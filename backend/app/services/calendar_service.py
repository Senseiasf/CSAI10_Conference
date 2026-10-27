import datetime
import uuid
import re
from typing import Dict, Any, Optional


class CalendarService:
    """
    Генератор iCalendar-файлов (.ics) по стандарту RFC 5545.
    Создает события для Google Calendar, Apple Calendar и Outlook:
    - Вылет туда (Flight Outbound)
    - Проживание в отеле (Check-in to Check-out)
    - Вылет обратно (Flight Inbound)
    """

    @staticmethod
    def _parse_date(date_str: Optional[str], default_date: datetime.date) -> datetime.date:
        if not date_str:
            return default_date
        
        # Try YYYY-MM-DD
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if m:
            try:
                return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        
        # Try DD.MM.YYYY
        m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", date_str)
        if m:
            try:
                return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                pass

        # Try Russian month names: "15 ноября", "10–17 ноября"
        months = {
            "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5,
            "июн": 6, "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12
        }
        for ru_m, m_num in months.items():
            if ru_m in date_str.lower():
                day_m = re.search(r"(\d{1,2})", date_str)
                if day_m:
                    day = int(day_m.group(1))
                    year = default_date.year if default_date.year >= 2026 else 2026
                    return datetime.date(year, m_num, min(day, 28))

        return default_date

    @classmethod
    def generate_ics(cls, cart: Dict[str, Any], booking_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Формирует содержимое .ics календаря на основе данных корзины/бронирования.
        """
        booking_id = (booking_info or {}).get("booking_id") or cart.get("booking_id") or f"OST-{uuid.uuid4().hex[:6].upper()}"
        destination = cart.get("destination", "Отпуск с Островком")
        hotel_name = cart.get("hotel", "Отель")
        nights = cart.get("nights", 7)
        flight_text = cart.get("flight", "Регулярный авиарейс")
        transfer_text = cart.get("transfer", "Индивидуальный трансфер")

        # Базовая дата
        today = datetime.date.today()
        base_start = datetime.date(today.year, 11, 15) if today.month > 10 else datetime.date(today.year, today.month + 1, 10)
        
        # Разбор дат заезда и выезда
        checkin_date = cls._parse_date(cart.get("checkin_date"), base_start)
        checkout_date = cls._parse_date(cart.get("checkout_date"), checkin_date + datetime.timedelta(days=nights))
        if checkout_date <= checkin_date:
            checkout_date = checkin_date + datetime.timedelta(days=max(1, nights))

        now_utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        
        # Времена перелета
        dep_time_str = "08:30"
        arr_time_str = "13:00"
        ret_dep_time_str = "15:45"
        ret_arr_time_str = "20:15"

        logistics = cart.get("logistics") or {}
        timeline = logistics.get("timeline") or {}
        if timeline.get("flight_departure"):
            dep_time_str = timeline.get("flight_departure")
        if timeline.get("flight_arrival"):
            arr_time_str = timeline.get("flight_arrival")

        dep_h, dep_m = [int(x) for x in dep_time_str.split(":")[:2]] if ":" in dep_time_str else (8, 30)
        arr_h, arr_m = [int(x) for x in arr_time_str.split(":")[:2]] if ":" in arr_time_str else (13, 0)
        ret_h, ret_m = [int(x) for x in ret_dep_time_str.split(":")[:2]] if ":" in ret_dep_time_str else (15, 45)
        ret_arr_h, ret_arr_m = [int(x) for x in ret_arr_time_str.split(":")[:2]] if ":" in ret_arr_time_str else (20, 15)

        dt_flight_out_start = datetime.datetime(checkin_date.year, checkin_date.month, checkin_date.day, dep_h, dep_m)
        dt_flight_out_end = datetime.datetime(checkin_date.year, checkin_date.month, checkin_date.day, arr_h, arr_m)
        if dt_flight_out_end <= dt_flight_out_start:
            dt_flight_out_end = dt_flight_out_start + datetime.timedelta(hours=4)

        dt_checkin = datetime.datetime(checkin_date.year, checkin_date.month, checkin_date.day, 15, 0)
        dt_checkout = datetime.datetime(checkout_date.year, checkout_date.month, checkout_date.day, 12, 0)

        dt_flight_ret_start = datetime.datetime(checkout_date.year, checkout_date.month, checkout_date.day, ret_h, ret_m)
        dt_flight_ret_end = datetime.datetime(checkout_date.year, checkout_date.month, checkout_date.day, ret_arr_h, ret_arr_m)
        if dt_flight_ret_end <= dt_flight_ret_start:
            dt_flight_ret_end = dt_flight_ret_start + datetime.timedelta(hours=4)

        def to_ics_dt(dt: datetime.datetime) -> str:
            return dt.strftime("%Y%m%dT%H%M%S")

        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Ostrovok AI Travel Agent//RU",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            f"X-WR-CALNAME:Тур: {destination} ({booking_id})",
            "X-WR-TIMEZONE:Europe/Moscow",
            
            # 1. Event: Вылет туда
            "BEGIN:VEVENT",
            f"UID:flight-out-{booking_id}@ostrovok.ai",
            f"DTSTAMP:{now_utc}",
            f"DTSTART:{to_ics_dt(dt_flight_out_start)}",
            f"DTEND:{to_ics_dt(dt_flight_out_end)}",
            f"SUMMARY:🛫 Вылет в {destination}: {flight_text}",
            f"DESCRIPTION:Перелет туда в {destination}. Ваучер заказа: {booking_id}. Трансфер: {transfer_text}.",
            f"LOCATION:{destination}",
            "STATUS:CONFIRMED",
            "END:VEVENT",

            # 2. Event: Проживание в отеле
            "BEGIN:VEVENT",
            f"UID:hotel-{booking_id}@ostrovok.ai",
            f"DTSTAMP:{now_utc}",
            f"DTSTART:{to_ics_dt(dt_checkin)}",
            f"DTEND:{to_ics_dt(dt_checkout)}",
            f"SUMMARY:🏨 Проживание: {hotel_name} ({destination})",
            f"DESCRIPTION:Заселение в отель {hotel_name} на {nights} ночей. Заказ: {booking_id}. Заселение с 15:00, выезд до 12:00.",
            f"LOCATION:{hotel_name}, {destination}",
            "STATUS:CONFIRMED",
            "END:VEVENT",

            # 3. Event: Вылет обратно
            "BEGIN:VEVENT",
            f"UID:flight-ret-{booking_id}@ostrovok.ai",
            f"DTSTAMP:{now_utc}",
            f"DTSTART:{to_ics_dt(dt_flight_ret_start)}",
            f"DTEND:{to_ics_dt(dt_flight_ret_end)}",
            f"SUMMARY:🛬 Обратный перелет: {destination} ➔ Домой",
            f"DESCRIPTION:Обратный рейс из {destination}. Рекомендуется прибыть в аэропорт за 2-3 часа до вылета.",
            f"LOCATION:{destination}",
            "STATUS:CONFIRMED",
            "END:VEVENT",

            "END:VCALENDAR"
        ]

        return "\r\n".join(ics_lines) + "\r\n"
