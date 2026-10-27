"""Calendar updates independent of databases and language model providers."""
import re
from datetime import date, timedelta


RU_MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12
}


def update_trip_dates(message: str, profile: dict, cart: dict | None = None) -> str | None:
    """Update supported dates; return a clarification instead of guessing a shift."""
    text = message.lower()
    shift = re.search(
        r"(?:на\s+)?(\d+)\s*(?:дня|дней|день)\s*(позже|раньше)", text
    ) or re.search(r"(позже|раньше)\s+на\s+(\d+)\s*(?:дня|дней|день)", text)
    if shift:
        values = shift.groups()
        amount = int(next(v for v in values if v.isdigit()))
        direction = next(v for v in values if not v.isdigit())
        if not profile.get("checkin_date"):
            if cart and cart.get("checkin_date"):
                profile["checkin_date"] = cart["checkin_date"]
                profile["checkout_date"] = cart.get("checkout_date")
            elif cart and (cart.get("destination") or cart.get("hotel") or cart.get("dates")):
                base_dt = date.today() + timedelta(days=14)
                nights_cnt = profile.get("nights", cart.get("nights", 7))
                profile["checkin_date"] = base_dt.isoformat()
                profile["checkout_date"] = (base_dt + timedelta(days=nights_cnt)).isoformat()
                profile["dates"] = f"{base_dt:%d.%m.%Y} — {base_dt + timedelta(days=nights_cnt):%d.%m.%Y} ({nights_cnt} ночей)"
        if not profile.get("checkin_date"):
            return "С какой даты сдвигаем поездку? Укажите дату заезда в формате ДД.ММ.ГГГГ."
        start = date.fromisoformat(profile["checkin_date"])
        try:
            start += timedelta(days=amount * (-1 if direction == "раньше" else 1))
        except OverflowError:
            return "Слишком большой сдвиг. Укажите новую дату заезда в формате ДД.ММ.ГГГГ."
    else:
        iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        dotted = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text)
        ru_range = re.search(
            r"(?:с\s+)?(\d{1,2})\s*(?:-|по|до)\s*(\d{1,2})\s*(янв\w*|фев\w*|мар\w*|апр\w*|ма[йея]\w*|июн\w*|июл\w*|авг\w*|сен\w*|окт\w*|ноя\w*|дек\w*)(?:\s*(\d{4}))?",
            text
        )
        ru_single = re.search(
            r"(?:заезд\s+|с\s+|на\s+)?(\d{1,2})\s*(янв\w*|фев\w*|мар\w*|апр\w*|ма[йея]\w*|июн\w*|июл\w*|авг\w*|сен\w*|окт\w*|ноя\w*|дек\w*)(?:\s*(\d{4}))?",
            text
        ) if not ru_range else None
        dotted_short = re.search(r"\b(\d{1,2})\.(\d{1,2})\b", text) if (not dotted and not iso and not ru_range and not ru_single) else None

        if iso or dotted:
            parts = [int(x) for x in (iso or dotted).groups()]
            try:
                start = date(*parts) if iso else date(parts[2], parts[1], parts[0])
            except ValueError:
                return "Такой даты нет в календаре. Укажите дату заезда в формате ДД.ММ.ГГГГ."
        elif ru_range:
            d1, d2, m_str, y_str = ru_range.groups()
            m_num = next((v for k, v in RU_MONTHS.items() if m_str.startswith(k)), None)
            y_num = int(y_str) if y_str else date.today().year
            try:
                start = date(y_num, m_num, int(d1))
                end_day = int(d2)
                if end_day > int(d1):
                    profile["nights"] = end_day - int(d1)
            except ValueError:
                return "Такой даты нет в календаре. Укажите дату заезда в формате ДД.ММ.ГГГГ."
        elif ru_single:
            d1, m_str, y_str = ru_single.groups()
            m_num = next((v for k, v in RU_MONTHS.items() if m_str.startswith(k)), None)
            y_num = int(y_str) if y_str else date.today().year
            try:
                start = date(y_num, m_num, int(d1))
                if y_str is None and start < date.today():
                    start = date(y_num + 1, m_num, int(d1))
            except ValueError:
                return "Такой даты нет в календаре. Укажите дату заезда в формате ДД.ММ.ГГГГ."
        elif dotted_short:
            d1, m1 = [int(x) for x in dotted_short.groups()]
            y_num = date.today().year
            try:
                start = date(y_num, m1, d1)
                if start < date.today():
                    start = date(y_num + 1, m1, d1)
            except ValueError:
                return "Такой даты нет в календаре. Укажите дату заезда в формате ДД.ММ.ГГГГ."
        else:
            if re.search(r"\b(?:январ|феврал|март|апрел|ма[йея]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*", text):
                return "На какую дату планируем заезд? Укажите её в формате ДД.ММ.ГГГГ."
            is_flight_time = any(w in text for w in ("вылет", "самолет", "самолёт", "рейс", "утр", "вечер", "в 6", "в 7", "время", "попозже", "пораньше")) and not any(w in text for w in ("дату", "даты", "заезд", "выезд", "дня позже", "дней позже", "дня раньше", "дней раньше", "сдвинь поездку", "сдвинь даты"))
            if not is_flight_time and any(w in text for w in ("позже", "раньше", "сдвинь", "другие даты")):
                return "Укажите новую дату заезда (ДД.ММ.ГГГГ) или на сколько дней сдвинуть поездку."
            start = date.fromisoformat(profile["checkin_date"]) if profile.get("checkin_date") else None

    # A relative date shift must not change the length of the stay.
    if not shift:
        duration = re.search(r"(?:на\s+)?(\d+)\s*(?:ноч\w*|дн\w*|сут\w*)", text)
        nights = int(duration.group(1)) if duration else None
        if nights is not None and not 1 <= nights <= 60:
            return "Укажите длительность поездки от 1 до 60 ночей."
        if nights is None:
            for phrases, value in (
                (("две недели", "2 недели"), 14),
                (("три недели", "3 недели"), 21),
                (("выходн", "уикенд", "weekend", "пару дней"), 2),
                (("на неделю", "неделю", "неделя"), 7),
                (("на месяц",), 30),
                (("на день", "на сутки"), 1),
            ):
                if any(phrase in text for phrase in phrases):
                    nights = value
                    break
        if nights is not None:
            profile["nights"] = nights

    nights = profile.get("nights", 7)
    if start:
        try:
            end = start + timedelta(days=nights)
        except OverflowError:
            return "Дата выезда выходит за пределы календаря. Укажите другую дату заезда."
        profile.update(checkin_date=start.isoformat(), checkout_date=end.isoformat())
        profile["dates"] = f"{start:%d.%m.%Y} — {end:%d.%m.%Y} ({nights} ночей)"
    else:
        profile["dates"] = f"{nights} ночей"
    return None
