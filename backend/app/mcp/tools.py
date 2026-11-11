import json
import logging
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import select

from app.database import async_session_maker
from app.models import DestinationModel, HotelModel, RiskAlertModel, UserSessionModel
from app.services.graph_repository import graph_repository

logger = logging.getLogger("mcp_tools")

# Координаты курортов для реального метео-сервиса Open-Meteo
DESTINATION_COORDINATES = {
    "sochi": {"lat": 43.59, "lon": 39.73, "name": "Сочи"},
    "uae": {"lat": 25.20, "lon": 55.27, "name": "Дубай"},
    "turkey": {"lat": 36.89, "lon": 30.71, "name": "Анталья"},
    "thailand": {"lat": 7.89, "lon": 98.39, "name": "Пхукет"},
    "maldives": {"lat": 4.18, "lon": 73.51, "name": "Мале"},
    "egypt": {"lat": 27.92, "lon": 34.33, "name": "Шарм-эль-Шейх"},
    "spb": {"lat": 59.93, "lon": 30.31, "name": "Санкт-Петербург"},
    "altai": {"lat": 51.83, "lon": 85.78, "name": "Манжерок / Горно-Алтайск"},
    "georgia": {"lat": 41.72, "lon": 44.78, "name": "Тбилиси"}
}

# Расшифровка WMO weather codes на русский язык
WMO_CODES = {
    0: "☀️ Ясно",
    1: "🌤️ Преимущественно ясно",
    2: "⛅ Переменная облачность",
    3: "☁️ Пасмурно",
    45: "🌫️ Туман",
    48: "🌫️ Изморозь",
    51: "🌦️ Небольшой дождь",
    53: "🌧️ Умеренный дождь",
    55: "🌧️ Сильный дождь",
    61: "🌧️ Дождь",
    63: "🌧️ Затяжной дождь",
    65: "🌧️ Ливень",
    71: "🌨️ Небольшой снегопад",
    73: "🌨️ Снегопад",
    75: "❄️ Сильный снегопад",
    80: "🌦️ Кратковременный ливень",
    81: "🌧️ Ливневый дождь",
    82: "⛈️ Шквалистый ливень",
    95: "⛈️ Гроза",
    96: "⛈️ Гроза с градом",
    99: "⛈️ Сильная гроза с градом"
}

# Кэш данных направлений для быстрых расчетов
_DESTINATIONS_DATA_CACHE = None

def _get_destinations_json():
    global _DESTINATIONS_DATA_CACHE
    if _DESTINATIONS_DATA_CACHE is None:
        p = Path(__file__).resolve().parent.parent / "data" / "destinations.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _DESTINATIONS_DATA_CACHE = json.load(f)
        else:
            _DESTINATIONS_DATA_CACHE = {"destinations": []}
    return _DESTINATIONS_DATA_CACHE


class MCPTools:
    """
    Реестр инструментов агента, совместимых со спецификацией Model Context Protocol (MCP)
    и Function Calling OpenAI.
    """

    @staticmethod
    async def get_live_weather(destination_id: str) -> Dict[str, Any]:
        """
        MCP Tool: Запрашивает реальный прогноз погоды на ближайшие 7 дней (неделю)
        через открытый глобальный метео-сервис Open-Meteo.
        """
        coords = DESTINATION_COORDINATES.get(destination_id)
        if not coords:
            return {"available": False, "summary": "Координаты региона не найдены"}

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={coords['lat']}&longitude={coords['lon']}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
                f"&timezone=auto"
            )
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    daily = data.get("daily", {})
                    t_mins = daily.get("temperature_2m_min", [])[:7]
                    t_maxs = daily.get("temperature_2m_max", [])[:7]
                    precs = daily.get("precipitation_probability_max", [])[:7]
                    w_codes = daily.get("weather_code", [])[:7]

                    if t_mins and t_maxs:
                        today_code = w_codes[0] if w_codes else 0
                        today_cond = WMO_CODES.get(today_code, "⛅ Переменная облачность")
                        min_w = round(min(t_mins))
                        max_w = round(max(t_maxs))
                        max_prec = max(precs) if precs else 0
                        today_max = round(t_maxs[0])
                        today_min = round(t_mins[0])

                        prec_str = f", осадки до {max_prec}%" if max_prec > 20 else ", без существенных осадков"
                        summary = f"Сегодня: {today_cond}, +{today_min}..+{today_max}°C. На 7 дней: от +{min_w}°C до +{max_w}°C{prec_str}."
                        return {
                            "available": True,
                            "city": coords["name"],
                            "today": {"condition": today_cond, "temp_min": today_min, "temp_max": today_max},
                            "week_range": {"min": min_w, "max": max_w, "max_precipitation_pct": max_prec},
                            "summary": summary,
                            "source": "Open-Meteo Live API"
                        }
        except Exception as e:
            logger.warning(f"Live weather API error for {destination_id}: {e}")

        return {
            "available": False,
            "summary": "Прогноз формируется на основе сезонных климатических норм.",
            "source": "PostgreSQL Climate Cache"
        }

    @staticmethod
    async def get_destination_risks(destination_id: str, month: Optional[str] = None) -> Dict[str, Any]:
        """
        MCP Tool: Запрашивает актуальные риски, сезонные опасности (наводнения, штормы, медузы, жара)
        и геополитическую обстановку по направлению.
        """
        logger.info(f"[MCP] Calling get_destination_risks for {destination_id}, month={month}")
        async with async_session_maker() as session:
            stmt = select(RiskAlertModel).where(RiskAlertModel.destination_id == destination_id)
            res = await session.execute(stmt)
            alerts = res.scalars().all()

            dest_res = await session.execute(select(DestinationModel).where(DestinationModel.id == destination_id))
            dest = dest_res.scalar_one_or_none()

            dest_name = dest.name if dest else destination_id

        alerts_data = []
        has_high_severity = False

        for a in alerts:
            alerts_data.append({
                "type": a.risk_type,
                "severity": a.severity,
                "title": a.title,
                "description": a.description
            })
            if a.severity == "high":
                has_high_severity = True

        recommendations = [
            "Рекомендуем оформить комплексную страховку с опцией защиты от задержки рейсов и задержки багажа.",
            "Все трансферы работают с flight tracking: при задержке рейса водитель ожидает бесплатно до 90 минут.",
            "Вещи первой необходимости, зарядки и сменную одежду держите в ручной клади на случай задержки выдачи багажа."
        ]
        if destination_id == "thailand":
            recommendations.append("Ни в коем случае не заходите в воду при установленных красных флагах (опасные отбойные течения).")
            recommendations.append("На Пхукете закладывайте 80 минут на прохождение границы и получение багажа.")
            recommendations.append("При стыковке через ОАЭ используйте сквозной билет для автоматической перегрузки багажа.")
        elif destination_id == "uae":
            recommendations.append("В светлое время суток избегайте открытого солнца, пейте больше электролитов.")
            recommendations.append("В терминалах Дубая (DXB) закладывайте не менее 75-80 минут на границу и выдачу багажа.")
        elif destination_id == "sochi":
            recommendations.append("В межсезонье в горах проверяйте статус канатных дорог на официальном сайте курорта.")
            recommendations.append("При штормовых задержках прилёта в Сочи трансфер переносится без доплат.")
        elif destination_id == "turkey":
            recommendations.append("При стыковках через аэропорты Стамбула (IST/SAW) выбирайте интервал пересадки не менее 2 часов.")
            recommendations.append("При поездках по Стамбулу закладывайте дополнительно 40-50 минут на дорожные пробки.")
        elif destination_id == "maldives":
            recommendations.append("Гидросамолеты летают строго при дневном свете (до 16:30); при вечернем прилёте требуется транзитная ночь в Мале.")
            recommendations.append("На локальных островах алкоголь строго запрещён законами Мальдивской Республики.")
            recommendations.append("При снорклинге на внешних рифах атоллов обязательно используйте спасательный жилет из-за сильных океанических течений.")
        elif destination_id == "egypt":
            recommendations.append("Купание на коралловых рифах разрешено только в защитной обуви (опасность морских ежей и рыб-камней).")
            recommendations.append("В Шарм-эль-Шейхе действует бесплатный штамп Sinai Only на 14 дней.")
        elif destination_id == "spb":
            recommendations.append("В период навигации с 01:10 до 05:00 разводятся мосты через Неву; проезд на Васильевский остров на авто заблокирован.")
            recommendations.append("В Санкт-Петербурге ветреная влажная погода с Финского залива; берите непродуваемую куртку.")
        elif destination_id == "altai":
            recommendations.append("В мае-июне активны таёжные клещи: оформите противоклещевую страховку и используйте репелленты.")
            recommendations.append("В горных ущельях сотовая связь может отсутствовать: скачайте офлайн-карты заранее.")
        elif destination_id == "georgia":
            recommendations.append("Перед поездкой по Военно-Грузинской дороге проверяйте статус Крестового перевала и лавинную сводку.")
            recommendations.append("Для поездок в Гудаури и Казбеги в зимне-весенний период используйте только полноприводные автомобили 4x4 с цепями противоскольжения.")
            recommendations.append("Карты РФ (включая «Мир») в Грузии не работают: имейте запас наличных (USD/EUR/GEL) или зарубежную карту.")
            recommendations.append("Оформите полис туристического медицинского страхования с покрытием от 30 000 GEL (обязательное требование законодательства 2026 г.).")

        return {
            "destination_id": destination_id,
            "destination_name": dest_name,
            "month_context": month or "текущий сезон",
            "overall_threat_level": "HIGH" if has_high_severity else "MEDIUM",
            "active_alerts": alerts_data,
            "safety_recommendations": recommendations,
            "source": "Ostrovok Safety & Force-Majeure MCP Provider"
        }

    @staticmethod
    async def compare_hotels(
        destination_id: str,
        pet_friendly: Optional[bool] = None,
        family_friendly: Optional[bool] = None,
        max_price_per_night: Optional[int] = None,
        exclude_hotel_ids: Optional[List[str]] = None,
        preferred_tags: Optional[List[str]] = None,
        preferred_hotel_id: Optional[str] = None,
        limit: Optional[int] = 3,
        private_room_required: bool = False,
        city: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        MCP Tool: Находит и сравнивает 2-3 отеля по ключевым параметрам:
        цена, условия отмены, питание, удаленность от моря, pet-friendly, теги интересов,
        исключая ранее отклоненные отели (Point 2 Re-planning).
        """
        logger.info(f"[MCP] Calling compare_hotels for {destination_id}, city={city}, exclude={exclude_hotel_ids}, tags={preferred_tags}")
        async with async_session_maker() as session:
            stmt = select(HotelModel).where(HotelModel.destination_id == destination_id)
            if city:
                stmt = stmt.where(HotelModel.city.ilike(f"%{city}%"))
            if pet_friendly:
                stmt = stmt.where(HotelModel.pet_friendly == True)
            if family_friendly:
                stmt = stmt.where(HotelModel.family_friendly == True)
            if max_price_per_night is not None:
                stmt = stmt.where(HotelModel.price_per_night_rub <= max_price_per_night)
            if exclude_hotel_ids:
                stmt = stmt.where(~HotelModel.id.in_(exclude_hotel_ids))

            stmt = stmt.order_by(HotelModel.price_per_night_rub.asc())
            res = await session.execute(stmt)
            hotels = list(res.scalars().all())

        # Возрастное ограничение обязательно, даже при противоречивых данных отеля.
        if family_friendly:
            hotels = [h for h in hotels if not any(
                "adults_only" in str(feature).lower() for feature in (h.features or [])
            )]

        if private_room_required:
            hotels = [h for h in hotels if not (
                any(tag in str(feature).lower()
                    for feature in (h.features or [])
                    for tag in ("hostel_dorm", "shared_bathroom", "shared_room"))
                or any(word in f"{h.name} {h.category}".lower()
                       for word in ("хостел", "hostel", "dorm"))
            )]

        if not hotels:
            return {
                "compared_hotels": [],
                "tradeoff_analysis": [],
                "recommended_choice": None,
                "status": "no_matches",
            }

        # Ранжирование по интересам и тегам
        def hotel_tag_score(h: HotelModel) -> int:
            score = 0
            feats = [str(f).lower() for f in (h.features or [])]
            cat = h.category.lower()
            tags = preferred_tags or []

            # Если путешествуют с детьми: жестко отсекаем Adults Only!
            if family_friendly is True and (not h.family_friendly or any("adults_only" in f for f in feats)):
                return -9999

            # Если отель экстремально дорогой (люкс > 80k), но пользователь НЕ просил luxury:
            if h.price_per_night_rub > 80000 and not any(t in ("luxury", "extreme_luxury") for t in tags):
                score -= 10

            # Если отель 1★ хостел / дорм, но пользователь НЕ просил budget_friendly:
            if any(k in feats for k in ("hostel_dorm", "shared_bathroom")) and not any(t in ("budget_friendly", "hostel_dorm") for t in tags):
                score -= 10

            for tag in tags:
                t = tag.lower()
                if any(t in f for f in feats) or t in cat:
                    score += 3
                if t == "budget_friendly":
                    if h.price_per_night_rub <= 6000:
                        score += 10
                    elif h.price_per_night_rub <= 12000:
                        score += 6
                    elif h.price_per_night_rub > 25000:
                        score -= 8
                if t == "luxury" and h.price_per_night_rub >= 30000:
                    score += 4
                if t == "extreme_luxury" and h.price_per_night_rub >= 100000:
                    score += 6
                if t == "beach" and h.distance_to_sea_meters <= 100:
                    score += 2
                if t == "family_kids" and h.family_friendly:
                    score += 3
            return score

        is_any_lux = any(t in ("luxury", "extreme_luxury", "luxury_extreme") for t in (preferred_tags or []))
        if is_any_lux:
            hotels.sort(key=lambda h: (-hotel_tag_score(h), -h.price_per_night_rub, h.id))
        else:
            hotels.sort(key=lambda h: (-hotel_tag_score(h), h.price_per_night_rub, h.id))

        if preferred_hotel_id:
            hotels.sort(key=lambda h: 0 if h.id == preferred_hotel_id else 1)

        selected_hotels = hotels if limit is None else hotels[:limit]

        comparison_table = []
        for h in selected_hotels:
            hotel_dict = {
                "id": h.id,
                "name": h.name,
                "city": getattr(h, "city", ""),
                "stars": f"{h.stars}★",
                "price_per_night_rub": h.price_per_night_rub,
                "price_per_night_formatted": f"{h.price_per_night_rub:,} ₽",
                "pet_surcharge_per_night_rub": getattr(h, "pet_surcharge_per_night_rub", 0) or 0,
                "meals": h.meals,
                "cancellation": h.cancellation_policy,
                "pet_friendly": "✅ Да" if h.pet_friendly else "❌ Нет",
                "distance_to_sea": f"{h.distance_to_sea_meters} м" if h.distance_to_sea_meters > 0 else "Пляж / Первая линия",
                "features": [f for f in (h.features or []) if isinstance(f, str)]
            }
            # Extract metadata (coordinates, rating, reviews, website) if stored in features
            for f in (h.features or []):
                if isinstance(f, dict):
                    if "coordinates" in f:
                        hotel_dict["coordinates"] = f["coordinates"]
                    if "rating_score" in f:
                        hotel_dict["rating_score"] = f["rating_score"]
                    if "reviews_count" in f:
                        hotel_dict["reviews_count"] = f["reviews_count"]
                    if "website_url" in f:
                        hotel_dict["website_url"] = f["website_url"]
                    if "real_description" in f:
                        hotel_dict["real_description"] = f["real_description"]

            comparison_table.append(hotel_dict)

        tradeoffs = []
        if len(selected_hotels) >= 2:
            h1 = selected_hotels[0]
            h2 = selected_hotels[1]
            price_diff = abs(h2.price_per_night_rub - h1.price_per_night_rub)
            tradeoffs.append(
                f"• Вариант «{h1.name}» доступнее на {price_diff:,} ₽/ночь ({h1.price_per_night_rub:,} ₽ vs {h2.price_per_night_rub:,} ₽), отмена: «{h1.cancellation_policy}»."
            )
            tradeoffs.append(
                f"• Вариант «{h2.name}» предлагает формат «{h2.category}» ({h2.meals})."
            )

        return {
            "destination_id": destination_id,
            "compared_hotels": comparison_table,
            "tradeoff_analysis": tradeoffs,
            "recommended_choice": comparison_table[0]
        }

    @staticmethod
    async def check_logistics_compatibility(
        airport_code: str,
        hotel_id: str,
        origin_airport_code: Optional[str] = "SVO",
        prefer_direct_flight: Optional[bool] = None,
        without_transfer: Optional[bool] = False,
        flight_time_pref: Optional[str] = None,
        travelers_count: Optional[int] = None,
        strict_graph_validation: bool = True,
        return_airport_code: str | None = None,
        transfer_arrival_time: str | None = None,
    ) -> Dict[str, Any]:
        """
        MCP Tool: Запрашивает графовую базу Neo4j для проверки сквозной цепочки:
        (Аэропорт вылета) -[:FLIGHT_TO]-> (Аэропорт прибытия) -[:TRANSFER_TO]-> (Отель)
        """
        logger.info(f"[MCP] Calling check_logistics_compatibility: {origin_airport_code} -> {airport_code} -> {hotel_id}, direct={prefer_direct_flight}, no_transfer={without_transfer}, time_pref={flight_time_pref}")
        if transfer_arrival_time is not None:
            from app.services.graph_service import GraphService
            return await GraphService.validate_transfer(
                airport_code, hotel_id, transfer_arrival_time, max(1, travelers_count or 1)
            )
        return await graph_repository.check_compatibility(
            airport_code,
            hotel_id,
            origin_airport_code=origin_airport_code,
            prefer_direct_flight=prefer_direct_flight,
            without_transfer=without_transfer,
            flight_time_pref=flight_time_pref,
            travelers_count=travelers_count,
            strict_graph_validation=strict_graph_validation,
            return_airport_code=return_airport_code,
        )

    @staticmethod
    def calculate_budget_breakdown(
        destination_id: str,
        flight_price_roundtrip: int,
        hotel_price_per_night: int,
        transfer_price: int,
        meals_type: str = "Завтрак включен",
        nights: int = 7,
        pet_surcharge_per_night: int = 0,
        rooms_count: int = 1,
        travelers_count: int = 1
    ) -> Dict[str, Any]:
        """
        MCP Tool / Calculator: Рассчитывает детальную структуру расходов:
        1. Обязательные платежи (сразу под ключ): перелет туда-обратно + проживание (с учетом количества номеров) + трансфер + доплата за питомца.
        2. Траты не сразу (карманные расходы на месте): питание (с учетом типа питания в отеле) + развлечения.
        3. Итоговый ориентир общего бюджета поездки.
        """
        data = _get_destinations_json()
        dest_meta = next((d for d in data.get("destinations", []) if d["id"] == destination_id), {})
        onsite_meta = dest_meta.get("onsite_expenses_rub", {})

        food_daily = onsite_meta.get("food_per_day", {"min": 2000, "max": 4500})
        ent_tour = onsite_meta.get("entertainment_7d", {"min": 10000, "max": 25000})
        actual_travelers = max(1, travelers_count or 1)

        # Корректировка расходов на еду в зависимости от включенного питания
        is_all_inclusive = "all inclusive" in meals_type.lower() or "все включено" in meals_type.lower()
        is_half_board = "полупансион" in meals_type.lower() or "ужин" in meals_type.lower() or "half board" in meals_type.lower()

        if is_all_inclusive:
            food_min = int(600 * nights * actual_travelers)
            food_max = int(1500 * nights * actual_travelers)
            food_note = f"All-Inclusive в отеле: основное питание включено, заложены перекусы и напитки ({food_min:,}–{food_max:,} ₽ за {nights} н., {actual_travelers} путешеств.)"
        elif is_half_board:
            food_min = int(food_daily["min"] * 0.5 * nights * actual_travelers)
            food_max = int(food_daily["max"] * 0.5 * nights * actual_travelers)
            food_note = f"Полупансион (завтрак + ужин): заложены обеды в кафе и ресторанах ({food_min:,}–{food_max:,} ₽ за {nights} н., {actual_travelers} путешеств.)"
        else:
            food_min = int(food_daily["min"] * nights * actual_travelers)
            food_max = int(food_daily["max"] * nights * actual_travelers)
            food_note = f"Завтрак включен: заложены обеды и ужины в кафе и ресторанах ({food_min:,}–{food_max:,} ₽ за {nights} н., {actual_travelers} путешеств.)"

        ent_min = max(2000, int(ent_tour.get("min", 10000) / 7.0 * nights * actual_travelers))
        ent_max = max(5000, int(ent_tour.get("max", 25000) / 7.0 * nights * actual_travelers))

        actual_rooms = max(1, rooms_count or 1)
        hotel_total = hotel_price_per_night * nights * actual_rooms
        pet_total = (pet_surcharge_per_night or 0) * nights
        flight_total = flight_price_roundtrip * actual_travelers
        mandatory_total = flight_total + hotel_total + transfer_price + pet_total

        onsite_min = food_min + ent_min
        onsite_max = food_max + ent_max

        total_min = mandatory_total + onsite_min
        total_max = mandatory_total + onsite_max

        return {
            "nights": nights,
            "mandatory": {
                "flight_roundtrip": flight_total,
                "travelers_count": actual_travelers,
                "rooms_count": actual_rooms,
                "hotel_total": hotel_total,
                "hotel_per_night": hotel_price_per_night,
                "transfer": transfer_price,
                "pet_surcharge_per_night": pet_surcharge_per_night or 0,
                "pet_total": pet_total,
                "total_mandatory_rub": mandatory_total
            },
            "onsite_flexible": {
                "travelers_count": actual_travelers,
                "food_range_rub": {"min": food_min, "max": food_max},
                "food_note": food_note,
                "entertainment_range_rub": {"min": ent_min, "max": ent_max},
                "total_onsite_range_rub": {"min": onsite_min, "max": onsite_max},
                "description": onsite_meta.get("description", "Экскурсии, активности, местный транспорт, сувениры.")
            },
            "total_trip_budget_range_rub": {
                "min": total_min,
                "max": total_max
            }
        }

    @staticmethod
    async def checkout_cart(user_id: str, payment_method: str = "Банковская карта онлайн") -> Dict[str, Any]:
        """Return the explicit provider availability status for checkout."""
        from app.services.booking_stub import booking_stub
        async with async_session_maker() as session:
            current = await session.get(UserSessionModel, user_id)
            if current is None:
                return {"status": "error", "message": "Сессия не найдена"}
            return booking_stub("book", current.current_cart or {})
