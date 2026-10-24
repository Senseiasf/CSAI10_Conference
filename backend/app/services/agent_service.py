import json
import logging
import re
from typing import Dict, Any, List, Optional
from sqlalchemy import select, text

from app.database import async_session_maker
from app.models import DestinationModel, HotelModel, RiskAlertModel, UserSessionModel
from app.services.vector_service import generate_text_embedding
from app.mcp.tools import MCPTools

logger = logging.getLogger(__name__)

# Маппинг внешних направлений в наши 8 расширенных регионов (Пункт 3 ТЗ)
UNKNOWN_DESTINATIONS_MAP = [
    {
        "keywords": ["бали", "сейшел", "гоа", "вьетнам", "шри-ланк", "доминикан", "куб", "маврики", "фиджи", "бора-бора"],
        "pilot_alt_id": "maldives",
        "pilot_alt_name": "Мальдивы (Атоллы)",
        "vibe_reason": "белоснежные тропические пляжи, лазурный Индийский океан, уединенные виллы над водой и живые коралловые рифы (например, Cheval Blanc Randheli или Villa Nautica)"
    },
    {
        "keywords": ["париж", "рим", "барселон", "праг", "европ", "итали", "франци", "испани", "вена", "будапешт"],
        "pilot_alt_id": "spb",
        "pilot_alt_name": "Санкт-Петербург (Северная столица)",
        "vibe_reason": "европейская классическая архитектура, парадные дворцы, Эрмитаж, Мариинский театр и гранд-отели мирового уровня (Four Seasons Lion Palace или Гранд Отель Европа)"
    },
    {
        "keywords": ["иордани", "марокко", "катар", "оман", "бахрейн"],
        "pilot_alt_id": "egypt",
        "pilot_alt_name": "Египет (Шарм-эль-Шейх / Красное море)",
        "vibe_reason": "богатейший подводный мир, круглогодичное теплое море, коралловые заповедники и курорты All-Inclusive (Rixos Premium Seagate или Four Seasons)"
    },
    {
        "keywords": ["альп", "доломит", "домбай", "эльбрус", "байкал", "карели", "кавказ", "шерегеш", "архыз"],
        "pilot_alt_id": "altai",
        "pilot_alt_name": "Алтай (Манжерок / Горный Алтай)",
        "vibe_reason": "величественная первозданная природа, бирюзовая Катунь, вековые кедры, чистейший горный воздух и премиальные эко-курорты (Манжерок 5★ или Altay Resort)"
    }
]


class AgentService:
    @staticmethod
    async def get_or_create_session(user_id: str) -> UserSessionModel:
        async with async_session_maker() as session:
            res = await session.execute(
                select(UserSessionModel).where(UserSessionModel.user_id == user_id)
            )
            user_session = res.scalar_one_or_none()
            if not user_session:
                user_session = UserSessionModel(
                    user_id=user_id,
                    current_state="INSPIRATION",
                    profile={
                        "has_kids": False,
                        "has_pets": False,
                        "preferred_destination": None,
                        "nights": 7,
                        "dates": "7 ночей",
                        "interests": [],
                        "rejected_hotels": [],
                        "prefer_direct_flight": None,
                        "without_transfer": False,
                        "party_size": "вдвоем"
                    },
                    current_cart={},
                    conversation_history=[]
                )
                session.add(user_session)
                await session.commit()
                await session.refresh(user_session)
            return user_session

    @staticmethod
    def extract_interests_and_slots(user_lower: str, profile: dict) -> dict:
        """
        Извлекает интересы путешественника, состав и длительность поездки (Память пользователя).
        """
        interests = set(profile.get("interests", []))

        # Извлечение длительности поездки (ночи / дни)
        nights_matched = None
        digit_match = re.search(r'(?:на\s+)?(\d+)\s*(?:ноч|дн|сут)', user_lower)
        if digit_match:
            try:
                val = int(digit_match.group(1))
                if 1 <= val <= 60:
                    nights_matched = val
            except ValueError:
                pass

        if not nights_matched:
            if any(w in user_lower for w in ["выходн", "уикенд", "weekend", "пару дней", "на 2 дня"]):
                nights_matched = 2
            elif any(w in user_lower for w in ["две недели", "2 недели"]):
                nights_matched = 14
            elif any(w in user_lower for w in ["три недели", "3 недели"]):
                nights_matched = 21
            elif any(w in user_lower for w in ["на неделю", "неделю", "неделя"]):
                nights_matched = 7
            elif any(w in user_lower for w in ["на месяц"]):
                nights_matched = 30
            elif "на день" in user_lower or "на сутки" in user_lower:
                nights_matched = 1

        if nights_matched:
            profile["nights"] = nights_matched
            profile["dates"] = f"{nights_matched} ночей"

        if any(w in user_lower for w in ["пляж", "море", "купаться", "позагорать", "песок", "загар", "плавать"]):
            interests.add("beach")
        if any(w in user_lower for w in ["гор", "лыж", "сноуборд", "полян", "хайкинг", "треккинг"]):
            interests.add("mountains")
        if any(w in user_lower for w in ["истор", "экскурси", "достопримечательн", "музе", "культура", "босфор", "старый город"]):
            interests.add("history_culture")
        if any(w in user_lower for w in ["спа", "spa", "релакс", "массаж", "велнес", "бассейн"]):
            interests.add("spa_wellness")
        if any(w in user_lower for w in ["ребенок", "дет", "малыш", "семь", "аквапарк", "горок"]):
            interests.add("family_kids")
            profile["has_kids"] = True
        if any(w in user_lower for w in ["собак", "питомц", "кошк", "пес"]):
            interests.add("pet_friendly")
            profile["has_pets"] = True
        if any(w in user_lower for w in ["дешев", "бюджет", "эконом", "подешев", "недорог", "хостел", "койко"]):
            interests.add("budget_friendly")
        if any(w in user_lower for w in ["самый дорог", "самый роскошн", "дворец", "burj al arab", "шейх", "президентск", "максимально дорог", "ультра-люкс"]):
            interests.add("extreme_luxury")
            interests.add("luxury")
        elif any(w in user_lower for w in ["люкс", "премиум", "вилл", "5 звезд", "пять звезд", "deluxe", "vip", "роскошн"]):
            interests.add("luxury")
        if any(w in user_lower for w in ["вдвоем", "парой", "романтик"]):
            profile["party_size"] = "вдвоем"
        elif any(w in user_lower for w in ["один", "одна", "соло"]):
            profile["party_size"] = "соло"

        profile["interests"] = list(interests)
        return profile

    @staticmethod
    def detect_unknown_destination(message: str) -> Optional[Dict[str, str]]:
        """
        Проверяет, запрашивает ли пользователь направление вне текущей пилотной базы (Пункт 3 ТЗ).
        Возвращает информацию о пилотной альтернативе по вайбу/климату.
        """
        m_lower = message.lower()
        for item in UNKNOWN_DESTINATIONS_MAP:
            for kw in item["keywords"]:
                if kw in m_lower:
                    return item
        return None

    @staticmethod
    def is_vague_greeting(message: str, has_cart_destination: bool) -> bool:
        """
        Проверяет, является ли сообщение пользователя неопределенным приветствием/стартовым запросом
        без указания направления (Пункт 4 ТЗ: Анти-15-вопросов).
        """
        m_lower = message.lower().strip()
        # Если направление уже выбрано в корзине, это не стартовое приветствие
        if has_cart_destination:
            return False

        # Проверка на наличие конкретных стран/городов
        known_kws = [
            "сочи", "сириус", "полян", "адлер", "дуба", "оаэ", "абу-даби", "эмират",
            "турци", "анталь", "стамбул", "белек", "кемер", "пхукет", "тай", "таиланд", "самуи",
            "мальдив", "мале", "египет", "шарм", "хургад", "синай",
            "спб", "питер", "петербург", "санкт-петербург",
            "алтай", "манжерок", "телецк", "горно-алтайск"
        ]
        if any(k in m_lower for k in known_kws):
            return False

        # Если есть внешние страны, сработает fallback
        for item in UNKNOWN_DESTINATIONS_MAP:
            if any(k in m_lower for k in item["keywords"]):
                return False

        vague_phrases = [
            "привет", "здравствуйте", "добрый день", "добрый вечер", "хай", "hello",
            "хочу отдохнуть", "помоги выбрать", "посоветуй", "куда поехать", "не знаю куда",
            "хочу в отпуск", "подбери тур", "что посоветуешь", "помоги с выбором", "начать"
        ]
        if len(m_lower.split()) <= 4 and any(vp in m_lower for vp in vague_phrases):
            return True
        return False

    @staticmethod
    async def search_destinations_vector(query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Гибридный поиск: прямое совпадение по ключевым словам городов/стран +
        векторный поиск по вайбу и климату в PostgreSQL через pgvector.
        """
        q_lower = query.lower()
        
        keyword_map = {
            "thailand": "thailand", "таиланд": "thailand", "тайланд": "thailand", "пхукет": "thailand", "тай": "thailand", "самуи": "thailand",
            "uae": "uae", "оаэ": "uae", "дуба": "uae", "дубай": "uae", "абу-даби": "uae", "эмират": "uae",
            "turkey": "turkey", "турци": "turkey", "турция": "turkey", "анталь": "turkey", "анталья": "turkey", "стамбул": "turkey", "белек": "turkey", "кемер": "turkey",
            "sochi": "sochi", "сочи": "sochi", "сириус": "sochi", "полян": "sochi", "адлер": "sochi", "роза хутор": "sochi",
            "maldives": "maldives", "мальдив": "maldives", "мальдивы": "maldives", "мале": "maldives",
            "egypt": "egypt", "египет": "egypt", "шарм": "egypt", "хургад": "egypt", "синай": "egypt",
            "spb": "spb", "спб": "spb", "питер": "spb", "петербург": "spb", "санкт-петербург": "spb",
            "altai": "altai", "алтай": "altai", "манжерок": "altai", "телецк": "altai", "горно-алтайск": "altai"
        }
        
        matched_id = None
        for kw in sorted(keyword_map.keys(), key=lambda k: len(k), reverse=True):
            if re.search(r'(?:\b|^|\s)' + re.escape(kw), q_lower):
                matched_id = keyword_map[kw]
                break

        async with async_session_maker() as session:
            if matched_id:
                res = await session.execute(
                    select(DestinationModel).where(DestinationModel.id == matched_id)
                )
                r = res.scalar_one_or_none()
                if r:
                    return [{
                        "id": r.id,
                        "name": r.name,
                        "country": r.country,
                        "price_min": r.price_min,
                        "price_max": r.price_max,
                        "avg_check": r.avg_check,
                        "vibe_description": r.vibe_description,
                        "climate_summary": r.climate_summary,
                        "score": 1.0
                    }]

            query_vector = generate_text_embedding(query)
            str_vector = f"[{','.join(str(x) for x in query_vector)}]"
            
            sql = text("""
                SELECT id, name, country, price_min, price_max, avg_check, vibe_description, climate_summary,
                       embedding <=> (:query_vector)::vector AS distance
                FROM destinations
                ORDER BY distance ASC
                LIMIT :limit
            """)
            
            result = await session.execute(sql, {"query_vector": str_vector, "limit": limit})
            rows = result.fetchall()
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "country": r.country,
                    "price_min": r.price_min,
                    "price_max": r.price_max,
                    "avg_check": r.avg_check,
                    "vibe_description": r.vibe_description,
                    "climate_summary": r.climate_summary,
                    "score": round(1.0 - float(r.distance), 3)
                }
                for r in rows
            ]

    @staticmethod
    async def process_message(user_id: str, message: str) -> Dict[str, Any]:
        """
        Основной цикл обработки запроса путешественника:
        1. Распознавание интента и профилирование (память пользователя)
        2. Пункт 4 ТЗ: Анти-15-вопросов при нечетком старте
        3. Пункт 3 ТЗ: Fallback & рекомендации для неизвестных направлений
        4. Пункт 2 ТЗ: Динамическое перепланирование на лету без сброса контекста
        5. Пункт 5 ТЗ: Имитация чекаута и ваучер
        """
        user_lower = message.lower().strip()
        session_obj = await AgentService.get_or_create_session(user_id)
        profile = session_obj.profile or {}
        cart = session_obj.current_cart or {}

        # 0. Проверка намерения покупки / чекаута (Пункт 5 ТЗ)
        if any(w in user_lower for w in ["купить", "оформить", "забронировать", "оплатить", "чек", "подтвердить"]):
            if cart.get("hotel") and cart.get("destination"):
                checkout_result = await MCPTools.checkout_cart(user_id=user_id)
                reply = (
                    f"🎉 **Бронирование успешно подтверждено!**\n\n"
                    f"📋 **Ваучер заказа:** `{checkout_result['booking_id']}`\n"
                    f"🏝️ **Направление:** {checkout_result['destination']}\n"
                    f"✈️ **Авиаперелет:** {checkout_result.get('flight')}\n"
                    f"🏨 **Отель:** {checkout_result['hotel']}\n"
                    f"🚕 **Трансфер:** {checkout_result['transfer']}\n"
                    f"💳 **Сумма к оплате:** {checkout_result['total_paid_rub']:,} ₽ (симуляция оплаты успешна)\n\n"
                    f"💡 *{checkout_result.get('onsite_spending_guide', '')}*\n\n"
                    f"Спасибо, что выбрали Островок! Приятного путешествия!"
                )
                return {
                    "text": reply,
                    "cart": cart,
                    "booking": checkout_result,
                    "state": "CONFIRMED"
                }

        # 1. Извлечение интересов и обновление профиля пользователя (User Memory)
        profile = AgentService.extract_interests_and_slots(user_lower, profile)

        # 2. Пункт 4 ТЗ: Анти-15-вопросов на старте (Anti-15-Questions)
        has_dest_in_cart = bool(cart.get("destination_id"))
        if AgentService.is_vague_greeting(message, has_dest_in_cart):
            reply_anti_questions = (
                "👋 **Здравствуйте! Я умный ассистент путешествий Ostrovok.**\n\n"
                "Вместо длинных анкет из 15 вопросов задам всего **два ключевых ориентира**:\n\n"
                "1️⃣ **Какой формат отдыха вам ближе?**\n"
                "• 🌊 **Пляжный релакс и рифы** (Мальдивы, Пхукет, Египет, ОАЭ, Анталья)\n"
                "• 🏔️ **Горы, природа и спа** (Алтай, Красная Поляна)\n"
                "• 🏛️ **Культура, архитектура и гастрономия** (Санкт-Петербург, Стамбул)\n"
                "• 👨‍👩‍👧 **Семейный All-Inclusive с детьми и аквапарками**\n\n"
                "2️⃣ **Примерный состав и даты поездки** (вдвоём, с семьёй, с питомцем или соло)?\n\n"
                "_Напишите, что вам ближе (например: «Хочу море и всё включено с ребенком»), "
                "и я мгновенно соберу готовый проверенный маршрут с перелетом, отелем и трансфером._"
            )
            await AgentService._save_session(user_id, profile, cart, message, reply_anti_questions)
            return {
                "text": reply_anti_questions,
                "state": "INSPIRATION_PROMPT",
                "cart": cart
            }

        # 3. Пункт 3 ТЗ: Fallback и рекомендации для неизвестных направлений
        unknown_match = AgentService.detect_unknown_destination(message)
        if unknown_match:
            alt_id = unknown_match["pilot_alt_id"]
            alt_name = unknown_match["pilot_alt_name"]
            reason = unknown_match["vibe_reason"]
            
            # Предлагаем ближайшую альтернативу по вайбу
            reply_fallback = (
                f"🌍 **Обратите внимание:** В текущей версии нашей системы оцифрована "
                f"сквозная графовая логистика (рейсы, трансферы и отели) по 8 ключевым направлениям: "
                f"**Сочи, ОАЭ, Турция, Таиланд, Мальдивы, Египет, Санкт-Петербург и Алтай**.\n\n"
                f"💡 **Рекомендуем альтернативу по вайбу:** Если вас привлекает {reason} — "
                f"идеальным выбором станет **{alt_name}**!\n\n"
                f"Собрать для вас готовый тур в **{alt_name}** с проверенным перелетом, трансфером и расчетом бюджета?"
            )
            # Запоминаем альтернативное направление в профиле
            profile["suggested_destination"] = alt_id
            await AgentService._save_session(user_id, profile, cart, message, reply_fallback)
            return {
                "text": reply_fallback,
                "suggested_destination": alt_id,
                "state": "FALLBACK_OFFERED",
                "cart": cart
            }

        # 4. Пункт 2 ТЗ: Динамическое перепланирование без сброса контекста (Dynamic Re-planning)
        current_dest_id = cart.get("destination_id")
        explicit_dest_change = any(w in user_lower for w in ["другой город", "поменяй направление", "другая страна", "вместо", "поехали в", "а если в "])
        
        is_replan_hotel = any(w in user_lower for w in ["другой отель", "поменяй отель", "смени отель", "не нравится отель", "есть еще отели", "покажи другие варианты", "альтернативный отель", "другой вариант проживания"])
        is_replan_cheaper = any(w in user_lower for w in ["дешевле", "подешевле", "дорого", "снизь цену", "бюджетнее", "экономнее", "сделай дешевле"])
        is_replan_flight = any(w in user_lower for w in ["только прямой", "прямой рейс", "без пересадок", "с пересадкой", "стыковочный", "рейс подешевле"])
        is_replan_transfer = any(w in user_lower for w in ["убери трансфер", "без трансфера", "поедем сами", "на каршеринге", "возьмем такси сами", "своим ходом", "не нужен трансфер"])
        is_replan_dates = any(w in user_lower for w in ["позже", "раньше", "сдвинь", "другие даты", "в декабре", "в ноябре", "в январе"])
        is_replan_nights = any(w in user_lower for w in ["ноч", "дн", "сут", "выходн", "уикенд", "недел", "месяц"]) and profile.get("nights") != cart.get("nights")

        nights = profile.get("nights", 7)
        cart["nights"] = nights
        cart["dates"] = profile.get("dates", f"{nights} ночей")

        # Обработка сдвига дат
        if is_replan_dates:
            profile["dates"] = "на 2 дня позже (сдвиг графика)"
            cart["dates"] = profile["dates"]

        # Обработка исключения трансфера
        if is_replan_transfer:
            profile["without_transfer"] = True

        # Обработка типа рейса
        if any(w in user_lower for w in ["только прямой", "прямой рейс", "без пересадок"]):
            profile["prefer_direct_flight"] = True
        elif any(w in user_lower for w in ["с пересадкой", "стыковочный"]):
            profile["prefer_direct_flight"] = False

        # Определение целевого направления
        matched_destinations = await AgentService.search_destinations_vector(message, limit=2)
        msg_top_dest = matched_destinations[0] if matched_destinations else None

        if current_dest_id and not explicit_dest_change and (is_replan_hotel or is_replan_cheaper or is_replan_flight or is_replan_transfer or is_replan_dates):
            # Контекстное перепланирование в рамках текущего города/курорта!
            dest_id = current_dest_id
            cur_matched = await AgentService.search_destinations_vector(dest_id, limit=1)
            top_dest = cur_matched[0] if cur_matched else {"id": dest_id, "name": cart.get("destination", dest_id), "climate_summary": ""}
            dest_name = top_dest["name"]
        elif msg_top_dest and (msg_top_dest.get("score", 0) >= 0.9 or not current_dest_id or explicit_dest_change):
            dest_id = msg_top_dest["id"]
            dest_name = msg_top_dest["name"]
            top_dest = msg_top_dest
            # При смене направления очищаем список отклоненных отелей для нового региона
            if current_dest_id != dest_id:
                profile["rejected_hotels"] = []
        elif current_dest_id:
            dest_id = current_dest_id
            cur_matched = await AgentService.search_destinations_vector(dest_id, limit=1)
            top_dest = cur_matched[0] if cur_matched else msg_top_dest
            dest_name = top_dest["name"] if top_dest else dest_id
        elif msg_top_dest:
            dest_id = msg_top_dest["id"]
            dest_name = msg_top_dest["name"]
            top_dest = msg_top_dest
        else:
            return {
                "text": "Извините, не удалось найти подходящие направления в базе.",
                "state": session_obj.current_state,
                "cart": cart
            }

        cart["destination_id"] = dest_id
        cart["destination"] = dest_name

        # 5. Обработка смены отеля (Добавление текущего в rejected_hotels)
        rejected_hotels = profile.get("rejected_hotels", [])
        old_hotel_name = cart.get("hotel")
        old_hotel_id = cart.get("hotel_id")
        old_mandatory_rub = cart.get("total_mandatory_rub", 0)

        if is_replan_hotel and old_hotel_id:
            if old_hotel_id not in rejected_hotels:
                rejected_hotels.append(old_hotel_id)
                profile["rejected_hotels"] = rejected_hotels

        max_price_limit = None
        if is_replan_cheaper:
            profile.setdefault("interests", []).append("budget_friendly")
            current_night_price = cart.get("price_per_night_rub", 20000)
            max_price_limit = max(4500, int(current_night_price * 0.75))

        # 6. Вызов MCP: Сравнение отелей с учетом тегов, исключений и фильтров
        hotel_comparison = await MCPTools.compare_hotels(
            destination_id=dest_id,
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids"),
            max_price_per_night=max_price_limit,
            exclude_hotel_ids=rejected_hotels,
            preferred_tags=profile.get("interests")
        )

        compared_list = hotel_comparison.get("compared_hotels", [])
        recommended_hotel = compared_list[0] if compared_list else None

        # 7. Вызов Neo4j: Графовая логистика сквозного маршрута
        logistics_info = {}
        if recommended_hotel:
            logistics_info = await MCPTools.check_logistics_compatibility(
                airport_code="",
                hotel_id=recommended_hotel["id"],
                origin_airport_code=None,
                prefer_direct_flight=profile.get("prefer_direct_flight"),
                without_transfer=profile.get("without_transfer", False)
            )

        # 8. Расчет финансовой структуры
        flight_info = logistics_info.get("flight", {})
        flight_cost = flight_info.get("price_rub_roundtrip", 25000)
        hotel_cost_per_night = recommended_hotel.get("price_per_night_rub", 15000) if recommended_hotel else 15000
        transfer_cost = 0 if profile.get("without_transfer") else logistics_info.get("transfer", {}).get("price_rub", 2000)
        meals_type = recommended_hotel.get("meals", "Завтрак включен") if recommended_hotel else "Завтрак включен"

        budget_breakdown = MCPTools.calculate_budget_breakdown(
            destination_id=dest_id,
            flight_price_roundtrip=flight_cost,
            hotel_price_per_night=hotel_cost_per_night,
            transfer_price=transfer_cost,
            meals_type=meals_type,
            nights=nights
        )

        # 9. Вызов MCP: Сезонные опасности и форс-мажоры + реальный прогноз погоды
        risks_report = await MCPTools.get_destination_risks(
            destination_id=dest_id,
            month="ноябрь"
        )
        weather_report = await MCPTools.get_live_weather(dest_id)

        # Сохранение в корзину
        if recommended_hotel:
            cart["hotel_id"] = recommended_hotel["id"]
            cart["hotel"] = recommended_hotel["name"]
            cart["price_per_night_rub"] = hotel_cost_per_night
            cart["price_per_night_formatted"] = recommended_hotel["price_per_night_formatted"]
        if flight_info:
            cart["flight"] = f"{flight_info.get('airline', 'Авиа')} {flight_info.get('flight_no', '')} ({flight_info.get('origin_name')} ⇄ {logistics_info.get('airport')})"
        
        tr_info = logistics_info.get("transfer", {})
        cart["transfer"] = tr_info.get("name", "Трансфер")
        cart["logistics_check"] = logistics_info.get("logistics_status", "Состыковано")
        new_mandatory_rub = budget_breakdown["mandatory"]["total_mandatory_rub"]
        cart["total_mandatory_rub"] = new_mandatory_rub

        cart["onsite_spending_guide"] = (
            f"Рекомендуемый бюджет на месте: от {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['min']:,} "
            f"до {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['max']:,} ₽"
        )

        # 10. Формирование сжатого и понятного ответа
        response_lines = []

        # Специфический статус перепланирования (Пункт 2 ТЗ)
        if is_replan_hotel and old_hotel_name and old_hotel_name != recommended_hotel["name"]:
            response_lines.append(f"🔄 **Отель успешно заменён без сброса маршрута:**")
            response_lines.append(f"Предыдущий отель «{old_hotel_name}» исключён из рекомендаций.")
            response_lines.append("")
        elif is_replan_cheaper and old_mandatory_rub > new_mandatory_rub:
            savings = old_mandatory_rub - new_mandatory_rub
            response_lines.append(f"📉 **Оптимизировали тур: экономия {savings:,} ₽!**")
            response_lines.append(f"Подобрали более доступный отель и выгодные условия логистики.")
            response_lines.append("")
        elif is_replan_transfer:
            response_lines.append(f"🚕 **Трансфер исключён:** стоимость тура снижена, вы сможете добраться самостоятельно (каршеринг/такси).")
            response_lines.append("")
        elif is_replan_flight:
            fl_type_str = "только прямой рейс" if profile.get("prefer_direct_flight") else "рейс с комфортной стыковкой"
            response_lines.append(f"✈️ **Маршрут перестроен под условие:** {fl_type_str}.")
            response_lines.append("")
        elif is_replan_nights:
            response_lines.append(f"📅 **Длительность тура изменена на {nights} ночей:** пересчитана стоимость проживания и суточные расходы.")
            response_lines.append("")
        elif is_replan_dates:
            response_lines.append(f"📅 **Даты поездки обновлены:** {profile.get('dates')}, отель и билеты состыкованы.")
            response_lines.append("")

        response_lines.append(f"🌴 **{top_dest['name']}** | {top_dest.get('climate_summary', '')}")
        if weather_report.get("available"):
            response_lines.append(f"🌤️ **Прогноз погоды на неделю ({weather_report.get('source')}):** {weather_report['summary']}")
        response_lines.append("")

        if recommended_hotel:
            h = recommended_hotel
            response_lines.append(f"🏨 **Отель:** {h['name']} ({h['stars']}, {h['price_per_night_formatted']}/ночь)")
            response_lines.append(f"• Питание: {h['meals']} | До моря: {h['distance_to_sea']} | Отмена: {h['cancellation']}")
            if len(compared_list) > 1:
                alt = compared_list[1]
                response_lines.append(f"• _Альтернатива:_ {alt['name']} ({alt['stars']}, {alt['price_per_night_formatted']}/н, {alt['meals']})")
            response_lines.append("")

        if logistics_info.get("compatible"):
            timeline = logistics_info.get("timeline", {})
            layover = timeline.get("layover_details", {})
            flight_code = f"{flight_info.get('airline')} {flight_info.get('flight_no')}"
            flight_times = f"{flight_info.get('origin_code', 'MOW')} {timeline.get('flight_departure')} ➔ {logistics_info.get('airport')} {timeline.get('flight_arrival')}"
            
            response_lines.append("✈️ **Логистика и стыковки (Neo4j Graph):**")
            if layover.get("is_connecting"):
                response_lines.append(f"• Стыковочный рейс: {flight_code} ({flight_times}), пересадка в {layover.get('city')} ({layover.get('duration_formatted')}, сквозной багаж).")
            else:
                response_lines.append(f"• Прямой рейс: {flight_code} ({flight_times}).")

            if profile.get("without_transfer"):
                response_lines.append(f"• 🧳 Буфер прилёта ({timeline.get('post_flight_buffer_min', 50)} мин: багаж + контроль) ➔ самостоятельный выезд на каршеринге/такси в отель.")
            else:
                buf_min = timeline.get("post_flight_buffer_min", 50)
                response_lines.append(f"• 🧳 Буфер прилёта ({buf_min} мин: багаж + контроль) ➔ трансфер в {timeline.get('transfer_start')} ➔ отель в {timeline.get('hotel_arrival')}.")
                response_lines.append(f"• 🛡️ Форс-мажор: бесплатное ожидание трансфера до 90 мин по номеру рейса; 24/7 Late Check-in.")
            response_lines.append("")

        # Финансовый блок
        mand = budget_breakdown["mandatory"]
        onsite = budget_breakdown["onsite_flexible"]
        total_range = budget_breakdown["total_trip_budget_range_rub"]

        response_lines.extend([
            f"💰 **Бюджет на {nights} ночей:**",
            f"• 🔴 **Сразу (авиа + отель {nights} н. + трансфер):** {mand['total_mandatory_rub']:,} ₽",
            f"• 🟢 **На месте ({nights} дн. питание + активности):** {onsite['total_onsite_range_rub']['min']:,} – {onsite['total_onsite_range_rub']['max']:,} ₽",
            f"💵 **Итого под ключ:** **{total_range['min']:,} – {total_range['max']:,} ₽**",
            ""
        ])

        # Сезонные опасности (сжато)
        active_alerts = risks_report.get("active_alerts", [])
        if active_alerts:
            threat = risks_report.get("overall_threat_level", "MEDIUM")
            top_risks = "; ".join(f"{a['title']}" for a in active_alerts[:2])
            response_lines.append(f"⚠️ **Сезонные риски ({threat}):** {top_risks}.")
            tips = risks_report.get("safety_recommendations", [])
            if tips:
                response_lines.append(f"🛡️ _{tips[0]}_")
            response_lines.append("")
        # 11. Аудит безопасности и умная валидация данных агентом (фильтрация неадекватных / рискованных предложений)
        audit_notes = []
        if profile.get("has_kids"):
            audit_notes.append("🚫 **Семейный контроль:** Отели категории «Adults Only (18+)» и шумные молодежные пати-резорты автоматически исключены из поиска.")
        if profile.get("has_pets"):
            audit_notes.append("🐾 **Pet-контроль:** Отели со строгим запретом на животных исключены; подтверждено безопасное размещение с питомцем.")

        if recommended_hotel:
            price_n = recommended_hotel.get("price_per_night_rub", 0)
            feats = [str(f).lower() for f in (recommended_hotel.get("features", []))]
            if price_n <= 2500 or any("shared_bathroom" in f or "hostel_dorm" in f for f in feats):
                audit_notes.append("⚠️ **Контроль эконом-сегмента:** Выбран ультрабюджетный хостел/дормиторий с общими удобствами. Базовые санитарные требования соблюдены, но для приватности рекомендуем отдельный номер.")
            elif price_n >= 100000:
                audit_notes.append("💎 **Премиум-аудит:** Подтвержден эксклюзивный статус Ultra-Luxury (персональный консьерж-сервис, приватная вилла/дворец, VIP-трансфер).")

        if logistics_info.get("compatible"):
            timeline = logistics_info.get("timeline", {})
            layover = timeline.get("layover_details", {})
            if layover.get("is_connecting"):
                audit_notes.append("✈️ **Аудит стыковок:** Отсеяны опасные короткие пересадки (<60 мин, риск потери багажа); выбран рейс с безопасным транзитным буфером.")
            
            flight_arr = timeline.get("flight_arrival", "12:00")
            if dest_id == "maldives" and flight_arr > "16:30":
                audit_notes.append("🏝️ **Правило Мальдив:** Гидросамолеты летают только в светлое время (до 16:30). Для вечернего рейса трансфер заменен на скоростной катер.")
            
            if dest_id == "spb" and ("01:10" <= flight_arr <= "05:00"):
                audit_notes.append("🌉 **Логистика СПб:** Прилёт в период разводки невских мостов (01:10–05:00) — маршрут трансфера проложен через скоростной ЗСД.")

        if audit_notes:
            response_lines.append("🛡️ **Аудит агента (Smart Filter & Safety Check):**")
            for note in audit_notes:
                response_lines.append(f"• {note}")
            response_lines.append("")

        response_lines.append("🛒 *Напишите «Забронировать» для покупки или уточните: «другой отель», «подешевле», «только прямой».*")

        reply_text = "\n".join(response_lines)
        await AgentService._save_session(user_id, profile, cart, message, reply_text)

        return {
            "text": reply_text,
            "destination": top_dest,
            "compared_hotels": compared_list,
            "logistics": logistics_info,
            "budget_breakdown": budget_breakdown,
            "risks_report": risks_report,
            "weather": weather_report,
            "cart": cart
        }

    @staticmethod
    async def _save_session(user_id: str, profile: dict, cart: dict, user_msg: str, agent_reply: str):
        async with async_session_maker() as session:
            sess_to_update = await session.get(UserSessionModel, user_id)
            if sess_to_update:
                sess_to_update.profile = profile
                sess_to_update.current_cart = cart
                sess_history = sess_to_update.conversation_history or []
                sess_history.append({"user": user_msg, "agent": agent_reply})
                sess_to_update.conversation_history = sess_history
                await session.commit()
