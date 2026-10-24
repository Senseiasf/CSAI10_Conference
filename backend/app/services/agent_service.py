import json
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import select, text

from app.config import settings
from app.database import async_session_maker
from app.models import DestinationModel, HotelModel, RiskAlertModel, UserSessionModel
from app.services.vector_service import generate_text_embedding
from app.mcp.tools import MCPTools

logger = logging.getLogger(__name__)


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
                        "dates": "ноябрь (7 ночей)"
                    },
                    current_cart={},
                    conversation_history=[]
                )
                session.add(user_session)
                await session.commit()
                await session.refresh(user_session)
            return user_session

    @staticmethod
    async def search_destinations_vector(query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Гибридный поиск: прямое совпадение по ключевым словам городов/стран +
        векторный поиск по вайбу и климату в PostgreSQL через pgvector.
        """
        q_lower = query.lower()
        
        keyword_map = {
            "thailand": "thailand", "таиланд": "thailand", "тайланд": "thailand", "пхукет": "thailand", "тай": "thailand",
            "uae": "uae", "оаэ": "uae", "дуба": "uae", "абу-даби": "uae", "эмират": "uae",
            "turkey": "turkey", "турци": "turkey", "анталь": "turkey", "стамбул": "turkey", "белек": "turkey", "кемер": "turkey",
            "sochi": "sochi", "сочи": "sochi", "сириус": "sochi", "полян": "sochi", "адлер": "sochi"
        }
        
        matched_id = None
        for kw, dest_id in keyword_map.items():
            if kw in q_lower:
                matched_id = dest_id
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
        - Понимание скрытых параметров
        - Сравнение отелей (MCP)
        - Графовая логистика перелёта и трансфера (Neo4j)
        - Сезонные риски и форс-мажоры (MCP)
        - Детальный финансовый расчёт (Обязательные платежи vs Траты не сразу)
        - Имитация бронирования
        """
        user_lower = message.lower()
        session_obj = await AgentService.get_or_create_session(user_id)
        profile = session_obj.profile or {}
        cart = session_obj.current_cart or {}

        # 0. Проверка намерения покупки / чекаута (Пункт 5 ТЗ)
        if any(w in user_lower for w in ["купить", "оформить", "забронировать", "оплатить", "чек"]):
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

        # 1. Извлечение условий (Slot Filling)
        if any(w in user_lower for w in ["собак", "питомц", "кошк", "пес"]):
            profile["has_pets"] = True
        if any(w in user_lower for w in ["ребенок", "дет", "малыш", "семь"]):
            profile["has_kids"] = True
        if "позже" in user_lower or "сдвин" in user_lower:
            profile["dates"] = "на 2 дня позже (сдвиг графика)"
            cart["dates"] = profile["dates"]

        # 2. Направление (сохраняем при контекстном уточнении, либо обновляем при новом запросе с упоминанием направления)
        current_dest_id = cart.get("destination_id")
        explicit_dest_change = any(w in user_lower for w in ["другой город", "поменяй направление", "другая страна", "вместо"])
        
        matched_destinations = await AgentService.search_destinations_vector(message, limit=2)
        msg_top_dest = matched_destinations[0] if matched_destinations else None

        # Если в запросе пользователя явно указано направление или сессия пуста / смена направления
        if msg_top_dest and (msg_top_dest.get("score", 0) >= 0.9 or not current_dest_id or explicit_dest_change):
            dest_id = msg_top_dest["id"]
            dest_name = msg_top_dest["name"]
            top_dest = msg_top_dest
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

        # 3. Вызов MCP: Сравнение отелей (Пункт 3 ТЗ)
        hotel_comparison = await MCPTools.compare_hotels(
            destination_id=dest_id,
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids")
        )

        compared_list = hotel_comparison.get("compared_hotels", [])
        recommended_hotel = compared_list[0] if compared_list else None

        # 4. Вызов MCP: Сезонные опасности и форс-мажоры (Пункт 4 ТЗ)
        risks_report = await MCPTools.get_destination_risks(
            destination_id=dest_id,
            month="ноябрь"
        )

        # 5. Вызов Neo4j: Графовая логистика сквозного маршрута (Авиа + Трансфер + Отель)
        logistics_info = {}
        if recommended_hotel:
            logistics_info = await MCPTools.check_logistics_compatibility(
                airport_code="",
                hotel_id=recommended_hotel["id"],
                origin_airport_code=None
            )

        # 6. Расчет детальной финансовой структуры (Обязательные платежи vs Траты не сразу)
        flight_info = logistics_info.get("flight", {})
        flight_cost = flight_info.get("price_rub_roundtrip", 25000)
        hotel_cost_per_night = recommended_hotel.get("price_per_night_rub", 15000) if recommended_hotel else 15000
        transfer_cost = logistics_info.get("transfer", {}).get("price_rub", 2000)
        meals_type = recommended_hotel.get("meals", "Завтрак включен") if recommended_hotel else "Завтрак включен"

        budget_breakdown = MCPTools.calculate_budget_breakdown(
            destination_id=dest_id,
            flight_price_roundtrip=flight_cost,
            hotel_price_per_night=hotel_cost_per_night,
            transfer_price=transfer_cost,
            meals_type=meals_type,
            nights=7
        )

        # Сохранение в корзину
        if recommended_hotel:
            cart["hotel"] = recommended_hotel["name"]
            cart["price_per_night_rub"] = hotel_cost_per_night
            cart["price_per_night_formatted"] = recommended_hotel["price_per_night_formatted"]
        if flight_info:
            cart["flight"] = f"{flight_info.get('airline', 'Авиа')} {flight_info.get('flight_no', '')} ({flight_info.get('origin_name')} ⇄ {logistics_info.get('airport')})"
        if logistics_info.get("transfer"):
            cart["transfer"] = logistics_info["transfer"]["name"]
            cart["logistics_check"] = logistics_info.get("logistics_status", "Состыковано")

        cart["total_mandatory_rub"] = budget_breakdown["mandatory"]["total_mandatory_rub"]
        cart["onsite_spending_guide"] = (
            f"Рекомендуемый бюджет на месте: от {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['min']:,} "
            f"до {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['max']:,} ₽"
        )

        # 7. Формирование сжатого и структурированного ответа пользователю
        response_lines = [
            f"🌴 **{top_dest['name']}** | {top_dest['climate_summary']}",
            ""
        ]

        if recommended_hotel:
            h = recommended_hotel
            response_lines.append(f"🏨 **Отель:** {h['name']} ({h['stars']}, {h['price_per_night_formatted']}/ночь)")
            response_lines.append(f"• Питание: {h['meals']} | До моря: {h['distance_to_sea']} | Отмена: {h['cancellation']}")
            if len(compared_list) > 1:
                alt = compared_list[1]
                response_lines.append(f"• _Альтернатива:_ {alt['name']} ({alt['stars']}, {alt['price_per_night_formatted']}/н, {alt['meals']})")
            response_lines.append("")

        if logistics_info.get("compatible"):
            tr = logistics_info.get("transfer", {})
            timeline = logistics_info.get("timeline", {})
            layover = timeline.get("layover_details", {})
            
            flight_code = f"{flight_info.get('airline')} {flight_info.get('flight_no')}"
            flight_times = f"{flight_info.get('origin_code', 'MOW')} {timeline.get('flight_departure')} ➔ {logistics_info.get('airport')} {timeline.get('flight_arrival')}"
            
            response_lines.append("✈️ **Логистика и стыковки (Neo4j Graph):**")
            if layover.get("is_connecting"):
                response_lines.append(f"• Рейс с пересадкой: {flight_code} ({flight_times}), стыковка {layover.get('city')} ({layover.get('duration_formatted')}, сквозной билет).")
            else:
                response_lines.append(f"• Прямой рейс: {flight_code} ({flight_times}).")

            buf_min = timeline.get("post_flight_buffer_min", 50)
            response_lines.append(f"• 🧳 Буфер прилёта ({buf_min} мин: багаж + контроль) ➔ трансфер в {timeline.get('transfer_start')} ➔ отель в {timeline.get('hotel_arrival')}.")
            response_lines.append(f"• 🛡️ Форс-мажор: бесплатное ожидание трансфера до 90 мин по номеру рейса; 24/7 Late Check-in в отеле.")
            response_lines.append("")

        # Финансовый блок (компактно)
        mand = budget_breakdown["mandatory"]
        onsite = budget_breakdown["onsite_flexible"]
        total_range = budget_breakdown["total_trip_budget_range_rub"]

        response_lines.extend([
            "💰 **Бюджет на 7 дней:**",
            f"• 🔴 **Сразу (авиа + отель + трансфер):** {mand['total_mandatory_rub']:,} ₽",
            f"• 🟢 **На месте (питание + активности):** {onsite['total_onsite_range_rub']['min']:,} – {onsite['total_onsite_range_rub']['max']:,} ₽",
            f"💵 **Итого под ключ:** **{total_range['min']:,} – {total_range['max']:,} ₽**",
            ""
        ])

        # Сезонные опасности (кратко)
        active_alerts = risks_report.get("active_alerts", [])
        if active_alerts:
            threat = risks_report.get("overall_threat_level", "MEDIUM")
            top_risks = "; ".join(f"{a['title']}" for a in active_alerts[:2])
            response_lines.append(f"⚠️ **Сезонные риски ({threat}):** {top_risks}.")
            tips = risks_report.get("safety_recommendations", [])
            if tips:
                response_lines.append(f"🛡️ _{tips[0]}_")
            response_lines.append("")

        response_lines.append("🛒 *Напишите «Забронировать» для покупки или уточните параметры: даты, отель, бюджет.*")

        reply_text = "\n".join(response_lines)

        # Сохранение сессии
        async with async_session_maker() as session:
            sess_to_update = await session.get(UserSessionModel, user_id)
            if sess_to_update:
                sess_to_update.profile = profile
                sess_to_update.current_cart = cart
                sess_history = sess_to_update.conversation_history or []
                sess_history.append({"user": message, "agent": reply_text})
                sess_to_update.conversation_history = sess_history
                await session.commit()

        return {
            "text": reply_text,
            "destination": top_dest,
            "compared_hotels": compared_list,
            "logistics": logistics_info,
            "budget_breakdown": budget_breakdown,
            "risks_report": risks_report,
            "cart": cart
        }
