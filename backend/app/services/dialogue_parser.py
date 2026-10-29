import json
import re
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.domain.dialogue import ChangeOperation, DialogueDecision, DialogueIntent, StateChange
from app.domain.trip import TripQuote, TripState
from app.services.llm_service import LLMService, LLMServiceError
from app.services.trip_logistics import is_logistics_question
from app.services.trip_state import (
    current_trip_question, estimate_request, hotel_selection, is_trip_edit,
    matching_hotels, transfer_preference,
)


DIALOGUE_DECISION_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_dialogue_decision",
        "description": "Вернуть единственное решение диалога без выполнения изменений.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": ["question", "comparison", "modify_trip", "plan_trip", "clarify"]},
                "topic": {"type": ["string", "null"]},
                "mode": {"type": ["string", "null"], "enum": ["min", "max", None]},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "field": {"type": "string"},
                            "operation": {"type": "string", "enum": ["set", "clear", "add", "remove"]},
                            "value": {},
                        },
                        "required": ["field", "operation"],
                    },
                },
                "clarification": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
                "narrative": {"type": ["string", "null"]},
            },
            "required": ["intent", "changes"],
        },
    },
}]


class DialogueParserError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.public_message = message


class DialogueParser:
    async def parse(
        self,
        message: str,
        state: TripState,
        context: dict[str, Any] | None = None,
        quote: TripQuote | None = None,
    ) -> DialogueDecision:
        context = context or {}
        if settings.is_mock_llm:
            return self._parse_mock(message, state, context)
        try:
            return await self._parse_live(message, state, context, quote)
        except (LLMServiceError, DialogueParserError):
            raise
        except Exception as exc:
            raise DialogueParserError("api", self._safe_error(exc)) from exc

    async def _parse_live(
        self,
        message: str,
        state: TripState,
        context: dict,
        quote: TripQuote | None,
    ) -> DialogueDecision:
        client = LLMService.get_client()
        if client is None:
            raise DialogueParserError("configuration", "Live parser требует LLM_MODE=live")
        payload = {
            "trip_state": state.model_dump(mode="json"),
            "trip_quote": quote.model_dump(mode="json") if quote else None,
            "selected_offers": context.get("selected_offers") or state.selected_offers,
            "alternative_offers": context.get("alternative_offers") or (context.get("cart") or {}).get("compared_hotels") or [],
            "conversation": (context.get("conversation_history") or [])[-3:],
        }
        prompt = (
            "Ты основной языковой обработчик туристического диалога. Верни только DialogueDecision через инструмент. "
            "question никогда не меняет поездку и имеет changes=[]; comparison рассчитывает альтернативы без смены выбранного варианта; "
            "modify_trip содержит только явно запрошенные изменения; plan_trip — первичный подбор; clarify — неясность. "
            "Отсутствие операции сохраняет прежнее значение, явное false снимает условие. Не угадывай даты, суммы и идентификаторы. "
            "Ссылки 'тот отель', 'первый вариант' разрешай только по alternative_offers/selected_offers до hotel_id; если ссылка неоднозначна — clarify. "
            "Для отказа от трансфера без других правок верни ровно одну операцию field=without_transfer, operation=set, value=true. "
            "Допустимые поля: destination_id, checkin_date, checkout_date, nights, adults_count, kids_count, has_pets, rooms_count, "
            "budget_rub, without_transfer, prefer_direct_flight, flight_time_pref, origin_airport, private_room_required, interests, "
            "hotel_id, rejected_destinations, rejected_hotels, replan_action. Для вопроса о пересадке topic=connection_risk, "
            "о сохранённом плане topic=current_trip, о границе цены topic=budget_estimate. Не выполняй денежные расчёты."
        )
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({"message": message, "context": payload}, ensure_ascii=False)},
                ],
                tools=DIALOGUE_DECISION_TOOL,
                tool_choice={"type": "function", "function": {"name": "submit_dialogue_decision"}},
                temperature=0,
                max_tokens=1200,
            )
            message_obj = response.choices[0].message
            raw = None
            if message_obj.tool_calls:
                raw = message_obj.tool_calls[0].function.arguments
            elif message_obj.content:
                raw = message_obj.content
            if not raw:
                raise DialogueParserError("invalid_response", "LLM не вернула DialogueDecision")
            decision = DialogueDecision.model_validate(json.loads(raw))
            LLMService._last_error = None
            return self._resolve_hotel_reference(decision, payload["alternative_offers"])
        except DialogueParserError:
            raise
        except (ValidationError, json.JSONDecodeError) as exc:
            raise DialogueParserError("invalid_response", f"Некорректный DialogueDecision: {exc}") from exc
        except Exception as exc:
            raise DialogueParserError("api", self._safe_error(exc)) from exc

    def _resolve_hotel_reference(self, decision: DialogueDecision, offers: list[dict]) -> DialogueDecision:
        for change in decision.changes:
            if change.field != "hotel_id" or not isinstance(change.value, str):
                continue
            value = change.value
            candidates = [offer for offer in offers if offer.get("id")]
            if any(offer["id"] == value for offer in candidates):
                continue
            normalized = value.casefold().strip()
            matches = [offer for offer in candidates if normalized in str(offer.get("name", "")).casefold()]
            if len(matches) == 1:
                change.value = matches[0]["id"]
            elif len(matches) > 1:
                return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification="Какой именно отель вы имеете в виду?", changes=[])
        return decision

    def _parse_mock(self, message: str, state: TripState, context: dict) -> DialogueDecision:
        if is_logistics_question(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="connection_risk")
        if current_trip_question(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="current_trip")
        estimate = estimate_request(message)
        if estimate and not is_trip_edit(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="budget_estimate", mode=estimate)
        if re.search(r"\b(?:сравни|соседн\w*\s+дат|календар\w*\s+(?:цен|тариф|выгод))", message.lower()):
            return DialogueDecision(intent=DialogueIntent.COMPARISON, topic="dates")
        if is_trip_edit(message):
            selected_query = hotel_selection(message)
            if selected_query:
                offers = context.get("alternative_offers") or []
                if re.fullmatch(r"(?:тот|этот|прежний|текущий)\s+отель", selected_query.lower()):
                    hotel_id = state.selected_offers.get("hotel_id")
                    if hotel_id:
                        return DialogueDecision(intent=DialogueIntent.MODIFY, changes=[
                            StateChange(field="hotel_id", operation=ChangeOperation.SET, value=hotel_id),
                        ])
                    if len(offers) != 1:
                        return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification="Какой именно отель вы имеете в виду?")
                matches = matching_hotels(selected_query, offers)
                if len(matches) == 1:
                    return DialogueDecision(intent=DialogueIntent.MODIFY, changes=[
                        StateChange(field="hotel_id", operation=ChangeOperation.SET, value=matches[0]["id"]),
                    ])
                if len(matches) > 1:
                    return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification="Какой именно отель вы имеете в виду?")
            changes = []
            transfer = transfer_preference(message)
            if transfer is not None:
                changes.append(StateChange(field="without_transfer", operation=ChangeOperation.SET, value=transfer))
            nights = re.search(r"\b(\d{1,2})\s*(?:ноч|дн|сут)", message.lower())
            if nights:
                changes.append(StateChange(field="nights", operation=ChangeOperation.SET, value=int(nights.group(1))))
            if not changes:
                changes.append(StateChange(field="replan_action", operation=ChangeOperation.SET, value="modify"))
            return DialogueDecision(intent=DialogueIntent.MODIFY, changes=changes)
        return DialogueDecision(intent=DialogueIntent.PLAN)

    def _safe_error(self, exc: Exception) -> str:
        message = str(exc)
        key = getattr(settings, "OPENAI_API_KEY", "")
        if key:
            message = message.replace(key, "<redacted>")
        return message[:500]


def decision_context(session: Any, state: TripState, cart: dict) -> dict[str, Any]:
    return {
        "selected_offers": deepcopy(state.selected_offers),
        "alternative_offers": deepcopy(cart.get("compared_hotels") or []),
        "conversation_history": deepcopy(getattr(session, "conversation_history", []) or []),
        "cart": deepcopy(cart),
    }
