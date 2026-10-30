from copy import deepcopy
from datetime import date
from typing import Any, Awaitable, Callable

from app.config import settings
from app.domain.dialogue import ChangeOperation, DialogueDecision, DialogueIntent, StateChange
from app.domain.trip import PendingAction, TripQuote, TripState
from app.services.booking_stub import booking_stub, detect_booking_action
from app.services.dialogue_parser import DialogueParser, DialogueParserError, decision_context
from app.services.llm_service import LLMService
from app.services.quote_service import compare_quotes
from app.services.trip_logistics import connection_summary, describe_saved_logistics
from app.services.trip_state import describe_budget_estimate, describe_current_trip
from app.services.trip_state_store import TripStateStore


class TripValidationError(ValueError):
    pass


STATE_CHANGING_INTENTS = {
    DialogueIntent.MODIFY,
    DialogueIntent.PLAN,
    DialogueIntent.SEARCH_ALTERNATIVES,
    DialogueIntent.SELECT_OFFER,
    DialogueIntent.CORRECTION,
    DialogueIntent.NEW_TRIP,
}
ALTERNATIVE_INTENTS = {DialogueIntent.COMPARISON, DialogueIntent.SEARCH_ALTERNATIVES}


class TripController:
    def __init__(self, parser: DialogueParser | None = None):
        self.parser = parser or DialogueParser()

    async def handle(
        self,
        user_id: str,
        message: str,
        session: Any,
        plan_executor: Callable[..., Awaitable[dict]],
        save_session: Callable[..., Awaitable[None]],
        progress_callback: Any = None,
    ) -> dict:
        profile = deepcopy(session.profile or {})
        cart = deepcopy(session.current_cart or {})
        state, quote = TripStateStore.load(session)
        previous_state = state.model_copy(deep=True)

        booking_action = detect_booking_action(message)
        if booking_action:
            result = booking_stub(booking_action, cart)
            await save_session(user_id, profile, cart, message, result["text"])
            return self._enrich(result, None, profile, cart, quote, state=state)

        try:
            decision = await self.parser.parse(
                message,
                state,
                decision_context(session, state, cart),
                quote,
            )
            LLMService._last_error = None
            if decision.intent == DialogueIntent.NEW_TRIP:
                state = TripState()
                profile = {}
                cart = {}
                quote = None
            if decision.intent in STATE_CHANGING_INTENTS and decision.changes:
                state = self.apply_changes(state, decision.changes)
            if decision.intent in STATE_CHANGING_INTENTS:
                state.revision = previous_state.revision + 1
                state.explicit_fields = sorted(set(state.explicit_fields) | {
                    change.field for change in decision.changes if change.field != "replan_action"
                })
                profile = state.apply_to_profile(profile)
            if decision.intent == DialogueIntent.CLARIFY:
                state.pending_action = self.pending_for_decision(decision, state)
                profile = state.apply_to_profile(profile)
            if decision.narrative:
                profile["llm_ai_intro"] = decision.narrative
            hotel_change = next((change for change in decision.changes if change.field == "hotel_id"), None)
            if hotel_change:
                profile["_resolved_hotel_id"] = hotel_change.value
                profile["_resolved_hotel_query"] = hotel_change.value
            destination_change = next((change for change in decision.changes if change.field == "destination_id"), None)
            if destination_change and destination_change.value != cart.get("destination_id"):
                profile["_structured_destination_changed"] = True
            profile["_structured_decision"] = decision.model_dump(mode="json")
            profile["_structured_changed_fields"] = [change.field for change in decision.changes]
            profile["_structured_budget_changed"] = state.budget_rub != previous_state.budget_rub
            profile["_structured_dates_changed"] = state.dates != previous_state.dates
        except (DialogueParserError, TripValidationError) as exc:
            code = getattr(exc, "code", "validation")
            public_message = getattr(exc, "public_message", str(exc))
            LLMService._last_error = {"code": code, "message": public_message}
            diagnostics = {"llm": {**LLMService.get_diagnostics(), "status": "error", "error": {"code": code, "message": public_message}}}
            reasoning = deepcopy(cart.get("reasoning_steps") or [])
            reasoning.append({"stage": "llm_error", "title": "Диагностика языкового обработчика", "detail": public_message})
            reply = "Не удалось надёжно обработать запрос. Текущая поездка сохранена без изменений. Уточните, пожалуйста, действие."
            await save_session(user_id, session.profile or {}, cart, message, reply)
            return {
                "text": reply,
                "state": "CLARIFICATION_REQUIRED",
                "cart": cart,
                "reasoning_steps": reasoning,
                "diagnostics": diagnostics,
                "trip_state": state.model_dump(mode="json"),
                "trip_quote": quote.model_dump(mode="json") if quote else None,
                "dialogue_decision": None,
            }

        if decision.intent in (DialogueIntent.QUESTION, DialogueIntent.CLARIFY):
            for key in (
                "_structured_decision", "_structured_changed_fields", "_structured_budget_changed",
                "_structured_dates_changed", "_structured_destination_changed",
                "_resolved_hotel_id", "_resolved_hotel_query",
            ):
                profile.pop(key, None)
            result = self._read_only_result(decision, profile, cart, message)
            session.profile = deepcopy(profile)
            session.current_cart = deepcopy(cart)
            await save_session(user_id, profile, cart, message, result["text"])
            return self._enrich(result, decision, profile, cart, quote, state=state)

        if decision.intent == DialogueIntent.NEW_TRIP and not decision.changes and not decision.search:
            reply = "Начинаем новую поездку. Расскажите, какой отдых хотите, когда и кто едет."
            session.profile = deepcopy(profile)
            session.current_cart = deepcopy(cart)
            await save_session(user_id, profile, cart, message, reply)
            return self._enrich({"text": reply, "state": "NEW_TRIP", "cart": cart}, decision, profile, cart, None, state=state)

        missing = self.outstanding_fields(decision, state)
        if missing:
            kind = "search_alternatives" if decision.intent == DialogueIntent.SEARCH_ALTERNATIVES else "plan_trip"
            attempts = (state.pending_action.attempts + 1) if state.pending_action else 0
            state.pending_action = PendingAction(kind=kind, missing_fields=missing, question=self.missing_question(missing), attempts=attempts)
            profile = state.apply_to_profile(profile)
            reply = state.pending_action.question
            result = {"text": reply, "state": "CLARIFICATION_REQUIRED", "cart": cart, "pending_action": state.pending_action.model_dump(mode="json")}
            session.profile = deepcopy(profile)
            session.current_cart = deepcopy(cart)
            await save_session(user_id, profile, cart, message, reply)
            return self._enrich(result, decision, profile, cart, quote, state=state)
        if decision.intent in STATE_CHANGING_INTENTS:
            state.pending_action = None
            profile = state.apply_to_profile(profile)
        if (decision.intent != DialogueIntent.PLAN
                and any(change.field != "replan_action" for change in decision.changes)
                and not profile.get("llm_ai_intro")):
            profile["llm_ai_intro"] = self.change_explanation(decision.changes)

        original_profile = deepcopy(profile)
        original_cart = deepcopy(cart)
        profile["_skip_llm_dispatch"] = True
        profile["_current_quote_id"] = quote.quote_id if quote else None
        profile["_structured_compound_edit"] = any(
            change.field not in ("without_transfer", "replan_action") for change in decision.changes
        )
        if decision.intent in ALTERNATIVE_INTENTS:
            profile["_comparison_only"] = True
            profile["_structured_comparison"] = True
        result = await plan_executor(
            user_id,
            message,
            profile,
            cart,
            session,
            progress_callback,
        )
        final_cart = result.get("cart") or cart
        final_quote = TripStateStore.quote_from_cart(final_cart) or quote
        if decision.intent in ALTERNATIVE_INTENTS:
            result["quote_difference"] = compare_quotes(quote, final_quote) if final_quote else None
            result["alternative_quote"] = final_quote.model_dump(mode="json") if final_quote else None
            result["alternative_cart"] = deepcopy(final_cart)
            profile = original_profile
            final_cart = original_cart
            final_quote = quote
            result["cart"] = original_cart
        for key in (
            "_current_quote_id", "_resolved_hotel_id", "_resolved_hotel_query",
            "_structured_compound_edit", "_structured_comparison", "_structured_decision",
            "_structured_changed_fields", "_structured_budget_changed", "_structured_dates_changed",
            "_structured_destination_changed", "_comparison_only", "destination_flexible",
        ):
            profile.pop(key, None)
        if decision.intent in ALTERNATIVE_INTENTS:
            await save_session(user_id, profile, final_cart, message, result.get("text", ""))
        difference = result.get("quote_difference") or final_cart.get("quote_difference")
        if difference:
            difference["narrative"] = await LLMService.narrate_quote_difference(difference)
            result["quote_difference"] = difference
        session.profile = deepcopy(profile)
        session.current_cart = deepcopy(final_cart)
        final_state = self._state_after_execution(state, profile, final_cart)
        return self._enrich(result, decision, profile, final_cart, final_quote, state=final_state)

    def _read_only_result(self, decision: DialogueDecision, profile: dict, cart: dict, message: str) -> dict:
        if decision.intent == DialogueIntent.CLARIFY:
            return {
                "text": decision.clarification or decision.answer or "Уточните, пожалуйста, запрос.",
                "state": "CLARIFICATION_REQUIRED",
                "cart": cart,
            }
        if decision.topic == "connection_risk":
            reasoning = deepcopy(cart.get("reasoning_steps") or [])
            step = next((item for item in reasoning if item.get("stage") == "logistics"), None)
            if step is None:
                step = {"stage": "logistics", "title": "Данные сохранённого рейса"}
                reasoning.append(step)
            step["detail"] = connection_summary(cart.get("logistics") or {})
            return {"text": describe_saved_logistics(cart), "state": "CURRENT_LOGISTICS", "cart": cart, "reasoning_steps": reasoning}
        if decision.topic == "internal_schedule":
            from app.services.agent_service import AgentService
            reply = AgentService.answer_internal_route_timing(message.lower(), cart)
            if reply:
                return {"text": reply, "state": "INTERNAL_SCHEDULE_INFO", "cart": cart}
        if decision.topic == "budget_estimate" and cart.get("budget_quote"):
            mode = decision.mode or "max"
            reply = describe_budget_estimate(cart["budget_quote"], mode, profile.get("budget_rub"))
            return {"text": reply, "state": "BUDGET_ESTIMATE", "cart": cart}
        if decision.answer:
            return {"text": decision.answer, "state": "CONSULTATION", "cart": cart}
        return {"text": describe_current_trip(profile, cart), "state": "CURRENT_TRIP", "cart": cart}

    def pending_for_decision(self, decision: DialogueDecision, state: TripState) -> PendingAction:
        missing = list(decision.search.missing_fields if decision.search else [])
        if not missing and state.pending_action:
            missing = list(state.pending_action.missing_fields)
        kind = "search_alternatives" if decision.search and decision.search.scope == "other_destinations" else "plan_trip"
        if decision.references and not missing:
            kind = "select_offer"
            missing = ["hotel_id"]
        return PendingAction(kind=kind, missing_fields=missing, question=self.missing_question(missing) if missing else decision.clarification)

    def outstanding_fields(self, decision: DialogueDecision, state: TripState) -> list[str]:
        requested = list(decision.search.missing_fields if decision.search else [])
        inherited = list(state.pending_action.missing_fields) if state.pending_action and decision.intent != DialogueIntent.NEW_TRIP else []
        unresolved = []
        for field in dict.fromkeys(requested + inherited):
            if field in ("direction", "destination_id") and decision.search and decision.search.destination_query:
                continue
            if (field in requested and decision.clarification) or self.is_missing(state, field):
                unresolved.append(field)
        return unresolved

    def is_missing(self, state: TripState, field: str) -> bool:
        if field in ("direction", "destination_id"):
            return not (state.destination_id or state.preferences.interests)
        if field == "checkin_date":
            return not state.dates.checkin_date
        if field == "nights":
            return not state.dates.nights
        if field == "adults_count":
            return not state.passengers.adults_count
        if field == "kids_ages":
            return (state.passengers.kids_count or 0) > len(state.kids_ages)
        if field == "rooms_count":
            travelers = (state.passengers.adults_count or 2) + (state.passengers.kids_count or 0)
            return travelers > 4 and not state.rooms_count
        if field == "budget_rub":
            return state.budget_rub is None
        if field == "hotel_id":
            return not state.selected_offers.get("hotel_id")
        return False

    def missing_question(self, fields: list[str]) -> str:
        labels = {
            "direction": "какой отдых или направление вам ближе",
            "destination_id": "какое направление выбрать",
            "checkin_date": "с какой даты начинается поездка (например, «1 ноября» или «ближайшие выходные»)",
            "nights": "сколько ночей вы едете",
            "adults_count": "сколько взрослых едет, включая вас",
            "kids_ages": "какой возраст у детей",
            "rooms_count": "сколько отдельных номеров нужно",
            "budget_rub": "какой общий лимит поездки в рублях",
            "hotel_id": "какой именно вариант выбрать",
        }
        unknown = [field for field in fields if field not in labels]
        if unknown:
            return "Уточните, пожалуйста, недостающие условия поездки."
        if not fields:
            return "Уточните, пожалуйста, запрос."
        return "Уточните, пожалуйста: " + ", ".join(labels[field] for field in fields) + "?"

    def change_explanation(self, changes: list[StateChange]) -> str:
        labels = {
            "nights": "длительность",
            "adults_count": "число взрослых",
            "kids_count": "число детей",
            "kids_ages": "возраст детей",
            "rooms_count": "количество номеров",
            "budget_rub": "общий лимит",
            "has_pets": "поездка с питомцем",
            "without_transfer": "трансфер",
            "destination_id": "направление",
            "interests": "критерии направления",
            "hotel_id": "выбор отеля",
            "flight_time_pref": "время вылета",
            "prefer_direct_flight": "тип рейса",
            "origin_airport": "аэропорт вылета",
            "checkin_date": "дата заезда",
            "checkout_date": "дата выезда",
        }
        changed = [labels.get(change.field, change.field) for change in changes if change.field != "replan_action"]
        return "Обновляю условия: " + ", ".join(changed) + ". Проверяю совместимость и пересчитываю стоимость."

    def apply_changes(self, state: TripState, changes: list[StateChange]) -> TripState:
        updated = state.model_copy(deep=True)
        for change in changes:
            self._validate_change(change)
            if change.field == "replan_action":
                continue
            value = change.value
            if change.operation == ChangeOperation.CLEAR:
                value = [] if change.field in ("interests", "rejected_destinations", "rejected_hotels", "kids_ages") else None
            if change.field in ("interests", "rejected_destinations", "rejected_hotels"):
                target = getattr(updated.preferences, change.field)
                values = value if isinstance(value, list) else [value]
                if change.operation == ChangeOperation.ADD:
                    target.extend(item for item in values if item not in target)
                elif change.operation == ChangeOperation.REMOVE:
                    target[:] = [item for item in target if item not in values]
                else:
                    setattr(updated.preferences, change.field, list(values))
            elif change.field == "hotel_id":
                if change.operation == ChangeOperation.CLEAR:
                    updated.selected_offers.pop("hotel_id", None)
                else:
                    updated.selected_offers["hotel_id"] = value
            elif change.field == "destination_id":
                updated.destination_id = value
            elif change.field in ("checkin_date", "checkout_date", "nights"):
                setattr(updated.dates, change.field, value)
            elif change.field in ("adults_count", "kids_count", "has_pets"):
                setattr(updated.passengers, change.field, value)
                if change.field == "kids_count":
                    updated.passengers.has_kids = value > 0
            elif change.field == "kids_ages":
                updated.kids_ages = list(value or [])
            elif change.field in ("without_transfer", "prefer_direct_flight", "flight_time_pref", "origin_airport"):
                setattr(updated.services, change.field, value)
            elif change.field == "private_room_required":
                updated.preferences.private_room_required = bool(value) if value is not None else False
            elif change.field == "is_multi_city":
                updated.preferences.is_multi_city = bool(value)
            else:
                setattr(updated, change.field, value)
        self._validate_state(updated)
        return updated

    def _validate_change(self, change: StateChange) -> None:
        allowed = {
            "destination_id", "checkin_date", "checkout_date", "nights", "adults_count", "kids_count",
            "has_pets", "rooms_count", "budget_rub", "without_transfer", "prefer_direct_flight",
            "flight_time_pref", "origin_airport", "private_room_required", "interests", "hotel_id",
            "rejected_destinations", "rejected_hotels", "replan_action", "kids_ages", "is_multi_city",
        }
        if change.field not in allowed:
            raise TripValidationError(f"Неизвестное поле поездки: {change.field}")
        if change.operation in (ChangeOperation.ADD, ChangeOperation.REMOVE) and change.field not in (
            "interests", "rejected_destinations", "rejected_hotels",
        ):
            raise TripValidationError(f"Операция {change.operation} недопустима для {change.field}")
        if change.operation == ChangeOperation.CLEAR:
            return
        value = change.value
        if change.field in ("nights", "adults_count", "kids_count", "rooms_count", "budget_rub") and (
            type(value) is not int
            or (change.field == "kids_count" and value < 0)
            or (change.field != "kids_count" and value <= 0)
        ):
            raise TripValidationError(f"Некорректное числовое значение: {change.field}")
        if change.field in ("has_pets", "without_transfer", "prefer_direct_flight", "private_room_required", "is_multi_city") and type(value) is not bool:
            raise TripValidationError(f"Ожидается true/false: {change.field}")
        if change.field == "kids_ages" and (
            not isinstance(value, list) or any(type(age) is not int or age < 0 or age > 17 for age in value)
        ):
            raise TripValidationError("Возраст детей должен быть списком целых чисел от 0 до 17")
        if change.field in ("checkin_date", "checkout_date"):
            try:
                date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise TripValidationError(f"Некорректная дата: {change.field}") from exc

    def _validate_state(self, state: TripState) -> None:
        if state.dates.checkin_date and state.dates.checkout_date:
            checkin = date.fromisoformat(state.dates.checkin_date)
            checkout = date.fromisoformat(state.dates.checkout_date)
            if checkout <= checkin:
                raise TripValidationError("Дата выезда должна быть позже даты заезда")
        if (state.passengers.adults_count or 2) + (state.passengers.kids_count or 0) > 50:
            raise TripValidationError("Слишком большая группа")

    def _state_after_execution(self, state: TripState, profile: dict, cart: dict) -> TripState:
        updated = state.model_copy(deep=True)
        updated.destination_id = cart.get("destination_id") or updated.destination_id
        updated.destination_name = cart.get("destination") or updated.destination_name
        updated.current_version_id = cart.get("tour_version_id") or updated.current_version_id
        for field in ("checkin_date", "checkout_date", "nights"):
            value = profile.get(field, cart.get(field))
            if value is not None:
                setattr(updated.dates, field, value)
        offer_keys = (
            "hotel_id", "hotel", "hotel_city", "flight", "logistics", "compared_hotels",
            "is_multi_city", "multi_city_segments",
        )
        updated.selected_offers = {key: deepcopy(cart[key]) for key in offer_keys if key in cart}
        return updated

    def _enrich(
        self,
        result: dict,
        decision: DialogueDecision | None,
        profile: dict,
        cart: dict,
        quote: TripQuote | None,
        state: TripState | None = None,
    ) -> dict:
        state = state or TripState.from_legacy(profile, cart)
        enriched = deepcopy(result)
        enriched["cart"] = cart
        enriched["trip_state"] = state.model_dump(mode="json")
        enriched["trip_quote"] = quote.model_dump(mode="json") if quote else None
        enriched["dialogue_decision"] = decision.model_dump(mode="json") if decision else None
        enriched["search_request"] = decision.search.model_dump(mode="json") if decision and decision.search else None
        enriched["_effective_profile"] = deepcopy(profile)
        enriched.setdefault("diagnostics", {})["llm"] = LLMService.get_diagnostics()
        return enriched
