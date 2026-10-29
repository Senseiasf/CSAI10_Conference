from copy import deepcopy
from datetime import date
from typing import Any, Awaitable, Callable

from app.config import settings
from app.domain.dialogue import ChangeOperation, DialogueDecision, DialogueIntent, StateChange
from app.domain.trip import TripQuote, TripState
from app.services.booking_stub import booking_stub, detect_booking_action
from app.services.dialogue_parser import DialogueParser, DialogueParserError, decision_context
from app.services.llm_service import LLMService
from app.services.quote_service import compare_quotes
from app.services.trip_logistics import connection_summary, describe_saved_logistics
from app.services.trip_state import describe_budget_estimate, describe_current_trip
from app.services.trip_state_store import TripStateStore


class TripValidationError(ValueError):
    pass


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

        booking_action = detect_booking_action(message)
        if booking_action:
            result = booking_stub(booking_action, cart)
            await save_session(user_id, profile, cart, message, result["text"])
            return self._enrich(result, None, profile, cart, quote)

        try:
            decision = await self.parser.parse(
                message,
                state,
                decision_context(session, state, cart),
                quote,
            )
            LLMService._last_error = None
            if decision.narrative:
                profile["llm_ai_intro"] = decision.narrative
            if decision.intent in (DialogueIntent.MODIFY, DialogueIntent.PLAN) and decision.changes:
                state = self.apply_changes(state, decision.changes)
                profile = state.apply_to_profile(profile)
            hotel_change = next((change for change in decision.changes if change.field == "hotel_id"), None)
            if hotel_change:
                profile["_resolved_hotel_id"] = hotel_change.value
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
            result = self._read_only_result(decision, profile, cart)
            await save_session(user_id, profile, cart, message, result["text"])
            return self._enrich(result, decision, profile, cart, quote)

        original_profile = deepcopy(profile)
        original_cart = deepcopy(cart)
        profile["_skip_llm_dispatch"] = True
        profile["_current_quote_id"] = quote.quote_id if quote else None
        if decision.intent == DialogueIntent.COMPARISON:
            profile["_comparison_only"] = True
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
        if decision.intent == DialogueIntent.COMPARISON:
            result["quote_difference"] = compare_quotes(quote, final_quote) if final_quote else None
            result["alternative_quote"] = final_quote.model_dump(mode="json") if final_quote else None
            profile = original_profile
            final_cart = original_cart
            final_quote = quote
            result["cart"] = original_cart
            await save_session(user_id, profile, final_cart, message, result.get("text", ""))
        difference = result.get("quote_difference") or final_cart.get("quote_difference")
        if difference:
            difference["narrative"] = await LLMService.narrate_quote_difference(difference)
            result["quote_difference"] = difference
        session.profile = deepcopy(profile)
        session.current_cart = deepcopy(final_cart)
        return self._enrich(result, decision, profile, final_cart, final_quote)

    def _read_only_result(self, decision: DialogueDecision, profile: dict, cart: dict) -> dict:
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
        if decision.topic == "budget_estimate" and cart.get("budget_quote"):
            mode = decision.mode or "max"
            reply = describe_budget_estimate(cart["budget_quote"], mode, profile.get("budget_rub"))
            return {"text": reply, "state": "BUDGET_ESTIMATE", "cart": cart}
        if decision.answer:
            return {"text": decision.answer, "state": "CONSULTATION", "cart": cart}
        return {"text": describe_current_trip(profile, cart), "state": "CURRENT_TRIP", "cart": cart}

    def apply_changes(self, state: TripState, changes: list[StateChange]) -> TripState:
        updated = state.model_copy(deep=True)
        for change in changes:
            self._validate_change(change)
            if change.field == "replan_action":
                continue
            value = change.value
            if change.operation == ChangeOperation.CLEAR:
                value = [] if change.field in ("interests", "rejected_destinations", "rejected_hotels") else None
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
            elif change.field in ("without_transfer", "prefer_direct_flight", "flight_time_pref", "origin_airport"):
                setattr(updated.services, change.field, value)
            elif change.field == "private_room_required":
                updated.preferences.private_room_required = bool(value) if value is not None else False
            else:
                setattr(updated, change.field, value)
        self._validate_state(updated)
        return updated

    def _validate_change(self, change: StateChange) -> None:
        allowed = {
            "destination_id", "checkin_date", "checkout_date", "nights", "adults_count", "kids_count",
            "has_pets", "rooms_count", "budget_rub", "without_transfer", "prefer_direct_flight",
            "flight_time_pref", "origin_airport", "private_room_required", "interests", "hotel_id",
            "rejected_destinations", "rejected_hotels", "replan_action",
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
        if change.field in ("has_pets", "without_transfer", "prefer_direct_flight", "private_room_required") and type(value) is not bool:
            raise TripValidationError(f"Ожидается true/false: {change.field}")
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
        if (state.passengers.adults_count or 2) + state.passengers.kids_count > 50:
            raise TripValidationError("Слишком большая группа")

    def _enrich(
        self,
        result: dict,
        decision: DialogueDecision | None,
        profile: dict,
        cart: dict,
        quote: TripQuote | None,
    ) -> dict:
        state = TripState.from_legacy(profile, cart)
        enriched = deepcopy(result)
        enriched["cart"] = cart
        enriched["trip_state"] = state.model_dump(mode="json")
        enriched["trip_quote"] = quote.model_dump(mode="json") if quote else None
        enriched["dialogue_decision"] = decision.model_dump(mode="json") if decision else None
        enriched["_effective_profile"] = deepcopy(profile)
        enriched.setdefault("diagnostics", {})["llm"] = LLMService.get_diagnostics()
        return enriched
