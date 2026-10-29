import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from app.domain.trip import TripQuote, TripState


class TripStateStore:
    @staticmethod
    def load(session: Any) -> tuple[TripState, TripQuote | None]:
        raw_state = getattr(session, "trip_state", None)
        if raw_state:
            try:
                state = TripState.model_validate(raw_state)
            except ValidationError:
                state = TripState.from_legacy(session.profile, session.current_cart)
        else:
            state = TripState.from_legacy(session.profile, session.current_cart)
        raw_quote = getattr(session, "trip_quote", None)
        if raw_quote:
            try:
                return state, TripQuote.model_validate(raw_quote)
            except ValidationError:
                pass
        return state, TripStateStore.quote_from_cart(session.current_cart or {})

    @staticmethod
    def quote_from_cart(cart: dict) -> TripQuote | None:
        breakdown = cart.get("budget_breakdown")
        if not breakdown or not all(key in breakdown for key in (
            "checkin_date", "checkout_date", "nights", "mandatory",
            "onsite_flexible", "total_trip_budget_range_rub",
        )):
            return None
        quote = TripQuote.from_breakdown(breakdown, breakdown.get("previous_quote_id"))
        if breakdown.get("quote_id"):
            quote.quote_id = breakdown["quote_id"]
        else:
            fingerprint = {
                "hotel_id": cart.get("hotel_id"),
                "checkin_date": breakdown["checkin_date"],
                "checkout_date": breakdown["checkout_date"],
                "mandatory": breakdown["mandatory"],
                "onsite": breakdown["onsite_flexible"],
                "total": breakdown["total_trip_budget_range_rub"],
            }
            quote.quote_id = str(uuid5(NAMESPACE_URL, "trip-quote:" + json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)))
        if breakdown.get("calculation_version"):
            quote.calculation_version = breakdown["calculation_version"]
        if breakdown.get("quote_created_at"):
            quote.created_at = breakdown["quote_created_at"]
        return quote

    @staticmethod
    def project(session: Any, state: TripState, quote: TripQuote | None) -> None:
        session.trip_state = state.model_dump(mode="json")
        session.trip_quote = quote.model_dump(mode="json") if quote else None

    @staticmethod
    def from_legacy(profile: dict, cart: dict) -> tuple[TripState, TripQuote | None]:
        state = TripState.from_legacy(profile, cart)
        return state, TripStateStore.quote_from_cart(cart)
