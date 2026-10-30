import json
from copy import deepcopy
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

    @staticmethod
    def with_execution_results(state: TripState, profile: dict, cart: dict) -> TripState:
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
        updated.search_results = deepcopy(profile.get("_search_results") or updated.search_results)
        if updated.pending_action:
            missing = [field for field in updated.pending_action.missing_fields
                       if TripStateStore.is_missing(updated, field)]
            if missing:
                updated.pending_action.missing_fields = missing
            else:
                updated.pending_action = None
        return updated

    @staticmethod
    def is_missing(state: TripState, field: str) -> bool:
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
