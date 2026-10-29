from copy import deepcopy
from typing import Any

from app.domain.route import RouteGraph
from app.domain.trip import TourVersion, TripQuote


CALCULATION_VERSION = "calendar-v1"


def version_breakdown(breakdown: dict, previous_quote_id: str | None = None) -> dict:
    quote = TripQuote.from_breakdown(breakdown, previous_quote_id)
    quote.calculation_version = CALCULATION_VERSION
    return quote.enrich_breakdown(breakdown)


def quote_from_breakdown(breakdown: dict) -> TripQuote:
    quote = TripQuote.from_breakdown(breakdown, breakdown.get("previous_quote_id"))
    quote.quote_id = breakdown.get("quote_id") or quote.quote_id
    quote.previous_quote_id = breakdown.get("previous_quote_id")
    quote.calculation_version = breakdown.get("calculation_version") or CALCULATION_VERSION
    quote.created_at = breakdown.get("quote_created_at") or quote.created_at
    return quote


def compare_quotes(previous: TripQuote | None, current: TripQuote) -> dict[str, Any] | None:
    if previous is None:
        return None
    old = previous.mandatory
    new = current.mandatory
    components = []
    for key, label in (
        ("hotel_total", "Проживание"),
        ("flight_roundtrip", "Перелёты"),
        ("transfer", "Трансфер"),
        ("pet_total", "Доплаты"),
    ):
        before = int(old.get(key) or 0)
        after = int(new.get(key) or 0)
        components.append({"component": key, "label": label, "previous_rub": before, "current_rub": after, "delta_rub": after - before})
    old_onsite = previous.onsite_flexible.get("total_onsite_range_rub") or {}
    new_onsite = current.onsite_flexible.get("total_onsite_range_rub") or {}
    components.append({
        "component": "onsite",
        "label": "Расходы на месте",
        "previous_rub": deepcopy(old_onsite),
        "current_rub": deepcopy(new_onsite),
        "delta_rub": {key: int(new_onsite.get(key) or 0) - int(old_onsite.get(key) or 0) for key in ("min", "max")},
    })
    totals = {
        "mandatory_delta_rub": int(new.get("total_mandatory_rub") or 0) - int(old.get("total_mandatory_rub") or 0),
        "total_delta_rub": {
            key: current.total_trip_budget_range_rub[key] - previous.total_trip_budget_range_rub[key]
            for key in ("min", "max")
        },
    }
    result = {
        "previous_quote_id": previous.quote_id,
        "current_quote_id": current.quote_id,
        "components": components,
        "totals": totals,
    }
    result["text"] = format_quote_difference(result)
    return result


def format_quote_difference(difference: dict) -> str:
    parts = []
    for component in difference["components"]:
        delta = component["delta_rub"]
        if component["component"] == "onsite" or not isinstance(delta, int) or delta == 0:
            continue
        direction = "подорожал" if delta > 0 else "подешевел"
        if component["component"] == "hotel_total":
            direction = "подорожало" if delta > 0 else "подешевело"
        elif component["component"] == "flight_roundtrip":
            direction = "подорожали" if delta > 0 else "подешевели"
        parts.append(f"{component['label']} {direction} на {abs(delta):,} ₽".replace(",", " "))
    total = difference["totals"]["mandatory_delta_rub"]
    if not parts and total == 0:
        return "Обязательные расходы не изменились."
    parts.append(f"итоговая разница — {total:,} ₽".replace(",", " ").replace("-", "−"))
    return ", ".join(parts) + "."


def build_tour_version(
    cart: dict,
    profile: dict,
    quote: TripQuote,
    previous_version_id: str | None,
    reason: str,
) -> TourVersion:
    route = RouteGraph.from_legacy(cart, profile)
    warnings = []
    for connection in route.connections:
        if connection.status == "unknown":
            warnings.append(f"Пересадка {connection.airport.code}: недостаточно данных для оценки билета и багажа.")
        elif connection.status == "warning":
            warnings.append(f"Пересадка {connection.airport.code}: {connection.risk_note or 'короткая стыковка'}.")
    sources = [quote.pricing_source, "Neo4j logistics graph"]
    return TourVersion(
        previous_version_id=previous_version_id,
        change_reason=reason,
        route=route,
        constraints={
            "budget_rub": profile.get("budget_rub"),
            "rooms_count": profile.get("rooms_count") or 1,
            "adults_count": profile.get("adults_count") or 2,
            "kids_count": profile.get("kids_count") or 0,
            "has_pets": bool(profile.get("has_pets")),
            "without_transfer": bool(profile.get("without_transfer")),
            "prefer_direct_flight": profile.get("prefer_direct_flight"),
        },
        quote=quote,
        sources=sources,
        warnings=warnings,
    )
