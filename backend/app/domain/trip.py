from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.route import RouteGraph


class TripDates(BaseModel):
    checkin_date: str | None = None
    checkout_date: str | None = None
    nights: int | None = Field(default=None, ge=1, le=60)


class Passengers(BaseModel):
    adults_count: int | None = Field(default=None, ge=1, le=30)
    kids_count: int = Field(default=0, ge=0, le=20)
    has_kids: bool | None = None
    has_pets: bool = False


class TripServices(BaseModel):
    without_transfer: bool = False
    prefer_direct_flight: bool | None = None
    flight_time_pref: str | None = None
    origin_airport: str | None = None


class TripPreferences(BaseModel):
    interests: list[str] = Field(default_factory=list)
    private_room_required: bool = False
    domestic_only: bool = False
    abroad_only: bool = False
    rejected_hotels: list[str] = Field(default_factory=list)
    rejected_destinations: list[str] = Field(default_factory=list)


class TripState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_id: str | None = None
    destination_name: str | None = None
    stops: list[dict[str, Any]] = Field(default_factory=list)
    dates: TripDates = Field(default_factory=TripDates)
    passengers: Passengers = Field(default_factory=Passengers)
    rooms_count: int | None = Field(default=None, ge=1, le=20)
    budget_rub: int | None = Field(default=None, gt=0)
    budget_estimate: Literal["min", "max"] = "max"
    services: TripServices = Field(default_factory=TripServices)
    preferences: TripPreferences = Field(default_factory=TripPreferences)
    selected_offers: dict[str, Any] = Field(default_factory=dict)
    current_version_id: str | None = None

    @classmethod
    def from_legacy(cls, profile: dict | None, cart: dict | None) -> "TripState":
        profile = profile or {}
        cart = cart or {}
        selected_offers = {
            key: deepcopy(cart[key]) for key in (
                "hotel_id", "hotel", "hotel_city", "flight", "logistics", "compared_hotels",
                "is_multi_city", "multi_city_segments",
            ) if key in cart
        }
        return cls(
            destination_id=cart.get("destination_id") or profile.get("preferred_destination"),
            destination_name=cart.get("destination"),
            stops=deepcopy(cart.get("stops") or cart.get("multi_city_segments") or []),
            dates=TripDates(
                checkin_date=profile.get("checkin_date") or cart.get("checkin_date"),
                checkout_date=profile.get("checkout_date") or cart.get("checkout_date"),
                nights=profile.get("nights") or cart.get("nights"),
            ),
            passengers=Passengers(
                adults_count=profile.get("adults_count") or cart.get("adults_count"),
                kids_count=profile.get("kids_count", cart.get("kids_count", 0)) or 0,
                has_kids=profile.get("has_kids") if "has_kids" in profile else (True if profile.get("kids_count", 0) > 0 else None),
                has_pets=bool(profile.get("has_pets", False)),
            ),
            rooms_count=profile.get("rooms_count") or cart.get("rooms_count"),
            budget_rub=profile.get("budget_rub", cart.get("budget_rub")),
            budget_estimate=profile.get("budget_estimate") or cart.get("budget_estimate") or "max",
            services=TripServices(
                without_transfer=bool(profile.get("without_transfer", cart.get("without_transfer", False))),
                prefer_direct_flight=profile.get("prefer_direct_flight"),
                flight_time_pref=profile.get("flight_time_pref"),
                origin_airport=profile.get("origin_airport"),
            ),
            preferences=TripPreferences(
                interests=list(profile.get("interests") or []),
                private_room_required=bool(profile.get("private_room_required", False)),
                domestic_only=bool(profile.get("domestic_only", False)),
                abroad_only=bool(profile.get("abroad_only", False)),
                rejected_hotels=list(profile.get("rejected_hotels") or []),
                rejected_destinations=list(profile.get("rejected_destinations") or []),
            ),
            selected_offers=selected_offers,
            current_version_id=cart.get("tour_version_id") or profile.get("tour_version_id"),
        )

    def apply_to_profile(self, profile: dict | None = None) -> dict:
        result = deepcopy(profile or {})
        result.update({
            "preferred_destination": self.destination_id,
            "checkin_date": self.dates.checkin_date,
            "checkout_date": self.dates.checkout_date,
            "nights": self.dates.nights,
            "adults_count": self.passengers.adults_count,
            "kids_count": self.passengers.kids_count,
            "has_kids": self.passengers.has_kids,
            "has_pets": self.passengers.has_pets,
            "rooms_count": self.rooms_count,
            "budget_rub": self.budget_rub,
            "budget_estimate": self.budget_estimate,
            "without_transfer": self.services.without_transfer,
            "prefer_direct_flight": self.services.prefer_direct_flight,
            "flight_time_pref": self.services.flight_time_pref,
            "origin_airport": self.services.origin_airport,
            "interests": list(self.preferences.interests),
            "private_room_required": self.preferences.private_room_required,
            "domestic_only": self.preferences.domestic_only,
            "abroad_only": self.preferences.abroad_only,
            "rejected_hotels": list(self.preferences.rejected_hotels),
            "rejected_destinations": list(self.preferences.rejected_destinations),
            "tour_version_id": self.current_version_id,
        })
        for key in ("adults_count", "nights", "rooms_count", "budget_rub", "prefer_direct_flight", "flight_time_pref", "origin_airport", "tour_version_id"):
            if result.get(key) is None:
                result.pop(key, None)
        if self.dates.checkin_date and self.dates.checkout_date:
            result["dates"] = f"{self.dates.checkin_date} — {self.dates.checkout_date} ({self.dates.nights} ночей)"
        return result

    def apply_to_cart(self, cart: dict | None = None) -> dict:
        result = deepcopy(cart or {})
        result.update({
            "destination_id": self.destination_id,
            "destination": self.destination_name or result.get("destination"),
            "nights": self.dates.nights,
            "checkin_date": self.dates.checkin_date,
            "checkout_date": self.dates.checkout_date,
            "rooms_count": self.rooms_count,
            "budget_rub": self.budget_rub,
            "budget_estimate": self.budget_estimate,
            "without_transfer": self.services.without_transfer,
            "tour_version_id": self.current_version_id,
        })
        result.update(deepcopy(self.selected_offers))
        return result


class TripQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote_id: str = Field(default_factory=lambda: str(uuid4()))
    previous_quote_id: str | None = None
    calculation_version: str = "calendar-v1"
    checkin_date: str
    checkout_date: str
    nights: int = Field(ge=1)
    source_prices: dict[str, Any] = Field(default_factory=dict)
    mandatory: dict[str, Any]
    onsite_flexible: dict[str, Any]
    total_trip_budget_range_rub: dict[str, int]
    pricing_source: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_breakdown(cls, breakdown: dict, previous_quote_id: str | None = None) -> "TripQuote":
        return cls(
            previous_quote_id=previous_quote_id,
            checkin_date=breakdown["checkin_date"],
            checkout_date=breakdown["checkout_date"],
            nights=breakdown["nights"],
            source_prices=deepcopy(breakdown.get("pricing_inputs") or {}),
            mandatory=deepcopy(breakdown["mandatory"]),
            onsite_flexible=deepcopy(breakdown["onsite_flexible"]),
            total_trip_budget_range_rub=deepcopy(breakdown["total_trip_budget_range_rub"]),
            pricing_source=breakdown.get("pricing_source") or "unknown",
        )

    def enrich_breakdown(self, breakdown: dict) -> dict:
        result = deepcopy(breakdown)
        result.update({
            "quote_id": self.quote_id,
            "previous_quote_id": self.previous_quote_id,
            "calculation_version": self.calculation_version,
            "quote_created_at": self.created_at,
        })
        return result


class TourVersion(BaseModel):
    version_id: str = Field(default_factory=lambda: str(uuid4()))
    previous_version_id: str | None = None
    change_reason: str
    route: RouteGraph
    constraints: dict[str, Any] = Field(default_factory=dict)
    quote: TripQuote
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    persistence_status: Literal["pending", "saved", "failed"] = "pending"
