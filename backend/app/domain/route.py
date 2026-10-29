from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RegionRef(BaseModel):
    id: str
    name: str


class CityRef(BaseModel):
    id: str
    name: str
    region_id: str | None = None


class AirportRef(BaseModel):
    code: str
    name: str | None = None
    city_id: str | None = None


class FlightSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    origin: AirportRef
    destination: AirportRef
    direction: Literal["outbound", "return", "intercity"] = "outbound"
    flight_no: str | None = None
    airline: str | None = None
    departure_time: str | None = None
    arrival_time: str | None = None
    price_rub: int | None = None
    baggage_allowance: str | None = None
    data_status: Literal["known", "partial", "unknown"] = "partial"


class Connection(BaseModel):
    id: str
    airport: AirportRef
    incoming_flight_id: str
    outgoing_flight_id: str
    duration_minutes: int | None = None
    through_ticket: bool | None = None
    through_baggage: bool | None = None
    baggage_status: Literal["known", "unknown"] = "unknown"
    risk_note: str | None = None
    status: Literal["assessed", "warning", "unknown"] = "unknown"


class TransferSegment(BaseModel):
    id: str
    name: str
    type: str
    from_airport: AirportRef | None = None
    to_hotel_id: str | None = None
    duration_minutes: int | None = None
    price_rub: int | None = None


class HotelOffer(BaseModel):
    id: str
    name: str
    city: CityRef | None = None
    price_per_night_rub: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Stay(BaseModel):
    id: str
    hotel_id: str
    checkin_date: str | None = None
    checkout_date: str | None = None
    nights: int


class OfferRef(BaseModel):
    id: str
    kind: Literal["hotel", "flight", "transfer", "package"]
    source_id: str
    selected: bool = False


class RouteConstraint(BaseModel):
    field: str
    value: Any


class RouteStop(BaseModel):
    id: str
    city: CityRef
    airport_codes: list[str] = Field(default_factory=list)
    hotel_id: str | None = None
    nights: int | None = None


class RouteGraph(BaseModel):
    region: RegionRef | None = None
    stops: list[RouteStop] = Field(default_factory=list)
    flights: list[FlightSegment] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    transfers: list[TransferSegment] = Field(default_factory=list)
    hotels: list[HotelOffer] = Field(default_factory=list)
    stays: list[Stay] = Field(default_factory=list)
    offers: list[OfferRef] = Field(default_factory=list)
    constraints: list[RouteConstraint] = Field(default_factory=list)

    @classmethod
    def from_legacy(cls, cart: dict, profile: dict) -> "RouteGraph":
        logistics = cart.get("logistics") or {}
        flight = logistics.get("flight") or {}
        destination_id = cart.get("destination_id") or "unknown"
        destination_name = cart.get("destination") or destination_id
        origin = AirportRef(
            code=flight.get("origin_code") or "UNKNOWN",
            name=flight.get("origin_name"),
        )
        return_flight = flight.get("return_flight") or {}
        destination = AirportRef(
            code=flight.get("dest_code") or return_flight.get("origin_code") or "UNKNOWN",
            name=logistics.get("airport"),
        )
        parent_id = str(flight.get("flight_no") or uuid4())
        flights: list[FlightSegment] = []
        connections: list[Connection] = []
        if flight.get("flight_type") == "connecting":
            connection_airport = AirportRef(
                code=flight.get("layover_airport") or "UNKNOWN",
                name=flight.get("layover_city"),
            )
            first = FlightSegment(
                id=f"{parent_id}:outbound:1",
                origin=origin,
                destination=connection_airport,
                flight_no=flight.get("flight_no"),
                airline=flight.get("airline"),
                departure_time=flight.get("departure_time"),
                arrival_time=None,
                price_rub=flight.get("price_rub_roundtrip"),
                baggage_allowance=flight.get("baggage_allowance"),
                data_status="partial",
            )
            second = FlightSegment(
                id=f"{parent_id}:outbound:2",
                origin=connection_airport,
                destination=destination,
                direction="outbound",
                flight_no=None,
                airline=flight.get("airline"),
                departure_time=None,
                arrival_time=flight.get("arrival_time"),
                price_rub=None,
                baggage_allowance=None,
                data_status="partial",
            )
            flights.extend((first, second))
            duration = flight.get("layover_duration_minutes")
            through_ticket = flight.get("through_ticket")
            through_baggage = flight.get("through_baggage")
            baggage_status = "known" if through_baggage is not None else "unknown"
            status = "unknown"
            if duration is not None and duration < 60:
                status = "warning"
            elif duration is not None and through_ticket is not None and through_baggage is not None:
                status = "assessed"
            connections.append(Connection(
                id=f"{parent_id}:connection:1",
                airport=connection_airport,
                incoming_flight_id=first.id,
                outgoing_flight_id=second.id,
                duration_minutes=duration,
                through_ticket=through_ticket,
                through_baggage=through_baggage,
                baggage_status=baggage_status,
                risk_note=flight.get("layover_risk_note") or None,
                status=status,
            ))
        elif flight:
            flights.append(FlightSegment(
                id=f"{parent_id}:outbound:1",
                origin=origin,
                destination=destination,
                flight_no=flight.get("flight_no"),
                airline=flight.get("airline"),
                departure_time=flight.get("departure_time"),
                arrival_time=flight.get("arrival_time"),
                price_rub=flight.get("price_rub_roundtrip"),
                baggage_allowance=flight.get("baggage_allowance"),
                data_status="known" if flight.get("flight_no") else "partial",
            ))
        if return_flight:
            flights.append(FlightSegment(
                id=f"{return_flight.get('flight_no') or parent_id}:return:1",
                origin=destination,
                destination=origin,
                direction="return",
                flight_no=return_flight.get("flight_no"),
                airline=return_flight.get("airline"),
                departure_time=return_flight.get("departure_time"),
                arrival_time=return_flight.get("arrival_time"),
                price_rub=None,
                baggage_allowance=return_flight.get("baggage_allowance"),
                data_status="known" if return_flight.get("flight_no") else "partial",
            ))
        hotel_id = cart.get("hotel_id")
        hotels = []
        stays = []
        offers = []
        transfers = []
        if hotel_id:
            city = CityRef(id=cart.get("hotel_city") or destination_name, name=cart.get("hotel_city") or destination_name, region_id=destination_id)
            hotels.append(HotelOffer(
                id=hotel_id,
                name=cart.get("hotel") or hotel_id,
                city=city,
                price_per_night_rub=cart.get("price_per_night_rub"),
            ))
            stays.append(Stay(
                id=f"{hotel_id}:stay",
                hotel_id=hotel_id,
                checkin_date=cart.get("checkin_date"),
                checkout_date=cart.get("checkout_date"),
                nights=cart.get("nights") or profile.get("nights") or 7,
            ))
            offers.append(OfferRef(id=f"hotel:{hotel_id}", kind="hotel", source_id=hotel_id, selected=True))
        transfer = logistics.get("transfer") or {}
        if transfer:
            transfers.append(TransferSegment(
                id=f"{hotel_id or destination_id}:transfer",
                name=transfer.get("name") or "Трансфер",
                type=transfer.get("type") or "unknown",
                from_airport=destination,
                to_hotel_id=hotel_id,
                duration_minutes=transfer.get("duration_min"),
                price_rub=transfer.get("price_rub"),
            ))
        city = CityRef(id=destination_id, name=destination_name, region_id=destination_id)
        return cls(
            region=RegionRef(id=destination_id, name=destination_name),
            stops=[RouteStop(id=destination_id, city=city, airport_codes=[destination.code], hotel_id=hotel_id, nights=cart.get("nights"))],
            flights=flights,
            connections=connections,
            transfers=transfers,
            hotels=hotels,
            stays=stays,
            offers=offers,
            constraints=[RouteConstraint(field="without_transfer", value=bool(profile.get("without_transfer")))] + [
                RouteConstraint(field=field, value=profile[field])
                for field in ("budget_rub", "rooms_count", "adults_count", "kids_count", "has_pets", "prefer_direct_flight")
                if profile.get(field) is not None
            ],
        )
