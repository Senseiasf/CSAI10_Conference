"""Build connected catalogue packages and evaluate the benchmark predicates."""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from math import isfinite

from app.services.graph_service import MIN_LAYOVER_MINUTES, minutes_since_midnight, post_flight_buffer, transfer_rejection


@dataclass(frozen=True)
class Route:
    destination: str
    hotel: dict
    flight: dict
    transfer: dict
    returns: tuple

    @property
    def key(self):
        return (self.destination, self.hotel["id"], self.flight["from"], self.flight["to"],
                self.flight["flight_no"], self.transfer["type"], self.transfer["name"])


def build_routes(catalog):
    origins = {airport["code"] for airport in catalog["origin_airports"]}
    flights = defaultdict(list)
    for flight in chain.from_iterable(d.get("flights", []) for d in catalog["destinations"]):
        flights[flight["from"], flight["to"]].append(flight)
    routes = defaultdict(list)
    for dest in catalog["destinations"]:
        hotels = {hotel["id"]: hotel for hotel in dest["hotels"]}
        airports = {airport["code"] for airport in dest["airports"]}
        for transfer in dest.get("transfers", []):
            airport = transfer["from"]
            if airport not in airports or transfer["to"] not in hotels:
                continue
            for origin in sorted(origins):
                for flight in flights[origin, airport]:
                    routes[dest["id"]].append(Route(dest["id"], hotels[transfer["to"]], flight,
                                                    transfer, tuple(flights[airport, origin])))
    return routes


async def read_graph_routes(driver, catalog):
    """Read candidate paths and their properties from the seeded Neo4j snapshot."""
    hotels = {h["id"]: h for d in catalog["destinations"] for h in d["hotels"]}
    rows, _, _ = await driver.execute_query(
        """
        MATCH (o:Airport)-[f:FLIGHT_TO]->(a:Airport)-[t:TRANSFER_TO]->(h:Hotel)
        MATCH (a)-[:LOCATED_IN]->(d:City)<-[:LOCATED_IN]-(h)
        WHERE o.code IN $origins AND d.id IN $destinations
        OPTIONAL MATCH (a)-[back:FLIGHT_TO]->(o)
        RETURN d.id AS dest, h.id AS hotel, o.code AS origin, a.code AS airport,
               properties(f) AS flight, properties(t) AS transfer, collect(properties(back)) AS returns
        """,
        origins=[a["code"] for a in catalog["origin_airports"]],
        destinations=[d["id"] for d in catalog["destinations"]],
        routing_="r",
    )
    routes = defaultdict(list)
    for row in rows:
        origin, airport = row["origin"], row["airport"]
        flight = {**row["flight"], "from": origin, "to": airport}
        transfer = {**row["transfer"], "from": airport, "to": row["hotel"]}
        returns = tuple({**f, "from": airport, "to": origin} for f in row["returns"])
        routes[row["dest"]].append(Route(row["dest"], hotels[row["hotel"]], flight, transfer, returns))
    return routes


def flight_errors(flight):
    return _flight_errors(flight.get("flight_type"), flight.get("layover_duration_minutes"),
                          flight.get("departure_time"), flight.get("arrival_time"), flight.get("duration_minutes"))


@lru_cache(maxsize=4096)
def _flight_errors(kind, layover, departure, arrival, duration):
    errors = set()
    if kind != "direct":
        if kind != "connecting" or not isinstance(layover, (int, float)) or not isfinite(layover):
            errors.add("unknown_connection")
        elif layover < MIN_LAYOVER_MINUTES:
            errors.add("layover_under_60min")
    try:
        minutes_since_midnight(departure)
        minutes_since_midnight(arrival)
        if not isfinite(duration) or duration <= 0:
            raise ValueError("nonpositive duration")
    except (KeyError, TypeError, ValueError):
        errors.add("invalid_timeline")
    return frozenset(errors)


def select_return(route, direct_only=False):
    return min(route.returns, default=None, key=lambda f: (
        bool(flight_errors(f) or (direct_only and f.get("flight_type") != "direct")),
        bool(flight_errors(f)), f.get("airline") != route.flight.get("airline"),
        f.get("departure_time") or "", f["flight_no"],
    ))


def package_price(route, request):
    days = request["days"]
    travelers = request["adults_count"] + request["kids_count"]
    rooms = request.get("rooms_count", 1)
    flight = route.flight.get("price_rub_roundtrip")
    hotel = route.hotel.get("price_per_night_rub")
    transfer = route.transfer.get("price_rub")
    pet = (route.hotel.get("pet_surcharge_per_night_rub") or 0) if request.get("has_pets") else 0
    if any(not isinstance(n, (int, float)) or not isfinite(n) or n < 0 for n in (flight, hotel, transfer, pet)):
        return None
    return flight * travelers + (hotel * rooms + pet) * days + transfer


def assess(route, request, back):
    """Return intrinsic logistics violations, user constraint violations, and total price."""
    flight, hotel, transfer = route.flight, route.hotel, route.transfer
    logistics = set(flight_errors(flight))
    constraints = set()
    if back is None:
        logistics.add("no_return_flight")
    else:
        logistics.update(flight_errors(back))
    if request.get("prefer_direct_flight") and any(
        f and f.get("flight_type") != "direct" for f in (flight, back)
    ):
        constraints.add("prefer_direct_violated")
    if request.get("origin_airport") and flight["from"] != request["origin_airport"]:
        constraints.add("origin_airport_violated")
    time_ranges = {"morning": (360, 720), "day": (720, 1080), "evening": (1080, 1440), "later": (600, 1440)}
    try:
        depart = minutes_since_midnight(flight["departure_time"])
        if request.get("flight_time_pref") in time_ranges:
            start, end = time_ranges[request["flight_time_pref"]]
            if not start <= depart < end:
                constraints.add("flight_time_pref_violated")
        arrival = minutes_since_midnight(flight["arrival_time"])
        if not isfinite(transfer["duration_minutes"]) or transfer["duration_minutes"] < 0:
            raise ValueError("negative transfer duration")
        failure = transfer_rejection({
            **transfer, "dest_code": flight["to"], "transfer_type": transfer["type"],
            "transfer_capacity": transfer.get("capacity", transfer.get("max_passengers")),
        }, arrival + post_flight_buffer(flight["to"]), request["adults_count"] + request["kids_count"], True)
        if failure:
            logistics.add(failure)
    except (KeyError, TypeError, ValueError):
        logistics.add("invalid_timeline")
    if request.get("has_pets") and not hotel.get("pet_friendly"):
        constraints.add("pet_friendly_violated")
    if request.get("has_kids") and (not hotel.get("family_friendly") or "adults_only" in hotel.get("features", [])):
        constraints.add("adults_only_with_kids")
    if request.get("private_room_required") and any(
        tag in hotel.get("features", []) for tag in ("hostel_dorm", "shared_bathroom", "shared_room")
    ):
        constraints.add("private_room_violated")
    price = package_price(route, request)
    if price is None:
        constraints.add("unknown_price")
    elif request.get("budget") is not None and price > request["budget"]:
        constraints.add("budget_exceeded")
    return logistics, constraints, price
