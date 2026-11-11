import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from app.database import get_neo4j_driver
from app.domain.route import RouteGraph
from app.domain.trip import TourVersion
from app.services.graph_service import GraphService


class GraphRepository(ABC):
    @abstractmethod
    async def find_routes(self, airport_code: str, hotel_id: str, **preferences) -> list[RouteGraph]:
        raise NotImplementedError

    @abstractmethod
    async def check_compatibility(self, airport_code: str, hotel_id: str, **preferences) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def find_alternatives(self, airport_code: str, hotel_id: str, **preferences) -> list[RouteGraph]:
        raise NotImplementedError

    @abstractmethod
    async def replace_hotel(self, cart: dict, new_hotel: dict, profile: dict, **preferences) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def save_tour_version(self, version: TourVersion) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_tour_version(self, version_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class Neo4jGraphRepository(GraphRepository):
    async def find_routes(self, airport_code: str, hotel_id: str, **preferences) -> list[RouteGraph]:
        result = await self.check_compatibility(airport_code, hotel_id, **preferences)
        if not result.get("compatible") or not result.get("route"):
            return []
        return [RouteGraph.model_validate(result["route"])]

    async def check_compatibility(self, airport_code: str, hotel_id: str, **preferences) -> dict[str, Any]:
        travelers = max(1, preferences.get("travelers_count") or 1)
        result = await GraphService.validate_logistics(
            airport_code=airport_code,
            hotel_id=hotel_id,
            origin_airport_code=preferences.get("origin_airport_code"),
            prefer_direct_flight=preferences.get("prefer_direct_flight"),
            without_transfer=preferences.get("without_transfer", False),
            flight_time_pref=preferences.get("flight_time_pref"),
            travelers_count=travelers,
            strict_graph_validation=preferences.get("strict_graph_validation", True),
            return_airport_code=preferences.get("return_airport_code"),
        )
        transfer = result.get("transfer") or {}
        capacity = transfer.get("capacity") or transfer.get("max_passengers")
        if result.get("compatible") and capacity is not None and travelers > capacity:
            return {
                "compatible": False,
                "reason": f"Трансфер вмещает {capacity} пассажиров, а едет {travelers}.",
                "travelers_count": travelers,
            }
        if transfer and transfer.get("type") != "self":
            transfer["capacity_status"] = "known" if capacity is not None else "unknown"
            if transfer.get("price_per_person_rub") is not None:
                transfer["price_rub"] = transfer["price_per_person_rub"] * travelers
        result["travelers_count"] = travelers
        result["room_capacity_status"] = "unknown"
        if result.get("compatible"):
            cart = {
                "destination_id": preferences.get("destination_id") or "unknown",
                "destination": preferences.get("destination_name") or preferences.get("destination_id") or "unknown",
                "hotel_id": hotel_id,
                "hotel": result.get("hotel"),
                "hotel_city": preferences.get("hotel_city"),
                "nights": preferences.get("nights") or 7,
                "checkin_date": preferences.get("checkin_date"),
                "checkout_date": preferences.get("checkout_date"),
                "logistics": result,
            }
            profile = {
                "without_transfer": preferences.get("without_transfer", False),
                "nights": preferences.get("nights") or 7,
            }
            result["route"] = RouteGraph.from_legacy(cart, profile).model_dump(mode="json")
        return result

    async def find_alternatives(self, airport_code: str, hotel_id: str, **preferences) -> list[RouteGraph]:
        alternatives = []
        for direct in (True, False):
            if preferences.get("prefer_direct_flight") is not None and direct != preferences["prefer_direct_flight"]:
                continue
            route = await self.find_routes(airport_code, hotel_id, **{**preferences, "prefer_direct_flight": direct})
            alternatives.extend(route)
        return alternatives

    async def replace_hotel(self, cart: dict, new_hotel: dict, profile: dict, **preferences) -> dict[str, Any]:
        result = await self.check_compatibility(
            preferences.get("airport_code") or "",
            new_hotel["id"],
            origin_airport_code=profile.get("origin_airport"),
            prefer_direct_flight=profile.get("prefer_direct_flight"),
            without_transfer=profile.get("without_transfer", False),
            flight_time_pref=profile.get("flight_time_pref"),
            travelers_count=max(1, profile.get("adults_count") or 2) + (profile.get("kids_count") or 0),
            destination_id=cart.get("destination_id"),
            destination_name=cart.get("destination"),
            hotel_city=new_hotel.get("city"),
            nights=profile.get("nights"),
            checkin_date=profile.get("checkin_date"),
            checkout_date=profile.get("checkout_date"),
        )
        if not result.get("compatible"):
            return result
        updated_cart = deepcopy(cart)
        updated_cart.update({
            "hotel_id": new_hotel["id"],
            "hotel": new_hotel.get("name") or new_hotel["id"],
            "hotel_city": new_hotel.get("city"),
            "logistics": result,
        })
        route = RouteGraph.from_legacy(updated_cart, profile)
        previous_route = RouteGraph.from_legacy(cart, profile)
        preserved = {flight.id for flight in previous_route.flights}
        result["preserved_segment_ids"] = [flight.id for flight in route.flights if flight.id in preserved]
        result["route"] = route.model_dump(mode="json")
        return result

    def preserve_compatible_hotel_change(self, cart: dict, new_hotel: dict, profile: dict, result: dict) -> dict:
        if not result.get("compatible"):
            return result
        updated_cart = deepcopy(cart)
        updated_cart.update({
            "hotel_id": new_hotel["id"],
            "hotel": new_hotel.get("name") or new_hotel["id"],
            "hotel_city": new_hotel.get("city"),
            "logistics": result,
        })
        route = RouteGraph.from_legacy(updated_cart, profile)
        previous_route = RouteGraph.from_legacy(cart, profile)
        preserved = {flight.id for flight in previous_route.flights}
        result["preserved_segment_ids"] = [flight.id for flight in route.flights if flight.id in preserved]
        result["route"] = route.model_dump(mode="json")
        return result

    async def save_tour_version(self, version: TourVersion) -> None:
        driver = get_neo4j_driver()
        payload = version.model_dump(mode="json")
        route = payload["route"]
        async with driver.session() as session:
            transaction = await session.begin_transaction()
            try:
                await transaction.run("""
                    MERGE (v:TourVersion {id: $version_id})
                    SET v.change_reason = $change_reason,
                        v.created_at = $created_at,
                        v.previous_version_id = $previous_version_id,
                        v.payload_json = $payload_json
                    MERGE (r:RouteSnapshot {id: $version_id})
                    MERGE (v)-[:HAS_ROUTE]->(r)
                    MERGE (c:ConstraintSet {id: $version_id})
                    SET c.payload_json = $constraints_json
                    MERGE (v)-[:HAS_CONSTRAINTS]->(c)
                    MERGE (q:QuoteVersion {id: $quote_id})
                    SET q.previous_quote_id = $previous_quote_id,
                        q.calculation_version = $calculation_version,
                        q.checkin_date = $checkin_date,
                        q.checkout_date = $checkout_date,
                        q.source_prices_json = $source_prices_json,
                        q.payload_json = $quote_json
                    MERGE (v)-[:HAS_QUOTE]->(q)
                    WITH v
                    OPTIONAL MATCH (previous:TourVersion {id: $previous_version_id})
                    FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
                        MERGE (v)-[:PREVIOUS_VERSION]->(previous)
                    )
                """, version_id=version.version_id, change_reason=version.change_reason,
                    created_at=version.created_at, previous_version_id=version.previous_version_id,
                    payload_json=json.dumps(payload, ensure_ascii=False), constraints_json=json.dumps(payload["constraints"], ensure_ascii=False),
                    quote_id=version.quote.quote_id, previous_quote_id=version.quote.previous_quote_id,
                    calculation_version=version.quote.calculation_version, checkin_date=version.quote.checkin_date,
                    checkout_date=version.quote.checkout_date, source_prices_json=json.dumps(version.quote.source_prices, ensure_ascii=False),
                    quote_json=json.dumps(payload["quote"], ensure_ascii=False))
                await transaction.run("""
                    UNWIND $flights AS item
                    MERGE (f:FlightSegment {id: item.id})
                    SET f.direction = item.direction,
                        f.flight_no = item.flight_no,
                        f.airline = item.airline,
                        f.departure_time = item.departure_time,
                        f.arrival_time = item.arrival_time,
                        f.price_rub = item.price_rub,
                        f.baggage_allowance = item.baggage_allowance,
                        f.data_status = item.data_status,
                        f.payload_json = item.payload_json
                    WITH f, item
                    MATCH (r:RouteSnapshot {id: $version_id})
                    MERGE (r)-[:CONTAINS]->(f)
                    MERGE (origin:Airport {code: item.origin.code})
                    MERGE (destination:Airport {code: item.destination.code})
                    MERGE (f)-[:DEPARTS_FROM]->(origin)
                    MERGE (f)-[:ARRIVES_AT]->(destination)
                """, version_id=version.version_id,
                    flights=[{**item, "payload_json": json.dumps(item, ensure_ascii=False)} for item in route["flights"]])
                await transaction.run("""
                    UNWIND $connections AS item
                    MERGE (c:Connection {id: item.id})
                    SET c.duration_minutes = item.duration_minutes,
                        c.through_ticket = item.through_ticket,
                        c.through_baggage = item.through_baggage,
                        c.baggage_status = item.baggage_status,
                        c.risk_note = item.risk_note,
                        c.status = item.status,
                        c.payload_json = item.payload_json
                    WITH c, item
                    MATCH (r:RouteSnapshot {id: $version_id})
                    MATCH (incoming:FlightSegment {id: item.incoming_flight_id})
                    MATCH (outgoing:FlightSegment {id: item.outgoing_flight_id})
                    MERGE (r)-[:CONTAINS]->(c)
                    MERGE (c)-[:AFTER]->(incoming)
                    MERGE (c)-[:BEFORE]->(outgoing)
                    MERGE (airport:Airport {code: item.airport.code})
                    MERGE (c)-[:AT_AIRPORT]->(airport)
                """, version_id=version.version_id,
                    connections=[{**item, "payload_json": json.dumps(item, ensure_ascii=False)} for item in route["connections"]])
                await transaction.run("""
                    UNWIND $hotels AS item
                    MERGE (h:Hotel {id: item.id})
                    WITH h, item
                    MATCH (r:RouteSnapshot {id: $version_id})
                    MERGE (r)-[:CONTAINS]->(h)
                """, version_id=version.version_id, hotels=route["hotels"])
                await transaction.run("""
                    UNWIND $stays AS item
                    MERGE (s:Stay {id: item.id})
                    SET s += item
                    WITH s, item
                    MATCH (r:RouteSnapshot {id: $version_id})
                    MATCH (h:Hotel {id: item.hotel_id})
                    MERGE (r)-[:CONTAINS]->(s)
                    MERGE (s)-[:AT_HOTEL]->(h)
                """, version_id=version.version_id, stays=route["stays"])
                await transaction.run("""
                    UNWIND $transfers AS item
                    MERGE (t:Transfer {id: item.id})
                    SET t.name = item.name,
                        t.type = item.type,
                        t.to_hotel_id = item.to_hotel_id,
                        t.duration_minutes = item.duration_minutes,
                        t.price_rub = item.price_rub,
                        t.payload_json = item.payload_json
                    WITH t, item
                    MATCH (r:RouteSnapshot {id: $version_id})
                    MERGE (r)-[:CONTAINS]->(t)
                """, version_id=version.version_id,
                    transfers=[{**item, "payload_json": json.dumps(item, ensure_ascii=False)} for item in route["transfers"]])
                await transaction.run("""
                    UNWIND $offers AS item
                    MERGE (o:Offer {id: item.id})
                    SET o += item
                    WITH o
                    MATCH (r:RouteSnapshot {id: $version_id})
                    MERGE (r)-[:CONTAINS]->(o)
                """, version_id=version.version_id, offers=route["offers"])
                await transaction.run("""
                    UNWIND $constraints AS item
                    MERGE (c:Constraint {id: $version_id + ':constraint:' + item.index})
                    SET c.field = item.value.field, c.value = item.value.value
                    WITH c
                    MATCH (set:ConstraintSet {id: $version_id})
                    MERGE (set)-[:CONTAINS]->(c)
                """, version_id=version.version_id,
                    constraints=[{"index": index, "value": value} for index, value in enumerate(route["constraints"])])
                await transaction.run("""
                    UNWIND $sources AS item
                    MERGE (s:Source {id: $version_id + ':source:' + item.index})
                    SET s.uri = item.value
                    WITH s
                    MATCH (v:TourVersion {id: $version_id})
                    MERGE (v)-[:HAS_SOURCE]->(s)
                """, version_id=version.version_id,
                    sources=[{"index": index, "value": value} for index, value in enumerate(version.sources)])
                await transaction.run("""
                    UNWIND $warnings AS item
                    MERGE (w:Warning {id: $version_id + ':warning:' + item.index})
                    SET w.text = item.value
                    WITH w
                    MATCH (v:TourVersion {id: $version_id})
                    MERGE (v)-[:HAS_WARNING]->(w)
                """, version_id=version.version_id,
                    warnings=[{"index": index, "value": value} for index, value in enumerate(version.warnings)])
                await transaction.commit()
            except Exception:
                await transaction.rollback()
                raise

    async def get_tour_version(self, version_id: str) -> dict[str, Any] | None:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            result = await session.run("""
                MATCH (v:TourVersion {id: $version_id})
                OPTIONAL MATCH (v)-[:HAS_QUOTE]->(q:QuoteVersion)
                OPTIONAL MATCH (v)-[:HAS_ROUTE]->(r:RouteSnapshot)
                RETURN v.payload_json AS payload, q.id AS quote_id, r.id AS route_id
            """, version_id=version_id)
            record = await result.single()
            if not record:
                return None
            payload = json.loads(record["payload"])
            payload["graph_saved"] = True
            payload["quote_id"] = record["quote_id"]
            payload["route_id"] = record["route_id"]
            return payload


graph_repository = Neo4jGraphRepository()
