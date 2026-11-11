"""Run with CSAI_NEO4J_TEST_URI pointing to a disposable Neo4j database."""

import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from neo4j import AsyncGraphDatabase

from app.mcp.tools import MCPTools

pytestmark = pytest.mark.skipif(
    not os.environ.get("CSAI_NEO4J_TEST_URI"), reason="Requires a local Neo4j test database"
)


@pytest.mark.asyncio
async def test_catalog_routes_are_checked_before_selection():
    fixture = uuid4().hex
    origin = f"origin-{fixture}"
    hotels = {kind: f"{kind}-{fixture}" for kind in ("mle", "led", "self_hotel")}
    async with AsyncGraphDatabase.driver(os.environ["CSAI_NEO4J_TEST_URI"]) as driver:
        await driver.execute_query(
            """
            CREATE (o:Airport {code: $origin, name: 'Test origin', validation_fixture: $fixture})
            MERGE (mle:Airport {code: 'MLE'}) ON CREATE SET mle.name = 'MLE', mle.validation_fixture = $fixture
            MERGE (led:Airport {code: 'LED'}) ON CREATE SET led.name = 'LED', led.validation_fixture = $fixture
            CREATE (city:City {id: $fixture, validation_fixture: $fixture})
            CREATE (hm:Hotel {id: $mle, name: 'Island', checkin_time: '15:00', validation_fixture: $fixture})
            CREATE (hl:Hotel {id: $led, name: 'Island city', checkin_time: '15:00', validation_fixture: $fixture})
            CREATE (hs:Hotel {id: $self_hotel, name: 'Self', checkin_time: '15:00', validation_fixture: $fixture})
            CREATE (led)-[:LOCATED_IN]->(city)<-[:LOCATED_IN]-(hs)
            CREATE (mle)-[:TRANSFER_TO {name: 'Plane', type: 'vip_seaplane',
                     duration_minutes: 20, price_rub: 100, capacity: 8}]->(hm)
            CREATE (mle)-[:TRANSFER_TO {name: 'Small boat', type: 'speedboat',
                     duration_minutes: 25, price_rub: 200, capacity: 2}]->(hm)
            CREATE (mle)-[:TRANSFER_TO {name: 'Large boat', type: 'speedboat',
                     duration_minutes: 30, price_rub: 300, capacity: 8}]->(hm)
            CREATE (led)-[:TRANSFER_TO {name: 'Bridge', type: 'taxi', duration_minutes: 25,
                     price_rub: 100, bridge_crossing_minutes: 20,
                     bridge_closed_from: '01:10', bridge_closed_until: '05:00'}]->(hl)
            CREATE (led)-[:TRANSFER_TO {name: 'Bypass', type: 'taxi', duration_minutes: 30,
                     price_rub: 200}]->(hl)
            """,
            origin=origin, fixture=fixture, **hotels,
        )

        async def flight(source, destination, number, *, arrival="15:11", layover=None):
            await driver.execute_query(
                """
                MATCH (o:Airport {code: $source}), (d:Airport {code: $destination})
                MERGE (o)-[f:FLIGHT_TO {flight_no: $number}]->(d)
                SET f.airline = 'Test Air', f.departure_time = '14:00', f.arrival_time = $arrival,
                    f.duration_minutes = 120, f.price_rub_roundtrip = 20000,
                    f.flight_type = $flight_type, f.layover_duration_minutes = $layover
                """,
                source=source, destination=destination, number=f"{number}-{fixture}",
                arrival=arrival, flight_type="connecting" if layover is not None else "direct",
                layover=layover,
            )

        async def validate(airport="MLE", hotel="mle", **options):
            return await MCPTools.check_logistics_compatibility(
                airport, hotels[hotel], origin_airport_code=origin, **options
            )

        try:
            with patch("app.services.graph_service.get_neo4j_driver", return_value=driver):
                assert (await validate())["reason_code"] == "no_outbound_route"
                await flight(origin, "MLE", "out", layover=59)
                assert (await validate())["reason_code"] == "no_outbound_route"

                await flight(origin, "MLE", "out", layover=60)
                assert (await validate())["reason_code"] == "no_return_flight"
                await flight("MLE", origin, "back", layover=59)
                assert (await validate())["reason_code"] == "no_return_flight"
                assert (await validate(strict_graph_validation=False))["compatible"]
                await flight("MLE", origin, "back", layover=60)

                assert (await validate(travelers_count=3))["transfer"]["name"] == "Large boat"
                assert (await validate(prefer_direct_flight=True))["reason_code"] == "no_outbound_route"
                assert (await validate(travelers_count=9))["reason_code"] == "transfer_capacity"
                await flight(origin, "MLE", "out", arrival="15:10", layover=60)
                result = await validate(travelers_count=3)
                assert result["transfer"]["name"] == "Plane"
                assert result["flight"]["return_flight"]["layover_duration_minutes"] == 60

                await flight(origin, "MLE", "out", arrival="invalid", layover=60)
                assert (await validate())["reason_code"] == "invalid_timeline"
                await flight(origin, "LED", "led-out", arrival="00:00")
                await flight("LED", origin, "led-back")
                assert (await validate("LED", "led"))["transfer"]["name"] == "Bypass"
                await flight(origin, "LED", "led-out", arrival="03:50")
                bridge = (await validate("LED", "led"))["transfer"]
                assert bridge["name"] == "Bridge"
                assert bridge["capacity_status"] == "unknown"

                assert (await validate("LED", "self_hotel"))["reason_code"] == "no_outbound_route"
                self_transfer = await validate("LED", "self_hotel", without_transfer=True)
                assert self_transfer["compatible"]
                assert "duration_min" not in self_transfer["transfer"]

                cross_city = await validate("LED", "led", return_airport_code="MLE")
                assert cross_city["flight"]["return_flight"]["origin_code"] == "MLE"
        finally:
            await driver.execute_query(
                "MATCH (n {validation_fixture: $fixture}) DETACH DELETE n", fixture=fixture
            )
