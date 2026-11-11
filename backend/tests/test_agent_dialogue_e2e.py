import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService


class FakeSession:
    def __init__(self, user):
        self.user = user

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, model, user_id):
        return self.user

    async def commit(self):
        return None


def hotel(hotel_id, name, stars, price):
    return {
        "id": hotel_id,
        "name": name,
        "city": "Сочи",
        "stars": stars,
        "price_per_night_rub": price,
        "price_per_night_formatted": f"{price:,} ₽",
        "meals": "Завтрак",
        "distance_to_sea": "100 м",
        "cancellation": "Бесплатно",
        "features": [],
    }


class DialogueTestDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 30)


class AgentDialogueE2ETest(unittest.TestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            profile={},
            current_cart={},
            current_state="INSPIRATION",
            conversation_history=[],
            trip_state=None,
            trip_quote=None,
        )
        self.hotels = [
            hotel("hostel", "Хостел У Моря", "1★", 3000),
            hotel("comfort", "Гранд Отель Сочи", "4★", 7000),
            hotel("premium", "Сочи Палас", "5★", 14000),
        ]
        logistics = {
            "compatible": True,
            "airport": "AER",
            "flight": {
                "price_rub_roundtrip": 20000,
                "origin_code": "SVO",
                "origin_name": "Москва",
                "airline": "Test Air",
                "flight_no": "T1",
            },
            "transfer": {"price_rub": 2000, "name": "Трансфер"},
            "timeline": {},
        }
        self.patches = [
            patch("app.services.trip_dates.date", DialogueTestDate),
            patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=self.user)),
            patch("app.services.agent_service.async_session_maker", side_effect=lambda: FakeSession(self.user)),
            patch("app.services.graph_repository.graph_repository.save_tour_version", new_callable=AsyncMock),
            patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[
                {"id": "sochi", "name": "Сочи", "score": 1.0},
            ])),
            patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": self.hotels})),
            patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)),
            patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})),
            patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()

    def post(self, client, endpoint, message):
        response = client.post(endpoint, json={"user_id": "web:e2e-dialogue", "message": message})
        self.assertEqual(response.status_code, 200, response.text)
        if endpoint == "/api/chat":
            return response.json()
        for block in response.text.split("\n\n"):
            if block.startswith("event: result\n"):
                return json.loads(next(line[5:] for line in block.splitlines() if line.startswith("data:")))
        self.fail(f"SSE result event not found: {response.text}")

    def assert_dialogue(self, endpoint):
        client = TestClient(app)
        first = self.post(client, endpoint, "Хочу на море")
        self.assertEqual(first["state"], "CLARIFICATION_REQUIRED")
        self.assertTrue(first["trip_state"]["pending_action"]["missing_fields"])

        second = self.post(client, endpoint, "В Сочи, с 10 по 15 ноября, 2 взрослых, бюджет 180 000")
        self.assertNotEqual(second.get("state"), "CLARIFICATION_REQUIRED", second["text"])
        self.assertEqual(second["trip_state"]["budget_rub"], 180000)
        self.assertEqual(second["trip_state"]["dates"], {
            "checkin_date": "2026-11-10",
            "checkout_date": "2026-11-15",
            "nights": 5,
        })
        self.assertEqual(second["trip_state"]["passengers"]["adults_count"], 2)
        self.assertIsNone(second["trip_state"]["pending_action"])
        self.assertEqual(second["cart"]["hotel_id"], "comfort")
        self.assertLessEqual(second["cart"]["total_trip_budget_range_rub"]["max"], 180000)

        third = self.post(client, endpoint, "А если бюджет 220 000 и отель получше?")
        self.assertNotEqual(third.get("state"), "CLARIFICATION_REQUIRED", third["text"])
        self.assertEqual(third["trip_state"]["budget_rub"], 220000)
        self.assertEqual(third["trip_state"]["dates"], second["trip_state"]["dates"])
        self.assertEqual(third["trip_state"]["passengers"]["adults_count"], 2)
        self.assertIsNone(third["trip_state"]["pending_action"])
        upgrade = third.get("alternative_cart") or third["cart"]
        self.assertEqual(upgrade["hotel_id"], "premium")
        self.assertGreater(
            upgrade["price_per_night_rub"],
            second["cart"]["price_per_night_rub"],
        )
        self.assertLessEqual(upgrade["total_trip_budget_range_rub"]["max"], 220000)

    def test_three_turn_dialogue_via_rest(self):
        self.assert_dialogue("/api/chat")

    def test_three_turn_dialogue_via_sse(self):
        self.assert_dialogue("/api/chat/stream")


if __name__ == "__main__":
    unittest.main()
