"""Regression checks without PostgreSQL, Neo4j, or external API calls.

Run from backend: python -m unittest discover -s tests -v
"""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.main import app
from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService


class ChatSessionsTest(unittest.TestCase):
    def test_separate_browsers_and_returning_browser(self):
        with patch.object(AgentService, "process_message", new_callable=AsyncMock) as process:
            process.return_value = {"text": "ok"}
            with_browser = TestClient(app)
            other_browser = TestClient(app)
            first = with_browser.post("/api/chat", json={"message": "Привет"})
            first_id = process.call_args.kwargs["user_id"]
            self.assertEqual(first.status_code, 200)
            self.assertIn("HttpOnly", first.headers["set-cookie"])
            with_browser.post("/api/chat", json={"message": "Продолжим"})
            self.assertEqual(first_id, process.call_args.kwargs["user_id"])
            other_browser.post("/api/chat", json={"message": "Привет"})
            self.assertNotEqual(first_id, process.call_args.kwargs["user_id"])

    def test_invalid_cookie_is_replaced(self):
        with patch.object(AgentService, "process_message", new_callable=AsyncMock) as process:
            process.return_value = {"text": "ok"}
            browser = TestClient(app)
            browser.cookies.set("ostrovok_session", "invalid")
            response = browser.post("/api/chat", json={"message": "Привет"})
            self.assertEqual(response.status_code, 200)
            self.assertNotEqual(process.call_args.kwargs["user_id"], "web:invalid")

    def test_explicit_api_user_is_preserved(self):
        with patch.object(AgentService, "process_message", new_callable=AsyncMock) as process:
            process.return_value = {"text": "ok"}
            TestClient(app).post("/api/chat", json={"message": "Привет", "user_id": "api-user"})
            self.assertEqual(process.call_args.kwargs["user_id"], "api-user")


class HotelConstraintsTest(unittest.IsolatedAsyncioTestCase):
    async def test_matching_family_hotel_is_still_recommended(self):
        hotel = SimpleNamespace(
            id="family", name="Family hotel", stars=4, category="family",
            features=["family_kids"], family_friendly=True, pet_friendly=True,
            price_per_night_rub=5000, meals="Завтрак", cancellation_policy="Бесплатно",
            distance_to_sea_meters=100,
        )
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.execute.return_value = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [hotel])
        )
        with patch("app.mcp.tools.async_session_maker", return_value=db):
            result = await MCPTools.compare_hotels(
                "sochi", family_friendly=True, pet_friendly=True,
            )
        self.assertEqual(result["recommended_choice"]["id"], "family")

    async def test_no_match_does_not_retry_without_constraints(self):
        db = AsyncMock()
        db.execute.return_value = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [])
        )
        db.__aenter__.return_value = db
        with patch("app.mcp.tools.async_session_maker", return_value=db):
            result = await MCPTools.compare_hotels(
                "sochi", pet_friendly=True, family_friendly=True,
                max_price_per_night=1000, exclude_hotel_ids=["rejected-a", "rejected-b"],
            )
        self.assertEqual(result["compared_hotels"], [])
        self.assertIsNone(result["recommended_choice"])
        db.execute.assert_awaited_once()
        query = str(db.execute.call_args.args[0].compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        ))
        for required in ("pet_friendly = true", "family_friendly = true",
                         "price_per_night_rub <= 1000", "rejected-a", "rejected-b"):
            self.assertIn(required, query)

    async def test_adults_only_cannot_be_recommended_to_family(self):
        # An inconsistent record marked family-friendly must still be excluded.
        hotel = SimpleNamespace(features=["adults_only"], family_friendly=True)
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.execute.return_value = SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: [hotel])
        )
        with patch("app.mcp.tools.async_session_maker", return_value=db):
            result = await MCPTools.compare_hotels("sochi", family_friendly=True)
        self.assertEqual(result["compared_hotels"], [])

    async def test_no_match_invalidates_old_cart_and_skips_logistics(self):
        old_session = SimpleNamespace(
            profile={"has_pets": True, "has_kids": True, "nights": 7},
            current_cart={"destination_id": "sochi", "destination": "Сочи",
                          "hotel": "Old hotel", "hotel_id": "old", "nights": 7,
                          "price_per_night_rub": 10000, "total_mandatory_rub": 90000},
            current_state="INSPIRATION",
        )
        destination = {"id": "sochi", "name": "Сочи", "score": 1.0}
        with (
            patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=old_session)),
            patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[destination])),
            patch.object(AgentService, "_save_session", new_callable=AsyncMock) as save,
            patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": []})),
            patch.object(MCPTools, "check_logistics_compatibility", new_callable=AsyncMock) as logistics,
            patch.object(MCPTools, "checkout_cart", new_callable=AsyncMock) as checkout,
        ):
            result = await AgentService.process_message("test", "покажи другой отель")
            self.assertEqual(result["state"], "NO_MATCHES")
            self.assertNotIn("hotel", result["cart"])
            self.assertNotIn("total_mandatory_rub", result["cart"])
            self.assertIn("с питомцем", result["text"])
            self.assertIn("с детьми", result["text"])
            self.assertIn("old", save.call_args.args[1]["rejected_hotels"])
            old_session.current_cart = result["cart"]
            await AgentService.process_message("test", "забронировать")
            logistics.assert_not_awaited()
            checkout.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
