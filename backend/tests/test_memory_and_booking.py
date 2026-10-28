import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from app.main import app
from app.services.agent_service import AgentService
from app.services.booking_stub import detect_booking_action
from app.services.llm_service import LLMService
from app.services.preference_memory import remembered_preferences
from app.mcp.tools import MCPTools


class BookingStubsTest(unittest.IsolatedAsyncioTestCase):
    async def test_operations_do_not_mutate_cart_or_call_checkout(self):
        for message, action in (("Забронируй отель", "book"), ("Хочу забронировать", "book"),
                                ("Хочу отменить бронь", "cancel"), ("Перенеси бронь на 2 дня", "reschedule")):
            with self.subTest(message=message):
                cart = {"hotel": "Hotel", "destination": "Сочи", "checkin_date": "2027-01-10"}
                profile = {"has_pets": True, "budget_rub": 100000}
                original = copy.deepcopy(cart)
                user = SimpleNamespace(profile=profile, current_cart=cart)
                with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)),
                      patch.object(AgentService, "_save_session", new_callable=AsyncMock),
                      patch.object(MCPTools, "checkout_cart", new_callable=AsyncMock) as checkout,
                      patch.object(LLMService, "dispatch_intent", new_callable=AsyncMock) as llm):
                    result = await AgentService.process_message("test-stub", message)
                self.assertEqual(result["state"], "BOOKING_UNAVAILABLE")
                self.assertEqual(result["booking_action"], {"action": action, "available": False})
                self.assertEqual(result["cart"], original)
                self.assertNotIn("booking", result)
                checkout.assert_not_awaited()
                llm.assert_not_awaited()

    def test_planning_and_policy_questions_are_not_booking_operations(self):
        for message in ("Сдвинь поездку на 2 дня позже", "Отмени трансфер", "Условия отмены брони?",
                        "Не хочу бронировать", "Не отменяй бронь", "Какие отели с бесплатной отменой?"):
            with self.subTest(message=message):
                self.assertIsNone(detect_booking_action(message))

    async def test_stub_without_cart_is_still_explicit(self):
        user = SimpleNamespace(profile={}, current_cart={})
        with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)),
              patch.object(AgentService, "_save_session", new_callable=AsyncMock)):
            result = await AgentService.process_message("test-empty", "Забронировать")
        self.assertEqual(result["state"], "BOOKING_UNAVAILABLE")
        self.assertIn("Бронь не создана", result["text"])

    async def test_llm_booking_intent_returns_stub(self):
        user = SimpleNamespace(profile={}, current_cart={}, conversation_history=[])
        with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)),
              patch.object(AgentService, "_save_session", new_callable=AsyncMock),
              patch("app.services.agent_service.settings") as settings,
              patch.object(LLMService, "dispatch_intent", new=AsyncMock(return_value={"action": "cancel_booking"})),
              patch.object(MCPTools, "checkout_cart", new_callable=AsyncMock) as checkout):
            settings.is_mock_llm = False
            result = await AgentService.process_message("web:llm-test", "Убери мой ранее оформленный заказ")
        self.assertEqual(result["booking_action"]["action"], "cancel")
        checkout.assert_not_awaited()


class PreferenceMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_trip_clears_itinerary_and_keeps_preferences(self):
        user = SimpleNamespace(
            profile={"has_kids": True, "has_pets": True, "private_room_required": True,
                     "interests": ["beach"], "origin_airport": "LED", "nights": 7,
                     "budget_rub": 100000, "checkin_date": "2027-01-10",
                     "llm_destination_id": "sochi", "rejected_hotels": ["old"]},
            current_cart={"hotel": "Old"}, conversation_history=[{"user": "Old"}], current_state="CONFIRMED")
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.get.return_value = user
        with patch("app.services.agent_service.async_session_maker", return_value=db):
            prefs = await AgentService.reset_trip("web:user")
        self.assertTrue(prefs["has_kids"])
        self.assertTrue(prefs["has_pets"])
        self.assertEqual(prefs["interests"], ["beach"])
        self.assertEqual(prefs["origin_airport"], "LED")
        for key in ("budget_rub", "nights", "checkin_date", "llm_destination_id", "rejected_hotels"):
            self.assertNotIn(key, user.profile)
        self.assertEqual(user.current_cart, {})
        self.assertEqual(user.conversation_history, [])
        self.assertEqual(user.current_state, "INSPIRATION")
        db.commit.assert_awaited_once()
        db.get.assert_awaited_once()
        self.assertEqual(db.get.call_args.args[1], "web:user")

    async def test_forget_preferences_clears_saved_data(self):
        user = SimpleNamespace(profile={"has_pets": True}, current_cart={"hotel": "Old"}, conversation_history=[])
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.get.return_value = user
        with patch("app.services.agent_service.async_session_maker", return_value=db):
            self.assertEqual(await AgentService.reset_trip("web:user", True), {})
        self.assertEqual(user.profile, {})

    def test_explicit_new_conditions_override_memory(self):
        profile = {"has_kids": True, "has_pets": True, "kids_count": 1,
                   "interests": ["family_kids", "pet_friendly"]}
        AgentService.extract_interests_and_slots("на этот раз без детей и без собаки", profile)
        self.assertFalse(profile["has_kids"])
        self.assertFalse(profile["has_pets"])
        self.assertEqual(profile["kids_count"], 0)
        self.assertNotIn("family_kids", profile["interests"])
        self.assertNotIn("pet_friendly", profile["interests"])

    def test_preference_snapshot_does_not_share_mutable_values(self):
        profile = {"interests": ["beach"]}
        prefs = remembered_preferences(profile)
        prefs["interests"].append("mountains")
        self.assertEqual(profile["interests"], ["beach"])


class MemoryApiTest(unittest.TestCase):
    def test_profile_survives_new_trip_same_browser(self):
        saved = {"has_pets": True, "private_room_required": True}
        with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=SimpleNamespace(profile=saved))) as get,
              patch.object(AgentService, "reset_trip", new=AsyncMock(return_value=saved)) as reset):
            browser = TestClient(app)
            first = browser.get("/api/profile")
            identity = get.call_args.args[0]
            self.assertEqual(first.json()["preferences"], saved)
            response = browser.post("/api/session/reset")
            self.assertEqual(reset.call_args.args[0], identity)
            self.assertFalse(reset.call_args.kwargs["forget_preferences"])
            self.assertIn("сохранены", response.json()["message"])
            browser.get("/api/profile")
            self.assertEqual(get.call_args.args[0], identity)
            other = TestClient(app)
            other.get("/api/profile")
            self.assertNotEqual(get.call_args.args[0], identity)

    def test_stream_sets_identity_cookie_for_followup_requests(self):
        with patch.object(AgentService, "process_message", new=AsyncMock(return_value={"text": "Демо", "booking_action": {"action": "book"}})) as process:
            browser = TestClient(app)
            first = browser.post("/api/chat/stream", json={"message": "Забронировать"})
            identity = process.call_args.kwargs["user_id"]
            self.assertIn("ostrovok_session", browser.cookies)
            self.assertIn("httponly", first.headers["set-cookie"].lower())
            self.assertNotIn("event: status", first.text)
            browser.post("/api/chat", json={"message": "Дальше"})
            self.assertEqual(process.call_args.kwargs["user_id"], identity)

    def test_explicit_forget_flag_is_passed(self):
        with patch.object(AgentService, "reset_trip", new=AsyncMock(return_value={})) as reset:
            response = TestClient(app).post("/api/session/reset", json={"forget_preferences": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(reset.call_args.kwargs["forget_preferences"])


class LlmMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_llm_receives_saved_preferences(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[], content="Уточните даты"))])))))
        with patch.object(LLMService, "get_client", return_value=client):
            await LLMService.dispatch_intent("Новая поездка", {"has_pets": True, "private_room_required": True}, {})
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        context = json.loads(messages[1]["content"].split(": ", 1)[1])
        self.assertTrue(context["saved_preferences"]["has_pets"])
        self.assertTrue(context["saved_preferences"]["private_room_required"])


if __name__ == "__main__":
    unittest.main()
