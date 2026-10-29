import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

from app.config import settings
from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService
from app.services.llm_service import LLMService
from app.services.trip_budget import update_trip_budget


def hotel(identifier="hotel", price=1000):
    return {
        "id": identifier,
        "name": identifier,
        "city": "Дубай",
        "price_per_night_rub": price,
        "price_per_night_formatted": f"{price} ₽",
        "meals": "Завтрак",
        "stars": "4★",
        "distance_to_sea": "100 м",
        "cancellation": "Бесплатно",
    }


class ReportedRegressionsTest(unittest.IsolatedAsyncioTestCase):
    def patches(self, user, destination_id="uae"):
        destination = {"id": destination_id, "name": "ОАЭ", "score": 1.0}
        logistics = {
            "compatible": True,
            "flight": {"price_rub_roundtrip": 20000},
            "transfer": {"price_rub": 2000},
            "timeline": {},
        }
        stack = ExitStack()
        stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
        stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[destination])))
        stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
        stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": [hotel()]})))
        stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)))
        stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
        stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))
        return stack

    def test_budget_is_not_invented_and_explicit_value_overwrites_old_one(self):
        profile = {}
        self.assertIsNone(update_trip_budget("ОАЭ на 5 ночей, отель 5 звезд", profile, {}))
        self.assertNotIn("budget_rub", profile)

        profile["budget_rub"] = 5000
        update_trip_budget("поставь лимит 85000 рублей", profile, {})
        self.assertEqual(profile["budget_rub"], 85000)
        update_trip_budget("бюджет 95000", profile, {})
        self.assertEqual(profile["budget_rub"], 95000)

    async def test_uae_slot_filling_keeps_all_slots_and_dot_budget(self):
        user = SimpleNamespace(profile={"nights": 7}, current_cart={}, current_state="INSPIRATION")
        message = "Хотим в ОАЭ, нас 3, так что нужно 2 комнаты. Общий бюджет - 200.000, на 5 ночей"
        with self.patches(user):
            result = await AgentService.process_message("test", message)
            saved_profile = user.profile
        self.assertEqual(result["cart"]["destination_id"], "uae")
        self.assertEqual(saved_profile["budget_rub"], 200000)
        self.assertEqual(saved_profile["adults_count"], 3)
        self.assertEqual(saved_profile["rooms_count"], 2)
        self.assertEqual(saved_profile["nights"], 5)

        user.current_cart = result["cart"]
        user.profile["awaiting_budget"] = True
        with self.patches(user):
            followup = await AgentService.process_message("test", "200.000")
        self.assertEqual(followup["cart"]["destination_id"], "uae")
        self.assertEqual(user.profile["budget_rub"], 200000)

    def test_sea_request_excludes_altai_before_ranking(self):
        dest_id, _, _ = AgentService.select_multi_city_route(
            "Составь тур по городам на море", {"interests": ["beach"]}, current_dest_id="altai"
        )
        self.assertNotEqual(dest_id, "altai")

    async def test_pet_followup_does_not_replace_saved_sochi_with_vector_match(self):
        user = SimpleNamespace(
            profile={"nights": 5, "adults_count": 3, "rooms_count": 2},
            current_cart={"destination_id": "sochi", "destination": "Сочи"},
            current_state="PLANNING",
        )

        async def destinations(query, **kwargs):
            if query == "А ещё с нами будет собака":
                return [{"id": "turkey", "name": "Турция", "score": 0.95}]
            return [{"id": "sochi", "name": "Сочи", "score": 1.0}]

        with self.patches(user) as stack:
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(side_effect=destinations)))
            result = await AgentService.process_message("test", "А ещё с нами будет собака")

        self.assertEqual(result["cart"]["destination_id"], "sochi")
        self.assertTrue(user.profile["has_pets"])
        self.assertNotIn("Турция", result["text"])

    async def test_response_has_no_ai_concierge_or_model_name(self):
        user = SimpleNamespace(profile={"nights": 5}, current_cart={}, current_state="INSPIRATION")
        with self.patches(user) as stack:
            stack.enter_context(patch.object(type(settings), "is_mock_llm", new_callable=PropertyMock, return_value=False))
            stack.enter_context(patch.object(LLMService, "dispatch_intent", new=AsyncMock(return_value={"action": "plan"})))
            advice = stack.enter_context(patch.object(LLMService, "generate_concierge_advice", new_callable=AsyncMock))
            result = await AgentService.process_message("real-user", "ОАЭ, бюджет 200.000")
        advice.assert_not_awaited()
        self.assertNotIn("AI-консьерж", result["text"])
        self.assertNotIn(settings.OPENAI_MODEL, result["text"])

    async def test_flight_prices_include_every_traveler(self):
        user = SimpleNamespace(
            profile={
                "nights": 5,
                "adults_count": 3,
                "kids_count": 0,
                "rooms_count": 2,
                "checkin_date": "2026-11-11",
                "checkout_date": "2026-11-16",
            },
            current_cart={},
            current_state="INSPIRATION",
        )
        with self.patches(user):
            result = await AgentService.process_message("test", "ОАЭ, нас 3, бюджет 200000")

        mandatory = result["budget_breakdown"]["mandatory"]
        self.assertEqual(mandatory["flight_price_roundtrip"], 36000)
        self.assertEqual(mandatory["flight_out_price"], 18000)
        self.assertEqual(mandatory["flight_ret_price"], 18000)
        self.assertEqual(mandatory["total_mandatory_rub"], 47600)
        self.assertEqual(result["cart"]["total_mandatory_rub"], 47600)

    async def test_identical_requests_keep_price_and_cart_in_sync(self):
        user = SimpleNamespace(
            profile={"nights": 6, "checkin_date": "2026-11-13", "checkout_date": "2026-11-19"},
            current_cart={"destination_id": "uae", "destination": "ОАЭ"},
            current_state="INSPIRATION",
        )
        message = "самый дешевый вариант в любое время"
        with self.patches(user):
            first = await AgentService.process_message("test", message)
            user.profile.update(first["cart"])
            second = await AgentService.process_message("test", message)
        self.assertEqual(first["cart"]["total_mandatory_rub"], second["cart"]["total_mandatory_rub"])
        self.assertEqual(first["cart"]["total_trip_budget_range_rub"], second["cart"]["total_trip_budget_range_rub"])
        self.assertEqual(first["cart"]["checkin_date"], second["cart"]["checkin_date"])
        self.assertEqual(second["cart"]["total_mandatory_rub"], second["budget_breakdown"]["mandatory"]["total_mandatory_rub"])


if __name__ == "__main__":
    unittest.main()
