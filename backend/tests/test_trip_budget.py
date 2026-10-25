import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.trip_budget import update_trip_budget
from app.services.agent_service import AgentService
from app.mcp.tools import MCPTools


class BudgetParsingTest(unittest.TestCase):
    def test_amount_formats(self):
        for message, expected in (
            ("Сочи до 100 тысяч", 100000), ("бюджет 100 000 рублей", 100000),
            ("до 100к", 100000), ("бюджет: 1,5 млн", 1500000),
            ("уложиться в 85 тыс.", 85000), ("не более 100000 ₽", 100000),
            ("бюджет 100\u202f000", 100000),
        ):
            with self.subTest(message=message):
                profile = {}
                self.assertIsNone(update_trip_budget(message, profile, {}))
                self.assertEqual(profile["budget_rub"], expected)

    def test_reduction_uses_total_including_onsite_expenses(self):
        profile = {}
        cart = {"total_mandatory_rub": 90000,
                "total_trip_budget_range_rub": {"min": 110000, "max": 140000}}
        self.assertIsNone(update_trip_budget("сделай дешевле на 30 тысяч", profile, cart))
        self.assertEqual(profile["budget_rub"], 110000)

    def test_reduction_does_not_raise_existing_limit(self):
        profile = {"budget_rub": 100000}
        cart = {"total_trip_budget_range_rub": {"max": 140000}}
        update_trip_budget("дешевле на 10 тысяч", profile, cart)
        self.assertEqual(profile["budget_rub"], 100000)

    def test_unclear_scope_or_impossible_amount_asks_question(self):
        for message in ("бюджет 0", "бюджет -100", "бюджет $100", "бюджет сто тысяч", "бюджет 100 евро", "до 10000 за ночь",
                        "бюджет 100 тысяч на человека", "дешевле на 30 тысяч"):
            with self.subTest(message=message):
                profile = {"budget_rub": 100000}
                self.assertIsNotNone(update_trip_budget(message, profile, {}))
                self.assertEqual(profile["budget_rub"], 100000)

    def test_unrelated_numbers_do_not_overwrite_budget(self):
        profile = {"budget_rub": 100000}
        for message in ("до 7 ночей", "до 10.11.2026", "на 2 дня позже", "ребёнку 5 лет", "другой отель"):
            self.assertIsNone(update_trip_budget(message, profile, {}))
            self.assertEqual(profile["budget_rub"], 100000)


def hotel(identifier, price):
    return {"id": identifier, "name": identifier, "price_per_night_rub": price,
            "price_per_night_formatted": f"{price} ₽", "meals": "Завтрак",
            "stars": "4★", "distance_to_sea": "100 м", "cancellation": "Бесплатно"}


class BudgetSelectionTest(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, message, hotels, *, profile=None, cart=None, compatible=True):
        user = SimpleNamespace(profile=profile or {"nights": 7, "has_pets": True, "has_kids": True},
                               current_cart=cart or {}, current_state="INSPIRATION")
        destination = {"id": "sochi", "name": "Сочи", "score": 1.0}
        logistics = {"compatible": compatible, "flight": {"price_rub_roundtrip": 20000},
                     "transfer": {"price_rub": 2000}, "timeline": {}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[destination])))
            save = stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            compare = stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": hotels})))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))
            checkout = stack.enter_context(patch.object(MCPTools, "checkout_cart", new_callable=AsyncMock))
            result = await AgentService.process_message("test", message)
        return result, save, compare, checkout

    async def test_finds_affordable_hotel_beyond_top_three(self):
        result, save, compare, _ = await self.run_agent(
            "Сочи до 100 тысяч", [hotel("luxury", 50000), hotel("second", 40000),
                                    hotel("third", 30000), hotel("affordable", 1000)],
        )
        self.assertEqual(result["cart"]["hotel_id"], "affordable")
        self.assertLessEqual(result["budget_breakdown"]["total_trip_budget_range_rub"]["max"], 100000)
        self.assertEqual(compare.call_args.kwargs["limit"], None)
        self.assertTrue(compare.call_args.kwargs["pet_friendly"])
        self.assertTrue(compare.call_args.kwargs["family_friendly"])
        self.assertEqual(save.call_args.args[1]["budget_rub"], 100000)
        self.assertEqual([h["id"] for h in result["compared_hotels"]], ["affordable"])

    async def test_onsite_expenses_can_put_trip_over_budget(self):
        result, _, _, _ = await self.run_agent("Сочи до 60 тысяч", [hotel("cheap-room", 4000)])
        self.assertEqual(result["state"], "BUDGET_EXCEEDED")
        self.assertNotIn("hotel", result["cart"])
        self.assertIn("выше лимита", result["text"])

    async def test_remembered_budget_applies_to_followup(self):
        result, _, _, _ = await self.run_agent("другой отель", [hotel("expensive", 50000)],
                                             profile={"nights": 7, "budget_rub": 100000})
        self.assertEqual(result["state"], "BUDGET_EXCEEDED")

    async def test_exact_reduction_is_applied_to_full_cost(self):
        result, _, _, _ = await self.run_agent(
            "сделай дешевле на 30 тысяч", [hotel("affordable", 1000)],
            cart={"destination_id": "sochi", "destination": "Сочи", "nights": 7,
                  "total_mandatory_rub": 90000,
                  "total_trip_budget_range_rub": {"min": 110000, "max": 140000}},
        )
        self.assertEqual(result["cart"]["budget_rub"], 110000)
        self.assertLessEqual(result["budget_breakdown"]["total_trip_budget_range_rub"]["max"], 110000)
        self.assertIn("41,000", result["text"])

    async def test_exact_ceiling_is_allowed(self):
        result, _, _, _ = await self.run_agent("Сочи до 99 тысяч", [hotel("affordable", 1000)])
        self.assertEqual(result["cart"]["total_trip_budget_range_rub"]["max"], 99000)

    async def test_checkout_with_new_budget_does_not_book_old_cart(self):
        result, _, _, checkout = await self.run_agent(
            "забронировать, но бюджет до 60 тысяч", [hotel("too-much", 4000)],
            cart={"hotel": "Old", "destination": "Сочи", "destination_id": "sochi",
                  "total_trip_budget_range_rub": {"max": 200000}},
        )
        checkout.assert_not_awaited()
        self.assertEqual(result["state"], "BUDGET_EXCEEDED")

    async def test_no_route_does_not_use_invented_flight_price(self):
        result, _, _, _ = await self.run_agent("Сочи до 100 тысяч", [hotel("hotel", 3000)], compatible=False)
        self.assertEqual(result["state"], "NO_ROUTE")
        self.assertNotIn("hotel", result["cart"])


if __name__ == "__main__":
    unittest.main()
