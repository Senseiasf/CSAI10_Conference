import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService


def make_hotel(h_id, name, price, pet_friendly=True, pet_surcharge=1500):
    return {
        "id": h_id,
        "name": name,
        "price_per_night_rub": price,
        "price_per_night_formatted": f"{price:,} ₽",
        "meals": "Завтрак включен",
        "stars": "4★",
        "distance_to_sea": "100 м",
        "cancellation": "Бесплатно",
        "pet_friendly": pet_friendly,
        "pet_surcharge_per_night_rub": pet_surcharge if pet_friendly else 0,
        "features": ["pet_friendly"] if pet_friendly else []
    }


class AgentFixesTest(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, message, hotels, *, profile=None, cart=None):
        user = SimpleNamespace(
            profile=profile or {"nights": 7, "has_pets": False, "has_kids": False},
            current_cart=cart or {},
            current_state="INSPIRATION"
        )
        destination = {"id": "sochi", "name": "Сочи", "score": 1.0}
        logistics = {
            "compatible": True,
            "flight": {"price_rub_roundtrip": 20000},
            "transfer": {"price_rub": 2000},
            "timeline": {}
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[destination])))
            save = stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": hotels})))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))
            result = await AgentService.process_message("test", message)
        return result, save

    async def test_small_budget_picks_cheapest_hotel_first(self):
        """Fix 1: When user specifies a budget, cheapest affordable hotel is recommended first."""
        hotels = [
            make_hotel("expensive", "Гранд Люкс Отель", 12000),
            make_hotel("mid", "Комфорт Отель", 6000),
            make_hotel("cheap", "Эконом Резорт", 3000),
        ]
        result, _ = await self.run_agent("Сочи бюджет до 150 тысяч", hotels)
        # All 3 fit into 150k, but the cheapest (3000/night) must be recommended first
        self.assertEqual(result["cart"]["hotel_id"], "cheap")
        self.assertEqual(result["cart"]["hotel"], "Эконом Резорт")

    async def test_internal_route_timing_response(self):
        """Fix 2: Internal route schedule questions are answered with departure, arrival, duration."""
        cart = {"destination_id": "turkey", "is_multi_city": True}
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart=cart,
            current_state="PLANNED"
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            result = await AgentService.process_message("test", "Во сколько внутренний рейс?")

        self.assertEqual(result["state"], "INTERNAL_SCHEDULE_INFO")
        self.assertIn("11:30", result["text"])
        self.assertIn("12:45", result["text"])
        self.assertIn("1 ч 15 мин", result["text"])
        self.assertIn("Turkish Airlines", result["text"])
        self.assertIn("Анталья ➔ Стамбул", result["text"])

    async def test_pet_surcharge_included_in_mandatory_and_breakdown(self):
        """Fix 3: When traveling with pets, pet surcharge is included in budget and displayed."""
        hotel = make_hotel("pet_hotel", "Pet Friendly Hotel", 5000, pet_friendly=True, pet_surcharge=1500)
        profile = {"nights": 7, "has_pets": True, "has_kids": False, "interests": ["pet_friendly"]}
        result, _ = await self.run_agent("Сочи", [hotel], profile=profile)

        # 1500 * 7 = 10,500 ₽ pet surcharge
        # hotel = 35,000, flight = 20,000, transfer = 2,000 -> mandatory without pet = 57,000; with pet = 67,500
        self.assertEqual(result["cart"]["total_mandatory_rub"], 67500)
        self.assertIn("Доплата за питомца", result["text"])
        self.assertIn("1,500", result["text"])
        self.assertIn("10,500", result["text"])

    async def test_budget_exceeded_does_not_say_fits_budget(self):
        """Fix 4: When budget limit is exceeded, does not say 'укладывается в лимит'."""
        hotels = [make_hotel("expensive", "Дорогой Отель", 10000)]
        # 10k budget is way below 10k/night * 7 nights + flight 20k
        result, _ = await self.run_agent("Сочи бюджет 10 тысяч", hotels)
        self.assertNotIn("укладывается в лимит", result["text"])
        self.assertNotIn("укладывается в бюджет", result["text"])
        self.assertIn("выше лимита", result["text"])
        self.assertIn("уложиться нереально", result["text"])

    async def test_splurge_and_luxury_picks_premium_hotel_first(self):
        """Fix: When user is ready to splurge / wants luxury, premium hotel is recommended, not cheap."""
        hotels = [
            make_hotel("budget", "Эконом Номер", 3000),
            make_hotel("mid", "Комфорт Отель", 10000),
            make_hotel("luxury", "Radisson Collection Luxury Palace", 45000),
        ]
        result, _ = await self.run_agent("Сочи, готов раскошелиться, пятизвездочный люкс", hotels)
        self.assertEqual(result["cart"]["hotel_id"], "luxury")
        self.assertEqual(result["cart"]["hotel"], "Radisson Collection Luxury Palace")
