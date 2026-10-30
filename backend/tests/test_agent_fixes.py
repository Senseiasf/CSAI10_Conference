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

    async def test_budget_limit_does_not_enable_cheapest_mode(self):
        hotels = [
            make_hotel("expensive", "Гранд Люкс Отель", 12000),
            make_hotel("mid", "Комфорт Отель", 6000),
            make_hotel("cheap", "Эконом Резорт", 3000),
        ]
        result, _ = await self.run_agent("Сочи бюджет до 300 тысяч", hotels)
        self.assertEqual(result["ltr_ranking"]["selection_policy"], "ltr")
        self.assertLessEqual(result["cart"]["budget_comparison"]["compared_amount_rub"], 300000)
        self.assertEqual(result["cart"]["budget_comparison"]["basis"], "full_max")

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

        # 1500 * 7 = 10,500 ₽ pet surcharge; cart uses the same calendar rate as the response.
        self.assertEqual(result["cart"]["total_mandatory_rub"],
                         result["budget_breakdown"]["mandatory"]["total_mandatory_rub"])
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
        self.assertEqual(result["state"], "BUDGET_EXCEEDED")
        self.assertIn("подходящих вариантов нет", result["text"])
        self.assertEqual(len(result["compared_hotels"]), 1)

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

    def test_destinations_json_no_duplicate_ids_and_full_transfer_coverage(self):
        """Block 1 & 6: destinations.json has 9 unique IDs, all hotels have airport transfers."""
        import json
        import os
        dest_path = os.path.join(os.path.dirname(__file__), "..", "app", "data", "destinations.json")
        with open(dest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        destinations = data.get("destinations", [])
        ids = [d["id"] for d in destinations]
        self.assertEqual(len(ids), 9)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("georgia", ids)

        for dest in destinations:
            airport_codes = {a["code"] if isinstance(a, dict) else a for a in dest.get("airports", [])}
            hotels = dest.get("hotels", [])
            transfers = dest.get("transfers", [])

            covered_hotels = {
                t["to"] for t in transfers if t.get("from") in airport_codes
            }
            for hotel in hotels:
                self.assertIn(
                    hotel["id"],
                    covered_hotels,
                    f"Hotel {hotel['id']} in {dest['id']} lacks an airport transfer"
                )

    async def test_vague_greeting_sea_returns_inspiration_prompt_without_multi_city(self):
        """Вдохновение сохраняет критерий и pending без сброса текущей поездки."""
        cart = {"destination_id": "turkey", "destination": "Турция (Анталья и Стамбул)", "is_multi_city": True}
        profile = {"interests": ["mountains", "beach"], "is_multi_city": True}
        user = SimpleNamespace(
            profile=profile,
            current_cart=cart,
            current_state="PLANNED"
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            result = await AgentService.process_message("test_sea", "Хочу на море")

        self.assertEqual(result["state"], "CLARIFICATION_REQUIRED")
        self.assertEqual(result["cart"]["destination_id"], "turkey")
        self.assertFalse(user.profile.get("is_multi_city", False))
        self.assertFalse(result["trip_state"]["preferences"]["is_multi_city"])
        self.assertNotIn("Стамбул", result.get("text", ""))
        self.assertNotIn("Красная Поляна", result.get("text", ""))

    def test_month_without_day_requires_exact_day(self):
        from app.services.trip_dates import update_trip_dates
        profile = {}
        err = update_trip_dates("Хочу в Турцию в ноябре на 7 ночей", profile, {})
        self.assertIsNotNone(err)
        self.assertNotIn("checkin_date", profile)

    def test_unified_pricing_math_between_base_and_flexible_dates(self):
        """Block 4 & 6: calculate_real_trip_cost_by_date matches calculate_flexible_dates_options savings exactly."""
        from datetime import date
        from app.services.agent_service import AgentService
        base_d_in = date(2026, 11, 14)
        nights = 7
        raw_flight_rt = 20000
        hotel_base_price = 5000
        fixed_costs = 2000

        cur_calc = AgentService.calculate_real_trip_cost_by_date(
            base_d_in, nights, raw_flight_rt, hotel_base_price, fixed_costs
        )
        flex_options = AgentService.calculate_flexible_dates_options(
            dest_id="turkey",
            base_mandatory_rub=cur_calc["total_mandatory"],
            checkin_date_str=base_d_in.isoformat(),
            nights=nights,
            base_flight_price=raw_flight_rt,
            base_hotel_price=hotel_base_price,
            fixed_costs=fixed_costs
        )
        best_opt = flex_options[0]
        expected_diff = cur_calc["total_mandatory"] - best_opt["target_mandatory_rub"]
        self.assertEqual(expected_diff, best_opt["savings_rub"])

