"""Multi-turn checks for saved destinations, repricing and price-bound questions."""
import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import inspect
from sqlalchemy.orm.attributes import set_committed_value

from app.mcp.tools import MCPTools
from app.models import UserSessionModel
from app.services.agent_service import AgentService
from app.services.llm_service import LLMService
from app.domain.dialogue import DialogueDecision, DialogueIntent
from app.services.dialogue_parser import DialogueParser
from app.services.trip_dates import update_trip_dates
from app.services.trip_budget import validate_package_totals
from app.services.trip_state import requested_destination, rejected_destinations, hotel_selection, destination_choice, transfer_preference


def make_user(**overrides):
    profile = dict(nights=11, adults_count=3, kids_count=0, rooms_count=2,
                   checkin_date="2027-10-15", checkout_date="2027-10-26",
                   dates="15.10.2027 — 26.10.2027 (11 ночей)", budget_rub=85000)
    profile.update(overrides)
    return SimpleNamespace(profile=profile, current_state="PLANNING", conversation_history=[],
                           current_cart={"destination_id": "spb", "destination": "Санкт-Петербург",
                                         "nights": profile["nights"], "checkin_date": profile["checkin_date"],
                                         "checkout_date": profile["checkout_date"]})


class DialogueRegressionTest(unittest.IsolatedAsyncioTestCase):
    def fixture(self, user, *, hotel_rate=1000, transfer_price=2000, flight_price=20000,
                destination_hotels=None, destination_rows=None):
        stack = ExitStack()
        stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))

        async def save(_user_id, profile, cart, message, reply):
            user.profile = copy.deepcopy(profile)
            user.current_cart = copy.deepcopy(cart)
            user.conversation_history = user.conversation_history + [{"user": message, "agent": reply}]

        self.save = stack.enter_context(patch.object(AgentService, "_save_session", new=AsyncMock(side_effect=save)))

        async def search(query, **kwargs):
            if destination_rows:
                if kwargs.get("limit") == 9:
                    return copy.deepcopy(destination_rows)
                identifier = query if query in {row["id"] for row in destination_rows} else requested_destination(query)
                return copy.deepcopy([row for row in destination_rows if row["id"] == identifier] or destination_rows[:1])
            destination = query if query in ("spb", "sochi") else requested_destination(query) or "sochi"
            return [{"id": destination, "name": {"spb": "Санкт-Петербург", "sochi": "Сочи"}[destination], "score": 1.0}]

        self.search = stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(side_effect=search)))
        hotel = {"id": "hotel-1", "name": "Тестовый отель", "city": "Санкт-Петербург", "stars": "4★",
                 "price_per_night_rub": hotel_rate, "price_per_night_formatted": f"{hotel_rate:,} ₽",
                 "meals": "Завтрак", "distance_to_sea": "нет данных", "cancellation": "Бесплатно"}
        async def compare(destination_id, **kwargs):
            candidates = (destination_hotels or {}).get(destination_id, [hotel])
            return {"compared_hotels": copy.deepcopy([hotel for hotel in candidates if hotel["id"] not in kwargs.get("exclude_hotel_ids", [])])}
        self.compare = stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(side_effect=compare)))
        logistics = {"compatible": True, "airport": "LED", "flight": {"price_rub_roundtrip": flight_price,
                     "origin_code": "SVO", "origin_name": "Москва", "airline": "Test Air", "flight_no": "T1"},
                     "transfer": {"price_rub": transfer_price, "name": "Трансфер"}, "timeline": {}}
        self.logistics = stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)))
        stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
        stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))
        stack.enter_context(patch("app.services.agent_service.settings", SimpleNamespace(is_mock_llm=False)))
        self.dispatch = stack.enter_context(patch.object(LLMService, "dispatch_intent", new=AsyncMock(
            return_value={"action": "consultation", "text": "Старый ответ без пересчёта"})))
        return stack

    async def test_remove_transfer_preserves_every_other_quoted_segment(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            first = await AgentService.process_message("web:transfer", "Пересчитай на 5 ночей")
            before = copy.deepcopy(first["cart"])
            for mock in (self.search, self.compare, self.logistics, self.dispatch):
                mock.reset_mock()
            result = await AgentService.process_message("web:transfer", "Трансфер не нужен, оставь остальной план")
            for mock in (self.search, self.compare, self.logistics, self.dispatch):
                mock.assert_not_awaited()
        self.assertEqual(result["state"], "TRANSFER_REMOVED")
        after = result["cart"]
        for key in ("hotel_id", "hotel", "flight", "destination_id", "dates", "checkin_date", "checkout_date",
                    "nights", "travelers_count", "rooms_count", "price_per_night_rub"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after["logistics"]["flight"], before["logistics"]["flight"])
        self.assertEqual(after["logistics"]["transfer"]["type"], "self")
        self.assertEqual(after["logistics"]["transfer"]["price_rub"], 0)
        self.assertIsNone(after["logistics"]["timeline"]["hotel_arrival"])
        self.assertTrue(result["trip_conditions"]["without_transfer"])
        self.assertEqual(after["total_mandatory_rub"], before["total_mandatory_rub"] - 2000)
        for bound in ("min", "max"):
            self.assertEqual(after["total_trip_budget_range_rub"][bound], before["total_trip_budget_range_rub"][bound] - 2000)
        self.assertEqual(result["budget_breakdown"], after["budget_breakdown"])
        self.assertEqual(result["logistics"], after["logistics"])
        validate_package_totals(after["budget_breakdown"])

    async def test_transfer_removal_is_idempotent_and_can_restore_original_quote(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            first = await AgentService.process_message("web:transfer", "Пересчитай на 5 ночей")
            before = copy.deepcopy(first["cart"])
            removed = await AgentService.process_message("web:transfer", "Убери трансфер")
            again = await AgentService.process_message("web:transfer", "Не нужен трансфер")
            self.assertEqual(again["cart"]["total_mandatory_rub"], removed["cart"]["total_mandatory_rub"])
            self.assertEqual(again["cart"]["total_trip_budget_range_rub"], removed["cart"]["total_trip_budget_range_rub"])
            restored = await AgentService.process_message("web:transfer", "Верни трансфер")
        self.assertEqual(restored["state"], "TRANSFER_INCLUDED")
        self.assertFalse(restored["trip_conditions"]["without_transfer"])
        self.assertEqual(restored["cart"]["total_trip_budget_range_rub"], before["total_trip_budget_range_rub"])
        self.assertEqual(restored["logistics"], before["logistics"])
        validate_package_totals(restored["budget_breakdown"])

    async def test_transfer_removal_supports_already_saved_carts_without_full_ledger(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            first = await AgentService.process_message("web:transfer", "Пересчитай на 5 ночей")
            user.current_cart.pop("budget_breakdown", None)
            self.search.reset_mock()
            self.compare.reset_mock()
            result = await AgentService.process_message("web:transfer", "Трансфер нам не нужен")
            self.search.assert_not_awaited()
            self.compare.assert_not_awaited()
        self.assertEqual(result["state"], "TRANSFER_REMOVED")
        self.assertEqual(result["cart"]["total_mandatory_rub"], first["cart"]["total_mandatory_rub"] - 2000)
        validate_package_totals(result["budget_breakdown"])

    async def test_transfer_restoration_respects_budget_and_preserves_current_plan(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            await AgentService.process_message("web:transfer", "Пересчитай на 5 ночей")
            removed = await AgentService.process_message("web:transfer", "Без трансфера")
            user.profile["budget_rub"] = removed["cart"]["total_trip_budget_range_rub"]["max"]
            result = await AgentService.process_message("web:transfer", "Добавь трансфер")
        self.assertEqual(result["state"], "TRANSFER_OVER_BUDGET")
        self.assertEqual(result["cart"], removed["cart"])
        self.assertTrue(user.profile["without_transfer"])

    async def test_compound_transfer_and_duration_edit_reprices_new_duration(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            await AgentService.process_message("web:transfer", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:transfer", "Трансфер не нужен, и на 8 ночей")
        self.assertEqual(result["cart"]["nights"], 8)
        self.assertTrue(user.profile["without_transfer"])
        self.assertEqual(result["budget_breakdown"]["mandatory"]["transfer"], 0)
        self.assertNotIn("с перелётом, отелем и трансфером", result["text"])

    async def test_question_about_eight_nights_recalculates_without_free_form_reply(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "А на 8 ночей?")
            self.dispatch.assert_not_awaited()
        self.assertEqual(result["cart"]["destination_id"], "spb")
        self.assertEqual(result["cart"]["nights"], 8)
        self.assertEqual(result["cart"]["checkout_date"], "2027-10-23")
        self.assertIn("8 ночей", result["text"])
        self.assertNotIn("Старый ответ", result["text"])

    async def test_cheapest_preserves_budget_dates_and_direct_flight(self):
        user = make_user(prefer_direct_flight=True)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "Сделай подешевле в любое время?")
            self.dispatch.assert_not_awaited()
        self.assertEqual(user.profile["budget_rub"], 85000)
        self.assertEqual(result["trip_conditions"]["budget_rub"], 85000)
        self.assertTrue(user.profile["prefer_direct_flight"])

    async def test_cheaper_without_date_permission_keeps_dates(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "Сделай самый дешевый вариант")
        self.assertEqual(result["cart"]["checkin_date"], "2027-10-15")
        self.assertEqual(result["cart"]["checkout_date"], "2027-10-26")

    async def test_lower_bound_uses_same_quote_without_search_or_llm(self):
        user = make_user(nights=1, adults_count=1, rooms_count=1, budget_rub=45000)
        with self.fixture(user, hotel_rate=10000):
            first = await AgentService.process_message("web:dialogue", "Пересчитай на 1 ночь")
            self.assertEqual(first["state"], "BUDGET_EXCEEDED")
            bounds = copy.deepcopy(first["alternative_cart"]["total_trip_budget_range_rub"])
            self.assertEqual(bounds, {"min": 40300, "max": 46300})
            self.search.reset_mock()
            self.compare.reset_mock()
            second = await AgentService.process_message("web:dialogue", "А по нижней оценке?")
            self.search.assert_not_awaited()
            self.compare.assert_not_awaited()
            self.dispatch.assert_not_awaited()
        self.assertEqual(second["state"], "BUDGET_ESTIMATE")
        self.assertEqual(second["trip_state"]["search_results"]["alternative_cart"]["total_trip_budget_range_rub"], bounds)
        self.assertIn("40,300", second["text"])
        self.assertIn("Остаток", second["text"])
        self.assertEqual(user.profile["budget_rub"], 45000)

    async def test_original_screenshot_chain_keeps_destination_and_updated_quote(self):
        user = make_user()
        with self.fixture(user):
            first = await AgentService.process_message("web:dialogue", "Сделай подешевле")
            second = await AgentService.process_message("web:dialogue", "А на 8 ночей?")
            third = await AgentService.process_message("web:dialogue", "А по нижней оценке?")
        self.assertEqual(first["cart"]["destination_id"], "spb")
        self.assertEqual(second["cart"]["destination_id"], "spb")
        self.assertEqual(third["cart"]["destination_id"], "spb")
        self.assertEqual(third["trip_state"]["search_results"]["alternative_cart"]["budget_quote"]["nights"], 8)
        self.assertEqual(
            third["trip_state"]["search_results"]["alternative_cart"]["total_trip_budget_range_rub"],
            second["alternative_cart"]["total_trip_budget_range_rub"],
        )

    async def test_destination_question_uses_saved_facts(self):
        user = make_user()
        before = copy.deepcopy(user.current_cart)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "Почему Сочи, ты сам писал про Питер?")
            self.dispatch.assert_not_awaited()
            self.search.assert_not_awaited()
        self.assertEqual(result["state"], "CURRENT_TRIP")
        self.assertEqual(result["cart"], before)
        self.assertIn("Санкт-Петербург", result["text"])
        self.assertNotIn("вашей новой просьбы", result["text"])

    async def test_party_edit_does_not_change_destination_on_perfect_vector_match(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "Нас 2 взрослых")
        self.assertEqual(result["cart"]["destination_id"], "spb")
        self.assertEqual(result["trip_conditions"]["adults_count"], 2)

    async def test_parsed_pet_edit_cannot_become_an_unpriced_consultation(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "А ещё с нами будет собака")
            self.dispatch.assert_not_awaited()
            self.assertTrue(self.compare.call_args.kwargs["pet_friendly"])
        self.assertTrue(result["trip_conditions"]["has_pets"])
        self.assertEqual(result["cart"]["destination_id"], "spb")

    async def test_unclear_budget_preserves_selected_offer_and_saves_other_conditions(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            selected = await AgentService.process_message("web:dialogue", "Пересчитай на 11 ночей")
            result = await AgentService.process_message("web:dialogue", "А на 8 ночей с бюджетом сто тысяч?")
        self.assertEqual(result["state"], "CLARIFICATION_REQUIRED")
        self.assertEqual(result["trip_state"]["dates"]["nights"], 8)
        self.assertEqual(result["cart"], selected["cart"])
        self.assertIn("hotel", result["cart"])

    async def test_explicit_destination_change_still_works(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "Вместо Питера поехали в Сочи на 8 ночей")
        self.assertEqual(result["cart"]["destination_id"], "sochi")

    async def test_short_city_question_changes_destination_without_free_form_reply(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            result = await AgentService.process_message("web:screenshot", "А Сочи?")
            self.dispatch.assert_not_awaited()
        self.assertEqual(result["cart"]["destination_id"], "sochi")
        self.assertEqual(result["trip_conditions"]["adults_count"], 3)

    async def test_city_in_advice_question_does_not_change_trip(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user), patch.object(DialogueParser, "parse", new=AsyncMock(
                return_value=DialogueDecision(intent=DialogueIntent.QUESTION, answer="Старый ответ без пересчёта"))) as parse:
            result = await AgentService.process_message("web:screenshot", "Какая погода в Сочи?")
            parse.assert_awaited_once()
        self.assertEqual(result["state"], "CONSULTATION")
        self.assertEqual(result["cart"]["destination_id"], "spb")

    async def test_hotel_filter_uses_final_calendar_adjusted_price(self):
        user = make_user(nights=1, adults_count=1, rooms_count=1, budget_rub=45000)
        with self.fixture(user, hotel_rate=10000):
            result = await AgentService.process_message("web:dialogue", "Пересчитай на 1 ночь")
        self.assertEqual(result["state"], "BUDGET_EXCEEDED")
        self.assertNotIn("hotel", result["cart"])
        self.assertEqual(result["alternative_cart"]["total_trip_budget_range_rub"]["max"], 46300)
        self.assertEqual(len(result["compared_hotels"]), 1)

    async def test_exact_final_ceiling_is_accepted(self):
        user = make_user(nights=1, adults_count=1, rooms_count=1, budget_rub=46300)
        with self.fixture(user, hotel_rate=10000):
            result = await AgentService.process_message("web:dialogue", "Пересчитай на 1 ночь")
        self.assertEqual(result["cart"]["hotel_id"], "hotel-1")
        self.assertEqual(result["cart"]["total_trip_budget_range_rub"]["max"], 46300)
        self.assertEqual(result["budget_breakdown"]["mandatory"]["flight_roundtrip"], 23300)
        self.assertEqual(result["cart"]["price_per_night_rub"], 11500)
        self.assertEqual(result["compared_hotels"][0]["price_per_night_rub"], 11500)
        self.assertEqual(result["cart"]["total_trip_budget_range_rub"], result["budget_breakdown"]["total_trip_budget_range_rub"])

    async def test_calendar_comparison_reuses_original_prices(self):
        user = make_user(nights=1, adults_count=1, rooms_count=1, budget_rub=1000000)
        with self.fixture(user, hotel_rate=10000):
            first = await AgentService.process_message("web:dialogue", "Пересчитай на 1 ночь")
            second = await AgentService.process_message("web:dialogue", "Сравни даты")
        self.assertEqual(second["state"], "DATES_COMPARISON")
        self.assertEqual(first["cart"]["total_mandatory_rub"], 36800)
        self.assertIn("36,800", second["text"])

    async def test_stale_llm_hints_do_not_survive_next_turn(self):
        user = make_user(budget_rub=1000000, llm_destination_id="sochi", llm_ai_intro="Сочи на 11 ночей",
                         llm_replan_action="change_destination")
        with self.fixture(user):
            result = await AgentService.process_message("web:dialogue", "А на 8 ночей?")
        self.assertEqual(result["cart"]["destination_id"], "spb")
        self.assertNotIn("Сочи на 11 ночей", result["text"])

    async def test_zero_transfer_price_is_not_replaced_by_default(self):
        user = make_user(nights=1, adults_count=1, rooms_count=1, budget_rub=1000000)
        with self.fixture(user, hotel_rate=10000, transfer_price=0):
            result = await AgentService.process_message("web:dialogue", "Пересчитай на 1 ночь")
        self.assertEqual(result["budget_breakdown"]["mandatory"]["transfer"], 0)
        self.assertEqual(result["cart"]["total_mandatory_rub"], 34800)
        self.assertEqual(result["cart"]["pricing_inputs"]["fixed_costs_rub"], 0)

    async def test_repeated_cheaper_request_explains_no_savings(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            first = await AgentService.process_message("web:dialogue", "Сделай подешевле")
            second = await AgentService.process_message("web:dialogue", "Сделай подешевле")
        self.assertEqual(first["cart"]["total_trip_budget_range_rub"], second["cart"]["total_trip_budget_range_rub"])
        self.assertIn("Более дешёвого варианта среди найденных нет", second["text"])

    @staticmethod
    def named_hotels():
        return [{"id": identifier, "name": name, "city": "Сочи", "stars": "3★",
                 "price_per_night_rub": price, "price_per_night_formatted": f"{price:,} ₽",
                 "meals": "Завтрак", "distance_to_sea": "150 м", "cancellation": "Бесплатно"}
                for identifier, name, price in (("golden", "Golden Dolphin Hotel", 3500),
                                               ("burgas", "Pansionat Burgas", 3510),
                                               ("medovaya", "Medovaya Hotel", 3860))]

    async def test_single_sochi_trip_uses_actual_destination_in_every_surface(self):
        rows = [{"id": "sochi", "name": "Сочи и Красная Поляна", "score": 1.0}]
        for city in ("Сочи", "Адлер", "Сириус", "Сочи (Центр)"):
            with self.subTest(city=city):
                user = make_user(budget_rub=1000000)
                hotel = {**self.named_hotels()[0], "city": city}
                with self.fixture(user, destination_rows=rows, destination_hotels={"sochi": [hotel]}):
                    result = await AgentService.process_message("web:route-name", "Хочу в Сочи на 5 ночей")
                self.assertEqual(result["destination"]["name"], "Сочи")
                self.assertEqual(result["cart"]["destination"], "Сочи")
                self.assertEqual(result["cart"]["hotel_city"], city)
                self.assertEqual(result["cart"]["budget_quote"]["destination"], "Сочи")
                self.assertNotIn("Сочи и Красная Поляна", result["text"])
                self.assertIn("🌴 **Сочи**", result["text"])
                self.assertEqual(user.current_cart["destination"], "Сочи")
        self.assertEqual(rows[0]["name"], "Сочи и Красная Поляна")

    async def test_single_mountain_hotel_does_not_become_a_coastal_or_combined_trip(self):
        user = make_user(budget_rub=1000000)
        rows = [{"id": "sochi", "name": "Сочи и Красная Поляна", "score": 1.0}]
        hotel = {**self.named_hotels()[0], "city": "Красная Поляна"}
        with self.fixture(user, destination_rows=rows, destination_hotels={"sochi": [hotel]}):
            result = await AgentService.process_message("web:route-name", "Хочу в Красную Поляну на 5 ночей")
        self.assertEqual(result["destination"]["name"], "Красная Поляна")
        self.assertEqual(result["cart"]["destination"], "Красная Поляна")
        self.assertIn("🌴 **Красная Поляна**", result["text"])

    async def test_old_sochi_cart_is_corrected_before_price_question_without_replanning(self):
        user = make_user(budget_rub=1000000)
        rows = [{"id": "sochi", "name": "Сочи и Красная Поляна", "score": 1.0}]
        with self.fixture(user, destination_rows=rows, destination_hotels={"sochi": self.named_hotels()}):
            await AgentService.process_message("web:route-name", "Хочу в Сочи на 5 ночей")
            user.current_cart["destination"] = "Сочи и Красная Поляна"
            user.current_cart["budget_quote"]["destination"] = "Сочи и Красная Поляна"
            user.current_cart.pop("hotel_city", None)
            before_total = user.current_cart["total_mandatory_rub"]
            self.search.reset_mock()
            self.compare.reset_mock()
            db = AsyncMock()
            db.__aenter__.return_value = db
            db.get.return_value = SimpleNamespace(city="Сириус")
            with patch("app.services.agent_service.async_session_maker", return_value=db):
                result = await AgentService.process_message("web:route-name", "А по нижней оценке?")
                db.get.assert_awaited_once()
            self.search.assert_not_awaited()
            self.compare.assert_not_awaited()
            self.dispatch.assert_not_awaited()
        self.assertEqual(result["cart"]["destination"], "Сочи")
        self.assertNotIn("Красная Поляна", result["text"])
        self.assertEqual(result["cart"]["budget_quote"]["destination"], "Сочи")
        self.assertEqual(result["cart"]["total_mandatory_rub"], before_total)

    async def test_pet_filters_and_surcharge_work_without_audit_boilerplate(self):
        user = make_user(budget_rub=1000000)
        hotel = {**self.named_hotels()[0], "pet_friendly": True, "pet_surcharge_per_night_rub": 1500}
        with self.fixture(user, destination_hotels={"spb": [hotel]}):
            with_pet = await AgentService.process_message("web:audit", "На 5 ночей, с собакой")
            self.assertTrue(self.compare.call_args.kwargs["pet_friendly"])
            self.assertGreater(with_pet["budget_breakdown"]["mandatory"]["pet_total"], 0)
            without_pet = await AgentService.process_message("web:audit", "Без питомцев")
            self.assertFalse(self.compare.call_args.kwargs["pet_friendly"])
        self.assertFalse(without_pet["trip_conditions"]["has_pets"])
        self.assertEqual(without_pet["budget_breakdown"]["mandatory"]["pet_total"], 0)
        for result in (with_pet, without_pet):
            for unwanted in ("Аудит агента", "Smart Filter", "Pet-контроль", "подтверждено безопасное размещение"):
                self.assertNotIn(unwanted, result["text"])

    async def test_model_intro_cannot_restore_combined_title_or_audit(self):
        rows = [{"id": "sochi", "name": "Сочи и Красная Поляна", "score": 1.0}]
        for intro in ("Подобрал тур в Сочи и Красную Поляну", "Аудит агента: подтверждено безопасное размещение"):
            with self.subTest(intro=intro):
                user = make_user(budget_rub=1000000)
                user.current_cart = {"destination_id": "sochi", "destination": "Сочи"}
                with self.fixture(user, destination_rows=rows, destination_hotels={"sochi": self.named_hotels()}), patch.object(
                        DialogueParser, "parse", new=AsyncMock(return_value=DialogueDecision(
                            intent=DialogueIntent.PLAN, narrative=intro))) as parse:
                    result = await AgentService.process_message("web:route-name", "Покажи подходящий вариант")
                    parse.assert_awaited_once()
                self.assertNotIn(intro, result["text"])
                self.assertNotIn("Аудит агента", result["text"])
                self.assertIn("🌴 **Сочи**", result["text"])

    async def test_screenshot_flights_show_per_person_times_three_without_double_multiplication(self):
        user = make_user(nights=5, checkin_date="2026-11-12", checkout_date="2026-11-17", budget_rub=300000)
        with self.fixture(user, hotel_rate=3500, flight_price=13800):
            result = await AgentService.process_message("web:screenshot", "Едем втроём на 5 ночей, нужно 2 номера")
        mandatory = result["budget_breakdown"]["mandatory"]
        self.assertEqual(mandatory["travelers_count"], 3)
        self.assertEqual(mandatory["flight_out_price"], 19800)
        self.assertEqual(mandatory["flight_ret_price"], 18300)
        self.assertIn("6,600 ₽ × 3", result["text"])
        self.assertIn("6,100 ₽ × 3", result["text"])
        self.assertIn("3 пассажира", result["text"])

    async def test_unknown_destination_searches_all_allowed_regions_before_exceeding_budget(self):
        user = make_user(nights=5, budget_rub=300000)
        user.current_cart = {}
        rows = [{"id": identifier, "name": name, "score": score}
                for identifier, name, score in (("maldives", "Мальдивы", 1.0), ("turkey", "Турция", .9), ("sochi", "Сочи", .8))]
        hotels = self.named_hotels()
        overpriced = {**hotels[0], "id": "costly", "price_per_night_rub": 30000}
        with self.fixture(user, destination_rows=rows,
                          destination_hotels={"maldives": [overpriced], "turkey": [overpriced], "sochi": hotels}):
            result = await AgentService.process_message("web:screenshot", "По направлению пока не знаю, едем втроём на 5 ночей, нужно будет 2 номера, бюджет 300.000")
        self.assertEqual(result["cart"]["destination_id"], "sochi")
        self.assertLessEqual(result["budget_breakdown"]["total_trip_budget_range_rub"]["max"], 300000)
        self.assertEqual(result["trip_conditions"]["adults_count"], 3)

    async def test_budget_and_other_destinations_preserve_all_trip_conditions(self):
        user = make_user(nights=5, budget_rub=120000)
        selected = copy.deepcopy(user.current_cart)
        rows = [{"id": "spb", "name": "Санкт-Петербург", "score": 1.0},
                {"id": "sochi", "name": "Сочи", "score": .8}]
        affordable = {**self.named_hotels()[0], "price_per_night_rub": 0}
        with self.fixture(user, destination_rows=rows, destination_hotels={"sochi": [affordable]}, flight_price=1000):
            result = await AgentService.process_message(
                "web:budget-directions", "У меня бюджет 180 000, может есть другие направления?"
            )
        self.assertEqual(result["trip_state"]["budget_rub"], 180000)
        self.assertEqual(result["trip_state"]["dates"], {
            "checkin_date": "2027-10-15", "checkout_date": "2027-10-26", "nights": 5,
        })
        self.assertEqual(result["search_request"]["scope"], "other_destinations")
        self.assertIsNone(result["search_request"]["price_preference"])
        self.assertNotIn("budget_friendly", result["trip_state"]["preferences"]["interests"])
        self.assertEqual(result["cart"], selected)
        self.assertEqual(result["alternative_cart"]["destination_id"], "sochi")
        self.assertLessEqual(result["alternative_cart"]["total_trip_budget_range_rub"]["max"], 180000)
        self.assertIn("даты", result["preserved_conditions"])
        self.assertNotIn("Сохранены: даты", result["text"])

    async def test_rejected_region_is_excluded_from_budget_search(self):
        user = make_user(nights=5, budget_rub=300000, destination_flexible=True)
        user.current_cart.update(destination_id="maldives", destination="Мальдивы")
        rows = [{"id": "maldives", "name": "Мальдивы", "score": 1.0},
                {"id": "sochi", "name": "Сочи", "score": .8}]
        with self.fixture(user, destination_rows=rows, destination_hotels={"sochi": self.named_hotels()}):
            result = await AgentService.process_message("web:screenshot", "А если не Мальдивы?")
        self.assertEqual(result["cart"]["destination_id"], "maldives")
        self.assertEqual(result["alternative_cart"]["destination_id"], "sochi")
        self.assertIn("maldives", user.profile["excluded_destinations"])

    async def test_named_hotel_is_selected_and_retained_until_cheaper_request(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user, destination_hotels={"spb": self.named_hotels()}):
            first = await AgentService.process_message("web:screenshot", "Пересчитай на 5 ночей")
            self.assertEqual(first["cart"]["hotel_id"], "golden")
            selected = await AgentService.process_message("web:screenshot", "Выбери Medovaya Hotel")
            self.assertEqual(selected["cart"]["hotel_id"], "medovaya")
            updated = await AgentService.process_message("web:screenshot", "А на 8 ночей?")
            self.assertEqual(updated["cart"]["hotel_id"], "medovaya")
            cheaper = await AgentService.process_message("web:screenshot", "Как уменьшить общую стоимость поездки?")
            self.dispatch.assert_not_awaited()
        self.assertEqual(cheaper["cart"]["hotel_id"], "golden")
        self.assertLess(cheaper["cart"]["total_trip_budget_range_rub"]["max"], updated["cart"]["total_trip_budget_range_rub"]["max"])
        self.assertEqual(cheaper["cart"]["nights"], 8)
        self.assertEqual(cheaper["trip_conditions"]["adults_count"], 3)
        self.assertEqual(cheaper["trip_conditions"]["rooms_count"], 2)

    async def test_other_accommodation_under_my_budget_reuses_limit_and_changes_hotel(self):
        user = make_user(nights=5, budget_rub=300000, prefer_direct_flight=True, without_transfer=True,
                         private_room_required=True)
        with self.fixture(user, destination_hotels={"spb": self.named_hotels()}):
            await AgentService.process_message("web:budget-reference", "Пересчитай на 5 ночей")
            selected = await AgentService.process_message("web:budget-reference", "Выбери Medovaya Hotel")
            before = copy.deepcopy(selected["cart"])
            user.profile["awaiting_budget"] = True
            self.dispatch.reset_mock()
            result = await AgentService.process_message("web:budget-reference", "Предложи другой вариант проживания под мой бюджет")
            self.dispatch.assert_not_awaited()
            self.assertIn("medovaya", self.compare.call_args.kwargs["exclude_hotel_ids"])
        self.assertNotEqual(result.get("state"), "NEEDS_BUDGET")
        self.assertNotEqual(result["cart"]["hotel_id"], before["hotel_id"])
        self.assertEqual(result["cart"]["budget_rub"], 300000)
        self.assertEqual(result["trip_conditions"]["budget_rub"], 300000)
        self.assertLessEqual(result["cart"]["total_trip_budget_range_rub"]["max"], 300000)
        for key in ("nights", "dates", "checkin_date", "checkout_date", "travelers_count", "rooms_count"):
            self.assertEqual(result["cart"][key], before[key], key)
        self.assertTrue(user.profile["prefer_direct_flight"])
        self.assertTrue(user.profile["without_transfer"])
        self.assertTrue(user.profile["private_room_required"])
        self.assertEqual(result["budget_breakdown"]["mandatory"]["transfer"], 0)
        self.assertNotIn("awaiting_budget", user.profile)

    async def test_other_accommodation_does_not_offer_over_budget_hotel(self):
        user = make_user(nights=5, budget_rub=300000)
        hotels = self.named_hotels()[:1] + [{**self.named_hotels()[1], "price_per_night_rub": 30000}]
        with self.fixture(user, destination_hotels={"spb": hotels}):
            await AgentService.process_message("web:budget-reference", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:budget-reference", "Предложи другой вариант проживания под мой бюджет")
        self.assertEqual(result["state"], "BUDGET_EXCEEDED")
        self.assertEqual(result["cart"]["hotel_id"], "golden")
        self.assertEqual(result["alternative_cart"]["hotel_id"], "burgas")
        self.assertTrue(result["alternative_cart"]["over_budget_alternative"])
        self.assertEqual(result["trip_conditions"]["budget_rub"], 300000)
        self.assertEqual(user.profile["budget_rub"], 300000)

    async def test_layover_concern_preserves_quote_without_model_or_replanning(self):
        user = make_user(budget_rub=300000, nights=5)
        risk = "Критический риск: самостоятельная пересадка 75 минут, требуется повторная регистрация багажа."
        with self.fixture(user):
            await AgentService.process_message("web:layover", "Пересчитай на 5 ночей")
            user.current_cart["logistics"]["flight"].update(flight_type="connecting", layover_airport="MRV",
                layover_city="Минеральные Воды", layover_duration_minutes=75, layover_risk_note=risk)
            user.current_cart["logistics"]["timeline"]["layover_details"] = {
                "is_connecting": True, "city": "Минеральные Воды", "airport": "MRV", "duration_minutes": 75,
                "risk_evaluation": risk, "safety_guarantee": "Багаж сквозной, авиакомпания бесплатно пересадит."}
            user.current_cart["reasoning_steps"] = [{"stage": "logistics", "detail": "Рейс с безопасным транзитным буфером."}]
            before = copy.deepcopy(user.current_cart)
            profile = copy.deepcopy(user.profile)
            for mock in (self.dispatch, self.search, self.compare, self.logistics):
                mock.reset_mock()
            self.dispatch.return_value = {"action": "plan", "ai_reply_intro": "Пересчитал поездку"}
            for question in ("А там же очень мало времени на пересадку", "Почему такая короткая пересадка?",
                             "А хватит ли времени на пересадку с багажом?", "Сколько времени на пересадке?",
                             "Как устроена эта пересадка", "Хочу узнать, достаточно ли времени на пересадку"):
                with self.subTest(question=question):
                    result = await AgentService.process_message("web:layover", question)
                    self.assertEqual(result["state"], "CURRENT_LOGISTICS")
                    self.assertEqual(result["cart"], before)
                    self.assertEqual(user.profile, profile)
                    self.assertIn(risk, result["text"])
                    self.assertIn("75", result["text"])
                    self.assertNotIn("бесплатно пересадит", result["text"])
                    self.assertIn(risk, result["reasoning_steps"][0]["detail"])
                    self.assertNotIn("безопасным транзитным буфером", result["reasoning_steps"][0]["detail"])
                    for mock in (self.dispatch, self.search, self.compare, self.logistics):
                        mock.assert_not_awaited()

    async def test_explicit_direct_flight_request_still_replans(self):
        user = make_user(budget_rub=300000, nights=5)
        with self.fixture(user):
            await AgentService.process_message("web:layover", "Пересчитай на 5 ночей")
            self.logistics.reset_mock()
            result = await AgentService.process_message("web:layover", "Только прямой рейс, без пересадок")
            self.logistics.assert_awaited()
            self.assertTrue(self.logistics.call_args.kwargs["prefer_direct_flight"])
        self.assertNotEqual(result.get("state"), "CURRENT_LOGISTICS")

    async def test_connection_diagnostics_repeat_catalogue_risk_instead_of_safety_claim(self):
        user = make_user(budget_rub=300000, nights=5)
        risk = "Критический риск: отдельные билеты, короткая пересадка."
        with self.fixture(user):
            self.logistics.return_value["flight"].update(flight_type="connecting", layover_duration_minutes=75,
                                                        layover_airport="MRV", layover_risk_note=risk)
            self.logistics.return_value["timeline"]["layover_details"] = {
                "is_connecting": True, "airport": "MRV", "duration_minutes": 75, "risk_evaluation": risk}
            result = await AgentService.process_message("web:layover", "Пересчитай на 5 ночей")
        detail = next(step["detail"] for step in result["reasoning_steps"] if step["stage"] == "logistics")
        self.assertIn(risk, detail)
        self.assertNotIn("безопасный транзитный буфер", detail)

    async def test_missing_named_hotel_does_not_silently_select_something_else(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user, destination_hotels={"spb": self.named_hotels()}):
            first = await AgentService.process_message("web:screenshot", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:screenshot", "Выбери Missing Hotel")
        self.assertEqual(result["state"], "HOTEL_NOT_AVAILABLE")
        self.assertEqual(result["cart"]["hotel_id"], first["cart"]["hotel_id"])

    async def test_named_hotel_above_ceiling_does_not_replace_an_affordable_plan(self):
        user = make_user(budget_rub=300000)
        hotels = self.named_hotels()
        hotels.append({**hotels[0], "id": "luxury", "name": "Expensive Hotel", "price_per_night_rub": 100000})
        with self.fixture(user, destination_hotels={"spb": hotels}):
            first = await AgentService.process_message("web:screenshot", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:screenshot", "Выбери Expensive Hotel")
        self.assertEqual(result["state"], "HOTEL_OVER_BUDGET")
        self.assertEqual(result["cart"]["hotel_id"], first["cart"]["hotel_id"])
        self.assertEqual(user.profile["budget_rub"], 300000)

    async def test_city_in_hotel_name_does_not_change_destination(self):
        user = make_user(budget_rub=1000000)
        hotels = self.named_hotels() + [{**self.named_hotels()[0], "id": "city-name", "name": "Сочи Парк Hotel"}]
        with self.fixture(user, destination_hotels={"spb": hotels}):
            result = await AgentService.process_message("web:screenshot", "Выбери Сочи Парк Hotel")
        self.assertEqual(result["cart"]["destination_id"], "spb")
        self.assertEqual(result["cart"]["hotel_id"], "city-name")

    async def test_no_cheaper_option_returns_a_clear_short_reply(self):
        user = make_user(budget_rub=1000000)
        with self.fixture(user):
            await AgentService.process_message("web:screenshot", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:screenshot", "Как уменьшить общую стоимость поездки?")
        self.assertEqual(result["state"], "NO_CHEAPER_OPTION")
        self.assertNotIn("Программа и расчёт тура", result["text"])
        self.assertIn("Более дешёвого", result["text"])


class StatePersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_history_append_is_detected_by_sqlalchemy_and_keeps_old_messages(self):
        user = UserSessionModel(user_id="persisted", profile={}, current_cart={}, conversation_history=[])
        previous_history = [{"user": "Питер", "agent": "Санкт-Петербург"}]
        set_committed_value(user, "conversation_history", previous_history)
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.get.return_value = user
        with patch("app.services.agent_service.async_session_maker", return_value=db):
            await AgentService._save_session("persisted", {"nights": 8}, {}, "А на 8 ночей?", "Пересчитал")
        self.assertTrue(inspect(user).attrs.conversation_history.history.has_changes())
        self.assertEqual(len(user.conversation_history), 2)
        self.assertEqual(previous_history, [{"user": "Питер", "agent": "Санкт-Петербург"}])
        db.commit.assert_awaited_once()


class ExplicitSlotsTest(unittest.TestCase):
    def test_transfer_preference_word_order_and_negation(self):
        for text in ("Трансфер не нужен, оставь остальной план", "Не нужен трансфер", "Трансфер нам не нужен",
                     "Без трансфера", "Убери трансфер", "Поедем сами"):
            with self.subTest(text=text):
                self.assertIs(transfer_preference(text), True)
        for text in ("Не убирай трансфер", "Верни трансфер", "Добавь трансфер", "Трансфер нужен"):
            with self.subTest(text=text):
                self.assertIs(transfer_preference(text), False)
        self.assertIsNone(transfer_preference("Какие условия отмены трансфера?"))

    def test_generic_hotel_requests_are_not_mistaken_for_hotel_names(self):
        for message in ("Выбери спокойный отель в Сочи для двоих", "Выбери отель в Дубае рядом с метро",
                        "Выбери семейный отель в ОАЭ с детским бассейном", "Выбери загородный отель для двоих",
                        "Выбери тихий отель с бассейном и сауной на Алтае"):
            with self.subTest(message=message):
                self.assertIsNone(hotel_selection(message))
        self.assertEqual(hotel_selection('Выбери «Отель в облаках»'), "Отель в облаках")
        self.assertEqual(hotel_selection('Выбери отель «Отель в облаках»'), "Отель в облаках")
        self.assertEqual(destination_choice("Выбери тихий отель с бассейном на Алтае"), "altai")

    def test_price_validator_rejects_missing_or_double_counted_tickets(self):
        profile = dict(nights=5, adults_count=3, rooms_count=2, checkin_date="2026-11-12")
        hotel = {"price_per_night_rub": 3500, "meals": "Завтрак"}
        logistics = {"flight": {"price_rub_roundtrip": 13800}, "transfer": {"price_rub": 1440}}
        correct = AgentService.calculate_package_budget("sochi", hotel, logistics, profile)
        validate_package_totals(correct)
        for price in (6600, 59400):
            with self.subTest(price=price):
                broken = copy.deepcopy(correct)
                broken["mandatory"]["flight_out_price"] = price
                with self.assertRaises(ValueError):
                    validate_package_totals(broken)

    def test_price_validator_rejects_inconsistent_full_range(self):
        profile = dict(nights=5, adults_count=3, rooms_count=2, checkin_date="2026-11-12")
        hotel = {"price_per_night_rub": 3500, "meals": "Завтрак"}
        logistics = {"flight": {"price_rub_roundtrip": 13800}, "transfer": {"price_rub": 1440}}
        broken = AgentService.calculate_package_budget("sochi", hotel, logistics, profile)
        broken["total_trip_budget_range_rub"]["max"] -= 1
        with self.assertRaises(ValueError):
            validate_package_totals(broken)

    def test_origin_is_not_destination(self):
        self.assertEqual(requested_destination("Из Санкт-Петербурга в Сочи"), "sochi")
        self.assertIsNone(requested_destination("Вылет из Сочи"))
        self.assertEqual(requested_destination("Вместо Питера в Сочи"), "sochi")
        self.assertEqual(requested_destination("Хочу в Грузию"), "georgia")
        self.assertIsNone(requested_destination("Вылет из Санкт-Петербурга"))
        self.assertIsNone(requested_destination("Маленький бюджет"))
        self.assertIsNone(requested_destination("Сочини план"))
        self.assertIsNone(requested_destination("А если не на Мальдивы?"))
        self.assertEqual(rejected_destinations("А если не на Мальдивы?"), {"maldives"})

    def test_descriptive_words_do_not_change_party_or_pets(self):
        profile = {"adults_count": 2}
        AgentService.extract_interests_and_slots("один номер, песок на пляже", profile)
        self.assertEqual(profile["adults_count"], 2)
        self.assertFalse(profile.get("has_pets", False))

    def test_unknown_prices_are_not_reconstructed_from_percentages(self):
        self.assertEqual(AgentService.calculate_flexible_dates_options("spb", 70000, "2027-10-15", 8), [])
        reply = AgentService.answer_dates_comparison("сравни даты", {"nights": 8}, {"destination_id": "spb"})
        self.assertIn("нет исходных тарифов", reply)

    def test_explicit_rooms_survive_followup(self):
        profile = {"adults_count": 2}
        AgentService.extract_interests_and_slots("нужно 2 комнаты", profile)
        AgentService.extract_interests_and_slots("на этот раз на 8 ночей", profile)
        self.assertEqual(profile["rooms_count"], 2)

    def test_request_for_two_rooms_does_not_mean_two_dormitory_beds(self):
        profile = {}
        AgentService.extract_interests_and_slots("Едем втроём, нужно будет 2 номера", profile)
        self.assertEqual(profile["adults_count"], 3)
        self.assertEqual(profile["rooms_count"], 2)
        self.assertTrue(profile["private_room_required"])
        AgentService.extract_interests_and_slots("Сделай подешевле", profile)
        self.assertTrue(profile["private_room_required"])

    def test_numeric_date_range_sets_nights(self):
        profile = {"nights": 7}
        self.assertIsNone(update_trip_dates("с 15.10.2027 по 26.10.2027", profile))
        self.assertEqual(profile["nights"], 11)
        self.assertEqual(profile["checkout_date"], "2027-10-26")

    def test_conflicting_or_invalid_date_edits_are_atomic(self):
        for message in ("с 26.10.2027 по 15.10.2027", "с 15 по 32 октября 2027", "заезд 15.10.2027 на 0 ночей",
                        "с 15.10.2027 по 26.10.2027 на 8 ночей"):
            with self.subTest(message=message):
                profile = {"nights": 7, "checkin_date": "2027-11-10", "checkout_date": "2027-11-17"}
                before = copy.deepcopy(profile)
                self.assertIsNotNone(update_trip_dates(message, profile))
                self.assertEqual(profile, before)


if __name__ == "__main__":
    unittest.main()
