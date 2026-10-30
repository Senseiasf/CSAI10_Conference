import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_trip_dialogue_regressions as dialogue_fixtures
from test_trip_dialogue_regressions import make_user

from app.domain.dialogue import DialogueDecision, DialogueIntent
from app.domain.trip import TripState
from app.services.agent_service import AgentService
from app.services.dialogue_parser import DialogueParser
from app.services.llm_service import LLMService
from app.services.trip_controller import TripController
from app.services.trip_state import matching_hotels


class DialogueLiveQualityRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_response_has_no_think_json_reasoning_or_change_prefix(self):
        dirty = (
            "  \n<think>Сначала проверю поля</think>\n"
            "applied_changes: [{\"field\": \"nights\"}]\n"
            "```json\n{\"change_explanation\": \"technical\"}\n```\n"
            "Готово! Обновил поездку на пять ночей."
        )
        cleaned = LLMService.clean_response_text(dirty)
        self.assertEqual(cleaned, "Готово! Обновил поездку на пять ночей.")

        session = SimpleNamespace(profile={}, current_cart={}, conversation_history=[], trip_state=None, trip_quote=None)
        parser = AsyncMock()
        parser.parse.return_value = DialogueDecision(
            intent=DialogueIntent.PLAN,
            narrative="С удовольствием обновил план — вот что получилось:",
        )
        result = await TripController(parser).handle(
            "web:clean", "Измени поездку", session,
            AsyncMock(return_value={"text": "Полный план", "cart": {}}),
            AsyncMock(),
        )
        self.assertTrue(result["text"].startswith("С удовольствием обновил план"))
        self.assertNotIn("Изменения:", result["text"])
        self.assertNotIn("applied_changes", result["text"])
        self.assertNotIn("<think>", result["text"])

    async def test_fuzzy_hotel_name_and_transliteration_resolve_to_id(self):
        hotels = [
            {"id": "burgas", "name": "Pansionat Burgas"},
            {"id": "apartments", "name": "Sochi Rent A Home Apartments"},
        ]
        self.assertEqual(matching_hotels("бургас", hotels)[0]["id"], "burgas")
        self.assertEqual(matching_hotels("апартаменты в сочи", hotels)[0]["id"], "apartments")
        decision = await DialogueParser().parse(
            "Выбери бургас",
            TripState(destination_id="sochi"),
            {"catalog_offers": hotels},
        )
        self.assertEqual(decision.intent, DialogueIntent.SELECT_OFFER)
        self.assertEqual(decision.changes[0].field, "hotel_id")
        self.assertEqual(decision.changes[0].value, "burgas")

    async def test_transfer_refusal_returns_complete_updated_plan(self):
        case = dialogue_fixtures.DialogueRegressionTest()
        user = make_user(budget_rub=1000000)
        with case.fixture(user):
            await AgentService.process_message("web:full-transfer-plan", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:full-transfer-plan", "Трансфер не нужен")
        self.assertEqual(result["state"], "TRANSFER_REMOVED")
        self.assertTrue(result["text"].startswith("Убрал трансфер из плана"))
        for fragment in ("Перелёт туда", "Проживание в отеле", "Перелёт обратно", "Расчёт стоимости всей поездки без трансфера"):
            self.assertIn(fragment, result["text"])
        self.assertIn("0 ₽", result["text"])
        self.assertNotIn("agent-reasoning", result["text"])

    async def test_week_later_shifts_both_dates_and_preserves_nights(self):
        case = dialogue_fixtures.DialogueRegressionTest()
        user = make_user(nights=5, checkin_date="2027-10-15", checkout_date="2027-10-20",
                         dates="15.10.2027 — 20.10.2027 (5 ночей)", budget_rub=1000000)
        user.current_cart.update(checkout_date="2027-10-20")
        with case.fixture(user):
            result = await AgentService.process_message("web:week-later", "Перенеси тур на неделю позже")
        self.assertEqual(result["trip_state"]["dates"], {
            "checkin_date": "2027-10-22",
            "checkout_date": "2027-10-27",
            "nights": 5,
        })
        changes = {change["field"]: change["value"] for change in result["dialogue_decision"]["changes"]}
        self.assertEqual(changes["date_shift_days"], 7)
        self.assertNotIn("nights", changes)

    async def test_sea_to_mountains_replaces_destination_and_uses_live_narrative(self):
        case = dialogue_fixtures.DialogueRegressionTest()
        user = make_user(interests=["beach"], budget_rub=1000000)
        user.current_cart.update(destination_id="sochi", destination="Сочи")
        rows = [{"id": "altai", "name": "Алтай", "score": 1.0}]
        hotels = {"altai": [{**case.named_hotels()[0], "city": "Манжерок"}]}
        with case.fixture(user, destination_rows=rows, destination_hotels=hotels):
            result = await AgentService.process_message("web:mountains", "А давай в горы")
        self.assertEqual(result["cart"]["destination_id"], "altai")
        self.assertEqual(result["trip_state"]["preferences"]["interests"], ["mountains"])
        self.assertTrue(result["text"].startswith("Отличная идея! Меняем морское побережье на горные вершины"))
        self.assertEqual(result["trip_state"]["dates"], {
            "checkin_date": "2027-10-15", "checkout_date": "2027-10-26", "nights": 11,
        })

    async def test_party_slang_becomes_nightlife_and_all_inclusive(self):
        decision = await DialogueParser().parse("Хочу убухаться и оторваться по полной с друзьями", TripState())
        interests = next(change.value for change in decision.changes if change.field == "interests")
        self.assertIn("nightlife", interests)
        self.assertIn("all_inclusive", interests)

    async def test_jail_request_never_becomes_destination(self):
        user = make_user()
        before = copy.deepcopy((user.profile, user.current_cart))
        with patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)), \
             patch.object(AgentService, "_save_session", new_callable=AsyncMock), \
             patch.object(AgentService, "search_destinations_vector", new=AsyncMock()) as search:
            result = await AgentService.process_message("web:invalid", "Хочу в СИЗО")
        self.assertEqual(result["state"], "INVALID_DESTINATION")
        self.assertEqual(result["dialogue_decision"]["intent"], "invalid_or_joke_request")
        self.assertIn("Туда мы туры, к счастью, не возим", result["text"])
        self.assertEqual((user.profile, user.current_cart), before)
        self.assertEqual(result["trip_state"]["destination_id"], "spb")
        search.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
