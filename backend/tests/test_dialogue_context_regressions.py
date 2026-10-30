import copy
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.domain.dialogue import ChangeOperation, DialogueDecision, DialogueIntent, StateChange
from app.main import app
from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService
from app.services.dialogue_parser import DialogueParser, DialogueParserError
from app.services.graph_repository import Neo4jGraphRepository
from app.services.graph_service import GraphService
from app.services.trip_controller import TripController
from test_trip_dialogue_regressions import make_user


class ControllerHarness:
    def make_session(self, profile=None, cart=None):
        return SimpleNamespace(
            profile=copy.deepcopy(profile or {}),
            current_cart=copy.deepcopy(cart or {}),
            conversation_history=[],
        )

    def save(self, session):
        async def save_session(_user_id, profile, cart, message, reply):
            session.profile = copy.deepcopy(profile)
            session.current_cart = copy.deepcopy(cart)
            session.conversation_history = session.conversation_history + [{"user": message, "agent": reply}]
        return AsyncMock(side_effect=save_session)

    def plan(self, session, calls):
        async def execute(_user_id, _message, profile, cart, _session, _progress_callback):
            calls.append((copy.deepcopy(profile), copy.deepcopy(cart)))
            return {
                "text": profile.get("llm_ai_intro") or "План проверен",
                "state": "PLAN_READY",
                "cart": cart,
            }
        return AsyncMock(side_effect=execute)


class DialogueContextTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.harness = ControllerHarness()

    async def test_inspiration_correction_mountains_and_pending_chain_keep_context(self):
        session = self.harness.make_session()
        calls = []
        controller = TripController()
        save = self.harness.save(session)
        plan = self.harness.plan(session, calls)
        results = []
        for message in (
            "Хочется к морю и солнцу",
            "Я же сказал, что к морю",
            "А давай в горы всё-таки",
            "на 5 ночей",
            "с 10.11.2027",
            "нас 3 взрослых, бюджет 180000",
            "Условия всё те же",
        ):
            results.append(await controller.handle("web:chain", message, session, plan, save))

        self.assertEqual(results[0]["state"], "CLARIFICATION_REQUIRED")
        self.assertIn("beach", results[0]["trip_state"]["preferences"]["interests"])
        self.assertNotIn("direction", results[0]["pending_action"]["missing_fields"])
        self.assertEqual(results[1]["dialogue_decision"]["intent"], "correction")
        self.assertEqual(results[2]["trip_state"]["preferences"]["interests"], ["mountains"])
        self.assertEqual(results[3]["trip_state"]["dates"]["nights"], 5)
        self.assertEqual(results[4]["trip_state"]["dates"]["checkin_date"], "2027-11-10")
        self.assertEqual(results[5]["trip_state"]["passengers"]["adults_count"], 3)
        self.assertEqual(results[5]["trip_state"]["budget_rub"], 180000)
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotIn("INSPIRATION_PROMPT", [result["state"] for result in results])
        self.assertNotIn("Здравствуйте", results[-1]["text"])

    async def test_nights_and_friends_are_applied_atomically_with_explanation(self):
        session = self.harness.make_session(
            profile={"checkin_date": "2027-11-10", "checkout_date": "2027-11-17", "nights": 7},
            cart={"destination_id": "sochi", "destination": "Сочи"},
        )
        calls = []
        result = await TripController().handle(
            "web:party",
            "Я еду на 5 ночей теперь и с двумя друзьями",
            session,
            self.harness.plan(session, calls),
            self.harness.save(session),
        )
        changes = {change["field"]: change["value"] for change in result["dialogue_decision"]["changes"]}
        self.assertEqual(changes["nights"], 5)
        self.assertEqual(changes["adults_count"], 3)
        self.assertEqual(result["trip_state"]["dates"]["nights"], 5)
        self.assertEqual(result["trip_state"]["passengers"]["adults_count"], 3)
        self.assertIn("длительность", result["text"])
        self.assertIn("число взрослых", result["text"])
        self.assertNotIn("Даты поездки успешно обновлены", result["text"])

    async def test_budget_and_other_destinations_are_separate_from_price_preference(self):
        selected = {"destination_id": "spb", "destination": "Санкт-Петербург", "hotel_id": "hotel-1", "hotel": "Hotel 1"}
        session = self.harness.make_session(
            profile={"nights": 5, "adults_count": 3, "budget_rub": 120000, "checkin_date": "2027-11-10"},
            cart=selected,
        )
        calls = []
        result = await TripController().handle(
            "web:alternatives",
            "У меня бюджет 180 000, может есть другие направления?",
            session,
            self.harness.plan(session, calls),
            self.harness.save(session),
        )
        self.assertEqual(result["dialogue_decision"]["intent"], "search_alternatives")
        self.assertEqual(result["trip_state"]["budget_rub"], 180000)
        self.assertEqual(result["search_request"]["scope"], "other_destinations")
        self.assertIsNone(result["search_request"]["price_preference"])
        self.assertNotIn("search", result["trip_state"])
        self.assertEqual(result["trip_state"]["dates"]["nights"], 5)
        self.assertEqual(result["cart"], selected)
        self.assertEqual(calls[0][0]["_structured_decision"]["search"]["scope"], "other_destinations")

    async def test_more_expensive_is_a_criterion_not_a_budget_change(self):
        session = self.harness.make_session(profile={"budget_rub": 180000})
        result = await TripController().handle(
            "web:expensive",
            "Я имею в виду дороже",
            session,
            self.harness.plan(session, []),
            self.harness.save(session),
        )
        self.assertEqual(result["trip_state"]["budget_rub"], 180000)
        self.assertEqual(result["search_request"]["price_preference"], "more_expensive")
        self.assertNotIn("search", result["trip_state"])

    async def test_reset_requires_explicit_new_trip(self):
        selected = {"destination_id": "sochi", "hotel_id": "hotel-1"}
        session = self.harness.make_session(cart=selected)
        controller = TripController()
        plan = self.harness.plan(session, [])
        save = self.harness.save(session)
        vague = await controller.handle("web:reset", "Хочется к морю и солнцу", session, plan, save)
        self.assertEqual(vague["cart"], selected)
        reset = await controller.handle("web:reset", "Начать заново, новая поездка", session, plan, save)
        self.assertEqual(reset["state"], "NEW_TRIP")
        self.assertEqual(reset["cart"], {})

    async def test_llm_failure_keeps_pending_action(self):
        pending = {"kind": "plan_trip", "missing_fields": ["budget_rub"], "question": "Бюджет?", "attempts": 0}
        session = self.harness.make_session(profile={"pending_action": pending, "nights": 5}, cart={"hotel_id": "hotel-1"})
        parser = AsyncMock()
        parser.parse.side_effect = DialogueParserError("api", "provider unavailable")
        result = await TripController(parser).handle(
            "web:error", "Продолжим", session, AsyncMock(), self.harness.save(session),
        )
        self.assertEqual(result["trip_state"]["pending_action"], pending)
        self.assertEqual(result["trip_state"]["dates"]["nights"], 5)
        self.assertEqual(result["diagnostics"]["llm"]["error"]["code"], "api")

    async def test_pet_change_does_not_clear_dates(self):
        session = self.harness.make_session(profile={"checkin_date": "2027-11-10", "checkout_date": "2027-11-15", "nights": 5})
        result = await TripController().handle(
            "web:pet", "Я еду с собакой", session, self.harness.plan(session, []), self.harness.save(session),
        )
        self.assertTrue(result["trip_state"]["passengers"]["has_pets"])
        self.assertEqual(result["trip_state"]["dates"]["checkin_date"], "2027-11-10")
        self.assertEqual(result["trip_state"]["dates"]["nights"], 5)

    async def test_natural_date_followup_completes_pending_trip(self):
        pending = {"kind": "plan_trip", "missing_fields": ["checkin_date"], "question": "Когда едете?", "attempts": 0}
        session = self.harness.make_session(
            profile={"destination_id": "sochi", "nights": 5, "adults_count": 2, "pending_action": pending},
            cart={"destination_id": "sochi", "destination": "Сочи"},
        )
        result = await TripController().handle(
            "web:natural-date", "1 ноября", session,
            self.harness.plan(session, []), self.harness.save(session),
        )
        expected = date(date.today().year, 11, 1)
        if expected < date.today():
            expected = expected.replace(year=expected.year + 1)
        self.assertEqual(result["trip_state"]["dates"]["checkin_date"], expected.isoformat())
        self.assertEqual(result["trip_state"]["dates"]["checkout_date"], (expected + timedelta(days=5)).isoformat())
        self.assertIsNone(result["trip_state"]["pending_action"])

    async def test_weekend_followup_sets_real_dates(self):
        session = self.harness.make_session(profile={"destination_id": "sochi", "adults_count": 2})
        result = await TripController().handle(
            "web:weekend", "ближайшие выходные", session,
            self.harness.plan(session, []), self.harness.save(session),
        )
        expected = date.today() + timedelta(days=(5 - date.today().weekday()) % 7)
        self.assertEqual(result["trip_state"]["dates"]["checkin_date"], expected.isoformat())
        self.assertEqual(result["trip_state"]["dates"]["checkout_date"], (expected + timedelta(days=2)).isoformat())
        self.assertEqual(result["trip_state"]["dates"]["nights"], 2)

    async def test_repeated_empty_clarification_changes_response_instead_of_looping(self):
        session = self.harness.make_session()
        controller = TripController()
        first = await controller.handle("web:loop", "Хочется к морю", session,
                                        self.harness.plan(session, []), self.harness.save(session))
        second = await controller.handle("web:loop", "продолжим", session,
                                         self.harness.plan(session, []), self.harness.save(session))
        third = await controller.handle("web:loop", "продолжим", session,
                                        self.harness.plan(session, []), self.harness.save(session))
        self.assertEqual(first["state"], "CLARIFICATION_REQUIRED")
        self.assertEqual(second["trip_state"]["pending_action"]["attempts"], 1)
        self.assertEqual(third["trip_state"]["pending_action"]["attempts"], 2)
        self.assertNotEqual(second["text"], third["text"])
        self.assertIn("Новых данных пока нет", third["text"])


class PresentationAndPlanningTest(unittest.IsolatedAsyncioTestCase):
    async def test_model_narrative_reaches_api_text(self):
        from test_trip_dialogue_regressions import DialogueRegressionTest
        case = DialogueRegressionTest()
        user = make_user(budget_rub=1000000)
        narrative = "Понял пожелание: сохраняю маршрут и обновляю только длительность."
        decision = DialogueDecision(
            intent=DialogueIntent.MODIFY,
            changes=[StateChange(field="nights", operation=ChangeOperation.SET, value=5)],
            narrative=narrative,
        )
        with case.fixture(user), patch.object(DialogueParser, "parse", new=AsyncMock(return_value=decision)):
            result = await AgentService.process_message("web:narrative", "Пересчитай на 5 ночей")
        self.assertIn(narrative, result["text"])

    async def test_sse_preserves_same_narrative(self):
        narrative = "Вступление модели не должно исчезнуть."
        with patch.object(AgentService, "process_message", new=AsyncMock(return_value={"text": narrative + "\nОсновной ответ", "cart": {}})):
            response = TestClient(app).post("/api/chat/stream", json={"message": "Продолжи"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(narrative, response.text)
        self.assertIn("Основной ответ", response.text)

    async def test_no_better_option_is_reported_without_replacing_current_tour(self):
        from test_trip_dialogue_regressions import DialogueRegressionTest
        case = DialogueRegressionTest()
        user = make_user(budget_rub=1000000)
        with case.fixture(user):
            first = await AgentService.process_message("web:expensive", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:expensive", "Я имею в виду дороже")
        self.assertEqual(result["state"], "NO_IMPROVEMENT_OPTION")
        self.assertEqual(result["cart"], first["cart"])
        self.assertIn("нет", result["text"])

    async def test_more_expensive_offer_explains_upgrade_and_preserves_limit(self):
        from test_trip_dialogue_regressions import DialogueRegressionTest
        case = DialogueRegressionTest()
        user = make_user(nights=5, budget_rub=300000)
        with case.fixture(user, destination_hotels={"spb": case.named_hotels()}):
            selected = await AgentService.process_message("web:upgrade", "Пересчитай на 5 ночей")
            result = await AgentService.process_message("web:upgrade", "Я имею в виду дороже")
        self.assertEqual(result["cart"], selected["cart"])
        self.assertEqual(result["alternative_cart"]["hotel_id"], "medovaya")
        self.assertEqual(result["alternative_cart"]["budget_rub"], 300000)
        self.assertLessEqual(result["alternative_cart"]["total_trip_budget_range_rub"]["max"], 300000)
        self.assertIn("За доплату", result["text"])
        self.assertIn("питание", result["text"])
        self.assertEqual(result["trip_state"]["dates"], selected["trip_state"]["dates"])

    async def test_onsite_expenses_and_flights_scale_with_travelers(self):
        kwargs = dict(
            destination_id="unknown",
            flight_price_roundtrip=10000,
            hotel_price_per_night=5000,
            transfer_price=1000,
            nights=5,
        )
        solo = MCPTools.calculate_budget_breakdown(**kwargs, travelers_count=1)
        group = MCPTools.calculate_budget_breakdown(**kwargs, travelers_count=3)
        self.assertEqual(group["mandatory"]["flight_roundtrip"], solo["mandatory"]["flight_roundtrip"] * 3)
        self.assertEqual(group["onsite_flexible"]["food_range_rub"]["min"], solo["onsite_flexible"]["food_range_rub"]["min"] * 3)
        self.assertEqual(group["onsite_flexible"]["entertainment_range_rub"]["min"], int(10000 / 7 * 5 * 3))
        self.assertEqual(group["onsite_flexible"]["travelers_count"], 3)

    async def test_transfer_capacity_is_checked_or_marked_unknown(self):
        repository = Neo4jGraphRepository()
        logistics = {
            "compatible": True,
            "flight": {"flight_no": "T1", "price_rub_roundtrip": 10000},
            "transfer": {"name": "Минивэн", "price_rub": 1000, "capacity": 2},
            "timeline": {},
        }
        with patch.object(GraphService, "validate_logistics", new=AsyncMock(return_value=copy.deepcopy(logistics))):
            result = await repository.check_compatibility("AER", "hotel", travelers_count=3)
        self.assertFalse(result["compatible"])
        self.assertIn("вмещает 2", result["reason"])
        logistics["transfer"].pop("capacity")
        with patch.object(GraphService, "validate_logistics", new=AsyncMock(return_value=copy.deepcopy(logistics))):
            result = await repository.check_compatibility("AER", "hotel", travelers_count=3)
        self.assertEqual(result["transfer"]["capacity_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
