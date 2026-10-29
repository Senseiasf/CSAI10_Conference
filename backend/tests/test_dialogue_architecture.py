import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.domain.dialogue import ChangeOperation, DialogueDecision, DialogueIntent, StateChange
from app.domain.route import RouteGraph
from app.domain.trip import TripQuote, TripState
from app.services.agent_service import AgentService
from app.services.dialogue_parser import DialogueParser, DialogueParserError
from app.services.graph_repository import Neo4jGraphRepository
from app.services.llm_service import LLMService
from app.services.quote_service import compare_quotes, version_breakdown
from app.services.trip_controller import TripController
from app.services.trip_state_store import TripStateStore


def breakdown(hotel_total=20000, transfer=1000):
    flight_out = 5000
    flight_ret = 5000
    mandatory_total = flight_out + flight_ret + hotel_total + transfer
    return {
        "nights": 5,
        "checkin_date": "2027-01-10",
        "checkout_date": "2027-01-15",
        "mandatory": {
            "travelers_count": 1,
            "rooms_count": 1,
            "flight_out_price": flight_out,
            "flight_ret_price": flight_ret,
            "flight_out_per_person_rub": flight_out,
            "flight_ret_per_person_rub": flight_ret,
            "flight_roundtrip": 10000,
            "flight_price_roundtrip": 10000,
            "hotel_per_night": hotel_total // 5,
            "hotel_total": hotel_total,
            "transfer": transfer,
            "pet_total": 0,
            "total_mandatory_rub": mandatory_total,
        },
        "onsite_flexible": {"total_onsite_range_rub": {"min": 5000, "max": 7000}},
        "total_trip_budget_range_rub": {"min": mandatory_total + 5000, "max": mandatory_total + 7000},
        "pricing_inputs": {"hotel_per_room_night_rub": hotel_total // 5, "transfer_rub": transfer},
        "pricing_source": "test_catalog",
    }


def cart_with_quote(quote_breakdown):
    return {
        "destination_id": "sochi",
        "destination": "Сочи",
        "hotel_id": "hotel-1",
        "hotel": "Hotel 1",
        "hotel_city": "Сочи",
        "nights": 5,
        "checkin_date": "2027-01-10",
        "checkout_date": "2027-01-15",
        "budget_breakdown": quote_breakdown,
        "logistics": {
            "flight": {
                "flight_no": "T1",
                "airline": "Test Air",
                "origin_code": "SVO",
                "origin_name": "Москва",
                "departure_time": "10:00",
                "arrival_time": "12:00",
                "flight_type": "direct",
                "price_rub_roundtrip": 10000,
                "return_flight": {"flight_no": "T2", "origin_code": "AER", "departure_time": "15:00"},
            },
            "airport": "Сочи",
            "transfer": {"name": "Такси", "type": "taxi", "price_rub": quote_breakdown["mandatory"]["transfer"]},
        },
    }


class DialogueContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_transfer_refusal_has_exactly_one_operation(self):
        decision = await DialogueParser().parse("Трансфер не нужен, оставь остальной план", TripState())
        self.assertEqual(decision.intent, DialogueIntent.MODIFY)
        self.assertEqual(len(decision.changes), 1)
        self.assertEqual(decision.changes[0].field, "without_transfer")
        self.assertIs(decision.changes[0].value, True)

    async def test_question_contract_forbids_changes(self):
        with self.assertRaises(ValueError):
            DialogueDecision(
                intent=DialogueIntent.QUESTION,
                topic="connection_risk",
                changes=[StateChange(field="without_transfer", operation=ChangeOperation.SET, value=True)],
            )

    async def test_explicit_false_and_long_clarification_chain_preserve_other_fields(self):
        controller = TripController()
        state = TripState(budget_rub=100000, rooms_count=2)
        state.passengers.has_pets = True
        state.dates.nights = 5
        updated = controller.apply_changes(state, [
            StateChange(field="has_pets", operation=ChangeOperation.SET, value=False),
            StateChange(field="nights", operation=ChangeOperation.SET, value=8),
        ])
        self.assertFalse(updated.passengers.has_pets)
        self.assertEqual(updated.dates.nights, 8)
        self.assertEqual(updated.rooms_count, 2)
        self.assertEqual(updated.budget_rub, 100000)
        cleared = controller.apply_changes(updated, [StateChange(field="budget_rub", operation=ChangeOperation.CLEAR)])
        self.assertIsNone(cleared.budget_rub)
        self.assertEqual(cleared.dates.nights, 8)
        self.assertTrue(state.passengers.has_pets)

    async def test_question_and_api_error_do_not_mutate_trip_or_call_planning(self):
        for decision, error in (
            (DialogueDecision(intent=DialogueIntent.QUESTION, topic="current_trip"), None),
            (None, DialogueParserError("api", "provider unavailable")),
        ):
            with self.subTest(error=error):
                session = SimpleNamespace(
                    profile={"nights": 5, "budget_rub": 90000},
                    current_cart={"destination_id": "spb", "destination": "Санкт-Петербург"},
                    conversation_history=[],
                )
                before = copy.deepcopy((session.profile, session.current_cart))
                parser = AsyncMock()
                if error:
                    parser.parse.side_effect = error
                else:
                    parser.parse.return_value = decision
                controller = TripController(parser)
                planning = AsyncMock()
                saving = AsyncMock()
                result = await controller.handle("web:test", "вопрос", session, planning, saving)
                self.assertEqual((session.profile, session.current_cart), before)
                planning.assert_not_awaited()
                self.assertEqual(result["trip_state"]["destination_id"], "spb")
                if error:
                    self.assertEqual(result["diagnostics"]["llm"]["error"]["code"], "api")

    async def test_comparison_keeps_selected_cart_and_exposes_alternative_quote(self):
        old_breakdown = version_breakdown(breakdown())
        old_cart = cart_with_quote(old_breakdown)
        session = SimpleNamespace(profile={"nights": 5}, current_cart=old_cart, conversation_history=[])
        parser = AsyncMock()
        parser.parse.return_value = DialogueDecision(intent=DialogueIntent.COMPARISON)
        changed_cart = cart_with_quote(version_breakdown(breakdown(hotel_total=24000, transfer=500), old_breakdown["quote_id"]))
        planning = AsyncMock(return_value={"text": "Сравнение", "state": "COMPARISON", "cart": changed_cart})
        saving = AsyncMock()
        result = await TripController(parser).handle("web:compare", "Сравни", session, planning, saving)
        self.assertEqual(result["cart"], old_cart)
        self.assertEqual(session.current_cart, old_cart)
        self.assertNotEqual(result["alternative_quote"]["quote_id"], old_breakdown["quote_id"])
        self.assertEqual(result["quote_difference"]["totals"]["mandatory_delta_rub"], 3500)
        saving.assert_awaited_once()

    def test_hotel_reference_resolves_only_to_known_offer(self):
        parser = DialogueParser()
        decision = DialogueDecision(
            intent=DialogueIntent.MODIFY,
            changes=[StateChange(field="hotel_id", operation=ChangeOperation.SET, value="Medovaya")],
        )
        offers = [{"id": "hotel-1", "name": "Medovaya Hotel"}, {"id": "hotel-2", "name": "Other Hotel"}]
        resolved = parser._resolve_hotel_reference(decision, offers)
        self.assertEqual(resolved.changes[0].value, "hotel-1")

    async def test_live_parser_validates_structured_decision(self):
        tool_call = SimpleNamespace(function=SimpleNamespace(
            arguments='{"intent":"question","topic":"connection_risk","changes":[]}'))
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
        with (patch.object(LLMService, "get_client", return_value=client),
              patch("app.services.dialogue_parser.settings", SimpleNamespace(
                  is_mock_llm=False, OPENAI_MODEL="test-model", OPENAI_API_KEY="test-key"))):
            decision = await DialogueParser().parse("Хватит ли времени на пересадку?", TripState())
        self.assertEqual(decision.intent, DialogueIntent.QUESTION)
        self.assertEqual(decision.topic, "connection_risk")
        self.assertEqual(decision.changes, [])
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "test-model")


class RouteAndQuoteTest(unittest.TestCase):
    def test_mrv_connection_has_two_flights_and_unknown_baggage_status(self):
        cart = cart_with_quote(version_breakdown(breakdown()))
        flight = cart["logistics"]["flight"]
        flight.update({
            "flight_type": "connecting",
            "layover_airport": "MRV",
            "layover_city": "Минеральные Воды",
            "layover_duration_minutes": 75,
            "layover_risk_note": "Самостоятельная пересадка",
        })
        route = RouteGraph.from_legacy(cart, {})
        self.assertEqual(len(route.flights), 3)
        self.assertEqual(len([item for item in route.flights if item.direction == "outbound"]), 2)
        connection = route.connections[0]
        self.assertEqual(connection.airport.code, "MRV")
        self.assertEqual(connection.duration_minutes, 75)
        self.assertEqual(connection.baggage_status, "unknown")
        self.assertEqual(connection.status, "unknown")
        self.assertIsNone(route.flights[1].flight_no)

    def test_hotel_only_change_preserves_compatible_flight_ids(self):
        cart = cart_with_quote(version_breakdown(breakdown()))
        new_result = copy.deepcopy(cart["logistics"])
        new_result.update({"compatible": True, "hotel": "Hotel 2"})
        result = Neo4jGraphRepository().preserve_compatible_hotel_change(
            cart,
            {"id": "hotel-2", "name": "Hotel 2", "city": "Сочи"},
            {"nights": 5},
            new_result,
        )
        self.assertEqual(result["flight"], cart["logistics"]["flight"])
        self.assertIn("T1:outbound:1", result["preserved_segment_ids"])
        self.assertEqual(result["route"]["hotels"][0]["id"], "hotel-2")

    def test_quote_versioning_and_component_difference(self):
        old_breakdown = version_breakdown(breakdown())
        new_breakdown = version_breakdown(breakdown(hotel_total=24000, transfer=500), old_breakdown["quote_id"])
        self.assertNotEqual(old_breakdown["quote_id"], new_breakdown["quote_id"])
        self.assertEqual(new_breakdown["previous_quote_id"], old_breakdown["quote_id"])
        old_quote = TripQuote.from_breakdown(old_breakdown)
        old_quote.quote_id = old_breakdown["quote_id"]
        new_quote = TripQuote.from_breakdown(new_breakdown, old_quote.quote_id)
        new_quote.quote_id = new_breakdown["quote_id"]
        difference = compare_quotes(old_quote, new_quote)
        deltas = {item["component"]: item["delta_rub"] for item in difference["components"]}
        self.assertEqual(deltas["hotel_total"], 4000)
        self.assertEqual(deltas["transfer"], -500)
        self.assertEqual(difference["totals"]["mandatory_delta_rub"], 3500)
        self.assertEqual(
            difference["text"],
            "Проживание подорожало на 4 000 ₽, Трансфер подешевел на 500 ₽, итоговая разница — 3 500 ₽.",
        )

    def test_trip_state_store_round_trip(self):
        session = SimpleNamespace()
        state = TripState(destination_id="spb", destination_name="Санкт-Петербург")
        state.dates.checkin_date = "2027-01-10"
        state.dates.checkout_date = "2027-01-15"
        state.dates.nights = 5
        quote_breakdown = version_breakdown(breakdown())
        quote = TripStateStore.quote_from_cart(cart_with_quote(quote_breakdown))
        TripStateStore.project(session, state, quote)
        loaded_state, loaded_quote = TripStateStore.load(session)
        self.assertEqual(loaded_state, state)
        self.assertEqual(loaded_quote.quote_id, quote.quote_id)


class TourVersionPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_save_session_versions_selected_quote_but_not_read_only_repeat(self):
        initial_breakdown = version_breakdown(breakdown())
        user = SimpleNamespace(
            profile={"nights": 5, "checkin_date": "2027-01-10", "checkout_date": "2027-01-15"},
            current_cart={},
            trip_state=None,
            trip_quote=None,
            conversation_history=[],
        )
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.get.return_value = user
        with (patch("app.services.agent_service.async_session_maker", return_value=db),
              patch("app.services.graph_repository.graph_repository.save_tour_version", new_callable=AsyncMock) as save_version):
            cart = cart_with_quote(initial_breakdown)
            await AgentService._save_session("web:version", user.profile, cart, "подбери", "готово")
            self.assertEqual(save_version.await_count, 1)
            first_version = user.current_cart["tour_version_id"]
            self.assertIsNotNone(user.trip_state)
            self.assertEqual(user.trip_quote["quote_id"], initial_breakdown["quote_id"])

            await AgentService._save_session("web:version", user.profile, user.current_cart, "вопрос", "ответ")
            self.assertEqual(save_version.await_count, 1)

            changed = version_breakdown(breakdown(hotel_total=24000, transfer=500), initial_breakdown["quote_id"])
            changed_cart = cart_with_quote(changed)
            await AgentService._save_session("web:version", user.profile, changed_cart, "измени", "готово")
            self.assertEqual(save_version.await_count, 2)
            self.assertEqual(user.current_cart["previous_tour_version_id"], first_version)
            self.assertEqual(user.current_cart["quote_difference"]["totals"]["mandatory_delta_rub"], 3500)
            self.assertEqual(user.trip_quote["previous_quote_id"], initial_breakdown["quote_id"])


if __name__ == "__main__":
    unittest.main()
