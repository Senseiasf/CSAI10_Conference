"""Условия поиска и точное ожидание заселения для карточки поездки."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.agent_service import AgentService
from app.services.graph_service import GraphService


class TripConditionsTest(unittest.IsolatedAsyncioTestCase):
    async def test_conditions_follow_replanning_and_do_not_expose_service_fields(self):
        user = SimpleNamespace(profile={"has_pets": True, "budget_rub": 150000,
                                        "rejected_hotels": ["old"], "llm_ai_intro": "internal"},
                               current_cart={"hotel": "Old"}, conversation_history=[])

        async def replan(user_id, message, profile, cart, session_obj, progress_callback):
            profile.update(dates="10.11.2026 — 17.11.2026", budget_rub=120000,
                           adults_count=2, has_kids=True, has_pets=False,
                           private_room_required=True)
            return {"cart": {"hotel": "New"}, "text": "Готово"}

        with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)),
              patch.object(AgentService, "_process_message", side_effect=replan)):
            result = await AgentService.process_message("test-conditions", "Измени условия")
        conditions = result["trip_conditions"]
        self.assertEqual(conditions["dates"], "10.11.2026 — 17.11.2026")
        self.assertEqual(conditions["budget_rub"], 120000)
        self.assertEqual(conditions["adults_count"], 2)
        self.assertTrue(conditions["has_kids"])
        self.assertIsNone(conditions["kids_count"])  # Число детей не выдумывается.
        self.assertFalse(conditions["has_pets"])
        self.assertTrue(conditions["private_room_required"])
        self.assertNotIn("rejected_hotels", conditions)
        self.assertNotIn("llm_ai_intro", conditions)

    async def test_empty_selection_keeps_conditions_and_unknown_values(self):
        user = SimpleNamespace(profile={"has_pets": False, "budget_rub": 100000}, current_cart={})
        empty = {"state": "NO_MATCHES", "compared_hotels": [],
                 "cart": {"selection_status": "no_matches"}}
        with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)),
              patch.object(AgentService, "_process_message", new=AsyncMock(return_value=empty))):
            result = await AgentService.process_message("test-empty-conditions", "Другой отель")
        self.assertEqual(result["state"], "NO_MATCHES")
        self.assertEqual(result["trip_conditions"]["budget_rub"], 100000)
        self.assertFalse(result["trip_conditions"]["has_pets"])
        self.assertIsNone(result["trip_conditions"]["has_kids"])
        self.assertIsNone(result["trip_conditions"]["adults_count"])


class CheckinWaitTest(unittest.IsolatedAsyncioTestCase):
    async def logistics(self, arrival, transfer_minutes, flight_fields=None):
        row = {"airline": "Test Air", "flight_no": "TA-1", "origin_name": "Москва",
               "departure_time": "01:00", "flight_arrival_time": arrival,
               "flight_duration": 180, "flight_price_roundtrip": 20000,
               "dest_code": "AER", "dest_airport_name": "Сочи", "hotel_name": "Hotel",
               "checkin_time": "15:00", "transfer_name": "Taxi", "transfer_type": "taxi",
               "transfer_duration": transfer_minutes, "transfer_price": 1500}
        row.update(flight_fields or {})
        records = AsyncMock()
        records.data.return_value = [row]
        db = AsyncMock()
        db.__aenter__.return_value = db
        db.run.return_value = records
        driver = MagicMock()
        driver.session.return_value = db
        with patch("app.services.graph_service.get_neo4j_driver", return_value=driver):
            return await GraphService.validate_logistics("AER", "hotel")

    async def test_wait_uses_hotel_arrival_after_baggage_and_transfer(self):
        result = await self.logistics("05:00", 10)
        self.assertTrue(result["compatible"])
        self.assertEqual(result["timeline"]["hotel_arrival"], "06:00")
        self.assertEqual(result["timeline"]["checkin_wait_minutes"], 540)

    async def test_connection_does_not_invent_ticket_or_baggage_guarantees(self):
        note = "Отдельные билеты; получить багаж и заново зарегистрироваться."
        result = await self.logistics("17:20", 30, {"flight_type": "connecting", "layover_airport": "MRV",
            "layover_city": "Минеральные Воды", "layover_duration_minutes": 75, "layover_risk_note": note})
        connection = result["timeline"]["layover_details"]
        self.assertEqual(connection["risk_evaluation"], note)
        self.assertEqual(connection["duration_minutes"], 75)
        self.assertNotIn("safety_guarantee", connection)

    async def test_arrival_at_or_after_checkin_has_no_wait(self):
        for arrival in ("14:00", "17:00"):
            with self.subTest(arrival=arrival):
                result = await self.logistics(arrival, 10)
                self.assertEqual(result["timeline"]["checkin_wait_minutes"], 0)

    async def test_transfer_crossing_midnight_does_not_create_false_wait(self):
        result = await self.logistics("23:30", 60)
        self.assertEqual(result["timeline"]["hotel_arrival"], "01:20")
        self.assertEqual(result["timeline"]["hotel_arrival_day_offset"], 1)
        self.assertEqual(result["timeline"]["checkin_wait_minutes"], 0)
