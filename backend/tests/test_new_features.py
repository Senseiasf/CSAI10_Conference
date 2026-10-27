import unittest
import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.services.compliance_service import ComplianceService
from app.services.calendar_service import CalendarService
from app.services.agent_service import AgentService
from app.mcp.tools import MCPTools


class NewFeaturesTest(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_compliance_domestic_sochi(self):
        info = ComplianceService.get_compliance_info("sochi", {"has_kids": False})
        self.assertTrue(info["is_domestic"])
        self.assertIn("Внутренний паспорт", info["passport_required"])
        self.assertIn("Безвизовый", info["visa_policy"])
        self.assertIn("Правила въезда", info["summary"])

    def test_compliance_international_maldives_imuga(self):
        info = ComplianceService.get_compliance_info("maldives", {"has_kids": True})
        self.assertFalse(info["is_domestic"])
        self.assertIn("IMUGA", info["special_notes"])
        self.assertIn("декларация", info["summary"].lower())
        self.assertIn("загранпаспорт", info["passport_required"].lower())

    def test_compliance_uae_strict_validity(self):
        info = ComplianceService.get_compliance_info("uae", {"has_kids": False})
        self.assertIn("6 месяцев", info["passport_validity"])

    def test_calendar_service_rfc5545(self):
        cart = {
            "booking_id": "OST-TEST123",
            "destination": "Сочи (Красная Поляна)",
            "hotel": "Radisson Rosa Khutor",
            "nights": 5,
            "checkin_date": "2026-11-20",
            "checkout_date": "2026-11-25",
            "flight": "Аэрофлот SU-1124",
            "transfer": "Индивидуальный трансфер"
        }
        ics = CalendarService.generate_ics(cart)
        self.assertIn("BEGIN:VCALENDAR", ics)
        self.assertIn("VERSION:2.0", ics)
        self.assertIn("BEGIN:VEVENT", ics)
        self.assertIn("UID:flight-out-OST-TEST123@ostrovok.ai", ics)
        self.assertIn("UID:hotel-OST-TEST123@ostrovok.ai", ics)
        self.assertIn("UID:flight-ret-OST-TEST123@ostrovok.ai", ics)
        self.assertIn("END:VCALENDAR", ics)

    def test_multi_room_detection(self):
        profile = {}
        # 4 adults
        AgentService.extract_interests_and_slots("нас 4 взрослых, ищем хороший отель", profile)
        self.assertEqual(profile.get("adults_count"), 4)
        self.assertEqual(profile.get("rooms_count"), 2)
        self.assertIn("Connecting Rooms", profile.get("room_type"))

        # Family with kids
        profile_fam = {}
        AgentService.extract_interests_and_slots("едем семьей, двое взрослых и двое детей", profile_fam)
        self.assertTrue(profile_fam.get("has_kids"))
        self.assertEqual(profile_fam.get("adults_count"), 2)
        self.assertEqual(profile_fam.get("kids_count"), 2)
        self.assertEqual(profile_fam.get("rooms_count"), 2)

    def test_multi_room_budget_scaling(self):
        # 1 room
        breakdown_1 = MCPTools.calculate_budget_breakdown(
            destination_id="sochi",
            flight_price_roundtrip=25000,
            hotel_price_per_night=10000,
            transfer_price=3000,
            meals_type="Завтрак включен",
            nights=5,
            rooms_count=1
        )
        # 2 rooms
        breakdown_2 = MCPTools.calculate_budget_breakdown(
            destination_id="sochi",
            flight_price_roundtrip=25000,
            hotel_price_per_night=10000,
            transfer_price=3000,
            meals_type="Завтрак включен",
            nights=5,
            rooms_count=2
        )
        self.assertEqual(breakdown_1["mandatory"]["hotel_total"], 50000)
        self.assertEqual(breakdown_2["mandatory"]["hotel_total"], 100000)
        self.assertEqual(
            breakdown_2["mandatory"]["total_mandatory_rub"] - breakdown_1["mandatory"]["total_mandatory_rub"],
            50000
        )

    def test_flexible_dates_options(self):
        options = AgentService.calculate_flexible_dates_options(
            dest_id="turkey",
            base_mandatory_rub=100000,
            checkin_date_str="2026-11-15",
            nights=7
        )
        self.assertTrue(len(options) >= 2)
        self.assertTrue(any(opt["days_shift"] < 0 for opt in options))
        self.assertTrue(any(opt["days_shift"] > 0 for opt in options))
        for opt in options:
            self.assertIn("savings_rub", opt)
            self.assertIn("label", opt)
            self.assertIn("checkin_date", opt)

    def test_api_compliance_endpoint(self):
        resp = self.client.get("/api/compliance/thailand")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Таиланд", data["country"])
        self.assertIn("6 месяцев", data["passport_validity"])

    def test_api_calendar_export_get(self):
        resp = self.client.get("/api/cart/export-calendar")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "text/calendar; charset=utf-8")
        self.assertIn("BEGIN:VCALENDAR", resp.text)
        self.assertIn("attachment", resp.headers.get("content-disposition", ""))

    def test_api_calendar_export_post(self):
        payload = {
            "cart": {
                "booking_id": "OST-POST-TEST",
                "destination": "ОАЭ (Дубай)",
                "hotel": "Atlantis The Palm",
                "nights": 6,
                "flight": "Emirates EK-132",
                "transfer": "VIP-трансфер"
            }
        }
        resp = self.client.post("/api/cart/export-calendar", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("BEGIN:VCALENDAR", resp.text)
        self.assertIn("Atlantis The Palm", resp.text)
        self.assertIn("OST-POST-TEST", resp.text)

    def test_api_chat_stream_endpoint(self):
        from unittest.mock import AsyncMock, patch
        with patch.object(AgentService, "process_message", new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value = {
                "text": "Отличный выбор Сочи!\nВот готовый тур.",
                "reasoning_steps": [{"stage": "nlu", "title": "Анализ", "detail": "Сочи"}]
            }
            with self.client.stream("POST", "/api/chat/stream", json={"message": "Сочи на 5 ночей", "user_id": "test_stream_user"}) as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers.get("content-type", ""))
                
                lines = []
                for line in response.iter_lines():
                    if line:
                        lines.append(line)
                    if len(lines) >= 15:
                        break
                self.assertTrue(any("event: status" in l for l in lines))
                self.assertTrue(any("event: token" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
