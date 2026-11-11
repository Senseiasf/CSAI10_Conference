"""LTR после жёстких фильтров, согласованность API/LLM и сохранение предыдущих функций."""
import json
import unittest
from collections import deque
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService
from app.services.graph_service import GraphService
from app.services.llm_service import LLMService
from app.services.ltr_service import LTRService


def hotel(index, price=3000):
    return {"id": f"hotel-{index}", "name": f"Hotel {index}", "city": "Сочи", "stars": "4★",
            "price_per_night_rub": price, "price_per_night_formatted": f"{price:,} ₽",
            "meals": "Завтрак включен", "cancellation": "Бесплатно", "distance_to_sea": "200 м",
            "features": ["pet_friendly", "family_kids"], "pet_friendly": True}


def logistics():
    return {"compatible": True, "airport": "Сочи", "flight": {"price_rub_roundtrip": 20000,
            "airline": "Test Air", "flight_no": "TA-1", "origin_name": "Москва", "origin_code": "SVO",
            "departure_time": "01:00", "arrival_time": "05:00", "flight_type": "direct",
            "return_flight": {"flight_no": "TA-2", "airline": "Test Air", "origin_code": "AER",
                              "departure_time": "15:00", "arrival_time": "18:00"}},
            "transfer": {"price_rub": 2000, "name": "Taxi", "type": "taxi"},
            "timeline": {"flight_departure": "01:00", "flight_arrival": "05:00", "transfer_start": "05:50",
                         "hotel_arrival": "06:00", "hotel_checkin_policy": "15:00", "checkin_wait_minutes": 540}}


class AgentLTRIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, *, profile=None, hotels=None, message="Сочи на 7 ночей"):
        profile = dict({"nights": 7, "checkin_date": "2026-11-10", "checkout_date": "2026-11-17",
                        "dates": "10.11.2026 — 17.11.2026"}, **(profile or {}))
        hotels = hotels if hotels is not None else [hotel(index, 3000 + index * 100) for index in range(5)]
        user = SimpleNamespace(profile=profile, current_cart={}, current_state="INSPIRATION", conversation_history=[])
        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[{"id": "sochi", "name": "Сочи", "score": 1.0}])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            compare = stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": hotels})))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics())))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))
            advice = stack.enter_context(patch.object(LLMService, "generate_concierge_advice", new=AsyncMock(return_value="Совет")))
            stack.enter_context(patch.object(LLMService, "dispatch_intent", new=AsyncMock(return_value=None)))
            stack.enter_context(patch("app.services.agent_service.settings", SimpleNamespace(is_mock_llm=False, OPENAI_MODEL="test")))
            stack.enter_context(patch.object(LTRService, "score_features_batch", side_effect=lambda batch: ([float(idx) for idx in range(len(batch))], .1, "test scorer")))
            result = await AgentService.process_message("web:ltr-fixture", message)
        return result, compare, advice

    async def test_default_top_three_match_scores_llm_cart_and_reasoning(self):
        result, compare, advice = await self.run_agent()
        self.assertEqual([item["id"] for item in result["compared_hotels"]], ["hotel-4", "hotel-3", "hotel-2"])
        self.assertEqual(compare.call_args.kwargs["limit"], 30)
        self.assertEqual(result["ltr_ranking"], result["cart"]["ltr_ranking"])
        self.assertEqual(result["ltr_ranking"]["top_hotel"], result["cart"]["hotel"])
        self.assertEqual(result["ltr_ranking"]["top_k"], 3)
        self.assertEqual(result["ltr_ranking"]["candidates_count"], 5)
        self.assertEqual(result["ltr_ranking"]["selection_policy"], "ltr")
        advice.assert_not_awaited()
        self.assertNotIn("AI-консьерж", result["text"])
        self.assertIn("ltr_ranking", [step["stage"] for step in result["reasoning_steps"]])
        self.assertIn("Оценено 5 пакетов", next(step["detail"] for step in result["reasoning_steps"] if step["stage"] == "filtering"))
        self.assertNotIn('class="agent-reasoning"', result["text"])
        self.assertNotIn("ML-ранжирование", result["text"])
        self.assertIn("trip_conditions", result)
        self.assertEqual(result["logistics"]["timeline"]["checkin_wait_minutes"], 540)

    async def test_explicit_budget_preference_beats_higher_ml_scores(self):
        for profile, message in (({"interests": ["budget_friendly"]}, "Сочи на 7 ночей"),
                                 ({"interests": ["luxury"]}, "Сделай самый дешевый вариант в Сочи")):
            with self.subTest(profile=profile, message=message):
                result, _, _ = await self.run_agent(profile=profile, message=message)
                self.assertEqual(result["cart"]["hotel_id"], "hotel-0")
                self.assertEqual(result["ltr_ranking"]["top_hotel"], "Hotel 0")
                self.assertEqual(result["ltr_ranking"]["selection_policy"], "budget")

    async def test_luxury_priority_and_price_ties_use_consistent_metadata(self):
        hotels = [hotel(0, 10000), hotel(1, 3000), hotel(2, 4000)]
        result, _, _ = await self.run_agent(profile={"interests": ["luxury"]}, hotels=hotels)
        self.assertEqual(result["cart"]["hotel_id"], "hotel-0")
        self.assertEqual(result["ltr_ranking"]["top_score"], result["compared_hotels"][0]["ltr_score"])
        tied, _, _ = await self.run_agent(profile={"budget_rub": 200000}, hotels=[hotel(index) for index in range(3)])
        self.assertEqual(tied["ltr_ranking"]["selection_policy"], "ltr")

    async def test_price_filter_cannot_be_undone_by_ltr(self):
        result, _, _ = await self.run_agent(profile={"budget_rub": 100000, "adults_count": 1},
                                           hotels=[hotel(0, 1000), hotel(1, 100000)])
        self.assertEqual([item["id"] for item in result["compared_hotels"]], ["hotel-0"])
        self.assertLessEqual(result["budget_breakdown"]["total_trip_budget_range_rub"]["max"], 100000)

    async def test_hard_constraints_still_reach_filter_and_conditions(self):
        result, compare, _ = await self.run_agent(profile={"has_kids": True, "has_pets": True,
                                                           "private_room_required": True, "budget_rub": 200000})
        for key in ("family_friendly", "pet_friendly", "private_room_required"):
            self.assertTrue(compare.call_args.kwargs[key])
        for key in ("has_kids", "has_pets", "private_room_required"):
            self.assertTrue(result["trip_conditions"][key])
        self.assertEqual(result["trip_conditions"]["budget_rub"], 200000)

    async def test_empty_candidates_preserve_conditions_and_skip_advice(self):
        result, _, advice = await self.run_agent(profile={"private_room_required": True}, hotels=[])
        self.assertEqual(result["state"], "NO_MATCHES")
        self.assertTrue(result["trip_conditions"]["private_room_required"])
        self.assertNotIn("hotel", result["cart"])
        advice.assert_not_awaited()

    async def test_multi_city_ranks_both_cities_and_exposes_metadata(self):
        profile = {"nights": 7, "checkin_date": "2026-11-10", "checkout_date": "2026-11-17",
                   "has_kids": True, "has_pets": True, "private_room_required": True}
        hotels_1, hotels_2 = [hotel(0), hotel(1)], [hotel(2), hotel(3)]
        with (patch.object(MCPTools, "compare_hotels", new=AsyncMock(side_effect=[{"compared_hotels": hotels_1}, {"compared_hotels": hotels_2}])) as compare,
              patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics())),
              patch.object(GraphService, "find_intercity_logistics", new=AsyncMock(return_value=None)),
              patch.object(AgentService, "_save_session", new_callable=AsyncMock)):
            result = await AgentService.assemble_multi_city_itinerary("sochi", {"id": "sochi", "name": "Сочи"}, profile, {},
                7, None, "test-multi-ltr", "Составь тур по городам", False, False, SimpleNamespace())
        self.assertTrue(all("ltr_score" in item for item in result["compared_hotels"]))
        self.assertEqual(result["ltr_ranking"], result["cart"]["ltr_ranking"])
        self.assertEqual(len(result["ltr_ranking"]["cities"]), 2)
        self.assertEqual(result["ltr_ranking"]["top_k"], 2)
        self.assertIn("ltr_ranking", [step["stage"] for step in result["reasoning_steps"]])
        for call in compare.call_args_list:
            self.assertTrue(call.kwargs["pet_friendly"])
            self.assertTrue(call.kwargs["family_friendly"])
            self.assertTrue(call.kwargs["private_room_required"])

    async def test_empty_multi_city_does_not_invent_hotels(self):
        with (patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": []})),
              patch.object(MCPTools, "check_logistics_compatibility", new_callable=AsyncMock) as route,
              patch.object(AgentService, "_save_session", new_callable=AsyncMock)):
            result = await AgentService.assemble_multi_city_itinerary("sochi", {"id": "sochi"},
                {"has_kids": True, "has_pets": True}, {}, 7, 100000, "test-empty", "Тур по городам", False, False, SimpleNamespace())
        self.assertEqual(result["state"], "NO_MATCHES")
        self.assertEqual(result["compared_hotels"], [])
        self.assertNotIn("hotel", result["cart"])
        route.assert_not_awaited()


class ConciergeContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_llm_receives_only_final_top_three_candidates_and_scores(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Совет"))])))))
        candidates = [dict(hotel(index), ltr_score=index + .5) for index in range(4)]
        with patch.object(LLMService, "get_client", return_value=client):
            result = await LLMService.generate_concierge_advice("Сочи", "Hotel 0", 7, "Отдых", {}, candidates)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(result, "Совет")
        self.assertIn("Топ-3 кандидата", prompt)
        self.assertIn("LTR=0.500", prompt)
        self.assertIn("Hotel 2", prompt)
        self.assertNotIn("Hotel 3", prompt)


class LTRApiTest(unittest.TestCase):
    def test_status_and_health_contain_real_model_metadata(self):
        browser = TestClient(app)
        status = browser.get("/api/ltr/status")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["features_count"], 10)
        self.assertIn("engine", status.json())
        with (patch("app.main.check_postgres_health", new=AsyncMock(return_value=True)),
              patch("app.main.check_neo4j_health", new=AsyncMock(return_value=True)),
              patch.object(GraphService, "get_graph_stats", new=AsyncMock(return_value={}))):
            health = browser.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["ltr_ranker"]["engine"], status.json()["engine"])

    def test_feedback_changes_status_and_rejects_invalid_requests(self):
        with (patch.object(LTRService, "_online_feedback_buffer", deque(maxlen=3)),
              patch.object(LTRService, "_online_feedback_events", 0)):
            browser = TestClient(app)
            payload = {"archetype": "family", "chosen_tour": {"price": 10000}, "rejected_tour": {"price": 20000}, "margin": 2}
            response = browser.post("/api/ltr/feedback", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["buffer_size"], 1)
            status = browser.get("/api/ltr/status").json()
            self.assertEqual(status["online_feedback_events"], 1)
            for invalid in ({"archetype": "unknown"}, {"margin": 0}, {"margin": 11}, {"margin": True}, {"margin": "2"}):
                with self.subTest(invalid=invalid):
                    self.assertEqual(browser.post("/api/ltr/feedback", json={**payload, **invalid}).status_code, 422)
            self.assertEqual(browser.get("/api/ltr/status").json()["online_feedback_events"], 1)

    def test_sse_ranking_stage_is_only_emitted_for_ranked_trips_and_cookie_is_preserved(self):
        for result, expected in (({"text": "Готово", "ltr_ranking": {"engine": "test"}}, True),
                                 ({"text": "Уточните даты"}, False),
                                 ({"text": "Бронирование недоступно", "booking_action": {"action": "book"}}, False)):
            with self.subTest(result=result), patch.object(AgentService, "process_message", new=AsyncMock(return_value=result)):
                response = TestClient(app).post("/api/chat/stream", json={"message": "Сочи"})
                self.assertEqual(response.status_code, 200)
                self.assertIn("HttpOnly", response.headers["set-cookie"])
                status_events = [json.loads(event.split("data: ", 1)[1]) for event in response.text.split("\n\n") if event.startswith("event: status")]
                self.assertEqual(any(event["stage"] == "ltr_ranking" for event in status_events), expected)
                self.assertIn("event: result", response.text)
