"""
Модульные тесты для табличного ранжировщика туров CatBoost LTR (`LTRService`).
Проверяют:
- определение архетипа пользователя (`pet`, `family`, `luxury`, `budget`, `solo`),
- извлечение 12 признаков (`FEATURE_COLUMNS`) из карточек отелей, логистики Neo4j и расчёта бюджета,
- пересортировку пакетов `rank_evaluated_packages` и кандидатов отелей `rank_hotel_candidates`,
- запись онлайн-конверсий `record_online_feedback` (Шаг 4 концепта Synthetic User Simulation).
"""

import unittest
from app.services.ltr_service import LTRService, FEATURE_COLUMNS


class LTRServiceTest(unittest.TestCase):
    def test_resolve_archetype_all_profiles(self):
        self.assertEqual(LTRService.resolve_archetype({"has_pets": True}), "pet")
        self.assertEqual(LTRService.resolve_archetype({"has_kids": True}), "family")
        self.assertEqual(LTRService.resolve_archetype({"interests": ["luxury"]}), "luxury")
        self.assertEqual(LTRService.resolve_archetype({"interests": ["budget_friendly"]}), "budget")
        self.assertEqual(LTRService.resolve_archetype({"budget_rub": 85000}), "budget")
        self.assertEqual(LTRService.resolve_archetype({"party_size": "соло", "adults_count": 1}), "solo")
        self.assertEqual(LTRService.resolve_archetype({}), "solo")

    def test_extract_candidate_features_matches_schema(self):
        hotel = {
            "id": "sochi_radisson",
            "name": "Radisson Collection Paradise",
            "city": "Сириус",
            "stars": "5★",
            "price_per_night_rub": 15000,
            "meals": "Завтрак включен",
            "distance_to_sea": "150 м",
            "pet_friendly": True,
        }
        logistics = {
            "flight": {
                "flight_type": "connecting",
                "layover_duration_minutes": 180,
                "price_rub_roundtrip": 24000,
            },
            "timeline": {
                "layover_details": {"is_connecting": True, "duration_minutes": 180},
                "post_flight_buffer_min": 50,
            },
        }
        breakdown = {
            "mandatory": {"total_mandatory_rub": 131500}
        }
        profile = {"has_pets": True, "nights": 7}

        feat = LTRService.extract_candidate_features(
            hotel=hotel,
            destination_id="sochi",
            profile=profile,
            nights=7,
            logistics=logistics,
            breakdown=breakdown,
        )

        for col in FEATURE_COLUMNS:
            self.assertIn(col, feat)
        self.assertIn("price", feat)
        self.assertEqual(feat["archetype"], "pet")
        self.assertEqual(feat["cities"], "Сочи -> Красная Поляна -> Сириус")
        self.assertEqual(feat["days"], 7)
        self.assertEqual(feat["price"], 131500.0)
        self.assertEqual(feat["transfers"], 2)  # connecting flight + ground transfer
        self.assertEqual(feat["layover_hours"], 3.0)
        self.assertEqual(feat["hotel_stars"], 5)
        self.assertAlmostEqual(feat["dist_center_km"], 0.15, places=2)
        self.assertEqual(feat["pet_friendly"], 1)
        self.assertEqual(feat["breakfast"], 1)
        self.assertGreater(feat["log_price"], 0)
        self.assertGreater(feat["log_price_per_day"], 0)
        self.assertGreater(feat["stars_per_10k"], 0)

    def test_rank_evaluated_packages_prioritizes_pet_friendly_for_pet_archetype(self):
        h_no_pet = {
            "id": "h_no_pet",
            "name": "Отель Без Животных",
            "stars": "4★",
            "price_per_night_rub": 6000,
            "meals": "Завтрак включен",
            "distance_to_sea": "500 м",
            "pet_friendly": False,
        }
        h_pet = {
            "id": "h_pet",
            "name": "Pet Friendly Resort",
            "stars": "4★",
            "price_per_night_rub": 6500,
            "meals": "Завтрак включен",
            "distance_to_sea": "300 м",
            "pet_friendly": True,
        }
        logistics = {"flight": {"flight_type": "direct", "price_rub_roundtrip": 20000}, "timeline": {}}
        b1 = {"mandatory": {"total_mandatory_rub": 64000}}
        b2 = {"mandatory": {"total_mandatory_rub": 67500}}

        ranked, meta = LTRService.rank_evaluated_packages(
            evaluated_options=[(h_no_pet, logistics, b1), (h_pet, logistics, b2)],
            destination_id="sochi",
            profile={"has_pets": True},
            nights=7,
        )

        self.assertEqual(meta["archetype"], "pet")
        self.assertEqual(meta["candidates_count"], 2)
        self.assertEqual(ranked[0][0]["id"], "h_pet")
        self.assertIn("ltr_score", ranked[0][0])

    def test_online_feedback_recording(self):
        tour_a = {"cities": "Сочи -> Красная Поляна -> Сириус", "days": 7, "price": 85000}
        tour_b = {"cities": "Сочи -> Красная Поляна -> Сириус", "days": 7, "price": 120000}
        res = LTRService.record_online_feedback("family", tour_a, tour_b, margin=2)
        self.assertTrue(res["recorded"])
        self.assertEqual(res["event"]["winner"], "A")
        self.assertEqual(res["event"]["archetype"], "family")


if __name__ == "__main__":
    unittest.main()
