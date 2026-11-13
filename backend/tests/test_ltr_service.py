"""
Модульные тесты для табличного ранжировщика туров CatBoost LTR (`LTRService`).
Проверяют:
- определение архетипа пользователя (`pet`, `family`, `luxury`, `budget`, `solo`),
- извлечение 12 признаков (`FEATURE_COLUMNS`) из карточек отелей, логистики Neo4j и расчёта бюджета,
- пересортировку пакетов `rank_evaluated_packages` и кандидатов отелей `rank_hotel_candidates`,
- запись онлайн-конверсий `record_online_feedback` (Шаг 4 концепта Synthetic User Simulation).
"""

import unittest
from pathlib import Path
from app.services.ltr_service import (
    LTRService,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_SCHEMA_V1,
    FEATURE_SCHEMA_V2,
    DESTINATION_TO_CANONICAL_CITIES_V2,
)


class LTRServiceTest(unittest.TestCase):
    def test_resolve_archetype_all_profiles(self):
        self.assertEqual(LTRService.resolve_archetype({"has_pets": True}), "pet")
        self.assertEqual(LTRService.resolve_archetype({"has_kids": True}), "family")
        self.assertEqual(LTRService.resolve_archetype({"interests": ["luxury"]}), "luxury")
        self.assertEqual(LTRService.resolve_archetype({"interests": ["budget_friendly"]}), "budget")
        self.assertEqual(LTRService.resolve_archetype({"budget_rub": 85000}), "budget")
        self.assertEqual(LTRService.resolve_archetype({"budget": 85000}), "budget")
        self.assertEqual(LTRService.resolve_archetype({"archetype": "luxury"}), "luxury")
        self.assertEqual(LTRService.resolve_archetype({"archetype": "budget"}), "budget")
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

    def test_schema_v2_canonical_cities_mapping(self):
        # В схеме V2 каждый регион обязан иметь собственный канонический маршрут, отличный от Турции
        for dest_id in ("uae", "thailand", "maldives", "egypt"):
            canon_v2 = LTRService.resolve_canonical_cities(dest_id, schema=FEATURE_SCHEMA_V2)
            self.assertNotEqual(canon_v2, "Стамбул -> Анталья -> Каппадокия")
            self.assertEqual(canon_v2, DESTINATION_TO_CANONICAL_CITIES_V2[dest_id])

    def test_schema_v2_center_vs_sea_distance_separation(self):
        # Отель у моря без указания центра: море не должно подменяться на центр!
        hotel_sea_only = {
            "id": "h_sea",
            "name": "Beach Resort",
            "distance_to_sea_meters": 200,
        }
        dist_c, has_c = LTRService._parse_dist_center_km_v2(hotel_sea_only)
        dist_s, has_s = LTRService._parse_dist_sea_km_v2(hotel_sea_only)

        self.assertEqual(has_c, 0)
        self.assertEqual(dist_c, 0.0)
        self.assertEqual(has_s, 1)
        self.assertEqual(dist_s, 0.2)

        # Отель в центре с тегом city_center
        hotel_center = {
            "id": "h_center",
            "name": "Urban Boutique",
            "distance_to_sea_meters": 0,
            "features": ["city_center"],
        }
        dist_c2, has_c2 = LTRService._parse_dist_center_km_v2(hotel_center)
        self.assertEqual(has_c2, 1)
        self.assertEqual(dist_c2, 0.5)

    def test_schema_v2_departure_hour_and_preference_matching(self):
        # 08:45 -> 8.75 часа
        hour, has_hour = LTRService._parse_departure_hour("08:45")
        self.assertEqual(has_hour, 1)
        self.assertAlmostEqual(hour, 8.75, places=2)

        # 14:30 -> 14.5 часа
        hour2, has_hour2 = LTRService._parse_departure_hour("14:30")
        self.assertEqual(has_hour2, 1)
        self.assertAlmostEqual(hour2, 14.5, places=2)

        # Проверка предпочтений по времени
        flight_morning = {"flight_type": "direct", "departure_time": "09:00"}
        matches_m = LTRService._check_preference_matches(
            flight_morning, {}, {"flight_time_pref": "morning"}
        )
        self.assertEqual(matches_m["is_direct_flight"], 1)
        self.assertEqual(matches_m["matches_time_pref"], 1)

        matches_d = LTRService._check_preference_matches(
            flight_morning, {}, {"flight_time_pref": "day"}
        )
        self.assertEqual(matches_d["matches_time_pref"], 0)

    def test_schema_v2_extract_candidate_features(self):
        hotel = {
            "id": "dubai_marina",
            "name": "Marina Hotel",
            "city": "Дубай",
            "stars": "5★",
            "price_per_night_rub": 20000,
            "meals": "Завтрак включен",
            "distance_to_sea_meters": 350,
            "pet_friendly": False,
        }
        logistics = {
            "flight": {
                "flight_type": "direct",
                "departure_time": "07:15",
                "layover_duration_minutes": 0,
                "price_rub_roundtrip": 35000,
            },
            "timeline": {},
        }
        profile = {
            "interests": ["luxury"],
            "nights": 6,
            "prefer_direct_flight": True,
            "flight_time_pref": "morning",
        }
        feat = LTRService.extract_candidate_features_v2(
            hotel=hotel,
            destination_id="uae",
            profile=profile,
            nights=6,
            logistics=logistics,
            reference_price=100000.0,
        )

        for col in FEATURE_COLUMNS_V2:
            self.assertIn(col, feat, f"Missing column {col} in V2 features")

        self.assertEqual(feat["archetype"], "luxury")
        self.assertEqual(feat["cities"], DESTINATION_TO_CANONICAL_CITIES_V2["uae"])
        self.assertEqual(feat["is_direct_flight"], 1)
        self.assertEqual(feat["matches_direct_pref"], 1)
        self.assertEqual(feat["matches_time_pref"], 1)
        self.assertEqual(feat["has_dist_sea"], 1)
        self.assertEqual(feat["has_dist_center"], 0)
        self.assertAlmostEqual(feat["departure_hour"], 7.25, places=2)
        self.assertGreater(feat["price_delta_vs_cheapest"], 0.0)
        self.assertGreater(feat["relative_surcharge_pct"], 0.0)

    def test_fail_fast_load_model(self):
        with self.assertRaises(RuntimeError):
            LTRService.load_model(Path("non_existent_model_file.cbm"), fail_fast=True)

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

    def test_rank_evaluated_packages_v2_features(self):
        v2_path = Path(__file__).resolve().parents[1] / "app" / "data" / "catboost_travel_ranker_v2.cbm"
        if v2_path.exists():
            orig_path = LTRService._model_path
            LTRService.load_model(str(v2_path))
            self.addCleanup(lambda: LTRService.load_model(str(orig_path)))
            self.assertEqual(LTRService.get_feature_schema(), FEATURE_SCHEMA_V2)

        h = {"id": "h1", "name": "Hotel", "stars": 4, "price_per_night_rub": 5000, "meals": "Завтрак включен"}
        logistics = {"flight": {"flight_type": "direct", "departure_time": "09:00"}, "timeline": {}}
        b1 = {"mandatory": {"total_mandatory_rub": 60000}}
        b2 = {"mandatory": {"total_mandatory_rub": 80000}}
        ranked, meta = LTRService.rank_evaluated_packages(
            [(h, logistics, b1), (h, logistics, b2)],
            destination_id="sochi",
            profile={"archetype": "budget"},
            nights=5,
        )
        self.assertEqual(len(ranked), 2)
        # Cheaper package must score higher or equal
        self.assertGreaterEqual(ranked[0][0]["ltr_score"], ranked[1][0]["ltr_score"])

    def test_pareto_monotonicity_all_archetypes(self):
        v2_path = Path(__file__).resolve().parents[1] / "app" / "data" / "catboost_travel_ranker_v2.cbm"
        if v2_path.exists():
            orig_path = LTRService._model_path
            LTRService.load_model(str(v2_path))
            self.addCleanup(lambda: LTRService.load_model(str(orig_path)))

        from app.services.ltr_service import ARCHETYPES
        for arch in ARCHETYPES:
            with self.subTest(archetype=arch):
                feat_cheap = {
                    "archetype": arch, "cities": "Сочи -> Красная Поляна -> Сириус", "days": 5,
                    "price": 50000.0, "log_price": 10.8, "log_price_per_day": 9.2,
                    "transfers": 1, "layover_hours": 0.0, "hotel_stars": 4,
                    "stars_per_10k": 4.0 / (50000.0 / 5 / 10000.0),
                    "pet_friendly": 1, "breakfast": 1, "dist_center_km": 1.0, "has_dist_center": 1,
                    "dist_sea_km": 0.5, "has_dist_sea": 1, "departure_hour": 10.0, "has_departure_time": 1,
                    "is_direct_flight": 1, "matches_time_pref": 1, "matches_direct_pref": 1,
                    "price_delta_vs_cheapest": 0.0, "relative_surcharge_pct": 0.0,
                }
                feat_exp = dict(feat_cheap)
                feat_exp["price"] = 80000.0
                feat_exp["log_price"] = 11.3
                feat_exp["log_price_per_day"] = 9.7
                feat_exp["stars_per_10k"] = 4.0 / (80000.0 / 5 / 10000.0)
                feat_exp["price_delta_vs_cheapest"] = 30000.0
                feat_exp["relative_surcharge_pct"] = 0.60

                # Surrogate score must respect Pareto monotonicity
                score_cheap = LTRService._surrogate_score(feat_cheap)
                score_exp = LTRService._surrogate_score(feat_exp)
                self.assertGreaterEqual(score_cheap, score_exp)

                # Batch scoring must respect Pareto monotonicity
                scores, _, _ = LTRService.score_features_batch([feat_cheap, feat_exp])
                self.assertGreaterEqual(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
