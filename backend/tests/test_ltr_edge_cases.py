"""Контракты признаков, реальная модель, отказоустойчивость и парные обучающие данные."""
import copy
import json
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import app.services.ltr_service as ltr
from app.services.ltr_service import LTRService, FEATURE_COLUMNS, CAT_FEATURES

try:
    from app.scripts import train_ltr
except ImportError:
    train_ltr = None


class LTRStateTest(unittest.TestCase):
    def setUp(self):
        for key in ("_model", "_model_loaded", "_model_path", "_load_error", "_inference_failed",
                    "_online_feedback_buffer", "_online_feedback_events"):
            value = getattr(LTRService, key)
            self.addCleanup(setattr, LTRService, key, value)


class FeatureEdgeCasesTest(unittest.TestCase):
    def test_archetype_priority_and_incomplete_profiles(self):
        cases = [(None, "solo"), ({}, "solo"),
                 ({"has_pets": True, "has_kids": True}, "pet"),
                 ({"has_kids": False, "kids_count": 2}, "family"),
                 ({"budget_rub": 90000}, "budget"), ({"budget_rub": 250000}, "solo"),
                 ({"budget_rub": 90000, "interests": ["extreme_luxury"]}, "luxury"),
                 ({"budget_rub": 90000, "interests": ["splurge"]}, "luxury"),
                 ({"budget_rub": "invalid"}, "solo"), ({"budget_rub": float("nan")}, "solo")]
        for profile, expected in cases:
            with self.subTest(profile=profile):
                self.assertEqual(LTRService.resolve_archetype(profile), expected)

    def test_canonical_routes_cover_catalog_and_city_fallbacks(self):
        for destination in ("sochi", "georgia", "turkey", "spb", "altai", "uae", "thailand", "maldives", "egypt"):
            with self.subTest(destination=destination):
                self.assertEqual(LTRService.resolve_canonical_cities(destination), ltr.DESTINATION_TO_CANONICAL_CITIES[destination])
        for city, destination in (("Красная Поляна", "sochi"), ("Батуми", "georgia"),
                                  ("Анталья", "turkey"), ("Петергоф", "spb"), ("Манжерок", "altai")):
            self.assertEqual(LTRService.resolve_canonical_cities("bali", city), ltr.DESTINATION_TO_CANONICAL_CITIES[destination])
        self.assertEqual(LTRService.resolve_canonical_cities("", "unknown"), "unknown")

    def test_stars_are_bounded_and_missing_values_are_safe(self):
        for value, expected in (("5★", 5), ("3 звезды", 3), (4, 4), (0, 1), (7, 5),
                                (None, 4), ("", 4), ("без звёзд", 4), (float("nan"), 4)):
            with self.subTest(value=value):
                self.assertEqual(LTRService._parse_stars(value), expected)

    def test_distance_formats_and_invalid_numbers(self):
        cases = [({"dist_center_km": 0}, 0), ({"dist_center_km": 2.5}, 2.5),
                 ({"dist_center_km": "abc"}, 1.5), ({"dist_center_km": float("nan")}, 1.5),
                 ({"distance_to_sea_meters": 0}, .1), ({"distance_to_sea_meters": 1500}, 1.5)]
        cases += [({"distance_to_sea": text}, expected) for text, expected in
                  (("350 м", .35), ("2,4 км", 2.4), ("1.8 км", 1.8), ("Первая линия", .2),
                   ("Пляж / Первая линия", .2), ("", 1.5), (None, 1.5))]
        for hotel, expected in cases:
            with self.subTest(hotel=hotel):
                self.assertAlmostEqual(LTRService._parse_dist_center_km(hotel), expected)

    def test_pet_flags_and_explicit_denial_override_tags(self):
        for value, expected in ((True, 1), (False, 0), (1, 1), (0, 0), ("✅ Да", 1),
                                ("❌ Нет", 0), ("true", 1), ("false", 0), ("1", 1)):
            with self.subTest(value=value):
                self.assertEqual(LTRService._parse_pet_friendly({"pet_friendly": value}), expected)
        self.assertEqual(LTRService._parse_pet_friendly({"features": ["wifi", "pet_friendly"]}), 1)
        self.assertEqual(LTRService._parse_pet_friendly({"pet_friendly": False, "features": ["pet_friendly"]}), 0)
        self.assertEqual(LTRService._parse_pet_friendly({"features": ["no_pets"]}), 0)
        self.assertEqual(LTRService._parse_pet_friendly({}), 0)

    def test_meals_and_explicit_breakfast_flag(self):
        for meals, expected in (("Завтрак включен", 1), ("All Inclusive (Все включено)", 1),
                                ("Полупансион", 1), ("Без питания (общая кухня)", 0),
                                ("Room only", 0), ("", 0), (None, 0), ("Неизвестно", 0)):
            with self.subTest(meals=meals):
                self.assertEqual(LTRService._parse_breakfast({"meals": meals}), expected)
        self.assertEqual(LTRService._parse_breakfast({"breakfast": False, "meals": "Завтрак"}), 0)
        self.assertEqual(LTRService._parse_breakfast({"breakfast": "false"}), 0)

    def test_transfer_and_connection_features(self):
        for connecting, without_transfer, transfers, hours in ((False, False, 1, 80 / 60),
                                                              (False, True, 0, 80 / 60),
                                                              (True, False, 2, 2.5), (True, True, 1, 2.5)):
            with self.subTest(connecting=connecting, without_transfer=without_transfer):
                logistics = {"flight": {"flight_type": "connecting" if connecting else "direct",
                                          "layover_duration_minutes": 150 if connecting else 0},
                             "timeline": {"post_flight_buffer_min": 80}}
                features = LTRService.extract_candidate_features({}, "sochi", {"without_transfer": without_transfer}, logistics=logistics)
                self.assertEqual(features["transfers"], transfers)
                self.assertAlmostEqual(features["layover_hours"], hours, places=2)
        for missing in (None, {}):
            features = LTRService.extract_candidate_features({}, "sochi", logistics=missing)
            self.assertEqual((features["transfers"], features["layover_hours"]), (0, 0))

    def test_package_price_room_scaling_and_positive_duration(self):
        features = LTRService.extract_candidate_features({"price_per_night_rub": 5000}, "sochi",
                                                         {"rooms_count": 2}, 10)
        self.assertEqual(features["price"], 100000)
        actual = LTRService.extract_candidate_features({}, "sochi", breakdown={"mandatory": {"total_mandatory_rub": 145000}})
        self.assertEqual(actual["price"], 145000)
        for nights in (0, -3, None, "bad"):
            self.assertGreaterEqual(LTRService.extract_candidate_features({}, "sochi", nights=nights)["days"], 1)


class ModelAndRankingTest(LTRStateTest):
    def features(self):
        return LTRService.extract_candidate_features({"stars": "4★"}, "sochi")

    def test_empty_batches_do_not_load_a_model(self):
        with patch.object(LTRService, "load_model") as load:
            for method in (LTRService.rank_hotel_candidates, LTRService.rank_evaluated_packages):
                ranked, meta = method([], "sochi")
                self.assertEqual(ranked, [])
                self.assertEqual((meta["candidates_count"], meta["top_k"], meta["engine"]), (0, 0, "none"))
            load.assert_not_called()

    def test_stable_ties_top_k_and_input_immutability(self):
        hotels = [{"id": str(idx), "name": str(idx), "features": ["wifi"]} for idx in range(3)]
        original = copy.deepcopy(hotels)
        options = [(hotel, {}, {"mandatory": {"total_mandatory_rub": 50000}}) for hotel in hotels]
        with patch.object(LTRService, "score_features_batch", return_value=([1.0, 1.0, 1.0], .1, "test")):
            for top_k, count in ((None, 3), (0, 0), (1, 1), (3, 3), (10, 3)):
                with self.subTest(top_k=top_k):
                    ranked, meta = LTRService.rank_hotel_candidates(hotels, "sochi", top_k=top_k)
                    packages, package_meta = LTRService.rank_evaluated_packages(options, "sochi", top_k=top_k)
                    self.assertEqual([hotel["id"] for hotel in ranked], ["0", "1", "2"][:count])
                    self.assertEqual([package[0]["id"] for package in packages], ["0", "1", "2"][:count])
                    self.assertEqual(meta["top_k"], count)
                    self.assertEqual(package_meta["top_k"], count)
                    if ranked:
                        self.assertIsNot(ranked[0], hotels[0])
                        self.assertIn("ltr_score", ranked[0])
            with self.assertRaises(ValueError):
                LTRService.rank_hotel_candidates(hotels, "sochi", top_k=-1)
        self.assertEqual(hotels, original)

    def test_missing_and_corrupt_models_use_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.cbm"
            corrupt.write_bytes(b"invalid model")
            for path in (Path(directory) / "missing.cbm", corrupt):
                with self.subTest(path=path):
                    self.assertIsNone(LTRService.load_model(path))
                    scores, _, engine = LTRService.score_features_batch([self.features()])
                    self.assertEqual(engine, ltr.FALLBACK_ENGINE)
                    self.assertTrue(math.isfinite(scores[0]))
                    self.assertFalse(LTRService.get_status()["available"])

    def test_missing_catboost_does_not_claim_native_inference(self):
        with patch.object(ltr, "HAS_CATBOOST", False):
            LTRService.load_model(ltr.DEFAULT_MODEL_PATH)
            status = LTRService.get_status()
            self.assertTrue(status["model_present"])
            self.assertFalse(status["available"])
            self.assertFalse(status["catboost_installed"])
            self.assertEqual(status["engine"], ltr.FALLBACK_ENGINE)

    @unittest.skipUnless(ltr.HAS_CATBOOST, "CatBoost is not installed")
    def test_native_model_schema_and_cached_loading(self):
        LTRService._model_loaded = False
        LTRService._model_path = ltr.DEFAULT_MODEL_PATH
        with patch.object(ltr, "CatBoostRanker", wraps=ltr.CatBoostRanker) as constructor:
            model = LTRService.load_model()
            self.assertIsNotNone(model)
            self.assertIs(LTRService.load_model(), model)
            self.assertEqual(constructor.call_count, 1)
        self.assertEqual(model.feature_names_, FEATURE_COLUMNS)
        self.assertEqual(model.get_cat_feature_indices(), [0, 1])
        scores, latency, engine = LTRService.score_features_batch([self.features()] * 3)
        self.assertEqual(engine, ltr.NATIVE_ENGINE)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertGreaterEqual(latency, 0)

    @unittest.skipUnless(ltr.HAS_CATBOOST, "CatBoost is not installed")
    def test_incompatible_feature_schema_is_rejected(self):
        fake = MagicMock(feature_names_=list(reversed(FEATURE_COLUMNS)))
        fake.get_cat_feature_indices.return_value = [0, 1]
        with patch.object(ltr, "CatBoostRanker", return_value=fake):
            self.assertIsNone(LTRService.load_model(ltr.DEFAULT_MODEL_PATH))
        self.assertEqual(LTRService.get_status()["fallback_reason"], "model_load_failed")

    def test_prediction_errors_lengths_and_nonfinite_scores_fall_back(self):
        for prediction in (RuntimeError("failed"), [], [1.0], [float("nan"), 1], [float("inf"), 2]):
            with self.subTest(prediction=prediction):
                fake = MagicMock()
                if isinstance(prediction, Exception):
                    fake.predict.side_effect = prediction
                else:
                    fake.predict.return_value = prediction
                with patch.object(LTRService, "load_model", return_value=fake):
                    scores, _, engine = LTRService.score_features_batch([self.features(), self.features()])
                    self.assertEqual(engine, ltr.FALLBACK_ENGINE)
                    self.assertEqual(len(scores), 2)
                    self.assertTrue(all(math.isfinite(score) for score in scores))
                    self.assertFalse(LTRService.get_status()["available"])

    def test_dirty_features_are_normalized_before_both_engines(self):
        features = self.features()
        features.update(archetype=None, cities=float("nan"), price=float("inf"),
                        days=-1, pet_friendly="false", breakfast="1")
        normalized = LTRService.normalize_features(features)
        self.assertEqual(list(normalized), FEATURE_COLUMNS)
        self.assertTrue(all(isinstance(normalized[name], str) for name in CAT_FEATURES))
        self.assertEqual((normalized["pet_friendly"], normalized["breakfast"]), (0, 1))
        self.assertTrue(all(math.isfinite(normalized[name]) for name in FEATURE_COLUMNS if name not in CAT_FEATURES))
        with patch.object(LTRService, "load_model", return_value=None):
            scores, _, engine = LTRService.score_features_batch([features])
            self.assertEqual(engine, ltr.FALLBACK_ENGINE)
            self.assertTrue(math.isfinite(scores[0]))

    def test_feedback_is_bounded_unique_and_copied(self):
        LTRService._online_feedback_buffer = deque(maxlen=2)
        LTRService._online_feedback_events = 0
        chosen = {"days": 7, "nested": {"price": 10000}}
        events = [LTRService.record_online_feedback("solo", chosen, {}, margin=2) for _ in range(3)]
        self.assertEqual(len({event["event"]["pair_id"] for event in events}), 3)
        chosen["nested"]["price"] = 999
        events[-1]["event"]["tour_a"]["nested"]["price"] = 888
        self.assertEqual(LTRService._online_feedback_buffer[-1]["tour_a"]["nested"]["price"], 10000)
        status = LTRService.get_status()
        self.assertEqual((status["online_feedback_events"], status["feedback_buffer_size"]), (3, 2))

    def test_invalid_feedback_does_not_enter_buffer(self):
        count = len(LTRService._online_feedback_buffer)
        for archetype, margin in (("unknown", 1), ("solo", 0), ("solo", 11), ("solo", True), ("solo", "2")):
            with self.subTest(archetype=archetype, margin=margin), self.assertRaises(ValueError):
                LTRService.record_online_feedback(archetype, {}, {}, margin)
        self.assertEqual(len(LTRService._online_feedback_buffer), count)


@unittest.skipUnless(train_ltr is not None, "Training dependencies are not installed")
class TrainingPipelineTest(unittest.TestCase):
    def pairs(self):
        features = {"cities": "Сочи -> Красная Поляна -> Сириус", "days": 7, "price": 85000,
                    "transfers": 1, "layover_hours": 1.5, "hotel_stars": 4,
                    "dist_center_km": .5, "pet_friendly": "false", "breakfast": "1"}
        return [{"pair_id": idx, "archetype": "solo", "winner": "A" if idx % 2 else "B",
                 "tour_a": dict(features, price=85000 + idx * 100),
                 "tour_b": dict(features, price=120000 + idx * 100)} for idx in range(5)]

    def load(self, pairs):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "pairs.jsonl"
            file.write_text("\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs), encoding="utf-8")
            return train_ltr.load_and_flatten_jsonl(file)

    def test_training_and_inference_feature_schemas_match(self):
        self.assertEqual(train_ltr.FEATURE_COLUMNS, FEATURE_COLUMNS)
        self.assertEqual(train_ltr.CAT_FEATURES, CAT_FEATURES)

    def test_pair_labels_binary_strings_and_group_split(self):
        pairs = self.pairs()
        data = self.load(pairs)
        self.assertEqual(len(data), 10)
        self.assertEqual(set(data["pet_friendly"]), {0})
        self.assertEqual(set(data["breakfast"]), {1})
        for pair in pairs:
            self.assertEqual(data.loc[data.group_id == pair["pair_id"], "target"].tolist(),
                             [1, 0] if pair["winner"] == "A" else [0, 1])
        train, test = train_ltr.split_by_group(data, test_size=.4)
        self.assertFalse(set(train.group_id) & set(test.group_id))
        self.assertEqual(len(train) + len(test), len(data))
        for subset in (train, test):
            self.assertEqual(subset.group_id.tolist(), sorted(subset.group_id.tolist()))
            self.assertTrue((subset.groupby("group_id").size() == 2).all())

    def test_invalid_winner_duplicate_groups_empty_and_nonfinite_data(self):
        cases = [[], [dict(self.pairs()[0], winner="C")], [self.pairs()[0]] * 2]
        dirty = self.pairs()
        dirty[0]["tour_a"]["price"] = float("nan")
        cases.append(dirty)
        for pairs in cases:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.load(pairs)

    def test_split_rejects_broken_pairs_and_single_group(self):
        data = self.load(self.pairs())
        for invalid in (data.iloc[:1], data.iloc[:2]):
            with self.assertRaises(ValueError):
                train_ltr.split_by_group(invalid)

    def test_dataset_fallback_and_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "part_4060ti_2.jsonl"
            with self.assertRaises(FileNotFoundError):
                train_ltr.resolve_dataset_path(str(requested))
            fallback = requested.with_name("part_4060ti.jsonl")
            fallback.touch()
            self.assertEqual(train_ltr.resolve_dataset_path(str(requested)), fallback)
            requested.touch()
            self.assertEqual(train_ltr.resolve_dataset_path(str(requested)), requested)

    def test_native_training_pool_and_metrics_smoke(self):
        train, test = train_ltr.split_by_group(self.load(self.pairs()), test_size=.4)
        train_pool, test_pool = train_ltr.create_catboost_pool(train), train_ltr.create_catboost_pool(test)
        with tempfile.TemporaryDirectory() as directory:
            model = ltr.CatBoostRanker(iterations=5, depth=2, loss_function="PairLogit", verbose=False,
                                      thread_count=1, allow_writing_files=False, train_dir=directory)
            model.fit(train_pool)
            self.assertEqual(model.feature_names_, FEATURE_COLUMNS)
            for value in train_ltr.evaluate_ndcg(model, test, test_pool).values():
                self.assertTrue(0 <= value <= 1)
