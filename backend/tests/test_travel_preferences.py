import unittest
from types import SimpleNamespace

from app.scripts.run_csai_benchmark import request_from_prediction
from app.services.travel_preferences import parse_preference, prioritize_preference, select_offers


def offer(key, price, destination="sochi", kind="direct"):
    route = SimpleNamespace(key=key, destination=destination,
                            flight={"flight_type": kind, "departure_time": "09:15"})
    return {"route": route, "features": {"price": price, "hotel_stars": 4, "breakfast": 1}}


class PreferenceTest(unittest.TestCase):
    def test_premium_is_distinct_from_trip_budget(self):
        text = "Общий бюджет 180 000 ₽. Предпочитаю прямой, если доплата до 15 000 ₽."
        slots = request_from_prediction({"action": "plan", "nights": 5, "budget_rub": 180000,
                                         "adults_count": 2, "prefer_direct_flight": True}, text)
        self.assertEqual(slots["budget"], 180000)
        self.assertFalse(slots["prefer_direct_flight"])
        self.assertEqual(slots["soft_preference"], {"attribute": "direct", "max_extra_rub": 15000})

    def test_hard_flight_requirement_is_preserved(self):
        text = "Нужен только прямой рейс. Утренний вылет желателен."
        slots = request_from_prediction({"action": "plan", "nights": 5, "budget_rub": 180000,
                                         "adults_count": 2, "prefer_direct_flight": True,
                                         "flight_time_pref": "morning"}, text)
        self.assertTrue(slots["prefer_direct_flight"])
        self.assertIsNone(slots["flight_time_pref"])
        self.assertEqual(slots["soft_preference"], {"attribute": "morning", "max_extra_rub": None})

    def test_numeric_formats_and_foreign_currency(self):
        self.assertEqual(parse_preference("Готов доплатить до 15,5 тыс. рублей за завтрак")["max_extra_rub"], 15500)
        self.assertEqual(parse_preference("Хочу прямой, если доплата до 0 ₽")["max_extra_rub"], 0)
        self.assertIsNone(parse_preference("Хочу прямой, если доплата до 1,5 тыс. долларов"))

    def test_unverified_llm_limit_requires_clarification(self):
        slots = request_from_prediction({
            "action": "plan", "nights": 5, "budget_rub": 180000, "adults_count": 2,
            "soft_preference": {"attribute": "morning", "max_extra_rub": 17000},
        }, "Утренний вылет желателен, но сильно переплачивать не хочется.")
        self.assertTrue(slots["preference_needs_clarification"])
        self.assertNotIn("soft_preference", slots)

    def test_exact_premium_boundary(self):
        cheap = offer("cheap", 100000, kind="connecting")
        direct = offer("direct", 115000)
        self.assertIs(prioritize_preference([cheap, direct], {"attribute": "direct", "max_extra_rub": 15000})[0], direct)
        self.assertIs(prioritize_preference([direct, cheap], {"attribute": "direct", "max_extra_rub": 14999})[0], cheap)

    def test_qualitative_does_not_impute_max_price(self):
        a = offer("a", 120000, kind="connecting")
        b = offer("b", 100000, kind="connecting")
        c = offer("c", 140000)
        self.assertEqual(prioritize_preference([a, b, c], {"attribute": "direct", "max_extra_rub": None}), [a, c, b])

    def test_diversification_retains_first_and_deduplicates_destinations(self):
        a, b, c = (offer(key, 100000) for key in ("a", "b", "c"))
        d = offer("d", 100000, destination="spb")
        self.assertEqual(select_offers([a, b, c, d], diversify=True), [a, d, b])
        self.assertEqual(select_offers([a, b, c], diversify=True), [a, b, c])
        self.assertEqual(select_offers([], diversify=True), [])

    def test_parse_preference_morphology_all_forms(self):
        cases = [
            ("Предпочтителен отель с включенным завтраком.", {"attribute": "breakfast", "max_extra_rub": None}),
            ("Отели более высокой категории предпочтительнее.", {"attribute": "stars", "max_extra_rub": None}),
            ("Предпочтительна первая линия.", None), # not in attributes
            ("Желателен утренний вылет.", {"attribute": "morning", "max_extra_rub": None}),
            ("Хотелось бы прямой авиаперелет.", {"attribute": "direct", "max_extra_rub": None}),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = parse_preference(text)
                if expected is None:
                    self.assertIsNone(res)
                else:
                    self.assertIsNotNone(res)
                    self.assertEqual(res, expected)

    def test_merge_preferences_no_phantom_cap(self):
        text = "Предпочтителен отель с включенным завтраком."
        request = {"archetype": "solo", "soft_preference": None}
        merged = request_from_prediction({"action": "plan", "nights": 5, "budget_rub": 150000, "adults_count": 1}, text)
        self.assertIsNotNone(merged)
        self.assertEqual(merged["soft_preference"], {"attribute": "breakfast", "max_extra_rub": None})

    def test_prioritize_preference_luxury_fallback_preserves_top_rank(self):
        # Luxury 5* hotel without breakfast is ranked #1 (score 0.95, price 500k)
        # 1* hostel without breakfast is cheapest (price 30k)
        # 5* hotel with breakfast costs 560k (surcharge 530k vs cheapest, cap is 20k)
        luxury_top = offer("luxury_top", 500000, destination="uae")
        luxury_top["features"] = {"price": 500000, "hotel_stars": 5, "breakfast": 0, "archetype": "luxury"}
        
        hostel = offer("hostel", 30000, destination="uae")
        hostel["features"] = {"price": 30000, "hotel_stars": 1, "breakfast": 0, "archetype": "luxury"}
        
        luxury_bf = offer("luxury_bf", 560000, destination="uae")
        luxury_bf["features"] = {"price": 560000, "hotel_stars": 5, "breakfast": 1, "archetype": "luxury"}
        
        ranked = [luxury_top, luxury_bf, hostel]
        res = prioritize_preference(ranked, {"attribute": "breakfast", "max_extra_rub": 20000})
        # Since luxury_bf is over cap, the fallback MUST be luxury_top, NOT the 1* hostel!
        self.assertIs(res[0], luxury_top)
        self.assertIsNot(res[0], hostel)
