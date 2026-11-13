from dataclasses import replace
import json

import unittest

from app.scripts.benchmark_routes import Route, assess, build_routes, select_return
from app.scripts.run_csai_benchmark import (
    DATA, LABELS, candidate_pool, explicit_destinations, pair_outcome, request_from_prediction,
    retrieve, summary_latex,
)


def make_route(airport="MLE", arrival="15:11", flight_type="direct", layover=None, destination="maldives"):
    flight = {"from": "SVO", "to": airport, "flight_no": "F1", "flight_type": flight_type,
              "departure_time": "07:00", "arrival_time": arrival, "duration_minutes": 120,
              "price_rub_roundtrip": 10000, "layover_duration_minutes": layover}
    back = {**flight, "from": airport, "to": "SVO", "flight_no": "F2", "flight_type": "direct"}
    hotel = {"id": "H", "stars": 4, "price_per_night_rub": 1000, "pet_friendly": True}
    transfer = {"type": "seaplane", "name": "Boat plane", "price_rub": 1000, "duration_minutes": 40}
    return Route(destination, hotel, flight, transfer, (back,))


REQUEST = {"days": 5, "adults_count": 2, "kids_count": 0, "budget": 30000,
           "archetype": "budget"}


class BenchmarkTest(unittest.IsolatedAsyncioTestCase):
    def test_seaplane_uses_airport_ready_time(self):
        for arrival, expected in [("15:10", False), ("15:11", True)]:
            with self.subTest(arrival=arrival):
                r = make_route(arrival=arrival)
                self.assertEqual("seaplane_closed" in assess(r, REQUEST, select_return(r))[0], expected)


    def test_bridge_applies_only_to_the_crossing_and_capacity_to_the_whole_party(self):
        r = make_route(airport="LED", arrival="00:20")
        plain = replace(r, transfer={**r.transfer, "type": "taxi"})
        assert "bridge_closed" not in assess(plain, REQUEST, select_return(plain))[0]
        bridge = replace(plain, transfer={**plain.transfer, "capacity": 2, "bridge_crossing_minutes": 10,
                                         "bridge_closed_from": "01:10", "bridge_closed_until": "05:00"})
        assert "bridge_closed" in assess(bridge, REQUEST, select_return(bridge))[0]
        assert "transfer_capacity" in assess(bridge, {**REQUEST, "kids_count": 1}, select_return(bridge))[0]


    def test_strict_modes_do_not_rank_rejected_candidates(self):
        bad = make_route(arrival="15:00", flight_type="connecting", layover=59)
        good = make_route(arrival="15:00", flight_type="connecting", layover=60)
        candidates, reasons = candidate_pool([bad, good], REQUEST, strict=True)
        assert [c["route"] for c in candidates] == [good]
        assert reasons["layover_under_60min"] == 1
        assert not candidate_pool([bad], REQUEST, strict=True)[0]
        assert candidate_pool([bad], REQUEST, strict=False)[0]
        assert not candidate_pool([replace(good, returns=())], REQUEST, strict=True)[0]
        assert not candidate_pool([good], {**REQUEST, "prefer_direct_flight": True}, strict=True)[0]


    def test_budget_accounts_for_people_and_rooms(self):
        r = make_route(arrival="15:00")
        log, constraints, price = assess(r, {**REQUEST, "rooms_count": 2}, select_return(r))
        assert log == set()
        assert price == 31000
        assert constraints == {"budget_exceeded"}


    def test_catalog_routes_are_connected_at_both_ends(self):
        catalog = json.loads((DATA / "destinations.json").read_text(encoding="utf-8"))
        routes = [r for group in build_routes(catalog).values() for r in group]
        assert len(routes) > 1000
        assert all(r.flight["to"] == r.transfer["from"] and r.hotel["id"] == r.transfer["to"] for r in routes)
        assert all((f["from"], f["to"]) == (r.flight["to"], r.flight["from"]) for r in routes for f in r.returns)


    async def test_explicit_dubai_beats_vector_search_and_lexical_abstains_without_match(self):
        catalog = json.loads((DATA / "destinations.json").read_text(encoding="utf-8"))["destinations"]
        assert await retrieve("Хочу в Дубай с детьми", catalog, {}, None, True) == ["uae"]
        assert explicit_destinations("Летим из Дубая в Сочи", catalog) == ["sochi"]
        assert await retrieve("Тёплое море и горы", catalog, {}, None, False) == []


    def test_confusion_matrix_uses_stable_pair_order_with_both_reference_classes(self):
        assert pair_outcome("A", 2, 1, 1, 1) == "TP"
        assert pair_outcome("A", 1, 2, 1, 1) == "FN"
        assert pair_outcome("B", 2, 1, 1, 1) == "FP"
        assert pair_outcome("B", 1, 2, 1, 1) == "TN"
        assert pair_outcome("B", 1, 1, 1, 1) == "Tie_B"


    def test_nlu_slots_are_mapped_without_reference_values(self):
        slots = request_from_prediction({"action": "plan", "nights": 5, "budget_rub": 30000,
                                         "adults_count": 2, "kids_count": 1, "flight_time_pref": "day"})
        assert slots["budget"] == 30000 and slots["kids_count"] == 1
        assert slots["flight_time_pref"] == "day"
        assert request_from_prediction({"action": "plan", "nights": 5}) is None


    def test_latex_uses_same_metric_columns_as_csv(self):
        row = {key: 0 for key in LABELS}
        rendered = summary_latex([row], LABELS)
        table_rows = [line for line in rendered.splitlines() if "&" in line]
        assert all(line.count("&") == len(row) - 1 for line in table_rows)
        assert "Success@3" in rendered

    def test_rank_is_symmetric_across_all_modes(self):
        from app.scripts.run_csai_benchmark import rank
        # Create candidates where A is cheap without pref, B is slightly more expensive with pref
        r_a = make_route(flight_type="connecting")
        r_b = make_route(flight_type="direct")
        cand_a = {"route": r_a, "back": None, "features": {"price": 50000, "hotel_stars": 3, "breakfast": 0, "pet_friendly": 0, "archetype": "solo", "days": 5, "transfers": 2, "layover_hours": 2.0, "dist_center_km": 1.0, "price_delta_vs_cheapest": 0.0, "relative_surcharge_pct": 0.0}}
        cand_b = {"route": r_b, "back": None, "features": {"price": 55000, "hotel_stars": 3, "breakfast": 0, "pet_friendly": 0, "archetype": "solo", "days": 5, "transfers": 1, "layover_hours": 0.0, "dist_center_km": 1.0, "price_delta_vs_cheapest": 5000.0, "relative_surcharge_pct": 0.10}}
        candidates = [cand_a, cand_b]
        pref = {"attribute": "direct", "max_extra_rub": 10000}

        for mode in ("price", "heuristic"):
            pure = rank(candidates, mode, model=None, preference=None)
            self.assertEqual(len(pure), 2)
            with_pref = rank(candidates, mode, model=None, preference=pref)
            self.assertIs(with_pref[0], cand_b)

    def test_offer_is_success_multi_destination_localized_baseline(self):
        from app.scripts.run_csai_benchmark import offer_is_success
        r_spb = make_route(destination="spb", airport="LED", flight_type="direct")
        r_geo_base = make_route(destination="georgia", airport="TBS", flight_type="connecting", layover=120)
        r_geo_pref = make_route(destination="georgia", airport="TBS", flight_type="direct")

        cand_spb = {"route": r_spb, "back": r_spb.returns[0], "features": {"price": 20000, "hotel_stars": 2, "breakfast": 0, "archetype": "solo"}}
        cand_geo_base = {"route": r_geo_base, "back": r_geo_base.returns[0], "features": {"price": 100000, "hotel_stars": 3, "breakfast": 0, "archetype": "solo"}}
        cand_geo_pref = {"route": r_geo_pref, "back": r_geo_pref.returns[0], "features": {"price": 110000, "hotel_stars": 3, "breakfast": 0, "archetype": "solo"}}

        pool = [cand_spb, cand_geo_base, cand_geo_pref]
        ref = {
            "expected_destinations": ["spb", "georgia"],
            "days": 5, "adults_count": 1, "kids_count": 0, "budget": 200000, "archetype": "solo",
            "soft_preference": {"attribute": "direct", "max_extra_rub": 20000},
        }
        self.assertTrue(offer_is_success(cand_geo_pref, ref, candidate_pool=pool))

    def test_offer_is_success_strict_false_invalid_candidates(self):
        from app.scripts.run_csai_benchmark import offer_is_success
        r_broken = Route("sochi", {"id": "h_broken", "stars": 1, "price_per_night_rub": 500},
                         {"from": "SVO", "to": "AER", "flight_no": "F_BROKEN", "flight_type": "direct", "departure_time": "08:00"},
                         {"type": "bus", "name": "Bus", "price_rub": 500}, ())
        r_valid_base = make_route(airport="AER", flight_type="connecting", layover=120, destination="sochi")
        r_valid_pref = make_route(airport="AER", flight_type="direct", destination="sochi")

        cand_broken = {"route": r_broken, "back": None, "features": {"price": 10000, "hotel_stars": 1, "breakfast": 0, "archetype": "solo"}}
        cand_valid_base = {"route": r_valid_base, "back": r_valid_base.returns[0], "features": {"price": 50000, "hotel_stars": 4, "breakfast": 0, "archetype": "solo"}}
        cand_valid_pref = {"route": r_valid_pref, "back": r_valid_pref.returns[0], "features": {"price": 60000, "hotel_stars": 4, "breakfast": 0, "archetype": "solo"}}

        pool = [cand_broken, cand_valid_base, cand_valid_pref]
        ref = {
            "expected_destinations": ["sochi"],
            "days": 5, "adults_count": 2, "kids_count": 0, "budget": 100000, "archetype": "solo",
            "soft_preference": {"attribute": "direct", "max_extra_rub": 15000},
        }
        self.assertTrue(offer_is_success(cand_valid_pref, ref, candidate_pool=pool))
