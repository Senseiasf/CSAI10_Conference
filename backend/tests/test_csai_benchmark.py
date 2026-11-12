from dataclasses import replace
import json

import unittest

from app.scripts.benchmark_routes import Route, assess, build_routes, select_return
from app.scripts.run_csai_benchmark import (
    DATA, LABELS, candidate_pool, explicit_destinations, pair_outcome, request_from_prediction,
    retrieve, summary_latex,
)


def make_route(airport="MLE", arrival="15:11", flight_type="direct", layover=None):
    flight = {"from": "SVO", "to": airport, "flight_no": "F1", "flight_type": flight_type,
              "departure_time": "07:00", "arrival_time": arrival, "duration_minutes": 120,
              "price_rub_roundtrip": 10000, "layover_duration_minutes": layover}
    back = {**flight, "from": airport, "to": "SVO", "flight_no": "F2", "flight_type": "direct"}
    hotel = {"id": "H", "stars": 4, "price_per_night_rub": 1000, "pet_friendly": True}
    transfer = {"type": "seaplane", "name": "Boat plane", "price_rub": 1000, "duration_minutes": 40}
    return Route("maldives", hotel, flight, transfer, (back,))


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
