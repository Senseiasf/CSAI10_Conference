"""Controlled paired test of explicit travel preferences on admissible Neo4j routes."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random

from catboost import CatBoostRanker, Pool
from neo4j import AsyncGraphDatabase
import numpy as np
import pandas as pd

from app.config import settings
from app.scripts.benchmark_routes import assess, build_routes, read_graph_routes
from app.scripts.run_csai_benchmark import DATA, OUT_DIR, candidate_pool, file_hash, scores
from app.services.graph_service import minutes_since_midnight
from app.services.ltr_service import ARCHETYPES, CAT_FEATURES, FEATURE_COLUMNS, LTRService


BENEFITS = {
    "direct": "прямой рейс туда",
    "stars": "отель более высокой категории",
    "breakfast": "включённый завтрак",
    "morning": "утренний вылет",
}
PARTIES = {
    "budget": "двое взрослых", "family": "двое взрослых и двое детей",
    "pet": "двое взрослых с собакой", "luxury": "двое взрослых", "solo": "один взрослый",
}
TEST_DESTINATIONS = frozenset(("turkey", "spb", "uae"))
VALIDATION_DESTINATIONS = frozenset(("egypt", "georgia"))
PAIRS_PER_CELL = 4
MIN_PREMIUM = 1500
MAX_PREMIUM = 50000
OUTPUT = OUT_DIR / "tradeoffs"


def comparable(candidate, kind):
    """Group comparable tours and return the attribute that improves with the upgrade."""
    route, feat = candidate["route"], candidate["features"]
    flight = route.flight
    airports = (flight["from"], flight["to"])
    if kind in ("direct", "morning"):
        key = (*airports, route.hotel["id"], route.transfer["type"], route.transfer["name"])
        if kind == "direct":
            return key, int(flight["flight_type"] == "direct")
        depart = minutes_since_midnight(flight["departure_time"])
        return (*key, flight["flight_type"]), int(360 <= depart < 720)
    key = (*airports, flight["flight_no"])
    if kind == "stars":
        return (*key, feat["breakfast"]), feat["hotel_stars"]
    return (*key, feat["hotel_stars"]), feat["breakfast"]


def select_tradeoffs(candidates, kind, seed, used):
    groups = defaultdict(lambda: defaultdict(list))
    for candidate in sorted(candidates, key=lambda c: c["route"].key):
        key, value = comparable(candidate, kind)
        groups[key][value].append(candidate)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    picked = []
    # First diversify by comparable group. Each returned pair has two distinct routes.
    for key in keys:
        by_value = groups[key]
        if len(by_value) < 2:
            continue
        levels = sorted(by_value)
        combinations = [(lo, hi) for lo in levels for hi in levels if lo < hi]
        rng.shuffle(combinations)
        pair = None
        for lo, hi in combinations:
            expensive = rng.sample(by_value[hi], len(by_value[hi]))
            for premium in expensive:
                price = premium["features"]["price"]
                affordable = [
                    c for c in by_value[lo]
                    if MIN_PREMIUM <= price - c["features"]["price"] <= MAX_PREMIUM
                    and (c["route"].key, premium["route"].key) not in used
                ]
                if affordable:
                    pair = (rng.choice(affordable), premium)
                    break
            if pair:
                break
        if pair:
            used.add(tuple(c["route"].key for c in pair))
            picked.append(pair)
        if len(picked) == PAIRS_PER_CELL:
            break
    return picked


def snapshot(candidate, upgrade):
    route, back = candidate["route"], candidate["back"]
    return {
        "route": route.key, "hotel": route.hotel["name"],
        "outbound": route.flight["flight_no"], "outbound_departure": route.flight["departure_time"],
        "return": back["flight_no"], "return_departure": back["departure_time"],
        "transfer_capacity_known": route.transfer.get("capacity", route.transfer.get("max_passengers")) is not None,
        "has_upgrade": upgrade, "features": candidate["features"],
    }


def make_case(dest, arch, kind, index, pair, request):
    cheap, upgrade = pair
    gap = upgrade["features"]["price"] - cheap["features"]["price"]
    budget = int(np.ceil(upgrade["features"]["price"] / 5000) * 5000)
    key = f"{dest['id']}/{arch}/{kind}/{index}"
    rng = random.Random(f"limits/v1/{key}")
    low = int(gap * rng.uniform(.2, .85) // 500 * 500)
    high = max(int(np.ceil(gap)), int(gap * rng.uniform(1.15, 1.9) // 500 * 500))
    slots = {**request, "budget": budget}
    for c in pair:
        log, con, _ = assess(c["route"], slots, c["back"])
        if log or con:
            raise ValueError(f"Trade-off pair must be admissible: {key} {log} {con}")
    left, right = snapshot(cheap, False), snapshot(upgrade, True)
    if rng.choice((False, True)):
        left, right = right, left
    prompts = []
    for limit in (low, high):
        want_upgrade = limit >= gap
        winner = "A" if left["has_upgrade"] == want_upgrade else "B"
        prompts.append({
            "query": f"Поездка в регион «{dest['name']}» на {slots['days']} ночей, "
                     f"{PARTIES[arch]}. Общий лимит {budget} ₽. Из этих двух допустимых "
                     f"туров выберу более дешёвый, если доплата за {BENEFITS[kind]} выше "
                     f"{limit} ₽. Если доплата не превышает эту сумму, выберу тур с этим "
                     "преимуществом. Остальные различия для меня несущественны.",
            "max_extra_rub": limit, "winner": winner,
        })
    # Keep both requests for the same A/B routes together in every split.
    split = "test" if dest["id"] in TEST_DESTINATIONS else "validation" if dest["id"] in VALIDATION_DESTINATIONS else "train"
    return {"pair_id": key, "destination": dest["id"], "archetype": arch, "benefit": kind,
            "split": split, "slots": slots, "premium_rub": gap,
            "a": left, "b": right, "requests": prompts}


async def prepare(output):
    catalog_path = DATA / "destinations.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    async with AsyncGraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    ) as driver:
        graph = await read_graph_routes(driver, catalog)
    reference = build_routes(catalog)
    cases = []
    used = set()
    for dest in catalog["destinations"]:
        identifier = dest["id"]
        if sorted(r.key for r in graph[identifier]) != sorted(r.key for r in reference[identifier]):
            raise RuntimeError(f"Neo4j paths differ from the frozen catalogue for {identifier}")
        for arch in ARCHETYPES:
            request = {"archetype": arch, "days": 5, "adults_count": 1 if arch == "solo" else 2,
                       "kids_count": 2 if arch == "family" else 0,
                       "has_kids": arch == "family", "has_pets": arch == "pet"}
            pool, _ = candidate_pool(graph[identifier], request, strict=True)
            for kind in BENEFITS:
                pairs = select_tradeoffs(pool, kind, f"{identifier}/{arch}/{kind}", used)
                cases.extend(make_case(dest, arch, kind, i, pair, request) for i, pair in enumerate(pairs, 1))
    output.mkdir(parents=True, exist_ok=True)
    dataset = {"protocol": "explicit-willingness-to-pay-v1", "catalog_sha256": file_hash(catalog_path),
               "pairs": cases}
    (output / "tradeoff_dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Frozen {len(cases)} base pairs before inference", flush=True)


def control_pool(cases, with_preference, include_labels=True):
    features, labels, groups = [], [], []
    for case in cases:
        for request in case["requests"]:
            group = groups[-1] + 1 if groups else 0
            for side in ("a", "b"):
                candidate = case[side]
                row = {col: candidate["features"][col] for col in FEATURE_COLUMNS}
                row.update(benefit=case["benefit"], upgrade=int(candidate["has_upgrade"]),
                           premium_thousands=case["premium_rub"] / 1000)
                if with_preference:
                    row["max_extra_thousands"] = request["max_extra_rub"] / 1000
                features.append(row)
                labels.append(int(request["winner"] == side.upper()))
                groups.append(group)
    return Pool(pd.DataFrame(features), label=labels if include_labels else None, group_id=groups,
                cat_features=[*CAT_FEATURES, "benefit"], thread_count=2)


def preferred(score_a, score_b, case):
    if abs(score_a - score_b) < 1e-8:
        return "A" if case["a"]["features"]["price"] < case["b"]["features"]["price"] else "B"
    return "A" if score_a > score_b else "B"


def validate_partitions(dataset):
    partitions = defaultdict(list)
    owners = {}
    identifiers = set()
    for case in dataset["pairs"]:
        pair_id = case["pair_id"]
        if pair_id in identifiers:
            raise ValueError(f"Repeated pair ID: {pair_id}")
        identifiers.add(pair_id)
        split = case["split"]
        destination = case["destination"]
        if owners.setdefault(destination, split) != split:
            raise ValueError(f"Destination leaks across splits: {destination}")
        a, b = case["a"], case["b"]
        if a["route"] == b["route"] or a["has_upgrade"] == b["has_upgrade"]:
            raise ValueError(f"Pair has no contrasting routes: {pair_id}")
        prices = [c["features"]["price"] for c in (a, b)]
        upgrade_side = "A" if a["has_upgrade"] else "B"
        gap = prices[0] - prices[1] if upgrade_side == "A" else prices[1] - prices[0]
        if gap != case["premium_rub"] or gap <= 0 or max(prices) > case["slots"]["budget"]:
            raise ValueError(f"Invalid pair pricing: {pair_id}")
        requests = case["requests"]
        if len(requests) != 2 or {p["winner"] for p in requests} != {"A", "B"}:
            raise ValueError(f"Missing counterfactual: {pair_id}")
        for request in requests:
            if (request["winner"] == upgrade_side) != (request["max_extra_rub"] >= gap):
                raise ValueError(f"Label disagrees with the stated preference: {pair_id}")
        partitions[split].append(case)
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("Trade-off dataset requires three nonempty destination-disjoint splits")
    return partitions


def evaluate(output, dataset_path=None):
    dataset_path = dataset_path or output / "tradeoff_dataset.json"
    dataset_bytes = dataset_path.read_bytes()
    if dataset_path.suffix == ".gz":
        dataset_bytes = gzip.decompress(dataset_bytes)
    dataset = json.loads(dataset_bytes)
    partitions = validate_partitions(dataset)
    test = partitions["test"]
    checkpoint = DATA / "catboost_travel_ranker.cbm"
    model = LTRService.load_model(checkpoint)
    if model is None:
        raise RuntimeError("Current C4 checkpoint failed to load")
    original_input = [case[side] for case in test for side in ("a", "b")]
    original = {name: scores(original_input, mode, model) for name, mode in
                (("price", "price"), ("C3", "heuristic"), ("C4", "catboost"))}
    controls = {}
    trees = {}
    for context in (False, True):
        name = "fresh_catboost_with_priority" if context else "fresh_catboost_without_priority"
        ranker = CatBoostRanker(loss_function="PairLogit", eval_metric="PairAccuracy",
                               iterations=300, depth=5, learning_rate=.08, random_seed=42,
                               thread_count=2, verbose=False, allow_writing_files=False)
        ranker.fit(control_pool(partitions["train"], context),
                   eval_set=control_pool(partitions["validation"], context), use_best_model=True)
        controls[name] = ranker.predict(control_pool(test, context, include_labels=False))
        trees[name] = ranker.tree_count_
    rows = []
    for i, case in enumerate(test):
        for j, request in enumerate(case["requests"]):
            idx = (i * 2 + j) * 2
            predictions = {name: preferred(values[2*i], values[2*i+1], case)
                           for name, values in original.items()}
            predictions.update({name: preferred(values[idx], values[idx+1], case)
                                for name, values in controls.items()})
            rows.extend({
                "pair_id": case["pair_id"], "destination": case["destination"], "benefit": case["benefit"],
                "archetype": case["archetype"], "max_extra_rub": request["max_extra_rub"],
                "premium_wins": request["max_extra_rub"] >= case["premium_rub"],
                "reference": request["winner"], "ranker": name, "prediction": pred,
                "correct": pred == request["winner"],
            } for name, pred in predictions.items())
    frame = pd.DataFrame(rows)
    metrics = frame.groupby("ranker")["correct"].agg(correct="sum", requests="count", accuracy="mean")
    by_case = frame.groupby(["ranker", "pair_id"])["correct"].sum()
    both = by_case.eq(2).groupby("ranker").sum()
    summary = metrics.assign(accuracy_pct=(metrics["accuracy"] * 100).round(2),
                             both_requests_correct=both).drop(columns="accuracy")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "tradeoff_predictions.csv", index=False)
    summary.to_csv(output / "tradeoff_summary.csv")
    frame.groupby(["ranker", "benefit", "premium_wins"])["correct"].agg(["sum", "count", "mean"]).to_csv(
        output / "tradeoff_slices.csv"
    )
    (output / "tradeoff_run.json").write_text(json.dumps({
        "dataset_sha256": sha256(dataset_bytes).hexdigest(),
        "checkpoint_sha256": file_hash(checkpoint), "control_trees": trees,
        "splits": {name: {"base_pairs": len(cases), "destinations": sorted({c["destination"] for c in cases})}
                   for name, cases in partitions.items()},
    }, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--dataset", type=Path, help="Frozen JSON or JSON.gz input for evaluate")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.dataset:
            parser.error("--dataset is only available for evaluate")
        asyncio.run(prepare(args.output_dir))
    else:
        evaluate(args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
