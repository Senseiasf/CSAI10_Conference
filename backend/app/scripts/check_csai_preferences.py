"""Evaluate preference wiring on frozen Neo4j routes and a separate qualitative test."""
import argparse
import asyncio
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from neo4j import AsyncGraphDatabase
import pandas as pd
from scipy.stats import binomtest

from app.config import settings
from app.scripts.benchmark_routes import read_graph_routes
from app.scripts.run_csai_benchmark import DATA, OUT_DIR, candidate_pool, file_hash, rank
from app.services.ltr_service import LTRService
from app.services.travel_preferences import merge_preferences


async def check(output):
    dataset = DATA / "csai_tradeoff_dataset.json.gz"
    qualitative_file = DATA / "csai_qualitative_holdout.json"
    cases = json.loads(gzip.decompress(dataset.read_bytes()))["pairs"]
    qualitative = {q["pair_id"]: q for q in json.loads(qualitative_file.read_text(encoding="utf-8"))}
    catalog = json.loads((DATA / "destinations.json").read_text(encoding="utf-8"))
    checkpoint = DATA / "catboost_travel_ranker.cbm"
    model = LTRService.load_model(checkpoint)
    if model is None:
        raise RuntimeError("C4 checkpoint could not be loaded")
    async with AsyncGraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    ) as driver:
        graph = await read_graph_routes(driver, catalog)
    route_index = {route.key: route for routes in graph.values() for route in routes}
    rows, soft_rows = [], []
    extraction = Counter()
    seen_qualitative = set()
    for case in cases:
        slots = case["slots"]
        identifiers = {tuple(case[side]["route"]): side.upper() for side in ("a", "b")}
        candidates, _ = candidate_pool([route_index[key] for key in identifiers], slots, strict=True)
        if len(candidates) != 2:
            raise ValueError(f"Pair {case['pair_id']} no longer has two admissible routes")
        if any(c["features"]["price"] != case[identifiers[c["route"].key].lower()]["features"]["price"]
               for c in candidates):
            raise ValueError(f"Price changed for frozen pair {case['pair_id']}")
        for query in case["requests"]:
            request = merge_preferences(slots, query["query"])
            preference = request.get("soft_preference")
            exact = preference == {"attribute": case["benefit"], "max_extra_rub": query["max_extra_rub"]}
            extraction[case["split"]] += int(exact)
            for config, mode in (("C3", "heuristic"), ("C4", "catboost")):
                for policy, applied in (("original", None), ("explicit_premium", preference)):
                    top = rank(candidates, mode, model, applied)[0]
                    side = identifiers[top["route"].key]
                    rows.append({"pair_id": case["pair_id"], "split": case["split"], "config": config,
                                 "policy": policy, "max_extra_rub": query["max_extra_rub"],
                                 "expected": query["winner"], "actual": side, "correct": side == query["winner"]})
        sample = qualitative.get(case["pair_id"])
        if sample:
            if case["split"] != "test":
                raise ValueError("Qualitative pairs must come from held-out destinations")
            seen_qualitative.add(case["pair_id"])
            for config, mode in (("price", "price"), ("C3", "heuristic"), ("C4", "catboost")):
                top = rank(candidates, mode, model)[0]
                side = identifiers[top["route"].key]
                soft_rows.append({"pair_id": case["pair_id"], "config": config, "benefit": case["benefit"],
                                  "expected": sample["winner"], "actual": side,
                                  "correct": None if sample["winner"] == "TIE" else side == sample["winner"]})
    if seen_qualitative != set(qualitative):
        raise ValueError("Missing qualitative pair in frozen routes")

    explicit = pd.DataFrame(rows)
    held_out = explicit.loc[explicit["split"] == "test"]
    explicit_summary = held_out.groupby(["config", "policy"])["correct"].agg(correct="sum", total="count")
    qualitative_results = pd.DataFrame(soft_rows)
    judged = qualitative_results.loc[qualitative_results["expected"] != "TIE"]
    soft_summary = judged.groupby("config")["correct"].agg(correct="sum", total="count")
    paired = defaultdict(dict)
    for row in judged.to_dict(orient="records"):
        paired[row["pair_id"]][row["config"]] = row["correct"]
    c4_only = sum(p["C4"] and not p["C3"] for p in paired.values())
    c3_only = sum(p["C3"] and not p["C4"] for p in paired.values())
    audit = {
        "explicit_extracted": dict(extraction),
        "explicit_requests": dict(Counter(c["split"] for c in cases)),
        "qualitative_ambiguous": sum(q["winner"] == "TIE" for q in qualitative.values()),
        "qualitative_c4_only": c4_only, "qualitative_c3_only": c3_only,
        "qualitative_paired_p": binomtest(c4_only, c4_only + c3_only).pvalue if c4_only + c3_only else 1,
        "explicit_data_sha256": file_hash(dataset), "qualitative_sha256": file_hash(qualitative_file),
        "checkpoint_sha256": file_hash(checkpoint),
    }
    audit["explicit_requests"] = {key: count * 2 for key, count in audit["explicit_requests"].items()}
    output.mkdir(parents=True, exist_ok=True)
    explicit.to_csv(output / "explicit_predictions.csv", index=False)
    explicit_summary.to_csv(output / "explicit_summary.csv")
    qualitative_results.to_csv(output / "qualitative_predictions.csv", index=False)
    soft_summary.to_csv(output / "qualitative_summary.csv")
    (output / "preference_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(explicit_summary.to_string())
    print(soft_summary.to_string())
    print(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR / "preferences")
    asyncio.run(check(parser.parse_args().output_dir))


if __name__ == "__main__":
    main()
