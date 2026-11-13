"""Evaluate C1–C4 against a frozen catalogue and a live Neo4j snapshot."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections import Counter
from contextlib import AsyncExitStack
from pathlib import Path
import random
import re
import sys
import time

import asyncpg
from catboost import Pool
from neo4j import AsyncGraphDatabase
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import settings
from app.scripts.benchmark_routes import assess, build_routes, read_graph_routes, select_return
from app.services.llm_service import LLMService
from app.services.ltr_service import (
    ARCHETYPES,
    CAT_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    CAT_FEATURES_V1,
    CAT_FEATURES_V2,
    FEATURE_SCHEMA_V1,
    FEATURE_SCHEMA_V2,
    LTRService,
)
from app.services.travel_preferences import matches_preference, merge_preferences, prioritize_preference, select_offers
from app.services.vector_service import EMBEDDING_VERSION, generate_text_embedding

DATA = BACKEND / "app" / "data"
OUT_DIR = (BACKEND.parent if (BACKEND.parent / "docker-compose.yml").exists() else BACKEND) / "experiment_outputs"
CONFIGS = (
    ("C1_Baseline_SQL", "C1: Lexical / Price", False, False, "price"),
    ("C2_Vector_RAG", "C2: Hybrid / Price", True, False, "price"),
    ("C2G_Graph_Price", "C2-G: Graph / Price", True, True, "price"),
    ("C3_Graph_Heuristic", "C3: Graph / Heuristic", True, True, "heuristic"),
    ("C4_Full_CatBoost", "C4: Graph / CatBoost", True, True, "catboost"),
)
# Morphological variants of catalogue toponyms, independent of the evaluation set.
ALIASES = {
    "uae": r"оаэ|эмират(?:ы|ах|ов)?|дуба(?:й|я|е|ю|ем)|абу[- ]даби|шардж[аиуе]",
    "sochi": r"сочи|адлер(?:е|а)?|сириус(?:е|а)?|красн\w+ полян\w+",
    "spb": r"спб|питер(?:е|а)?|(?:санкт[- ]?)?петербург(?:е|а)?",
    "altai": r"алта[йеяю]|горно[- ]алтайск(?:е|а)?",
    "turkey": r"турци[яюие]|анталь[яюие]|стамбул(?:е|а)?|кемер(?:е|а)?|белек(?:е|а)?",
    "thailand": r"(?:таи|тай)ланд(?:е|а)?|пхукет(?:е|а)?|бангкок(?:е|а)?|самуи",
    "egypt": r"египет|египт[аеу]|хургад[аыуе]|шарм(?:[- ]эль[- ]шейх)?",
    "georgia": r"грузи[яюие]|тбилиси|батуми|гудаури|казбеги",
    "maldives": r"мальдив[ыаху]*|мале",
}
LOG = logging.getLogger(__name__)


def explicit_destinations(text, destinations):
    text = text.casefold().replace("ё", "е")
    matches = []
    countries = Counter(d["country"] for d in destinations)
    for d in destinations:
        names = d["cities"] + ([d["country"]] if countries[d["country"]] == 1 else [])
        aliases = [re.escape(x.casefold().replace("ё", "е")) for x in names]
        pattern = "|".join([ALIASES.get(d["id"], "(?!)"), *aliases])
        for match in re.finditer(r"(?<!\w)(?:" + pattern + r")(?!\w)", text):
            if re.search(r"(?:\bиз|\bне|\bне в|\bкроме)\s+$", text[:match.start()]):
                continue
            matches.append((match.start(), d["id"]))
    return list(dict.fromkeys(identifier for _, identifier in sorted(matches)))


async def retrieve(text, destinations, embeddings, pg, vector, explicit=None):
    named = explicit_destinations(text, destinations)
    if named:
        return named[:2]
    if explicit and explicit in embeddings:
        return [explicit]
    if not vector or not re.search(r"\w", text):
        return []
    query = generate_text_embedding(text)
    if pg is not None:
        vector_str = "[" + ",".join(map(str, query)) + "]"
        rows = await pg.fetch(
            """SELECT id FROM destinations WHERE id = ANY($2::text[])
               AND metadata_json->>'embedding_version' = $3 AND embedding IS NOT NULL
               ORDER BY embedding <=> $1::text::vector, id LIMIT 2""",
            vector_str, list(embeddings), EMBEDDING_VERSION,
        )
        if len(rows) != min(2, len(embeddings)):
            raise RuntimeError("pgvector snapshot incomplete. Seed the benchmark catalogue and model version.")
        return [row["id"] for row in rows]
    return sorted(embeddings, key=lambda identifier: (-float(np.dot(query, embeddings[identifier])), identifier))[:2]


def request_from_prediction(pred, text=""):
    """Convert tool slots to the benchmark schema. Missing essential slots require clarification."""
    if not pred or pred.get("action") != "plan":
        return None
    days = pred.get("nights")
    budget = pred.get("budget_rub")
    adults, kids = pred.get("adults_count"), pred.get("kids_count")
    if not isinstance(days, int) or days <= 0 or not isinstance(budget, int) or budget <= 0:
        return None
    if not isinstance(adults, int) or adults <= 0:
        return None
    if kids is None:
        if pred.get("family_friendly"):
            return None
        kids = 0
    if not isinstance(kids, int) or kids < 0:
        return None
    profile = {
        "has_pets": bool(pred.get("pet_friendly")), "has_kids": kids > 0 or bool(pred.get("family_friendly")),
        "interests": pred.get("tags") or [], "budget_rub": budget, "adults_count": adults,
    }
    request = {**profile, "budget": budget, "days": days, "kids_count": kids,
            "rooms_count": pred.get("rooms_count") or 1,
            "prefer_direct_flight": bool(pred.get("prefer_direct_flight")),
            "private_room_required": bool(pred.get("private_room_required")),
            "flight_time_pref": pred.get("flight_time_pref"), "origin_airport": pred.get("origin_airport"),
            "destination_id": pred.get("destination_id"), "destination_query": pred.get("destination_query"),
            "archetype": LTRService.resolve_archetype(profile)}
    return merge_preferences(request, text, pred.get("soft_preference"), from_model=True)


def nlu_slot_metrics(queries, predictions):
    fields = {
        "days": "nights", "budget": "budget_rub", "adults_count": "adults_count",
        "kids_count": "kids_count", "has_pets": "pet_friendly", "has_kids": "family_friendly",
        "prefer_direct_flight": "prefer_direct_flight", "flight_time_pref": "flight_time_pref",
    }
    metrics = {}
    for reference, predicted in fields.items():
        expected = [(q[reference], (predictions.get(q["id"]) or {}).get(predicted))
                    for q in queries if q.get(reference) is not None]
        correct = sum(a == b for a, b in expected)
        metrics[reference] = {"correct": correct, "evaluated": len(expected),
                              "accuracy_pct": percent(correct, len(expected))}
    return metrics


def features(route, request, price, min_price=None):
    schema = LTRService.get_feature_schema()
    if schema == FEATURE_SCHEMA_V2:
        return LTRService.extract_candidate_features_v2(
            hotel=route.hotel,
            destination_id=route.destination,
            profile=request,
            nights=request.get("days", 7),
            logistics={"flight": route.flight, "timeline": {}},
            breakdown={"mandatory": {"total_mandatory_rub": price}},
            reference_price=min_price,
        )
    flight = route.flight
    days = request["days"]
    per_day = price / days
    stars = LTRService._parse_stars(route.hotel["stars"])
    layover = (flight.get("layover_duration_minutes") or 0) / 60
    return {
        "archetype": request["archetype"],
        "cities": LTRService.resolve_canonical_cities(route.destination, route.hotel.get("city")),
        "days": days, "price": price, "log_price": np.log1p(price),
        "log_price_per_day": np.log1p(per_day),
        "transfers": 1 + int(flight.get("flight_type") != "direct"), "layover_hours": layover,
        "hotel_stars": stars, "dist_center_km": LTRService._parse_dist_center_km(route.hotel),
        "pet_friendly": LTRService._parse_pet_friendly(route.hotel),
        "breakfast": LTRService._parse_breakfast(route.hotel),
        "stars_per_10k": stars / max(0.1, per_day / 10000),
    }


def candidate_pool(routes, request, strict):
    candidates, rejected = [], Counter()
    returns = {}
    valid_assessed = []
    for route in routes:
        flight = route.flight
        key = (flight["from"], flight["to"], flight["flight_no"])
        if key not in returns:
            returns[key] = select_return(route, request.get("prefer_direct_flight"))
        back = returns[key]
        logistics, constraints, price = assess(route, request, back)
        rejected.update(logistics | constraints)
        if price is None or (strict and (logistics or constraints)):
            continue
        valid_assessed.append((route, back, price, not (logistics or constraints)))
    
    admissible_prices = [price for _, _, price, is_admissible in valid_assessed if is_admissible]
    pool_min_price = min(admissible_prices) if admissible_prices else min((price for _, _, price, _ in valid_assessed), default=None)

    for route, back, price, _ in valid_assessed:
        candidates.append({"route": route, "back": back, "features": features(route, request, price, pool_min_price)})
    return candidates, rejected


def scores(candidates, mode, model):
    feats = [c["features"] for c in candidates]
    if not feats:
        return []
    if mode == "price":
        return [-f["price"] for f in feats]
    if mode == "heuristic":
        return [LTRService._surrogate_score(f) for f in feats]
    schema = LTRService.get_feature_schema()
    cols = FEATURE_COLUMNS_V2 if schema == FEATURE_SCHEMA_V2 else FEATURE_COLUMNS_V1
    cat_cols = CAT_FEATURES_V2 if schema == FEATURE_SCHEMA_V2 else CAT_FEATURES_V1
    return model.predict(Pool(
        [[f.get(name, 0.0) for name in cols] for f in feats],
        cat_features=cat_cols, feature_names=cols, thread_count=2,
    ), thread_count=2)


def rank(candidates, mode, model, preference=None):
    ranked = [candidate for candidate, _ in sorted(
        zip(candidates, scores(candidates, mode, model)),
        key=lambda item: (-item[1], item[0]["features"]["price"], item[0]["route"].key),
    )]
    if preference:
        return prioritize_preference(ranked, preference)
    return ranked


def offer_is_success(candidate, reference, candidate_pool=None):
    route = candidate["route"]
    if route.destination not in reference["expected_destinations"]:
        return False
    logistics, constraints, _ = assess(route, reference, candidate["back"])
    if logistics or constraints:
        return False

    pref = reference.get("soft_preference")
    if pref and candidate_pool:
        attr = pref.get("attribute")
        cap = pref.get("max_extra_rub")
        dest = route.destination
        admissible_dest = []
        for c in candidate_pool:
            if c["route"].destination == dest:
                log, con, _ = assess(c["route"], reference, c["back"])
                if not log and not con:
                    admissible_dest.append(c)
        if cap is not None and admissible_dest:
            cheapest = min(admissible_dest, key=lambda c: (c["features"]["price"], c["route"].key))
            min_p = cheapest["features"]["price"]
            has_matching = any(
                matches_preference(c, attr, cheapest) and (c["features"]["price"] - min_p <= cap)
                for c in admissible_dest
            )
            if has_matching:
                if not matches_preference(candidate, attr, cheapest):
                    return False
                if candidate["features"]["price"] - min_p > cap:
                    return False
    return True


def pareto_winner(a, b):
    """Diagnostic labels based on objective dominance among admissible options."""
    def utility(f):
        return (-f["price"], -f["layover_hours"], -f["transfers"], -f["dist_center_km"],
                f["hotel_stars"], f["breakfast"], f["pet_friendly"])
    u, v = utility(a), utility(b)
    if all(x >= y for x, y in zip(u, v)) and u != v:
        return "A"
    if all(y >= x for x, y in zip(u, v)) and u != v:
        return "B"
    return None


def pair_candidate(candidate):
    route = candidate["route"]
    return {**candidate["features"], "flight_departure": route.flight["departure_time"],
            "return_departure": candidate["back"]["departure_time"], "hotel_id": route.hotel["id"],
            "transfer_type": route.transfer["type"], "transfer_minutes": route.transfer["duration_minutes"]}


async def judge_pair(client, query, a, b):
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "Compare two admissible travel packages using the user's preferences. Reply exactly A, B or TIE. Do not infer absent amenities."},
            {"role": "user", "content": json.dumps({
                "query": query["query"], "requested_departure_window": query.get("flight_time_pref"),
                "A": pair_candidate(a), "B": pair_candidate(b),
            }, ensure_ascii=False)},
        ],
    )
    choice = (response.choices[0].message.content or "").strip().upper()
    if choice not in {"A", "B", "TIE"}:
        raise ValueError(f"Invalid pair judgment for {query['id']}")
    return None if choice == "TIE" else choice


async def build_pairs(queries, graph, client):
    pairs = []
    for q in queries:
        routes = [route for dest in q["expected_destinations"] for route in graph[dest]]
        pool, _ = candidate_pool(routes, q, True)
        pool.sort(key=lambda c: c["route"].key)
        rng = random.Random(f"csai-pairs-v2:{q['id']}")
        if len(pool) < 2:
            continue
        # A/B orientation is chosen before labeling, and recorded verbatim.
        for _ in range(1 if client else 128):
            a, b = rng.sample(pool, 2)
            winner = await judge_pair(client, q, a, b) if client else pareto_winner(a["features"], b["features"])
            if winner:
                pairs.append({"query_id": q["id"], "archetype": q["archetype"], "a": a, "b": b, "winner": winner})
                break
    return pairs


def pair_outcome(true_winner, score_a, score_b, price_a, price_b):
    if abs(score_a - score_b) > 1e-5:
        pred = "A" if score_a > score_b else "B"
    elif price_a != price_b:
        pred = "A" if price_a < price_b else "B"
    else:
        return "Tie_A" if true_winner == "A" else "Tie_B"
    return {("A", "A"): "TP", ("B", "A"): "FP", ("B", "B"): "TN", ("A", "B"): "FN"}[true_winner, pred]


def percent(numerator, denominator):
    return round(100 * numerator / denominator, 2) if denominator else None


def ranking_metrics(pairs, mode, model):
    predicted = scores([p[side] for p in pairs for side in ("a", "b")], mode, model)
    judgments = []
    for i, p in enumerate(pairs):
        sa, sb = map(float, predicted[2*i:2*i+2])
        outcome = pair_outcome(p["winner"], sa, sb, p["a"]["features"]["price"], p["b"]["features"]["price"])
        # Report NDCG using the same effective order as Pairwise Accuracy.
        order = {"TP": [1, 0], "TN": [0, 1], "FP": [1, 0], "FN": [0, 1]}.get(outcome, [0, 0])
        ndcg = ndcg_score([[int(p["winner"] == "A"), int(p["winner"] == "B")]], [order], k=2)
        judgments.append({"query_id": p["query_id"], "archetype": p["archetype"], "outcome": outcome,
                          "winner": p["winner"], "score_a": sa, "score_b": sb, "ndcg": ndcg})
    aggregates = {}
    for arch in ("ALL", *ARCHETYPES):
        subset = [p for p in judgments if arch == "ALL" or p["archetype"] == arch]
        counts = Counter(p["outcome"] for p in subset)
        tp, fp, tn, fn = (counts[k] for k in ("TP", "FP", "TN", "FN"))
        aggregates[arch] = {
            **{k: counts[k] for k in ("TP", "FP", "TN", "FN", "Tie_A", "Tie_B")},
            "total": len(subset),
            "pairwise_acc": percent(tp + tn + (counts["Tie_A"] + counts["Tie_B"]) / 2, len(subset)),
            "precision_a": percent(tp, tp + fp), "recall_a": percent(tp, tp + fn + counts["Tie_A"]),
            "ndcg_at_2": round(float(np.mean([p["ndcg"] for p in subset])), 4) if subset else None,
        }
    return aggregates, judgments


def summary_latex(rows, labels):
    def cell(value):
        if value is None:
            return "--"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("&", r"\&").replace("_", r"\_")
    body = [" & ".join(cell(row[key]) for key in labels) + r" \\" for row in rows]
    return "\n".join([
        r"\begin{tabular}{l" + "r" * (len(labels) - 1) + "}",
        r"\hline", " & ".join(labels.values()) + r" \\", r"\hline", *body, r"\hline", r"\end{tabular}",
    ]) + "\n"


LABELS = {
    "Configuration": "Configuration", "Destination_Hit_at_2_pct": r"Hit@2 (\%)",
    "Logistics_Feasibility_pct": r"LFR@3 (\%)", "Constraint_Satisfaction_pct": r"CSR@3 (\%)",
    "Success_at_3_pct": r"Success@3 (\%)", "Coverage_pct": r"Coverage (\%)",
    "Pairwise_Accuracy_pct": r"Pair acc. (\%)", "NDCG_at_2": "NDCG@2",
    "Ranking_Latency_ms": "Ranking (ms)",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(args):
    data_path, queries_path = DATA / "destinations.json", args.queries
    model_path = Path(args.model) if getattr(args, "model", None) else DATA / "catboost_travel_ranker.cbm"
    catalog = json.loads(data_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    model = LTRService.load_model(model_path, fail_fast=True)
    if model is None:
        raise RuntimeError("C4 requires a loadable CatBoost model with matching feature schema.")
    destinations = catalog["destinations"]
    catalogue_routes = build_routes(catalog)
    embeddings = {d["id"]: generate_text_embedding(d["vibe_description"]) for d in destinations}
    query_runs, summaries, pair_rows, pair_audits, failures, archetypes = [], [], [], [], [], {}
    retrieval = {}
    async with AsyncExitStack() as stack:
        neo = await stack.enter_async_context(AsyncGraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)))
        graph = await read_graph_routes(neo, catalog)
        for dest in destinations:
            identifier = dest["id"]
            if Counter(r.key for r in graph[identifier]) != Counter(r.key for r in catalogue_routes[identifier]):
                raise RuntimeError(f"Neo4j paths differ from the catalogue for {identifier}. Seed a clean graph.")
        pg = None
        if args.retrieval == "pgvector":
            pg = await asyncpg.connect(host=settings.POSTGRES_HOST, port=settings.POSTGRES_PORT,
                                       user=settings.POSTGRES_USER, password=settings.POSTGRES_PASSWORD,
                                       database=settings.POSTGRES_DB)
            stack.push_async_callback(pg.close)
            stored = await pg.fetch(
                """SELECT id, embedding::text AS vector FROM destinations
                   WHERE metadata_json->>'embedding_version' = $1 AND id = ANY($2::text[])""",
                EMBEDDING_VERSION, list(embeddings),
            )
            if len(stored) != len(embeddings) or any(
                row["vector"] is None or not np.allclose(
                    json.loads(row["vector"]), embeddings[row["id"]], atol=1e-5, rtol=1e-4
                )
                for row in stored
            ):
                raise RuntimeError("pgvector snapshot differs from the catalogue embeddings. Re-seed PostgreSQL.")
        client = LLMService.get_client() if args.judge == "live" or args.slots == "live" else None
        if (args.judge == "live" or args.slots == "live") and client is None:
            raise RuntimeError("Live NLU/judging requires a configured LLM. Use --slots oracle for component evaluation.")
        predictions = {}
        for q in queries:
            if args.slots == "live":
                pred = await LLMService.dispatch_intent(q["query"], profile={}, current_cart={})
                predictions[q["id"]] = pred
                request = request_from_prediction(pred, q["query"])
            else:
                request = merge_preferences(q, q["query"])
            for vector in (False, True):
                ids = await retrieve(q["query"], destinations, embeddings, pg, vector,
                                     request.get("destination_id") if request and args.slots == "live" else None)
                retrieval[q["id"], vector] = (ids, request)
        pairs = await build_pairs(queries, graph, client if args.judge == "live" else None)

    for key, name, vector, strict, mode in CONFIGS:
        results = []
        detected, emitted = Counter(), Counter()
        for q in queries:
            ids, request = retrieval[q["id"], vector]
            paths = [r for identifier in ids for r in (graph if strict else catalogue_routes)[identifier]]
            candidates, rejected = candidate_pool(paths, request, strict) if request else ([], Counter())
            detected.update(rejected)
            t0 = time.perf_counter()
            preference = request.get("soft_preference") if strict and request else None
            ranked = rank(candidates, mode, model, preference)
            pure_ranked = rank(candidates, mode, model, preference=None)
            top = select_offers(ranked, diversify=strict and args.top3 == "diverse")
            elapsed = (time.perf_counter() - t0) * 1000
            score_top = select_offers(ranked)
            diverse_top = select_offers(ranked, diversify=strict)
            eval_ref = {**q, **(request or {})}
            emitted_packages = []
            for c in top:
                route = c["route"]
                log, con, price = assess(route, q, c["back"])
                emitted.update(log | con)
                emitted_packages.append({
                    "route": route.key, "return_flight": c["back"]["flight_no"] if c["back"] else None,
                    "estimated_price": c["features"]["price"], "reference_price": price,
                    "logistics_errors": sorted(log), "constraint_errors": sorted(con),
                    "destination_match": route.destination in q["expected_destinations"],
                })
            n = len(top)
            row = {
                "config": key, "query_id": q["id"], "archetype": q["archetype"], "top2": ids,
                "expected": q["expected_destinations"], "slots": request,
                "hit2": bool(set(ids) & set(q["expected_destinations"])), "returned": n,
                "lfr": sum(not p["logistics_errors"] for p in emitted_packages) / n if n else 0,
                "csr": sum(not p["constraint_errors"] for p in emitted_packages) / n if n else 0,
                "success": any(offer_is_success(c, eval_ref, candidates) for c in top),
                "success_at_1": bool(top and offer_is_success(top[0], eval_ref, candidates)),
                "pure_success_at_1": bool(pure_ranked and offer_is_success(pure_ranked[0], eval_ref, candidates)),
                "success_score_at_3": any(offer_is_success(c, eval_ref, candidates) for c in score_top),
                "success_diverse_at_3": any(offer_is_success(c, eval_ref, candidates) for c in diverse_top),
                "score_top_destinations": len({c["route"].destination for c in score_top}),
                "diverse_top_destinations": len({c["route"].destination for c in diverse_top}),
                "ranking_ms": elapsed, "packages": emitted_packages,
                "candidate_count": len(paths), "rankable_count": len(candidates),
                "rejection_reasons": dict(rejected), "needs_clarification": request is None,
            }
            results.append(row)
        query_runs.extend(results)
        aggregate, judgments = ranking_metrics(pairs, mode, model)
        archetypes[key] = {arch: aggregate[arch] for arch in ARCHETYPES}
        pair_rows.extend({"Config_ID": key, "Archetype": arch, **result} for arch, result in aggregate.items())
        pair_audits.extend({"config": key, **p} for p in judgments)
        failures.extend({"Config_ID": key, "stage": stage, "reason": reason, "count": count}
                        for stage, counter in (("candidate", detected), ("top3", emitted))
                        for reason, count in sorted(counter.items()))
        summaries.append({
            "Config_ID": key, "Configuration": name,
            "Destination_Hit_at_2_pct": percent(sum(r["hit2"] for r in results), len(results)),
            "Logistics_Feasibility_pct": percent(sum(r["lfr"] for r in results), len(results)),
            "Constraint_Satisfaction_pct": percent(sum(r["csr"] for r in results), len(results)),
            "Success_at_3_pct": percent(sum(r["success"] for r in results), len(results)),
            "Coverage_pct": percent(sum(bool(r["returned"]) for r in results), len(results)),
            "Pairwise_Accuracy_pct": aggregate["ALL"]["pairwise_acc"], "NDCG_at_2": aggregate["ALL"]["ndcg_at_2"],
            "Ranking_Latency_ms": round(float(np.mean([r["ranking_ms"] for r in results])), 3),
        })
        print(summaries[-1], flush=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"slots": args.slots, "retrieval": args.retrieval, "graph": "live_neo4j",
                "top3": args.top3,
                "pair_labels": "llm" if args.judge == "live" else "pareto_dominance",
                "llm_model": settings.OPENAI_MODEL if client else None,
                "query_count": len(queries), "pair_count": len(pairs), "embedding_version": EMBEDDING_VERSION,
                "catalog_sha256": file_hash(data_path), "queries_sha256": file_hash(queries_path),
                "model_sha256": file_hash(model_path)}
    pd.DataFrame(summaries).to_csv(out / "summary_table.csv", index=False)
    pd.DataFrame(query_runs).groupby("config")[[
        "pure_success_at_1", "success_at_1", "success_score_at_3", "success_diverse_at_3",
        "score_top_destinations", "diverse_top_destinations",
    ]].mean().to_csv(out / "top3_comparison.csv")
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_confusion_matrix.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "failure_modes_breakdown.csv", index=False)
    (out / "summary_table.tex").write_text(summary_latex(summaries, LABELS), encoding="utf-8")
    pair_labels = [
        {**{k: p[k] for k in ("query_id", "archetype", "winner")},
         **{side: {"route": p[side]["route"].key, **pair_candidate(p[side])} for side in ("a", "b")}}
        for p in pairs
    ]
    dest_audit = {}
    for key, _, _, _, _ in CONFIGS:
        rows = [r for r in query_runs if r["config"] == key]
        frame = pd.DataFrame({"reference_primary": [r["expected"][0] for r in rows],
                              "retrieved_top1": [r["top2"][0] if r["top2"] else "no_hit" for r in rows]})
        pd.crosstab(frame["reference_primary"], frame["retrieved_top1"]).to_csv(
            out / f"destination_confusion_matrix_{key.split('_')[0].lower()}.csv")
        dest_audit[key] = {
            dest: {"queries": len(subset),
                   "primary_hit_at_1_pct": percent(sum(r["top2"][:1] == [dest] for r in subset), len(subset)),
                   "primary_hit_at_2_pct": percent(sum(dest in r["top2"] for r in subset), len(subset))}
            for dest in sorted(embeddings)
            if (subset := [r for r in rows if r["expected"][0] == dest])
        }
    active_cols = FEATURE_COLUMNS_V2 if LTRService.get_feature_schema() == FEATURE_SCHEMA_V2 else FEATURE_COLUMNS_V1
    output = {
        "benchmark_results.json": {
            "summary": summaries, "metadata": metadata, "archetype_comparison": archetypes,
            "nlu_slot_metrics": nlu_slot_metrics(queries, predictions) if args.slots == "live" else None,
            "feature_importances": dict(zip(active_cols, model.get_feature_importance(type="PredictionValuesChange").tolist())),
        },
        "per_query_audit.json": query_runs,
        "in_query_pairs_judged.json": pair_labels,
        "pair_predictions.json": pair_audits,
        "confusion_matrices.json": {"pairwise": pair_rows, "destination": dest_audit},
        "nlu_predictions.json": predictions,
    }
    for filename, value in output.items():
        (out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results written to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", choices=("oracle", "live"), default="oracle")
    parser.add_argument("--retrieval", choices=("numpy", "pgvector"), default="numpy")
    parser.add_argument("--judge", choices=("pareto", "live"), default="pareto")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--queries", type=Path, default=DATA / "csai_200_benchmark_queries.json")
    parser.add_argument("--top3", choices=("score", "diverse"), default="diverse")
    parser.add_argument("--model", type=Path, default=DATA / "catboost_travel_ranker_v2.cbm" if (DATA / "catboost_travel_ranker_v2.cbm").exists() else DATA / "catboost_travel_ranker.cbm")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
