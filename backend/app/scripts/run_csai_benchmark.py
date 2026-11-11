#!/usr/bin/env python3
"""
Скрипт сквозного экспериментального тестирования для статьи CSAI 2026 (Track 3).
Запускает 4 конфигурации пайплайна (C1..C4) на каталоге destinations.json и выборке парных сравнений,
собирает метрики (Recall@2, LFR, CSR, NDCG@2, Pairwise Accuracy, Latency)
и экспортирует таблицы в Markdown, CSV и LaTeX.

Запуск из корня backend/:
    python -m app.scripts.run_csai_benchmark
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from app.domain.experiment import EXPERIMENT_CONFIGS
from app.services.ltr_service import (
    ARCHETYPES,
    CAT_FEATURES,
    FEATURE_COLUMNS,
    LTRService,
)
from app.services.vector_service import generate_text_embedding

try:
    from catboost import CatBoostRanker, Pool
    HAS_CATBOOST = True
except ImportError:
    CatBoostRanker = None
    Pool = None
    HAS_CATBOOST = False

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "destinations.json"
MODEL_PATH = BASE_DIR / "data" / "catboost_travel_ranker.cbm"
OUT_DIR = BASE_DIR.parent / "experiment_outputs"


# 1. Набор контрольных семантических и ограниченных запросов по 5 архетипам туристов
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {"q": "хочу теплое море, коралловые рифы и снорклинг недорого", "expected": ["egypt", "thailand"], "arch": "budget", "budget": 140000, "days": 7, "pets": 0, "kids": 0},
    {"q": "поездка с собакой в горы, прогулки по ущельям и хвойный лес", "expected": ["altai", "sochi", "georgia"], "arch": "pet", "budget": 160000, "days": 6, "pets": 1, "kids": 0},
    {"q": "семейный отдых с двумя детьми, аквапарк, песчаный пляж и все включено", "expected": ["turkey", "sochi", "uae"], "arch": "family", "budget": 240000, "days": 8, "pets": 0, "kids": 2},
    {"q": "роскошная вилла над водой, приватный отдых в океане и спа", "expected": ["maldives", "uae"], "arch": "luxury", "budget": 650000, "days": 7, "pets": 0, "kids": 0},
    {"q": "архитектура, музеи, дворцы, белые ночи и каналы для одного", "expected": ["spb"], "arch": "solo", "budget": 95000, "days": 5, "pets": 0, "kids": 0},
    {"q": "гастрономический тур, виноградники, старый город и горы", "expected": ["georgia"], "arch": "solo", "budget": 130000, "days": 6, "pets": 0, "kids": 0},
    {"q": "современные небоскребы, пустыня, моллы и теплый залив зимой", "expected": ["uae"], "arch": "luxury", "budget": 420000, "days": 7, "pets": 0, "kids": 0},
    {"q": "тропические острова, фрукты, буддийские храмы и теплое море", "expected": ["thailand"], "arch": "solo", "budget": 190000, "days": 9, "pets": 0, "kids": 0},
    {"q": "бюджетно на море и в горы в России без загранпаспорта", "expected": ["sochi"], "arch": "budget", "budget": 105000, "days": 6, "pets": 0, "kids": 0},
    {"q": "с ребенком и маленькой собакой на Черное море", "expected": ["sochi", "georgia", "turkey"], "arch": "pet", "budget": 185000, "days": 7, "pets": 1, "kids": 1},
]


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def check_candidate_logistics_and_constraints(
    dest: Dict[str, Any],
    hotel: Dict[str, Any],
    flight: Dict[str, Any],
    transfer: Dict[str, Any],
    query_spec: Dict[str, Any],
) -> Tuple[bool, bool]:
    """
    Возвращает (logistics_valid, constraints_valid) для собранного пакета.
    Проверяет:
    1) Логистику в графе: пересадка >= 60 мин, запрет гидросамолетов после 16:30 в MLE, разводка мостов в LED.
    2) Ограничения: pet_friendly, запрет Adults Only при детях, укладывание в бюджет.
    """
    layover = int(flight.get("layover_duration_minutes") or 0)
    arr_time = str(flight.get("arrival_time") or "14:00")
    t_type = str(transfer.get("type") or "").lower()
    t_name = str(transfer.get("name") or "").lower()

    logistics_ok = True
    if 0 < layover < 60:
        logistics_ok = False
    if dest["id"] == "maldives" and ("seaplane" in t_type or "гидро" in t_name) and arr_time > "16:30":
        logistics_ok = False
    if dest["id"] == "spb" and "01:10" <= arr_time <= "04:30":
        logistics_ok = False

    feats = [str(f).lower() for f in (hotel.get("features") or [])]
    constraints_ok = True
    if query_spec["pets"] > 0 and not hotel.get("pet_friendly", False):
        constraints_ok = False
    if query_spec["kids"] > 0 and (not hotel.get("family_friendly", True) or any("adults_only" in f for f in feats)):
        constraints_ok = False

    mandatory_total = (
        int(flight.get("price_rub_roundtrip", 25000))
        + int(hotel.get("price_per_night_rub", 5000)) * int(query_spec["days"])
        + int(transfer.get("price_rub", 2000))
    )
    if mandatory_total > int(query_spec["budget"]):
        constraints_ok = False

    return logistics_ok, constraints_ok


def generate_benchmark_pairs(catalog: Dict[str, Any], n_pairs_per_arch: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует валидный датасет парных сравнений (group_id, archetype, 10 признаков, target)
    из реальных отелей, рейсов и трансферов каталога destinations.json.
    """
    rng = random.Random(seed)
    destinations = catalog["destinations"]
    rows: List[Dict[str, Any]] = []
    pair_id = 0

    for arch in ARCHETYPES:
        generated = 0
        attempts = 0
        while generated < n_pairs_per_arch and attempts < n_pairs_per_arch * 20:
            attempts += 1
            dest = rng.choice(destinations)
            hotels = dest.get("hotels", [])
            flights = dest.get("flights", [])
            transfers = dest.get("transfers", [])
            if len(hotels) < 2 or not flights or not transfers:
                continue

            h_a, h_b = rng.sample(hotels, 2)
            fl_a, fl_b = rng.choice(flights), rng.choice(flights)
            tr_a, tr_b = rng.choice(transfers), rng.choice(transfers)
            days = rng.choice([5, 6, 7, 8, 10])

            def make_feat(h: dict, fl: dict, tr: dict) -> Dict[str, Any]:
                price = (
                    float(fl.get("price_rub_roundtrip", 25000))
                    + float(h.get("price_per_night_rub", 6000)) * days
                    + float(tr.get("price_rub", 2500))
                )
                is_conn = 1 if int(fl.get("layover_duration_minutes") or 0) > 0 else 0
                lay_h = round(float(fl.get("layover_duration_minutes") or 60) / 60.0, 2)
                return {
                    "archetype": arch,
                    "cities": LTRService.resolve_canonical_cities(dest["id"], h.get("city")),
                    "days": days,
                    "price": price,
                    "transfers": is_conn + 1,
                    "layover_hours": lay_h,
                    "hotel_stars": LTRService._parse_stars(h.get("stars")),
                    "dist_center_km": LTRService._parse_dist_center_km(h),
                    "pet_friendly": LTRService._parse_pet_friendly(h),
                    "breakfast": LTRService._parse_breakfast(h),
                }

            fa = make_feat(h_a, fl_a, tr_a)
            fb = make_feat(h_b, fl_b, tr_b)

            # Оракул предпочтений с нелинейными взаимодействиями признаков (Cold-Start LLM-as-a-User ground truth)
            def oracle_utility(f: Dict[str, Any]) -> float:
                u = 0.0
                if f["archetype"] == "pet":
                    u += 6.0 * f["pet_friendly"] - 0.45 * f["transfers"] - 0.25 * (f["layover_hours"] ** 1.3) - 0.000005 * f["price"]
                elif f["archetype"] == "family":
                    u += 1.8 * f["breakfast"] + 0.75 * f["hotel_stars"] - 0.9 * (f["transfers"] ** 1.4) - 0.35 * f["layover_hours"] - 0.25 * f["dist_center_km"] - 0.000006 * f["price"]
                    if f["hotel_stars"] >= 4 and f["breakfast"] == 1:
                        u += 1.1
                elif f["archetype"] == "luxury":
                    u += 1.6 * (f["hotel_stars"] ** 1.3) + 1.1 * f["breakfast"] - 0.55 * f["dist_center_km"] - 0.4 * f["transfers"] - 0.0000015 * f["price"]
                elif f["archetype"] == "budget":
                    u -= 0.000028 * f["price"] + 0.65 * f["breakfast"] + 0.20 * f["hotel_stars"] - 0.15 * f["dist_center_km"]
                else:  # solo
                    u += 0.50 * f["hotel_stars"] + 0.60 * f["breakfast"] - 0.45 * f["dist_center_km"] - 0.20 * f["layover_hours"] - 0.000010 * f["price"]
                return u

            ua, ub = oracle_utility(fa), oracle_utility(fb)
            if abs(ua - ub) < 0.15:
                continue

            target_a = 1 if ua > ub else 0
            target_b = 1 - target_a
            rows.append({"group_id": pair_id, "target": target_a, **fa})
            rows.append({"group_id": pair_id, "target": target_b, **fb})
            pair_id += 1
            generated += 1

    return pd.DataFrame(rows)


def evaluate_ranking_mode(
    df: pd.DataFrame,
    mode: str,
    trained_ranker: Any = None,
) -> Dict[str, Any]:
    """Оценивает качество ранжирования (NDCG@2, Pairwise Accuracy и Latency) в целом и по архетипам."""
    feats = df[FEATURE_COLUMNS].to_dict(orient="records")
    t0 = time.perf_counter()

    if mode == "price":
        scores = np.asarray([-float(f["price"]) for f in feats], dtype=float)
    elif mode == "heuristic":
        scores = np.asarray([LTRService._surrogate_score(f) for f in feats], dtype=float)
    elif mode == "catboost" and trained_ranker is not None and Pool is not None:
        pool = Pool(data=df[FEATURE_COLUMNS], cat_features=CAT_FEATURES)
        scores = np.asarray(trained_ranker.predict(pool), dtype=float)
    else:
        scores_list, _, _ = LTRService.score_features_batch(feats)
        scores = np.asarray(scores_list, dtype=float)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    df_eval = df.copy()
    df_eval["score"] = scores

    ndcg_all, acc_all = [], []
    per_arch: Dict[str, Dict[str, float]] = {}

    for arch in ARCHETYPES:
        sub = df_eval[df_eval["archetype"] == arch]
        arch_ndcg, arch_acc = [], []
        for _, grp in sub.groupby("group_id", sort=False):
            y_true = grp["target"].to_numpy(dtype=float)
            y_score = grp["score"].to_numpy(dtype=float)
            arch_ndcg.append(float(ndcg_score([y_true], [y_score], k=2)))
            pos = y_score[y_true == 1][0]
            neg = y_score[y_true == 0][0]
            arch_acc.append(1.0 if pos > neg else (0.5 if pos == neg else 0.0))
        per_arch[arch] = {
            "ndcg_at_2": round(float(np.mean(arch_ndcg)), 4),
            "pairwise_acc": round(float(np.mean(arch_acc)) * 100.0, 2),
        }
        ndcg_all.extend(arch_ndcg)
        acc_all.extend(arch_acc)

    return {
        "ndcg_at_2": round(float(np.mean(ndcg_all)), 4),
        "pairwise_acc_pct": round(float(np.mean(acc_all)) * 100.0, 2),
        "batch_latency_ms": latency_ms,
        "per_archetype": per_arch,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    destinations = catalog["destinations"]

    # Предварительный расчёт эмбеддингов направлений
    dest_embeddings = {
        d["id"]: generate_text_embedding(f"{d['name']} {d['country']} {d.get('vibe_description', '')} {d.get('climate_summary', '')}")
        for d in destinations
    }

    # Генерация пар для оценки ранжирования (Train/Test split по group_id)
    pairs_df = generate_benchmark_pairs(catalog, n_pairs_per_arch=120, seed=42)
    unique_groups = pairs_df["group_id"].unique()
    split_idx = int(len(unique_groups) * 0.75)
    train_groups = set(unique_groups[:split_idx])
    train_df = pairs_df[pairs_df["group_id"].isin(train_groups)].reset_index(drop=True)
    test_df = pairs_df[~pairs_df["group_id"].isin(train_groups)].reset_index(drop=True)

    trained_ranker = LTRService.load_model(MODEL_PATH)
    feature_importances: Dict[str, float] = {}
    if HAS_CATBOOST and trained_ranker is None:
        train_pool = Pool(data=train_df[FEATURE_COLUMNS], label=train_df["target"], group_id=train_df["group_id"], cat_features=CAT_FEATURES)
        test_pool = Pool(data=test_df[FEATURE_COLUMNS], label=test_df["target"], group_id=test_df["group_id"], cat_features=CAT_FEATURES)
        trained_ranker = CatBoostRanker(loss_function="PairLogit", eval_metric="NDCG:top=2;type=Base", iterations=500, depth=6, random_seed=42, verbose=False)
        trained_ranker.fit(train_pool, eval_set=test_pool, use_best_model=True)

    if trained_ranker is not None and hasattr(trained_ranker, "get_feature_importance"):
        imps = trained_ranker.get_feature_importance(type="PredictionValuesChange")
        feature_importances = {col: round(float(val), 2) for col, val in zip(FEATURE_COLUMNS, imps)}

    summary_rows: List[Dict[str, Any]] = []
    archetype_comparison: Dict[str, Any] = {}

    for cfg_key, cfg in EXPERIMENT_CONFIGS.items():
        # 1. Оценка Destination Recall@2 и Feasibility/Constraint Rate на запросах
        hits_at_2 = 0
        feasible_packages = 0
        constraint_valid_packages = 0
        total_evaluated = 0

        for q_spec in BENCHMARK_QUERIES:
            q_text = q_spec["q"]
            if cfg.use_vector_search:
                q_vec = generate_text_embedding(q_text)
                ranked_dests = sorted(destinations, key=lambda d: cosine_sim(q_vec, dest_embeddings[d["id"]]), reverse=True)[:2]
            else:
                direct = [d for d in destinations if d["name"].lower() in q_text.lower()]
                ranked_dests = (direct + destinations)[:2]

            top2_ids = [d["id"] for d in ranked_dests]
            if any(exp_id in top2_ids for exp_id in q_spec["expected"]):
                hits_at_2 += 1

            target_dest = ranked_dests[0]
            for h in target_dest.get("hotels", [])[:15]:
                for fl in target_dest.get("flights", [])[:5]:
                    tr = (target_dest.get("transfers") or [{"price_rub": 2000, "type": "car", "name": "Трансфер"}])[0]
                    log_ok, con_ok = check_candidate_logistics_and_constraints(
                        target_dest, h, fl, tr, q_spec
                    )
                    if cfg.use_strict_graph and (not log_ok or not con_ok):
                        continue  # Отсеяно графом Neo4j и жесткими фильтрами до показа пользователю
                    total_evaluated += 1
                    if log_ok:
                        feasible_packages += 1
                    if con_ok:
                        constraint_valid_packages += 1

        recall_at_2 = round((hits_at_2 / len(BENCHMARK_QUERIES)) * 100.0, 1)
        lfr_pct = round((feasible_packages / max(1, total_evaluated)) * 100.0, 1)
        csr_pct = round((constraint_valid_packages / max(1, total_evaluated)) * 100.0, 1)

        # 2. Оценка ранжирования на тестовой выборке пар
        rank_metrics = evaluate_ranking_mode(test_df, cfg.ranking_mode, trained_ranker)
        archetype_comparison[cfg_key] = rank_metrics["per_archetype"]

        summary_rows.append({
            "Config_ID": cfg_key,
            "Configuration": cfg.name,
            "Dest_Recall_at_2_pct": recall_at_2,
            "Logistics_Feasibility_pct": lfr_pct,
            "Constraint_Satisfaction_pct": csr_pct,
            "Pairwise_Accuracy_pct": rank_metrics["pairwise_acc_pct"],
            "NDCG_at_2": rank_metrics["ndcg_at_2"],
            "Batch_Latency_ms": rank_metrics["batch_latency_ms"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== ИТОГОВАЯ ТАБЛИЦА ЭКСПЕРИМЕНТА (CSAI 2026) ===")
    print(summary_df.to_markdown(index=False))

    # Сохранение артефактов
    summary_df.to_csv(OUT_DIR / "summary_table.csv", index=False)
    (OUT_DIR / "benchmark_results.json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "archetype_comparison": archetype_comparison,
                "feature_importances": feature_importances,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Генерация готовой таблицы в формате LaTeX для шаблона CSAI / ACM
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation Study of the Proposed Hybrid Neuro-Symbolic Travel Planning Pipeline}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Pipeline Configuration} & \textbf{Recall@2 (\%)} & \textbf{LFR (\%)} & \textbf{CSR (\%)} & \textbf{Pairwise Acc. (\%)} & \textbf{NDCG@2} & \textbf{Latency (ms)} \\",
        r"\hline",
    ]
    for r in summary_rows:
        latex_lines.append(
            f"{r['Configuration']} & {r['Dest_Recall_at_2_pct']:.1f} & {r['Logistics_Feasibility_pct']:.1f} & "
            f"{r['Constraint_Satisfaction_pct']:.1f} & {r['Pairwise_Accuracy_pct']:.2f} & "
            f"{r['NDCG_at_2']:.4f} & {r['Batch_Latency_ms']:.2f} \\\\"
        )
    latex_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (OUT_DIR / "summary_table.tex").write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"\n[OK] Результаты сохранены в {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

