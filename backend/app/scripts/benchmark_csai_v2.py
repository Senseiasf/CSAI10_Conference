#!/usr/bin/env python3
"""
Скрипт сравнения моделей на закрытом тесте компромиссов (Этапы 6 и 7 CSAI 2026).

Сравнивает 5 конфигураций на абсолютно одинаковом входе:
1. Price: Сортировка по минимальной цене
2. C3: Действующая графовая эвристика (heuristic)
3. C4 (Legacy): Действующий CatBoost чекпойнт (12 признаков V1)
4. C4 (V2 Full): Новый CatBoost чекпойнт (22 признака V2, включая предпочтения)
5. C4 (V2 Ablation): Контрольный CatBoost V2 без признаков предпочтений/доплаты

Вычисляет:
- Парную точность (Pairwise Accuracy %) по однозначным меткам
- Точность в разрезе 5 архетипов (family, budget, solo, luxury, pet)
- Точность в разрезе 5 видов компромиссов (direct_vs_price, time_vs_price, stars_vs_price, etc.)
- Парный статистический тест Стьюдента / Биномиальный тест побед C4 vs C3 (p-value)
- Задержку инференса (Ranking Latency ms на CPU)
- Подробные кейсы побед и поражений новой модели для отчета.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from scipy.stats import binomtest

from app.services.ltr_service import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    CAT_FEATURES_V1,
    CAT_FEATURES_V2,
    LTRService,
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "app" / "data"
DEFAULT_TEST_PATH = DATA_DIR / "csai_tradeoffs_v2_test.jsonl"
DEFAULT_MODEL_V1_PATH = DATA_DIR / "catboost_travel_ranker.cbm"
DEFAULT_MODEL_V2_PATH = DATA_DIR / "catboost_travel_ranker_v2.cbm"
DEFAULT_MODEL_ABL_PATH = DATA_DIR / "catboost_travel_ranker_v2_ablation.cbm"
DEFAULT_OUTPUT_DIR = (BACKEND_DIR.parent if (BACKEND_DIR.parent / "docker-compose.yml").exists() else BACKEND_DIR) / "experiment_outputs" / "csai_v2_evaluation"


def load_test_pairs(test_file: Path) -> List[Dict[str, Any]]:
    pairs = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def score_candidate(
    feat: Dict[str, Any],
    mode: str,
    model: Optional[CatBoostRanker] = None,
    cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> float:
    if mode == "price":
        return -float(feat.get("price", 0.0))
    if mode == "heuristic":
        return LTRService._surrogate_score(feat)
    if mode.startswith("catboost") and model is not None and cols is not None and cat_cols is not None:
        norm = LTRService.normalize_features(feat, schema="v2" if len(cols) > 12 else "v1")
        if mode == "catboost_v2_abl":
            for col in ("matches_time_pref", "matches_direct_pref", "price_delta_vs_cheapest", "relative_surcharge_pct"):
                norm[col] = 0.0
        pool = Pool(
            data=[[norm.get(col, 0.0) for col in cols]],
            feature_names=cols,
            cat_features=cat_cols,
        )
        return float(model.predict(pool)[0])
    raise ValueError(f"Unknown scoring mode: {mode}")


def evaluate_models_on_test(
    test_pairs: List[Dict[str, Any]],
    model_v1: Optional[CatBoostRanker],
    model_v2: Optional[CatBoostRanker],
    model_abl: Optional[CatBoostRanker],
) -> Dict[str, Any]:
    configs = [
        ("Price_Baseline", "price", None, None, None),
        ("C3_Graph_Heuristic", "heuristic", None, None, None),
    ]
    if model_v1 is not None:
        configs.append(("C4_Legacy_V1", "catboost_v1", model_v1, FEATURE_COLUMNS_V1, CAT_FEATURES_V1))
    if model_abl is not None:
        # Аблация: признаки V2 без преференсов (зануляются)
        configs.append(("C4_V2_Ablation", "catboost_v2_abl", model_abl, FEATURE_COLUMNS_V2, CAT_FEATURES_V2))
    if model_v2 is not None:
        configs.append(("C4_V2_Full_Context", "catboost_v2", model_v2, FEATURE_COLUMNS_V2, CAT_FEATURES_V2))

    rows = []
    latencies = defaultdict(list)

    for pair in test_pairs:
        pair_id = pair["pair_id"]
        true_winner = pair["winner"]
        arch = pair["archetype"]
        fam = pair["tradeoff_family"]
        dest = pair["destination_id"]
        tour_a = pair["tour_a"]
        tour_b = pair["tour_b"]

        for cfg_name, mode, mdl, cols, cats in configs:
            t0 = time.perf_counter()
            score_a = score_candidate(tour_a, mode, mdl, cols, cats)
            score_b = score_candidate(tour_b, mode, mdl, cols, cats)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies[cfg_name].append(latency)

            if abs(score_a - score_b) < 1e-6:
                # Тай-брейк по минимальной цене
                pred_winner = "A" if tour_a["price"] < tour_b["price"] else "B"
            else:
                pred_winner = "A" if score_a > score_b else "B"

            correct = int(pred_winner == true_winner)

            rows.append({
                "pair_id": pair_id,
                "config": cfg_name,
                "archetype": arch,
                "tradeoff_family": fam,
                "destination_id": dest,
                "expected": true_winner,
                "predicted": pred_winner,
                "correct": correct,
                "score_a": round(score_a, 4),
                "score_b": round(score_b, 4),
                "price_a": tour_a["price"],
                "price_b": tour_b["price"],
            })

    df = pd.DataFrame(rows)

    # 1. Общие метрики
    summary_list = []
    for cfg_name in df["config"].unique():
        sub = df[df["config"] == cfg_name]
        acc = sub["correct"].mean() * 100.0
        n_correct = sub["correct"].sum()
        total = len(sub)
        lat = np.mean(latencies[cfg_name])
        summary_list.append({
            "Configuration": cfg_name,
            "Pairwise_Accuracy_pct": round(acc, 2),
            "Correct_Pairs": int(n_correct),
            "Total_Pairs": int(total),
            "Latency_ms_per_pair": round(lat, 3),
        })

    summary_df = pd.DataFrame(summary_list)

    # 2. Метрики по архетипам
    arch_df = df.pivot_table(index="archetype", columns="config", values="correct", aggfunc=lambda x: round(np.mean(x) * 100.0, 1))

    # 3. Метрики по видам компромиссов
    tradeoff_df = df.pivot_table(index="tradeoff_family", columns="config", values="correct", aggfunc=lambda x: round(np.mean(x) * 100.0, 1))

    # 4. Парное сравнение C4 V2 Full vs C3 Heuristic
    paired_c4_c3 = {}
    c4_name = "C4_V2_Full_Context" if "C4_V2_Full_Context" in df["config"].unique() else "C4_Legacy_V1"
    c3_name = "C3_Graph_Heuristic"

    piv = df.pivot(index="pair_id", columns="config", values="correct")
    c4_only = int(((piv[c4_name] == 1) & (piv[c3_name] == 0)).sum())
    c3_only = int(((piv[c3_name] == 1) & (piv[c4_name] == 0)).sum())
    both_correct = int(((piv[c4_name] == 1) & (piv[c3_name] == 1)).sum())
    both_incorrect = int(((piv[c4_name] == 0) & (piv[c3_name] == 0)).sum())

    total_discordant = c4_only + c3_only
    p_val = binomtest(c4_only, total_discordant).pvalue if total_discordant > 0 else 1.0

    stat_test = {
        "c4_config": c4_name,
        "c3_config": c3_name,
        "c4_wins_only": c4_only,
        "c3_wins_only": c3_only,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "binomial_p_value": round(float(p_val), 5),
        "statistically_significant": bool(p_val < 0.05),
    }

    # 5. Примеры выигрышей и ошибок C4
    wins_df = df[(df["config"] == c4_name) & (df["correct"] == 1) & (df["pair_id"].isin(df[(df["config"] == c3_name) & (df["correct"] == 0)]["pair_id"]))].head(5)
    losses_df = df[(df["config"] == c4_name) & (df["correct"] == 0) & (df["pair_id"].isin(df[(df["config"] == c3_name) & (df["correct"] == 1)]["pair_id"]))].head(5)

    return {
        "summary": summary_df.to_dict(orient="records"),
        "by_archetype": arch_df.to_dict(),
        "by_tradeoff": tradeoff_df.to_dict(),
        "statistical_comparison": stat_test,
        "c4_win_examples": wins_df.to_dict(orient="records"),
        "c4_loss_examples": losses_df.to_dict(orient="records"),
        "detailed_predictions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение моделей ранжирования на закрытом тесте")
    parser.add_argument("--test-file", type=str, default=str(DEFAULT_TEST_PATH), help="Путь к csai_tradeoffs_v2_test.jsonl")
    parser.add_argument("--model-v1", type=str, default=str(DEFAULT_MODEL_V1_PATH), help="Модель V1")
    parser.add_argument("--model-v2", type=str, default=str(DEFAULT_MODEL_V2_PATH), help="Модель V2")
    parser.add_argument("--model-abl", type=str, default=str(DEFAULT_MODEL_ABL_PATH), help="Модель V2 Ablation")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Каталог для отчетов")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_pairs = load_test_pairs(Path(args.test_file))
    logger.info("Загружено %d тестовых пар из %s", len(test_pairs), args.test_file)

    # Загрузка моделей
    mdl_v1, mdl_v2, mdl_abl = None, None, None
    if Path(args.model_v1).is_file():
        mdl_v1 = CatBoostRanker()
        mdl_v1.load_model(args.model_v1)
        logger.info("Модель V1 загружена: %s", args.model_v1)
    if Path(args.model_v2).is_file():
        mdl_v2 = CatBoostRanker()
        mdl_v2.load_model(args.model_v2)
        logger.info("Модель V2 загружена: %s", args.model_v2)
    if Path(args.model_abl).is_file():
        mdl_abl = CatBoostRanker()
        mdl_abl.load_model(args.model_abl)
        logger.info("Модель V2 Ablation загружена: %s", args.model_abl)

    results = evaluate_models_on_test(test_pairs, mdl_v1, mdl_v2, mdl_abl)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("CSAI 2026: ИТОГОВЫЕ РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕЛЕЙ НА ЗАКРЫТОМ ТЕСТЕ")
    print("=" * 80)
    summary_df = pd.DataFrame(results["summary"])
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("Парное статистическое сравнение (C4 vs C3):")
    st = results["statistical_comparison"]
    print(f"  - Побед только {st['c4_config']} : {st['c4_wins_only']}")
    print(f"  - Побед только {st['c3_config']} : {st['c3_wins_only']}")
    print(f"  - Совпадение (оба верны)   : {st['both_correct']}")
    print(f"  - Совпадение (оба неверны) : {st['both_incorrect']}")
    print(f"  - Binomial Test p-value    : {st['binomial_p_value']} (значимо: {st['statistically_significant']})")


if __name__ == "__main__":
    main()
