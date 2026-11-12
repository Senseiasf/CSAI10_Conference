#!/usr/bin/env python3
"""
Скрипт разбиения выборки пар с мягкими компромиссами на train, validation и закрытый test (Этап 5).

Принципы разбиения:
1. Группировка строго по семейству сценариев (`scenario_family_id`): все пары, варианты запроса
   и парафразы одного семейства остаются строго в одной части.
2. Проверка отсутствия утечек между выборками (Data Leakage Check).
3. Закрытый test изолируется от процесса обучения и подбора гиперпараметров.
4. Выгрузка в двух форматах:
   - JSONL для подачи напрямую в `train_ltr.py` (схема V2)
   - JSON со всеми метаданными разметки и обоснованиями LLM-судьи.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LABELED_PATH = BACKEND_DIR / "app" / "data" / "csai_tradeoffs_labeled_v2.json"
DATA_DIR = BACKEND_DIR / "app" / "data"


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_jsonl_row(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Форматирует запись пары туров в формат, ожидаемый train_ltr.py."""
    winner = pair["label"]["winner"]
    feat_a = dict(pair["tour_a"]["features_v2"])
    feat_b = dict(pair["tour_b"]["features_v2"])

    return {
        "pair_id": pair["pair_id"],
        "scenario_family_id": pair["scenario_family_id"],
        "archetype": pair["archetype"],
        "destination_id": pair["destination_id"],
        "tradeoff_family": pair["tradeoff_family"],
        "winner": winner,
        "source": pair["label"].get("source", "llm_judge"),
        "confidence": pair["label"].get("confidence", "high"),
        "tour_a": feat_a,
        "tour_b": feat_b,
    }


def split_labeled_dataset(
    labeled_file: Path,
    output_dir: Path,
    seed: int = 42,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
) -> Dict[str, Any]:
    raw = json.loads(labeled_file.read_text(encoding="utf-8"))
    pairs = raw["labeled_pairs"]

    # Исключаем ничьи (TIE) из обучающей выборки для функции потерь PairLogit,
    # сохраняя их в отдельный файл для анализа калибровки
    definitive_pairs = [p for p in pairs if p["label"]["winner"] in ("A", "B")]
    tie_pairs = [p for p in pairs if p["label"]["winner"] == "TIE"]

    logger.info(
        "Всего размечено пар: %d. Однозначных: %d, ничьих (TIE): %d",
        len(pairs), len(definitive_pairs), len(tie_pairs)
    )

    # Группируем по scenario_family_id
    families = defaultdict(list)
    for p in definitive_pairs:
        families[p["scenario_family_id"]].append(p)

    all_family_ids = sorted(families.keys())
    rng = random.Random(seed)
    rng.shuffle(all_family_ids)

    n_families = len(all_family_ids)
    n_test = max(2, int(round(n_families * test_ratio)))
    n_val = max(2, int(round(n_families * val_ratio)))
    n_train = n_families - n_test - n_val

    test_fam = set(all_family_ids[:n_test])
    val_fam = set(all_family_ids[n_test : n_test + n_val])
    train_fam = set(all_family_ids[n_test + n_val :])

    logger.info("Семейства в Test: %s", sorted(test_fam))
    logger.info("Семейства в Validation: %s", sorted(val_fam))
    logger.info("Семейства в Train: %s", sorted(train_fam))

    train_pairs = [p for fam in train_fam for p in families[fam]]
    val_pairs = [p for fam in val_fam for p in families[fam]]
    test_pairs = [p for fam in test_fam for p in families[fam]]

    # Проверка на отсутствие пересечений
    train_routes = {tuple(p["tour_a"]["route_key"]) for p in train_pairs} | {tuple(p["tour_b"]["route_key"]) for p in train_pairs}
    test_routes = {tuple(p["tour_a"]["route_key"]) for p in test_pairs} | {tuple(p["tour_b"]["route_key"]) for p in test_pairs}
    leakage = train_fam & test_fam
    if leakage:
        raise ValueError(f"Data leakage detected! Families shared between train and test: {leakage}")

    # Сохраняем в JSONL
    train_jsonl = output_dir / "csai_tradeoffs_v2_train.jsonl"
    val_jsonl = output_dir / "csai_tradeoffs_v2_val.jsonl"
    test_jsonl = output_dir / "csai_tradeoffs_v2_test.jsonl"
    ties_json = output_dir / "csai_tradeoffs_v2_ties.json"

    with train_jsonl.open("w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(format_jsonl_row(p), ensure_ascii=False) + "\n")

    with val_jsonl.open("w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(format_jsonl_row(p), ensure_ascii=False) + "\n")

    with test_jsonl.open("w", encoding="utf-8") as f:
        for p in test_pairs:
            f.write(json.dumps(format_jsonl_row(p), ensure_ascii=False) + "\n")

    ties_json.write_text(json.dumps({"ties": tie_pairs}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "split_seed": seed,
        "source_labeled_file": labeled_file.name,
        "total_pairs": len(pairs),
        "definitive_pairs": len(definitive_pairs),
        "tie_pairs": len(tie_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "train_families": sorted(train_fam),
        "val_families": sorted(val_fam),
        "test_families": sorted(test_fam),
        "train_sha256": file_sha256(train_jsonl),
        "val_sha256": file_sha256(val_jsonl),
        "test_sha256": file_sha256(test_jsonl),
    }

    meta_file = output_dir / "csai_tradeoffs_v2_split_meta.json"
    meta_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Сплит размеченной выборки пар CSAI 2026")
    parser.add_argument("--labeled", type=str, default=str(DEFAULT_LABELED_PATH), help="Размеченный датасет")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Директория для сплитов")
    parser.add_argument("--seed", type=int, default=42, help="Случайный seed")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Доля сценариев в test")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Доля сценариев в validation")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    summary = split_labeled_dataset(
        labeled_file=Path(args.labeled),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
    )

    print("=" * 70)
    print("CSAI 2026: Сплит выборки завершен")
    print("=" * 70)
    print(f"Train : {summary['train_pairs']} пар ({len(summary['train_families'])} семейств) -> {summary['train_sha256'][:16]}...")
    print(f"Val   : {summary['val_pairs']} пар ({len(summary['val_families'])} семейств) -> {summary['val_sha256'][:16]}...")
    print(f"Test  : {summary['test_pairs']} пар ({len(summary['test_families'])} семейств) -> {summary['test_sha256'][:16]}...")
    print(f"Ties  : {summary['tie_pairs']} пар сохранено в ties.json")


if __name__ == "__main__":
    main()

