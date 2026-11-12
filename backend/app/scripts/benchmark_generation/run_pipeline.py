#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт полного цикла генерации и многоступенчатой фильтрации эталонного датасета
из ровно 200 запросов (по 40 на каждый из 5 архетипов: budget, pet, family, luxury, solo)
для статьи CSAI 2026.

Сохраняет результат в backend/app/data/csai_150_benchmark_queries.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Обеспечиваем корректное добавление корня backend в sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parents[2]
REPO_ROOT = SCRIPT_DIR.parents[3]

for p in (str(REPO_ROOT), str(BACKEND_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from filter_diversity import (
    audit_catalog_feasibility,
    clean_and_validate_query,
    deduplicate_archetype_pool,
)
from generate_raw_queries import generate_raw_queries

DESTINATIONS_FILE = BACKEND_DIR / "app" / "data" / "destinations.json"
OUTPUT_FILE = BACKEND_DIR / "app" / "data" / "csai_150_benchmark_queries.json"


def main() -> None:
    print("=" * 70, flush=True)
    print("CSAI 2026: Генерация и фильтрация 200 эталонных запросов (по 40 на архетип)", flush=True)
    print("=" * 70, flush=True)

    # 1. Генерация первичного сырого пула (500-550 сценариев)
    print("\n[Шаг 1/4] Генерация расширенного сырого пула сценариев (по ~110 на архетип)...", flush=True)
    raw_queries = generate_raw_queries(target_count_per_arch=110)
    print(f"          Всего сгенерировано: {len(raw_queries)} сырых сценариев.", flush=True)

    # 2. Построчная очистка и валидация (03_clean_and_filter.py)
    print("\n[Шаг 2/4] Глубокая очистка и валидация текста (фильтр метаданных, стоп-слов, баланс языка)...", flush=True)
    cleaned_pool = []
    rejected_count = 0
    for item in raw_queries:
        cleaned_text = clean_and_validate_query(item["query"])
        if cleaned_text is not None:
            item_copy = dict(item)
            item_copy["query"] = cleaned_text
            cleaned_pool.append(item_copy)
        else:
            rejected_count += 1
    print(f"          Прошли очистку: {len(cleaned_pool)} (отсеяно: {rejected_count})", flush=True)

    # 3. Трёхступенчатая дедупликация (04_merge_synthetic_pools.py)
    print("\n[Шаг 3/4] Трёхступенчатая дедупликация по 5 архетипам...", flush=True)
    archetypes = ["budget", "pet", "family", "luxury", "solo"]
    global_seen_norms = set()
    deduped_by_arch = {}

    for arch in archetypes:
        arch_items = [q for q in cleaned_pool if q["archetype"] == arch]
        filtered = deduplicate_archetype_pool(
            items=arch_items,
            archetype=arch,
            max_first_word=6,
            max_bigram_prefix=3,
            max_trigram=5,
            sim_thresh=0.65,
            global_seen_norms=global_seen_norms,
            seed=42,
        )
        deduped_by_arch[arch] = filtered

    # 4. Аудит выполнимости по каталогу и выравнивание квоты (06_finalise_datasets.py)
    print("\n[Шаг 4/4] Аудит выполнимости по каталогу destinations.json и выравнивание квоты по 40 запросов...")
    catalog = json.loads(DESTINATIONS_FILE.read_text(encoding="utf-8"))

    final_pool = []
    QUOTA_PER_ARCH = 40

    for arch in archetypes:
        items = deduped_by_arch[arch]
        audited = audit_catalog_feasibility(items, catalog)

        if len(audited) < QUOTA_PER_ARCH:
            raise RuntimeError(f"Only {len(audited)} diverse queries for {arch}; generate a larger raw pool")

        selected = audited[:QUOTA_PER_ARCH]
        final_pool.extend(selected)

        implicit_cnt = sum(1 for it in selected if it.get("is_implicit_vibe", False))
        implicit_pct = round(implicit_cnt / len(selected) * 100, 1)
        print(f"  -> Архетип {arch:<8}: отобрано {len(selected)}/40 (Implicit Vibe: {implicit_cnt}/40 = {implicit_pct}%)")

    # Нумерация CSAI_001 .. CSAI_200
    for idx, item in enumerate(final_pool, start=1):
        item["id"] = f"CSAI_{idx:03d}"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(final_pool, ensure_ascii=False, indent=2), encoding="utf-8")

    total_implicit = sum(1 for it in final_pool if it.get("is_implicit_vibe", False))
    print("\n" + "=" * 70)
    print(f"[УСПЕХ] Эталонный датасет сформирован: {len(final_pool)} запросов (5 архетипов по 40).")
    print(f"        Доля неявных запросов (Implicit Vibe): {total_implicit}/{len(final_pool)} ({total_implicit / len(final_pool) * 100:.1f}%)")
    print(f"        Файл сохранён: {OUTPUT_FILE.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
