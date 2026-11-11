#!/usr/bin/env python3
"""
Скрипт для парсинга синтетического датасета парных сравнений туров (Cold-Start)
и обучения модели ранжирования CatBoostRanker (Learning to Rank).
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit


# Список признаков для обучения модели
FEATURE_COLUMNS: List[str] = [
    "archetype",
    "cities",
    "days",
    "price",
    "transfers",
    "layover_hours",
    "hotel_stars",
    "dist_center_km",
    "pet_friendly",
    "breakfast",
]

# Категориальные признаки
CAT_FEATURES: List[str] = ["archetype", "cities"]


def parse_binary(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("0", "false", "no", "нет"):
            return 0
        if value in ("1", "true", "yes", "да"):
            return 1
        raise ValueError(f"Некорректный бинарный признак: {value!r}")
    if value not in (0, 1, False, True):
        raise ValueError(f"Некорректный бинарный признак: {value!r}")
    return int(value)


def validate_groups(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Датасет не содержит пар сравнений")
    for group_id, group in df.groupby("group_id", sort=False):
        if len(group) != 2 or sorted(group["target"].tolist()) != [0, 1]:
            raise ValueError(f"Группа {group_id} должна содержать ровно два тура и одного победителя")


def resolve_dataset_path(requested_path: str) -> Path:
    """
    Проверяет наличие файла датасета. Если указанный файл (например, part_4060ti_2.jsonl)
    не найден, автоматически проверяет резервный путь part_4060ti.jsonl.
    """
    path = Path(requested_path)
    if path.exists():
        return path

    fallback = path.with_name("part_4060ti.jsonl")
    if fallback.exists():
        print(f"[INFO] Файл '{path}' не найден. Используется найденный файл '{fallback}'.")
        return fallback

    raise FileNotFoundError(
        f"Файл датасета не найден ни по пути '{path}', ни по резервному пути '{fallback}'."
    )


def load_and_flatten_jsonl(file_path: Path) -> pd.DataFrame:
    """
    Читает JSONL-файл с парными сравнениями туров и разворачивает (flattens) каждую строку
    в ДВЕ записи DataFrame (для tour_a и tour_b), объединённые общим group_id (pair_id).

    Правила разметки target:
    - Для записи из tour_a: target = 1, если winner == "A", иначе 0.
    - Для записи из tour_b: target = 1, если winner == "B", иначе 0.
    Поля reasoning и margin игнорируются в базовой реализации.
    """
    records: List[Dict[str, Any]] = []
    seen_groups = set()

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            group_id = int(row["pair_id"])
            if not 0 <= group_id < 2 ** 63 or group_id in seen_groups:
                raise ValueError(f"Строка {line_num}: pair_id должен быть уникальным неотрицательным int64")
            seen_groups.add(group_id)
            archetype = str(row["archetype"])
            winner = str(row["winner"]).strip().upper()
            if winner not in ("A", "B"):
                raise ValueError(f"Строка {line_num}: winner должен быть A или B")
            if row["archetype"] is None or not archetype.strip():
                raise ValueError(f"Строка {line_num}: archetype не должен быть пустым")

            for tour_key, winner_code in (("tour_a", "A"), ("tour_b", "B")):
                tour_data = row[tour_key]
                target = 1 if winner == winner_code else 0
                if tour_data["cities"] is None or not str(tour_data["cities"]).strip():
                    raise ValueError(f"Строка {line_num}: cities не должен быть пустым")
                for column in FEATURE_COLUMNS[2:-2]:
                    if not math.isfinite(float(tour_data[column])):
                        raise ValueError(f"Строка {line_num}: {column} должен быть конечным числом")
                if float(tour_data["days"]) < 1 or not 1 <= float(tour_data["hotel_stars"]) <= 5:
                    raise ValueError(f"Строка {line_num}: некорректная длительность или звёздность")
                if any(float(tour_data[column]) < 0 for column in ("price", "transfers", "layover_hours", "dist_center_km")):
                    raise ValueError(f"Строка {line_num}: числовые признаки не должны быть отрицательными")

                records.append(
                    {
                        "group_id": group_id,
                        "archetype": archetype,
                        "cities": str(tour_data["cities"]),
                        "days": int(tour_data["days"]),
                        "price": float(tour_data["price"]),
                        "transfers": int(tour_data["transfers"]),
                        "layover_hours": float(tour_data["layover_hours"]),
                        "hotel_stars": int(tour_data["hotel_stars"]),
                        "dist_center_km": float(tour_data["dist_center_km"]),
                        "pet_friendly": parse_binary(tour_data["pet_friendly"]),
                        "breakfast": parse_binary(tour_data["breakfast"]),
                        "target": int(target),
                    }
                )

    df = pd.DataFrame(records)
    validate_groups(df)
    return df


def split_by_group(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет DataFrame на train и test по группам (group_id), гарантируя, что
    пары туров из одного сравнения (одного group_id) не попадут одновременно в train и test,
    и строки внутри одного group_id остаются сгруппированными вместе для CatBoost Pool.
    """
    validate_groups(df)
    if df["group_id"].nunique() < 2:
        raise ValueError("Для разбиения нужны минимум две группы")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, df["target"], groups=df["group_id"]))

    # Сортируем по group_id (с сохранением порядка внутри группы), так как CatBoost Pool
    # требует, чтобы объекты с одинаковым group_id шли подряд.
    train_df = (
        df.iloc[train_idx]
        .sort_values(by="group_id", kind="mergesort")
        .reset_index(drop=True)
    )
    test_df = (
        df.iloc[test_idx]
        .sort_values(by="group_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return train_df, test_df


def create_catboost_pool(df: pd.DataFrame) -> Pool:
    """
    Создаёт объект catboost.Pool с передачей матрицы признаков data,
    целевой переменной label, идентификатора группы group_id и списка категориальных фич.
    """
    validate_groups(df)
    return Pool(
        data=df[FEATURE_COLUMNS],
        label=df["target"],
        group_id=df["group_id"],
        cat_features=CAT_FEATURES,
    )


def evaluate_ndcg(
    model: CatBoostRanker,
    test_df: pd.DataFrame,
    test_pool: Pool,
) -> Dict[str, float]:
    """
    Вычисляет метрику NDCG на тестовой выборке двумя способами:
    1. Встроенный метод CatBoost (`eval_metrics` на `test_pool`).
    2. Усреднённый по группам `ndcg_score` из `scikit-learn`.
    """
    # 1. Встроенный расчёт метрики NDCG средствами CatBoost
    cb_metrics = model.eval_metrics(
        data=test_pool,
        metrics=["NDCG:top=2;type=Base"],
    )
    cb_ndcg_last = float(cb_metrics["NDCG:top=2;type=Base"][-1])

    # 2. Расчёт NDCG через scikit-learn по каждой паре (group_id)
    preds = model.predict(test_pool)
    eval_df = test_df[["group_id", "target"]].copy()
    eval_df["score"] = preds

    # Так как в каждой группе ровно 2 тура и выборка отсортирована по group_id,
    # формируем матрицы формы (n_groups, 2) для быстрого расчёта
    y_true = eval_df["target"].to_numpy().reshape(-1, 2)
    y_score = eval_df["score"].to_numpy().reshape(-1, 2)
    sklearn_ndcg = float(ndcg_score(y_true, y_score, k=2))

    # Дополнительно посчитаем Pairwise Accuracy (долю угаданных победителей в парах)
    pred_winner = np.argmax(y_score, axis=1)
    true_winner = np.argmax(y_true, axis=1)
    ties = y_score[:, 0] == y_score[:, 1]
    pairwise_acc = float(np.mean(np.where(ties, 0.5, pred_winner == true_winner)))

    return {
        "catboost_ndcg": cb_ndcg_last,
        "sklearn_ndcg": sklearn_ndcg,
        "pairwise_accuracy": pairwise_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение табличного ранжировщика туров на базе CatBoostRanker (LTR)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="part_4060ti_2.jsonl",
        help="Путь к входному JSONL-файлу (по умолчанию: part_4060ti_2.jsonl)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "catboost_travel_ranker.cbm"),
        help="Путь для сохранения обученной модели (по умолчанию: app/data/catboost_travel_ranker.cbm)",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="PairLogit",
        choices=["PairLogit", "PairLogitPairwise", "YetiRank", "YetiRankPairwise"],
        help="Функция потерь для ранжирования (по умолчанию: PairLogit)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Количество итераций бустинга (по умолчанию: 1000)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Глубина деревьев (по умолчанию: 6)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля групп в тестовой выборке (по умолчанию: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости (по умолчанию: 42)",
    )
    args = parser.parse_args()

    # 1. Парсинг и разворачивание (Flattening) данных
    dataset_path = resolve_dataset_path(args.input)
    print(f"[1/4] Чтение и разворачивание датасета из '{dataset_path}'...")
    df = load_and_flatten_jsonl(dataset_path)
    print(
        f"      Загружено строк в DataFrame: {len(df)} "
        f"(уникальных пар group_id: {df['group_id'].nunique()})"
    )

    # 2. Разделение на train и test по group_id (без перемешивания внутри групп)
    print(f"[2/4] Разбиение на train/test по group_id (test_size={args.test_size})...")
    train_df, test_df = split_by_group(
        df,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(
        f"      Train: {len(train_df)} строк ({train_df['group_id'].nunique()} групп) | "
        f"Test: {len(test_df)} строк ({test_df['group_id'].nunique()} групп)"
    )

    train_df, validation_df = split_by_group(train_df, test_size=0.2, random_state=args.seed + 1)

    # Создание объектов catboost.Pool
    train_pool = create_catboost_pool(train_df)
    test_pool = create_catboost_pool(test_df)
    validation_pool = create_catboost_pool(validation_df)

    # 3. Инициализация и обучение CatBoostRanker
    print(
        f"[3/4] Обучение CatBoostRanker "
        f"(loss_function='{args.loss_function}', iterations={args.iterations}, depth={args.depth}, task_type='CPU')..."
    )
    ranker = CatBoostRanker(
        loss_function=args.loss_function,
        eval_metric="NDCG:top=2;type=Base",
        iterations=args.iterations,
        depth=args.depth,
        task_type="CPU",
        random_seed=args.seed,
        verbose=100,
        allow_writing_files=False,
    )

    ranker.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
    )

    # Оценка качества на тестовой выборке
    metrics = evaluate_ndcg(ranker, test_df, test_pool)
    print("\n=== Результаты на тестовой выборке ===")
    print(f"NDCG@2 (CatBoost)      : {metrics['catboost_ndcg']:.5f}")
    print(f"NDCG@2 (scikit-learn)  : {metrics['sklearn_ndcg']:.5f}")
    print(f"Pairwise Accuracy      : {metrics['pairwise_accuracy']:.5f}")

    # 4. Сохранение модели
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(output_path))
    print(f"\n[4/4] Модель успешно сохранена в файл: '{output_path.resolve()}'")


if __name__ == "__main__":
    main()
