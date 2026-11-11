#!/usr/bin/env python3
"""
Скрипт для обучения модели ранжирования CatBoostRanker (Learning to Rank)
на объединенном датасете парных сравнений туров (CSAI 2026).
Поддерживает вычисление нормализованных признаков (log_price, log_price_per_day, stars_per_10k).
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


# Список 12 признаков для обучения модели
FEATURE_COLUMNS: List[str] = [
    "archetype",
    "cities",
    "days",
    "log_price",
    "log_price_per_day",
    "transfers",
    "layover_hours",
    "hotel_stars",
    "dist_center_km",
    "pet_friendly",
    "breakfast",
    "stars_per_10k",
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


def resolve_dataset_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    fallback = p.with_name("part_4060ti.jsonl")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Dataset path {path_str} not found")


def load_and_flatten_jsonl(file_path: Path | str) -> pd.DataFrame:
    return load_and_flatten_jsonl_sources([Path(file_path)])


def load_and_flatten_jsonl_sources(file_paths: List[Path]) -> pd.DataFrame:
    """
    Читает список JSONL-файлов с парными сравнениями туров и разворачивает каждую строку
    в ДВЕ записи DataFrame (для tour_a и tour_b), объединённые общим group_id.
    """
    records: List[Dict[str, Any]] = []
    seen_groups = set()
    pair_id_offset = 0

    for fpath in file_paths:
        if not fpath.exists():
            print(f"[WARN] Файл не найден: {fpath}, пропускаем...")
            continue

        print(f"      Считывание данных из: {fpath} (размер: {fpath.stat().st_size / (1024*1024):.1f} МБ)...")
        with fpath.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                raw_pair_id = int(row["pair_id"])
                group_id = raw_pair_id + pair_id_offset
                if raw_pair_id in seen_groups:
                    raise ValueError(f"Дублирующийся pair_id={raw_pair_id} в строке {line_num}")
                seen_groups.add(raw_pair_id)

                archetype = str(row["archetype"])
                winner = str(row["winner"]).strip().upper()
                if winner not in ("A", "B"):
                    raise ValueError(f"Некорректный winner: {winner!r} в строке {line_num}")

                for tour_key, winner_code in (("tour_a", "A"), ("tour_b", "B")):
                    tour_data = row[tour_key]
                    target = 1 if winner == winner_code else 0
                    days = max(1, int(tour_data["days"]))
                    price = float(tour_data["price"])
                    if not math.isfinite(price):
                        raise ValueError(f"Нечисловая цена в строке {line_num}")
                    price_per_day = price / days
                    stars = int(tour_data["hotel_stars"])
                    pet_friendly = parse_binary(tour_data["pet_friendly"])
                    breakfast = parse_binary(tour_data["breakfast"])
                    transfers = int(tour_data["transfers"])
                    layover_hours = float(tour_data["layover_hours"])
                    if not math.isfinite(layover_hours):
                        raise ValueError(f"Нечисловой layover_hours в строке {line_num}")
                    dist_center_km = float(tour_data["dist_center_km"])
                    if not math.isfinite(dist_center_km):
                        raise ValueError(f"Нечисловой dist_center_km в строке {line_num}")

                    log_price = float(np.log1p(price))
                    log_price_per_day = float(np.log1p(price_per_day))
                    stars_per_10k = float(stars / max(0.1, price_per_day / 10000.0))

                    records.append(
                        {
                            "group_id": group_id,
                            "archetype": archetype,
                            "cities": str(tour_data["cities"]),
                            "days": days,
                            "price": price,
                            "log_price": log_price,
                            "log_price_per_day": log_price_per_day,
                            "transfers": transfers,
                            "layover_hours": layover_hours,
                            "hotel_stars": stars,
                            "dist_center_km": dist_center_km,
                            "pet_friendly": pet_friendly,
                            "breakfast": breakfast,
                            "stars_per_10k": stars_per_10k,
                            "target": int(target),
                        }
                    )

        pair_id_offset += 1_000_000

    if not records:
        raise ValueError("Датасет пуст или не содержит корректных записей")

    df = pd.DataFrame(records)
    # Сортируем по group_id (CatBoost требует, чтобы объекты группы шли подряд)
    df = df.sort_values(by="group_id", kind="mergesort").reset_index(drop=True)
    validate_groups(df)
    return df


def split_by_group(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет DataFrame на train и test по group_id."""
    validate_groups(df)
    if df["group_id"].nunique() < 2:
        raise ValueError("Для сплита требуется минимум 2 группы")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["target"], groups=df["group_id"]))
    train_df = df.iloc[train_idx].sort_values(by="group_id", kind="mergesort").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values(by="group_id", kind="mergesort").reset_index(drop=True)
    return train_df, test_df


def create_catboost_pool(df: pd.DataFrame) -> Pool:
    """Создаёт catboost.Pool с матрицей 12 признаков."""
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
    """Вычисляет NDCG@2 и Pairwise Accuracy на отложенной выборке."""
    preds = model.predict(test_pool)
    eval_df = test_df[["group_id", "target"]].copy()
    eval_df["score"] = preds

    y_true = eval_df["target"].to_numpy().reshape(-1, 2)
    y_score = eval_df["score"].to_numpy().reshape(-1, 2)
    sklearn_ndcg = float(ndcg_score(y_true, y_score, k=2))

    pred_winner = np.argmax(y_score, axis=1)
    true_winner = np.argmax(y_true, axis=1)
    pairwise_acc = float(np.mean(pred_winner == true_winner))

    return {
        "sklearn_ndcg": sklearn_ndcg,
        "pairwise_accuracy": pairwise_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение CatBoostRanker на объединенном пуле парных сравнений (CSAI 2026)."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=[
            "D:/Repos/ostrovok-data/part_4060ti.jsonl",
            "D:/Repos/ostrovok-data/part_4080s.jsonl",
        ],
        help="Пути к входным JSONL-файлам с парами туров",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "catboost_travel_ranker.cbm"),
        help="Путь для сохранения обученной модели",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="PairLogit",
        choices=["PairLogit", "PairLogitPairwise", "YetiRank"],
        help="Функция потерь (по умолчанию: PairLogit)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Количество итераций бустинга (по умолчанию: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        help="Скорость обучения (learning rate, по умолчанию: 0.08)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Глубина деревьев (по умолчанию: 6)",
    )
    parser.add_argument(
        "--border-count",
        type=int,
        default=254,
        help="Количество корзин квантования для числовых признаков (по умолчанию: 254)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Доля групп в тестовой выборке (по умолчанию: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (по умолчанию: 42)",
    )
    args = parser.parse_args()

    # 1. Чтение и сведение данных
    file_paths = [Path(p) for p in args.inputs]
    print("=" * 70)
    print("CSAI 2026: Обучение CatBoostRanker на объединенном датасете")
    print("=" * 70)
    print(f"\n[1/4] Считывание и векторизация признаков из {len(file_paths)} файлов...")
    df = load_and_flatten_jsonl_sources(file_paths)
    print(f"      Всего загружено: {len(df)} записей ({df['group_id'].nunique()} парных сравнений)")

    # 2. Train/Test Split
    print(f"\n[2/4] Разделение на train и validation по группам (test_size={args.test_size})...")
    train_df, test_df = split_by_group(df, test_size=args.test_size, random_state=args.seed)
    print(f"      Train : {len(train_df)} записей ({train_df['group_id'].nunique()} пар)")
    print(f"      Valid : {len(test_df)} записей ({test_df['group_id'].nunique()} пар)")

    train_pool = create_catboost_pool(train_df)
    test_pool = create_catboost_pool(test_df)

    # 3. Обучение модели
    print(f"\n[3/4] Обучение CatBoostRanker (loss='{args.loss_function}', depth={args.depth}, lr={args.learning_rate}, borders={args.border_count})...")
    ranker = CatBoostRanker(
        loss_function=args.loss_function,
        eval_metric="NDCG:top=2;type=Base",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        border_count=args.border_count,
        task_type="CPU",
        random_seed=args.seed,
        verbose=100,
        allow_writing_files=False,
    )

    ranker.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
    )

    best_iter = ranker.get_best_iteration()
    print(f"\n[INFO] Лучшая итерация на валидации: {best_iter}")

    metrics = evaluate_ndcg(ranker, test_df, test_pool)
    print("\n=== Результаты на отложенной валидационной выборке ===")
    print(f"NDCG@2            : {metrics['sklearn_ndcg']:.5f}")
    print(f"Pairwise Accuracy : {metrics['pairwise_accuracy']*100:.2f}%")

    # Важность признаков
    print("\n=== Важность признаков (Feature Importance) ===")
    imps = ranker.get_feature_importance(type="PredictionValuesChange")
    for feat, imp in sorted(zip(FEATURE_COLUMNS, imps), key=lambda x: -x[1]):
        print(f"  {feat:<20}: {imp:6.2f}%")

    # 4. Сохранение модели
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(out_path))
    print(f"\n[4/4] Модель успешно сохранена в: {out_path.resolve()}")
    print("Готово!")


if __name__ == "__main__":
    main()
