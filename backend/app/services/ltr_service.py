"""
Сервис сверхбыстрого табличного ранжирования туров на базе CatBoost LTR (Learning-to-Rank).

Реализует архитектурный концепт Synthetic User Simulation & Distillation (Cold-Start RecSys):
1. Профиль клиента параметризуется через 5 базовых архетипов:
   `family` (семьи), `budget` (эконом), `solo` (соло), `luxury` (люкс), `pet` (пет-туристы).
2. Обученная на синтетических парных сравнениях модель `catboost_travel_ranker.cbm`
   за миллисекунды на CPU пересортировывает кандидатов по 10 признакам:
   `archetype`, `cities`, `days`, `price`, `transfers`, `layover_hours`,
   `hotel_stars`, `dist_center_km`, `pet_friendly`, `breakfast`.
3. После фильтрации и учёта приоритетов пользователю и LLM передаётся top-3.
4. Парные выборы временно накапливаются в памяти процесса для последующего анализа.
   Автоматическое дообучение и постоянное хранение событий не реализованы.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostRanker, Pool
    HAS_CATBOOST = True
except (ImportError, OSError):
    CatBoostRanker = None  # type: ignore
    Pool = None  # type: ignore
    HAS_CATBOOST = False

# Точный порядок и состав признаков, на которых обучена модель catboost_travel_ranker.cbm
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

CAT_FEATURES: List[str] = ["archetype", "cities"]
ARCHETYPES = ("pet", "family", "luxury", "budget", "solo")
NATIVE_ENGINE = "CatBoostRanker (.cbm)"
FALLBACK_ENGINE = "Analytical fallback"


def finite_number(value: Any, default: float = 0.0) -> float:
    """Не допускает NaN/Infinity и некорректные числа в признаках модели."""
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def binary_flag(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in ("1", "true", "yes", "да", "✅ да", "✅"))
    return int(bool(finite_number(value)))

# Канонические маршруты из синтетического датасета для точного попадания в категориальный словарь CatBoost
DESTINATION_TO_CANONICAL_CITIES: Dict[str, str] = {
    "sochi": "Сочи -> Красная Поляна -> Сириус",
    "georgia": "Тбилиси -> Батуми -> Кутаиси",
    "turkey": "Стамбул -> Анталья -> Каппадокия",
    "spb": "Москва -> СПб -> Казань",
    "altai": "Екатеринбург -> Тюмень -> Тобольск",
    # Ближайшие по структуре и логистике маршруты для остальных курортных направлений
    "uae": "Стамбул -> Анталья -> Каппадокия",
    "thailand": "Стамбул -> Анталья -> Каппадокия",
    "maldives": "Стамбул -> Анталья -> Каппадокия",
    "egypt": "Стамбул -> Анталья -> Каппадокия",
}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "catboost_travel_ranker.cbm"


class LTRService:
    """
    Сервис ранжирования кандидатов туров и отелей с помощью CatBoostRanker (`catboost_travel_ranker.cbm`).
    Использует ленивую загрузку (Singleton) и безопасный суррогатный fallback, если библиотека
    `catboost` не установлена в локальном тестовом окружении без Docker.
    """

    _model: Optional[Any] = None
    _model_loaded: bool = False
    _model_path: Path = DEFAULT_MODEL_PATH
    _lock = RLock()
    _load_error: Optional[str] = None
    _inference_failed = False
    _online_feedback_buffer = deque(maxlen=10000)
    _online_feedback_events = 0

    @classmethod
    def get_model_path(cls) -> Path:
        return cls._model_path

    @classmethod
    def load_model(cls, model_path: Optional[Path] = None) -> Optional[Any]:
        """Загружает модель CatBoostRanker из `.cbm` файла (один раз на процесс)."""
        with cls._lock:
            return cls._load_model(model_path)

    @classmethod
    def _load_model(cls, model_path: Optional[Path] = None) -> Optional[Any]:
        if model_path is not None:
            cls._model_path = Path(model_path)
            cls._model_loaded = False
            cls._model = None

        if cls._model_loaded:
            return cls._model

        cls._model_loaded = True
        cls._load_error = None
        cls._inference_failed = False
        target_path = cls.get_model_path()

        if not target_path.is_file():
            cls._load_error = "model_missing"
            logger.warning("Файл модели CatBoost LTR не найден по пути %s. Активен суррогатный LTR-ранжировщик.", target_path)
            return None

        if not HAS_CATBOOST:
            cls._load_error = "catboost_unavailable"
            logger.info(
                "Библиотека catboost не установлена в текущем окружении (файл модели %s найден, %d байт). "
                "Активен резервный аналитический скоринг.",
                target_path.name,
                target_path.stat().st_size,
            )
            return None

        try:
            ranker = CatBoostRanker()
            ranker.load_model(str(target_path))
            if list(ranker.feature_names_) != FEATURE_COLUMNS or ranker.get_cat_feature_indices() != [0, 1]:
                raise ValueError("Схема признаков модели не совпадает с FEATURE_COLUMNS / CAT_FEATURES")
            cls._model = ranker
            logger.info("Модель CatBoostRanker успешно загружена из %s", target_path)
            return cls._model
        except Exception as exc:
            cls._load_error = "model_load_failed"
            logger.warning("Не удалось загрузить CatBoostRanker из %s: %s", target_path, exc)
            cls._model = None
            return None

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Возвращает диагностический статус LTR-ранжировщика для `/health` и UI."""
        path = cls.get_model_path()
        model = cls.load_model()
        return {
            "available": model is not None and not cls._inference_failed,
            "engine": NATIVE_ENGINE if model is not None and not cls._inference_failed else FALLBACK_ENGINE,
            "model_present": path.is_file(),
            "catboost_installed": HAS_CATBOOST,
            "model_loaded": model is not None,
            "fallback_reason": "inference_failed" if cls._inference_failed else cls._load_error,
            "model_file": path.name,
            "model_size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0.0,
            "features_count": len(FEATURE_COLUMNS),
            "online_feedback_events": cls._online_feedback_events,
            "feedback_buffer_size": len(cls._online_feedback_buffer),
            "feedback_buffer_capacity": cls._online_feedback_buffer.maxlen,
        }

    @staticmethod
    def resolve_archetype(profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Определяет архетип путешественника из `profile` в соответствии с пространством
        признаков модели: `pet`, `family`, `luxury`, `budget`, `solo`.
        """
        if not profile:
            return "solo"

        interests = [str(x).lower() for x in (profile.get("interests") or [])]

        if binary_flag(profile.get("has_pets")) or "pet_friendly" in interests:
            return "pet"
        if binary_flag(profile.get("has_kids")) or "family_kids" in interests or finite_number(profile.get("kids_count")) > 0:
            return "family"
        if any(t in interests for t in ("luxury", "extreme_luxury", "luxury_extreme", "splurge")):
            return "luxury"
        if "budget_friendly" in interests or (
            0 < finite_number(profile.get("budget_rub")) <= 110000
        ):
            return "budget"
        if profile.get("party_size") == "соло" or profile.get("adults_count") == 1:
            return "solo"
        return "solo"

    @staticmethod
    def resolve_canonical_cities(destination_id: str, city_name: Optional[str] = None) -> str:
        """
        Сопоставляет `destination_id` или название города с каноническим маршрутом `cities`,
        использованным при обучении CatBoost LTR.
        """
        dest_key = str(destination_id or "").lower().strip()
        if dest_key in DESTINATION_TO_CANONICAL_CITIES:
            return DESTINATION_TO_CANONICAL_CITIES[dest_key]

        city_lower = str(city_name or "").lower()
        if any(w in city_lower for w in ("сочи", "сириус", "полян", "адлер")):
            return "Сочи -> Красная Поляна -> Сириус"
        if any(w in city_lower for w in ("тбилис", "батум", "кутаис", "гудаур", "казбег")):
            return "Тбилиси -> Батуми -> Кутаиси"
        if any(w in city_lower for w in ("стамбул", "анталь", "каппадок", "кемер", "белек", "каш", "алань")):
            return "Стамбул -> Анталья -> Каппадокия"
        if any(w in city_lower for w in ("петербург", "спб", "москв", "казан", "петергоф")):
            return "Москва -> СПб -> Казань"
        if any(w in city_lower for w in ("екатеринбург", "тюмен", "тобольск", "алтай", "манжерок", "чемал")):
            return "Екатеринбург -> Тюмень -> Тобольск"

        return "Сочи -> Красная Поляна -> Сириус"

    @staticmethod
    def _parse_stars(raw_stars: Any) -> int:
        if isinstance(raw_stars, (int, float)):
            return max(1, min(5, int(finite_number(raw_stars, 4))))
        s = str(raw_stars or "4")
        m = re.search(r"(\d+)", s)
        return max(1, min(5, int(m.group(1)))) if m else 4

    @staticmethod
    def _parse_dist_center_km(hotel: Dict[str, Any]) -> float:
        """Преобразует расстояние до моря/центра в километры (`dist_center_km`)."""
        if "dist_center_km" in hotel and hotel["dist_center_km"] is not None:
            try:
                value = float(hotel["dist_center_km"])
                if math.isfinite(value):
                    return max(0.0, value)
            except (ValueError, TypeError):
                pass

        if "distance_to_sea_meters" in hotel and hotel["distance_to_sea_meters"] is not None:
            try:
                meters = finite_number(hotel["distance_to_sea_meters"], 1500)
                return round(max(0.1, meters / 1000.0), 2)
            except (ValueError, TypeError):
                pass

        dist_str = str(hotel.get("distance_to_sea", "") or "")
        m_km = re.search(r"(\d+(?:[.,]\d+)?)\s*км", dist_str.lower())
        if m_km:
            return float(m_km.group(1).replace(",", "."))
        m_meters = re.search(r"(\d+)\s*м\b", dist_str.lower())
        if m_meters:
            return round(max(0.1, float(m_meters.group(1)) / 1000.0), 2)
        if "первая линия" in dist_str.lower() or "пляж" in dist_str.lower():
            return 0.2
        return 1.5

    @staticmethod
    def _parse_pet_friendly(hotel: Dict[str, Any]) -> int:
        val = hotel.get("pet_friendly")
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, float)):
            return binary_flag(val)
        s = str(val or "").strip().lower()
        if s in ("0", "false", "no", "нет", "❌ нет", "❌"):
            return 0
        if binary_flag(s):
            return 1
        feats = [str(f).lower() for f in (hotel.get("features") or [])]
        return int("pet_friendly" in feats)

    @staticmethod
    def _parse_breakfast(hotel: Dict[str, Any]) -> int:
        if hotel.get("breakfast") is not None:
            return binary_flag(hotel["breakfast"])
        meals = str(hotel.get("meals", "") or "").lower()
        if not meals or "без питания" in meals or "room only" in meals:
            return 0
        if any(w in meals for w in ("завтрак", "all inclusive", "все включено", "всё включено", "полупансион", "пансион")):
            return 1
        return 0

    @classmethod
    def extract_candidate_features(
        cls,
        hotel: Dict[str, Any],
        destination_id: str,
        profile: Optional[Dict[str, Any]] = None,
        nights: int = 7,
        logistics: Optional[Dict[str, Any]] = None,
        breakdown: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Формирует вектор из 10 признаков кандидата для подачи в `CatBoostRanker`:
        - `archetype` (str)
        - `cities` (str)
        - `days` (int)
        - `price` (float)
        - `transfers` (int)
        - `layover_hours` (float)
        - `hotel_stars` (int)
        - `dist_center_km` (float)
        - `pet_friendly` (int: 1/0)
        - `breakfast` (int: 1/0)
        """
        profile = profile or {}
        archetype = cls.resolve_archetype(profile)
        cities = cls.resolve_canonical_cities(destination_id, hotel.get("city"))
        days = max(1, int(finite_number(nights if nights is not None else profile.get("nights"), 7)))

        # Полная стоимость пакета или оценка на основе цены номера за ночь
        if breakdown and "mandatory" in breakdown:
            price = max(0.0, finite_number((breakdown["mandatory"] or {}).get("total_mandatory_rub")))
        else:
            per_night = max(0.0, finite_number(hotel.get("price_per_night_rub"), 5000))
            rooms = max(1, int(finite_number(profile.get("rooms_count"), 1)))
            price = per_night * days * rooms + 25000.0

        # Параметры пересадок и стыковок из графа логистики Neo4j
        transfers_count = 1
        layover_hours = 1.5
        if logistics:
            flight = logistics.get("flight") or {}
            timeline = logistics.get("timeline") or {}
            layover_details = timeline.get("layover_details") or {}
            is_connecting = (
                flight.get("flight_type") == "connecting"
                or bool(layover_details.get("is_connecting"))
                or finite_number(flight.get("layover_duration_minutes")) > 0
            )
            has_ground_transfer = not bool(profile.get("without_transfer"))
            transfers_count = (1 if is_connecting else 0) + (1 if has_ground_transfer else 0)
            layover_min = finite_number(
                layover_details.get("duration_minutes")
                or flight.get("layover_duration_minutes")
                or 0
            )
            if layover_min > 0:
                layover_hours = round(float(layover_min) / 60.0, 2)
            else:
                buffer_min = max(0.0, finite_number(timeline.get("post_flight_buffer_min"), 60))
                layover_hours = round(float(buffer_min) / 60.0, 2)

        return {
            "archetype": archetype,
            "cities": cities,
            "days": days,
            "price": price,
            "transfers": int(transfers_count),
            "layover_hours": float(layover_hours),
            "hotel_stars": cls._parse_stars(hotel.get("stars")),
            "dist_center_km": cls._parse_dist_center_km(hotel),
            "pet_friendly": cls._parse_pet_friendly(hotel),
            "breakfast": cls._parse_breakfast(hotel),
        }

    @classmethod
    def _surrogate_score(cls, feat: Dict[str, Any]) -> float:
        """
        Резервная аналитическая формула при недоступности CatBoost.
        Не заменяет обучение и не считается результатом инференса бинарной модели.
        """
        arch = feat["archetype"]
        price = float(feat["price"])
        stars = int(feat["hotel_stars"])
        dist = float(feat["dist_center_km"])
        transfers = int(feat["transfers"])
        layover = float(feat["layover_hours"])
        pet = int(feat["pet_friendly"])
        bf = int(feat["breakfast"])
        days = int(feat["days"])

        score = 0.0
        # Общие факторы комфорта логистики
        score -= 0.35 * transfers
        score -= 0.12 * layover
        score -= 0.18 * dist
        score += 0.04 * days

        if arch == "pet":
            score += 3.5 if pet == 1 else -3.5
            score += 0.35 * stars
            score -= 0.000004 * price
        elif arch == "family":
            score += 1.2 if bf == 1 else -0.8
            score += 0.55 * stars
            score -= 0.55 * transfers
            score -= 0.20 * layover
            score -= 0.000006 * price
        elif arch == "luxury":
            score += 1.15 * stars
            score += 0.8 if bf == 1 else -0.3
            score -= 0.35 * dist
            score -= 0.000002 * price
        elif arch == "budget":
            score -= 0.000022 * price
            score += 0.5 if bf == 1 else 0.0
            score += 0.15 * stars
        else:  # solo
            score += 0.40 * stars
            score += 0.55 if bf == 1 else 0.0
            score -= 0.25 * dist
            score -= 0.000009 * price

        return round(float(score), 5)

    @classmethod
    def score_features_batch(cls, features_list: List[Dict[str, Any]]) -> Tuple[List[float], float, str]:
        """
        Вычисляет LTR-скоры для списка векторов признаков.
        Возвращает кортеж: `(scores, latency_ms, engine_name)`.
        """
        if not features_list:
            return [], 0.0, "empty"

        t0 = time.perf_counter()
        model = cls.load_model()

        # Категории всегда строки, числовые фичи конечны даже для неполного входа.
        features_list = [cls.normalize_features(feat) for feat in features_list]
        if model is not None and Pool is not None:
            try:
                pool = Pool(
                    data=[[feat[column] for column in FEATURE_COLUMNS] for feat in features_list],
                    feature_names=FEATURE_COLUMNS,
                    cat_features=CAT_FEATURES,
                )
                preds = model.predict(pool)
                values = np.asarray(preds, dtype=float).ravel()
                if len(values) != len(features_list) or not np.isfinite(values).all():
                    raise ValueError("Модель вернула некорректные скоры")
                scores = [round(float(x), 5) for x in values]
                cls._inference_failed = False
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return scores, latency_ms, NATIVE_ENGINE
            except Exception as exc:
                cls._inference_failed = True
                logger.warning("Ошибка инференса CatBoostRanker, переключение на суррогатный скорер: %s", exc)

        scores = [cls._surrogate_score(f) for f in features_list]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return scores, latency_ms, FALLBACK_ENGINE

    @staticmethod
    def normalize_features(feat: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for column in FEATURE_COLUMNS:
            value = feat.get(column)
            if column in CAT_FEATURES:
                result[column] = str(value) if value is not None and not (isinstance(value, float) and not math.isfinite(value)) else "unknown"
            elif column in ("pet_friendly", "breakfast"):
                result[column] = binary_flag(value)
            else:
                result[column] = max(0.0, finite_number(value))
        result["days"] = max(1, int(result["days"]))
        result["hotel_stars"] = max(1, min(5, int(result["hotel_stars"])))
        result["transfers"] = int(result["transfers"])
        return result

    @classmethod
    def rank_hotel_candidates(
        cls,
        hotels: List[Dict[str, Any]],
        destination_id: str,
        profile: Optional[Dict[str, Any]] = None,
        nights: int = 7,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Быстро пересортировывает кандидатов отелей (например, для каждого города в Multi-City туре)
        на основе `CatBoostRanker` и возвращает отсортированный список + метаданные.
        """
        if not hotels:
            return [], {
                "candidates_count": 0,
                "top_k": 0,
                "archetype": cls.resolve_archetype(profile),
                "latency_ms": 0.0,
                "engine": "none",
                "top_score": 0.0,
            }

        features_batch = [
            cls.extract_candidate_features(
                hotel=h,
                destination_id=destination_id,
                profile=profile,
                nights=nights,
            )
            for h in hotels
        ]
        scores, latency_ms, engine_name = cls.score_features_batch(features_batch)

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, (h, score, feat) in enumerate(zip(hotels, scores, features_batch)):
            h_copy = dict(h)
            h_copy["ltr_score"] = score
            h_copy["ltr_archetype"] = feat["archetype"]
            scored.append((score, idx, h_copy))

        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [item[2] for item in scored]
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k не должен быть отрицательным")
            ranked = ranked[:top_k]

        meta = {
            "candidates_count": len(hotels),
            "top_k": len(ranked),
            "archetype": features_batch[0]["archetype"],
            "canonical_cities": features_batch[0]["cities"],
            "latency_ms": latency_ms,
            "engine": engine_name,
            "top_score": ranked[0]["ltr_score"] if ranked else 0.0,
            "top_hotel": ranked[0].get("name") if ranked else None,
        }
        return ranked, meta

    @classmethod
    def rank_evaluated_packages(
        cls,
        evaluated_options: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
        destination_id: str,
        profile: Optional[Dict[str, Any]] = None,
        nights: int = 7,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]:
        """
        Пересортировывает список собранных пакетов `(hotel, logistics, budget_breakdown)`
        с помощью модели `CatBoostRanker` за ~2–3 мс и возвращает отранжированный список
        вместе с метаданными ранжирования для блока `Agent Reasoning`.
        """
        if not evaluated_options:
            return [], {
                "candidates_count": 0,
                "top_k": 0,
                "archetype": cls.resolve_archetype(profile),
                "latency_ms": 0.0,
                "engine": "none",
                "top_score": 0.0,
            }

        features_batch = [
            cls.extract_candidate_features(
                hotel=opt[0],
                destination_id=destination_id,
                profile=profile,
                nights=nights,
                logistics=opt[1],
                breakdown=opt[2],
            )
            for opt in evaluated_options
        ]

        scores, latency_ms, engine_name = cls.score_features_batch(features_batch)

        scored_options: List[Tuple[float, int, Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]] = []
        for idx, (opt, score, feat) in enumerate(zip(evaluated_options, scores, features_batch)):
            hotel_copy = dict(opt[0])
            hotel_copy["ltr_score"] = score
            hotel_copy["ltr_archetype"] = feat["archetype"]
            scored_options.append((score, idx, (hotel_copy, opt[1], opt[2])))

        # Сортируем по убыванию LTR-скора (при равенстве сохраняем исходный порядок)
        scored_options.sort(key=lambda item: (-item[0], item[1]))

        ranked = [item[2] for item in scored_options]
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k не должен быть отрицательным")
            ranked = ranked[:top_k]

        meta = {
            "candidates_count": len(evaluated_options),
            "top_k": len(ranked),
            "archetype": features_batch[0]["archetype"],
            "canonical_cities": features_batch[0]["cities"],
            "latency_ms": latency_ms,
            "engine": engine_name,
            "top_score": ranked[0][0]["ltr_score"] if ranked else 0.0,
            "top_hotel": ranked[0][0].get("name") if ranked else None,
        }
        return ranked, meta

    @classmethod
    def record_online_feedback(
        cls,
        archetype: str,
        chosen_tour: Dict[str, Any],
        rejected_tour: Dict[str, Any],
        margin: int = 1,
    ) -> Dict[str, Any]:
        """
        Шаг 4 концепта Synthetic User Simulation:
        Записывает реальный сигнал конверсии/выбора пользователя (онлайн-лог кликстрима)
        в формате парного сравнения для последующего онлайн-обновления весов CatBoost LTR.
        """
        if archetype not in ARCHETYPES or not isinstance(margin, int) or isinstance(margin, bool) or not 1 <= margin <= 10:
            raise ValueError("Некорректный архетип или margin (ожидается целое число от 1 до 10)")
        event = {
            "pair_id": uuid4().int & ((1 << 63) - 1),
            "archetype": archetype,
            "tour_a": deepcopy(chosen_tour),
            "tour_b": deepcopy(rejected_tour),
            "winner": "A",
            "margin": margin,
            "source": "real_clickstream_conversion",
        }
        with cls._lock:
            cls._online_feedback_buffer.append(event)
            cls._online_feedback_events += 1
            buffer_size = len(cls._online_feedback_buffer)
        return {
            "recorded": True,
            "buffer_size": buffer_size,
            "event": deepcopy(event),
        }
