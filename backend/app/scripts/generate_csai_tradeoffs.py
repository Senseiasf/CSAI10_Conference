#!/usr/bin/env python3
"""
Генератор выборки допустимых пар туров с мягкими компромиссами (CSAI 2026, Этап 3).

Собирает пары туров из снимка каталога и графа Neo4j, удовлетворяющие требованиям:
1. Обе стороны (Tour A и Tour B) строго проходят единый жесткий фильтр слотов и лимит бюджета.
2. Пары представляют реальные нетривиальные компромиссы (trade-offs):
   - Цена против прямого рейса (direct_vs_price)
   - Время вылета против цены (time_vs_price)
   - Класс отеля / звезды против цены (stars_vs_price)
   - Завтрак / тип питания против цены (breakfast_vs_price)
   - Логистический комфорт / первая линия против цены (comfort_vs_price)
3. Покрывает все 5 архетипов: family, budget, solo, luxury, pet.
4. Отфильтровывает тривиальные пары:
   - Одинаковые туры (A == B)
   - Пары доминирования (где один тур и дешевле, и превосходит другой по всем характеристикам)
   - Пары с явной числовой суммой доплаты (решаемые детерминированным правилом сравнения)
5. Для каждого тура формирует снимок признаков FEATURE_SCHEMA_V2 (22 признака).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.scripts.benchmark_routes import (
    Route,
    assess,
    build_routes,
    read_graph_routes,
    select_return,
)
from app.services.graph_service import minutes_since_midnight
from app.services.ltr_service import (
    ARCHETYPES,
    DESTINATION_TO_CANONICAL_CITIES_V2,
    FEATURE_COLUMNS_V2,
    FEATURE_SCHEMA_V2,
    LTRService,
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = BACKEND_DIR / "app" / "data" / "destinations.json"
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "app" / "data" / "csai_tradeoff_candidates_v2.json"

# Сценарии и запросы по архетипам и направлениям (без явных числовых сумм доплаты)
SCENARIO_TEMPLATES: List[Dict[str, Any]] = [
    # --- 1. FAMILY ARCHETYPE ---
    {
        "family_id": "family_direct_vs_price_turkey",
        "archetype": "family",
        "destination_id": "turkey",
        "adults_count": 2,
        "kids_count": 1,
        "has_pets": False,
        "has_kids": True,
        "nights": 7,
        "budget": 350000,
        "origin_airport": "VKO",
        "tradeoff_family": "direct_vs_price",
        "soft_preference": {"attribute": "direct", "max_extra_rub": None},
        "queries": [
            "Летим на отдых в Турцию семьей с ребенком на неделю, бюджет до 350 тыс. Предпочитаем прямой рейс, но ради хорошей экономии готовы рассмотреть пересадку.",
            "Семья из троих человек в Турцию на 7 ночей. Хотелось бы прямой перелет без стыковок, но если с пересадкой существенно дешевле, согласны лететь с ней.",
        ],
    },
    {
        "family_id": "family_time_vs_price_sochi",
        "archetype": "family",
        "destination_id": "sochi",
        "adults_count": 2,
        "kids_count": 2,
        "has_pets": False,
        "has_kids": True,
        "nights": 6,
        "budget": 240000,
        "origin_airport": "SVO",
        "tradeoff_family": "time_vs_price",
        "soft_preference": {"attribute": "morning", "max_extra_rub": None},
        "queries": [
            "Едем с двумя детьми в Сочи на 6 ночей. Удобнее вылететь утром, чтобы дети не капризничали, но из-за ощутимой переплаты готовы полететь днем.",
            "Отпуск в Сочи с детьми на 6 дней. Хотелось бы утренний вылет, но если дневной рейс заметно выгоднее, рассмотрим его.",
        ],
    },
    {
        "family_id": "family_breakfast_vs_price_egypt",
        "archetype": "family",
        "destination_id": "egypt",
        "adults_count": 2,
        "kids_count": 1,
        "has_pets": False,
        "has_kids": True,
        "nights": 8,
        "budget": 380000,
        "origin_airport": "DME",
        "tradeoff_family": "breakfast_vs_price",
        "soft_preference": {"attribute": "breakfast", "max_extra_rub": None},
        "queries": [
            "Тур в Египет для семьи с ребенком на 8 ночей. Предпочтительно питание все включено или завтрак, но при существенной экономии рассмотрим варианты без питания.",
        ],
    },

    # --- 2. BUDGET ARCHETYPE ---
    {
        "family_id": "budget_direct_vs_price_georgia",
        "archetype": "budget",
        "destination_id": "georgia",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 6,
        "budget": 160000,
        "origin_airport": "VKO",
        "tradeoff_family": "direct_vs_price",
        "soft_preference": {"attribute": "direct", "max_extra_rub": None},
        "queries": [
            "Недорогой отдых в Грузии вдвоем на 6 дней. Хочется прямой рейс, но если стыковочный рейс ощутимо дешевле, полетим с пересадкой.",
            "Вдвоем в Тбилиси экономно. Лучше бы без пересадок, однако при существенной разнице в цене выберем пересадку.",
        ],
    },
    {
        "family_id": "budget_stars_vs_price_spb",
        "archetype": "budget",
        "destination_id": "spb",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 5,
        "budget": 120000,
        "origin_airport": "SVO",
        "tradeoff_family": "stars_vs_price",
        "soft_preference": {"attribute": "stars", "max_extra_rub": None},
        "queries": [
            "Поездка в Санкт-Петербург на 5 дней для двоих из Шереметьево. Хотелось бы отель 4 звезды, но если уютная трешка обойдется существенно дешевле, возьмем 3 звезды.",
            "Экономный тур в Питер на 5 дней из SVO. Предпочтительнее комфортный отель повыше классом, если разница в цене оправдана.",
        ],
    },
    {
        "family_id": "budget_time_vs_price_altai",
        "archetype": "budget",
        "destination_id": "altai",
        "adults_count": 1,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 7,
        "budget": 110000,
        "origin_airport": "SVO",
        "tradeoff_family": "time_vs_price",
        "soft_preference": {"attribute": "morning", "max_extra_rub": None},
        "queries": [
            "Один человек на Алтай на 7 дней в пределах 110 тыс. Хотелось бы вылететь утром, но если вечерний рейс ощутимо дешевле, полечу вечером.",
        ],
    },

    # --- 3. SOLO ARCHETYPE ---
    {
        "family_id": "solo_time_vs_price_uae",
        "archetype": "solo",
        "destination_id": "uae",
        "adults_count": 1,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 6,
        "budget": 220000,
        "origin_airport": "DME",
        "tradeoff_family": "time_vs_price",
        "soft_preference": {"attribute": "day", "max_extra_rub": None},
        "queries": [
            "Лечу один в Дубай на 6 ночей. Предпочитаю вылет днем без раннего подъема, но при заметной разнице в цене готов лететь утренним рейсом.",
            "Соло-тур в ОАЭ на 6 ночей. Хотелось бы дневной рейс, но если утренний существенно дешевле, выберу утро.",
        ],
    },
    {
        "family_id": "solo_stars_vs_price_thailand",
        "archetype": "solo",
        "destination_id": "thailand",
        "adults_count": 1,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 9,
        "budget": 280000,
        "origin_airport": "DME",
        "tradeoff_family": "stars_vs_price",
        "soft_preference": {"attribute": "stars", "max_extra_rub": None},
        "queries": [
            "Соло путешествие в Таиланд на Пхукет на 9 ночей из Домодедово. Хотелось бы резорт 5 звезд, но если отель 4 звезды ощутимо дешевле, остановлюсь в 4*.",
        ],
    },
    {
        "family_id": "solo_direct_vs_price_sochi",
        "archetype": "solo",
        "destination_id": "sochi",
        "adults_count": 1,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 5,
        "budget": 120000,
        "origin_airport": "VKO",
        "tradeoff_family": "direct_vs_price",
        "soft_preference": {"attribute": "direct", "max_extra_rub": None},
        "queries": [
            "Соло тур в Сочи на 5 дней. Хотелось бы прямой рейс, но если с пересадкой значительно дешевле, полечу со стыковкой.",
        ],
    },

    # --- 4. LUXURY ARCHETYPE ---
    {
        "family_id": "luxury_stars_vs_price_maldives",
        "archetype": "luxury",
        "destination_id": "maldives",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 7,
        "budget": 950000,
        "origin_airport": "SVO",
        "tradeoff_family": "stars_vs_price",
        "soft_preference": {"attribute": "stars", "max_extra_rub": None},
        "queries": [
            "Премиальный отдых на Мальдивах для двоих на неделю. Желателен топовый 5* резорт люкс, но если отличный отель 4* или 5* стандарт заметно дешевле, рассмотрим компромисс.",
            "Люкс-тур на Мальдивы на 7 ночей. Предпочитаем высший класс отеля, если доплата за дополнительную звезду оправдана качеством.",
        ],
    },
    {
        "family_id": "luxury_direct_vs_price_uae",
        "archetype": "luxury",
        "destination_id": "uae",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 7,
        "budget": 600000,
        "origin_airport": "DME",
        "tradeoff_family": "direct_vs_price",
        "soft_preference": {"attribute": "direct", "max_extra_rub": None},
        "queries": [
            "Люксовый отдых в Дубае на двоих на 7 ночей. Предпочтителен прямой комфортный перелет, но при очень большой переплате рассмотрим премиальную стыковку.",
        ],
    },
    {
        "family_id": "luxury_comfort_vs_price_turkey",
        "archetype": "luxury",
        "destination_id": "turkey",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": False,
        "has_kids": False,
        "nights": 7,
        "budget": 500000,
        "origin_airport": "VKO",
        "tradeoff_family": "comfort_vs_price",
        "soft_preference": {"attribute": "proximity", "max_extra_rub": None},
        "queries": [
            "Элитный тур в Турцию (Белек или Бодрум) на неделю. Хотелось бы отель прямо на первой линии моря, но если второй ряд заметно дешевле при равном сервисе, рассмотрим его.",
        ],
    },

    # --- 5. PET ARCHETYPE ---
    {
        "family_id": "pet_direct_vs_price_sochi",
        "archetype": "pet",
        "destination_id": "sochi",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": True,
        "has_kids": False,
        "nights": 6,
        "budget": 200000,
        "origin_airport": "SVO",
        "tradeoff_family": "direct_vs_price",
        "soft_preference": {"attribute": "direct", "max_extra_rub": None},
        "queries": [
            "Едем в Сочи с собакой на 6 ночей, отель строго дог-френдли. С питомцем лучше лететь прямым рейсом, но если рейс с пересадкой существенно дешевле, рассмотрим стыковку.",
            "Отдых в Сочи с домашним животным на двоих на 6 дней. Хотелось бы прямой рейс для питомца, но если пересадка позволит заметно сэкономить, согласны.",
        ],
    },
    {
        "family_id": "pet_time_vs_price_georgia",
        "archetype": "pet",
        "destination_id": "georgia",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": True,
        "has_kids": False,
        "nights": 7,
        "budget": 230000,
        "origin_airport": "VKO",
        "tradeoff_family": "time_vs_price",
        "soft_preference": {"attribute": "morning", "max_extra_rub": None},
        "queries": [
            "Тур в Грузию с собакой на 7 ночей. Удобнее вылететь утром, чтобы днем успеть заселиться с питомцем, но из-за ощутимой разницы в цене готовы вылететь днем.",
        ],
    },
    {
        "family_id": "pet_stars_vs_price_altai",
        "archetype": "pet",
        "destination_id": "altai",
        "adults_count": 2,
        "kids_count": 0,
        "has_pets": True,
        "has_kids": False,
        "nights": 7,
        "budget": 210000,
        "origin_airport": "SVO",
        "tradeoff_family": "stars_vs_price",
        "soft_preference": {"attribute": "stars", "max_extra_rub": None},
        "queries": [
            "Тур на Алтай с собакой на 7 ночей в пет-френдли отель. Хотелось бы эко-отель 4-5 звезд, но если хорошая база 3-4 звезды заметно дешевле, выберем ее.",
        ],
    },
]


def file_sha256(path: Path) -> str:
    """Вычисляет sha256 хеш файла."""
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_tradeoff_admissibility(
    candidate: Dict[str, Any],
    scenario: Dict[str, Any],
) -> bool:
    """
    Проверяет, что тур строго удовлетворяет жестким ограничениям запроса и бюджету.
    """
    route = candidate["route"]
    flight = route.flight
    hotel = route.hotel

    # 1. Происхождение и аэропорт
    req_origin = scenario.get("origin_airport")
    if req_origin and flight.get("from") != req_origin:
        return False

    # 2. Питомцы (pet_friendly)
    if scenario.get("has_pets") and not hotel.get("pet_friendly"):
        return False

    # 3. Дети (family_friendly и запрет adults_only)
    if scenario.get("has_kids"):
        if not hotel.get("family_friendly") or "adults_only" in (hotel.get("features") or []):
            return False

    # 4. Бюджет
    budget = scenario.get("budget")
    price = candidate["price"]
    if budget is not None and price > budget:
        return False

    return True


def evaluate_candidate_advantages(
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
    tradeoff_family: str,
    pref_attr: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Анализирует преимущества каждого из двух туров.
    Возвращает (advantages_a, advantages_b).
    """
    fl_a = candidate_a["route"].flight
    fl_b = candidate_b["route"].flight
    h_a = candidate_a["route"].hotel
    h_b = candidate_b["route"].hotel
    p_a = candidate_a["price"]
    p_b = candidate_b["price"]

    adv_a = []
    adv_b = []

    # 1. Цена
    if p_a < p_b:
        adv_a.append("lower_price")
    elif p_b < p_a:
        adv_b.append("lower_price")

    # 2. Прямой перелет
    is_dir_a = fl_a.get("flight_type") == "direct"
    is_dir_b = fl_b.get("flight_type") == "direct"
    if is_dir_a and not is_dir_b:
        adv_a.append("direct_flight")
    elif is_dir_b and not is_dir_a:
        adv_b.append("direct_flight")

    # 3. Время вылета
    dep_a = fl_a.get("departure_time")
    dep_b = fl_b.get("departure_time")
    if dep_a and dep_b and pref_attr in ("morning", "day", "evening"):
        m_a = minutes_since_midnight(dep_a)
        m_b = minutes_since_midnight(dep_b)
        ranges = {"morning": (360, 720), "day": (720, 1080), "evening": (1080, 1440)}
        start, end = ranges[pref_attr]
        match_a = start <= m_a < end
        match_b = start <= m_b < end
        if match_a and not match_b:
            adv_a.append(f"{pref_attr}_departure")
        elif match_b and not match_a:
            adv_b.append(f"{pref_attr}_departure")

    # 4. Звезды отеля
    stars_a = LTRService._parse_stars(h_a.get("stars"))
    stars_b = LTRService._parse_stars(h_b.get("stars"))
    if stars_a > stars_b:
        adv_a.append(f"higher_hotel_stars_{stars_a}_vs_{stars_b}")
    elif stars_b > stars_a:
        adv_b.append(f"higher_hotel_stars_{stars_b}_vs_{stars_a}")

    # 5. Завтрак / питание
    bf_a = LTRService._parse_breakfast(h_a)
    bf_b = LTRService._parse_breakfast(h_b)
    if bf_a and not bf_b:
        adv_a.append("breakfast_included")
    elif bf_b and not bf_a:
        adv_b.append("breakfast_included")

    # 6. Близость к морю
    sea_a, has_sea_a = LTRService._parse_dist_sea_km_v2(h_a)
    sea_b, has_sea_b = LTRService._parse_dist_sea_km_v2(h_b)
    if has_sea_a and has_sea_b and abs(sea_a - sea_b) >= 0.3:
        if sea_a < sea_b:
            adv_a.append("closer_to_sea")
        elif sea_b < sea_a:
            adv_b.append("closer_to_sea")

    return adv_a, adv_b


def is_valid_tradeoff_pair(
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any],
    tradeoff_family: str,
    pref_attr: Optional[str] = None,
    min_price_delta: float = 2000.0,
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Проверяет пару на нетривиальность компромисса и отсутствие строгого доминирования.
    Возвращает (is_valid, reason, adv_a, adv_b).
    """
    # 1. Не должны совпадать
    if cand_a["route"].key == cand_b["route"].key:
        return False, "identical_routes", [], []

    # 2. Разница цен должна быть не менее min_price_delta
    price_delta = abs(cand_a["price"] - cand_b["price"])
    if price_delta < min_price_delta:
        return False, "negligible_price_delta", [], []

    adv_a, adv_b = evaluate_candidate_advantages(cand_a, cand_b, tradeoff_family, pref_attr)

    # 3. Обе стороны обязаны иметь хотя бы одно преимущество (нет строгого доминирования)
    if not adv_a or not adv_b:
        return False, "strict_dominance_detected", adv_a, adv_b

    # 4. Проверка релевантности компромисса заданному tradeoff_family
    relevant = False
    if tradeoff_family == "direct_vs_price":
        relevant = ("direct_flight" in adv_a and "lower_price" in adv_b) or \
                   ("direct_flight" in adv_b and "lower_price" in adv_a)
    elif tradeoff_family == "time_vs_price":
        target = f"{pref_attr}_departure"
        relevant = (target in adv_a and "lower_price" in adv_b) or \
                   (target in adv_b and "lower_price" in adv_a)
    elif tradeoff_family == "stars_vs_price":
        has_stars_a = any("higher_hotel_stars" in x for x in adv_a)
        has_stars_b = any("higher_hotel_stars" in x for x in adv_b)
        relevant = (has_stars_a and "lower_price" in adv_b) or \
                   (has_stars_b and "lower_price" in adv_a)
    elif tradeoff_family == "breakfast_vs_price":
        relevant = ("breakfast_included" in adv_a and "lower_price" in adv_b) or \
                   ("breakfast_included" in adv_b and "lower_price" in adv_a)
    elif tradeoff_family == "comfort_vs_price":
        relevant = (("closer_to_sea" in adv_a and "lower_price" in adv_b) or
                    ("closer_to_sea" in adv_b and "lower_price" in adv_a))

    if not relevant:
        return False, f"tradeoff_family_mismatch_{tradeoff_family}", adv_a, adv_b

    return True, "valid_tradeoff", adv_a, adv_b


async def collect_admissible_candidates(
    routes_map: Dict[str, List[Route]],
    scenario: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Отбирает все допустимые маршруты для заданного сценария через жесткую проверку правил.
    """
    dest_id = scenario["destination_id"]
    routes = routes_map.get(dest_id, [])
    request_slots = {
        "days": scenario["nights"],
        "adults_count": scenario["adults_count"],
        "kids_count": scenario["kids_count"],
        "has_pets": scenario["has_pets"],
        "has_kids": scenario["has_kids"],
        "budget": scenario["budget"],
        "origin_airport": scenario.get("origin_airport"),
        "prefer_direct_flight": False,  # Мягкое предпочтение НЕ является жестким фильтром
        "flight_time_pref": None,       # Время вылета мягкое
        "rooms_count": 1,
    }

    admissible = []
    for r in routes:
        back = select_return(r, direct_only=False)
        logistics, constraints, price = assess(r, request_slots, back)
        if logistics or constraints or price is None or price <= 0:
            continue

        cand = {
            "route": r,
            "back": back,
            "price": price,
            "destination_id": dest_id,
        }
        if check_tradeoff_admissibility(cand, scenario):
            admissible.append(cand)

    return admissible


def build_pair_record(
    pair_id: int,
    cand_1: Dict[str, Any],
    cand_2: Dict[str, Any],
    scenario: Dict[str, Any],
    query_text: str,
    adv_1: List[str],
    adv_2: List[str],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Создает сериализуемую запись пары со случайным ослеплением порядка A/B и снимком признаков V2.
    """
    # Случайно перемешиваем сторону A и B (50% / 50%) для исключения позиционного сдвига
    swap = rng.choice([False, True])
    c_a, c_b = (cand_2, cand_1) if swap else (cand_1, cand_2)
    adv_a, adv_b = (adv_2, adv_1) if swap else (adv_1, adv_2)

    min_p = min(c_a["price"], c_b["price"])
    cheaper_side = "A" if c_a["price"] < c_b["price"] else "B"
    price_delta = round(abs(c_a["price"] - c_b["price"]), 2)
    rel_surcharge = round(price_delta / max(1.0, min_p), 4)

    req_profile = {
        "archetype": scenario["archetype"],
        "interests": [scenario["archetype"]],
        "nights": scenario["nights"],
        "days": scenario["nights"],
        "adults_count": scenario["adults_count"],
        "kids_count": scenario["kids_count"],
        "has_pets": scenario["has_pets"],
        "has_kids": scenario["has_kids"],
        "budget": scenario["budget"],
        "soft_preference": scenario["soft_preference"],
        "prefer_direct_flight": scenario["soft_preference"]["attribute"] == "direct",
        "flight_time_pref": scenario["soft_preference"]["attribute"] if scenario["soft_preference"]["attribute"] in ("morning", "day", "evening") else None,
    }

    # Вычисляем признаки схемы V2 для обоих туров
    feat_a = LTRService.extract_candidate_features_v2(
        hotel=c_a["route"].hotel,
        destination_id=scenario["destination_id"],
        profile=req_profile,
        nights=scenario["nights"],
        logistics={"flight": c_a["route"].flight, "timeline": {}},
        reference_price=min_p,
    )
    feat_b = LTRService.extract_candidate_features_v2(
        hotel=c_b["route"].hotel,
        destination_id=scenario["destination_id"],
        profile=req_profile,
        nights=scenario["nights"],
        logistics={"flight": c_b["route"].flight, "timeline": {}},
        reference_price=min_p,
    )

    r_a = c_a["route"]
    r_b = c_b["route"]

    return {
        "pair_id": pair_id,
        "scenario_family_id": scenario["family_id"],
        "archetype": scenario["archetype"],
        "destination_id": scenario["destination_id"],
        "tradeoff_family": scenario["tradeoff_family"],
        "query_text": query_text,
        "soft_preference": scenario["soft_preference"],
        "request_slots": {
            "nights": scenario["nights"],
            "adults_count": scenario["adults_count"],
            "kids_count": scenario["kids_count"],
            "has_pets": scenario["has_pets"],
            "has_kids": scenario["has_kids"],
            "budget": scenario["budget"],
            "origin_airport": scenario.get("origin_airport"),
        },
        "tour_a": {
            "route_key": list(r_a.key),
            "hotel_id": r_a.hotel["id"],
            "hotel_name": r_a.hotel["name"],
            "hotel_stars": LTRService._parse_stars(r_a.hotel["stars"]),
            "flight_no": r_a.flight["flight_no"],
            "flight_type": r_a.flight.get("flight_type"),
            "departure_time": r_a.flight.get("departure_time"),
            "price": round(c_a["price"], 2),
            "features_v2": feat_a,
        },
        "tour_b": {
            "route_key": list(r_b.key),
            "hotel_id": r_b.hotel["id"],
            "hotel_name": r_b.hotel["name"],
            "hotel_stars": LTRService._parse_stars(r_b.hotel["stars"]),
            "flight_no": r_b.flight["flight_no"],
            "flight_type": r_b.flight.get("flight_type"),
            "departure_time": r_b.flight.get("departure_time"),
            "price": round(c_b["price"], 2),
            "features_v2": feat_b,
        },
        "tradeoff_metadata": {
            "cheaper_side": cheaper_side,
            "min_price": min_p,
            "price_delta": price_delta,
            "relative_surcharge_pct": rel_surcharge,
            "advantages_tour_a": adv_a,
            "advantages_tour_b": adv_b,
            "swapped_for_blinding": swap,
        },
    }


async def generate_dataset(
    catalog_path: Path,
    output_path: Path,
    target_pairs: int = 250,
    seed: int = 42,
    use_neo4j: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Основная процедура сборки датасета пар с компромиссами.
    """
    rng = random.Random(seed)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_hash = file_sha256(catalog_path)

    # 1. Загрузка маршрутов
    if use_neo4j:
        from neo4j import AsyncGraphDatabase
        async with AsyncGraphDatabase.driver(
            settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        ) as driver:
            routes_map = await read_graph_routes(driver, catalog)
        logger.info("Маршруты успешно прочитаны из живого графа Neo4j")
    else:
        routes_map = build_routes(catalog)
        logger.info("Маршруты построены из снимка каталога destinations.json")

    total_routes = sum(len(routes) for routes in routes_map.values())
    logger.info("Всего доступно маршрутов по всем направлениям: %d", total_routes)

    all_pairs: List[Dict[str, Any]] = []
    seen_route_pairs: Set[Tuple[Tuple[Any, ...], Tuple[Any, ...]]] = set()
    pair_counter = 1

    archetype_stats = Counter()
    tradeoff_stats = Counter()
    destination_stats = Counter()

    # 2. Итерация по сценариям
    for scenario in SCENARIO_TEMPLATES:
        dest_id = scenario["destination_id"]
        tradeoff_fam = scenario["tradeoff_family"]
        pref_attr = scenario["soft_preference"]["attribute"]
        candidates = await collect_admissible_candidates(routes_map, scenario)

        if len(candidates) < 2:
            logger.warning("Сценарий %s: найдено менее 2 допустимых туров (%d)", scenario["family_id"], len(candidates))
            continue

        # Ограничиваем пул кандидатов до 40 туров с равномерным шагом по цене для быстрого и репрезентативного отбора
        if len(candidates) > 40:
            candidates.sort(key=lambda c: c["price"])
            step = len(candidates) / 40.0
            sampled_candidates = [candidates[int(k * step)] for k in range(40)]
        else:
            sampled_candidates = list(candidates)

        max_per_scenario = max(15, target_pairs // len(SCENARIO_TEMPLATES) + 5)
        needed_pairs = max_per_scenario * 2

        scenario_pairs = []
        for i in range(len(sampled_candidates)):
            for j in range(i + 1, len(sampled_candidates)):
                cand_1 = sampled_candidates[i]
                cand_2 = sampled_candidates[j]

                r_key_1 = cand_1["route"].key
                r_key_2 = cand_2["route"].key
                canonical_pair_key = tuple(sorted([r_key_1, r_key_2]))
                if canonical_pair_key in seen_route_pairs:
                    continue

                valid, reason, adv_1, adv_2 = is_valid_tradeoff_pair(
                    cand_1, cand_2, tradeoff_fam, pref_attr
                )
                if not valid:
                    continue

                seen_route_pairs.add(canonical_pair_key)
                query_text = rng.choice(scenario["queries"])

                record = build_pair_record(
                    pair_id=pair_counter,
                    cand_1=cand_1,
                    cand_2=cand_2,
                    scenario=scenario,
                    query_text=query_text,
                    adv_1=adv_1,
                    adv_2=adv_2,
                    rng=rng,
                )
                scenario_pairs.append(record)
                pair_counter += 1
                if len(scenario_pairs) >= needed_pairs:
                    break
            if len(scenario_pairs) >= needed_pairs:
                break

        # Ограничиваем количество пар на один сценарий для балансировки
        rng.shuffle(scenario_pairs)
        max_per_scenario = max(15, target_pairs // len(SCENARIO_TEMPLATES) + 5)
        selected_for_scenario = scenario_pairs[:max_per_scenario]

        for p in selected_for_scenario:
            all_pairs.append(p)
            archetype_stats[p["archetype"]] += 1
            tradeoff_stats[p["tradeoff_family"]] += 1
            destination_stats[p["destination_id"]] += 1

    # 3. Финальная балансировка и усечение до target_pairs
    rng.shuffle(all_pairs)
    if len(all_pairs) > target_pairs:
        all_pairs = all_pairs[:target_pairs]

    # Перенумеровываем pair_id
    for idx, p in enumerate(all_pairs, start=1):
        p["pair_id"] = idx

    output_payload = {
        "dataset_name": "CSAI-2026 Soft Preference Trade-Offs (Candidate Pool)",
        "feature_schema": FEATURE_SCHEMA_V2,
        "feature_columns": FEATURE_COLUMNS_V2,
        "catalog_sha256": catalog_hash,
        "target_pairs": target_pairs,
        "total_pairs_generated": len(all_pairs),
        "random_seed": seed,
        "distributions": {
            "by_archetype": dict(archetype_stats),
            "by_tradeoff_family": dict(tradeoff_stats),
            "by_destination": dict(destination_stats),
        },
        "pairs": all_pairs,
    }

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Датасет успешно сохранен в: %s (размер: %.1f КБ)", output_path, output_path.stat().st_size / 1024)

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор пар туров с мягкими компромиссами (Этап 3 CSAI 2026)")
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG_PATH), help="Путь к destinations.json")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Путь для сохранения результата")
    parser.add_argument("--target-pairs", type=int, default=250, help="Целевое количество пар")
    parser.add_argument("--seed", type=int, default=42, help="Случайный seed")
    parser.add_argument("--neo4j", action="store_true", help="Читать маршруты из живого Neo4j вместо каталога")
    parser.add_argument("--dry-run", action="store_true", help="Не записывать файл на диск, только вывести статистику")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод логов")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = asyncio.run(
        generate_dataset(
            catalog_path=Path(args.catalog),
            output_path=Path(args.output),
            target_pairs=args.target_pairs,
            seed=args.seed,
            use_neo4j=args.neo4j,
            dry_run=args.dry_run,
        )
    )

    print("=" * 70)
    print("CSAI 2026: Генерация пар с мягкими компромиссами завершена")
    print("=" * 70)
    print(f"Всего сформировано пар: {result['total_pairs_generated']}")
    print(f"Схема признаков       : {result['feature_schema']} ({len(result['feature_columns'])} признаков)")
    print(f"Хеш каталога (sha256) : {result['catalog_sha256']}")
    print("\nРаспределение по архетипам:")
    for arch, cnt in result["distributions"]["by_archetype"].items():
        print(f"  - {arch:<12}: {cnt} пар")
    print("\nРаспределение по видам компромиссов:")
    for fam, cnt in result["distributions"]["by_tradeoff_family"].items():
        print(f"  - {fam:<25}: {cnt} пар")
    print("\nРаспределение по направлениям:")
    for dest, cnt in result["distributions"]["by_destination"].items():
        print(f"  - {dest:<12}: {cnt} пар")


if __name__ == "__main__":
    main()
