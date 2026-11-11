#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль генерации сырого пула запросов (250-300 сценариев) по 5 архетипам туристов
с естественными формулировками, карточками ground_truth, неявным вайбом (is_implicit_vibe)
и чекпоинтами прогресса (по аналогии с 01_generate_base_phishing.py).
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Обеспечиваем корректный импорт app
BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from app.config import settings
except ImportError:
    from backend.app.config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "raw_queries_pool_cache.json"
SEED = 42

LEN_SHORT = ("короткий", "1-2 коротких предложения (до 140 символов), лаконично")
LEN_MED = ("средний", "2-3 предложения (200-300 символов), средний объем")
LEN_LONG = ("развернутый", "3-5 предложений (400-500 символов), подробный рассказ о пожеланиях")

LENGTH_BUCKETS = [LEN_SHORT, LEN_MED, LEN_MED, LEN_LONG]

FLIGHT_PREFS: List[Tuple[Optional[str], str]] = [
    ("morning", "вылет утром пораньше"),
    ("day", "вылет днем в обеденное время"),
    ("evening", "вылет вечером после работы"),
    (None, "время вылета не имеет значения"),
]

# Спецификация 5 архетипов и их семантических привязок к 9 направлениям
ARCHETYPE_SCENARIOS = {
    "budget": [
        {
            "dests": ["egypt", "thailand"],
            "vibe": "теплое море, красивые коралловые рифы, снорклинг и купание",
            "explicit_name": "Египет или Таиланд",
            "b_range": (85000, 135000),
            "days_range": (5, 8),
        },
        {
            "dests": ["sochi"],
            "vibe": "отдых на море и прогулки в горах в России без загранпаспорта",
            "explicit_name": "Сочи",
            "b_range": (65000, 110000),
            "days_range": (4, 7),
        },
        {
            "dests": ["georgia"],
            "vibe": "старый колоритный город, хинкали, хачапури, горы и недорогие кафе",
            "explicit_name": "Грузия, Тбилиси",
            "b_range": (75000, 125000),
            "days_range": (5, 7),
        },
        {
            "dests": ["turkey"],
            "vibe": "недорогой отель рядом с морем, прогулки по набережной и восточный колорит",
            "explicit_name": "Турция",
            "b_range": (85000, 140000),
            "days_range": (5, 7),
        },
        {
            "dests": ["spb"],
            "vibe": "музеи, старинная архитектура, каналы и бюджетный хостел или мини-отель у метро",
            "explicit_name": "Санкт-Петербург",
            "b_range": (45000, 85000),
            "days_range": (4, 6),
        },
        {
            "dests": ["altai"],
            "vibe": "треккинг по горным тропам, хвойный лес, горная река и эконом турбаза",
            "explicit_name": "Алтай",
            "b_range": (70000, 120000),
            "days_range": (5, 8),
        },
    ],
    "pet": [
        {
            "dests": ["altai", "sochi", "georgia"],
            "vibe": "горы, хвойный лес, треккинг с собакой вдоль рек и пет-френдли коттедж",
            "explicit_name": "Алтай или Сочи",
            "b_range": (115000, 195000),
            "days_range": (5, 8),
            "pets_desc": "с джек-расселом",
        },
        {
            "dests": ["sochi"],
            "vibe": "прогулки по морской набережной и паркам с маленькой собачкой, отель с приемом животных",
            "explicit_name": "Сочи",
            "b_range": (110000, 180000),
            "days_range": (5, 7),
            "pets_desc": "со шпицем",
        },
        {
            "dests": ["spb"],
            "vibe": "парки, скверы, исторический центр и прогулки с питомцем по набережным",
            "explicit_name": "Санкт-Петербург",
            "b_range": (85000, 155000),
            "days_range": (4, 7),
            "pets_desc": "с корги",
        },
        {
            "dests": ["georgia"],
            "vibe": "просторные луга, горы Казбека, гостеприимные пет-френдли гестхаусы",
            "explicit_name": "Грузия",
            "b_range": (120000, 205000),
            "days_range": (6, 8),
            "pets_desc": "с собакой среднего размера",
        },
        {
            "dests": ["turkey"],
            "vibe": "зеленый отель с собственной территорией у моря, где разрешено проживание с животными",
            "explicit_name": "Турция",
            "b_range": (145000, 235000),
            "days_range": (6, 8),
            "pets_desc": "с лабрадором",
        },
        {
            "dests": ["uae"],
            "vibe": "современный курорт, кондиционированные пет-френдли апартаменты у залива",
            "explicit_name": "Дубай",
            "b_range": (195000, 310000),
            "days_range": (5, 8),
            "pets_desc": "с карликовым пуделем",
        },
    ],
    "family": [
        {
            "dests": ["turkey", "uae", "egypt"],
            "vibe": "аквапарк, песчаный пологий пляж, детская комната, анимация и все включено",
            "explicit_name": "Турция или Египет",
            "b_range": (210000, 350000),
            "days_range": (6, 9),
        },
        {
            "dests": ["sochi"],
            "vibe": "семейный курорт с подогреваемым бассейном, детской игровой и парком аттракционов",
            "explicit_name": "Сочи",
            "b_range": (140000, 240000),
            "days_range": (5, 8),
        },
        {
            "dests": ["egypt"],
            "vibe": "закрытая от ветра песчаная лагуна для безопасного купания малышей без кораллов у берега",
            "explicit_name": "Египет",
            "b_range": (185000, 310000),
            "days_range": (7, 10),
        },
        {
            "dests": ["uae"],
            "vibe": "тематические парки развлечений, безопасный чистый залив и семейные просторные номера",
            "explicit_name": "Дубай, ОАЭ",
            "b_range": (280000, 440000),
            "days_range": (6, 8),
        },
        {
            "dests": ["spb"],
            "vibe": "познавательная поездка для школьника, музей гранд макет, мосты и крейсер Аврора",
            "explicit_name": "Санкт-Петербург",
            "b_range": (110000, 185000),
            "days_range": (5, 7),
        },
        {
            "dests": ["thailand"],
            "vibe": "спокойная бухта с мягким песком, свежие фрукты, слоновья ферма и чистый океан",
            "explicit_name": "Таиланд, Пхукет",
            "b_range": (230000, 375000),
            "days_range": (7, 10),
        },
    ],
    "luxury": [
        {
            "dests": ["maldives"],
            "vibe": "приватная вилла над бирюзовой лагуной, трансфер на гидросамолете и спа-процедуры",
            "explicit_name": "Мальдивы",
            "b_range": (550000, 920000),
            "days_range": (6, 10),
        },
        {
            "dests": ["uae"],
            "vibe": "пятизвездочный отель люкс с видом на небоскребы, индивидуальный трансфер и шопинг",
            "explicit_name": "Дубай",
            "b_range": (380000, 690000),
            "days_range": (6, 9),
        },
        {
            "dests": ["turkey"],
            "vibe": "премиум резорт в уединенной бухте, частный пляж, изысканная гастрономия и консьерж",
            "explicit_name": "Бодрум или Анталья",
            "b_range": (350000, 620000),
            "days_range": (6, 9),
        },
        {
            "dests": ["sochi"],
            "vibe": "эксклюзивный пятизвездочный спа-отель в горах Красной Поляны с открытым бассейном",
            "explicit_name": "Красная Поляна, Сочи",
            "b_range": (260000, 490000),
            "days_range": (5, 8),
        },
        {
            "dests": ["altai"],
            "vibe": "роскошное уединенное шале на берегу реки Катунь с вертолетной экскурсией",
            "explicit_name": "Алтай",
            "b_range": (330000, 580000),
            "days_range": (5, 8),
        },
        {
            "dests": ["spb"],
            "vibe": "исторический пятизвездочный гранд-отель с видом на собор и высокой кухней",
            "explicit_name": "Санкт-Петербург",
            "b_range": (220000, 410000),
            "days_range": (4, 7),
        },
    ],
    "solo": [
        {
            "dests": ["spb"],
            "vibe": "архитектура, парадные, набережные каналов, музеи, кофейни и белые ночи для одного",
            "explicit_name": "Санкт-Петербург",
            "b_range": (70000, 130000),
            "days_range": (4, 7),
        },
        {
            "dests": ["georgia"],
            "vibe": "винные погреба, узкие улочки старого города, горы и колорит в одиночку",
            "explicit_name": "Тбилиси или Батуми",
            "b_range": (85000, 155000),
            "days_range": (5, 8),
        },
        {
            "dests": ["altai"],
            "vibe": "соло перезагрузка, медитация на природе, горный воздух и тишина у реки",
            "explicit_name": "Алтай",
            "b_range": (90000, 165000),
            "days_range": (5, 8),
        },
        {
            "dests": ["thailand"],
            "vibe": "тропические храмы, фруктовые рынки, массаж и закаты на пляже соло",
            "explicit_name": "Таиланд",
            "b_range": (135000, 230000),
            "days_range": (7, 10),
        },
        {
            "dests": ["turkey"],
            "vibe": "исторические улочки Стамбула, паромы через пролив, кофейни и смотровые площадки",
            "explicit_name": "Стамбул",
            "b_range": (80000, 145000),
            "days_range": (5, 7),
        },
        {
            "dests": ["sochi"],
            "vibe": "активный треккинг по горным тропам одному, подъемники и свежий морской бриз",
            "explicit_name": "Красная Поляна и Сочи",
            "b_range": (75000, 130000),
            "days_range": (5, 7),
        },
    ],
}


def get_style_instruction(archetype: str) -> str:
    """Стилевая инструкция для модели (по аналогии с 01_generate_base_phishing.py)."""
    if archetype == "budget":
        return (
            "Пиши как экономный практичный путешественник: четко указывает лимит бюджета, "
            "ищет максимальную отдачу за скромные деньги. Простой разговорный тон, без лишнего пафоса."
        )
    if archetype == "pet":
        return (
            "Пиши как заботливый владелец домашнего животного: упоминает питомца (породу/размер), "
            "интересуется пет-френдли отелем и возможностью комфортных прогулок на свежем воздухе."
        )
    if archetype == "family":
        return (
            "Пиши как заботливый родитель, планирующий семейный отпуск с детьми: важны безопасность, "
            "хорошее питание, пологий вход в воду/бассейн и спокойный удобный перелет."
        )
    if archetype == "luxury":
        return (
            "Пиши как состоятельный клиент: ценит пятизвездочный сервис, красивый вид, "
            "приватность, спа, индивидуальный подход и комфорт без компромиссов."
        )
    return (
        "Пиши как самостоятельный путешественник (соло): едет один развеяться, важны атмосфера, "
        "красивые места, прогулки по городу или единение с природой, гастрономия и свобода передвижения."
    )


def format_days_phrase(days: int) -> str:
    """Формирует грамматически корректную фразу о длительности."""
    if days in (2, 3, 4):
        return f"на {days} дня"
    return f"на {days} дней"


def generate_single_llm_query(
    client: Any,
    archetype: str,
    scenario: Dict[str, Any],
    is_implicit: bool,
    budget: int,
    days: int,
    f_code: Optional[str],
    f_desc: str,
    prefer_direct: bool,
    length_bucket: Tuple[str, str],
) -> str:
    """Генерирует живой запрос пользователя через OpenAI-совместимый API (DeepSeek Flash)."""
    style = get_style_instruction(archetype)
    days_str = format_days_phrase(days)
    budget_str = f"{budget:,} руб".replace(",", " ")

    if is_implicit:
        location_instruction = (
            f"КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО прямо называть город или страну ({scenario.get('explicit_name')})! "
            f"Опиши только желаемую природу, атмосферу, климат или активности: {scenario['vibe']}."
        )
    else:
        location_instruction = (
            f"Прямо назови в тексте желаемое направление: {scenario['explicit_name']}. "
            f"Опиши желаемое: {scenario['vibe']}."
        )

    flight_instruction = f"Пожелание по рейсу: {f_desc}." if f_code else ""
    direct_instruction = "Нужен только прямой перелет без пересадок." if prefer_direct else ""
    pet_instruction = f"Едем с питомцем ({scenario.get('pets_desc', 'с собакой')})." if archetype == "pet" else ""
    family_instruction = "Едем всей семьей с детьми." if archetype == "family" else ""
    solo_instruction = "Путешествую один (соло-поездка)." if archetype == "solo" else ""

    starter_instruction = (
        "ВАЖНО ДЛЯ РАЗНООБРАЗИЯ: используй оригинальный зачин предложения, например: "
        "'Планируем...', 'Подскажите вариант...', 'Нам нужен...', 'Мечтаем о...', 'Собираемся...', "
        "'Интересует...', 'Подыскиваем...', 'Летим...', 'Организуйте поездку...', 'Рассматриваем...'. "
        "НЕ начинай каждый раз с 'Хочу' или 'Ищу'!"
    )

    prompt = (
        f"Ты турист, который пишет в поисковый чат турагента Ostrovok. Напиши естественный текст запроса на русском языке.\n\n"
        f"ПАРАМЕТРЫ ЗАПРОСА:\n"
        f"- Стиль: {style}\n"
        f"- Зачин: {starter_instruction}\n"
        f"- Требование к локации: {location_instruction}\n"
        f"- Бюджет: около {budget_str} (упомяни сумму естественно).\n"
        f"- Срок: {days_str}.\n"
        f"{pet_instruction}\n{family_instruction}\n{solo_instruction}\n"
        f"{flight_instruction}\n{direct_instruction}\n"
        f"- Формат длины: {length_bucket[1]}.\n\n"
        f"ТРЕБОВАНИЯ К ВЫДАЧЕ:\n"
        f"- Выдай ТОЛЬКО сам текст запроса туриста.\n"
        f"- Никаких кавычек, никаких приветствий 'Здравствуйте/Привет', никаких вступительных фраз ('Вот ваш запрос:', 'Пример:').\n"
        f"- Соблюдай русскую грамматику (не пиши 'на 4 дней', пиши 'на 4 дня')."
    )

    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Ты обычный пользователь, пишущий естественный запрос в туристический сервис."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.82,
        max_tokens=1500,
    )
    return (resp.choices[0].message.content or "").strip()


def build_fallback_query(
    archetype: str,
    scenario: Dict[str, Any],
    is_implicit: bool,
    budget: int,
    days: int,
    f_code: Optional[str],
    f_desc: str,
    prefer_direct: bool,
    rng: random.Random,
) -> str:
    """Высококачественный детерминированный фоллбэк генерации живых запросов при недоступности сети."""
    budget_fmt = f"{budget:,} руб".replace(",", " ")
    days_fmt = format_days_phrase(days)

    vibe_text = scenario["vibe"]
    target_name = scenario.get("explicit_name", "")

    flight_part = ""
    if f_code == "morning":
        flight_part = rng.choice(["вылет утром пораньше", "желательно утренний рейс", "чтобы вылететь с утра"])
    elif f_code == "day":
        flight_part = rng.choice(["вылет днем в обед", "желательно дневной рейс", "удобный рейс в середине дня"])
    elif f_code == "evening":
        flight_part = rng.choice(["вылет вечером после работы", "лучше вечерний рейс", "поздний вылет"])

    direct_part = rng.choice(["только прямой рейс", "без пересадок", "прямой перелет"]) if prefer_direct else ""

    addons = [a for a in [flight_part, direct_part] if a]
    addons_str = (", " + ", ".join(addons)) if addons else ""

    starters_explicit = [
        f"Планируем поездку в {target_name}: {vibe_text}",
        f"Хотим поехать в {target_name}, интересует {vibe_text}",
        f"Подберите тур в {target_name}, чтобы было {vibe_text}",
        f"Собираемся отдохнуть в {target_name}: {vibe_text}",
        f"Мечтаем посетить {target_name}, важны {vibe_text}",
        f"Нам нужен тур в {target_name}, главное {vibe_text}",
        f"Рассматриваем отпуск в {target_name}: {vibe_text}",
        f"Подыскиваем вариант в {target_name}, где есть {vibe_text}",
        f"Интересует путешествие в {target_name}: {vibe_text}",
        f"Организуйте отдых в {target_name}, хочется {vibe_text}",
        f"Летим в {target_name}, пожелания: {vibe_text}",
        f"Помогите выбрать отель в {target_name}, нужно {vibe_text}",
    ]
    starters_implicit = [
        f"Ищем тур: {vibe_text}",
        f"Хочется отдохнуть: {vibe_text}",
        f"Подберите направление, где есть {vibe_text}",
        f"Мечтаю о поездке, где будет {vibe_text}",
        f"Планируем отпуск, где есть {vibe_text}",
        f"Собираемся в тихое место: {vibe_text}",
        f"Нам необходим отдых: {vibe_text}",
        f"Подыскиваем спокойный курорт: {vibe_text}",
        f"Интересует направление, где можно найти {vibe_text}",
        f"Помогите найти курорт: {vibe_text}",
        f"Рассматриваем поездку на природу: {vibe_text}",
        f"Хотим выбраться туда, где {vibe_text}",
    ]

    if not is_implicit:
        lead = rng.choice(starters_explicit)
    else:
        lead = rng.choice(starters_implicit)

    if archetype == "pet":
        pet_info = scenario.get("pets_desc", "с собакой")
        text = f"{lead}, едем {pet_info} {days_fmt}, бюджет до {budget_fmt}{addons_str}."
    elif archetype == "family":
        text = f"{lead}, семейный отдых с детьми {days_fmt}, бюджет в пределах {budget_fmt}{addons_str}."
    elif archetype == "luxury":
        text = f"{lead}, высокий уровень сервиса {days_fmt}, ориентируемся на бюджет {budget_fmt}{addons_str}."
    elif archetype == "budget":
        text = f"{lead}, экономно {days_fmt}, строгий лимит бюджета до {budget_fmt}{addons_str}."
    else:  # solo
        text = f"{lead}, соло поездка для одного {days_fmt}, рассчитываю на бюджет {budget_fmt}{addons_str}."

    return text


from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_raw_queries(target_count_per_arch: int = 110) -> List[Dict[str, Any]]:
    """
    Генерирует первичный пул из не менее 280-300 сценариев (по 55-60 на каждый из 5 архетипов).
    Использует параллельные потоки и чекпоинты прогресса (по аналогии с 01_generate_base_phishing.py).
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.is_file():
        try:
            cached_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if len(cached_data) >= target_count_per_arch * 5:
                print(f"[CACHE] Загружен готовый пул из {len(cached_data)} сырых запросов из {CACHE_FILE.name}")
                return cached_data
        except Exception as e:
            print(f"[CACHE WARN] Не удалось прочитать кэш: {e}")

    rng = random.Random(SEED)
    tasks = []
    idx = 1

    for archetype, scenarios in ARCHETYPE_SCENARIOS.items():
        for i in range(target_count_per_arch):
            sc = scenarios[i % len(scenarios)]
            is_implicit = (rng.random() < 0.68)
            b_min, b_max = sc["b_range"]
            d_min, d_max = sc["days_range"]

            days = rng.randint(d_min, d_max)
            budget = int(rng.randint(b_min, b_max) // 5000 * 5000)

            f_code, f_desc = rng.choice(FLIGHT_PREFS)
            prefer_direct = rng.choice([True, False])
            length_bucket = rng.choice(LENGTH_BUCKETS)

            has_pets = (archetype == "pet")
            has_kids = (archetype == "family")

            tasks.append({
                "idx": idx,
                "archetype": archetype,
                "scenario": sc,
                "is_implicit": is_implicit,
                "budget": budget,
                "days": days,
                "f_code": f_code,
                "f_desc": f_desc,
                "prefer_direct": prefer_direct,
                "length_bucket": length_bucket,
                "has_pets": has_pets,
                "has_kids": has_kids,
                "kids_count": rng.choice([1, 2]) if has_kids else 0,
                "adults_count": 1 if archetype == "solo" else (2 if archetype in ("family", "luxury") else rng.choice([1, 2])),
            })
            idx += 1

    # Подключаем клиент OpenAI, если ключ задан
    client = None
    if not settings.is_mock_llm and settings.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
            print("[INFO] Активен параллельный пул DeepSeek Flash (6 воркеров) для генерации естественных запросов")
        except Exception as e:
            print(f"[WARN] Ошибка инициализации live OpenAI клиента: {e}")

    def worker(task: Dict[str, Any]) -> Dict[str, Any]:
        local_rng = random.Random(SEED + task["idx"])
        q_text = ""
        if client is not None:
            for _attempt in range(2):
                try:
                    q_text = generate_single_llm_query(
                        client=client,
                        archetype=task["archetype"],
                        scenario=task["scenario"],
                        is_implicit=task["is_implicit"],
                        budget=task["budget"],
                        days=task["days"],
                        f_code=task["f_code"],
                        f_desc=task["f_desc"],
                        prefer_direct=task["prefer_direct"],
                        length_bucket=task["length_bucket"],
                    )
                    if q_text and len(q_text) >= 30:
                        break
                except Exception:
                    time.sleep(0.5)

        if not q_text or len(q_text) < 30:
            q_text = build_fallback_query(
                archetype=task["archetype"],
                scenario=task["scenario"],
                is_implicit=task["is_implicit"],
                budget=task["budget"],
                days=task["days"],
                f_code=task["f_code"],
                f_desc=task["f_desc"],
                prefer_direct=task["prefer_direct"],
                rng=local_rng,
            )

        return {
            "id": f"RAW_{task['idx']:03d}",
            "query": q_text,
            "archetype": task["archetype"],
            "expected_destinations": task["scenario"]["dests"],
            "is_implicit_vibe": task["is_implicit"],
            "budget": task["budget"],
            "days": task["days"],
            "has_pets": task["has_pets"],
            "has_kids": task["has_kids"],
            "kids_count": task["kids_count"],
            "adults_count": task["adults_count"],
            "flight_time_pref": task["f_code"],
            "prefer_direct_flight": task["prefer_direct"],
        }

    raw_dataset: List[Dict[str, Any]] = []
    print(f"[INFO] Запуск генерации {len(tasks)} сценариев...", flush=True)
    max_workers = 6 if client is not None else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, t): t["idx"] for t in tasks}
        completed = 0
        for f in as_completed(futures):
            try:
                res = f.result()
                raw_dataset.append(res)
            except Exception as e:
                print(f"[ERROR in worker]: {e}", flush=True)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"      Прогресс генерации: {completed}/{len(tasks)} ({(completed / len(tasks) * 100):.1f}%)", flush=True)

    # Сортируем обратно по id
    raw_dataset.sort(key=lambda x: x["id"])

    # Сохраняем в кэш
    CACHE_FILE.write_text(json.dumps(raw_dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Сгенерирован и сохранён пул из {len(raw_dataset)} запросов в {CACHE_FILE.name}")
    return raw_dataset


if __name__ == "__main__":
    pool = generate_raw_queries(target_count_per_arch=58)
    print(f"Всего сгенерировано: {len(pool)} сырых сценариев.")
