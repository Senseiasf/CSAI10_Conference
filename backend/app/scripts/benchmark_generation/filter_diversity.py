#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль фильтрации, дедупликации и аудита запросов для CSAI 2026.
Перенос алгоритмов из 03_clean_and_filter.py, 04_merge_synthetic_pools.py, 06_finalise_datasets.py.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Регулярные выражения для глубокой очистки (из 03_clean_and_filter.py)
META_PATTERNS = re.compile(
    r"(###|\bпример\w*\s*\d|\bобразец\s*\d|\bпримечание|\bконец\s+примеров|\bот\s+лица\s|"
    r"\bсгенерирован|\bтекст\s+запроса|\bвариант\s+\d|\bзапрос\s+\d|\bвот\s+ваш\s+запрос|"
    r"\bниже\s+представлен|\bданный\s+запрос|\bв\s+качестве\s+примера|\bобратите\s+внимание)",
    flags=re.IGNORECASE,
)
PLACEHOLDER_PATTERN = re.compile(r"[\[\]<>{}]|example\.com", flags=re.IGNORECASE)
LEAD_JUNK = re.compile(r"^[\s\"\'\`\*\#\-\–\—\•\d\.\)\:]+")
MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")

PREFIX_STRIP = re.compile(
    r"^(?:[^\wА-Яа-яЁё]*"
    r"(?:уважаем\w+|здравствуйте|здравствуй|приветствую(?:\s+вас)?|привет(?:ик)?|"
    r"добрый\s+(?:день|вечер|утро)|доброго\s+времени(?:\s+суток)?)"
    r"[!,\.\:\—\-\s]*)+",
    flags=re.IGNORECASE,
)

TRUNCATED_END = re.compile(
    r"(\b(и|в|на|с|по|для|от|к|о|из|за|при|до|без|через|или|что|как|но|а|причём|также)\b|[,:;—\-])\s*$",
    flags=re.IGNORECASE,
)

BAD_STARTS = (
    "я не могу", "i cannot", "к сожалению", "однако", "извините", "простите",
    "as an ai", "вот ", "конечно", "ниже представлен", "сгенерированный",
    "здесь представлен", "образец", "пример:",
)

BAD_WORDS = ["example", "sample", "test", "placeholder", "вставь", "промпт", "prompt"]

GRAMMAR_FIXES = [
    (re.compile(r"\bна\s+2\s+дней\b", re.IGNORECASE), "на 2 дня"),
    (re.compile(r"\bна\s+3\s+дней\b", re.IGNORECASE), "на 3 дня"),
    (re.compile(r"\bна\s+4\s+дней\b", re.IGNORECASE), "на 4 дня"),
]


def fix_grammar_artifacts(text: str) -> str:
    """Исправляет распространенные шаблонные огрехи генерации (напр. 'на 4 дней' -> 'на 4 дня')."""
    res = text
    for pattern, repl in GRAMMAR_FIXES:
        res = pattern.sub(repl, res)
    return res


def clean_and_validate_query(raw_text: str) -> Optional[str]:
    """
    Построчная глубокая очистка и валидация сырого текста запроса (перенос из 03_clean_and_filter.py).
    Возвращает очищенную строку или None, если текст забракован.
    """
    if not isinstance(raw_text, str):
        return None
    raw_stripped = raw_text.strip()
    if not raw_stripped:
        return None

    # 1. Отбраковка метакомментариев LLM и плейсхолдеров
    if META_PATTERNS.search(raw_stripped) or PLACEHOLDER_PATTERN.search(raw_stripped):
        return None

    # 2. Снятие Markdown-выделений
    line = MD_BOLD.sub(r"\1", raw_stripped).strip()
    if not line or line.startswith("*"):
        return None

    # 3. Проверка недопустимых стоп-слов
    clean_lower = line.lower()
    if any(bw in clean_lower for bw in BAD_WORDS):
        return None
    if clean_lower.startswith(BAD_STARTS):
        return None

    # 4. Очистка лидирующего мусора, кавычек и приветствий
    line = LEAD_JUNK.sub("", line).strip()
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()
    elif line.endswith('"') or line.endswith("'"):
        line = line[:-1].strip()

    line = PREFIX_STRIP.sub("", line).strip()
    line = LEAD_JUNK.sub("", line).strip()
    line = re.sub(r"\s+", " ", line).strip()

    # 5. Исправление грамматических артефактов
    line = fix_grammar_artifacts(line)

    if not line:
        return None

    # 6. Проверка баланса кириллицы и длины
    ru_chars = len(re.findall(r"[А-Яа-яЁё]", line))
    en_chars = len(re.findall(r"[A-Za-z]", line))
    if ru_chars < 20 or en_chars > ru_chars:
        return None

    # Минимальная длина, отсутствие оборванного окончания и наличие терминальной пунктуации
    if len(line) < 35 or TRUNCATED_END.search(line) or not re.search(r"[\.!\?…]\s*$", line):
        return None

    words = re.findall(r"[а-яёa-z0-9]+", line.lower())
    if len(words) < 6:
        return None

    # Заглавная буква в начале
    if line[0].islower() and line[0].isalpha():
        line = line[0].upper() + line[1:]

    return line


def deduplicate_archetype_pool(
    items: List[Dict[str, Any]],
    archetype: str,
    max_first_word: int = 4,
    max_bigram_prefix: int = 2,
    max_trigram: int = 4,
    sim_thresh: float = 0.62,
    global_seen_norms: Optional[Set[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Трёхступенчатая дедупликация пула запросов одного архетипа (перенос из 04_merge_synthetic_pools.py):
    1. Stage 1: Alphanumeric normalization and exact duplicate pruning.
    2. Stage 2: N-gram frequency control (prefix unigram, prefix bigram, internal trigrams).
    3. Stage 3: Batched Char-WB TF-IDF cosine deduplication with <num> token masking.
    """
    if global_seen_norms is None:
        global_seen_norms = set()

    rng = np.random.default_rng(seed)

    # Stage 1: Alphanumeric deduplication
    stage1: List[Dict[str, Any]] = []
    for item in items:
        text = item.get("query", "")
        words = re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))
        if len(words) < 6:
            continue
        alpha_norm = " ".join(words)
        if alpha_norm in global_seen_norms:
            continue
        global_seen_norms.add(alpha_norm)
        stage1.append(item)

    # Перемешивание для объективного частотного отбора
    shuffled_indices = rng.permutation(len(stage1))
    stage1 = [stage1[i] for i in shuffled_indices]

    # Stage 2: N-gram frequency filtering
    stage2: List[Dict[str, Any]] = []
    fw_counts: Counter = Counter()
    bp_counts: Counter = Counter()
    tg_counts: Counter = Counter()

    for item in stage1:
        text = item.get("query", "")
        words = re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))
        if len(words) < 6:
            continue
        fw = words[0]
        bp = f"{words[0]} {words[1]}"

        if fw_counts[fw] >= max_first_word or bp_counts[bp] >= max_bigram_prefix:
            continue

        tgs = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        if any(tg_counts[tg] >= max_trigram for tg in tgs):
            continue

        fw_counts[fw] += 1
        bp_counts[bp] += 1
        for tg in tgs:
            tg_counts[tg] += 1
        stage2.append(item)

    # Stage 3: Batched Char-WB TF-IDF Cosine Deduplication with <num> masking
    if len(stage2) >= 2:
        # Маскируем все числа на <num>, чтобы запросы с одинаковой структурой, но разными цифрами, распознавались как дубликаты
        masked_texts = [
            re.sub(r"\d+", "<num>", it["query"].lower().replace("ё", "е"))
            for it in stage2
        ]

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=25000,
            dtype=np.float32,
        )
        X = vec.fit_transform(masked_texts)
        n = len(stage2)
        removed: Set[int] = set()
        keep_indices: List[int] = []

        batch_size = 500
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sim_chunk = (X[start:end] @ X.T).toarray()
            for idx_in_chunk in range(end - start):
                i = start + idx_in_chunk
                if i in removed:
                    continue
                keep_indices.append(i)
                dups = np.where(sim_chunk[idx_in_chunk, i + 1:] >= sim_thresh)[0] + (i + 1)
                removed.update(dups.tolist())

        final_items = [stage2[i] for i in keep_indices]
    else:
        final_items = list(stage2)

    print(
        f"[DEDUP: {archetype:<8}] Вход: {len(items):>3} | "
        f"Stage 1 (Alpha): {len(stage1):>3} | "
        f"Stage 2 (N-gram): {len(stage2):>3} | "
        f"Stage 3 (TF-IDF): {len(final_items):>3}"
    )
    return final_items


def audit_catalog_feasibility(
    items: List[Dict[str, Any]],
    catalog: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Record catalogue feasibility without changing the request or its labels."""
    from app.scripts.benchmark_routes import assess, build_routes, select_return

    routes = build_routes(catalog)
    audited = []
    for item in items:
        feasible = False
        for dest in item["expected_destinations"]:
            for route in routes[dest]:
                back = select_return(route, item.get("prefer_direct_flight"))
                logistics, constraints, _ = assess(route, item, back)
                if not logistics and not constraints:
                    feasible = True
                    break
            if feasible:
                break
        audited.append({**item, "catalog_feasible": feasible})
    return audited
