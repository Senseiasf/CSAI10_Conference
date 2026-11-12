#!/usr/bin/env python3
"""
Скрипт слепой разметки пар туров с мягкими компромиссами с помощью LLM-судьи (Этап 4).

Принципы разметки:
1. Оценщику предъявляется исходный пользовательский запрос, контекст профиля и две карточки туров
   (Tour A и Tour B) с идентичным стандартизированным набором характеристик.
2. Порядок A/B уже рандомизирован на этапе генерации пула.
3. Дополнительно на подвыборке проверяется swap consistency (проверка симметричности решения).
4. Судья обязан явно обосновать выбор (rationale) и определить победителя: A, B или TIE (ничья).
5. Результаты разметки сохраняются со всеми метаданными (модель, температура, промпт, сырой ответ).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BACKEND_DIR / "app" / "data" / "csai_tradeoff_candidates_v2.json"
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "app" / "data" / "csai_tradeoffs_labeled_v2.json"

JUDGE_SYSTEM_PROMPT = """Ты — объективный эксперт по оценке туристических предложений и поведенческому ранжированию туров.
Тебе предложен запрос пользователя (с его мягкими пожеланиями и приоритетами) и два ДОПУСТИМЫХ варианта тура: Вариант A и Вариант B.
Оба варианта полностью удовлетворяют обязательным требованиям (бюджет, состав гостей, допустимость дат).

Твоя задача — объективно определить, какой из двух вариантов лучше соответствует потребностям и мягким предпочтениям туриста, учитывая баланс между ценой и комфортом (trade-off):
- Если пользователь готов пойти на компромисс ради экономии, и скидка действительно существенна относительно уступки — побеждает более экономичный тур.
- Если пользователю принципиально важен заявленный комфорт (например, прямой рейс с детьми или домашним животным, или удобный утренний вылет, или питание в отеле), а доплата разумна — побеждает более удобный тур.
- Если варианты равнозначны по полезности, либо разница в цене точно компенсирует разницу в комфорте — ответь TIE.

Формат твоего ответа строго JSON:
{
  "winner": "A" | "B" | "TIE",
  "confidence": "high" | "medium" | "low",
  "key_tradeoff": "краткое описание компромисса (1 предложение)",
  "rationale": "подробное объяснение (2-3 предложения), почему выбранный вариант лучше для данного пользователя"
}
Отвечай ТОЛЬКО валидным JSON-объектом без лишнего текста вокруг."""


def format_card(tour_data: Dict[str, Any]) -> Dict[str, Any]:
    """Форматирует стандартизированную карточку тура для судьи."""
    f = tour_data.get("features_v2") or {}
    return {
        "полная_цена_руб": tour_data.get("price"),
        "отель": {
            "название": tour_data.get("hotel_name"),
            "звезды": f"{tour_data.get('hotel_stars')}★",
            "завтрак": "Включен" if f.get("breakfast") else "Без питания (Room Only)",
            "расстояние_до_пляжа_км": f.get("dist_sea_km") if f.get("has_dist_sea") else "Не применимо / нет данных",
            "расстояние_до_центра_км": f.get("dist_center_km") if f.get("has_dist_center") else "Нет данных",
            "разрешено_с_животными": "Да" if f.get("pet_friendly") else "Нет",
        },
        "авиаперелет": {
            "номер_рейса": tour_data.get("flight_no"),
            "тип_перелета": "Прямой рейс" if f.get("is_direct_flight") else f"Рейс с пересадкой ({f.get('transfers')} сегм., стыковка {f.get('layover_hours')} ч)",
            "время_вылета": tour_data.get("departure_time") or f"{int(f.get('departure_hour', 0)):02d}:00",
        },
    }


async def judge_single_pair(
    client: Any,
    pair: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    model_name: str,
    swap_order: bool = False,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Отправляет одну пару на оценку в LLM.
    При swap_order=True меняет карточки A и B местами.
    """
    async with semaphore:
        card_a = format_card(pair["tour_b"] if swap_order else pair["tour_a"])
        card_b = format_card(pair["tour_a"] if swap_order else pair["tour_b"])

        user_content = {
            "запрос_туриста": pair["query_text"],
            "архетип_клиента": pair["archetype"],
            "направление": pair["destination_id"],
            "состав_гостей": f"{pair['request_slots']['adults_count']} взр., {pair['request_slots']['kids_count']} детей",
            "длительность_ночей": pair["request_slots"]["nights"],
            "бюджет_лимит_руб": pair["request_slots"]["budget"],
            "Вариант A": card_a,
            "Вариант B": card_b,
        }

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False, indent=2)},
                    ],
                    temperature=0.0,
                    max_tokens=1200,
                )
                raw_text = (resp.choices[0].message.content or "").strip()
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""

                # Очистка markdown блоков
                cleaned = raw_text
                if "```" in cleaned:
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip())
                    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

                match = re.search(r"\{[\s\S]*\}", cleaned)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    parsed = json.loads(cleaned)

                raw_winner = str(parsed.get("winner", "TIE")).strip().upper()
                if raw_winner not in ("A", "B", "TIE"):
                    raw_winner = "TIE"

                # Если мы меняли порядок, инвертируем победителя обратно
                if swap_order:
                    if raw_winner == "A":
                        final_winner = "B"
                    elif raw_winner == "B":
                        final_winner = "A"
                    else:
                        final_winner = "TIE"
                else:
                    final_winner = raw_winner

                return {
                    "pair_id": pair["pair_id"],
                    "winner": final_winner,
                    "confidence": parsed.get("confidence", "medium"),
                    "key_tradeoff": parsed.get("key_tradeoff", ""),
                    "rationale": parsed.get("rationale", ""),
                    "reasoning_trace": reasoning[:300] if reasoning else "",
                    "status": "success",
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue

        logger.warning("Ошибка разметки пары %d после %d попыток: %s", pair["pair_id"], max_retries + 1, last_error)
        return {
            "pair_id": pair["pair_id"],
            "winner": "TIE",
            "confidence": "none",
            "key_tradeoff": "error",
            "rationale": last_error,
            "reasoning_trace": "",
            "status": "error",
        }


async def label_dataset(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    concurrency: int = 5,
    swap_consistency_sample: int = 25,
) -> Dict[str, Any]:
    """
    Выполняет разметку всех пар в пуле с параллелизацией и проверкой swap-consistency.
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    if limit is not None and limit > 0:
        pairs = pairs[:limit]

    client = LLMService.get_client()
    if client is None:
        raise RuntimeError("LLMService client is not configured. Check OPENAI_API_KEY.")

    model_name = settings.OPENAI_MODEL
    logger.info("Старт разметки %d пар с помощью LLM-судьи (%s)...", len(pairs), model_name)

    sem = asyncio.Semaphore(concurrency)

    # 1. Основной прогон разметки
    tasks = [judge_single_pair(client, p, sem, model_name, swap_order=False) for p in pairs]
    results = await asyncio.gather(*tasks)

    # 2. Swap consistency check на подвыборке
    swap_pairs = pairs[:swap_consistency_sample]
    swap_tasks = [judge_single_pair(client, p, sem, model_name, swap_order=True) for p in swap_pairs]
    swap_results = await asyncio.gather(*swap_tasks)

    # Сравнение консистентности
    swap_map = {res["pair_id"]: res["winner"] for res in swap_results}
    consistent_count = 0
    for res in results[:swap_consistency_sample]:
        pid = res["pair_id"]
        if pid in swap_map and res["winner"] == swap_map[pid]:
            consistent_count += 1
    swap_consistency_pct = round(100.0 * consistent_count / max(1, len(swap_pairs)), 2)
    logger.info("Swap Consistency Check на %d парах: %.2f%% совпадений", len(swap_pairs), swap_consistency_pct)

    # Объединяем разметку с исходными парами
    labeled_pairs = []
    winner_counts = {"A": 0, "B": 0, "TIE": 0}
    for orig_pair, judgment in zip(pairs, results):
        p_copy = dict(orig_pair)
        winner = judgment["winner"]
        p_copy["label"] = {
            "winner": winner,
            "confidence": judgment["confidence"],
            "key_tradeoff": judgment["key_tradeoff"],
            "rationale": judgment["rationale"],
            "judge_model": model_name,
            "source": f"llm_judge:{model_name}",
        }
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        labeled_pairs.append(p_copy)

    output_payload = {
        "dataset_name": "CSAI-2026 Labeled Preference Trade-Offs (V2)",
        "source_candidates_file": input_path.name,
        "judge_model": model_name,
        "judge_temperature": 0.0,
        "total_pairs_labeled": len(labeled_pairs),
        "winner_distribution": winner_counts,
        "swap_consistency_sample_size": len(swap_pairs),
        "swap_consistency_pct": swap_consistency_pct,
        "labeled_pairs": labeled_pairs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Размеченный датасет сохранен в %s (размер: %.1f КБ)", output_path, output_path.stat().st_size / 1024)

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Слепая разметка пар туров с помощью LLM-судьи")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Входной файл кандидатов")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Выходной файл с метками")
    parser.add_argument("--limit", type=int, default=None, help="Лимит пар для быстрой проверки")
    parser.add_argument("--concurrency", type=int, default=5, help="Количество параллельных запросов к API")
    parser.add_argument("--swap-sample", type=int, default=30, help="Размер подвыборки для проверки swap-consistency")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    res = asyncio.run(
        label_dataset(
            input_path=Path(args.input),
            output_path=Path(args.output),
            limit=args.limit,
            concurrency=args.concurrency,
            swap_consistency_sample=args.swap_sample,
        )
    )

    print("=" * 70)
    print("CSAI 2026: Разметка пула пар завершена")
    print("=" * 70)
    print(f"Всего размечено пар       : {res['total_pairs_labeled']}")
    print(f"Модель судьи              : {res['judge_model']}")
    print(f"Swap Consistency          : {res['swap_consistency_pct']}% (на {res['swap_consistency_sample_size']} парах)")
    print(f"Распределение меток       : A={res['winner_distribution']['A']}, B={res['winner_distribution']['B']}, TIE={res['winner_distribution']['TIE']}")


if __name__ == "__main__":
    main()
