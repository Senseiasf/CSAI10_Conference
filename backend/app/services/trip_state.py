"""Recognize explicit trip edits and answer questions from saved trip facts."""
import re


DESTINATION_KEYWORDS = {
    "thailand": ("thailand", "таиланд", "тайланд", "пхукет", "тай", "самуи"),
    "uae": ("uae", "оаэ", "дуба", "абу-даби", "эмират"),
    "turkey": ("turkey", "турци", "анталь", "стамбул", "белек", "кемер"),
    "sochi": ("sochi", "сочи", "сириус", "полян", "адлер", "роза хутор"),
    "maldives": ("maldives", "мальдив", "мале"),
    "egypt": ("egypt", "египет", "шарм", "хургад", "синай"),
    "spb": ("spb", "спб", "питер", "петербург", "санкт-петербург"),
    "altai": ("altai", "алтай", "манжерок", "телецк", "горно-алтайск"),
    "georgia": ("georgia", "грузи", "тбилиси", "батуми", "гудаури", "казбеги"),
}
EXACT_ALIASES = set(DESTINATION_KEYWORDS) | {
    "тай", "самуи", "оаэ", "спб", "сочи", "мале", "абу-даби", "тбилиси", "батуми", "гудаури", "казбеги",
}


def requested_destination(message: str) -> str | None:
    text = message.lower().replace("ё", "е")
    matches = []
    aliases = {keyword: destination for destination, keywords in DESTINATION_KEYWORDS.items() for keyword in keywords}
    pattern = r"\b(" + "|".join(
        re.escape(keyword) + (r"(?=\W|$)" if keyword in EXACT_ALIASES else "")
        for keyword in sorted(aliases, key=len, reverse=True)
    ) + r")\w*"
    for match in re.finditer(pattern, text):
        prefix = text[:match.start()]
        # Match the whole city once, so "Петербург" inside an origin such as
        # "из Санкт-Петербурга" cannot become a second destination match.
        if re.search(r"\b(?:(?:из|с)(?:\s+города)?|вместо|кроме|не(?:\s+в)?)\s*$", prefix):
            continue
        matches.append((match.start(), aliases[match.group(1)]))
    return max(matches)[1] if matches else None


def is_trip_edit(message: str) -> bool:
    text = message.lower().replace("ё", "е")
    return bool(re.search(
        r"дешев|подешев|бюджет|лимит|дорого|экономнее|снизь\s+цену|"
        r"(?:другой|поменяй|смени)\s+(?:отель|город|номер|направление)|"
        r"трансфер|только\s+прямой|без\s+пересадок|вылет|рейс\s+попозже|"
        r"\d+\s*(?:ноч|дн|сут|взросл|человек|дет|ребен|номер|комнат)|"
        r"\bнас\s+\d+|\d{1,2}\.\d{1,2}|\d{4}-\d{2}-\d{2}|"
        r"\b(?:вдвоем|втроем|вчетвером|впятером|соло|заезд|выезд|сдвинь)|"
        r"на\s+(?:неделю|месяц|выходные)|с\s+(?:собак|кошк|питомц)|без\s+(?:детей|собаки|кошки|питомц)",
        text,
    ))


def estimate_request(message: str) -> str | None:
    text = message.lower().replace("ё", "е")
    if re.search(r"(?:нижн\w*|минимальн\w*)\s+(?:оценк|границ)|по\s+минимуму", text):
        return "min"
    if re.search(r"(?:верхн\w*|максимальн\w*)\s+(?:оценк|границ)|по\s+максимуму", text):
        return "max"
    return None


def current_trip_question(message: str) -> bool:
    text = message.lower().replace("ё", "е")
    return bool(re.search(
        r"(?:почему|откуда).*(?:сочи|питер|петербург|направлен|город)|"
        r"ты.*(?:писал|говорил).*(?:питер|сочи|направлен)|"
        r"(?:какое|какой|куда|где).*(?:сейчас|выбран|в\s+плане|едем)|"
        r"(?:текущий|сохраненный|мой)\s+план",
        text,
    )) and not is_trip_edit(message)


def describe_current_trip(profile: dict, cart: dict) -> str:
    destination = cart.get("destination")
    if not destination:
        return "В текущем плане направление ещё не выбрано. Куда хотите поехать?"
    parts = [f"В сохранённом плане направление — **{destination}**."]
    for label, value in (("Даты", profile.get("dates") or cart.get("dates")),
                         ("Ночей", profile.get("nights", cart.get("nights"))),
                         ("Взрослых", profile.get("adults_count")),
                         ("Детей", profile.get("kids_count"))):
        if value is not None:
            parts.append(f"{label}: {value}.")
    if profile.get("budget_rub") is not None:
        parts.append(f"Общий бюджет: {profile['budget_rub']:,} ₽.")
    parts.append("Именно эти параметры используются для расчёта поездки.")
    return "\n".join(parts)


def describe_budget_estimate(quote: dict, mode: str, budget: int | None) -> str:
    bounds = quote["total_trip_budget_range_rub"]
    label = "нижняя" if mode == "min" else "верхняя"
    lines = [f"Для того же варианта «{quote['hotel']}» в направлении «{quote['destination']}» "
             f"на {quote['nights']} ночей {label} оценка — **{bounds[mode]:,} ₽**.",
             f"Даты сохранённого расчёта: {quote['checkin_date']} — {quote['checkout_date']}.",
             f"Полный диапазон: **{bounds['min']:,}–{bounds['max']:,} ₽**."]
    if budget is not None:
        difference = bounds[mode] - budget
        lines.append(f"Общий лимит: {budget:,} ₽. " + (
            f"Превышение по этой оценке: {difference:,} ₽." if difference > 0
            else f"Остаток по этой оценке: {-difference:,} ₽."
        ))
    lines.append("Нижняя граница предполагает минимальные расходы на месте; это ориентир, а не гарантия итоговой цены.")
    return "\n\n".join(lines)
