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
    "altai": ("altai", "алтай", "алтае", "алтая", "алтаю", "манжерок", "телецк", "горно-алтайск"),
    "georgia": ("georgia", "грузи", "тбилиси", "батуми", "гудаури", "казбеги"),
}
EXACT_ALIASES = set(DESTINATION_KEYWORDS) | {
    "тай", "самуи", "оаэ", "спб", "сочи", "мале", "абу-даби", "тбилиси", "батуми", "гудаури", "казбеги",
    "алтае", "алтая", "алтаю",
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
        if re.search(r"\b(?:(?:из|с)(?:\s+города)?|вместо|кроме|не(?:\s+(?:в|на))?)\s*$", prefix):
            continue
        matches.append((match.start(), aliases[match.group(1)]))
    return max(matches)[1] if matches else None


def is_trip_edit(message: str) -> bool:
    text = message.lower().replace("ё", "е")
    return bool(re.search(
        r"дешев|подешев|бюджет|лимит|дорого|экономнее|снизь\s+цену|"
        r"(?:уменьш\w*|сниз\w*|сократ\w*)\s+(?:(?:общ\w*|полную)\s+)?(?:стоимост\w*|цен\w*|расход\w*)|"
        r"\b(?:выбери|выберите|выбираю|переключи)\b|"
        r"(?:другой|поменяй|смени)\s+(?:отель|город|номер|направление)|"
        r"трансфер|только\s+прямой|без\s+пересадок|вылет|рейс\s+попозже|"
        r"\d+\s*(?:ноч|дн|сут|взросл|человек|дет|ребен|номер|комнат)|"
        r"\bнас\s+\d+|\d{1,2}\.\d{1,2}|\d{4}-\d{2}-\d{2}|"
        r"\b(?:вдвоем|втроем|вчетвером|впятером|соло|заезд|выезд|сдвинь)|"
        r"на\s+(?:неделю|месяц|выходные)|с\s+(?:собак|кошк|питомц)|без\s+(?:детей|собаки|кошки|питомц)",
        text,
    ))


def destination_choice(message: str) -> str | None:
    text = message.lower().replace("ё", "е").strip()
    advice = re.match(r"^(?:а\s+)?(?:почему|откуда|что|как|расскажи|какая|какой|какие|нужна|сколько)\b", text)
    if advice and not re.search(r"\b(?:подбери|собери|поедем|хочу|выбери|смени|поменяй)\b", text):
        return None
    return requested_destination(message)


def wants_cheaper(message: str) -> bool:
    text = message.lower().replace("ё", "е")
    return bool(re.search(
        r"дешев|подешев|бюджетнее|экономнее|сэконом|дорого|"
        r"(?:уменьш\w*|сниз\w*|сократ\w*)\s+(?:(?:общ\w*|полную)\s+)?(?:стоимост\w*|цен\w*|расход\w*)",
        text,
    ))


def transfer_preference(message: str) -> bool | None:
    """Return True for self-organized transport, False for an included transfer."""
    text = message.lower().replace("ё", "е")
    if re.search(r"\bне\s+(?:убирай|исключай|отменяй)\s+трансфер|\bне\s+хочу\s+без\s+трансфера", text):
        return False
    remove = (
        r"\b(?:без\s+трансфера|(?:убери|уберите|исключи|удали|отмени)\s+трансфер\w*|"
        r"трансфер\w*\s+(?:(?:мне|нам)\s+)?не\s+(?:нуж\w*|надо|требуется)|"
        r"не\s+(?:нуж\w*|надо|требуется)\s+(?:(?:мне|нам)\s+)?трансфер\w*)\b"
    )
    if re.search(remove, text) or any(phrase in text for phrase in (
        "поедем сами", "на каршеринге", "возьмем такси сами", "своим ходом",
    )):
        return True
    if re.search(r"\b(?:верни|добавь|включи)\s+(?:обратно\s+)?трансфер\w*|\bтрансфер\w*\s+нужен\b|\bнужен\s+трансфер\w*", text):
        return False
    return None


def hotel_selection(message: str) -> str | None:
    match = re.match(r"\s*(?:выбери(?:те)?|выбираю|переключи(?:сь)?\s+на)\s+(.+?)\s*[.!?]?\s*$", message, re.IGNORECASE)
    if not match:
        return None
    raw_query = re.sub(r"^(?:отель|гостиницу)\s+", "", match[1], flags=re.IGNORECASE)
    is_quoted = raw_query.lstrip().startswith(('«', '"', '“'))
    query = raw_query.strip(' «»"“”.!?')
    generic_request = re.search(r"\b(?:отел\w*|гостиниц\w*|резорт\w*)\s+(?:в|на|для|с|у|рядом|где)\b", match[1], re.IGNORECASE)
    if generic_request and not is_quoted:
        return None
    if not is_quoted and requested_destination(query) and len(query.split()) <= 2 and not re.search(r"отел|hotel|гостиниц", query, re.IGNORECASE):
        return None
    if query.casefold() in ("другой", "другой отель", "подешевле", "самый дешевый", "самый дешёвый"):
        return None
    return query or None


def matching_hotels(query: str, hotels: list[dict]) -> list[dict]:
    def normalize(value):
        return re.sub(r"[^\w]+", " ", value.casefold().replace("ё", "е")).strip()
    name = normalize(query)
    exact = [hotel for hotel in hotels if name in (normalize(hotel["name"]), normalize(hotel["id"]))]
    return exact or [hotel for hotel in hotels if len(name) >= 4 and name in normalize(hotel["name"])]


def rejected_destinations(message: str) -> set[str]:
    text = message.lower().replace("ё", "е")
    rejected = set()
    for destination, keywords in DESTINATION_KEYWORDS.items():
        for keyword in keywords:
            suffix = r"\b" if keyword in EXACT_ALIASES else r"\w*\b"
            if re.search(r"\b(?:не\s+(?:(?:в|на)\s+)?|кроме\s+|без\s+)" + re.escape(keyword) + suffix, text):
                rejected.add(destination)
    return rejected


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


def single_destination_name(destination_id: str, region_name: str, hotel_city: str | None) -> str:
    """Name the selected stay rather than every city covered by its catalogue region."""
    if destination_id == "sochi":
        if hotel_city and re.search(r"красн\w*\s+полян|эсто[ -]садок|роза\s+хутор", hotel_city, re.IGNORECASE):
            return hotel_city
        return "Сочи"
    if hotel_city and "(" in region_name and " и " in region_name:
        return f"{region_name.split('(')[0].strip()} ({hotel_city})"
    return region_name


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
