"""Preserve soft travel preferences separately from eligibility constraints."""
import re
from decimal import Decimal

from app.services.graph_service import minutes_since_midnight


ATTRIBUTES = {
    "direct": r"прям\w*\s+(?:авиа)?(?:рейс\w*|перел[её]т\w*)(?:\s+туда)?|прямой|без пересадок",
    "morning": r"утренн\w*(?:\s+(?:вылет\w*|рейс\w*))?|вылет\w*\s+утром|утром",
    "day": r"дневн\w*\s+(?:вылет\w*|рейс\w*)|вылет\w*\s+днем",
    "evening": r"вечерн\w*\s+(?:вылет\w*|рейс\w*)|вылет\w*\s+вечером",
    "breakfast": r"(?:включ[её]нн\w+\s+)?завтрак\w*",
    "stars": r"отел\w*\s+(?:более высок\w+\s+категори\w+|повыше классом|с большим числом звезд)|больше звезд",
}
SOFT_WORDS = r"предпочита\w+|предпочтительн\w*|желател(?:ен|ьн\w*)|лучше бы|хотелось бы|главное"
HARD_WORDS = r"только|обязател(?:ен|ьн\w*)|строго|исключительно|непременно"
NUMBER = r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+(?:[,.]\d+)?"
AMOUNT = (rf"(?P<amount>{NUMBER})(?![\d,.]|[ \u00a0\u202f]\d{{3}}\b)\s*(?P<thousands>тыс\w*\.?|[кk]\b)?"
          rf"\s*(?:₽|руб(?:лей|ля|\.)?|р\.)?(?!\s*(?:\.\s*)?(?:[$€]|доллар|евро|usd\b|eur\b|тыс|[-–]\s*\d))")
LIMIT = r"(?P<comparison>до|не\s+(?:более|выше|дороже|больше)|не\s+превыша\w*|максимум|выше|больше|превыша\w*)"
SURCHARGE = r"(?:доплат\w*|переплат\w*|разниц\w*\s+в\s+цене)"


def normalize_preference(value):
    if not isinstance(value, dict):
        return None
    attribute = value.get("attribute")
    if not isinstance(attribute, str) or attribute not in ATTRIBUTES:
        return None
    cap = value.get("max_extra_rub")
    if cap is not None and (type(cap) is not int or cap < 0):
        return None
    return {"attribute": attribute, "max_extra_rub": cap}


def _amount(match):
    amount = Decimal(re.sub(r"\s+", "", match["amount"]).replace(",", "."))
    if match["thousands"]:
        amount *= 1000
    return int(amount) if amount == amount.to_integral_value() else None


def parse_preference(text):
    """Recognize clear Russian preference statements and explicit ruble premiums."""
    text = text.casefold().replace("ё", "е")
    precise, qualitative = set(), set()
    for attribute, words in ATTRIBUTES.items():
        term = rf"(?<!\w)(?:{words})(?!\w)"
        patterns = (
            rf"{SURCHARGE}\s+за\s+{term}\s+{LIMIT}\s+{AMOUNT}",
            rf"{term}[^.;!?]{{0,48}}?{SURCHARGE}\s*{LIMIT}\s+{AMOUNT}",
            rf"(?:готов\w*|могу|соглас\w+)\s+{SURCHARGE}\s*(?:{LIMIT}\s+)?{AMOUNT}\s+за\s+{term}",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if re.search(r"(?:\bне|\bне\s+готов\w*)\s+$", text[:match.start()]):
                    continue
                comparison = match["comparison"] or ""
                if comparison.startswith(("выше", "больше", "превыша")):
                    before = text[max(0, match.start() - 80):match.start()]
                    surrounding = before + text[match.start():match.end() + 80]
                    if not re.search(r"\bесли\b", before) or not re.search(
                        r"дешев|откаж|выберу\s+рейс\s+с\s+пересадк", surrounding
                    ):
                        continue
                cap = _amount(match)
                if cap is not None:
                    precise.add((attribute, cap))
        if re.search(rf"(?:{SOFT_WORDS})\s+(?:(?!не\b)\w+\s+){{0,4}}?{term}"
                     rf"|{term}\s+(?:{SOFT_WORDS})(?!\w)", text):
            qualitative.add(attribute)
    if len(precise) == 1:
        attribute, cap = next(iter(precise))
        return {"attribute": attribute, "max_extra_rub": cap}
    if precise:
        return None
    if len(qualitative) == 1:
        return {"attribute": next(iter(qualitative)), "max_extra_rub": None}
    return None


def merge_preferences(request, text, model_preference=None, *, from_model=False):
    """Attach the preference and undo an LLM's promotion of the same soft slot to hard."""
    parsed = parse_preference(text)
    supplied = normalize_preference(model_preference)
    if supplied and supplied["max_extra_rub"] is not None and not (
        parsed and parsed["max_extra_rub"] is not None
    ):
        return {**request, "preference_needs_clarification": True}
    preference = parsed or supplied or normalize_preference(request.get("soft_preference"))
    if preference is None:
        return request
    result = {**request, "soft_preference": preference}
    attribute = preference["attribute"]
    if from_model:
        words = ATTRIBUTES[attribute]
        normalized = text.casefold().replace("ё", "е")
        pattern = (rf"(?:{HARD_WORDS})\s+(?:нужен\s+|нужно\s+)?(?:{words})(?!\w)"
                   rf"|(?:{words})\s+(?:{HARD_WORDS})\b(?!\s+если)")
        hard = any(not re.search(r"\bне\s+$", normalized[:match.start()])
                   for match in re.finditer(pattern, normalized))
        if not hard:
            if attribute == "direct":
                result["prefer_direct_flight"] = False
            if result.get("flight_time_pref") == attribute:
                result["flight_time_pref"] = None
    return result


def matches_preference(candidate, attribute, reference):
    route, feat = candidate["route"], candidate["features"]
    if attribute == "direct":
        return route.flight["flight_type"] == "direct"
    if attribute == "stars":
        return feat["hotel_stars"] > reference["features"]["hotel_stars"]
    if attribute == "breakfast":
        return bool(feat["breakfast"])
    start = {"morning": 360, "day": 720, "evening": 1080}[attribute]
    return start <= minutes_since_midnight(route.flight["departure_time"]) < start + 360


def prioritize_preference(ranked, value):
    """Use a stated premium exactly. Without a price limit, expose a preferred alternative."""
    preference = normalize_preference(value)
    if not ranked or preference is None:
        return ranked
    attribute, cap = preference["attribute"], preference["max_extra_rub"]
    cheapest = min(ranked, key=lambda c: (c["features"]["price"], c["route"].key))
    reference_price = cheapest["features"]["price"]
    preferred = next((c for c in ranked if matches_preference(c, attribute, cheapest)
                      and (cap is None or c["features"]["price"] - reference_price <= cap)), None)
    first = ranked[0] if cap is None else preferred or cheapest
    leaders = [first]
    if cap is None and preferred is not None and preferred is not first:
        leaders.append(preferred)
    return leaders + [c for c in ranked if all(c is not leader for leader in leaders)]


def select_offers(ranked, limit=3, diversify=False):
    """Preserve the best-ranked offer and reserve room for other retrieved destinations."""
    if limit <= 0:
        return []
    if not diversify or limit < 2 or len(ranked) < 2:
        return ranked[:limit]
    selected, skipped, destinations = [], [], set()
    for candidate in ranked:
        dest = candidate["route"].destination
        if dest in destinations:
            skipped.append(candidate)
        else:
            selected.append(candidate)
            destinations.add(dest)
            if len(selected) == limit:
                return selected
    return selected + skipped[:limit - len(selected)]
