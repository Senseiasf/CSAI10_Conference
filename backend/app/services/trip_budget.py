"""Parse explicit trip budgets in rubles without guessing dates or guest counts."""
import re
from decimal import Decimal


AMOUNT = r"(?P<amount>-?(?:\d{1,3}(?:[ \u00a0\u202f.]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?))"
UNIT = r"\s*(?P<unit>тыс(?:яч\w*)?\.?|[кk](?!\w)|млн\.?|миллион\w*)?"
MONEY = AMOUNT + UNIT + r"\s*(?:₽|руб\w*)?"
CHEAPER = r"(?:дешевле|подешевле|бюджетнее|экономнее|сниз\w*\s+(?:цену|бюджет)|уменьш\w*\s+(?:цену|бюджет))"


def validate_package_totals(breakdown: dict) -> None:
    """Reject inconsistent package arithmetic before it can be recommended."""
    mandatory = breakdown["mandatory"]
    travelers = mandatory["travelers_count"]
    if travelers < 1 or mandatory["rooms_count"] < 1 or breakdown["nights"] < 1:
        raise ValueError("Invalid traveler, room or night count")
    money_fields = ("flight_out_price", "flight_ret_price", "flight_out_per_person_rub", "flight_ret_per_person_rub",
                    "flight_roundtrip", "flight_price_roundtrip", "hotel_per_night", "hotel_total", "transfer", "pet_total",
                    "total_mandatory_rub")
    if any(type(mandatory[field]) is not int or mandatory[field] < 0 for field in money_fields):
        raise ValueError("Package prices must be nonnegative whole rubles")
    for direction in ("out", "ret"):
        if mandatory[f"flight_{direction}_price"] != mandatory[f"flight_{direction}_per_person_rub"] * travelers:
            raise ValueError("Flight price does not include every traveler exactly once")
    flight_total = mandatory["flight_out_price"] + mandatory["flight_ret_price"]
    if mandatory["flight_roundtrip"] != flight_total or mandatory["flight_price_roundtrip"] != flight_total:
        raise ValueError("Round-trip flight total is inconsistent")
    if mandatory["hotel_total"] != mandatory["hotel_per_night"] * breakdown["nights"] * mandatory["rooms_count"]:
        raise ValueError("Hotel total is inconsistent")
    total = flight_total + mandatory["hotel_total"] + mandatory["transfer"] + mandatory["pet_total"]
    if mandatory["total_mandatory_rub"] != total:
        raise ValueError("Mandatory package total is inconsistent")
    bounds = breakdown["total_trip_budget_range_rub"]
    onsite = breakdown["onsite_flexible"]["total_onsite_range_rub"]
    if any(bounds[key] != total + onsite[key] for key in ("min", "max")) or bounds["min"] > bounds["max"]:
        raise ValueError("Full trip price range is inconsistent")


def update_trip_budget(message: str, profile: dict, cart: dict) -> str | None:
    """Store a total ceiling; return a question when the requested scope is unclear."""
    text = message.lower()
    reduction = re.search(CHEAPER + r"\s+на\s+" + MONEY, text)
    absolute = re.search(r"\b(?:бюджет(?:ом)?(?:\s+(?:до|не более))?|лимит|не больше|не более|не дороже|улож\w*\s+в|до)(?:\s*[-:—]\s+|\s*)" + MONEY, text)
    match = reduction or absolute
    if not match and profile.get("awaiting_budget"):
        match = re.fullmatch(r"\s*" + MONEY + r"\s*", text)
    if not match:
        if re.search(CHEAPER + r"\s+на\b", text):
            return "На сколько рублей снизить полную стоимость поездки? Укажите сумму числом."
        if re.search(r"\b(?:бюджет|лимит)(?:ом|а|у|е)?\b", text):
            profile["awaiting_budget"] = True
            return "Укажите общий бюджет всей поездки числом в рублях, например: «бюджет до 100 тысяч»."
        return None

    tail = text[match.end():]
    if re.match(r"\.\d{4}\b", tail):
        return None  # A date such as 'до 10.11.2026'.
    if re.match(r"\s*(?:доллар\w*|евро|usd|eur|\$|€)", tail):
        return "Пока считаю бюджет в рублях. Укажите общий лимит поездки в ₽."
    if re.match(r"\s*(?:за\s+(?:ночь|сутки|день)|в\s+(?:день|сутки)|на\s+(?:человека|одного)|с\s+человека)", tail):
        return "Укажите общий бюджет всей поездки в рублях, включая перелёт, отель, трансфер и расходы на месте."
    # Bare 'до 7 ночей' is a duration, not a monetary constraint.
    if re.match(r"\s*(?:ноч\w*|дн\w*|сут\w*|недел\w*|человек\w*|лет)\b", tail):
        return None
    raw_amount = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", match["amount"])
    value = Decimal(re.sub(r"[ \u00a0\u202f]", "", raw_amount).replace(",", "."))
    unit = match["unit"] or ""
    factor = 1_000_000 if unit.startswith(("млн", "миллион")) else 1000 if unit else 1
    amount = int(value * factor)
    if amount <= 0:
        return "Укажите положительный общий бюджет поездки в рублях."

    if reduction:
        previous_total = cart.get("total_trip_budget_range_rub", {}).get("max")
        if previous_total is None:
            return "Какой общий лимит поездки установить в рублях? Пока нет полной оценки, от которой можно посчитать экономию."
        limit = previous_total - amount
        if limit <= 0:
            return "Такое снижение оставляет нулевой или отрицательный бюджет. Укажите новый общий лимит поездки в рублях."
        # A reduction must never silently raise an existing ceiling.
        limit = min(limit, profile.get("budget_rub", limit))
    else:
        limit = amount

    profile["budget_rub"] = limit
    profile.pop("awaiting_budget", None)
    return None
