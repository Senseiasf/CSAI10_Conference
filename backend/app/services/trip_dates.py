from copy import deepcopy
from datetime import date, timedelta
from typing import Iterable

from app.domain.trip import TripDates


MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}
DAY_WORDS = {"день", "дня", "дней", "сутки", "суток"}
NIGHT_WORDS = {"ночь", "ночи", "ночей", "сутки", "суток"}
WEEK_WORDS = {"неделя", "недели", "недель", "неделю"}
DIRECTIONS = {"позже": 1, "вперед": 1, "вперёд": 1, "раньше": -1, "назад": -1}
NUMBER_WORDS = {
    "один": 1, "одну": 1, "два": 2, "две": 2, "пару": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}


class DateResolutionError(ValueError):
    pass


class TripDateResolver:
    def __init__(self, today: date | None = None):
        self.today = today or date.today()

    def shift(self, dates: TripDates, days: int) -> TripDates:
        if not dates.checkin_date or not dates.checkout_date:
            raise DateResolutionError("Сначала укажите даты поездки, а затем период сдвига.")
        checkin = date.fromisoformat(dates.checkin_date) + timedelta(days=days)
        checkout = date.fromisoformat(dates.checkout_date) + timedelta(days=days)
        if checkin < self.today:
            raise DateResolutionError("Новая дата заезда уже прошла. Укажите другой перенос.")
        return TripDates(checkin_date=checkin.isoformat(), checkout_date=checkout.isoformat(), nights=dates.nights)

    def resolve(self, previous: TripDates, candidate: TripDates, changed_fields: Iterable[str]) -> TripDates:
        changed = set(changed_fields)
        checkin = self._parse_iso(candidate.checkin_date, "заезда") if candidate.checkin_date else None
        checkout = self._parse_iso(candidate.checkout_date, "выезда") if candidate.checkout_date else None
        nights = candidate.nights

        if "checkin_date" in changed and checkin and checkin < self.today:
            raise DateResolutionError("Дата заезда уже прошла. Укажите дату в будущем.")
        if "checkin_date" in changed:
            if "checkout_date" not in changed:
                checkout = checkin + timedelta(days=nights) if checkin and nights else None
            if "nights" not in changed and checkin and checkout:
                nights = (checkout - checkin).days
        elif "checkout_date" in changed:
            if "nights" not in changed and checkin and checkout:
                nights = (checkout - checkin).days
        elif "nights" in changed and checkin and nights:
            checkout = checkin + timedelta(days=nights)

        if checkin and checkout:
            derived_nights = (checkout - checkin).days
            if derived_nights <= 0:
                raise DateResolutionError("Дата выезда должна быть позже даты заезда.")
            if nights is not None and nights != derived_nights:
                raise DateResolutionError("Даты заезда и выезда не совпадают с указанным числом ночей.")
            nights = derived_nights
        if nights is not None and not 1 <= nights <= 60:
            raise DateResolutionError("Укажите длительность поездки от 1 до 60 ночей.")
        return TripDates(
            checkin_date=checkin.isoformat() if checkin else None,
            checkout_date=checkout.isoformat() if checkout else None,
            nights=nights,
        )

    def _parse_iso(self, value: str, label: str) -> date:
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise DateResolutionError(f"Некорректная дата {label}.") from exc


def update_trip_dates(message: str, profile: dict, cart: dict | None = None, today: date | None = None) -> str | None:
    next_profile = deepcopy(profile)
    next_cart = deepcopy(cart) if cart is not None else None
    previous = TripDates(
        checkin_date=profile.get("checkin_date") or (cart or {}).get("checkin_date"),
        checkout_date=profile.get("checkout_date") or (cart or {}).get("checkout_date"),
        nights=profile.get("nights") or (cart or {}).get("nights"),
    )
    candidate = previous.model_copy(deep=True)
    changed: set[str] = set()
    text = message.lower().replace("ё", "е")
    tokens = [_clean_token(token) for token in text.split()]
    tokens = [token for token in tokens if token]
    resolver = TripDateResolver(today)

    shift = _relative_shift(tokens)
    weekend = any(token.startswith(("выходн", "уикенд")) for token in tokens) or "weekend" in tokens
    numeric_dates = [(index, parsed) for index, token in enumerate(tokens) if (parsed := _numeric_date(token, resolver.today))]
    russian_range = _russian_range(tokens, resolver.today)
    russian_single = None if russian_range else _russian_single(tokens, resolver.today)
    duration, duration_index = _duration(tokens)
    if not numeric_dates and any(_looks_like_date(token) for token in tokens):
        return "Такой даты нет в календаре. Укажите корректную дату."

    try:
        if shift:
            amount, direction = shift
            if not previous.checkin_date:
                return "С какой даты сдвигаем поездку? Укажите дату заезда."
            start = date.fromisoformat(previous.checkin_date) + timedelta(days=amount * direction)
            candidate.checkin_date = start.isoformat()
            changed.add("checkin_date")
        elif len(numeric_dates) >= 2:
            (first_index, start), (second_index, end) = numeric_dates[:2]
            if not _is_range_separator(tokens[first_index + 1:second_index]):
                return "Уточните даты заезда и выезда."
            candidate.checkin_date = start.isoformat()
            candidate.checkout_date = end.isoformat()
            changed.update(("checkin_date", "checkout_date"))
        elif russian_range:
            start, end = russian_range
            candidate.checkin_date = start.isoformat()
            candidate.checkout_date = end.isoformat()
            changed.update(("checkin_date", "checkout_date"))
        elif numeric_dates:
            candidate.checkin_date = numeric_dates[0][1].isoformat()
            changed.add("checkin_date")
        elif russian_single:
            candidate.checkin_date = russian_single.isoformat()
            changed.add("checkin_date")
        elif weekend:
            days_to_saturday = (5 - resolver.today.weekday()) % 7
            candidate.checkin_date = (resolver.today + timedelta(days=days_to_saturday)).isoformat()
            changed.add("checkin_date")
            if duration is None:
                candidate.nights = 2
                changed.add("nights")
        elif _month_mentioned(tokens):
            return "Уточните день заезда, например: «1 ноября»."

        if duration is not None and not shift:
            if "checkout_date" in changed:
                checkin = date.fromisoformat(candidate.checkin_date)
                checkout = date.fromisoformat(candidate.checkout_date)
                if (checkout - checkin).days != duration:
                    return "Даты заезда и выезда не совпадают с указанным числом ночей."
            candidate.nights = duration
            changed.add("nights")
        elif duration_index is not None and not shift:
            return "Укажите длительность поездки от 1 до 60 ночей."

        if not changed:
            if any(token in ("сдвинь", "сдвиньте") for token in tokens) or "другие даты" in text:
                return "Укажите новую дату заезда или на сколько дней сдвинуть поездку."
            return None
        resolved = resolver.resolve(previous, candidate, changed)
    except (DateResolutionError, ValueError, OverflowError) as exc:
        if isinstance(exc, DateResolutionError):
            return str(exc)
        return "Такой даты нет в календаре. Укажите корректные даты поездки."

    next_profile.update(
        checkin_date=resolved.checkin_date,
        checkout_date=resolved.checkout_date,
        nights=resolved.nights,
    )
    if resolved.checkin_date and resolved.checkout_date:
        next_profile["dates"] = _format_range(resolved)
    elif resolved.nights:
        next_profile["dates"] = f"{resolved.nights} ночей"
    if next_cart is not None:
        next_cart.update(checkin_date=resolved.checkin_date, checkout_date=resolved.checkout_date)
        if "dates" in next_profile:
            next_cart["dates"] = next_profile["dates"]
        if shift:
            next_cart["date_shifted"] = True
    if shift:
        next_profile["date_shifted"] = True
        next_profile["date_shift_days"] = shift[0] * shift[1]
    profile.clear()
    profile.update(next_profile)
    if cart is not None:
        cart.clear()
        cart.update(next_cart)
    return None


def _clean_token(token: str) -> str:
    return token.strip(" \t\n\r,!?;()[]{}«»\"'")


def _looks_like_date(token: str) -> bool:
    if len(token) == 10 and token[4] == token[7] == "-":
        return token[:4].isdigit() and token[5:7].isdigit() and token[8:].isdigit()
    parts = token.split(".")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        return False
    return 1 <= int(parts[0]) <= 31 and 1 <= int(parts[1]) <= 12


def _numeric_date(token: str, today: date) -> date | None:
    try:
        if len(token) == 10 and token[4] == token[7] == "-":
            return date.fromisoformat(token)
        parts = token.split(".")
        if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
            return None
        day, month = int(parts[0]), int(parts[1])
        year = today.year if len(parts) == 2 else int(parts[2])
        if len(parts) == 3 and len(parts[2]) == 2:
            year += 2000
        result = date(year, month, day)
        if len(parts) == 2 and result < today:
            result = date(year + 1, month, day)
        return result
    except ValueError:
        return None


def _month_number(token: str) -> int | None:
    if token in ("май", "мая", "мае", "маю"):
        return 5
    for prefix, number in MONTHS.items():
        if prefix != "ма" and token.startswith(prefix):
            return number
    return None


def _year_after(tokens: list[str], index: int) -> int | None:
    if index + 1 < len(tokens) and tokens[index + 1].isdigit() and len(tokens[index + 1]) == 4:
        return int(tokens[index + 1])
    return None


def _russian_single(tokens: list[str], today: date) -> date | None:
    for index in range(len(tokens) - 1):
        if not tokens[index].isdigit():
            continue
        month = _month_number(tokens[index + 1])
        if not month:
            continue
        year = _year_after(tokens, index + 1) or today.year
        try:
            result = date(year, month, int(tokens[index]))
            if _year_after(tokens, index + 1) is None and result < today:
                result = date(year + 1, month, int(tokens[index]))
            return result
        except ValueError:
            return None
    return None


def _russian_range(tokens: list[str], today: date) -> tuple[date, date] | None:
    for index in range(len(tokens) - 3):
        if not (tokens[index].isdigit() and tokens[index + 1] in ("по", "до", "-") and tokens[index + 2].isdigit()):
            continue
        month = _month_number(tokens[index + 3])
        if not month:
            continue
        year = _year_after(tokens, index + 3) or today.year
        try:
            start = date(year, month, int(tokens[index]))
            end = date(year, month, int(tokens[index + 2]))
            if _year_after(tokens, index + 3) is None and start < today:
                start = date(year + 1, month, start.day)
                end = date(year + 1, month, end.day)
            return start, end
        except ValueError:
            return None
    return None


def _is_range_separator(tokens: list[str]) -> bool:
    return len(tokens) == 1 and tokens[0] in ("по", "до", "-", "—", "–")


def _number(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def relative_date_shift(message: str) -> int | None:
    text = message.lower().replace("ё", "е")
    tokens = [_clean_token(token) for token in text.split()]
    shift = _relative_shift([token for token in tokens if token])
    return shift[0] * shift[1] if shift else None


def _relative_shift(tokens: list[str]) -> tuple[int, int] | None:
    for index, token in enumerate(tokens):
        if token in DIRECTIONS:
            direction = DIRECTIONS[token]
            window = tokens[max(0, index - 3):index] + tokens[index + 1:index + 4]
            for position, item in enumerate(window):
                amount = _number(item)
                if amount is None:
                    continue
                following = window[position + 1:position + 3]
                if any(word in DAY_WORDS for word in following):
                    return amount, direction
                if any(word in WEEK_WORDS for word in following):
                    return amount * 7, direction
            before = tokens[max(0, index - 2):index]
            if any(word in WEEK_WORDS for word in before):
                return 7, direction
            if any(word in DAY_WORDS for word in before):
                return 1, direction
    return None


def _duration(tokens: list[str]) -> tuple[int | None, int | None]:
    for index, token in enumerate(tokens):
        amount = _number(token)
        if amount is None or index + 1 >= len(tokens):
            continue
        unit = tokens[index + 1]
        following = tokens[index + 2:index + 4]
        if unit in WEEK_WORDS:
            return amount * 7, index
        if unit in NIGHT_WORDS and not any(word in DIRECTIONS for word in following):
            return amount, index
    return None, None


def _month_mentioned(tokens: list[str]) -> bool:
    return any(_month_number(token) for token in tokens)


def _format_range(dates: TripDates) -> str:
    start = date.fromisoformat(dates.checkin_date)
    end = date.fromisoformat(dates.checkout_date)
    return f"{start:%d.%m.%Y} — {end:%d.%m.%Y} ({dates.nights} ночей)"
