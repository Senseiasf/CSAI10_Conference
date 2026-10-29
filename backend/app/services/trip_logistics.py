"""Explain saved flight facts without changing a quoted trip."""
import re


def is_logistics_question(message: str) -> bool:
    text = message.lower().replace("ё", "е").strip()
    if not re.search(r"пересад|стыков|багаж|задерж\w*.*рейс|рейс.*задерж", text):
        return False
    if re.search(r"\b(?:подбери|подберите|выбери|выберите|замени|поменяй|смени|сделай|измени|перенеси|пересобери|составь|собери|организуй|спланируй|убери|добавь|увеличь|уменьши|нужен|нужна|нужны|давай)\b|\bхочу\s+(?!узнать|понять|уточнить|спросить|проверить)|подешев|дешевле", text):
        return False
    if re.fullmatch(r"(?:а\s+)?(?:можно\s+)?(?:только\s+прямой(?:\s+рейс)?|без\s+пересадок)[?.!]*", text):
        return False
    return bool(re.search(r"\?|\b(?:как|когда|где|что|какой|какая|какие|достаточно)\b|сколько|почему|хватит|успе\w*|риск\w*|безопасн\w*|мало|коротк\w*|поясни|объясни|расскажи", text))


def connection_summary(logistics: dict) -> str:
    connections = (logistics.get("route") or {}).get("connections") or []
    if connections:
        connection = connections[0]
        airport = connection.get("airport") or {}
        city = airport.get("name")
        code = airport.get("code")
        place = " ".join(str(value) for value in (city, f"({code})" if code else None) if value) or "аэропорт не указан"
        minutes = connection.get("duration_minutes")
        duration = f"{minutes} мин" if minutes is not None else "длительность не указана"
        note = connection.get("risk_note")
        facts = f"Пересадка: {place}, {duration}. " + (note or "Данных о риске пересадки недостаточно.")
        if connection.get("status") == "unknown" or connection.get("baggage_status") == "unknown":
            facts += " Статус пересадки: неизвестно; единый билет и сквозной багаж не подтверждены."
        return facts
    flight = logistics.get("flight", {})
    layover = logistics.get("timeline", {}).get("layover_details", {})
    if layover.get("is_connecting") or flight.get("flight_type") == "connecting":
        city = layover.get("city") or flight.get("layover_city")
        airport = layover.get("airport") or flight.get("layover_airport")
        place = " ".join(str(value) for value in (city, f"({airport})" if airport else None) if value) or "аэропорт не указан"
        minutes = layover.get("duration_minutes") or flight.get("layover_duration_minutes")
        duration = f"{minutes} мин" if minutes else "длительность не указана"
        note = layover.get("risk_evaluation") or flight.get("layover_risk_note")
        return f"Пересадка: {place}, {duration}. " + (note or "Данных для оценки риска и правил перевозки багажа недостаточно.")
    if flight.get("flight_type") == "direct" or layover.get("is_connecting") is False:
        return "В выбранном рейсе пересадка не предусмотрена."
    return "Подробные данные о пересадке в текущем плане не сохранены."


def describe_saved_logistics(cart: dict) -> str:
    parts = [connection_summary(cart.get("logistics", {}))]
    bounds = cart.get("total_trip_budget_range_rub")
    if bounds:
        parts.append(f"Полная оценка сохранённой поездки: {bounds['min']:,}–{bounds['max']:,} ₽.")
    parts.append("Отель, рейсы, даты и стоимость поездки сохранены без изменений.")
    return "\n\n".join(parts)
