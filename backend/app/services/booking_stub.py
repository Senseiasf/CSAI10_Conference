"""Explicit demo responses without booking, payments or changes to existing bookings."""
import re

def detect_booking_action(message: str) -> str | None:
    text = message.lower().replace("ё", "е")
    booking = bool(re.search(r"\b(?:брон\w*|бронирован\w*|заказ\w*)", text))
    if re.search(r"\bне\s+(?:(?:хочу|нужно|надо|буду)\s+)?(?:брони\w*|заброни\w*|отмен\w*|перен[ео]\w*|оплач\w*|оплат\w*)", text):
        return None
    if booking and re.search(r"\b(?:отмени\w*|отменить|аннулир\w*)", text):
        return "cancel"
    if booking and re.search(r"\b(?:перенес\w*|перенос\w*|перенест\w*|сдвин\w*|измени\w*|поменя\w*)", text):
        return "reschedule"
    if re.search(r"\b(?:заброни\w*|бронировать|оплатить|оплати|купить|купи)\b", text):
        return "book"
    if re.search(r"\b(?:оформи\w*|подтверди\w*)", text) and (booking or re.search(r"\b(?:тур\w*|отел\w*|поездк\w*)", text)):
        return "book"
    return None

def booking_stub(action: str, cart: dict) -> dict:
    messages = {
        "book": "Бронирование отеля пока недоступно: это демонстрационная версия планировщика. Бронь не создана, оплата не выполнялась. Ваш план сохранён — его можно продолжить редактировать.",
        "cancel": "Отмена брони пока недоступна в демонстрационной версии. Бронирование не отменено. Если вы оформляли его в другом сервисе, обратитесь туда. Текущий план поездки сохранён.",
        "reschedule": "Перенос оформленной брони пока недоступен в демонстрационной версии. Даты бронирования не изменены. Могу отдельно пересчитать план поездки на другие даты — это не изменит существующую бронь.",
    }
    return {"text": messages[action], "state": "BOOKING_UNAVAILABLE",
            "booking_action": {"action": action, "available": False}, "cart": cart}
