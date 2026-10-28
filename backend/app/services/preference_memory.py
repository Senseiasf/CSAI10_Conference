"""Preferences retained between trips; dates, budget and cart are trip-specific."""
from copy import deepcopy

PREFERENCE_KEYS = (
    "interests", "has_kids", "has_pets", "private_room_required",
    "prefer_direct_flight", "origin_airport", "flight_time_pref",
    "adults_count", "kids_count", "party_size",
)

def remembered_preferences(profile: dict) -> dict:
    return {key: deepcopy(profile[key]) for key in PREFERENCE_KEYS if key in profile}

def preference_summary(profile: dict) -> str:
    labels = []
    for key, label in (("has_kids", "поездка с детьми"), ("has_pets", "размещение с питомцем"),
                       ("private_room_required", "отдельный номер"), ("prefer_direct_flight", "прямой рейс")):
        if profile.get(key):
            labels.append(label)
    if profile.get("origin_airport"):
        labels.append(f"вылет из {profile['origin_airport']}")
    interests = {"beach": "море", "mountains": "горы", "spa_wellness": "спа",
                 "history_culture": "экскурсии", "luxury": "премиальный отдых", "budget_friendly": "экономный отдых"}
    labels.extend(interests[tag] for tag in profile.get("interests", []) if tag in interests)
    return "Помню ваши предпочтения: " + ", ".join(labels) + ". Их можно изменить в чате." if labels else ""
