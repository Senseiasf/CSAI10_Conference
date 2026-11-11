"""Select feasible hotel pairs using catalogue transport, before rendering a tour."""
from datetime import datetime, timedelta

from app.mcp.tools import MCPTools
from app.services.graph_service import GraphService


async def select_route(cfg, hotels_1, hotels_2, profile, nights_1, nights_2, budget, policy):
    travelers = max(1, profile.get("adults_count") or 2) + (profile.get("kids_count") or 0)
    rooms = max(1, profile.get("rooms_count") or 1)
    flying = cfg["intercity"]["type"] == "flight"
    flight = None
    if flying:
        flight = await GraphService.find_intercity_logistics(cfg["city_1"]["airport"], cfg["city_2"]["airport"])
        if not flight:
            return None
    outbound, second = {}, {}
    choices = []
    for h1 in hotels_1:
        outbound[h1["id"]] = await MCPTools.check_logistics_compatibility(
            airport_code=cfg["city_1"]["airport"], hotel_id=h1["id"],
            origin_airport_code=profile.get("origin_airport"),
            prefer_direct_flight=profile.get("prefer_direct_flight"),
            without_transfer=profile.get("without_transfer", False),
            flight_time_pref=profile.get("flight_time_pref"), travelers_count=travelers,
            return_airport_code=cfg["city_2"]["airport"],
        )
        logistics = outbound[h1["id"]]
        if not logistics.get("compatible"):
            continue
        for h2 in hotels_2:
            if flying:
                intercity = {
                    "type": "flight", "name": f"{flight['airline']} {flight['flight_no']}",
                    "operator": flight["airline"], "departure": flight["departure_time"],
                    "arrival": flight["arrival_time"], "duration": f"{flight['duration_minutes']} мин",
                    "price_rub": flight["price_rub"],
                    "description": "Рейс из каталога; в оценке использован доступный тариф туда-обратно. Цена одного сегмента не подтверждена.",
                }
                intercity_cost = flight["price_rub"] * travelers
                source, ready = cfg["city_2"]["airport"], flight["arrival_time"]
            else:
                source, ready = h1["id"], "12:00"
            key = (source, h2["id"], ready)
            if flying and profile.get("without_transfer"):
                leg = {"compatible": True, "transfer": {"price_rub": 0}}
            else:
                if key not in second:
                    second[key] = await MCPTools.check_logistics_compatibility(
                        airport_code=source, hotel_id=h2["id"], transfer_arrival_time=ready,
                        travelers_count=travelers,
                    )
                leg = second[key]
            if not leg.get("compatible"):
                continue
            transfer = leg["transfer"]
            if not flying:
                duration = transfer["duration_min"]
                arrival = datetime.strptime(ready, "%H:%M") + timedelta(minutes=duration)
                intercity = {
                    "type": "ground", "name": transfer["name"], "operator": "Не указан в каталоге",
                    "departure": ready, "arrival": arrival.strftime("%H:%M"),
                    "duration": f"{duration} мин", "price_rub": transfer["price_rub"],
                    "description": "Наземный маршрут из графа. Отправление в 12:00 — плановое время, требует согласования с перевозчиком.",
                }
                intercity_cost = transfer["price_rub"]
                transfer = {"price_rub": 0}
            transfer_1 = logistics.get("transfer", {})
            cost = (h1["price_per_night_rub"] * nights_1 + h2["price_per_night_rub"] * nights_2) * rooms
            cost += logistics["flight"]["price_rub_roundtrip"] * travelers + intercity_cost
            cost += transfer_1.get("price_rub", 0) + transfer["price_rub"]
            if profile.get("has_pets"):
                cost += (h1.get("pet_surcharge_per_night_rub") or 0) * nights_1
                cost += (h2.get("pet_surcharge_per_night_rub") or 0) * nights_2
            if budget is not None and cost + (nights_1 + nights_2) * 5500 > budget:
                continue
            score = h1["ltr_score"] + h2["ltr_score"]
            key = (cost, -score) if policy == "budget" else (-score, cost)
            choices.append((key, {"hotel_1": h1, "hotel_2": h2, "logistics": logistics,
                                  "intercity": intercity, "intercity_cost": intercity_cost,
                                  "transfer_1": transfer_1, "transfer_2": transfer}))
    return min(choices, key=lambda c: c[0])[1] if choices else None
