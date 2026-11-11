import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from app.database import get_neo4j_driver

logger = logging.getLogger(__name__)

MIN_LAYOVER_MINUTES = 60
INTERNATIONAL_AIRPORTS = {
    "DXB", "DWC", "AUH", "AYT", "IST", "GZP", "DLM", "HKT", "BKK",
    "USM", "URT", "MLE", "SSH", "HRG", "TBS", "BUS",
}

REJECTION_MESSAGES = {
    "no_outbound_route": "В графе нет подходящего рейса до отеля.",
    "no_return_flight": "В графе нет подходящего обратного рейса.",
    "invalid_timeline": "Данных каталога недостаточно для расчёта времени маршрута.",
    "transfer_capacity": "В трансфере недостаточно мест.",
    "seaplane_closed": "Путешественники не успевают к отправлению гидросамолёта.",
    "bridge_closed": "Мост закрыт в расчётное время проезда.",
}


def minutes_since_midnight(clock: str) -> int:
    parsed = datetime.strptime(clock, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def post_flight_buffer(airport_code: str) -> int:
    return 80 if airport_code in INTERNATIONAL_AIRPORTS else 50


def transfer_rejection(candidate: dict, ready_minutes: int, travelers_count: int, strict: bool) -> str | None:
    capacity = candidate.get("transfer_capacity")
    if capacity is not None and capacity < travelers_count:
        return "transfer_capacity"
    if not strict:
        return None
    if candidate["dest_code"] == "MLE" and candidate["transfer_type"] in {"seaplane", "vip_seaplane"}:
        if ready_minutes > 16 * 60 + 30:
            return "seaplane_closed"
    crossing = candidate.get("bridge_crossing_minutes")
    if crossing is not None:
        crossing_time = (ready_minutes + crossing) % (24 * 60)
        closed_from = minutes_since_midnight(candidate["bridge_closed_from"])
        closed_until = minutes_since_midnight(candidate["bridge_closed_until"])
        if (crossing_time - closed_from) % (24 * 60) < (closed_until - closed_from) % (24 * 60):
            return "bridge_closed"
    return None


def rejection(code: str) -> dict:
    return {"compatible": False, "reason_code": code, "reason": REJECTION_MESSAGES[code]}


class GraphService:
    @staticmethod
    async def init_schema():
        """Создает индексы и ограничения уникальности в Neo4j."""
        driver = get_neo4j_driver()
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Airport) REQUIRE a.code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.id IS UNIQUE"
        ]
        async with driver.session() as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:
                    logger.warning(f"Neo4j constraint creation notice: {e}")

    @staticmethod
    async def seed_graph(data: Dict[str, Any]):
        """
        Заполняет Neo4j графом логистики:
        - Аэропорты вылета (Москва SVO, VKO, DME)
        - Аэропорты прибытия
        - Рейсы (:Airport)-[:FLIGHT_TO]->(:Airport) с ценами туда-обратно и расписанием
        - Отели (:Hotel)
        - Трансферы (:Airport)-[:TRANSFER_TO]->(:Hotel)
        """
        from app.services.catalog_validation import validate_catalog
        validate_catalog(data)
        driver = get_neo4j_driver()
        await GraphService.init_schema()

        destinations_data = data.get("destinations", [])
        origin_airports = data.get("origin_airports", [])

        async with driver.session() as session:
            # 1. Аэропорты вылета (Origin Airports)
            for orig in origin_airports:
                await session.run(
                    """
                    MERGE (a:Airport {code: $code})
                    SET a.name = $name, a.city = $city, a.is_origin = true
                    """,
                    code=orig["code"],
                    name=orig["name"],
                    city=orig["city"]
                )

            # 2. Направления, аэропорты назначения, рейсы, отели и трансферы
            for dest in destinations_data:
                # Город / регион
                await session.run(
                    """
                    MERGE (c:City {id: $id})
                    SET c.name = $name, c.country = $country
                    """,
                    id=dest["id"],
                    name=dest["name"],
                    country=dest["country"]
                )

                # Аэропорты назначения
                for ap in dest.get("airports", []):
                    await session.run(
                        """
                        MERGE (a:Airport {code: $code})
                        ON CREATE SET a.is_origin = false
                        SET a.name = $name
                        WITH a
                        MATCH (c:City {id: $dest_id})
                        MERGE (a)-[:LOCATED_IN]->(c)
                        """,
                        code=ap["code"],
                        name=ap["name"],
                        dest_id=dest["id"]
                    )

                # Отели (батчевая вставка через UNWIND)
                hotels_batch = [
                    {
                        "id": hotel["id"],
                        "name": hotel["name"],
                        "city": hotel["city"],
                        "stars": hotel["stars"],
                        "pet_friendly": hotel["pet_friendly"],
                        "price": hotel["price_per_night_rub"],
                        "checkin": hotel["checkin_time"],
                        "checkout": hotel["checkout_time"],
                        "dest_id": dest["id"]
                    }
                    for hotel in dest.get("hotels", [])
                ]
                if hotels_batch:
                    await session.run(
                        """
                        UNWIND $batch AS item
                        MERGE (h:Hotel {id: item.id})
                        SET h.name = item.name,
                            h.city = item.city,
                            h.stars = item.stars,
                            h.pet_friendly = item.pet_friendly,
                            h.price_per_night = item.price,
                            h.checkin_time = item.checkin,
                            h.checkout_time = item.checkout
                        WITH h, item
                        MATCH (c:City {id: item.dest_id})
                        MERGE (h)-[:LOCATED_IN]->(c)
                        """,
                        batch=hotels_batch
                    )

            # Связи создаются после загрузки всех аэропортов и отелей.
            for dest in destinations_data:
                # Рейсы (FLIGHT_TO с расписанием, ценами, пересадками, пунктуальностью и типами ВС - батчевая вставка)
                flights_batch = [
                    {
                        "from_code": fl["from"],
                        "to_code": fl["to"],
                        "flight_no": fl["flight_no"],
                        "airline": fl["airline"],
                        "departure_time": fl["departure_time"],
                        "arrival_time": fl["arrival_time"],
                        "duration": fl["duration_minutes"],
                        "price": fl["price_rub_roundtrip"],
                        "flight_type": fl.get("flight_type"),
                        "layover_airport": fl.get("layover_airport"),
                        "layover_city": fl.get("layover_city"),
                        "layover_duration_minutes": fl.get("layover_duration_minutes"),
                        "layover_risk_note": fl.get("layover_risk_note", ""),
                        "on_time_pct": fl.get("on_time_pct"),
                        "aircraft_type": fl.get("aircraft_type"),
                        "baggage_allowance": fl.get("baggage_allowance"),
                        "meal_service": fl.get("meal_service"),
                        "seat_pitch": fl.get("seat_pitch")
                    }
                    for fl in dest.get("flights", [])
                ]
                if flights_batch:
                    await session.run(
                        """
                        UNWIND $batch AS item
                        MATCH (orig:Airport {code: item.from_code})
                        MATCH (dest:Airport {code: item.to_code})
                        MERGE (orig)-[f:FLIGHT_TO {flight_no: item.flight_no}]->(dest)
                        SET f.airline = item.airline,
                            f.departure_time = item.departure_time,
                            f.arrival_time = item.arrival_time,
                            f.duration_minutes = item.duration,
                            f.price_rub_roundtrip = item.price,
                            f.flight_type = item.flight_type,
                            f.layover_airport = item.layover_airport,
                            f.layover_city = item.layover_city,
                            f.layover_duration_minutes = item.layover_duration_minutes,
                            f.layover_risk_note = item.layover_risk_note,
                            f.on_time_pct = item.on_time_pct,
                            f.aircraft_type = item.aircraft_type,
                            f.baggage_allowance = item.baggage_allowance,
                            f.meal_service = item.meal_service,
                            f.seat_pitch = item.seat_pitch
                        """,
                        batch=flights_batch
                    )

                # Связи трансферов (батчевая вставка через UNWIND)
                airport_transfers_batch = []
                interhotel_transfers_batch = []
                for tr in dest.get("transfers", []):
                    tr_item = {
                        "from_code": tr["from"],
                        "to_id": tr["to"],
                        "type": tr["type"],
                        "name": tr["name"],
                        "duration": tr["duration_minutes"],
                        "price": tr["price_rub"],
                        "capacity": tr.get("capacity", tr.get("max_passengers")),
                        "bridge_crossing_minutes": tr.get("bridge_crossing_minutes"),
                        "bridge_closed_from": tr.get("bridge_closed_from"),
                        "bridge_closed_until": tr.get("bridge_closed_until"),
                    }
                    if len(tr.get("from", "")) == 3 and tr["from"].isupper():
                        airport_transfers_batch.append(tr_item)
                    else:
                        interhotel_transfers_batch.append(tr_item)

                if airport_transfers_batch:
                    await session.run(
                        """
                        UNWIND $batch AS item
                        MATCH (a:Airport {code: item.from_code})
                        MATCH (h:Hotel {id: item.to_id})
                        MERGE (a)-[r:TRANSFER_TO {type: item.type, name: item.name}]->(h)
                        SET r.duration_minutes = item.duration,
                            r.price_rub = item.price,
                            r.capacity = item.capacity,
                            r.bridge_crossing_minutes = item.bridge_crossing_minutes,
                            r.bridge_closed_from = item.bridge_closed_from,
                            r.bridge_closed_until = item.bridge_closed_until
                        """,
                        batch=airport_transfers_batch
                    )

                if interhotel_transfers_batch:
                    await session.run(
                        """
                        UNWIND $batch AS item
                        MATCH (h1:Hotel {id: item.from_code})
                        MATCH (h2:Hotel {id: item.to_id})
                        MERGE (h1)-[r:INTERHOTEL_TRANSFER_TO {type: item.type, name: item.name}]->(h2)
                        SET r.duration_minutes = item.duration,
                            r.price_rub = item.price,
                            r.capacity = item.capacity,
                            r.bridge_crossing_minutes = item.bridge_crossing_minutes,
                            r.bridge_closed_from = item.bridge_closed_from,
                            r.bridge_closed_until = item.bridge_closed_until
                        """,
                        batch=interhotel_transfers_batch
                    )

        logger.info("Neo4j graph seeded with airports, flights, hotels, and transfers.")

    @staticmethod
    async def find_intercity_logistics(origin_airport: str, dest_airport: str) -> Optional[Dict[str, Any]]:
        """
        Ищет в графе Neo4j прямое межгородское транспортное ребро между аэропортами/городами:
        (a1:Airport)-[f:FLIGHT_TO]->(a2:Airport)
        """
        driver = get_neo4j_driver()
        query = """
        MATCH (a1:Airport {code: $orig})-[f:FLIGHT_TO]->(a2:Airport {code: $dest})
        WHERE coalesce(f.flight_type, 'direct') = 'direct'
          AND f.price_rub_roundtrip >= 0 AND f.duration_minutes > 0
        RETURN f.airline AS airline,
               f.flight_no AS flight_no,
               f.departure_time AS departure_time,
               f.arrival_time AS arrival_time,
               f.duration_minutes AS duration_minutes,
               f.price_rub_roundtrip AS price_rub
        ORDER BY f.price_rub_roundtrip, f.duration_minutes, f.flight_no
        LIMIT 1
        """
        try:
            async with driver.session() as session:
                result = await session.run(query, orig=origin_airport, dest=dest_airport)
                record = await result.single()
                if record:
                    data = dict(record)
                    minutes_since_midnight(data["departure_time"])
                    minutes_since_midnight(data["arrival_time"])
                    return data
        except Exception as e:
            logger.warning("Intercity route lookup failed: %s", e)
        return None

    @staticmethod
    async def validate_logistics(
        airport_code: str,
        hotel_id: str,
        origin_airport_code: Optional[str] = None,
        prefer_direct_flight: Optional[bool] = None,
        without_transfer: Optional[bool] = False,
        flight_time_pref: Optional[str] = None,
        travelers_count: int = 1,
        strict_graph_validation: bool = True,
        return_airport_code: str | None = None,
    ) -> Dict[str, Any]:
        """
        Проверяет сквозную логистику от вылета до заселения:
        (Аэропорт вылета) -[:FLIGHT_TO]-> (Аэропорт прилёта) -[:TRANSFER_TO]-> (Отель)
        Рассчитывает время прилёта, детальный буфер после посадки, пересадки и стыковки,
        обратный рейс, время трансфера, расчётное прибытие в отель и статус готовности номера.
        """
        driver = get_neo4j_driver()
        query = """
        MATCH (orig:Airport)-[f:FLIGHT_TO]->(dest:Airport)
        WHERE ($airport_code IS NULL OR $airport_code = '' OR dest.code = $airport_code)
          AND (($origin_code IS NULL AND orig.is_origin = true) OR orig.code = $origin_code)
          AND (NOT $strict OR $prefer_direct IS NULL OR $prefer_direct = false OR f.flight_type = 'direct')
          AND (NOT $strict OR f.flight_type = 'direct'
               OR (f.flight_type = 'connecting' AND f.layover_duration_minutes >= $min_layover))
        MATCH (h:Hotel {id: $hotel_id})
        OPTIONAL MATCH (dest)-[t:TRANSFER_TO]->(h)
        WITH orig, dest, f, h, t
        WHERE t IS NOT NULL OR ($without_transfer AND EXISTS {
            MATCH (dest)-[:LOCATED_IN]->(:City)<-[:LOCATED_IN]-(h)
        })
        RETURN orig.code AS origin_code,
               orig.name AS origin_name,
               f.airline AS airline,
               f.flight_no AS flight_no,
               f.departure_time AS departure_time,
               f.arrival_time AS flight_arrival_time,
               f.duration_minutes AS flight_duration,
               f.price_rub_roundtrip AS flight_price_roundtrip,
               f.flight_type AS flight_type,
               f.layover_airport AS layover_airport,
               f.layover_city AS layover_city,
               f.layover_duration_minutes AS layover_duration_minutes,
               f.layover_risk_note AS layover_risk_note,
               f.on_time_pct AS on_time_pct,
               f.aircraft_type AS aircraft_type,
               f.baggage_allowance AS baggage_allowance,
               f.meal_service AS meal_service,
               f.seat_pitch AS seat_pitch,
               dest.code AS dest_code,
               dest.name AS dest_airport_name,
               h.name AS hotel_name,
               h.checkin_time AS checkin_time,
               t.type AS transfer_type,
               t.name AS transfer_name,
               t.duration_minutes AS transfer_duration,
               t.price_rub AS transfer_price,
               t.capacity AS transfer_capacity,
               t.bridge_crossing_minutes AS bridge_crossing_minutes,
               t.bridge_closed_from AS bridge_closed_from,
               t.bridge_closed_until AS bridge_closed_until
        ORDER BY
            CASE 
                WHEN $flight_time_pref = 'day' THEN (
                    CASE 
                        WHEN f.departure_time >= '11:00' AND f.departure_time < '17:00' THEN 0 
                        WHEN f.departure_time >= '10:00' AND f.departure_time < '19:00' THEN 1 
                        WHEN f.departure_time >= '08:00' THEN 3
                        ELSE 10 
                    END
                )
                WHEN $flight_time_pref = 'later' THEN (
                    CASE 
                        WHEN f.departure_time >= '11:00' THEN 0 
                        WHEN f.departure_time >= '09:30' THEN 1 
                        WHEN f.departure_time >= '08:00' THEN 3
                        ELSE 10 
                    END
                )
                WHEN $flight_time_pref = 'evening' THEN (
                    CASE 
                        WHEN f.departure_time >= '17:00' THEN 0 
                        WHEN f.departure_time >= '14:00' THEN 1 
                        ELSE 10 
                    END
                )
                WHEN $flight_time_pref = 'morning' THEN (
                    CASE 
                        WHEN f.departure_time >= '07:30' AND f.departure_time < '11:30' THEN 0 
                        WHEN f.departure_time < '12:00' THEN 1 
                        ELSE 10 
                    END
                )
                ELSE 0
            END,
            CASE 
                WHEN $prefer_direct = true THEN (CASE WHEN f.flight_type = 'direct' THEN 0 ELSE 1 END)
                WHEN $prefer_direct = false THEN (CASE WHEN f.flight_type = 'connecting' THEN 0 ELSE 1 END)
                ELSE 0
            END,
            f.price_rub_roundtrip ASC, 
            t.duration_minutes ASC, f.flight_no, t.name
        """
        async with driver.session() as session:
            result = await session.run(
                query,
                airport_code=airport_code,
                hotel_id=hotel_id,
                origin_code=origin_airport_code,
                prefer_direct=prefer_direct_flight,
                flight_time_pref=flight_time_pref,
                without_transfer=bool(without_transfer),
                strict=strict_graph_validation,
                min_layover=MIN_LAYOVER_MINUTES,
            )
            returns: dict[tuple[str, str, str], dict | None] = {}
            failure = "no_outbound_route"
            for candidate in await result.data():
                try:
                    arrival = minutes_since_midnight(candidate["flight_arrival_time"])
                    minutes_since_midnight(candidate["departure_time"])
                    if candidate["flight_price_roundtrip"] < 0 or candidate["flight_duration"] <= 0:
                        raise ValueError("invalid flight duration or price")
                    if not without_transfer:
                        minutes_since_midnight(candidate["checkin_time"])
                        if candidate["transfer_duration"] < 0 or candidate["transfer_price"] < 0:
                            raise ValueError("invalid transfer duration or price")
                        buffer = post_flight_buffer(candidate["dest_code"])
                        transfer_failure = transfer_rejection(
                            candidate, arrival + buffer, travelers_count, strict_graph_validation
                        )
                        if transfer_failure:
                            failure = transfer_failure
                            continue
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Invalid outbound schedule or transfer: %s", exc)
                    failure = "invalid_timeline"
                    continue

                return_key = (return_airport_code or candidate["dest_code"], candidate["origin_code"], candidate["airline"])
                if return_key not in returns:
                    returns[return_key] = await GraphService._find_return_flight(
                        session, *return_key, strict_graph_validation, bool(prefer_direct_flight)
                    )
                return_flight = returns[return_key]
                if return_flight is None:
                    failure = "no_return_flight"
                    continue
                best = candidate
                break
            else:
                return rejection(failure)

            flight_info = {
                "airline": best["airline"],
                "flight_no": best["flight_no"],
                "origin_name": best["origin_name"],
                "origin_code": best["origin_code"],
                "departure_time": best["departure_time"],
                "arrival_time": best["flight_arrival_time"],
                "duration_minutes": best["flight_duration"],
                "price_rub_roundtrip": best["flight_price_roundtrip"],
                "flight_type": best["flight_type"],
                "layover_airport": best.get("layover_airport"),
                "layover_city": best.get("layover_city"),
                "layover_duration_minutes": best.get("layover_duration_minutes"),
                "layover_risk_note": best.get("layover_risk_note", ""),
                "on_time_pct": best.get("on_time_pct"),
                "aircraft_type": best.get("aircraft_type"),
                "baggage_allowance": best.get("baggage_allowance"),
                "meal_service": best.get("meal_service"),
                "seat_pitch": best.get("seat_pitch"),
                "return_flight": return_flight,
            }
            if without_transfer:
                return {
                    "compatible": True,
                    "flight": flight_info,
                    "airport": best["dest_airport_name"],
                    "hotel": best["hotel_name"],
                    "transfer": {"type": "self", "name": "Самостоятельный проезд", "price_rub": 0},
                    "logistics_status": "Перелёты найдены. Проезд до отеля путешественник организует самостоятельно.",
                    "mandatory_logistics_cost": flight_info["price_rub_roundtrip"],
                }

            checkin_str = best.get("checkin_time") or "15:00"
            flight_arr_str = flight_info["arrival_time"]

            try:
                is_international = best["dest_code"] in INTERNATIONAL_AIRPORTS

                # Детализированный буфер после прилёта: высадка, паспортный контроль, выдача багажа на ленте, таможня
                if is_international:
                    deplaning_min = 15
                    passport_control_min = 25
                    baggage_claim_min = 30
                    customs_exit_min = 10
                    total_post_flight_buffer_min = post_flight_buffer(best["dest_code"])
                    buffer_desc = (
                        f"Международный буфер {total_post_flight_buffer_min} мин "
                        f"(руление и высадка 15 мин ➔ паспортный контроль 25 мин ➔ выдача багажа на ленте 30 мин ➔ таможня и выход к трансферу 10 мин)"
                    )
                    buffer_breakdown = {
                        "deplaning_min": deplaning_min,
                        "passport_control_min": passport_control_min,
                        "baggage_claim_min": baggage_claim_min,
                        "customs_exit_min": customs_exit_min,
                        "total_buffer_min": total_post_flight_buffer_min,
                        "summary": "Высадка (15м) + Граница (25м) + Багажная лента (30м) + Выход к трансферу (10м)"
                    }
                else:
                    deplaning_min = 15
                    passport_control_min = 0
                    baggage_claim_min = 25
                    customs_exit_min = 10
                    total_post_flight_buffer_min = post_flight_buffer(best["dest_code"])
                    buffer_desc = (
                        f"Внутренний буфер {total_post_flight_buffer_min} мин "
                        f"(высадка и выход в терминал 15 мин ➔ выдача багажа на карусели 25 мин ➔ выход к автостоянке трансфера 10 мин)"
                    )
                    buffer_breakdown = {
                        "deplaning_min": deplaning_min,
                        "passport_control_min": 0,
                        "baggage_claim_min": baggage_claim_min,
                        "customs_exit_min": customs_exit_min,
                        "total_buffer_min": total_post_flight_buffer_min,
                        "summary": "Высадка (15м) + Багажная карусель (25м) + Выход к автомобилю (10м)"
                    }

                arr_dt = datetime.strptime(flight_arr_str, "%H:%M")
                transfer_start_dt = arr_dt + timedelta(minutes=total_post_flight_buffer_min)
                hotel_arrival_dt = transfer_start_dt + timedelta(minutes=best["transfer_duration"])
                checkin_dt = datetime.strptime(checkin_str, "%H:%M")

                hotel_arr_str = hotel_arrival_dt.strftime("%H:%M")
                transfer_start_str = transfer_start_dt.strftime("%H:%M")

                # Оценка пересадки (стыковки)
                is_connecting = flight_info.get("flight_type") == "connecting"
                if is_connecting:
                    layover_dur = flight_info["layover_duration_minutes"]
                    dur_str = f"{layover_dur // 60}ч {layover_dur % 60}м" if layover_dur is not None else None
                    layover_details = {
                        "is_connecting": True,
                        "airport": flight_info.get("layover_airport"),
                        "city": flight_info.get("layover_city"),
                        "duration_minutes": layover_dur,
                        "duration_formatted": dur_str,
                        "risk_evaluation": flight_info.get("layover_risk_note", ""),
                    }
                else:
                    layover_details = {
                        "is_connecting": False,
                        "airport": None,
                        "city": None,
                        "duration_minutes": 0,
                        "duration_formatted": None,
                        "risk_evaluation": "Прямой рейс; пересадка не предусмотрена.",
                    }

                if hotel_arrival_dt < checkin_dt:
                    diff_hours = (checkin_dt - hotel_arrival_dt).total_seconds() / 3600
                    status = (
                        f"Вылет из {flight_info['origin_name']} в {flight_info['departure_time']} → "
                        f"прилёт в {best['dest_airport_name']} в {flight_arr_str}. "
                        f"Закладываем {total_post_flight_buffer_min} мин на прохождение процедур и получение багажа. "
                        f"Отправление трансфера в {transfer_start_str}, прибытие в отель в {hotel_arr_str}. "
                        f"До заселения (с {checkin_str}) остаётся {diff_hours:.1f} ч."
                    )
                else:
                    diff_hours = (hotel_arrival_dt - checkin_dt).total_seconds() / 3600
                    status = (
                        f"Вылет из {flight_info['origin_name']} в {flight_info['departure_time']} → "
                        f"прилёт в {best['dest_airport_name']} в {flight_arr_str}. "
                        f"Закладываем {total_post_flight_buffer_min} мин на прохождение процедур и получение багажа. "
                        f"Отправление трансфера в {transfer_start_str}, прибытие в отель в {hotel_arr_str}. "
                        f"Прибытие после начала заселения (check-in с {checkin_str})."
                    )

                return {
                    "compatible": True,
                    "flight": flight_info,
                    "airport": best["dest_airport_name"],
                    "hotel": best["hotel_name"],
                    "transfer": {
                        "name": best["transfer_name"],
                        "type": best["transfer_type"],
                        "duration_min": best["transfer_duration"],
                        "price_rub": best["transfer_price"],
                        "capacity": best.get("transfer_capacity"),
                        "capacity_status": "known" if best.get("transfer_capacity") is not None else "unknown",
                    },
                    "timeline": {
                        "flight_departure": flight_info["departure_time"],
                        "flight_arrival": flight_arr_str,
                        "is_international": is_international,
                        "post_flight_buffer_min": total_post_flight_buffer_min,
                        "buffer_breakdown": buffer_breakdown,
                        "buffer_description": buffer_desc,
                        "transfer_start": transfer_start_str,
                        "hotel_arrival": hotel_arr_str,
                        "hotel_arrival_day_offset": (hotel_arrival_dt.date() - arr_dt.date()).days,
                        "hotel_checkin_policy": checkin_str,
                        "checkin_wait_minutes": max(0, int((checkin_dt - hotel_arrival_dt).total_seconds() / 60)),
                        "layover_details": layover_details
                    },
                    "logistics_status": status,
                    "mandatory_logistics_cost": flight_info["price_rub_roundtrip"] + best["transfer_price"]
                }
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Invalid route timeline: %s", exc)
                return rejection("invalid_timeline")

    @staticmethod
    async def _find_return_flight(
        session, origin: str, destination: str, airline: str, strict: bool, direct_only: bool,
    ) -> dict | None:
        result = await session.run(
            """
            MATCH (:Airport {code: $origin})-[f:FLIGHT_TO]->(:Airport {code: $destination})
            WHERE (NOT $strict OR f.flight_type = 'direct'
                OR (f.flight_type = 'connecting' AND f.layover_duration_minutes >= $min_layover))
              AND (NOT $strict OR NOT $direct_only OR f.flight_type = 'direct')
            RETURN f.airline AS airline, f.flight_no AS flight_no,
                   f.departure_time AS departure_time, f.arrival_time AS arrival_time,
                   f.duration_minutes AS duration_minutes, f.flight_type AS flight_type,
                   f.layover_duration_minutes AS layover_duration_minutes,
                   f.on_time_pct AS on_time_pct, f.aircraft_type AS aircraft_type,
                   f.baggage_allowance AS baggage_allowance, f.meal_service AS meal_service,
                   f.seat_pitch AS seat_pitch
            ORDER BY CASE WHEN f.airline = $airline THEN 0 ELSE 1 END,
                     CASE WHEN f.departure_time >= '12:00' THEN 0 ELSE 1 END,
                     f.departure_time, f.flight_no
            """,
            origin=origin,
            destination=destination,
            airline=airline,
            strict=strict,
            direct_only=direct_only,
            min_layover=MIN_LAYOVER_MINUTES,
        )
        for row in await result.data():
            try:
                minutes_since_midnight(row["departure_time"])
                minutes_since_midnight(row["arrival_time"])
                if row["duration_minutes"] > 0:
                    return {**row, "origin_code": origin, "dest_code": destination}
            except (KeyError, TypeError, ValueError):
                continue
        return None


    @staticmethod
    async def validate_transfer(airport_code: str, hotel_id: str, arrival_time: str,
                                travelers_count: int = 1) -> dict:
        """Validate the ground leg after arrival of an intercity service."""
        try:
            ready = minutes_since_midnight(arrival_time)
        except (ValueError, TypeError):
            return rejection("invalid_timeline")
        async with get_neo4j_driver().session() as session:
            result = await session.run(
                """
                MATCH (start)-[t:TRANSFER_TO|INTERHOTEL_TRANSFER_TO]->(:Hotel {id: $hotel})
                WHERE start.code = $airport OR start.id = $airport
                RETURN start:Airport AS is_airport, $airport AS dest_code, t.name AS transfer_name, t.type AS transfer_type,
                       t.duration_minutes AS transfer_duration, t.price_rub AS transfer_price,
                       t.capacity AS transfer_capacity,
                       t.bridge_crossing_minutes AS bridge_crossing_minutes,
                       t.bridge_closed_from AS bridge_closed_from, t.bridge_closed_until AS bridge_closed_until
                ORDER BY t.price_rub, t.duration_minutes, t.name
                """, airport=airport_code, hotel=hotel_id,
            )
            for row in await result.data():
                try:
                    if row["transfer_price"] < 0 or row["transfer_duration"] < 0:
                        continue
                    if transfer_rejection(row, ready + (post_flight_buffer(airport_code) if row["is_airport"] else 0), travelers_count, True):
                        continue
                except (KeyError, ValueError, TypeError):
                    continue
                return {"compatible": True, "transfer": {
                    "name": row["transfer_name"], "type": row["transfer_type"],
                    "price_rub": row["transfer_price"], "duration_min": row["transfer_duration"],
                    "capacity": row["transfer_capacity"],
                }}
        return rejection("no_outbound_route")

    @staticmethod
    async def get_graph_stats() -> Dict[str, int]:
        driver = get_neo4j_driver()
        query = """
        MATCH (c:City) WITH count(c) as cities
        MATCH (a:Airport) WITH cities, count(a) as airports
        MATCH (h:Hotel) WITH cities, airports, count(h) as hotels
        OPTIONAL MATCH ()-[r:TRANSFER_TO]->() WITH cities, airports, hotels, count(r) as transfers
        OPTIONAL MATCH ()-[f:FLIGHT_TO]->() WITH cities, airports, hotels, transfers, count(f) as flights
        RETURN cities, airports, hotels, transfers, flights
        """
        try:
            async with driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                if record:
                    return {
                        "cities": record["cities"],
                        "airports": record["airports"],
                        "hotels": record["hotels"],
                        "transfers": record["transfers"],
                        "flights": record["flights"]
                    }
        except Exception as e:
            logger.warning(f"Could not fetch graph stats: {e}")
        return {"cities": 0, "airports": 0, "hotels": 0, "transfers": 0, "flights": 0}
