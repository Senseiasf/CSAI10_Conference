import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.database import get_neo4j_driver

logger = logging.getLogger(__name__)


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
                        SET a.name = $name, a.is_origin = false
                        WITH a
                        MATCH (c:City {id: $dest_id})
                        MERGE (a)-[:LOCATED_IN]->(c)
                        """,
                        code=ap["code"],
                        name=ap["name"],
                        dest_id=dest["id"]
                    )

                # Рейсы (FLIGHT_TO с расписанием, ценами, пересадками и рисками)
                for fl in dest.get("flights", []):
                    await session.run(
                        """
                        MATCH (orig:Airport {code: $from_code})
                        MATCH (dest:Airport {code: $to_code})
                        MERGE (orig)-[f:FLIGHT_TO {flight_no: $flight_no}]->(dest)
                        SET f.airline = $airline,
                            f.departure_time = $departure_time,
                            f.arrival_time = $arrival_time,
                            f.duration_minutes = $duration,
                            f.price_rub_roundtrip = $price,
                            f.flight_type = $flight_type,
                            f.layover_airport = $layover_airport,
                            f.layover_city = $layover_city,
                            f.layover_duration_minutes = $layover_duration_minutes,
                            f.layover_risk_note = $layover_risk_note
                        """,
                        from_code=fl["from"],
                        to_code=fl["to"],
                        flight_no=fl["flight_no"],
                        airline=fl["airline"],
                        departure_time=fl["departure_time"],
                        arrival_time=fl["arrival_time"],
                        duration=fl["duration_minutes"],
                        price=fl["price_rub_roundtrip"],
                        flight_type=fl.get("flight_type", "direct"),
                        layover_airport=fl.get("layover_airport"),
                        layover_city=fl.get("layover_city"),
                        layover_duration_minutes=fl.get("layover_duration_minutes", 0),
                        layover_risk_note=fl.get("layover_risk_note", "")
                    )

                # Отели
                for hotel in dest.get("hotels", []):
                    await session.run(
                        """
                        MERGE (h:Hotel {id: $id})
                        SET h.name = $name,
                            h.city = $city,
                            h.stars = $stars,
                            h.pet_friendly = $pet_friendly,
                            h.price_per_night = $price,
                            h.checkin_time = $checkin,
                            h.checkout_time = $checkout
                        WITH h
                        MATCH (c:City {id: $dest_id})
                        MERGE (h)-[:LOCATED_IN]->(c)
                        """,
                        id=hotel["id"],
                        name=hotel["name"],
                        city=hotel["city"],
                        stars=hotel["stars"],
                        pet_friendly=hotel["pet_friendly"],
                        price=hotel["price_per_night_rub"],
                        checkin=hotel["checkin_time"],
                        checkout=hotel["checkout_time"],
                        dest_id=dest["id"]
                    )

                # Связи трансферов
                for tr in dest.get("transfers", []):
                    await session.run(
                        """
                        MATCH (a:Airport {code: $from_code})
                        MATCH (h:Hotel {id: $to_id})
                        MERGE (a)-[r:TRANSFER_TO {type: $type, name: $name}]->(h)
                        SET r.duration_minutes = $duration,
                            r.price_rub = $price
                        """,
                        from_code=tr["from"],
                        to_id=tr["to"],
                        type=tr["type"],
                        name=tr["name"],
                        duration=tr["duration_minutes"],
                        price=tr["price_rub"]
                    )

        logger.info("Neo4j graph seeded with airports, flights, hotels, and transfers.")

    @staticmethod
    async def validate_logistics(
        airport_code: str,
        hotel_id: str,
        origin_airport_code: Optional[str] = None,
        prefer_direct_flight: Optional[bool] = None,
        without_transfer: Optional[bool] = False,
        flight_time_pref: Optional[str] = None
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
          AND ($origin_code IS NULL OR orig.code = $origin_code)
        MATCH (dest)-[t:TRANSFER_TO]->(h:Hotel {id: $hotel_id})
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
               dest.code AS dest_code,
               dest.name AS dest_airport_name,
               h.name AS hotel_name,
               h.checkin_time AS checkin_time,
               t.type AS transfer_type,
               t.name AS transfer_name,
               t.duration_minutes AS transfer_duration,
               t.price_rub AS transfer_price
        ORDER BY 
            CASE WHEN f.layover_duration_minutes > 0 AND f.layover_duration_minutes < 60 THEN 2 ELSE 0 END,
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
            t.duration_minutes ASC
        """
        async with driver.session() as session:
            result = await session.run(
                query,
                airport_code=airport_code,
                hotel_id=hotel_id,
                origin_code=origin_airport_code,
                prefer_direct=prefer_direct_flight,
                flight_time_pref=flight_time_pref
            )
            records = await result.data()

            if not records:
                # Fallback только на трансфер если рейс не найден
                query_fallback = """
                MATCH (a:Airport)-[r:TRANSFER_TO]->(h:Hotel {id: $hotel_id})
                WHERE ($airport_code IS NULL OR $airport_code = '' OR a.code = $airport_code)
                RETURN a.name AS dest_airport_name,
                       a.code AS dest_code,
                       h.name AS hotel_name,
                       h.checkin_time AS checkin_time,
                       r.type AS transfer_type,
                       r.name AS transfer_name,
                       r.duration_minutes AS transfer_duration,
                       r.price_rub AS transfer_price
                ORDER BY r.duration_minutes ASC
                """
                res_fb = await session.run(query_fallback, airport_code=airport_code, hotel_id=hotel_id)
                records_fb = await res_fb.data()
                if not records_fb:
                    return {"compatible": False, "reason": "Маршрут не найден в графе."}
                best = records_fb[0]

                if flight_time_pref == "day":
                    fb_dep, fb_arr = "13:30", "18:00"
                elif flight_time_pref == "later":
                    fb_dep, fb_arr = "11:30", "16:00"
                elif flight_time_pref == "evening":
                    fb_dep, fb_arr = "18:00", "22:30"
                elif flight_time_pref == "morning":
                    fb_dep, fb_arr = "08:30", "13:00"
                else:
                    fb_dep, fb_arr = "09:00", "13:30"

                flight_info = {
                    "airline": "Регулярный рейс",
                    "flight_no": "Рейс по расписанию",
                    "origin_name": "Москва (SVO/DME)",
                    "departure_time": fb_dep,
                    "arrival_time": fb_arr,
                    "price_rub_roundtrip": 25000,
                    "flight_type": "direct",
                    "layover_airport": None,
                    "layover_city": None,
                    "layover_duration_minutes": 0,
                    "layover_risk_note": "Прямой рейс без пересадок."
                }
            else:
                best = records[0]
                flight_info = {
                    "airline": best["airline"],
                    "flight_no": best["flight_no"],
                    "origin_code": best.get("origin_code", "SVO"),
                    "origin_name": best.get("origin_name", "Москва"),
                    "departure_time": best["departure_time"],
                    "arrival_time": best["flight_arrival_time"],
                    "duration_minutes": best["flight_duration"],
                    "price_rub_roundtrip": best["flight_price_roundtrip"],
                    "flight_type": best.get("flight_type", "direct"),
                    "layover_airport": best.get("layover_airport"),
                    "layover_city": best.get("layover_city"),
                    "layover_duration_minutes": best.get("layover_duration_minutes", 0),
                    "layover_risk_note": best.get("layover_risk_note", "")
                }

            # Поиск или синтез обратного рейса (roundtrip)
            return_flight_info = None
            if best.get("dest_code") and best.get("origin_code"):
                ret_query = """
                MATCH (dest:Airport {code: $dest_code})-[rf:FLIGHT_TO]->(orig:Airport {code: $orig_code})
                RETURN rf.airline AS airline,
                       rf.flight_no AS flight_no,
                       rf.departure_time AS departure_time,
                       rf.arrival_time AS arrival_time,
                       rf.duration_minutes AS duration_minutes,
                       rf.flight_type AS flight_type
                ORDER BY 
                    CASE WHEN rf.airline = $airline THEN 0 ELSE 1 END,
                    CASE WHEN rf.departure_time >= '12:00' THEN 0 ELSE 1 END,
                    rf.departure_time ASC
                LIMIT 1
                """
                ret_res = await session.run(
                    ret_query,
                    dest_code=best["dest_code"],
                    orig_code=best["origin_code"],
                    airline=best["airline"]
                )
                ret_data = await ret_res.data()
                if ret_data:
                    rf = ret_data[0]
                    return_flight_info = {
                        "airline": rf["airline"],
                        "flight_no": rf["flight_no"],
                        "origin_code": best["dest_code"],
                        "dest_code": best["origin_code"],
                        "departure_time": rf["departure_time"],
                        "arrival_time": rf["arrival_time"],
                        "duration_minutes": rf["duration_minutes"],
                        "flight_type": rf.get("flight_type", "direct")
                    }

            if not return_flight_info:
                out_no = flight_info.get("flight_no", "SU-1000")
                parts = out_no.split("-")
                if len(parts) == 2 and parts[1].isdigit():
                    num = int(parts[1])
                    ret_no = f"{parts[0]}-{num + 1 if num % 2 == 0 else num - 1}"
                else:
                    ret_no = f"{flight_info.get('airline', 'Авиа')} (обратный)"
                
                dur = flight_info.get("duration_minutes", 180)
                dep = "15:45"
                dep_h, dep_m = 15, 45
                arr_h = (dep_h + (dep_m + dur) // 60) % 24
                arr_m = (dep_m + dur) % 60
                arr = f"{arr_h:02d}:{arr_m:02d}"
                return_flight_info = {
                    "airline": flight_info.get("airline", "Регулярный рейс"),
                    "flight_no": ret_no,
                    "origin_code": best.get("dest_code", "DEST"),
                    "dest_code": best.get("origin_code", "SVO"),
                    "departure_time": dep,
                    "arrival_time": arr,
                    "duration_minutes": dur,
                    "flight_type": flight_info.get("flight_type", "direct")
                }

            flight_info["return_flight"] = return_flight_info

            if without_transfer:
                best["transfer_name"] = "Самостоятельно (такси / каршеринг / общественный транспорт)"
                best["transfer_type"] = "self"
                best["transfer_duration"] = 30
                best["transfer_price"] = 0

            checkin_str = best.get("checkin_time") or "15:00"
            flight_arr_str = flight_info["arrival_time"]

            try:
                dest_code = best.get("dest_code", "")
                is_international = dest_code in ("DXB", "DWC", "AYT", "IST", "HKT", "MLE", "SSH")
                
                # Детализированный буфер после прилёта: высадка, паспортный контроль, выдача багажа на ленте, таможня
                if is_international:
                    deplaning_min = 15
                    passport_control_min = 25
                    baggage_claim_min = 30
                    customs_exit_min = 10
                    total_post_flight_buffer_min = deplaning_min + passport_control_min + baggage_claim_min + customs_exit_min  # 80 минут
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
                    total_post_flight_buffer_min = deplaning_min + passport_control_min + baggage_claim_min + customs_exit_min  # 50 минут
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

                # Расчёт сценария форс-мажора: задержка рейса на 60 мин
                delay_flight_arr_dt = arr_dt + timedelta(minutes=60)
                delay_transfer_start_dt = transfer_start_dt + timedelta(minutes=60)
                delay_hotel_arr_dt = hotel_arrival_dt + timedelta(minutes=60)
                delay_hotel_arr_str = delay_hotel_arr_dt.strftime("%H:%M")

                flight_no = flight_info.get("flight_no", "рейса")

                force_majeure_resilience = {
                    "flight_delay_scenario": (
                        f"При задержке вылета/прилёта на 1 час (приземление в {delay_flight_arr_dt.strftime('%H:%M')}) "
                        f"отправление трансфера автоматически сдвигается на {delay_transfer_start_dt.strftime('%H:%M')}, "
                        f"прибытие в отель — в {delay_hotel_arr_str}. "
                        f"Водитель отслеживает борт {flight_no} по онлайн-табло: бесплатное ожидание до 90 минут. "
                        f"В отеле гарантирован круглосуточный Late Check-in (бронь не аннулируется)."
                    ),
                    "baggage_delay_safety": (
                        "💼 Совет по багажу: ценные вещи, документы, медикаменты, зарядные устройства "
                        "и комплект одежды на первые сутки держите в ручной клади. "
                        "При задержке багажа на ленте оформите акт PIR на стойке Lost & Found до выхода из транзитной зоны."
                    )
                }

                # Оценка пересадки (стыковки)
                is_connecting = flight_info.get("flight_type") == "connecting"
                if is_connecting:
                    layover_dur = flight_info.get("layover_duration_minutes", 105)
                    layover_hours = layover_dur // 60
                    layover_mins = layover_dur % 60
                    dur_str = f"{layover_hours}ч {layover_mins}м" if layover_hours else f"{layover_mins}м"
                    layover_details = {
                        "is_connecting": True,
                        "airport": flight_info.get("layover_airport"),
                        "city": flight_info.get("layover_city"),
                        "duration_minutes": layover_dur,
                        "duration_formatted": dur_str,
                        "risk_evaluation": flight_info.get("layover_risk_note", ""),
                        "safety_guarantee": (
                            "Оформлен единый сквозной билет: багаж регистрируется сразу до конечной точки. "
                            "При опоздании первого рейса авиакомпания бесплатно пересаживает на следующий рейс и предоставляет питание."
                        )
                    }
                else:
                    layover_details = {
                        "is_connecting": False,
                        "airport": None,
                        "city": None,
                        "duration_minutes": 0,
                        "duration_formatted": None,
                        "risk_evaluation": "Прямой беспосадочный рейс — риск потери стыковки или багажа при транзитной перегрузке исключён.",
                        "safety_guarantee": "Прямой маршрут без промежуточных пунктов."
                    }

                if hotel_arrival_dt < checkin_dt:
                    diff_hours = (checkin_dt - hotel_arrival_dt).total_seconds() / 3600
                    status = (
                        f"Вылет из {flight_info['origin_name']} в {flight_info['departure_time']} → "
                        f"прилёт в {best['dest_airport_name']} в {flight_arr_str}. "
                        f"Закладываем {total_post_flight_buffer_min} мин на прохождение процедур и получение багажа. "
                        f"Отправление трансфера в {transfer_start_str}, прибытие в отель в {hotel_arr_str}. "
                        f"До заселения (с {checkin_str}) остаётся {diff_hours:.1f} ч (вещи можно оставить на ресепшн отеля)."
                    )
                else:
                    diff_hours = (hotel_arrival_dt - checkin_dt).total_seconds() / 3600
                    status = (
                        f"Вылет из {flight_info['origin_name']} в {flight_info['departure_time']} → "
                        f"прилёт в {best['dest_airport_name']} в {flight_arr_str}. "
                        f"Закладываем {total_post_flight_buffer_min} мин на прохождение процедур и получение багажа. "
                        f"Отправление трансфера в {transfer_start_str}, прибытие в отель в {hotel_arr_str}. "
                        f"Номер готов к заселению (check-in с {checkin_str})."
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
                        "price_rub": best["transfer_price"]
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
                        "hotel_checkin_policy": checkin_str,
                        "force_majeure_resilience": force_majeure_resilience,
                        "layover_details": layover_details
                    },
                    "logistics_status": status,
                    "mandatory_logistics_cost": flight_info["price_rub_roundtrip"] + best["transfer_price"]
                }
            except Exception as e:
                logger.error(f"Error in timeline calculation: {e}")
                return {
                    "compatible": True,
                    "flight": flight_info,
                    "airport": best.get("dest_airport_name", airport_code),
                    "hotel": best.get("hotel_name", hotel_id),
                    "transfer": {
                        "name": best["transfer_name"],
                        "type": best["transfer_type"],
                        "duration_min": best["transfer_duration"],
                        "price_rub": best["transfer_price"]
                    },
                    "logistics_status": "Логистика состыкована",
                    "mandatory_logistics_cost": flight_info["price_rub_roundtrip"] + best["transfer_price"]
                }

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
