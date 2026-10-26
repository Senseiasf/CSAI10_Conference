import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_flights")

DESTINATIONS_FILE = Path("backend/app/data/destinations.json")
AIRLINE_DATASET_FILE = Path("/home/mironovnp/.cache/kagglehub/datasets/iamsouravbanerjee/airline-dataset/versions/4/Airline Dataset Updated - v2.csv")

TARGET_AIRPORTS = {"SVO", "VKO", "DME", "LED", "AER", "DXB", "AUH", "AYT", "IST", "SAW", "HKT", "BKK", "MLE", "SSH", "HRG", "RGK"}


def compute_punctuality():
    logger.info("Computing airport punctuality and delay rates from Airline Dataset...")
    stats = {code: {"on_time": 0, "total": 0} for code in TARGET_AIRPORTS}

    if not AIRLINE_DATASET_FILE.exists():
        logger.warning("Airline dataset file not found: %s", AIRLINE_DATASET_FILE)
        return {code: 85.0 for code in TARGET_AIRPORTS}

    with open(AIRLINE_DATASET_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arr = row.get("Arrival Airport", "").strip()
            st = row.get("Flight Status", "").strip()
            if arr in stats:
                stats[arr]["total"] += 1
                if st == "On Time":
                    stats[arr]["on_time"] += 1

    rates = {}
    for code, s in stats.items():
        if s["total"] >= 5:
            rates[code] = round((s["on_time"] / s["total"]) * 100, 1)
        else:
            rates[code] = 84.5

    logger.info("Computed punctuality rates: %s", rates)
    return rates


def assign_aircraft(airline: str, duration_min: int) -> str:
    al = airline.lower()
    if "победа" in al or "pegasus" in al or "smartavia" in al:
        return "Boeing 737-800 NextGen"
    if "emirates" in al:
        return "Airbus A380-800" if duration_min > 300 else "Boeing 777-300ER"
    if "flydubai" in al:
        return "Boeing 737 MAX 8"
    if "turkish" in al or "qatar" in al:
        return "Airbus A350-900" if duration_min > 300 else "Airbus A321neo"
    if "аэрофлот" in al:
        return "Boeing 777-300ER" if duration_min > 360 else ("Airbus A321neo" if duration_min > 180 else "Airbus A320")
    if "s7" in al:
        return "Airbus A320neo"
    if "nordwind" in al or "red wings" in al:
        return "Boeing 777-200ER" if duration_min > 360 else "Airbus A321"
    return "Airbus A321neo"


def assign_amenities(airline: str, price: int):
    al = airline.lower()
    if "победа" in al or "pegasus" in al or "smartavia" in al:
        return {
            "baggage": "Ручная кладь 10 кг (багаж от 1 500 ₽)",
            "meals": "Платное бортовое меню и напитки",
            "seat_pitch": "74-76 см"
        }
    if "emirates" in al or "qatar" in al or "etihad" in al:
        return {
            "baggage": "Багаж 2×23 кг + ручная кладь 10 кг",
            "meals": "Премиальное многоразовое горячее питание и бар",
            "seat_pitch": "84-86 см (ICE Entertainment)"
        }
    if "аэрофлот" in al or "turkish" in al:
        return {
            "baggage": "Багаж 23 кг + ручная кладь 10 кг",
            "meals": "Горячее питание, десерты и напитки включены",
            "seat_pitch": "81-83 см"
        }
    return {
        "baggage": "Багаж 23 кг + ручная кладь 10 кг",
        "meals": "Снэки и прохладительные напитки",
        "seat_pitch": "78-81 см"
    }


def enrich_flights():
    with open(DESTINATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rates = compute_punctuality()
    dest_map = {d["id"]: d for d in data["destinations"]}

    # 1. Enrich existing flights
    enriched_existing = 0
    for dest in data["destinations"]:
        for fl in dest.get("flights", []):
            arr_code = fl.get("to", "")
            on_time = rates.get(arr_code, 82.0)
            fl["on_time_pct"] = on_time
            fl["aircraft_type"] = assign_aircraft(fl.get("airline", ""), fl.get("duration_minutes", 180))
            amenities = assign_amenities(fl.get("airline", ""), fl.get("price_rub_roundtrip", 20000))
            fl["baggage_allowance"] = amenities["baggage"]
            fl["meal_service"] = amenities["meals"]
            fl["seat_pitch"] = amenities["seat_pitch"]
            enriched_existing += 1

    logger.info("Enriched %d existing flights with punctuality and aircraft specs.", enriched_existing)

    # 2. Add realistic scheduled flights to cover all departure hubs and time slots
    new_flights_definitions = [
        # --- СОЧИ (AER) ---
        {
            "dest_id": "sochi",
            "from": "LED", "to": "AER", "airline": "Аэрофлот (Россия)", "flight_no": "SU-6561",
            "departure_time": "12:15", "arrival_time": "16:45", "duration_minutes": 270,
            "price_rub_roundtrip": 14200, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой дневной рейс из Пулково в Сочи (Адлер)."
        },
        {
            "dest_id": "sochi",
            "from": "AER", "to": "LED", "airline": "Аэрофлот (Россия)", "flight_no": "SU-6562",
            "departure_time": "17:45", "arrival_time": "22:15", "duration_minutes": 270,
            "price_rub_roundtrip": 14200, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой вечерний рейс Сочи ➔ Санкт-Петербург."
        },
        {
            "dest_id": "sochi",
            "from": "DME", "to": "AER", "airline": "S7 Airlines", "flight_no": "S7-2045",
            "departure_time": "14:10", "arrival_time": "18:05", "duration_minutes": 235,
            "price_rub_roundtrip": 13500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной комфортный рейс S7 Airlines из Домодедово."
        },
        {
            "dest_id": "sochi",
            "from": "AER", "to": "DME", "airline": "S7 Airlines", "flight_no": "S7-2046",
            "departure_time": "19:00", "arrival_time": "22:50", "duration_minutes": 230,
            "price_rub_roundtrip": 13500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Вечерний обратный рейс S7 Airlines Сочи ➔ Домодедово."
        },
        {
            "dest_id": "sochi",
            "from": "VKO", "to": "AER", "airline": "Победа (Вечерний)", "flight_no": "DP-119",
            "departure_time": "18:40", "arrival_time": "22:35", "duration_minutes": 235,
            "price_rub_roundtrip": 10500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Вечерний бюджетный рейс Победы."
        },
        {
            "dest_id": "sochi",
            "from": "AER", "to": "VKO", "airline": "Победа (Вечерний)", "flight_no": "DP-120",
            "departure_time": "23:25", "arrival_time": "03:15", "duration_minutes": 230,
            "price_rub_roundtrip": 10500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Ночной обратный рейс Победы в Москву."
        },

        # --- ОАЭ (DXB / AUH) ---
        {
            "dest_id": "uae",
            "from": "LED", "to": "DXB", "airline": "Emirates (Пулково ➔ Дубай)", "flight_no": "EK-176",
            "departure_time": "16:45", "arrival_time": "23:15", "duration_minutes": 330,
            "price_rub_roundtrip": 48500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой рейс Emirates на широкофюзеляжном Boeing 777-300ER из Санкт-Петербурга."
        },
        {
            "dest_id": "uae",
            "from": "DXB", "to": "LED", "airline": "Emirates (Дубай ➔ Пулково)", "flight_no": "EK-175",
            "departure_time": "09:30", "arrival_time": "15:05", "duration_minutes": 335,
            "price_rub_roundtrip": 48500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной прямой рейс Emirates в Пулково."
        },
        {
            "dest_id": "uae",
            "from": "SVO", "to": "AUH", "airline": "Etihad Airways (Дневной)", "flight_no": "EY-68",
            "departure_time": "12:55", "arrival_time": "19:25", "duration_minutes": 330,
            "price_rub_roundtrip": 42000, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Премиальный дневной беспосадочный рейс в Абу-Даби (терминал Zayed International)."
        },
        {
            "dest_id": "uae",
            "from": "AUH", "to": "SVO", "airline": "Etihad Airways (Обратный)", "flight_no": "EY-67",
            "departure_time": "08:15", "arrival_time": "12:50", "duration_minutes": 335,
            "price_rub_roundtrip": 42000, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Утренний обратный рейс Etihad Airways в Москву."
        },
        {
            "dest_id": "uae",
            "from": "DME", "to": "DXB", "airline": "FlyDubai (Дневной)", "flight_no": "FZ-992",
            "departure_time": "13:30", "arrival_time": "19:40", "duration_minutes": 310,
            "price_rub_roundtrip": 34900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной рейс FlyDubai без ночных подъемов."
        },
        {
            "dest_id": "uae",
            "from": "DXB", "to": "DME", "airline": "FlyDubai (Обратный)", "flight_no": "FZ-991",
            "departure_time": "08:00", "arrival_time": "12:20", "duration_minutes": 320,
            "price_rub_roundtrip": 34900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Утренний рейс FlyDubai Дубай ➔ Домодедово."
        },

        # --- ТУРЦИЯ (AYT / IST) ---
        {
            "dest_id": "turkey",
            "from": "LED", "to": "AYT", "airline": "Turkish Airlines (Пулково ➔ Анталья)", "flight_no": "TK-3673",
            "departure_time": "13:10", "arrival_time": "17:40", "duration_minutes": 270,
            "price_rub_roundtrip": 26900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой дневной регулярный рейс Turkish Airlines в Анталью."
        },
        {
            "dest_id": "turkey",
            "from": "AYT", "to": "LED", "airline": "Turkish Airlines (Анталья ➔ Пулково)", "flight_no": "TK-3672",
            "departure_time": "08:00", "arrival_time": "12:15", "duration_minutes": 255,
            "price_rub_roundtrip": 26900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Утренний обратный рейс Анталья ➔ Санкт-Петербург."
        },
        {
            "dest_id": "turkey",
            "from": "VKO", "to": "AYT", "airline": "Turkish Airlines (Дневной)", "flight_no": "TK-212",
            "departure_time": "14:15", "arrival_time": "18:40", "duration_minutes": 265,
            "price_rub_roundtrip": 23500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной рейс во Внуково."
        },
        {
            "dest_id": "turkey",
            "from": "AYT", "to": "VKO", "airline": "Turkish Airlines (Обратный)", "flight_no": "TK-211",
            "departure_time": "09:00", "arrival_time": "13:15", "duration_minutes": 255,
            "price_rub_roundtrip": 23500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Утренний обратный рейс Анталья ➔ Внуково."
        },
        {
            "dest_id": "turkey",
            "from": "DME", "to": "IST", "airline": "Pegasus Airlines (Сабиха Гёкчен)", "flight_no": "PC-389",
            "departure_time": "17:20", "arrival_time": "21:35", "duration_minutes": 255,
            "price_rub_roundtrip": 16500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Вечерний бюджетный рейс Pegasus в Стамбул (SAW)."
        },
        {
            "dest_id": "turkey",
            "from": "IST", "to": "DME", "airline": "Pegasus Airlines (Обратный)", "flight_no": "PC-388",
            "departure_time": "11:45", "arrival_time": "16:15", "duration_minutes": 270,
            "price_rub_roundtrip": 16500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной обратный рейс Стамбул ➔ Домодедово."
        },

        # --- ТАИЛАНД (HKT) ---
        {
            "dest_id": "thailand",
            "from": "LED", "to": "HKT", "airline": "Аэрофлот (Пулково ➔ Пхукет)", "flight_no": "SU-629",
            "departure_time": "19:50", "arrival_time": "09:40", "duration_minutes": 590,
            "price_rub_roundtrip": 58900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой беспосадочный рейс из Санкт-Петербурга в Таиланд (9ч 50м)."
        },
        {
            "dest_id": "thailand",
            "from": "HKT", "to": "LED", "airline": "Аэрофлот (Пхукет ➔ Пулково)", "flight_no": "SU-630",
            "departure_time": "11:50", "arrival_time": "18:25", "duration_minutes": 635,
            "price_rub_roundtrip": 58900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной обратный беспосадочный рейс Пхукет ➔ Санкт-Петербург."
        },
        {
            "dest_id": "thailand",
            "from": "DME", "to": "HKT", "airline": "Qatar Airways (Комфортная стыковка)", "flight_no": "QR-338",
            "departure_time": "13:10", "arrival_time": "06:20", "duration_minutes": 610,
            "price_rub_roundtrip": 62500, "flight_type": "connecting",
            "layover_airport": "DOH", "layover_city": "Доха", "layover_duration_minutes": 135,
            "layover_risk_note": "Комфортная пересадка в лучшем аэропорту мира Хамад (2ч 15м)."
        },
        {
            "dest_id": "thailand",
            "from": "HKT", "to": "DME", "airline": "Qatar Airways (Обратный)", "flight_no": "QR-339",
            "departure_time": "08:10", "arrival_time": "18:35", "duration_minutes": 685,
            "price_rub_roundtrip": 62500, "flight_type": "connecting",
            "layover_airport": "DOH", "layover_city": "Доха", "layover_duration_minutes": 140,
            "layover_risk_note": "Сквозной билет через Доху: багаж перегружается автоматически."
        },

        # --- ЕГИПЕТ (SSH / HRG) ---
        {
            "dest_id": "egypt",
            "from": "LED", "to": "SSH", "airline": "Red Wings (Пулково ➔ Шарм)", "flight_no": "WZ-3027",
            "departure_time": "07:30", "arrival_time": "13:20", "duration_minutes": 350,
            "price_rub_roundtrip": 29800, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой рейс из Пулково к Красному морю."
        },
        {
            "dest_id": "egypt",
            "from": "SSH", "to": "LED", "airline": "Red Wings (Шарм ➔ Пулково)", "flight_no": "WZ-3028",
            "departure_time": "14:40", "arrival_time": "20:30", "duration_minutes": 350,
            "price_rub_roundtrip": 29800, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной обратный рейс Шарм-эль-Шейх ➔ Санкт-Петербург."
        },
        {
            "dest_id": "egypt",
            "from": "SVO", "to": "SSH", "airline": "Аэрофлот (Дневной)", "flight_no": "SU-426",
            "departure_time": "11:55", "arrival_time": "17:10", "duration_minutes": 315,
            "price_rub_roundtrip": 34500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Регулярный прямой дневной рейс Аэрофлота."
        },
        {
            "dest_id": "egypt",
            "from": "SSH", "to": "SVO", "airline": "Аэрофлот (Обратный)", "flight_no": "SU-427",
            "departure_time": "18:30", "arrival_time": "23:45", "duration_minutes": 315,
            "price_rub_roundtrip": 34500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Вечерний обратный рейс Аэрофлота в Шереметьево."
        },
        {
            "dest_id": "egypt",
            "from": "DME", "to": "HRG", "airline": "Air Cairo (Хургада)", "flight_no": "SM-902",
            "departure_time": "13:40", "arrival_time": "18:45", "duration_minutes": 305,
            "price_rub_roundtrip": 27500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Прямой рейс в Хургаду."
        },
        {
            "dest_id": "egypt",
            "from": "HRG", "to": "DME", "airline": "Air Cairo (Обратный)", "flight_no": "SM-901",
            "departure_time": "07:30", "arrival_time": "12:35", "duration_minutes": 305,
            "price_rub_roundtrip": 27500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Утренний рейс Хургада ➔ Москва."
        },

        # --- МАЛЬДИВЫ (MLE) ---
        {
            "dest_id": "maldives",
            "from": "SVO", "to": "MLE", "airline": "Аэрофлот (Дневной вылет)", "flight_no": "SU-320",
            "departure_time": "14:20", "arrival_time": "01:05", "duration_minutes": 525,
            "price_rub_roundtrip": 78900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной беспосадочный рейс. Прилёт ночью: требуется скоростной катер или ночь в Мале."
        },
        {
            "dest_id": "maldives",
            "from": "MLE", "to": "SVO", "airline": "Аэрофлот (Обратный)", "flight_no": "SU-321",
            "departure_time": "09:40", "arrival_time": "16:55", "duration_minutes": 555,
            "price_rub_roundtrip": 78900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной обратный рейс в Шереметьево."
        },
        {
            "dest_id": "maldives",
            "from": "VKO", "to": "MLE", "airline": "FlyDubai + Emirates (Стыковка в Дубае)", "flight_no": "FZ-872",
            "departure_time": "14:40", "arrival_time": "06:30", "duration_minutes": 530,
            "price_rub_roundtrip": 68500, "flight_type": "connecting",
            "layover_airport": "DXB", "layover_city": "Дубай", "layover_duration_minutes": 150,
            "layover_risk_note": "Комфортная пересадка 2ч 30м в терминале 3 Дубая. Сквозной багаж."
        },
        {
            "dest_id": "maldives",
            "from": "MLE", "to": "VKO", "airline": "FlyDubai + Emirates (Обратный)", "flight_no": "FZ-871",
            "departure_time": "09:20", "arrival_time": "19:40", "duration_minutes": 560,
            "price_rub_roundtrip": 68500, "flight_type": "connecting",
            "layover_airport": "DXB", "layover_city": "Дубай", "layover_duration_minutes": 135,
            "layover_risk_note": "Сквозной билет через Дубай."
        },

        # --- САНКТ-ПЕТЕРБУРГ (LED) ---
        {
            "dest_id": "spb",
            "from": "SVO", "to": "LED", "airline": "Аэрофлот (Шаттл Утренний)", "flight_no": "SU-14",
            "departure_time": "08:15", "arrival_time": "09:45", "duration_minutes": 90,
            "price_rub_roundtrip": 5400, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Регулярный авиашаттл Аэрофлота Москва ➔ Санкт-Петербург."
        },
        {
            "dest_id": "spb",
            "from": "LED", "to": "SVO", "airline": "Аэрофлот (Шаттл Вечерний)", "flight_no": "SU-29",
            "departure_time": "19:30", "arrival_time": "21:00", "duration_minutes": 90,
            "price_rub_roundtrip": 5400, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Вечерний шаттл Санкт-Петербург ➔ Москва."
        },
        {
            "dest_id": "spb",
            "from": "DME", "to": "LED", "airline": "S7 Airlines (Дневной)", "flight_no": "S7-1011",
            "departure_time": "13:20", "arrival_time": "14:55", "duration_minutes": 95,
            "price_rub_roundtrip": 4900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной рейс S7 Airlines."
        },
        {
            "dest_id": "spb",
            "from": "LED", "to": "DME", "airline": "S7 Airlines (Обратный)", "flight_no": "S7-1012",
            "departure_time": "16:00", "arrival_time": "17:35", "duration_minutes": 95,
            "price_rub_roundtrip": 4900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной обратный рейс S7 Airlines."
        },

        # --- ГОРНЫЙ АЛТАЙ (RGK) ---
        {
            "dest_id": "altai",
            "from": "SVO", "to": "RGK", "airline": "Аэрофлот (Дневной)", "flight_no": "SU-1524",
            "departure_time": "11:40", "arrival_time": "20:00", "duration_minutes": 260,
            "price_rub_roundtrip": 23500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Дневной беспосадочный рейс в Горно-Алтайск (разница во времени +4 часа)."
        },
        {
            "dest_id": "altai",
            "from": "RGK", "to": "SVO", "airline": "Аэрофлот (Обратный)", "flight_no": "SU-1525",
            "departure_time": "21:10", "arrival_time": "21:40", "duration_minutes": 270,
            "price_rub_roundtrip": 23500, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Вечерний рейс Горно-Алтайск ➔ Шереметьево."
        },
        {
            "dest_id": "altai",
            "from": "VKO", "to": "RGK", "airline": "Победа (Бюджетный)", "flight_no": "DP-463",
            "departure_time": "23:55", "arrival_time": "08:15", "duration_minutes": 260,
            "price_rub_roundtrip": 15900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Ночной бюджетный перелёт Победы на Алтай."
        },
        {
            "dest_id": "altai",
            "from": "RGK", "to": "VKO", "airline": "Победа (Обратный)", "flight_no": "DP-464",
            "departure_time": "09:20", "arrival_time": "09:50", "duration_minutes": 270,
            "price_rub_roundtrip": 15900, "flight_type": "direct",
            "layover_airport": None, "layover_city": None, "layover_duration_minutes": 0,
            "layover_risk_note": "Утренний обратный рейс Победы в Москву."
        }
    ]

    existing_flight_keys = set()
    for dest in data["destinations"]:
        for fl in dest.get("flights", []):
            existing_flight_keys.add((fl["from"], fl["to"], fl["flight_no"]))

    added_count = 0
    for fl_def in new_flights_definitions:
        dest_id = fl_def.pop("dest_id")
        key = (fl_def["from"], fl_def["to"], fl_def["flight_no"])
        if key in existing_flight_keys:
            continue

        arr_code = fl_def["to"]
        fl_def["on_time_pct"] = rates.get(arr_code, 83.5)
        fl_def["aircraft_type"] = assign_aircraft(fl_def["airline"], fl_def["duration_minutes"])
        amenities = assign_amenities(fl_def["airline"], fl_def["price_rub_roundtrip"])
        fl_def["baggage_allowance"] = amenities["baggage"]
        fl_def["meal_service"] = amenities["meals"]
        fl_def["seat_pitch"] = amenities["seat_pitch"]

        target_dest = dest_map.get(dest_id)
        if target_dest:
            target_dest.setdefault("flights", []).append(fl_def)
            existing_flight_keys.add(key)
            added_count += 1

    logger.info("Added %d new high-demand scheduled flights covering all hubs and time slots.", added_count)

    # Save destinations.json
    with open(DESTINATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_flights_now = sum(len(d.get("flights", [])) for d in data["destinations"])
    logger.info("Updated destinations.json! Total flights now: %d (previously 87).", total_flights_now)


if __name__ == "__main__":
    enrich_flights()
