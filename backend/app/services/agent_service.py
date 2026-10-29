import json
import logging
import re
from copy import deepcopy
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import select, text

from app.database import async_session_maker
from app.models import DestinationModel, HotelModel, RiskAlertModel, UserSessionModel
from app.config import settings
from app.services.vector_service import generate_text_embedding
from app.mcp.tools import MCPTools
from app.services.trip_dates import update_trip_dates
from app.services.trip_budget import update_trip_budget, validate_package_totals
from app.services.trip_state import (
    current_trip_question, describe_budget_estimate, describe_current_trip,
    estimate_request, is_trip_edit, requested_destination,
    hotel_selection, matching_hotels, rejected_destinations, wants_cheaper,
    destination_choice,
    transfer_preference,
    single_destination_name,
)
from app.services.booking_stub import detect_booking_action, booking_stub
from app.services.preference_memory import remembered_preferences
from app.services.llm_service import LLMService
from app.services.graph_service import GraphService
from app.services.compliance_service import ComplianceService
from app.services.calendar_service import CalendarService
from app.services.ltr_service import LTRService

logger = logging.getLogger(__name__)

MONTH_NAMES_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая", 6: "июня",
    7: "июля", 8: "августа", 9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
}
MONTH_SHORT_RU = {
    1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "мая", 6: "июн",
    7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек"
}
DAY_OF_WEEK_RU = {
    0: "пн", 1: "вт", 2: "ср", 3: "чт", 4: "пт", 5: "сб", 6: "вс"
}


def format_russian_date_range(d_in_str: Optional[str], d_out_str: Optional[str], nights: int) -> str:
    if not d_in_str or not d_out_str:
        return f"{nights} ночей"
    try:
        d1 = date.fromisoformat(d_in_str)
        d2 = date.fromisoformat(d_out_str)
        return f"{d1.day} {MONTH_NAMES_RU[d1.month]} — {d2.day} {MONTH_NAMES_RU[d2.month]} {d2.year} г. ({nights} ночей)"
    except Exception:
        return f"{d_in_str} — {d_out_str} ({nights} ночей)"


def format_flight_date(d_str: Optional[str]) -> str:
    if not d_str:
        return ""
    try:
        d = date.fromisoformat(d_str)
        return f" ({d.day} {MONTH_SHORT_RU[d.month]}, {DAY_OF_WEEK_RU[d.weekday()]})"
    except Exception:
        return f" ({d_str})"


def format_hotel_dates(d_in_str: Optional[str], d_out_str: Optional[str], nights: int) -> str:
    if not d_in_str or not d_out_str:
        return f"на {nights} ночей"
    try:
        d1 = date.fromisoformat(d_in_str)
        d2 = date.fromisoformat(d_out_str)
        return f"{d1.day} {MONTH_SHORT_RU[d1.month]} — {d2.day} {MONTH_SHORT_RU[d2.month]}, {nights} ночей"
    except Exception:
        return f"{d_in_str} — {d_out_str}, {nights} ночей"


# Маппинг внешних направлений в наши 8 расширенных регионов (Пункт 3 ТЗ)
UNKNOWN_DESTINATIONS_MAP = [
    {
        "keywords": ["бали", "сейшел", "гоа", "вьетнам", "шри-ланк", "доминикан", "куб", "маврики", "фиджи", "бора-бора"],
        "pilot_alt_id": "maldives",
        "pilot_alt_name": "Мальдивы (Атоллы)",
        "vibe_reason": "белоснежные тропические пляжи, лазурный Индийский океан, уединенные виллы над водой и живые коралловые рифы (например, Cheval Blanc Randheli или Villa Nautica)"
    },
    {
        "keywords": ["париж", "рим", "барселон", "праг", "европ", "итали", "франци", "испани", "вена", "будапешт"],
        "pilot_alt_id": "spb",
        "pilot_alt_name": "Санкт-Петербург (Северная столица)",
        "vibe_reason": "европейская классическая архитектура, парадные дворцы, Эрмитаж, Мариинский театр и гранд-отели мирового уровня (Four Seasons Lion Palace или Гранд Отель Европа)"
    },
    {
        "keywords": ["иордани", "марокко", "катар", "оман", "бахрейн"],
        "pilot_alt_id": "egypt",
        "pilot_alt_name": "Египет (Шарм-эль-Шейх / Красное море)",
        "vibe_reason": "богатейший подводный мир, круглогодичное теплое море, коралловые заповедники и курорты All-Inclusive (Rixos Premium Seagate или Four Seasons)"
    },
    {
        "keywords": ["альп", "доломит", "домбай", "эльбрус", "байкал", "карели", "кавказ", "шерегеш", "архыз"],
        "pilot_alt_id": "altai",
        "pilot_alt_name": "Алтай (Манжерок / Горный Алтай)",
        "vibe_reason": "величественная первозданная природа, бирюзовая Катунь, вековые кедры, чистейший горный воздух и премиальные эко-курорты (Манжерок 5★ или Altay Resort)"
    }
]

MULTI_CITY_CONFIG = {
    "turkey": {
        "title": "Турция (Анталья + Стамбул)",
        "country": "Турция",
        "climate": "Средиземноморское побережье и величие Босфора",
        "tags": ["море", "пляж", "культура", "история", "босфор", "город у моря", "много городов у моря", "европа и азия", "архитектура", "шопинг"],
        "selection_rationale": "По вашему запросу на тур по городам («{criteria}») идеально подходит связка Анталья + Стамбул: прямой перелёт между городами (1 ч 15 мин), объединяющий лазурные пляжи Средиземного моря и богатое культурное наследие Босфора.",
        "city_1": {
            "name": "Анталья",
            "nights_share": 0.55,
            "vibe": "пляжный релакс, лазурное Средиземное море и старый город Калеичи",
            "activities": "отдых на пляжах Лара и Коньяалты, старинный квартал Калеичи, ворота Адриана и водопады Дюден",
            "airport": "AYT",
            "weather": "Сегодня: +23°C (море +22°C), солнечно, без осадков."
        },
        "city_2": {
            "name": "Стамбул",
            "nights_share": 0.45,
            "vibe": "культура, архитектура, шопинг и круизы по Босфору",
            "activities": "круиз по проливу Босфор на закате, собор Айя-София, дворец Топкапы, колоритный Гранд-базар и аутентичные кофейни",
            "airport": "IST",
            "weather": "Сегодня: +18°C, комфортно для прогулок, переменная облачность."
        },
        "intercity": {
            "type": "flight",
            "name": "Внутренний авиаперелёт TK-2415 Анталья (AYT) ➔ Стамбул (IST)",
            "operator": "Turkish Airlines (TK-2415)",
            "departure": "11:30",
            "arrival": "12:45",
            "duration": "1 ч 15 мин",
            "price_rub": 3800,
            "description": "Прямой внутренний перелёт Turkish Airlines TK-2415 (1 ч 15 мин) прямо в Новый аэропорт Стамбула."
        }
    },
    "sochi": {
        "title": "Сочи (Побережье + Красная Поляна)",
        "country": "Россия",
        "climate": "Субтропическое море и чистейший горный воздух Кавказа",
        "tags": ["море", "пляж", "горы", "горный", "хайкинг", "спа", "природа", "море и горы", "горы и море", "лыжи", "активный отдых"],
        "selection_rationale": "По вашему запросу («{criteria}») я подобрал связку Сочи (Сириус) + Красная Поляна: первые дни вы наслаждаетесь теплым морем и променадом Имеретинки, а затем за 45 минут на комфортабельной «Ласточке» перемещаетесь в горы на высоту 960–2200м для хайкинга и альпийского спа.",
        "city_1": {
            "name": "Сочи (Сириус / Побережье)",
            "nights_share": 0.55,
            "vibe": "морской променад и Олимпийский парк",
            "activities": "морской бриз, прогулки по набережной Имеретинки, поющие фонтаны и Олимпийский парк",
            "airport": "AER",
            "weather": "Сегодня: +19°C, лёгкий морской бриз, ясно."
        },
        "city_2": {
            "name": "Красная Поляна",
            "nights_share": 0.45,
            "vibe": "альпийские панорамы, реликтовые леса и горные спа",
            "activities": "подъем на канатных дорогах на высоту 2200м, прогулки по эко-тропам, банные комплексы и горный релакс",
            "airport": "AER",
            "weather": "Сегодня: +14°C в горах, чистый горный воздух, солнечно."
        },
        "intercity": {
            "type": "train",
            "name": "Скоростной электропоезд «Ласточка»",
            "operator": "РЖД Экспресс",
            "departure": "12:15",
            "arrival": "13:00",
            "duration": "45 мин",
            "price_rub": 450,
            "description": "Комфортабельный поезд «Ласточка» с панорамными окнами вдоль живописного ущелья реки Мзымта."
        }
    },
    "uae": {
        "title": "ОАЭ (Дубай + Абу-Даби)",
        "country": "ОАЭ",
        "climate": "Современные мегаполисы, Персидский залив и арабская роскошь",
        "tags": ["небоскребы", "шопинг", "роскошь", "футуризм", "мегаполисы", "залив", "персидский залив", "парки развлечений", "лувр"],
        "selection_rationale": "Для ценителей современного ритма и ярких впечатлений («{criteria}») идеально подходит связка Дубай + Абу-Даби: ультрасовременные небоскребы, шопинг мирового уровня, пляжи Персидского залива и шедевры Лувра Абу-Даби с комфортным экспресс-трансфером.",
        "city_1": {
            "name": "Дубай",
            "nights_share": 0.6,
            "vibe": "футуристические небоскребы, пляжи JBR и шопинг",
            "activities": "Бурдж-Халифа, шоу фонтанов, пляжи Dubai Marina, шопинг в Dubai Mall",
            "airport": "DXB",
            "weather": "Сегодня: +31°C (вода +28°C), солнечно, штиль."
        },
        "city_2": {
            "name": "Абу-Даби",
            "nights_share": 0.4,
            "vibe": "восточная культура, шедевры Лувра и остров Яс",
            "activities": "белоснежная мечеть Шейха Зайда, Лувр Абу-Даби, парк развлечений Ferrari World",
            "airport": "DXB",
            "weather": "Сегодня: +32°C, солнечно, без осадков."
        },
        "intercity": {
            "type": "transfer",
            "name": "Междугородний экспресс-трансфер Дубай ➔ Абу-Даби",
            "operator": "Ostrovok Express Transfer",
            "departure": "11:00",
            "arrival": "12:15",
            "duration": "1 ч 15 мин",
            "price_rub": 3500,
            "description": "Поездка на комфортабельном экспресс-трансфере по скоростному шоссе Sheikh Zayed Road (E11)."
        }
    },
    "altai": {
        "title": "Алтай (Горно-Алтайск + Чемал)",
        "country": "Россия",
        "climate": "Место силы, хвойные леса и бирюзовая река Катунь",
        "tags": ["горы", "реки", "катунь", "природа", "места силы", "эко-туризм", "хвойные леса", "чистый воздух", "телецкое озеро", "чемал"],
        "selection_rationale": "По вашему запросу на единение с природой («{criteria}») составлен живописный маршрут по Алтаю: Горно-Алтайск (Телецкое озеро) + Чемал вдоль бирюзовой Катуни с трансфером по легендарному Чуйскому тракту.",
        "city_1": {
            "name": "Горно-Алтайск / Майма",
            "nights_share": 0.45,
            "vibe": "врата Алтая и Телецкое озеро",
            "activities": "катерная экскурсия по Телецкому озеру, дегустация алтайского меда и травяных чаев",
            "airport": "RGK",
            "weather": "Сегодня: +12°C, горная свежесть, солнечно."
        },
        "city_2": {
            "name": "Чемал",
            "nights_share": 0.55,
            "vibe": "скалистые берега Катуни и сосновый воздух",
            "activities": "остров Патмос, скалы «Зубы Дракона», Чемальская ГЭС, прогулки по козьей тропе",
            "airport": "RGK",
            "weather": "Сегодня: +13°C, ясная погода, запах кедровой хвои."
        },
        "intercity": {
            "type": "transfer",
            "name": "Трансфер по Чуйскому тракту",
            "operator": "Алтай-Тур Трансфер",
            "departure": "10:30",
            "arrival": "11:50",
            "duration": "1 ч 20 мин",
            "price_rub": 2500,
            "description": "Живописная поездка по легендарному Чуйскому тракту вдоль бурлящей реки Катунь."
        }
    },
    "thailand": {
        "title": "Таиланд (Пхукет: Патонг + Банг Тао)",
        "country": "Таиланд",
        "climate": "Тропическое Андаманское море, белоснежные пляжи и экзотика",
        "tags": ["острова", "экзотика", "тропики", "джунгли", "море", "пляж", "буддизм", "дайвинг", "зимовка"],
        "selection_rationale": "По вашему запросу на тропическую экзотику («{criteria}») составлен тур по Таиланду: яркий динамичный Патонг/Ката и уединенные белоснежные пляжи Банг Тао с панорамными закатами.",
        "city_1": {
            "name": "Патонг (Пхукет)",
            "nights_share": 0.55,
            "vibe": "активная жизнь, лазурные пляжи и катамараны на острова",
            "activities": "пляжный отдых, морские прогулки на спидботах к островам, тайский стрит-фуд и ночные рынки",
            "airport": "HKT",
            "weather": "Сегодня: +31°C (море +29°C), солнце, легкий бриз."
        },
        "city_2": {
            "name": "Банг Тао (Пхукет)",
            "nights_share": 0.45,
            "vibe": "респектабельный уединенный релакс, лагуны и спа",
            "activities": "белоснежная песчаная коса Банг Тао, спа-центры мирового уровня, вечерний релакс в бич-клабах",
            "airport": "HKT",
            "weather": "Сегодня: +30°C, тропическая идиллия, штиль."
        },
        "intercity": {
            "type": "transfer",
            "name": "Премиум-трансфер по побережью Пхукета",
            "operator": "Phuket Shuttle Express",
            "departure": "11:00",
            "arrival": "11:50",
            "duration": "50 мин",
            "price_rub": 2200,
            "description": "Комфортный трансфер на кондиционированном авто вдоль живописного побережья острова."
        }
    },
    "spb": {
        "title": "Санкт-Петербург (Исторический центр + Петергоф)",
        "country": "Россия",
        "climate": "Северная Венеция, парадные набережные и царские резиденции",
        "tags": ["культура", "архитектура", "музеи", "эрмитаж", "дворцы", "фонтаны", "каналы", "театры", "северная столица", "европейская классика"],
        "selection_rationale": "По вашему запросу на культурное путешествие («{criteria}») составлен классический маршрут: исторический центр Петербурга (каналы, театры, Эрмитаж) и царский Петергоф со скоростным «Метеором» по Финскому заливу.",
        "city_1": {
            "name": "Санкт-Петербург (Центр)",
            "nights_share": 0.6,
            "vibe": "Эрмитаж, набережные Невы, Мариинский театр и крыши",
            "activities": "прогулка на катере по рекам и каналам, экскурсия в Эрмитаж и Спас на Крови, авторские гастробары",
            "airport": "LED",
            "weather": "Сегодня: +11°C, лёгкий морской бриз, переменная облачность."
        },
        "city_2": {
            "name": "Петергоф",
            "nights_share": 0.4,
            "vibe": "фонтаны, царские парки и дворцовый шик",
            "activities": "Большой Петергофский каскад, парк Александрия, дворец Монплезир и прогулки вдоль залива",
            "airport": "LED",
            "weather": "Сегодня: +12°C, свежий морской воздух, ясно."
        },
        "intercity": {
            "type": "boat",
            "name": "Скоростной теплоход «Метеор» по Финскому заливу",
            "operator": "Петергоф Экспресс",
            "departure": "10:30",
            "arrival": "11:05",
            "duration": "35 мин",
            "price_rub": 1400,
            "description": "Захватывающая поездка на подводных крыльях из самого центра Петербурга (причал у Эрмитажа) прямо к Нижнему парку Петергофа."
        }
    },
    "egypt": {
        "title": "Египет (Шарм-эль-Шейх: Наама Бей + Хадаба)",
        "country": "Египет",
        "climate": "Круглогодичное Красное море, коралловые сады и Синайская пустыня",
        "tags": ["море", "пляж", "кораллы", "рифы", "дайвинг", "снорклинг", "рыбки", "красное море", "пустыня", "all-inclusive"],
        "selection_rationale": "По вашему запросу на теплое море и кораллы («{criteria}») подобран комбинированный отдых в Египте: пляжный отдых и променад в Наама Бей + лучшие рифы Хадабы для снорклинга.",
        "city_1": {
            "name": "Шарм-эль-Шейх (Наама Бей)",
            "nights_share": 0.5,
            "vibe": "променад, восточные базары и пляжи с песчаным заходом",
            "activities": "вечерний променад Наама Бей, кальяны и бедуинский чай, шоппинг и рестораны морепродуктов",
            "airport": "SSH",
            "weather": "Сегодня: +29°C (вода +27°C), солнечно, штиль."
        },
        "city_2": {
            "name": "Шарм (Хадаба)",
            "nights_share": 0.5,
            "vibe": "заповедные коралловые рифы Рас-Мохаммед и дайвинг",
            "activities": "снорклинг прямо у отельного рифа, дайв-сафари к затонувшим кораблям, сафари на квадроциклах на закате",
            "airport": "SSH",
            "weather": "Сегодня: +28°C (вода +27°C), кристальная видимость в море."
        },
        "intercity": {
            "type": "transfer",
            "name": "Видовой экспресс-трансфер по Синайскому побережью",
            "operator": "Red Sea Express",
            "departure": "11:30",
            "arrival": "12:00",
            "duration": "30 мин",
            "price_rub": 1200,
            "description": "Быстрый кондиционированный трансфер между районами с панорамой Синайских гор."
        }
    },
    "georgia": {
        "title": "Грузия (Тбилиси + Гудаури / Казбеги)",
        "country": "Грузия",
        "climate": "Старинный Тбилиси и заснеженные вершины Большого Кавказа",
        "tags": ["горы", "культура", "вино", "гастрономия", "хайкинг", "природа", "тбилиси", "гудаури", "казбеги", "батуми", "горы и культура"],
        "selection_rationale": "По вашему запросу («{criteria}») подобран комбинированный тур по Грузии: колоритный старинный Тбилиси с серными банями и гастрономией плюс величественные вершины Гудаури и Казбеги по Военно-Грузинской дороге.",
        "city_1": {
            "name": "Тбилиси",
            "nights_share": 0.5,
            "vibe": "старинные улочки Сололаки, серные бани Абанотубани и грузинское гостеприимство",
            "activities": "крепость Нарикала, фуникулер на Мтацминду, дегустация оранжевых вин и хинкали в аутентичных духанах",
            "airport": "TBS",
            "weather": "Сегодня: +18°C, тепло, солнечно."
        },
        "city_2": {
            "name": "Гудаури / Степанцминда",
            "nights_share": 0.5,
            "vibe": "альпийские панорамы Большого Кавказа и вид на гору Казбек",
            "activities": "подъем к церкви Гергети у подножия Казбека, панорамные спа с видом на горы, треккинг и чистейший воздух",
            "airport": "TBS",
            "weather": "Сегодня: +9°C в горах, чистый горный воздух, ясно."
        },
        "intercity": {
            "type": "transfer",
            "name": "Gotrip.ge 4x4 / Джип 4x4 по Военно-Грузинской дороге через Ананури и Жинвали",
            "operator": "Gotrip Georgia 4x4",
            "departure": "10:00",
            "arrival": "12:30",
            "duration": "150 мин",
            "price_rub": 8500,
            "description": "Поездка на полноприводном внедорожнике с остановками у крепости Ананури, Жинвальского водохранилища и Арки Дружбы Народов."
        }
    }
}


class AgentService:
    @staticmethod
    async def get_or_create_session(user_id: str) -> UserSessionModel:
        async with async_session_maker() as session:
            res = await session.execute(
                select(UserSessionModel).where(UserSessionModel.user_id == user_id)
            )
            user_session = res.scalar_one_or_none()
            if not user_session:
                user_session = UserSessionModel(
                    user_id=user_id,
                    current_state="INSPIRATION",
                    profile={
                        "has_kids": False,
                        "has_pets": False,
                        "preferred_destination": None,
                        "nights": 7,
                        "dates": "7 ночей",
                        "interests": [],
                        "rejected_hotels": [],
                        "prefer_direct_flight": None,
                        "without_transfer": False,
                        "party_size": "вдвоем"
                    },
                    current_cart={},
                    conversation_history=[]
                )
                session.add(user_session)
                await session.commit()
                await session.refresh(user_session)
            return user_session

    @staticmethod
    def extract_interests_and_slots(user_lower: str, profile: dict) -> dict:
        """
        Извлекает интересы путешественника, состав и длительность поездки (Память пользователя).
        """
        user_lower = user_lower.lower().replace("ё", "е")
        previous_party = (profile.get("adults_count", 2), profile.get("kids_count", 0))
        previous_rooms = profile.get("rooms_count", 1)
        interests = set(profile.get("interests", []))
        private_room = bool(re.search(
            r"\b(?:не\s+хостел\w*|без\s+хостел\w*|хостел\w*\s+не\s+(?:нуж\w*|подход\w*)|отдельн\w*\s+номер\w*|собственн\w*\s+ванн\w*)", user_lower
        ))
        if private_room:
            profile["private_room_required"] = True
            interests.discard("hostel_dorm")

        if any(w in user_lower for w in ["пляж", "море", "купаться", "позагорать", "песок", "загар", "плавать"]):
            interests.add("beach")
        if any(w in user_lower for w in ["гор", "лыж", "сноуборд", "полян", "хайкинг", "треккинг"]):
            interests.add("mountains")
        if any(w in user_lower for w in ["истор", "экскурси", "достопримечательн", "музе", "культура", "босфор", "старый город"]):
            interests.add("history_culture")
        if any(w in user_lower for w in ["спа", "spa", "релакс", "массаж", "велнес", "бассейн"]):
            interests.add("spa_wellness")
        children_text = re.sub(r"\bбез\s+(?:детей|ребенка)\b", "", user_lower)
        if re.search(r"\b(?:ребен\w*|дети|детьми|детей|малыш\w*|семь\w*|аквапарк\w*)", children_text):
            interests.add("family_kids")
            profile["has_kids"] = True
        if re.search(r"\b(?:собак\w*|питомц\w*|кошк\w*|пес|пса|псом)\b", user_lower):
            interests.add("pet_friendly")
            profile["has_pets"] = True
        if any(w in user_lower for w in [
            "дешев", "эконом", "подешев", "недорог", "бюджетн", "маленький бюджет",
            "скромный бюджет", "небольшой бюджет", "минимальный бюджет", "ограниченный бюджет"
        ]) or (
            not profile.get("private_room_required") and any(w in user_lower for w in ["хостел", "койко"])
        ):
            interests.add("budget_friendly")
        if any(w in user_lower for w in [
            "самый дорог", "самый роскошн", "дворец", "burj al arab", "шейх",
            "президентск", "максимально дорог", "ультра-люкс", "любой бюджет", "денег не жалко"
        ]):
            interests.add("extreme_luxury")
            interests.add("luxury")
            interests.discard("budget_friendly")
        elif not any(neg in user_lower for neg in ["не дорог", "недорог"]) and any(w in user_lower for w in [
            "люкс", "премиум", "вилл", "5 звезд", "пять звезд", "deluxe", "vip",
            "роскошн", "раскошел", "не эконом", "шикануть", "лучший отель"
        ]):
            interests.add("luxury")
            interests.discard("budget_friendly")
        if any(w in user_lower for w in ["вдвоем", "парой", "романтик", "двое взрослых", "два взрослых", "двумя взрослыми", "двух взрослых"]):
            profile["party_size"] = "вдвоем"
            profile["adults_count"] = 2
        elif re.search(r"\b(?:(?:я|еду|путешествую)\s+(?:один|одна)|соло|один\s+взрослый)\b", user_lower) or user_lower.strip() in ("один", "одна"):
            profile["party_size"] = "соло"
            profile["adults_count"] = 1
        elif any(w in user_lower for w in ["втроем", "3 взрослых", "три взрослых", "трое взрослых", "тремя взрослыми", "на троих", "из трех друзей", "трое друзей"]):
            profile["party_size"] = "втроем (3 взр.)"
            profile["adults_count"] = 3
        elif any(w in user_lower for w in ["вчетвером", "4 взрослых", "четыре взрослых", "четверо взрослых", "четырьмя взрослыми"]):
            profile["party_size"] = "вчетвером (4 взр.)"
            profile["adults_count"] = 4
        elif any(w in user_lower for w in ["впятером", "5 взрослых", "пять взрослых"]):
            profile["party_size"] = "впятером (5 взр.)"
            profile["adults_count"] = 5
        else:
            m_adults = re.search(r"\bнас\s+(\d+)\b|\b(\d+)\s*(?:взросл\w*|человек\w*|пассажир\w*|турист\w*)", user_lower)
            if m_adults:
                try:
                    c = int(m_adults.group(1) or m_adults.group(2))
                    if 1 <= c <= 20:
                        profile["adults_count"] = c
                        profile["party_size"] = f"{c} чел."
                except ValueError:
                    pass

        # Дети
        m_kids = re.search(r"(\d+)\s*(?:ребен\w*|детей|дет\w*|малыш\w*)", user_lower)
        if m_kids:
            try:
                kc = int(m_kids.group(1))
                if 1 <= kc <= 10:
                    profile["kids_count"] = kc
                    profile["has_kids"] = True
            except ValueError:
                pass
        elif any(w in user_lower for w in ["двое детей", "двумя детьми", "2 детей", "двух детей"]):
            profile["kids_count"] = 2
            profile["has_kids"] = True
        elif any(w in user_lower for w in ["трое детей", "тремя детьми", "3 детей", "трех детей"]):
            profile["kids_count"] = 3
            profile["has_kids"] = True

        # Логика мульти-номеров (Family / Multi-Room)
        # Стандартный номер вмещает до 2 взрослых. При >=3 взрослых или от 4 человек с детьми бронируются смежные номера
        if re.search(r"\bбез\s+(?:детей|ребенка)\b", user_lower):
            profile["has_kids"] = False
            profile["kids_count"] = 0
            interests.discard("family_kids")
        if re.search(r"\bбез\s+(?:собаки|кошки|питомц\w*|животных)\b|\b(?:собака|кошка|питомец)\s+не\s+едет\b", user_lower):
            profile["has_pets"] = False
            interests.discard("pet_friendly")

        adults = profile.get("adults_count", 2)
        kids = profile.get("kids_count", 0)
        if adults >= 3 or (adults >= 2 and kids >= 2) or (adults + kids >= 4):
            rooms = max(2, (adults + 1) // 2)
            profile["rooms_count"] = rooms
            profile["room_type"] = f"{rooms} смежных номера Standard (Connecting Rooms)" if rooms == 2 else f"{rooms} отдельных номеров"
        else:
            profile["rooms_count"] = 1
            profile["room_type"] = "1 стандартный номер (Standard Room)"

        explicit_rooms = re.search(r"\b(\d+)\s*(?:комнат\w*|номер\w*)", user_lower)
        if explicit_rooms and 1 <= int(explicit_rooms.group(1)) <= 20:
            profile["requested_rooms_count"] = int(explicit_rooms.group(1))
            profile["private_room_required"] = True
            interests.discard("hostel_dorm")
        if profile.get("requested_rooms_count"):
            profile["rooms_count"] = max(profile["rooms_count"], profile["requested_rooms_count"])
            profile["room_type"] = f"{profile['rooms_count']} отдельных номеров"
        elif previous_party == (adults, kids) and previous_rooms > profile["rooms_count"]:
            profile["rooms_count"] = previous_rooms
            profile["room_type"] = f"{previous_rooms} отдельных номеров"

        for token in ["пулково", "led", "спб", "питер", "санкт-петербург"]:
            if f"из {token}" in user_lower or f"с {token}" in user_lower:
                profile["origin_airport"] = "LED"
        for token in ["сочи", "адлер", "aer"]:
            if f"из {token}" in user_lower:
                profile["origin_airport"] = "AER"
        for token in ["алтай", "горно-алтайск", "rgk"]:
            if f"из {token}" in user_lower:
                profile["origin_airport"] = "RGK"

        # Предпочтения по географии (заграница / Россия)
        if any(w in user_lower for w in ["заграниц", "за границу", "за границей", "за рубеж", "зарубеж", "не в росси", "не по росси", "не в рф", "другая страна", "другую страну", "только не россия", "только заграниц", "международн"]):
            profile["abroad_only"] = True
            profile.pop("domestic_only", None)
        elif any(w in user_lower for w in ["по россии", "внутри россии", "по рф", "в россии", "отечественн", "без загранпаспорта", "без визы и заграна"]):
            profile["domestic_only"] = True
            profile.pop("abroad_only", None)

        # Предпочтения по времени вылета
        if any(w in user_lower for w in ["днем", "днём", "в обед", "дневной вылет", "дневным вылетом"]):
            profile["flight_time_pref"] = "day"
        elif any(w in user_lower for w in ["вечером", "вечерний вылет", "ночью", "поздний вылет"]):
            profile["flight_time_pref"] = "evening"
        elif any(w in user_lower for w in ["попозже", "самолет попозже", "вылет попозже", "рейс попозже", "не в 6 утра", "не в 6", "не в 7", "не рано", "позже 10", "после 10", "после 12", "вылетать позже", "выспаться"]):
            profile["flight_time_pref"] = "later"
        elif any(w in user_lower for w in ["пораньше", "утром", "утренний вылет", "самолет пораньше", "рейс пораньше", "ранний вылет"]):
            profile["flight_time_pref"] = "morning"

        # Комбинированный тур по городам (Multi-City Tour)
        # Устанавливаем True только при явном запросе на комбинированный тур, либо при правке существующего
        is_explicit = AgentService.is_explicit_multi_city_request(user_lower)
        if is_explicit:
            profile["is_multi_city"] = True
        else:
            replan_multi = profile.get("is_multi_city") and any(w in user_lower for w in [
                "попозже", "пораньше", "сдвинь", "дешев", "дорож", "отель", "вылет", "рейс", "трансфер", "номер", "бюджет"
            ])
            if not replan_multi:
                profile["is_multi_city"] = False

        profile["interests"] = list(interests)
        return profile

    @staticmethod
    def detect_unknown_destination(message: str) -> Optional[Dict[str, str]]:
        """
        Проверяет, запрашивает ли пользователь направление вне текущей пилотной базы (Пункт 3 ТЗ).
        Возвращает информацию о пилотной альтернативе по вайбу/климату.
        """
        m_lower = message.lower()
        for item in UNKNOWN_DESTINATIONS_MAP:
            for kw in item["keywords"]:
                if kw in m_lower:
                    return item
        return None

    @staticmethod
    def get_clarification_prompt(message: str) -> str:
        """
        Формирует лаконичные ориентирующие вопросы (Анти-15-вопросов)
        в зависимости от общего направления мысли пользователя (море/пляж, горы/природа или общий отпуск).
        """
        m_lower = message.lower()
        if any(w in m_lower for w in ["море", "пляж", "морск", "побереж", "купат", "в тепло", "теплое море"]):
            return (
                "🌊 **Отличная идея — пляжный отдых!**\n\n"
                "Чтобы подобрать идеальный тур под ключ с проверенным перелётом, отелем и трансфером, уточните пару моментов:\n\n"
                "1️⃣ **Какое направление вам ближе?**\n"
                "• 🌴 **Турция (Анталья / Кемер / Белек)** — Тёплое море, любимый All-Inclusive, прямые рейсы без виз\n"
                "• 🌊 **Сочи / Сириус** — Родное побережье, быстрый перелёт по паспорту РФ, развитая инфраструктура\n"
                "• ✨ **ОАЭ (Дубай / Абу-Даби)** — Футуристичные пляжи, шопинг и первоклассный сервис\n"
                "• 🐠 **Египет (Шарм-эль-Шейх)** — Тёплое Красное море и живые коралловые рифы\n"
                "• 🏝️ **Мальдивы или Пхукет (Таиланд)** — Белоснежные пляжи, бирюзовые лагуны и экзотика\n\n"
                "2️⃣ **Когда и на сколько дней планируете поездку?** (например: «в ноябре на 7 ночей» или точные даты)\n"
                "3️⃣ **Кто едет и есть ли пожелания по бюджету?** (вдвоём, с семьёй/детьми, соло, комфортный или люкс)\n\n"
                "_Напишите ваши пожелания (например: «В Турцию на двоих в ноябре, всё включено»), и я сразу соберу готовый тур!_"
            )
        elif any(w in m_lower for w in ["гор", "лыж", "хайкинг", "канатк", "альп"]):
            return (
                "🏔️ **Прекрасный выбор — горы, чистый воздух и альпийские панорамы!**\n\n"
                "Чтобы подобрать лучший тур, уточните пару деталей:\n\n"
                "1️⃣ **Какой формат горного отдыха вам ближе?**\n"
                "• 🌲 **Красная Поляна (Сочи)** — Канатные дороги, пик 2200м, альпийские спа-отели и горный воздух\n"
                "• 🇬🇪 **Грузия (Гудаури / Казбеги)** — Величественные вершины Большого Кавказа, вид на гору Казбек, аутентичная кухня и вино\n"
                "• 🏞️ **Алтай (Манжерок / Чемал)** — Первозданная природа, бирюзовая Катунь, кедровые леса и места силы\n\n"
                "2️⃣ **Когда и на сколько дней планируете поездку?**\n"
                "3️⃣ **Состав путешественников и пожелания по бюджету?**\n\n"
                "_Напишите, что вам ближе, и я мгновенно составлю маршрут под ключ!_"
            )
        else:
            return (
                "👋 **Здравствуйте! Я умный ассистент путешествий Ostrovok.**\n\n"
                "Вместо длинных анкет из 15 вопросов задам всего **два ключевых ориентира**:\n\n"
                "1️⃣ **Какой формат отдыха вам ближе?**\n"
                "• 🌊 **Пляжный релакс и рифы** (Мальдивы, Пхукет/Самуи, Египет, ОАЭ, Анталья/Каш/Аланья)\n"
                "• 🏔️ **Горы, природа и спа** (Алтай, Красная Поляна, Грузия / Гудаури / Казбеги)\n"
                "• 🏛️ **Культура, архитектура и гастрономия** (Санкт-Петербург, Стамбул, Тбилиси)\n"
                "• 👨‍👩‍👧 **Семейный All-Inclusive с детьми и аквапарками**\n\n"
                "2️⃣ **Примерный состав и даты поездки** (вдвоём, с семьёй, с питомцем или соло)?\n\n"
                "_Напишите, что вам ближе (например: «Хочу море и всё включено с ребенком»), "
                "и я мгновенно соберу готовый проверенный маршрут с перелетом, отелем и трансфером._"
            )

    @staticmethod
    def is_explicit_multi_city_request(message: str) -> bool:
        """
        Проверяет, запрашивал ли пользователь явно мульти-городской/комбинированный тур.
        Исключает ложные срабатывания на одиночные запросы вроде «Хочу на море».
        """
        m = message.lower()
        explicit_keywords = [
            "тур по городам", "по городам", "несколько городов", "нескольких городов",
            "много городов", "комбинированный тур", "комбинированн", "составной тур",
            "составной маршрут", "комбо тур", "комбо-тур", "маршрут по городам", "два города",
            "двум городам", "разные города", "несколько курортов", "совместить города", "маршрут из двух",
            "города у моря", "несколько городов у моря"
        ]
        if any(k in m for k in explicit_keywords):
            return True
        if bool(re.search(
            r"(?:по\s+(?:нескольк\w*\s+)?город\w*|нескольк\w*\s+(?:город\w*|курорт\w*)|мног\w*\s+город\w*|разн\w*\s+город\w*|комбинир\w*|составн\w*\s+(?:тур\w*|маршрут\w*)|комбо[- ]тур\w*)",
            m
        )):
            return True
        city_pairs = [
            ("сочи", "полян"), ("сириус", "полян"),
            ("побереж", "гор"), ("гор", "побереж"),
            ("сочи", "побереж"), ("побереж", "сочи"),
            ("побереж", "полян"), ("полян", "побереж"),
            ("анталь", "стамбул"), ("стамбул", "анталь"),
            ("анталь", "каш"), ("каш", "анталь"),
            ("анталь", "алань"), ("алань", "анталь"),
            ("дубай", "абу-даби"), ("абу-даби", "дубай"),
            ("дуба", "абу даби"), ("абу даби", "дуба"),
            ("манжерок", "чемал"), ("чемал", "манжерок"),
            ("горно-алтайск", "чемал"), ("чемал", "горно-алтайск"),
            ("пхукет", "краби"), ("краби", "пхукет"),
            ("бангкок", "пхукет"), ("пхукет", "бангкок"),
            ("бангкок", "самуи"), ("самуи", "бангкок"),
            ("тбилис", "гудаур"), ("гудаур", "тбилис"),
            ("тбилис", "казбег"), ("казбег", "тбилис"),
            ("тбилис", "степанцминд"), ("степанцминд", "тбилис"),
            ("тбилис", "батум"), ("батум", "тбилис"),
            ("питер", "петергоф"), ("петербург", "петергоф"),
            ("шарм", "дахаб"), ("дахаб", "шарм"),
            ("шарм", "хадаб"), ("хадаб", "шарм")
        ]
        return any(c1 in m and c2 in m for c1, c2 in city_pairs)

    @staticmethod
    def sanitize_ai_intro(intro: Optional[str], dest_id: str, default_intro: str, route_name: Optional[str] = None) -> str:
        """
        Проверяет, не противоречит ли реплика ИИ выбранному региону тура
        (например, когда модель упоминает Турцию при туре в Сочи).
        При обнаружении несоответствия заменяет реплику на проверенную системную.
        """
        if not intro:
            return default_intro
        intro_lower = intro.lower()
        if re.search(r"аудит\s+агента|smart\s+filter|pet[ -]контроль", intro_lower):
            return default_intro
        if dest_id == "sochi" and route_name == "Сочи" and re.search(r"красн\w*\s+полян|эсто[ -]садок|роза\s+хутор", intro_lower):
            return default_intro
        forbidden_map = {
            "sochi": ["турци", "стамбул", "анталь", "оаэ", "дуба", "египет", "мальдив", "тайланд", "таиланд", "пхукет", "питер", "петербург", "алтай", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок", "каш", "алань"],
            "turkey": ["сочи", "сириус", "полян", "адлер", "оаэ", "дуба", "египет", "мальдив", "тайланд", "таиланд", "пхукет", "питер", "петербург", "алтай", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок"],
            "uae": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "египет", "мальдив", "тайланд", "таиланд", "пхукет", "питер", "петербург", "алтай", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок", "каш", "алань"],
            "egypt": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "оаэ", "дуба", "мальдив", "тайланд", "таиланд", "пхукет", "питер", "петербург", "алтай", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок", "каш", "алань"],
            "maldives": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "оаэ", "дуба", "египет", "тайланд", "таиланд", "пхукет", "питер", "петербург", "алтай", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок", "каш", "алань"],
            "thailand": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "оаэ", "дуба", "египет", "мальдив", "питер", "петербург", "алтай", "грузи", "тбилис", "гудаур", "казбег", "каш", "алань"],
            "spb": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "оаэ", "дуба", "египет", "мальдив", "тайланд", "таиланд", "пхукет", "алтай", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок", "каш", "алань"],
            "altai": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "оаэ", "дуба", "египет", "мальдив", "тайланд", "таиланд", "пхукет", "питер", "петербург", "грузи", "тбилис", "гудаур", "казбег", "самуи", "бангкок", "каш", "алань"],
            "georgia": ["сочи", "сириус", "полян", "турци", "анталь", "стамбул", "оаэ", "дуба", "египет", "мальдив", "тайланд", "таиланд", "пхукет", "питер", "петербург", "алтай", "самуи", "бангкок", "каш", "алань"]
        }
        forbidden = forbidden_map.get(dest_id, [])
        if any(fw in intro_lower for fw in forbidden):
            return default_intro
        return intro

    @staticmethod
    def is_vague_greeting(message: str, has_cart_destination: bool = False) -> bool:
        """
        Проверяет, является ли сообщение пользователя неопределенным приветствием/стартовым запросом
        без указания направления (Пункт 4 ТЗ: Анти-15-вопросов).
        """
        m_lower = message.lower().strip()
        if re.fullmatch(r"\d{1,3}(?:[ .]\d{3})+(?:\s*(?:₽|руб\w*))?", m_lower):
            return False

        # Проверка на наличие конкретных стран/городов (включая Грузию и новые курорты)
        known_kws = [
            "сочи", "сириус", "полян", "адлер", "дуба", "оаэ", "абу-даби", "эмират",
            "турци", "анталь", "стамбул", "белек", "кемер", "каш", "калкан", "алань", "алани", "даламан", "газипаш", "сиде",
            "пхукет", "тай", "таиланд", "самуи", "бангкок", "сураттхани", "патонг", "карон", "ката", "раваи", "банг тао",
            "мальдив", "мале", "египет", "шарм", "хургад", "синай",
            "спб", "питер", "петербург", "санкт-петербург",
            "алтай", "манжерок", "телецк", "горно-алтайск",
            "грузи", "грузия", "тбилис", "гудаур", "казбег", "степанцминд", "батум"
        ]
        if any(k in m_lower for k in known_kws):
            return False

        # Если есть внешние страны, сработает fallback
        for item in UNKNOWN_DESTINATIONS_MAP:
            if any(k in m_lower for k in item["keywords"]):
                return False

        # Если есть параметры/команды перепланирования или фильтров — это не приветствие
        replan_keywords = [
            "другой отель", "смени отель", "поменяй отель", "дешев", "дорог", "попозже", "пораньше",
            "убери трансфер", "без трансфера", "забронир", "оформ", "оплат", "купит"
        ]
        if any(rw in m_lower for rw in replan_keywords):
            return False

        # Если указаны конкретные календарные даты
        if re.search(r"\b\d{1,2}\.\d{1,2}\b", m_lower) or re.search(r"\b\d{4}-\d{2}-\d{2}\b", m_lower):
            return False

        # Если указан бюджет
        if re.search(r"\b(?:\d+[\s\d]*(?:тыс|к|руб|₽|млн)|бюджет|до \d+)\b", m_lower):
            return False

        vague_phrases = [
            "привет", "здравствуйте", "добрый день", "добрый вечер", "хай", "hello",
            "хочу отдохнуть", "помоги выбрать", "посоветуй", "куда поехать", "не знаю куда",
            "хочу в отпуск", "подбери тур", "что посоветуешь", "помоги с выбором", "начать",
            "хочу поехать", "хочу слетать", "подобрать тур", "куда слетать", "планирую отдых",
            "планирую отпуск", "посоветуйте", "ищу тур", "хочу путешествовать", "помоги с поездкой",
            "хочу в путешествие", "поехать куда-нибудь", "где лучше отдохнуть", "какие есть варианты",
            "помоги спланировать", "выбрать тур", "собери тур",
            "хочу на море", "на море", "к морю", "хочу на пляж", "на пляж", "пляжный отдых", "хочу искупаться",
            "хочу в тепло", "в тепло", "хочу в горы", "в горы", "горный отдых", "покататься на лыжах"
        ]
        # Если совпала явная стартовая фраза из списка — срабатываем БЕЗУСЛОВНО (даже если в старой корзине что-то лежало)
        if any(vp in m_lower for vp in vague_phrases):
            return True

        # Если направление уже выбрано в корзине и это не стартовая фраза
        if has_cart_destination:
            return False

        if len(m_lower.split()) <= 4 and not any(k in m_lower for k in known_kws):
            return True

        return False

    @staticmethod
    def answer_internal_route_timing(user_lower: str, cart: dict) -> Optional[str]:
        """
        Проверяет, спрашивает ли пользователь о расписании / времени внутреннего маршрута
        (начало, конец, длительность рейса/поезда/трансфера между городами).
        """
        time_keywords = [
            "во сколько", "какое время", "время вылета", "время прилета", "время отправления",
            "время прибытия", "расписание", "когда вылет", "когда отправление", "когда прилет",
            "начало и конец", "во сколько начало", "во сколько конец", "длительность",
            "сколько лететь", "сколько ехать", "график", "время маршрута", "время рейса",
            "время поезда", "время переезда", "время трансфера", "время внутреннего"
        ]
        has_time_kw = any(tk in user_lower for tk in time_keywords) or (
            any(w in user_lower for w in ["время", "во сколько", "когда", "тайминг", "часы", "час"]) and
            any(w in user_lower for w in ["начал", "конец", "конца", "прибыт", "вылет", "отправлен", "маршрут", "рейс", "поезд", "переезд", "трансфер"])
        )
        if not has_time_kw:
            return None

        route_keywords = [
            "внутренн", "между город", "междугород", "переезд", "ласточк", "между отел",
            "между част", "анталья и стамбул", "из антальи в стамбул", "из стамбула",
            "сочи и поляна", "из сочи в поляну", "в красную поляну",
            "дубай и абу-даби", "из дубая в абу-даби",
            "горно-алтайск и чемал", "из горно-алтайска в чемал",
            "шарм и дахаб", "из наама бей", "петергоф", "метеор"
        ]
        has_route_kw = any(rk in user_lower for rk in route_keywords) or cart.get("is_multi_city")
        if not (has_time_kw and has_route_kw):
            return None

        # Определяем направление мульти-городского тура
        dest_id = cart.get("destination_id")
        if not dest_id or dest_id not in MULTI_CITY_CONFIG:
            if any(w in user_lower for w in ["анталь", "стамбул", "турци"]):
                dest_id = "turkey"
            elif any(w in user_lower for w in ["сочи", "полян", "сириус"]):
                dest_id = "sochi"
            elif any(w in user_lower for w in ["дуба", "абу-даби", "оаэ"]):
                dest_id = "uae"
            elif any(w in user_lower for w in ["алтай", "чемал", "катунь"]):
                dest_id = "altai"
            elif any(w in user_lower for w in ["питер", "петербург", "петергоф"]):
                dest_id = "spb"
            elif any(w in user_lower for w in ["египет", "шарм", "хадаб"]):
                dest_id = "egypt"
            elif cart.get("is_multi_city"):
                dest_id = "turkey"
            else:
                return None

        cfg = MULTI_CITY_CONFIG[dest_id]
        inter = cfg["intercity"]
        city_1 = cfg["city_1"]["name"]
        city_2 = cfg["city_2"]["name"]

        reply = (
            f"🕒 **Расписание и тайминг внутреннего маршрута ({city_1} ➔ {city_2}):**\n\n"
            f"• 🚆/✈️ **Транспорт:** {inter['name']}\n"
            f"• 🛫 **Начало (отправление):** **{inter.get('departure', '11:30')}**\n"
            f"• 🛬 **Конец (прибытие):** **{inter.get('arrival', '12:45')}**\n"
            f"• ⏱️ **Время в пути:** **{inter.get('duration', '1 ч 15 мин')}**\n"
            f"• 🏢 **Перевозчик:** {inter.get('operator', 'Партнёр Островок')}\n"
            f"• ℹ️ **Маршрут:** {inter.get('description', '')}\n\n"
            f"✅ **Сквозная стыковка с отелями:** График рассчитан идеально: "
            f"вы выселяетесь из первого отеля в {city_1} после завтрака (до 11:00/12:00), "
            f"комфортно совершаете переезд и прибываете во второй отель в {city_2} "
            f"к началу стандартного заселения (после 14:00)."
        )
        return reply

    @staticmethod
    async def search_destinations_vector(
        query: str,
        limit: int = 2,
        abroad_only: bool = False,
        domestic_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Гибридный поиск: прямое совпадение по ключевым словам городов/стран +
        векторный поиск по вайбу и климату в PostgreSQL через pgvector с поддержкой фильтрации заграница/РФ.
        """
        q_lower = query.lower()
        sea_only = any(w in q_lower for w in ["море", "морю", "морск", "пляж", "побереж", "купаться", "позагорать"])
        sea_destination_ids = ("turkey", "sochi", "uae", "thailand", "maldives", "egypt")
        
        matched_id = requested_destination(query)

        # Если запрошена строго заграница — отсекаем города РФ
        if abroad_only and matched_id in ("sochi", "spb", "altai"):
            matched_id = None
        # Если запрошена строго Россия — отсекаем зарубежные курорты
        if domestic_only and matched_id in ("uae", "turkey", "thailand", "maldives", "egypt"):
            matched_id = None
        if sea_only and matched_id not in sea_destination_ids:
            matched_id = None

        async with async_session_maker() as session:
            if matched_id:
                res = await session.execute(
                    select(DestinationModel).where(DestinationModel.id == matched_id)
                )
                r = res.scalar_one_or_none()
                if r:
                    return [{
                        "id": r.id,
                        "name": r.name,
                        "country": r.country,
                        "price_min": r.price_min,
                        "price_max": r.price_max,
                        "avg_check": r.avg_check,
                        "vibe_description": r.vibe_description,
                        "climate_summary": r.climate_summary,
                        "score": 1.0
                    }]

            query_vector = generate_text_embedding(query)
            str_vector = f"[{','.join(str(x) for x in query_vector)}]"
            
            where_clauses = []
            if abroad_only:
                where_clauses.append("country != 'Россия'")
            elif domestic_only:
                where_clauses.append("country = 'Россия'")
            if sea_only:
                where_clauses.append("id IN ('turkey', 'sochi', 'uae', 'thailand', 'maldives', 'egypt')")
            
            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            sql = text(f"""
                SELECT id, name, country, price_min, price_max, avg_check, vibe_description, climate_summary,
                       embedding <=> (:query_vector)::vector AS distance
                FROM destinations
                {where_sql}
                ORDER BY distance ASC
                LIMIT :limit
            """)
            
            result = await session.execute(sql, {"query_vector": str_vector, "limit": limit})
            rows = result.fetchall()
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "country": r.country,
                    "price_min": r.price_min,
                    "price_max": r.price_max,
                    "avg_check": r.avg_check,
                    "vibe_description": r.vibe_description,
                    "climate_summary": r.climate_summary,
                    "score": round(1.0 - float(r.distance), 3)
                }
                for r in rows
            ]

    @staticmethod
    def select_multi_city_route(
        user_message: str,
        profile: Dict[str, Any],
        current_dest_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Интеллектуальный параметрический подбор связки городов и маршрута мульти-городского тура.
        Анализирует параметры запроса (климат, пляж/горы, культура/шопинг, природа, ограничения),
        проверяет связки и генерирует персональное обоснование выбора.
        """
        user_lower = user_message.lower()
        interests = [str(x).lower() for x in profile.get("interests", [])]
        budget_limit = profile.get("budget_limit")
        domestic_only = profile.get("domestic_only", False) or any(w in user_lower for w in ["росси", "по рф", "без загран", "внутри страны"])

        scores: Dict[str, float] = {k: 0.0 for k in MULTI_CITY_CONFIG}
        matched_reasons: Dict[str, str] = {}

        # 1. Прямые упоминания городов или стран (наивысший приоритет)
        geo_keywords = {
            "turkey": ["турци", "анталь", "стамбул", "босфор"],
            "sochi": ["сочи", "сириус", "полян", "адлер", "имеретинк"],
            "uae": ["дуба", "абу-даби", "абу даби", "оаэ", "эмират"],
            "altai": ["алтай", "чемал", "манжерок", "катунь", "майма", "горно-алтайск"],
            "thailand": ["тайланд", "таиланд", "пхукет", "краби", "патонг", "банг тао"],
            "spb": ["петербург", "питер", "спб", "петергоф", "эрмитаж", "царское село"],
            "egypt": ["египет", "шарм", "дахаб", "наама", "хадаб"]
        }
        for dest_key, kws in geo_keywords.items():
            if any(kw in user_lower for kw in kws):
                scores[dest_key] += 60.0
                matched_reasons[dest_key] = "указанный вами маршрут"

        # 2. Параметрический анализ связок (вайб, специфика сочетания городов)
        has_sea = any(w in user_lower for w in ["море", "пляж", "морск", "побереж", "купат"]) or any("sea" in i or "beach" in i for i in interests)
        if has_sea:
            for non_sea_dest in ("altai", "spb", "georgia"):
                scores[non_sea_dest] = -10000.0
        has_mountains = any(w in user_lower for w in ["гор", "хайкинг", "канатк", "лыж", "полян", "вершин", "альпийск"]) or any("mountain" in i for i in interests)
        if has_sea and has_mountains:
            scores["sochi"] += 45.0
            matched_reasons["sochi"] = "сочетание теплого моря и горных панорам"

        has_culture = any(w in user_lower for w in ["культур", "истори", "архитектур", "босфор", "старый город", "достопримечательн", "экскурси"]) or any("culture" in i for i in interests)
        if has_sea and has_culture:
            scores["turkey"] += 40.0
            matched_reasons["turkey"] = "сочетание средиземноморского пляжа и богатейшей культуры"
        if any(w in user_lower for w in ["много городов у моря", "города у моря", "несколько городов у моря"]):
            scores["turkey"] += 35.0
            matched_reasons["turkey"] = "маршрут по приморским городам"

        has_shopping = any(w in user_lower for w in ["шопинг", "шоппинг", "небоскреб", "молл", "роскош", "футуризм", "мегаполис"]) or any("shopping" in i for i in interests)
        if has_shopping:
            scores["uae"] += 40.0
            matched_reasons["uae"] = "футуристические мегаполисы, шопинг и роскошный отдых"

        has_nature_wild = any(w in user_lower for w in ["природ", "река", "реки", "катунь", "кедр", "места силы", "эко", "чистый воздух", "озер"]) or any("nature" in i for i in interests)
        if has_nature_wild and not has_sea:
            scores["altai"] += 40.0
            matched_reasons["altai"] = "первозданная природа, бирюзовые реки и хвойный воздух"

        has_palaces = any(w in user_lower for w in ["дворц", "музе", "эрмитаж", "фонтан", "канал", "театр", "северная столица"])
        if has_palaces:
            scores["spb"] += 40.0
            matched_reasons["spb"] = "классическая архитектура, шедевры Эрмитажа и дворцовые парки"

        has_corals = any(w in user_lower for w in ["коралл", "риф", "дайвинг", "снорклинг", "рыбк", "подводн", "красное море"])
        if has_corals:
            scores["egypt"] += 45.0
            matched_reasons["egypt"] = "живые коралловые рифы и теплейшее Красное море"

        has_exotic = any(w in user_lower for w in ["остров", "острова", "экзотик", "джунгл", "тропик", "спидбот", "буддизм"])
        if has_exotic:
            scores["thailand"] += 40.0
            matched_reasons["thailand"] = "тропические пляжи, острова и экзотическая атмосфера"

        # 3. Фильтрация и веса ограничений (Бюджет и Domestic)
        if domestic_only:
            for foreign_key in ["turkey", "uae", "thailand", "egypt"]:
                scores[foreign_key] -= 100.0

        if budget_limit and budget_limit < 90000:
            scores["sochi"] += 15.0
            scores["spb"] += 15.0
            scores["altai"] += 10.0
            scores["thailand"] -= 20.0
            scores["uae"] -= 20.0

        # Преемственность текущего региона, если он уже был выбран
        if current_dest_id and current_dest_id in scores:
            scores[current_dest_id] += 5.0

        best_dest = max(scores, key=scores.get)
        if scores[best_dest] <= 0 and not current_dest_id:
            best_dest = "turkey"

        best_cfg = MULTI_CITY_CONFIG.get(best_dest, MULTI_CITY_CONFIG["turkey"])
        criteria_reason = matched_reasons.get(best_dest, "ваши пожелания к комбинированному маршруту")
        template = best_cfg.get("selection_rationale", "Собрал для вас готовый комбинированный тур «{title}» под ключ с логистикой между городами, отелями и расчётом полной стоимости:")
        rationale = template.format(criteria=criteria_reason, title=best_cfg["title"])

        return best_dest, best_cfg, rationale

    @staticmethod
    async def assemble_multi_city_itinerary(
        dest_id: str,
        top_dest: Dict[str, Any],
        profile: Dict[str, Any],
        cart: Dict[str, Any],
        nights: int,
        budget_limit: Optional[int],
        user_id: str,
        message: str,
        is_replan_flight_day: bool,
        is_replan_flight_later: bool,
        session_obj: Any,
        selection_rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Формирует полноценный комбинированный маршрут тура по нескольким городам (Multi-City Tour).
        Подбирает проживание в обоих ключевых городах региона, рассчитывает межгородскую логистику
        (внутренний авиарейс через Neo4j, поезд «Ласточка» или комфорт-трансфер) и международные авиаперелёты.
        """
        cfg = MULTI_CITY_CONFIG.get(dest_id, MULTI_CITY_CONFIG["turkey"])
        nights_1 = max(1, int(round(nights * cfg["city_1"]["nights_share"])))
        nights_2 = max(1, nights - nights_1)
        user_lower = message.lower().strip()

        is_cheapest_requested = any(w in user_lower for w in [
            "самый дешев", "самые дешев", "самая дешев", "самое дешев",
            "самый эконом", "самый бюджет", "минимальн", "как можно дешев",
            "максимально дешев", "подешевле", "подешевле вариант", "дешевый вариант", "дешевые билеты", "дешевыми билетами"
        ])

        # 1. Подбор отелей для обоих городов через MCP
        h_res_1 = await MCPTools.compare_hotels(
            destination_id=dest_id,
            city=cfg["city_1"]["name"].split("/")[0].strip(),
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids"),
            private_room_required=profile.get("private_room_required", False),
            preferred_tags=profile.get("interests"),
            limit=None if budget_limit is not None else 30
        )
        hotels_1 = h_res_1.get("compared_hotels", [])

        h_res_2 = await MCPTools.compare_hotels(
            destination_id=dest_id,
            city=cfg["city_2"]["name"].split("/")[0].strip(),
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids"),
            private_room_required=profile.get("private_room_required", False),
            preferred_tags=profile.get("interests"),
            limit=None if budget_limit is not None else 30
        )
        hotels_2 = h_res_2.get("compared_hotels", [])

        is_luxury = any(t in ("luxury", "extreme_luxury") for t in profile.get("interests", []))
        is_budget = "budget_friendly" in profile.get("interests", [])

        # Если отель не найден по точному фильтру города — берем альтернативы из общего каталога
        if not hotels_1 or not hotels_2:
            all_h = (await MCPTools.compare_hotels(
                destination_id=dest_id,
                pet_friendly=profile.get("has_pets"),
                family_friendly=profile.get("has_kids"),
                private_room_required=profile.get("private_room_required", False),
                preferred_tags=profile.get("interests"),
                limit=None if budget_limit is not None else 30
            )).get("compared_hotels", [])
            if not hotels_1 and all_h:
                hotels_1 = [all_h[0]]
            if not hotels_2:
                hotels_2 = [h for h in all_h if not hotels_1 or h["id"] != hotels_1[0]["id"]] or hotels_1

        if not hotels_1 or not hotels_2:
            reply_text = "Не нашлось отелей для обеих частей маршрута под ваши условия. Можно изменить города или условия поиска."
            cart = {"destination_id": dest_id, "destination": cfg["title"], "nights": nights,
                    "dates": profile.get("dates"), "selection_status": "no_matches"}
            await AgentService._save_session(user_id, profile, cart, message, reply_text)
            return {"text": reply_text, "state": "NO_MATCHES", "cart": cart, "compared_hotels": []}

        hotels_1, ltr_meta_1 = LTRService.rank_hotel_candidates(hotels_1, dest_id, profile, nights_1)
        hotels_2, ltr_meta_2 = LTRService.rank_hotel_candidates(hotels_2, dest_id, profile, nights_2)
        policy = "ltr"
        if is_luxury and not is_cheapest_requested:
            policy = "luxury"
            hotels_1.sort(key=lambda h: (-h.get("price_per_night_rub", 0), -h["ltr_score"]))
            hotels_2.sort(key=lambda h: (-h.get("price_per_night_rub", 0), -h["ltr_score"]))
        elif is_cheapest_requested or is_budget or budget_limit is not None:
            policy = "budget"
            hotels_1.sort(key=lambda h: (h.get("price_per_night_rub", 0), -h["ltr_score"]))
            hotels_2.sort(key=lambda h: (h.get("price_per_night_rub", 0), -h["ltr_score"]))
        hotel_1, hotel_2 = hotels_1[0], hotels_2[0]
        for meta, hotel in ((ltr_meta_1, hotel_1), (ltr_meta_2, hotel_2)):
            meta.update(top_k=1, top_score=hotel["ltr_score"], top_hotel=hotel["name"], selection_policy=policy)
        ltr_meta = {
            "engine": ltr_meta_1["engine"] if ltr_meta_1["engine"] == ltr_meta_2["engine"] else "Mixed ranking engines",
            "archetype": ltr_meta_1["archetype"], "selection_policy": policy,
            "candidates_count": ltr_meta_1["candidates_count"] + ltr_meta_2["candidates_count"],
            "latency_ms": round(ltr_meta_1["latency_ms"] + ltr_meta_2["latency_ms"], 2),
            "top_k": 2, "cities": [ltr_meta_1, ltr_meta_2],
        }

        # 2. Логистика: международный перелёт с учетом пожеланий по времени (день / вечер / утро)
        logistics_1 = await MCPTools.check_logistics_compatibility(
            airport_code=cfg["city_1"]["airport"],
            hotel_id=hotel_1.get("id", ""),
            origin_airport_code=profile.get("origin_airport"),
            prefer_direct_flight=profile.get("prefer_direct_flight"),
            without_transfer=profile.get("without_transfer", False),
            flight_time_pref=profile.get("flight_time_pref")
        )
        outbound_flight = logistics_1.get("flight", {})

        # Проверяем в графе Neo4j наличие прямого межгородского транспортного ребра
        graph_intercity = await GraphService.find_intercity_logistics(
            cfg["city_1"]["airport"],
            cfg["city_2"]["airport"]
        )
        if graph_intercity:
            intercity = {
                "type": "flight",
                "name": f"Внутренний авиарейс {graph_intercity.get('flight_no', 'TK-2415')} {cfg['city_1']['airport']} ➔ {cfg['city_2']['airport']}",
                "operator": graph_intercity.get("airline", "Turkish Airlines"),
                "departure": graph_intercity.get("departure_time", "11:30"),
                "arrival": graph_intercity.get("arrival_time", "12:45"),
                "duration": f"{graph_intercity.get('duration_minutes', 75)} мин",
                "price_rub": graph_intercity.get("price_rub", 3800),
                "description": f"Прямой авиаперелёт {graph_intercity.get('airline')} ({graph_intercity.get('flight_no')}) между городами подтверждён графом логистики Neo4j."
            }
        else:
            intercity = cfg["intercity"]

        ret_fl = outbound_flight.get("return_flight", {})
        ret_airline = ret_fl.get("airline", outbound_flight.get("airline", "Аэрофлот"))
        ret_no = ret_fl.get("flight_no", "SU-2137")
        ret_dep = ret_fl.get("departure_time", "15:45")
        ret_arr = ret_fl.get("arrival_time", "20:30")
        dest_code = cfg["city_2"]["airport"]
        orig_code = profile.get("origin_airport") or outbound_flight.get("origin_code", "SVO")

        # 3. Даты и логистика
        is_any_time = any(w in user_lower for w in [
            "в любое время", "в любой день", "любое время", "время не важно", "не важно когда",
            "без разницы во сколько", "в любое время суток", "любое время вылета", "когда угодно"
        ])

        if not profile.get("checkin_date"):
            today = date.today()
            base_date = today + timedelta(days=14)
            profile["checkin_date"] = base_date.isoformat()
            profile["checkout_date"] = (base_date + timedelta(days=nights)).isoformat()
            profile["dates"] = f"{base_date:%d.%m.%Y} — {base_date + timedelta(days=nights):%d.%m.%Y} ({nights} ночей)"

        rooms_cnt = max(1, profile.get("rooms_count", 1))
        travelers_count = max(1, profile.get("adults_count", 2) + profile.get("kids_count", 0))
        raw_flight_rt = outbound_flight.get("price_rub_roundtrip", 27000)
        best_mc_flex = None

        if (is_cheapest_requested or is_any_time) and not profile.get("date_shifted"):
            avg_hotel_price = int((hotel_1.get("price_per_night_rub", 6000) * nights_1 + hotel_2.get("price_per_night_rub", 8500) * nights_2) / max(1, nights))
            intercity_cost = intercity.get("price_rub", 3800) * travelers_count
            tr_cost = 0 if profile.get("without_transfer") else 3500
            pf1 = (hotel_1.get("pet_surcharge_per_night_rub", 0) or 0) * nights_1 if profile.get("has_pets") else 0
            pf2 = (hotel_2.get("pet_surcharge_per_night_rub", 0) or 0) * nights_2 if profile.get("has_pets") else 0
            fixed_mc = intercity_cost + tr_cost + pf1 + pf2
            flex_pre = AgentService.calculate_flexible_dates_options(
                dest_id=dest_id,
                base_mandatory_rub=avg_hotel_price * nights * rooms_cnt + raw_flight_rt * travelers_count + fixed_mc,
                checkin_date_str=profile.get("checkin_date"),
                nights=nights,
                base_flight_price=raw_flight_rt,
                base_hotel_price=avg_hotel_price,
                fixed_costs=fixed_mc,
                rooms_count=rooms_cnt,
                travelers_count=travelers_count
            )
            if flex_pre and flex_pre[0]["savings_rub"] > 0:
                best_mc_flex = flex_pre[0]
                profile["checkin_date"] = best_mc_flex["checkin_date"]
                profile["checkout_date"] = best_mc_flex["checkout_date"]
                profile["dates"] = f"{best_mc_flex['dates_formatted']} ({nights} ночей)"
                profile["date_shifted"] = True
                cart["date_shifted"] = True
                cart["checkin_date"] = best_mc_flex["checkin_date"]
                cart["checkout_date"] = best_mc_flex["checkout_date"]
                cart["dates"] = profile["dates"]

        d_in = date.fromisoformat(profile["checkin_date"])
        d_mid = d_in + timedelta(days=nights_1)
        d_out = date.fromisoformat(profile["checkout_date"])

        date_out_fmt = format_flight_date(d_in.isoformat())
        date_mid_fmt = format_flight_date(d_mid.isoformat())
        date_ret_fmt = format_flight_date(d_out.isoformat())
        date_range_str = format_russian_date_range(d_in.isoformat(), d_out.isoformat(), nights)

        hotel_1_dates = format_hotel_dates(d_in.isoformat(), d_mid.isoformat(), nights_1)
        hotel_2_dates = format_hotel_dates(d_mid.isoformat(), d_out.isoformat(), nights_2)

        # 4. Финансовый расчёт всего тура с учётом реального календарного тарифа
        is_date_shifted = bool(profile.get("date_shifted") or cart.get("date_shifted"))

        # Реальный тариф на авиарейсы
        raw_flight_rt = outbound_flight.get("price_rub_roundtrip", 27000)
        base_flight_half = raw_flight_rt * 0.5
        if is_date_shifted:
            f_out = AgentService.FLIGHT_DAY_FACTORS.get(d_in.weekday(), 1.0)
            f_ret = AgentService.FLIGHT_DAY_FACTORS.get(d_out.weekday(), 1.0)
            flight_out_price = int(round(base_flight_half * f_out / 100) * 100) * travelers_count
            flight_ret_price = int(round(base_flight_half * f_ret / 100) * 100) * travelers_count
            intl_flight_price = flight_out_price + flight_ret_price

            # Реальный тариф на отели
            h1_factors = [AgentService.HOTEL_NIGHT_FACTORS.get((d_in + timedelta(days=i)).weekday(), 1.0) for i in range(nights_1)]
            avg_h1 = sum(h1_factors) / max(1, nights_1)
            rate_h1 = int(round(hotel_1.get("price_per_night_rub", 6000) * avg_h1 / 100) * 100)
            cost_hotel_1 = rate_h1 * nights_1 * rooms_cnt

            h2_factors = [AgentService.HOTEL_NIGHT_FACTORS.get((d_mid + timedelta(days=i)).weekday(), 1.0) for i in range(nights_2)]
            avg_h2 = sum(h2_factors) / max(1, nights_2)
            rate_h2 = int(round(hotel_2.get("price_per_night_rub", 8500) * avg_h2 / 100) * 100)
            cost_hotel_2 = rate_h2 * nights_2 * rooms_cnt
        else:
            intl_flight_price = raw_flight_rt * travelers_count
            flight_out_price = int(round(intl_flight_price * 0.5))
            flight_ret_price = int(intl_flight_price - flight_out_price)
            cost_hotel_1 = hotel_1.get("price_per_night_rub", 6000) * nights_1 * rooms_cnt
            cost_hotel_2 = hotel_2.get("price_per_night_rub", 8500) * nights_2 * rooms_cnt

        intercity_price = intercity.get("price_rub", 3800) * travelers_count
        transfers_price = 0 if profile.get("without_transfer") else 3500
        tr_1_price = transfers_price // 2
        tr_2_price = transfers_price - tr_1_price

        pet_fee_1 = (hotel_1.get("pet_surcharge_per_night_rub", 0) or 0) * nights_1 if profile.get("has_pets") else 0
        pet_fee_2 = (hotel_2.get("pet_surcharge_per_night_rub", 0) or 0) * nights_2 if profile.get("has_pets") else 0
        pet_total = pet_fee_1 + pet_fee_2

        mand_total = cost_hotel_1 + cost_hotel_2 + intl_flight_price + intercity_price + transfers_price + pet_total
        onsite_min = nights * 2500
        onsite_max = nights * 5500
        total_min = mand_total + onsite_min
        total_max = mand_total + onsite_max

        # 5. Формирование ответа
        response_lines = []
        raw_intro = profile.pop("llm_ai_intro", None) or selection_rationale
        out_dep = outbound_flight.get("departure_time", "13:40")
        if is_cheapest_requested:
            if best_mc_flex:
                raw_intro = f"Подобрал для вас самый дешёвый вариант комбинированного тура под ключ: сдвинул вылет на {abs(best_mc_flex['days_shift'])} дня {best_mc_flex['direction']} ({best_mc_flex['target_weekday']}, {best_mc_flex['dates_formatted']}), чтобы билеты и отели вышли по минимальному тарифу:"
            else:
                raw_intro = "Собрал для вас самый экономичный вариант комбинированного тура под ключ с самыми дешёвыми билетами и отелями:"
        elif not raw_intro:
            if budget_limit is not None and total_max > budget_limit:
                raw_intro = (
                    f"⚠️ К сожалению, с бюджетом {budget_limit:,} ₽ составить комбинированный тур невозможно. "
                    f"Ближайший доступный вариант — {total_max:,} ₽ (на {total_max - budget_limit:,} ₽ выше лимита). "
                    f"Ниже подобран самый экономичный вариант под ключ:"
                )
            elif is_replan_flight_day:
                raw_intro = f"Понял вас — никаких подъёмов в 6 утра: подобрал дневной рейс в {out_dep}, чтобы можно было выспаться. Собрал полноценный комбинированный тур по обоим городам:"
            elif is_replan_flight_later:
                raw_intro = f"Принято! Обновил время вылета на комфортное дневное ({out_dep}) и пересчитал программу тура по городам:"
            else:
                raw_intro = f"Собрал для вас готовый комбинированный тур «{cfg['title']}» под ключ с логистикой между городами, отелями и расчётом полной стоимости:"

        ai_intro = AgentService.sanitize_ai_intro(
            intro=raw_intro,
            dest_id=dest_id,
            default_intro=f"Собрал для вас готовый комбинированный тур «{cfg['title']}» под ключ с логистикой между городами, отелями и расчётом полной стоимости:"
        )

        response_lines.append(f"💬 {ai_intro}")
        response_lines.append("")
        response_lines.append(f"🌴 **Комбинированный тур: {cfg['title']}** | {cfg['climate']}")
        response_lines.append(f"📅 **Даты поездки:** {date_range_str} *(даты можно изменить: напишите «смени даты на ...»)*")
        response_lines.append(f"🌤️ **Прогноз погоды по маршруту:** {cfg['city_1']['name']}: {cfg['city_1']['weather']} | {cfg['city_2']['name']}: {cfg['city_2']['weather']}")
        response_lines.append("")

        response_lines.append(f"🗺️ **Программа и маршрут тура по городам ({nights} ночей):**")
        response_lines.append("")

        fl_code_out = f"{outbound_flight.get('airline', 'Авиа')} {outbound_flight.get('flight_no', '')}"
        times_out = f"{outbound_flight.get('origin_code', 'MOW')} {outbound_flight.get('departure_time', '13:40')} ➔ {cfg['city_1']['airport']} {outbound_flight.get('arrival_time', '18:35')}"
        fl_code_ret = f"{ret_airline} {ret_no}"
        times_ret = f"{dest_code} {ret_dep} ➔ {orig_code} {ret_arr}"

        # 1. Перелет туда
        response_lines.append(f"1️⃣ 🛫 **Перелёт туда в {cfg['city_1']['name']}{date_out_fmt}:** Прямой рейс {fl_code_out} ({times_out}) — **{flight_out_price:,} ₽**")

        # 2. Трансфер в отель 1
        if profile.get("without_transfer"):
            response_lines.append(f"2️⃣ 🧳 **Трансфер в отель 1:** Самостоятельно на такси (Трансферы исключены по вашему запросу) — **0 ₽**")
        else:
            response_lines.append(f"2️⃣ 🚗 **Трансфер в отель 1:** Аэропорт {cfg['city_1']['airport']} ➔ отель (~30 мин) — **{tr_1_price:,} ₽**")

        # 3. Отель 1
        h1_formatted = hotel_1.get('price_per_night_formatted', f"{hotel_1.get('price_per_night_rub', 6000):,} ₽")
        if rooms_cnt > 1:
            response_lines.append(f"3️⃣ 🏨 **Часть 1: Отель в {cfg['city_1']['name']} ({nights_1} ночи/ночей, {hotel_1_dates}):** **«{hotel_1['name']}»** ({hotel_1['stars']}) — **{h1_formatted}/ночь за номер** ({cost_hotel_1:,} ₽ за {rooms_cnt} ном. на {nights_1} н.) — **{cost_hotel_1:,} ₽**")
            response_lines.append(f"• 🛏️ **Размещение:** {profile.get('room_type', 'смежные номера')}.")
        else:
            response_lines.append(f"3️⃣ 🏨 **Часть 1: Отель в {cfg['city_1']['name']} ({nights_1} ночи/ночей, {hotel_1_dates}):** **«{hotel_1['name']}»** ({hotel_1['stars']}) — **{h1_formatted}/ночь** ({cost_hotel_1:,} ₽ за {nights_1} ночей) — **{cost_hotel_1:,} ₽**")
        response_lines.append(f"• Питание: {hotel_1.get('meals', 'Завтрак')} | До моря: {hotel_1.get('distance_to_sea', 'Первая линия')} | Отмена: {hotel_1.get('cancellation', 'Бесплатно')}")
        response_lines.append(f"• 🏖️ **Вайб и программа:** {cfg['city_1']['activities']}.")

        # 4. Переезд между городами
        response_lines.append(f"4️⃣ 🚄 / ✈️ **Переезд между городами{date_mid_fmt}:** **{intercity['name']}** ({intercity['duration']}): {intercity['description']} — **{intercity_price:,} ₽**")

        # 5. Отель 2
        h2_formatted = hotel_2.get('price_per_night_formatted', f"{hotel_2.get('price_per_night_rub', 8500):,} ₽")
        if rooms_cnt > 1:
            response_lines.append(f"5️⃣ 🏨 **Часть 2: Отель в {cfg['city_2']['name']} ({nights_2} ночи/ночей, {hotel_2_dates}):** **«{hotel_2['name']}»** ({hotel_2['stars']}) — **{h2_formatted}/ночь за номер** ({cost_hotel_2:,} ₽ за {rooms_cnt} ном. на {nights_2} н.) — **{cost_hotel_2:,} ₽**")
            response_lines.append(f"• 🛏️ **Размещение:** {profile.get('room_type', 'смежные номера')}.")
        else:
            response_lines.append(f"5️⃣ 🏨 **Часть 2: Отель в {cfg['city_2']['name']} ({nights_2} ночи/ночей, {hotel_2_dates}):** **«{hotel_2['name']}»** ({hotel_2['stars']}) — **{h2_formatted}/ночь** ({cost_hotel_2:,} ₽ за {nights_2} ночей) — **{cost_hotel_2:,} ₽**")
        response_lines.append(f"• Питание: {hotel_2.get('meals', 'Завтрак')} | Отмена: {hotel_2.get('cancellation', 'Бесплатно')}")
        response_lines.append(f"• 🏛️ **Вайб и программа:** {cfg['city_2']['activities']}.")

        # 6. Трансфер в аэропорт 2
        if profile.get("without_transfer"):
            response_lines.append(f"6️⃣ 🧳 **Трансфер в аэропорт:** Самостоятельно на такси (Трансферы исключены по вашему запросу) — **0 ₽**")
        else:
            response_lines.append(f"6️⃣ 🚗 **Трансфер в аэропорт {cfg['city_2']['name']}:** Отель ➔ аэропорт {cfg['city_2']['airport']} (~30 мин) — **{tr_2_price:,} ₽**")

        # 7. Перелет обратно
        response_lines.append(f"7️⃣ 🛬 **Перелёт обратно из {cfg['city_2']['name']}{date_ret_fmt}:** {fl_code_ret} ({times_ret}) — **{flight_ret_price:,} ₽**")
        response_lines.append("")

        # Финансовый блок
        response_lines.append(f"💰 **Расчёт стоимости всей поездки (комбинированный тур на {nights} н.):**")
        response_lines.append(f"• 🔴 **Обязательные расходы (все перелёты + 2 отеля на {nights} н. + переезд между городами + трансферы):** {mand_total:,} ₽")
        if profile.get("has_pets") and pet_total > 0:
            response_lines.append(f"• 🐾 **Доплата за питомца (в обоих отелях):** {pet_total:,} ₽ (за {nights} н.) — включена в обязательные расходы")
        response_lines.append(f"• 🟢 **Расходы на месте ({nights} дн. питание + досуг в обоих городах):** {onsite_min:,} – {onsite_max:,} ₽")
        response_lines.append(f"💵 **ИТОГО под ключ:** **{total_min:,} – {total_max:,} ₽**")
        response_lines.append("")

        if budget_limit is not None and total_max > budget_limit:
            response_lines.append(
                f"⚠️ Лимит всей поездки: {budget_limit:,} ₽ превышен на {total_max - budget_limit:,} ₽. "
                f"Верхняя оценка расходов {total_max:,} ₽ — это ближайший доступный вариант для комбинированного тура. "
                f"Рекомендуем увеличить бюджет или сократить количество дней.\n"
            )

        # Паспортный и визовый комплаенс
        compliance_info = ComplianceService.get_compliance_info(dest_id, profile)
        response_lines.append(compliance_info["summary"])
        response_lines.append("")

        # Календарь выгоды (Гибкие даты ±2 дня)
        avg_hotel_price = int((cost_hotel_1 + cost_hotel_2) / max(1, nights * rooms_cnt))
        flexible_dates = AgentService.calculate_flexible_dates_options(
            dest_id=dest_id,
            base_mandatory_rub=mand_total,
            checkin_date_str=profile.get("checkin_date"),
            nights=nights,
            base_flight_price=raw_flight_rt,
            base_hotel_price=avg_hotel_price,
            fixed_costs=intercity_price + transfers_price + pet_total,
            rooms_count=rooms_cnt,
            travelers_count=travelers_count
        )
        if flexible_dates:
            best_flex = flexible_dates[0]
            if best_flex["savings_rub"] > 0:
                response_lines.append(f"📅 **Календарь выгоды (Гибкие даты ±2 дня):**")
                response_lines.append(
                    f"• 💡 Вылет на {abs(best_flex['days_shift'])} дня {best_flex['direction']} "
                    f"({best_flex['dates_formatted']}, {best_flex['target_weekday']}) "
                    f"сэкономит **~{best_flex['savings_rub']:,} ₽** (поездка обойдётся в {best_flex['target_mandatory_rub']:,} ₽ вместо {mand_total:,} ₽)."
                )
                response_lines.append(f"• *Нажмите на подсказку ниже или напишите «{best_flex['message']}», чтобы пересчитать.*")
                response_lines.append("")
            else:
                cur_day_ru = AgentService.DAY_NAMES_RU[d_in.weekday()]
                response_lines.append(f"📅 **Календарь тарифов (Гибкие даты ±2 дня):**")
                response_lines.append(
                    f"• 🌟 **У вас выбран самый выгодный день вылета:** {cur_day_ru} ({date_range_str}) — "
                    f"минимальный тариф **{mand_total:,} ₽**. Вылеты на соседние даты стоят дороже или столько же."
                )
                response_lines.append("")

        response_lines.append("🛒 *Напишите «Забронировать» для покупки или уточните: «другой отель», «подешевле», «смени даты».*")

        # Формируем аккордеон хода рассуждений агента
        reasoning_steps = [
            {"stage": "nlu", "title": "Анализ запроса и профиля", "detail": f"Составной тур «{cfg['title']}». Состав: {profile.get('party_size', 'стандартный')}, номеров: {rooms_cnt} ({profile.get('room_type', '1 номер')})."},
            {"stage": "ltr_ranking", "title": "ML-ранжирование (CatBoost LTR)", "detail": f"Движок {ltr_meta['engine']}: оценено {ltr_meta['candidates_count']} кандидатов для двух городов за {ltr_meta['latency_ms']} мс. Архетип: {ltr_meta['archetype']}; приоритет: {policy}."},
            {"stage": "multi_city", "title": "Сквозной маршрут", "detail": f"Программа: {cfg['city_1']['name']} ({nights_1} н.) ➔ {cfg['city_2']['name']} ({nights_2} н.)."},
            {"stage": "logistics", "title": "Межгородская логистика в Neo4j", "detail": f"Сегмент: {intercity['name']} ({intercity['duration']}). Сквозная синхронизация чек-ина."},
            {"stage": "compliance", "title": "Паспортный и визовый комплаенс", "detail": f"{compliance_info.get('passport_required')} ({compliance_info.get('passport_validity')}), {compliance_info.get('visa_policy')}."},
            {"stage": "budget", "title": "Двухкомпонентный расчёт", "detail": f"Обязательные расходы: {mand_total:,} ₽, на месте: {onsite_min:,}–{onsite_max:,} ₽."}
        ]
        if flexible_dates:
            reasoning_steps.append({"stage": "dates", "title": "Календарь выгоды", "detail": f"Проверены соседние даты (±2 дня): экономия до {flexible_dates[0]['savings_rub']:,} ₽."})

        reasoning_html = [
            f"<details class=\"agent-reasoning\">",
            f"<summary>🧠 <b>Ход рассуждений агента ({len(reasoning_steps)} проверок)</b></summary>",
            f"<div class=\"reasoning-body\">"
        ]
        for idx, s in enumerate(reasoning_steps, 1):
            reasoning_html.append(f"<div class=\"reasoning-step\"><span class=\"step-num\">{idx}.</span> <strong>{s['title']}:</strong> {s['detail']}</div>")
        reasoning_html.append("</div>")
        reasoning_html.append("</details>")
        reasoning_block = "\n".join(reasoning_html)

        reply_text = f"{reasoning_block}\n\n" + "\n".join(response_lines)

        # Сохранение в корзину
        cart["destination_id"] = dest_id
        cart["destination"] = cfg["title"]
        cart["is_multi_city"] = True
        cart["hotel"] = f"«{hotel_1['name']}» ({nights_1} н.) + «{hotel_2['name']}» ({nights_2} н.)"
        cart["flight"] = f"{fl_code_out} ➔ {fl_code_ret}"
        cart["transfer"] = f"Трансферы + {intercity['name']}"
        cart["nights"] = nights
        cart["dates"] = profile.get("dates", f"{nights} ночей")
        cart["checkin_date"] = profile.get("checkin_date")
        cart["checkout_date"] = profile.get("checkout_date")
        cart["total_mandatory_rub"] = mand_total
        cart["total_trip_budget_range_rub"] = {"min": total_min, "max": total_max}
        cart["onsite_spending_guide"] = f"Рекомендуемый бюджет на месте: от {onsite_min:,} до {onsite_max:,} ₽"
        cart["rooms_count"] = rooms_cnt
        cart["room_type"] = profile.get("room_type", "1 стандартный номер (Standard Room)")
        cart["compliance"] = compliance_info

        cart["ltr_ranking"] = ltr_meta

        await AgentService._save_session(user_id, profile, cart, message, reply_text)

        return {
            "text": reply_text,
            "destination": {"id": dest_id, "name": cfg["title"]},
            "compared_hotels": [hotel_1, hotel_2],
            "logistics": logistics_1,
            "is_multi_city": True,
            "ltr_ranking": ltr_meta,
            "cart": cart,
            "reasoning_steps": reasoning_steps,
            "flexible_dates": flexible_dates,
            "compliance_check": compliance_info,
            "multi_room": {
                "adults_count": profile.get("adults_count", 2),
                "kids_count": profile.get("kids_count", 0),
                "rooms_count": rooms_cnt,
                "room_type": profile.get("room_type", "1 стандартный номер (Standard Room)")
            }
        }

    FLIGHT_DAY_FACTORS = {
        0: 0.98,  # Пн
        1: 0.88,  # Вт (самый дешёвый день для авиабилетов)
        2: 0.90,  # Ср (выгодный день)
        3: 0.95,  # Чт
        4: 1.18,  # Пт (вечерний пик выходного дня)
        5: 1.15,  # Сб (утренний пик на уикенд)
        6: 1.05,  # Вс
    }

    HOTEL_NIGHT_FACTORS = {
        0: 0.96,  # Вс/Пн
        1: 0.96,  # Пн/Вт
        2: 0.96,  # Вт/Ср
        3: 0.98,  # Ср/Чт
        4: 1.15,  # Пт/Сб (уикенд)
        5: 1.15,  # Сб/Вс (уикенд)
        6: 0.96   # Вс
    }

    DAY_NAMES_RU = {
        0: "понедельник",
        1: "вторник",
        2: "среда",
        3: "четверг",
        4: "пятница",
        5: "суббота",
        6: "воскресенье"
    }

    DAY_PREP_RU = {
        0: "в понедельник",
        1: "во вторник",
        2: "в среду",
        3: "в четверг",
        4: "в пятницу",
        5: "в субботу",
        6: "в воскресенье"
    }

    MONTH_RU_SHORT = {
        1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "мая", 6: "июн",
        7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек"
    }

    @staticmethod
    def calculate_real_trip_cost_by_date(
        checkin: date,
        nights: int,
        base_flight_roundtrip: int,
        base_hotel_per_night: int,
        fixed_costs: int = 0,
        rooms_count: int = 1,
        travelers_count: int = 1
    ) -> Dict[str, Any]:
        """
        Рассчитывает фактическую стоимость поездки с привязкой к реальным календарным датам:
        - Авиабилеты: скидки во вторник-среду (-12%), наценки в пятницу-субботу (+18%).
        - Отели: наценки на ночи уикенда (пятница-суббота).
        """
        checkout = checkin + timedelta(days=nights)
        f_out = AgentService.FLIGHT_DAY_FACTORS.get(checkin.weekday(), 1.0)
        f_ret = AgentService.FLIGHT_DAY_FACTORS.get(checkout.weekday(), 1.0)

        half_flight = base_flight_roundtrip * 0.5
        actual_travelers = max(1, travelers_count)
        flight_out = int(round(half_flight * f_out / 100) * 100) * actual_travelers
        flight_ret = int(round(half_flight * f_ret / 100) * 100) * actual_travelers
        flight_total = flight_out + flight_ret

        h_factors = [AgentService.HOTEL_NIGHT_FACTORS.get((checkin + timedelta(days=i)).weekday(), 1.0) for i in range(nights)]
        avg_h = sum(h_factors) / max(1, nights)
        rate_h = int(round(base_hotel_per_night * avg_h / 100) * 100)
        hotel_total = rate_h * nights * rooms_count

        total_mandatory = flight_total + hotel_total + fixed_costs
        return {
            "checkin": checkin,
            "checkout": checkout,
            "flight_out": flight_out,
            "flight_ret": flight_ret,
            "flight_total": flight_total,
            "hotel_rate": rate_h,
            "hotel_total": hotel_total,
            "total_mandatory": total_mandatory
        }

    @staticmethod
    def calculate_package_budget(dest_id: str, hotel: dict, logistics: dict,
                                 profile: dict, *, flexible_dates: bool = False) -> dict:
        """Price a candidate once, before filtering, ranking or rendering it."""
        nights = profile.get("nights", 7)
        checkin = date.fromisoformat(profile["checkin_date"])
        flight_base = logistics["flight"]["price_rub_roundtrip"]
        transfer = 0 if profile.get("without_transfer") else logistics["transfer"]["price_rub"]
        pet_rate = (hotel.get("pet_surcharge_per_night_rub") or 0) if profile.get("has_pets") else 0
        rooms = max(1, profile.get("rooms_count", 1))
        travelers = max(1, profile.get("adults_count", 2) + profile.get("kids_count", 0))
        fixed_costs = transfer + pet_rate * nights
        calculation = AgentService.calculate_real_trip_cost_by_date(
            checkin, nights, flight_base, hotel["price_per_night_rub"],
            fixed_costs, rooms, travelers,
        )
        selected_flex = None
        if flexible_dates:
            options = AgentService.calculate_flexible_dates_options(
                dest_id, calculation["total_mandatory"], checkin.isoformat(), nights,
                flight_base, hotel["price_per_night_rub"], fixed_costs, rooms, travelers,
            )
            if options and options[0]["savings_rub"] > 0:
                selected_flex = options[0]
                checkin = date.fromisoformat(selected_flex["checkin_date"])
                calculation = AgentService.calculate_real_trip_cost_by_date(
                    checkin, nights, flight_base, hotel["price_per_night_rub"],
                    fixed_costs, rooms, travelers,
                )
        breakdown = MCPTools.calculate_budget_breakdown(
            destination_id=dest_id, flight_price_roundtrip=flight_base,
            hotel_price_per_night=hotel["price_per_night_rub"], transfer_price=transfer,
            meals_type=hotel.get("meals", "Завтрак включен"), nights=nights,
            pet_surcharge_per_night=pet_rate, rooms_count=rooms, travelers_count=travelers,
        )
        breakdown["mandatory"].update(
            flight_roundtrip=calculation["flight_total"],
            flight_price_roundtrip=calculation["flight_total"],
            flight_out_price=calculation["flight_out"], flight_ret_price=calculation["flight_ret"],
            flight_out_per_person_rub=calculation["flight_out"] // travelers,
            flight_ret_per_person_rub=calculation["flight_ret"] // travelers,
            hotel_per_night=calculation["hotel_rate"], hotel_total=calculation["hotel_total"],
            total_mandatory_rub=calculation["total_mandatory"],
        )
        onsite = breakdown["onsite_flexible"]["total_onsite_range_rub"]
        breakdown["total_trip_budget_range_rub"] = {
            key: calculation["total_mandatory"] + onsite[key] for key in ("min", "max")
        }
        breakdown["checkin_date"] = checkin.isoformat()
        breakdown["checkout_date"] = calculation["checkout"].isoformat()
        breakdown["selected_flexible_date"] = selected_flex
        breakdown["pricing_source"] = "synthetic_catalog_with_calendar_factors"
        breakdown["pricing_inputs"] = {
            "flight_roundtrip_per_person_rub": flight_base,
            "hotel_per_room_night_rub": hotel["price_per_night_rub"],
            "fixed_costs_rub": fixed_costs, "rooms_count": rooms, "travelers_count": travelers,
        }
        validate_package_totals(breakdown)
        return breakdown

    @staticmethod
    async def evaluate_hotels(dest_id: str, hotels: list[dict], profile: dict, *, flexible_dates: bool = False) -> list:
        evaluated = []
        for hotel in hotels:
            logistics = await MCPTools.check_logistics_compatibility(
                airport_code="", hotel_id=hotel["id"], origin_airport_code=profile.get("origin_airport"),
                prefer_direct_flight=profile.get("prefer_direct_flight"),
                without_transfer=profile.get("without_transfer", False), flight_time_pref=profile.get("flight_time_pref"),
            )
            if not logistics.get("compatible") or "price_rub_roundtrip" not in logistics.get("flight", {}):
                continue
            if not profile.get("without_transfer") and "price_rub" not in logistics.get("transfer", {}):
                continue
            try:
                breakdown = AgentService.calculate_package_budget(dest_id, hotel, logistics, profile, flexible_dates=flexible_dates)
            except ValueError as error:
                logger.warning("Package price validation failed for hotel %s: %s", hotel["id"], error)
                continue
            evaluated.append((hotel, logistics, breakdown))
        return evaluated

    @staticmethod
    def budget_quote(dest_name: str, hotel: dict, breakdown: dict) -> dict:
        return {
            "destination": dest_name, "hotel": hotel["name"], "nights": breakdown["nights"],
            "checkin_date": breakdown["checkin_date"], "checkout_date": breakdown["checkout_date"],
            "total_trip_budget_range_rub": deepcopy(breakdown["total_trip_budget_range_rub"]),
        }

    @staticmethod
    def change_saved_transfer(profile: dict, cart: dict, without_transfer: bool) -> Optional[dict]:
        """Edit only the included transfer of an already quoted single-hotel trip."""
        if not cart.get("hotel_id") or cart.get("is_multi_city") or not cart.get("logistics") or not cart.get("pricing_inputs"):
            return None
        updated = deepcopy(cart)
        logistics = updated["logistics"]
        current_transfer = logistics.get("transfer", {})
        original = updated.get("original_transfer")
        if current_transfer.get("type") != "self":
            original = {"transfer": deepcopy(current_transfer), "timeline": deepcopy(logistics.get("timeline", {})),
                        "logistics_status": logistics.get("logistics_status")}
        if not without_transfer and not original:
            return None
        breakdown = deepcopy(updated.get("budget_breakdown"))
        if not breakdown:
            # Earlier carts retain exact pricing inputs and the quoted range.
            # Reconstruct the ledger without querying or replacing any segment.
            inputs = updated["pricing_inputs"]
            nights = updated["nights"]
            calc = AgentService.calculate_real_trip_cost_by_date(
                date.fromisoformat(updated["checkin_date"]), nights,
                inputs["flight_roundtrip_per_person_rub"], inputs["hotel_per_room_night_rub"],
                inputs["fixed_costs_rub"], inputs["rooms_count"], inputs["travelers_count"],
            )
            old_transfer = current_transfer.get("price_rub", 0)
            travelers = inputs["travelers_count"]
            breakdown = {
                "nights": nights, "checkin_date": updated["checkin_date"], "checkout_date": updated["checkout_date"],
                "mandatory": {
                    "travelers_count": travelers, "rooms_count": inputs["rooms_count"],
                    "flight_roundtrip": calc["flight_total"], "flight_price_roundtrip": calc["flight_total"],
                    "flight_out_price": calc["flight_out"], "flight_ret_price": calc["flight_ret"],
                    "flight_out_per_person_rub": calc["flight_out"] // travelers,
                    "flight_ret_per_person_rub": calc["flight_ret"] // travelers,
                    "hotel_per_night": calc["hotel_rate"], "hotel_total": calc["hotel_total"],
                    "transfer": old_transfer, "pet_total": inputs["fixed_costs_rub"] - old_transfer,
                    "total_mandatory_rub": updated["total_mandatory_rub"],
                },
                "onsite_flexible": {"total_onsite_range_rub": {
                    key: updated["total_trip_budget_range_rub"][key] - updated["total_mandatory_rub"] for key in ("min", "max")
                }},
                "total_trip_budget_range_rub": deepcopy(updated["total_trip_budget_range_rub"]),
                "pricing_inputs": deepcopy(inputs), "pricing_source": "saved_quote",
            }
        validate_package_totals(breakdown)
        old_price = breakdown["mandatory"]["transfer"]
        new_price = 0 if without_transfer else original["transfer"]["price_rub"]
        delta = new_price - old_price
        breakdown["mandatory"]["transfer"] = new_price
        breakdown["mandatory"]["total_mandatory_rub"] += delta
        for bound in ("min", "max"):
            breakdown["total_trip_budget_range_rub"][bound] += delta
        breakdown["pricing_inputs"]["fixed_costs_rub"] += delta
        validate_package_totals(breakdown)
        estimate_key = "min" if profile.get("budget_estimate") == "min" else "max"
        limit = profile.get("budget_rub")
        if not without_transfer and limit is not None and breakdown["total_trip_budget_range_rub"][estimate_key] > limit:
            return {"text": f"С трансфером оценка превысит лимит {limit:,} ₽. Текущий план без трансфера сохранён.",
                    "state": "TRANSFER_OVER_BUDGET", "cart": cart}
        updated["original_transfer"] = original
        if without_transfer:
            logistics["transfer"] = {"name": "Самостоятельный проезд", "type": "self", "price_rub": 0, "duration_min": None}
            logistics.setdefault("timeline", {}).update(hotel_arrival=None, hotel_arrival_day_offset=None, checkin_wait_minutes=None)
            logistics["logistics_status"] = "Самостоятельный проезд; время прибытия в отель уточняется"
        else:
            logistics["transfer"] = deepcopy(original["transfer"])
            logistics["timeline"] = deepcopy(original["timeline"])
            if original["logistics_status"] is None:
                logistics.pop("logistics_status", None)
            else:
                logistics["logistics_status"] = original["logistics_status"]
        if "mandatory_logistics_cost" in logistics:
            logistics["mandatory_logistics_cost"] += delta
        updated.update(without_transfer=without_transfer, transfer=logistics["transfer"]["name"],
                       logistics_check=logistics.get("logistics_status", "Состыковано"), budget_breakdown=breakdown,
                       pricing_inputs=deepcopy(breakdown["pricing_inputs"]),
                       total_mandatory_rub=breakdown["mandatory"]["total_mandatory_rub"],
                       total_trip_budget_range_rub=deepcopy(breakdown["total_trip_budget_range_rub"]))
        updated["budget_quote"] = AgentService.budget_quote(updated["destination"], {"name": updated["hotel"]}, breakdown)
        for option in updated.get("flexible_dates", []):
            option["target_mandatory_rub"] += delta
        for step in updated.get("reasoning_steps", []):
            if step.get("stage") == "budget":
                step["detail"] = f"Обязательные расходы: {updated['total_mandatory_rub']:,} ₽; включённый трансфер: {new_price:,} ₽."
            elif step.get("stage") == "logistics":
                step["detail"] = logistics.get("logistics_status", "Состыковано")
        profile["without_transfer"] = without_transfer
        bounds = updated["total_trip_budget_range_rub"]
        action = ("Трансфер уже исключён." if current_transfer.get("type") == "self" else "Трансфер исключён из расчёта.") if without_transfer else "Трансфер возвращён в план."
        reply = (f"{action} Отель «{updated['hotel']}», рейсы, даты {updated['dates']}, состав и номера сохранены.\n\n"
                 f"Обязательные расходы: {updated['total_mandatory_rub']:,} ₽. Полная оценка: {bounds['min']:,}–{bounds['max']:,} ₽.")
        if without_transfer:
            reply += f"\n\nИз включённых расходов убрано {old_price:,} ₽. Самостоятельную дорогу организуете и оплачиваете отдельно; её стоимость здесь не учтена."
        return {"text": reply, "state": "TRANSFER_REMOVED" if without_transfer else "TRANSFER_INCLUDED", "cart": updated,
                "logistics": logistics, "budget_breakdown": breakdown, "reasoning_steps": updated.get("reasoning_steps", []),
                "flexible_dates": updated.get("flexible_dates", [])}

    @staticmethod
    def calculate_flexible_dates_options(
        dest_id: str,
        base_mandatory_rub: int,
        checkin_date_str: Optional[str],
        nights: int,
        base_flight_price: Optional[int] = None,
        base_hotel_price: Optional[int] = None,
        fixed_costs: Optional[int] = None,
        rooms_count: int = 1,
        travelers_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Рассчитывает честный календарь тарифов и выгоды (±2 дня) со смещением вылета.
        Использует реальные тарифные кривые авиакомпаний (низкие тарифы во вторник-среду,
        высокие пики в пятницу-субботу) и отелей (выходные/будни).
        """
        import datetime
        options = []
        base_dt = None
        if checkin_date_str:
            try:
                base_dt = datetime.date.fromisoformat(checkin_date_str)
            except Exception:
                pass
        if not base_dt:
            base_dt = datetime.date.today() + datetime.timedelta(days=14)

        if base_flight_price is None or base_hotel_price is None or fixed_costs is None:
            return []
        fl_base = base_flight_price
        ht_base = base_hotel_price
        fix_base = fixed_costs
        if fix_base < 0:
            fix_base = 0

        current_res = AgentService.calculate_real_trip_cost_by_date(
            base_dt, nights, fl_base, ht_base, fix_base, rooms_count, travelers_count
        )
        current_cost = current_res["total_mandatory"]

        # Оцениваем соседние дни (±2 дня): -2, -1, +1, +2
        shifts = [-2, -1, 1, 2]
        for days_diff in shifts:
            new_in = base_dt + datetime.timedelta(days=days_diff)
            new_out = new_in + datetime.timedelta(days=nights)
            cand_res = AgentService.calculate_real_trip_cost_by_date(
                new_in, nights, fl_base, ht_base, fix_base, rooms_count, travelers_count
            )
            cand_cost = cand_res["total_mandatory"]
            savings = current_cost - cand_cost
            dir_word = "позже" if days_diff > 0 else "раньше"
            day_name = AgentService.DAY_NAMES_RU[new_in.weekday()]
            prep_day = AgentService.DAY_PREP_RU[new_in.weekday()]
            date_label = f"{new_in.day} {AgentService.MONTH_RU_SHORT.get(new_in.month, '')} — {new_out.day} {AgentService.MONTH_RU_SHORT.get(new_out.month, '')}"

            if savings > 0:
                label_chip = f"📅 На {abs(days_diff)} дн. {dir_word} (-{savings:,} ₽)"
            elif savings < 0:
                label_chip = f"📅 На {abs(days_diff)} дн. {dir_word} (+{abs(savings):,} ₽)"
            else:
                label_chip = f"📅 На {abs(days_diff)} дн. {dir_word} (та же цена)"

            msg = f"Сдвинь поездку на {abs(days_diff)} дня {dir_word}"

            options.append({
                "days_shift": days_diff,
                "direction": dir_word,
                "target_weekday": day_name,
                "target_weekday_prep": prep_day,
                "checkin_date": new_in.isoformat(),
                "checkout_date": new_out.isoformat(),
                "dates_formatted": date_label,
                "savings_rub": savings,
                "target_mandatory_rub": cand_cost,
                "label": label_chip,
                "message": msg
            })

        options.sort(key=lambda x: -x["savings_rub"])
        return options

    @staticmethod
    def answer_dates_comparison(user_lower: str, profile: dict, cart: dict) -> Optional[str]:
        """Отвечает на запрос детального сравнения реальных тарифов по соседним датам (±2 дня)."""
        triggers = [
            "соседние даты", "соседних дат", "календарь цен", "цены на соседние даты",
            "самые дешевые даты", "самые дешёвые даты", "когда дешевле",
            "в какие даты дешевле", "самый дешевый вылет", "самый дешёвый вылет",
            "самая дешевая поездка", "самая дешёвая поездка", "проверить цены",
            "проверь самую дешевую", "проверь самую дешёвую", "реальные цены",
            "сравнить даты", "сравни даты", "сравнение дат", "какие даты выгоднее",
            "самый выгодный день", "выгодные даты", "выгодный день", "тарифы по дням"
        ]
        if not any(t in user_lower for t in triggers):
            return None
        if any(w in user_lower for w in ["сделай", "подбери", "собери", "пересчитай", "построй", "выбери"]) and any(w in user_lower for w in ["самый дешев", "самая дешев", "самое дешев", "самые дешев", "дешевле", "подешевле"]):
            return None
        if not cart.get("destination_id") and not profile.get("destination_id"):
            return None

        nights = profile.get("nights", cart.get("nights", 7))
        checkin_str = profile.get("checkin_date") or cart.get("checkin_date")
        if not checkin_str:
            base_dt = date.today() + timedelta(days=14)
        else:
            try:
                base_dt = date.fromisoformat(checkin_str)
            except Exception:
                base_dt = date.today() + timedelta(days=14)

        inputs = cart.get("pricing_inputs")
        if not inputs:
            return "Для сохранённого плана нет исходных тарифов. Пересчитайте поездку, чтобы сравнить соседние даты без выдуманных цен."
        rooms_cnt = inputs["rooms_count"]
        travelers_count = inputs["travelers_count"]
        base_fl = inputs["flight_roundtrip_per_person_rub"]
        base_ht = inputs["hotel_per_room_night_rub"]
        fixed_c = inputs["fixed_costs_rub"]

        cur_res = AgentService.calculate_real_trip_cost_by_date(
            base_dt, nights, base_fl, base_ht, fixed_c, rooms_cnt, travelers_count
        )

        all_dates = []
        for shift in (-2, -1, 0, 1, 2):
            dt = base_dt + timedelta(days=shift)
            dt_out = dt + timedelta(days=nights)
            res = AgentService.calculate_real_trip_cost_by_date(
                dt, nights, base_fl, base_ht, fixed_c, rooms_cnt, travelers_count
            )
            diff = res["total_mandatory"] - cur_res["total_mandatory"]
            day_name = AgentService.DAY_NAMES_RU[dt.weekday()]
            date_str = f"{dt.day} {AgentService.MONTH_RU_SHORT.get(dt.month, '')} — {dt_out.day} {AgentService.MONTH_RU_SHORT.get(dt_out.month, '')}"
            all_dates.append({
                "shift": shift,
                "dt": dt,
                "day_name": day_name,
                "date_str": date_str,
                "cost": res["total_mandatory"],
                "diff": diff,
                "is_current": (shift == 0)
            })

        min_date = min(all_dates, key=lambda x: x["cost"])

        lines = [
            f"📊 **Сравнение расчётных тарифов на соседние даты (±2 дня, тур на {nights} н.):**",
            "Демонстрационный каталог и календарные коэффициенты; цены поставщика не запрашивались.",
            ""
        ]
        for item in all_dates:
            flag = " *(текущий выбор)*" if item["is_current"] else ""
            if item["cost"] == min_date["cost"]:
                best_mark = " 🔥 **самый выгодный тариф!**"
            elif item["diff"] > 0:
                best_mark = f" (+{item['diff']:,} ₽)"
            elif item["diff"] < 0:
                best_mark = f" ({item['diff']:,} ₽)"
            else:
                best_mark = " (та же цена)"

            lines.append(f"• **{item['date_str']} ({item['day_name']}):** **{item['cost']:,} ₽**{best_mark}{flag}")

        lines.append("")
        if min_date["is_current"]:
            lines.append(f"✅ **У вас уже выбран самый дешёвый день вылета** ({min_date['day_name']}). Любой вылет на соседние даты обойдётся дороже.")
        else:
            diff_abs = abs(min_date["diff"])
            shift_word = f"на {abs(min_date['shift'])} дня {'позже' if min_date['shift'] > 0 else 'раньше'}"
            lines.append(
                f"💡 **Самый дешёвый вариант:** вылет {AgentService.DAY_PREP_RU[min_date['dt'].weekday()]} ({min_date['date_str']}) "
                f"сэкономит **{diff_abs:,} ₽** (стоимость составит {min_date['cost']:,} ₽ вместо {cur_res['total_mandatory']:,} ₽)."
            )
            lines.append(f"*Чтобы переключиться на эти даты, напишите: «Сдвинь поездку {shift_word}» или «заезд {min_date['dt']:%d.%m.%Y}».*")

        return "\n".join(lines)

    @staticmethod
    async def process_message(user_id: str, message: str, progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        """Обрабатывает запрос и возвращает учтённые условия текущей поездки."""
        session_obj = await AgentService.get_or_create_session(user_id)
        profile = session_obj.profile or {}
        cart = deepcopy(session_obj.current_cart or {})
        if (cart.get("destination_id") == "sochi" and cart.get("hotel_id")
                and cart.get("destination") == "Сочи и Красная Поляна" and not cart.get("is_multi_city")):
            # Existing JSONB carts may still contain the combined catalogue title.
            # Read the selected hotel's city once, without reselecting or repricing it.
            if not cart.get("hotel_city"):
                async with async_session_maker() as session:
                    hotel = await session.get(HotelModel, cart["hotel_id"])
                    cart["hotel_city"] = hotel.city if hotel else None
            cart["destination"] = single_destination_name("sochi", cart.get("destination", "Сочи"), cart.get("hotel_city"))
            if cart.get("budget_quote"):
                cart["budget_quote"]["destination"] = cart["destination"]
        result = await AgentService._process_message(
            user_id, message, profile, cart, session_obj, progress_callback
        )
        if not result.get("booking_action"):
            # Только условия поиска: служебные поля и история профиля в UI не нужны.
            keys = ("dates", "nights", "adults_count", "kids_count", "has_kids",
                    "has_pets", "private_room_required", "budget_rub", "rooms_count", "without_transfer")
            result["trip_conditions"] = {key: profile.get(key) for key in keys}
        return result

    @staticmethod
    async def _process_message(user_id: str, message: str, profile: dict, cart: dict,
                               session_obj: Any, progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Основной цикл обработки запроса путешественника:
        1. Распознавание интента и профилирование (память пользователя)
        2. Пункт 4 ТЗ: Анти-15-вопросов при нечетком старте
        3. Пункт 3 ТЗ: Fallback & рекомендации для неизвестных направлений
        4. Пункт 2 ТЗ: Динамическое перепланирование на лету без сброса контекста
        5. Пункт 5 ТЗ: Имитация чекаута и ваучер
        """
        user_lower = message.lower().replace("ё", "е").strip()
        previous_cart = deepcopy(cart)
        transfer_setting = transfer_preference(message)
        selected_hotel_query = hotel_selection(message)
        direct_destination_choice = destination_choice(message) if not selected_hotel_query else None
        slot_message = "" if selected_hotel_query else user_lower
        excluded_now = rejected_destinations(message) if not selected_hotel_query else set()
        slot_keys = ("nights", "adults_count", "kids_count", "has_kids", "has_pets",
                     "private_room_required", "origin_airport", "flight_time_pref",
                     "abroad_only", "domestic_only", "requested_rooms_count")
        previous_slots = {key: deepcopy(profile.get(key)) for key in slot_keys}

        # Model hints belong to one turn, never to the saved trip state.
        for transient in ("llm_destination_id", "llm_replan_action", "llm_ai_intro"):
            profile.pop(transient, None)
        if current_trip_question(message):
            reply = describe_current_trip(profile, cart)
            await AgentService._save_session(user_id, profile, cart, message, reply)
            return {"text": reply, "state": "CURRENT_TRIP", "cart": cart}

        requested_estimate = estimate_request(message)
        if requested_estimate:
            profile["budget_estimate"] = requested_estimate
            if cart.get("budget_quote") and not is_trip_edit(message):
                reply = describe_budget_estimate(cart["budget_quote"], requested_estimate, profile.get("budget_rub"))
                cart["budget_estimate"] = requested_estimate
                await AgentService._save_session(user_id, profile, cart, message, reply)
                return {"text": reply, "state": "BUDGET_ESTIMATE", "cart": cart}

        booking_action = detect_booking_action(message)
        if booking_action:
            result = booking_stub(booking_action, cart)
            await AgentService._save_session(user_id, profile, cart, message, result["text"])
            return result

        previous_budget = profile.get("budget_rub")
        budget_question = update_trip_budget(slot_message, profile, cart)
        if budget_question:
            profile = AgentService.extract_interests_and_slots(user_lower, profile)
            update_trip_dates(user_lower, profile, cart)
            explicit_destination = requested_destination(message)
            pending_destinations = await AgentService.search_destinations_vector(explicit_destination, limit=1) if explicit_destination else []
            if pending_destinations:
                cart["destination_id"] = pending_destinations[0]["id"]
                cart["destination"] = pending_destinations[0]["name"]
            cart = {key: cart[key] for key in ("destination_id", "destination") if key in cart}
            cart.update(nights=profile.get("nights", 7), dates=profile.get("dates"),
                        checkin_date=profile.get("checkin_date"), checkout_date=profile.get("checkout_date"),
                        selection_status="needs_budget")
            await AgentService._save_session(user_id, profile, cart, message, budget_question)
            return {"text": budget_question, "state": "NEEDS_BUDGET", "cart": cart}
        budget_changed = profile.get("budget_rub") != previous_budget
        budget_limit = profile.get("budget_rub")

        # Синхронизация дат: если в profile нет checkin_date, восстанавливаем из cart или текущего тура
        if not profile.get("checkin_date"):
            if cart.get("checkin_date"):
                profile["checkin_date"] = cart["checkin_date"]
                profile["checkout_date"] = cart.get("checkout_date")
            elif cart.get("hotel") or cart.get("destination") or cart.get("dates"):
                base_dt = date.today() + timedelta(days=14)
                nights_cnt = profile.get("nights", cart.get("nights", 7))
                profile["checkin_date"] = base_dt.isoformat()
                profile["checkout_date"] = (base_dt + timedelta(days=nights_cnt)).isoformat()
                profile["dates"] = f"{base_dt:%d.%m.%Y} — {base_dt + timedelta(days=nights_cnt):%d.%m.%Y} ({nights_cnt} ночей)"

        previous_dates = profile.get("dates")
        previous_calendar = (profile.get("checkin_date"), profile.get("checkout_date"))
        date_question = update_trip_dates(slot_message, profile, cart)
        if date_question:
            await AgentService._save_session(user_id, profile, cart, message, date_question)
            return {"text": date_question, "state": "NEEDS_DATES", "cart": cart}

        # 1. Извлечение интересов и обновление профиля пользователя (User Memory)
        profile = AgentService.extract_interests_and_slots(slot_message, profile)
        other_conditions_changed = budget_changed or previous_calendar != (
            profile.get("checkin_date"), profile.get("checkout_date")
        ) or any(
            profile.get(key) != previous_slots[key] for key in slot_keys
        )
        other_replan_requested = bool(selected_hotel_query or excluded_now or direct_destination_choice or wants_cheaper(message)) or bool(re.search(
            r"(?:другой|поменяй|смени)\s+(?:отель|город|номер|направление)|"
            r"прямой|пересад|вылет|рейс|самолет|попозже|пораньше|утром|вечером|днем|любое время",
            user_lower,
        ))
        if transfer_setting is not None and not other_conditions_changed and not other_replan_requested:
            result = AgentService.change_saved_transfer(profile, cart, transfer_setting)
            if result:
                await AgentService._save_session(user_id, profile, result["cart"], message, result["text"])
                return result
        deterministic_trip_edit = is_trip_edit(message) or transfer_setting is not None or bool(excluded_now) or bool(direct_destination_choice) or other_conditions_changed

        # Проверка вопроса о расписании внутреннего маршрута / переезда между городами
        internal_timing_reply = AgentService.answer_internal_route_timing(user_lower, cart)
        if internal_timing_reply:
            await AgentService._save_session(user_id, profile, cart, message, internal_timing_reply)
            return {
                "text": internal_timing_reply,
                "state": "INTERNAL_SCHEDULE_INFO",
                "cart": cart
            }

        # Проверка вопроса о сравнении реальных цен на соседние даты / поиске самого дешёвого дня
        dates_comparison_reply = AgentService.answer_dates_comparison(user_lower, profile, cart)
        if dates_comparison_reply:
            await AgentService._save_session(user_id, profile, cart, message, dates_comparison_reply)
            return {
                "text": dates_comparison_reply,
                "state": "DATES_COMPARISON",
                "cart": cart
            }

        # 2. Интеллектуальный Tool Calling диспетчер на базе DeepSeek Flash
        # ИИ сам извлекает параметры (город вылета, время, географию, неизвестные локации, тип перепланирования, теги)
        # без необходимости для пользователя говорить точные кодовые слова.
        has_dest_in_cart = bool(cart.get("destination_id"))
        if not settings.is_mock_llm and not user_id.startswith("test") and not (deterministic_trip_edit or requested_estimate):
            llm_decision = await LLMService.dispatch_intent(
                user_message=message,
                profile=profile,
                current_cart=cart,
                conversation_history=getattr(session_obj, "conversation_history", [])
            )
            if llm_decision:
                if llm_decision.get("action") in ("checkout", "cancel_booking", "reschedule_booking"):
                    action = {"checkout": "book", "cancel_booking": "cancel", "reschedule_booking": "reschedule"}[llm_decision["action"]]
                    result = booking_stub(action, cart)
                    await AgentService._save_session(user_id, profile, cart, message, result["text"])
                    return result
                if llm_decision.get("action") == "unknown_destination":
                    alt_id = llm_decision.get("suggested_pilot_id") or "maldives"
                    alt_name = llm_decision.get("suggested_pilot_name") or "Мальдивы"
                    reason = llm_decision.get("vibe_reason") or "отличный курортный отдых и море впечатлений"
                    
                    reply_fallback = (
                        f"🌍 **Обратите внимание:** В текущей версии нашей системы оцифрована "
                        f"сквозная графовая логистика (рейсы, трансферы и отели) по 9 ключевым направлениям: "
                        f"**Сочи, ОАЭ, Турция, Таиланд, Мальдивы, Египет, Санкт-Петербург, Алтай и Грузия**.\n\n"
                        f"💡 **Рекомендуем альтернативу по вайбу:** Если вас привлекает {reason} — "
                        f"идеальным выбором станет **{alt_name}**!\n\n"
                        f"Собрать для вас готовый тур в **{alt_name}** с проверенным перелетом, трансфером и расчетом бюджета?"
                    )
                    profile["suggested_destination"] = alt_id
                    await AgentService._save_session(user_id, profile, cart, message, reply_fallback)
                    return {
                        "text": reply_fallback,
                        "suggested_destination": alt_id,
                        "state": "FALLBACK_OFFERED",
                        "cart": cart
                    }
                elif llm_decision.get("action") == "consultation" and not any(w in user_lower for w in ["забронир", "оформи", "купи", "хочу поехать", "подбери тур", "собери тур"]):
                    consult_text = llm_decision.get("text", "")
                    if consult_text:
                        await AgentService._save_session(user_id, profile, cart, message, consult_text)
                        return {
                            "text": consult_text,
                            "state": "CONSULTATION",
                            "cart": cart
                        }
                elif llm_decision.get("action") == "plan":
                    # Если запрос приветственный/неопределенный без указания направления (Анти-15-вопросов)
                    if AgentService.is_vague_greeting(message, has_dest_in_cart):
                        profile["is_multi_city"] = False
                        cart.clear()
                        cart.pop("is_multi_city", None)
                        reply_anti_questions = AgentService.get_clarification_prompt(message)
                        await AgentService._save_session(user_id, profile, cart, message, reply_anti_questions)
                        return {
                            "text": reply_anti_questions,
                            "state": "INSPIRATION_PROMPT",
                            "cart": cart
                        }

                    if llm_decision.get("ai_reply_intro"):
                        profile["llm_ai_intro"] = llm_decision["ai_reply_intro"]
                    if llm_decision.get("origin_airport") and llm_decision["origin_airport"] != "any":
                        profile["origin_airport"] = llm_decision["origin_airport"]
                    if llm_decision.get("flight_time_pref") and llm_decision["flight_time_pref"] != "any":
                        profile["flight_time_pref"] = llm_decision["flight_time_pref"]
                    if llm_decision.get("geography") == "abroad":
                        profile["abroad_only"] = True
                        profile.pop("domestic_only", None)
                    elif llm_decision.get("geography") == "domestic":
                        profile["domestic_only"] = True
                        profile.pop("abroad_only", None)
                    if llm_decision.get("destination_id"):
                        profile["llm_destination_id"] = llm_decision["destination_id"]
                    if llm_decision.get("replan_action") and llm_decision["replan_action"] != "none":
                        profile["llm_replan_action"] = llm_decision["replan_action"]
                    if llm_decision.get("family_friendly"):
                        profile["has_kids"] = True
                        if "family_kids" not in profile.setdefault("interests", []):
                            profile["interests"].append("family_kids")
                    if llm_decision.get("private_room_required"):
                        profile["private_room_required"] = True
                    if llm_decision.get("pet_friendly"):
                        profile["has_pets"] = True
                        if "pet_friendly" not in profile.setdefault("interests", []):
                            profile["interests"].append("pet_friendly")
                    if llm_decision.get("prefer_direct_flight") is not None:
                        profile["prefer_direct_flight"] = llm_decision["prefer_direct_flight"]
                    if llm_decision.get("without_transfer"):
                        profile["without_transfer"] = True
                    if llm_decision.get("tags"):
                        for t in llm_decision["tags"]:
                            if t not in profile.setdefault("interests", []):
                                profile["interests"].append(t)
                    if llm_decision.get("is_multi_city") and AgentService.is_explicit_multi_city_request(message):
                        profile["is_multi_city"] = True

        # Детерминированный fallback (для mock LLM и unit-тестов):
        # Пункт 4 ТЗ: Анти-15-вопросов на старте (Anti-15-Questions)
        if not selected_hotel_query and AgentService.is_vague_greeting(message, has_dest_in_cart):
            profile["is_multi_city"] = False
            cart.clear()
            cart.pop("is_multi_city", None)
            reply_anti_questions = AgentService.get_clarification_prompt(message)
            await AgentService._save_session(user_id, profile, cart, message, reply_anti_questions)
            return {
                "text": reply_anti_questions,
                "state": "INSPIRATION_PROMPT",
                "cart": cart
            }

        # Детерминированный fallback для неизвестных направлений (если оффлайн / mock LLM)
        unknown_match = AgentService.detect_unknown_destination(message) if not selected_hotel_query else None
        if unknown_match and not profile.get("llm_destination_id"):
            alt_id = unknown_match["pilot_alt_id"]
            alt_name = unknown_match["pilot_alt_name"]
            reason = unknown_match["vibe_reason"]
            
            # Предлагаем ближайшую альтернативу по вайбу
            reply_fallback = (
                f"🌍 **Обратите внимание:** В текущей версии нашей системы оцифрована "
                f"сквозная графовая логистика (рейсы, трансферы и отели) по 8 ключевым направлениям: "
                f"**Сочи, ОАЭ, Турция, Таиланд, Мальдивы, Египет, Санкт-Петербург и Алтай**.\n\n"
                f"💡 **Рекомендуем альтернативу по вайбу:** Если вас привлекает {reason} — "
                f"идеальным выбором станет **{alt_name}**!\n\n"
                f"Собрать для вас готовый тур в **{alt_name}** с проверенным перелетом, трансфером и расчетом бюджета?"
            )
            # Запоминаем альтернативное направление в профиле
            profile["suggested_destination"] = alt_id
            await AgentService._save_session(user_id, profile, cart, message, reply_fallback)
            return {
                "text": reply_fallback,
                "suggested_destination": alt_id,
                "state": "FALLBACK_OFFERED",
                "cart": cart
            }

        # 4. Пункт 2 ТЗ: Динамическое перепланирование без сброса контекста (Dynamic Re-planning)
        current_dest_id = cart.get("destination_id")
        explicit_destination = direct_destination_choice
        excluded_ids = set(profile.get("excluded_destinations", [])) | excluded_now
        if explicit_destination:
            excluded_ids.discard(explicit_destination)
            profile["destination_flexible"] = False
        elif (not current_dest_id and budget_limit is not None) or current_dest_id in excluded_now or any(
            phrase in user_lower for phrase in ("направлению пока не знаю", "направление не знаю", "не определились", "любое направление", "другой город", "другое направление")
        ):
            profile["destination_flexible"] = True
        if selected_hotel_query:
            profile["destination_flexible"] = False
        profile["excluded_destinations"] = sorted(excluded_ids)
        search_multiple_destinations = bool(profile.get("destination_flexible")) and not profile.get("is_multi_city")
        explicit_dest_change = bool(explicit_destination and explicit_destination != current_dest_id) or any(
            w in user_lower for w in ["другой город", "поменяй направление", "другая страна", "поехали в", "а если в "]
        )
        
        is_replan_hotel = any(w in user_lower for w in ["другой отель", "поменяй отель", "смени отель", "не нравится отель", "есть еще отели", "покажи другие варианты", "альтернативный отель", "другой вариант проживания"])
        is_cheapest_requested = any(w in user_lower for w in [
            "самый дешев", "самые дешев", "самая дешев", "самое дешев",
            "самый эконом", "самый бюджет", "минимальн", "как можно дешев",
            "максимально дешев", "подешевле", "подешевле вариант", "дешевый вариант", "дешевые билеты", "дешевыми билетами"
        ])
        is_any_time = any(w in user_lower for w in [
            "в любое время", "в любой день", "любое время", "время не важно", "не важно когда",
            "без разницы во сколько", "в любое время суток", "любое время вылета", "когда угодно"
        ])
        is_replan_cheaper = not selected_hotel_query and wants_cheaper(message)
        is_replan_flight_day = any(w in user_lower for w in ["днем", "днём", "в обед", "дневной вылет", "дневным вылетом", "вылет днем", "вылет днём", "вылетать не в 6 утра", "не в 6 утра, а днем", "не в 6 утра, а днём"])
        is_replan_flight_later = any(w in user_lower for w in ["попозже", "самолет попозже", "вылет попозже", "рейс попозже", "не в 6 утра", "не в 6", "не в 7", "не рано", "позже 10", "после 10", "после 12", "вечером", "поздний вылет", "вылетать позже"]) or is_replan_flight_day
        is_replan_flight_morning = any(w in user_lower for w in ["пораньше", "утром", "утренний вылет", "самолет пораньше", "рейс пораньше", "ранний вылет"])
        is_replan_flight = is_replan_flight_later or is_replan_flight_morning or any(w in user_lower for w in ["только прямой", "прямой рейс", "без пересадок", "с пересадкой", "стыковочный", "рейс подешевле"])
        is_replan_transfer = transfer_setting is not None
        is_replan_dates = profile.get("dates") != previous_dates and bool(profile.get("checkin_date"))
        is_replan_nights = any(w in user_lower for w in ["ноч", "дн", "сут", "выходн", "уикенд", "недел", "месяц"]) and profile.get("nights") != cart.get("nights")

        # Если запрошено любое время вылета — сбрасываем ограничения времени и типа рейса
        if is_any_time:
            profile["flight_time_pref"] = None
            profile.pop("flight_time_pref", None)
            is_replan_flight_later = False
            is_replan_flight_day = False
            is_replan_flight_morning = False
            is_replan_flight = False

        if is_replan_cheaper:
            interests = profile.get("interests", [])
            profile["interests"] = [t for t in interests if t not in ("luxury", "extreme_luxury", "luxury_extreme", "splurge")]
            if "budget_friendly" not in profile["interests"]:
                profile["interests"].append("budget_friendly")

        # Применяем решения ИИ-диспетчера (Tool Calling DeepSeek Flash)
        llm_ra = profile.pop("llm_replan_action", None)
        if llm_ra == "change_hotel":
            is_replan_hotel = True
        elif llm_ra == "cheaper":
            is_replan_cheaper = True
        elif llm_ra == "change_flight_time" and not is_any_time:
            is_replan_flight = True
            if profile.get("flight_time_pref") == "day":
                is_replan_flight_day = True
                is_replan_flight_later = True
            elif profile.get("flight_time_pref") in ("later", "evening"):
                is_replan_flight_later = True
            elif profile.get("flight_time_pref") == "morning":
                is_replan_flight_morning = True
        elif llm_ra == "remove_transfer":
            is_replan_transfer = True
            profile["without_transfer"] = True
        elif llm_ra == "change_destination":
            explicit_dest_change = True

        if is_replan_cheaper or is_replan_hotel or explicit_dest_change:
            profile.pop("selected_hotel_id", None)

        if is_replan_flight_day and not is_any_time:
            profile["flight_time_pref"] = "day"

        is_abroad_request = profile.get("abroad_only", False)
        is_domestic_request = profile.get("domestic_only", False)
        if explicit_destination:
            if explicit_destination in ("sochi", "spb", "altai"):
                profile.pop("abroad_only", None)
                is_abroad_request = False
            else:
                profile.pop("domestic_only", None)
                is_domestic_request = False
        if is_abroad_request and current_dest_id in ("sochi", "spb", "altai"):
            explicit_dest_change = True
        elif is_domestic_request and current_dest_id not in ("sochi", "spb", "altai"):
            explicit_dest_change = True

        nights = profile.get("nights", 7)
        if not profile.get("checkin_date"):
            update_trip_dates((date.today() + timedelta(days=14)).isoformat(), profile)
        travelers_count = max(1, profile.get("adults_count", 2) + profile.get("kids_count", 0))
        cart["travelers_count"] = travelers_count
        cart["nights"] = nights
        cart["dates"] = profile.get("dates", f"{nights} ночей")

        # Обработка исключения трансфера
        if is_replan_transfer:
            profile["without_transfer"] = transfer_setting if transfer_setting is not None else True

        # Обработка типа рейса
        if any(w in user_lower for w in ["только прямой", "прямой рейс", "без пересадок"]):
            profile["prefer_direct_flight"] = True
        elif any(w in user_lower for w in ["с пересадкой", "стыковочный"]):
            profile["prefer_direct_flight"] = False

        # Определение целевого направления
        matched_destinations = await AgentService.search_destinations_vector(
            message,
            limit=9 if search_multiple_destinations else 2,
            abroad_only=is_abroad_request,
            domestic_only=is_domestic_request
        )
        if search_multiple_destinations:
            matched_destinations = [candidate for candidate in matched_destinations if candidate["id"] not in excluded_ids
                                    and (not is_domestic_request or candidate["id"] in ("sochi", "spb", "altai"))
                                    and (not is_abroad_request or candidate["id"] not in ("sochi", "spb", "altai"))]
        msg_top_dest = matched_destinations[0] if matched_destinations else None

        # Если ИИ точно определил destination_id через Tool Calling
        llm_dest = profile.pop("llm_destination_id", None)
        if llm_dest and (explicit_dest_change or not current_dest_id or (is_abroad_request and current_dest_id in ("sochi", "spb", "altai"))):
            cur_matched = await AgentService.search_destinations_vector(llm_dest, limit=1)
            if cur_matched:
                msg_top_dest = cur_matched[0]

        if search_multiple_destinations and msg_top_dest:
            top_dest = msg_top_dest
            dest_id, dest_name = top_dest["id"], top_dest["name"]
        elif search_multiple_destinations:
            reply = "После исключения направлений под ваши условия не осталось вариантов. Какое направление рассмотреть?"
            cart = {"nights": nights, "dates": profile.get("dates"), "selection_status": "no_matches"}
            await AgentService._save_session(user_id, profile, cart, message, reply)
            return {"text": reply, "state": "NO_MATCHES", "cart": cart, "compared_hotels": []}
        elif current_dest_id and not explicit_dest_change and not profile.get("is_multi_city"):
            # Контекстное перепланирование в рамках текущего города/курорта!
            dest_id = current_dest_id
            cur_matched = await AgentService.search_destinations_vector(dest_id, limit=1)
            top_dest = cur_matched[0] if cur_matched else {"id": dest_id, "name": cart.get("destination", dest_id), "climate_summary": ""}
            dest_name = top_dest["name"]
        elif msg_top_dest and (not current_dest_id or explicit_dest_change or msg_top_dest.get("score", 0) >= 0.999):
            dest_id = msg_top_dest["id"]
            dest_name = msg_top_dest["name"]
            top_dest = msg_top_dest
            # При смене направления очищаем список отклоненных отелей для нового региона
            if current_dest_id != dest_id:
                profile["rejected_hotels"] = []
        elif current_dest_id and not (is_abroad_request and current_dest_id in ("sochi", "spb", "altai")):
            dest_id = current_dest_id
            cur_matched = await AgentService.search_destinations_vector(dest_id, limit=1)
            top_dest = cur_matched[0] if cur_matched else msg_top_dest
            dest_name = top_dest["name"] if top_dest else dest_id
        elif msg_top_dest:
            dest_id = msg_top_dest["id"]
            dest_name = msg_top_dest["name"]
            top_dest = msg_top_dest
        else:
            return {
                "text": "Извините, не удалось найти подходящие направления в базе.",
                "state": session_obj.current_state,
                "cart": cart
            }

        cart["destination_id"] = dest_id
        cart["destination"] = dest_name

        # Пункт: Мульти-городской тур (Multi-City Itinerary с динамическим подбором связок городов)
        if profile.get("is_multi_city"):
            selected_dest, selected_cfg, rationale = AgentService.select_multi_city_route(
                user_message=message,
                profile=profile,
                current_dest_id=current_dest_id or dest_id
            )
            if selected_dest:
                dest_id = selected_dest
                dest_name = selected_cfg["title"]
                cart["destination_id"] = dest_id
                cart["destination"] = dest_name
                cur_matched = await AgentService.search_destinations_vector(dest_id, limit=1)
                top_dest = cur_matched[0] if cur_matched else {"id": dest_id, "name": dest_name, "climate_summary": ""}

            return await AgentService.assemble_multi_city_itinerary(
                dest_id=dest_id,
                top_dest=top_dest,
                profile=profile,
                cart=cart,
                nights=nights,
                budget_limit=budget_limit,
                user_id=user_id,
                message=message,
                is_replan_flight_day=is_replan_flight_day,
                is_replan_flight_later=is_replan_flight_later,
                session_obj=session_obj,
                selection_rationale=rationale
            )

        # 5. Обработка смены отеля (Добавление текущего в rejected_hotels)
        rejected_hotels = profile.get("rejected_hotels", [])
        old_hotel_name = cart.get("hotel")
        old_hotel_id = cart.get("hotel_id")

        if is_replan_hotel and old_hotel_id and not selected_hotel_query:
            if old_hotel_id not in rejected_hotels:
                rejected_hotels.append(old_hotel_id)
                profile["rejected_hotels"] = rejected_hotels

        if is_replan_cheaper:
            profile.setdefault("interests", []).append("budget_friendly")

        # 6. Вызов MCP: Сравнение отелей с учетом тегов, исключений и фильтров
        estimate_key = "min" if profile.get("budget_estimate") == "min" else "max"
        candidates = matched_destinations if search_multiple_destinations else [top_dest]
        evaluated_regions = []
        compared_list = []
        requested_hotel_id = None
        for candidate in candidates:
            hotel_comparison = await MCPTools.compare_hotels(
                destination_id=candidate["id"], pet_friendly=profile.get("has_pets"),
                family_friendly=profile.get("has_kids"), private_room_required=profile.get("private_room_required", False),
                exclude_hotel_ids=rejected_hotels, preferred_tags=profile.get("interests"),
                limit=None if budget_limit is not None or is_replan_cheaper or selected_hotel_query or profile.get("selected_hotel_id") else 30,
            )
            hotels = hotel_comparison.get("compared_hotels", [])
            if selected_hotel_query:
                matches = matching_hotels(selected_hotel_query, hotels)
                if len(matches) != 1:
                    reply = (f"По названию «{selected_hotel_query}» найдено несколько отелей. Уточните полное название."
                             if matches else f"Отель «{selected_hotel_query}» не найден среди доступных в текущем направлении с вашими условиями.")
                    await AgentService._save_session(user_id, profile, previous_cart, message, reply)
                    return {"text": reply, "state": "HOTEL_NOT_AVAILABLE", "cart": previous_cart}
                requested_hotel_id = matches[0]["id"]
                hotels = matches
            elif profile.get("selected_hotel_id"):
                hotels = [hotel for hotel in hotels if hotel["id"] == profile["selected_hotel_id"]]
            compared_list.extend(hotels)
            evaluated_regions.append((candidate, await AgentService.evaluate_hotels(
                candidate["id"], hotels, profile,
                flexible_dates=(is_replan_cheaper and is_any_time and not profile.get("date_shifted")),
            )))

        if selected_hotel_query:
            selected_options = evaluated_regions[0][1] if evaluated_regions else []
            if not selected_options:
                reply = f"Для отеля «{selected_hotel_query}» не найден совместимый маршрут с известной стоимостью. Текущий вариант сохранён."
                await AgentService._save_session(user_id, profile, previous_cart, message, reply)
                return {"text": reply, "state": "HOTEL_NOT_AVAILABLE", "cart": previous_cart}
            requested_total = selected_options[0][2]["total_trip_budget_range_rub"][estimate_key]
            if budget_limit is not None and requested_total > budget_limit:
                reply = (f"Отель «{selected_options[0][0]['name']}»: полная оценка {requested_total:,} ₽, "
                         f"на {requested_total - budget_limit:,} ₽ выше лимита {budget_limit:,} ₽. Текущий вариант сохранён.")
                await AgentService._save_session(user_id, profile, previous_cart, message, reply)
                return {"text": reply, "state": "HOTEL_OVER_BUDGET", "cart": previous_cart}

        evaluated = evaluated_regions[0][1] if evaluated_regions else []
        if search_multiple_destinations:
            regional_options = [(candidate, option) for candidate, options in evaluated_regions for option in options]
            affordable_regions = [(candidate, option) for candidate, option in regional_options if budget_limit is None
                                  or option[2]["total_trip_budget_range_rub"][estimate_key] <= budget_limit]
            if affordable_regions or regional_options:
                top_dest, _ = min(affordable_regions or regional_options,
                                  key=lambda item: item[1][2]["total_trip_budget_range_rub"][estimate_key])
                dest_id, dest_name = top_dest["id"], top_dest["name"]
                evaluated = next(options for candidate, options in evaluated_regions if candidate["id"] == dest_id)
                cart.update(destination_id=dest_id, destination=dest_name)
            cart["destination_search"] = {"considered": len(candidates), "affordable_options": len(affordable_regions)}
        recommended_hotel = compared_list[0] if compared_list else None

        if not recommended_hotel:
            constraints = []
            if profile.get("has_pets"):
                constraints.append("размещение с питомцем")
            if profile.get("has_kids"):
                constraints.append("размещение с детьми, без Adults Only")
            if profile.get("private_room_required"):
                constraints.append("отдельный номер, без хостелов и общих удобств")
            if budget_limit is not None:
                constraints.append(f"общий бюджет до {budget_limit:,} ₽")
            if rejected_hotels:
                constraints.append("без ранее отклонённых отелей")
            details = "; ".join(constraints) or "текущие условия поиска"
            reply_text = (
                f"В направлении «{dest_name}» нет отелей под ваши условия: {details}. "
                "Я сохранил ограничения. Можно выбрать другое направление или явно изменить условия. "
                "Бронирование недоступно, пока не подобран подходящий отель."
            )
            # Старое предложение больше не соответствует текущим условиям.
            # Убираем его из корзины, чтобы его нельзя было случайно оформить.
            cart = {
                "destination_id": dest_id, "destination": dest_name,
                "nights": nights, "dates": profile.get("dates"),
                "selection_status": "no_matches",
            }
            await AgentService._save_session(user_id, profile, cart, message, reply_text)
            return {
                "text": reply_text, "state": "NO_MATCHES",
                "compared_hotels": [], "cart": cart,
            }

        affordable = [option for option in evaluated if budget_limit is None or
                      option[2]["total_trip_budget_range_rub"][estimate_key] <= budget_limit]

        affordable, ltr_meta = LTRService.rank_evaluated_packages(affordable, dest_id, profile, nights)

        is_luxury = any(t in ("luxury", "extreme_luxury") for t in profile.get("interests", []))
        is_budget = "budget_friendly" in profile.get("interests", []) or is_replan_cheaper or is_cheapest_requested

        if is_luxury and not is_cheapest_requested:
            # Пользователь готов раскошелиться / выбрал люкс: рекомендуем премиум-отель с максимальным комфортом в рамках бюджета
            affordable.sort(key=lambda x: (-x[0]["price_per_night_rub"], -x[0].get("ltr_score", 0.0)))
        elif is_cheapest_requested or is_budget or (budget_limit is not None and not is_luxury):
            # Пользователь ищет доступный вариант или ограничил бюджет: рекомендуем наиболее экономичный вариант
            affordable.sort(key=lambda x: (x[2]["total_trip_budget_range_rub"][estimate_key], -x[0].get("ltr_score", 0.0), x[0]["price_per_night_rub"]))

        if not affordable:
            if evaluated and budget_limit is not None:
                lowest_option = min(evaluated, key=lambda x: x[2]["total_trip_budget_range_rub"][estimate_key])
                lowest = lowest_option[2]["total_trip_budget_range_rub"][estimate_key]
                closest_hotel = lowest_option[0]
                dest_name = single_destination_name(dest_id, dest_name, closest_hotel.get("city"))
                estimate_label = "нижней" if estimate_key == "min" else "верхней"
                reply_text = (
                    f"⚠️ Среди найденных вариантов нет поездки в пределах бюджета {budget_limit:,} ₽. "
                    f"В направлении «{dest_name}» ближайший найденный вариант — отель «{closest_hotel['name']}» "
                    f"с {estimate_label} оценкой полной стоимости {lowest:,} ₽ (на {lowest - budget_limit:,} ₽ выше лимита). "
                    "В расчёт включены перелёт, отель, трансфер, питание и активности. "
                    "Оценка получена по демонстрационному каталогу. "
                    "Можно сократить поездку, выбрать другое направление или явно увеличить бюджет."
                )
                state = "BUDGET_EXCEEDED"
                if search_multiple_destinations:
                    reply_text = f"Проверил {len(candidates)} направлений по вашим условиям.\n\n" + reply_text
            else:
                reply_text = "Не удалось собрать маршрут с известными ценами перелёта и трансфера. Бронирование пока недоступно."
                state = "NO_ROUTE"
            cart = {"destination_id": dest_id, "destination": dest_name,
                    "nights": nights, "dates": profile.get("dates"), "selection_status": state.lower(),
                    "checkin_date": profile.get("checkin_date"), "checkout_date": profile.get("checkout_date"),
                    "budget_rub": budget_limit, "budget_estimate": estimate_key}
            if evaluated and budget_limit is not None:
                update_trip_dates(lowest_option[2]["checkin_date"], profile, cart)
                cart["budget_quote"] = AgentService.budget_quote(dest_name, lowest_option[0], lowest_option[2])
                cart["total_trip_budget_range_rub"] = deepcopy(lowest_option[2]["total_trip_budget_range_rub"])
                cart["pricing_inputs"] = deepcopy(lowest_option[2]["pricing_inputs"])
            await AgentService._save_session(user_id, profile, cart, message, reply_text)
            return {"text": reply_text, "state": state, "cart": cart, "compared_hotels": []}

        recommended_hotel, logistics_info, budget_breakdown = affordable[0]
        dest_name = single_destination_name(dest_id, top_dest["name"], recommended_hotel.get("city"))
        top_dest = {**top_dest, "name": dest_name}
        cart["destination"] = dest_name
        cart["hotel_city"] = recommended_hotel.get("city")
        if requested_hotel_id:
            profile["selected_hotel_id"] = requested_hotel_id
        unchanged_conditions = all(profile.get(key) == previous_slots[key] for key in slot_keys)
        if (is_replan_cheaper and not budget_changed and unchanged_conditions
                and previous_cart.get("hotel_id") == recommended_hotel["id"]
                and previous_cart.get("checkin_date") == budget_breakdown["checkin_date"]
                and previous_cart.get("checkout_date") == budget_breakdown["checkout_date"]
                and previous_cart.get("total_trip_budget_range_rub") == budget_breakdown["total_trip_budget_range_rub"]):
            inputs = budget_breakdown["pricing_inputs"]
            options = AgentService.calculate_flexible_dates_options(
                dest_id, budget_breakdown["mandatory"]["total_mandatory_rub"], budget_breakdown["checkin_date"], nights,
                inputs["flight_roundtrip_per_person_rub"], inputs["hotel_per_room_night_rub"],
                inputs["fixed_costs_rub"], inputs["rooms_count"], inputs["travelers_count"],
            )
            bounds = budget_breakdown["total_trip_budget_range_rub"]
            reply = (f"Более дешёвого варианта среди найденных нет при тех же датах, направлении, "
                     f"{travelers_count} путешественниках и {profile.get('rooms_count', 1)} номерах. "
                     f"Текущий отель — «{recommended_hotel['name']}», полная оценка {bounds['min']:,}–{bounds['max']:,} ₽.")
            cheaper_dates = next((option for option in options if option["savings_rub"] > 0), None)
            if cheaper_dates:
                reply += (f"\n\nЕсли разрешите изменить даты: {cheaper_dates['dates_formatted']} — "
                          f"верхняя оценка {bounds['max'] - cheaper_dates['savings_rub']:,} ₽, "
                          f"экономия {cheaper_dates['savings_rub']:,} ₽. Для применения: «{cheaper_dates['message']}».")
            else:
                reply += "\n\nДля дальнейшего снижения можно сравнить другие направления или изменить длительность."
            await AgentService._save_session(user_id, profile, previous_cart, message, reply)
            return {"text": reply, "state": "NO_CHEAPER_OPTION", "cart": previous_cart,
                    "budget_breakdown": budget_breakdown, "flexible_dates": options}
        update_trip_dates(budget_breakdown["checkin_date"], profile)
        if budget_breakdown["selected_flexible_date"]:
            profile["date_shifted"] = True
            cart["date_shifted"] = True
        compared_list = [x[0] for x in affordable[:3]]
        policy = "user_choice" if profile.get("selected_hotel_id") else "luxury" if is_luxury and not is_cheapest_requested else "budget" if is_cheapest_requested or is_budget or budget_limit is not None else "ltr"
        ltr_meta.update(top_k=len(compared_list), top_score=recommended_hotel["ltr_score"],
                        top_hotel=recommended_hotel["name"], selection_policy=policy)
        flight_info = logistics_info["flight"]
        hotel_cost_per_night = recommended_hotel["price_per_night_rub"]

        # 9. Вызов MCP: Сезонные опасности и форс-мажоры + реальный прогноз погоды
        risks_report = await MCPTools.get_destination_risks(
            destination_id=dest_id,
            month=profile.get("checkin_date", "не указан")
        )
        weather_report = await MCPTools.get_live_weather(dest_id)

        # Сохранение в корзину
        cart.pop("selection_status", None)
        cart.pop("original_transfer", None)
        cart["without_transfer"] = profile.get("without_transfer", False)
        cart["budget_breakdown"] = deepcopy(budget_breakdown)
        if recommended_hotel:
            cart["hotel_id"] = recommended_hotel["id"]
            cart["hotel"] = recommended_hotel["name"]
            cart["price_per_night_rub"] = hotel_cost_per_night
            cart["price_per_night_formatted"] = recommended_hotel["price_per_night_formatted"]
        if flight_info:
            cart["flight"] = f"{flight_info.get('airline', 'Авиа')} {flight_info.get('flight_no', '')} ({flight_info.get('origin_name')} ⇄ {logistics_info.get('airport')})"
        
        tr_info = logistics_info.get("transfer", {})
        cart["transfer"] = tr_info.get("name", "Трансфер")
        cart["logistics_check"] = logistics_info.get("logistics_status", "Состыковано")
        new_mandatory_rub = budget_breakdown["mandatory"]["total_mandatory_rub"]
        cart["total_mandatory_rub"] = new_mandatory_rub
        old_total_max = cart.get("total_trip_budget_range_rub", {}).get("max")
        cart["total_trip_budget_range_rub"] = deepcopy(budget_breakdown["total_trip_budget_range_rub"])
        cart["budget_rub"] = budget_limit
        cart["budget_estimate"] = estimate_key
        cart["budget_quote"] = AgentService.budget_quote(dest_name, recommended_hotel, budget_breakdown)
        cart["pricing_inputs"] = deepcopy(budget_breakdown["pricing_inputs"])
        cart["price_per_night_rub"] = budget_breakdown["mandatory"]["hotel_per_night"]
        cart["price_per_night_formatted"] = f"{cart['price_per_night_rub']:,} ₽"

        cart["onsite_spending_guide"] = (
            f"Рекомендуемый бюджет на месте: от {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['min']:,} "
            f"до {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['max']:,} ₽"
        )

        # 10. Формирование сжатого и понятного ответа
        response_lines = []

        # Краткий вежливый ответ ИИ при уточнениях (Пункт 1 & 6 обратной связи пользователя)
        ai_intro = profile.pop("llm_ai_intro", None)
        if not ai_intro:
            if is_replan_flight_day:
                ai_intro = "Понял вас — никаких подъёмов в 6 утра: подобрал рейс с комфортным дневным вылетом, чтобы можно было выспаться, и пересчитал логистику:"
            elif is_replan_flight_later:
                ai_intro = "Понял вас! Подобрал рейс с более удобным вылетом после 10:00 (без раннего подъёма в 6 утра) и обновил расписание трансфера:"
            elif is_replan_flight_morning:
                ai_intro = "Принято! Выбрал утренний рейс и пересчитал логистику маршрута:"
            elif is_replan_hotel and old_hotel_name and old_hotel_name != recommended_hotel["name"]:
                ai_intro = f"Договорились! Исключил отель «{old_hotel_name}», подобрал альтернативу «{recommended_hotel['name']}» и пересчитал бюджет:"
            elif is_replan_cheaper and old_total_max is not None and old_total_max > budget_breakdown["total_trip_budget_range_rub"]["max"]:
                savings = old_total_max - budget_breakdown["total_trip_budget_range_rub"]["max"]
                ai_intro = f"Понял вас! Подобрал более доступный вариант, снизив бюджет поездки на {savings:,} ₽:"
            elif is_replan_cheaper and old_total_max is not None:
                ai_intro = "Пересчитал варианты с сохранением условий. Более дешёвого варианта среди найденных нет:"
            elif is_replan_transfer:
                ai_intro = ("Исключил трансфер из расчёта — самостоятельная дорога оплачивается отдельно:"
                            if profile.get("without_transfer") else "Вернул трансфер в расчёт поездки:")
            elif is_replan_flight:
                fl_type_str = "только прямой беспосадочный рейс" if profile.get("prefer_direct_flight") else "рейс с комфортной стыковкой"
                ai_intro = f"Маршрут перестроен: теперь подобран {fl_type_str}:"
            elif is_replan_nights:
                ai_intro = f"Принято! Изменил длительность тура на {nights} ночей и пересчитал полную стоимость:"
            elif is_replan_dates:
                ai_intro = f"Даты поездки успешно обновлены на {profile.get('dates')}:"
            elif is_abroad_request and current_dest_id in ("sochi", "spb", "altai"):
                ai_intro = f"Отличный выбор! Переключил маршрут на зарубежный курорт ({top_dest['name']}) и подобрал международный перелёт:"
            elif budget_changed and budget_limit is not None:
                ai_intro = f"Подобрал оптимальный тур, который укладывается в ваш бюджет {budget_limit:,} ₽:"

        # Даты тура
        if not profile.get("checkin_date"):
            today = date.today()
            base_date = today + timedelta(days=14)
            profile["checkin_date"] = base_date.isoformat()
            profile["checkout_date"] = (base_date + timedelta(days=nights)).isoformat()
            profile["dates"] = f"{base_date:%d.%m.%Y} — {base_date + timedelta(days=nights):%d.%m.%Y} ({nights} ночей)"

        d_in_str = profile.get("checkin_date")
        d_out_str = profile.get("checkout_date")
        try:
            d_in = date.fromisoformat(d_in_str) if d_in_str else date.today() + timedelta(days=14)
        except Exception:
            d_in = date.today() + timedelta(days=14)
        # Реальный тариф на авиарейсы и проживание через единый метод расчёта тарифа
        raw_flight_rt = budget_breakdown["pricing_inputs"]["flight_roundtrip_per_person_rub"]
        tr_info = logistics_info.get("transfer", {})
        tr_price = budget_breakdown["mandatory"]["transfer"]
        tr_name = tr_info.get("name", "Комфорт-трансфер")
        tr_min = tr_info.get("duration_min", 30)

        # Автоматический выбор самой выгодной даты вылета при запросе самого дешевого варианта
        best_flex_found = budget_breakdown["selected_flexible_date"]
        date_shifted_for_cheapest = best_flex_found is not None

        date_range_str = format_russian_date_range(d_in_str, d_out_str, nights)
        fl_date_out = format_flight_date(d_in_str)
        fl_date_ret = format_flight_date(d_out_str)
        hotel_dates_str = format_hotel_dates(d_in_str, d_out_str, nights)

        dest_display_name = dest_name

        if is_cheapest_requested and not (
            old_total_max is not None and budget_breakdown["total_trip_budget_range_rub"]["max"] >= old_total_max
        ):
            if date_shifted_for_cheapest and best_flex_found:
                ai_intro = f"Подобрал для вас самый дешёвый вариант под ключ с минимальным тарифом: сместил вылет на {abs(best_flex_found['days_shift'])} дня {best_flex_found['direction']} ({best_flex_found['target_weekday']}, {best_flex_found['dates_formatted']}), чтобы билеты и отель вышли максимально выгодными:"
            else:
                ai_intro = "Собрал для вас самый экономичный вариант тура под ключ с самыми дешёвыми билетами и отелем:"

        package_parts = "с перелётом и отелем; трансфер исключён" if profile.get("without_transfer") else "с перелётом, отелем и трансфером"
        default_intro = (
            f"Собрал для вас самый экономичный тур в {dest_display_name} под ключ с дешёвыми билетами и отелем:"
            if is_cheapest_requested else
            f"Подобрал для вас проверенный тур в {dest_display_name} {package_parts}:"
        )
        ai_intro = AgentService.sanitize_ai_intro(
            intro=ai_intro,
            dest_id=dest_id,
            default_intro=default_intro,
            route_name=dest_name,
        )

        if ai_intro:
            response_lines.append(f"💬 {ai_intro}")
            response_lines.append("")

        response_lines.append(f"🌴 **{dest_display_name}** | {top_dest.get('climate_summary', '')}")
        response_lines.append(f"📅 **Даты поездки:** {date_range_str} *(даты можно изменить: напишите «смени даты на ...»)*")
        if weather_report.get("available"):
            response_lines.append(f"🌤️ **Прогноз погоды на неделю ({weather_report.get('source')}):** {weather_report['summary']}")
        response_lines.append("")

        timeline = logistics_info.get("timeline", {})
        layover = timeline.get("layover_details", {})
        flight_code = f"{flight_info.get('airline')} {flight_info.get('flight_no')}"
        flight_times = f"{flight_info.get('origin_code', 'MOW')} {timeline.get('flight_departure')} ➔ {logistics_info.get('airport')} {timeline.get('flight_arrival')}"

        ret_fl = flight_info.get("return_flight", {})
        ret_airline = ret_fl.get("airline", flight_info.get("airline"))
        ret_no = ret_fl.get("flight_no", "")
        ret_dep = ret_fl.get("departure_time") or "нет данных"
        ret_arr = ret_fl.get("arrival_time") or "нет данных"
        dest_code = ret_fl.get("origin_code", logistics_info.get("airport", "DEST"))
        orig_code = ret_fl.get("dest_code", flight_info.get("origin_code", "SVO"))

        h = recommended_hotel
        rooms_cnt = profile.get("rooms_count", 1)

        priced_mandatory = budget_breakdown["mandatory"]
        flight_out_price = priced_mandatory["flight_out_price"]
        flight_ret_price = priced_mandatory["flight_ret_price"]
        hotel_cost_per_night = priced_mandatory["hotel_per_night"]
        hotel_total_stay = priced_mandatory["hotel_total"]
        quoted_hotel_rate = f"{hotel_cost_per_night:,} ₽"
        last_two = travelers_count % 100
        passenger_word = ("пассажир" if travelers_count % 10 == 1 and last_two != 11 else
                          "пассажира" if travelers_count % 10 in (2, 3, 4) and last_two not in (12, 13, 14) else "пассажиров")
        out_tickets = f"{priced_mandatory['flight_out_per_person_rub']:,} ₽ × {travelers_count} {passenger_word} = {flight_out_price:,} ₽"
        return_tickets = f"{priced_mandatory['flight_ret_per_person_rub']:,} ₽ × {travelers_count} {passenger_word} = {flight_ret_price:,} ₽"

        response_lines.append(f"🗺️ **Программа и расчёт тура ({nights} ночей):**")
        response_lines.append("")

        # 1. Перелет туда
        if layover.get("is_connecting"):
            response_lines.append(f"1️⃣ 🛫 **Перелёт туда{fl_date_out}:** {flight_code} ({flight_times}), пересадка в {layover.get('city')} ({layover.get('duration_formatted')}) — **{out_tickets}**")
        else:
            response_lines.append(f"1️⃣ 🛫 **Перелёт туда{fl_date_out}:** Прямой рейс {flight_code} ({flight_times}) — **{out_tickets}**")

        # 2. Трансфер в отель
        if profile.get("without_transfer"):
            response_lines.append("2️⃣ 🧳 **Трансфер в отель исключён.** Самостоятельная дорога оплачивается отдельно и в расчёт не входит.")
        else:
            response_lines.append(f"2️⃣ 🚗 **Трансфер в отель:** {tr_name} (~{tr_min} мин): встреча в аэропорту ({timeline.get('transfer_start')}) ➔ прибытие в отель к {timeline.get('hotel_arrival')} — **{tr_price:,} ₽**")

        # 3. Отель
        if rooms_cnt > 1:
            response_lines.append(f"3️⃣ 🏨 **Проживание в отеле ({hotel_dates_str}):** **«{h['name']}»** ({h['stars']}) — **{quoted_hotel_rate}/ночь за номер** ({hotel_total_stay:,} ₽ за {rooms_cnt} ном., {profile.get('room_type', '2 номера')}) — **{hotel_total_stay:,} ₽**")
        else:
            response_lines.append(f"3️⃣ 🏨 **Проживание в отеле ({hotel_dates_str}):** **«{h['name']}»** ({h['stars']}) — **{quoted_hotel_rate}/ночь** — **{hotel_total_stay:,} ₽**")
        response_lines.append(f"• Питание: {h['meals']} | До моря: {h['distance_to_sea']} | Отмена: {h['cancellation']}")

        # 4. Перелет обратно
        response_lines.append(f"4️⃣ 🛬 **Перелёт обратно{fl_date_ret}:** {ret_airline} {ret_no} ({dest_code} {ret_dep} ➔ {orig_code} {ret_arr}) — **{return_tickets}**")
        response_lines.append("")

        if dest_id in ("turkey", "sochi", "uae", "altai"):
            other_city = "Стамбул" if dest_id == "turkey" else ("Красную Поляну" if dest_id == "sochi" else ("Абу-Даби" if dest_id == "uae" else "Чемал"))
            response_lines.append(f"💡 *Хотите посмотреть несколько городов? Напишите «Составь тур по городам», чтобы объединить в одном туре {recommended_hotel.get('city', 'курорт')} и {other_city} с внутренним трансфером.*")
            response_lines.append("")

        # Финансовый блок
        mand = budget_breakdown["mandatory"]
        onsite = budget_breakdown["onsite_flexible"]
        total_range = budget_breakdown["total_trip_budget_range_rub"]

        fin_lines = [
            f"💰 **Расчёт стоимости всей поездки (для отеля «{recommended_hotel['name']}» на {nights} н.):**",
            f"• 🔴 **Обязательные расходы (перелёт туда + трансфер + проживание + перелёт обратно):** {mand['total_mandatory_rub']:,} ₽",
        ]
        if profile.get("has_pets") and mand.get("pet_total", 0) > 0:
            fin_lines.append(f"• 🐾 **Доплата за питомца в отеле:** {mand.get('pet_surcharge_per_night', 0):,} ₽/ночь ({mand['pet_total']:,} ₽ за {nights} н.) — включена в обязательные расходы")
        fin_lines.extend([
            f"• 🟢 **Расходы на месте ({nights} дн. питание + досуг):** {onsite['total_onsite_range_rub']['min']:,} – {onsite['total_onsite_range_rub']['max']:,} ₽",
            f"💵 **ИТОГО под ключ:** **{total_range['min']:,} – {total_range['max']:,} ₽**",
            ""
        ])
        response_lines.extend(fin_lines)
        response_lines.append("*Расчёт по демонстрационному каталогу с календарными коэффициентами; актуальная цена поставщика не запрашивалась.*\n")

        if budget_limit is not None and total_range['max'] > budget_limit:
            response_lines.append(
                f"⚠️ Лимит всей поездки: {budget_limit:,} ₽ превышен на {total_range['max'] - budget_limit:,} ₽. "
                f"Верхняя оценка расходов: {total_range['max']:,} ₽. Нижняя граница — оптимистичный ориентир. "
                f"Попробуйте увеличить бюджет, сократить длительность или выбрать другое направление.\n"
            )

        if len(compared_list) > 1:
            response_lines.append("📋 **Альтернативные отели (для сравнения):**")
            for alt in compared_list[1:3]:
                alt_breakdown = next(option[2] for option in affordable if option[0]["id"] == alt["id"])
                diff_rub = alt_breakdown["total_trip_budget_range_rub"][estimate_key] - total_range[estimate_key]
                if diff_rub > 0:
                    diff_str = f"+{diff_rub:,} ₽ к общему бюджету"
                elif diff_rub < 0:
                    diff_str = f"{diff_rub:,} ₽ к общему бюджету"
                else:
                    diff_str = "такая же стоимость"
                response_lines.append(f"• **«{alt['name']}»** ({alt['stars']}, {alt['price_per_night_formatted']}/ночь, {alt['meals']}) — {diff_str}. _(Напишите «выбери {alt['name']}», чтобы переключиться)_")
            response_lines.append("")

        # Сезонные особенности (сжато)
        active_alerts = risks_report.get("active_alerts", [])
        if active_alerts:
            top_risks = "; ".join(f"{a['title']}" for a in active_alerts[:2])
            response_lines.append(f"ℹ️ **Сезонные особенности:** {top_risks}.")
            response_lines.append("")
        # Паспортный и визовый комплаенс
        compliance_info = ComplianceService.get_compliance_info(dest_id, profile)
        response_lines.append(compliance_info["summary"])
        response_lines.append("")

        # Календарь выгоды (Гибкие даты ±2 дня)
        flexible_dates = AgentService.calculate_flexible_dates_options(
            dest_id=dest_id,
            base_mandatory_rub=mand["total_mandatory_rub"],
            checkin_date_str=profile.get("checkin_date"),
            nights=nights,
            base_flight_price=raw_flight_rt,
            base_hotel_price=h["price_per_night_rub"],
            fixed_costs=tr_price + (mand.get("pet_total", 0) or 0),
            rooms_count=rooms_cnt,
            travelers_count=travelers_count
        )
        if flexible_dates:
            best_flex = flexible_dates[0]
            if best_flex["savings_rub"] > 0:
                response_lines.append("📅 **Календарь выгоды (Гибкие даты ±2 дня):**")
                response_lines.append(
                    f"• 💡 Вылет на {abs(best_flex['days_shift'])} дня {best_flex['direction']} "
                    f"({best_flex['dates_formatted']}, {best_flex['target_weekday']}) "
                    f"сэкономит **~{best_flex['savings_rub']:,} ₽** (поездка обойдётся в {best_flex['target_mandatory_rub']:,} ₽ вместо {mand['total_mandatory_rub']:,} ₽)."
                )
                response_lines.append(f"• *Нажмите на подсказку ниже или напишите «{best_flex['message']}», чтобы пересчитать.*")
                response_lines.append("")
            else:
                cur_day_ru = AgentService.DAY_NAMES_RU[d_in.weekday()]
                response_lines.append("📅 **Календарь тарифов (Гибкие даты ±2 дня):**")
                response_lines.append(
                    f"• 🌟 **Вы выбрали самый выгодный день вылета:** {cur_day_ru} ({date_range_str}) — "
                    f"минимальный тариф **{mand['total_mandatory_rub']:,} ₽**. Вылеты на соседние даты стоят дороже или столько же."
                )
                response_lines.append("")

        response_lines.append("🛒 *Напишите «Забронировать» для покупки или уточните: «другой отель», «подешевле», «только прямой».*")

        rooms_cnt = profile.get("rooms_count", 1)
        reasoning_steps = [
            {"stage": "nlu", "title": "Анализ запроса и профиля", "detail": f"Направление «{top_dest.get('name')}», {nights} н., {profile.get('adults_count', 2)} взр., {rooms_cnt} ном. ({profile.get('room_type', '1 номер')})."},
            {"stage": "filtering", "title": "Фильтрация каталога", "detail": f"Оценено {len(evaluated)} пакетов с учётом ограничений (дети={profile.get('has_kids')}, питомцы={profile.get('has_pets')})."},
            {"stage": "ltr_ranking", "title": "ML-ранжирование (CatBoost LTR)", "detail": f"Движок {ltr_meta['engine']}: оценено {ltr_meta['candidates_count']} пакетов за {ltr_meta['latency_ms']} мс; выбрано {ltr_meta['top_k']}. Архетип: {ltr_meta['archetype']}; приоритет: {policy}."},
            {"stage": "logistics", "title": "Графовая логистика Neo4j", "detail": f"Рейс {flight_info.get('airline', '')} {flight_info.get('flight_no', '')}, трансфер {tr_info.get('name', 'стандарт')}, безопасный транзитный буфер."},
            {"stage": "compliance", "title": "Паспортный и визовый комплаенс", "detail": f"{compliance_info.get('passport_required')} ({compliance_info.get('passport_validity')}), {compliance_info.get('visa_policy')}."},
            {"stage": "budget", "title": "Двухкомпонентный расчёт", "detail": f"Обязательные расходы: {budget_breakdown['mandatory']['total_mandatory_rub']:,} ₽ (номеров: {rooms_cnt}), на месте: {budget_breakdown['onsite_flexible']['total_onsite_range_rub']['min']:,}–{budget_breakdown['onsite_flexible']['total_onsite_range_rub']['max']:,} ₽."}
        ]
        if flexible_dates:
            reasoning_steps.append({"stage": "dates", "title": "Календарь выгоды", "detail": f"Проверены соседние даты (±2 дня): максимальная выгода до {flexible_dates[0]['savings_rub']:,} ₽."})

        reasoning_html = [
            "<details class=\"agent-reasoning\">",
            f"<summary>🧠 <b>Ход рассуждений агента ({len(reasoning_steps)} проверок)</b></summary>",
            "<div class=\"reasoning-body\">"
        ]
        for idx, s in enumerate(reasoning_steps, 1):
            reasoning_html.append(f"<div class=\"reasoning-step\"><span class=\"step-num\">{idx}.</span> <strong>{s['title']}:</strong> {s['detail']}</div>")
        reasoning_html.append("</div>")
        reasoning_html.append("</details>")
        reasoning_block = "\n".join(reasoning_html)

        reply_text = f"{reasoning_block}\n\n" + "\n".join(response_lines)

        cart["rooms_count"] = rooms_cnt
        cart["room_type"] = profile.get("room_type", "Стандарт")
        cart["ltr_ranking"] = ltr_meta
        cart["compliance"] = compliance_info
        cart["flexible_dates"] = flexible_dates
        cart["reasoning_steps"] = reasoning_steps
        cart["logistics"] = logistics_info
        cart["checkin_date"] = profile.get("checkin_date")
        cart["checkout_date"] = profile.get("checkout_date")
        cart["dates"] = profile.get("dates")

        await AgentService._save_session(user_id, profile, cart, message, reply_text)

        return {
            "text": reply_text,
            "destination": top_dest,
            "compared_hotels": [
                {**hotel, "catalog_price_per_night_rub": hotel["price_per_night_rub"],
                 "price_per_night_rub": breakdown["mandatory"]["hotel_per_night"],
                 "price_per_night_formatted": f"{breakdown['mandatory']['hotel_per_night']:,} ₽",
                 "total_trip_budget_range_rub": deepcopy(breakdown["total_trip_budget_range_rub"])}
                for hotel, _, breakdown in affordable[:3]
            ],
            "logistics": logistics_info,
            "budget_breakdown": budget_breakdown,
            "risks_report": risks_report,
            "weather": weather_report,
            "cart": cart,
            "reasoning_steps": reasoning_steps,
            "flexible_dates": flexible_dates,
            "compliance_info": compliance_info,
            "rooms_count": rooms_cnt,
            "ltr_ranking": ltr_meta
        }

    @staticmethod
    async def reset_trip(user_id: str, forget_preferences: bool = False) -> dict:
        async with async_session_maker() as session:
            user_session = await session.get(UserSessionModel, user_id)
            preferences = {}
            if user_session:
                preferences = {} if forget_preferences else remembered_preferences(user_session.profile or {})
                user_session.profile = preferences
                user_session.current_cart = {}
                user_session.conversation_history = []
                user_session.current_state = "INSPIRATION"
                await session.commit()
            return preferences

    @staticmethod
    async def _save_session(user_id: str, profile: dict, cart: dict, user_msg: str, agent_reply: str):
        async with async_session_maker() as session:
            sess_to_update = await session.get(UserSessionModel, user_id)
            if sess_to_update:
                sess_to_update.profile = deepcopy(profile)
                sess_to_update.current_cart = deepcopy(cart)
                sess_history = list(sess_to_update.conversation_history or [])
                sess_history.append({"user": user_msg, "agent": agent_reply})
                sess_to_update.conversation_history = sess_history
                await session.commit()
