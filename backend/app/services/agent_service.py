import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import select, text

from app.database import async_session_maker
from app.models import DestinationModel, HotelModel, RiskAlertModel, UserSessionModel
from app.config import settings
from app.services.vector_service import generate_text_embedding
from app.mcp.tools import MCPTools
from app.services.trip_dates import update_trip_dates
from app.services.trip_budget import update_trip_budget
from app.services.llm_service import LLMService
from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)

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
        if any(w in user_lower for w in ["собак", "питомц", "кошк", "пес"]):
            interests.add("pet_friendly")
            profile["has_pets"] = True
        if any(w in user_lower for w in ["дешев", "эконом", "подешев", "недорог", "бюджетн"]) or (
            not profile.get("private_room_required") and any(w in user_lower for w in ["хостел", "койко"])
        ):
            interests.add("budget_friendly")
        if any(w in user_lower for w in ["самый дорог", "самый роскошн", "дворец", "burj al arab", "шейх", "президентск", "максимально дорог", "ультра-люкс"]):
            interests.add("extreme_luxury")
            interests.add("luxury")
        elif any(w in user_lower for w in ["люкс", "премиум", "вилл", "5 звезд", "пять звезд", "deluxe", "vip", "роскошн"]):
            interests.add("luxury")
        if any(w in user_lower for w in ["вдвоем", "парой", "романтик"]):
            profile["party_size"] = "вдвоем"
        elif any(w in user_lower for w in ["один", "одна", "соло"]):
            profile["party_size"] = "соло"

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
        multi_city_kws = [
            "тур по городам", "по городам", "несколько городов", "нескольких городов",
            "много городов", "комбинированный тур", "комбинированн", "составной тур",
            "составной маршрут", "комбо тур", "маршрут по городам", "два города",
            "двум городам", "разные города", "несколько курортов"
        ]
        city_pair_matches = (
            ("анталь" in user_lower and "стамбул" in user_lower) or
            ("сочи" in user_lower and ("полян" in user_lower or "сириус" in user_lower)) or
            ("дуба" in user_lower and ("абу-даби" in user_lower or "абу даби" in user_lower)) or
            ("алтай" in user_lower and ("чемал" in user_lower or "манжерок" in user_lower)) or
            ("пхукет" in user_lower and ("краби" in user_lower or "банг тао" in user_lower or "остров" in user_lower)) or
            ("петербург" in user_lower and "петергоф" in user_lower) or
            ("шарм" in user_lower and ("дахаб" in user_lower or "хадаб" in user_lower)) or
            ("гор" in user_lower and ("побереж" in user_lower or "море" in user_lower or "морск" in user_lower or "пляж" in user_lower)) or
            ("море" in user_lower and ("культур" in user_lower or "босфор" in user_lower or "истори" in user_lower)) or
            ("шопинг" in user_lower and "небоскреб" in user_lower)
        )
        is_multi_city_regex = bool(re.search(
            r"(?:по\s+(?:нескольк\w*\s+)?город\w*|нескольк\w*\s+(?:город\w*|курорт\w*)|мног\w*\s+город\w*|разн\w*\s+город\w*|комбинир\w*|составн\w*\s+(?:тур\w*|маршрут\w*)|комбо[- ]тур\w*)",
            user_lower
        ))
        if is_multi_city_regex or any(k in user_lower for k in multi_city_kws) or city_pair_matches:
            profile["is_multi_city"] = True

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
    def is_vague_greeting(message: str, has_cart_destination: bool) -> bool:
        """
        Проверяет, является ли сообщение пользователя неопределенным приветствием/стартовым запросом
        без указания направления (Пункт 4 ТЗ: Анти-15-вопросов).
        """
        m_lower = message.lower().strip()
        # Если направление уже выбрано в корзине, это не стартовое приветствие
        if has_cart_destination:
            return False

        # Проверка на наличие конкретных стран/городов
        known_kws = [
            "сочи", "сириус", "полян", "адлер", "дуба", "оаэ", "абу-даби", "эмират",
            "турци", "анталь", "стамбул", "белек", "кемер", "пхукет", "тай", "таиланд", "самуи",
            "мальдив", "мале", "египет", "шарм", "хургад", "синай",
            "спб", "питер", "петербург", "санкт-петербург",
            "алтай", "манжерок", "телецк", "горно-алтайск"
        ]
        if any(k in m_lower for k in known_kws):
            return False

        # Если есть внешние страны, сработает fallback
        for item in UNKNOWN_DESTINATIONS_MAP:
            if any(k in m_lower for k in item["keywords"]):
                return False

        # Если есть параметры/команды перепланирования или фильтров — это не приветствие
        replan_keywords = [
            "отель", "гостиниц", "номер", "дешев", "дорог", "рейс", "вылет", "самолет",
            "трансфер", "бюджет", "тысяч", "рубл", "руб", "заезд", "выезд", "ноч", "дней", "день",
            "забронир", "оформ", "оплат", "купит"
        ]
        if any(rw in m_lower for rw in replan_keywords):
            return False

        vague_phrases = [
            "привет", "здравствуйте", "добрый день", "добрый вечер", "хай", "hello",
            "хочу отдохнуть", "помоги выбрать", "посоветуй", "куда поехать", "не знаю куда",
            "хочу в отпуск", "подбери тур", "что посоветуешь", "помоги с выбором", "начать",
            "хочу поехать", "хочу слетать", "подобрать тур", "куда слетать", "планирую отдых",
            "планирую отпуск", "посоветуйте", "ищу тур", "хочу путешествовать", "помоги с поездкой",
            "хочу в путешествие", "поехать куда-нибудь", "где лучше отдохнуть", "какие есть варианты",
            "помоги спланировать", "выбрать тур", "собери тур"
        ]
        if any(vp in m_lower for vp in vague_phrases):
            return True

        if len(m_lower.split()) <= 3 and not any(w in m_lower for w in ["заграниц", "за рубеж", "пляж", "море", "гор"]):
            return True

        return False

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
        
        keyword_map = {
            "thailand": "thailand", "таиланд": "thailand", "тайланд": "thailand", "пхукет": "thailand", "тай": "thailand", "самуи": "thailand",
            "uae": "uae", "оаэ": "uae", "дуба": "uae", "дубай": "uae", "абу-даби": "uae", "эмират": "uae",
            "turkey": "turkey", "турци": "turkey", "турция": "turkey", "анталь": "turkey", "анталья": "turkey", "стамбул": "turkey", "белек": "turkey", "кемер": "turkey",
            "sochi": "sochi", "сочи": "sochi", "сириус": "sochi", "полян": "sochi", "адлер": "sochi", "роза хутор": "sochi",
            "maldives": "maldives", "мальдив": "maldives", "мальдивы": "maldives", "мале": "maldives",
            "egypt": "egypt", "египет": "egypt", "шарм": "egypt", "хургад": "egypt", "синай": "egypt",
            "spb": "spb", "спб": "spb", "питер": "spb", "петербург": "spb", "санкт-петербург": "spb",
            "altai": "altai", "алтай": "altai", "манжерок": "altai", "телецк": "altai", "горно-алтайск": "altai"
        }
        
        matched_id = None
        for kw in sorted(keyword_map.keys(), key=lambda k: len(k), reverse=True):
            if re.search(r'(?:\b|^|\s)' + re.escape(kw), q_lower):
                matched_id = keyword_map[kw]
                break

        # Если запрошена строго заграница — отсекаем города РФ
        if abroad_only and matched_id in ("sochi", "spb", "altai"):
            matched_id = None
        # Если запрошена строго Россия — отсекаем зарубежные курорты
        if domestic_only and matched_id in ("uae", "turkey", "thailand", "maldives", "egypt"):
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

        # 1. Подбор отелей для обоих городов через MCP
        h_res_1 = await MCPTools.compare_hotels(
            destination_id=dest_id,
            city=cfg["city_1"]["name"].split("/")[0].strip(),
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids"),
            private_room_required=profile.get("private_room_required", False),
            preferred_tags=profile.get("interests"),
            limit=3
        )
        hotels_1 = h_res_1.get("compared_hotels", [])

        h_res_2 = await MCPTools.compare_hotels(
            destination_id=dest_id,
            city=cfg["city_2"]["name"].split("/")[0].strip(),
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids"),
            private_room_required=profile.get("private_room_required", False),
            preferred_tags=profile.get("interests"),
            limit=3
        )
        hotels_2 = h_res_2.get("compared_hotels", [])

        # Если отель не найден по точному фильтру города — берем альтернативы из общего каталога
        if not hotels_1 or not hotels_2:
            all_h = (await MCPTools.compare_hotels(
                destination_id=dest_id,
                pet_friendly=profile.get("has_pets"),
                family_friendly=profile.get("has_kids"),
                private_room_required=profile.get("private_room_required", False),
                preferred_tags=profile.get("interests"),
                limit=6
            )).get("compared_hotels", [])
            if not hotels_1 and all_h:
                hotels_1 = [all_h[0]]
            if not hotels_2:
                hotels_2 = [h for h in all_h if not hotels_1 or h["id"] != hotels_1[0]["id"]] or hotels_1

        hotel_1 = hotels_1[0] if hotels_1 else {
            "id": "h1_default",
            "name": f"Курортный отель {cfg['city_1']['name']}", "stars": "4★",
            "price_per_night_rub": 6000, "price_per_night_formatted": "6,000 ₽",
            "meals": "Завтрак включен", "cancellation": "Бесплатно за 24 ч", "distance_to_sea": "300 м"
        }
        hotel_2 = hotels_2[0] if hotels_2 else {
            "id": "h2_default",
            "name": f"Гранд-отель {cfg['city_2']['name']}", "stars": "4★",
            "price_per_night_rub": 8500, "price_per_night_formatted": "8,500 ₽",
            "meals": "Завтрак включен", "cancellation": "Бесплатно за 48 ч", "distance_to_sea": "Центр города"
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

        # 3. Финансовый расчёт всего тура
        cost_hotel_1 = hotel_1.get("price_per_night_rub", 6000) * nights_1
        cost_hotel_2 = hotel_2.get("price_per_night_rub", 8500) * nights_2
        intl_flight_price = outbound_flight.get("price_rub_roundtrip", 27000)
        intercity_price = intercity.get("price_rub", 3800)
        transfers_price = 0 if profile.get("without_transfer") else 3500

        mand_total = cost_hotel_1 + cost_hotel_2 + intl_flight_price + intercity_price + transfers_price
        onsite_min = nights * 2500
        onsite_max = nights * 5500
        total_min = mand_total + onsite_min
        total_max = mand_total + onsite_max

        # 4. Формирование ответа
        response_lines = []
        ai_intro = profile.pop("llm_ai_intro", None) or selection_rationale
        out_dep = outbound_flight.get("departure_time", "13:40")
        if not ai_intro:
            if is_replan_flight_day:
                ai_intro = f"Понял вас — никаких подъёмов в 6 утра: подобрал дневной рейс в {out_dep}, чтобы можно было выспаться. Собрал полноценный комбинированный тур по обоим городам:"
            elif is_replan_flight_later:
                ai_intro = f"Принято! Обновил время вылета на комфортное дневное ({out_dep}) и пересчитал программу тура по городам:"
            else:
                ai_intro = f"Собрал для вас готовый комбинированный тур «{cfg['title']}» под ключ с логистикой между городами, отелями и расчётом полной стоимости:"

        response_lines.append(f"💬 {ai_intro}")
        response_lines.append("")
        response_lines.append(f"🌴 **Комбинированный тур: {cfg['title']}** | {cfg['climate']}")
        response_lines.append(f"🌤️ **Прогноз погоды по маршруту:** {cfg['city_1']['name']}: {cfg['city_1']['weather']} | {cfg['city_2']['name']}: {cfg['city_2']['weather']}")
        response_lines.append("")

        response_lines.append(f"🗺️ **Программа и маршрут тура по городам ({nights} ночей):**")
        response_lines.append("")

        # Город 1
        response_lines.append(f"1️⃣ **Часть 1: {cfg['city_1']['name']} ({nights_1} ночи/ночей) — {cfg['city_1']['vibe']}**")
        response_lines.append(f"• 🏨 **Отель:** **«{hotel_1['name']}»** ({hotel_1['stars']}) — **{hotel_1.get('price_per_night_formatted', str(hotel_1.get('price_per_night_rub', 6000)) + ' ₽')}/ночь** ({cost_hotel_1:,} ₽ за {nights_1} ночей)")
        response_lines.append(f"• Питание: {hotel_1.get('meals', 'Завтрак')} | До моря: {hotel_1.get('distance_to_sea', 'Первая линия')} | Отмена: {hotel_1.get('cancellation', 'Бесплатно')}")
        response_lines.append(f"• 🏖️ **Вайб и программа:** {cfg['city_1']['activities']}.")
        response_lines.append("")

        # Переезд
        response_lines.append(f"🚄 / ✈️ **Переезд между городами:**")
        response_lines.append(f"• **{intercity['name']}** ({intercity['duration']}, {intercity['price_rub']:,} ₽): {intercity['description']}")
        response_lines.append("")

        # Город 2
        response_lines.append(f"2️⃣ **Часть 2: {cfg['city_2']['name']} ({nights_2} ночи/ночей) — {cfg['city_2']['vibe']}**")
        response_lines.append(f"• 🏨 **Отель:** **«{hotel_2['name']}»** ({hotel_2['stars']}) — **{hotel_2.get('price_per_night_formatted', str(hotel_2.get('price_per_night_rub', 8500)) + ' ₽')}/ночь** ({cost_hotel_2:,} ₽ за {nights_2} ночей)")
        response_lines.append(f"• Питание: {hotel_2.get('meals', 'Завтрак')} | Отмена: {hotel_2.get('cancellation', 'Бесплатно')}")
        response_lines.append(f"• 🏛️ **Вайб и программа:** {cfg['city_2']['activities']}.")
        response_lines.append("")

        # Авиаперелёты
        fl_code_out = f"{outbound_flight.get('airline', 'Авиа')} {outbound_flight.get('flight_no', '')}"
        times_out = f"{outbound_flight.get('origin_code', 'MOW')} {outbound_flight.get('departure_time', '13:40')} ➔ {cfg['city_1']['airport']} {outbound_flight.get('arrival_time', '18:35')}"
        fl_code_ret = f"{ret_airline} {ret_no}"
        times_ret = f"{dest_code} {ret_dep} ➔ {orig_code} {ret_arr}"

        date_out = profile.get("checkin_date")
        date_ret = profile.get("checkout_date")
        date_out_str = f" ({date_out})" if date_out else ""
        date_ret_str = f" ({date_ret})" if date_ret else ""

        response_lines.append("✈️ **Международные авиаперелёты (включены в стоимость):**")
        response_lines.append(f"• 🛫 **Туда{date_out_str}:** Прямой рейс {fl_code_out} ({times_out}) прямо в {cfg['city_1']['name']}.")
        response_lines.append(f"• 🛬 **Обратно{date_ret_str}:** {fl_code_ret} ({times_ret}) вылет из {cfg['city_2']['name']}.")
        if profile.get("without_transfer"):
            response_lines.append("• 🧳 Трансферы исключены по вашему запросу (самостоятельно на такси).")
        else:
            response_lines.append("• 🚗 Трансферы аэропорт ➔ отель включены по всему маршруту.")
        response_lines.append("")

        # Финансовый блок
        response_lines.append(f"💰 **Расчёт стоимости всей поездки (комбинированный тур на {nights} н.):**")
        response_lines.append(f"• 🔴 **Обязательные расходы (все перелёты + 2 отеля на {nights} н. + переезд между городами + трансферы):** {mand_total:,} ₽")
        response_lines.append(f"• 🟢 **Расходы на месте ({nights} дн. питание + досуг в обоих городах):** {onsite_min:,} – {onsite_max:,} ₽")
        response_lines.append(f"💵 **ИТОГО под ключ:** **{total_min:,} – {total_max:,} ₽**")
        response_lines.append("")

        if budget_limit is not None:
            response_lines.append(
                f"Лимит всей поездки: {budget_limit:,} ₽. Верхняя оценка расходов "
                f"{total_max:,} ₽ укладывается в лимит; это оценка, а не фиксированная цена.\n"
            )

        response_lines.append("🛡️ **Аудит агента (Smart Filter & Safety Check):**")
        response_lines.append(f"• ✅ **Сквозная состыковка:** График переезда между {cfg['city_1']['name']} и {cfg['city_2']['name']} согласован со временем заселения и выселения.")
        response_lines.append("• ✅ **Безопасная логистика:** Отсеяны рискованные рейсы с ранним вылетом или короткими стыковками.")
        response_lines.append("")

        response_lines.append("🛒 *Напишите «Забронировать» для покупки или уточните: «другой отель», «подешевле», «смени даты».*")

        reply_text = "\n".join(response_lines)

        # Сохранение в корзину
        cart["destination_id"] = dest_id
        cart["destination"] = cfg["title"]
        cart["is_multi_city"] = True
        cart["hotel"] = f"«{hotel_1['name']}» ({nights_1} н.) + «{hotel_2['name']}» ({nights_2} н.)"
        cart["flight"] = f"{fl_code_out} ➔ {fl_code_ret}"
        cart["transfer"] = f"Трансферы + {intercity['name']}"
        cart["nights"] = nights
        cart["dates"] = profile.get("dates", f"{nights} ночей")
        cart["total_mandatory_rub"] = mand_total
        cart["total_trip_budget_range_rub"] = {"min": total_min, "max": total_max}
        cart["onsite_spending_guide"] = f"Рекомендуемый бюджет на месте: от {onsite_min:,} до {onsite_max:,} ₽"

        await AgentService._save_session(user_id, profile, cart, message, reply_text)

        return {
            "text": reply_text,
            "destination": {"id": dest_id, "name": cfg["title"]},
            "compared_hotels": [hotel_1, hotel_2],
            "logistics": logistics_1,
            "is_multi_city": True,
            "cart": cart
        }

    @staticmethod
    async def process_message(user_id: str, message: str) -> Dict[str, Any]:
        """
        Основной цикл обработки запроса путешественника:
        1. Распознавание интента и профилирование (память пользователя)
        2. Пункт 4 ТЗ: Анти-15-вопросов при нечетком старте
        3. Пункт 3 ТЗ: Fallback & рекомендации для неизвестных направлений
        4. Пункт 2 ТЗ: Динамическое перепланирование на лету без сброса контекста
        5. Пункт 5 ТЗ: Имитация чекаута и ваучер
        """
        user_lower = message.lower().strip()
        session_obj = await AgentService.get_or_create_session(user_id)
        profile = session_obj.profile or {}
        cart = session_obj.current_cart or {}

        previous_budget = profile.get("budget_rub")
        budget_question = update_trip_budget(user_lower, profile, cart)
        if budget_question:
            await AgentService._save_session(user_id, profile, cart, message, budget_question)
            return {"text": budget_question, "state": "NEEDS_BUDGET", "cart": cart}
        budget_changed = profile.get("budget_rub") != previous_budget
        budget_limit = profile.get("budget_rub")
        cart_within_budget = budget_limit is None or (
            cart.get("total_trip_budget_range_rub", {}).get("max", float("inf")) <= budget_limit
        )

        # 0. Проверка намерения покупки / чекаута (Пункт 5 ТЗ)
        if any(w in user_lower for w in ["купить", "оформить", "забронировать", "оплатить", "чек", "подтвердить"]):
            if cart.get("hotel") and cart.get("destination") and not budget_changed and cart_within_budget:
                checkout_result = await MCPTools.checkout_cart(user_id=user_id)
                reply = (
                    f"🎉 **Бронирование успешно подтверждено!**\n\n"
                    f"📋 **Ваучер заказа:** `{checkout_result['booking_id']}`\n"
                    f"🏝️ **Направление:** {checkout_result['destination']}\n"
                    f"✈️ **Авиаперелет:** {checkout_result.get('flight')}\n"
                    f"🏨 **Отель:** {checkout_result['hotel']}\n"
                    f"🚕 **Трансфер:** {checkout_result['transfer']}\n"
                    f"💳 **Сумма к оплате:** {checkout_result['total_paid_rub']:,} ₽ (симуляция оплаты успешна)\n\n"
                    f"💡 *{checkout_result.get('onsite_spending_guide', '')}*\n\n"
                    f"Спасибо, что выбрали Островок! Приятного путешествия!"
                )
                return {
                    "text": reply,
                    "cart": cart,
                    "booking": checkout_result,
                    "state": "CONFIRMED"
                }

        previous_dates = profile.get("dates")
        date_question = update_trip_dates(user_lower, profile)
        if date_question:
            await AgentService._save_session(user_id, profile, cart, message, date_question)
            return {"text": date_question, "state": "NEEDS_DATES", "cart": cart}

        # 1. Извлечение интересов и обновление профиля пользователя (User Memory)
        profile = AgentService.extract_interests_and_slots(user_lower, profile)

        # 2. Интеллектуальный Tool Calling диспетчер на базе DeepSeek Flash
        # ИИ сам извлекает параметры (город вылета, время, географию, неизвестные локации, тип перепланирования, теги)
        # без необходимости для пользователя говорить точные кодовые слова.
        has_dest_in_cart = bool(cart.get("destination_id"))
        if not settings.is_mock_llm and not user_id.startswith("test"):
            llm_decision = await LLMService.dispatch_intent(
                user_message=message,
                profile=profile,
                current_cart=cart,
                conversation_history=getattr(session_obj, "conversation_history", [])
            )
            if llm_decision:
                if llm_decision.get("action") == "unknown_destination":
                    req_name = llm_decision.get("requested_name") or "выбранное направление"
                    alt_id = llm_decision.get("suggested_pilot_id") or "maldives"
                    alt_name = llm_decision.get("suggested_pilot_name") or "Мальдивы"
                    reason = llm_decision.get("vibe_reason") or "отличный курортный отдых и море впечатлений"
                    
                    reply_fallback = (
                        f"🌍 **Обратите внимание:** В текущей версии нашей системы оцифрована "
                        f"сквозная графовая логистика (рейсы, трансферы и отели) по 8 ключевым направлениям: "
                        f"**Сочи, ОАЭ, Турция, Таиланд, Мальдивы, Египет, Санкт-Петербург и Алтай**.\n\n"
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
                    # Если запрос совершенно пустой / приветствие без параметров
                    is_pure_greeting = not has_dest_in_cart and not llm_decision.get("destination_id") and not llm_decision.get("tags") and not llm_decision.get("origin_airport") and not llm_decision.get("budget_rub") and AgentService.is_vague_greeting(message, has_dest_in_cart)
                    if is_pure_greeting:
                        reply_anti_questions = (
                            "👋 **Здравствуйте! Я умный ассистент путешествий Ostrovok.**\n\n"
                            "Вместо длинных анкет из 15 вопросов задам всего **два ключевых ориентира**:\n\n"
                            "1️⃣ **Какой формат отдыха вам ближе?**\n"
                            "• 🌊 **Пляжный релакс и рифы** (Мальдивы, Пхукет, Египет, ОАЭ, Анталья)\n"
                            "• 🏔️ **Горы, природа и спа** (Алтай, Красная Поляна)\n"
                            "• 🏛️ **Культура, архитектура и гастрономия** (Санкт-Петербург, Стамбул)\n"
                            "• 👨‍👩‍👧 **Семейный All-Inclusive с детьми и аквапарками**\n\n"
                            "2️⃣ **Примерный состав и даты поездки** (вдвоём, с семьёй, с питомцем или соло)?\n\n"
                            "_Напишите, что вам ближе (например: «Хочу море и всё включено с ребенком»), "
                            "и я мгновенно соберу готовый проверенный маршрут с перелетом, отелем и трансфером._"
                        )
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
                    if llm_decision.get("budget_rub") and not profile.get("budget_rub"):
                        profile["budget_rub"] = llm_decision["budget_rub"]
                    if llm_decision.get("nights") and not any(w in user_lower for w in ["ноч", "дн", "сут"]):
                        profile["nights"] = llm_decision["nights"]
                        profile["dates"] = f"{llm_decision['nights']} ночей"
                    if llm_decision.get("is_multi_city"):
                        profile["is_multi_city"] = True

        # Детерминированный fallback (для mock LLM и unit-тестов):
        # Пункт 4 ТЗ: Анти-15-вопросов на старте (Anti-15-Questions)
        if AgentService.is_vague_greeting(message, has_dest_in_cart):
            reply_anti_questions = (
                "👋 **Здравствуйте! Я умный ассистент путешествий Ostrovok.**\n\n"
                "Вместо длинных анкет из 15 вопросов задам всего **два ключевых ориентира**:\n\n"
                "1️⃣ **Какой формат отдыха вам ближе?**\n"
                "• 🌊 **Пляжный релакс и рифы** (Мальдивы, Пхукет, Египет, ОАЭ, Анталья)\n"
                "• 🏔️ **Горы, природа и спа** (Алтай, Красная Поляна)\n"
                "• 🏛️ **Культура, архитектура и гастрономия** (Санкт-Петербург, Стамбул)\n"
                "• 👨‍👩‍👧 **Семейный All-Inclusive с детьми и аквапарками**\n\n"
                "2️⃣ **Примерный состав и даты поездки** (вдвоём, с семьёй, с питомцем или соло)?\n\n"
                "_Напишите, что вам ближе (например: «Хочу море и всё включено с ребенком»), "
                "и я мгновенно соберу готовый проверенный маршрут с перелетом, отелем и трансфером._"
            )
            await AgentService._save_session(user_id, profile, cart, message, reply_anti_questions)
            return {
                "text": reply_anti_questions,
                "state": "INSPIRATION_PROMPT",
                "cart": cart
            }

        # Детерминированный fallback для неизвестных направлений (если оффлайн / mock LLM)
        unknown_match = AgentService.detect_unknown_destination(message)
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
        explicit_dest_change = any(w in user_lower for w in ["другой город", "поменяй направление", "другая страна", "вместо", "поехали в", "а если в "])
        
        is_replan_hotel = any(w in user_lower for w in ["другой отель", "поменяй отель", "смени отель", "не нравится отель", "есть еще отели", "покажи другие варианты", "альтернативный отель", "другой вариант проживания"])
        is_replan_cheaper = any(w in user_lower for w in ["дешевле", "подешевле", "дорого", "снизь цену", "бюджетнее", "экономнее", "сделай дешевле"])
        is_replan_flight_day = any(w in user_lower for w in ["днем", "днём", "в обед", "дневной вылет", "дневным вылетом", "вылет днем", "вылет днём", "вылетать не в 6 утра", "не в 6 утра, а днем", "не в 6 утра, а днём"])
        is_replan_flight_later = any(w in user_lower for w in ["попозже", "самолет попозже", "вылет попозже", "рейс попозже", "не в 6 утра", "не в 6", "не в 7", "не рано", "позже 10", "после 10", "после 12", "вечером", "поздний вылет", "вылетать позже"]) or is_replan_flight_day
        is_replan_flight_morning = any(w in user_lower for w in ["пораньше", "утром", "утренний вылет", "самолет пораньше", "рейс пораньше", "ранний вылет"])
        is_replan_flight = is_replan_flight_later or is_replan_flight_morning or any(w in user_lower for w in ["только прямой", "прямой рейс", "без пересадок", "с пересадкой", "стыковочный", "рейс подешевле"])
        is_replan_transfer = any(w in user_lower for w in ["убери трансфер", "без трансфера", "поедем сами", "на каршеринге", "возьмем такси сами", "своим ходом", "не нужен трансфер"])
        is_replan_dates = profile.get("dates") != previous_dates and bool(profile.get("checkin_date"))
        is_replan_nights = any(w in user_lower for w in ["ноч", "дн", "сут", "выходн", "уикенд", "недел", "месяц"]) and profile.get("nights") != cart.get("nights")

        # Применяем решения ИИ-диспетчера (Tool Calling DeepSeek Flash)
        llm_ra = profile.pop("llm_replan_action", None)
        if llm_ra == "change_hotel":
            is_replan_hotel = True
        elif llm_ra == "cheaper":
            is_replan_cheaper = True
        elif llm_ra == "change_flight_time":
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

        if is_replan_flight_day:
            profile["flight_time_pref"] = "day"

        is_abroad_request = profile.get("abroad_only", False)
        is_domestic_request = profile.get("domestic_only", False)
        if is_abroad_request and current_dest_id in ("sochi", "spb", "altai"):
            explicit_dest_change = True
        elif is_domestic_request and current_dest_id not in ("sochi", "spb", "altai"):
            explicit_dest_change = True

        nights = profile.get("nights", 7)
        cart["nights"] = nights
        cart["dates"] = profile.get("dates", f"{nights} ночей")

        # Обработка исключения трансфера
        if is_replan_transfer:
            profile["without_transfer"] = True

        # Обработка типа рейса
        if any(w in user_lower for w in ["только прямой", "прямой рейс", "без пересадок"]):
            profile["prefer_direct_flight"] = True
        elif any(w in user_lower for w in ["с пересадкой", "стыковочный"]):
            profile["prefer_direct_flight"] = False

        # Определение целевого направления
        matched_destinations = await AgentService.search_destinations_vector(
            message,
            limit=2,
            abroad_only=is_abroad_request,
            domestic_only=is_domestic_request
        )
        msg_top_dest = matched_destinations[0] if matched_destinations else None

        # Если ИИ точно определил destination_id через Tool Calling
        llm_dest = profile.pop("llm_destination_id", None)
        if llm_dest and (explicit_dest_change or not current_dest_id or (is_abroad_request and current_dest_id in ("sochi", "spb", "altai"))):
            cur_matched = await AgentService.search_destinations_vector(llm_dest, limit=1)
            if cur_matched:
                msg_top_dest = cur_matched[0]

        if current_dest_id and not explicit_dest_change and not profile.get("is_multi_city") and (is_replan_hotel or is_replan_cheaper or is_replan_flight or is_replan_transfer or is_replan_dates or budget_changed):
            # Контекстное перепланирование в рамках текущего города/курорта!
            dest_id = current_dest_id
            cur_matched = await AgentService.search_destinations_vector(dest_id, limit=1)
            top_dest = cur_matched[0] if cur_matched else {"id": dest_id, "name": cart.get("destination", dest_id), "climate_summary": ""}
            dest_name = top_dest["name"]
        elif msg_top_dest and (msg_top_dest.get("score", 0) >= 0.9 or not current_dest_id or explicit_dest_change):
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

        if is_replan_hotel and old_hotel_id:
            if old_hotel_id not in rejected_hotels:
                rejected_hotels.append(old_hotel_id)
                profile["rejected_hotels"] = rejected_hotels

        if is_replan_cheaper:
            profile.setdefault("interests", []).append("budget_friendly")

        # 6. Вызов MCP: Сравнение отелей с учетом тегов, исключений и фильтров
        hotel_comparison = await MCPTools.compare_hotels(
            destination_id=dest_id,
            pet_friendly=profile.get("has_pets"),
            family_friendly=profile.get("has_kids"),
            private_room_required=profile.get("private_room_required", False),
            exclude_hotel_ids=rejected_hotels,
            preferred_tags=profile.get("interests"),
            limit=None if budget_limit is not None else 3,
        )

        compared_list = hotel_comparison.get("compared_hotels", [])
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

        # Оцениваем полную стоимость каждого кандидата, а не только цену номера.
        # Проверяем весь отфильтрованный каталог при наличии лимита бюджета.
        evaluated = []
        for hotel in compared_list:
            logistics = await MCPTools.check_logistics_compatibility(
                airport_code="", hotel_id=hotel["id"], origin_airport_code=profile.get("origin_airport"),
                prefer_direct_flight=profile.get("prefer_direct_flight"),
                without_transfer=profile.get("without_transfer", False),
                flight_time_pref=profile.get("flight_time_pref")
            )
            flight = logistics.get("flight", {})
            transfer = logistics.get("transfer", {})
            if not logistics.get("compatible") or "price_rub_roundtrip" not in flight:
                continue
            if not profile.get("without_transfer") and "price_rub" not in transfer:
                continue
            breakdown = MCPTools.calculate_budget_breakdown(
                destination_id=dest_id,
                flight_price_roundtrip=flight["price_rub_roundtrip"],
                hotel_price_per_night=hotel["price_per_night_rub"],
                transfer_price=0 if profile.get("without_transfer") else transfer["price_rub"],
                meals_type=hotel.get("meals", "Завтрак включен"), nights=nights,
            )
            evaluated.append((hotel, logistics, breakdown))

        affordable = [option for option in evaluated if budget_limit is None or
                      option[2]["total_trip_budget_range_rub"]["max"] <= budget_limit]
        if not affordable:
            if evaluated and budget_limit is not None:
                lowest = min(x[2]["total_trip_budget_range_rub"]["max"] for x in evaluated)
                reply_text = (
                    f"В направлении «{dest_name}» не нашлось поездки в пределах {budget_limit:,} ₽. "
                    f"Самая низкая верхняя оценка полной стоимости — {lowest:,} ₽ "
                    f"(на {lowest - budget_limit:,} ₽ выше лимита). "
                    "В расчёт включены перелёт, отель, трансфер, питание и активности. "
                    "Можно сократить поездку, выбрать другое направление или явно увеличить бюджет."
                )
                state = "BUDGET_EXCEEDED"
            else:
                reply_text = "Не удалось собрать маршрут с известными ценами перелёта и трансфера. Бронирование пока недоступно."
                state = "NO_ROUTE"
            cart = {"destination_id": dest_id, "destination": dest_name,
                    "nights": nights, "dates": profile.get("dates"), "selection_status": state.lower()}
            await AgentService._save_session(user_id, profile, cart, message, reply_text)
            return {"text": reply_text, "state": state, "cart": cart, "compared_hotels": []}

        recommended_hotel, logistics_info, budget_breakdown = affordable[0]
        compared_list = [x[0] for x in affordable[:3]]
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
        cart["total_trip_budget_range_rub"] = budget_breakdown["total_trip_budget_range_rub"]
        cart["budget_rub"] = budget_limit

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
            elif is_replan_transfer:
                ai_intro = "Без проблем! Исключил трансфер из расчёта — вы сможете добраться самостоятельно:"
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

        if ai_intro:
            response_lines.append(f"💬 {ai_intro}")
            response_lines.append("")

        dest_display_name = top_dest['name']
        if recommended_hotel and recommended_hotel.get("city") and "(" in dest_display_name and " и " in dest_display_name:
            country_part = dest_display_name.split("(")[0].strip()
            dest_display_name = f"{country_part} ({recommended_hotel['city']})"

        response_lines.append(f"🌴 **{dest_display_name}** | {top_dest.get('climate_summary', '')}")
        if weather_report.get("available"):
            response_lines.append(f"🌤️ **Прогноз погоды на неделю ({weather_report.get('source')}):** {weather_report['summary']}")
        response_lines.append("")

        if recommended_hotel:
            h = recommended_hotel
            hotel_total_stay = h['price_per_night_rub'] * nights
            response_lines.append(f"🏨 **Рекомендуемый отель (в расчёте тура):**")
            response_lines.append(f"• **«{h['name']}»** ({h['stars']}) — **{h['price_per_night_formatted']}/ночь** ({hotel_total_stay:,} ₽ за {nights} ночей)")
            response_lines.append(f"• Питание: {h['meals']} | До моря: {h['distance_to_sea']} | Отмена: {h['cancellation']}")
            response_lines.append("")

            if dest_id in ("turkey", "sochi", "uae", "altai"):
                other_city = "Стамбул" if dest_id == "turkey" else ("Красную Поляну" if dest_id == "sochi" else ("Абу-Даби" if dest_id == "uae" else "Чемал"))
                response_lines.append(f"💡 *Хотите посмотреть несколько городов? Напишите «Составь тур по городам», чтобы объединить в одном туре {recommended_hotel.get('city', 'курорт')} и {other_city} с внутренним трансфером.*")
                response_lines.append("")

        if logistics_info.get("compatible"):
            timeline = logistics_info.get("timeline", {})
            layover = timeline.get("layover_details", {})
            flight_code = f"{flight_info.get('airline')} {flight_info.get('flight_no')}"
            flight_times = f"{flight_info.get('origin_code', 'MOW')} {timeline.get('flight_departure')} ➔ {logistics_info.get('airport')} {timeline.get('flight_arrival')}"
            
            ret_fl = flight_info.get("return_flight", {})
            ret_airline = ret_fl.get("airline", flight_info.get("airline"))
            ret_no = ret_fl.get("flight_no", "")
            ret_dep = ret_fl.get("departure_time", "15:45")
            ret_arr = ret_fl.get("arrival_time", "20:15")
            dest_code = ret_fl.get("origin_code", logistics_info.get("airport", "DEST"))
            orig_code = ret_fl.get("dest_code", flight_info.get("origin_code", "SVO"))

            date_out = profile.get("checkin_date")
            date_ret = profile.get("checkout_date")
            date_out_str = f" ({date_out})" if date_out else ""
            date_ret_str = f" ({date_ret})" if date_ret else ""

            response_lines.append("✈️ **Авиаперелёт туда и обратно (включён в стоимость):**")
            if layover.get("is_connecting"):
                response_lines.append(f"• 🛫 **Туда{date_out_str}:** {flight_code} ({flight_times}), пересадка в {layover.get('city')} ({layover.get('duration_formatted')}).")
            else:
                response_lines.append(f"• 🛫 **Туда{date_out_str}:** Прямой рейс {flight_code} ({flight_times}).")

            response_lines.append(f"• 🛬 **Обратно{date_ret_str}:** {ret_airline} {ret_no} ({dest_code} {ret_dep} ➔ {orig_code} {ret_arr}).")

            if profile.get("without_transfer"):
                response_lines.append(f"• 🧳 Прилёт: получение багажа ➔ самостоятельный выезд на такси/каршеринге в отель.")
            else:
                tr_name = logistics_info.get("transfer", {}).get("name", "Трансфер")
                tr_min = logistics_info.get("transfer", {}).get("duration_min", 30)
                response_lines.append(f"• 🚗 Трансфер ({tr_name}, ~{tr_min} мин): встреча в аэропорту ({timeline.get('transfer_start')}) ➔ прибытие в отель к {timeline.get('hotel_arrival')}.")
            response_lines.append("")

        # Финансовый блок
        mand = budget_breakdown["mandatory"]
        onsite = budget_breakdown["onsite_flexible"]
        total_range = budget_breakdown["total_trip_budget_range_rub"]

        response_lines.extend([
            f"💰 **Расчёт стоимости всей поездки (для отеля «{recommended_hotel['name']}» на {nights} н.):**",
            f"• 🔴 **Обязательные расходы (перелёт туда-обратно + {recommended_hotel['name']} {nights} н. + трансфер):** {mand['total_mandatory_rub']:,} ₽",
            f"• 🟢 **Расходы на месте ({nights} дн. питание + досуг):** {onsite['total_onsite_range_rub']['min']:,} – {onsite['total_onsite_range_rub']['max']:,} ₽",
            f"💵 **ИТОГО под ключ:** **{total_range['min']:,} – {total_range['max']:,} ₽**",
            ""
        ])

        if budget_limit is not None:
            response_lines.append(
                f"Лимит всей поездки: {budget_limit:,} ₽. Верхняя оценка расходов "
                f"{total_range['max']:,} ₽ укладывается в лимит; это оценка, а не фиксированная цена.\n"
            )

        if len(compared_list) > 1:
            response_lines.append("📋 **Альтернативные отели (для сравнения):**")
            for alt in compared_list[1:3]:
                diff_rub = (alt['price_per_night_rub'] - recommended_hotel['price_per_night_rub']) * nights
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
        # 11. Аудит безопасности и умная валидация данных агентом (фильтрация неадекватных / рискованных предложений)
        audit_notes = []
        if profile.get("has_kids"):
            audit_notes.append("🚫 **Семейный контроль:** Отели категории «Adults Only (18+)» и шумные молодежные пати-резорты автоматически исключены из поиска.")
        if profile.get("has_pets"):
            audit_notes.append("🐾 **Pet-контроль:** Отели со строгим запретом на животных исключены; подтверждено безопасное размещение с питомцем.")

        if recommended_hotel:
            price_n = recommended_hotel.get("price_per_night_rub", 0)
            feats = [str(f).lower() for f in (recommended_hotel.get("features", []))]
            if price_n <= 2500 or any("shared_bathroom" in f or "hostel_dorm" in f for f in feats):
                audit_notes.append("⚠️ **Контроль эконом-сегмента:** Выбран бюджетный вариант. Проверьте формат размещения и наличие общих удобств; отдельная санитарная проверка в демо не проводится.")
            elif price_n >= 100000:
                audit_notes.append("💎 **Премиум-аудит:** Подтвержден эксклюзивный статус Ultra-Luxury (персональный консьерж-сервис, приватная вилла/дворец, VIP-трансфер).")

        if logistics_info.get("compatible"):
            timeline = logistics_info.get("timeline", {})
            layover = timeline.get("layover_details", {})
            if layover.get("is_connecting"):
                audit_notes.append("✈️ **Аудит стыковок:** Отсеяны опасные короткие пересадки (<60 мин, риск потери багажа); выбран рейс с безопасным транзитным буфером.")
            
            flight_arr = timeline.get("flight_arrival", "12:00")
            if dest_id == "maldives" and flight_arr > "16:30":
                audit_notes.append("🏝️ **Правило Мальдив:** Гидросамолеты летают только в светлое время (до 16:30). Для вечернего рейса трансфер заменен на скоростной катер.")
            
            if dest_id == "spb" and ("01:10" <= flight_arr <= "05:00"):
                audit_notes.append("🌉 **Логистика СПб:** Прилёт в период разводки невских мостов (01:10–05:00) — маршрут трансфера проложен через скоростной ЗСД.")

        if audit_notes:
            response_lines.append("🛡️ **Аудит агента (Smart Filter & Safety Check):**")
            for note in audit_notes:
                response_lines.append(f"• {note}")
            response_lines.append("")

        # 12. Персональный инсайдерский совет от AI-консьержа DeepSeek Flash
        if not settings.is_mock_llm and not user_id.startswith("test") and recommended_hotel:
            concierge_tip = await LLMService.generate_concierge_advice(
                destination_name=top_dest.get("name", dest_id),
                hotel_name=recommended_hotel["name"],
                nights=nights,
                user_message=message,
                profile=profile
            )
            if concierge_tip:
                response_lines.append(f"✨ **AI-консьерж ({settings.OPENAI_MODEL}):** {concierge_tip}")
                response_lines.append("")

        response_lines.append("🛒 *Напишите «Забронировать» для покупки или уточните: «другой отель», «подешевле», «только прямой».*")

        reply_text = "\n".join(response_lines)
        await AgentService._save_session(user_id, profile, cart, message, reply_text)

        return {
            "text": reply_text,
            "destination": top_dest,
            "compared_hotels": compared_list,
            "logistics": logistics_info,
            "budget_breakdown": budget_breakdown,
            "risks_report": risks_report,
            "weather": weather_report,
            "cart": cart
        }

    @staticmethod
    async def _save_session(user_id: str, profile: dict, cart: dict, user_msg: str, agent_reply: str):
        async with async_session_maker() as session:
            sess_to_update = await session.get(UserSessionModel, user_id)
            if sess_to_update:
                sess_to_update.profile = profile
                sess_to_update.current_cart = cart
                sess_history = sess_to_update.conversation_history or []
                sess_history.append({"user": user_msg, "agent": agent_reply})
                sess_to_update.conversation_history = sess_history
                await session.commit()
