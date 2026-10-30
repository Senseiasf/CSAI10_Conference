import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService
from app.services.graph_service import GraphService


def sample_hotel(identifier, name, city, price):
    return {
        "id": identifier,
        "name": name,
        "city": city,
        "price_per_night_rub": price,
        "price_per_night_formatted": f"{price:,} ₽",
        "meals": "Завтрак включен",
        "stars": "4★",
        "distance_to_sea": "300 м",
        "cancellation": "Бесплатно за 24 ч",
        "features": ["wifi", "pool"]
    }


def make_logistics(airline="Аэрофлот", flight_no="SU-2144", dep="13:40", arr="18:35", orig="SVO", dest="AYT"):
    return {
        "compatible": True,
        "flight": {
            "airline": airline,
            "flight_no": flight_no,
            "origin_name": "Шереметьево",
            "origin_code": orig,
            "departure_time": dep,
            "arrival_time": arr,
            "dest_code": dest,
            "price_rub_roundtrip": 26500,
            "return_flight": {
                "airline": airline,
                "flight_no": "SU-2143",
                "departure_time": "15:45",
                "arrival_time": "20:30",
                "origin_code": dest,
                "dest_code": orig
            }
        },
        "transfer": {"price_rub": 2500, "name": "Комфорт-трансфер"},
        "timeline": {
            "flight_departure": dep,
            "flight_arrival": arr,
            "transfer_start": "19:05",
            "hotel_arrival": "19:50"
        }
    }


class SlotExtractionTest(unittest.TestCase):
    def test_day_flight_preference_extraction(self):
        phrases = [
            "хочу вылетать не в 6 утра, а днем",
            "не в 6 утра, а днём",
            "рейс днем",
            "дневной вылет",
            "хочу дневным вылетом",
            "вылет в обед"
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                profile = AgentService.extract_interests_and_slots(phrase, {})
                self.assertEqual(profile.get("flight_time_pref"), "day")

    def test_evening_and_morning_flight_preferences(self):
        profile_ev = AgentService.extract_interests_and_slots("вылет вечером после 19:00", {})
        self.assertEqual(profile_ev.get("flight_time_pref"), "evening")

        profile_mo = AgentService.extract_interests_and_slots("утренний рейс пораньше", {})
        self.assertEqual(profile_mo.get("flight_time_pref"), "morning")

        profile_lat = AgentService.extract_interests_and_slots("самолет попозже, после 12", {})
        self.assertEqual(profile_lat.get("flight_time_pref"), "later")

    def test_origin_airport_and_direct_flight_with_time_pref(self):
        profile = AgentService.extract_interests_and_slots("вылет из Пулково днем, только прямой рейс", {})
        self.assertEqual(profile.get("origin_airport"), "LED")
        self.assertEqual(profile.get("flight_time_pref"), "day")

    def test_multi_city_intent_extraction(self):
        phrases = [
            "составь тур по городам",
            "хочу тур по нескольким городам",
            "много городов у моря",
            "комбинированный тур",
            "хочу посмотреть и Анталью, и Стамбул",
            "хочу и в горы, и на побережье Сочи",
            "поездка в Дубай и Абу-Даби",
            "тур по двум городам"
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                profile = AgentService.extract_interests_and_slots(phrase, {})
                self.assertTrue(profile.get("is_multi_city"), f"Failed for phrase: {phrase}")


class MultiCityRegionsTest(unittest.IsolatedAsyncioTestCase):
    async def test_turkey_multi_city_itinerary(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={"destination_id": "turkey", "destination": "Турция (Анталья и Стамбул)"},
            current_state="INSPIRATION"
        )
        dest = {"id": "turkey", "name": "Турция (Анталья и Стамбул)", "climate_summary": "Средиземноморье", "score": 1.0}
        h1 = sample_hotel("h_ayt", "Antalya City Boutique", "Анталья", 5800)
        h2 = sample_hotel("h_ist", "Istanbul Old Town Hotel", "Стамбул", 7200)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            
            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Анталь" in city:
                    return {"compared_hotels": [h1]}
                return {"compared_hotels": [h2]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics())))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-tr", "составь тур по городам")

        self.assertTrue(result.get("is_multi_city"))
        self.assertIn("Анталья", result["text"])
        self.assertIn("Стамбул", result["text"])
        self.assertIn("Antalya City Boutique", result["text"])
        self.assertIn("Istanbul Old Town Hotel", result["text"])
        self.assertIn("TK-2415", result["text"])
        self.assertEqual(result["cart"]["destination"], "Турция (Анталья + Стамбул)")

    async def test_sochi_multi_city_coast_and_mountains(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={},
            current_state="INSPIRATION"
        )
        dest = {"id": "sochi", "name": "Сочи", "climate_summary": "Субтропики", "score": 1.0}
        h_sirius = sample_hotel("h_sir", "Сириус Ривьера", "Сириус", 6500)
        h_polyana = sample_hotel("h_pol", "Гранд Спа Поляна", "Красная Поляна", 9200)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Сириус" in city:
                    return {"compared_hotels": [h_sirius]}
                return {"compared_hotels": [h_polyana]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics(dest="AER"))))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-sochi", "комбинированный тур в Сочи: хочу и на побережье, и в Красную Поляну")

        self.assertTrue(result.get("is_multi_city"))
        self.assertIn("Сириус", result["text"])
        self.assertIn("Красная Поляна", result["text"])
        self.assertIn("Ласточка", result["text"])
        self.assertIn("Сириус Ривьера", result["text"])
        self.assertIn("Гранд Спа Поляна", result["text"])

    async def test_uae_multi_city_dubai_and_abu_dhabi(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={},
            current_state="INSPIRATION"
        )
        dest = {"id": "uae", "name": "ОАЭ", "climate_summary": "Персидский залив", "score": 1.0}
        h_dxb = sample_hotel("h_dxb", "Rixos Premium Dubai", "Дубай", 18500)
        h_auh = sample_hotel("h_auh", "Emirates Palace Abu Dhabi", "Абу-Даби", 22000)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Дубай" in city:
                    return {"compared_hotels": [h_dxb]}
                return {"compared_hotels": [h_auh]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics(dest="DXB"))))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-uae", "составь тур по городам в ОАЭ: в Дубай и Абу-Даби")

        self.assertTrue(result.get("is_multi_city"))
        self.assertIn("Дубай", result["text"])
        self.assertIn("Абу-Даби", result["text"])
        self.assertIn("экспресс-трансфер", result["text"])
        self.assertIn("Rixos Premium Dubai", result["text"])
        self.assertIn("Emirates Palace", result["text"])

    async def test_altai_multi_city_gorno_altaysk_and_chemal(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={},
            current_state="INSPIRATION"
        )
        dest = {"id": "altai", "name": "Алтай", "climate_summary": "Горный воздух", "score": 1.0}
        h_gorno = sample_hotel("h_alt1", "Эко-курорт Манжерок", "Горно-Алтайск", 12000)
        h_chemal = sample_hotel("h_alt2", "Чемальская Лагуна", "Чемал", 8500)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Горно" in city:
                    return {"compared_hotels": [h_gorno]}
                return {"compared_hotels": [h_chemal]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics(dest="RGK"))))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-altai", "комбинированный тур на Алтай в Горно-Алтайск и Чемал")

        self.assertTrue(result.get("is_multi_city"))
        self.assertIn("Горно-Алтайск", result["text"])
        self.assertIn("Чемал", result["text"])
        self.assertIn("Чуйскому тракту", result["text"])


class MultiCityConstraintsAndEdgeCasesTest(unittest.IsolatedAsyncioTestCase):
    async def test_multi_city_preserves_pet_and_kids_constraints(self):
        user = SimpleNamespace(
            profile={"nights": 7, "has_pets": True, "has_kids": True},
            current_cart={},
            current_state="INSPIRATION"
        )
        dest = {"id": "turkey", "name": "Турция (Анталья и Стамбул)", "climate_summary": "Средиземноморье", "score": 1.0}
        h1 = sample_hotel("h1", "Family Pet Resort", "Анталья", 6000)
        h2 = sample_hotel("h2", "Grand Family Istanbul", "Стамбул", 8000)

        received_calls = []

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                received_calls.append({"city": city, "kwargs": kwargs})
                if city and "Анталь" in city:
                    return {"compared_hotels": [h1]}
                return {"compared_hotels": [h2]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics())))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            await AgentService.process_message("test-constraints", "составь тур по городам в Турцию с ребенком и собакой")

        self.assertGreaterEqual(len(received_calls), 2)
        for call in received_calls:
            self.assertTrue(call["kwargs"].get("pet_friendly"), "pet_friendly must be passed to compare_hotels")
            self.assertTrue(call["kwargs"].get("family_friendly"), "family_friendly must be passed to compare_hotels")

    async def test_multi_city_without_transfer(self):
        user = SimpleNamespace(
            profile={"nights": 7, "without_transfer": True},
            current_cart={},
            current_state="INSPIRATION"
        )
        dest = {"id": "turkey", "name": "Турция (Анталья и Стамбул)", "climate_summary": "Средиземноморье", "score": 1.0}
        h1 = sample_hotel("h1", "Antalya Hotel", "Анталья", 5000)
        h2 = sample_hotel("h2", "Istanbul Hotel", "Стамбул", 7000)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Анталь" in city:
                    return {"compared_hotels": [h1]}
                return {"compared_hotels": [h2]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics())))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-notr", "комбинированный тур в Турцию, но без трансфера")

        self.assertIn("Трансферы исключены по вашему запросу", result["text"])

    async def test_single_city_to_multi_city_progression(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={"destination_id": "turkey", "destination": "Турция (Анталья)", "hotel": "Antalya City Boutique"},
            current_state="OFFER_PRESENTED"
        )
        dest = {"id": "turkey", "name": "Турция (Анталья и Стамбул)", "climate_summary": "Средиземноморье", "score": 1.0}
        h1 = sample_hotel("h1", "Antalya City Boutique", "Анталья", 5800)
        h2 = sample_hotel("h2", "Istanbul Old Town Hotel", "Стамбул", 7200)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Анталь" in city:
                    return {"compared_hotels": [h1]}
                return {"compared_hotels": [h2]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics())))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-progression", "а составь тур по городам")

        self.assertTrue(result.get("is_multi_city"))
        self.assertEqual(result["cart"]["destination_id"], "turkey")
        self.assertIn("Анталья", result["text"])
        self.assertIn("Стамбул", result["text"])

    async def test_dynamic_night_splits(self):
        user = SimpleNamespace(
            profile={"nights": 10},
            current_cart={},
            current_state="INSPIRATION"
        )
        dest = {"id": "turkey", "name": "Турция (Анталья и Стамбул)", "climate_summary": "Средиземноморье", "score": 1.0}
        h1 = sample_hotel("h1", "Antalya Hotel", "Анталья", 5000)
        h2 = sample_hotel("h2", "Istanbul Hotel", "Стамбул", 7000)

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Анталь" in city:
                    return {"compared_hotels": [h1]}
                return {"compared_hotels": [h2]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=make_logistics())))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-10n", "тур по городам на 10 ночей в Турцию")

        self.assertIn("10 ночей", result["text"])
        self.assertIn("6 ночи/ночей", result["text"])
        self.assertIn("4 ночи/ночей", result["text"])


class DaytimeFlightReplanningTest(unittest.IsolatedAsyncioTestCase):
    async def test_replan_flight_to_daytime(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={"destination_id": "turkey", "destination": "Турция (Анталья)"},
            current_state="OFFER_PRESENTED"
        )
        dest_turkey = {
            "id": "turkey",
            "name": "Турция (Анталья и Стамбул)",
            "country": "Турция",
            "climate_summary": "Средиземноморский климат",
            "score": 1.0
        }
        hotel = sample_hotel("h_ayt_1", "Antalya City Boutique Hotel", "Анталья", 5800)
        day_logistics = make_logistics(flight_no="SU-2144", dep="13:40", arr="18:35")

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest_turkey])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": [hotel]})))
            logistics_mock = stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=day_logistics)))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-day", "хочу вылетать не в 6 утра, а днем")

        self.assertEqual(logistics_mock.call_args.kwargs.get("flight_time_pref"), "day")
        self.assertIn("дневным вылетом", result["text"])
        self.assertIn("13:40", result["text"])
        self.assertIn("Турция (Анталья)", result["text"])
        self.assertIn("Составь тур по городам", result["text"])

    async def test_direct_flight_combined_with_day_flight(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={"destination_id": "sochi", "destination": "Сочи"},
            current_state="OFFER_PRESENTED"
        )
        dest_sochi = {"id": "sochi", "name": "Сочи", "climate_summary": "Субтропики", "score": 1.0}
        hotel = sample_hotel("h_sochi", "Сочи Парк Отель", "Сириус", 5000)
        day_logistics = make_logistics(airline="Победа", flight_no="DP-115", dep="12:20", arr="16:05", dest="AER")

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest_sochi])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))
            stack.enter_context(patch.object(MCPTools, "compare_hotels", new=AsyncMock(return_value={"compared_hotels": [hotel]})))
            logistics_mock = stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=day_logistics)))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-direct-day", "только прямой рейс и вылет днем")

        self.assertEqual(logistics_mock.call_args.kwargs.get("flight_time_pref"), "day")
        self.assertTrue(logistics_mock.call_args.kwargs.get("prefer_direct_flight"))
        self.assertIn("12:20", result["text"])


class DynamicMultiCityRouteSelectionTest(unittest.TestCase):
    def test_parametric_selection_sea_and_mountains(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "хочу тур по городам: море и горы",
            {}
        )
        self.assertEqual(dest_id, "sochi")
        self.assertIn("Сочи (Побережье + Красная Поляна)", cfg["title"])
        self.assertIn("сочетание теплого моря и горных панорам", rationale)

    def test_parametric_selection_culture_and_beach(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "хочу тур по городам: культура, Босфор и пляж",
            {}
        )
        self.assertEqual(dest_id, "turkey")
        self.assertIn("Турция", cfg["title"])
        self.assertIn("Босфор", rationale)

    def test_parametric_selection_shopping_skyscrapers(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "хочу тур по городам: шопинг и небоскребы",
            {}
        )
        self.assertEqual(dest_id, "uae")
        self.assertIn("ОАЭ", cfg["title"])
        self.assertIn("шопинг", rationale)

    def test_parametric_selection_nature_and_rivers(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "комбинированный тур: дикая природа, реки и чистый воздух",
            {}
        )
        self.assertEqual(dest_id, "altai")
        self.assertIn("Алтай", cfg["title"])
        self.assertIn("природ", rationale)

    def test_parametric_selection_palaces_and_museums(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "тур по нескольким городам: дворцы, каналы и Эрмитаж",
            {}
        )
        self.assertEqual(dest_id, "spb")
        self.assertIn("Санкт-Петербург", cfg["title"])
        self.assertIn("Эрмитаж", rationale)

    def test_parametric_selection_corals_and_diving(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "тур по нескольким курортам: коралловые рифы и дайвинг",
            {}
        )
        self.assertEqual(dest_id, "egypt")
        self.assertIn("Египет", cfg["title"])
        self.assertIn("коралл", rationale)

    def test_parametric_selection_islands_and_tropics(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "комбинированный тур: тропические острова и экзотика",
            {}
        )
        self.assertEqual(dest_id, "thailand")
        self.assertIn("Таиланд", cfg["title"])
        self.assertIn("экзотик", rationale)

    def test_domestic_constraint_filters_foreign_destinations(self):
        dest_id, cfg, rationale = AgentService.select_multi_city_route(
            "тур по городам по России: культура и дворцы",
            {"domestic_only": True}
        )
        self.assertEqual(dest_id, "spb")
        self.assertEqual(cfg["country"], "Россия")


class DynamicMultiCityProcessMessageTest(unittest.IsolatedAsyncioTestCase):
    async def test_process_message_dynamically_selects_sochi_for_sea_and_mountains(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={},
            current_state="OFFER_PRESENTED"
        )
        dest_sochi = {
            "id": "sochi",
            "name": "Сочи и Красная Поляна",
            "country": "Россия",
            "climate_summary": "Субтропики",
            "score": 1.0
        }
        h_sirius = sample_hotel("h_sirius", "Отель Сириус Побережье", "Сириус", 5200)
        h_polyana = sample_hotel("h_polyana", "Гранд Поляна Спа", "Красная Поляна", 8900)
        logistics = make_logistics(dest="AER")

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest_sochi])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and ("сириус" in city.lower() or "побереж" in city.lower()):
                    return {"compared_hotels": [h_sirius]}
                return {"compared_hotels": [h_polyana]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))

            result = await AgentService.process_message("test-dynamic-sochi", "хочу тур по городам: море и горы на 7 ночей")

        self.assertIn("Сочи (Побережье + Красная Поляна)", result["text"])
        self.assertIn("Сириус", result["text"])
        self.assertIn("Красная Поляна", result["text"])
        self.assertIn("Ласточка", result["text"])
        self.assertIn("сочетание теплого моря и горных панорам", result["text"])
        self.assertIn("Красная Поляна", result["destination"]["name"])
        self.assertEqual(result["cart"]["destination"], result["destination"]["name"])
        self.assertEqual(len(result["compared_hotels"]), 2)
        self.assertNotIn("Аудит агента", result["text"])
        self.assertNotIn("Safety Check", result["text"])

    async def test_assemble_multi_city_itinerary_with_graph_flight(self):
        user = SimpleNamespace(
            profile={"nights": 7},
            current_cart={"destination_id": "turkey", "destination": "Турция"},
            current_state="OFFER_PRESENTED"
        )
        dest_turkey = {
            "id": "turkey",
            "name": "Турция (Анталья + Стамбул)",
            "country": "Турция",
            "climate_summary": "Средиземноморье",
            "score": 1.0
        }
        h_ayt = sample_hotel("h_ayt", "Antalya Beach Resort", "Анталья", 6000)
        h_ist = sample_hotel("h_ist", "Bosphorus Palace Hotel", "Стамбул", 8500)
        logistics = make_logistics(dest="AYT")
        mock_graph_flight = {
            "flight_no": "TK-2415",
            "airline": "Turkish Airlines",
            "departure_time": "11:30",
            "arrival_time": "12:45",
            "duration_minutes": 75,
            "price_rub": 3800
        }

        with ExitStack() as stack:
            stack.enter_context(patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)))
            stack.enter_context(patch.object(AgentService, "search_destinations_vector", new=AsyncMock(return_value=[dest_turkey])))
            stack.enter_context(patch.object(AgentService, "_save_session", new_callable=AsyncMock))

            async def mock_compare_hotels(destination_id, city=None, **kwargs):
                if city and "Анталь" in city:
                    return {"compared_hotels": [h_ayt]}
                return {"compared_hotels": [h_ist]}

            stack.enter_context(patch.object(MCPTools, "compare_hotels", side_effect=mock_compare_hotels))
            stack.enter_context(patch.object(MCPTools, "check_logistics_compatibility", new=AsyncMock(return_value=logistics)))
            stack.enter_context(patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={})))
            stack.enter_context(patch.object(GraphService, "find_intercity_logistics", new=AsyncMock(return_value=mock_graph_flight)))

            result = await AgentService.process_message("test-graph-flight", "составь тур по городам в Турцию")

        self.assertIn("TK-2415", result["text"])
        self.assertIn("подтверждён графом логистики Neo4j", result["text"])


if __name__ == "__main__":
    unittest.main()
