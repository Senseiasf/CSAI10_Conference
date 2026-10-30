import json
import re
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.domain.dialogue import ChangeOperation, DialogueDecision, DialogueIntent, SearchCriteriaDecision, StateChange
from app.domain.trip import TripQuote, TripState
from app.services.llm_service import LLMService, LLMServiceError
from app.services.trip_budget import update_trip_budget
from app.services.trip_dates import update_trip_dates
from app.services.trip_logistics import is_logistics_question
from app.services.trip_state import (
    current_trip_question, destination_choice, estimate_request, hotel_selection, is_trip_edit,
    matching_hotels, rejected_destinations, transfer_preference, wants_cheaper,
)


DIALOGUE_DECISION_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_dialogue_decision",
        "description": "Вернуть единственное решение диалога без выполнения изменений.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": ["question", "comparison", "modify_trip", "plan_trip", "search_alternatives", "select_offer", "correction", "new_trip", "clarify"]},
                "topic": {"type": ["string", "null"]},
                "mode": {"type": ["string", "null"], "enum": ["min", "max", None]},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "field": {"type": "string"},
                            "operation": {"type": "string", "enum": ["set", "clear", "add", "remove"]},
                            "value": {},
                        },
                        "required": ["field", "operation"],
                    },
                },
                "search": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                        "scope": {"type": ["string", "null"], "enum": ["current", "other_destinations", "any", "hotels", "flights", "dates", None]},
                        "destination_query": {"type": ["string", "null"]},
                        "price_preference": {"type": ["string", "null"], "enum": ["cheaper", "more_expensive", "comfort", "any", None]},
                        "missing_fields": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "references": {"type": "array", "items": {"type": "string"}},
                "resume_pending": {"type": "boolean"},
                "clarification": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
                "narrative": {"type": ["string", "null"]},
            },
            "required": ["intent", "changes"],
        },
    },
}]


class DialogueParserError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.public_message = message


class DialogueParser:
    async def parse(
        self,
        message: str,
        state: TripState,
        context: dict[str, Any] | None = None,
        quote: TripQuote | None = None,
    ) -> DialogueDecision:
        context = context or {}
        if settings.is_mock_llm:
            return self._parse_mock(message, state, context)
        try:
            return await self._parse_live(message, state, context, quote)
        except (LLMServiceError, DialogueParserError):
            raise
        except Exception as exc:
            raise DialogueParserError("api", self._safe_error(exc)) from exc

    async def _parse_live(
        self,
        message: str,
        state: TripState,
        context: dict,
        quote: TripQuote | None,
    ) -> DialogueDecision:
        client = LLMService.get_client()
        if client is None:
            raise DialogueParserError("configuration", "Live parser требует LLM_MODE=live")
        payload = {
            "trip_state": state.model_dump(mode="json"),
            "trip_quote": quote.model_dump(mode="json") if quote else None,
            "selected_tour": deepcopy((context.get("cart") or {})),
            "selected_offers": context.get("selected_offers") or state.selected_offers,
            "alternative_offers": context.get("alternative_offers") or (context.get("cart") or {}).get("compared_hotels") or [],
            "conversation": (context.get("conversation_history") or [])[-3:],
            "expected_clarification": state.pending_action.model_dump(mode="json") if state.pending_action else None,
        }
        prompt = (
            "Ты основной языковой обработчик туристического диалога. Верни только DialogueDecision через инструмент. "
            "question никогда не меняет поездку и имеет changes=[]; comparison только сравнивает сохранённые факты; "
            "search_alternatives ищет новые варианты, не подменяя выбранный тур; select_offer выбирает вариант по ID; "
            "correction исправляет предыдущее условие; new_trip разрешён только при явном запросе начать новую поездку; "
            "modify_trip меняет условия; plan_trip продолжает подбор; clarify задаёт точечный вопрос. "
            "Одно сообщение может содержать несколько изменений — верни все операции вместе. 'С двумя друзьями' означает троих взрослых включая автора. "
            "Отсутствие операции сохраняет прежнее значение, явное false отменяет условие, null в set означает неизвестное и требует уточнения. "
            "Не сбрасывай поездку и не повторяй начальную анкету. Краткий ответ продолжает expected_clarification; 'всё то же' использует TripState, а не догадки из истории. "
            "Разделяй budget_rub (лимит), стоимость и search.price_preference. Больший лимит не означает экономию или трату всей суммы. "
            "Ссылки 'тот отель', 'первый вариант' разрешай только по alternative_offers/selected_offers до hotel_id; если неоднозначно — clarify. "
            "Для отказа от трансфера без других правок верни ровно одну операцию field=without_transfer, operation=set, value=true. "
            "Допустимые поля: destination_id, checkin_date, checkout_date, nights, adults_count, kids_count, kids_ages, has_pets, rooms_count, "
            "budget_rub, without_transfer, prefer_direct_flight, flight_time_pref, origin_airport, private_room_required, interests, "
            "hotel_id, rejected_destinations, rejected_hotels, price_preference, search_scope, destination_query, is_multi_city, replan_action. "
            "Для вопроса о пересадке topic=connection_risk, о сохранённом плане topic=current_trip, о границе цены topic=budget_estimate. "
            "Не выдумывай цены, наличие, свойства отелей, разрешение проживания с животными или успешное выполнение действий. Денежные расчёты выполняет код."
        )
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({"message": message, "context": payload}, ensure_ascii=False)},
                ],
                tools=DIALOGUE_DECISION_TOOL,
                tool_choice={"type": "function", "function": {"name": "submit_dialogue_decision"}},
                temperature=0,
                max_tokens=1200,
            )
            message_obj = response.choices[0].message
            raw = None
            if message_obj.tool_calls:
                raw = message_obj.tool_calls[0].function.arguments
            elif message_obj.content:
                raw = message_obj.content
            if not raw:
                raise DialogueParserError("invalid_response", "LLM не вернула DialogueDecision")
            decision = DialogueDecision.model_validate(json.loads(raw))
            LLMService._last_error = None
            return self._resolve_hotel_reference(decision, payload["alternative_offers"])
        except DialogueParserError:
            raise
        except (ValidationError, json.JSONDecodeError) as exc:
            raise DialogueParserError("invalid_response", f"Некорректный DialogueDecision: {exc}") from exc
        except Exception as exc:
            raise DialogueParserError("api", self._safe_error(exc)) from exc

    def _resolve_hotel_reference(self, decision: DialogueDecision, offers: list[dict]) -> DialogueDecision:
        for change in decision.changes:
            if change.field != "hotel_id" or not isinstance(change.value, str):
                continue
            value = change.value
            candidates = [offer for offer in offers if offer.get("id")]
            if any(offer["id"] == value for offer in candidates):
                continue
            normalized = value.casefold().strip()
            matches = [offer for offer in candidates if normalized in str(offer.get("name", "")).casefold()]
            if len(matches) == 1:
                change.value = matches[0]["id"]
            elif len(matches) > 1:
                return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification="Какой именно отель вы имеете в виду?", changes=[])
        return decision

    def _parse_mock(self, message: str, state: TripState, context: dict) -> DialogueDecision:
        lower = message.lower().replace("ё", "е").strip()
        if re.search(r"\b(?:новая поездка|начать заново|сбрось поездку|другое путешествие)\b", lower):
            return DialogueDecision(intent=DialogueIntent.NEW_TRIP)
        if is_logistics_question(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="connection_risk")
        if current_trip_question(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="current_trip")
        if re.search(r"во сколько.*(?:рейс|поезд|переезд)|внутренн\w*\s+рейс", lower) and not is_trip_edit(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="internal_schedule")
        estimate = estimate_request(message)
        if estimate and not is_trip_edit(message):
            return DialogueDecision(intent=DialogueIntent.QUESTION, topic="budget_estimate", mode=estimate)
        if re.search(r"\b(?:сравни|соседн\w*\s+дат|календар\w*\s+(?:цен|тариф|выгод))", lower):
            return DialogueDecision(intent=DialogueIntent.COMPARISON, topic="dates")
        if re.search(r"\bусловия\s+все\s+те\s+же\b|\bвсе\s+то\s+же\b", lower):
            return DialogueDecision(intent=DialogueIntent.PLAN, resume_pending=True)

        direct_destination = destination_choice(message) if not hotel_selection(message) else None
        changes = []
        interests = self._mock_interests(lower)
        if interests and re.search(r"\bя же\b|\bвсе-таки\b|\bвсё-таки\b|\bлучше\b|\bдавай\b", lower):
            changes.append(StateChange(field="interests", operation=ChangeOperation.SET, value=interests))
            return DialogueDecision(
                intent=DialogueIntent.CORRECTION,
                changes=changes,
                search=SearchCriteriaDecision(missing_fields=self._mock_missing(state)),
                references=["previous_message"],
            )
        if re.search(r"друг\w*\s+(?:направлен|город|стран)|другие\s+направления", lower):
            budget_profile = deepcopy(state.apply_to_profile({}))
            cart = context.get("cart") or {}
            update_trip_budget(message, budget_profile, cart)
            if budget_profile.get("budget_rub") != state.budget_rub:
                changes.append(StateChange(field="budget_rub", operation=ChangeOperation.SET, value=budget_profile["budget_rub"]))
            return DialogueDecision(
                intent=DialogueIntent.SEARCH_ALTERNATIVES,
                changes=changes,
                search=SearchCriteriaDecision(scope="other_destinations"),
            )
        if re.search(r"\bдороже\b|\bкомфортнее\b|\bболее\s+(?:дорог|комфорт|лучш)", lower):
            return DialogueDecision(
                intent=DialogueIntent.SEARCH_ALTERNATIVES,
                search=SearchCriteriaDecision(scope="hotels", price_preference="more_expensive"),
            )
        multi_city_request = bool(re.search(r"тур по городам|комбинированн|мульти|несколько городов", lower))
        pending_budget = bool(state.pending_action and "budget_rub" in state.pending_action.missing_fields)
        if (is_trip_edit(message) or multi_city_request or pending_budget or rejected_destinations(message)
                or re.search(r"\b(?:собак\w*|кошк\w*|питомц\w*|пес|пса|псом)\b", lower)):
            selected_query = hotel_selection(message)
            if selected_query:
                offers = context.get("alternative_offers") or []
                if re.fullmatch(r"(?:тот|этот|прежний|текущий)\s+отель", selected_query.lower()):
                    hotel_id = state.selected_offers.get("hotel_id")
                    if hotel_id:
                        return DialogueDecision(intent=DialogueIntent.SELECT_OFFER, changes=[
                            StateChange(field="hotel_id", operation=ChangeOperation.SET, value=hotel_id),
                        ])
                    if len(offers) != 1:
                        return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification="Какой именно отель вы имеете в виду?", references=["selected_hotel"])
                matches = matching_hotels(selected_query, offers)
                if len(matches) > 1:
                    return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification="Какой именно отель вы имеете в виду?", references=["hotel_name"])
                hotel_id = matches[0]["id"] if matches else selected_query
                return DialogueDecision(intent=DialogueIntent.SELECT_OFFER, changes=[
                    StateChange(field="hotel_id", operation=ChangeOperation.SET, value=hotel_id),
                ])
            if direct_destination:
                changes.append(StateChange(field="destination_id", operation=ChangeOperation.SET, value=direct_destination))
            if multi_city_request:
                changes.append(StateChange(field="is_multi_city", operation=ChangeOperation.SET, value=True))
            if interests:
                changes.append(StateChange(field="interests", operation=ChangeOperation.SET, value=interests))
            rejected = rejected_destinations(message)
            if rejected:
                changes.append(StateChange(field="rejected_destinations", operation=ChangeOperation.ADD, value=sorted(rejected)))
            transfer = transfer_preference(message)
            if transfer is not None:
                changes.append(StateChange(field="without_transfer", operation=ChangeOperation.SET, value=transfer))
            nights = re.search(r"\b(\d{1,2})\s*(?:ноч|дн|сут)", lower)
            if nights:
                changes.append(StateChange(field="nights", operation=ChangeOperation.SET, value=int(nights.group(1))))
            friends = re.search(r"\bс\s+(двумя|тремя|четырьмя|\d+)\s+друз", lower)
            if friends:
                friend_count = {"двумя": 2, "тремя": 3, "четырьмя": 4}.get(friends.group(1), friends.group(1))
                changes.append(StateChange(field="adults_count", operation=ChangeOperation.SET, value=int(friend_count) + 1))
            else:
                adults = re.search(r"\bнас\s+(\d+)\b|\b(\d+)\s*взросл", lower)
                if adults:
                    changes.append(StateChange(field="adults_count", operation=ChangeOperation.SET, value=int(adults.group(1) or adults.group(2))))
                elif re.search(r"\b(?:вдвоем|двое взрослых|два взрослых)\b", lower):
                    changes.append(StateChange(field="adults_count", operation=ChangeOperation.SET, value=2))
                elif re.search(r"\b(?:втроем|три взрослых|трое взрослых)\b", lower):
                    changes.append(StateChange(field="adults_count", operation=ChangeOperation.SET, value=3))
                elif re.search(r"\b(?:вчетвером|четыре взрослых|четверо взрослых)\b", lower):
                    changes.append(StateChange(field="adults_count", operation=ChangeOperation.SET, value=4))
            kids = re.search(r"\b(\d+)\s*(?:ребен\w*|детей|дет\w*|малыш\w*)", lower)
            if kids:
                changes.append(StateChange(field="kids_count", operation=ChangeOperation.SET, value=int(kids.group(1))))
            elif re.search(r"\bбез\s+(?:детей|ребенка)\b", lower):
                changes.append(StateChange(field="kids_count", operation=ChangeOperation.SET, value=0))
            rooms = re.search(r"\b(\d+)\s*(?:комнат\w*|номер\w*)", lower)
            if rooms:
                changes.append(StateChange(field="rooms_count", operation=ChangeOperation.SET, value=int(rooms.group(1))))
                changes.append(StateChange(field="private_room_required", operation=ChangeOperation.SET, value=True))
            if re.search(r"\b(?:не\s+хостел\w*|без\s+хостел\w*|отдельн\w*\s+номер\w*|собственн\w*\s+ванн\w*)", lower):
                changes.append(StateChange(field="private_room_required", operation=ChangeOperation.SET, value=True))
            date_profile = deepcopy(state.apply_to_profile({}))
            date_question = update_trip_dates(message, date_profile, context.get("cart") or {})
            if date_question:
                return DialogueDecision(intent=DialogueIntent.CLARIFY, clarification=date_question,
                                        search=SearchCriteriaDecision(missing_fields=["checkin_date"]))
            for field in ("checkin_date", "checkout_date"):
                if date_profile.get(field) and date_profile.get(field) != getattr(state.dates, field):
                    changes.append(StateChange(field=field, operation=ChangeOperation.SET, value=date_profile[field]))
            if not nights and date_profile.get("nights") != state.dates.nights:
                changes.append(StateChange(field="nights", operation=ChangeOperation.SET, value=date_profile["nights"]))
            if re.search(r"\b(?:собак\w*|кошк\w*|питомц\w*|пес|пса|псом)\b", lower) and not re.search(r"\bбез\s+(?:собаки|кошки|питомц\w*|животных)\b", lower):
                changes.append(StateChange(field="has_pets", operation=ChangeOperation.SET, value=True))
            if re.search(r"\bбез\s+(?:собаки|кошки|питомц\w*|животных)\b", lower):
                changes.append(StateChange(field="has_pets", operation=ChangeOperation.SET, value=False))
            if re.search(r"\b(?:только\s+прямой|прямой\s+рейс|без\s+пересадок)\b", lower):
                changes.append(StateChange(field="prefer_direct_flight", operation=ChangeOperation.SET, value=True))
            elif re.search(r"\b(?:с\s+пересадкой|стыковочный)\b", lower):
                changes.append(StateChange(field="prefer_direct_flight", operation=ChangeOperation.SET, value=False))
            flight_time = None
            if re.search(r"днем|днём|в\s+обед|дневн\w*\s+вылет", lower):
                flight_time = "day"
            elif re.search(r"попозже|не\s+в\s+[67]\b|не\s+рано|после\s+(?:10|12)|поздний\s+вылет", lower):
                flight_time = "later"
            elif re.search(r"вечером|вечерн\w*\s+вылет|ночью", lower):
                flight_time = "evening"
            elif re.search(r"пораньше|утром|утренн\w*\s+вылет|ранний\s+вылет", lower):
                flight_time = "morning"
            if flight_time:
                changes.append(StateChange(field="flight_time_pref", operation=ChangeOperation.SET, value=flight_time))
            budget_profile = deepcopy(state.apply_to_profile({}))
            if pending_budget:
                budget_profile["awaiting_budget"] = True
            budget_question = update_trip_budget(message, budget_profile, context.get("cart") or {})
            if budget_question:
                if not changes:
                    changes.append(StateChange(field="replan_action", operation=ChangeOperation.SET, value="modify"))
                return DialogueDecision(intent=DialogueIntent.MODIFY, changes=changes, clarification=budget_question,
                                        search=SearchCriteriaDecision(missing_fields=["budget_rub"]))
            if budget_profile.get("budget_rub") != state.budget_rub:
                changes.append(StateChange(field="budget_rub", operation=ChangeOperation.SET, value=budget_profile["budget_rub"]))
            search = None
            if wants_cheaper(message):
                search = SearchCriteriaDecision(scope="current", price_preference="cheaper",
                                                flexible_dates=bool(re.search(r"в\s+любое\s+время|в\s+любой\s+день|когда\s+угодно", lower)))
            elif rejected:
                search = SearchCriteriaDecision(scope="other_destinations")
            elif re.search(r"другой\s+отель|поменяй\s+отель|смени\s+отель|другой\s+вариант\s+проживания|другие\s+варианты", lower):
                search = SearchCriteriaDecision(scope="hotels")
                changes.append(StateChange(field="replan_action", operation=ChangeOperation.SET, value="change_hotel"))
            if not changes:
                changes.append(StateChange(field="replan_action", operation=ChangeOperation.SET, value="modify"))
            return DialogueDecision(intent=DialogueIntent.PLAN if multi_city_request else DialogueIntent.MODIFY,
                                    changes=changes, search=search)

        changes = [StateChange(field="interests", operation=ChangeOperation.SET, value=interests)] if interests else []
        if direct_destination:
            changes.append(StateChange(field="destination_id", operation=ChangeOperation.SET, value=direct_destination))
        budget_profile = deepcopy(state.apply_to_profile({}))
        update_trip_budget(message, budget_profile, context.get("cart") or {})
        if budget_profile.get("budget_rub") != state.budget_rub:
            changes.append(StateChange(field="budget_rub", operation=ChangeOperation.SET, value=budget_profile["budget_rub"]))
        vague = re.search(r"хоч\w*|посовету|помоги|куда|отдохн|отпуск|путешеств|море|горы|солнц", lower)
        specific_plan = bool(direct_destination or multi_city_request)
        if vague and not specific_plan and state.preferences.is_multi_city:
            changes.append(StateChange(field="is_multi_city", operation=ChangeOperation.SET, value=False))
        missing = self._mock_missing(state) if vague and not specific_plan else []
        return DialogueDecision(
            intent=DialogueIntent.PLAN,
            changes=changes,
            search=SearchCriteriaDecision(missing_fields=missing),
            resume_pending=bool(state.pending_action),
        )

    def _mock_interests(self, message: str) -> list[str]:
        interests = []
        if re.search(r"море|морю|моря|морск|пляж|солнц|побереж|купат|загар", message):
            interests.append("beach")
        if re.search(r"горы|горн|альп|лыж|хайкинг|треккинг", message):
            interests.append("mountains")
        if re.search(r"истор|экскурси|достопримечательн|музе|культура", message):
            interests.append("history_culture")
        if re.search(r"спа|spa|релакс|массаж|велнес", message):
            interests.append("spa_wellness")
        if re.search(r"дешев|эконом|подешев|недорог|бюджетн", message):
            interests.append("budget_friendly")
        if re.search(r"люкс|премиум|вилл|5\s*звезд|пять\s*звезд|deluxe|vip|роскошн|раскошел|лучший\s+отель", message):
            interests.append("luxury")
        return interests

    def _mock_missing(self, state: TripState) -> list[str]:
        missing = []
        if not (state.destination_id or state.preferences.interests):
            missing.append("direction")
        if not state.dates.checkin_date:
            missing.append("checkin_date")
        if not state.dates.nights:
            missing.append("nights")
        if not state.passengers.adults_count:
            missing.append("adults_count")
        return missing

    def _safe_error(self, exc: Exception) -> str:
        message = str(exc)
        key = getattr(settings, "OPENAI_API_KEY", "")
        if key:
            message = message.replace(key, "<redacted>")
        return message[:500]


def decision_context(session: Any, state: TripState, cart: dict) -> dict[str, Any]:
    return {
        "selected_offers": deepcopy(state.selected_offers),
        "alternative_offers": deepcopy(cart.get("compared_hotels") or []),
        "conversation_history": deepcopy(getattr(session, "conversation_history", []) or []),
        "cart": deepcopy(cart),
    }
