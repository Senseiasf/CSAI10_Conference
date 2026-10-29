import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config import settings
from app.database import (
    check_postgres_health, check_neo4j_health, close_connections,
    async_session_maker
)
from app.models import UserSessionModel
from app.seed import run_seed
from app.services.graph_service import GraphService
from app.services.agent_service import AgentService
from app.services.calendar_service import CalendarService
from app.services.preference_memory import remembered_preferences, preference_summary
from app.services.compliance_service import ComplianceService
from app.services.ltr_service import LTRService
from app.services.llm_service import LLMService
from app.bot.telegram_bot import start_telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Ostrovok AI Travel Agent backend...")
    
    # 1. Автоматическое сидирование и проверка баз данных
    try:
        logger.info("Executing zero-touch startup seeding...")
        await run_seed()
    except Exception as e:
        logger.error(f"Startup seeding encountered an error (will continue in degraded mode): {e}")

    # 2. Фоновый запуск Telegram-бота
    bot_task = asyncio.create_task(start_telegram_bot())

    yield

    # Корректное завершение
    logger.info("Shutting down backend and cancelling background tasks...")
    bot_task.cancel()
    await close_connections()


app = FastAPI(
    title="Ostrovok AI Travel Agent",
    description="Intelligent end-to-end trip planning agent with Neo4j logistics graph and pgvector inspiration search",
    version="1.0.0",
    lifespan=lifespan
)

# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Главная страница: интерактивный дашборд и тест-интерфейс на порту 8080."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "bot_username": settings.TELEGRAM_BOT_USERNAME,
            "llm_mode": "Mock Agent" if settings.is_mock_llm else f"DeepSeek Flash ({settings.OPENAI_MODEL})"
        }
    )


@app.get("/health")
async def health_check():
    """
    Обязательный эндпоинт проверки здоровья для Docker Compose и жюри (HTTP 200).
    """
    pg_ok = await check_postgres_health()
    neo_ok = await check_neo4j_health()
    graph_stats = await GraphService.get_graph_stats() if neo_ok else {}

    all_ok = pg_ok and neo_ok
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "degraded",
            "postgres": pg_ok,
            "neo4j": neo_ok,
            "llm_mode": "mock" if settings.is_mock_llm else settings.OPENAI_MODEL,
            "llm": LLMService.get_diagnostics(),
            "graph_stats": graph_stats,
            "ltr_ranker": LTRService.get_status()
        }
    )


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request, response: Response):
    """Эндпоинт диалога с AI-агентом для веб-интерфейса."""
    user_id = req.user_id or web_user_id(request, response)
    result = await AgentService.process_message(
        user_id=user_id,
        message=req.message
    )
    return result


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, request: Request, response: Response):
    """
    Server-Sent Events (SSE) эндпоинт для real-time reasoning и потоковой генерации ответа.
    События:
    - status: ход рассуждений и этапы графовой валидации
    - token: потоковые чанки текста для эффекта живой печати
    - result: финальный JSON с карточкой тура, чипами дат, комплаенсом и корзиной
    """
    user_id = req.user_id or web_user_id(request, response)

    # Выполняем расчёт тура в основном цикле запроса (для стабильности event loop и asyncpg)
    result = await AgentService.process_message(
        user_id=user_id,
        message=req.message
    )

    async def sse_event_stream():
        try:
            if not result.get("booking_action"):
                # 1. Этап: Анализ запроса
                yield f"event: status\ndata: {json.dumps({'stage': 'nlu', 'message': '🧠 Анализирую намерения, состав туристов и бюджет...'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.04)

                # 2. Этап: Графовая логистика
                yield f"event: status\ndata: {json.dumps({'stage': 'graph', 'message': '🕸️ Проверяю стыковки рейсов и трансферы в графе Neo4j...'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.04)

                if result.get("ltr_ranking"):
                    yield f"event: status\ndata: {json.dumps({'stage': 'ltr_ranking', 'message': 'Сравниваю подходящие варианты с учётом ваших предпочтений...'}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.04)

                # 3. Этап: Комплаенс и безопасность
                yield f"event: status\ndata: {json.dumps({'stage': 'compliance', 'message': '🛂 Сверяю визовый режим, паспортный комплаенс и правила въезда...'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.04)

            # 4. Потоковая передача текста частями (chunks)
            full_text = result.get("text", "")
            lines = full_text.split("\n")
            for idx, line in enumerate(lines):
                chunk = line + ("\n" if idx < len(lines) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.008)

            # 5. Финальное событие с полным объектом
            yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error in SSE chat stream: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            **({"Set-Cookie": response.headers["set-cookie"]} if "set-cookie" in response.headers else {})
        }
    )


def web_user_id(request: Request, response: Response) -> str:
    try:
        session_id = str(UUID(request.cookies.get("ostrovok_session", "")))
    except ValueError:
        session_id = str(uuid4())
    response.set_cookie("ostrovok_session", session_id, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https", max_age=60 * 60 * 24 * 30)
    return f"web:{session_id}"


@app.get("/api/profile")
async def api_profile(request: Request, response: Response):
    user = await AgentService.get_or_create_session(web_user_id(request, response))
    preferences = remembered_preferences(user.profile or {})
    return {"preferences": preferences, "summary": preference_summary(preferences)}


class ResetSessionRequest(BaseModel):
    forget_preferences: bool = False


@app.post("/api/session/reset")
async def api_session_reset(request: Request, response: Response, options: Optional[ResetSessionRequest] = None):
    preferences = await AgentService.reset_trip(
        web_user_id(request, response),
        forget_preferences=bool(options and options.forget_preferences),
    )
    return {"status": "ok", "preferences": preferences, "summary": preference_summary(preferences),
            "message": "Новая поездка: предпочтения сохранены." if preferences else "Начинаем новую поездку."}


class ExportCalendarRequest(BaseModel):
    cart: Optional[Dict[str, Any]] = None
    booking_id: Optional[str] = None


@app.get("/api/cart/export-calendar")
async def api_export_calendar_get(request: Request, user_id: Optional[str] = None):
    """Скачать файл календаря (.ics) для текущей корзины пользователя."""
    if not user_id:
        cookie = request.cookies.get("ostrovok_session", "")
        user_id = f"web:{cookie}" if cookie else "web:anonymous"

    cart = {}
    try:
        async with async_session_maker() as session:
            sess = await session.get(UserSessionModel, user_id)
            if sess and sess.current_cart:
                cart = sess.current_cart
    except Exception as e:
        logger.warning(f"Could not load cart for calendar export: {e}")

    if not cart or not cart.get("destination"):
        cart = {
            "booking_id": "OST-DEMO",
            "destination": "Сочи (Красная Поляна)",
            "hotel": "Radisson Hotel Rosa Khutor",
            "nights": 7,
            "flight": "Аэрофлот SU-1124 (SVO ⇄ AER)",
            "transfer": "Индивидуальный трансфер"
        }

    ics_content = CalendarService.generate_ics(cart)
    booking_id = cart.get("booking_id", "trip")
    filename = f"ostrovok_{booking_id}.ics"

    return Response(
        content=ics_content,
        media_type="text/calendar; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.post("/api/cart/export-calendar")
async def api_export_calendar_post(req: ExportCalendarRequest, request: Request):
    """Скачать файл календаря (.ics) с передачей корзины из клиента."""
    cart = req.cart
    if not cart:
        cookie = request.cookies.get("ostrovok_session", "")
        user_id = f"web:{cookie}" if cookie else "web:anonymous"
        try:
            async with async_session_maker() as session:
                sess = await session.get(UserSessionModel, user_id)
                if sess and sess.current_cart:
                    cart = sess.current_cart
        except Exception:
            pass

    if not cart:
        cart = {
            "booking_id": req.booking_id or "OST-TRIP",
            "destination": "Тур с Островком",
            "hotel": "Рекомендуемый отель",
            "nights": 7
        }

    ics_content = CalendarService.generate_ics(cart)
    booking_id = cart.get("booking_id", req.booking_id or "trip")
    filename = f"ostrovok_{booking_id}.ics"

    return Response(
        content=ics_content,
        media_type="text/calendar; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.get("/api/compliance/{dest_id}")
async def api_compliance(dest_id: str):
    """Проверка правил въезда и визовых требований по направлению."""
    return ComplianceService.get_compliance_info(dest_id)


class LTRFeedbackRequest(BaseModel):
    archetype: Literal["pet", "family", "luxury", "budget", "solo"] = "solo"
    chosen_tour: Dict[str, Any]
    rejected_tour: Dict[str, Any]
    margin: int = Field(default=1, ge=1, le=10, strict=True)


@app.get("/api/ltr/status")
async def api_ltr_status():
    """Реальный статус модели и резервного скоринга."""
    return LTRService.get_status()


@app.post("/api/ltr/feedback")
async def api_ltr_feedback(req: LTRFeedbackRequest):
    """Сохраняет парный выбор в ограниченный буфер текущего процесса."""
    return LTRService.record_online_feedback(req.archetype, req.chosen_tour, req.rejected_tour, req.margin)


@app.get("/api/stats")
async def api_stats():
    """Статистика по графу и базе знаний."""
    stats = await GraphService.get_graph_stats()
    return stats


@app.get("/api/voucher/{booking_id}", response_class=HTMLResponse)
async def get_voucher(request: Request, booking_id: str):
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserSessionModel).where(
                UserSessionModel.current_cart["booking_confirmation"]["booking_id"].astext == booking_id
            )
        )
        user_session = result.scalars().first()

    confirmation = (user_session.current_cart or {}).get("booking_confirmation") if user_session else None
    if not confirmation:
        raise HTTPException(status_code=404, detail="Ваучер не найден")

    return templates.TemplateResponse(
        request=request,
        name="voucher.html",
        context={
            "confirmation": confirmation,
            "total_paid": f"{confirmation['total_paid_rub']:,} ₽",
        },
    )
