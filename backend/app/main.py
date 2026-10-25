import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
from app.database import check_postgres_health, check_neo4j_health, close_connections
from app.seed import run_seed
from app.services.graph_service import GraphService
from app.services.agent_service import AgentService
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
            "llm_mode": "Mock Agent" if settings.is_mock_llm else "OpenAI Compatible"
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
            "llm_mode": "mock" if settings.is_mock_llm else "openai",
            "graph_stats": graph_stats
        }
    )


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request, response: Response):
    """Эндпоинт диалога с AI-агентом для веб-интерфейса."""
    user_id = req.user_id
    if not user_id:
        cookie = request.cookies.get("ostrovok_session", "")
        try:
            session_id = str(UUID(cookie))
        except ValueError:
            session_id = str(uuid4())
        user_id = f"web:{session_id}"
        response.set_cookie(
            "ostrovok_session", session_id, httponly=True, samesite="lax",
            secure=request.url.scheme == "https", max_age=60 * 60 * 24 * 30,
        )
    result = await AgentService.process_message(
        user_id=user_id,
        message=req.message
    )
    return result


@app.get("/api/stats")
async def api_stats():
    """Статистика по графу и базе знаний."""
    stats = await GraphService.get_graph_stats()
    return stats
