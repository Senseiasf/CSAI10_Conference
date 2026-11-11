import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.database import (
    check_postgres_health, check_neo4j_health, close_connections
)
from app.seed import run_seed
from app.services.graph_service import GraphService
from app.services.compliance_service import ComplianceService
from app.services.ltr_service import LTRService
from app.services.llm_service import LLMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up CSAI 2026 Neuro-Symbolic Benchmark backend...")

    # Automatic database seeding and health check
    try:
        logger.info("Executing zero-touch startup seeding...")
        await run_seed()
    except Exception as e:
        logger.error(f"Startup seeding encountered an error (will continue in degraded mode): {e}")

    yield

    logger.info("Shutting down backend...")
    await close_connections()


app = FastAPI(
    title="CSAI 2026 Neuro-Symbolic Travel Benchmark API",
    description="Scientific evaluation stand for Neuro-Symbolic Trip Planning (PostgreSQL + pgvector + Neo4j + CatBoost LTR)",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "title": "CSAI 2026 Neuro-Symbolic Travel Benchmark Stand",
        "description": "Hybrid Neuro-Symbolic Routing & Ranking Architecture for Complex Travel Planning",
        "health_check": "/health",
        "docs": "/docs",
        "ltr_status": "/api/ltr/status",
        "graph_stats": "/api/stats"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker Compose and evaluators."""
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


@app.get("/api/stats")
async def api_stats():
    """Knowledge graph and catalog statistics."""
    return await GraphService.get_graph_stats()


@app.get("/api/compliance/{dest_id}")
async def api_compliance(dest_id: str):
    """Visa compliance and entry requirements check by destination."""
    return ComplianceService.get_compliance_info(dest_id)


class LTRFeedbackRequest(BaseModel):
    archetype: str = "solo"
    chosen_tour: Dict[str, Any]
    rejected_tour: Dict[str, Any]
    margin: int = Field(default=1, ge=1, le=10)


@app.get("/api/ltr/status")
async def api_ltr_status():
    """CatBoost LTR model status and feature diagnostics."""
    return LTRService.get_status()


@app.post("/api/ltr/feedback")
async def api_ltr_feedback(req: LTRFeedbackRequest):
    """Records pairwise choice feedback."""
    return LTRService.record_online_feedback(req.archetype, req.chosen_tour, req.rejected_tour, req.margin)
