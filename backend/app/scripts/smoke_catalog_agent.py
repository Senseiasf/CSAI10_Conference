"""Manual dialogue probe: real catalogue/embeddings/CatBoost/Neo4j, memory sessions.

Run from backend with NEO4J_URI pointing to a seeded test instance:
python -m app.scripts.smoke_catalog_agent path/to/report.md
PostgreSQL, external LLM and weather are deliberately not exercised here.
"""
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from app.database import close_connections
from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService
from app.services.trip_state import requested_destination
from app.services.vector_service import generate_text_embedding


async def run():
    data = json.loads((Path(__file__).parents[1] / "data/destinations.json").read_text(encoding="utf-8"))
    destinations = {d["id"]: d for d in data["destinations"]}
    vectors = {d["id"]: generate_text_embedding(f"{d['name']} {d['country']} {d['vibe_description']} {d['climate_summary']}")
               for d in destinations.values()}
    user = SimpleNamespace(profile={}, current_cart={}, current_state="INSPIRATION", conversation_history=[],
                           trip_state=None, trip_quote=None)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args):
            return user

        async def commit(self):
            pass

    async def save(uid, profile, cart, message, response, **kwargs):
        user.profile, user.current_cart = deepcopy(profile), deepcopy(cart)

    async def search(query, limit=2, abroad_only=False, domestic_only=False):
        exact = requested_destination(query)
        vector = generate_text_embedding(query)
        rows = [{**d, "score": float(np.dot(vector, vectors[d["id"]]))} for d in destinations.values()
                if (not abroad_only or d["country"] != "Россия") and (not domestic_only or d["country"] == "Россия")]
        if exact:
            return [d for d in rows if d["id"] == exact]
        return sorted(rows, key=lambda d: -d["score"])[:limit]

    async def hotels(destination_id, city=None, pet_friendly=False, family_friendly=False,
                     exclude_hotel_ids=None, limit=30, **kwargs):
        rows = []
        for h in destinations[destination_id]["hotels"]:
            if city and city.casefold() not in h["city"].casefold():
                continue
            if h["id"] in (exclude_hotel_ids or []) or pet_friendly and not h.get("pet_friendly"):
                continue
            if family_friendly and not h.get("family_friendly"):
                continue
            rows.append({**h, "stars": f"{h['stars']}★", "cancellation": h.get("cancellation_policy", "Не указана"),
                         "distance_to_sea": f"{h.get('distance_to_sea_meters', 0)} м",
                         "price_per_night_formatted": f"{h['price_per_night_rub']:,} ₽"})
        return {"compared_hotels": rows[:limit] if limit else rows}

    messages = ["В Сочи с 20 по 25 ноября 2026, 2 взрослых, бюджет 250000 рублей",
                "Убери трансфер", "Сравни цены на соседние даты", "Сделай подешевле",
                "Забронируй", "Тур по городам в Сочи на 6 ночей, бюджет 500000 рублей"]
    lines = ["# Проверка диалога по каталогу", "", "Живые Neo4j, multilingual MiniLM и CatBoost. "
             "Сессия и выборка каталога в памяти; PostgreSQL, внешний LLM и погода заменены тестовыми адаптерами.", ""]
    with (patch.object(AgentService, "get_or_create_session", new=AsyncMock(return_value=user)),
          patch.object(AgentService, "_save_session", side_effect=save),
          patch("app.services.agent_service.async_session_maker", return_value=Session()),
          patch.object(AgentService, "search_destinations_vector", side_effect=search),
          patch.object(MCPTools, "compare_hotels", side_effect=hotels),
          patch.object(MCPTools, "get_destination_risks", new=AsyncMock(return_value={})),
          patch.object(MCPTools, "get_live_weather", new=AsyncMock(return_value={}))):
        for message in messages:
            result = await AgentService.process_message("test-catalog-smoke", message)
            lines += [f"## Пользователь: {message}", "", f"Состояние: {result.get('state', 'OFFER')}", "", result["text"], ""]
            print(message, "=>", result.get("state", "OFFER"), result.get("cart", {}).get("hotel_id"))
    Path(sys.argv[1]).write_text("\n".join(lines), encoding="utf-8")
    await close_connections()


if __name__ == "__main__":
    asyncio.run(run())
