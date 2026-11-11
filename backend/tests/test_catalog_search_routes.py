import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.mcp.tools import MCPTools
from app.services.agent_service import AgentService, MULTI_CITY_CONFIG
from app.services.catalog_validation import audit_catalog, validate_catalog
from app.services.graph_service import GraphService
from app.services.multi_city_routes import select_route
from app.services.vector_service import generate_text_embedding


def catalog():
    return json.loads((Path(__file__).parents[1] / "app/data/destinations.json").read_text(encoding="utf-8"))


def test_catalog_has_no_dangling_edges_or_uncovered_hotels():
    report = validate_catalog(catalog())
    assert report["hotels"] == 1327
    assert report["flights"] == 276
    assert report["transfers"] == 1360
    assert report["hotels_without_airport_transfer"] == []


def test_bad_transfer_reference_prevents_silent_seed_loss():
    data = catalog()
    data["destinations"][0]["transfers"][0]["to"] = "missing-hotel"
    assert audit_catalog(data)["errors"]
    with pytest.raises(ValueError, match="Dangling transfer"):
        validate_catalog(data)


@pytest.mark.skipif(not os.environ.get("CSAI_EMBEDDING_TEST"), reason="Opt-in: downloads the pinned embedding model")
def test_real_multilingual_embeddings_preserve_meaning_and_punctuation():
    sea, inflected, beach, irrelevant, punctuation = map(generate_text_embedding,
        ("море", "морю", "пляж", "бухгалтерия", "море,"))
    assert len(sea) == 128
    assert np.linalg.norm(sea) == pytest.approx(1)
    assert np.dot(sea, inflected) > .9
    assert np.dot(sea, beach) > np.dot(sea, irrelevant) + .3
    assert sea == punctuation


@pytest.mark.asyncio
async def test_empty_and_conflicting_search_never_query_databases():
    with patch("app.services.agent_service.async_session_maker") as db:
        assert await AgentService.search_destinations_vector("!!!") == []
        assert await AgentService.search_destinations_vector("пляж", domestic_only=True, abroad_only=True) == []
        db.assert_not_called()


def h(identifier, price, score):
    return {"id": identifier, "price_per_night_rub": price, "ltr_score": score}


@pytest.mark.asyncio
async def test_missing_intercity_flight_is_not_replaced_by_template():
    with patch.object(GraphService, "find_intercity_logistics", new=AsyncMock(return_value=None)):
        assert await select_route(MULTI_CITY_CONFIG["turkey"], [h("a", 1000, 2)], [h("b", 1000, 1)],
                                  {}, 2, 2, None, "ltr") is None


@pytest.mark.asyncio
async def test_pair_search_skips_unreachable_and_over_budget_hotels():
    async def logistics(**kw):
        if kw.get("transfer_arrival_time"):
            if kw["hotel_id"] == "unreachable":
                return {"compatible": False}
            return {"compatible": True, "transfer": {"name": "Real road", "duration_min": 70, "price_rub": 1700}}
        return {"compatible": True, "flight": {"price_rub_roundtrip": 10000}, "transfer": {"price_rub": 800}}

    with patch.object(MCPTools, "check_logistics_compatibility", side_effect=logistics):
        route = await select_route(MULTI_CITY_CONFIG["sochi"], [h("a", 1000, 1)],
            [h("unreachable", 1000, 100), h("expensive", 50000, 10), h("b", 1000, 2)],
            {"adults_count": 1}, 2, 2, 40000, "ltr")
    assert route["hotel_2"]["id"] == "b"
    assert route["intercity_cost"] == 1700
    assert route["transfer_2"]["price_rub"] == 0  # ground leg charged once
    assert route["intercity"]["arrival"] == "13:10"


@pytest.mark.asyncio
async def test_without_airport_transfer_keeps_required_ground_intercity_leg():
    async def logistics(**kw):
        if kw.get("transfer_arrival_time"):
            return {"compatible": False}
        return {"compatible": True, "flight": {"price_rub_roundtrip": 10000}, "transfer": {"price_rub": 0}}

    with patch.object(MCPTools, "check_logistics_compatibility", side_effect=logistics):
        assert await select_route(MULTI_CITY_CONFIG["sochi"], [h("a", 1000, 1)], [h("b", 1000, 2)],
                                  {"without_transfer": True}, 2, 2, None, "ltr") is None


@pytest.mark.asyncio
async def test_single_night_cannot_be_split_into_two_hotel_nights():
    cart = {"hotel_id": "keep"}
    result = await AgentService.assemble_multi_city_itinerary(
        "sochi", {}, {}, deepcopy(cart), 1, None, "test", "two cities", False, False, None)
    assert result["state"] == "NO_MATCHES"
    assert result["cart"] == cart


@pytest.mark.asyncio
async def test_checkout_does_not_mark_catalog_quote_paid():
    cart = {"hotel_id": "selected", "total_mandatory_rub": 10000}
    session = AsyncMock()
    session.get.return_value = SimpleNamespace(current_cart=cart)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    with patch("app.mcp.tools.async_session_maker", return_value=context):
        result = await MCPTools.checkout_cart("test")
    assert result["state"] == "BOOKING_UNAVAILABLE"
    assert cart == {"hotel_id": "selected", "total_mandatory_rub": 10000}
    session.commit.assert_not_awaited()
