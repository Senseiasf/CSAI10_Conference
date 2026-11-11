import asyncio
import json
import logging
from pathlib import Path
from sqlalchemy import text, select
from app.database import engine, async_session_maker, check_postgres_health, check_neo4j_health
from app.models import Base, DestinationModel, HotelModel, RiskAlertModel
from app.services.vector_service import generate_text_embedding, EMBEDDING_VERSION
from app.services.graph_service import GraphService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("seed")


async def wait_for_services(max_retries: int = 30, delay: float = 2.0):
    logger.info("Waiting for PostgreSQL and Neo4j to be healthy...")
    for i in range(max_retries):
        pg_ok = await check_postgres_health()
        neo_ok = await check_neo4j_health()
        if pg_ok and neo_ok:
            logger.info("PostgreSQL and Neo4j are healthy and ready!")
            return True
        logger.info(f"Retrying connections ({i+1}/{max_retries})... pg: {pg_ok}, neo4j: {neo_ok}")
        await asyncio.sleep(delay)
    raise RuntimeError("Timed out waiting for database services.")


async def seed_postgres(data: dict):
    logger.info("Initializing PostgreSQL schema and pgvector extension...")
    async with engine.begin() as conn:
        # Enable pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("UPDATE destinations SET embedding = NULL;"))
        await conn.execute(text("ALTER TABLE destinations ALTER COLUMN embedding TYPE vector(384);"))
        await conn.execute(text("ALTER TABLE hotels ADD COLUMN IF NOT EXISTS pet_surcharge_per_night_rub INTEGER DEFAULT 0;"))
        await conn.execute(text("ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS trip_state JSONB;"))
        await conn.execute(text("ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS trip_quote JSONB;"))

    async with async_session_maker() as session:
        for dest in data["destinations"]:
            # Проверяем наличие
            existing = await session.execute(
                select(DestinationModel).where(DestinationModel.id == dest["id"])
            )
            dest_obj = existing.scalar_one_or_none()

            # Честный 384-мерный вектор SentenceTransformer MiniLM из калиброванного описания
            embedding_text = dest["vibe_description"]
            vector = generate_text_embedding(embedding_text)

            price_info = dest.get("price_range_rub", {})
            min_price = price_info.get("min", 0)
            max_price = price_info.get("max", 0)
            avg_price = price_info.get("avg_check_per_person_tour_7d", 0)

            if not dest_obj:
                dest_obj = DestinationModel(
                    id=dest["id"],
                    name=dest["name"],
                    country=dest["country"],
                    price_min=min_price,
                    price_max=max_price,
                    avg_check=avg_price,
                    vibe_description=dest["vibe_description"],
                    climate_summary=dest["climate_summary"],
                    embedding=vector,
                    metadata_json={
                        "cities": dest.get("cities", []),
                        "airports": dest.get("airports", [])
                    }
                )
                session.add(dest_obj)
            else:
                dest_obj.price_min = min_price
                dest_obj.price_max = max_price
                dest_obj.avg_check = avg_price
                dest_obj.vibe_description = dest["vibe_description"]
                dest_obj.climate_summary = dest["climate_summary"]
                dest_obj.embedding = vector
                dest_obj.metadata_json = {
                    **(dest_obj.metadata_json or {}),
                    "cities": dest.get("cities", []),
                    "airports": dest.get("airports", []),
                    "embedding_version": EMBEDDING_VERSION,
                }

            # Отели
            for h in dest.get("hotels", []):
                hotel_res = await session.execute(select(HotelModel).where(HotelModel.id == h["id"]))
                hotel_obj = hotel_res.scalar_one_or_none()
                
                features_data = list(h.get("features", []))
                if "coordinates" in h:
                    features_data.append({"coordinates": h["coordinates"]})
                if "rating_score" in h:
                    features_data.append({"rating_score": h["rating_score"]})
                if "reviews_count" in h:
                    features_data.append({"reviews_count": h["reviews_count"]})
                if "website_url" in h:
                    features_data.append({"website_url": h["website_url"]})
                if "real_description" in h:
                    features_data.append({"real_description": h["real_description"]})

                if not hotel_obj:
                    hotel_obj = HotelModel(
                        id=h["id"],
                        destination_id=dest["id"],
                        name=h["name"],
                        city=h["city"],
                        stars=h.get("stars", 4),
                        category=h.get("category", ""),
                        pet_friendly=h.get("pet_friendly", False),
                        cancellation_policy=h.get("cancellation_policy", ""),
                        cancellation_type=h.get("cancellation_type", "flexible"),
                        meals=h.get("meals", ""),
                        checkin_time=h.get("checkin_time", "15:00"),
                        checkout_time=h.get("checkout_time", "12:00"),
                        price_per_night_rub=h.get("price_per_night_rub", 10000),
                        pet_surcharge_per_night_rub=h.get("pet_surcharge_per_night_rub", 1500 if h.get("pet_friendly") else 0),
                        distance_to_sea_meters=h.get("distance_to_sea_meters", 0),
                        family_friendly=h.get("family_friendly", True),
                        features=features_data
                    )
                    session.add(hotel_obj)
                else:
                    hotel_obj.name = h["name"]
                    hotel_obj.city = h["city"]
                    hotel_obj.stars = h.get("stars", 4)
                    hotel_obj.category = h.get("category", "")
                    hotel_obj.pet_friendly = h.get("pet_friendly", False)
                    hotel_obj.cancellation_policy = h.get("cancellation_policy", "")
                    hotel_obj.cancellation_type = h.get("cancellation_type", "flexible")
                    hotel_obj.meals = h.get("meals", "")
                    hotel_obj.checkin_time = h.get("checkin_time", "15:00")
                    hotel_obj.checkout_time = h.get("checkout_time", "12:00")
                    hotel_obj.price_per_night_rub = h.get("price_per_night_rub", 10000)
                    hotel_obj.pet_surcharge_per_night_rub = h.get("pet_surcharge_per_night_rub", 1500 if h.get("pet_friendly") else 0)
                    hotel_obj.distance_to_sea_meters = h.get("distance_to_sea_meters", 0)
                    hotel_obj.family_friendly = h.get("family_friendly", True)
                    hotel_obj.features = features_data

            # Внешние обстоятельства и риски
            for r in dest.get("risks_and_alerts", []):
                risk_res = await session.execute(
                    select(RiskAlertModel).where(
                        RiskAlertModel.destination_id == dest["id"],
                        RiskAlertModel.title == r["title"]
                    )
                )
                if not risk_res.scalar_one_or_none():
                    risk_obj = RiskAlertModel(
                        destination_id=dest["id"],
                        risk_type=r.get("type", "general"),
                        severity=r.get("severity", "medium"),
                        title=r["title"],
                        description=r["description"]
                    )
                    session.add(risk_obj)

        await session.commit()
    logger.info("PostgreSQL database successfully seeded with destinations, hotels, and risk alerts.")


async def run_seed():
    await wait_for_services()
    json_path = Path(__file__).parent / "data" / "destinations.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Сидируем Postgres
    await seed_postgres(data)

    # 2. Сидируем Neo4j (аэропорты вылета, рейсы, перелеты, отели, трансферы)
    logger.info("Seeding Neo4j logistics graph (flights and transfers)...")
    await GraphService.seed_graph(data)

    stats = await GraphService.get_graph_stats()
    logger.info(f"Graph stats: {stats}")
    logger.info("Seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_seed())
