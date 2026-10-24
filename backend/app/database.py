import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Async Engine
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Neo4j Driver
neo4j_driver: Optional[AsyncDriver] = None


def get_neo4j_driver() -> AsyncDriver:
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    return neo4j_driver


async def close_connections():
    global neo4j_driver
    if neo4j_driver:
        await neo4j_driver.close()
        neo4j_driver = None
    await engine.dispose()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def check_postgres_health() -> bool:
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.warning(f"PostgreSQL healthcheck failed: {e}")
        return False


async def check_neo4j_health() -> bool:
    try:
        driver = get_neo4j_driver()
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS num")
            record = await result.single()
            return record and record["num"] == 1
    except Exception as e:
        logger.warning(f"Neo4j healthcheck failed: {e}")
        return False
