from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_PORT: int = 8080
    APP_HOST: str = "0.0.0.0"
    
    # PostgreSQL
    POSTGRES_USER: str = "app"
    POSTGRES_PASSWORD: str = "app"
    POSTGRES_DB: str = "app"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Neo4j
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "secretpassword"
    
    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_BOT_USERNAME: str = "OstrovokTravelBot"
    
    # LLM (OpenAI-compatible / DeepSeek)
    LLM_MODE: Literal["mock", "live"] = "mock"
    DIALOGUE_HANDLER: Literal["legacy", "v2"] = "v2"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.deepseek.com"
    OPENAI_MODEL: str = "deepseek-flash"
    TRIP_TIMEZONE: str = "Europe/Moscow"

    @property
    def is_mock_llm(self) -> bool:
        return self.LLM_MODE == "mock"

    @property
    def llm_configuration_error(self) -> str | None:
        if self.is_mock_llm:
            return None
        if self.OPENAI_API_KEY.strip().lower() in ("", "mock", "stub", "dummy"):
            return "LLM_MODE=live требует действующий OPENAI_API_KEY"
        if not self.OPENAI_BASE_URL.strip() or not self.OPENAI_MODEL.strip():
            return "Для live-режима укажите OPENAI_BASE_URL и OPENAI_MODEL"
        return None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
