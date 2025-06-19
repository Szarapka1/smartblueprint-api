# app/core/config.py

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class AppSettings(BaseSettings):
    """
    Global application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- General Info ---
    PROJECT_NAME: str = "Smart Blueprint Chat API"
    DEBUG: bool = True

    # --- Server Settings ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- CORS Settings (Allow any origin) ---
    CORS_ORIGINS: List[str] = ["*"]

    # --- Azure Blob ---
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_CONTAINER_NAME: str = "blueprints"
    AZURE_CACHE_CONTAINER_NAME: str = "blueprints-cache"

    # --- OpenAI API ---
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 40000
    OPENAI_TEMPERATURE: float = 0.0

    # --- PDF ---
    PDF_PREVIEW_RESOLUTION: int = 150
    PDF_HIGH_RESOLUTION: int = 300

    # --- Cache ---
    MAX_MEMORY_CACHE_SIZE: int = 50

@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
