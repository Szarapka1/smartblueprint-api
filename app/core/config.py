# app/core/config.py

from functools import lru_cache
from typing import List
import os

class AppSettings:
    """
    Application settings loaded from Azure environment variables.
    """

    def __init__(self):
        # General
        self.PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Smart Blueprint Chat API")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

        # Server
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", 8000))

        # CORS
        cors_origins = os.getenv("CORS_ORIGINS", '["*"]')
        self.CORS_ORIGINS: List[str] = eval(cors_origins)

        # Azure Blob Storage
        self.AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.AZURE_CONTAINER_NAME: str = os.getenv("AZURE_CONTAINER_NAME", "blueprints")
        self.AZURE_CACHE_CONTAINER_NAME: str = os.getenv("AZURE_CACHE_CONTAINER_NAME", "blueprints-cache")

        # OpenAI
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", 4000))
        self.OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", 0.1))

        # PDF Processing
        self.PDF_PREVIEW_RESOLUTION: int = int(os.getenv("PDF_PREVIEW_RESOLUTION", 150))
        self.PDF_HIGH_RESOLUTION: int = int(os.getenv("PDF_HIGH_RESOLUTION", 200))
        self.PDF_IMAGE_DPI: int = int(os.getenv("PDF_IMAGE_DPI", 200))
        self.PDF_MAX_PAGES: int = int(os.getenv("PDF_MAX_PAGES", 100))
        self.PROCESSING_BATCH_SIZE: int = int(os.getenv("PROCESSING_BATCH_SIZE", 5))

        # Cache and Limits
        self.MAX_MEMORY_CACHE_SIZE: int = int(os.getenv("MAX_MEMORY_CACHE_SIZE", 100))
        self.ADMIN_SECRET_TOKEN: str = os.getenv("ADMIN_SECRET_TOKEN", "blueprintreader789")
        self.MAX_ACTIVITY_LOGS: int = int(os.getenv("MAX_ACTIVITY_LOGS", 1000))
        self.MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 60))
        self.MAX_CHAT_LOGS: int = int(os.getenv("MAX_CHAT_LOGS", 1000))
        self.MAX_USER_CHAT_HISTORY: int = int(os.getenv("MAX_USER_CHAT_HISTORY", 100))

@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
