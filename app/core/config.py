# app/core/config.py - OPTIMIZED FOR MULTI-USER COLLABORATION
from functools import lru_cache
from typing import List
import os

class AppSettings:
    """
    Application settings optimized for multi-user collaborative blueprint analysis.
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
        self.OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
        
        # PDF Processing - OPTIMIZED VALUES
        self.PDF_PREVIEW_RESOLUTION: int = int(os.getenv("PDF_PREVIEW_RESOLUTION", 72))  # For thumbnails
        self.PDF_HIGH_RESOLUTION: int = int(os.getenv("PDF_HIGH_RESOLUTION", 150))  # For storage
        self.PDF_IMAGE_DPI: int = int(os.getenv("PDF_IMAGE_DPI", 150))  # Reduced from 200
        self.PDF_AI_DPI: int = int(os.getenv("PDF_AI_DPI", 100))  # New - for AI processing
        self.PDF_THUMBNAIL_DPI: int = int(os.getenv("PDF_THUMBNAIL_DPI", 72))  # New - for thumbnails
        self.PDF_MAX_PAGES: int = int(os.getenv("PDF_MAX_PAGES", 100))
        self.PROCESSING_BATCH_SIZE: int = int(os.getenv("PROCESSING_BATCH_SIZE", 5))
        self.PDF_PNG_COMPRESSION: int = int(os.getenv("PDF_PNG_COMPRESSION", 6))  # New - PNG compression level
        self.PDF_JPEG_QUALITY: int = int(os.getenv("PDF_JPEG_QUALITY", 85))  # New - JPEG quality
        
        # Cache and Limits
        self.MAX_MEMORY_CACHE_SIZE: int = int(os.getenv("MAX_MEMORY_CACHE_SIZE", 100))
        self.ADMIN_SECRET_TOKEN: str = os.getenv("ADMIN_SECRET_TOKEN", "blueprintreader789")
        self.MAX_ACTIVITY_LOGS: int = int(os.getenv("MAX_ACTIVITY_LOGS", 1000))
        self.MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 60))
        self.MAX_CHAT_LOGS: int = int(os.getenv("MAX_CHAT_LOGS", 1000))
        self.MAX_USER_CHAT_HISTORY: int = int(os.getenv("MAX_USER_CHAT_HISTORY", 100))
        
        # --- NEW SETTINGS FOR NOTES AND HIGHLIGHTING ---
        
        # Note System Settings
        self.MAX_NOTES_PER_DOCUMENT: int = int(os.getenv("MAX_NOTES_PER_DOCUMENT", 500))
        self.MAX_NOTE_LENGTH: int = int(os.getenv("MAX_NOTE_LENGTH", 10000))
        self.MAX_TOTAL_NOTE_CHARS: int = int(os.getenv("MAX_TOTAL_NOTE_CHARS", 500000))
        self.NOTE_TYPES: List[str] = os.getenv("NOTE_TYPES", "general,question,issue,suggestion,review,coordination,warning").split(",")
        
        # Session Service Settings
        self.MAX_SESSIONS_IN_MEMORY: int = int(os.getenv("MAX_SESSIONS_IN_MEMORY", 100))
        self.SESSION_CLEANUP_HOURS: int = int(os.getenv("SESSION_CLEANUP_HOURS", 24))
        self.MAX_SESSION_MEMORY_MB: int = int(os.getenv("MAX_SESSION_MEMORY_MB", 500))
        self.MAX_ANNOTATIONS_PER_SESSION: int = int(os.getenv("MAX_ANNOTATIONS_PER_SESSION", 1000))
        
        # AI Visual Highlighting Settings
        self.AI_MAX_PAGES: int = int(os.getenv("AI_MAX_PAGES", 100))
        self.AI_BATCH_SIZE: int = int(os.getenv("AI_BATCH_SIZE", 10))
        self.AI_IMAGE_QUALITY: int = int(os.getenv("AI_IMAGE_QUALITY", 85))
        self.AI_MAX_IMAGE_DIMENSION: int = int(os.getenv("AI_MAX_IMAGE_DIMENSION", 2000))
        self.AI_ENABLE_HIGHLIGHTING: bool = os.getenv("AI_ENABLE_HIGHLIGHTING", "true").lower() == "true"
        
        # Grid System Settings
        self.GRID_DETECTION_CONFIDENCE: float = float(os.getenv("GRID_DETECTION_CONFIDENCE", 0.8))
        self.MAX_VISUAL_ELEMENTS: int = int(os.getenv("MAX_VISUAL_ELEMENTS", 200))  # Max elements to highlight
        
        # Storage Optimization Settings
        self.STORAGE_MAX_CONCURRENT_UPLOADS: int = int(os.getenv("STORAGE_MAX_CONCURRENT_UPLOADS", 5))
        self.STORAGE_MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("STORAGE_MAX_CONCURRENT_DOWNLOADS", 5))
        self.STORAGE_ENABLE_PAGINATION: bool = os.getenv("STORAGE_ENABLE_PAGINATION", "true").lower() == "true"

        # Background Processing Settings
        self.ENABLE_BACKGROUND_PROCESSING: bool = os.getenv("ENABLE_BACKGROUND_PROCESSING", "true").lower() == "true"
        self.MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", 5))
        self.JOB_TIMEOUT_SECONDS: int = int(os.getenv("JOB_TIMEOUT_SECONDS", 1800))  # 30 minutes
        self.PROCESSING_CHECK_INTERVAL: int = int(os.getenv("PROCESSING_CHECK_INTERVAL", 5))  # 5 seconds

        # Timeout Settings
        self.REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 600))  # 10 minutes
        self.UPLOAD_TIMEOUT: int = int(os.getenv("UPLOAD_TIMEOUT", 3600))  # 1 hour
        
        # Trade Configuration
        self.TRADES_LIST: List[str] = os.getenv(
            "TRADES_LIST", 
            "General,Electrical,Plumbing,HVAC,Fire Protection,Structural,Architectural"
        ).split(",")
        self.ENABLE_TRADE_COORDINATION: bool = os.getenv("ENABLE_TRADE_COORDINATION", "true").lower() == "true"
        
        # Feature Flags
        self.ENABLE_DOCUMENT_NOTES: bool = os.getenv("ENABLE_DOCUMENT_NOTES", "true").lower() == "true"
        self.ENABLE_AI_HIGHLIGHTING: bool = os.getenv("ENABLE_AI_HIGHLIGHTING", "true").lower() == "true"
        self.ENABLE_PRIVATE_NOTES: bool = os.getenv("ENABLE_PRIVATE_NOTES", "true").lower() == "true"
        self.ENABLE_NOTE_PUBLISHING: bool = os.getenv("ENABLE_NOTE_PUBLISHING", "true").lower() == "true"
        
        # Environment
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    def get_note_types(self) -> List[str]:
        """Get allowed note types"""
        return self.NOTE_TYPES
    
    def get_trades_list(self) -> List[str]:
        """Get list of construction trades"""
        return self.TRADES_LIST
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() in ["development", "dev"]
    
    def get_max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a proper list"""
        if isinstance(self.CORS_ORIGINS, str):
            try:
                return eval(self.CORS_ORIGINS)
            except:
                return ["*"]
        return self.CORS_ORIGINS


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached settings instance"""
    return AppSettings()
