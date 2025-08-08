# app/core/config.py - Complete Fixed Version with All Settings + SSE Support + Visual Grid Detection
"""
Configuration for Smart Blueprint Chat API with SSE Support and Visual Grid Detection

CORE PHILOSOPHY:
1. Load ALL thumbnails - no artificial limits
2. Smart page selection based on actual query needs  
3. Storage is not a constraint - accuracy is paramount
4. The system adapts to document size, not the other way around
5. Real-time updates via Server-Sent Events (SSE)
6. Visual grid detection for accurate blueprint analysis

DATA LOADING APPROACH:
- Check metadata for page count
- If no metadata, probe to find all pages (up to 200)
- Load ALL thumbnails found
- Smart AI selection from complete thumbnail set
- Load high-res only for selected pages
- Detect actual grid lines from blueprint visuals
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path


class AppSettings:
    """
    Application settings optimized for multi-user collaborative blueprint analysis.
    
    CORE PHILOSOPHY:
    - Load ALL available thumbnails from documents (no artificial limits)
    - Smart page selection based on actual needs
    - Comprehensive analysis with unlimited page support
    - Storage is not a constraint - accuracy is paramount
    - Real-time updates via Server-Sent Events
    - Visual grid detection for precise blueprint analysis
    """
    
    def __init__(self):
        # === GENERAL SETTINGS ===
        self.PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Smart Blueprint Chat API")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
        
        # === SERVER SETTINGS ===
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", 8000))
        
        # === CORS SETTINGS ===
        cors_origins = os.getenv("CORS_ORIGINS", '["*"]')
        try:
            # Using json.loads is safer than eval()
            self.CORS_ORIGINS: List[str] = json.loads(cors_origins) if isinstance(cors_origins, str) else cors_origins
        except:
            self.CORS_ORIGINS: List[str] = ["*"]
        
        # === SSE (SERVER-SENT EVENTS) SETTINGS ===
        self.SSE_KEEPALIVE_INTERVAL: int = int(os.getenv("SSE_KEEPALIVE_INTERVAL", 30))  # Keepalive ping every 30 seconds
        self.SSE_CLIENT_TIMEOUT: int = int(os.getenv("SSE_CLIENT_TIMEOUT", 300))  # Disconnect idle clients after 5 minutes
        self.SSE_MAX_CONNECTIONS_PER_DOCUMENT: int = int(os.getenv("SSE_MAX_CONNECTIONS_PER_DOCUMENT", 50))  # Allow many concurrent viewers
        self.SSE_RETRY_INTERVAL: int = int(os.getenv("SSE_RETRY_INTERVAL", 5000))  # Client retry after 5 seconds (ms)
        self.SSE_EVENT_QUEUE_SIZE: int = int(os.getenv("SSE_EVENT_QUEUE_SIZE", 1000))  # Max events to queue per document
        self.SSE_ENABLE_COMPRESSION: bool = os.getenv("SSE_ENABLE_COMPRESSION", "true").lower() == "true"  # Compress event data
        self.ENABLE_SSE: bool = os.getenv("ENABLE_SSE", "true").lower() == "true"  # Master feature flag
        
        # === NEW SSE SETTINGS FOR FIXES ===
        self.SSE_MAX_QUEUE_SIZE_PER_CONNECTION: int = int(os.getenv("SSE_MAX_QUEUE_SIZE_PER_CONNECTION", 1000))
        self.SSE_DROPPED_EVENT_THRESHOLD: int = int(os.getenv("SSE_DROPPED_EVENT_THRESHOLD", 100))
        self.SSE_SLOW_CONSUMER_THRESHOLD: int = int(os.getenv("SSE_SLOW_CONSUMER_THRESHOLD", 50))
        self.SSE_EVENT_HISTORY_CLEANUP_SIZE: int = int(os.getenv("SSE_EVENT_HISTORY_CLEANUP_SIZE", 25))
        
        # === AZURE BLOB STORAGE ===
        self.AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.AZURE_CONTAINER_NAME: str = os.getenv("AZURE_CONTAINER_NAME", "blueprints")
        self.AZURE_CACHE_CONTAINER_NAME: str = os.getenv("AZURE_CACHE_CONTAINER_NAME", "blueprints-cache")
        
        # === OPENAI SETTINGS ===
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", 4000))
        self.OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
        
        # === PDF PROCESSING SETTINGS ===
        self.PDF_PREVIEW_RESOLUTION: int = int(os.getenv("PDF_PREVIEW_RESOLUTION", 72))
        self.PDF_HIGH_RESOLUTION: int = int(os.getenv("PDF_HIGH_RESOLUTION", 150))
        self.PDF_IMAGE_DPI: int = int(os.getenv("PDF_IMAGE_DPI", 150))
        self.PDF_AI_DPI: int = int(os.getenv("PDF_AI_DPI", 700))
        self.PDF_THUMBNAIL_DPI: int = int(os.getenv("PDF_THUMBNAIL_DPI", 72))
        self.PDF_MAX_PAGES: int = int(os.getenv("PDF_MAX_PAGES", 200))  # Increased from 100
        self.PROCESSING_BATCH_SIZE: int = int(os.getenv("PROCESSING_BATCH_SIZE", 25))  # Updated: Increased from 10
        self.PDF_PNG_COMPRESSION: int = int(os.getenv("PDF_PNG_COMPRESSION", 6))
        self.PDF_JPEG_QUALITY: int = int(os.getenv("PDF_JPEG_QUALITY", 95))
        
        # === NEW PROCESSING SETTINGS FOR FIXES ===
        self.STATUS_UPDATE_LOCK_TIMEOUT: float = float(os.getenv("STATUS_UPDATE_LOCK_TIMEOUT", 5.0))
        self.PROCESSING_HEALTH_CHECK_INTERVAL: int = int(os.getenv("PROCESSING_HEALTH_CHECK_INTERVAL", 30))
        
        # === CACHE AND MEMORY LIMITS ===
        self.MAX_MEMORY_CACHE_SIZE: int = int(os.getenv("MAX_MEMORY_CACHE_SIZE", 100))
        self.MAX_CACHE_SIZE_MB: int = int(os.getenv("MAX_CACHE_SIZE_MB", 10000))  # Updated: 10GB (was 1000)
        self.CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/blueprint_cache")
        self.CACHE_DIR_PATH: Path = Path(self.CACHE_DIR)
        self.ENABLE_DISK_CACHE: bool = os.getenv("ENABLE_DISK_CACHE", "true").lower() == "true"
        
        # Cache TTL Settings - Updated for better performance with large documents
        self.CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 21600))  # Updated: 6 hours (was 7200)
        self.METADATA_CACHE_TTL: int = int(os.getenv("METADATA_CACHE_TTL", 14400))  # Updated: 4 hours (was 3600)
        self.IMAGE_CACHE_TTL: int = int(os.getenv("IMAGE_CACHE_TTL", 21600))  # Updated: 6 hours (was 7200)
        self.ANALYSIS_CACHE_TTL: int = int(os.getenv("ANALYSIS_CACHE_TTL", 28800))  # Updated: 8 hours (was 14400)
        self.THUMBNAIL_CACHE_TTL: int = int(os.getenv("THUMBNAIL_CACHE_TTL", 28800))  # Updated: 8 hours (was 10800)
        
        # Memory Cache Limits - Updated for large documents
        self.MAX_MEMORY_ITEMS: int = int(os.getenv("MAX_MEMORY_ITEMS", 5000))  # Updated from enhanced_cache.py
        self.MAX_THUMBNAIL_ITEMS: int = int(os.getenv("MAX_THUMBNAIL_ITEMS", 1500))  # Updated from enhanced_cache.py
        self.MAX_MEMORY_ITEM_SIZE_MB: int = int(os.getenv("MAX_MEMORY_ITEM_SIZE_MB", 50))  # Updated from enhanced_cache.py
        
        # === SECURITY AND ADMIN ===
        self.ADMIN_SECRET_TOKEN: str = os.getenv("ADMIN_SECRET_TOKEN", "blueprintreader789")
        
        # === ACTIVITY AND LOGGING ===
        self.MAX_ACTIVITY_LOGS: int = int(os.getenv("MAX_ACTIVITY_LOGS", 1000))
        self.MAX_CHAT_LOGS: int = int(os.getenv("MAX_CHAT_LOGS", 1000))
        self.MAX_USER_CHAT_HISTORY: int = int(os.getenv("MAX_USER_CHAT_HISTORY", 100))
        
        # === FILE SIZE LIMITS ===
        # Set to 0 for unlimited, but 100MB is reasonable default
        self.MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 100))  # Increased from 60MB
        
        # === NOTE SYSTEM SETTINGS ===
        self.MAX_NOTES_PER_DOCUMENT: int = int(os.getenv("MAX_NOTES_PER_DOCUMENT", 500))
        self.MAX_NOTE_LENGTH: int = int(os.getenv("MAX_NOTE_LENGTH", 10000))
        self.MAX_TOTAL_NOTE_CHARS: int = int(os.getenv("MAX_TOTAL_NOTE_CHARS", 500000))
        self.NOTE_TYPES: List[str] = os.getenv(
            "NOTE_TYPES", 
            "general,question,issue,suggestion,review,coordination,warning"
        ).split(",")
        
        # === SESSION MANAGEMENT SETTINGS ===
        self.MAX_SESSIONS_IN_MEMORY: int = int(os.getenv("MAX_SESSIONS_IN_MEMORY", 100))
        self.SESSION_CLEANUP_HOURS: int = int(os.getenv("SESSION_CLEANUP_HOURS", 24))
        self.SESSION_CLEANUP_INTERVAL_SECONDS: int = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", 3600))
        self.MAX_SESSION_MEMORY_MB: int = int(os.getenv("MAX_SESSION_MEMORY_MB", 500))
        self.MAX_ANNOTATIONS_PER_SESSION: int = int(os.getenv("MAX_ANNOTATIONS_PER_SESSION", 1000))
        
        # === AI VISUAL HIGHLIGHTING SETTINGS ===
        self.AI_MAX_PAGES: int = int(os.getenv("AI_MAX_PAGES", 200))  # Increased to support larger documents
        self.AI_BATCH_SIZE: int = int(os.getenv("AI_BATCH_SIZE", 25))  # Updated: Increased from 20
        self.AI_IMAGE_QUALITY: int = int(os.getenv("AI_IMAGE_QUALITY", 85))
        self.AI_MAX_IMAGE_DIMENSION: int = int(os.getenv("AI_MAX_IMAGE_DIMENSION", 2000))
        self.AI_ENABLE_HIGHLIGHTING: bool = os.getenv("AI_ENABLE_HIGHLIGHTING", "true").lower() == "true"
        
        # === VISION INTELLIGENCE SETTINGS - UPDATED FOR LARGE DOCUMENTS ===
        self.VISION_TIMEOUT: float = float(os.getenv("VISION_TIMEOUT", 300.0))  # Updated: 5 minutes (was 60.0)
        self.VISION_REQUEST_TIMEOUT: float = float(os.getenv("VISION_REQUEST_TIMEOUT", 600.0))  # Updated: 10 minutes (was 120.0)
        self.VISION_MAX_RETRIES: int = int(os.getenv("VISION_MAX_RETRIES", 3))
        self.VISION_INFERENCE_LIMIT: int = int(os.getenv("VISION_INFERENCE_LIMIT", 10))
        self.VISION_PRODUCTION_TOKENS: int = int(os.getenv("VISION_PRODUCTION_TOKENS", 8000))
        self.VISION_BATCH_SIZE: int = int(os.getenv("VISION_BATCH_SIZE", 20))  # Updated: Increased from 10
        self.VISION_PARALLEL_PROCESSING: bool = os.getenv("VISION_PARALLEL_PROCESSING", "true").lower() == "true"
        self.VISION_VALIDATION_TIMEOUT: float = float(os.getenv("VISION_VALIDATION_TIMEOUT", 120.0))  # Updated: 2 minutes (was 30.0)
        
        # === DATA LOADER SETTINGS ===
        # Philosophy: Load ALL available thumbnails - no artificial limits
        # The system should load as many thumbnails as exist in the document
        self.MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))
        self.RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", 1.0))
        self.RETRY_BACKOFF_FACTOR: float = float(os.getenv("RETRY_BACKOFF_FACTOR", 2.0))
        self.THUMBNAIL_PROBE_LIMIT: int = int(os.getenv("THUMBNAIL_PROBE_LIMIT", 200))  # Increased probe limit
        self.BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 25))  # Updated: Increased from 10
        self.PARALLEL_PROCESSING: bool = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
        self.AGGRESSIVE_CACHING: bool = os.getenv("AGGRESSIVE_CACHING", "true").lower() == "true"
        
        # Thumbnail specific settings - LOAD ALL AVAILABLE
        self.LOAD_ALL_THUMBNAILS: bool = os.getenv("LOAD_ALL_THUMBNAILS", "true").lower() == "true"  # Always true
        self.THUMBNAIL_LOAD_TIMEOUT: float = float(os.getenv("THUMBNAIL_LOAD_TIMEOUT", 900.0))  # Updated: 15 minutes (was 120.0)
        self.THUMBNAIL_BATCH_DELAY: float = float(os.getenv("THUMBNAIL_BATCH_DELAY", 0.1))
        self.THUMBNAIL_PROBE_MAX_PAGES: int = int(os.getenv("THUMBNAIL_PROBE_MAX_PAGES", 200))  # Probe up to 200 pages
        self.PAGE_LOAD_TIMEOUT: float = float(os.getenv("PAGE_LOAD_TIMEOUT", 600.0))  # Added: 10 minutes
        
        # === PAGE SELECTION SETTINGS ===
        # No artificial limits - smart selection based on query needs
        self.MAX_PAGES_FOR_ANALYSIS: int = int(os.getenv("MAX_PAGES_FOR_ANALYSIS", 100))  # Analyze up to 100 pages
        self.UNLIMITED_PAGE_LOADING: bool = os.getenv("UNLIMITED_PAGE_LOADING", "true").lower() == "true"
        self.PAGE_SELECTION_CHUNK_SIZE: int = int(os.getenv("PAGE_SELECTION_CHUNK_SIZE", 25))  # Updated: Increased from 10
        self.ENABLE_SMART_PAGE_SELECTION: bool = os.getenv("ENABLE_SMART_PAGE_SELECTION", "true").lower() == "true"
        
        # === GRID SYSTEM SETTINGS ===
        self.GRID_DETECTION_CONFIDENCE: float = float(os.getenv("GRID_DETECTION_CONFIDENCE", 0.8))
        self.MAX_VISUAL_ELEMENTS: int = int(os.getenv("MAX_VISUAL_ELEMENTS", 200))
        
        # === VISUAL GRID DETECTION SETTINGS (NEW) ===
        self.VISUAL_GRID_DETECTION_ENABLED: bool = os.getenv("VISUAL_GRID_DETECTION_ENABLED", "true").lower() == "true"
        self.GRID_DETECTION_MIN_CONFIDENCE: float = float(os.getenv("GRID_DETECTION_MIN_CONFIDENCE", 0.7))
        self.GRID_LINE_DETECTION_THRESHOLD: int = int(os.getenv("GRID_LINE_DETECTION_THRESHOLD", 100))
        self.GRID_MIN_LINE_LENGTH: int = int(os.getenv("GRID_MIN_LINE_LENGTH", 100))  # Minimum pixels for a grid line
        self.GRID_LINE_GAP_TOLERANCE: int = int(os.getenv("GRID_LINE_GAP_TOLERANCE", 50))  # Maximum gap in pixels
        self.GRID_LINE_THICKNESS_MIN: float = float(os.getenv("GRID_LINE_THICKNESS_MIN", 0.5))
        self.GRID_LINE_THICKNESS_MAX: float = float(os.getenv("GRID_LINE_THICKNESS_MAX", 3.0))
        self.GRID_DETECTION_DPI_MULTIPLIER: float = float(os.getenv("GRID_DETECTION_DPI_MULTIPLIER", 2.0))  # 2x zoom for detection
        self.GRID_LINE_ANGLE_TOLERANCE: float = float(os.getenv("GRID_LINE_ANGLE_TOLERANCE", 5.0))  # Degrees from horizontal/vertical
        self.GRID_LINE_GROUPING_DISTANCE: int = int(os.getenv("GRID_LINE_GROUPING_DISTANCE", 10))  # Pixels to group lines
        self.GRID_DETECTION_CANNY_LOW: int = int(os.getenv("GRID_DETECTION_CANNY_LOW", 50))  # Canny edge detection low threshold
        self.GRID_DETECTION_CANNY_HIGH: int = int(os.getenv("GRID_DETECTION_CANNY_HIGH", 150))  # Canny edge detection high threshold
        self.GRID_DETECTION_HOUGH_THRESHOLD: int = int(os.getenv("GRID_DETECTION_HOUGH_THRESHOLD", 100))  # Hough transform threshold
        self.GRID_MAX_LABELS_X: int = int(os.getenv("GRID_MAX_LABELS_X", 26))  # Maximum X labels (A-Z)
        self.GRID_MAX_LABELS_Y: int = int(os.getenv("GRID_MAX_LABELS_Y", 50))  # Maximum Y labels (1-50)
        
        # === STORAGE OPTIMIZATION SETTINGS - UPDATED ===
        self.STORAGE_MAX_CONCURRENT_UPLOADS: int = int(os.getenv("STORAGE_MAX_CONCURRENT_UPLOADS", 25))  # Updated: Increased from 10
        self.STORAGE_MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("STORAGE_MAX_CONCURRENT_DOWNLOADS", 25))  # Updated: Increased from 10
        self.STORAGE_ENABLE_PAGINATION: bool = os.getenv("STORAGE_ENABLE_PAGINATION", "true").lower() == "true"
        self.STORAGE_DOWNLOAD_TIMEOUT: float = float(os.getenv("STORAGE_DOWNLOAD_TIMEOUT", 300.0))  # Updated: 5 minutes (was 60.0)
        self.STORAGE_UPLOAD_TIMEOUT: float = float(os.getenv("STORAGE_UPLOAD_TIMEOUT", 600.0))  # Updated: 10 minutes (was 120.0)
        self.STORAGE_BATCH_TIMEOUT: float = float(os.getenv("STORAGE_BATCH_TIMEOUT", 1200.0))  # Added: 20 minutes
        
        # === NEW STORAGE RETRY SETTINGS FOR FIXES ===
        self.STORAGE_RETRY_ATTEMPTS: int = int(os.getenv("STORAGE_RETRY_ATTEMPTS", 3))
        self.STORAGE_RETRY_DELAY: float = float(os.getenv("STORAGE_RETRY_DELAY", 1.0))
        
        # === BACKGROUND PROCESSING SETTINGS ===
        self.ENABLE_BACKGROUND_PROCESSING: bool = os.getenv("ENABLE_BACKGROUND_PROCESSING", "true").lower() == "true"
        self.MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", 5))
        self.JOB_TIMEOUT_SECONDS: int = int(os.getenv("JOB_TIMEOUT_SECONDS", 7200))  # Updated: 2 hours (was 1800)
        self.PROCESSING_CHECK_INTERVAL: int = int(os.getenv("PROCESSING_CHECK_INTERVAL", 5))
        
        # === TIMEOUT SETTINGS - UPDATED FOR LARGE DOCUMENTS ===
        self.REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 3600))  # Updated: 60 minutes (was 600)
        self.UPLOAD_TIMEOUT: int = int(os.getenv("UPLOAD_TIMEOUT", 3600))  # Already 1 hour
        
        # === TRADE CONFIGURATION ===
        self.TRADES_LIST: List[str] = os.getenv(
            "TRADES_LIST", 
            "General,Electrical,Plumbing,HVAC,Fire Protection,Structural,Architectural"
        ).split(",")
        self.ENABLE_TRADE_COORDINATION: bool = os.getenv("ENABLE_TRADE_COORDINATION", "true").lower() == "true"
        
        # === FEATURE FLAGS ===
        self.ENABLE_DOCUMENT_NOTES: bool = os.getenv("ENABLE_DOCUMENT_NOTES", "true").lower() == "true"
        self.ENABLE_AI_HIGHLIGHTING: bool = os.getenv("ENABLE_AI_HIGHLIGHTING", "true").lower() == "true"
        self.ENABLE_PRIVATE_NOTES: bool = os.getenv("ENABLE_PRIVATE_NOTES", "true").lower() == "true"
        self.ENABLE_NOTE_PUBLISHING: bool = os.getenv("ENABLE_NOTE_PUBLISHING", "true").lower() == "true"
        
        # === VALIDATION SETTINGS - UPDATED ===
        self.VALIDATION_MAX_RETRIES: int = int(os.getenv("VALIDATION_MAX_RETRIES", 2))
        self.ENABLE_4X_VALIDATION: bool = os.getenv("ENABLE_4X_VALIDATION", "true").lower() == "true"
        self.MAX_CONCURRENT_VALIDATIONS: int = int(os.getenv("MAX_CONCURRENT_VALIDATIONS", 6))  # Updated: 6 (was 4)
        
        # === ELEMENT DETECTION SETTINGS - UPDATED ===
        self.ELEMENT_DETECTION_CONFIDENCE: float = float(os.getenv("ELEMENT_DETECTION_CONFIDENCE", 0.7))
        self.ELEMENT_DETECTION_TIMEOUT: float = float(os.getenv("ELEMENT_DETECTION_TIMEOUT", 60.0))  # Updated: 1 minute (was 10.0)
        
        # === MONITORING & WARNINGS ===
        self.WARNING_THRESHOLD: float = float(os.getenv("WARNING_THRESHOLD", 300.0))  # Added: 5 minutes
        
        # === CREATE CACHE DIRECTORY IF NEEDED ===
        if self.ENABLE_DISK_CACHE:
            try:
                self.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create cache directory: {e}")
    
    # === HELPER METHODS ===
    
    def get_sse_settings(self) -> Dict[str, Any]:
        """Get SSE configuration settings"""
        return {
            "enabled": self.ENABLE_SSE,
            "keepalive_interval": self.SSE_KEEPALIVE_INTERVAL,
            "client_timeout": self.SSE_CLIENT_TIMEOUT,
            "max_connections_per_document": self.SSE_MAX_CONNECTIONS_PER_DOCUMENT,
            "retry_interval": self.SSE_RETRY_INTERVAL,
            "event_queue_size": self.SSE_EVENT_QUEUE_SIZE,
            "enable_compression": self.SSE_ENABLE_COMPRESSION,
            "max_queue_size_per_connection": self.SSE_MAX_QUEUE_SIZE_PER_CONNECTION,
            "dropped_event_threshold": self.SSE_DROPPED_EVENT_THRESHOLD,
            "slow_consumer_threshold": self.SSE_SLOW_CONSUMER_THRESHOLD,
            "event_history_cleanup_size": self.SSE_EVENT_HISTORY_CLEANUP_SIZE
        }
    
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
        return self.CORS_ORIGINS
    
    def get_session_settings(self) -> Dict[str, Any]:
        """Get session management settings"""
        return {
            "cleanup_interval": self.SESSION_CLEANUP_INTERVAL_SECONDS,
            "max_sessions": self.MAX_SESSIONS_IN_MEMORY,
            "max_memory_mb": self.MAX_SESSION_MEMORY_MB,
            "expiry_hours": self.SESSION_CLEANUP_HOURS,
            "max_annotations": self.MAX_ANNOTATIONS_PER_SESSION
        }
    
    def get_storage_settings(self) -> Dict[str, Any]:
        """Get storage-specific settings"""
        return {
            "connection_string": self.AZURE_STORAGE_CONNECTION_STRING,
            "main_container": self.AZURE_CONTAINER_NAME,
            "cache_container": self.AZURE_CACHE_CONTAINER_NAME,
            "max_concurrent_uploads": self.STORAGE_MAX_CONCURRENT_UPLOADS,
            "max_concurrent_downloads": self.STORAGE_MAX_CONCURRENT_DOWNLOADS,
            "enable_pagination": self.STORAGE_ENABLE_PAGINATION,
            "download_timeout": self.STORAGE_DOWNLOAD_TIMEOUT,
            "upload_timeout": self.STORAGE_UPLOAD_TIMEOUT,
            "batch_timeout": self.STORAGE_BATCH_TIMEOUT,
            "retry_attempts": self.STORAGE_RETRY_ATTEMPTS,
            "retry_delay": self.STORAGE_RETRY_DELAY
        }
    
    def get_ai_settings(self) -> Dict[str, Any]:
        """Get AI service settings"""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.OPENAI_MODEL,
            "max_tokens": self.OPENAI_MAX_TOKENS,
            "temperature": self.OPENAI_TEMPERATURE,
            "max_pages": self.AI_MAX_PAGES,
            "batch_size": self.AI_BATCH_SIZE,
            "image_quality": self.AI_IMAGE_QUALITY,
            "max_image_dimension": self.AI_MAX_IMAGE_DIMENSION,
            "enable_highlighting": self.AI_ENABLE_HIGHLIGHTING
        }
    
    def get_vision_settings(self) -> Dict[str, Any]:
        """Get Vision Intelligence settings"""
        return {
            "vision_timeout": self.VISION_TIMEOUT,
            "request_timeout": self.VISION_REQUEST_TIMEOUT,
            "max_retries": self.VISION_MAX_RETRIES,
            "inference_limit": self.VISION_INFERENCE_LIMIT,
            "production_tokens": self.VISION_PRODUCTION_TOKENS,
            "batch_size": self.VISION_BATCH_SIZE,
            "parallel_processing": self.VISION_PARALLEL_PROCESSING,
            "validation_timeout": self.VISION_VALIDATION_TIMEOUT
        }
    
    def get_cache_settings(self) -> Dict[str, Any]:
        """Get cache-specific settings"""
        return {
            "max_size_mb": self.MAX_CACHE_SIZE_MB,
            "cache_dir": self.CACHE_DIR,
            "cache_dir_path": self.CACHE_DIR_PATH,
            "enable_disk_cache": self.ENABLE_DISK_CACHE,
            "metadata_ttl": self.METADATA_CACHE_TTL,
            "image_ttl": self.IMAGE_CACHE_TTL,
            "analysis_ttl": self.ANALYSIS_CACHE_TTL,
            "thumbnail_ttl": self.THUMBNAIL_CACHE_TTL,
            "default_ttl": self.CACHE_TTL_SECONDS,
            "max_memory_items": self.MAX_MEMORY_ITEMS,
            "max_thumbnail_items": self.MAX_THUMBNAIL_ITEMS,
            "max_memory_item_size_mb": self.MAX_MEMORY_ITEM_SIZE_MB
        }
    
    def get_data_loader_settings(self) -> Dict[str, Any]:
        """
        Get data loader settings
        
        Key principle: Load ALL thumbnails that exist in the document
        - Use metadata to determine page count
        - If no metadata, probe to find all pages
        - No artificial limits on thumbnail loading
        """
        return {
            "max_retries": self.MAX_RETRIES,
            "retry_delay": self.RETRY_DELAY,
            "retry_backoff_factor": self.RETRY_BACKOFF_FACTOR,
            "thumbnail_probe_limit": self.THUMBNAIL_PROBE_LIMIT,
            "batch_size": self.BATCH_SIZE,
            "parallel_processing": self.PARALLEL_PROCESSING,
            "aggressive_caching": self.AGGRESSIVE_CACHING,
            "load_all_thumbnails": self.LOAD_ALL_THUMBNAILS,
            "thumbnail_load_timeout": self.THUMBNAIL_LOAD_TIMEOUT,
            "thumbnail_batch_delay": self.THUMBNAIL_BATCH_DELAY,
            "thumbnail_probe_max_pages": self.THUMBNAIL_PROBE_MAX_PAGES,
            "page_load_timeout": self.PAGE_LOAD_TIMEOUT
        }
    
    def get_page_selection_settings(self) -> Dict[str, Any]:
        """
        Get page selection settings
        
        Key principle: Smart selection from ALL available thumbnails
        - Analyze all thumbnails to select relevant pages
        - No artificial limits on selection
        - Let the AI decide based on query needs
        """
        return {
            "max_pages_for_analysis": self.MAX_PAGES_FOR_ANALYSIS,
            "unlimited_page_loading": self.UNLIMITED_PAGE_LOADING,
            "page_selection_chunk_size": self.PAGE_SELECTION_CHUNK_SIZE,
            "enable_smart_page_selection": self.ENABLE_SMART_PAGE_SELECTION
        }
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get validation settings"""
        return {
            "timeout": self.VISION_VALIDATION_TIMEOUT, # Corrected to use the single source
            "max_retries": self.VALIDATION_MAX_RETRIES,
            "enable_4x": self.ENABLE_4X_VALIDATION,
            "max_concurrent": self.MAX_CONCURRENT_VALIDATIONS
        }
    
    def get_grid_detection_settings(self) -> Dict[str, Any]:
        """Get visual grid detection settings"""
        return {
            "enabled": self.VISUAL_GRID_DETECTION_ENABLED,
            "min_confidence": self.GRID_DETECTION_MIN_CONFIDENCE,
            "detection_threshold": self.GRID_LINE_DETECTION_THRESHOLD,
            "min_line_length": self.GRID_MIN_LINE_LENGTH,
            "line_gap_tolerance": self.GRID_LINE_GAP_TOLERANCE,
            "line_thickness_range": (self.GRID_LINE_THICKNESS_MIN, self.GRID_LINE_THICKNESS_MAX),
            "dpi_multiplier": self.GRID_DETECTION_DPI_MULTIPLIER,
            "angle_tolerance": self.GRID_LINE_ANGLE_TOLERANCE,
            "grouping_distance": self.GRID_LINE_GROUPING_DISTANCE,
            "canny_thresholds": (self.GRID_DETECTION_CANNY_LOW, self.GRID_DETECTION_CANNY_HIGH),
            "hough_threshold": self.GRID_DETECTION_HOUGH_THRESHOLD,
            "max_labels": {"x": self.GRID_MAX_LABELS_X, "y": self.GRID_MAX_LABELS_Y}
        }
    
    def get_unlimited_loading_settings(self) -> Dict[str, Any]:
        """
        Get settings for unlimited thumbnail/page loading
        
        These settings ensure the system loads ALL available content
        """
        return {
            "load_all_thumbnails": self.LOAD_ALL_THUMBNAILS,
            "unlimited_page_loading": self.UNLIMITED_PAGE_LOADING,
            "thumbnail_probe_max_pages": self.THUMBNAIL_PROBE_MAX_PAGES,
            "max_pages_for_analysis": self.MAX_PAGES_FOR_ANALYSIS,
            "pdf_max_pages": self.PDF_MAX_PAGES,
            "ai_max_pages": self.AI_MAX_PAGES,
            "max_visual_elements": self.MAX_VISUAL_ELEMENTS,
            "max_file_size_mb": self.MAX_FILE_SIZE_MB,
            "max_cache_size_mb": self.MAX_CACHE_SIZE_MB,
        }
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing-specific settings"""
        return {
            "status_update_lock_timeout": self.STATUS_UPDATE_LOCK_TIMEOUT,
            "processing_health_check_interval": self.PROCESSING_HEALTH_CHECK_INTERVAL,
            "processing_check_interval": self.PROCESSING_CHECK_INTERVAL,
            "job_timeout_seconds": self.JOB_TIMEOUT_SECONDS,
            "max_concurrent_jobs": self.MAX_CONCURRENT_JOBS,
            "enable_background_processing": self.ENABLE_BACKGROUND_PROCESSING
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        feature_checks = {
            "storage": bool(self.AZURE_STORAGE_CONNECTION_STRING),
            "ai": bool(self.OPENAI_API_KEY),
            "admin": bool(self.ADMIN_SECRET_TOKEN),
            "notes": self.ENABLE_DOCUMENT_NOTES,
            "highlighting": self.ENABLE_AI_HIGHLIGHTING,
            "private_notes": self.ENABLE_PRIVATE_NOTES,
            "note_publishing": self.ENABLE_NOTE_PUBLISHING,
            "trade_coordination": self.ENABLE_TRADE_COORDINATION,
            "pdf_processing": bool(self.AZURE_STORAGE_CONNECTION_STRING),
            "sessions": True,
            "disk_cache": self.ENABLE_DISK_CACHE,
            "background_processing": self.ENABLE_BACKGROUND_PROCESSING,
            "4x_validation": self.ENABLE_4X_VALIDATION,
            "unlimited_loading": self.UNLIMITED_PAGE_LOADING,
            "load_all_thumbnails": self.LOAD_ALL_THUMBNAILS,
            "sse": self.ENABLE_SSE,  # Added SSE feature check
            "visual_grid_detection": self.VISUAL_GRID_DETECTION_ENABLED,  # Added visual grid detection
        }
        return feature_checks.get(feature, False)
    
    def get_pdf_settings(self) -> Dict[str, Any]:
        """Get PDF processing settings"""
        return {
            "preview_resolution": self.PDF_PREVIEW_RESOLUTION,
            "high_resolution": self.PDF_HIGH_RESOLUTION,
            "image_dpi": self.PDF_IMAGE_DPI,
            "ai_dpi": self.PDF_AI_DPI,
            "thumbnail_dpi": self.PDF_THUMBNAIL_DPI,
            "max_pages": self.PDF_MAX_PAGES,
            "batch_size": self.PROCESSING_BATCH_SIZE,
            "png_compression": self.PDF_PNG_COMPRESSION,
            "jpeg_quality": self.PDF_JPEG_QUALITY
        }


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached settings instance"""
    return AppSettings()


# === CONFIG DICTIONARY FOR VISION_AI MODULE ===
# The vision_ai module expects CONFIG to be a dictionary
# Philosophy: Load ALL thumbnails, no artificial limits
_settings = get_settings()

CONFIG = {
    # Data loader settings
    "max_retries": _settings.MAX_RETRIES,
    "retry_delay": _settings.RETRY_DELAY,
    "retry_backoff_factor": _settings.RETRY_BACKOFF_FACTOR,
    "thumbnail_probe_limit": _settings.THUMBNAIL_PROBE_LIMIT,
    "batch_size": _settings.BATCH_SIZE,
    "parallel_processing": _settings.PARALLEL_PROCESSING,
    "aggressive_caching": _settings.AGGRESSIVE_CACHING,
    "load_all_thumbnails": _settings.LOAD_ALL_THUMBNAILS,
    "thumbnail_load_timeout": _settings.THUMBNAIL_LOAD_TIMEOUT,
    "thumbnail_batch_delay": _settings.THUMBNAIL_BATCH_DELAY,
    "thumbnail_probe_max_pages": _settings.THUMBNAIL_PROBE_MAX_PAGES,
    "page_load_timeout": _settings.PAGE_LOAD_TIMEOUT,
    
    # Cache settings
    "max_cache_size_mb": _settings.MAX_CACHE_SIZE_MB,
    "cache_dir": _settings.CACHE_DIR,
    "cache_dir_path": _settings.CACHE_DIR_PATH,
    "enable_disk_cache": _settings.ENABLE_DISK_CACHE,
    "metadata_cache_ttl": _settings.METADATA_CACHE_TTL,
    "image_cache_ttl": _settings.IMAGE_CACHE_TTL,
    "analysis_cache_ttl": _settings.ANALYSIS_CACHE_TTL,
    "thumbnail_cache_ttl": _settings.THUMBNAIL_CACHE_TTL,
    "cache_ttl_seconds": _settings.CACHE_TTL_SECONDS,
    "max_memory_items": _settings.MAX_MEMORY_ITEMS,
    "max_thumbnail_items": _settings.MAX_THUMBNAIL_ITEMS,
    "max_memory_item_size_mb": _settings.MAX_MEMORY_ITEM_SIZE_MB,
    
    # Page selection settings
    "max_pages_for_analysis": _settings.MAX_PAGES_FOR_ANALYSIS,
    "unlimited_page_loading": _settings.UNLIMITED_PAGE_LOADING,
    "page_selection_chunk_size": _settings.PAGE_SELECTION_CHUNK_SIZE,
    "enable_smart_page_selection": _settings.ENABLE_SMART_PAGE_SELECTION,
    
    # Vision settings
    "VISION_REQUEST_TIMEOUT": _settings.VISION_REQUEST_TIMEOUT,
    "VISION_MAX_RETRIES": _settings.VISION_MAX_RETRIES,
    "VISION_INFERENCE_LIMIT": _settings.VISION_INFERENCE_LIMIT,
    "VISION_TIMEOUT": _settings.VISION_TIMEOUT,
    "VISION_PRODUCTION_TOKENS": _settings.VISION_PRODUCTION_TOKENS,
    "VISION_BATCH_SIZE": _settings.VISION_BATCH_SIZE,
    "VISION_PARALLEL_PROCESSING": _settings.VISION_PARALLEL_PROCESSING,
    
    # Validation settings
    "validation_timeout": _settings.VISION_VALIDATION_TIMEOUT,
    "VALIDATION_MAX_RETRIES": _settings.VALIDATION_MAX_RETRIES,
    "enable_4x_validation": _settings.ENABLE_4X_VALIDATION,
    "max_concurrent_validations": _settings.MAX_CONCURRENT_VALIDATIONS,
    
    # Element detection settings
    "element_detection_confidence": _settings.ELEMENT_DETECTION_CONFIDENCE,
    "element_detection_timeout": _settings.ELEMENT_DETECTION_TIMEOUT,
    
    # Grid and visual settings
    "grid_detection_confidence": _settings.GRID_DETECTION_CONFIDENCE,
    "max_visual_elements": _settings.MAX_VISUAL_ELEMENTS,
    
    # Visual Grid Detection Settings (NEW)
    "visual_grid_detection_enabled": _settings.VISUAL_GRID_DETECTION_ENABLED,
    "grid_detection_min_confidence": _settings.GRID_DETECTION_MIN_CONFIDENCE,
    "grid_line_detection_threshold": _settings.GRID_LINE_DETECTION_THRESHOLD,
    "grid_min_line_length": _settings.GRID_MIN_LINE_LENGTH,
    "grid_line_gap_tolerance": _settings.GRID_LINE_GAP_TOLERANCE,
    "grid_line_thickness_min": _settings.GRID_LINE_THICKNESS_MIN,
    "grid_line_thickness_max": _settings.GRID_LINE_THICKNESS_MAX,
    "grid_detection_dpi_multiplier": _settings.GRID_DETECTION_DPI_MULTIPLIER,
    "grid_line_angle_tolerance": _settings.GRID_LINE_ANGLE_TOLERANCE,
    "grid_line_grouping_distance": _settings.GRID_LINE_GROUPING_DISTANCE,
    "grid_detection_canny_low": _settings.GRID_DETECTION_CANNY_LOW,
    "grid_detection_canny_high": _settings.GRID_DETECTION_CANNY_HIGH,
    "grid_detection_hough_threshold": _settings.GRID_DETECTION_HOUGH_THRESHOLD,
    "grid_max_labels_x": _settings.GRID_MAX_LABELS_X,
    "grid_max_labels_y": _settings.GRID_MAX_LABELS_Y,
    
    # Storage settings
    "storage_download_timeout": _settings.STORAGE_DOWNLOAD_TIMEOUT,
    "storage_upload_timeout": _settings.STORAGE_UPLOAD_TIMEOUT,
    "storage_batch_timeout": _settings.STORAGE_BATCH_TIMEOUT,
    "storage_max_concurrent_downloads": _settings.STORAGE_MAX_CONCURRENT_DOWNLOADS,
    "storage_max_concurrent_uploads": _settings.STORAGE_MAX_CONCURRENT_UPLOADS,
    "storage_retry_attempts": _settings.STORAGE_RETRY_ATTEMPTS,
    "storage_retry_delay": _settings.STORAGE_RETRY_DELAY,
    
    # General timeouts
    "request_timeout": _settings.REQUEST_TIMEOUT,
    "upload_timeout": _settings.UPLOAD_TIMEOUT,
    "job_timeout_seconds": _settings.JOB_TIMEOUT_SECONDS,
    "warning_threshold": _settings.WARNING_THRESHOLD,
    
    # SSE settings
    "sse_enabled": _settings.ENABLE_SSE,
    "sse_keepalive_interval": _settings.SSE_KEEPALIVE_INTERVAL,
    "sse_client_timeout": _settings.SSE_CLIENT_TIMEOUT,
    "sse_max_connections_per_document": _settings.SSE_MAX_CONNECTIONS_PER_DOCUMENT,
    "sse_retry_interval": _settings.SSE_RETRY_INTERVAL,
    "sse_event_queue_size": _settings.SSE_EVENT_QUEUE_SIZE,
    "sse_enable_compression": _settings.SSE_ENABLE_COMPRESSION,
    "sse_max_queue_size_per_connection": _settings.SSE_MAX_QUEUE_SIZE_PER_CONNECTION,
    "sse_dropped_event_threshold": _settings.SSE_DROPPED_EVENT_THRESHOLD,
    "sse_slow_consumer_threshold": _settings.SSE_SLOW_CONSUMER_THRESHOLD,
    "sse_event_history_cleanup_size": _settings.SSE_EVENT_HISTORY_CLEANUP_SIZE,
    
    # Processing settings
    "status_update_lock_timeout": _settings.STATUS_UPDATE_LOCK_TIMEOUT,
    "processing_health_check_interval": _settings.PROCESSING_HEALTH_CHECK_INTERVAL,
    "processing_check_interval": _settings.PROCESSING_CHECK_INTERVAL,
}

# Export settings for convenience
settings = get_settings()

"""
IMPORTANT NOTES FOR DATA LOADER IMPLEMENTATION:

1. The load_all_thumbnails method should:
   - Check metadata for page_count
   - If no metadata, probe up to THUMBNAIL_PROBE_MAX_PAGES
   - Load ALL thumbnails found, not just MIN_THUMBNAILS_TO_LOAD
   - Use batch loading with BATCH_SIZE for performance
   - Apply THUMBNAIL_BATCH_DELAY between batches

2. Page selection should:
   - Analyze ALL loaded thumbnails
   - Select relevant pages based on query
   - No artificial limits on selection
   - Use UNLIMITED_PAGE_LOADING flag

3. High-res loading should:
   - Only load selected pages
   - Use aggressive caching
   - Apply proper timeouts

4. SSE implementation should:
   - Send events during processing for real-time updates
   - Use keepalive events to maintain connection
   - Handle multiple connections per document
   - Clean up on disconnection

5. Visual Grid Detection should:
   - Use OpenCV to detect actual grid lines
   - Apply Hough transform with configured thresholds
   - Group nearby lines within GRID_LINE_GROUPING_DISTANCE
   - Maintain confidence scores for detected grids
   - Fall back to text-based detection when visual fails
"""