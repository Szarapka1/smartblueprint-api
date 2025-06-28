# main.py - PERMISSIVE TEST CONFIGURATION WITH PRODUCTION-GRADE CORE

import os
import logging
import uvicorn
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import settings first
from app.core.config import get_settings

# Import services
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService
from app.services.ai_service import ProfessionalBlueprintAI
from app.services.session_service import SessionService

# Import API routers
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router
from app.api.routes.note_routes import note_router

# --- Configure Logging ---
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level for test environment
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBlueprintAPI")

# Load settings
settings = get_settings()

# --- Application Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown with PERMISSIVE error handling for testing.
    """
    logger.info("="*60)
    logger.info("🚀 Initializing Smart Blueprint API (TEST MODE)...")
    logger.info("⚠️  WARNING: Running in UNSAFE test configuration!")
    logger.info("="*60)
    
    # Track initialization status
    initialization_status = {
        "storage": {"status": "not_started", "error": None},
        "pdf": {"status": "not_started", "error": None},
        "ai": {"status": "not_started", "error": None},
        "session": {"status": "not_started", "error": None}
    }
    
    try:
        # 1. Initialize Storage Service
        initialization_status["storage"]["status"] = "initializing"
        try:
            if settings.AZURE_STORAGE_CONNECTION_STRING:
                storage_service = StorageService(settings)
                await storage_service.verify_connection()
                app.state.storage_service = storage_service
                initialization_status["storage"]["status"] = "success"
                logger.info("✅ Storage Service initialized successfully")
            else:
                app.state.storage_service = None
                initialization_status["storage"]["status"] = "disabled"
                logger.warning("⚠️ Storage Service disabled - no connection string")
        except Exception as e:
            initialization_status["storage"]["status"] = "failed"
            initialization_status["storage"]["error"] = str(e)
            app.state.storage_service = None
            logger.error(f"❌ Storage Service failed: {e}")
            logger.error(traceback.format_exc())
        
        # 2. Initialize PDF Service
        initialization_status["pdf"]["status"] = "initializing"
        try:
            if app.state.storage_service:
                pdf_service = PDFService(settings)
                app.state.pdf_service = pdf_service
                initialization_status["pdf"]["status"] = "success"
                logger.info("✅ PDF Service initialized successfully")
            else:
                app.state.pdf_service = None
                initialization_status["pdf"]["status"] = "disabled"
                logger.warning("⚠️ PDF Service disabled - requires storage service")
        except Exception as e:
            initialization_status["pdf"]["status"] = "failed"
            initialization_status["pdf"]["error"] = str(e)
            app.state.pdf_service = None
            logger.error(f"❌ PDF Service failed: {e}")
            logger.error(traceback.format_exc())
        
        # 3. Initialize AI Service
        initialization_status["ai"]["status"] = "initializing"
        try:
            if settings.OPENAI_API_KEY:
                ai_service = ProfessionalBlueprintAI(settings)
                app.state.ai_service = ai_service
                initialization_status["ai"]["status"] = "success"
                logger.info("✅ AI Service initialized successfully")
            else:
                app.state.ai_service = None
                initialization_status["ai"]["status"] = "disabled"
                logger.warning("⚠️ AI Service disabled - no API key")
        except Exception as e:
            initialization_status["ai"]["status"] = "failed"
            initialization_status["ai"]["error"] = str(e)
            app.state.ai_service = None
            logger.error(f"❌ AI Service failed: {e}")
            logger.error(traceback.format_exc())
        
        # 4. Initialize Session Service
        initialization_status["session"]["status"] = "initializing"
        try:
            session_service = SessionService(settings, app.state.storage_service)
            await session_service.start_background_cleanup()
            app.state.session_service = session_service
            initialization_status["session"]["status"] = "success"
            logger.info("✅ Session Service initialized successfully")
        except Exception as e:
            initialization_status["session"]["status"] = "failed"
            initialization_status["session"]["error"] = str(e)
            app.state.session_service = None
            logger.error(f"❌ Session Service failed: {e}")
            logger.error(traceback.format_exc())
        
        # 5. Store initialization status
        app.state.initialization_status = initialization_status
        
        # 6. Log final status
        logger.info("="*60)
        logger.info("📊 Service Initialization Summary:")
        for service, status in initialization_status.items():
            emoji = "✅" if status["status"] == "success" else "❌" if status["status"] == "failed" else "⚠️"
            logger.info(f"   {emoji} {service}: {status['status']}")
            if status["error"]:
                logger.info(f"      Error: {status['error']}")
        
        logger.info("="*60)
        logger.info("🎉 API is ready! (TEST MODE)")
        logger.info(f"📚 Interactive docs available at:")
        logger.info(f"   - http://localhost:{settings.PORT}/docs")
        logger.info(f"   - http://localhost:{settings.PORT}/redoc")
        logger.info("="*60)

        yield  # Application is now running

        # --- Shutdown Logic ---
        logger.info("="*60)
        logger.info("🛑 Shutting down API...")
        
        # Gracefully shutdown services
        if hasattr(app.state, 'session_service') and app.state.session_service:
            try:
                await app.state.session_service.shutdown()
                logger.info("✅ Session Service shutdown complete")
            except Exception as e:
                logger.error(f"❌ Session Service shutdown error: {e}")
                logger.error(traceback.format_exc())
        
        if hasattr(app.state, 'ai_service') and app.state.ai_service:
            try:
                if hasattr(app.state.ai_service, '__aexit__'):
                    await app.state.ai_service.__aexit__(None, None, None)
                logger.info("✅ AI Service shutdown complete")
            except Exception as e:
                logger.error(f"❌ AI Service shutdown error: {e}")
                logger.error(traceback.format_exc())
        
        if hasattr(app.state, 'storage_service') and app.state.storage_service:
            try:
                if hasattr(app.state.storage_service, '__aexit__'):
                    await app.state.storage_service.__aexit__(None, None, None)
                logger.info("✅ Storage Service shutdown complete")
            except Exception as e:
                logger.error(f"❌ Storage Service shutdown error: {e}")
                logger.error(traceback.format_exc())
        
        logger.info("✅ Shutdown complete")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"❌ Critical error during application lifecycle: {e}")
        logger.error(traceback.format_exc())
        # In test mode, we continue anyway
        yield

# --- Create FastAPI Application ---

app = FastAPI(
    title="Smart Blueprint API (TEST MODE)",
    description="""
    An intelligent API for analyzing construction blueprints with AI-powered insights.
    
    ⚠️ **WARNING**: This instance is running in TEST MODE with:
    - All CORS origins allowed
    - Detailed error messages exposed
    - Full stack traces in responses
    - No authentication required
    
    DO NOT USE IN PRODUCTION!
    """,
    version=f"{settings.VERSION}-TEST",
    lifespan=lifespan,
    docs_url="/docs",  # Always enabled in test mode
    redoc_url="/redoc",  # Always enabled in test mode
    debug=True  # Always debug in test mode
)

# --- PERMISSIVE CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow ALL methods
    allow_headers=["*"],  # Allow ALL headers
    expose_headers=["*"]  # Expose ALL headers
)

logger.warning("⚠️ CORS: Allowing ALL origins, methods, and headers - TEST MODE ONLY!")

# --- VERBOSE Global Exception Handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    VERBOSE exception handler for TEST MODE - exposes everything for debugging.
    """
    # Get full traceback
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    
    # Log the full error
    logger.error(f"Unhandled exception for {request.method} {request.url}:")
    logger.error("".join(tb_lines))
    
    # Get request details for debugging
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else "No body"
    except:
        body_str = "Could not read body"
    
    # Return EVERYTHING for debugging
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "exception_args": exc.args if hasattr(exc, 'args') else None,
            "traceback": tb_lines,
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "path_params": request.path_params,
                "query_params": dict(request.query_params),
                "body": body_str
            },
            "file_location": {
                "file": tb_lines[-2].split('"')[1] if len(tb_lines) > 1 and '"' in tb_lines[-2] else "unknown",
                "line": tb_lines[-2].split("line ")[1].split(",")[0] if len(tb_lines) > 1 and "line " in tb_lines[-2] else "unknown"
            },
            "initialization_status": getattr(app.state, 'initialization_status', {})
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with VERBOSE formatting for testing."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail,
            "path": str(request.url),
            "method": request.method,
            "headers": dict(request.headers) if exc.status_code >= 400 else None
        }
    )

# --- Include API Routers ---

app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint Analysis"])
app.include_router(document_router, prefix="/api/v1", tags=["Document Management"])
app.include_router(annotation_router, prefix="/api/v1", tags=["Annotations"])
app.include_router(note_router, prefix="/api/v1", tags=["Notes"])

# --- Root Endpoints ---

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with DETAILED system information for testing."""
    return {
        "service": "Smart Blueprint API",
        "mode": "TEST/DEVELOPMENT - UNSAFE",
        "version": settings.VERSION,
        "status": "operational",
        "warning": "This API is running in TEST MODE with permissive settings!",
        "documentation": {
            "interactive": f"http://localhost:{settings.PORT}/docs",
            "redoc": f"http://localhost:{settings.PORT}/redoc"
        },
        "initialization_status": getattr(app.state, 'initialization_status', {}),
        "features": {
            "storage": settings.is_feature_enabled("storage"),
            "ai_analysis": settings.is_feature_enabled("ai"),
            "pdf_processing": settings.is_feature_enabled("pdf_processing"),
            "admin_access": settings.is_feature_enabled("admin"),
            "notes": settings.is_feature_enabled("notes"),
            "highlighting": settings.is_feature_enabled("highlighting"),
            "trade_coordination": settings.is_feature_enabled("trade_coordination")
        },
        "services_available": {
            "storage": hasattr(app.state, 'storage_service') and app.state.storage_service is not None,
            "pdf": hasattr(app.state, 'pdf_service') and app.state.pdf_service is not None,
            "ai": hasattr(app.state, 'ai_service') and app.state.ai_service is not None,
            "session": hasattr(app.state, 'session_service') and app.state.session_service is not None
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """DETAILED health check endpoint for testing."""
    def check_service(service_name: str) -> dict:
        """Check if a service is available and get detailed info."""
        service = getattr(app.state, service_name, None)
        if service is None:
            return {
                "status": "unavailable",
                "message": "Service not initialized",
                "initialization_error": getattr(app.state, 'initialization_status', {}).get(
                    service_name.replace('_service', ''), {}
                ).get('error')
            }
        
        try:
            # Try to get detailed service info
            result = {"status": "healthy", "details": {}}
            
            if hasattr(service, 'get_connection_info'):
                result["details"]["connection_info"] = service.get_connection_info()
            
            if hasattr(service, 'get_service_statistics'):
                result["details"]["statistics"] = service.get_service_statistics()
            
            if hasattr(service, 'get_processing_stats'):
                result["details"]["processing_stats"] = service.get_processing_stats()
            
            if hasattr(service, 'get_professional_capabilities'):
                result["details"]["capabilities"] = service.get_professional_capabilities()
            
            if hasattr(service, 'is_running'):
                result["details"]["is_running"] = service.is_running()
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc().splitlines()[-5:]  # Last 5 lines
            }
    
    health_status = {
        "timestamp": logging.Formatter().formatTime(logging.LogRecord(
            name="", level=0, pathname="", lineno=0,
            msg="", args=(), exc_info=None
        )),
        "overall_status": "healthy",
        "mode": "TEST/DEVELOPMENT",
        "version": settings.VERSION,
        "initialization_status": getattr(app.state, 'initialization_status', {}),
        "services": {
            "storage": check_service('storage_service'),
            "ai": check_service('ai_service'),
            "pdf": check_service('pdf_service'),
            "session": check_service('session_service')
        },
        "environment": {
            "debug": settings.DEBUG,
            "cors_origins": settings.CORS_ORIGINS,
            "python_version": os.sys.version
        }
    }
    
    # Determine overall status
    service_statuses = [s.get("status", "unknown") for s in health_status["services"].values()]
    if "error" in service_statuses:
        health_status["overall_status"] = "degraded"
    elif all(status == "unavailable" for status in service_statuses):
        health_status["overall_status"] = "critical"
    
    return health_status

@app.get("/config", tags=["System"])
async def get_configuration():
    """Get FULL configuration details for testing."""
    return {
        "mode": "TEST/DEVELOPMENT - UNSAFE",
        "features": {
            "storage": settings.is_feature_enabled("storage"),
            "ai_analysis": settings.is_feature_enabled("ai"),
            "pdf_processing": settings.is_feature_enabled("pdf_processing"),
            "admin_access": settings.is_feature_enabled("admin"),
            "notes": settings.is_feature_enabled("notes"),
            "sessions": settings.is_feature_enabled("sessions"),
            "highlighting": settings.is_feature_enabled("highlighting"),
            "private_notes": settings.is_feature_enabled("private_notes"),
            "note_publishing": settings.is_feature_enabled("note_publishing"),
            "trade_coordination": settings.is_feature_enabled("trade_coordination")
        },
        "limits": {
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "max_notes_per_document": settings.MAX_NOTES_PER_DOCUMENT,
            "max_sessions": settings.MAX_SESSIONS_IN_MEMORY,
            "pdf_max_pages": settings.PDF_MAX_PAGES,
            "ai_max_pages": settings.AI_MAX_PAGES,
            "max_note_length": settings.MAX_NOTE_LENGTH,
            "max_total_note_chars": settings.MAX_TOTAL_NOTE_CHARS,
            "max_visual_elements": settings.MAX_VISUAL_ELEMENTS,
            "session_cleanup_interval": settings.SESSION_CLEANUP_INTERVAL_SECONDS,
            "session_expiry_hours": settings.SESSION_CLEANUP_HOURS
        },
        "environment": {
            "debug_mode": True,  # Always true in test mode
            "cors_origins": ["*"],  # Always permissive in test mode
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "host": settings.HOST,
            "port": settings.PORT
        },
        "storage": {
            "main_container": settings.AZURE_CONTAINER_NAME,
            "cache_container": settings.AZURE_CACHE_CONTAINER_NAME,
            "connection_string_length": len(settings.AZURE_STORAGE_CONNECTION_STRING) if settings.AZURE_STORAGE_CONNECTION_STRING else 0
        },
        "ai": {
            "model": settings.OPENAI_MODEL,
            "max_tokens": settings.OPENAI_MAX_TOKENS,
            "temperature": settings.OPENAI_TEMPERATURE,
            "api_key_length": len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 0
        },
        "pdf": {
            "preview_resolution": settings.PDF_PREVIEW_RESOLUTION,
            "high_resolution": settings.PDF_HIGH_RESOLUTION,
            "ai_dpi": settings.PDF_AI_DPI,
            "thumbnail_dpi": settings.PDF_THUMBNAIL_DPI,
            "batch_size": settings.PROCESSING_BATCH_SIZE
        }
    }

@app.get("/debug/error-test", tags=["Debug"])
async def test_error_handling():
    """Test endpoint to verify verbose error handling is working."""
    # This will trigger the global exception handler
    raise ValueError("This is a test error to verify verbose error handling is working correctly!")

# --- Application Entry Point ---

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 Starting Smart Blueprint API in TEST MODE")
    logger.info("⚠️  WARNING: This configuration is UNSAFE for production!")
    logger.info("="*60)
    logger.info(f"🔧 Debug mode: ON")
    logger.info(f"🌐 CORS: Allowing ALL origins")
    logger.info(f"📝 Error details: FULLY EXPOSED")
    logger.info(f"🔓 Authentication: DISABLED")
    logger.info("="*60)
    
    # Run the application with reload for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,  # Auto-reload on code changes
        log_level="debug",  # Verbose logging
        access_log=True  # Log all requests
    )
