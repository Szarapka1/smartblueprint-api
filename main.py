# main.py - FIXED APPLICATION WITH PROPER ERROR HANDLING

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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBlueprintAPI")

# Load settings
settings = get_settings()

# --- Application Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown with proper error handling.
    """
    logger.info("="*60)
    logger.info("🚀 Initializing Smart Blueprint API...")
    logger.info("="*60)
    
    # Track which services were successfully initialized
    initialized_services = []
    
    try:
        # 1. Initialize Storage Service (required for most functionality)
        if settings.is_feature_enabled("storage"):
            try:
                storage_service = StorageService(settings)
                await storage_service.verify_connection()
                app.state.storage_service = storage_service
                initialized_services.append("storage")
                logger.info("✅ Storage Service initialized and verified")
            except Exception as e:
                logger.error(f"❌ Storage Service failed to initialize: {e}")
                # Continue without storage - some features will be limited
                app.state.storage_service = None
        else:
            logger.warning("⚠️ Storage Service disabled (missing connection string)")
            app.state.storage_service = None
        
        # 2. Initialize PDF Service (depends on storage)
        if settings.is_feature_enabled("pdf_processing") and app.state.storage_service:
            try:
                pdf_service = PDFService(settings)
                app.state.pdf_service = pdf_service
                initialized_services.append("pdf")
                logger.info("✅ PDF Service initialized")
            except Exception as e:
                logger.error(f"❌ PDF Service failed to initialize: {e}")
                app.state.pdf_service = None
        else:
            logger.warning("⚠️ PDF Service disabled (storage not available)")
            app.state.pdf_service = None
        
        # 3. Initialize AI Service
        if settings.is_feature_enabled("ai"):
            try:
                ai_service = ProfessionalBlueprintAI(settings)
                app.state.ai_service = ai_service
                initialized_services.append("ai")
                logger.info("✅ AI Service initialized")
            except Exception as e:
                logger.error(f"❌ AI Service failed to initialize: {e}")
                app.state.ai_service = None
        else:
            logger.warning("⚠️ AI Service disabled (missing API key)")
            app.state.ai_service = None
        
        # 4. Initialize Session Service (always try to initialize)
        try:
            # Pass storage service even if None - session service should handle gracefully
            session_service = SessionService(settings, app.state.storage_service)
            await session_service.start_background_cleanup()
            app.state.session_service = session_service
            initialized_services.append("session")
            logger.info("✅ Session Service initialized with background cleanup")
        except Exception as e:
            logger.error(f"❌ Session Service failed to initialize: {e}")
            logger.error(traceback.format_exc())
            app.state.session_service = None
        
        # 5. Log final initialization status
        logger.info("="*60)
        logger.info("📊 Service Initialization Summary:")
        logger.info(f"   ✅ Successfully initialized: {', '.join(initialized_services) or 'None'}")
        
        if len(initialized_services) >= 2:
            logger.info("🎉 API is ready for operations!")
        else:
            logger.warning("⚠️ API is running with limited functionality")
        
        if settings.DEBUG:
            logger.info(f"📚 API Documentation: http://localhost:{settings.PORT}/docs")
        
        logger.info("="*60)

        yield  # Application is now running

        # --- Shutdown Logic ---
        logger.info("="*60)
        logger.info("🛑 Shutting down API...")
        
        # Gracefully shutdown services
        shutdown_tasks = []
        
        if hasattr(app.state, 'session_service') and app.state.session_service:
            try:
                await app.state.session_service.shutdown()
                logger.info("✅ Session Service shutdown complete")
            except Exception as e:
                logger.error(f"❌ Session Service shutdown error: {e}")
        
        if hasattr(app.state, 'ai_service') and app.state.ai_service:
            try:
                if hasattr(app.state.ai_service, '__aexit__'):
                    await app.state.ai_service.__aexit__(None, None, None)
                logger.info("✅ AI Service shutdown complete")
            except Exception as e:
                logger.error(f"❌ AI Service shutdown error: {e}")
        
        if hasattr(app.state, 'storage_service') and app.state.storage_service:
            try:
                if hasattr(app.state.storage_service, '__aexit__'):
                    await app.state.storage_service.__aexit__(None, None, None)
                logger.info("✅ Storage Service shutdown complete")
            except Exception as e:
                logger.error(f"❌ Storage Service shutdown error: {e}")
        
        logger.info("✅ Shutdown complete")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"❌ Critical error during application lifecycle: {e}")
        logger.error(traceback.format_exc())
        raise

# --- Create FastAPI Application ---

app = FastAPI(
    title="Smart Blueprint API",
    description="An intelligent API for analyzing construction blueprints with AI-powered insights.",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    debug=settings.DEBUG
)

# --- Configure CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if "*" in settings.CORS_ORIGINS:
    logger.warning("⚠️ CORS: Allowing ALL origins - ensure this is intentional!")

# --- Global Exception Handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    Returns appropriate error responses and logs details.
    """
    # Log the error
    logger.error(f"Unhandled exception for {request.method} {request.url}:")
    logger.error(f"Error: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Return appropriate response based on environment
    if settings.DEBUG:
        # In debug mode, return detailed error information
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "type": type(exc).__name__,
                "path": str(request.url),
                "debug_info": {
                    "traceback": traceback.format_exc().splitlines()[-5:],  # Last 5 lines
                    "request_method": request.method,
                    "timestamp": str(traceback.extract_tb(exc.__traceback__)[-1])
                }
            }
        )
    else:
        # In production, return generic error
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": str(traceback.extract_tb(exc.__traceback__)[-1])
            }
        )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with consistent formatting."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail,
            "path": str(request.url)
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
    """Root endpoint with system information."""
    return {
        "service": "Smart Blueprint API",
        "version": settings.VERSION,
        "status": "operational",
        "environment": "development" if settings.DEBUG else "production",
        "documentation": f"/docs" if settings.DEBUG else "Not available in production",
        "features": {
            "storage": settings.is_feature_enabled("storage"),
            "ai_analysis": settings.is_feature_enabled("ai"),
            "pdf_processing": settings.is_feature_enabled("pdf_processing"),
            "admin_access": settings.is_feature_enabled("admin"),
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Comprehensive health check endpoint."""
    def check_service(service_name: str) -> dict:
        """Check if a service is available and healthy."""
        service = getattr(app.state, service_name, None)
        if service is None:
            return {"status": "unavailable", "message": "Service not initialized"}
        
        try:
            # Try to get service info to verify it's working
            if hasattr(service, 'get_connection_info'):
                info = service.get_connection_info()
                return {"status": "healthy", "info": info}
            elif hasattr(service, 'get_service_statistics'):
                stats = service.get_service_statistics()
                return {"status": "healthy", "stats": stats}
            elif hasattr(service, 'is_running'):
                running = service.is_running()
                return {"status": "healthy" if running else "stopped"}
            else:
                return {"status": "healthy", "message": "Service initialized"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    health_status = {
        "timestamp": logger.handlers[0].formatter.formatTime(logging.LogRecord(
            name="", level=0, pathname="", lineno=0,
            msg="", args=(), exc_info=None
        )) if logger.handlers else "unknown",
        "overall_status": "healthy",
        "version": settings.VERSION,
        "services": {
            "storage": check_service('storage_service'),
            "ai": check_service('ai_service'),
            "pdf": check_service('pdf_service'),
            "session": check_service('session_service')
        }
    }
    
    # Determine overall status
    service_statuses = [s.get("status", "unknown") for s in health_status["services"].values()]
    if "error" in service_statuses:
        health_status["overall_status"] = "degraded"
    elif all(status in ["unavailable", "not_initialized"] for status in service_statuses):
        health_status["overall_status"] = "critical"
    
    return health_status

@app.get("/config", tags=["System"])
async def get_configuration():
    """Get current configuration status (safe for production)."""
    return {
        "features": {
            "storage": settings.is_feature_enabled("storage"),
            "ai_analysis": settings.is_feature_enabled("ai"),
            "pdf_processing": settings.is_feature_enabled("pdf_processing"),
            "admin_access": settings.is_feature_enabled("admin"),
            "notes": settings.is_feature_enabled("notes"),
            "sessions": settings.is_feature_enabled("sessions")
        },
        "limits": {
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "max_notes_per_document": settings.MAX_NOTES_PER_DOCUMENT,
            "max_sessions": settings.MAX_SESSIONS_IN_MEMORY,
            "pdf_max_pages": settings.PDF_MAX_PAGES,
            "ai_max_pages": settings.AI_MAX_PAGES
        },
        "environment": {
            "debug_mode": settings.DEBUG,
            "cors_origins": settings.CORS_ORIGINS,
            "version": settings.VERSION
        }
    }

# --- Application Entry Point ---

if __name__ == "__main__":
    logger.info(f"🚀 Starting Smart Blueprint API on port {settings.PORT}")
    logger.info(f"🔧 Debug mode: {'ON' if settings.DEBUG else 'OFF'}")
    logger.info(f"🌐 CORS origins: {settings.CORS_ORIGINS}")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=settings.DEBUG
    )
