# app/main.py - COMPLETE FIXED VERSION

"""
Main FastAPI application entry point.
Initializes all services and routes for the Blueprint Analysis System.
"""

import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from app.core.config import get_settings
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService
from app.services.session_service import SessionService
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router
from app.api.routes.note_routes import note_router
from app.api.routes.collaboration_routes import collaboration_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.
    Initialize all services and clean up resources.
    """
    logger.info("🚀 Starting Blueprint Analysis System...")
    
    try:
        # Initialize Storage Service
        logger.info("Initializing Storage Service...")
        storage_service = StorageService(settings)
        await storage_service.initialize_containers()
        app.state.storage_service = storage_service
        logger.info("✅ Storage Service ready")
        
        # Initialize PDF Service
        logger.info("Initializing PDF Service...")
        pdf_service = PDFService(settings)
        app.state.pdf_service = pdf_service
        logger.info("✅ PDF Service ready")
        
        # Initialize Session Service
        logger.info("Initializing Session Service...")
        session_service = SessionService(settings)
        session_service.set_storage_service(storage_service)
        await session_service.start()
        app.state.session_service = session_service
        logger.info("✅ Session Service ready")
        
        # Initialize AI Service (placeholder - you'll add this when ready)
        app.state.ai_service = None
        logger.info("⚠️ AI Service not configured (add when ready)")
        
        logger.info("✅ All services initialized successfully!")
        logger.info(f"📍 API running at: http://localhost:{settings.PORT}")
        logger.info(f"📚 API docs available at: http://localhost:{settings.PORT}/docs")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("🛑 Shutting down Blueprint Analysis System...")
        
        # Stop session service
        if hasattr(app.state, 'session_service') and app.state.session_service:
            await app.state.session_service.stop()
        
        logger.info("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Blueprint Analysis API",
    description="AI-powered construction blueprint analysis system with collaboration features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Document-ID", "X-Page-Count", "X-Page-Number", "X-Session-ID"]
)


# --- Error Handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error['loc'])
        errors.append({
            "field": field,
            "message": error['msg'],
            "type": error['type']
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "type": type(exc).__name__
        }
    )


# --- Root Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Blueprint Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {}
    }
    
    # Check Storage Service
    try:
        if hasattr(request.app.state, 'storage_service') and request.app.state.storage_service:
            health_status["services"]["storage"] = {
                "status": "healthy",
                "info": request.app.state.storage_service.get_service_info()
            }
        else:
            health_status["services"]["storage"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["storage"] = {"status": "error", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check PDF Service
    try:
        if hasattr(request.app.state, 'pdf_service') and request.app.state.pdf_service:
            health_status["services"]["pdf"] = {"status": "healthy"}
        else:
            health_status["services"]["pdf"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["pdf"] = {"status": "error", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Session Service
    try:
        if hasattr(request.app.state, 'session_service') and request.app.state.session_service:
            health_status["services"]["session"] = {
                "status": "healthy",
                "statistics": request.app.state.session_service.get_service_statistics()
            }
        else:
            health_status["services"]["session"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["session"] = {"status": "error", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check AI Service
    health_status["services"]["ai"] = {
        "status": "not_configured",
        "message": "AI service will be added when ready"
    }
    
    # Set appropriate status code
    status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        content=health_status,
        status_code=status_code
    )

@app.get("/api/v1", tags=["API Info"])
async def api_info():
    """API version and capability information."""
    return {
        "version": "1.0.0",
        "capabilities": {
            "pdf_upload": True,
            "pdf_processing": True,
            "text_extraction": True,
            "image_generation": True,
            "thumbnail_generation": True,
            "grid_detection": True,
            "table_extraction": True,
            "ai_analysis": False,  # Will be True when AI service is added
            "collaboration": True,
            "annotations": True,
            "notes": True
        },
        "limits": {
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "max_pages": settings.PDF_MAX_PAGES,
            "supported_formats": ["pdf"]
        },
        "endpoints": {
            "upload": "/api/v1/documents/upload",
            "status": "/api/v1/documents/{document_id}/status",
            "download": "/api/v1/documents/{document_id}/download",
            "metadata": "/api/v1/documents/{document_id}",
            "images": "/api/v1/documents/{document_id}/images",
            "chat": "/api/v1/documents/{document_id}/chat"
        }
    }


# --- Register Routers ---

# Blueprint routes (upload, chat, etc.)
app.include_router(
    blueprint_router,
    prefix="/api/v1",
    tags=["Blueprints"]
)

# Document routes (metadata, images, etc.)
app.include_router(
    document_router,
    prefix="/api/v1",
    tags=["Documents"]
)

# Annotation routes
app.include_router(
    annotation_router,
    prefix="/api/v1",
    tags=["Annotations"]
)

# Note routes
app.include_router(
    note_router,
    prefix="/api/v1",
    tags=["Notes"]
)

# Collaboration routes
app.include_router(
    collaboration_router,
    prefix="/api/v1",
    tags=["Collaboration"]
)


# --- Request Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
    
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"📥 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response status
    status_emoji = "✅" if response.status_code < 400 else "❌"
    logger.info(f"{status_emoji} {request.method} {request.url.path} - {response.status_code}")
    
    return response


# --- Utility Endpoints ---

@app.post("/api/v1/test-upload", tags=["Testing"])
async def test_upload_endpoint(request: Request):
    """Test endpoint to verify file upload capability."""
    return {
        "status": "ready",
        "storage_service": hasattr(request.app.state, 'storage_service') and request.app.state.storage_service is not None,
        "pdf_service": hasattr(request.app.state, 'pdf_service') and request.app.state.pdf_service is not None,
        "session_service": hasattr(request.app.state, 'session_service') and request.app.state.session_service is not None,
        "message": "Ready to accept file uploads"
    }


# --- Main Entry Point ---

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    # Print startup banner
    print("\n" + "="*60)
    print("🏗️  Blueprint Analysis System")
    print("="*60)
    print(f"📅 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"🌐 Environment: {settings.ENVIRONMENT}")
    print(f"🔧 Debug mode: {settings.DEBUG}")
    print("="*60 + "\n")
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG
    )