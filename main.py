import datetime
import logging
import uvicorn
import traceback
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# App config and services
from app.core.config import get_settings

# --- Robust service imports with fallbacks ---
try:
    from app.services.storage_service import StorageService
except ImportError as e:
    logging.warning(f"Storage service import failed: {e}")
    StorageService = None

try:
    from app.services.pdf_service import PDFService
except ImportError as e:
    logging.warning(f"PDF service import failed: {e}")
    PDFService = None

try:
    from app.services.ai_service import AIService
except ImportError as e:
    logging.warning(f"AI service import failed: {e}")
    AIService = None

try:
    from app.services.session_service import SessionService
except ImportError as e:
    logging.warning(f"Session service import failed: {e}")
    SessionService = None

# --- Robust route imports with fallbacks ---
try:
    from app.api.routes.blueprint_routes import blueprint_router
except ImportError as e:
    logging.warning(f"Blueprint routes import failed: {e}")
    blueprint_router = None

try:
    from app.api.routes.document_routes import document_router
except ImportError as e:
    logging.warning(f"Document routes import failed: {e}")
    document_router = None

try:
    from app.api.routes.annotation_routes import annotation_router
except ImportError as e:
    logging.warning(f"Annotation routes import failed: {e}")
    annotation_router = None

# --- Init logging & settings ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartBlueprintAPI")

# Safe settings initialization
try:
    settings = get_settings()
    logger.info("✅ Settings loaded successfully")
except Exception as e:
    logger.error(f"❌ Settings failed to load: {e}")
    # Create minimal fallback settings
    class FallbackSettings:
        PROJECT_NAME = "Smart Blueprint Chat API"
        DEBUG = False
        HOST = "0.0.0.0"
        PORT = 8000
        CORS_ORIGINS = ["*"]
        AZURE_STORAGE_CONNECTION_STRING = None
        AZURE_CONTAINER_NAME = "blueprints"
        AZURE_CACHE_CONTAINER_NAME = "blueprints-cache"
        OPENAI_API_KEY = None
    
    settings = FallbackSettings()
    logger.info("⚠️ Using fallback settings")

# --- Service initialization helper ---
async def init_service(service_class, service_name: str, required_config: list = None):
    """Safely initialize a service with error handling"""
    if service_class is None:
        logger.warning(f"⚠️ {service_name} class not available")
        return None
    
    # Check required configuration
    if required_config:
        missing_config = []
        for config_key in required_config:
            if not hasattr(settings, config_key) or getattr(settings, config_key) is None:
                missing_config.append(config_key)
        
        if missing_config:
            logger.warning(f"⚠️ {service_name} disabled - missing config: {missing_config}")
            return None
    
    try:
        service = service_class(settings)
        logger.info(f"✅ {service_name} initialized")
        return service
    except Exception as e:
        logger.error(f"❌ {service_name} initialization failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None

# --- Storage verification helper ---
async def verify_storage_connection(storage_service):
    """Safely verify storage connection"""
    if not storage_service:
        return False
    
    try:
        await storage_service.verify_connection()
        logger.info("✅ Storage connection verified")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Storage verification failed: {e}")
        return False

# --- Health check helper ---
async def verify_system_health(app: FastAPI):
    """Comprehensive system health check"""
    health_status = {
        "storage": False,
        "ai": False,
        "pdf": False,
        "session": False
    }
    
    # Check storage health
    if app.state.storage_service:
        try:
            main_blobs = await app.state.storage_service.list_blobs(settings.AZURE_CONTAINER_NAME)
            cache_blobs = await app.state.storage_service.list_blobs(settings.AZURE_CACHE_CONTAINER_NAME)
            logger.info(f"✅ Blob containers: {len(main_blobs)} main / {len(cache_blobs)} cache")
            health_status["storage"] = True
        except Exception as e:
            logger.warning(f"⚠️ Storage health check failed: {e}")
    
    # Check other services
    health_status["ai"] = app.state.ai_service is not None
    health_status["pdf"] = app.state.pdf_service is not None
    health_status["session"] = app.state.session_service is not None
    
    logger.info(f"📊 System health: {health_status}")
    return health_status

# --- Lifecycle hooks ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Smart Blueprint API...")
    logger.info(f"🔧 Environment: {'Development' if settings.DEBUG else 'Production'}")
    
    # Initialize services with robust error handling
    app.state.storage_service = await init_service(
        StorageService, 
        "Storage Service", 
        ["AZURE_STORAGE_CONNECTION_STRING"]
    )
    
    # Verify storage connection if available
    if app.state.storage_service:
        storage_ok = await verify_storage_connection(app.state.storage_service)
        if not storage_ok:
            logger.warning("⚠️ Storage service disabled due to connection issues")
            app.state.storage_service = None
    
    app.state.pdf_service = await init_service(PDFService, "PDF Service")
    
    app.state.ai_service = await init_service(
        AIService, 
        "AI Service", 
        ["OPENAI_API_KEY"]
    )
    
    app.state.session_service = await init_service(SessionService, "Session Service")
    
    # Perform health check
    try:
        health_status = await verify_system_health(app)
        active_services = sum(health_status.values())
        logger.info(f"✅ Startup complete: {active_services}/4 services active")
    except Exception as e:
        logger.warning(f"⚠️ Health check failed: {e}")
    
    # Log startup summary
    services_status = []
    if app.state.storage_service:
        services_status.append("📁 Storage")
    if app.state.ai_service:
        services_status.append("🤖 AI")
    if app.state.pdf_service:
        services_status.append("📄 PDF")
    if app.state.session_service:
        services_status.append("👥 Sessions")
    
    logger.info(f"🎯 Active services: {', '.join(services_status) if services_status else 'Basic API only'}")
    logger.info("🌐 API is ready for requests")
    
    yield
    
    logger.info("🛑 Shutting down gracefully...")

# --- Create FastAPI app ---
app = FastAPI(
    title=getattr(settings, 'PROJECT_NAME', 'Smart Blueprint Chat API'),
    description="Robust collaborative chat, annotation, and analysis for construction blueprints.",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS Middleware ---
cors_origins = getattr(settings, 'CORS_ORIGINS', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Robust exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"🚨 Unhandled error: {exc}")
    logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Don't expose internal errors in production
    error_detail = str(exc) if settings.DEBUG else "Internal server error"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Server Error",
            "detail": error_detail,
            "url": str(request.url),
            "timestamp": str(datetime.datetime.now())  # ✅ FIXED: Use datetime.datetime instead of uvicorn.config.datetime
        }
    )

# --- Register routes safely ---
if blueprint_router:
    app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint Chat"])
    logger.info("✅ Blueprint routes registered")

if document_router:
    app.include_router(document_router, prefix="/api/v1", tags=["Documents"])
    logger.info("✅ Document routes registered")

if annotation_router:
    app.include_router(annotation_router, prefix="/api/v1", tags=["Annotations"])
    logger.info("✅ Annotation routes registered")

# --- Core API endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to Smart Blueprint Chat API",
        "version": "2.1.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Comprehensive health check endpoint"""
    storage_status = "connected" if hasattr(app.state, 'storage_service') and app.state.storage_service else "disabled"
    ai_status = "connected" if hasattr(app.state, 'ai_service') and app.state.ai_service else "disabled"
    pdf_status = "available" if hasattr(app.state, 'pdf_service') and app.state.pdf_service else "disabled"
    session_status = "active" if hasattr(app.state, 'session_service') and app.state.session_service else "disabled"
    
    return {
        "status": "healthy",
        "timestamp": str(datetime.datetime.now()),  # ✅ FIXED: Use datetime.datetime instead of uvicorn.config.datetime
        "services": {
            "storage": storage_status,
            "ai": ai_status,
            "pdf": pdf_status,
            "sessions": session_status
        },
        "version": "2.1.0"
    }

@app.get("/api/v1/system/info", tags=["General"])
async def system_info():
    """System information and capabilities"""
    features = ["Basic API", "Error Handling", "Health Monitoring"]
    
    if hasattr(app.state, 'storage_service') and app.state.storage_service:
        features.extend(["Blueprint Uploads", "File Storage", "Blob Integration"])
    
    if hasattr(app.state, 'ai_service') and app.state.ai_service:
        features.extend(["AI Chat", "Blueprint Analysis", "Smart Annotations"])
    
    if hasattr(app.state, 'pdf_service') and app.state.pdf_service:
        features.extend(["PDF Processing", "Document Parsing"])
    
    if hasattr(app.state, 'session_service') and app.state.session_service:
        features.extend(["User Sessions", "Team Collaboration"])
    
    return {
        "project": getattr(settings, 'PROJECT_NAME', 'Smart Blueprint Chat API'),
        "version": "2.1.0",
        "environment": "development" if settings.DEBUG else "production",
        "features": features,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1"
        }
    }

@app.get("/api/v1/system/status", tags=["General"])
async def detailed_status():
    """Detailed system status for monitoring"""
    try:
        # Perform live health checks
        health_status = await verify_system_health(app) if hasattr(app.state, 'storage_service') else {}
        
        return {
            "overall_status": "operational",
            "timestamp": str(datetime.datetime.now()),  # ✅ FIXED: Use datetime.datetime instead of uvicorn.config.datetime
            "version": "2.1.0",
            "health_checks": health_status,
            "configuration": {
                "debug_mode": settings.DEBUG,
                "cors_enabled": bool(cors_origins),
                "storage_configured": bool(getattr(settings, 'AZURE_STORAGE_CONNECTION_STRING', None)),
                "ai_configured": bool(getattr(settings, 'OPENAI_API_KEY', None))
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "overall_status": "degraded",
            "error": str(e),
            "timestamp": str(datetime.datetime.now())  # ✅ FIXED: Use datetime.datetime instead of uvicorn.config.datetime
        }

# --- Local development server ---
if __name__ == "__main__":
    logger.info("🚀 Starting development server...")
    uvicorn.run(
        "main:app",
        host=getattr(settings, 'HOST', '0.0.0.0'),
        port=getattr(settings, 'PORT', 8000),
        reload=getattr(settings, 'DEBUG', False),
        log_level="info"
    )
