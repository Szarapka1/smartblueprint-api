import datetime
import logging
import uvicorn
import traceback
from contextlib import asynccontextmanager
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
    # Use the ProfessionalAIService aliased as AIService
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

# --- Init logging & settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartBlueprintAPI")

try:
    settings = get_settings()
    logger.info("✅ Settings loaded successfully")
except Exception as e:
    logger.error(f"❌ Settings failed to load: {e}. The application may not function correctly.")
    settings = None

# --- Application Lifecycle (Startup & Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes all services and attaches them to the app state.
    """
    logger.info("🚀 Starting Smart Blueprint API...")
    logger.info(f"🔧 Environment: {'Development' if getattr(settings, 'DEBUG', False) else 'Production'}")

    # --- Initialize services explicitly and safely ---
    app.state.storage_service = None
    if StorageService and settings and settings.AZURE_STORAGE_CONNECTION_STRING:
        try:
            storage_service = StorageService(settings)
            await storage_service.verify_connection()
            app.state.storage_service = storage_service
            logger.info("✅ Storage Service initialized and connection verified")
        except Exception as e:
            logger.error(f"❌ Storage Service initialization failed: {e}")
    else:
        logger.warning("⚠️ Storage Service disabled: class not available or connection string missing.")

    app.state.ai_service = None
    if AIService and settings and settings.OPENAI_API_KEY:
        try:
            # FIX: Initialize the new AIService by passing the entire settings object
            app.state.ai_service = AIService(settings=settings)
            logger.info("✅ AI Service initialized")
        except Exception as e:
            logger.error(f"❌ AI Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ AI Service disabled: class not available or OPENAI_API_KEY missing.")

    app.state.pdf_service = None
    if PDFService:
        try:
            # Assuming PDFService has a simple constructor
            app.state.pdf_service = PDFService()
            logger.info("✅ PDF Service initialized")
        except Exception as e:
            logger.error(f"❌ PDF Service initialization failed: {e}")

    app.state.session_service = None
    if SessionService:
        try:
            # Assuming SessionService has a simple constructor
            app.state.session_service = SessionService()
            logger.info("✅ Session Service initialized")
        except Exception as e:
            logger.error(f"❌ Session Service initialization failed: {e}")

    # --- Log final status ---
    active_services = [
        "📁 Storage" if app.state.storage_service else None,
        "🤖 AI" if app.state.ai_service else None,
        "📄 PDF" if app.state.pdf_service else None,
        "👥 Sessions" if app.state.session_service else None
    ]
    active_services_str = ', '.join(filter(None, active_services))
    logger.info(f"🎯 Startup complete. Active services: {active_services_str if active_services_str else 'Basic API only'}")
    logger.info("🌐 API is ready for requests")

    yield

    logger.info("🛑 Shutting down gracefully...")
    if hasattr(app.state, 'ai_service') and app.state.ai_service:
        # Gracefully shutdown the AI service's thread pool
        await app.state.ai_service.__aexit__(None, None, None)
        logger.info("🤖 AI Service shutdown complete.")


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
cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# If settings.CORS_ORIGINS is a list, extend directly (no .split)
if hasattr(settings, 'CORS_ORIGINS') and isinstance(settings.CORS_ORIGINS, list):
    cors_origins.extend(origin.strip() for origin in settings.CORS_ORIGINS if origin.strip())

# Remove duplicates just in case
cors_origins = list(set(cors_origins))

logger.info(f"🌐 CORS enabled for origins: {cors_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Robust exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"🚨 Unhandled error: {exc}")
    logger.debug(f"Traceback: {traceback.format_exc()}")
    error_detail = str(exc) if getattr(settings, 'DEBUG', False) else "An internal server error occurred."
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": error_detail,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )

# --- Register routes safely ---
if blueprint_router:
    app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint Chat"])
    logger.info("✅ Blueprint routes registered")

# Add other routers if they exist
# if document_router: app.include_router(...)
# if annotation_router: app.include_router(...)

# --- Core API endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to Smart Blueprint Chat API", "version": "2.1.0", "docs": "/docs"}

@app.get("/health", tags=["General"])
async def health_check(request: Request):
    """Provides a health check of the API and its connected services."""
    app_state = request.app.state
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "storage": "connected" if hasattr(app_state, 'storage_service') and app_state.storage_service else "disabled",
            "ai": "connected" if hasattr(app_state, 'ai_service') and app_state.ai_service else "disabled",
            "pdf": "available" if hasattr(app_state, 'pdf_service') and app_state.pdf_service else "disabled",
            "sessions": "active" if hasattr(app_state, 'session_service') and app_state.session_service else "disabled"
        },
        "version": "2.1.0"
    }

# --- Local development server ---
if __name__ == "__main__":
    host = getattr(settings, 'HOST', '0.0.0.0')
    port = getattr(settings, 'PORT', 8000)
    reload = getattr(settings, 'DEBUG', False)
    
    logger.info(f"🚀 Starting development server at http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level="info")
