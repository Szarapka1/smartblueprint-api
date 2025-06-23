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
    from app.services.ai_service import ProfessionalAIService
except ImportError as e:
    logging.warning(f"AI service import failed: {e}")
    ProfessionalAIService = None

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
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    
    # Print environment info for debugging
    logger.info(f"📊 Python environment: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ Storage Service disabled: class not available or connection string missing.")

    app.state.ai_service = None
    if ProfessionalAIService and settings and settings.OPENAI_API_KEY:
        try:
            # Initialize the ProfessionalAIService by passing the entire settings object
            app.state.ai_service = ProfessionalAIService(settings=settings)
            logger.info("✅ AI Service initialized with Professional capabilities")
            
            # Log AI capabilities
            capabilities = app.state.ai_service.get_professional_capabilities()
            logger.info(f"🤖 AI Service capabilities loaded: {len(capabilities.get('building_codes', []))} building codes, "
                       f"{len(capabilities.get('engineering_disciplines', []))} disciplines")
        except Exception as e:
            logger.error(f"❌ AI Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ AI Service disabled: class not available or OPENAI_API_KEY missing.")

    app.state.pdf_service = None
    if PDFService and settings:
        try:
            app.state.pdf_service = PDFService(settings=settings)
            logger.info("✅ PDF Service initialized")
        except Exception as e:
            logger.error(f"❌ PDF Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ PDF Service disabled: class not available or settings missing.")

    app.state.session_service = None
    if SessionService and settings:
        try:
            app.state.session_service = SessionService(settings=settings)
            logger.info("✅ Session Service initialized")
        except Exception as e:
            logger.error(f"❌ Session Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ Session Service disabled: class not available or settings missing.")

    # --- Log final status ---
    active_services = [
        "📁 Storage" if app.state.storage_service else None,
        "🤖 AI (Professional)" if app.state.ai_service else None,
        "📄 PDF" if app.state.pdf_service else None,
        "👥 Sessions" if app.state.session_service else None
    ]
    active_services_str = ', '.join(filter(None, active_services))
    logger.info(f"🎯 Startup complete. Active services: {active_services_str if active_services_str else 'Basic API only'}")
    logger.info("🌐 API is ready for blueprint analysis requests")

    yield

    logger.info("🛑 Shutting down gracefully...")
    if hasattr(app.state, 'ai_service') and app.state.ai_service and hasattr(app.state.ai_service, '__aexit__'):
        await app.state.ai_service.__aexit__(None, None, None)
        logger.info("🤖 AI Service shutdown complete.")
    logger.info("👋 Smart Blueprint API shutdown complete")

# --- Create FastAPI app ---
app = FastAPI(
    title=getattr(settings, 'PROJECT_NAME', 'Smart Blueprint Chat API'),
    description="Professional blueprint analysis with AI-powered code compliance, calculations, and insights.",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS Middleware ---
cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# If CORS_ORIGINS is defined in settings as a list, extend the default list
if hasattr(settings, 'CORS_ORIGINS') and isinstance(settings.CORS_ORIGINS, list):
    cors_origins.extend(origin.strip() for origin in settings.CORS_ORIGINS if origin.strip())

# Deduplicate origins
cors_origins = list(set(cors_origins))

# Log which origins are allowed
logger.info(f"🌐 CORS enabled for origins: {cors_origins}")

# Add middleware
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
    app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint Analysis"])
    logger.info("✅ Blueprint routes registered for professional analysis")

# --- Core API endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to Smart Blueprint Chat API - Professional Edition",
        "version": "2.2.0",
        "docs": "/docs",
        "capabilities": [
            "Blueprint Visual Analysis",
            "Code Compliance Verification",
            "Professional Calculations",
            "ADA Compliance Checking",
            "Fire Safety Analysis",
            "Quantity Takeoffs"
        ]
    }

@app.get("/health", tags=["General"])
async def health_check(request: Request):
    """Provides a health check of the API and its connected services."""
    app_state = request.app.state
    
    # Check AI service capabilities
    ai_capabilities = None
    if hasattr(app_state, 'ai_service') and app_state.ai_service:
        try:
            ai_capabilities = app_state.ai_service.get_professional_capabilities()
            ai_status = "professional"
        except:
            ai_status = "basic"
    else:
        ai_status = "disabled"
    
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "storage": "connected" if hasattr(app_state, 'storage_service') and app_state.storage_service else "disabled",
            "ai": ai_status,
            "pdf": "available" if hasattr(app_state, 'pdf_service') and app_state.pdf_service else "disabled",
            "sessions": "active" if hasattr(app_state, 'session_service') and app_state.session_service else "disabled"
        },
        "ai_capabilities": {
            "vision_analysis": True if ai_status == "professional" else False,
            "code_compliance": True if ai_status == "professional" else False,
            "calculations": True if ai_status == "professional" else False
        },
        "version": "2.2.0",
        "environment": "Production" if not getattr(settings, 'DEBUG', False) else "Development"
    }

@app.get("/capabilities", tags=["General"])
async def get_capabilities(request: Request):
    """Returns the professional capabilities of the AI service."""
    app_state = request.app.state
    
    if hasattr(app_state, 'ai_service') and app_state.ai_service:
        try:
            return {
                "status": "available",
                "capabilities": app_state.ai_service.get_professional_capabilities()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    else:
        return {
            "status": "unavailable",
            "message": "AI service not initialized"
        }

# --- Local development server ---
if __name__ == "__main__":
    host = getattr(settings, 'HOST', '0.0.0.0')
    port = getattr(settings, 'PORT', 8000)
    reload = getattr(settings, 'DEBUG', False)
    
    logger.info(f"🚀 Starting Smart Blueprint API server at http://{host}:{port}")
    logger.info("📊 Professional blueprint analysis ready")
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level="info")
