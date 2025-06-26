# main.py - COMPLETE VERSION WITH CORS FIX FOR TESTING

import os
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

# --- Service imports with proper error handling ---
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
    from app.services.ai_service import ProfessionalBlueprintAI
except ImportError as e:
    logging.warning(f"AI service import failed: {e}")
    ProfessionalBlueprintAI = None

try:
    from app.services.session_service import SessionService
except ImportError as e:
    logging.warning(f"Session service import failed: {e}")
    SessionService = None

# --- Route imports ---
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

try:
    from app.api.routes.note_routes import note_router
except ImportError as e:
    logging.warning(f"Note routes import failed: {e}")
    note_router = None

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBlueprintAPI")

# Suppress noisy loggers
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Load settings ---
try:
    settings = get_settings()
    logger.info("✅ Settings loaded successfully")
    environment = getattr(settings, 'ENVIRONMENT', os.getenv('ENVIRONMENT', 'production'))
    logger.info(f"🔧 Environment: {environment}")
except Exception as e:
    logger.error(f"❌ Settings failed to load: {e}")
    settings = None
    raise

# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown with optimized service initialization"""
    logger.info("="*60)
    logger.info("🚀 SMART BLUEPRINT API - PROFESSIONAL EDITION")
    logger.info("="*60)
    logger.info(f"📅 Starting at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    environment = getattr(settings, 'ENVIRONMENT', os.getenv('ENVIRONMENT', 'production'))
    debug_mode = getattr(settings, 'DEBUG', False)
    
    logger.info(f"🔧 Environment: {environment}")
    logger.info(f"🐍 Debug Mode: {'ON' if debug_mode else 'OFF'}")
    
    # Initialize service states
    app.state.storage_service = None
    app.state.ai_service = None
    app.state.pdf_service = None
    app.state.session_service = None
    app.state.settings = settings
    app.state.environment = environment
    
    # --- 1. Initialize Storage Service (Required) ---
    azure_storage_conn = getattr(settings, 'AZURE_STORAGE_CONNECTION_STRING', None)
    if StorageService and azure_storage_conn:
        try:
            logger.info("📦 Initializing Storage Service...")
            storage_service = StorageService(settings)
            await storage_service.verify_connection()
            app.state.storage_service = storage_service
            
            storage_info = storage_service.get_connection_info()
            logger.info("✅ Storage Service Ready")
            logger.info(f"   - Main container: {storage_info['containers']['main']}")
            logger.info(f"   - Cache container: {storage_info['containers']['cache']}")
            logger.info(f"   - Optimizations: Parallel uploads/downloads enabled")
            
        except Exception as e:
            logger.error(f"❌ Storage Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError("Cannot start without Storage Service")
    else:
        logger.error("❌ Storage Service disabled: No connection string")
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is required")

    # --- 2. Initialize AI Service (Required) ---
    openai_key = getattr(settings, 'OPENAI_API_KEY', None)
    if ProfessionalBlueprintAI and openai_key:
        try:
            logger.info("🤖 Initializing Professional AI Service...")
            ai_service = ProfessionalBlueprintAI(settings)
            app.state.ai_service = ai_service
            
            capabilities = ai_service.get_professional_capabilities()
            logger.info("✅ Professional AI Service Ready")
            logger.info(f"   - Building codes: {len(capabilities['building_codes'])} supported")
            logger.info(f"   - Engineering disciplines: {len(capabilities['engineering_disciplines'])} covered")
            logger.info(f"   - Max pages: {ai_service.max_pages_to_load}")
            logger.info(f"   - Batch processing: {ai_service.batch_size} pages/batch")
            logger.info(f"   - Image optimization: Enabled (quality={ai_service.image_quality}%)")
            logger.info(f"   - Note suggestions: Enabled")
            
        except Exception as e:
            logger.error(f"❌ AI Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError("Cannot start without AI Service")
    else:
        logger.error("❌ AI Service disabled: No OpenAI API key")
        raise RuntimeError("OPENAI_API_KEY is required")

    # --- 3. Initialize PDF Service (Required) ---
    if PDFService and settings:
        try:
            logger.info("📄 Initializing PDF Service...")
            pdf_service = PDFService(settings)
            app.state.pdf_service = pdf_service
            
            stats = pdf_service.get_processing_stats()
            logger.info("✅ PDF Service Ready")
            logger.info(f"   - Max pages: {stats['capabilities']['max_pages']}")
            logger.info(f"   - Parallel processing: {stats['capabilities']['parallel_processing']}")
            logger.info(f"   - Image optimization: {stats['capabilities']['image_optimization']}")
            logger.info(f"   - Grid detection: {stats['capabilities']['grid_detection']}")
            logger.info(f"   - Workers: {stats['performance']['parallel_workers']}")
            
        except Exception as e:
            logger.error(f"❌ PDF Service initialization failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError("Cannot start without PDF Service")
    else:
        logger.error("❌ PDF Service disabled")
        raise RuntimeError("PDF Service is required")

    # --- 4. Initialize Session Service (Optional but recommended) ---
    if SessionService and settings:
        try:
            logger.info("👥 Initializing Session Service...")
            session_service = SessionService(settings)
            app.state.session_service = session_service
            
            # Start background cleanup
            await session_service.start_background_cleanup()
            
            logger.info("✅ Session Service Ready")
            logger.info(f"   - Max sessions: {session_service.max_sessions}")
            logger.info(f"   - Max annotations/session: {session_service.max_annotations_per_session}")
            logger.info(f"   - Memory management: LRU eviction enabled")
            logger.info(f"   - Background cleanup: Running")
            
        except Exception as e:
            logger.error(f"❌ Session Service initialization failed: {e}")
            logger.warning("⚠️  Continuing without Session Service")
    else:
        logger.warning("⚠️  Session Service disabled")

    # --- Startup Summary ---
    host = getattr(settings, 'HOST', '0.0.0.0')
    port = getattr(settings, 'PORT', 8000)
    
    logger.info("="*60)
    logger.info("🎯 STARTUP COMPLETE - SYSTEM STATUS")
    logger.info("="*60)
    logger.info(f"✅ Storage Service: ACTIVE")
    logger.info(f"✅ AI Service: PROFESSIONAL MODE")
    logger.info(f"✅ PDF Service: OPTIMIZED")
    logger.info(f"{'✅' if app.state.session_service else '⚠️ '} Session Service: {'ACTIVE' if app.state.session_service else 'DISABLED'}")
    logger.info("="*60)
    logger.info("🌐 API Ready for Blueprint Analysis")
    logger.info(f"📍 Access docs at: http://{host}:{port}/docs")
    logger.info("="*60)
    logger.info("📋 Available Features:")
    logger.info("   - Multi-page blueprint analysis (up to 100 pages)")
    logger.info("   - AI-powered visual highlighting")
    logger.info("   - Private user sessions for highlights")
    logger.info("   - Collaborative note system")
    logger.info("   - Building code compliance checking")
    logger.info("   - Trade coordination & conflict detection")
    logger.info("   - Automatic note suggestions from AI")
    logger.info("="*60)

    yield

    # --- Shutdown ---
    logger.info("="*60)
    logger.info("🛑 SHUTTING DOWN SMART BLUEPRINT API")
    logger.info("="*60)
    
    # Cleanup services
    if hasattr(app.state, 'ai_service') and app.state.ai_service:
        try:
            if hasattr(app.state.ai_service, '__aexit__'):
                await app.state.ai_service.__aexit__(None, None, None)
            logger.info("✅ AI Service shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️  AI Service cleanup warning: {e}")
    
    if hasattr(app.state, 'storage_service') and app.state.storage_service:
        try:
            if hasattr(app.state.storage_service, '__aexit__'):
                await app.state.storage_service.__aexit__(None, None, None)
            logger.info("✅ Storage Service shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️  Storage Service cleanup warning: {e}")
    
    logger.info("👋 Smart Blueprint API shutdown complete")
    logger.info("="*60)

# --- Create FastAPI Application ---
app = FastAPI(
    title="Smart Blueprint Chat API - Professional Edition",
    description="""
    **Professional Blueprint Analysis with AI-Powered Intelligence**
    
    This API provides comprehensive blueprint analysis including:
    - 🏗️ Multi-page document support (up to 100 pages)
    - 📐 Building code compliance verification (BCBC, NBC, NFPA, CEC)
    - 🔍 Professional calculations and quantity takeoffs
    - 👥 Collaborative annotations and discussions
    - 🤖 GPT-4 Vision powered analysis with visual highlighting
    - 📝 AI-suggested note creation for important findings
    - 🔒 Privacy-first design with user-specific highlight sessions
    
    ## Key Features:
    
    ### Visual Intelligence
    - Analyzes blueprint images with GPT-4 Vision
    - Identifies elements and their grid locations
    - Creates visual highlights visible only to the requesting user
    - Detects grid systems for precise coordinate mapping
    
    ### Collaboration
    - Private notes by default (only visible to author)
    - Publish notes to share findings with all users
    - Track which trades are impacted by issues
    - Export notes and findings
    
    ### Building Code Knowledge
    - Falls back to code requirements when information is missing
    - Supports BCBC 2018, NBC, NFPA, CEC standards
    - Identifies potential code violations
    - Suggests compliance verification steps
    
    Optimized for large construction documents with parallel processing and intelligent caching.
    """,
    version="2.3.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Blueprint Analysis",
            "description": "Upload and analyze construction blueprints with AI"
        },
        {
            "name": "Document Analytics",
            "description": "Get statistics and collaboration insights"
        },
        {
            "name": "Annotations",
            "description": "Manage visual highlights and annotations (private to users)"
        },
        {
            "name": "Notes",
            "description": "Create and manage collaborative notes (private/public)"
        },
        {
            "name": "General",
            "description": "Health checks and system information"
        }
    ]
)

# --- CORS Configuration - UPDATED FOR TESTING ---
logger.info("🌐 Configuring CORS for testing...")

# Get environment-specific CORS settings
environment = getattr(settings, 'ENVIRONMENT', os.getenv('ENVIRONMENT', 'production')) if settings else 'production'
debug_mode = getattr(settings, 'DEBUG', False) if settings else False

# Base CORS origins
cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
    "http://localhost:8080",
    "http://localhost:8000",
    "https://localhost:3000",  # HTTPS localhost
    "https://127.0.0.1:3000",  # HTTPS localhost
    "https://talktosmartblueprints.netlify.app",
]

# Add any additional origins from settings
if hasattr(settings, 'CORS_ORIGINS') and settings:
    cors_value = getattr(settings, 'CORS_ORIGINS', '')
    if isinstance(cors_value, str) and cors_value:
        additional_origins = [origin.strip() for origin in cors_value.split(',') if origin.strip()]
        cors_origins.extend(additional_origins)
    elif isinstance(cors_value, list):
        cors_origins.extend(cors_value)

# For testing/development: Allow all origins
if environment in ['development', 'testing'] or debug_mode:
    logger.info("🧪 TESTING MODE: Allowing all CORS origins")
    cors_origins = ["*"]
    allow_credentials = False  # Must be False when allowing all origins
else:
    logger.info("🔒 PRODUCTION MODE: Restricted CORS origins")
    allow_credentials = True

# Remove duplicates while preserving order
if cors_origins != ["*"]:
    cors_origins = list(dict.fromkeys(cors_origins))  # Preserves order, removes duplicates

logger.info(f"🌐 CORS Configuration:")
logger.info(f"   - Origins: {cors_origins}")
logger.info(f"   - Credentials: {allow_credentials}")
logger.info(f"   - Environment: {environment}")

# Add CORS middleware with testing-friendly settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"], # Expose all headers to client
    max_age=3600,        # Cache preflight for 1 hour
)

logger.info("✅ CORS middleware configured successfully")

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handles all unhandled exceptions with proper logging"""
    logger.error(f"🚨 Unhandled exception on {request.method} {request.url.path}")
    logger.error(f"Error: {exc}")
    logger.debug(f"Traceback: {traceback.format_exc()}")
    
    debug_mode = getattr(settings, 'DEBUG', False) if settings else False
    error_detail = str(exc) if debug_mode else "An internal server error occurred"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": error_detail,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# --- Register ALL API Routes ---

# Blueprint routes (main functionality)
if blueprint_router:
    app.include_router(
        blueprint_router, 
        prefix="/api/v1", 
        tags=["Blueprint Analysis"]
    )
    logger.info("✅ Blueprint analysis routes registered")
else:
    logger.error("❌ Blueprint routes not available")

# Document analytics routes
if document_router:
    app.include_router(
        document_router, 
        prefix="/api/v1", 
        tags=["Document Analytics"]
    )
    logger.info("✅ Document analytics routes registered")
else:
    logger.warning("⚠️  Document analytics routes not available")

# Annotation routes (visual highlights)
if annotation_router:
    app.include_router(
        annotation_router, 
        prefix="/api/v1", 
        tags=["Annotations"]
    )
    logger.info("✅ Annotation routes registered")
else:
    logger.warning("⚠️  Annotation routes not available")

# Note routes (collaborative notes)
if note_router:
    app.include_router(
        note_router, 
        prefix="/api/v1", 
        tags=["Notes"]
    )
    logger.info("✅ Note routes registered")
else:
    logger.warning("⚠️  Note routes not available")

# --- Core API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with API information and quick start guide"""
    return {
        "message": "Welcome to Smart Blueprint Chat API - Professional Edition",
        "version": "2.3.0",
        "documentation": "/docs",
        "health_check": "/health",
        "cors_enabled": True,
        "testing_mode": True,
        "quick_start": {
            "1_upload": "POST /api/v1/documents/upload - Upload a PDF blueprint",
            "2_chat": "POST /api/v1/documents/{document_id}/chat - Ask questions",
            "3_highlights": "GET /api/v1/documents/{document_id}/highlights/active - View your highlights",
            "4_notes": "POST /api/v1/documents/{document_id}/notes - Create notes"
        },
        "capabilities": {
            "blueprint_analysis": True,
            "code_compliance": True,
            "multi_page_support": True,
            "max_pages": 100,
            "collaboration": True,
            "private_highlights": True,
            "public_notes": True,
            "ai_suggestions": True
        },
        "supported_codes": [
            "2018 BCBC (British Columbia Building Code)",
            "NBC (National Building Code of Canada)",
            "CSA Standards",
            "NFPA Standards",
            "CEC (Canadian Electrical Code)"
        ],
        "privacy_model": {
            "highlights": "Private to each user",
            "notes": "Private by default, can be published",
            "sessions": "User-specific with 24h expiration"
        }
    }

@app.get("/health", tags=["General"])
async def health_check(request: Request):
    """Comprehensive health check of all services"""
    app_state = request.app.state
    
    services_status = {}
    
    # Storage Service
    if hasattr(app_state, 'storage_service') and app_state.storage_service:
        try:
            services_status['storage'] = {
                "status": "healthy",
                "type": "Azure Blob Storage",
                "containers": app_state.storage_service.get_connection_info()['containers']
            }
        except Exception as e:
            services_status['storage'] = {"status": "unhealthy", "error": str(e)}
    else:
        services_status['storage'] = {"status": "disabled"}
    
    # AI Service
    if hasattr(app_state, 'ai_service') and app_state.ai_service:
        try:
            capabilities = app_state.ai_service.get_professional_capabilities()
            services_status['ai'] = {
                "status": "healthy",
                "mode": "professional",
                "max_pages": app_state.ai_service.max_pages_to_load,
                "disciplines": len(capabilities['engineering_disciplines']),
                "note_suggestions": "enabled"
            }
        except Exception as e:
            services_status['ai'] = {"status": "unhealthy", "error": str(e)}
    else:
        services_status['ai'] = {"status": "disabled"}
    
    # PDF Service
    if hasattr(app_state, 'pdf_service') and app_state.pdf_service:
        try:
            stats = app_state.pdf_service.get_processing_stats()
            services_status['pdf'] = {
                "status": "healthy",
                "max_pages": stats['capabilities']['max_pages'],
                "workers": stats['performance']['parallel_workers'],
                "grid_detection": stats['capabilities']['grid_detection']
            }
        except Exception as e:
            services_status['pdf'] = {"status": "unhealthy", "error": str(e)}
    else:
        services_status['pdf'] = {"status": "disabled"}
    
    # Session Service
    if hasattr(app_state, 'session_service') and app_state.session_service:
        try:
            stats = app_state.session_service.get_session_statistics()
            services_status['sessions'] = {
                "status": "healthy",
                "active_sessions": stats['total_sessions'],
                "capacity_used": f"{stats['capacity_used_percent']}%",
                "highlight_sessions": stats['total_highlight_sessions']
            }
        except Exception as e:
            services_status['sessions'] = {"status": "unhealthy", "error": str(e)}
    else:
        services_status['sessions'] = {"status": "disabled"}
    
    # Overall health
    all_healthy = all(
        service.get('status') == 'healthy' 
        for service in services_status.values() 
        if service.get('status') != 'disabled'
    )
    
    environment = getattr(app_state, 'environment', 'production')
    
    # Calculate uptime
    start_time = getattr(app.state, 'start_time', datetime.datetime.now())
    uptime_seconds = int((datetime.datetime.now() - start_time).total_seconds())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "2.3.0",
        "environment": environment,
        "services": services_status,
        "uptime_seconds": uptime_seconds,
        "uptime_human": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m",
        "cors_enabled": True,
        "testing_mode": True
    }

@app.get("/api/v1/system/status", tags=["General"])
async def system_status(request: Request):
    """Detailed system status and capabilities"""
    app_state = request.app.state
    
    environment = getattr(app_state, 'environment', 'production')
    debug_mode = getattr(settings, 'DEBUG', False) if settings else False
    
    system_info = {
        "api_version": "2.3.0",
        "environment": environment,
        "debug_mode": debug_mode,
        "cors_enabled": True,
        "testing_mode": True,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "features_enabled": {
            "document_notes": settings.ENABLE_DOCUMENT_NOTES if settings else True,
            "ai_highlighting": settings.ENABLE_AI_HIGHLIGHTING if settings else True,
            "private_notes": settings.ENABLE_PRIVATE_NOTES if settings else True,
            "note_publishing": settings.ENABLE_NOTE_PUBLISHING if settings else True,
            "trade_coordination": settings.ENABLE_TRADE_COORDINATION if settings else True
        }
    }
    
    # Get detailed capabilities if AI service is available
    if hasattr(app_state, 'ai_service') and app_state.ai_service:
        system_info['ai_capabilities'] = app_state.ai_service.get_professional_capabilities()
    
    # Get PDF processing stats
    if hasattr(app_state, 'pdf_service') and app_state.pdf_service:
        system_info['pdf_processing'] = app_state.pdf_service.get_processing_stats()
    
    # Get session stats
    if hasattr(app_state, 'session_service') and app_state.session_service:
        system_info['session_stats'] = app_state.session_service.get_session_statistics()
    
    return system_info

@app.get("/api/v1/system/info", tags=["General"])
async def system_info():
    """Basic system information and limits"""
    return {
        "name": "Smart Blueprint Chat API",
        "version": "2.3.0",
        "description": "Professional blueprint analysis with AI-powered intelligence",
        "cors_enabled": True,
        "testing_mode": True,
        "features": [
            "Multi-page blueprint analysis (up to 100 pages)",
            "Building code compliance verification",
            "Professional calculations and takeoffs",
            "Private user-specific highlights",
            "Collaborative note system",
            "Real-time AI chat with visual responses",
            "Document version tracking",
            "Trade coordination support",
            "AI-suggested note creation"
        ],
        "limits": {
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "60")),
            "max_pages": int(os.getenv("PDF_MAX_PAGES", "100")),
            "max_chat_length": 2000,
            "max_annotation_length": 1000,
            "max_notes_per_document": int(os.getenv("MAX_NOTES_PER_DOCUMENT", "500")),
            "max_note_length": int(os.getenv("MAX_NOTE_LENGTH", "10000")),
            "highlight_expiration_hours": 24
        },
        "supported_formats": ["PDF"],
        "supported_trades": [
            "General",
            "Electrical", 
            "Plumbing",
            "HVAC",
            "Fire Protection",
            "Structural",
            "Architectural"
        ]
    }

# --- Store startup time for uptime calculation ---
app.state.start_time = datetime.datetime.now()

# --- Development Server ---
if __name__ == "__main__":
    # Get host and port safely with defaults
    host = getattr(settings, 'HOST', '0.0.0.0') if settings else '0.0.0.0'
    port = getattr(settings, 'PORT', 8000) if settings else 8000
    reload = getattr(settings, 'DEBUG', False) if settings else False
    
    logger.info("="*60)
    logger.info(f"🚀 Starting development server at http://{host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"🔄 Auto-reload: {'ON' if reload else 'OFF'}")
    logger.info(f"🌐 CORS: ENABLED FOR ALL ORIGINS (Testing Mode)")
    logger.info("="*60)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )
