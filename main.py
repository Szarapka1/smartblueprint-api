# main.py - OPTIMIZED FOR LARGE DOCUMENT PROCESSING

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
    # Import the main AI service class
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
    logger.info(f"🔧 Environment: {settings.ENVIRONMENT}")
except Exception as e:
    logger.error(f"❌ Settings failed to load: {e}")
    settings = None
    raise

# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown with optimized service initialization
    """
    logger.info("="*60)
    logger.info("🚀 SMART BLUEPRINT API - PROFESSIONAL EDITION")
    logger.info("="*60)
    logger.info(f"📅 Starting at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔧 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🐍 Debug Mode: {'ON' if settings.DEBUG else 'OFF'}")
    
    # Initialize service states
    app.state.storage_service = None
    app.state.ai_service = None
    app.state.pdf_service = None
    app.state.session_service = None
    app.state.settings = settings
    
    # --- 1. Initialize Storage Service (Required) ---
    if StorageService and settings.AZURE_STORAGE_CONNECTION_STRING:
        try:
            logger.info("📦 Initializing Storage Service...")
            storage_service = StorageService(settings)
            await storage_service.verify_connection()
            app.state.storage_service = storage_service
            
            # Log storage configuration
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
    if ProfessionalBlueprintAI and settings.OPENAI_API_KEY:
        try:
            logger.info("🤖 Initializing Professional AI Service...")
            ai_service = ProfessionalBlueprintAI(settings)
            app.state.ai_service = ai_service
            
            # Log AI capabilities
            capabilities = ai_service.get_professional_capabilities()
            logger.info("✅ Professional AI Service Ready")
            logger.info(f"   - Building codes: {len(capabilities['building_codes'])} supported")
            logger.info(f"   - Engineering disciplines: {len(capabilities['engineering_disciplines'])} covered")
            logger.info(f"   - Max pages: {ai_service.max_pages_to_load}")
            logger.info(f"   - Batch processing: {ai_service.batch_size} pages/batch")
            logger.info(f"   - Image optimization: Enabled (quality={ai_service.image_quality}%)")
            
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
            
            # Log PDF processing configuration
            stats = pdf_service.get_processing_stats()
            logger.info("✅ PDF Service Ready")
            logger.info(f"   - Max pages: {stats['capabilities']['max_pages']}")
            logger.info(f"   - Parallel processing: {stats['capabilities']['parallel_processing']}")
            logger.info(f"   - Image optimization: {stats['capabilities']['image_optimization']}")
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
            
            logger.info("✅ Session Service Ready")
            logger.info(f"   - Max sessions: {session_service.max_sessions}")
            logger.info(f"   - Max annotations/session: {session_service.max_annotations_per_session}")
            logger.info(f"   - Memory management: LRU eviction enabled")
            
        except Exception as e:
            logger.error(f"❌ Session Service initialization failed: {e}")
            logger.warning("⚠️  Continuing without Session Service")
    else:
        logger.warning("⚠️  Session Service disabled")

    # --- Startup Summary ---
    logger.info("="*60)
    logger.info("🎯 STARTUP COMPLETE - SYSTEM STATUS")
    logger.info("="*60)
    logger.info(f"✅ Storage Service: ACTIVE")
    logger.info(f"✅ AI Service: PROFESSIONAL MODE")
    logger.info(f"✅ PDF Service: OPTIMIZED")
    logger.info(f"{'✅' if app.state.session_service else '⚠️ '} Session Service: {'ACTIVE' if app.state.session_service else 'DISABLED'}")
    logger.info("="*60)
    logger.info("🌐 API Ready for Blueprint Analysis")
    logger.info(f"📍 Access docs at: http://{settings.HOST}:{settings.PORT}/docs")
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
    - 📐 Building code compliance verification
    - 🔍 Professional calculations and quantity takeoffs
    - 👥 Collaborative annotations and discussions
    - 🤖 GPT-4 Vision powered analysis
    
    Optimized for large construction documents with parallel processing and intelligent caching.
    """,
    version="2.2.0",
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
            "name": "General",
            "description": "Health checks and system information"
        }
    ]
)

# --- CORS Configuration ---
cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "https://talktosmartblueprints.netlify.app",  # Production URL
]

# Add any additional origins from settings
if hasattr(settings, 'CORS_ORIGINS'):
    if isinstance(settings.CORS_ORIGINS, str):
        cors_origins.extend([origin.strip() for origin in settings.CORS_ORIGINS.split(',') if origin.strip()])
    elif isinstance(settings.CORS_ORIGINS, list):
        cors_origins.extend(settings.CORS_ORIGINS)

# Deduplicate origins
cors_origins = list(set(cors_origins))

logger.info(f"🌐 CORS enabled for {len(cors_origins)} origins")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handles all unhandled exceptions with proper logging"""
    logger.error(f"🚨 Unhandled exception on {request.method} {request.url.path}")
    logger.error(f"Error: {exc}")
    logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Provide detailed error in debug mode only
    error_detail = str(exc) if settings.DEBUG else "An internal server error occurred"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": error_detail,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# --- Register API Routes ---
if blueprint_router:
    app.include_router(
        blueprint_router, 
        prefix="/api/v1", 
        tags=["Blueprint Analysis"]
    )
    logger.info("✅ Blueprint analysis routes registered")
else:
    logger.error("❌ Blueprint routes not available")

if document_router:
    app.include_router(
        document_router, 
        prefix="/api/v1", 
        tags=["Document Analytics"]
    )
    logger.info("✅ Document analytics routes registered")
else:
    logger.warning("⚠️  Document analytics routes not available")

# --- Core API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to Smart Blueprint Chat API - Professional Edition",
        "version": "2.2.0",
        "documentation": "/docs",
        "health_check": "/health",
        "capabilities": {
            "blueprint_analysis": True,
            "code_compliance": True,
            "multi_page_support": True,
            "max_pages": 100,
            "collaboration": True,
            "annotations": True
        },
        "supported_codes": [
            "2018 BCBC (British Columbia Building Code)",
            "NBC (National Building Code of Canada)",
            "CSA Standards",
            "NFPA Standards"
        ]
    }

@app.get("/health", tags=["General"])
async def health_check(request: Request):
    """Comprehensive health check of all services"""
    app_state = request.app.state
    
    # Check each service
    services_status = {}
    
    # Storage Service
    if hasattr(app_state, 'storage_service') and app_state.storage_service:
        try:
            # Quick connectivity check
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
                "disciplines": len(capabilities['engineering_disciplines'])
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
                "workers": stats['performance']['parallel_workers']
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
                "capacity_used": f"{stats['capacity_used_percent']}%"
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
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "2.2.0",
        "environment": settings.ENVIRONMENT,
        "services": services_status,
        "uptime_seconds": int((datetime.datetime.now() - app.state.get('start_time', datetime.datetime.now())).total_seconds())
    }

@app.get("/api/v1/system/status", tags=["General"])
async def system_status(request: Request):
    """Detailed system status and capabilities"""
    app_state = request.app.state
    
    system_info = {
        "api_version": "2.2.0",
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG,
        "timestamp": datetime.datetime.utcnow().isoformat()
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
    """Basic system information"""
    return {
        "name": "Smart Blueprint Chat API",
        "version": "2.2.0",
        "description": "Professional blueprint analysis with AI-powered intelligence",
        "features": [
            "Multi-page blueprint analysis (up to 100 pages)",
            "Building code compliance verification",
            "Professional calculations and takeoffs",
            "Collaborative annotations",
            "Real-time AI chat",
            "Document version tracking"
        ],
        "limits": {
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "60")),
            "max_pages": int(os.getenv("PDF_MAX_PAGES", "100")),
            "max_chat_length": 2000,
            "max_annotation_length": 1000
        }
    }

# --- Store startup time for uptime calculation ---
app.state.start_time = datetime.datetime.now()

# --- Development Server ---
if __name__ == "__main__":
    host = settings.HOST
    port = settings.PORT
    reload = settings.DEBUG
    
    logger.info("="*60)
    logger.info(f"🚀 Starting development server at http://{host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"🔄 Auto-reload: {'ON' if reload else 'OFF'}")
    logger.info("="*60)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )
