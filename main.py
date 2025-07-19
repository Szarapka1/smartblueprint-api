# main.py - ULTRA-RELIABLE VERSION FOR MAXIMUM COMPATIBILITY

import os
import logging
import uvicorn
import traceback
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime

# Import settings first
from app.core.config import get_settings

# Import services with error handling
try:
    from app.services.storage_service import StorageService
except ImportError as e:
    logging.error(f"Could not import StorageService: {e}")
    StorageService = None

try:
    from app.services.pdf_service import PDFService
except ImportError as e:
    logging.error(f"Could not import PDFService: {e}")
    PDFService = None

try:
    from app.services.vision_ai.ai_service_core import VisualIntelligenceFirst
except ImportError as e:
    logging.error(f"Could not import AI Service: {e}")
    VisualIntelligenceFirst = None

try:
    from app.services.session_service import SessionService
except ImportError as e:
    logging.error(f"Could not import SessionService: {e}")
    SessionService = None

# Import API routers
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router
from app.api.routes.note_routes import note_router

# --- Configure Logging ---
logging.basicConfig(
    level=logging.DEBUG,  # Maximum verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Quieten down noisy library loggers
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("azure").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)

# Get your application's specific logger
logger = logging.getLogger("SmartBlueprintAPI")
logger.setLevel(logging.DEBUG)

# Load settings
settings = get_settings()

# Define VERSION constant
API_VERSION = "1.0.0-ULTRA"

# --- Service Initialization Helpers ---

async def initialize_service_with_retry(service_name: str, init_func, max_attempts: int = 5):
    """Initialize a service with retry logic and detailed logging"""
    logger.info(f"🔄 Initializing {service_name}...")
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"   Attempt {attempt + 1}/{max_attempts}")
            service = await init_func()
            logger.info(f"✅ {service_name} initialized successfully")
            return service
        except Exception as e:
            logger.error(f"❌ {service_name} initialization failed (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"   Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ {service_name} initialization failed after {max_attempts} attempts")
                logger.error(traceback.format_exc())
                return None

# --- Application Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown with ULTRA-RELIABLE error handling.
    Services are initialized with retries and graceful degradation.
    """
    logger.info("="*80)
    logger.info("🚀 Initializing Smart Blueprint API (ULTRA-RELIABLE VERSION)...")
    logger.info("⚠️  WARNING: Running in maximum compatibility mode!")
    logger.info("="*80)
    
    # Track initialization status
    initialization_status = {
        "storage": {"status": "not_started", "error": None, "attempts": 0},
        "pdf": {"status": "not_started", "error": None, "attempts": 0},
        "ai": {"status": "not_started", "error": None, "attempts": 0},
        "session": {"status": "not_started", "error": None, "attempts": 0}
    }
    
    # Track startup time
    startup_start = time.time()
    
    try:
        # 1. Initialize Storage Service (CRITICAL - Required for all other services)
        initialization_status["storage"]["status"] = "initializing"
        
        async def init_storage():
            if not settings.AZURE_STORAGE_CONNECTION_STRING:
                raise ValueError("No Azure Storage connection string configured")
            
            storage_service = StorageService(settings)
            # Extended timeout for connection verification
            await asyncio.wait_for(storage_service.verify_connection(), timeout=60.0)
            return storage_service
        
        if settings.AZURE_STORAGE_CONNECTION_STRING and StorageService:
            app.state.storage_service = await initialize_service_with_retry(
                "Storage Service", 
                init_storage, 
                max_attempts=5
            )
            
            if app.state.storage_service:
                initialization_status["storage"]["status"] = "success"
            else:
                initialization_status["storage"]["status"] = "failed"
                initialization_status["storage"]["error"] = "Initialization failed after retries"
        else:
            app.state.storage_service = None
            initialization_status["storage"]["status"] = "disabled"
            initialization_status["storage"]["error"] = "No connection string or module not available"
            logger.warning("⚠️ Storage Service disabled - no connection string or import failed")
        
        # 2. Initialize PDF Service (Depends on Storage)
        initialization_status["pdf"]["status"] = "initializing"
        
        async def init_pdf():
            if not app.state.storage_service:
                raise ValueError("Storage service required but not available")
            if not PDFService:
                raise ValueError("PDFService module not available")
            
            return PDFService(settings)
        
        if app.state.storage_service and PDFService:
            app.state.pdf_service = await initialize_service_with_retry(
                "PDF Service",
                init_pdf,
                max_attempts=3
            )
            
            if app.state.pdf_service:
                initialization_status["pdf"]["status"] = "success"
            else:
                initialization_status["pdf"]["status"] = "failed"
                initialization_status["pdf"]["error"] = "Initialization failed"
        else:
            app.state.pdf_service = None
            initialization_status["pdf"]["status"] = "disabled"
            initialization_status["pdf"]["error"] = "Requires storage service or module not available"
            logger.warning("⚠️ PDF Service disabled")
        
        # 3. Initialize AI Service (Independent)
        initialization_status["ai"]["status"] = "initializing"
        
        async def init_ai():
            if not settings.OPENAI_API_KEY:
                raise ValueError("No OpenAI API key configured")
            if not VisualIntelligenceFirst:
                raise ValueError("AI Service module not available")
            
            return VisualIntelligenceFirst(settings)
        
        if settings.OPENAI_API_KEY and VisualIntelligenceFirst:
            app.state.ai_service = await initialize_service_with_retry(
                "AI Service",
                init_ai,
                max_attempts=3
            )
            
            if app.state.ai_service:
                initialization_status["ai"]["status"] = "success"
            else:
                initialization_status["ai"]["status"] = "failed"
                initialization_status["ai"]["error"] = "Initialization failed"
        else:
            app.state.ai_service = None
            initialization_status["ai"]["status"] = "disabled"
            initialization_status["ai"]["error"] = "No API key or module not available"
            logger.warning("⚠️ AI Service disabled")
        
        # 4. Initialize Session Service (Depends on Storage)
        initialization_status["session"]["status"] = "initializing"
        
        async def init_session():
            if not SessionService:
                raise ValueError("SessionService module not available")
            
            session_service = SessionService(settings, app.state.storage_service)
            # Start background tasks with extended timeout
            await asyncio.wait_for(session_service.start_background_cleanup(), timeout=30.0)
            return session_service
        
        if SessionService:
            # Session service can work with or without storage (degraded mode)
            app.state.session_service = await initialize_service_with_retry(
                "Session Service",
                init_session,
                max_attempts=3
            )
            
            if app.state.session_service:
                initialization_status["session"]["status"] = "success"
            else:
                initialization_status["session"]["status"] = "failed"
                initialization_status["session"]["error"] = "Initialization failed"
        else:
            app.state.session_service = None
            initialization_status["session"]["status"] = "disabled"
            initialization_status["session"]["error"] = "Module not available"
            logger.warning("⚠️ Session Service disabled")
        
        # 5. Store initialization status
        app.state.initialization_status = initialization_status
        
        # 6. Calculate startup time
        startup_time = time.time() - startup_start
        
        # 7. Log final status
        logger.info("="*80)
        logger.info("📊 Service Initialization Summary:")
        
        successful_services = 0
        for service, status in initialization_status.items():
            emoji = "✅" if status["status"] == "success" else "❌" if status["status"] == "failed" else "⚠️"
            logger.info(f"   {emoji} {service}: {status['status']}")
            if status["error"]:
                logger.info(f"      Error: {status['error']}")
            if status["status"] == "success":
                successful_services += 1
        
        logger.info(f"   ⏱️ Startup time: {startup_time:.1f}s")
        logger.info("="*80)
        
        if successful_services == 0:
            logger.error("❌ NO SERVICES INITIALIZED - API will have limited functionality")
        elif successful_services < 4:
            logger.warning(f"⚠️ Only {successful_services}/4 services initialized - Some features unavailable")
        else:
            logger.info("🎉 All services initialized successfully!")
        
        logger.info("🎉 API is ready! (ULTRA-RELIABLE MODE)")
        logger.info(f"📚 Interactive docs available at:")
        logger.info(f"   - http://localhost:{settings.PORT}/docs")
        logger.info(f"   - http://localhost:{settings.PORT}/redoc")
        logger.info("="*80)

        yield  # Application is now running

        # --- Shutdown Logic ---
        logger.info("="*80)
        logger.info("🛑 Shutting down API...")
        
        shutdown_start = time.time()
        
        # Gracefully shutdown services in reverse order
        services_to_shutdown = [
            ('session_service', 'Session Service'),
            ('ai_service', 'AI Service'),
            ('pdf_service', 'PDF Service'),
            ('storage_service', 'Storage Service')
        ]
        
        for service_attr, service_name in services_to_shutdown:
            service = getattr(app.state, service_attr, None)
            if service:
                try:
                    logger.info(f"   Shutting down {service_name}...")
                    
                    # Check for different shutdown methods
                    if hasattr(service, 'shutdown'):
                        await asyncio.wait_for(service.shutdown(), timeout=30.0)
                    elif hasattr(service, '__aexit__'):
                        await asyncio.wait_for(service.__aexit__(None, None, None), timeout=30.0)
                    elif hasattr(service, 'close'):
                        await asyncio.wait_for(service.close(), timeout=30.0)
                    
                    logger.info(f"   ✅ {service_name} shutdown complete")
                except asyncio.TimeoutError:
                    logger.error(f"   ⏱️ {service_name} shutdown timeout")
                except Exception as e:
                    logger.error(f"   ❌ {service_name} shutdown error: {e}")
                    logger.error(traceback.format_exc())
        
        shutdown_time = time.time() - shutdown_start
        logger.info(f"✅ Shutdown complete in {shutdown_time:.1f}s")
        logger.info("="*80)
    
    except Exception as e:
        logger.error(f"❌ Critical error during application lifecycle: {e}")
        logger.error(traceback.format_exc())
        
        # Store error in initialization status
        app.state.initialization_status = {
            "critical_error": str(e),
            "traceback": traceback.format_exc()
        }
        
        yield  # Let the app run even with errors

# --- Create FastAPI Application ---

app = FastAPI(
    title="Smart Blueprint API (ULTRA-RELIABLE)",
    description="""
    An intelligent API for analyzing construction blueprints with AI-powered insights.
    
    🛡️ **ULTRA-RELIABLE MODE**: This instance is running with:
    - Maximum retry attempts for all operations
    - Extended timeouts for slow connections
    - Graceful degradation when services fail
    - Comprehensive error recovery
    - Detailed logging and diagnostics
    
    ⚠️ **COMPATIBILITY MODE**: 
    - All CORS origins allowed
    - Detailed error messages exposed
    - Full stack traces in responses
    - No authentication required
    
    Perfect for testing and development!
    """,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=True
)

# --- PERMISSIVE CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow ALL methods
    allow_headers=["*"],  # Allow ALL headers
    expose_headers=["*"],  # Expose ALL headers
    max_age=3600  # Cache preflight requests for 1 hour
)

logger.info("⚠️ CORS: Allowing ALL origins, methods, and headers - DEVELOPMENT MODE!")

# --- Request/Response Logging Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses with timing"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - ERROR - {process_time:.3f}s - {e}")
        raise

# --- ULTRA-VERBOSE Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions with maximum detail"""
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error(f"Unhandled exception for {request.method} {request.url}:")
    logger.error("".join(tb_lines))
    
    # Try to get request body
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else "No body"
        if len(body_str) > 1000:
            body_str = body_str[:1000] + "... (truncated)"
    except:
        body_str = "Could not read body"
    
    # Get initialization status
    init_status = getattr(app.state, 'initialization_status', {})
    
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
            "initialization_status": init_status,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with extra detail"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": exc.detail,
            "path": str(request.url),
            "method": request.method,
            "headers": dict(request.headers) if exc.status_code >= 400 else None,
            "timestamp": datetime.utcnow().isoformat() + "Z"
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
    """Root endpoint with comprehensive system information"""
    init_status = getattr(app.state, 'initialization_status', {})
    
    # Calculate service availability
    services_available = {
        "storage": hasattr(app.state, 'storage_service') and app.state.storage_service is not None,
        "pdf": hasattr(app.state, 'pdf_service') and app.state.pdf_service is not None,
        "ai": hasattr(app.state, 'ai_service') and app.state.ai_service is not None,
        "session": hasattr(app.state, 'session_service') and app.state.session_service is not None
    }
    
    available_count = sum(1 for v in services_available.values() if v)
    
    return {
        "service": "Smart Blueprint API",
        "mode": "ULTRA-RELIABLE - Maximum Compatibility",
        "version": API_VERSION,
        "status": "operational" if available_count > 0 else "degraded",
        "warning": "This API is running in ULTRA-RELIABLE MODE with maximum compatibility settings!",
        "documentation": {
            "interactive": f"http://localhost:{settings.PORT}/docs",
            "redoc": f"http://localhost:{settings.PORT}/redoc"
        },
        "initialization_status": init_status,
        "features": {
            "storage": settings.is_feature_enabled("storage"),
            "ai_analysis": settings.is_feature_enabled("ai"),
            "pdf_processing": settings.is_feature_enabled("pdf_processing"),
            "admin_access": settings.is_feature_enabled("admin"),
            "notes": settings.is_feature_enabled("notes"),
            "highlighting": settings.is_feature_enabled("highlighting"),
            "trade_coordination": settings.is_feature_enabled("trade_coordination"),
            "unlimited_loading": settings.is_feature_enabled("unlimited_loading")
        },
        "services_available": services_available,
        "services_summary": f"{available_count}/4 services operational",
        "reliability_settings": {
            "max_retries": 10,
            "extended_timeouts": True,
            "graceful_degradation": True,
            "comprehensive_logging": True
        }
    }

@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Comprehensive health check with detailed service status"""
    def check_service(service_name: str) -> dict:
        """Check service health with timeout protection"""
        service = getattr(app.state, service_name, None)
        
        if service is None:
            return {
                "status": "unavailable",
                "message": "Service not initialized",
                "initialization_status": getattr(app.state, 'initialization_status', {}).get(
                    service_name.replace('_service', ''), {}
                )
            }
        
        try:
            result = {"status": "healthy", "details": {}}
            
            # Try various health check methods
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
            
            # Check for health indicators
            if hasattr(service, 'health_status'):
                health = service.health_status
                if isinstance(health, dict) and not health.get('healthy', True):
                    result["status"] = "degraded"
                    result["health_details"] = health
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc().splitlines()[-5:]
            }

    health_status = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "overall_status": "healthy",
        "mode": "ULTRA-RELIABLE",
        "version": API_VERSION,
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
            "python_version": os.sys.version,
            "host": settings.HOST,
            "port": settings.PORT
        },
        "reliability_features": {
            "retry_enabled": True,
            "extended_timeouts": True,
            "graceful_degradation": True,
            "error_recovery": True
        }
    }

    # Calculate overall status
    service_statuses = [s.get("status", "unknown") for s in health_status["services"].values()]
    
    if all(status == "unavailable" for status in service_statuses):
        health_status["overall_status"] = "critical"
    elif "error" in service_statuses:
        health_status["overall_status"] = "degraded"
    elif "degraded" in service_statuses:
        health_status["overall_status"] = "degraded"
    elif "unavailable" in service_statuses:
        health_status["overall_status"] = "partial"
    
    # Add recommendations
    if health_status["overall_status"] != "healthy":
        health_status["recommendations"] = []
        
        if not health_status["services"]["storage"]["status"] == "healthy":
            health_status["recommendations"].append("Check Azure Storage connection string and network connectivity")
        
        if not health_status["services"]["ai"]["status"] == "healthy":
            health_status["recommendations"].append("Verify OpenAI API key and check API status")
        
        if not health_status["services"]["pdf"]["status"] == "healthy":
            health_status["recommendations"].append("PDF service requires storage service to be operational")

    return health_status

@app.get("/config", tags=["System"])
async def get_configuration():
    """Get current configuration with reliability settings"""
    return {
        "mode": "ULTRA-RELIABLE - Maximum Compatibility",
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
            "trade_coordination": settings.is_feature_enabled("trade_coordination"),
            "unlimited_loading": settings.is_feature_enabled("unlimited_loading"),
            "load_all_thumbnails": settings.is_feature_enabled("load_all_thumbnails"),
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
        "reliability_settings": {
            "storage_max_retries": 10,
            "storage_timeout": 1800,
            "pdf_batch_size": 5,
            "pdf_timeout_per_page": 120,
            "session_persistence_batch": 3,
            "ai_request_timeout": settings.VISION_REQUEST_TIMEOUT,
            "global_request_timeout": settings.REQUEST_TIMEOUT
        },
        "environment": {
            "debug_mode": True,
            "cors_origins": ["*"],
            "version": API_VERSION,
            "environment": settings.ENVIRONMENT,
            "host": settings.HOST,
            "port": settings.PORT
        },
        "storage": {
            "main_container": settings.AZURE_CONTAINER_NAME,
            "cache_container": settings.AZURE_CACHE_CONTAINER_NAME,
            "connection_string_configured": bool(settings.AZURE_STORAGE_CONNECTION_STRING)
        },
        "ai": {
            "model": settings.OPENAI_MODEL,
            "max_tokens": settings.OPENAI_MAX_TOKENS,
            "temperature": settings.OPENAI_TEMPERATURE,
            "api_key_configured": bool(settings.OPENAI_API_KEY)
        },
        "pdf": {
            "preview_resolution": settings.PDF_PREVIEW_RESOLUTION,
            "high_resolution": settings.PDF_HIGH_RESOLUTION,
            "ai_dpi": settings.PDF_AI_DPI,
            "thumbnail_dpi": settings.PDF_THUMBNAIL_DPI,
            "batch_size": settings.PROCESSING_BATCH_SIZE,
            "jpeg_quality": settings.PDF_JPEG_QUALITY
        }
    }

@app.get("/debug/error-test", tags=["Debug"])
async def test_error_handling():
    """Test error handling to verify verbose error responses"""
    # This will trigger the global exception handler
    raise ValueError("This is a test error to verify ultra-verbose error handling is working correctly!")

@app.get("/debug/services", tags=["Debug"])
async def debug_services():
    """Debug endpoint to check service availability"""
    return {
        "services": {
            "storage": {
                "module_available": StorageService is not None,
                "instance_created": hasattr(app.state, 'storage_service'),
                "instance_active": getattr(app.state, 'storage_service', None) is not None
            },
            "pdf": {
                "module_available": PDFService is not None,
                "instance_created": hasattr(app.state, 'pdf_service'),
                "instance_active": getattr(app.state, 'pdf_service', None) is not None
            },
            "ai": {
                "module_available": VisualIntelligenceFirst is not None,
                "instance_created": hasattr(app.state, 'ai_service'),
                "instance_active": getattr(app.state, 'ai_service', None) is not None
            },
            "session": {
                "module_available": SessionService is not None,
                "instance_created": hasattr(app.state, 'session_service'),
                "instance_active": getattr(app.state, 'session_service', None) is not None
            }
        },
        "initialization_status": getattr(app.state, 'initialization_status', {})
    }

# --- Application Entry Point ---
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🚀 Starting Smart Blueprint API in ULTRA-RELIABLE MODE")
    logger.info("🛡️ Maximum compatibility and reliability settings enabled")
    logger.info("="*80)
    logger.info(f"🔧 Debug mode: ON")
    logger.info(f"🌐 CORS: Allowing ALL origins")
    logger.info(f"📝 Error details: FULLY EXPOSED")
    logger.info(f"🔓 Authentication: DISABLED")
    logger.info(f"🔄 Retry logic: MAXIMUM")
    logger.info(f"⏱️ Timeouts: EXTENDED")
    logger.info("="*80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,  # Auto-reload on code changes
        log_level="debug",  # Verbose logging
        access_log=True,  # Log all requests
        timeout_keep_alive=75,  # Extended keep-alive
        limit_concurrency=1000,  # High concurrency limit
        limit_max_requests=10000  # High request limit
    )