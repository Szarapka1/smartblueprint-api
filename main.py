# main.py - PERMISSIVE TEST CONFIGURATION WITH PRODUCTION-GRADE CORE + SSE SUPPORT

import os
import logging
import uvicorn
import traceback
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import settings first
from app.core.config import get_settings

# Import core services with error handling
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService

# Try to import optional services
try:
    from app.services.vision_ai.ai_service_core import VisualIntelligenceFirst
    AI_SERVICE_AVAILABLE = True
except ImportError:
    AI_SERVICE_AVAILABLE = False
    logging.warning("AI service module not found - AI features will be disabled")

try:
    from app.services.session_service import SessionService
    SESSION_SERVICE_AVAILABLE = True
except ImportError:
    SESSION_SERVICE_AVAILABLE = False
    logging.warning("Session service module not found - session tracking will be disabled")

# Import core API routers (these should exist)
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router

# Try to import optional routers
try:
    from app.api.routes.annotation_routes import annotation_router
    ANNOTATION_ROUTES_AVAILABLE = True
except ImportError:
    ANNOTATION_ROUTES_AVAILABLE = False
    logging.warning("Annotation routes not found - annotation endpoints will be unavailable")

try:
    from app.api.routes.note_routes import note_router
    NOTE_ROUTES_AVAILABLE = True
except ImportError:
    NOTE_ROUTES_AVAILABLE = False
    logging.warning("Note routes not found - note endpoints will be unavailable")

# --- Configure Logging ---
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level for test environment
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBlueprintAPI")

# Load settings
settings = get_settings()

# Define VERSION constant
API_VERSION = "1.0.0"

# --- SSE Event Management Classes ---

class SSEConnection:
    """Represents a single SSE connection"""
    def __init__(self, client_id: str, document_id: str):
        self.client_id = client_id
        self.document_id = document_id
        self.queue = asyncio.Queue(maxsize=settings.SSE_EVENT_QUEUE_SIZE)
        self.connected_at = asyncio.get_event_loop().time()
        self.last_activity = self.connected_at
        self.active = True

class SSEEventManager:
    """Manages SSE connections and event distribution"""
    def __init__(self, settings):
        self.settings = settings
        self.connections = defaultdict(dict)  # {document_id: {client_id: SSEConnection}}
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        
    async def start(self):
        """Start the event manager"""
        if self.settings.ENABLE_SSE:
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())
            logger.info("✅ SSE Event Manager started")
    
    async def stop(self):
        """Stop the event manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self.lock:
            for doc_connections in self.connections.values():
                for connection in doc_connections.values():
                    connection.active = False
            self.connections.clear()
        
        logger.info("✅ SSE Event Manager stopped")
    
    async def add_connection(self, document_id: str, client_id: str) -> SSEConnection:
        """Add a new SSE connection"""
        async with self.lock:
            # Check connection limit
            if len(self.connections[document_id]) >= self.settings.SSE_MAX_CONNECTIONS_PER_DOCUMENT:
                # Remove oldest connection
                oldest_client = min(
                    self.connections[document_id].items(),
                    key=lambda x: x[1].connected_at
                )[0]
                old_conn = self.connections[document_id].pop(oldest_client)
                old_conn.active = False
                logger.info(f"Removed oldest SSE connection for document {document_id}")
            
            # Create new connection
            connection = SSEConnection(client_id, document_id)
            self.connections[document_id][client_id] = connection
            
            logger.info(f"Added SSE connection: document={document_id}, client={client_id}")
            logger.info(f"Active connections for document: {len(self.connections[document_id])}")
            
            return connection
    
    async def remove_connection(self, document_id: str, client_id: str):
        """Remove an SSE connection"""
        async with self.lock:
            if document_id in self.connections and client_id in self.connections[document_id]:
                connection = self.connections[document_id].pop(client_id)
                connection.active = False
                
                # Clean up empty document entries
                if not self.connections[document_id]:
                    del self.connections[document_id]
                
                logger.info(f"Removed SSE connection: document={document_id}, client={client_id}")
    
    async def send_event(self, document_id: str, event_type: str, data: dict):
        """Send an event to all connected clients for a document"""
        if not self.settings.ENABLE_SSE:
            return
        
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time(),
            "document_id": document_id
        }
        
        async with self.lock:
            if document_id in self.connections:
                # Send to all connected clients
                disconnected = []
                for client_id, connection in self.connections[document_id].items():
                    if connection.active:
                        try:
                            # Non-blocking put with timeout
                            await asyncio.wait_for(
                                connection.queue.put(event),
                                timeout=1.0
                            )
                            connection.last_activity = asyncio.get_event_loop().time()
                        except (asyncio.TimeoutError, asyncio.QueueFull):
                            logger.warning(f"Failed to send event to client {client_id} - queue full")
                        except Exception as e:
                            logger.error(f"Error sending event to client {client_id}: {e}")
                            disconnected.append(client_id)
                    else:
                        disconnected.append(client_id)
                
                # Clean up disconnected clients
                for client_id in disconnected:
                    self.connections[document_id].pop(client_id, None)
                
                if not self.connections[document_id]:
                    del self.connections[document_id]
    
    async def _cleanup_idle_connections(self):
        """Periodically clean up idle connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = asyncio.get_event_loop().time()
                timeout = self.settings.SSE_CLIENT_TIMEOUT
                
                async with self.lock:
                    disconnected = []
                    
                    for document_id, doc_connections in self.connections.items():
                        for client_id, connection in doc_connections.items():
                            if current_time - connection.last_activity > timeout:
                                connection.active = False
                                disconnected.append((document_id, client_id))
                    
                    # Remove disconnected
                    for document_id, client_id in disconnected:
                        if document_id in self.connections:
                            self.connections[document_id].pop(client_id, None)
                            if not self.connections[document_id]:
                                del self.connections[document_id]
                        logger.info(f"Cleaned up idle SSE connection: {client_id}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SSE cleanup task: {e}")

# Store active processing tasks
processing_tasks = {}

# --- Stub Session Service if not available ---
if not SESSION_SERVICE_AVAILABLE:
    class SessionService:
        """Stub session service when real implementation is not available"""
        def __init__(self, settings):
            self.settings = settings
            self._running = False
            logger.info("Using stub SessionService")
        
        def set_storage_service(self, storage_service):
            pass
        
        async def start(self):
            self._running = True
            
        async def stop(self):
            self._running = False
            
        def is_running(self):
            return self._running
        
        async def create_session(self, document_id: str, original_filename: str):
            logger.debug(f"Stub: create_session called for {document_id}")
            
        async def update_session_metadata(self, document_id: str, metadata: dict):
            logger.debug(f"Stub: update_session_metadata called for {document_id}")
            
        async def record_chat_activity(self, document_id: str, user: str, chat_data: dict):
            logger.debug(f"Stub: record_chat_activity called for {document_id}")
            
        async def clear_session(self, document_id: str):
            logger.debug(f"Stub: clear_session called for {document_id}")
        
        async def get_session_info(self, document_id: str):
            return {"status": "stub_session", "document_id": document_id}

# --- Stub AI Service if not available ---
if not AI_SERVICE_AVAILABLE:
    class VisualIntelligenceFirst:
        """Stub AI service when real implementation is not available"""
        def __init__(self, settings):
            self.settings = settings
            logger.info("Using stub AI Service")
        
        async def get_ai_response(self, **kwargs):
            return {
                "ai_response": "AI service is not available. Please ensure the AI service module is installed and OPENAI_API_KEY is configured.",
                "visual_highlights": [],
                "note_suggestion": None
            }
        
        def get_professional_capabilities(self):
            return {"status": "stub", "message": "AI service not available"}
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

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
        "session": {"status": "not_started", "error": None},
        "sse": {"status": "not_started", "error": None}
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
            if settings.OPENAI_API_KEY and (AI_SERVICE_AVAILABLE or True):  # Always try
                ai_service = VisualIntelligenceFirst(settings)
                app.state.ai_service = ai_service
                initialization_status["ai"]["status"] = "success"
                logger.info("✅ AI Service initialized successfully")
            else:
                app.state.ai_service = None
                initialization_status["ai"]["status"] = "disabled"
                logger.warning("⚠️ AI Service disabled - no API key or module not available")
        except Exception as e:
            initialization_status["ai"]["status"] = "failed"
            initialization_status["ai"]["error"] = str(e)
            app.state.ai_service = None
            logger.error(f"❌ AI Service failed: {e}")
            logger.error(traceback.format_exc())
        
        # 4. Initialize Session Service
        initialization_status["session"]["status"] = "initializing"
        try:
            # Always use SessionService (stub or real)
            session_service = SessionService(settings)
            
            # Set storage service if available
            if app.state.storage_service:
                session_service.set_storage_service(app.state.storage_service)
            
            # Start the service
            await session_service.start()
            
            app.state.session_service = session_service
            initialization_status["session"]["status"] = "success"
            logger.info("✅ Session Service initialized successfully")
        except Exception as e:
            initialization_status["session"]["status"] = "failed"
            initialization_status["session"]["error"] = str(e)
            app.state.session_service = None
            logger.error(f"❌ Session Service failed: {e}")
            logger.error(traceback.format_exc())
        
        # 5. Initialize SSE Event Manager
        initialization_status["sse"]["status"] = "initializing"
        try:
            sse_manager = SSEEventManager(settings)
            await sse_manager.start()
            app.state.sse_manager = sse_manager
            initialization_status["sse"]["status"] = "success"
            logger.info("✅ SSE Event Manager initialized successfully")
        except Exception as e:
            initialization_status["sse"]["status"] = "failed"
            initialization_status["sse"]["error"] = str(e)
            app.state.sse_manager = None
            logger.error(f"❌ SSE Event Manager failed: {e}")
            logger.error(traceback.format_exc())
        
        # 6. Store initialization status and processing tasks
        app.state.initialization_status = initialization_status
        app.state.processing_tasks = processing_tasks
        
        # 7. Log final status
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
        if settings.ENABLE_SSE:
            logger.info(f"📡 SSE (Server-Sent Events) enabled")
        logger.info("="*60)

        yield  # Application is now running

        # --- Shutdown Logic ---
        logger.info("="*60)
        logger.info("🛑 Shutting down API...")
        
        # Cancel all processing tasks
        for task_id, task in processing_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled processing task: {task_id}")
        
        # Gracefully shutdown services
        if hasattr(app.state, 'sse_manager') and app.state.sse_manager:
            try:
                await app.state.sse_manager.stop()
                logger.info("✅ SSE Event Manager shutdown complete")
            except Exception as e:
                logger.error(f"❌ SSE Event Manager shutdown error: {e}")
                logger.error(traceback.format_exc())
        
        if hasattr(app.state, 'session_service') and app.state.session_service:
            try:
                await app.state.session_service.stop()
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
    
    🚀 **NEW**: Server-Sent Events (SSE) support for real-time status updates
    
    DO NOT USE IN PRODUCTION!
    """,
    version=f"{API_VERSION}-TEST",
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
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error(f"Unhandled exception for {request.method} {request.url}:")
    logger.error("".join(tb_lines))
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else "No body"
    except:
        body_str = "Could not read body"
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
# Always include core routers
app.include_router(blueprint_router, tags=["Blueprint Analysis"])
app.include_router(document_router, tags=["Document Management"])

# Conditionally include optional routers
if ANNOTATION_ROUTES_AVAILABLE:
    app.include_router(annotation_router, tags=["Annotations"])
else:
    logger.warning("⚠️ Annotation routes not included - module not found")

if NOTE_ROUTES_AVAILABLE:
    app.include_router(note_router, tags=["Notes"])
else:
    logger.warning("⚠️ Note routes not included - module not found")

# --- Root Endpoints ---
@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Smart Blueprint API",
        "mode": "TEST/DEVELOPMENT - UNSAFE",
        "version": API_VERSION,
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
            "trade_coordination": settings.is_feature_enabled("trade_coordination"),
            "sse": settings.is_feature_enabled("sse")
        },
        "services_available": {
            "storage": hasattr(app.state, 'storage_service') and app.state.storage_service is not None,
            "pdf": hasattr(app.state, 'pdf_service') and app.state.pdf_service is not None,
            "ai": hasattr(app.state, 'ai_service') and app.state.ai_service is not None,
            "session": hasattr(app.state, 'session_service') and app.state.session_service is not None,
            "sse": hasattr(app.state, 'sse_manager') and app.state.sse_manager is not None
        },
        "modules_available": {
            "ai_service": AI_SERVICE_AVAILABLE,
            "session_service": SESSION_SERVICE_AVAILABLE,
            "annotation_routes": ANNOTATION_ROUTES_AVAILABLE,
            "note_routes": NOTE_ROUTES_AVAILABLE
        },
        "sse_config": settings.get_sse_settings() if settings.ENABLE_SSE else None
    }

@app.get("/api/v1/health", tags=["System"])
async def health_check():
    def check_service(service_name: str) -> dict:
        service = getattr(app.state, service_name, None)
        if service is None:
            return {
                "status": "unavailable",
                "message": "Service not initialized",
                "initialization_error": getattr(app.state, 'initialization_status', {}).get(
                    service_name.replace('_service', '').replace('_manager', ''), {}
                ).get('error')
            }
        try:
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
            
            # SSE specific checks
            if service_name == 'sse_manager' and hasattr(service, 'connections'):
                result["details"]["active_connections"] = sum(
                    len(conns) for conns in service.connections.values()
                )
                result["details"]["monitored_documents"] = len(service.connections)
            
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc().splitlines()[-5:]
            }

    health_status = {
        "timestamp": logging.Formatter().formatTime(logging.LogRecord(
            name="", level=0, pathname="", lineno=0,
            msg="", args=(), exc_info=None
        )),
        "overall_status": "healthy",
        "mode": "TEST/DEVELOPMENT",
        "version": API_VERSION,
        "initialization_status": getattr(app.state, 'initialization_status', {}),
        "services": {
            "storage": check_service('storage_service'),
            "ai": check_service('ai_service'),
            "pdf": check_service('pdf_service'),
            "session": check_service('session_service'),
            "sse": check_service('sse_manager')
        },
        "environment": {
            "debug": settings.DEBUG,
            "cors_origins": settings.CORS_ORIGINS,
            "python_version": os.sys.version,
            "sse_enabled": settings.ENABLE_SSE
        }
    }

    service_statuses = [s.get("status", "unknown") for s in health_status["services"].values()]
    if "error" in service_statuses:
        health_status["overall_status"] = "degraded"
    elif all(status == "unavailable" for status in service_statuses):
        health_status["overall_status"] = "critical"

    return health_status

@app.get("/config", tags=["System"])
async def get_configuration():
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
            "trade_coordination": settings.is_feature_enabled("trade_coordination"),
            "sse": settings.is_feature_enabled("sse")
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
        },
        "sse": settings.get_sse_settings() if settings.ENABLE_SSE else None
    }

@app.get("/debug/error-test", tags=["Debug"])
async def test_error_handling():
    raise ValueError("This is a test error to verify verbose error handling is working correctly!")

@app.get("/debug/routes", tags=["Debug"])
async def list_routes():
    """List all registered routes for debugging"""
    routes = []
    for route in app.routes:
        if hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if hasattr(route, "methods") else None,
                "name": route.name if hasattr(route, "name") else None,
                "endpoint": route.endpoint.__name__ if hasattr(route, "endpoint") and hasattr(route.endpoint, "__name__") else None
            })
    return {
        "total_routes": len(routes),
        "routes": sorted(routes, key=lambda x: x["path"])
    }

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
    if settings.ENABLE_SSE:
        logger.info(f"📡 SSE: ENABLED")
    logger.info("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,  # Auto-reload on code changes
        log_level="debug",  # Verbose logging
        access_log=True  # Log all requests
    )