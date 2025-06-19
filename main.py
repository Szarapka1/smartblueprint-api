# main.py
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# App configuration
from app.core.config import get_settings

# Core services
from app.services.ai_service import AIService
from app.services.pdf_service import PDFService
from app.services.storage_service import StorageService
from app.services.session_service import SessionService

# API Routers
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router

# Initialize logger and settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartBlueprintChatAPI")
settings = get_settings()

# --- Application Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown with proper error handling
    and service initialization for collaborative blueprint management.
    """
    logger.info("Starting Smart Blueprint Chat API...")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    
    try:
        # Initialize core services in dependency order
        logger.info("Initializing core services...")
        
        # Storage service (foundation for everything)
        app.state.storage_service = StorageService(settings)
        await app.state.storage_service.verify_connection()
        logger.info("Storage service connected to Azure")
        
        # PDF processing service
        app.state.pdf_service = PDFService(settings)
        logger.info("PDF service initialized")
        
        # AI service with collaborative caching
        app.state.ai_service = AIService(settings)
        logger.info("AI service initialized with OpenAI")
        
        # Session service (legacy support)
        app.state.session_service = SessionService(settings)
        logger.info("Session service initialized")
        
        # Verify all critical components
        await verify_system_health(app)
        
        logger.info("All core services initialized successfully!")
        logger.info("Collaborative document system ready")
        
    except Exception as e:
        logger.critical(f"Failed to initialize core services: {e}")
        logger.critical("Application startup failed - check configuration and dependencies")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Smart Blueprint Chat API...")
    logger.info("Graceful shutdown complete")

async def verify_system_health(app: FastAPI):
    """Verify all systems are working correctly"""
    try:
        # Test storage containers
        storage = app.state.storage_service
        
        # Test main container
        blobs_main = await storage.list_blobs(container_name=settings.AZURE_CONTAINER_NAME)
        logger.info(f"Main container accessible: {len(blobs_main)} blobs found")
        
        # Test cache container  
        blobs_cache = await storage.list_blobs(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        logger.info(f"Cache container accessible: {len(blobs_cache)} blobs found")
        
        logger.info("Azure blob storage containers accessible")
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise

# --- FastAPI Application Setup ---
app = FastAPI(
    title=f"{settings.PROJECT_NAME} - Collaborative Edition",
    description="""
    Smart Blueprint Chat API - Collaborative Document Management
    
    ## Features
    - Shared Document Upload - Multiple users access the same blueprints
    - Intelligent Chat - AI-powered blueprint analysis with 87%+ token savings
    - Collaborative Annotations - Real-time annotation sharing across teams
    - Activity Tracking - Monitor document usage and collaboration
    - Advanced Caching - Lightning-fast responses with smart chunking
    
    ## Getting Started
    1. Upload a blueprint with a custom document_id
    2. Share the document_id with your team
    3. Everyone can chat and annotate collaboratively
    
    ## New Collaborative Endpoints
    - POST /api/v1/documents/upload - Upload shared documents
    - POST /api/v1/documents/{document_id}/chat - Ask questions
    - POST /api/v1/documents/{document_id}/annotations - Add annotations
    - GET /api/v1/documents/{document_id}/stats - View collaboration stats
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Enhanced CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "request_id": str(id(request))
        }
    )

# --- API Routes Registration ---
app.include_router(
    blueprint_router, 
    prefix="/api/v1", 
    tags=["Documents & Chat"]
)

app.include_router(
    document_router, 
    prefix="/api/v1", 
    tags=["Document Analytics"]
)

app.include_router(
    annotation_router, 
    prefix="/api/v1", 
    tags=["Collaborative Annotations"]
)

# --- API Status Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with system status"""
    return {
        "message": "Welcome to the Smart Blueprint Chat API - Collaborative Edition",
        "version": "2.0.0",
        "features": [
            "Shared document management",
            "Collaborative annotations", 
            "AI-powered chat with 87% token optimization",
            "Real-time activity tracking"
        ],
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """System health check endpoint"""
    try:
        # Quick health checks
        storage_ok = hasattr(app.state, 'storage_service')
        ai_ok = hasattr(app.state, 'ai_service')
        
        return {
            "status": "healthy" if (storage_ok and ai_ok) else "degraded",
            "services": {
                "storage": "OK" if storage_ok else "ERROR",
                "ai": "OK" if ai_ok else "ERROR",
                "pdf": "OK" if hasattr(app.state, 'pdf_service') else "ERROR"
            },
            "timestamp": "2025-06-18T22:30:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/api/v1/system/info", tags=["General"])
async def system_info():
    """Get system configuration and capabilities"""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": "2.0.0",
        "features": {
            "collaborative_documents": True,
            "ai_chat": True,
            "annotations": True,
            "activity_tracking": True,
            "token_optimization": True
        },
        "endpoints": {
            "upload": "/api/v1/documents/upload",
            "chat": "/api/v1/documents/{document_id}/chat", 
            "annotations": "/api/v1/documents/{document_id}/annotations",
            "document_list": "/api/v1/documents"
        }
    }

# --- Development Server ---
if __name__ == "__main__":
    logger.info("Starting development server...")
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG,
        log_level="info"
    )