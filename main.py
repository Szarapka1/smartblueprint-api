# main.py - UNSAFE TEST VERSION - ALLOW EVERYTHING
# ⚠️ WARNING: This is for testing only! Never use in production!

import os
import datetime
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# App config and services
from app.core.config import get_settings
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService
from app.services.ai_service import ProfessionalBlueprintAI
from app.services.session_service import SessionService

# Routes
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router
from app.api.routes.note_routes import note_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBlueprintAPI")

# Load settings
settings = get_settings()

# Application Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simple startup/shutdown"""
    logger.info("="*60)
    logger.info("🚀 SMART BLUEPRINT API - UNSAFE TEST VERSION")
    logger.info("⚠️  WARNING: ALL SECURITY DISABLED FOR TESTING")
    logger.info("="*60)
    
    # Initialize services without any checks
    try:
        # Storage Service
        app.state.storage_service = StorageService(settings)
        logger.info("✅ Storage Service initialized")
        
        # AI Service
        app.state.ai_service = ProfessionalBlueprintAI(settings)
        logger.info("✅ AI Service initialized")
        
        # PDF Service
        app.state.pdf_service = PDFService(settings)
        logger.info("✅ PDF Service initialized")
        
        # Session Service
        app.state.session_service = SessionService(settings, app.state.storage_service)
        await app.state.session_service.start_background_cleanup()
        logger.info("✅ Session Service initialized")
        
    except Exception as e:
        logger.error(f"Service initialization error: {e}")
        # Continue anyway for testing
    
    logger.info("="*60)
    logger.info("🌐 API Ready - http://0.0.0.0:8000")
    logger.info("📚 Docs - http://0.0.0.0:8000/docs")
    logger.info("⚠️  CORS: ALLOWING EVERYTHING")
    logger.info("⚠️  SECURITY: DISABLED")
    logger.info("="*60)
    
    yield
    
    logger.info("🛑 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Smart Blueprint API - UNSAFE TEST",
    description="⚠️ UNSAFE TEST VERSION - ALL SECURITY DISABLED",
    version="TEST",
    lifespan=lifespan
)

# CORS - ALLOW EVERYTHING - MUST BE FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=True,  # Allow credentials
    allow_methods=["*"],  # Allow ALL methods
    allow_headers=["*"],  # Allow ALL headers
    expose_headers=["*"],  # Expose ALL headers
    max_age=86400  # Cache for 24 hours
)

logger.info("⚠️  CORS: Allowing ALL origins, methods, and headers")

# Add custom middleware to ensure CORS headers on all responses
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Error during request: {e}")
        response = JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    # Always add CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# Global exception handler - always return details
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return full error details for debugging"""
    import traceback
    
    response = JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "path": str(request.url),
            "method": request.method
        }
    )
    
    # Add CORS headers to error responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# Include all routers
app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint"])
app.include_router(document_router, prefix="/api/v1", tags=["Documents"])
app.include_router(annotation_router, prefix="/api/v1", tags=["Annotations"])
app.include_router(note_router, prefix="/api/v1", tags=["Notes"])

# Root endpoint
@app.get("/")
async def root():
    """Test endpoint"""
    return {
        "status": "running",
        "mode": "UNSAFE_TEST",
        "cors": "ALLOW_ALL",
        "endpoints": {
            "upload": "POST /api/v1/documents/upload",
            "chat": "POST /api/v1/documents/{id}/chat",
            "health": "GET /health"
        }
    }

# Health check
@app.get("/health")
async def health():
    """Simple health check"""
    return {
        "status": "healthy",
        "mode": "UNSAFE_TEST",
        "services": {
            "storage": hasattr(app.state, 'storage_service'),
            "ai": hasattr(app.state, 'ai_service'),
            "pdf": hasattr(app.state, 'pdf_service'),
            "session": hasattr(app.state, 'session_service')
        },
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# Test CORS endpoint
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    """Handle OPTIONS requests for CORS"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# Debug endpoint
@app.get("/debug")
async def debug_info():
    """Show all debug information"""
    return {
        "settings": {
            "azure_storage": bool(settings.AZURE_STORAGE_CONNECTION_STRING),
            "openai_key": bool(settings.OPENAI_API_KEY),
            "environment": getattr(settings, 'ENVIRONMENT', 'unknown'),
            "debug": getattr(settings, 'DEBUG', False)
        },
        "services": {
            "storage": hasattr(app.state, 'storage_service'),
            "ai": hasattr(app.state, 'ai_service'),
            "pdf": hasattr(app.state, 'pdf_service'),
            "session": hasattr(app.state, 'session_service')
        },
        "routes": [route.path for route in app.routes],
        "cors": {
            "origins": "*",
            "methods": "*",
            "headers": "*",
            "credentials": True
        }
    }

# Test upload endpoint
@app.post("/test/upload")
async def test_upload():
    """Test endpoint to verify API is accessible"""
    return {
        "message": "Upload endpoint accessible",
        "cors": "enabled",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    logger.info("⚠️  Starting UNSAFE TEST server...")
    logger.info("⚠️  ALL SECURITY FEATURES DISABLED")
    logger.info("⚠️  DO NOT USE IN PRODUCTION")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        reload=True,  # Auto-reload on changes
        log_level="info",
        access_log=True
    )
