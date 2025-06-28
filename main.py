# main.py - SAFE FOR LOCAL DEVELOPMENT & TESTING
# ⚠️ WARNING: This version is configured for local development.
# It has open CORS policies and verbose error reporting.
# DO NOT deploy this version to a production environment.

import os
import logging
import uvicorn
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# --- App Modules (using the same namesakes) ---
from app.core.config import get_settings
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService
from app.services.ai_service import ProfessionalBlueprintAI
from app.services.session_service import SessionService

# --- API Routers (using the same namesakes) ---
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router
from app.api.routes.note_routes import note_router

# --- Application Setup ---

# 1. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBlueprintAPI-Dev")

# 2. Load application settings
settings = get_settings()

# 3. Define the application lifecycle (with robust startup)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.
    This corrected version ensures all services are initialized before serving requests.
    """
    logger.info("="*60)
    logger.info("🚀 Initializing Smart Blueprint API for LOCAL DEVELOPMENT...")
    logger.info("="*60)

    # Initialize and attach services to the application state.
    # The app will fail to start if a critical setting (e.g., connection string) is missing.
    storage_service = StorageService(settings)
    session_service = SessionService(settings, storage_service)
    
    app.state.storage_service = storage_service
    logger.info("✅ Storage Service initialized.")
    
    app.state.ai_service = ProfessionalBlueprintAI(settings)
    logger.info("✅ AI Service initialized.")
    
    app.state.pdf_service = PDFService(settings)
    logger.info("✅ PDF Service initialized.")
    
    app.state.session_service = session_service
    await app.state.session_service.start_background_cleanup()
    logger.info("✅ Session Service initialized with background cleanup task.")
    
    logger.info("="*60)
    logger.info("✅ All services initialized. API is ready.")
    logger.info(f"📚 API Docs available at http://localhost:{settings.PORT}/docs")
    logger.info("="*60)

    yield  # Application is now running

    # --- Shutdown Logic ---
    logger.info("="*60)
    logger.info("🛑 Shutting down API...")
    if hasattr(app.state, 'session_service') and app.state.session_service.is_running():
        await app.state.session_service.stop_background_cleanup()
        logger.info("✅ Session Service background task stopped.")
    logger.info("✅ Shutdown complete.")

# 4. Create the FastAPI application instance
app = FastAPI(
    title="Smart Blueprint API (Local Test Version)",
    description="An API for intelligent analysis of blueprints. This version has relaxed security for local development.",
    version="1.0.0-dev",
    lifespan=lifespan,
    docs_url="/docs",  # Ensure docs are always on for testing
    redoc_url="/redoc"
)

# 5. Configure WIDE OPEN CORS for local development
# This allows your local frontend to connect to the local backend.
logger.info("⚠️ CORS: Allowing ALL origins, methods, and headers for local testing.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DANGEROUS IN PRODUCTION
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Add a VERBOSE global exception handler for easy debugging
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    """
    Handles any un-caught exception and returns a detailed
    traceback in the response. ONLY FOR DEVELOPMENT.
    """
    logger.error(f"Unhandled exception for request {request.method} {request.url}:")
    logger.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc().splitlines(),
            "path": str(request.url),
        }
    )

# 7. Include API routers
app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint"])
app.include_router(document_router, prefix="/api/v1", tags=["Documents"])
app.include_router(annotation_router, prefix="/api/v1", tags=["Annotations"])
app.include_router(note_router, prefix="/api/v1", tags=["Notes"])

# 8. Define root and health check endpoints
@app.get("/", tags=["Root"])
async def root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Smart Blueprint API (Local Test Version) is running."}

@app.get("/health", tags=["Health"])
async def health_check():
    """Provides a simple health check of the API and its services."""
    return {
        "status": "healthy",
        "mode": "local_testing",
        "services": {
            "storage": "ok" if hasattr(app.state, 'storage_service') else "error",
            "ai": "ok" if hasattr(app.state, 'ai_service') else "error",
            "pdf": "ok" if hasattr(app.state, 'pdf_service') else "error",
            "session": "ok" if hasattr(app.state, 'session_service') else "error",
        }
    }

# 9. Main entry point to run the server for local development
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on http://localhost:{settings.PORT}")
    logger.info("Auto-reload is enabled.")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
