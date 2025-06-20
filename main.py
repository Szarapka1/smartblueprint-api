import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# App config and services
from app.core.config import get_settings
from app.services.storage_service import StorageService
from app.services.pdf_service import PDFService
from app.services.ai_service import AIService
from app.services.session_service import SessionService

# API routes
from app.api.routes.blueprint_routes import blueprint_router
from app.api.routes.document_routes import document_router
from app.api.routes.annotation_routes import annotation_router

# --- Init logging & settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartBlueprintAPI")
settings = get_settings()

# --- Lifecycle hooks ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Smart Blueprint API...")

    try:
        # Init services
        app.state.storage_service = StorageService(settings)
        await app.state.storage_service.verify_connection()
        logger.info("✅ Azure Blob Storage connected")

        app.state.pdf_service = PDFService(settings)
        logger.info("✅ PDF Processor ready")

        app.state.ai_service = AIService(settings)
        logger.info("✅ OpenAI Service ready")

        app.state.session_service = SessionService(settings)
        logger.info("✅ Session service active")

        await verify_system_health(app)
        logger.info("✅ All systems verified. Startup successful.")
    except Exception as e:
        logger.critical(f"❌ Startup failed: {e}")
        raise

    yield
    logger.info("🛑 Shutting down gracefully...")

# --- Health Check Helper ---
async def verify_system_health(app: FastAPI):
    try:
        main_blobs = await app.state.storage_service.list_blobs(settings.AZURE_CONTAINER_NAME)
        cache_blobs = await app.state.storage_service.list_blobs(settings.AZURE_CACHE_CONTAINER_NAME)
        logger.info(f"✅ Blob containers: {len(main_blobs)} main / {len(cache_blobs)} cache")
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise

# --- Create FastAPI app ---
app = FastAPI(
    title="Smart Blueprint Chat API",
    description="Collaborative chat, annotation, and analysis for construction blueprints.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Server Error",
            "detail": str(exc),
            "url": str(request.url)
        }
    )

# --- Register routes ---
app.include_router(blueprint_router, prefix="/api/v1", tags=["Blueprint Chat"])
app.include_router(document_router, prefix="/api/v1", tags=["Documents"])
app.include_router(annotation_router, prefix="/api/v1", tags=["Annotations"])

# --- Public endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to Smart Blueprint Chat API",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["General"])
async def health():
    return {"status": "healthy"}

@app.get("/api/v1/system/info", tags=["General"])
async def system_info():
    return {
        "project": settings.PROJECT_NAME,
        "version": "2.0.0",
        "features": [
            "Blueprint Uploads", "AI Chat", "Annotations",
            "Blob Storage Integration", "Team Collaboration"
        ]
    }

# --- Local development server ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
