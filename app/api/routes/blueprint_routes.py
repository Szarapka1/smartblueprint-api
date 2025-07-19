# app/api/routes/blueprint_routes.py - OPTIMIZED WITH THUMBNAILS AND SINGLE HIGH-QUALITY JPEG

"""
Blueprint Analysis Routes - Optimized with thumbnails for fast preview and high-quality JPEGs for analysis
"""

import traceback
import uuid
import json
import logging
import os
import re
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Header, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

# Core imports
from app.core.config import get_settings

# Schema imports
from app.models.schemas import (
    ChatRequest, ChatResponse, DocumentUploadResponse, DocumentInfoResponse,
    DocumentListResponse, SuccessResponse, ErrorResponse, NoteSuggestion,
    QuickNoteCreate, UserPreferences, Note, VisualHighlight
)

# Initialize router and settings
blueprint_router = APIRouter(
    tags=["Blueprint Analysis"],
    responses={
        503: {"description": "Service temporarily unavailable"},
        500: {"description": "Internal server error"},
        413: {"description": "Payload too large"},
        425: {"description": "Too early - processing not complete"}
    }
)

settings = get_settings()
logger = logging.getLogger(__name__)

# Global processing state tracker (use Redis in production)
processing_state: Dict[str, Dict[str, Any]] = {}
processing_lock = asyncio.Lock()

# ===== CONSTANTS =====
PDF_HEADER_BYTES = [b'%PDF', b'\x25\x50\x44\x46']  # Standard and hex-encoded PDF headers
MAX_RETRIES = 3
RETRY_DELAY = 1.0
PROCESSING_STATUS_TTL = 86400  # 24 hours
CACHE_CONTROL_METADATA = "public, max-age=3600"  # 1 hour cache for metadata
CACHE_CONTROL_IMAGES = "public, max-age=86400"  # 24 hour cache for images
CACHE_CONTROL_THUMBNAILS = "public, max-age=604800"  # 7 day cache for thumbnails

# ===== VALIDATION FUNCTIONS =====

def validate_document_id(document_id: str) -> str:
    """
    Validate and sanitize document ID with enhanced security.
    """
    if not document_id or not isinstance(document_id, str):
        logger.warning(f"Invalid document ID type: {type(document_id)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Document ID must be a non-empty string"
        )
    
    # Allow alphanumeric, underscore, hyphen (UUID format)
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '', document_id.strip())
    
    if not clean_id or len(clean_id) < 3:
        logger.warning(f"Document ID too short after sanitization: '{clean_id}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Document ID must be at least 3 characters long"
        )
    
    if len(clean_id) > 50:
        logger.warning(f"Document ID too long: {len(clean_id)} characters")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID must be 50 characters or less"
        )
    
    return clean_id

def validate_pdf_content(content: bytes, filename: str) -> bool:
    """
    Thoroughly validate PDF content with multiple checks.
    """
    if not content:
        logger.error("Empty file content")
        return False
    
    # Check multiple PDF header formats
    header = content.lstrip()[:4]
    if not any(header.startswith(pdf_header) for pdf_header in PDF_HEADER_BYTES):
        logger.error(f"Invalid PDF header for {filename}: {header[:20]}")
        return False
    
    # Check for PDF end marker
    if not (b'%%EOF' in content[-1024:] or b'%%EOF\n' in content[-1024:] or 
            b'%%EOF\r\n' in content[-1024:] or b'%%EOF\r' in content[-1024:]):
        logger.warning(f"PDF {filename} missing EOF marker (non-critical)")
    
    return True

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename with enhanced security and Unicode support.
    """
    if not filename:
        return "unnamed.pdf"
    
    # Get base name without path
    filename = os.path.basename(filename)
    
    # Replace problematic characters but keep Unicode letters
    clean_name = re.sub(r'[^\w\s\-_\.]', '_', filename, flags=re.UNICODE)
    clean_name = re.sub(r'[\s]+', '_', clean_name)  # Replace spaces with underscore
    clean_name = re.sub(r'[_]+', '_', clean_name)   # Collapse multiple underscores
    
    # Ensure it ends with .pdf
    if not clean_name.lower().endswith('.pdf'):
        clean_name += '.pdf'
    
    # Limit length
    if len(clean_name) > 100:
        name_part = clean_name[:-4][:96]  # Keep extension
        clean_name = f"{name_part}.pdf"
    
    return clean_name

# ===== NEW COMPREHENSIVE DATA ENDPOINT WITH THUMBNAILS =====

@blueprint_router.get(
    "/documents/{document_id}/complete-data",
    summary="Get all document data in one request",
    description="Retrieve all processed document data including metadata, text, grids, thumbnails, and image URLs"
)
async def get_complete_document_data(
    request: Request,
    document_id: str,
    include_text: bool = True,
    include_tables: bool = True
):
    """
    Get all document data in a single comprehensive response.
    This eliminates the need for multiple API calls from the frontend.
    Includes both thumbnail and high-quality image URLs.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Check if document exists and is processed
        if not await storage_service.blob_exists(cache_container, f"{clean_document_id}_metadata.json"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{clean_document_id}' not found or not yet processed"
            )
        
        # Prepare response data
        response_data = {
            "document_id": clean_document_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # 1. Load metadata
        try:
            metadata_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_metadata.json"
            )
            response_data["metadata"] = json.loads(metadata_text)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            response_data["metadata"] = None
        
        # 2. Load grid systems
        try:
            grid_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_grid_systems.json"
            )
            response_data["grid_systems"] = json.loads(grid_text)
        except Exception:
            response_data["grid_systems"] = {}
        
        # 3. Load document index
        try:
            index_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_document_index.json"
            )
            response_data["document_index"] = json.loads(index_text)
        except Exception:
            response_data["document_index"] = None
        
        # 4. Load context text (optional)
        if include_text:
            try:
                context_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=f"{clean_document_id}_context.txt"
                )
                response_data["context_text"] = context_text
            except Exception:
                response_data["context_text"] = None
        
        # 5. Load tables (optional)
        if include_tables:
            try:
                tables_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=f"{clean_document_id}_tables.json"
                )
                response_data["tables"] = json.loads(tables_text)
            except Exception:
                response_data["tables"] = []
        
        # 6. Generate page image URLs (BOTH thumbnails and high-quality)
        page_count = response_data.get("metadata", {}).get("page_count", 0)
        base_url = f"https://{storage_service.blob_service_client.account_name}.blob.core.windows.net/{cache_container}"
        
        page_images = {}
        for page_num in range(1, page_count + 1):
            # Thumbnail and high-quality JPEG
            thumbnail_blob = f"{clean_document_id}_page_{page_num}_thumb.jpg"
            jpeg_blob = f"{clean_document_id}_page_{page_num}.jpg"
            
            page_images[str(page_num)] = {
                "thumbnail_url": f"{base_url}/{thumbnail_blob}",
                "jpeg_url": f"{base_url}/{jpeg_blob}",
                "thumbnail_blob": thumbnail_blob,
                "jpeg_blob": jpeg_blob
            }
        
        response_data["page_images"] = page_images
        
        # 7. Add processing summary
        response_data["summary"] = {
            "total_pages": page_count,
            "has_grids": bool(response_data.get("grid_systems")),
            "has_tables": bool(response_data.get("tables")),
            "has_thumbnails": response_data.get("metadata", {}).get("extraction_summary", {}).get("has_thumbnails", True),
            "drawing_types": list(response_data.get("document_index", {}).get("drawing_types", {}).keys()),
            "grid_pages": response_data.get("document_index", {}).get("grid_pages", []),
            "table_pages": response_data.get("document_index", {}).get("table_pages", []),
            "image_settings": response_data.get("metadata", {}).get("image_settings", {})
        }
        
        # Return with cache headers
        return JSONResponse(
            content=response_data,
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id,
                "X-Page-Count": str(page_count)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get complete document data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document data: {str(e)}"
        )

# ===== PAGE IMAGE ENDPOINTS WITH THUMBNAIL SUPPORT =====

@blueprint_router.get(
    "/documents/{document_id}/pages/{page_number}/thumbnail",
    summary="Get page thumbnail for quick preview",
    description="Retrieve the thumbnail JPEG for a specific page"
)
async def get_page_thumbnail(
    request: Request,
    document_id: str,
    page_number: int
):
    """
    Get the thumbnail JPEG for fast page preview.
    """
    clean_document_id = validate_document_id(document_id)
    
    if page_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be 1 or greater"
        )
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        blob_name = f"{clean_document_id}_page_{page_number}_thumb.jpg"
        
        # Check if thumbnail exists
        if not await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, blob_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thumbnail for page {page_number} not found"
            )
        
        # Download thumbnail
        image_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        # Return with aggressive caching for thumbnails
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": CACHE_CONTROL_THUMBNAILS,
                "Content-Disposition": f"inline; filename=\"{clean_document_id}_page_{page_number}_thumb.jpg\"",
                "X-Page-Number": str(page_number),
                "X-Document-ID": clean_document_id,
                "X-Image-Type": "thumbnail"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get page thumbnail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve page thumbnail: {str(e)}"
        )

@blueprint_router.get(
    "/documents/{document_id}/pages/{page_number}/image",
    summary="Get high-quality page image",
    description="Retrieve the high-quality JPEG for a specific page"
)
async def get_page_image(
    request: Request,
    document_id: str,
    page_number: int
):
    """
    Get the single high-quality JPEG for a page with proper caching headers.
    """
    clean_document_id = validate_document_id(document_id)
    
    if page_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be 1 or greater"
        )
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        blob_name = f"{clean_document_id}_page_{page_number}.jpg"
        
        # Check if image exists
        if not await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, blob_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page {page_number} not found for document {clean_document_id}"
            )
        
        # Download image
        image_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        # Return with caching headers
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": CACHE_CONTROL_IMAGES,
                "Content-Disposition": f"inline; filename=\"{clean_document_id}_page_{page_number}.jpg\"",
                "X-Page-Number": str(page_number),
                "X-Document-ID": clean_document_id,
                "X-Image-Type": "high_quality"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get page image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve page image: {str(e)}"
        )

@blueprint_router.get(
    "/documents/{document_id}/thumbnails/all",
    summary="Get all thumbnail URLs",
    description="Get URLs for all page thumbnails in the document"
)
async def get_all_thumbnail_urls(
    request: Request,
    document_id: str
):
    """
    Get URLs for all thumbnails - useful for preloading or gallery views.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        # Load metadata to get page count
        metadata = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        
        page_count = metadata.get('page_count', 0)
        base_url = f"https://{storage_service.blob_service_client.account_name}.blob.core.windows.net/{settings.AZURE_CACHE_CONTAINER_NAME}"
        
        thumbnails = []
        for page_num in range(1, page_count + 1):
            blob_name = f"{clean_document_id}_page_{page_num}_thumb.jpg"
            thumbnails.append({
                "page": page_num,
                "url": f"{base_url}/{blob_name}",
                "api_url": f"/api/v1/documents/{clean_document_id}/pages/{page_num}/thumbnail"
            })
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "total_pages": page_count,
                "thumbnails": thumbnails,
                "thumbnail_settings": metadata.get('image_settings', {}).get('thumbnail', {
                    'dpi': 72,
                    'quality': 75
                })
            },
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get thumbnail URLs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve thumbnail URLs: {str(e)}"
        )

# ===== ASYNC PROCESSING WITH ERROR RECOVERY =====

async def process_pdf_with_retry(
    session_id: str,
    contents: bytes,
    clean_filename: str,
    author: str,
    storage_service,
    pdf_service,
    session_service,
    max_retries: int = MAX_RETRIES
):
    """
    Process PDF with automatic retry and comprehensive error handling.
    """
    start_time = time.time()
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            logger.info(f"🔄 Processing attempt {retry_count + 1}/{max_retries} for {session_id}")
            
            # Update processing state
            async with processing_lock:
                processing_state[session_id] = {
                    'status': 'processing',
                    'attempt': retry_count + 1,
                    'started_at': datetime.utcnow().isoformat(),
                    'filename': clean_filename,
                    'author': author,
                    'progress': 0
                }
            
            # Update status in storage
            await update_processing_status(
                storage_service, session_id, 'processing',
                {'attempt': retry_count + 1, 'progress': 0}
            )
            
            # Process the PDF with progress tracking
            await pdf_service.process_and_cache_pdf(
                session_id=session_id,
                pdf_bytes=contents,
                storage_service=storage_service
            )
            
            # Success! Update status
            elapsed_time = time.time() - start_time
            
            # Get processing results
            metadata = await get_document_metadata(storage_service, session_id)
            
            await update_processing_status(
                storage_service, session_id, 'ready',
                {
                    'processing_time': elapsed_time,
                    'pages_processed': metadata.get('page_count', 0),
                    'total_pages': metadata.get('total_pages', 0),
                    'grid_systems_detected': metadata.get('grid_systems_detected', 0),
                    'has_tables': metadata.get('extraction_summary', {}).get('has_tables', False),
                    'has_thumbnails': metadata.get('extraction_summary', {}).get('has_thumbnails', True)
                }
            )
            
            # Update session if available
            if session_service:
                try:
                    await session_service.update_session_metadata(
                        document_id=session_id,
                        metadata={
                            'processing_complete': True,
                            'status': 'ready',
                            'processing_time': elapsed_time,
                            'pages_processed': metadata.get('page_count', 0),
                            'has_thumbnails': True
                        }
                    )
                except Exception as e:
                    logger.warning(f"Session update failed (non-critical): {e}")
            
            # Clean up processing state
            async with processing_lock:
                processing_state.pop(session_id, None)
            
            logger.info(f"✅ Processing completed for {session_id} in {elapsed_time:.2f}s")
            return True
            
        except asyncio.CancelledError:
            logger.warning(f"Processing cancelled for {session_id}")
            raise
        except Exception as e:
            retry_count += 1
            last_error = e
            logger.error(f"❌ Processing attempt {retry_count} failed for {session_id}: {e}")
            logger.error(traceback.format_exc())
            
            if retry_count < max_retries:
                delay = RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                logger.info(f"⏰ Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                # Final failure
                await handle_processing_failure(
                    storage_service, session_id, clean_filename, 
                    author, str(last_error), traceback.format_exc()
                )
                
                async with processing_lock:
                    processing_state.pop(session_id, None)
                
                return False

async def update_processing_status(
    storage_service, session_id: str, status: str, 
    additional_data: Dict[str, Any] = None
):
    """Update processing status in storage."""
    try:
        status_data = {
            'document_id': session_id,
            'status': status,
            'updated_at': datetime.utcnow().isoformat() + 'Z',
            **(additional_data or {})
        }
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(status_data, indent=2).encode('utf-8'),
            content_type="application/json"
        )
    except Exception as e:
        logger.error(f"Failed to update status for {session_id}: {e}")

async def handle_processing_failure(
    storage_service, session_id: str, filename: str,
    author: str, error: str, traceback_str: str
):
    """Handle processing failure with detailed error logging."""
    error_data = {
        'document_id': session_id,
        'status': 'error',
        'error': error,
        'traceback': traceback_str,
        'filename': filename,
        'author': author,
        'failed_at': datetime.utcnow().isoformat() + 'Z',
        'retry_available': True
    }
    
    try:
        # Save error status
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(error_data, indent=2).encode('utf-8'),
            content_type="application/json"
        )
        
        # Save detailed error log
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_error_log.json",
            data=json.dumps(error_data, indent=2).encode('utf-8'),
            content_type="application/json"
        )
    except Exception as e:
        logger.critical(f"Failed to save error status: {e}")

async def get_document_metadata(storage_service, session_id: str) -> Dict[str, Any]:
    """Get document metadata if available."""
    try:
        metadata_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_metadata.json"
        )
        return json.loads(metadata_text)
    except:
        return {}

# ===== MAIN UPLOAD ENDPOINT =====

@blueprint_router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a blueprint PDF",
    description="Upload a construction blueprint PDF for AI analysis. Supports files up to 100MB."
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload"),
    author: Optional[str] = Form(default="Anonymous", description="Name of the person uploading"),
    trade: Optional[str] = Form(default=None, description="Trade/discipline associated with the document")
):
    """
    Upload a PDF document for processing with enhanced validation and error handling.
    Returns immediately with processing status, processes asynchronously.
    """
    upload_start = time.time()
    
    # Validate file presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file provided"
        )
    
    # Validate file type by extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # Sanitize inputs
    clean_filename = sanitize_filename(file.filename)
    author = re.sub(r'[^\w\s\-_@.]', '', author)[:100]  # Clean author name
    
    logger.info(f"📤 Upload request: {clean_filename} by {author}")
    
    # Read file content with size tracking
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        logger.info(f"📊 File stats: {clean_filename} - {file_size_mb:.2f}MB")
        
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file. Please try again."
        )
    
    # Validate file size
    max_size_mb = settings.MAX_FILE_SIZE_MB
    if file_size_mb > max_size_mb:
        logger.warning(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum size is {max_size_mb}MB."
        )
    
    # Validate PDF content
    if not validate_pdf_content(contents, clean_filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file. The file appears to be corrupted or is not a valid PDF."
        )
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    logger.info(f"📋 Created session: {session_id} for {clean_filename}")
    
    # Get services
    storage_service = request.app.state.storage_service
    if not storage_service:
        logger.error("Storage service not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service is temporarily unavailable. Please try again later."
        )
    
    try:
        # Upload original PDF first (quick operation)
        pdf_blob_name = f"{session_id}.pdf"
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=pdf_blob_name,
            data=contents,
            content_type="application/pdf"
        )
        
        upload_time = time.time() - upload_start
        logger.info(f"✅ PDF uploaded in {upload_time:.2f}s: {pdf_blob_name}")
        
        # Create initial processing status
        estimated_time = max(30, int(file_size_mb * 3))  # 3 seconds per MB estimate
        
        initial_status = {
            'document_id': session_id,
            'status': 'uploaded',
            'filename': clean_filename,
            'author': author,
            'trade': trade,
            'file_size_mb': round(file_size_mb, 2),
            'uploaded_at': datetime.utcnow().isoformat() + 'Z',
            'estimated_processing_time': estimated_time
        }
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(initial_status, indent=2).encode('utf-8'),
            content_type="application/json"
        )
        
        # Initialize session if service available
        session_service = request.app.state.session_service
        if session_service:
            try:
                await session_service.create_session(
                    document_id=session_id,
                    original_filename=clean_filename
                )
                logger.info("✅ Session created")
            except Exception as e:
                logger.warning(f"Session creation failed (non-critical): {e}")
        
        # Check if PDF processing is available
        pdf_service = request.app.state.pdf_service
        if not pdf_service:
            logger.warning("PDF service not available - upload only mode")
            return DocumentUploadResponse(
                document_id=session_id,
                filename=clean_filename,
                status="uploaded",
                message="Document uploaded successfully. Processing service unavailable - please check back later.",
                file_size_mb=round(file_size_mb, 2)
            )
        
        # Start async processing
        processing_task = asyncio.create_task(
            process_pdf_with_retry(
                session_id=session_id,
                contents=contents,
                clean_filename=clean_filename,
                author=author,
                storage_service=storage_service,
                pdf_service=pdf_service,
                session_service=session_service
            )
        )
        
        # Don't await - let it run in background
        processing_task.add_done_callback(
            lambda t: logger.info(f"Processing task completed for {session_id}")
        )
        
        # Return success immediately
        return DocumentUploadResponse(
            document_id=session_id,
            filename=clean_filename,
            status="processing",
            message=f"Document uploaded successfully! Processing will take approximately {estimated_time} seconds. Thumbnails will be generated for quick preview.",
            file_size_mb=round(file_size_mb, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to clean up
        try:
            if 'session_id' in locals():
                await storage_service.delete_blob(
                    settings.AZURE_CONTAINER_NAME, 
                    f"{session_id}.pdf"
                )
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Upload failed: {str(e)}"
        )

# ===== STATUS CHECK ENDPOINT =====

@blueprint_router.get(
    "/documents/{document_id}/status",
    summary="Check document processing status",
    description="Check if a document has finished processing"
)
async def check_document_status(
    request: Request,
    document_id: str
):
    """
    Check the processing status of a document with detailed progress information.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        # Check in-memory processing state first
        async with processing_lock:
            if clean_document_id in processing_state:
                state = processing_state[clean_document_id]
                return JSONResponse(
                    content={
                        "document_id": clean_document_id,
                        "status": state.get('status', 'processing'),
                        "progress": state.get('progress', 0),
                        "message": f"Processing... Attempt {state.get('attempt', 1)}"
                    },
                    headers={"Cache-Control": "no-cache"}
                )
        
        # Check persisted status
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        status_blob = f"{clean_document_id}_status.json"
        
        if await storage_service.blob_exists(cache_container, status_blob):
            status_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=status_blob
            )
            status_data = json.loads(status_text)
            
            # Add time calculations for processing status
            if status_data.get('status') == 'processing':
                if 'started_at' in status_data:
                    started = datetime.fromisoformat(status_data['started_at'].rstrip('Z'))
                    elapsed = (datetime.utcnow() - started).total_seconds()
                    estimated_total = status_data.get('estimated_processing_time', 60)
                    remaining = max(0, int(estimated_total - elapsed))
                    status_data['estimated_time_remaining'] = remaining
                    status_data['progress_percentage'] = min(100, int((elapsed / estimated_total) * 100))
            
            return JSONResponse(
                content={
                    "document_id": clean_document_id,
                    "status": status_data.get('status', 'unknown'),
                    "message": status_data.get('message', ''),
                    "pages_processed": status_data.get('pages_processed'),
                    "total_pages": status_data.get('total_pages'),
                    "has_tables": status_data.get('has_tables', False),
                    "has_thumbnails": status_data.get('has_thumbnails', True),
                    "error": status_data.get('error'),
                    "uploaded_at": status_data.get('uploaded_at'),
                    "completed_at": status_data.get('completed_at'),
                    "estimated_time_remaining": status_data.get('estimated_time_remaining'),
                    "progress_percentage": status_data.get('progress_percentage'),
                    "retry_available": status_data.get('retry_available', False)
                },
                headers={"Cache-Control": "no-cache"}
            )
        
        # Fallback checks
        context_blob = f"{clean_document_id}_context.txt"
        if await storage_service.blob_exists(cache_container, context_blob):
            return JSONResponse(
                content={
                    "document_id": clean_document_id,
                    "status": "ready",
                    "message": "Document is ready for analysis",
                    "has_thumbnails": True
                },
                headers={"Cache-Control": CACHE_CONTROL_METADATA}
            )
        
        main_container = settings.AZURE_CONTAINER_NAME
        if await storage_service.blob_exists(main_container, f"{clean_document_id}.pdf"):
            return JSONResponse(
                content={
                    "document_id": clean_document_id,
                    "status": "uploaded",
                    "message": "Document uploaded, processing pending"
                },
                headers={"Cache-Control": "no-cache"}
            )
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "status": "not_found",
                "message": "Document not found"
            },
            status_code=status.HTTP_404_NOT_FOUND
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check status: {str(e)}"
        )

# ===== DOWNLOAD ENDPOINT =====

@blueprint_router.get("/documents/{document_id}/download")
async def download_document_pdf(
    request: Request,
    document_id: str
):
    """
    Download the original PDF file for viewing with streaming support.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        container_name = settings.AZURE_CONTAINER_NAME
        blob_name = f"{clean_document_id}.pdf"
        
        # Check existence first
        if not await storage_service.blob_exists(container_name, blob_name):
            logger.warning(f"PDF not found: {blob_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {clean_document_id} not found"
            )
        
        # Get PDF data
        logger.info(f"📥 Downloading PDF: {blob_name}")
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=container_name,
            blob_name=blob_name
        )
        
        logger.info(f"✅ PDF downloaded: {len(pdf_bytes)} bytes")
        
        # Return with proper headers for browser viewing
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename=\"{clean_document_id}.pdf\"",
                "Content-Length": str(len(pdf_bytes)),
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for {clean_document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Download failed: {str(e)}"
        )

# ===== CHAT ENDPOINT =====

@blueprint_router.post(
    "/documents/{document_id}/chat",
    response_model=ChatResponse,
    summary="Chat with a blueprint",
    description="Send a question about an uploaded blueprint and receive AI analysis"
)
async def chat_with_document(
    request: Request,
    document_id: str,
    chat_request: ChatRequest
):
    """
    Chat with an uploaded document using AI analysis.
    Enhanced with better error handling and status checking.
    """
    clean_document_id = validate_document_id(document_id)
    
    # Validate request
    if not chat_request.prompt or len(chat_request.prompt.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Prompt cannot be empty"
        )
    
    # Clean inputs
    prompt = chat_request.prompt.strip()[:2000]  # Enforce limit
    author = re.sub(r'[^\w\s\-_@.]', '', chat_request.author)[:100]
    
    # Get services
    ai_service = request.app.state.ai_service
    storage_service = request.app.state.storage_service
    
    if not ai_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI service is not available. Please try again later."
        )
    
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service is not available. Please try again later."
        )
    
    try:
        logger.info(f"💬 Chat request for {clean_document_id} from {author}")
        logger.info(f"   Prompt: {prompt[:100]}...")
        
        # Check document readiness with detailed status
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Check status file for detailed info
        status_blob = f"{clean_document_id}_status.json"
        if await storage_service.blob_exists(cache_container, status_blob):
            status_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=status_blob
            )
            status_data = json.loads(status_text)
            
            if status_data.get('status') == 'processing':
                # Calculate time remaining
                estimated_remaining = 30
                if status_data.get('started_at'):
                    started = datetime.fromisoformat(status_data['started_at'].rstrip('Z'))
                    elapsed = (datetime.utcnow() - started).total_seconds()
                    estimated_total = status_data.get('estimated_processing_time', 60)
                    estimated_remaining = max(5, int(estimated_total - elapsed))
                
                raise HTTPException(
                    status_code=status.HTTP_425_TOO_EARLY,
                    detail=f"Document is still processing. Estimated time remaining: {estimated_remaining} seconds.",
                    headers={"Retry-After": str(estimated_remaining)}
                )
            
            elif status_data.get('status') == 'error':
                # Check if retry is available
                if status_data.get('retry_available', False):
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Document processing failed but can be retried. Please re-upload the document."
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Document processing failed: {status_data.get('error', 'Unknown error')}"
                    )
        
        # Verify document is ready
        context_blob = f"{clean_document_id}_context.txt"
        if not await storage_service.blob_exists(cache_container, context_blob):
            # Check if PDF exists
            main_container = settings.AZURE_CONTAINER_NAME
            if await storage_service.blob_exists(main_container, f"{clean_document_id}.pdf"):
                raise HTTPException(
                    status_code=status.HTTP_425_TOO_EARLY,
                    detail="Document is pending processing. Please try again in a few moments.",
                    headers={"Retry-After": "30"}
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{clean_document_id}' not found"
                )
        
        # Process AI request with timeout
        try:
            ai_result = await asyncio.wait_for(
                ai_service.get_ai_response(
                    prompt=prompt,
                    document_id=clean_document_id,
                    storage_service=storage_service,
                    author=author,
                    current_page=chat_request.current_page,
                    request_highlights=True,
                    reference_previous=chat_request.reference_previous,
                    preserve_existing=chat_request.preserve_existing,
                    show_trade_info=chat_request.show_trade_info,
                    detect_conflicts=chat_request.detect_conflicts,
                    auto_suggest_notes=chat_request.auto_suggest_notes,
                    note_suggestion_threshold=chat_request.note_suggestion_threshold
                ),
                timeout=settings.VISION_REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"AI request timeout for {clean_document_id}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="AI analysis timed out. Please try with a simpler question or specific page reference."
            )
        
        # Save chat history
        await save_chat_to_history(
            document_id=clean_document_id,
            author=author,
            prompt=prompt,
            response=ai_result.get("ai_response", ""),
            storage_service=storage_service,
            metadata={
                "has_highlights": bool(ai_result.get("visual_highlights")),
                "note_suggested": bool(ai_result.get("note_suggestion")),
                "processing_time": ai_result.get("processing_time"),
                "pages_analyzed": len(ai_result.get("source_pages", []))
            }
        )
        
        # Update session activity
        session_service = request.app.state.session_service
        if session_service:
            try:
                await session_service.record_chat_activity(
                    document_id=clean_document_id,
                    user=author,
                    chat_data={
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "author": author,
                        "prompt": prompt[:200],
                        "has_response": bool(ai_result.get("ai_response"))
                    }
                )
            except Exception as e:
                logger.warning(f"Session activity update failed: {e}")
        
        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_result.get("ai_response", "I was unable to analyze the blueprint. Please try rephrasing your question."),
            visual_highlights=ai_result.get("visual_highlights"),
            current_page=chat_request.current_page,
            query_session_id=ai_result.get("query_session_id"),
            all_highlight_pages=ai_result.get("all_highlight_pages"),
            trade_summary=ai_result.get("trade_summary"),
            note_suggestion=ai_result.get("note_suggestion")
        )
        
        logger.info(f"✅ Chat response generated successfully")
        if ai_result.get("visual_highlights"):
            logger.info(f"   Highlights: {len(ai_result['visual_highlights'])}")
        if ai_result.get("note_suggestion"):
            logger.info(f"   Note suggested: {ai_result['note_suggestion'].id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Provide user-friendly error message
        error_message = "An unexpected error occurred while processing your question."
        if "timeout" in str(e).lower():
            error_message = "The analysis took too long. Please try a more specific question."
        elif "memory" in str(e).lower():
            error_message = "The document is too large to analyze. Please specify a page number."
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=error_message
        )

# ===== QUICK NOTE CREATION =====

@blueprint_router.post("/documents/{document_id}/notes/quick-create")
async def quick_create_note_from_suggestion(
    request: Request,
    document_id: str,
    quick_note: QuickNoteCreate
):
    """
    Quick endpoint to create a note from AI suggestion with validation.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        notes_blob = f"{clean_document_id}_notes.json"
        
        # Load existing notes with error handling
        all_notes = []
        try:
            notes_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=notes_blob
            )
            all_notes = json.loads(notes_text)
        except FileNotFoundError:
            logger.info(f"Creating first note for document {clean_document_id}")
        except json.JSONDecodeError:
            logger.error(f"Corrupted notes file for {clean_document_id}, starting fresh")
            all_notes = []
        
        # Validate note limits
        max_notes = settings.MAX_NOTES_PER_DOCUMENT
        user_notes = [n for n in all_notes if n.get('author') == quick_note.author]
        
        if len(user_notes) >= max_notes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Note limit reached. Maximum {max_notes} notes per user per document."
            )
        
        # Calculate total character count
        total_chars = sum(len(n.get('text', '')) for n in all_notes)
        if total_chars + len(quick_note.text) > settings.MAX_TOTAL_NOTE_CHARS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document note character limit exceeded."
            )
        
        # Create note with validation
        new_note = {
            "note_id": f"note_{uuid.uuid4().hex[:8]}",
            "document_id": clean_document_id,
            "text": quick_note.text[:settings.MAX_NOTE_LENGTH],
            "note_type": quick_note.note_type.value,
            "author": quick_note.author,
            "impacts_trades": quick_note.impacts_trades[:10],  # Limit trades
            "priority": quick_note.priority.value,
            "is_private": quick_note.is_private,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "char_count": len(quick_note.text),
            "status": "open",
            "ai_suggested": quick_note.ai_suggested,
            "suggestion_confidence": quick_note.suggestion_confidence,
            "related_query_sessions": quick_note.related_query_sessions[:10],
            "related_element_ids": quick_note.related_highlights[:20],
            "source_pages": sorted(list(set(quick_note.source_pages)))[:50]
        }
        
        # Add to notes
        all_notes.append(new_note)
        
        # Save atomically
        await storage_service.upload_file(
            container_name=cache_container,
            blob_name=notes_blob,
            data=json.dumps(all_notes, indent=2).encode('utf-8'),
            content_type="application/json"
        )
        
        logger.info(f"✅ Note created: {new_note['note_id']} (AI: {quick_note.ai_suggested})")
        
        return {
            "status": "success",
            "note_id": new_note["note_id"],
            "message": "Note created successfully",
            "note_details": {
                "type": quick_note.note_type.value,
                "priority": quick_note.priority.value,
                "is_private": quick_note.is_private,
                "ai_suggested": quick_note.ai_suggested
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Note creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to create note. Please try again."
        )

# ===== DOCUMENT INFO ENDPOINT =====

@blueprint_router.get(
    "/documents/{document_id}/info",
    response_model=DocumentInfoResponse,
    summary="Get document information",
    description="Get processing status and metadata for a specific document"
)
async def get_document_info(request: Request, document_id: str):
    """
    Get comprehensive information about a document including collaboration stats.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        main_container = settings.AZURE_CONTAINER_NAME
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Check document existence
        pdf_exists = await storage_service.blob_exists(
            container_name=main_container,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        context_exists = await storage_service.blob_exists(
            container_name=cache_container,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not pdf_exists and not context_exists:
            return DocumentInfoResponse(
                document_id=clean_document_id,
                status="not_found",
                message="Document not found",
                exists=False
            )
        
        # Get status information
        status_data = None
        status_blob = f"{clean_document_id}_status.json"
        if await storage_service.blob_exists(cache_container, status_blob):
            status_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=status_blob
            )
            status_data = json.loads(status_text)
        
        # Get metadata if processed
        metadata = None
        if context_exists:
            metadata = await get_document_metadata(storage_service, clean_document_id)
        
        # Get collaboration statistics
        published_notes = 0
        active_collaborators = set()
        recent_activity = False
        
        try:
            # Count public notes
            notes_blob = f"{clean_document_id}_notes.json"
            if await storage_service.blob_exists(cache_container, notes_blob):
                notes_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=notes_blob
                )
                all_notes = json.loads(notes_text)
                
                for note in all_notes:
                    if not note.get('is_private', True):
                        published_notes += 1
                        active_collaborators.add(note.get('author'))
                        
                        # Check for recent activity (last 24 hours)
                        if note.get('timestamp'):
                            note_time = datetime.fromisoformat(note['timestamp'].rstrip('Z'))
                            if (datetime.utcnow() - note_time).total_seconds() < 86400:
                                recent_activity = True
        except Exception as e:
            logger.warning(f"Failed to load collaboration stats: {e}")
        
        # Determine overall status
        if status_data:
            status_value = status_data.get('status', 'unknown')
            message = status_data.get('message', '')
            
            if status_value == 'error':
                message = f"Processing failed: {status_data.get('error', 'Unknown error')}"
            elif status_value == 'processing':
                message = "Document is being processed"
            elif status_value == 'ready':
                page_count = metadata.get('page_count', 0) if metadata else 0
                message = f"Document ready for analysis. {page_count} pages processed with thumbnails."
        else:
            if context_exists and metadata:
                status_value = "ready"
                message = f"Document ready. {metadata.get('page_count', 0)} pages available."
            elif pdf_exists:
                status_value = "uploaded"
                message = "Document uploaded, processing pending"
            else:
                status_value = "partial"
                message = "Document partially processed"
        
        # Get session info if available
        session_info = None
        session_service = request.app.state.session_service
        if session_service:
            try:
                session_info = await session_service.get_session_info(clean_document_id)
            except:
                pass
        
        # Include image settings info
        image_info = None
        if metadata:
            image_info = metadata.get('image_settings', {
                'high_quality': {
                    'dpi': 150,
                    'quality': 90,
                    'progressive': True
                },
                'thumbnail': {
                    'dpi': 72,
                    'quality': 75
                }
            })
        
        return DocumentInfoResponse(
            document_id=clean_document_id,
            status=status_value,
            message=message,
            exists=True,
            metadata={
                "filename": status_data.get('filename') if status_data else None,
                "author": status_data.get('author') if status_data else None,
                "uploaded_at": status_data.get('uploaded_at') if status_data else None,
                "file_size_mb": status_data.get('file_size_mb') if status_data else None,
                "page_count": metadata.get('page_count') if metadata else None,
                "total_pages": metadata.get('total_pages') if metadata else None,
                "processing_time": metadata.get('processing_time') if metadata else None,
                "grid_systems_detected": metadata.get('grid_systems_detected', 0) if metadata else None,
                "has_text": metadata.get('extraction_summary', {}).get('has_text', False) if metadata else None,
                "has_tables": metadata.get('extraction_summary', {}).get('has_tables', False) if metadata else None,
                "has_thumbnails": metadata.get('extraction_summary', {}).get('has_thumbnails', True) if metadata else None,
                "drawing_types": list(metadata.get('drawing_types', {}).keys()) if metadata else None,
                "image_settings": image_info
            },
            total_published_notes=published_notes,
            active_collaborators=len(active_collaborators),
            recent_public_activity=recent_activity,
            session_info=session_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to retrieve document information"
        )

# ===== SERVICE HEALTH CHECK =====

@blueprint_router.get("/health/services", include_in_schema=False)
async def check_services(request: Request):
    """
    Check health status of all services.
    """
    def check_service(service_name: str) -> Dict[str, Any]:
        """Check if a service exists and get its status"""
        service = getattr(request.app.state, service_name, None)
        
        if service is None:
            return {
                "available": False, 
                "status": "not_initialized"
            }
        
        # Get service-specific health info
        health_info = {"available": True, "status": "healthy"}
        
        try:
            if hasattr(service, 'get_connection_info'):
                health_info["connection_info"] = service.get_connection_info()
            elif hasattr(service, 'get_service_statistics'):
                health_info["statistics"] = service.get_service_statistics()
            elif hasattr(service, 'is_running'):
                health_info["is_running"] = service.is_running()
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
        
        return health_info
    
    # Check processing tasks
    async with processing_lock:
        active_processing = len(processing_state)
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "storage": check_service('storage_service'),
            "ai": check_service('ai_service'),
            "pdf": check_service('pdf_service'),
            "session": check_service('session_service')
        },
        "processing": {
            "active_tasks": active_processing,
            "task_ids": list(processing_state.keys()) if active_processing > 0 else []
        },
        "configuration": {
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "pdf_max_pages": settings.PDF_MAX_PAGES,
            "unlimited_loading": settings.UNLIMITED_PAGE_LOADING,
            "image_generation": {
                "thumbnails_enabled": True,
                "high_quality_enabled": True,
                "thumbnail_dpi": settings.PDF_THUMBNAIL_DPI,
                "high_quality_dpi": 150
            }
        }
    }

# ===== HELPER FUNCTIONS =====

async def save_chat_to_history(
    document_id: str,
    author: str,
    prompt: str,
    response: str,
    storage_service,
    metadata: Dict[str, Any] = None
):
    """
    Save chat interaction to history with metadata.
    """
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        chat_blob = f"{document_id}_all_chats.json"
        
        # Load existing history
        chat_history = []
        try:
            chat_data = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=chat_blob
            )
            chat_history = json.loads(chat_data)
        except:
            pass
        
        # Create chat entry
        chat_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "author": author,
            "prompt": prompt[:500],  # Limit prompt size
            "response": response[:1000],  # Limit response size
            "metadata": metadata or {}
        }
        
        chat_history.append(chat_entry)
        
        # Keep reasonable history size
        max_chats = settings.MAX_CHAT_LOGS
        if len(chat_history) > max_chats:
            chat_history = chat_history[-max_chats:]
        
        # Save updated history
        await storage_service.upload_file(
            container_name=cache_container,
            blob_name=chat_blob,
            data=json.dumps(chat_history, indent=2).encode('utf-8'),
            content_type="application/json"
        )
        
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")
        # Non-critical, don't fail the request

# === END OF FILE ===