# app/api/routes/blueprint_routes.py - WRITE OPERATIONS WITH SSE SUPPORT

"""
Blueprint Write Operations - Handles document upload, processing, chat, and annotations
Now includes Server-Sent Events (SSE) for real-time status updates
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
from typing import Optional, Dict, List, Any, Union, AsyncGenerator
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from collections import defaultdict

# Core imports
from app.core.config import get_settings

# Schema imports
from app.models.schemas import (
    ChatRequest, ChatResponse, DocumentUploadResponse,
    SuccessResponse, ErrorResponse, NoteSuggestion,
    QuickNoteCreate, UserPreferences, Note,
    SSEEventType, SSEEvent, StatusUpdateEvent, ResourceReadyEvent,
    ProcessingCompleteEvent, ProcessingErrorEvent
)

# Initialize router and settings
blueprint_router = APIRouter(
    prefix="/api/v1",
    tags=["Blueprint Write Operations"],
    responses={
        503: {"description": "Service temporarily unavailable"},
        500: {"description": "Internal server error"}
    }
)

settings = get_settings()
logger = logging.getLogger(__name__)

# Store active processing tasks
processing_tasks = {}

# SSE Event Management
document_event_queues = defaultdict(list)  # document_id -> list of event queues
document_event_locks = defaultdict(asyncio.Lock)  # document_id -> lock for queue management
active_sse_connections = defaultdict(set)  # document_id -> set of connection IDs

# ===== UTILITY FUNCTIONS =====

def validate_document_id(document_id: str) -> str:
    """
    Validate and sanitize document ID to prevent injection attacks.
    Returns cleaned document ID.
    """
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Document ID must be a non-empty string"
        )
    
    # Remove any potentially dangerous characters
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    clean_id = clean_id.strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Document ID must be at least 3 characters long after sanitization"
        )
    
    if len(clean_id) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID must be 50 characters or less"
        )
    
    return clean_id

def validate_file_type(filename: str) -> bool:
    """Validate that uploaded file is a PDF"""
    if not filename:
        return False
    return filename.lower().endswith('.pdf')

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    filename = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return clean_name

# ===== SSE EVENT MANAGEMENT =====

async def emit_sse_event(document_id: str, event: SSEEvent):
    """Emit an SSE event to all connected clients for a document"""
    if not settings.ENABLE_SSE:
        return
    
    async with document_event_locks[document_id]:
        # Add event to all active queues for this document
        event_data = {
            'event_type': event.event_type,
            'data': event.data,
            'timestamp': event.timestamp,
            'document_id': event.document_id
        }
        
        # Serialize event data
        event_json = json.dumps(event_data)
        
        # Add to each queue
        for queue in document_event_queues[document_id]:
            try:
                # Check queue size limit
                if len(queue) < settings.SSE_EVENT_QUEUE_SIZE:
                    await queue.put(event_json)
                else:
                    logger.warning(f"Event queue full for document {document_id}, dropping oldest event")
                    # Remove oldest event if queue is full
                    try:
                        queue.get_nowait()
                        await queue.put(event_json)
                    except asyncio.QueueEmpty:
                        await queue.put(event_json)
            except Exception as e:
                logger.error(f"Failed to emit event to queue: {e}")

async def create_status_update_event(
    document_id: str,
    stage: str,
    message: str,
    progress_percent: int,
    pages_processed: int,
    estimated_time: Optional[int] = None
) -> StatusUpdateEvent:
    """Create a status update event"""
    return StatusUpdateEvent(
        stage=stage,
        message=message,
        progress_percent=progress_percent,
        pages_processed=pages_processed,
        estimated_time=estimated_time
    )

async def create_resource_ready_event(
    document_id: str,
    resource_type: str,
    resource_id: str,
    page_number: Optional[int] = None,
    url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ResourceReadyEvent:
    """Create a resource ready event"""
    return ResourceReadyEvent(
        resource_type=resource_type,
        resource_id=resource_id,
        page_number=page_number,
        url=url,
        metadata=metadata or {}
    )

# ===== SSE STREAMING ENDPOINT =====

@blueprint_router.get(
    "/documents/{document_id}/status/stream",
    summary="Stream document processing status via SSE",
    description="Real-time status updates using Server-Sent Events",
    response_class=StreamingResponse
)
async def stream_document_status(
    request: Request,
    document_id: str
):
    """
    Stream real-time document processing status using Server-Sent Events.
    Clients receive updates as processing happens.
    """
    if not settings.ENABLE_SSE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SSE is not enabled on this server"
        )
    
    clean_document_id = validate_document_id(document_id)
    connection_id = str(uuid.uuid4())
    
    logger.info(f"📡 SSE connection request for document: {clean_document_id}")
    logger.info(f"   Connection ID: {connection_id}")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the client"""
        queue = asyncio.Queue(maxsize=settings.SSE_EVENT_QUEUE_SIZE)
        
        # Register this connection
        async with document_event_locks[clean_document_id]:
            document_event_queues[clean_document_id].append(queue)
            active_sse_connections[clean_document_id].add(connection_id)
        
        logger.info(f"✅ SSE connection established: {connection_id}")
        logger.info(f"   Active connections for document: {len(active_sse_connections[clean_document_id])}")
        
        try:
            # Send initial connection event
            yield f"event: connected\ndata: {{\"document_id\": \"{clean_document_id}\", \"connection_id\": \"{connection_id}\"}}\n\n"
            
            # Check current document status
            storage_service = getattr(request.app.state, 'storage_service', None)
            if storage_service:
                try:
                    status_blob = f"{clean_document_id}_status.json"
                    if await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, status_blob):
                        status_text = await storage_service.download_blob_as_text(
                            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                            blob_name=status_blob
                        )
                        status_data = json.loads(status_text)
                        
                        # Send current status
                        current_status = {
                            "stage": status_data.get("status", "unknown"),
                            "message": f"Current status: {status_data.get('status', 'unknown')}",
                            "progress_percent": round(
                                (status_data.get('pages_processed', 0) / max(status_data.get('total_pages', 1), 1)) * 100
                            ),
                            "pages_processed": status_data.get('pages_processed', 0)
                        }
                        yield f"event: status_update\ndata: {json.dumps(current_status)}\n\n"
                except Exception as e:
                    logger.error(f"Failed to get initial status: {e}")
            
            # Main event loop
            last_keepalive = time.time()
            
            while True:
                try:
                    # Wait for events with timeout for keepalive
                    event_data = await asyncio.wait_for(
                        queue.get(),
                        timeout=settings.SSE_KEEPALIVE_INTERVAL
                    )
                    
                    # Parse and send event
                    event = json.loads(event_data)
                    event_type = event['event_type']
                    data = event['data']
                    
                    # Format SSE event
                    if settings.SSE_ENABLE_COMPRESSION and len(json.dumps(data)) > 1000:
                        # For large events, send a simplified version
                        if event_type == SSEEventType.resource_ready and 'metadata' in data:
                            data['metadata'] = {'compressed': True, 'original_size': len(str(data['metadata']))}
                    
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\nid: {event.get('timestamp', '')}\n\n"
                    
                    # Check if processing is complete
                    if event_type == SSEEventType.processing_complete:
                        logger.info(f"📍 Processing complete, will close SSE connection soon: {connection_id}")
                        # Send a few more events if any, then close
                        await asyncio.sleep(2)
                        break
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    current_time = time.time()
                    if current_time - last_keepalive >= settings.SSE_KEEPALIVE_INTERVAL:
                        yield f"event: keepalive\ndata: {{\"timestamp\": \"{datetime.utcnow().isoformat()}Z\"}}\n\n"
                        last_keepalive = current_time
                
                except asyncio.CancelledError:
                    logger.info(f"🔌 SSE connection cancelled: {connection_id}")
                    break
                
                except Exception as e:
                    logger.error(f"Error in SSE event loop: {e}")
                    yield f"event: error\ndata: {{\"error\": \"Internal error in event stream\"}}\n\n"
                    break
                
                # Check if client is still connected
                if await request.is_disconnected():
                    logger.info(f"🔌 Client disconnected: {connection_id}")
                    break
                
                # Check connection timeout
                if time.time() - last_keepalive > settings.SSE_CLIENT_TIMEOUT:
                    logger.info(f"⏱️ SSE connection timeout: {connection_id}")
                    break
        
        finally:
            # Cleanup connection
            async with document_event_locks[clean_document_id]:
                if queue in document_event_queues[clean_document_id]:
                    document_event_queues[clean_document_id].remove(queue)
                active_sse_connections[clean_document_id].discard(connection_id)
                
                # Clean up empty structures
                if not document_event_queues[clean_document_id]:
                    del document_event_queues[clean_document_id]
                if not active_sse_connections[clean_document_id]:
                    del active_sse_connections[clean_document_id]
            
            logger.info(f"✅ SSE connection closed: {connection_id}")
            logger.info(f"   Remaining connections for document: {len(active_sse_connections.get(clean_document_id, []))}")
    
    # Return SSE response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
            "Access-Control-Allow-Origin": "*",  # CORS support
        }
    )

# ===== ASYNC PROCESSING WITH SSE EVENTS =====

async def process_pdf_async(
    session_id: str,
    contents: bytes,
    clean_filename: str,
    author: str,
    storage_service,
    pdf_service,
    session_service
):
    """
    Process PDF asynchronously with SSE event emission
    """
    try:
        logger.info(f"🔄 Starting async processing for document: {session_id}")
        
        # Initial processing status
        processing_status = {
            'document_id': session_id,
            'status': 'processing',
            'started_at': datetime.utcnow().isoformat() + 'Z',
            'filename': clean_filename,
            'author': author,
            'pages_processed': 0,
            'total_pages': 0,
            'current_batch': 0,
            'total_batches': 0
        }
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Emit initial status event
        if settings.ENABLE_SSE:
            await emit_sse_event(
                session_id,
                SSEEvent(
                    event_type=SSEEventType.status_update,
                    data=StatusUpdateEvent(
                        stage="processing_started",
                        message="Document processing has started",
                        progress_percent=0,
                        pages_processed=0
                    ).dict(),
                    timestamp=datetime.utcnow().isoformat() + 'Z',
                    document_id=session_id
                )
            )
        
        # Define event callback for PDF service
        async def processing_event_callback(event_type: str, event_data: Dict[str, Any]):
            """Callback to emit SSE events during processing"""
            if not settings.ENABLE_SSE:
                return
            
            try:
                if event_type == "page_processed":
                    await emit_sse_event(
                        session_id,
                        SSEEvent(
                            event_type=SSEEventType.status_update,
                            data=StatusUpdateEvent(
                                stage="processing_pages",
                                message=f"Processing page {event_data['page_number']} of {event_data['total_pages']}",
                                progress_percent=event_data.get('progress_percent', 0),
                                pages_processed=event_data['page_number'],
                                estimated_time=event_data.get('estimated_time')
                            ).dict(),
                            timestamp=datetime.utcnow().isoformat() + 'Z',
                            document_id=session_id
                        )
                    )
                
                elif event_type == "resource_ready":
                    resource_type = event_data.get('resource_type', 'unknown')
                    page_number = event_data.get('page_number')
                    
                    # Build resource URL
                    url = None
                    if resource_type == "thumbnail" and page_number:
                        url = f"/api/v1/documents/{session_id}/pages/{page_number}/image?type=thumb"
                    elif resource_type == "full_image" and page_number:
                        url = f"/api/v1/documents/{session_id}/pages/{page_number}/image"
                    elif resource_type == "ai_image" and page_number:
                        url = f"/api/v1/documents/{session_id}/pages/{page_number}/image?type=ai"
                    elif resource_type == "context_text":
                        url = f"/api/v1/documents/{session_id}/context"
                    
                    await emit_sse_event(
                        session_id,
                        SSEEvent(
                            event_type=SSEEventType.resource_ready,
                            data=ResourceReadyEvent(
                                resource_type=resource_type,
                                resource_id=event_data.get('resource_id', resource_type),
                                page_number=page_number,
                                url=url,
                                metadata=event_data.get('metadata', {})
                            ).dict(),
                            timestamp=datetime.utcnow().isoformat() + 'Z',
                            document_id=session_id
                        )
                    )
                
                elif event_type == "batch_complete":
                    await emit_sse_event(
                        session_id,
                        SSEEvent(
                            event_type=SSEEventType.status_update,
                            data=StatusUpdateEvent(
                                stage="batch_complete",
                                message=f"Completed batch {event_data['batch_number']} of {event_data['total_batches']}",
                                progress_percent=event_data.get('progress_percent', 0),
                                pages_processed=event_data.get('pages_processed', 0)
                            ).dict(),
                            timestamp=datetime.utcnow().isoformat() + 'Z',
                            document_id=session_id
                        )
                    )
                
            except Exception as e:
                logger.error(f"Failed to emit processing event: {e}")
        
        # Process the PDF with event callback
        await pdf_service.process_and_cache_pdf(
            session_id=session_id,
            pdf_bytes=contents,
            storage_service=storage_service,
            event_callback=processing_event_callback if settings.ENABLE_SSE else None
        )
        
        # Final status update
        processing_status['status'] = 'ready'
        processing_status['completed_at'] = datetime.utcnow().isoformat() + 'Z'
        
        # Get final metadata
        try:
            metadata_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json"
            )
            metadata = json.loads(metadata_text)
            processing_status['pages_processed'] = metadata.get('page_count', 0)
            processing_status['total_pages'] = metadata.get('total_pages', 0)
        except:
            pass
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Emit completion event
        if settings.ENABLE_SSE:
            await emit_sse_event(
                session_id,
                SSEEvent(
                    event_type=SSEEventType.processing_complete,
                    data=ProcessingCompleteEvent(
                        status="ready",
                        total_pages=processing_status.get('pages_processed', 0),
                        processing_time=time.time() - time.mktime(
                            datetime.fromisoformat(processing_status['started_at'].rstrip('Z')).timetuple()
                        ),
                        resources_summary={
                            "pages_with_images": processing_status.get('pages_processed', 0),
                            "pages_with_thumbnails": processing_status.get('pages_processed', 0),
                            "context_available": True,
                            "metadata_available": True
                        }
                    ).dict(),
                    timestamp=datetime.utcnow().isoformat() + 'Z',
                    document_id=session_id
                )
            )
        
        # Update session if available
        if session_service and hasattr(session_service, 'update_session_metadata'):
            try:
                await session_service.update_session_metadata(
                    document_id=session_id,
                    metadata={
                        'processing_complete': True,
                        'status': 'ready',
                        'pages_processed': processing_status.get('pages_processed', 0)
                    }
                )
            except Exception as e:
                logger.warning(f"Session update failed: {e}")
        
        logger.info(f"✅ Async processing completed for: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Async processing failed for {session_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Save error status
        error_status = {
            'document_id': session_id,
            'status': 'error',
            'error': str(e),
            'failed_at': datetime.utcnow().isoformat() + 'Z',
            'filename': clean_filename,
            'author': author
        }
        
        try:
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_status.json",
                data=json.dumps(error_status).encode('utf-8')
            )
        except Exception as storage_error:
            logger.error(f"Failed to save error status: {storage_error}")
        
        # Emit error event
        if settings.ENABLE_SSE:
            await emit_sse_event(
                session_id,
                SSEEvent(
                    event_type=SSEEventType.error,
                    data=ProcessingErrorEvent(
                        error_type="processing_failed",
                        message=str(e),
                        is_fatal=True,
                        retry_possible=False
                    ).dict(),
                    timestamp=datetime.utcnow().isoformat() + 'Z',
                    document_id=session_id
                )
            )
    
    finally:
        # Remove from processing tasks
        if session_id in processing_tasks:
            del processing_tasks[session_id]

# ===== WRITE OPERATION ROUTES (UNCHANGED) =====

@blueprint_router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a blueprint PDF",
    description="Upload a construction blueprint PDF for AI analysis. Returns immediately with processing status."
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload"),
    author: Optional[str] = Form(default="Anonymous", description="Name of the person uploading"),
    trade: Optional[str] = Form(default=None, description="Trade/discipline associated with the document")
):
    """
    Upload a PDF document for processing.
    Returns immediately with document ID while processing happens asynchronously.
    """
    
    # Validate request
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file provided"
        )
    
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # Read file content
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file"
        )
    
    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    max_size_mb = settings.MAX_FILE_SIZE_MB
    
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum size is {max_size_mb}MB."
        )
    
    # Validate it's a valid PDF
    if not contents.startswith(b'%PDF'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format"
        )
    
    # Sanitize filename and generate session ID
    clean_filename = sanitize_filename(file.filename)
    session_id = str(uuid.uuid4())
    
    try:
        # Get storage service
        storage_service = getattr(request.app.state, 'storage_service', None)
        if not storage_service:
            logger.error("Storage service not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="Storage service is not available. Please try again later."
            )
        
        logger.info(f"📤 Uploading document: {clean_filename} ({file_size_mb:.1f}MB)")
        logger.info(f"   Session ID: {session_id}")
        logger.info(f"   Author: {author}")
        if trade:
            logger.info(f"   Trade: {trade}")
        
        # Upload original PDF
        pdf_blob_name = f"{session_id}.pdf"
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=pdf_blob_name,
            data=contents
        )
        
        logger.info(f"✅ PDF uploaded successfully to blob: {pdf_blob_name}")
        
        # Estimate processing time
        estimated_processing_time = max(30, int(file_size_mb * 3))  # ~3 seconds per MB
        
        # Create initial status file
        initial_status = {
            'document_id': session_id,
            'status': 'processing',
            'filename': clean_filename,
            'author': author,
            'trade': trade,
            'file_size_mb': round(file_size_mb, 2),
            'uploaded_at': datetime.utcnow().isoformat() + 'Z',
            'estimated_processing_time': estimated_processing_time,
            'pages_processed': 0,
            'sse_enabled': settings.ENABLE_SSE,
            'sse_url': f"/api/v1/documents/{session_id}/status/stream" if settings.ENABLE_SSE else None
        }
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(initial_status).encode('utf-8')
        )
        
        # Create session if service available
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service:
            try:
                await session_service.create_session(
                    document_id=session_id,
                    original_filename=clean_filename
                )
                logger.info("✅ Created session for document")
            except Exception as e:
                logger.warning(f"Session creation failed (non-critical): {e}")
        
        # Get PDF service
        pdf_service = getattr(request.app.state, 'pdf_service', None)
        if not pdf_service:
            logger.warning("⚠️  PDF service not available - upload only mode")
            
            return DocumentUploadResponse(
                document_id=session_id,
                filename=clean_filename,
                status="uploaded",
                message="Document uploaded successfully. Processing service unavailable.",
                file_size_mb=round(file_size_mb, 2)
            )
        
        # Start async processing
        task = asyncio.create_task(
            process_pdf_async(
                session_id=session_id,
                contents=contents,
                clean_filename=clean_filename,
                author=author,
                storage_service=storage_service,
                pdf_service=pdf_service,
                session_service=session_service
            )
        )
        
        # Store task reference
        processing_tasks[session_id] = task
        
        logger.info(f"📋 Returning processing status for: {session_id}")
        
        return DocumentUploadResponse(
            document_id=session_id,
            filename=clean_filename,
            status="processing",
            message=f"Document uploaded! Processing will take approximately {estimated_processing_time} seconds.",
            file_size_mb=round(file_size_mb, 2),
            estimated_time=estimated_processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Upload failed: {str(e)}"
        )

# ===== REMAINING ENDPOINTS (UNCHANGED) =====

@blueprint_router.post(
    "/documents/{document_id}/chat",
    response_model=ChatResponse,
    summary="Chat with a blueprint",
    description="Send a question about an uploaded blueprint and receive AI analysis with visual highlights"
)
async def chat_with_document(
    request: Request,
    document_id: str,
    chat_request: ChatRequest
):
    """
    Chat with an uploaded document using AI analysis.
    Returns AI response with visual highlights and note suggestions.
    """
    
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    # Validate request
    if not chat_request.prompt or len(chat_request.prompt.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Prompt cannot be empty"
        )
    
    # Get services
    ai_service = getattr(request.app.state, 'ai_service', None)
    storage_service = getattr(request.app.state, 'storage_service', None)
    
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
        logger.info(f"💬 Chat request for document: {clean_document_id}")
        logger.info(f"   Author: {chat_request.author}")
        logger.info(f"   Prompt: {chat_request.prompt[:100]}...")
        
        # Quick check if document is ready
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        context_exists = await storage_service.blob_exists(
            container_name=cache_container,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not context_exists:
            # Check status for better error message
            status_blob = f"{clean_document_id}_status.json"
            if await storage_service.blob_exists(cache_container, status_blob):
                status_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=status_blob
                )
                status_data = json.loads(status_text)
                
                if status_data.get('status') == 'processing':
                    # Calculate estimated time
                    estimated_time = 30
                    if status_data.get('started_at'):
                        try:
                            started = datetime.fromisoformat(status_data['started_at'].rstrip('Z'))
                            elapsed = (datetime.utcnow() - started).total_seconds()
                            estimated_total = status_data.get('estimated_processing_time', 60)
                            estimated_time = max(5, int(estimated_total - elapsed))
                        except:
                            pass
                    
                    raise HTTPException(
                        status_code=status.HTTP_425_TOO_EARLY,
                        detail=f"Document is still being processed. Please try again in {estimated_time} seconds."
                    )
                elif status_data.get('status') == 'error':
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Document processing failed: {status_data.get('error', 'Unknown error')}"
                    )
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{clean_document_id}' not found or not ready"
            )
        
        # Get AI response
        ai_result = await ai_service.get_ai_response(
            prompt=chat_request.prompt,
            document_id=clean_document_id,
            storage_service=storage_service,
            author=chat_request.author,
            current_page=chat_request.current_page,
            request_highlights=True,
            reference_previous=chat_request.reference_previous,
            preserve_existing=chat_request.preserve_existing,
            show_trade_info=chat_request.show_trade_info,
            detect_conflicts=chat_request.detect_conflicts,
            auto_suggest_notes=chat_request.auto_suggest_notes,
            note_suggestion_threshold=chat_request.note_suggestion_threshold
        )
        
        # Save chat to history
        await save_chat_to_history(
            document_id=clean_document_id,
            author=chat_request.author,
            prompt=chat_request.prompt,
            response=ai_result.get("ai_response", ""),
            storage_service=storage_service,
            has_highlights=bool(ai_result.get("visual_highlights")),
            note_suggested=bool(ai_result.get("note_suggestion"))
        )
        
        # Record activity in session
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service and hasattr(session_service, 'record_chat_activity'):
            try:
                await session_service.record_chat_activity(
                    document_id=clean_document_id,
                    user=chat_request.author,
                    chat_data={
                        'prompt': chat_request.prompt,
                        'current_page': chat_request.current_page,
                        'highlights_generated': len(ai_result.get("visual_highlights", []))
                    }
                )
            except Exception as e:
                logger.warning(f"Session update failed (non-critical): {e}")
        
        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_result.get("ai_response", "I apologize, but I was unable to analyze the blueprint."),
            visual_highlights=ai_result.get("visual_highlights"),
            current_page=chat_request.current_page,
            query_session_id=ai_result.get("query_session_id"),
            all_highlight_pages=ai_result.get("all_highlight_pages"),
            trade_summary=ai_result.get("trade_summary"),
            note_suggestion=ai_result.get("note_suggestion")
        )
        
        logger.info(f"✅ Chat response generated successfully")
        if ai_result.get("visual_highlights"):
            logger.info(f"   Highlights created: {len(ai_result['visual_highlights'])}")
        if ai_result.get("note_suggestion"):
            logger.info(f"   Note suggested: {ai_result['note_suggestion'].title}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to process chat: {str(e)}"
        )

@blueprint_router.post(
    "/documents/{document_id}/notes/quick-create",
    summary="Quick create note from AI suggestion",
    description="Create a note with one click from AI suggestion"
)
async def quick_create_note_from_suggestion(
    request: Request,
    document_id: str,
    quick_note: QuickNoteCreate
):
    """
    Quick endpoint to create a note from AI suggestion.
    Streamlined for one-click note creation from chat responses.
    """
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    # Get storage service
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        notes_blob = f"{clean_document_id}_notes.json"
        
        # Load existing notes
        try:
            notes_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=notes_blob
            )
            all_notes = json.loads(notes_text)
        except:
            all_notes = []
        
        # Check note limit
        max_notes = settings.MAX_NOTES_PER_DOCUMENT
        user_notes = [n for n in all_notes if n.get('author') == quick_note.author]
        if len(user_notes) >= max_notes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Note limit reached ({max_notes} notes per document)"
            )
        
        # Create note with all metadata
        new_note = {
            "note_id": str(uuid.uuid4())[:8],
            "document_id": clean_document_id,
            "text": quick_note.text,
            "note_type": quick_note.note_type,
            "author": quick_note.author,
            "impacts_trades": quick_note.impacts_trades or [],
            "priority": quick_note.priority,
            "is_private": quick_note.is_private,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "char_count": len(quick_note.text),
            "status": "open",
            "ai_suggested": quick_note.ai_suggested,
            "suggestion_confidence": quick_note.suggestion_confidence,
            "related_query_sessions": quick_note.related_query_sessions or [],
            "related_element_ids": quick_note.related_highlights or [],
            "source_pages": quick_note.source_pages or []
        }
        
        # Add to notes
        all_notes.append(new_note)
        
        # Save back
        await storage_service.upload_file(
            container_name=cache_container,
            blob_name=notes_blob,
            data=json.dumps(all_notes, indent=2).encode('utf-8')
        )
        
        logger.info(f"✅ Quick note created from AI suggestion")
        logger.info(f"   Note ID: {new_note['note_id']}")
        logger.info(f"   Type: {quick_note.note_type}")
        logger.info(f"   Priority: {quick_note.priority}")
        
        return {
            "status": "success",
            "note_id": new_note["note_id"],
            "message": "Note created successfully from AI suggestion"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick note creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@blueprint_router.delete(
    "/documents/{document_id}/clear",
    summary="Clear document and all associated data",
    description="Delete a document and all its associated data (annotations, notes, cache)"
)
async def clear_document(
    request: Request,
    document_id: str,
    confirm: bool = Query(False, description="Confirm deletion")
):
    """
    Clear a document and all associated data.
    Requires confirmation parameter to prevent accidental deletion.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deletion not confirmed. Set confirm=true to proceed."
        )
    
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        logger.info(f"🗑️ Clearing document: {clean_document_id}")
        
        # Delete from main container
        deleted_count = 0
        
        # Delete original PDF
        if await storage_service.delete_blob(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        ):
            deleted_count += 1
        
        # Delete all cached files
        cache_deleted = await storage_service.delete_blobs_with_prefix(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            prefix=clean_document_id
        )
        deleted_count += cache_deleted
        
        # Clear session if available
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service:
            try:
                await session_service.clear_session(clean_document_id)
            except Exception as e:
                logger.warning(f"Session clear failed: {e}")
        
        logger.info(f"✅ Deleted {deleted_count} files for document {clean_document_id}")
        
        return {
            "status": "success",
            "message": f"Document {clean_document_id} and all associated data cleared",
            "files_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear document: {str(e)}"
        )

# ===== HELPER FUNCTIONS =====

async def save_chat_to_history(
    document_id: str,
    author: str,
    prompt: str,
    response: str,
    storage_service,
    has_highlights: bool = False,
    note_suggested: bool = False
):
    """Save chat interaction to history for activity tracking"""
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        chat_blob = f"{document_id}_all_chats.json"
        
        # Load existing chat history
        try:
            chat_data = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=chat_blob
            )
            chat_history = json.loads(chat_data)
        except:
            chat_history = []
        
        # Add new chat entry
        chat_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "author": author,
            "prompt": prompt,
            "response": response[:1000],  # Limit response size
            "has_highlights": has_highlights,
            "note_suggested": note_suggested
        }
        
        chat_history.append(chat_entry)
        
        # Keep only recent chats
        max_chats = settings.MAX_CHAT_LOGS
        if len(chat_history) > max_chats:
            chat_history = chat_history[-max_chats:]
        
        # Save updated history
        await storage_service.upload_file(
            container_name=cache_container,
            blob_name=chat_blob,
            data=json.dumps(chat_history, indent=2).encode('utf-8')
        )
        
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")

# ===== SSE CONNECTION MONITORING =====

@blueprint_router.get(
    "/sse/connections",
    summary="Get active SSE connections",
    description="Monitor active SSE connections (admin endpoint)"
)
async def get_sse_connections(
    request: Request,
    admin_token: str = Query(..., description="Admin token for authentication")
):
    """Get information about active SSE connections"""
    if admin_token != settings.ADMIN_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )
    
    connections_info = {}
    for doc_id, connections in active_sse_connections.items():
        connections_info[doc_id] = {
            "active_connections": len(connections),
            "connection_ids": list(connections),
            "event_queues": len(document_event_queues.get(doc_id, []))
        }
    
    return {
        "total_documents": len(active_sse_connections),
        "total_connections": sum(len(conns) for conns in active_sse_connections.values()),
        "documents": connections_info,
        "sse_enabled": settings.ENABLE_SSE
    }

# NOTE: READ operations (status, download, info) remain in document_routes.py