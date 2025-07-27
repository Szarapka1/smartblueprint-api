# app/api/routes/blueprint_routes.py - FIXED VERSION WITH PROPER ERROR HANDLING

"""
Blueprint Write Operations - Handles document upload, processing, chat, and annotations
FIXED: Proper error handling and status updates to prevent 30% stuck issue
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
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import weakref
import gc

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

# ===== SSE EVENT MANAGEMENT =====

class SSEConnectionManager:
    """Production-ready SSE connection and event manager"""
    
    def __init__(self):
        self._connections = defaultdict(dict)
        self._connection_metadata = {}
        self._locks = defaultdict(asyncio.Lock)
        self._event_history = defaultdict(lambda: deque(maxlen=settings.SSE_EVENT_HISTORY_CLEANUP_SIZE))
        self._history_limit = settings.SSE_EVENT_HISTORY_CLEANUP_SIZE
        self._total_connections = 0
        self._total_events_sent = 0
        self._connection_errors = 0
        self._dropped_events = 0
        self._cleanup_tasks = weakref.WeakValueDictionary()
        self._last_health_check = time.time()
        self._health_check_interval = settings.PROCESSING_HEALTH_CHECK_INTERVAL
        
    async def add_connection(self, document_id: str, connection_id: str, send_history: bool = True) -> asyncio.Queue:
        """Add a new SSE connection"""
        async with self._locks[document_id]:
            queue = asyncio.Queue(maxsize=settings.SSE_MAX_QUEUE_SIZE_PER_CONNECTION)
            self._connections[document_id][connection_id] = queue
            self._connection_metadata[connection_id] = {
                'document_id': document_id,
                'connected_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'events_sent': 0,
                'errors': 0,
                'dropped_events': 0,
                'slow_consumer': False,
                'health_status': 'healthy'
            }
            self._total_connections += 1
            
            if send_history and document_id in self._event_history:
                history_events = list(self._event_history[document_id])[-10:]
                for event in history_events:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full when sending history to {connection_id}")
                        break
            
            logger.info(f"✅ SSE connection added: doc={document_id}, conn={connection_id}")
            return queue
    
    async def remove_connection(self, document_id: str, connection_id: str):
        """Remove SSE connection with cleanup"""
        async with self._locks[document_id]:
            if connection_id in self._connections[document_id]:
                queue = self._connections[document_id].pop(connection_id)
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                
                metadata = self._connection_metadata.pop(connection_id, {})
                if metadata:
                    logger.info(f"📊 Connection stats for {connection_id}:")
                    logger.info(f"   Events sent: {metadata.get('events_sent', 0)}")
                    logger.info(f"   Events dropped: {metadata.get('dropped_events', 0)}")
                
                if not self._connections[document_id]:
                    del self._connections[document_id]
                    if document_id in self._event_history:
                        del self._event_history[document_id]
                
                logger.info(f"🔌 SSE connection removed: doc={document_id}, conn={connection_id}")
    
    async def emit_event(self, document_id: str, event_type: str, event_data: Dict[str, Any], store_history: bool = True):
        """Emit event to all connections"""
        if not settings.ENABLE_SSE:
            return
        
        event = SSEEvent(
            event_type=event_type,
            data=event_data,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            document_id=document_id
        )
        
        event_json = json.dumps({
            'event_type': event.event_type,
            'data': event.data,
            'timestamp': event.timestamp,
            'document_id': event.document_id
        })
        
        if store_history:
            async with self._locks[document_id]:
                self._event_history[document_id].append(event_json)
        
        async with self._locks[document_id]:
            if document_id not in self._connections:
                return
            
            failed_connections = []
            
            for connection_id, queue in self._connections[document_id].items():
                metadata = self._connection_metadata.get(connection_id, {})
                
                try:
                    if metadata.get('slow_consumer', False) and event_type == SSEEventType.keepalive:
                        metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                        self._dropped_events += 1
                        continue
                    
                    try:
                        await asyncio.wait_for(queue.put(event_json), timeout=0.1)
                        metadata['events_sent'] = metadata.get('events_sent', 0) + 1
                        metadata['last_activity'] = datetime.utcnow()
                        metadata['errors'] = 0
                        self._total_events_sent += 1
                    except asyncio.TimeoutError:
                        metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                        self._dropped_events += 1
                        
                        if metadata['dropped_events'] > settings.SSE_SLOW_CONSUMER_THRESHOLD:
                            metadata['slow_consumer'] = True
                            logger.warning(f"Marking {connection_id} as slow consumer")
                        
                        if event_type in [SSEEventType.processing_complete, SSEEventType.error]:
                            metadata['errors'] = metadata.get('errors', 0) + 1
                            if metadata['errors'] > 5:
                                failed_connections.append(connection_id)
                except Exception as e:
                    logger.error(f"Error sending event to {connection_id}: {e}")
                    metadata['errors'] = metadata.get('errors', 0) + 1
                    self._connection_errors += 1
                    
                    if metadata['errors'] > 10:
                        failed_connections.append(connection_id)
            
            for conn_id in failed_connections:
                logger.warning(f"Removing failed connection {conn_id}")
                await self.remove_connection(document_id, conn_id)
    
    def get_connection_info(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Get connection statistics"""
        if document_id:
            return {
                'document_id': document_id,
                'active_connections': len(self._connections.get(document_id, {})),
                'event_history_size': len(self._event_history.get(document_id, []))
            }
        else:
            return {
                'total_documents': len(self._connections),
                'total_active_connections': sum(len(conns) for conns in self._connections.values()),
                'total_connections_created': self._total_connections,
                'total_events_sent': self._total_events_sent,
                'total_events_dropped': self._dropped_events,
                'connection_errors': self._connection_errors
            }

# Global SSE manager instance
sse_manager = SSEConnectionManager()

# Store active processing tasks
processing_tasks = {}
processing_tasks_lock = asyncio.Lock()

# ===== UTILITY FUNCTIONS =====

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID"""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(status_code=400, detail="Document ID must be a non-empty string")
    
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip()).strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(status_code=400, detail="Document ID must be at least 3 characters long")
    
    if len(clean_id) > 50:
        raise HTTPException(status_code=400, detail="Document ID must be 50 characters or less")
    
    return clean_id

def validate_file_type(filename: str) -> bool:
    """For now, accept any file type"""
    return bool(filename)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename"""
    filename = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return clean_name

# ===== FIXED ASYNC PROCESSING WITH PROPER ERROR HANDLING =====

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
    Process PDF asynchronously with FIXED error handling and status updates
    """
    processing_start_time = time.time()
    
    logger.info("="*50)
    logger.info(f"🔄 ASYNC PROCESSING STARTED: {session_id}")
    logger.info("="*50)
    
    # Initialize processing status
    processing_status = {
        'document_id': session_id,
        'status': 'processing',
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'filename': clean_filename,
        'author': author,
        'pages_processed': 0,
        'total_pages': 0,
        'current_batch': 0,
        'total_batches': 0,
        'sse_enabled': settings.ENABLE_SSE,
        'processing_stage': 'initializing',
        'last_update': datetime.utcnow().isoformat() + 'Z'
    }
    
    try:
        # Save initial processing status
        logger.info("📝 Saving initial processing status...")
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Emit initial status event
        await sse_manager.emit_event(
            session_id,
            SSEEventType.status_update,
            StatusUpdateEvent(
                stage="processing_started",
                message="Document processing has started",
                progress_percent=0,
                pages_processed=0
            ).dict()
        )
        
        # Define event callback for PDF service
        async def processing_event_callback(event_type: str, event_data: Dict[str, Any]):
            """Callback to emit SSE events and update status during processing"""
            try:
                logger.debug(f"📡 Processing event: {event_type}")
                
                if event_type == "page_processed":
                    processing_status['pages_processed'] = event_data.get('page_number', 0)
                    processing_status['total_pages'] = event_data.get('total_pages', 0)
                    processing_status['processing_stage'] = f"processing_page_{event_data.get('page_number', 0)}"
                    processing_status['last_update'] = datetime.utcnow().isoformat() + 'Z'
                    
                    # Save status update
                    await storage_service.upload_file(
                        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_status.json",
                        data=json.dumps(processing_status).encode('utf-8')
                    )
                    
                    # Emit SSE event
                    await sse_manager.emit_event(
                        session_id,
                        SSEEventType.status_update,
                        StatusUpdateEvent(
                            stage="processing_pages",
                            message=f"Processing page {event_data['page_number']} of {event_data['total_pages']}",
                            progress_percent=event_data.get('progress_percent', 0),
                            pages_processed=event_data['page_number']
                        ).dict()
                    )
                    
                    logger.info(f"✅ Page {event_data['page_number']}/{event_data['total_pages']} processed")
                
                elif event_type == "resource_ready":
                    await sse_manager.emit_event(
                        session_id,
                        SSEEventType.resource_ready,
                        ResourceReadyEvent(
                            resource_type=event_data.get('resource_type', 'unknown'),
                            resource_id=event_data.get('resource_id', ''),
                            page_number=event_data.get('page_number'),
                            metadata=event_data.get('metadata', {})
                        ).dict()
                    )
                
            except Exception as e:
                logger.error(f"Error in processing callback: {e}")
        
        # Check if PDF service is available
        if not pdf_service:
            raise RuntimeError("PDF processing service is not available")
        
        # Process the PDF
        logger.info("🚀 Starting PDF processing...")
        await pdf_service.process_and_cache_pdf(
            session_id=session_id,
            pdf_bytes=contents,
            storage_service=storage_service,
            event_callback=processing_event_callback if settings.ENABLE_SSE else None
        )
        
        logger.info("✅ PDF processing completed successfully")
        
        # Calculate processing time
        processing_time = time.time() - processing_start_time
        
        # Update final status to 'ready'
        final_status = {
            'document_id': session_id,
            'status': 'ready',
            'filename': clean_filename,
            'author': author,
            'completed_at': datetime.utcnow().isoformat() + 'Z',
            'processing_time_seconds': round(processing_time, 2),
            'processing_stage': 'completed',
            'pages_processed': processing_status.get('pages_processed', 0),
            'total_pages': processing_status.get('total_pages', 0),
            'last_update': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Try to get final metadata
        try:
            metadata = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json"
            )
            final_status['pages_processed'] = metadata.get('page_count', 0)
            final_status['total_pages'] = metadata.get('total_pages', 0)
        except:
            pass
        
        # Save final status
        logger.info("📝 Saving final status...")
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(final_status).encode('utf-8')
        )
        
        # Emit completion event
        await sse_manager.emit_event(
            session_id,
            SSEEventType.processing_complete,
            ProcessingCompleteEvent(
                status="ready",
                total_pages=final_status.get('pages_processed', 0),
                processing_time=processing_time,
                resources_summary={
                    "pages_with_images": final_status.get('pages_processed', 0),
                    "context_available": True,
                    "metadata_available": True
                }
            ).dict()
        )
        
        # Update session if available
        if session_service:
            try:
                await session_service.update_session_metadata(
                    document_id=session_id,
                    metadata={
                        'processing_complete': True,
                        'status': 'ready',
                        'pages_processed': final_status.get('pages_processed', 0)
                    }
                )
            except:
                pass
        
        logger.info("="*50)
        logger.info(f"✅ PROCESSING COMPLETED: {session_id}")
        logger.info(f"   Total time: {processing_time:.2f}s")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f"❌ PROCESSING FAILED: {session_id}")
        logger.error(f"   Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("="*50)
        
        processing_time = time.time() - processing_start_time
        
        # Create error status
        error_status = {
            'document_id': session_id,
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'failed_at': datetime.utcnow().isoformat() + 'Z',
            'filename': clean_filename,
            'author': author,
            'processing_time_before_error': round(processing_time, 2),
            'pages_processed': processing_status.get('pages_processed', 0),
            'processing_stage': processing_status.get('processing_stage', 'unknown'),
            'last_update': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Save error status with retries
        saved = False
        for attempt in range(3):
            try:
                await storage_service.upload_file(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_status.json",
                    data=json.dumps(error_status).encode('utf-8')
                )
                logger.info(f"✅ Error status saved (attempt {attempt + 1})")
                saved = True
                break
            except Exception as save_error:
                logger.error(f"Failed to save error status (attempt {attempt + 1}): {save_error}")
                await asyncio.sleep(0.5)
        
        if not saved:
            logger.critical(f"Could not save error status for {session_id}")
        
        # Emit error event
        await sse_manager.emit_event(
            session_id,
            SSEEventType.error,
            ProcessingErrorEvent(
                error_type="processing_failed",
                message=str(e),
                is_fatal=True,
                retry_possible=False,
                details={
                    'error_type': type(e).__name__,
                    'processing_time_before_error': processing_time,
                    'pages_processed': processing_status.get('pages_processed', 0)
                }
            ).dict()
        )
    
    finally:
        # Clean up task reference
        async with processing_tasks_lock:
            task = processing_tasks.pop(session_id, None)
            if task:
                logger.info(f"📋 Removed task from tracking. Active tasks: {len(processing_tasks)}")

# ===== FIXED UPLOAD ENDPOINT =====

@blueprint_router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for processing",
    description="Upload any document for processing. Returns immediately with processing status."
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="File to upload"),
    author: Optional[str] = Form(default="Anonymous", description="Name of the person uploading"),
    trade: Optional[str] = Form(default=None, description="Trade/discipline associated with the document")
):
    """
    Upload a document with FIXED error handling and status updates
    """
    logger.info("="*50)
    logger.info("📤 UPLOAD ENDPOINT CALLED")
    logger.info("="*50)
    
    # Validate request
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Read file content
    try:
        contents = await file.read()
        logger.info(f"✅ File read successfully: {len(contents)} bytes")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    
    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    max_size_mb = settings.MAX_FILE_SIZE_MB
    
    logger.info(f"📊 File size: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")
    
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum size is {max_size_mb}MB."
        )
    
    # Generate session ID and clean filename
    clean_filename = sanitize_filename(file.filename)
    session_id = str(uuid.uuid4())
    
    logger.info(f"📋 Session ID: {session_id}")
    logger.info(f"📄 Filename: {clean_filename}")
    logger.info(f"👤 Author: {author}")
    
    try:
        # Check services
        storage_service = getattr(request.app.state, 'storage_service', None)
        pdf_service = getattr(request.app.state, 'pdf_service', None)
        session_service = getattr(request.app.state, 'session_service', None)
        
        logger.info(f"   Storage Service: {'✅' if storage_service else '❌'}")
        logger.info(f"   PDF Service: {'✅' if pdf_service else '❌'}")
        logger.info(f"   Session Service: {'✅' if session_service else '❌'}")
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")
        
        # Upload original file
        pdf_blob_name = f"{session_id}.pdf"
        logger.info(f"📤 Uploading to blob: {pdf_blob_name}")
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=pdf_blob_name,
            data=contents
        )
        
        logger.info(f"✅ File uploaded successfully")
        
        # Estimate processing time
        estimated_processing_time = max(30, int(file_size_mb * 3))
        
        # Create initial status
        initial_status = {
            'document_id': session_id,
            'status': 'uploaded',
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
        
        # Save initial status
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(initial_status).encode('utf-8')
        )
        
        # Create session if available
        if session_service:
            try:
                await session_service.create_session(
                    document_id=session_id,
                    original_filename=clean_filename
                )
            except Exception as e:
                logger.warning(f"Session creation failed (non-critical): {e}")
        
        # Check if PDF service is available
        if not pdf_service:
            logger.warning("⚠️ PDF service not available - upload only mode")
            
            # Update status to indicate no processing
            no_processing_status = {
                **initial_status,
                'status': 'uploaded_only',
                'message': 'Document uploaded but processing service unavailable'
            }
            
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_status.json",
                data=json.dumps(no_processing_status).encode('utf-8')
            )
            
            return DocumentUploadResponse(
                document_id=session_id,
                filename=clean_filename,
                status="uploaded",
                message="Document uploaded successfully. Processing service unavailable.",
                file_size_mb=round(file_size_mb, 2)
            )
        
        # Start async processing
        logger.info("🚀 Creating async processing task...")
        
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
        async with processing_tasks_lock:
            processing_tasks[session_id] = task
        
        logger.info(f"✅ Async task created. Active tasks: {len(processing_tasks)}")
        
        # Add callback for cleanup
        def task_done_callback(future):
            try:
                if future.exception():
                    logger.error(f"Task failed for {session_id}: {future.exception()}")
                else:
                    logger.info(f"Task completed for {session_id}")
            except:
                pass
        
        task.add_done_callback(task_done_callback)
        
        # Give task a moment to start and check for immediate failures
        await asyncio.sleep(0.1)
        
        if task.done() and task.exception():
            logger.error(f"Task failed immediately: {task.exception()}")
            
            # Update status to error
            error_status = {
                'document_id': session_id,
                'status': 'error',
                'error': f"Processing failed to start: {str(task.exception())}",
                'filename': clean_filename,
                'author': author,
                'uploaded_at': datetime.utcnow().isoformat() + 'Z'
            }
            
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_status.json",
                data=json.dumps(error_status).encode('utf-8')
            )
            
            return DocumentUploadResponse(
                document_id=session_id,
                filename=clean_filename,
                status="error",
                message=f"Processing failed to start: {str(task.exception())}",
                file_size_mb=round(file_size_mb, 2)
            )
        
        # Update status to processing
        processing_status = {
            **initial_status,
            'status': 'processing',
            'started_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        logger.info(f"📋 Returning success response for: {session_id}")
        
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
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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
    """Stream real-time document processing status using Server-Sent Events"""
    if not settings.ENABLE_SSE:
        raise HTTPException(status_code=501, detail="SSE is not enabled on this server")
    
    clean_document_id = validate_document_id(document_id)
    connection_id = str(uuid.uuid4())
    
    logger.info(f"📡 SSE connection request for document: {clean_document_id}")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the client"""
        queue = None
        
        try:
            # Add connection to manager
            queue = await sse_manager.add_connection(clean_document_id, connection_id, send_history=True)
            
            logger.info(f"✅ SSE connection established: {connection_id}")
            
            # Send initial connection event
            yield f"event: connected\ndata: {{\"document_id\": \"{clean_document_id}\", \"connection_id\": \"{connection_id}\"}}\n\n"
            
            # Send current status
            storage_service = getattr(request.app.state, 'storage_service', None)
            if storage_service:
                try:
                    status_data = await storage_service.download_blob_as_json(
                        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{clean_document_id}_status.json"
                    )
                    
                    if status_data:
                        current_status = {
                            "stage": status_data.get("status", "unknown"),
                            "message": f"Current status: {status_data.get('status', 'unknown')}",
                            "progress_percent": round(
                                (status_data.get('pages_processed', 0) / max(status_data.get('total_pages', 1), 1)) * 100
                            ),
                            "pages_processed": status_data.get('pages_processed', 0),
                            "total_pages": status_data.get('total_pages', 0)
                        }
                        yield f"event: status_update\ndata: {json.dumps(current_status)}\n\n"
                except:
                    pass
            
            # Main event loop
            last_keepalive = time.time()
            
            while True:
                if await request.is_disconnected():
                    break
                
                try:
                    # Wait for events with timeout for keepalive
                    try:
                        event_data = await asyncio.wait_for(queue.get(), timeout=settings.SSE_KEEPALIVE_INTERVAL)
                        
                        # Parse and send event
                        event = json.loads(event_data)
                        event_type = event['event_type']
                        data = event['data']
                        
                        # Format SSE event
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                        
                        # Check if processing is complete
                        if event_type == SSEEventType.processing_complete:
                            logger.info(f"📍 Processing complete for {connection_id}")
                            await asyncio.sleep(2)
                            break
                        
                    except asyncio.TimeoutError:
                        # Send keepalive
                        current_time = time.time()
                        if current_time - last_keepalive >= settings.SSE_KEEPALIVE_INTERVAL:
                            yield f"event: keepalive\ndata: {{\"timestamp\": \"{datetime.utcnow().isoformat()}Z\"}}\n\n"
                            last_keepalive = current_time
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in SSE event loop: {e}")
                    yield f"event: error\ndata: {{\"error\": \"Internal error in event stream\"}}\n\n"
                    break
            
        except Exception as e:
            logger.error(f"Fatal error in SSE generator: {e}")
            yield f"event: fatal_error\ndata: {{\"error\": \"Fatal error occurred\"}}\n\n"
        
        finally:
            # Clean up connection
            logger.info(f"🧹 Cleaning up SSE connection: {connection_id}")
            await sse_manager.remove_connection(clean_document_id, connection_id)
            yield f"event: close\ndata: {{\"message\": \"Connection closed\"}}\n\n"
    
    # Return SSE response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "X-Content-Type-Options": "nosniff",
        }
    )

# ===== CHAT ENDPOINT =====

@blueprint_router.post(
    "/documents/{document_id}/chat",
    response_model=ChatResponse,
    summary="Chat with a document",
    description="Send a question about an uploaded document and receive AI analysis"
)
async def chat_with_document(
    request: Request,
    document_id: str,
    chat_request: ChatRequest
):
    """Chat with an uploaded document using AI analysis"""
    clean_document_id = validate_document_id(document_id)
    
    if not chat_request.prompt or len(chat_request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Get services
    ai_service = getattr(request.app.state, 'ai_service', None)
    storage_service = getattr(request.app.state, 'storage_service', None)
    
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service is not available")
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service is not available")
    
    try:
        logger.info(f"💬 Chat request for document: {clean_document_id}")
        
        # Check if document is ready
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        context_exists = await storage_service.blob_exists(
            container_name=cache_container,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not context_exists:
            # Check status for better error message
            try:
                status_data = await storage_service.download_blob_as_json(
                    container_name=cache_container,
                    blob_name=f"{clean_document_id}_status.json"
                )
                
                if status_data.get('status') == 'processing':
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
                        status_code=425,
                        detail=f"Document is still being processed. Please try again in {estimated_time} seconds."
                    )
                elif status_data.get('status') == 'error':
                    raise HTTPException(
                        status_code=500,
                        detail=f"Document processing failed: {status_data.get('error', 'Unknown error')}"
                    )
            except HTTPException:
                raise
            except:
                pass
            
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found or not ready")
        
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
        
        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_result.get("ai_response", "I apologize, but I was unable to analyze the document."),
            visual_highlights=ai_result.get("visual_highlights"),
            current_page=chat_request.current_page,
            query_session_id=ai_result.get("query_session_id"),
            all_highlight_pages=ai_result.get("all_highlight_pages"),
            trade_summary=ai_result.get("trade_summary"),
            note_suggestion=ai_result.get("note_suggestion")
        )
        
        logger.info(f"✅ Chat response generated successfully")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

# ===== DELETE ENDPOINT =====

@blueprint_router.delete(
    "/documents/{document_id}/clear",
    summary="Clear document and all associated data",
    description="Delete a document and all its associated data"
)
async def clear_document(
    request: Request,
    document_id: str,
    confirm: bool = Query(False, description="Confirm deletion")
):
    """Clear a document and all associated data"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Deletion not confirmed. Set confirm=true to proceed.")
    
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        logger.info(f"🗑️ Clearing document: {clean_document_id}")
        
        # Cancel any active processing tasks
        async with processing_tasks_lock:
            task = processing_tasks.pop(clean_document_id, None)
            if task and not task.done():
                task.cancel()
                logger.info(f"Cancelled active processing task for {clean_document_id}")
        
        # Delete from storage
        deleted_count = 0
        
        # Delete original file
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
            except:
                pass
        
        logger.info(f"✅ Deleted {deleted_count} files for document {clean_document_id}")
        
        return {
            "status": "success",
            "message": f"Document {clean_document_id} and all associated data cleared",
            "files_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear document: {str(e)}")

# ===== DEBUG ENDPOINTS =====

@blueprint_router.get(
    "/debug/document-status/{document_id}",
    summary="Debug document processing status"
)
async def debug_document_status(
    request: Request,
    document_id: str
):
    """Debug endpoint to check detailed document status"""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        return {"error": "Storage service not available"}
    
    try:
        # Check status file
        status_data = None
        try:
            status_data = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_status.json"
            )
        except:
            pass
        
        # Check if original file exists
        pdf_exists = await storage_service.blob_exists(
            settings.AZURE_CONTAINER_NAME,
            f"{clean_document_id}.pdf"
        )
        
        # Check processing task
        task_status = None
        async with processing_tasks_lock:
            if clean_document_id in processing_tasks:
                task = processing_tasks[clean_document_id]
                task_status = {
                    "exists": True,
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                    "exception": str(task.exception()) if task.done() and task.exception() else None
                }
        
        # Check key resources
        resources = {
            "context": await storage_service.blob_exists(
                settings.AZURE_CACHE_CONTAINER_NAME,
                f"{clean_document_id}_context.txt"
            ),
            "metadata": await storage_service.blob_exists(
                settings.AZURE_CACHE_CONTAINER_NAME,
                f"{clean_document_id}_metadata.json"
            ),
            "page_1_image": await storage_service.blob_exists(
                settings.AZURE_CACHE_CONTAINER_NAME,
                f"{clean_document_id}_page_1.jpg"
            )
        }
        
        return {
            "document_id": clean_document_id,
            "status_data": status_data,
            "pdf_exists": pdf_exists,
            "task_status": task_status,
            "resources": resources,
            "services": {
                "storage": hasattr(request.app.state, 'storage_service') and request.app.state.storage_service is not None,
                "pdf": hasattr(request.app.state, 'pdf_service') and request.app.state.pdf_service is not None,
                "ai": hasattr(request.app.state, 'ai_service') and request.app.state.ai_service is not None,
                "session": hasattr(request.app.state, 'session_service') and request.app.state.session_service is not None
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ===== PERIODIC CLEANUP =====

async def periodic_cleanup():
    """Periodic cleanup of stale connections and tasks"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Cleanup completed tasks
            async with processing_tasks_lock:
                completed_tasks = [
                    task_id for task_id, task in processing_tasks.items()
                    if task.done()
                ]
                for task_id in completed_tasks:
                    processing_tasks.pop(task_id, None)
                
                if completed_tasks:
                    logger.info(f"Cleaned up {len(completed_tasks)} completed tasks")
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# Start cleanup task when module loads
try:
    asyncio.create_task(periodic_cleanup())
except RuntimeError:
    pass