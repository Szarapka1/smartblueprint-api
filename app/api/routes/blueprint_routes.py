# app/api/routes/blueprint_routes.py - WRITE OPERATIONS WITH PRODUCTION SSE SUPPORT (FIXED)

"""
Blueprint Write Operations - Handles document upload, processing, chat, and annotations
Production-ready Server-Sent Events (SSE) implementation with robust error handling
FIXED VERSION: Proper queue management, connection health checks, and memory leak prevention
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

# ===== PRODUCTION SSE EVENT MANAGEMENT WITH FIXES =====

class SSEConnectionManager:
    """Production-ready SSE connection and event manager with queue overflow handling"""
    
    def __init__(self):
        # Connection tracking
        self._connections = defaultdict(dict)  # {document_id: {connection_id: queue}}
        self._connection_metadata = {}  # {connection_id: metadata}
        self._locks = defaultdict(asyncio.Lock)
        
        # Event history for late joiners with automatic cleanup
        self._event_history = defaultdict(lambda: deque(maxlen=settings.SSE_EVENT_HISTORY_CLEANUP_SIZE))
        self._history_limit = settings.SSE_EVENT_HISTORY_CLEANUP_SIZE
        
        # Metrics
        self._total_connections = 0
        self._total_events_sent = 0
        self._connection_errors = 0
        self._dropped_events = 0
        
        # Cleanup tracking
        self._cleanup_tasks = weakref.WeakValueDictionary()
        
        # Connection health tracking
        self._last_health_check = time.time()
        self._health_check_interval = settings.PROCESSING_HEALTH_CHECK_INTERVAL
        
    async def add_connection(
        self, 
        document_id: str, 
        connection_id: str,
        send_history: bool = True
    ) -> asyncio.Queue:
        """Add a new SSE connection with production safeguards"""
        async with self._locks[document_id]:
            # Create queue with configured size (not double for production)
            queue = asyncio.Queue(maxsize=settings.SSE_MAX_QUEUE_SIZE_PER_CONNECTION)
            
            # Store connection
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
            
            # Update metrics
            self._total_connections += 1
            
            # Send event history if requested (for late joiners)
            if send_history and document_id in self._event_history:
                # Send only recent events
                history_events = list(self._event_history[document_id])[-10:]
                for event in history_events:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full when sending history to {connection_id}")
                        break
            
            logger.info(f"✅ SSE connection added: doc={document_id}, conn={connection_id}")
            logger.info(f"   Active connections for document: {len(self._connections[document_id])}")
            
            return queue
    
    async def remove_connection(self, document_id: str, connection_id: str):
        """Remove SSE connection with cleanup"""
        async with self._locks[document_id]:
            if connection_id in self._connections[document_id]:
                # Get queue for cleanup
                queue = self._connections[document_id].pop(connection_id)
                
                # Clear any remaining items in queue
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                
                # Remove metadata
                metadata = self._connection_metadata.pop(connection_id, {})
                
                # Log connection statistics
                if metadata:
                    logger.info(f"📊 Connection stats for {connection_id}:")
                    logger.info(f"   Events sent: {metadata.get('events_sent', 0)}")
                    logger.info(f"   Events dropped: {metadata.get('dropped_events', 0)}")
                    logger.info(f"   Errors: {metadata.get('errors', 0)}")
                    logger.info(f"   Slow consumer: {metadata.get('slow_consumer', False)}")
                
                # Clean up empty document entries
                if not self._connections[document_id]:
                    del self._connections[document_id]
                    # Clear history when no connections remain
                    if document_id in self._event_history:
                        del self._event_history[document_id]
                
                logger.info(f"🔌 SSE connection removed: doc={document_id}, conn={connection_id}")
    
    async def emit_event(
        self, 
        document_id: str, 
        event_type: str,
        event_data: Dict[str, Any],
        store_history: bool = True
    ):
        """Emit event to all connections with improved queue overflow handling"""
        if not settings.ENABLE_SSE:
            return
        
        # Create event
        event = SSEEvent(
            event_type=event_type,
            data=event_data,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            document_id=document_id
        )
        
        # Serialize event
        event_json = json.dumps({
            'event_type': event.event_type,
            'data': event.data,
            'timestamp': event.timestamp,
            'document_id': event.document_id
        })
        
        # Store in history if requested (using deque with maxlen for automatic cleanup)
        if store_history:
            async with self._locks[document_id]:
                self._event_history[document_id].append(event_json)
        
        # Send to all connections
        async with self._locks[document_id]:
            if document_id not in self._connections:
                return
            
            # Track failed connections
            failed_connections = []
            
            for connection_id, queue in self._connections[document_id].items():
                metadata = self._connection_metadata.get(connection_id, {})
                
                try:
                    # Check if connection is marked as slow consumer
                    if metadata.get('slow_consumer', False):
                        # For slow consumers, drop less important events
                        if event_type == SSEEventType.keepalive:
                            metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                            self._dropped_events += 1
                            continue
                    
                    # Try to put event in queue with short timeout
                    try:
                        await asyncio.wait_for(
                            queue.put(event_json),
                            timeout=0.1  # 100ms timeout for fast failure
                        )
                        
                        # Update metadata
                        metadata['events_sent'] = metadata.get('events_sent', 0) + 1
                        metadata['last_activity'] = datetime.utcnow()
                        metadata['errors'] = 0  # Reset error count on success
                        self._total_events_sent += 1
                        
                    except asyncio.TimeoutError:
                        # Queue is full or slow - handle gracefully
                        metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                        self._dropped_events += 1
                        
                        # Mark as slow consumer if dropping too many events
                        if metadata['dropped_events'] > settings.SSE_SLOW_CONSUMER_THRESHOLD:
                            metadata['slow_consumer'] = True
                            logger.warning(f"Marking {connection_id} as slow consumer")
                        
                        # Only fail connection if dropping critical events
                        if event_type in [SSEEventType.processing_complete, SSEEventType.error]:
                            metadata['errors'] = metadata.get('errors', 0) + 1
                            if metadata['errors'] > 5:
                                failed_connections.append(connection_id)
                    
                except asyncio.QueueFull:
                    # Queue is full - try to remove oldest event for critical events only
                    if event_type in [SSEEventType.processing_complete, SSEEventType.error]:
                        try:
                            # Remove oldest event
                            old_event = await asyncio.wait_for(queue.get(), timeout=0.05)
                            # Add new event
                            await asyncio.wait_for(queue.put(event_json), timeout=0.05)
                            
                            metadata['events_sent'] = metadata.get('events_sent', 0) + 1
                            metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                            self._dropped_events += 1
                            
                        except:
                            metadata['errors'] = metadata.get('errors', 0) + 1
                            metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                            self._dropped_events += 1
                    else:
                        # For non-critical events, just drop
                        metadata['dropped_events'] = metadata.get('dropped_events', 0) + 1
                        self._dropped_events += 1
                
                except Exception as e:
                    logger.error(f"Error sending event to {connection_id}: {e}")
                    metadata['errors'] = metadata.get('errors', 0) + 1
                    self._connection_errors += 1
                    
                    # Only remove on persistent errors
                    if metadata['errors'] > 10:
                        failed_connections.append(connection_id)
            
            # Clean up failed connections
            for conn_id in failed_connections:
                logger.warning(f"Removing failed connection {conn_id}")
                await self.remove_connection(document_id, conn_id)
    
    async def check_connection_health(self, connection_id: str) -> bool:
        """Check if a connection is healthy"""
        metadata = self._connection_metadata.get(connection_id, {})
        
        # Check if connection exists
        if not metadata:
            return False
        
        # Check for excessive errors
        if metadata.get('errors', 0) > 10:
            metadata['health_status'] = 'unhealthy'
            return False
        
        # Check for excessive dropped events
        if metadata.get('dropped_events', 0) > settings.SSE_DROPPED_EVENT_THRESHOLD:
            metadata['health_status'] = 'degraded'
            # Still return True as connection is alive, just degraded
        
        # Check last activity
        last_activity = metadata.get('last_activity', datetime.utcnow())
        if (datetime.utcnow() - last_activity).total_seconds() > settings.SSE_CLIENT_TIMEOUT:
            metadata['health_status'] = 'inactive'
            return False
        
        return True
    
    def get_connection_info(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Get connection statistics"""
        if document_id:
            connections_info = []
            for conn_id in self._connections.get(document_id, {}):
                metadata = self._connection_metadata.get(conn_id, {})
                connections_info.append({
                    'connection_id': conn_id,
                    'health_status': metadata.get('health_status', 'unknown'),
                    'slow_consumer': metadata.get('slow_consumer', False),
                    'events_sent': metadata.get('events_sent', 0),
                    'dropped_events': metadata.get('dropped_events', 0),
                    'errors': metadata.get('errors', 0)
                })
            
            return {
                'document_id': document_id,
                'active_connections': len(self._connections.get(document_id, {})),
                'connections': connections_info,
                'event_history_size': len(self._event_history.get(document_id, []))
            }
        else:
            return {
                'total_documents': len(self._connections),
                'total_active_connections': sum(len(conns) for conns in self._connections.values()),
                'total_connections_created': self._total_connections,
                'total_events_sent': self._total_events_sent,
                'total_events_dropped': self._dropped_events,
                'connection_errors': self._connection_errors,
                'documents': {
                    doc_id: {
                        'connections': len(conns),
                        'history_size': len(self._event_history.get(doc_id, [])),
                        'slow_consumers': sum(
                            1 for conn_id in conns
                            if self._connection_metadata.get(conn_id, {}).get('slow_consumer', False)
                        )
                    }
                    for doc_id, conns in self._connections.items()
                }
            }
    
    async def cleanup_stale_connections(self, max_idle_seconds: int = 300):
        """Clean up stale connections"""
        now = datetime.utcnow()
        stale_connections = []
        
        for conn_id, metadata in list(self._connection_metadata.items()):
            last_activity = metadata.get('last_activity', now)
            if (now - last_activity).total_seconds() > max_idle_seconds:
                stale_connections.append((metadata['document_id'], conn_id))
        
        for doc_id, conn_id in stale_connections:
            logger.info(f"Cleaning up stale connection: {conn_id}")
            await self.remove_connection(doc_id, conn_id)
        
        # Force garbage collection if many connections were cleaned
        if len(stale_connections) > 10:
            gc.collect()

# Global SSE manager instance
sse_manager = SSEConnectionManager()

# Store active processing tasks with safe cleanup
processing_tasks = {}
processing_tasks_lock = asyncio.Lock()

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

# ===== SSE STREAMING ENDPOINT WITH CONNECTION HEALTH CHECKS =====

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
    Production-ready implementation with proper error handling and health checks.
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
    
    async def check_connection_health() -> bool:
        """Check if client is still connected"""
        try:
            if await request.is_disconnected():
                return False
            # Additional health check via SSE manager
            return await sse_manager.check_connection_health(connection_id)
        except:
            return False
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the client with health monitoring"""
        queue = None
        health_check_task = None
        
        try:
            # Add connection to manager
            queue = await sse_manager.add_connection(
                clean_document_id, 
                connection_id,
                send_history=True
            )
            
            logger.info(f"✅ SSE connection established: {connection_id}")
            
            # Send initial connection event
            yield f"event: connected\ndata: {{\"document_id\": \"{clean_document_id}\", \"connection_id\": \"{connection_id}\"}}\n\n"
            
            # Send current document status
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
                            "pages_processed": status_data.get('pages_processed', 0),
                            "total_pages": status_data.get('total_pages', 0)
                        }
                        yield f"event: status_update\ndata: {json.dumps(current_status)}\n\n"
                except Exception as e:
                    logger.error(f"Failed to get initial status: {e}")
            
            # Start periodic health check
            async def periodic_health_check():
                while True:
                    await asyncio.sleep(10)  # Check every 10 seconds
                    if not await check_connection_health():
                        logger.info(f"Health check failed for {connection_id}")
                        break
            
            health_check_task = asyncio.create_task(periodic_health_check())
            
            # Main event loop
            last_keepalive = time.time()
            connection_start = time.time()
            max_connection_time = 3600  # 1 hour max
            
            while True:
                # Check connection health
                if not await check_connection_health():
                    logger.info(f"🏥 Connection health check failed: {connection_id}")
                    yield f"event: health_failed\ndata: {{\"message\": \"Connection health check failed\"}}\n\n"
                    break
                
                # Check connection time limit
                if time.time() - connection_start > max_connection_time:
                    logger.info(f"⏰ Max connection time reached for {connection_id}")
                    yield f"event: connection_timeout\ndata: {{\"message\": \"Maximum connection time reached\"}}\n\n"
                    break
                
                try:
                    # Wait for events with timeout for keepalive
                    try:
                        event_data = await asyncio.wait_for(
                            queue.get(),
                            timeout=settings.SSE_KEEPALIVE_INTERVAL
                        )
                        
                        # Parse and send event
                        event = json.loads(event_data)
                        event_type = event['event_type']
                        data = event['data']
                        
                        # Format SSE event
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\nid: {event.get('timestamp', '')}\n\n"
                        
                        # Check if processing is complete
                        if event_type == SSEEventType.processing_complete:
                            logger.info(f"📍 Processing complete for {connection_id}")
                            # Give client time to process final event
                            await asyncio.sleep(5)
                            break
                        
                    except asyncio.TimeoutError:
                        # Send keepalive
                        current_time = time.time()
                        if current_time - last_keepalive >= settings.SSE_KEEPALIVE_INTERVAL:
                            yield f"event: keepalive\ndata: {{\"timestamp\": \"{datetime.utcnow().isoformat()}Z\"}}\n\n"
                            last_keepalive = current_time
                    
                except asyncio.CancelledError:
                    logger.info(f"🔌 SSE connection cancelled: {connection_id}")
                    yield f"event: cancelled\ndata: {{\"message\": \"Connection cancelled by server\"}}\n\n"
                    break
                
                except Exception as e:
                    logger.error(f"Error in SSE event loop: {e}")
                    logger.error(traceback.format_exc())
                    yield f"event: error\ndata: {{\"error\": \"Internal error in event stream\", \"details\": \"{str(e)}\"}}\n\n"
                    break
            
        except Exception as e:
            logger.error(f"Fatal error in SSE generator: {e}")
            logger.error(traceback.format_exc())
            yield f"event: fatal_error\ndata: {{\"error\": \"Fatal error occurred\", \"details\": \"{str(e)}\"}}\n\n"
        
        finally:
            # Cancel health check task
            if health_check_task and not health_check_task.done():
                health_check_task.cancel()
                try:
                    await health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Always clean up connection
            logger.info(f"🧹 Cleaning up SSE connection: {connection_id}")
            await sse_manager.remove_connection(clean_document_id, connection_id)
            yield f"event: close\ndata: {{\"message\": \"Connection closed\"}}\n\n"
    
    # Return SSE response with production headers
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
            "Access-Control-Allow-Origin": "*",  # CORS support
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "X-Content-Type-Options": "nosniff",
        }
    )

# ===== ASYNC PROCESSING WITH SSE EVENTS AND SAFE TASK CLEANUP =====

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
    Process PDF asynchronously with production SSE event emission and safe cleanup
    """
    processing_start_time = time.time()
    
    logger.info("="*50)
    logger.info(f"🔄 ASYNC PROCESSING STARTED: {session_id}")
    logger.info("="*50)
    
    try:
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
            'total_batches': 0,
            'sse_enabled': settings.ENABLE_SSE,
            'processing_stage': 'initializing'
        }
        
        logger.info("📝 Updating status to 'processing'...")
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Emit initial status event
        logger.info("📡 Emitting SSE event: processing_started")
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
        
        # Define robust event callback for PDF service
        async def processing_event_callback(event_type: str, event_data: Dict[str, Any]):
            """Callback to emit SSE events during processing"""
            logger.debug(f"📡 Event callback: {event_type} - {event_data}")
            
            try:
                if event_type == "page_processed":
                    # Update status
                    processing_status['pages_processed'] = event_data.get('page_number', 0)
                    processing_status['total_pages'] = event_data.get('total_pages', 0)
                    processing_status['processing_stage'] = f"processing_page_{event_data.get('page_number', 0)}"
                    
                    # Save status (will use thread-safe update in pdf_service)
                    await storage_service.upload_file(
                        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_status.json",
                        data=json.dumps(processing_status).encode('utf-8')
                    )
                    
                    # Emit event
                    await sse_manager.emit_event(
                        session_id,
                        SSEEventType.status_update,
                        StatusUpdateEvent(
                            stage="processing_pages",
                            message=f"Processing page {event_data['page_number']} of {event_data['total_pages']}",
                            progress_percent=event_data.get('progress_percent', 0),
                            pages_processed=event_data['page_number'],
                            estimated_time=event_data.get('estimated_time')
                        ).dict()
                    )
                    
                    logger.info(f"✅ Page {event_data['page_number']}/{event_data['total_pages']} processed")
                
                elif event_type == "resource_ready":
                    resource_type = event_data.get('resource_type', 'unknown')
                    page_number = event_data.get('page_number')
                    
                    logger.info(f"📦 Resource ready: {resource_type} for page {page_number}")
                    
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
                    elif resource_type == "grid_system":
                        url = f"/api/v1/documents/{session_id}/pages/{page_number}/grid"
                    
                    await sse_manager.emit_event(
                        session_id,
                        SSEEventType.resource_ready,
                        ResourceReadyEvent(
                            resource_type=resource_type,
                            resource_id=event_data.get('resource_id', f"{resource_type}_{page_number}"),
                            page_number=page_number,
                            url=url,
                            metadata=event_data.get('metadata', {})
                        ).dict()
                    )
                
                elif event_type == "batch_complete":
                    logger.info(f"📦 Batch {event_data['batch_number']}/{event_data['total_batches']} complete")
                    
                    await sse_manager.emit_event(
                        session_id,
                        SSEEventType.status_update,
                        StatusUpdateEvent(
                            stage="batch_complete",
                            message=f"Completed batch {event_data['batch_number']} of {event_data['total_batches']}",
                            progress_percent=event_data.get('progress_percent', 0),
                            pages_processed=event_data.get('pages_processed', 0)
                        ).dict()
                    )
                
                elif event_type == "extraction_complete":
                    logger.info("📝 Text extraction completed")
                    
                    await sse_manager.emit_event(
                        session_id,
                        SSEEventType.status_update,
                        StatusUpdateEvent(
                            stage="extraction_complete",
                            message="Text extraction completed",
                            progress_percent=90,
                            pages_processed=processing_status.get('pages_processed', 0)
                        ).dict()
                    )
                
            except Exception as e:
                logger.error(f"Failed to emit processing event: {e}")
                logger.error(traceback.format_exc())
        
        # Process the PDF with event callback
        logger.info("🚀 Calling pdf_service.process_and_cache_pdf...")
        
        await pdf_service.process_and_cache_pdf(
            session_id=session_id,
            pdf_bytes=contents,
            storage_service=storage_service,
            event_callback=processing_event_callback if settings.ENABLE_SSE else None
        )
        
        logger.info("✅ PDF processing completed successfully")
        
        # Calculate processing time
        processing_time = time.time() - processing_start_time
        
        # Final status update
        processing_status['status'] = 'ready'
        processing_status['completed_at'] = datetime.utcnow().isoformat() + 'Z'
        processing_status['processing_time_seconds'] = round(processing_time, 2)
        processing_status['processing_stage'] = 'completed'
        
        # Get final metadata
        try:
            metadata_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json"
            )
            metadata = json.loads(metadata_text)
            processing_status['pages_processed'] = metadata.get('page_count', 0)
            processing_status['total_pages'] = metadata.get('total_pages', 0)
            processing_status['metadata'] = metadata
            
            logger.info(f"📊 Metadata loaded: {metadata.get('page_count', 0)} pages")
            
        except Exception as e:
            logger.warning(f"Could not load final metadata: {e}")
        
        logger.info("📝 Updating final status to 'ready'...")
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Emit completion event
        logger.info("📡 Emitting SSE event: processing_complete")
        await sse_manager.emit_event(
            session_id,
            SSEEventType.processing_complete,
            ProcessingCompleteEvent(
                status="ready",
                total_pages=processing_status.get('pages_processed', 0),
                processing_time=processing_time,
                resources_summary={
                    "pages_with_images": processing_status.get('pages_processed', 0),
                    "pages_with_thumbnails": processing_status.get('pages_processed', 0),
                    "context_available": True,
                    "metadata_available": True,
                    "grid_systems_available": True
                }
            ).dict()
        )
        
        # Update session if available
        if session_service and hasattr(session_service, 'update_session_metadata'):
            try:
                await session_service.update_session_metadata(
                    document_id=session_id,
                    metadata={
                        'processing_complete': True,
                        'status': 'ready',
                        'pages_processed': processing_status.get('pages_processed', 0),
                        'processing_time_seconds': processing_time
                    }
                )
                logger.info("✅ Session metadata updated")
            except Exception as e:
                logger.warning(f"Session update failed: {e}")
        
        logger.info("="*50)
        logger.info(f"✅ ASYNC PROCESSING COMPLETED: {session_id}")
        logger.info(f"   Total time: {processing_time:.2f}s")
        logger.info(f"   Pages: {processing_status.get('pages_processed', 0)}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f"❌ ASYNC PROCESSING FAILED: {session_id}")
        logger.error(f"   Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("="*50)
        
        # Calculate how long we processed before failure
        processing_time = time.time() - processing_start_time
        
        # Save error status
        error_status = {
            'document_id': session_id,
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'error_traceback': traceback.format_exc(),
            'failed_at': datetime.utcnow().isoformat() + 'Z',
            'filename': clean_filename,
            'author': author,
            'processing_time_before_error': round(processing_time, 2),
            'pages_processed': processing_status.get('pages_processed', 0),
            'processing_stage': processing_status.get('processing_stage', 'unknown')
        }
        
        try:
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_status.json",
                data=json.dumps(error_status).encode('utf-8')
            )
            logger.info("✅ Error status saved")
        except Exception as storage_error:
            logger.error(f"Failed to save error status: {storage_error}")
        
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
                    'pages_processed': processing_status.get('pages_processed', 0),
                    'stage': processing_status.get('processing_stage', 'unknown')
                }
            ).dict()
        )
    
    finally:
        # Safe task cleanup with lock
        async with processing_tasks_lock:
            # Use pop to safely remove without KeyError
            task = processing_tasks.pop(session_id, None)
            if task:
                logger.info(f"📋 Removed from processing tasks. Active tasks: {len(processing_tasks)}")
            else:
                logger.warning(f"Task {session_id} was already removed from processing tasks")
        
        # Log final metrics
        logger.info(f"📊 Processing metrics for {session_id}:")
        logger.info(f"   Total time: {time.time() - processing_start_time:.2f}s")
        logger.info(f"   Pages processed: {processing_status.get('pages_processed', 0)}")
        logger.info(f"   SSE connections: {sse_manager.get_connection_info(session_id)}")

# ===== WRITE OPERATION ROUTES =====

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
    Upload a PDF document for processing with enhanced debugging.
    Returns immediately with document ID while processing happens asynchronously.
    """
    
    logger.info("="*50)
    logger.info("📤 UPLOAD ENDPOINT CALLED")
    logger.info("="*50)
    
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
        logger.info(f"✅ File read successfully: {len(contents)} bytes")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file"
        )
    
    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    max_size_mb = settings.MAX_FILE_SIZE_MB
    
    logger.info(f"📊 File size: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")
    
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
    
    logger.info(f"📋 Session ID: {session_id}")
    logger.info(f"📄 Filename: {clean_filename}")
    logger.info(f"👤 Author: {author}")
    if trade:
        logger.info(f"🔨 Trade: {trade}")
    
    try:
        # Check available services
        logger.info("🔍 Checking available services...")
        
        storage_service = getattr(request.app.state, 'storage_service', None)
        pdf_service = getattr(request.app.state, 'pdf_service', None)
        session_service = getattr(request.app.state, 'session_service', None)
        
        logger.info(f"   Storage Service: {'✅ Available' if storage_service else '❌ NOT AVAILABLE'}")
        logger.info(f"   PDF Service: {'✅ Available' if pdf_service else '❌ NOT AVAILABLE'}")
        logger.info(f"   Session Service: {'✅ Available' if session_service else '❌ NOT AVAILABLE'}")
        
        if not storage_service:
            logger.error("Storage service not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="Storage service is not available. Please try again later."
            )
        
        # Upload original PDF
        pdf_blob_name = f"{session_id}.pdf"
        logger.info(f"📤 Uploading to blob: {pdf_blob_name}")
        
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
        
        logger.info("📝 Creating status file...")
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_status.json",
            data=json.dumps(initial_status).encode('utf-8')
        )
        
        logger.info("✅ Status file created")
        
        # Create session if service available
        if session_service:
            try:
                logger.info("📊 Creating session...")
                await session_service.create_session(
                    document_id=session_id,
                    original_filename=clean_filename
                )
                logger.info("✅ Created session for document")
            except Exception as e:
                logger.warning(f"Session creation failed (non-critical): {e}")
        
        # Check PDF service
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
        logger.info("🚀 Starting async processing task...")
        
        try:
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
            
            # Store task reference with lock
            async with processing_tasks_lock:
                processing_tasks[session_id] = task
            
            logger.info(f"✅ Async task created successfully")
            logger.info(f"📋 Active processing tasks: {len(processing_tasks)}")
            
            # Add a callback to log when task completes (with safe cleanup)
            def task_done_callback(future):
                try:
                    result = future.result()
                    logger.info(f"✅ Task completed for {session_id}")
                except Exception as e:
                    logger.error(f"❌ Task failed for {session_id}: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    # Safe cleanup
                    asyncio.create_task(async_task_cleanup(session_id))
            
            async def async_task_cleanup(task_id: str):
                """Async cleanup with lock"""
                async with processing_tasks_lock:
                    processing_tasks.pop(task_id, None)
            
            task.add_done_callback(task_done_callback)
            
        except Exception as e:
            logger.error(f"❌ Failed to create async task: {e}")
            logger.error(traceback.format_exc())
            
            # Update status to error
            error_status = {
                'document_id': session_id,
                'status': 'error',
                'error': f"Failed to start processing: {str(e)}",
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
                message=f"Processing failed to start: {str(e)}",
                file_size_mb=round(file_size_mb, 2)
            )
        
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
    Requires confirmation parameter to prevent accidental deletioghfn.
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
        
        # Cancel any active processing tasks
        async with processing_tasks_lock:
            task = processing_tasks.pop(clean_document_id, None)
            if task and not task.done():
                task.cancel()
                logger.info(f"Cancelled active processing task for {clean_document_id}")
        
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
    
    # Get connection info
    connection_info = sse_manager.get_connection_info()
    
    # Get processing tasks info with lock
    async with processing_tasks_lock:
        tasks_info = {
            doc_id: "running" if not task.done() else "completed"
            for doc_id, task in processing_tasks.items()
        }
    
    return {
        **connection_info,
        "processing_tasks": tasks_info
    }

# ===== DEBUG ENDPOINTS =====

@blueprint_router.get(
    "/debug/service-health",
    summary="Check service health and initialization",
    description="Debug endpoint to verify all services are properly initialized"
)
async def check_service_health(request: Request):
    """
    Check the health and availability of all services
    """
    health_status = {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "services": {},
        "settings": {},
        "tasks": {}
    }
    
    # Check storage service
    storage_service = getattr(request.app.state, 'storage_service', None)
    if storage_service:
        try:
            # Try to list blobs to verify connection
            test_list = await storage_service.list_blobs(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                prefix="_test_"
            )
            health_status["services"]["storage"] = {
                "available": True,
                "status": "healthy",
                "connection_info": storage_service.get_connection_info()
            }
        except Exception as e:
            health_status["services"]["storage"] = {
                "available": True,
                "status": "error",
                "error": str(e)
            }
    else:
        health_status["services"]["storage"] = {
            "available": False,
            "status": "not_initialized"
        }
    
    # Check PDF service
    pdf_service = getattr(request.app.state, 'pdf_service', None)
    if pdf_service:
        try:
            stats = pdf_service.get_processing_stats()
            health_status["services"]["pdf"] = {
                "available": True,
                "status": "healthy",
                "stats": stats
            }
        except Exception as e:
            health_status["services"]["pdf"] = {
                "available": True,
                "status": "error",
                "error": str(e)
            }
    else:
        health_status["services"]["pdf"] = {
            "available": False,
            "status": "not_initialized"
        }
    
    # Check session service
    session_service = getattr(request.app.state, 'session_service', None)
    health_status["services"]["session"] = {
        "available": session_service is not None,
        "status": "healthy" if session_service else "not_initialized"
    }
    
    # Check AI service
    ai_service = getattr(request.app.state, 'ai_service', None)
    health_status["services"]["ai"] = {
        "available": ai_service is not None,
        "status": "healthy" if ai_service else "not_initialized"
    }
    
    # Check SSE manager
    health_status["services"]["sse"] = {
        "available": True,
        "enabled": settings.ENABLE_SSE,
        "connections": sse_manager.get_connection_info()
    }
    
    # Check important settings
    health_status["settings"] = {
        "ENABLE_SSE": settings.ENABLE_SSE,
        "MAX_FILE_SIZE_MB": settings.MAX_FILE_SIZE_MB,
        "PDF_MAX_PAGES": settings.PDF_MAX_PAGES,
        "AZURE_CONTAINER_NAME": settings.AZURE_CONTAINER_NAME,
        "AZURE_CACHE_CONTAINER_NAME": settings.AZURE_CACHE_CONTAINER_NAME,
        "AZURE_STORAGE_CONNECTION_STRING": bool(settings.AZURE_STORAGE_CONNECTION_STRING),
        "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
        "SSE_SLOW_CONSUMER_THRESHOLD": settings.SSE_SLOW_CONSUMER_THRESHOLD,
        "SSE_DROPPED_EVENT_THRESHOLD": settings.SSE_DROPPED_EVENT_THRESHOLD
    }
    
    # Check active processing tasks
    async with processing_tasks_lock:
        health_status["tasks"] = {
            "active_processing_tasks": len(processing_tasks),
            "task_ids": list(processing_tasks.keys()),
            "task_status": {
                task_id: "running" if not task.done() else "completed"
                for task_id, task in processing_tasks.items()
            }
        }
    
    # Overall health
    all_services_healthy = all(
        service.get("available", False) and service.get("status") != "error"
        for service in health_status["services"].values()
    )
    
    health_status["overall_health"] = "healthy" if all_services_healthy else "degraded"
    
    return health_status

@blueprint_router.post(
    "/debug/test-processing",
    summary="Test PDF processing with a small file",
    description="Debug endpoint to test PDF processing pipeline"
)
async def test_pdf_processing(
    request: Request,
    test_type: str = Query("minimal", description="Type of test: minimal, full")
):
    """
    Test PDF processing with a minimal PDF file
    """
    try:
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        return {
            "error": "reportlab not installed",
            "message": "Please install reportlab: pip install reportlab"
        }
    
    # Create a simple test PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    if test_type == "minimal":
        # Single page PDF
        c.drawString(100, 750, "Test PDF Document")
        c.drawString(100, 700, "This is a test page for debugging.")
        c.drawString(100, 650, "Grid Reference: A-1")
        c.showPage()
    else:
        # Multi-page PDF
        for i in range(3):
            c.drawString(100, 750, f"Test PDF - Page {i+1}")
            c.drawString(100, 700, f"Grid Reference: {chr(65+i)}-{i+1}")
            c.showPage()
    
    c.save()
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Create a mock upload file
    test_file = UploadFile(
        filename="test_debug.pdf",
        file=io.BytesIO(pdf_bytes)
    )
    
    # Process through normal upload endpoint
    try:
        response = await upload_document(
            request=request,
            file=test_file,
            author="Debug Test",
            trade="Testing"
        )
        
        return {
            "test_type": test_type,
            "pdf_size_bytes": len(pdf_bytes),
            "upload_response": response.dict(),
            "debug_info": {
                "services_available": {
                    "storage": getattr(request.app.state, 'storage_service', None) is not None,
                    "pdf": getattr(request.app.state, 'pdf_service', None) is not None,
                    "session": getattr(request.app.state, 'session_service', None) is not None,
                    "ai": getattr(request.app.state, 'ai_service', None) is not None
                }
            }
        }
    except Exception as e:
        return {
            "test_type": test_type,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@blueprint_router.post(
    "/debug/retry-processing/{document_id}",
    summary="Manually retry processing for a stuck document",
    description="Debug endpoint to retry processing if async task failed"
)
async def retry_document_processing(
    request: Request,
    document_id: str
):
    """
    Manually retry processing for a document
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    pdf_service = getattr(request.app.state, 'pdf_service', None)
    
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    if not pdf_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF service unavailable"
        )
    
    try:
        # Check if document exists
        pdf_exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        if not pdf_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {clean_document_id} not found"
            )
        
        # Download PDF
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        # Get current status
        try:
            status_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_status.json"
            )
            current_status = json.loads(status_text)
        except:
            current_status = {}
        
        # Start processing
        task = asyncio.create_task(
            process_pdf_async(
                session_id=clean_document_id,
                contents=pdf_bytes,
                clean_filename=current_status.get('filename', 'unknown.pdf'),
                author=current_status.get('author', 'Unknown'),
                storage_service=storage_service,
                pdf_service=pdf_service,
                session_service=getattr(request.app.state, 'session_service', None)
            )
        )
        
        # Store task with lock
        async with processing_tasks_lock:
            processing_tasks[clean_document_id] = task
        
        return {
            "status": "success",
            "message": f"Reprocessing started for document {clean_document_id}",
            "document_id": clean_document_id,
            "active_tasks": len(processing_tasks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry processing: {str(e)}"
        )

# ===== CLEANUP TASK =====

async def periodic_cleanup():
    """Periodic cleanup of stale connections and tasks"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Cleanup stale SSE connections
            await sse_manager.cleanup_stale_connections(max_idle_seconds=600)  # 10 minute timeout
            
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
            
            # Force garbage collection if needed
            if len(completed_tasks) > 5:
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# Start cleanup task when module loads
try:
    asyncio.create_task(periodic_cleanup())
except RuntimeError:
    # Ignore if no event loop is running yet
    pass

# NOTE: READ operations (status, download, info) remain in document_routes.py
