# app/api/routes/document_routes.py - READ OPERATIONS WITH SSE AWARENESS

"""
Document Read Operations - Handles all document queries, status checks, and data retrieval
Optimized for progressive loading and efficient resource checking
Now includes SSE connection information for real-time updates
"""

from fastapi import APIRouter, Request, HTTPException, Query, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
import logging
import re
import asyncio
from app.core.config import get_settings

# Schema imports
from app.models.schemas import (
    DocumentInfoResponse, DocumentStatsResponse, 
    DocumentActivityResponse, CollaboratorsResponse,
    CollaboratorStats
)

document_router = APIRouter(
    prefix="/api/v1",
    tags=["Document Read Operations"]
)

settings = get_settings()
logger = logging.getLogger(__name__)

# ===== UTILITY FUNCTIONS =====

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID"""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be a non-empty string"
        )
    
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip()).strip('_')

    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters long"
        )
    
    if len(clean_id) > 50:
        raise HTTPException(
            status_code=400,
            detail="Document ID must be 50 characters or less"
        )
    
    return clean_id

async def load_document_resources(document_id: str, storage_service) -> Dict[str, Any]:
    """Load commonly used document resources"""
    cache_container = settings.AZURE_CACHE_CONTAINER_NAME
    
    # Parallel load of common resources
    tasks = {
        'status': storage_service.download_blob_as_json(
            container_name=cache_container,
            blob_name=f"{document_id}_status.json"
        ),
        'metadata': storage_service.download_blob_as_json(
            container_name=cache_container,
            blob_name=f"{document_id}_metadata.json"
        ),
        'annotations': storage_service.download_blob_as_json(
            container_name=cache_container,
            blob_name=f"{document_id}_annotations.json"
        ),
        'chats': storage_service.download_blob_as_json(
            container_name=cache_container,
            blob_name=f"{document_id}_all_chats.json"
        ),
        'notes': storage_service.download_blob_as_json(
            container_name=cache_container,
            blob_name=f"{document_id}_notes.json"
        )
    }
    
    results = {}
    for key, task in tasks.items():
        try:
            results[key] = await task
        except:
            results[key] = None if key in ['metadata', 'status'] else []
    
    return results

# ===== ENHANCED STATUS ENDPOINT WITH SSE INFO =====

class DocumentStatusResponse(BaseModel):
    """Enhanced status response for progressive loading with SSE support"""
    document_id: str
    status: str
    message: Optional[str] = ""
    total_pages: Optional[int] = None
    pages_processed: int = 0
    available_resources: Dict[str, Any]
    error: Optional[str] = None
    uploaded_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    processing_progress_percent: int = 0
    # SSE Support
    sse_enabled: bool = False
    sse_connection_url: Optional[str] = None
    include_sse_info: Optional[bool] = None

@document_router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Check document processing status with resource availability",
    description="Get detailed processing status including which resources are ready for progressive loading and SSE connection info"
)
async def check_document_status_enhanced(
    request: Request,
    document_id: str,
    include_resource_check: bool = Query(True, description="Check individual resource availability"),
    include_sse_info: bool = Query(True, description="Include SSE connection information")
):
    """
    Enhanced status check that includes detailed resource availability
    for progressive loading support and SSE connection information.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        main_container = settings.AZURE_CONTAINER_NAME
        
        # Check for status file
        status_blob = f"{clean_document_id}_status.json"
        status_data = {}
        
        if await storage_service.blob_exists(cache_container, status_blob):
            status_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=status_blob
            )
            status_data = json.loads(status_text)
        
        # Initialize resource availability
        available_resources = {
            "pdf": False,
            "pages_with_images": [],
            "pages_with_thumbnails": [],
            "pages_with_ai_images": [],
            "context_text": False,
            "metadata": False,
            "grid_systems": False,
            "document_index": False
        }
        
        if include_resource_check:
            # Check PDF availability
            available_resources["pdf"] = await storage_service.blob_exists(
                main_container, f"{clean_document_id}.pdf"
            )
            
            # Check metadata files
            metadata_checks = await asyncio.gather(
                storage_service.blob_exists(cache_container, f"{clean_document_id}_context.txt"),
                storage_service.blob_exists(cache_container, f"{clean_document_id}_metadata.json"),
                storage_service.blob_exists(cache_container, f"{clean_document_id}_grid_systems.json"),
                storage_service.blob_exists(cache_container, f"{clean_document_id}_document_index.json"),
                return_exceptions=True
            )
            
            available_resources["context_text"] = metadata_checks[0] if not isinstance(metadata_checks[0], Exception) else False
            available_resources["metadata"] = metadata_checks[1] if not isinstance(metadata_checks[1], Exception) else False
            available_resources["grid_systems"] = metadata_checks[2] if not isinstance(metadata_checks[2], Exception) else False
            available_resources["document_index"] = metadata_checks[3] if not isinstance(metadata_checks[3], Exception) else False
            
            # Check page resources efficiently
            pages_to_check = status_data.get('pages_processed', 0)
            if pages_to_check > 0:
                # Batch check for efficiency
                batch_size = 10
                for batch_start in range(0, pages_to_check, batch_size):
                    batch_end = min(batch_start + batch_size, pages_to_check)
                    
                    # Create tasks for parallel checking
                    tasks = []
                    for page in range(batch_start + 1, batch_end + 1):
                        tasks.extend([
                            storage_service.blob_exists(cache_container, f"{clean_document_id}_page_{page}.jpg"),
                            storage_service.blob_exists(cache_container, f"{clean_document_id}_page_{page}_thumb.jpg"),
                            storage_service.blob_exists(cache_container, f"{clean_document_id}_page_{page}_ai.jpg")
                        ])
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Process results
                        for i, page in enumerate(range(batch_start + 1, batch_end + 1)):
                            base_idx = i * 3
                            if base_idx < len(results):
                                if results[base_idx] and not isinstance(results[base_idx], Exception):
                                    available_resources["pages_with_images"].append(page)
                                if base_idx + 1 < len(results) and results[base_idx + 1] and not isinstance(results[base_idx + 1], Exception):
                                    available_resources["pages_with_thumbnails"].append(page)
                                if base_idx + 2 < len(results) and results[base_idx + 2] and not isinstance(results[base_idx + 2], Exception):
                                    available_resources["pages_with_ai_images"].append(page)
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        if status_data.get('status') == 'processing' and status_data.get('started_at'):
            try:
                started = datetime.fromisoformat(status_data['started_at'].rstrip('Z'))
                elapsed = (datetime.utcnow() - started).total_seconds()
                
                pages_done = status_data.get('pages_processed', 0)
                total_pages = status_data.get('total_pages', 1)
                
                if pages_done > 0 and pages_done < total_pages:
                    time_per_page = elapsed / pages_done
                    pages_remaining = total_pages - pages_done
                    estimated_time_remaining = int(time_per_page * pages_remaining)
                else:
                    estimated_time_remaining = max(0, status_data.get('estimated_processing_time', 60) - int(elapsed))
            except:
                pass
        
        # Determine overall status
        if not status_data and not available_resources["pdf"]:
            status = "not_found"
            message = "Document not found"
        else:
            status = status_data.get('status', 'unknown')
            message = status_data.get('message', '')
            
            if status == 'processing':
                pages_done = status_data.get('pages_processed', 0)
                total_pages = status_data.get('total_pages', 0)
                message = f"Processing... {pages_done}/{total_pages} pages complete"
            elif status == 'ready':
                message = "Document ready for viewing"
            elif status == 'error':
                message = f"Processing failed: {status_data.get('error', 'Unknown error')}"
        
        # Prepare SSE information
        sse_enabled = settings.ENABLE_SSE and status_data.get('sse_enabled', settings.ENABLE_SSE)
        sse_connection_url = None
        
        if include_sse_info and sse_enabled and status == 'processing':
            # Only provide SSE URL during processing
            sse_connection_url = f"/api/v1/documents/{clean_document_id}/status/stream"
        
        return DocumentStatusResponse(
            document_id=clean_document_id,
            status=status,
            message=message,
            total_pages=status_data.get('total_pages'),
            pages_processed=status_data.get('pages_processed', 0),
            available_resources=available_resources,
            error=status_data.get('error'),
            uploaded_at=status_data.get('uploaded_at'),
            started_at=status_data.get('started_at'),
            completed_at=status_data.get('completed_at'),
            estimated_time_remaining=estimated_time_remaining,
            processing_progress_percent=round(
                (status_data.get('pages_processed', 0) / max(status_data.get('total_pages', 1), 1)) * 100
            ),
            sse_enabled=sse_enabled,
            sse_connection_url=sse_connection_url,
            include_sse_info=include_sse_info
        )
        
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")

# ===== PAGE RESOURCE ENDPOINTS =====

@document_router.get(
    "/documents/{document_id}/pages/{page_num}/resources",
    summary="Check page resource availability",
    description="Check what resources are available for a specific page"
)
async def get_page_resources(
    request: Request,
    document_id: str,
    page_num: int  # ✅ Path parameter
):
    """Check what resources are available for a specific page."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Parallel check all resources for this page
        resources = await asyncio.gather(
            storage_service.blob_exists(cache_container, f"{clean_document_id}_page_{page_num}.jpg"),
            storage_service.blob_exists(cache_container, f"{clean_document_id}_page_{page_num}_ai.jpg"),
            storage_service.blob_exists(cache_container, f"{clean_document_id}_page_{page_num}_thumb.jpg"),
            return_exceptions=True
        )
        
        # Get grid system for this page
        grid_system = None
        try:
            grid_blob = f"{clean_document_id}_grid_systems.json"
            if await storage_service.blob_exists(cache_container, grid_blob):
                grid_data = await storage_service.download_blob_as_json(
                    container_name=cache_container,
                    blob_name=grid_blob
                )
                grid_system = grid_data.get(str(page_num))
        except:
            pass
        
        return {
            "page": page_num,
            "has_full_image": resources[0] if not isinstance(resources[0], Exception) else False,
            "has_ai_image": resources[1] if not isinstance(resources[1], Exception) else False,
            "has_thumbnail": resources[2] if not isinstance(resources[2], Exception) else False,
            "grid_system": grid_system,
            "urls": {
                "full_image": f"/api/v1/documents/{clean_document_id}/pages/{page_num}/image" if resources[0] else None,
                "ai_image": f"/api/v1/documents/{clean_document_id}/pages/{page_num}/image?type=ai" if resources[1] else None,
                "thumbnail": f"/api/v1/documents/{clean_document_id}/pages/{page_num}/image?type=thumb" if resources[2] else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to check page resources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check page resources: {str(e)}")

@document_router.get(
    "/documents/{document_id}/pages/{page_num}/image",
    summary="Get page image",
    description="Get a specific page image (full, ai, or thumbnail)"
)
async def get_page_image(
    request: Request,
    document_id: str,
    page_num: int,  # ✅ Path parameter - no Query()
    type: str = Query("full", regex="^(full|ai|thumb)$", description="Image type")
):
    """Get a specific page image."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Determine blob name
        suffix_map = {"full": "", "ai": "_ai", "thumb": "_thumb"}
        blob_name = f"{clean_document_id}_page_{page_num}{suffix_map[type]}.jpg"
        
        # Get image
        image_data = await storage_service.download_blob_as_bytes(
            container_name=cache_container,
            blob_name=blob_name
        )
        
        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Length": str(len(image_data))
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found for page {page_num}")
    except Exception as e:
        logger.error(f"Failed to get page image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get page image: {str(e)}")

# ===== DOCUMENT DOWNLOAD =====

@document_router.get(
    "/documents/{document_id}/download",
    summary="Download original PDF",
    description="Download the original PDF file for viewing"
)
async def download_document_pdf(
    request: Request,
    document_id: str
):
    """Download the original PDF file."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        logger.info(f"📥 Downloading PDF for document: {clean_document_id}")
        
        # Get PDF from main container
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        logger.info(f"✅ PDF downloaded successfully: {len(pdf_bytes)} bytes")
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={clean_document_id}.pdf",
                "Cache-Control": "public, max-age=3600",
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"PDF not found for document {clean_document_id}")
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")

# ===== DOCUMENT INFO WITH SSE SUPPORT =====

@document_router.get(
    "/documents/{document_id}/info",
    response_model=DocumentInfoResponse,
    summary="Get document information",
    description="Get comprehensive document information including metadata and collaboration stats"
)
async def get_document_info(request: Request, document_id: str):
    """Get comprehensive document information."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Check basic existence
        main_container = settings.AZURE_CONTAINER_NAME
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        pdf_exists = await storage_service.blob_exists(
            container_name=main_container,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        if not pdf_exists:
            return DocumentInfoResponse(
                document_id=clean_document_id,
                status="not_found",
                message="Document not found",
                exists=False
            )
        
        # Load resources
        resources = await load_document_resources(clean_document_id, storage_service)
        
        # Determine status
        status_data = resources.get('status', {})
        metadata = resources.get('metadata', {})
        
        if status_data.get('status') == 'error':
            status = "error"
            message = f"Processing failed: {status_data.get('error', 'Unknown error')}"
        elif status_data.get('status') == 'processing':
            status = "processing"
            message = f"Processing... {status_data.get('pages_processed', 0)}/{status_data.get('total_pages', 0)} pages"
        elif metadata:
            status = "ready"
            message = f"Document ready. {metadata.get('page_count', 0)} pages processed."
        else:
            status = "uploaded"
            message = "Document uploaded, processing pending"
        
        # Calculate collaboration stats
        notes = resources.get('notes', [])
        published_notes = len([n for n in notes if not n.get('is_private', True)])
        active_collaborators = len(set(n.get('author') for n in notes if not n.get('is_private', True)))
        
        # Get session info
        session_info = None
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service:
            try:
                session_info = await session_service.get_session_info(clean_document_id)
            except:
                pass
        
        # Add SSE info to metadata if processing
        if status == "processing" and settings.ENABLE_SSE:
            if not metadata:
                metadata = {}
            metadata["sse_available"] = True
            metadata["sse_url"] = f"/api/v1/documents/{clean_document_id}/status/stream"
        
        return DocumentInfoResponse(
            document_id=clean_document_id,
            status=status,
            message=message,
            exists=True,
            metadata={
                "page_count": metadata.get('page_count'),
                "total_pages": metadata.get('total_pages'),
                "processing_time": metadata.get('processing_time'),
                "file_size_mb": metadata.get('file_size_mb'),
                "grid_systems_detected": metadata.get('grid_systems_detected', 0),
                "has_text": metadata.get('extraction_summary', {}).get('has_text', False),
                "uploaded_at": status_data.get('uploaded_at'),
                "completed_at": status_data.get('completed_at'),
                "sse_available": metadata.get('sse_available', False),
                "sse_url": metadata.get('sse_url')
            },
            total_published_notes=published_notes,
            active_collaborators=active_collaborators,
            recent_public_activity=published_notes > 0,
            session_info=session_info
        )
        
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

# ===== DOCUMENT STATISTICS =====

@document_router.get(
    "/documents/{document_id}/stats",
    response_model=DocumentStatsResponse,
    summary="Get document statistics",
    description="Get detailed statistics about document usage and collaboration"
)
async def get_document_statistics(request: Request, document_id: str):
    """Get comprehensive document statistics."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = getattr(request.app.state, 'storage_service', None)
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")

        # Load all resources
        resources = await load_document_resources(clean_document_id, storage_service)
        
        if not resources.get('metadata'):
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        # Calculate statistics
        annotations = resources.get('annotations', [])
        chats = resources.get('chats', [])
        metadata = resources.get('metadata', {})
        
        # Get unique collaborators
        all_authors = set()
        all_authors.update(ann.get('author') for ann in annotations if ann.get('author'))
        all_authors.update(chat.get('author') for chat in chats if chat.get('author'))
        
        # Count annotation types
        annotation_types = {}
        for ann in annotations:
            ann_type = ann.get('annotation_type', 'note')
            annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
        
        # Find last activity
        all_timestamps = []
        for item in annotations + chats:
            if item.get('timestamp'):
                all_timestamps.append(item['timestamp'])
        last_activity = max(all_timestamps) if all_timestamps else None
        
        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=len(annotations),
            total_pages=metadata.get('total_pages', metadata.get('page_count', 1)),
            total_characters=len(resources.get('context', '')),
            estimated_tokens=len(resources.get('context', '')) // 4,
            unique_collaborators=list(all_authors),
            collaborator_count=len(all_authors),
            annotation_types=annotation_types,
            last_activity=last_activity,
            status="ready" if metadata else "processing"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")

# ===== DOCUMENT ACTIVITY =====

@document_router.get(
    "/documents/{document_id}/activity",
    response_model=DocumentActivityResponse,
    summary="Get recent document activity",
    description="Get recent annotations and chats for a document"
)
async def get_document_activity(
    request: Request, 
    document_id: str, 
    limit: int = Query(10, ge=1, le=50, description="Number of recent items")
):
    """Get recent activity for a document."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = getattr(request.app.state, 'storage_service', None)
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")

        # Load resources
        resources = await load_document_resources(clean_document_id, storage_service)
        
        if not resources.get('metadata'):
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        annotations = resources.get('annotations', [])
        chats = resources.get('chats', [])
        
        # Combine and sort by timestamp
        all_activity = []
        
        for ann in annotations:
            if ann.get('timestamp'):
                all_activity.append({**ann, 'activity_type': 'annotation'})
        
        for chat in chats:
            if chat.get('timestamp'):
                all_activity.append({**chat, 'activity_type': 'chat'})
        
        # Sort by timestamp (most recent first)
        all_activity.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Get recent items
        recent_activity = all_activity[:limit]
        
        # Separate and get active authors
        recent_annotations = [item for item in recent_activity if item.get('activity_type') == 'annotation']
        recent_chats = [item for item in recent_activity if item.get('activity_type') == 'chat']
        active_authors = list(set(item.get('author') for item in recent_activity if item.get('author')))
        
        return DocumentActivityResponse(
            document_id=clean_document_id,
            recent_annotations=recent_annotations,
            recent_chats=recent_chats,
            active_authors=active_authors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document activity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document activity: {str(e)}")

# ===== COLLABORATORS =====

@document_router.get(
    "/documents/{document_id}/collaborators",
    response_model=CollaboratorsResponse,
    summary="Get document collaborators",
    description="Get list of all collaborators and their activity statistics"
)
async def get_document_collaborators(request: Request, document_id: str):
    """Get collaborator statistics for a document."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = getattr(request.app.state, 'storage_service', None)
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")

        # Load resources
        resources = await load_document_resources(clean_document_id, storage_service)
        
        if not resources.get('metadata'):
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        annotations = resources.get('annotations', [])
        chats = resources.get('chats', [])
        
        # Calculate per-author statistics
        author_stats = {}
        
        for ann in annotations:
            author = ann.get('author')
            if author:
                if author not in author_stats:
                    author_stats[author] = {'annotations': 0, 'chats': 0}
                author_stats[author]['annotations'] += 1
        
        for chat in chats:
            author = chat.get('author')
            if author:
                if author not in author_stats:
                    author_stats[author] = {'annotations': 0, 'chats': 0}
                author_stats[author]['chats'] += 1
        
        # Convert to response format
        collaborators = []
        for author, stats in author_stats.items():
            collaborators.append(CollaboratorStats(
                author=author,
                annotations_count=stats['annotations'],
                chats_count=stats['chats'],
                total_interactions=stats['annotations'] + stats['chats']
            ))
        
        # Sort by total interactions
        collaborators.sort(key=lambda x: x.total_interactions, reverse=True)
        
        return CollaboratorsResponse(
            document_id=clean_document_id,
            collaborators=collaborators,
            total_collaborators=len(collaborators)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collaborators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collaborators: {str(e)}")

# ===== DOCUMENT SUMMARY WITH SSE INFO =====

@document_router.get(
    "/documents/{document_id}/summary",
    summary="Get document summary",
    description="Get a comprehensive summary of document activity and statistics"
)
async def get_document_summary(request: Request, document_id: str):
    """Get comprehensive document summary."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = getattr(request.app.state, 'storage_service', None)
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")

        # Get all data in parallel
        results = await asyncio.gather(
            get_document_info(request, document_id),
            get_document_statistics(request, document_id),
            get_document_activity(request, document_id, limit=5),
            get_document_collaborators(request, document_id),
            return_exceptions=True
        )
        
        # Handle results
        info = results[0] if not isinstance(results[0], Exception) else None
        stats = results[1] if not isinstance(results[1], Exception) else None
        activity = results[2] if not isinstance(results[2], Exception) else None
        collaborators = results[3] if not isinstance(results[3], Exception) else None
        
        if not info or not info.exists:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")
        
        return {
            "document_id": clean_document_id,
            "overview": {
                "status": info.status,
                "message": info.message,
                "total_pages": stats.total_pages if stats else 0,
                "total_characters": stats.total_characters if stats else 0,
                "estimated_tokens": stats.estimated_tokens if stats else 0,
                "last_activity": stats.last_activity if stats else None,
                "sse_available": info.metadata.get('sse_available', False) if info.metadata else False,
                "sse_url": info.metadata.get('sse_url') if info.metadata else None
            },
            "collaboration": {
                "total_collaborators": collaborators.total_collaborators if collaborators else 0,
                "most_active_collaborators": collaborators.collaborators[:3] if collaborators else [],
                "total_annotations": stats.total_annotations if stats else 0,
                "total_chats": len(activity.recent_chats) if activity else 0
            },
            "annotation_breakdown": stats.annotation_types if stats else {},
            "recent_activity_preview": {
                "recent_annotations": len(activity.recent_annotations) if activity else 0,
                "recent_chats": len(activity.recent_chats) if activity else 0,
                "active_authors": activity.active_authors if activity else []
            },
            "metadata": info.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document summary: {str(e)}")

# ===== THUMBNAIL NAVIGATION =====

@document_router.get(
    "/documents/{document_id}/thumbnails",
    summary="Get all available thumbnails",
    description="Get list of all available thumbnail URLs for navigation"
)
async def get_document_thumbnails(
    request: Request,
    document_id: str,
    max_pages: int = Query(200, ge=1, le=500, description="Maximum pages to check")
):
    """Get all available thumbnails for document navigation."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Get document metadata to know total pages
        try:
            metadata = await storage_service.download_blob_as_json(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_metadata.json"
            )
            total_pages = metadata.get('total_pages', metadata.get('page_count', max_pages))
        except:
            total_pages = max_pages
        
        # Check which thumbnails exist
        pages_to_check = min(total_pages, max_pages)
        available_thumbnails = []
        
        # Batch check for efficiency
        batch_size = 20
        for batch_start in range(0, pages_to_check, batch_size):
            batch_end = min(batch_start + batch_size, pages_to_check)
            
            tasks = []
            for page in range(batch_start + 1, batch_end + 1):
                tasks.append(storage_service.blob_exists(
                    cache_container, 
                    f"{clean_document_id}_page_{page}_thumb.jpg"
                ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, exists in enumerate(results):
                page_num = batch_start + i + 1
                if exists and not isinstance(exists, Exception):
                    available_thumbnails.append({
                        "page": page_num,
                        "url": f"/api/v1/documents/{clean_document_id}/pages/{page_num}/image?type=thumb"
                    })
        
        return {
            "document_id": clean_document_id,
            "total_pages": total_pages,
            "thumbnails_available": len(available_thumbnails),
            "thumbnails": available_thumbnails
        }
        
    except Exception as e:
        logger.error(f"Failed to get thumbnails: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get thumbnails: {str(e)}")

# ===== CONTEXT AND METADATA =====

@document_router.get(
    "/documents/{document_id}/context",
    summary="Get document text context",
    description="Get the extracted text content for search and analysis"
)
async def get_document_context(
    request: Request,
    document_id: str,
    page: Optional[int] = Query(None, ge=1, description="Get text for specific page only")
):
    """Get document text context."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Get full context
        context_text = await storage_service.download_blob_as_text(
            container_name=cache_container,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if page:
            # Extract specific page text
            page_marker = f"--- PAGE {page} ---"
            next_page_marker = f"--- PAGE {page + 1} ---"
            
            start_idx = context_text.find(page_marker)
            if start_idx == -1:
                return {"page": page, "text": "", "found": False}
            
            end_idx = context_text.find(next_page_marker, start_idx)
            if end_idx == -1:
                page_text = context_text[start_idx:]
            else:
                page_text = context_text[start_idx:end_idx]
            
            return {"page": page, "text": page_text.strip(), "found": True}
        
        return {
            "document_id": clean_document_id,
            "total_length": len(context_text),
            "text": context_text
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document context not found")
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")

# ===== SSE SUPPORT ENDPOINTS =====

@document_router.get(
    "/documents/{document_id}/sse-status",
    summary="Check SSE availability for document",
    description="Check if SSE is available and get connection details"
)
async def check_sse_status(
    request: Request,
    document_id: str
):
    """Check SSE availability and connection status for a document."""
    clean_document_id = validate_document_id(document_id)
    
    # Check if SSE is enabled globally
    if not settings.ENABLE_SSE:
        return {
            "document_id": clean_document_id,
            "sse_enabled": False,
            "message": "SSE is not enabled on this server"
        }
    
    # Check document status
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        status_blob = f"{clean_document_id}_status.json"
        status_data = {}
        
        if await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, status_blob):
            status_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=status_blob
            )
            status_data = json.loads(status_text)
        
        # Determine if SSE is available for this document
        document_status = status_data.get('status', 'unknown')
        sse_available = document_status == 'processing' and settings.ENABLE_SSE
        
        response = {
            "document_id": clean_document_id,
            "sse_enabled": settings.ENABLE_SSE,
            "sse_available": sse_available,
            "document_status": document_status,
            "connection_config": {
                "retry_interval": settings.SSE_RETRY_INTERVAL,
                "keepalive_interval": settings.SSE_KEEPALIVE_INTERVAL,
                "max_reconnect_attempts": 10
            }
        }
        
        if sse_available:
            response["sse_url"] = f"/api/v1/documents/{clean_document_id}/status/stream"
            response["message"] = "SSE available for real-time updates"
        else:
            response["message"] = f"SSE not available - document status: {document_status}"
            response["fallback_polling_interval"] = settings.POLLING_INTERVAL if hasattr(settings, 'POLLING_INTERVAL') else 5000
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to check SSE status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check SSE status: {str(e)}")

# NOTE: Write operations (annotations, notes) remain in blueprint_routes.py