# app/api/routes/annotation_routes.py - COMPLETE FIXED VERSION

"""
Annotation routes for managing visual highlights and AI-generated annotations.
Handles both temporary session-based highlights and persistent annotations.
"""

import logging
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.schemas import (
    Annotation, AnnotationBase, AnnotationResponse,
    VisualHighlight, HighlightCreate, HighlightResponse
)

logger = logging.getLogger(__name__)
settings = get_settings()

annotation_router = APIRouter(
    prefix="/documents/{document_id}/annotations",
    tags=["Annotations & Highlights"]
)

# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID."""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID must be a non-empty string"
        )
    
    # Basic sanitization
    clean_id = document_id.strip()
    if len(clean_id) < 3 or len(clean_id) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID must be between 3 and 50 characters"
        )
    
    return clean_id

async def verify_document_exists(storage_service, document_id: str) -> bool:
    """Verify that a document exists and is ready."""
    try:
        # Check for context file (indicates document is processed)
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        context_exists = await storage_service.blob_exists(
            cache_container,
            f"{document_id}_context.txt"
        )
        
        if not context_exists:
            # Check status
            status_blob = f"{document_id}_status.json"
            if await storage_service.blob_exists(cache_container, status_blob):
                status_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=status_blob
                )
                status_data = json.loads(status_text)
                if status_data.get('status') == 'processing':
                    raise HTTPException(
                        status_code=status.HTTP_425_TOO_EARLY,
                        detail="Document is still processing"
                    )
                elif status_data.get('status') == 'error':
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Document processing failed"
                    )
        
        return context_exists or await storage_service.blob_exists(
            settings.AZURE_CONTAINER_NAME,
            f"{document_id}.pdf"
        )
    except HTTPException:
        raise
    except Exception:
        return False

# --- Annotation CRUD Operations ---

@annotation_router.post("/", 
                       response_model=AnnotationResponse,
                       status_code=status.HTTP_201_CREATED,
                       summary="Create a new annotation")
async def create_annotation(
    request: Request,
    document_id: str,
    annotation: AnnotationBase
):
    """Create a new annotation for a document."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    # Verify document exists
    if not await verify_document_exists(storage_service, clean_document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{clean_document_id}' not found"
        )
    
    try:
        # Generate annotation ID
        annotation_id = f"ann_{uuid.uuid4().hex[:12]}"
        
        # Create full annotation object
        full_annotation = Annotation(
            annotation_id=annotation_id,
            document_id=clean_document_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            **annotation.dict()
        )
        
        # Load existing annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations = []
        
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            annotations = json.loads(annotations_text)
        except FileNotFoundError:
            logger.info(f"Creating first annotation for document {clean_document_id}")
        except Exception as e:
            logger.warning(f"Failed to load existing annotations: {e}")
        
        # Add new annotation
        annotations.append(full_annotation.dict())
        
        # Save updated annotations
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob,
            data=json.dumps(annotations, indent=2).encode('utf-8'),
            content_type="application/json"
        )
        
        # Update session if available
        session_service = request.app.state.session_service
        if session_service:
            try:
                await session_service.add_annotation(
                    document_id=clean_document_id,
                    annotation_data=full_annotation.dict()
                )
            except Exception as e:
                logger.warning(f"Failed to update session: {e}")
        
        logger.info(f"✅ Created annotation {annotation_id} for document {clean_document_id}")
        
        return AnnotationResponse(
            annotation_id=annotation_id,
            document_id=clean_document_id,
            page_number=full_annotation.page_number,
            element_type=full_annotation.element_type,
            grid_reference=full_annotation.grid_reference,
            query_session_id=full_annotation.query_session_id,
            created_at=full_annotation.created_at,
            assigned_trade=full_annotation.assigned_trade
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create annotation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create annotation"
        )

@annotation_router.get("/",
                      summary="Get all annotations for a document")
async def get_annotations(
    request: Request,
    document_id: str,
    page: Optional[int] = Query(None, ge=1, description="Filter by page number"),
    author: Optional[str] = Query(None, description="Filter by author"),
    element_type: Optional[str] = Query(None, description="Filter by element type"),
    include_private: bool = Query(True, description="Include private annotations"),
    query_session_id: Optional[str] = Query(None, description="Filter by query session")
):
    """Get all annotations for a document with optional filtering."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    # Verify document exists
    if not await verify_document_exists(storage_service, clean_document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{clean_document_id}' not found"
        )
    
    try:
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations = []
        
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            annotations = json.loads(annotations_text)
        except FileNotFoundError:
            logger.info(f"No annotations found for document {clean_document_id}")
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
        
        # Apply filters
        filtered_annotations = []
        current_user = request.headers.get("X-User-Name", "Anonymous")
        
        for ann in annotations:
            # Filter by privacy
            if not include_private and ann.get('is_private', True):
                if ann.get('author') != current_user:
                    continue
            
            # Filter by page
            if page is not None and ann.get('page_number') != page:
                continue
            
            # Filter by author
            if author is not None and ann.get('author') != author:
                continue
            
            # Filter by element type
            if element_type is not None and ann.get('element_type') != element_type:
                continue
            
            # Filter by query session
            if query_session_id is not None and ann.get('query_session_id') != query_session_id:
                continue
            
            filtered_annotations.append(ann)
        
        # Sort by creation time (newest first)
        filtered_annotations.sort(
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "annotations": filtered_annotations,
                "total": len(filtered_annotations),
                "filters_applied": {
                    "page": page,
                    "author": author,
                    "element_type": element_type,
                    "include_private": include_private,
                    "query_session_id": query_session_id
                }
            },
            headers={"Cache-Control": "no-cache, must-revalidate"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get annotations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve annotations"
        )

@annotation_router.get("/{annotation_id}",
                      summary="Get a specific annotation")
async def get_annotation(
    request: Request,
    document_id: str,
    annotation_id: str
):
    """Get a specific annotation by ID."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob
        )
        annotations = json.loads(annotations_text)
        
        # Find annotation
        for ann in annotations:
            if ann.get('annotation_id') == annotation_id:
                # Check privacy
                current_user = request.headers.get("X-User-Name", "Anonymous")
                if ann.get('is_private', True) and ann.get('author') != current_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to private annotation"
                    )
                
                return JSONResponse(
                    content=ann,
                    headers={"Cache-Control": "public, max-age=300"}
                )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation '{annotation_id}' not found"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No annotations found for this document"
        )
    except Exception as e:
        logger.error(f"Failed to get annotation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve annotation"
        )

@annotation_router.delete("/{annotation_id}",
                         status_code=status.HTTP_204_NO_CONTENT,
                         summary="Delete an annotation")
async def delete_annotation(
    request: Request,
    document_id: str,
    annotation_id: str
):
    """Delete an annotation (only by its author)."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob
        )
        annotations = json.loads(annotations_text)
        
        # Find and verify ownership
        current_user = request.headers.get("X-User-Name", "Anonymous")
        updated_annotations = []
        found = False
        
        for ann in annotations:
            if ann.get('annotation_id') == annotation_id:
                found = True
                if ann.get('author') != current_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Only the author can delete an annotation"
                    )
                logger.info(f"Deleting annotation {annotation_id}")
            else:
                updated_annotations.append(ann)
        
        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Annotation '{annotation_id}' not found"
            )
        
        # Save updated annotations
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob,
            data=json.dumps(updated_annotations, indent=2).encode('utf-8'),
            content_type="application/json"
        )
        
        logger.info(f"✅ Deleted annotation {annotation_id}")
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No annotations found for this document"
        )
    except Exception as e:
        logger.error(f"Failed to delete annotation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete annotation"
        )

# --- Highlight Session Management ---

@annotation_router.get("/highlights/sessions",
                      summary="Get active highlight sessions")
async def get_highlight_sessions(
    request: Request,
    document_id: str,
    active_only: bool = Query(True, description="Only show active sessions")
):
    """Get highlight sessions for a document."""
    clean_document_id = validate_document_id(document_id)
    
    session_service = request.app.state.session_service
    if not session_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session service unavailable"
        )
    
    try:
        # Get document session
        doc_session = await session_service.get_session(clean_document_id)
        if not doc_session:
            return JSONResponse(
                content={
                    "document_id": clean_document_id,
                    "sessions": [],
                    "total": 0
                }
            )
        
        # Get highlight sessions
        sessions_data = []
        for session_id, highlight_session in doc_session.highlight_sessions.items():
            if active_only and highlight_session.is_expired():
                continue
            
            sessions_data.append({
                "query_session_id": session_id,
                "user": highlight_session.user,
                "query": highlight_session.query,
                "created_at": highlight_session.created_at.isoformat(),
                "expires_at": highlight_session.expires_at.isoformat(),
                "is_active": highlight_session.is_active and not highlight_session.is_expired(),
                "total_highlights": highlight_session.total_highlights,
                "pages_with_highlights": dict(highlight_session.pages_with_highlights),
                "element_types": list(highlight_session.element_types)
            })
        
        # Sort by creation time (newest first)
        sessions_data.sort(key=lambda x: x['created_at'], reverse=True)
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "sessions": sessions_data,
                "total": len(sessions_data)
            },
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Failed to get highlight sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve highlight sessions"
        )

@annotation_router.get("/highlights/{query_session_id}",
                      summary="Get highlights for a specific query session")
async def get_session_highlights(
    request: Request,
    document_id: str,
    query_session_id: str
):
    """Get all highlights for a specific query session."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    session_service = request.app.state.session_service
    
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        # Check session service first for active session
        if session_service:
            doc_session = await session_service.get_session(clean_document_id)
            if doc_session and query_session_id in doc_session.highlight_sessions:
                highlight_session = doc_session.highlight_sessions[query_session_id]
                
                # Return session info
                return JSONResponse(
                    content={
                        "document_id": clean_document_id,
                        "query_session_id": query_session_id,
                        "user": highlight_session.user,
                        "query": highlight_session.query,
                        "created_at": highlight_session.created_at.isoformat(),
                        "expires_at": highlight_session.expires_at.isoformat(),
                        "is_active": highlight_session.is_active and not highlight_session.is_expired(),
                        "total_highlights": highlight_session.total_highlights,
                        "pages_with_highlights": dict(highlight_session.pages_with_highlights),
                        "element_types": list(highlight_session.element_types)
                    },
                    headers={"Cache-Control": "no-cache"}
                )
        
        # Otherwise look in stored annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        highlights = []
        
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            annotations = json.loads(annotations_text)
            
            # Filter by query session
            for ann in annotations:
                if ann.get('query_session_id') == query_session_id:
                    highlights.append(ann)
            
        except FileNotFoundError:
            pass
        
        if not highlights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No highlights found for session '{query_session_id}'"
            )
        
        # Group by page
        pages_with_highlights = {}
        element_types = set()
        
        for h in highlights:
            page = h.get('page_number', 0)
            pages_with_highlights[page] = pages_with_highlights.get(page, 0) + 1
            if h.get('element_type'):
                element_types.add(h['element_type'])
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "query_session_id": query_session_id,
                "highlights": highlights,
                "total_highlights": len(highlights),
                "pages_with_highlights": pages_with_highlights,
                "element_types": list(element_types)
            },
            headers={"Cache-Control": "public, max-age=300"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session highlights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve highlights"
        )

# --- Visual Highlight Creation ---

@annotation_router.post("/highlights",
                       response_model=HighlightResponse,
                       status_code=status.HTTP_201_CREATED,
                       summary="Create a visual highlight")
async def create_highlight(
    request: Request,
    document_id: str,
    highlight: HighlightCreate
):
    """Create a visual highlight (rectangle, circle, etc.)."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    # Verify document exists
    if not await verify_document_exists(storage_service, clean_document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{clean_document_id}' not found"
        )
    
    try:
        # Create visual highlight
        visual_highlight = VisualHighlight(
            id=str(uuid.uuid4()),
            type=highlight.type.value,
            coordinates=highlight.coordinates,
            page=highlight.page,
            label=highlight.label,
            color=highlight.color,
            created_at=datetime.utcnow(),
            created_by=request.headers.get("X-User-Name", "Anonymous")
        )
        
        # If note content provided, create annotation too
        if highlight.note_content:
            annotation = AnnotationBase(
                page_number=highlight.page,
                element_type="visual_highlight",
                grid_reference=f"HL-{highlight.page}",
                x=int(highlight.coordinates[0]) if highlight.coordinates else 0,
                y=int(highlight.coordinates[1]) if highlight.coordinates else 0,
                width=100,
                height=100,
                text=highlight.note_content,
                author=visual_highlight.created_by,
                annotation_type="user_annotation"
            )
            
            # Create annotation
            ann_response = await create_annotation(
                request, clean_document_id, annotation
            )
            
            visual_highlight.annotation_id = ann_response.annotation_id
        
        # Store visual highlight
        highlights_blob = f"{clean_document_id}_visual_highlights.json"
        highlights = []
        
        try:
            highlights_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=highlights_blob
            )
            highlights = json.loads(highlights_text)
        except FileNotFoundError:
            pass
        
        highlights.append(visual_highlight.dict())
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=highlights_blob,
            data=json.dumps(highlights, indent=2, default=str).encode('utf-8'),
            content_type="application/json"
        )
        
        logger.info(f"✅ Created visual highlight {visual_highlight.id}")
        
        return HighlightResponse(
            highlight=visual_highlight,
            status="created",
            message="Highlight created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create highlight: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create highlight"
        )

# --- Bulk Operations ---

@annotation_router.post("/bulk-delete",
                       summary="Delete multiple annotations")
async def bulk_delete_annotations(
    request: Request,
    document_id: str,
    annotation_ids: List[str] = Body(..., description="List of annotation IDs to delete")
):
    """Delete multiple annotations at once (only by their author)."""
    clean_document_id = validate_document_id(document_id)
    
    if not annotation_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No annotation IDs provided"
        )
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob
        )
        annotations = json.loads(annotations_text)
        
        # Process deletions
        current_user = request.headers.get("X-User-Name", "Anonymous")
        updated_annotations = []
        deleted_count = 0
        unauthorized_count = 0
        
        for ann in annotations:
            if ann.get('annotation_id') in annotation_ids:
                if ann.get('author') == current_user:
                    deleted_count += 1
                    logger.info(f"Deleting annotation {ann.get('annotation_id')}")
                else:
                    unauthorized_count += 1
                    updated_annotations.append(ann)
            else:
                updated_annotations.append(ann)
        
        if deleted_count > 0:
            # Save updated annotations
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob,
                data=json.dumps(updated_annotations, indent=2).encode('utf-8'),
                content_type="application/json"
            )
        
        return JSONResponse(
            content={
                "deleted": deleted_count,
                "unauthorized": unauthorized_count,
                "total_requested": len(annotation_ids)
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No annotations found for this document"
        )
    except Exception as e:
        logger.error(f"Failed to bulk delete annotations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete annotations"
        )

# --- Analytics ---

@annotation_router.get("/stats",
                      summary="Get annotation statistics")
async def get_annotation_stats(
    request: Request,
    document_id: str
):
    """Get statistics about annotations for a document."""
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations = []
        
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            annotations = json.loads(annotations_text)
        except FileNotFoundError:
            pass
        
        # Calculate statistics
        stats = {
            "total_annotations": len(annotations),
            "by_author": {},
            "by_page": {},
            "by_element_type": {},
            "by_annotation_type": {},
            "private_count": 0,
            "public_count": 0,
            "ai_generated_count": 0,
            "with_query_session": 0
        }
        
        for ann in annotations:
            # By author
            author = ann.get('author', 'Unknown')
            stats['by_author'][author] = stats['by_author'].get(author, 0) + 1
            
            # By page
            page = str(ann.get('page_number', 0))
            stats['by_page'][page] = stats['by_page'].get(page, 0) + 1
            
            # By element type
            elem_type = ann.get('element_type', 'unknown')
            stats['by_element_type'][elem_type] = stats['by_element_type'].get(elem_type, 0) + 1
            
            # By annotation type
            ann_type = ann.get('annotation_type', 'note')
            stats['by_annotation_type'][ann_type] = stats['by_annotation_type'].get(ann_type, 0) + 1
            
            # Privacy
            if ann.get('is_private', True):
                stats['private_count'] += 1
            else:
                stats['public_count'] += 1
            
            # AI generated
            if ann.get('annotation_type') == 'ai_highlight':
                stats['ai_generated_count'] += 1
            
            # With query session
            if ann.get('query_session_id'):
                stats['with_query_session'] += 1
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "statistics": stats
            },
            headers={"Cache-Control": "public, max-age=300"}
        )
        
    except Exception as e:
        logger.error(f"Failed to get annotation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )

# --- Export/Import ---

@annotation_router.get("/export",
                      summary="Export all annotations")
async def export_annotations(
    request: Request,
    document_id: str,
    format: str = Query("json", description="Export format (json, csv)")
):
    """Export all annotations for a document."""
    clean_document_id = validate_document_id(document_id)
    
    if format not in ["json", "csv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'json' or 'csv'"
        )
    
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable"
        )
    
    try:
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        annotations = []
        
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            annotations = json.loads(annotations_text)
        except FileNotFoundError:
            pass
        
        if format == "json":
            # Return as JSON
            return JSONResponse(
                content={
                    "document_id": clean_document_id,
                    "export_date": datetime.utcnow().isoformat() + "Z",
                    "annotations": annotations
                },
                headers={
                    "Content-Disposition": f"attachment; filename=\"{clean_document_id}_annotations.json\""
                }
            )
        else:
            # Convert to CSV
            import csv
            import io
            
            output = io.StringIO()
            if annotations:
                fieldnames = list(annotations[0].keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(annotations)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=\"{clean_document_id}_annotations.csv\""
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to export annotations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export annotations"
        )