# app/api/routes/annotation_routes.py - COMPLETE ANNOTATION MANAGEMENT ROUTES

from fastapi import APIRouter, Request, HTTPException, Query, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging
import uuid
import re

from app.core.config import get_settings

# Schema imports
from app.models.schemas import (
    AnnotationCreate, AnnotationUpdate, AnnotationResponse, 
    AnnotationListResponse, BulkAnnotationRequest
)

annotation_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID - consistent with other routes"""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be a non-empty string"
        )
    
    # Remove invalid characters and strip underscores from edges
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

async def load_annotations(document_id: str, storage_service) -> List[Dict[str, Any]]:
    """Load annotations from storage"""
    try:
        annotations_blob = f"{document_id}_annotations.json"
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob
        )
        return json.loads(annotations_data)
    except Exception as e:
        logger.debug(f"No annotations found for {document_id}: {e}")
        return []

async def save_annotations(document_id: str, annotations: List[Dict[str, Any]], storage_service):
    """Save annotations to storage"""
    try:
        annotations_blob = f"{document_id}_annotations.json"
        annotations_data = json.dumps(annotations, indent=2).encode('utf-8')
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob,
            data=annotations_data,
            content_type="application/json"
        )
        logger.info(f"Saved {len(annotations)} annotations for {document_id}")
    except Exception as e:
        logger.error(f"Failed to save annotations: {e}")
        raise

def filter_annotations_for_user(annotations: List[Dict], author: str, include_published: bool = True) -> List[Dict]:
    """Filter annotations based on user and privacy settings"""
    filtered = []
    
    for ann in annotations:
        # Include user's own annotations
        if ann.get('author') == author:
            filtered.append(ann)
        # Include published (non-private) annotations if requested
        elif include_published and not ann.get('is_private', False):
            filtered.append(ann)
    
    return filtered

# --- API Routes ---

@annotation_router.get("/documents/{document_id}/annotations", response_model=AnnotationListResponse)
async def get_annotations(
    request: Request,
    document_id: str,
    author: str = Query(..., description="Current user's name"),
    page: Optional[int] = Query(None, ge=1, description="Filter by specific page"),
    annotation_type: Optional[str] = Query(None, description="Filter by annotation type"),
    include_highlights: bool = Query(True, description="Include AI highlights"),
    include_published: bool = Query(True, description="Include published annotations from others")
):
    """
    Get annotations for a document with privacy filtering.
    Returns user's own annotations plus published annotations from others.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load all annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Filter based on user and privacy
        filtered_annotations = filter_annotations_for_user(
            all_annotations, author, include_published
        )
        
        # Apply additional filters
        if page is not None:
            filtered_annotations = [
                ann for ann in filtered_annotations 
                if ann.get('page_number') == page
            ]
        
        if annotation_type:
            filtered_annotations = [
                ann for ann in filtered_annotations 
                if ann.get('annotation_type') == annotation_type
            ]
        
        # Load AI highlights if requested
        highlights = []
        if include_highlights:
            try:
                session_service = request.app.state.session_service
                if session_service:
                    session = await session_service.get_session(clean_document_id)
                    if session:
                        for hs in session.highlight_sessions.values():
                            if hs.is_active and not hs.is_expired():
                                highlight_data = {
                                    'query_session_id': hs.query_session_id,
                                    'pages': hs.pages_with_highlights,
                                    'total_highlights': hs.total_highlights,
                                    'user': hs.user,
                                    'created_at': hs.created_at.isoformat()
                                }
                                highlights.append(highlight_data)
            except Exception as e:
                logger.warning(f"Could not load highlights: {e}")
        
        return AnnotationListResponse(
            document_id=clean_document_id,
            annotations=filtered_annotations,
            highlights=highlights,
            total_count=len(filtered_annotations),
            page_filter=page,
            type_filter=annotation_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@annotation_router.post("/documents/{document_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(
    request: Request,
    document_id: str,
    annotation: AnnotationCreate
):
    """Create a new annotation on a document"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load existing annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Create new annotation
        new_annotation = {
            "annotation_id": str(uuid.uuid4())[:8],
            "document_id": clean_document_id,
            "annotation_type": annotation.annotation_type,
            "page_number": annotation.page_number,
            "author": annotation.author,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "is_private": annotation.is_private,
            "published": not annotation.is_private,  # Published if not private
            **annotation.dict(exclude={'annotation_type', 'page_number', 'author', 'is_private'})
        }
        
        # Handle specific annotation types
        if annotation.annotation_type == "pen":
            new_annotation.update({
                "points": annotation.points,
                "color": annotation.color or "#FF0000",
                "lineWidth": annotation.line_width or 2
            })
        elif annotation.annotation_type == "note":
            new_annotation.update({
                "x": annotation.x,
                "y": annotation.y,
                "text": annotation.text,
                "note_type": annotation.note_type or "general"
            })
        elif annotation.annotation_type == "highlight":
            new_annotation.update({
                "element_id": annotation.element_id,
                "grid_reference": annotation.grid_reference,
                "confidence": annotation.confidence or 1.0
            })
        
        # Add to annotations
        all_annotations.append(new_annotation)
        
        # Save back
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        # Update session if available
        session_service = request.app.state.session_service
        if session_service:
            await session_service.add_annotation(clean_document_id, new_annotation)
        
        logger.info(f"Created annotation {new_annotation['annotation_id']} for {clean_document_id}")
        
        return AnnotationResponse(**new_annotation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@annotation_router.put("/documents/{document_id}/annotations/{annotation_id}")
async def update_annotation(
    request: Request,
    document_id: str,
    annotation_id: str,
    update: AnnotationUpdate,
    author: str = Query(..., description="User making the update")
):
    """Update an existing annotation"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Find annotation
        annotation_index = None
        for i, ann in enumerate(all_annotations):
            if ann.get('annotation_id') == annotation_id:
                # Check permission
                if ann.get('author') != author:
                    raise HTTPException(
                        status_code=403, 
                        detail="You can only edit your own annotations"
                    )
                annotation_index = i
                break
        
        if annotation_index is None:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # Update annotation
        update_data = update.dict(exclude_unset=True)
        all_annotations[annotation_index].update(update_data)
        all_annotations[annotation_index]['last_modified'] = datetime.utcnow().isoformat() + "Z"
        
        # Save back
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        logger.info(f"Updated annotation {annotation_id}")
        
        return {"status": "success", "annotation_id": annotation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@annotation_router.delete("/documents/{document_id}/annotations/{annotation_id}")
async def delete_annotation(
    request: Request,
    document_id: str,
    annotation_id: str,
    author: str = Query(..., description="User requesting deletion")
):
    """Delete an annotation"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Find and remove annotation
        removed = False
        filtered_annotations = []
        
        for ann in all_annotations:
            if ann.get('annotation_id') == annotation_id:
                # Check permission
                if ann.get('author') != author:
                    raise HTTPException(
                        status_code=403, 
                        detail="You can only delete your own annotations"
                    )
                removed = True
                logger.info(f"Removing annotation {annotation_id}")
            else:
                filtered_annotations.append(ann)
        
        if not removed:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # Save updated list
        await save_annotations(clean_document_id, filtered_annotations, storage_service)
        
        return {"status": "success", "message": "Annotation deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@annotation_router.post("/documents/{document_id}/annotations/bulk")
async def create_bulk_annotations(
    request: Request,
    document_id: str,
    bulk_request: BulkAnnotationRequest
):
    """Create multiple annotations at once"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load existing annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Create new annotations
        created_ids = []
        for ann_data in bulk_request.annotations:
            new_annotation = {
                "annotation_id": str(uuid.uuid4())[:8],
                "document_id": clean_document_id,
                "annotation_type": ann_data.annotation_type,
                "page_number": ann_data.page_number,
                "author": ann_data.author,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "is_private": ann_data.is_private,
                "published": not ann_data.is_private,
                **ann_data.dict(exclude={'annotation_type', 'page_number', 'author', 'is_private'})
            }
            
            all_annotations.append(new_annotation)
            created_ids.append(new_annotation['annotation_id'])
        
        # Save all at once
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        logger.info(f"Created {len(created_ids)} annotations in bulk")
        
        return {
            "status": "success",
            "created_count": len(created_ids),
            "annotation_ids": created_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create bulk annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@annotation_router.put("/documents/{document_id}/annotations/{annotation_id}/publish")
async def publish_annotation(
    request: Request,
    document_id: str,
    annotation_id: str,
    author: str = Query(..., description="Annotation owner")
):
    """Publish a private annotation to make it visible to others"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Find and update annotation
        updated = False
        for ann in all_annotations:
            if ann.get('annotation_id') == annotation_id:
                # Check permission
                if ann.get('author') != author:
                    raise HTTPException(
                        status_code=403, 
                        detail="You can only publish your own annotations"
                    )
                
                ann['is_private'] = False
                ann['published'] = True
                ann['published_at'] = datetime.utcnow().isoformat() + "Z"
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # Save back
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        logger.info(f"Published annotation {annotation_id}")
        
        return {"status": "success", "message": "Annotation published"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@annotation_router.delete("/documents/{document_id}/annotations")
async def clear_user_annotations(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User whose annotations to clear"),
    page: Optional[int] = Query(None, description="Clear only from specific page")
):
    """Clear all annotations for a user on a document or specific page"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Filter out user's annotations
        original_count = len(all_annotations)
        filtered_annotations = []
        removed_count = 0
        
        for ann in all_annotations:
            should_remove = ann.get('author') == author
            if page is not None:
                should_remove = should_remove and ann.get('page_number') == page
            
            if should_remove:
                removed_count += 1
            else:
                filtered_annotations.append(ann)
        
        # Save if any were removed
        if removed_count > 0:
            await save_annotations(clean_document_id, filtered_annotations, storage_service)
            logger.info(f"Cleared {removed_count} annotations for {author}")
        
        return {
            "status": "success",
            "removed_count": removed_count,
            "remaining_count": len(filtered_annotations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))