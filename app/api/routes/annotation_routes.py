# app/api/routes/annotation_routes.py - MULTI-PAGE HIGHLIGHT STORAGE

from fastapi import APIRouter, Request, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import json
import uuid
import os
import re
import logging
from app.core.config import get_settings
from app.models.schemas import Annotation, AnnotationResponse

annotation_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate document ID"""
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip()).strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters"
        )
    
    return clean_id

def validate_admin_access(admin_token: str) -> bool:
    """Validate admin access using environment variable"""
    expected_token = os.getenv("ADMIN_SECRET_TOKEN", settings.ADMIN_SECRET_TOKEN)
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Admin access not configured"
        )
    return admin_token == expected_token

async def load_all_annotations(document_id: str, storage_service) -> List[dict]:
    """Load ALL annotations including AI highlights"""
    annotations_blob_name = f"{document_id}_annotations.json"
    
    try:
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob_name
        )
        return json.loads(annotations_data)
    except Exception:
        return []

async def save_all_annotations(document_id: str, annotations: List[dict], storage_service):
    """Save ALL annotations including AI highlights"""
    annotations_blob_name = f"{document_id}_annotations.json"
    annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
    
    await storage_service.upload_file(
        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
        blob_name=annotations_blob_name,
        data=annotations_json.encode('utf-8')
    )

async def clear_expired_highlights(annotations: List[dict]) -> List[dict]:
    """Remove expired AI highlights"""
    current_time = datetime.utcnow()
    active_annotations = []
    
    for ann in annotations:
        # Skip if it has an expiry time that's passed
        if ann.get("expires_at"):
            try:
                expires = datetime.fromisoformat(ann["expires_at"].replace('Z', '+00:00'))
                if expires < current_time:
                    continue
            except:
                pass
        active_annotations.append(ann)
    
    return active_annotations

async def get_active_highlights_for_session(document_id: str, query_session_id: str, 
                                          storage_service) -> List[dict]:
    """Get all highlights for a specific query session"""
    all_annotations = await load_all_annotations(document_id, storage_service)
    
    # Filter to just this query session
    session_highlights = [
        ann for ann in all_annotations 
        if ann.get("query_session_id") == query_session_id
    ]
    
    return session_highlights

# --- NEW: Multi-Page Highlight Management ---

@annotation_router.post("/documents/{document_id}/highlights/create-batch")
async def create_highlight_batch(
    request: Request,
    document_id: str,
    highlights: List[Annotation]
):
    """Create multiple highlights across pages from AI analysis"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Generate query session ID if not provided
        query_session_id = highlights[0].query_session_id if highlights else str(uuid.uuid4())
        
        # Load existing annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Clear old highlights from previous queries by this user
        # Keep only non-AI highlights and highlights from other query sessions
        filtered_annotations = [
            ann for ann in all_annotations
            if ann.get("annotation_type") != "ai_highlight" or 
               ann.get("query_session_id") != query_session_id
        ]
        
        # Add new highlights
        created_highlights = []
        for highlight in highlights:
            new_annotation = highlight.dict()
            new_annotation["annotation_id"] = str(uuid.uuid4())[:8]
            new_annotation["created_at"] = datetime.utcnow().isoformat() + "Z"
            new_annotation["document_id"] = clean_document_id
            new_annotation["query_session_id"] = query_session_id
            
            filtered_annotations.append(new_annotation)
            created_highlights.append(new_annotation)
        
        # Save all annotations
        await save_all_annotations(clean_document_id, filtered_annotations, storage_service)
        
        # Log the activity
        logger.info(f"Created {len(highlights)} highlights for document {clean_document_id}, session {query_session_id}")
        
        return {
            "status": "success",
            "query_session_id": query_session_id,
            "highlights_created": len(created_highlights),
            "pages_affected": list(set(h.page_number for h in highlights))
        }
    
    except Exception as e:
        logger.error(f"Failed to create highlight batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create highlights: {str(e)}")

@annotation_router.get("/documents/{document_id}/highlights/active")
async def get_active_highlights(
    request: Request,
    document_id: str,
    query_session_id: str = Query(..., description="Query session to get highlights for"),
    author: str = Query(..., description="User requesting highlights"),
    page_number: Optional[int] = Query(None, description="Filter by specific page")
):
    """Get active highlights for current query session - only visible to the user who created them"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Get all highlights for this session
        session_highlights = await get_active_highlights_for_session(
            clean_document_id, query_session_id, storage_service
        )
        
        # Filter to only highlights created by this user
        user_highlights = [
            h for h in session_highlights
            if h.get("author") == author or h.get("author") == "ai_system"
        ]
        
        # Filter by page if requested
        if page_number is not None:
            user_highlights = [
                h for h in user_highlights 
                if h.get("page_number") == page_number
            ]
        
        # Group by page for summary
        highlights_by_page = {}
        
        for highlight in user_highlights:
            page = highlight.get("page_number", 0)
            if page not in highlights_by_page:
                highlights_by_page[page] = []
            highlights_by_page[page].append(highlight)
        
        return {
            "document_id": clean_document_id,
            "query_session_id": query_session_id,
            "author": author,
            "total_highlights": len(user_highlights),
            "highlights": user_highlights,
            "pages_with_highlights": sorted(highlights_by_page.keys()),
            "highlights_by_page": {
                page: len(items) for page, items in highlights_by_page.items()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get active highlights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get highlights: {str(e)}")

@annotation_router.delete("/documents/{document_id}/highlights/clear")
async def clear_highlight_session(
    request: Request,
    document_id: str,
    query_session_id: str = Query(..., description="Query session to clear")
):
    """Clear all highlights from a specific query session"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Load all annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Remove highlights from this session
        initial_count = len(all_annotations)
        filtered_annotations = [
            ann for ann in all_annotations
            if ann.get("query_session_id") != query_session_id
        ]
        removed_count = initial_count - len(filtered_annotations)
        
        # Save updated annotations
        await save_all_annotations(clean_document_id, filtered_annotations, storage_service)
        
        logger.info(f"Cleared {removed_count} highlights from session {query_session_id}")
        
        return {
            "status": "success",
            "query_session_id": query_session_id,
            "highlights_removed": removed_count
        }
    
    except Exception as e:
        logger.error(f"Failed to clear highlight session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear highlights: {str(e)}")

# --- EXISTING ENDPOINTS (Updated for open access) ---

@annotation_router.get("/documents/{document_id}/annotations")
async def get_user_visible_annotations(
    request: Request, 
    document_id: str,
    author: str = Query(..., description="User requesting annotations"),
    include_highlights: bool = Query(True, description="Include your AI highlights"),
    include_published: bool = Query(True, description="Include published notes from all users")
):
    """
    Get annotations visible to user:
    - Their own AI highlights (private)
    - Their own personal notes (private)
    - Published notes from all users (public)
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Load ALL annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Clear expired highlights
        all_annotations = await clear_expired_highlights(all_annotations)
        
        # Filter annotations based on privacy rules
        visible_annotations = []
        for annotation in all_annotations:
            # User's own AI highlights (private to them)
            if (annotation.get("annotation_type") == "ai_highlight" and 
                include_highlights and 
                annotation.get("author") == author):
                visible_annotations.append(annotation)
            # Published notes (visible to everyone)
            elif not annotation.get("is_private", True) and include_published:
                visible_annotations.append(annotation)
            # User's own private notes
            elif annotation.get("is_private", True) and annotation.get("author") == author:
                visible_annotations.append(annotation)
        
        return {
            "document_id": clean_document_id,
            "annotations": visible_annotations,
            "visible_count": len(visible_annotations),
            "requesting_author": author,
            "filters_applied": {
                "include_highlights": include_highlights,
                "include_published": include_published
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch annotations: {str(e)}")

@annotation_router.post("/documents/{document_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(
    request: Request,
    document_id: str,
    annotation: Annotation
):
    """
    Create annotation (can be user annotation OR AI highlight)
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Verify document exists
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
        except:
            raise HTTPException(
                status_code=404, 
                detail=f"Document '{clean_document_id}' not found"
            )
        
        # Load existing annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Create new annotation
        new_annotation = annotation.dict()
        new_annotation["annotation_id"] = str(uuid.uuid4())[:8]
        new_annotation["document_id"] = clean_document_id
        new_annotation["created_at"] = datetime.utcnow().isoformat() + "Z"
        
        # If it's an AI highlight and no expiry set, expire after 24 hours
        if annotation.annotation_type == "ai_highlight" and not annotation.expires_at:
            expiry = datetime.utcnow() + timedelta(hours=24)
            new_annotation["expires_at"] = expiry.isoformat() + "Z"
        
        # Add to annotations
        all_annotations.append(new_annotation)
        await save_all_annotations(clean_document_id, all_annotations, storage_service)
        
        return AnnotationResponse(
            annotation_id=new_annotation["annotation_id"],
            document_id=clean_document_id,
            page_number=annotation.page_number,
            element_type=annotation.element_type,
            grid_reference=annotation.grid_reference,
            query_session_id=annotation.query_session_id,
            created_at=new_annotation["created_at"],
            assigned_trade=new_annotation.get("assigned_trade")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@annotation_router.get("/documents/{document_id}/highlights/summary")
async def get_highlights_summary(
    request: Request,
    document_id: str,
    include_expired: bool = Query(False, description="Include expired highlights")
):
    """Get summary of all highlights for a document - accessible to all users"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Get ALL annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Filter to highlights
        if not include_expired:
            all_annotations = await clear_expired_highlights(all_annotations)
        
        highlights = [
            ann for ann in all_annotations 
            if ann.get("annotation_type") == "ai_highlight"
        ]
        
        # Group by session
        sessions_summary = {}
        for highlight in highlights:
            session_id = highlight.get("query_session_id", "unknown")
            if session_id not in sessions_summary:
                sessions_summary[session_id] = {
                    "created_at": highlight.get("created_at"),
                    "expires_at": highlight.get("expires_at"),
                    "pages": set(),
                    "element_types": set(),
                    "total": 0
                }
            
            sessions_summary[session_id]["pages"].add(highlight.get("page_number"))
            sessions_summary[session_id]["element_types"].add(highlight.get("element_type"))
            sessions_summary[session_id]["total"] += 1
        
        # Convert sets to lists
        for session in sessions_summary.values():
            session["pages"] = sorted(list(session["pages"]))
            session["element_types"] = sorted(list(session["element_types"]))
        
        return {
            "document_id": clean_document_id,
            "total_highlights": len(highlights),
            "active_sessions": len(sessions_summary),
            "sessions": sessions_summary
        }
    
    except Exception as e:
        logger.error(f"Failed to get highlights summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ADMIN ENDPOINTS ---

@annotation_router.get("/admin/documents/{document_id}/all-highlights")
async def admin_get_all_highlights(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    ADMIN: Get ALL highlights across all sessions for analytics
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Get ALL annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Filter to just AI highlights
        all_highlights = [
            ann for ann in all_annotations 
            if ann.get("annotation_type") == "ai_highlight"
        ]
        
        # Group by query session with more details
        sessions = {}
        trade_breakdown = {}
        
        for highlight in all_highlights:
            session_id = highlight.get("query_session_id", "unknown")
            if session_id not in sessions:
                sessions[session_id] = {
                    "highlights": [],
                    "pages": set(),
                    "element_types": set(),
                    "trades": set(),
                    "created_at": highlight.get("created_at"),
                    "author": highlight.get("author", "ai_system")
                }
            sessions[session_id]["highlights"].append(highlight)
            sessions[session_id]["pages"].add(highlight.get("page_number"))
            sessions[session_id]["element_types"].add(highlight.get("element_type"))
            
            # Track trades
            trade = highlight.get("assigned_trade", "Unassigned")
            sessions[session_id]["trades"].add(trade)
            
            if trade not in trade_breakdown:
                trade_breakdown[trade] = 0
            trade_breakdown[trade] += 1
        
        # Convert sets to lists for JSON
        for session in sessions.values():
            session["pages"] = sorted(list(session["pages"]))
            session["element_types"] = sorted(list(session["element_types"]))
            session["trades"] = sorted(list(session["trades"]))
            session["highlight_count"] = len(session["highlights"])
        
        return {
            "document_id": clean_document_id,
            "total_highlights": len(all_highlights),
            "total_sessions": len(sessions),
            "trade_breakdown": trade_breakdown,
            "sessions": sessions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")
