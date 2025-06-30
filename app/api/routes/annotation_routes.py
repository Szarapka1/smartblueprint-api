# app/api/routes/annotation_routes.py - FIXED MULTI-USER HIGHLIGHT STORAGE

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

@annotation_router = APIRouter()
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
                expires = datetime.fromisoformat(ann["expires_at"].replace('Z', ''))
                if expires < current_time:
                    logger.debug(f"Removing expired highlight: {ann.get('annotation_id')}")
                    continue
            except:
                pass
        active_annotations.append(ann)
    
    return active_annotations

async def get_active_highlights_for_session(document_id: str, query_session_id: str, 
                                          author: str, storage_service) -> List[dict]:
    """Get all highlights for a specific query session and user"""
    all_annotations = await load_all_annotations(document_id, storage_service)
    
    # Filter to just this query session AND this author
    session_highlights = [
        ann for ann in all_annotations 
        if ann.get("query_session_id") == query_session_id
        and ann.get("author") == author
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
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Validate that all highlights have the same author
        if not highlights:
            raise HTTPException(status_code=400, detail="No highlights provided")
            
        authors = set(h.author for h in highlights)
        if len(authors) > 1:
            raise HTTPException(status_code=400, detail="All highlights must have the same author")
        
        author = highlights[0].author
        
        # Generate query session ID if not provided
        query_session_id = highlights[0].query_session_id if highlights[0].query_session_id else str(uuid.uuid4())
        
        # Ensure all highlights have the same session ID
        for h in highlights:
            h.query_session_id = query_session_id
        
        # Load existing annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Clear expired highlights first
        all_annotations = await clear_expired_highlights(all_annotations)
        
        # FIXED: Only remove THIS USER's old AI highlights for THIS DOCUMENT
        # Keep all other users' highlights and all non-AI annotations
        filtered_annotations = []
        removed_count = 0
        
        for ann in all_annotations:
            # Remove only if ALL conditions are met:
            # 1. It's an AI highlight
            # 2. It belongs to the same author
            # 3. It's for the same document
            # 4. It's NOT from the current query session
            if (ann.get("annotation_type") == "ai_highlight" and 
                ann.get("author") == author and
                ann.get("document_id") == clean_document_id and
                ann.get("query_session_id") != query_session_id):
                removed_count += 1
                logger.debug(f"Removing old highlight from {author}: {ann.get('annotation_id')}")
            else:
                filtered_annotations.append(ann)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old highlights for user {author}")
        
        # Add new highlights with proper expiration
        created_highlights = []
        expiry_time = datetime.utcnow() + timedelta(hours=24)
        
        for highlight in highlights:
            new_annotation = highlight.dict()
            new_annotation["annotation_id"] = str(uuid.uuid4())[:8]
            new_annotation["created_at"] = datetime.utcnow().isoformat()
            new_annotation["document_id"] = clean_document_id
            new_annotation["query_session_id"] = query_session_id
            new_annotation["expires_at"] = expiry_time.isoformat()
            
            # Ensure author is set
            if "author" not in new_annotation or not new_annotation["author"]:
                new_annotation["author"] = author
            
            filtered_annotations.append(new_annotation)
            created_highlights.append(new_annotation)
        
        # Save all annotations
        await save_all_annotations(clean_document_id, filtered_annotations, storage_service)
        
        # Log the activity
        logger.info(f"Created {len(highlights)} highlights for document {clean_document_id}, session {query_session_id}, user {author}")
        
        # Update session service if available
        session_service = request.app.state.session_service
        if session_service:
            try:
                # Calculate pages with highlights
                pages_with_highlights = {}
                element_types = set()
                
                for h in created_highlights:
                    page = h.get("page_number", 0)
                    pages_with_highlights[page] = pages_with_highlights.get(page, 0) + 1
                    element_types.add(h.get("element_type", "unknown"))
                
                await session_service.update_highlight_session(
                    document_id=clean_document_id,
                    query_session_id=query_session_id,
                    pages_with_highlights=pages_with_highlights,
                    element_types=list(element_types),
                    total_highlights=len(created_highlights)
                )
            except Exception as e:
                logger.warning(f"Failed to update session service: {e}")
        
        return {
            "status": "success",
            "query_session_id": query_session_id,
            "highlights_created": len(created_highlights),
            "pages_affected": list(set(h.page_number for h in highlights)),
            "author": author
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create highlight batch: {e}", exc_info=True)
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
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Get all highlights for this session AND user
        session_highlights = await get_active_highlights_for_session(
            clean_document_id, query_session_id, author, storage_service
        )
        
        # No need for additional filtering - get_active_highlights_for_session already filters by author
        user_highlights = session_highlights
        
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active highlights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get highlights: {str(e)}")

@annotation_router.delete("/documents/{document_id}/highlights/clear")
async def clear_highlight_session(
    request: Request,
    document_id: str,
    query_session_id: str = Query(..., description="Query session to clear"),
    author: str = Query(..., description="User clearing highlights")
):
    """Clear all highlights from a specific query session for a specific user"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load all annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Remove highlights from this session AND this author only
        initial_count = len(all_annotations)
        filtered_annotations = []
        removed_count = 0
        
        for ann in all_annotations:
            # Only remove if it matches BOTH session and author
            if (ann.get("query_session_id") == query_session_id and
                ann.get("author") == author):
                removed_count += 1
                logger.debug(f"Removing highlight: {ann.get('annotation_id')}")
            else:
                filtered_annotations.append(ann)
        
        # Save updated annotations
        await save_all_annotations(clean_document_id, filtered_annotations, storage_service)
        
        logger.info(f"Cleared {removed_count} highlights from session {query_session_id} for user {author}")
        
        return {
            "status": "success",
            "query_session_id": query_session_id,
            "author": author,
            "highlights_removed": removed_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear highlight session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear highlights: {str(e)}")

# --- EXISTING ENDPOINTS (Updated for proper multi-user support) ---

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
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
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
            elif (annotation.get("is_private", True) and 
                  annotation.get("author") == author and
                  annotation.get("annotation_type") != "ai_highlight"):
                visible_annotations.append(annotation)
        
        # Group by type for summary
        summary = {
            "ai_highlights": 0,
            "private_notes": 0,
            "published_notes": 0
        }
        
        for ann in visible_annotations:
            if ann.get("annotation_type") == "ai_highlight":
                summary["ai_highlights"] += 1
            elif ann.get("is_private", True):
                summary["private_notes"] += 1
            else:
                summary["published_notes"] += 1
        
        return {
            "document_id": clean_document_id,
            "annotations": visible_annotations,
            "visible_count": len(visible_annotations),
            "requesting_author": author,
            "summary": summary,
            "filters_applied": {
                "include_highlights": include_highlights,
                "include_published": include_published
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch annotations: {e}", exc_info=True)
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
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
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
        
        # Clear expired highlights
        all_annotations = await clear_expired_highlights(all_annotations)
        
        # Create new annotation
        new_annotation = annotation.dict()
        new_annotation["annotation_id"] = str(uuid.uuid4())[:8]
        new_annotation["document_id"] = clean_document_id
        new_annotation["created_at"] = datetime.utcnow().isoformat()
        
        # If it's an AI highlight and no expiry set, expire after 24 hours
        if annotation.annotation_type == "ai_highlight" and not annotation.expires_at:
            expiry = datetime.utcnow() + timedelta(hours=24)
            new_annotation["expires_at"] = expiry.isoformat()
        
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
        logger.error(f"Failed to create annotation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@annotation_router.get("/documents/{document_id}/highlights/summary")
async def get_highlights_summary(
    request: Request,
    document_id: str,
    author: Optional[str] = Query(None, description="Filter by specific author"),
    include_expired: bool = Query(False, description="Include expired highlights")
):
    """Get summary of highlights for a document - optionally filtered by author"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Get ALL annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Filter to highlights
        if not include_expired:
            all_annotations = await clear_expired_highlights(all_annotations)
        
        highlights = [
            ann for ann in all_annotations 
            if ann.get("annotation_type") == "ai_highlight"
        ]
        
        # Filter by author if specified
        if author:
            highlights = [h for h in highlights if h.get("author") == author]
        
        # Group by session and author
        sessions_summary = {}
        authors_summary = {}
        
        for highlight in highlights:
            session_id = highlight.get("query_session_id", "unknown")
            highlight_author = highlight.get("author", "unknown")
            
            # Session summary
            if session_id not in sessions_summary:
                sessions_summary[session_id] = {
                    "created_at": highlight.get("created_at"),
                    "expires_at": highlight.get("expires_at"),
                    "author": highlight_author,
                    "pages": set(),
                    "element_types": set(),
                    "total": 0
                }
            
            sessions_summary[session_id]["pages"].add(highlight.get("page_number"))
            sessions_summary[session_id]["element_types"].add(highlight.get("element_type"))
            sessions_summary[session_id]["total"] += 1
            
            # Author summary
            if highlight_author not in authors_summary:
                authors_summary[highlight_author] = {
                    "total_highlights": 0,
                    "active_sessions": set(),
                    "pages": set()
                }
            
            authors_summary[highlight_author]["total_highlights"] += 1
            authors_summary[highlight_author]["active_sessions"].add(session_id)
            authors_summary[highlight_author]["pages"].add(highlight.get("page_number"))
        
        # Convert sets to lists
        for session in sessions_summary.values():
            session["pages"] = sorted(list(session["pages"]))
            session["element_types"] = sorted(list(session["element_types"]))
        
        for author_data in authors_summary.values():
            author_data["active_sessions"] = len(author_data["active_sessions"])
            author_data["pages"] = sorted(list(author_data["pages"]))
        
        return {
            "document_id": clean_document_id,
            "total_highlights": len(highlights),
            "active_sessions": len(sessions_summary),
            "active_authors": len(authors_summary),
            "filter_applied": {"author": author} if author else None,
            "sessions": sessions_summary,
            "authors": authors_summary
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get highlights summary: {e}", exc_info=True)
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
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
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
        author_breakdown = {}
        
        for highlight in all_highlights:
            session_id = highlight.get("query_session_id", "unknown")
            if session_id not in sessions:
                sessions[session_id] = {
                    "highlights": [],
                    "pages": set(),
                    "element_types": set(),
                    "trades": set(),
                    "created_at": highlight.get("created_at"),
                    "author": highlight.get("author", "ai_system"),
                    "expires_at": highlight.get("expires_at")
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
            
            # Track authors
            author = highlight.get("author", "unknown")
            if author not in author_breakdown:
                author_breakdown[author] = 0
            author_breakdown[author] += 1
        
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
            "unique_authors": len(author_breakdown),
            "trade_breakdown": trade_breakdown,
            "author_breakdown": author_breakdown,
            "sessions": sessions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin access failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@annotation_router.delete("/admin/documents/{document_id}/cleanup-expired")
async def admin_cleanup_expired_highlights(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    ADMIN: Force cleanup of expired highlights
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        # Load all annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        initial_count = len(all_annotations)
        
        # Clear expired highlights
        active_annotations = await clear_expired_highlights(all_annotations)
        removed_count = initial_count - len(active_annotations)
        
        # Save cleaned annotations
        await save_all_annotations(clean_document_id, active_annotations, storage_service)
        
        logger.info(f"Admin cleanup: removed {removed_count} expired highlights from {clean_document_id}")
        
        return {
            "status": "success",
            "document_id": clean_document_id,
            "initial_count": initial_count,
            "removed_count": removed_count,
            "remaining_count": len(active_annotations)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin cleanup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Admin cleanup failed: {str(e)}")
