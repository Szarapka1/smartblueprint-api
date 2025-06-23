# app/api/routes/document_routes.py
# (Renamed from session_routes.py to better reflect shared document functionality)

from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import json
import logging
import re
from app.core.config import get_settings

document_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class DocumentStatsResponse(BaseModel):
    document_id: str
    total_annotations: int
    total_pages: int
    total_characters: int
    estimated_tokens: int
    unique_collaborators: List[str]
    collaborator_count: int
    annotation_types: Dict[str, int]
    last_activity: Optional[str]
    status: str

class DocumentActivityResponse(BaseModel):
    document_id: str
    recent_annotations: List[dict]
    recent_chats: List[dict]
    active_authors: List[str]

# --- Helper Functions (Re-usable and robust) ---

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID for shared use."""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(status_code=400, detail="Document ID must be a non-empty string")
    
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip()).strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(status_code=400, detail="Document ID must be at least 3 characters")
    return clean_id

# FIX: Corrected function name to match what is used in blueprint_routes.py
async def load_annotations(document_id: str, storage_service) -> List[dict]:
    """Load all annotations for a document."""
    # FIX: Corrected blob name from '_all_annotations.json' to '_annotations.json'
    annotations_blob_name = f"{document_id}_annotations.json"
    try:
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob_name
        )
        return json.loads(annotations_data)
    except Exception:
        logger.info(f"No annotations file found for {document_id}")
        return []

# FIX: Corrected function to load from the correct chat log file
async def load_all_chats(document_id: str, storage_service) -> List[dict]:
    """Load all chat activity for a document."""
    # FIX: Corrected blob name from '_service_activity.json' to '_all_chats.json'
    activity_blob_name = f"{document_id}_all_chats.json"
    try:
        activity_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name
        )
        return json.loads(activity_data)
    except Exception:
        logger.info(f"No chat activity file found for {document_id}")
        return []

# --- API Routes ---

@document_router.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(request: Request, document_id: str):
    """Get comprehensive statistics for a specific document."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service

        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        # FIX: Removed dependency on non-existent ai_service.get_document_info
        # Instead, we directly check for the document's existence and get its content length.
        try:
            context_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
            doc_status = "ready"
            char_count = len(context_text)
            token_count = char_count // 4
        except Exception:
            logger.warning(f"Could not find context file for {clean_document_id}")
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        annotations = await load_annotations(clean_document_id, storage_service)
        chats = await load_all_chats(clean_document_id, storage_service)

        # Calculate collaborator and activity stats
        annotation_authors = {ann.get("author") for ann in annotations if ann.get("author")}
        chat_authors = {chat.get("author") for chat in chats if chat.get("author")}
        all_authors = list(annotation_authors.union(chat_authors))

        annotation_types = {}
        for ann in annotations:
            ann_type = ann.get("annotation_type", "note")
            annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1

        all_timestamps = [item.get("timestamp") for item in annotations + chats if item.get("timestamp")]
        last_activity = max(all_timestamps) if all_timestamps else None

        page_numbers = [ann.get("page_number", 1) for ann in annotations if isinstance(ann.get("page_number"), int)]
        max_page = max(page_numbers) if page_numbers else 1

        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=len(annotations),
            total_pages=max_page,
            total_characters=char_count,
            estimated_tokens=token_count,
            unique_collaborators=all_authors,
            collaborator_count=len(all_authors),
            annotation_types=annotation_types,
            last_activity=last_activity,
            status=doc_status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_document_statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")

@document_router.get("/documents/{document_id}/activity", response_model=DocumentActivityResponse)
async def get_document_activity(request: Request, document_id: str, limit: int = Query(10, ge=1, le=50)):
    """Get recent activity (chats and annotations) for a document."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        chats = await load_all_chats(clean_document_id, storage_service)
        annotations = await load_annotations(clean_document_id, storage_service)

        # Sort all activities by timestamp to get the most recent ones
        all_activity = sorted(chats + annotations, key=lambda x: x.get("timestamp", ""), reverse=True)
        active_authors = list(set(act.get("author") for act in all_activity[:limit] if act.get("author")))

        # Get the most recent annotations and chats separately
        recent_annotations = sorted(annotations, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        recent_chats = sorted(chats, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

        return DocumentActivityResponse(
            document_id=clean_document_id,
            recent_annotations=recent_annotations,
            recent_chats=recent_chats,
            active_authors=active_authors
        )
    except Exception as e:
        logger.error(f"Failed to get document activity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document activity: {str(e)}")

@document_router.get("/documents/{document_id}/collaborators")
async def get_document_collaborators(request: Request, document_id: str):
    """Get a list of all collaborators and their interaction counts."""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        annotations = await load_annotations(clean_document_id, storage_service)
        chats = await load_all_chats(clean_document_id, storage_service)

        annotation_authors = {ann.get("author") for ann in annotations if ann.get("author")}
        chat_authors = {chat.get("author") for chat in chats if chat.get("author")}
        all_collaborators = list(annotation_authors.union(chat_authors))

        collaborator_stats = []
        for author in all_collaborators:
            author_annotations = len([ann for ann in annotations if ann.get("author") == author])
            author_chats = len([chat for chat in chats if chat.get("author") == author])
            collaborator_stats.append({
                "author": author,
                "annotations_count": author_annotations,
                "chats_count": author_chats,
                "total_interactions": author_annotations + author_chats
            })

        collaborator_stats.sort(key=lambda x: x["total_interactions"], reverse=True)

        return {
            "document_id": clean_document_id,
            "collaborators": collaborator_stats,
            "total_collaborators": len(collaborator_stats)
        }
    except Exception as e:
        logger.error(f"Failed to get collaborators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collaborators: {str(e)}")

# NOTE: The DELETE endpoint was removed from this file to prevent conflicts
# with the one defined in `blueprint_routes.py`. It's best practice to
# define a given route in only one place. The one in `blueprint_routes.py`
# is more comprehensive.
