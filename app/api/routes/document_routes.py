# app/api/routes/document_routes.py - COMPLETE REWRITTEN VERSION
# Provides document statistics, activity tracking, and collaboration analytics

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

class CollaboratorStats(BaseModel):
    author: str
    annotations_count: int
    chats_count: int
    total_interactions: int

class CollaboratorsResponse(BaseModel):
    document_id: str
    collaborators: List[CollaboratorStats]
    total_collaborators: int

# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID - consistent with blueprint_routes.py"""
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

async def load_annotations(document_id: str, storage_service) -> List[dict]:
    """Load all annotations for a document - unified blob naming"""
    annotations_blob_name = f"{document_id}_annotations.json"
    try:
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob_name
        )
        annotations = json.loads(annotations_data)
        logger.info(f"Loaded {len(annotations)} annotations for {document_id}")
        return annotations
    except Exception as e:
        logger.info(f"No annotations file found for {document_id}: {e}")
        return []

async def load_all_chats(document_id: str, storage_service) -> List[dict]:
    """Load all chat activity for a document - unified blob naming"""
    activity_blob_name = f"{document_id}_all_chats.json"
    try:
        activity_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name
        )
        chats = json.loads(activity_data)
        logger.info(f"Loaded {len(chats)} chat messages for {document_id}")
        return chats
    except Exception as e:
        logger.info(f"No chat activity file found for {document_id}: {e}")
        return []

async def get_document_context(document_id: str, storage_service) -> Dict[str, any]:
    """Get document context and basic info"""
    try:
        context_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{document_id}_context.txt"
        )
        return {
            "exists": True,
            "status": "ready",
            "char_count": len(context_text),
            "token_count": len(context_text) // 4,  # Rough estimate
            "context_text": context_text
        }
    except Exception as e:
        logger.warning(f"Could not find context file for {document_id}: {e}")
        return {
            "exists": False,
            "status": "not_found",
            "char_count": 0,
            "token_count": 0,
            "context_text": ""
        }

def calculate_collaborator_stats(annotations: List[dict], chats: List[dict]) -> Dict[str, any]:
    """Calculate comprehensive collaborator statistics"""
    # Get all unique authors
    annotation_authors = {ann.get("author") for ann in annotations if ann.get("author")}
    chat_authors = {chat.get("author") for chat in chats if chat.get("author")}
    all_authors = annotation_authors.union(chat_authors)
    
    collaborator_stats = []
    for author in all_authors:
        author_annotations = len([ann for ann in annotations if ann.get("author") == author])
        author_chats = len([chat for chat in chats if chat.get("author") == author])
        
        collaborator_stats.append({
            "author": author,
            "annotations_count": author_annotations,
            "chats_count": author_chats,
            "total_interactions": author_annotations + author_chats
        })
    
    # Sort by total interactions (most active first)
    collaborator_stats.sort(key=lambda x: x["total_interactions"], reverse=True)
    
    return {
        "collaborators": collaborator_stats,
        "unique_authors": list(all_authors),
        "total_collaborators": len(all_authors)
    }

def analyze_annotation_types(annotations: List[dict]) -> Dict[str, int]:
    """Analyze and count annotation types"""
    annotation_types = {}
    for ann in annotations:
        ann_type = ann.get("annotation_type", "note")
        annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
    return annotation_types

def get_recent_activity(annotations: List[dict], chats: List[dict], limit: int = 10) -> Dict[str, any]:
    """Get recent activity sorted by timestamp"""
    # Combine and sort all activities by timestamp
    all_activity = []
    
    # Add annotations with activity type
    for ann in annotations:
        if ann.get("timestamp"):
            all_activity.append({**ann, "activity_type": "annotation"})
    
    # Add chats with activity type
    for chat in chats:
        if chat.get("timestamp"):
            all_activity.append({**chat, "activity_type": "chat"})
    
    # Sort by timestamp (most recent first)
    all_activity.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Get recent items
    recent_activity = all_activity[:limit]
    
    # Separate back into annotations and chats for response
    recent_annotations = [item for item in recent_activity if item.get("activity_type") == "annotation"]
    recent_chats = [item for item in recent_activity if item.get("activity_type") == "chat"]
    
    # Get active authors from recent activity
    active_authors = list(set(
        item.get("author") for item in recent_activity 
        if item.get("author")
    ))
    
    return {
        "recent_annotations": recent_annotations,
        "recent_chats": recent_chats,
        "active_authors": active_authors,
        "total_recent_items": len(recent_activity)
    }

def calculate_document_metrics(annotations: List[dict], chats: List[dict], context_info: Dict[str, any]) -> Dict[str, any]:
    """Calculate comprehensive document metrics"""
    # Get all timestamps for last activity calculation
    all_timestamps = []
    
    for item in annotations + chats:
        if item.get("timestamp"):
            all_timestamps.append(item["timestamp"])
    
    last_activity = max(all_timestamps) if all_timestamps else None
    
    # Calculate page statistics from annotations
    page_numbers = [
        ann.get("page_number", 1) 
        for ann in annotations 
        if isinstance(ann.get("page_number"), int)
    ]
    max_page = max(page_numbers) if page_numbers else 1
    
    return {
        "total_annotations": len(annotations),
        "total_chats": len(chats),
        "total_pages": max_page,
        "total_characters": context_info.get("char_count", 0),
        "estimated_tokens": context_info.get("token_count", 0),
        "last_activity": last_activity,
        "status": context_info.get("status", "unknown")
    }

# --- API Routes ---

@document_router.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(request: Request, document_id: str):
    """Get comprehensive statistics for a specific document"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service

        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        # Get document context and verify existence
        context_info = await get_document_context(clean_document_id, storage_service)
        if not context_info["exists"]:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        # Load annotations and chats
        annotations = await load_annotations(clean_document_id, storage_service)
        chats = await load_all_chats(clean_document_id, storage_service)

        # Calculate statistics
        collaborator_data = calculate_collaborator_stats(annotations, chats)
        annotation_types = analyze_annotation_types(annotations)
        metrics = calculate_document_metrics(annotations, chats, context_info)

        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=metrics["total_annotations"],
            total_pages=metrics["total_pages"],
            total_characters=metrics["total_characters"],
            estimated_tokens=metrics["estimated_tokens"],
            unique_collaborators=collaborator_data["unique_authors"],
            collaborator_count=collaborator_data["total_collaborators"],
            annotation_types=annotation_types,
            last_activity=metrics["last_activity"],
            status=metrics["status"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_document_statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")

@document_router.get("/documents/{document_id}/activity", response_model=DocumentActivityResponse)
async def get_document_activity(
    request: Request, 
    document_id: str, 
    limit: int = Query(10, ge=1, le=50, description="Number of recent items to return")
):
    """Get recent activity (chats and annotations) for a document"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        # Verify document exists
        context_info = await get_document_context(clean_document_id, storage_service)
        if not context_info["exists"]:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        # Load activity data
        annotations = await load_annotations(clean_document_id, storage_service)
        chats = await load_all_chats(clean_document_id, storage_service)

        # Get recent activity
        activity_data = get_recent_activity(annotations, chats, limit)

        return DocumentActivityResponse(
            document_id=clean_document_id,
            recent_annotations=activity_data["recent_annotations"],
            recent_chats=activity_data["recent_chats"],
            active_authors=activity_data["active_authors"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document activity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document activity: {str(e)}")

@document_router.get("/documents/{document_id}/collaborators", response_model=CollaboratorsResponse)
async def get_document_collaborators(request: Request, document_id: str):
    """Get a list of all collaborators and their interaction statistics"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        # Verify document exists
        context_info = await get_document_context(clean_document_id, storage_service)
        if not context_info["exists"]:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        # Load activity data
        annotations = await load_annotations(clean_document_id, storage_service)
        chats = await load_all_chats(clean_document_id, storage_service)

        # Calculate collaborator statistics
        collaborator_data = calculate_collaborator_stats(annotations, chats)

        # Convert to Pydantic models
        collaborator_models = [
            CollaboratorStats(**collab) for collab in collaborator_data["collaborators"]
        ]

        return CollaboratorsResponse(
            document_id=clean_document_id,
            collaborators=collaborator_models,
            total_collaborators=collaborator_data["total_collaborators"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collaborators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collaborators: {str(e)}")

@document_router.get("/documents/{document_id}/summary")
async def get_document_summary(request: Request, document_id: str):
    """Get a comprehensive summary of document activity and statistics"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service is not available")

        # Get document context and verify existence
        context_info = await get_document_context(clean_document_id, storage_service)
        if not context_info["exists"]:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        # Load all data
        annotations = await load_annotations(clean_document_id, storage_service)
        chats = await load_all_chats(clean_document_id, storage_service)

        # Calculate comprehensive statistics
        collaborator_data = calculate_collaborator_stats(annotations, chats)
        annotation_types = analyze_annotation_types(annotations)
        metrics = calculate_document_metrics(annotations, chats, context_info)
        recent_activity = get_recent_activity(annotations, chats, 5)

        return {
            "document_id": clean_document_id,
            "overview": {
                "status": metrics["status"],
                "total_characters": metrics["total_characters"],
                "estimated_tokens": metrics["estimated_tokens"],
                "total_pages": metrics["total_pages"],
                "last_activity": metrics["last_activity"]
            },
            "collaboration": {
                "total_collaborators": collaborator_data["total_collaborators"],
                "most_active_collaborators": collaborator_data["collaborators"][:3],
                "total_annotations": metrics["total_annotations"],
                "total_chats": metrics["total_chats"]
            },
            "annotation_breakdown": annotation_types,
            "recent_activity_preview": {
                "recent_annotations": len(recent_activity["recent_annotations"]),
                "recent_chats": len(recent_activity["recent_chats"]),
                "active_authors": recent_activity["active_authors"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document summary: {str(e)}")

# NOTE: Delete endpoints are handled in blueprint_routes.py to avoid route conflicts
# This file focuses on read-only analytics and statistics
