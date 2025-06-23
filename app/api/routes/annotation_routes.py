# app/api/routes/annotation_routes.py

from fastapi import APIRouter, Request, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
import uuid
import os
import re
from app.core.config import get_settings

annotation_router = APIRouter()
settings = get_settings()

class CreateAnnotationRequest(BaseModel):
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str = "note"  # note, issue, question, highlight, etc.
    author: str

class AnnotationResponse(BaseModel):
    id: str
    document_id: str
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str
    author: str
    is_private: bool
    created_at: str
    updated_at: Optional[str] = None
    published_at: Optional[str] = None

def validate_document_id(document_id: str) -> str:
    """Validate document ID"""
    import re
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters"
        )
    
    return clean_id

def validate_admin_access(admin_token: str) -> bool:
    """Validate admin access using environment variable"""
    expected_token = os.getenv("ADMIN_SECRET_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Admin access not configured"
        )
    return admin_token == expected_token

async def load_all_annotations(document_id: str, storage_service) -> List[dict]:
    """Load ALL annotations for the service - both private and public"""
    annotations_blob_name = f"{document_id}_annotations.json"  # FIXED: unified naming
    
    try:
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob_name
        )
        return json.loads(annotations_data)
    except Exception:
        return []

async def save_all_annotations(document_id: str, annotations: List[dict], storage_service):
    """Save ALL annotations - service has access to everything"""
    annotations_blob_name = f"{document_id}_annotations.json"  # FIXED: unified naming
    annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
    
    await storage_service.upload_file(
        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
        blob_name=annotations_blob_name,
        data=annotations_json.encode('utf-8')
    )

async def log_user_activity(document_id: str, activity_type: str, author: str, details: dict, storage_service):
    """Log all user activity for service analytics"""
    activity_blob_name = f"{document_id}_all_chats.json"  # FIXED: unified naming
    
    try:
        activity_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name
        )
        activities = json.loads(activity_data)
    except:
        activities = []
    
    new_activity = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": activity_type,
        "author": author,
        "details": details,
        "document_id": document_id
    }
    
    activities.append(new_activity)
    
    # Keep configurable number of activities
    max_activities = getattr(settings, 'MAX_ACTIVITY_LOGS', 1000)  # FIXED: use settings
    if len(activities) > max_activities:
        activities = activities[-max_activities:]
    
    activity_json = json.dumps(activities, indent=2, ensure_ascii=False)
    await storage_service.upload_file(
        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
        blob_name=activity_blob_name,
        data=activity_json.encode('utf-8')
    )

def filter_user_visible_annotations(annotations: List[dict], requesting_author: str) -> List[dict]:
    """Filter what the user can see: their private + everyone's public"""
    visible = []
    for annotation in annotations:
        # Show public annotations to everyone
        if not annotation.get("is_private", True):
            visible.append(annotation)
        # Show private annotations only to their author
        elif annotation.get("author") == requesting_author:
            visible.append(annotation)
    return visible

@annotation_router.get("/documents/{document_id}/annotations")
async def get_user_visible_annotations(
    request: Request, 
    document_id: str,
    author: str = Query(...)  # Query parameter: who is requesting?
):
    """
    USER VIEW: Returns only annotations the user should see
    SERVICE: Logs the request and has access to ALL data
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # SERVICE: Load ALL annotations (you have everything)
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # SERVICE: Log this request for analytics
        await log_user_activity(
            document_id=clean_document_id,
            activity_type="view_annotations",
            author=author,
            details={"total_annotations_available": len(all_annotations)},
            storage_service=storage_service
        )
        
        # USER: Filter to what they're allowed to see
        visible_annotations = filter_user_visible_annotations(all_annotations, author)
        
        return {
            "document_id": clean_document_id,
            "annotations": visible_annotations,
            "visible_count": len(visible_annotations),
            "requesting_author": author
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch annotations: {str(e)}")

@annotation_router.post("/documents/{document_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(
    request: Request,
    document_id: str,
    annotation_request: CreateAnnotationRequest
):
    """
    USER: Creates annotation (private by default)
    SERVICE: Stores everything and logs all activity
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
        
        # Validate input
        if len(annotation_request.text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Annotation text cannot be empty"
            )
        
        if annotation_request.page_number < 1:
            raise HTTPException(
                status_code=400,
                detail="Page number must be greater than 0"
            )
        
        # SERVICE: Load all existing annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Create new annotation
        annotation_id = str(uuid.uuid4())[:8]
        current_time = datetime.utcnow().isoformat() + "Z"
        
        new_annotation = {
            "id": annotation_id,
            "document_id": clean_document_id,
            "page_number": annotation_request.page_number,
            "x": annotation_request.x,
            "y": annotation_request.y,
            "text": annotation_request.text.strip(),
            "annotation_type": annotation_request.annotation_type,
            "author": annotation_request.author,
            "is_private": True,  # Always private by default
            "created_at": current_time,
            "updated_at": None,
            "published_at": None
        }
        
        # SERVICE: Store everything
        all_annotations.append(new_annotation)
        await save_all_annotations(clean_document_id, all_annotations, storage_service)
        
        # SERVICE: Log this activity
        await log_user_activity(
            document_id=clean_document_id,
            activity_type="create_annotation",
            author=annotation_request.author,
            details={
                "annotation_id": annotation_id,
                "annotation_type": annotation_request.annotation_type,
                "page_number": annotation_request.page_number,
                "text_length": len(annotation_request.text)
            },
            storage_service=storage_service
        )
        
        return AnnotationResponse(**new_annotation)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@annotation_router.post("/documents/{document_id}/annotations/{annotation_id}/publish")
async def publish_annotation(
    request: Request,
    document_id: str,
    annotation_id: str,
    author: str = Query(...)  # Query parameter
):
    """
    USER: Publishes their private annotation to make it visible to others
    SERVICE: Tracks all publishing activity
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # SERVICE: Load all annotations
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Find the annotation
        annotation_to_publish = None
        annotation_index = None
        
        for i, annotation in enumerate(all_annotations):
            if annotation.get("id") == annotation_id:
                annotation_to_publish = annotation
                annotation_index = i
                break
        
        if not annotation_to_publish:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # Verify the user owns this annotation
        if annotation_to_publish.get("author") != author:
            raise HTTPException(status_code=403, detail="You can only publish your own annotations")
        
        # Already public?
        if not annotation_to_publish.get("is_private", True):
            raise HTTPException(status_code=400, detail="Annotation is already public")
        
        # Publish it
        current_time = datetime.utcnow().isoformat() + "Z"
        annotation_to_publish["is_private"] = False
        annotation_to_publish["published_at"] = current_time
        
        # SERVICE: Save everything
        all_annotations[annotation_index] = annotation_to_publish
        await save_all_annotations(clean_document_id, all_annotations, storage_service)
        
        # SERVICE: Log publishing activity
        await log_user_activity(
            document_id=clean_document_id,
            activity_type="publish_annotation",
            author=author,
            details={
                "annotation_id": annotation_id,
                "annotation_type": annotation_to_publish.get("annotation_type"),
                "published_at": current_time
            },
            storage_service=storage_service
        )
        
        return {
            "message": "Annotation published successfully",
            "annotation_id": annotation_id,
            "published_at": current_time
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish annotation: {str(e)}")

# SERVICE ADMIN ENDPOINTS - Protected by environment variable

@annotation_router.get("/admin/documents/{document_id}/all-annotations")
async def admin_get_all_annotations(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    SERVICE ADMIN: Get ALL annotations from ALL users (private + public)
    Use this for analytics, moderation, backups, etc.
    """
    try:
        # Validate admin access
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # SERVICE: Get ALL annotations from ALL users
        all_annotations = await load_all_annotations(clean_document_id, storage_service)
        
        # Analytics data
        analytics = {
            "total_annotations": len(all_annotations),
            "private_annotations": len([a for a in all_annotations if a.get("is_private", True)]),
            "public_annotations": len([a for a in all_annotations if not a.get("is_private", True)]),
            "unique_authors": len(set(a.get("author") for a in all_annotations)),
            "annotation_types": {}
        }
        
        # Count annotation types
        for annotation in all_annotations:
            ann_type = annotation.get("annotation_type", "unknown")
            analytics["annotation_types"][ann_type] = analytics["annotation_types"].get(ann_type, 0) + 1
        
        return {
            "document_id": clean_document_id,
            "all_annotations": all_annotations,  # Everything from everyone
            "analytics": analytics
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@annotation_router.get("/admin/documents/{document_id}/activity")
async def admin_get_all_activity(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    SERVICE ADMIN: Get complete activity log for analytics
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        activity_blob_name = f"{clean_document_id}_all_chats.json"
        try:
            activity_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=activity_blob_name
            )
            activities = json.loads(activity_data)
        except:
            activities = []
        
        return {
            "document_id": clean_document_id,
            "total_activities": len(activities),
            "activities": activities
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")
