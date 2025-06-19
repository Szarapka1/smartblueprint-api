# app/api/routes/document_routes.py
# (Renamed from session_routes.py to better reflect shared document functionality)

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import json
from app.core.config import get_settings

document_router = APIRouter()
settings = get_settings()

class DocumentStatsResponse(BaseModel):
    document_id: str
    total_annotations: int
    total_pages: int
    total_characters: int
    estimated_tokens: int
    unique_authors: List[str]
    annotation_types: Dict[str, int]
    last_activity: Optional[str]
    status: str

class DocumentActivityResponse(BaseModel):
    document_id: str
    recent_annotations: List[dict]
    recent_chats: List[dict]
    active_authors: List[str]

def validate_document_id(document_id: str) -> str:
    """Validate document ID matches other routes"""
    import re
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters"
        )
    
    return clean_id

async def load_document_annotations(document_id: str, storage_service) -> List[dict]:
    """Load annotations for a document"""
    annotations_blob_name = f"{document_id}_all_annotations.json"
    
    try:
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob_name
        )
        return json.loads(annotations_data)
    except Exception:
        return []

async def load_document_activity(document_id: str, storage_service) -> List[dict]:
    """Load chat activity log for a document"""
    activity_blob_name = f"{document_id}_service_activity.json"
    
    try:
        activity_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name
        )
        return json.loads(activity_data)
    except Exception:
        return []

@document_router.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(
    request: Request,
    document_id: str
):
    """
    Get comprehensive statistics about a shared document including annotations,
    activity, and processing information.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        ai_service = request.app.state.ai_service
        
        # Get document processing info
        doc_info = await ai_service.get_document_info(clean_document_id, storage_service)
        
        if doc_info.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail=f"Document '{clean_document_id}' not found"
            )
        
        # Get annotations
        annotations = await load_document_annotations(clean_document_id, storage_service)
        
        # Calculate statistics
        unique_authors = list(set(ann.get("author", "Unknown") for ann in annotations))
        
        annotation_types = {}
        for ann in annotations:
            ann_type = ann.get("annotation_type", "note")
            annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
        
        # Get last activity timestamp
        last_activity = None
        if annotations:
            last_activity = max(
                ann.get("created_at", ann.get("updated_at", "")) 
                for ann in annotations
            )
        
        # Estimate total pages from page numbers in annotations
        page_numbers = [ann.get("page_number", 1) for ann in annotations]
        max_page = max(page_numbers) if page_numbers else 1
        
        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=len(annotations),
            total_pages=max_page,
            total_characters=doc_info.get("total_characters", 0),
            estimated_tokens=doc_info.get("estimated_tokens", 0),
            unique_authors=unique_authors,
            annotation_types=annotation_types,
            last_activity=last_activity,
            status=doc_info.get("status", "unknown")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")

@document_router.get("/documents/{document_id}/activity", response_model=DocumentActivityResponse)
async def get_document_activity(
    request: Request,
    document_id: str,
    limit: int = 20
):
    """
    Get recent activity on a shared document including annotations and chats.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Get recent activity log
        activities = await load_document_activity(clean_document_id, storage_service)
        
        # Get recent annotations (last 10)
        annotations = await load_document_annotations(clean_document_id, storage_service)
        recent_annotations = sorted(
            annotations, 
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )[:10]
        
        # Get recent chat activities
        recent_chats = [
            activity for activity in activities 
            if activity.get("type") == "chat"
        ][-10:]  # Last 10 chats
        
        # Get active authors (those with activity in recent activities)
        active_authors = list(set(
            activity.get("author", "Unknown") 
            for activity in activities[-limit:]
        ))
        
        return DocumentActivityResponse(
            document_id=clean_document_id,
            recent_annotations=recent_annotations,
            recent_chats=recent_chats,
            active_authors=active_authors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document activity: {str(e)}")

@document_router.get("/documents/{document_id}/collaborators")
async def get_document_collaborators(
    request: Request,
    document_id: str
):
    """
    Get list of all users who have interacted with this document.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Get authors from annotations
        annotations = await load_document_annotations(clean_document_id, storage_service)
        annotation_authors = set(ann.get("author", "Unknown") for ann in annotations)
        
        # Get authors from activity log
        activities = await load_document_activity(clean_document_id, storage_service)
        activity_authors = set(activity.get("author", "Unknown") for activity in activities)
        
        # Combine all collaborators
        all_collaborators = list(annotation_authors.union(activity_authors))
        all_collaborators = [author for author in all_collaborators if author != "Unknown"]
        
        # Get collaboration stats
        collaborator_stats = []
        for author in all_collaborators:
            author_annotations = len([ann for ann in annotations if ann.get("author") == author])
            author_chats = len([act for act in activities if act.get("author") == author and act.get("type") == "chat"])
            
            collaborator_stats.append({
                "author": author,
                "annotations_count": author_annotations,
                "chats_count": author_chats,
                "total_interactions": author_annotations + author_chats
            })
        
        # Sort by total interactions
        collaborator_stats.sort(key=lambda x: x["total_interactions"], reverse=True)
        
        return {
            "document_id": clean_document_id,
            "collaborators": collaborator_stats,
            "total_collaborators": len(collaborator_stats)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collaborators: {str(e)}")

@document_router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    author: str
):
    """
    Delete a shared document and all associated data (annotations, activity, cached chunks).
    Use with caution - this affects all users.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # List of blobs to delete
        blobs_to_delete = [
            f"{clean_document_id}.pdf",  # Original PDF
            f"{clean_document_id}_context.txt",  # Extracted text
            f"{clean_document_id}_chunks.json",  # Cached chunks
            f"{clean_document_id}_all_annotations.json",  # Annotations
            f"{clean_document_id}_service_activity.json",  # Activity log
            f"{clean_document_id}_all_chats.json"  # Chat log
        ]
        
        # Delete from main container
        try:
            await storage_service.delete_blob(
                container_name=settings.AZURE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}.pdf"
            )
        except:
            pass  # PDF might not exist in main container
        
        # Delete from cache container
        deleted_count = 0
        for blob_name in blobs_to_delete[1:]:  # Skip PDF, already tried above
            try:
                await storage_service.delete_blob(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
                deleted_count += 1
            except:
                pass  # Some files might not exist
        
        # Also delete any page images and user chat files
        try:
            all_blobs = await storage_service.list_blobs(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME
            )
            document_blobs = [blob for blob in all_blobs if blob.startswith(f"{clean_document_id}_")]
            
            for blob in document_blobs:
                try:
                    await storage_service.delete_blob(
                        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=blob
                    )
                    deleted_count += 1
                except:
                    pass
        except:
            pass
        
        return {
            "message": f"Document '{clean_document_id}' deleted successfully",
            "document_id": clean_document_id,
            "deleted_by": author,
            "files_deleted": deleted_count,
            "warning": "This action affects all users who had access to this document"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")