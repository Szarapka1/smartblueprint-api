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
    import re
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(status_code=400, detail="Document ID must be at least 3 characters")
    return clean_id

async def load_document_annotations(document_id: str, storage_service) -> List[dict]:
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
    activity_blob_name = f"{document_id}_service_activity.json"
    try:
        activity_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name
        )
        return json.loads(activity_data)
    except Exception:
        return []

async def resolve_real_filename(document_id: str, storage_service) -> str:
    """Look up actual filename for a document_id using a mapping json."""
    try:
        map_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name="document_map.json"
        )
        mapping = json.loads(map_data)
        return mapping.get(document_id, f"{document_id}.pdf")
    except Exception:
        return f"{document_id}.pdf"

@document_router.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(request: Request, document_id: str):
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        ai_service = request.app.state.ai_service

        # Resolve the real file name from the ID
        real_filename = await resolve_real_filename(clean_document_id, storage_service)

        # Try getting document info from AI service
        try:
            doc_info = await ai_service.get_document_info(real_filename, storage_service)
            if not isinstance(doc_info, dict):
                raise ValueError("AI service did not return a valid dictionary")
        except Exception as e:
            logging.warning(f"AI service failed: {e}")
            doc_info = {
                "total_characters": 0,
                "estimated_tokens": 0,
                "status": "unknown"
            }

        # Check if the document wasn't found explicitly
        if doc_info.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")

        # Load annotations
        annotations = await load_document_annotations(clean_document_id, storage_service)
        unique_authors = list(set(ann.get("author", "Unknown") for ann in annotations))

        annotation_types = {}
        for ann in annotations:
            ann_type = ann.get("annotation_type", "note")
            annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1

        # Get the latest activity timestamp
        last_activity = None
        if annotations:
            last_activity = max(
                ann.get("created_at") or ann.get("updated_at", "") for ann in annotations
            )

        # Get page stats
        page_numbers = [ann.get("page_number", 1) for ann in annotations]
        max_page = max(page_numbers) if page_numbers else 1

        # Final structured response
        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=len(annotations),
            total_pages=max_page,
            total_characters=doc_info.get("total_characters", 0),
            estimated_tokens=doc_info.get("estimated_tokens", 0),
            unique_authors=unique_authors,
            annotation_types=annotation_types,
            last_activity=last_activity,
            status=doc_info.get("status", "ok")
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in get_document_statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")


@document_router.get("/documents/{document_id}/activity", response_model=DocumentActivityResponse)
async def get_document_activity(request: Request, document_id: str, limit: int = 20):
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service

        activities = await load_document_activity(clean_document_id, storage_service)
        annotations = await load_document_annotations(clean_document_id, storage_service)
        recent_annotations = sorted(annotations, key=lambda x: x.get("created_at", ""), reverse=True)[:10]
        recent_chats = [activity for activity in activities if activity.get("type") == "chat"][-10:]
        active_authors = list(set(activity.get("author", "Unknown") for activity in activities[-limit:]))

        return DocumentActivityResponse(
            document_id=clean_document_id,
            recent_annotations=recent_annotations,
            recent_chats=recent_chats,
            active_authors=active_authors
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document activity: {str(e)}")

@document_router.get("/documents/{document_id}/collaborators")
async def get_document_collaborators(request: Request, document_id: str):
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        annotations = await load_document_annotations(clean_document_id, storage_service)
        activities = await load_document_activity(clean_document_id, storage_service)

        annotation_authors = set(ann.get("author", "Unknown") for ann in annotations)
        activity_authors = set(activity.get("author", "Unknown") for activity in activities)
        all_collaborators = list(annotation_authors.union(activity_authors))
        all_collaborators = [author for author in all_collaborators if author != "Unknown"]

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

        collaborator_stats.sort(key=lambda x: x["total_interactions"], reverse=True)

        return {
            "document_id": clean_document_id,
            "collaborators": collaborator_stats,
            "total_collaborators": len(collaborator_stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collaborators: {str(e)}")

@document_router.delete("/documents/{document_id}")
async def delete_document(request: Request, document_id: str, author: str):
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service

        blobs_to_delete = [
            f"{clean_document_id}.pdf",
            f"{clean_document_id}_context.txt",
            f"{clean_document_id}_chunks.json",
            f"{clean_document_id}_all_annotations.json",
            f"{clean_document_id}_service_activity.json",
            f"{clean_document_id}_all_chats.json"
        ]

        try:
            await storage_service.delete_blob(
                container_name=settings.AZURE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}.pdf"
            )
        except:
            pass

        deleted_count = 0
        for blob_name in blobs_to_delete[1:]:
            try:
                await storage_service.delete_blob(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
                deleted_count += 1
            except:
                pass

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
