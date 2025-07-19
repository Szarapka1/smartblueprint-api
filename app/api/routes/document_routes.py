# app/api/routes/document_routes.py - FINAL, UNIFIED, AND CORRECTED VERSION

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import re

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

document_router = APIRouter()

# --- Pydantic Models for Analytics ---

class DocumentStatsResponse(BaseModel):
    document_id: str
    total_annotations: int
    total_chats: int
    total_pages: int
    collaborator_count: int
    last_activity: Optional[str]
    status: str

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
    """Validate and sanitize document ID to prevent injection attacks."""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(status_code=400, detail="Document ID must be a non-empty string")
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip()).strip('_')
    if not clean_id or len(clean_id) < 3 or len(clean_id) > 50:
        raise HTTPException(status_code=400, detail="Invalid Document ID format.")
    return clean_id

async def _get_service(request: Request, service_name: str):
    """Helper to safely retrieve a service from the application state."""
    service = getattr(request.app.state, service_name, None)
    if not service:
        raise HTTPException(status_code=503, detail=f"{service_name.replace('_', ' ').title()} is not available.")
    return service

# --- CORE DATA ENDPOINTS ---

@document_router.get("/documents", summary="List all processed documents")
async def list_all_documents(request: Request):
    """Lists all document IDs that have been successfully processed."""
    storage_service = await _get_service(request, 'storage_service')
    try:
        document_ids = await storage_service.list_document_ids(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        return {"document_ids": document_ids, "count": len(document_ids)}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve document list.")

@document_router.get("/documents/{document_id}", summary="Get document metadata")
async def get_document_details(request: Request, document_id: str):
    """Retrieves the main metadata file for a single, processed document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    try:
        metadata = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        return JSONResponse(content=metadata)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Metadata for document '{clean_document_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get metadata for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching metadata.")

@document_router.get("/documents/{document_id}/grid-systems", summary="Get the document's grid system data")
async def get_document_grid_systems(request: Request, document_id: str):
    """Retrieves the processed grid systems JSON file for a document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    try:
        blob_name = f"{clean_document_id}_grid_systems.json"
        grid_data = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        return JSONResponse(content=grid_data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Grid system data for document '{clean_document_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get grid system for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching grid system data.")

@document_router.get("/documents/{document_id}/document-index", summary="Get the document's index data")
async def get_document_index(request: Request, document_id: str):
    """Retrieves the processed document index JSON file."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    try:
        blob_name = f"{clean_document_id}_document_index.json"
        index_data = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        return JSONResponse(content=index_data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document index for '{clean_document_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get document index for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching the document index.")

# --- ANALYTICS & COLLABORATION ENDPOINTS ---

@document_router.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(request: Request, document_id: str):
    """Get comprehensive statistics for a specific document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Fetch all required data in parallel
        metadata_task = storage_service.download_blob_as_json(settings.AZURE_CACHE_CONTAINER_NAME, f"{clean_document_id}_metadata.json")
        notes_task = storage_service.download_blob_as_json(settings.AZURE_CACHE_CONTAINER_NAME, f"{clean_document_id}_annotations.json")
        chats_task = storage_service.download_blob_as_json(settings.AZURE_CACHE_CONTAINER_NAME, f"{clean_document_id}_all_chats.json")
        
        results = await asyncio.gather(metadata_task, notes_task, chats_task, return_exceptions=True)
        
        metadata = results[0] if not isinstance(results[0], Exception) else {}
        annotations = results[1] if not isinstance(results[1], Exception) else []
        chats = results[2] if not isinstance(results[2], Exception) else []
        
        if not metadata:
             raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found or is still processing.")

        collaborators = {ann.get("author") for ann in annotations if ann.get("author")}
        collaborators.update({chat.get("author") for chat in chats if chat.get("author")})
        
        all_timestamps = [item.get("timestamp") for item in annotations + chats if item.get("timestamp")]
        
        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=len(annotations),
            total_chats=len(chats),
            total_pages=metadata.get('page_count', 0),
            collaborator_count=len(collaborators),
            last_activity=max(all_timestamps) if all_timestamps else None,
            status=metadata.get('status', 'ready')
        )
    except Exception as e:
        logger.error(f"Failed to get document statistics for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document statistics.")

@document_router.get("/documents/{document_id}/collaborators", response_model=CollaboratorsResponse)
async def get_document_collaborators(request: Request, document_id: str):
    """Get a list of all collaborators and their interaction statistics."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    try:
        notes_task = storage_service.download_blob_as_json(settings.AZURE_CACHE_CONTAINER_NAME, f"{clean_document_id}_annotations.json")
        chats_task = storage_service.download_blob_as_json(settings.AZURE_CACHE_CONTAINER_NAME, f"{clean_document_id}_all_chats.json")
        
        results = await asyncio.gather(notes_task, chats_task, return_exceptions=True)
        annotations = results[0] if not isinstance(results[0], Exception) else []
        chats = results[1] if not isinstance(results[1], Exception) else []

        authors = {ann.get("author") for ann in annotations if ann.get("author")}
        authors.update({chat.get("author") for chat in chats if chat.get("author")})

        collaborator_stats = []
        for author in authors:
            ann_count = sum(1 for ann in annotations if ann.get("author") == author)
            chat_count = sum(1 for chat in chats if chat.get("author") == author)
            collaborator_stats.append(CollaboratorStats(
                author=author,
                annotations_count=ann_count,
                chats_count=chat_count,
                total_interactions=ann_count + chat_count
            ))

        collaborator_stats.sort(key=lambda x: x.total_interactions, reverse=True)

        return CollaboratorsResponse(
            document_id=clean_document_id,
            collaborators=collaborator_stats,
            total_collaborators=len(collaborator_stats)
        )
    except Exception as e:
        logger.error(f"Failed to get collaborators for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve collaborator data.")
