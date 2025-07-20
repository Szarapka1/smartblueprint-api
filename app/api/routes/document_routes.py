# app/api/routes/document_routes.py - COMPLETE FIXED VERSION

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import json
import re
import asyncio
from datetime import datetime
from collections import defaultdict

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

document_router = APIRouter()

# Cache control headers
CACHE_CONTROL_METADATA = "public, max-age=3600"  # 1 hour
CACHE_CONTROL_STATIC = "public, max-age=86400"  # 24 hours
CACHE_CONTROL_THUMBNAILS = "public, max-age=604800"  # 7 days
CACHE_CONTROL_DYNAMIC = "no-cache, must-revalidate"

# --- Pydantic Models ---

class DocumentStatsResponse(BaseModel):
    document_id: str
    total_annotations: int
    total_chats: int
    total_pages: int
    collaborator_count: int
    last_activity: Optional[str]
    status: str
    has_tables: bool
    has_grids: bool
    has_thumbnails: bool
    grid_pages: List[int]
    table_pages: List[int]

class CollaboratorStats(BaseModel):
    author: str
    annotations_count: int
    chats_count: int
    total_interactions: int
    last_activity: Optional[str]

class CollaboratorsResponse(BaseModel):
    document_id: str
    collaborators: List[CollaboratorStats]
    total_collaborators: int

class DocumentMetadataResponse(BaseModel):
    document_id: str
    page_count: int
    total_pages: int
    processing_time: float
    file_size_mb: float
    grid_systems_detected: int
    has_tables: bool
    has_thumbnails: bool
    table_count: int
    jpeg_settings: Dict[str, Any]
    thumbnail_settings: Dict[str, Any]
    drawing_types: Dict[str, List[int]]
    sheet_numbers: Dict[str, int]
    scales_found: Dict[str, str]

class PageDataResponse(BaseModel):
    page_number: int
    thumbnail_url: str
    image_url: str
    has_grid: bool
    grid_system: Optional[Dict[str, Any]]
    has_tables: bool
    table_count: int
    drawing_type: Optional[str]
    sheet_number: Optional[str]
    scale: Optional[str]
    image_dimensions: Dict[str, int]

class BatchPageDataResponse(BaseModel):
    document_id: str
    pages: List[PageDataResponse]
    total_pages: int

class CompleteDocumentDataResponse(BaseModel):
    document_id: str
    metadata: Optional[Dict[str, Any]]
    grid_systems: Optional[Dict[str, Any]]
    document_index: Optional[Dict[str, Any]]
    context_text: Optional[str]
    tables: Optional[List[Dict[str, Any]]]
    page_images: Dict[str, Dict[str, str]]
    summary: Dict[str, Any]
    timestamp: str

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

def generate_image_url(storage_service, container_name: str, blob_name: str) -> str:
    """Generate a full URL for a blob."""
    base_url = f"https://{storage_service.blob_service_client.account_name}.blob.core.windows.net"
    return f"{base_url}/{container_name}/{blob_name}"

async def _check_document_ready(storage_service, document_id: str) -> bool:
    """Check if document is fully processed and ready."""
    try:
        # Check for context file (indicates processing complete)
        context_exists = await storage_service.blob_exists(
            settings.AZURE_CACHE_CONTAINER_NAME,
            f"{document_id}_context.txt"
        )
        
        if not context_exists:
            # Check status file
            status_blob = f"{document_id}_status.json"
            if await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, status_blob):
                status_text = await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=status_blob
                )
                status_data = json.loads(status_text)
                return status_data.get('status') in ['ready', 'completed']
        
        return context_exists
    except:
        return False

# --- DOCUMENT LISTING ---

@document_router.get("/documents", summary="List all processed documents")
async def list_all_documents(request: Request):
    """Lists all document IDs that have been successfully processed."""
    storage_service = await _get_service(request, 'storage_service')
    try:
        document_ids = await storage_service.list_document_ids(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        
        processed_documents = []
        for doc_id in document_ids:
            # Check if document is ready
            if await _check_document_ready(storage_service, doc_id):
                processed_documents.append(doc_id)
        
        return JSONResponse(
            content={
                "document_ids": processed_documents, 
                "count": len(processed_documents),
                "total_found": len(document_ids)
            },
            headers={"Cache-Control": CACHE_CONTROL_METADATA}
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve document list.")

# --- DOCUMENT METADATA ---

@document_router.get("/documents/{document_id}", 
                    response_model=DocumentMetadataResponse,
                    summary="Get document metadata")
async def get_document_details(request: Request, document_id: str):
    """Retrieves the complete metadata for a processed document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing or not found"
            )
        
        metadata_json = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        
        # Try to get document index for additional info
        try:
            index_json = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_document_index.json"
            )
        except:
            index_json = {}
        
        # Get image settings from metadata or use defaults
        image_settings = metadata_json.get('image_settings', {
            'high_quality': {'dpi': 150, 'quality': 90, 'progressive': True},
            'thumbnail': {'dpi': 72, 'quality': 75}
        })
        
        response_data = DocumentMetadataResponse(
            document_id=clean_document_id,
            page_count=metadata_json.get('page_count', 0),
            total_pages=metadata_json.get('total_pages', 0),
            processing_time=metadata_json.get('processing_time', 0),
            file_size_mb=metadata_json.get('file_size_mb', 0),
            grid_systems_detected=metadata_json.get('grid_systems_detected', 0),
            has_tables=metadata_json.get('extraction_summary', {}).get('has_tables', False),
            has_thumbnails=metadata_json.get('extraction_summary', {}).get('has_thumbnails', True),
            table_count=metadata_json.get('extraction_summary', {}).get('table_count', 0),
            jpeg_settings=image_settings.get('high_quality', {}),
            thumbnail_settings=image_settings.get('thumbnail', {}),
            drawing_types=index_json.get('drawing_types', {}),
            sheet_numbers=index_json.get('sheet_numbers', {}),
            scales_found=index_json.get('scales_found', {})
        )
        
        return JSONResponse(
            content=response_data.dict(),
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Metadata for document '{clean_document_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get metadata for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching metadata.")

# --- COMPLETE DATA ENDPOINT ---

@document_router.get("/documents/{document_id}/complete-data",
                    response_model=CompleteDocumentDataResponse,
                    summary="Get all document data in one request")
async def get_complete_document_data(
    request: Request,
    document_id: str,
    include_text: bool = Query(True, description="Include full context text"),
    include_tables: bool = Query(True, description="Include extracted tables")
):
    """
    Get all document data in a single comprehensive response.
    This eliminates the need for multiple API calls from the frontend.
    """
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            # Check if it's still processing
            status_blob = f"{clean_document_id}_status.json"
            if await storage_service.blob_exists(cache_container, status_blob):
                status_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=status_blob
                )
                status_data = json.loads(status_text)
                if status_data.get('status') == 'processing':
                    raise HTTPException(
                        status_code=status.HTTP_425_TOO_EARLY,
                        detail=f"Document is still processing"
                    )
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{clean_document_id}' not found or not yet processed"
            )
        
        response_data = {
            "document_id": clean_document_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # 1. Load metadata
        try:
            metadata_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_metadata.json"
            )
            response_data["metadata"] = json.loads(metadata_text)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            response_data["metadata"] = None
        
        # 2. Load grid systems
        try:
            grid_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_grid_systems.json"
            )
            response_data["grid_systems"] = json.loads(grid_text)
        except Exception:
            response_data["grid_systems"] = {}
        
        # 3. Load document index
        try:
            index_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=f"{clean_document_id}_document_index.json"
            )
            response_data["document_index"] = json.loads(index_text)
        except Exception:
            response_data["document_index"] = None
        
        # 4. Load context text (optional)
        if include_text:
            try:
                context_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=f"{clean_document_id}_context.txt"
                )
                response_data["context_text"] = context_text
            except Exception:
                response_data["context_text"] = None
        else:
            response_data["context_text"] = None
        
        # 5. Load tables (optional)
        if include_tables:
            try:
                tables_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=f"{clean_document_id}_tables.json"
                )
                response_data["tables"] = json.loads(tables_text)
            except Exception:
                response_data["tables"] = []
        else:
            response_data["tables"] = []
        
        # 6. Generate page image URLs
        page_count = response_data.get("metadata", {}).get("page_count", 0)
        base_url = f"https://{storage_service.blob_service_client.account_name}.blob.core.windows.net/{cache_container}"
        
        page_images = {}
        for page_num in range(1, page_count + 1):
            thumbnail_blob = f"{clean_document_id}_page_{page_num}_thumb.jpg"
            jpeg_blob = f"{clean_document_id}_page_{page_num}.jpg"
            
            page_images[str(page_num)] = {
                "thumbnail_url": f"{base_url}/{thumbnail_blob}",
                "jpeg_url": f"{base_url}/{jpeg_blob}",
                "thumbnail_blob": thumbnail_blob,
                "jpeg_blob": jpeg_blob
            }
        
        response_data["page_images"] = page_images
        
        # 7. Add processing summary
        response_data["summary"] = {
            "total_pages": page_count,
            "has_grids": bool(response_data.get("grid_systems")),
            "has_tables": bool(response_data.get("tables")),
            "has_thumbnails": response_data.get("metadata", {}).get("extraction_summary", {}).get("has_thumbnails", True),
            "drawing_types": list(response_data.get("document_index", {}).get("drawing_types", {}).keys()) if response_data.get("document_index") else [],
            "grid_pages": response_data.get("document_index", {}).get("grid_pages", []) if response_data.get("document_index") else [],
            "table_pages": response_data.get("document_index", {}).get("table_pages", []) if response_data.get("document_index") else [],
            "image_settings": response_data.get("metadata", {}).get("image_settings", {}) if response_data.get("metadata") else {}
        }
        
        response = CompleteDocumentDataResponse(**response_data)
        
        return JSONResponse(
            content=response.dict(),
            headers={
                "Cache-Control": "public, max-age=600",
                "X-Document-ID": clean_document_id,
                "X-Page-Count": str(page_count)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get complete document data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document data: {str(e)}"
        )

# --- SPECIFIC DATA ENDPOINTS ---

@document_router.get("/documents/{document_id}/grid-systems", 
                    summary="Get comprehensive grid system data")
async def get_document_grid_systems(request: Request, document_id: str):
    """Retrieves the comprehensive grid systems with pixel-accurate coordinates."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        blob_name = f"{clean_document_id}_grid_systems.json"
        grid_data = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        summary = {
            "total_pages_with_grids": len([k for k, v in grid_data.items() if v.get('confidence', 0) > 0.3]),
            "detection_methods": list(set(v.get('detection_method', 'none') for v in grid_data.values())),
            "average_confidence": sum(v.get('confidence', 0) for v in grid_data.values()) / len(grid_data) if grid_data else 0
        }
        
        response = {
            "document_id": clean_document_id,
            "grid_systems": grid_data,
            "summary": summary
        }
        
        return JSONResponse(
            content=response,
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "grid_systems": {},
                "summary": {
                    "total_pages_with_grids": 0,
                    "detection_methods": [],
                    "average_confidence": 0
                }
            },
            headers={"Cache-Control": CACHE_CONTROL_METADATA}
        )
    except Exception as e:
        logger.error(f"Failed to get grid system for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching grid system data.")

@document_router.get("/documents/{document_id}/tables",
                    summary="Get extracted tables data")
async def get_document_tables(request: Request, document_id: str):
    """Retrieves all extracted tables from the document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        blob_name = f"{clean_document_id}_tables.json"
        tables_data = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        pages_with_tables = set()
        total_rows = 0
        
        for table in tables_data:
            pages_with_tables.add(table.get('page_number'))
            total_rows += len(table.get('rows', []))
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "tables": tables_data,
                "summary": {
                    "total_tables": len(tables_data),
                    "pages_with_tables": sorted(list(pages_with_tables)),
                    "total_rows": total_rows
                }
            },
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "tables": [],
                "summary": {
                    "total_tables": 0,
                    "pages_with_tables": [],
                    "total_rows": 0
                }
            },
            headers={"Cache-Control": CACHE_CONTROL_METADATA}
        )
    except Exception as e:
        logger.error(f"Failed to get tables for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching tables.")

@document_router.get("/documents/{document_id}/document-index", 
                    summary="Get the document's index data")
async def get_document_index(request: Request, document_id: str):
    """Retrieves the processed document index JSON file."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        blob_name = f"{clean_document_id}_document_index.json"
        index_data = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        return JSONResponse(
            content=index_data,
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document index for '{clean_document_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get document index for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching the document index.")

@document_router.get("/documents/{document_id}/context",
                    summary="Get document text context")
async def get_document_context(request: Request, document_id: str):
    """Retrieves the full text context of the document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        blob_name = f"{clean_document_id}_context.txt"
        context_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        return Response(
            content=context_text,
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id,
                "Content-Disposition": f"inline; filename=\"{clean_document_id}_context.txt\""
            }
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Context for document '{clean_document_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get context for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching document context.")

# --- PAGE DATA ENDPOINTS ---

@document_router.get("/documents/{document_id}/pages/{page_number}",
                    response_model=PageDataResponse,
                    summary="Get comprehensive data for a specific page")
async def get_page_data(request: Request, document_id: str, page_number: int):
    """Get all data for a specific page including thumbnail URL, image URL, grid, and tables."""
    clean_document_id = validate_document_id(document_id)
    
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be 1 or greater")
    
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        metadata = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        
        if page_number > metadata.get('page_count', 0):
            raise HTTPException(status_code=404, detail=f"Page {page_number} not found")
        
        page_details = None
        for detail in metadata.get('page_details', []):
            if detail.get('page_number') == page_number:
                page_details = detail
                break
        
        grid_system = None
        try:
            grid_systems = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_grid_systems.json"
            )
            grid_system = grid_systems.get(str(page_number))
        except:
            pass
        
        table_count = 0
        try:
            tables = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_tables.json"
            )
            table_count = sum(1 for t in tables if t.get('page_number') == page_number)
        except:
            pass
        
        thumbnail_blob = f"{clean_document_id}_page_{page_number}_thumb.jpg"
        image_blob = f"{clean_document_id}_page_{page_number}.jpg"
        thumbnail_url = generate_image_url(storage_service, settings.AZURE_CACHE_CONTAINER_NAME, thumbnail_blob)
        image_url = generate_image_url(storage_service, settings.AZURE_CACHE_CONTAINER_NAME, image_blob)
        
        response = PageDataResponse(
            page_number=page_number,
            thumbnail_url=thumbnail_url,
            image_url=image_url,
            has_grid=page_details.get('has_grid', False) if page_details else False,
            grid_system=grid_system,
            has_tables=page_details.get('has_tables', False) if page_details else False,
            table_count=table_count,
            drawing_type=page_details.get('drawing_type') if page_details else None,
            sheet_number=page_details.get('sheet_number') if page_details else None,
            scale=page_details.get('scale') if page_details else None,
            image_dimensions=page_details.get('image_dimensions', {}) if page_details else {}
        )
        
        return JSONResponse(
            content=response.dict(),
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id,
                "X-Page-Number": str(page_number)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get page data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve page data")

@document_router.get("/documents/{document_id}/pages",
                    response_model=BatchPageDataResponse,
                    summary="Get data for multiple pages")
async def get_batch_page_data(
    request: Request, 
    document_id: str,
    start_page: int = Query(1, ge=1, description="Starting page number"),
    end_page: Optional[int] = Query(None, ge=1, description="Ending page number (inclusive)"),
    page_list: Optional[str] = Query(None, description="Comma-separated list of page numbers")
):
    """Get data for multiple pages in a single request with thumbnail and image URLs."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        metadata = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        
        total_pages = metadata.get('page_count', 0)
        
        if page_list:
            page_numbers = []
            for p in page_list.split(','):
                try:
                    page_num = int(p.strip())
                    if 1 <= page_num <= total_pages:
                        page_numbers.append(page_num)
                except:
                    pass
        else:
            if not end_page:
                end_page = min(start_page + 9, total_pages)
            else:
                end_page = min(end_page, total_pages)
            
            page_numbers = list(range(start_page, end_page + 1))
        
        page_numbers = page_numbers[:50]
        
        try:
            grid_systems = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_grid_systems.json"
            )
        except:
            grid_systems = {}
        
        try:
            tables = await storage_service.download_blob_as_json(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_tables.json"
            )
        except:
            tables = []
        
        pages = []
        for page_num in page_numbers:
            page_details = None
            for detail in metadata.get('page_details', []):
                if detail.get('page_number') == page_num:
                    page_details = detail
                    break
            
            table_count = sum(1 for t in tables if t.get('page_number') == page_num)
            
            thumbnail_blob = f"{clean_document_id}_page_{page_num}_thumb.jpg"
            image_blob = f"{clean_document_id}_page_{page_num}.jpg"
            thumbnail_url = generate_image_url(storage_service, settings.AZURE_CACHE_CONTAINER_NAME, thumbnail_blob)
            image_url = generate_image_url(storage_service, settings.AZURE_CACHE_CONTAINER_NAME, image_blob)
            
            pages.append(PageDataResponse(
                page_number=page_num,
                thumbnail_url=thumbnail_url,
                image_url=image_url,
                has_grid=page_details.get('has_grid', False) if page_details else False,
                grid_system=grid_systems.get(str(page_num)),
                has_tables=page_details.get('has_tables', False) if page_details else False,
                table_count=table_count,
                drawing_type=page_details.get('drawing_type') if page_details else None,
                sheet_number=page_details.get('sheet_number') if page_details else None,
                scale=page_details.get('scale') if page_details else None,
                image_dimensions=page_details.get('image_dimensions', {}) if page_details else {}
            ))
        
        response = BatchPageDataResponse(
            document_id=clean_document_id,
            pages=pages,
            total_pages=total_pages
        )
        
        return JSONResponse(
            content=response.dict(),
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id,
                "X-Page-Count": str(len(pages))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch page data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve page data")

# --- IMAGE SERVING ENDPOINTS ---

@document_router.get("/documents/{document_id}/images/page-{page_number}_thumb.jpg")
async def get_page_thumbnail_direct(
    request: Request,
    document_id: str,
    page_number: int
):
    """Direct URL endpoint for page thumbnails with aggressive caching."""
    clean_document_id = validate_document_id(document_id)
    
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Invalid page number")
    
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        blob_name = f"{clean_document_id}_page_{page_number}_thumb.jpg"
        
        if not await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, blob_name):
            raise HTTPException(status_code=404, detail="Page thumbnail not found")
        
        image_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": CACHE_CONTROL_THUMBNAILS,
                "X-Document-ID": clean_document_id,
                "X-Page-Number": str(page_number),
                "X-Image-Type": "thumbnail",
                "Content-Length": str(len(image_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve thumbnail: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve thumbnail")

@document_router.get("/documents/{document_id}/images/page-{page_number}.jpg")
async def get_page_image_direct(
    request: Request,
    document_id: str,
    page_number: int
):
    """Direct URL endpoint for page images with aggressive caching."""
    clean_document_id = validate_document_id(document_id)
    
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Invalid page number")
    
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        blob_name = f"{clean_document_id}_page_{page_number}.jpg"
        
        if not await storage_service.blob_exists(settings.AZURE_CACHE_CONTAINER_NAME, blob_name):
            raise HTTPException(status_code=404, detail="Page image not found")
        
        image_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=604800, immutable",
                "X-Document-ID": clean_document_id,
                "X-Page-Number": str(page_number),
                "X-Image-Type": "high_quality",
                "Content-Length": str(len(image_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")

@document_router.get("/documents/{document_id}/image-urls",
                    summary="Get all page image URLs for a document")
async def get_all_image_urls(request: Request, document_id: str):
    """Get URLs for all page images and thumbnails in a document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        metadata = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        
        page_count = metadata.get('page_count', 0)
        base_url = f"https://{storage_service.blob_service_client.account_name}.blob.core.windows.net/{settings.AZURE_CACHE_CONTAINER_NAME}"
        
        image_urls = {}
        for page_num in range(1, page_count + 1):
            thumbnail_blob = f"{clean_document_id}_page_{page_num}_thumb.jpg"
            image_blob = f"{clean_document_id}_page_{page_num}.jpg"
            
            image_urls[str(page_num)] = {
                "thumbnail": {
                    "direct_url": f"{base_url}/{thumbnail_blob}",
                    "api_url": f"/api/v1/documents/{clean_document_id}/images/page-{page_num}_thumb.jpg",
                    "blob_name": thumbnail_blob
                },
                "high_quality": {
                    "direct_url": f"{base_url}/{image_blob}",
                    "api_url": f"/api/v1/documents/{clean_document_id}/images/page-{page_num}.jpg",
                    "blob_name": image_blob
                }
            }
        
        # Get image settings from metadata
        image_settings = metadata.get('image_settings', {
            'high_quality': {'dpi': 150, 'quality': 90, 'progressive': True},
            'thumbnail': {'dpi': 72, 'quality': 75}
        })
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "total_pages": page_count,
                "image_urls": image_urls,
                "image_settings": image_settings
            },
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image URLs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve image URLs")

@document_router.get("/documents/{document_id}/thumbnails",
                    summary="Get all thumbnail URLs for quick gallery view")
async def get_all_thumbnails(request: Request, document_id: str):
    """Get URLs for all page thumbnails - optimized for gallery/preview views."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        metadata = await storage_service.download_blob_as_json(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_metadata.json"
        )
        
        page_count = metadata.get('page_count', 0)
        base_url = f"https://{storage_service.blob_service_client.account_name}.blob.core.windows.net/{settings.AZURE_CACHE_CONTAINER_NAME}"
        
        thumbnails = []
        for page_num in range(1, page_count + 1):
            page_details = None
            for detail in metadata.get('page_details', []):
                if detail.get('page_number') == page_num:
                    page_details = detail
                    break
            
            thumbnail_blob = f"{clean_document_id}_page_{page_num}_thumb.jpg"
            
            thumbnails.append({
                "page": page_num,
                "url": f"{base_url}/{thumbnail_blob}",
                "api_url": f"/api/v1/documents/{clean_document_id}/images/page-{page_num}_thumb.jpg",
                "sheet_number": page_details.get('sheet_number') if page_details else None,
                "drawing_type": page_details.get('drawing_type') if page_details else None,
                "has_grid": page_details.get('has_grid', False) if page_details else False,
                "has_tables": page_details.get('has_tables', False) if page_details else False
            })
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "total_pages": page_count,
                "thumbnails": thumbnails,
                "thumbnail_settings": metadata.get('image_settings', {}).get('thumbnail', {
                    'dpi': 72,
                    'quality': 75
                })
            },
            headers={
                "Cache-Control": CACHE_CONTROL_METADATA,
                "X-Document-ID": clean_document_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thumbnails: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve thumbnails")

# --- ANALYTICS & COLLABORATION ---

@document_router.get("/documents/{document_id}/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(request: Request, document_id: str):
    """Get comprehensive statistics for a specific document."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        tasks = {
            'metadata': storage_service.download_blob_as_json(
                settings.AZURE_CACHE_CONTAINER_NAME, 
                f"{clean_document_id}_metadata.json"
            ),
            'notes': storage_service.download_blob_as_json(
                settings.AZURE_CACHE_CONTAINER_NAME, 
                f"{clean_document_id}_annotations.json"
            ),
            'chats': storage_service.download_blob_as_json(
                settings.AZURE_CACHE_CONTAINER_NAME, 
                f"{clean_document_id}_all_chats.json"
            ),
            'index': storage_service.download_blob_as_json(
                settings.AZURE_CACHE_CONTAINER_NAME,
                f"{clean_document_id}_document_index.json"
            )
        }
        
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except FileNotFoundError:
                if key == 'metadata':
                    raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found.")
                results[key] = [] if key in ['notes', 'chats'] else {}
            except Exception:
                results[key] = [] if key in ['notes', 'chats'] else {}

        metadata = results['metadata']
        annotations = results['notes']
        chats = results['chats']
        index = results['index']
        
        collaborators = {ann.get("author") for ann in annotations if ann.get("author")}
        collaborators.update({chat.get("author") for chat in chats if chat.get("author")})
        
        all_timestamps = [
            item.get("timestamp") for item in annotations + chats 
            if item.get("timestamp")
        ]
        
        grid_pages = index.get('grid_pages', [])
        table_pages = index.get('table_pages', [])
        
        return DocumentStatsResponse(
            document_id=clean_document_id,
            total_annotations=len(annotations),
            total_chats=len(chats),
            total_pages=metadata.get('page_count', 0),
            collaborator_count=len(collaborators),
            last_activity=max(all_timestamps) if all_timestamps else None,
            status=metadata.get('status', 'ready'),
            has_tables=metadata.get('extraction_summary', {}).get('has_tables', False),
            has_grids=len(grid_pages) > 0,
            has_thumbnails=metadata.get('extraction_summary', {}).get('has_thumbnails', True),
            grid_pages=grid_pages,
            table_pages=table_pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document statistics for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document statistics.")

@document_router.get("/documents/{document_id}/collaborators", response_model=CollaboratorsResponse)
async def get_document_collaborators(request: Request, document_id: str):
    """Get a list of all collaborators and their interaction statistics."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        tasks = {
            'notes': storage_service.download_blob_as_json(
                settings.AZURE_CACHE_CONTAINER_NAME, 
                f"{clean_document_id}_annotations.json"
            ),
            'chats': storage_service.download_blob_as_json(
                settings.AZURE_CACHE_CONTAINER_NAME, 
                f"{clean_document_id}_all_chats.json"
            )
        }
        
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except:
                results[key] = []
        
        annotations = results['notes']
        chats = results['chats']

        collaborator_data = {}
        
        for ann in annotations:
            author = ann.get("author")
            if author:
                if author not in collaborator_data:
                    collaborator_data[author] = {
                        'annotations': 0,
                        'chats': 0,
                        'last_activity': None
                    }
                collaborator_data[author]['annotations'] += 1
                
                timestamp = ann.get('timestamp')
                if timestamp:
                    if not collaborator_data[author]['last_activity'] or timestamp > collaborator_data[author]['last_activity']:
                        collaborator_data[author]['last_activity'] = timestamp
        
        for chat in chats:
            author = chat.get("author")
            if author:
                if author not in collaborator_data:
                    collaborator_data[author] = {
                        'annotations': 0,
                        'chats': 0,
                        'last_activity': None
                    }
                collaborator_data[author]['chats'] += 1
                
                timestamp = chat.get('timestamp')
                if timestamp:
                    if not collaborator_data[author]['last_activity'] or timestamp > collaborator_data[author]['last_activity']:
                        collaborator_data[author]['last_activity'] = timestamp

        collaborator_stats = []
        for author, data in collaborator_data.items():
            collaborator_stats.append(CollaboratorStats(
                author=author,
                annotations_count=data['annotations'],
                chats_count=data['chats'],
                total_interactions=data['annotations'] + data['chats'],
                last_activity=data['last_activity']
            ))

        collaborator_stats.sort(key=lambda x: x.total_interactions, reverse=True)

        return CollaboratorsResponse(
            document_id=clean_document_id,
            collaborators=collaborator_stats,
            total_collaborators=len(collaborator_stats)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collaborators for {clean_document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve collaborator data.")

# --- PREFETCH ENDPOINT ---

@document_router.post("/documents/{document_id}/prefetch",
                     summary="Prefetch document data for performance")
async def prefetch_document_data(
    request: Request,
    document_id: str,
    pages: Optional[List[int]] = None,
    include_context: bool = False,
    include_thumbnails: bool = True
):
    """Prefetch document data to warm up caches."""
    clean_document_id = validate_document_id(document_id)
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        # Check if document is ready
        if not await _check_document_ready(storage_service, clean_document_id):
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail="Document is still processing"
            )
        
        tasks = []
        
        tasks.append(storage_service.download_blob_as_json(
            settings.AZURE_CACHE_CONTAINER_NAME,
            f"{clean_document_id}_metadata.json"
        ))
        
        tasks.append(storage_service.download_blob_as_json(
            settings.AZURE_CACHE_CONTAINER_NAME,
            f"{clean_document_id}_grid_systems.json"
        ))
        
        tasks.append(storage_service.download_blob_as_json(
            settings.AZURE_CACHE_CONTAINER_NAME,
            f"{clean_document_id}_document_index.json"
        ))
        
        if include_context:
            tasks.append(storage_service.download_blob_as_text(
                settings.AZURE_CACHE_CONTAINER_NAME,
                f"{clean_document_id}_context.txt"
            ))
        
        if pages:
            for page_num in pages[:10]:
                if include_thumbnails:
                    thumbnail_blob = f"{clean_document_id}_page_{page_num}_thumb.jpg"
                    tasks.append(storage_service.blob_exists(
                        settings.AZURE_CACHE_CONTAINER_NAME,
                        thumbnail_blob
                    ))
                
                image_blob = f"{clean_document_id}_page_{page_num}.jpg"
                tasks.append(storage_service.blob_exists(
                    settings.AZURE_CACHE_CONTAINER_NAME,
                    image_blob
                ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "prefetched_items": successful,
                "total_requested": len(tasks),
                "status": "complete",
                "thumbnails_included": include_thumbnails
            },
            headers={"Cache-Control": CACHE_CONTROL_DYNAMIC}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prefetch failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "document_id": clean_document_id,
                "status": "partial",
                "error": str(e)
            },
            status_code=200
        )

# --- DOCUMENT SEARCH ---

@document_router.get("/documents/search",
                    summary="Search across documents")
async def search_documents(
    request: Request,
    q: str = Query(..., min_length=2, description="Search query"),
    document_ids: Optional[str] = Query(None, description="Comma-separated document IDs to search in"),
    search_in: str = Query("all", description="Search in: all, metadata, context, sheets")
):
    """Search for documents based on various criteria."""
    storage_service = await _get_service(request, 'storage_service')
    
    try:
        if document_ids:
            doc_list = [d.strip() for d in document_ids.split(',')]
        else:
            doc_list = await storage_service.list_document_ids(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME
            )
        
        search_results = []
        query_lower = q.lower()
        
        for doc_id in doc_list[:50]:
            # Only search ready documents
            if not await _check_document_ready(storage_service, doc_id):
                continue
            
            try:
                result = {
                    'document_id': doc_id,
                    'matches': []
                }
                
                if search_in in ['all', 'metadata']:
                    try:
                        metadata = await storage_service.download_blob_as_json(
                            settings.AZURE_CACHE_CONTAINER_NAME,
                            f"{doc_id}_metadata.json"
                        )
                        
                        doc_info = metadata.get('document_info', {})
                        for key, value in doc_info.items():
                            if query_lower in str(value).lower():
                                result['matches'].append({
                                    'type': 'metadata',
                                    'field': key,
                                    'value': str(value)[:100]
                                })
                    except:
                        pass
                
                if search_in in ['all', 'sheets']:
                    try:
                        index = await storage_service.download_blob_as_json(
                            settings.AZURE_CACHE_CONTAINER_NAME,
                            f"{doc_id}_document_index.json"
                        )
                        
                        for sheet_num, page in index.get('sheet_numbers', {}).items():
                            if query_lower in sheet_num.lower():
                                result['matches'].append({
                                    'type': 'sheet',
                                    'sheet_number': sheet_num,
                                    'page': page
                                })
                    except:
                        pass
                
                if result['matches']:
                    search_results.append(result)
                    
            except Exception as e:
                logger.debug(f"Error searching document {doc_id}: {e}")
        
        return JSONResponse(
            content={
                "query": q,
                "search_in": search_in,
                "total_results": len(search_results),
                "results": search_results[:20]
            },
            headers={"Cache-Control": CACHE_CONTROL_DYNAMIC}
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search operation failed")