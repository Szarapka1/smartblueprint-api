# app/api/routes/blueprint_routes.py - COMPLETE FIXED VERSION

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import json
import uuid
import re
import os
import logging
from app.core.config import get_settings

logger = logging.getLogger(__name__)
blueprint_router = APIRouter()
settings = get_settings()

# --- Pydantic Models ---

class DocumentChatRequest(BaseModel):
    document_id: str = Field(..., min_length=3, max_length=50)
    prompt: str = Field(..., min_length=1, max_length=2000)
    author: str = Field(..., min_length=1, max_length=100)

class DocumentChatResponse(BaseModel):
    document_id: str
    ai_response: str
    author: str
    chat_id: str
    timestamp: str
    source_pages: List[int] = []

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    file_size_mb: float

class DocumentInfoResponse(BaseModel):
    document_id: str
    status: str
    message: str
    exists: bool

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, str]]
    total_count: int

class AnnotationCreate(BaseModel):
    document_id: str
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str
    author: str
    is_private: bool = True

class Annotation(BaseModel):
    annotation_id: str
    document_id: str
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str
    author: str
    is_private: bool
    timestamp: str

# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID"""
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

def validate_author(author: str) -> str:
    """Validate and sanitize author name"""
    if not author or not isinstance(author, str):
        raise HTTPException(
            status_code=400,
            detail="Author name must be a non-empty string"
        )
    
    clean_author = author.strip()
    
    if not clean_author:
        raise HTTPException(status_code=400, detail="Author name cannot be empty")
    
    if len(clean_author) > 100:
        raise HTTPException(status_code=400, detail="Author name must be 100 characters or less")
    
    return clean_author

def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400, 
            detail="File must have a valid filename"
        )
    
    logger.info(f"File validation passed: {file.filename}, type: {file.content_type}")

# --- Chat History Management ---

async def load_user_chat_history(document_id: str, author: str, storage_service) -> List[dict]:
    """Load a specific user's private chat history"""
    try:
        safe_author = re.sub(r'[^a-zA-Z0-9_-]', '_', author.lower().replace(' ', '_'))
        chat_blob_name = f"{document_id}_chat_{safe_author}.json"
        
        chat_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=chat_blob_name
        )
        chats = json.loads(chat_data)
        logger.info(f"Loaded {len(chats)} chat messages for {author} in {document_id}")
        return chats
    except Exception as e:
        logger.info(f"No existing chat history for {author} in {document_id}: {e}")
        return []

async def save_user_chat_history(document_id: str, author: str, chat_history: List[dict], storage_service):
    """Save a user's private chat history"""
    try:
        safe_author = re.sub(r'[^a-zA-Z0-9_-]', '_', author.lower().replace(' ', '_'))
        chat_blob_name = f"{document_id}_chat_{safe_author}.json"
        chat_json = json.dumps(chat_history, indent=2, ensure_ascii=False)
        
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=chat_blob_name,
            data=chat_json.encode('utf-8')
        )
        logger.info(f"Saved {len(chat_history)} chat messages for {author} in {document_id}")
    except Exception as e:
        logger.error(f"Failed to save chat history for {author} in {document_id}: {e}")
        raise

async def log_all_chat_activity(document_id: str, author: str, prompt: str, ai_response: str, storage_service):
    """Log ALL chat activity for service analytics"""
    try:
        activity_blob_name = f"{document_id}_all_chats.json"
        
        # Load existing activity
        try:
            activity_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=activity_blob_name
            )
            all_chats = json.loads(activity_data)
        except Exception:
            all_chats = []
        
        # Add new activity
        all_chats.append({
            "chat_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "document_id": document_id,
            "author": author,
            "prompt": prompt,
            "ai_response": ai_response,
            "prompt_length": len(prompt),
            "response_length": len(ai_response)
        })
        
        # Limit size
        max_chats = int(os.getenv("MAX_CHAT_LOGS", "1000"))
        if len(all_chats) > max_chats:
            all_chats = all_chats[-max_chats:]
        
        # Save back
        activity_json = json.dumps(all_chats, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name,
            data=activity_json.encode('utf-8')
        )
        logger.info(f"Logged chat activity for {author} in {document_id}")
    except Exception as e:
        logger.error(f"Failed to log chat activity: {e}")

# --- Annotation Management ---

async def load_annotations(document_id: str, storage_service) -> List[dict]:
    """Load all annotations for a document"""
    try:
        blob_name = f"{document_id}_annotations.json"
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        annotations = json.loads(annotations_data)
        logger.info(f"Loaded {len(annotations)} annotations for {document_id}")
        return annotations
    except Exception:
        logger.info(f"No annotations found for {document_id}")
        return []

async def save_annotations(document_id: str, annotations: List[dict], storage_service):
    """Save all annotations for a document"""
    try:
        blob_name = f"{document_id}_annotations.json"
        annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name,
            data=annotations_json.encode('utf-8')
        )
        logger.info(f"Saved {len(annotations)} annotations for {document_id}")
    except Exception as e:
        logger.error(f"Failed to save annotations for {document_id}: {e}")
        raise

# --- MAIN API ROUTES ---

@blueprint_router.post("/documents/upload", status_code=201, response_model=DocumentUploadResponse)
async def upload_shared_document(
    request: Request,
    document_id: str = Query(..., description="Custom document ID"),
    force_reprocess: bool = Query(False, description="Force reprocessing"),
    file: UploadFile = File(...)
):
    """Upload and process a PDF document"""
    try:
        logger.info(f"Upload request: id={document_id}, force={force_reprocess}, file={file.filename}")
        validate_file_upload(file)
        
        # Check file size
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "60")) * 1024 * 1024
        pdf_bytes = await file.read()
        if len(pdf_bytes) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size is {max_size // (1024*1024)}MB"
            )

        clean_document_id = validate_document_id(document_id)
        
        # Check services
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service
        
        if not pdf_service or not storage_service:
            raise HTTPException(
                status_code=503, 
                detail="Required services are not available"
            )

        # Check if document exists
        document_exists = False
        if not force_reprocess:
            try:
                await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{clean_document_id}_context.txt"
                )
                document_exists = True
                logger.info(f"Document {clean_document_id} already exists")
                
                return DocumentUploadResponse(
                    document_id=clean_document_id,
                    filename=file.filename,
                    status="already_exists",
                    message=f"Document '{clean_document_id}' already exists. Use force_reprocess=true to reprocess.",
                    file_size_mb=round(len(pdf_bytes) / (1024*1024), 2)
                )
            except Exception:
                logger.info(f"Document {clean_document_id} not found, proceeding with upload")

        # Store original PDF
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf",
            data=pdf_bytes
        )

        # Process PDF
        await pdf_service.process_and_cache_pdf(
            session_id=clean_document_id,
            pdf_bytes=pdf_bytes,
            storage_service=storage_service
        )

        status_message = "reprocessed" if (force_reprocess and document_exists) else "processing_complete"
        
        return DocumentUploadResponse(
            document_id=clean_document_id,
            filename=file.filename,
            status=status_message,
            message=f"Document '{clean_document_id}' uploaded and ready for use",
            file_size_mb=round(len(pdf_bytes) / (1024*1024), 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@blueprint_router.get("/documents/{document_id}/info", response_model=DocumentInfoResponse)
async def get_document_info(request: Request, document_id: str):
    """Get information about a specific document"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Check if document exists
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
            return DocumentInfoResponse(
                document_id=clean_document_id,
                status="ready",
                message="Document is ready for use",
                exists=True
            )
        except Exception:
            raise HTTPException(
                status_code=404, 
                detail=f"Document '{clean_document_id}' not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request):
    """List all available documents"""
    try:
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # List all context files to find documents
        blobs = await storage_service.list_blobs(settings.AZURE_CACHE_CONTAINER_NAME)
        context_files = [blob for blob in blobs if blob.endswith('_context.txt')]
        
        documents = []
        for context_file in context_files:
            doc_id = context_file.replace('_context.txt', '')
            documents.append({
                "document_id": doc_id,
                "status": "ready"
            })
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@blueprint_router.post("/documents/{document_id}/chat", response_model=DocumentChatResponse)
async def chat_with_shared_document(request: Request, document_id: str, chat: DocumentChatRequest):
    """Chat with AI about a document"""
    try:
        clean_path_id = validate_document_id(document_id)
        if clean_path_id != validate_document_id(chat.document_id):
            raise HTTPException(
                status_code=400, 
                detail="Document ID in URL and body must match"
            )
        
        clean_author = validate_author(chat.author)
        clean_prompt = chat.prompt.strip()
        if not clean_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Check services
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service
        if not ai_service or not storage_service:
            raise HTTPException(
                status_code=503, 
                detail="Required services are not available"
            )

        # Verify document exists
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_path_id}_context.txt"
            )
        except Exception:
            raise HTTPException(
                status_code=404, 
                detail=f"Document '{clean_path_id}' not found"
            )

        # Get AI response
        ai_response_text = await ai_service.get_ai_response(
            prompt=clean_prompt,
            document_id=clean_path_id,
            storage_service=storage_service,
            author=clean_author
        )
        
        chat_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Save to user's chat history
        chat_history = await load_user_chat_history(clean_path_id, clean_author, storage_service)
        chat_history.append({
            "chat_id": chat_id,
            "timestamp": timestamp,
            "prompt": clean_prompt,
            "ai_response": ai_response_text
        })
        
        # Limit history size
        max_chats = int(os.getenv("MAX_USER_CHAT_HISTORY", "100"))
        await save_user_chat_history(
            clean_path_id, 
            clean_author, 
            chat_history[-max_chats:], 
            storage_service
        )
        
        # Log for analytics
        await log_all_chat_activity(
            clean_path_id, 
            clean_author, 
            clean_prompt, 
            ai_response_text, 
            storage_service
        )

        return DocumentChatResponse(
            document_id=clean_path_id,
            ai_response=ai_response_text,
            author=clean_author,
            chat_id=chat_id,
            timestamp=timestamp,
            source_pages=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

@blueprint_router.get("/documents/{document_id}/my-chats")
async def get_my_chat_history(
    request: Request, 
    document_id: str, 
    author: str = Query(...), 
    limit: int = Query(20, ge=1, le=100)
):
    """Get user's private chat history for a document"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        chat_history = await load_user_chat_history(clean_document_id, clean_author, storage_service)
        recent_chats = chat_history[-limit:]
        
        return {
            "document_id": clean_document_id,
            "author": clean_author,
            "chat_history": list(reversed(recent_chats)),
            "total_conversations": len(chat_history),
            "showing": len(recent_chats)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat history for {author}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

# --- ANNOTATION ROUTES ---

@blueprint_router.post("/documents/{document_id}/annotations", response_model=Annotation)
async def create_annotation(request: Request, document_id: str, annotation_data: AnnotationCreate):
    """Create a new annotation"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(annotation_data.author)
        
        # Validate annotation data
        if not annotation_data.text.strip() or len(annotation_data.text.strip()) > 1000:
            raise HTTPException(
                status_code=400, 
                detail="Annotation text must be between 1 and 1000 characters"
            )
        if annotation_data.page_number < 1:
            raise HTTPException(
                status_code=400, 
                detail="Page number must be positive"
            )
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Load existing annotations
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Create new annotation
        new_annotation = Annotation(
            annotation_id=str(uuid.uuid4())[:8],
            document_id=clean_document_id,
            page_number=annotation_data.page_number,
            x=annotation_data.x,
            y=annotation_data.y,
            text=annotation_data.text.strip(),
            annotation_type=annotation_data.annotation_type,
            author=clean_author,
            is_private=annotation_data.is_private,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        all_annotations.append(new_annotation.dict())
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        return new_annotation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create annotation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@blueprint_router.get("/documents/{document_id}/annotations")
async def get_user_visible_annotations(
    request: Request, 
    document_id: str, 
    author: str = Query(...)
):
    """Get annotations visible to the user (their private + all public)"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Filter to visible annotations
        visible_annotations = [
            ann for ann in all_annotations 
            if not ann.get("is_private", True) or ann.get("author") == clean_author
        ]
        
        return {"annotations": visible_annotations}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get annotations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get annotations: {str(e)}")

@blueprint_router.delete("/documents/{document_id}/annotations/{annotation_id}")
async def delete_annotation(
    request: Request, 
    document_id: str, 
    annotation_id: str, 
    author: str = Query(...)
):
    """Delete user's own annotation"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Find annotation to delete
        annotation_to_delete = next(
            (ann for ann in all_annotations if ann.get('annotation_id') == annotation_id), 
            None
        )
        
        if not annotation_to_delete:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        if annotation_to_delete.get('author') != clean_author:
            raise HTTPException(
                status_code=403, 
                detail="You can only delete your own annotations"
            )
            
        # Remove annotation
        updated_annotations = [
            ann for ann in all_annotations 
            if ann.get('annotation_id') != annotation_id
        ]
        
        await save_annotations(clean_document_id, updated_annotations, storage_service)
        
        return {"status": "deleted", "deleted_id": annotation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete annotation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete annotation: {str(e)}")

@blueprint_router.post("/documents/{document_id}/annotations/{annotation_id}/publish")
async def publish_annotation(
    request: Request, 
    document_id: str, 
    annotation_id: str, 
    author: str = Query(...)
):
    """Make a private annotation public"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Find and update annotation
        updated = False
        for annotation in all_annotations:
            if (annotation.get("annotation_id") == annotation_id and 
                annotation.get("author") == clean_author):
                
                if not annotation.get("is_private", True):
                    raise HTTPException(
                        status_code=400, 
                        detail="Annotation is already public"
                    )
                
                annotation["is_private"] = False
                annotation["published_at"] = datetime.utcnow().isoformat() + "Z"
                updated = True
                break
        
        if not updated:
            raise HTTPException(
                status_code=404, 
                detail="Private annotation not found or not owned by you"
            )
        
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        return {"status": "published", "annotation_id": annotation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish annotation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish annotation: {str(e)}")
