# app/api/routes/blueprint_routes.py - Complete Fixed Version

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header, Query
from pydantic import BaseModel, Field
from typing import Optional, List
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

class Annotation(BaseModel):
    annotation_id: str
    document_id: str
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str  # e.g., 'note', 'highlight'
    author: str
    is_private: bool
    timestamp: str

class AnnotationCreate(BaseModel):
    document_id: str
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str
    author: str
    is_private: bool = True

class AnnotationDeleteResponse(BaseModel):
    status: str
    deleted_id: str

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    file_size_mb: float


# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID for shared use, ensuring no trailing underscores."""
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be a non-empty string"
        )
    
    # Replace non-allowed characters with underscores
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    
    # Remove any trailing underscores that might result from sanitization or initial input
    clean_id = clean_id.strip('_') 

    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters and contain only letters, numbers, underscores, or hyphens"
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
        raise HTTPException(
            status_code=400,
            detail="Author name cannot be empty"
        )
    
    if len(clean_author) > 100:
        raise HTTPException(
            status_code=400,
            detail="Author name must be 100 characters or less"
        )
    
    return clean_author

def validate_admin_access(admin_token: str) -> bool:
    """Validate admin access using environment variable"""
    expected_token = os.getenv("ADMIN_SECRET_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Admin access not configured"
        )
    return admin_token == expected_token

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


# --- Chat History Helpers ---

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


# --- Analytics Helpers ---

async def log_all_chat_activity(document_id: str, author: str, prompt: str, ai_response: str, storage_service):
    """SERVICE: Log ALL chat activity from ALL users for analytics"""
    try:
        activity_blob_name = f"{document_id}_all_chats.json"
        
        try:
            activity_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=activity_blob_name
            )
            all_chats = json.loads(activity_data)
        except Exception:
            all_chats = []
        
        chat_entry = {
            "chat_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "document_id": document_id,
            "author": author,
            "prompt": prompt,
            "ai_response": ai_response,
            "prompt_length": len(prompt),
            "response_length": len(ai_response)
        }
        
        all_chats.append(chat_entry)
        
        max_chats = int(os.getenv("MAX_CHAT_LOGS", "1000"))
        if len(all_chats) > max_chats:
            all_chats = all_chats[-max_chats:]
        
        activity_json = json.dumps(all_chats, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name,
            data=activity_json.encode('utf-8')
        )
        logger.info(f"Logged chat activity for {author} in {document_id}")
    except Exception as e:
        logger.error(f"Failed to log chat activity: {e}")


# --- Annotation Helpers ---

async def load_annotations(document_id: str, storage_service) -> List[dict]:
    """Load all annotations for a document."""
    try:
        blob_name = f"{document_id}_annotations.json"
        annotations_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        annotations = json.loads(annotations_data)
        logger.info(f"Loaded {len(annotations)} annotations for {document_id}")
        return annotations
    except Exception as e:
        logger.info(f"No existing annotations for {document_id}: {e}")
        return []

async def save_annotations(document_id: str, annotations: List[dict], storage_service):
    """Save all annotations for a document."""
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

async def filter_user_visible_annotations(annotations: List[dict], requesting_author: str) -> List[dict]:
    """Filter annotations to show only what the user should see"""
    visible = []
    for annotation in annotations:
        if not annotation.get("is_private", True):
            visible.append(annotation)
        elif annotation.get("author") == requesting_author:
            visible.append(annotation)
    
    logger.info(f"Filtered {len(visible)} visible annotations out of {len(annotations)} total for {requesting_author}")
    return visible


# --- API Routes ---

@blueprint_router.post("/documents/upload", status_code=201, response_model=DocumentUploadResponse)
async def upload_shared_document(
    request: Request,
    document_id: str = Query(..., description="Custom document ID"),
    force_reprocess: bool = Query(False, description="Force reprocessing even if document exists"),
    file: UploadFile = File(...)
):
    """
    Upload a PDF blueprint as a shared document that everyone can access.
    """
    try:
        logger.info(f"Upload request received: document_id={document_id}, force_reprocess={force_reprocess}, file={file.filename}")
        validate_file_upload(file)
        
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "60")) * 1024 * 1024
        pdf_bytes = await file.read()
        if len(pdf_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(pdf_bytes) / (1024*1024):.1f}MB. Maximum size is {max_size // (1024*1024)}MB"
            )

        clean_document_id = validate_document_id(document_id)
        
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service
        
        if not pdf_service:
            raise HTTPException(status_code=503, detail="PDF processing service not available")
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")

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
                logger.info(f"Document {clean_document_id} does not exist, proceeding with upload")

        logger.info(f"Uploading PDF to storage: {clean_document_id}.pdf")
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf",
            data=pdf_bytes
        )

        logger.info(f"Processing PDF for AI analysis: {clean_document_id}")
        await pdf_service.process_and_cache_pdf(
            session_id=clean_document_id, 
            pdf_bytes=pdf_bytes,
            storage_service=storage_service
        )

        status_message = "reprocessed" if (force_reprocess and document_exists) else "processing_complete"
        logger.info(f"Upload completed successfully: {clean_document_id}")
        
        return DocumentUploadResponse(
            document_id=clean_document_id,
            filename=file.filename,
            status=status_message,
            message=f"Document '{clean_document_id}' uploaded and ready for collaborative use",
            file_size_mb=round(len(pdf_bytes) / (1024*1024), 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@blueprint_router.post("/documents/{document_id}/chat", response_model=DocumentChatResponse)
async def chat_with_shared_document(
    request: Request,
    document_id: str,
    chat: DocumentChatRequest 
):
    """
    USER: Ask private questions about a shared document
    SERVICE: Store all conversations and provide analytics
    """
    try:
        logger.info(f"Chat request: {document_id} from {chat.author}")
        
        clean_path_document_id = validate_document_id(document_id)
        clean_chat_body_document_id = validate_document_id(chat.document_id)

        if clean_path_document_id != clean_chat_body_document_id:
            raise HTTPException(
                status_code=400, 
                detail="Document ID in URL must match Document ID in request body"
            )
        
        final_document_id = clean_path_document_id 
        
        if not chat.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        clean_author = validate_author(chat.author)
        clean_prompt = chat.prompt.strip()
        
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service

        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")

        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{final_document_id}_context.txt"
            )
        except Exception: 
            raise HTTPException(
                status_code=404,
                detail=f"Document '{final_document_id}' not found. Please upload it first."
            )

        logger.info(f"Getting AI response for: {clean_prompt[:50]}...")
        ai_response_text = await ai_service.get_ai_response(
            prompt=clean_prompt,
            document_id=final_document_id,
            storage_service=storage_service,
            author=clean_author
        )

        chat_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        user_chat_history = await load_user_chat_history(final_document_id, clean_author, storage_service)
        
        chat_entry = {
            "chat_id": chat_id,
            "timestamp": timestamp,
            "prompt": clean_prompt,
            "ai_response": ai_response_text,
            "page_number": None
        }
        
        user_chat_history.append(chat_entry)
        
        max_user_chats = int(os.getenv("MAX_USER_CHAT_HISTORY", "100"))
        if len(user_chat_history) > max_user_chats:
            user_chat_history = user_chat_history[-max_user_chats:]
        
        await save_user_chat_history(final_document_id, clean_author, user_chat_history, storage_service)

        await log_all_chat_activity(
            document_id=final_document_id,
            author=clean_author,
            prompt=clean_prompt,
            ai_response=ai_response_text,
            storage_service=storage_service
        )

        logger.info(f"Chat completed successfully for {clean_author}")
        
        return DocumentChatResponse(
            document_id=final_document_id,
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
    author: str = Query(..., description="Author name"),
    limit: int = Query(20, ge=1, le=100, description="Number of recent chats to return")
):
    """
    USER: Get their own private chat history for a specific document.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # FIX: Correct logic to load chat history
        chat_history = await load_user_chat_history(clean_document_id, clean_author, storage_service)
        
        recent_chats = chat_history[-limit:]
        
        logger.info(f"Retrieved {len(recent_chats)} chat messages for {clean_author}")
        
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

# FIX: Added back the get_document_info endpoint
@blueprint_router.get("/documents/{document_id}/info")
async def get_document_info(
    request: Request,
    document_id: str
):
    """
    Get information about a shared document including processing status.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service

        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Check for the existence of the context file as a proxy for 'ready' status
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
            status = "ready"
            error_message = None
        except Exception:
            status = "not_found"
            error_message = f"Document '{clean_document_id}' has not been processed or does not exist."

        if status == "not_found":
            raise HTTPException(status_code=404, detail=error_message)

        # You can add more details here if needed, like fetching metadata
        return {
            "document_id": clean_document_id,
            "status": status,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents")
async def list_shared_documents(request: Request):
    """
    List all available shared documents in the system.
    """
    try:
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        blobs = await storage_service.list_blobs(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        
        documents = []
        processed_ids = set()

        for blob_name in blobs:
            if '_context.txt' in blob_name:
                doc_id = blob_name.replace('_context.txt', '')
                if doc_id in processed_ids:
                    continue
                
                try:
                    clean_document_id = validate_document_id(doc_id)
                    documents.append({
                        "document_id": clean_document_id,
                        "status": "ready"
                    })
                    processed_ids.add(doc_id)
                except HTTPException:
                    logger.warning(f"Skipping invalid document ID found in storage: {doc_id}")
                    continue
        
        logger.info(f"Listed {len(documents)} available documents")
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


# --- ANNOTATION ENDPOINTS ---

@blueprint_router.post("/documents/{document_id}/annotations", response_model=Annotation)
async def create_annotation(
    request: Request,
    document_id: str,
    annotation_data: AnnotationCreate
):
    """Create a new annotation (note or highlight) on a document."""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(annotation_data.author)
        
        if not annotation_data.text.strip():
            raise HTTPException(status_code=400, detail="Annotation text cannot be empty")
        if annotation_data.page_number < 1:
            raise HTTPException(status_code=400, detail="Page number must be greater than 0")
        if len(annotation_data.text.strip()) > 1000:
            raise HTTPException(status_code=400, detail="Annotation text too long (max 1000 characters)")
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
        except Exception:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        new_annotation = Annotation(
            annotation_id=str(uuid.uuid4())[:8],
            document_id=clean_document_id,
            page_number=annotation_data.page_number,
            x=annotation_data.x,
            y=annotation_data.y,
            text=annotation_data.text.strip(),
            annotation_type=annotation_data.annotation_type,
            author=clean_auth
