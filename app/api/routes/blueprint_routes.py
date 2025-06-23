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
    # Check file type
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    # Check filename
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400,
            detail="File must have a valid filename"
        )
    
    # Additional file validation can be added here
    logger.info(f"File validation passed: {file.filename}, type: {file.content_type}")


# --- Chat History Helpers ---

async def load_user_chat_history(document_id: str, author: str, storage_service) -> List[dict]:
    """Load a specific user's private chat history"""
    try:
        # Sanitize author name for filename
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
        
        # Keep configurable number of chats for service analytics
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
        # Don't raise - this is analytics, not critical


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
        # Show public annotations to everyone
        if not annotation.get("is_private", True):
            visible.append(annotation)
        # Show private annotations only to their author
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
    FIXED: Now supports force reprocessing for fresh AI analysis.
    """
    try:
        logger.info(f"Upload request received: document_id={document_id}, force_reprocess={force_reprocess}, file={file.filename}")
        
        # Validate file
        validate_file_upload(file)
        
        # Validate file size
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "60")) * 1024 * 1024
        pdf_bytes = await file.read()
        if len(pdf_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(pdf_bytes) / (1024*1024):.1f}MB. Maximum size is {max_size // (1024*1024)}MB"
            )

        # Validate and clean the document ID
        clean_document_id = validate_document_id(document_id)
        
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service
        
        if not pdf_service:
            raise HTTPException(status_code=503, detail="PDF processing service not available")
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Check if document already exists (only if not forcing reprocess)
        document_exists = False
        if not force_reprocess:
            try:
                existing_context = await storage_service.download_blob_as_text(
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

        # Upload original PDF to main container (always overwrite if force_reprocess=True)
        logger.info(f"Uploading PDF to storage: {clean_document_id}.pdf")
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf",
            data=pdf_bytes
        )

        # Process PDF for AI analysis (this will overwrite existing processed data)
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
    FIXED: Proper AI service integration
    """
    try:
        logger.info(f"Chat request: {document_id} from {chat.author}")
        
        # Validate document ID consistency
        clean_path_document_id = validate_document_id(document_id)
        clean_chat_body_document_id = validate_document_id(chat.document_id)

        if clean_path_document_id != clean_chat_body_document_id:
            raise HTTPException(
                status_code=400, 
                detail="Document ID in URL must match Document ID in request body"
            )
        
        final_document_id = clean_path_document_id 
        
        # Validate input
        if not chat.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        clean_author = validate_author(chat.author)
        clean_prompt = chat.prompt.strip()
        
        # Get services
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service

        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Verify document exists
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

        # FIXED: Get AI response using the correct method name
        logger.info(f"Getting AI response for: {clean_prompt[:50]}...")
        ai_response_text = await ai_service.get_ai_response(
            prompt=clean_prompt,
            document_id=final_document_id,
            storage_service=storage_service,
            author=clean_author
        )

        # Generate unique chat ID and timestamp
        chat_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        # USER: Save to their private chat history
        user_chat_history = await load_user_chat_history(final_document_id, clean_author, storage_service)
        
        chat_entry = {
            "chat_id": chat_id,
            "timestamp": timestamp,
            "prompt": clean_prompt,
            "ai_response": ai_response_text,
            "page_number": None  # AI determines this via tools
        }
        
        user_chat_history.append(chat_entry)
        
        # Keep configurable number of conversations per user
        max_user_chats = int(os.getenv("MAX_USER_CHAT_HISTORY", "100"))
        if len(user_chat_history) > max_user_chats:
            user_chat_history = user_chat_history[-max_user_chats:]
        
        await save_user_chat_history(final_document_id, clean_author, user_chat_history, storage_service)

        # SERVICE: Log everything for analytics
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
            source_pages=[]  # AI can determine this via tools if needed
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
    USER: Get their own private chat history
    Each user only sees their own conversations
    """
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        annotation_to_publish = None
        annotation_index = None
        
        for i, annotation in enumerate(all_annotations):
            if annotation.get("annotation_id") == clean_annotation_id:
                annotation_to_publish = annotation
                annotation_index = i
                break
        
        if not annotation_to_publish:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        if annotation_to_publish.get("author") != clean_author:
            raise HTTPException(status_code=403, detail="You can only publish your own annotations")
        
        if not annotation_to_publish.get("is_private", True):
            raise HTTPException(status_code=400, detail="Annotation is already public")
        
        # Publish it
        current_time = datetime.utcnow().isoformat() + "Z"
        annotation_to_publish["is_private"] = False
        annotation_to_publish["published_at"] = current_time
        
        all_annotations[annotation_index] = annotation_to_publish
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        logger.info(f"Published annotation {clean_annotation_id} by {clean_author}")
        
        return {
            "message": "Annotation published successfully",
            "annotation_id": clean_annotation_id,
            "published_at": current_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish annotation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish annotation: {str(e)}")

# --- ADMIN ENDPOINTS ---

@blueprint_router.get("/admin/documents/{document_id}/all-chats")
async def admin_get_all_chats(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    SERVICE ADMIN: Get ALL chat conversations from ALL users
    Use this for analytics, understanding usage patterns, etc.
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get ALL chats from ALL users
        activity_blob_name = f"{clean_document_id}_all_chats.json"
        try:
            activity_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=activity_blob_name
            )
            all_chats = json.loads(activity_data)
        except Exception: 
            all_chats = []
        
        # Analytics
        analytics = {
            "total_conversations": len(all_chats),
            "unique_users": len(set(chat.get("author") for chat in all_chats)),
            "avg_prompt_length": sum(chat.get("prompt_length", 0) for chat in all_chats) / len(all_chats) if all_chats else 0,
            "avg_response_length": sum(chat.get("response_length", 0) for chat in all_chats) / len(all_chats) if all_chats else 0
        }
        
        logger.info(f"Admin retrieved {len(all_chats)} chats for {clean_document_id}")
        
        return {
            "document_id": clean_document_id,
            "all_chats": all_chats, 
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin chat access failed: {e}")
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@blueprint_router.get("/admin/documents/{document_id}/user-chat/{author}")
async def admin_get_user_chat(
    request: Request,
    document_id: str,
    author: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    SERVICE ADMIN: Get specific user's complete chat history
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get specific user's chat history
        user_chats = await load_user_chat_history(clean_document_id, clean_author, storage_service)
        
        logger.info(f"Admin retrieved {len(user_chats)} chats for {clean_author}")
        
        return {
            "document_id": clean_document_id,
            "author": clean_author,
            "chat_history": user_chats,
            "total_conversations": len(user_chats)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin user chat access failed: {e}")
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@blueprint_router.get("/admin/documents/{document_id}/all-annotations")
async def admin_get_all_annotations(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    SERVICE ADMIN: Get ALL annotations from ALL users (private + public)
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get ALL annotations from ALL users
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
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
        
        logger.info(f"Admin retrieved {len(all_annotations)} annotations for {clean_document_id}")
        
        return {
            "document_id": clean_document_id,
            "all_annotations": all_annotations,
            "analytics": analytics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin annotation access failed: {e}")
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@blueprint_router.get("/admin/system/stats")
async def admin_get_system_stats(
    request: Request,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    SERVICE ADMIN: Get overall system statistics
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get basic storage stats
        main_blobs = await storage_service.list_blobs(container_name=settings.AZURE_CONTAINER_NAME)
        cache_blobs = await storage_service.list_blobs(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        
        # Count document types
        pdf_documents = len([b for b in main_blobs if b.endswith('.pdf')])
        processed_documents = len([b for b in cache_blobs if b.endswith('_chunks.json')])
        
        # Count different file types in cache
        cache_file_types = {}
        for blob in cache_blobs:
            if '_' in blob:
                file_type = blob.split('_')[-1].split('.')[0]
                cache_file_types[file_type] = cache_file_types.get(file_type, 0) + 1
        
        logger.info(f"Admin retrieved system stats: {pdf_documents} PDFs, {processed_documents} processed")
        
        return {
            "system_stats": {
                "total_pdf_documents": pdf_documents,
                "processed_documents": processed_documents,
                "total_cache_files": len(cache_blobs),
                "total_main_files": len(main_blobs),
                "cache_file_breakdown": cache_file_types
            },
            "service_status": {
                "pdf_service": bool(request.app.state.pdf_service),
                "ai_service": bool(request.app.state.ai_service),
                "storage_service": bool(request.app.state.storage_service),
                "session_service": bool(request.app.state.session_service)
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin system stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

# --- UTILITY ENDPOINTS ---

@blueprint_router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    author: str = Query(..., description="Author name requesting deletion"),
    confirm: bool = Query(False, description="Confirm deletion")
):
    """
    Delete entire document and all associated data.
    WARNING: This affects all users who had access to this document.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion by setting confirm=true"
            )
        
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # List of blobs to delete
        blobs_to_delete = [
            f"{clean_document_id}.pdf",  # Main PDF
            f"{clean_document_id}_context.txt",  # Text content
            f"{clean_document_id}_chunks.json",  # Processed chunks
            f"{clean_document_id}_annotations.json",  # Annotations
            f"{clean_document_id}_all_chats.json",  # All chat logs
        ]
        
        # Delete from main container
        try:
            await storage_service.delete_blob(
                container_name=settings.AZURE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}.pdf"
            )
            logger.info(f"Deleted main PDF: {clean_document_id}.pdf")
        except Exception as e:
            logger.warning(f"Could not delete main PDF: {e}")
        
        # Delete from cache container
        deleted_count = 0
        for blob_name in blobs_to_delete[1:]:  # Skip the PDF (already handled)
            try:
                await storage_service.delete_blob(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
                deleted_count += 1
                logger.info(f"Deleted cache file: {blob_name}")
            except Exception as e:
                logger.warning(f"Could not delete {blob_name}: {e}")
        
        # Delete any remaining document-related blobs
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
                    logger.info(f"Deleted additional file: {blob}")
                except Exception as e:
                    logger.warning(f"Could not delete {blob}: {e}")
        except Exception as e:
            logger.warning(f"Could not list blobs for cleanup: {e}")
        
        logger.info(f"Document {clean_document_id} deleted by {clean_author}, {deleted_count} files removed")
        
        return {
            "message": f"Document '{clean_document_id}' deleted successfully",
            "document_id": clean_document_id,
            "deleted_by": clean_author,
            "files_deleted": deleted_count,
            "warning": "This action affects all users who had access to this document",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@blueprint_router.get("/documents/{document_id}/stats")
async def get_document_statistics(
    request: Request,
    document_id: str
):
    """
    Get comprehensive statistics about a document.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get document info
        try:
            context_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
        except Exception:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")
        
        # Get annotations
        annotations = await load_annotations(clean_document_id, storage_service)
        
        # Get all chats
        all_chats = []
        try:
            activity_blob_name = f"{clean_document_id}_all_chats.json"
            activity_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=activity_blob_name
            )
            all_chats = json.loads(activity_data)
        except Exception:
            pass
        
        # Calculate statistics
        unique_authors = set()
        for ann in annotations:
            if ann.get("author"):
                unique_authors.add(ann.get("author"))
        for chat in all_chats:
            if chat.get("author"):
                unique_authors.add(chat.get("author"))
        
        annotation_types = {}
        for ann in annotations:
            ann_type = ann.get("annotation_type", "unknown")
            annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
        
        page_numbers = [ann.get("page_number", 1) for ann in annotations if ann.get("page_number")]
        max_page = max(page_numbers) if page_numbers else 1
        
        last_activity = None
        all_timestamps = []
        for ann in annotations:
            if ann.get("timestamp"):
                all_timestamps.append(ann.get("timestamp"))
        for chat in all_chats:
            if chat.get("timestamp"):
                all_timestamps.append(chat.get("timestamp"))
        
        if all_timestamps:
            last_activity = max(all_timestamps)
        
        stats = {
            "document_id": clean_document_id,
            "total_annotations": len(annotations),
            "total_chats": len(all_chats),
            "total_pages": max_page,
            "total_characters": len(context_text),
            "estimated_tokens": len(context_text) // 4,
            "unique_collaborators": list(unique_authors),
            "collaborator_count": len(unique_authors),
            "annotation_types": annotation_types,
            "last_activity": last_activity,
            "status": "active" if (annotations or all_chats) else "inactive",
            "created": datetime.utcnow().isoformat() + "Z"
        }
        
        logger.info(f"Generated stats for {clean_document_id}: {len(annotations)} annotations, {len(all_chats)} chats")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")# Load user's private chat history
        chat_history = await load_user_chat_history(clean_document_id, clean_author, storage_service)
        
        # Return most recent chats
        recent_chats = chat_history[-limit:] if len(chat_history) > limit else chat_history
        
        logger.info(f"Retrieved {len(recent_chats)} chat messages for {clean_author}")
        
        return {
            "document_id": clean_document_id,
            "author": clean_author,
            "chat_history": list(reversed(recent_chats)),  # Most recent first
            "total_conversations": len(chat_history),
            "showing": len(recent_chats)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@blueprint_router.get("/documents/{document_id}/info")
async def get_document_info(
    request: Request,
    document_id: str
):
    """
    Get information about a shared document including processing status and stats.
    FIXED: Proper AI service integration
    """
    try:
        clean_document_id = validate_document_id(document_id)
        
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service

        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # FIXED: Use the correct method name
        document_info = await ai_service.get_document_info(
            document_id=clean_document_id,
            storage_service=storage_service
        )

        logger.info(f"Retrieved document info for {clean_document_id}: {document_info.get('status', 'unknown')}")
        
        return document_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents")
async def list_shared_documents(request: Request):
    """
    List all available shared documents in the system.
    """
    try:
        storage_service = request.app.state.storage_service
        ai_service = request.app.state.ai_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # List all cached chunks files to find available documents
        blobs = await storage_service.list_blobs(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        
        documents = []
        for blob_name in blobs:
            if blob_name.endswith('_chunks.json'):
                # Extract document_id from blob name
                document_id = blob_name.replace('_chunks.json', '')
                
                try:
                    clean_document_id = validate_document_id(document_id)

                    # Get document info if AI service is available
                    if ai_service:
                        try:
                            doc_info = await ai_service.get_document_info(clean_document_id, storage_service)
                            documents.append(doc_info)
                        except Exception as e:
                            logger.warning(f"Could not get info for {clean_document_id}: {e}")
                            documents.append({
                                "document_id": clean_document_id,
                                "status": "available",
                                "warning": "Could not retrieve full info"
                            })
                    else:
                        documents.append({
                            "document_id": clean_document_id,
                            "status": "available",
                            "warning": "AI service not available"
                        })
                        
                except Exception as e:
                    logger.warning(f"Invalid document ID {document_id}: {e}")
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
        
        # Validate annotation data
        if not annotation_data.text.strip():
            raise HTTPException(status_code=400, detail="Annotation text cannot be empty")
        
        if annotation_data.page_number < 1:
            raise HTTPException(status_code=400, detail="Page number must be greater than 0")
        
        if len(annotation_data.text.strip()) > 1000:
            raise HTTPException(status_code=400, detail="Annotation text too long (max 1000 characters)")
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Verify document exists
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
            author=clean_author,
            is_private=annotation_data.is_private,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        all_annotations.append(new_annotation.dict())
        await save_annotations(clean_document_id, all_annotations, storage_service)
        
        logger.info(f"Created annotation {new_annotation.annotation_id} by {clean_author}")
        
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
    author: str = Query(..., description="Author name")
):
    """Get all annotations for a document that are visible to the current user."""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        # Filter to only show annotations the user should see
        visible_annotations = await filter_user_visible_annotations(all_annotations, clean_author)
        
        logger.info(f"Retrieved {len(visible_annotations)} visible annotations for {clean_author}")
        
        return {
            "document_id": clean_document_id,
            "annotations": visible_annotations,
            "visible_count": len(visible_annotations),
            "requesting_author": clean_author
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get annotations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get annotations: {str(e)}")

@blueprint_router.delete("/documents/{document_id}/annotations/{annotation_id}", response_model=AnnotationDeleteResponse)
async def delete_annotation(
    request: Request,
    document_id: str,
    annotation_id: str,
    author: str = Query(..., description="Author name")
):
    """Delete an annotation, if the user is the author."""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        clean_annotation_id = annotation_id.strip()
        
        if not clean_annotation_id:
            raise HTTPException(status_code=400, detail="Annotation ID cannot be empty")
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_annotations = await load_annotations(clean_document_id, storage_service)
        
        annotation_to_delete = None
        for ann in all_annotations:
            if ann.get('annotation_id') == clean_annotation_id:
                annotation_to_delete = ann
                break
        
        if not annotation_to_delete:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        if annotation_to_delete.get('author') != clean_author:
            raise HTTPException(status_code=403, detail="You can only delete your own annotations")
            
        updated_annotations = [ann for ann in all_annotations if ann.get('annotation_id') != clean_annotation_id]
        await save_annotations(clean_document_id, updated_annotations, storage_service)
        
        logger.info(f"Deleted annotation {clean_annotation_id} by {clean_author}")
        
        return AnnotationDeleteResponse(status="deleted", deleted_id=clean_annotation_id)

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
    author: str = Query(..., description="Author name")
):
    """Publish a private annotation to make it visible to others."""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        clean_annotation_id = annotation_id.strip()
        
        if not clean_annotation_id:
            raise HTTPException(status_code=400, detail="Annotation ID cannot be empty")
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
