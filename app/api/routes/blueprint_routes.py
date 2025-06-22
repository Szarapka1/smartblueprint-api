# app/api/routes/blueprint_routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import json
import uuid
import re
import os
from app.core.config import get_settings

blueprint_router = APIRouter()
settings = get_settings()

# ================================
# PYDANTIC MODELS
# ================================

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

class CreateAnnotationRequest(BaseModel):
    document_id: str = Field(..., min_length=3, max_length=50)
    page_number: int = Field(..., ge=1)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    text: str = Field(..., min_length=1, max_length=1000)
    annotation_type: str = Field(..., regex="^(note|highlight|pen)$")
    author: str = Field(..., min_length=1, max_length=100)
    is_private: bool = Field(default=True)

class CreateAnnotationResponse(BaseModel):
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

class AnnotationsListResponse(BaseModel):
    document_id: str
    author: str
    annotations: List[dict]
    total_count: int

# ================================
# UTILITY FUNCTIONS
# ================================

def validate_document_id(document_id: str) -> str:
    """Validate and sanitize document ID for shared use"""
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    
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

def validate_admin_access(admin_token: str) -> bool:
    """Validate admin access using environment variable"""
    expected_token = os.getenv("ADMIN_SECRET_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Admin access not configured"
        )
    return admin_token == expected_token

def sanitize_author_name(author: str) -> str:
    """Convert author name to safe filename format"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', author.lower().replace(' ', '_'))

# ================================
# CHAT HISTORY FUNCTIONS
# ================================

async def load_user_chat_history(document_id: str, author: str, storage_service) -> List[dict]:
    """Load a specific user's private chat history"""
    safe_author = sanitize_author_name(author)
    chat_blob_name = f"{document_id}_chat_{safe_author}.json"
    
    try:
        chat_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=chat_blob_name
        )
        return json.loads(chat_data)
    except Exception:
        return []

async def save_user_chat_history(document_id: str, author: str, chat_history: List[dict], storage_service):
    """Save a user's private chat history"""
    safe_author = sanitize_author_name(author)
    chat_blob_name = f"{document_id}_chat_{safe_author}.json"
    chat_json = json.dumps(chat_history, indent=2, ensure_ascii=False)
    
    await storage_service.upload_file(
        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
        blob_name=chat_blob_name,
        data=chat_json.encode('utf-8')
    )

async def log_all_chat_activity(document_id: str, author: str, prompt: str, ai_response: str, storage_service):
    """Log ALL chat activity from ALL users for analytics"""
    activity_blob_name = f"{document_id}_all_chats.json"
    
    try:
        activity_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name
        )
        all_chats = json.loads(activity_data)
    except:
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

# ================================
# ANNOTATION FUNCTIONS
# ================================

async def load_user_annotations(document_id: str, author: str, storage_service) -> List[dict]:
    """Load user's annotations (private + public from others)"""
    safe_author = sanitize_author_name(author)
    
    # Load user's private annotations
    private_blob_name = f"{document_id}_annotations_{safe_author}.json"
    try:
        private_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=private_blob_name
        )
        private_annotations = json.loads(private_data)
    except Exception:
        private_annotations = []
    
    # Load public annotations from all users
    public_blob_name = f"{document_id}_public_annotations.json"
    try:
        public_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=public_blob_name
        )
        public_annotations = json.loads(public_data)
        # Filter out this user's public annotations to avoid duplicates
        public_annotations = [ann for ann in public_annotations if ann.get('author') != author]
    except Exception:
        public_annotations = []
    
    return private_annotations + public_annotations

async def save_annotation(annotation_data: dict, storage_service):
    """Save annotation to appropriate storage (private or public)"""
    document_id = annotation_data['document_id']
    author = annotation_data['author']
    is_private = annotation_data['is_private']
    
    if is_private:
        # Save to user's private annotations
        safe_author = sanitize_author_name(author)
        private_blob_name = f"{document_id}_annotations_{safe_author}.json"
        
        try:
            existing_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=private_blob_name
            )
            annotations = json.loads(existing_data)
        except Exception:
            annotations = []
        
        annotations.append(annotation_data)
        
        annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=private_blob_name,
            data=annotations_json.encode('utf-8')
        )
    else:
        # Save to public annotations
        public_blob_name = f"{document_id}_public_annotations.json"
        
        try:
            existing_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=public_blob_name
            )
            annotations = json.loads(existing_data)
        except Exception:
            annotations = []
        
        annotations.append(annotation_data)
        
        annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=public_blob_name,
            data=annotations_json.encode('utf-8')
        )

async def delete_user_annotation(document_id: str, annotation_id: str, author: str, storage_service) -> bool:
    """Delete an annotation (only the author can delete their own annotations)"""
    safe_author = sanitize_author_name(author)
    
    # Try private annotations first
    private_blob_name = f"{document_id}_annotations_{safe_author}.json"
    try:
        private_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=private_blob_name
        )
        annotations = json.loads(private_data)
        
        original_count = len(annotations)
        annotations = [ann for ann in annotations if ann.get('annotation_id') != annotation_id]
        
        if len(annotations) < original_count:
            # Found and removed from private annotations
            annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=private_blob_name,
                data=annotations_json.encode('utf-8')
            )
            return True
    except Exception:
        pass
    
    # Try public annotations if not found in private
    public_blob_name = f"{document_id}_public_annotations.json"
    try:
        public_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=public_blob_name
        )
        annotations = json.loads(public_data)
        
        original_count = len(annotations)
        # Only allow deletion if the author matches
        annotations = [ann for ann in annotations 
                      if not (ann.get('annotation_id') == annotation_id and ann.get('author') == author)]
        
        if len(annotations) < original_count:
            # Found and removed from public annotations
            annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
            await storage_service.upload_file(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=public_blob_name,
                data=annotations_json.encode('utf-8')
            )
            return True
    except Exception:
        pass
    
    return False

# ================================
# DOCUMENT ROUTES
# ================================

@blueprint_router.post("/documents/upload", status_code=201)
async def upload_shared_document(
    request: Request,
    document_id: str,
    file: UploadFile = File(...)
):
    """
    Uploads a PDF blueprint as a shared document that everyone can access.
    Uses a custom document_id instead of random session_id.
    """
    try:
        # Validate file type
        if not file.content_type or file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only PDF files are allowed."
            )

        # Validate file size (60MB limit)
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "60")) * 1024 * 1024
        pdf_bytes = await file.read()
        if len(pdf_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            )

        # Validate and clean the document ID
        clean_document_id = validate_document_id(document_id)
        
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service

        # Check if document already exists
        try:
            existing_context = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
            return {
                "document_id": clean_document_id,
                "filename": file.filename,
                "status": "already_exists",
                "message": f"Document '{clean_document_id}' already exists and is ready for use",
                "file_size_mb": round(len(pdf_bytes) / (1024*1024), 2)
            }
        except Exception:
            # Document doesn't exist, proceed with upload/processing
            pass

        # Upload original PDF to main container
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf",
            data=pdf_bytes
        )

        # Process PDF for shared AI and chat use
        await pdf_service.process_and_cache_pdf(
            session_id=clean_document_id,
            pdf_bytes=pdf_bytes,
            storage_service=storage_service
        )

        return {
            "document_id": clean_document_id,
            "filename": file.filename,
            "status": "processing_complete",
            "message": f"Document '{clean_document_id}' uploaded and ready for collaborative use",
            "file_size_mb": round(len(pdf_bytes) / (1024*1024), 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@blueprint_router.get("/documents/{document_id}/info")
async def get_document_info(
    request: Request,
    document_id: str
):
    """
    Get information about a shared document including processing status and stats.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service

        document_info = await ai_service.get_document_info(
            document_id=clean_document_id,
            storage_service=storage_service
        )

        return document_info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents")
async def list_shared_documents(request: Request):
    """
    List all available shared documents in the system.
    """
    try:
        storage_service = request.app.state.storage_service
        
        # List all cached chunks files to find available documents
        blobs = await storage_service.list_blobs(container_name=settings.AZURE_CACHE_CONTAINER_NAME)
        
        documents = []
        for blob_name in blobs:
            if blob_name.endswith('_chunks.json'):
                # Extract document_id from blob name
                document_id = blob_name.replace('_chunks.json', '')
                
                try:
                    # Get document info
                    ai_service = request.app.state.ai_service
                    doc_info = await ai_service.get_document_info(document_id, storage_service)
                    documents.append(doc_info)
                except Exception:
                    # If we can't get info, still list the document as available
                    documents.append({
                        "document_id": document_id,
                        "status": "available",
                        "warning": "Could not retrieve full info"
                    })
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

# ================================
# CHAT ROUTES
# ================================

@blueprint_router.post("/documents/{document_id}/chat", response_model=DocumentChatResponse)
async def chat_with_shared_document(
    request: Request,
    document_id: str,
    chat: DocumentChatRequest
):
    """
    Ask questions about a shared document.
    AI will automatically determine if it needs page images via tool calling.
    """
    try:
        # Validate document ID matches the one in the body
        if document_id != chat.document_id:
            raise HTTPException(
                status_code=400, 
                detail="Document ID in URL must match document ID in request body"
            )
        
        clean_document_id = validate_document_id(document_id)
        
        # Validate input
        if not chat.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        if not chat.author.strip():
            raise HTTPException(
                status_code=400,
                detail="Author name cannot be empty"
            )
        
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service

        # Verify document exists
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{clean_document_id}' not found. Please upload it first."
            )

        # Get AI response (AI will decide if it needs page images via tools)
        ai_response_text = await ai_service.get_ai_response(
            prompt=chat.prompt.strip(),
            document_id=clean_document_id,
            storage_service=storage_service,
            author=chat.author.strip()
        )

        # Generate unique chat ID and timestamp
        chat_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Save to user's private chat history
        user_chat_history = await load_user_chat_history(clean_document_id, chat.author, storage_service)
        
        chat_entry = {
            "chat_id": chat_id,
            "timestamp": timestamp,
            "prompt": chat.prompt.strip(),
            "ai_response": ai_response_text
        }
        
        user_chat_history.append(chat_entry)
        
        # Keep configurable number of conversations per user
        max_user_chats = int(os.getenv("MAX_USER_CHAT_HISTORY", "100"))
        if len(user_chat_history) > max_user_chats:
            user_chat_history = user_chat_history[-max_user_chats:]
        
        await save_user_chat_history(clean_document_id, chat.author, user_chat_history, storage_service)

        # Log all activity for analytics
        await log_all_chat_activity(
            document_id=clean_document_id,
            author=chat.author,
            prompt=chat.prompt.strip(),
            ai_response=ai_response_text,
            storage_service=storage_service
        )

        return DocumentChatResponse(
            document_id=clean_document_id,
            ai_response=ai_response_text,
            author=chat.author,
            chat_id=chat_id,
            timestamp=timestamp,
            source_pages=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

@blueprint_router.get("/documents/{document_id}/my-chats")
async def get_my_chat_history(
    request: Request,
    document_id: str,
    author: str,
    limit: int = 20
):
    """
    Get user's own private chat history.
    Each user only sees their own conversations.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        
        if not author.strip():
            raise HTTPException(
                status_code=400,
                detail="Author parameter is required"
            )
        
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 100"
            )
        
        storage_service = request.app.state.storage_service
        
        # Load user's private chat history
        chat_history = await load_user_chat_history(clean_document_id, author, storage_service)
        
        # Return most recent chats
        recent_chats = chat_history[-limit:] if len(chat_history) > limit else chat_history
        
        return {
            "document_id": clean_document_id,
            "author": author,
            "chat_history": list(reversed(recent_chats)),  # Most recent first
            "total_conversations": len(chat_history),
            "showing": len(recent_chats)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

# ================================
# ANNOTATION ROUTES
# ================================

@blueprint_router.post("/documents/{document_id}/annotations", response_model=CreateAnnotationResponse)
async def create_annotation(
    request: Request,
    document_id: str,
    annotation: CreateAnnotationRequest
):
    """
    Create a new annotation (note, highlight, or drawing).
    Annotations can be private (only visible to creator) or public (visible to all).
    """
    try:
        # Validate document ID matches
        if document_id != annotation.document_id:
            raise HTTPException(
                status_code=400,
                detail="Document ID in URL must match document ID in request body"
            )
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Verify document exists
        try:
            await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{clean_document_id}_context.txt"
            )
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{clean_document_id}' not found"
            )
        
        # Create annotation data
        annotation_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        annotation_data = {
            "annotation_id": annotation_id,
            "document_id": clean_document_id,
            "page_number": annotation.page_number,
            "x": annotation.x,
            "y": annotation.y,
            "text": annotation.text,
            "annotation_type": annotation.annotation_type,
            "author": annotation.author.strip(),
            "is_private": annotation.is_private,
            "timestamp": timestamp
        }
        
        # Save annotation
        await save_annotation(annotation_data, storage_service)
        
        return CreateAnnotationResponse(**annotation_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@blueprint_router.get("/documents/{document_id}/annotations", response_model=AnnotationsListResponse)
async def get_user_visible_annotations(
    request: Request,
    document_id: str,
    author: str
):
    """
    Get all annotations visible to a user.
    Returns user's private annotations + public annotations from other users.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        
        if not author.strip():
            raise HTTPException(
                status_code=400,
                detail="Author parameter is required"
            )
        
        storage_service = request.app.state.storage_service
        
        # Load user's visible annotations
        annotations = await load_user_annotations(clean_document_id, author.strip(), storage_service)
        
        return AnnotationsListResponse(
            document_id=clean_document_id,
            author=author.strip(),
            annotations=annotations,
            total_count=len(annotations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get annotations: {str(e)}")

@blueprint_router.delete("/documents/{document_id}/annotations/{annotation_id}")
async def delete_annotation(
    request: Request,
    document_id: str,
    annotation_id: str,
    author: str
):
    """
    Delete an annotation.
    Only the author can delete their own annotations.
    """
    try:
        clean_document_id = validate_document_id(document_id)
        
        if not author.strip():
            raise HTTPException(
                status_code=400,
                detail="Author parameter is required"
            )
        
        storage_service = request.app.state.storage_service
        
        # Attempt to delete the annotation
        deleted = await delete_user_annotation(
            clean_document_id, 
            annotation_id, 
            author.strip(), 
            storage_service
        )
        
        if not deleted:
            raise HTTPException(
                status_code=404, 
                detail="Annotation not found or not owned by user"
            )
        
        return {
            "message": "Annotation deleted successfully", 
            "annotation_id": annotation_id,
            "document_id": clean_document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete annotation: {str(e)}")

# ================================
# ADMIN ROUTES
# ================================

@blueprint_router.get("/admin/documents/{document_id}/all-chats")
async def admin_get_all_chats(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    ADMIN: Get ALL chat conversations from ALL users for analytics.
    Requires admin token in X-Admin-Token header.
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
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
        
        # Calculate analytics
        analytics = {
            "total_conversations": len(all_chats),
            "unique_users": len(set(chat.get("author") for chat in all_chats)),
            "avg_prompt_length": sum(chat.get("prompt_length", 0) for chat in all_chats) / len(all_chats) if all_chats else 0,
            "avg_response_length": sum(chat.get("response_length", 0) for chat in all_chats) / len(all_chats) if all_chats else 0,
            "most_active_users": {}
        }
        
        # Calculate user activity
        user_counts = {}
        for chat in all_chats:
            author = chat.get("author")
            if author:
                user_counts[author] = user_counts.get(author, 0) + 1
        
        analytics["most_active_users"] = dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            "document_id": clean_document_id,
            "all_chats": all_chats,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@blueprint_router.get("/admin/documents/{document_id}/user-chat/{author}")
async def admin_get_user_chat(
    request: Request,
    document_id: str,
    author: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    ADMIN: Get specific user's complete chat history.
    Requires admin token in X-Admin-Token header.
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
        
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        # Get specific user's chat history
        user_chats = await load_user_chat_history(clean_document_id, author, storage_service)
        
        return {
            "document_id": clean_document_id,
            "author": author,
            "chat_history": user_chats,
            "total_conversations": len(user_chats)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin access failed: {str(e)}")

@blueprint_router.get("/admin/system/stats")
async def admin_get_system_stats(
    request: Request,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """
    ADMIN: Get overall system statistics.
    Requires admin token in X-Admin-Token header.
    """
    try:
        if not admin_token:
            raise HTTPException(status_code=401, detail="Admin token required")
