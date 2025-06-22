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

class DocumentChatRequest(BaseModel):
    document_id: str = Field(..., min_length=3, max_length=50)
    prompt: str = Field(..., min_length=1, max_length=2000)
    page_number: Optional[int] = Field(None, ge=1)
    author: str = Field(..., min_length=1, max_length=100)

class DocumentChatResponse(BaseModel):
    document_id: str
    ai_response: str
    author: str
    chat_id: str
    timestamp: str
    source_pages: List[int] = []

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

async def load_user_chat_history(document_id: str, author: str, storage_service) -> List[dict]:
    """Load a specific user's private chat history"""
    # Sanitize author name for filename
    safe_author = re.sub(r'[^a-zA-Z0-9_-]', '_', author.lower().replace(' ', '_'))
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
    safe_author = re.sub(r'[^a-zA-Z0-9_-]', '_', author.lower().replace(' ', '_'))
    chat_blob_name = f"{document_id}_chat_{safe_author}.json"
    chat_json = json.dumps(chat_history, indent=2, ensure_ascii=False)
    
    await storage_service.upload_file(
        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
        blob_name=chat_blob_name,
        data=chat_json.encode('utf-8')
    )

async def log_all_chat_activity(document_id: str, author: str, prompt: str, ai_response: str, storage_service):
    """SERVICE: Log ALL chat activity from ALL users for analytics"""
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
        
        # Check if required services are available
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(
                status_code=503,
                detail="Storage service unavailable. Please try again later."
            )
        
        if not pdf_service:
            raise HTTPException(
                status_code=503,
                detail="PDF processing service unavailable. Please try again later."
            )

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
        except:
            # Document doesn't exist, proceed with upload
            pass

        # Upload original PDF to main container
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf",
            data=pdf_bytes
        )

        # Process PDF for shared AI and chat use
        await pdf_service.process_and_cache_pdf(
            session_id=clean_document_id,  # Use document_id as session_id for processing
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

# ... [rest of the routes remain the same] ...

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
        except:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{clean_document_id}' not found. Please upload it first."
            )

        # Get AI response for shared document (uses shared cached chunks)
        ai_response_text = await ai_service.get_ai_response(
            prompt=chat.prompt.strip(),
            document_id=clean_document_id,
            storage_service=storage_service,
            page_number=chat.page_number,
            author=chat.author.strip()
        )

        # Generate unique chat ID and timestamp
        chat_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        # USER: Save to their private chat history
        user_chat_history = await load_user_chat_history(clean_document_id, chat.author, storage_service)
        
        chat_entry = {
            "chat_id": chat_id,
            "timestamp": timestamp,
            "prompt": chat.prompt.strip(),
            "ai_response": ai_response_text,
            "page_number": chat.page_number
        }
        
        user_chat_history.append(chat_entry)
        
        # Keep configurable number of conversations per user
        max_user_chats = int(os.getenv("MAX_USER_CHAT_HISTORY", "100"))
        if len(user_chat_history) > max_user_chats:
            user_chat_history = user_chat_history[-max_user_chats:]
        
        await save_user_chat_history(clean_document_id, chat.author, user_chat_history, storage_service)

        # SERVICE: Log everything for analytics (you get all data)
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
            source_pages=[]  # Can be enhanced later with specific page references
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

# [Include all other routes from your original file...]
