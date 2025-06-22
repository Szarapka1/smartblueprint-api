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

# Maximum file size (60MB)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "60")) * 1024 * 1024

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    file_size: int
    page_count: int = 0
    text_length: int = 0
    image_count: int = 0

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

class CreateAnnotationRequest(BaseModel):
    document_id: str = Field(..., min_length=3, max_length=50)
    page_number: int = Field(..., ge=1)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    text: str = Field(..., min_length=1, max_length=1000)
    annotation_type: str = Field(..., pattern="^(note|highlight|pen)$")
    author: str = Field(..., min_length=1, max_length=100)
    is_private: bool = Field(default=True)

class AnnotationResponse(BaseModel):
    annotation_id: str
    document_id: str
    page_number: int
    x: float
    y: float
    text: str
    annotation_type: str
    author: str
    timestamp: str
    is_private: bool

def generate_document_id() -> str:
    """Generate a unique document ID"""
    return f"DOC-{uuid.uuid4().hex[:8].upper()}-{uuid.uuid4().hex[:5].upper()}"

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

# === UPLOAD ROUTES ===

@blueprint_router.post("/upload", response_model=DocumentUploadResponse)
async def upload_blueprint(
    file: UploadFile = File(...),
    request: Request = None
):
    """Upload and process a new blueprint document with auto-generated ID"""
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file data
        file_data = await file.read()
        if len(file_data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")
        
        # Generate document ID
        document_id = generate_document_id()
        
        # Get services
        storage_service = request.app.state.storage_service
        pdf_service = request.app.state.pdf_service
        
        if not storage_service or not pdf_service:
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Save original file
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{document_id}.pdf",
            data=file_data
        )
        
        # Process PDF using existing method
        await pdf_service.process_and_cache_pdf(
            session_id=document_id,
            pdf_bytes=file_data,
            storage_service=storage_service
        )
        
        # Get processing results for response
        try:
            # Get full text to calculate length
            context_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_context.txt"
            )
            text_length = len(context_data)
            
            # Get chunks to estimate page count
            chunks_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_chunks.json"
            )
            chunks = json.loads(chunks_data)
            
            # Estimate page count and image count from blob storage
            all_blobs = await storage_service.list_blobs(settings.AZURE_CACHE_CONTAINER_NAME)
            page_images = [blob for blob in all_blobs if blob.startswith(f"{document_id}_page_") and blob.endswith('.png')]
            
        except Exception as e:
            # If we can't get detailed info, provide basic response
            text_length = 0
            page_images = []
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            file_size=len(file_data),
            page_count=len(page_images),
            text_length=text_length,
            image_count=len(page_images)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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

        # Validate file size
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
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

# === CHAT ROUTES ===

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

# === ANNOTATION ROUTES ===

@blueprint_router.post("/documents/{document_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(
    request: Request,
    document_id: str,
    annotation: CreateAnnotationRequest
):
    """Create a new annotation on a document"""
    try:
        # Validate document ID matches
        if document_id != annotation.document_id:
            raise HTTPException(
                status_code=400,
                detail="Document ID in URL must match document ID in request body"
            )
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
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
        
        # Generate annotation ID
        annotation_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Create annotation data
        annotation_data = {
            "annotation_id": annotation_id,
            "document_id": clean_document_id,
            "page_number": annotation.page_number,
            "x": annotation.x,
            "y": annotation.y,
            "text": annotation.text,
            "annotation_type": annotation.annotation_type,
            "author": annotation.author,
            "timestamp": timestamp,
            "is_private": annotation.is_private
        }
        
        # Load existing annotations
        annotations_blob_name = f"{clean_document_id}_annotations.json"
        try:
            annotations_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob_name
            )
            annotations = json.loads(annotations_data)
        except:
            annotations = []
        
        # Add new annotation
        annotations.append(annotation_data)
        
        # Save updated annotations
        annotations_json = json.dumps(annotations, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob_name,
            data=annotations_json.encode('utf-8')
        )
        
        return AnnotationResponse(**annotation_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create annotation: {str(e)}")

@blueprint_router.get("/documents/{document_id}/annotations")
async def get_annotations(
    request: Request,
    document_id: str,
    page_number: Optional[int] = None,
    author: Optional[str] = None
):
    """Get annotations for a document, optionally filtered by page or author"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Load annotations
        annotations_blob_name = f"{clean_document_id}_annotations.json"
        try:
            annotations_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob_name
            )
            annotations = json.loads(annotations_data)
        except:
            annotations = []
        
        # Filter annotations
        filtered_annotations = annotations
        
        if page_number is not None:
            filtered_annotations = [ann for ann in filtered_annotations if ann["page_number"] == page_number]
        
        if author is not None:
            filtered_annotations = [ann for ann in filtered_annotations if ann["author"] == author]
        
        return {
            "document_id": clean_document_id,
            "annotations": filtered_annotations,
            "total_count": len(filtered_annotations),
            "filters": {
                "page_number": page_number,
                "author": author
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get annotations: {str(e)}")

# === DOCUMENT INFO ROUTES ===

@blueprint_router.get("/documents/{document_id}")
async def get_document_info(request: Request, document_id: str):
    """Get information about a document"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        ai_service = request.app.state.ai_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get document info from AI service
        if ai_service:
            doc_info = await ai_service.get_document_info(clean_document_id, storage_service)
            return doc_info
        else:
            # Basic info without AI service
            try:
                context_data = await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{clean_document_id}_context.txt"
                )
                return {
                    "document_id": clean_document_id,
                    "status": "ready",
                    "total_characters": len(context_data),
                    "ai_service": "unavailable"
                }
            except:
                raise HTTPException(status_code=404, detail="Document not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents")
async def list_documents(request: Request):
    """List all available documents"""
    try:
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get all PDF files from main container
        all_blobs = await storage_service.list_blobs(settings.AZURE_CONTAINER_NAME)
        pdf_files = [blob for blob in all_blobs if blob.endswith('.pdf')]
        
        documents = []
        for pdf_file in pdf_files:
            document_id = pdf_file.replace('.pdf', '')
            
            # Check if document is processed
            try:
                await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                status = "ready"
            except:
                status = "processing"
            
            documents.append({
                "document_id": document_id,
                "filename": pdf_file,
                "status": status
            })
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

# === ADMIN ROUTES ===

@blueprint_router.get("/admin/analytics/{document_id}")
async def get_document_analytics(
    request: Request,
    document_id: str,
    admin_token: str = Header(...)
):
    """Get analytics for a specific document (admin only)"""
    try:
        # Validate admin access
        validate_admin_access(admin_token)
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get all chat activity
        activity_blob_name = f"{clean_document_id}_all_chats.json"
        try:
            activity_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=activity_blob_name
            )
            all_chats = json.loads(activity_data)
        except:
            all_chats = []
        
        # Generate analytics
        total_chats = len(all_chats)
        unique_users = len(set(chat["author"] for chat in all_chats))
        avg_prompt_length = sum(chat["prompt_length"] for chat in all_chats) / total_chats if total_chats > 0 else 0
        avg_response_length = sum(chat["response_length"] for chat in all_chats) / total_chats if total_chats > 0 else 0
        
        return {
            "document_id": clean_document_id,
            "analytics": {
                "total_chats": total_chats,
                "unique_users": unique_users,
                "avg_prompt_length": round(avg_prompt_length, 2),
                "avg_response_length": round(avg_response_length, 2),
                "recent_activity": all_chats[-10:] if all_chats else []
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")
