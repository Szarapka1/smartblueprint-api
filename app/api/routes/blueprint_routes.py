# app/api/routes/blueprint_routes.py - WITH NOTES AND HIGHLIGHTING

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header, Query, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import uuid
import re
import os
import logging
from app.core.config import get_settings
from app.models.schemas import (
    Note, NoteCreate, NoteUpdate, NoteBatch, NoteList,
    ChatRequest, ChatResponse, VisualElement, DrawingGrid,
    DocumentUploadResponse, DocumentInfoResponse, DocumentListResponse,
    SuccessResponse, ErrorResponse,
    Annotation  # For backward compatibility
)

logger = logging.getLogger(__name__)
blueprint_router = APIRouter()
settings = get_settings()

# --- Helper Functions (KEEPING ALL EXISTING ONES) ---

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

# --- Chat History Management (KEEPING EXISTING) ---

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

async def log_all_chat_activity(document_id: str, author: str, prompt: str, 
                               ai_response: str, visual_highlights: Optional[List[Dict]], 
                               storage_service):
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
        chat_entry = {
            "chat_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "document_id": document_id,
            "author": author,
            "prompt": prompt,
            "ai_response": ai_response,
            "prompt_length": len(prompt),
            "response_length": len(ai_response),
            "had_visual_highlights": bool(visual_highlights),
            "highlight_count": len(visual_highlights) if visual_highlights else 0
        }
        
        all_chats.append(chat_entry)
        
        # Limit size
        max_chats = getattr(settings, 'MAX_CHAT_LOGS', 1000)
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

# --- NEW: Note Management Functions ---

async def load_notes(document_id: str, storage_service) -> List[dict]:
    """Load all notes for a document"""
    try:
        blob_name = f"{document_id}_notes.json"
        notes_data = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name
        )
        notes = json.loads(notes_data)
        logger.info(f"Loaded {len(notes)} notes for {document_id}")
        return notes
    except Exception:
        logger.info(f"No notes found for {document_id}")
        return []

async def save_notes(document_id: str, notes: List[dict], storage_service):
    """Save all notes for a document"""
    try:
        blob_name = f"{document_id}_notes.json"
        notes_json = json.dumps(notes, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=blob_name,
            data=notes_json.encode('utf-8')
        )
        logger.info(f"Saved {len(notes)} notes for {document_id}")
    except Exception as e:
        logger.error(f"Failed to save notes for {document_id}: {e}")
        raise

async def get_document_metadata(document_id: str, storage_service) -> Optional[Dict]:
    """Get document metadata if available"""
    try:
        metadata_blob = f"{document_id}_metadata.json"
        metadata_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=metadata_blob
        )
        return json.loads(metadata_text)
    except Exception:
        return None

# --- MAIN API ROUTES (KEEPING ALL EXISTING ENDPOINTS) ---

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
        document_exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if document_exists and not force_reprocess:
            logger.info(f"Document {clean_document_id} already exists")
            
            return DocumentUploadResponse(
                document_id=clean_document_id,
                filename=file.filename,
                status="already_exists",
                message=f"Document '{clean_document_id}' already exists. Use force_reprocess=true to reprocess.",
                file_size_mb=round(len(pdf_bytes) / (1024*1024), 2)
            )

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
        exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not exists:
            raise HTTPException(
                status_code=404, 
                detail=f"Document '{clean_document_id}' not found"
            )
        
        # Get metadata
        metadata = await get_document_metadata(clean_document_id, storage_service)
        
        return DocumentInfoResponse(
            document_id=clean_document_id,
            status="ready",
            message="Document is ready for use",
            exists=True,
            metadata=metadata
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """List all available documents with pagination"""
    try:
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Use optimized method
        if hasattr(storage_service, 'list_document_ids'):
            document_ids = await storage_service.list_document_ids(settings.AZURE_CACHE_CONTAINER_NAME)
        else:
            context_files = await storage_service.list_blobs(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                suffix="_context.txt"
            )
            document_ids = [f.replace('_context.txt', '') for f in context_files]
        
        # Apply pagination
        paginated_ids = document_ids[offset:offset + limit]
        has_more = len(document_ids) > offset + limit
        
        documents = []
        for doc_id in paginated_ids:
            doc_info = {
                "document_id": doc_id,
                "status": "ready"
            }
            
            # Try to get metadata
            try:
                metadata = await get_document_metadata(doc_id, storage_service)
                if metadata:
                    doc_info["page_count"] = metadata.get("page_count", 0)
                    doc_info["file_size_mb"] = metadata.get("file_size_mb", 0)
            except:
                pass
            
            documents.append(doc_info)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(document_ids),
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@blueprint_router.get("/documents/{document_id}/pdf")
async def download_document_pdf(request: Request, document_id: str):
    """Download the original PDF for viewing in shared documents"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Check existence first
        exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        if not exists:
            raise HTTPException(status_code=404, detail=f"PDF for document '{document_id}' not found")
        
        # Download PDF
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={clean_document_id}.pdf",
                "Cache-Control": "public, max-age=3600"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")

# --- ENHANCED CHAT ENDPOINT WITH HIGHLIGHTING ---

@blueprint_router.post("/documents/{document_id}/chat", response_model=ChatResponse)
async def chat_with_shared_document(
    request: Request, 
    document_id: str, 
    chat_request: ChatRequest
):
    """Chat with AI about a document with optional visual highlighting"""
    try:
        clean_document_id = validate_document_id(document_id)
        
        # Validate author from body (for backward compatibility)
        author = getattr(chat_request, 'author', 'anonymous')
        clean_author = validate_author(author) if author != 'anonymous' else author
        
        clean_prompt = chat_request.prompt.strip()
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
        exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not exists:
            raise HTTPException(
                status_code=404, 
                detail=f"Document '{clean_document_id}' not found"
            )

        # Get AI response with optional highlighting
        ai_result = await ai_service.get_ai_response(
            prompt=clean_prompt,
            document_id=clean_document_id,
            storage_service=storage_service,
            author=clean_author,
            current_page=chat_request.current_page,
            request_highlights=True  # Always allow highlights if page specified
        )
        
        # Extract components from result
        if isinstance(ai_result, dict):
            ai_response_text = ai_result.get("ai_response", "")
            visual_highlights = ai_result.get("visual_highlights")
            drawing_grid = ai_result.get("drawing_grid")
            highlight_summary = ai_result.get("highlight_summary")
            current_page = ai_result.get("current_page")
        else:
            # Backward compatibility
            ai_response_text = ai_result
            visual_highlights = None
            drawing_grid = None
            highlight_summary = None
            current_page = None
        
        chat_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Save to user's chat history
        chat_history = await load_user_chat_history(clean_document_id, clean_author, storage_service)
        chat_entry = {
            "chat_id": chat_id,
            "timestamp": timestamp,
            "prompt": clean_prompt,
            "ai_response": ai_response_text,
            "had_highlights": bool(visual_highlights)
        }
        chat_history.append(chat_entry)
        
        # Limit history size
        max_chats = int(os.getenv("MAX_USER_CHAT_HISTORY", "100"))
        await save_user_chat_history(
            clean_document_id, 
            clean_author, 
            chat_history[-max_chats:], 
            storage_service
        )
        
        # Log for analytics
        await log_all_chat_activity(
            clean_document_id, 
            clean_author, 
            clean_prompt, 
            ai_response_text,
            visual_highlights,
            storage_service
        )

        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_response_text,
            source_pages=[current_page] if current_page else [],
            visual_highlights=visual_highlights,
            drawing_grid=drawing_grid,
            highlight_summary=highlight_summary,
            current_page=current_page
        )
        
        return response
        
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

# --- NEW NOTE ENDPOINTS (Replacing Annotations) ---

@blueprint_router.post("/documents/{document_id}/notes", response_model=Note)
async def create_note(
    request: Request, 
    document_id: str, 
    note_data: NoteCreate,
    author: str = Query(..., description="Author of the note")
):
    """Create a new note about the document"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        
        storage_service = request.app.state.storage_service
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Load existing notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Create new note
        note = Note(
            note_id=str(uuid.uuid4())[:8],
            document_id=clean_document_id,
            text=note_data.text.strip(),
            note_type=note_data.note_type,
            author=clean_author,
            is_private=note_data.is_private,
            timestamp=datetime.utcnow().isoformat() + "Z",
            char_count=len(note_data.text.strip())
        )
        
        all_notes.append(note.dict())
        await save_notes(clean_document_id, all_notes, storage_service)
        
        logger.info(f"Created note {note.note_id} for document {clean_document_id}")
        return note
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create note: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create note: {str(e)}")

@blueprint_router.get("/documents/{document_id}/notes", response_model=NoteList)
async def get_document_notes(
    request: Request, 
    document_id: str, 
    author: str = Query(..., description="Viewing user"),
    note_type: Optional[str] = Query(None, description="Filter by note type"),
    include_private: bool = Query(True, description="Include your private notes")
):
    """Get notes visible to the user (their private + all public)"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Filter to visible notes
        visible_notes = []
        for note_dict in all_notes:
            # Include if public OR authored by current user
            if not note_dict.get("is_private", True) or note_dict.get("author") == clean_author:
                # Apply type filter if specified
                if note_type and note_dict.get("note_type") != note_type:
                    continue
                    
                # Convert dict to Note object
                note = Note(**note_dict)
                visible_notes.append(note)
        
        # Sort by timestamp (newest first)
        visible_notes.sort(key=lambda n: n.timestamp, reverse=True)
        
        return NoteList(
            notes=visible_notes,
            total_count=len(visible_notes),
            filter_applied={"note_type": note_type} if note_type else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get notes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get notes: {str(e)}")

@blueprint_router.put("/documents/{document_id}/notes/{note_id}", response_model=Note)
async def update_note(
    request: Request,
    document_id: str,
    note_id: str,
    update_data: NoteUpdate,
    author: str = Query(..., description="Author updating the note")
):
    """Update a note (only by its author)"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find and update note
        updated = False
        updated_note = None
        
        for i, note_dict in enumerate(all_notes):
            if note_dict.get("note_id") == note_id:
                if note_dict.get("author") != clean_author:
                    raise HTTPException(status_code=403, detail="You can only edit your own notes")
                
                # Update fields
                if update_data.text is not None:
                    note_dict["text"] = update_data.text.strip()
                    note_dict["char_count"] = len(update_data.text.strip())
                
                if update_data.note_type is not None:
                    note_dict["note_type"] = update_data.note_type
                
                note_dict["edited_at"] = datetime.utcnow().isoformat() + "Z"
                
                updated_note = Note(**note_dict)
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Note not found")
        
        await save_notes(clean_document_id, all_notes, storage_service)
        return updated_note
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update note: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update note: {str(e)}")

@blueprint_router.delete("/documents/{document_id}/notes/{note_id}")
async def delete_note(
    request: Request, 
    document_id: str, 
    note_id: str, 
    author: str = Query(...)
):
    """Delete user's own note"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find note to delete
        note_to_delete = None
        for note in all_notes:
            if note.get('note_id') == note_id:
                note_to_delete = note
                break
        
        if not note_to_delete:
            raise HTTPException(status_code=404, detail="Note not found")
        
        if note_to_delete.get('author') != clean_author:
            raise HTTPException(status_code=403, detail="You can only delete your own notes")
            
        # Remove note
        updated_notes = [n for n in all_notes if n.get('note_id') != note_id]
        
        await save_notes(clean_document_id, updated_notes, storage_service)
        
        return {"status": "deleted", "deleted_id": note_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete note: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete note: {str(e)}")

@blueprint_router.post("/documents/{document_id}/notes/{note_id}/publish")
async def publish_note(
    request: Request, 
    document_id: str, 
    note_id: str, 
    author: str = Query(...)
):
    """Make a private note public"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find and update note
        updated = False
        for note in all_notes:
            if note.get("note_id") == note_id and note.get("author") == clean_author:
                if not note.get("is_private", True):
                    raise HTTPException(status_code=400, detail="Note is already public")
                
                note["is_private"] = False
                note["published_at"] = datetime.utcnow().isoformat() + "Z"
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Private note not found or not owned by you")
        
        await save_notes(clean_document_id, all_notes, storage_service)
        
        return {"status": "published", "note_id": note_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish note: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish note: {str(e)}")

@blueprint_router.post("/documents/{document_id}/notes/publish-batch")
async def publish_multiple_notes(
    request: Request, 
    document_id: str, 
    batch: NoteBatch,
    author: str = Query(...)
):
    """Publish multiple private notes at once"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        all_notes = await load_notes(clean_document_id, storage_service)
        published_count = 0
        
        for note in all_notes:
            if (note.get("note_id") in batch.note_ids and 
                note.get("author") == clean_author and
                note.get("is_private", True)):
                
                note["is_private"] = False
                note["published_at"] = datetime.utcnow().isoformat() + "Z"
                published_count += 1
        
        await save_notes(clean_document_id, all_notes, storage_service)
        
        return {
            "status": "published", 
            "published_count": published_count,
            "note_ids": batch.note_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch publish: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- BACKWARD COMPATIBILITY ENDPOINTS ---

@blueprint_router.post("/documents/{document_id}/annotations")
async def create_annotation_compat(
    request: Request,
    document_id: str,
    annotation: Annotation
):
    """DEPRECATED: Compatibility endpoint - redirects to notes"""
    # Convert annotation to note
    note_data = NoteCreate(
        text=annotation.text,
        note_type=annotation.annotation_type,
        is_private=annotation.is_private
    )
    
    return await create_note(
        request=request,
        document_id=document_id,
        note_data=note_data,
        author=annotation.author
    )

@blueprint_router.get("/documents/{document_id}/annotations")
async def get_annotations_compat(
    request: Request,
    document_id: str,
    author: str = Query(...)
):
    """DEPRECATED: Compatibility endpoint - returns notes as annotations"""
    notes_response = await get_document_notes(
        request=request,
        document_id=document_id,
        author=author
    )
    
    # Convert notes to annotation format
    annotations = []
    for note in notes_response.notes:
        annotations.append({
            "annotation_id": note.note_id,
            "document_id": note.document_id,
            "page_number": 1,  # Default
            "x": 0,
            "y": 0,
            "text": note.text,
            "annotation_type": note.note_type,
            "author": note.author,
            "is_private": note.is_private,
            "timestamp": note.timestamp
        })
    
    return {"annotations": annotations}

# --- ADMIN ENDPOINTS ---

@blueprint_router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """Delete a document and all associated files (admin only)"""
    try:
        # Verify admin token
        expected_token = getattr(settings, 'ADMIN_SECRET_TOKEN', None)
        if not expected_token or admin_token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid admin token")
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Use optimized deletion
        if hasattr(storage_service, 'delete_document_files'):
            deleted_counts = await storage_service.delete_document_files(clean_document_id)
            
            if deleted_counts['total'] == 0:
                raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")
            
            return {
                "status": "deleted",
                "document_id": clean_document_id,
                "files_deleted": deleted_counts
            }
        else:
            # Fallback
            deleted_count = 0
            
            # Delete original PDF
            try:
                await storage_service.delete_blob(settings.AZURE_CONTAINER_NAME, f"{clean_document_id}.pdf")
                deleted_count += 1
            except:
                pass
            
            # Delete all cache files
            cache_files = await storage_service.list_blobs(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=f"{clean_document_id}_"
            )
            
            for filename in cache_files:
                try:
                    await storage_service.delete_blob(settings.AZURE_CACHE_CONTAINER_NAME, filename)
                    deleted_count += 1
                except:
                    pass
            
            if deleted_count == 0:
                raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")
            
            return {
                "status": "deleted",
                "document_id": clean_document_id,
                "files_deleted": deleted_count
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
