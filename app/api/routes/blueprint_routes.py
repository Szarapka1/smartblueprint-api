# app/api/routes/blueprint_routes.py - COMPLETE INTEGRATION WITH VISUAL HIGHLIGHTING

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header, Query, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import uuid
import re
import os
import logging
import asyncio

from app.core.config import get_settings
from app.models.schemas import (
    Note, NoteCreate, NoteUpdate, NoteBatch, NoteList,
    ChatRequest, ChatResponse, VisualElement, DrawingGrid, GridReference,
    DocumentUploadResponse, DocumentInfoResponse, DocumentListResponse,
    SuccessResponse, ErrorResponse,
    Annotation  # For backward compatibility
)

logger = logging.getLogger(__name__)
blueprint_router = APIRouter()
settings = get_settings()

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

async def log_all_chat_activity(document_id: str, author: str, prompt: str, 
                               ai_response: str, query_session_id: Optional[str],
                               highlight_stats: Dict[str, Any],
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
        
        # Add new activity with comprehensive tracking
        chat_entry = {
            "chat_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "document_id": document_id,
            "author": author,
            "prompt": prompt,
            "ai_response": ai_response[:1000],  # Truncate long responses
            "prompt_length": len(prompt),
            "response_length": len(ai_response),
            "query_session_id": query_session_id,
            "highlight_stats": highlight_stats,
            "reference_previous": bool("reference_previous" in prompt.lower()),
        }
        
        all_chats.append(chat_entry)
        
        # Limit size
        max_chats = settings.MAX_CHAT_LOGS
        if len(all_chats) > max_chats:
            all_chats = all_chats[-max_chats:]
        
        # Save back
        activity_json = json.dumps(all_chats, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=activity_blob_name,
            data=activity_json.encode('utf-8')
        )
        logger.info(f"Logged chat activity for {author} in {document_id} (session: {query_session_id})")
    except Exception as e:
        logger.error(f"Failed to log chat activity: {e}")

# --- Note Management Functions ---

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

# --- MAIN API ROUTES ---

@blueprint_router.post("/documents/upload", status_code=201, response_model=DocumentUploadResponse)
async def upload_shared_document(
    request: Request,
    document_id: str = Query(..., description="Custom document ID"),
    force_reprocess: bool = Query(False, description="Force reprocessing"),
    file: UploadFile = File(...)
):
    """Upload and process a PDF document with grid detection"""
    try:
        logger.info(f"Upload request: id={document_id}, force={force_reprocess}, file={file.filename}")
        validate_file_upload(file)
        
        # Check file size
        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        pdf_bytes = await file.read()
        if len(pdf_bytes) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size is {settings.MAX_FILE_SIZE_MB}MB"
            )

        clean_document_id = validate_document_id(document_id)
        
        # Check services
        pdf_service = request.app.state.pdf_service
        storage_service = request.app.state.storage_service
        session_service = request.app.state.session_service
        
        if not all([pdf_service, storage_service, session_service]):
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
            # Create/update session
            session_service.create_session(clean_document_id, file.filename)
            
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

        # Process PDF with grid detection
        await pdf_service.process_and_cache_pdf(
            session_id=clean_document_id,
            pdf_bytes=pdf_bytes,
            storage_service=storage_service
        )
        
        # Create session
        session_service.create_session(clean_document_id, file.filename)
        
        # Update session with metadata
        metadata = await get_document_metadata(clean_document_id, storage_service)
        if metadata:
            session_service.update_session_metadata(clean_document_id, {
                'page_count': metadata.get('page_count', 0),
                'has_grid_systems': metadata.get('extraction_summary', {}).get('has_grid_systems', False),
                'grid_systems_detected': metadata.get('optimization_stats', {}).get('grid_systems_detected', 0)
            })

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
        logger.error(f"Upload failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@blueprint_router.get("/documents/{document_id}/info", response_model=DocumentInfoResponse)
async def get_document_info(request: Request, document_id: str):
    """Get information about a specific document"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        session_service = request.app.state.session_service
        
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
        
        # Get session info if available
        session_info = None
        if session_service:
            session_info = session_service.get_session_info(clean_document_id)
        
        return DocumentInfoResponse(
            document_id=clean_document_id,
            status="ready",
            message="Document is ready for use",
            exists=True,
            metadata={
                **(metadata or {}),
                'session_info': session_info
            }
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
        session_service = request.app.state.session_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Get document IDs
        document_ids = await storage_service.list_document_ids(settings.AZURE_CACHE_CONTAINER_NAME)
        
        # Apply pagination
        paginated_ids = document_ids[offset:offset + limit]
        has_more = len(document_ids) > offset + limit
        
        # Get session data if available
        active_sessions = {}
        if session_service:
            active_sessions = session_service.get_all_sessions(limit=200)
        
        documents = []
        for doc_id in paginated_ids:
            doc_info = {
                "document_id": doc_id,
                "status": "ready"
            }
            
            # Add session info
            if doc_id in active_sessions:
                doc_info["session_data"] = active_sessions[doc_id]
            
            # Try to get metadata
            try:
                metadata = await get_document_metadata(doc_id, storage_service)
                if metadata:
                    doc_info["page_count"] = metadata.get("page_count", 0)
                    doc_info["file_size_mb"] = metadata.get("file_size_mb", 0)
                    doc_info["has_grid_systems"] = metadata.get("extraction_summary", {}).get("has_grid_systems", False)
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
    """Download the original PDF"""
    try:
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Check existence
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

# --- ENHANCED CHAT ENDPOINT WITH VISUAL HIGHLIGHTING ---

@blueprint_router.post("/documents/{document_id}/chat", response_model=ChatResponse)
async def chat_with_shared_document(
    request: Request, 
    document_id: str, 
    chat_request: ChatRequest
):
    """Chat with AI about a document with visual highlighting support"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(chat_request.author)
        clean_prompt = chat_request.prompt.strip()
        
        if not clean_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Check services
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service
        session_service = request.app.state.session_service
        
        if not all([ai_service, storage_service, session_service]):
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
        
        # Create highlight session
        highlight_session = session_service.create_highlight_session(
            document_id=clean_document_id,
            user=clean_author,
            query=clean_prompt
        )
        
        logger.info(f"Created highlight session {highlight_session.query_session_id} for {clean_author}")

        # Get AI response with visual highlighting
        ai_result = await ai_service.get_ai_response(
            prompt=clean_prompt,
            document_id=clean_document_id,
            storage_service=storage_service,
            author=clean_author,
            current_page=chat_request.current_page,
            request_highlights=True,
            reference_previous=chat_request.reference_previous,
            preserve_existing=chat_request.preserve_existing
        )
        
        # Extract components from result
        if isinstance(ai_result, dict):
            ai_response_text = ai_result.get("ai_response", "")
            visual_highlights = ai_result.get("visual_highlights")
            drawing_grid = ai_result.get("drawing_grid")
            highlight_summary = ai_result.get("highlight_summary")
            current_page = ai_result.get("current_page", chat_request.current_page)
            query_session_id = highlight_session.query_session_id
            all_highlight_pages = ai_result.get("all_highlight_pages", {})
            total_highlights = ai_result.get("total_highlights", 0)
        else:
            # Backward compatibility
            ai_response_text = str(ai_result)
            visual_highlights = None
            drawing_grid = None
            highlight_summary = None
            current_page = chat_request.current_page
            query_session_id = highlight_session.query_session_id
            all_highlight_pages = {}
            total_highlights = 0
        
        # Update highlight session with results
        if all_highlight_pages:
            element_types = list(set(
                h.element_type for h in visual_highlights
            )) if visual_highlights else []
            
            session_service.update_highlight_session(
                document_id=clean_document_id,
                query_session_id=query_session_id,
                pages_with_highlights=all_highlight_pages,
                element_types=element_types,
                total_highlights=total_highlights
            )
        
        # Save to user's chat history
        chat_history = await load_user_chat_history(clean_document_id, clean_author, storage_service)
        chat_entry = {
            "chat_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompt": clean_prompt,
            "ai_response": ai_response_text,
            "had_highlights": bool(visual_highlights),
            "query_session_id": query_session_id,
            "reference_previous": chat_request.reference_previous,
            "pages_with_highlights": list(all_highlight_pages.keys()) if all_highlight_pages else []
        }
        chat_history.append(chat_entry)
        
        # Limit history size
        max_chats = settings.MAX_USER_CHAT_HISTORY
        await save_user_chat_history(
            clean_document_id, 
            clean_author, 
            chat_history[-max_chats:], 
            storage_service
        )
        
        # Record chat activity in session
        session_service.record_chat_activity(clean_document_id, clean_author)
        
        # Log for analytics
        highlight_stats = {
            "total_highlights": total_highlights,
            "pages_with_highlights": len(all_highlight_pages),
            "highlight_counts": all_highlight_pages
        }
        
        await log_all_chat_activity(
            clean_document_id, 
            clean_author, 
            clean_prompt, 
            ai_response_text,
            query_session_id,
            highlight_stats,
            storage_service
        )

        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_response_text,
            source_pages=list(all_highlight_pages.keys()) if all_highlight_pages else [],
            visual_highlights=visual_highlights,
            drawing_grid=drawing_grid,
            highlight_summary=highlight_summary,
            current_page=current_page,
            query_session_id=query_session_id,
            all_highlight_pages=all_highlight_pages
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

# --- NEW HIGHLIGHT MANAGEMENT ENDPOINTS ---

@blueprint_router.get("/documents/{document_id}/highlights/generate")
async def generate_highlighted_page(
    request: Request,
    document_id: str,
    page_number: int = Query(..., description="Page number to highlight"),
    query_session_id: str = Query(..., description="Query session ID for highlights")
):
    """Generate highlighted version of a page on-demand"""
    try:
        clean_document_id = validate_document_id(document_id)
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service
        
        if not ai_service or not storage_service:
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Generate highlighted page
        highlighted_image = await ai_service.generate_highlighted_page(
            document_id=clean_document_id,
            page_num=page_number,
            query_session_id=query_session_id,
            storage_service=storage_service
        )
        
        if not highlighted_image:
            raise HTTPException(
                status_code=404,
                detail="No highlights found for this page and session"
            )
        
        # Return as base64 data URL
        return {
            "document_id": clean_document_id,
            "page_number": page_number,
            "query_session_id": query_session_id,
            "highlighted_image_url": highlighted_image
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate highlighted page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.get("/documents/{document_id}/highlights/sessions")
async def get_highlight_sessions(
    request: Request,
    document_id: str,
    author: Optional[str] = Query(None, description="Filter by author"),
    include_expired: bool = Query(False, description="Include expired sessions")
):
    """Get all highlight sessions for a document"""
    try:
        clean_document_id = validate_document_id(document_id)
        session_service = request.app.state.session_service
        
        if not session_service:
            raise HTTPException(status_code=503, detail="Session service not available")
        
        sessions = session_service.get_highlight_sessions_for_document(
            clean_document_id,
            include_expired=include_expired
        )
        
        # Filter by author if requested
        if author:
            clean_author = validate_author(author)
            sessions = [s for s in sessions if s['user'] == clean_author]
        
        return {
            "document_id": clean_document_id,
            "highlight_sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get highlight sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.get("/documents/{document_id}/highlights/active")
async def get_user_active_highlights(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User to get highlights for")
):
    """Get user's active highlight session"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        session_service = request.app.state.session_service
        
        if not session_service:
            raise HTTPException(status_code=503, detail="Session service not available")
        
        active_session = session_service.get_user_active_highlights(
            clean_document_id,
            clean_author
        )
        
        if not active_session:
            return {
                "document_id": clean_document_id,
                "author": clean_author,
                "active_session": None,
                "message": "No active highlight session"
            }
        
        return {
            "document_id": clean_document_id,
            "author": clean_author,
            "active_session": active_session
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active highlights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.post("/documents/{document_id}/highlights/activate")
async def activate_highlight_session(
    request: Request,
    document_id: str,
    query_session_id: str = Query(..., description="Session to activate"),
    author: str = Query(..., description="User activating the session")
):
    """Activate a specific highlight session for a user"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        session_service = request.app.state.session_service
        
        if not session_service:
            raise HTTPException(status_code=503, detail="Session service not available")
        
        success = session_service.set_user_active_session(
            clean_document_id,
            clean_author,
            query_session_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Highlight session not found"
            )
        
        return {
            "status": "success",
            "document_id": clean_document_id,
            "author": clean_author,
            "active_session_id": query_session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.delete("/documents/{document_id}/highlights/clear")
async def clear_user_highlights(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User to clear highlights for")
):
    """Clear user's active highlights"""
    try:
        clean_document_id = validate_document_id(document_id)
        clean_author = validate_author(author)
        
        storage_service = request.app.state.storage_service
        session_service = request.app.state.session_service
        
        if not storage_service or not session_service:
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Get user's active session
        active_session = session_service.get_user_active_highlights(
            clean_document_id,
            clean_author
        )
        
        if not active_session:
            return {
                "status": "success",
                "message": "No active highlights to clear"
            }
        
        # Clear from annotations storage
        query_session_id = active_session['query_session_id']
        
        # Load annotations
        annotations_blob = f"{clean_document_id}_annotations.json"
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            annotations = json.loads(annotations_text)
        except:
            annotations = []
        
        # Remove highlights from this session
        filtered_annotations = [
            ann for ann in annotations
            if ann.get('query_session_id') != query_session_id
        ]
        
        # Save back
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob,
            data=json.dumps(filtered_annotations, indent=2).encode('utf-8')
        )
        
        # Clear from session
        session_service.set_user_active_session(clean_document_id, clean_author, "")
        
        return {
            "status": "success",
            "message": f"Cleared highlights from session {query_session_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear highlights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- CHAT HISTORY ENDPOINTS ---

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

# --- NOTE MANAGEMENT ENDPOINTS ---

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
        
        # Check note limit
        if len(all_notes) >= settings.MAX_NOTES_PER_DOCUMENT:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum number of notes ({settings.MAX_NOTES_PER_DOCUMENT}) reached"
            )
        
        # Create new note
        note = Note(
            note_id=str(uuid.uuid4())[:8],
            document_id=clean_document_id,
            text=note_data.text.strip(),
            note_type=note_data.note_type,
            author=clean_author,
            author_trade=note_data.impacts_trades[0] if note_data.impacts_trades else None,
            impacts_trades=note_data.impacts_trades,
            priority=note_data.priority,
            is_private=note_data.is_private,
            timestamp=datetime.utcnow().isoformat() + "Z",
            char_count=len(note_data.text.strip()),
            status="open"
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
    """Get notes visible to the user"""
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
                    
                if update_data.impacts_trades is not None:
                    note_dict["impacts_trades"] = update_data.impacts_trades
                    
                if update_data.priority is not None:
                    note_dict["priority"] = update_data.priority
                    
                if update_data.status is not None:
                    note_dict["status"] = update_data.status
                
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

# --- SESSION MANAGEMENT ENDPOINTS ---

@blueprint_router.get("/documents/{document_id}/session")
async def get_document_session(request: Request, document_id: str):
    """Get session information for a document"""
    try:
        clean_document_id = validate_document_id(document_id)
        session_service = request.app.state.session_service
        
        if not session_service:
            raise HTTPException(status_code=503, detail="Session service not available")
        
        session_info = session_service.get_session_info(clean_document_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        if not admin_token or admin_token != settings.ADMIN_SECRET_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid admin token")
        
        clean_document_id = validate_document_id(document_id)
        storage_service = request.app.state.storage_service
        session_service = request.app.state.session_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service not available")
        
        # Delete from storage
        deleted_counts = await storage_service.delete_document_files(clean_document_id)
        
        if deleted_counts['total'] == 0:
            raise HTTPException(status_code=404, detail=f"Document '{clean_document_id}' not found")
        
        # Delete from session if exists
        if session_service:
            session_service.delete_session(clean_document_id)
        
        return {
            "status": "deleted",
            "document_id": clean_document_id,
            "files_deleted": deleted_counts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@blueprint_router.get("/admin/sessions/stats")
async def get_session_statistics(
    request: Request,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """Get session service statistics (admin only)"""
    try:
        # Verify admin token
        if not admin_token or admin_token != settings.ADMIN_SECRET_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid admin token")
        
        session_service = request.app.state.session_service
        
        if not session_service:
            raise HTTPException(status_code=503, detail="Session service not available")
        
        return session_service.get_session_statistics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.post("/admin/cleanup/expired")
async def cleanup_expired_highlights(
    request: Request,
    admin_token: str = Header(None, alias="X-Admin-Token")
):
    """Clean up expired highlights (admin only)"""
    try:
        # Verify admin token
        if not admin_token or admin_token != settings.ADMIN_SECRET_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid admin token")
        
        session_service = request.app.state.session_service
        ai_service = request.app.state.ai_service
        storage_service = request.app.state.storage_service
        
        if not all([session_service, ai_service, storage_service]):
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Clean up in session service
        session_results = await session_service.cleanup_all_expired()
        
        # Clean up in storage
        storage_count = await ai_service.cleanup_expired_highlights(storage_service)
        
        return {
            "status": "success",
            "session_cleanup": session_results,
            "storage_cleanup": {
                "highlights_removed": storage_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))
