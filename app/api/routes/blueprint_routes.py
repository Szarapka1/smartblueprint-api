# app/api/routes/blueprint_routes.py - ENHANCED WITH NOTE SUGGESTIONS

import traceback
import uuid
import json
import logging
import os
import re
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app.core.config import get_settings
from app.models.schemas import (
    ChatRequest, ChatResponse, DocumentUploadResponse, DocumentInfoResponse,
    DocumentListResponse, SuccessResponse, ErrorResponse, NoteSuggestion,
    QuickNoteCreate, UserPreferences, Note
)

# Initialize router and settings
blueprint_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def validate_document_id(document_id: str) -> str:
    """
    Validate and sanitize document ID to prevent injection attacks.
    Returns cleaned document ID.
    """
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be a non-empty string"
        )
    
    # Remove any potentially dangerous characters
    # Allow only alphanumeric, underscore, and hyphen
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    
    # Remove leading/trailing underscores
    clean_id = clean_id.strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters long after sanitization"
        )
    
    if len(clean_id) > 50:
        raise HTTPException(
            status_code=400,
            detail="Document ID must be 50 characters or less"
        )
    
    return clean_id

def validate_file_type(filename: str) -> bool:
    """Validate that uploaded file is a PDF"""
    if not filename:
        return False
    return filename.lower().endswith('.pdf')

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    # Remove any path components
    filename = os.path.basename(filename)
    # Remove special characters except dots and hyphens
    clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return clean_name

# --- Main Routes ---

@blueprint_router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    author: Optional[str] = Form(None),
    trade: Optional[str] = Form(None)
):
    """
    Upload a PDF document for processing.
    
    This endpoint:
    1. Validates the uploaded file
    2. Generates a unique session ID
    3. Saves the PDF to Azure Storage
    4. Initiates async processing of the PDF
    5. Returns immediately with session ID
    
    The PDF processing happens in the background and includes:
    - Text extraction
    - Page rendering at multiple resolutions
    - Metadata extraction
    - Index creation
    """
    
    # Validate request
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # Validate file size
    file_size_mb = 0
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Sanitize filename
    clean_filename = sanitize_filename(file.filename)
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Get services from app state
        storage_service = request.app.state.storage_service
        pdf_service = request.app.state.pdf_service
        session_service = request.app.state.session_service
        
        if not storage_service:
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        
        logger.info(f"📤 Uploading document: {clean_filename} ({file_size_mb:.1f}MB)")
        logger.info(f"   Session ID: {session_id}")
        if author:
            logger.info(f"   Author: {author}")
        if trade:
            logger.info(f"   Trade: {trade}")
        
        # Upload original PDF to main container
        pdf_blob_name = f"{session_id}.pdf"
        await storage_service.upload_file(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=pdf_blob_name,
            data=contents
        )
        
        logger.info(f"✅ PDF uploaded successfully to blob: {pdf_blob_name}")
        
        # Create session if session service is available
        if session_service:
            try:
                session_service.create_session(
                    document_id=session_id,
                    original_filename=clean_filename
                )
                logger.info(f"✅ Created session for document")
            except Exception as e:
                logger.warning(f"Session creation failed (non-critical): {e}")
        
        # Process PDF
        if pdf_service:
            try:
                logger.info(f"🔄 Starting PDF processing...")
                await pdf_service.process_and_cache_pdf(
                    session_id=session_id,
                    pdf_bytes=contents,
                    storage_service=storage_service
                )
                
                # Load metadata to get processing results
                metadata_blob = f"{session_id}_metadata.json"
                metadata_text = await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=metadata_blob
                )
                metadata = json.loads(metadata_text)
                
                # Update session with metadata
                if session_service:
                    session_service.update_session_metadata(
                        document_id=session_id,
                        metadata={
                            'page_count': metadata.get('page_count'),
                            'total_pages': metadata.get('total_pages'),
                            'has_text': metadata['extraction_summary'].get('has_text', False),
                            'grid_systems_detected': metadata['optimization_stats'].get('grid_systems_detected', 0)
                        }
                    )
                
                logger.info(f"✅ PDF processing completed successfully")
                logger.info(f"   Pages processed: {metadata.get('page_count')}")
                logger.info(f"   Grid systems: {metadata['optimization_stats'].get('grid_systems_detected', 0)}")
                
                # Return detailed response
                return DocumentUploadResponse(
                    document_id=session_id,
                    filename=clean_filename,
                    status="success",
                    message=f"Document uploaded and processed successfully. {metadata.get('page_count')} pages ready for analysis.",
                    file_size_mb=round(file_size_mb, 2),
                    pages_processed=metadata.get('page_count'),
                    grid_systems_detected=metadata['optimization_stats'].get('grid_systems_detected', 0),
                    drawing_types_found=list(set(
                        page.get('drawing_type') 
                        for page in metadata.get('page_details', []) 
                        if page.get('drawing_type')
                    ))
                )
                
            except Exception as e:
                logger.error(f"❌ PDF processing failed: {e}")
                # Don't fail the upload - document is still available
                return DocumentUploadResponse(
                    document_id=session_id,
                    filename=clean_filename,
                    status="partial",
                    message=f"Document uploaded but processing encountered issues: {str(e)}",
                    file_size_mb=round(file_size_mb, 2)
                )
        else:
            # PDF service not available
            return DocumentUploadResponse(
                document_id=session_id,
                filename=clean_filename,
                status="uploaded",
                message="Document uploaded successfully. Processing service unavailable.",
                file_size_mb=round(file_size_mb, 2)
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@blueprint_router.get("/documents/{document_id}/download")
async def download_document_pdf(
    request: Request,
    document_id: str
):
    """
    Download the original PDF file for viewing.
    Required for the "Load Document" functionality.
    """
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        logger.info(f"📥 Downloading PDF for document: {clean_document_id}")
        
        # Get PDF from main container where original files are stored
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        logger.info(f"✅ PDF downloaded successfully: {len(pdf_bytes)} bytes")
        
        # Return PDF with proper headers for inline viewing
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={clean_document_id}.pdf",
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to download PDF {clean_document_id}: {e}")
        raise HTTPException(
            status_code=404, 
            detail=f"PDF file not found for document {clean_document_id}"
        )

@blueprint_router.post("/documents/{document_id}/chat", response_model=ChatResponse)
async def chat_with_document(
    request: Request,
    document_id: str,
    chat_request: ChatRequest
):
    """
    Chat with an uploaded document using AI analysis with note suggestions.
    
    This endpoint:
    1. Validates the document exists
    2. Sends the query to the AI service
    3. Returns AI response with optional visual highlights
    4. Suggests creating notes for important findings
    5. Tracks the chat in session history
    """
    
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    # Validate request
    try:
        # Additional validation if needed
        if not chat_request.prompt or len(chat_request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Get services
    ai_service = request.app.state.ai_service
    storage_service = request.app.state.storage_service
    session_service = request.app.state.session_service
    
    if not ai_service or not storage_service:
        raise HTTPException(
            status_code=503, 
            detail="Required services are not available"
        )
    
    try:
        # Log chat request
        logger.info(f"💬 Chat request for document: {clean_document_id}")
        logger.info(f"   Author: {chat_request.author}")
        logger.info(f"   Prompt: {chat_request.prompt[:100]}...")
        
        # Check if document exists
        context_blob = f"{clean_document_id}_context.txt"
        if not await storage_service.blob_exists(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=context_blob
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Document '{clean_document_id}' not found or not processed yet"
            )
        
        # Get AI response with note suggestions
        ai_result = await ai_service.get_ai_response(
            prompt=chat_request.prompt,
            document_id=clean_document_id,
            storage_service=storage_service,
            author=chat_request.author,
            current_page=chat_request.current_page,
            request_highlights=True,  # Always request highlights for better analysis
            reference_previous=chat_request.reference_previous,
            preserve_existing=chat_request.preserve_existing,
            show_trade_info=chat_request.show_trade_info,
            detect_conflicts=chat_request.detect_conflicts,
            auto_suggest_notes=chat_request.auto_suggest_notes,
            note_suggestion_threshold=chat_request.note_suggestion_threshold
        )
        
        # Save chat to history
        await save_chat_to_history(
            document_id=clean_document_id,
            author=chat_request.author,
            prompt=chat_request.prompt,
            response=ai_result.get("ai_response", ""),
            storage_service=storage_service,
            has_highlights=bool(ai_result.get("visual_highlights")),
            note_suggested=bool(ai_result.get("note_suggestion"))
        )
        
        # Record activity in session
        if session_service:
            try:
                session_service.record_chat_activity(
                    document_id=clean_document_id,
                    user=chat_request.author
                )
                
                # If highlights were created, update session
                if ai_result.get("query_session_id") and ai_result.get("visual_highlights"):
                    session_service.update_highlight_session(
                        document_id=clean_document_id,
                        query_session_id=ai_result["query_session_id"],
                        pages_with_highlights=ai_result.get("all_highlight_pages", {}),
                        element_types=list(set(h.element_type for h in ai_result["visual_highlights"])),
                        total_highlights=ai_result.get("total_highlights", 0)
                    )
            except Exception as e:
                logger.warning(f"Session update failed (non-critical): {e}")
        
        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_result.get("ai_response", ""),
            visual_highlights=ai_result.get("visual_highlights"),
            current_page=chat_request.current_page,
            query_session_id=ai_result.get("query_session_id"),
            all_highlight_pages=ai_result.get("all_highlight_pages"),
            trade_summary=ai_result.get("trade_summary"),
            note_suggestion=ai_result.get("note_suggestion")
        )
        
        logger.info(f"✅ Chat response generated successfully")
        if ai_result.get("visual_highlights"):
            logger.info(f"   Highlights created: {len(ai_result['visual_highlights'])}")
        if ai_result.get("note_suggestion"):
            logger.info(f"   Note suggested: {ai_result['note_suggestion'].reason}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process chat: {str(e)}"
        )

@blueprint_router.post("/documents/{document_id}/notes/quick-create")
async def quick_create_note_from_suggestion(
    request: Request,
    document_id: str,
    quick_note: QuickNoteCreate
):
    """
    Quick endpoint to create a note from AI suggestion with one click.
    """
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    # Get storage service
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load existing notes
        notes_blob = f"{clean_document_id}_notes.json"
        try:
            notes_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=notes_blob
            )
            all_notes = json.loads(notes_text)
        except:
            all_notes = []
        
        # Check note limit
        user_notes = [n for n in all_notes if n.get('author') == quick_note.author]
        if len(user_notes) >= settings.MAX_NOTES_PER_DOCUMENT:
            raise HTTPException(
                status_code=400,
                detail=f"Note limit reached ({settings.MAX_NOTES_PER_DOCUMENT} notes per document)"
            )
        
        # Create note
        new_note = {
            "note_id": str(uuid.uuid4())[:8],
            "document_id": clean_document_id,
            "text": quick_note.text,
            "note_type": quick_note.note_type,
            "author": quick_note.author,
            "impacts_trades": quick_note.impacts_trades,
            "priority": quick_note.priority,
            "is_private": quick_note.is_private,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "char_count": len(quick_note.text),
            "status": "open",
            "ai_suggested": quick_note.ai_suggested,
            "suggestion_confidence": quick_note.suggestion_confidence,
            "related_query_sessions": quick_note.related_query_sessions,
            "related_element_ids": quick_note.related_highlights,
            "source_pages": quick_note.source_pages
        }
        
        # Add to notes
        all_notes.append(new_note)
        
        # Save back
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=notes_blob,
            data=json.dumps(all_notes, indent=2).encode('utf-8')
        )
        
        logger.info(f"✅ Quick note created from AI suggestion")
        logger.info(f"   Note ID: {new_note['note_id']}")
        logger.info(f"   Type: {quick_note.note_type}")
        logger.info(f"   Priority: {quick_note.priority}")
        
        return {
            "status": "success",
            "note_id": new_note["note_id"],
            "message": "Note created successfully from AI suggestion"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick note creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.put("/documents/{document_id}/user-preferences")
async def update_user_preferences(
    request: Request,
    document_id: str,
    preferences: UserPreferences,
    author: str = Header(None, alias="X-Author")
):
    """
    Update user preferences for AI behavior (stored per user, not per document).
    """
    if not author:
        raise HTTPException(status_code=400, detail="Author header required")
    
    # This would typically be stored in a user preferences database
    # For now, we'll store in a simple JSON file in blob storage
    storage_service = request.app.state.storage_service
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        prefs_blob = f"user_preferences_{author}.json"
        
        # Save preferences
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=prefs_blob,
            data=json.dumps(preferences.dict(), indent=2).encode('utf-8')
        )
        
        logger.info(f"✅ Updated preferences for user: {author}")
        
        return {
            "status": "success",
            "message": "Preferences updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@blueprint_router.get("/documents/{document_id}/info", response_model=DocumentInfoResponse)
async def get_document_info(request: Request, document_id: str):
    """
    Get information about a specific document.
    
    Returns:
    - Document existence status
    - Processing status
    - Basic metadata (page count, etc.)
    - Public collaboration info (published notes count)
    """
    
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    session_service = request.app.state.session_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Check if document exists
        pdf_exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        context_exists = await storage_service.blob_exists(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not pdf_exists and not context_exists:
            return DocumentInfoResponse(
                document_id=clean_document_id,
                status="not_found",
                message="Document not found",
                exists=False
            )
        
        # Get metadata if processed
        metadata = None
        if context_exists:
            try:
                metadata_blob = f"{clean_document_id}_metadata.json"
                metadata_text = await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=metadata_blob
                )
                metadata = json.loads(metadata_text)
            except:
                pass
        
        # Get public collaboration info
        published_notes = 0
        active_collaborators = set()
        
        try:
            notes_blob = f"{clean_document_id}_notes.json"
            notes_text = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=notes_blob
            )
            all_notes = json.loads(notes_text)
            
            # Count only published notes
            for note in all_notes:
                if not note.get('is_private', True):
                    published_notes += 1
                    active_collaborators.add(note.get('author'))
                    
        except:
            pass
        
        # Get session info if available
        session_info = None
        if session_service:
            session_info = session_service.get_session_info(clean_document_id)
        
        # Determine status
        if context_exists and metadata:
            status = "ready"
            message = f"Document ready for analysis. {metadata.get('page_count', 0)} pages processed."
        elif pdf_exists:
            status = "processing"
            message = "Document uploaded, processing in progress"
        else:
            status = "partial"
            message = "Document partially processed"
        
        return DocumentInfoResponse(
            document_id=clean_document_id,
            status=status,
            message=message,
            exists=True,
            metadata={
                "page_count": metadata.get('page_count') if metadata else None,
                "total_pages": metadata.get('total_pages') if metadata else None,
                "processing_time": metadata.get('processing_time') if metadata else None,
                "grid_systems_detected": metadata.get('optimization_stats', {}).get('grid_systems_detected', 0) if metadata else None,
                "has_text": metadata.get('extraction_summary', {}).get('has_text', False) if metadata else None,
                "file_size_mb": metadata.get('file_size_mb') if metadata else None
            },
            total_published_notes=published_notes,
            active_collaborators=len(active_collaborators),
            recent_public_activity=published_notes > 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@blueprint_router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    author: Optional[str] = None
):
    """
    List all available documents with pagination.
    
    Parameters:
    - limit: Maximum number of documents to return (1-100)
    - offset: Number of documents to skip
    - author: Filter by documents with activity from this author
    
    Returns paginated list of documents with basic metadata.
    """
    
    # Validate parameters
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")
    
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Get all document IDs efficiently
        document_ids = await storage_service.list_document_ids(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME
        )
        
        if not document_ids:
            return DocumentListResponse(
                documents=[],
                total_count=0,
                has_more=False
            )
        
        # Sort by ID (which includes timestamp for UUID-based IDs)
        document_ids.sort(reverse=True)  # Most recent first
        
        # Apply author filter if provided
        filtered_ids = document_ids
        if author:
            # This would require checking each document's activity
            # For now, we'll return all documents
            # In production, you'd want an index for this
            pass
        
        # Apply pagination
        total_count = len(filtered_ids)
        paginated_ids = filtered_ids[offset:offset + limit]
        has_more = (offset + limit) < total_count
        
        # Load metadata for each document in parallel
        documents = []
        for doc_id in paginated_ids:
            try:
                # Get basic metadata
                metadata_blob = f"{doc_id}_metadata.json"
                metadata_text = await storage_service.download_blob_as_text(
                    container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=metadata_blob
                )
                metadata = json.loads(metadata_text)
                
                # Get activity summary
                activity = await get_document_activity_summary(
                    doc_id, storage_service
                )
                
                documents.append({
                    "document_id": doc_id,
                    "filename": metadata.get('document_info', {}).get('title', 'Unknown'),
                    "page_count": metadata.get('page_count', 0),
                    "uploaded_at": metadata.get('processing_timestamp', ''),
                    "status": "ready",
                    "file_size_mb": metadata.get('file_size_mb', 0),
                    "total_chats": activity.get('total_chats', 0),
                    "total_annotations": activity.get('total_annotations', 0),
                    "last_activity": activity.get('last_activity')
                })
                
            except Exception as e:
                # Document might be partially processed
                logger.warning(f"Could not load metadata for {doc_id}: {e}")
                documents.append({
                    "document_id": doc_id,
                    "filename": "Unknown",
                    "status": "processing",
                    "page_count": 0
                })
        
        return DocumentListResponse(
            documents=documents,
            total_count=total_count,
            has_more=has_more,
            filter_applied={"author": author} if author else None
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@blueprint_router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    admin_token: Optional[str] = Header(None, alias="X-Admin-Token")
):
    """
    Delete a document and all associated data.
    
    Requires admin authentication via X-Admin-Token header.
    
    This will delete:
    - Original PDF
    - All processed pages
    - All metadata
    - All annotations
    - All chat history
    - Session data
    """
    
    # Validate admin access
    if not admin_token or admin_token != settings.ADMIN_SECRET_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Admin token required."
        )
    
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    storage_service = request.app.state.storage_service
    session_service = request.app.state.session_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        logger.info(f"🗑️ Starting deletion of document: {clean_document_id}")
        
        # Delete all files associated with the document
        deletion_results = await storage_service.delete_document_files(clean_document_id)
        
        # Delete session if available
        if session_service:
            try:
                session_service.delete_session(clean_document_id)
                logger.info(f"✅ Deleted session data")
            except Exception as e:
                logger.warning(f"Session deletion failed (non-critical): {e}")
        
        logger.info(f"✅ Document deletion complete")
        logger.info(f"   Files deleted: {deletion_results['total']}")
        
        return {
            "status": "success",
            "message": f"Document {clean_document_id} and all associated data deleted successfully",
            "files_deleted": deletion_results
        }
        
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

# --- Helper Functions ---

async def save_chat_to_history(
    document_id: str,
    author: str,
    prompt: str,
    response: str,
    storage_service,
    has_highlights: bool = False,
    note_suggested: bool = False
):
    """Save chat interaction to history"""
    try:
        # Load existing chat history
        chat_blob = f"{document_id}_all_chats.json"
        
        try:
            chat_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chat_blob
            )
            chat_history = json.loads(chat_data)
        except:
            chat_history = []
        
        # Add new chat entry
        chat_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "author": author,
            "prompt": prompt,
            "response": response[:1000],  # Limit response size
            "has_highlights": has_highlights,
            "note_suggested": note_suggested
        }
        
        chat_history.append(chat_entry)
        
        # Keep only recent chats
        if len(chat_history) > settings.MAX_CHAT_LOGS:
            chat_history = chat_history[-settings.MAX_CHAT_LOGS:]
        
        # Save updated history
        await storage_service.upload_file(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=chat_blob,
            data=json.dumps(chat_history, indent=2).encode('utf-8')
        )
        
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")

async def get_document_activity_summary(
    document_id: str,
    storage_service
) -> Dict:
    """Get activity summary for a document"""
    summary = {
        'total_chats': 0,
        'total_annotations': 0,
        'last_activity': None
    }
    
    try:
        # Count chats
        chat_blob = f"{document_id}_all_chats.json"
        try:
            chat_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chat_blob
            )
            chats = json.loads(chat_data)
            summary['total_chats'] = len(chats)
            if chats:
                summary['last_activity'] = chats[-1].get('timestamp')
        except:
            pass
        
        # Count annotations
        ann_blob = f"{document_id}_annotations.json"
        try:
            ann_data = await storage_service.download_blob_as_text(
                container_name=settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=ann_blob
            )
            annotations = json.loads(ann_data)
            summary['total_annotations'] = len(annotations)
            
            # Update last activity if annotation is more recent
            if annotations:
                last_ann_time = max(ann.get('created_at', '') for ann in annotations)
                if not summary['last_activity'] or last_ann_time > summary['last_activity']:
                    summary['last_activity'] = last_ann_time
        except:
            pass
            
    except Exception as e:
        logger.warning(f"Failed to get activity summary: {e}")
    
    return summary
