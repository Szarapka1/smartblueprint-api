# app/api/routes/blueprint_routes.py - TRUE ASYNC VERSION

"""
Blueprint Analysis Routes - Fixed for true asynchronous upload and processing
"""

import traceback
import uuid
import json
import logging
import os
import re
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Header, status, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

# Core imports
from app.core.config import get_settings

# Schema imports
from app.models.schemas import (
    ChatRequest, ChatResponse, DocumentUploadResponse, DocumentInfoResponse,
    DocumentListResponse, SuccessResponse, ErrorResponse, NoteSuggestion,
    QuickNoteCreate, UserPreferences, Note, VisualHighlight
)

# Initialize router and settings
blueprint_router = APIRouter(
    tags=["Blueprint Analysis"],
    responses={
        503: {"description": "Service temporarily unavailable"},
        500: {"description": "Internal server error"}
    }
)

settings = get_settings()
logger = logging.getLogger(__name__)

# ===== UTILITY FUNCTIONS =====

def validate_document_id(document_id: str) -> str:
    """
    Validate and sanitize document ID to prevent injection attacks.
    Returns cleaned document ID.
    """
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Document ID must be a non-empty string"
        )
    
    # Remove any potentially dangerous characters
    # Allow only alphanumeric, underscore, and hyphen
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip())
    
    # Remove leading/trailing underscores
    clean_id = clean_id.strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Document ID must be at least 3 characters long after sanitization"
        )
    
    if len(clean_id) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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

# ===== BACKGROUND PROCESSING FUNCTION =====

async def process_pdf_in_background(
    session_id: str,
    contents: bytes,
    clean_filename: str,
    author: str,
    storage_service,
    pdf_service,
    session_service
):
    """
    Process PDF in the background after returning response to user.
    This function runs after the HTTP response is sent.
    """
    try:
        logger.info(f"🔄 Starting background processing for document: {session_id}")
        
        # Mark as processing in storage
        processing_status = {
            'document_id': session_id,
            'status': 'processing',
            'started_at': datetime.utcnow().isoformat() + 'Z',
            'filename': clean_filename,
            'author': author
        }
        
        await storage_service.upload_file(
            container_name=getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache'),
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Process the PDF
        await pdf_service.process_and_cache_pdf(
            session_id=session_id,
            pdf_bytes=contents,
            storage_service=storage_service
        )
        
        # Update status to ready
        processing_status['status'] = 'ready'
        processing_status['completed_at'] = datetime.utcnow().isoformat() + 'Z'
        
        # Get metadata for page count
        try:
            metadata_blob = f"{session_id}_metadata.json"
            metadata_text = await storage_service.download_blob_as_text(
                container_name=getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache'),
                blob_name=metadata_blob
            )
            metadata = json.loads(metadata_text)
            processing_status['pages_processed'] = metadata.get('page_count', 0)
            processing_status['total_pages'] = metadata.get('total_pages', 0)
        except:
            pass
        
        await storage_service.upload_file(
            container_name=getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache'),
            blob_name=f"{session_id}_status.json",
            data=json.dumps(processing_status).encode('utf-8')
        )
        
        # Update session if available
        if session_service and hasattr(session_service, 'update_session_metadata'):
            try:
                session_service.update_session_metadata(
                    document_id=session_id,
                    metadata={
                        'processing_complete': True,
                        'status': 'ready',
                        'pages_processed': processing_status.get('pages_processed', 0)
                    }
                )
            except Exception as e:
                logger.warning(f"Session update failed: {e}")
        
        logger.info(f"✅ Background processing completed for: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Background processing failed for {session_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Save error status
        error_status = {
            'document_id': session_id,
            'status': 'error',
            'error': str(e),
            'failed_at': datetime.utcnow().isoformat() + 'Z',
            'filename': clean_filename,
            'author': author
        }
        
        try:
            await storage_service.upload_file(
                container_name=getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache'),
                blob_name=f"{session_id}_status.json",
                data=json.dumps(error_status).encode('utf-8')
            )
            
            # Also save detailed error info
            await storage_service.upload_file(
                container_name=getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache'),
                blob_name=f"{session_id}_error.json",
                data=json.dumps({
                    'document_id': session_id,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }).encode('utf-8')
            )
        except Exception as storage_error:
            logger.error(f"Failed to save error status: {storage_error}")

# ===== MAIN ROUTES =====

@blueprint_router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a blueprint PDF",
    description="Upload a construction blueprint PDF for AI analysis. Supports files up to 60MB."
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    author: Optional[str] = Form(default="Anonymous", description="Name of the person uploading"),
    trade: Optional[str] = Form(default=None, description="Trade/discipline associated with the document")
):
    """
    Upload a PDF document for processing - TRUE ASYNC VERSION
    Returns immediately with processing status, processes in background
    """
    
    # Validate request
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file provided"
        )
    
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # Read file content
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file"
        )
    
    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    max_size_mb = getattr(settings, 'MAX_FILE_SIZE_MB', 60)
    
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum size is {max_size_mb}MB."
        )
    
    # Validate it's a valid PDF
    if not contents.startswith(b'%PDF'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format"
        )
    
    # Sanitize filename
    clean_filename = sanitize_filename(file.filename)
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Get storage service
        storage_service = getattr(request.app.state, 'storage_service', None)
        if not storage_service:
            logger.error("Storage service not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="Storage service is not available. Please try again later."
            )
        
        logger.info(f"📤 Uploading document: {clean_filename} ({file_size_mb:.1f}MB)")
        logger.info(f"   Session ID: {session_id}")
        logger.info(f"   Author: {author}")
        if trade:
            logger.info(f"   Trade: {trade}")
        
        # Upload original PDF to main container - this is quick
        pdf_blob_name = f"{session_id}.pdf"
        
        await storage_service.upload_file(
            container_name=getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs'),
            blob_name=pdf_blob_name,
            data=contents
        )
        
        logger.info(f"✅ PDF uploaded successfully to blob: {pdf_blob_name}")
        
        # Create initial status file
        initial_status = {
            'document_id': session_id,
            'status': 'processing',
            'filename': clean_filename,
            'author': author,
            'trade': trade,
            'file_size_mb': round(file_size_mb, 2),
            'uploaded_at': datetime.utcnow().isoformat() + 'Z',
            'estimated_processing_time': max(30, int(file_size_mb * 3))  # Rough estimate: 3 seconds per MB
        }
        
        await storage_service.upload_file(
            container_name=getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache'),
            blob_name=f"{session_id}_status.json",
            data=json.dumps(initial_status).encode('utf-8')
        )
        
        # Create session if service available
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service:
            try:
                if hasattr(session_service, 'create_session'):
                    session_service.create_session(
                        document_id=session_id,
                        original_filename=clean_filename
                    )
                    logger.info("✅ Created session for document")
            except Exception as e:
                logger.warning(f"Session creation failed (non-critical): {e}")
        
        # Get PDF service
        pdf_service = getattr(request.app.state, 'pdf_service', None)
        if not pdf_service:
            logger.warning("⚠️  PDF service not available - upload only mode")
            
            return DocumentUploadResponse(
                document_id=session_id,
                filename=clean_filename,
                status="uploaded",
                message="Document uploaded successfully. Processing service unavailable.",
                file_size_mb=round(file_size_mb, 2)
            )
        
        # Add background task for PDF processing
        background_tasks.add_task(
            process_pdf_in_background,
            session_id=session_id,
            contents=contents,
            clean_filename=clean_filename,
            author=author,
            storage_service=storage_service,
            pdf_service=pdf_service,
            session_service=session_service
        )
        
        # Return immediately with processing status
        logger.info(f"📋 Returning processing status for: {session_id}")
        
        return DocumentUploadResponse(
            document_id=session_id,
            filename=clean_filename,
            status="processing",
            message=f"Document uploaded successfully! Processing will take approximately {initial_status['estimated_processing_time']} seconds.",
            file_size_mb=round(file_size_mb, 2),
            estimated_time=initial_status['estimated_processing_time']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Upload failed: {str(e)}"
        )

@blueprint_router.get(
    "/documents/{document_id}/status",
    summary="Check document processing status",
    description="Check if a document has finished processing"
)
async def check_document_status(
    request: Request,
    document_id: str
):
    """
    Check the processing status of a document.
    Returns current status: processing, ready, or error
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
        
        # Check for status file
        status_blob = f"{clean_document_id}_status.json"
        if await storage_service.blob_exists(cache_container, status_blob):
            status_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=status_blob
            )
            status_data = json.loads(status_text)
            
            return {
                "document_id": clean_document_id,
                "status": status_data.get('status', 'unknown'),
                "message": status_data.get('message', ''),
                "pages_processed": status_data.get('pages_processed'),
                "total_pages": status_data.get('total_pages'),
                "error": status_data.get('error'),
                "uploaded_at": status_data.get('uploaded_at'),
                "completed_at": status_data.get('completed_at'),
                "estimated_time_remaining": status_data.get('estimated_time_remaining')
            }
        
        # Fallback: check if context exists (old method)
        context_blob = f"{clean_document_id}_context.txt"
        if await storage_service.blob_exists(cache_container, context_blob):
            return {
                "document_id": clean_document_id,
                "status": "ready",
                "message": "Document is ready for analysis"
            }
        
        # Check if PDF exists
        main_container = getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs')
        if await storage_service.blob_exists(main_container, f"{clean_document_id}.pdf"):
            return {
                "document_id": clean_document_id,
                "status": "processing",
                "message": "Document is being processed"
            }
        
        return {
            "document_id": clean_document_id,
            "status": "not_found",
            "message": "Document not found"
        }
        
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check status: {str(e)}"
        )

@blueprint_router.get("/documents/{document_id}/download")
async def download_document_pdf(
    request: Request,
    document_id: str
):
    """
    Download the original PDF file for viewing.
    Required for the frontend "Load Document" functionality.
    """
    clean_document_id = validate_document_id(document_id)
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        logger.info(f"📥 Downloading PDF for document: {clean_document_id}")
        
        # Check if PDF exists first
        container_name = getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs')
        blob_name = f"{clean_document_id}.pdf"
        
        if not await storage_service.blob_exists(container_name, blob_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF not found for document {clean_document_id}"
            )
        
        # Get PDF from main container where original files are stored
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=container_name,
            blob_name=blob_name
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download PDF {clean_document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Failed to download PDF: {str(e)}"
        )

@blueprint_router.post(
    "/documents/{document_id}/chat",
    response_model=ChatResponse,
    summary="Chat with a blueprint",
    description="Send a question about an uploaded blueprint and receive AI analysis with visual highlights"
)
async def chat_with_document(
    request: Request,
    document_id: str,
    chat_request: ChatRequest
):
    """
    Chat with an uploaded document using AI analysis with note suggestions.
    """
    
    # Validate document ID
    clean_document_id = validate_document_id(document_id)
    
    # Validate request
    if not chat_request.prompt or len(chat_request.prompt.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Prompt cannot be empty"
        )
    
    # Get services
    ai_service = getattr(request.app.state, 'ai_service', None)
    storage_service = getattr(request.app.state, 'storage_service', None)
    
    if not ai_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI service is not available. Please try again later."
        )
    
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service is not available. Please try again later."
        )
    
    try:
        # Log chat request
        logger.info(f"💬 Chat request for document: {clean_document_id}")
        logger.info(f"   Author: {chat_request.author}")
        logger.info(f"   Prompt: {chat_request.prompt[:100]}...")
        
        # Check document status first
        cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
        
        # Check status file
        status_blob = f"{clean_document_id}_status.json"
        status_data = None
        
        if await storage_service.blob_exists(cache_container, status_blob):
            try:
                status_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=status_blob
                )
                status_data = json.loads(status_text)
            except:
                pass
        
        # Determine actual status
        if status_data:
            if status_data.get('status') == 'processing':
                estimated_time = status_data.get('estimated_time_remaining', 30)
                raise HTTPException(
                    status_code=status.HTTP_425_TOO_EARLY,
                    detail=f"Document is still being processed. Please try again in {estimated_time} seconds."
                )
            elif status_data.get('status') == 'error':
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Document processing failed: {status_data.get('error', 'Unknown error')}"
                )
        
        # Check if context exists (document is ready)
        context_blob = f"{clean_document_id}_context.txt"
        context_exists = await storage_service.blob_exists(
            container_name=cache_container,
            blob_name=context_blob
        )
        
        if not context_exists:
            # Check if PDF exists
            main_container = getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs')
            pdf_exists = await storage_service.blob_exists(
                container_name=main_container,
                blob_name=f"{clean_document_id}.pdf"
            )
            
            if pdf_exists:
                raise HTTPException(
                    status_code=status.HTTP_425_TOO_EARLY,
                    detail="Document is still being processed. Please try again in a few seconds."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{clean_document_id}' not found"
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
        
        # Record activity in session if available
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service and hasattr(session_service, 'record_chat_activity'):
            try:
                session_service.record_chat_activity(
                    document_id=clean_document_id,
                    user=chat_request.author
                )
                
                # Update highlight session if created
                if (ai_result.get("query_session_id") and 
                    ai_result.get("visual_highlights") and 
                    hasattr(session_service, 'update_highlight_session')):
                    
                    element_types = list(set(
                        h.element_type for h in ai_result["visual_highlights"]
                    ))
                    
                    session_service.update_highlight_session(
                        document_id=clean_document_id,
                        query_session_id=ai_result["query_session_id"],
                        pages_with_highlights=ai_result.get("all_highlight_pages", {}),
                        element_types=element_types,
                        total_highlights=ai_result.get("total_highlights", 0)
                    )
            except Exception as e:
                logger.warning(f"Session update failed (non-critical): {e}")
        
        # Build response
        response = ChatResponse(
            session_id=clean_document_id,
            ai_response=ai_result.get("ai_response", "I apologize, but I was unable to analyze the blueprint."),
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
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
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
        notes_blob = f"{clean_document_id}_notes.json"
        
        # Load existing notes
        try:
            notes_text = await storage_service.download_blob_as_text(
                container_name=cache_container,
                blob_name=notes_blob
            )
            all_notes = json.loads(notes_text)
        except:
            all_notes = []
        
        # Check note limit
        max_notes = getattr(settings, 'MAX_NOTES_PER_DOCUMENT', 500)
        user_notes = [n for n in all_notes if n.get('author') == quick_note.author]
        if len(user_notes) >= max_notes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Note limit reached ({max_notes} notes per document)"
            )
        
        # Create note with all metadata
        new_note = {
            "note_id": str(uuid.uuid4())[:8],
            "document_id": clean_document_id,
            "text": quick_note.text,
            "note_type": quick_note.note_type,
            "author": quick_note.author,
            "impacts_trades": quick_note.impacts_trades or [],
            "priority": quick_note.priority,
            "is_private": quick_note.is_private,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "char_count": len(quick_note.text),
            "status": "open",
            "ai_suggested": quick_note.ai_suggested,
            "suggestion_confidence": quick_note.suggestion_confidence,
            "related_query_sessions": quick_note.related_query_sessions or [],
            "related_element_ids": quick_note.related_highlights or [],
            "source_pages": quick_note.source_pages or []
        }
        
        # Add to notes
        all_notes.append(new_note)
        
        # Save back
        await storage_service.upload_file(
            container_name=cache_container,
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@blueprint_router.get(
    "/documents/{document_id}/info",
    response_model=DocumentInfoResponse,
    summary="Get document information",
    description="Get processing status and metadata for a specific document"
)
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
    
    storage_service = getattr(request.app.state, 'storage_service', None)
    if not storage_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Storage service unavailable"
        )
    
    try:
        # Check containers
        main_container = getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs')
        cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
        
        # Check for status file first
        status_blob = f"{clean_document_id}_status.json"
        status_data = None
        
        if await storage_service.blob_exists(cache_container, status_blob):
            try:
                status_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
                    blob_name=status_blob
                )
                status_data = json.loads(status_text)
            except:
                pass
        
        # Check if document exists
        pdf_exists = await storage_service.blob_exists(
            container_name=main_container,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        context_exists = await storage_service.blob_exists(
            container_name=cache_container,
            blob_name=f"{clean_document_id}_context.txt"
        )
        
        if not pdf_exists and not context_exists:
            return DocumentInfoResponse(
                document_id=clean_document_id,
                status="not_found",
                message="Document not found",
                exists=False
            )
        
        # Use status data if available
        if status_data:
            if status_data.get('status') == 'error':
                return DocumentInfoResponse(
                    document_id=clean_document_id,
                    status="error",
                    message=f"Processing failed: {status_data.get('error', 'Unknown error')}",
                    exists=True
                )
            elif status_data.get('status') == 'processing':
                return DocumentInfoResponse(
                    document_id=clean_document_id,
                    status="processing",
                    message="Document is being processed",
                    exists=True,
                    metadata={
                        "estimated_time_remaining": status_data.get('estimated_time_remaining'),
                        "uploaded_at": status_data.get('uploaded_at')
                    }
                )
        
        # Get metadata if processed
        metadata = None
        if context_exists:
            try:
                metadata_blob = f"{clean_document_id}_metadata.json"
                metadata_text = await storage_service.download_blob_as_text(
                    container_name=cache_container,
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
                container_name=cache_container,
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
        session_service = getattr(request.app.state, 'session_service', None)
        if session_service and hasattr(session_service, 'get_session_info'):
            try:
                session_info = session_service.get_session_info(clean_document_id)
            except:
                pass
        
        # Determine status
        if context_exists and metadata:
            status_value = "ready"
            message = f"Document ready for analysis. {metadata.get('page_count', 0)} pages processed."
        elif pdf_exists:
            status_value = "processing"
            message = "Document uploaded, processing in progress"
        else:
            status_value = "partial"
            message = "Document partially processed"
        
        return DocumentInfoResponse(
            document_id=clean_document_id,
            status=status_value,
            message=message,
            exists=True,
            metadata={
                "page_count": metadata.get('page_count') if metadata else None,
                "total_pages": metadata.get('total_pages') if metadata else None,
                "processing_time": metadata.get('processing_time') if metadata else None,
                "grid_systems_detected": metadata.get('grid_systems_detected', 0) if metadata else None,
                "has_text": metadata.get('extraction_summary', {}).get('has_text', False) if metadata else None,
                "file_size_mb": metadata.get('file_size_mb') if metadata else None
            },
            total_published_notes=published_notes,
            active_collaborators=len(active_collaborators),
            recent_public_activity=published_notes > 0,
            session_info=session_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get document info: {str(e)}"
        )

@blueprint_router.get("/health/services")
async def check_services(request: Request):
    """
    Check which services are currently available.
    Useful for debugging and monitoring service health.
    """
    def check_service(service_name: str) -> Dict[str, Any]:
        """Check if a service exists and is functional"""
        service = getattr(request.app.state, service_name, None)
        if service is None:
            return {"available": False, "status": "not_initialized"}
        
        # Try to call a method to verify it's working
        try:
            if hasattr(service, 'get_connection_info'):
                info = service.get_connection_info()
                return {"available": True, "status": "healthy", "info": info}
            elif hasattr(service, 'get_processing_stats'):
                stats = service.get_processing_stats()
                return {"available": True, "status": "healthy", "stats": stats}
            elif hasattr(service, 'get_professional_capabilities'):
                caps = service.get_professional_capabilities()
                return {"available": True, "status": "healthy", "capabilities": caps}
            else:
                return {"available": True, "status": "initialized"}
        except Exception as e:
            return {"available": True, "status": "error", "error": str(e)}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "storage": check_service('storage_service'),
            "ai": check_service('ai_service'),
            "pdf": check_service('pdf_service'),
            "session": check_service('session_service')
        }
    }

# ===== HELPER FUNCTIONS =====

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
        cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
        chat_blob = f"{document_id}_all_chats.json"
        
        # Load existing chat history
        try:
            chat_data = await storage_service.download_blob_as_text(
                container_name=cache_container,
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
        max_chats = getattr(settings, 'MAX_CHAT_LOGS', 100)
        if len(chat_history) > max_chats:
            chat_history = chat_history[-max_chats:]
        
        # Save updated history
        await storage_service.upload_file(
            container_name=cache_container,
            blob_name=chat_blob,
            data=json.dumps(chat_history, indent=2).encode('utf-8')
        )
        
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")
