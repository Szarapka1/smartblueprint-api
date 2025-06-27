# app/api/routes/blueprint_routes.py - COMPLETE REWRITE WITH PROPER SESSION HANDLING

"""
Blueprint Analysis Routes - Main API endpoints for document upload and chat

This module provides the core functionality for:
- Uploading blueprint PDFs
- Chatting with AI about blueprints
- Managing document status and metadata
- Creating visual highlights and annotations
- Suggesting and creating notes from AI analysis
"""

import traceback
import uuid
import json
import logging
import os
import re
import asyncio  # THIS WAS MISSING - NOW ADDED!
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Header, status
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
    file: UploadFile = File(..., description="PDF file to upload"),
    author: Optional[str] = Form(default="Anonymous", description="Name of the person uploading"),
    trade: Optional[str] = Form(default=None, description="Trade/discipline associated with the document")
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
    - Grid detection for coordinate mapping
    - Metadata extraction
    - Index creation for fast searching
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
            detail=f"File too large. Maximum size is {max_size_mb}MB, got {file_size_mb:.1f}MB"
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
        
        # Upload original PDF to main container with retry logic
        pdf_blob_name = f"{session_id}.pdf"
        max_retries = 3
        upload_success = False
        
        for retry in range(max_retries):
            try:
                # Fix the TypeError by removing content_type parameter
                upload_success = await storage_service.upload_file(
                    container_name=getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs'),
                    blob_name=pdf_blob_name,
                    data=contents
                    # REMOVED: content_type='application/pdf'  # This was causing the error
                )
                if upload_success:
                    break
            except Exception as e:
                logger.error(f"Upload attempt {retry + 1} failed: {e}")
                if retry == max_retries - 1:
                    raise
                await asyncio.sleep(1)  # Wait before retry - NOW THIS WILL WORK!
        
        if not upload_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload file to storage after multiple attempts"
            )
        
        logger.info(f"✅ PDF uploaded successfully to blob: {pdf_blob_name}")
        
        # === FIXED SESSION SERVICE HANDLING ===
        session_data = None
        session_created = False
        
        try:
            # Safely check if session service exists
            session_service = None
            if hasattr(request.app.state, 'session_service'):
                session_service = getattr(request.app.state, 'session_service', None)
            
            if session_service is not None:
                try:
                    logger.info("   Creating document session...")
                    # Use the session service method
                    if hasattr(session_service, 'create_session'):
                        session_service.create_session(
                            document_id=session_id,
                            original_filename=clean_filename
                        )
                        session_created = True
                        logger.info("✅ Created session for document")
                    else:
                        logger.warning("Session service exists but missing create_session method")
                except Exception as e:
                    logger.warning(f"Session creation failed (non-critical): {e}")
                    # Continue without session
            else:
                logger.info("⚠️  Session service not available - continuing without session tracking")
        except Exception as e:
            logger.error(f"Session service check failed: {e}")
            # Continue without session
        
        # Process PDF if service is available
        processing_started = False
        processing_error = None
        pages_processed = 0
        grid_systems_detected = 0
        drawing_types_found = []
        
        try:
            pdf_service = getattr(request.app.state, 'pdf_service', None)
            if pdf_service:
                logger.info("🔄 Starting PDF processing...")
                
                # Process PDF synchronously to get immediate results
                await pdf_service.process_and_cache_pdf(
                    session_id=session_id,
                    pdf_bytes=contents,
                    storage_service=storage_service
                )
                
                processing_started = True
                
                # Try to load metadata to get processing results
                try:
                    metadata_blob = f"{session_id}_metadata.json"
                    cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
                    
                    metadata_text = await storage_service.download_blob_as_text(
                        container_name=cache_container,
                        blob_name=metadata_blob
                    )
                    metadata = json.loads(metadata_text)
                    
                    # Extract processing results
                    pages_processed = metadata.get('page_count', 0)
                    grid_systems_detected = metadata.get('optimization_stats', {}).get('grid_systems_detected', 0)
                    
                    # Get unique drawing types
                    drawing_types = set()
                    for page in metadata.get('page_details', []):
                        if page.get('drawing_type'):
                            drawing_types.add(page['drawing_type'])
                    drawing_types_found = list(drawing_types)
                    
                    # Update session with metadata if available
                    if session_service and hasattr(session_service, 'update_session_metadata'):
                        try:
                            session_service.update_session_metadata(
                                document_id=session_id,
                                metadata={
                                    'page_count': pages_processed,
                                    'total_pages': metadata.get('total_pages'),
                                    'has_text': metadata.get('extraction_summary', {}).get('has_text', False),
                                    'grid_systems_detected': grid_systems_detected
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Session metadata update failed: {e}")
                    
                    logger.info(f"✅ PDF processing completed successfully")
                    logger.info(f"   Pages processed: {pages_processed}")
                    logger.info(f"   Grid systems: {grid_systems_detected}")
                    
                except Exception as e:
                    logger.warning(f"Could not load processing metadata: {e}")
                    # Processing happened but metadata not immediately available
                    
            else:
                logger.warning("⚠️  PDF service not available - upload only mode")
                
        except Exception as e:
            processing_error = str(e)
            logger.error(f"❌ PDF processing failed: {e}")
            # Don't fail the upload - document is still available
        
        # Determine response status
        if processing_started and not processing_error and pages_processed > 0:
            status_msg = "success"
            message = f"Document uploaded and processed successfully. {pages_processed} pages ready for analysis."
        elif processing_started and not processing_error:
            status_msg = "processing"
            message = "Document uploaded successfully. Processing in background."
        elif processing_error:
            status_msg = "partial"
            message = f"Document uploaded but processing encountered issues: {processing_error}"
        else:
            status_msg = "uploaded"
            message = "Document uploaded successfully. Processing service unavailable."
        
        # Build response
        response = DocumentUploadResponse(
            document_id=session_id,
            filename=clean_filename,
            status=status_msg,
            message=message,
            file_size_mb=round(file_size_mb, 2),
            pages_processed=pages_processed if pages_processed > 0 else None,
            grid_systems_detected=grid_systems_detected if grid_systems_detected > 0 else None,
            drawing_types_found=drawing_types_found if drawing_types_found else None,
            author=author,
            trade=trade,
            session_created=session_created
        )
        
        logger.info(f"✅ Upload complete for document {session_id}")
        logger.info(f"   Status: {status_msg}")
        logger.info(f"   Session created: {session_created}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Upload failed: {str(e)}"
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
        
        # Get PDF from main container where original files are stored
        container_name = getattr(settings, 'AZURE_CONTAINER_NAME', 'pdfs')
        pdf_bytes = await storage_service.download_blob_as_bytes(
            container_name=container_name,
            blob_name=f"{clean_document_id}.pdf"
        )
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found for document {clean_document_id}"
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
    
    This endpoint:
    1. Validates the document exists and is processed
    2. Sends the query to the AI service
    3. Returns AI response with optional visual highlights
    4. Suggests creating notes for important findings
    5. Tracks the chat in session history
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
        
        # Check if document exists and is processed
        cache_container = getattr(settings, 'AZURE_CACHE_CONTAINER_NAME', 'cache')
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
                    detail="Document is still being processed. Please try again in a few moments."
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
                        h.get('element_type', 'unknown') 
                        for h in ai_result["visual_highlights"]
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
                "grid_systems_detected": metadata.get('optimization_stats', {}).get('grid_systems_detected', 0) if metadata else None,
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

# Additional routes can be added here following the same pattern...
