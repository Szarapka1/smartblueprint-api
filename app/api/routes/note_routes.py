# app/api/routes/note_routes.py - NOTE MANAGEMENT WITH AI INTEGRATION

from fastapi import APIRouter, Request, HTTPException, Query
from typing import Optional, List, Dict
from datetime import datetime
import json
import uuid
import logging

from app.core.config import get_settings
from app.models.schemas import (
    Note, NoteCreate, NoteUpdate, NoteList, NoteBatch,
    SuccessResponse, ErrorResponse
)

note_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def validate_document_id(document_id: str) -> str:
    """Validate document ID"""
    import re
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document_id.strip()).strip('_')
    
    if not clean_id or len(clean_id) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Document ID must be at least 3 characters"
        )
    
    return clean_id

async def load_notes(document_id: str, storage_service) -> List[dict]:
    """Load all notes for a document"""
    notes_blob = f"{document_id}_notes.json"
    
    try:
        notes_text = await storage_service.download_blob_as_text(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=notes_blob
        )
        return json.loads(notes_text)
    except:
        return []

async def save_notes(document_id: str, notes: List[dict], storage_service):
    """Save notes to storage"""
    notes_blob = f"{document_id}_notes.json"
    
    await storage_service.upload_file(
        container_name=settings.AZURE_CACHE_CONTAINER_NAME,
        blob_name=notes_blob,
        data=json.dumps(notes, indent=2).encode('utf-8')
    )

def filter_notes_for_user(notes: List[dict], author: str) -> List[dict]:
    """Filter notes based on privacy rules"""
    visible_notes = []
    
    for note in notes:
        # User sees their own notes (private or public)
        if note.get('author') == author:
            visible_notes.append(note)
        # User sees published notes from others
        elif not note.get('is_private', True):
            visible_notes.append(note)
    
    return visible_notes

# --- Note Routes ---

@note_router.post("/documents/{document_id}/notes", response_model=Note)
async def create_note(
    request: Request,
    document_id: str,
    note_create: NoteCreate,
    author: str = Query(..., description="Author creating the note")
):
    """
    Create a new note for the document.
    Notes are private by default unless explicitly set to public.
    """
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Verify document exists
        context_blob = f"{clean_document_id}_context.txt"
        if not await storage_service.blob_exists(
            container_name=settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=context_blob
        ):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Load existing notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Check limits
        user_notes = [n for n in all_notes if n.get('author') == author]
        if len(user_notes) >= settings.MAX_NOTES_PER_DOCUMENT:
            raise HTTPException(
                status_code=400,
                detail=f"Note limit reached ({settings.MAX_NOTES_PER_DOCUMENT} notes per document)"
            )
        
        # Check total character limit
        total_chars = sum(n.get('char_count', 0) for n in user_notes)
        if total_chars + len(note_create.text) > settings.MAX_TOTAL_NOTE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Character limit would be exceeded (max {settings.MAX_TOTAL_NOTE_CHARS} total)"
            )
        
        # Validate note type
        if note_create.note_type not in settings.NOTE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid note type. Must be one of: {settings.NOTE_TYPES}"
            )
        
        # Create note
        new_note = Note(
            note_id=str(uuid.uuid4())[:8],
            document_id=clean_document_id,
            text=note_create.text,
            note_type=note_create.note_type,
            author=author,
            impacts_trades=note_create.impacts_trades,
            priority=note_create.priority,
            is_private=note_create.is_private,
            timestamp=datetime.utcnow().isoformat() + "Z",
            char_count=len(note_create.text),
            status="open",
            related_element_ids=note_create.related_element_ids or [],
            related_query_sessions=note_create.related_query_sessions or [],
            ai_suggested=note_create.ai_suggested,
            suggestion_confidence=note_create.suggestion_confidence
        )
        
        # Add to notes
        all_notes.append(new_note.dict())
        
        # Save
        await save_notes(clean_document_id, all_notes, storage_service)
        
        logger.info(f"✅ Created note {new_note.note_id} for document {clean_document_id}")
        if note_create.ai_suggested:
            logger.info(f"   Note was AI suggested (confidence: {note_create.suggestion_confidence})")
        
        return new_note
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.get("/documents/{document_id}/notes", response_model=NoteList)
async def get_notes(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User requesting notes"),
    note_type: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    include_ai_suggested: bool = True
):
    """
    Get notes visible to the user based on privacy rules.
    Users see:
    - All their own notes (private and public)
    - Published notes from other users
    """
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Filter based on privacy
        visible_notes = filter_notes_for_user(all_notes, author)
        
        # Apply filters
        filtered_notes = visible_notes
        
        if note_type:
            filtered_notes = [n for n in filtered_notes if n.get('note_type') == note_type]
        
        if priority:
            filtered_notes = [n for n in filtered_notes if n.get('priority') == priority]
        
        if status:
            filtered_notes = [n for n in filtered_notes if n.get('status') == status]
        
        if not include_ai_suggested:
            filtered_notes = [n for n in filtered_notes if not n.get('ai_suggested', False)]
        
        # Convert to Note models
        note_models = [Note(**note) for note in filtered_notes]
        
        # Sort by timestamp (newest first)
        note_models.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Calculate statistics
        private_count = sum(1 for n in visible_notes if n.get('is_private', True) and n.get('author') == author)
        published_count = sum(1 for n in visible_notes if not n.get('is_private', True))
        ai_suggested_count = sum(1 for n in visible_notes if n.get('ai_suggested', False))
        
        status_breakdown = {}
        for note in visible_notes:
            s = note.get('status', 'open')
            status_breakdown[s] = status_breakdown.get(s, 0) + 1
        
        return NoteList(
            notes=note_models,
            total_count=len(note_models),
            filter_applied={
                "note_type": note_type,
                "priority": priority,
                "status": status
            } if any([note_type, priority, status]) else None,
            private_notes_count=private_count,
            published_notes_count=published_count,
            notes_by_status=status_breakdown,
            ai_suggested_count=ai_suggested_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.get("/documents/{document_id}/notes/{note_id}", response_model=Note)
async def get_note(
    request: Request,
    document_id: str,
    note_id: str,
    author: str = Query(..., description="User requesting the note")
):
    """Get a specific note if visible to the user"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find the note
        note = None
        for n in all_notes:
            if n.get('note_id') == note_id:
                note = n
                break
        
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Check visibility
        if note.get('is_private', True) and note.get('author') != author:
            raise HTTPException(status_code=403, detail="Note is private")
        
        return Note(**note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.put("/documents/{document_id}/notes/{note_id}")
async def update_note(
    request: Request,
    document_id: str,
    note_id: str,
    note_update: NoteUpdate,
    author: str = Query(..., description="User updating the note")
):
    """Update a note (only by the author)"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find and update the note
        updated = False
        for i, note in enumerate(all_notes):
            if note.get('note_id') == note_id:
                # Check ownership
                if note.get('author') != author:
                    raise HTTPException(status_code=403, detail="Only the author can update this note")
                
                # Update fields
                if note_update.text is not None:
                    note['text'] = note_update.text
                    note['char_count'] = len(note_update.text)
                
                if note_update.note_type is not None:
                    if note_update.note_type not in settings.NOTE_TYPES:
                        raise HTTPException(status_code=400, detail="Invalid note type")
                    note['note_type'] = note_update.note_type
                
                if note_update.impacts_trades is not None:
                    note['impacts_trades'] = note_update.impacts_trades
                
                if note_update.priority is not None:
                    note['priority'] = note_update.priority
                
                if note_update.status is not None:
                    note['status'] = note_update.status
                    if note_update.status == 'resolved':
                        note['resolved_at'] = datetime.utcnow().isoformat() + "Z"
                        note['resolved_by'] = author
                
                if note_update.resolution_notes is not None:
                    note['resolution_notes'] = note_update.resolution_notes
                
                if note_update.related_element_ids is not None:
                    note['related_element_ids'] = note_update.related_element_ids
                
                note['edited_at'] = datetime.utcnow().isoformat() + "Z"
                
                all_notes[i] = note
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Save
        await save_notes(clean_document_id, all_notes, storage_service)
        
        logger.info(f"✅ Updated note {note_id}")
        
        return {"status": "success", "message": "Note updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.put("/documents/{document_id}/notes/{note_id}/publish")
async def publish_note(
    request: Request,
    document_id: str,
    note_id: str,
    author: str = Query(..., description="User publishing the note")
):
    """Publish a private note to make it visible to all users"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find and publish the note
        published = False
        for i, note in enumerate(all_notes):
            if note.get('note_id') == note_id:
                # Check ownership
                if note.get('author') != author:
                    raise HTTPException(status_code=403, detail="Only the author can publish this note")
                
                # Check if already published
                if not note.get('is_private', True):
                    return {"status": "success", "message": "Note is already published"}
                
                # Publish
                note['is_private'] = False
                note['published_at'] = datetime.utcnow().isoformat() + "Z"
                
                all_notes[i] = note
                published = True
                break
        
        if not published:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Save
        await save_notes(clean_document_id, all_notes, storage_service)
        
        logger.info(f"✅ Published note {note_id}")
        
        return {"status": "success", "message": "Note published successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.delete("/documents/{document_id}/notes/{note_id}")
async def delete_note(
    request: Request,
    document_id: str,
    note_id: str,
    author: str = Query(..., description="User deleting the note")
):
    """Delete a note (only by the author)"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Find and delete the note
        deleted = False
        filtered_notes = []
        
        for note in all_notes:
            if note.get('note_id') == note_id:
                # Check ownership
                if note.get('author') != author:
                    raise HTTPException(status_code=403, detail="Only the author can delete this note")
                deleted = True
                logger.info(f"🗑️ Deleting note {note_id}")
            else:
                filtered_notes.append(note)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Note not found")
        
        # Save
        await save_notes(clean_document_id, filtered_notes, storage_service)
        
        return {"status": "success", "message": "Note deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.post("/documents/{document_id}/notes/batch", response_model=SuccessResponse)
async def batch_update_notes(
    request: Request,
    document_id: str,
    batch: NoteBatch,
    author: str = Query(..., description="User performing batch operation")
):
    """Perform batch operations on multiple notes"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        updated_count = 0
        
        for i, note in enumerate(all_notes):
            if note.get('note_id') in batch.note_ids:
                # Check ownership
                if note.get('author') != author:
                    continue
                
                if batch.operation == "update" and batch.update_data:
                    # Apply updates
                    if batch.update_data.status:
                        note['status'] = batch.update_data.status
                        if batch.update_data.status == 'resolved':
                            note['resolved_at'] = datetime.utcnow().isoformat() + "Z"
                            note['resolved_by'] = author
                    
                    if batch.update_data.priority:
                        note['priority'] = batch.update_data.priority
                    
                    note['edited_at'] = datetime.utcnow().isoformat() + "Z"
                    all_notes[i] = note
                    updated_count += 1
                
                elif batch.operation == "resolve":
                    note['status'] = 'resolved'
                    note['resolved_at'] = datetime.utcnow().isoformat() + "Z"
                    note['resolved_by'] = author
                    all_notes[i] = note
                    updated_count += 1
                
                elif batch.operation == "delete":
                    # Mark for deletion (we'll filter later)
                    note['_delete'] = True
                    updated_count += 1
        
        # Filter out deleted notes
        if batch.operation == "delete":
            all_notes = [n for n in all_notes if not n.get('_delete')]
        
        # Save
        await save_notes(clean_document_id, all_notes, storage_service)
        
        logger.info(f"✅ Batch {batch.operation} completed: {updated_count} notes affected")
        
        return SuccessResponse(
            status="success",
            message=f"Batch operation completed. {updated_count} notes {batch.operation}d."
        )
        
    except Exception as e:
        logger.error(f"Failed to perform batch operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.get("/documents/{document_id}/notes/stats")
async def get_note_statistics(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User requesting statistics")
):
    """Get statistics about notes for the document"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Calculate statistics
        user_notes = [n for n in all_notes if n.get('author') == author]
        published_notes = [n for n in all_notes if not n.get('is_private', True)]
        ai_suggested_notes = [n for n in all_notes if n.get('ai_suggested', False)]
        
        # Type breakdown
        type_breakdown = {}
        for note in all_notes:
            if not note.get('is_private', True) or note.get('author') == author:
                ntype = note.get('note_type', 'general')
                type_breakdown[ntype] = type_breakdown.get(ntype, 0) + 1
        
        # Priority breakdown
        priority_breakdown = {}
        for note in all_notes:
            if not note.get('is_private', True) or note.get('author') == author:
                priority = note.get('priority', 'normal')
                priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
        
        # Trade impact analysis
        trade_impacts = {}
        for note in all_notes:
            if not note.get('is_private', True) or note.get('author') == author:
                for trade in note.get('impacts_trades', []):
                    trade_impacts[trade] = trade_impacts.get(trade, 0) + 1
        
        # AI suggestion statistics
        ai_stats = {
            'total_suggested': len(ai_suggested_notes),
            'user_created_from_ai': len([n for n in user_notes if n.get('ai_suggested', False)]),
            'avg_confidence': sum(n.get('suggestion_confidence', 0) for n in ai_suggested_notes) / len(ai_suggested_notes) if ai_suggested_notes else 0
        }
        
        return {
            "document_id": clean_document_id,
            "user_stats": {
                "private_notes": len([n for n in user_notes if n.get('is_private', True)]),
                "published_notes": len([n for n in user_notes if not n.get('is_private', True)]),
                "total_notes": len(user_notes),
                "total_characters": sum(n.get('char_count', 0) for n in user_notes)
            },
            "global_stats": {
                "total_published_notes": len(published_notes),
                "total_contributors": len(set(n.get('author') for n in published_notes))
            },
            "breakdown": {
                "by_type": type_breakdown,
                "by_priority": priority_breakdown,
                "by_trade_impact": trade_impacts
            },
            "ai_suggestion_stats": ai_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get note statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.get("/documents/{document_id}/notes/search")
async def search_notes(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User performing search"),
    q: str = Query(..., description="Search query"),
    search_in: str = Query("all", description="Search in: all, text, trades")
):
    """Search notes visible to the user"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Filter based on privacy
        visible_notes = filter_notes_for_user(all_notes, author)
        
        # Search
        search_results = []
        query_lower = q.lower()
        
        for note in visible_notes:
            match = False
            
            if search_in in ["all", "text"]:
                if query_lower in note.get('text', '').lower():
                    match = True
                elif query_lower in note.get('resolution_notes', '').lower():
                    match = True
            
            if search_in in ["all", "trades"]:
                for trade in note.get('impacts_trades', []):
                    if query_lower in trade.lower():
                        match = True
                        break
            
            if match:
                search_results.append(note)
        
        # Convert to Note models
        note_models = [Note(**note) for note in search_results]
        
        # Sort by relevance (simple: most recent first)
        note_models.sort(key=lambda x: x.timestamp, reverse=True)
        
        return {
            "query": q,
            "search_in": search_in,
            "results_count": len(note_models),
            "notes": note_models
        }
        
    except Exception as e:
        logger.error(f"Failed to search notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@note_router.get("/documents/{document_id}/notes/export")
async def export_notes(
    request: Request,
    document_id: str,
    author: str = Query(..., description="User exporting notes"),
    format: str = Query("json", description="Export format: json, csv")
):
    """Export notes visible to the user"""
    clean_document_id = validate_document_id(document_id)
    storage_service = request.app.state.storage_service
    
    if not storage_service:
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    
    try:
        # Load all notes
        all_notes = await load_notes(clean_document_id, storage_service)
        
        # Filter based on privacy
        visible_notes = filter_notes_for_user(all_notes, author)
        
        if format == "json":
            return {
                "document_id": clean_document_id,
                "export_date": datetime.utcnow().isoformat() + "Z",
                "exported_by": author,
                "notes_count": len(visible_notes),
                "notes": visible_notes
            }
        
        elif format == "csv":
            # Simple CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'note_id', 'type', 'priority', 'status', 'text', 
                'author', 'created', 'is_private', 'ai_suggested'
            ])
            
            writer.writeheader()
            for note in visible_notes:
                writer.writerow({
                    'note_id': note.get('note_id'),
                    'type': note.get('note_type'),
                    'priority': note.get('priority'),
                    'status': note.get('status'),
                    'text': note.get('text', '').replace('\n', ' '),
                    'author': note.get('author'),
                    'created': note.get('timestamp'),
                    'is_private': note.get('is_private', True),
                    'ai_suggested': note.get('ai_suggested', False)
                })
            
            from fastapi.responses import Response
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=notes_{clean_document_id}.csv"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use json or csv")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
