# app/services/session_service.py - COMPLETE FIXED VERSION

"""
Session Service for managing document sessions, collaboration state, and real-time highlights.
Provides persistent session management with Azure Blob Storage backend.
FIXED: Memory leak prevention, better error recovery, and improved cleanup
"""

import logging
import json
import asyncio
from typing import Dict, Optional, List, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import uuid
import weakref
import gc

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass
class HighlightSession:
    """Represents a temporary highlight session from AI analysis."""
    query_session_id: str
    user: str
    query: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True
    total_highlights: int = 0
    pages_with_highlights: Dict[int, int] = field(default_factory=dict)
    element_types: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if this highlight session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'query_session_id': self.query_session_id,
            'user': self.user,
            'query': self.query,
            'created_at': self.created_at.isoformat() + 'Z',
            'expires_at': self.expires_at.isoformat() + 'Z',
            'is_active': self.is_active,
            'total_highlights': self.total_highlights,
            'pages_with_highlights': dict(self.pages_with_highlights),
            'element_types': list(self.element_types)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HighlightSession':
        """Create from dictionary."""
        return cls(
            query_session_id=data['query_session_id'],
            user=data['user'],
            query=data['query'],
            created_at=datetime.fromisoformat(data['created_at'].rstrip('Z')),
            expires_at=datetime.fromisoformat(data['expires_at'].rstrip('Z')),
            is_active=data.get('is_active', True),
            total_highlights=data.get('total_highlights', 0),
            pages_with_highlights=data.get('pages_with_highlights', {}),
            element_types=set(data.get('element_types', []))
        )


@dataclass
class DocumentSession:
    """Represents an active document session with collaboration features."""
    document_id: str
    created_at: datetime
    last_activity: datetime
    original_filename: str
    active_users: Set[str] = field(default_factory=set)
    annotation_count: int = 0
    chat_count: int = 0
    note_count: int = 0
    highlight_sessions: Dict[str, HighlightSession] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self, user: Optional[str] = None):
        """Update last activity timestamp and optionally add user."""
        self.last_activity = datetime.utcnow()
        if user:
            self.active_users.add(user)
            # Limit active users to prevent memory growth
            if len(self.active_users) > 100:
                # Keep only the 100 most recent users (this is a simple approach)
                self.active_users = set(list(self.active_users)[-100:])
    
    def add_highlight_session(self, session: HighlightSession):
        """Add a new highlight session with size limit."""
        # First cleanup expired sessions
        self.cleanup_expired_highlights()
        
        # Check if we're at the limit
        max_highlight_sessions = 50  # Reasonable limit per document
        if len(self.highlight_sessions) >= max_highlight_sessions:
            # Remove oldest expired session
            oldest_expired = None
            for sid, hl_session in self.highlight_sessions.items():
                if hl_session.is_expired():
                    if oldest_expired is None or hl_session.created_at < self.highlight_sessions[oldest_expired].created_at:
                        oldest_expired = sid
            
            if oldest_expired:
                del self.highlight_sessions[oldest_expired]
            else:
                # If no expired sessions, remove oldest
                oldest = min(self.highlight_sessions.items(), key=lambda x: x[1].created_at)[0]
                del self.highlight_sessions[oldest]
        
        self.highlight_sessions[session.query_session_id] = session
        self.update_activity(session.user)
    
    def cleanup_expired_highlights(self):
        """Remove expired highlight sessions."""
        expired_ids = [
            sid for sid, session in self.highlight_sessions.items()
            if session.is_expired()
        ]
        for sid in expired_ids:
            del self.highlight_sessions[sid]
        
        # Force garbage collection if many were removed
        if len(expired_ids) > 10:
            gc.collect(0)  # Collect youngest generation only
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Limit metadata size to prevent memory issues
        safe_metadata = dict(self.metadata)
        if 'large_data' in safe_metadata:
            safe_metadata['large_data'] = '[truncated]'
        
        return {
            'document_id': self.document_id,
            'created_at': self.created_at.isoformat() + 'Z',
            'last_activity': self.last_activity.isoformat() + 'Z',
            'original_filename': self.original_filename,
            'active_users': list(self.active_users)[:100],  # Limit to 100 users
            'annotation_count': self.annotation_count,
            'chat_count': self.chat_count,
            'note_count': self.note_count,
            'highlight_sessions': {
                sid: session.to_dict() 
                for sid, session in list(self.highlight_sessions.items())[:50]  # Limit to 50
            },
            'metadata': safe_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentSession':
        """Create from dictionary with error recovery."""
        try:
            session = cls(
                document_id=data['document_id'],
                created_at=datetime.fromisoformat(data['created_at'].rstrip('Z')),
                last_activity=datetime.fromisoformat(data['last_activity'].rstrip('Z')),
                original_filename=data['original_filename'],
                active_users=set(data.get('active_users', [])[:100]),  # Limit users
                annotation_count=data.get('annotation_count', 0),
                chat_count=data.get('chat_count', 0),
                note_count=data.get('note_count', 0),
                metadata=data.get('metadata', {})
            )
            
            # Restore highlight sessions with limit
            highlight_data = data.get('highlight_sessions', {})
            for sid, hl_data in list(highlight_data.items())[:50]:  # Limit to 50
                try:
                    session.highlight_sessions[sid] = HighlightSession.from_dict(hl_data)
                except Exception as e:
                    logger.warning(f"Failed to restore highlight session {sid}: {e}")
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to restore session from dict: {e}")
            # Return a minimal session on error
            return cls(
                document_id=data.get('document_id', 'unknown'),
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                original_filename=data.get('original_filename', 'unknown.pdf')
            )


class SessionService:
    """Service for managing document sessions with persistent storage and memory leak prevention."""
    
    def __init__(self, settings: AppSettings):
        """Initialize session service with settings only."""
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        self.storage_service: Optional[StorageService] = None
        
        # Session configuration
        self.session_timeout_hours = settings.SESSION_CLEANUP_HOURS
        self.highlight_session_duration_hours = 2
        self.max_sessions_per_document = 100
        self.cleanup_interval_minutes = 30
        
        # In-memory cache for active sessions with size limit
        self._sessions: Dict[str, DocumentSession] = {}
        self._session_access_times: Dict[str, datetime] = {}  # Track access for LRU
        self._last_cleanup = datetime.utcnow()
        self._lock = asyncio.Lock()
        
        # Use weak references to prevent memory leaks
        self._weak_sessions: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        
        logger.info("✅ Session Service initialized")
        logger.info(f"   Session timeout: {self.session_timeout_hours} hours")
        logger.info(f"   Highlight duration: {self.highlight_session_duration_hours} hours")
        logger.info(f"   Max sessions in memory: {self.settings.MAX_SESSIONS_IN_MEMORY}")
    
    def set_storage_service(self, storage_service: StorageService):
        """Set the storage service for persistence."""
        self.storage_service = storage_service
        logger.info("✅ Storage service connected to session service")
    
    async def start(self):
        """Start the session service and background tasks."""
        if self.storage_service:
            # Load existing sessions from storage
            await self._load_sessions_from_storage()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Start persist task
        self._persist_task = asyncio.create_task(self._periodic_persist())
        
        logger.info("✅ Session service started")
    
    async def stop(self):
        """Stop the session service and save state."""
        # Cancel background tasks
        for task in [self._cleanup_task, self._persist_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save all sessions to storage
        if self.storage_service:
            await self._save_sessions_to_storage()
        
        # Clear memory
        self._sessions.clear()
        self._session_access_times.clear()
        self._weak_sessions.clear()
        
        logger.info("✅ Session service stopped")
    
    # Legacy method names for backward compatibility
    async def start_background_cleanup(self):
        """Legacy method name - redirects to start()"""
        await self.start()
    
    async def shutdown(self):
        """Legacy method name - redirects to stop()"""
        await self.stop()
    
    async def create_session(self, document_id: str, original_filename: str) -> DocumentSession:
        """Create a new document session with memory management."""
        async with self._lock:
            # Check if session already exists
            if document_id in self._sessions:
                logger.info(f"Session already exists for {document_id}")
                self._session_access_times[document_id] = datetime.utcnow()
                return self._sessions[document_id]
            
            # Check memory limit and evict if necessary
            await self._check_memory_limit()
            
            # Create new session
            session = DocumentSession(
                document_id=document_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                original_filename=original_filename
            )
            
            self._sessions[document_id] = session
            self._session_access_times[document_id] = datetime.utcnow()
            self._weak_sessions[document_id] = session
            
            # Persist to storage
            await self._persist_session(session)
            
            logger.info(f"✅ Created session for document {document_id}")
            return session
    
    async def get_session(self, document_id: str) -> Optional[DocumentSession]:
        """Get an existing document session with LRU tracking."""
        async with self._lock:
            # Check in-memory cache
            if document_id in self._sessions:
                session = self._sessions[document_id]
                session.cleanup_expired_highlights()
                self._session_access_times[document_id] = datetime.utcnow()
                return session
            
            # Check weak references
            if document_id in self._weak_sessions:
                session = self._weak_sessions[document_id]
                # Promote back to strong reference
                self._sessions[document_id] = session
                self._session_access_times[document_id] = datetime.utcnow()
                session.cleanup_expired_highlights()
                await self._check_memory_limit()
                return session
            
            # Try to load from storage
            if self.storage_service:
                session = await self._load_session_from_storage(document_id)
                if session:
                    # Check memory limit before adding
                    await self._check_memory_limit()
                    self._sessions[document_id] = session
                    self._session_access_times[document_id] = datetime.utcnow()
                    self._weak_sessions[document_id] = session
                    session.cleanup_expired_highlights()
                    return session
            
            return None
    
    async def update_session_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update session metadata with size limits."""
        async with self._lock:
            session = await self.get_session(document_id)
            if session:
                # Limit metadata size to prevent memory issues
                safe_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        safe_metadata[key] = value
                    elif isinstance(value, (list, dict)):
                        # Limit complex types
                        if len(str(value)) < 1000:
                            safe_metadata[key] = value
                        else:
                            safe_metadata[key] = f"[truncated - {type(value).__name__}]"
                
                session.metadata.update(safe_metadata)
                session.update_activity()
                await self._persist_session(session)
                logger.info(f"✅ Updated metadata for session {document_id}")
    
    async def add_annotation(self, document_id: str, annotation_data: Dict[str, Any]):
        """Add annotation to session tracking."""
        async with self._lock:
            session = await self.get_session(document_id)
            if session:
                session.annotation_count += 1
                session.update_activity(annotation_data.get('author'))
                # Don't persist on every annotation to reduce I/O
                if session.annotation_count % 10 == 0:
                    await self._persist_session(session)
    
    async def record_chat_activity(self, document_id: str, user: str, chat_data: Dict[str, Any]):
        """Record chat activity in session."""
        async with self._lock:
            session = await self.get_session(document_id)
            if session:
                session.chat_count += 1
                session.update_activity(user)
                # Don't persist on every chat to reduce I/O
                if session.chat_count % 5 == 0:
                    await self._persist_session(session)
    
    async def add_highlight_session(
        self, 
        document_id: str, 
        query_session_id: str,
        user: str,
        query: str,
        total_highlights: int,
        pages_with_highlights: Dict[int, int],
        element_types: List[str]
    ):
        """Add a new highlight session from AI analysis."""
        async with self._lock:
            session = await self.get_session(document_id)
            if session:
                highlight_session = HighlightSession(
                    query_session_id=query_session_id,
                    user=user,
                    query=query[:500],  # Limit query length
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=self.highlight_session_duration_hours),
                    total_highlights=total_highlights,
                    pages_with_highlights=pages_with_highlights,
                    element_types=set(element_types[:20])  # Limit element types
                )
                
                session.add_highlight_session(highlight_session)
                await self._persist_session(session)
                
                logger.info(f"✅ Added highlight session {query_session_id} to document {document_id}")
    
    async def get_session_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get session information for a document."""
        session = await self.get_session(document_id)
        if not session:
            return None
        
        # Cleanup expired highlights
        session.cleanup_expired_highlights()
        
        # Calculate session age
        session_age = datetime.utcnow() - session.created_at
        
        return {
            'document_id': document_id,
            'created_at': session.created_at.isoformat() + 'Z',
            'last_activity': session.last_activity.isoformat() + 'Z',
            'session_age_hours': session_age.total_seconds() / 3600,
            'active_users': list(session.active_users)[:50],  # Limit to 50 users
            'active_user_count': len(session.active_users),
            'statistics': {
                'annotations': session.annotation_count,
                'chats': session.chat_count,
                'notes': session.note_count,
                'active_highlights': len(session.highlight_sessions)
            },
            'has_active_highlights': len(session.highlight_sessions) > 0,
            'original_filename': session.original_filename
        }
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions."""
        async with self._lock:
            active_sessions = []
            
            for doc_id, session in list(self._sessions.items()):
                # Check if session is still active
                time_since_activity = datetime.utcnow() - session.last_activity
                if time_since_activity.total_seconds() < self.session_timeout_hours * 3600:
                    try:
                        session_info = await self.get_session_info(doc_id)
                        if session_info:
                            active_sessions.append(session_info)
                    except Exception as e:
                        logger.error(f"Error getting session info for {doc_id}: {e}")
            
            return active_sessions
    
    async def clear_session(self, document_id: str):
        """Clear a document session."""
        async with self._lock:
            # Remove from all caches
            if document_id in self._sessions:
                del self._sessions[document_id]
            
            if document_id in self._session_access_times:
                del self._session_access_times[document_id]
            
            if document_id in self._weak_sessions:
                del self._weak_sessions[document_id]
            
            # Remove from storage
            if self.storage_service:
                try:
                    await self.storage_service.delete_blob(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{document_id}_session.json"
                    )
                except Exception as e:
                    logger.error(f"Failed to delete session from storage: {e}")
            
            logger.info(f"✅ Cleared session for document {document_id}")
    
    # --- Private Methods ---
    
    async def _check_memory_limit(self):
        """Check and enforce memory limits using LRU eviction."""
        max_sessions = self.settings.MAX_SESSIONS_IN_MEMORY
        
        if len(self._sessions) >= max_sessions:
            # Find least recently used session
            if self._session_access_times:
                lru_doc_id = min(self._session_access_times.items(), key=lambda x: x[1])[0]
                
                # Move to weak reference only
                if lru_doc_id in self._sessions:
                    session = self._sessions.pop(lru_doc_id)
                    self._weak_sessions[lru_doc_id] = session
                    del self._session_access_times[lru_doc_id]
                    
                    # Persist before evicting
                    try:
                        await self._persist_session(session)
                    except Exception as e:
                        logger.error(f"Failed to persist LRU session {lru_doc_id}: {e}")
                    
                    logger.info(f"Evicted LRU session {lru_doc_id} from memory")
    
    async def _persist_session(self, session: DocumentSession):
        """Persist session to storage with error handling."""
        if not self.storage_service:
            return
        
        try:
            session_data = json.dumps(session.to_dict(), indent=2)
            
            await self.storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session.document_id}_session.json",
                data=session_data.encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to persist session {session.document_id}: {e}")
    
    async def _load_session_from_storage(self, document_id: str) -> Optional[DocumentSession]:
        """Load a session from storage with error recovery."""
        if not self.storage_service:
            return None
        
        try:
            session_data = await self.storage_service.download_blob_as_json(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_session.json"
            )
            
            if not session_data:
                return None
            
            return DocumentSession.from_dict(session_data)
            
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to load session {document_id}: {e}")
            return None
    
    async def _load_sessions_from_storage(self):
        """Load all sessions from storage on startup with memory limits."""
        if not self.storage_service:
            return
        
        try:
            # List all session blobs
            blobs = await self.storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                suffix="_session.json"
            )
            
            session_count = 0
            max_to_load = min(len(blobs), self.settings.MAX_SESSIONS_IN_MEMORY)
            
            # Sort by modification time if possible to load most recent
            for blob_name in blobs[:max_to_load]:
                if blob_name.endswith('_session.json'):
                    doc_id = blob_name.replace('_session.json', '')
                    session = await self._load_session_from_storage(doc_id)
                    if session:
                        # Only load recent sessions
                        time_since_activity = datetime.utcnow() - session.last_activity
                        if time_since_activity.total_seconds() < self.session_timeout_hours * 3600:
                            self._sessions[doc_id] = session
                            self._session_access_times[doc_id] = session.last_activity
                            self._weak_sessions[doc_id] = session
                            session_count += 1
                            
                            # Cleanup expired highlights on load
                            session.cleanup_expired_highlights()
            
            logger.info(f"✅ Loaded {session_count} active sessions from storage")
            
        except Exception as e:
            logger.error(f"Failed to load sessions from storage: {e}")
    
    async def _save_sessions_to_storage(self):
        """Save all sessions to storage."""
        if not self.storage_service:
            return
        
        saved_count = 0
        failed_count = 0
        
        # Save all sessions (both strong and weak references)
        all_sessions = {}
        all_sessions.update(self._sessions)
        
        # Add weak references that aren't in strong references
        for doc_id, session in self._weak_sessions.items():
            if doc_id not in all_sessions:
                all_sessions[doc_id] = session
        
        for session in all_sessions.values():
            try:
                await self._persist_session(session)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save session {session.document_id}: {e}")
                failed_count += 1
        
        logger.info(f"✅ Saved {saved_count} sessions to storage ({failed_count} failed)")
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions and highlights."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                
                async with self._lock:
                    # Find expired sessions
                    expired_sessions = []
                    current_time = datetime.utcnow()
                    
                    for doc_id, session in list(self._sessions.items()):
                        # Cleanup expired highlights
                        session.cleanup_expired_highlights()
                        
                        # Check if session is expired
                        time_since_activity = current_time - session.last_activity
                        if time_since_activity.total_seconds() > self.session_timeout_hours * 3600:
                            expired_sessions.append(doc_id)
                    
                    # Remove expired sessions
                    for doc_id in expired_sessions:
                        await self.clear_session(doc_id)
                    
                    if expired_sessions:
                        logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                    
                    # Force garbage collection after cleanup
                    if len(expired_sessions) > 5:
                        gc.collect()
                    
                    self._last_cleanup = current_time
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _periodic_persist(self):
        """Periodically persist active sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                async with self._lock:
                    persist_count = 0
                    for session in self._sessions.values():
                        try:
                            await self._persist_session(session)
                            persist_count += 1
                        except Exception as e:
                            logger.error(f"Failed to persist session {session.document_id}: {e}")
                    
                    if persist_count > 0:
                        logger.info(f"💾 Persisted {persist_count} active sessions")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic persist: {e}")
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        total_highlights = sum(
            len(session.highlight_sessions) 
            for session in self._sessions.values()
        )
        
        total_users = len(set(
            user 
            for session in self._sessions.values() 
            for user in session.active_users
        ))
        
        return {
            "total_sessions": len(self._sessions),
            "weak_sessions": len(self._weak_sessions),
            "total_active_highlights": total_highlights,
            "total_active_users": total_users,
            "last_cleanup": self._last_cleanup.isoformat() + 'Z',
            "session_timeout_hours": self.session_timeout_hours,
            "highlight_duration_hours": self.highlight_session_duration_hours,
            "memory_limit": self.settings.MAX_SESSIONS_IN_MEMORY,
            "cleanup_interval_minutes": self.cleanup_interval_minutes
        }
    
    def is_running(self) -> bool:
        """Check if the service is running."""
        return (
            self._cleanup_task is not None and not self._cleanup_task.done() and
            self._persist_task is not None and not self._persist_task.done()
        )