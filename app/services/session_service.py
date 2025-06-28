# app/services/session_service.py - FIXED AND COMPLETE SESSION SERVICE

import uuid
import logging
import asyncio
import json
import sys
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta, timezone
from collections import OrderedDict, defaultdict

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# --- Data Models ---

class HighlightSession:
    """Represents a highlight session from a user query with timezone-aware datetimes."""
    
    def __init__(self, query_session_id: str, document_id: str, user: str, query: str):
        self.query_session_id = query_session_id
        self.document_id = document_id
        self.user = user
        self.query = query
        # Use timezone-aware UTC datetime for all timestamping
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.created_at
        self.expires_at: datetime = self.created_at + timedelta(hours=24)
        self.pages_with_highlights: Dict[int, int] = {}
        self.element_types: Set[str] = set()
        self.total_highlights: int = 0
        self.is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to a dictionary, ensuring ISO 8601 format for datetimes."""
        return {
            'query_session_id': self.query_session_id,
            'document_id': self.document_id,
            'user': self.user,
            'query': self.query[:100],  # Truncate long queries
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'pages_with_highlights': self.pages_with_highlights,
            'element_types': list(self.element_types),
            'total_highlights': self.total_highlights,
            'is_active': self.is_active
        }
    
    def update_access(self):
        """Updates the last access time with a timezone-aware datetime."""
        self.last_accessed = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Checks if the session has expired using timezone-aware comparison."""
        return datetime.now(timezone.utc) > self.expires_at

class DocumentSession:
    """Represents a document being viewed, with timezone-aware datetimes."""
    
    def __init__(self, document_id: str, filename: str):
        self.document_id = document_id
        self.filename = filename
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.created_at
        self.annotations: List[Dict] = []
        self.highlight_sessions: Dict[str, HighlightSession] = {}
        self.active_users: Dict[str, str] = {}
        self.chat_count: int = 0
        self.annotation_count: int = 0
        self.metadata: Dict[str, Any] = {}

    def get_memory_usage_estimate(self) -> int:
        """Provides a more accurate memory usage estimate."""
        try:
            size = sys.getsizeof(self)
            size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.__dict__.items())
            # Add size of complex collections
            size += sys.getsizeof(self.annotations) + sum(sys.getsizeof(ann) for ann in self.annotations)
            size += sys.getsizeof(self.highlight_sessions) + sum(sys.getsizeof(hs) for hs in self.highlight_sessions.values())
            return size
        except Exception:
            # Fallback estimation
            return 1024 * len(str(self.__dict__))

# --- Main Service Class ---

class SessionService:
    """
    Manages complex, multi-user document and highlight sessions.
    This version is thread-safe, stable, and includes proper error handling.
    """
    
    def __init__(self, settings: AppSettings, storage_service: StorageService):
        self.settings = settings
        self.storage_service = storage_service
        
        # Use OrderedDict for LRU (Least Recently Used) cache behavior
        self.document_sessions: OrderedDict[str, DocumentSession] = OrderedDict()
        
        # Single lock for thread safety
        self.lock = asyncio.Lock()
        
        # Configuration with safe defaults
        session_config = settings.get_session_settings()
        self.max_sessions = session_config.get('max_sessions', 100)
        self.max_session_memory_mb = session_config.get('max_memory_mb', 512)
        self.session_cleanup_interval_seconds = session_config.get('cleanup_interval', 3600)
        self.session_expiry_hours = session_config.get('expiry_hours', 24)
        
        # Statistics and cleanup
        self.stats = defaultdict(int)
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("✅ SessionService initialized successfully")
        logger.info(f"   📊 Max sessions: {self.max_sessions}")
        logger.info(f"   💾 Max memory: {self.max_session_memory_mb}MB")
        logger.info(f"   🧹 Cleanup interval: {self.session_cleanup_interval_seconds}s")
        logger.info(f"   ⏰ Session expiry: {self.session_expiry_hours}h")

    # --- Lifecycle Management ---

    async def start_background_cleanup(self):
        """Starts the background task for cleaning up old and oversized sessions."""
        if not self.is_running():
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                logger.info(f"🧹 Background session cleanup started (interval: {self.session_cleanup_interval_seconds}s)")
            except Exception as e:
                logger.error(f"Failed to start cleanup task: {e}")

    async def stop_background_cleanup(self):
        """Stops the background cleanup task gracefully."""
        if self.is_running() and self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation
            except Exception as e:
                logger.warning(f"Cleanup task stop error: {e}")
            finally:
                self._cleanup_task = None
                logger.info("🛑 Background session cleanup stopped")

    def is_running(self) -> bool:
        """Checks if the background task is active."""
        return self._cleanup_task is not None and not self._cleanup_task.done()

    async def _cleanup_loop(self):
        """The main loop that periodically triggers session cleanup."""
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval_seconds)
                logger.info("🧹 Running periodic session cleanup...")
                
                cleanup_results = await self._perform_cleanup_and_eviction()
                
                logger.info("🧹 Cleanup complete:")
                logger.info(f"   📊 Sessions checked: {len(self.document_sessions)}")
                logger.info(f"   ❌ Highlights expired: {cleanup_results.get('highlights_expired', 0)}")
                logger.info(f"   🗑️ Sessions evicted: {cleanup_results.get('sessions_evicted', 0)}")
                
            except asyncio.CancelledError:
                logger.info("🧹 Cleanup loop is shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error in session cleanup loop: {e}", exc_info=True)

    # --- Core Thread-Safe Methods ---

    async def get_or_create_session(self, document_id: str, filename: str) -> DocumentSession:
        """
        Gets an existing session or creates a new one. This is the primary
        entry point for interacting with sessions. Fully thread-safe.
        """
        async with self.lock:
            # If session exists, mark as recently used and return
            if document_id in self.document_sessions:
                session = self.document_sessions[document_id]
                session.last_accessed = datetime.now(timezone.utc)
                self.document_sessions.move_to_end(document_id)
                logger.debug(f"📄 Accessed existing session for document: {document_id}")
                return session
            
            # Create new session
            logger.info(f"📄 Creating new session for document '{filename}' (ID: {document_id})")
            session = DocumentSession(document_id, filename)
            self.document_sessions[document_id] = session
            self.stats['sessions_created'] += 1
            
            # Check if eviction is needed
            if len(self.document_sessions) > self.max_sessions:
                await self._evict_oldest_session("max session count exceeded")

            return session

    async def get_session(self, document_id: str) -> Optional[DocumentSession]:
        """Safely retrieves a session by its ID, if it exists."""
        async with self.lock:
            session = self.document_sessions.get(document_id)
            if session:
                session.last_accessed = datetime.now(timezone.utc)
                self.document_sessions.move_to_end(document_id)
            return session

    async def record_chat_activity(self, document_id: str, user: str):
        """Record chat activity for a document session."""
        async with self.lock:
            session = self.document_sessions.get(document_id)
            if session:
                session.chat_count += 1
                session.active_users[user] = datetime.now(timezone.utc).isoformat()
                session.last_accessed = datetime.now(timezone.utc)
                self.document_sessions.move_to_end(document_id)

    async def update_highlight_session(self, document_id: str, query_session_id: str, 
                                     pages_with_highlights: Dict[int, int],
                                     element_types: List[str], total_highlights: int):
        """Update or create a highlight session."""
        async with self.lock:
            session = self.document_sessions.get(document_id)
            if session:
                if query_session_id not in session.highlight_sessions:
                    # Create new highlight session
                    highlight_session = HighlightSession(
                        query_session_id=query_session_id,
                        document_id=document_id,
                        user="system",  # Could be passed as parameter
                        query=""  # Could be passed as parameter
                    )
                    session.highlight_sessions[query_session_id] = highlight_session
                else:
                    highlight_session = session.highlight_sessions[query_session_id]
                
                # Update highlight session data
                highlight_session.pages_with_highlights = pages_with_highlights
                highlight_session.element_types = set(element_types)
                highlight_session.total_highlights = total_highlights
                highlight_session.update_access()

    async def add_annotation(self, document_id: str, annotation_data: Dict[str, Any]) -> bool:
        """Adds an annotation to a document session in a thread-safe manner."""
        session = await self.get_session(document_id)
        if not session:
            logger.warning(f"Attempted to add annotation to non-existent session: {document_id}")
            return False
            
        async with self.lock:
            session.annotations.append(annotation_data)
            session.annotation_count += 1
            self.stats['total_annotations'] += 1
        return True

    # --- Eviction and Cleanup ---

    async def _perform_cleanup_and_eviction(self) -> Dict[str, int]:
        """
        Centralized logic to enforce all cleanup rules inside a lock.
        This prevents race conditions between different cleanup types.
        """
        results = {'highlights_expired': 0, 'sessions_evicted': 0}
        
        async with self.lock:
            # 1. Clean up expired highlight sessions within each document session
            total_expired_highlights = 0
            for doc_session in self.document_sessions.values():
                expired_count = self._cleanup_expired_highlight_sessions(doc_session)
                total_expired_highlights += expired_count
            
            results['highlights_expired'] = total_expired_highlights
            if total_expired_highlights > 0:
                logger.info(f"🧹 Cleaned up {total_expired_highlights} expired highlight sessions")
                self.stats['highlights_expired'] += total_expired_highlights

            # 2. Enforce total memory limit by evicting least recently used sessions
            max_memory_bytes = self.max_session_memory_mb * 1024 * 1024
            evicted_count = 0
            
            while self._get_total_memory_usage() > max_memory_bytes and self.document_sessions:
                await self._evict_oldest_session("memory pressure")
                evicted_count += 1
            
            results['sessions_evicted'] = evicted_count
        
        return results

    def _cleanup_expired_highlight_sessions(self, doc_session: DocumentSession) -> int:
        """Internal method to clean highlight sessions for a given document session."""
        expired_ids = [
            sid for sid, hs in doc_session.highlight_sessions.items() 
            if hs.is_expired()
        ]
        
        if not expired_ids:
            return 0
            
        for session_id in expired_ids:
            del doc_session.highlight_sessions[session_id]
            # Clean up any active user references to this expired session
            users_to_clean = [
                user for user, sid in doc_session.active_users.items() 
                if sid == session_id
            ]
            for user in users_to_clean:
                del doc_session.active_users[user]
        
        return len(expired_ids)

    async def _evict_oldest_session(self, reason: str):
        """
        Safely evicts the single oldest session.
        This function MUST be called from within a locked context.
        """
        if not self.document_sessions:
            return
            
        # Get the ID of the oldest session (first item in OrderedDict)
        oldest_id, session_to_evict = self.document_sessions.popitem(last=False)
        self.stats['sessions_evicted'] += 1
        
        logger.warning(f"🗑️ Evicting session {oldest_id} for '{session_to_evict.filename}' due to: {reason}")
        
        # Persist important data if storage service available
        if self.storage_service:
            try:
                if session_to_evict.annotations:
                    logger.info(f"💾 Persisting {len(session_to_evict.annotations)} annotations for evicted session {oldest_id}")
                    await self._persist_session_data(oldest_id, session_to_evict)
                    logger.info(f"✅ Successfully persisted data for {oldest_id}")
            except Exception as e:
                logger.error(f"❌ Failed to persist data for evicted session {oldest_id}: {e}")

    async def _persist_session_data(self, session_id: str, session: DocumentSession):
        """Persist session data to storage before eviction."""
        try:
            if session.annotations:
                annotations_data = json.dumps(session.annotations, indent=2)
                await self.storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_session_backup.json",
                    data=annotations_data.encode('utf-8')
                )
        except Exception as e:
            logger.error(f"Failed to persist session data: {e}")
            raise

    def _get_total_memory_usage(self) -> int:
        """Calculates the total estimated memory usage of all sessions."""
        if not self.document_sessions:
            return 0
        try:
            return sum(s.get_memory_usage_estimate() for s in self.document_sessions.values())
        except Exception as e:
            logger.warning(f"Memory calculation error: {e}")
            return len(self.document_sessions) * 1024 * 1024  # Rough estimate

    # --- Information and Statistics ---

    def get_session_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        session = self.document_sessions.get(document_id)
        if not session:
            return None
        
        return {
            "document_id": session.document_id,
            "filename": session.filename,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "chat_count": session.chat_count,
            "annotation_count": session.annotation_count,
            "active_highlight_sessions": len(session.highlight_sessions),
            "active_users": list(session.active_users.keys()),
            "memory_usage_estimate": session.get_memory_usage_estimate()
        }

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        total_memory = self._get_total_memory_usage()
        
        return {
            "service_name": "SessionService",
            "status": "running" if self.is_running() else "stopped",
            "configuration": {
                "max_sessions": self.max_sessions,
                "max_memory_mb": self.max_session_memory_mb,
                "cleanup_interval_seconds": self.session_cleanup_interval_seconds,
                "session_expiry_hours": self.session_expiry_hours
            },
            "current_state": {
                "active_sessions": len(self.document_sessions),
                "total_memory_usage_mb": round(total_memory / (1024 * 1024), 2),
                "memory_utilization_percent": round((total_memory / (self.max_session_memory_mb * 1024 * 1024)) * 100, 1) if self.max_session_memory_mb > 0 else 0
            },
            "lifetime_statistics": dict(self.stats),
            "cleanup_task_running": self.is_running()
        }

    # --- Service Management ---

    async def create_session(self, document_id: str, original_filename: str):
        """Create a new session (alternative method name for compatibility)."""
        return await self.get_or_create_session(document_id, original_filename)

    async def update_session_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update session metadata."""
        async with self.lock:
            session = self.document_sessions.get(document_id)
            if session:
                session.metadata.update(metadata)
                session.last_accessed = datetime.now(timezone.utc)

    async def shutdown(self):
        """Graceful shutdown of the session service."""
        logger.info("🛑 SessionService shutting down...")
        await self.stop_background_cleanup()
        
        # Persist any important data
        if self.storage_service:
            try:
                for session_id, session in self.document_sessions.items():
                    if session.annotations:
                        await self._persist_session_data(session_id, session)
            except Exception as e:
                logger.error(f"Error during shutdown persistence: {e}")
        
        logger.info("✅ SessionService shutdown complete")

    # --- Context Manager Support ---

    async def __aenter__(self):
        await self.start_background_cleanup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
