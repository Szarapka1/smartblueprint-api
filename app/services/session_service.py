# app/services/session_service.py - OPTIMIZED SESSION SERVICE WITH BOUNDED MEMORY

import uuid
import logging
import asyncio
import json
import sys
import gzip
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta, timezone
from collections import OrderedDict, defaultdict, deque

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
   """Represents a document being viewed with bounded memory usage."""
   
   def __init__(self, document_id: str, filename: str, max_items_in_memory: int = 100):
       self.document_id = document_id
       self.filename = filename
       self.created_at: datetime = datetime.now(timezone.utc)
       self.last_accessed: datetime = self.created_at
       
       # Use deque for automatic size limits (FIFO)
       self.max_items_in_memory = max_items_in_memory
       self.recent_annotations = deque(maxlen=max_items_in_memory)
       self.recent_chats = deque(maxlen=max_items_in_memory)
       
       # Keep full index in memory but not full data
       self.annotation_ids: Set[str] = set()
       self.chat_timestamps: List[str] = []
       
       # Highlight sessions stay in memory (they auto-expire)
       self.highlight_sessions: Dict[str, HighlightSession] = {}
       
       # Active users and metadata
       self.active_users: Dict[str, str] = {}
       self.chat_count: int = 0
       self.annotation_count: int = 0
       self.metadata: Dict[str, Any] = {}
       
       # Track what's been persisted
       self.last_persisted_at: datetime = datetime.now(timezone.utc)
       self.needs_persistence: bool = False

   def add_annotation(self, annotation: Dict[str, Any]):
       """Add annotation with automatic persistence tracking."""
       self.recent_annotations.append(annotation)
       self.annotation_ids.add(annotation.get('annotation_id', ''))
       self.annotation_count += 1
       self.needs_persistence = True
       self.last_accessed = datetime.now(timezone.utc)
   
   def add_chat(self, chat_data: Dict[str, Any]):
       """Add chat with automatic persistence tracking."""
       self.recent_chats.append(chat_data)
       self.chat_timestamps.append(chat_data.get('timestamp', ''))
       self.chat_count += 1
       self.needs_persistence = True
       self.last_accessed = datetime.now(timezone.utc)

   def get_memory_usage_estimate(self) -> int:
       """Get accurate memory usage estimate."""
       try:
           # Base object size
           size = sys.getsizeof(self)
           
           # Recent items (bounded by deque)
           size += sys.getsizeof(self.recent_annotations)
           size += sum(sys.getsizeof(ann) for ann in self.recent_annotations)
           size += sys.getsizeof(self.recent_chats)
           size += sum(sys.getsizeof(chat) for chat in self.recent_chats)
           
           # Index data (unbounded but small)
           size += sys.getsizeof(self.annotation_ids)
           size += sys.getsizeof(self.chat_timestamps)
           
           # Highlight sessions
           size += sys.getsizeof(self.highlight_sessions)
           for session in self.highlight_sessions.values():
               size += sys.getsizeof(session.to_dict())
           
           # Metadata
           size += sys.getsizeof(self.metadata)
           size += sys.getsizeof(self.active_users)
           
           return size
       except Exception:
           # Conservative estimate: 10KB per recent item + 5KB base
           return 5000 + (len(self.recent_annotations) + len(self.recent_chats)) * 10000

   def to_summary_dict(self) -> Dict[str, Any]:
       """Get summary for persistence without full data."""
       return {
           "document_id": self.document_id,
           "filename": self.filename,
           "created_at": self.created_at.isoformat(),
           "last_accessed": self.last_accessed.isoformat(),
           "chat_count": self.chat_count,
           "annotation_count": self.annotation_count,
           "active_users": list(self.active_users.keys()),
           "highlight_sessions_count": len(self.highlight_sessions),
           "annotation_ids_count": len(self.annotation_ids),
           "metadata": self.metadata
       }

# --- Main Service Class ---

class SessionService:
   """
   Manages document sessions with bounded memory and automatic persistence.
   Optimized for multi-user collaboration with data preservation.
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
       
       # Memory management settings
       self.max_items_per_session = 100  # Recent items kept in memory
       self.persistence_interval_seconds = 300  # Persist every 5 minutes
       self.compress_persistence = True  # Use gzip compression
       
       # Statistics and cleanup
       self.stats = defaultdict(int)
       self._cleanup_task: Optional[asyncio.Task] = None
       self._persistence_task: Optional[asyncio.Task] = None
       
       logger.info("✅ SessionService initialized with bounded memory")
       logger.info(f"   📊 Max sessions: {self.max_sessions}")
       logger.info(f"   💾 Max memory: {self.max_session_memory_mb}MB")
       logger.info(f"   📝 Max items per session in memory: {self.max_items_per_session}")
       logger.info(f"   💿 Persistence interval: {self.persistence_interval_seconds}s")
       logger.info(f"   🗜️ Compression: {'enabled' if self.compress_persistence else 'disabled'}")

   # --- Lifecycle Management ---

   async def start_background_cleanup(self):
       """Starts background tasks for cleanup and persistence."""
       if not self.is_running():
           try:
               self._cleanup_task = asyncio.create_task(self._cleanup_loop())
               self._persistence_task = asyncio.create_task(self._persistence_loop())
               logger.info("🧹 Background tasks started (cleanup + persistence)")
           except Exception as e:
               logger.error(f"Failed to start background tasks: {e}")

   async def stop_background_cleanup(self):
       """Stops all background tasks gracefully."""
       tasks_to_stop = []
       
       if self._cleanup_task and not self._cleanup_task.done():
           self._cleanup_task.cancel()
           tasks_to_stop.append(self._cleanup_task)
           
       if self._persistence_task and not self._persistence_task.done():
           self._persistence_task.cancel()
           tasks_to_stop.append(self._persistence_task)
       
       if tasks_to_stop:
           await asyncio.gather(*tasks_to_stop, return_exceptions=True)
           
       self._cleanup_task = None
       self._persistence_task = None
       logger.info("🛑 Background tasks stopped")

   def is_running(self) -> bool:
       """Checks if the background tasks are active."""
       return (self._cleanup_task is not None and not self._cleanup_task.done()) or \
              (self._persistence_task is not None and not self._persistence_task.done())

   async def _cleanup_loop(self):
       """Main loop for session cleanup."""
       while True:
           try:
               await asyncio.sleep(self.session_cleanup_interval_seconds)
               logger.info("🧹 Running periodic session cleanup...")
               
               cleanup_results = await self._perform_cleanup()
               
               logger.info(f"🧹 Cleanup complete: {cleanup_results}")
               
           except asyncio.CancelledError:
               logger.info("🧹 Cleanup loop shutting down")
               break
           except Exception as e:
               logger.error(f"Error in cleanup loop: {e}", exc_info=True)

   async def _persistence_loop(self):
       """Background loop for automatic persistence."""
       while True:
           try:
               await asyncio.sleep(self.persistence_interval_seconds)
               logger.info("💾 Running automatic persistence...")
               
               persist_results = await self._persist_all_sessions()
               
               if persist_results['persisted_count'] > 0:
                   logger.info(f"💾 Persisted {persist_results['persisted_count']} sessions")
               
           except asyncio.CancelledError:
               logger.info("💾 Persistence loop shutting down")
               break
           except Exception as e:
               logger.error(f"Error in persistence loop: {e}", exc_info=True)

   # --- Core Session Management ---

   async def get_or_create_session(self, document_id: str, filename: str) -> DocumentSession:
       """Gets existing session or creates new one with bounded memory."""
       async with self.lock:
           # If session exists, update access time
           if document_id in self.document_sessions:
               session = self.document_sessions[document_id]
               session.last_accessed = datetime.now(timezone.utc)
               self.document_sessions.move_to_end(document_id)
               return session
           
           # Create new session with memory limits
           logger.info(f"📄 Creating new session for '{filename}' (ID: {document_id})")
           session = DocumentSession(document_id, filename, self.max_items_per_session)
           
           # Load recent user data from storage if exists
           await self._load_user_recent_data(session)
           
           self.document_sessions[document_id] = session
           self.stats['sessions_created'] += 1
           
           # Check if eviction needed
           if len(self.document_sessions) > self.max_sessions:
               await self._evict_oldest_session("max session count exceeded")
           
           # Check memory limits
           if self._get_total_memory_usage() > self.max_session_memory_mb * 1024 * 1024:
               await self._evict_sessions_for_memory()

           return session

   async def get_session(self, document_id: str) -> Optional[DocumentSession]:
       """Get existing session."""
       async with self.lock:
           session = self.document_sessions.get(document_id)
           if session:
               session.last_accessed = datetime.now(timezone.utc)
               self.document_sessions.move_to_end(document_id)
           return session

   async def record_chat_activity(self, document_id: str, user: str, chat_data: Optional[Dict] = None):
       """Record chat activity with automatic persistence."""
       async with self.lock:
           session = self.document_sessions.get(document_id)
           if session:
               session.active_users[user] = datetime.now(timezone.utc).isoformat()
               
               if chat_data is None:
                   chat_data = {
                       "timestamp": datetime.now(timezone.utc).isoformat(),
                       "author": user,
                       "type": "activity"
                   }
               
               session.add_chat(chat_data)
               
               # Persist if needed
               if session.needs_persistence:
                   await self._persist_session_data(document_id, session)

   async def update_highlight_session(self, document_id: str, query_session_id: str, 
                                    pages_with_highlights: Dict[int, int],
                                    element_types: List[str], total_highlights: int,
                                    user: str = "system", query: str = ""):
       """Update or create a highlight session."""
       async with self.lock:
           session = self.document_sessions.get(document_id)
           if session:
               if query_session_id not in session.highlight_sessions:
                   highlight_session = HighlightSession(
                       query_session_id=query_session_id,
                       document_id=document_id,
                       user=user,
                       query=query
                   )
                   session.highlight_sessions[query_session_id] = highlight_session
               else:
                   highlight_session = session.highlight_sessions[query_session_id]
               
               # Update data
               highlight_session.pages_with_highlights = pages_with_highlights
               highlight_session.element_types = set(element_types)
               highlight_session.total_highlights = total_highlights
               highlight_session.update_access()
               
               session.needs_persistence = True

   async def add_annotation(self, document_id: str, annotation_data: Dict[str, Any]) -> bool:
       """Add annotation with bounded memory."""
       session = await self.get_session(document_id)
       if not session:
           logger.warning(f"No session for document: {document_id}")
           return False
           
       async with self.lock:
           session.add_annotation(annotation_data)
           self.stats['total_annotations'] += 1
           
           # Auto-persist if many changes
           if len(session.recent_annotations) >= session.max_items_in_memory:
               await self._persist_session_data(document_id, session)
       
       return True

   # --- Persistence Methods ---

   async def _persist_session_data(self, session_id: str, session: DocumentSession):
       """Persist session data to storage with compression."""
       try:
           # Create persistence timestamp
           persist_time = datetime.now(timezone.utc).isoformat()
           
           # Prepare data for persistence
           session_data = {
               "session_id": session_id,
               "persisted_at": persist_time,
               "summary": session.to_summary_dict(),
               "recent_annotations": list(session.recent_annotations),
               "recent_chats": list(session.recent_chats),
               "highlight_sessions": {
                   sid: hs.to_dict() 
                   for sid, hs in session.highlight_sessions.items()
               },
               "active_users": session.active_users
           }
           
           # Convert to JSON
           json_data = json.dumps(session_data, indent=2, ensure_ascii=False)
           
           # Compress if enabled
           if self.compress_persistence:
               data_bytes = gzip.compress(json_data.encode('utf-8'))
               blob_name = f"{session_id}_session_{persist_time.replace(':', '-')}.json.gz"
           else:
               data_bytes = json_data.encode('utf-8')
               blob_name = f"{session_id}_session_{persist_time.replace(':', '-')}.json"
           
           # Upload to storage
           await self.storage_service.upload_file(
               container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
               blob_name=blob_name,
               data=data_bytes
           )
           
           # Update session state
           session.last_persisted_at = datetime.now(timezone.utc)
           session.needs_persistence = False
           
           self.stats['sessions_persisted'] += 1
           logger.debug(f"💾 Persisted session {session_id} ({len(data_bytes)} bytes)")
           
       except Exception as e:
           logger.error(f"Failed to persist session {session_id}: {e}")
           raise

   async def _load_user_recent_data(self, session: DocumentSession):
       """Load recent user data when creating session."""
       try:
           # Try to find most recent session file
           session_files = await self.storage_service.list_blobs(
               container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
               prefix=f"{session.document_id}_session_"
           )
           
           if not session_files:
               return
           
           # Get most recent file
           session_files.sort(reverse=True)
           latest_file = session_files[0]
           
           # Download and decompress
           data_bytes = await self.storage_service.download_blob_as_bytes(
               container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
               blob_name=latest_file
           )
           
           if latest_file.endswith('.gz'):
               json_data = gzip.decompress(data_bytes).decode('utf-8')
           else:
               json_data = data_bytes.decode('utf-8')
           
           # Load data
           session_data = json.loads(json_data)
           
           # Restore recent items (they're already limited by deque maxlen)
           for ann in session_data.get('recent_annotations', [])[-session.max_items_in_memory:]:
               session.recent_annotations.append(ann)
               session.annotation_ids.add(ann.get('annotation_id', ''))
           
           for chat in session_data.get('recent_chats', [])[-session.max_items_in_memory:]:
               session.recent_chats.append(chat)
               session.chat_timestamps.append(chat.get('timestamp', ''))
           
           # Restore counts
           if 'summary' in session_data:
               summary = session_data['summary']
               session.chat_count = summary.get('chat_count', 0)
               session.annotation_count = summary.get('annotation_count', 0)
               session.metadata = summary.get('metadata', {})
           
           logger.debug(f"📥 Loaded recent data for session {session.document_id}")
           
       except Exception as e:
           logger.debug(f"No recent session data to load: {e}")

   async def _persist_all_sessions(self) -> Dict[str, int]:
       """Persist all sessions that need it."""
       persisted_count = 0
       
       async with self.lock:
           for session_id, session in self.document_sessions.items():
               if session.needs_persistence:
                   try:
                       await self._persist_session_data(session_id, session)
                       persisted_count += 1
                   except Exception as e:
                       logger.error(f"Failed to persist {session_id}: {e}")
       
       return {"persisted_count": persisted_count}

   # --- Cleanup and Eviction ---

   async def _perform_cleanup(self) -> Dict[str, int]:
       """Perform cleanup of expired data."""
       results = {'highlights_expired': 0, 'sessions_evicted': 0}
       
       async with self.lock:
           # Clean expired highlight sessions
           for doc_session in self.document_sessions.values():
               expired_ids = [
                   sid for sid, hs in doc_session.highlight_sessions.items() 
                   if hs.is_expired()
               ]
               
               for sid in expired_ids:
                   del doc_session.highlight_sessions[sid]
                   results['highlights_expired'] += 1
           
           # Persist before evicting
           await self._persist_all_sessions()
           
           # Check memory usage
           while self._get_total_memory_usage() > self.max_session_memory_mb * 1024 * 1024:
               await self._evict_oldest_session("memory limit exceeded")
               results['sessions_evicted'] += 1
       
       return results

   async def _evict_oldest_session(self, reason: str):
       """Evict oldest session after persisting."""
       if not self.document_sessions:
           return
       
       # Get oldest session
       oldest_id, oldest_session = self.document_sessions.popitem(last=False)
       
       # Persist if needed
       if oldest_session.needs_persistence:
           await self._persist_session_data(oldest_id, oldest_session)
       
       self.stats['sessions_evicted'] += 1
       logger.info(f"🗑️ Evicted session {oldest_id} ({reason})")

   async def _evict_sessions_for_memory(self):
       """Evict sessions to meet memory limit."""
       target_memory = self.max_session_memory_mb * 1024 * 1024 * 0.8  # 80% target
       
       while self._get_total_memory_usage() > target_memory and len(self.document_sessions) > 1:
           await self._evict_oldest_session("memory pressure")

   def _get_total_memory_usage(self) -> int:
       """Get total memory usage."""
       if not self.document_sessions:
           return 0
       return sum(s.get_memory_usage_estimate() for s in self.document_sessions.values())

   # --- Information and Statistics ---

   def get_session_info(self, document_id: str) -> Optional[Dict[str, Any]]:
       """Get session information."""
       session = self.document_sessions.get(document_id)
       if not session:
           return None
       
       return {
           "document_id": session.document_id,
           "filename": session.filename,
           "created_at": session.created_at.isoformat(),
           "last_accessed": session.last_accessed.isoformat(),
           "last_persisted_at": session.last_persisted_at.isoformat(),
           "needs_persistence": session.needs_persistence,
           "chat_count": session.chat_count,
           "annotation_count": session.annotation_count,
           "recent_items_in_memory": len(session.recent_annotations) + len(session.recent_chats),
           "active_highlight_sessions": len(session.highlight_sessions),
           "active_users": list(session.active_users.keys()),
           "memory_usage_bytes": session.get_memory_usage_estimate()
       }

   def get_service_statistics(self) -> Dict[str, Any]:
       """Get service statistics."""
       total_memory = self._get_total_memory_usage()
       
       return {
           "service_name": "SessionService",
           "status": "running" if self.is_running() else "stopped",
           "configuration": {
               "max_sessions": self.max_sessions,
               "max_memory_mb": self.max_session_memory_mb,
               "max_items_per_session": self.max_items_per_session,
               "persistence_interval_seconds": self.persistence_interval_seconds,
               "cleanup_interval_seconds": self.session_cleanup_interval_seconds,
               "compression_enabled": self.compress_persistence
           },
           "current_state": {
               "active_sessions": len(self.document_sessions),
               "total_memory_usage_mb": round(total_memory / (1024 * 1024), 2),
               "memory_utilization_percent": round((total_memory / (self.max_session_memory_mb * 1024 * 1024)) * 100, 1)
           },
           "lifetime_statistics": dict(self.stats)
       }

   # --- Compatibility Methods ---

   async def create_session(self, document_id: str, original_filename: str):
       """Create a new session (compatibility method)."""
       return await self.get_or_create_session(document_id, original_filename)

   async def update_session_metadata(self, document_id: str, metadata: Dict[str, Any]):
       """Update session metadata."""
       async with self.lock:
           session = self.document_sessions.get(document_id)
           if session:
               session.metadata.update(metadata)
               session.last_accessed = datetime.now(timezone.utc)
               session.needs_persistence = True

   # --- Shutdown ---

   async def shutdown(self):
       """Graceful shutdown with final persistence."""
       logger.info("🛑 SessionService shutting down...")
       
       # Stop background tasks
       await self.stop_background_cleanup()
       
       # Final persistence of all sessions
       await self._persist_all_sessions()
       
       logger.info("✅ SessionService shutdown complete")

   # --- Context Manager ---

   async def __aenter__(self):
       await self.start_background_cleanup()
       return self

   async def __aexit__(self, exc_type, exc_val, exc_tb):
       await self.shutdown()
