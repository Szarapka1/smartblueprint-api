# app/services/session_service.py - BULLETPROOF PRODUCTION VERSION

import uuid
import logging
import asyncio
import json
import sys
import gzip
import time
import os
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta, timezone
from collections import OrderedDict, defaultdict, deque
from functools import wraps
import traceback

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# --- Decorators for reliability ---

def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"⏰ Timeout in {func.__name__} after {timeout_seconds}s")
                raise TimeoutError(f"Operation {func.__name__} timed out after {timeout_seconds}s")
        return wrapper
    return decorator

def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to add retry logic to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except asyncio.CancelledError:
                    raise  # Don't retry on cancellation
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ All {max_attempts} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

# --- Data Models with validation ---

class HighlightSession:
    """Represents a highlight session with comprehensive validation and error handling"""
    
    def __init__(self, query_session_id: str, document_id: str, user: str, query: str):
        # Validate inputs
        if not query_session_id or not isinstance(query_session_id, str):
            raise ValueError("Invalid query_session_id")
        if not document_id or not isinstance(document_id, str):
            raise ValueError("Invalid document_id")
        if not user or not isinstance(user, str):
            raise ValueError("Invalid user")
        
        self.query_session_id = query_session_id
        self.document_id = document_id
        self.user = user[:100]  # Limit user length
        self.query = (query or "")[:500]  # Limit query length
        
        # Use timezone-aware UTC datetime
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.created_at
        self.expires_at: datetime = self.created_at + timedelta(hours=24)
        
        # Data storage with limits
        self.pages_with_highlights: Dict[int, int] = {}
        self.element_types: Set[str] = set()
        self.total_highlights: int = 0
        self.is_active: bool = True
        self.access_count: int = 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary with error handling"""
        try:
            return {
                'query_session_id': self.query_session_id,
                'document_id': self.document_id,
                'user': self.user,
                'query': self.query,
                'created_at': self.created_at.isoformat(),
                'last_accessed': self.last_accessed.isoformat(),
                'expires_at': self.expires_at.isoformat(),
                'pages_with_highlights': dict(self.pages_with_highlights),  # Ensure dict
                'element_types': list(self.element_types)[:50],  # Limit size
                'total_highlights': self.total_highlights,
                'is_active': self.is_active,
                'access_count': self.access_count
            }
        except Exception as e:
            logger.error(f"Error converting HighlightSession to dict: {e}")
            # Return minimal valid dict
            return {
                'query_session_id': self.query_session_id,
                'document_id': self.document_id,
                'error': str(e)
            }
    
    def update_access(self):
        """Updates access time and count"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """Checks if session expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def extend_expiry(self, hours: int = 24):
        """Extends expiry time"""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)

class DocumentSession:
    """Document session with bounded memory and error recovery"""
    
    def __init__(self, document_id: str, filename: str, max_items_in_memory: int = 100):
        # Validate inputs
        if not document_id or not isinstance(document_id, str):
            raise ValueError("Invalid document_id")
        
        self.document_id = document_id
        self.filename = (filename or "unknown.pdf")[:255]  # Limit filename length
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.created_at
        
        # Memory bounds
        self.max_items_in_memory = min(max_items_in_memory, 1000)  # Hard limit
        
        # Use deque for automatic size limits (FIFO)
        self.recent_annotations = deque(maxlen=self.max_items_in_memory)
        self.recent_chats = deque(maxlen=self.max_items_in_memory)
        
        # Indexes for fast lookup
        self.annotation_ids: Set[str] = set()
        self.chat_timestamps: deque = deque(maxlen=self.max_items_in_memory * 2)
        
        # Highlight sessions with limit
        self.highlight_sessions: OrderedDict[str, HighlightSession] = OrderedDict()
        self.max_highlight_sessions = 100
        
        # Activity tracking
        self.active_users: OrderedDict[str, str] = OrderedDict()
        self.max_active_users = 50
        
        # Counters and metadata
        self.chat_count: int = 0
        self.annotation_count: int = 0
        self.access_count: int = 0
        self.metadata: Dict[str, Any] = {}
        
        # Persistence tracking
        self.last_persisted_at: datetime = datetime.now(timezone.utc)
        self.needs_persistence: bool = False
        self.persistence_failures: int = 0
        
        # Error tracking
        self.errors: deque = deque(maxlen=10)
        
    def add_annotation(self, annotation: Dict[str, Any]) -> bool:
        """Add annotation with validation"""
        try:
            # Validate annotation
            if not isinstance(annotation, dict):
                raise ValueError("Annotation must be a dictionary")
            
            # Ensure required fields
            if 'annotation_id' not in annotation:
                annotation['annotation_id'] = str(uuid.uuid4())
            
            # Add timestamp if missing
            if 'timestamp' not in annotation:
                annotation['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Limit annotation size
            annotation_str = json.dumps(annotation)
            if len(annotation_str) > 10000:  # 10KB limit per annotation
                logger.warning(f"Annotation too large: {len(annotation_str)} bytes")
                return False
            
            self.recent_annotations.append(annotation)
            self.annotation_ids.add(annotation['annotation_id'])
            self.annotation_count += 1
            self.needs_persistence = True
            self.last_accessed = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add annotation: {e}")
            self.errors.append({'type': 'annotation_add', 'error': str(e), 'time': datetime.now(timezone.utc).isoformat()})
            return False
    
    def add_chat(self, chat_data: Dict[str, Any]) -> bool:
        """Add chat with validation"""
        try:
            # Validate chat data
            if not isinstance(chat_data, dict):
                raise ValueError("Chat data must be a dictionary")
            
            # Add timestamp if missing
            if 'timestamp' not in chat_data:
                chat_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Limit chat size
            if 'prompt' in chat_data and len(chat_data['prompt']) > 2000:
                chat_data['prompt'] = chat_data['prompt'][:2000] + "..."
            if 'response' in chat_data and len(chat_data['response']) > 5000:
                chat_data['response'] = chat_data['response'][:5000] + "..."
            
            self.recent_chats.append(chat_data)
            self.chat_timestamps.append(chat_data['timestamp'])
            self.chat_count += 1
            self.needs_persistence = True
            self.last_accessed = datetime.now(timezone.utc)
            
            # Track active user
            if 'author' in chat_data:
                self.track_active_user(chat_data['author'])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chat: {e}")
            self.errors.append({'type': 'chat_add', 'error': str(e), 'time': datetime.now(timezone.utc).isoformat()})
            return False
    
    def track_active_user(self, user: str):
        """Track active user with bounds"""
        if user and isinstance(user, str):
            self.active_users[user] = datetime.now(timezone.utc).isoformat()
            self.active_users.move_to_end(user)
            
            # Limit active users
            while len(self.active_users) > self.max_active_users:
                self.active_users.popitem(last=False)
    
    def add_highlight_session(self, highlight_session: HighlightSession) -> bool:
        """Add highlight session with bounds"""
        try:
            session_id = highlight_session.query_session_id
            
            # Check limits
            if len(self.highlight_sessions) >= self.max_highlight_sessions:
                # Remove oldest expired session
                expired_removed = False
                for sid, session in list(self.highlight_sessions.items()):
                    if session.is_expired():
                        del self.highlight_sessions[sid]
                        expired_removed = True
                        break
                
                # If no expired sessions, remove oldest
                if not expired_removed and self.highlight_sessions:
                    self.highlight_sessions.popitem(last=False)
            
            self.highlight_sessions[session_id] = highlight_session
            self.needs_persistence = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add highlight session: {e}")
            return False
    
    def get_memory_usage_estimate(self) -> int:
        """Get memory usage estimate with error handling"""
        try:
            size = 0
            
            # Base object
            size += sys.getsizeof(self)
            
            # Collections
            size += sys.getsizeof(self.recent_annotations) + sum(sys.getsizeof(json.dumps(a)) for a in self.recent_annotations)
            size += sys.getsizeof(self.recent_chats) + sum(sys.getsizeof(json.dumps(c)) for c in self.recent_chats)
            size += sys.getsizeof(self.annotation_ids) + sum(sys.getsizeof(aid) for aid in self.annotation_ids)
            size += sys.getsizeof(self.chat_timestamps) + sum(sys.getsizeof(ts) for ts in self.chat_timestamps)
            
            # Highlight sessions
            size += sys.getsizeof(self.highlight_sessions)
            for session in self.highlight_sessions.values():
                size += sys.getsizeof(json.dumps(session.to_dict()))
            
            # Other data
            size += sys.getsizeof(self.active_users) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.active_users.items())
            size += sys.getsizeof(self.metadata) + sys.getsizeof(json.dumps(self.metadata))
            
            return size
            
        except Exception as e:
            logger.error(f"Error calculating memory usage: {e}")
            # Return conservative estimate
            return 10000 + (len(self.recent_annotations) + len(self.recent_chats)) * 5000
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary for persistence"""
        try:
            return {
                "document_id": self.document_id,
                "filename": self.filename,
                "created_at": self.created_at.isoformat(),
                "last_accessed": self.last_accessed.isoformat(),
                "chat_count": self.chat_count,
                "annotation_count": self.annotation_count,
                "access_count": self.access_count,
                "active_users": list(self.active_users.keys())[:20],  # Limit
                "highlight_sessions_count": len(self.highlight_sessions),
                "annotation_ids_count": len(self.annotation_ids),
                "metadata": self.metadata,
                "last_persisted_at": self.last_persisted_at.isoformat(),
                "persistence_failures": self.persistence_failures,
                "error_count": len(self.errors)
            }
        except Exception as e:
            logger.error(f"Error creating summary dict: {e}")
            return {
                "document_id": self.document_id,
                "error": str(e)
            }

# --- Main Service Class ---

class SessionService:
    """
    Bulletproof session management with automatic recovery and persistence
    """
    
    def __init__(self, settings: AppSettings, storage_service: StorageService):
        if not settings:
            raise ValueError("AppSettings required")
        if not storage_service:
            raise ValueError("StorageService required")
        
        self.settings = settings
        self.storage_service = storage_service
        
        # Use OrderedDict for LRU behavior
        self.document_sessions: OrderedDict[str, DocumentSession] = OrderedDict()
        
        # Thread safety
        self.lock = asyncio.Lock()
        self.persistence_lock = asyncio.Lock()
        
        # Configuration with safe defaults
        self.max_sessions = min(settings.MAX_SESSIONS_IN_MEMORY, 1000)  # Hard limit
        self.max_session_memory_mb = min(settings.MAX_SESSION_MEMORY_MB, 2048)  # 2GB hard limit
        self.session_cleanup_interval = max(settings.SESSION_CLEANUP_INTERVAL_SECONDS, 300)  # Min 5 minutes
        self.session_expiry_hours = min(settings.SESSION_CLEANUP_HOURS, 168)  # Max 1 week
        
        # Memory management
        self.max_items_per_session = min(100, settings.MAX_ANNOTATIONS_PER_SESSION)
        self.persistence_interval = 300  # 5 minutes
        self.persistence_batch_size = 10  # Sessions to persist at once
        self.compress_persistence = True
        self.compression_level = 6  # Balance speed vs size
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Statistics and health
        self.stats = defaultdict(int)
        self.health_status = {
            "healthy": True,
            "last_cleanup": None,
            "last_persistence": None,
            "consecutive_failures": 0
        }
        
        # Error recovery
        self.max_consecutive_failures = 5
        self.recovery_delay = 30  # seconds
        
        logger.info("🚀 SessionService initialized (BULLETPROOF VERSION)")
        logger.info(f"📊 Configuration:")
        logger.info(f"   Max sessions: {self.max_sessions}")
        logger.info(f"   Max memory: {self.max_session_memory_mb}MB")
        logger.info(f"   Max items/session: {self.max_items_per_session}")
        logger.info(f"   Persistence interval: {self.persistence_interval}s")
        logger.info(f"   Cleanup interval: {self.session_cleanup_interval}s")
        logger.info(f"   Session expiry: {self.session_expiry_hours}h")

    # --- Lifecycle Management ---

    async def start_background_cleanup(self):
        """Start all background tasks with error recovery"""
        try:
            if not self.is_running():
                logger.info("🚀 Starting background tasks...")
                
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                self._persistence_task = asyncio.create_task(self._persistence_loop())
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                # Give tasks names for better debugging
                self._cleanup_task.set_name("session_cleanup")
                self._persistence_task.set_name("session_persistence")
                self._health_check_task.set_name("session_health_check")
                
                logger.info("✅ Background tasks started successfully")
                
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            logger.error(traceback.format_exc())
            self.health_status["healthy"] = False

    async def stop_background_cleanup(self):
        """Stop all background tasks gracefully"""
        logger.info("🛑 Stopping background tasks...")
        
        tasks_to_stop = []
        
        for task_name, task in [
            ("cleanup", self._cleanup_task),
            ("persistence", self._persistence_task),
            ("health_check", self._health_check_task)
        ]:
            if task and not task.done():
                logger.info(f"   Cancelling {task_name} task...")
                task.cancel()
                tasks_to_stop.append(task)
        
        if tasks_to_stop:
            # Wait for tasks to finish with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_stop, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Some tasks didn't stop cleanly within timeout")
        
        self._cleanup_task = None
        self._persistence_task = None
        self._health_check_task = None
        
        logger.info("✅ Background tasks stopped")

    def is_running(self) -> bool:
        """Check if background tasks are running"""
        return any(
            task and not task.done() 
            for task in [self._cleanup_task, self._persistence_task, self._health_check_task]
        )

    # --- Background Loops with Error Recovery ---

    async def _cleanup_loop(self):
        """Main cleanup loop with error recovery"""
        consecutive_failures = 0
        
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval)
                
                logger.info("🧹 Running periodic cleanup...")
                start_time = time.time()
                
                cleanup_results = await self._perform_cleanup()
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Cleanup complete in {elapsed:.1f}s: {cleanup_results}")
                
                self.health_status["last_cleanup"] = datetime.now(timezone.utc).isoformat()
                consecutive_failures = 0
                
            except asyncio.CancelledError:
                logger.info("🛑 Cleanup loop cancelled")
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"❌ Cleanup loop error (attempt {consecutive_failures}): {e}")
                logger.error(traceback.format_exc())
                
                if consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"⚠️ Too many cleanup failures, pausing for {self.recovery_delay}s")
                    await asyncio.sleep(self.recovery_delay)
                    consecutive_failures = 0

    async def _persistence_loop(self):
        """Persistence loop with batching and error recovery"""
        consecutive_failures = 0
        
        while True:
            try:
                await asyncio.sleep(self.persistence_interval)
                
                logger.info("💾 Running automatic persistence...")
                start_time = time.time()
                
                persist_results = await self._persist_all_sessions()
                
                elapsed = time.time() - start_time
                
                if persist_results['persisted_count'] > 0:
                    logger.info(f"✅ Persisted {persist_results['persisted_count']} sessions in {elapsed:.1f}s")
                    if persist_results['failed_count'] > 0:
                        logger.warning(f"⚠️ Failed to persist {persist_results['failed_count']} sessions")
                
                self.health_status["last_persistence"] = datetime.now(timezone.utc).isoformat()
                consecutive_failures = 0
                
            except asyncio.CancelledError:
                logger.info("🛑 Persistence loop cancelled")
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"❌ Persistence loop error (attempt {consecutive_failures}): {e}")
                logger.error(traceback.format_exc())
                
                if consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"⚠️ Too many persistence failures, pausing for {self.recovery_delay}s")
                    await asyncio.sleep(self.recovery_delay)
                    consecutive_failures = 0

    async def _health_check_loop(self):
        """Health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check memory usage
                memory_usage = self._get_total_memory_usage()
                memory_percent = (memory_usage / (self.max_session_memory_mb * 1024 * 1024)) * 100
                
                if memory_percent > 90:
                    logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}%")
                    # Force cleanup
                    await self._perform_cleanup()
                
                # Check consecutive failures
                if self.health_status["consecutive_failures"] > self.max_consecutive_failures:
                    logger.error("❌ Service unhealthy - too many consecutive failures")
                    self.health_status["healthy"] = False
                else:
                    self.health_status["healthy"] = True
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    # --- Core Session Management ---

    @with_retry(max_attempts=3, delay=0.5)
    @with_timeout(30.0)
    async def get_or_create_session(self, document_id: str, filename: str) -> DocumentSession:
        """Get existing or create new session with error recovery"""
        # Validate inputs
        if not document_id or not isinstance(document_id, str):
            raise ValueError("Invalid document_id")
        
        document_id = document_id[:100]  # Limit length
        
        async with self.lock:
            try:
                # Check if session exists
                if document_id in self.document_sessions:
                    session = self.document_sessions[document_id]
                    session.last_accessed = datetime.now(timezone.utc)
                    session.access_count += 1
                    
                    # Move to end (LRU)
                    self.document_sessions.move_to_end(document_id)
                    
                    logger.debug(f"📄 Retrieved existing session for {document_id}")
                    return session
                
                # Create new session
                logger.info(f"📄 Creating new session for '{filename}' (ID: {document_id})")
                
                session = DocumentSession(document_id, filename, self.max_items_per_session)
                
                # Try to load recent data
                try:
                    await self._load_recent_session_data(session)
                except Exception as e:
                    logger.warning(f"Could not load recent data: {e}")
                    # Continue anyway - new session
                
                self.document_sessions[document_id] = session
                self.stats['sessions_created'] += 1
                
                # Check limits
                await self._enforce_session_limits()
                
                return session
                
            except Exception as e:
                self.stats['session_creation_errors'] += 1
                logger.error(f"Failed to create session: {e}")
                raise

    async def get_session(self, document_id: str) -> Optional[DocumentSession]:
        """Get existing session"""
        if not document_id:
            return None
        
        async with self.lock:
            session = self.document_sessions.get(document_id)
            if session:
                session.last_accessed = datetime.now(timezone.utc)
                session.access_count += 1
                self.document_sessions.move_to_end(document_id)
            return session

    async def create_session(self, document_id: str, original_filename: str):
        """Create a new session (compatibility method)"""
        return await self.get_or_create_session(document_id, original_filename)

    @with_retry(max_attempts=2, delay=0.5)
    async def record_chat_activity(self, document_id: str, user: str, chat_data: Optional[Dict] = None):
        """Record chat activity with error recovery"""
        try:
            session = await self.get_session(document_id)
            if not session:
                logger.warning(f"No session found for document: {document_id}")
                return
            
            async with self.lock:
                # Track user
                session.track_active_user(user)
                
                # Prepare chat data
                if chat_data is None:
                    chat_data = {}
                
                chat_data.update({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "author": user[:100],  # Limit length
                    "type": "activity"
                })
                
                # Add chat
                if session.add_chat(chat_data):
                    self.stats['chat_activities_recorded'] += 1
                    
                    # Auto-persist if many changes
                    if len(session.recent_chats) % 50 == 0:
                        asyncio.create_task(self._persist_session_data(document_id, session))
                
        except Exception as e:
            self.stats['chat_record_errors'] += 1
            logger.error(f"Failed to record chat activity: {e}")

    async def add_annotation(self, document_id: str, annotation_data: Dict[str, Any]) -> bool:
        """Add annotation with validation"""
        try:
            session = await self.get_session(document_id)
            if not session:
                logger.warning(f"No session for document: {document_id}")
                return False
            
            async with self.lock:
                if session.add_annotation(annotation_data):
                    self.stats['annotations_added'] += 1
                    
                    # Auto-persist if many changes
                    if len(session.recent_annotations) % 25 == 0:
                        asyncio.create_task(self._persist_session_data(document_id, session))
                    
                    return True
                
                return False
                
        except Exception as e:
            self.stats['annotation_add_errors'] += 1
            logger.error(f"Failed to add annotation: {e}")
            return False

    @with_retry(max_attempts=2, delay=0.5)
    async def update_highlight_session(
        self, document_id: str, query_session_id: str, 
        pages_with_highlights: Dict[int, int], element_types: List[str], 
        total_highlights: int, user: str = "system", query: str = ""
    ):
        """Update or create highlight session"""
        try:
            session = await self.get_session(document_id)
            if not session:
                logger.warning(f"No session for document: {document_id}")
                return
            
            async with self.lock:
                # Create or update highlight session
                if query_session_id in session.highlight_sessions:
                    highlight_session = session.highlight_sessions[query_session_id]
                    highlight_session.update_access()
                else:
                    highlight_session = HighlightSession(
                        query_session_id=query_session_id,
                        document_id=document_id,
                        user=user,
                        query=query
                    )
                
                # Update data with limits
                highlight_session.pages_with_highlights = dict(list(pages_with_highlights.items())[:100])
                highlight_session.element_types = set(element_types[:50])
                highlight_session.total_highlights = min(total_highlights, 10000)
                
                # Add to session
                if session.add_highlight_session(highlight_session):
                    self.stats['highlight_sessions_updated'] += 1
                
        except Exception as e:
            self.stats['highlight_update_errors'] += 1
            logger.error(f"Failed to update highlight session: {e}")

    async def update_session_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update session metadata"""
        try:
            session = await self.get_session(document_id)
            if not session:
                return
            
            async with self.lock:
                # Merge metadata with size limit
                metadata_str = json.dumps(metadata)
                if len(metadata_str) > 50000:  # 50KB limit
                    logger.warning("Metadata too large, truncating")
                    return
                
                session.metadata.update(metadata)
                session.needs_persistence = True
                
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")

    # --- Persistence Methods ---

    @with_retry(max_attempts=3, delay=1.0)
    @with_timeout(60.0)
    async def _persist_session_data(self, session_id: str, session: DocumentSession) -> bool:
        """Persist session data with compression and error recovery"""
        async with self.persistence_lock:
            try:
                start_time = time.time()
                
                # Skip if recently persisted
                time_since_last = (datetime.now(timezone.utc) - session.last_persisted_at).total_seconds()
                if time_since_last < 30 and not session.needs_persistence:
                    return True
                
                # Prepare data
                session_data = {
                    "session_id": session_id,
                    "persisted_at": datetime.now(timezone.utc).isoformat(),
                    "summary": session.to_summary_dict(),
                    "recent_annotations": list(session.recent_annotations)[-50:],  # Last 50
                    "recent_chats": list(session.recent_chats)[-50:],  # Last 50
                    "highlight_sessions": {
                        sid: hs.to_dict() 
                        for sid, hs in list(session.highlight_sessions.items())[-20:]  # Last 20
                    },
                    "active_users": dict(list(session.active_users.items())[-20:])  # Last 20
                }
                
                # Convert to JSON
                json_data = json.dumps(session_data, ensure_ascii=False, separators=(',', ':'))
                
                # Generate safe filename
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                
                # Compress if enabled
                if self.compress_persistence:
                    data_bytes = gzip.compress(
                        json_data.encode('utf-8'), 
                        compresslevel=self.compression_level
                    )
                    blob_name = f"{session_id}_session_{timestamp}.json.gz"
                    content_type = "application/gzip"
                else:
                    data_bytes = json_data.encode('utf-8')
                    blob_name = f"{session_id}_session_{timestamp}.json"
                    content_type = "application/json"
                
                # Upload to storage
                await self.storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name,
                    data=data_bytes,
                    content_type=content_type,
                    metadata={
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "compressed": str(self.compress_persistence)
                    }
                )
                
                # Update session state
                session.last_persisted_at = datetime.now(timezone.utc)
                session.needs_persistence = False
                session.persistence_failures = 0
                
                elapsed = time.time() - start_time
                self.stats['sessions_persisted'] += 1
                
                logger.debug(f"💾 Persisted session {session_id} ({len(data_bytes)} bytes) in {elapsed:.1f}s")
                
                return True
                
            except Exception as e:
                session.persistence_failures += 1
                self.stats['persistence_errors'] += 1
                self.health_status["consecutive_failures"] += 1
                
                logger.error(f"Failed to persist session {session_id}: {e}")
                logger.error(traceback.format_exc())
                
                return False

    @with_timeout(120.0)
    async def _load_recent_session_data(self, session: DocumentSession):
        """Load recent session data with error recovery"""
        try:
            # Find most recent session file
            session_files = await self.storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=f"{session.document_id}_session_",
                max_results=10
            )
            
            if not session_files:
                logger.debug(f"No previous session data for {session.document_id}")
                return
            
            # Sort by timestamp (newest first)
            session_files.sort(reverse=True)
            
            # Try to load most recent valid file
            for session_file in session_files[:3]:  # Try up to 3 most recent
                try:
                    # Download data
                    data_bytes = await self.storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=session_file
                    )
                    
                    # Decompress if needed
                    if session_file.endswith('.gz'):
                        json_data = gzip.decompress(data_bytes).decode('utf-8')
                    else:
                        json_data = data_bytes.decode('utf-8')
                    
                    # Parse JSON
                    session_data = json.loads(json_data)
                    
                    # Restore data with validation
                    self._restore_session_data(session, session_data)
                    
                    logger.info(f"📥 Loaded session data from {session_file}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {session_file}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not load session history: {e}")
            # Not critical - continue with empty session

    def _restore_session_data(self, session: DocumentSession, session_data: Dict[str, Any]):
        """Restore session data with validation"""
        try:
            # Restore recent items (already limited by deque maxlen)
            for ann in session_data.get('recent_annotations', []):
                if isinstance(ann, dict):
                    session.add_annotation(ann)
            
            for chat in session_data.get('recent_chats', []):
                if isinstance(chat, dict):
                    session.add_chat(chat)
            
            # Restore counts from summary
            if 'summary' in session_data:
                summary = session_data['summary']
                session.chat_count = max(session.chat_count, summary.get('chat_count', 0))
                session.annotation_count = max(session.annotation_count, summary.get('annotation_count', 0))
                
                # Restore metadata
                if isinstance(summary.get('metadata'), dict):
                    session.metadata.update(summary['metadata'])
            
            # Mark as not needing immediate persistence
            session.needs_persistence = False
            
        except Exception as e:
            logger.error(f"Error restoring session data: {e}")

    async def _persist_all_sessions(self) -> Dict[str, int]:
        """Persist all sessions that need it"""
        results = {"persisted_count": 0, "failed_count": 0}
        
        # Get sessions needing persistence
        sessions_to_persist = []
        
        async with self.lock:
            for session_id, session in self.document_sessions.items():
                if session.needs_persistence or session.persistence_failures > 0:
                    sessions_to_persist.append((session_id, session))
        
        if not sessions_to_persist:
            return results
        
        # Persist in batches
        for i in range(0, len(sessions_to_persist), self.persistence_batch_size):
            batch = sessions_to_persist[i:i + self.persistence_batch_size]
            
            # Persist each session in batch
            tasks = []
            for session_id, session in batch:
                task = self._persist_session_data(session_id, session)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            for result in batch_results:
                if isinstance(result, bool) and result:
                    results["persisted_count"] += 1
                else:
                    results["failed_count"] += 1
        
        return results

    # --- Cleanup and Memory Management ---

    async def _perform_cleanup(self) -> Dict[str, int]:
        """Perform comprehensive cleanup"""
        results = {
            'expired_highlights': 0,
            'evicted_sessions': 0,
            'cleaned_items': 0
        }
        
        try:
            async with self.lock:
                # Clean expired highlight sessions
                for doc_session in list(self.document_sessions.values()):
                    expired_ids = [
                        sid for sid, hs in doc_session.highlight_sessions.items() 
                        if hs.is_expired()
                    ]
                    
                    for sid in expired_ids:
                        del doc_session.highlight_sessions[sid]
                        results['expired_highlights'] += 1
                
                # Clean old items from sessions
                for doc_session in list(self.document_sessions.values()):
                    # Clean old error logs
                    while len(doc_session.errors) > 5:
                        doc_session.errors.popleft()
                        results['cleaned_items'] += 1
                
                # Persist before evicting
                await self._persist_all_sessions()
                
                # Evict old sessions
                results['evicted_sessions'] = await self._evict_old_sessions()
                
                # Memory pressure check
                await self._check_memory_pressure()
            
            self.stats['cleanups_performed'] += 1
            
        except Exception as e:
            self.stats['cleanup_errors'] += 1
            logger.error(f"Cleanup error: {e}")
            logger.error(traceback.format_exc())
        
        return results

    async def _evict_old_sessions(self) -> int:
        """Evict old sessions based on age and activity"""
        evicted = 0
        
        # Calculate expiry time
        expiry_cutoff = datetime.now(timezone.utc) - timedelta(hours=self.session_expiry_hours)
        
        # Find sessions to evict
        sessions_to_evict = []
        
        for session_id, session in self.document_sessions.items():
            # Check if session is old and inactive
            if session.last_accessed < expiry_cutoff and session.access_count < 5:
                sessions_to_evict.append(session_id)
        
        # Evict sessions
        for session_id in sessions_to_evict:
            session = self.document_sessions.pop(session_id, None)
            if session:
                # Try to persist one last time
                if session.needs_persistence:
                    await self._persist_session_data(session_id, session)
                
                evicted += 1
                logger.info(f"🗑️ Evicted old session: {session_id}")
        
        return evicted

    async def _enforce_session_limits(self):
        """Enforce session count and memory limits"""
        # Check session count
        while len(self.document_sessions) > self.max_sessions:
            # Evict oldest (LRU)
            if self.document_sessions:
                session_id, session = self.document_sessions.popitem(last=False)
                
                # Persist if needed
                if session.needs_persistence:
                    await self._persist_session_data(session_id, session)
                
                self.stats['sessions_evicted'] += 1
                logger.info(f"🗑️ Evicted session {session_id} (max sessions exceeded)")
        
        # Check memory usage
        await self._check_memory_pressure()

    async def _check_memory_pressure(self):
        """Check and handle memory pressure"""
        total_memory = self._get_total_memory_usage()
        memory_limit = self.max_session_memory_mb * 1024 * 1024
        
        if total_memory > memory_limit * 0.9:  # 90% threshold
            logger.warning(f"⚠️ High memory usage: {total_memory / (1024 * 1024):.1f}MB")
            
            # Evict sessions until under 80%
            target_memory = memory_limit * 0.8
            
            while total_memory > target_memory and len(self.document_sessions) > 1:
                # Evict oldest
                session_id, session = self.document_sessions.popitem(last=False)
                
                # Persist if needed
                if session.needs_persistence:
                    await self._persist_session_data(session_id, session)
                
                self.stats['sessions_evicted_memory'] += 1
                logger.info(f"🗑️ Evicted session {session_id} (memory pressure)")
                
                # Recalculate
                total_memory = self._get_total_memory_usage()

    def _get_total_memory_usage(self) -> int:
        """Calculate total memory usage"""
        try:
            if not self.document_sessions:
                return 0
            
            total = sum(
                session.get_memory_usage_estimate() 
                for session in self.document_sessions.values()
            )
            
            # Add overhead estimate
            total += len(self.document_sessions) * 1000
            
            return total
            
        except Exception as e:
            logger.error(f"Error calculating memory: {e}")
            # Return conservative estimate
            return len(self.document_sessions) * 100000

    # --- Information and Statistics ---

    async def get_session_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = await self.get_session(document_id)
        if not session:
            return None
        
        try:
            return {
                "document_id": session.document_id,
                "filename": session.filename,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "last_persisted_at": session.last_persisted_at.isoformat(),
                "needs_persistence": session.needs_persistence,
                "persistence_failures": session.persistence_failures,
                "chat_count": session.chat_count,
                "annotation_count": session.annotation_count,
                "access_count": session.access_count,
                "recent_items_in_memory": len(session.recent_annotations) + len(session.recent_chats),
                "active_highlight_sessions": len(session.highlight_sessions),
                "active_users": list(session.active_users.keys())[:10],  # First 10
                "memory_usage_bytes": session.get_memory_usage_estimate(),
                "errors": len(session.errors)
            }
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {"document_id": document_id, "error": str(e)}

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        try:
            total_memory = self._get_total_memory_usage()
            
            return {
                "service_name": "SessionService",
                "version": "2.0-bulletproof",
                "status": "healthy" if self.health_status["healthy"] else "degraded",
                "running": self.is_running(),
                "uptime_seconds": None,  # Could track if needed
                "configuration": {
                    "max_sessions": self.max_sessions,
                    "max_memory_mb": self.max_session_memory_mb,
                    "max_items_per_session": self.max_items_per_session,
                    "persistence_interval": self.persistence_interval,
                    "cleanup_interval": self.session_cleanup_interval,
                    "session_expiry_hours": self.session_expiry_hours,
                    "compression_enabled": self.compress_persistence
                },
                "current_state": {
                    "active_sessions": len(self.document_sessions),
                    "total_memory_mb": round(total_memory / (1024 * 1024), 2),
                    "memory_usage_percent": round((total_memory / (self.max_session_memory_mb * 1024 * 1024)) * 100, 1),
                    "total_annotations": sum(s.annotation_count for s in self.document_sessions.values()),
                    "total_chats": sum(s.chat_count for s in self.document_sessions.values()),
                    "total_highlights": sum(len(s.highlight_sessions) for s in self.document_sessions.values())
                },
                "health": {
                    "healthy": self.health_status["healthy"],
                    "last_cleanup": self.health_status["last_cleanup"],
                    "last_persistence": self.health_status["last_persistence"],
                    "consecutive_failures": self.health_status["consecutive_failures"]
                },
                "lifetime_stats": dict(self.stats)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    # --- Shutdown ---

    async def shutdown(self):
        """Graceful shutdown with final persistence"""
        logger.info("🛑 SessionService shutting down...")
        
        try:
            # Stop background tasks
            await self.stop_background_cleanup()
            
            # Final persistence of all sessions
            logger.info("💾 Performing final persistence...")
            final_results = await self._persist_all_sessions()
            logger.info(f"   Persisted {final_results['persisted_count']} sessions")
            
            if final_results['failed_count'] > 0:
                logger.warning(f"   Failed to persist {final_results['failed_count']} sessions")
            
            # Clear sessions
            self.document_sessions.clear()
            
            logger.info("✅ SessionService shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.error(traceback.format_exc())

    # --- Context Manager ---

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_background_cleanup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()

# === END OF FILE ===