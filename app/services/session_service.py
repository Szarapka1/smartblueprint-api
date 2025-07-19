# app/services/session_service.py - ULTRA-RELIABLE VERSION

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

# --- Enhanced Decorators for Ultra-Reliability ---

def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions with logging"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                logger.debug(f"⏱️ Starting {func.__name__} with {timeout_seconds}s timeout")
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                logger.debug(f"✅ {func.__name__} completed successfully")
                return result
            except asyncio.TimeoutError:
                logger.error(f"⏰ Timeout in {func.__name__} after {timeout_seconds}s")
                raise TimeoutError(f"Operation {func.__name__} timed out after {timeout_seconds}s")
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def with_retry(max_attempts: int = 5, delay: float = 2.0, backoff: float = 2.0):
    """Decorator to add retry logic with detailed logging"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    logger.debug(f"🔄 Attempt {attempt + 1}/{max_attempts} for {func.__name__}")
                    return await func(*args, **kwargs)
                except asyncio.CancelledError:
                    logger.warning(f"🛑 Operation cancelled: {func.__name__}")
                    raise  # Don't retry on cancellation
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"⚠️ Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                        logger.info(f"⏸️ Waiting {wait_time:.1f}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ All {max_attempts} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

# --- Data Models with Enhanced Validation ---

class HighlightSession:
    """Represents a highlight session with comprehensive validation and error handling"""
    
    def __init__(self, query_session_id: str, document_id: str, user: str, query: str):
        # Validate inputs with detailed error messages
        if not query_session_id or not isinstance(query_session_id, str):
            raise ValueError(f"Invalid query_session_id: {query_session_id}")
        if not document_id or not isinstance(document_id, str):
            raise ValueError(f"Invalid document_id: {document_id}")
        if not user or not isinstance(user, str):
            raise ValueError(f"Invalid user: {user}")
        
        self.query_session_id = query_session_id
        self.document_id = document_id
        self.user = user[:100]  # Limit user length
        self.query = (query or "")[:500]  # Limit query length
        
        # Use timezone-aware UTC datetime
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.created_at
        self.expires_at: datetime = self.created_at + timedelta(hours=48)  # Extended from 24
        
        # Data storage with limits
        self.pages_with_highlights: Dict[int, int] = {}
        self.element_types: Set[str] = set()
        self.total_highlights: int = 0
        self.is_active: bool = True
        self.access_count: int = 1
        
        logger.debug(f"Created HighlightSession: {query_session_id}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary with comprehensive error handling"""
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
            logger.error(traceback.format_exc())
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
        logger.debug(f"Updated access for session {self.query_session_id}: count={self.access_count}")
    
    def is_expired(self) -> bool:
        """Checks if session expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def extend_expiry(self, hours: int = 48):
        """Extends expiry time"""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        logger.debug(f"Extended expiry for session {self.query_session_id} by {hours} hours")

class DocumentSession:
    """Document session with enhanced reliability and error recovery"""
    
    def __init__(self, document_id: str, filename: str, max_items_in_memory: int = 50):
        # Validate inputs
        if not document_id or not isinstance(document_id, str):
            raise ValueError(f"Invalid document_id: {document_id}")
        
        self.document_id = document_id
        self.filename = (filename or "unknown.pdf")[:255]  # Limit filename length
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.created_at
        
        # Memory bounds - MORE CONSERVATIVE
        self.max_items_in_memory = min(max_items_in_memory, 100)  # Hard limit
        
        # Use deque for automatic size limits (FIFO)
        self.recent_annotations = deque(maxlen=self.max_items_in_memory)
        self.recent_chats = deque(maxlen=self.max_items_in_memory)
        
        # Indexes for fast lookup
        self.annotation_ids: Set[str] = set()
        self.chat_timestamps: deque = deque(maxlen=self.max_items_in_memory * 2)
        
        # Highlight sessions with limit
        self.highlight_sessions: OrderedDict[str, HighlightSession] = OrderedDict()
        self.max_highlight_sessions = 50  # Reduced from 100
        
        # Activity tracking
        self.active_users: OrderedDict[str, str] = OrderedDict()
        self.max_active_users = 25  # Reduced from 50
        
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
        
        logger.info(f"📄 Created DocumentSession for {document_id} ({filename})")
        
    def add_annotation(self, annotation: Dict[str, Any]) -> bool:
        """Add annotation with enhanced validation"""
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
            
            logger.debug(f"Added annotation {annotation['annotation_id']} to session {self.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add annotation: {e}")
            self.errors.append({
                'type': 'annotation_add', 
                'error': str(e), 
                'time': datetime.now(timezone.utc).isoformat()
            })
            return False
    
    def add_chat(self, chat_data: Dict[str, Any]) -> bool:
        """Add chat with enhanced validation"""
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
            
            logger.debug(f"Added chat to session {self.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chat: {e}")
            self.errors.append({
                'type': 'chat_add', 
                'error': str(e), 
                'time': datetime.now(timezone.utc).isoformat()
            })
            return False
    
    def track_active_user(self, user: str):
        """Track active user with bounds"""
        if user and isinstance(user, str):
            self.active_users[user] = datetime.now(timezone.utc).isoformat()
            self.active_users.move_to_end(user)
            
            # Limit active users
            while len(self.active_users) > self.max_active_users:
                removed_user = self.active_users.popitem(last=False)[0]
                logger.debug(f"Removed inactive user from tracking: {removed_user}")
    
    def add_highlight_session(self, highlight_session: HighlightSession) -> bool:
        """Add highlight session with enhanced bounds checking"""
        try:
            session_id = highlight_session.query_session_id
            
            # Check limits
            if len(self.highlight_sessions) >= self.max_highlight_sessions:
                logger.info(f"Highlight session limit reached ({self.max_highlight_sessions}), removing oldest")
                
                # Remove oldest expired session
                expired_removed = False
                for sid, session in list(self.highlight_sessions.items()):
                    if session.is_expired():
                        del self.highlight_sessions[sid]
                        logger.debug(f"Removed expired highlight session: {sid}")
                        expired_removed = True
                        break
                
                # If no expired sessions, remove oldest
                if not expired_removed and self.highlight_sessions:
                    removed_id, removed_session = self.highlight_sessions.popitem(last=False)
                    logger.debug(f"Removed oldest highlight session: {removed_id}")
            
            self.highlight_sessions[session_id] = highlight_session
            self.needs_persistence = True
            
            logger.debug(f"Added highlight session {session_id} to document {self.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add highlight session: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_memory_usage_estimate(self) -> int:
        """Get memory usage estimate with comprehensive calculation"""
        try:
            size = 0
            
            # Base object
            size += sys.getsizeof(self)
            
            # Collections
            size += sys.getsizeof(self.recent_annotations) + sum(
                sys.getsizeof(json.dumps(a)) for a in self.recent_annotations
            )
            size += sys.getsizeof(self.recent_chats) + sum(
                sys.getsizeof(json.dumps(c)) for c in self.recent_chats
            )
            size += sys.getsizeof(self.annotation_ids) + sum(
                sys.getsizeof(aid) for aid in self.annotation_ids
            )
            size += sys.getsizeof(self.chat_timestamps) + sum(
                sys.getsizeof(ts) for ts in self.chat_timestamps
            )
            
            # Highlight sessions
            size += sys.getsizeof(self.highlight_sessions)
            for session in self.highlight_sessions.values():
                size += sys.getsizeof(json.dumps(session.to_dict()))
            
            # Other data
            size += sys.getsizeof(self.active_users) + sum(
                sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.active_users.items()
            )
            size += sys.getsizeof(self.metadata) + sys.getsizeof(json.dumps(self.metadata))
            
            return size
            
        except Exception as e:
            logger.error(f"Error calculating memory usage: {e}")
            # Return conservative estimate
            return 10000 + (len(self.recent_annotations) + len(self.recent_chats)) * 5000
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary for persistence with error handling"""
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
                "error_count": len(self.errors),
                "memory_usage_bytes": self.get_memory_usage_estimate()
            }
        except Exception as e:
            logger.error(f"Error creating summary dict: {e}")
            logger.error(traceback.format_exc())
            return {
                "document_id": self.document_id,
                "error": str(e)
            }

# --- Main Service Class ---

class SessionService:
    """
    Ultra-reliable session management with comprehensive error recovery
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
        
        # Configuration with ULTRA-SAFE defaults
        self.max_sessions = min(settings.MAX_SESSIONS_IN_MEMORY, 50)  # Reduced hard limit
        self.max_session_memory_mb = min(settings.MAX_SESSION_MEMORY_MB, 1024)  # 1GB hard limit
        self.session_cleanup_interval = max(settings.SESSION_CLEANUP_INTERVAL_SECONDS, 900)  # Min 15 minutes
        self.session_expiry_hours = min(settings.SESSION_CLEANUP_HOURS, 168)  # Max 1 week
        
        # Memory management - CONSERVATIVE
        self.max_items_per_session = min(50, settings.MAX_ANNOTATIONS_PER_SESSION)
        self.persistence_interval = 600  # 10 minutes
        self.persistence_batch_size = 3  # Very small batches
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
            "consecutive_failures": 0,
            "last_error": None
        }
        
        # Error recovery
        self.max_consecutive_failures = 3  # Reduced for faster recovery
        self.recovery_delay = 60  # seconds
        
        logger.info("🚀 SessionService initialized (ULTRA-RELIABLE VERSION)")
        logger.info(f"📊 Configuration:")
        logger.info(f"   Max sessions: {self.max_sessions}")
        logger.info(f"   Max memory: {self.max_session_memory_mb}MB")
        logger.info(f"   Max items/session: {self.max_items_per_session}")
        logger.info(f"   Persistence interval: {self.persistence_interval}s")
        logger.info(f"   Cleanup interval: {self.session_cleanup_interval}s")
        logger.info(f"   Session expiry: {self.session_expiry_hours}h")

    # --- Lifecycle Management ---

    async def start_background_cleanup(self):
        """Start all background tasks with comprehensive error recovery"""
        try:
            if not self.is_running():
                logger.info("🚀 Starting background tasks...")
                
                # Create tasks with meaningful names
                self._cleanup_task = asyncio.create_task(
                    self._cleanup_loop(), 
                    name="session_cleanup"
                )
                self._persistence_task = asyncio.create_task(
                    self._persistence_loop(), 
                    name="session_persistence"
                )
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop(), 
                    name="session_health_check"
                )
                
                logger.info("✅ Background tasks started successfully")
                
                # Initial health check
                await asyncio.sleep(1)
                if self.is_running():
                    logger.info("✅ All background tasks confirmed running")
                
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            logger.error(traceback.format_exc())
            self.health_status["healthy"] = False
            self.health_status["last_error"] = str(e)

    async def stop_background_cleanup(self):
        """Stop all background tasks gracefully with timeout"""
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
                    timeout=30.0  # Increased timeout
                )
                logger.info("✅ All tasks stopped cleanly")
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

    # --- Background Loops with Enhanced Error Recovery ---

    async def _cleanup_loop(self):
        """Main cleanup loop with comprehensive error recovery"""
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
                self.health_status["consecutive_failures"] = 0
                consecutive_failures = 0
                
            except asyncio.CancelledError:
                logger.info("🛑 Cleanup loop cancelled")
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"❌ Cleanup loop error (attempt {consecutive_failures}): {e}")
                logger.error(traceback.format_exc())
                
                self.health_status["consecutive_failures"] = consecutive_failures
                self.health_status["last_error"] = str(e)
                
                if consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"⚠️ Too many cleanup failures, pausing for {self.recovery_delay}s")
                    await asyncio.sleep(self.recovery_delay)
                    consecutive_failures = 0
                else:
                    await asyncio.sleep(10)  # Short delay before retry

    async def _persistence_loop(self):
        """Persistence loop with enhanced reliability"""
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
                else:
                    await asyncio.sleep(10)

    async def _health_check_loop(self):
        """Health monitoring loop with detailed checks"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check memory usage
                memory_usage = self._get_total_memory_usage()
                memory_percent = (memory_usage / (self.max_session_memory_mb * 1024 * 1024)) * 100
                
                logger.debug(f"💾 Memory usage: {memory_percent:.1f}% ({memory_usage / (1024*1024):.1f}MB)")
                
                if memory_percent > 80:
                    logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}%")
                    # Force cleanup
                    await self._perform_cleanup()
                
                # Check consecutive failures
                if self.health_status["consecutive_failures"] > self.max_consecutive_failures:
                    logger.error("❌ Service unhealthy - too many consecutive failures")
                    self.health_status["healthy"] = False
                else:
                    self.health_status["healthy"] = True
                
                # Log session stats
                if len(self.document_sessions) > 0:
                    logger.debug(f"📊 Active sessions: {len(self.document_sessions)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)

    # --- Core Session Management ---

    @with_retry(max_attempts=5, delay=1.0)
    @with_timeout(60.0)
    async def get_or_create_session(self, document_id: str, filename: str) -> DocumentSession:
        """Get existing or create new session with comprehensive error recovery"""
        # Validate inputs
        if not document_id or not isinstance(document_id, str):
            raise ValueError(f"Invalid document_id: {document_id}")
        
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
                
                # Try to load recent data with timeout
                try:
                    await asyncio.wait_for(
                        self._load_recent_session_data(session),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout loading session history - continuing with new session")
                except Exception as e:
                    logger.warning(f"Could not load recent data: {e}")
                    # Continue anyway - new session
                
                self.document_sessions[document_id] = session
                self.stats['sessions_created'] += 1
                
                # Check limits
                await self._enforce_session_limits()
                
                logger.info(f"✅ Session created for {document_id}")
                return session
                
            except Exception as e:
                self.stats['session_creation_errors'] += 1
                logger.error(f"Failed to create session: {e}")
                logger.error(traceback.format_exc())
                raise

    async def get_session(self, document_id: str) -> Optional[DocumentSession]:
        """Get existing session with validation"""
        if not document_id:
            return None
        
        async with self.lock:
            session = self.document_sessions.get(document_id)
            if session:
                session.last_accessed = datetime.now(timezone.utc)
                session.access_count += 1
                self.document_sessions.move_to_end(document_id)
                logger.debug(f"Retrieved session for {document_id}")
            return session

    async def create_session(self, document_id: str, original_filename: str):
        """Create a new session (compatibility method)"""
        return await self.get_or_create_session(document_id, original_filename)

    @with_retry(max_attempts=3, delay=0.5)
    @with_timeout(30.0)
    async def record_chat_activity(self, document_id: str, user: str, chat_data: Optional[Dict] = None):
        """Record chat activity with enhanced error recovery"""
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
                    if len(session.recent_chats) % 25 == 0:  # Reduced from 50
                        logger.debug("Auto-persisting due to chat activity")
                        asyncio.create_task(self._persist_session_data(document_id, session))
                
        except Exception as e:
            self.stats['chat_record_errors'] += 1
            logger.error(f"Failed to record chat activity: {e}")
            logger.error(traceback.format_exc())

    async def add_annotation(self, document_id: str, annotation_data: Dict[str, Any]) -> bool:
        """Add annotation with comprehensive validation"""
        try:
            session = await self.get_session(document_id)
            if not session:
                logger.warning(f"No session for document: {document_id}")
                return False
            
            async with self.lock:
                if session.add_annotation(annotation_data):
                    self.stats['annotations_added'] += 1
                    
                    # Auto-persist if many changes
                    if len(session.recent_annotations) % 10 == 0:  # Reduced from 25
                        logger.debug("Auto-persisting due to annotation activity")
                        asyncio.create_task(self._persist_session_data(document_id, session))
                    
                    return True
                
                return False
                
        except Exception as e:
            self.stats['annotation_add_errors'] += 1
            logger.error(f"Failed to add annotation: {e}")
            logger.error(traceback.format_exc())
            return False

    @with_retry(max_attempts=3, delay=0.5)
    @with_timeout(30.0)
    async def update_highlight_session(
        self, document_id: str, query_session_id: str, 
        pages_with_highlights: Dict[int, int], element_types: List[str], 
        total_highlights: int, user: str = "system", query: str = ""
    ):
        """Update or create highlight session with validation"""
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
                    highlight_session.extend_expiry()  # Extend on update
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
                    logger.debug(f"Updated highlight session {query_session_id}")
                
        except Exception as e:
            self.stats['highlight_update_errors'] += 1
            logger.error(f"Failed to update highlight session: {e}")
            logger.error(traceback.format_exc())

    async def update_session_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update session metadata with size limits"""
        try:
            session = await self.get_session(document_id)
            if not session:
                return
            
            async with self.lock:
                # Merge metadata with size limit
                metadata_str = json.dumps(metadata)
                if len(metadata_str) > 50000:  # 50KB limit
                    logger.warning("Metadata too large, truncating")
                    # Keep only essential fields
                    essential_keys = ['status', 'processing_complete', 'pages_processed', 'has_thumbnails']
                    metadata = {k: v for k, v in metadata.items() if k in essential_keys}
                
                session.metadata.update(metadata)
                session.needs_persistence = True
                logger.debug(f"Updated metadata for session {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            logger.error(traceback.format_exc())

    # --- Persistence Methods ---

    @with_retry(max_attempts=5, delay=2.0)
    @with_timeout(120.0)
    async def _persist_session_data(self, session_id: str, session: DocumentSession) -> bool:
        """Persist session data with compression and comprehensive error recovery"""
        async with self.persistence_lock:
            try:
                start_time = time.time()
                
                # Skip if recently persisted
                time_since_last = (datetime.now(timezone.utc) - session.last_persisted_at).total_seconds()
                if time_since_last < 60 and not session.needs_persistence:  # Increased from 30
                    logger.debug(f"Skipping persistence for {session_id} - too recent")
                    return True
                
                logger.info(f"💾 Persisting session {session_id}...")
                
                # Prepare data with limits
                session_data = {
                    "session_id": session_id,
                    "persisted_at": datetime.now(timezone.utc).isoformat(),
                    "summary": session.to_summary_dict(),
                    "recent_annotations": list(session.recent_annotations)[-25:],  # Reduced from 50
                    "recent_chats": list(session.recent_chats)[-25:],  # Reduced from 50
                    "highlight_sessions": {
                        sid: hs.to_dict() 
                        for sid, hs in list(session.highlight_sessions.items())[-10:]  # Reduced from 20
                    },
                    "active_users": dict(list(session.active_users.items())[-10:])  # Reduced from 20
                }
                
                # Convert to JSON
                json_data = json.dumps(session_data, ensure_ascii=False, separators=(',', ':'))
                
                # Generate safe filename
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                
                # Compress if enabled
                if self.compress_persistence:
                    logger.debug("   Compressing data...")
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
                
                logger.debug(f"   Uploading {len(data_bytes)} bytes...")
                
                # Upload to storage
                await self.storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name,
                    data=data_bytes,
                    content_type=content_type,
                    metadata={
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "compressed": str(self.compress_persistence),
                        "items_count": str(len(session.recent_annotations) + len(session.recent_chats))
                    }
                )
                
                # Update session state
                session.last_persisted_at = datetime.now(timezone.utc)
                session.needs_persistence = False
                session.persistence_failures = 0
                
                elapsed = time.time() - start_time
                self.stats['sessions_persisted'] += 1
                
                logger.info(f"✅ Persisted session {session_id} ({len(data_bytes)} bytes) in {elapsed:.1f}s")
                
                return True
                
            except Exception as e:
                session.persistence_failures += 1
                self.stats['persistence_errors'] += 1
                self.health_status["consecutive_failures"] += 1
                
                logger.error(f"Failed to persist session {session_id}: {e}")
                logger.error(traceback.format_exc())
                
                return False

    @with_timeout(180.0)
    async def _load_recent_session_data(self, session: DocumentSession):
        """Load recent session data with timeout protection"""
        try:
            logger.debug(f"Loading session history for {session.document_id}...")
            
            # Find most recent session file
            session_files = await self.storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=f"{session.document_id}_session_",
                max_results=5  # Reduced from 10
            )
            
            if not session_files:
                logger.debug(f"No previous session data for {session.document_id}")
                return
            
            # Sort by timestamp (newest first)
            session_files.sort(reverse=True)
            
            # Try to load most recent valid file
            for session_file in session_files[:3]:  # Try up to 3 most recent
                try:
                    logger.debug(f"   Trying to load {session_file}...")
                    
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
        """Restore session data with comprehensive validation"""
        try:
            restored_count = 0
            
            # Restore recent items (already limited by deque maxlen)
            for ann in session_data.get('recent_annotations', []):
                if isinstance(ann, dict) and session.add_annotation(ann):
                    restored_count += 1
            
            for chat in session_data.get('recent_chats', []):
                if isinstance(chat, dict) and session.add_chat(chat):
                    restored_count += 1
            
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
            
            logger.debug(f"Restored {restored_count} items to session")
            
        except Exception as e:
            logger.error(f"Error restoring session data: {e}")
            logger.error(traceback.format_exc())

    async def _persist_all_sessions(self) -> Dict[str, int]:
        """Persist all sessions that need it with small batches"""
        results = {"persisted_count": 0, "failed_count": 0}
        
        # Get sessions needing persistence
        sessions_to_persist = []
        
        async with self.lock:
            for session_id, session in self.document_sessions.items():
                if session.needs_persistence or session.persistence_failures > 0:
                    sessions_to_persist.append((session_id, session))
        
        if not sessions_to_persist:
            return results
        
        logger.info(f"Found {len(sessions_to_persist)} sessions needing persistence")
        
        # Persist in small batches
        for i in range(0, len(sessions_to_persist), self.persistence_batch_size):
            batch = sessions_to_persist[i:i + self.persistence_batch_size]
            
            logger.debug(f"Persisting batch {i//self.persistence_batch_size + 1}")
            
            # Persist each session in batch
            for session_id, session in batch:
                try:
                    success = await self._persist_session_data(session_id, session)
                    if success:
                        results["persisted_count"] += 1
                    else:
                        results["failed_count"] += 1
                except Exception as e:
                    logger.error(f"Persistence error for {session_id}: {e}")
                    results["failed_count"] += 1
            
            # Small delay between batches
            if i + self.persistence_batch_size < len(sessions_to_persist):
                await asyncio.sleep(1)
        
        return results

    # --- Cleanup and Memory Management ---

    async def _perform_cleanup(self) -> Dict[str, int]:
        """Perform comprehensive cleanup with detailed tracking"""
        results = {
            'expired_highlights': 0,
            'evicted_sessions': 0,
            'cleaned_items': 0
        }
        
        try:
            async with self.lock:
                logger.debug("🧹 Starting cleanup operations...")
                
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
                logger.debug("   Persisting sessions before cleanup...")
                await self._persist_all_sessions()
                
                # Evict old sessions
                results['evicted_sessions'] = await self._evict_old_sessions()
                
                # Memory pressure check
                await self._check_memory_pressure()
            
            self.stats['cleanups_performed'] += 1
            logger.debug(f"✅ Cleanup complete: {results}")
            
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
        
        logger.debug(f"Found {len(sessions_to_evict)} sessions to evict")
        
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
        
        if total_memory > memory_limit * 0.8:  # 80% threshold
            logger.warning(f"⚠️ High memory usage: {total_memory / (1024 * 1024):.1f}MB")
            
            # Evict sessions until under 70%
            target_memory = memory_limit * 0.7
            
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
        """Calculate total memory usage with error handling"""
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
        """Get session information with error handling"""
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
                "errors": len(session.errors),
                "health": "healthy" if session.persistence_failures < 3 else "degraded"
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
                "version": "Ultra-Reliable",
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
                    "compression_enabled": self.compress_persistence,
                    "persistence_batch_size": self.persistence_batch_size
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
                    "consecutive_failures": self.health_status["consecutive_failures"],
                    "last_error": self.health_status["last_error"]
                },
                "lifetime_stats": dict(self.stats)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    # --- Shutdown ---

    async def shutdown(self):
        """Graceful shutdown with final persistence"""
        logger.info("🛑 SessionService shutting down...")
        
        try:
            # Stop background tasks first
            await self.stop_background_cleanup()
            
            # Final persistence of all sessions
            logger.info("💾 Performing final persistence...")
            
            # Force persistence for all sessions
            async with self.lock:
                for session in self.document_sessions.values():
                    session.needs_persistence = True
            
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
        if exc_type:
            logger.error(f"Exception in session context: {exc_type.__name__}: {exc_val}")
        await self.shutdown()

# === END OF FILE ===