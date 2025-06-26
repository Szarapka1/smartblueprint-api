# app/services/session_service.py - ENHANCED WITH HIGHLIGHT SESSION MANAGEMENT

import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import json
import os

from app.core.config import AppSettings, get_settings
from app.models.schemas import Annotation, VisualElement, GridReference

logger = logging.getLogger(__name__)


class HighlightSession:
    """Represents a highlight session from a user query"""
    
    def __init__(self, query_session_id: str, document_id: str, user: str, query: str):
        self.query_session_id = query_session_id
        self.document_id = document_id
        self.user = user
        self.query = query
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=24)
        self.pages_with_highlights: Dict[int, int] = {}  # page_num: count
        self.element_types: Set[str] = set()
        self.total_highlights = 0
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
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
        """Update last access time"""
        self.last_accessed = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() > self.expires_at


class DocumentSession:
    """Represents a document being viewed with all associated data"""
    
    def __init__(self, document_id: str, filename: str):
        self.document_id = document_id
        self.filename = filename
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.annotations: List[Dict] = []
        self.highlight_sessions: Dict[str, HighlightSession] = {}  # query_session_id: session
        self.active_users: Dict[str, str] = {}  # user: active_query_session_id
        self.chat_count = 0
        self.annotation_count = 0
        self.metadata: Dict[str, Any] = {}
    
    def add_highlight_session(self, session: HighlightSession):
        """Add a new highlight session"""
        self.highlight_sessions[session.query_session_id] = session
        self.active_users[session.user] = session.query_session_id
        self.last_accessed = datetime.utcnow()
    
    def get_active_session_for_user(self, user: str) -> Optional[HighlightSession]:
        """Get user's active highlight session"""
        session_id = self.active_users.get(user)
        if session_id and session_id in self.highlight_sessions:
            session = self.highlight_sessions[session_id]
            if not session.is_expired():
                session.update_access()
                return session
            else:
                # Clean up expired session
                del self.highlight_sessions[session_id]
                del self.active_users[user]
        return None
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired highlight sessions"""
        expired_count = 0
        expired_ids = []
        
        for session_id, session in self.highlight_sessions.items():
            if session.is_expired():
                expired_ids.append(session_id)
                expired_count += 1
        
        # Remove expired sessions
        for session_id in expired_ids:
            del self.highlight_sessions[session_id]
            # Remove from active users
            users_to_update = [
                user for user, sid in self.active_users.items() 
                if sid == session_id
            ]
            for user in users_to_update:
                del self.active_users[user]
        
        return expired_count
    
    def get_memory_usage_estimate(self) -> int:
        """Estimate memory usage in bytes"""
        # Rough estimation
        base_size = 1024  # 1KB base
        annotation_size = len(self.annotations) * 200  # ~200 bytes per annotation
        highlight_size = len(self.highlight_sessions) * 500  # ~500 bytes per session
        metadata_size = len(json.dumps(self.metadata)) if self.metadata else 0
        
        return base_size + annotation_size + highlight_size + metadata_size


class SessionService:
    """Enhanced session management with highlight session tracking"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        
        # Document sessions with LRU eviction
        self.document_sessions: OrderedDict[str, DocumentSession] = OrderedDict()
        
        # Configuration
        self.max_sessions = settings.MAX_SESSIONS_IN_MEMORY
        self.max_annotations_per_session = settings.MAX_ANNOTATIONS_PER_SESSION
        self.session_cleanup_hours = settings.SESSION_CLEANUP_HOURS
        self.max_session_memory_mb = settings.MAX_SESSION_MEMORY_MB
        self.max_highlight_sessions_per_doc = 50  # Limit highlight sessions per document
        
        # Statistics tracking
        self.stats = {
            'sessions_created': 0,
            'sessions_evicted': 0,
            'highlights_created': 0,
            'highlights_expired': 0,
            'total_annotations': 0,
            'total_chats': 0
        }
        
        # Background cleanup task
        self.cleanup_task = None
        
        logger.info("✅ Enhanced SessionService initialized")
        logger.info(f"   📊 Max sessions: {self.max_sessions}")
        logger.info(f"   📝 Max annotations per session: {self.max_annotations_per_session}")
        logger.info(f"   🎯 Max highlight sessions per doc: {self.max_highlight_sessions_per_doc}")
        logger.info(f"   💾 Max memory per session: {self.max_session_memory_mb}MB")
        logger.info(f"   ⏰ Session cleanup: {self.session_cleanup_hours}h")

    async def start_background_cleanup(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self.cleanup_all_expired()
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("🔄 Started background cleanup task")

    def create_session(self, document_id: str, original_filename: str) -> str:
        """Create a new document session"""
        try:
            # Check if session already exists
            if document_id in self.document_sessions:
                session = self.document_sessions[document_id]
                session.last_accessed = datetime.utcnow()
                # Move to end (LRU)
                self.document_sessions.move_to_end(document_id)
                logger.info(f"♻️ Reusing existing session for document {document_id}")
                return document_id
            
            # Check if we need to evict old sessions
            if len(self.document_sessions) >= self.max_sessions:
                self._evict_oldest_session()
            
            # Create new session
            session = DocumentSession(document_id, original_filename)
            self.document_sessions[document_id] = session
            self.stats['sessions_created'] += 1
            
            logger.info(f"✅ Created new document session:")
            logger.info(f"   🆔 Document ID: {document_id}")
            logger.info(f"   📄 Filename: '{original_filename}'")
            logger.info(f"   📊 Total sessions: {len(self.document_sessions)}")
            
            return document_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise

    def create_highlight_session(
        self, 
        document_id: str, 
        user: str, 
        query: str,
        query_session_id: Optional[str] = None
    ) -> HighlightSession:
        """Create a new highlight session for a user query"""
        try:
            if document_id not in self.document_sessions:
                raise ValueError(f"Document session {document_id} not found")
            
            doc_session = self.document_sessions[document_id]
            
            # Generate query session ID if not provided
            if not query_session_id:
                query_session_id = str(uuid.uuid4())
            
            # Check highlight session limit
            if len(doc_session.highlight_sessions) >= self.max_highlight_sessions_per_doc:
                # Remove oldest expired session
                doc_session.cleanup_expired_sessions()
                
                # If still over limit, remove oldest
                if len(doc_session.highlight_sessions) >= self.max_highlight_sessions_per_doc:
                    oldest_id = min(
                        doc_session.highlight_sessions.keys(),
                        key=lambda k: doc_session.highlight_sessions[k].created_at
                    )
                    del doc_session.highlight_sessions[oldest_id]
                    logger.info(f"♻️ Evicted oldest highlight session {oldest_id}")
            
            # Create new highlight session
            highlight_session = HighlightSession(
                query_session_id=query_session_id,
                document_id=document_id,
                user=user,
                query=query
            )
            
            doc_session.add_highlight_session(highlight_session)
            self.stats['highlights_created'] += 1
            
            # Update document access
            doc_session.last_accessed = datetime.utcnow()
            self.document_sessions.move_to_end(document_id)
            
            logger.info(f"✅ Created highlight session:")
            logger.info(f"   🆔 Session ID: {query_session_id}")
            logger.info(f"   👤 User: {user}")
            logger.info(f"   ❓ Query: {query[:50]}...")
            
            return highlight_session
            
        except Exception as e:
            logger.error(f"❌ Failed to create highlight session: {e}")
            raise

    def update_highlight_session(
        self,
        document_id: str,
        query_session_id: str,
        pages_with_highlights: Dict[int, int],
        element_types: List[str],
        total_highlights: int
    ) -> bool:
        """Update highlight session with results"""
        try:
            if document_id not in self.document_sessions:
                return False
            
            doc_session = self.document_sessions[document_id]
            
            if query_session_id not in doc_session.highlight_sessions:
                return False
            
            highlight_session = doc_session.highlight_sessions[query_session_id]
            highlight_session.pages_with_highlights = pages_with_highlights
            highlight_session.element_types = set(element_types)
            highlight_session.total_highlights = total_highlights
            highlight_session.update_access()
            
            logger.info(f"📝 Updated highlight session {query_session_id}:")
            logger.info(f"   📄 Pages with highlights: {len(pages_with_highlights)}")
            logger.info(f"   🎯 Total highlights: {total_highlights}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update highlight session: {e}")
            return False

    def get_user_active_highlights(
        self,
        document_id: str,
        user: str
    ) -> Optional[Dict[str, Any]]:
        """Get user's active highlight session"""
        try:
            if document_id not in self.document_sessions:
                return None
            
            doc_session = self.document_sessions[document_id]
            highlight_session = doc_session.get_active_session_for_user(user)
            
            if highlight_session:
                return {
                    'query_session_id': highlight_session.query_session_id,
                    'query': highlight_session.query,
                    'created_at': highlight_session.created_at.isoformat(),
                    'pages_with_highlights': highlight_session.pages_with_highlights,
                    'element_types': list(highlight_session.element_types),
                    'total_highlights': highlight_session.total_highlights,
                    'expires_at': highlight_session.expires_at.isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user highlights: {e}")
            return None

    def set_user_active_session(
        self,
        document_id: str,
        user: str,
        query_session_id: str
    ) -> bool:
        """Set user's active highlight session"""
        try:
            if document_id not in self.document_sessions:
                return False
            
            doc_session = self.document_sessions[document_id]
            
            if query_session_id in doc_session.highlight_sessions:
                doc_session.active_users[user] = query_session_id
                doc_session.highlight_sessions[query_session_id].update_access()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set active session: {e}")
            return False

    def add_annotation(self, document_id: str, annotation: Annotation) -> bool:
        """Add annotation to document session"""
        try:
            if document_id not in self.document_sessions:
                logger.warning(f"Document session {document_id} not found")
                return False
            
            doc_session = self.document_sessions[document_id]
            
            # Check annotation limit
            if len(doc_session.annotations) >= self.max_annotations_per_session:
                logger.warning(f"⚠️ Annotation limit reached for {document_id}")
                # Remove oldest annotation
                doc_session.annotations.pop(0)
            
            # Add annotation
            annotation_data = annotation.model_dump() if hasattr(annotation, 'model_dump') else annotation.dict()
            annotation_data['timestamp'] = datetime.utcnow().isoformat()
            doc_session.annotations.append(annotation_data)
            doc_session.annotation_count += 1
            self.stats['total_annotations'] += 1
            
            # Update access
            doc_session.last_accessed = datetime.utcnow()
            self.document_sessions.move_to_end(document_id)
            
            logger.debug(f"Added annotation to document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add annotation: {e}")
            return False

    def get_annotations_for_page(
        self,
        document_id: str,
        page_number: int,
        include_highlights: bool = True
    ) -> List[Dict]:
        """Get annotations for a specific page"""
        try:
            if document_id not in self.document_sessions:
                return []
            
            doc_session = self.document_sessions[document_id]
            doc_session.last_accessed = datetime.utcnow()
            self.document_sessions.move_to_end(document_id)
            
            # Get regular annotations
            page_annotations = [
                ann for ann in doc_session.annotations
                if ann.get('page_number') == page_number
            ]
            
            # Note: Highlights are now stored separately in storage service
            # This is just for compatibility
            
            return page_annotations
            
        except Exception as e:
            logger.error(f"Failed to get annotations: {e}")
            return []

    def update_session_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update session metadata"""
        try:
            if document_id not in self.document_sessions:
                return False
            
            doc_session = self.document_sessions[document_id]
            doc_session.metadata.update(metadata)
            doc_session.last_accessed = datetime.utcnow()
            self.document_sessions.move_to_end(document_id)
            
            # Track page count if provided
            if 'page_count' in metadata:
                logger.info(f"📋 Updated metadata for {document_id}: {metadata.get('page_count')} pages")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False

    def record_chat_activity(self, document_id: str, user: str) -> bool:
        """Record chat activity for a document"""
        try:
            if document_id not in self.document_sessions:
                return False
            
            doc_session = self.document_sessions[document_id]
            doc_session.chat_count += 1
            self.stats['total_chats'] += 1
            
            # Update access
            doc_session.last_accessed = datetime.utcnow()
            self.document_sessions.move_to_end(document_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record chat: {e}")
            return False

    def get_session_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session information"""
        try:
            if document_id not in self.document_sessions:
                return None
            
            doc_session = self.document_sessions[document_id]
            doc_session.last_accessed = datetime.utcnow()
            self.document_sessions.move_to_end(document_id)
            
            # Count active highlights
            active_highlights = sum(
                1 for session in doc_session.highlight_sessions.values()
                if not session.is_expired()
            )
            
            # Get unique users
            unique_users = set()
            for session in doc_session.highlight_sessions.values():
                unique_users.add(session.user)
            
            return {
                'document_id': document_id,
                'filename': doc_session.filename,
                'created_at': doc_session.created_at.isoformat(),
                'last_accessed': doc_session.last_accessed.isoformat(),
                'metadata': doc_session.metadata,
                'statistics': {
                    'annotation_count': len(doc_session.annotations),
                    'chat_count': doc_session.chat_count,
                    'highlight_sessions': len(doc_session.highlight_sessions),
                    'active_highlight_sessions': active_highlights,
                    'unique_users': len(unique_users),
                    'memory_usage_bytes': doc_session.get_memory_usage_estimate()
                },
                'active_users': list(doc_session.active_users.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return None

    def get_all_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get paginated session list"""
        try:
            all_sessions = {}
            
            # Get sessions with pagination
            session_items = list(self.document_sessions.items())
            paginated_items = session_items[offset:offset + limit]
            
            for doc_id, doc_session in paginated_items:
                # Count active highlights
                if include_expired:
                    highlight_count = len(doc_session.highlight_sessions)
                else:
                    highlight_count = sum(
                        1 for s in doc_session.highlight_sessions.values()
                        if not s.is_expired()
                    )
                
                all_sessions[doc_id] = {
                    'filename': doc_session.filename,
                    'created_at': doc_session.created_at.isoformat(),
                    'last_accessed': doc_session.last_accessed.isoformat(),
                    'annotation_count': len(doc_session.annotations),
                    'chat_count': doc_session.chat_count,
                    'highlight_sessions': highlight_count,
                    'active_users': len(doc_session.active_users),
                    'page_count': doc_session.metadata.get('page_count', 0)
                }
            
            logger.info(f"📊 Retrieved {len(all_sessions)} sessions")
            return all_sessions
            
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return {}

    def get_highlight_sessions_for_document(
        self,
        document_id: str,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all highlight sessions for a document"""
        try:
            if document_id not in self.document_sessions:
                return []
            
            doc_session = self.document_sessions[document_id]
            sessions = []
            
            for session in doc_session.highlight_sessions.values():
                if include_expired or not session.is_expired():
                    sessions.append(session.to_dict())
            
            # Sort by creation time (newest first)
            sessions.sort(key=lambda x: x['created_at'], reverse=True)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get highlight sessions: {e}")
            return []

    def delete_session(self, document_id: str) -> bool:
        """Delete a document session and all associated data"""
        try:
            if document_id not in self.document_sessions:
                logger.warning(f"Session {document_id} not found")
                return False
            
            doc_session = self.document_sessions[document_id]
            
            # Log deletion stats
            logger.info(f"🗑️ Deleting session {document_id}:")
            logger.info(f"   📄 Filename: {doc_session.filename}")
            logger.info(f"   📝 Annotations: {len(doc_session.annotations)}")
            logger.info(f"   💬 Chats: {doc_session.chat_count}")
            logger.info(f"   🎯 Highlight sessions: {len(doc_session.highlight_sessions)}")
            
            del self.document_sessions[document_id]
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    async def cleanup_all_expired(self) -> Dict[str, int]:
        """Clean up all expired highlight sessions"""
        try:
            total_expired = 0
            sessions_checked = 0
            
            for doc_id, doc_session in self.document_sessions.items():
                sessions_checked += 1
                expired_count = doc_session.cleanup_expired_sessions()
                total_expired += expired_count
                
                if expired_count > 0:
                    logger.info(f"🧹 Cleaned {expired_count} expired highlights from {doc_id}")
            
            self.stats['highlights_expired'] += total_expired
            
            logger.info(f"🧹 Cleanup complete:")
            logger.info(f"   📊 Sessions checked: {sessions_checked}")
            logger.info(f"   ❌ Highlights expired: {total_expired}")
            
            return {
                'sessions_checked': sessions_checked,
                'highlights_expired': total_expired
            }
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {'error': str(e)}

    def cleanup_old_sessions(self, hours: int = None) -> int:
        """Remove document sessions older than specified hours"""
        if hours is None:
            hours = self.session_cleanup_hours
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            sessions_to_remove = []
            
            for doc_id, doc_session in self.document_sessions.items():
                if doc_session.last_accessed < cutoff_time:
                    sessions_to_remove.append(doc_id)
            
            for doc_id in sessions_to_remove:
                self.delete_session(doc_id)
            
            logger.info(f"🧹 Removed {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all sessions"""
        try:
            total_annotations = 0
            total_chats = 0
            total_highlights = 0
            active_highlights = 0
            total_users = set()
            memory_usage = 0
            
            for doc_session in self.document_sessions.values():
                total_annotations += len(doc_session.annotations)
                total_chats += doc_session.chat_count
                total_highlights += len(doc_session.highlight_sessions)
                
                # Count active highlights
                for h_session in doc_session.highlight_sessions.values():
                    if not h_session.is_expired():
                        active_highlights += 1
                    total_users.add(h_session.user)
                
                memory_usage += doc_session.get_memory_usage_estimate()
            
            capacity_percent = (len(self.document_sessions) / self.max_sessions * 100) if self.max_sessions > 0 else 0
            
            return {
                'total_sessions': len(self.document_sessions),
                'capacity_used_percent': round(capacity_percent, 1),
                'total_annotations': total_annotations,
                'total_chats': total_chats,
                'total_highlight_sessions': total_highlights,
                'active_highlight_sessions': active_highlights,
                'unique_users': len(total_users),
                'memory_usage_mb': round(memory_usage / (1024 * 1024), 2),
                'lifetime_stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}

    def _evict_oldest_session(self):
        """Evict oldest session (LRU)"""
        if not self.document_sessions:
            return
        
        # Get oldest (first item in OrderedDict)
        oldest_id, oldest_session = self.document_sessions.popitem(last=False)
        self.stats['sessions_evicted'] += 1
        
        logger.info(f"♻️ Evicted oldest session:")
        logger.info(f"   🆔 Document: {oldest_id}")
        logger.info(f"   📄 Filename: {oldest_session.filename}")
        logger.info(f"   ⏰ Last accessed: {oldest_session.last_accessed}")

    def export_session_data(self, document_id: str) -> Optional[str]:
        """Export session data as JSON"""
        try:
            if document_id not in self.document_sessions:
                return None
            
            doc_session = self.document_sessions[document_id]
            
            export_data = {
                'document_id': document_id,
                'filename': doc_session.filename,
                'created_at': doc_session.created_at.isoformat(),
                'metadata': doc_session.metadata,
                'annotations': doc_session.annotations,
                'highlight_sessions': [
                    session.to_dict() for session in doc_session.highlight_sessions.values()
                ],
                'statistics': {
                    'chat_count': doc_session.chat_count,
                    'annotation_count': len(doc_session.annotations),
                    'highlight_session_count': len(doc_session.highlight_sessions)
                }
            }
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to export session: {e}")
            return None

    def clear_all_sessions(self) -> int:
        """Clear all sessions for maintenance"""
        try:
            session_count = len(self.document_sessions)
            
            # Calculate totals for logging
            total_annotations = sum(
                len(s.annotations) for s in self.document_sessions.values()
            )
            total_highlights = sum(
                len(s.highlight_sessions) for s in self.document_sessions.values()
            )
            
            self.document_sessions.clear()
            
            logger.info(f"🧹 Cleared all sessions:")
            logger.info(f"   📊 Sessions: {session_count}")
            logger.info(f"   📝 Annotations: {total_annotations}")
            logger.info(f"   🎯 Highlight sessions: {total_highlights}")
            
            return session_count
            
        except Exception as e:
            logger.error(f"Failed to clear sessions: {e}")
            return 0

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_background_cleanup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
