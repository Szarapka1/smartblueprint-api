# app/services/session_service.py - OPTIMIZED FOR LARGE DOCUMENTS

import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import OrderedDict
import json
from app.core.config import AppSettings, get_settings
from app.models.schemas import Annotation

logger = logging.getLogger(__name__)

class SessionService:
    """Session management service optimized for handling multiple large documents"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        
        # Use OrderedDict for LRU-style memory management
        self.sessions: OrderedDict[str, Any] = OrderedDict()
        
        # Configuration for optimization
        self.max_sessions = int(os.getenv("MAX_SESSIONS_IN_MEMORY", "100"))
        self.max_annotations_per_session = int(os.getenv("MAX_ANNOTATIONS_PER_SESSION", "1000"))
        
        logger.info("✅ SessionService initialized (optimized in-memory storage)")
        logger.info(f"   📊 Max sessions: {self.max_sessions}")
        logger.info(f"   📝 Max annotations per session: {self.max_annotations_per_session}")
        logger.info(f"   💾 Initial session count: 0")

    def create_session(self, original_filename: str) -> str:
        """Creates a new session record with memory management"""
        try:
            # Check if we need to evict old sessions (LRU)
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session (first item in OrderedDict)
                oldest_id, oldest_data = self.sessions.popitem(last=False)
                logger.info(f"♻️ Evicted oldest session {oldest_id} to maintain memory limit")
            
            session_id = str(uuid.uuid4())
            
            session_data = {
                "filename": original_filename,
                "annotations": [],
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "status": "created",
                "metadata": {
                    "file_size": 0,
                    "page_count": 0,
                    "annotation_count": 0
                }
            }
            
            self.sessions[session_id] = session_data
            
            logger.info(f"✅ Created new session:")
            logger.info(f"   🆔 Session ID: {session_id}")
            logger.info(f"   📄 Filename: '{original_filename}'")
            logger.info(f"   📊 Total sessions: {len(self.sessions)}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create session for file '{original_filename}': {e}")
            raise

    def add_annotation(self, session_id: str, annotation: Annotation) -> None:
        """Adds an annotation with overflow protection"""
        try:
            if session_id in self.sessions:
                # Update last accessed time
                self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
                
                # Move to end (most recently used)
                self.sessions.move_to_end(session_id)
                
                # Check annotation limit
                current_annotations = self.sessions[session_id]["annotations"]
                if len(current_annotations) >= self.max_annotations_per_session:
                    logger.warning(f"⚠️ Session {session_id} reached annotation limit, removing oldest")
                    current_annotations.pop(0)
                
                annotation_data = annotation.model_dump()
                annotation_data["timestamp"] = datetime.utcnow().isoformat()
                current_annotations.append(annotation_data)
                
                # Update metadata
                self.sessions[session_id]["metadata"]["annotation_count"] = len(current_annotations)
                
                logger.info(f"✅ Added annotation to session:")
                logger.info(f"   🆔 Session: {session_id}")
                logger.info(f"   📄 Page: {annotation.page_number}")
                logger.info(f"   📝 Content: '{annotation.text[:50]}{'...' if len(annotation.text) > 50 else ''}'")
                logger.info(f"   📊 Total annotations: {len(current_annotations)}")
                
            else:
                logger.warning(f"⚠️ Attempted to add annotation to non-existent session: {session_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to add annotation to session {session_id}: {e}")
            raise

    def get_annotations_for_page(self, session_id: str, page_number: int) -> List[Dict]:
        """Retrieves annotations for a specific page with caching"""
        try:
            if session_id in self.sessions:
                # Update last accessed and move to end
                self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
                self.sessions.move_to_end(session_id)
                
                all_annotations = self.sessions[session_id].get("annotations", [])
                
                # Use list comprehension for efficiency
                page_annotations = [
                    ann for ann in all_annotations 
                    if ann.get("page_number") == page_number
                ]
                
                logger.debug(f"📋 Retrieved {len(page_annotations)} annotations for page {page_number}")
                
                return page_annotations
            else:
                logger.warning(f"⚠️ Session {session_id} not found")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get annotations: {e}")
            return []

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information with usage tracking"""
        try:
            if session_id in self.sessions:
                # Update access time and move to end
                self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
                self.sessions.move_to_end(session_id)
                
                session_data = self.sessions[session_id].copy()
                
                # Calculate additional stats
                annotations = session_data.get("annotations", [])
                
                # Page distribution
                page_counts = {}
                annotation_types = {}
                
                for ann in annotations:
                    # Count by page
                    page_num = ann.get("page_number", 0)
                    page_counts[page_num] = page_counts.get(page_num, 0) + 1
                    
                    # Count by type
                    ann_type = ann.get("annotation_type", "unknown")
                    annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
                
                # Add computed stats
                session_data["statistics"] = {
                    "annotation_count": len(annotations),
                    "annotations_per_page": page_counts,
                    "annotation_types": annotation_types,
                    "pages_with_annotations": len(page_counts),
                    "most_annotated_page": max(page_counts.items(), key=lambda x: x[1])[0] if page_counts else None
                }
                
                return session_data
            else:
                logger.warning(f"⚠️ Session {session_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get session info: {e}")
            return None

    def get_all_sessions(self, limit: int = 50, offset: int = 0) -> Dict[str, Dict[str, Any]]:
        """Get paginated session list for better performance"""
        try:
            all_sessions = {}
            
            # Get sessions with pagination
            session_items = list(self.sessions.items())
            paginated_items = session_items[offset:offset + limit]
            
            for session_id, session_data in paginated_items:
                all_sessions[session_id] = {
                    "filename": session_data.get("filename", "unknown"),
                    "annotation_count": len(session_data.get("annotations", [])),
                    "created_at": session_data.get("created_at", "unknown"),
                    "last_accessed": session_data.get("last_accessed", "unknown"),
                    "status": session_data.get("status", "unknown"),
                    "metadata": session_data.get("metadata", {})
                }
            
            logger.info(f"📊 Retrieved {len(all_sessions)} sessions (offset: {offset}, limit: {limit})")
            return all_sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to get sessions: {e}")
            return {}

    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Update session metadata (e.g., page count, file size)"""
        try:
            if session_id in self.sessions:
                self.sessions[session_id]["metadata"].update(metadata)
                self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
                self.sessions.move_to_end(session_id)
                
                logger.info(f"📋 Updated metadata for session {session_id}")
                return True
            else:
                logger.warning(f"⚠️ Session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update metadata: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and free memory"""
        try:
            if session_id in self.sessions:
                session_data = self.sessions[session_id]
                annotation_count = len(session_data.get("annotations", []))
                filename = session_data.get("filename", "unknown")
                
                del self.sessions[session_id]
                
                logger.info(f"🗑️ Deleted session:")
                logger.info(f"   🆔 Session: {session_id}")
                logger.info(f"   📄 Filename: {filename}")
                logger.info(f"   📝 Annotations removed: {annotation_count}")
                logger.info(f"   📊 Remaining sessions: {len(self.sessions)}")
                
                return True
            else:
                logger.warning(f"⚠️ Attempted to delete non-existent session: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete session {session_id}: {e}")
            return False

    def update_session_status(self, session_id: str, status: str) -> bool:
        """Update session status with tracking"""
        try:
            if session_id in self.sessions:
                old_status = self.sessions[session_id].get("status", "unknown")
                self.sessions[session_id]["status"] = status
                self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
                self.sessions.move_to_end(session_id)
                
                logger.info(f"📋 Updated session status: {old_status} → {status}")
                return True
            else:
                logger.warning(f"⚠️ Session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update status: {e}")
            return False

    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """Remove sessions older than specified hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            sessions_to_remove = []
            
            for session_id, session_data in self.sessions.items():
                last_accessed = session_data.get("last_accessed", session_data.get("created_at"))
                if last_accessed:
                    try:
                        access_time = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                        if access_time < cutoff_time:
                            sessions_to_remove.append(session_id)
                    except:
                        continue
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
            
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup sessions: {e}")
            return 0

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get optimized statistics about all sessions"""
        try:
            total_sessions = len(self.sessions)
            total_annotations = 0
            total_pages = set()
            status_counts = {}
            file_types = {}
            
            for session in self.sessions.values():
                # Count annotations
                annotations = session.get("annotations", [])
                total_annotations += len(annotations)
                
                # Track unique pages
                for ann in annotations:
                    if "page_number" in ann:
                        total_pages.add(ann["page_number"])
                
                # Status breakdown
                status = session.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # File type breakdown
                filename = session.get("filename", "")
                if "." in filename:
                    ext = filename.split(".")[-1].lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            # Memory usage estimate (rough)
            memory_usage_mb = len(json.dumps(dict(self.sessions))) / (1024 * 1024)
            
            statistics = {
                "total_sessions": total_sessions,
                "total_annotations": total_annotations,
                "unique_pages_annotated": len(total_pages),
                "average_annotations_per_session": total_annotations / total_sessions if total_sessions > 0 else 0,
                "status_breakdown": status_counts,
                "file_type_breakdown": file_types,
                "memory_usage_mb": round(memory_usage_mb, 2),
                "capacity_used_percent": round((total_sessions / self.max_sessions) * 100, 1)
            }
            
            logger.info(f"📊 Session statistics calculated")
            return statistics
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {
                "error": str(e),
                "total_sessions": len(self.sessions)
            }

    def export_session(self, session_id: str) -> Optional[str]:
        """Export session data as JSON string"""
        try:
            if session_id in self.sessions:
                session_data = self.sessions[session_id]
                return json.dumps(session_data, indent=2)
            return None
        except Exception as e:
            logger.error(f"Failed to export session: {e}")
            return None

    def import_session(self, session_id: str, session_json: str) -> bool:
        """Import session from JSON string"""
        try:
            session_data = json.loads(session_json)
            self.sessions[session_id] = session_data
            logger.info(f"✅ Imported session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return False

    def clear_all_sessions(self) -> int:
        """Clear all sessions for cleanup"""
        try:
            session_count = len(self.sessions)
            total_annotations = sum(
                len(session.get("annotations", [])) 
                for session in self.sessions.values()
            )
            
            self.sessions.clear()
            
            logger.info(f"🧹 Cleared all sessions:")
            logger.info(f"   📊 Sessions removed: {session_count}")
            logger.info(f"   📝 Annotations removed: {total_annotations}")
            
            return session_count
            
        except Exception as e:
            logger.error(f"❌ Failed to clear sessions: {e}")
            return 0
