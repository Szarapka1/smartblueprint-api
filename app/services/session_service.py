# app/services/session_service.py - Enhanced with Better Logging and Debugging
import uuid
import logging
from typing import Dict, Any, List, Optional
from app.core.config import AppSettings, get_settings
from app.models.schemas import Annotation

logger = logging.getLogger(__name__)

class SessionService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.sessions: Dict[str, Any] = {}
        logger.info("✅ SessionService initialized (in-memory storage)")
        logger.info(f"📊 Initial session count: 0")

    def create_session(self, original_filename: str) -> str:
        """Creates a new session record and returns the session ID."""
        try:
            session_id = str(uuid.uuid4())
            
            session_data = {
                "filename": original_filename,
                "annotations": [],
                "created_at": logger.handlers[0].formatter.formatTime(logging.LogRecord('', 0, '', 0, '', (), None), '%Y-%m-%d %H:%M:%S') if logger.handlers else "unknown",
                "status": "created"
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
        """Adds an annotation to a specific session."""
        try:
            if session_id in self.sessions:
                annotation_data = annotation.model_dump()
                self.sessions[session_id]["annotations"].append(annotation_data)
                
                logger.info(f"✅ Added annotation to session:")
                logger.info(f"   🆔 Session: {session_id}")
                logger.info(f"   📄 Page: {annotation.page_number}")
                logger.info(f"   📝 Content: '{annotation.content[:50]}{'...' if len(annotation.content) > 50 else ''}'")
                logger.info(f"   📊 Total annotations for session: {len(self.sessions[session_id]['annotations'])}")
                
            else:
                logger.warning(f"⚠️ Attempted to add annotation to non-existent session: {session_id}")
                logger.warning(f"📊 Available sessions: {list(self.sessions.keys())}")
                
        except Exception as e:
            logger.error(f"❌ Failed to add annotation to session {session_id}: {e}")
            raise

    def get_annotations_for_page(self, session_id: str, page_number: int) -> List[Dict]:
        """Retrieves all annotations for a specific page within a session."""
        try:
            if session_id in self.sessions:
                all_annotations = self.sessions[session_id].get("annotations", [])
                page_annotations = [
                    ann for ann in all_annotations if ann.get("page_number") == page_number
                ]
                
                logger.info(f"📋 Retrieved annotations for session {session_id}, page {page_number}: {len(page_annotations)} found")
                
                return page_annotations
            else:
                logger.warning(f"⚠️ Session {session_id} not found when getting annotations for page {page_number}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get annotations for session {session_id}, page {page_number}: {e}")
            return []

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        try:
            if session_id in self.sessions:
                session_data = self.sessions[session_id].copy()
                session_data["annotation_count"] = len(session_data.get("annotations", []))
                
                # Count annotations per page
                annotations = session_data.get("annotations", [])
                page_counts = {}
                for ann in annotations:
                    page_num = ann.get("page_number", 0)
                    page_counts[page_num] = page_counts.get(page_num, 0) + 1
                
                session_data["annotations_per_page"] = page_counts
                
                logger.info(f"📊 Session info for {session_id}: {session_data['annotation_count']} annotations across {len(page_counts)} pages")
                return session_data
            else:
                logger.warning(f"⚠️ Session {session_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get session info for {session_id}: {e}")
            return None

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all sessions."""
        try:
            all_sessions = {}
            for session_id, session_data in self.sessions.items():
                session_summary = {
                    "filename": session_data.get("filename", "unknown"),
                    "annotation_count": len(session_data.get("annotations", [])),
                    "created_at": session_data.get("created_at", "unknown"),
                    "status": session_data.get("status", "unknown")
                }
                all_sessions[session_id] = session_summary
            
            logger.info(f"📊 Retrieved info for {len(all_sessions)} sessions")
            return all_sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to get all sessions: {e}")
            return {}

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its annotations."""
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
        """Update the status of a session."""
        try:
            if session_id in self.sessions:
                old_status = self.sessions[session_id].get("status", "unknown")
                self.sessions[session_id]["status"] = status
                
                logger.info(f"📋 Updated session status:")
                logger.info(f"   🆔 Session: {session_id}")
                logger.info(f"   📊 Status: {old_status} → {status}")
                
                return True
            else:
                logger.warning(f"⚠️ Cannot update status for non-existent session: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update session status for {session_id}: {e}")
            return False

    def clear_all_sessions(self) -> int:
        """Clear all sessions (useful for testing or cleanup)."""
        try:
            session_count = len(self.sessions)
            total_annotations = sum(len(session.get("annotations", [])) for session in self.sessions.values())
            
            self.sessions.clear()
            
            logger.info(f"🧹 Cleared all sessions:")
            logger.info(f"   📊 Sessions removed: {session_count}")
            logger.info(f"   📝 Annotations removed: {total_annotations}")
            
            return session_count
            
        except Exception as e:
            logger.error(f"❌ Failed to clear all sessions: {e}")
            return 0

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about all sessions."""
        try:
            total_sessions = len(self.sessions)
            total_annotations = sum(len(session.get("annotations", [])) for session in self.sessions.values())
            
            # Status breakdown
            status_counts = {}
            for session in self.sessions.values():
                status = session.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # File type breakdown (from filenames)
            file_types = {}
            for session in self.sessions.values():
                filename = session.get("filename", "")
                if "." in filename:
                    ext = filename.split(".")[-1].lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                else:
                    file_types["no_extension"] = file_types.get("no_extension", 0) + 1
            
            statistics = {
                "total_sessions": total_sessions,
                "total_annotations": total_annotations,
                "average_annotations_per_session": total_annotations / total_sessions if total_sessions > 0 else 0,
                "status_breakdown": status_counts,
                "file_type_breakdown": file_types
            }
            
            logger.info(f"📊 Session statistics: {statistics}")
            return statistics
            
        except Exception as e:
            logger.error(f"❌ Failed to get session statistics: {e}")
            return {
                "total_sessions": 0,
                "total_annotations": 0,
                "average_annotations_per_session": 0,
                "status_breakdown": {},
                "file_type_breakdown": {},
                "error": str(e)
            }
