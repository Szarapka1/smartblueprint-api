# app/services/session_service.py
import uuid
import logging
from typing import Dict, Any, List
from app.core.config import AppSettings
from app.models.schemas import Annotation

logger = logging.getLogger(__name__)

class SessionService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.sessions: Dict[str, Any] = {}
        logger.info("SessionService initialized (in-memory).")

    def create_session(self, original_filename: str) -> str:
        """Creates a new session record and returns the session ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "filename": original_filename,
            "annotations": []
        }
        logger.info(f"Created new session {session_id} for file '{original_filename}'.")
        return session_id

    def add_annotation(self, session_id: str, annotation: Annotation) -> None:
        """Adds an annotation to a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id]["annotations"].append(annotation.model_dump())
            logger.info(f"Added annotation to session {session_id} on page {annotation.page_number}.")
        else:
            logger.warning(f"Attempted to add annotation to non-existent session {session_id}.")

    def get_annotations_for_page(self, session_id: str, page_number: int) -> List[Dict]:
        """Retrieves all annotations for a specific page within a session."""
        if session_id in self.sessions:
            all_annotations = self.sessions[session_id].get("annotations", [])
            page_annotations = [
                ann for ann in all_annotations if ann.get("page_number") == page_number
            ]
            return page_annotations
        return []