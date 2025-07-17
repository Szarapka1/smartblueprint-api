from .ai_service_core import VisualIntelligenceFirst
from .question_analyzer import QuestionAnalyzer
from .vision_intelligence import VisionIntelligence
from .validation_system import ValidationSystem
from .response_formatter import ResponseFormatter
from .data_loader import DataLoader
from .semantic_highlighter import SemanticHighlighter
from .note_generator import NoteGenerator

# FIXED: Import patterns
from .patterns import VISUAL_PATTERNS, VISION_CONFIG, VISION_PHILOSOPHY, ELEMENT_VARIATIONS

from app.models.schemas import (
    VisualIntelligenceResult,
    ValidationResult,
    TrustMetrics,
    QuestionType,
    NoteSuggestion,
    ElementGeometry,
    SemanticHighlight
)

ProfessionalBlueprintAI = VisualIntelligenceFirst

__all__ = [
    "VisualIntelligenceFirst",
    "ProfessionalBlueprintAI",
    "QuestionAnalyzer", 
    "VisionIntelligence",
    "ValidationSystem",
    "ResponseFormatter",
    "DataLoader",
    "SemanticHighlighter",
    "NoteGenerator",
    "VisualIntelligenceResult",
    "ValidationResult",
    "TrustMetrics",
    "QuestionType",
    "NoteSuggestion",
    "ElementGeometry",
    "SemanticHighlight",
    # FIXED: Add pattern exports
    "VISUAL_PATTERNS",
    "VISION_CONFIG",
    "VISION_PHILOSOPHY",
    "ELEMENT_VARIATIONS"
]
