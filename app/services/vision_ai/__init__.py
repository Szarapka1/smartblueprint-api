from .ai_service_core import VisualIntelligenceFirst
from .question_analyzer import QuestionAnalyzer
from .vision_intelligence import VisionIntelligence
from .validation_system import ValidationSystem
from .response_formatter import ResponseFormatter
from .data_loader import DataLoader
from .semantic_highlighter import SemanticHighlighter
from .note_generator import NoteGenerator
from .page_selection import PageSelector
from .post_processor import PostProcessor
from .enhanced_cache import EnhancedCache
from .calculation_engine import CalculationEngine

# Import patterns
from .patterns import VISUAL_PATTERNS, VISION_CONFIG, VISION_PHILOSOPHY, ELEMENT_VARIATIONS

# Import models from schemas
from app.models.schemas import (
    VisualIntelligenceResult,
    ValidationResult,
    TrustMetrics,
    QuestionType,
    NoteSuggestion,
    ElementGeometry,
    SemanticHighlight
)

# Alias for backward compatibility
ProfessionalBlueprintAI = VisualIntelligenceFirst

__all__ = [
    # Main orchestrator
    "VisualIntelligenceFirst",
    "ProfessionalBlueprintAI",
    
    # Core modules
    "QuestionAnalyzer", 
    "VisionIntelligence",
    "ValidationSystem",
    "ResponseFormatter",
    "DataLoader",
    "SemanticHighlighter",
    "NoteGenerator",
    "PageSelector",
    "PostProcessor",
    "EnhancedCache",
    "CalculationEngine",
    
    # Models
    "VisualIntelligenceResult",
    "ValidationResult",
    "TrustMetrics",
    "QuestionType",
    "NoteSuggestion",
    "ElementGeometry",
    "SemanticHighlight",
    
    # Pattern definitions
    "VISUAL_PATTERNS",
    "VISION_CONFIG",
    "VISION_PHILOSOPHY",
    "ELEMENT_VARIATIONS"
]