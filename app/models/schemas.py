# app/models/schemas.py - FIXED AND OPTIMIZED VERSION WITH SSE SUPPORT

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

# --- Enums for type safety ---

class SSEEventType(str, Enum):
    """Server-Sent Event types"""
    status_update = "status_update"
    resource_ready = "resource_ready"
    processing_complete = "processing_complete"
    error = "error"
    keepalive = "keepalive"

class AnnotationType(str, Enum):
    note = "note"
    ai_highlight = "ai_highlight"
    user_annotation = "user_annotation"

class NoteType(str, Enum):
    general = "general"
    question = "question"
    issue = "issue"
    warning = "warning"
    coordination = "coordination"
    suggestion = "suggestion"
    review = "review"

class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    critical = "critical"

class Status(str, Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"
    closed = "closed"

class HighlightType(str, Enum):
    AREA = "area"
    LINE = "line"
    POINT = "point"
    TEXT = "text"

class ConflictType(str, Enum):
    spatial = "spatial"
    scheduling = "scheduling"
    system = "system"
    access = "access"

class NotificationCategory(str, Enum):
    code_issue = "code_issue"
    coordination = "coordination"
    safety = "safety"
    calculation = "calculation"
    follow_up = "follow_up"

class QuestionType(Enum):
    """Types of questions users can ask about blueprints"""
    COUNT = ("count", "How many X are there?")
    LOCATION = ("location", "Where are the X located?")
    IDENTIFY = ("identify", "What is this? What type?")
    SPECIFICATION = ("specification", "What are the specs/dimensions?")
    COMPLIANCE = ("compliance", "Does this meet code?")
    DETAILED = ("detailed", "Detailed technical analysis")
    ESTIMATE = ("estimate", "Estimate cost/materials/time")
    GENERAL = ("general", "General or open-ended question")

# --- SSE Event Models ---

class SSEEvent(BaseModel):
    """Base SSE event model"""
    event_type: SSEEventType
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    document_id: str

class StatusUpdateEvent(BaseModel):
    """Status update event data"""
    stage: str = Field(..., description="Processing stage")
    message: str = Field(..., description="Human-readable status message")
    progress_percent: int = Field(0, ge=0, le=100, description="Progress percentage")
    pages_processed: int = Field(0, ge=0, description="Number of pages processed")
    estimated_time: Optional[int] = Field(None, description="Estimated time remaining in seconds")

class ResourceReadyEvent(BaseModel):
    """Resource ready event data"""
    resource_type: str = Field(..., description="Type of resource: thumbnail, full_image, ai_image, context_text, metadata, grid_systems")
    resource_id: Optional[str] = Field(None, description="Resource identifier")
    page_number: Optional[int] = Field(None, ge=1, description="Page number for page-specific resources")
    url: Optional[str] = Field(None, description="URL to access the resource")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional resource metadata")

class ProcessingCompleteEvent(BaseModel):
    """Processing complete event data"""
    status: str = Field(..., description="Final status: ready or error")
    total_pages: int = Field(..., ge=0, description="Total pages processed")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    resources_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of available resources")

class ProcessingErrorEvent(BaseModel):
    """Processing error event data"""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    is_fatal: bool = Field(True, description="Whether error is fatal")
    retry_possible: bool = Field(False, description="Whether retry is possible")

# --- Core Models ---

class GridReference(BaseModel):
    """Grid reference on a drawing"""
    grid_ref: str = Field(..., description="Full grid reference (e.g., W2-WA)")
    x_grid: str = Field(..., description="X-axis grid label")
    y_grid: Optional[str] = Field(None, description="Y-axis grid label")

class DrawingGrid(BaseModel):
    """Drawing grid system information"""
    x_labels: List[str]
    y_labels: List[str]
    scale: Optional[str] = None

class VisualElement(BaseModel):
    """Visual element to highlight on drawing"""
    element_id: str
    element_type: str  # column, outlet, catch_basin, sprinkler_head, etc.
    grid_location: GridReference
    label: str
    dimensions: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    trade: Optional[str] = None  # Which trade this belongs to
    page_number: int  # Which page this element is on
    # Enhanced trade coordination
    related_trades: Optional[List[str]] = Field(default_factory=list, description="Other trades affected")
    coordination_notes: Optional[str] = None

class VisualHighlight(BaseModel):
    """Visual highlight on a document"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(..., description="Highlight type: area, line, point, text")
    coordinates: List[float] = Field(..., description="Format depends on type")
    page: int = Field(..., ge=1)
    label: Optional[str] = None
    color: Optional[str] = "#FF0000"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    # Link to annotation if created from AI
    annotation_id: Optional[str] = None
    query_session_id: Optional[str] = None

# --- Vision Intelligence Models ---

class ElementGeometry(BaseModel):
    """Geometry information for a detected element"""
    element_type: str
    geometry_type: str
    center_point: Dict[str, float]
    boundary_points: List[Dict[str, float]] = Field(default_factory=list)
    dimensions: Dict[str, float] = Field(default_factory=dict)
    orientation: float = 0.0
    special_features: Dict[str, Any] = Field(default_factory=dict)

class VisualIntelligenceResult(BaseModel):
    """Result from visual intelligence analysis"""
    element_type: str
    count: int
    locations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float
    visual_evidence: List[str] = Field(default_factory=list)
    pattern_matches: List[str] = Field(default_factory=list)
    grid_references: List[str] = Field(default_factory=list)
    verification_notes: List[str] = Field(default_factory=list)
    page_number: int
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    element_geometries: List[ElementGeometry] = Field(default_factory=list)
    semantic_highlights: List['SemanticHighlight'] = Field(default_factory=list)

class SemanticHighlight(BaseModel):
    """Semantic highlight for AI-detected elements"""
    element_id: str = ""
    element_type: str
    geometry_type: str
    path_points: List[Dict[str, float]] = Field(default_factory=list)
    path_type: str = "polygon"  # polygon, polyline, composite
    visual_description: str = ""
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    page: int
    stroke_color: str = "#FFD700"
    stroke_width: float = 3.0
    fill_opacity: float = 0.2

# FIXED: These are the correct models for validation_system.py
class ValidationResult(BaseModel):
    """Result from a single validation pass"""
    pass_number: int
    methodology: str
    findings: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    trust_score: float = Field(0.0, ge=0.0, le=1.0)
    discrepancies: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    pattern_matches_verified: bool = False

class TrustMetrics(BaseModel):
    """Trust metrics for validation results"""
    visual_intelligence_score: float = Field(0.0, ge=0.0, le=1.0)
    perfect_accuracy_achieved: bool = False
    reliability_score: float = Field(0.0, ge=0.0, le=1.0)
    confidence_basis: str = ""
    validation_consensus: bool = False
    accuracy_sources: List[str] = Field(default_factory=list)
    uncertainty_factors: List[str] = Field(default_factory=list)
    source_quality_scores: Dict[str, float] = Field(default_factory=dict)

# Legacy validation models (renamed to avoid conflicts)
class LegacyValidationResult(BaseModel):
    """Legacy validation result - kept for compatibility"""
    is_valid: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    trust_metrics: 'LegacyTrustMetrics'
    validation_messages: List[str] = Field(default_factory=list)
    element_count: int = 0
    validated_elements: List[Dict[str, Any]] = Field(default_factory=list)
    discrepancies: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    validation_method: str = "multi_phase_validation"

class LegacyTrustMetrics(BaseModel):
    """Legacy trust metrics - kept for compatibility"""
    visual_confidence: float = Field(0.0, ge=0.0, le=1.0)
    ocr_confidence: float = Field(0.0, ge=0.0, le=1.0)
    context_confidence: float = Field(0.0, ge=0.0, le=1.0)
    pattern_match_score: float = Field(0.0, ge=0.0, le=1.0)
    cross_reference_score: float = Field(0.0, ge=0.0, le=1.0)
    overall_trust: float = Field(0.0, ge=0.0, le=1.0)
    confidence_factors: Dict[str, float] = Field(default_factory=dict)

# --- Request Models ---

class ChatRequest(BaseModel):
    """Request for chatting with AI about a document"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="User's question")
    author: str = Field(..., min_length=1, max_length=100, description="User making the request")
    current_page: Optional[int] = Field(None, ge=1, description="Current page being viewed")
    trade: Optional[str] = Field(None, description="User's trade (optional - for context only)")
    reference_previous: Optional[List[str]] = Field(None, description="Element types to include from previous queries")
    preserve_existing: bool = Field(False, description="Keep existing highlights when adding new ones")
    # Optional features
    show_trade_info: bool = Field(False, description="Include trade information in response")
    detect_conflicts: bool = Field(False, description="Detect potential conflicts between trades")
    # User preferences for note suggestions
    auto_suggest_notes: bool = Field(True, description="AI should suggest creating notes for important findings")
    note_suggestion_threshold: str = Field("medium", description="Threshold for suggestions: low/medium/high")
    
    @validator('note_suggestion_threshold')
    def validate_threshold(cls, v):
        if v not in ["low", "medium", "high"]:
            raise ValueError("Threshold must be low, medium, or high")
        return v

# --- Note Suggestion Models ---

class NoteSuggestion(BaseModel):
    """AI-suggested note based on findings"""
    id: str = Field(default_factory=lambda: f"note_{uuid.uuid4().hex[:8]}")
    element_id: str
    title: str
    content: str
    element_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    author: str
    timestamp: str
    tags: List[str] = Field(default_factory=list)

class BatchNoteSuggestion(BaseModel):
    """Multiple note suggestions from comprehensive analysis"""
    total_suggestions: int
    critical_count: int = 0
    high_priority_count: int = 0
    normal_priority_count: int = 0
    
    suggestions: List[NoteSuggestion]
    summary: str = Field(..., description="Summary of what was found")

# --- Note Models ---

class NoteCreate(BaseModel):
    """Create a document-level note"""
    text: str = Field(..., min_length=1, max_length=10000)
    note_type: NoteType = Field(NoteType.general, description="Type of note")
    impacts_trades: List[str] = Field(default_factory=list, description="Trades impacted by this note")
    priority: Priority = Field(Priority.normal, description="Priority level")
    is_private: bool = Field(True, description="Private to author or public to all")
    # Related elements
    related_element_ids: Optional[List[str]] = Field(default_factory=list, description="Related visual elements")
    related_query_sessions: Optional[List[str]] = Field(default_factory=list, description="Related highlight sessions")
    # AI suggestion tracking
    ai_suggested: bool = Field(False, description="Was this suggested by AI?")
    suggestion_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence if suggested")

class QuickNoteCreate(BaseModel):
    """Quick note creation from AI suggestion"""
    text: str = Field(..., min_length=1, max_length=5000)
    note_type: NoteType = Field(NoteType.general)
    author: str = Field(..., min_length=1, max_length=100)
    priority: Priority = Field(Priority.normal)
    impacts_trades: List[str] = Field(default_factory=list)
    is_private: bool = Field(True)
    
    # Auto-populated from AI context
    ai_suggested: bool = Field(True)
    suggestion_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    related_query_sessions: Optional[List[str]] = Field(default_factory=list)
    related_highlights: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)

class NoteUpdate(BaseModel):
    """Update a note"""
    text: Optional[str] = Field(None, min_length=1, max_length=10000)
    note_type: Optional[NoteType] = None
    impacts_trades: Optional[List[str]] = None
    priority: Optional[Priority] = None
    status: Optional[Status] = None
    resolution_notes: Optional[str] = Field(None, max_length=2000)
    related_element_ids: Optional[List[str]] = None

class Note(BaseModel):
    """Document-level note - private by default until published"""
    note_id: str
    document_id: str
    text: str
    note_type: NoteType
    author: str
    author_trade: Optional[str] = None
    impacts_trades: List[str] = Field(default_factory=list)
    priority: Priority = Priority.normal
    is_private: bool = True  # Private by default - user must explicitly publish
    timestamp: str
    edited_at: Optional[str] = None
    published_at: Optional[str] = None  # When made public
    char_count: int
    status: Status = Status.open
    # Related elements
    related_element_ids: List[str] = Field(default_factory=list)
    related_query_sessions: List[str] = Field(default_factory=list)
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    # AI suggestion tracking
    ai_suggested: bool = False
    suggestion_confidence: Optional[float] = None
    source_pages: List[int] = Field(default_factory=list)

class NoteList(BaseModel):
    """List of notes with metadata"""
    notes: List[Note]
    total_count: int
    filter_applied: Optional[Dict[str, Any]] = None
    # Note breakdown
    private_notes_count: int = 0
    published_notes_count: int = 0
    notes_by_status: Dict[str, int] = Field(default_factory=dict)
    ai_suggested_count: int = 0

class BatchUpdateData(BaseModel):
    """Data for batch updates"""
    status: Optional[Status] = None
    priority: Optional[Priority] = None

class NoteBatch(BaseModel):
    """Batch operations on notes"""
    note_ids: List[str] = Field(..., min_items=1)
    operation: str = Field(..., description="Operation: update, resolve, delete")
    update_data: Optional[BatchUpdateData] = None
    
    @validator('operation')
    def validate_operation(cls, v):
        if v not in ["update", "resolve", "delete"]:
            raise ValueError("Operation must be update, resolve, or delete")
        return v

# --- Trade Coordination Models ---

class TradeConflict(BaseModel):
    """Cross-trade conflict detection"""
    conflict_id: str
    location: GridReference
    trades_involved: List[str]
    conflict_type: ConflictType
    severity: Priority
    description: str
    resolution_notes: Optional[str] = None
    status: Status = Status.open
    # Enhanced conflict tracking
    detected_at: str
    detected_by_session: str
    element_ids: List[str] = Field(default_factory=list)
    page_numbers: List[int] = Field(default_factory=list)
    suggested_resolution: Optional[str] = None
    assigned_to_trade: Optional[str] = None
    ai_suggested_note: bool = False

# --- Response Models ---

class ChatResponse(BaseModel):
    """Enhanced chat response with note suggestions"""
    session_id: str
    ai_response: str
    source_pages: List[int] = Field(default_factory=list)
    visual_highlights: Optional[List[VisualElement]] = None
    drawing_grid: Optional[DrawingGrid] = None
    highlight_summary: Optional[Dict[str, int]] = None
    current_page: Optional[int] = None
    trade_conflicts: Optional[List[TradeConflict]] = None
    # Multi-page highlighting
    query_session_id: Optional[str] = None
    all_highlight_pages: Optional[Dict[int, int]] = None
    # Trade analysis
    trade_summary: Optional[Dict[str, Dict[str, int]]] = None
    detected_conflicts: Optional[List[TradeConflict]] = None
    
    # Note suggestions from AI
    note_suggestion: Optional[NoteSuggestion] = None
    batch_suggestions: Optional[BatchNoteSuggestion] = None

class AnnotationBase(BaseModel):
    """Base annotation model"""
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    element_type: str = Field(..., description="Type of element being annotated")
    grid_reference: str = Field(..., description="Grid reference for the element")
    x: int = Field(0, description="X coordinate on page")
    y: int = Field(0, description="Y coordinate on page")
    width: int = Field(100, description="Width of annotation area")
    height: int = Field(100, description="Height of annotation area")
    text: str = Field("", description="Annotation text or description")
    author: str = Field(..., description="Author of the annotation")
    annotation_type: AnnotationType = Field(AnnotationType.note, description="Type of annotation")
    is_private: bool = Field(True, description="Whether annotation is private to author")
    query_session_id: Optional[str] = Field(None, description="Associated query session ID")
    expires_at: Optional[str] = Field(None, description="When the annotation expires (ISO format)")
    assigned_trade: Optional[str] = Field(None, description="Trade responsible for this element")

class Annotation(AnnotationBase):
    """Visual highlight storage - private to the user who created the query"""
    annotation_id: Optional[str] = Field(None, description="Unique identifier")
    document_id: Optional[str] = Field(None, description="Document this annotation belongs to")
    created_at: Optional[str] = Field(None, description="When annotation was created")
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    # Trade assignment
    related_trades: List[str] = Field(default_factory=list)
    coordination_required: bool = Field(False)

class AnnotationResponse(BaseModel):
    """Response when creating/updating annotations"""
    annotation_id: str
    document_id: str
    page_number: int
    element_type: str
    grid_reference: str
    query_session_id: Optional[str] = None
    created_at: str
    assigned_trade: Optional[str] = None

# --- Document Management Models ---

class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: str
    filename: str
    status: str
    message: str
    file_size_mb: float
    pages_processed: Optional[int] = None
    grid_systems_detected: Optional[int] = None
    drawing_types_found: Optional[List[str]] = None
    estimated_time: Optional[int] = None

class DocumentInfoResponse(BaseModel):
    """Document information response"""
    document_id: str
    status: str
    message: str
    exists: bool
    metadata: Optional[Dict[str, Any]] = None
    total_published_notes: int = 0
    active_collaborators: int = 0
    recent_public_activity: bool = False
    session_info: Optional[Dict[str, Any]] = None

class DocumentListResponse(BaseModel):
    """List of documents response"""
    documents: List[DocumentInfoResponse]
    total_count: int

# Enhanced status response for progressive loading with SSE info
class DocumentStatusResponse(BaseModel):
    """Enhanced status response for progressive loading"""
    document_id: str
    status: str
    message: Optional[str] = ""
    total_pages: Optional[int] = None
    pages_processed: int = 0
    available_resources: Dict[str, Any]
    error: Optional[str] = None
    uploaded_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    processing_progress_percent: int = 0
    sse_connection_url: Optional[str] = Field(None, description="SSE connection URL if SSE is enabled")

# --- Additional document route models ---

class DocumentStatsResponse(BaseModel):
    """Document statistics response"""
    document_id: str
    total_annotations: int
    total_pages: int
    total_characters: int
    estimated_tokens: int
    unique_collaborators: List[str]
    collaborator_count: int
    annotation_types: Dict[str, int]
    last_activity: Optional[str]
    status: str

class DocumentActivityResponse(BaseModel):
    """Document activity response"""
    document_id: str
    recent_annotations: List[Dict[str, Any]]
    recent_chats: List[Dict[str, Any]]
    active_authors: List[str]

class CollaboratorStats(BaseModel):
    """Collaborator statistics"""
    author: str
    annotations_count: int
    chats_count: int
    total_interactions: int

class CollaboratorsResponse(BaseModel):
    """Collaborators response"""
    document_id: str
    collaborators: List[CollaboratorStats]
    total_collaborators: int

# --- User Preferences ---

class UserPreferences(BaseModel):
    """User preferences for AI behavior"""
    auto_suggest_notes: bool = Field(True, description="Auto-suggest note creation")
    note_suggestion_threshold: str = Field("medium", description="low/medium/high")
    default_note_priority: Priority = Field(Priority.normal)
    quick_save_enabled: bool = Field(True, description="Enable one-click save")
    preferred_note_type: NoteType = Field(NoteType.general)
    auto_link_highlights: bool = Field(True, description="Auto-link notes to current highlights")
    show_trade_info: bool = Field(False)
    detect_conflicts: bool = Field(False)
    default_note_privacy: bool = Field(True)
    
    @validator('note_suggestion_threshold')
    def validate_threshold(cls, v):
        if v not in ["low", "medium", "high"]:
            raise ValueError("Threshold must be low, medium, or high")
        return v

# --- Generic Response Models ---

class SuccessResponse(BaseModel):
    """Generic success response"""
    status: str = "success"
    message: str
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Generic error response"""
    status: str = "error"
    message: str
    details: Optional[str] = None
    timestamp: Optional[str] = None

# --- Additional Utility Models ---

class HighlightCreate(BaseModel):
    """Create a highlight request"""
    type: HighlightType
    coordinates: List[float]
    page: int = Field(..., ge=1)
    label: Optional[str] = None
    color: Optional[str] = "#FF0000"
    note_content: Optional[str] = None

class HighlightResponse(BaseModel):
    """Highlight response"""
    highlight: VisualHighlight
    status: str = "created"
    message: str = "Highlight created successfully"

class ServiceStatus(BaseModel):
    """Service status information"""
    service_name: str
    status: str
    message: Optional[str] = None
    last_check: str

class HealthCheck(BaseModel):
    """Health check response"""
    overall_status: str
    timestamp: str
    services: Dict[str, Dict[str, Any]]
    version: str

class ConfigurationStatus(BaseModel):
    """Configuration status response"""
    features: Dict[str, bool]
    limits: Dict[str, int]
    environment: Dict[str, Any]

# --- Export all models ---
__all__ = [
    # Enums
    "SSEEventType", "AnnotationType", "NoteType", "Priority", "Status", "HighlightType", 
    "ConflictType", "NotificationCategory", "QuestionType",
    
    # SSE Event Models
    "SSEEvent", "StatusUpdateEvent", "ResourceReadyEvent", "ProcessingCompleteEvent", "ProcessingErrorEvent",
    
    # Core Models
    "GridReference", "DrawingGrid", "VisualElement", "VisualHighlight",
    
    # Vision Intelligence Models
    "ElementGeometry", "VisualIntelligenceResult", "SemanticHighlight",
    "TrustMetrics", "ValidationResult",
    
    # Legacy Models (kept for compatibility)
    "LegacyTrustMetrics", "LegacyValidationResult",
    
    # Request Models
    "ChatRequest", "NoteCreate", "QuickNoteCreate", "NoteUpdate", 
    "NoteBatch", "BatchUpdateData", "HighlightCreate",
    
    # Response Models
    "ChatResponse", "NoteSuggestion", "BatchNoteSuggestion", "Note", 
    "NoteList", "DocumentUploadResponse", "DocumentInfoResponse", 
    "DocumentListResponse", "DocumentStatusResponse", "DocumentStatsResponse",
    "DocumentActivityResponse", "CollaboratorsResponse", "CollaboratorStats",
    "AnnotationResponse", "HighlightResponse",
    
    # Annotation Models
    "Annotation", "AnnotationBase",
    
    # Trade Models
    "TradeConflict",
    
    # User Models
    "UserPreferences",
    
    # Utility Models
    "SuccessResponse", "ErrorResponse", "ServiceStatus", "HealthCheck", 
    "ConfigurationStatus"
]

# Fix forward references after all models are defined
ChatResponse.update_forward_refs()
VisualIntelligenceResult.update_forward_refs()
LegacyValidationResult.update_forward_refs()