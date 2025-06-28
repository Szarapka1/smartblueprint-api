# app/models/schemas.py - FIXED AND OPTIMIZED VERSION

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

# --- Enums for type safety ---

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
    should_create_note: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI confidence this should be noted")
    reason: str = Field(..., description="Why AI thinks this should be noted")
    category: NotificationCategory = Field(..., description="Category of the suggestion")
    
    # Suggested note properties
    suggested_text: str
    suggested_type: NoteType = Field(NoteType.general, description="Suggested note type")
    suggested_priority: Priority = Field(Priority.normal, description="Suggested priority")
    suggested_impacts_trades: List[str] = Field(default_factory=list)
    
    # Context
    related_pages: List[int] = Field(default_factory=list)
    related_grid_refs: List[str] = Field(default_factory=list)
    related_elements: List[str] = Field(default_factory=list)
    related_query_sessions: List[str] = Field(default_factory=list)
    source_quote: Optional[str] = Field(None, description="Relevant quote from AI response")

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
    "AnnotationType", "NoteType", "Priority", "Status", "HighlightType", 
    "ConflictType", "NotificationCategory",
    
    # Core Models
    "GridReference", "DrawingGrid", "VisualElement", "VisualHighlight",
    
    # Request Models
    "ChatRequest", "NoteCreate", "QuickNoteCreate", "NoteUpdate", 
    "NoteBatch", "BatchUpdateData", "HighlightCreate",
    
    # Response Models
    "ChatResponse", "NoteSuggestion", "BatchNoteSuggestion", "Note", 
    "NoteList", "DocumentUploadResponse", "DocumentInfoResponse", 
    "DocumentListResponse", "AnnotationResponse", "HighlightResponse",
    
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
