# app/models/schemas.py - COMPLETE VERSION WITH ALL FEATURES

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

# === ADD THIS SECTION - Missing models that might be imported elsewhere ===

class HighlightType(str):
    """Highlight type constants"""
    AREA = "area"
    LINE = "line"
    POINT = "point"
    TEXT = "text"

class VisualHighlight(BaseModel):
    """Visual highlight on a document - this was the missing import!"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # area, line, point, text
    coordinates: List[float]  # Format depends on type
    page: int
    label: Optional[str] = None
    color: Optional[str] = "#FF0000"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    # Link to annotation if created from AI
    annotation_id: Optional[str] = None
    query_session_id: Optional[str] = None

# === END OF ADDED SECTION ===

# --- Document Chat Models ---

class ChatRequest(BaseModel):
    """Request for chatting with AI about a document"""
    session_id: str = Field(..., description="Session ID for the uploaded document")
    prompt: str = Field(..., min_length=1, max_length=2000, description="User's question")
    current_page: Optional[int] = Field(None, description="Current page being viewed")
    author: str = Field(..., description="User making the request")
    trade: Optional[str] = Field(None, description="User's trade (optional - for context only)")
    reference_previous: Optional[List[str]] = Field(None, description="Element types to include from previous queries")
    preserve_existing: bool = Field(False, description="Keep existing highlights when adding new ones")
    # Optional features - anyone can use these
    show_trade_info: bool = Field(False, description="Include trade information in response")
    detect_conflicts: bool = Field(False, description="Detect potential conflicts between trades")
    # NEW: User preferences for note suggestions
    auto_suggest_notes: bool = Field(True, description="AI should suggest creating notes for important findings")
    note_suggestion_threshold: str = Field("medium", description="Threshold for suggestions: low/medium/high")

# --- NEW: Note Suggestion Models ---

class NoteSuggestion(BaseModel):
    """AI-suggested note based on findings"""
    should_create_note: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="AI confidence this should be noted")
    reason: str = Field(..., description="Why AI thinks this should be noted")
    category: str = Field(..., description="Category: code_issue, coordination, safety, calculation, follow_up")
    
    # Suggested note properties
    suggested_text: str
    suggested_type: str  # general, question, issue, warning, coordination
    suggested_priority: str  # low, normal, high, critical
    suggested_impacts_trades: List[str] = Field(default_factory=list)
    
    # Context
    related_pages: List[int] = Field(default_factory=list)
    related_grid_refs: List[str] = Field(default_factory=list)
    related_elements: List[str] = Field(default_factory=list)
    source_quote: Optional[str] = Field(None, description="Relevant quote from AI response")

class BatchNoteSuggestion(BaseModel):
    """Multiple note suggestions from comprehensive analysis"""
    total_suggestions: int
    critical_count: int = 0
    high_priority_count: int = 0
    normal_priority_count: int = 0
    
    suggestions: List[NoteSuggestion]
    summary: str = Field(..., description="Summary of what was found")

# --- Visual Highlighting Models ---

class GridReference(BaseModel):
    """Grid reference on a drawing"""
    grid_ref: str = Field(..., description="Full grid reference (e.g., W2-WA)")
    x_grid: str = Field(..., description="X-axis grid label")
    y_grid: Optional[str] = Field(None, description="Y-axis grid label")

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
    # NEW: Enhanced trade coordination
    related_trades: Optional[List[str]] = Field(default_factory=list, description="Other trades affected")
    coordination_notes: Optional[str] = None

class DrawingGrid(BaseModel):
    """Drawing grid system information"""
    x_labels: List[str]
    y_labels: List[str]
    scale: Optional[str] = None

class ChatResponse(BaseModel):
    """Enhanced chat response with note suggestions"""
    session_id: str
    ai_response: str
    source_pages: List[int] = Field(default_factory=list)
    visual_highlights: Optional[List[VisualElement]] = None  # Current page highlights for this user
    drawing_grid: Optional[DrawingGrid] = None
    highlight_summary: Optional[Dict[str, int]] = None
    current_page: Optional[int] = None
    trade_conflicts: Optional[List[Dict]] = None  # For cross-trade conflicts
    # NEW fields for multi-page highlighting
    query_session_id: Optional[str] = None  # Groups highlights from this query (private to user)
    all_highlight_pages: Optional[Dict[int, int]] = None  # {page_num: element_count}
    # NEW: Trade analysis
    trade_summary: Optional[Dict[str, Dict[str, int]]] = None  # {trade: {element_type: count}}
    detected_conflicts: Optional[List['TradeConflict']] = None
    
    # NEW: Note suggestions from AI
    note_suggestion: Optional[NoteSuggestion] = None
    batch_suggestions: Optional[BatchNoteSuggestion] = None

# --- Enhanced Note Models ---

class QuickNoteCreate(BaseModel):
    """Quick note creation from AI suggestion"""
    text: str
    note_type: str
    priority: str
    impacts_trades: List[str] = Field(default_factory=list)
    is_private: bool = Field(True)
    
    # Auto-populated from AI context
    ai_suggested: bool = Field(True)
    suggestion_confidence: float
    related_query_session: Optional[str] = None
    related_highlights: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)

class NoteCreate(BaseModel):
    """Create a document-level note"""
    text: str = Field(..., min_length=1, max_length=10000)
    note_type: str = Field("general", description="Type: general, question, issue, warning, coordination")
    impacts_trades: List[str] = Field(default_factory=list, description="Trades impacted by this note")
    priority: str = Field("normal", description="Priority: low, normal, high, critical")
    is_private: bool = Field(True, description="Private to author or public to all")
    # NEW: Related elements
    related_element_ids: Optional[List[str]] = Field(default_factory=list, description="Related visual elements")
    related_query_sessions: Optional[List[str]] = Field(default_factory=list, description="Related highlight sessions")
    # NEW: AI suggestion tracking
    ai_suggested: bool = Field(False, description="Was this suggested by AI?")
    suggestion_confidence: Optional[float] = Field(None, description="AI confidence if suggested")

class Note(BaseModel):
    """Document-level note - private by default until published"""
    note_id: str
    document_id: str
    text: str
    note_type: str
    author: str
    author_trade: Optional[str] = None
    impacts_trades: List[str] = Field(default_factory=list)
    priority: str = "normal"
    is_private: bool = True  # Private by default - user must explicitly publish
    timestamp: str
    edited_at: Optional[str] = None
    published_at: Optional[str] = None  # When made public
    char_count: int
    status: str = "open"  # open, resolved, in_progress
    # NEW: Related elements
    related_element_ids: List[str] = Field(default_factory=list)
    related_query_sessions: List[str] = Field(default_factory=list)
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    # NEW: AI suggestion tracking
    ai_suggested: bool = False
    suggestion_confidence: Optional[float] = None
    source_pages: List[int] = Field(default_factory=list)

class NoteUpdate(BaseModel):
    """Update a note"""
    text: Optional[str] = None
    note_type: Optional[str] = None
    impacts_trades: Optional[List[str]] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    # NEW: Resolution tracking
    resolution_notes: Optional[str] = None
    related_element_ids: Optional[List[str]] = None

class NoteList(BaseModel):
    """List of notes with metadata - includes both private and published notes visible to user"""
    notes: List[Note]
    total_count: int
    filter_applied: Optional[Dict[str, str]] = None
    # NEW: Note breakdown
    private_notes_count: Optional[int] = None  # User's private notes
    published_notes_count: Optional[int] = None  # Public notes from all users
    notes_by_status: Optional[Dict[str, int]] = None
    ai_suggested_count: Optional[int] = None  # How many were AI suggested

class NoteBatch(BaseModel):
    """Batch operations on notes"""
    note_ids: List[str]
    # NEW: Batch operations
    operation: Optional[str] = Field("update", description="Operation: update, resolve, delete")
    update_data: Optional[NoteUpdate] = None

# --- User Preferences ---

class UserPreferences(BaseModel):
    """User preferences for AI behavior"""
    auto_suggest_notes: bool = Field(True, description="Auto-suggest note creation")
    suggestion_threshold: str = Field("medium", description="low/medium/high")
    default_note_priority: str = Field("normal")
    quick_save_enabled: bool = Field(True, description="Enable one-click save")
    preferred_note_type: str = Field("general")
    auto_link_highlights: bool = Field(True, description="Auto-link notes to current highlights")

# --- Trade Coordination Models ---

class TradeConflict(BaseModel):
    """Cross-trade conflict detection"""
    conflict_id: str
    location: GridReference
    trades_involved: List[str]
    conflict_type: str  # spatial, scheduling, system, access
    severity: str  # low, medium, high, critical
    description: str
    resolution_notes: Optional[str] = None
    status: str = "unresolved"  # unresolved, in_progress, resolved
    # NEW: Enhanced conflict tracking
    detected_at: str
    detected_by_session: str
    element_ids: List[str] = Field(default_factory=list, description="Conflicting element IDs")
    page_numbers: List[int] = Field(default_factory=list, description="Pages with conflicts")
    suggested_resolution: Optional[str] = None
    assigned_to_trade: Optional[str] = None
    # NEW: AI suggested this be noted
    ai_suggested_note: bool = False

class TradeNotification(BaseModel):
    """Notification for trade coordination"""
    notification_id: str
    from_trade: str
    to_trades: List[str]
    note_id: Optional[str] = None
    conflict_id: Optional[str] = None
    message: str
    priority: str
    timestamp: str
    read_by: Dict[str, bool] = Field(default_factory=dict)
    # NEW: Enhanced notifications
    notification_type: str = Field("general", description="Type: general, conflict, resolution, update")
    related_elements: List[str] = Field(default_factory=list)
    action_required: bool = Field(False)
    expires_at: Optional[str] = None

# --- Annotation Models (For Visual Highlights Storage) ---

class Annotation(BaseModel):
    """Visual highlight storage - private to the user who created the query"""
    annotation_id: str
    document_id: str
    page_number: int
    element_type: str  # catch_basin, column, sprinkler_head, etc.
    grid_reference: str  # W2-WA, B-3, etc.
    label: Optional[str] = None  # CB-301, C-101, etc.
    x: float = 0  # Keep for backward compatibility
    y: float = 0  # Keep for backward compatibility
    text: str  # Description of element
    annotation_type: str = "ai_highlight"  # Distinguish from user annotations
    author: str  # User who asked the question (owns these highlights)
    is_private: bool = True  # AI highlights are private to requesting user
    query_session_id: str  # Groups all highlights from one question
    created_at: str
    expires_at: Optional[str] = None  # When to auto-clear (e.g., 24 hours)
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    # NEW: Trade assignment
    assigned_trade: Optional[str] = None
    related_trades: List[str] = Field(default_factory=list)
    coordination_required: bool = Field(False)

class AnnotationResponse(BaseModel):
    """Response when creating/updating annotations"""
    annotation_id: str
    document_id: str
    page_number: int
    element_type: str
    grid_reference: str
    query_session_id: str
    created_at: str
    # NEW: Trade info
    assigned_trade: Optional[str] = None

# --- Document Management Models ---

class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: str
    filename: str
    status: str
    message: str
    file_size_mb: float
    # NEW: Processing summary
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
    # NEW: Public collaboration info only
    total_published_notes: Optional[int] = None  # Count of public notes
    active_collaborators: Optional[int] = None  # Users who have published notes
    recent_public_activity: Optional[bool] = None  # Has recent published content

class DocumentListResponse(BaseModel):
    """List of documents response"""
    documents: List[Dict[str, Any]]
    total_count: int
    has_more: bool
    # NEW: Filtering info
    filter_applied: Optional[Dict[str, Any]] = None

# --- Enhanced Response Models ---

class HighlightSessionInfo(BaseModel):
    """Information about a highlight session - private to the user who created it"""
    query_session_id: str
    query: str
    created_at: str
    expires_at: str
    pages_with_highlights: Dict[int, int]
    element_types: List[str]
    total_highlights: int
    user: str  # Owner of this highlight session
    trade: Optional[str] = None
    is_active: bool
    can_merge: bool = True
    # NEW: Notes created from this session
    notes_created: int = 0
    ai_suggestions_made: int = 0

class TradeFilterRequest(BaseModel):
    """Request for filtering by trade"""
    trades: List[str]
    include_related: bool = Field(True, description="Include elements that affect multiple trades")
    
class ConflictResolutionRequest(BaseModel):
    """Request to resolve a conflict"""
    conflict_id: str
    resolution_notes: str
    resolved_by: str
    resolved_by_trade: str
    notify_trades: bool = Field(True)

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

# === Additional Models that might be referenced elsewhere ===

class SessionResponse(BaseModel):
    """Session information response"""
    session_id: str
    document_id: str
    filename: str
    created_at: datetime
    page_count: int
    status: str = "active"

class HighlightCreate(BaseModel):
    """Create a highlight request"""
    type: str  # area, line, point, text
    coordinates: List[float]
    page: int
    label: Optional[str] = None
    color: Optional[str] = "#FF0000"
    note_content: Optional[str] = None

class HighlightResponse(BaseModel):
    """Highlight response"""
    highlight: VisualHighlight
    status: str = "created"
    message: str = "Highlight created successfully"
