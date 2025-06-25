# app/models/schemas.py - DOCUMENT NOTES WITH MULTI-PAGE HIGHLIGHTING

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

# --- Document Chat Models ---

class ChatRequest(BaseModel):
    """Request for chatting with AI about a document"""
    session_id: str = Field(..., description="Session ID for the uploaded document")
    prompt: str = Field(..., min_length=1, max_length=2000, description="User's question")
    current_page: Optional[int] = Field(None, description="Current page being viewed")
    author: str = Field(..., description="User making the request")
    trade: Optional[str] = Field(None, description="User's trade (Electrical, Plumbing, etc.)")
    reference_previous: Optional[List[str]] = Field(None, description="Element types to include from previous queries")
    preserve_existing: bool = Field(False, description="Keep existing highlights when adding new ones")

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

class DrawingGrid(BaseModel):
    """Drawing grid system information"""
    x_labels: List[str]
    y_labels: List[str]
    scale: Optional[str] = None

class ChatResponse(BaseModel):
    """Enhanced chat response with multi-page visual highlights"""
    session_id: str
    ai_response: str
    source_pages: List[int] = Field(default_factory=list)
    visual_highlights: Optional[List[VisualElement]] = None  # Current page highlights
    drawing_grid: Optional[DrawingGrid] = None
    highlight_summary: Optional[Dict[str, int]] = None
    current_page: Optional[int] = None
    trade_conflicts: Optional[List[Dict]] = None  # For cross-trade conflicts
    # NEW fields for multi-page highlighting
    query_session_id: Optional[str] = None  # Groups highlights from this query
    all_highlight_pages: Optional[Dict[int, int]] = None  # {page_num: element_count}

# --- Document Note Models (NO COORDINATES) ---

class NoteCreate(BaseModel):
    """Create a document-level note"""
    text: str = Field(..., min_length=1, max_length=10000)
    note_type: str = Field("general", description="Type: general, question, issue, warning, coordination")
    impacts_trades: List[str] = Field(default_factory=list, description="Trades impacted by this note")
    priority: str = Field("normal", description="Priority: low, normal, high, critical")
    is_private: bool = Field(True, description="Private to author or public to all")

class Note(BaseModel):
    """Document-level note (no page/coordinates)"""
    note_id: str
    document_id: str
    text: str
    note_type: str
    author: str
    author_trade: Optional[str] = None
    impacts_trades: List[str] = Field(default_factory=list)
    priority: str = "normal"
    is_private: bool = True
    timestamp: str
    edited_at: Optional[str] = None
    published_at: Optional[str] = None
    char_count: int
    status: str = "open"  # open, resolved, in_progress

class NoteUpdate(BaseModel):
    """Update a note"""
    text: Optional[str] = None
    note_type: Optional[str] = None
    impacts_trades: Optional[List[str]] = None
    priority: Optional[str] = None
    status: Optional[str] = None

class NoteList(BaseModel):
    """List of notes with metadata"""
    notes: List[Note]
    total_count: int
    filter_applied: Optional[Dict[str, str]] = None

class NoteBatch(BaseModel):
    """Batch operations on notes"""
    note_ids: List[str]

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

# --- Annotation Models (For Visual Highlights Storage) ---

class Annotation(BaseModel):
    """Visual highlight storage - saved per element across all pages"""
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
    author: str = "ai_system"
    is_private: bool = False  # AI highlights are visible to all
    query_session_id: str  # Groups all highlights from one question
    created_at: str
    expires_at: Optional[str] = None  # When to clear (next question)
    confidence: float = Field(0.9, ge=0.0, le=1.0)

class AnnotationResponse(BaseModel):
    """Response when creating/updating annotations"""
    annotation_id: str
    document_id: str
    page_number: int
    element_type: str
    grid_reference: str
    query_session_id: str
    created_at: str

# --- Document Management Models ---

class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: str
    filename: str
    status: str
    message: str
    file_size_mb: float

class DocumentInfoResponse(BaseModel):
    """Document information response"""
    document_id: str
    status: str
    message: str
    exists: bool
    metadata: Optional[Dict[str, any]] = None

class DocumentListResponse(BaseModel):
    """List of documents response"""
    documents: List[Dict[str, any]]
    total_count: int
    has_more: bool

# --- Generic Response Models ---

class SuccessResponse(BaseModel):
    """Generic success response"""
    status: str = "success"
    message: str
    data: Optional[Dict[str, any]] = None

class ErrorResponse(BaseModel):
    """Generic error response"""
    status: str = "error"
    message: str
    details: Optional[str] = None
