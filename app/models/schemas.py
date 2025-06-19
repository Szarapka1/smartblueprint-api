# app/models/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List

# --- Request Schema: Asking a question about a blueprint session ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for the uploaded blueprint document.")
    prompt: str = Field(..., min_length=1, description="The user's question for the AI.")
    page_number: Optional[int] = Field(None, description="Optional page number to focus the question.")

# --- Response Schema: What the AI sends back ---
class ChatResponse(BaseModel):
    session_id: str
    ai_response: str
    source_pages: List[int] = Field(default_factory=list, description="Pages used in generating the answer.")

# --- Annotation Model: Marks on a specific page ---
class Annotation(BaseModel):
    page_number: int = Field(..., description="Page number where annotation is placed.")
    x: float = Field(..., description="X-coordinate on the page (in pixels or percent).")
    y: float = Field(..., description="Y-coordinate on the page.")
    text: str = Field(..., description="The content or label of the annotation.")

# --- Response for a visual page with annotations ---
class PageResponse(BaseModel):
    session_id: str
    page_number: int
    image_url: str = Field(..., description="URL to the rendered page image.")
    annotations: List[Annotation] = Field(default_factory=list, description="List of annotations for this page.")
