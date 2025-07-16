# app/services/ai_service.py - ENHANCED WITH AI NOTE SUGGESTIONS

import os
import asyncio
import base64
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, OrderedDict
from PIL import Image, ImageDraw, ImageFont
import io
import uuid

# Set up logger
logger = logging.getLogger(__name__)

# OpenAI imports
try:
    from openai import OpenAI

    logger.info("✅ OpenAI SDK imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import OpenAI SDK: {e}")
    raise

# Optional imports for grid detection
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
    logger.info("✅ OpenCV available for grid detection")
except ImportError:
    OPENCV_AVAILABLE = False
    logger.info("⚠️ OpenCV not available - using fallback grid estimation")

# Internal imports
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService
from app.models.schemas import (
    VisualElement,
    GridReference,
    DrawingGrid,
    NoteSuggestion,
    BatchNoteSuggestion,
)


@dataclass
class GridSystem:
    """Represents a drawing's grid system"""

    page_number: int
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    x_coordinates: Dict[str, int] = field(default_factory=dict)  # label -> pixel x
    y_coordinates: Dict[str, int] = field(default_factory=dict)  # label -> pixel y
    cell_width: int = 100
    cell_height: int = 100
    origin_x: int = 100
    origin_y: int = 100
    scale: Optional[str] = None
    confidence: float = 0.0

    def get_pixel_coords(self, grid_ref: str) -> Optional[Dict[str, int]]:
        """Convert grid reference to pixel coordinates"""
        # Parse various grid reference formats
        patterns = [
            (r"^([A-Z]+)[-\s]?(\d+)$", "letter-number"),  # A-1, B2
            (r"^(\d+)[-\s]?([A-Z]+)$", "number-letter"),  # 1-A, 2B
            (r"^([A-Z]+\d*)[-\s]?([A-Z]+\d*)$", "complex"),  # W2-WA
        ]

        for pattern, pattern_type in patterns:
            match = re.match(pattern, grid_ref.upper().strip())
            if match:
                x_ref = match.group(1)
                y_ref = match.group(2)

                # Get coordinates from known positions or estimate
                x = self.x_coordinates.get(x_ref)
                y = self.y_coordinates.get(y_ref)

                # If not found, try swapped
                if x is None and pattern_type != "complex":
                    x = self.x_coordinates.get(y_ref)
                    y = self.y_coordinates.get(x_ref)

                # If still not found, estimate
                if x is None:
                    x = self._estimate_coordinate(x_ref, "x")
                if y is None:
                    y = self._estimate_coordinate(y_ref, "y")

                if x is not None and y is not None:
                    return {
                        "x": int(x),
                        "y": int(y),
                        "width": self.cell_width,
                        "height": self.cell_height,
                        "center_x": int(x + self.cell_width // 2),
                        "center_y": int(y + self.cell_height // 2),
                    }

        # If no pattern matches, return estimated center position
        return self._get_fallback_position()

    def _estimate_coordinate(self, ref: str, axis: str) -> Optional[int]:
        """Estimate coordinate based on reference"""
        if axis == "x":
            if ref.isalpha() and len(ref) == 1:
                # Simple letter: A=0, B=1, etc.
                index = ord(ref) - ord("A")
                return self.origin_x + (index * self.cell_width)
            elif ref.isdigit():
                # Number on x-axis (less common)
                index = int(ref) - 1
                return self.origin_x + (index * self.cell_width)
        else:  # y-axis
            if ref.isdigit():
                # Simple number: 1=0, 2=1, etc.
                index = int(ref) - 1
                return self.origin_y + (index * self.cell_height)
            elif ref.isalpha() and len(ref) == 1:
                # Letter on y-axis (less common)
                index = ord(ref) - ord("A")
                return self.origin_y + (index * self.cell_height)

        return None

    def _get_fallback_position(self) -> Dict[str, int]:
        """Get a fallback position when grid reference can't be parsed"""
        # Return a position in the middle of the drawing
        return {
            "x": self.origin_x + (5 * self.cell_width),  # Middle of typical grid
            "y": self.origin_y + (5 * self.cell_height),
            "width": self.cell_width,
            "height": self.cell_height,
            "center_x": self.origin_x + (5 * self.cell_width) + self.cell_width // 2,
            "center_y": self.origin_y + (5 * self.cell_height) + self.cell_height // 2,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "page_number": self.page_number,
            "x_labels": self.x_labels,
            "y_labels": self.y_labels,
            "x_coordinates": self.x_coordinates,
            "y_coordinates": self.y_coordinates,
            "cell_width": self.cell_width,
            "cell_height": self.cell_height,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "scale": self.scale,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridSystem":
        """Create from dictionary"""
        return cls(
            page_number=data.get("page_number", 1),
            x_labels=data.get("x_labels", []),
            y_labels=data.get("y_labels", []),
            x_coordinates=data.get("x_coordinates", {}),
            y_coordinates=data.get("y_coordinates", {}),
            cell_width=data.get("cell_width", 100),
            cell_height=data.get("cell_height", 100),
            origin_x=data.get("origin_x", 100),
            origin_y=data.get("origin_y", 100),
            scale=data.get("scale"),
            confidence=data.get("confidence", 0.0),
        )


class ProfessionalBlueprintAI:
    """Professional AI service with note suggestion capabilities"""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY

        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")

        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info(
                "✅ Professional Blueprint AI initialized with note suggestions"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Configuration
        self.max_pages_to_load = self.settings.AI_MAX_PAGES
        self.batch_size = self.settings.AI_BATCH_SIZE
        self.image_quality = self.settings.AI_IMAGE_QUALITY
        self.max_image_dimension = self.settings.AI_MAX_IMAGE_DIMENSION

        # Storage optimization settings
        self.store_highlighted_images = False  # Never store highlighted images
        self.highlight_cache_ttl_hours = 24  # Expire highlight data after 24 hours
        self.grid_cache_size = 100  # Max documents to cache grids for
        self.max_grid_data_size = 1024 * 1024  # 1MB max per document grid data

        # Visual highlight settings
        self.highlight_color = (255, 255, 0, 80)  # Yellow with transparency
        self.highlight_border = (255, 0, 0)  # Red border
        self.highlight_border_width = 3
        self.label_font_size = 14

        # Grid cache with LRU eviction and size limits
        self.grid_cache = OrderedDict()
        self.grid_cache_sizes = {}  # Track size of each cached grid

        # Patterns for element detection
        self.element_patterns = {
            "window": r"(?:TYPE|type)\s*(\d+)|W\d+|WINDOW|window",
            "door": r"(?:TYPE|type)\s*([A-Z]\d*)|D\d+|DOOR|door",
            "column": r"C\d+|COL\.?\s*\d+|COLUMN",
            "beam": r"B\d+|BM\.?\s*\d+|BEAM",
            "catch_basin": r"CB[-\s]?\d+|CATCH\s*BASIN",
            "sprinkler": r"SP[-\s]?\d+|SPRINKLER",
            "outlet": r"OUTLET|RECEPTACLE|DUPLEX",
            "panel": r"PANEL\s*[A-Z]\d*|EP[-\s]?\d+|MDP",
            "equipment": r"UNIT|AHU|RTU|VAV|FCU",
            "fixture": r"WC|LAV|SINK|FIXTURE",
        }

        # Building code requirements for reference
        self.code_requirements = {
            "sprinkler": {
                "NFPA_13": {
                    "light_hazard": {"max_spacing": 15, "max_coverage": 225},
                    "ordinary_hazard": {"max_spacing": 15, "max_coverage": 130},
                    "extra_hazard": {"max_spacing": 12, "max_coverage": 100},
                },
                "BCBC_2018": {
                    "section": "3.2.5.12",
                    "min_from_wall": 4,  # inches
                    "max_from_wall": 72,  # inches (6 feet)
                },
            },
            "outlet": {
                "CEC": {
                    "residential": {"max_spacing": 12},  # feet
                    "commercial": {"workstation_spacing": 6},  # feet
                },
                "BCBC_2018": {"section": "9.23.11", "counter_spacing": 48},  # inches
            },
            "exit": {
                "BCBC_2018": {
                    "section": "3.4",
                    "max_travel_distance": {
                        "sprinklered": 150,  # meters
                        "non_sprinklered": 45,  # meters
                    },
                    "min_width": 1100,  # mm
                }
            },
            "guard": {
                "BCBC_2018": {
                    "section": "9.8.8.3",
                    "min_height": 1070,  # mm
                    "max_opening": 100,  # mm sphere
                }
            },
        }

        # NEW: Note suggestion triggers
        self.note_suggestion_triggers = {
            "code_issue": {
                "keywords": [
                    "not shown",
                    "missing",
                    "should be",
                    "must be",
                    "required",
                    "not specified",
                    "cannot find",
                    "not indicated",
                    "verify compliance",
                ],
                "priority": "high",
                "confidence_boost": 0.2,
            },
            "coordination": {
                "keywords": [
                    "conflict",
                    "interfere",
                    "overlap",
                    "coordination needed",
                    "appears to be in the same location",
                    "clash",
                    "collision",
                    "spatial conflict",
                ],
                "priority": "high",
                "confidence_boost": 0.25,
            },
            "safety": {
                "keywords": [
                    "violation",
                    "dangerous",
                    "non-compliant",
                    "safety concern",
                    "hazard",
                    "does not meet",
                    "below minimum",
                    "exceeds maximum",
                ],
                "priority": "critical",
                "confidence_boost": 0.3,
            },
            "calculation": {
                "keywords": [
                    "total",
                    "calculated",
                    "sum",
                    "area",
                    "length",
                    "count",
                    "square feet",
                    "linear feet",
                    "quantity",
                ],
                "priority": "normal",
                "confidence_boost": 0.1,
            },
            "follow_up": {
                "keywords": [
                    "verify",
                    "confirm",
                    "check with",
                    "RFI needed",
                    "clarification required",
                    "contractor should",
                    "field verify",
                    "coordinate with",
                ],
                "priority": "normal",
                "confidence_boost": 0.15,
            },
        }

        logger.info(f"Configuration:")
        logger.info(f"  📄 Max pages: {self.max_pages_to_load}")
        logger.info(f"  📦 Batch size: {self.batch_size}")
        logger.info(f"  🖼️ Image quality: {self.image_quality}%")
        logger.info(f"  💾 Storage mode: ON-DEMAND (no storage)")
        logger.info(f"  ⏰ Highlight TTL: {self.highlight_cache_ttl_hours}h")
        logger.info(
            f"  🎯 Grid detection: {'OpenCV' if OPENCV_AVAILABLE else 'Estimation'}"
        )
        logger.info(f"  📝 Note suggestions: ENABLED")
        logger.info(
            f"  🧠 Grid cache: {self.grid_cache_size} documents, {self.max_grid_data_size/1024/1024:.1f}MB max each"
        )

    async def get_ai_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        author: str = None,
        current_page: Optional[int] = None,
        request_highlights: bool = True,
        reference_previous: Optional[List[str]] = None,
        preserve_existing: bool = False,
        show_trade_info: bool = False,
        detect_conflicts: bool = False,
        auto_suggest_notes: bool = True,
        note_suggestion_threshold: str = "medium",
    ) -> Dict[str, Any]:
        """Process blueprint queries with visual highlighting and note suggestions"""
        try:
            logger.info(f"📐 Processing query for document: {document_id}")
            logger.info(f"   Current page: {current_page}")
            logger.info(f"   Request highlights: {request_highlights}")
            logger.info(f"   Auto suggest notes: {auto_suggest_notes}")
            if show_trade_info:
                logger.info(f"   Show trade info: {show_trade_info}")
            if detect_conflicts:
                logger.info(f"   Detect conflicts: {detect_conflicts}")

            # Load document context
            document_text, metadata = await self._load_document_context(
                document_id, storage_service
            )

            if not document_text and not metadata:
                return {
                    "ai_response": "Document not found or not properly processed.",
                    "visual_highlights": None,
                    "note_suggestion": None,
                }

            # Check for previous highlights to reuse
            reused_highlights = []
            if reference_previous:
                reused_highlights = await self._get_previous_highlights(
                    document_id, reference_previous, storage_service
                )
                logger.info(f"♻️ Found {len(reused_highlights)} highlights to reuse")

            # Classify query type
            query_type = self._classify_query(prompt, current_page, reference_previous)
            logger.info(f"📋 Query type: {query_type}")

            # Determine which pages need visual analysis
            pages_to_load = await self._determine_pages_to_load(
                query_type, prompt, document_text, metadata, current_page
            )

            # Load pages if needed
            page_images = []
            if pages_to_load:
                page_images = await self._load_specific_pages(
                    document_id, pages_to_load, storage_service
                )
                logger.info(f"📄 Loaded {len(page_images)} pages for analysis")

            # Get AI analysis
            result = await self._analyze_blueprint(
                prompt=prompt,
                document_text=document_text,
                page_images=page_images,
                document_id=document_id,
                author=author,
                current_page=current_page,
                request_highlights=request_highlights,
                reference_previous=reference_previous,
                preserve_existing=preserve_existing,
                storage_service=storage_service,
                metadata=metadata,
                reused_highlights=reused_highlights,
                show_trade_info=show_trade_info,
                detect_conflicts=detect_conflicts,
                auto_suggest_notes=auto_suggest_notes,
                note_suggestion_threshold=note_suggestion_threshold,
            )

            return result

        except Exception as e:
            logger.error(f"Error in get_ai_response: {e}", exc_info=True)
            return {
                "ai_response": f"An error occurred: {str(e)}",
                "visual_highlights": None,
                "note_suggestion": None,
            }

    def _classify_query(
        self,
        prompt: str,
        current_page: Optional[int],
        reference_previous: Optional[List[str]],
    ) -> str:
        """Classify the type of query to optimize processing"""
        prompt_lower = prompt.lower()

        # Reference previous only
        if reference_previous and not any(
            keyword in prompt_lower
            for keyword in ["also", "add", "more", "other", "additional", "besides"]
        ):
            return "reference_only"

        # Page-specific query
        if current_page and any(
            keyword in prompt_lower
            for keyword in [
                "this page",
                "on page",
                f"page {current_page}",
                "here",
                "this drawing",
            ]
        ):
            return "page_specific"

        # Document-wide visual search
        if any(
            keyword in prompt_lower
            for keyword in [
                "all",
                "every",
                "total",
                "count",
                "list",
                "find all",
                "show all",
                "where are all",
            ]
        ):
            return "document_wide_visual"

        # Context-only query
        if any(
            keyword in prompt_lower
            for keyword in [
                "square footage",
                "project name",
                "specifications",
                "general notes",
                "title",
                "summary",
            ]
        ):
            return "context_only"

        # Visual element query
        if any(
            keyword in prompt_lower
            for keyword in [
                "show",
                "highlight",
                "where",
                "locate",
                "find",
                "identify",
                "mark",
                "point out",
            ]
        ):
            return "visual_search"

        return "general"

    async def _determine_pages_to_load(
        self,
        query_type: str,
        prompt: str,
        document_text: str,
        metadata: Optional[Dict],
        current_page: Optional[int],
    ) -> List[int]:
        """Determine which pages need to be loaded based on query type"""

        if query_type == "reference_only":
            return []  # No new pages needed

        if query_type == "context_only":
            return []  # Text context is sufficient

        if query_type == "page_specific" and current_page:
            return [current_page]  # Just the current page

        if query_type == "document_wide_visual":
            # For comprehensive searches, load all pages
            total_pages = metadata.get("page_count", 10) if metadata else 10
            return list(range(1, min(total_pages + 1, self.max_pages_to_load + 1)))

        # For visual searches, try to identify relevant pages from context
        relevant_pages = self._identify_relevant_pages(prompt, document_text, metadata)

        # If current page is set, prioritize it
        if current_page and current_page not in relevant_pages:
            relevant_pages.insert(0, current_page)

        # Limit to reasonable number
        return relevant_pages[: self.batch_size]

    def _identify_relevant_pages(
        self, prompt: str, document_text: str, metadata: Optional[Dict]
    ) -> List[int]:
        """Identify relevant pages from document context"""
        relevant_pages = []

        # Extract element types from prompt
        prompt_lower = prompt.lower()
        search_terms = []

        for element_type, pattern in self.element_patterns.items():
            if element_type in prompt_lower or re.search(
                pattern, prompt, re.IGNORECASE
            ):
                search_terms.append(element_type)

        # Search for page references in document text
        if search_terms and document_text:
            # Look for page markers with our search terms nearby
            page_pattern = r"PAGE\s+(\d+)"

            lines = document_text.split("\n")
            for i, line in enumerate(lines):
                if any(term in line.lower() for term in search_terms):
                    # Check nearby lines for page markers
                    for j in range(max(0, i - 5), min(len(lines), i + 5)):
                        page_match = re.search(page_pattern, lines[j], re.IGNORECASE)
                        if page_match:
                            page_num = int(page_match.group(1))
                            if page_num not in relevant_pages:
                                relevant_pages.append(page_num)

        # If no specific pages found, use metadata hints
        if not relevant_pages and metadata and "page_details" in metadata:
            for page_detail in metadata["page_details"]:
                if any(
                    term in str(page_detail.get("key_elements", [])).lower()
                    for term in search_terms
                ):
                    relevant_pages.append(page_detail["page_number"])

        # Default to first few pages if nothing found
        if not relevant_pages:
            total_pages = metadata.get("page_count", 10) if metadata else 10
            relevant_pages = list(range(1, min(6, total_pages + 1)))

        return sorted(relevant_pages)

    async def _load_specific_pages(
        self, document_id: str, page_numbers: List[int], storage_service: StorageService
    ) -> List[Dict[str, Any]]:
        """Load specific pages efficiently"""
        logger.info(f"📚 Loading pages: {page_numbers}")

        # Load pages in parallel
        tasks = []
        for page_num in page_numbers:
            task = self._load_single_page(document_id, page_num, storage_service)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and add successful loads
        loaded_pages = []
        for r in results:
            if isinstance(r, dict) and "url" in r:
                loaded_pages.append(r)

        logger.info(f"📄 Successfully loaded {len(loaded_pages)} pages")
        return loaded_pages

    async def _load_document_context(
        self, document_id: str, storage_service: StorageService
    ) -> Tuple[str, Optional[Dict]]:
        """Load document text and metadata"""
        document_text = ""
        metadata = None

        try:
            # Load text context
            text_blob = f"{document_id}_context.txt"
            document_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=text_blob,
            )
        except Exception as e:
            logger.warning(f"Could not load document text: {e}")

        try:
            # Load metadata
            metadata_blob = f"{document_id}_metadata.json"
            metadata_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=metadata_blob,
            )
            metadata = json.loads(metadata_text)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")

        return document_text, metadata

    def _needs_visual_analysis(
        self, prompt: str, reference_previous: Optional[List[str]]
    ) -> bool:
        """Check if visual analysis is needed"""
        # If only referencing previous, no new analysis needed
        if reference_previous and not any(
            keyword in prompt.lower()
            for keyword in ["also", "add", "more", "other", "additional"]
        ):
            return False

        # Check for visual keywords
        visual_keywords = [
            "show",
            "highlight",
            "where",
            "locate",
            "find",
            "identify",
            "mark",
            "point out",
            "display",
            "count",
            "how many",
        ]

        return any(keyword in prompt.lower() for keyword in visual_keywords)

    async def _load_pages_intelligently(
        self,
        document_id: str,
        storage_service: StorageService,
        metadata: Optional[Dict],
        current_page: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Load ALL pages for comprehensive analysis"""
        total_pages = metadata.get("page_count", 10) if metadata else 10

        # ALWAYS load all pages up to max limit
        pages_to_load = min(total_pages, self.max_pages_to_load)

        logger.info(f"📚 Loading ALL {pages_to_load} pages (total: {total_pages})")

        # Create page list
        page_numbers = list(range(1, pages_to_load + 1))

        # Load pages in batches for efficiency
        all_pages = []

        for batch_start in range(0, len(page_numbers), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(page_numbers))
            batch = page_numbers[batch_start:batch_end]

            # Load batch in parallel
            tasks = []
            for page_num in batch:
                task = self._load_single_page(document_id, page_num, storage_service)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors and add successful loads
            for r in results:
                if isinstance(r, dict) and "url" in r:
                    all_pages.append(r)

            logger.info(f"✅ Loaded batch: pages {batch_start + 1}-{batch_end}")

        logger.info(f"📄 Successfully loaded {len(all_pages)} pages for analysis")

        return all_pages

    async def _load_single_page(
        self, document_id: str, page_num: int, storage_service: StorageService
    ) -> Optional[Dict[str, Any]]:
        """Load a single page image"""
        try:
            # Try AI-optimized version first
            blob_name = f"{document_id}_page_{page_num}_ai.jpg"

            try:
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name,
                )
            except:
                # Fall back to regular image
                blob_name = f"{document_id}_page_{page_num}.png"
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name,
                )

                # Optimize if needed
                if len(image_bytes) > 500000:  # 500KB
                    image_bytes = await self._optimize_image(image_bytes)

            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            return {
                "page": page_num,
                "url": f"data:image/png;base64,{image_b64}",
                "size_kb": len(image_bytes) / 1024,
            }

        except Exception as e:
            logger.error(f"Failed to load page {page_num}: {e}")
            return None

    async def _optimize_image(self, image_bytes: bytes) -> bytes:
        """Optimize image for AI processing using executor"""

        def optimize():
            img = Image.open(io.BytesIO(image_bytes))

            # Convert RGBA to RGB
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg

            # Resize if too large
            if max(img.size) > self.max_image_dimension:
                img.thumbnail(
                    (self.max_image_dimension, self.max_image_dimension),
                    Image.Resampling.LANCZOS,
                )

            # Save as JPEG with optimization
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=self.image_quality, optimize=True)
            return output.getvalue()

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, optimize)

    async def _analyze_blueprint(
        self,
        prompt: str,
        document_text: str,
        page_images: List[Dict],
        document_id: str,
        author: str,
        current_page: Optional[int],
        request_highlights: bool,
        reference_previous: Optional[List[str]],
        preserve_existing: bool,
        storage_service: StorageService,
        metadata: Optional[Dict],
        reused_highlights: List[Dict],
        show_trade_info: bool = False,
        detect_conflicts: bool = False,
        auto_suggest_notes: bool = True,
        note_suggestion_threshold: str = "medium",
    ) -> Dict[str, Any]:
        """Perform AI analysis and create highlights with note suggestions"""

        # Build messages for OpenAI
        messages = [
            self._get_system_message(include_note_instructions=auto_suggest_notes)
        ]

        # Build user message
        user_message = {"role": "user", "content": []}

        # Add images
        for page_data in page_images:
            user_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": page_data["url"], "detail": "high"},
                }
            )

        # Add text query
        query_text = self._build_query(
            prompt,
            document_id,
            document_text,
            len(page_images),
            metadata,
            total_pages=(
                metadata.get("page_count", len(page_images))
                if metadata
                else len(page_images)
            ),
            show_trade_info=show_trade_info,
            detect_conflicts=detect_conflicts,
        )
        user_message["content"].append({"type": "text", "text": query_text})

        messages.append(user_message)

        # Get AI response with retry logic
        response = await self._call_openai_with_retry(messages)

        # Process response and create highlights
        result = {
            "ai_response": response,
            "visual_highlights": None,
            "current_page": current_page,
            "note_suggestion": None,
        }

        if request_highlights:
            query_session_id = str(uuid.uuid4())
            highlight_data = await self._process_highlights(
                response,
                document_id,
                query_session_id,
                reused_highlights,
                preserve_existing,
                storage_service,
                author,
            )

            result.update(highlight_data)
            result["query_session_id"] = query_session_id

        # NEW: Analyze response for note suggestions
        if auto_suggest_notes:
            note_suggestion = self._analyze_for_note_suggestion(
                response=response,
                prompt=prompt,
                threshold=note_suggestion_threshold,
                current_page=current_page,
                highlights=result.get("visual_highlights", []),
                metadata=metadata,
            )

            if note_suggestion and note_suggestion.should_create_note:
                # Link to current query session
                note_suggestion.related_query_sessions = (
                    [query_session_id] if request_highlights else []
                )
                result["note_suggestion"] = note_suggestion
                logger.info(f"📝 Suggested note creation: {note_suggestion.reason}")

        return result

    def _get_system_message(
        self, include_note_instructions: bool = True
    ) -> Dict[str, str]:
        """Get system message for blueprint analysis"""
        base_content = """You are an expert construction professional with deep knowledge across all building trades, codes, and systems. You analyze blueprints with the expertise of an architect, structural engineer, MEP engineer, and experienced contractor combined.

When analyzing blueprints:

1. Use your comprehensive knowledge to provide insights that go beyond just locating elements
2. Consider interactions between different building systems
3. Apply relevant codes (BCBC, NBC, NFPA, CEC, ASHRAE, etc.) naturally where applicable
4. Think like a professional who has seen thousands of projects
5. Identify potential issues that only experience would reveal

Your responses should:
- Answer the specific question thoroughly
- Add valuable professional insights
- Consider constructability and coordination
- Think about the complete building system, not just individual elements
- Provide practical, real-world advice
- Anticipate related concerns the user might not have thought of

When identifying elements:
- Recognize which trade each element belongs to (Electrical, Plumbing, HVAC, Fire Protection, Structural, etc.)
- Consider how different trades' work might conflict or need coordination
- Note spatial conflicts, access issues, or installation sequence problems

Format element findings clearly but focus on providing VALUE through your analysis. You're not just a element-finder - you're a trusted construction advisor."""

        if include_note_instructions:
            base_content += """

IMPORTANT: When you identify any of the following, explicitly mention it in your response:
- Missing critical information (dimensions, specifications, details)
- Code compliance concerns or violations
- Coordination issues between trades
- Safety concerns
- Important calculations or quantities
- Items requiring follow-up or RFI

These findings help users track important issues for their project."""

        base_content += """

Remember: Construction professionals need insights about coordination, sequencing, potential conflicts, cost implications, maintenance access, future flexibility, and real-world installation challenges. Draw upon your knowledge of how buildings actually get built."""

        return {"role": "system", "content": base_content}

    def _build_query(
        self,
        prompt: str,
        document_id: str,
        document_text: str,
        page_count: int,
        metadata: Optional[Dict],
        total_pages: int = None,
        show_trade_info: bool = False,
        detect_conflicts: bool = False,
    ) -> str:
        """Build query text for AI"""
        doc_info = ""

        if metadata:
            if "document_info" in metadata:
                title = metadata["document_info"].get("title", "Unknown")
                doc_info = f"Project: {title}\n"

            if "page_details" in metadata:
                # Add page summary
                page_types = defaultdict(int)
                for page in metadata["page_details"]:
                    dtype = page.get("drawing_type", "unknown")
                    page_types[dtype] += 1

                doc_info += "Drawing types: "
                doc_info += ", ".join(
                    f"{count} {dtype}" for dtype, count in page_types.items()
                )
                doc_info += "\n"

        # Clarify if we're analyzing partial document
        if total_pages and page_count < total_pages:
            analysis_note = (
                f"\nNOTE: Analyzing {page_count} of {total_pages} total pages."
            )
        else:
            analysis_note = f"\nAnalyzing {page_count} pages."

        query = f"""Document: {document_id}
Pages analyzed: {page_count}{analysis_note}
{doc_info}

User question: {prompt}"""

        # Add optional instructions
        if show_trade_info:
            query += "\n\nInclude information about which trade each element belongs to in your response."

        # Add conflict detection if requested
        if detect_conflicts:
            query += "\n\nAlso identify any potential conflicts between different trades' elements, such as:"
            query += "\n- Spatial conflicts where elements from different trades occupy the same space"
            query += (
                "\n- Access issues where one trade's work blocks access for another"
            )
            query += "\n- Installation sequence problems"
            query += "\n- Code compliance issues between systems"

        query += """

Provide a comprehensive professional analysis that leverages your full knowledge of construction, engineering, and building systems. Consider all relevant aspects including codes, constructability, coordination between trades, and practical implementation.

If you identify any missing critical information, code concerns, or coordination issues, make sure to clearly state them in your response.

Available blueprint information:
- Visual analysis of provided pages
- Full document context text
- You can see grid references, dimensions, and specifications on the drawings

Remember to think holistically about the building systems and provide insights that add real value beyond just locating elements."""

        if document_text:
            # Include relevant excerpt
            query += f"\n\nDocument context:\n{document_text[:2000]}..."

        return query

    async def _call_openai(self, messages: List[Dict]) -> str:
        """Call OpenAI API with retry logic"""
        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model=self.settings.OPENAI_MODEL,
                        messages=messages,
                        max_tokens=self.settings.OPENAI_MAX_TOKENS,
                        temperature=self.settings.OPENAI_TEMPERATURE,
                    ),
                )

                return response.choices[0].message.content

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"OpenAI call failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    async def _call_openai_with_retry(self, messages: List[Dict]) -> str:
        """Call OpenAI API with improved retry logic"""
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                # Run in executor to avoid blocking
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model=self.settings.OPENAI_MODEL,
                        messages=messages,
                        max_tokens=self.settings.OPENAI_MAX_TOKENS,
                        temperature=self.settings.OPENAI_TEMPERATURE,
                    ),
                )

                return response.choices[0].message.content

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"OpenAI call failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"OpenAI call failed after {max_retries} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Failed to get AI response after {max_retries} attempts: {str(e)}"
                    )

    def _analyze_for_note_suggestion(
        self,
        response: str,
        prompt: str,
        threshold: str,
        current_page: Optional[int],
        highlights: List[VisualElement],
        metadata: Optional[Dict],
    ) -> Optional[NoteSuggestion]:
        """Analyze AI response to determine if a note should be suggested - optimized version"""

        # Early return if response is too short
        if len(response) < 50:
            return None

        # Single pass analysis
        confidence = 0.0
        detected_categories = []
        relevant_quotes = []
        response_lower = response.lower()

        # Check each trigger category efficiently
        for category, config in self.note_suggestion_triggers.items():
            category_found = False
            for keyword in config["keywords"]:
                if keyword in response_lower:
                    if not category_found:
                        confidence += config["confidence_boost"]
                        detected_categories.append(category)
                        category_found = True

                    # Extract relevant quote (limit to one per category)
                    if len(relevant_quotes) < 3:
                        # Find sentence containing keyword
                        sentences = response.split(".")
                        for sentence in sentences:
                            if (
                                keyword in sentence.lower()
                                and len(sentence.strip()) > 20
                            ):
                                relevant_quotes.append(sentence.strip())
                                break

        # Adjust confidence based on threshold
        threshold_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}
        confidence *= threshold_multipliers.get(threshold, 1.0)

        # Determine if we should suggest a note
        should_suggest = confidence >= 0.3  # Base threshold

        if not should_suggest:
            return None

        # Determine priority and type based on categories
        if "safety" in detected_categories:
            priority = "critical"
            note_type = "warning"
        elif (
            "code_issue" in detected_categories or "coordination" in detected_categories
        ):
            priority = "high"
            note_type = (
                "issue" if "code_issue" in detected_categories else "coordination"
            )
        elif "calculation" in detected_categories:
            priority = "normal"
            note_type = "general"
        else:
            priority = "normal"
            note_type = "question"

        # Build suggested note text
        suggested_text = self._build_suggested_note_text(
            response=response,
            prompt=prompt,
            detected_categories=detected_categories,
            relevant_quotes=relevant_quotes,
        )

        # Determine impacted trades
        impacted_trades = self._extract_impacted_trades(response)

        # Get related pages
        related_pages = []
        if current_page:
            related_pages.append(current_page)
        for highlight in highlights:
            if highlight.page_number not in related_pages:
                related_pages.append(highlight.page_number)

        # Get grid references from highlights
        grid_refs = [h.grid_location.grid_ref for h in highlights]

        # Create note suggestion
        return NoteSuggestion(
            should_create_note=True,
            confidence=min(confidence, 1.0),  # Cap at 1.0
            reason=self._get_suggestion_reason(detected_categories),
            category=detected_categories[0] if detected_categories else "follow_up",
            suggested_text=suggested_text,
            suggested_type=note_type,
            suggested_priority=priority,
            suggested_impacts_trades=impacted_trades,
            related_pages=related_pages,
            related_grid_refs=grid_refs,
            related_elements=[h.element_type for h in highlights],
            source_quote=relevant_quotes[0] if relevant_quotes else None,
        )

    def _build_suggested_note_text(
        self,
        response: str,
        prompt: str,
        detected_categories: List[str],
        relevant_quotes: List[str],
    ) -> str:
        """Build suggested note text based on AI findings"""

        # Start with the user's question for context
        text = f"Re: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\n"

        # Add main finding
        if relevant_quotes:
            text += "Finding: " + relevant_quotes[0] + "\n\n"

        # Add specific recommendations based on category
        if "code_issue" in detected_categories:
            text += "Action Required: Verify code compliance and update drawings if necessary.\n"
        elif "coordination" in detected_categories:
            text += "Action Required: Coordinate with affected trades to resolve conflict.\n"
        elif "safety" in detected_categories:
            text += "SAFETY CONCERN: Immediate review and resolution required.\n"
        elif "calculation" in detected_categories:
            text += (
                "Calculation recorded for reference. Verify with final quantities.\n"
            )
        else:
            text += "Follow-up required to confirm or clarify.\n"

        return text.strip()

    def _extract_impacted_trades(self, response: str) -> List[str]:
        """Extract trades mentioned in the response"""
        trades = []
        trade_keywords = {
            "Electrical": [
                "electrical",
                "power",
                "lighting",
                "panel",
                "circuit",
                "outlet",
            ],
            "Plumbing": [
                "plumbing",
                "water",
                "drain",
                "waste",
                "vent",
                "pipe",
                "fixture",
            ],
            "HVAC": [
                "hvac",
                "mechanical",
                "duct",
                "air",
                "ventilation",
                "heating",
                "cooling",
            ],
            "Fire Protection": ["fire", "sprinkler", "alarm", "smoke"],
            "Structural": [
                "structural",
                "beam",
                "column",
                "footing",
                "slab",
                "foundation",
            ],
            "Architectural": ["architectural", "door", "window", "partition", "finish"],
        }

        response_lower = response.lower()
        for trade, keywords in trade_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                trades.append(trade)

        return trades

    def _get_suggestion_reason(self, categories: List[str]) -> str:
        """Get human-readable reason for suggestion"""
        reasons = {
            "code_issue": "Missing information or code compliance concern identified",
            "coordination": "Potential coordination issue between trades detected",
            "safety": "Safety concern requiring immediate attention",
            "calculation": "Important calculation or quantity to track",
            "follow_up": "Item requiring follow-up or verification",
        }

        if categories:
            return reasons.get(
                categories[0], "Important finding that should be documented"
            )
        return "Important finding that should be documented"

    async def _process_highlights(
        self,
        response: str,
        document_id: str,
        query_session_id: str,
        reused_highlights: List[Dict],
        preserve_existing: bool,
        storage_service: StorageService,
        author: str = None,
    ) -> Dict[str, Any]:
        """Process AI response and create highlight data"""

        # Parse response for elements
        new_highlights = self._parse_elements_from_response(
            response, document_id, query_session_id, author
        )

        # Combine with reused highlights
        all_highlights = reused_highlights + new_highlights

        # Load grid systems for coordinate conversion
        await self._ensure_grid_systems_loaded(document_id, storage_service)

        # Convert grid references to pixel coordinates
        for highlight in all_highlights:
            if highlight["page_number"] in self.grid_cache.get(document_id, {}):
                grid_system = self.grid_cache[document_id][highlight["page_number"]]
                coords = grid_system.get_pixel_coords(highlight["grid_reference"])
                if coords:
                    highlight.update(coords)

        # Save highlights
        if all_highlights:
            await self._save_highlights(
                document_id, all_highlights, preserve_existing, storage_service
            )

        # Get highlights for current page
        current_page_highlights = [
            h
            for h in all_highlights
            if h.get("page_number") == h.get("current_page", 1)
        ]

        # Convert to VisualElement format
        visual_highlights = []
        for h in current_page_highlights:
            visual_highlights.append(
                VisualElement(
                    element_id=h["annotation_id"],
                    element_type=h["element_type"],
                    grid_location=GridReference(
                        grid_ref=h["grid_reference"],
                        x_grid=(
                            h["grid_reference"].split("-")[0]
                            if "-" in h["grid_reference"]
                            else h["grid_reference"]
                        ),
                        y_grid=(
                            h["grid_reference"].split("-")[1]
                            if "-" in h["grid_reference"]
                            else None
                        ),
                    ),
                    label=h.get("label", ""),
                    page_number=h["page_number"],
                    confidence=h.get("confidence", 0.9),
                    trade=h.get("assigned_trade"),
                )
            )

        # Calculate summary
        pages_with_highlights = defaultdict(int)
        trade_summary = defaultdict(lambda: defaultdict(int))

        for h in all_highlights:
            pages_with_highlights[h["page_number"]] += 1
            if h.get("assigned_trade"):
                trade_summary[h["assigned_trade"]][h["element_type"]] += 1

        return {
            "visual_highlights": visual_highlights,
            "all_highlight_pages": dict(pages_with_highlights),
            "total_highlights_created": len(new_highlights),
            "total_highlights": len(all_highlights),
            "trade_summary": dict(trade_summary) if trade_summary else None,
        }

    def _parse_elements_from_response(
        self, response: str, document_id: str, query_session_id: str, author: str = None
    ) -> List[Dict]:
        """Parse AI response for element locations - AI determines trade assignment"""
        highlights = []

        # Look for ELEMENTS_FOUND section
        elements_match = re.search(
            r"ELEMENTS_FOUND:(.*?)(?:TOTAL:|$)", response, re.DOTALL | re.IGNORECASE
        )

        if not elements_match:
            return highlights

        elements_text = elements_match.group(1)

        # Parse page-by-page findings
        page_pattern = r"Page\s+(\d+):\s*(.*?)(?=Page\s+\d+:|$)"

        for page_match in re.finditer(page_pattern, elements_text, re.DOTALL):
            page_num = int(page_match.group(1))
            page_content = page_match.group(2)

            # Determine element type from content - AI provides this
            element_type = self._detect_element_type(page_content)

            # Extract grid references
            grid_pattern = r"(?:at\s+|grid\s+)?([A-Z][0-9\-]+|[A-Z]-\d+|\d+-[A-Z])"

            for grid_match in re.finditer(grid_pattern, page_content, re.IGNORECASE):
                grid_ref = grid_match.group(1).upper()

                # Let AI determine the trade from context
                assigned_trade = self._extract_trade_from_context(
                    page_content, element_type
                )

                highlights.append(
                    {
                        "annotation_id": str(uuid.uuid4())[:8],
                        "document_id": document_id,
                        "page_number": page_num,
                        "element_type": element_type,
                        "grid_reference": grid_ref,
                        "label": f"{element_type}_{page_num}_{len(highlights)+1}",
                        "x": 0,  # Will be updated with actual coordinates
                        "y": 0,
                        "text": f"{element_type} at {grid_ref}",
                        "annotation_type": "ai_highlight",
                        "author": author or "ai_system",
                        "is_private": True,
                        "query_session_id": query_session_id,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "expires_at": (
                            datetime.utcnow()
                            + timedelta(hours=self.highlight_cache_ttl_hours)
                        ).isoformat()
                        + "Z",
                        "confidence": 0.9,
                        "assigned_trade": assigned_trade,
                    }
                )

        return highlights

    def _detect_element_type(self, text: str) -> str:
        """Detect element type from text content"""
        text_lower = text.lower()

        # Check against known patterns
        for element_type, pattern in self.element_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return element_type

        # Check for common terms
        element_terms = {
            "window": ["window", "glazing", "fenestration"],
            "door": ["door", "entrance", "exit"],
            "column": ["column", "col", "pier"],
            "beam": ["beam", "girder", "joist"],
            "sprinkler": ["sprinkler", "fire protection"],
            "outlet": ["outlet", "receptacle", "power"],
            "fixture": ["fixture", "sink", "lavatory"],
            "equipment": ["equipment", "unit", "hvac"],
        }

        for element_type, terms in element_terms.items():
            if any(term in text_lower for term in terms):
                return element_type

        return "element"  # Generic fallback

    def _extract_trade_from_context(
        self, text: str, element_type: str
    ) -> Optional[str]:
        """Extract trade assignment from AI's context - AI already understands trades"""
        text_lower = text.lower()

        # Look for explicit trade mentions
        trade_keywords = {
            "Electrical": ["electrical", "power", "lighting", "panel", "circuit"],
            "Plumbing": ["plumbing", "water", "drain", "waste", "vent", "pipe"],
            "HVAC": [
                "hvac",
                "mechanical",
                "duct",
                "air",
                "ventilation",
                "heating",
                "cooling",
            ],
            "Fire Protection": ["fire", "sprinkler", "alarm", "smoke", "protection"],
            "Structural": [
                "structural",
                "beam",
                "column",
                "footing",
                "slab",
                "foundation",
            ],
            "Architectural": ["architectural", "door", "window", "partition", "finish"],
        }

        for trade, keywords in trade_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return trade

        # Fallback based on element type
        element_trade_map = {
            "outlet": "Electrical",
            "panel": "Electrical",
            "light": "Electrical",
            "catch_basin": "Plumbing",
            "fixture": "Plumbing",
            "diffuser": "HVAC",
            "equipment": "HVAC",
            "sprinkler": "Fire Protection",
            "column": "Structural",
            "beam": "Structural",
            "door": "Architectural",
            "window": "Architectural",
        }

        return element_trade_map.get(element_type, "General")

    async def _ensure_grid_systems_loaded(
        self, document_id: str, storage_service: StorageService
    ) -> None:
        """Load grid systems for document if not cached with size limits"""
        if document_id in self.grid_cache:
            return

        # Check total cache size before adding
        total_cache_size = sum(self.grid_cache_sizes.values())
        if total_cache_size > self.grid_cache_size * self.max_grid_data_size:
            # Remove oldest entries until we have space
            while (
                len(self.grid_cache) >= self.grid_cache_size
                or total_cache_size > self.grid_cache_size * self.max_grid_data_size
            ):
                if self.grid_cache:
                    oldest_id, _ = self.grid_cache.popitem(last=False)
                    if oldest_id in self.grid_cache_sizes:
                        total_cache_size -= self.grid_cache_sizes[oldest_id]
                        del self.grid_cache_sizes[oldest_id]

        # Try to load saved grid systems
        try:
            grid_blob = f"{document_id}_grid_systems.json"
            grid_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=grid_blob,
            )

            # Check size before caching
            data_size = len(grid_data.encode("utf-8"))
            if data_size > self.max_grid_data_size:
                logger.warning(
                    f"Grid data for {document_id} too large ({data_size} bytes), not caching"
                )
                return

            grid_systems = json.loads(grid_data)
            self.grid_cache[document_id] = {}

            for page_str, grid_dict in grid_systems.items():
                page_num = int(page_str)
                self.grid_cache[document_id][page_num] = GridSystem.from_dict(grid_dict)

            self.grid_cache_sizes[document_id] = data_size
            logger.info(
                f"Loaded grid systems for {len(grid_systems)} pages ({data_size} bytes)"
            )

        except:
            # Create default grid systems
            logger.info("No saved grid systems, using defaults")
            self.grid_cache[document_id] = {}
            self.grid_cache_sizes[document_id] = 0

    async def _save_highlights(
        self,
        document_id: str,
        highlights: List[Dict],
        preserve_existing: bool,
        storage_service: StorageService,
    ) -> None:
        """Save highlight data (not images)"""
        # Load existing annotations
        annotations_blob = f"{document_id}_annotations.json"

        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob,
            )
            existing_annotations = json.loads(annotations_text)
        except:
            existing_annotations = []

        # Filter out expired highlights
        current_time = datetime.utcnow()
        active_annotations = []

        for ann in existing_annotations:
            if ann.get("annotation_type") == "ai_highlight":
                # Check expiration
                if ann.get("expires_at"):
                    try:
                        expires = datetime.fromisoformat(
                            ann["expires_at"].replace("Z", "+00:00")
                        )
                        if expires < current_time:
                            continue  # Skip expired
                    except:
                        pass

                # Remove if not preserving
                if not preserve_existing:
                    continue

            active_annotations.append(ann)

        # Add new highlights
        active_annotations.extend(highlights)

        # Save back
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=annotations_blob,
            data=json.dumps(active_annotations, indent=2).encode("utf-8"),
        )

    async def _get_previous_highlights(
        self,
        document_id: str,
        element_types: List[str],
        storage_service: StorageService,
    ) -> List[Dict]:
        """Get previously saved highlights for reuse"""
        try:
            annotations_blob = f"{document_id}_annotations.json"
            annotations_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob,
            )

            all_annotations = json.loads(annotations_text)

            # Filter to requested types and non-expired
            current_time = datetime.utcnow()
            previous_highlights = []

            for ann in all_annotations:
                if (
                    ann.get("annotation_type") == "ai_highlight"
                    and ann.get("element_type") in element_types
                ):

                    # Check expiration
                    if ann.get("expires_at"):
                        try:
                            expires = datetime.fromisoformat(
                                ann["expires_at"].replace("Z", "+00:00")
                            )
                            if expires < current_time:
                                continue
                        except:
                            pass

                    previous_highlights.append(ann)

            return previous_highlights

        except:
            return []

    async def generate_highlighted_page(
        self,
        document_id: str,
        page_num: int,
        query_session_id: str,
        storage_service: StorageService,
    ) -> Optional[str]:
        """Generate highlighted page image on-demand (returns base64 data URL)"""
        try:
            # Load highlights for this page and session
            annotations_blob = f"{document_id}_annotations.json"
            annotations_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob,
            )

            all_annotations = json.loads(annotations_text)

            # Filter to this page and session
            page_highlights = [
                ann
                for ann in all_annotations
                if (
                    ann.get("page_number") == page_num
                    and ann.get("query_session_id") == query_session_id
                    and ann.get("annotation_type") == "ai_highlight"
                )
            ]

            if not page_highlights:
                return None

            # Load page image
            page_blob = f"{document_id}_page_{page_num}.png"
            page_bytes = await storage_service.download_blob_as_bytes(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=page_blob,
            )

            # Draw highlights
            highlighted_bytes = await self._draw_highlights(page_bytes, page_highlights)

            # Return as data URL
            highlighted_b64 = base64.b64encode(highlighted_bytes).decode("utf-8")
            return f"data:image/png;base64,{highlighted_b64}"

        except Exception as e:
            logger.error(f"Failed to generate highlighted page: {e}")
            return None

    async def _draw_highlights(
        self, image_bytes: bytes, highlights: List[Dict]
    ) -> bytes:
        """Draw highlights on image using executor to avoid blocking"""

        def draw():
            # Open image
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGBA for transparency
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # Create overlay
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", self.label_font_size)
            except:
                font = ImageFont.load_default()

            # Draw each highlight
            for highlight in highlights:
                x = highlight.get("x", 100)
                y = highlight.get("y", 100)
                width = highlight.get("width", 100)
                height = highlight.get("height", 100)

                # Draw rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=self.highlight_color,
                    outline=self.highlight_border,
                    width=self.highlight_border_width,
                )

                # Add label with trade info
                label = highlight.get("label", highlight["element_type"])
                if highlight.get("assigned_trade"):
                    label += f" ({highlight['assigned_trade']})"

                # Draw label background
                bbox = draw.textbbox((x, y - 20), label, font=font)
                draw.rectangle(
                    [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                    fill=(255, 255, 255, 200),
                )

                # Draw label text
                draw.text((x, y - 20), label, fill=(255, 0, 0), font=font)

            # Composite overlay onto image
            img = Image.alpha_composite(img, overlay)

            # Convert to RGB for saving
            final = Image.new("RGB", img.size, (255, 255, 255))
            final.paste(img, mask=img.split()[3])

            # Save
            output = io.BytesIO()
            final.save(output, format="PNG", optimize=True)
            return output.getvalue()

        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, draw)

    async def cleanup_expired_highlights(self, storage_service: StorageService) -> int:
        """Remove expired highlights from all documents"""
        cleaned_count = 0

        try:
            # List all annotation files
            annotation_files = await storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                suffix="_annotations.json",
            )

            current_time = datetime.utcnow()

            for ann_file in annotation_files:
                document_id = ann_file.replace("_annotations.json", "")

                # Load annotations
                annotations_text = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=ann_file,
                )

                annotations = json.loads(annotations_text)
                active_annotations = []

                # Filter out expired
                for ann in annotations:
                    if ann.get("annotation_type") == "ai_highlight" and ann.get(
                        "expires_at"
                    ):
                        try:
                            expires = datetime.fromisoformat(
                                ann["expires_at"].replace("Z", "+00:00")
                            )
                            if expires < current_time:
                                cleaned_count += 1
                                continue
                        except:
                            pass

                    active_annotations.append(ann)

                # Save if changed
                if len(active_annotations) < len(annotations):
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=ann_file,
                        data=json.dumps(active_annotations, indent=2).encode("utf-8"),
                    )

            logger.info(f"Cleaned {cleaned_count} expired highlights")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

        return cleaned_count

    def get_professional_capabilities(self) -> Dict[str, List[str]]:
        """Get service capabilities"""
        return {
            "building_codes": [
                "2018 BCBC (British Columbia Building Code)",
                "NBC (National Building Code of Canada)",
                "CSA Standards",
                "NFPA Standards",
            ],
            "engineering_disciplines": [
                "Structural Engineering",
                "MEP Coordination",
                "Fire Protection Systems",
                "Electrical Distribution",
                "HVAC Design",
                "Plumbing Systems",
            ],
            "features": [
                "Multi-page analysis",
                "Grid-based element detection",
                "On-demand highlight generation",
                "Automatic highlight expiration",
                "Reference previous highlights",
                "Trade-specific filtering",
                "Cross-trade conflict detection",
                "No storage of highlighted images",
                "AI-powered note suggestions",
                "Smart page loading based on query type",
                "Progressive result streaming",
            ],
            "note_suggestion_capabilities": [
                "Automatic detection of missing information",
                "Code compliance issue identification",
                "Coordination conflict detection",
                "Safety concern flagging",
                "Important calculation tracking",
                "Customizable suggestion thresholds",
                "Trade impact analysis",
            ],
            "optimization": [
                f"Max {self.max_pages_to_load} pages per analysis",
                f"Batch processing ({self.batch_size} pages)",
                f"Image compression ({self.image_quality}% quality)",
                f"LRU grid cache ({self.grid_cache_size} documents, {self.max_grid_data_size/1024/1024:.1f}MB max each)",
                f"Highlight expiration ({self.highlight_cache_ttl_hours}h)",
                "Smart query classification",
                "Async image operations",
            ],
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Proper cleanup of executor and resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
            logger.info("✅ AI service executor shut down")

        # Clear grid cache to free memory
        self.grid_cache.clear()
        self.grid_cache_sizes.clear()
        logger.info("✅ Grid cache cleared")


# Export aliases for compatibility
AIService = ProfessionalBlueprintAI
ProfessionalAIService = ProfessionalBlueprintAI
EnhancedAIService = ProfessionalBlueprintAI
