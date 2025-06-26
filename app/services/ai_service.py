# app/services/ai_service.py - PRODUCTION-READY WITH STORAGE OPTIMIZATION

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
    VisualElement, GridReference, DrawingGrid
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
            (r'^([A-Z]+)[-\s]?(\d+)$', 'letter-number'),  # A-1, B2
            (r'^(\d+)[-\s]?([A-Z]+)$', 'number-letter'),  # 1-A, 2B
            (r'^([A-Z]+\d*)[-\s]?([A-Z]+\d*)$', 'complex'),  # W2-WA
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
                if x is None and pattern_type != 'complex':
                    x = self.x_coordinates.get(y_ref)
                    y = self.y_coordinates.get(x_ref)
                
                # If still not found, estimate
                if x is None:
                    x = self._estimate_coordinate(x_ref, 'x')
                if y is None:
                    y = self._estimate_coordinate(y_ref, 'y')
                
                if x is not None and y is not None:
                    return {
                        "x": int(x),
                        "y": int(y),
                        "width": self.cell_width,
                        "height": self.cell_height,
                        "center_x": int(x + self.cell_width // 2),
                        "center_y": int(y + self.cell_height // 2)
                    }
        
        # If no pattern matches, return estimated center position
        return self._get_fallback_position()
    
    def _estimate_coordinate(self, ref: str, axis: str) -> Optional[int]:
        """Estimate coordinate based on reference"""
        if axis == 'x':
            if ref.isalpha() and len(ref) == 1:
                # Simple letter: A=0, B=1, etc.
                index = ord(ref) - ord('A')
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
                index = ord(ref) - ord('A')
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
            "center_y": self.origin_y + (5 * self.cell_height) + self.cell_height // 2
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
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridSystem':
        """Create from dictionary"""
        return cls(
            page_number=data.get('page_number', 1),
            x_labels=data.get('x_labels', []),
            y_labels=data.get('y_labels', []),
            x_coordinates=data.get('x_coordinates', {}),
            y_coordinates=data.get('y_coordinates', {}),
            cell_width=data.get('cell_width', 100),
            cell_height=data.get('cell_height', 100),
            origin_x=data.get('origin_x', 100),
            origin_y=data.get('origin_y', 100),
            scale=data.get('scale'),
            confidence=data.get('confidence', 0.0)
        )


class ProfessionalBlueprintAI:
    """Professional AI service with storage-optimized visual highlighting"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Professional Blueprint AI initialized")
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
        self.highlight_cache_ttl_hours = 24   # Expire highlight data after 24 hours
        self.grid_cache_size = 100  # Max documents to cache grids for
        
        # Visual highlight settings
        self.highlight_color = (255, 255, 0, 80)  # Yellow with transparency
        self.highlight_border = (255, 0, 0)       # Red border
        self.highlight_border_width = 3
        self.label_font_size = 14
        
        # Grid cache with LRU eviction
        self.grid_cache = OrderedDict()
        
        # Patterns for element detection
        self.element_patterns = {
            'window': r'(?:TYPE|type)\s*(\d+)|W\d+|WINDOW|window',
            'door': r'(?:TYPE|type)\s*([A-Z]\d*)|D\d+|DOOR|door',
            'column': r'C\d+|COL\.?\s*\d+|COLUMN',
            'beam': r'B\d+|BM\.?\s*\d+|BEAM',
            'catch_basin': r'CB[-\s]?\d+|CATCH\s*BASIN',
            'sprinkler': r'SP[-\s]?\d+|SPRINKLER',
            'outlet': r'OUTLET|RECEPTACLE|DUPLEX',
            'panel': r'PANEL\s*[A-Z]\d*|EP[-\s]?\d+|MDP',
            'equipment': r'UNIT|AHU|RTU|VAV|FCU',
            'fixture': r'WC|LAV|SINK|FIXTURE'
        }
        
        # Building code requirements for reference
        self.code_requirements = {
            'sprinkler': {
                'NFPA_13': {
                    'light_hazard': {'max_spacing': 15, 'max_coverage': 225},
                    'ordinary_hazard': {'max_spacing': 15, 'max_coverage': 130},
                    'extra_hazard': {'max_spacing': 12, 'max_coverage': 100}
                },
                'BCBC_2018': {
                    'section': '3.2.5.12',
                    'min_from_wall': 4,  # inches
                    'max_from_wall': 72  # inches (6 feet)
                }
            },
            'outlet': {
                'CEC': {
                    'residential': {'max_spacing': 12},  # feet
                    'commercial': {'workstation_spacing': 6}  # feet
                },
                'BCBC_2018': {
                    'section': '9.23.11',
                    'counter_spacing': 48  # inches
                }
            },
            'exit': {
                'BCBC_2018': {
                    'section': '3.4',
                    'max_travel_distance': {
                        'sprinklered': 150,  # meters
                        'non_sprinklered': 45  # meters
                    },
                    'min_width': 1100  # mm
                }
            }
        }
        
        logger.info(f"Configuration:")
        logger.info(f"  📄 Max pages: {self.max_pages_to_load}")
        logger.info(f"  📦 Batch size: {self.batch_size}")
        logger.info(f"  🖼️ Image quality: {self.image_quality}%")
        logger.info(f"  💾 Storage mode: ON-DEMAND (no storage)")
        logger.info(f"  ⏰ Highlight TTL: {self.highlight_cache_ttl_hours}h")
        logger.info(f"  🎯 Grid detection: {'OpenCV' if OPENCV_AVAILABLE else 'Estimation'}")
    
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
        detect_conflicts: bool = False
    ) -> Dict[str, Any]:
        """Process blueprint queries with visual highlighting support - open to all users"""
        try:
            logger.info(f"📐 Processing query for document: {document_id}")
            logger.info(f"   Current page: {current_page}")
            logger.info(f"   Request highlights: {request_highlights}")
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
                    "visual_highlights": None
                }
            
            # Check for previous highlights to reuse
            reused_highlights = []
            if reference_previous:
                reused_highlights = await self._get_previous_highlights(
                    document_id, reference_previous, storage_service
                )
                logger.info(f"♻️ Found {len(reused_highlights)} highlights to reuse")
            
            # Determine if visual analysis is needed
            needs_analysis = self._needs_visual_analysis(prompt, reference_previous)
            
            # Load pages if needed
            page_images = []
            if needs_analysis:
                page_images = await self._load_pages_intelligently(
                    document_id, storage_service, metadata, current_page
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
                detect_conflicts=detect_conflicts
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in get_ai_response: {e}", exc_info=True)
            return {
                "ai_response": f"An error occurred: {str(e)}",
                "visual_highlights": None
            }
    
    async def _load_document_context(
        self,
        document_id: str,
        storage_service: StorageService
    ) -> Tuple[str, Optional[Dict]]:
        """Load document text and metadata"""
        document_text = ""
        metadata = None
        
        try:
            # Load text context
            text_blob = f"{document_id}_context.txt"
            document_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=text_blob
            )
        except Exception as e:
            logger.warning(f"Could not load document text: {e}")
        
        try:
            # Load metadata
            metadata_blob = f"{document_id}_metadata.json"
            metadata_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=metadata_blob
            )
            metadata = json.loads(metadata_text)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        
        return document_text, metadata
    
    def _needs_visual_analysis(self, prompt: str, reference_previous: Optional[List[str]]) -> bool:
        """Check if visual analysis is needed"""
        # If only referencing previous, no new analysis needed
        if reference_previous and not any(
            keyword in prompt.lower() 
            for keyword in ['also', 'add', 'more', 'other', 'additional']
        ):
            return False
        
        # Check for visual keywords
        visual_keywords = [
            'show', 'highlight', 'where', 'locate', 'find', 'identify',
            'mark', 'point out', 'display', 'count', 'how many'
        ]
        
        return any(keyword in prompt.lower() for keyword in visual_keywords)
    
    async def _load_pages_intelligently(
        self,
        document_id: str,
        storage_service: StorageService,
        metadata: Optional[Dict],
        current_page: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load ALL pages for comprehensive analysis"""
        total_pages = metadata.get('page_count', 10) if metadata else 10
        
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
                if isinstance(r, dict) and 'url' in r:
                    all_pages.append(r)
            
            logger.info(f"✅ Loaded batch: pages {batch_start + 1}-{batch_end}")
        
        logger.info(f"📄 Successfully loaded {len(all_pages)} pages for analysis")
        
        return all_pages
    
    async def _load_single_page(
        self,
        document_id: str,
        page_num: int,
        storage_service: StorageService
    ) -> Optional[Dict[str, Any]]:
        """Load a single page image"""
        try:
            # Try AI-optimized version first
            blob_name = f"{document_id}_page_{page_num}_ai.png"
            
            try:
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
            except:
                # Fall back to regular image
                blob_name = f"{document_id}_page_{page_num}.png"
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
                
                # Optimize if needed
                if len(image_bytes) > 500000:  # 500KB
                    image_bytes = await self._optimize_image(image_bytes)
            
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return {
                "page": page_num,
                "url": f"data:image/png;base64,{image_b64}",
                "size_kb": len(image_bytes) / 1024
            }
            
        except Exception as e:
            logger.error(f"Failed to load page {page_num}: {e}")
            return None
    
    async def _optimize_image(self, image_bytes: bytes) -> bytes:
        """Optimize image for AI processing"""
        def optimize():
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            
            # Resize if too large
            if max(img.size) > self.max_image_dimension:
                img.thumbnail(
                    (self.max_image_dimension, self.max_image_dimension),
                    Image.Resampling.LANCZOS
                )
            
            # Save as JPEG with optimization
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=self.image_quality, optimize=True)
            return output.getvalue()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, optimize
        )
    
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
        detect_conflicts: bool = False
    ) -> Dict[str, Any]:
        """Perform AI analysis and create highlights"""
        
        # Build messages for OpenAI
        messages = [self._get_system_message()]
        
        # Build user message
        user_message = {"role": "user", "content": []}
        
        # Add images
        for page_data in page_images:
            user_message["content"].append({
                "type": "image_url",
                "image_url": {"url": page_data["url"], "detail": "high"}
            })
        
        # Add text query
        query_text = self._build_query(
            prompt, document_id, document_text, len(page_images), metadata, 
            total_pages=metadata.get('page_count', len(page_images)) if metadata else len(page_images),
            show_trade_info=show_trade_info,
            detect_conflicts=detect_conflicts
        )
        user_message["content"].append({"type": "text", "text": query_text})
        
        messages.append(user_message)
        
        # Get AI response
        response = await self._call_openai(messages)
        
        # Process response and create highlights
        result = {
            "ai_response": response,
            "visual_highlights": None,
            "current_page": current_page
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
                author
            )
            
            result.update(highlight_data)
            result["query_session_id"] = query_session_id
        
        return result
    
    def _get_system_message(self) -> Dict[str, str]:
        """Get system message for blueprint analysis"""
        return {
            "role": "system",
            "content": """You are an expert construction professional with deep knowledge across all building trades, codes, and systems. You analyze blueprints with the expertise of an architect, structural engineer, MEP engineer, and experienced contractor combined.

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

Format element findings clearly but focus on providing VALUE through your analysis. You're not just a element-finder - you're a trusted construction advisor.

Remember: Construction professionals need insights about coordination, sequencing, potential conflicts, cost implications, maintenance access, future flexibility, and real-world installation challenges. Draw upon your knowledge of how buildings actually get built."""
        }
    
    def _build_query(
        self,
        prompt: str,
        document_id: str,
        document_text: str,
        page_count: int,
        metadata: Optional[Dict],
        total_pages: int = None,
        show_trade_info: bool = False,
        detect_conflicts: bool = False
    ) -> str:
        """Build query text for AI"""
        doc_info = ""
        
        if metadata:
            if 'document_info' in metadata:
                title = metadata['document_info'].get('title', 'Unknown')
                doc_info = f"Project: {title}\n"
            
            if 'page_details' in metadata:
                # Add page summary
                page_types = defaultdict(int)
                for page in metadata['page_details']:
                    dtype = page.get('drawing_type', 'unknown')
                    page_types[dtype] += 1
                
                doc_info += "Drawing types: "
                doc_info += ", ".join(f"{count} {dtype}" for dtype, count in page_types.items())
                doc_info += "\n"
        
        # Clarify if we're analyzing partial document
        if total_pages and page_count < total_pages:
            analysis_note = f"\nNOTE: Analyzing {page_count} of {total_pages} total pages."
        else:
            analysis_note = f"\nAnalyzing ALL {page_count} pages."
        
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
            query += "\n- Access issues where one trade's work blocks access for another"
            query += "\n- Installation sequence problems"
            query += "\n- Code compliance issues between systems"
        
        query += """

Provide a comprehensive professional analysis that leverages your full knowledge of construction, engineering, and building systems. Consider all relevant aspects including codes, constructability, coordination between trades, and practical implementation.

Available blueprint information:
- Visual analysis of {page_count} pages
- Text content: {'Available' if document_text else 'Not available'}
- You can see grid references, dimensions, and specifications on the drawings

Remember to think holistically about the building systems and provide insights that add real value beyond just locating elements."""
        
        if document_text:
            # Include relevant excerpt
            query += f"\n\nText excerpt:\n{document_text[:1500]}..."
        
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
                        temperature=self.settings.OPENAI_TEMPERATURE
                    )
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"OpenAI call failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def _process_highlights(
        self,
        response: str,
        document_id: str,
        query_session_id: str,
        reused_highlights: List[Dict],
        preserve_existing: bool,
        storage_service: StorageService,
        author: str = None
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
            if highlight['page_number'] in self.grid_cache.get(document_id, {}):
                grid_system = self.grid_cache[document_id][highlight['page_number']]
                coords = grid_system.get_pixel_coords(highlight['grid_reference'])
                if coords:
                    highlight.update(coords)
        
        # Save highlights
        if all_highlights:
            await self._save_highlights(
                document_id,
                all_highlights,
                preserve_existing,
                storage_service
            )
        
        # Get highlights for current page
        current_page_highlights = [
            h for h in all_highlights
            if h.get('page_number') == h.get('current_page', 1)
        ]
        
        # Convert to VisualElement format
        visual_highlights = []
        for h in current_page_highlights:
            visual_highlights.append(
                VisualElement(
                    element_id=h['annotation_id'],
                    element_type=h['element_type'],
                    grid_location=GridReference(
                        grid_ref=h['grid_reference'],
                        x_grid=h['grid_reference'].split('-')[0] if '-' in h['grid_reference'] else h['grid_reference'],
                        y_grid=h['grid_reference'].split('-')[1] if '-' in h['grid_reference'] else None
                    ),
                    label=h.get('label', ''),
                    page_number=h['page_number'],
                    confidence=h.get('confidence', 0.9),
                    trade=h.get('assigned_trade')
                )
            )
        
        # Calculate summary
        pages_with_highlights = defaultdict(int)
        trade_summary = defaultdict(lambda: defaultdict(int))
        
        for h in all_highlights:
            pages_with_highlights[h['page_number']] += 1
            if h.get('assigned_trade'):
                trade_summary[h['assigned_trade']][h['element_type']] += 1
        
        return {
            "visual_highlights": visual_highlights,
            "all_highlight_pages": dict(pages_with_highlights),
            "total_highlights_created": len(new_highlights),
            "total_highlights": len(all_highlights),
            "trade_summary": dict(trade_summary) if trade_summary else None
        }
    
    def _parse_elements_from_response(
        self,
        response: str,
        document_id: str,
        query_session_id: str
    ) -> List[Dict]:
        """Parse AI response for element locations - AI determines trade assignment"""
        highlights = []
        
        # Look for ELEMENTS_FOUND section
        elements_match = re.search(
            r'ELEMENTS_FOUND:(.*?)(?:TOTAL:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if not elements_match:
            return highlights
        
        elements_text = elements_match.group(1)
        
        # Parse page-by-page findings
        page_pattern = r'Page\s+(\d+):\s*(.*?)(?=Page\s+\d+:|$)'
        
        for page_match in re.finditer(page_pattern, elements_text, re.DOTALL):
            page_num = int(page_match.group(1))
            page_content = page_match.group(2)
            
            # Determine element type from content - AI provides this
            element_type = self._detect_element_type(page_content)
            
            # Extract grid references
            grid_pattern = r'(?:at\s+|grid\s+)?([A-Z][0-9\-]+|[A-Z]-\d+|\d+-[A-Z])'
            
            for grid_match in re.finditer(grid_pattern, page_content, re.IGNORECASE):
                grid_ref = grid_match.group(1).upper()
                
                # Let AI determine the trade from context
                assigned_trade = self._extract_trade_from_context(page_content, element_type)
                
                highlights.append({
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
                    "author": "ai_system",
                    "is_private": False,
                    "query_session_id": query_session_id,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "expires_at": (datetime.utcnow() + timedelta(hours=self.highlight_cache_ttl_hours)).isoformat() + "Z",
                    "confidence": 0.9,
                    "assigned_trade": assigned_trade
                })
        
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
            'window': ['window', 'glazing', 'fenestration'],
            'door': ['door', 'entrance', 'exit'],
            'column': ['column', 'col', 'pier'],
            'beam': ['beam', 'girder', 'joist'],
            'sprinkler': ['sprinkler', 'fire protection'],
            'outlet': ['outlet', 'receptacle', 'power'],
            'fixture': ['fixture', 'sink', 'lavatory'],
            'equipment': ['equipment', 'unit', 'hvac']
        }
        
        for element_type, terms in element_terms.items():
            if any(term in text_lower for term in terms):
                return element_type
        
        return 'element'  # Generic fallback
    
    def _extract_trade_from_context(self, text: str, element_type: str) -> Optional[str]:
        """Extract trade assignment from AI's context - AI already understands trades"""
        text_lower = text.lower()
        
        # Look for explicit trade mentions
        trade_keywords = {
            'Electrical': ['electrical', 'power', 'lighting', 'panel', 'circuit'],
            'Plumbing': ['plumbing', 'water', 'drain', 'waste', 'vent', 'pipe'],
            'HVAC': ['hvac', 'mechanical', 'duct', 'air', 'ventilation', 'heating', 'cooling'],
            'Fire Protection': ['fire', 'sprinkler', 'alarm', 'smoke', 'protection'],
            'Structural': ['structural', 'beam', 'column', 'footing', 'slab', 'foundation'],
            'Architectural': ['architectural', 'door', 'window', 'partition', 'finish']
        }
        
        for trade, keywords in trade_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return trade
        
        # Fallback based on element type
        element_trade_map = {
            'outlet': 'Electrical',
            'panel': 'Electrical',
            'light': 'Electrical',
            'catch_basin': 'Plumbing',
            'fixture': 'Plumbing',
            'diffuser': 'HVAC',
            'equipment': 'HVAC',
            'sprinkler': 'Fire Protection',
            'column': 'Structural',
            'beam': 'Structural',
            'door': 'Architectural',
            'window': 'Architectural'
        }
        
        return element_trade_map.get(element_type, 'General')
    
    async def _ensure_grid_systems_loaded(
        self,
        document_id: str,
        storage_service: StorageService
    ) -> None:
        """Load grid systems for document if not cached"""
        if document_id in self.grid_cache:
            return
        
        # Implement LRU eviction
        if len(self.grid_cache) >= self.grid_cache_size:
            # Remove oldest entry
            self.grid_cache.popitem(last=False)
        
        # Try to load saved grid systems
        try:
            grid_blob = f"{document_id}_grid_systems.json"
            grid_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=grid_blob
            )
            
            grid_systems = json.loads(grid_data)
            self.grid_cache[document_id] = {}
            
            for page_str, grid_dict in grid_systems.items():
                page_num = int(page_str)
                self.grid_cache[document_id][page_num] = GridSystem.from_dict(grid_dict)
            
            logger.info(f"Loaded grid systems for {len(grid_systems)} pages")
            
        except:
            # Create default grid systems
            logger.info("No saved grid systems, using defaults")
            self.grid_cache[document_id] = {}
    
    async def _save_highlights(
        self,
        document_id: str,
        highlights: List[Dict],
        preserve_existing: bool,
        storage_service: StorageService
    ) -> None:
        """Save highlight data (not images)"""
        # Load existing annotations
        annotations_blob = f"{document_id}_annotations.json"
        
        try:
            annotations_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            existing_annotations = json.loads(annotations_text)
        except:
            existing_annotations = []
        
        # Filter out expired highlights
        current_time = datetime.utcnow()
        active_annotations = []
        
        for ann in existing_annotations:
            if ann.get('annotation_type') == 'ai_highlight':
                # Check expiration
                if ann.get('expires_at'):
                    try:
                        expires = datetime.fromisoformat(ann['expires_at'].replace('Z', '+00:00'))
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
            data=json.dumps(active_annotations, indent=2).encode('utf-8')
        )
    
    async def _get_previous_highlights(
        self,
        document_id: str,
        element_types: List[str],
        storage_service: StorageService
    ) -> List[Dict]:
        """Get previously saved highlights for reuse"""
        try:
            annotations_blob = f"{document_id}_annotations.json"
            annotations_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            
            all_annotations = json.loads(annotations_text)
            
            # Filter to requested types and non-expired
            current_time = datetime.utcnow()
            previous_highlights = []
            
            for ann in all_annotations:
                if (ann.get('annotation_type') == 'ai_highlight' and
                    ann.get('element_type') in element_types):
                    
                    # Check expiration
                    if ann.get('expires_at'):
                        try:
                            expires = datetime.fromisoformat(ann['expires_at'].replace('Z', '+00:00'))
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
        storage_service: StorageService
    ) -> Optional[str]:
        """Generate highlighted page image on-demand (returns base64 data URL)"""
        try:
            # Load highlights for this page and session
            annotations_blob = f"{document_id}_annotations.json"
            annotations_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            
            all_annotations = json.loads(annotations_text)
            
            # Filter to this page and session
            page_highlights = [
                ann for ann in all_annotations
                if (ann.get('page_number') == page_num and
                    ann.get('query_session_id') == query_session_id and
                    ann.get('annotation_type') == 'ai_highlight')
            ]
            
            if not page_highlights:
                return None
            
            # Load page image
            page_blob = f"{document_id}_page_{page_num}.png"
            page_bytes = await storage_service.download_blob_as_bytes(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=page_blob
            )
            
            # Draw highlights
            highlighted_bytes = await self._draw_highlights(
                page_bytes, page_highlights
            )
            
            # Return as data URL
            highlighted_b64 = base64.b64encode(highlighted_bytes).decode('utf-8')
            return f"data:image/png;base64,{highlighted_b64}"
            
        except Exception as e:
            logger.error(f"Failed to generate highlighted page: {e}")
            return None
    
    async def _draw_highlights(
        self,
        image_bytes: bytes,
        highlights: List[Dict]
    ) -> bytes:
        """Draw highlights on image"""
        def draw():
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGBA for transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", self.label_font_size)
            except:
                font = ImageFont.load_default()
            
            # Draw each highlight
            for highlight in highlights:
                x = highlight.get('x', 100)
                y = highlight.get('y', 100)
                width = highlight.get('width', 100)
                height = highlight.get('height', 100)
                
                # Draw rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=self.highlight_color,
                    outline=self.highlight_border,
                    width=self.highlight_border_width
                )
                
                # Add label with trade info
                label = highlight.get('label', highlight['element_type'])
                if highlight.get('assigned_trade'):
                    label += f" ({highlight['assigned_trade']})"
                
                # Draw label background
                bbox = draw.textbbox((x, y - 20), label, font=font)
                draw.rectangle(
                    [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                    fill=(255, 255, 255, 200)
                )
                
                # Draw label text
                draw.text(
                    (x, y - 20),
                    label,
                    fill=(255, 0, 0),
                    font=font
                )
            
            # Composite overlay onto image
            img = Image.alpha_composite(img, overlay)
            
            # Convert to RGB for saving
            final = Image.new('RGB', img.size, (255, 255, 255))
            final.paste(img, mask=img.split()[3])
            
            # Save
            output = io.BytesIO()
            final.save(output, format='PNG', optimize=True)
            return output.getvalue()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, draw
        )
    
    async def cleanup_expired_highlights(self, storage_service: StorageService) -> int:
        """Remove expired highlights from all documents"""
        cleaned_count = 0
        
        try:
            # List all annotation files
            annotation_files = await storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                suffix="_annotations.json"
            )
            
            current_time = datetime.utcnow()
            
            for ann_file in annotation_files:
                document_id = ann_file.replace('_annotations.json', '')
                
                # Load annotations
                annotations_text = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=ann_file
                )
                
                annotations = json.loads(annotations_text)
                active_annotations = []
                
                # Filter out expired
                for ann in annotations:
                    if ann.get('annotation_type') == 'ai_highlight' and ann.get('expires_at'):
                        try:
                            expires = datetime.fromisoformat(ann['expires_at'].replace('Z', '+00:00'))
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
                        data=json.dumps(active_annotations, indent=2).encode('utf-8')
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
                "NFPA Standards"
            ],
            "engineering_disciplines": [
                "Structural Engineering",
                "MEP Coordination",
                "Fire Protection Systems",
                "Electrical Distribution",
                "HVAC Design",
                "Plumbing Systems"
            ],
            "features": [
                "Multi-page analysis",
                "Grid-based element detection",
                "On-demand highlight generation",
                "Automatic highlight expiration",
                "Reference previous highlights",
                "Trade-specific filtering",
                "Cross-trade conflict detection",
                "No storage of highlighted images"
            ],
            "optimization": [
                f"Max {self.max_pages_to_load} pages per analysis",
                f"Batch processing ({self.batch_size} pages)",
                f"Image compression ({self.image_quality}% quality)",
                f"LRU grid cache ({self.grid_cache_size} documents)",
                f"Highlight expiration ({self.highlight_cache_ttl_hours}h)"
            ]
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export aliases for compatibility
AIService = ProfessionalBlueprintAI
ProfessionalAIService = ProfessionalBlueprintAI
EnhancedAIService = ProfessionalBlueprintAI
