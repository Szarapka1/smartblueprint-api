# app/services/ai_service.py - WITH GRID-BASED HIGHLIGHTING

import os
import asyncio
import base64
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from PIL import Image
import io

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# OpenAI imports
try:
    from openai import OpenAI
    logger.info("✅ OpenAI SDK imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import OpenAI SDK: {e}")
    raise

# Internal imports
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService
from app.models.schemas import VisualElement, GridReference, DrawingGrid


@dataclass
class GridSystem:
    """Represents a drawing's grid system"""
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    x_spacing: Dict[str, float] = field(default_factory=dict)
    y_spacing: Dict[str, float] = field(default_factory=dict)
    scale: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_labels": self.x_labels,
            "y_labels": self.y_labels,
            "x_spacing": self.x_spacing,
            "y_spacing": self.y_spacing,
            "scale": self.scale
        }


class ProfessionalBlueprintAI:
    """Professional AI service with grid-based visual element highlighting"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Professional Blueprint AI initialized with highlighting support")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration for optimization
        self.max_pages_to_load = int(os.getenv("AI_MAX_PAGES", "100"))
        self.batch_size = int(os.getenv("AI_BATCH_SIZE", "10"))
        self.image_quality = int(os.getenv("AI_IMAGE_QUALITY", "85"))
        self.max_image_dimension = int(os.getenv("AI_MAX_IMAGE_DIMENSION", "2000"))
        
        # Grid detection patterns
        self.grid_patterns = {
            'structural': r'([A-Z]\d+|W\d+|[A-Z]+\d*)',
            'architectural': r'([A-Z]\.?\d+|\d+\.?\d*|[A-Z]-[A-Z])',
            'coordinate': r'([A-Z]+)-([A-Z]+\d*|\d+)'
        }
        
        logger.info(f"   📄 Max pages: {self.max_pages_to_load}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   🖼️ Image quality: {self.image_quality}%")
        logger.info(f"   🎯 Grid highlighting: Enabled")
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, 
                              author: str = None,
                              current_page: Optional[int] = None,
                              request_highlights: bool = True) -> Dict[str, Any]:
        """Process blueprint queries with optional grid-based highlighting"""
        try:
            logger.info(f"📐 Processing blueprint analysis for {document_id}")
            logger.info(f"   📄 Current page: {current_page}")
            logger.info(f"   🎯 Highlights requested: {request_highlights}")
            
            analysis_start = asyncio.get_event_loop().time()
            
            # Load document context and metadata
            document_text = ""
            metadata = None
            
            try:
                # Load text context
                context_task = storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                document_text = await asyncio.wait_for(context_task, timeout=30.0)
                logger.info(f"✅ Loaded text from all pages: {len(document_text)} characters")
                
                # Load metadata
                try:
                    metadata_task = storage_service.download_blob_as_text(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{document_id}_metadata.json"
                    )
                    metadata_text = await asyncio.wait_for(metadata_task, timeout=10.0)
                    metadata = json.loads(metadata_text)
                    logger.info(f"✅ Loaded metadata: {metadata.get('page_count', 0)} pages")
                except:
                    logger.info("ℹ️ No metadata found, will discover pages")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return {
                    "ai_response": "Unable to load the blueprint. Please ensure the document is properly uploaded and processed.",
                    "visual_highlights": None
                }
            
            # Determine if we need visual analysis
            needs_visual_analysis = self._needs_visual_highlighting(prompt, current_page, request_highlights)
            
            if needs_visual_analysis and current_page:
                # Load specific page for visual analysis
                page_images = await self._load_specific_page(
                    document_id, current_page, storage_service
                )
            else:
                # Load all pages for comprehensive analysis
                page_images = await self._load_all_pages_optimized(
                    document_id, storage_service, metadata
                )
            
            loading_time = asyncio.get_event_loop().time() - analysis_start
            logger.info(f"⏱️ Document loaded in {loading_time:.2f}s")
            
            # Process with professional analysis
            result = await self._analyze_blueprint_with_highlighting(
                prompt=prompt,
                document_text=document_text,
                image_urls=page_images,
                document_id=document_id,
                author=author,
                current_page=current_page,
                request_highlights=request_highlights and needs_visual_analysis
            )
            
            total_time = asyncio.get_event_loop().time() - analysis_start
            logger.info(f"✅ Analysis complete in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return {
                "ai_response": f"Error analyzing blueprint: {str(e)}",
                "visual_highlights": None
            }
    
    def _needs_visual_highlighting(self, prompt: str, current_page: Optional[int], 
                                  request_highlights: bool) -> bool:
        """Determine if the query needs visual highlighting"""
        if not request_highlights or not current_page:
            return False
        
        # Keywords that indicate visual search
        visual_keywords = [
            'show', 'highlight', 'where', 'locate', 'find', 'identify',
            'point out', 'mark', 'indicate', 'display', 'circle'
        ]
        
        # Element types that can be highlighted
        element_keywords = [
            'column', 'beam', 'outlet', 'window', 'door', 'sprinkler',
            'duct', 'pipe', 'equipment', 'fixture', 'panel', 'valve',
            'switch', 'light', 'vent', 'drain', 'wall', 'opening'
        ]
        
        prompt_lower = prompt.lower()
        
        # Check if prompt contains visual request
        has_visual_keyword = any(keyword in prompt_lower for keyword in visual_keywords)
        has_element_keyword = any(keyword in prompt_lower for keyword in element_keywords)
        
        return has_visual_keyword or has_element_keyword
    
    async def _load_specific_page(self, document_id: str, page_num: int,
                                 storage_service: StorageService) -> List[Dict[str, any]]:
        """Load a specific page for visual analysis"""
        try:
            page_data = await self._load_single_page_optimized(
                document_id, page_num, storage_service
            )
            
            if page_data:
                logger.info(f"✅ Loaded page {page_num} for visual analysis")
                return [page_data]
            else:
                logger.warning(f"⚠️ Could not load page {page_num}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading page {page_num}: {e}")
            return []
    
    async def _analyze_blueprint_with_highlighting(self, prompt: str, document_text: str, 
                                                  image_urls: List[Dict[str, any]], 
                                                  document_id: str,
                                                  author: str,
                                                  current_page: Optional[int],
                                                  request_highlights: bool) -> Dict[str, Any]:
        """Analyze blueprint with optional grid-based highlighting"""
        try:
            logger.info("="*50)
            logger.info("📊 PROFESSIONAL BLUEPRINT ANALYSIS WITH HIGHLIGHTING")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"🎯 Current Page: {current_page}")
            logger.info(f"✨ Highlights Requested: {request_highlights}")
            logger.info("="*50)
            
            # Enhanced system message for grid-based highlighting
            system_message = self._get_enhanced_system_message(request_highlights, current_page)
            
            messages = [system_message]
            
            # Build user message
            user_message = {"role": "user", "content": []}
            
            # Add page images
            if image_urls:
                for page_data in image_urls:
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {"url": page_data["url"], "detail": "high"}
                    })
            
            # Build comprehensive query
            query_text = self._build_query_text(
                prompt, document_id, document_text, 
                current_page, request_highlights, len(image_urls)
            )
            
            user_message["content"].append({"type": "text", "text": query_text})
            messages.append(user_message)
            
            logger.info("📤 Requesting professional analysis with grid detection")
            
            # Get AI response
            response = await self._get_ai_completion(messages)
            
            # Parse response for visual elements if requested
            if request_highlights and current_page:
                parsed_response = self._parse_response_with_highlights(response, current_page)
            else:
                parsed_response = {
                    "ai_response": response,
                    "visual_highlights": None,
                    "drawing_grid": None,
                    "highlight_summary": None,
                    "current_page": current_page
                }
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "ai_response": f"Error performing analysis: {str(e)}",
                "visual_highlights": None
            }
    
    def _get_enhanced_system_message(self, request_highlights: bool, 
                                    current_page: Optional[int]) -> Dict[str, str]:
        """Get system message with grid highlighting instructions"""
        
        base_message = """You are a professional blueprint analyst with extensive experience across all construction trades. You analyze MULTI-SHEET blueprint sets, ALWAYS provide code-based recommendations when information is missing, AND ask clarifying questions."""
        
        highlighting_instructions = """

🎯 VISUAL ELEMENT IDENTIFICATION AND GRID HIGHLIGHTING:

When asked to show/highlight/identify specific elements on a drawing:

1. IDENTIFY GRID SYSTEM
• Look for grid lines with labels (e.g., A, B, C... or 1, 2, 3... or W1, W2...)
• Note grid intersections (e.g., A-1, B-2, W2-WA)
• Identify scale from title block

2. LOCATE ELEMENTS USING GRID REFERENCES
• For each requested element, provide its grid location
• Use format: "Column at W2-WA" or "Outlet near grid B-3"
• Be specific about grid intersections or areas

3. PROVIDE STRUCTURED OUTPUT FOR HIGHLIGHTING
When identifying elements for highlighting, include:
```
VISUAL_ELEMENTS:
- Element: [type] at grid [reference]
  Label: [identifier or description]
  Size: [dimensions if shown]
```

4. GRID SYSTEM INFORMATION
If grid system is visible, provide:
```
GRID_SYSTEM:
X-axis: [list of horizontal grid labels]
Y-axis: [list of vertical grid labels]
Scale: [drawing scale]
```

Example for columns:
```
VISUAL_ELEMENTS:
- Element: Column at grid W2-WA
  Label: C-101
  Size: 600x600mm
- Element: Column at grid W3-WC
  Label: C-102
  Size: 600x600mm

GRID_SYSTEM:
X-axis: W1, W2, W3, W4, W5
Y-axis: WA, WB, WC, WD, WE
Scale: 1/8" = 1'-0"
```"""
        
        comprehensive_instructions = """

📍 COMPREHENSIVE ANALYSIS APPROACH:

1. ANALYZE ALL SHEETS PROVIDED
• Identify what's on each sheet
• Cross-reference between sheets
• Note what's missing

2. ALWAYS PROVIDE COMPLETE ANSWERS USING CODES
• If sizes aren't shown → Use code minimums and typical sizes
• If quantities are missing → Calculate per code requirements
• Never say "not enough information" → Give code-based answer

3. ASK CLARIFYING QUESTIONS
• To verify assumptions
• To find additional sheets
• To provide better accuracy"""
        
        if request_highlights and current_page:
            return {
                "role": "system",
                "content": base_message + highlighting_instructions + comprehensive_instructions
            }
        else:
            return {
                "role": "system",
                "content": base_message + comprehensive_instructions
            }
    
    def _build_query_text(self, prompt: str, document_id: str, document_text: str,
                         current_page: Optional[int], request_highlights: bool,
                         image_count: int) -> str:
        """Build the query text with appropriate context"""
        
        base_query = f"""Document: {document_id}
Question: {prompt}"""
        
        if request_highlights and current_page:
            highlight_context = f"""

VISUAL HIGHLIGHTING REQUEST:
- Current page: {current_page}
- Identify and locate elements using grid references
- Provide structured VISUAL_ELEMENTS and GRID_SYSTEM sections
- Be specific about grid locations for each element"""
        else:
            highlight_context = """

Note: If this query is about locating specific elements, please specify which page you're viewing for visual highlighting."""
        
        context = f"""{base_query}{highlight_context}

Total pages provided: {image_count}
Text from all pages: {'Available' if document_text else 'Not available'}

Drawing text content (from all sheets):
{document_text}"""
        
        return context
    
    async def _get_ai_completion(self, messages: List[Dict]) -> str:
        """Get completion from OpenAI"""
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.0
                    )
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Retry {retry_count}/{max_retries}: {e}")
                else:
                    raise
    
    def _parse_response_with_highlights(self, response: str, current_page: int) -> Dict[str, Any]:
        """Parse AI response to extract visual elements and grid information"""
        
        # Extract visual elements section
        visual_elements = []
        grid_system = None
        
        # Look for VISUAL_ELEMENTS section
        visual_match = re.search(r'VISUAL_ELEMENTS:(.*?)(?=GRID_SYSTEM:|$)', response, re.DOTALL)
        if visual_match:
            elements_text = visual_match.group(1)
            visual_elements = self._parse_visual_elements(elements_text)
        
        # Look for GRID_SYSTEM section
        grid_match = re.search(r'GRID_SYSTEM:(.*?)(?=\n\n|$)', response, re.DOTALL)
        if grid_match:
            grid_text = grid_match.group(1)
            grid_system = self._parse_grid_system(grid_text)
        
        # Clean the response text (remove structured sections)
        clean_response = response
        if visual_match:
            clean_response = clean_response.replace(visual_match.group(0), '')
        if grid_match:
            clean_response = clean_response.replace(grid_match.group(0), '')
        
        # Create highlight summary
        highlight_summary = {}
        if visual_elements:
            for element in visual_elements:
                elem_type = element.element_type
                highlight_summary[elem_type] = highlight_summary.get(elem_type, 0) + 1
        
        return {
            "ai_response": clean_response.strip(),
            "visual_highlights": visual_elements if visual_elements else None,
            "drawing_grid": grid_system,
            "highlight_summary": highlight_summary if highlight_summary else None,
            "current_page": current_page
        }
    
    def _parse_visual_elements(self, elements_text: str) -> List[VisualElement]:
        """Parse visual elements from structured text"""
        elements = []
        
        # Pattern to match element entries
        element_pattern = r'- Element: (.+?) at grid (.+?)(?:\n\s+Label: (.+?))?(?:\n\s+Size: (.+?))?'
        
        matches = re.finditer(element_pattern, elements_text)
        
        for i, match in enumerate(matches):
            element_type = match.group(1).lower().replace(' ', '_')
            grid_ref = match.group(2).strip()
            label = match.group(3).strip() if match.group(3) else f"{element_type}_{i+1}"
            size = match.group(4).strip() if match.group(4) else None
            
            # Parse grid reference
            grid_parts = grid_ref.split('-')
            if len(grid_parts) == 2:
                x_grid, y_grid = grid_parts
            else:
                x_grid = grid_ref
                y_grid = None
            
            element = VisualElement(
                element_id=f"{element_type}_{i+1}",
                element_type=element_type,
                grid_location=GridReference(
                    grid_ref=grid_ref,
                    x_grid=x_grid,
                    y_grid=y_grid
                ),
                label=label,
                dimensions=size,
                confidence=0.9
            )
            
            elements.append(element)
        
        logger.info(f"📍 Parsed {len(elements)} visual elements")
        return elements
    
    def _parse_grid_system(self, grid_text: str) -> DrawingGrid:
        """Parse grid system information"""
        x_labels = []
        y_labels = []
        scale = None
        
        # Parse X-axis
        x_match = re.search(r'X-axis:\s*(.+)', grid_text)
        if x_match:
            x_labels = [label.strip() for label in x_match.group(1).split(',')]
        
        # Parse Y-axis
        y_match = re.search(r'Y-axis:\s*(.+)', grid_text)
        if y_match:
            y_labels = [label.strip() for label in y_match.group(1).split(',')]
        
        # Parse scale
        scale_match = re.search(r'Scale:\s*(.+)', grid_text)
        if scale_match:
            scale = scale_match.group(1).strip()
        
        if x_labels or y_labels:
            grid = DrawingGrid(
                x_labels=x_labels,
                y_labels=y_labels,
                scale=scale
            )
            logger.info(f"📐 Parsed grid system: {len(x_labels)}x{len(y_labels)}")
            return grid
        
        return None
    
    # --- Keep all existing methods from original file ---
    
    async def _load_all_pages_optimized(self, document_id: str, 
                                       storage_service: StorageService,
                                       metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """Load all pages with optimizations for large documents"""
        try:
            # Determine total pages
            if metadata and 'page_count' in metadata:
                total_pages = metadata['page_count']
            else:
                # Discover pages by checking what exists
                total_pages = await self._discover_page_count(document_id, storage_service)
            
            pages_to_load = min(total_pages, self.max_pages_to_load)
            logger.info(f"📄 Loading {pages_to_load} pages (total: {total_pages})")
            
            all_page_urls = []
            
            # Process pages in batches for efficiency
            for batch_start in range(0, pages_to_load, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pages_to_load)
                
                # Create tasks for parallel loading
                tasks = []
                for page_num in range(batch_start + 1, batch_end + 1):
                    task = self._load_single_page_optimized(
                        document_id, page_num, storage_service
                    )
                    tasks.append(task)
                
                # Execute batch in parallel
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(batch_results):
                    if isinstance(result, dict) and not isinstance(result, Exception):
                        all_page_urls.append(result)
                    else:
                        logger.warning(f"Failed to load page {batch_start + i + 1}: {result}")
                
                logger.info(f"✅ Loaded batch: pages {batch_start + 1}-{batch_end}")
            
            logger.info(f"✅ Successfully loaded {len(all_page_urls)} pages")
            return all_page_urls
            
        except Exception as e:
            logger.error(f"Error loading pages: {e}")
            return []
    
    async def _discover_page_count(self, document_id: str, storage_service: StorageService) -> int:
        """Discover how many pages exist for a document"""
        # Check for AI-optimized images first, then regular images
        page_count = 0
        
        # Binary search for efficiency
        low, high = 1, 200  # Assume max 200 pages
        
        while low <= high:
            mid = (low + high) // 2
            
            # Check if this page exists
            exists = await storage_service.blob_exists(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_page_{mid}_ai.png"
            )
            
            if not exists:
                # Try regular image
                exists = await storage_service.blob_exists(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_page_{mid}.png"
                )
            
            if exists:
                page_count = mid
                low = mid + 1
            else:
                high = mid - 1
        
        logger.info(f"📄 Discovered {page_count} pages for document")
        return page_count
    
    async def _load_single_page_optimized(self, document_id: str, page_num: int, 
                                         storage_service: StorageService) -> Optional[Dict]:
        """Load a single page with optimization"""
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
                
                # Optimize on the fly if needed
                image_bytes = await self._optimize_image_for_ai(image_bytes)
            
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
    
    async def _optimize_image_for_ai(self, image_bytes: bytes) -> bytes:
        """Optimize image for AI processing to reduce memory usage"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._compress_image,
            image_bytes
        )
    
    def _compress_image(self, image_bytes: bytes) -> bytes:
        """Compress image while maintaining readability"""
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            
            # Resize if too large
            if max(img.size) > self.max_image_dimension:
                img.thumbnail(
                    (self.max_image_dimension, self.max_image_dimension), 
                    Image.Resampling.LANCZOS
                )
            
            # Save with optimization
            output = io.BytesIO()
            img.save(
                output, 
                format='JPEG', 
                quality=self.image_quality, 
                optimize=True
            )
            
            compressed = output.getvalue()
            
            # Log compression
            compression_ratio = (1 - len(compressed) / len(image_bytes)) * 100
            if compression_ratio > 0:
                logger.debug(f"Image compressed by {compression_ratio:.1f}%")
            
            return compressed
            
        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return image_bytes
    
    def get_professional_capabilities(self) -> Dict[str, List[str]]:
        """Return professional capabilities of the service"""
        return {
            "building_codes": [
                "2018 BCBC (British Columbia Building Code)",
                "NBC (National Building Code of Canada)",
                "CSA Standards",
                "NFPA Standards",
                "Local Municipal Codes"
            ],
            "engineering_disciplines": [
                "Architectural",
                "Structural", 
                "Mechanical (HVAC)",
                "Electrical",
                "Plumbing",
                "Fire Protection",
                "Civil/Site"
            ],
            "analysis_features": [
                "Multi-page document support",
                "Cross-trade coordination",
                "Code compliance checking",
                "Quantity takeoffs",
                "Professional calculations",
                "Drawing cross-referencing",
                "Grid-based element highlighting"
            ],
            "visual_features": [
                "Grid system detection",
                "Element location identification",
                "Dynamic highlighting",
                "Multi-element selection",
                "Drawing scale recognition"
            ],
            "optimization_features": [
                f"Handles up to {self.max_pages_to_load} pages",
                f"Parallel loading in batches of {self.batch_size}",
                "Image compression for efficiency",
                "Automatic retry on context limits",
                "Optimized for large documents"
            ]
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the professional AI service
ProfessionalAIService = ProfessionalBlueprintAI
AIService = ProfessionalBlueprintAI
EnhancedAIService = ProfessionalBlueprintAI
