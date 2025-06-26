# app/services/ai_service.py - COMPLETE VISUAL HIGHLIGHTING SYSTEM

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
from PIL import Image, ImageDraw, ImageFont
import io
import uuid
import numpy as np
import cv2

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
from app.models.schemas import (
    VisualElement, GridReference, DrawingGrid, ChatRequest
)


@dataclass
class GridSystem:
    """Represents a drawing's grid system with pixel mappings"""
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
        """Convert grid reference to pixel coordinates with bounding box"""
        # Parse grid reference (e.g., "B-3", "W2-WA", "A2")
        patterns = [
            r'^([A-Z]+)[-\s]?(\d+)$',  # A-1, B2, C 3
            r'^([A-Z]+\d+)[-\s]?([A-Z]+\d*)$',  # W2-WA
            r'^(\d+)[-\s]?([A-Z]+)$',  # 1-A, 2-B
        ]
        
        for pattern in patterns:
            match = re.match(pattern, grid_ref.upper())
            if match:
                x_ref = match.group(1)
                y_ref = match.group(2)
                
                # Get coordinates
                x = self.x_coordinates.get(x_ref)
                y = self.y_coordinates.get(y_ref)
                
                if x is None:
                    x = self._estimate_x_coord(x_ref)
                if y is None:
                    y = self._estimate_y_coord(y_ref)
                
                if x is not None and y is not None:
                    return {
                        "x": x,
                        "y": y,
                        "width": self.cell_width,
                        "height": self.cell_height,
                        "center_x": x + self.cell_width // 2,
                        "center_y": y + self.cell_height // 2
                    }
        
        return None
    
    def _estimate_x_coord(self, x_ref: str) -> Optional[int]:
        """Estimate X coordinate for unknown reference"""
        if x_ref.isalpha() and len(x_ref) == 1:
            # Simple letter reference
            index = ord(x_ref) - ord('A')
            return self.origin_x + (index * self.cell_width)
        elif x_ref in self.x_labels:
            # Find position in known labels
            index = self.x_labels.index(x_ref)
            return self.origin_x + (index * self.cell_width)
        return None
    
    def _estimate_y_coord(self, y_ref: str) -> Optional[int]:
        """Estimate Y coordinate for unknown reference"""
        if y_ref.isdigit():
            # Simple number reference
            index = int(y_ref) - 1
            return self.origin_y + (index * self.cell_height)
        elif y_ref in self.y_labels:
            # Find position in known labels
            index = self.y_labels.index(y_ref)
            return self.origin_y + (index * self.cell_height)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "x_labels": self.x_labels,
            "y_labels": self.y_labels,
            "scale": self.scale,
            "confidence": self.confidence,
            "cell_dimensions": {
                "width": self.cell_width,
                "height": self.cell_height
            }
        }


class ProfessionalBlueprintAI:
    """Professional AI service with complete visual highlighting system"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Professional Blueprint AI initialized with visual highlighting")
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
        
        # Visual highlighting configuration
        self.highlight_color = (255, 255, 0, 100)  # Yellow with transparency
        self.highlight_border = (255, 0, 0, 255)   # Red border
        self.highlight_border_width = 3
        self.label_font_size = 16
        
        # Grid detection patterns (keeping existing)
        self.grid_patterns = {
            'structural': r'([A-Z]\d+|W\d+|[A-Z]+\d*)',
            'architectural': r'([A-Z]\.?\d+|\d+\.?\d*|[A-Z]-[A-Z])',
            'coordinate': r'([A-Z]+)-([A-Z]+\d*|\d+)'
        }
        
        # Element patterns for detection (keeping existing)
        self.element_patterns = {
            'window': r'(?:TYPE|type)\s*(\d+)|W\d+|WINDOW|window',
            'door': r'(?:TYPE|type)\s*([A-Z]\d*)|D\d+|DOOR|door',
            'column': r'C\d+|COL\.?\s*\d+|COLUMN',
            'beam': r'B\d+|BM\.?\s*\d+|BEAM',
            'catch_basin': r'CB[-\s]?\d+|CATCH\s*BASIN',
            'sprinkler': r'SP[-\s]?\d+|SPRINKLER',
            'outlet': r'OUTLET|RECEPTACLE|DUPLEX',
            'panel': r'PANEL\s*[A-Z]\d*|EP[-\s]?\d+|MDP'
        }
        
        # Cache for grid systems per document
        self.grid_cache = {}
        
        logger.info(f"   📄 Max pages: {self.max_pages_to_load}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   🖼️ Image quality: {self.image_quality}%")
        logger.info(f"   🎯 Visual highlighting: Enabled")
        logger.info(f"   🎨 Highlight color: Yellow with red border")
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, 
                              author: str = None,
                              current_page: Optional[int] = None,
                              request_highlights: bool = True,
                              reference_previous: Optional[List[str]] = None,
                              preserve_existing: bool = False) -> Dict[str, Any]:
        """Process blueprint queries with visual highlighting support"""
        try:
            logger.info(f"📐 Processing blueprint analysis for {document_id}")
            logger.info(f"   📄 Current page: {current_page}")
            logger.info(f"   🎯 Highlights requested: {request_highlights}")
            logger.info(f"   📚 Reference previous: {reference_previous}")
            logger.info(f"   🔒 Preserve existing: {preserve_existing}")
            
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
            
            # Load grid systems for the document
            await self._load_or_detect_grid_systems(document_id, storage_service, metadata)
            
            # Check if we can reuse previous highlights
            if reference_previous:
                reused_highlights = await self._get_previous_highlights(
                    document_id, reference_previous, storage_service
                )
                if reused_highlights:
                    logger.info(f"♻️ Reusing {len(reused_highlights)} previous highlights")
            
            # Determine if we need new visual analysis
            needs_new_analysis = self._needs_visual_analysis(prompt, reference_previous)
            
            if needs_new_analysis:
                # Load ALL pages for comprehensive analysis
                page_images = await self._load_all_pages_optimized(
                    document_id, storage_service, metadata
                )
                logger.info(f"📄 Loaded {len(page_images)} pages for analysis")
            else:
                # Just load current page for context
                if current_page:
                    page_images = await self._load_specific_page(
                        document_id, current_page, storage_service
                    )
                else:
                    page_images = []
            
            loading_time = asyncio.get_event_loop().time() - analysis_start
            logger.info(f"⏱️ Document loaded in {loading_time:.2f}s")
            
            # Process with professional analysis
            result = await self._analyze_blueprint_with_visual_highlighting(
                prompt=prompt,
                document_text=document_text,
                image_urls=page_images,
                document_id=document_id,
                author=author,
                current_page=current_page,
                request_highlights=request_highlights,
                reference_previous=reference_previous,
                preserve_existing=preserve_existing,
                storage_service=storage_service,
                metadata=metadata
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
    
    async def _load_or_detect_grid_systems(self, document_id: str, 
                                          storage_service: StorageService,
                                          metadata: Optional[Dict]):
        """Load existing grid systems or detect them for each page"""
        try:
            # Try to load existing grid data
            grid_blob = f"{document_id}_grid_systems.json"
            try:
                grid_data = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=grid_blob
                )
                grid_systems = json.loads(grid_data)
                
                # Convert to GridSystem objects
                self.grid_cache[document_id] = {}
                for page_str, grid_dict in grid_systems.items():
                    page_num = int(page_str)
                    self.grid_cache[document_id][page_num] = GridSystem(
                        page_number=page_num,
                        x_labels=grid_dict.get('x_labels', []),
                        y_labels=grid_dict.get('y_labels', []),
                        x_coordinates=grid_dict.get('x_coordinates', {}),
                        y_coordinates=grid_dict.get('y_coordinates', {}),
                        cell_width=grid_dict.get('cell_width', 100),
                        cell_height=grid_dict.get('cell_height', 100),
                        origin_x=grid_dict.get('origin_x', 100),
                        origin_y=grid_dict.get('origin_y', 100),
                        scale=grid_dict.get('scale'),
                        confidence=grid_dict.get('confidence', 0.8)
                    )
                
                logger.info(f"✅ Loaded grid systems for {len(self.grid_cache[document_id])} pages")
                return
                
            except:
                logger.info("🔍 No existing grid data, will detect grids")
            
            # Detect grids for each page
            if metadata and 'page_count' in metadata:
                page_count = min(metadata['page_count'], 20)  # Limit grid detection
                
                grid_systems = {}
                for page_num in range(1, page_count + 1):
                    try:
                        # Load page image
                        page_blob = f"{document_id}_page_{page_num}.png"
                        page_bytes = await storage_service.download_blob_as_bytes(
                            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                            blob_name=page_blob
                        )
                        
                        # Detect grid
                        grid_system = await self._detect_grid_on_page(
                            page_bytes, page_num
                        )
                        
                        if grid_system and grid_system.confidence > 0.5:
                            grid_systems[str(page_num)] = {
                                'x_labels': grid_system.x_labels,
                                'y_labels': grid_system.y_labels,
                                'x_coordinates': grid_system.x_coordinates,
                                'y_coordinates': grid_system.y_coordinates,
                                'cell_width': grid_system.cell_width,
                                'cell_height': grid_system.cell_height,
                                'origin_x': grid_system.origin_x,
                                'origin_y': grid_system.origin_y,
                                'scale': grid_system.scale,
                                'confidence': grid_system.confidence
                            }
                            
                            if document_id not in self.grid_cache:
                                self.grid_cache[document_id] = {}
                            self.grid_cache[document_id][page_num] = grid_system
                        
                    except Exception as e:
                        logger.warning(f"Failed to detect grid for page {page_num}: {e}")
                
                # Save grid systems
                if grid_systems:
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=grid_blob,
                        data=json.dumps(grid_systems).encode('utf-8')
                    )
                    logger.info(f"💾 Saved grid systems for {len(grid_systems)} pages")
                    
        except Exception as e:
            logger.error(f"Error loading/detecting grids: {e}")
            # Continue without grids - will use estimated positions
            self.grid_cache[document_id] = {}
    
    async def _detect_grid_on_page(self, page_bytes: bytes, page_num: int) -> Optional[GridSystem]:
        """Detect grid system on a blueprint page using image analysis"""
        try:
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(page_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height, width = gray.shape
            
            # Detect lines using Hough transform
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect horizontal lines
            h_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                      minLineLength=width*0.3, maxLineGap=50)
            
            # Detect vertical lines
            v_lines = cv2.HoughLinesP(edges, 1, np.pi/2, 100, 
                                      minLineLength=height*0.3, maxLineGap=50)
            
            if h_lines is None or v_lines is None:
                # Fallback to default grid
                return self._create_default_grid(page_num, width, height)
            
            # Extract unique X and Y positions
            x_positions = set()
            y_positions = set()
            
            # Process vertical lines for X coordinates
            if v_lines is not None:
                for line in v_lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x1 - x2) < 10:  # Nearly vertical
                        x_positions.add((x1 + x2) // 2)
            
            # Process horizontal lines for Y coordinates
            if h_lines is not None:
                for line in h_lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 10:  # Nearly horizontal
                        y_positions.add((y1 + y2) // 2)
            
            # Sort positions
            x_sorted = sorted(list(x_positions))
            y_sorted = sorted(list(y_positions))
            
            # Filter to get evenly spaced grid lines
            x_grid = self._filter_grid_lines(x_sorted, min_spacing=50)
            y_grid = self._filter_grid_lines(y_sorted, min_spacing=50)
            
            if len(x_grid) < 2 or len(y_grid) < 2:
                return self._create_default_grid(page_num, width, height)
            
            # Detect labels using OCR or pattern matching
            # For now, use simple letter/number labels
            x_labels = [chr(65 + i) for i in range(len(x_grid))]  # A, B, C...
            y_labels = [str(i + 1) for i in range(len(y_grid))]   # 1, 2, 3...
            
            # Create grid system
            grid = GridSystem(
                page_number=page_num,
                x_labels=x_labels,
                y_labels=y_labels,
                confidence=0.8
            )
            
            # Map coordinates
            for i, x in enumerate(x_grid):
                if i < len(x_labels):
                    grid.x_coordinates[x_labels[i]] = x
            
            for i, y in enumerate(y_grid):
                if i < len(y_labels):
                    grid.y_coordinates[y_labels[i]] = y
            
            # Calculate cell dimensions
            if len(x_grid) > 1:
                grid.cell_width = int(np.mean(np.diff(x_grid)))
            if len(y_grid) > 1:
                grid.cell_height = int(np.mean(np.diff(y_grid)))
            
            grid.origin_x = x_grid[0] if x_grid else 100
            grid.origin_y = y_grid[0] if y_grid else 100
            
            logger.info(f"🎯 Detected grid on page {page_num}: {len(x_grid)}x{len(y_grid)}")
            return grid
            
        except Exception as e:
            logger.error(f"Grid detection failed: {e}")
            return self._create_default_grid(page_num, 2000, 2000)
    
    def _filter_grid_lines(self, positions: List[int], min_spacing: int = 50) -> List[int]:
        """Filter positions to get evenly spaced grid lines"""
        if not positions:
            return []
        
        filtered = [positions[0]]
        for pos in positions[1:]:
            if pos - filtered[-1] >= min_spacing:
                filtered.append(pos)
        
        return filtered
    
    def _create_default_grid(self, page_num: int, width: int, height: int) -> GridSystem:
        """Create a default grid system for pages where detection fails"""
        # Create a reasonable default grid
        num_x = min(20, width // 100)
        num_y = min(30, height // 100)
        
        x_labels = [chr(65 + i) for i in range(num_x)]  # A-T
        y_labels = [str(i + 1) for i in range(num_y)]   # 1-30
        
        grid = GridSystem(
            page_number=page_num,
            x_labels=x_labels,
            y_labels=y_labels,
            cell_width=width // (num_x + 2),
            cell_height=height // (num_y + 2),
            origin_x=100,
            origin_y=100,
            confidence=0.3  # Low confidence for default grid
        )
        
        # Set coordinates
        for i, label in enumerate(x_labels):
            grid.x_coordinates[label] = grid.origin_x + (i * grid.cell_width)
        
        for i, label in enumerate(y_labels):
            grid.y_coordinates[label] = grid.origin_y + (i * grid.cell_height)
        
        return grid
    
    async def _analyze_blueprint_with_visual_highlighting(
        self, prompt: str, document_text: str, image_urls: List[Dict[str, any]], 
        document_id: str, author: str, current_page: Optional[int],
        request_highlights: bool, reference_previous: Optional[List[str]],
        preserve_existing: bool, storage_service: StorageService,
        metadata: Optional[Dict]) -> Dict[str, Any]:
        """Analyze blueprint and create visual highlights across ALL pages"""
        try:
            logger.info("="*50)
            logger.info("📊 VISUAL BLUEPRINT ANALYSIS")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"📑 Total pages: {len(image_urls)}")
            logger.info("="*50)
            
            # Get system message for comprehensive analysis
            system_message = self._get_multi_page_system_message()
            
            messages = [system_message]
            
            # Build user message with all pages
            user_message = {"role": "user", "content": []}
            
            # Add ALL page images for analysis
            if image_urls:
                for i, page_data in enumerate(image_urls):
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {"url": page_data["url"], "detail": "high"}
                    })
            
            # Build comprehensive query
            query_text = self._build_multi_page_query(
                prompt, document_id, document_text, 
                len(image_urls), metadata
            )
            
            user_message["content"].append({"type": "text", "text": query_text})
            messages.append(user_message)
            
            logger.info("📤 Requesting multi-page analysis")
            
            # Get AI response
            response = await self._get_ai_completion(messages)
            
            # Parse response and create visual highlights
            if request_highlights:
                query_session_id = str(uuid.uuid4())
                parsed_response = await self._create_visual_highlights(
                    response, document_id, query_session_id, 
                    reference_previous, preserve_existing, storage_service
                )
                parsed_response['query_session_id'] = query_session_id
                
                # Generate highlighted images
                if parsed_response.get('all_highlights'):
                    highlighted_urls = await self._generate_highlighted_images(
                        document_id, parsed_response['all_highlights'], 
                        storage_service
                    )
                    parsed_response['highlighted_image_urls'] = highlighted_urls
            else:
                parsed_response = {
                    "ai_response": response,
                    "visual_highlights": None,
                    "query_session_id": None
                }
            
            # Add current page and highlight summary
            parsed_response['current_page'] = current_page
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "ai_response": f"Error performing analysis: {str(e)}",
                "visual_highlights": None
            }
    
    async def _create_visual_highlights(self, response: str, document_id: str,
                                       query_session_id: str,
                                       reference_previous: Optional[List[str]],
                                       preserve_existing: bool,
                                       storage_service: StorageService) -> Dict[str, Any]:
        """Parse response and create visual highlights with pixel coordinates"""
        
        # Parse response for element locations
        all_highlights = []
        pages_with_highlights = {}
        
        # Extract structured element data from response
        elements_section = re.search(r'ELEMENTS_FOUND:(.*?)(?=TOTAL:|$)', response, re.DOTALL)
        
        if elements_section:
            elements_text = elements_section.group(1)
            
            # Parse page-by-page findings
            page_pattern = r'Page\s+(\d+):\s*(.*?)(?=Page\s+\d+:|$)'
            page_matches = re.finditer(page_pattern, elements_text, re.DOTALL)
            
            for match in page_matches:
                page_num = int(match.group(1))
                page_content = match.group(2)
                
                # Get grid system for this page
                grid_system = self.grid_cache.get(document_id, {}).get(page_num)
                
                # Extract elements from this page
                page_elements = self._parse_page_elements(page_content, page_num)
                
                for element in page_elements:
                    # Convert grid reference to pixel coordinates
                    pixel_coords = None
                    if grid_system:
                        pixel_coords = grid_system.get_pixel_coords(
                            element['grid_reference']
                        )
                    
                    if not pixel_coords:
                        # Estimate coordinates if grid not available
                        pixel_coords = self._estimate_element_position(
                            element['grid_reference'], page_num
                        )
                    
                    highlight_data = {
                        "annotation_id": str(uuid.uuid4())[:8],
                        "document_id": document_id,
                        "page_number": page_num,
                        "element_type": element['element_type'],
                        "grid_reference": element['grid_reference'],
                        "label": element.get('label', ''),
                        "x": pixel_coords['x'],
                        "y": pixel_coords['y'],
                        "width": pixel_coords['width'],
                        "height": pixel_coords['height'],
                        "pixel_coordinates": pixel_coords,  # Full coordinate data
                        "text": element.get('description', ''),
                        "annotation_type": "ai_highlight",
                        "author": "ai_system",
                        "is_private": False,
                        "query_session_id": query_session_id,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "confidence": element.get('confidence', 0.9)
                    }
                    
                    all_highlights.append(highlight_data)
                
                pages_with_highlights[page_num] = len(page_elements)
        
        # Handle reference_previous - add existing highlights
        if reference_previous:
            previous_highlights = await self._get_previous_highlights(
                document_id, reference_previous, storage_service
            )
            
            # Update their session ID if not preserving
            if not preserve_existing:
                for ph in previous_highlights:
                    ph['query_session_id'] = query_session_id
            
            all_highlights.extend(previous_highlights)
        
        # Save all highlights
        if all_highlights:
            # Import here to avoid circular dependency
            from app.api.routes.annotation_routes import save_all_annotations, load_all_annotations
            
            # Load existing annotations
            existing_annotations = await load_all_annotations(document_id, storage_service)
            
            # Clear previous AI highlights unless preserving
            if not preserve_existing:
                existing_annotations = [
                    ann for ann in existing_annotations
                    if ann.get('annotation_type') != 'ai_highlight'
                ]
            
            # Add new highlights
            existing_annotations.extend(all_highlights)
            
            # Save back
            await save_all_annotations(document_id, existing_annotations, storage_service)
            
            logger.info(f"💾 Saved {len(all_highlights)} visual highlights across {len(pages_with_highlights)} pages")
        
        # Clean response text
        clean_response = response
        if elements_section:
            clean_response = clean_response.replace(elements_section.group(0), '')
        
        # Return current page highlights only for display
        current_page_highlights = None
        if all_highlights and 'current_page' in locals():
            # Convert to VisualElement format for current page
            current_page_highlights = []
            for h in all_highlights:
                if h.get('page_number') == current_page:
                    current_page_highlights.append(
                        VisualElement(
                            element_id=h['annotation_id'],
                            element_type=h['element_type'],
                            grid_location=GridReference(
                                grid_ref=h['grid_reference'],
                                x_grid=h['grid_reference'].split('-')[0] if '-' in h['grid_reference'] else h['grid_reference'],
                                y_grid=h['grid_reference'].split('-')[1] if '-' in h['grid_reference'] else None
                            ),
                            label=h['label'],
                            page_number=h['page_number'],
                            confidence=h.get('confidence', 0.9)
                        )
                    )
        
        return {
            "ai_response": clean_response.strip(),
            "visual_highlights": current_page_highlights,
            "all_highlights": all_highlights,  # Include full highlight data
            "all_highlight_pages": pages_with_highlights,
            "total_highlights_created": len(all_highlights)
        }
    
    def _estimate_element_position(self, grid_ref: str, page_num: int) -> Dict[str, int]:
        """Estimate element position when grid system not available"""
        # Default estimation based on typical blueprint layout
        x_base = 200
        y_base = 200
        cell_width = 100
        cell_height = 100
        
        # Try to parse grid reference
        match = re.match(r'([A-Z]+)[-\s]?(\d+)', grid_ref.upper())
        if match:
            letter = match.group(1)
            number = match.group(2)
            
            # Calculate position
            if len(letter) == 1:
                x = x_base + (ord(letter) - ord('A')) * cell_width
            else:
                x = x_base
            
            y = y_base + (int(number) - 1) * cell_height
            
            return {
                "x": x,
                "y": y,
                "width": cell_width,
                "height": cell_height,
                "center_x": x + cell_width // 2,
                "center_y": y + cell_height // 2
            }
        
        # Fallback to center of page
        return {
            "x": 1000,
            "y": 1000,
            "width": 150,
            "height": 150,
            "center_x": 1075,
            "center_y": 1075
        }
    
    async def _generate_highlighted_images(self, document_id: str, 
                                          highlights: List[Dict],
                                          storage_service: StorageService) -> Dict[int, str]:
        """Generate new images with visual highlights drawn on them"""
        highlighted_urls = {}
        
        # Group highlights by page
        highlights_by_page = defaultdict(list)
        for h in highlights:
            highlights_by_page[h['page_number']].append(h)
        
        # Process each page with highlights
        for page_num, page_highlights in highlights_by_page.items():
            try:
                # Load original page image
                page_blob = f"{document_id}_page_{page_num}.png"
                page_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=page_blob
                )
                
                # Create highlighted version
                highlighted_bytes = await self._create_highlighted_image(
                    page_bytes, page_highlights
                )
                
                # Save highlighted image
                highlighted_blob = f"{document_id}_page_{page_num}_highlighted_{highlights[0]['query_session_id']}.png"
                url = await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=highlighted_blob,
                    data=highlighted_bytes
                )
                
                highlighted_urls[page_num] = url
                logger.info(f"✅ Created highlighted image for page {page_num}")
                
            except Exception as e:
                logger.error(f"Failed to create highlighted image for page {page_num}: {e}")
        
        return highlighted_urls
    
    async def _create_highlighted_image(self, page_bytes: bytes, 
                                       highlights: List[Dict]) -> bytes:
        """Create a new image with highlight boxes drawn on it"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._draw_highlights_on_image,
            page_bytes,
            highlights
        )
    
    def _draw_highlights_on_image(self, page_bytes: bytes, 
                                  highlights: List[Dict]) -> bytes:
        """Draw highlight boxes and labels on image"""
        try:
            # Open image
            img = Image.open(io.BytesIO(page_bytes))
            
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create overlay for highlights
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", self.label_font_size)
            except:
                font = ImageFont.load_default()
            
            # Draw each highlight
            for highlight in highlights:
                coords = highlight.get('pixel_coordinates', {})
                if not coords:
                    continue
                
                x = coords.get('x', 0)
                y = coords.get('y', 0)
                width = coords.get('width', 100)
                height = coords.get('height', 100)
                
                # Draw semi-transparent rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=self.highlight_color,
                    outline=self.highlight_border,
                    width=self.highlight_border_width
                )
                
                # Add label with background
                label = f"{highlight['element_type']}: {highlight.get('label', highlight['grid_reference'])}"
                
                # Get text bbox
                try:
                    bbox = draw.textbbox((x, y - 25), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    # Fallback for older PIL versions
                    text_width, text_height = draw.textsize(label, font=font)
                
                # Draw text background
                padding = 4
                draw.rectangle(
                    [x - padding, y - 25 - padding, 
                     x + text_width + padding, y - 25 + text_height + padding],
                    fill=(255, 255, 255, 200)  # White background
                )
                
                # Draw text
                draw.text(
                    (x, y - 25),
                    label,
                    fill=(255, 0, 0, 255),  # Red text
                    font=font
                )
            
            # Composite overlay onto original image
            img = Image.alpha_composite(img, overlay)
            
            # Convert back to RGB for saving
            final_img = Image.new('RGB', img.size, (255, 255, 255))
            final_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            
            # Save to bytes
            output = io.BytesIO()
            final_img.save(output, format='PNG', optimize=True, compress_level=6)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error drawing highlights: {e}")
            # Return original image if highlighting fails
            return page_bytes
    
    # ===== KEEP ALL EXISTING METHODS FROM ORIGINAL =====
    
    def _needs_visual_analysis(self, prompt: str, reference_previous: Optional[List[str]]) -> bool:
        """Determine if we need to analyze pages for new elements"""
        # If just referencing previous, no new analysis needed
        if reference_previous and not self._contains_new_element_request(prompt):
            return False
        
        # Check if prompt asks for visual elements
        visual_keywords = [
            'show', 'highlight', 'where', 'locate', 'find', 'identify',
            'point out', 'mark', 'indicate', 'display', 'how many', 'count'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in visual_keywords)
    
    def _contains_new_element_request(self, prompt: str) -> bool:
        """Check if prompt requests new element types beyond reference_previous"""
        prompt_lower = prompt.lower()
        
        # Check for element types in prompt
        for element_type in self.element_patterns.keys():
            if element_type in prompt_lower:
                return True
        
        return False
    
    async def _get_previous_highlights(self, document_id: str, element_types: List[str],
                                      storage_service: StorageService) -> List[Dict]:
        """Retrieve previously saved highlights for specified element types"""
        try:
            # Load all annotations
            annotations_blob = f"{document_id}_annotations.json"
            annotations_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=annotations_blob
            )
            all_annotations = json.loads(annotations_data)
            
            # Filter to requested element types
            previous_highlights = [
                ann for ann in all_annotations
                if ann.get('annotation_type') == 'ai_highlight' and
                   ann.get('element_type') in element_types
            ]
            
            logger.info(f"✅ Retrieved {len(previous_highlights)} previous highlights")
            return previous_highlights
            
        except Exception as e:
            logger.info(f"No previous highlights found: {e}")
            return []
    
    def _get_multi_page_system_message(self) -> Dict[str, str]:
        """System message for analyzing entire document"""
        return {
            "role": "system",
            "content": """You are a professional blueprint analyst examining construction documents. 

CRITICAL INSTRUCTIONS FOR MULTI-PAGE ANALYSIS:

1. ALWAYS provide complete, professional answers with code references
2. When identifying elements (windows, doors, columns, etc.), analyze ALL pages provided
3. For each element found, note:
   - Page number where it appears
   - Grid location (e.g., "W2-WA", "Grid B-3", "A-5")
   - Element label/ID if visible
   - Quantity and dimensions

4. FORMAT YOUR FINDINGS:
When listing elements across pages, use this format:
```
ELEMENTS_FOUND:
Page 3: 8 windows (Type 1 at B-2, Type 2 at E-5...)
Page 4: 6 windows (Type 1 at B-2, Type 3 at F-7...)
Page 5: 8 windows (Type 1 at B-2, Type 2 at E-5...)

TOTAL: 22 windows across 3 pages
```

5. BUILDING CODES:
- Reference BCBC 2018 (British Columbia Building Code)
- Include CSA standards where applicable
- Mention local Burnaby requirements when relevant

6. ALWAYS SUGGEST FOLLOW-UP:
End responses with helpful options like:
"Would you like me to:
1. Show specific element types on other levels?
2. Calculate quantities for specific trades?
3. Identify potential conflicts?"

Remember: You're helping construction professionals who need accurate counts, locations, and code compliance information."""
        }
    
    def _build_multi_page_query(self, prompt: str, document_id: str, 
                                document_text: str, page_count: int,
                                metadata: Optional[Dict]) -> str:
        """Build query for multi-page analysis"""
        
        # Extract document info from metadata
        doc_info = ""
        if metadata:
            if 'document_info' in metadata:
                doc_info = f"Project: {metadata['document_info'].get('title', 'Unknown')}\n"
            if 'page_details' in metadata:
                sheet_info = []
                for page in metadata['page_details']:
                    if page.get('sheet_number'):
                        sheet_info.append(f"Page {page['page_number']}: Sheet {page['sheet_number']} - {page.get('drawing_type', 'Unknown type')}")
                if sheet_info:
                    doc_info += "Drawing sheets:\n" + "\n".join(sheet_info[:10])  # First 10
        
        query = f"""Document: {document_id}
Total pages provided: {page_count}

{doc_info}

User Question: {prompt}

IMPORTANT: 
- Analyze ALL {page_count} pages provided
- Identify requested elements on EVERY page where they appear
- Provide total counts across all pages
- Note page numbers and grid locations for each finding

Drawing text content available: {'Yes' if document_text else 'No'}"""
        
        if document_text:
            query += f"\n\nText extracted from drawings:\n{document_text[:2000]}..."
        
        return query
    
    def _parse_page_elements(self, page_content: str, page_num: int) -> List[Dict]:
        """Parse elements from page description"""
        elements = []
        
        # Common patterns for element detection
        patterns = [
            # "Type 1 at B-2"
            r'Type\s+(\d+)\s+at\s+([A-Z0-9-]+)',
            # "8 windows (details)"
            r'(\d+)\s+(\w+)\s+\((.*?)\)',
            # "CB-301 at grid W2-WA"
            r'(CB-\d+)\s+at\s+(?:grid\s+)?([A-Z0-9-]+)',
            # Generic "element at location"
            r'(\w+)\s+at\s+(?:grid\s+)?([A-Z0-9-]+)'
        ]
        
        content_lower = page_content.lower()
        
        # Determine element type from content
        element_type = 'unknown'
        for etype, keywords in [
            ('window', ['window', 'glazing', 'type']),
            ('door', ['door', 'entrance']),
            ('catch_basin', ['cb', 'catch basin', 'drain']),
            ('column', ['column', 'col']),
            ('sprinkler', ['sprinkler', 'sp'])
        ]:
            if any(kw in content_lower for kw in keywords):
                element_type = etype
                break
        
        # Extract grid locations
        grid_pattern = r'[A-Z]\d+[-\s]?[A-Z]*\d*|[A-Z][-\s]\d+|\d+[-\s][A-Z]'
        grid_matches = re.finditer(grid_pattern, page_content)
        
        for i, match in enumerate(grid_matches):
            elements.append({
                'element_type': element_type,
                'grid_reference': match.group(0),
                'label': f"{element_type}_{page_num}_{i+1}",
                'confidence': 0.9,
                'description': f"{element_type} on page {page_num}"
            })
        
        return elements
    
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
            "multi_page_features": [
                "Analyzes entire document at once",
                "Finds elements across all pages",
                "Creates persistent highlights",
                "Reuses previous analysis",
                "Provides page-by-page breakdowns"
            ],
            "visual_features": [
                "Grid system detection",
                "Element location identification",
                "Multi-page highlighting",
                "Reference previous highlights",
                "Drawing scale recognition",
                "Visual highlight overlays",  # NEW
                "Pixel coordinate mapping"     # NEW
            ],
            "optimization_features": [
                f"Handles up to {self.max_pages_to_load} pages",
                f"Parallel loading in batches of {self.batch_size}",
                "Image compression for efficiency",
                "Highlight reuse system",
                "Optimized for large documents",
                "Visual highlight generation"   # NEW
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
