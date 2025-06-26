# app/services/ai_service.py - COMPLETE MULTI-PAGE HIGHLIGHTING SYSTEM

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
import uuid

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
    """Professional AI service with multi-page grid-based visual element highlighting"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Professional Blueprint AI initialized with multi-page highlighting")
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
        
        # Element patterns for detection
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
        
        logger.info(f"   📄 Max pages: {self.max_pages_to_load}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   🖼️ Image quality: {self.image_quality}%")
        logger.info(f"   🎯 Multi-page highlighting: Enabled")
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, 
                              author: str = None,
                              current_page: Optional[int] = None,
                              request_highlights: bool = True,
                              reference_previous: Optional[List[str]] = None,
                              preserve_existing: bool = False) -> Dict[str, Any]:
        """Process blueprint queries with multi-page highlighting support"""
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
            result = await self._analyze_blueprint_with_multi_page_highlighting(
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
    
    async def _analyze_blueprint_with_multi_page_highlighting(
        self, prompt: str, document_text: str, image_urls: List[Dict[str, any]], 
        document_id: str, author: str, current_page: Optional[int],
        request_highlights: bool, reference_previous: Optional[List[str]],
        preserve_existing: bool, storage_service: StorageService,
        metadata: Optional[Dict]) -> Dict[str, Any]:
        """Analyze blueprint and create highlights across ALL pages"""
        try:
            logger.info("="*50)
            logger.info("📊 MULTI-PAGE BLUEPRINT ANALYSIS")
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
            
            # Parse response and create highlights for ALL pages
            if request_highlights:
                query_session_id = str(uuid.uuid4())
                parsed_response = await self._create_multi_page_highlights(
                    response, document_id, query_session_id, 
                    reference_previous, preserve_existing, storage_service
                )
                parsed_response['query_session_id'] = query_session_id
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
   - Grid location (e.g., "W2-WA", "Grid B-3")
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
- Note page numbers for each finding

Drawing text content available: {'Yes' if document_text else 'No'}"""
        
        if document_text:
            query += f"\n\nText extracted from drawings:\n{document_text[:2000]}..."
        
        return query
    
    async def _create_multi_page_highlights(self, response: str, document_id: str,
                                           query_session_id: str,
                                           reference_previous: Optional[List[str]],
                                           preserve_existing: bool,
                                           storage_service: StorageService) -> Dict[str, Any]:
        """Parse response and create highlights across multiple pages"""
        
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
                
                # Extract elements from this page
                page_elements = self._parse_page_elements(page_content, page_num)
                
                for element in page_elements:
                    all_highlights.append({
                        "annotation_id": str(uuid.uuid4())[:8],
                        "document_id": document_id,
                        "page_number": page_num,
                        "element_type": element['element_type'],
                        "grid_reference": element['grid_reference'],
                        "label": element.get('label', ''),
                        "x": 0,  # Grid-based, not pixel
                        "y": 0,
                        "text": element.get('description', ''),
                        "annotation_type": "ai_highlight",
                        "author": "ai_system",
                        "is_private": False,
                        "query_session_id": query_session_id,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "confidence": element.get('confidence', 0.9)
                    })
                
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
        
        # Save all highlights using annotation route
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
            
            logger.info(f"💾 Saved {len(all_highlights)} highlights across {len(pages_with_highlights)} pages")
        
        # Clean response text
        clean_response = response
        if elements_section:
            clean_response = clean_response.replace(elements_section.group(0), '')
        
        # Return current page highlights only for display
        current_page_highlights = None
        if all_highlights:
            # Convert to VisualElement format for current page
            current_page_highlights = []
            for h in all_highlights:
                if h.get('page_number') == h.get('current_page', 1):
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
                            page_number=h['page_number']
                        )
                    )
        
        return {
            "ai_response": clean_response.strip(),
            "visual_highlights": current_page_highlights,
            "all_highlight_pages": pages_with_highlights,
            "total_highlights_created": len(all_highlights)
        }
    
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
    
    # --- Keep all existing helper methods ---
    
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
                "Drawing scale recognition"
            ],
            "optimization_features": [
                f"Handles up to {self.max_pages_to_load} pages",
                f"Parallel loading in batches of {self.batch_size}",
                "Image compression for efficiency",
                "Highlight reuse system",
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
