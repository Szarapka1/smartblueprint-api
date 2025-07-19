# app/services/pdf_service.py - PRODUCTION-GRADE PDF PROCESSING (REVISED FOR ACCURACY)

import logging
import fitz  # PyMuPDF
import json
import os
import asyncio
import gc
from PIL import Image
import io
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import time

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass
class GridSystem:
    """Grid system for a blueprint page with proper validation"""
    page_number: int
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    x_coordinates: Dict[str, int] = field(default_factory=dict)
    y_coordinates: Dict[str, int] = field(default_factory=dict)
    x_lines: List[int] = field(default_factory=list)
    y_lines: List[int] = field(default_factory=list)
    cell_width: int = 100
    cell_height: int = 100
    origin_x: int = 100
    origin_y: int = 100
    confidence: float = 0.0
    scale: Optional[str] = None
    detection_method: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'page_number': self.page_number,
            'x_labels': self.x_labels,
            'y_labels': self.y_labels,
            'x_coordinates': {str(k): v for k, v in self.x_coordinates.items()},
            'y_coordinates': {str(k): v for k, v in self.y_coordinates.items()},
            'x_lines': self.x_lines,
            'y_lines': self.y_lines,
            'cell_width': self.cell_width,
            'cell_height': self.cell_height,
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'confidence': self.confidence,
            'scale': self.scale,
            'detection_method': self.detection_method
        }


class PDFService:
    """Production-grade PDF processing with enhanced grid detection and robustness"""

    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")

        self.settings = settings
        self._lock = asyncio.Lock()

        # Resolution settings
        self.storage_dpi = settings.PDF_HIGH_RESOLUTION
        self.ai_image_dpi = settings.PDF_AI_DPI
        self.thumbnail_dpi = settings.PDF_THUMBNAIL_DPI

        # Processing settings
        self.max_pages = settings.PDF_MAX_PAGES
        self.batch_size = settings.PROCESSING_BATCH_SIZE

        # Image optimization
        self.jpeg_quality = settings.PDF_JPEG_QUALITY
        self.ai_max_dimension = settings.AI_MAX_IMAGE_DIMENSION

        # Text extraction and grid detection settings
        self.max_text_per_page = 150000  # Increased limit for comprehensive text
        self.enable_grid_detection = True
        self.grid_line_tolerance = 5  # Tolerance in pixels for clustering grid lines

        # Memory management
        self.gc_frequency = 3
        self.processing_delay = 0.25  # Reduced delay for faster testing

        logger.info("✅ PDFService initialized (REVISED FOR ACCURACY)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ DPI: AI={self.ai_image_dpi}, Thumbnail={self.thumbnail_dpi}")
        logger.info(f"   📦 Batch size: {self.batch_size} pages")
        logger.info(f"   🔍 Grid Detection: VISUAL_FIRST")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes,
                                   storage_service: StorageService):
        """Process PDF with robust error handling, memory management, and advanced grid detection"""
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session ID")
        if not pdf_bytes or not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b'%PDF'):
            raise ValueError("Invalid PDF data provided")

        async with self._lock:
            try:
                logger.info(f"🚀 Starting PDF processing for: {session_id}")
                pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
                logger.info(f"📄 File Size: {pdf_size_mb:.1f}MB")

                if pdf_size_mb > self.settings.MAX_FILE_SIZE_MB:
                    raise ValueError(f"PDF too large: {pdf_size_mb:.1f}MB (max: {self.settings.MAX_FILE_SIZE_MB}MB)")

                processing_start = time.time()

                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                except Exception as e:
                    raise RuntimeError(f"Failed to open PDF stream: {e}")

                try:
                    total_pages = len(doc)
                    pages_to_process = min(total_pages, self.max_pages)
                    logger.info(f"📄 Processing {pages_to_process} of {total_pages} pages")

                    metadata = self._initialize_metadata(session_id, doc, pages_to_process, total_pages, pdf_size_mb)
                    all_text_parts, all_grid_systems, pages_processed = [], {}, 0

                    for batch_start in range(0, pages_to_process, self.batch_size):
                        batch_end = min(batch_start + self.batch_size, pages_to_process)
                        batch_pages = list(range(batch_start, batch_end))
                        logger.info(f"📦 Processing batch: pages {batch_start + 1}-{batch_end}")

                        batch_results = await self._process_batch_safe(doc, batch_pages, session_id, storage_service)

                        for result in batch_results:
                            if result.get('success'):
                                all_text_parts.append(result['text'])
                                metadata['page_details'].append(result['metadata'])
                                if result.get('grid_system'):
                                    page_num = result['metadata']['page_number']
                                    all_grid_systems[str(page_num)] = result['grid_system'].to_dict()
                                    metadata['extraction_summary']['has_grid_systems'] = True
                                self._update_extraction_summary(metadata, result)
                                pages_processed += 1
                        
                        gc.collect()
                        await asyncio.sleep(self.processing_delay)
                        logger.info(f"📊 Progress: {(pages_processed / pages_to_process) * 100:.1f}%")

                    await self._save_processing_results(
                        session_id, all_text_parts, all_grid_systems, metadata,
                        processing_start, storage_service
                    )

                    logger.info(f"✅ Processing complete for {session_id} in {metadata['processing_time']:.2f}s")
                finally:
                    doc.close()
                    gc.collect()
            except Exception as e:
                logger.error(f"❌ Processing failed: {e}", exc_info=True)
                await self._save_error_state(session_id, str(e), storage_service)
                raise RuntimeError(f"PDF processing failed: {str(e)}")

    # --- Grid Detection Overhaul ---

    def _detect_grid_patterns(self, page: fitz.Page, page_text: str, page_num: int) -> GridSystem:
        """
        Detects grid systems using a robust, visual-first approach.
        Always returns a GridSystem object to prevent downstream errors.
        """
        # 1. Attempt Visual Detection (High Confidence)
        try:
            grid = self._detect_grid_visually(page, page_num)
            if grid.confidence > 0.6:  # High confidence from visual analysis
                logger.info(f"🎯 [Page {page_num}] High-confidence grid found visually: {len(grid.x_labels)}x{len(grid.y_labels)}")
                grid.detection_method = "visual"
                return grid
        except Exception as e:
            logger.warning(f"Visual grid detection failed for page {page_num}: {e}")
            grid = GridSystem(page_number=page_num) # Start with a blank grid

        # 2. Fallback to Text-Based Detection (Lower Confidence)
        try:
            text_grid = self._detect_grid_by_text(page_text, page_num)
            if text_grid.confidence > grid.confidence: # If text is better, use it
                logger.info(f"🎯 [Page {page_num}] Grid found via text analysis: {len(text_grid.x_labels)}x{len(text_grid.y_labels)}")
                text_grid.detection_method = "text"
                return text_grid
        except Exception as e:
            logger.warning(f"Text grid detection failed for page {page_num}: {e}")

        # 3. Return best effort or an empty grid object
        logger.warning(f"⚠️ [Page {page_num}] Low-confidence or no grid system detected. Returning best effort.")
        grid.detection_method = "none" if grid.confidence == 0.0 else grid.detection_method
        return grid

    def _detect_grid_visually(self, page: fitz.Page, page_num: int) -> GridSystem:
        """Analyzes drawing vectors to find grid lines."""
        paths = page.get_drawings()
        horiz_lines, vert_lines = [], []
        page_width, page_height = page.rect.width, page.rect.height

        for path in paths:
            for item in path["items"]:
                if item[0] == "l":  # Line item
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < self.grid_line_tolerance and abs(p1.x - p2.x) > page_width * 0.1:
                        horiz_lines.append(p1.y)
                    elif abs(p1.x - p2.x) < self.grid_line_tolerance and abs(p1.y - p2.y) > page_height * 0.1:
                        vert_lines.append(p1.x)

        if not vert_lines or not horiz_lines:
            return GridSystem(page_number=page_num, confidence=0.0)

        # Cluster and average line positions to find the main grid
        final_vert = self._cluster_lines(vert_lines)
        final_horiz = self._cluster_lines(horiz_lines)

        if len(final_vert) < 2 or len(final_horiz) < 2:
            return GridSystem(page_number=page_num, confidence=0.2) # Not enough lines for a grid

        # For now, labels are generic, but can be enhanced by OCR'ing text near line ends
        grid = GridSystem(
            page_number=page_num,
            x_labels=[chr(65 + i) for i in range(len(final_vert))],
            y_labels=[str(i + 1) for i in range(len(final_horiz))],
            x_lines=[int(x) for x in sorted(final_vert)],
            y_lines=[int(y) for y in sorted(final_horiz)],
            confidence=min(0.9, (len(final_vert) * len(final_horiz)) / 100.0)
        )
        return grid

    def _cluster_lines(self, lines: List[float]) -> List[float]:
        """Clusters close lines together to find the primary grid lines."""
        if not lines:
            return []
        lines.sort()
        clusters = []
        current_cluster = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - current_cluster[-1] <= self.grid_line_tolerance:
                current_cluster.append(lines[i])
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [lines[i]]
        clusters.append(sum(current_cluster) / len(current_cluster))
        return clusters

    def _detect_grid_by_text(self, page_text: str, page_num: int) -> GridSystem:
        """Original text-based grid detection as a fallback."""
        x_refs, y_refs = set(), set()
        search_text = page_text[:15000]
        grid_patterns = {
            'grid_ref': re.compile(r'([A-Z]+)[-/]([0-9]+)', re.IGNORECASE),
            'column': re.compile(r'(?:COLUMN|COL\.?|C\.?)\s*([A-Z]+)(?:\s|$)', re.IGNORECASE),
            'row': re.compile(r'(?:ROW|R\.?)\s*([0-9]+)', re.IGNORECASE)
        }
        for p_name, p in grid_patterns.items():
            for match in p.finditer(search_text):
                if p_name == 'grid_ref':
                    x_refs.add(match.group(1).upper())
                    y_refs.add(match.group(2))
                elif p_name == 'column':
                    x_refs.add(match.group(1).upper())
                elif p_name == 'row':
                    y_refs.add(match.group(1))

        if not x_refs or not y_refs:
            return GridSystem(page_number=page_num, confidence=0.0)

        return GridSystem(
            page_number=page_num,
            x_labels=sorted(list(x_refs))[:30],
            y_labels=sorted(list(y_refs), key=lambda x: int(x))[:50],
            confidence=0.5
        )

    # --- Core Processing Logic (Mostly Unchanged but with Minor Enhancements) ---

    async def _process_single_page_safe(self, doc: fitz.Document, page_num: int,
                                       session_id: str, storage_service: StorageService) -> Dict:
        """Process a single page, now using the robust grid detection."""
        try:
            page = doc[page_num]
            page_actual = page_num + 1

            page_text = page.get_text("text")[:self.max_text_per_page]
            page_analysis = self._analyze_page_content(page_text, page_actual)

            # Use the new robust grid detection
            grid_system = self._detect_grid_patterns(page, page_text, page_actual)

            await self._generate_and_upload_page_images_safe(page, page_actual, session_id, storage_service)

            page_metadata = {
                'page_number': page_actual,
                'text_length': len(page_text),
                'drawing_type': page_analysis.get('drawing_type'),
                'sheet_number': page_analysis.get('sheet_number'),
                'scale': page_analysis.get('scale') or grid_system.scale,
                'has_grid': grid_system.confidence > 0.3
            }

            formatted_text = self._format_page_text(page_actual, page_analysis, grid_system, page_text)
            page.clean_contents()

            return {
                'success': True,
                'text': formatted_text,
                'metadata': page_metadata,
                'grid_system': grid_system,
            }
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}", exc_info=True)
            raise

    async def _save_processing_results(self, session_id: str, all_text_parts: List[str],
                                     all_grid_systems: Dict[str, Any], metadata: Dict[str, Any],
                                     processing_start: float, storage_service: StorageService):
        """Save all processing results, including the now-guaranteed grid file."""
        try:
            full_text = '\n'.join(all_text_parts)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_context.txt",
                data=full_text.encode('utf-8'),
                content_type="text/plain"
            )

            # Save grid systems file. It will contain all detected grids or be an empty object.
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_grid_systems.json",
                data=json.dumps(all_grid_systems, indent=2).encode('utf-8'),
                content_type="application/json"
            )

            await self._create_document_index(session_id, metadata['page_details'], storage_service)

            metadata['processing_time'] = time.time() - processing_start
            metadata['grid_systems_detected'] = len(all_grid_systems)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json",
                data=json.dumps(metadata, indent=2).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to save processing results: {e}", exc_info=True)
            raise

    # Other helper methods (_initialize_metadata, _update_extraction_summary, _process_batch_safe,
    # _format_page_text, _generate_and_upload_page_images_safe, etc.) would be here.
    # They are largely unchanged from your provided file but I will include them for completeness.

    def _initialize_metadata(self, session_id: str, doc: fitz.Document,
                           pages_to_process: int, total_pages: int,
                           pdf_size_mb: float) -> Dict[str, Any]:
        """Initialize metadata structure"""
        return {
            'document_id': session_id,
            'page_count': pages_to_process,
            'total_pages': total_pages,
            'document_info': dict(doc.metadata) if hasattr(doc, 'metadata') else {},
            'processing_time': 0,
            'file_size_mb': round(pdf_size_mb, 2),
            'page_details': [],
            'grid_detection_enabled': self.enable_grid_detection,
            'extraction_summary': {
                'has_text': False,
                'has_images': True,
                'has_grid_systems': False,
            }
        }

    def _update_extraction_summary(self, metadata: Dict[str, Any], result: Dict[str, Any]):
        """Update extraction summary based on page results"""
        if result.get('text', '').strip():
            metadata['extraction_summary']['has_text'] = True

    async def _process_batch_safe(self, doc: fitz.Document, page_numbers: List[int],
                                 session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process batch with error recovery"""
        tasks = [self._process_single_page_safe(doc, num, session_id, storage_service) for num in page_numbers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                processed_results.append({
                    'success': False, 'page_num': page_numbers[i] + 1, 'error': str(res),
                    'text': '', 'metadata': {'page_number': page_numbers[i] + 1}
                })
            else:
                processed_results.append(res)
        return processed_results


    def _format_page_text(self, page_num: int, page_analysis: Dict[str, Any],
                         grid_system: Optional[GridSystem], page_text: str) -> str:
        """Format page text with metadata"""
        header = [f"\n--- PAGE {page_num} ---"]
        if page_analysis.get('sheet_number'): header.append(f"Sheet: {page_analysis['sheet_number']}")
        if page_analysis.get('drawing_type'): header.append(f"Type: {page_analysis['drawing_type']}")
        if grid_system and grid_system.confidence > 0.3: header.append(f"Grid: {len(grid_system.x_labels)}x{len(grid_system.y_labels)}")
        if page_analysis.get('scale'): header.append(f"Scale: {page_analysis['scale']}")
        return "\n".join(header) + f"\n\n{page_text}"

    async def _generate_and_upload_page_images_safe(self, page: fitz.Page, page_num: int,
                                                   session_id: str, storage_service: StorageService):
        """Generate and upload all required images for a page."""
        try:
            # AI optimized (JPEG)
            await self._generate_ai_image(page, page_num, session_id, storage_service)
            # Thumbnail (JPEG)
            await self._generate_thumbnail(page, page_num, session_id, storage_service)
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")

    async def _generate_ai_image(self, page: fitz.Page, page_num: int,
                                session_id: str, storage_service: StorageService):
        """Generate AI-optimized image."""
        ai_matrix = fitz.Matrix(self.ai_image_dpi / 72, self.ai_image_dpi / 72)
        pix = page.get_pixmap(matrix=ai_matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if max(img.size) > self.ai_max_dimension:
            img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=self.jpeg_quality, optimize=True)
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_page_{page_num}_ai.jpg",
            data=output.getvalue(), content_type="image/jpeg"
        )

    async def _generate_thumbnail(self, page: fitz.Page, page_num: int,
                                 session_id: str, storage_service: StorageService):
        """Generate thumbnail image."""
        thumb_matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
        pix = page.get_pixmap(matrix=thumb_matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=75, optimize=True)
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_page_{page_num}_thumb.jpg",
            data=output.getvalue(), content_type="image/jpeg"
        )
        
    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content with enhanced pattern matching"""
        info = {}
        text_upper = text.upper()
        
        drawing_patterns = {'floor_plan': ['FLOOR PLAN', 'LEVEL'], 'electrical': ['ELECTRICAL'], 'structural': ['STRUCTURAL']}
        for dtype, patterns in drawing_patterns.items():
            if any(p in text_upper for p in patterns):
                info['drawing_type'] = dtype
                break
        
        scale_match = re.search(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+)', text_upper)
        if scale_match: info['scale'] = scale_match.group(1)

        sheet_match = re.search(r'(?:SHEET|DWG)[\s#:]*([A-Z]*[-\s]?[0-9]+\.?[0-9]*)', text_upper)
        if sheet_match: info['sheet_number'] = sheet_match.group(1).strip()
        
        return info
        
    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   storage_service: StorageService):
        """Create a searchable index of the document's contents."""
        try:
            index = {
                'document_id': session_id, 'total_pages': len(page_details),
                'page_index': {}, 'drawing_types': defaultdict(list),
                'sheet_numbers': {}, 'grid_pages': []
            }
            for detail in page_details:
                page_num = detail['page_number']
                index['page_index'][page_num] = {k: v for k, v in detail.items() if k != 'page_number'}
                if detail.get('drawing_type'): index['drawing_types'][detail['drawing_type']].append(page_num)
                if detail.get('sheet_number'): index['sheet_numbers'][detail['sheet_number']] = page_num
                if detail.get('has_grid'): index['grid_pages'].append(page_num)
            
            index['drawing_types'] = dict(index['drawing_types'])
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_document_index.json",
                data=json.dumps(index, indent=2).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to create document index: {e}")

    async def _save_error_state(self, session_id: str, error: str,
                               storage_service: StorageService):
        """Save error information for debugging."""
        error_info = {'document_id': session_id, 'timestamp': datetime.now().isoformat(), 'error': error}
        try:
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_error.json",
                data=json.dumps(error_info).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to save error state: {e}")