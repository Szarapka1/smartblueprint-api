# app/services/pdf_service.py - INTELLIGENT PDF PREPROCESSING FOR AI-READY DOCUMENT DECOMPOSITION

import logging
import fitz  # PyMuPDF
import json
import os
import asyncio
import gc
from PIL import Image
import io
import re
from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import time
import numpy as np
import traceback

# Safe OpenCV import
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("⚠️ OpenCV not available - visual grid detection disabled")

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass
class GridSystem:
    """Grid System for AI Navigation - Saved as {document_id}_grid_systems.json"""
    page_number: int
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    x_coordinates: Dict[str, int] = field(default_factory=dict)
    y_coordinates: Dict[str, int] = field(default_factory=dict)
    x_lines: List[int] = field(default_factory=list)
    y_lines: List[int] = field(default_factory=list)
    cell_width: int = 100
    cell_height: int = 100
    origin_x: int = 0
    origin_y: int = 0
    confidence: float = 0.0
    scale: Optional[str] = None
    page_width: int = 0
    page_height: int = 0
    grid_type: str = "detected"
    
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
            'page_width': self.page_width,
            'page_height': self.page_height,
            'grid_type': self.grid_type
        }


class PDFService:
    """PDF Processing Service - Decomposes PDFs into AI-Ready Components"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        self._lock = asyncio.Lock()
        self._status_lock = asyncio.Lock()
        self._status_update_locks: Dict[str, asyncio.Lock] = {}
        
        # Resolution settings
        self.storage_dpi = settings.PDF_HIGH_RESOLUTION  # For full images
        self.ai_image_dpi = settings.PDF_AI_DPI         # For AI processing
        self.thumbnail_dpi = settings.PDF_THUMBNAIL_DPI  # For quick scanning
        
        # Processing settings
        self.max_pages = settings.PDF_MAX_PAGES
        self.batch_size = settings.PROCESSING_BATCH_SIZE
        self.max_concurrent_images = 2
        
        # Image optimization
        self.png_compression = settings.PDF_PNG_COMPRESSION
        self.jpeg_quality = settings.PDF_JPEG_QUALITY
        self.ai_max_dimension = settings.AI_MAX_IMAGE_DIMENSION
        
        # Text extraction settings
        self.max_text_per_page = 100000
        self.enable_tables = True
        self.enable_grid_detection = True
        
        # Memory management
        self.gc_frequency = 3
        self.processing_delay = 0.5
        
        # Grid detection patterns
        self.grid_patterns = {
            'column_line': re.compile(r'(?:COLUMN\s*LINE|COL\.?\s*LINE|C\.?L\.?)\s*([A-Z]+)', re.IGNORECASE),
            'grid_line': re.compile(r'(?:GRID\s*LINE|GRID)\s*([A-Z0-9]+)', re.IGNORECASE),
            'axis': re.compile(r'(?:AXIS|AX\.?)\s*([A-Z0-9]+)', re.IGNORECASE),
            'row': re.compile(r'(?:ROW|R\.?)\s*([0-9]+)', re.IGNORECASE),
            'column': re.compile(r'(?:COLUMN|COL\.?|C\.?)\s*([A-Z]+)(?:\s|$)', re.IGNORECASE),
            'grid_ref': re.compile(r'([A-Z]+)[-/]([0-9]+)', re.IGNORECASE),
            'coordinate': re.compile(r'(?:@|AT)\s*([A-Z]+)[-/]([0-9]+)', re.IGNORECASE)
        }
        
        # Element patterns for indexing
        self.element_patterns = {
            'door': re.compile(r'(?:DOOR|DR|D)[-\s]?([A-Z0-9\-]+)', re.IGNORECASE),
            'window': re.compile(r'(?:WINDOW|WIN|W)[-\s]?([A-Z0-9\-]+)', re.IGNORECASE),
            'room': re.compile(r'(?:ROOM|RM|SPACE)\s*([A-Z0-9\-]+)', re.IGNORECASE),
            'equipment': re.compile(r'(?:EQUIPMENT|EQUIP|EQ)[-\s]?([A-Z0-9\-]+)', re.IGNORECASE),
            'dimension': re.compile(r'(\d+[\'"](?:\s*-?\s*\d+(?:\s*\d+/\d+)?[\'"]?)?)', re.IGNORECASE),
            'elevation': re.compile(r'(?:EL\.|ELEV\.?|ELEVATION)\s*([\+\-]?\d+(?:\.\d+)?)', re.IGNORECASE),
            'section': re.compile(r'(?:SECTION|SEC|S)[-\s]?([A-Z0-9\-]+)', re.IGNORECASE),
            'detail': re.compile(r'(?:DETAIL|DET|DTL)[-\s]?([A-Z0-9\-/]+)', re.IGNORECASE)
        }
        
        # Visual detection settings
        self.min_line_length = 100
        self.line_gap_tolerance = 50
        self.grid_line_thickness_range = (0.5, 3)
        
        logger.info("🏗️ PDF Decomposition Service initialized")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ Image resolutions - Storage: {self.storage_dpi}, AI: {self.ai_image_dpi}, Thumbnail: {self.thumbnail_dpi}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   📐 Grid Detection: ENABLED")
        logger.info(f"   🗂️ File Decomposition: READY")

    def _get_status_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a status lock for a specific document"""
        if session_id not in self._status_update_locks:
            self._status_update_locks[session_id] = asyncio.Lock()
        return self._status_update_locks[session_id]

    async def _safe_event_callback(self, event_callback: Optional[Callable], event_type: str, event_data: Dict[str, Any]):
        """Safely call event callback with error handling"""
        if not event_callback:
            return
            
        try:
            await event_callback(event_type, event_data)
        except Exception as e:
            logger.error(f"Error in event callback for {event_type}: {e}")
            logger.error(traceback.format_exc())

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService,
                                   event_callback: Optional[Callable] = None):
        """
        Process PDF and decompose into AI-ready files:
        - {session_id}_context.txt - All extracted text
        - {session_id}_page_{n}_thumb.jpg - Thumbnails for AI scanning
        - {session_id}_page_{n}.jpg - Full resolution images
        - {session_id}_page_{n}_ai.jpg - AI optimized images
        - {session_id}_grid_systems.json - Grid data for navigation
        - {session_id}_navigation_index.json - Quick lookup index
        - {session_id}_metadata.json - Document metadata
        - {session_id}_document_index.json - Page content index
        - {session_id}_status.json - Processing status
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session ID")
        
        if not pdf_bytes or not isinstance(pdf_bytes, bytes):
            raise ValueError("Invalid PDF data")
        
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValueError("Not a valid PDF file")
        
        async with self._lock:
            try:
                logger.info(f"🏗️ Starting PDF decomposition for: {session_id}")
                pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
                logger.info(f"📄 File Size: {pdf_size_mb:.1f}MB")
                
                if pdf_size_mb > self.settings.MAX_FILE_SIZE_MB:
                    raise ValueError(f"PDF too large: {pdf_size_mb:.1f}MB (max: {self.settings.MAX_FILE_SIZE_MB}MB)")
                
                processing_start = time.time()
                
                # Open PDF
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                except Exception as e:
                    raise RuntimeError(f"Failed to open PDF: {e}")
                
                try:
                    total_pages = len(doc)
                    pages_to_process = min(total_pages, self.max_pages)
                    
                    logger.info(f"📄 Decomposing {pages_to_process} of {total_pages} pages")
                    
                    # Initialize metadata
                    metadata = self._initialize_metadata(session_id, doc, pages_to_process, total_pages, pdf_size_mb)
                    
                    # Collections for aggregated data
                    all_text_parts = []          # For _context.txt
                    all_grid_systems = {}        # For _grid_systems.json
                    all_tables = {}              # For table data
                    navigation_data = {          # For _navigation_index.json
                        'page_thumbnails': [],
                        'page_summaries': {},
                        'element_locations': defaultdict(list),
                        'drawing_types': defaultdict(list),
                        'quick_references': {}
                    }
                    
                    pages_processed = 0
                    total_batches = (pages_to_process + self.batch_size - 1) // self.batch_size
                    
                    # Process pages in batches
                    for batch_num, batch_start in enumerate(range(0, pages_to_process, self.batch_size), 1):
                        batch_end = min(batch_start + self.batch_size, pages_to_process)
                        batch_pages = list(range(batch_start, batch_end))
                        
                        logger.info(f"📦 Processing batch {batch_num}/{total_batches}: pages {batch_start + 1}-{batch_end}")
                        
                        # Emit batch start event
                        await self._safe_event_callback(event_callback, "batch_start", {
                            "batch_number": batch_num,
                            "total_batches": total_batches,
                            "start_page": batch_start + 1,
                            "end_page": batch_end
                        })
                        
                        # Process batch
                        batch_results = await self._process_batch_safe(
                            doc, batch_pages, session_id, storage_service, event_callback
                        )
                        
                        # Collect results from batch
                        for result in batch_results:
                            if result['success']:
                                page_num = result['metadata']['page_number']
                                
                                # Collect text for _context.txt
                                all_text_parts.append(result['text'])
                                
                                # Collect page metadata
                                metadata['page_details'].append(result['metadata'])
                                
                                # Collect grid system for _grid_systems.json
                                if result.get('grid_system'):
                                    all_grid_systems[str(page_num)] = result['grid_system'].to_dict()
                                    metadata['extraction_summary']['has_grid_systems'] = True
                                
                                # Collect tables
                                if result.get('tables'):
                                    all_tables[str(page_num)] = result['tables']
                                
                                # Build navigation data
                                self._update_navigation_data(
                                    navigation_data, 
                                    page_num, 
                                    result['metadata'],
                                    result.get('grid_system')
                                )
                                
                                self._update_extraction_summary(metadata, result)
                                pages_processed += 1
                        
                        # Update status file
                        metadata['pages_processed'] = pages_processed
                        await self._update_status(session_id, 'processing', metadata, storage_service)
                        
                        # Emit batch complete event
                        progress_percent = round((pages_processed / pages_to_process) * 100)
                        await self._safe_event_callback(event_callback, "batch_complete", {
                            "batch_number": batch_num,
                            "total_batches": total_batches,
                            "pages_processed": pages_processed,
                            "progress_percent": progress_percent
                        })
                        
                        # Memory management
                        if (batch_end % self.gc_frequency) == 0:
                            gc.collect()
                            logger.debug(f"🧹 Memory cleanup after {batch_end} pages")
                        
                        await asyncio.sleep(self.processing_delay)
                        
                        progress = (pages_processed / pages_to_process) * 100
                        logger.info(f"📊 Progress: {progress:.1f}% ({pages_processed}/{pages_to_process})")
                    
                    # Save all decomposed files
                    await self._save_all_decomposed_files(
                        session_id=session_id,
                        all_text_parts=all_text_parts,
                        all_grid_systems=all_grid_systems,
                        all_tables=all_tables,
                        navigation_data=navigation_data,
                        metadata=metadata,
                        processing_start=processing_start,
                        storage_service=storage_service,
                        event_callback=event_callback
                    )
                    
                    # Update final status
                    metadata['status'] = 'ready'
                    metadata['processing_time'] = round(time.time() - processing_start, 2)
                    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
                    await self._update_status(session_id, 'ready', metadata, storage_service)
                    
                    # Emit processing complete event
                    await self._safe_event_callback(event_callback, "processing_complete", {
                        "status": "ready",
                        "total_pages": pages_processed,
                        "processing_time": metadata['processing_time'],
                        "files_created": {
                            "text_context": f"{session_id}_context.txt",
                            "grid_systems": f"{session_id}_grid_systems.json",
                            "navigation_index": f"{session_id}_navigation_index.json",
                            "metadata": f"{session_id}_metadata.json",
                            "document_index": f"{session_id}_document_index.json",
                            "thumbnails": pages_processed,
                            "full_images": pages_processed,
                            "ai_images": pages_processed
                        }
                    })
                    
                    logger.info(f"✅ PDF decomposition complete for {session_id}")
                    logger.info(f"   📝 Pages processed: {pages_processed}")
                    logger.info(f"   ⏱️ Total time: {metadata['processing_time']}s")
                    logger.info(f"   🗂️ Files created for AI access")
                    
                finally:
                    doc.close()
                    gc.collect()
                    if session_id in self._status_update_locks:
                        del self._status_update_locks[session_id]
                    
            except Exception as e:
                logger.error(f"❌ Processing failed: {e}", exc_info=True)
                await self._save_error_state(session_id, str(e), storage_service)
                await self._safe_event_callback(event_callback, "processing_error", {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "is_fatal": True
                })
                raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _update_status(self, session_id: str, status: str, metadata: Dict, storage_service: StorageService):
        """Update status file - {session_id}_status.json"""
        status_lock = self._get_status_lock(session_id)
        
        async with status_lock:
            try:
                async with asyncio.timeout(self.settings.STATUS_UPDATE_LOCK_TIMEOUT):
                    status_data = {
                        'document_id': session_id,
                        'status': status,
                        'updated_at': datetime.utcnow().isoformat() + 'Z',
                        'pages_processed': metadata.get('pages_processed', 0),
                        'total_pages': metadata.get('total_pages', 0),
                        'started_at': metadata.get('started_at'),
                        'completed_at': metadata.get('completed_at'),
                        'error': metadata.get('error'),
                        'processing_time': metadata.get('processing_time'),
                        'grid_systems_detected': metadata.get('grid_systems_detected', 0),
                        'last_update_timestamp': time.time()
                    }
                    
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_status.json",
                        data=json.dumps(status_data).encode('utf-8'),
                        content_type="application/json"
                    )
                    
            except asyncio.TimeoutError:
                logger.error(f"Status update timeout for {session_id}")
            except Exception as e:
                logger.error(f"Failed to update status for {session_id}: {e}")

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
            'file_size_mb': pdf_size_mb,
            'page_details': [],
            'grid_detection_enabled': self.enable_grid_detection,
            'extraction_summary': {
                'has_text': False,
                'has_images': True,
                'has_tables': False,
                'has_grid_systems': False,
                'total_tables_extracted': 0,
                'total_tables_found': 0
            },
            'started_at': datetime.utcnow().isoformat() + 'Z',
            'pages_processed': 0,
            'grid_systems_detected': 0
        }

    def _update_extraction_summary(self, metadata: Dict[str, Any], result: Dict[str, Any]):
        """Update extraction summary based on page results"""
        if result['text'].strip():
            metadata['extraction_summary']['has_text'] = True
        
        if result.get('has_tables'):
            metadata['extraction_summary']['has_tables'] = True
        
        if 'tables_found' in result:
            metadata['extraction_summary']['total_tables_found'] += result['tables_found']
        if 'tables_extracted' in result:
            metadata['extraction_summary']['total_tables_extracted'] += result['tables_extracted']
        
        if result.get('grid_system'):
            metadata['grid_systems_detected'] = metadata.get('grid_systems_detected', 0) + 1

    async def _process_batch_safe(self, doc: fitz.Document, page_numbers: List[int], 
                                 session_id: str, storage_service: StorageService,
                                 event_callback: Optional[Callable] = None) -> List[Dict]:
        """Process batch with error recovery"""
        results = []
        
        for page_num in page_numbers:
            try:
                result = await self._process_single_page_safe(
                    doc, page_num, session_id, storage_service, event_callback
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {e}")
                results.append({
                    'success': False,
                    'page_num': page_num + 1,
                    'error': str(e),
                    'text': '',
                    'metadata': {'page_number': page_num + 1}
                })
                
                await self._safe_event_callback(event_callback, "page_error", {
                    "page_number": page_num + 1,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
        
        return results

    async def _process_single_page_safe(self, doc: fitz.Document, page_num: int, 
                                       session_id: str, storage_service: StorageService,
                                       event_callback: Optional[Callable] = None) -> Dict:
        """Process single page and extract all components"""
        try:
            page = doc[page_num]
            page_actual = page_num + 1
            
            # 1. Extract text for _context.txt
            page_text = page.get_text()
            if len(page_text) > self.max_text_per_page:
                page_text = page_text[:self.max_text_per_page] + "\n[Text truncated]"
            
            # 2. Analyze page content
            page_analysis = self._analyze_page_content(page_text, page_actual)
            
            # 3. Extract tables
            tables = []
            tables_found = 0
            tables_extracted = 0
            if self.enable_tables:
                table_result = await self._extract_tables_safe(page)
                tables = table_result['tables']
                tables_found = table_result['found']
                tables_extracted = table_result['extracted']
            
            # 4. Detect grid system for _grid_systems.json
            grid_system = None
            if self.enable_grid_detection:
                grid_system = await self._detect_grid_patterns(page, page_text, page_actual)
            
            # 5. Generate all image versions
            await self._generate_all_page_images(
                page, page_actual, session_id, storage_service, event_callback
            )
            
            # Prepare metadata
            page_metadata = {
                'page_number': page_actual,
                'text_length': len(page_text),
                'has_text': bool(page_text.strip()),
                'has_tables': len(tables) > 0,
                'table_count': len(tables),
                'tables_found': tables_found,
                'tables_extracted': tables_extracted,
                'table_details': [
                    {
                        'rows': t.get('row_count', 0),
                        'cols': t.get('col_count', 0),
                        'extraction': t.get('extraction_note', 'complete')
                    } for t in tables
                ] if tables else [],
                'drawing_type': page_analysis.get('drawing_type'),
                'sheet_number': page_analysis.get('sheet_number'),
                'scale': page_analysis.get('scale'),
                'key_elements': page_analysis.get('key_elements', []),
                'has_grid': grid_system is not None,
                'grid_confidence': grid_system.confidence if grid_system else 0.0,
                'grid_type': grid_system.grid_type if grid_system else None
            }
            
            # Format text for context file
            formatted_text = self._format_page_text(page_actual, page_analysis, grid_system, page_text)
            
            # Clean up page
            page.clean_contents()
            
            # Emit page processed event
            total_pages = doc.page_count
            progress_percent = round((page_actual / total_pages) * 100)
            estimated_time = int((total_pages - page_actual) * self.processing_delay)
            
            await self._safe_event_callback(event_callback, "page_processed", {
                "page_number": page_actual,
                "total_pages": total_pages,
                "progress_percent": progress_percent,
                "estimated_time": estimated_time,
                "has_text": page_metadata['has_text'],
                "has_tables": page_metadata['has_tables'],
                "has_grid": page_metadata['has_grid'],
                "images_generated": {
                    "thumbnail": f"{session_id}_page_{page_actual}_thumb.jpg",
                    "full": f"{session_id}_page_{page_actual}.jpg",
                    "ai": f"{session_id}_page_{page_actual}_ai.jpg"
                }
            })
            
            return {
                'success': True,
                'page_num': page_actual,
                'text': formatted_text,
                'metadata': page_metadata,
                'tables': tables,
                'grid_system': grid_system,
                'has_tables': len(tables) > 0,
                'tables_found': tables_found,
                'tables_extracted': tables_extracted
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            raise

    async def _detect_grid_patterns(self, page: fitz.Page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Detect grid patterns for navigation"""
        try:
            page_width = page.rect.width
            page_height = page.rect.height
            
            logger.info(f"📐 Grid detection for page {page_num}: {page_width}x{page_height}px")
            
            # Try visual detection first (most accurate)
            visual_grid = None
            if OPENCV_AVAILABLE:
                visual_grid = await self._detect_visual_grid(page, page_num)
            
            # Try text-based detection
            text_grid = self._detect_text_based_grid(page, page_text, page_num)
            
            # Choose best grid or create universal grid
            if visual_grid and visual_grid.confidence > 0.7:
                grid_system = visual_grid
                logger.info(f"✅ Using visual grid with confidence {visual_grid.confidence:.2f}")
            elif text_grid and (len(text_grid.x_labels) > 0 or len(text_grid.y_labels) > 0):
                grid_system = text_grid
                logger.info(f"✅ Using text-based grid with {len(text_grid.x_labels)}x{len(text_grid.y_labels)} labels")
            else:
                # Create universal grid
                grid_system = self._create_universal_grid(page_width, page_height, page_num)
                logger.info(f"📏 Created universal grid for page {page_num}")
            
            return grid_system
            
        except Exception as e:
            logger.error(f"Grid detection failed: {e}")
            # Always return a basic grid
            return self._create_universal_grid(page.rect.width, page.rect.height, page_num)

    async def _detect_visual_grid(self, page: fitz.Page, page_num: int) -> Optional[GridSystem]:
        """Detect grid lines visually using OpenCV"""
        if not OPENCV_AVAILABLE:
            return None
            
        try:
            # Extract page as image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better line detection
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            
            # Detect lines
            horizontal_lines = self._detect_horizontal_lines(gray)
            vertical_lines = self._detect_vertical_lines(gray)
            
            # Clean up
            pix = None
            
            if not horizontal_lines and not vertical_lines:
                return None
            
            # Extract text for label mapping
            page_text = page.get_text()
            
            # Create grid system from detected lines
            grid_system = self._create_grid_from_visual_lines(
                horizontal_lines, vertical_lines, page_text, page_num,
                page.rect.width, page.rect.height
            )
            
            return grid_system
            
        except Exception as e:
            logger.error(f"Visual grid detection failed: {e}")
            return None

    def _detect_horizontal_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect horizontal lines in image"""
        lines = []
        
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        detected_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_line_length,
            maxLineGap=self.line_gap_tolerance
        )
        
        if detected_lines is None:
            return lines
        
        # Process horizontal lines
        horizontal_positions = {}
        
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is horizontal
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 5 or angle > 175:
                y_avg = (y1 + y2) / 2
                
                # Group nearby lines
                found_group = False
                for y_pos in list(horizontal_positions.keys()):
                    if abs(y_pos - y_avg) < 10:
                        horizontal_positions[y_pos].append((x1, y1, x2, y2))
                        found_group = True
                        break
                
                if not found_group:
                    horizontal_positions[y_avg] = [(x1, y1, x2, y2)]
        
        # Create line objects
        for y_pos, segments in horizontal_positions.items():
            min_x = min(min(seg[0], seg[2]) for seg in segments)
            max_x = max(max(seg[0], seg[2]) for seg in segments)
            
            total_length = sum(abs(seg[2] - seg[0]) for seg in segments)
            span_length = max_x - min_x
            confidence = total_length / span_length if span_length > 0 else 0
            
            if confidence > 0.5:
                lines.append({
                    'position': y_pos / 2,  # Adjust for 2x zoom
                    'confidence': confidence
                })
        
        return sorted(lines, key=lambda l: l['position'])

    def _detect_vertical_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vertical lines in image"""
        lines = []
        
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        detected_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_line_length,
            maxLineGap=self.line_gap_tolerance
        )
        
        if detected_lines is None:
            return lines
        
        # Process vertical lines
        vertical_positions = {}
        
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is vertical
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 85 < angle < 95:
                x_avg = (x1 + x2) / 2
                
                # Group nearby lines
                found_group = False
                for x_pos in list(vertical_positions.keys()):
                    if abs(x_pos - x_avg) < 10:
                        vertical_positions[x_pos].append((x1, y1, x2, y2))
                        found_group = True
                        break
                
                if not found_group:
                    vertical_positions[x_avg] = [(x1, y1, x2, y2)]
        
        # Create line objects
        for x_pos, segments in vertical_positions.items():
            min_y = min(min(seg[1], seg[3]) for seg in segments)
            max_y = max(max(seg[1], seg[3]) for seg in segments)
            
            total_length = sum(abs(seg[3] - seg[1]) for seg in segments)
            span_length = max_y - min_y
            confidence = total_length / span_length if span_length > 0 else 0
            
            if confidence > 0.5:
                lines.append({
                    'position': x_pos / 2,  # Adjust for 2x zoom
                    'confidence': confidence
                })
        
        return sorted(lines, key=lambda l: l['position'])

    def _create_grid_from_visual_lines(self, h_lines: List[Dict], v_lines: List[Dict], 
                                      page_text: str, page_num: int, 
                                      page_width: float, page_height: float) -> GridSystem:
        """Create grid system from detected visual lines"""
        # Extract potential labels from text
        x_labels, y_labels = self._extract_grid_labels_from_text(page_text)
        
        # Create grid
        grid = GridSystem(
            page_number=page_num,
            page_width=int(page_width),
            page_height=int(page_height),
            grid_type="visual",
            confidence=0.9 if (h_lines and v_lines) else 0.7
        )
        
        # Map vertical lines
        if v_lines:
            for i, line in enumerate(v_lines):
                label = x_labels[i] if i < len(x_labels) else chr(65 + i)
                grid.x_labels.append(label)
                grid.x_coordinates[label] = int(line['position'])
                grid.x_lines.append(int(line['position']))
            
            if len(v_lines) > 1:
                spacings = [v_lines[i+1]['position'] - v_lines[i]['position'] 
                           for i in range(len(v_lines)-1)]
                grid.cell_width = int(sum(spacings) / len(spacings))
        
        # Map horizontal lines
        if h_lines:
            for i, line in enumerate(h_lines):
                label = y_labels[i] if i < len(y_labels) else str(i + 1)
                grid.y_labels.append(label)
                grid.y_coordinates[label] = int(line['position'])
                grid.y_lines.append(int(line['position']))
            
            if len(h_lines) > 1:
                spacings = [h_lines[i+1]['position'] - h_lines[i]['position'] 
                           for i in range(len(h_lines)-1)]
                grid.cell_height = int(sum(spacings) / len(spacings))
        
        return grid

    def _detect_text_based_grid(self, page: fitz.Page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Detect grid from text patterns"""
        try:
            x_refs = set()
            y_refs = set()
            
            # Search for grid patterns
            search_text = page_text[:10000] if len(page_text) > 10000 else page_text
            
            for pattern_name, pattern in self.grid_patterns.items():
                matches = list(pattern.finditer(search_text))[:20]
                for match in matches:
                    if pattern_name in ['column', 'column_line']:
                        x_refs.add(match.group(1).upper())
                    elif pattern_name == 'row':
                        y_refs.add(match.group(1))
                    elif pattern_name in ['grid_ref', 'coordinate']:
                        x_refs.add(match.group(1).upper())
                        y_refs.add(match.group(2))
            
            if not x_refs and not y_refs:
                return None
            
            # Create grid
            grid = GridSystem(
                page_number=page_num,
                x_labels=sorted(list(x_refs))[:20],
                y_labels=sorted(list(y_refs), key=lambda x: int(x) if x.isdigit() else 0)[:30],
                confidence=0.6,
                page_width=int(page.rect.width),
                page_height=int(page.rect.height),
                grid_type="text"
            )
            
            # Distribute coordinates
            if grid.x_labels:
                spacing = page.rect.width / (len(grid.x_labels) + 1)
                for i, label in enumerate(grid.x_labels):
                    x_pos = (i + 1) * spacing
                    grid.x_coordinates[label] = int(x_pos)
                    grid.x_lines.append(int(x_pos))
                grid.cell_width = int(spacing)
            
            if grid.y_labels:
                spacing = page.rect.height / (len(grid.y_labels) + 1)
                for i, label in enumerate(grid.y_labels):
                    y_pos = (i + 1) * spacing
                    grid.y_coordinates[label] = int(y_pos)
                    grid.y_lines.append(int(y_pos))
                grid.cell_height = int(spacing)
            
            return grid
            
        except Exception as e:
            logger.error(f"Text-based grid detection failed: {e}")
            return None

    def _create_universal_grid(self, page_width: float, page_height: float, page_num: int) -> GridSystem:
        """Create universal grid for any document"""
        grid = GridSystem(
            page_number=page_num,
            page_width=int(page_width),
            page_height=int(page_height),
            grid_type="universal",
            confidence=1.0
        )
        
        # Determine optimal grid size
        aspect_ratio = page_width / page_height
        
        if aspect_ratio > 1.3:  # Landscape
            num_cols = 12
            num_rows = 8
        else:  # Portrait
            num_cols = 10
            num_rows = 12
        
        # Create grid
        col_spacing = page_width / (num_cols + 1)
        row_spacing = page_height / (num_rows + 1)
        
        for i in range(num_cols):
            label = chr(65 + i) if i < 26 else f"X{i}"
            x_pos = (i + 1) * col_spacing
            grid.x_labels.append(label)
            grid.x_coordinates[label] = int(x_pos)
            grid.x_lines.append(int(x_pos))
        
        for i in range(num_rows):
            label = str(i + 1)
            y_pos = (i + 1) * row_spacing
            grid.y_labels.append(label)
            grid.y_coordinates[label] = int(y_pos)
            grid.y_lines.append(int(y_pos))
        
        grid.cell_width = int(col_spacing)
        grid.cell_height = int(row_spacing)
        
        logger.info(f"Created universal {num_cols}x{num_rows} grid for page {page_num}")
        
        return grid

    def _extract_grid_labels_from_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract grid labels from page text"""
        x_labels = set()
        y_labels = set()
        
        search_text = text[:10000] if len(text) > 10000 else text
        
        for pattern_name, pattern in self.grid_patterns.items():
            for match in pattern.finditer(search_text):
                if pattern_name in ['column', 'column_line']:
                    x_labels.add(match.group(1).upper())
                elif pattern_name == 'row':
                    y_labels.add(match.group(1))
                elif pattern_name in ['grid_ref', 'coordinate']:
                    x_labels.add(match.group(1).upper())
                    y_labels.add(match.group(2))
        
        return sorted(list(x_labels)), sorted(list(y_labels), key=lambda x: int(x) if x.isdigit() else 0)

    async def _extract_tables_safe(self, page: fitz.Page) -> Dict[str, Any]:
        """Extract tables from page"""
        tables = []
        tables_found = 0
        tables_extracted = 0
        
        try:
            page_tables = page.find_tables()
            if page_tables:
                tables_found = len(page_tables)
                
                for i, table in enumerate(page_tables):
                    try:
                        table_data = table.extract()
                        
                        if table_data and len(table_data) > 0:
                            cleaned_data = self._clean_table_data(table_data)
                            
                            if cleaned_data:
                                tables.append({
                                    'index': i,
                                    'data': cleaned_data,
                                    'row_count': len(cleaned_data),
                                    'col_count': max(len(row) for row in cleaned_data) if cleaned_data else 0
                                })
                                tables_extracted += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract table {i+1}: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return {
            'tables': tables,
            'found': tables_found,
            'extracted': tables_extracted
        }

    def _clean_table_data(self, table_data: List[List[Any]]) -> List[List[str]]:
        """Clean and normalize table data"""
        if not table_data:
            return []
        
        cleaned = []
        
        for row in table_data:
            cleaned_row = []
            for cell in row:
                cell_str = str(cell) if cell is not None else ""
                cell_str = " ".join(cell_str.strip().split())
                cleaned_row.append(cell_str)
            
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned

    def _format_page_text(self, page_num: int, page_analysis: Dict[str, Any], 
                         grid_system: Optional[GridSystem], page_text: str) -> str:
        """Format page text for context file"""
        formatted_text = f"\n--- PAGE {page_num} ---\n"
        
        if page_analysis.get('sheet_number'):
            formatted_text += f"Sheet: {page_analysis['sheet_number']}\n"
        
        if page_analysis.get('drawing_type'):
            formatted_text += f"Type: {page_analysis['drawing_type']}\n"
        
        if grid_system:
            formatted_text += f"Grid: {len(grid_system.x_labels)}x{len(grid_system.y_labels)} ({grid_system.grid_type})\n"
            if grid_system.confidence > 0:
                formatted_text += f"Grid Confidence: {grid_system.confidence:.2f}\n"
        
        if page_analysis.get('scale'):
            formatted_text += f"Scale: {page_analysis['scale']}\n"
        
        formatted_text += page_text
        
        return formatted_text

    async def _generate_all_page_images(self, page: fitz.Page, page_num: int, 
                                       session_id: str, storage_service: StorageService,
                                       event_callback: Optional[Callable] = None):
        """Generate all image versions for a page"""
        try:
            # 1. Thumbnail for AI quick scanning - {session_id}_page_{n}_thumb.jpg
            await self._generate_jpeg_image(page, page_num, session_id, storage_service,
                                          dpi=self.thumbnail_dpi, quality=70, suffix="_thumb",
                                          resource_type="thumbnail", event_callback=event_callback)
            
            # 2. Full resolution for detailed viewing - {session_id}_page_{n}.jpg
            await self._generate_jpeg_image(page, page_num, session_id, storage_service, 
                                          dpi=self.storage_dpi, quality=90, suffix="", 
                                          resource_type="full_image", event_callback=event_callback)
            
            # 3. AI optimized for analysis - {session_id}_page_{n}_ai.jpg
            await self._generate_jpeg_image(page, page_num, session_id, storage_service,
                                          dpi=self.ai_image_dpi, quality=85, suffix="_ai",
                                          resource_type="ai_image", event_callback=event_callback)
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")

    async def _generate_jpeg_image(self, page: fitz.Page, page_num: int, session_id: str,
                                  storage_service: StorageService, dpi: int, quality: int, suffix: str,
                                  resource_type: str, event_callback: Optional[Callable] = None):
        """Generate and upload JPEG image"""
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        
        try:
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Resize if needed
            if suffix == "_ai" and max(img.size) > self.ai_max_dimension:
                img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            
            # Upload
            blob_name = f"{session_id}_page_{page_num}{suffix}.jpg"
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=blob_name,
                data=output.getvalue(),
                content_type="image/jpeg"
            )
            
            await self._safe_event_callback(event_callback, "resource_ready", {
                "resource_type": resource_type,
                "resource_id": f"page_{page_num}_{resource_type}",
                "page_number": page_num,
                "filename": blob_name
            })
            
        finally:
            pix = None
            if 'img' in locals():
                img.close()
            if 'output' in locals():
                output.close()

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content"""
        info = {
            'page_number': page_num,
            'drawing_type': None,
            'title': None,
            'scale': None,
            'sheet_number': None,
            'key_elements': []
        }
        
        text_upper = text.upper()
        
        # Identify drawing type
        drawing_patterns = [
            ('floor_plan', ['FLOOR PLAN', 'LEVEL', r'\d+(?:ST|ND|RD|TH)\s*FLOOR']),
            ('foundation', ['FOUNDATION PLAN', 'FOOTING PLAN']),
            ('electrical', ['ELECTRICAL PLAN', 'POWER PLAN', 'LIGHTING PLAN']),
            ('plumbing', ['PLUMBING PLAN', 'PIPING PLAN']),
            ('mechanical', ['MECHANICAL PLAN', 'HVAC PLAN']),
            ('structural', ['STRUCTURAL PLAN', 'FRAMING PLAN']),
            ('detail', ['DETAIL', 'SECTION']),
            ('elevation', ['ELEVATION']),
            ('site', ['SITE PLAN'])
        ]
        
        for dtype, patterns in drawing_patterns:
            for pattern in patterns:
                if isinstance(pattern, str) and pattern in text_upper:
                    info['drawing_type'] = dtype
                    break
        
        # Extract scale
        scale_match = re.search(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+|[0-9]+:[0-9]+)', text_upper)
        if scale_match:
            info['scale'] = scale_match.group(1)
        
        # Extract sheet number
        sheet_match = re.search(r'(?:SHEET|DWG)[\s#:]*([A-Z]*[-\s]?[0-9]+\.?[0-9]*)', text_upper)
        if sheet_match:
            info['sheet_number'] = sheet_match.group(1).strip()
        
        # Identify key elements
        element_keywords = {
            'columns': ['COLUMN', 'COL.'],
            'beams': ['BEAM', 'BM.'],
            'doors': ['DOOR', 'DR.'],
            'windows': ['WINDOW', 'WIN.'],
            'equipment': ['EQUIPMENT', 'UNIT'],
            'dimensions': ['DIM', 'DIMENSION']
        }
        
        for element, keywords in element_keywords.items():
            for keyword in keywords:
                if keyword in text_upper:
                    info['key_elements'].append(element)
                    break
        
        info['key_elements'] = list(set(info['key_elements']))[:10]
        
        return info

    def _update_navigation_data(self, nav_data: Dict[str, Any], page_num: int, 
                               page_metadata: Dict[str, Any], grid_system: Optional[GridSystem]):
        """Update navigation data for AI access"""
        # Add thumbnail reference
        nav_data['page_thumbnails'].append({
            'page': page_num,
            'thumbnail': f"_page_{page_num}_thumb.jpg",
            'has_grid': page_metadata.get('has_grid', False)
        })
        
        # Add page summary
        nav_data['page_summaries'][str(page_num)] = {
            'drawing_type': page_metadata.get('drawing_type'),
            'sheet_number': page_metadata.get('sheet_number'),
            'has_text': page_metadata.get('has_text'),
            'has_tables': page_metadata.get('has_tables'),
            'key_elements': page_metadata.get('key_elements', []),
            'grid_confidence': page_metadata.get('grid_confidence', 0)
        }
        
        # Index element locations
        for element in page_metadata.get('key_elements', []):
            nav_data['element_locations'][element].append(page_num)
        
        # Index drawing types
        if page_metadata.get('drawing_type'):
            nav_data['drawing_types'][page_metadata['drawing_type']].append(page_num)
        
        # Add quick references
        if page_metadata.get('sheet_number'):
            nav_data['quick_references'][page_metadata['sheet_number']] = page_num

    async def _save_all_decomposed_files(self, session_id: str, all_text_parts: List[str],
                                        all_grid_systems: Dict[str, Any], all_tables: Dict[str, Any],
                                        navigation_data: Dict[str, Any], metadata: Dict[str, Any],
                                        processing_start: float, storage_service: StorageService,
                                        event_callback: Optional[Callable]):
        """Save all decomposed files for AI access"""
        try:
            # 1. Save text context - {session_id}_context.txt
            full_text = '\n'.join(all_text_parts)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_context.txt",
                data=full_text.encode('utf-8'),
                content_type="text/plain"
            )
            logger.info(f"📝 Saved {session_id}_context.txt ({len(full_text)} chars)")
            
            # 2. Save grid systems - {session_id}_grid_systems.json
            if all_grid_systems:
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_grid_systems.json",
                    data=json.dumps(all_grid_systems, indent=2).encode('utf-8'),
                    content_type="application/json"
                )
                logger.info(f"📐 Saved {session_id}_grid_systems.json ({len(all_grid_systems)} grids)")
            
            # 3. Save navigation index - {session_id}_navigation_index.json
            # Convert defaultdicts to regular dicts
            navigation_data['element_locations'] = dict(navigation_data['element_locations'])
            navigation_data['drawing_types'] = dict(navigation_data['drawing_types'])
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_navigation_index.json",
                data=json.dumps(navigation_data, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            logger.info(f"🗺️ Saved {session_id}_navigation_index.json")
            
            # 4. Create and save document index - {session_id}_document_index.json
            document_index = await self._create_document_index(
                session_id, metadata['page_details'], full_text, navigation_data
            )
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_document_index.json",
                data=json.dumps(document_index, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            logger.info(f"📑 Saved {session_id}_document_index.json")
            
            # 5. Save tables if any - {session_id}_tables.json
            if all_tables:
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_tables.json",
                    data=json.dumps(all_tables, indent=2).encode('utf-8'),
                    content_type="application/json"
                )
                logger.info(f"📊 Saved {session_id}_tables.json")
            
            # 6. Update and save metadata - {session_id}_metadata.json
            processing_end = time.time()
            metadata['processing_time'] = round(processing_end - processing_start, 2)
            metadata['grid_systems_detected'] = len(all_grid_systems)
            metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
            metadata['files_created'] = {
                'context': f"{session_id}_context.txt",
                'grid_systems': f"{session_id}_grid_systems.json" if all_grid_systems else None,
                'navigation_index': f"{session_id}_navigation_index.json",
                'document_index': f"{session_id}_document_index.json",
                'tables': f"{session_id}_tables.json" if all_tables else None,
                'thumbnails': metadata['page_count'],
                'full_images': metadata['page_count'],
                'ai_images': metadata['page_count']
            }
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json",
                data=json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            logger.info(f"📋 Saved {session_id}_metadata.json")
            
            await self._safe_event_callback(event_callback, "files_saved", {
                "files_created": metadata['files_created'],
                "total_text_length": len(full_text),
                "total_grids": len(all_grid_systems),
                "total_tables": sum(len(tables) for tables in all_tables.values()) if all_tables else 0
            })
            
        except Exception as e:
            logger.error(f"Failed to save decomposed files: {e}")
            raise

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   full_text: str, navigation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive document index for AI"""
        index = {
            'document_id': session_id,
            'total_pages': len(page_details),
            'page_index': {},
            'drawing_types': navigation_data.get('drawing_types', {}),
            'sheet_numbers': navigation_data.get('quick_references', {}),
            'element_pages': navigation_data.get('element_locations', {}),
            'grid_pages': [],
            'table_pages': [],
            'text_stats': {
                'total_length': len(full_text),
                'has_content': bool(full_text.strip())
            }
        }
        
        for page_detail in page_details:
            page_num = page_detail['page_number']
            
            index['page_index'][str(page_num)] = {
                'has_text': page_detail['has_text'],
                'text_length': page_detail.get('text_length', 0),
                'drawing_type': page_detail.get('drawing_type'),
                'sheet_number': page_detail.get('sheet_number'),
                'has_grid': page_detail.get('has_grid', False),
                'grid_confidence': page_detail.get('grid_confidence', 0.0),
                'grid_type': page_detail.get('grid_type'),
                'has_tables': page_detail.get('has_tables', False),
                'table_count': page_detail.get('table_count', 0),
                'key_elements': page_detail.get('key_elements', [])
            }
            
            if page_detail.get('has_grid'):
                index['grid_pages'].append(page_num)
            
            if page_detail.get('has_tables'):
                index['table_pages'].append(page_num)
        
        return index

    async def _save_error_state(self, session_id: str, error: str, 
                               storage_service: StorageService):
        """Save error information"""
        error_info = {
            'document_id': session_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'error': str(error),
            'status': 'error',
            'error_type': 'processing_failed'
        }
        
        try:
            await self._update_status(session_id, 'error', error_info, storage_service)
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_error.json",
                data=json.dumps(error_info, indent=2).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to save error state: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "PDFService",
            "version": "5.0.0-AI-READY-DECOMPOSITION",
            "mode": "document_decomposition",
            "capabilities": {
                "max_pages": self.max_pages,
                "batch_size": self.batch_size,
                "text_extraction": True,
                "table_extraction": True,
                "grid_detection": True,
                "visual_grid_detection": OPENCV_AVAILABLE,
                "file_outputs": {
                    "context_text": True,
                    "thumbnails": True,
                    "full_images": True,
                    "ai_optimized_images": True,
                    "grid_systems": True,
                    "navigation_index": True,
                    "document_index": True,
                    "metadata": True,
                    "tables": True
                },
                "multi_resolution": {
                    "thumbnail_dpi": self.thumbnail_dpi,
                    "full_dpi": self.storage_dpi,
                    "ai_dpi": self.ai_image_dpi
                }
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        gc.collect()
        self._status_update_locks.clear()
        logger.info("✅ PDF decomposition service cleaned up")