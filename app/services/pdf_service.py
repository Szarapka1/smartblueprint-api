# app/services/pdf_service.py - FIXED VERSION WITH RELIABLE UPLOAD FLOW

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
import traceback

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass
class GridSystem:
    """Enhanced grid system with pixel-accurate coordinates"""
    page_number: int
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    x_coordinates: Dict[str, int] = field(default_factory=dict)
    y_coordinates: Dict[str, int] = field(default_factory=dict)
    x_lines: List[int] = field(default_factory=list)
    y_lines: List[int] = field(default_factory=list)
    scale: Optional[str] = None
    confidence: float = 0.0
    detection_method: str = "none"
    image_width: int = 0
    image_height: int = 0
    grid_bounds: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'x_labels': self.x_labels,
            'y_labels': self.y_labels,
            'x_coordinates': {str(k): v for k, v in self.x_coordinates.items()},
            'y_coordinates': {str(k): v for k, v in self.y_coordinates.items()},
            'scale': self.scale,
            'confidence': self.confidence,
            'detection_method': self.detection_method,
            'image_dimensions': {
                'width': self.image_width,
                'height': self.image_height
            },
            'grid_bounds': self.grid_bounds
        }


class PDFService:
    """Ultra-reliable PDF processing with thumbnails and single high-quality JPEG per page"""

    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")

        self.settings = settings
        self._lock = asyncio.Lock()

        # Image settings - THUMBNAILS + HIGH QUALITY JPEG
        self.high_quality_dpi = 150
        self.thumbnail_dpi = settings.PDF_THUMBNAIL_DPI
        self.jpeg_quality = 90
        self.thumbnail_quality = 75
        self.jpeg_progressive = True

        # Processing settings - CONSERVATIVE FOR RELIABILITY
        self.max_pages = settings.PDF_MAX_PAGES
        self.batch_size = min(5, settings.PROCESSING_BATCH_SIZE)
        self.processing_timeout_per_page = 120
        self.image_generation_timeout = 180

        # Text and table extraction settings
        self.max_text_per_page = 150000
        self.enable_grid_detection = True
        self.enable_table_extraction = True
        self.grid_line_tolerance = 5

        # Memory management
        self.gc_frequency = 2
        self.processing_delay = 0.5
        self.memory_warning_threshold_mb = 500

        logger.info("✅ PDFService initialized (FIXED VERSION)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   📦 Batch size: {self.batch_size} pages")
        logger.info(f"   🖼️ High-quality: {self.high_quality_dpi} DPI, Quality={self.jpeg_quality}")
        logger.info(f"   🖼️ Thumbnails: {self.thumbnail_dpi} DPI, Quality={self.thumbnail_quality}")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes,
                                   storage_service: StorageService):
        """Process PDF with ultra-reliable thumbnail and high-quality JPEG generation"""
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

                # Process with proper error handling
                await self._process_pdf_internal(session_id, pdf_bytes, storage_service)

            except Exception as e:
                logger.error(f"❌ Processing failed: {e}")
                logger.error(traceback.format_exc())
                
                # Update status to failed
                await self._update_processing_status(
                    storage_service, session_id, 'failed', 
                    {'error': str(e), 'traceback': traceback.format_exc()}
                )
                
                raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _process_pdf_internal(self, session_id: str, pdf_bytes: bytes,
                                   storage_service: StorageService):
        """Internal PDF processing with detailed progress tracking"""
        processing_start = time.time()
        
        # Open PDF
        try:
            logger.info("📖 Opening PDF document...")
            doc = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: fitz.open(stream=pdf_bytes, filetype="pdf")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}")

        try:
            total_pages = len(doc)
            pages_to_process = min(total_pages, self.max_pages)
            logger.info(f"📄 Document has {total_pages} pages, processing {pages_to_process}")

            # Initialize tracking
            metadata = self._initialize_metadata(session_id, doc, pages_to_process, total_pages, len(pdf_bytes) / (1024 * 1024))
            all_text_parts = []
            all_tables = []
            all_grid_systems = {}
            pages_processed = 0

            # Update status to processing
            await self._update_processing_status(
                storage_service, session_id, 'processing',
                {'total_pages': total_pages, 'processing_pages': pages_to_process}
            )

            # Process in batches
            for batch_start in range(0, pages_to_process, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pages_to_process)
                batch_pages = list(range(batch_start, batch_end))
                
                batch_num = (batch_start // self.batch_size) + 1
                total_batches = (pages_to_process + self.batch_size - 1) // self.batch_size
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches}: pages {batch_start + 1}-{batch_end}")

                # Process batch
                batch_results = await self._process_batch_safe(doc, batch_pages, session_id, storage_service)

                # Aggregate results
                for result in batch_results:
                    if result.get('success'):
                        all_text_parts.append(result['text'])
                        metadata['page_details'].append(result['metadata'])
                        
                        if result.get('tables'):
                            all_tables.extend(result['tables'])
                        
                        if result.get('grid_system'):
                            page_num = result['metadata']['page_number']
                            all_grid_systems[str(page_num)] = result['grid_system'].to_dict()
                            metadata['extraction_summary']['has_grid_systems'] = True
                        
                        self._update_extraction_summary(metadata, result)
                        pages_processed += 1
                    else:
                        logger.error(f"❌ Failed to process page {result.get('page_num', '?')}: {result.get('error')}")
                
                # Update progress
                progress = (pages_processed / pages_to_process) * 100
                await self._update_processing_status(
                    storage_service, session_id, 'processing',
                    {'progress': progress, 'pages_processed': pages_processed}
                )
                
                # Memory management
                if batch_num % self.gc_frequency == 0:
                    gc.collect()
                    await asyncio.sleep(0.1)
                
                # Delay between batches
                if batch_end < pages_to_process:
                    await asyncio.sleep(self.processing_delay)

            # Save results
            logger.info("💾 Saving processing results...")
            await self._save_processing_results(
                session_id, all_text_parts, all_tables, all_grid_systems, 
                metadata, processing_start, storage_service
            )

            # Update final status
            await self._update_processing_status(
                storage_service, session_id, 'completed',
                {
                    'pages_processed': pages_processed,
                    'processing_time': time.time() - processing_start,
                    'has_thumbnails': True,
                    'has_text': bool(all_text_parts),
                    'has_tables': bool(all_tables),
                    'has_grids': bool(all_grid_systems)
                }
            )

            logger.info(f"✅ Processing complete for {session_id} in {metadata['processing_time']:.2f}s")
            
        finally:
            doc.close()
            gc.collect()

    async def _update_processing_status(self, storage_service: StorageService, 
                                      session_id: str, status: str, 
                                      additional_data: Dict[str, Any] = None):
        """Update processing status in storage"""
        try:
            status_data = {
                'document_id': session_id,
                'status': status,
                'updated_at': datetime.utcnow().isoformat() + 'Z',
                **(additional_data or {})
            }
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_processing_status.json",
                data=json.dumps(status_data, indent=2).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")

    async def _generate_page_images(self, page: fitz.Page, page_num: int,
                                   session_id: str, storage_service: StorageService) -> Tuple[int, int]:
        """Generate BOTH thumbnail and high-quality JPEG for a page"""
        try:
            return await self._generate_images_internal(page, page_num, session_id, storage_service)
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")
            raise

    async def _generate_images_internal(self, page: fitz.Page, page_num: int,
                                       session_id: str, storage_service: StorageService) -> Tuple[int, int]:
        """Internal image generation with proper error handling"""
        logger.info(f"🖼️ Generating images for page {page_num}...")
        
        # 1. Generate thumbnail
        await self._generate_thumbnail(page, page_num, session_id, storage_service)
        
        # Small delay between operations
        await asyncio.sleep(0.1)
        
        # 2. Generate high-quality JPEG
        image_width, image_height = await self._generate_high_quality_jpeg(
            page, page_num, session_id, storage_service
        )
        
        logger.info(f"✅ Images generated for page {page_num}")
        gc.collect()
        
        return image_width, image_height

    async def _generate_thumbnail(self, page: fitz.Page, page_num: int,
                                 session_id: str, storage_service: StorageService):
        """Generate thumbnail with proper error handling"""
        try:
            # Create thumbnail pixmap
            thumb_matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
            
            pix = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: page.get_pixmap(matrix=thumb_matrix, alpha=False)
            )
            
            # Convert to JPEG
            img_data = pix.tobytes("jpeg")
            img = Image.open(io.BytesIO(img_data))
            
            # Optimize
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=self.thumbnail_quality, optimize=True)
            output.seek(0)
            
            # Upload
            blob_name = f"{session_id}_page_{page_num}_thumb.jpg"
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=blob_name,
                data=output.getvalue(),
                content_type="image/jpeg",
                metadata={
                    "page_number": str(page_num),
                    "type": "thumbnail",
                    "dpi": str(self.thumbnail_dpi),
                    "width": str(img.width),
                    "height": str(img.height)
                }
            )
            
            logger.debug(f"✅ Thumbnail uploaded: {blob_name}")
            
            # Clean up
            pix = None
            img.close()
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for page {page_num}: {e}")
            raise

    async def _generate_high_quality_jpeg(self, page: fitz.Page, page_num: int,
                                         session_id: str, storage_service: StorageService) -> Tuple[int, int]:
        """Generate high-quality JPEG with proper error handling"""
        try:
            # Create high-quality pixmap
            matrix = fitz.Matrix(self.high_quality_dpi / 72, self.high_quality_dpi / 72)
            
            pix = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: page.get_pixmap(matrix=matrix, alpha=False)
            )
            
            width, height = pix.width, pix.height
            
            # Convert to JPEG
            img_data = pix.tobytes("jpeg")
            img = Image.open(io.BytesIO(img_data))
            
            # Save with high quality
            output = io.BytesIO()
            img.save(
                output, 
                format='JPEG', 
                quality=self.jpeg_quality,
                optimize=True,
                progressive=self.jpeg_progressive
            )
            output.seek(0)
            
            # Upload
            blob_name = f"{session_id}_page_{page_num}.jpg"
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=blob_name,
                data=output.getvalue(),
                content_type="image/jpeg",
                metadata={
                    "page_number": str(page_num),
                    "type": "high_quality",
                    "dpi": str(self.high_quality_dpi),
                    "width": str(width),
                    "height": str(height)
                }
            )
            
            logger.info(f"✅ High-quality JPEG uploaded: {blob_name}")
            
            # Clean up
            pix = None
            img.close()
            gc.collect()
            
            return width, height
            
        except Exception as e:
            logger.error(f"Failed to generate high-quality JPEG for page {page_num}: {e}")
            raise

    def _detect_comprehensive_grid(self, page: fitz.Page, page_text: str, page_num: int, 
                                  image_width: int, image_height: int) -> GridSystem:
        """Simplified grid detection focused on reliability"""
        logger.debug(f"🔍 Detecting grid system for page {page_num}...")
        
        grid = GridSystem(
            page_number=page_num,
            image_width=image_width,
            image_height=image_height
        )

        try:
            # Get page dimensions
            page_rect = page.rect
            scale_x = image_width / page_rect.width
            scale_y = image_height / page_rect.height

            # Try visual detection (simplified)
            try:
                paths = page.get_drawings()
                x_lines = []
                y_lines = []
                
                for path in paths:
                    for item in path["items"]:
                        if item[0] == "l":  # Line
                            p1, p2 = item[1], item[2]
                            
                            # Horizontal lines
                            if abs(p1.y - p2.y) < self.grid_line_tolerance:
                                if abs(p1.x - p2.x) > page_rect.width * 0.3:
                                    y_lines.append(int(p1.y * scale_y))
                            
                            # Vertical lines
                            elif abs(p1.x - p2.x) < self.grid_line_tolerance:
                                if abs(p1.y - p2.y) > page_rect.height * 0.3:
                                    x_lines.append(int(p1.x * scale_x))
                
                # Remove duplicates
                if x_lines:
                    grid.x_lines = sorted(list(set(x_lines)))
                if y_lines:
                    grid.y_lines = sorted(list(set(y_lines)))
                
                if len(grid.x_lines) >= 2 and len(grid.y_lines) >= 2:
                    grid.confidence = 0.8
                    grid.detection_method = "visual"
                    
            except Exception as e:
                logger.debug(f"Visual grid detection error: {e}")

            # Extract scale if available
            scale_match = re.search(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+|[0-9]+:[0-9]+)', page_text, re.IGNORECASE)
            if scale_match:
                grid.scale = scale_match.group(1)

        except Exception as e:
            logger.error(f"Grid detection error: {e}")

        return grid

    def _extract_tables_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Simplified table extraction"""
        tables = []
        
        try:
            page_tables = page.find_tables()
            
            for idx, table in enumerate(page_tables):
                table_data = {
                    'page_number': page_num,
                    'table_index': idx,
                    'rows': []
                }
                
                # Extract content
                for row in table.extract():
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    table_data['rows'].append(cleaned_row)
                
                if table_data['rows']:
                    tables.append(table_data)
                    
        except Exception as e:
            logger.warning(f"Table extraction failed for page {page_num}: {e}")
        
        return tables

    async def _process_single_page_safe(self, doc: fitz.Document, page_num: int,
                                       session_id: str, storage_service: StorageService) -> Dict:
        """Process a single page with comprehensive error handling"""
        page_start = time.time()
        
        try:
            page = doc[page_num]
            page_actual = page_num + 1
            
            logger.info(f"📄 Processing page {page_actual}...")

            # Extract text
            page_text = page.get_text("text")[:self.max_text_per_page]
            page_analysis = self._analyze_page_content(page_text, page_actual)

            # Generate images
            image_width, image_height = await self._generate_page_images(
                page, page_actual, session_id, storage_service
            )

            # Detect grid
            grid_system = None
            if self.enable_grid_detection:
                grid_system = self._detect_comprehensive_grid(
                    page, page_text, page_actual, image_width, image_height
                )

            # Extract tables
            tables = []
            if self.enable_table_extraction:
                tables = self._extract_tables_from_page(page, page_actual)

            # Build metadata
            page_metadata = {
                'page_number': page_actual,
                'text_length': len(page_text),
                'drawing_type': page_analysis.get('drawing_type'),
                'sheet_number': page_analysis.get('sheet_number'),
                'scale': grid_system.scale if grid_system else page_analysis.get('scale'),
                'has_grid': grid_system.confidence > 0.3 if grid_system else False,
                'has_tables': len(tables) > 0,
                'table_count': len(tables),
                'image_dimensions': {
                    'width': image_width,
                    'height': image_height
                }
            }

            # Format text
            formatted_text = self._format_page_text(page_actual, page_analysis, grid_system, page_text)
            
            # Clean up
            page.clean_contents()
            
            elapsed = time.time() - page_start
            logger.info(f"✅ Page {page_actual} processed in {elapsed:.1f}s")

            return {
                'success': True,
                'text': formatted_text,
                'metadata': page_metadata,
                'grid_system': grid_system,
                'tables': tables
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing page {page_num + 1}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'page_num': page_num + 1,
                'error': str(e),
                'text': '',
                'metadata': {'page_number': page_num + 1}
            }

    async def _process_batch_safe(self, doc: fitz.Document, page_numbers: List[int],
                                 session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process batch with proper error recovery"""
        results = []
        
        for page_num in page_numbers:
            try:
                result = await asyncio.wait_for(
                    self._process_single_page_safe(doc, page_num, session_id, storage_service),
                    timeout=self.processing_timeout_per_page
                )
                results.append(result)
                
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout processing page {page_num + 1}")
                results.append({
                    'success': False,
                    'page_num': page_num + 1,
                    'error': 'Processing timeout',
                    'text': '',
                    'metadata': {'page_number': page_num + 1}
                })
            
            await asyncio.sleep(0.1)
        
        return results

    async def _save_processing_results(self, session_id: str, all_text_parts: List[str],
                                     all_tables: List[Dict], all_grid_systems: Dict[str, Any], 
                                     metadata: Dict[str, Any], processing_start: float, 
                                     storage_service: StorageService):
        """Save all processing results with proper error handling"""
        try:
            # Save text
            full_text = '\n'.join(all_text_parts)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_context.txt",
                data=full_text.encode('utf-8'),
                content_type="text/plain"
            )

            # Save grid systems
            if all_grid_systems:
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_grid_systems.json",
                    data=json.dumps(all_grid_systems, indent=2).encode('utf-8'),
                    content_type="application/json"
                )

            # Save tables
            if all_tables:
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_tables.json",
                    data=json.dumps(all_tables, indent=2).encode('utf-8'),
                    content_type="application/json"
                )

            # Create document index
            await self._create_document_index(session_id, metadata['page_details'], storage_service)

            # Update metadata
            metadata['processing_time'] = time.time() - processing_start
            metadata['grid_systems_detected'] = len(all_grid_systems)
            metadata['status'] = 'ready'
            metadata['processing_complete'] = True
            metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
            metadata['extraction_summary']['has_thumbnails'] = True
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json",
                data=json.dumps(metadata, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            
            logger.info("✅ All results saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save processing results: {e}")
            raise

    def _initialize_metadata(self, session_id: str, doc: fitz.Document,
                           pages_to_process: int, total_pages: int,
                           pdf_size_mb: float) -> Dict[str, Any]:
        """Initialize metadata structure"""
        doc_info = {}
        try:
            doc_info = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
        except:
            pass
            
        return {
            'document_id': session_id,
            'status': 'processing',
            'page_count': pages_to_process,
            'total_pages': total_pages,
            'document_info': doc_info,
            'processing_time': 0,
            'file_size_mb': round(pdf_size_mb, 2),
            'page_details': [],
            'extraction_summary': {
                'has_text': False,
                'has_images': True,
                'has_grid_systems': False,
                'has_tables': False,
                'table_count': 0,
                'has_thumbnails': True
            },
            'image_settings': {
                'high_quality': {
                    'dpi': self.high_quality_dpi,
                    'quality': self.jpeg_quality,
                    'progressive': self.jpeg_progressive
                },
                'thumbnail': {
                    'dpi': self.thumbnail_dpi,
                    'quality': self.thumbnail_quality
                }
            }
        }

    def _update_extraction_summary(self, metadata: Dict[str, Any], result: Dict[str, Any]):
        """Update extraction summary"""
        if result.get('text', '').strip():
            metadata['extraction_summary']['has_text'] = True
        
        if result.get('tables'):
            metadata['extraction_summary']['has_tables'] = True

    def _format_page_text(self, page_num: int, page_analysis: Dict[str, Any],
                         grid_system: GridSystem, page_text: str) -> str:
        """Format page text with metadata"""
        header = [f"\n--- PAGE {page_num} ---"]
        if page_analysis.get('sheet_number'): 
            header.append(f"Sheet: {page_analysis['sheet_number']}")
        if page_analysis.get('drawing_type'): 
            header.append(f"Type: {page_analysis['drawing_type']}")
        if grid_system and grid_system.confidence > 0.3: 
            header.append(f"Grid: Detected")
        if grid_system and grid_system.scale: 
            header.append(f"Scale: {grid_system.scale}")
        return "\n".join(header) + f"\n\n{page_text}"

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content"""
        info = {}
        text_upper = text.upper()
        
        # Drawing type detection
        drawing_patterns = {
            'floor_plan': ['FLOOR PLAN', 'LEVEL'],
            'electrical': ['ELECTRICAL', 'POWER', 'LIGHTING'],
            'plumbing': ['PLUMBING', 'WATER', 'DRAINAGE'],
            'hvac': ['HVAC', 'MECHANICAL', 'VENTILATION'],
            'structural': ['STRUCTURAL', 'FOUNDATION', 'FRAMING']
        }
        
        for dtype, patterns in drawing_patterns.items():
            if any(p in text_upper for p in patterns):
                info['drawing_type'] = dtype
                break
        
        # Scale detection
        scale_match = re.search(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+|[0-9]+:[0-9]+)', text_upper)
        if scale_match: 
            info['scale'] = scale_match.group(1)

        # Sheet number
        sheet_match = re.search(r'(?:SHEET|DWG)[\s#:]*([A-Z]*[-\s]?[0-9]+\.?[0-9]*)', text_upper)
        if sheet_match:
            info['sheet_number'] = sheet_match.group(1).strip()
        
        return info

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   storage_service: StorageService):
        """Create document index"""
        try:
            index = {
                'document_id': session_id,
                'total_pages': len(page_details),
                'page_index': {},
                'drawing_types': defaultdict(list),
                'sheet_numbers': {},
                'grid_pages': [],
                'table_pages': []
            }
            
            for detail in page_details:
                page_num = detail['page_number']
                index['page_index'][page_num] = detail
                
                if detail.get('drawing_type'):
                    index['drawing_types'][detail['drawing_type']].append(page_num)
                
                if detail.get('sheet_number'):
                    index['sheet_numbers'][detail['sheet_number']] = page_num
                
                if detail.get('has_grid'):
                    index['grid_pages'].append(page_num)
                
                if detail.get('has_tables'):
                    index['table_pages'].append(page_num)
            
            index['drawing_types'] = dict(index['drawing_types'])
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_document_index.json",
                data=json.dumps(index, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            
        except Exception as e:
            logger.error(f"Failed to create document index: {e}")