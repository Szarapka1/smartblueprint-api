# app/services/pdf_service.py - ULTRA-RELIABLE VERSION WITH THUMBNAILS AND HIGH-QUALITY JPEG

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
    x_coordinates: Dict[str, int] = field(default_factory=dict)  # Pixel coordinates
    y_coordinates: Dict[str, int] = field(default_factory=dict)  # Pixel coordinates
    x_lines: List[int] = field(default_factory=list)  # Raw line positions
    y_lines: List[int] = field(default_factory=list)  # Raw line positions
    scale: Optional[str] = None
    confidence: float = 0.0
    detection_method: str = "none"
    image_width: int = 0  # Width of the rendered JPEG
    image_height: int = 0  # Height of the rendered JPEG
    grid_bounds: Dict[str, int] = field(default_factory=dict)  # min_x, max_x, min_y, max_y

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
        self.high_quality_dpi = 150  # For detailed viewing and AI analysis
        self.thumbnail_dpi = settings.PDF_THUMBNAIL_DPI  # 72 DPI for thumbnails
        self.jpeg_quality = 90  # High quality for main images
        self.thumbnail_quality = 75  # Lower quality for thumbnails
        self.jpeg_progressive = True  # Progressive encoding for better loading

        # Processing settings - CONSERVATIVE FOR RELIABILITY
        self.max_pages = settings.PDF_MAX_PAGES
        self.batch_size = min(5, settings.PROCESSING_BATCH_SIZE)  # Max 5 pages at once
        self.processing_timeout_per_page = 120  # 2 minutes per page
        self.image_generation_timeout = 180  # 3 minutes for image generation

        # Text and table extraction settings
        self.max_text_per_page = 150000
        self.enable_grid_detection = True
        self.enable_table_extraction = True
        self.grid_line_tolerance = 5  # pixels

        # Memory management - AGGRESSIVE
        self.gc_frequency = 2  # Run GC every 2 pages
        self.processing_delay = 0.5  # Delay between batches
        self.memory_warning_threshold_mb = 500  # Warn if process uses > 500MB

        logger.info("✅ PDFService initialized (ULTRA-RELIABLE VERSION)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   📦 Batch size: {self.batch_size} pages")
        logger.info(f"   🖼️ High-quality: {self.high_quality_dpi} DPI, Quality={self.jpeg_quality}")
        logger.info(f"   🖼️ Thumbnails: {self.thumbnail_dpi} DPI, Quality={self.thumbnail_quality}")
        logger.info(f"   ⏱️ Timeout: {self.processing_timeout_per_page}s per page")

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

                # Calculate total timeout based on file size
                estimated_pages = min(int(pdf_size_mb * 2), self.max_pages)  # Rough estimate
                total_timeout = estimated_pages * self.processing_timeout_per_page
                
                logger.info(f"⏱️ Using timeout: {total_timeout}s for estimated {estimated_pages} pages")

                # Process with timeout
                try:
                    await asyncio.wait_for(
                        self._process_pdf_internal(session_id, pdf_bytes, storage_service),
                        timeout=total_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"❌ Processing timeout after {total_timeout}s")
                    raise RuntimeError(f"PDF processing timeout after {total_timeout}s")

            except Exception as e:
                logger.error(f"❌ Processing failed: {e}")
                logger.error(traceback.format_exc())
                await self._save_error_state(session_id, str(e), storage_service)
                raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _process_pdf_internal(self, session_id: str, pdf_bytes: bytes,
                                   storage_service: StorageService):
        """Internal PDF processing with detailed progress tracking"""
        processing_start = time.time()
        
        try:
            # Open PDF with timeout protection
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

            # Initialize metadata
            metadata = self._initialize_metadata(session_id, doc, pages_to_process, total_pages, len(pdf_bytes) / (1024 * 1024))
            all_text_parts = []
            all_tables = []
            all_grid_systems = {}
            pages_processed = 0

            # Process in small batches
            for batch_start in range(0, pages_to_process, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pages_to_process)
                batch_pages = list(range(batch_start, batch_end))
                
                batch_num = (batch_start // self.batch_size) + 1
                total_batches = (pages_to_process + self.batch_size - 1) // self.batch_size
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches}: pages {batch_start + 1}-{batch_end}")

                # Process batch with error recovery
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
                
                # Memory management
                if batch_num % self.gc_frequency == 0:
                    logger.debug("🧹 Running garbage collection...")
                    gc.collect()
                    await asyncio.sleep(0.1)  # Give system time to breathe
                
                # Progress update
                progress = (pages_processed / pages_to_process) * 100
                elapsed = time.time() - processing_start
                rate = pages_processed / elapsed if elapsed > 0 else 0
                eta = (pages_to_process - pages_processed) / rate if rate > 0 else 0
                
                logger.info(f"📊 Progress: {progress:.1f}% ({pages_processed}/{pages_to_process} pages)")
                logger.info(f"   Speed: {rate:.1f} pages/sec, ETA: {eta:.0f}s")
                
                # Delay between batches
                if batch_end < pages_to_process:
                    logger.debug(f"⏸️ Pausing {self.processing_delay}s before next batch...")
                    await asyncio.sleep(self.processing_delay)

            # Save all results
            logger.info("💾 Saving processing results...")
            await self._save_processing_results(
                session_id, all_text_parts, all_tables, all_grid_systems, 
                metadata, processing_start, storage_service
            )

            logger.info(f"✅ Processing complete for {session_id} in {metadata['processing_time']:.2f}s")
            logger.info(f"   Pages: {pages_processed}, Tables: {len(all_tables)}, Grids: {len(all_grid_systems)}")
            
        finally:
            # Clean up
            logger.debug("🧹 Cleaning up PDF document...")
            doc.close()
            gc.collect()

    async def _generate_page_images(self, page: fitz.Page, page_num: int,
                                   session_id: str, storage_service: StorageService) -> Tuple[int, int]:
        """Generate BOTH thumbnail and high-quality JPEG for a page with timeout protection"""
        try:
            # Use timeout for image generation
            return await asyncio.wait_for(
                self._generate_images_internal(page, page_num, session_id, storage_service),
                timeout=self.image_generation_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Image generation timeout for page {page_num}")
            raise RuntimeError(f"Image generation timeout for page {page_num}")
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")
            raise

    async def _generate_images_internal(self, page: fitz.Page, page_num: int,
                                       session_id: str, storage_service: StorageService) -> Tuple[int, int]:
        """Internal image generation with memory management"""
        logger.info(f"🖼️ Generating images for page {page_num}...")
        
        # 1. Generate thumbnail first (faster, lower memory)
        logger.debug(f"   Creating thumbnail...")
        await self._generate_thumbnail(page, page_num, session_id, storage_service)
        
        # Small delay and GC between operations
        await asyncio.sleep(0.1)
        gc.collect()
        
        # 2. Generate high-quality JPEG
        logger.debug(f"   Creating high-quality JPEG...")
        image_width, image_height = await self._generate_high_quality_jpeg(
            page, page_num, session_id, storage_service
        )
        
        logger.info(f"✅ Images generated for page {page_num}")
        gc.collect()
        
        return image_width, image_height

    async def _generate_thumbnail(self, page: fitz.Page, page_num: int,
                                 session_id: str, storage_service: StorageService):
        """Generate thumbnail image for quick preview with error handling"""
        try:
            # Create thumbnail matrix
            thumb_matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
            
            # Render page at thumbnail resolution
            logger.debug(f"   Rendering thumbnail at {self.thumbnail_dpi} DPI...")
            pix = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: page.get_pixmap(matrix=thumb_matrix, alpha=False)
            )
            
            # Convert to PIL Image
            img_data = pix.tobytes("jpeg")
            img = Image.open(io.BytesIO(img_data))
            
            # Optimize thumbnail
            output = io.BytesIO()
            img.save(
                output, 
                format='JPEG', 
                quality=self.thumbnail_quality,
                optimize=True
            )
            output.seek(0)
            
            # Upload to storage with retry
            blob_name = f"{session_id}_page_{page_num}_thumb.jpg"
            thumb_size = len(output.getvalue())
            
            logger.debug(f"   Uploading thumbnail ({thumb_size / 1024:.1f}KB)...")
            
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
                    "height": str(img.height),
                    "quality": str(self.thumbnail_quality)
                }
            )
            
            logger.debug(f"✅ Thumbnail uploaded: {img.width}x{img.height} pixels")
            
            # Clean up immediately
            pix = None
            img.close()
            del img
            del output
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for page {page_num}: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _generate_high_quality_jpeg(self, page: fitz.Page, page_num: int,
                                         session_id: str, storage_service: StorageService) -> Tuple[int, int]:
        """Generate high-quality JPEG for detailed viewing and AI analysis"""
        try:
            # Create high-quality matrix
            matrix = fitz.Matrix(self.high_quality_dpi / 72, self.high_quality_dpi / 72)
            
            # Render page at high resolution
            logger.debug(f"   Rendering high-quality at {self.high_quality_dpi} DPI...")
            pix = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: page.get_pixmap(matrix=matrix, alpha=False)
            )
            
            # Get dimensions before conversion
            width, height = pix.width, pix.height
            logger.debug(f"   Rendered size: {width}x{height} pixels")
            
            # Convert to PIL Image for better compression
            img_data = pix.tobytes("jpeg")
            img = Image.open(io.BytesIO(img_data))
            
            # Save as progressive JPEG with high quality
            output = io.BytesIO()
            img.save(
                output, 
                format='JPEG', 
                quality=self.jpeg_quality,
                optimize=True,
                progressive=self.jpeg_progressive
            )
            output.seek(0)
            
            # Upload to storage
            blob_name = f"{session_id}_page_{page_num}.jpg"
            jpeg_size = len(output.getvalue())
            
            logger.debug(f"   Uploading high-quality JPEG ({jpeg_size / (1024*1024):.1f}MB)...")
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=blob_name,
                data=output.getvalue(),
                content_type="image/jpeg",
                metadata={
                    "page_number": str(page_num),
                    "type": "high_quality",
                    "dpi": str(self.high_quality_dpi),
                    "quality": str(self.jpeg_quality),
                    "width": str(width),
                    "height": str(height),
                    "progressive": str(self.jpeg_progressive)
                }
            )
            
            logger.info(f"✅ High-quality JPEG uploaded: {width}x{height} pixels ({jpeg_size / (1024*1024):.1f}MB)")
            
            # Clean up immediately
            pix = None
            img.close()
            del img
            del output
            gc.collect()
            
            return width, height
            
        except Exception as e:
            logger.error(f"Failed to generate high-quality JPEG for page {page_num}: {e}")
            logger.error(traceback.format_exc())
            raise

    def _detect_comprehensive_grid(self, page: fitz.Page, page_text: str, page_num: int, 
                                  image_width: int, image_height: int) -> GridSystem:
        """
        Detect grid with pixel-accurate coordinates relative to the JPEG image.
        Combines visual line detection and text analysis for maximum accuracy.
        """
        logger.debug(f"🔍 Detecting grid system for page {page_num}...")
        
        grid = GridSystem(
            page_number=page_num,
            image_width=image_width,
            image_height=image_height
        )

        try:
            # Get page-to-image transformation matrix
            page_rect = page.rect
            scale_x = image_width / page_rect.width
            scale_y = image_height / page_rect.height

            # 1. Visual Grid Detection (High Priority)
            try:
                visual_grid = self._detect_visual_grid_lines(page, scale_x, scale_y)
                if visual_grid and visual_grid['confidence'] > 0.6:
                    grid.x_lines = visual_grid['x_lines']
                    grid.y_lines = visual_grid['y_lines']
                    grid.confidence = visual_grid['confidence']
                    grid.detection_method = "visual"
                    logger.debug(f"   Visual grid: {len(grid.x_lines)}x{len(grid.y_lines)} lines")
            except Exception as e:
                logger.warning(f"Visual grid detection failed: {e}")

            # 2. Text-based Grid Label Detection
            try:
                text_grid = self._extract_grid_labels_from_text(page, page_text, scale_x, scale_y)
                if text_grid:
                    grid.x_labels = text_grid['x_labels']
                    grid.y_labels = text_grid['y_labels']
                    grid.x_coordinates = text_grid['x_coordinates']
                    grid.y_coordinates = text_grid['y_coordinates']
                    
                    if grid.detection_method == "none":
                        grid.detection_method = "text"
                        grid.confidence = 0.5
                    else:
                        grid.detection_method = "combined"
                        grid.confidence = min(0.95, grid.confidence + 0.2)
                    
                    logger.debug(f"   Text grid: {len(grid.x_labels)} x-labels, {len(grid.y_labels)} y-labels")
            except Exception as e:
                logger.warning(f"Text grid extraction failed: {e}")

            # 3. If we have lines but no labels, generate synthetic labels
            if grid.x_lines and not grid.x_labels:
                grid.x_labels = [chr(65 + i) for i in range(len(grid.x_lines))]  # A, B, C...
                grid.x_coordinates = {label: int(line) for label, line in zip(grid.x_labels, sorted(grid.x_lines))}

            if grid.y_lines and not grid.y_labels:
                grid.y_labels = [str(i + 1) for i in range(len(grid.y_lines))]  # 1, 2, 3...
                grid.y_coordinates = {label: int(line) for label, line in zip(grid.y_labels, sorted(grid.y_lines))}

            # 4. Extract scale from title block
            grid.scale = self._extract_scale_from_page(page_text, page)

            # 5. Calculate grid bounds
            if grid.x_coordinates:
                x_values = list(grid.x_coordinates.values())
                grid.grid_bounds['min_x'] = min(x_values)
                grid.grid_bounds['max_x'] = max(x_values)
            
            if grid.y_coordinates:
                y_values = list(grid.y_coordinates.values())
                grid.grid_bounds['min_y'] = min(y_values)
                grid.grid_bounds['max_y'] = max(y_values)

            logger.debug(f"✅ Grid detection complete: {grid.detection_method} method, confidence: {grid.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Grid detection error: {e}")
            logger.error(traceback.format_exc())

        return grid

    def _detect_visual_grid_lines(self, page: fitz.Page, scale_x: float, scale_y: float) -> Dict[str, Any]:
        """Detect grid lines visually from the page"""
        try:
            paths = page.get_drawings()
            x_lines = []
            y_lines = []
            page_width = page.rect.width
            page_height = page.rect.height

            for path in paths:
                for item in path["items"]:
                    if item[0] == "l":  # Line
                        p1, p2 = item[1], item[2]
                        
                        # Horizontal lines
                        if abs(p1.y - p2.y) < self.grid_line_tolerance:
                            line_length = abs(p1.x - p2.x)
                            if line_length > page_width * 0.3:  # Significant line
                                y_lines.append(int(p1.y * scale_y))
                        
                        # Vertical lines
                        elif abs(p1.x - p2.x) < self.grid_line_tolerance:
                            line_length = abs(p1.y - p2.y)
                            if line_length > page_height * 0.3:  # Significant line
                                x_lines.append(int(p1.x * scale_x))

            # Cluster lines to remove duplicates
            x_lines = self._cluster_lines(x_lines)
            y_lines = self._cluster_lines(y_lines)

            confidence = 0.0
            if len(x_lines) >= 2 and len(y_lines) >= 2:
                confidence = min(0.9, (len(x_lines) * len(y_lines)) / 100.0)

            return {
                'x_lines': sorted(x_lines),
                'y_lines': sorted(y_lines),
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Visual grid detection error: {e}")
            return {'x_lines': [], 'y_lines': [], 'confidence': 0.0}

    def _extract_grid_labels_from_text(self, page: fitz.Page, page_text: str, 
                                      scale_x: float, scale_y: float) -> Dict[str, Any]:
        """Extract grid labels and their pixel coordinates from text"""
        try:
            x_labels = []
            y_labels = []
            x_coordinates = {}
            y_coordinates = {}

            # Get text with position information
            text_instances = page.get_text("dict")
            
            # Pattern for grid references
            grid_patterns = [
                re.compile(r'^([A-Z]+)$'),  # Single letters
                re.compile(r'^([0-9]+)$'),  # Single numbers
                re.compile(r'([A-Z]+)[-/]([0-9]+)'),  # Combined references
            ]

            for block in text_instances.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            bbox = span.get("bbox", [])
                            
                            if len(bbox) == 4 and text:
                                # Center of text
                                center_x = int((bbox[0] + bbox[2]) / 2 * scale_x)
                                center_y = int((bbox[1] + bbox[3]) / 2 * scale_y)
                                
                                # Check if it's a grid label
                                for pattern in grid_patterns:
                                    match = pattern.match(text)
                                    if match:
                                        if text.isalpha() and len(text) <= 2:  # X-axis label
                                            if text not in x_labels:
                                                x_labels.append(text)
                                                x_coordinates[text] = center_x
                                        elif text.isdigit() and int(text) <= 100:  # Y-axis label
                                            if text not in y_labels:
                                                y_labels.append(text)
                                                y_coordinates[text] = center_y

            return {
                'x_labels': sorted(x_labels),
                'y_labels': sorted(y_labels, key=lambda x: int(x) if x.isdigit() else 0),
                'x_coordinates': x_coordinates,
                'y_coordinates': y_coordinates
            }
        except Exception as e:
            logger.error(f"Text grid extraction error: {e}")
            return {'x_labels': [], 'y_labels': [], 'x_coordinates': {}, 'y_coordinates': {}}

    def _extract_scale_from_page(self, page_text: str, page: fitz.Page) -> Optional[str]:
        """Extract scale from title block or page text"""
        try:
            # Common scale patterns
            scale_patterns = [
                re.compile(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+)', re.IGNORECASE),
                re.compile(r'SCALE[\s:]+([0-9]+:[0-9]+)', re.IGNORECASE),
                re.compile(r'([0-9/]+)"\s*=\s*([0-9]+\'-[0-9]+"?)', re.IGNORECASE),
                re.compile(r'SCALE[\s:]+FULL\s*SIZE', re.IGNORECASE),
                re.compile(r'SCALE[\s:]+N\.?T\.?S\.?', re.IGNORECASE),  # Not to scale
            ]

            # Search in page text
            for pattern in scale_patterns:
                match = pattern.search(page_text)
                if match:
                    if "FULL" in match.group(0).upper():
                        return "FULL SIZE"
                    elif "N.T.S" in match.group(0).upper() or "NTS" in match.group(0).upper():
                        return "N.T.S."
                    else:
                        return match.group(1) if match.lastindex else match.group(0)

            # Look specifically in title block area (typically bottom right)
            try:
                page_rect = page.rect
                title_block_rect = fitz.Rect(
                    page_rect.width * 0.6,  # Right 40% of page
                    page_rect.height * 0.7,  # Bottom 30% of page
                    page_rect.width,
                    page_rect.height
                )
                
                title_text = page.get_textbox(title_block_rect)
                for pattern in scale_patterns:
                    match = pattern.search(title_text)
                    if match:
                        return match.group(1) if match.lastindex else match.group(0)
            except:
                pass

            return None
        except Exception as e:
            logger.error(f"Scale extraction error: {e}")
            return None

    def _cluster_lines(self, lines: List[int], tolerance: int = 10) -> List[int]:
        """Cluster nearby lines to remove duplicates"""
        if not lines:
            return []
        
        lines.sort()
        clusters = []
        current_cluster = [lines[0]]
        
        for i in range(1, len(lines)):
            if lines[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(lines[i])
            else:
                # Add average of cluster
                clusters.append(int(sum(current_cluster) / len(current_cluster)))
                current_cluster = [lines[i]]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(int(sum(current_cluster) / len(current_cluster)))
        
        return clusters

    def _extract_tables_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables with their content"""
        tables = []
        
        try:
            logger.debug(f"📊 Extracting tables from page {page_num}...")
            
            # Get page tables
            page_tables = page.find_tables()
            
            for idx, table in enumerate(page_tables):
                table_data = {
                    'page_number': page_num,
                    'table_index': idx,
                    'bbox': list(table.bbox) if hasattr(table, 'bbox') else None,
                    'rows': []
                }
                
                # Extract table content
                for row in table.extract():
                    # Clean row data
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    table_data['rows'].append(cleaned_row)
                
                # Only add tables with content
                if table_data['rows'] and any(any(cell for cell in row) for row in table_data['rows']):
                    tables.append(table_data)
                    
            if tables:
                logger.debug(f"   Found {len(tables)} tables")
                    
        except Exception as e:
            logger.warning(f"Table extraction failed for page {page_num}: {e}")
        
        return tables

    async def _process_single_page_safe(self, doc: fitz.Document, page_num: int,
                                       session_id: str, storage_service: StorageService) -> Dict:
        """Process a single page with comprehensive extraction and error recovery"""
        page_start = time.time()
        
        try:
            page = doc[page_num]
            page_actual = page_num + 1
            
            logger.info(f"📄 Processing page {page_actual}...")

            # Extract text with size limit
            logger.debug("   Extracting text...")
            page_text = page.get_text("text")[:self.max_text_per_page]
            page_analysis = self._analyze_page_content(page_text, page_actual)

            # Generate BOTH thumbnail and high-quality JPEG
            logger.debug("   Generating images...")
            image_width, image_height = await self._generate_page_images(
                page, page_actual, session_id, storage_service
            )

            # Detect comprehensive grid system with pixel coordinates
            grid_system = None
            if self.enable_grid_detection:
                logger.debug("   Detecting grid system...")
                grid_system = self._detect_comprehensive_grid(
                    page, page_text, page_actual, image_width, image_height
                )

            # Extract tables
            tables = []
            if self.enable_table_extraction:
                logger.debug("   Extracting tables...")
                tables = self._extract_tables_from_page(page, page_actual)

            # Build page metadata
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

            # Format text with metadata
            formatted_text = self._format_page_text(page_actual, page_analysis, grid_system, page_text)
            
            # Clean up page resources
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
            elapsed = time.time() - page_start
            logger.error(f"❌ Error processing page {page_num + 1} after {elapsed:.1f}s: {e}")
            logger.error(traceback.format_exc())
            
            # Return partial result
            return {
                'success': False,
                'page_num': page_num + 1,
                'error': str(e),
                'text': '',
                'metadata': {'page_number': page_num + 1}
            }

    async def _process_batch_safe(self, doc: fitz.Document, page_numbers: List[int],
                                 session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process batch with error recovery and timeout management"""
        # Process pages sequentially to avoid memory issues
        results = []
        
        for page_num in page_numbers:
            try:
                # Process with timeout
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
                
            # Small delay between pages
            await asyncio.sleep(0.1)
        
        return results

    async def _save_processing_results(self, session_id: str, all_text_parts: List[str],
                                     all_tables: List[Dict], all_grid_systems: Dict[str, Any], 
                                     metadata: Dict[str, Any], processing_start: float, 
                                     storage_service: StorageService):
        """Save all processing results including tables with progress tracking"""
        try:
            logger.info("💾 Saving processing results...")
            save_start = time.time()
            
            # Save full text
            logger.debug("   Saving context text...")
            full_text = '\n'.join(all_text_parts)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_context.txt",
                data=full_text.encode('utf-8'),
                content_type="text/plain"
            )

            # Save comprehensive grid systems
            if all_grid_systems:
                logger.debug("   Saving grid systems...")
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_grid_systems.json",
                    data=json.dumps(all_grid_systems, indent=2).encode('utf-8'),
                    content_type="application/json"
                )

            # Save tables if any
            if all_tables:
                logger.debug(f"   Saving {len(all_tables)} tables...")
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_tables.json",
                    data=json.dumps(all_tables, indent=2).encode('utf-8'),
                    content_type="application/json"
                )
                metadata['extraction_summary']['has_tables'] = True
                metadata['extraction_summary']['table_count'] = len(all_tables)

            # Create document index
            logger.debug("   Creating document index...")
            await self._create_document_index(session_id, metadata['page_details'], storage_service)

            # Update and save metadata
            metadata['processing_time'] = time.time() - processing_start
            metadata['grid_systems_detected'] = len(all_grid_systems)
            metadata['image_settings'] = {
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
            
            logger.debug("   Saving metadata...")
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json",
                data=json.dumps(metadata, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            
            save_elapsed = time.time() - save_start
            logger.info(f"✅ Results saved in {save_elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to save processing results: {e}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_metadata(self, session_id: str, doc: fitz.Document,
                           pages_to_process: int, total_pages: int,
                           pdf_size_mb: float) -> Dict[str, Any]:
        """Initialize metadata structure"""
        doc_info = {}
        try:
            doc_info = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
        except:
            logger.warning("Could not extract document metadata")
            
        return {
            'document_id': session_id,
            'page_count': pages_to_process,
            'total_pages': total_pages,
            'document_info': doc_info,
            'processing_time': 0,
            'file_size_mb': round(pdf_size_mb, 2),
            'page_details': [],
            'grid_detection_enabled': self.enable_grid_detection,
            'table_extraction_enabled': self.enable_table_extraction,
            'extraction_summary': {
                'has_text': False,
                'has_images': True,
                'has_grid_systems': False,
                'has_tables': False,
                'table_count': 0,
                'has_thumbnails': True  # Always true now
            },
            'processing_settings': {
                'batch_size': self.batch_size,
                'timeout_per_page': self.processing_timeout_per_page,
                'high_quality_dpi': self.high_quality_dpi,
                'thumbnail_dpi': self.thumbnail_dpi
            }
        }

    def _update_extraction_summary(self, metadata: Dict[str, Any], result: Dict[str, Any]):
        """Update extraction summary based on page results"""
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
            header.append(f"Grid: {len(grid_system.x_labels)}x{len(grid_system.y_labels)}")
        if grid_system and grid_system.scale: 
            header.append(f"Scale: {grid_system.scale}")
        return "\n".join(header) + f"\n\n{page_text}"

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content with enhanced pattern matching"""
        info = {}
        text_upper = text.upper()
        
        # Drawing type detection
        drawing_patterns = {
            'floor_plan': ['FLOOR PLAN', 'LEVEL', 'STORY'],
            'electrical': ['ELECTRICAL', 'POWER', 'LIGHTING'],
            'plumbing': ['PLUMBING', 'WATER', 'DRAINAGE', 'SEWER'],
            'hvac': ['HVAC', 'MECHANICAL', 'VENTILATION', 'AIR CONDITIONING'],
            'structural': ['STRUCTURAL', 'FOUNDATION', 'FRAMING', 'BEAM', 'COLUMN'],
            'architectural': ['ARCHITECTURAL', 'ELEVATION', 'SECTION', 'DETAIL'],
            'site': ['SITE PLAN', 'LANDSCAPE', 'GRADING'],
            'fire': ['FIRE PROTECTION', 'SPRINKLER', 'FIRE ALARM']
        }
        
        for dtype, patterns in drawing_patterns.items():
            if any(p in text_upper for p in patterns):
                info['drawing_type'] = dtype
                break
        
        # Scale extraction (handled more comprehensively in grid detection)
        scale_match = re.search(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+)', text_upper)
        if scale_match: 
            info['scale'] = scale_match.group(1)

        # Sheet number extraction
        sheet_patterns = [
            re.compile(r'(?:SHEET|DWG)[\s#:]*([A-Z]*[-\s]?[0-9]+\.?[0-9]*)', re.IGNORECASE),
            re.compile(r'([A-Z]-[0-9]+(?:\.[0-9]+)?)', re.IGNORECASE),  # A-101 format
            re.compile(r'([A-Z]{1,2}[0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)  # A101 format
        ]
        
        for pattern in sheet_patterns:
            match = pattern.search(text_upper)
            if match:
                info['sheet_number'] = match.group(1).strip()
                break
        
        return info

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   storage_service: StorageService):
        """Create a searchable index of the document's contents"""
        try:
            logger.debug("📑 Creating document index...")
            
            index = {
                'document_id': session_id,
                'total_pages': len(page_details),
                'page_index': {},
                'drawing_types': defaultdict(list),
                'sheet_numbers': {},
                'grid_pages': [],
                'table_pages': [],
                'scales_found': {}
            }
            
            for detail in page_details:
                page_num = detail['page_number']
                index['page_index'][page_num] = {k: v for k, v in detail.items() if k != 'page_number'}
                
                if detail.get('drawing_type'):
                    index['drawing_types'][detail['drawing_type']].append(page_num)
                
                if detail.get('sheet_number'):
                    index['sheet_numbers'][detail['sheet_number']] = page_num
                
                if detail.get('has_grid'):
                    index['grid_pages'].append(page_num)
                
                if detail.get('has_tables'):
                    index['table_pages'].append(page_num)
                
                if detail.get('scale'):
                    index['scales_found'][page_num] = detail['scale']
            
            index['drawing_types'] = dict(index['drawing_types'])
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_document_index.json",
                data=json.dumps(index, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            
            logger.debug("✅ Document index created")
            
        except Exception as e:
            logger.error(f"Failed to create document index: {e}")
            logger.error(traceback.format_exc())

    async def _save_error_state(self, session_id: str, error: str,
                               storage_service: StorageService):
        """Save error information for debugging"""
        error_info = {
            'document_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'processing_config': {
                'high_quality_dpi': self.high_quality_dpi,
                'thumbnail_dpi': self.thumbnail_dpi,
                'jpeg_quality': self.jpeg_quality,
                'thumbnail_quality': self.thumbnail_quality,
                'max_pages': self.max_pages,
                'batch_size': self.batch_size,
                'timeout_per_page': self.processing_timeout_per_page
            }
        }
        try:
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_error.json",
                data=json.dumps(error_info, indent=2).encode('utf-8'),
                content_type="application/json"
            )
            logger.info("💾 Error state saved for debugging")
        except Exception as e:
            logger.error(f"Failed to save error state: {e}")