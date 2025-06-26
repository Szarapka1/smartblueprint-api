# app/services/pdf_service.py - ENHANCED WITH GRID DETECTION

import logging
import fitz  # PyMuPDF
import json
import os
import asyncio
import gc
from PIL import Image, ImageDraw
import io
import re
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

# Optional imports for grid detection
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass
class GridSystem:
    """Grid system for a blueprint page"""
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
            'scale': self.scale
        }


class PDFService:
    """PDF processing service with grid detection for visual highlighting"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        
        # Resolution settings for different purposes
        self.image_dpi = self.settings.PDF_IMAGE_DPI  # 150 - Storage quality
        self.ai_image_dpi = self.settings.PDF_AI_DPI  # 100 - AI processing
        self.thumbnail_dpi = self.settings.PDF_THUMBNAIL_DPI  # 72 - Previews
        
        # Processing settings
        self.max_pages = self.settings.PDF_MAX_PAGES
        self.batch_size = self.settings.PROCESSING_BATCH_SIZE
        self.max_workers = min(4, os.cpu_count() or 2)
        
        # Image optimization settings
        self.png_compression_level = self.settings.PDF_PNG_COMPRESSION
        self.jpeg_quality = self.settings.PDF_JPEG_QUALITY
        self.ai_image_quality = self.settings.AI_IMAGE_QUALITY
        self.ai_max_dimension = self.settings.AI_MAX_IMAGE_DIMENSION
        
        # Grid detection settings
        self.enable_grid_detection = True
        self.grid_line_threshold = 100  # Minimum line length for grid detection
        self.grid_detection_method = 'opencv' if OPENCV_AVAILABLE else 'pattern'
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Grid patterns for text-based detection
        self.grid_patterns = {
            'column_line': re.compile(r'COLUMN\s+LINE\s+([A-Z0-9]+)', re.IGNORECASE),
            'grid_line': re.compile(r'GRID\s+(?:LINE\s+)?([A-Z0-9]+)', re.IGNORECASE),
            'axis': re.compile(r'(?:AXIS|GRID)\s+([A-Z])[^\w]', re.IGNORECASE),
            'number': re.compile(r'(?:GRID|LINE)\s+(\d+)[^\w]', re.IGNORECASE),
            'coordinate': re.compile(r'\b([A-Z])-(\d+)\b'),
        }
        
        logger.info(f"✅ PDFService initialized with grid detection")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ Image resolutions: Storage={self.image_dpi}, AI={self.ai_image_dpi}, Thumb={self.thumbnail_dpi}")
        logger.info(f"   🎯 Grid detection: {self.grid_detection_method}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   ⚡ Workers: {self.max_workers}")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService):
        """Process PDF with grid detection and multi-resolution image generation"""
        try:
            logger.info(f"🚀 Starting enhanced PDF processing for: {session_id}")
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 PDF Size: {pdf_size_mb:.1f}MB")
            
            processing_start = asyncio.get_event_loop().time()
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                total_pages = len(doc)
                pages_to_process = min(total_pages, self.max_pages)
                
                logger.info(f"📄 Processing {pages_to_process} of {total_pages} pages")
                
                # Initialize metadata
                metadata = {
                    'document_id': session_id,
                    'page_count': pages_to_process,
                    'total_pages': total_pages,
                    'document_info': dict(doc.metadata),
                    'processing_time': 0,
                    'file_size_mb': pdf_size_mb,
                    'page_details': [],
                    'grid_detection_enabled': self.enable_grid_detection,
                    'extraction_summary': {
                        'has_text': False,
                        'has_images': True,
                        'has_tables': False,
                        'has_forms': False,
                        'has_grid_systems': False
                    },
                    'optimization_stats': {
                        'total_storage_size_mb': 0,
                        'total_ai_size_mb': 0,
                        'total_thumbnail_size_mb': 0,
                        'grid_systems_detected': 0
                    }
                }
                
                # Process pages in batches
                all_text_parts = []
                all_grid_systems = {}
                total_storage_size = 0
                total_ai_size = 0
                total_thumb_size = 0
                
                for batch_start in range(0, pages_to_process, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, pages_to_process)
                    batch_pages = list(range(batch_start, batch_end))
                    
                    logger.info(f"📦 Processing batch: pages {batch_start + 1}-{batch_end}")
                    
                    # Process batch in parallel
                    batch_results = await self._process_page_batch(
                        doc, batch_pages, session_id, storage_service
                    )
                    
                    # Aggregate results
                    for page_result in batch_results:
                        if page_result['success']:
                            # Add text
                            all_text_parts.append(page_result['text'])
                            
                            # Add page metadata
                            metadata['page_details'].append(page_result['metadata'])
                            
                            # Add grid system if detected
                            if page_result.get('grid_system'):
                                page_num = page_result['metadata']['page_number']
                                all_grid_systems[str(page_num)] = page_result['grid_system'].to_dict()
                                metadata['extraction_summary']['has_grid_systems'] = True
                                metadata['optimization_stats']['grid_systems_detected'] += 1
                            
                            # Track sizes
                            sizes = page_result['metadata'].get('image_sizes', {})
                            total_storage_size += sizes.get('storage', 0)
                            total_ai_size += sizes.get('ai', 0)
                            total_thumb_size += sizes.get('thumbnail', 0)
                            
                            # Update extraction summary
                            if page_result['text'].strip():
                                metadata['extraction_summary']['has_text'] = True
                            if page_result['metadata'].get('has_tables'):
                                metadata['extraction_summary']['has_tables'] = True
                    
                    # Force garbage collection after each batch
                    gc.collect()
                    
                    # Progress update
                    progress = (batch_end / pages_to_process) * 100
                    logger.info(f"📊 Progress: {progress:.1f}%")
                
                # Save all text content
                full_text = '\n'.join(all_text_parts)
                await self._save_text_content(session_id, full_text, storage_service)
                
                # Save grid systems if any were detected
                if all_grid_systems:
                    await self._save_grid_systems(session_id, all_grid_systems, storage_service)
                    logger.info(f"🎯 Saved grid systems for {len(all_grid_systems)} pages")
                
                # Create document index for fast searching
                document_index = await self._create_document_index(
                    session_id, metadata['page_details'], full_text, storage_service
                )
                
                # Update metadata with final stats
                processing_end = asyncio.get_event_loop().time()
                metadata['processing_time'] = round(processing_end - processing_start, 2)
                metadata['optimization_stats']['total_storage_size_mb'] = round(total_storage_size / 1024, 2)
                metadata['optimization_stats']['total_ai_size_mb'] = round(total_ai_size / 1024, 2)
                metadata['optimization_stats']['total_thumbnail_size_mb'] = round(total_thumb_size / 1024, 2)
                
                # Save metadata
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_metadata.json",
                    data=json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                )
                
                # Create processing summary
                await self._create_processing_summary(session_id, metadata, storage_service)
                
                logger.info(f"✅ Processing complete for {session_id}")
                logger.info(f"   📝 Text extracted: {len(full_text)} characters")
                logger.info(f"   🖼️ Pages processed: {pages_to_process}")
                logger.info(f"   🎯 Grid systems detected: {metadata['optimization_stats']['grid_systems_detected']}")
                logger.info(f"   ⏱️ Processing time: {metadata['processing_time']}s")
                logger.info(f"   💾 Storage breakdown:")
                logger.info(f"      - Storage images: {metadata['optimization_stats']['total_storage_size_mb']:.1f}MB")
                logger.info(f"      - AI images: {metadata['optimization_stats']['total_ai_size_mb']:.1f}MB")
                logger.info(f"      - Thumbnails: {metadata['optimization_stats']['total_thumbnail_size_mb']:.1f}MB")
                
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}", exc_info=True)
            await self._save_error_state(session_id, str(e), storage_service)
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _process_page_batch(self, doc, page_numbers: List[int], 
                                 session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process a batch of pages in parallel"""
        tasks = []
        
        for page_num in page_numbers:
            task = self._process_single_page(doc[page_num], page_num, session_id, storage_service)
            tasks.append(task)
        
        # Process pages in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to process page {page_numbers[i] + 1}: {result}")
                processed_results.append({
                    'success': False,
                    'page_num': page_numbers[i] + 1,
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results

    async def _process_single_page(self, page, page_num: int, session_id: str, 
                                  storage_service: StorageService) -> Dict[str, Any]:
        """Process a single page with text extraction, image generation, and grid detection"""
        try:
            page_actual = page_num + 1
            
            # Extract text
            page_text = page.get_text()
            
            # Analyze page content
            page_analysis = self._analyze_page_content(page_text, page_actual)
            
            # Extract tables
            tables = []
            try:
                page_tables = page.find_tables()
                if page_tables:
                    for table in page_tables:
                        tables.append({
                            'data': table.extract(),
                            'bbox': list(table.bbox)
                        })
                    page_analysis['has_tables'] = True
            except:
                pass
            
            # Generate images at different resolutions
            image_results = await self._generate_all_page_images(page, page_actual)
            
            # Detect grid system
            grid_system = None
            if self.enable_grid_detection and 'storage' in image_results:
                grid_system = await self._detect_page_grid_system(
                    page, page_actual, image_results['storage'], page_text
                )
                if grid_system and grid_system.confidence > 0.3:
                    logger.info(f"🎯 Grid detected on page {page_actual}: {len(grid_system.x_labels)}x{len(grid_system.y_labels)} grid")
            
            # Upload images
            upload_tasks = []
            
            # Storage quality image
            if 'storage' in image_results:
                upload_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}.png",
                    data=image_results['storage']
                )
                upload_tasks.append(('storage', upload_task))
            
            # AI optimized image
            if 'ai' in image_results:
                upload_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}_ai.png",
                    data=image_results['ai']
                )
                upload_tasks.append(('ai', upload_task))
            
            # Thumbnail (first 20 pages only)
            if 'thumbnail' in image_results and page_actual <= 20:
                upload_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}_thumb.png",
                    data=image_results['thumbnail']
                )
                upload_tasks.append(('thumbnail', upload_task))
            
            # Wait for uploads
            await asyncio.gather(*[task for _, task in upload_tasks])
            
            # Prepare page metadata
            page_metadata = {
                'page_number': page_actual,
                'text_length': len(page_text),
                'has_text': bool(page_text.strip()),
                'has_tables': len(tables) > 0,
                'table_count': len(tables),
                'drawing_type': page_analysis.get('drawing_type'),
                'sheet_number': page_analysis.get('sheet_number'),
                'scale': page_analysis.get('scale'),
                'key_elements': page_analysis.get('key_elements', []),
                'has_grid': grid_system is not None and grid_system.confidence > 0.3,
                'grid_confidence': grid_system.confidence if grid_system else 0.0,
                'image_sizes': {
                    'storage': len(image_results.get('storage', b'')) / 1024,  # KB
                    'ai': len(image_results.get('ai', b'')) / 1024,
                    'thumbnail': len(image_results.get('thumbnail', b'')) / 1024
                }
            }
            
            # Format text for context
            formatted_text = f"\n--- PAGE {page_actual} ---\n"
            if page_analysis.get('sheet_number'):
                formatted_text += f"Sheet: {page_analysis['sheet_number']}\n"
            if page_analysis.get('drawing_type'):
                formatted_text += f"Type: {page_analysis['drawing_type']}\n"
            if grid_system and grid_system.confidence > 0.3:
                formatted_text += f"Grid: {len(grid_system.x_labels)}x{len(grid_system.y_labels)}\n"
            formatted_text += page_text
            
            return {
                'success': True,
                'page_num': page_actual,
                'text': formatted_text,
                'metadata': page_metadata,
                'tables': tables,
                'grid_system': grid_system
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return {
                'success': False,
                'page_num': page_num + 1,
                'error': str(e),
                'text': '',
                'metadata': {}
            }

    async def _generate_all_page_images(self, page, page_num: int) -> Dict[str, bytes]:
        """Generate page images at different resolutions"""
        
        async def generate_image(dpi: int, image_type: str):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._generate_page_image_optimized,
                    page, dpi, image_type
                )
            except Exception as e:
                logger.error(f"Failed to generate {image_type} image for page {page_num}: {e}")
                return None
        
        # Generate all image types in parallel
        tasks = {
            'storage': generate_image(self.image_dpi, 'storage'),
            'ai': generate_image(self.ai_image_dpi, 'ai'),
            'thumbnail': generate_image(self.thumbnail_dpi, 'thumbnail')
        }
        
        results = {}
        for img_type, task in tasks.items():
            result = await task
            if result:
                results[img_type] = result
        
        return results

    def _generate_page_image_optimized(self, page, dpi: int, image_type: str) -> bytes:
        """Generate optimized page image with proper compression"""
        try:
            # Create transformation matrix
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            
            # Generate pixmap without alpha channel (smaller files)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert based on image type
            if image_type == 'thumbnail':
                # JPEG for thumbnails (much smaller)
                img_data = pix.pil_tobytes(format="JPEG", optimize=True, quality=self.jpeg_quality)
            
            elif image_type == 'ai':
                # Optimized PNG for AI processing
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize if too large
                if max(img.size) > self.ai_max_dimension:
                    img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
                
                # Save as optimized PNG
                output = io.BytesIO()
                img.save(output, format='PNG', optimize=True, compress_level=self.png_compression_level)
                img_data = output.getvalue()
            
            else:  # storage
                # High quality compressed PNG
                # FIX: Remove the compress_level parameter
                img_data = pix.tobytes(output="png")
            
            # Clean up
            pix = None
            
            return img_data
            
        except Exception as e:
            logger.error(f"Image generation error for {image_type}: {e}")
            raise

    async def _detect_page_grid_system(self, page, page_num: int, 
                                      image_bytes: bytes, page_text: str) -> Optional[GridSystem]:
        """Detect grid system on a page using multiple methods"""
        
        grid_system = None
        
        # Try OpenCV detection first if available
        if OPENCV_AVAILABLE and self.grid_detection_method == 'opencv':
            grid_system = await self._detect_grid_opencv(image_bytes, page_num)
        
        # If OpenCV fails or not available, try pattern-based detection
        if not grid_system or grid_system.confidence < 0.5:
            pattern_grid = await self._detect_grid_patterns(page, page_text, page_num)
            if pattern_grid and (not grid_system or pattern_grid.confidence > grid_system.confidence):
                grid_system = pattern_grid
        
        # If still no good grid, create estimated grid
        if not grid_system or grid_system.confidence < 0.3:
            grid_system = self._create_estimated_grid(page, page_num)
        
        return grid_system

    async def _detect_grid_opencv(self, image_bytes: bytes, page_num: int) -> Optional[GridSystem]:
        """Detect grid using OpenCV line detection"""
        if not OPENCV_AVAILABLE:
            return None
        
        def detect():
            try:
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                height, width = gray.shape
                
                # Edge detection
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                
                # Detect lines using Hough transform
                lines = cv2.HoughLinesP(
                    edges,
                    rho=1,
                    theta=np.pi/180,
                    threshold=100,
                    minLineLength=self.grid_line_threshold,
                    maxLineGap=50
                )
                
                if lines is None:
                    return None
                
                # Separate horizontal and vertical lines
                h_lines = []
                v_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate angle
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    
                    # Vertical lines (near 90 degrees)
                    if 85 <= angle <= 95:
                        v_lines.append((x1 + x2) // 2)
                    # Horizontal lines (near 0 or 180 degrees)
                    elif angle <= 5 or angle >= 175:
                        h_lines.append((y1 + y2) // 2)
                
                # Remove duplicates and sort
                v_lines = sorted(list(set(v_lines)))
                h_lines = sorted(list(set(h_lines)))
                
                # Filter to get evenly spaced grid lines
                v_grid = self._filter_grid_lines(v_lines, min_spacing=width // 30)
                h_grid = self._filter_grid_lines(h_lines, min_spacing=height // 40)
                
                if len(v_grid) < 3 or len(h_grid) < 3:
                    return None
                
                # Create grid system
                grid = GridSystem(
                    page_number=page_num,
                    x_lines=v_grid,
                    y_lines=h_grid,
                    confidence=0.7
                )
                
                # Generate labels
                grid.x_labels = [chr(65 + i) for i in range(len(v_grid))]  # A, B, C...
                grid.y_labels = [str(i + 1) for i in range(len(h_grid))]   # 1, 2, 3...
                
                # Map coordinates
                for i, (label, x) in enumerate(zip(grid.x_labels, v_grid)):
                    grid.x_coordinates[label] = x
                
                for i, (label, y) in enumerate(zip(grid.y_labels, h_grid)):
                    grid.y_coordinates[label] = y
                
                # Calculate cell dimensions
                if len(v_grid) > 1:
                    grid.cell_width = int(np.mean(np.diff(v_grid)))
                if len(h_grid) > 1:
                    grid.cell_height = int(np.mean(np.diff(h_grid)))
                
                grid.origin_x = v_grid[0] if v_grid else 100
                grid.origin_y = h_grid[0] if h_grid else 100
                
                return grid
                
            except Exception as e:
                logger.error(f"OpenCV grid detection failed: {e}")
                return None
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, detect)

    async def _detect_grid_patterns(self, page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Detect grid using text pattern matching"""
        
        def detect():
            try:
                # Find grid references in text
                x_refs = set()
                y_refs = set()
                
                # Look for grid labels
                for pattern_name, pattern in self.grid_patterns.items():
                    matches = pattern.finditer(page_text)
                    for match in matches:
                        if pattern_name in ['column_line', 'grid_line', 'axis']:
                            ref = match.group(1)
                            if ref.isalpha():
                                x_refs.add(ref)
                            elif ref.isdigit():
                                y_refs.add(ref)
                        elif pattern_name == 'number':
                            y_refs.add(match.group(1))
                        elif pattern_name == 'coordinate':
                            x_refs.add(match.group(1))
                            y_refs.add(match.group(2))
                
                if not x_refs or not y_refs:
                    return None
                
                # Sort references
                x_sorted = sorted(list(x_refs))
                y_sorted = sorted(list(y_refs), key=lambda x: int(x) if x.isdigit() else 0)
                
                # Create grid system
                grid = GridSystem(
                    page_number=page_num,
                    x_labels=x_sorted[:20],  # Limit to reasonable number
                    y_labels=y_sorted[:30],
                    confidence=0.5
                )
                
                # Try to find actual positions using text search
                width = page.rect.width
                height = page.rect.height
                
                # Estimate positions
                if grid.x_labels:
                    spacing = width / (len(grid.x_labels) + 1)
                    for i, label in enumerate(grid.x_labels):
                        grid.x_coordinates[label] = int((i + 1) * spacing)
                
                if grid.y_labels:
                    spacing = height / (len(grid.y_labels) + 1)
                    for i, label in enumerate(grid.y_labels):
                        grid.y_coordinates[label] = int((i + 1) * spacing)
                
                # Calculate cell dimensions
                if len(grid.x_labels) > 1:
                    grid.cell_width = int(width / (len(grid.x_labels) + 1))
                if len(grid.y_labels) > 1:
                    grid.cell_height = int(height / (len(grid.y_labels) + 1))
                
                return grid
                
            except Exception as e:
                logger.error(f"Pattern grid detection failed: {e}")
                return None
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, detect)

    def _filter_grid_lines(self, positions: List[int], min_spacing: int = 50) -> List[int]:
        """Filter positions to get evenly spaced grid lines"""
        if not positions:
            return []
        
        filtered = [positions[0]]
        
        for pos in positions[1:]:
            # Only add if sufficiently far from last position
            if pos - filtered[-1] >= min_spacing:
                filtered.append(pos)
        
        return filtered

    def _create_estimated_grid(self, page, page_num: int) -> GridSystem:
        """Create an estimated grid when detection fails"""
        width = page.rect.width
        height = page.rect.height
        
        # Create reasonable default grid
        num_x = min(12, max(6, int(width / 150)))  # 6-12 columns
        num_y = min(20, max(8, int(height / 150)))  # 8-20 rows
        
        grid = GridSystem(
            page_number=page_num,
            x_labels=[chr(65 + i) for i in range(num_x)],  # A-L
            y_labels=[str(i + 1) for i in range(num_y)],   # 1-20
            cell_width=int(width / (num_x + 2)),
            cell_height=int(height / (num_y + 2)),
            origin_x=int(width / (num_x + 2)),
            origin_y=int(height / (num_y + 2)),
            confidence=0.2  # Low confidence
        )
        
        # Set coordinates
        for i, label in enumerate(grid.x_labels):
            grid.x_coordinates[label] = grid.origin_x + (i * grid.cell_width)
        
        for i, label in enumerate(grid.y_labels):
            grid.y_coordinates[label] = grid.origin_y + (i * grid.cell_height)
        
        return grid

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content to identify drawing type and key information"""
        
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
            ('foundation', ['FOUNDATION PLAN', 'FOOTING PLAN', 'PILE PLAN']),
            ('framing', ['FRAMING PLAN', 'STRUCTURAL PLAN', 'BEAM LAYOUT']),
            ('roof', ['ROOF PLAN', 'ROOF FRAMING']),
            ('electrical', ['ELECTRICAL PLAN', 'POWER PLAN', 'LIGHTING PLAN']),
            ('plumbing', ['PLUMBING PLAN', 'PIPING PLAN', 'DRAINAGE']),
            ('mechanical', ['MECHANICAL PLAN', 'HVAC PLAN', 'DUCTWORK']),
            ('fire', ['FIRE PROTECTION', 'SPRINKLER PLAN', 'FIRE ALARM']),
            ('detail', ['DETAIL', 'SECTION', 'CONNECTION']),
            ('elevation', ['ELEVATION', 'BUILDING ELEVATION']),
            ('site', ['SITE PLAN', 'PLOT PLAN', 'LANDSCAPE']),
            ('schedule', ['SCHEDULE', 'EQUIPMENT LIST'])
        ]
        
        for dtype, patterns in drawing_patterns:
            for pattern in patterns:
                if isinstance(pattern, str) and pattern in text_upper:
                    info['drawing_type'] = dtype
                    break
                elif hasattr(pattern, 'search') and pattern.search(text_upper):
                    info['drawing_type'] = dtype
                    break
            if info['drawing_type']:
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
            'columns': ['COLUMN', 'COL.', r'\bC\d+\b'],
            'beams': ['BEAM', 'BM.', r'\bB\d+\b'],
            'doors': ['DOOR', 'DR.', r'\bD\d+\b'],
            'windows': ['WINDOW', 'WIN.', r'\bW\d+\b'],
            'equipment': ['EQUIPMENT', 'UNIT', 'AHU', 'RTU'],
            'sprinklers': ['SPRINKLER', 'SP', 'HEAD'],
            'outlets': ['OUTLET', 'RECEPTACLE', 'POWER'],
            'fixtures': ['FIXTURE', 'WC', 'LAV', 'SINK'],
            'catch_basins': ['CATCH BASIN', 'CB', 'DRAIN']
        }
        
        for element, keywords in element_keywords.items():
            for keyword in keywords:
                if isinstance(keyword, str) and keyword in text_upper:
                    info['key_elements'].append(element)
                    break
                elif hasattr(keyword, 'search') and keyword.search(text_upper):
                    info['key_elements'].append(element)
                    break
        
        # Remove duplicates
        info['key_elements'] = list(set(info['key_elements']))
        
        return info

    async def _save_text_content(self, session_id: str, full_text: str, 
                                storage_service: StorageService):
        """Save extracted text content"""
        context_blob = f"{session_id}_context.txt"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=context_blob,
            data=full_text.encode('utf-8')
        )
        logger.info(f"✅ Saved text content ({len(full_text)} characters)")

    async def _save_grid_systems(self, session_id: str, grid_systems: Dict[str, Dict],
                                storage_service: StorageService):
        """Save detected grid systems"""
        grid_blob = f"{session_id}_grid_systems.json"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=grid_blob,
            data=json.dumps(grid_systems, ensure_ascii=False).encode('utf-8')
        )

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   full_text: str, storage_service: StorageService) -> Dict:
        """Create searchable index for fast information retrieval"""
        index = {
            'document_id': session_id,
            'total_pages': len(page_details),
            'page_index': {},
            'drawing_types': defaultdict(list),
            'sheet_numbers': {},
            'grid_pages': [],
            'element_locations': defaultdict(list)
        }
        
        # Build index
        for page_detail in page_details:
            page_num = page_detail['page_number']
            
            # Page reference
            index['page_index'][page_num] = {
                'has_text': page_detail['has_text'],
                'drawing_type': page_detail.get('drawing_type'),
                'sheet_number': page_detail.get('sheet_number'),
                'has_grid': page_detail.get('has_grid', False),
                'key_elements': page_detail.get('key_elements', [])
            }
            
            # Drawing types
            if page_detail.get('drawing_type'):
                index['drawing_types'][page_detail['drawing_type']].append(page_num)
            
            # Sheet numbers
            if page_detail.get('sheet_number'):
                index['sheet_numbers'][page_detail['sheet_number']] = page_num
            
            # Grid pages
            if page_detail.get('has_grid'):
                index['grid_pages'].append(page_num)
            
            # Elements
            for element in page_detail.get('key_elements', []):
                index['element_locations'][element].append(page_num)
        
        # Save index
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_document_index.json",
            data=json.dumps(index, ensure_ascii=False).encode('utf-8')
        )
        
        return index

    async def _create_processing_summary(self, session_id: str, metadata: Dict,
                                       storage_service: StorageService):
        """Create processing summary"""
        summary = {
            'document_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'processing_stats': {
                'total_pages': metadata['page_count'],
                'processing_time_seconds': metadata['processing_time'],
                'file_size_mb': metadata['file_size_mb'],
                'pages_per_second': round(metadata['page_count'] / metadata['processing_time'], 2) if metadata['processing_time'] > 0 else 0
            },
            'content_summary': metadata['extraction_summary'],
            'drawing_breakdown': defaultdict(int),
            'sheets_found': [],
            'grid_detection': {
                'enabled': metadata['grid_detection_enabled'],
                'pages_with_grids': metadata['optimization_stats']['grid_systems_detected'],
                'detection_method': self.grid_detection_method
            },
            'storage_breakdown': metadata['optimization_stats']
        }
        
        # Analyze drawing types
        for page_detail in metadata['page_details']:
            if page_detail.get('drawing_type'):
                summary['drawing_breakdown'][page_detail['drawing_type']] += 1
            if page_detail.get('sheet_number'):
                summary['sheets_found'].append(page_detail['sheet_number'])
        
        summary['drawing_breakdown'] = dict(summary['drawing_breakdown'])
        summary['sheets_found'] = sorted(list(set(summary['sheets_found'])))
        
        # Save summary
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_processing_summary.json",
            data=json.dumps(summary, ensure_ascii=False).encode('utf-8')
        )

    async def _save_error_state(self, session_id: str, error: str, 
                               storage_service: StorageService):
        """Save error information for debugging"""
        error_info = {
            'document_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'status': 'failed'
        }
        
        try:
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_error.json",
                data=json.dumps(error_info).encode('utf-8')
            )
        except:
            pass

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "PDFService",
            "version": "3.0.0",
            "mode": "enhanced_with_grid_detection",
            "capabilities": {
                "max_pages": self.max_pages,
                "parallel_processing": True,
                "image_optimization": True,
                "text_indexing": True,
                "table_extraction": True,
                "grid_detection": self.enable_grid_detection,
                "multi_resolution": True
            },
            "performance": {
                "batch_size": self.batch_size,
                "parallel_workers": self.max_workers,
                "image_resolutions": {
                    "storage": self.image_dpi,
                    "ai_processing": self.ai_image_dpi,
                    "thumbnails": self.thumbnail_dpi
                },
                "compression": {
                    "png_level": self.png_compression_level,
                    "jpeg_quality": self.jpeg_quality
                },
                "grid_detection": {
                    "method": self.grid_detection_method,
                    "opencv_available": OPENCV_AVAILABLE
                }
            }
        }

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
