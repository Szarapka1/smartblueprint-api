# app/services/pdf_service.py - ENHANCED WITH GRID DETECTION (OPTIMIZED)

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing
import time

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
        
        # Processing settings - OPTIMIZED
        self.max_pages = self.settings.PDF_MAX_PAGES
        self.batch_size = min(10, self.settings.PROCESSING_BATCH_SIZE * 2)  # Larger batches
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # Use all available cores
        
        # Image optimization settings
        self.png_compression_level = 6  # Faster compression (was 9)
        self.jpeg_quality = self.settings.PDF_JPEG_QUALITY
        self.ai_image_quality = self.settings.AI_IMAGE_QUALITY
        self.ai_max_dimension = self.settings.AI_MAX_IMAGE_DIMENSION
        
        # Grid detection settings - OPTIMIZED
        self.enable_grid_detection = True
        self.grid_line_threshold = 100
        self.grid_detection_method = 'pattern'  # Use pattern first, it's faster
        self.defer_opencv_grid = True  # Process OpenCV grid detection after initial response
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Separate executor for CPU-intensive tasks
        self.cpu_executor = ProcessPoolExecutor(max_workers=max(2, self.max_workers // 2))
        
        # Grid patterns for text-based detection
        self.grid_patterns = {
            'column_line': re.compile(r'COLUMN\s+LINE\s+([A-Z0-9]+)', re.IGNORECASE),
            'grid_line': re.compile(r'GRID\s+(?:LINE\s+)?([A-Z0-9]+)', re.IGNORECASE),
            'axis': re.compile(r'(?:AXIS|GRID)\s+([A-Z])[^\w]', re.IGNORECASE),
            'number': re.compile(r'(?:GRID|LINE)\s+(\d+)[^\w]', re.IGNORECASE),
            'coordinate': re.compile(r'\b([A-Z])-(\d+)\b'),
        }
        
        # Cache for frequently accessed data
        self._page_cache = {}
        self._grid_cache = {}
        
        logger.info(f"✅ PDFService initialized with optimizations")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ Image resolutions: Storage={self.image_dpi}, AI={self.ai_image_dpi}, Thumb={self.thumbnail_dpi}")
        logger.info(f"   🎯 Grid detection: {self.grid_detection_method} (deferred OpenCV: {self.defer_opencv_grid})")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   ⚡ Workers: {self.max_workers} threads, {self.cpu_executor._max_workers} processes")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService):
        """Process PDF with grid detection and multi-resolution image generation"""
        try:
            logger.info(f"🚀 Starting enhanced PDF processing for: {session_id}")
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 PDF Size: {pdf_size_mb:.1f}MB")
            
            processing_start = time.time()
            
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
                
                # Create all page objects once
                pages = [doc[i] for i in range(pages_to_process)]
                
                for batch_start in range(0, pages_to_process, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, pages_to_process)
                    batch_pages = list(range(batch_start, batch_end))
                    
                    logger.info(f"📦 Processing batch: pages {batch_start + 1}-{batch_end}")
                    batch_time = time.time()
                    
                    # Process batch in parallel with optimizations
                    batch_results = await self._process_page_batch_optimized(
                        pages, batch_pages, session_id, storage_service
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
                    
                    # Log batch performance
                    batch_elapsed = time.time() - batch_time
                    pages_per_second = len(batch_pages) / batch_elapsed
                    logger.info(f"📊 Batch completed in {batch_elapsed:.1f}s ({pages_per_second:.1f} pages/sec)")
                    
                    # Progress update
                    progress = (batch_end / pages_to_process) * 100
                    logger.info(f"📊 Progress: {progress:.1f}%")
                
                # Save all text content
                full_text = '\n'.join(all_text_parts)
                text_task = self._save_text_content(session_id, full_text, storage_service)
                
                # Save grid systems if any were detected
                grid_task = None
                if all_grid_systems:
                    grid_task = self._save_grid_systems(session_id, all_grid_systems, storage_service)
                    logger.info(f"🎯 Saved grid systems for {len(all_grid_systems)} pages")
                
                # Create document index for fast searching
                index_task = self._create_document_index(
                    session_id, metadata['page_details'], full_text, storage_service
                )
                
                # Update metadata with final stats
                processing_end = time.time()
                metadata['processing_time'] = round(processing_end - processing_start, 2)
                metadata['optimization_stats']['total_storage_size_mb'] = round(total_storage_size / 1024, 2)
                metadata['optimization_stats']['total_ai_size_mb'] = round(total_ai_size / 1024, 2)
                metadata['optimization_stats']['total_thumbnail_size_mb'] = round(total_thumb_size / 1024, 2)
                
                # Save metadata
                metadata_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_metadata.json",
                    data=json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                )
                
                # Create processing summary
                summary_task = self._create_processing_summary(session_id, metadata, storage_service)
                
                # Wait for all async saves to complete
                await asyncio.gather(text_task, metadata_task, summary_task)
                if grid_task:
                    await grid_task
                await index_task
                
                # Schedule deferred OpenCV grid detection if needed
                if self.defer_opencv_grid and OPENCV_AVAILABLE:
                    asyncio.create_task(self._process_deferred_grid_detection(
                        session_id, pages_to_process, storage_service
                    ))
                
                logger.info(f"✅ Processing complete for {session_id}")
                logger.info(f"   📝 Text extracted: {len(full_text)} characters")
                logger.info(f"   🖼️ Pages processed: {pages_to_process}")
                logger.info(f"   🎯 Grid systems detected: {metadata['optimization_stats']['grid_systems_detected']}")
                logger.info(f"   ⏱️ Processing time: {metadata['processing_time']}s")
                logger.info(f"   📈 Processing speed: {pages_to_process / metadata['processing_time']:.1f} pages/sec")
                logger.info(f"   💾 Storage breakdown:")
                logger.info(f"      - Storage images: {metadata['optimization_stats']['total_storage_size_mb']:.1f}MB")
                logger.info(f"      - AI images: {metadata['optimization_stats']['total_ai_size_mb']:.1f}MB")
                logger.info(f"      - Thumbnails: {metadata['optimization_stats']['total_thumbnail_size_mb']:.1f}MB")
                
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}", exc_info=True)
            await self._save_error_state(session_id, str(e), storage_service)
            raise RuntimeError(f"PDF processing failed: {str(e)}")
        finally:
            # Clear caches
            self._page_cache.clear()
            self._grid_cache.clear()
            gc.collect()

    async def _process_page_batch_optimized(self, pages: List, page_numbers: List[int], 
                                          session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process a batch of pages in parallel with optimizations"""
        # Pre-extract text and analyze in parallel
        text_tasks = []
        for page_num in page_numbers:
            task = asyncio.create_task(self._extract_page_data_fast(pages[page_num], page_num))
            text_tasks.append(task)
        
        # Wait for text extraction
        page_data = await asyncio.gather(*text_tasks)
        
        # Process pages with extracted data
        process_tasks = []
        for i, (page_num, data) in enumerate(zip(page_numbers, page_data)):
            task = self._process_single_page_optimized(
                pages[page_num], page_num, data, session_id, storage_service
            )
            process_tasks.append(task)
        
        # Process all pages in parallel
        results = await asyncio.gather(*process_tasks, return_exceptions=True)
        
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

    async def _extract_page_data_fast(self, page, page_num: int) -> Dict:
        """Fast extraction of page text and basic analysis"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._extract_page_data_sync,
            page,
            page_num
        )
    
    def _extract_page_data_sync(self, page, page_num: int) -> Dict:
        """Synchronous page data extraction"""
        try:
            # Extract text
            page_text = page.get_text()
            
            # Quick analysis
            page_analysis = self._analyze_page_content(page_text, page_num + 1)
            
            # Extract tables (fast check)
            tables = []
            try:
                page_tables = page.find_tables()
                if page_tables:
                    for table in page_tables[:5]:  # Limit to first 5 tables for speed
                        tables.append({
                            'data': table.extract(),
                            'bbox': list(table.bbox)
                        })
            except:
                pass
            
            return {
                'text': page_text,
                'analysis': page_analysis,
                'tables': tables,
                'has_tables': len(tables) > 0
            }
        except Exception as e:
            logger.error(f"Error extracting page {page_num + 1} data: {e}")
            return {
                'text': '',
                'analysis': {},
                'tables': [],
                'has_tables': False
            }

    async def _process_single_page_optimized(self, page, page_num: int, page_data: Dict,
                                           session_id: str, storage_service: StorageService) -> Dict[str, Any]:
        """Process a single page with pre-extracted data"""
        try:
            page_actual = page_num + 1
            
            # Use pre-extracted data
            page_text = page_data['text']
            page_analysis = page_data['analysis']
            tables = page_data['tables']
            
            # Generate images in parallel
            image_task = self._generate_all_page_images_optimized(page, page_actual)
            
            # Quick grid detection (pattern-based only for speed)
            grid_system = None
            if self.enable_grid_detection and page_text:
                grid_system = await self._detect_grid_patterns_fast(page, page_text, page_actual)
            
            # Wait for images
            image_results = await image_task
            
            # Upload images in parallel
            upload_tasks = []
            
            # Storage quality image
            if 'storage' in image_results:
                upload_task = storage_service.upload_file_async(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}.png",
                    data=image_results['storage']
                )
                upload_tasks.append(('storage', upload_task))
            
            # AI optimized image
            if 'ai' in image_results:
                upload_task = storage_service.upload_file_async(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}_ai.jpg",
                    data=image_results['ai']
                )
                upload_tasks.append(('ai', upload_task))
            
            # Thumbnail (first 10 pages only for speed)
            if 'thumbnail' in image_results and page_actual <= 10:
                upload_task = storage_service.upload_file_async(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}_thumb.jpg",
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
                'has_tables': page_data['has_tables'],
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

    async def _generate_all_page_images_optimized(self, page, page_num: int) -> Dict[str, bytes]:
        """Generate page images at different resolutions - optimized"""
        
        # Generate base pixmap once and reuse
        base_matrix = fitz.Matrix(self.image_dpi / 72, self.image_dpi / 72)
        base_pix = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: page.get_pixmap(matrix=base_matrix, alpha=False)
        )
        
        # Convert to PIL Image once
        base_img = Image.frombytes("RGB", [base_pix.width, base_pix.height], base_pix.samples)
        
        async def generate_from_base(img_type: str):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._convert_image_optimized,
                    base_img,
                    img_type
                )
            except Exception as e:
                logger.error(f"Failed to generate {img_type} image for page {page_num}: {e}")
                return None
        
        # Generate all image types in parallel from base image
        tasks = {
            'storage': generate_from_base('storage'),
            'ai': generate_from_base('ai'),
            'thumbnail': generate_from_base('thumbnail')
        }
        
        results = {}
        for img_type, task in tasks.items():
            result = await task
            if result:
                results[img_type] = result
        
        # Clean up
        base_pix = None
        
        return results

    def _convert_image_optimized(self, base_img: Image.Image, image_type: str) -> bytes:
        """Convert base image to optimized format"""
        try:
            output = io.BytesIO()
            
            if image_type == 'thumbnail':
                # Create small thumbnail
                thumb = base_img.copy()
                thumb.thumbnail((400, 400), Image.Resampling.LANCZOS)
                thumb.save(output, format='JPEG', optimize=True, quality=70)
            
            elif image_type == 'ai':
                # AI processing image
                ai_img = base_img.copy()
                if max(ai_img.size) > self.ai_max_dimension:
                    ai_img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), 
                                   Image.Resampling.LANCZOS)
                ai_img.save(output, format='JPEG', optimize=True, quality=self.ai_image_quality)
            
            else:  # storage
                # Use WebP for storage - better compression than PNG
                if hasattr(Image, 'WEBP'):
                    base_img.save(output, format='WEBP', quality=90, method=4)
                else:
                    # Fallback to optimized PNG
                    base_img.save(output, format='PNG', optimize=True, compress_level=self.png_compression_level)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Image conversion error for {image_type}: {e}")
            raise

    async def _detect_grid_patterns_fast(self, page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Fast pattern-based grid detection"""
        try:
            # Check cache first
            cache_key = f"{page_num}_grid"
            if cache_key in self._grid_cache:
                return self._grid_cache[cache_key]
            
            # Quick pattern search
            x_refs = set()
            y_refs = set()
            
            # Limit text search to first 5000 characters for speed
            search_text = page_text[:5000] if len(page_text) > 5000 else page_text
            
            # Look for grid labels
            for pattern_name, pattern in self.grid_patterns.items():
                matches = list(pattern.finditer(search_text))[:10]  # Limit matches
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
            
            # Create quick grid system
            grid = GridSystem(
                page_number=page_num,
                x_labels=sorted(list(x_refs))[:15],  # Limit size
                y_labels=sorted(list(y_refs), key=lambda x: int(x) if x.isdigit() else 0)[:20],
                confidence=0.4
            )
            
            # Quick position estimation
            width = page.rect.width
            height = page.rect.height
            
            if grid.x_labels:
                spacing = width / (len(grid.x_labels) + 1)
                for i, label in enumerate(grid.x_labels):
                    grid.x_coordinates[label] = int((i + 1) * spacing)
                grid.cell_width = int(spacing)
            
            if grid.y_labels:
                spacing = height / (len(grid.y_labels) + 1)
                for i, label in enumerate(grid.y_labels):
                    grid.y_coordinates[label] = int((i + 1) * spacing)
                grid.cell_height = int(spacing)
            
            # Cache result
            self._grid_cache[cache_key] = grid
            
            return grid
            
        except Exception as e:
            logger.error(f"Fast grid detection failed: {e}")
            return None

    async def _process_deferred_grid_detection(self, session_id: str, page_count: int,
                                             storage_service: StorageService):
        """Process OpenCV grid detection in background after initial processing"""
        try:
            logger.info(f"🔍 Starting deferred OpenCV grid detection for {session_id}")
            
            # Load existing grid systems
            try:
                grid_data = await storage_service.download_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_grid_systems.json"
                )
                existing_grids = json.loads(grid_data.decode('utf-8')) if grid_data else {}
            except:
                existing_grids = {}
            
            updated = False
            
            # Process pages that need better grid detection
            for page_num in range(1, min(page_count + 1, 20)):  # Limit to first 20 pages
                str_page = str(page_num)
                
                # Skip if already has good grid
                if str_page in existing_grids and existing_grids[str_page].get('confidence', 0) > 0.6:
                    continue
                
                try:
                    # Download page image
                    img_data = await storage_service.download_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_page_{page_num}.png"
                    )
                    
                    if img_data:
                        # Run OpenCV detection
                        grid = await self._detect_grid_opencv(img_data, page_num)
                        
                        if grid and grid.confidence > existing_grids.get(str_page, {}).get('confidence', 0):
                            existing_grids[str_page] = grid.to_dict()
                            updated = True
                            logger.info(f"🎯 Enhanced grid detection for page {page_num}")
                
                except Exception as e:
                    logger.error(f"Failed deferred grid detection for page {page_num}: {e}")
            
            # Save updated grids if any improvements
            if updated:
                await self._save_grid_systems(session_id, existing_grids, storage_service)
                logger.info(f"✅ Completed deferred grid detection for {session_id}")
            
        except Exception as e:
            logger.error(f"Deferred grid detection failed: {e}")

    # Keep all the existing helper methods unchanged
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
            "version": "3.1.0",
            "mode": "enhanced_with_grid_detection_optimized",
            "capabilities": {
                "max_pages": self.max_pages,
                "parallel_processing": True,
                "image_optimization": True,
                "text_indexing": True,
                "table_extraction": True,
                "grid_detection": self.enable_grid_detection,
                "multi_resolution": True,
                "deferred_processing": self.defer_opencv_grid
            },
            "performance": {
                "batch_size": self.batch_size,
                "parallel_workers": self.max_workers,
                "process_workers": self.cpu_executor._max_workers,
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
                    "opencv_available": OPENCV_AVAILABLE,
                    "deferred": self.defer_opencv_grid
                }
            }
        }

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'cpu_executor'):
            self.cpu_executor.shutdown(wait=True)
