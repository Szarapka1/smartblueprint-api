# app/services/pdf_service.py - PRODUCTION-GRADE PDF PROCESSING

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
    """Production-grade PDF processing with proper error handling and memory management"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        self._lock = asyncio.Lock()  # Thread safety
        
        # Resolution settings - balanced for quality and performance
        self.storage_dpi = settings.PDF_HIGH_RESOLUTION  # 150 DPI for storage
        self.ai_image_dpi = settings.PDF_AI_DPI  # 100 DPI for AI
        self.thumbnail_dpi = settings.PDF_THUMBNAIL_DPI  # 72 DPI for thumbnails
        
        # Processing settings
        self.max_pages = settings.PDF_MAX_PAGES
        self.batch_size = settings.PROCESSING_BATCH_SIZE
        self.max_concurrent_images = 2
        
        # Image optimization
        self.png_compression = settings.PDF_PNG_COMPRESSION
        self.jpeg_quality = settings.PDF_JPEG_QUALITY
        self.ai_max_dimension = settings.AI_MAX_IMAGE_DIMENSION
        
        # Text extraction settings
        self.max_text_per_page = 100000  # 100KB per page
        self.enable_tables = True
        self.enable_grid_detection = True
        
        # Memory management
        self.gc_frequency = 3  # Garbage collect every 3 pages
        self.processing_delay = 0.5  # Delay between batches
        
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
        
        logger.info("✅ PDFService initialized (Production Mode)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ DPI: Storage={self.storage_dpi}, AI={self.ai_image_dpi}")
        logger.info(f"   📦 Batch size: {self.batch_size} pages")
        logger.info(f"   🔒 Thread safety: Enabled")
        logger.info(f"   💾 Memory management: Active")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService):
        """Process PDF with production-grade error handling and memory management"""
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session ID")
        
        if not pdf_bytes or not isinstance(pdf_bytes, bytes):
            raise ValueError("Invalid PDF data")
        
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValueError("Not a valid PDF file")
        
        async with self._lock:  # Thread safety
            try:
                logger.info(f"🚀 Starting PDF processing for: {session_id}")
                pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
                logger.info(f"📄 File Size: {pdf_size_mb:.1f}MB")
                
                if pdf_size_mb > self.settings.MAX_FILE_SIZE_MB:
                    raise ValueError(f"PDF too large: {pdf_size_mb:.1f}MB (max: {self.settings.MAX_FILE_SIZE_MB}MB)")
                
                processing_start = time.time()
                
                # Open PDF with error handling
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                except Exception as e:
                    raise RuntimeError(f"Failed to open PDF: {e}")
                
                try:
                    total_pages = len(doc)
                    pages_to_process = min(total_pages, self.max_pages)
                    
                    logger.info(f"📄 Processing {pages_to_process} of {total_pages} pages")
                    
                    # Initialize metadata
                    metadata = self._initialize_metadata(session_id, doc, pages_to_process, total_pages, pdf_size_mb)
                    
                    # Process in batches
                    all_text_parts = []
                    all_grid_systems = {}
                    pages_processed = 0
                    
                    for batch_start in range(0, pages_to_process, self.batch_size):
                        batch_end = min(batch_start + self.batch_size, pages_to_process)
                        batch_pages = list(range(batch_start, batch_end))
                        
                        logger.info(f"📦 Processing batch: pages {batch_start + 1}-{batch_end}")
                        
                        # Process batch
                        batch_results = await self._process_batch_safe(
                            doc, batch_pages, session_id, storage_service
                        )
                        
                        # Collect results
                        for result in batch_results:
                            if result['success']:
                                all_text_parts.append(result['text'])
                                metadata['page_details'].append(result['metadata'])
                                
                                if result.get('grid_system'):
                                    page_num = result['metadata']['page_number']
                                    all_grid_systems[str(page_num)] = result['grid_system'].to_dict()
                                    metadata['extraction_summary']['has_grid_systems'] = True
                                
                                self._update_extraction_summary(metadata, result)
                                pages_processed += 1
                        
                        # Memory management
                        if (batch_end % self.gc_frequency) == 0:
                            gc.collect()
                            logger.debug(f"🧹 Memory cleanup after {batch_end} pages")
                        
                        # Prevent overload
                        await asyncio.sleep(self.processing_delay)
                        
                        # Progress update
                        progress = (pages_processed / pages_to_process) * 100
                        logger.info(f"📊 Progress: {progress:.1f}% ({pages_processed}/{pages_to_process})")
                    
                    # Save results
                    await self._save_processing_results(
                        session_id, all_text_parts, all_grid_systems, metadata, 
                        processing_start, storage_service
                    )
                    
                    logger.info(f"✅ Processing complete for {session_id}")
                    logger.info(f"   📝 Pages processed: {pages_processed}")
                    logger.info(f"   ⏱️ Total time: {metadata['processing_time']}s")
                    
                finally:
                    doc.close()
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"❌ Processing failed: {e}", exc_info=True)
                await self._save_error_state(session_id, str(e), storage_service)
                raise RuntimeError(f"PDF processing failed: {str(e)}")

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
                'has_grid_systems': False
            }
        }

    def _update_extraction_summary(self, metadata: Dict[str, Any], result: Dict[str, Any]):
        """Update extraction summary based on page results"""
        if result['text'].strip():
            metadata['extraction_summary']['has_text'] = True
        
        if result.get('has_tables'):
            metadata['extraction_summary']['has_tables'] = True

    async def _process_batch_safe(self, doc: fitz.Document, page_numbers: List[int], 
                                 session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process batch with error recovery"""
        results = []
        
        for page_num in page_numbers:
            try:
                result = await self._process_single_page_safe(
                    doc, page_num, session_id, storage_service
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
        
        return results

    async def _process_single_page_safe(self, doc: fitz.Document, page_num: int, 
                                       session_id: str, storage_service: StorageService) -> Dict:
        """Process single page with all safety checks"""
        try:
            page = doc[page_num]
            page_actual = page_num + 1
            
            # Extract text with length limit
            page_text = page.get_text()
            if len(page_text) > self.max_text_per_page:
                page_text = page_text[:self.max_text_per_page] + "\n[Text truncated]"
            
            # Analyze page content
            page_analysis = self._analyze_page_content(page_text, page_actual)
            
            # Extract tables if enabled and present
            tables = []
            if self.enable_tables:
                tables = await self._extract_tables_safe(page)
            
            # Grid detection
            grid_system = None
            if self.enable_grid_detection:
                grid_system = self._detect_grid_patterns(page, page_text, page_actual)
            
            # Generate images
            await self._generate_and_upload_page_images_safe(
                page, page_actual, session_id, storage_service
            )
            
            # Prepare metadata
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
                'has_grid': grid_system is not None
            }
            
            # Format text for context
            formatted_text = self._format_page_text(page_actual, page_analysis, grid_system, page_text)
            
            # Clean up page
            page.clean_contents()
            
            return {
                'success': True,
                'page_num': page_actual,
                'text': formatted_text,
                'metadata': page_metadata,
                'tables': tables,
                'grid_system': grid_system,
                'has_tables': len(tables) > 0
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            raise

    async def _extract_tables_safe(self, page: fitz.Page) -> List[Dict]:
        """Safely extract tables from page"""
        tables = []
        try:
            page_tables = page.find_tables()
            if page_tables:
                for i, table in enumerate(page_tables[:3]):  # Limit to 3 tables
                    try:
                        tables.append({
                            'index': i,
                            'data': table.extract(),
                            'bbox': list(table.bbox) if hasattr(table, 'bbox') else None
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract table {i}: {e}")
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables

    def _format_page_text(self, page_num: int, page_analysis: Dict[str, Any], 
                         grid_system: Optional[GridSystem], page_text: str) -> str:
        """Format page text with metadata"""
        formatted_text = f"\n--- PAGE {page_num} ---\n"
        
        if page_analysis.get('sheet_number'):
            formatted_text += f"Sheet: {page_analysis['sheet_number']}\n"
        
        if page_analysis.get('drawing_type'):
            formatted_text += f"Type: {page_analysis['drawing_type']}\n"
        
        if grid_system:
            formatted_text += f"Grid: {len(grid_system.x_labels)}x{len(grid_system.y_labels)}\n"
        
        if page_analysis.get('scale'):
            formatted_text += f"Scale: {page_analysis['scale']}\n"
        
        formatted_text += page_text
        
        return formatted_text

    async def _generate_and_upload_page_images_safe(self, page: fitz.Page, page_num: int, 
                                                   session_id: str, storage_service: StorageService):
        """Generate and upload page images with proper error handling"""
        try:
            # Storage quality (PNG)
            await self._generate_storage_image(page, page_num, session_id, storage_service)
            
            # AI optimized (JPEG)
            await self._generate_ai_image(page, page_num, session_id, storage_service)
            
            # Thumbnail (first 5 pages only)
            if page_num <= 5:
                await self._generate_thumbnail(page, page_num, session_id, storage_service)
            
            # Force cleanup
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")
            # Don't fail the entire page processing for image generation errors
            pass

    async def _generate_storage_image(self, page: fitz.Page, page_num: int, 
                                     session_id: str, storage_service: StorageService):
        """Generate high-quality storage image"""
        storage_matrix = fitz.Matrix(self.storage_dpi / 72, self.storage_dpi / 72)
        storage_pix = page.get_pixmap(matrix=storage_matrix, alpha=False)
        
        try:
            # Convert and save as PNG
            storage_img = Image.frombytes("RGB", [storage_pix.width, storage_pix.height], storage_pix.samples)
            storage_output = io.BytesIO()
            storage_img.save(storage_output, format='PNG', optimize=True, compress_level=self.png_compression)
            
            # Upload
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_page_{page_num}.png",
                data=storage_output.getvalue(),
                content_type="image/png"
            )
        finally:
            # Clean up
            storage_pix = None
            if 'storage_img' in locals():
                storage_img.close()
            if 'storage_output' in locals():
                storage_output.close()

    async def _generate_ai_image(self, page: fitz.Page, page_num: int, 
                                session_id: str, storage_service: StorageService):
        """Generate AI-optimized image"""
        ai_matrix = fitz.Matrix(self.ai_image_dpi / 72, self.ai_image_dpi / 72)
        ai_pix = page.get_pixmap(matrix=ai_matrix, alpha=False)
        
        try:
            # Convert to PIL
            ai_img = Image.frombytes("RGB", [ai_pix.width, ai_pix.height], ai_pix.samples)
            
            # Resize if too large
            if max(ai_img.size) > self.ai_max_dimension:
                ai_img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            ai_output = io.BytesIO()
            ai_img.save(ai_output, format='JPEG', quality=self.jpeg_quality, optimize=True)
            
            # Upload
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_page_{page_num}_ai.jpg",
                data=ai_output.getvalue(),
                content_type="image/jpeg"
            )
        finally:
            # Clean up
            ai_pix = None
            if 'ai_img' in locals():
                ai_img.close()
            if 'ai_output' in locals():
                ai_output.close()

    async def _generate_thumbnail(self, page: fitz.Page, page_num: int, 
                                 session_id: str, storage_service: StorageService):
        """Generate thumbnail image"""
        thumb_matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
        thumb_pix = page.get_pixmap(matrix=thumb_matrix, alpha=False)
        
        try:
            thumb_img = Image.frombytes("RGB", [thumb_pix.width, thumb_pix.height], thumb_pix.samples)
            thumb_output = io.BytesIO()
            thumb_img.save(thumb_output, format='JPEG', quality=70, optimize=True)
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_page_{page_num}_thumb.jpg",
                data=thumb_output.getvalue(),
                content_type="image/jpeg"
            )
        finally:
            thumb_pix = None
            if 'thumb_img' in locals():
                thumb_img.close()
            if 'thumb_output' in locals():
                thumb_output.close()

    def _detect_grid_patterns(self, page: fitz.Page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Detect grid system from page with validation"""
        try:
            # Quick text-based detection
            x_refs = set()
            y_refs = set()
            
            # Search in first 10000 chars for performance
            search_text = page_text[:10000] if len(page_text) > 10000 else page_text
            
            # Look for grid patterns
            for pattern_name, pattern in self.grid_patterns.items():
                matches = list(pattern.finditer(search_text))[:20]  # Limit matches
                for match in matches:
                    if pattern_name == 'grid_ref':
                        x_refs.add(match.group(1))
                        y_refs.add(match.group(2))
                    elif pattern_name == 'coordinate':
                        x_refs.add(match.group(1))
                        y_refs.add(match.group(2))
                    elif pattern_name in ['column_line', 'grid_line', 'axis', 'column']:
                        ref = match.group(1)
                        if ref.isalpha():
                            x_refs.add(ref)
                        elif ref.isdigit():
                            y_refs.add(ref)
                    elif pattern_name == 'row':
                        y_refs.add(match.group(1))
            
            if not x_refs and not y_refs:
                return None
            
            # Create grid system with validation
            grid = GridSystem(
                page_number=page_num,
                x_labels=sorted(list(x_refs))[:20],  # Limit to 20 columns
                y_labels=sorted(list(y_refs), key=lambda x: int(x) if x.isdigit() else 0)[:30],  # Limit to 30 rows
                confidence=0.5 if x_refs and y_refs else 0.3
            )
            
            # Estimate positions
            page_width = page.rect.width
            page_height = page.rect.height
            
            if grid.x_labels:
                spacing = page_width / (len(grid.x_labels) + 1)
                for i, label in enumerate(grid.x_labels):
                    grid.x_coordinates[label] = int((i + 1) * spacing)
                grid.cell_width = int(spacing)
            
            if grid.y_labels:
                spacing = page_height / (len(grid.y_labels) + 1)
                for i, label in enumerate(grid.y_labels):
                    grid.y_coordinates[label] = int((i + 1) * spacing)
                grid.cell_height = int(spacing)
            
            logger.info(f"🎯 Grid detected on page {page_num}: {len(grid.x_labels)}x{len(grid.y_labels)}")
            
            return grid
            
        except Exception as e:
            logger.error(f"Grid detection failed: {e}")
            return None

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content with enhanced pattern matching"""
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
                elif isinstance(pattern, str):
                    # Try regex pattern
                    try:
                        if re.search(pattern, text_upper):
                            info['drawing_type'] = dtype
                            break
                    except:
                        pass
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
            'columns': ['COLUMN', 'COL.'],
            'beams': ['BEAM', 'BM.'],
            'doors': ['DOOR', 'DR.'],
            'windows': ['WINDOW', 'WIN.'],
            'equipment': ['EQUIPMENT', 'UNIT'],
            'sprinklers': ['SPRINKLER'],
            'outlets': ['OUTLET', 'RECEPTACLE']
        }
        
        for element, keywords in element_keywords.items():
            for keyword in keywords:
                if keyword in text_upper:
                    info['key_elements'].append(element)
                    break
        
        info['key_elements'] = list(set(info['key_elements']))[:10]  # Limit to 10 unique elements
        
        return info

    async def _save_processing_results(self, session_id: str, all_text_parts: List[str],
                                     all_grid_systems: Dict[str, Any], metadata: Dict[str, Any],
                                     processing_start: float, storage_service: StorageService):
        """Save all processing results with proper error handling"""
        try:
            # Save text content
            full_text = '\n'.join(all_text_parts)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_context.txt",
                data=full_text.encode('utf-8'),
                content_type="text/plain"
            )
            
            # Save grid systems if detected
            if all_grid_systems:
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_grid_systems.json",
                    data=json.dumps(all_grid_systems).encode('utf-8'),
                    content_type="application/json"
                )
            
            # Create document index
            await self._create_document_index(
                session_id, metadata['page_details'], full_text, storage_service
            )
            
            # Update and save metadata
            processing_end = time.time()
            metadata['processing_time'] = round(processing_end - processing_start, 2)
            metadata['grid_systems_detected'] = len(all_grid_systems)
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json",
                data=json.dumps(metadata, ensure_ascii=False).encode('utf-8'),
                content_type="application/json"
            )
        
        except Exception as e:
            logger.error(f"Failed to save processing results: {e}")
            raise

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   full_text: str, storage_service: StorageService):
        """Create searchable index with error handling"""
        try:
            index = {
                'document_id': session_id,
                'total_pages': len(page_details),
                'page_index': {},
                'drawing_types': defaultdict(list),
                'sheet_numbers': {},
                'grid_pages': []
            }
            
            for page_detail in page_details:
                page_num = page_detail['page_number']
                
                index['page_index'][page_num] = {
                    'has_text': page_detail['has_text'],
                    'drawing_type': page_detail.get('drawing_type'),
                    'sheet_number': page_detail.get('sheet_number'),
                    'has_grid': page_detail.get('has_grid', False)
                }
                
                if page_detail.get('drawing_type'):
                    index['drawing_types'][page_detail['drawing_type']].append(page_num)
                
                if page_detail.get('sheet_number'):
                    index['sheet_numbers'][page_detail['sheet_number']] = page_num
                
                if page_detail.get('has_grid'):
                    index['grid_pages'].append(page_num)
            
            # Convert defaultdict to regular dict for JSON serialization
            index['drawing_types'] = dict(index['drawing_types'])
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_document_index.json",
                data=json.dumps(index, ensure_ascii=False).encode('utf-8'),
                content_type="application/json"
            )
        
        except Exception as e:
            logger.error(f"Failed to create document index: {e}")
            # Non-critical error, don't fail the entire process

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
                data=json.dumps(error_info).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to save error state: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "PDFService",
            "version": "4.0.0-PRODUCTION",
            "mode": "production_grade",
            "capabilities": {
                "max_pages": self.max_pages,
                "batch_size": self.batch_size,
                "full_text_extraction": True,
                "table_extraction": True,
                "grid_detection": True,
                "multi_resolution_images": True,
                "ai_optimized": True,
                "thread_safe": True,
                "memory_managed": True
            },
            "security": {
                "input_validation": True,
                "size_limits": True,
                "thread_safety": True,
                "error_recovery": True
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        # Force garbage collection
        gc.collect()
        logger.info("✅ PDF service cleaned up")
