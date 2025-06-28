# app/services/pdf_service.py - SMART MEMORY-OPTIMIZED VERSION WITH FULL VISUAL CAPABILITY

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
    """Smart PDF processing - high quality for AI while managing memory"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        
        # SMART RESOLUTION SETTINGS - Balance quality and memory
        # For visual AI analysis, we need good quality
        self.storage_dpi = 150  # High quality for storage/viewing
        self.ai_image_dpi = 120  # Good quality for AI analysis (was 100)
        self.thumbnail_dpi = 72  # Low quality for previews
        
        # Processing settings - SMART BATCHING
        self.max_pages = self.settings.PDF_MAX_PAGES
        self.batch_size = 3  # Process 3 pages at a time (was 10)
        self.max_concurrent_images = 2  # Limit concurrent image operations
        
        # Image optimization settings - BALANCED
        self.png_compression = 6  # Balanced compression
        self.jpeg_quality = 85  # Good quality for AI
        self.ai_max_dimension = 2000  # Keep full size for AI accuracy
        
        # Text extraction settings
        self.max_text_per_page = 100000  # 100KB per page
        self.enable_tables = True  # Keep table extraction
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
        
        logger.info(f"✅ PDFService initialized - SMART MEMORY MODE")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ Image quality: Storage={self.storage_dpi}, AI={self.ai_image_dpi}")
        logger.info(f"   📦 Batch size: {self.batch_size} pages")
        logger.info(f"   🎯 Full visual analysis: ENABLED")
        logger.info(f"   💾 Memory management: ACTIVE")
        logger.info(f"   🎯 Grid detection patterns: {len(self.grid_patterns)}")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService):
        """Process PDF with smart memory management while maintaining quality"""
        try:
            logger.info(f"🚀 Starting smart PDF processing for: {session_id}")
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 File Size: {pdf_size_mb:.1f}MB")
            
            processing_start = time.time()
            
            # Open PDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            try:
                total_pages = len(doc)
                pages_to_process = min(total_pages, self.max_pages)
                
                logger.info(f"📄 Processing {pages_to_process} of {total_pages} pages")
                
                # Initialize tracking
                metadata = {
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
                
                # Process in smart batches
                all_text_parts = []
                all_grid_systems = {}
                pages_processed = 0
                
                for batch_start in range(0, pages_to_process, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, pages_to_process)
                    batch_pages = list(range(batch_start, batch_end))
                    
                    logger.info(f"📦 Processing batch: pages {batch_start + 1}-{batch_end}")
                    
                    # Process batch with memory management
                    batch_results = await self._process_batch_smart(
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
                            
                            if result['text'].strip():
                                metadata['extraction_summary']['has_text'] = True
                            
                            if result.get('has_tables'):
                                metadata['extraction_summary']['has_tables'] = True
                            
                            pages_processed += 1
                    
                    # Memory management
                    if (batch_end % self.gc_frequency) == 0:
                        gc.collect()
                        logger.info(f"🧹 Memory cleanup after {batch_end} pages")
                    
                    # Prevent overload
                    await asyncio.sleep(self.processing_delay)
                    
                    # Progress update
                    progress = (pages_processed / pages_to_process) * 100
                    logger.info(f"📊 Progress: {progress:.1f}% ({pages_processed}/{pages_to_process} pages)")
                
                # Save all text content
                full_text = '\n'.join(all_text_parts)
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_context.txt",
                    data=full_text.encode('utf-8')
                )
                
                # Save grid systems if detected
                if all_grid_systems:
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_grid_systems.json",
                        data=json.dumps(all_grid_systems).encode('utf-8')
                    )
                
                # Create document index
                await self._create_document_index(
                    session_id, metadata['page_details'], full_text, storage_service
                )
                
                # Save metadata
                processing_end = time.time()
                metadata['processing_time'] = round(processing_end - processing_start, 2)
                
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_metadata.json",
                    data=json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                )
                
                logger.info(f"✅ Processing complete for {session_id}")
                logger.info(f"   📝 Pages processed: {pages_processed}")
                logger.info(f"   ⏱️ Processing time: {metadata['processing_time']}s")
                logger.info(f"   📈 Speed: {pages_processed / metadata['processing_time']:.1f} pages/sec")
                
            finally:
                doc.close()
                gc.collect()
                
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}", exc_info=True)
            await self._save_error_state(session_id, str(e), storage_service)
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _process_batch_smart(self, doc, page_numbers: List[int], 
                                  session_id: str, storage_service: StorageService) -> List[Dict]:
        """Process a batch of pages with smart memory management"""
        results = []
        
        for page_num in page_numbers:
            try:
                result = await self._process_single_page_smart(
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
                    'metadata': {}
                })
        
        return results

    async def _process_single_page_smart(self, doc, page_num: int, 
                                        session_id: str, storage_service: StorageService) -> Dict:
        """Process single page with full quality for AI"""
        try:
            page = doc[page_num]
            page_actual = page_num + 1
            
            # Extract text with full context
            page_text = page.get_text()
            
            # Analyze page content
            page_analysis = self._analyze_page_content(page_text, page_actual)
            
            # Extract tables if present
            tables = []
            if self.enable_tables:
                try:
                    page_tables = page.find_tables()
                    if page_tables:
                        for table in page_tables[:3]:  # Limit to 3 tables
                            tables.append({
                                'data': table.extract(),
                                'bbox': list(table.bbox)
                            })
                except:
                    pass
            
            # Grid detection
            grid_system = None
            if self.enable_grid_detection:
                grid_system = self._detect_grid_patterns(page, page_text, page_actual)
            
            # Generate images with proper quality
            await self._generate_and_upload_page_images(
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
            formatted_text = f"\n--- PAGE {page_actual} ---\n"
            if page_analysis.get('sheet_number'):
                formatted_text += f"Sheet: {page_analysis['sheet_number']}\n"
            if page_analysis.get('drawing_type'):
                formatted_text += f"Type: {page_analysis['drawing_type']}\n"
            if grid_system:
                formatted_text += f"Grid: {len(grid_system.x_labels)}x{len(grid_system.y_labels)}\n"
            formatted_text += page_text
            
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
            return {
                'success': False,
                'page_num': page_num + 1,
                'error': str(e),
                'text': '',
                'metadata': {}
            }

    async def _generate_and_upload_page_images(self, page, page_num: int, 
                                              session_id: str, storage_service: StorageService):
        """Generate multiple quality images for different purposes"""
        try:
            # 1. Storage quality (PNG) - for viewing
            storage_matrix = fitz.Matrix(self.storage_dpi / 72, self.storage_dpi / 72)
            storage_pix = page.get_pixmap(matrix=storage_matrix, alpha=False)
            
            # Convert and save as PNG
            storage_img = Image.frombytes("RGB", [storage_pix.width, storage_pix.height], storage_pix.samples)
            storage_output = io.BytesIO()
            storage_img.save(storage_output, format='PNG', optimize=True, compress_level=self.png_compression)
            
            # Upload storage image
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_page_{page_num}.png",
                data=storage_output.getvalue()
            )
            
            # Clean up
            storage_pix = None
            storage_img.close()
            storage_output.close()
            
            # 2. AI optimized (JPEG) - for AI analysis
            ai_matrix = fitz.Matrix(self.ai_image_dpi / 72, self.ai_image_dpi / 72)
            ai_pix = page.get_pixmap(matrix=ai_matrix, alpha=False)
            
            # Convert to PIL
            ai_img = Image.frombytes("RGB", [ai_pix.width, ai_pix.height], ai_pix.samples)
            
            # Resize if too large
            if max(ai_img.size) > self.ai_max_dimension:
                ai_img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            ai_output = io.BytesIO()
            ai_img.save(ai_output, format='JPEG', quality=self.jpeg_quality, optimize=True)
            
            # Upload AI image
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_page_{page_num}_ai.jpg",
                data=ai_output.getvalue()
            )
            
            # Clean up
            ai_pix = None
            ai_img.close()
            ai_output.close()
            
            # 3. Thumbnail (only for first 5 pages)
            if page_num <= 5:
                thumb_matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
                thumb_pix = page.get_pixmap(matrix=thumb_matrix, alpha=False)
                
                thumb_img = Image.frombytes("RGB", [thumb_pix.width, thumb_pix.height], thumb_pix.samples)
                thumb_output = io.BytesIO()
                thumb_img.save(thumb_output, format='JPEG', quality=70, optimize=True)
                
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_num}_thumb.jpg",
                    data=thumb_output.getvalue()
                )
                
                thumb_pix = None
                thumb_img.close()
                thumb_output.close()
            
            # Force cleanup
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")

    def _detect_grid_patterns(self, page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Detect grid system from page"""
        try:
            # Quick text-based detection
            x_refs = set()
            y_refs = set()
            
            # Search in first 10000 chars
            search_text = page_text[:10000] if len(page_text) > 10000 else page_text
            
            # Look for grid patterns
            for pattern_name, pattern in self.grid_patterns.items():
                matches = list(pattern.finditer(search_text))[:20]
                for match in matches:
                    if pattern_name == 'grid_ref':
                        # Handle A-1 style references
                        x_refs.add(match.group(1))
                        y_refs.add(match.group(2))
                    elif pattern_name == 'coordinate':
                        # Handle @A-1 style references
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
            
            # Create grid system
            grid = GridSystem(
                page_number=page_num,
                x_labels=sorted(list(x_refs))[:20],
                y_labels=sorted(list(y_refs), key=lambda x: int(x) if x.isdigit() else 0)[:30],
                confidence=0.5
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
        
        info['key_elements'] = list(set(info['key_elements']))
        
        return info

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   full_text: str, storage_service: StorageService):
        """Create searchable index"""
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
        
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_document_index.json",
            data=json.dumps(index, ensure_ascii=False).encode('utf-8')
        )

    async def _save_error_state(self, session_id: str, error: str, 
                               storage_service: StorageService):
        """Save error information"""
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
            "version": "3.0.0-SMART-MEMORY",
            "mode": "balanced_quality_memory",
            "capabilities": {
                "max_pages": self.max_pages,
                "batch_size": self.batch_size,
                "full_text_extraction": True,
                "table_extraction": True,
                "grid_detection": True,
                "multi_resolution_images": True,
                "ai_optimized": True
            }
        }
