# app/services/pdf_service.py - OPTIMIZED FOR LARGE MULTI-PAGE DOCUMENTS

import logging
import fitz  # PyMuPDF
import json
import os
import asyncio
import gc
from PIL import Image
import io
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class PDFService:
    """Optimized PDF processing service for large multi-page documents"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        
        # OPTIMIZED SETTINGS - Lower DPI for reasonable file sizes
        self.image_dpi = int(os.getenv("PDF_IMAGE_DPI", "150"))  # Down from 200
        self.ai_image_dpi = int(os.getenv("PDF_AI_DPI", "100"))  # Down from 150
        self.thumbnail_dpi = int(os.getenv("PDF_THUMBNAIL_DPI", "72"))
        self.max_pages = int(os.getenv("PDF_MAX_PAGES", "500"))
        self.batch_size = int(os.getenv("PDF_BATCH_SIZE", "5"))
        self.max_workers = int(os.getenv("PDF_MAX_WORKERS", "4"))
        
        # Image optimization settings
        self.png_compression_level = int(os.getenv("PDF_PNG_COMPRESSION", "6"))  # 0-9, 6 is balanced
        self.jpeg_quality = int(os.getenv("PDF_JPEG_QUALITY", "85"))  # For thumbnails
        self.ai_image_quality = int(os.getenv("PDF_AI_IMAGE_QUALITY", "85"))
        self.ai_max_dimension = int(os.getenv("PDF_AI_MAX_DIMENSION", "3000"))
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"✅ PDFService initialized (Optimized for Large Documents)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ Storage DPI: {self.image_dpi} (optimized)")
        logger.info(f"   🤖 AI DPI: {self.ai_image_dpi} (optimized)")
        logger.info(f"   🖼️ Thumbnail DPI: {self.thumbnail_dpi}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   ⚡ Parallel workers: {self.max_workers}")
        logger.info(f"   🗜️ PNG compression: Level {self.png_compression_level}")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService):
        """Process large PDFs efficiently with optimized image generation"""
        try:
            logger.info(f"🚀 Starting optimized PDF processing for: {session_id}")
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 PDF Size: {pdf_size_mb:.1f}MB")
            
            # Initialize processing tracking
            processing_start = asyncio.get_event_loop().time()
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                total_pages = len(doc)
                pages_to_process = min(total_pages, self.max_pages)
                
                logger.info(f"📄 Total pages: {total_pages} (processing {pages_to_process})")
                
                # Initialize metadata
                metadata = {
                    'document_id': session_id,
                    'page_count': pages_to_process,
                    'total_pages': total_pages,
                    'document_info': dict(doc.metadata),
                    'processing_time': 0,
                    'file_size_mb': pdf_size_mb,
                    'page_details': [],
                    'extraction_summary': {
                        'has_text': False,
                        'has_images': True,
                        'has_tables': False,
                        'has_forms': False
                    },
                    'optimization_stats': {
                        'total_storage_size_mb': 0,
                        'total_ai_size_mb': 0,
                        'average_page_size_mb': 0
                    }
                }
                
                # Process pages in batches
                all_text_parts = []
                total_storage_size = 0
                total_ai_size = 0
                
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
                            all_text_parts.append(page_result['text'])
                            metadata['page_details'].append(page_result['metadata'])
                            
                            # Track sizes
                            total_storage_size += page_result['metadata'].get('image_sizes', {}).get('storage', 0)
                            total_ai_size += page_result['metadata'].get('image_sizes', {}).get('ai', 0)
                            
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
                
                # Save combined text content
                full_text = ''.join(all_text_parts)
                await self._save_text_content(session_id, full_text, storage_service)
                
                # Create and save document index
                document_index = await self._create_document_index(
                    session_id, metadata['page_details'], full_text, storage_service
                )
                
                # Update metadata with processing time and optimization stats
                processing_end = asyncio.get_event_loop().time()
                metadata['processing_time'] = round(processing_end - processing_start, 2)
                metadata['optimization_stats']['total_storage_size_mb'] = round(total_storage_size / 1024, 2)
                metadata['optimization_stats']['total_ai_size_mb'] = round(total_ai_size / 1024, 2)
                metadata['optimization_stats']['average_page_size_mb'] = round((total_storage_size / pages_to_process) / 1024, 2)
                
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
                logger.info(f"   ⏱️ Processing time: {metadata['processing_time']}s")
                logger.info(f"   💾 Pages/second: {pages_to_process/metadata['processing_time']:.2f}")
                logger.info(f"   📊 Storage size: {metadata['optimization_stats']['total_storage_size_mb']:.1f}MB")
                logger.info(f"   📊 AI size: {metadata['optimization_stats']['total_ai_size_mb']:.1f}MB")
                logger.info(f"   📊 Avg page size: {metadata['optimization_stats']['average_page_size_mb']:.1f}MB")
                
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
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
        """Process a single page with optimized image generation"""
        try:
            page_actual = page_num + 1
            
            # Extract text
            page_text = page.get_text()
            
            # Analyze page content
            page_analysis = self._analyze_page_content(page_text, page_actual)
            
            # Extract tables if present
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
            
            # Generate images in parallel with OPTIMIZED settings
            image_tasks = []
            
            # High quality image for storage (with compression)
            image_task = asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_page_image_optimized,
                page, self.image_dpi, 'storage'
            )
            image_tasks.append(('storage', image_task))
            
            # Optimized image for AI
            ai_image_task = asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._generate_page_image_optimized,
                page, self.ai_image_dpi, 'ai'
            )
            image_tasks.append(('ai', ai_image_task))
            
            # Thumbnail for UI (only for first 10 pages)
            if page_actual <= 10:
                thumb_task = asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._generate_page_image_optimized,
                    page, self.thumbnail_dpi, 'thumbnail'
                )
                image_tasks.append(('thumbnail', thumb_task))
            
            # Wait for all images
            image_results = {}
            for img_type, task in image_tasks:
                try:
                    image_results[img_type] = await task
                except Exception as e:
                    logger.error(f"Failed to generate {img_type} image for page {page_actual}: {e}")
            
            # Upload images
            upload_tasks = []
            
            # Upload storage quality image
            if 'storage' in image_results:
                upload_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}.png",
                    data=image_results['storage']
                )
                upload_tasks.append(upload_task)
            
            # Upload AI optimized image
            if 'ai' in image_results:
                upload_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}_ai.png",
                    data=image_results['ai']
                )
                upload_tasks.append(upload_task)
            
            # Upload thumbnail
            if 'thumbnail' in image_results:
                upload_task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_page_{page_actual}_thumb.png",
                    data=image_results['thumbnail']
                )
                upload_tasks.append(upload_task)
            
            # Wait for uploads to complete
            await asyncio.gather(*upload_tasks)
            
            # Prepare page metadata with size tracking
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
                'image_sizes': {
                    'storage': len(image_results.get('storage', b'')) / 1024,  # KB
                    'ai': len(image_results.get('ai', b'')) / 1024,  # KB
                    'thumbnail': len(image_results.get('thumbnail', b'')) / 1024 if 'thumbnail' in image_results else 0
                }
            }
            
            # Log compression results
            if 'storage' in image_results:
                logger.debug(f"Page {page_actual} - Storage: {page_metadata['image_sizes']['storage']:.1f}KB, "
                           f"AI: {page_metadata['image_sizes']['ai']:.1f}KB")
            
            # Format text for storage
            formatted_text = f"\n--- PAGE {page_actual} ---\n"
            if page_analysis.get('sheet_number'):
                formatted_text += f"Sheet: {page_analysis['sheet_number']}\n"
            if page_analysis.get('drawing_type'):
                formatted_text += f"Type: {page_analysis['drawing_type']}\n"
            formatted_text += page_text
            
            return {
                'success': True,
                'page_num': page_actual,
                'text': formatted_text,
                'metadata': page_metadata,
                'tables': tables
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

    def _generate_page_image_optimized(self, page, dpi: int, image_type: str) -> bytes:
        """Generate optimized page image with proper compression"""
        try:
            # Create transformation matrix
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            
            # Generate pixmap
            pix = page.get_pixmap(matrix=matrix, alpha=False)  # No alpha for smaller files
            
            # Convert based on image type
            if image_type == 'thumbnail':
                # Thumbnails as JPEG for smallest size
                img_data = pix.pil_tobytes(format="JPEG", optimize=True, quality=self.jpeg_quality)
            elif image_type == 'ai':
                # AI images need balance of quality and size
                # First get as PIL image for further optimization
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize if too large
                if max(img.size) > self.ai_max_dimension:
                    img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
                
                # Save as optimized PNG
                output = io.BytesIO()
                img.save(output, format='PNG', optimize=True, compress_level=self.png_compression_level)
                img_data = output.getvalue()
            else:
                # Storage images - compressed PNG
                # Use PyMuPDF's built-in PNG compression
                img_data = pix.tobytes(output="png", compress_level=self.png_compression_level)
            
            # Clean up
            pix = None
            
            return img_data
            
        except Exception as e:
            logger.error(f"Image generation error for {image_type}: {e}")
            # Fallback to basic generation
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            pix = None
            return img_data

    async def _optimize_image_for_ai(self, image_bytes: bytes) -> bytes:
        """Further optimize image for AI processing if needed"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._compress_image,
            image_bytes
        )
    
    def _compress_image(self, image_bytes: bytes) -> bytes:
        """Compress image to reduce memory usage"""
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            
            # Resize if too large
            if max(img.size) > self.ai_max_dimension:
                img.thumbnail((self.ai_max_dimension, self.ai_max_dimension), Image.Resampling.LANCZOS)
            
            # Save with optimization
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=self.ai_image_quality, optimize=True)
            
            compressed = output.getvalue()
            
            # Log compression ratio
            ratio = (1 - len(compressed) / len(image_bytes)) * 100
            if ratio > 0:
                logger.debug(f"Image compressed by {ratio:.1f}%")
            
            return compressed
            
        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return image_bytes  # Return original if compression fails

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content to identify drawing type and key information"""
        import re
        
        info = {
            'page_number': page_num,
            'drawing_type': None,
            'title': None,
            'scale': None,
            'sheet_number': None,
            'key_elements': []
        }
        
        text_upper = text.upper()
        
        # Identify drawing type with priority
        drawing_patterns = [
            ('floor_plan', ['FLOOR PLAN', 'LEVEL P', 'LEVEL [0-9]', r'\d+(?:ST|ND|RD|TH)\s*FLOOR']),
            ('foundation', ['FOUNDATION PLAN', 'FOOTING PLAN', 'PILE PLAN']),
            ('framing', ['FRAMING PLAN', 'STRUCTURAL PLAN', 'BEAM LAYOUT']),
            ('electrical', ['ELECTRICAL PLAN', 'POWER PLAN', 'LIGHTING PLAN', 'PANEL SCHEDULE']),
            ('plumbing', ['PLUMBING PLAN', 'PIPING PLAN', 'DRAINAGE PLAN', 'RISER DIAGRAM']),
            ('mechanical', ['MECHANICAL PLAN', 'HVAC PLAN', 'DUCTWORK', 'EQUIPMENT SCHEDULE']),
            ('fire', ['FIRE PROTECTION', 'SPRINKLER PLAN', 'FIRE ALARM']),
            ('detail', ['DETAIL', 'SECTION', 'CONNECTION DETAIL', 'TYPICAL DETAIL']),
            ('schedule', ['SCHEDULE', 'EQUIPMENT LIST', 'FIXTURE SCHEDULE']),
            ('site', ['SITE PLAN', 'PLOT PLAN', 'LANDSCAPE PLAN']),
            ('elevation', ['ELEVATION', 'BUILDING ELEVATION', 'EXTERIOR ELEVATION']),
            ('roof', ['ROOF PLAN', 'ROOF FRAMING', 'ROOF DRAINAGE'])
        ]
        
        for dtype, patterns in drawing_patterns:
            for pattern in patterns:
                if isinstance(pattern, str):
                    if pattern in text_upper:
                        info['drawing_type'] = dtype
                        break
                else:
                    if re.search(pattern, text_upper):
                        info['drawing_type'] = dtype
                        break
            if info['drawing_type']:
                break
        
        # Extract scale
        scale_patterns = [
            r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+)',
            r'SCALE[\s:]+([0-9]+:[0-9]+)',
            r'SCALE[\s:]+(\d+/\d+"\s*=\s*\d+\'-\d+")',
        ]
        for pattern in scale_patterns:
            match = re.search(pattern, text_upper)
            if match:
                info['scale'] = match.group(1)
                break
        
        # Extract sheet number
        sheet_patterns = [
            r'(?:SHEET|DWG)[\s#:]*([A-Z]*[-\s]?[0-9]+\.?[0-9]*)',
            r'([A-Z]{1,2}[-\s]?[0-9]+\.?[0-9]*)\s*(?:SHEET|DWG)',
        ]
        for pattern in sheet_patterns:
            match = re.search(pattern, text_upper)
            if match:
                info['sheet_number'] = match.group(1).strip()
                break
        
        # Identify key elements
        element_keywords = {
            'columns': ['COLUMN', 'COL.', 'C[0-9]+'],
            'beams': ['BEAM', 'BM.', 'B[0-9]+'],
            'doors': ['DOOR', 'DR.', 'D[0-9]+'],
            'windows': ['WINDOW', 'WIN.', 'W[0-9]+'],
            'equipment': ['EQUIPMENT', 'UNIT', 'AHU', 'RTU', 'VAV'],
            'rooms': ['ROOM', 'RM.', 'SPACE'],
            'fixtures': ['FIXTURE', 'WC', 'LAV', 'SINK'],
            'panels': ['PANEL', 'ELECTRICAL PANEL', 'DISTRIBUTION'],
            'sprinklers': ['SPRINKLER', 'SP', 'HEAD'],
            'parking': ['PARKING', 'STALL', 'SPACE', 'GARAGE']
        }
        
        for element, keywords in element_keywords.items():
            for keyword in keywords:
                if isinstance(keyword, str) and keyword in text_upper:
                    info['key_elements'].append(element)
                    break
                elif isinstance(keyword, str) and re.search(keyword, text_upper):
                    info['key_elements'].append(element)
                    break
        
        # Remove duplicates
        info['key_elements'] = list(set(info['key_elements']))
        
        return info

    async def _save_text_content(self, session_id: str, full_text: str, 
                                storage_service: StorageService):
        """Save text content efficiently"""
        # Save full text
        context_blob = f"{session_id}_context.txt"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=context_blob,
            data=full_text.encode('utf-8')
        )
        
        logger.info(f"✅ Saved text content ({len(full_text)} characters)")

    async def _create_document_index(self, session_id: str, page_details: List[Dict],
                                   full_text: str, storage_service: StorageService) -> Dict:
        """Create searchable index for fast information retrieval"""
        index = {
            'document_id': session_id,
            'total_pages': len(page_details),
            'page_index': {},
            'drawing_types': defaultdict(list),
            'sheet_numbers': {},
            'keyword_index': defaultdict(set),
            'element_locations': defaultdict(list)
        }
        
        # Build page index
        for page_detail in page_details:
            page_num = page_detail['page_number']
            
            # Page quick reference
            index['page_index'][page_num] = {
                'has_text': page_detail['has_text'],
                'text_length': page_detail['text_length'],
                'drawing_type': page_detail.get('drawing_type'),
                'sheet_number': page_detail.get('sheet_number'),
                'has_tables': page_detail.get('has_tables', False),
                'key_elements': page_detail.get('key_elements', [])
            }
            
            # Index by drawing type
            if page_detail.get('drawing_type'):
                index['drawing_types'][page_detail['drawing_type']].append(page_num)
            
            # Index by sheet number
            if page_detail.get('sheet_number'):
                index['sheet_numbers'][page_detail['sheet_number']] = page_num
            
            # Index elements
            for element in page_detail.get('key_elements', []):
                index['element_locations'][element].append(page_num)
        
        # Build keyword index for common search terms
        keywords_to_index = [
            'schedule', 'detail', 'section', 'plan', 'elevation',
            'foundation', 'roof', 'floor', 'ceiling', 'wall',
            'column', 'beam', 'slab', 'footing',
            'electrical', 'mechanical', 'plumbing', 'fire',
            'equipment', 'panel', 'fixture'
        ]
        
        # Search for keywords in text (simplified for performance)
        text_lower = full_text.lower()
        for keyword in keywords_to_index:
            if keyword in text_lower:
                # Find pages containing this keyword
                for page_detail in page_details:
                    page_num = page_detail['page_number']
                    # Check if keyword is in this page's key elements
                    if keyword in [e.lower() for e in page_detail.get('key_elements', [])]:
                        index['keyword_index'][keyword].add(page_num)
        
        # Convert sets to lists for JSON serialization
        for keyword in index['keyword_index']:
            index['keyword_index'][keyword] = list(index['keyword_index'][keyword])
        
        # Save index
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_document_index.json",
            data=json.dumps(index, ensure_ascii=False).encode('utf-8')
        )
        
        logger.info(f"✅ Created document index with {len(index['sheet_numbers'])} sheets")
        
        return index

    async def _create_processing_summary(self, session_id: str, metadata: Dict,
                                       storage_service: StorageService):
        """Create a summary for quick document overview"""
        summary = {
            'document_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_pages': metadata['page_count'],
                'processing_time_seconds': metadata['processing_time'],
                'file_size_mb': metadata['file_size_mb'],
                'pages_per_second': round(metadata['page_count'] / metadata['processing_time'], 2)
            },
            'content_summary': metadata['extraction_summary'],
            'drawing_types_found': {},
            'sheets_found': [],
            'optimization_summary': metadata['optimization_stats']
        }
        
        # Summarize drawing types
        drawing_type_counts = defaultdict(int)
        sheets = set()
        
        for page_detail in metadata['page_details']:
            if page_detail.get('drawing_type'):
                drawing_type_counts[page_detail['drawing_type']] += 1
            if page_detail.get('sheet_number'):
                sheets.add(page_detail['sheet_number'])
        
        summary['drawing_types_found'] = dict(drawing_type_counts)
        summary['sheets_found'] = sorted(list(sheets))
        
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
            pass  # Don't fail on error logging

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "PDFService",
            "version": "2.1.0",
            "mode": "optimized_large_document_processing",
            "capabilities": {
                "max_pages": self.max_pages,
                "parallel_processing": True,
                "image_optimization": True,
                "text_indexing": True,
                "table_extraction": True
            },
            "performance": {
                "batch_size": self.batch_size,
                "parallel_workers": self.max_workers,
                "image_dpi": {
                    "storage": self.image_dpi,
                    "ai_processing": self.ai_image_dpi,
                    "thumbnails": self.thumbnail_dpi
                },
                "compression": {
                    "png_level": self.png_compression_level,
                    "jpeg_quality": self.jpeg_quality,
                    "ai_image_quality": self.ai_image_quality
                }
            }
        }

    # Cleanup
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
