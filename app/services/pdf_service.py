# app/services/pdf_service.py - PRODUCTION-GRADE PDF PROCESSING WITH ENHANCED GRID DETECTION AND THREAD SAFETY

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
import weakref

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
    origin_x: int = 0  # Always start from page origin
    origin_y: int = 0  # Always start from page origin
    confidence: float = 0.0
    scale: Optional[str] = None
    page_width: int = 0  # Actual PDF page width
    page_height: int = 0  # Actual PDF page height
    grid_type: str = "generated"  # "embedded", "text_based", or "generated"
    
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


@dataclass
class VisualGridLine:
    """Represents a detected grid line in the blueprint"""
    orientation: str  # 'horizontal' or 'vertical'
    position: float   # x-coordinate for vertical, y-coordinate for horizontal
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    confidence: float
    label: Optional[str] = None


class PDFService:
    """Production-grade PDF processing with enhanced grid detection, proper error handling, memory management, thread safety, and SSE event support"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        self._lock = asyncio.Lock()  # Thread safety for general operations
        self._status_lock = asyncio.Lock()  # Dedicated lock for status updates
        self._status_update_locks: Dict[str, asyncio.Lock] = {}  # Per-document status locks
        
        # Resolution settings - balanced for quality and performance
        self.storage_dpi = settings.PDF_HIGH_RESOLUTION  # 150 DPI for storage
        self.ai_image_dpi = settings.PDF_AI_DPI  # 250 DPI for AI
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
        self.max_sessions_in_memory = 10  # Maximum sessions to keep in memory
        self.session_timeout_seconds = 3600  # 1 hour
        
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
        
        # Visual grid detection settings
        self.min_line_length = 100  # Minimum pixels for a line to be considered a grid line
        self.line_gap_tolerance = 50  # Maximum gap in pixels to consider lines as continuous
        self.grid_line_thickness_range = (0.5, 3)  # Expected line thickness in pixels
        
        # Universal grid settings for documents without grids
        self.default_grid_columns = 10  # Default 10x10 grid for documents without grids
        self.default_grid_rows = 10
        
        # Revolutionary Features - Living Document Enhancements
        self.enable_ai_annotations = True  # Add AI-friendly coordinate markers
        self.enable_quality_scoring = True  # Automatic document quality assessment
        self.enable_page_fingerprinting = True  # Detect duplicate/similar pages
        self.enable_smart_caching = True  # Predictive caching for faster access
        self.enable_collaboration_markers = True  # Real-time collaboration support
        self.enable_anomaly_detection = True  # Detect unusual patterns
        self.enable_performance_tracking = True  # Track processing metrics
        self.enable_accessibility_scoring = True  # WCAG compliance scoring
        self.enable_semantic_analysis = True  # Understand document relationships
        self.enable_progressive_enhancement = True  # Gradual quality improvements
        
        # MEMORY LEAK FIX: Use bounded collections with automatic cleanup
        self.collaboration_sessions = {}  # Track active collaborators
        self.performance_metrics = {}  # Changed from defaultdict to regular dict
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        
        self.quality_thresholds = {
            'text_clarity': 0.8,
            'grid_confidence': 0.7,
            'table_extraction': 0.6,
            'overall_quality': 0.75
        }
        
        # AI Enhancement Settings
        self.ai_coordinate_format = "{{coord:X{x}Y{y}}}"  # Embedded coordinate markers
        self.ai_grid_reference_format = "{{grid:{label}}}"  # Grid reference markers
        self.ai_confidence_threshold = 0.85  # Minimum confidence for AI annotations
        
        logger.info("✅ PDFService initialized (Production Mode with Enhanced Grid Detection)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ DPI: Storage={self.storage_dpi}, AI={self.ai_image_dpi}")
        logger.info(f"   📦 Batch size: {self.batch_size} pages")
        logger.info(f"   🔒 Thread safety: Enhanced with status locks")
        logger.info(f"   💾 Memory management: Active with automatic cleanup")
        logger.info(f"   📡 SSE Events: Supported")
        logger.info(f"   🎯 Visual Grid Detection: {'Enabled' if OPENCV_AVAILABLE else 'Disabled (OpenCV not available)'}")
        logger.info(f"   📐 Universal Grid System: Enabled")
        logger.info(f"   🤖 AI Enhancements: Active")
        logger.info(f"   📊 Quality Scoring: Enabled")
        logger.info(f"   🔍 Anomaly Detection: Active")
        logger.info(f"   👥 Collaboration Ready: Yes")
        logger.info(f"   ♿ Accessibility Scoring: Enabled")

    def _cleanup_old_data(self):
        """Periodic cleanup of old session data to prevent memory leaks"""
        current_time = time.time()
        
        # Skip if cleaned up recently
        if current_time - self._last_cleanup_time < self._cleanup_interval:
            return
        
        try:
            # Clean old collaboration sessions
            sessions_to_remove = []
            for session_id, data in self.collaboration_sessions.items():
                if current_time - data.get('last_activity', 0) > self.session_timeout_seconds:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.collaboration_sessions[session_id]
            
            # Clean old performance metrics - keep only recent sessions
            if len(self.performance_metrics) > self.max_sessions_in_memory:
                # Sort by most recent activity (assuming last entry has latest timestamp)
                sorted_sessions = sorted(
                    self.performance_metrics.items(),
                    key=lambda x: x[1][-1]['processing_time'] if x[1] else 0,
                    reverse=True
                )
                # Keep only the most recent sessions
                self.performance_metrics = dict(sorted_sessions[:self.max_sessions_in_memory])
            
            # Clean orphaned status locks
            if len(self._status_update_locks) > self.max_sessions_in_memory * 2:
                # Keep only locks that are currently locked
                active_locks = {}
                for session_id, lock in self._status_update_locks.items():
                    if lock.locked():
                        active_locks[session_id] = lock
                self._status_update_locks = active_locks
            
            self._last_cleanup_time = current_time
            
            if sessions_to_remove or len(self.performance_metrics) > self.max_sessions_in_memory:
                logger.info(f"🧹 Memory cleanup: Removed {len(sessions_to_remove)} old sessions, "
                          f"{len(self.performance_metrics)} sessions in memory")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

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
            # Don't propagate callback errors to main processing

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService,
                                   event_callback: Optional[Callable] = None,
                                   collaboration_session: Optional[str] = None):
        """
        Process PDF with production-grade error handling, memory management, thread safety, and SSE event emission
        
        Args:
            session_id: Document ID
            pdf_bytes: PDF file content
            storage_service: Storage service instance
            event_callback: Optional async callback for SSE events
            collaboration_session: Optional collaboration session ID for real-time features
        """
        # Perform cleanup at the start of each processing
        self._cleanup_old_data()
        
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session ID")
        
        if not pdf_bytes or not isinstance(pdf_bytes, bytes):
            raise ValueError("Invalid PDF data")
        
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValueError("Not a valid PDF file")
        
        # Revolutionary: Track collaboration session
        if collaboration_session and self.enable_collaboration_markers:
            self.collaboration_sessions[session_id] = {
                'session_id': collaboration_session,
                'started_at': datetime.utcnow().isoformat() + 'Z',
                'participants': 1,
                'last_activity': time.time()
            }
        
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
                    
                    # MEMORY LEAK FIX: Initialize performance metrics as list instead of using defaultdict
                    if self.enable_performance_tracking:
                        self.performance_metrics[session_id] = []
                    
                    # Process in batches
                    all_text_parts = []
                    all_grid_systems = {}
                    pages_processed = 0
                    total_batches = (pages_to_process + self.batch_size - 1) // self.batch_size
                    
                    for batch_num, batch_start in enumerate(range(0, pages_to_process, self.batch_size), 1):
                        batch_end = min(batch_start + self.batch_size, pages_to_process)
                        batch_pages = list(range(batch_start, batch_end))
                        
                        # Revolutionary: Adaptive batch sizing based on document complexity
                        if self.enable_progressive_enhancement and batch_num > 1:
                            # Adjust batch size based on previous performance
                            avg_time = self._get_recent_page_processing_time(session_id)
                            if avg_time > 2.0:  # Slow processing
                                batch_pages = batch_pages[:max(1, len(batch_pages) // 2)]
                                logger.info(f"📊 Adaptive batching: Reduced batch size due to complexity")
                        
                        logger.info(f"📦 Processing batch {batch_num}/{total_batches}: pages {batch_start + 1}-{batch_start + len(batch_pages)}")
                        
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
                        
                        # Update status with thread safety
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
                        
                        # Prevent overload
                        await asyncio.sleep(self.processing_delay)
                        
                        # Progress update
                        progress = (pages_processed / pages_to_process) * 100
                        logger.info(f"📊 Progress: {progress:.1f}% ({pages_processed}/{pages_to_process})")
                    
                    # Save results
                    await self._save_processing_results(
                        session_id, all_text_parts, all_grid_systems, metadata, 
                        processing_start, storage_service, event_callback
                    )
                    
                    # Update final status to 'ready'
                    metadata['status'] = 'ready'
                    metadata['processing_time'] = round(time.time() - processing_start, 2)
                    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
                    await self._update_status(session_id, 'ready', metadata, storage_service)
                    
                    # Emit processing complete event
                    await self._safe_event_callback(event_callback, "processing_complete", {
                        "status": "ready",
                        "total_pages": pages_processed,
                        "processing_time": metadata['processing_time'],
                        "grid_systems_detected": metadata.get('grid_systems_detected', 0)
                    })
                    
                    logger.info(f"✅ Processing complete for {session_id}")
                    logger.info(f"   📝 Pages processed: {pages_processed}")
                    logger.info(f"   ⏱️ Total time: {metadata['processing_time']}s")
                    
                    # Revolutionary: Provide optimization suggestions for future processing
                    if self.enable_progressive_enhancement:
                        suggestions = self._generate_optimization_suggestions(metadata, all_grid_systems)
                        if suggestions:
                            logger.info("💡 Optimization suggestions for similar documents:")
                            for suggestion in suggestions:
                                logger.info(f"   - {suggestion}")
                            metadata['optimization_suggestions'] = suggestions
                    
                finally:
                    doc.close()
                    gc.collect()
                    
                    # MEMORY LEAK FIX: Clean up session-specific data
                    # Clean up status lock
                    if session_id in self._status_update_locks:
                        del self._status_update_locks[session_id]
                    
                    # Clean up old performance metrics if too many sessions
                    if len(self.performance_metrics) > self.max_sessions_in_memory * 2:
                        self._cleanup_old_data()
                    
            except Exception as e:
                logger.error(f"❌ Processing failed: {e}", exc_info=True)
                await self._save_error_state(session_id, str(e), storage_service)
                # Emit error event
                await self._safe_event_callback(event_callback, "processing_error", {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "is_fatal": True
                })
                raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _update_status(self, session_id: str, status: str, metadata: Dict, storage_service: StorageService):
        """Update status file with thread safety to prevent race conditions"""
        # Get document-specific lock
        status_lock = self._get_status_lock(session_id)
        
        async with status_lock:
            try:
                # Add timeout to prevent deadlocks
                async with asyncio.timeout(self.settings.STATUS_UPDATE_LOCK_TIMEOUT):
                    status_data = {
                        'document_id': session_id,
                        'status': status,  # 'processing', 'ready', or 'error'
                        'updated_at': datetime.utcnow().isoformat() + 'Z',
                        'pages_processed': metadata.get('pages_processed', 0),
                        'total_pages': metadata.get('total_pages', 0),
                        'started_at': metadata.get('started_at'),
                        'completed_at': metadata.get('completed_at'),
                        'error': metadata.get('error'),
                        'processing_time': metadata.get('processing_time'),
                        'last_update_timestamp': time.time()  # For debugging race conditions
                    }
                    
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_status.json",
                        data=json.dumps(status_data).encode('utf-8'),
                        content_type="application/json"
                    )
                    
            except asyncio.TimeoutError:
                logger.error(f"Status update timeout for {session_id}")
                # Continue without failing the entire process
            except Exception as e:
                logger.error(f"Failed to update status for {session_id}: {e}")
                # Continue without failing the entire process

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
        
        # Update table counts
        if 'tables_found' in result:
            metadata['extraction_summary']['total_tables_found'] += result['tables_found']
        if 'tables_extracted' in result:
            metadata['extraction_summary']['total_tables_extracted'] += result['tables_extracted']
        
        # Update grid systems count
        if result.get('grid_system'):
            metadata['grid_systems_detected'] = metadata.get('grid_systems_detected', 0) + 1

    async def _process_batch_safe(self, doc: fitz.Document, page_numbers: List[int], 
                                 session_id: str, storage_service: StorageService,
                                 event_callback: Optional[Callable] = None) -> List[Dict]:
        """Process batch with error recovery and event emission"""
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
                
                # Emit page error event
                await self._safe_event_callback(event_callback, "page_error", {
                    "page_number": page_num + 1,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
        
        return results

    async def _process_single_page_safe(self, doc: fitz.Document, page_num: int, 
                                       session_id: str, storage_service: StorageService,
                                       event_callback: Optional[Callable] = None) -> Dict:
        """Process single page with all safety checks and event emission"""
        page_start_time = time.time()  # Revolutionary: Track page processing time
        
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
            tables_found = 0
            tables_extracted = 0
            if self.enable_tables:
                table_result = await self._extract_tables_safe(page)
                tables = table_result['tables']
                tables_found = table_result['found']
                tables_extracted = table_result['extracted']
            
            # Enhanced grid detection with visual analysis
            grid_system = None
            if self.enable_grid_detection:
                grid_system = await self._detect_grid_patterns_enhanced(page, page_text, page_actual)
            
            # Generate images (using JPEG as per the working version)
            await self._generate_and_upload_page_images_safe(
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
            
            # Revolutionary: Add quality scoring
            if self.enable_quality_scoring:
                quality_score = self._calculate_page_quality_score(
                    page_text, tables, grid_system, page_analysis
                )
                page_metadata['quality_score'] = quality_score
                page_metadata['quality_factors'] = {
                    'text_clarity': self._assess_text_clarity(page_text),
                    'grid_quality': grid_system.confidence if grid_system else 0.0,
                    'table_quality': tables_extracted / tables_found if tables_found > 0 else 1.0,
                    'metadata_completeness': self._assess_metadata_completeness(page_analysis)
                }
            
            # Revolutionary: Add page fingerprint for duplicate detection
            if self.enable_page_fingerprinting:
                page_fingerprint = self._generate_page_fingerprint(
                    page_text, grid_system, tables, page.rect
                )
                page_metadata['fingerprint'] = page_fingerprint
            
            # Revolutionary: Add AI coordinate annotations
            if self.enable_ai_annotations and grid_system:
                ai_enhanced_text = self._add_ai_coordinate_annotations(
                    page_text, grid_system, page_analysis
                )
                page_metadata['ai_annotations'] = {
                    'coordinate_markers_added': ai_enhanced_text != page_text,
                    'grid_reference_system': grid_system.grid_type,
                    'confidence': grid_system.confidence
                }
            else:
                ai_enhanced_text = page_text
            
            # Revolutionary: Detect anomalies
            if self.enable_anomaly_detection:
                anomalies = self._detect_page_anomalies(
                    page, page_text, tables, grid_system
                )
                if anomalies:
                    page_metadata['anomalies'] = anomalies
            
            # Revolutionary: Track performance metrics with memory management
            if self.enable_performance_tracking and session_id in self.performance_metrics:
                page_processing_time = time.time() - page_start_time
                self.performance_metrics[session_id].append({
                    'page': page_actual,
                    'processing_time': page_processing_time,
                    'text_length': len(page_text),
                    'tables_found': tables_found,
                    'quality_score': page_metadata.get('quality_score', 0)
                })
                # MEMORY LEAK FIX: Limit metrics per session
                if len(self.performance_metrics[session_id]) > 100:
                    self.performance_metrics[session_id] = self.performance_metrics[session_id][-50:]
            
            # Revolutionary: Accessibility scoring
            if self.enable_accessibility_scoring:
                accessibility_score = self._calculate_accessibility_score(
                    page_text, tables, page_metadata
                )
                page_metadata['accessibility'] = {
                    'score': accessibility_score,
                    'has_alt_text': False,  # Would need OCR analysis
                    'table_headers': self._check_table_headers(tables),
                    'text_contrast': self._estimate_text_contrast(page)
                }
            
            # Format text for context with AI enhancements
            formatted_text = self._format_page_text(page_actual, page_analysis, grid_system, ai_enhanced_text)
            
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
                "grid_confidence": page_metadata['grid_confidence'],
                "grid_type": page_metadata['grid_type']
            })
            
            # Clean up page
            page.clean_contents()
            
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

    async def _detect_grid_patterns_enhanced(self, page: fitz.Page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """
        Universal grid detection that creates an intelligent reference system for ANY document.
        This grid serves as an invisible coordinate system for AI validation.
        """
        try:
            # Get actual page dimensions - critical for all documents
            page_width = page.rect.width
            page_height = page.rect.height
            
            logger.info(f"📐 Analyzing page {page_num}: {page_width}x{page_height}px")
            
            # First, try to detect if document has its own grid system
            has_embedded_grid = False
            visual_grid = None
            
            # Try visual detection if OpenCV is available
            if OPENCV_AVAILABLE:
                visual_grid = await self._detect_visual_grid(page, page_num)
                if visual_grid and visual_grid.confidence > 0.7:
                    has_embedded_grid = True
                    visual_grid.grid_type = "embedded"
                    logger.info(f"✅ High confidence embedded grid detected on page {page_num}")
                    return visual_grid
            else:
                logger.info(f"ℹ️ Visual grid detection not available, using text-based detection for page {page_num}")
            
            # Try text-based detection
            text_grid = self._detect_grid_patterns(page, page_text, page_num)
            
            if text_grid and (len(text_grid.x_labels) > 0 or len(text_grid.y_labels) > 0):
                # Document has text-based grid references
                text_grid.grid_type = "text_based"
                
                # If we have both visual and text grids, merge them
                if visual_grid:
                    # Prefer visual grid coordinates but use text labels if better
                    if len(text_grid.x_labels) > len(visual_grid.x_labels):
                        visual_grid.x_labels = text_grid.x_labels
                    if len(text_grid.y_labels) > len(visual_grid.y_labels):
                        visual_grid.y_labels = text_grid.y_labels
                    
                    visual_grid.scale = text_grid.scale or visual_grid.scale
                    visual_grid.grid_type = "embedded"  # Visual takes precedence
                    return visual_grid
                
                return text_grid
            
            # No grid detected - create universal reference grid
            # This ensures EVERY document has a coordinate system
            logger.info(f"📏 Creating universal reference grid for page {page_num}")
            
            universal_grid = self._create_universal_grid(page_width, page_height, page_num)
            return universal_grid
            
        except Exception as e:
            logger.error(f"Enhanced grid detection failed: {e}")
            # Fallback to universal grid
            return self._create_universal_grid(page.rect.width, page.rect.height, page_num)

    def _create_universal_grid(self, page_width: float, page_height: float, page_num: int) -> GridSystem:
        """
        Create a universal reference grid for documents without embedded grids.
        This provides a consistent coordinate system for AI reference.
        """
        grid = GridSystem(
            page_number=page_num,
            page_width=int(page_width),
            page_height=int(page_height),
            grid_type="generated",
            confidence=1.0  # High confidence since we're creating it
        )
        
        # Determine optimal grid size based on page dimensions
        # For letter/A4 size, 10x10 is good. Scale for other sizes.
        aspect_ratio = page_width / page_height
        
        if aspect_ratio > 1.3:  # Landscape
            num_cols = max(12, min(20, int(page_width / 50)))
            num_rows = max(8, min(15, int(page_height / 50)))
        else:  # Portrait or square
            num_cols = max(8, min(15, int(page_width / 70)))
            num_rows = max(10, min(20, int(page_height / 50)))
        
        # Create evenly spaced grid
        col_spacing = page_width / (num_cols + 1)
        row_spacing = page_height / (num_rows + 1)
        
        # Generate column labels (A, B, C, ..., Z, AA, AB, ...)
        for i in range(num_cols):
            if i < 26:
                label = chr(65 + i)  # A-Z
            else:
                # AA, AB, AC, etc.
                label = chr(65 + (i // 26) - 1) + chr(65 + (i % 26))
            
            x_pos = (i + 1) * col_spacing
            grid.x_labels.append(label)
            grid.x_coordinates[label] = int(x_pos)
            grid.x_lines.append(int(x_pos))
        
        # Generate row labels (1, 2, 3, ...)
        for i in range(num_rows):
            label = str(i + 1)
            y_pos = (i + 1) * row_spacing
            grid.y_labels.append(label)
            grid.y_coordinates[label] = int(y_pos)
            grid.y_lines.append(int(y_pos))
        
        grid.cell_width = int(col_spacing)
        grid.cell_height = int(row_spacing)
        grid.origin_x = 0
        grid.origin_y = 0
        
        logger.info(f"📐 Created universal {num_cols}x{num_rows} grid for page {page_num}")
        
        return grid

    async def _detect_visual_grid(self, page: fitz.Page, page_num: int) -> Optional[GridSystem]:
        """
        Detect actual grid lines from the visual content of the page
        """
        if not OPENCV_AVAILABLE:
            return None
        
        pix = None
        img_data = None
        
        try:
            # Extract page as image for analysis
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better line detection
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            
            # Detect lines using Hough transform
            horizontal_lines = self._detect_horizontal_lines(gray)
            vertical_lines = self._detect_vertical_lines(gray)
            
            # MEMORY LEAK FIX: Properly clean up
            gray = None
            
            if not horizontal_lines and not vertical_lines:
                logger.info(f"No visual grid lines detected on page {page_num}")
                return None
            
            # Extract text for label mapping
            page_text = page.get_text()
            
            # Map grid labels to detected lines
            grid_system = self._create_grid_system_from_lines(
                horizontal_lines, 
                vertical_lines, 
                page_text, 
                page_num,
                page.rect.width,
                page.rect.height
            )
            
            return grid_system
            
        except Exception as e:
            logger.error(f"Visual grid detection failed for page {page_num}: {e}")
            return None
        finally:
            # MEMORY LEAK FIX: Properly clean up pixmap and numpy array
            if pix:
                pix.clear_with()  # Proper cleanup method
                pix = None
            if img_data is not None:
                del img_data
            gc.collect()

    def _detect_horizontal_lines(self, image: np.ndarray) -> List[VisualGridLine]:
        """Detect horizontal grid lines"""
        lines = []
        
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Probabilistic Hough Transform
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
        
        # Filter and group horizontal lines
        horizontal_positions = {}
        
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is horizontal (within 5 degrees)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 5 or angle > 175:
                # Group lines that are close together
                y_avg = (y1 + y2) / 2
                
                # Find nearby line group
                found_group = False
                for y_pos in list(horizontal_positions.keys()):
                    if abs(y_pos - y_avg) < 10:  # Within 10 pixels
                        horizontal_positions[y_pos].append((x1, y1, x2, y2))
                        found_group = True
                        break
                
                if not found_group:
                    horizontal_positions[y_avg] = [(x1, y1, x2, y2)]
        
        # Create VisualGridLine objects for each group
        for y_pos, line_segments in horizontal_positions.items():
            # Find the extent of all segments
            min_x = min(min(seg[0], seg[2]) for seg in line_segments)
            max_x = max(max(seg[0], seg[2]) for seg in line_segments)
            
            # Calculate total length vs gaps to determine confidence
            total_length = sum(abs(seg[2] - seg[0]) for seg in line_segments)
            span_length = max_x - min_x
            confidence = total_length / span_length if span_length > 0 else 0
            
            if confidence > 0.5:  # At least 50% coverage
                lines.append(VisualGridLine(
                    orientation='horizontal',
                    position=y_pos / 2,  # Adjust for 2x zoom
                    start_point=(min_x / 2, y_pos / 2),
                    end_point=(max_x / 2, y_pos / 2),
                    confidence=confidence
                ))
        
        return sorted(lines, key=lambda l: l.position)

    def _detect_vertical_lines(self, image: np.ndarray) -> List[VisualGridLine]:
        """Detect vertical grid lines"""
        lines = []
        
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Probabilistic Hough Transform
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
        
        # Filter and group vertical lines
        vertical_positions = {}
        
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is vertical (within 5 degrees of 90)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 85 < angle < 95:
                # Group lines that are close together
                x_avg = (x1 + x2) / 2
                
                # Find nearby line group
                found_group = False
                for x_pos in list(vertical_positions.keys()):
                    if abs(x_pos - x_avg) < 10:  # Within 10 pixels
                        vertical_positions[x_pos].append((x1, y1, x2, y2))
                        found_group = True
                        break
                
                if not found_group:
                    vertical_positions[x_avg] = [(x1, y1, x2, y2)]
        
        # Create VisualGridLine objects for each group
        for x_pos, line_segments in vertical_positions.items():
            # Find the extent of all segments
            min_y = min(min(seg[1], seg[3]) for seg in line_segments)
            max_y = max(max(seg[1], seg[3]) for seg in line_segments)
            
            # Calculate total length vs gaps to determine confidence
            total_length = sum(abs(seg[3] - seg[1]) for seg in line_segments)
            span_length = max_y - min_y
            confidence = total_length / span_length if span_length > 0 else 0
            
            if confidence > 0.5:  # At least 50% coverage
                lines.append(VisualGridLine(
                    orientation='vertical',
                    position=x_pos / 2,  # Adjust for 2x zoom
                    start_point=(x_pos / 2, min_y / 2),
                    end_point=(x_pos / 2, max_y / 2),
                    confidence=confidence
                ))
        
        return sorted(lines, key=lambda l: l.position)

    def _create_grid_system_from_lines(
        self, 
        horizontal_lines: List[VisualGridLine],
        vertical_lines: List[VisualGridLine],
        page_text: str,
        page_num: int,
        page_width: float,
        page_height: float
    ) -> GridSystem:
        """Create GridSystem from detected lines and text labels"""
        
        # Extract grid labels from text
        x_labels, y_labels = self._extract_grid_labels_from_text(page_text)
        
        # Create grid system with page dimensions
        grid = GridSystem(
            page_number=page_num,
            page_width=int(page_width),
            page_height=int(page_height),
            grid_type="embedded"
        )
        
        # Always start from page origin
        grid.origin_x = 0
        grid.origin_y = 0
        
        # Map vertical lines to X coordinates/labels
        if vertical_lines:
            # Try to match labels to lines
            if x_labels and len(x_labels) == len(vertical_lines):
                # Direct mapping
                for i, (label, line) in enumerate(zip(sorted(x_labels), vertical_lines)):
                    grid.x_labels.append(label)
                    grid.x_coordinates[label] = int(line.position)
                    grid.x_lines.append(int(line.position))
            else:
                # Use line positions directly
                for i, line in enumerate(vertical_lines):
                    label = chr(65 + i) if i < 26 else f"X{i}"  # A, B, C... then X26, X27...
                    grid.x_labels.append(label)
                    grid.x_coordinates[label] = int(line.position)
                    grid.x_lines.append(int(line.position))
            
            # Calculate average cell width
            if len(vertical_lines) > 1:
                spacings = [vertical_lines[i+1].position - vertical_lines[i].position 
                           for i in range(len(vertical_lines)-1)]
                grid.cell_width = int(sum(spacings) / len(spacings))
        
        # Map horizontal lines to Y coordinates/labels  
        if horizontal_lines:
            # Try to match labels to lines
            if y_labels and len(y_labels) == len(horizontal_lines):
                # Direct mapping
                for i, (label, line) in enumerate(zip(sorted(y_labels, key=lambda x: int(x) if x.isdigit() else 0), horizontal_lines)):
                    grid.y_labels.append(label)
                    grid.y_coordinates[label] = int(line.position)
                    grid.y_lines.append(int(line.position))
            else:
                # Use line positions directly
                for i, line in enumerate(horizontal_lines):
                    label = str(i + 1)
                    grid.y_labels.append(label)
                    grid.y_coordinates[label] = int(line.position)
                    grid.y_lines.append(int(line.position))
            
            # Calculate average cell height
            if len(horizontal_lines) > 1:
                spacings = [horizontal_lines[i+1].position - horizontal_lines[i].position 
                           for i in range(len(horizontal_lines)-1)]
                grid.cell_height = int(sum(spacings) / len(spacings))
        
        # Calculate confidence based on line detection quality
        avg_confidence = 0
        if vertical_lines:
            avg_confidence += sum(l.confidence for l in vertical_lines) / len(vertical_lines) * 0.5
        if horizontal_lines:
            avg_confidence += sum(l.confidence for l in horizontal_lines) / len(horizontal_lines) * 0.5
        
        grid.confidence = avg_confidence
        
        # Detect scale if present
        grid.scale = self._extract_scale_from_text(page_text)
        
        logger.info(f"Created embedded grid: {len(grid.x_labels)}x{len(grid.y_labels)} with confidence {grid.confidence:.2f}")
        
        return grid

    def _extract_grid_labels_from_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract grid labels from page text"""
        x_labels = set()
        y_labels = set()
        
        # Search for patterns
        search_text = text[:10000] if len(text) > 10000 else text
        
        for pattern_name, pattern in self.grid_patterns.items():
            for match in pattern.finditer(search_text):
                if pattern_name in ['column', 'column_line']:
                    x_labels.add(match.group(1).upper())
                elif pattern_name == 'row':
                    y_labels.add(match.group(1))
                elif pattern_name == 'grid_ref' or pattern_name == 'coordinate':
                    x_labels.add(match.group(1).upper())
                    y_labels.add(match.group(2))
                elif pattern_name in ['grid_line', 'axis']:
                    ref = match.group(1)
                    if ref.isalpha():
                        x_labels.add(ref.upper())
                    elif ref.isdigit():
                        y_labels.add(ref)
        
        return sorted(list(x_labels)), sorted(list(y_labels), key=lambda x: int(x) if x.isdigit() else 0)

    def _extract_scale_from_text(self, text: str) -> Optional[str]:
        """Extract scale information from text"""
        scale_patterns = [
            re.compile(r'SCALE[\s:]+([0-9/]+"\s*=\s*[0-9\'-]+)', re.IGNORECASE),
            re.compile(r'SCALE[\s:]+([0-9]+:[0-9]+)', re.IGNORECASE),
            re.compile(r'([0-9/]+"\s*=\s*[0-9\'-]+)', re.IGNORECASE),
        ]
        
        for pattern in scale_patterns:
            match = pattern.search(text[:5000])  # Search in first 5000 chars
            if match:
                return match.group(1)
        
        return None

    def _detect_grid_patterns(self, page: fitz.Page, page_text: str, page_num: int) -> Optional[GridSystem]:
        """Original text-based grid detection as fallback"""
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
            
            # Get actual page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Create grid system with proper dimensions
            grid = GridSystem(
                page_number=page_num,
                x_labels=sorted(list(x_refs))[:20],  # Limit to 20 columns
                y_labels=sorted(list(y_refs), key=lambda x: int(x) if x.isdigit() else 0)[:30],  # Limit to 30 rows
                confidence=0.5 if x_refs and y_refs else 0.3,
                page_width=int(page_width),
                page_height=int(page_height),
                grid_type="text_based"
            )
            
            # Always start from page origin
            grid.origin_x = 0
            grid.origin_y = 0
            
            # Distribute grid coordinates evenly across the page
            if grid.x_labels:
                spacing = page_width / (len(grid.x_labels) + 1)
                for i, label in enumerate(grid.x_labels):
                    x_pos = (i + 1) * spacing
                    grid.x_coordinates[label] = int(x_pos)
                    grid.x_lines.append(int(x_pos))
                grid.cell_width = int(spacing)
            
            if grid.y_labels:
                spacing = page_height / (len(grid.y_labels) + 1)
                for i, label in enumerate(grid.y_labels):
                    y_pos = (i + 1) * spacing
                    grid.y_coordinates[label] = int(y_pos)
                    grid.y_lines.append(int(y_pos))
                grid.cell_height = int(spacing)
            
            logger.info(f"🎯 Text-based grid detected on page {page_num}: {len(grid.x_labels)}x{len(grid.y_labels)}")
            
            return grid
            
        except Exception as e:
            logger.error(f"Grid detection failed: {e}")
            return None

    async def _extract_tables_safe(self, page: fitz.Page) -> Dict[str, Any]:
        """Safely extract ALL tables from page with robust error handling - FIXED VERSION"""
        tables = []
        tables_found = 0
        tables_extracted = 0
        tables_failed = 0
        
        try:
            # FIXED: Don't call len() on TableFinder directly
            page_tables = page.find_tables()
            
            # Convert to list to get count - this is the fix!
            if page_tables:
                # Try to convert to list to count tables
                try:
                    table_list = list(page_tables)
                    tables_found = len(table_list)
                except Exception as e:
                    # If we can't convert to list, try to iterate and count
                    logger.warning(f"Could not convert TableFinder to list: {e}")
                    table_list = []
                    try:
                        for table in page_tables:
                            table_list.append(table)
                        tables_found = len(table_list)
                    except Exception as e2:
                        logger.warning(f"Could not iterate TableFinder: {e2}")
                        return {
                            'tables': tables,
                            'found': 0,
                            'extracted': 0
                        }
                
                if tables_found > 0:
                    logger.info(f"📊 Found {tables_found} tables on page")
                
                for i, table in enumerate(table_list):  # Use the list, not the TableFinder
                    try:
                        # Try to extract table data
                        table_data = None
                        bbox_data = None
                        
                        # Safely extract table content with specific error handling
                        try:
                            table_data = table.extract()
                            
                            # Validate extracted data
                            if table_data and len(table_data) > 0:
                                # Count non-empty cells
                                non_empty_cells = sum(
                                    1 for row in table_data 
                                    for cell in row 
                                    if cell and str(cell).strip()
                                )
                                
                                # Skip tables that are mostly empty
                                total_cells = sum(len(row) for row in table_data)
                                if total_cells > 0 and non_empty_cells / total_cells < 0.1:
                                    logger.debug(f"Skipping mostly empty table {i+1}")
                                    continue
                                    
                        except TypeError as e:
                            if "slice" in str(e) and "int" in str(e):
                                logger.warning(f"Complex table structure in table {i+1}/{tables_found}, attempting workaround...")
                                # Try to get basic table info without full extraction
                                try:
                                    # Get table boundaries at least
                                    if hasattr(table, 'bbox'):
                                        bbox_data = list(table.bbox)
                                        table_data = [["[Complex table - see PDF image]"]]
                                        logger.info(f"Captured table {i+1} location, full extraction failed")
                                except:
                                    logger.warning(f"Could not extract table {i+1} at all")
                                    tables_failed += 1
                                    continue
                            else:
                                raise
                        
                        # Safely get bbox if available
                        try:
                            if hasattr(table, 'bbox') and bbox_data is None:
                                bbox_data = list(table.bbox)
                        except Exception:
                            bbox_data = None
                        
                        if table_data:
                            # Clean up table data - remove excessive empty rows/columns
                            cleaned_data = self._clean_table_data(table_data)
                            
                            if cleaned_data:  # Only add if table has content after cleaning
                                tables.append({
                                    'index': i,
                                    'data': cleaned_data,
                                    'bbox': bbox_data,
                                    'row_count': len(cleaned_data),
                                    'col_count': max(len(row) for row in cleaned_data) if cleaned_data else 0,
                                    'extraction_note': 'partial' if "[Complex table" in str(cleaned_data) else 'complete'
                                })
                                tables_extracted += 1
                                
                                # Log large tables
                                if len(cleaned_data) > 20:
                                    logger.info(f"📋 Extracted large table {i+1}: {len(cleaned_data)} rows")
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract table {i+1}/{tables_found}: {str(e)[:100]}")
                        tables_failed += 1
                        # Continue to next table instead of failing completely
                        continue
                
                # Summary logging
                if tables_found > 0:
                    logger.info(f"📊 Table extraction complete: {tables_extracted}/{tables_found} extracted, {tables_failed} failed")
                        
        except Exception as e:
            # Check if it's the specific type comparison error
            if "object of type 'TableFinder' has no len()" in str(e):
                logger.warning(f"Table detection failed: TableFinder doesn't support len() - using workaround")
                # Try alternative approach
                try:
                    page_tables = page.find_tables()
                    if page_tables:
                        # Don't use len(), just iterate
                        for i, table in enumerate(page_tables):
                            try:
                                table_data = table.extract()
                                if table_data:
                                    tables_found += 1
                                    cleaned_data = self._clean_table_data(table_data)
                                    if cleaned_data:
                                        tables.append({
                                            'index': i,
                                            'data': cleaned_data,
                                            'bbox': list(table.bbox) if hasattr(table, 'bbox') else None,
                                            'row_count': len(cleaned_data),
                                            'col_count': max(len(row) for row in cleaned_data) if cleaned_data else 0,
                                            'extraction_note': 'complete'
                                        })
                                        tables_extracted += 1
                            except Exception as table_error:
                                logger.warning(f"Failed to extract table {i+1}: {str(table_error)[:100]}")
                                continue
                except Exception as e2:
                    logger.warning(f"Alternative table extraction also failed: {str(e2)[:100]}")
            else:
                logger.warning(f"Table extraction failed: {str(e)[:200]}")
        
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
            # Convert all cells to strings and strip whitespace
            cleaned_row = []
            for cell in row:
                cell_str = str(cell) if cell is not None else ""
                cell_str = cell_str.strip()
                # Replace multiple spaces with single space
                cell_str = " ".join(cell_str.split())
                cleaned_row.append(cell_str)
            
            # Only include rows that have at least one non-empty cell
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        # Remove columns that are completely empty
        if cleaned:
            # Find columns with any content
            col_count = max(len(row) for row in cleaned)
            non_empty_cols = []
            
            for col_idx in range(col_count):
                col_has_content = False
                for row in cleaned:
                    if col_idx < len(row) and row[col_idx]:
                        col_has_content = True
                        break
                if col_has_content:
                    non_empty_cols.append(col_idx)
            
            # Keep only non-empty columns
            if non_empty_cols:
                filtered_cleaned = []
                for row in cleaned:
                    filtered_row = [row[idx] if idx < len(row) else "" for idx in non_empty_cols]
                    filtered_cleaned.append(filtered_row)
                cleaned = filtered_cleaned
        
        return cleaned

    def _format_page_text(self, page_num: int, page_analysis: Dict[str, Any], 
                         grid_system: Optional[GridSystem], page_text: str) -> str:
        """Format page text with metadata"""
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

    async def _generate_and_upload_page_images_safe(self, page: fitz.Page, page_num: int, 
                                                   session_id: str, storage_service: StorageService,
                                                   event_callback: Optional[Callable] = None):
        """Generate and upload page images with proper error handling and event emission"""
        try:
            # High quality JPEG for viewing
            await self._generate_jpeg_image(page, page_num, session_id, storage_service, 
                                          dpi=self.storage_dpi, quality=90, suffix="", 
                                          resource_type="full_image", event_callback=event_callback)
            
            # AI optimized JPEG
            await self._generate_jpeg_image(page, page_num, session_id, storage_service,
                                          dpi=self.ai_image_dpi, quality=85, suffix="_ai",
                                          resource_type="ai_image", event_callback=event_callback)
            
            # Thumbnail JPEG for all pages
            await self._generate_jpeg_image(page, page_num, session_id, storage_service,
                                        dpi=self.thumbnail_dpi, quality=70, suffix="_thumb",
                                        resource_type="thumbnail", event_callback=event_callback)
            
            # Force cleanup
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to generate images for page {page_num}: {e}")
            # Don't fail the entire page processing for image generation errors
            # Emit image generation error event
            await self._safe_event_callback(event_callback, "image_error", {
                "page_number": page_num,
                "error": str(e),
                "error_type": type(e).__name__
            })

    async def _generate_jpeg_image(self, page: fitz.Page, page_num: int, session_id: str,
                                  storage_service: StorageService, dpi: int, quality: int, suffix: str,
                                  resource_type: str, event_callback: Optional[Callable] = None):
        """Generate and upload JPEG image with event emission and proper cleanup"""
        pix = None
        img = None
        output = None
        
        try:
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Resize if needed for AI images
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
            
            # Emit resource ready event
            await self._safe_event_callback(event_callback, "resource_ready", {
                "resource_type": resource_type,
                "resource_id": f"page_{page_num}_{resource_type}",
                "page_number": page_num,
                "metadata": {
                    "dpi": dpi,
                    "quality": quality,
                    "size": len(output.getvalue())
                }
            })
            
        finally:
            # MEMORY LEAK FIX: Proper cleanup
            if pix:
                pix.clear_with()  # Proper cleanup method
                pix = None
            if img:
                img.close()
                img = None
            if output:
                output.close()
                output = None
            gc.collect()

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
                                     processing_start: float, storage_service: StorageService,
                                     event_callback: Optional[Callable] = None):
        """Save all processing results with proper error handling and event emission"""
        try:
            # Save text content
            full_text = '\n'.join(all_text_parts)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_context.txt",
                data=full_text.encode('utf-8'),
                content_type="text/plain"
            )
            
            # Emit context ready event
            await self._safe_event_callback(event_callback, "resource_ready", {
                "resource_type": "context_text",
                "resource_id": "context",
                "available": True,
                "metadata": {
                    "text_length": len(full_text),
                    "has_content": bool(full_text.strip())
                }
            })
            
            # Save grid systems if detected
            if all_grid_systems:
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{session_id}_grid_systems.json",
                    data=json.dumps(all_grid_systems).encode('utf-8'),
                    content_type="application/json"
                )
                
                # Emit grid systems ready event
                await self._safe_event_callback(event_callback, "resource_ready", {
                    "resource_type": "grid_systems",
                    "resource_id": "grid_systems",
                    "metadata": {
                        "grid_count": len(all_grid_systems),
                        "pages_with_grids": list(all_grid_systems.keys())
                    }
                })
            
            # Revolutionary: Detect semantic relationships between pages
            if self.enable_semantic_analysis and metadata.get('page_details'):
                page_relationships = self._analyze_page_relationships(metadata['page_details'])
                if page_relationships:
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{session_id}_page_relationships.json",
                        data=json.dumps(page_relationships).encode('utf-8'),
                        content_type="application/json"
                    )
                    metadata['has_page_relationships'] = True
                    metadata['relationship_summary'] = {
                        'total_relationships': len(page_relationships.get('relationships', [])),
                        'page_groups': len(page_relationships.get('groups', [])),
                        'document_flow': page_relationships.get('flow_type', 'linear')
                    }
            
            # Create document index
            await self._create_document_index(
                session_id, metadata['page_details'], full_text, storage_service
            )
            
            # Update and save metadata
            processing_end = time.time()
            metadata['processing_time'] = round(processing_end - processing_start, 2)
            metadata['grid_systems_detected'] = len(all_grid_systems)
            metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
            
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_metadata.json",
                data=json.dumps(metadata, ensure_ascii=False).encode('utf-8'),
                content_type="application/json"
            )
            
            # Emit metadata ready event
            await self._safe_event_callback(event_callback, "resource_ready", {
                "resource_type": "metadata",
                "resource_id": "metadata",
                "metadata": {
                    "page_count": metadata['page_count'],
                    "processing_time": metadata['processing_time']
                }
            })
        
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
                'grid_pages': [],
                'grid_confidence': {},
                'grid_types': {},
                'table_summary': {
                    'total_tables_found': 0,
                    'total_tables_extracted': 0,
                    'pages_with_tables': []
                }
            }
            
            for page_detail in page_details:
                page_num = page_detail['page_number']
                
                index['page_index'][page_num] = {
                    'has_text': page_detail['has_text'],
                    'drawing_type': page_detail.get('drawing_type'),
                    'sheet_number': page_detail.get('sheet_number'),
                    'has_grid': page_detail.get('has_grid', False),
                    'grid_confidence': page_detail.get('grid_confidence', 0.0),
                    'grid_type': page_detail.get('grid_type'),
                    'has_tables': page_detail.get('has_tables', False),
                    'table_count': page_detail.get('table_count', 0)
                }
                
                if page_detail.get('drawing_type'):
                    index['drawing_types'][page_detail['drawing_type']].append(page_num)
                
                if page_detail.get('sheet_number'):
                    index['sheet_numbers'][page_detail['sheet_number']] = page_num
                
                if page_detail.get('has_grid'):
                    index['grid_pages'].append(page_num)
                    index['grid_confidence'][page_num] = page_detail.get('grid_confidence', 0.0)
                    index['grid_types'][page_num] = page_detail.get('grid_type', 'unknown')
                
                # Track table information
                if page_detail.get('has_tables'):
                    index['table_summary']['pages_with_tables'].append(page_num)
                
                index['table_summary']['total_tables_found'] += page_detail.get('tables_found', 0)
                index['table_summary']['total_tables_extracted'] += page_detail.get('tables_extracted', 0)
            
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
        """Save error information for debugging with thread safety"""
        error_info = {
            'document_id': session_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'error': str(error),
            'status': 'error',
            'error_type': 'processing_failed'
        }
        
        try:
            # Update status file with thread safety
            await self._update_status(session_id, 'error', error_info, storage_service)
            
            # Also save detailed error info
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{session_id}_error.json",
                data=json.dumps(error_info).encode('utf-8'),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to save error state: {e}")

    def _calculate_page_quality_score(self, text: str, tables: List[Dict], 
                                     grid_system: Optional[GridSystem], 
                                     page_analysis: Dict[str, Any]) -> float:
        """Revolutionary: Calculate comprehensive page quality score"""
        scores = []
        
        # Text quality (0-1)
        if text.strip():
            text_score = min(len(text.strip()) / 1000, 1.0)  # More text = better
            scores.append(text_score * 0.3)  # 30% weight
        else:
            scores.append(0)
        
        # Grid quality (0-1)
        if grid_system:
            scores.append(grid_system.confidence * 0.3)  # 30% weight
        else:
            scores.append(0.3)  # Universal grid gets base score
        
        # Table extraction quality (0-1)
        if tables:
            table_score = sum(1 for t in tables if t.get('extraction_note') == 'complete') / len(tables)
            scores.append(table_score * 0.2)  # 20% weight
        else:
            scores.append(0.2)  # No tables = neutral score
        
        # Metadata completeness (0-1)
        metadata_score = sum([
            0.25 if page_analysis.get('drawing_type') else 0,
            0.25 if page_analysis.get('sheet_number') else 0,
            0.25 if page_analysis.get('scale') else 0,
            0.25 if page_analysis.get('key_elements') else 0
        ])
        scores.append(metadata_score * 0.2)  # 20% weight
        
        return round(sum(scores), 3)

    def _assess_text_clarity(self, text: str) -> float:
        """Assess text extraction clarity"""
        if not text:
            return 0.0
        
        # Simple heuristics for text quality
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Check for garbage characters
        readable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
        clarity = readable_chars / total_chars
        
        # Check for reasonable word lengths
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if 3 <= avg_word_length <= 10:
                clarity *= 1.1  # Bonus for reasonable word lengths
        
        return min(clarity, 1.0)

    def _assess_metadata_completeness(self, page_analysis: Dict[str, Any]) -> float:
        """Assess how complete the extracted metadata is"""
        important_fields = ['drawing_type', 'sheet_number', 'scale', 'key_elements']
        found = sum(1 for field in important_fields if page_analysis.get(field))
        return found / len(important_fields)

    def _generate_page_fingerprint(self, text: str, grid_system: Optional[GridSystem],
                                  tables: List[Dict], page_rect: fitz.Rect) -> str:
        """Generate unique fingerprint for duplicate detection"""
        import hashlib
        
        components = []
        
        # Page dimensions
        components.append(f"dim:{int(page_rect.width)}x{int(page_rect.height)}")
        
        # Text signature (first 500 chars)
        if text:
            text_sig = hashlib.md5(text[:500].encode()).hexdigest()[:8]
            components.append(f"text:{text_sig}")
        
        # Grid signature
        if grid_system:
            grid_sig = f"grid:{len(grid_system.x_labels)}x{len(grid_system.y_labels)}-{grid_system.grid_type}"
            components.append(grid_sig)
        
        # Table signature
        if tables:
            table_sig = f"tables:{len(tables)}-{sum(t['row_count'] for t in tables)}"
            components.append(table_sig)
        
        return "|".join(components)

    def _add_ai_coordinate_annotations(self, text: str, grid_system: GridSystem,
                                     page_analysis: Dict[str, Any]) -> str:
        """Revolutionary: Add AI-friendly coordinate annotations to text"""
        if not self.enable_ai_annotations or not grid_system:
            return text
        
        # Only annotate if we have high confidence
        if grid_system.confidence < self.ai_confidence_threshold:
            return text
        
        # Add grid reference header
        ai_header = f"\n[AI-GRID-REFERENCE: Type={grid_system.grid_type}, "
        ai_header += f"Size={len(grid_system.x_labels)}x{len(grid_system.y_labels)}, "
        ai_header += f"Origin=({grid_system.origin_x},{grid_system.origin_y}), "
        ai_header += f"CellSize={grid_system.cell_width}x{grid_system.cell_height}]\n"
        
        # Add coordinate mapping
        coord_map = "\n[AI-COORDINATES: "
        coord_map += f"X={','.join(grid_system.x_labels[:10])}"
        if len(grid_system.x_labels) > 10:
            coord_map += "..."
        coord_map += f" Y={','.join(grid_system.y_labels[:10])}"
        if len(grid_system.y_labels) > 10:
            coord_map += "..."
        coord_map += "]\n"
        
        return ai_header + coord_map + text

    def _detect_page_anomalies(self, page: fitz.Page, text: str, 
                              tables: List[Dict], grid_system: Optional[GridSystem]) -> List[Dict]:
        """Revolutionary: Detect unusual patterns or potential issues"""
        anomalies = []
        
        # Check for rotated or skewed content
        if hasattr(page, 'rotation') and page.rotation != 0:
            anomalies.append({
                'type': 'rotation',
                'severity': 'medium',
                'details': f'Page rotated {page.rotation} degrees'
            })
        
        # Check for unusual page size
        width, height = page.rect.width, page.rect.height
        standard_sizes = [(612, 792), (792, 612)]  # Letter portrait/landscape
        if not any(abs(width - w) < 50 and abs(height - h) < 50 for w, h in standard_sizes):
            anomalies.append({
                'type': 'unusual_size',
                'severity': 'low',
                'details': f'Non-standard page size: {int(width)}x{int(height)}'
            })
        
        # Check for mostly empty pages with grids
        if grid_system and len(text.strip()) < 100:
            anomalies.append({
                'type': 'empty_grid_page',
                'severity': 'low',
                'details': 'Page has grid but minimal text'
            })
        
        # Check for table extraction issues
        if tables:
            failed_tables = sum(1 for t in tables if t.get('extraction_note') != 'complete')
            if failed_tables > 0:
                anomalies.append({
                    'type': 'table_extraction_issues',
                    'severity': 'medium',
                    'details': f'{failed_tables} tables partially extracted'
                })
        
        return anomalies

    def _calculate_accessibility_score(self, text: str, tables: List[Dict],
                                     page_metadata: Dict[str, Any]) -> float:
        """Revolutionary: Calculate WCAG-inspired accessibility score"""
        scores = []
        
        # Text presence (required for screen readers)
        if text.strip():
            scores.append(0.4)  # 40% weight
        else:
            scores.append(0)
        
        # Table structure (headers needed for accessibility)
        if tables:
            tables_with_structure = sum(1 for t in tables if t.get('row_count', 0) > 1)
            table_score = tables_with_structure / len(tables) if tables else 1.0
            scores.append(table_score * 0.3)  # 30% weight
        else:
            scores.append(0.3)
        
        # Metadata presence (helps with navigation)
        metadata_score = 0
        if page_metadata.get('drawing_type'):
            metadata_score += 0.5
        if page_metadata.get('sheet_number'):
            metadata_score += 0.5
        scores.append(metadata_score * 0.3)  # 30% weight
        
        return round(sum(scores), 2)

    def _check_table_headers(self, tables: List[Dict]) -> bool:
        """Check if tables have proper header rows"""
        if not tables:
            return True
        
        tables_with_headers = 0
        for table in tables:
            if table.get('data') and len(table['data']) > 1:
                # Simple heuristic: first row has different content pattern
                first_row = table['data'][0]
                if any(cell for cell in first_row if cell):
                    tables_with_headers += 1
        
        return tables_with_headers == len(tables)

    def _estimate_text_contrast(self, page: fitz.Page) -> float:
        """Estimate text contrast (simplified - would need full image analysis)"""
        # This is a placeholder - real implementation would analyze the rendered page
        # For now, return a reasonable default
        return 0.85  # Assume good contrast

    def _analyze_page_relationships(self, page_details: List[Dict]) -> Dict[str, Any]:
        """Revolutionary: Analyze semantic relationships between pages"""
        relationships = {
            'relationships': [],
            'groups': [],
            'flow_type': 'linear',
            'cross_references': []
        }
        
        # Group pages by drawing type
        type_groups = defaultdict(list)
        for page in page_details:
            if page.get('drawing_type'):
                type_groups[page['drawing_type']].append(page['page_number'])
        
        relationships['groups'] = [
            {'type': dtype, 'pages': pages, 'count': len(pages)}
            for dtype, pages in type_groups.items()
        ]
        
        # Detect page sequences (e.g., consecutive detail pages)
        for i in range(len(page_details) - 1):
            current = page_details[i]
            next_page = page_details[i + 1]
            
            # Same drawing type = likely related
            if current.get('drawing_type') == next_page.get('drawing_type'):
                relationships['relationships'].append({
                    'from': current['page_number'],
                    'to': next_page['page_number'],
                    'type': 'continuation',
                    'confidence': 0.8
                })
            
            # Sheet number sequence
            if current.get('sheet_number') and next_page.get('sheet_number'):
                try:
                    curr_num = ''.join(filter(str.isdigit, current['sheet_number']))
                    next_num = ''.join(filter(str.isdigit, next_page['sheet_number']))
                    if curr_num and next_num and int(next_num) == int(curr_num) + 1:
                        relationships['relationships'].append({
                            'from': current['page_number'],
                            'to': next_page['page_number'],
                            'type': 'sequence',
                            'confidence': 0.9
                        })
                except:
                    pass
        
        # Detect document flow type
        if len(type_groups) > 3:
            relationships['flow_type'] = 'mixed'
        elif any(dtype in ['detail', 'section'] for dtype in type_groups):
            relationships['flow_type'] = 'hierarchical'
        
        return relationships

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        # Cleanup before reporting stats
        self._cleanup_old_data()
        
        return {
            "service": "PDFService",
            "version": "7.0.0-REVOLUTIONARY-LIVING-DOCUMENT-LEAK-FIXED",
            "mode": "production_grade_with_universal_grid_detection_and_ai_enhancements",
            "capabilities": {
                "max_pages": self.max_pages,
                "batch_size": self.batch_size,
                "full_text_extraction": True,
                "table_extraction": True,
                "unlimited_tables": True,
                "grid_detection": True,
                "visual_grid_detection": OPENCV_AVAILABLE,
                "universal_grid_system": True,
                "grid_confidence_scoring": True,
                "multi_resolution_images": True,
                "ai_optimized": True,
                "thread_safe": True,
                "memory_managed": True,
                "memory_leak_fixed": True,
                "sse_events": True,
                "status_locking": True
            },
            "memory_management": {
                "max_sessions_in_memory": self.max_sessions_in_memory,
                "session_timeout_seconds": self.session_timeout_seconds,
                "cleanup_interval_seconds": self._cleanup_interval,
                "active_sessions": len(self.performance_metrics),
                "active_collaboration_sessions": len(self.collaboration_sessions),
                "active_status_locks": len(self._status_update_locks)
            },
            "revolutionary_features": {
                "ai_coordinate_annotations": self.enable_ai_annotations,
                "quality_scoring": self.enable_quality_scoring,
                "page_fingerprinting": self.enable_page_fingerprinting,
                "smart_caching": self.enable_smart_caching,
                "collaboration_markers": self.enable_collaboration_markers,
                "anomaly_detection": self.enable_anomaly_detection,
                "performance_tracking": self.enable_performance_tracking,
                "accessibility_scoring": self.enable_accessibility_scoring,
                "semantic_analysis": self.enable_semantic_analysis,
                "progressive_enhancement": self.enable_progressive_enhancement,
                "living_document": True
            },
            "grid_detection": {
                "text_based": True,
                "visual_based": OPENCV_AVAILABLE,
                "universal_grid": True,
                "grid_types": ["embedded", "text_based", "generated"],
                "line_detection": "Hough Transform" if OPENCV_AVAILABLE else "Not Available",
                "confidence_scoring": True,
                "fallback_support": True,
                "adaptive_sizing": True,
                "ai_coordinate_format": self.ai_coordinate_format,
                "ai_grid_reference": self.ai_grid_reference_format
            },
            "quality_metrics": {
                "thresholds": self.quality_thresholds,
                "tracking_enabled": self.enable_quality_scoring,
                "accessibility_compliance": self.enable_accessibility_scoring,
                "anomaly_detection_active": self.enable_anomaly_detection
            },
            "collaboration": {
                "real_time_markers": self.enable_collaboration_markers,
                "active_sessions": len(self.collaboration_sessions),
                "performance_tracking": len(self.performance_metrics),
                "living_document_mode": True
            },
            "security": {
                "input_validation": True,
                "size_limits": True,
                "thread_safety": True,
                "error_recovery": True,
                "status_locking": True,
                "race_condition_prevention": True,
                "fingerprint_tracking": self.enable_page_fingerprinting
            },
            "sse_events": {
                "page_processed": True,
                "resource_ready": True,
                "batch_complete": True,
                "processing_complete": True,
                "error_events": True,
                "safe_callbacks": True,
                "quality_updates": self.enable_quality_scoring,
                "anomaly_alerts": self.enable_anomaly_detection,
                "collaboration_events": self.enable_collaboration_markers
            },
            "performance": {
                "metrics_collected": sum(len(metrics) for metrics in self.performance_metrics.values()),
                "average_page_time": self._get_average_processing_time(),
                "quality_scores": self._get_average_quality_scores()
            }
        }

    def _get_average_processing_time(self) -> float:
        """Get average page processing time across all documents"""
        all_times = []
        for metrics in self.performance_metrics.values():
            all_times.extend(m.get('processing_time', 0) for m in metrics)
        return round(sum(all_times) / len(all_times), 3) if all_times else 0.0

    def _get_average_quality_scores(self) -> Dict[str, float]:
        """Get average quality scores across all processed pages"""
        all_scores = []
        for metrics in self.performance_metrics.values():
            all_scores.extend(m.get('quality_score', 0) for m in metrics)
        return {
            'average': round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0,
            'total_pages_scored': len(all_scores)
        }

    def _get_recent_page_processing_time(self, session_id: str) -> float:
        """Get average processing time for recent pages in this session"""
        if session_id not in self.performance_metrics:
            return 0.5  # Default
        
        recent_metrics = self.performance_metrics[session_id][-5:]  # Last 5 pages
        if not recent_metrics:
            return 0.5
        
        times = [m.get('processing_time', 0.5) for m in recent_metrics]
        return sum(times) / len(times)

    def _generate_optimization_suggestions(self, metadata: Dict[str, Any], 
                                          grid_systems: Dict[str, Any]) -> List[str]:
        """Revolutionary: Generate intelligent optimization suggestions"""
        suggestions = []
        
        # Analyze processing patterns
        avg_time_per_page = metadata.get('processing_time', 0) / metadata.get('pages_processed', 1)
        
        # Performance suggestions
        if avg_time_per_page > 3.0:
            suggestions.append("Consider reducing DPI to 100 for faster processing")
        
        # Grid system suggestions
        grid_types = []
        for grid in grid_systems.values():
            grid_types.append(grid.get('grid_type', 'unknown'))
        
        if grid_types.count('generated') > len(grid_types) * 0.8:
            suggestions.append("Document lacks embedded grids - consider pre-processing to add grid references")
        
        # Table extraction suggestions
        summary = metadata.get('extraction_summary', {})
        if summary.get('total_tables_found', 0) > summary.get('total_tables_extracted', 0) * 2:
            suggestions.append("Many complex tables detected - consider using specialized table extraction tools")
        
        # Quality suggestions
        page_details = metadata.get('page_details', [])
        if page_details:
            avg_quality = sum(p.get('quality_score', 0) for p in page_details) / len(page_details)
            if avg_quality < 0.6:
                suggestions.append("Low average quality score - check source document resolution")
        
        # Batch size suggestions
        if self.performance_metrics.get(metadata['document_id']):
            processing_times = [m['processing_time'] for m in self.performance_metrics[metadata['document_id']]]
            if len(processing_times) > 5:
                variance = np.var(processing_times) if processing_times else 0
                if variance > 2.0:
                    suggestions.append("High processing time variance - enable adaptive batch sizing")
        
        return suggestions[:5]  # Limit to top 5 suggestions

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        # Force final cleanup
        self._cleanup_old_data()
        
        # Clear all remaining data
        self.performance_metrics.clear()
        self.collaboration_sessions.clear()
        self._status_update_locks.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ PDF service cleaned up") 