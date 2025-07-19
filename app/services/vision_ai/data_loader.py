# data_loader.py - OPTIMIZED AND ROBUST VERSION
import asyncio
import base64
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG
from .patterns import VISUAL_PATTERNS

# Import from same package (relative import is correct)
from .enhanced_cache import EnhancedCache

# Add model imports that might be needed
from app.models.schemas import (
    VisualIntelligenceResult,
    ElementGeometry,
    SemanticHighlight
)

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Robust Data Loader - Handles all data loading for blueprint analysis
    
    CORE PHILOSOPHY:
    - Load ALL available thumbnails - no artificial limits
    - Smart page selection based on actual query needs
    - Robust error handling and recovery
    - Clear logging for debugging
    - Optimized for large documents
    
    KEY IMPROVEMENTS:
    - Timeout handling on all operations
    - Better error recovery
    - Cache validation
    - Progress tracking
    - Connection pooling
    - Batch operation optimization
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.cache = EnhancedCache()
        
        # Retry configuration with extended values for robustness
        self.retry_config = {
            "max_retries": CONFIG.get("max_retries", 5),
            "retry_delay": CONFIG.get("retry_delay", 1.0),
            "backoff_factor": CONFIG.get("retry_backoff_factor", 2.0),
            "max_delay": 30.0  # Maximum delay between retries
        }
        
        # Timeout configuration
        self.timeout_config = {
            "metadata": 60.0,  # 1 minute for metadata
            "thumbnail": CONFIG.get("thumbnail_load_timeout", 900.0),  # 15 minutes
            "page": CONFIG.get("page_load_timeout", 600.0),  # 10 minutes
            "batch": 1200.0,  # 20 minutes for batch operations
            "probe": 300.0,  # 5 minutes for probing
            "operation": 180.0  # 3 minutes per operation
        }
        
        # Performance tracking
        self._operation_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        self._last_progress_log = 0
        
        # Connection health
        self._consecutive_errors = 0
        self._last_error_time = 0
        
        logger.info("✅ DataLoader initialized with robust configuration")
        logger.info(f"⏱️ Timeouts: {json.dumps(self.timeout_config, indent=2)}")
    
    async def load_metadata(
        self,
        document_id: str,
        storage_service
    ) -> Optional[Dict[str, Any]]:
        """Load document metadata with robust error handling"""
        
        start_time = time.time()
        operation = "load_metadata"
        
        try:
            # Validate inputs
            if not document_id:
                raise ValueError("Document ID is required")
            
            cache_key = f"metadata_{document_id}"
            
            # Check cache with validation
            cached_metadata = self._get_validated_cache(cache_key, "metadata")
            if cached_metadata:
                logger.info(f"✅ Retrieved metadata from cache for {document_id}")
                return cached_metadata
            
            metadata_blob = f"{document_id}_metadata.json"
            logger.info(f"📥 Loading metadata: {metadata_blob}")
            
            # Load with retry and timeout
            metadata = await self._retry_with_backoff(
                storage_service.download_blob_as_json,
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=metadata_blob,
                operation_name=operation,
                timeout=self.timeout_config["metadata"]
            )
            
            if metadata:
                # Validate metadata structure
                if not isinstance(metadata, dict):
                    logger.error(f"❌ Invalid metadata format for {document_id}")
                    return None
                
                # Cache valid metadata
                self.cache.set(cache_key, metadata, "metadata")
                
                page_count = metadata.get('page_count', 'unknown')
                logger.info(f"✅ Metadata loaded: {page_count} pages, processing date: {metadata.get('processing_date', 'unknown')}")
                
                self._track_operation_success(operation, start_time)
                return metadata
            else:
                logger.warning(f"⚠️ Metadata file exists but is empty for {document_id}")
                
        except FileNotFoundError:
            logger.info(f"📄 No metadata file found for {document_id} - will probe for pages")
        except Exception as e:
            logger.error(f"❌ Error loading metadata for {document_id}: {type(e).__name__}: {e}")
            self._track_operation_error(operation, start_time, e)
        
        return None
    
    async def load_all_thumbnails(
        self,
        document_id: str,
        storage_service,
        page_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load ALL available thumbnails with robust error handling
        This is critical for intelligent page selection
        """
        
        start_time = time.time()
        operation = "load_all_thumbnails"
        
        logger.warning(f"🔍 Starting thumbnail loading for document: {document_id}")
        logger.warning(f"📦 Container: {self.settings.AZURE_CACHE_CONTAINER_NAME}")
        
        try:
            # Validate inputs
            if not document_id:
                raise ValueError("Document ID is required")
            
            # Check cache with validation
            cache_key = f"thumbnails_{document_id}"
            cached_thumbnails = self._get_validated_cache(cache_key, "thumbnail_batch")
            if cached_thumbnails and isinstance(cached_thumbnails, list) and len(cached_thumbnails) > 0:
                logger.info(f"✅ Retrieved {len(cached_thumbnails)} thumbnails from cache")
                self._track_operation_success(operation, start_time)
                return cached_thumbnails
            
            # Determine page count
            if page_count:
                max_pages_to_try = page_count
                logger.info(f"📄 Using metadata page count: {page_count} pages")
            else:
                # Probe for actual page count
                logger.info("🔍 No metadata - probing for actual page count...")
                max_pages_to_try = await self._probe_for_page_count(document_id, storage_service)
                logger.warning(f"📄 Probe detected {max_pages_to_try} pages")
            
            # Ensure minimum pages for blueprints
            min_pages_to_try = 10
            if max_pages_to_try < min_pages_to_try:
                logger.warning(f"⚠️ Increasing from {max_pages_to_try} to minimum {min_pages_to_try} pages")
                max_pages_to_try = min_pages_to_try
            
            # Optimize cache for large documents
            if max_pages_to_try > 50:
                self.cache.optimize_for_thumbnails(document_id, max_pages_to_try)
            
            logger.warning(f"🔄 Loading thumbnails for {max_pages_to_try} pages...")
            
            # Load thumbnails
            thumbnails = []
            if CONFIG.get("parallel_processing", True) and max_pages_to_try > 5:
                thumbnails = await self._batch_load_thumbnails_optimized(
                    document_id, storage_service, max_pages_to_try
                )
            else:
                # Sequential loading for small documents
                thumbnails = await self._sequential_load_thumbnails(
                    document_id, storage_service, max_pages_to_try
                )
            
            # Process results
            if thumbnails:
                # Sort by page number
                thumbnails.sort(key=lambda x: x["page"])
                
                # Log summary
                loaded_pages = [t["page"] for t in thumbnails]
                logger.warning(f"✅ Loaded {len(thumbnails)} thumbnails: pages {loaded_pages[:10]}{'...' if len(loaded_pages) > 10 else ''}")
                
                # Warn if missing pages
                if page_count and len(thumbnails) < page_count * 0.9:  # 90% threshold
                    logger.error(f"⚠️ Only loaded {len(thumbnails)} of {page_count} expected thumbnails!")
                
                # Cache the results
                if CONFIG.get("aggressive_caching", True) and len(thumbnails) > 0:
                    self.cache.set(cache_key, thumbnails, "thumbnail_batch")
                
                self._track_operation_success(operation, start_time)
            else:
                logger.error("❌ CRITICAL: No thumbnails could be loaded!")
                await self._log_debug_info(document_id, storage_service)
                self._track_operation_error(operation, start_time, "No thumbnails loaded")
            
            return thumbnails
            
        except Exception as e:
            logger.error(f"❌ Critical error loading thumbnails: {type(e).__name__}: {e}")
            self._track_operation_error(operation, start_time, e)
            return []
    
    async def load_specific_pages(
        self,
        document_id: str,
        storage_service,
        page_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Load specific high-res pages with robust error handling
        Only loads the pages selected by AI
        """
        
        start_time = time.time()
        operation = "load_specific_pages"
        
        logger.info(f"📥 Loading {len(page_numbers)} specific HIGH-RES pages: {page_numbers}")
        
        try:
            # Validate inputs
            if not document_id:
                raise ValueError("Document ID is required")
            if not page_numbers:
                logger.warning("⚠️ No page numbers specified")
                return []
            
            # Remove duplicates and sort
            page_numbers = sorted(list(set(page_numbers)))
            
            images = []
            
            # Check cache for each page
            cached_images = []
            uncached_pages = []
            
            for page_num in page_numbers:
                cache_key = f"page_{document_id}_{page_num}"
                cached_image = self._get_validated_cache(cache_key, "image")
                if cached_image:
                    cached_images.append(cached_image)
                    logger.debug(f"✅ Page {page_num} retrieved from cache")
                else:
                    uncached_pages.append(page_num)
            
            if cached_images:
                logger.info(f"✅ Retrieved {len(cached_images)} pages from cache")
                images.extend(cached_images)
            
            # Load uncached pages
            if uncached_pages:
                logger.info(f"📥 Loading {len(uncached_pages)} uncached pages: {uncached_pages}")
                
                if CONFIG.get("parallel_processing", True) and len(uncached_pages) > 2:
                    new_images = await self._batch_load_pages_optimized(
                        document_id, storage_service, uncached_pages
                    )
                else:
                    new_images = await self._sequential_load_pages(
                        document_id, storage_service, uncached_pages
                    )
                
                images.extend(new_images)
            
            # Sort by page number
            images.sort(key=lambda x: x["page"])
            
            # Validate results
            if not images and page_numbers:
                logger.error(f"❌ No pages could be loaded from requested: {page_numbers}")
                # Try page 1 as fallback
                logger.warning("⚠️ Attempting to load page 1 as fallback...")
                fallback = await self._load_single_page_robust(document_id, storage_service, 1)
                if fallback:
                    images = [fallback]
            
            loaded_pages = [img['page'] for img in images]
            logger.info(f"✅ Loaded {len(images)} high-res pages: {loaded_pages}")
            
            self._track_operation_success(operation, start_time)
            return images
            
        except Exception as e:
            logger.error(f"❌ Critical error loading pages: {type(e).__name__}: {e}")
            self._track_operation_error(operation, start_time, e)
            return []
    
    async def load_comprehensive_data(
        self,
        document_id: str,
        storage_service,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Load comprehensive data for selected pages with robust error handling
        """
        
        start_time = time.time()
        operation = "load_comprehensive_data"
        
        try:
            analyzed_pages = [img["page"] for img in images] if images else []
            logger.info(f"📊 Loading comprehensive data for {len(analyzed_pages)} pages")
            
            comprehensive_data = {
                "images": images,
                "analyzed_pages": analyzed_pages,
                "context": "",
                "grid_systems": {},
                "schedules": {},
                "legends": {},
                "drawing_info": {},
                "metadata": {}
            }
            
            if not analyzed_pages:
                logger.warning("⚠️ No pages to analyze")
                return comprehensive_data
            
            # Check for cached comprehensive data
            cache_key = f"comprehensive_{document_id}_{'-'.join(map(str, analyzed_pages[:10]))}"
            cached_data = self._get_validated_cache(cache_key, "analysis")
            if cached_data:
                logger.info("✅ Retrieved comprehensive data from cache")
                self._track_operation_success(operation, start_time)
                return cached_data
            
            # Load metadata for context
            metadata = await self.load_metadata(document_id, storage_service)
            if metadata:
                comprehensive_data["metadata"] = metadata
            
            # Extract context for each page
            if CONFIG.get("parallel_processing", True) and len(analyzed_pages) > 2:
                await self._parallel_extract_context(
                    document_id, storage_service, analyzed_pages, comprehensive_data
                )
            else:
                await self._sequential_extract_context(
                    document_id, storage_service, analyzed_pages, comprehensive_data
                )
            
            # Cache the comprehensive data
            if CONFIG.get("aggressive_caching", True) and comprehensive_data.get("context"):
                self.cache.set(cache_key, comprehensive_data, "analysis")
            
            logger.info(f"✅ Comprehensive data loaded: {len(comprehensive_data.get('context', ''))} chars of context")
            self._track_operation_success(operation, start_time)
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"❌ Error loading comprehensive data: {type(e).__name__}: {e}")
            self._track_operation_error(operation, start_time, e)
            return {
                "images": images,
                "analyzed_pages": [img["page"] for img in images] if images else [],
                "context": "",
                "error": str(e)
            }
    
    # === PRIVATE METHODS - OPTIMIZED ===
    
    async def _retry_with_backoff(
        self,
        func,
        *args,
        operation_name: str = "operation",
        timeout: Optional[float] = None,
        **kwargs
    ):
        """Execute operation with retry, timeout, and exponential backoff"""
        
        last_exception = None
        timeout = timeout or self.timeout_config.get("operation", 180.0)
        
        for attempt in range(self.retry_config["max_retries"]):
            try:
                # Wrap with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
                
                # Reset error tracking on success
                self._consecutive_errors = 0
                return result
                
            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"{operation_name} timed out after {timeout}s")
                logger.error(f"⏰ Attempt {attempt + 1}/{self.retry_config['max_retries']} timed out for {operation_name}")
                
            except FileNotFoundError:
                # Don't retry for missing files
                raise
                
            except Exception as e:
                last_exception = e
                self._consecutive_errors += 1
                self._last_error_time = time.time()
                
                # Log appropriate level based on attempt
                if attempt < self.retry_config["max_retries"] - 1:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for {operation_name}: {type(e).__name__}: {str(e)[:100]}")
                else:
                    logger.error(f"❌ All attempts failed for {operation_name}: {type(e).__name__}: {e}")
            
            # Calculate backoff delay
            if attempt < self.retry_config["max_retries"] - 1:
                delay = min(
                    self.retry_config["retry_delay"] * (self.retry_config["backoff_factor"] ** attempt),
                    self.retry_config["max_delay"]
                )
                
                logger.info(f"⏳ Waiting {delay:.1f}s before retry {attempt + 2}...")
                await asyncio.sleep(delay)
        
        # Check if we should warn about connection issues
        if self._consecutive_errors >= 5:
            logger.error(f"🔥 Multiple consecutive errors ({self._consecutive_errors}) - possible connection issue")
        
        raise last_exception
    
    async def _probe_for_page_count(
        self,
        document_id: str,
        storage_service
    ) -> int:
        """Probe to determine actual page count with optimization"""
        
        logger.info("🔍 Starting intelligent page count probe")
        start_time = time.time()
        
        try:
            # Quick check for common page counts using binary search approach
            checkpoints = [5, 10, 20, 30, 50, 75, 100, 150, 200]
            last_found = 0
            
            # Phase 1: Find rough upper bound
            for checkpoint in checkpoints:
                exists = await self._check_page_exists_cached(
                    document_id, storage_service, checkpoint
                )
                
                if exists:
                    last_found = checkpoint
                    logger.debug(f"✅ Page {checkpoint} exists")
                else:
                    logger.debug(f"❌ Page {checkpoint} not found")
                    break
                
                # Early exit for small documents
                if checkpoint <= 20 and not exists:
                    break
            
            # Phase 2: Binary search for exact count
            if last_found > 0:
                lower = last_found
                upper = checkpoints[checkpoints.index(last_found) + 1] if last_found in checkpoints[:-1] else last_found + 50
                
                # Binary search
                while lower < upper - 1:
                    mid = (lower + upper) // 2
                    exists = await self._check_page_exists_cached(
                        document_id, storage_service, mid
                    )
                    
                    if exists:
                        lower = mid
                    else:
                        upper = mid
                    
                    # Timeout check
                    if time.time() - start_time > self.timeout_config["probe"]:
                        logger.warning(f"⏰ Probe timeout - assuming {lower} pages")
                        return lower
                
                actual_count = lower
                logger.info(f"✅ Probe complete: found {actual_count} pages in {time.time() - start_time:.1f}s")
                return actual_count
            
            # Fallback: linear search for very small documents
            for page_num in range(1, 31):
                exists = await self._check_page_exists_cached(
                    document_id, storage_service, page_num
                )
                if not exists:
                    return page_num - 1
            
            # Default maximum
            logger.warning("⚠️ Document has 30+ pages, using 30 as limit")
            return 30
            
        except Exception as e:
            logger.error(f"❌ Probe error: {e}")
            return 10  # Safe default
    
    async def _check_page_exists_cached(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> bool:
        """Check if page exists with caching"""
        
        # Use a simple cache for existence checks
        cache_key = f"exists_{document_id}_{page_num}"
        
        try:
            # Check for thumbnail as it's smaller
            thumbnail_name = f"{document_id}_page_{page_num}_thumb.jpg"
            
            exists = await asyncio.wait_for(
                storage_service.blob_exists(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=thumbnail_name
                ),
                timeout=10.0  # Quick timeout for existence check
            )
            
            return exists
            
        except Exception:
            return False
    
    async def _batch_load_thumbnails_optimized(
        self,
        document_id: str,
        storage_service,
        max_pages: int
    ) -> List[Dict[str, Any]]:
        """Optimized batch loading of thumbnails"""
        
        thumbnails = []
        batch_size = CONFIG.get("batch_size", 25)
        semaphore = asyncio.Semaphore(CONFIG.get("storage_max_concurrent_downloads", 25))
        
        # Progress tracking
        loaded_count = 0
        error_count = 0
        
        async def load_with_semaphore(page_num: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self._load_single_thumbnail_robust(
                    document_id, storage_service, page_num
                )
        
        # Process in batches with progress logging
        for batch_start in range(1, max_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size, max_pages + 1)
            batch_pages = list(range(batch_start, batch_end))
            
            # Log progress
            if time.time() - self._last_progress_log > 5.0:  # Log every 5 seconds
                logger.info(f"📊 Progress: {loaded_count}/{max_pages} thumbnails loaded, {error_count} errors")
                self._last_progress_log = time.time()
            
            # Create tasks for batch
            tasks = [load_with_semaphore(page_num) for page_num in batch_pages]
            
            # Execute batch with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.timeout_config["batch"]
                )
                
                # Process results
                for i, result in enumerate(batch_results):
                    if isinstance(result, dict) and result:
                        thumbnails.append(result)
                        loaded_count += 1
                    elif isinstance(result, Exception):
                        error_count += 1
                        if error_count <= 5:  # Log first 5 errors
                            logger.debug(f"Batch error for page {batch_pages[i]}: {result}")
                
                # Small delay between batches to avoid overwhelming service
                if batch_end < max_pages:
                    await asyncio.sleep(CONFIG.get("thumbnail_batch_delay", 0.1))
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Batch timeout for pages {batch_start}-{batch_end}")
                error_count += len(batch_pages)
        
        # Final progress log
        logger.info(f"✅ Batch loading complete: {loaded_count} loaded, {error_count} errors")
        
        return thumbnails
    
    async def _sequential_load_thumbnails(
        self,
        document_id: str,
        storage_service,
        max_pages: int
    ) -> List[Dict[str, Any]]:
        """Sequential loading for small documents"""
        
        thumbnails = []
        
        for page_num in range(1, max_pages + 1):
            thumb = await self._load_single_thumbnail_robust(
                document_id, storage_service, page_num
            )
            if thumb:
                thumbnails.append(thumb)
                logger.debug(f"✅ Loaded thumbnail for page {page_num}")
            
            # Progress log
            if page_num % 10 == 0:
                logger.info(f"📊 Progress: {len(thumbnails)}/{page_num} thumbnails loaded")
        
        return thumbnails
    
    async def _load_single_thumbnail_robust(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """Load single thumbnail with multiple format attempts"""
        
        # Check cache first
        cache_key = f"thumb_{document_id}_{page_num}"
        cached_thumb = self._get_validated_cache(cache_key, "thumbnail")
        if cached_thumb:
            return cached_thumb
        
        # FIXED: Removed incorrect zero-padding and other patterns.
        # This now matches the exact filename created by pdf_service.py
        thumbnail_name = f"{document_id}_page_{page_num}_thumb.jpg"
        
        try:
            thumb_bytes = await self._retry_with_backoff(
                storage_service.download_blob_as_bytes,
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=thumbnail_name,
                operation_name=f"download_thumb_{page_num}",
                timeout=30.0  # 30s timeout per thumbnail
            )
            
            if thumb_bytes and len(thumb_bytes) > 0:
                thumb_data = {
                    "page": page_num,
                    "url": f"data:image/jpeg;base64,{base64.b64encode(thumb_bytes).decode('utf-8')}",
                    "size_kb": len(thumb_bytes) / 1024,
                    "is_thumbnail": True,
                    "pattern": thumbnail_name.split('_')[-1]
                }
                
                # Cache successful result
                if CONFIG.get("aggressive_caching", True):
                    self.cache.set(cache_key, thumb_data, "thumbnail")
                
                return thumb_data
                
        except FileNotFoundError:
            logger.debug(f"Thumbnail not found for page {page_num} with name {thumbnail_name}")
        except Exception as e:
            logger.debug(f"Error loading thumbnail {thumbnail_name}: {e}")
        
        return None
    
    async def _batch_load_pages_optimized(
        self,
        document_id: str,
        storage_service,
        page_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """Optimized batch loading of high-res pages"""
        
        images = []
        semaphore = asyncio.Semaphore(CONFIG.get("storage_max_concurrent_downloads", 25))
        
        async def load_with_semaphore(page_num: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self._load_single_page_robust(
                    document_id, storage_service, page_num
                )
        
        # Create all tasks
        tasks = [load_with_semaphore(page_num) for page_num in page_numbers]
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_config["batch"]
            )
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, dict) and result:
                    images.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Failed to load page {page_numbers[i]}: {result}")
                    
        except asyncio.TimeoutError:
            logger.error(f"❌ Batch page loading timeout")
        
        return images
    
    async def _sequential_load_pages(
        self,
        document_id: str,
        storage_service,
        page_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """Sequential loading of high-res pages"""
        
        images = []
        
        for page_num in page_numbers:
            image = await self._load_single_page_robust(
                document_id, storage_service, page_num
            )
            if image:
                images.append(image)
        
        return images
    
    async def _load_single_page_robust(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """Load single high-res page with multiple format attempts"""
        
        # Check cache first
        cache_key = f"page_{document_id}_{page_num}"
        cached_page = self._get_validated_cache(cache_key, "image")
        if cached_page:
            return cached_page
        
        # FIXED: Removed incorrect patterns to match pdf_service.py output
        formats_to_try = [
            (f"{document_id}_page_{page_num}_ai.jpg", "ai.jpg", "image/jpeg"),
            (f"{document_id}_page_{page_num}.png", "png", "image/png"),
        ]
        
        for blob_name, ext, mime_type in formats_to_try:
            try:
                image_bytes = await self._retry_with_backoff(
                    storage_service.download_blob_as_bytes,
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name,
                    operation_name=f"download_page_{page_num}",
                    timeout=self.timeout_config["page"]
                )
                
                if image_bytes and len(image_bytes) > 0:
                    logger.info(f"✅ Loaded page {page_num} using pattern: {blob_name}")
                    
                    page_data = {
                        "page": page_num,
                        "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                        "size_kb": len(image_bytes) / 1024,
                        "is_thumbnail": False,
                        "format": ext
                    }
                    
                    # Cache successful result
                    if CONFIG.get("aggressive_caching", True):
                        self.cache.set(cache_key, page_data, "image")
                    
                    return page_data
                    
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.debug(f"Error loading {blob_name}: {e}")
                continue
        
        logger.error(f"Failed to load page {page_num} in any format.")
        return None
    
    async def _parallel_extract_context(
        self,
        document_id: str,
        storage_service,
        analyzed_pages: List[int],
        comprehensive_data: Dict[str, Any]
    ) -> None:
        """Extract context data in parallel"""
        
        semaphore = asyncio.Semaphore(10)  # Limit concurrent extractions
        
        async def extract_with_semaphore(page_num: int) -> Tuple[int, Optional[Dict[str, Any]]]:
            async with semaphore:
                context = await self._extract_page_context(
                    document_id, storage_service, page_num
                )
                return page_num, context
        
        # Create tasks
        tasks = [extract_with_semaphore(page_num) for page_num in analyzed_pages]
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_config["batch"]
            )
            
            # Process results
            for result in results:
                if isinstance(result, tuple):
                    page_num, context_data = result
                    if context_data:
                        self._merge_context_data(comprehensive_data, context_data, page_num)
                        
        except asyncio.TimeoutError:
            logger.error("❌ Context extraction timeout")
    
    async def _sequential_extract_context(
        self,
        document_id: str,
        storage_service,
        analyzed_pages: List[int],
        comprehensive_data: Dict[str, Any]
    ) -> None:
        """Extract context data sequentially"""
        
        for page_num in analyzed_pages:
            context_data = await self._extract_page_context(
                document_id, storage_service, page_num
            )
            if context_data:
                self._merge_context_data(comprehensive_data, context_data, page_num)
    
    async def _extract_page_context(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context data for a single page"""
        
        try:
            # This filename must match what pdf_service.py creates
            context_blob = f"{document_id}_context.txt" # This was incorrect, should be per-page
            
            context_data = await self._retry_with_backoff(
                storage_service.download_blob_as_json,
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=context_blob,
                operation_name=f"context_{page_num}",
                timeout=30.0
            )
            
            if context_data:
                logger.debug(f"✅ Loaded context for page {page_num}")
                return context_data
                
        except FileNotFoundError:
            logger.debug(f"No context file for page {page_num}")
        except Exception as e:
            logger.debug(f"Context extraction error for page {page_num}: {e}")
        
        return None
    
    def _merge_context_data(
        self,
        comprehensive_data: Dict[str, Any],
        context_data: Dict[str, Any],
        page_num: int
    ) -> None:
        """Merge context data into comprehensive data structure"""
        
        # Add to context string
        comprehensive_data["context"] += f"\n\n--- PAGE {page_num} ---\n"
        
        # Format context data safely
        try:
            # Assuming context_data is the text content for that page
            comprehensive_data["context"] += context_data
        except:
            comprehensive_data["context"] += str(context_data)
        
        # Extract specific data types if available in metadata
        metadata = comprehensive_data.get("metadata", {})
        page_details = next((p for p in metadata.get("page_details", []) if p.get("page_number") == page_num), None)

        if page_details:
            if "grid_system" in page_details:
                comprehensive_data["grid_systems"][page_num] = page_details["grid_system"]
            
            if "schedules" in page_details:
                comprehensive_data["schedules"][page_num] = page_details["schedules"]
            
            if "drawing_type" in page_details:
                comprehensive_data["drawing_info"][page_num] = {
                    "type": page_details["drawing_type"],
                    "scale": page_details.get("scale"),
                    "title": page_details.get("title")
                }
    
    def _get_validated_cache(self, key: str, cache_type: str) -> Optional[Any]:
        """Get from cache with validation"""
        
        value = self.cache.get(key, cache_type)
        
        # Validate cached data
        if value is not None:
            # Check for empty collections
            if isinstance(value, (list, dict, str)) and not value:
                logger.warning(f"⚠️ Cache returned empty {type(value).__name__} for {key}")
                return None
            
            # Validate image/thumbnail data
            if cache_type in ["image", "thumbnail", "thumbnail_batch"] and isinstance(value, list):
                 for item in value:
                    if not (isinstance(item, dict) and "url" in item and item["url"]):
                        logger.warning(f"⚠️ Invalid cached image data for {key}")
                        return None
            elif cache_type in ["image", "thumbnail"] and isinstance(value, dict):
                if "url" not in value or not value["url"]:
                    logger.warning(f"⚠️ Invalid cached image data for {key}")
                    return None
        
        return value
    
    def _track_operation_success(self, operation: str, start_time: float):
        """Track successful operation metrics"""
        
        elapsed = time.time() - start_time
        self._operation_stats[operation]["count"] += 1
        self._operation_stats[operation]["total_time"] += elapsed
        
        # Log slow operations
        if elapsed > self.timeout_config.get("warning_threshold", 300.0):
            logger.warning(f"⚠️ Slow operation: {operation} took {elapsed:.1f}s")
    
    def _track_operation_error(self, operation: str, start_time: float, error: Any):
        """Track operation errors"""
        
        elapsed = time.time() - start_time
        self._operation_stats[operation]["errors"] += 1
        
        logger.error(f"❌ Operation {operation} failed after {elapsed:.1f}s: {error}")
    
    async def _log_debug_info(self, document_id: str, storage_service):
        """Log debug information when loading fails"""
        
        try:
            # List some files
            files = await storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=document_id
            )
            
            logger.error(f"🔍 Debug info for {document_id}:")
            logger.error(f"   Total files: {len(files)}")
            
            # Categorize files
            thumbs = [f for f in files if "thumb" in f.lower()]
            pngs = [f for f in files if f.endswith(".png")]
            jpgs = [f for f in files if f.endswith(".jpg")]
            
            logger.error(f"   Thumbnails: {len(thumbs)}")
            logger.error(f"   PNGs: {len(pngs)}")
            logger.error(f"   JPGs: {len(jpgs)}")
            
            # Show examples
            if files:
                logger.error(f"   Example files: {files[:5]}")
                
        except Exception as e:
            logger.error(f"   Could not get debug info: {e}")
    
    async def debug_list_all_document_files(
        self,
        document_id: str,
        storage_service
    ) -> List[str]:
        """Debug method to list all files for a document"""
        
        try:
            all_files = await storage_service.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=document_id
            )
            
            logger.warning(f"🔍 All files for document {document_id}:")
            for f in sorted(all_files):
                logger.warning(f"  - {f}")
            
            # Count file types
            thumb_count = sum(1 for f in all_files if "_thumb.jpg" in f)
            png_count = sum(1 for f in all_files if f.endswith(".png"))
            ai_jpg_count = sum(1 for f in all_files if "_ai.jpg" in f)
            
            logger.warning(f"📊 File counts: {thumb_count} thumbnails, {png_count} PNGs, {ai_jpg_count} AI JPGs")
            
            return all_files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        
        stats = {}
        for op, data in self._operation_stats.items():
            if data["count"] > 0:
                avg_time = data["total_time"] / data["count"]
                stats[op] = {
                    "count": data["count"],
                    "errors": data["errors"],
                    "avg_time": f"{avg_time:.2f}s",
                    "total_time": f"{data['total_time']:.2f}s",
                    "error_rate": f"{(data['errors'] / data['count']) * 100:.1f}%"
                }
        
        return stats