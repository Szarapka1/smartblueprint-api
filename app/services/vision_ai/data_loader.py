# data_loader.py
import asyncio
import base64
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

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
    """Handles all data loading - thumbnails for analysis, high-res only for selected pages"""
    
    def __init__(self, settings):
        self.settings = settings
        self.cache = EnhancedCache()
        
        # Retry configuration
        self.retry_config = {
            "max_retries": CONFIG["max_retries"],
            "retry_delay": CONFIG["retry_delay"],
            "backoff_factor": 2.0
        }
    
    async def load_metadata(
        self,
        document_id: str,
        storage_service
    ) -> Optional[Dict[str, Any]]:
        """Load document metadata to get page count and other info"""
        
        cache_key = f"metadata_{document_id}"
        cached_metadata = self.cache.get(cache_key, "metadata")
        
        if cached_metadata:
            logger.info(f"✅ Retrieved metadata from cache")
            return cached_metadata
        
        try:
            metadata_blob = f"{document_id}_metadata.json"
            
            # Use the new download_blob_as_json method
            metadata = await self._retry_with_backoff(
                storage_service.download_blob_as_json,
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=metadata_blob
            )
            
            if metadata:
                self.cache.set(cache_key, metadata, "metadata")
                logger.info(f"📋 Metadata loaded: {metadata.get('page_count', 'unknown')} pages")
                return metadata
            else:
                logger.warning(f"⚠️ Metadata file exists but is empty for {document_id}")
                
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        
        return None
    
    async def load_all_thumbnails(
        self,
        document_id: str,
        storage_service,
        page_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load ALL available thumbnails for complete document analysis
        This is critical for intelligent page selection
        """
        
        # Check cache first
        cache_key = f"thumbnails_{document_id}"
        cached_thumbnails = self.cache.get(cache_key, "thumbnail")
        if cached_thumbnails:
            logger.info(f"✅ Retrieved {len(cached_thumbnails)} thumbnails from cache")
            return cached_thumbnails
        
        # Determine how many pages to try loading
        if page_count:
            max_pages_to_try = page_count
            logger.info(f"📄 Loading thumbnails for {page_count} pages (from metadata)")
        else:
            # Probe for actual page count
            logger.info("🔍 Probing for actual page count...")
            max_pages_to_try = await self._probe_for_page_count(
                document_id, storage_service
            )
            logger.warning(f"⚠️ No metadata found - probe detected {max_pages_to_try} pages")
        
        # IMPORTANT: Ensure we try to load at least 10 pages for blueprints
        # Many construction documents have 10+ pages
        min_pages_to_try = 10
        if max_pages_to_try < min_pages_to_try:
            logger.warning(f"⚠️ Only {max_pages_to_try} pages detected, but will try {min_pages_to_try} minimum")
            max_pages_to_try = min_pages_to_try
        
        logger.info(f"🔄 Loading thumbnails for up to {max_pages_to_try} pages...")
        
        # Load thumbnails in parallel batches
        thumbnails = []
        if CONFIG["parallel_processing"]:
            thumbnails = await self._batch_load_thumbnails(
                document_id, storage_service, max_pages_to_try
            )
        else:
            # Sequential loading
            for page_num in range(1, max_pages_to_try + 1):
                thumb = await self._load_single_thumbnail(
                    document_id, storage_service, page_num
                )
                if thumb:
                    thumbnails.append(thumb)
        
        if thumbnails:
            logger.info(f"✅ Successfully loaded {len(thumbnails)} thumbnails")
            
            # Warn if we got fewer than expected
            if page_count and len(thumbnails) < page_count:
                logger.warning(f"⚠️ Only loaded {len(thumbnails)} of {page_count} expected thumbnails")
            
            # Sort thumbnails by page number
            thumbnails.sort(key=lambda x: x["page"])
            
            # Cache the thumbnails
            if CONFIG["aggressive_caching"]:
                self.cache.set(cache_key, thumbnails, "thumbnail")
        else:
            logger.error("❌ CRITICAL: No thumbnails could be loaded!")
        
        return thumbnails
    
    async def load_specific_pages(
        self,
        document_id: str,
        storage_service,
        page_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Load ONLY the specific high-res pages selected by AI
        This is the key optimization - we don't load all pages
        """
        
        logger.info(f"📥 Loading {len(page_numbers)} specific HIGH-RES pages: {page_numbers}")
        
        images = []
        
        # Check cache for each page
        cached_images = []
        uncached_pages = []
        
        for page_num in page_numbers:
            cache_key = f"page_{document_id}_{page_num}"
            cached_image = self.cache.get(cache_key, "image")
            if cached_image:
                cached_images.append(cached_image)
                logger.debug(f"✅ Page {page_num} retrieved from cache")
            else:
                uncached_pages.append(page_num)
        
        if cached_images:
            logger.info(f"✅ Retrieved {len(cached_images)} pages from cache")
            images.extend(cached_images)
        
        # Load only the uncached pages
        if uncached_pages:
            logger.info(f"📥 Loading {len(uncached_pages)} uncached pages: {uncached_pages}")
            
            if CONFIG["parallel_processing"] and len(uncached_pages) > 2:
                # Batch loading for performance
                new_images = await self._batch_load_pages(
                    document_id, storage_service, uncached_pages
                )
            else:
                # Sequential loading for small numbers
                tasks = [
                    self._load_single_page(document_id, storage_service, page_num)
                    for page_num in uncached_pages
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                new_images = [img for img in results if img and not isinstance(img, Exception)]
            
            images.extend(new_images)
        
        # Sort by page number
        images.sort(key=lambda x: x["page"])
        
        # Fallback if no images loaded
        if not images and page_numbers:
            logger.warning("⚠️ No requested pages found, falling back to page 1")
            fallback = await self._load_single_page(document_id, storage_service, 1)
            if fallback:
                images = [fallback]
        
        logger.info(f"✅ Loaded {len(images)} high-resolution pages: {[img['page'] for img in images]}")
        
        return images
    
    async def load_comprehensive_data(
        self,
        document_id: str,
        storage_service,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Load comprehensive data for the SELECTED pages only
        Includes extracted text, grid systems, schedules, etc.
        """
        
        analyzed_pages = [img["page"] for img in images]
        logger.info(f"📊 Loading comprehensive data for {len(analyzed_pages)} selected pages")
        
        comprehensive_data = {
            "images": images,
            "analyzed_pages": analyzed_pages,
            "context": "",
            "grid_systems": {},
            "schedules": {},
            "legends": {},
            "drawing_info": {}
        }
        
        # Check for cached comprehensive data
        cache_key = f"comprehensive_{document_id}_{'-'.join(map(str, analyzed_pages))}"
        cached_data = self.cache.get(cache_key, "analysis")
        if cached_data:
            logger.info("✅ Retrieved comprehensive data from cache")
            return cached_data
        
        # Extract context for each selected page
        if CONFIG["parallel_processing"]:
            # Parallel context extraction
            context_tasks = []
            for page_num in analyzed_pages:
                context_tasks.append(
                    self._extract_page_context(document_id, storage_service, page_num)
                )
            
            context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
            
            for i, context_data in enumerate(context_results):
                if context_data and not isinstance(context_data, Exception):
                    page_num = analyzed_pages[i]
                    self._merge_context_data(comprehensive_data, context_data, page_num)
        else:
            # Sequential extraction
            for page_num in analyzed_pages:
                context_data = await self._extract_page_context(
                    document_id, storage_service, page_num
                )
                if context_data:
                    self._merge_context_data(comprehensive_data, context_data, page_num)
        
        # Cache the comprehensive data
        if CONFIG["aggressive_caching"]:
            self.cache.set(cache_key, comprehensive_data, "analysis")
        
        logger.info(f"✅ Comprehensive data loaded for pages {analyzed_pages}")
        
        return comprehensive_data
    
    async def _probe_for_page_count(
        self,
        document_id: str,
        storage_service
    ) -> int:
        """Probe to determine actual page count"""
        
        logger.info("🔍 Starting page count probe")
        
        # First, try a quick linear check for common page counts
        common_page_counts = [5, 10, 15, 20, 25, 30]
        last_found = 0
        
        for check_page in common_page_counts:
            exists = await self._check_page_exists(
                document_id, storage_service, check_page
            )
            if exists:
                last_found = check_page
                logger.debug(f"✅ Page {check_page} exists")
            else:
                logger.debug(f"❌ Page {check_page} does not exist")
                # If we found pages before but not this one, do detailed check
                if last_found > 0:
                    break
        
        # Now do a detailed check between last_found and the next checkpoint
        if last_found > 0:
            start = last_found
            end = common_page_counts[common_page_counts.index(last_found) + 1] if last_found in common_page_counts[:-1] else last_found + 5
            
            for page_num in range(start + 1, end):
                exists = await self._check_page_exists(
                    document_id, storage_service, page_num
                )
                if not exists:
                    actual_count = page_num - 1
                    logger.info(f"✅ Probe complete: found {actual_count} pages")
                    return actual_count
            
            # If all exist up to end, that's our count
            logger.info(f"✅ Probe complete: found at least {end - 1} pages")
            return end - 1
        
        # If nothing found in common counts, do linear search from 1
        for page_num in range(1, 31):  # Check up to 30 pages
            exists = await self._check_page_exists(
                document_id, storage_service, page_num
            )
            if not exists:
                actual_count = page_num - 1
                logger.info(f"✅ Probe complete: found {actual_count} pages")
                return actual_count
        
        # If we get here, there are 30+ pages
        logger.warning("⚠️ Document has 30+ pages, using 30 as limit")
        return 30
    
    async def _check_page_exists(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> bool:
        """Check if a specific page exists"""
        
        # Check for thumbnail as it's smaller and faster
        thumbnail_name = f"{document_id}_page_{page_num}_thumb.jpg"
        try:
            exists = await storage_service.blob_exists(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=thumbnail_name
            )
            logger.debug(f"Checking {thumbnail_name}: exists = {exists}")
            return exists
        except:
            return False
    
    async def _batch_load_thumbnails(
        self,
        document_id: str,
        storage_service,
        max_pages: int
    ) -> List[Dict[str, Any]]:
        """Batch load thumbnails for better performance"""
        
        thumbnails = []
        batch_size = CONFIG["batch_size"]
        
        for batch_start in range(1, max_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size, max_pages + 1)
            batch_pages = range(batch_start, batch_end)
            
            logger.debug(f"Loading thumbnail batch: pages {batch_start}-{batch_end-1}")
            
            batch_tasks = [
                self._load_single_thumbnail(document_id, storage_service, page_num)
                for page_num in batch_pages
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if result and not isinstance(result, Exception):
                    thumbnails.append(result)
            
            # Small delay between batches to avoid overwhelming the service
            if batch_end < max_pages:
                await asyncio.sleep(0.1)
        
        return thumbnails
    
    async def _batch_load_pages(
        self,
        document_id: str,
        storage_service,
        page_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """Batch load specific high-res pages"""
        
        images = []
        batch_size = CONFIG["batch_size"]
        
        # Process in batches
        for i in range(0, len(page_numbers), batch_size):
            batch = page_numbers[i:i+batch_size]
            
            logger.debug(f"Loading page batch: {batch}")
            
            batch_tasks = [
                self._load_single_page(document_id, storage_service, page_num)
                for page_num in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if result and not isinstance(result, Exception):
                    images.append(result)
            
            # Small delay between batches
            if i + batch_size < len(page_numbers):
                await asyncio.sleep(0.1)
        
        return images
    
    async def _load_single_thumbnail(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """Load a single thumbnail"""
        
        # Check cache first
        cache_key = f"thumb_{document_id}_{page_num}"
        cached_thumb = self.cache.get(cache_key, "thumbnail")
        if cached_thumb:
            return cached_thumb
        
        try:
            # Thumbnail naming pattern from your file listing
            thumbnail_name = f"{document_id}_page_{page_num}_thumb.jpg"
            
            thumb_bytes = await storage_service.download_blob_as_bytes(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=thumbnail_name
            )
            
            if thumb_bytes:
                logger.debug(f"✅ Loaded thumbnail for page {page_num}")
                
                thumb_data = {
                    "page": page_num,
                    "url": f"data:image/jpeg;base64,{base64.b64encode(thumb_bytes).decode('utf-8')}",
                    "size_kb": len(thumb_bytes) / 1024,
                    "is_thumbnail": True
                }
                
                # Cache the thumbnail
                if CONFIG["aggressive_caching"]:
                    self.cache.set(cache_key, thumb_data, "thumbnail")
                
                return thumb_data
                
        except Exception as e:
            if "not found" in str(e).lower():
                logger.debug(f"📭 Thumbnail for page {page_num} does not exist")
            else:
                logger.error(f"❌ Error loading thumbnail for page {page_num}: {e}")
        
        return None
    
    async def _load_single_page(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """Load a single high-res page with retry"""
        
        # Check cache first
        cache_key = f"page_{document_id}_{page_num}"
        cached_page = self.cache.get(cache_key, "image")
        if cached_page:
            logger.debug(f"✅ Page {page_num} from cache")
            return cached_page
        
        # Try different formats based on what we see in your file listing
        # Pattern: {document_id}_page_{num}.png and {document_id}_page_{num}_ai.jpg
        formats_to_try = [
            (f"{document_id}_page_{page_num}_ai.jpg", "ai.jpg", "image/jpeg"),
            (f"{document_id}_page_{page_num}.png", "png", "image/png"),
            (f"{document_id}_page_{page_num}.jpg", "jpg", "image/jpeg")
        ]
        
        for blob_name, ext, mime_type in formats_to_try:
            try:
                logger.debug(f"🔍 Trying to load: {blob_name}")
                
                image_bytes = await self._retry_with_backoff(
                    storage_service.download_blob_as_bytes,
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
                
                if image_bytes:
                    logger.info(f"✅ Loaded HIGH-RES page {page_num} ({ext})")
                    
                    page_data = {
                        "page": page_num,
                        "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                        "size_kb": len(image_bytes) / 1024,
                        "is_thumbnail": False,
                        "format": ext
                    }
                    
                    # Cache the page
                    if CONFIG["aggressive_caching"]:
                        self.cache.set(cache_key, page_data, "image")
                    
                    return page_data
                    
            except Exception as e:
                logger.debug(f"Could not load page {page_num} as {ext}: {e}")
        
        logger.error(f"❌ Failed to load page {page_num} in any format")
        return None
    
    async def _extract_page_context(
        self,
        document_id: str,
        storage_service,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context data for a single page"""
        
        try:
            context_blob = f"{document_id}_page_{page_num}_context.json"
            
            # Use the new download_blob_as_json method
            context_data = await storage_service.download_blob_as_json(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=context_blob
            )
            
            if context_data:
                logger.debug(f"✅ Loaded context for page {page_num}")
                return context_data
            
        except Exception as e:
            logger.debug(f"No pre-extracted context for page {page_num}: {e}")
            
        return None
    
    def _merge_context_data(
        self,
        comprehensive_data: Dict[str, Any],
        context_data: Dict[str, Any],
        page_num: int
    ) -> None:
        """Merge context data into comprehensive data structure"""
        
        # Add to context string
        comprehensive_data["context"] += f"\n\nPage {page_num} Context:\n"
        
        # Format context data
        import json
        comprehensive_data["context"] += json.dumps(context_data, indent=2)
        
        # Extract specific data types
        if "grid_system" in context_data:
            comprehensive_data["grid_systems"][page_num] = context_data["grid_system"]
        
        if "schedules" in context_data:
            comprehensive_data["schedules"][page_num] = context_data["schedules"]
        
        if "drawing_type" in context_data:
            comprehensive_data["drawing_info"][page_num] = {
                "type": context_data["drawing_type"],
                "scale": context_data.get("scale"),
                "title": context_data.get("title")
            }
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(self.retry_config["max_retries"]):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = self.retry_config["retry_delay"] * (
                        self.retry_config["backoff_factor"] ** attempt
                    )
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.retry_config['max_retries']} attempts failed")
        
        raise last_exception
    
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
