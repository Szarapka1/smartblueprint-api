# data_loader.py - FINAL, ROBUST, AND COMPLETE VERSION

import asyncio
import base64
import logging
import time
import json
from typing import List, Dict, Any, Optional

from app.core.config import CONFIG
from .enhanced_cache import EnhancedCache

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Robust Data Loader - Handles all data loading for blueprint analysis.
    This definitive version combines robust, production-grade features with a
    corrected data pipeline to load all necessary ground-truth data.
    """

    def __init__(self, settings):
        self.settings = settings
        self.cache = EnhancedCache()

        # Configuration for timeouts and retries from the application config
        self.timeout_config = {
            "metadata": 60.0,
            "thumbnail_batch": CONFIG.get("thumbnail_load_timeout", 900.0),
            "page_batch": CONFIG.get("page_load_timeout", 600.0),
            "probe": 300.0,
            "operation": 180.0
        }
        self.retry_config = {
            "max_retries": CONFIG.get("max_retries", 3),
            "retry_delay": CONFIG.get("retry_delay", 1.0),
            "backoff_factor": CONFIG.get("retry_backoff_factor", 2.0)
        }
        logger.info("✅ DataLoader initialized (Definitive Version)")

    async def _retry_operation(self, func, *args, **kwargs):
        """A simplified retry wrapper for core operations."""
        last_exception = None
        for attempt in range(self.retry_config["max_retries"]):
            try:
                # Use a specific timeout for the operation if provided
                timeout = kwargs.pop("timeout", self.timeout_config["operation"])
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except FileNotFoundError as e:
                # Do not retry if the file simply doesn't exist
                raise e
            except Exception as e:
                last_exception = e
                logger.warning(f"Operation {func.__name__} failed on attempt {attempt + 1}. Retrying...")
                await asyncio.sleep(self.retry_config["retry_delay"] * (self.retry_config["backoff_factor"] ** attempt))
        raise last_exception

    async def load_metadata(self, document_id: str, storage_service) -> Optional[Dict[str, Any]]:
        """Loads the primary metadata file for a document."""
        cache_key = f"metadata_{document_id}"
        cached_metadata = self.cache.get(cache_key, "metadata")
        if cached_metadata:
            logger.info(f"✅ Retrieved metadata for {document_id} from cache.")
            return cached_metadata

        try:
            metadata_blob = f"{document_id}_metadata.json"
            logger.info(f"📥 Loading metadata from: {metadata_blob}")
            metadata = await self._retry_operation(
                storage_service.download_blob_as_json,
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=metadata_blob,
                timeout=self.timeout_config["metadata"]
            )
            if metadata:
                self.cache.set(cache_key, metadata, "metadata")
                logger.info(f"✅ Metadata for {document_id} loaded and cached.")
            return metadata
        except FileNotFoundError:
            logger.warning(f"⚠️ Metadata file not found for {document_id}.")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading metadata for {document_id}: {e}")
            return None

    async def load_all_thumbnails(self, document_id: str, storage_service, page_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads all available thumbnails for a document to enable intelligent page selection."""
        cache_key = f"thumbnails_{document_id}"
        cached_thumbnails = self.cache.get(cache_key, "thumbnail_batch")
        if cached_thumbnails:
            logger.info(f"✅ Retrieved {len(cached_thumbnails)} thumbnails for {document_id} from cache.")
            return cached_thumbnails

        if not page_count:
            page_count = await self._probe_for_page_count(document_id, storage_service)
            logger.info(f"📄 Probe determined page count: {page_count}")

        logger.info(f"📥 Loading all thumbnails for {page_count} pages of document {document_id}...")
        
        async def _load_thumb(p_num):
            try:
                blob_name = f"{document_id}_page_{p_num}_thumb.jpg"
                thumb_bytes = await storage_service.download_blob_as_bytes(self.settings.AZURE_CACHE_CONTAINER_NAME, blob_name)
                return {
                    "page": p_num,
                    "url": f"data:image/jpeg;base64,{base64.b64encode(thumb_bytes).decode('utf-8')}",
                }
            except FileNotFoundError:
                return None
            except Exception as e:
                logger.warning(f"Could not load thumbnail for page {p_num}: {e}")
                return None

        tasks = [_load_thumb(i) for i in range(1, page_count + 1)]
        results = await asyncio.gather(*tasks)
        thumbnails = [res for res in results if res is not None]
        
        if thumbnails:
            self.cache.set(cache_key, thumbnails, "thumbnail_batch")
        
        logger.info(f"✅ Loaded {len(thumbnails)} of {page_count} thumbnails for {document_id}.")
        return thumbnails

    async def load_specific_pages(self, document_id: str, storage_service, page_numbers: List[int]) -> List[Dict[str, Any]]:
        """Loads specific high-resolution page images required for analysis."""
        logger.info(f"📥 Loading {len(page_numbers)} high-res pages for {document_id}: {page_numbers}")

        async def _load_page(p_num):
            cache_key = f"page_{document_id}_{p_num}"
            cached_page = self.cache.get(cache_key, "image")
            if cached_page:
                return cached_page
            try:
                blob_name = f"{document_id}_page_{p_num}_ai.jpg"
                image_bytes = await storage_service.download_blob_as_bytes(self.settings.AZURE_CACHE_CONTAINER_NAME, blob_name)
                page_data = {
                    "page": p_num,
                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                }
                self.cache.set(cache_key, page_data, "image")
                return page_data
            except FileNotFoundError:
                logger.error(f"❌ High-res image for page {p_num} not found.")
                return None
            except Exception as e:
                logger.error(f"❌ Failed to load page {p_num}: {e}")
                return None

        tasks = [_load_page(p_num) for p_num in sorted(list(set(page_numbers)))]
        results = await asyncio.gather(*tasks)
        images = [res for res in results if res is not None]
        logger.info(f"✅ Loaded {len(images)} high-res pages for {document_id}.")
        return images

    async def load_comprehensive_data(
        self,
        document_id: str,
        storage_service,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Loads all data needed for a full analysis, including the critical
        ground-truth data from grid system and document index JSON files.
        """
        logger.info(f"📊 Loading comprehensive data for document: {document_id}")

        comprehensive_data = {
            "images": images,
            "context": "",
            "grid_systems": None,
            "document_index": None
        }

        container = self.settings.AZURE_CACHE_CONTAINER_NAME

        # Define tasks to run in parallel
        async def _load_context():
            try:
                comprehensive_data["context"] = await storage_service.download_blob_as_text(container, f"{document_id}_context.txt")
                logger.info(f"✅ Loaded context file ({len(comprehensive_data['context'])} chars).")
            except FileNotFoundError:
                logger.warning(f"⚠️ Context file not found for {document_id}.")
            except Exception as e:
                logger.error(f"❌ Failed to load context file for {document_id}: {e}")

        async def _load_grids():
            try:
                comprehensive_data["grid_systems"] = await storage_service.download_blob_as_json(container, f"{document_id}_grid_systems.json")
                logger.info(f"✅ Loaded grid systems file with data for {len(comprehensive_data['grid_systems'])} page(s).")
            except FileNotFoundError:
                logger.warning(f"⚠️ Grid systems file not found. Spatial validation will be limited.")
            except Exception as e:
                logger.error(f"❌ Failed to load or parse grid systems file for {document_id}: {e}")

        async def _load_index():
            try:
                comprehensive_data["document_index"] = await storage_service.download_blob_as_json(container, f"{document_id}_document_index.json")
                logger.info(f"✅ Loaded document index file with {len(comprehensive_data['document_index'].get('page_index', []))} indexed pages.")
            except FileNotFoundError:
                logger.warning(f"⚠️ Document index not found. Count reconciliation will be limited.")
            except Exception as e:
                logger.error(f"❌ Failed to load or parse document index file for {document_id}: {e}")
        
        # Run all data loading tasks concurrently
        await asyncio.gather(_load_context(), _load_grids(), _load_index())

        return comprehensive_data

    async def _probe_for_page_count(self, document_id: str, storage_service) -> int:
        """Probe storage to determine the actual number of pages available."""
        # This is a simplified version of the robust probing logic from your original file.
        # A full implementation would use a more efficient search.
        max_probe = CONFIG.get("thumbnail_probe_max_pages", 200)
        for i in range(1, max_probe + 2):
            exists = await storage_service.blob_exists(
                self.settings.AZURE_CACHE_CONTAINER_NAME,
                f"{document_id}_page_{i}_thumb.jpg"
            )
            if not exists:
                return i - 1
        return max_probe