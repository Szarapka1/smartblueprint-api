# app/services/storage_service.py - ULTRA-RELIABLE VERSION FOR TESTING

import logging
import asyncio
import json
import time
import os
import re
import gzip
import hashlib
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import wraps

from app.core.config import get_settings, CONFIG
from azure.storage.blob.aio import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import (
    ResourceExistsError, ResourceNotFoundError, AzureError, 
    ClientAuthenticationError, HttpResponseError, ServiceRequestError
)
from azure.storage.blob import BlobPrefix, ContentSettings, BlobProperties

logger = logging.getLogger(__name__)

# --- Utility Functions ---

def retry_with_backoff(max_attempts: int = 10, initial_delay: float = 2.0, max_delay: float = 120.0):
    """Decorator for retry with exponential backoff - EXTRA PATIENT VERSION"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_attempts):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_attempts} for {func.__name__}")
                    return await func(*args, **kwargs)
                except (ResourceNotFoundError, ClientAuthenticationError):
                    # Don't retry these
                    logger.error(f"Non-retryable error in {func.__name__}: {type(last_exception).__name__}")
                    raise
                except asyncio.CancelledError:
                    logger.warning(f"Operation cancelled: {func.__name__}")
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}")
                        logger.info(f"Waiting {delay}s before retry...")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

class StorageService:
    """
    Ultra-reliable Azure Blob Storage service optimized for testing
    
    Features:
    - Maximum retry attempts with patient backoff
    - Extended timeouts for all operations
    - Small batch sizes to prevent overload
    - Detailed progress tracking
    - Comprehensive error recovery
    - Graceful degradation
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
        # Validate connection string
        if not self.settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
        
        # Service client (will be initialized on first use)
        self.blob_service_client: Optional[BlobServiceClient] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Container client cache
        self._container_clients: Dict[str, ContainerClient] = {}
        
        # Operation tracking
        self._stats = {
            "uploads": {"count": 0, "bytes": 0, "errors": 0},
            "downloads": {"count": 0, "bytes": 0, "errors": 0},
            "operations": {"total": 0, "failed": 0}
        }
        
        # Health monitoring
        self._health = {
            "is_healthy": True,
            "last_success": None,
            "consecutive_failures": 0,
            "last_error": None
        }
        
        # Configuration - OPTIMIZED FOR RELIABILITY
        self.config = {
            # Retry settings - VERY PATIENT
            "max_retries": 10,
            "retry_delay": 2.0,
            "max_retry_delay": 120.0,
            
            # Timeouts (seconds) - VERY GENEROUS
            "operation_timeout": 1800,  # 30 minutes per operation
            "download_timeout": 3600,   # 60 minutes for downloads
            "upload_timeout": 3600,     # 60 minutes for uploads
            
            # Chunk sizes - SMALLER FOR RELIABILITY
            "download_chunk_size": 4 * 1024 * 1024,   # 4MB chunks
            "upload_chunk_size": 4 * 1024 * 1024,     # 4MB chunks
            "max_single_put_size": 64 * 1024 * 1024,  # 64MB
            "max_single_get_size": 64 * 1024 * 1024,  # 64MB
            
            # Concurrency - CONSERVATIVE
            "max_concurrency": 3,
            "batch_size": 5,
            
            # Features
            "enable_compression": True,
            "compression_level": 6,
            
            # Progress tracking
            "progress_interval": 1.0,  # Report progress every second
            "enable_detailed_logging": True
        }
        
        logger.info("🚀 StorageService initialized (ULTRA-RELIABLE VERSION)")
        logger.info(f"   Max retries: {self.config['max_retries']}")
        logger.info(f"   Timeouts: {self.config['operation_timeout']}s per operation")
        logger.info(f"   Chunk sizes: {self.config['download_chunk_size'] / (1024*1024):.0f}MB")
        logger.info(f"   Max concurrency: {self.config['max_concurrency']}")

    # --- Initialization ---
    
    async def _ensure_initialized(self):
        """Lazy initialization of blob service client with retries"""
        if self._initialized:
            return
            
        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return
                
            max_init_attempts = 5
            for attempt in range(max_init_attempts):
                try:
                    logger.info(f"🔄 Initializing Azure Blob Storage client (attempt {attempt + 1}/{max_init_attempts})...")
                    
                    # Create client with conservative settings
                    self.blob_service_client = BlobServiceClient.from_connection_string(
                        self.settings.AZURE_STORAGE_CONNECTION_STRING,
                        max_single_get_size=self.config['max_single_get_size'],
                        max_chunk_get_size=self.config['download_chunk_size'],
                        max_single_put_size=self.config['max_single_put_size'],
                        max_block_size=self.config['upload_chunk_size'],
                        # Extended connection settings
                        connection_timeout=60,
                        read_timeout=600,
                        retry_total=self.config['max_retries']
                    )
                    
                    # Test connection with timeout
                    await asyncio.wait_for(self._test_connection(), timeout=30.0)
                    
                    self._initialized = True
                    logger.info("✅ Azure Blob Storage client initialized successfully")
                    break
                    
                except asyncio.TimeoutError:
                    logger.error(f"❌ Connection test timeout (attempt {attempt + 1})")
                    if attempt < max_init_attempts - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                except Exception as e:
                    logger.error(f"❌ Failed to initialize storage client (attempt {attempt + 1}): {e}")
                    self._health["is_healthy"] = False
                    self._health["last_error"] = str(e)
                    if attempt < max_init_attempts - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise ConnectionError(f"Storage initialization failed after {max_init_attempts} attempts: {e}")
    
    async def _test_connection(self):
        """Test storage connection with detailed logging"""
        try:
            logger.info("Testing Azure Storage connection...")
            account_info = await self.blob_service_client.get_account_information()
            account_type = account_info.get('account_kind', 'Unknown')
            
            logger.info(f"✅ Connected to Azure Storage")
            logger.info(f"   Account type: {account_type}")
            logger.info(f"   SKU: {account_info.get('sku_name', 'Unknown')}")
            
            self._record_success()
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            self._record_failure(e)
            raise

    async def _get_container_client(self, container_name: str) -> ContainerClient:
        """Get or create container client with initialization check"""
        await self._ensure_initialized()
        
        if not container_name:
            raise ValueError("Container name is required")
            
        # Cache container clients
        if container_name not in self._container_clients:
            logger.debug(f"Creating container client for: {container_name}")
            self._container_clients[container_name] = self.blob_service_client.get_container_client(container_name)
            
        return self._container_clients[container_name]

    # --- Health Monitoring ---
    
    def _record_success(self):
        """Record successful operation"""
        self._health["is_healthy"] = True
        self._health["last_success"] = datetime.utcnow()
        self._health["consecutive_failures"] = 0
        self._stats["operations"]["total"] += 1
        
        if self._health["consecutive_failures"] > 0:
            logger.info("✅ Service recovered after failures")
    
    def _record_failure(self, error: Exception):
        """Record failed operation with detailed logging"""
        self._health["consecutive_failures"] += 1
        self._health["last_error"] = str(error)
        self._stats["operations"]["failed"] += 1
        self._stats["operations"]["total"] += 1
        
        logger.error(f"Operation failure #{self._health['consecutive_failures']}: {error}")
        
        if self._health["consecutive_failures"] >= 5:
            self._health["is_healthy"] = False
            logger.error(f"⚠️ Storage service unhealthy after {self._health['consecutive_failures']} failures")

    # --- Validation ---
    
    def _validate_blob_name(self, blob_name: str) -> str:
        """Validate and sanitize blob name with detailed checks"""
        if not blob_name or not isinstance(blob_name, str):
            raise ValueError("Blob name must be a non-empty string")
        
        # Remove path traversal attempts
        original_name = blob_name
        blob_name = blob_name.replace("../", "").replace("..\\", "")
        blob_name = blob_name.strip().strip("/\\")
        
        if blob_name != original_name:
            logger.warning(f"Sanitized blob name: '{original_name}' -> '{blob_name}'")
        
        # Check length
        if len(blob_name) == 0:
            raise ValueError("Blob name cannot be empty after sanitization")
        if len(blob_name) > 1024:
            raise ValueError(f"Blob name too long: {len(blob_name)} chars (max 1024)")
        
        return blob_name

    def _determine_content_type(self, blob_name: str, content_type: Optional[str] = None) -> str:
        """Determine content type from extension with logging"""
        if content_type:
            logger.debug(f"Using provided content type: {content_type}")
            return content_type
            
        ext = os.path.splitext(blob_name.lower())[1]
        content_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.xml': 'application/xml',
            '.csv': 'text/csv',
            '.gz': 'application/gzip',
            '.zip': 'application/zip'
        }
        
        determined_type = content_types.get(ext, 'application/octet-stream')
        logger.debug(f"Determined content type for {ext}: {determined_type}")
        
        return determined_type

    # --- Core Operations ---
    
    @retry_with_backoff(max_attempts=10, initial_delay=2.0)
    async def upload_file(
        self, 
        container_name: str, 
        blob_name: str, 
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload file to blob storage with ultra-reliable retry logic
        
        Args:
            container_name: Target container
            blob_name: Name for the blob
            data: File content as bytes
            content_type: MIME type (auto-detected if not provided)
            metadata: Optional metadata dict
            
        Returns:
            Blob URL
        """
        start_time = time.time()
        blob_name = self._validate_blob_name(blob_name)
        
        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes")
        
        file_size_mb = len(data) / (1024 * 1024)
        logger.info(f"📤 Uploading {blob_name} ({file_size_mb:.1f}MB)")
        
        # Calculate dynamic timeout based on size
        base_timeout = 300  # 5 minutes base
        timeout_per_mb = 30  # 30 seconds per MB
        timeout = min(self.config['upload_timeout'], base_timeout + (file_size_mb * timeout_per_mb))
        
        logger.info(f"   Using timeout: {timeout}s for {file_size_mb:.1f}MB")
        
        try:
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Prepare content settings
            content_settings = ContentSettings(
                content_type=self._determine_content_type(blob_name, content_type)
            )
            
            # Upload with optimized settings
            upload_kwargs = {
                'data': data,
                'overwrite': True,
                'content_settings': content_settings,
                'metadata': metadata or {},
                'validate_content': False,  # Skip MD5 for performance
                'timeout': timeout
            }
            
            # Use minimal concurrency for reliability
            if file_size_mb > 10:
                upload_kwargs['max_concurrency'] = 2
                logger.info(f"   Using concurrent upload with 2 workers")
            
            # Progress tracking for large files
            if file_size_mb > 5:
                progress_task = asyncio.create_task(self._track_progress("Upload", file_size_mb, start_time))
            
            await blob_client.upload_blob(**upload_kwargs)
            
            if file_size_mb > 5 and 'progress_task' in locals():
                progress_task.cancel()
            
            elapsed = time.time() - start_time
            speed_mbps = (file_size_mb * 8) / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Uploaded {blob_name} in {elapsed:.1f}s @ {speed_mbps:.1f} Mbps")
            
            self._record_success()
            self._stats["uploads"]["count"] += 1
            self._stats["uploads"]["bytes"] += len(data)
            
            return blob_client.url
            
        except asyncio.TimeoutError:
            self._record_failure(TimeoutError(f"Upload timeout after {timeout}s"))
            self._stats["uploads"]["errors"] += 1
            raise TimeoutError(f"Upload of {blob_name} ({file_size_mb:.1f}MB) timed out after {timeout}s")
        except Exception as e:
            self._record_failure(e)
            self._stats["uploads"]["errors"] += 1
            logger.error(f"❌ Upload failed for {blob_name}: {e}")
            raise

    @retry_with_backoff(max_attempts=10, initial_delay=2.0)
    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """
        Download blob as bytes with ultra-reliable retry logic
        
        Args:
            container_name: Source container
            blob_name: Blob to download
            
        Returns:
            File content as bytes
        """
        start_time = time.time()
        blob_name = self._validate_blob_name(blob_name)
        
        logger.info(f"📥 Downloading {blob_name}")
        
        try:
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Get blob properties first to know size
            try:
                properties = await blob_client.get_blob_properties()
                blob_size_mb = properties.size / (1024 * 1024)
                logger.info(f"   Blob size: {blob_size_mb:.1f}MB")
            except:
                blob_size_mb = 0
                logger.warning("Could not get blob size")
            
            # Calculate dynamic timeout
            base_timeout = 300  # 5 minutes base
            timeout_per_mb = 30  # 30 seconds per MB
            timeout = min(self.config['download_timeout'], base_timeout + (blob_size_mb * timeout_per_mb))
            
            logger.info(f"   Using timeout: {timeout}s")
            
            # Progress tracking for large files
            if blob_size_mb > 5:
                progress_task = asyncio.create_task(self._track_progress("Download", blob_size_mb, start_time))
            
            # Download blob with minimal concurrency
            downloader = await blob_client.download_blob(max_concurrency=2, timeout=timeout)
            
            # Read content in chunks with progress
            chunks = []
            bytes_downloaded = 0
            
            async for chunk in downloader.chunks():
                chunks.append(chunk)
                bytes_downloaded += len(chunk)
                
                if blob_size_mb > 5 and bytes_downloaded % (1024 * 1024) == 0:
                    progress_mb = bytes_downloaded / (1024 * 1024)
                    logger.debug(f"   Downloaded: {progress_mb:.1f}/{blob_size_mb:.1f}MB")
            
            if blob_size_mb > 5 and 'progress_task' in locals():
                progress_task.cancel()
            
            data = b''.join(chunks)
            
            elapsed = time.time() - start_time
            actual_size_mb = len(data) / (1024 * 1024)
            speed_mbps = (actual_size_mb * 8) / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Downloaded {blob_name} ({actual_size_mb:.1f}MB) in {elapsed:.1f}s @ {speed_mbps:.1f} Mbps")
            
            self._record_success()
            self._stats["downloads"]["count"] += 1
            self._stats["downloads"]["bytes"] += len(data)
            
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob not found: {blob_name}")
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except asyncio.TimeoutError:
            self._record_failure(TimeoutError(f"Download timeout"))
            self._stats["downloads"]["errors"] += 1
            raise TimeoutError(f"Download of {blob_name} timed out after {timeout}s")
        except Exception as e:
            self._record_failure(e)
            self._stats["downloads"]["errors"] += 1
            logger.error(f"❌ Download failed for {blob_name}: {e}")
            raise

    async def _track_progress(self, operation: str, size_mb: float, start_time: float):
        """Track progress for long operations"""
        while True:
            await asyncio.sleep(self.config['progress_interval'])
            elapsed = time.time() - start_time
            logger.info(f"   {operation} in progress: {elapsed:.0f}s elapsed for {size_mb:.1f}MB")

    async def download_blob_as_text(self, container_name: str, blob_name: str, encoding: str = "utf-8") -> str:
        """Download blob as text with encoding detection"""
        logger.debug(f"Downloading text blob: {blob_name}")
        data = await self.download_blob_as_bytes(container_name, blob_name)
        
        # Try specified encoding first
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with {encoding}, trying alternatives")
        
        # Try common encodings
        for enc in ['utf-8-sig', 'latin-1', 'cp1252', 'utf-16']:
            try:
                logger.debug(f"Trying encoding: {enc}")
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        
        # Fallback with error handling
        logger.warning(f"Using lossy decoding for {blob_name}")
        return data.decode('utf-8', errors='ignore')

    async def download_blob_as_json(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Download and parse JSON blob with validation"""
        logger.debug(f"Downloading JSON blob: {blob_name}")
        text = await self.download_blob_as_text(container_name, blob_name)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {blob_name}: {e}")
            logger.debug(f"JSON content preview: {text[:200]}...")
            raise ValueError(f"Blob contains invalid JSON: {e}")

    @retry_with_backoff(max_attempts=5, initial_delay=1.0)
    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if blob exists with retries"""
        try:
            blob_name = self._validate_blob_name(blob_name)
            
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            exists = await blob_client.exists()
            logger.debug(f"Blob exists check: {blob_name} = {exists}")
            
            return exists
            
        except Exception as e:
            logger.debug(f"Error checking blob existence: {e}")
            return False

    @retry_with_backoff(max_attempts=5, initial_delay=1.0)
    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob with retries"""
        blob_name = self._validate_blob_name(blob_name)
        
        try:
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            await blob_client.delete_blob()
            
            logger.info(f"✅ Deleted {blob_name}")
            self._record_success()
            
            return True
            
        except ResourceNotFoundError:
            logger.debug(f"Blob {blob_name} already deleted or doesn't exist")
            return False
        except Exception as e:
            self._record_failure(e)
            logger.error(f"Failed to delete {blob_name}: {e}")
            raise

    @retry_with_backoff(max_attempts=5, initial_delay=1.0)
    async def list_blobs(
        self, 
        container_name: str, 
        prefix: str = "",
        suffix: str = "",
        max_results: Optional[int] = None
    ) -> List[str]:
        """List blobs with optional filtering and pagination"""
        try:
            container_client = await self._get_container_client(container_name)
            
            blob_names = []
            count = 0
            
            logger.info(f"Listing blobs in {container_name} with prefix='{prefix}', suffix='{suffix}'")
            
            # List blobs with pagination
            async for page in container_client.list_blobs(name_starts_with=prefix).by_page():
                async for blob in page:
                    blob_name = blob.name
                    
                    # Apply suffix filter
                    if suffix and not blob_name.endswith(suffix):
                        continue
                    
                    blob_names.append(blob_name)
                    count += 1
                    
                    if max_results and count >= max_results:
                        logger.info(f"Reached max results: {max_results}")
                        break
                
                if max_results and count >= max_results:
                    break
                
                # Small delay between pages to be nice to Azure
                await asyncio.sleep(0.1)
            
            blob_names.sort()
            logger.info(f"Found {len(blob_names)} blobs")
            self._record_success()
            
            return blob_names
            
        except Exception as e:
            self._record_failure(e)
            logger.error(f"Failed to list blobs: {e}")
            raise

    # --- Batch Operations ---
    
    async def batch_download_blobs(
        self,
        container_name: str,
        blob_names: List[str],
        max_concurrent: int = 3
    ) -> Dict[str, bytes]:
        """Download multiple blobs concurrently with conservative settings"""
        if not blob_names:
            return {}
        
        # Validate names
        valid_names = []
        for name in blob_names:
            try:
                valid_names.append(self._validate_blob_name(name))
            except ValueError:
                logger.warning(f"Skipping invalid blob name: {name}")
        
        if not valid_names:
            return {}
        
        # Limit concurrency for reliability
        max_concurrent = min(max_concurrent, len(valid_names), self.config['max_concurrency'])
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"📥 Batch downloading {len(valid_names)} blobs with {max_concurrent} workers")
        
        async def download_one(blob_name: str) -> Tuple[str, Optional[bytes]]:
            async with semaphore:
                try:
                    logger.debug(f"Downloading: {blob_name}")
                    data = await self.download_blob_as_bytes(container_name, blob_name)
                    return (blob_name, data)
                except Exception as e:
                    logger.error(f"Failed to download {blob_name}: {e}")
                    return (blob_name, None)
        
        # Download all with progress tracking
        start_time = time.time()
        
        # Process in smaller batches for better progress tracking
        batch_size = self.config['batch_size']
        downloaded = {}
        failed = []
        
        for i in range(0, len(valid_names), batch_size):
            batch = valid_names[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_names) + batch_size - 1)//batch_size}")
            
            results = await asyncio.gather(*[download_one(name) for name in batch])
            
            for blob_name, data in results:
                if data is not None:
                    downloaded[blob_name] = data
                else:
                    failed.append(blob_name)
            
            # Small delay between batches
            if i + batch_size < len(valid_names):
                await asyncio.sleep(0.5)
        
        elapsed = time.time() - start_time
        total_bytes = sum(len(data) for data in downloaded.values())
        
        logger.info(f"✅ Batch download complete: {len(downloaded)}/{len(valid_names)} successful")
        logger.info(f"   {total_bytes / (1024*1024):.1f}MB in {elapsed:.1f}s")
        
        if failed:
            logger.warning(f"   Failed: {len(failed)} blobs - {failed[:5]}...")
        
        return downloaded

    async def batch_upload_blobs(
        self,
        container_name: str,
        blobs: Dict[str, bytes],
        max_concurrent: int = 3
    ) -> Dict[str, bool]:
        """Upload multiple blobs concurrently with conservative settings"""
        if not blobs:
            return {}
        
        # Validate
        valid_blobs = {}
        for name, data in blobs.items():
            try:
                valid_name = self._validate_blob_name(name)
                if isinstance(data, bytes):
                    valid_blobs[valid_name] = data
                else:
                    logger.warning(f"Skipping non-bytes data for: {name}")
            except ValueError:
                logger.warning(f"Skipping invalid blob: {name}")
        
        if not valid_blobs:
            return {}
        
        # Limit concurrency
        max_concurrent = min(max_concurrent, len(valid_blobs), self.config['max_concurrency'])
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"📤 Batch uploading {len(valid_blobs)} blobs with {max_concurrent} workers")
        
        async def upload_one(blob_name: str, data: bytes) -> Tuple[str, bool]:
            async with semaphore:
                try:
                    logger.debug(f"Uploading: {blob_name} ({len(data)} bytes)")
                    await self.upload_file(container_name, blob_name, data)
                    return (blob_name, True)
                except Exception as e:
                    logger.error(f"Failed to upload {blob_name}: {e}")
                    return (blob_name, False)
        
        # Upload in batches
        start_time = time.time()
        batch_size = self.config['batch_size']
        upload_results = {}
        
        items = list(valid_blobs.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
            
            results = await asyncio.gather(*[
                upload_one(name, data) 
                for name, data in batch
            ])
            
            upload_results.update(dict(results))
            
            # Small delay between batches
            if i + batch_size < len(items):
                await asyncio.sleep(0.5)
        
        successful = sum(1 for success in upload_results.values() if success)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Batch upload complete: {successful}/{len(valid_blobs)} successful in {elapsed:.1f}s")
        
        return upload_results

    # --- Utility Operations ---
    
    async def list_document_ids(self, container_name: str) -> List[str]:
        """List all document IDs by finding _context.txt files"""
        logger.info("Listing all document IDs...")
        
        context_files = await self.list_blobs(
            container_name=container_name,
            suffix="_context.txt"
        )
        
        document_ids = []
        for filename in context_files:
            if filename.endswith("_context.txt"):
                doc_id = filename[:-12]  # Remove suffix
                document_ids.append(doc_id)
        
        unique_ids = sorted(list(set(document_ids)))
        logger.info(f"Found {len(unique_ids)} unique document IDs")
        
        return unique_ids

    async def delete_document_files(self, document_id: str) -> Dict[str, int]:
        """Delete all files for a document with detailed tracking"""
        if not re.match(r'^[\w\-]+$', document_id):
            raise ValueError("Invalid document ID format")
        
        logger.info(f"Deleting all files for document: {document_id}")
        
        counts = {
            "pdfs": 0,
            "pages": 0,
            "metadata": 0,
            "total": 0
        }
        
        # Delete from main container
        if hasattr(self.settings, 'AZURE_CONTAINER_NAME'):
            try:
                if await self.delete_blob(self.settings.AZURE_CONTAINER_NAME, f"{document_id}.pdf"):
                    counts["pdfs"] = 1
                    logger.info(f"Deleted main PDF: {document_id}.pdf")
            except Exception as e:
                logger.warning(f"Could not delete PDF: {e}")
        
        # Delete from cache container
        if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME'):
            # List all related files
            files = await self.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=f"{document_id}_"
            )
            
            logger.info(f"Found {len(files)} cache files to delete")
            
            # Delete in small batches
            batch_size = 5
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                logger.debug(f"Deleting batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
                
                delete_tasks = [
                    self.delete_blob(self.settings.AZURE_CACHE_CONTAINER_NAME, f)
                    for f in batch
                ]
                
                results = await asyncio.gather(*delete_tasks, return_exceptions=True)
                
                for filename, result in zip(batch, results):
                    if result is True:
                        counts["total"] += 1
                        
                        # Categorize
                        if "_page_" in filename and filename.endswith(".jpg"):
                            counts["pages"] += 1
                        elif filename.endswith((".json", ".txt", ".gz")):
                            counts["metadata"] += 1
                
                # Delay between batches
                if i + batch_size < len(files):
                    await asyncio.sleep(0.2)
        
        counts["total"] += counts["pdfs"]
        
        logger.info(f"✅ Deleted {counts['total']} files for document {document_id}")
        logger.info(f"   PDFs: {counts['pdfs']}, Pages: {counts['pages']}, Metadata: {counts['metadata']}")
        
        return counts

    # --- Status and Health ---
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        return {
            "service": "Azure Blob Storage",
            "version": "Ultra-Reliable",
            "initialized": self._initialized,
            "health": self._health,
            "statistics": self._stats,
            "configuration": {
                "max_retries": self.config['max_retries'],
                "timeouts": {
                    "operation": self.config['operation_timeout'],
                    "download": self.config['download_timeout'],
                    "upload": self.config['upload_timeout']
                },
                "chunk_sizes": {
                    "download_mb": self.config['download_chunk_size'] / (1024*1024),
                    "upload_mb": self.config['upload_chunk_size'] / (1024*1024)
                },
                "concurrency": {
                    "max": self.config['max_concurrency'],
                    "batch_size": self.config['batch_size']
                }
            },
            "containers": {
                "main": getattr(self.settings, 'AZURE_CONTAINER_NAME', None),
                "cache": getattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME', None)
            }
        }

    async def verify_connection(self):
        """Verify storage connection and container access with retries"""
        logger.info("🔍 Verifying storage connection...")
        
        await self._ensure_initialized()
        
        # Test containers
        containers = []
        if hasattr(self.settings, 'AZURE_CONTAINER_NAME'):
            containers.append(self.settings.AZURE_CONTAINER_NAME)
        if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME'):
            containers.append(self.settings.AZURE_CACHE_CONTAINER_NAME)
        
        for container_name in set(containers):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    container_client = await self._get_container_client(container_name)
                    props = await container_client.get_container_properties()
                    logger.info(f"✅ Container '{container_name}' accessible")
                    logger.debug(f"   Last modified: {props.get('last_modified', 'Unknown')}")
                    break
                except ResourceNotFoundError:
                    logger.warning(f"⚠️ Container '{container_name}' not found")
                    # Optionally create it
                    try:
                        await container_client.create_container()
                        logger.info(f"✅ Created container '{container_name}'")
                        break
                    except ResourceExistsError:
                        logger.info(f"Container '{container_name}' was created by another process")
                        break
                    except Exception as e:
                        logger.error(f"Failed to create container: {e}")
                except Exception as e:
                    logger.error(f"❌ Cannot access container '{container_name}' (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2 * (attempt + 1))
        
        logger.info("✅ Storage verification complete")

    # --- Context Manager ---
    
    async def __aenter__(self):
        """Async context manager entry with verification"""
        await self.verify_connection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        try:
            # Log any exception that occurred
            if exc_type:
                logger.error(f"Exception in storage context: {exc_type.__name__}: {exc_val}")
            
            # Clear caches
            self._container_clients.clear()
            
            # Close client
            if self.blob_service_client:
                await self.blob_service_client.close()
            
            logger.info("✅ Storage service closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# --- Singleton Management ---

_instance: Optional[StorageService] = None
_lock = asyncio.Lock()

async def get_storage_service() -> StorageService:
    """Get or create storage service instance"""
    global _instance
    
    async with _lock:
        if _instance is None:
            logger.info("Creating new StorageService instance...")
            _instance = StorageService()
            await _instance.verify_connection()
    
    return _instance

def get_storage_service_sync() -> StorageService:
    """Get storage service for dependency injection (sync version)"""
    global _instance
    
    if _instance is None:
        logger.info("Creating new StorageService instance (sync)...")
        _instance = StorageService()
    
    return _instance

def reset_storage_service():
    """Reset storage service (for testing)"""
    global _instance
    logger.info("Resetting StorageService instance")
    _instance = None

# --- Exports ---

__all__ = [
    'StorageService',
    'get_storage_service',
    'get_storage_service_sync',
    'reset_storage_service'
]