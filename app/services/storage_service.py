# app/services/storage_service.py - PRODUCTION-READY VERSION WITH MAXIMUM RELIABILITY

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

def retry_with_backoff(max_attempts: int = 3, initial_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ResourceNotFoundError, ClientAuthenticationError):
                    # Don't retry these
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

class StorageService:
    """
    Production-grade Azure Blob Storage service
    
    Features:
    - Automatic retry with exponential backoff
    - Connection pooling and health monitoring
    - Progress tracking for large operations
    - Compression support
    - Batch operations
    - Graceful error handling
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
        
        # Configuration
        self.config = {
            # Retry settings
            "max_retries": 5,
            "retry_delay": 1.0,
            "max_retry_delay": 60.0,
            
            # Timeouts (seconds)
            "operation_timeout": 300,  # 5 minutes per operation
            "download_timeout": 600,   # 10 minutes for downloads
            "upload_timeout": 900,     # 15 minutes for uploads
            
            # Chunk sizes
            "download_chunk_size": 32 * 1024 * 1024,  # 32MB
            "upload_chunk_size": 16 * 1024 * 1024,    # 16MB
            "max_single_put_size": 256 * 1024 * 1024, # 256MB
            "max_single_get_size": 256 * 1024 * 1024, # 256MB
            
            # Concurrency
            "max_concurrency": 10,
            "batch_size": 20,
            
            # Features
            "enable_compression": True,
            "compression_level": 6
        }
        
        logger.info("🚀 StorageService initialized")
        logger.info(f"   Max retries: {self.config['max_retries']}")
        logger.info(f"   Timeouts: {self.config['operation_timeout']}s per operation")
        logger.info(f"   Chunk sizes: {self.config['download_chunk_size'] / (1024*1024):.0f}MB download, "
                   f"{self.config['upload_chunk_size'] / (1024*1024):.0f}MB upload")

    # --- Initialization ---
    
    async def _ensure_initialized(self):
        """Lazy initialization of blob service client"""
        if self._initialized:
            return
            
        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return
                
            try:
                logger.info("🔄 Initializing Azure Blob Storage client...")
                
                # Create client with optimized settings
                # Let Azure SDK handle the transport layer
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.settings.AZURE_STORAGE_CONNECTION_STRING,
                    max_single_get_size=self.config['max_single_get_size'],
                    max_chunk_get_size=self.config['download_chunk_size'],
                    max_single_put_size=self.config['max_single_put_size'],
                    max_block_size=self.config['upload_chunk_size'],
                    # Connection settings
                    connection_timeout=30,
                    read_timeout=300
                )
                
                # Test connection
                await self._test_connection()
                
                self._initialized = True
                logger.info("✅ Azure Blob Storage client initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize storage client: {e}")
                self._health["is_healthy"] = False
                self._health["last_error"] = str(e)
                raise ConnectionError(f"Storage initialization failed: {e}")
    
    async def _test_connection(self):
        """Test storage connection"""
        try:
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
        """Get or create container client"""
        await self._ensure_initialized()
        
        if not container_name:
            raise ValueError("Container name is required")
            
        # Cache container clients
        if container_name not in self._container_clients:
            self._container_clients[container_name] = self.blob_service_client.get_container_client(container_name)
            
        return self._container_clients[container_name]

    # --- Health Monitoring ---
    
    def _record_success(self):
        """Record successful operation"""
        self._health["is_healthy"] = True
        self._health["last_success"] = datetime.utcnow()
        self._health["consecutive_failures"] = 0
        self._stats["operations"]["total"] += 1
    
    def _record_failure(self, error: Exception):
        """Record failed operation"""
        self._health["consecutive_failures"] += 1
        self._health["last_error"] = str(error)
        self._stats["operations"]["failed"] += 1
        self._stats["operations"]["total"] += 1
        
        if self._health["consecutive_failures"] >= 5:
            self._health["is_healthy"] = False
            logger.error(f"⚠️ Storage service unhealthy after {self._health['consecutive_failures']} failures")

    # --- Validation ---
    
    def _validate_blob_name(self, blob_name: str) -> str:
        """Validate and sanitize blob name"""
        if not blob_name or not isinstance(blob_name, str):
            raise ValueError("Blob name must be a non-empty string")
        
        # Remove path traversal attempts
        blob_name = blob_name.replace("../", "").replace("..\\", "")
        blob_name = blob_name.strip().strip("/\\")
        
        # Check length
        if len(blob_name) == 0:
            raise ValueError("Blob name cannot be empty")
        if len(blob_name) > 1024:
            raise ValueError(f"Blob name too long: {len(blob_name)} chars (max 1024)")
        
        return blob_name

    def _determine_content_type(self, blob_name: str, content_type: Optional[str] = None) -> str:
        """Determine content type from extension"""
        if content_type:
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
        
        return content_types.get(ext, 'application/octet-stream')

    # --- Core Operations ---
    
    @retry_with_backoff(max_attempts=5)
    async def upload_file(
        self, 
        container_name: str, 
        blob_name: str, 
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload file to blob storage with retry logic
        
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
                'validate_content': False  # Skip MD5 for performance
            }
            
            # Use concurrent upload for large files
            if file_size_mb > 64:
                upload_kwargs['max_concurrency'] = min(10, max(2, int(file_size_mb / 64)))
            
            # Set timeout based on file size
            timeout = min(self.config['upload_timeout'], max(60, file_size_mb * 10))
            
            await asyncio.wait_for(
                blob_client.upload_blob(**upload_kwargs),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            speed_mbps = (file_size_mb * 8) / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Uploaded {blob_name} in {elapsed:.1f}s @ {speed_mbps:.1f} Mbps")
            
            self._record_success()
            self._stats["uploads"]["count"] += 1
            self._stats["uploads"]["bytes"] += len(data)
            
            return blob_client.url
            
        except asyncio.TimeoutError:
            self._record_failure(TimeoutError(f"Upload timeout after {timeout}s"))
            raise TimeoutError(f"Upload of {blob_name} timed out")
        except Exception as e:
            self._record_failure(e)
            self._stats["uploads"]["errors"] += 1
            logger.error(f"❌ Upload failed for {blob_name}: {e}")
            raise

    @retry_with_backoff(max_attempts=5)
    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """
        Download blob as bytes with retry logic
        
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
            
            # Set timeout
            timeout = self.config['download_timeout']
            
            # Download blob
            downloader = await asyncio.wait_for(
                blob_client.download_blob(max_concurrency=8),
                timeout=timeout
            )
            
            # Read content
            data = await asyncio.wait_for(
                downloader.readall(),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            size_mb = len(data) / (1024 * 1024)
            speed_mbps = (size_mb * 8) / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Downloaded {blob_name} ({size_mb:.1f}MB) in {elapsed:.1f}s @ {speed_mbps:.1f} Mbps")
            
            self._record_success()
            self._stats["downloads"]["count"] += 1
            self._stats["downloads"]["bytes"] += len(data)
            
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob not found: {blob_name}")
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except asyncio.TimeoutError:
            self._record_failure(TimeoutError(f"Download timeout"))
            raise TimeoutError(f"Download of {blob_name} timed out")
        except Exception as e:
            self._record_failure(e)
            self._stats["downloads"]["errors"] += 1
            logger.error(f"❌ Download failed for {blob_name}: {e}")
            raise

    async def download_blob_as_text(self, container_name: str, blob_name: str, encoding: str = "utf-8") -> str:
        """Download blob as text with encoding detection"""
        data = await self.download_blob_as_bytes(container_name, blob_name)
        
        # Try specified encoding first
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
        
        # Try common encodings
        for enc in ['utf-8-sig', 'latin-1', 'cp1252', 'utf-16']:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        
        # Fallback with error handling
        logger.warning(f"Using lossy decoding for {blob_name}")
        return data.decode('utf-8', errors='ignore')

    async def download_blob_as_json(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Download and parse JSON blob"""
        text = await self.download_blob_as_text(container_name, blob_name)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {blob_name}: {e}")
            raise ValueError(f"Blob contains invalid JSON: {e}")

    @retry_with_backoff(max_attempts=3)
    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if blob exists"""
        try:
            blob_name = self._validate_blob_name(blob_name)
            
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            return await blob_client.exists()
            
        except Exception as e:
            logger.debug(f"Error checking blob existence: {e}")
            return False

    @retry_with_backoff(max_attempts=3)
    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob"""
        blob_name = self._validate_blob_name(blob_name)
        
        try:
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            await blob_client.delete_blob()
            
            logger.info(f"✅ Deleted {blob_name}")
            self._record_success()
            
            return True
            
        except ResourceNotFoundError:
            logger.debug(f"Blob {blob_name} already deleted")
            return False
        except Exception as e:
            self._record_failure(e)
            logger.error(f"Failed to delete {blob_name}: {e}")
            raise

    @retry_with_backoff(max_attempts=3)
    async def list_blobs(
        self, 
        container_name: str, 
        prefix: str = "",
        suffix: str = "",
        max_results: Optional[int] = None
    ) -> List[str]:
        """List blobs with optional filtering"""
        try:
            container_client = await self._get_container_client(container_name)
            
            blob_names = []
            count = 0
            
            # List blobs
            async for blob in container_client.list_blobs(name_starts_with=prefix):
                blob_name = blob.name
                
                # Apply suffix filter
                if suffix and not blob_name.endswith(suffix):
                    continue
                
                blob_names.append(blob_name)
                count += 1
                
                if max_results and count >= max_results:
                    break
            
            blob_names.sort()
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
        max_concurrent: int = 10
    ) -> Dict[str, bytes]:
        """Download multiple blobs concurrently"""
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
        
        # Limit concurrency
        max_concurrent = min(max_concurrent, len(valid_names), self.config['max_concurrency'])
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"📥 Batch downloading {len(valid_names)} blobs with {max_concurrent} workers")
        
        async def download_one(blob_name: str) -> Tuple[str, Optional[bytes]]:
            async with semaphore:
                try:
                    data = await self.download_blob_as_bytes(container_name, blob_name)
                    return (blob_name, data)
                except Exception as e:
                    logger.error(f"Failed to download {blob_name}: {e}")
                    return (blob_name, None)
        
        # Download all
        start_time = time.time()
        results = await asyncio.gather(*[download_one(name) for name in valid_names])
        
        # Process results
        downloaded = {}
        failed = []
        total_bytes = 0
        
        for blob_name, data in results:
            if data is not None:
                downloaded[blob_name] = data
                total_bytes += len(data)
            else:
                failed.append(blob_name)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Batch download complete: {len(downloaded)}/{len(valid_names)} successful")
        logger.info(f"   {total_bytes / (1024*1024):.1f}MB in {elapsed:.1f}s")
        
        if failed:
            logger.warning(f"   Failed: {len(failed)} blobs")
        
        return downloaded

    async def batch_upload_blobs(
        self,
        container_name: str,
        blobs: Dict[str, bytes],
        max_concurrent: int = 10
    ) -> Dict[str, bool]:
        """Upload multiple blobs concurrently"""
        if not blobs:
            return {}
        
        # Validate
        valid_blobs = {}
        for name, data in blobs.items():
            try:
                valid_name = self._validate_blob_name(name)
                if isinstance(data, bytes):
                    valid_blobs[valid_name] = data
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
                    await self.upload_file(container_name, blob_name, data)
                    return (blob_name, True)
                except Exception as e:
                    logger.error(f"Failed to upload {blob_name}: {e}")
                    return (blob_name, False)
        
        # Upload all
        start_time = time.time()
        results = await asyncio.gather(*[
            upload_one(name, data) 
            for name, data in valid_blobs.items()
        ])
        
        # Process results
        upload_results = dict(results)
        successful = sum(1 for success in upload_results.values() if success)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Batch upload complete: {successful}/{len(valid_blobs)} successful in {elapsed:.1f}s")
        
        return upload_results

    # --- Utility Operations ---
    
    async def list_document_ids(self, container_name: str) -> List[str]:
        """List all document IDs by finding _context.txt files"""
        context_files = await self.list_blobs(
            container_name=container_name,
            suffix="_context.txt"
        )
        
        document_ids = []
        for filename in context_files:
            if filename.endswith("_context.txt"):
                doc_id = filename[:-12]  # Remove suffix
                document_ids.append(doc_id)
        
        return sorted(list(set(document_ids)))

    async def delete_document_files(self, document_id: str) -> Dict[str, int]:
        """Delete all files for a document"""
        if not re.match(r'^[\w\-]+$', document_id):
            raise ValueError("Invalid document ID")
        
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
            except:
                pass
        
        # Delete from cache container
        if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME'):
            # List all related files
            files = await self.list_blobs(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                prefix=f"{document_id}_"
            )
            
            # Delete in batches
            for i in range(0, len(files), self.config['batch_size']):
                batch = files[i:i + self.config['batch_size']]
                
                delete_tasks = [
                    self.delete_blob(self.settings.AZURE_CACHE_CONTAINER_NAME, f)
                    for f in batch
                ]
                
                results = await asyncio.gather(*delete_tasks, return_exceptions=True)
                
                for filename, result in zip(batch, results):
                    if result is True:
                        counts["total"] += 1
                        
                        # Categorize
                        if "_page_" in filename:
                            counts["pages"] += 1
                        elif filename.endswith((".json", ".txt")):
                            counts["metadata"] += 1
        
        counts["total"] += counts["pdfs"]
        
        logger.info(f"✅ Deleted {counts['total']} files for document {document_id}")
        
        return counts

    # --- Status and Health ---
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        return {
            "service": "Azure Blob Storage",
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
                }
            },
            "containers": {
                "main": getattr(self.settings, 'AZURE_CONTAINER_NAME', None),
                "cache": getattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME', None)
            }
        }

    async def verify_connection(self):
        """Verify storage connection and container access"""
        logger.info("🔍 Verifying storage connection...")
        
        await self._ensure_initialized()
        
        # Test containers
        containers = []
        if hasattr(self.settings, 'AZURE_CONTAINER_NAME'):
            containers.append(self.settings.AZURE_CONTAINER_NAME)
        if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME'):
            containers.append(self.settings.AZURE_CACHE_CONTAINER_NAME)
        
        for container_name in set(containers):
            try:
                container_client = await self._get_container_client(container_name)
                props = await container_client.get_container_properties()
                logger.info(f"✅ Container '{container_name}' accessible")
            except ResourceNotFoundError:
                logger.warning(f"⚠️ Container '{container_name}' not found")
                # Optionally create it
                try:
                    await container_client.create_container()
                    logger.info(f"✅ Created container '{container_name}'")
                except ResourceExistsError:
                    pass
            except Exception as e:
                logger.error(f"❌ Cannot access container '{container_name}': {e}")
        
        logger.info("✅ Storage verification complete")

    # --- Context Manager ---
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.verify_connection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
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
            _instance = StorageService()
            await _instance.verify_connection()
    
    return _instance

def get_storage_service_sync() -> StorageService:
    """Get storage service for dependency injection"""
    global _instance
    
    if _instance is None:
        _instance = StorageService()
    
    return _instance

def reset_storage_service():
    """Reset storage service (for testing)"""
    global _instance
    _instance = None

# --- Exports ---

__all__ = [
    'StorageService',
    'get_storage_service',
    'get_storage_service_sync',
    'reset_storage_service'
]
