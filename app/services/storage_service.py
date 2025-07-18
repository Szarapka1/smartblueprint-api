# app/services/storage_service.py - ENHANCED PRODUCTION VERSION

import logging
import asyncio
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import time
from app.core.config import get_settings, CONFIG
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, AzureError
from azure.storage.blob import BlobPrefix, ContentSettings
import re

logger = logging.getLogger(__name__)

class StorageService:
    """
    Enhanced Azure Blob Storage service optimized for blueprint analysis pipeline
    
    Key improvements:
    - No container validation - works with any container
    - Better error handling and logging
    - Retry logic with exponential backoff
    - Connection pooling and reuse
    - Optimized for thumbnail -> selection -> full image workflow
    - Better debugging information
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
        if not self.settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
        
        self._lock = asyncio.Lock()
        self._container_clients: Dict[str, ContainerClient] = {}  # Container client cache
        self._last_error_time = 0
        self._error_count = 0
        
        # Check if extended timeouts should be used (from env or config)
        self.use_extended_timeouts = (
            getattr(settings, 'USE_EXTENDED_TIMEOUTS', None) or 
            CONFIG.get('USE_EXTENDED_TIMEOUTS', True)  # Default to True for testing
        )
        
        if self.use_extended_timeouts:
            # EXTENDED TIMEOUTS FOR TESTING
            self.retry_config = {
                "max_retries": 5,           # Increased from 3
                "initial_delay": 1.0,       # Increased from 0.5
                "max_delay": 30,            # Increased from 10
                "backoff_factor": 2
            }
            
            self.timeout_config = {
                "connection_timeout": 300,   # 5 minutes (was 30s)
                "read_timeout": 3600,        # 1 hour (was 5min)
                "write_timeout": 3600,       # 1 hour
                "operation_timeout": 1800,   # 30 minutes per operation
                "total_timeout": 7200        # 2 hours total
            }
            
            logger.warning("⏰ EXTENDED TIMEOUTS ENABLED FOR TESTING:")
            logger.warning(f"  - Connection: {self.timeout_config['connection_timeout']}s (5 min)")
            logger.warning(f"  - Read: {self.timeout_config['read_timeout']}s (1 hour)")
            logger.warning(f"  - Operation: {self.timeout_config['operation_timeout']}s (30 min)")
            logger.warning(f"  - Total: {self.timeout_config['total_timeout']}s (2 hours)")
            logger.warning(f"  - Retries: {self.retry_config['max_retries']} attempts")
        else:
            # PRODUCTION TIMEOUTS
            self.retry_config = {
                "max_retries": 3,
                "initial_delay": 0.5,
                "max_delay": 10,
                "backoff_factor": 2
            }
            
            self.timeout_config = {
                "connection_timeout": 30,
                "read_timeout": 300,
                "write_timeout": 300,
                "operation_timeout": 180,
                "total_timeout": 600
            }
            
            logger.info("⏰ Using production timeouts")
        
        try:
            # Create blob service client with EXTENDED timeout settings
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.settings.AZURE_STORAGE_CONNECTION_STRING,
                max_single_get_size=128*1024*1024,     # 128MB chunks (increased)
                max_chunk_get_size=16*1024*1024,       # 16MB streaming chunks (increased)
                max_single_put_size=256*1024*1024,     # 256MB single upload (increased)
                max_block_size=16*1024*1024,           # 16MB blocks (increased)
                connection_timeout=self.timeout_config['connection_timeout'],
                read_timeout=self.timeout_config['read_timeout']
            )
            
            logger.info("✅ StorageService initialized successfully")
            
            # Test connection by getting account info
            asyncio.create_task(self._test_connection())
            
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Azure Blob Storage client: {e}")
            raise ValueError(f"Azure Storage initialization failed: {e}")

    async def _test_connection(self):
        """Test connection in background with extended timeout"""
        try:
            # Extended timeout for initial connection test
            account_info = await asyncio.wait_for(
                self.blob_service_client.get_account_information(),
                timeout=60.0  # 1 minute for connection test
            )
            logger.info(f"✅ Connected to Azure Storage: {account_info.get('sku_name', 'Unknown')} tier")
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Connection test timed out after 60s")
        except Exception as e:
            logger.warning(f"⚠️ Connection test failed: {e}")

    async def _get_container_client(self, container_name: str) -> ContainerClient:
        """Get or create a cached container client"""
        if container_name not in self._container_clients:
            self._container_clients[container_name] = self.blob_service_client.get_container_client(container_name)
            logger.debug(f"📦 Created container client for: {container_name}")
        return self._container_clients[container_name]

    def _validate_blob_name(self, blob_name: str) -> str:
        """Validate and sanitize blob name"""
        if not blob_name or not isinstance(blob_name, str):
            raise ValueError("Blob name must be a non-empty string")
        
        # Remove any path traversal attempts
        blob_name = blob_name.replace("../", "").replace("..\\", "")
        
        # Azure blob names are case-sensitive but let's log if there might be issues
        if blob_name != blob_name.lower():
            logger.debug(f"📝 Note: Blob name contains uppercase: {blob_name}")
        
        # Ensure valid characters
        if not re.match(r'^[\w\-./]+$', blob_name):
            raise ValueError(f"Blob name contains invalid characters: {blob_name}")
        
        # Strip leading/trailing slashes
        blob_name = blob_name.strip("/")
        
        if not blob_name:
            raise ValueError("Blob name cannot be empty after sanitization")
            
        return blob_name

    async def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with retry logic and extended timeouts"""
        last_error = None
        
        for attempt in range(self.retry_config["max_retries"]):
            try:
                # Wrap operation with extended timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.timeout_config["operation_timeout"]
                )
                
                # Reset error count on success
                if self._error_count > 0:
                    self._error_count = 0
                    logger.info("✅ Connection restored after errors")
                
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Operation timed out after {self.timeout_config['operation_timeout']}s")
                logger.error(f"⏰ Timeout on attempt {attempt + 1}/{self.retry_config['max_retries']}")
                self._error_count += 1
                
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = min(
                        self.retry_config["initial_delay"] * (self.retry_config["backoff_factor"] ** attempt),
                        self.retry_config["max_delay"]
                    )
                    logger.warning(f"⏰ Operation timeout. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                
            except ResourceNotFoundError:
                # Don't retry if resource doesn't exist
                raise
                
            except Exception as e:
                last_error = e
                self._error_count += 1
                
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = min(
                        self.retry_config["initial_delay"] * (self.retry_config["backoff_factor"] ** attempt),
                        self.retry_config["max_delay"]
                    )
                    
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ All {self.retry_config['max_retries']} attempts failed")
        
        raise last_error

    async def verify_connection(self):
        """Verify connection and test container access"""
        async with self._lock:
            logger.info("🔍 Verifying Azure Storage connection...")
            
            try:
                # Test basic connectivity with extended timeout
                account_info = await asyncio.wait_for(
                    self.blob_service_client.get_account_information(),
                    timeout=120.0  # 2 minutes for account info
                )
                logger.info(f"✅ Connected to account: {account_info.get('account_kind', 'Unknown')}")
                
                # Test container access if configured
                test_containers = []
                if hasattr(self.settings, 'AZURE_CONTAINER_NAME') and self.settings.AZURE_CONTAINER_NAME:
                    test_containers.append(self.settings.AZURE_CONTAINER_NAME)
                if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME') and self.settings.AZURE_CACHE_CONTAINER_NAME:
                    test_containers.append(self.settings.AZURE_CACHE_CONTAINER_NAME)
                
                for container_name in test_containers:
                    try:
                        container_client = await self._get_container_client(container_name)
                        properties = await asyncio.wait_for(
                            container_client.get_container_properties(),
                            timeout=60.0  # 1 minute per container
                        )
                        logger.info(f"✅ Container '{container_name}' accessible (Last modified: {properties.get('last_modified', 'Unknown')})")
                    except ResourceNotFoundError:
                        logger.warning(f"⚠️ Container '{container_name}' not found - will be created if needed")
                    except Exception as e:
                        logger.error(f"❌ Cannot access container '{container_name}': {e}")
                
                logger.info("✅ Storage connection verified")
                
            except Exception as e:
                logger.error(f"❌ Connection verification failed: {e}")
                raise ConnectionError(f"Azure Storage connection failed: {e}")

    async def upload_file(self, container_name: str, blob_name: str, data: bytes, 
                         content_type: Optional[str] = None) -> str:
        """Upload data to Azure Blob Storage"""
        start_time = time.time()
        blob_name = self._validate_blob_name(blob_name)
        
        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes")
        
        if len(data) == 0:
            raise ValueError("Cannot upload empty data")
        
        try:
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Prepare upload options
            upload_options = {
                'data': data,
                'overwrite': True,
                'validate_content': len(data) < 10 * 1024 * 1024 if self.use_extended_timeouts else len(data) < 4 * 1024 * 1024  # Only validate smaller files
            }
            
            # Set content type
            if content_type:
                upload_options['content_settings'] = ContentSettings(content_type=content_type)
            else:
                # Auto-detect content type
                if blob_name.lower().endswith('.png'):
                    upload_options['content_settings'] = ContentSettings(content_type='image/png')
                elif blob_name.lower().endswith(('.jpg', '.jpeg')):
                    upload_options['content_settings'] = ContentSettings(content_type='image/jpeg')
                elif blob_name.lower().endswith('.pdf'):
                    upload_options['content_settings'] = ContentSettings(content_type='application/pdf')
                elif blob_name.lower().endswith('.json'):
                    upload_options['content_settings'] = ContentSettings(content_type='application/json')
                elif blob_name.lower().endswith('.txt'):
                    upload_options['content_settings'] = ContentSettings(content_type='text/plain')
            
            # Optimize for large files - INCREASED THRESHOLDS
            file_size_mb = len(data) / (1024 * 1024)
            if file_size_mb > 10:  # Increased from 4MB
                upload_options['max_concurrency'] = min(16, int(file_size_mb / 8))  # More aggressive parallelism
                logger.debug(f"📤 Using {upload_options['max_concurrency']} concurrent connections for {file_size_mb:.1f}MB upload")
            
            # Upload with retry
            await self._retry_operation(blob_client.upload_blob, **upload_options)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Uploaded '{blob_name}' ({file_size_mb:.1f}MB in {elapsed:.1f}s)")
            
            return blob_client.url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload blob '{blob_name}' to '{container_name}': {e}")
            raise RuntimeError(f"Blob upload failed: {e}")

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Download blob content as bytes"""
        start_time = time.time()
        blob_name = self._validate_blob_name(blob_name)
        
        logger.debug(f"📥 Downloading: {blob_name} from {container_name}")
        
        try:
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Download with retry and extended timeout
            download_stream = await self._retry_operation(
                blob_client.download_blob,
                max_concurrency=8,              # Increased concurrency
                validate_content=False,         # Skip MD5 validation for speed
                timeout=self.timeout_config["read_timeout"]  # Use extended read timeout
            )
            
            data = await download_stream.readall()
            
            elapsed = time.time() - start_time
            size_mb = len(data) / (1024 * 1024)
            speed_mbps = (size_mb * 8) / elapsed if elapsed > 0 else 0
            
            logger.debug(f"✅ Downloaded '{blob_name}' ({size_mb:.1f}MB in {elapsed:.1f}s @ {speed_mbps:.1f}Mbps)")
            
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob not found: '{blob_name}' in container '{container_name}'")
            # List similar files to help debugging
            try:
                prefix = blob_name.split('_page_')[0] if '_page_' in blob_name else blob_name[:20]
                similar = await self.list_blobs(container_name, prefix=prefix)
                if similar:
                    logger.info(f"💡 Similar files found: {similar[:5]}")
            except:
                pass
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except Exception as e:
            logger.error(f"❌ Failed to download blob '{blob_name}' from '{container_name}': {type(e).__name__}: {e}")
            raise RuntimeError(f"Blob download failed: {e}")

    async def download_blob_as_text(self, container_name: str, blob_name: str, 
                                   encoding: str = "utf-8") -> str:
        """Download blob content as text"""
        blob_bytes = await self.download_blob_as_bytes(container_name, blob_name)
        
        try:
            return blob_bytes.decode(encoding)
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    logger.warning(f"⚠️ Using {alt_encoding} encoding for {blob_name}")
                    return blob_bytes.decode(alt_encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode blob '{blob_name}' with any supported encoding")

    async def download_blob_as_json(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Download blob content as JSON"""
        try:
            text = await self.download_blob_as_text(container_name, blob_name)
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in blob '{blob_name}': {e}")
            raise ValueError(f"Blob '{blob_name}' contains invalid JSON")

    async def list_blobs(self, container_name: str, prefix: str = "", 
                        suffix: str = "") -> List[str]:
        """List blob names with optional prefix/suffix filters"""
        # Validate inputs
        if prefix and not re.match(r'^[\w\-./]*$', prefix):
            raise ValueError("Prefix contains invalid characters")
        if suffix and not re.match(r'^[\w\-./]*$', suffix):
            raise ValueError("Suffix contains invalid characters")
        
        try:
            container_client = await self._get_container_client(container_name)
            
            blob_names = []
            
            # Use prefix for server-side filtering (more efficient)
            list_kwargs = {"name_starts_with": prefix} if prefix else {}
            
            # List blobs
            async for blob in container_client.list_blobs(**list_kwargs):
                blob_name = blob.name
                
                # Apply suffix filter if specified
                if suffix and not blob_name.endswith(suffix):
                    continue
                
                blob_names.append(blob_name)
            
            # Sort for consistent ordering
            blob_names.sort()
            
            logger.debug(f"✅ Listed {len(blob_names)} blobs from '{container_name}' "
                        f"(prefix='{prefix}', suffix='{suffix}')")
            
            return blob_names
            
        except Exception as e:
            logger.error(f"❌ Failed to list blobs in '{container_name}': {e}")
            raise RuntimeError(f"Blob listing failed: {e}")

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists"""
        try:
            blob_name = self._validate_blob_name(blob_name)
            
            container_client = await self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Use exists() method with retry
            exists = await self._retry_operation(blob_client.exists)
            
            logger.debug(f"🔍 Blob exists check: '{blob_name}' = {exists}")
            
            return exists
            
        except Exception as e:
            logger.debug(f"❌ Error checking blob existence: {e}")
            return False

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob"""
        blob_name = self._validate_blob_name(blob_name)
        
        async with self._lock:
            try:
                container_client = await self._get_container_client(container_name)
                blob_client = container_client.get_blob_client(blob=blob_name)
                
                await self._retry_operation(blob_client.delete_blob)
                
                logger.info(f"✅ Deleted '{blob_name}' from '{container_name}'")
                return True
                
            except ResourceNotFoundError:
                logger.debug(f"⚠️ Blob '{blob_name}' not found (already deleted)")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to delete '{blob_name}': {e}")
                raise RuntimeError(f"Blob deletion failed: {e}")

    async def batch_download_blobs(self, container_name: str, blob_names: List[str], 
                                  max_concurrent: int = 10) -> Dict[str, bytes]:
        """Download multiple blobs in parallel with extended timeouts"""
        # Validate all blob names
        validated_names = []
        for name in blob_names:
            try:
                validated_names.append(self._validate_blob_name(name))
            except ValueError as e:
                logger.warning(f"Skipping invalid blob name '{name}': {e}")
        
        if not validated_names:
            return {}
        
        # Optimize concurrency based on number of blobs - INCREASED LIMITS
        max_concurrent = min(max_concurrent, len(validated_names), 20)  # Increased from 10
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"📥 Batch downloading {len(validated_names)} blobs with {max_concurrent} concurrent connections")
        
        async def download_with_semaphore(blob_name: str) -> tuple[str, Optional[bytes]]:
            async with semaphore:
                try:
                    # Extended timeout for individual downloads
                    data = await asyncio.wait_for(
                        self.download_blob_as_bytes(container_name, blob_name),
                        timeout=self.timeout_config["operation_timeout"]
                    )
                    return (blob_name, data)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout downloading {blob_name} after {self.timeout_config['operation_timeout']}s")
                    return (blob_name, None)
                except FileNotFoundError:
                    logger.warning(f"Blob not found during batch: {blob_name}")
                    return (blob_name, None)
                except Exception as e:
                    logger.error(f"Failed to download {blob_name}: {e}")
                    return (blob_name, None)
        
        # Download all blobs in parallel with extended timeout
        start_time = time.time()
        try:
            tasks = [download_with_semaphore(name) for name in validated_names]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.timeout_config["total_timeout"]  # 2 hours total
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Batch download timed out after {self.timeout_config['total_timeout']}s")
            return {}
        
        # Process results
        downloaded = {}
        total_size = 0
        for blob_name, data in results:
            if data is not None:
                downloaded[blob_name] = data
                total_size += len(data)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Batch downloaded {len(downloaded)}/{len(validated_names)} blobs "
                   f"({total_size/1024/1024:.1f}MB in {elapsed:.1f}s)")
        
        return downloaded

    async def batch_upload_blobs(self, container_name: str, blobs: Dict[str, bytes], 
                                max_concurrent: int = 10) -> Dict[str, bool]:
        """Upload multiple blobs in parallel with extended timeouts"""
        # Validate all blob names and data
        validated_blobs = {}
        for name, data in blobs.items():
            try:
                validated_name = self._validate_blob_name(name)
                if isinstance(data, bytes) and len(data) > 0:
                    validated_blobs[validated_name] = data
                else:
                    logger.warning(f"Skipping invalid data for blob '{name}'")
            except ValueError as e:
                logger.warning(f"Skipping invalid blob name '{name}': {e}")
        
        if not validated_blobs:
            return {}
        
        # Optimize concurrency - INCREASED LIMITS
        max_concurrent = min(max_concurrent, len(validated_blobs), 20)  # Increased from 10
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"📤 Batch uploading {len(validated_blobs)} blobs with {max_concurrent} concurrent connections")
        
        async def upload_with_semaphore(blob_name: str, data: bytes) -> tuple[str, bool]:
            async with semaphore:
                try:
                    # Extended timeout for individual uploads
                    await asyncio.wait_for(
                        self.upload_file(container_name, blob_name, data),
                        timeout=self.timeout_config["operation_timeout"]
                    )
                    return (blob_name, True)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout uploading {blob_name} after {self.timeout_config['operation_timeout']}s")
                    return (blob_name, False)
                except Exception as e:
                    logger.error(f"Failed to upload {blob_name}: {e}")
                    return (blob_name, False)
        
        # Upload all blobs in parallel with extended timeout
        start_time = time.time()
        try:
            tasks = [upload_with_semaphore(name, data) for name, data in validated_blobs.items()]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.timeout_config["total_timeout"]  # 2 hours total
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Batch upload timed out after {self.timeout_config['total_timeout']}s")
            return {name: False for name in validated_blobs}
        
        # Return upload results
        upload_results = {name: success for name, success in results}
        
        successful = sum(1 for success in upload_results.values() if success)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Batch uploaded {successful}/{len(validated_blobs)} blobs in {elapsed:.1f}s")
        
        return upload_results

    async def list_document_ids(self, container_name: str) -> List[str]:
        """List all document IDs by finding _context.txt files"""
        try:
            # Find all context files
            context_files = await self.list_blobs(
                container_name=container_name,
                suffix="_context.txt"
            )
            
            # Extract document IDs
            document_ids = []
            for filename in context_files:
                if filename.endswith("_context.txt"):
                    doc_id = filename[:-12]  # Remove "_context.txt"
                    document_ids.append(doc_id)
            
            # Remove duplicates and sort
            document_ids = sorted(list(set(document_ids)))
            
            logger.info(f"✅ Found {len(document_ids)} documents in '{container_name}'")
            return document_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to list document IDs: {e}")
            raise RuntimeError(f"Document listing failed: {e}")

    async def delete_document_files(self, document_id: str) -> Dict[str, int]:
        """Delete all files associated with a document ID"""
        if not document_id or not re.match(r'^[\w\-]+$', document_id):
            raise ValueError("Invalid document ID")
        
        async with self._lock:
            try:
                deleted_counts = {
                    "pdfs": 0,
                    "pages": 0,
                    "metadata": 0,
                    "annotations": 0,
                    "chats": 0,
                    "total": 0
                }
                
                # Try to delete from main container
                if hasattr(self.settings, 'AZURE_CONTAINER_NAME') and self.settings.AZURE_CONTAINER_NAME:
                    try:
                        await self.delete_blob(self.settings.AZURE_CONTAINER_NAME, f"{document_id}.pdf")
                        deleted_counts["pdfs"] = 1
                    except:
                        pass
                
                # Delete from cache container
                if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME') and self.settings.AZURE_CACHE_CONTAINER_NAME:
                    # List all files for this document
                    cache_files = await self.list_blobs(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        prefix=f"{document_id}_"
                    )
                    
                    # Batch delete for better performance
                    delete_tasks = []
                    
                    for filename in cache_files:
                        # Categorize files
                        if "_page_" in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            deleted_counts["pages"] += 1
                        elif filename.endswith(("_metadata.json", "_context.txt", "_document_index.json")):
                            deleted_counts["metadata"] += 1
                        elif "_annotations.json" in filename:
                            deleted_counts["annotations"] += 1
                        elif "_chat" in filename or "_all_chats.json" in filename:
                            deleted_counts["chats"] += 1
                        
                        delete_tasks.append(self.delete_blob(self.settings.AZURE_CACHE_CONTAINER_NAME, filename))
                    
                    # Execute deletes in batches
                    if delete_tasks:
                        batch_size = 10
                        for i in range(0, len(delete_tasks), batch_size):
                            batch = delete_tasks[i:i+batch_size]
                            results = await asyncio.gather(*batch, return_exceptions=True)
                            
                            for result in results:
                                if result is True:
                                    deleted_counts["total"] += 1
                                elif isinstance(result, Exception):
                                    logger.debug(f"Delete error (expected for missing files): {result}")
                
                deleted_counts["total"] += deleted_counts["pdfs"]
                
                logger.info(f"✅ Deleted {deleted_counts['total']} files for document '{document_id}'")
                return deleted_counts
                
            except Exception as e:
                logger.error(f"❌ Failed to delete document files: {e}")
                raise RuntimeError(f"Document deletion failed: {e}")

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the storage connection"""
        info = {
            "service_type": "Azure Blob Storage",
            "containers": {},
            "optimizations": {
                "chunk_size": "16MB streaming, 128MB single get",
                "upload_chunk_size": "16MB blocks, 256MB single put",
                "parallel_downloads": "Supported with retry (up to 20 concurrent)",
                "parallel_uploads": "Supported with retry (up to 20 concurrent)",
                "connection_pooling": "Container clients cached",
                "thread_safety": "Full asyncio locking",
                "batch_operations": "Optimized with high concurrency",
                "retry_logic": f"Max {self.retry_config['max_retries']} retries with exponential backoff",
                "timeout_settings": f"{self.timeout_config['connection_timeout']}s connection, {self.timeout_config['read_timeout']}s read, {self.timeout_config['operation_timeout']}s per operation",
                "extended_timeouts": "ENABLED FOR TESTING"
            },
            "supported_formats": {
                "documents": ["pdf"],
                "images": ["png", "jpg", "jpeg"],
                "metadata": ["json", "txt"]
            },
            "connection_status": "Unknown",
            "error_count": self._error_count
        }
        
        # Add configured containers
        if hasattr(self.settings, 'AZURE_CONTAINER_NAME') and self.settings.AZURE_CONTAINER_NAME:
            info["containers"]["main"] = self.settings.AZURE_CONTAINER_NAME
        if hasattr(self.settings, 'AZURE_CACHE_CONTAINER_NAME') and self.settings.AZURE_CACHE_CONTAINER_NAME:
            info["containers"]["cache"] = self.settings.AZURE_CACHE_CONTAINER_NAME
        
        # Connection status
        if self._error_count == 0:
            info["connection_status"] = "Healthy"
        elif self._error_count < 5:
            info["connection_status"] = "Degraded"
        else:
            info["connection_status"] = "Unhealthy"
        
        # Connection string info (without exposing it)
        if self.settings.AZURE_STORAGE_CONNECTION_STRING:
            conn_str = self.settings.AZURE_STORAGE_CONNECTION_STRING
            if "AccountName=" in conn_str:
                account_match = re.search(r'AccountName=([^;]+)', conn_str)
                if account_match:
                    info["account_name"] = account_match.group(1)
            info["has_connection_string"] = True
            info["connection_string_length"] = len(conn_str)
        else:
            info["has_connection_string"] = False
        
        return info

    async def __aenter__(self):
        """Async context manager entry"""
        await self.verify_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        try:
            # Close all container clients
            self._container_clients.clear()
            
            # Close the main client
            await self.blob_service_client.close()
            
            logger.info("✅ Storage service cleaned up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Storage service cleanup warning: {e}")

# Singleton instance management
_storage_instance = None
_storage_lock = asyncio.Lock()

async def get_storage_service():
    """Get or create storage service instance (async)"""
    global _storage_instance
    
    async with _storage_lock:
        if _storage_instance is None:
            _storage_instance = StorageService()
            await _storage_instance.verify_connection()
    
    return _storage_instance

def get_storage_service_sync():
    """Get or create storage service instance (sync) - for dependency injection"""
    global _storage_instance
    
    if _storage_instance is None:
        _storage_instance = StorageService()
    
    return _storage_instance

def configure_timeouts(extended: bool = True):
    """Configure timeout mode - call this before using the service"""
    CONFIG['USE_EXTENDED_TIMEOUTS'] = extended
    logger.info(f"⏰ Configured timeouts: {'EXTENDED' if extended else 'PRODUCTION'}")

# For backward compatibility
get_storage_service = get_storage_service_sync