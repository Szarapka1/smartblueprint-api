# app/services/storage_service.py - PRODUCTION-GRADE WITH PROPER ERROR HANDLING

import logging
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from app.core.config import get_settings, CONFIG  # Fixed: Import actual config
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobPrefix, ContentSettings
import re

logger = logging.getLogger(__name__)

class StorageService:
    """Production-grade Azure Blob Storage service with robust error handling"""
    
    def __init__(self, settings=None):
        # Use provided settings or get from config
        self.settings = settings or get_settings()
        
        if not self.settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
        
        self._lock = asyncio.Lock()  # Thread safety for operations
        
        try:
            # Create blob service client with optimized settings
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.settings.AZURE_STORAGE_CONNECTION_STRING,
                max_single_get_size=32*1024*1024,  # 32MB chunks for large files
                max_chunk_get_size=4*1024*1024     # 4MB chunks for streaming
            )
            
            # Pre-create container clients for better performance
            self.main_container_client = self.blob_service_client.get_container_client(
                self.settings.AZURE_CONTAINER_NAME
            )
            self.cache_container_client = self.blob_service_client.get_container_client(
                self.settings.AZURE_CACHE_CONTAINER_NAME
            )
            
            logger.info("✅ StorageService initialized successfully")
            
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Azure Blob Storage client: {e}")
            raise ValueError(f"Azure Storage initialization failed: {e}")

    async def verify_connection(self):
        """Verify connection and ensure required containers exist"""
        async with self._lock:
            logger.info("🔍 Verifying Azure Storage connection and containers...")
            
            containers = [
                (self.settings.AZURE_CONTAINER_NAME, self.main_container_client),
                (self.settings.AZURE_CACHE_CONTAINER_NAME, self.cache_container_client)
            ]
            
            for container_name, client in containers:
                try:
                    logger.info(f"🔍 Checking container: {container_name}")
                    await client.create_container()
                    logger.info(f"✅ Container '{container_name}' created successfully")
                except ResourceExistsError:
                    logger.info(f"✅ Container '{container_name}' already exists")
                except Exception as e:
                    logger.critical(f"❌ Container verification failed for '{container_name}': {e}")
                    raise ConnectionError(f"Azure Storage container '{container_name}' verification failed: {e}")
            
            logger.info("✅ All Azure Storage containers verified successfully")

    def _validate_blob_name(self, blob_name: str) -> str:
        """Validate and sanitize blob name to prevent security issues"""
        if not blob_name or not isinstance(blob_name, str):
            raise ValueError("Blob name must be a non-empty string")
        
        # Remove any path traversal attempts
        blob_name = blob_name.replace("../", "").replace("..\\", "")
        
        # Ensure valid characters (optimized regex)
        if not re.match(r'^[\w\-./]+$', blob_name):
            raise ValueError("Blob name contains invalid characters")
        
        # Strip leading/trailing slashes
        blob_name = blob_name.strip("/")
        
        if not blob_name:
            raise ValueError("Blob name cannot be empty after sanitization")
            
        return blob_name

    async def upload_file(self, container_name: str, blob_name: str, data: bytes, 
                         content_type: Optional[str] = None) -> str:
        """Upload data to Azure Blob Storage with validation and error handling"""
        blob_name = self._validate_blob_name(blob_name)
        
        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes")
        
        if len(data) == 0:
            raise ValueError("Cannot upload empty data")
        
        # Validate container name
        if container_name not in [self.settings.AZURE_CONTAINER_NAME, 
                                  self.settings.AZURE_CACHE_CONTAINER_NAME]:
            raise ValueError(f"Invalid container name: {container_name}")
        
        try:
            # Use pre-created container clients
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Prepare upload options
            upload_options = {
                'data': data,
                'overwrite': True
            }
            
            # Add content type if provided (with smart defaults for images)
            if content_type:
                upload_options['content_settings'] = ContentSettings(content_type=content_type)
            elif blob_name.lower().endswith('.png'):
                upload_options['content_settings'] = ContentSettings(content_type='image/png')
            elif blob_name.lower().endswith(('.jpg', '.jpeg')):
                upload_options['content_settings'] = ContentSettings(content_type='image/jpeg')
            elif blob_name.lower().endswith('.pdf'):
                upload_options['content_settings'] = ContentSettings(content_type='application/pdf')
            
            # For large files (>4MB), use optimized upload settings
            if len(data) > 4 * 1024 * 1024:
                upload_options['max_concurrency'] = 4  # Parallel upload chunks
                upload_options['validate_content'] = False  # Skip MD5 validation for speed
            
            await blob_client.upload_blob(**upload_options)
            
            logger.info(f"✅ Uploaded '{blob_name}' ({len(data)/1024/1024:.1f}MB)")
            return blob_client.url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload blob '{blob_name}': {e}")
            raise RuntimeError(f"Blob upload failed: {e}")

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Download blob content as bytes with validation"""
        blob_name = self._validate_blob_name(blob_name)
        
        # Validate container name
        if container_name not in [self.settings.AZURE_CONTAINER_NAME, 
                                  self.settings.AZURE_CACHE_CONTAINER_NAME]:
            raise ValueError(f"Invalid container name: {container_name}")
        
        try:
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Stream download for efficiency
            download_stream = await blob_client.download_blob(max_concurrency=4)
            data = await download_stream.readall()
            
            logger.debug(f"✅ Downloaded '{blob_name}' ({len(data)/1024/1024:.1f}MB)")
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob '{blob_name}' not found in container '{container_name}'")
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except Exception as e:
            logger.error(f"❌ Failed to download blob '{blob_name}': {e}")
            raise RuntimeError(f"Blob download failed: {e}")

    async def download_blob_as_text(self, container_name: str, blob_name: str, 
                                   encoding: str = "utf-8") -> str:
        """Download blob content as text with encoding handling"""
        blob_bytes = await self.download_blob_as_bytes(container_name, blob_name)
        
        try:
            return blob_bytes.decode(encoding)
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    logger.warning(f"⚠️ Using {alt_encoding} encoding for {blob_name}")
                    return blob_bytes.decode(alt_encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode blob '{blob_name}' with any supported encoding")

    async def list_blobs(self, container_name: str, prefix: str = "", 
                        suffix: str = "") -> List[str]:
        """List blob names with validation and error handling"""
        # Validate container name
        if container_name not in [self.settings.AZURE_CONTAINER_NAME, 
                                  self.settings.AZURE_CACHE_CONTAINER_NAME]:
            raise ValueError(f"Invalid container name: {container_name}")
        
        # Validate prefix and suffix (optimized regex)
        if prefix and not re.match(r'^[\w\-./]*$', prefix):
            raise ValueError("Prefix contains invalid characters")
        if suffix and not re.match(r'^[\w\-./]*$', suffix):
            raise ValueError("Suffix contains invalid characters")
        
        try:
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_names = []
            
            # Use prefix for server-side filtering
            list_kwargs = {"name_starts_with": prefix} if prefix else {}
            
            # List blobs with prefix filter
            async for blob in container_client.list_blobs(**list_kwargs):
                # Apply suffix filter client-side if needed
                if suffix and not blob.name.endswith(suffix):
                    continue
                blob_names.append(blob.name)
            
            logger.debug(f"✅ Listed {len(blob_names)} blobs from '{container_name}'")
            return blob_names
            
        except Exception as e:
            logger.error(f"❌ Failed to list blobs: {e}")
            raise RuntimeError(f"Blob listing failed: {e}")

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists with validation"""
        try:
            blob_name = self._validate_blob_name(blob_name)
            
            # Validate container name
            if container_name not in [self.settings.AZURE_CONTAINER_NAME, 
                                      self.settings.AZURE_CACHE_CONTAINER_NAME]:
                return False
            
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            return await blob_client.exists()
            
        except Exception:
            return False

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob with validation"""
        blob_name = self._validate_blob_name(blob_name)
        
        # Validate container name
        if container_name not in [self.settings.AZURE_CONTAINER_NAME, 
                                  self.settings.AZURE_CACHE_CONTAINER_NAME]:
            raise ValueError(f"Invalid container name: {container_name}")
        
        async with self._lock:  # Thread safety for delete operations
            try:
                container_client = (self.cache_container_client 
                                  if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                                  else self.main_container_client)
                
                blob_client = container_client.get_blob_client(blob=blob_name)
                await blob_client.delete_blob()
                
                logger.info(f"✅ Deleted '{blob_name}'")
                return True
                
            except ResourceNotFoundError:
                logger.warning(f"⚠️ Blob '{blob_name}' not found (already deleted)")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to delete '{blob_name}': {e}")
                raise RuntimeError(f"Blob deletion failed: {e}")

    async def batch_download_blobs(self, container_name: str, blob_names: List[str], 
                                  max_concurrent: int = 5) -> Dict[str, bytes]:
        """Download multiple blobs in parallel with validation"""
        # Validate all blob names first
        validated_names = []
        for name in blob_names:
            try:
                validated_names.append(self._validate_blob_name(name))
            except ValueError as e:
                logger.warning(f"Skipping invalid blob name '{name}': {e}")
        
        if not validated_names:
            return {}
        
        # Limit concurrency to reasonable value
        max_concurrent = min(max_concurrent, 10)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(blob_name: str) -> tuple[str, Optional[bytes]]:
            async with semaphore:
                try:
                    data = await self.download_blob_as_bytes(container_name, blob_name)
                    return (blob_name, data)
                except FileNotFoundError:
                    logger.warning(f"Blob not found: {blob_name}")
                    return (blob_name, None)
                except Exception as e:
                    logger.error(f"Failed to download {blob_name}: {e}")
                    return (blob_name, None)
        
        # Download all blobs in parallel
        tasks = [download_with_semaphore(name) for name in validated_names]
        results = await asyncio.gather(*tasks)
        
        # Process results
        downloaded = {}
        for blob_name, data in results:
            if data is not None:
                downloaded[blob_name] = data
        
        logger.info(f"✅ Batch downloaded {len(downloaded)}/{len(validated_names)} blobs")
        return downloaded

    async def batch_upload_blobs(self, container_name: str, blobs: Dict[str, bytes], 
                                max_concurrent: int = 5) -> Dict[str, bool]:
        """Upload multiple blobs in parallel with validation"""
        # Validate all blob names first
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
        
        # Limit concurrency
        max_concurrent = min(max_concurrent, 10)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def upload_with_semaphore(blob_name: str, data: bytes) -> tuple[str, bool]:
            async with semaphore:
                try:
                    await self.upload_file(container_name, blob_name, data)
                    return (blob_name, True)
                except Exception as e:
                    logger.error(f"Failed to upload {blob_name}: {e}")
                    return (blob_name, False)
        
        # Upload all blobs in parallel
        tasks = [upload_with_semaphore(name, data) for name, data in validated_blobs.items()]
        results = await asyncio.gather(*tasks)
        
        # Return upload results
        upload_results = {name: success for name, success in results}
        
        successful = sum(1 for success in upload_results.values() if success)
        logger.info(f"✅ Batch uploaded {successful}/{len(validated_blobs)} blobs")
        
        return upload_results

    async def list_document_ids(self, container_name: str) -> List[str]:
        """List all document IDs by finding _context.txt files"""
        try:
            # Use suffix filter to only get context files
            context_files = await self.list_blobs(
                container_name=container_name,
                suffix="_context.txt"
            )
            
            # Extract document IDs from filenames
            document_ids = []
            for filename in context_files:
                # Remove _context.txt to get document ID
                if filename.endswith("_context.txt"):
                    doc_id = filename[:-12]  # Remove "_context.txt" (12 chars)
                    document_ids.append(doc_id)
            
            logger.info(f"✅ Found {len(document_ids)} documents in '{container_name}'")
            return document_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to list document IDs: {e}")
            raise RuntimeError(f"Document listing failed: {e}")

    async def delete_document_files(self, document_id: str) -> Dict[str, int]:
        """Delete all files associated with a document ID"""
        # Validate document ID (optimized regex)
        if not document_id or not re.match(r'^[\w\-]+$', document_id):
            raise ValueError("Invalid document ID")
        
        async with self._lock:  # Thread safety for bulk delete
            try:
                deleted_counts = {
                    "pdfs": 0,
                    "pages": 0,
                    "metadata": 0,
                    "annotations": 0,
                    "chats": 0,
                    "total": 0
                }
                
                # Delete from main container (original PDF)
                try:
                    await self.delete_blob(self.settings.AZURE_CONTAINER_NAME, f"{document_id}.pdf")
                    deleted_counts["pdfs"] = 1
                except:
                    pass
                
                # List all files for this document in cache container
                cache_files = await self.list_blobs(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    prefix=f"{document_id}_"
                )
                
                # Batch delete operations for better performance
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
                    
                    # Create delete task
                    delete_tasks.append(self.delete_blob(self.settings.AZURE_CACHE_CONTAINER_NAME, filename))
                
                # Execute all deletes in parallel (with controlled concurrency)
                if delete_tasks:
                    # Process in batches of 10
                    for i in range(0, len(delete_tasks), 10):
                        batch = delete_tasks[i:i+10]
                        results = await asyncio.gather(*batch, return_exceptions=True)
                        
                        # Count successful deletes
                        for result in results:
                            if result is True:
                                deleted_counts["total"] += 1
                            elif isinstance(result, Exception):
                                logger.warning(f"Delete error: {result}")
                
                deleted_counts["total"] += deleted_counts["pdfs"]
                
                logger.info(f"✅ Deleted {deleted_counts['total']} files for document '{document_id}'")
                return deleted_counts
                
            except Exception as e:
                logger.error(f"❌ Failed to delete document files: {e}")
                raise RuntimeError(f"Document deletion failed: {e}")

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the storage connection"""
        return {
            "service_type": "Azure Blob Storage",
            "containers": {
                "main": self.settings.AZURE_CONTAINER_NAME,
                "cache": self.settings.AZURE_CACHE_CONTAINER_NAME
            },
            "optimizations": {
                "chunk_size": "4MB streaming, 32MB single",
                "parallel_downloads": "Supported",
                "parallel_uploads": "Supported",
                "pre_created_clients": True,
                "thread_safety": "Full asyncio locking",
                "batch_operations": "Optimized with controlled concurrency"
            },
            "supported_formats": {
                "documents": ["pdf"],
                "images": ["png", "jpg", "jpeg"],  # Optimized for these formats
                "metadata": ["json", "txt"]
            },
            "has_connection_string": bool(self.settings.AZURE_STORAGE_CONNECTION_STRING),
            "connection_string_length": len(self.settings.AZURE_STORAGE_CONNECTION_STRING) if self.settings.AZURE_STORAGE_CONNECTION_STRING else 0
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        try:
            await self.blob_service_client.close()
            logger.info("✅ Storage service cleaned up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Storage service cleanup warning: {e}")

# Singleton instance for easy access
_storage_instance = None

def get_storage_service():
    """Get or create storage service instance"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageService()
    return _storage_instance
