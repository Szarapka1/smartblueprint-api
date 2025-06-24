# app/services/storage_service.py - OPTIMIZED VERSION

import logging
import asyncio
from typing import List, Optional, Dict, Any
from app.core.config import AppSettings
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

logger = logging.getLogger(__name__)

class StorageService:
    """Azure Blob Storage service optimized for blueprint handling"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        
        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.AZURE_STORAGE_CONNECTION_STRING
            )
            # Pre-create container clients for performance
            self.main_container_client = self.blob_service_client.get_container_client(
                settings.AZURE_CONTAINER_NAME
            )
            self.cache_container_client = self.blob_service_client.get_container_client(
                settings.AZURE_CACHE_CONTAINER_NAME
            )
            logger.info("✅ StorageService initialized with optimized clients")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Azure Blob Storage client: {e}")
            raise ValueError(f"Azure Storage initialization failed: {e}")

    async def verify_connection(self):
        """Verify connection and ensure required containers exist"""
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

    async def upload_file(self, container_name: str, blob_name: str, data: bytes) -> str:
        """Upload data to Azure Blob Storage with optimizations"""
        if not blob_name or not blob_name.strip():
            raise ValueError("Blob name cannot be empty")
        
        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes")
        
        try:
            # Use pre-created container clients for performance
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # For large files (>4MB), use optimized upload
            if len(data) > 4 * 1024 * 1024:
                await blob_client.upload_blob(
                    data, 
                    overwrite=True,
                    max_concurrency=4  # Parallel upload for speed
                )
            else:
                await blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"✅ Uploaded '{blob_name}' ({len(data)/1024/1024:.1f}MB)")
            return blob_client.url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload blob '{blob_name}': {e}")
            raise RuntimeError(f"Blob upload failed: {e}")

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Download blob content as bytes with streaming for large files"""
        if not blob_name or not blob_name.strip():
            raise ValueError("Blob name cannot be empty")
        
        try:
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            
            # Stream download for memory efficiency
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            
            logger.debug(f"✅ Downloaded '{blob_name}' ({len(data)/1024/1024:.1f}MB)")
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob '{blob_name}' not found")
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except Exception as e:
            logger.error(f"❌ Failed to download blob '{blob_name}': {e}")
            raise RuntimeError(f"Blob download failed: {e}")

    async def download_blob_as_text(self, container_name: str, blob_name: str, 
                                   encoding: str = "utf-8") -> str:
        """Download blob content as text"""
        blob_bytes = await self.download_blob_as_bytes(container_name, blob_name)
        
        try:
            return blob_bytes.decode(encoding)
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ['latin-1', 'cp1252']:
                try:
                    return blob_bytes.decode(alt_encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode blob '{blob_name}' with any supported encoding")

    async def list_blobs(self, container_name: str, prefix: str = "") -> List[str]:
        """List blob names in a container with optional prefix filter"""
        try:
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_names = []
            list_kwargs = {"name_starts_with": prefix} if prefix else {}
            
            async for blob in container_client.list_blobs(**list_kwargs):
                blob_names.append(blob.name)
            
            logger.debug(f"✅ Listed {len(blob_names)} blobs from '{container_name}'")
            return blob_names
            
        except Exception as e:
            logger.error(f"❌ Failed to list blobs: {e}")
            raise RuntimeError(f"Blob listing failed: {e}")

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists - optimized version"""
        if not blob_name or not blob_name.strip():
            return False
        
        try:
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            return await blob_client.exists()
            
        except Exception:
            return False

    async def get_blob_info(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Get blob information"""
        try:
            container_client = (self.cache_container_client 
                              if container_name == self.settings.AZURE_CACHE_CONTAINER_NAME 
                              else self.main_container_client)
            
            blob_client = container_client.get_blob_client(blob=blob_name)
            properties = await blob_client.get_blob_properties()
            
            return {
                "name": blob_name,
                "container": container_name,
                "size_bytes": properties.size,
                "size_mb": round(properties.size / (1024 * 1024), 2),
                "content_type": properties.content_settings.content_type if properties.content_settings else None,
                "last_modified": properties.last_modified.isoformat() if properties.last_modified else None,
                "exists": True
            }
            
        except ResourceNotFoundError:
            return {"name": blob_name, "exists": False}
        except Exception as e:
            return {"name": blob_name, "exists": False, "error": str(e)}

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob from storage"""
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

    async def batch_download_blobs(self, container_name: str, blob_names: List[str]) -> Dict[str, bytes]:
        """Download multiple blobs in parallel for performance"""
        async def download_single(blob_name: str) -> tuple[str, Optional[bytes]]:
            try:
                data = await self.download_blob_as_bytes(container_name, blob_name)
                return (blob_name, data)
            except Exception as e:
                logger.error(f"Failed to download {blob_name}: {e}")
                return (blob_name, None)
        
        # Download in parallel
        tasks = [download_single(name) for name in blob_names]
        results = await asyncio.gather(*tasks)
        
        # Return as dict, excluding failed downloads
        return {name: data for name, data in results if data is not None}

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        try:
            await self.blob_service_client.close()
        except Exception as e:
            logger.warning(f"⚠️ Storage service cleanup warning: {e}")
