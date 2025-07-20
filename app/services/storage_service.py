# app/services/storage_service.py - COMPLETE FIXED VERSION

"""
Azure Blob Storage Service with enhanced reliability and performance.
Handles all document storage operations with proper error handling and retries.
"""

import logging
import asyncio
import aiofiles
import json
from typing import Optional, List, Dict, Any, Union, BinaryIO
from datetime import datetime, timedelta
import io
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import ContentSettings, BlobSasPermissions, generate_blob_sas
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, AzureError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import AppSettings

logger = logging.getLogger(__name__)


class StorageService:
    """Azure Blob Storage service with comprehensive operations and error handling."""
    
    def __init__(self, settings: AppSettings):
        """Initialize storage service with Azure Blob Storage client."""
        if not settings:
            raise ValueError("AppSettings instance is required")
            
        self.settings = settings
        
        # Initialize blob service client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY
        )
        
        # Container names
        self.main_container = settings.AZURE_CONTAINER_NAME
        self.cache_container = settings.AZURE_CACHE_CONTAINER_NAME
        
        # Performance settings
        self.max_single_put_size = 256 * 1024 * 1024  # 256MB
        self.max_block_size = 4 * 1024 * 1024  # 4MB
        self.max_concurrency = 10
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # SAS token settings
        self.sas_token_expiry_hours = 24
        
        logger.info("✅ Storage Service initialized")
        logger.info(f"   Account: {settings.AZURE_STORAGE_ACCOUNT_NAME}")
        logger.info(f"   Main container: {self.main_container}")
        logger.info(f"   Cache container: {self.cache_container}")
    
    async def initialize_containers(self) -> None:
        """Ensure required containers exist with proper configuration."""
        containers = [
            (self.main_container, "Main storage for PDF documents"),
            (self.cache_container, "Cache storage for processed data")
        ]
        
        for container_name, description in containers:
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                
                # Check if container exists
                try:
                    await asyncio.to_thread(container_client.get_container_properties)
                    logger.info(f"✓ Container '{container_name}' exists")
                except ResourceNotFoundError:
                    # Create container
                    await asyncio.to_thread(
                        container_client.create_container,
                        public_access=None,
                        metadata={"description": description}
                    )
                    logger.info(f"✅ Created container '{container_name}'")
                    
            except Exception as e:
                logger.error(f"Failed to initialize container '{container_name}': {e}")
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(AzureError),
        reraise=True
    )
    async def upload_file(
        self,
        container_name: str,
        blob_name: str,
        data: Union[bytes, BinaryIO],
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Azure Blob Storage with automatic retry and error handling.
        
        Args:
            container_name: Target container name
            blob_name: Blob name (path)
            data: File data as bytes or file-like object
            content_type: MIME type of the content
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with upload details including URL and etag
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Prepare content settings
            content_settings = ContentSettings(content_type=content_type)
            
            # Convert data to bytes if necessary
            if hasattr(data, 'read'):
                file_data = data.read()
                if hasattr(data, 'close'):
                    data.close()
            else:
                file_data = data
            
            # Upload with progress tracking for large files
            if len(file_data) > self.max_single_put_size:
                logger.info(f"📤 Uploading large file {blob_name} ({len(file_data) / (1024*1024):.1f}MB) in blocks...")
                
                # Upload in blocks for large files
                response = await asyncio.to_thread(
                    blob_client.upload_blob,
                    data=file_data,
                    content_settings=content_settings,
                    metadata=metadata,
                    overwrite=True,
                    max_concurrency=self.max_concurrency
                )
            else:
                # Single shot upload for smaller files
                response = await asyncio.to_thread(
                    blob_client.upload_blob,
                    data=file_data,
                    content_settings=content_settings,
                    metadata=metadata,
                    overwrite=True
                )
            
            # Get blob URL
            blob_url = blob_client.url
            
            logger.info(f"✅ Uploaded {blob_name} to {container_name} ({len(file_data)} bytes)")
            
            return {
                "blob_name": blob_name,
                "container": container_name,
                "url": blob_url,
                "etag": response.get('etag'),
                "last_modified": response.get('last_modified'),
                "size": len(file_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to upload {blob_name} to {container_name}: {e}")
            raise

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Download a blob as bytes with automatic retry."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            logger.debug(f"📥 Downloading {blob_name} from {container_name}")
            
            # Download blob
            download_stream = await asyncio.to_thread(blob_client.download_blob)
            data = await asyncio.to_thread(download_stream.readall)
            
            logger.debug(f"✅ Downloaded {blob_name} ({len(data)} bytes)")
            
            return data
            
        except ResourceNotFoundError:
            logger.error(f"Blob not found: {blob_name} in {container_name}")
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            raise

    async def download_blob_as_text(self, container_name: str, blob_name: str, encoding: str = 'utf-8') -> str:
        """Download a blob as text with specified encoding."""
        try:
            data = await self.download_blob_as_bytes(container_name, blob_name)
            return data.decode(encoding)
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode {blob_name} as {encoding}: {e}")
            raise ValueError(f"Failed to decode blob as {encoding}")

    async def download_blob_as_json(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Download and parse a JSON blob."""
        try:
            text = await self.download_blob_as_text(container_name, blob_name)
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {blob_name}: {e}")
            raise ValueError(f"Invalid JSON in blob '{blob_name}'")

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists in the specified container."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Try to get blob properties
            await asyncio.to_thread(blob_client.get_blob_properties)
            return True
            
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking blob existence: {e}")
            return False

    async def list_blobs(
        self,
        container_name: str,
        prefix: Optional[str] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """List blobs in a container with optional prefix filter."""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            blobs = []
            
            # List blobs with optional prefix
            blob_list = container_client.list_blobs(
                name_starts_with=prefix,
                include=['metadata'] if include_metadata else None
            )
            
            async for blob in blob_list:
                blob_info = {
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    "etag": blob.etag
                }
                
                if include_metadata and blob.metadata:
                    blob_info["metadata"] = blob.metadata
                
                blobs.append(blob_info)
            
            return blobs
            
        except Exception as e:
            logger.error(f"Failed to list blobs in {container_name}: {e}")
            raise

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob from storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            await asyncio.to_thread(blob_client.delete_blob)
            logger.info(f"🗑️ Deleted {blob_name} from {container_name}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found for deletion: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete blob {blob_name}: {e}")
            raise

    async def delete_blobs_with_prefix(self, container_name: str, prefix: str) -> int:
        """Delete all blobs with a specific prefix."""
        try:
            # List blobs with prefix
            blobs = await self.list_blobs(container_name, prefix=prefix)
            
            # Delete each blob
            deleted_count = 0
            for blob in blobs:
                if await self.delete_blob(container_name, blob['name']):
                    deleted_count += 1
            
            logger.info(f"🗑️ Deleted {deleted_count} blobs with prefix '{prefix}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete blobs with prefix {prefix}: {e}")
            raise

    async def copy_blob(
        self,
        source_container: str,
        source_blob: str,
        dest_container: str,
        dest_blob: str
    ) -> Dict[str, Any]:
        """Copy a blob between containers or rename within same container."""
        try:
            # Get source blob URL
            source_client = self.blob_service_client.get_blob_client(
                container=source_container,
                blob=source_blob
            )
            
            # Generate SAS token for source blob
            sas_token = generate_blob_sas(
                account_name=self.settings.AZURE_STORAGE_ACCOUNT_NAME,
                container_name=source_container,
                blob_name=source_blob,
                account_key=self.settings.AZURE_STORAGE_ACCOUNT_KEY,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1)
            )
            
            source_url = f"{source_client.url}?{sas_token}"
            
            # Get destination client
            dest_client = self.blob_service_client.get_blob_client(
                container=dest_container,
                blob=dest_blob
            )
            
            # Start copy operation
            copy_result = await asyncio.to_thread(
                dest_client.start_copy_from_url,
                source_url
            )
            
            logger.info(f"📋 Copied {source_blob} to {dest_blob}")
            
            return {
                "source": f"{source_container}/{source_blob}",
                "destination": f"{dest_container}/{dest_blob}",
                "copy_id": copy_result.get('copy_id'),
                "copy_status": copy_result.get('copy_status')
            }
            
        except Exception as e:
            logger.error(f"Failed to copy blob: {e}")
            raise

    async def get_blob_metadata(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Get blob properties and metadata."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            properties = await asyncio.to_thread(blob_client.get_blob_properties)
            
            return {
                "name": blob_name,
                "size": properties.size,
                "content_type": properties.content_settings.content_type if properties.content_settings else None,
                "last_modified": properties.last_modified.isoformat() if properties.last_modified else None,
                "etag": properties.etag,
                "metadata": properties.metadata or {}
            }
            
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob '{blob_name}' not found")
        except Exception as e:
            logger.error(f"Failed to get blob metadata: {e}")
            raise

    async def update_blob_metadata(
        self,
        container_name: str,
        blob_name: str,
        metadata: Dict[str, str]
    ) -> None:
        """Update blob metadata without re-uploading the content."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            await asyncio.to_thread(
                blob_client.set_blob_metadata,
                metadata=metadata
            )
            
            logger.info(f"📝 Updated metadata for {blob_name}")
            
        except Exception as e:
            logger.error(f"Failed to update blob metadata: {e}")
            raise

    async def generate_blob_sas_url(
        self,
        container_name: str,
        blob_name: str,
        expiry_hours: Optional[int] = None,
        permission: str = "r"
    ) -> str:
        """Generate a SAS URL for temporary blob access."""
        try:
            expiry_hours = expiry_hours or self.sas_token_expiry_hours
            
            # Set permissions
            permissions = BlobSasPermissions(
                read="r" in permission,
                write="w" in permission,
                delete="d" in permission
            )
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.settings.AZURE_STORAGE_ACCOUNT_NAME,
                container_name=container_name,
                blob_name=blob_name,
                account_key=self.settings.AZURE_STORAGE_ACCOUNT_KEY,
                permission=permissions,
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            
            # Build URL
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            sas_url = f"{blob_client.url}?{sas_token}"
            
            logger.debug(f"🔗 Generated SAS URL for {blob_name} (expires in {expiry_hours}h)")
            
            return sas_url
            
        except Exception as e:
            logger.error(f"Failed to generate SAS URL: {e}")
            raise

    async def list_document_ids(self, container_name: str) -> List[str]:
        """List unique document IDs from blob names in a container."""
        try:
            # List all blobs
            blobs = await self.list_blobs(container_name)
            
            # Extract document IDs
            document_ids = set()
            
            for blob in blobs:
                # Extract document ID from blob name
                # Assumes format: {document_id}_{suffix}.{extension}
                blob_name = blob['name']
                
                # Handle different blob name patterns
                if '_' in blob_name:
                    doc_id = blob_name.split('_')[0]
                else:
                    # For blobs without underscore, use name without extension
                    doc_id = blob_name.rsplit('.', 1)[0]
                
                # Validate document ID format
                if len(doc_id) >= 3 and doc_id.replace('-', '').replace('_', '').isalnum():
                    document_ids.add(doc_id)
            
            return sorted(list(document_ids))
            
        except Exception as e:
            logger.error(f"Failed to list document IDs: {e}")
            raise

    async def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get statistics about a container."""
        try:
            blobs = await self.list_blobs(container_name, include_metadata=True)
            
            total_size = sum(blob['size'] for blob in blobs)
            
            # Group by content type
            by_type = {}
            for blob in blobs:
                content_type = blob.get('content_type', 'unknown')
                if content_type not in by_type:
                    by_type[content_type] = {"count": 0, "size": 0}
                by_type[content_type]["count"] += 1
                by_type[content_type]["size"] += blob['size']
            
            return {
                "container": container_name,
                "total_blobs": len(blobs),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "by_content_type": by_type,
                "last_updated": datetime.utcnow().isoformat() + 'Z'
            }
            
        except Exception as e:
            logger.error(f"Failed to get container stats: {e}")
            raise

    def get_service_info(self) -> Dict[str, Any]:
        """Get storage service configuration and status."""
        return {
            "service": "Azure Blob Storage",
            "account": self.settings.AZURE_STORAGE_ACCOUNT_NAME,
            "containers": {
                "main": self.main_container,
                "cache": self.cache_container
            },
            "settings": {
                "max_single_put_size_mb": self.max_single_put_size / (1024 * 1024),
                "max_block_size_mb": self.max_block_size / (1024 * 1024),
                "max_concurrency": self.max_concurrency,
                "sas_token_expiry_hours": self.sas_token_expiry_hours
            },
            "status": "operational"
        }

    async def cleanup_old_blobs(self, container_name: str, days_old: int = 30) -> int:
        """Clean up blobs older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            blobs = await self.list_blobs(container_name)
            
            deleted_count = 0
            for blob in blobs:
                if blob.get('last_modified'):
                    last_modified = datetime.fromisoformat(blob['last_modified'].rstrip('Z'))
                    if last_modified < cutoff_date:
                        if await self.delete_blob(container_name, blob['name']):
                            deleted_count += 1
            
            logger.info(f"🧹 Cleaned up {deleted_count} blobs older than {days_old} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old blobs: {e}")
            raise

    async def move_blob(
        self,
        source_container: str,
        source_blob: str,
        dest_container: str,
        dest_blob: str
    ) -> bool:
        """Move a blob (copy then delete original)."""
        try:
            # Copy blob
            await self.copy_blob(source_container, source_blob, dest_container, dest_blob)
            
            # Delete original
            await self.delete_blob(source_container, source_blob)
            
            logger.info(f"➡️ Moved {source_blob} to {dest_container}/{dest_blob}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move blob: {e}")
            raise