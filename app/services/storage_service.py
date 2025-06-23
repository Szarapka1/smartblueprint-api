# app/services/storage_service.py - COMPLETE FIXED VERSION

import logging
from typing import List, Optional
from app.core.config import AppSettings
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

logger = logging.getLogger(__name__)

class StorageService:
    """Azure Blob Storage service with comprehensive error handling"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        
        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.AZURE_STORAGE_CONNECTION_STRING
            )
            logger.info("✅ StorageService initialized with Azure Blob Storage")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Azure Blob Storage client: {e}")
            raise ValueError(f"Azure Storage initialization failed: {e}")

    async def verify_connection(self):
        """Verify connection and ensure required containers exist"""
        logger.info("🔍 Verifying Azure Storage connection and containers...")
        
        containers = [
            self.settings.AZURE_CONTAINER_NAME,
            self.settings.AZURE_CACHE_CONTAINER_NAME
        ]
        
        for container_name in containers:
            try:
                logger.info(f"🔍 Checking container: {container_name}")
                container_client = self.blob_service_client.get_container_client(container_name)
                
                # Try to create container (idempotent operation)
                await container_client.create_container()
                logger.info(f"✅ Container '{container_name}' created successfully")
                
            except ResourceExistsError:
                logger.info(f"✅ Container '{container_name}' already exists")
                
            except Exception as e:
                logger.critical(f"❌ Container verification failed for '{container_name}': {e}")
                logger.critical(f"❌ Error type: {type(e).__name__}")
                raise ConnectionError(f"Azure Storage container '{container_name}' verification failed: {e}")
        
        logger.info("✅ All Azure Storage containers verified successfully")

    async def upload_file(self, container_name: str, blob_name: str, data: bytes) -> str:
        """Upload data to Azure Blob Storage"""
        if not blob_name or not blob_name.strip():
            raise ValueError("Blob name cannot be empty")
        
        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes")
        
        try:
            logger.info(f"⬆️ Uploading blob: {blob_name} to container: {container_name}")
            logger.info(f"⬆️ Data size: {len(data):,} bytes ({len(data) / (1024*1024):.2f} MB)")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            # Upload with overwrite=True to replace existing blobs
            await blob_client.upload_blob(data, overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"✅ Successfully uploaded '{blob_name}' to container '{container_name}'")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload blob '{blob_name}' to container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            raise RuntimeError(f"Blob upload failed: {e}")

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Download blob content as bytes"""
        if not blob_name or not blob_name.strip():
            raise ValueError("Blob name cannot be empty")
        
        try:
            logger.info(f"⬇️ Downloading blob: {blob_name} from container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            # Get blob properties first for size info
            try:
                blob_properties = await blob_client.get_blob_properties()
                blob_size = blob_properties.size
                blob_size_mb = blob_size / (1024 * 1024)
                logger.info(f"📊 Blob size: {blob_size:,} bytes ({blob_size_mb:.2f} MB)")
                
                if blob_size_mb > 50:  # Warn for large blobs
                    logger.warning(f"⚠️ Large blob download: {blob_size_mb:.2f} MB")
                    
            except Exception as prop_error:
                logger.warning(f"⚠️ Could not get blob properties: {prop_error}")
            
            # Download the blob
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            
            downloaded_size_mb = len(data) / (1024 * 1024)
            logger.info(f"✅ Downloaded '{blob_name}' successfully: {len(data):,} bytes ({downloaded_size_mb:.2f} MB)")
            
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob '{blob_name}' not found in container '{container_name}'")
            raise FileNotFoundError(f"Blob '{blob_name}' not found in container '{container_name}'")
        except Exception as e:
            logger.error(f"❌ Failed to download blob '{blob_name}' from container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            raise RuntimeError(f"Blob download failed: {e}")

    async def download_blob_as_text(self, container_name: str, blob_name: str, encoding: str = "utf-8") -> str:
        """Download blob content as text with encoding handling"""
        try:
            logger.info(f"📄 Downloading text blob: {blob_name}")
            blob_bytes = await self.download_blob_as_bytes(container_name, blob_name)
            
            # Try primary encoding
            try:
                text_content = blob_bytes.decode(encoding)
                logger.info(f"✅ Successfully decoded text blob ({encoding}): {len(text_content):,} characters")
                return text_content
            except UnicodeDecodeError as decode_error:
                logger.warning(f"⚠️ Failed to decode blob as {encoding}: {decode_error}")
                
                # Try alternative encodings
                alternative_encodings = ['latin-1', 'cp1252', 'iso-8859-1']
                for alt_encoding in alternative_encodings:
                    try:
                        text_content = blob_bytes.decode(alt_encoding)
                        logger.warning(f"⚠️ Successfully decoded using {alt_encoding}: {len(text_content):,} characters")
                        return text_content
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail
                raise ValueError(f"Could not decode blob '{blob_name}' with any supported encoding")
                
        except (FileNotFoundError, RuntimeError):
            raise  # Re-raise these as-is
        except Exception as e:
            logger.error(f"❌ Failed to download text blob '{blob_name}': {e}")
            raise RuntimeError(f"Text blob download failed: {e}")

    async def list_blobs(self, container_name: str, prefix: str = "") -> List[str]:
        """List blob names in a container with optional prefix filter"""
        try:
            logger.info(f"📋 Listing blobs in container: {container_name}" + (f" with prefix: {prefix}" if prefix else ""))
            
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_names = []
            
            # List blobs with optional prefix
            list_kwargs = {"name_starts_with": prefix} if prefix else {}
            
            async for blob in container_client.list_blobs(**list_kwargs):
                blob_names.append(blob.name)
                logger.debug(f"📁 Found blob: {blob.name}")
            
            logger.info(f"✅ Listed {len(blob_names)} blobs from container '{container_name}'")
            
            # Log summary of blob types
            if blob_names:
                blob_types = {}
                for name in blob_names:
                    if name.endswith('.png'):
                        blob_types['PNG images'] = blob_types.get('PNG images', 0) + 1
                    elif name.endswith('.txt'):
                        blob_types['Text files'] = blob_types.get('Text files', 0) + 1
                    elif name.endswith('.json'):
                        blob_types['JSON files'] = blob_types.get('JSON files', 0) + 1
                    elif name.endswith('.pdf'):
                        blob_types['PDF files'] = blob_types.get('PDF files', 0) + 1
                    else:
                        blob_types['Other files'] = blob_types.get('Other files', 0) + 1
                
                logger.info(f"📊 Blob type summary: {blob_types}")
            
            return blob_names
            
        except Exception as e:
            logger.error(f"❌ Failed to list blobs in container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            raise RuntimeError(f"Blob listing failed: {e}")

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Delete a blob from storage"""
        if not blob_name or not blob_name.strip():
            raise ValueError("Blob name cannot be empty")
        
        try:
            logger.info(f"🗑️ Deleting blob: {blob_name} from container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            await blob_client.delete_blob()
            
            logger.info(f"✅ Successfully deleted '{blob_name}' from container '{container_name}'")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"⚠️ Blob '{blob_name}' not found in container '{container_name}' (already deleted)")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete '{blob_name}' from container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            raise RuntimeError(f"Blob deletion failed: {e}")

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists in storage"""
        if not blob_name or not blob_name.strip():
            return False
        
        try:
            logger.debug(f"🔍 Checking if blob exists: {blob_name} in container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            await blob_client.get_blob_properties()
            
            logger.debug(f"✅ Blob exists: {blob_name}")
            return True
            
        except ResourceNotFoundError:
            logger.debug(f"❌ Blob does not exist: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking if blob '{blob_name}' exists: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            return False

    async def get_blob_info(self, container_name: str, blob_name: str) -> dict:
        """Get detailed information about a blob"""
        try:
            logger.info(f"ℹ️ Getting blob info: {blob_name}")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            properties = await blob_client.get_blob_properties()
            
            blob_info = {
                "name": blob_name,
                "container": container_name,
                "size_bytes": properties.size,
                "size_mb": round(properties.size / (1024 * 1024), 2),
                "content_type": properties.content_settings.content_type if properties.content_settings else None,
                "last_modified": properties.last_modified.isoformat() if properties.last_modified else None,
                "etag": properties.etag,
                "exists": True,
                "metadata": dict(properties.metadata) if properties.metadata else {}
            }
            
            logger.info(f"✅ Blob info retrieved: {blob_info['size_mb']} MB, {blob_info['content_type']}")
            return blob_info
            
        except ResourceNotFoundError:
            return {
                "name": blob_name,
                "container": container_name,
                "exists": False,
                "error": "Blob not found"
            }
        except Exception as e:
            logger.error(f"❌ Error getting blob info for '{blob_name}': {e}")
            return {
                "name": blob_name,
                "container": container_name,
                "exists": False,
                "error": str(e)
            }

    async def copy_blob(self, source_container: str, source_blob: str, 
                       dest_container: str, dest_blob: str) -> bool:
        """Copy a blob from one location to another"""
        try:
            logger.info(f"📋 Copying blob: {source_container}/{source_blob} -> {dest_container}/{dest_blob}")
            
            # Get source blob URL
            source_blob_client = self.blob_service_client.get_blob_client(
                container=source_container, 
                blob=source_blob
            )
            source_url = source_blob_client.url
            
            # Copy to destination
            dest_blob_client = self.blob_service_client.get_blob_client(
                container=dest_container, 
                blob=dest_blob
            )
            await dest_blob_client.start_copy_from_url(source_url)
            
            logger.info(f"✅ Blob copy initiated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy blob: {e}")
            return False

    async def test_storage_connection(self) -> dict:
        """Comprehensive storage connection test"""
        test_results = {
            "connection_test": False,
            "containers": {},
            "test_upload": False,
            "test_download": False,
            "test_list": False,
            "errors": [],
            "timestamp": None
        }
        
        try:
            import datetime
            test_results["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
            
            # Test 1: Basic connection and container verification
            logger.info("🧪 Testing Azure Storage connection...")
            await self.verify_connection()
            test_results["connection_test"] = True
            logger.info("✅ Connection test passed")
            
            # Test 2: Container access
            containers = [self.settings.AZURE_CONTAINER_NAME, self.settings.AZURE_CACHE_CONTAINER_NAME]
            for container_name in containers:
                try:
                    blobs = await self.list_blobs(container_name)
                    test_results["containers"][container_name] = {
                        "accessible": True,
                        "blob_count": len(blobs)
                    }
                except Exception as e:
                    test_results["containers"][container_name] = {
                        "accessible": False,
                        "error": str(e)
                    }
                    test_results["errors"].append(f"Container {container_name}: {e}")
            
            test_results["test_list"] = True
            
            # Test 3: Upload/Download cycle
            test_blob_name = f"storage_test_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
            test_data = "This is a test file for storage verification.".encode('utf-8')
            
            try:
                # Test upload
                logger.info("🧪 Testing file upload...")
                await self.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=test_blob_name,
                    data=test_data
                )
                test_results["test_upload"] = True
                
                # Test download
                logger.info("🧪 Testing file download...")
                downloaded_data = await self.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=test_blob_name
                )
                
                if downloaded_data == test_data:
                    test_results["test_download"] = True
                    logger.info("✅ Upload/Download test passed")
                else:
                    test_results["errors"].append("Downloaded data doesn't match uploaded data")
                
                # Clean up test file
                await self.delete_blob(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=test_blob_name
                )
                
            except Exception as e:
                test_results["errors"].append(f"Upload/Download test failed: {e}")
                logger.error(f"❌ Upload/Download test failed: {e}")
        
        except Exception as e:
            test_results["errors"].append(f"Connection test failed: {e}")
            logger.error(f"❌ Storage connection test failed: {e}")
        
        # Calculate overall success
        test_results["overall_success"] = (
            test_results["connection_test"] and 
            test_results["test_upload"] and 
            test_results["test_download"] and
            len(test_results["errors"]) == 0
        )
        
        logger.info(f"🧪 Storage test completed: {test_results}")
        return test_results

    async def get_container_stats(self, container_name: str) -> dict:
        """Get statistics about a container"""
        try:
            blobs = await self.list_blobs(container_name)
            
            total_size = 0
            file_types = {}
            
            # Get detailed info for each blob (limited to avoid timeouts)
            sample_size = min(len(blobs), 100)  # Sample first 100 blobs
            for blob_name in blobs[:sample_size]:
                try:
                    blob_info = await self.get_blob_info(container_name, blob_name)
                    if blob_info.get("exists"):
                        total_size += blob_info.get("size_bytes", 0)
                        
                        # Count file types
                        if "." in blob_name:
                            ext = blob_name.split(".")[-1].lower()
                            file_types[ext] = file_types.get(ext, 0) + 1
                        else:
                            file_types["no_extension"] = file_types.get("no_extension", 0) + 1
                except:
                    continue  # Skip blobs that can't be accessed
            
            return {
                "container_name": container_name,
                "total_blobs": len(blobs),
                "sampled_blobs": sample_size,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "largest_files": [],  # Could be implemented if needed
                "success": True
            }
            
        except Exception as e:
            return {
                "container_name": container_name,
                "error": str(e),
                "success": False
            }

    def get_connection_info(self) -> dict:
        """Get information about the storage connection (safe for logging)"""
        return {
            "service_type": "Azure Blob Storage",
            "containers": {
                "main": self.settings.AZURE_CONTAINER_NAME,
                "cache": self.settings.AZURE_CACHE_CONTAINER_NAME
            },
            "has_connection_string": bool(self.settings.AZURE_STORAGE_CONNECTION_STRING),
            "connection_string_length": len(self.settings.AZURE_STORAGE_CONNECTION_STRING) if self.settings.AZURE_STORAGE_CONNECTION_STRING else 0
        }

    # Context manager support for async operations
    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        try:
            # Close blob service client if it has a close method
            if hasattr(self.blob_service_client, 'close'):
                await self.blob_service_client.close()
        except Exception as e:
            logger.warning(f"⚠️ Storage service cleanup warning: {e}")

# Utility functions for storage operations
class StorageError(Exception):
    """Custom exception for storage-related errors"""
    pass

def validate_container_name(name: str) -> bool:
    """Validate Azure container name according to Azure rules"""
    if not name:
        return False
    if len(name) < 3 or len(name) > 63:
        return False
    if not name.islower():
        return False
    if not name.replace('-', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').isalpha():
        return False
    return True

def validate_blob_name(name: str) -> bool:
    """Validate blob name for Azure storage"""
    if not name or len(name) > 1024:
        return False
    # Azure blob names cannot end with dot or slash
    if name.endswith('.') or name.endswith('/'):
        return False
    return True
