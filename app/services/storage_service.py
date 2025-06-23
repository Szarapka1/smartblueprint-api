# app/services/storage_service.py - Enhanced with Better Error Handling and Debugging
import logging
from typing import List
from app.core.config import AppSettings
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.settings.AZURE_STORAGE_CONNECTION_STRING
            )
            logger.info("✅ StorageService initialized with Azure Blob Storage")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Azure Blob Storage client: {e}")
            raise

    async def verify_connection(self):
        """Ensures both required containers exist and the connection is valid."""
        logger.info("🔍 Verifying Azure Storage connection and containers...")
        
        for container_name in [self.settings.AZURE_CONTAINER_NAME, self.settings.AZURE_CACHE_CONTAINER_NAME]:
            try:
                logger.info(f"🔍 Checking container: {container_name}")
                container_client = self.blob_service_client.get_container_client(container_name)
                
                # Try to create container (will succeed if doesn't exist, or raise ResourceExistsError if it does)
                await container_client.create_container()
                logger.info(f"✅ Container '{container_name}' created successfully.")
                
            except ResourceExistsError:
                logger.info(f"✅ Container '{container_name}' already exists.")
                
            except Exception as e:
                logger.critical(f"❌ Azure Storage connection failed for container '{container_name}': {e}")
                logger.critical(f"❌ Error type: {type(e).__name__}")
                logger.critical(f"❌ Error details: {str(e)}")
                raise
        
        logger.info("✅ All Azure Storage containers verified successfully")

    async def upload_file(self, container_name: str, blob_name: str, data: bytes) -> str:
        """Uploads data to a blob and returns its URL."""
        try:
            logger.info(f"⬆️ Uploading blob: {blob_name} to container: {container_name}")
            logger.info(f"⬆️ Data size: {len(data)} bytes ({len(data) / (1024*1024):.2f} MB)")
            
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            
            # Upload with overwrite=True to replace existing blobs
            await blob_client.upload_blob(data, overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"✅ Successfully uploaded '{blob_name}' to container '{container_name}'")
            logger.debug(f"🔗 Blob URL: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload blob '{blob_name}' to container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            raise

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Downloads a blob from storage and returns its content as bytes."""
        try:
            logger.info(f"⬇️ Downloading blob: {blob_name} from container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            
            # Get blob properties first to check size
            try:
                blob_properties = await blob_client.get_blob_properties()
                blob_size = blob_properties.size
                blob_size_mb = blob_size / (1024 * 1024)
                logger.info(f"📊 Blob size: {blob_size} bytes ({blob_size_mb:.2f} MB)")
                
                if blob_size_mb > 50:  # Warn for large blobs
                    logger.warning(f"⚠️ Large blob download: {blob_size_mb:.2f} MB")
                    
            except Exception as prop_error:
                logger.warning(f"⚠️ Could not get blob properties: {prop_error}")
            
            # Download the blob
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            
            downloaded_size_mb = len(data) / (1024 * 1024)
            logger.info(f"✅ Downloaded '{blob_name}' successfully: {len(data)} bytes ({downloaded_size_mb:.2f} MB)")
            
            return data
            
        except ResourceNotFoundError:
            logger.error(f"❌ Blob '{blob_name}' not found in container '{container_name}'")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to download blob '{blob_name}' from container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            raise

    async def download_blob_as_text(self, container_name: str, blob_name: str) -> str:
        """Downloads a text blob and returns its content as a string."""
        try:
            logger.info(f"📄 Downloading text blob: {blob_name}")
            blob_bytes = await self.download_blob_as_bytes(container_name, blob_name)
            
            # Try to decode as UTF-8
            try:
                text_content = blob_bytes.decode("utf-8")
                logger.info(f"✅ Successfully decoded text blob: {len(text_content)} characters")
                return text_content
            except UnicodeDecodeError as decode_error:
                logger.error(f"❌ Failed to decode blob as UTF-8: {decode_error}")
                # Try alternative encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text_content = blob_bytes.decode(encoding)
                        logger.warning(f"⚠️ Successfully decoded using {encoding}: {len(text_content)} characters")
                        return text_content
                    except:
                        continue
                
                # If all encodings fail, return error
                raise ValueError(f"Could not decode blob '{blob_name}' with any supported encoding")
                
        except Exception as e:
            logger.error(f"❌ Failed to download text blob '{blob_name}': {e}")
            raise

    async def list_blobs(self, container_name: str) -> List[str]:
        """Lists all blob names in a container."""
        try:
            logger.info(f"📋 Listing blobs in container: {container_name}")
            
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_names = []
            
            # List all blobs in the container
            async for blob in container_client.list_blobs():
                blob_names.append(blob.name)
                logger.debug(f"📁 Found blob: {blob.name}")
            
            logger.info(f"✅ Listed {len(blob_names)} blobs from container '{container_name}'")
            
            # Log summary of blob types if there are any
            if blob_names:
                blob_types = {}
                for name in blob_names:
                    if name.endswith('.png'):
                        blob_types['PNG images'] = blob_types.get('PNG images', 0) + 1
                    elif name.endswith('.txt'):
                        blob_types['Text files'] = blob_types.get('Text files', 0) + 1
                    elif name.endswith('.json'):
                        blob_types['JSON files'] = blob_types.get('JSON files', 0) + 1
                    else:
                        blob_types['Other files'] = blob_types.get('Other files', 0) + 1
                
                logger.info(f"📊 Blob type summary: {blob_types}")
            
            return blob_names
            
        except Exception as e:
            logger.error(f"❌ Failed to list blobs in container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            raise

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Deletes a blob from storage."""
        try:
            logger.info(f"🗑️ Deleting blob: {blob_name} from container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            await blob_client.delete_blob()
            
            logger.info(f"✅ Successfully deleted '{blob_name}' from container '{container_name}'")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"⚠️ Blob '{blob_name}' not found in container '{container_name}' (already deleted)")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete '{blob_name}' from container '{container_name}': {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            raise

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists in storage."""
        try:
            logger.debug(f"🔍 Checking if blob exists: {blob_name} in container: {container_name}")
            
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
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
        """Get detailed information about a blob."""
        try:
            logger.info(f"ℹ️ Getting blob info: {blob_name}")
            
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            properties = await blob_client.get_blob_properties()
            
            blob_info = {
                "name": blob_name,
                "container": container_name,
                "size_bytes": properties.size,
                "size_mb": properties.size / (1024 * 1024),
                "content_type": properties.content_settings.content_type,
                "last_modified": properties.last_modified,
                "etag": properties.etag,
                "exists": True
            }
            
            logger.info(f"✅ Blob info retrieved: {blob_info['size_mb']:.2f} MB, {blob_info['content_type']}")
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

    async def test_storage_connection(self) -> dict:
        """Test the storage connection and return detailed results."""
        test_results = {
            "connection_test": False,
            "containers": {},
            "test_upload": False,
            "test_download": False,
            "errors": []
        }
        
        try:
            # Test 1: Basic connection
            logger.info("🧪 Testing Azure Storage connection...")
            await self.verify_connection()
            test_results["connection_test"] = True
            logger.info("✅ Connection test passed")
            
            # Test 2: Container access
            for container_name in [self.settings.AZURE_CONTAINER_NAME, self.settings.AZURE_CACHE_CONTAINER_NAME]:
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
            
            # Test 3: Upload/Download cycle
            test_blob_name = "storage_test.txt"
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
        
        logger.info(f"🧪 Storage test completed: {test_results}")
        return test_results
