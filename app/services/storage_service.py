# app/services/storage_service.py
import logging
from typing import List
from app.core.config import AppSettings
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.settings.AZURE_STORAGE_CONNECTION_STRING
        )

    async def verify_connection(self):
        """Ensures both required containers exist and the connection is valid."""
        for container_name in [self.settings.AZURE_CONTAINER_NAME, self.settings.AZURE_CACHE_CONTAINER_NAME]:
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                await container_client.create_container()
                logger.info(f"Container '{container_name}' is ready.")
            except ResourceExistsError:
                logger.info(f"Container '{container_name}' already exists.")
            except Exception as e:
                logger.critical(f"Azure Storage connection failed for container '{container_name}': {e}")
                raise

    async def upload_file(self, container_name: str, blob_name: str, data: bytes) -> str:
        """Uploads data to a blob and returns its URL."""
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        await blob_client.upload_blob(data, overwrite=True)
        logger.info(f"Uploaded '{blob_name}' to container '{container_name}'.")
        return blob_client.url

    async def download_blob_as_bytes(self, container_name: str, blob_name: str) -> bytes:
        """Downloads a blob from storage and returns its content as bytes."""
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            logger.info(f"Downloaded '{blob_name}' from container '{container_name}'.")
            return data
        except ResourceNotFoundError:
            logger.error(f"Blob '{blob_name}' not found in container '{container_name}'.")
            raise

    async def download_blob_as_text(self, container_name: str, blob_name: str) -> str:
        """Downloads a text blob and returns its content as a string."""
        blob_bytes = await self.download_blob_as_bytes(container_name, blob_name)
        return blob_bytes.decode("utf-8")

    async def list_blobs(self, container_name: str) -> List[str]:
        """Lists all blob names in a container."""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_names = []
            async for blob in container_client.list_blobs():
                blob_names.append(blob.name)
            logger.info(f"Listed {len(blob_names)} blobs from container '{container_name}'.")
            return blob_names
        except Exception as e:
            logger.error(f"Failed to list blobs in container '{container_name}': {e}")
            raise

    async def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """Deletes a blob from storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            await blob_client.delete_blob()
            logger.info(f"Deleted '{blob_name}' from container '{container_name}'.")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Blob '{blob_name}' not found in container '{container_name}' (already deleted).")
            return False
        except Exception as e:
            logger.error(f"Failed to delete '{blob_name}' from container '{container_name}': {e}")
            raise

    async def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists in storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            await blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking if blob '{blob_name}' exists: {e}")
            return False