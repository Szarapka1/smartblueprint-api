# app/services/pdf_service.py
import logging
import fitz  # PyMuPDF library
from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        logger.info("PDFService initialized.")

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extracts all text from a PDF for general analysis."""
        full_text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text() + "\n\n"
        logger.info(f"Extracted {len(full_text)} characters of text from PDF.")
        return full_text

    def _extract_page_images(self, pdf_bytes: bytes) -> list[bytes]:
        """Renders each page of a PDF into a list of PNG image bytes."""
        images = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=self.settings.PDF_PREVIEW_RESOLUTION)
                images.append(pix.tobytes("png"))
        logger.info(f"Rendered {len(images)} page images from PDF.")
        return images

    async def process_and_cache_pdf(
        self,
        session_id: str,
        pdf_bytes: bytes,
        storage_service: StorageService
    ):
        """
        Orchestrates the processing of a PDF and saves its text and page images
        to the Azure cache container.
        """
        full_text = self._extract_text_from_pdf(pdf_bytes)
        text_blob_name = f"{session_id}_context.txt"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=text_blob_name,
            data=full_text.encode('utf-8')
        )

        page_images = self._extract_page_images(pdf_bytes)
        for i, image_bytes in enumerate(page_images):
            image_blob_name = f"{session_id}_page_{i + 1}.png"
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=image_blob_name,
                data=image_bytes
            )