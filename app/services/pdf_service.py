# app/services/pdf_service.py
import logging
import fitz  # PyMuPDF library
import json # Added for JSON operations
from app.core.config import AppSettings
from app.services.storage_service import StorageService
# Import AIService only for its _chunk_document method, or copy the logic
# For simplicity and to avoid circular imports if AIService imports PDFService,
# we will duplicate the _chunk_document logic here or ensure it's made available.
# A better architectural approach might be a shared utility for chunking,
# but for now, we'll integrate the chunking directly or ensure it's accessible.

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

    def _chunk_document(self, text: str, chunk_size: int = 2500) -> List[Dict[str, str]]:
        """
        Break document into numbered chunks with previews for caching.
        (Copied from ai_service.py to avoid circular dependency for this direct call)
        """
        logger.info(f"📑 Chunking document into sections of max {chunk_size} characters for PDFService")
        
        # Split by double newlines first (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 1
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > chunk_size and current_chunk:
                preview = current_chunk.strip()[:200]
                if len(current_chunk) > 200:
                    preview += "..."
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": current_chunk.strip(),
                    "preview": preview
                })
                
                current_chunk = paragraph
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk
        if current_chunk.strip():
            preview = current_chunk.strip()[:200]
            if len(current_chunk) > 200:
                preview += "..."
                
            chunks.append({
                "chunk_id": chunk_id,
                "content": current_chunk.strip(),
                "preview": preview
            })
        
        logger.info(f"📑 Document chunked into {len(chunks)} sections by PDFService")
        return chunks


    async def process_and_cache_pdf(
        self,
        session_id: str,
        pdf_bytes: bytes,
        storage_service: StorageService
    ):
        """
        Orchestrates the processing of a PDF and saves its text, page images,
        and generated text chunks to the Azure cache container.
        """
        # 1. Extract full text
        full_text = self._extract_text_from_pdf(pdf_bytes)
        text_blob_name = f"{session_id}_context.txt"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=text_blob_name,
            data=full_text.encode('utf-8')
        )
        logger.info(f"Uploaded full text context to '{text_blob_name}'.")


        # 2. Extract and upload page images
        page_images = self._extract_page_images(pdf_bytes)
        for i, image_bytes in enumerate(page_images):
            image_blob_name = f"{session_id}_page_{i + 1}.png"
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=image_blob_name,
                data=image_bytes
            )
        logger.info(f"Uploaded {len(page_images)} page images.")

        # 3. Generate and cache chunks immediately
        chunks = self._chunk_document(full_text)
        chunks_blob_name = f"{session_id}_chunks.json"
        chunks_json = json.dumps(chunks, indent=2, ensure_ascii=False)
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=chunks_blob_name,
            data=chunks_json.encode('utf-8')
        )
        logger.info(f"💾 ✅ Cached {len(chunks)} chunks to '{chunks_blob_name}' during PDF processing.")
