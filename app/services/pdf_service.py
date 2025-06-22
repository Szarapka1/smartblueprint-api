# app/services/pdf_service.py
import logging
import fitz  # PyMuPDF library
import json 
from typing import List, Dict
from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        logger.info("✅ PDFService initialized successfully.")

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extracts all text from a PDF for general analysis."""
        try:
            full_text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                logger.info(f"📄 Processing PDF with {len(doc)} pages")
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    full_text += page_text + "\n\n"
                    logger.debug(f"📄 Page {page_num + 1}: extracted {len(page_text)} characters")
            
            logger.info(f"✅ Extracted {len(full_text)} total characters from PDF")
            return full_text
        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF: {e}")
            raise

    def _extract_page_images(self, pdf_bytes: bytes) -> List[bytes]:
        """Renders each page of a PDF into a list of PNG image bytes."""
        try:
            images = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                logger.info(f"🖼️ Rendering {len(doc)} pages as images")
                for page_num, page in enumerate(doc):
                    try:
                        # Use settings for DPI, with fallback
                        dpi = getattr(self.settings, 'PDF_PREVIEW_RESOLUTION', 150)
                        pix = page.get_pixmap(dpi=dpi)
                        image_bytes = pix.tobytes("png")
                        images.append(image_bytes)
                        logger.debug(f"🖼️ Page {page_num + 1}: rendered {len(image_bytes)} bytes")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to render page {page_num + 1}: {e}")
                        # Continue with other pages
                        continue
            
            logger.info(f"✅ Rendered {len(images)} page images from PDF")
            return images
        except Exception as e:
            logger.error(f"❌ Failed to extract page images: {e}")
            raise

    def _chunk_document(self, text: str, chunk_size: int = 2500) -> List[Dict[str, str]]:
        """
        Break document into numbered chunks with previews for caching.
        """
        try:
            logger.info(f"📑 Chunking document into sections of max {chunk_size} characters")
            
            if not text.strip():
                logger.warning("⚠️ Empty text provided for chunking")
                return [{
                    "chunk_id": 1,
                    "content": "No text content found in document",
                    "preview": "No text content found in document"
                }]
            
            # Split by double newlines first (paragraphs)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                # If no paragraphs, split by single newlines
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            if not paragraphs:
                # If still no paragraphs, use the whole text
                paragraphs = [text.strip()]
            
            chunks = []
            current_chunk = ""
            chunk_id = 1
            
            for paragraph in paragraphs:
                if len(current_chunk + paragraph) > chunk_size and current_chunk:
                    # Save current chunk
                    preview = current_chunk.strip()[:200]
                    if len(current_chunk) > 200:
                        preview += "..."
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": current_chunk.strip(),
                        "preview": preview
                    })
                    
                    # Start new chunk
                    current_chunk = paragraph
                    chunk_id += 1
                else:
                    # Add to current chunk
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
            
            logger.info(f"✅ Document chunked into {len(chunks)} sections")
            
            # Log chunk sizes for debugging
            for chunk in chunks[:3]:  # Log first 3 chunks
                logger.debug(f"📑 Chunk {chunk['chunk_id']}: {len(chunk['content'])} chars - '{chunk['preview'][:50]}...'")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to chunk document: {e}")
            # Return a basic chunk on failure
            return [{
                "chunk_id": 1,
                "content": text[:2500] if text else "Error processing document",
                "preview": (text[:200] + "...") if len(text) > 200 else text
            }]

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
        try:
            logger.info(f"🚀 Starting PDF processing for document '{session_id}'")
            
            # Validate inputs
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")
            
            if not pdf_bytes:
                raise ValueError("PDF bytes cannot be empty")
            
            if not storage_service:
                raise ValueError("Storage service is required")
            
            logger.info(f"📄 Processing PDF: {len(pdf_bytes)} bytes for document '{session_id}'")
            
            # 1. Extract full text
            logger.info("🔍 Step 1: Extracting text from PDF...")
            full_text = self._extract_text_from_pdf(pdf_bytes)
            
            if not full_text.strip():
                logger.warning(f"⚠️ No text extracted from PDF for '{session_id}' - might be image-only")
                full_text = f"Document '{session_id}' processed but no extractable text found. This may be an image-based PDF."
            
            # Save full text context
            text_blob_name = f"{session_id}_context.txt"
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=text_blob_name,
                data=full_text.encode('utf-8')
            )
            logger.info(f"✅ Step 1 complete: Uploaded text context to '{text_blob_name}'")

            # 2. Extract and upload page images
            logger.info("🖼️ Step 2: Extracting page images...")
            page_images = self._extract_page_images(pdf_bytes)
            
            for i, image_bytes in enumerate(page_images):
                image_blob_name = f"{session_id}_page_{i + 1}.png"
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=image_blob_name,
                    data=image_bytes
                )
                logger.debug(f"🖼️ Uploaded page {i + 1} image: {len(image_bytes)} bytes")
            
            logger.info(f"✅ Step 2 complete: Uploaded {len(page_images)} page images")

            # 3. Generate and cache chunks
            logger.info("📑 Step 3: Generating text chunks...")
            chunks = self._chunk_document(full_text)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks generated for '{session_id}', creating default chunk")
                chunks = [{
                    "chunk_id": 1,
                    "content": full_text or f"Document '{session_id}' processed but no content available",
                    "preview": (full_text[:200] + "...") if len(full_text) > 200 else full_text
                }]
            
            chunks_blob_name = f"{session_id}_chunks.json"
            chunks_json = json.dumps(chunks, indent=2, ensure_ascii=False)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name,
                data=chunks_json.encode('utf-8')
            )
            logger.info(f"✅ Step 3 complete: Cached {len(chunks)} chunks to '{chunks_blob_name}'")
            
            # Final verification
            logger.info(f"🎯 PDF processing completed successfully for '{session_id}':")
            logger.info(f"   📄 Text: {len(full_text)} characters")
            logger.info(f"   🖼️ Images: {len(page_images)} pages")
            logger.info(f"   📑 Chunks: {len(chunks)} sections")
            logger.info(f"   🆔 Document ID: '{session_id}' is ready for chat!")
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed for '{session_id}': {e}")
            logger.error(f"❌ Full error details: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")
