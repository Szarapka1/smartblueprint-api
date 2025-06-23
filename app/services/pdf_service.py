# app/services/pdf_service.py - Enhanced with High Resolution Images and Debugging
import logging
import fitz  # PyMuPDF library
import json 
import os
from typing import List, Dict
from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        # Set high DPI for technical drawings - configurable via environment
        self.image_dpi = int(os.getenv("PDF_IMAGE_DPI", "300"))  # Default 300 DPI
        self.thumbnail_dpi = int(os.getenv("PDF_THUMBNAIL_DPI", "150"))  # For thumbnails if needed
        logger.info(f"✅ PDFService initialized with {self.image_dpi} DPI for analysis images")

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
        """Renders each page of a PDF into high-resolution PNG images for AI analysis."""
        try:
            images = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                logger.info(f"🖼️ Rendering {len(doc)} pages as HIGH RESOLUTION images ({self.image_dpi} DPI)")
                
                for page_num, page in enumerate(doc):
                    try:
                        logger.info(f"🖼️ Rendering page {page_num + 1} at {self.image_dpi} DPI...")
                        
                        # Get pixmap with high DPI
                        matrix = fitz.Matrix(self.image_dpi / 72, self.image_dpi / 72)  # DPI conversion
                        pix = page.get_pixmap(matrix=matrix)
                        
                        # Convert to PNG bytes
                        image_bytes = pix.tobytes("png")
                        images.append(image_bytes)
                        
                        # Log the size and validate PNG
                        estimated_size_mb = len(image_bytes) / (1024 * 1024)
                        is_valid_png = image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
                        
                        logger.info(f"🖼️ Page {page_num + 1}: {estimated_size_mb:.1f}MB at {self.image_dpi}DPI ({pix.width}x{pix.height}px)")
                        logger.info(f"🖼️ Page {page_num + 1}: Valid PNG = {is_valid_png}")
                        
                        if not is_valid_png:
                            logger.warning(f"⚠️ Page {page_num + 1} may not be a valid PNG!")
                        
                        # Clean up memory
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to render page {page_num + 1}: {e}")
                        # Continue with other pages
                        continue
            
            total_size_mb = sum(len(img) for img in images) / (1024 * 1024)
            logger.info(f"✅ Rendered {len(images)} HIGH-RES page images (Total: {total_size_mb:.1f}MB)")
            
            # Warn if images are very large
            if total_size_mb > 100:
                logger.warning(f"⚠️ Large image cache: {total_size_mb:.1f}MB. Consider reducing DPI if storage is a concern.")
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Failed to extract page images: {e}")
            raise

    def _create_thumbnail_images(self, pdf_bytes: bytes) -> List[bytes]:
        """Create lower resolution thumbnail images for UI/preview purposes."""
        try:
            thumbnails = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                logger.info(f"🖼️ Creating thumbnails at {self.thumbnail_dpi} DPI for UI")
                
                for page_num, page in enumerate(doc):
                    try:
                        # Lower DPI for thumbnails/UI display
                        matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
                        pix = page.get_pixmap(matrix=matrix)
                        image_bytes = pix.tobytes("png")
                        thumbnails.append(image_bytes)
                        
                        logger.debug(f"📸 Thumbnail {page_num + 1}: {len(image_bytes)/1024:.0f}KB")
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to create thumbnail for page {page_num + 1}: {e}")
                        continue
            
            logger.info(f"✅ Created {len(thumbnails)} thumbnail images")
            return thumbnails
            
        except Exception as e:
            logger.error(f"❌ Failed to create thumbnails: {e}")
            return []

    def _chunk_document(self, text: str, chunk_size: int = 2500) -> List[Dict[str, str]]:
        """Break document into numbered chunks with previews for caching."""
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
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            if not paragraphs:
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
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to chunk document: {e}")
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
        Process PDF and save high-resolution images for AI analysis + optional thumbnails.
        """
        try:
            logger.info(f"🚀 Starting HIGH-RESOLUTION PDF processing for document '{session_id}'")
            
            # Validate inputs
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")
            
            if not pdf_bytes:
                raise ValueError("PDF bytes cannot be empty")
            
            if not storage_service:
                raise ValueError("Storage service is required")
            
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 Processing PDF: {pdf_size_mb:.1f}MB for document '{session_id}' at {self.image_dpi} DPI")
            
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
            logger.info(f"✅ Step 1 complete: Uploaded text context to {text_blob_name}")

            # 2. Extract HIGH-RESOLUTION page images for AI analysis
            logger.info(f"🖼️ Step 2: Extracting HIGH-RESOLUTION page images ({self.image_dpi} DPI)...")
            page_images = self._extract_page_images(pdf_bytes)
            
            uploaded_images = 0
            for i, image_bytes in enumerate(page_images):
                try:
                    image_blob_name = f"{session_id}_page_{i + 1}.png"
                    
                    # Validate image before upload
                    is_valid_png = image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
                    if not is_valid_png:
                        logger.error(f"❌ Invalid PNG for page {i + 1}, skipping upload")
                        continue
                    
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=image_blob_name,
                        data=image_bytes
                    )
                    
                    size_mb = len(image_bytes) / (1024 * 1024)
                    logger.info(f"🖼️ ✅ Uploaded HIGH-RES page {i + 1}: {size_mb:.1f}MB -> {image_blob_name}")
                    uploaded_images += 1
                    
                except Exception as upload_error:
                    logger.error(f"❌ Failed to upload page {i + 1}: {upload_error}")
                    continue
            
            logger.info(f"✅ Step 2 complete: Uploaded {uploaded_images}/{len(page_images)} HIGH-RESOLUTION images")

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
            logger.info(f"✅ Step 3 complete: Cached {len(chunks)} chunks to {chunks_blob_name}")
            
            # Optional Step 4: Create lower-res thumbnails for UI if needed
            create_thumbnails = os.getenv("CREATE_THUMBNAILS", "false").lower() == "true"
            if create_thumbnails:
                logger.info(f"🖼️ Step 4: Creating UI thumbnails ({self.thumbnail_dpi} DPI)...")
                thumbnails = self._create_thumbnail_images(pdf_bytes)
                
                for i, thumb_bytes in enumerate(thumbnails):
                    thumb_blob_name = f"{session_id}_thumb_{i + 1}.png"
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=thumb_blob_name,
                        data=thumb_bytes
                    )
                
                logger.info(f"✅ Step 4 complete: Created {len(thumbnails)} thumbnails")
            
            # Final status
            total_storage_mb = sum(len(img) for img in page_images) / (1024 * 1024)
            logger.info(f"🎯 HIGH-RESOLUTION PDF processing completed for '{session_id}':")
            logger.info(f"   📄 Text: {len(full_text)} characters")
            logger.info(f"   🖼️ HIGH-RES Images: {uploaded_images} pages ({total_storage_mb:.1f}MB)")
            logger.info(f"   📑 Chunks: {len(chunks)} sections")
            logger.info(f"   🔍 DPI: {self.image_dpi} (optimized for symbol recognition)")
            logger.info(f"   🆔 Document ID: '{session_id}' is ready for VISUAL AI analysis!")
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed for '{session_id}': {e}")
            logger.error(f"❌ Full error details: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")

    def get_recommended_dpi_settings(self) -> Dict[str, int]:
        """Return recommended DPI settings for different use cases"""
        return {
            "analysis_dpi": self.image_dpi,
            "thumbnail_dpi": self.thumbnail_dpi,
            "recommended_for_blueprints": 300,
            "recommended_for_text_docs": 200,
            "recommended_for_thumbnails": 150,
            "maximum_recommended": 600  # Beyond this, file sizes become very large
        }
