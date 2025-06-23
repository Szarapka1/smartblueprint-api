# app/services/pdf_service.py - COMPLETE FIXED VERSION

import logging
import fitz  # PyMuPDF library
import json 
import os
from typing import List, Dict, Optional
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class PDFService:
    """PDF processing service with high-resolution image generation and text extraction"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        
        # Set high DPI for technical drawings - configurable via environment
        self.image_dpi = int(os.getenv("PDF_IMAGE_DPI", "300"))  # Default 300 DPI
        self.thumbnail_dpi = int(os.getenv("PDF_THUMBNAIL_DPI", "150"))  # For thumbnails
        self.max_pages = int(os.getenv("PDF_MAX_PAGES", "20"))  # Limit pages to process
        
        logger.info(f"✅ PDFService initialized:")
        logger.info(f"   🖼️ Analysis DPI: {self.image_dpi}")
        logger.info(f"   📸 Thumbnail DPI: {self.thumbnail_dpi}")
        logger.info(f"   📄 Max pages: {self.max_pages}")

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract all text from PDF for analysis"""
        try:
            full_text = ""
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                logger.info(f"📄 Processing PDF with {len(doc)} pages")
                
                # Limit pages to process to avoid timeouts
                pages_to_process = min(len(doc), self.max_pages)
                if len(doc) > self.max_pages:
                    logger.warning(f"⚠️ PDF has {len(doc)} pages, processing only first {self.max_pages}")
                
                for page_num in range(pages_to_process):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        
                        if page_text.strip():  # Only add if page has text
                            full_text += f"\n--- PAGE {page_num + 1} ---\n"
                            full_text += page_text + "\n"
                            logger.debug(f"📄 Page {page_num + 1}: extracted {len(page_text)} characters")
                        else:
                            logger.debug(f"📄 Page {page_num + 1}: no text found (may be image-based)")
                    
                    except Exception as page_error:
                        logger.warning(f"⚠️ Failed to extract text from page {page_num + 1}: {page_error}")
                        continue
            
            logger.info(f"✅ Extracted {len(full_text):,} total characters from {pages_to_process} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF: {e}")
            raise RuntimeError(f"PDF text extraction failed: {e}")

    def _extract_page_images(self, pdf_bytes: bytes) -> List[bytes]:
        """Render PDF pages as high-resolution PNG images for AI analysis"""
        try:
            images = []
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                pages_to_process = min(len(doc), self.max_pages)
                logger.info(f"🖼️ Rendering {pages_to_process} pages as HIGH RESOLUTION images ({self.image_dpi} DPI)")
                
                for page_num in range(pages_to_process):
                    try:
                        page = doc[page_num]
                        logger.info(f"🖼️ Rendering page {page_num + 1} at {self.image_dpi} DPI...")
                        
                        # Calculate transformation matrix for high DPI
                        matrix = fitz.Matrix(self.image_dpi / 72, self.image_dpi / 72)  # DPI conversion
                        
                        # Get pixmap with high DPI
                        pix = page.get_pixmap(matrix=matrix)
                        
                        # Convert to PNG bytes
                        image_bytes = pix.tobytes("png")
                        images.append(image_bytes)
                        
                        # Validate and log image details
                        estimated_size_mb = len(image_bytes) / (1024 * 1024)
                        is_valid_png = image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
                        
                        logger.info(f"🖼️ Page {page_num + 1}: {estimated_size_mb:.1f}MB at {self.image_dpi}DPI ({pix.width}x{pix.height}px)")
                        
                        if not is_valid_png:
                            logger.error(f"❌ Page {page_num + 1} generated invalid PNG!")
                            images.pop()  # Remove invalid image
                            continue
                        
                        # Clean up memory
                        pix = None
                        
                    except Exception as page_error:
                        logger.error(f"❌ Failed to render page {page_num + 1}: {page_error}")
                        continue  # Continue with other pages
            
            total_size_mb = sum(len(img) for img in images) / (1024 * 1024)
            logger.info(f"✅ Rendered {len(images)} HIGH-RES page images (Total: {total_size_mb:.1f}MB)")
            
            # Warn if images are very large
            if total_size_mb > 100:
                logger.warning(f"⚠️ Large image cache: {total_size_mb:.1f}MB. Consider reducing DPI if storage is a concern.")
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Failed to extract page images: {e}")
            raise RuntimeError(f"PDF image extraction failed: {e}")

    def _create_thumbnail_images(self, pdf_bytes: bytes) -> List[bytes]:
        """Create lower resolution thumbnails for UI/preview"""
        try:
            thumbnails = []
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                pages_to_process = min(len(doc), self.max_pages)
                logger.info(f"🖼️ Creating thumbnails at {self.thumbnail_dpi} DPI for UI")
                
                for page_num in range(pages_to_process):
                    try:
                        page = doc[page_num]
                        
                        # Lower DPI for thumbnails/UI display
                        matrix = fitz.Matrix(self.thumbnail_dpi / 72, self.thumbnail_dpi / 72)
                        pix = page.get_pixmap(matrix=matrix)
                        image_bytes = pix.tobytes("png")
                        thumbnails.append(image_bytes)
                        
                        logger.debug(f"📸 Thumbnail {page_num + 1}: {len(image_bytes)/1024:.0f}KB")
                        pix = None
                        
                    except Exception as page_error:
                        logger.warning(f"⚠️ Failed to create thumbnail for page {page_num + 1}: {page_error}")
                        continue
            
            logger.info(f"✅ Created {len(thumbnails)} thumbnail images")
            return thumbnails
            
        except Exception as e:
            logger.error(f"❌ Failed to create thumbnails: {e}")
            return []

    def _chunk_document(self, text: str, chunk_size: int = 2500) -> List[Dict[str, str]]:
        """Break document into logical chunks for better AI processing"""
        try:
            logger.info(f"📑 Chunking document into sections of max {chunk_size} characters")
            
            if not text.strip():
                logger.warning("⚠️ Empty text provided for chunking")
                return [{
                    "chunk_id": 1,
                    "content": "No text content found in document",
                    "preview": "No text content found in document",
                    "type": "empty"
                }]
            
            # Split by page markers first, then by paragraphs
            page_sections = text.split('--- PAGE')
            chunks = []
            chunk_id = 1
            
            for i, section in enumerate(page_sections):
                if not section.strip():
                    continue
                
                # Restore page marker if it was split
                if i > 0:
                    section = '--- PAGE' + section
                
                # If section is small enough, use as-is
                if len(section) <= chunk_size:
                    preview = section.strip()[:200]
                    if len(section) > 200:
                        preview += "..."
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": section.strip(),
                        "preview": preview,
                        "type": "page_section" if "PAGE" in section else "content"
                    })
                    chunk_id += 1
                else:
                    # Break large sections into smaller chunks
                    paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        if len(current_chunk + paragraph) > chunk_size and current_chunk:
                            # Save current chunk
                            preview = current_chunk.strip()[:200]
                            if len(current_chunk) > 200:
                                preview += "..."
                            
                            chunks.append({
                                "chunk_id": chunk_id,
                                "content": current_chunk.strip(),
                                "preview": preview,
                                "type": "content_chunk"
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
                    
                    # Add the final chunk from this section
                    if current_chunk.strip():
                        preview = current_chunk.strip()[:200]
                        if len(current_chunk) > 200:
                            preview += "..."
                            
                        chunks.append({
                            "chunk_id": chunk_id,
                            "content": current_chunk.strip(),
                            "preview": preview,
                            "type": "content_chunk"
                        })
                        chunk_id += 1
            
            logger.info(f"✅ Document chunked into {len(chunks)} logical sections")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to chunk document: {e}")
            # Return safe fallback
            return [{
                "chunk_id": 1,
                "content": text[:2500] if text else "Error processing document",
                "preview": (text[:200] + "...") if len(text) > 200 else text,
                "type": "fallback"
            }]

    async def process_and_cache_pdf(
        self,
        session_id: str,
        pdf_bytes: bytes,
        storage_service: StorageService
    ):
        """
        Complete PDF processing pipeline:
        1. Extract text content
        2. Generate high-resolution page images
        3. Create text chunks for AI processing
        4. Cache everything to Azure storage
        """
        try:
            logger.info(f"🚀 Starting PDF processing for document '{session_id}'")
            
            # Input validation
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")
            
            if not pdf_bytes or len(pdf_bytes) == 0:
                raise ValueError("PDF bytes cannot be empty")
            
            if not storage_service:
                raise ValueError("Storage service is required")
            
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 Processing PDF: {pdf_size_mb:.1f}MB for document '{session_id}'")
            
            # Step 1: Extract full text
            logger.info("🔍 Step 1: Extracting text from PDF...")
            try:
                full_text = self._extract_text_from_pdf(pdf_bytes)
            except Exception as e:
                logger.error(f"❌ Text extraction failed: {e}")
                full_text = f"Document '{session_id}' processed but text extraction failed: {str(e)}"
            
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
            logger.info(f"✅ Step 1 complete: Uploaded text context ({len(full_text):,} chars)")

            # Step 2: Extract HIGH-RESOLUTION page images
            logger.info(f"🖼️ Step 2: Extracting HIGH-RESOLUTION page images ({self.image_dpi} DPI)...")
            try:
                page_images = self._extract_page_images(pdf_bytes)
            except Exception as e:
                logger.error(f"❌ Image extraction failed: {e}")
                page_images = []
            
            uploaded_images = 0
            for i, image_bytes in enumerate(page_images):
                try:
                    image_blob_name = f"{session_id}_page_{i + 1}.png"
                    
                    # Validate image before upload
                    if not image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                        logger.error(f"❌ Invalid PNG for page {i + 1}, skipping upload")
                        continue
                    
                    await storage_service.upload_file(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=image_blob_name,
                        data=image_bytes
                    )
                    
                    size_mb = len(image_bytes) / (1024 * 1024)
                    logger.info(f"🖼️ ✅ Uploaded HIGH-RES page {i + 1}: {size_mb:.1f}MB")
                    uploaded_images += 1
                    
                except Exception as upload_error:
                    logger.error(f"❌ Failed to upload page {i + 1}: {upload_error}")
                    continue
            
            logger.info(f"✅ Step 2 complete: Uploaded {uploaded_images}/{len(page_images)} HIGH-RESOLUTION images")

            # Step 3: Generate and cache text chunks
            logger.info("📑 Step 3: Generating text chunks...")
            try:
                chunks = self._chunk_document(full_text)
            except Exception as e:
                logger.error(f"❌ Chunking failed: {e}")
                chunks = [{
                    "chunk_id": 1,
                    "content": full_text or f"Document '{session_id}' processed but chunking failed",
                    "preview": (full_text[:200] + "...") if len(full_text) > 200 else full_text,
                    "type": "fallback"
                }]
            
            if not chunks:
                logger.warning(f"⚠️ No chunks generated for '{session_id}', creating default chunk")
                chunks = [{
                    "chunk_id": 1,
                    "content": full_text or f"Document '{session_id}' processed but no content available",
                    "preview": (full_text[:200] + "...") if len(full_text) > 200 else full_text,
                    "type": "default"
                }]
            
            chunks_blob_name = f"{session_id}_chunks.json"
            chunks_json = json.dumps(chunks, indent=2, ensure_ascii=False)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name,
                data=chunks_json.encode('utf-8')
            )
            logger.info(f"✅ Step 3 complete: Cached {len(chunks)} chunks")
            
            # Optional Step 4: Create UI thumbnails if enabled
            create_thumbnails = os.getenv("CREATE_THUMBNAILS", "false").lower() == "true"
            thumbnail_count = 0
            
            if create_thumbnails:
                logger.info(f"🖼️ Step 4: Creating UI thumbnails ({self.thumbnail_dpi} DPI)...")
                try:
                    thumbnails = self._create_thumbnail_images(pdf_bytes)
                    
                    for i, thumb_bytes in enumerate(thumbnails):
                        try:
                            thumb_blob_name = f"{session_id}_thumb_{i + 1}.png"
                            await storage_service.upload_file(
                                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                                blob_name=thumb_blob_name,
                                data=thumb_bytes
                            )
                            thumbnail_count += 1
                        except Exception as thumb_error:
                            logger.warning(f"⚠️ Failed to upload thumbnail {i + 1}: {thumb_error}")
                    
                    logger.info(f"✅ Step 4 complete: Created {thumbnail_count} thumbnails")
                except Exception as e:
                    logger.warning(f"⚠️ Thumbnail creation failed: {e}")
            
            # Step 5: Create processing summary
            processing_summary = {
                "document_id": session_id,
                "processing_timestamp": logger.handlers[0].formatter.formatTime(logging.LogRecord('', 0, '', 0, '', (), None), '%Y-%m-%d %H:%M:%S') if logger.handlers else "unknown",
                "original_size_mb": round(pdf_size_mb, 2),
                "text_length": len(full_text),
                "pages_processed": uploaded_images,
                "chunks_created": len(chunks),
                "thumbnails_created": thumbnail_count,
                "image_dpi": self.image_dpi,
                "processing_status": "complete"
            }
            
            summary_blob_name = f"{session_id}_processing_summary.json"
            summary_json = json.dumps(processing_summary, indent=2, ensure_ascii=False)
            await storage_service.upload_file(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=summary_blob_name,
                data=summary_json.encode('utf-8')
            )
            
            # Final status log
            total_storage_mb = sum(len(img) for img in page_images) / (1024 * 1024)
            logger.info(f"🎯 PDF processing completed for '{session_id}':")
            logger.info(f"   📄 Text: {len(full_text):,} characters")
            logger.info(f"   🖼️ HIGH-RES Images: {uploaded_images} pages ({total_storage_mb:.1f}MB)")
            logger.info(f"   📑 Chunks: {len(chunks)} sections")
            logger.info(f"   📸 Thumbnails: {thumbnail_count}")
            logger.info(f"   🔍 DPI: {self.image_dpi} (optimized for AI analysis)")
            logger.info(f"   🆔 Document '{session_id}' is ready for AI analysis!")
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed for '{session_id}': {e}")
            logger.error(f"❌ Full error details: {str(e)}")
            
            # Try to save error summary
            try:
                error_summary = {
                    "document_id": session_id,
                    "processing_timestamp": logger.handlers[0].formatter.formatTime(logging.LogRecord('', 0, '', 0, '', (), None), '%Y-%m-%d %H:%M:%S') if logger.handlers else "unknown",
                    "processing_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                error_blob_name = f"{session_id}_processing_error.json"
                error_json = json.dumps(error_summary, indent=2, ensure_ascii=False)
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=error_blob_name,
                    data=error_json.encode('utf-8')
                )
            except:
                pass  # Don't fail if we can't save error summary
            
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    def get_processing_stats(self) -> Dict[str, any]:
        """Get current processing configuration and stats"""
        return {
            "service_name": "PDFService",
            "version": "2.1.0",
            "configuration": {
                "image_dpi": self.image_dpi,
                "thumbnail_dpi": self.thumbnail_dpi,
                "max_pages": self.max_pages
            },
            "capabilities": {
                "text_extraction": True,
                "high_res_images": True,
                "thumbnail_creation": True,
                "intelligent_chunking": True,
                "multi_page_support": True
            },
            "recommended_settings": {
                "blueprints_dpi": 300,
                "documents_dpi": 200,
                "thumbnails_dpi": 150,
                "max_recommended_dpi": 600
            }
        }

    def validate_pdf(self, pdf_bytes: bytes) -> Dict[str, any]:
        """Validate PDF file and return information"""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                return {
                    "valid": True,
                    "page_count": len(doc),
                    "size_bytes": len(pdf_bytes),
                    "size_mb": round(len(pdf_bytes) / (1024 * 1024), 2),
                    "encrypted": doc.needs_pass,
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "creator": doc.metadata.get("creator", "")
                }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "size_bytes": len(pdf_bytes) if pdf_bytes else 0
            }
