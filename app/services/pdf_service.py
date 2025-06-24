# app/services/pdf_service.py - HYBRID TEXT + VISUAL PDF PROCESSING

import logging
import fitz  # PyMuPDF
import json
import os
import asyncio
import gc
from typing import List, Dict, Optional, Any
from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class PDFService:
    """PDF processing service that extracts BOTH text and visual data for comprehensive analysis"""
    
    def __init__(self, settings: AppSettings):
        if not settings:
            raise ValueError("AppSettings instance is required")
        
        self.settings = settings
        self.image_dpi = int(os.getenv("PDF_IMAGE_DPI", "200"))
        self.thumbnail_dpi = int(os.getenv("PDF_THUMBNAIL_DPI", "150"))
        self.max_pages = int(os.getenv("PDF_MAX_PAGES", "100"))
        self.batch_size = int(os.getenv("PROCESSING_BATCH_SIZE", "5"))
        
        logger.info(f"✅ PDFService initialized (Hybrid Mode)")
        logger.info(f"   📄 Max pages: {self.max_pages}")
        logger.info(f"   🖼️ Image DPI: {self.image_dpi}")

    async def process_and_cache_pdf(self, session_id: str, pdf_bytes: bytes, 
                                   storage_service: StorageService):
        """Process PDF extracting both text and visual data"""
        try:
            logger.info(f"🚀 Processing PDF: {session_id}")
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.info(f"📄 Size: {pdf_size_mb:.1f}MB")
            
            # Step 1: Extract comprehensive data
            extracted_data = await self._extract_hybrid_data(pdf_bytes)
            
            # Step 2: Save text content
            await self._save_text_content(
                session_id,
                extracted_data['text_content'],
                extracted_data['structured_text'],
                storage_service
            )
            
            # Step 3: Save visual data
            await self._save_visual_pages(
                session_id,
                extracted_data['pages'],
                storage_service
            )
            
            # Step 4: Save metadata and summaries
            await self._save_metadata(
                session_id,
                extracted_data['metadata'],
                storage_service
            )
            
            # Step 5: Create searchable chunks
            await self._create_smart_chunks(
                session_id,
                extracted_data,
                storage_service
            )
            
            logger.info(f"✅ Processing complete for {session_id}")
            logger.info(f"   📝 Text: {len(extracted_data['text_content'])} chars")
            logger.info(f"   🖼️ Pages: {extracted_data['metadata']['page_count']}")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    async def _extract_hybrid_data(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract both text and visual data from PDF"""
        data = {
            'text_content': '',
            'structured_text': {},
            'pages': [],
            'metadata': {
                'page_count': 0,
                'total_size_mb': 0,
                'has_text': False,
                'has_images': False,
                'drawing_types': set(),
                'document_info': {}
            }
        }
        
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            # Document metadata
            data['metadata']['document_info'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'pages': len(doc)
            }
            
            page_count = min(len(doc), self.max_pages)
            all_text = []
            
            # Process pages in batches
            for batch_start in range(0, page_count, self.batch_size):
                batch_end = min(batch_start + self.batch_size, page_count)
                logger.info(f"📄 Processing pages {batch_start + 1}-{batch_end}")
                
                for page_num in range(batch_start, batch_end):
                    page = doc[page_num]
                    
                    # Extract text
                    page_text = page.get_text()
                    if page_text.strip():
                        data['metadata']['has_text'] = True
                        all_text.append(f"\n--- PAGE {page_num + 1} ---\n{page_text}")
                        
                        # Structured text extraction
                        data['structured_text'][f'page_{page_num + 1}'] = {
                            'text': page_text,
                            'tables': self._extract_tables(page),
                            'annotations': self._extract_annotations(page)
                        }
                    
                    # Extract visual
                    matrix = fitz.Matrix(self.image_dpi / 72, self.image_dpi / 72)
                    pix = page.get_pixmap(matrix=matrix)
                    image_bytes = pix.tobytes("png")
                    
                    # Analyze content
                    drawing_info = self._analyze_page_content(page_text, page_num + 1)
                    if drawing_info['drawing_type']:
                        data['metadata']['drawing_types'].add(drawing_info['drawing_type'])
                    
                    page_data = {
                        'page_num': page_num + 1,
                        'image_bytes': image_bytes,
                        'size_mb': len(image_bytes) / (1024 * 1024),
                        'has_text': bool(page_text.strip()),
                        'text_length': len(page_text),
                        'drawing_info': drawing_info,
                        'dimensions': {'width': pix.width, 'height': pix.height}
                    }
                    
                    data['pages'].append(page_data)
                    data['metadata']['has_images'] = True
                    
                    # Cleanup
                    pix = None
                
                gc.collect()
            
            # Compile results
            data['text_content'] = ''.join(all_text)
            data['metadata']['page_count'] = len(data['pages'])
            data['metadata']['total_size_mb'] = sum(p['size_mb'] for p in data['pages'])
            data['metadata']['drawing_types'] = list(data['metadata']['drawing_types'])
            
        return data

    def _extract_tables(self, page) -> List[Dict]:
        """Extract tables from page"""
        tables = []
        try:
            # PyMuPDF table extraction
            tabs = page.find_tables()
            for tab in tabs:
                tables.append({
                    'rows': tab.extract(),
                    'bbox': tab.bbox
                })
        except:
            pass
        return tables

    def _extract_annotations(self, page) -> List[Dict]:
        """Extract annotations and markup"""
        annotations = []
        try:
            for annot in page.annots():
                annotations.append({
                    'type': annot.type[1],
                    'content': annot.info.get('content', ''),
                    'author': annot.info.get('title', '')
                })
        except:
            pass
        return annotations

    def _analyze_page_content(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze page content to identify drawing type and key information"""
        import re
        
        info = {
            'page_number': page_num,
            'drawing_type': None,
            'title': None,
            'scale': None,
            'sheet_number': None,
            'key_elements': []
        }
        
        text_upper = text.upper()
        
        # Identify drawing type
        drawing_patterns = {
            'floor_plan': ['FLOOR PLAN', 'LEVEL', r'[0-9]+(?:ST|ND|RD|TH)\s+FLOOR'],
            'electrical': ['ELECTRICAL', 'POWER', 'LIGHTING', 'PANEL'],
            'plumbing': ['PLUMBING', 'PIPING', 'DRAINAGE', 'WATER'],
            'mechanical': ['MECHANICAL', 'HVAC', 'DUCTWORK', 'EQUIPMENT'],
            'structural': ['STRUCTURAL', 'FRAMING', 'FOUNDATION', 'BEAM'],
            'detail': ['DETAIL', 'SECTION', 'CONNECTION'],
            'schedule': ['SCHEDULE', 'LEGEND', 'TABLE']
        }
        
        for dtype, patterns in drawing_patterns.items():
            for pattern in patterns:
                if isinstance(pattern, str):
                    if pattern in text_upper:
                        info['drawing_type'] = dtype
                        break
                else:
                    if re.search(pattern, text_upper):
                        info['drawing_type'] = dtype
                        break
        
        # Extract key information
        patterns = {
            'scale': r'SCALE[\s:]+([^\n]+)',
            'sheet_number': r'(?:SHEET|DWG)[\s:#]*([A-Z0-9\-\.]+)',
            'title': r'(?:TITLE|DRAWING)[\s:]+([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text_upper)
            if match:
                info[key] = match.group(1).strip()
        
        # Identify key elements mentioned
        elements = ['DOOR', 'WINDOW', 'ROOM', 'COLUMN', 'BEAM', 'WALL', 
                   'EQUIPMENT', 'FIXTURE', 'PANEL', 'DUCT', 'PIPE']
        info['key_elements'] = [e for e in elements if e in text_upper]
        
        return info

    async def _save_text_content(self, session_id: str, full_text: str, 
                                structured_text: Dict, storage_service: StorageService):
        """Save text content for fast retrieval"""
        # Save full text
        context_blob = f"{session_id}_context.txt"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=context_blob,
            data=full_text.encode('utf-8')
        )
        
        # Save structured text
        structured_blob = f"{session_id}_structured_text.json"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=structured_blob,
            data=json.dumps(structured_text, ensure_ascii=False).encode('utf-8')
        )
        
        logger.info(f"✅ Saved text content ({len(full_text)} chars)")

    async def _save_visual_pages(self, session_id: str, pages: List[Dict],
                                storage_service: StorageService):
        """Save page images efficiently"""
        total = len(pages)
        
        for i in range(0, total, self.batch_size):
            batch = pages[i:i+self.batch_size]
            tasks = []
            
            for page_data in batch:
                blob_name = f"{session_id}_page_{page_data['page_num']}.png"
                task = storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name,
                    data=page_data['image_bytes']
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            logger.info(f"✅ Uploaded pages {i+1}-{min(i+self.batch_size, total)}")

    async def _save_metadata(self, session_id: str, metadata: Dict,
                            storage_service: StorageService):
        """Save comprehensive metadata"""
        # Visual summary for image analysis
        visual_summary = {
            "document_id": session_id,
            "page_count": metadata['page_count'],
            "document_type": "blueprint",
            "drawing_types": metadata['drawing_types'],
            "has_text": metadata['has_text'],
            "has_images": metadata['has_images'],
            "pages": []
        }
        
        # Add page details
        for page in metadata.get('pages', []):
            visual_summary['pages'].append({
                "page_number": page['page_num'],
                "drawing_type": page['drawing_info'].get('drawing_type'),
                "sheet_number": page['drawing_info'].get('sheet_number'),
                "has_text": page['has_text'],
                "key_elements": page['drawing_info'].get('key_elements', [])
            })
        
        # Save visual summary
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_visual_summary.json",
            data=json.dumps(visual_summary, ensure_ascii=False).encode('utf-8')
        )
        
        # Save full metadata
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=f"{session_id}_metadata.json",
            data=json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        )

    async def _create_smart_chunks(self, session_id: str, data: Dict,
                                  storage_service: StorageService):
        """Create intelligent chunks for fast retrieval"""
        chunks = []
        
        # Create chunks by page with context
        for page_data in data['pages']:
            page_num = page_data['page_num']
            page_text = data['structured_text'].get(f'page_{page_num}', {}).get('text', '')
            
            if page_text:
                chunk = {
                    "chunk_id": page_num,
                    "page_number": page_num,
                    "content": page_text[:2500],  # Limit chunk size
                    "drawing_type": page_data['drawing_info'].get('drawing_type'),
                    "key_elements": page_data['drawing_info'].get('key_elements', []),
                    "has_image": True,
                    "preview": page_text[:200] + "..." if len(page_text) > 200 else page_text
                }
                chunks.append(chunk)
        
        # Add a summary chunk
        summary_chunk = {
            "chunk_id": 0,
            "page_number": 0,
            "content": f"Document contains {data['metadata']['page_count']} pages. Types: {', '.join(data['metadata']['drawing_types'])}",
            "drawing_type": "summary",
            "preview": "Document summary"
        }
        chunks.insert(0, summary_chunk)
        
        # Save chunks
        chunks_blob = f"{session_id}_chunks.json"
        await storage_service.upload_file(
            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
            blob_name=chunks_blob,
            data=json.dumps(chunks, ensure_ascii=False).encode('utf-8')
        )
        
        logger.info(f"✅ Created {len(chunks)} smart chunks")

    # Keep legacy methods for compatibility
    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Legacy text extraction"""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except:
            return ""

    def _extract_page_images(self, pdf_bytes: bytes) -> List[bytes]:
        """Legacy image extraction"""
        images = []
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap()
                    images.append(pix.tobytes("png"))
        except:
            pass
        return images

    def _chunk_document(self, text: str, chunk_size: int = 2500) -> List[Dict[str, str]]:
        """Legacy chunking"""
        if not text:
            return [{"chunk_id": 1, "content": "No text content", "preview": "No text"}]
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "content": chunk_text,
                "preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "type": "text"
            })
        return chunks

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "PDFService",
            "mode": "hybrid_processing",
            "capabilities": {
                "text_extraction": True,
                "visual_extraction": True,
                "table_extraction": True,
                "smart_chunking": True,
                "max_pages": self.max_pages
            },
            "configuration": {
                "image_dpi": self.image_dpi,
                "batch_size": self.batch_size
            }
        }

    def validate_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Validate PDF"""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                return {
                    "valid": True,
                    "page_count": len(doc),
                    "size_mb": round(len(pdf_bytes) / (1024 * 1024), 2),
                    "title": doc.metadata.get("title", "")
                }
        except Exception as e:
            return {"valid": False, "error": str(e)}
