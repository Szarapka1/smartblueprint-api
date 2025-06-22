from PyPDF2 import PdfReader
import json
from io import BytesIO
from datetime import datetime

class PDFService:
    def __init__(self, logger=None):
        self.logger = logger

    async def process_and_cache_pdf(self, session_id, pdf_bytes, storage_service):
        try:
            pdf_reader = PdfReader(BytesIO(pdf_bytes))
            chunks = []
            full_context = ""

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text() or ""
                chunk = {
                    "page_number": page_number,
                    "text": text.strip(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                chunks.append(chunk)
                full_context += f"\n\n--- Page {page_number} ---\n{text}"

            # Save _chunks.json
            await storage_service.upload_file(
                container_name="smartblueprint-cache",  # or use settings.AZURE_CACHE_CONTAINER_NAME
                blob_name=f"{session_id}_chunks.json",
                data=json.dumps(chunks, ensure_ascii=False, indent=2).encode("utf-8")
            )

            # Save _context.txt
            await storage_service.upload_file(
                container_name="smartblueprint-cache",
                blob_name=f"{session_id}_context.txt",
                data=full_context.encode("utf-8")
            )

            # Optionally store metadata
            metadata = {
                "total_pages": len(chunks),
                "session_id": session_id,
                "processed_at": datetime.utcnow().isoformat()
            }
            await storage_service.upload_file(
                container_name="smartblueprint-cache",
                blob_name=f"{session_id}_meta.json",
                data=json.dumps(metadata).encode("utf-8")
            )

            if self.logger:
                self.logger.info(f"[PDF] Processed and cached: {session_id} with {len(chunks)} pages")

        except Exception as e:
            if self.logger:
                self.logger.error(f"[PDF ERROR] Failed to process: {e}")
            raise
