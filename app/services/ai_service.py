# app/services/ai_service.py

import logging
import base64
import re
import json
from typing import Optional, List, Dict
from openai import OpenAI, APIError

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        try:
            self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
            logger.info("✅ AIService initialized with OpenAI.")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize OpenAI client: {e}")
            raise

    # Removed _chunk_document from here as it's now in pdf_service.py
    # and called during initial processing for consistency.
    # The _load_or_create_chunks will now strictly load, not create.

    async def _load_chunks(self, document_id: str, storage_service: StorageService) -> List[Dict[str, str]]:
        """
        Load cached chunks from blob storage. Assumes chunks are already created during PDF processing.
        """
        chunks_blob_name = f"{document_id}_chunks.json"
        
        try:
            # Try to load existing cached chunks
            logger.info(f"🔍 Loading cached chunks: {chunks_blob_name}")
            chunks_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name
            )
            chunks = json.loads(chunks_data)
            logger.info(f"📋 ✅ Loaded {len(chunks)} cached chunks for document '{document_id}'")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to load chunks for document '{document_id}'. They might not have been created during upload or there's a storage issue: {e}")
            raise # Re-raise if chunks are essential and not found/loaded

    async def get_ai_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        page_number: Optional[int] = None,
        author: Optional[str] = None
    ) -> str:
        """
        Generate AI response for a shared document that everyone can access.
        """
        logger.info(f"🧠 Generating AI response for shared document '{document_id}'" + (f" by {author}" if author else ""))
        logger.info("📋 Using SHARED CACHED TWO-PASS approach")

        try:
            # Step 1: Load cached chunks for this shared document (they should already exist from PDFService)
            chunks = await self._load_chunks(document_id, storage_service)
            
            # Step 2: PASS 1 - AI identifies relevant sections using cached previews
            logger.info("🔍 PASS 1: AI identifying relevant sections from shared cached data")
            relevant_chunk_ids = await self._identify_relevant_sections(prompt, chunks, document_id)
            
            # Step 3: PASS 2 - AI answers using focused sections
            logger.info(f"🎯 PASS 2: AI answering using shared cached sections {relevant_chunk_ids}")
            return await self._get_focused_response(
                prompt, chunks, relevant_chunk_ids, storage_service, document_id, page_number, author
            )

        except Exception as e:
            logger.error(f"❌ Error in shared document AI response for '{document_id}': {e}")
            return f"An error occurred while analyzing the shared document: {str(e)}"

    async def _identify_relevant_sections(self, prompt: str, chunks: List[Dict[str, str]], document_id: str) -> List[int]:
        """
        PASS 1: AI reads cached chunk previews and identifies which sections are most relevant.
        """
        logger.info(f"🔍 Analyzing {len(chunks)} cached sections for document '{document_id}' to find relevance")
        
        # Create section summaries from cached previews
        chunk_summaries = []
        for chunk in chunks:
            chunk_summaries.append(f"SECTION {chunk['chunk_id']}: {chunk['preview']}")
        
        all_summaries = "\n\n".join(chunk_summaries)
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a construction document analyst. Your job is to identify which sections "
                            "of a blueprint document are most relevant to answer a specific question.\n\n"
                            "You will be shown previews of all document sections. Identify the 3-5 most "
                            "relevant section numbers that would help answer the user's question.\n\n"
                            "Respond with ONLY the section numbers, separated by commas. For example: 2,7,12,15\n"
                            "If you're unsure, include a few extra sections rather than missing important ones."
                        )
                    },
                    {
                        "role": "user", 
                        "content": (
                            f"QUESTION TO ANSWER: {prompt}\n\n"
                            f"DOCUMENT SECTION PREVIEWS:\n{all_summaries}\n\n"
                            f"Which section numbers (3-5 max) are most relevant to answer this question?"
                        )
                    }
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            # Parse the AI response to extract section numbers
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"🔍 AI section analysis result for '{document_id}': '{ai_response}'")
            
            # Extract numbers from the response
            section_numbers = []
            for num_str in re.findall(r'\d+', ai_response):
                section_num = int(num_str)
                if 1 <= section_num <= len(chunks):
                    section_numbers.append(section_num)
            
            # Ensure we have reasonable number of sections
            if not section_numbers:
                logger.warning(f"⚠️ No valid sections identified for '{document_id}', using default sections 1-3")
                section_numbers = [1, 2, 3]
            elif len(section_numbers) > 5:
                logger.info(f"📊 Too many sections identified ({len(section_numbers)}) for '{document_id}', limiting to first 5")
                section_numbers = section_numbers[:5]
            
            logger.info(f"✅ Selected sections for '{document_id}': {section_numbers}")
            return section_numbers
            
        except APIError as e:
            logger.error(f"🚫 OpenAI API error in section identification for '{document_id}': {e}")
            return [1, 2, 3]  # Fallback
        except Exception as e:
            logger.error(f"❌ Error in section identification for '{document_id}': {e}")
            return [1, 2, 3]  # Fallback

    async def _get_focused_response(
        self,
        prompt: str,
        chunks: List[Dict[str, str]],
        relevant_chunk_ids: List[int],
        storage_service: StorageService,
        document_id: str,
        page_number: Optional[int] = None,
        author: Optional[str] = None
    ) -> str:
        """
        PASS 2: AI answers the question using only the identified relevant sections from shared cache.
        """
        logger.info(f"🎯 Building focused response for '{document_id}' using sections: {relevant_chunk_ids}")
        
        # Gather the relevant sections from cached chunks
        relevant_sections = []
        total_focused_chars = 0
        
        for chunk_id in relevant_chunk_ids:
            # Find the chunk with this ID
            for chunk in chunks:
                if chunk['chunk_id'] == chunk_id:
                    section_content = f"=== SECTION {chunk_id} ===\n{chunk['content']}"
                    relevant_sections.append(section_content)
                    total_focused_chars += len(chunk['content'])
                    logger.info(f"📄 Including Section {chunk_id}: {len(chunk['content'])} chars")
                    break
        
        # Combine all relevant sections
        focused_content = "\n\n".join(relevant_sections)
        
        # Calculate token savings for logging
        total_document_chars = sum(len(chunk['content']) for chunk in chunks)
        token_savings_percent = ((total_document_chars - total_focused_chars) / total_document_chars) * 100
        estimated_tokens = total_focused_chars // 4  # Rough token estimation
        
        logger.info(f"📊 Token optimization summary for '{document_id}':")
        logger.info(f"   - Full document: {total_document_chars} chars (~{total_document_chars//4} tokens)")
        logger.info(f"   - Focused content: {total_focused_chars} chars (~{estimated_tokens} tokens)")
        logger.info(f"   - Savings: {token_savings_percent:.1f}% token reduction")
        
        # Build the user message content
        user_content = [
            {
                "type": "text",
                "text": (
                    "You are an experienced construction superintendent reading blueprint documents. "
                    "Answer questions directly using the specific information provided in the document sections.\n\n"
                    f"DOCUMENT SECTIONS:\n\n{focused_content}\n\n"
                    f"QUESTION: {prompt}\n\n"
                    "INSTRUCTIONS:\n"
                    "- Give direct, specific answers using the exact information from the sections above\n"
                    "- Quote specific measurements, materials, quantities, and specifications when available\n"
                    "- If you see window schedules, door schedules, or quantity lists - state the exact numbers\n"
                    "- Don't hedge or say 'not specified' unless truly missing\n"
                    "- Be concise and factual - no unnecessary analysis or recommendations\n"
                    "- Answer like you're reading directly from the plans"
                )
            }
        ]

        # Add page image if provided
        if page_number is not None:
            image_blob_name = f"{document_id}_page_{page_number}.png"
            try:
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=image_blob_name
                )
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })
                logger.info(f"🖼️ Enhanced response with image from page {page_number} of '{document_id}'")
            except Exception as e:
                logger.warning(f"⚠️ Could not load image for page {page_number} of '{document_id}': {e}")

        try:
            # Call OpenAI with the focused content
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an experienced construction superintendent who reads blueprints daily. "
                            "Read the document sections provided and answer questions with specific facts and numbers. "
                            "Give direct answers like you're looking right at the plans - no hedging or analysis."
                        )
                    },
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ],
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS
            )

            ai_message = response.choices[0].message.content
            logger.info(f"✅ Focused AI response received for shared document '{document_id}'" + (f" (requested by {author})" if author else ""))
            
            if ai_message:
                return ai_message.strip()
            else:
                return "The AI returned an empty response. Please try rephrasing your question."

        except APIError as e:
            logger.error(f"🚫 OpenAI API error in focused response for '{document_id}': {e}")
            return f"OpenAI API error: {e}"
        except Exception as e:
            logger.error(f"❌ Unexpected error in focused response for '{document_id}': {e}")
            return f"An error occurred while generating your answer: {str(e)}"

    async def get_document_info(self, document_id: str, storage_service: StorageService) -> Dict[str, any]:
        """
        Get information about a shared document including chunk count and processing status.
        """
        try:
            # Check if chunks exist (document is processed)
            chunks_blob_name = f"{document_id}_chunks.json"
            chunks_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name
            )
            chunks = json.loads(chunks_data)
            
            # Get full text info
            context_blob_name = f"{document_id}_context.txt"
            full_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=context_blob_name
            )
            
            return {
                "document_id": document_id,
                "status": "ready",
                "total_chunks": len(chunks),
                "total_characters": len(full_text),
                "estimated_tokens": len(full_text) // 4,
                "last_processed": "Available"  # Could add timestamp metadata later
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting document info for '{document_id}': {e}")
            return {
                "document_id": document_id,
                "status": "not_found",
                "error": str(e)
            }
