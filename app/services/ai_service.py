# app/services/ai_service.py - Enhanced with Visual Analysis

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

    async def _load_chunks(self, document_id: str, storage_service: StorageService) -> List[Dict[str, str]]:
        """Load cached chunks from blob storage."""
        chunks_blob_name = f"{document_id}_chunks.json"
        
        try:
            logger.info(f"🔍 Loading cached chunks: {chunks_blob_name}")
            chunks_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name
            )
            chunks = json.loads(chunks_data)
            logger.info(f"📋 ✅ Loaded {len(chunks)} cached chunks for document '{document_id}'")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to load chunks for document '{document_id}': {e}")
            raise

    async def _get_page_images(self, document_id: str, storage_service: StorageService) -> List[Dict[str, any]]:
        """Get all available page images for a document"""
        try:
            # List all blobs in cache container
            all_blobs = await storage_service.list_blobs(self.settings.AZURE_CACHE_CONTAINER_NAME)
            
            # Find page images for this document
            page_images = []
            for blob_name in all_blobs:
                if blob_name.startswith(f"{document_id}_page_") and blob_name.endswith('.png'):
                    # Extract page number from blob name
                    page_match = re.search(rf"{document_id}_page_(\d+)\.png", blob_name)
                    if page_match:
                        page_num = int(page_match.group(1))
                        page_images.append({
                            "page_number": page_num,
                            "blob_name": blob_name
                        })
            
            # Sort by page number
            page_images.sort(key=lambda x: x["page_number"])
            logger.info(f"🖼️ Found {len(page_images)} page images for document '{document_id}'")
            return page_images
            
        except Exception as e:
            logger.error(f"❌ Error getting page images for '{document_id}': {e}")
            return []

    async def _detect_visual_question(self, prompt: str) -> bool:
        """Detect if the question requires visual analysis of drawings"""
        visual_keywords = [
            "count", "see", "drawing", "visual", "image", "diagram", "plan", "blueprint",
            "sprinkler", "heads", "symbols", "elements", "objects", "items", "components",
            "how many", "locate", "find", "identify", "point out", "show", "visible",
            "appears", "shown", "depicted", "illustrated", "marked", "labeled"
        ]
        
        prompt_lower = prompt.lower()
        has_visual_keywords = any(keyword in prompt_lower for keyword in visual_keywords)
        
        logger.info(f"🔍 Visual question detection: {has_visual_keywords} for prompt: '{prompt[:50]}...'")
        return has_visual_keywords

    async def get_ai_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        page_number: Optional[int] = None,
        author: Optional[str] = None
    ) -> str:
        """Generate AI response with visual analysis capability"""
        logger.info(f"🧠 Generating AI response for document '{document_id}'" + (f" by {author}" if author else ""))
        
        try:
            # Detect if this is a visual question
            is_visual_question = await self._detect_visual_question(prompt)
            
            if is_visual_question:
                logger.info("👁️ VISUAL ANALYSIS MODE: Using image analysis")
                return await self._get_visual_response(prompt, document_id, storage_service, page_number, author)
            else:
                logger.info("📋 TEXT ANALYSIS MODE: Using cached text chunks")
                return await self._get_text_response(prompt, document_id, storage_service, page_number, author)

        except Exception as e:
            logger.error(f"❌ Error in AI response for '{document_id}': {e}")
            return f"An error occurred while analyzing the document: {str(e)}"

    async def _get_visual_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        page_number: Optional[int] = None,
        author: Optional[str] = None
    ) -> str:
        """Generate response using visual analysis of page images"""
        logger.info(f"👁️ Starting visual analysis for '{document_id}'")
        
        try:
            # Get available page images
            page_images = await self._get_page_images(document_id, storage_service)
            
            if not page_images:
                return "No page images are available for visual analysis. The document may not have been fully processed."
            
            # Determine which pages to analyze
            if page_number is not None:
                # Analyze specific page
                target_pages = [img for img in page_images if img["page_number"] == page_number]
                if not target_pages:
                    return f"Page {page_number} is not available for analysis."
                logger.info(f"🎯 Analyzing specific page {page_number}")
            else:
                # Analyze all pages (limit to first 5 for performance)
                target_pages = page_images[:5]
                logger.info(f"🔍 Analyzing first {len(target_pages)} pages for visual content")
            
            # Build message content for OpenAI
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"You are analyzing blueprint/construction drawings. Look carefully at the images and answer this question: {prompt}\n\n"
                        "VISUAL ANALYSIS INSTRUCTIONS:\n"
                        "- Examine the drawings carefully for visual elements\n"
                        "- Count objects, symbols, or components if asked\n"
                        "- Describe what you actually see in the drawings\n"
                        "- Be specific about quantities, locations, and visual details\n"
                        "- If you can't see something clearly, say so\n"
                        "- Focus on the visual content, not just text labels\n\n"
                        f"Question: {prompt}"
                    )
                }
            ]
            
            # Add page images to the request
            for page_info in target_pages:
                try:
                    # Load the page image
                    image_bytes = await storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=page_info["blob_name"]
                    )
                    
                    # Convert to base64
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    
                    # Add to message content
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                    
                    logger.info(f"🖼️ Added page {page_info['page_number']} image for analysis")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not load image for page {page_info['page_number']}: {e}")
                    continue
            
            # Call OpenAI with visual analysis
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 Vision model
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert construction blueprint analyst with excellent visual perception. "
                            "You can identify, count, and describe elements in technical drawings. "
                            "When analyzing blueprints, you look for symbols, components, annotations, and spatial relationships. "
                            "You provide accurate visual analysis based on what you actually observe in the images."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                temperature=0.1,  # Lower temperature for more consistent visual analysis
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"✅ Visual analysis completed for '{document_id}'")
            
            if ai_response:
                return ai_response.strip()
            else:
                return "The visual analysis did not produce a response. Please try rephrasing your question."
                
        except APIError as e:
            logger.error(f"🚫 OpenAI API error in visual analysis: {e}")
            return f"OpenAI API error during visual analysis: {e}"
        except Exception as e:
            logger.error(f"❌ Error in visual analysis: {e}")
            return f"An error occurred during visual analysis: {str(e)}"

    async def _get_text_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        page_number: Optional[int] = None,
        author: Optional[str] = None
    ) -> str:
        """Generate response using text analysis (your existing system)"""
        logger.info("📋 Using text-based analysis with cached chunks")
        
        try:
            # Load cached chunks
            chunks = await self._load_chunks(document_id, storage_service)
            
            # PASS 1: Identify relevant sections
            relevant_chunk_ids = await self._identify_relevant_sections(prompt, chunks, document_id)
            
            # PASS 2: Generate focused response
            return await self._get_focused_response(
                prompt, chunks, relevant_chunk_ids, storage_service, document_id, page_number, author
            )
            
        except Exception as e:
            logger.error(f"❌ Error in text analysis: {e}")
            return f"An error occurred during text analysis: {str(e)}"

    async def _identify_relevant_sections(self, prompt: str, chunks: List[Dict[str, str]], document_id: str) -> List[int]:
        """PASS 1: AI identifies relevant text sections"""
        logger.info(f"🔍 Analyzing {len(chunks)} sections for relevance")
        
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
                            "You are a construction document analyst. Identify which sections "
                            "are most relevant to answer a specific question. "
                            "Respond with ONLY the section numbers, separated by commas. For example: 2,7,12,15"
                        )
                    },
                    {
                        "role": "user", 
                        "content": (
                            f"QUESTION: {prompt}\n\n"
                            f"DOCUMENT SECTIONS:\n{all_summaries}\n\n"
                            f"Which section numbers (3-5 max) are most relevant?"
                        )
                    }
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            ai_response = response.choices[0].message.content.strip()
            section_numbers = []
            for num_str in re.findall(r'\d+', ai_response):
                section_num = int(num_str)
                if 1 <= section_num <= len(chunks):
                    section_numbers.append(section_num)
            
            if not section_numbers:
                section_numbers = [1, 2, 3]
            elif len(section_numbers) > 5:
                section_numbers = section_numbers[:5]
            
            logger.info(f"✅ Selected sections: {section_numbers}")
            return section_numbers
            
        except Exception as e:
            logger.error(f"❌ Error in section identification: {e}")
            return [1, 2, 3]

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
        """PASS 2: Generate focused response using relevant text sections"""
        logger.info(f"🎯 Building focused response using sections: {relevant_chunk_ids}")
        
        # Gather relevant sections
        relevant_sections = []
        for chunk_id in relevant_chunk_ids:
            for chunk in chunks:
                if chunk['chunk_id'] == chunk_id:
                    section_content = f"=== SECTION {chunk_id} ===\n{chunk['content']}"
                    relevant_sections.append(section_content)
                    break
        
        focused_content = "\n\n".join(relevant_sections)
        
        # Build user message
        user_content = [
            {
                "type": "text",
                "text": (
                    "You are an experienced construction superintendent reading blueprint documents. "
                    "Answer questions directly using the specific information provided.\n\n"
                    f"DOCUMENT SECTIONS:\n\n{focused_content}\n\n"
                    f"QUESTION: {prompt}\n\n"
                    "Give direct, specific answers using the exact information from the sections above."
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
                logger.info(f"🖼️ Enhanced with page {page_number} image")
            except Exception as e:
                logger.warning(f"⚠️ Could not load page {page_number} image: {e}")

        try:
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an experienced construction superintendent who reads blueprints daily. "
                            "Give direct answers with specific facts and numbers from the provided sections."
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
            logger.info(f"✅ Focused response completed")
            
            return ai_message.strip() if ai_message else "No response generated."

        except Exception as e:
            logger.error(f"❌ Error in focused response: {e}")
            return f"An error occurred while generating your answer: {str(e)}"

    async def get_document_info(self, document_id: str, storage_service: StorageService) -> Dict[str, any]:
        """Get information about a document"""
        try:
            chunks_blob_name = f"{document_id}_chunks.json"
            chunks_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name
            )
            chunks = json.loads(chunks_data)
            
            context_blob_name = f"{document_id}_context.txt"
            full_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=context_blob_name
            )
            
            # Count page images
            page_images = await self._get_page_images(document_id, storage_service)
            
            return {
                "document_id": document_id,
                "status": "ready",
                "total_chunks": len(chunks),
                "total_characters": len(full_text),
                "estimated_tokens": len(full_text) // 4,
                "page_count": len(page_images),
                "visual_analysis_available": len(page_images) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting document info: {e}")
            return {
                "document_id": document_id,
                "status": "not_found",
                "error": str(e)
            }
