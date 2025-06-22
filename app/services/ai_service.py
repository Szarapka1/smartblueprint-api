# app/services/ai_service.py

import logging
import base64
import re
import json
from typing import Optional, List, Dict
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionToolParam
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

    def _chunk_document(self, text: str, chunk_size: int = 2500) -> List[Dict[str, str]]:
        """
        Break document into numbered chunks with previews for caching.
        """
        logger.info(f"📑 Chunking document into sections of max {chunk_size} characters")
        
        # Split by double newlines first (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 1
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
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
        
        logger.info(f"📑 Document chunked into {len(chunks)} sections")
        for chunk in chunks:
            logger.info(f"  - Section {chunk['chunk_id']}: {len(chunk['content'])} chars")
        
        return chunks

    async def _load_or_create_chunks(self, document_id: str, storage_service: StorageService) -> List[Dict[str, str]]:
        """
        Load cached chunks from blob storage, or create and cache them if they don't exist.
        """
        chunks_blob_name = f"{document_id}_chunks.json"
        
        try:
            logger.info(f"🔍 Looking for cached chunks: {chunks_blob_name}")
            chunks_data = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=chunks_blob_name
            )
            chunks = json.loads(chunks_data)
            logger.info(f"📋 ✅ Loaded {len(chunks)} cached chunks for document '{document_id}'")
            return chunks
            
        except Exception as e:
            logger.info(f"📝 No cached chunks found for document '{document_id}', creating new ones: {str(e)[:100]}...")
            
            try:
                # Load full text and create chunks
                full_text = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                
                logger.info(f"📄 Loaded full document text ({len(full_text)} characters)")
                
                # Create chunks
                chunks = self._chunk_document(full_text)
                
                # Cache the chunks for future use
                chunks_json = json.dumps(chunks, indent=2, ensure_ascii=False)
                await storage_service.upload_file(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=chunks_blob_name,
                    data=chunks_json.encode('utf-8')
                )
                logger.info(f"💾 ✅ Cached {len(chunks)} chunks for shared document '{document_id}'")
                
                return chunks
                
            except Exception as create_error:
                logger.error(f"❌ Failed to create chunks for document '{document_id}': {create_error}")
                raise

    async def _get_document_page_count(self, document_id: str, storage_service: StorageService) -> int:
        """Get the total number of pages in the document by checking for page images"""
        try:
            blobs = await storage_service.list_blobs(container_name=self.settings.AZURE_CACHE_CONTAINER_NAME)
            page_blobs = [blob for blob in blobs if blob.startswith(f"{document_id}_page_") and blob.endswith('.png')]
            
            if not page_blobs:
                return 1  # Default to 1 page if no page images found
            
            # Extract page numbers and find the maximum
            page_numbers = []
            for blob in page_blobs:
                try:
                    # Extract number from pattern: {document_id}_page_{number}.png
                    page_num_str = blob.replace(f"{document_id}_page_", "").replace(".png", "")
                    page_numbers.append(int(page_num_str))
                except ValueError:
                    continue
            
            return max(page_numbers) if page_numbers else 1
        except Exception as e:
            logger.warning(f"Could not determine page count for document '{document_id}': {e}")
            return 1

    # Enhanced tools for comprehensive document search
    _TOOLS = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_page_image",
                "description": (
                    "Retrieves the image of a specific page from the blueprint document. "
                    "Use this when you need to examine visual elements on a specific page."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "The 1-based page number to retrieve.",
                            "minimum": 1
                        }
                    },
                    "required": ["page_number"]
                },
            }
        ),
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "search_all_pages_for_symbols",
                "description": (
                    "Searches through ALL pages of the document to find specific symbols, elements, or features. "
                    "Use this when the user asks about finding symbols, elements, or features throughout the entire document "
                    "(e.g., 'find sprinkler heads', 'show me fire alarms', 'where are the electrical outlets', 'find all doors'). "
                    "This tool will examine every page and provide a comprehensive analysis."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "What to search for (e.g., 'sprinkler head symbols', 'fire alarm symbols', 'electrical outlets', 'doors', 'windows')"
                        }
                    },
                    "required": ["search_query"]
                },
            }
        )
    ]

    async def get_ai_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        author: Optional[str] = None
    ) -> str:
        """
        Generate AI response with enhanced multi-page search capabilities.
        """
        logger.info(f"🧠 Generating AI response for document '{document_id}'" + (f" by {author}" if author else ""))

        try:
            # Load cached chunks for context
            chunks = await self._load_or_create_chunks(document_id, storage_service)
            
            # Get document page count
            total_pages = await self._get_document_page_count(document_id, storage_service)
            
            # Identify relevant sections from text
            relevant_chunk_ids = await self._identify_relevant_sections(prompt, chunks, document_id)
            relevant_sections_text = self._format_relevant_sections(chunks, relevant_chunk_ids)

            # Enhanced system message with multi-page search capabilities
            system_message = (
                f"You are an expert construction superintendent and blueprint reader. "
                f"You have access to a {total_pages}-page blueprint document. "
                f"You can search through ALL pages to find symbols, elements, or features anywhere in the document.\n\n"
                f"IMPORTANT CAPABILITIES:\n"
                f"- Use 'search_all_pages_for_symbols' when users ask about finding symbols/elements throughout the document\n"
                f"- Use 'get_page_image' when you need to examine a specific page in detail\n"
                f"- Always provide specific page numbers when you find elements\n"
                f"- Give direct, factual answers with precise locations\n\n"
                f"DOCUMENT TEXT CONTEXT:\n{relevant_sections_text}"
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]

            # Initial AI call with tool capabilities
            logger.info("🤖 Making initial AI call with enhanced search tools...")
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                tools=self._TOOLS,
                tool_choice="auto",
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                logger.info(f"🔧 AI requested tool: {tool_calls[0].function.name}")
                
                if tool_calls[0].function.name == "search_all_pages_for_symbols":
                    return await self._handle_multi_page_search(
                        tool_calls[0], messages, response_message, document_id, 
                        storage_service, total_pages
                    )
                
                elif tool_calls[0].function.name == "get_page_image":
                    return await self._handle_single_page_request(
                        tool_calls[0], messages, response_message, document_id, 
                        storage_service, prompt
                    )
                
                else:
                    return f"Unsupported tool requested: {tool_calls[0].function.name}"
            else:
                # AI responded directly without tools
                logger.info("💬 AI responded directly (no tool needed)")
                return response_message.content.strip() if response_message.content else "No response generated."

        except APIError as e:
            logger.error(f"🚫 OpenAI API error: {e}")
            return f"OpenAI API error: {e}"
        except Exception as e:
            logger.error(f"❌ Unexpected error in AI response: {e}")
            return f"An error occurred while generating your answer: {str(e)}"

    async def _handle_multi_page_search(
        self, tool_call, messages, response_message, document_id, storage_service, total_pages
    ) -> str:
        """Handle searching through all pages for symbols/elements"""
        try:
            arguments = json.loads(tool_call.function.arguments)
            search_query = arguments.get("search_query", "")
            
            logger.info(f"🔍 Searching all {total_pages} pages for: {search_query}")
            
            # Collect all page images
            page_images = []
            successful_pages = []
            
            for page_num in range(1, total_pages + 1):
                try:
                    image_blob_name = f"{document_id}_page_{page_num}.png"
                    image_bytes = await storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=image_blob_name
                    )
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    page_images.append({
                        "page": page_num,
                        "image": base64_image
                    })
                    successful_pages.append(page_num)
                    logger.info(f"📄 Loaded page {page_num}")
                except Exception as e:
                    logger.warning(f"Could not load page {page_num}: {e}")
            
            if not page_images:
                return f"I couldn't access any page images to search for {search_query}. Please try again."
            
            # Add tool call response to conversation
            messages.append(response_message)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "search_all_pages_for_symbols",
                "content": json.dumps({
                    "status": "success",
                    "search_query": search_query,
                    "pages_searched": successful_pages,
                    "total_pages": len(page_images)
                })
            })
            
            # Create comprehensive message with all page images
            content_items = [
                {
                    "type": "text", 
                    "text": f"I've loaded all {len(page_images)} pages of the blueprint. Please analyze each page carefully to find all instances of '{search_query}'. For each instance you find, specify the exact page number and describe the location on that page."
                }
            ]
            
            # Add all page images to the message
            for page_data in page_images:
                content_items.append({
                    "type": "text",
                    "text": f"\n--- PAGE {page_data['page']} ---"
                })
                content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page_data['image']}",
                        "detail": "high"
                    }
                })
            
            messages.append({
                "role": "user",
                "content": content_items
            })
            
            # Make final AI call with all page images
            logger.info(f"🤖 Making comprehensive analysis call with {len(page_images)} page images...")
            final_response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS
            )
            
            return final_response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in multi-page search: {e}")
            return f"I encountered an error while searching through the pages: {str(e)}"

    async def _handle_single_page_request(
        self, tool_call, messages, response_message, document_id, storage_service, original_prompt
    ) -> str:
        """Handle single page image requests"""
        try:
            arguments = json.loads(tool_call.function.arguments)
            page_number = arguments.get("page_number")
            
            if page_number is None:
                return "I needed a page number to answer your question, but it wasn't specified."
            
            # Get the specific page image
            image_blob_name = f"{document_id}_page_{page_number}.png"
            image_bytes = await storage_service.download_blob_as_bytes(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=image_blob_name
            )
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            logger.info(f"🖼️ Retrieved page {page_number} for analysis")
            
            # Add tool response and image to conversation
            messages.append(response_message)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "get_page_image",
                "content": json.dumps({"status": "success", "page_number": page_number})
            })
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": original_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            })
            
            # Get AI response with the page image
            logger.info("🔄 Making follow-up call with page image...")
            final_response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS
            )
            
            return final_response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in single page request: {e}")
            return f"I tried to get page {page_number} but encountered an error: {str(e)}"

    async def _identify_relevant_sections(self, prompt: str, chunks: List[Dict[str, str]], document_id: str) -> List[int]:
        """
        AI identifies which document sections are most relevant to the user's question.
        """
        logger.info("🔍 AI identifying relevant sections from document chunks")
        
        # Create previews for AI to choose from
        chunk_previews = []
        for chunk in chunks:
            chunk_previews.append(f"Section {chunk['chunk_id']}: {chunk['preview']}")
        
        previews_text = "\n".join(chunk_previews)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use faster model for section identification
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing a blueprint document. "
                            "The user will ask a question, and you need to identify which sections are most relevant. "
                            "Return ONLY a JSON array of section numbers, like [1, 3, 5]. "
                            "If the question seems to require visual analysis of the actual blueprint drawings, "
                            "still return the most relevant text sections that might contain related information."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {prompt}\n\nAvailable sections:\n{previews_text}\n\nWhich sections are most relevant? Return only JSON array of numbers."
                    }
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response to get section numbers
            try:
                relevant_sections = json.loads(ai_response)
                if isinstance(relevant_sections, list) and all(isinstance(x, int) for x in relevant_sections):
                    # Validate section numbers exist
                    valid_sections = [s for s in relevant_sections if 1 <= s <= len(chunks)]
                    if valid_sections:
                        logger.info(f"📋 AI selected sections: {valid_sections}")
                        return valid_sections
            except json.JSONDecodeError:
                pass
            
            # Fallback: return first few sections
            logger.warning("Could not parse AI section selection, using fallback")
            return [1, 2, 3][:len(chunks)]
            
        except Exception as e:
            logger.error(f"Error in section identification: {e}")
            # Fallback: return first few sections
            return [1, 2, 3][:len(chunks)]

    def _format_relevant_sections(self, chunks: List[Dict[str, str]], relevant_chunk_ids: List[int]) -> str:
        """Format relevant sections into text for AI context"""
        relevant_sections = []
        for chunk_id in relevant_chunk_ids:
            for chunk in chunks:
                if chunk['chunk_id'] == chunk_id:
                    relevant_sections.append(f"=== SECTION {chunk_id} ===\n{chunk['content']}")
                    break
        return "\n\n".join(relevant_sections)

    async def get_document_info(self, document_id: str, storage_service: StorageService) -> Dict[str, any]:
        """Get information about a document including processing status"""
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
            
            # Get page count
            total_pages = await self._get_document_page_count(document_id, storage_service)
            
            return {
                "document_id": document_id,
                "status": "ready",
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "total_characters": len(full_text),
                "estimated_tokens": len(full_text) // 4,
                "last_processed": "Available"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting document info for '{document_id}': {e}")
            return {
                "document_id": document_id,
                "status": "not_found",
                "error": str(e)
            }
