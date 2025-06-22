# app/services/ai_service.py

import logging
import base64
import re
import json
from typing import Optional, List, Dict
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionToolParam # Import for tool definition
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
        Uses document_id instead of session_id for shared access.
        """
        chunks_blob_name = f"{document_id}_chunks.json"
        
        try:
            # Try to load existing cached chunks
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
                
                # Cache the chunks for future use by anyone
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

    # Define the tool for getting page images
    _TOOLS = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_page_image",
                "description": (
                    "Retrieves the image of a specific page from the blueprint document to answer visual questions. "
                    "Use this tool when the user asks about visual elements, layouts, specific symbols, or something "
                    "that requires looking at the actual blueprint drawing (e.g., 'What does the symbol for fire alarms look like on page 5?', "
                    "'Show me the layout of page 2', 'Where is the main entrance on this page?')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "The 1-based page number of the document to retrieve the image for.",
                            "minimum": 1
                        }
                    },
                    "required": ["page_number"]
                },
            }
        )
    ]

    async def get_ai_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        # page_number: Optional[int] = None, # No longer directly passed for AI decision
        author: Optional[str] = None
    ) -> str:
        """
        Generate AI response for a shared document that everyone can access.
        The AI can now decide to use tools (like getting page images).
        """
        logger.info(f"🧠 Generating AI response for shared document '{document_id}'" + (f" by {author}" if author else ""))
        logger.info("📋 Using SHARED CACHED TWO-PASS approach with potential Tool Calling")

        try:
            # Step 1: Load cached chunks for this shared document
            chunks = await self._load_or_create_chunks(document_id, storage_service)
            
            # Step 2: PASS 1 - AI identifies relevant sections using cached previews
            logger.info("🔍 PASS 1: AI identifying relevant sections from shared cached data")
            relevant_chunk_ids = await self._identify_relevant_sections(prompt, chunks, document_id)
            
            # Step 3: Prepare system message and relevant content for AI
            relevant_sections_text = self._format_relevant_sections(chunks, relevant_chunk_ids)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an experienced construction superintendent who reads blueprint documents. "
                        "Read the document sections provided and answer questions with specific facts and numbers. "
                        "Give direct answers like you're looking right at the plans - no hedging or analysis."
                        "If the user asks a visual question (e.g., about a symbol, layout, or something on 'this page'), "
                        "you MUST use the `get_page_image` tool to retrieve the relevant page's image. "
                        "When using the tool, ensure the page number is valid for the document."
                        "\n\nDOCUMENT SECTIONS:\n" + relevant_sections_text
                    )
                },
                {"role": "user", "content": prompt}
            ]

            # Step 4: Initial AI call - AI decides if it needs a tool
            logger.info("🤖 Making initial AI call (tool-enabled)...")
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                tools=self._TOOLS, # Pass the defined tools
                tool_choice="auto", # Allow AI to choose a tool
                temperature=self.settings.OPENAI_TEMPERATURE,
                max_tokens=self.settings.OPENAI_MAX_TOKENS
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # Step 5: AI wants to use a tool (e.g., get_page_image)
                logger.info(f"🔧 AI requested tool call: {tool_calls[0].function.name}")
                if tool_calls[0].function.name == "get_page_image":
                    try:
                        arguments = json.loads(tool_calls[0].function.arguments)
                        page_number_for_tool = arguments.get("page_number")
                        
                        if page_number_for_tool is None:
                            logger.error("❌ Tool call missing page_number argument.")
                            return "I needed a page number to answer that visual question, but it wasn't specified."

                        # Retrieve and encode image
                        image_blob_name = f"{document_id}_page_{page_number_for_tool}.png"
                        image_bytes = await storage_service.download_blob_as_bytes(
                            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                            blob_name=image_blob_name
                        )
                        base64_image = base64.b64encode(image_bytes).decode("utf-8")
                        
                        logger.info(f"🖼️ Retrieved image for page {page_number_for_tool}.")

                        # Add image to messages for a follow-up AI call
                        messages.append(response_message) # The AI's tool call response
                        messages.append(
                            {
                                "tool_call_id": tool_calls[0].id,
                                "role": "tool",
                                "name": tool_calls[0].function.name,
                                "content": json.dumps({"status": "success", "page_number": page_number_for_tool}) # Tool output confirmation
                            }
                        )
                        
                        # Add image content for the final AI call
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}, # Original prompt
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                            ]
                        })
                        logger.info("🔄 Making follow-up AI call with image context...")
                        second_response = self.client.chat.completions.create(
                            model=self.settings.OPENAI_MODEL,
                            messages=messages,
                            temperature=self.settings.OPENAI_TEMPERATURE,
                            max_tokens=self.settings.OPENAI_MAX_TOKENS
                        )
                        return second_response.choices[0].message.content.strip()

                    except Exception as tool_error:
                        logger.error(f"❌ Error processing tool call for '{document_id}': {tool_error}")
                        return f"I tried to get a page image but encountered an error: {str(tool_error)}. Please try again or rephrase."
                else:
                    return f"The AI requested an unsupported tool: {tool_calls[0].function.name}."
            else:
                # Step 5: AI responded directly (no tool needed)
                logger.info("💬 AI responded directly (no tool call).")
                ai_message = response_message.content
                if ai_message:
                    return ai_message.strip()
                else:
                    return "The AI returned an empty response. Please try rephrasing your question."

        except APIError as e:
            logger.error(f"🚫 OpenAI API error: {e}")
            return f"OpenAI API error: {e}"
        except Exception as e:
            logger.error(f"❌ Unexpected error in AI response for '{document_id}': {e}")
            return f"An error occurred while generating your answer: {str(e)}"

    def _format_relevant_sections(self, chunks: List[Dict[str, str]], relevant_chunk_ids: List[int]) -> str:
        """Helper to format relevant sections into a single string for the AI."""
        relevant_sections = []
        for chunk_id in relevant_chunk_ids:
            for chunk in chunks:
                if chunk['chunk_id'] == chunk_id:
                    relevant_sections.append(f"=== SECTION {chunk_id} ===\n{chunk['content']}")
                    break
        return "\n\n".join(relevant_sections)

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
