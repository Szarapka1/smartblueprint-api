# app/services/ai_service.py - Enhanced with Visual Analysis and Comprehensive Debugging

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
            logger.info(f"🔍 Searching for page images for document '{document_id}'")
            
            # List all blobs in cache container
            all_blobs = await storage_service.list_blobs(self.settings.AZURE_CACHE_CONTAINER_NAME)
            logger.info(f"🔍 Found {len(all_blobs)} total blobs in cache container")
            
            # Find page images for this document
            page_images = []
            pattern = f"{document_id}_page_"
            
            for blob_name in all_blobs:
                if blob_name.startswith(pattern) and blob_name.endswith('.png'):
                    # Extract page number from blob name
                    page_match = re.search(rf"{document_id}_page_(\d+)\.png", blob_name)
                    if page_match:
                        page_num = int(page_match.group(1))
                        page_images.append({
                            "page_number": page_num,
                            "blob_name": blob_name
                        })
                        logger.debug(f"🖼️ Found page image: {blob_name} (page {page_num})")
            
            # Sort by page number
            page_images.sort(key=lambda x: x["page_number"])
            logger.info(f"🖼️ ✅ Found {len(page_images)} page images for document '{document_id}'")
            
            for img in page_images:
                logger.info(f"🖼️   - Page {img['page_number']}: {img['blob_name']}")
            
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
            "appears", "shown", "depicted", "illustrated", "marked", "labeled", "look",
            "examine", "observe", "spot", "detect", "notice", "view", "display"
        ]
        
        prompt_lower = prompt.lower()
        found_keywords = [keyword for keyword in visual_keywords if keyword in prompt_lower]
        has_visual_keywords = len(found_keywords) > 0
        
        # Enhanced logging
        logger.info(f"🔍 ================================")
        logger.info(f"🔍 VISUAL QUESTION DETECTION")
        logger.info(f"🔍 ================================")
        logger.info(f"📝 Original prompt: '{prompt}'")
        logger.info(f"📝 Lowercase prompt: '{prompt_lower}'")
        logger.info(f"🎯 Found keywords: {found_keywords}")
        logger.info(f"✅ Should use visual analysis: {has_visual_keywords}")
        logger.info(f"🔍 ================================")
        
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
        logger.info(f"🧠 =================================")
        logger.info(f"🧠 STARTING AI RESPONSE GENERATION")
        logger.info(f"🧠 =================================")
        logger.info(f"🧠 Document: '{document_id}'")
        logger.info(f"🧠 Prompt: '{prompt}'")
        logger.info(f"🧠 Page: {page_number}")
        logger.info(f"🧠 Author: {author}")
        logger.info(f"🧠 =================================")
        
        try:
            # Step 1: Detect if this is a visual question
            logger.info("🔍 STEP 1: Detecting question type...")
            is_visual_question = await self._detect_visual_question(prompt)
            
            if is_visual_question:
                logger.info("👁️ ✅ VISUAL ANALYSIS MODE SELECTED")
                logger.info("👁️ Proceeding with image-based analysis...")
                
                try:
                    response = await self._get_visual_response(prompt, document_id, storage_service, page_number, author)
                    logger.info("👁️ ✅ Visual analysis completed successfully")
                    return response
                except Exception as visual_error:
                    logger.error(f"👁️ ❌ Visual analysis failed: {visual_error}")
                    logger.error(f"👁️ ❌ Error type: {type(visual_error).__name__}")
                    logger.info("📋 🔄 Falling back to text analysis...")
                    # Fall back to text analysis
                    return await self._get_text_response(prompt, document_id, storage_service, page_number, author)
            else:
                logger.info("📋 ✅ TEXT ANALYSIS MODE SELECTED")
                logger.info("📋 Proceeding with text-based analysis...")
                return await self._get_text_response(prompt, document_id, storage_service, page_number, author)

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in AI response: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
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
        logger.info(f"👁️ ===============================")
        logger.info(f"👁️ STARTING VISUAL ANALYSIS")
        logger.info(f"👁️ ===============================")
        logger.info(f"👁️ Document: '{document_id}'")
        logger.info(f"👁️ Target page: {page_number or 'All pages'}")
        logger.info(f"👁️ ===============================")
        
        try:
            # Step 1: Get available page images
            logger.info("👁️ STEP 1: Finding available page images...")
            page_images = await self._get_page_images(document_id, storage_service)
            
            if not page_images:
                logger.error("👁️ ❌ No page images available for visual analysis")
                # Try to provide helpful information
                try:
                    # Check if document exists at all
                    chunks = await self._load_chunks(document_id, storage_service)
                    if chunks:
                        logger.info("👁️ ℹ️ Document has text but no images - may be text-only PDF")
                        return "This document appears to be text-only without visual elements. No page images are available for visual analysis. Please try asking about the text content instead."
                    else:
                        return "No document data found. The document may not have been processed yet."
                except:
                    return "No page images are available for visual analysis. The document may not have been fully processed or may be text-only."
            
            # Step 2: Determine which pages to analyze
            logger.info("👁️ STEP 2: Selecting pages for analysis...")
            if page_number is not None:
                target_pages = [img for img in page_images if img["page_number"] == page_number]
                if not target_pages:
                    logger.error(f"👁️ ❌ Page {page_number} not found in available images")
                    available_pages = [img["page_number"] for img in page_images]
                    return f"Page {page_number} is not available for analysis. Available pages: {available_pages}"
                logger.info(f"👁️ ✅ Analyzing specific page {page_number}")
            else:
                # Analyze first 3 pages for performance (can be configured)
                max_pages = int(os.getenv("MAX_VISUAL_PAGES", "3"))
                target_pages = page_images[:max_pages]
                logger.info(f"👁️ ✅ Analyzing first {len(target_pages)} pages (max: {max_pages})")
            
            # Step 3: Build message content for OpenAI
            logger.info("👁️ STEP 3: Building OpenAI request...")
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"You are analyzing blueprint/construction drawings. Look carefully at the images and answer this question: {prompt}\n\n"
                        "VISUAL ANALYSIS INSTRUCTIONS:\n"
                        "- Examine ALL drawings carefully for visual elements\n"
                        "- Count objects, symbols, or components EXACTLY as they appear\n"
                        "- Describe what you actually see in the drawings\n"
                        "- Be specific about quantities, locations, and visual details\n"
                        "- If counting symbols, look for small circles, crosses, or other technical symbols\n"
                        "- Pay attention to legends, keys, or symbol explanations\n"
                        "- If you can't see something clearly, say so\n"
                        "- Focus on the visual content, not just text labels\n"
                        "- For sprinkler heads, look for circular symbols, often with crosses or dots inside\n\n"
                        f"Question: {prompt}"
                    )
                }
            ]
            
            # Step 4: Add page images to the request
            logger.info("👁️ STEP 4: Loading and encoding page images...")
            successfully_loaded_images = 0
            total_image_size_mb = 0
            
            for page_info in target_pages:
                try:
                    logger.info(f"👁️ Loading image: {page_info['blob_name']}")
                    
                    # Load the page image
                    image_bytes = await storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=page_info["blob_name"]
                    )
                    
                    image_size_mb = len(image_bytes) / (1024 * 1024)
                    total_image_size_mb += image_size_mb
                    logger.info(f"👁️ Image loaded: {image_size_mb:.2f}MB")
                    
                    # Validate it's a PNG
                    is_valid_png = image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
                    if not is_valid_png:
                        logger.warning(f"👁️ ⚠️ Image may not be valid PNG: {page_info['blob_name']}")
                        logger.warning(f"👁️ ⚠️ First 16 bytes: {image_bytes[:16]}")
                    else:
                        logger.info(f"👁️ ✅ Valid PNG confirmed")
                    
                    # Convert to base64
                    try:
                        base64_image = base64.b64encode(image_bytes).decode("utf-8")
                        base64_size_mb = len(base64_image) / (1024 * 1024)
                        logger.info(f"👁️ Base64 encoded: {base64_size_mb:.2f}MB")
                        
                        # Add to message content
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"  # Use high detail for better symbol recognition
                            }
                        })
                        
                        successfully_loaded_images += 1
                        logger.info(f"👁️ ✅ Successfully added page {page_info['page_number']} to analysis")
                        
                    except Exception as b64_error:
                        logger.error(f"👁️ ❌ Base64 encoding failed for {page_info['blob_name']}: {b64_error}")
                        continue
                    
                except Exception as e:
                    logger.error(f"👁️ ❌ Failed to load image {page_info['blob_name']}: {e}")
                    logger.error(f"👁️ ❌ Error type: {type(e).__name__}")
                    continue
            
            if successfully_loaded_images == 0:
                logger.error("👁️ ❌ No images could be loaded for analysis")
                return "Failed to load any images for visual analysis. There may be an issue with the stored images. Please try reprocessing the document."
            
            logger.info(f"👁️ ✅ Successfully loaded {successfully_loaded_images} images for analysis")
            logger.info(f"👁️ ✅ Total image payload: {total_image_size_mb:.2f}MB")
            
            # Check if payload is too large
            if total_image_size_mb > 20:  # OpenAI has size limits
                logger.warning(f"👁️ ⚠️ Large image payload ({total_image_size_mb:.2f}MB) - may hit API limits")
            
            # Step 5: Call OpenAI with visual analysis
            logger.info("👁️ STEP 5: Calling OpenAI Vision API...")
            
            try:
                # Determine the best model to use
                model_to_use = "gpt-4o"  # Use GPT-4o for vision tasks
                logger.info(f"👁️ Using model: {model_to_use}")
                
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert construction blueprint analyst with excellent visual perception. "
                                "You can identify, count, and describe elements in technical drawings with high accuracy. "
                                "When analyzing blueprints, you look for symbols, components, annotations, and spatial relationships. "
                                "You provide accurate visual analysis based on what you actually observe in the images. "
                                "When counting elements, you examine each image systematically and provide exact counts. "
                                "For sprinkler heads and similar symbols, you look for circular markers, crosses, or standardized symbols. "
                                "You always describe your methodology and what you observe."
                            )
                        },
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ],
                    temperature=0.1,  # Lower temperature for more consistent visual analysis
                    max_tokens=1500  # Allow for detailed responses
                )
                
                ai_response = response.choices[0].message.content
                logger.info(f"👁️ ✅ OpenAI Vision API responded successfully")
                logger.info(f"👁️ Response length: {len(ai_response) if ai_response else 0} characters")
                logger.info(f"👁️ Response preview: {ai_response[:200] if ai_response else 'No response'}...")
                
                if ai_response:
                    return ai_response.strip()
                else:
                    logger.error("👁️ ❌ OpenAI returned empty response")
                    return "The visual analysis did not produce a response. Please try rephrasing your question."
                    
            except APIError as openai_error:
                logger.error(f"👁️ ❌ OpenAI API error: {openai_error}")
                logger.error(f"👁️ ❌ OpenAI error type: {type(openai_error).__name__}")
                logger.error(f"👁️ ❌ OpenAI error details: {str(openai_error)}")
                
                # Provide more specific error messages
                if "rate_limit" in str(openai_error).lower():
                    return "OpenAI API rate limit reached. Please try again in a moment."
                elif "context_length" in str(openai_error).lower():
                    return "The images are too large for analysis. Please try analyzing specific pages or reduce image quality."
                elif "insufficient_quota" in str(openai_error).lower():
                    return "OpenAI API quota exceeded. Please check your API usage."
                else:
                    return f"OpenAI API error during visual analysis: {str(openai_error)}"
                    
            except Exception as general_error:
                logger.error(f"👁️ ❌ General error during OpenAI call: {general_error}")
                logger.error(f"👁️ ❌ Error type: {type(general_error).__name__}")
                raise general_error
            
        except Exception as e:
            logger.error(f"👁️ ❌ Visual analysis failed: {e}")
            logger.error(f"👁️ ❌ Error type: {type(e).__name__}")
            raise e

    async def _get_text_response(
        self,
        prompt: str,
        document_id: str,
        storage_service: StorageService,
        page_number: Optional[int] = None,
        author: Optional[str] = None
    ) -> str:
        """Generate response using text analysis (cached chunks)"""
        logger.info("📋 ===============================")
        logger.info("📋 STARTING TEXT ANALYSIS")
        logger.info("📋 ===============================")
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
                "visual_analysis_available": len(page_images) > 0,
                "available_pages": [img["page_number"] for img in page_images] if page_images else []
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting document info: {e}")
            return {
                "document_id": document_id,
                "status": "not_found",
                "error": str(e)
            }

    async def test_visual_analysis_pipeline(self, document_id: str, storage_service: StorageService) -> Dict[str, any]:
        """Test the visual analysis pipeline for debugging"""
        logger.info(f"🧪 Testing visual analysis pipeline for document '{document_id}'")
        
        test_results = {
            "document_id": document_id,
            "tests": []
        }
        
        try:
            # Test 1: Check if document chunks exist
            try:
                chunks = await self._load_chunks(document_id, storage_service)
                test_results["tests"].append({
                    "test": "chunks_exist",
                    "status": "pass",
                    "details": f"Found {len(chunks)} chunks"
                })
            except Exception as e:
                test_results["tests"].append({
                    "test": "chunks_exist",
                    "status": "fail",
                    "details": str(e)
                })
            
            # Test 2: Check if page images exist
            try:
                page_images = await self._get_page_images(document_id, storage_service)
                test_results["tests"].append({
                    "test": "images_exist",
                    "status": "pass" if page_images else "fail",
                    "details": f"Found {len(page_images)} page images: {[img['blob_name'] for img in page_images]}"
                })
                
                # Test 3: Try to load first image
                if page_images:
                    try:
                        first_image = page_images[0]
                        image_bytes = await storage_service.download_blob_as_bytes(
                            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                            blob_name=first_image["blob_name"]
                        )
                        
                        is_valid_png = image_bytes.startswith(b'\x89PNG\r\n\x1a\n')
                        size_mb = len(image_bytes) / (1024 * 1024)
                        
                        test_results["tests"].append({
                            "test": "image_loading",
                            "status": "pass",
                            "details": f"Loaded {first_image['blob_name']}: {size_mb:.2f}MB, valid PNG: {is_valid_png}"
                        })
                        
                        # Test 4: Base64 encoding
                        try:
                            base64_image = base64.b64encode(image_bytes).decode("utf-8")
                            test_results["tests"].append({
                                "test": "base64_encoding",
                                "status": "pass",
                                "details": f"Base64 length: {len(base64_image)} characters"
                            })
                        except Exception as e:
                            test_results["tests"].append({
                                "test": "base64_encoding",
                                "status": "fail",
                                "details": str(e)
                            })
                            
                    except Exception as e:
                        test_results["tests"].append({
                            "test": "image_loading",
                            "status": "fail",
                            "details": str(e)
                        })
                
            except Exception as e:
                test_results["tests"].append({
                    "test": "images_exist",
                    "status": "fail",
                    "details": str(e)
                })
            
            # Test 5: Visual question detection
            test_prompts = [
                "count the sprinkler heads",
                "how many symbols are there",
                "what does the blueprint show"
            ]
            
            for prompt in test_prompts:
                is_visual = await self._detect_visual_question(prompt)
                test_results["tests"].append({
                    "test": f"visual_detection",
                    "status": "pass",
                    "details": f"Prompt: '{prompt}' -> Visual: {is_visual}"
                })
                
        except Exception as e:
            test_results["tests"].append({
                "test": "pipeline_test",
                "status": "fail",
                "details": str(e)
            })
        
        logger.info(f"🧪 Pipeline test completed: {json.dumps(test_results, indent=2)}")
        return test_results
