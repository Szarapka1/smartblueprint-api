# app/services/ai_service.py - OPTIMIZED FOR LARGE MULTI-PAGE DOCUMENTS

import asyncio
import base64
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from PIL import Image
import io

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# OpenAI imports
try:
    from openai import OpenAI
    logger.info("✅ OpenAI SDK imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import OpenAI SDK: {e}")
    raise

# Internal imports
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService


class ProfessionalBlueprintAI:
    """Professional AI service optimized for large multi-page blueprint analysis"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Professional Blueprint AI initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration for optimization
        self.max_pages_to_load = int(os.getenv("AI_MAX_PAGES", "100"))
        self.batch_size = int(os.getenv("AI_BATCH_SIZE", "10"))
        self.image_quality = int(os.getenv("AI_IMAGE_QUALITY", "85"))
        self.max_image_dimension = int(os.getenv("AI_MAX_IMAGE_DIMENSION", "2000"))
        
        logger.info(f"   📄 Max pages: {self.max_pages_to_load}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   🖼️ Image quality: {self.image_quality}%")
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process blueprint queries with optimized multi-page support"""
        try:
            logger.info(f"📐 Processing blueprint analysis for {document_id}")
            analysis_start = asyncio.get_event_loop().time()
            
            # Load document context and metadata
            document_text = ""
            metadata = None
            
            try:
                # Load text context (contains text from ALL pages)
                context_task = storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                document_text = await asyncio.wait_for(context_task, timeout=30.0)
                logger.info(f"✅ Loaded text from all pages: {len(document_text)} characters")
                
                # Try to load metadata for optimization
                try:
                    metadata_task = storage_service.download_blob_as_text(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{document_id}_metadata.json"
                    )
                    metadata_text = await asyncio.wait_for(metadata_task, timeout=10.0)
                    metadata = json.loads(metadata_text)
                    logger.info(f"✅ Loaded metadata: {metadata.get('page_count', 0)} pages")
                except:
                    logger.info("ℹ️ No metadata found, will discover pages")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return "Unable to load the blueprint. Please ensure the document is properly uploaded and processed."
            
            # Load ALL page images efficiently
            image_urls = await self._load_all_pages_optimized(
                document_id, 
                storage_service, 
                metadata
            )
            
            loading_time = asyncio.get_event_loop().time() - analysis_start
            logger.info(f"⏱️ Document loaded in {loading_time:.2f}s")
            
            # Process with professional analysis
            result = await self._analyze_blueprint_professionally(
                prompt=prompt,
                document_text=document_text,
                image_urls=image_urls,
                document_id=document_id,
                author=author
            )
            
            total_time = asyncio.get_event_loop().time() - analysis_start
            logger.info(f"✅ Analysis complete in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return f"Error analyzing blueprint: {str(e)}"
    
    async def _load_all_pages_optimized(self, document_id: str, 
                                       storage_service: StorageService,
                                       metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """Load all pages with optimizations for large documents"""
        try:
            # Determine total pages
            if metadata and 'page_count' in metadata:
                total_pages = metadata['page_count']
            else:
                # Discover pages by checking what exists
                total_pages = await self._discover_page_count(document_id, storage_service)
            
            pages_to_load = min(total_pages, self.max_pages_to_load)
            logger.info(f"📄 Loading {pages_to_load} pages (total: {total_pages})")
            
            all_page_urls = []
            
            # Process pages in batches for efficiency
            for batch_start in range(0, pages_to_load, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pages_to_load)
                
                # Create tasks for parallel loading
                tasks = []
                for page_num in range(batch_start + 1, batch_end + 1):
                    task = self._load_single_page_optimized(
                        document_id, page_num, storage_service
                    )
                    tasks.append(task)
                
                # Execute batch in parallel
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(batch_results):
                    if isinstance(result, dict) and not isinstance(result, Exception):
                        all_page_urls.append(result)
                    else:
                        logger.warning(f"Failed to load page {batch_start + i + 1}: {result}")
                
                logger.info(f"✅ Loaded batch: pages {batch_start + 1}-{batch_end}")
            
            logger.info(f"✅ Successfully loaded {len(all_page_urls)} pages")
            return all_page_urls
            
        except Exception as e:
            logger.error(f"Error loading pages: {e}")
            return []
    
    async def _discover_page_count(self, document_id: str, storage_service: StorageService) -> int:
        """Discover how many pages exist for a document"""
        # Check for AI-optimized images first, then regular images
        page_count = 0
        
        # Binary search for efficiency
        low, high = 1, 200  # Assume max 200 pages
        
        while low <= high:
            mid = (low + high) // 2
            
            # Check if this page exists
            exists = await storage_service.blob_exists(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_page_{mid}_ai.png"
            )
            
            if not exists:
                # Try regular image
                exists = await storage_service.blob_exists(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_page_{mid}.png"
                )
            
            if exists:
                page_count = mid
                low = mid + 1
            else:
                high = mid - 1
        
        logger.info(f"📄 Discovered {page_count} pages for document")
        return page_count
    
    async def _load_single_page_optimized(self, document_id: str, page_num: int, 
                                         storage_service: StorageService) -> Optional[Dict]:
        """Load a single page with optimization"""
        try:
            # Try AI-optimized version first
            blob_name = f"{document_id}_page_{page_num}_ai.png"
            
            try:
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
            except:
                # Fall back to regular image
                blob_name = f"{document_id}_page_{page_num}.png"
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=blob_name
                )
                
                # Optimize on the fly if needed
                image_bytes = await self._optimize_image_for_ai(image_bytes)
            
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return {
                "page": page_num,
                "url": f"data:image/png;base64,{image_b64}",
                "size_kb": len(image_bytes) / 1024
            }
            
        except Exception as e:
            logger.error(f"Failed to load page {page_num}: {e}")
            return None
    
    async def _optimize_image_for_ai(self, image_bytes: bytes) -> bytes:
        """Optimize image for AI processing to reduce memory usage"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._compress_image,
            image_bytes
        )
    
    def _compress_image(self, image_bytes: bytes) -> bytes:
        """Compress image while maintaining readability"""
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            
            # Resize if too large
            if max(img.size) > self.max_image_dimension:
                img.thumbnail(
                    (self.max_image_dimension, self.max_image_dimension), 
                    Image.Resampling.LANCZOS
                )
            
            # Save with optimization
            output = io.BytesIO()
            img.save(
                output, 
                format='JPEG', 
                quality=self.image_quality, 
                optimize=True
            )
            
            compressed = output.getvalue()
            
            # Log compression
            compression_ratio = (1 - len(compressed) / len(image_bytes)) * 100
            if compression_ratio > 0:
                logger.debug(f"Image compressed by {compression_ratio:.1f}%")
            
            return compressed
            
        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return image_bytes
    
    async def _analyze_blueprint_professionally(self, prompt: str, document_text: str, 
                                               image_urls: List[Dict[str, any]] = None, 
                                               document_id: str = None,
                                               author: str = None) -> str:
        """Professional blueprint analysis with optimized multi-page support"""
        try:
            # Log analysis details
            logger.info("="*50)
            logger.info("📊 PROFESSIONAL BLUEPRINT ANALYSIS")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"📝 Text Data: {'Available' if document_text else 'None'}")
            logger.info(f"🖼️ Images: {len(image_urls) if image_urls else 0} pages")
            
            # Calculate total image size
            if image_urls:
                total_size_mb = sum(img.get('size_kb', 0) for img in image_urls) / 1024
                logger.info(f"💾 Total image size: {total_size_mb:.1f}MB")
            
            logger.info("="*50)
            
            # Professional system message (keeping your exact message)
            system_message = {
                "role": "system",
                "content": """You are a professional blueprint analyst with extensive experience across all construction trades. You analyze MULTI-SHEET blueprint sets, ALWAYS provide code-based recommendations when information is missing, AND ask clarifying questions.

🏗️ COMPREHENSIVE ANALYSIS APPROACH:

1. ANALYZE ALL SHEETS PROVIDED
• Identify what's on each sheet
• Cross-reference between sheets
• Note what's missing

2. ALWAYS PROVIDE COMPLETE ANSWERS USING CODES
• If sizes aren't shown → Use code minimums and typical sizes
• If quantities are missing → Calculate per code requirements
• Never say "not enough information" → Give code-based answer

3. ASK CLARIFYING QUESTIONS
• To verify assumptions
• To find additional sheets
• To provide better accuracy

📍 RESPONSE FORMAT:

"Analyzing [number] sheets for [address] (Scale: [scale] from title block):

**Sheets Provided:**
• Sheet [number]: [description]
• [List all sheets identified]

**Drawing Analysis:**
From Sheet [number]:
• [Specific findings with citations]
• [Counts, measurements, locations]

**Building Code Requirements - [Local code based on address]:**
• [Specific requirements with section numbers]
• [Standard sizes and minimums]
• [Calculations based on code]

**Calculations:**
Based on drawing + code requirements:
• [Show all math]
• [Include code assumptions]
• = **[Actionable result]**

**Professional Recommendations:**
• [What to order/build based on analysis]
• [Code minimums if specifics not shown]
• [Industry standard practices]

**To refine these recommendations, I need clarification:**
1. [Question about specific sheets]
2. [Question about project requirements]
3. [Question to verify assumptions]

[Explain how answers would improve accuracy]"

🎯 EXAMPLE - COMPLETE RESPONSE WITH CODES + QUESTIONS:

"Analyzing Sheet AW-1.05 for 4572 Dawson Street, Burnaby, BC (Scale: 1/8" = 1'-0"):

**Sheets Provided:**
• Sheet AW-1.05: Level P3 Overall Floor Plan (Architectural)

**Drawing Analysis:**
From Sheet AW-1.05:
• Column grid: W2-W9 x WA-WE [shown on plan]
• Column count: 25 columns at grid intersections
• Column sizes: NOT SHOWN on architectural
• Area: 2,720.2 m² (29,277 sq ft) [stated on drawing]
• Parking: 87 stalls [per summary box]

**Building Code Requirements - 2018 BCBC (Burnaby):**
• CSA A23.3-14 Clause 10.5: Minimum column 300mm for buildings
• CSA A23.3-14 Table 10: Seismic Category D requirements
• NBC Table 4.1.5.10: 40 PSF live load for S-2 parking
• Industry standard: 600mm x 600mm for parking columns
• Concrete: 25 MPa minimum per BCBC Table 9.3.1.1

**Calculations:**
Using code minimums since sizes not shown:
• Assume 600mm x 600mm columns (typical for parking)
• Height: 3.0m floor-to-floor (standard parking)
• Volume: 0.6 × 0.6 × 3.0 = 1.08 m³ per column
• Total: 25 columns × 1.08 = 27.0 m³
• Add 10% waste: 29.7 m³ = **33 cubic yards**

**Professional Recommendations:**
1. Order 33 cubic yards of 25 MPa concrete for columns
2. Each column requires:
   - Vertical: 8-25M bars minimum (1% reinforcement)
   - Ties: 10M @ 300mm o.c. (150mm at top/bottom)
   - Approximately 150 kg rebar per column
3. Formwork: 600mm × 600mm × 3000mm = 7.2 m² per column

**To refine these recommendations, I need clarification:**
1. Do you have structural drawings (S2.1-S2.5)? These would show:
   - Exact column sizes (might be 500mm or 700mm)
   - Actual reinforcement schedules
   - Special requirements at transfer levels

2. What level is this for?
   - P3 continuing to P4? (full height columns)
   - P3 only? (might have different details)
   - Top of parking? (might have transfers)

3. Are there any special conditions?
   - Equipment loads requiring larger columns?
   - Architectural features requiring specific sizes?
   - Seismic joints requiring special details?

With structural drawings, I can provide exact sizes rather than typical assumptions, potentially saving 10-20% on concrete if columns are smaller than assumed."

CRITICAL BEHAVIORS:
• ALWAYS provide usable answers even without complete info
• ALWAYS cite specific code sections and requirements  
• ALWAYS calculate quantities using code minimums if needed
• ALWAYS ask questions that would improve accuracy
• NEVER just say "information not available"
• REFERENCE multiple sheets when provided
• EXPLAIN the value of additional information"""
            }
            
            messages = [system_message]
            
            # Build user message with optimized multi-page support
            user_message = {"role": "user", "content": []}
            
            # Add page images efficiently
            if image_urls:
                # Add images
                for page_data in image_urls:
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {"url": page_data["url"], "detail": "high"}
                    })
                
                # Log pages being sent
                page_numbers = [p["page"] for p in image_urls]
                logger.info(f"📤 Sending pages: {page_numbers[:10]}{'...' if len(page_numbers) > 10 else ''}")
            
            # Add comprehensive query
            query_text = f"""Document: {document_id}
Question: {prompt}

MULTI-SHEET ANALYSIS INSTRUCTIONS:
1. Identify ALL sheets provided (look at title blocks)
2. Note which sheet contains what information
3. Cross-reference between sheets when applicable
4. Cite specific sheet numbers for all information
5. Ask about sheets that would provide missing information

Total pages provided: {len(image_urls) if image_urls else 0}
Text from all pages: {'Available' if document_text else 'Not available'}

Drawing text content (from all sheets):
{document_text}"""
            
            user_message["content"].append({"type": "text", "text": query_text})
            messages.append(user_message)
            
            logger.info("📤 Requesting professional analysis")
            
            # Get AI response with retry logic for large documents
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=4000,
                            temperature=0.0  # Consistent, professional responses
                        )
                    )
                    
                    ai_response = response.choices[0].message.content
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if "context_length_exceeded" in str(e) and retry_count <= max_retries:
                        logger.warning(f"Context too large, retrying with fewer images ({retry_count}/{max_retries})")
                        # Reduce image count by 20%
                        reduce_by = int(len(image_urls) * 0.2)
                        if reduce_by > 0:
                            image_urls = image_urls[:-reduce_by]
                            # Rebuild message with fewer images
                            user_message["content"] = user_message["content"][-1:]  # Keep only text
                            for page_data in image_urls:
                                user_message["content"].insert(0, {
                                    "type": "image_url",
                                    "image_url": {"url": page_data["url"], "detail": "high"}
                                })
                    else:
                        raise
            
            # Verify response quality
            logger.info("="*50)
            logger.info("✅ ANALYSIS COMPLETE")
            
            # Check key elements
            has_location = any(term in ai_response.lower() for term in ['located at', 'address', 'burnaby', 'vancouver'])
            has_scale = 'scale:' in ai_response.lower()
            has_counts = bool(re.findall(r'\*\*\d+', ai_response))
            
            logger.info(f"📍 Location identified: {'YES' if has_location else 'NO'}")
            logger.info(f"📐 Scale referenced: {'YES' if has_scale else 'NO'}")
            logger.info(f"🔢 Specific counts: {'YES' if has_counts else 'NO'}")
            logger.info(f"📏 Response length: {len(ai_response)} characters")
            logger.info("="*50)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"Error performing analysis: {str(e)}"
    
    def get_professional_capabilities(self) -> Dict[str, List[str]]:
        """Return professional capabilities of the service"""
        return {
            "building_codes": [
                "2018 BCBC (British Columbia Building Code)",
                "NBC (National Building Code of Canada)",
                "CSA Standards",
                "NFPA Standards",
                "Local Municipal Codes"
            ],
            "engineering_disciplines": [
                "Architectural",
                "Structural", 
                "Mechanical (HVAC)",
                "Electrical",
                "Plumbing",
                "Fire Protection",
                "Civil/Site"
            ],
            "analysis_features": [
                "Multi-page document support",
                "Cross-trade coordination",
                "Code compliance checking",
                "Quantity takeoffs",
                "Professional calculations",
                "Drawing cross-referencing"
            ],
            "optimization_features": [
                f"Handles up to {self.max_pages_to_load} pages",
                f"Parallel loading in batches of {self.batch_size}",
                "Image compression for efficiency",
                "Automatic retry on context limits",
                "Optimized for large documents"
            ]
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the professional AI service
ProfessionalAIService = ProfessionalBlueprintAI
AIService = ProfessionalBlueprintAI
EnhancedAIService = ProfessionalBlueprintAI
