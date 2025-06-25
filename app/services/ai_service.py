# app/services/ai_service.py - PROFESSIONAL BLUEPRINT ANALYSIS AI

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
    """Professional AI service for comprehensive blueprint analysis across all trades"""
    
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
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process blueprint queries with professional analysis"""
        try:
            logger.info(f"📐 Processing blueprint analysis for {document_id}")
            
            # Load document context with multi-page support
            document_text = ""
            image_urls = []  # List to hold multiple page images
            
            try:
                # Load text context (contains text from ALL pages)
                context_task = storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                document_text = await asyncio.wait_for(context_task, timeout=30.0)
                logger.info(f"✅ Loaded text from all pages: {len(document_text)} characters")
                
                # Try to load ALL page images
                page_num = 1
                max_pages = 20  # Reasonable limit
                
                while page_num <= max_pages:
                    try:
                        image_task = storage_service.download_blob_as_bytes(
                            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                            blob_name=f"{document_id}_page_{page_num}.png"
                        )
                        page_bytes = await asyncio.wait_for(image_task, timeout=30.0)
                        page_b64 = base64.b64encode(page_bytes).decode('utf-8')
                        image_url = f"data:image/png;base64,{page_b64}"
                        image_urls.append({
                            "page": page_num,
                            "url": image_url
                        })
                        logger.info(f"✅ Loaded page {page_num} image")
                        page_num += 1
                    except:
                        # No more pages found
                        break
                
                logger.info(f"✅ Total pages loaded: {len(image_urls)}")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return "Unable to load the blueprint. Please ensure the document is properly uploaded and processed."
            
            # Process with professional analysis
            result = await self._analyze_blueprint_professionally(
                prompt=prompt,
                document_text=document_text,
                image_urls=image_urls,  # Now passing list of images
                document_id=document_id,
                author=author
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return f"Error analyzing blueprint: {str(e)}"
    
    async def _analyze_blueprint_professionally(self, prompt: str, document_text: str, 
                                               image_urls: List[Dict[str, any]] = None, document_id: str = None,
                                               author: str = None) -> str:
        """Professional blueprint analysis with multi-page support"""
        try:
            # Log analysis details
            logger.info("="*50)
            logger.info("📊 PROFESSIONAL BLUEPRINT ANALYSIS")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"📝 Text Data: {'Available' if document_text else 'None'}")
            logger.info(f"🖼️ Images: {len(image_urls) if image_urls else 0} pages")
            logger.info("="*50)
            
            # Professional system message with multi-sheet awareness AND code recommendations
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
            
            # Build user message with multi-page support
            user_message = {"role": "user", "content": []}
            
            # Add all page images if available
            if image_urls:
                for page_data in image_urls:
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {"url": page_data["url"], "detail": "high"}
                    })
                    
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
            
            # Get AI response
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
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the professional AI service
ProfessionalAIService = ProfessionalBlueprintAI
AIService = ProfessionalBlueprintAI
EnhancedAIService = ProfessionalBlueprintAI
