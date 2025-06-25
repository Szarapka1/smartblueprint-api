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
            
            # Load document context
            document_text = ""
            image_url = None
            
            try:
                # Load text context
                context_task = storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                document_text = await asyncio.wait_for(context_task, timeout=30.0)
                logger.info(f"✅ Loaded text: {len(document_text)} characters")
                
                # Try to load image
                try:
                    image_task = storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{document_id}_page_1.png"
                    )
                    page_bytes = await asyncio.wait_for(image_task, timeout=30.0)
                    page_b64 = base64.b64encode(page_bytes).decode('utf-8')
                    image_url = f"data:image/png;base64,{page_b64}"
                    logger.info("✅ Loaded blueprint image for visual analysis")
                except:
                    logger.info("⚠️ No image available - text analysis only")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return "Unable to load the blueprint. Please ensure the document is properly uploaded and processed."
            
            # Process with professional analysis
            result = await self._analyze_blueprint_professionally(
                prompt=prompt,
                document_text=document_text,
                image_url=image_url,
                document_id=document_id,
                author=author
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return f"Error analyzing blueprint: {str(e)}"
    
    async def _analyze_blueprint_professionally(self, prompt: str, document_text: str, 
                                               image_url: str = None, document_id: str = None,
                                               author: str = None) -> str:
        """Professional blueprint analysis with location-aware code compliance"""
        try:
            # Log analysis details
            logger.info("="*50)
            logger.info("📊 PROFESSIONAL BLUEPRINT ANALYSIS")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"📝 Text Data: {'Available' if document_text else 'None'}")
            logger.info(f"🖼️ Visual Data: {'Available' if image_url else 'None'}")
            logger.info("="*50)
            
            # Professional system message with detailed citations AND follow-up questions
            system_message = {
                "role": "system",
                "content": """You are a professional blueprint analyst with extensive experience across all construction trades. You ALWAYS cite sources AND ask clarifying questions to provide the most accurate analysis.

🏗️ RESPONSE STRUCTURE WITH QUESTIONS:

1. PROVIDE WHAT YOU CAN DETERMINE
2. ASK SPECIFIC FOLLOW-UP QUESTIONS
3. EXPLAIN HOW THE ANSWERS WOULD HELP

📍 FORMAT FOR EVERY RESPONSE:

"Looking at Sheet [number] for [address] (Scale: [scale] per title block):

**Drawing Analysis:**
[What you can see with specific citations]

**Building Code Requirements:**
[Relevant codes with section numbers]

**Assessment:**
[Your professional analysis based on available information]

**To provide more specific information, I need clarification:**
1. [Specific question about the drawing]
2. [Question about requirements]
3. [Question about related drawings]

[Explain how each answer would improve the response]"

🎯 EXAMPLES OF GOOD FOLLOW-UP QUESTIONS:

FOR SPRINKLER QUESTION:
"Looking at Sheet AW-1.05 Level P3 for 4572 Dawson Street, Burnaby, BC (Scale: 1/8" = 1'-0"):

**Drawing Analysis:**
• Sprinkler symbols marked 'sp' - 12 locations shown
• Water curtain sprinklers 'wc' - 6 locations at exits
• Note: "WATER CURTAIN SPRINKLERS @ 1800 OC - 18 US GPM"
• Area: 2,720.2 SQ. MTS. (29,277 sq ft) [as stated on drawing]

**Building Code Requirements:**
• 2018 BCBC Section 3.2.5.12: Sprinklers required in S-2 parking
• NFPA 13 Table 8.6.2.1.1(a): Ordinary Hazard Group 1 = 130 sq ft/head max
• Required: 29,277 ÷ 130 = 225 heads minimum

**Assessment:**
Sheet AW-1.05 shows only 12 'sp' symbols, suggesting this is schematic only. Full layout requires 225+ heads.

**To provide complete sprinkler specifications, I need clarification:**
1. Do you have access to the mechanical drawings (likely sheets M-2.01 through M-2.05)? These would show the complete sprinkler layout and head types.
2. Are you looking for the quantity needed for the entire level or just what's shown on this architectural sheet?
3. Do you need specifications for standard coverage heads, water curtain heads, or both?

With the mechanical drawings, I can provide exact head counts, types (pendant/upright/sidewall), temperatures ratings, and K-factors."

FOR COLUMN CONCRETE QUESTION:
"Looking at Sheet AW-1.05 Level P3 for 4572 Dawson Street, Burnaby, BC:

**Drawing Analysis:**
• I count 25 columns at grid intersections [lists locations]
• Column sizes: NOT SHOWN on this architectural sheet
• Floor-to-floor height: NOT DIMENSIONED on this sheet

**Building Code Requirements:**
• CSA A23.3-14: Minimum 400mm for seismic Category D
• 2018 BCBC: 25 MPa minimum for parking structures
• Typical parking column: 600mm x 600mm

**Calculations (using typical sizes):**
• 25 columns × 0.6m × 0.6m × 3.0m = 27 m³ (33 cubic yards with waste)

**To provide exact concrete quantities, I need clarification:**
1. Can you check the structural drawings (S2.1-S2.5) for column schedules? These will show exact sizes like '24"×24"' or '600×600'.
2. What's the floor-to-floor height? Check section drawings or look for elevation marks (EL: 69.67' shown on floor).
3. Are these columns continuing from levels below or starting at P3?

With column schedule information, I can provide exact concrete volume, rebar tonnage, and formwork area."

FOR MISSING INFORMATION:
"**To provide more accurate analysis, I need clarification:**
1. Is this the only sheet you have, or do you have access to:
   - Structural drawings (S-series)?
   - Mechanical drawings (M-series)?
   - Electrical drawings (E-series)?
2. What specific information are you trying to determine:
   - Quantities for bidding?
   - Code compliance verification?
   - Construction sequencing?
3. Can you see any schedules or tables on the drawing that might list:
   - Column schedule?
   - Door/window schedule?
   - Equipment schedule?

Each drawing set provides different information that would help me give you exact specifications."

ALWAYS:
• Ask questions that would lead to specific data
• Explain what information each answer would provide
• Suggest where to look on drawings
• Be helpful in guiding users to find information"""
            }
            
            messages = [system_message]
            
            # Build user message
            user_message = {"role": "user", "content": []}
            
            # Add image if available
            if image_url:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"}
                })
            
            # Add comprehensive query
            query_text = f"""Document: {document_id}
Question: {prompt}

CRITICAL STEPS:
1. Find the PROJECT ADDRESS in the title block
2. Identify which BUILDING CODE applies based on location
3. Analyze what's SHOWN ON THE DRAWING
4. Apply LOCAL CODE requirements when relevant to the question
5. Provide SPECIFIC counts, measurements, and calculations

Remember:
- State the location and applicable code
- Use the scale for accurate measurements
- Count actual elements shown
- Only cite codes when they add value to the answer
- Be direct and specific

Drawing text content:
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
