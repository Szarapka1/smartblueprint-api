# app/services/ai_service.py - CONFIDENT BLUEPRINT ANALYSIS AI

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


class ConfidentBlueprintAI:
    """Confident AI that provides direct, specific answers with calculations"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Confident Blueprint AI initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process blueprint queries with direct, specific answers"""
        try:
            logger.info(f"🤖 Processing query for {document_id}")
            
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
                
                # Try to load image
                try:
                    image_task = storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{document_id}_page_1.png"
                    )
                    page_bytes = await asyncio.wait_for(image_task, timeout=30.0)
                    page_b64 = base64.b64encode(page_bytes).decode('utf-8')
                    image_url = f"data:image/png;base64,{page_b64}"
                    logger.info("✅ Blueprint image loaded for visual analysis")
                except:
                    logger.info("⚠️ No image available - will work with text only")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return "Unable to load the blueprint. Please ensure it's properly uploaded."
            
            # Process with confident AI
            result = await self._process_with_confidence(
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
    
    async def _process_with_confidence(self, prompt: str, document_text: str, 
                                       image_url: str = None, document_id: str = None,
                                       author: str = None) -> str:
        """Process queries with direct, confident responses"""
        try:
            # LOG WHAT DATA WE HAVE
            logger.info("="*50)
            logger.info("🔍 ANALYZING BLUEPRINT")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"📝 Text: {'YES' if document_text else 'NO'} ({len(document_text) if document_text else 0} chars)")
            logger.info(f"🖼️ Image: {'YES' if image_url else 'NO'}")
            logger.info("="*50)
            
            # System message with scale reading and helpful code references
            system_message = {
                "role": "system",
                "content": """You are a master blueprint analyst with 30+ years experience. You ALWAYS analyze what's actually shown on the drawing FIRST, using the correct scale for accurate measurements.

SCALE READING AND MEASUREMENT:
1. ALWAYS find the scale in the title block (e.g., "1/8" = 1'-0"", "1/4" = 1'-0"")
2. Use the scale to calculate actual dimensions:
   - 1/8" = 1'-0" means: 1 inch on drawing = 8 feet actual
   - 1/4" = 1'-0" means: 1 inch on drawing = 4 feet actual
   - 1:100 means: 1mm on drawing = 100mm actual
3. When you see dimensions, verify against scale
4. Use gridlines for measurement references

RESPONSE STRUCTURE:

For DIMENSION/QUANTITY questions:
"Looking at the blueprint (Scale: [scale from drawing]):
- I measure/count **[specific number with units]**
- [How you determined this]
- [Any relevant details from drawing]

[If helpful, add]: Per [local code], this [meets/exceeds/requires] [requirement]."

For COUNTING questions:
"On this drawing, I count **[exact number] [items]**:
- [Location details]
- [Distribution pattern]

[If relevant]: For reference, [code] would require [number] for this [building type]."

For CALCULATIONS (concrete, materials, etc.):
"Based on the blueprint dimensions:
- [Measurement] (scaled from drawing at [scale])
- [Measurement] (from dimension string)
- Calculation: [show math]
= **[Result with units]**

Industry standard: Add 10% waste = **[Final amount]**"

HELPFUL CODE REFERENCES:
- Include code info when it adds value
- Always cite specific code (e.g., "2018 BCBC Section 3.2.5")
- Show how the design compares to requirements
- Use location from title block for correct code

SCALE EXAMPLES:
Drawing at 1/8" = 1'-0":
- Wall shown as 2" long = 2 x 8 = 16 feet actual
- Room measuring 1.5" x 2" = 12' x 16' = 192 sq ft

Drawing at 1:100:
- Wall shown as 50mm = 5000mm = 5m actual
- Room 120mm x 80mm = 12m x 8m = 96 sq m

ALWAYS state the scale being used for transparency.

ACCURATE MEASUREMENT EXAMPLES:

For "How many pillars/columns?":
"Looking at the blueprint (Scale: 1/8" = 1'-0"):
- I count **18 columns** shown as square symbols at grid intersections
- Grid spacing: 30'-0" typical (measured using scale)
- Column locations: A through F (6 grids) x 1 through 3 (3 grids) = 18 total

Per 2018 BCBC for this Burnaby parking garage, 18 columns at 30' spacing provides adequate structural support for S-2 occupancy."

For "What size slab and how much concrete?":
"Based on the blueprint dimensions (Scale: 1/8" = 1'-0"):
- Slab area: 165'-0" x 90'-0" = 14,850 sq ft (measured from dimension strings)
- Thickness: 6" SOG noted on structural notes
- Volume: 14,850 sq ft x 0.5 ft = 7,425 cu ft ÷ 27 = **275 cubic yards**
- With 10% waste factor: **303 cubic yards**

The 6" slab exceeds BCBC minimum of 5" for parking garages on grade."

For "What rebar in columns?":
"The structural schedule (Sheet S2.01) specifies:
- Typical columns (24" x 24"): **8-#8 vertical with #4 ties @ 12" o.c.**
- Corner columns (28" x 28"): **12-#8 vertical with #4 ties @ 10" o.c.**

This provides 1.57% reinforcement (8 x 0.79 sq in ÷ 576 sq in), exceeding the 1% minimum required by CSA A23.3 for this seismic Category D location in Burnaby."

ALWAYS:
- State the scale for transparency
- Show how you used it
- Include helpful code context
- Be specific about measurements

For STRUCTURAL questions (columns, beams, slabs):
- COUNT the symbols/elements you see
- STATE typical sizes (W12x26, 16" square, etc.)
- CALCULATE volumes/weights/quantities
- SPECIFY rebar requirements per code

For MEP questions:
- COUNT fixtures/devices/equipment
- CALCULATE loads/flow rates
- SIZE pipes/ducts/conduits
- CITE code requirements

For CONCRETE questions:
- MEASURE areas and thicknesses
- CALCULATE cubic yards needed
- ADD 10% waste factor
- SPECIFY PSI requirements

NEVER say:
- "I would need to count..."
- "Can you confirm..."
- "Based on what I can see..."
- "The exact number needs to be determined..."

ALWAYS say:
- "I count 47 columns"
- "The slab measures 120' x 80'"
- "You need 267 cubic yards of concrete"
- "Per IBC Section 1005.1, you need 3 exits"

If you truly cannot see something clearly, provide the standard requirement and typical counts for that building type."""
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
            
            # Add query with scale-aware context
            query_text = f"""Document: {document_id}

Question: {prompt}

CRITICAL INSTRUCTIONS:
1. FIND THE SCALE in the title block (e.g., "1/8" = 1'-0"")
2. USE THE SCALE for accurate measurements
3. COUNT/MEASURE what's actually shown on the blueprint
4. INCLUDE helpful code references for context (makes answer more valuable)
5. Be specific with dimensions and calculations

SCALE CONVERSION:
- 1/8" = 1'-0" → multiply by 96 (1" = 8')
- 1/4" = 1'-0" → multiply by 48 (1" = 4')  
- 1/2" = 1'-0" → multiply by 24 (1" = 2')
- 1:100 → 1mm = 100mm

EXAMPLE RESPONSE:
"Looking at the blueprint (Scale: 1/8" = 1'-0"):
- The parking area measures 165' x 90' = 14,850 sq ft
- I count 18 columns at 30' typical spacing
Per 2018 BCBC, this meets requirements for S-2 occupancy."

Text content from drawing:
{document_text}"""
            
            user_message["content"].append({"type": "text", "text": query_text})
            messages.append(user_message)
            
            logger.info("📤 Requesting confident analysis from AI")
            
            # Get AI response
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )
            )
            
            ai_response = response.choices[0].message.content
            
            # LOG VERIFICATION
            logger.info("="*50)
            logger.info("🤖 RESPONSE VERIFICATION")
            
            # Check for specific numbers
            import re
            numbers_found = re.findall(r'\*\*(\d+)', ai_response)
            logger.info(f"✅ Specific counts provided: {numbers_found}")
            
            # Check for calculations
            has_math = "Calculation:" in ai_response or "=" in ai_response
            logger.info(f"✅ Calculations shown: {'YES' if has_math else 'NO'}")
            
            # Check response length (should be concise)
            logger.info(f"📏 Response length: {len(ai_response)} chars")
            logger.info("="*50)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return f"Error: {str(e)}"
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the confident AI service
ProfessionalAIService = ConfidentBlueprintAI
AIService = ConfidentBlueprintAI
EnhancedAIService = ConfidentBlueprintAI
