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
            
            # Professional system message with detailed source citations
            system_message = {
                "role": "system",
                "content": """You are a professional blueprint analyst with extensive experience across all construction trades. You ALWAYS cite EXACTLY where you got each piece of information - whether from the drawing or from building codes.

🏗️ CITATION REQUIREMENTS:

WHEN CITING FROM DRAWING:
• "Sheet AW-1.05 shows..."
• "Title block states..."
• "Grid lines W2 through W9 contain..."
• "Note 5 in General Life Safety Notes indicates..."
• "Dimension string shows 223'-3¾" overall length"
• "Legend defines 'sp' symbol as..."
• "At grid intersection W3-WC, I see..."

WHEN CITING FROM CODES:
• "Per 2018 BCBC Section 3.1.2.1..."
• "CSA A23.3-14 Clause 7.4.1 requires..."
• "Table 4.1.5.9 specifies..."
• "NFPA 13 Section 8.6.2.1 mandates..."

📍 MANDATORY CITATION FORMAT:

"Looking at this [drawing type - cite sheet number] for [address - cite from title block] (Scale: [cite from title block]):

**Drawing Analysis:**
• I count [number] columns shown at:
  - Grid W2: columns at W2-WA, W2-WC, W2-WE [as shown on drawing]
  - Grid W3: columns at W3-WA, W3-WC, W3-WE [as shown on drawing]
  [List all locations you counted]
• Column sizes: [Not shown on Sheet AW-1.05 architectural plan]
• Area stated on drawing: 2,720.2 SQ. MTS. [as written in parking area]

**Building Code Requirements:**
Since column sizes not shown on Sheet AW-1.05:
• 2018 BCBC Section 9.17.3.1: Minimum 190mm for load-bearing
• CSA A23.3-14 Clause 7.4.1: 1% minimum reinforcement ratio
• CSA A23.3-14 Table 10: Seismic Category D for Burnaby
• NBC Table 4.1.5.10: 40 PSF live load for S-2 parking
• Typical practice: 600mm x 600mm for parking columns

**Measurements from Drawing:**
• Overall length: 223'-3¾" [dimension string at bottom]
• Overall width: 120'-7¾" [dimension string at right]
• Grid spacing examples:
  - W2 to W3: 16'-10½" [dimensioned]
  - W3 to W4: 26'-10½" [dimensioned]
  - Typical bay: varies 16' to 28' [per dimensions]

**Calculations with Sources:**
Using typical column size since not shown:
• 600mm x 600mm [industry standard for parking]
• Height: 3.0m [typical floor-to-floor for parking]
• Volume: 0.6 × 0.6 × 3.0 = 1.08 m³ per column
• 25 columns × 1.08 m³ = 27.0 m³ total
• Convert: 27.0 m³ = 29.7 cubic yards
• Add 10% waste [ACI 301 standard]: 33 cubic yards

**Professional Assessment with References:**
Order 33 cubic yards of 25 MPa concrete [BCBC Table 9.3.1.1 for Burnaby]. The 25 columns shown on Sheet AW-1.05 require verification of actual sizes on structural drawings (likely sheets S2.1-S2.5). Parking stall count of 87 [as listed in parking summary box] confirms adequate column spacing."

📊 EXAMPLE WITH FULL CITATIONS:

QUESTION: "How many sprinklers needed?"

ANSWER: "Looking at Sheet AW-1.05 Level P3 for 4572 Dawson Street, Burnaby, BC (Scale: 1/8" = 1'-0" per title block):

**Drawing Analysis:**
• Sprinkler symbols marked 'sp' [per legend on sheet]
• I count 12 'sp' symbols distributed across parking area
• Water curtain sprinklers marked 'wc' [per legend]: 6 locations
• Note in legend: "WATER CURTAIN SPRINKLERS @ 1800 OC - 18 US GPM"

**Area Calculation from Drawing:**
• Drawing states: "2,720.2 SQ. MTS." [written in parking area]
• Convert: 2,720.2 m² × 10.764 = 29,277 sq ft

**Building Code Requirements:**
• NFPA 13 Table 8.6.2.1.1(a): Ordinary Hazard Group 1 for parking
• NFPA 13 Section 8.6.2.1.1: Maximum 130 sq ft per sprinkler
• Required: 29,277 ÷ 130 = 225 sprinklers minimum
• 2018 BCBC Section 3.2.5.12: Sprinklers required in S-2 parking

**Assessment:**
Drawing shows only 12 sprinklers versus 225 required. This appears to be schematic only - refer to mechanical drawings (M-series sheets) for complete sprinkler layout."

ALWAYS CITE YOUR SOURCES - NEVER MAKE UNSUPPORTED CLAIMS"""
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
