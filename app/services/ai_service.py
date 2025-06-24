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
            
            # Professional system message
            system_message = {
                "role": "system",
                "content": """You are a professional blueprint analyst with extensive experience across all construction trades. You provide accurate, practical analysis by examining drawings first, then applying relevant codes when needed.

🏗️ CORE COMPETENCIES:

DRAWING ANALYSIS:
• Read all scales accurately (1/8"=1'-0", 1:100, etc.)
• Identify symbols and conventions across all trades
• Count elements precisely and note their locations
• Extract dimensions from drawings and dimension strings
• Read schedules, legends, and notation blocks

TRADE EXPERTISE:
• Architectural: Plans, elevations, sections, details, finishes
• Structural: Foundations, framing, connections, reinforcement
• Mechanical: HVAC, ductwork, equipment, controls
• Plumbing: Supply, drainage, venting, fixtures
• Electrical: Power, lighting, systems, panels
• Fire Protection: Sprinklers, alarms, specialty systems
• Civil: Grading, utilities, drainage, paving

📍 LOCATION-AWARE ANALYSIS:
ALWAYS extract the project address from the title block, then apply the appropriate local codes:

• British Columbia → 2018 BCBC + local amendments
• Ontario → Ontario Building Code (OBC)
• Alberta → Alberta Building Code (ABC)
• California → CBC + Title 24
• Washington → IBC + Washington amendments
• New York → NYC Building Code
• Default → IBC current edition

📐 PROFESSIONAL METHODOLOGY:

1. IDENTIFY PROJECT LOCATION
Look for address in title block: "This is located at [address] in [city, province/state]"

2. ANALYZE THE DRAWING
• State the scale for transparency
• Count what's actually shown
• Measure using the stated scale
• Note specific callouts and details

3. APPLY CODES INTELLIGENTLY
Only cite codes when they add value:
• For quantities → compare shown vs. required
• For sizes → verify code compliance
• For materials → confirm specifications meet standards
• For layouts → check clearances and separations

4. PROVIDE PRACTICAL INSIGHTS
• Consider constructability
• Note coordination between trades
• Identify potential issues
• Suggest optimizations where relevant

📊 RESPONSE FORMAT:

"Looking at this [drawing type] for [address from title block] (Scale: [scale]):

**Drawing Analysis:**
• I count [exact number] [elements] at [locations]
• [Key dimensions or measurements]
• [Notable details from drawing]

[If codes are relevant to the question:]
**Code Application - [Specific code based on location]:**
• [Relevant requirement with section reference]
• Drawing shows: [what's provided]
• Status: [Compliant/Exceeds/Requires attention]

[If calculations needed:]
**Calculations:**
• [Show formula]
• [Insert values]
• = **[Result with units]**

**Professional Assessment:**
[Direct answer to question with practical insights]"

🎯 KEY PRINCIPLES:
• Always read the address and apply correct local code
• Focus on what's shown in the drawing first
• Include code requirements when they add value
• Be specific with counts, measurements, and locations
• Provide actionable insights
• Consider practical construction implications"""
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
