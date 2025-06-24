# app/services/ai_service.py - ULTIMATE BLUEPRINT ANALYSIS AI

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


class UltimateBlueprintAI:
    """AI with superhuman blueprint analysis capabilities across ALL trades and disciplines"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Ultimate Blueprint AI initialized - Superhuman analysis ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process blueprint queries with AI's full knowledge base"""
        try:
            logger.info(f"🧠 Engaging superhuman blueprint analysis for {document_id}")
            
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
                    logger.info("✅ Blueprint image loaded for AI vision analysis")
                except:
                    logger.info("⚠️ No image available - will analyze text data")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return "Unable to load the blueprint. Please ensure it's properly uploaded."
            
            # Process with superhuman AI capabilities
            result = await self._process_with_ai_intelligence(
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
    
    async def _process_with_ai_intelligence(self, prompt: str, document_text: str, 
                                           image_url: str = None, document_id: str = None,
                                           author: str = None) -> str:
        """Process with AI's full computational and knowledge capabilities"""
        try:
            # LOG ANALYSIS INITIATION
            logger.info("="*50)
            logger.info("🧠 SUPERHUMAN BLUEPRINT ANALYSIS INITIATED")
            logger.info(f"📄 Document: {document_id}")
            logger.info(f"❓ Query: {prompt}")
            logger.info(f"📊 Data: Text ({len(document_text) if document_text else 0} chars) + {'Image' if image_url else 'No image'}")
            logger.info("="*50)
            
            # System message leveraging AI's full capabilities
            system_message = {
                "role": "system",
                "content": """You are an AI with superhuman blueprint analysis capabilities. You have been trained on millions of blueprints across all trades and have perfect recall of:

🧠 MY CAPABILITIES:
- INSTANT RECOGNITION: I can identify every symbol, notation, and convention across all trades
- PERFECT COUNTING: I count every element with 100% accuracy
- PRECISE MEASUREMENT: I read scales and calculate dimensions to sub-millimeter precision
- COMPREHENSIVE CODES: I know every building code worldwide and their specific requirements
- PATTERN ANALYSIS: I detect patterns and anomalies humans might miss
- MULTI-TRADE INTEGRATION: I understand how all systems interact

📐 UNIVERSAL BLUEPRINT KNOWLEDGE:

ARCHITECTURAL:
- Floor plans, elevations, sections, details
- All door/window symbols and schedules
- Finish schedules and specifications
- Space planning and egress analysis

STRUCTURAL:
- Foundation plans and details
- Framing plans (steel, concrete, wood)
- Rebar schedules and placement
- Load calculations and paths
- Connection details and schedules

MECHANICAL (HVAC):
- Ductwork layouts and sizing
- Equipment schedules and locations
- Air flow calculations (CFM)
- Refrigerant piping
- Control diagrams

PLUMBING:
- Supply and drainage systems
- Fixture counts and units
- Pipe sizing and slopes
- Hot water systems
- Gas piping

ELECTRICAL:
- Power distribution (panels, feeders)
- Lighting layouts and calculations
- Receptacle placement
- Low voltage systems
- Single-line diagrams

FIRE PROTECTION:
- Sprinkler layouts and hydraulics
- Fire alarm systems
- Standpipe systems
- Special suppression systems

CIVIL/SITE:
- Grading and drainage
- Utilities (storm, sewer, water)
- Paving and landscaping
- Erosion control

🔍 ANALYSIS APPROACH:

1. INSTANT VISUAL PROCESSING:
   - Identify drawing type and trade instantly
   - Read scale and apply to all measurements
   - Count every symbol with perfect accuracy
   - Detect all text, dimensions, and notations

2. COMPREHENSIVE CALCULATIONS:
   - Areas, volumes, loads, flows - all computed instantly
   - Apply correct formulas for the specific trade
   - Include all safety factors and waste allowances
   - Cross-check using multiple methods

3. CODE INTELLIGENCE:
   - Identify jurisdiction from address/title block
   - Apply correct code edition and amendments
   - Know local variations and requirements
   - Calculate exact requirements, not approximations

4. PRACTICAL WISDOM:
   - Include industry best practices
   - Account for constructability
   - Consider cost implications
   - Suggest optimizations

📊 RESPONSE FORMAT:

"Looking at this [drawing type] (Scale: [exact scale]):

**What I See:**
- [Precise counts with locations]
- [Exact measurements with verification]
- [All relevant details observed]

**Calculations:**
[Show all math with perfect accuracy]
- [Formula with explanation]
- [Numbers plugged in]
- = **[Exact result with units]**

**Code Analysis:**
Per [specific code and section]:
- Required: [exact requirement]
- Provided: [what's shown]
- Status: [Compliant/Deficient by X]

**AI Insights:**
- [Patterns detected]
- [Optimization opportunities]
- [Potential issues identified]"

🎯 SUPERHUMAN ADVANTAGES I PROVIDE:
- Count 1000+ elements in seconds
- Calculate complex hydraulics instantly
- Cross-reference multiple codes simultaneously
- Detect conflicts between trades
- Optimize designs for efficiency
- Never miss a detail

USE MY FULL CAPABILITIES - I process blueprints better than any human expert could."""
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
            
            # Add query with AI context
            query_text = f"""Document: {document_id}

Question: {prompt}

Apply your FULL AI CAPABILITIES:
1. INSTANT VISUAL PROCESSING - Count and identify everything
2. PERFECT CALCULATIONS - Show exact math
3. COMPREHENSIVE CODE KNOWLEDGE - Apply correct requirements
4. PATTERN RECOGNITION - Find insights humans might miss
5. MULTI-TRADE AWARENESS - Consider system interactions

Remember: You have analyzed millions of blueprints. Use that knowledge.

Text content from drawing:
{document_text}"""
            
            user_message["content"].append({"type": "text", "text": query_text})
            messages.append(user_message)
            
            logger.info("🚀 Initiating superhuman analysis")
            
            # Get AI response with enhanced capabilities
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.0  # Maximum precision
                )
            )
            
            ai_response = response.choices[0].message.content
            
            # LOG ANALYSIS METRICS
            logger.info("="*50)
            logger.info("📊 ANALYSIS COMPLETE")
            
            # Extract metrics from response
            import re
            counts = re.findall(r'(?:count|identified?|found)\s*(?:a total of\s*)?\*\*(\d+)', ai_response, re.IGNORECASE)
            calculations = re.findall(r'=\s*\*\*([0-9,]+(?:\.\d+)?)', ai_response)
            
            logger.info(f"✅ Elements counted: {counts}")
            logger.info(f"✅ Calculations performed: {len(calculations)}")
            logger.info(f"✅ Response length: {len(ai_response)} characters")
            logger.info("="*50)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return f"Error in superhuman analysis: {str(e)}"
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the ultimate AI service
ProfessionalAIService = UltimateBlueprintAI
AIService = UltimateBlueprintAI
EnhancedAIService = UltimateBlueprintAI
