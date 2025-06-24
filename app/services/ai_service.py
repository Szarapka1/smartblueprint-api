# app/services/ai_service.py - INTERACTIVE BLUEPRINT ANALYSIS AI

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


class InteractiveBlueprintAI:
    """Interactive AI that asks clarifying questions and provides comprehensive multi-perspective answers"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Interactive Blueprint AI initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process blueprint queries with interactive clarification when needed"""
        try:
            logger.info(f"🤖 Processing interactive query for {document_id}")
            
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
                return self._generate_document_error_response()
            
            # Process with interactive AI
            result = await self._process_with_interaction(
                prompt=prompt,
                document_text=document_text,
                image_url=image_url,
                document_id=document_id,
                author=author
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return self._generate_error_response(str(e))
    
    async def _process_with_interaction(self, prompt: str, document_text: str, 
                                       image_url: str = None, document_id: str = None,
                                       author: str = None) -> str:
        """Process queries with comprehensive responses and clarifying questions when needed"""
        try:
            # System message that enforces comprehensive responses and interactive clarification
            system_message = {
                "role": "system",
                "content": """You are an expert blueprint analyst with 30+ years of experience. You ALWAYS provide comprehensive, multi-perspective responses, and when you need clarification, you ask specific questions.

RESPONSE FRAMEWORK - USE FOR EVERY QUERY:

1. **INTERPRET THE QUESTION**
   - Identify ALL possible interpretations
   - If ambiguous, list the interpretations and ask which one they meant
   - Example: "This question could mean: (a) actual count shown, (b) code-required amount, or (c) recommended quantity. Which perspective would you like?"

2. **COMPREHENSIVE ANALYSIS STRUCTURE**
   When you CAN answer (even partially):
   
   **A. What's SHOWN on the Drawing**
   - "After performing a visual analysis of the blueprint..."
   - Count actual elements/symbols
   - Note locations (use grid references if available)
   - Describe patterns or distribution
   
   **B. What's REQUIRED by Code**
   - State applicable codes (NFPA 13, IBC, NEC, etc.)
   - Show calculations step-by-step
   - Include formulas with explanations
   - Provide specific requirements
   
   **C. Industry Best Practices**
   - What experienced professionals typically do
   - Common variations or alternatives
   - Practical considerations
   
   **D. Calculations & Analysis**
   - Show ALL math clearly
   - Use proper units
   - Include safety factors
   - Explain assumptions

3. **WHEN YOU NEED CLARIFICATION**
   Ask SPECIFIC questions about:
   
   **Visual Elements:**
   - "I need to see the drawing more clearly. Can you tell me:
     • What type of drawing is this? (floor plan, elevation, section, detail)
     • What's the scale shown in the title block?
     • Are there grid lines? If so, what are they labeled?
     • Do you see a legend or symbol list?"
   
   **Specific Counts:**
   - "To count [items] accurately, please help me identify:
     • What symbols represent [items]? (describe their appearance)
     • Which areas of the drawing show these symbols?
     • Are they labeled with tags or numbers?"
   
   **Drawing Details:**
   - "For code compliance calculations, I need:
     • The total square footage (usually shown in title block or as a dimension)
     • The occupancy type (office, retail, warehouse, etc.)
     • Ceiling height if shown
     • Any special conditions noted"
   
   **Missing Information:**
   - "The drawing might not show everything. Please check:
     • Is this the only sheet, or part of a set?
     • Are there detail callouts referencing other sheets?
     • Are there schedules or tables on the drawing?"

4. **RESPONSE FORMAT**
   
   Start with: "Certainly. Let me analyze this comprehensively..."
   
   Then either:
   a) Provide the full multi-perspective analysis, OR
   b) Say: "To provide the most accurate analysis, I need some clarification:"
      [List specific questions]
      
      "However, based on what I can determine so far:"
      [Provide partial analysis with caveats]

5. **EXAMPLE RESPONSES**

   **When you CAN fully answer:**
   "Certainly. This question can be interpreted in multiple ways:
   1. How many [items] are *shown* on the current drawing?
   2. How many [items] are *required* by building codes?
   3. How many [items] are *recommended* for optimal performance?
   
   I'll analyze all three perspectives:
   
   **1. [Items] Shown on Drawing**
   After performing a visual analysis, I've identified **X [items]** marked as [description of symbols]...
   
   **2. Code Requirements ([Standard])**
   • Occupancy Classification: [type]
   • Applicable Code: [specific section]
   • Calculation:
     - Formula: [show formula]
     - Values: [plug in numbers]
     - Result: **Y [items] required**
   
   **3. Best Practice Recommendations**
   Industry professionals typically...
   
   **Conclusion:**
   The drawing shows X, code requires Y, best practice suggests Z."
   
   **When you need clarification:**
   "I'd be happy to analyze the sprinkler head requirements comprehensively. To provide the most accurate analysis, I need some clarification:
   
   1. **Drawing Type**: What type of drawing are you looking at? (floor plan, reflected ceiling plan, or sprinkler riser diagram?)
   
   2. **Visual Symbols**: Can you see circular symbols on the ceiling/plan? They might look like:
      - Simple circles (○)
      - Circles with a dot (⊙)
      - Circles with lines extending down
      - Letters 'SP' or 'S' in circles
   
   3. **Space Information**: To calculate code requirements, I need:
      - Total square footage (check title block or large dimension)
      - Building use (office, retail, warehouse, etc.)
      - Any areas labeled as different hazard levels?
   
   **Based on standard practices while I await your clarification:**
   For a typical office space (Light Hazard per NFPA 13):
   - Maximum coverage: 200 sq ft per head
   - For a 10,000 sq ft space: 50 heads minimum
   - Actual requirements vary based on obstructions and ceiling configuration
   
   Please provide the above details so I can give you exact counts and requirements."

CRITICAL RULES:
- ALWAYS provide multi-perspective analysis when possible
- ALWAYS show calculations and cite codes
- When uncertain, ask SPECIFIC visual questions
- Never just say "I can't see" - explain what you need to see
- Even with limited info, provide useful context and standard practices
- Guide the user to help you see what's important
- Use **bold** for key findings and numbers
- Maintain professional expertise throughout"""
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
            
            # Add query with context
            query_text = f"""Document: {document_id}
Author: {author or 'Anonymous'}

Query: {prompt}

Available Information:
- Document text: {'Available' if document_text else 'Not available'}
- Visual image: {'Available' if image_url else 'Not available'}
- Text preview: {document_text[:500] + '...' if len(document_text) > 500 else document_text}"""
            
            user_message["content"].append({"type": "text", "text": query_text})
            messages.append(user_message)
            
            # Get AI response
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1  # Lower temperature for consistency
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return self._generate_error_response(str(e))
    
    def _generate_document_error_response(self) -> str:
        """Generate helpful response when document can't be loaded"""
        return """I'm unable to access the blueprint document. To help you analyze blueprints effectively, I need:

**Document Requirements:**
1. A properly uploaded PDF blueprint
2. The document to be processed and cached

**To proceed, please:**
1. Ensure your blueprint PDF is uploaded correctly
2. Verify the document ID is correct
3. Check that the document has finished processing

**What I can help with once the document is ready:**
- Count specific elements (doors, windows, fixtures, etc.)
- Calculate code requirements (egress, fire protection, accessibility)
- Analyze spatial layouts and dimensions
- Verify compliance with building codes
- Perform quantity takeoffs
- Answer technical questions about the design

Please upload your blueprint and try again, or contact support if you continue to have issues."""
    
    def _generate_error_response(self, error: str) -> str:
        """Generate helpful error response"""
        return f"""I encountered an error while analyzing your blueprint query. 

**Error Details:**
{error}

**To help me assist you better:**

1. **If asking about specific elements:**
   - Describe what you're looking for (e.g., "round symbols labeled 'S'")
   - Mention which drawing sheet you're viewing
   - Note any relevant grid references

2. **If asking about calculations:**
   - Provide the space type (office, retail, warehouse, etc.)
   - Include any dimensions or square footage shown
   - Specify which code or standard you need to follow

3. **If the drawing is unclear:**
   - Describe what you can see
   - Note the drawing title and scale
   - Mention any schedules or legends visible

**I can still help with:**
- General code requirements and standards
- Typical calculation methods
- Industry best practices
- What to look for on blueprints

Please provide more details about your specific question, and I'll do my best to help!"""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the interactive AI service
ProfessionalAIService = InteractiveBlueprintAI
AIService = InteractiveBlueprintAI
EnhancedAIService = InteractiveBlueprintAI
