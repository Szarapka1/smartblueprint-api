# app/services/ai_service.py - SMART BLUEPRINT PORTAL
# Acts as an intelligent interface to your blueprint API with contextual understanding

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

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


class ProfessionalAIService:
    """
    Smart portal that understands your blueprint analysis context
    and provides intelligent responses through your API
    """
    
    def __init__(self, settings: AppSettings = None):
        if settings is None:
            settings = get_settings()
            
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
        
        # System context about what this API does
        self.system_context = """You are an intelligent assistant for a blueprint analysis API. 

WHAT THIS SYSTEM DOES:
- Analyzes construction blueprints (architectural, structural, MEP, civil drawings)
- Extracts information from uploaded PDF blueprints
- Answers questions about building plans, specifications, and codes
- Performs calculations (areas, quantities, code compliance)
- Identifies symbols, dimensions, and drawing elements

YOUR ROLE:
- Help users understand their blueprints
- Answer construction and building-related questions
- Guide users on how to use the system effectively
- Provide expert analysis while being approachable

CAPABILITIES YOU CAN LEVERAGE:
- Visual analysis of blueprint pages
- Text extraction from drawings
- Building code knowledge (IBC, NFPA, NEC, ADA)
- Construction methods and materials
- Engineering calculations
- Symbol recognition

INTERACTION STYLE:
- Be conversational but professional
- Explain technical concepts clearly
- Show calculations when relevant
- Reference specific parts of drawings
- Acknowledge limitations honestly"""
        
        logger.info("✅ Smart Blueprint Portal initialized")
    
    def _smart_page_selection(self, query: str, visual_summary: Dict[str, Any], 
                            structured_text: Dict[str, Any], max_pages: int = 3) -> List[int]:
        """
        Simple page selection based on query context and available pages
        """
        total_pages = visual_summary.get('page_count', 1)
        
        if total_pages <= max_pages:
            return list(range(1, total_pages + 1))
        
        # Basic heuristic - can be enhanced based on your needs
        query_lower = query.lower()
        
        # First page often has overview/title block
        if any(term in query_lower for term in ["what is", "project", "overview", "title"]):
            return [1]
        
        # Last pages often have details
        elif "detail" in query_lower:
            return list(range(max(1, total_pages - max_pages + 1), total_pages + 1))
        
        # Default to first few pages
        return list(range(1, min(max_pages + 1, total_pages + 1)))
    
    async def analyze_blueprint(self, document_id: str, query: str, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze blueprint with understanding of the system's purpose
        """
        try:
            # Build conversation with context
            messages = [
                {
                    "role": "system",
                    "content": self.system_context
                },
                {
                    "role": "assistant",
                    "content": f"I'm analyzing the blueprint '{document_id}' for you. Let me help with your question."
                }
            ]
            
            # Determine if visual analysis would help
            query_lower = query.lower()
            needs_visual = any(term in query_lower for term in [
                "show", "where", "locate", "count", "identify", "what is",
                "symbol", "dimension", "area", "detail", "plan", "section"
            ])
            
            # Build user message
            user_content = []
            selected_pages = []
            
            if needs_visual and context.get('pages'):
                # Select relevant pages
                visual_summary = context.get('visual_summary', {})
                structured_text = context.get('structured_text', {})
                selected_pages = self._smart_page_selection(query, visual_summary, structured_text)
                
                # Add query text
                user_content.append({
                    "type": "text",
                    "text": f"{query}\n\nContext: {context.get('text', '')[:1500]}"
                })
                
                # Add page images
                for page_num in selected_pages:
                    page_key = f'page_{page_num}'
                    if page_key in context.get('pages', {}):
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{context['pages'][page_key]}",
                                "detail": "high"
                            }
                        })
            else:
                # Text-based query
                user_content = f"""{query}

Available blueprint information:
{context.get('text', '')[:2000]}

Please provide a helpful response based on the blueprint data."""
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Get response
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "intent": "blueprint_analysis",
                "confidence": 0.95,
                "components_used": ["visual"] if needs_visual else ["text"],
                "pages_analyzed": selected_pages,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "error": str(e),
                "answer": "I had trouble analyzing that. Could you rephrase your question or ensure the blueprint is properly uploaded?"
            }
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """
        Main entry point - processes questions about blueprints
        """
        try:
            # Log the interaction
            logger.info(f"Processing question from {author or 'user'} about {document_id}: {prompt[:100]}...")
            
            # Build context from available data
            context = {}
            
            # Load text context
            try:
                context['text'] = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
            except:
                context['text'] = ""
            
            # Load metadata
            try:
                metadata_json = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_visual_summary.json"
                )
                context['visual_summary'] = json.loads(metadata_json)
            except:
                context['visual_summary'] = {}
            
            # Load structured text
            try:
                structured_json = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_structured_text.json"
                )
                context['structured_text'] = json.loads(structured_json)
            except:
                context['structured_text'] = {}
            
            # Load page images if needed (based on query)
            if any(term in prompt.lower() for term in ["show", "identify", "count", "where"]):
                context['pages'] = {}
                for i in range(1, 4):  # Load up to 3 pages
                    try:
                        page_data = await storage_service.download_blob_as_bytes(
                            container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                            blob_name=f"{document_id}_page_{i}.png"
                        )
                        context['pages'][f'page_{i}'] = base64.b64encode(page_data).decode('utf-8')
                    except:
                        break
            
            # Check if we have any data
            if not context.get('text') and not context.get('pages'):
                return f"""I don't see any blueprint data for '{document_id}'. 

Please make sure:
1. The blueprint PDF has been uploaded
2. The document ID '{document_id}' is correct
3. The processing has completed

You can upload blueprints through the upload endpoint and then ask questions about them."""
            
            # Analyze the blueprint
            result = await self.analyze_blueprint(document_id, prompt, context)
            
            return result.get('answer', 'Unable to process your question.')
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return f"I encountered an error: {str(e)}. Please try again or contact support if the issue persists."
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Export with expected aliases
AIService = ProfessionalAIService
EnhancedAIService = ProfessionalAIService
ExpertBlueprintAI = ProfessionalAIService
