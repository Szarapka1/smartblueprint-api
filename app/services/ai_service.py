# app/services/ai_service.py - INTELLIGENT BLUEPRINT UNDERSTANDING WITH CODE KNOWLEDGE

import asyncio
import base64
import json
import logging
import re
import math
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from openai import OpenAI
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class BlueprintKnowledge:
    """Comprehensive blueprint and building code knowledge base"""
    
    # Building codes and standards
    BUILDING_CODES = {
        "egress": {
            "corridor_width_min": 44,  # inches
            "door_width_min": 32,      # inches
            "stair_width_min": 44,     # inches
            "occupant_load_factors": {
                "office": 100,         # sq ft per person
                "assembly": 15,
                "retail": 30,
                "residential": 200,
                "storage": 300
            },
            "travel_distance": {
                "sprinklered": {"office": 300, "assembly": 250, "residential": 250},
                "unsprinklered": {"office": 200, "assembly": 150, "residential": 200}
            }
        },
        "ada": {
            "door_clear_width": 32,
            "corridor_width": 48,
            "turning_radius": 60,
            "ramp_max_slope": 8.33,  # percent (1:12)
            "parking_ratio": {1: 1, 25: 1, 50: 2, 75: 3, 100: 4, 150: 5, 200: 6}
        },
        "fire": {
            "sprinkler_spacing": {
                "light_hazard": 225,    # sq ft max per head
                "ordinary_hazard": 130,
                "extra_hazard": 90
            },
            "fire_ratings": {
                "exit_enclosure": 2,    # hours
                "exit_passageway": 2,
                "corridor": 1,
                "tenant_separation": 1
            }
        },
        "structural": {
            "live_loads_psf": {
                "office": 50,
                "corridor": 100,
                "retail": 100,
                "residential": 40,
                "parking": 40,
                "roof": 20
            },
            "deflection_limits": {
                "floor_live": 360,  # L/360
                "floor_total": 240,
                "roof_live": 240,
                "roof_total": 180
            }
        }
    }
    
    # Standard dimensions and measurements
    STANDARD_DIMENSIONS = {
        "door_sizes": {
            "single": {"width": [30, 32, 36], "height": [80, 84, 96]},
            "double": {"width": [60, 64, 72], "height": [80, 84, 96]},
            "overhead": {"width": [96, 108, 120, 144], "height": [84, 96, 108, 120]}
        },
        "ceiling_heights": {
            "residential": [96, 108, 120],  # 8', 9', 10'
            "commercial": [108, 120, 144],  # 9', 10', 12'
            "retail": [144, 168, 192]       # 12', 14', 16'
        },
        "parking_spaces": {
            "standard": {"width": 108, "length": 216},     # 9' x 18'
            "compact": {"width": 96, "length": 192},       # 8' x 16'
            "accessible": {"width": 144, "length": 216},   # 12' x 18' (includes aisle)
            "van_accessible": {"width": 192, "length": 216} # 16' x 18'
        }
    }
    
    # Drawing scale conversions
    SCALE_CONVERSIONS = {
        "1/16\" = 1'-0\"": 192,
        "1/8\" = 1'-0\"": 96,
        "3/16\" = 1'-0\"": 64,
        "1/4\" = 1'-0\"": 48,
        "3/8\" = 1'-0\"": 32,
        "1/2\" = 1'-0\"": 24,
        "3/4\" = 1'-0\"": 16,
        "1\" = 1'-0\"": 12,
        "1 1/2\" = 1'-0\"": 8,
        "3\" = 1'-0\"": 4
    }
    
    # Symbol recognition patterns
    BLUEPRINT_SYMBOLS = {
        "doors": {
            "single_swing": "arc with line",
            "double_swing": "two arcs",
            "sliding": "rectangle with arrow",
            "overhead": "dashed rectangle"
        },
        "windows": {
            "fixed": "double line",
            "casement": "line with arc",
            "sliding": "overlapping rectangles"
        },
        "fixtures": {
            "wc": "elongated oval",
            "lavatory": "rectangle with bowl",
            "shower": "square with X",
            "kitchen_sink": "rectangle with drainboard"
        }
    }


class IntentAnalyzer:
    """Analyzes user intent and determines required analysis approach"""
    
    INTENT_PATTERNS = {
        "code_compliance": {
            "keywords": ["code", "compliant", "comply", "legal", "requirement", "allowed", "minimum", "maximum"],
            "patterns": [r"does.*meet.*code", r"is.*compliant", r"check.*compliance"],
            "requires": ["visual", "calculation", "code_knowledge"]
        },
        "quantity_takeoff": {
            "keywords": ["count", "how many", "number of", "total", "quantity"],
            "patterns": [r"count.*(?:door|window|room|fixture)", r"how many"],
            "requires": ["visual", "calculation"]
        },
        "measurement": {
            "keywords": ["dimension", "size", "width", "length", "height", "area", "square"],
            "patterns": [r"what.*(?:size|dimension)", r"how.*(?:wide|long|tall|big)"],
            "requires": ["visual", "scale_calculation"]
        },
        "identification": {
            "keywords": ["what is", "identify", "type of", "which", "locate"],
            "patterns": [r"what.*type", r"identify.*(?:system|material)"],
            "requires": ["visual", "text"]
        },
        "calculation": {
            "keywords": ["calculate", "determine", "capacity", "load", "sizing"],
            "patterns": [r"calculate.*(?:load|capacity)", r"size.*(?:beam|duct|pipe)"],
            "requires": ["visual", "calculation", "engineering"]
        },
        "location": {
            "keywords": ["where", "location", "find", "locate", "position"],
            "patterns": [r"where.*(?:is|are)", r"locate.*(?:room|equipment)"],
            "requires": ["visual", "grid_reference"]
        }
    }
    
    @classmethod
    def analyze_intent(cls, prompt: str) -> Dict[str, Any]:
        """Deeply analyze user intent"""
        prompt_lower = prompt.lower()
        
        intents = []
        required_capabilities = set()
        
        for intent_type, config in cls.INTENT_PATTERNS.items():
            # Check keywords
            if any(keyword in prompt_lower for keyword in config["keywords"]):
                intents.append(intent_type)
                required_capabilities.update(config["requires"])
                continue
                
            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, prompt_lower):
                    intents.append(intent_type)
                    required_capabilities.update(config["requires"])
                    break
        
        # Default if no specific intent found
        if not intents:
            intents = ["general_inquiry"]
            required_capabilities = {"text", "visual"}
        
        return {
            "primary_intent": intents[0] if intents else "general_inquiry",
            "all_intents": intents,
            "required_capabilities": list(required_capabilities),
            "complexity": len(required_capabilities)
        }


class ProfessionalAIService:
    """AI service with deep blueprint understanding and code knowledge"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.openai_api_key, timeout=30.0)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.max_pages = getattr(settings, 'PDF_MAX_PAGES', 100)
        self.knowledge = BlueprintKnowledge()
        self.intent_analyzer = IntentAnalyzer()
        
        logger.info("✅ Professional AI Service with Blueprint Intelligence initialized")
        
    async def get_ai_response(self, prompt: str, document_id: str, 
                             storage_service: StorageService, author: str = None) -> str:
        """Process queries with deep understanding of intent and blueprint knowledge"""
        try:
            # Step 1: Analyze intent
            intent_analysis = self.intent_analyzer.analyze_intent(prompt)
            logger.info(f"🧠 Intent: {intent_analysis['primary_intent']}")
            logger.info(f"📋 Required: {intent_analysis['required_capabilities']}")
            
            # Step 2: Load document context
            context = await self._load_comprehensive_context(document_id, storage_service)
            
            # Step 3: Route to appropriate handler based on intent
            if intent_analysis['primary_intent'] == "code_compliance":
                return await self._handle_code_compliance(prompt, document_id, storage_service, context)
                
            elif intent_analysis['primary_intent'] == "quantity_takeoff":
                return await self._handle_quantity_takeoff(prompt, document_id, storage_service, context)
                
            elif intent_analysis['primary_intent'] == "measurement":
                return await self._handle_measurement(prompt, document_id, storage_service, context)
                
            elif intent_analysis['primary_intent'] == "calculation":
                return await self._handle_engineering_calculation(prompt, document_id, storage_service, context)
                
            else:
                # Use intelligent routing for other intents
                return await self._handle_intelligent_query(
                    prompt, document_id, storage_service, context, intent_analysis
                )
                
        except Exception as e:
            logger.error(f"❌ AI response error: {e}")
            return f"I encountered an error analyzing the blueprint. Please try rephrasing your question."
    
    async def _handle_code_compliance(self, prompt: str, document_id: str,
                                     storage_service: StorageService, 
                                     context: Dict[str, Any]) -> str:
        """Handle code compliance checks with deep knowledge"""
        
        # Determine which code area
        code_area = self._identify_code_area(prompt)
        
        # Load relevant drawings
        if "egress" in code_area or "ada" in code_area:
            pages = await self._get_floor_plan_pages(context)
        elif "fire" in code_area:
            pages = await self._get_fire_protection_pages(context)
        else:
            pages = await self._get_relevant_pages_by_type(prompt, context)
        
        # Load images
        page_images = await self._load_page_images(document_id, pages[:5], storage_service)
        
        # Prepare code-aware prompt
        system_prompt = f"""You are a code compliance expert analyzing blueprints. You have deep knowledge of:

BUILDING CODES:
{json.dumps(self.knowledge.BUILDING_CODES, indent=2)}

STANDARD DIMENSIONS:
{json.dumps(self.knowledge.STANDARD_DIMENSIONS, indent=2)}

Analyze the drawings for code compliance. Be specific about:
1. What code requirements apply
2. What you observe in the drawings
3. Whether it meets code (with specific measurements)
4. Any violations or concerns

Always cite specific code requirements and measured dimensions."""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        user_content = [
            {"type": "text", "text": f"Code compliance check: {prompt}"}
        ]
        
        for page_num, image_url in page_images:
            user_content.extend([
                {"type": "text", "text": f"\nPage {page_num}:"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
            ])
        
        messages.append({"role": "user", "content": user_content})
        
        response = await self._call_openai_vision(messages)
        return response
    
    async def _handle_quantity_takeoff(self, prompt: str, document_id: str,
                                      storage_service: StorageService,
                                      context: Dict[str, Any]) -> str:
        """Handle quantity takeoff with visual counting"""
        
        # Identify what to count
        count_target = self._identify_count_target(prompt)
        
        # Get appropriate pages
        if count_target in ["doors", "windows", "rooms"]:
            pages = await self._get_floor_plan_pages(context)
        elif count_target in ["fixtures", "equipment"]:
            pages = await self._get_mep_pages(context)
        else:
            pages = await self._get_all_relevant_pages(context)
        
        # For counting, we might need multiple batches
        all_counts = {}
        
        for i in range(0, len(pages), 5):
            batch_pages = pages[i:i+5]
            batch_images = await self._load_page_images(document_id, batch_pages, storage_service)
            
            if batch_images:
                count_prompt = f"""Count all {count_target} visible in these drawings.

For each item found:
1. Give the count per page
2. Note the location (grid reference if visible)
3. Note any identifying marks or labels

Use the standard symbols:
{json.dumps(self.knowledge.BLUEPRINT_SYMBOLS, indent=2)}

Total everything at the end."""
                
                batch_result = await self._analyze_with_vision(
                    count_prompt, batch_images, document_id
                )
                
                # Parse counts from result
                self._parse_counts_from_result(batch_result, all_counts)
        
        # Compile final count
        return self._compile_quantity_takeoff(count_target, all_counts, prompt)
    
    async def _handle_measurement(self, prompt: str, document_id: str,
                                 storage_service: StorageService,
                                 context: Dict[str, Any]) -> str:
        """Handle measurement queries with scale calculations"""
        
        # Identify what to measure
        measurement_target = self._identify_measurement_target(prompt)
        
        # Get scale from context
        scale = await self._determine_drawing_scale(document_id, storage_service, context)
        
        # Get relevant pages
        pages = await self._get_pages_with_dimensions(context, measurement_target)
        page_images = await self._load_page_images(document_id, pages[:3], storage_service)
        
        scale_prompt = f"""Measure {measurement_target} using the drawing scale.

DRAWING SCALE: {scale}
SCALE FACTOR: {self.knowledge.SCALE_CONVERSIONS.get(scale, 'unknown')}

Steps:
1. Identify what needs to be measured
2. Find dimension strings or measure using scale
3. Convert to real-world measurements
4. Double-check using any written dimensions

Be precise with measurements and show your calculation."""
        
        return await self._analyze_with_vision(scale_prompt, page_images, document_id)
    
    async def _handle_engineering_calculation(self, prompt: str, document_id: str,
                                            storage_service: StorageService,
                                            context: Dict[str, Any]) -> str:
        """Handle engineering calculations with code knowledge"""
        
        calc_type = self._identify_calculation_type(prompt)
        
        if calc_type == "structural":
            return await self._calculate_structural(prompt, document_id, storage_service, context)
        elif calc_type == "mep":
            return await self._calculate_mep(prompt, document_id, storage_service, context)
        elif calc_type == "occupancy":
            return await self._calculate_occupancy(prompt, document_id, storage_service, context)
        else:
            return await self._general_calculation(prompt, document_id, storage_service, context)
    
    async def _calculate_occupancy(self, prompt: str, document_id: str,
                                  storage_service: StorageService,
                                  context: Dict[str, Any]) -> str:
        """Calculate occupancy loads using code factors"""
        
        # Get floor plans
        pages = await self._get_floor_plan_pages(context)
        page_images = await self._load_page_images(document_id, pages[:5], storage_service)
        
        calc_prompt = f"""Calculate occupancy loads for this building.

OCCUPANT LOAD FACTORS (sq ft per person):
{json.dumps(self.knowledge.BUILDING_CODES['egress']['occupant_load_factors'], indent=2)}

Steps:
1. Identify room types and uses
2. Measure or read area for each space
3. Apply correct occupant load factor
4. Sum total occupancy per floor
5. Check egress capacity

Show all calculations."""
        
        return await self._analyze_with_vision(calc_prompt, page_images, document_id)
    
    async def _handle_intelligent_query(self, prompt: str, document_id: str,
                                       storage_service: StorageService,
                                       context: Dict[str, Any],
                                       intent_analysis: Dict[str, Any]) -> str:
        """Handle queries with intelligent routing based on intent"""
        
        # Build comprehensive system prompt with all knowledge
        system_content = self._build_intelligent_system_prompt(intent_analysis)
        
        # Determine optimal page selection
        if "visual" in intent_analysis["required_capabilities"]:
            pages = await self._smart_page_selection(prompt, context, intent_analysis)
            page_images = await self._load_page_images(document_id, pages, storage_service)
            
            # Use vision with blueprint knowledge
            messages = [
                {"role": "system", "content": system_content}
            ]
            
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "text", "text": f"\nContext: {context.get('text', '')[:1000]}"}
            ]
            
            for page_num, image_url in page_images:
                user_content.extend([
                    {"type": "text", "text": f"\nPage {page_num}:"},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                ])
            
            messages.append({"role": "user", "content": user_content})
            
            return await self._call_openai_vision(messages)
        else:
            # Text-based with blueprint knowledge
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{prompt}\n\nDocument context:\n{context.get('text', '')[:3000]}"}
            ]
            
            return await self._call_openai_text(messages)
    
    def _build_intelligent_system_prompt(self, intent_analysis: Dict[str, Any]) -> str:
        """Build a comprehensive system prompt with all relevant knowledge"""
        
        base_prompt = """You are a master blueprint analyst with deep knowledge of building codes, 
construction standards, and engineering principles. You understand drawings as both visual 
documents and technical specifications."""
        
        # Add specific knowledge based on intent
        if "code_knowledge" in intent_analysis["required_capabilities"]:
            base_prompt += f"\n\nBUILDING CODES:\n{json.dumps(self.knowledge.BUILDING_CODES, indent=2)}"
        
        if "calculation" in intent_analysis["required_capabilities"]:
            base_prompt += f"\n\nSTANDARD DIMENSIONS:\n{json.dumps(self.knowledge.STANDARD_DIMENSIONS, indent=2)}"
        
        if "scale_calculation" in intent_analysis["required_capabilities"]:
            base_prompt += f"\n\nSCALE CONVERSIONS:\n{json.dumps(self.knowledge.SCALE_CONVERSIONS, indent=2)}"
        
        base_prompt += """

When analyzing:
1. Apply your professional knowledge
2. Reference specific codes when relevant
3. Perform calculations when needed
4. Be precise with measurements and counts
5. Explain your reasoning"""
        
        return base_prompt
    
    # Helper methods for specific analyses
    async def _get_floor_plan_pages(self, context: Dict[str, Any]) -> List[int]:
        """Get pages containing floor plans"""
        pages = []
        for page_info in context.get('metadata', {}).get('pages', []):
            if page_info.get('drawing_type') == 'floor_plan':
                pages.append(page_info['page_number'])
        
        # If no floor plans identified, check first few pages
        if not pages and context.get('page_count', 0) > 0:
            pages = list(range(1, min(6, context['page_count'] + 1)))
        
        return pages
    
    async def _get_fire_protection_pages(self, context: Dict[str, Any]) -> List[int]:
        """Get fire protection related pages"""
        pages = []
        keywords = ['fire', 'sprinkler', 'alarm', 'life safety']
        
        for page_info in context.get('metadata', {}).get('pages', []):
            if any(kw in str(page_info).lower() for kw in keywords):
                pages.append(page_info['page_number'])
        
        return pages if pages else [1]
    
    def _identify_code_area(self, prompt: str) -> List[str]:
        """Identify which code areas are relevant"""
        prompt_lower = prompt.lower()
        areas = []
        
        if any(word in prompt_lower for word in ['egress', 'exit', 'stair', 'corridor']):
            areas.append('egress')
        if any(word in prompt_lower for word in ['ada', 'accessible', 'disability']):
            areas.append('ada')
        if any(word in prompt_lower for word in ['fire', 'sprinkler', 'rating']):
            areas.append('fire')
        if any(word in prompt_lower for word in ['structural', 'load', 'beam', 'column']):
            areas.append('structural')
        
        return areas if areas else ['general']
    
    def _identify_count_target(self, prompt: str) -> str:
        """Identify what needs to be counted"""
        prompt_lower = prompt.lower()
        
        targets = ['doors', 'windows', 'rooms', 'fixtures', 'sprinklers', 
                   'columns', 'beams', 'parking spaces', 'equipment']
        
        for target in targets:
            if target in prompt_lower:
                return target
        
        # Try singular forms
        for target in targets:
            if target.rstrip('s') in prompt_lower:
                return target
        
        return "elements"
    
    async def _load_page_images(self, document_id: str, page_numbers: List[int],
                               storage_service: StorageService) -> List[Tuple[int, str]]:
        """Load page images efficiently"""
        if not page_numbers:
            return []
        
        async def load_single_page(page_num: int):
            try:
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_page_{page_num}.png"
                )
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                return (page_num, f"data:image/png;base64,{image_b64}")
            except:
                return None
        
        tasks = [load_single_page(pn) for pn in page_numbers]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]
    
    async def _analyze_with_vision(self, prompt: str, page_images: List[Tuple[int, str]], 
                                  document_id: str) -> str:
        """Analyze with GPT-4 Vision"""
        messages = [
            {
                "role": "system",
                "content": "You are analyzing technical blueprints. Be precise and thorough."
            }
        ]
        
        user_content = [{"type": "text", "text": prompt}]
        
        for page_num, image_url in page_images:
            user_content.extend([
                {"type": "text", "text": f"\nPage {page_num}:"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
            ])
        
        messages.append({"role": "user", "content": user_content})
        
        return await self._call_openai_vision(messages)
    
    async def _call_openai_vision(self, messages: List[Dict]) -> str:
        """Call OpenAI with vision capabilities"""
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4000,
                temperature=0.1
            )
        )
        return response.choices[0].message.content
    
    async def _call_openai_text(self, messages: List[Dict]) -> str:
        """Call OpenAI for text analysis"""
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
        )
        return response.choices[0].message.content
    
    async def _load_comprehensive_context(self, document_id: str, 
                                         storage_service: StorageService) -> Dict[str, Any]:
        """Load all available context"""
        context = {
            "text": "",
            "metadata": {},
            "page_count": 0,
            "has_visual": False,
            "structured_text": {}
        }
        
        # Load text context
        try:
            text_content = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_context.txt"
            )
            context["text"] = text_content
        except:
            pass
        
        # Load visual summary
        try:
            summary_json = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_visual_summary.json"
            )
            context["metadata"] = json.loads(summary_json)
            context["page_count"] = context["metadata"].get("page_count", 0)
            context["has_visual"] = True
        except:
            pass
        
        # Load structured text if available
        try:
            structured_json = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=f"{document_id}_structured_text.json"
            )
            context["structured_text"] = json.loads(structured_json)
        except:
            pass
        
        return context
    
    def get_professional_capabilities(self) -> Dict[str, List[str]]:
        """Return enhanced AI capabilities"""
        return {
            "intent_understanding": [
                "Code compliance checking",
                "Quantity takeoff and counting",
                "Measurement and scaling",
                "Engineering calculations",
                "Location identification",
                "Document navigation"
            ],
            "code_knowledge": [
                "IBC (International Building Code)",
                "ADA compliance requirements",
                "NFPA fire protection standards",
                "Structural load requirements",
                "MEP sizing standards"
            ],
            "calculation_abilities": [
                "Occupancy load calculations",
                "Egress capacity analysis",
                "Structural load analysis",
                "Fire protection coverage",
                "ADA compliance verification"
            ],
            "visual_capabilities": [
                "Symbol recognition with meaning",
                "Scale-accurate measurements",
                "Grid-based location reference",
                "Cross-sheet coordination"
            ]
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Maintain compatibility
AIService = ProfessionalAIService
