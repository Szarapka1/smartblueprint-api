# vision_intelligence.py
import asyncio
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from openai import AsyncOpenAI

# Fix: Import from app.models.schemas
from app.models.schemas import VisualIntelligenceResult, ElementGeometry
from app.core.config import CONFIG

# FIXED: Import patterns from dedicated file
from .patterns import VISUAL_PATTERNS, VISION_CONFIG, VISION_PHILOSOPHY, ELEMENT_VARIATIONS

logger = logging.getLogger(__name__)


class VisionIntelligence:
    """
    Advanced Visual Intelligence using GPT-4 Vision
    
    Philosophy: Let GPT-4V SEE and UNDERSTAND naturally, don't constrain it with rigid patterns.
    We guide it to think like a master engineer who can recognize ANY element intelligently.
    
    ENHANCED: Better multi-page analysis support
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        # Lazy initialization
        self._client = None
        self.vision_semaphore = None
        self.inference_semaphore = None
        
        self.visual_patterns = VISUAL_PATTERNS
        self.deterministic_seed = 42
        
        # Enhanced confidence calculation
        self.confidence_weights = {
            "visual_clarity": 0.20,
            "pattern_recognition": 0.20,
            "context_consistency": 0.20,
            "cross_reference": 0.20,
            "professional_judgment": 0.20
        }
        
        # Track analysis context - ENHANCED for multi-page
        self.analysis_context = {
            "drawing_type": None,
            "scale": None,
            "discipline": None,
            "element_focus": None,
            "pages_analyzed": 0,  # NEW
            "total_elements_found": 0  # NEW
        }
    
    @property
    def client(self):
        """Lazy client initialization"""
        if self._client is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required")
            
            self._client = AsyncOpenAI(
                api_key=self.openai_api_key,
                timeout=CONFIG["VISION_REQUEST_TIMEOUT"],
                max_retries=CONFIG["VISION_MAX_RETRIES"]
            )
        return self._client
    
    def _ensure_semaphores_initialized(self):
    """Initialize semaphores for rate limiting"""
    if self.vision_semaphore is None:
        # Use CONFIG value instead of hardcoded 3
        vision_limit = CONFIG.get("VISION_INFERENCE_LIMIT", 10)
        self.vision_semaphore = asyncio.Semaphore(vision_limit)
        logger.info(f"🔧 Vision semaphore initialized with limit: {vision_limit}")
        
    if self.inference_semaphore is None:
        # Use the same limit from CONFIG
        self.inference_semaphore = asyncio.Semaphore(CONFIG.get("VISION_INFERENCE_LIMIT", 10))
    
    async def analyze(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]],
        page_number: int,
        comprehensive_data: Optional[Dict[str, Any]] = None
    ) -> VisualIntelligenceResult:
        """
        Core visual analysis method - True Visual Intelligence Approach
        
        ENHANCED: Better handling of multiple pages
        - page_number can be a single page or represent multiple pages
        - images can contain multiple pages for comprehensive analysis
        
        1. SEE - Let GPT-4V observe and understand the drawings naturally
        2. UNDERSTAND - Comprehend the question in context of what it sees
        3. RECOGNIZE - Use intelligence to identify elements, not rigid patterns
        4. ANALYZE - Apply engineering expertise comprehensively
        5. VERIFY - Cross-reference all available information
        6. DELIVER - Provide accurate, professional results
        """
        
        logger.info("🧠 Starting Visual Intelligence Analysis")
        logger.info(f"📄 Analyzing {len(images)} images for: {prompt}")
        
        # Update context for multi-page
        self.analysis_context["pages_analyzed"] = len(images)
        
        extracted_text = ""
        if comprehensive_data:
            extracted_text = comprehensive_data.get("context", "")
            # Update analysis context
            self._update_analysis_context(comprehensive_data)
        
        try:
            # PHASE 1: Visual Understanding - Let GPT SEE first
            logger.info("👁️ PHASE 1: Visual Understanding")
            visual_context = await self._visual_understanding_phase(
                prompt, images, question_analysis
            )
            
            # PHASE 2: Intelligent Recognition - Understand what to look for
            logger.info("🎯 PHASE 2: Intelligent Element Recognition")
            element_understanding = await self._intelligent_recognition_phase(
                prompt, images, visual_context, question_analysis
            )
            
            # PHASE 3: Comprehensive Analysis - Full engineering analysis
            logger.info("📊 PHASE 3: Comprehensive Engineering Analysis")
            
            # For multi-page analysis, we may need to analyze in batches
            if len(images) > 5 and question_analysis.get("scope") == "comprehensive":
                # Analyze in batches for large documents
                analysis_result = await self._comprehensive_analysis_phase_batched(
                    prompt, images, element_understanding, visual_context,
                    extracted_text, question_analysis
                )
            else:
                # Standard analysis for smaller sets
                analysis_result = await self._comprehensive_analysis_phase(
                    prompt, images, element_understanding, visual_context,
                    extracted_text, question_analysis
                )
            
            # PHASE 4: Verification & Cross-Reference
            logger.info("✅ PHASE 4: Verification & Cross-Reference")
            verified_result = await self._verification_phase(
                analysis_result, images, extracted_text
            )
            
            # Build final result - ENHANCED for multi-page
            visual_result = self._build_visual_result(
                verified_result, page_number, element_understanding
            )
            
            logger.info(f"✅ Analysis complete: {visual_result.count} {visual_result.element_type}(s) found")
            return visual_result
            
        except Exception as e:
            logger.error(f"Visual Intelligence error: {e}", exc_info=True)
            return self._create_error_result(page_number, str(e))
    
    async def _visual_understanding_phase(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 1: Let GPT-4V SEE and UNDERSTAND the drawings naturally
        No constraints - just observation and understanding
        
        ENHANCED: Better handling of multiple pages
        """
        
        # Adjust prompt based on number of images
        if len(images) > 1:
            understanding_prompt = f"""You are a Master Construction Professional with expertise in all disciplines.

I'm showing you {len(images)} pages from construction drawings.

FIRST, simply OBSERVE these construction drawings and tell me:
1. What types of drawings am I looking at? (floor plans, elevations, details, etc.)
2. What discipline(s)? (architectural, structural, MEP, etc.)
3. What's the scale and level of detail?
4. What are the main elements/systems visible?
5. Any text, labels, schedules, or legends visible?
6. Do these pages appear to be related or from the same project?

Just OBSERVE and DESCRIBE what you see. Be specific about drawing characteristics."""
        else:
            understanding_prompt = """You are a Master Construction Professional with expertise in all disciplines.

FIRST, simply OBSERVE these construction drawings and tell me:
1. What type of drawings am I looking at? (floor plan, elevation, detail, etc.)
2. What discipline? (architectural, structural, MEP, etc.)
3. What's the scale and level of detail?
4. What are the main elements/systems visible?
5. Any text, labels, schedules, or legends visible?

Just OBSERVE and DESCRIBE what you see. Be specific about drawing characteristics."""

        try:
            # For many images, sample a subset for understanding
            sample_images = images[:5] if len(images) > 5 else images
            content = self._prepare_vision_content(sample_images, understanding_prompt)
            
            self._ensure_semaphores_initialized()
            async with self.vision_semaphore:
                response = await self._make_vision_request(
                    content,
                    system_prompt="You are an expert at reading construction drawings. Simply observe and describe what you see.",
                    max_tokens=2000
                )
            
            if response:
                return self._parse_understanding_response(response)
            
        except Exception as e:
            logger.error(f"Visual understanding error: {e}")
        
        return {"drawing_type": "unknown", "discipline": "unknown", "is_multi_page": len(images) > 1}
    
    async def _intelligent_recognition_phase(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        visual_context: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 2: Intelligent Recognition - Understand what elements to find
        Use context and intelligence, not rigid pattern matching
        """
        
        pages_context = f"{len(images)} pages of" if len(images) > 1 else ""
        
        recognition_prompt = f"""Based on your observation of these {pages_context} {visual_context.get('drawing_type', 'construction')} drawings:

USER QUESTION: {prompt}

You observed this is a {visual_context.get('discipline', '')} drawing showing {visual_context.get('main_elements', 'various elements')}.

Now, using your INTELLIGENCE:
1. What SPECIFIC element type is the user asking about? (Consider their terminology)
2. What would these elements LOOK LIKE in this type of drawing?
3. Are there industry variations in how these might be shown?
4. What associated information should I look for? (tags, labels, schedules)
5. Should I look across all {len(images)} pages or focus on specific types?

Don't just match keywords - UNDERSTAND what they need based on the drawing context."""

        try:
            # Use sample images for recognition
            sample_images = images[:3] if len(images) > 3 else images
            content = self._prepare_vision_content(sample_images, recognition_prompt)
            
            async with self.vision_semaphore:
                response = await self._make_vision_request(
                    content,
                    system_prompt="You understand construction terminology and can recognize what users are asking about even with non-standard terms.",
                    max_tokens=1500
                )
            
            if response:
                return self._parse_recognition_response(response, question_analysis)
                
        except Exception as e:
            logger.error(f"Recognition phase error: {e}")
        
        return {"element_type": question_analysis.get("element_focus", "element")}
    
    async def _comprehensive_analysis_phase(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        element_understanding: Dict[str, Any],
        visual_context: Dict[str, Any],
        extracted_text: str,
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 3: Comprehensive Engineering Analysis
        Full professional analysis with all context
        
        ENHANCED: Better handling of multiple pages in single request
        """
        
        element_type = element_understanding.get("element_type", "element")
        visual_hints = element_understanding.get("visual_appearance", "")
        
        # Adjust prompt based on scope
        if question_analysis.get("scope") == "comprehensive" or len(images) > 1:
            analysis_prompt = f"""MASTER ENGINEERING ANALYSIS - {element_type}s ACROSS {len(images)} PAGES

CONTEXT:
- Drawing Type: {visual_context.get('drawing_type', 'construction drawing')}
- Discipline: {visual_context.get('discipline', 'not specified')}
- Scale: {visual_context.get('scale', 'not specified')}
- User Question: {prompt}
- Analyzing {len(images)} pages

YOU ARE ANALYZING FOR: {element_type}s
These typically appear as: {visual_hints}

COMPREHENSIVE MULTI-PAGE ANALYSIS APPROACH:
1. VISUAL SCAN - Find ALL {element_type}s across ALL pages
2. PAGE-BY-PAGE COUNT - Track which page each element is on
3. SYSTEMATIC TOTAL - Count every instance across all pages
4. LOCATION MAPPING - Note page number and grid references
5. AVOID DUPLICATES - Don't count the same element twice
6. INFORMATION GATHERING - Read all labels, tags, notes
7. CROSS-REFERENCE - Check schedules, legends, notes
8. VERIFICATION - Double-check your findings

{self._add_extracted_text_context(extracted_text)}

PROVIDE DETAILED FINDINGS:
- TOTAL COUNT across all pages
- COUNT PER PAGE breakdown
- LOCATION of each (page number + grid reference)
- Any TAGS/LABELS/IDENTIFIERS
- VISUAL EVIDENCE supporting your findings
- CROSS-REFERENCES found (schedules, notes, etc.)

Format your response clearly with page-by-page breakdown."""
        else:
            # Single page analysis
            analysis_prompt = f"""MASTER ENGINEERING ANALYSIS - {element_type}s

CONTEXT:
- Drawing Type: {visual_context.get('drawing_type', 'construction drawing')}
- Discipline: {visual_context.get('discipline', 'not specified')}
- Scale: {visual_context.get('scale', 'not specified')}
- User Question: {prompt}

YOU ARE ANALYZING FOR: {element_type}s
These typically appear as: {visual_hints}

COMPREHENSIVE ANALYSIS APPROACH:
1. VISUAL SCAN - Find ALL {element_type}s using your visual intelligence
2. SYSTEMATIC COUNT - Count every instance you can see
3. LOCATION MAPPING - Note grid references or spatial locations
4. INFORMATION GATHERING - Read all labels, tags, notes
5. CROSS-REFERENCE - Check schedules, legends, notes
6. VERIFICATION - Double-check your findings

{self._add_extracted_text_context(extracted_text)}

PROVIDE DETAILED FINDINGS:
- EXACT COUNT of {element_type}s found
- LOCATION of each (with grid references if visible)
- Any TAGS/LABELS/IDENTIFIERS
- VISUAL EVIDENCE supporting your findings
- CROSS-REFERENCES found (schedules, notes, etc.)

Be thorough - construction accuracy depends on this."""

        try:
            content = self._prepare_vision_content(images, analysis_prompt)
            
            async with self.vision_semaphore:
                response = await self._make_vision_request(
                    content,
                    system_prompt=self._get_analysis_system_prompt(element_type),
                    max_tokens=CONFIG["VISION_PRODUCTION_TOKENS"]
                )
            
            if response:
                return self._parse_analysis_response(response, element_type)
                
        except Exception as e:
            logger.error(f"Analysis phase error: {e}")
        
        return {"count": 0, "locations": [], "evidence": [], "page_counts": {}}
    
    async def _comprehensive_analysis_phase_batched(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        element_understanding: Dict[str, Any],
        visual_context: Dict[str, Any],
        extracted_text: str,
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        NEW: Batched analysis for large documents
        Analyzes images in batches and aggregates results
        """
        
        element_type = element_understanding.get("element_type", "element")
        batch_size = 5  # Analyze 5 pages at a time
        
        # Initialize aggregate results
        aggregate_result = {
            "count": 0,
            "locations": [],
            "evidence": [],
            "grid_references": [],
            "verification_notes": [],
            "page_counts": {}
        }
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_pages = [img.get("page", i+j+1) for j, img in enumerate(batch)]
            
            logger.info(f"📊 Analyzing batch: pages {batch_pages}")
            
            # Analyze this batch
            batch_result = await self._comprehensive_analysis_phase(
                prompt, batch, element_understanding, visual_context,
                extracted_text, question_analysis
            )
            
            # Aggregate results
            aggregate_result["count"] += batch_result.get("count", 0)
            aggregate_result["locations"].extend(batch_result.get("locations", []))
            aggregate_result["evidence"].extend(batch_result.get("evidence", []))
            aggregate_result["grid_references"].extend(batch_result.get("grid_references", []))
            aggregate_result["verification_notes"].extend(batch_result.get("verification_notes", []))
            
            # Merge page counts
            for page, count in batch_result.get("page_counts", {}).items():
                aggregate_result["page_counts"][page] = count
        
        logger.info(f"📊 Batch analysis complete: {aggregate_result['count']} total {element_type}s found")
        
        return aggregate_result
    
    async def _verification_phase(
        self,
        analysis_result: Dict[str, Any],
        images: List[Dict[str, Any]],
        extracted_text: str
    ) -> Dict[str, Any]:
        """
        Phase 4: Verification & Cross-Reference
        Verify findings and check for any missed elements
        
        ENHANCED: Better multi-page verification
        """
        
        # Adjust verification based on result size
        page_info = f"across {len(images)} pages" if len(images) > 1 else ""
        
        verification_prompt = f"""VERIFICATION CHECK

My analysis found:
- Count: {analysis_result.get('count', 0)} {analysis_result.get('element_type', 'element')}s {page_info}
- Locations: {len(analysis_result.get('locations', []))} identified

VERIFY:
1. Did I miss any {analysis_result.get('element_type', 'element')}s?
2. Are my locations accurate?
3. Do my findings match any schedules or notes?
4. Is the count reasonable for this type/size of project?
5. Are there any duplicates counted across pages?

{self._add_extracted_text_context(extracted_text, brief=True)}

If you find discrepancies, provide:
- VERIFIED COUNT: [number]
- MISSED ELEMENTS: [list if any]
- CORRECTIONS: [any needed]"""

        try:
            # Only verify if we found something or if count is suspiciously low
            if analysis_result.get('count', 0) > 0 or len(images) > 1:
                # For large sets, verify a sample
                sample_images = images[:3] if len(images) > 3 else images
                content = self._prepare_vision_content(sample_images, verification_prompt)
                
                async with self.vision_semaphore:
                    response = await self._make_vision_request(
                        content,
                        system_prompt="You are verifying the accuracy of a construction analysis. Be thorough.",
                        max_tokens=1500
                    )
                
                if response:
                    verification = self._parse_verification_response(response, analysis_result)
                    # Merge verification results
                    analysis_result.update(verification)
                    
        except Exception as e:
            logger.error(f"Verification phase error: {e}")
        
        return analysis_result
    
    def _prepare_vision_content(
        self,
        images: List[Dict[str, Any]],
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare content for vision API request
        ENHANCED: Better handling of multiple images with page labels
        """
        content = []
        
        # Add images with high detail and page labels
        for i, image in enumerate(images):
            # Add page label if multiple images
            if len(images) > 1:
                page_num = image.get("page", i + 1)
                content.append({
                    "type": "text",
                    "text": f"\n--- PAGE {page_num} ---"
                })
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image["url"],
                    "detail": "high"  # Always use high detail for accuracy
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        return content
    
    async def _make_vision_request(
        self,
        content: List[Dict[str, Any]],
        system_prompt: str,
        max_tokens: int = 3000
    ) -> Optional[str]:
        """Make a vision API request with error handling"""
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=self.deterministic_seed
                ),
                timeout=CONFIG["VISION_TIMEOUT"]
            )
            
            if response and response.choices:
                return response.choices[0].message.content or ""
                
        except asyncio.TimeoutError:
            logger.error("Vision request timeout")
        except Exception as e:
            logger.error(f"Vision request error: {e}")
        
        return None
    
    def _parse_understanding_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the visual understanding response
        ENHANCED: Handle multi-page context
        """
        context = {
            "drawing_type": "construction drawing",
            "discipline": "general",
            "scale": "not specified",
            "main_elements": [],
            "is_multi_page": False,  # NEW
            "page_consistency": "unknown"  # NEW
        }
        
        # Check for multi-page indicators
        if "pages" in response.lower() or "multiple" in response.lower():
            context["is_multi_page"] = True
        
        # Extract drawing type
        if "floor plan" in response.lower():
            context["drawing_type"] = "floor plan"
        elif "elevation" in response.lower():
            context["drawing_type"] = "elevation"
        elif "section" in response.lower():
            context["drawing_type"] = "section"
        elif "detail" in response.lower():
            context["drawing_type"] = "detail"
        elif "site plan" in response.lower():
            context["drawing_type"] = "site plan"
        
        # Extract discipline
        if "architectural" in response.lower() or "arch" in response.lower():
            context["discipline"] = "architectural"
        elif "structural" in response.lower():
            context["discipline"] = "structural"
        elif "electrical" in response.lower():
            context["discipline"] = "electrical"
        elif "mechanical" in response.lower() or "hvac" in response.lower():
            context["discipline"] = "mechanical"
        elif "plumbing" in response.lower():
            context["discipline"] = "plumbing"
        
        # Extract scale
        scale_match = re.search(r'(\d+["\']?\s*=\s*\d+["\']?|1:\d+|1/\d+)', response)
        if scale_match:
            context["scale"] = scale_match.group(1)
        
        # Check page consistency
        if "same project" in response.lower() or "related" in response.lower():
            context["page_consistency"] = "consistent"
        elif "different" in response.lower() or "unrelated" in response.lower():
            context["page_consistency"] = "mixed"
        
        # Store full response for context
        context["full_observation"] = response[:500]
        
        return context
    
    def _parse_recognition_response(
        self,
        response: str,
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse the recognition phase response"""
        
        element_understanding = {
            "element_type": question_analysis.get("element_focus", "element"),
            "visual_appearance": "",
            "search_hints": [],
            "associated_info": [],
            "multi_page_search": False  # NEW
        }
        
        # Check if multi-page search is recommended
        if "all pages" in response.lower() or "across pages" in response.lower():
            element_understanding["multi_page_search"] = True
        
        # Extract element type understanding
        element_match = re.search(
            r'element\s+type.*?:.*?(\w+(?:\s+\w+)*)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        if element_match:
            detected_type = element_match.group(1).lower().strip()
            
            # Map to known element types
            for known_type in self.visual_patterns.keys():
                if known_type in detected_type or detected_type in known_type:
                    element_understanding["element_type"] = known_type
                    break
            
            # Check element variations
            for variation, mapped_type in ELEMENT_VARIATIONS.items():
                if variation in detected_type:
                    element_understanding["element_type"] = mapped_type
                    break
        
        # Extract visual appearance description
        appearance_match = re.search(
            r'look\s+like.*?:.*?([^\n]+(?:\n[^\n]+)*)',
            response,
            re.IGNORECASE
        )
        if appearance_match:
            element_understanding["visual_appearance"] = appearance_match.group(1).strip()
        
        # Extract any specific search hints
        if "tag" in response.lower():
            element_understanding["search_hints"].append("Look for element tags")
        if "schedule" in response.lower():
            element_understanding["search_hints"].append("Check schedules")
        if "legend" in response.lower():
            element_understanding["search_hints"].append("Reference legend")
        
        return element_understanding
    
    def _parse_analysis_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """
        Parse the comprehensive analysis response
        ENHANCED: Handle multi-page results
        """
        
        result = {
            "element_type": element_type,
            "count": 0,
            "locations": [],
            "evidence": [],
            "grid_references": [],
            "verification_notes": [],
            "page_counts": {}  # NEW: Count per page
        }
        
        # Extract total count - try multiple patterns
        count_patterns = [
            r'(?:TOTAL COUNT|EXACT COUNT|COUNT|TOTAL|FOUND).*?(\d+)',
            r'(\d+)\s+' + element_type + r's?\b',
            r'found\s+(\d+)',
            r'identified\s+(\d+)',
            r'total.*?(\d+)'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))
                break
        
        # Extract per-page counts if present in response
        page_count_pattern = r'(?:PAGE|Page)\s*(\d+).*?:.*?(\d+)\s*(?:' + element_type + r's?)?'
        for match in re.finditer(page_count_pattern, response, re.IGNORECASE | re.MULTILINE):
            page_num = int(match.group(1))
            count = int(match.group(2))
            result["page_counts"][page_num] = count
        
        # Extract locations with details
        location_section = re.search(
            r'(?:LOCATION|COUNT PER PAGE|BREAKDOWN).*?:(.*?)(?:TAGS|VISUAL|CROSS|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        if location_section:
            location_text = location_section.group(1)
            
            # Parse individual locations with page numbers
            location_patterns = [
                r'(?:Page\s*)?(\d+)\s*[-–:]\s*(?:Grid\s+)?([A-Z]-?\d+)(?:\s*[-–]\s*)?(.+?)(?=\n|$)',
                r'(\d+)\.\s*(?:Page\s*)?(\d+)\s*[-–:]\s*(?:Grid\s+)?([A-Z]-?\d+)(?:\s*[-–]\s*)?(.+?)(?=\n|$)',
                r'(?:Grid\s+)?([A-Z]-?\d+)(?:\s*[-–:]\s*)?(.+?)(?=\n|$)'
            ]
            
            for pattern in location_patterns:
                for match in re.finditer(pattern, location_text, re.MULTILINE):
                    groups = match.groups()
                    
                    location_info = {
                        "visual_details": groups[-1].strip() if groups else "",
                        "element_tag": None
                    }
                    
                    # Extract grid reference
                    for g in groups[:-1]:
                        if g and re.match(r'[A-Z]-?\d+', g):
                            location_info["grid_ref"] = g
                            result["grid_references"].append(g)
                            break
                    
                    # Extract page number if present
                    for g in groups[:-1]:
                        if g and g.isdigit():
                            location_info["page"] = int(g)
                            break
                    
                    # Extract element tag if present
                    tag_match = re.search(r'\b([A-Z]+\d+)\b', location_info["visual_details"])
                    if tag_match:
                        location_info["element_tag"] = tag_match.group(1)
                    
                    result["locations"].append(location_info)
        
        # Extract evidence
        evidence_section = re.search(
            r'(?:VISUAL EVIDENCE|EVIDENCE|CROSS-REFERENCES?).*?:(.*?)(?:VERIFICATION|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        if evidence_section:
            evidence_text = evidence_section.group(1)
            evidence_items = [
                line.strip() 
                for line in evidence_text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            result["evidence"] = evidence_items[:10]
        
        # Extract verification notes
        if "schedule" in response.lower():
            result["verification_notes"].append("Cross-referenced with schedule")
        if "verified" in response.lower() or "confirmed" in response.lower():
            result["verification_notes"].append("Visual verification completed")
        if "tag" in response.lower():
            result["verification_notes"].append("Element tags identified")
        
        return result
    
    def _parse_verification_response(
        self,
        response: str,
        original_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse verification response and merge with original"""
        
        updates = {}
        
        # Check for verified count
        verified_match = re.search(r'VERIFIED COUNT.*?(\d+)', response, re.IGNORECASE)
        if verified_match:
            new_count = int(verified_match.group(1))
            if new_count != original_result.get('count', 0):
                updates['count'] = new_count
                updates['verification_notes'] = original_result.get('verification_notes', [])
                updates['verification_notes'].append(
                    f"Count adjusted from {original_result.get('count', 0)} to {new_count} after verification"
                )
        
        # Check for missed elements
        if "missed" in response.lower() and "none" not in response.lower():
            missed_match = re.search(r'MISSED.*?:(.*?)(?:CORRECTIONS|$)', response, re.IGNORECASE | re.DOTALL)
            if missed_match:
                updates['verification_notes'] = original_result.get('verification_notes', [])
                updates['verification_notes'].append(f"Additional elements found during verification")
        
        # Check for duplicates
        if "duplicate" in response.lower():
            updates['verification_notes'] = original_result.get('verification_notes', [])
            updates['verification_notes'].append("Duplicates removed during verification")
        
        return updates
    
    def _build_visual_result(
        self,
        analysis_data: Dict[str, Any],
        page_number: int,
        element_understanding: Dict[str, Any]
    ) -> VisualIntelligenceResult:
        """
        Build the final visual intelligence result
        ENHANCED: Include multi-page context from analysis_data
        """
        
        # Calculate confidence score
        confidence = self._calculate_comprehensive_confidence(analysis_data)
        
        # Handle page context from analysis data
        pages_analyzed = self.analysis_context.get("pages_analyzed", 1)
        if pages_analyzed > 1:
            # For multi-page, extract page info from locations
            analyzed_pages = []
            for loc in analysis_data.get("locations", []):
                if isinstance(loc, dict) and "page" in loc:
                    analyzed_pages.append(loc["page"])
            
            if not analyzed_pages and analysis_data.get("page_counts"):
                analyzed_pages = list(analysis_data["page_counts"].keys())
            
            if analyzed_pages:
                page_context = f"Pages {min(analyzed_pages)}-{max(analyzed_pages)}"
            else:
                page_context = f"{pages_analyzed} pages analyzed"
        else:
            analyzed_pages = [page_number]
            page_context = f"Page {page_number}"
        
        # Create result
        result = VisualIntelligenceResult(
            element_type=analysis_data.get("element_type", "element"),
            count=analysis_data.get("count", 0),
            locations=analysis_data.get("locations", []),
            confidence=confidence,
            visual_evidence=analysis_data.get("evidence", []),
            pattern_matches=element_understanding.get("search_hints", []),
            grid_references=analysis_data.get("grid_references", []),
            verification_notes=analysis_data.get("verification_notes", []),
            page_number=page_number
        )
        
        # Add analysis metadata - ENHANCED
        result.analysis_metadata = {
            "visual_understanding": element_understanding.get("visual_appearance", ""),
            "context": self.analysis_context,
            "verification_performed": len(result.verification_notes) > 0,
            "pages_analyzed": pages_analyzed,  # From context
            "page_context": page_context,  # NEW
            "page_counts": analysis_data.get("page_counts", {}),  # NEW
            "analyzed_pages": analyzed_pages  # NEW
        }
        
        return result
    
    def _calculate_comprehensive_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """
        Calculate confidence using multiple factors
        ENHANCED: Consider multi-page factors
        """
        
        scores = {
            "visual_clarity": 0.8,  # Base score
            "pattern_recognition": 0.8,
            "context_consistency": 0.8,
            "cross_reference": 0.5,  # Lower if no cross-references
            "professional_judgment": 0.8
        }
        
        # Adjust scores based on findings
        
        # Visual clarity - do we have clear locations?
        if analysis_data.get("locations"):
            location_quality = len(analysis_data["locations"]) / max(1, analysis_data.get("count", 1))
            scores["visual_clarity"] = min(0.95, 0.7 + location_quality * 0.25)
        
        # Pattern recognition - did we find expected patterns?
        if analysis_data.get("evidence"):
            scores["pattern_recognition"] = 0.9
        
        # Context consistency - does count make sense?
        count = analysis_data.get("count", 0)
        pages_analyzed = len(analysis_data.get("page_counts", {})) or 1
        
        if count == 0:
            scores["context_consistency"] = 0.6  # Zero finds need extra verification
        elif count > 100 * pages_analyzed:  # Very high count per page
            scores["context_consistency"] = 0.7  # Very high counts need verification
        else:
            # Check consistency across pages
            if analysis_data.get("page_counts"):
                counts = list(analysis_data["page_counts"].values())
                if counts:
                    avg_count = sum(counts) / len(counts)
                    variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
                    if variance < avg_count * 2:  # Reasonable variance
                        scores["context_consistency"] = 0.9
        
        # Cross-reference - any supporting evidence?
        if any("schedule" in str(note).lower() for note in analysis_data.get("verification_notes", [])):
            scores["cross_reference"] = 0.95
        elif analysis_data.get("evidence"):
            scores["cross_reference"] = 0.8
        
        # Professional judgment - any verification performed?
        if analysis_data.get("verification_notes"):
            scores["professional_judgment"] = 0.9
        
        # Calculate weighted average
        total_weight = sum(self.confidence_weights.values())
        weighted_sum = sum(
            scores[factor] * self.confidence_weights[factor]
            for factor in scores
        )
        
        confidence = weighted_sum / total_weight
        
        # Apply bounds
        return max(0.3, min(0.99, confidence))
    
    def _update_analysis_context(self, comprehensive_data: Dict[str, Any]):
        """Update analysis context from comprehensive data"""
        
        if comprehensive_data.get("drawing_info"):
            # Extract drawing type and scale from first page info
            first_page_info = next(iter(comprehensive_data["drawing_info"].values()), {})
            self.analysis_context["drawing_type"] = first_page_info.get("type")
            self.analysis_context["scale"] = first_page_info.get("scale")
        
        # Update total elements found (cumulative)
        self.analysis_context["total_elements_found"] = 0
    
    def _add_extracted_text_context(self, extracted_text: str, brief: bool = False) -> str:
        """Add extracted text context to prompts"""
        
        if not extracted_text:
            return ""
        
        if brief:
            text_sample = extracted_text[:500]
            return f"\nEXTRACTED TEXT (for reference):\n{text_sample}..."
        else:
            text_sample = extracted_text[:2000]
            return f"""
EXTRACTED TEXT AND ANNOTATIONS:
{text_sample}

USE THIS TEXT TO:
- Find element tags and labels
- Check schedules and counts
- Verify specifications
- Cross-reference visual findings"""
    
    def _get_analysis_system_prompt(self, element_type: str) -> str:
        """Get comprehensive system prompt for analysis"""
        
        # Get element-specific hints
        element_hints = self.visual_patterns.get(element_type, {})
        visual_guidance = element_hints.get("vision_guidance", "")
        
        return f"""You are a MASTER CONSTRUCTION PROFESSIONAL with decades of experience.

CURRENT FOCUS: {element_type}s

GUIDANCE FOR THIS ELEMENT:
{visual_guidance}

YOUR EXPERTISE INCLUDES:
- Architectural design and construction
- All engineering disciplines (structural, MEP, civil)
- Construction codes and standards
- Drawing conventions across all trades
- Cost estimation and scheduling
- Field construction experience

YOUR APPROACH:
1. Use VISUAL INTELLIGENCE - See what's actually there, not what you expect
2. Be SYSTEMATIC - Check every area of every drawing
3. READ EVERYTHING - Labels, tags, notes, schedules, legends
4. THINK LIKE A BUILDER - What makes sense in real construction?
5. CROSS-REFERENCE - Verify using multiple information sources
6. BE PRECISE - Construction tolerances are tight

For MULTI-PAGE analysis:
- Track elements page by page
- Avoid counting duplicates across pages
- Note which page each element is on
- Provide total count AND per-page breakdown

Remember: Accuracy is critical. Missing elements costs money. False counts waste time.
Be thorough, be accurate, be professional."""
    
    def _create_error_result(self, page_number: int, error_message: str) -> VisualIntelligenceResult:
        """Create an error result"""
        
        return VisualIntelligenceResult(
            element_type="element",
            count=0,
            locations=[],
            confidence=0.0,
            visual_evidence=[f"Analysis error: {error_message}"],
            pattern_matches=[],
            grid_references=[],
            verification_notes=["Analysis failed - manual review required"],
            page_number=page_number
        )
    
    async def detect_element_focus(self, prompt: str) -> str:
        """
        Intelligently detect what element the user is asking about
        Uses GPT-4's understanding, not just keyword matching
        """
        
        logger.info(f"🎯 Detecting element focus for: '{prompt}'")
        
        detection_prompt = f"""Analyze this construction/blueprint question:

"{prompt}"

What PRIMARY construction element is the user asking about?
Consider:
- Common terminology and abbreviations
- Context clues
- Industry jargon
- What would make sense on blueprints

Return ONLY the element type in lowercase.
Examples: door, window, outlet, panel, column, beam, pipe, sprinkler, etc.

If unclear, return "element".

ELEMENT TYPE:"""

        try:
            self._ensure_semaphores_initialized()
            async with self.inference_semaphore:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert at understanding construction terminology and user intent."
                            },
                            {"role": "user", "content": detection_prompt}
                        ],
                        max_tokens=50,
                        temperature=0.0
                    ),
                    timeout=10.0
                )
            
            if response and response.choices:
                detected = response.choices[0].message.content.strip().lower()
                
                # Validate against known types
                if detected in self.visual_patterns:
                    logger.info(f"✅ Detected element type: '{detected}'")
                    return detected
                
                # Check variations
                for variation, mapped in ELEMENT_VARIATIONS.items():
                    if variation in detected:
                        logger.info(f"✅ Detected element type: '{mapped}' (from '{detected}')")
                        return mapped
                
                # Check partial matches
                for known_element in self.visual_patterns.keys():
                    if known_element in detected or detected in known_element:
                        return known_element
                
                # If it's a reasonable but unknown type, return it
                if detected != "element" and detected != "unknown" and len(detected.split()) <= 2:
                    logger.info(f"✅ Detected custom element type: '{detected}'")
                    return detected
                    
        except Exception as e:
            logger.debug(f"Element detection error: {e}")
        
        # Fallback to keyword detection
        return self._fallback_keyword_detection(prompt.lower())
    
    def _fallback_keyword_detection(self, prompt_lower: str) -> str:
        """Fallback keyword-based detection"""
        
        # Direct element type matches
        for element_type in self.visual_patterns.keys():
            if element_type in prompt_lower:
                return element_type
            # Check plural
            if element_type + 's' in prompt_lower:
                return element_type
            # Check without spaces
            if element_type.replace(' ', '') in prompt_lower.replace(' ', ''):
                return element_type
        
        # Check variations and synonyms
        for variation, element_type in ELEMENT_VARIATIONS.items():
            if variation in prompt_lower:
                return element_type
        
        # Check for partial matches
        words = prompt_lower.split()
        for word in words:
            for element_type in self.visual_patterns.keys():
                if word in element_type or element_type in word:
                    return element_type
        
        return "element"
    
    async def detect_precise_geometries(
        self,
        visual_result: VisualIntelligenceResult,
        images: List[Dict[str, Any]]
    ) -> List[ElementGeometry]:
        """
        Detect precise element geometries for accurate highlighting
        Uses GPT-4V's spatial understanding
        
        ENHANCED: Handle multi-page geometry detection
        """
        
        if not visual_result.locations:
            return []
        
        logger.info(f"📐 Detecting geometries for {len(visual_result.locations)} elements")
        
        # Group locations by page for efficient processing
        locations_by_page = defaultdict(list)
        for loc in visual_result.locations:
            page = loc.get("page", 1)
            locations_by_page[page].append(loc)
        
        all_geometries = []
        
        # Process each page's elements
        for page, page_locations in locations_by_page.items():
            # Find the corresponding image
            page_image = next((img for img in images if img.get("page", 1) == page), None)
            if not page_image:
                continue
            
            geometry_prompt = f"""PRECISE GEOMETRY DETECTION for {visual_result.element_type}s on PAGE {page}

I found {len(page_locations)} {visual_result.element_type}(s) on this page at these locations:
{self._format_locations_for_geometry(page_locations)}

For EACH element, provide PRECISE spatial information:

1. BOUNDING BOX: The exact rectangular area containing the element
   - Top-left corner (x, y)
   - Width and height
   
2. ELEMENT SHAPE: The actual shape outline
   - Shape type: rectangle, circle, polygon, line, arc, composite
   - Key points defining the shape
   
3. ORIENTATION: Rotation angle if applicable

4. SPECIAL FEATURES:
   - For doors: swing direction and arc
   - For windows: sill projection
   - For linear elements: start and end points
   
Assume the image dimensions are 1000x1000 pixels for coordinate reference.

FORMAT each element as:
ELEMENT [#] at [grid]:
- Bounding Box: x=[?], y=[?], width=[?], height=[?]
- Shape: [type]
- Points: [(x1,y1), (x2,y2), ...]
- Orientation: [degrees]
- Special: [any special features]"""

            try:
                content = self._prepare_vision_content([page_image], geometry_prompt)
                
                async with self.vision_semaphore:
                    response = await self._make_vision_request(
                        content,
                        system_prompt="You are a precision spatial analysis system. Provide exact coordinates.",
                        max_tokens=3000
                    )
                
                if response:
                    geometries = self._parse_geometry_data(response, visual_result.element_type)
                    # Add page information to each geometry
                    for geom in geometries:
                        if hasattr(geom, 'page'):
                            geom.page = page
                        else:
                            # Store in special_features if page attribute doesn't exist
                            geom.special_features["page"] = page
                    all_geometries.extend(geometries)
                    
            except Exception as e:
                logger.error(f"Geometry detection error for page {page}: {e}")
        
        return all_geometries
    
    def _format_locations_for_geometry(self, locations: List[Dict[str, Any]]) -> str:
        """Format locations for geometry detection prompt"""
        formatted = []
        for i, loc in enumerate(locations[:10], 1):  # Limit to 10
            formatted.append(
                f"{i}. Grid {loc.get('grid_ref', 'Unknown')}: {loc.get('visual_details', 'Element')}"
            )
        if len(locations) > 10:
            formatted.append(f"... and {len(locations) - 10} more")
        return "\n".join(formatted)
    
    def _parse_geometry_data(self, response: str, element_type: str) -> List[ElementGeometry]:
        """Parse geometry detection response into ElementGeometry objects"""
        
        geometries = []
        
        # Split by element sections
        element_sections = re.split(r'ELEMENT\s*\d+', response, flags=re.IGNORECASE)
        
        for section in element_sections[1:]:  # Skip first empty section
            geometry = ElementGeometry(
                element_type=element_type,
                geometry_type="auto_detect",
                center_point={"x": 0, "y": 0},
                boundary_points=[],
                dimensions={},
                orientation=0.0
            )
            
            # Parse bounding box
            bbox_match = re.search(
                r'Bounding Box:.*?x=(\d+).*?y=(\d+).*?width=(\d+).*?height=(\d+)',
                section,
                re.IGNORECASE
            )
            if bbox_match:
                x, y, w, h = map(int, bbox_match.groups())
                geometry.center_point = {"x": x + w/2, "y": y + h/2}
                geometry.dimensions = {"width": w, "height": h}
            
            # Parse shape type
            shape_match = re.search(r'Shape:\s*(\w+)', section, re.IGNORECASE)
            if shape_match:
                shape = shape_match.group(1).lower()
                geometry.geometry_type = shape
            
            # Parse points
            points_match = re.search(r'Points:.*?\[(.*?)\]', section, re.DOTALL)
            if points_match:
                points_str = points_match.group(1)
                for pt_match in re.finditer(r'\((\d+),\s*(\d+)\)', points_str):
                    geometry.boundary_points.append({
                        "x": int(pt_match.group(1)),
                        "y": int(pt_match.group(2))
                    })
            
            # Parse orientation
            orient_match = re.search(r'Orientation:\s*([\d.]+)', section)
            if orient_match:
                geometry.orientation = float(orient_match.group(1))
            
            # Parse special features
            if element_type == "door":
                if "swing" in section.lower():
                    swing_match = re.search(r'swing.*?(left|right|up|down)', section, re.IGNORECASE)
                    if swing_match:
                        geometry.special_features["swing_direction"] = swing_match.group(1)
            
            geometries.append(geometry)
        
        return geometries
    
    def _generate_default_geometries(self, visual_result: VisualIntelligenceResult) -> List[ElementGeometry]:
        """Generate default geometries based on element type"""
        
        geometries = []
        element_type = visual_result.element_type
        
        # Get typical dimensions from patterns
        typical_dims = {
            "door": {"width": 36, "height": 84},
            "window": {"width": 48, "height": 36},
            "outlet": {"radius": 15},
            "panel": {"width": 24, "height": 36},
            "light fixture": {"width": 48, "height": 24},
            "sprinkler": {"radius": 10},
            "column": {"width": 24, "height": 24},
            "parking": {"width": 108, "height": 216}  # 9' x 18'
        }
        
        for i, location in enumerate(visual_result.locations):
            geometry = ElementGeometry(
                element_type=element_type,
                geometry_type=self.visual_patterns.get(element_type, {}).get(
                    "highlight_geometry", "auto_detect"
                ),
                center_point={"x": 100 + (i % 5) * 200, "y": 100 + (i // 5) * 200},
                boundary_points=[],
                dimensions=typical_dims.get(element_type, {"width": 50, "height": 50}),
                orientation=0.0
            )
            
            # Add page info if available
            if hasattr(location, 'get') and location.get("page"):
                if hasattr(geometry, 'page'):
                    geometry.page = location["page"]
                else:
                    geometry.special_features["page"] = location["page"]
            
            # Add element-specific features
            if element_type == "door":
                geometry.special_features["swing_direction"] = "right"
            elif element_type == "window":
                geometry.special_features["has_sill"] = True
            
            geometries.append(geometry)
        
        return geometries