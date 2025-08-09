# vision_intelligence.py
import asyncio
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter

from openai import AsyncOpenAI

# Fix: Import from app.models.schemas
from app.models.schemas import VisualIntelligenceResult, ElementGeometry
from app.core.config import CONFIG

# Import patterns from dedicated file
from .patterns import VISUAL_PATTERNS, VISION_CONFIG, VISION_PHILOSOPHY, ELEMENT_VARIATIONS

logger = logging.getLogger(__name__)


class VisionIntelligence:
    """
    Triple Verification Visual Intelligence using GPT-4 Vision
    
    Philosophy: GPT-4V already knows construction. Just ask it clearly 3 times:
    1. Visual count
    2. Visual + spatial verification
    3. Text review across entire document
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
            self.vision_semaphore = asyncio.Semaphore(3)
        if self.inference_semaphore is None:
            self.inference_semaphore = asyncio.Semaphore(CONFIG["VISION_INFERENCE_LIMIT"])
    
    async def analyze(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]],
        page_number: int,
        comprehensive_data: Optional[Dict[str, Any]] = None
    ) -> VisualIntelligenceResult:
        """
        Core visual analysis method - Triple Verification Approach
        
        Pass 1: Direct visual count
        Pass 2: Visual + spatial verification  
        Pass 3: Text review across entire document
        
        If all 3 agree = near 100% confidence
        """
        
        element_type = question_analysis.get("element_focus", "element")
        logger.info(f"🧠 Starting Triple Verification for {element_type}s")
        logger.info(f"📄 Analyzing {len(images)} images for: {prompt}")
        
        # Extract text context if available
        extracted_text = ""
        if comprehensive_data:
            extracted_text = comprehensive_data.get("context", "")
        
        try:
            # Ensure semaphores are initialized
            self._ensure_semaphores_initialized()
            
            # PASS 1: Direct Visual Count
            logger.info("👁️ PASS 1: Direct Visual Count")
            pass1_result = await self._pass1_visual_count(
                element_type, images, prompt
            )
            
            # PASS 2: Visual + Spatial Verification
            logger.info("📍 PASS 2: Visual + Spatial Verification")
            pass2_result = await self._pass2_spatial_verification(
                element_type, images, prompt
            )
            
            # PASS 3: Text Review
            logger.info("📄 PASS 3: Document Text Review")
            pass3_result = await self._pass3_text_review(
                element_type, images, extracted_text, prompt
            )
            
            # Build consensus result
            final_result = self._build_consensus_result(
                pass1_result, pass2_result, pass3_result,
                element_type, page_number
            )
            
            logger.info(f"✅ Triple verification complete: {final_result.count} {element_type}(s) " +
                       f"with {int(final_result.confidence * 100)}% confidence")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Visual Intelligence error: {e}", exc_info=True)
            return self._create_error_result(element_type, page_number, str(e))
    
    async def _pass1_visual_count(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Pass 1: Simple direct visual count
        Just ask GPT-4V to count the elements
        """
        
        prompt = f"""PASS 1: VISUAL COUNT

Count ALL {element_type}s you can see in these construction drawings.

Just count them. Every single one.
Look carefully at all pages provided.

IMPORTANT: You already know what {element_type}s look like. Trust your vision.

Original question: {original_prompt}

Provide:
COUNT: [exact number]
CONFIDENCE: [HIGH/MEDIUM/LOW]
FOUND_ON_PAGES: [list which pages have {element_type}s]"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are counting {element_type}s in construction drawings. Be thorough and accurate.",
                max_tokens=1000
            )
        
        if response:
            return self._parse_pass1_response(response, element_type)
        
        return {"count": 0, "confidence": "LOW", "pages": []}
    
    async def _pass2_spatial_verification(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Pass 2: Count with spatial verification
        Ask for count WITH grid locations
        """
        
        prompt = f"""PASS 2: SPATIAL VERIFICATION

Count ALL {element_type}s again, but this time tell me WHERE each one is located.

For EACH {element_type}:
- Give its grid reference (like A-1, B-2, etc.)
- Brief description of what you see
- Any labels or tags

Original question: {original_prompt}

FORMAT:
TOTAL COUNT: [number]
LOCATIONS:
1. Grid [?] - [description]
2. Grid [?] - [description]
...

VERIFICATION: Does this count match what you'd expect for this type of drawing?"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are verifying {element_type} locations in construction drawings. Be precise with grid references.",
                max_tokens=3000
            )
        
        if response:
            return self._parse_pass2_response(response, element_type)
        
        return {"count": 0, "locations": [], "grid_references": []}
    
    async def _pass3_text_review(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        extracted_text: str,
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Pass 3: Review ALL text in document
        Not just schedules - ALL text, notes, labels, specs
        """
        
        # Add extracted text context if available
        text_context = ""
        if extracted_text:
            text_context = f"""
EXTRACTED TEXT FROM DOCUMENT:
{extracted_text[:2000]}...

Review this text for any mentions of {element_type}s, quantities, specifications, or related information.
"""
        
        prompt = f"""PASS 3: COMPREHENSIVE TEXT REVIEW

Review ALL text in these drawings for information about {element_type}s:
- Title blocks
- General notes  
- Schedules (if any)
- Specifications
- Labels and tags
- Legend items
- Any text mentioning {element_type}s

{text_context}

Original question: {original_prompt}

PROVIDE:
TEXT-BASED COUNT: [number if found in text]
SOURCES:
- [list where you found quantity information]
KEY FINDINGS:
- [any important text about {element_type}s]
SCHEDULE_COUNT: [number if schedule exists, or "No schedule found"]"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are reviewing all text and documentation about {element_type}s. Look everywhere, not just schedules.",
                max_tokens=2000
            )
        
        if response:
            return self._parse_pass3_response(response, element_type)
        
        return {"text_count": None, "schedule_count": None, "sources": [], "findings": []}
    
    def _build_consensus_result(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        element_type: str,
        page_number: int
    ) -> VisualIntelligenceResult:
        """
        Build consensus from triple verification
        If all 3 agree = high confidence
        """
        
        # Extract counts
        counts = []
        if pass1.get("count") is not None:
            counts.append(pass1["count"])
        if pass2.get("count") is not None:
            counts.append(pass2["count"])
        if pass3.get("text_count") is not None:
            counts.append(pass3["text_count"])
        
        # Determine consensus count
        if counts:
            # Use mode (most common) or average if no mode
            count_freq = Counter(counts)
            mode_count = count_freq.most_common(1)[0][0]
            consensus_count = mode_count
        else:
            consensus_count = 0
        
        # Calculate confidence based on agreement
        confidence = self._calculate_consensus_confidence(
            pass1, pass2, pass3, counts
        )
        
        # Build verification notes
        verification_notes = []
        verification_notes.append(f"Pass 1 (Visual): {pass1.get('count', 0)} {element_type}s")
        verification_notes.append(f"Pass 2 (Spatial): {pass2.get('count', 0)} {element_type}s")
        
        if pass3.get('schedule_count') is not None:
            verification_notes.append(f"Pass 3 (Text): Schedule shows {pass3['schedule_count']} {element_type}s")
        elif pass3.get('text_count') is not None:
            verification_notes.append(f"Pass 3 (Text): Documentation indicates {pass3['text_count']} {element_type}s")
        else:
            verification_notes.append(f"Pass 3 (Text): No quantity found in text")
        
        # Add consensus note
        if len(set(counts)) == 1 and len(counts) >= 2:
            verification_notes.append("✅ PERFECT CONSENSUS achieved")
        elif confidence >= 0.90:
            verification_notes.append("✅ Strong agreement between passes")
        else:
            verification_notes.append("⚠️ Some variation between verification passes")
        
        # Build visual evidence
        visual_evidence = []
        if pass2.get("locations"):
            visual_evidence.append(f"Found in {len(set(pass2.get('grid_references', [])))} grid locations")
        if pass1.get("pages"):
            visual_evidence.append(f"Appears on pages: {', '.join(map(str, pass1['pages']))}")
        visual_evidence.extend(pass3.get("findings", [])[:3])  # Add top 3 text findings
        
        # Create result
        result = VisualIntelligenceResult(
            element_type=element_type,
            count=consensus_count,
            locations=pass2.get("locations", []),
            confidence=confidence,
            visual_evidence=visual_evidence,
            pattern_matches=[],  # Not used in triple verification
            grid_references=pass2.get("grid_references", []),
            verification_notes=verification_notes,
            page_number=page_number
        )
        
        # Add metadata
        result.analysis_metadata = {
            "method": "triple_verification",
            "pass1_count": pass1.get("count", 0),
            "pass2_count": pass2.get("count", 0),
            "pass3_count": pass3.get("text_count"),
            "schedule_count": pass3.get("schedule_count"),
            "consensus_achieved": len(set(counts)) == 1 and len(counts) >= 2
        }
        
        return result
    
    def _calculate_consensus_confidence(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        counts: List[int]
    ) -> float:
        """Calculate confidence based on consensus between passes"""
        
        if not counts:
            return 0.5
        
        # Perfect consensus
        if len(set(counts)) == 1:
            if len(counts) == 3:
                return 0.99  # All 3 agree
            elif len(counts) == 2:
                return 0.95  # 2 agree (3rd had no data)
        
        # Near consensus (within 1)
        if len(counts) >= 2:
            count_range = max(counts) - min(counts)
            if count_range <= 1:
                return 0.90
            elif count_range <= 2:
                return 0.85
        
        # Check pass confidence levels
        base_confidence = 0.70
        
        if pass1.get("confidence") == "HIGH":
            base_confidence += 0.05
        if pass2.get("locations") and len(pass2["locations"]) > 0:
            base_confidence += 0.05
        if pass3.get("schedule_count") is not None:
            base_confidence += 0.10
        
        return min(base_confidence, 0.95)
    
    def _prepare_vision_content(
        self,
        images: List[Dict[str, Any]],
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare content for vision API request"""
        content = []
        
        # Add note about number of images
        if len(images) > 1:
            content.append({
                "type": "text",
                "text": f"You are looking at {len(images)} pages of construction drawings."
            })
        
        # Add images
        for i, image in enumerate(images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image["url"],
                    "detail": "high"
                }
            })
        
        # Add the main prompt
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
    
    def _parse_pass1_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 1 response"""
        result = {
            "count": 0,
            "confidence": "MEDIUM",
            "pages": []
        }
        
        # Extract count
        count_match = re.search(r'COUNT:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            result["count"] = int(count_match.group(1))
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()
        
        # Extract pages
        pages_match = re.search(r'FOUND_ON_PAGES:.*?(\[.*?\]|[\d,\s]+)', response, re.IGNORECASE | re.DOTALL)
        if pages_match:
            pages_text = pages_match.group(1)
            page_numbers = re.findall(r'\d+', pages_text)
            result["pages"] = [int(p) for p in page_numbers]
        
        return result
    
    def _parse_pass2_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 2 response with locations"""
        result = {
            "count": 0,
            "locations": [],
            "grid_references": []
        }
        
        # Extract total count
        count_match = re.search(r'TOTAL COUNT:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            result["count"] = int(count_match.group(1))
        
        # Extract locations
        location_pattern = r'(\d+)\.\s*Grid\s*([A-Z]-?\d+)\s*[-–]\s*(.+?)(?=\n\d+\.|$)'
        
        for match in re.finditer(location_pattern, response, re.MULTILINE | re.DOTALL):
            grid_ref = match.group(2)
            description = match.group(3).strip()
            
            # Extract element tag if present
            tag_match = re.search(r'\b([A-Z]+\d+)\b', description)
            
            location_info = {
                "grid_ref": grid_ref,
                "visual_details": description,
                "element_tag": tag_match.group(1) if tag_match else None
            }
            
            result["locations"].append(location_info)
            result["grid_references"].append(grid_ref)
        
        # If no structured format, try to extract count
        if result["count"] == 0 and element_type in response:
            numbers = re.findall(r'\b(\d+)\s*' + element_type, response, re.IGNORECASE)
            if numbers:
                result["count"] = int(numbers[0])
        
        return result
    
    def _parse_pass3_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 3 text review response"""
        result = {
            "text_count": None,
            "schedule_count": None,
            "sources": [],
            "findings": []
        }
        
        # Extract text-based count
        text_count_match = re.search(r'TEXT-BASED COUNT:\s*(\d+)', response, re.IGNORECASE)
        if text_count_match:
            result["text_count"] = int(text_count_match.group(1))
        
        # Extract schedule count
        schedule_match = re.search(r'SCHEDULE_COUNT:\s*(\d+)', response, re.IGNORECASE)
        if schedule_match:
            result["schedule_count"] = int(schedule_match.group(1))
        elif "no schedule" in response.lower():
            result["schedule_count"] = None
        
        # Extract sources
        sources_section = re.search(r'SOURCES:(.*?)(?:KEY FINDINGS:|SCHEDULE_COUNT:|$)', 
                                   response, re.IGNORECASE | re.DOTALL)
        if sources_section:
            sources_text = sources_section.group(1)
            sources = [s.strip() for s in sources_text.split('\n') if s.strip() and s.strip().startswith('-')]
            result["sources"] = [s.lstrip('- ') for s in sources]
        
        # Extract key findings
        findings_section = re.search(r'KEY FINDINGS:(.*?)(?:SCHEDULE_COUNT:|$)', 
                                    response, re.IGNORECASE | re.DOTALL)
        if findings_section:
            findings_text = findings_section.group(1)
            findings = [f.strip() for f in findings_text.split('\n') if f.strip() and f.strip().startswith('-')]
            result["findings"] = [f.lstrip('- ') for f in findings]
        
        return result
    
    def _create_error_result(self, element_type: str, page_number: int, error_message: str) -> VisualIntelligenceResult:
        """Create an error result"""
        return VisualIntelligenceResult(
            element_type=element_type,
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
        (Kept for compatibility with question_analyzer.py)
        """
        
        logger.info(f"🎯 Detecting element focus for: '{prompt}'")
        
        detection_prompt = f"""What construction element is this question about?

Question: "{prompt}"

Consider common terminology, abbreviations, and context.
Return ONLY the element type in lowercase (e.g., door, outlet, window, panel, etc.)

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
                                "content": "You understand construction terminology."
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
                
                # Validate and map to known types
                if detected in self.visual_patterns:
                    return detected
                
                # Check variations
                for variation, mapped in ELEMENT_VARIATIONS.items():
                    if variation in detected or detected in variation:
                        return mapped
                
                # Return if reasonable
                if detected and len(detected.split()) <= 3:
                    return detected
                    
        except Exception as e:
            logger.debug(f"Element detection error: {e}")
        
        return "element"
    
    async def detect_precise_geometries(
        self,
        visual_result: VisualIntelligenceResult,
        images: List[Dict[str, Any]]
    ) -> List[ElementGeometry]:
        """
        Detect precise element geometries for highlighting
        (Kept for compatibility with semantic_highlighter.py)
        """
        
        if not visual_result.locations:
            return []
        
        # For now, generate basic geometries
        # This could be enhanced to use GPT-4V for precise detection
        geometries = []
        
        for i, location in enumerate(visual_result.locations):
            geometry = ElementGeometry(
                element_type=visual_result.element_type,
                geometry_type="auto_detect",
                center_point={"x": 100 + (i % 5) * 200, "y": 100 + (i // 5) * 200},
                boundary_points=[],
                dimensions=self._get_typical_dimensions(visual_result.element_type),
                orientation=0.0
            )
            geometries.append(geometry)
        
        return geometries
    
    def _get_typical_dimensions(self, element_type: str) -> Dict[str, float]:
        """Get typical dimensions for element type"""
        dimensions = {
            "door": {"width": 36, "height": 84},
            "window": {"width": 48, "height": 36},
            "outlet": {"radius": 15},
            "panel": {"width": 24, "height": 36},
            "light fixture": {"width": 48, "height": 24},
            "sprinkler": {"radius": 10},
            "column": {"width": 24, "height": 24}
        }
        return dimensions.get(element_type, {"width": 50, "height": 50})
