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
    Penta Verification Visual Intelligence using GPT-4 Vision
    
    Philosophy: GPT-4V already knows construction. Ask it 5 different ways:
    0. Pure GPT-4V - User prompt only (baseline)
    1. Visual count with guidance
    2. Visual + spatial verification
    3. Text review across entire document
    4. Cross-reference & deduplication check
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
        Core visual analysis method - Penta Verification Approach
        
        Pass 0: Pure GPT-4V with user prompt only (baseline)
        Pass 1: Direct visual count with guidance
        Pass 2: Visual + spatial verification  
        Pass 3: Text review across entire document
        Pass 4: Cross-reference & deduplication
        
        If all 5 agree = near 100% confidence
        """
        
        element_type = question_analysis.get("element_focus", "element")
        logger.info(f"🧠 Starting Penta Verification for {element_type}s")
        logger.info(f"📄 Analyzing {len(images)} images for: {prompt}")
        
        # Extract text context if available
        extracted_text = ""
        if comprehensive_data:
            extracted_text = comprehensive_data.get("context", "")
        
        try:
            # Ensure semaphores are initialized
            self._ensure_semaphores_initialized()
            
            # PASS 0: Pure GPT-4V - User prompt only
            logger.info("🎯 PASS 0: Pure GPT-4V Analysis (User Prompt Only)")
            pass0_result = await self._pass0_pure_vision(
                element_type, images, prompt
            )
            
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
            
            # PASS 4: Cross-Reference & Deduplication
            logger.info("🔄 PASS 4: Cross-Reference & Deduplication")
            pass4_result = await self._pass4_cross_reference(
                element_type, images, prompt, pass1_result, pass2_result, pass3_result
            )
            
            # Build consensus result
            final_result = self._build_consensus_result(
                pass0_result, pass1_result, pass2_result, pass3_result, pass4_result,
                element_type, page_number
            )
            
            logger.info(f"✅ Penta verification complete: {final_result.count} {element_type}(s) " +
                       f"with {int(final_result.confidence * 100)}% confidence")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Visual Intelligence error: {e}", exc_info=True)
            return self._create_error_result(element_type, page_number, str(e))
    
    async def _pass0_pure_vision(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Pass 0: Pure GPT-4V with ONLY the user's original prompt
        No guidance, no structure - just let GPT-4V use its inherent understanding
        """
        
        # Use ONLY the original user prompt - nothing else
        prompt = original_prompt

        content = self._prepare_vision_content_minimal(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt="You are looking at construction drawings.",  # Minimal system prompt
                max_tokens=1000
            )
        
        if response:
            return self._parse_pass0_response(response, element_type)
        
        return {"count": 0, "raw_response": "", "confidence": "LOW"}
    
    def _prepare_vision_content_minimal(
        self,
        images: List[Dict[str, Any]],
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare minimal content for pure vision request"""
        content = []
        
        # Add images
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image["url"],
                    "detail": "high"
                }
            })
        
        # Add the prompt
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        return content
    
    def _parse_pass0_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 0 pure response - extract whatever count GPT naturally provides"""
        result = {
            "count": 0,
            "raw_response": response,
            "confidence": "MEDIUM",
            "natural_count": None
        }
        
        # Try to extract any number that appears to be a count
        # Look for patterns like "33 windows", "I count 33", "There are 33", etc.
        count_patterns = [
            r'(\d+)\s+' + element_type + r's?\b',
            r'count(?:ed)?\s+(\d+)\b',
            r'(?:there\s+are|I\s+see|found|total(?:s)?)\s+(\d+)\b',
            r'(\d+)\s+total\b',
            r'exactly\s+(\d+)\b',
            r'(\d+)\s+(?:unique|different|individual)\s+' + element_type
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))
                result["natural_count"] = int(match.group(1))
                break
        
        # If we found a count, confidence is higher
        if result["count"] > 0:
            # Check if response seems confident
            if any(word in response.lower() for word in ['exactly', 'definitely', 'clearly', 'precisely']):
                result["confidence"] = "HIGH"
            elif any(word in response.lower() for word in ['approximately', 'about', 'around', 'roughly']):
                result["confidence"] = "MEDIUM"
            else:
                result["confidence"] = "MEDIUM"
        
        return result
    
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
        
        prompt = f"""PASS 1: VISUAL COUNT - FIND EVERY SINGLE {element_type.upper()}

You are examining {len(images)} page(s) of construction drawings.

YOUR TASK: Count EVERY SINGLE {element_type} visible in these drawings. Do not miss any!

CRITICAL INSTRUCTIONS:
1. Examine EVERY part of EVERY page thoroughly
2. Look in ALL drawing areas:
   - Main floor plans
   - Enlarged plans or details
   - Elevation views
   - Section views
   - Detail bubbles
   - Typical details that may show multiple instances
   - Partial plans
   - Any other drawing views

3. Count ALL {element_type}s including:
   - Those shown with solid lines (new/proposed)
   - Those shown with dashed lines (existing)
   - Those in detailed areas
   - Those partially visible at drawing edges
   - Those in different sizes or types
   - Those with tags/labels (like {element_type[0].upper()}1, {element_type[0].upper()}2)
   - Multiple instances of the same tag number

4. {element_type.capitalize()}s appear as standard construction symbols in drawings
   - Each drawing discipline has its own symbol conventions
   - Count based on visual symbols, not just labels

IMPORTANT: This is for accurate construction estimation. Missing even one {element_type} affects costs and compliance.

Original question: {original_prompt}

RESPOND WITH:
COUNT: [exact total number - be extremely thorough]
CONFIDENCE: [HIGH if symbols are clear, MEDIUM if some uncertainty, LOW if unclear]
FOUND_ON_PAGES: [list every page number where you found {element_type}s]
ELEMENT_TAGS: [list all tags like {element_type[0].upper()}1, {element_type[0].upper()}2, etc.]
AREAS_CHECKED: [confirm you checked plans, elevations, sections, details, etc.]"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are a meticulous construction estimator counting {element_type}s. Accuracy is critical - missing items causes budget overruns. Be extremely thorough.",
                max_tokens=1500
            )
        
        if response:
            return self._parse_pass1_response(response, element_type)
        
        return {"count": 0, "confidence": "LOW", "pages": [], "tags": [], "areas_checked": []}
    
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
        
        prompt = f"""PASS 2: SPATIAL VERIFICATION - LOCATE EVERY {element_type.upper()}

Count ALL {element_type}s again by listing WHERE each one is located. This verifies nothing was missed.

GRID REFERENCE SYSTEM:
- Look for grid lines on the drawings
- Horizontal: Usually letters (A, B, C, D...)
- Vertical: Usually numbers (1, 2, 3, 4...)
- Reference format: Letter-Number (e.g., A-1, B-3)
- Between grids: Use decimals (e.g., B-2.5)
- No grid? Use room names, compass directions, or descriptive locations

YOUR TASK: List EVERY SINGLE {element_type} with its location
- Start from top-left of each drawing
- Work systematically across and down
- Check all drawing types (plans, elevations, sections, details)
- Don't skip partial views or edges

FORMAT YOUR RESPONSE:
TOTAL COUNT: [number]
GRID_SYSTEM: [YES/NO - describe what you see]

DETAILED LOCATIONS:
1. [Location] - [Tag if any] - [Description/Type]
2. [Location] - [Tag if any] - [Description/Type]
...
[Continue numbering until you've listed EVERY {element_type}]

COVERAGE CHECK:
- Main plans: [number found]
- Elevations: [number found]  
- Sections: [number found]
- Details: [number found]
- Other areas: [number found]

VERIFICATION: 
- Did you check every drawing area? [YES/NO]
- Any areas difficult to read? [describe]

Original question: {original_prompt}"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are verifying {element_type} locations for construction coordination. List every single one - missing locations causes installation errors.",
                max_tokens=4000
            )
        
        if response:
            return self._parse_pass2_response(response, element_type)
        
        return {"count": 0, "locations": [], "grid_references": [], "coverage": {}}
    
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
        
        prompt = f"""PASS 3: COMPREHENSIVE TEXT REVIEW - VERIFY {element_type.upper()} COUNT

Review ALL text, schedules, and written information about {element_type}s in these drawings.

SYSTEMATIC REVIEW:

1. SCHEDULES (Highest Priority):
   - Look for "{element_type.upper()} SCHEDULE" or similar tables
   - Equipment schedules listing {element_type}s
   - Count total quantities in schedules
   - Note each type/model and its quantity

2. GENERAL/KEYNOTES:
   - General notes sections
   - Keynotes
   - Installation notes
   - Quantity statements (e.g., "provide {element_type}s at...")

3. DRAWING LABELS:
   - Count how many times each {element_type} tag appears (e.g., {element_type[0].upper()}1, {element_type[0].upper()}2)
   - Room labels mentioning {element_type}s
   - Detail callouts

4. SPECIFICATIONS ON SHEETS:
   - Written specifications
   - Performance requirements
   - Material lists
   - Code requirements

5. LEGENDS:
   - Symbol definitions
   - {element_type.capitalize()} types shown

6. TITLE BLOCK:
   - Project data
   - Drawing lists
   - Quantities summary

{text_context}

Original question: {original_prompt}

PROVIDE:
SCHEDULE_FOUND: [YES with exact schedule name / NO]
SCHEDULE_COUNT: [exact total from schedule, or "No schedule found"]
SCHEDULE_BREAKDOWN: [if schedule exists, list each type with quantity]

TEXT_REFERENCES:
- [List every text reference to {element_type} quantities]
- [Include where you found each reference]

TAG_COUNT: [Count of unique {element_type} tags found]
TAG_LIST: [List all unique tags]

KEY_FINDINGS:
- [Important specifications]
- [Installation requirements]
- [Any quantity statements]

CROSS-CHECK:
- Visual count suggests approximately how many {element_type}s?
- Does text/schedule support this count? [explain]"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are reviewing documentation for {element_type}s. Accurate counts prevent material shortages and change orders. Check everything.",
                max_tokens=2500
            )
        
        if response:
            return self._parse_pass3_response(response, element_type)
        
        return {"text_count": None, "schedule_count": None, "sources": [], "findings": [], "tags": []}
    
    async def _pass4_cross_reference(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        original_prompt: str,
        pass1_result: Dict[str, Any],
        pass2_result: Dict[str, Any],
        pass3_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 4: Cross-Reference & Deduplication
        Identify elements that might be counted multiple times across different views
        """
        
        # Reference previous counts for context
        visual_count = pass1_result.get("count", 0)
        spatial_count = pass2_result.get("count", 0)
        schedule_count = pass3_result.get("schedule_count", 0) if pass3_result.get("schedule_count") != "No schedule found" else 0
        
        prompt = f"""PASS 4: CROSS-REFERENCE & DEDUPLICATION CHECK

Previous counts found:
- Visual Count: {visual_count} {element_type}s
- Spatial Count: {spatial_count} {element_type}s  
- Schedule Count: {schedule_count} {element_type}s (if any)

YOUR CRITICAL TASK: Determine the TRUE count by identifying:

1. MULTI-VIEW ANALYSIS:
   - Are the same {element_type}s shown in BOTH plan AND elevation views?
   - Do section cuts show {element_type}s already counted in plans?
   - Are detail drawings showing the SAME {element_type}s as the main plans?

2. SCALE OVERLAP CHECK:
   - Look for "ENLARGED PLAN" or "DETAIL" areas
   - These often show the SAME {element_type}s at larger scale
   - Match reference bubbles (like 1/A5.1) to their source

3. TYPICAL DETAILS:
   - Does any detail say "TYPICAL" or "TYP"?
   - If yes, how many times should this typical detail be applied?
   - Look for notes like "SEE TYPICAL DETAIL FOR ALL {element_type.upper()}S"

4. DRAWING BOUNDARIES:
   - Check for "MATCH LINE" or "SEE SHEET XX"
   - Are {element_type}s split across drawing boundaries?
   - Look for continuation symbols

5. UNIQUE IDENTIFICATION:
   - Use {element_type} tags (like {element_type[0].upper()}1, {element_type[0].upper()}2) to identify unique elements
   - If an element has the same tag in multiple views, it's the SAME element
   - Count each unique tag only ONCE

SYSTEMATIC VERIFICATION:
- First, identify all UNIQUE {element_type} tags/marks
- Then, count unmarked {element_type}s that are clearly different
- Finally, check if "typical" conditions multiply the count

CRITICAL QUESTIONS:
- Are elevations showing the SAME {element_type}s as the floor plans? [YES/NO]
- Are enlarged details showing NEW {element_type}s or just bigger views? [NEW/SAME]
- Do sections cut through {element_type}s already counted? [YES/NO]

PROVIDE:
VERIFIED_COUNT: [the TRUE total after deduplication]
UNIQUE_TAGS: [list all unique {element_type} identifiers]
DUPLICATION_FOUND:
- Plan/Elevation overlap: [describe any]
- Detail/Plan overlap: [describe any]
- Section duplicates: [describe any]
TYPICAL_MULTIPLIER: [if typical details apply to multiple locations]
CONFIDENCE: [HIGH/MEDIUM/LOW in this verified count]

EXPLANATION: [Explain how you arrived at the true count]

Original question: {original_prompt}"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are a construction document coordinator preventing double-counting of {element_type}s across multiple drawing views. Accurate deduplication prevents material over-ordering.",
                max_tokens=3000
            )
        
        if response:
            return self._parse_pass4_response(response, element_type)
        
        return {"verified_count": None, "unique_tags": [], "duplication_found": {}, "confidence": "LOW"}
    
    def _parse_pass4_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 4 cross-reference response"""
        result = {
            "verified_count": None,
            "unique_tags": [],
            "duplication_found": {},
            "typical_multiplier": 1,
            "confidence": "MEDIUM",
            "explanation": ""
        }
        
        # Extract verified count
        verified_match = re.search(r'VERIFIED_COUNT:\s*(\d+)', response, re.IGNORECASE)
        if verified_match:
            result["verified_count"] = int(verified_match.group(1))
        
        # Extract unique tags
        tags_section = re.search(r'UNIQUE_TAGS:(.*?)(?:DUPLICATION_FOUND:|TYPICAL_MULTIPLIER:|$)', response, re.IGNORECASE | re.DOTALL)
        if tags_section:
            tags_text = tags_section.group(1)
            found_tags = re.findall(r'\b[A-Z]+\d+\b', tags_text)
            result["unique_tags"] = sorted(list(set(found_tags)))
        
        # Extract duplication information
        dup_section = re.search(r'DUPLICATION_FOUND:(.*?)(?:TYPICAL_MULTIPLIER:|CONFIDENCE:|$)', response, re.IGNORECASE | re.DOTALL)
        if dup_section:
            dup_text = dup_section.group(1)
            
            # Look for specific overlap types
            if "plan/elevation overlap:" in dup_text.lower():
                overlap_match = re.search(r'plan/elevation overlap:\s*(.+?)(?:\n|$)', dup_text, re.IGNORECASE)
                if overlap_match:
                    result["duplication_found"]["plan_elevation"] = overlap_match.group(1).strip()
            
            if "detail/plan overlap:" in dup_text.lower():
                detail_match = re.search(r'detail/plan overlap:\s*(.+?)(?:\n|$)', dup_text, re.IGNORECASE)
                if detail_match:
                    result["duplication_found"]["detail_plan"] = detail_match.group(1).strip()
        
        # Extract typical multiplier
        multiplier_match = re.search(r'TYPICAL_MULTIPLIER:\s*(\d+)', response, re.IGNORECASE)
        if multiplier_match:
            result["typical_multiplier"] = int(multiplier_match.group(1))
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()
        
        # Extract explanation
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:Original question:|$)', response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()
        
        return result
    
    def _build_consensus_result(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        pass4: Dict[str, Any],
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
        
        # Determine consensus count - prefer higher counts to avoid missing items
        if counts:
            # If counts vary, consider using the maximum to avoid undercounting
            count_freq = Counter(counts)
            mode_count = count_freq.most_common(1)[0][0]
            max_count = max(counts)
            
            # If there's significant variance, lean toward higher count
            if max_count - min(counts) > 2:
                consensus_count = max_count
                logger.warning(f"Count variance detected: {counts}. Using maximum: {max_count}")
            else:
                consensus_count = mode_count
        else:
            consensus_count = 0
        
        # Calculate confidence based on agreement
        confidence = self._calculate_consensus_confidence(
            pass1, pass2, pass3, counts
        )
        
        # Build verification notes
        verification_notes = []
        
        # Add Pass 0 if it found something
        if pass0.get("count", 0) > 0:
            verification_notes.append(f"Pass 0 (Pure GPT-4V): {pass0.get('count', 0)} {element_type}s")
        
        verification_notes.append(f"Pass 1 (Visual): {pass1.get('count', 0)} {element_type}s")
        verification_notes.append(f"Pass 2 (Spatial): {pass2.get('count', 0)} {element_type}s")
        
        if pass3.get('schedule_count') is not None and pass3['schedule_count'] != "No schedule found":
            verification_notes.append(f"Pass 3 (Text): Schedule shows {pass3['schedule_count']} {element_type}s")
        elif pass3.get('text_count') is not None:
            verification_notes.append(f"Pass 3 (Text): Documentation indicates {pass3['text_count']} {element_type}s")
        else:
            verification_notes.append(f"Pass 3 (Text): No quantity found in text")
        
        # Add areas checked from pass1
        if pass1.get("areas_checked"):
            verification_notes.append(f"Areas verified: {', '.join(pass1['areas_checked'])}")
        
        # Add consensus note
        if len(set(counts)) == 1 and len(counts) >= 2:
            verification_notes.append("✅ PERFECT CONSENSUS achieved")
        elif confidence >= 0.90:
            verification_notes.append("✅ Strong agreement between passes")
        else:
            verification_notes.append("⚠️ Some variation between verification passes - using highest count to avoid missing items")
        
        # Build visual evidence
        visual_evidence = []
        
        # Add Pass 0 insight if it was close
        if pass0.get("count", 0) > 0:
            visual_evidence.append(f"Pure GPT-4V detected {pass0['count']} {element_type}s")
        
        if pass2.get("locations"):
            visual_evidence.append(f"Found in {len(set(pass2.get('grid_references', [])))} grid locations")
        if pass1.get("pages"):
            visual_evidence.append(f"Appears on pages: {', '.join(map(str, pass1['pages']))}")
        if pass1.get("tags"):
            visual_evidence.append(f"Element tags found: {', '.join(pass1['tags'][:10])}")
        if pass2.get("coverage"):
            coverage = pass2["coverage"]
            visual_evidence.append(f"Distribution: Plans({coverage.get('plans', 0)}), Elevations({coverage.get('elevations', 0)}), Details({coverage.get('details', 0)})")
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
            "consensus_achieved": len(set(counts)) == 1 and len(counts) >= 2,
            "element_tags": pass1.get("tags", []),
            "areas_checked": pass1.get("areas_checked", []),
            "coverage_breakdown": pass2.get("coverage", {})
        }
        
        return result
    
    def _calculate_consensus_confidence_with_pass4(
        self,
        pass0: Dict[str, Any],
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        pass4: Dict[str, Any],
        counts: List[int],
        count_sources: List[Tuple[str, int]],
        consensus_count: int
    ) -> float:
        """Calculate confidence based on consensus between all 5 passes"""
        
        if not counts:
            return 0.5
        
        # Special weight if Pass 0 (pure GPT) agrees with final consensus
        pass0_agrees = (pass0.get("count") == consensus_count and pass0.get("count", 0) > 0)
        
        # If Pass 4 verified with high confidence and it matches consensus
        if (pass4.get("verified_count") == consensus_count and 
            pass4.get("confidence") == "HIGH"):
            if pass0_agrees:
                return 0.995  # Highest - both pure and verified agree
            return 0.99  # Very high - verified deduplication matches
        
        # Check for perfect consensus across all passes
        unique_counts = set(counts)
        if len(unique_counts) == 1:
            if len(counts) >= 5:
                return 0.99  # All 5 data points agree
            elif len(counts) >= 4:
                return 0.97  # 4 agree
            elif len(counts) >= 3:
                return 0.95  # 3 agree
            elif len(counts) == 2:
                return 0.92  # 2 agree
        
        # Near consensus (within 1-2)
        if len(counts) >= 2:
            count_range = max(counts) - min(counts)
            if count_range <= 1:
                base_confidence = 0.90
            elif count_range <= 2:
                base_confidence = 0.85
            elif count_range <= 3:
                base_confidence = 0.80
            else:
                base_confidence = 0.70
        else:
            base_confidence = 0.70
        
        # Confidence adjustments
        confidence_adjustment = 0.0
        
        # Pass quality bonuses
        if pass0.get("confidence") == "HIGH" and pass0.get("count", 0) > 0:
            confidence_adjustment += 0.04  # Pure GPT confidence bonus
        if pass1.get("confidence") == "HIGH":
            confidence_adjustment += 0.03
        if pass2.get("locations") and len(pass2["locations"]) > 0:
            confidence_adjustment += 0.03
        if pass3.get("schedule_count") is not None and pass3.get("schedule_count") != "No schedule found":
            confidence_adjustment += 0.05
        if pass4.get("confidence") == "HIGH":
            confidence_adjustment += 0.05
        elif pass4.get("confidence") == "MEDIUM":
            confidence_adjustment += 0.03
        
        # Deduplication bonus - finding duplicates increases confidence
        if pass4.get("duplication_found"):
            has_duplication = any(
                desc and desc.lower() not in ["none", "no", "n/a", ""]
                for desc in pass4["duplication_found"].values()
            )
            if has_duplication:
                confidence_adjustment += 0.04  # Successfully identified duplicates
        
        # Coverage bonus
        if pass1.get("areas_checked") and len(pass1["areas_checked"]) >= 4:
            confidence_adjustment += 0.02
        
        # Unique tags bonus - more unique tags = better tracking
        all_tags = set()
        if pass1.get("tags"):
            all_tags.update(pass1["tags"])
        if pass4.get("unique_tags"):
            all_tags.update(pass4["unique_tags"])
        if len(all_tags) >= consensus_count * 0.7:  # Most elements are tagged
            confidence_adjustment += 0.03
        
        final_confidence = base_confidence + confidence_adjustment
        return min(max(final_confidence, 0.5), 0.99)
    
    def _calculate_consensus_confidence(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        counts: List[int]
    ) -> float:
        """Calculate confidence based on consensus between passes (kept for compatibility - not used in penta verification)"""
        
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
            elif count_range <= 3:
                return 0.80
            else:
                # Large variance - lower confidence
                return 0.70
        
        # Check pass confidence levels
        base_confidence = 0.70
        
        if pass1.get("confidence") == "HIGH":
            base_confidence += 0.05
        if pass2.get("locations") and len(pass2["locations"]) > 0:
            base_confidence += 0.05
        if pass3.get("schedule_count") is not None and pass3.get("schedule_count") != "No schedule found":
            base_confidence += 0.10
        
        # Check if all areas were verified
        if pass1.get("areas_checked") and len(pass1["areas_checked"]) >= 3:
            base_confidence += 0.03
        
        return min(base_confidence, 0.95)
    
    def _prepare_vision_content(
        self,
        images: List[Dict[str, Any]],
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare content for vision API request"""
        content = []
        
        # Enhanced context about thoroughness
        content.append({
            "type": "text",
            "text": f"""CRITICAL: You are examining {len(images)} page(s) of construction drawings.
            
Your analysis affects construction costs and compliance. Be EXTREMELY thorough.
Missing even one element causes problems. Check every area of every drawing.
Count everything - it's better to overcount than undercount."""
        })
        
        # Add images
        for i, image in enumerate(images):
            if len(images) > 1:
                content.append({
                    "type": "text",
                    "text": f"=== PAGE {i+1} of {len(images)} ==="
                })
            
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
        """Parse Pass 1 response - enhanced to capture tags and areas"""
        result = {
            "count": 0,
            "confidence": "MEDIUM",
            "pages": [],
            "tags": [],
            "areas_checked": []
        }
        
        # Extract count - try multiple patterns
        count_patterns = [
            r'COUNT:\s*(\d+)',
            r'TOTAL:\s*(\d+)',
            r'Total Count:\s*(\d+)',
            r'Found:\s*(\d+)',
            r'(\d+)\s+total'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))
                break
        
        # If still no count, look for any number followed by element type
        if result["count"] == 0:
            pattern = r'(\d+)\s*' + element_type
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()
        
        # Extract pages
        pages_section = re.search(r'FOUND_ON_PAGES:(.*?)(?:ELEMENT_TAGS:|AREAS_CHECKED:|$)', response, re.IGNORECASE | re.DOTALL)
        if pages_section:
            pages_text = pages_section.group(1)
            page_numbers = re.findall(r'\d+', pages_text)
            result["pages"] = sorted(list(set(int(p) for p in page_numbers)))
        
        # Extract element tags
        tags_section = re.search(r'ELEMENT_TAGS:(.*?)(?:AREAS_CHECKED:|$)', response, re.IGNORECASE | re.DOTALL)
        if tags_section:
            tags_text = tags_section.group(1)
            # Look for patterns like W1, D2, P3, etc.
            found_tags = re.findall(r'\b[A-Z]+\d+\b', tags_text)
            result["tags"] = sorted(list(set(found_tags)))
        
        # Extract areas checked
        areas_section = re.search(r'AREAS_CHECKED:(.*?)$', response, re.IGNORECASE | re.DOTALL)
        if areas_section:
            areas_text = areas_section.group(1)
            # Common area names
            area_keywords = ['plan', 'elevation', 'section', 'detail', 'schedule', 'typical', 'enlarged']
            areas_found = []
            for keyword in area_keywords:
                if keyword in areas_text.lower():
                    areas_found.append(keyword.capitalize() + 's')
            result["areas_checked"] = areas_found
        
        return result
    
    def _parse_pass2_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 2 response with locations and coverage"""
        result = {
            "count": 0,
            "locations": [],
            "grid_references": [],
            "coverage": {}
        }
        
        # Extract total count
        count_match = re.search(r'TOTAL COUNT:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            result["count"] = int(count_match.group(1))
        
        # Extract detailed locations
        locations_section = re.search(
            r'DETAILED LOCATIONS:(.*?)(?:COVERAGE CHECK:|VERIFICATION:|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if locations_section:
            location_text = locations_section.group(1)
            
            # Multiple patterns for different formatting
            location_patterns = [
                # Standard: 1. A-1 - W1 - Description
                r'(\d+)\.\s*([A-Z]-?\d+(?:\.\d+)?)\s*[-–]\s*(?:([A-Z]+\d+)\s*[-–]\s*)?(.+?)(?=\n\d+\.|$)',
                # Alternative: 1. Grid A-1 - Description
                r'(\d+)\.\s*(?:Grid\s+)?([A-Z]-?\d+(?:\.\d+)?)\s*[-–]\s*(.+?)(?=\n\d+\.|$)',
                # Room-based: 1. Room 101 - Description
                r'(\d+)\.\s*(Room\s+\w+|Area\s+\w+|[NSEW]+\s+wall)\s*[-–]\s*(.+?)(?=\n\d+\.|$)',
                # Simple numbered list
                r'(\d+)\.\s*(.+?)(?=\n\d+\.|$)'
            ]
            
            for pattern in location_patterns:
                matches = list(re.finditer(pattern, location_text, re.MULTILINE))
                if matches:
                    for match in matches:
                        groups = match.groups()
                        if len(groups) >= 4:  # Full format with tag
                            grid_ref = groups[1]
                            tag = groups[2] or ""
                            description = groups[3].strip()
                        elif len(groups) >= 3:  # Grid without tag
                            grid_ref = groups[1]
                            description = groups[2].strip()
                            tag = ""
                        else:  # Simple format
                            grid_ref = f"Location {groups[0]}"
                            description = groups[1].strip()
                            tag = ""
                        
                        # Extract tag from description if not found
                        if not tag:
                            tag_match = re.search(r'\b([A-Z]+\d+)\b', description)
                            tag = tag_match.group(1) if tag_match else ""
                        
                        location_info = {
                            "grid_ref": grid_ref,
                            "element_tag": tag if tag else None,
                            "visual_details": description
                        }
                        
                        result["locations"].append(location_info)
                        if not grid_ref.startswith("Location") and "Room" not in grid_ref:
                            result["grid_references"].append(grid_ref)
                    
                    break  # Use first pattern that works
        
        # Extract coverage information
        coverage_section = re.search(
            r'COVERAGE CHECK:(.*?)(?:VERIFICATION:|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if coverage_section:
            coverage_text = coverage_section.group(1)
            
            # Extract numbers for each area type
            coverage_patterns = [
                (r'Main plans?:\s*(\d+)', 'plans'),
                (r'Elevations?:\s*(\d+)', 'elevations'),
                (r'Sections?:\s*(\d+)', 'sections'),
                (r'Details?:\s*(\d+)', 'details'),
                (r'Other areas?:\s*(\d+)', 'other')
            ]
            
            for pattern, key in coverage_patterns:
                match = re.search(pattern, coverage_text, re.IGNORECASE)
                if match:
                    result["coverage"][key] = int(match.group(1))
        
        # If no count from total but we have locations, count them
        if result["count"] == 0 and result["locations"]:
            result["count"] = len(result["locations"])
        
        return result
    
    def _parse_pass3_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parse Pass 3 text review response"""
        result = {
            "text_count": None,
            "schedule_count": None,
            "sources": [],
            "findings": [],
            "tags": [],
            "schedule_breakdown": {}
        }
        
        # Extract schedule found status
        schedule_found = re.search(r'SCHEDULE_FOUND:\s*(YES|NO)', response, re.IGNORECASE)
        if schedule_found and schedule_found.group(1).upper() == "YES":
            # Extract schedule count
            schedule_count_match = re.search(r'SCHEDULE_COUNT:\s*(\d+)', response, re.IGNORECASE)
            if schedule_count_match:
                result["schedule_count"] = int(schedule_count_match.group(1))
            
            # Extract schedule breakdown
            breakdown_section = re.search(
                r'SCHEDULE_BREAKDOWN:(.*?)(?:TEXT_REFERENCES:|TAG_COUNT:|KEY_FINDINGS:|$)',
                response, re.IGNORECASE | re.DOTALL
            )
            if breakdown_section:
                breakdown_text = breakdown_section.group(1)
                # Parse entries like "Type W1: 15 units"
                type_pattern = r'(?:Type\s+)?([A-Z]+\d+)[:\s]+(\d+)'
                for match in re.finditer(type_pattern, breakdown_text):
                    type_code = match.group(1)
                    quantity = int(match.group(2))
                    result["schedule_breakdown"][type_code] = quantity
        
        # Extract text references
        text_refs_section = re.search(
            r'TEXT_REFERENCES:(.*?)(?:TAG_COUNT:|KEY_FINDINGS:|CROSS-CHECK:|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        if text_refs_section:
            refs_text = text_refs_section.group(1)
            refs = [r.strip() for r in refs_text.split('\n') if r.strip() and r.strip().startswith('-')]
            result["sources"] = [r.lstrip('- ') for r in refs]
            
            # Try to extract a count from text references
            for ref in result["sources"]:
                count_match = re.search(r'(\d+)\s*' + element_type, ref, re.IGNORECASE)
                if count_match and result["text_count"] is None:
                    result["text_count"] = int(count_match.group(1))
        
        # Extract tag information
        tag_count_match = re.search(r'TAG_COUNT:\s*(\d+)', response, re.IGNORECASE)
        tag_list_section = re.search(r'TAG_LIST:(.*?)(?:KEY_FINDINGS:|CROSS-CHECK:|$)', response, re.IGNORECASE | re.DOTALL)
        
        if tag_list_section:
            tag_text = tag_list_section.group(1)
            found_tags = re.findall(r'\b[A-Z]+\d+\b', tag_text)
            result["tags"] = sorted(list(set(found_tags)))
        
        # Extract key findings
        findings_section = re.search(
            r'KEY_FINDINGS:(.*?)(?:CROSS-CHECK:|$)',
            response, re.IGNORECASE | re.DOTALL
        )
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
Common elements include: door, window, outlet, panel, light, fixture, sprinkler, diffuser, column, beam, etc.

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
            "light": {"width": 48, "height": 24},  # Alias
            "fixture": {"width": 48, "height": 24},  # Alias
            "sprinkler": {"radius": 10},
            "column": {"width": 24, "height": 24},
            "diffuser": {"width": 24, "height": 24},
            "switch": {"width": 4, "height": 4},
            "thermostat": {"radius": 8},
            "vav box": {"width": 36, "height": 24},
            "beam": {"width": 12, "height": 24}
        }
        return dimensions.get(element_type, {"width": 50, "height": 50})