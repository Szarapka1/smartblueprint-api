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
        
        # Enhanced element knowledge base
        self.element_knowledge = self._build_element_knowledge()
    
    def _build_element_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive knowledge base for all construction elements"""
        return {
            "electrical": {
                "elements": ["outlet", "switch", "panel", "junction box", "disconnect", "transformer"],
                "symbols": "circles, squares with internal markings, rectangles with labels",
                "tags": ["E", "P", "J", "T", "XFMR"],
                "schedules": ["panel schedule", "electrical equipment schedule", "load schedule"]
            },
            "plumbing": {
                "elements": ["fixture", "valve", "cleanout", "floor drain", "water heater"],
                "symbols": "circles, triangles, special fixture symbols per plumbing code",
                "tags": ["P", "FD", "CO", "V", "WH"],
                "schedules": ["plumbing fixture schedule", "equipment schedule"]
            },
            "hvac": {
                "elements": ["diffuser", "grille", "vav box", "fan", "unit heater", "thermostat"],
                "symbols": "squares with cross patterns, rectangles with diagonals, circles with letters",
                "tags": ["AD", "RG", "VAV", "UH", "T", "RTU"],
                "schedules": ["hvac equipment schedule", "diffuser schedule", "air balance schedule"]
            },
            "fire_life_safety": {
                "elements": ["sprinkler", "smoke detector", "pull station", "fire extinguisher", "exit sign"],
                "symbols": "circles with cross, SD in circle, squares with F, triangles",
                "tags": ["SP", "SD", "FA", "FE", "EXIT"],
                "schedules": ["fire alarm device schedule", "sprinkler schedule"]
            },
            "architectural": {
                "elements": ["door", "window", "wall", "column", "stair", "elevator"],
                "symbols": "arc in wall opening, parallel lines in wall, heavy lines, circles/squares with fill",
                "tags": ["D", "W", "C", "S", "ELEV"],
                "schedules": ["door schedule", "window schedule", "finish schedule", "room schedule"]
            },
            "structural": {
                "elements": ["beam", "column", "footing", "slab", "brace", "joist"],
                "symbols": "heavy lines, rectangles with hatching, dashed lines for hidden",
                "tags": ["B", "C", "F", "S", "W", "J"],
                "schedules": ["column schedule", "beam schedule", "footing schedule"]
            }
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
            self.vision_semaphore = asyncio.Semaphore(3)
        if self.inference_semaphore is None:
            self.inference_semaphore = asyncio.Semaphore(CONFIG["VISION_INFERENCE_LIMIT"])
    
    def _get_element_category(self, element_type: str) -> str:
        """Determine which category an element belongs to"""
        element_lower = element_type.lower()
        for category, info in self.element_knowledge.items():
            if any(element_lower in elem or elem in element_lower for elem in info["elements"]):
                return category
        return "general"
    
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
            
            # Get element category for better prompting
            element_category = self._get_element_category(element_type)
            
            # PASS 1: Direct Visual Count
            logger.info("👁️ PASS 1: Direct Visual Count")
            pass1_result = await self._pass1_visual_count(
                element_type, images, prompt, element_category
            )
            
            # PASS 2: Visual + Spatial Verification
            logger.info("📍 PASS 2: Visual + Spatial Verification")
            pass2_result = await self._pass2_spatial_verification(
                element_type, images, prompt, element_category, pass1_result
            )
            
            # PASS 3: Text Review
            logger.info("📄 PASS 3: Document Text Review")
            pass3_result = await self._pass3_text_review(
                element_type, images, extracted_text, prompt, element_category, pass1_result, pass2_result
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
        original_prompt: str,
        element_category: str
    ) -> Dict[str, Any]:
        """
        Pass 1: Enhanced visual count with detailed guidance
        """
        
        # Get category-specific guidance
        category_info = self.element_knowledge.get(element_category, {})
        
        prompt = f"""PASS 1: COMPREHENSIVE VISUAL COUNT - {element_type.upper()}S

You are a master construction drawing analyst examining professional architectural/engineering drawings. These drawings follow standard industry conventions (NCS, AIA, CSI).

DRAWING CONTEXT:
- Total pages provided: {len(images)}
- Drawing types may include: floor plans, reflected ceiling plans, elevations, sections, details, schedules
- Scale varies by drawing type - pay attention to scale notations

{element_type.upper()} IDENTIFICATION GUIDANCE:
- Category: {element_category.upper()} elements
- Typical symbols: {category_info.get('symbols', 'Standard construction symbols')}
- Common tags: {', '.join(category_info.get('tags', ['Check for alphanumeric tags']))}
- Related elements: {', '.join(category_info.get('elements', []))}

SYSTEMATIC COUNTING APPROACH:
1. First, scan each drawing to identify drawing type and scale
2. Look for {element_type} symbols in:
   - Main drawing areas (floor plans, elevations, etc.)
   - Detail bubbles and enlarged areas
   - Typical details that may show multiple instances
   - Legend/symbol keys that define what to look for

3. Count methodically:
   - Start from top-left, work systematically across and down
   - Count EVERY instance you see
   - If an element has a tag (like {element_type[0].upper()}1), count each physical instance
   - Don't skip partial views or sectioned elements

4. Special considerations:
   - Elements shown dashed are existing or hidden - still count them
   - Elements in different line weights are different types - count all
   - Check both architectural AND engineering drawings if multiple disciplines shown

Original user question: {original_prompt}

REQUIRED OUTPUT:
COUNT: [exact number - be precise]
CONFIDENCE: [HIGH/MEDIUM/LOW based on symbol clarity]
FOUND_ON_PAGES: [list all page numbers where {element_type}s appear]
DRAWING_TYPES: [list drawing types where found - plan, elevation, section, etc.]
SYMBOL_TYPES: [describe the different symbols used for {element_type}s]
TAGS_FOUND: [list all element tags like {element_type[0].upper()}1, {element_type[0].upper()}2, etc.]
DISTRIBUTION: [brief description of where most {element_type}s are located]

Be thorough and systematic. This count is critical for construction cost estimation and code compliance verification."""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are a professional construction estimator and drawing analyst. You must accurately count {element_type}s for accurate pricing and compliance. Be meticulous.",
                max_tokens=2000
            )
        
        if response:
            return self._parse_pass1_response_enhanced(response, element_type)
        
        return {"count": 0, "confidence": "LOW", "pages": [], "tags": [], "symbol_types": []}
    
    async def _pass2_spatial_verification(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        original_prompt: str,
        element_category: str,
        pass1_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 2: Enhanced spatial verification with detailed location tracking
        """
        
        # Reference Pass 1 findings
        pass1_count = pass1_result.get("count", 0)
        pass1_tags = pass1_result.get("tags", [])
        
        prompt = f"""PASS 2: DETAILED SPATIAL VERIFICATION - {element_type.upper()}S WITH PRECISE LOCATIONS

You are performing spatial verification of the {pass1_count} {element_type}s found in Pass 1.

GRID REFERENCE SYSTEM:
- Horizontal grid lines: Letters (A, B, C, D, etc.) 
- Vertical grid lines: Numbers (1, 2, 3, 4, etc.)
- Grid reference format: Letter-Number (e.g., A-1, B-3, C-2.5)
- If between grid lines, use decimals (e.g., A-1.5 means halfway between 1 and 2)
- If no grid system visible, use compass directions and room names

VERIFICATION APPROACH:
1. For each {element_type}, provide:
   - Exact grid location or descriptive position
   - Element tag if visible (from list: {', '.join(pass1_tags[:10])})
   - Visual description (size, type, special features)
   - What room/space it serves
   - Any adjacent elements or systems

2. Look for patterns:
   - Are {element_type}s evenly distributed?
   - Do they follow structural grid?
   - Any areas with unusual density?
   - Compliance with typical spacing requirements?

3. Cross-check with Pass 1:
   - Expected to find approximately {pass1_count} {element_type}s
   - Verify tags found: {', '.join(pass1_tags[:5])}...

Original question: {original_prompt}

REQUIRED FORMAT:
TOTAL COUNT: [your verified count]
GRID_SYSTEM: [YES with description / NO use alternate location method]

DETAILED LOCATIONS:
1. Grid [location] - [{element_type} tag] - [room/area] - [description]
2. Grid [location] - [{element_type} tag] - [room/area] - [description]
[... continue for ALL {element_type}s found ...]

SPATIAL PATTERNS:
- Distribution: [describe the overall pattern]
- Density: [areas with high/low concentration]
- Typical spacing: [distance between elements]
- Anomalies: [any unusual placements]

VERIFICATION_NOTES:
- Count matches Pass 1: [YES/NO - explain any difference]
- Confidence in locations: [HIGH/MEDIUM/LOW]
- Areas requiring field verification: [list any unclear areas]"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are a construction coordinator verifying {element_type} locations for installation crews. Accurate spatial data is critical for material ordering and labor planning.",
                max_tokens=4000
            )
        
        if response:
            return self._parse_pass2_response_enhanced(response, element_type)
        
        return {"count": 0, "locations": [], "grid_references": [], "spatial_patterns": {}}
    
    async def _pass3_text_review(
        self,
        element_type: str,
        images: List[Dict[str, Any]],
        extracted_text: str,
        original_prompt: str,
        element_category: str,
        pass1_result: Dict[str, Any],
        pass2_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 3: Comprehensive text and schedule review
        """
        
        # Get expected schedule types for this element
        category_info = self.element_knowledge.get(element_category, {})
        expected_schedules = category_info.get('schedules', ['schedule', 'equipment list'])
        
        # Reference previous passes
        pass1_count = pass1_result.get("count", 0)
        pass2_count = pass2_result.get("count", 0)
        visual_consensus = (pass1_count + pass2_count) / 2 if pass2_count > 0 else pass1_count
        
        # Add extracted text context if available
        text_context = ""
        if extracted_text:
            text_context = f"""
EXTRACTED TEXT FROM DOCUMENT (OCR/Text Layer):
{extracted_text[:3000]}...

This extracted text should be cross-referenced with visible text in the drawings.
"""
        
        prompt = f"""PASS 3: COMPREHENSIVE TEXT, SCHEDULE, AND DOCUMENTATION REVIEW - {element_type.upper()}S

You are performing the final verification pass by reviewing ALL text-based information about {element_type}s in these construction documents.

PREVIOUS FINDINGS REFERENCE:
- Pass 1 Visual Count: {pass1_count} {element_type}s
- Pass 2 Spatial Count: {pass2_count} {element_type}s
- Visual consensus suggests approximately {int(visual_consensus)} {element_type}s

SYSTEMATIC TEXT REVIEW APPROACH:

1. SCHEDULES (HIGHEST PRIORITY):
   Look for these schedule types:
   {chr(10).join(f'   - {schedule}' for schedule in expected_schedules)}
   
   For any schedule found:
   - Note the schedule title exactly as shown
   - Count total quantity of {element_type}s listed
   - List each {element_type} type/model with its quantity
   - Note any remarks or special conditions

2. GENERAL NOTES:
   Scan all general notes sections for:
   - {element_type} specifications
   - Installation requirements
   - Quantity statements (e.g., "provide {element_type}s at all...")
   - Performance criteria
   - Code requirements mentioning {element_type}s

3. LEGENDS AND KEYS:
   - Symbol definitions for {element_type}s
   - Abbreviation lists
   - Different {element_type} type indicators

4. KEYED NOTES:
   - Numbered or lettered notes that might reference {element_type}s
   - Detail references
   - Installation notes

5. SPECIFICATIONS ON DRAWINGS:
   - Written specs in drawing margins
   - Detail callouts
   - Section markers with {element_type} info

6. TITLE BLOCK INFORMATION:
   - Project statistics
   - Drawing index mentioning {element_type} drawings
   - Discipline-specific information

{text_context}

Original question: {original_prompt}

REQUIRED OUTPUT:

SCHEDULE_FOUND: [YES with schedule name(s) / NO schedule found]
SCHEDULE_COUNT: [exact total from schedule(s) if found, or "N/A"]
SCHEDULE_BREAKDOWN:
[If schedule exists, list each type with quantity, e.g.:]
- Type W1: 15 units
- Type W2: 8 units
- Type W3: 12 units
SCHEDULE_TOTAL: [sum of all types]

TEXT_QUANTITY_REFERENCES:
[List any text that mentions quantities, e.g.:]
- "Provide outlets at 6' o.c. maximum" 
- "Install 2 per room minimum"
- [Include source location of each reference]

KEY_SPECIFICATIONS:
- [List important specs found in notes]
- [Include sizes, models, ratings]
- [Note installation requirements]

CRITICAL_NOTES:
- [Any code requirements]
- [Special conditions]
- [Coordination notes]

DISCREPANCY_ANALYSIS:
Visual count suggests {int(visual_consensus)} but schedule shows: [compare]
Explanation for any difference: [analyze why counts might differ]

CONFIDENCE_IN_DOCUMENTATION: [HIGH/MEDIUM/LOW]
RECOMMEND_FIELD_VERIFICATION: [YES/NO and why]"""

        content = self._prepare_vision_content(images, prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"You are a construction document controller verifying {element_type} quantities for accurate procurement and avoiding costly change orders. Schedule accuracy is critical.",
                max_tokens=3000
            )
        
        if response:
            return self._parse_pass3_response_enhanced(response, element_type)
        
        return {"text_count": None, "schedule_count": None, "sources": [], "findings": [], "schedule_breakdown": {}}
    
    def _build_consensus_result(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        element_type: str,
        page_number: int
    ) -> VisualIntelligenceResult:
        """
        Enhanced consensus building with detailed verification notes
        """
        
        # Extract all counts
        counts = []
        count_sources = []
        
        if pass1.get("count") is not None:
            counts.append(pass1["count"])
            count_sources.append(("Visual Count", pass1["count"]))
            
        if pass2.get("count") is not None:
            counts.append(pass2["count"])
            count_sources.append(("Spatial Verification", pass2["count"]))
            
        if pass3.get("schedule_count") is not None and pass3["schedule_count"] != "N/A":
            try:
                schedule_count = int(pass3["schedule_count"])
                counts.append(schedule_count)
                count_sources.append(("Schedule Count", schedule_count))
            except:
                pass
                
        if pass3.get("text_count") is not None:
            counts.append(pass3["text_count"])
            count_sources.append(("Text Reference", pass3["text_count"]))
        
        # Determine consensus count using sophisticated logic
        if counts:
            # If we have a schedule count, it often takes precedence
            schedule_count = next((c[1] for c in count_sources if c[0] == "Schedule Count"), None)
            
            # Calculate mode (most common count)
            count_freq = Counter(counts)
            mode_count, mode_freq = count_freq.most_common(1)[0]
            
            # Determine final count
            if schedule_count is not None and mode_freq >= 2:
                # If schedule agrees with majority, use it
                consensus_count = schedule_count if schedule_count == mode_count else mode_count
            elif mode_freq >= 2:
                # Use mode if it appears multiple times
                consensus_count = mode_count
            elif schedule_count is not None:
                # Trust schedule if no strong consensus
                consensus_count = schedule_count
            else:
                # Use average rounded to nearest integer
                consensus_count = round(sum(counts) / len(counts))
        else:
            consensus_count = 0
        
        # Calculate enhanced confidence
        confidence = self._calculate_enhanced_confidence(
            pass1, pass2, pass3, counts, count_sources, consensus_count
        )
        
        # Build detailed verification notes
        verification_notes = self._build_detailed_verification_notes(
            pass1, pass2, pass3, count_sources, consensus_count, confidence
        )
        
        # Build comprehensive visual evidence
        visual_evidence = self._build_visual_evidence(
            pass1, pass2, pass3, element_type
        )
        
        # Extract all unique locations
        all_locations = []
        if pass2.get("locations"):
            all_locations.extend(pass2["locations"])
        
        # Add locations from Pass 1 tags if not in Pass 2
        if pass1.get("tags"):
            for tag in pass1["tags"]:
                if not any(tag in str(loc) for loc in all_locations):
                    all_locations.append({
                        "element_tag": tag,
                        "visual_details": f"{element_type} with tag {tag}",
                        "grid_ref": "See drawings"
                    })
        
        # Create final result
        result = VisualIntelligenceResult(
            element_type=element_type,
            count=consensus_count,
            locations=all_locations[:50],  # Limit to prevent overwhelming
            confidence=confidence,
            visual_evidence=visual_evidence,
            pattern_matches=pass1.get("symbol_types", []),
            grid_references=pass2.get("grid_references", []),
            verification_notes=verification_notes,
            page_number=page_number
        )
        
        # Add comprehensive metadata
        result.analysis_metadata = {
            "method": "triple_verification_enhanced",
            "pass1_count": pass1.get("count", 0),
            "pass2_count": pass2.get("count", 0),
            "pass3_text_count": pass3.get("text_count"),
            "schedule_count": pass3.get("schedule_count"),
            "schedule_breakdown": pass3.get("schedule_breakdown", {}),
            "consensus_achieved": len(set(counts)) == 1 and len(counts) >= 2,
            "all_counts": count_sources,
            "pages_analyzed": pass1.get("pages", []),
            "drawing_types": pass1.get("drawing_types", []),
            "spatial_patterns": pass2.get("spatial_patterns", {}),
            "critical_notes": pass3.get("findings", [])[:5]
        }
        
        return result
    
    def _calculate_enhanced_confidence(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        counts: List[int],
        count_sources: List[Tuple[str, int]],
        consensus_count: int
    ) -> float:
        """Calculate confidence with sophisticated logic"""
        
        if not counts:
            return 0.5
        
        # Perfect consensus scenarios
        unique_counts = set(counts)
        if len(unique_counts) == 1:
            if len(counts) >= 3:
                return 0.99  # All 3+ sources agree perfectly
            elif len(counts) == 2:
                return 0.95  # 2 sources agree perfectly
        
        # Check if we have schedule verification
        has_schedule = any(source[0] == "Schedule Count" for source in count_sources)
        schedule_matches = any(
            source[0] == "Schedule Count" and source[1] == consensus_count 
            for source in count_sources
        )
        
        # Strong confidence if schedule matches consensus
        if has_schedule and schedule_matches:
            return 0.93
        
        # Calculate variance in counts
        if len(counts) >= 2:
            count_variance = max(counts) - min(counts)
            if count_variance <= 1:
                base_confidence = 0.90
            elif count_variance <= 2:
                base_confidence = 0.85
            elif count_variance <= 3:
                base_confidence = 0.80
            else:
                base_confidence = 0.70
        else:
            base_confidence = 0.75
        
        # Adjust based on pass confidence levels
        confidence_adjustment = 0.0
        
        if pass1.get("confidence") == "HIGH":
            confidence_adjustment += 0.03
        elif pass1.get("confidence") == "LOW":
            confidence_adjustment -= 0.05
            
        if pass2.get("locations") and len(pass2["locations"]) > 0:
            confidence_adjustment += 0.03
            
        if has_schedule:
            confidence_adjustment += 0.04
        
        # Spatial pattern bonus
        if pass2.get("spatial_patterns", {}).get("distribution") == "even":
            confidence_adjustment += 0.02
        
        final_confidence = base_confidence + confidence_adjustment
        return min(max(final_confidence, 0.5), 0.99)
    
    def _build_detailed_verification_notes(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        count_sources: List[Tuple[str, int]],
        consensus_count: int,
        confidence: float
    ) -> List[str]:
        """Build detailed verification notes for the response"""
        
        notes = []
        
        # Add count summary
        count_summary = " | ".join([f"{source}: {count}" for source, count in count_sources])
        notes.append(f"Count Verification: {count_summary}")
        
        # Add consensus analysis
        unique_counts = set(count for _, count in count_sources)
        if len(unique_counts) == 1 and len(count_sources) >= 2:
            notes.append("✅ PERFECT CONSENSUS - All verification methods agree")
        elif confidence >= 0.90:
            notes.append("✅ Strong agreement between verification methods")
        elif confidence >= 0.80:
            notes.append("✓ Good agreement with minor variations")
        else:
            notes.append("⚠️ Some disagreement between verification methods - field verification recommended")
        
        # Add method-specific notes
        if pass1.get("pages"):
            notes.append(f"Found on pages: {', '.join(map(str, pass1['pages']))}")
            
        if pass1.get("drawing_types"):
            notes.append(f"Drawing types analyzed: {', '.join(pass1['drawing_types'])}")
        
        # Add spatial distribution note
        if pass2.get("spatial_patterns"):
            distribution = pass2["spatial_patterns"].get("distribution", "unknown")
            notes.append(f"Spatial distribution: {distribution}")
        
        # Add schedule verification note
        schedule_count = next((c[1] for c in count_sources if c[0] == "Schedule Count"), None)
        if schedule_count is not None:
            if schedule_count == consensus_count:
                notes.append("✅ Schedule count matches field count")
            else:
                notes.append(f"⚠️ Schedule shows {schedule_count} but field count is {consensus_count}")
        else:
            if pass3.get("schedule_found") == "NO":
                notes.append("📋 No schedule found - count based on visual verification only")
        
        # Add confidence statement
        if confidence >= 0.95:
            notes.append(f"Confidence: {int(confidence * 100)}% - EXCELLENT RELIABILITY")
        elif confidence >= 0.90:
            notes.append(f"Confidence: {int(confidence * 100)}% - HIGH RELIABILITY")
        elif confidence >= 0.80:
            notes.append(f"Confidence: {int(confidence * 100)}% - GOOD RELIABILITY")
        else:
            notes.append(f"Confidence: {int(confidence * 100)}% - MODERATE RELIABILITY")
        
        return notes
    
    def _build_visual_evidence(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        element_type: str
    ) -> List[str]:
        """Build comprehensive visual evidence list"""
        
        evidence = []
        
        # Add symbol types found
        if pass1.get("symbol_types"):
            evidence.append(f"Symbol types identified: {', '.join(pass1['symbol_types'][:3])}")
        
        # Add distribution info
        if pass1.get("distribution"):
            evidence.append(f"Distribution: {pass1['distribution']}")
        
        # Add grid coverage
        if pass2.get("grid_references"):
            unique_grids = len(set(pass2["grid_references"]))
            evidence.append(f"Found in {unique_grids} grid locations")
        
        # Add spatial patterns
        if pass2.get("spatial_patterns"):
            patterns = pass2["spatial_patterns"]
            if patterns.get("typical_spacing"):
                evidence.append(f"Typical spacing: {patterns['typical_spacing']}")
            if patterns.get("anomalies"):
                evidence.append(f"Special conditions: {patterns['anomalies']}")
        
        # Add specification highlights
        if pass3.get("findings"):
            for finding in pass3["findings"][:2]:  # Top 2 findings
                if len(finding) < 100:  # Only short findings
                    evidence.append(finding)
        
        # Add tag summary if many found
        if pass1.get("tags") and len(pass1["tags"]) > 5:
            evidence.append(f"Element tags: {', '.join(pass1['tags'][:5])}...")
        elif pass1.get("tags"):
            evidence.append(f"Element tags: {', '.join(pass1['tags'])}")
        
        return evidence
    
    def _prepare_vision_content(
        self,
        images: List[Dict[str, Any]],
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare content for vision API request"""
        content = []
        
        # Add enhanced context about the images
        if len(images) > 1:
            content.append({
                "type": "text",
                "text": f"""You are examining {len(images)} pages of professional construction drawings.
                
IMPORTANT: Analyze EVERY page thoroughly. Elements may appear on any page.
Each page may contain different drawing types (plans, elevations, sections, details, schedules).
Pay special attention to drawing titles and scales."""
            })
        
        # Add images with page indicators
        for i, image in enumerate(images):
            # Add page number reference
            if len(images) > 1:
                content.append({
                    "type": "text",
                    "text": f"--- PAGE {i+1} of {len(images)} ---"
                })
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image["url"],
                    "detail": "high"  # Always use high detail for accuracy
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
        """Make a vision API request with enhanced error handling"""
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o",  # Latest model for best accuracy
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # Deterministic for consistency
                    seed=self.deterministic_seed
                ),
                timeout=CONFIG["VISION_TIMEOUT"]
            )
            
            if response and response.choices:
                return response.choices[0].message.content or ""
                
        except asyncio.TimeoutError:
            logger.error("Vision request timeout - consider increasing timeout for complex drawings")
        except Exception as e:
            logger.error(f"Vision request error: {e}")
        
        return None
    
    def _parse_pass1_response_enhanced(self, response: str, element_type: str) -> Dict[str, Any]:
        """Enhanced parsing for Pass 1 with more fields"""
        result = {
            "count": 0,
            "confidence": "MEDIUM",
            "pages": [],
            "drawing_types": [],
            "symbol_types": [],
            "tags": [],
            "distribution": ""
        }
        
        # Extract count - try multiple patterns
        count_patterns = [
            r'COUNT:\s*(\d+)',
            r'Total:\s*(\d+)',
            r'Found:\s*(\d+)',
            r'(\d+)\s+' + element_type
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["count"] = int(match.group(1))
                break
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()
        
        # Extract pages
        pages_match = re.search(r'FOUND_ON_PAGES:([^\n]+)', response, re.IGNORECASE)
        if pages_match:
            pages_text = pages_match.group(1)
            page_numbers = re.findall(r'\d+', pages_text)
            result["pages"] = sorted(list(set(int(p) for p in page_numbers)))
        
        # Extract drawing types
        drawing_match = re.search(r'DRAWING_TYPES:([^\n]+)', response, re.IGNORECASE)
        if drawing_match:
            drawing_text = drawing_match.group(1)
            result["drawing_types"] = [
                dt.strip() for dt in re.split(r'[,;]', drawing_text)
                if dt.strip()
            ]
        
        # Extract symbol types
        symbol_match = re.search(r'SYMBOL_TYPES:([^\n]+)', response, re.IGNORECASE)
        if symbol_match:
            result["symbol_types"] = [
                st.strip() for st in re.split(r'[,;]', symbol_match.group(1))
                if st.strip()
            ]
        
        # Extract tags
        tags_match = re.search(r'TAGS_FOUND:([^\n]+)', response, re.IGNORECASE)
        if tags_match:
            tags_text = tags_match.group(1)
            # Extract alphanumeric tags
            found_tags = re.findall(r'\b[A-Z]+\d+\b', tags_text)
            result["tags"] = sorted(list(set(found_tags)))
        
        # Extract distribution
        dist_match = re.search(r'DISTRIBUTION:([^\n]+)', response, re.IGNORECASE)
        if dist_match:
            result["distribution"] = dist_match.group(1).strip()
        
        return result
    
    def _parse_pass2_response_enhanced(self, response: str, element_type: str) -> Dict[str, Any]:
        """Enhanced parsing for Pass 2 with spatial patterns"""
        result = {
            "count": 0,
            "locations": [],
            "grid_references": [],
            "spatial_patterns": {
                "distribution": "",
                "density": "",
                "typical_spacing": "",
                "anomalies": ""
            }
        }
        
        # Extract total count
        count_match = re.search(r'TOTAL COUNT:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            result["count"] = int(count_match.group(1))
        
        # Enhanced location parsing
        location_section = re.search(
            r'DETAILED LOCATIONS:(.*?)(?:SPATIAL PATTERNS:|VERIFICATION_NOTES:|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if location_section:
            location_text = location_section.group(1)
            
            # Multiple patterns for different formatting styles
            location_patterns = [
                # Standard format: 1. Grid A-1 - [tag] - [room] - [description]
                r'(\d+)\.\s*Grid\s*([A-Z]-?\d+(?:\.\d+)?)\s*[-–]\s*(?:\[?([A-Z]+\d+)\]?\s*[-–]\s*)?(?:\[?([^][-]+?)\]?\s*[-–]\s*)?(.+?)(?=\n\d+\.|$)',
                # Alternative: Grid A-1: [description]
                r'Grid\s*([A-Z]-?\d+(?:\.\d+)?)[:\s]+(.+?)(?=Grid|$)',
                # Numbered without "Grid": 1. A-1 - [description]
                r'(\d+)\.\s*([A-Z]-?\d+(?:\.\d+)?)\s*[-–:]\s*(.+?)(?=\n\d+\.|$)'
            ]
            
            locations_found = []
            
            for pattern in location_patterns:
                for match in re.finditer(pattern, location_text, re.MULTILINE):
                    if len(match.groups()) >= 5:  # Full format
                        grid_ref = match.group(2)
                        tag = match.group(3) or ""
                        room = match.group(4) or ""
                        description = match.group(5).strip()
                    elif len(match.groups()) >= 3:  # Simplified format
                        grid_ref = match.group(2) if match.group(1) else match.group(1)
                        description = match.group(3) if match.group(1) else match.group(2)
                        tag = ""
                        room = ""
                    else:
                        continue
                    
                    # Extract tag from description if not found
                    if not tag:
                        tag_match = re.search(r'\b([A-Z]+\d+)\b', description)
                        if tag_match:
                            tag = tag_match.group(1)
                    
                    location_info = {
                        "grid_ref": grid_ref,
                        "element_tag": tag,
                        "room": room,
                        "visual_details": description.strip()
                    }
                    
                    locations_found.append(location_info)
                    result["grid_references"].append(grid_ref)
            
            result["locations"] = locations_found
        
        # Extract spatial patterns
        patterns_section = re.search(
            r'SPATIAL PATTERNS:(.*?)(?:VERIFICATION_NOTES:|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if patterns_section:
            patterns_text = patterns_section.group(1)
            
            # Distribution
            dist_match = re.search(r'Distribution:\s*([^\n]+)', patterns_text, re.IGNORECASE)
            if dist_match:
                result["spatial_patterns"]["distribution"] = dist_match.group(1).strip()
            
            # Density
            density_match = re.search(r'Density:\s*([^\n]+)', patterns_text, re.IGNORECASE)
            if density_match:
                result["spatial_patterns"]["density"] = density_match.group(1).strip()
            
            # Spacing
            spacing_match = re.search(r'spacing:\s*([^\n]+)', patterns_text, re.IGNORECASE)
            if spacing_match:
                result["spatial_patterns"]["typical_spacing"] = spacing_match.group(1).strip()
            
            # Anomalies
            anomaly_match = re.search(r'Anomalies:\s*([^\n]+)', patterns_text, re.IGNORECASE)
            if anomaly_match:
                result["spatial_patterns"]["anomalies"] = anomaly_match.group(1).strip()
        
        # If no count from TOTAL COUNT, count the locations
        if result["count"] == 0 and result["locations"]:
            result["count"] = len(result["locations"])
        
        return result
    
    def _parse_pass3_response_enhanced(self, response: str, element_type: str) -> Dict[str, Any]:
        """Enhanced parsing for Pass 3 with schedule breakdown"""
        result = {
            "text_count": None,
            "schedule_count": None,
            "schedule_found": "NO",
            "schedule_breakdown": {},
            "sources": [],
            "findings": [],
            "specifications": [],
            "discrepancy_note": ""
        }
        
        # Check if schedule was found
        schedule_found_match = re.search(r'SCHEDULE_FOUND:\s*(YES|NO)', response, re.IGNORECASE)
        if schedule_found_match:
            result["schedule_found"] = schedule_found_match.group(1).upper()
        elif "schedule" in response.lower() and ("found" in response.lower() or "shows" in response.lower()):
            result["schedule_found"] = "YES"
        
        # Extract schedule count
        schedule_count_match = re.search(r'SCHEDULE_COUNT:\s*(\d+)', response, re.IGNORECASE)
        if schedule_count_match:
            result["schedule_count"] = int(schedule_count_match.group(1))
        elif re.search(r'SCHEDULE_TOTAL:\s*(\d+)', response, re.IGNORECASE):
            total_match = re.search(r'SCHEDULE_TOTAL:\s*(\d+)', response, re.IGNORECASE)
            result["schedule_count"] = int(total_match.group(1))
        
        # Extract schedule breakdown
        breakdown_section = re.search(
            r'SCHEDULE_BREAKDOWN:(.*?)(?:TEXT_QUANTITY|KEY_SPECIFICATIONS|SCHEDULE_TOTAL|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if breakdown_section:
            breakdown_text = breakdown_section.group(1)
            # Parse type and quantity entries
            type_pattern = r'[-•]\s*Type\s*([A-Z0-9-]+)[:\s]+(\d+)\s*(?:units?|each)?'
            for match in re.finditer(type_pattern, breakdown_text, re.IGNORECASE):
                type_code = match.group(1)
                quantity = int(match.group(2))
                result["schedule_breakdown"][type_code] = quantity
        
        # Extract text quantity references
        text_qty_section = re.search(
            r'TEXT_QUANTITY_REFERENCES:(.*?)(?:KEY_SPECIFICATIONS|CRITICAL_NOTES|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if text_qty_section:
            qty_text = text_qty_section.group(1)
            qty_refs = [
                ref.strip() for ref in qty_text.split('\n')
                if ref.strip() and ref.strip().startswith('-')
            ]
            result["sources"].extend([ref.lstrip('- ') for ref in qty_refs])
        
        # Extract specifications
        spec_section = re.search(
            r'KEY_SPECIFICATIONS:(.*?)(?:CRITICAL_NOTES|DISCREPANCY|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if spec_section:
            spec_text = spec_section.group(1)
            specs = [
                spec.strip() for spec in spec_text.split('\n')
                if spec.strip() and spec.strip().startswith('-')
            ]
            result["specifications"] = [spec.lstrip('- ') for spec in specs]
        
        # Extract critical notes as findings
        notes_section = re.search(
            r'CRITICAL_NOTES:(.*?)(?:DISCREPANCY|CONFIDENCE|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if notes_section:
            notes_text = notes_section.group(1)
            notes = [
                note.strip() for note in notes_text.split('\n')
                if note.strip() and note.strip().startswith('-')
            ]
            result["findings"].extend([note.lstrip('- ') for note in notes])
        
        # Extract discrepancy analysis
        discrepancy_match = re.search(
            r'DISCREPANCY_ANALYSIS:(.*?)(?:CONFIDENCE|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        
        if discrepancy_match:
            result["discrepancy_note"] = discrepancy_match.group(1).strip()
        
        # Try to extract a text-based count if mentioned
        text_count_patterns = [
            r'documentation indicates?\s*(\d+)',
            r'notes indicate\s*(\d+)',
            r'specifications show\s*(\d+)',
            r'text shows?\s*(\d+)'
        ]
        
        for pattern in text_count_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["text_count"] = int(match.group(1))
                break
        
        return result
    
    def _create_error_result(self, element_type: str, page_number: int, error_message: str) -> VisualIntelligenceResult:
        """Create an error result with helpful information"""
        return VisualIntelligenceResult(
            element_type=element_type,
            count=0,
            locations=[],
            confidence=0.0,
            visual_evidence=[f"Analysis error: {error_message}"],
            pattern_matches=[],
            grid_references=[],
            verification_notes=[
                "Analysis failed - manual review required",
                "Please ensure drawings are clear and properly formatted",
                "Consider breaking complex drawings into smaller sections"
            ],
            page_number=page_number
        )
    
    async def detect_element_focus(self, prompt: str) -> str:
        """
        Intelligently detect what element the user is asking about
        Enhanced with better context understanding
        """
        
        logger.info(f"🎯 Detecting element focus for: '{prompt}'")
        
        # First check common patterns
        prompt_lower = prompt.lower()
        
        # Direct element mentions
        for element_type in VISUAL_PATTERNS.keys():
            if element_type in prompt_lower:
                return element_type
            # Check plural
            if element_type + 's' in prompt_lower:
                return element_type
        
        # Check variations
        for variation, element_type in ELEMENT_VARIATIONS.items():
            if variation in prompt_lower:
                return element_type
        
        # Use AI for complex detection
        detection_prompt = f"""Identify the construction element type from this question.

Question: "{prompt}"

Consider:
- Common construction terminology
- Trade-specific terms (electrical, plumbing, HVAC, etc.)
- Abbreviations and informal names
- Context clues about what's being counted or analyzed

Return ONLY the element type in lowercase, singular form.
Examples: door, window, outlet, panel, beam, column, sprinkler, etc.
If genuinely unclear, return "element".

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
                                "content": "You are an expert in construction terminology across all trades."
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
                
                # Remove common suffixes
                detected = detected.rstrip('s')  # Remove plural
                
                # Validate and map to known types
                if detected in self.visual_patterns:
                    return detected
                
                # Check if it's a valid construction element
                if detected and len(detected.split()) <= 3 and detected != "element":
                    logger.info(f"Detected element type: {detected}")
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
        Enhanced with actual coordinate detection
        """
        
        if not visual_result.locations:
            return []
        
        # For now, generate estimated geometries based on grid references
        # This could be enhanced to use GPT-4V for precise coordinate detection
        geometries = []
        
        # Estimate grid spacing (typical is 20-30 feet)
        grid_spacing_pixels = 300  # Approximate pixels per grid square
        
        for i, location in enumerate(visual_result.locations):
            grid_ref = location.get("grid_ref", "")
            
            # Parse grid reference (e.g., "B-3" -> col=2, row=3)
            grid_match = re.match(r'([A-Z])-?(\d+(?:\.\d+)?)', grid_ref)
            if grid_match:
                col = ord(grid_match.group(1)) - ord('A') + 1
                row = float(grid_match.group(2))
                
                # Calculate approximate position
                x = col * grid_spacing_pixels
                y = row * grid_spacing_pixels
            else:
                # Default positioning if no grid
                x = 100 + (i % 5) * 200
                y = 100 + (i // 5) * 200
            
            geometry = ElementGeometry(
                element_type=visual_result.element_type,
                geometry_type="auto_detect",
                center_point={"x": x, "y": y},
                boundary_points=[],
                dimensions=self._get_typical_dimensions(visual_result.element_type),
                orientation=0.0
            )
            geometries.append(geometry)
        
        return geometries
    
    def _get_typical_dimensions(self, element_type: str) -> Dict[str, float]:
        """Get typical dimensions for element type - enhanced with more types"""
        dimensions = {
            # Architectural
            "door": {"width": 36, "height": 84},
            "window": {"width": 48, "height": 36},
            "column": {"width": 24, "height": 24},
            "wall": {"width": 8, "height": 120},
            
            # Electrical
            "outlet": {"radius": 15},
            "switch": {"width": 4, "height": 4},
            "panel": {"width": 24, "height": 36},
            "light fixture": {"width": 48, "height": 24},
            "junction box": {"width": 12, "height": 12},
            
            # Mechanical
            "diffuser": {"width": 24, "height": 24},
            "grille": {"width": 24, "height": 12},
            "vav box": {"width": 36, "height": 24},
            "thermostat": {"radius": 8},
            
            # Plumbing
            "fixture": {"width": 24, "height": 30},
            "floor drain": {"radius": 12},
            "cleanout": {"radius": 10},
            
            # Fire/Life Safety
            "sprinkler": {"radius": 10},
            "smoke detector": {"radius": 12},
            "pull station": {"width": 8, "height": 8},
            "exit sign": {"width": 24, "height": 8}
        }
        
        return dimensions.get(element_type, {"width": 50, "height": 50})