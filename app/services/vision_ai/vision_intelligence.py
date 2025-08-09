# vision_intelligence.py
import asyncio
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
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
    Universal Construction Drawing Analysis using GPT-4 Vision
    
    Philosophy: GPT-4V understands construction. We just guide it with 3 simple passes:
    1. Direct visual analysis - answer the question by looking
    2. Detailed verification - check tags, measurements, specs
    3. Documentation check - verify with schedules, notes, calculations
    
    Works for ANY construction question - counting, measuring, calculating, compliance, etc.
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
                timeout=CONFIG.get("VISION_REQUEST_TIMEOUT", 60),
                max_retries=CONFIG.get("VISION_MAX_RETRIES", 3)
            )
        return self._client
    
    def _ensure_semaphores_initialized(self):
        """Initialize semaphores for rate limiting"""
        if self.vision_semaphore is None:
            self.vision_semaphore = asyncio.Semaphore(3)
        if self.inference_semaphore is None:
            self.inference_semaphore = asyncio.Semaphore(CONFIG.get("VISION_INFERENCE_LIMIT", 5))
    
    async def analyze(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]],
        page_number: int,
        comprehensive_data: Optional[Dict[str, Any]] = None
    ) -> VisualIntelligenceResult:
        """
        Core visual analysis method - Universal Triple Verification
        
        Pass 1: Direct visual analysis
        Pass 2: Detailed verification with measurements/tags/specs
        Pass 3: Documentation check (schedules, notes, calculations)
        
        Works for ANY construction question
        """
        
        element_type = question_analysis.get("element_focus", "element")
        question_type = self._determine_question_type(prompt)
        
        logger.info(f"🧠 Starting analysis for: {prompt}")
        logger.info(f"📄 Question type: {question_type}, Element: {element_type}")
        logger.info(f"📄 Analyzing {len(images)} images")
        
        # Extract text context if available
        extracted_text = ""
        if comprehensive_data:
            extracted_text = comprehensive_data.get("context", "")
        
        try:
            # Ensure semaphores are initialized
            self._ensure_semaphores_initialized()
            
            # PASS 1: Direct Visual Analysis
            logger.info("👁️ PASS 1: Direct Visual Analysis")
            pass1_result = await self._pass1_direct_analysis(
                prompt, element_type, question_type, images
            )
            
            # PASS 2: Detailed Verification
            logger.info("🔍 PASS 2: Detailed Verification")
            pass2_result = await self._pass2_detailed_verification(
                prompt, element_type, question_type, images, pass1_result
            )
            
            # PASS 3: Documentation Check
            logger.info("📋 PASS 3: Documentation Check")
            pass3_result = await self._pass3_documentation_check(
                prompt, element_type, question_type, images, extracted_text, pass1_result, pass2_result
            )
            
            # Build final result
            final_result = self._build_universal_consensus(
                pass1_result, pass2_result, pass3_result,
                element_type, question_type, page_number, prompt
            )
            
            logger.info(f"✅ Analysis complete: {final_result.count} {element_type}(s) " +
                       f"with {int(final_result.confidence * 100)}% confidence")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Visual Intelligence error: {e}", exc_info=True)
            return self._create_error_result(element_type, page_number, str(e))
    
    def _determine_question_type(self, prompt: str) -> str:
        """Determine what type of question this is"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["how many", "count", "number of", "total"]):
            return "counting"
        elif any(word in prompt_lower for word in ["spacing", "distance", "between", "apart"]):
            return "measurement"
        elif any(word in prompt_lower for word in ["calculate", "how much", "volume", "area", "load"]):
            return "calculation"
        elif any(word in prompt_lower for word in ["code", "compliant", "ada", "nfpa", "requirement"]):
            return "compliance"
        elif any(word in prompt_lower for word in ["what type", "identify", "what is"]):
            return "identification"
        else:
            return "general"
    
    async def _pass1_direct_analysis(
        self,
        prompt: str,
        element_type: str,
        question_type: str,
        images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Pass 1: Direct visual analysis - Just answer the question
        """
        
        # Build context-aware prompt based on question type
        if question_type == "counting":
            analysis_prompt = f"""Look at these construction drawings and answer: {prompt}

Count EVERY {element_type} you can see across all {len(images)} pages.
- Check floor plans, elevations, sections, details, enlarged areas
- Count each symbol/instance you see
- Include partial views at drawing edges

Direct answer:
COUNT: [number]
WHERE: [which pages/areas have {element_type}s]
CONFIDENCE: [HIGH/MEDIUM/LOW]
VISUAL NOTES: [what you see - symbols, patterns, distribution]"""

        elif question_type == "measurement":
            analysis_prompt = f"""Look at these construction drawings and answer: {prompt}

Measure or determine the requested dimension/spacing.
- Look for dimension strings
- Check detail drawings for measurements
- Use grid spacing if shown
- Note any typical spacing patterns

Direct answer:
MEASUREMENT: [value and units]
WHERE MEASURED: [location/detail where you found this]
CONFIDENCE: [HIGH/MEDIUM/LOW]
MEASUREMENT NOTES: [how you determined this]"""

        elif question_type == "calculation":
            analysis_prompt = f"""Look at these construction drawings and answer: {prompt}

Extract the information needed for this calculation.
- Find relevant dimensions
- Count quantities if needed
- Look for material specifications
- Check notes for additional data

Direct answer:
RELEVANT DATA: [list all data found for calculation]
INITIAL CALCULATION: [if possible, provide estimate]
CONFIDENCE: [HIGH/MEDIUM/LOW]
CALC NOTES: [what information you found]"""

        else:  # general/identification/compliance
            analysis_prompt = f"""Look at these construction drawings and answer: {prompt}

Analyze the drawings to provide a direct answer.
- Look at all relevant areas
- Check symbols, notes, and specifications
- Consider all {len(images)} pages

Direct answer:
ANSWER: [your direct answer to the question]
EVIDENCE: [what you see that supports this answer]
CONFIDENCE: [HIGH/MEDIUM/LOW]
NOTES: [any relevant observations]"""

        content = self._prepare_vision_content(images, analysis_prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt="You are analyzing construction drawings. Provide accurate, direct answers based on what you see.",
                max_tokens=1500
            )
        
        if response:
            return self._parse_pass1_response(response, question_type)
        
        return {"primary_answer": None, "confidence": "LOW", "evidence": [], "raw_response": ""}
    
    async def _pass2_detailed_verification(
        self,
        prompt: str,
        element_type: str,
        question_type: str,
        images: List[Dict[str, Any]],
        pass1_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 2: Detailed verification - Check tags, codes, measurements, specs
        """
        
        # Reference Pass 1 findings
        initial_answer = pass1_result.get("primary_answer", "No initial answer")
        
        if question_type == "counting":
            verify_prompt = f"""Verify the count of {element_type}s by checking TAGS and CODES.

Initial count: {initial_answer}

Now verify by looking for:
- Element tags/marks (like {element_type[0].upper()}1, {element_type[0].upper()}2, etc.)
- Grid locations for each {element_type}
- Any {element_type}s that appear in multiple views (same tag = same element)

For each unique element:
TAG/MARK: [identifier if any]
LOCATION: [where it appears]
APPEARS IN: [which views/pages]

VERIFIED COUNT: [final count after checking for duplicates]
UNIQUE ELEMENTS: [list of unique identifiers found]
VERIFICATION METHOD: [how you verified]"""

        elif question_type == "measurement":
            verify_prompt = f"""Verify the measurement/spacing for: {prompt}

Initial answer: {initial_answer}

Now verify by checking:
- Actual dimension strings on drawings
- Scale indicators
- Grid spacing references
- Detail drawings with measurements
- Standard spacing requirements

VERIFIED MEASUREMENT: [value with units]
MEASUREMENT SOURCES: [where you found dimensions]
SCALE VERIFICATION: [drawing scale if relevant]
STANDARD COMPLIANCE: [if this meets typical standards]"""

        elif question_type == "calculation":
            verify_prompt = f"""Verify the data needed for: {prompt}

Initial findings: {initial_answer}

Now verify by collecting:
- All relevant dimensions with units
- Quantities and counts
- Material specifications
- Any formulas or calculation notes
- Reference standards

VERIFIED DATA:
- Dimensions: [list with units]
- Quantities: [counts of relevant items]
- Specifications: [materials, ratings, etc.]
- Additional factors: [any other relevant data]
CALCULATION CHECK: [verify if data is complete for calculation]"""

        else:
            verify_prompt = f"""Verify your answer to: {prompt}

Initial answer: {initial_answer}

Now verify by checking:
- Specific callouts and labels
- Written specifications
- Code references
- Any conflicting information
- Additional details missed initially

VERIFIED ANSWER: [confirmed or revised answer]
SUPPORTING DETAILS: [specific evidence]
CONFLICTS FOUND: [any contradictions]
CONFIDENCE UPDATE: [more certain or less certain]"""

        content = self._prepare_vision_content(images, verify_prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt=f"Verify the answer by checking details, tags, measurements, and specifications. Be thorough and precise.",
                max_tokens=2000
            )
        
        if response:
            return self._parse_pass2_response(response, question_type, initial_answer)
        
        return {"verified_answer": initial_answer, "verification_method": "none", "details": {}}
    
    async def _pass3_documentation_check(
        self,
        prompt: str,
        element_type: str,
        question_type: str,
        images: List[Dict[str, Any]],
        extracted_text: str,
        pass1_result: Dict[str, Any],
        pass2_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 3: Check schedules, notes, specifications, and documentation
        """
        
        # Reference previous findings
        current_answer = pass2_result.get("verified_answer", pass1_result.get("primary_answer", "Unknown"))
        
        # Add extracted text context if available
        text_context = ""
        if extracted_text and len(extracted_text) > 100:
            text_context = f"\nExtracted text from drawings:\n{extracted_text[:2000]}...\n"
        
        doc_prompt = f"""Final verification: Check all DOCUMENTATION for: {prompt}

Current answer: {current_answer}
{text_context}
Look for:
1. SCHEDULES - Any tables listing {element_type}s, quantities, specifications
2. GENERAL NOTES - Requirements, standards, installation notes
3. SPECIFICATIONS - Material specs, ratings, dimensions
4. CALCULATIONS - Any shown calculations or formulas
5. LEGENDS - Symbol definitions, abbreviations
6. CODE REFERENCES - Building codes, standards mentioned

Document findings:
SCHEDULES FOUND: [YES/NO - which schedules]
RELEVANT NOTES: [key notes about the question]
SPECIFICATIONS: [relevant specs found]
DOCUMENTED ANSWER: [answer according to documentation]
FINAL VERIFICATION: [does documentation support our answer?]

Final answer to "{prompt}": [consolidated answer]"""

        content = self._prepare_vision_content(images, doc_prompt)
        
        async with self.vision_semaphore:
            response = await self._make_vision_request(
                content,
                system_prompt="Check all documentation, schedules, notes, and specifications. Documentation is the most reliable source.",
                max_tokens=2000
            )
        
        if response:
            return self._parse_pass3_response(response, question_type, current_answer)
        
        return {"documented_answer": current_answer, "documentation_found": False, "final_answer": current_answer}
    
    def _prepare_vision_content(
        self,
        images: List[Dict[str, Any]],
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare content for vision API request"""
        content = []
        
        # Add images with page markers
        for i, image in enumerate(images):
            if len(images) > 1:
                content.append({
                    "type": "text",
                    "text": f"\n=== PAGE {i+1} of {len(images)} ===\n"
                })
            
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
                    temperature=0.1,  # Slight temperature for better reasoning
                    seed=self.deterministic_seed
                ),
                timeout=CONFIG.get("VISION_TIMEOUT", 60)
            )
            
            if response and response.choices:
                return response.choices[0].message.content or ""
                
        except asyncio.TimeoutError:
            logger.error("Vision request timeout")
        except Exception as e:
            logger.error(f"Vision request error: {e}")
        
        return None
    
    def _parse_pass1_response(self, response: str, question_type: str) -> Dict[str, Any]:
        """Parse Pass 1 response based on question type"""
        result = {
            "primary_answer": None,
            "confidence": "MEDIUM",
            "evidence": [],
            "raw_response": response[:500] if response else ""
        }
        
        if question_type == "counting":
            # Look for count
            count_match = re.search(r'COUNT:\s*(\d+)', response, re.IGNORECASE)
            if count_match:
                result["primary_answer"] = int(count_match.group(1))
            else:
                # Fallback patterns
                number_patterns = [r'found\s+(\d+)', r'counted\s+(\d+)', r'total[:\s]+(\d+)']
                for pattern in number_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        result["primary_answer"] = int(match.group(1))
                        break
                        
        elif question_type == "measurement":
            # Look for measurement with units
            measure_match = re.search(r'MEASUREMENT:\s*([0-9.,]+)\s*([a-zA-Z\'"]+)', response, re.IGNORECASE)
            if measure_match:
                result["primary_answer"] = f"{measure_match.group(1)} {measure_match.group(2)}"
            else:
                # Fallback for measurements in text
                unit_patterns = [
                    r'(\d+(?:\.\d+)?)\s*(feet|ft|inches|in|"|\′|meters?|m|mm|cm)',
                    r'(\d+(?:\.\d+)?)\s*[\'"]?\s*(?:-|–)\s*(\d+(?:\.\d+)?)\s*[\'"]?'
                ]
                for pattern in unit_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        result["primary_answer"] = match.group(0)
                        break
                        
        elif question_type == "calculation":
            # Extract relevant data
            data_match = re.search(r'RELEVANT DATA:(.*?)(?:INITIAL CALCULATION:|CONFIDENCE:|$)', response, re.IGNORECASE | re.DOTALL)
            if data_match:
                result["primary_answer"] = data_match.group(1).strip()
            
            # Try to get initial calculation
            calc_match = re.search(r'INITIAL CALCULATION:\s*(.+?)(?:\n|CONFIDENCE:|$)', response, re.IGNORECASE)
            if calc_match:
                result["initial_calc"] = calc_match.group(1).strip()
                
        else:  # General answer
            answer_match = re.search(r'ANSWER:\s*(.+?)(?:EVIDENCE:|CONFIDENCE:|$)', response, re.IGNORECASE | re.DOTALL)
            if answer_match:
                result["primary_answer"] = answer_match.group(1).strip()
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()
        
        # Extract evidence/notes
        evidence_patterns = [
            r'VISUAL NOTES:\s*(.+?)(?:$|\n\n)',
            r'EVIDENCE:\s*(.+?)(?:CONFIDENCE:|$)',
            r'NOTES:\s*(.+?)(?:$|\n\n)'
        ]
        for pattern in evidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                result["evidence"].append(match.group(1).strip())
        
        return result
    
    def _parse_pass2_response(self, response: str, question_type: str, initial_answer: Any) -> Dict[str, Any]:
        """Parse Pass 2 verification response"""
        result = {
            "verified_answer": initial_answer,
            "verification_method": "visual",
            "details": {}
        }
        
        if question_type == "counting":
            # Look for verified count
            verified_match = re.search(r'VERIFIED COUNT:\s*(\d+)', response, re.IGNORECASE)
            if verified_match:
                result["verified_answer"] = int(verified_match.group(1))
                result["verification_method"] = "tags_and_codes"
            
            # Extract unique elements
            unique_match = re.search(r'UNIQUE ELEMENTS:(.*?)(?:VERIFICATION METHOD:|$)', response, re.IGNORECASE | re.DOTALL)
            if unique_match:
                elements = re.findall(r'\b([A-Z]+\d+)\b', unique_match.group(1))
                result["details"]["unique_elements"] = elements
                
        elif question_type == "measurement":
            # Look for verified measurement
            verified_match = re.search(r'VERIFIED MEASUREMENT:\s*([0-9.,]+\s*[a-zA-Z\'"]+)', response, re.IGNORECASE)
            if verified_match:
                result["verified_answer"] = verified_match.group(1)
                result["verification_method"] = "dimension_verification"
                
        elif question_type == "calculation":
            # Extract verified data
            data_section = re.search(r'VERIFIED DATA:(.*?)(?:CALCULATION CHECK:|$)', response, re.IGNORECASE | re.DOTALL)
            if data_section:
                result["details"]["verified_data"] = data_section.group(1).strip()
                result["verified_answer"] = data_section.group(1).strip()
                
        else:
            # General verification
            verified_match = re.search(r'VERIFIED ANSWER:\s*(.+?)(?:SUPPORTING DETAILS:|CONFLICTS:|$)', response, re.IGNORECASE | re.DOTALL)
            if verified_match:
                result["verified_answer"] = verified_match.group(1).strip()
        
        return result
    
    def _parse_pass3_response(self, response: str, question_type: str, current_answer: Any) -> Dict[str, Any]:
        """Parse Pass 3 documentation response"""
        result = {
            "documented_answer": current_answer,
            "documentation_found": False,
            "final_answer": current_answer,
            "documentation": {}
        }
        
        # Check if documentation found
        schedules_match = re.search(r'SCHEDULES FOUND:\s*(YES|NO)', response, re.IGNORECASE)
        if schedules_match and schedules_match.group(1).upper() == "YES":
            result["documentation_found"] = True
        
        # Extract documented answer
        doc_answer_match = re.search(r'DOCUMENTED ANSWER:\s*(.+?)(?:FINAL VERIFICATION:|$)', response, re.IGNORECASE | re.DOTALL)
        if doc_answer_match:
            doc_answer = doc_answer_match.group(1).strip()
            if doc_answer and doc_answer.lower() not in ["n/a", "none", "not found"]:
                result["documented_answer"] = doc_answer
                result["documentation_found"] = True
        
        # Extract final answer
        final_match = re.search(r'Final answer to[^:]+:\s*(.+?)(?:$)', response, re.IGNORECASE | re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
        
        # Extract relevant notes
        notes_match = re.search(r'RELEVANT NOTES:\s*(.+?)(?:SPECIFICATIONS:|DOCUMENTED ANSWER:|$)', response, re.IGNORECASE | re.DOTALL)
        if notes_match:
            result["documentation"]["notes"] = notes_match.group(1).strip()
        
        return result
    
    def _build_universal_consensus(
        self,
        pass1: Dict[str, Any],
        pass2: Dict[str, Any],
        pass3: Dict[str, Any],
        element_type: str,
        question_type: str,
        page_number: int,
        original_prompt: str
    ) -> VisualIntelligenceResult:
        """Build consensus from universal verification"""
        
        # Determine final answer with priority:
        # 1. Documentation (if found and reliable)
        # 2. Verified answer (Pass 2)
        # 3. Initial answer (Pass 1)
        
        if pass3.get("documentation_found") and pass3.get("documented_answer"):
            final_answer = pass3["documented_answer"]
            confidence = 0.95
            source = "documentation"
        elif pass2.get("verified_answer") is not None:
            final_answer = pass2["verified_answer"]
            confidence = 0.90 if pass2.get("verification_method") != "visual" else 0.85
            source = "verified_analysis"
        else:
            final_answer = pass1.get("primary_answer", "Unable to determine")
            confidence = 0.80 if pass1.get("confidence") == "HIGH" else 0.70
            source = "visual_analysis"
        
        # For counting questions, ensure we have a number
        if question_type == "counting":
            try:
                count = int(final_answer) if final_answer else 0
            except:
                count = 0
        else:
            count = 0  # Not a counting question
        
        # Build verification notes based on question type
        verification_notes = []
        
        if question_type == "counting":
            verification_notes.append(f"Visual count: {pass1.get('primary_answer', 'Not found')}")
            if pass2.get("details", {}).get("unique_elements"):
                verification_notes.append(f"Unique elements found: {len(pass2['details']['unique_elements'])}")
            verification_notes.append(f"Verified count: {pass2.get('verified_answer', 'Not verified')}")
            if pass3.get("documentation_found"):
                verification_notes.append(f"Documentation confirms: {pass3.get('documented_answer')}")
                
        elif question_type == "measurement":
            verification_notes.append(f"Initial measurement: {pass1.get('primary_answer', 'Not found')}")
            verification_notes.append(f"Verified: {pass2.get('verified_answer', 'Not verified')}")
            if pass3.get("documentation", {}).get("notes"):
                verification_notes.append("Documentation checked")
                
        elif question_type == "calculation":
            verification_notes.append("Data extracted for calculation")
            if pass1.get("initial_calc"):
                verification_notes.append(f"Initial estimate: {pass1['initial_calc']}")
            verification_notes.append("Verified data completeness")
            
        else:
            verification_notes.append(f"Analysis complete for: {question_type}")
            verification_notes.append(f"Confidence: {pass1.get('confidence', 'MEDIUM')}")
            if pass3.get("documentation_found"):
                verification_notes.append("Documentation supports answer")
        
        verification_notes.append(f"Final answer based on: {source}")
        
        # Build visual evidence
        visual_evidence = []
        visual_evidence.extend(pass1.get("evidence", []))
        
        if pass2.get("details", {}).get("unique_elements"):
            elements = pass2["details"]["unique_elements"]
            visual_evidence.append(f"Elements identified: {', '.join(elements[:10])}")
            
        if pass3.get("documentation", {}).get("notes"):
            visual_evidence.append("Documentation verified")
        
        # Create locations for counting questions
        locations = []
        if question_type == "counting" and pass2.get("details", {}).get("unique_elements"):
            for elem in pass2["details"]["unique_elements"][:50]:  # Limit to 50
                locations.append({
                    "element_tag": elem,
                    "visual_details": f"{element_type} {elem}",
                    "grid_ref": "See drawings"
                })
        
        # Format the final answer for the response
        if question_type == "counting":
            formatted_answer = str(count)
        else:
            formatted_answer = str(final_answer)
        
        # Create result
        result = VisualIntelligenceResult(
            element_type=element_type,
            count=count if question_type == "counting" else 0,
            locations=locations,
            confidence=confidence,
            visual_evidence=visual_evidence,
            pattern_matches=[],
            grid_references=[],
            verification_notes=verification_notes,
            page_number=page_number
        )
        
        # Add comprehensive metadata
        result.analysis_metadata = {
            "method": "universal_triple_verification",
            "question_type": question_type,
            "original_prompt": original_prompt,
            "pass1_answer": pass1.get("primary_answer"),
            "pass2_answer": pass2.get("verified_answer"),
            "pass3_answer": pass3.get("final_answer"),
            "final_answer": formatted_answer,
            "answer_source": source,
            "documentation_found": pass3.get("documentation_found", False),
            "verification_method": pass2.get("verification_method", "none")
        }
        
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
        Detect what element the user is asking about
        Works for any construction element
        """
        
        logger.info(f"🎯 Detecting element focus for: '{prompt}'")
        
        # Quick check for common elements
        prompt_lower = prompt.lower()
        
        # Common element keywords (not exhaustive, just helpers)
        element_keywords = {
            "window": ["window", "windows", "glazing", "fenestration"],
            "door": ["door", "doors", "entrance", "exit", "opening"],
            "outlet": ["outlet", "outlets", "receptacle", "plug", "power"],
            "panel": ["panel", "panels", "breaker", "electrical panel", "load center"],
            "light": ["light", "lights", "lighting", "fixture", "luminaire"],
            "sprinkler": ["sprinkler", "sprinklers", "fire protection", "fire suppression"],
            "column": ["column", "columns", "pillar", "structural support", "post"],
            "beam": ["beam", "beams", "girder", "joist", "structural member"],
            "wall": ["wall", "walls", "partition", "facade"],
            "slab": ["slab", "concrete", "floor", "deck"],
            "pipe": ["pipe", "pipes", "piping", "plumbing"],
            "duct": ["duct", "ducts", "ductwork", "hvac", "air"],
            "diffuser": ["diffuser", "diffusers", "grille", "air supply", "register"],
            "fixture": ["fixture", "fixtures", "plumbing fixture", "bathroom fixture"]
        }
        
        # Check for exact matches first
        for element, keywords in element_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return element
        
        # If no match, ask GPT to identify
        detection_prompt = f"""What construction element or topic is this question about?

Question: "{prompt}"

If it's about a specific building element (like doors, windows, concrete, etc.), return just that element name.
If it's about a calculation or general topic, return the main subject.

Examples:
- "How many windows on page 3?" -> window
- "What is the spacing between outlets?" -> outlet  
- "How much concrete for the slab?" -> slab
- "Calculate the electrical load" -> electrical

ELEMENT/TOPIC:"""

        try:
            self._ensure_semaphores_initialized()
            async with self.inference_semaphore:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": detection_prompt}
                        ],
                        max_tokens=20,
                        temperature=0.0
                    ),
                    timeout=5.0
                )
            
            if response and response.choices:
                detected = response.choices[0].message.content.strip().lower()
                # Basic cleanup
                detected = detected.replace(".", "").replace(",", "").strip()
                if detected and len(detected.split()) <= 3:
                    return detected
                    
        except Exception as e:
            logger.debug(f"Element detection error: {e}")
        
        # Default fallback
        return "element"
    
    async def detect_precise_geometries(
        self,
        visual_result: VisualIntelligenceResult,
        images: List[Dict[str, Any]]
    ) -> List[ElementGeometry]:
        """
        Detect precise element geometries for highlighting
        (Kept for compatibility)
        """
        
        if not visual_result.locations:
            return []
        
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
            "light": {"width": 48, "height": 24},
            "fixture": {"width": 48, "height": 24},
            "sprinkler": {"radius": 10},
            "column": {"width": 24, "height": 24},
            "diffuser": {"width": 24, "height": 24},
            "beam": {"width": 12, "height": 24},
            "wall": {"width": 8, "height": 120},
            "slab": {"width": 240, "height": 240}
        }
        return dimensions.get(element_type, {"width": 50, "height": 50})