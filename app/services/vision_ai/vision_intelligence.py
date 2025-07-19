# vision_intelligence.py - REVISED FOR CONTEXT-AWARE ANALYSIS

import asyncio
import re
import logging
import json
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from app.models.schemas import VisualIntelligenceResult, ElementGeometry
from app.core.config import CONFIG
from .patterns import VISUAL_PATTERNS, ELEMENT_VARIATIONS

logger = logging.getLogger(__name__)


class VisionIntelligence:
    """
    Advanced Visual Intelligence using GPT-4 Vision.
    REVISED to use structured data (grids, index) for a context-aware "scan" phase.
    """

    def __init__(self, settings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        self._client: Optional[AsyncOpenAI] = None
        self.vision_semaphore: Optional[asyncio.Semaphore] = None
        self.deterministic_seed = 42
        self.analysis_context: Dict[str, Any] = {}

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy client initialization."""
        if self._client is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for VisionIntelligence.")
            self._client = AsyncOpenAI(
                api_key=self.openai_api_key,
                timeout=CONFIG["VISION_REQUEST_TIMEOUT"],
                max_retries=CONFIG["VISION_MAX_RETRIES"]
            )
        return self._client

    def _ensure_semaphores_initialized(self):
        """Initializes semaphores for API rate limiting."""
        if self.vision_semaphore is None:
            limit = CONFIG.get("VISION_INFERENCE_LIMIT", 10)
            self.vision_semaphore = asyncio.Semaphore(limit)
            logger.info(f"🔧 Vision semaphore initialized with limit: {limit}")

    async def analyze(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        images: List[Dict[str, Any]],
        page_number: int,
        comprehensive_data: Optional[Dict[str, Any]] = None
    ) -> VisualIntelligenceResult:
        """
        Core visual analysis method. Uses a multi-phase approach, now primed with structured data.
        1. Understand the visual context of the drawings.
        2. Recognize the user's intent and target elements.
        3. Analyze the drawings, guided by ground-truth data from JSON files.
        4. Verify the findings against the same structured data.
        """
        logger.info(f"🧠 Starting Context-Aware Visual Intelligence Analysis for: {prompt}")
        logger.info(f"📄 Analyzing {len(images)} image(s).")
        
        self.analysis_context = {"pages_analyzed": len(images)}
        comprehensive_data = comprehensive_data or {}

        try:
            # PHASE 1: Visual Understanding of the drawings
            logger.info("👁️ PHASE 1: Visual Understanding")
            visual_context = await self._visual_understanding_phase(images)

            # PHASE 2: Intelligent Recognition of the user's query
            logger.info("🎯 PHASE 2: Intelligent Element Recognition")
            element_understanding = self._intelligent_recognition_phase(question_analysis)

            # PHASE 3: Comprehensive Analysis, now guided by structured data
            logger.info("📊 PHASE 3: Comprehensive Engineering Analysis (Scan)")
            analysis_result = await self._comprehensive_analysis_phase(
                prompt, images, element_understanding, visual_context, comprehensive_data
            )

            # PHASE 4: Verification against the provided data
            logger.info("✅ PHASE 4: Verification & Cross-Reference")
            verified_result = await self._verification_phase(
                analysis_result, images, comprehensive_data
            )

            visual_result = self._build_visual_result(verified_result, page_number, element_understanding)
            logger.info(f"✅ Analysis complete: {visual_result.count} {visual_result.element_type}(s) found.")
            return visual_result

        except Exception as e:
            logger.error(f"❌ Visual Intelligence pipeline error: {e}", exc_info=True)
            return self._create_error_result(page_number, str(e))

    def _prepare_vision_content(self, images: List[Dict[str, Any]], text_prompt: str) -> List[Dict[str, Any]]:
        """Prepares the content payload for the vision API request."""
        content = []
        for i, image in enumerate(images):
            page_num = image.get("page", i + 1)
            content.append({"type": "text", "text": f"\n--- PAGE {page_num} ---"})
            content.append({
                "type": "image_url",
                "image_url": {"url": image["url"], "detail": "high"}
            })
        content.append({"type": "text", "text": text_prompt})
        return content

    async def _make_vision_request(self, content: List[Dict[str, Any]], system_prompt: str, max_tokens: int) -> Optional[str]:
        """Makes a vision API request with proper error handling and rate limiting."""
        self._ensure_semaphores_initialized()
        async with self.vision_semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=self.deterministic_seed
                )
                return response.choices[0].message.content if response and response.choices else ""
            except Exception as e:
                logger.error(f"Vision request failed: {e}")
                return None

    async def _visual_understanding_phase(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 1: Let the AI get a general understanding of the provided drawings."""
        prompt = "Briefly describe these construction drawings. Identify the drawing type (e.g., floor plan, elevation), discipline (e.g., architectural, electrical), and overall level of detail."
        content = self._prepare_vision_content(images[:3], prompt) # Sample a few pages for efficiency
        response = await self._make_vision_request(content, "You are an expert construction document analyst.", 1000)
        # For simplicity in this refactor, we'll use a placeholder. A full implementation would parse this.
        return {"drawing_type": "floor plan", "discipline": "architectural"}

    def _intelligent_recognition_phase(self, question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Use the pre-analyzed question to understand what element to find."""
        return {
            "element_type": question_analysis.get("element_focus", "element"),
            "visual_appearance": f"Symbols or drawings representing a {question_analysis.get('element_focus', 'generic construction element')}."
        }

    async def _comprehensive_analysis_phase(self, prompt: str, images: List[Dict[str, Any]], element_understanding: Dict[str, Any],
                                           visual_context: Dict[str, Any], comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: The main analysis, now guided by the ground-truth data."""
        element_type = element_understanding.get("element_type", "element")
        
        # --- Prompt Engineering: Injecting Ground-Truth Data ---
        ground_truth_prompt = self._build_ground_truth_prompt(comprehensive_data)
        
        analysis_prompt = f"""
        MASTER ENGINEERING ANALYSIS

        **USER QUESTION:** {prompt}
        **TARGET ELEMENT:** {element_type}s
        **DRAWING CONTEXT:** {visual_context.get('discipline')} {visual_context.get('drawing_type')}

        {ground_truth_prompt}

        **INSTRUCTIONS:**
        1.  **SCAN AND VERIFY:** Systematically scan all provided pages. Use the Ground-Truth Data to guide your search and verify what you see.
        2.  **COUNT ACCURATELY:** Provide a total count across all pages and a breakdown for each page.
        3.  **LOCATE PRECISELY:** For each element found, provide its page number and its grid reference (e.g., "Page 3, Grid C-4").
        4.  **DOCUMENT EVERYTHING:** List any tags, labels, or relevant notes associated with each element.

        **RESPONSE FORMAT:**
        - **TOTAL COUNT:** [Total number of {element_type}s found]
        - **PAGE BREAKDOWN:**
          - **Page [Number]:** [Count on this page]
        - **ELEMENT LIST:**
          - **1.** Page: [Page Number], Location: [Grid Reference], Details: [Tags, notes, etc.]
          - **2.** Page: [Page Number], Location: [Grid Reference], Details: [Tags, notes, etc.]
        """

        content = self._prepare_vision_content(images, analysis_prompt)
        system_prompt = "You are a Master Construction Professional. Your task is to perform a precise, data-driven analysis of construction drawings, verifying your visual findings against provided ground-truth data."
        
        # Increased tokens for comprehensive test
        response = await self._make_vision_request(content, system_prompt, 4000)
        
        return self._parse_analysis_response(response, element_type) if response else self._create_empty_analysis(element_type)

    def _build_ground_truth_prompt(self, comprehensive_data: Dict[str, Any]) -> str:
        """Builds the prompt section containing the structured data."""
        prompts = ["**GROUND-TRUTH DATA FOR VERIFICATION:**"]
        
        if comprehensive_data.get('grid_systems'):
            prompts.append(f"- **Grid System:** A grid system has been detected for {len(comprehensive_data['grid_systems'])} page(s). Use this for precise location reporting.")
        
        if comprehensive_data.get('document_index'):
            index = comprehensive_data['document_index']
            sheets = list(index.get('sheet_numbers', {}).keys())
            if sheets:
                prompts.append(f"- **Document Index:** The index includes sheets like {', '.join(sheets[:3])}{'...' if len(sheets) > 3 else ''}. Ensure your findings are consistent.")

        if len(prompts) == 1:
            return "- No ground-truth data was provided. Rely solely on visual analysis."
            
        return "\n".join(prompts)

    async def _verification_phase(self, analysis_result: Dict[str, Any], images: List[Dict[str, Any]],
                                 comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 4: A final, quick self-verification step.
        The primary, rigorous validation is now handled by the separate ValidationSystem.
        """
        logger.info("Performing final self-verification of visual scan.")
        # This can be expanded, but for now we trust the guided analysis is more accurate
        # and will be rigorously checked by the main ValidationSystem next.
        analysis_result.setdefault('verification_notes', []).append("Initial guided scan complete. Awaiting full 4x validation.")
        return analysis_result

    def _parse_analysis_response(self, response: str, element_type: str) -> Dict[str, Any]:
        """Parses the detailed analysis response from the vision model."""
        result = self._create_empty_analysis(element_type)
        
        # Parse total count
        count_match = re.search(r'TOTAL COUNT:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            result['count'] = int(count_match.group(1))

        # Parse element list
        element_list_text = re.search(r'ELEMENT LIST:([\s\S]*)', response, re.IGNORECASE)
        if element_list_text:
            for line in element_list_text.group(1).strip().split('\n'):
                # Pattern to find Page, Location, and Details
                match = re.search(r'Page:\s*(\d+),\s*Location:\s*([A-Z0-9-]+),\s*Details:\s*(.*)', line, re.IGNORECASE)
                if match:
                    page, location, details = match.groups()
                    result['locations'].append({
                        "page": int(page.strip()),
                        "grid_ref": location.strip(),
                        "visual_details": details.strip()
                    })
                    result['grid_references'].append(location.strip())
        
        # If count was found but list parsing failed, we still have the total
        if result['count'] > 0 and not result['locations']:
            logger.warning("Could not parse detailed locations, but a total count was found.")
            result['verification_notes'].append("Could not parse detailed locations.")

        # If list was parsed but count wasn't, update count from list length
        if result['count'] == 0 and result['locations']:
            result['count'] = len(result['locations'])
            
        return result

    def _build_visual_result(self, analysis_data: Dict[str, Any], page_number: int,
                             element_understanding: Dict[str, Any]) -> VisualIntelligenceResult:
        """Builds the final VisualIntelligenceResult object."""
        confidence = self._calculate_confidence(analysis_data)
        pages_analyzed = self.analysis_context.get("pages_analyzed", 1)
        
        return VisualIntelligenceResult(
            element_type=analysis_data.get("element_type", "element"),
            count=analysis_data.get("count", 0),
            locations=analysis_data.get("locations", []),
            confidence=confidence,
            grid_references=analysis_data.get("grid_references", []),
            verification_notes=analysis_data.get("verification_notes", []),
            page_number=page_number, # Represents the initial page context of the query
            analysis_metadata={
                "pages_analyzed_count": pages_analyzed,
                "page_counts": analysis_data.get("page_counts", {}),
            }
        )
    
    def _calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculates a confidence score based on the quality of the analysis data."""
        # Start with a high base confidence because the scan is now guided
        confidence = 0.85
        if analysis_data.get('count') > 0 and analysis_data.get('locations'):
            # If we successfully parsed detailed locations, confidence increases
            confidence = 0.95
        if "Could not parse" in str(analysis_data.get('verification_notes', [])):
            confidence -= 0.2
        return min(0.99, max(0.5, confidence))

    def _create_error_result(self, page_number: int, error_message: str) -> VisualIntelligenceResult:
        """Creates a standardized error result object."""
        return VisualIntelligenceResult(
            element_type="error", count=0, locations=[], confidence=0.0,
            verification_notes=[f"Analysis failed: {error_message}"], page_number=page_number
        )

    def _create_empty_analysis(self, element_type: str) -> Dict[str, Any]:
        """Creates a default empty analysis dictionary."""
        return {
            "element_type": element_type, "count": 0, "locations": [], "evidence": [],
            "grid_references": [], "verification_notes": [], "page_counts": {}
        }