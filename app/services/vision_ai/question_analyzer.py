# question_analyzer.py
import re
import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime

# Import CONFIG and patterns
from app.core.config import CONFIG
from .patterns import VISUAL_PATTERNS, ELEMENT_VARIATIONS

from app.models.schemas import (
    QuestionType,
    VisualIntelligenceResult,
    ElementGeometry,
    SemanticHighlight
)

logger = logging.getLogger(__name__)


class QuestionAnalyzer:
    """
    Analyzes user questions to understand intent and extract key information
    Simplified to trust GPT-4V and focus on what matters
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None  # Will be set by core
        
        # Simplified question type keywords
        self.question_keywords = {
            QuestionType.COUNT: [
                'how many', 'count', 'number of', 'total', 'quantity'
            ],
            QuestionType.LOCATION: [
                'where', 'locate', 'find', 'location', 'position'
            ],
            QuestionType.IDENTIFY: [
                'what is', 'what type', 'identify', 'which', 'what kind'
            ],
            QuestionType.SPECIFICATION: [
                'specification', 'spec', 'size', 'dimension', 'model', 'rating'
            ],
            QuestionType.COMPLIANCE: [
                'code', 'compliant', 'compliance', 'ada', 'nfpa', 'ibc', 'nec'
            ],
            QuestionType.DETAILED: [
                'analyze', 'detailed', 'review', 'comprehensive', 'coverage'
            ],
            QuestionType.ESTIMATE: [
                'estimate', 'calculate', 'cost', 'price', 'budget', 'load',
                'area', 'square footage', 'material', 'electrical load'
            ]
        }
        
        # Calculation trigger keywords
        self.calculation_keywords = [
            'calculate', 'calculation', 'compute', 'determine',
            'cost', 'price', 'budget', 'expense',
            'load', 'electrical load', 'power requirement',
            'area', 'square footage', 'sq ft', 'square feet',
            'coverage', 'spacing', 'distance',
            'quantity', 'material', 'how much',
            'estimate', 'estimation', 'approximate'
        ]
        
        # Use imported ELEMENT_VARIATIONS
        self.element_variations = ELEMENT_VARIATIONS
    
    def set_vision_client(self, client):
        """Set the vision client for AI-based analysis"""
        self.vision_client = client
    
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Main method to analyze a question
        Simplified and focused on what matters
        """
        
        logger.info(f"🤔 Analyzing question: '{prompt}'")
        
        prompt_lower = prompt.lower().strip()
        
        # Initialize analysis result
        analysis = {
            "original_prompt": prompt,
            "normalized_prompt": prompt_lower,
            "type": QuestionType.GENERAL,
            "element_focus": "element",
            "highlight_request": False,
            "requested_page": None,
            "drawing_type": None,
            "wants_total": False,
            "specific_area": None,
            "comparison_request": False,
            "temporal_context": None,
            "needs_calculation": False,  # NEW: Flag for calculation needs
            "confidence_in_analysis": 0.9
        }
        
        # Check for highlight request
        if prompt_lower.startswith('highlight'):
            analysis["highlight_request"] = True
            prompt_lower = prompt_lower.replace('highlight', '', 1).strip()
        
        # Detect question type
        analysis["type"] = self._detect_question_type(prompt_lower)
        
        # Check if calculation is needed
        analysis["needs_calculation"] = self._needs_calculation(prompt_lower, analysis["type"])
        
        # Detect element focus - simplified to trust GPT-4V
        analysis["element_focus"] = await self._detect_element_focus_simple(prompt_lower)
        
        # Extract page references
        analysis["requested_page"] = self._extract_page_reference(prompt_lower)
        
        # Check for total count request
        analysis["wants_total"] = self._wants_total_count(prompt_lower)
        
        # Extract specific area references (if any)
        analysis["specific_area"] = self._extract_area_reference(prompt_lower)
        
        logger.info(f"✅ Analysis complete: Type={analysis['type'].value}, "
                   f"Element={analysis['element_focus']}, "
                   f"Calculation={analysis['needs_calculation']}")
        
        return analysis
    
    def _detect_question_type(self, prompt_lower: str) -> QuestionType:
        """Detect the type of question being asked - simplified"""
        
        # Check ESTIMATE first (often includes count keywords)
        if any(keyword in prompt_lower for keyword in self.question_keywords[QuestionType.ESTIMATE]):
            return QuestionType.ESTIMATE
        
        # Check other types
        best_match = QuestionType.GENERAL
        best_score = 0
        
        for question_type, keywords in self.question_keywords.items():
            if question_type == QuestionType.ESTIMATE:
                continue  # Already checked
            
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > best_score:
                best_score = score
                best_match = question_type
        
        # Quick overrides for clear patterns
        if 'how many' in prompt_lower or 'count' in prompt_lower:
            return QuestionType.COUNT
        elif prompt_lower.startswith('where'):
            return QuestionType.LOCATION
        elif prompt_lower.startswith('what is') or prompt_lower.startswith('what type'):
            return QuestionType.IDENTIFY
        
        return best_match
    
    def _needs_calculation(self, prompt_lower: str, question_type: QuestionType) -> bool:
        """
        Determine if calculation engine should be used
        This is CRITICAL for integration
        """
        
        # ESTIMATE questions always need calculations
        if question_type == QuestionType.ESTIMATE:
            return True
        
        # Check for calculation keywords
        if any(keyword in prompt_lower for keyword in self.calculation_keywords):
            return True
        
        # Specific patterns that need calculations
        calculation_patterns = [
            r'how much (?:will|would|does)',
            r'total (?:cost|price|load|area)',
            r'calculate',
            r'what is the (?:cost|price|load|area)',
            r'\d+\s*(?:watts?|kw|amps?|volts?)',  # Electrical units
            r'sq(?:uare)?\s*f(?:ee|oo)?t',        # Square footage
            r'per\s+(?:square|sq|linear)\s+(?:foot|feet|ft)'  # Unit costs
        ]
        
        for pattern in calculation_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        return False
    
    async def _detect_element_focus_simple(self, prompt_lower: str) -> str:
        """
        Simplified element detection - trust GPT-4V
        Don't duplicate what vision_intelligence.py does
        """
        
        # Quick keyword check first
        for element_type in VISUAL_PATTERNS.keys():
            if element_type in prompt_lower:
                return element_type
            # Check plural
            if element_type + 's' in prompt_lower:
                return element_type
        
        # Check variations
        for variation, element_type in self.element_variations.items():
            if variation in prompt_lower:
                return element_type
        
        # If GPT-4V is available, ask it (but keep it simple)
        if self.vision_client:
            try:
                element = await self._ai_detect_element_quick(prompt_lower)
                if element and element != "element":
                    return element
            except Exception as e:
                logger.debug(f"AI element detection skipped: {e}")
        
        return "element"
    
    async def _ai_detect_element_quick(self, prompt_lower: str) -> Optional[str]:
        """Quick AI element detection - one simple question"""
        
        detection_prompt = f"""What construction element is this about?
Question: "{prompt_lower}"
Return ONLY the element type in lowercase (door, outlet, window, etc.) or "element" if unclear.
ELEMENT:"""

        try:
            response = await asyncio.wait_for(
                self.vision_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You understand construction."},
                        {"role": "user", "content": detection_prompt}
                    ],
                    max_tokens=20,
                    temperature=0.0
                ),
                timeout=5.0  # Quick timeout
            )
            
            if response and response.choices:
                detected = response.choices[0].message.content.strip().lower()
                
                # Quick validation
                if detected in VISUAL_PATTERNS:
                    return detected
                if detected in self.element_variations:
                    return self.element_variations[detected]
                if detected and len(detected.split()) <= 2 and detected != "element":
                    return detected
                    
        except Exception as e:
            logger.debug(f"Quick AI detection error: {e}")
        
        return None
    
    def _extract_page_reference(self, prompt_lower: str) -> Optional[int]:
        """Extract specific page number from prompt"""
        
        page_patterns = [
            r'page\s+(\d+)',
            r'on\s+page\s+(\d+)',
            r'sheet\s+(\d+)',
            r'p\.?\s*(\d+)',
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _wants_total_count(self, prompt_lower: str) -> bool:
        """Check if user wants total count across all pages"""
        
        total_indicators = [
            'total', 'all pages', 'whole file', 'entire',
            'document', 'complete set', 'throughout'
        ]
        
        return any(indicator in prompt_lower for indicator in total_indicators)
    
    def _extract_area_reference(self, prompt_lower: str) -> Optional[str]:
        """Extract specific area references (room, grid, floor, etc.)"""
        
        # Floor references (most common)
        floor_patterns = [
            r'(?:on|for|at)\s+(?:the\s+)?(\w+)\s+floor',
            r'floor\s+(\d+)',
            r'floors?\s+(\d+)(?:\s*(?:to|through|-)\s*(\d+))?'
        ]
        
        for pattern in floor_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                if match.lastindex and match.lastindex > 1:  # Range of floors
                    return f"floors_{match.group(1)}_to_{match.group(match.lastindex)}"
                else:
                    return f"floor_{match.group(1)}"
        
        # Room references
        room_match = re.search(r'(?:in|at|for)\s+(?:room|rm\.?)\s*([A-Z0-9]+)', prompt_lower, re.IGNORECASE)
        if room_match:
            return f"room_{room_match.group(1)}"
        
        # Grid references
        grid_match = re.search(r'(?:at|in|near)\s+grid\s*([A-Z]-?\d+)', prompt_lower, re.IGNORECASE)
        if grid_match:
            return f"grid_{grid_match.group(1)}"
        
        return None
    
    def get_element_context(self, element_type: str) -> Dict[str, Any]:
        """
        Get context information for a specific element type
        (Kept for compatibility but simplified)
        """
        
        context = {
            "element_type": element_type,
            "patterns": VISUAL_PATTERNS.get(element_type, VISUAL_PATTERNS["generic_element"]),
            "variations": [],
            "common_questions": [],
            "typical_locations": [],
            "calculation_relevant": False
        }
        
        # Find variations
        for variation, mapped_type in self.element_variations.items():
            if mapped_type == element_type:
                context["variations"].append(variation)
        
        # Check if calculations are typically needed for this element
        calculable_elements = [
            'outlet', 'panel', 'light fixture',  # Electrical load
            'sprinkler', 'diffuser',              # Coverage
            'door', 'window',                     # Cost
            'parking'                             # Area/spacing
        ]
        
        context["calculation_relevant"] = element_type in calculable_elements
        
        # Add common questions
        if element_type == "outlet":
            context["common_questions"] = [
                "How many outlets are there?",
                "Calculate the electrical load",
                "Are outlets spaced to code?"
            ]
            context["typical_locations"] = ["walls", "countertops", "floors"]
        elif element_type == "door":
            context["common_questions"] = [
                "How many doors?",
                "Are door widths ADA compliant?",
                "What is the door schedule?"
            ]
            context["typical_locations"] = ["walls", "openings", "entries"]
        
        return context
    
    def _extract_temporal_context(self, prompt_lower: str) -> Optional[str]:
        """Extract temporal context (simplified)"""
        
        if 'existing' in prompt_lower or 'current' in prompt_lower:
            return 'existing'
        elif 'proposed' in prompt_lower or 'new' in prompt_lower:
            return 'proposed'
        elif 'demo' in prompt_lower or 'remove' in prompt_lower:
            return 'demolition'
        
        return None
    
    def _extract_drawing_type(self, prompt_lower: str) -> Optional[str]:
        """Extract drawing type preference (simplified)"""
        
        drawing_types = {
            'floor plan': ['floor plan', 'plan view'],
            'elevation': ['elevation'],
            'section': ['section'],
            'schedule': ['schedule']
        }
        
        for dtype, keywords in drawing_types.items():
            if any(kw in prompt_lower for kw in keywords):
                return dtype
        
        return None