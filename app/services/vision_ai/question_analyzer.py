# question_analyzer.py
import re
import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from datetime import datetime

# Import CONFIG from config and patterns
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
    
    CORE PHILOSOPHY:
    - Identify queries that need comprehensive multi-page analysis
    - Detect specific location references (floors, rooms, areas)
    - Understand when to analyze ALL pages vs specific pages
    - Support unlimited page analysis for accuracy
    - ENHANCED: Full support for calculation queries matching calculation_engine.py
    
    NON-BREAKING: All original methods preserved with original signatures
    Enhanced functionality available through new methods
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None  # Will be set by core
        
        # Question type keywords - ENHANCED FOR CALCULATIONS
        self.question_keywords = {
            QuestionType.COUNT: [
                'how many', 'count', 'number of', 'total', 'quantity',
                'how much', 'enumerate', 'tally', 'sum of', 'amount',
                'are there', 'do you see', 'can you count', 'list all',
                'give me all', 'show me all', 'find all'
            ],
            QuestionType.LOCATION: [
                'where', 'locate', 'find', 'show', 'location', 'position',
                'where are', 'point out', 'mark', 'indicate', 'placement',
                'which grid', 'what grid', 'whereabouts', 'situated',
                'on what page', 'which floor', 'what room'
            ],
            QuestionType.IDENTIFY: [
                'what is', 'what type', 'identify', 'specify', 'which',
                'what kind', 'describe', 'tell me about', 'explain',
                'what are these', 'what am i looking at', 'recognize'
            ],
            QuestionType.SPECIFICATION: [
                'dimension', 'size', 'gauge', 'rating', 'specification',
                'capacity', 'model', 'thickness', 'diameter', 'width',
                'height', 'length', 'weight', 'load', 'pressure', 'spec',
                'measurement', 'details', 'technical', 'properties'
            ],
            QuestionType.COMPLIANCE: [
                'code', 'compliant', 'compliance', 'standard', 'regulation',
                'ada', 'nfpa', 'ibc', 'nec', 'meets', 'violate', 'legal',
                'requirement', 'allowed', 'permitted', 'safety', 'approved'
            ],
            QuestionType.DETAILED: [
                'analyze', 'evaluate', 'design', 'review', 'assess',
                'verify', 'check', 'inspect', 'coordinate', 'detailed',
                'comprehensive', 'thorough', 'deep dive', 'examine',
                'complete analysis', 'full review', 'audit'
            ],
            QuestionType.ESTIMATE: [
            # Primary calculation keywords
                'calculate', 'estimate', 'computation', 'figure out', 'work out',
            # Cost estimation
                'cost', 'price', 'budget', 'dollar',
            # Area/quantity estimation  
                'area', 'square footage', 'material', 'quantity needed',
                'how much will', 'rough estimate', 'approximate'
            ]
        }
        
        # Drawing type patterns
        self.drawing_types = {
            'floor plan': ['floor plan', 'plan view', 'level', 'floor', 'flr'],
            'elevation': ['elevation', 'elev', 'facade', 'exterior view'],
            'section': ['section', 'cross section', 'sect', 'cut'],
            'detail': ['detail', 'enlarged', 'typical', 'dtl'],
            'schedule': ['schedule', 'table', 'legend', 'equipment list'],
            'site plan': ['site plan', 'site', 'civil', 'survey', 'plot plan'],
            'structural': ['structural', 'framing', 'foundation', 'struct'],
            'electrical': ['electrical', 'power', 'lighting', 'elec'],
            'plumbing': ['plumbing', 'piping', 'waste', 'water', 'plmb'],
            'mechanical': ['mechanical', 'hvac', 'ductwork', 'mech'],
            'fire': ['fire protection', 'fire alarm', 'sprinkler', 'fp']
        }
        
        # Use imported ELEMENT_VARIATIONS
        self.element_variations = ELEMENT_VARIATIONS
        
        # Contextual patterns - ENHANCED
        self.contextual_patterns = {
            "total_count": [
                "entire document", "whole file", "all pages", "total in",
                "throughout", "complete", "full set", "entire set",
                "across all", "in total", "whole building", "entire building",
                "all floors", "every floor", "all levels", "entire project",
                "complete count", "comprehensive count", "full count"
            ],
            "specific_area": [
                "in room", "on floor", "in area", "at grid", "near",
                "adjacent to", "next to", "around", "between",
                "within", "inside", "outside", "above", "below"
            ],
            "comparison": [
                "compare", "versus", "vs", "difference between",
                "more than", "less than", "same as", "match",
                "differ", "contrast"
            ],
            "multi_location": [
                "on floors", "in rooms", "across levels", "multiple areas",
                "various locations", "different floors", "several rooms",
                "floors \\d+ (?:to|through|-) \\d+", "levels \\d+ and \\d+"
            ]
        }
        
        # Scope indicators - NEW
        self.scope_indicators = {
            "comprehensive": [
                "all", "every", "entire", "complete", "whole", "total",
                "throughout", "across", "comprehensive", "full"
            ],
            "specific": [
                "only", "just", "specific", "particular", "single",
                "one", "this", "that"
            ],
            "multiple": [
                "and", "&", "plus", "also", "as well as", "both",
                "multiple", "various", "several", "many"
            ]
        }
        
        # Calculation type patterns - NEW
        self.calculation_patterns = {
            "area": [
                r'\b(?:square|sq\.?)\s*(?:feet|ft|foot|footage)\b',
                r'\b(?:area|coverage|surface)\b',
                r'\bsqft\b', r'\bsf\b'
            ],
            "cost": [
                r'\$\s*\d+', r'\bdollar\b', r'\bcost\b', r'\bprice\b',
                r'\bbudget\b', r'\bexpense\b', r'\bquote\b'
            ],
            "load": [
                r'\b(?:electrical|power|hvac|cooling|heating)\s*load\b',
                r'\bwatts?\b', r'\bamps?\b', r'\bvoltage\b',
                r'\bbtu\b', r'\btonnage\b', r'\bcfm\b'
            ],
            "material": [
                r'\bhow\s+(?:much|many)\s+\w+\s+(?:do|will|needed)\b',
                r'\b(?:material|supplies?|quantity)\s+needed\b',
                r'\bgallons?\b', r'\bsheets?\b', r'\bunits?\b'
            ],
            "spacing": [
                r'\b(?:spacing|distance|apart|between)\b',
                r'\b(?:interval|gap|separation|clearance)\b'
            ],
            "time": [
                r'\b(?:hours?|days?|weeks?|months?)\b',
                r'\b(?:duration|timeline|schedule)\b',
                r'\bhow\s+long\b', r'\blabor\s+hours?\b'
            ],
            "percentage": [
                r'\d+\s*%', r'\bpercent(?:age)?\b',
                r'\bratio\b', r'\bproportion\b'
            ]
        }
    
    def set_vision_client(self, client):
        """Set the vision client for AI-based analysis"""
        self.vision_client = client
    
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Main method to analyze a question
        Returns comprehensive analysis of the user's intent
        
        ENHANCED: Better detection of queries needing multi-page analysis
        Enhanced calculation detection and categorization
        Backward compatible - all original fields preserved
        """
        
        logger.info(f"🤔 Analyzing question: '{prompt}'")
        
        prompt_lower = prompt.lower().strip()
        
        # Initialize analysis result - ALL ORIGINAL FIELDS PRESERVED
        analysis = {
            "original_prompt": prompt,
            "normalized_prompt": prompt_lower,
            "type": QuestionType.GENERAL,
            "element_focus": "element",
            "highlight_request": False,
            "requested_page": None,  # Original field preserved
            "drawing_type": None,
            "wants_total": False,
            "specific_area": None,  # Original field preserved
            "comparison_request": False,
            "temporal_context": None,
            "confidence_in_analysis": 0.9
        }
        
        # Check for highlight request
        if prompt_lower.startswith('highlight'):
            analysis["highlight_request"] = True
            prompt_lower = prompt_lower.replace('highlight', '', 1).strip()
        
        # Detect question type - ENHANCED FOR CALCULATIONS
        analysis["type"] = self._detect_question_type(prompt_lower)
        
        # Detect element focus (AI-first with fallback)
        analysis["element_focus"] = await self._detect_element_focus(prompt_lower)
        
        # Extract page references - USING ORIGINAL METHOD
        analysis["requested_page"] = self._extract_page_reference(prompt_lower)
        
        # Detect drawing type preference
        analysis["drawing_type"] = self._detect_drawing_type(prompt_lower)
        
        # Check for total count request - USING ORIGINAL METHOD
        analysis["wants_total"] = self._wants_total_count(prompt_lower)
        
        # Extract specific area references - USING ORIGINAL METHOD
        analysis["specific_area"] = self._extract_area_reference(prompt_lower)
        
        # Check for comparison requests
        analysis["comparison_request"] = self._is_comparison(prompt_lower)
        
        # Extract temporal context
        analysis["temporal_context"] = self._extract_temporal_context(prompt_lower)
        
        # NEW: Add enhanced analysis using new methods
        analysis = self._add_enhanced_analysis(analysis, prompt_lower)
        
        # Validate and adjust analysis
        analysis = self._validate_analysis(analysis)
        
        logger.info(f"✅ Question analysis complete: Type={analysis['type'].value}, "
                   f"Element={analysis['element_focus']}, Scope={analysis.get('scope', 'specific')}, "
                   f"Pages={analysis.get('requested_pages', []) or analysis['requested_page'] or 'auto'}")
        
        return analysis
    
    def _detect_question_type(self, prompt_lower: str) -> QuestionType:
        """Detect the type of question being asked"""
        
        # Check each question type's keywords
        best_match = QuestionType.GENERAL
        best_score = 0
        
        for question_type, keywords in self.question_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > best_score:
                best_score = score
                best_match = question_type
        
        # Special cases and overrides
        if best_match == QuestionType.GENERAL:
            # Check for implicit patterns
            if any(word in prompt_lower for word in ['list', 'show me all', 'give me']):
                if 'where' in prompt_lower:
                    best_match = QuestionType.LOCATION
                else:
                    best_match = QuestionType.COUNT
            elif '?' in prompt_lower and any(word in prompt_lower for word in ['is', 'are']):
                best_match = QuestionType.IDENTIFY
        
        # Override for certain patterns
        if 'how many' in prompt_lower or 'count' in prompt_lower:
            # Check if it's actually a calculation
            if any(calc_word in prompt_lower for calc_word in ['cost', 'need', 'gallons', 'sheets']):
                best_match = QuestionType.ESTIMATE
            else:
                best_match = QuestionType.COUNT
        elif 'where' in prompt_lower and best_match != QuestionType.COUNT:
            best_match = QuestionType.LOCATION
        elif 'calculate' in prompt_lower or 'estimate' in prompt_lower:
            best_match = QuestionType.ESTIMATE
        
        return best_match
    
    def _is_calculation_question(self, prompt_lower: str) -> bool:
        """Detect if this is a calculation question - NEW METHOD"""
        
        # Check for explicit calculation keywords
        calculation_indicators = [
            'calculate', 'compute', 'figure out', 'work out',
            'how much paint', 'how many gallons', 'how much material',
            'what is the cost', 'price', 'estimate',
            'square footage', 'sq ft', 'area',
            'electrical load', 'hvac load', 'cooling load',
            'spacing', 'distance between',
            'labor hours', 'man hours', 'duration',
            'percentage', 'ratio', 'proportion'
        ]
        
        for indicator in calculation_indicators:
            if indicator in prompt_lower:
                return True
        
        # Check for calculation patterns
        for calc_type, patterns in self.calculation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    return True
        
        # Check for unit mentions that suggest calculations
        unit_patterns = [
            r'\b\d+\s*(?:sq|square)\s*(?:ft|feet|foot)\b',
            r'\b\d+\s*(?:gallons?|gal)\b',
            r'\b\d+\s*(?:watts?|w)\b',
            r'\b\d+\s*(?:amps?|a)\b',
            r'\b\d+\s*(?:tons?)\b',
            r'\b\d+\s*(?:hours?|hrs?)\b',
            r'\b\d+\s*%'
        ]
        
        for pattern in unit_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        return False
    
    async def _detect_element_focus(self, prompt_lower: str) -> str:
        """
        Detect what element the user is asking about
        Uses AI first, then falls back to keyword matching
        """
        
        # Try AI detection first
        if self.vision_client and CONFIG.get("element_detection_confidence", 0.7) > 0:
            try:
                ai_element = await self._ai_detect_element(prompt_lower)
                if ai_element and ai_element != "element":
                    logger.info(f"🧠 AI detected element: '{ai_element}'")
                    return ai_element
            except Exception as e:
                logger.debug(f"AI element detection failed, using fallback: {e}")
        
        # Fallback to keyword detection
        return self._keyword_detect_element(prompt_lower)
    
    async def _ai_detect_element(self, prompt_lower: str) -> Optional[str]:
        """Use AI to understand what element type the user is asking about"""
        
        detection_prompt = f"""Analyze this construction/blueprint question and identify the PRIMARY element type being asked about.

QUESTION: {prompt_lower}

Consider:
- Complex phrases and technical jargon
- Abbreviations and shortcuts (e.g., "receps" for receptacles, "AHU" for air handling unit)
- Context clues about what they're looking for
- Industry-specific terminology
- Plural forms and variations
- If it's a calculation question, identify the element being calculated about

If the user is asking about multiple elements, identify the PRIMARY one.
If it's a general question not about a specific element, return "element".

Common construction elements include:
outlet, door, window, light, fixture, panel, breaker, sprinkler, column, beam, 
pipe, duct, equipment, diffuser, vav, pump, valve, sensor, detector, etc.

Return ONLY the element type in lowercase singular form (e.g., door not doors, outlet not outlets)

ELEMENT TYPE:"""

        try:
            response = await asyncio.wait_for(
                self.vision_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at understanding construction terminology and identifying element types from questions. You understand abbreviations, slang, and technical jargon."
                        },
                        {"role": "user", "content": detection_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.0
                ),
                timeout=CONFIG.get("element_detection_timeout", 10.0)
            )
            
            if response and response.choices:
                detected = response.choices[0].message.content.strip().lower()
                
                # Remove plural 's' if present
                if detected.endswith('s') and detected[:-1] in VISUAL_PATTERNS:
                    detected = detected[:-1]
                
                # Validate against known elements
                if detected in VISUAL_PATTERNS:
                    return detected
                
                # Check if it's a variation we know
                if detected in self.element_variations:
                    return self.element_variations[detected]
                
                # Check partial matches
                for known_element in VISUAL_PATTERNS.keys():
                    if known_element in detected or detected in known_element:
                        return known_element
                
                # If it seems valid but unknown, return it
                if detected and detected != "unknown" and detected != "element" and len(detected.split()) <= 2:
                    logger.info(f"🆕 New element type detected: '{detected}'")
                    return detected
                    
        except Exception as e:
            logger.debug(f"AI element detection error: {e}")
        
        return None
    
    def _keyword_detect_element(self, prompt_lower: str) -> str:
        """Keyword-based element detection"""
        
        # First check for exact matches
        for element_type in VISUAL_PATTERNS.keys():
            if element_type in prompt_lower:
                return element_type
            # Check plural
            if element_type + 's' in prompt_lower:
                return element_type
            # Check without spaces
            if element_type.replace(' ', '') in prompt_lower.replace(' ', ''):
                return element_type
        
        # Check variations and synonyms
        for variation, element_type in self.element_variations.items():
            if variation in prompt_lower:
                return element_type
        
        # Check for partial matches
        words = prompt_lower.split()
        for word in words:
            for element_type in VISUAL_PATTERNS.keys():
                if word in element_type or element_type in word:
                    return element_type
        
        return "element"
    
    def _extract_page_reference(self, prompt_lower: str) -> Optional[int]:
        """
        ORIGINAL METHOD - Extract specific page number from prompt
        Returns single page number for backward compatibility
        """
        
        # Page reference patterns
        page_patterns = [
            r'page\s+(\d+)',
            r'on\s+page\s+(\d+)',
            r'sheet\s+(\d+)',
            r'drawing\s+(\d+)',
            r'p\.?\s*(\d+)',
            r'pg\.?\s*(\d+)',
            r'page\s+number\s+(\d+)',
            r'look\s+at\s+page\s+(\d+)',
            r'check\s+page\s+(\d+)',
            r'see\s+page\s+(\d+)'
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                page_num = int(match.group(1))
                logger.info(f"📍 Detected page reference: {page_num}")
                return page_num
        
        return None
    
    def _extract_page_references_enhanced(self, prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Extract page references with support for multiple pages and ranges
        """
        
        result = {
            "single": None,
            "multiple": [],
            "ranges": []
        }
        
        # Get single page using original method
        result["single"] = self._extract_page_reference(prompt_lower)
        
        # Page reference patterns
        all_patterns = [
            (r'pages?\s+(\d+)\s*(?:to|through|-)\s*(\d+)', 'range'),
            (r'pages?\s+(\d+)\s*(?:and|&)\s*(\d+)', 'multiple'),
            (r'page\s+(\d+)', 'single'),
            (r'on\s+page\s+(\d+)', 'single'),
            (r'sheet\s+(\d+)', 'single'),
            (r'drawing\s+(\d+)', 'single'),
            (r'p\.?\s*(\d+)', 'single'),
            (r'pg\.?\s*(\d+)', 'single'),
            (r'pages?\s+(\d+(?:\s*,\s*\d+)+)', 'list'),
        ]
        
        all_pages = set()
        
        for pattern, ptype in all_patterns:
            matches = re.finditer(pattern, prompt_lower)
            for match in matches:
                if ptype == 'single':
                    page_num = int(match.group(1))
                    all_pages.add(page_num)
                elif ptype == 'range':
                    start = int(match.group(1))
                    end = int(match.group(2))
                    result["ranges"].append((start, end))
                    all_pages.update(range(start, end + 1))
                elif ptype == 'multiple':
                    page1 = int(match.group(1))
                    page2 = int(match.group(2))
                    all_pages.add(page1)
                    all_pages.add(page2)
                elif ptype == 'list':
                    page_list = match.group(1)
                    pages = [int(p.strip()) for p in page_list.split(',')]
                    all_pages.update(pages)
        
        result["multiple"] = sorted(list(all_pages))
        
        return result
    
    def _detect_drawing_type(self, prompt_lower: str) -> Optional[str]:
        """Detect if user is asking about a specific drawing type"""
        
        for dtype, terms in self.drawing_types.items():
            if any(term in prompt_lower for term in terms):
                return dtype
        
        return None
    
    def _wants_total_count(self, prompt_lower: str) -> bool:
        """
        ORIGINAL METHOD - Check if user wants total count across all pages
        Returns boolean for backward compatibility
        """
        
        total_indicators = [
            'total', 'all pages', 'whole file', 'entire',
            'document', 'complete set', 'throughout',
            'across all', 'in total', 'sum', 'aggregate'
        ]
        
        return any(indicator in prompt_lower for indicator in total_indicators)
    
    def _analyze_count_scope_enhanced(self, prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Enhanced detection of total/comprehensive count requests
        """
        
        result = {
            "wants_total": False,
            "wants_comprehensive": False,
            "confidence": 0.0
        }
        
        # Direct total indicators
        total_indicators = [
            'total', 'all pages', 'whole file', 'entire',
            'document', 'complete set', 'throughout',
            'across all', 'in total', 'sum', 'aggregate',
            'whole building', 'entire building', 'all floors',
            'every floor', 'all levels', 'entire project',
            'complete count', 'comprehensive', 'full count'
        ]
        
        # Check for total indicators
        total_score = sum(1 for indicator in total_indicators if indicator in prompt_lower)
        
        # Regex patterns for total count
        total_patterns = [
            r'how many.*?(?:total|all|entire)',
            r'total.*?(?:number|count|quantity)',
            r'count.*?(?:all|entire|whole)',
            r'(?:all|every|entire).*?(?:floor|level|page|room)',
            r'across\s+(?:all|the\s+entire)',
            r'throughout\s+(?:the\s+)?(?:building|document|project)'
        ]
        
        for pattern in total_patterns:
            if re.search(pattern, prompt_lower):
                total_score += 2
        
        # Comprehensive analysis indicators
        comprehensive_patterns = [
            'comprehensive', 'complete analysis', 'full review',
            'detailed count', 'thorough count', 'analyze all',
            'review all', 'check all', 'inspect all'
        ]
        
        comprehensive_score = sum(1 for pattern in comprehensive_patterns if pattern in prompt_lower)
        
        # Set results based on scores
        if total_score >= 2:
            result["wants_total"] = True
            result["confidence"] = min(1.0, total_score * 0.3)
        
        if comprehensive_score >= 1 or total_score >= 3:
            result["wants_comprehensive"] = True
            result["confidence"] = max(result["confidence"], 0.8)
        
        # Check for limiting words that negate total count
        limiting_words = ['only', 'just', 'specific', 'particular', 'single']
        if any(word in prompt_lower for word in limiting_words):
            result["wants_total"] = False
            result["wants_comprehensive"] = False
            result["confidence"] *= 0.5
        
        return result
    
    def _extract_area_reference(self, prompt_lower: str) -> Optional[str]:
        """
        ORIGINAL METHOD - Extract specific area references (room, grid, etc.)
        Returns single area reference for backward compatibility
        """
        
        # Room references
        room_match = re.search(r'(?:in|at|for)\s+(?:room|rm\.?)\s*([A-Z0-9]+)', prompt_lower, re.IGNORECASE)
        if room_match:
            return f"room_{room_match.group(1)}"
        
        # Grid references
        grid_match = re.search(r'(?:at|in|near)\s+grid\s*([A-Z]-?\d+)', prompt_lower, re.IGNORECASE)
        if grid_match:
            return f"grid_{grid_match.group(1)}"
        
        # Floor references
        floor_match = re.search(r'(?:on|at)\s+(?:floor|level)\s*(\d+|[A-Z]+)', prompt_lower, re.IGNORECASE)
        if floor_match:
            return f"floor_{floor_match.group(1)}"
        
        # Area descriptions
        for pattern in self.contextual_patterns["specific_area"]:
            if pattern in prompt_lower:
                # Extract the area description
                area_match = re.search(f'{pattern}\\s+([\\w\\s]+?)(?:\\?|$|,)', prompt_lower)
                if area_match:
                    return f"area_{area_match.group(1).strip()}"
        
        return None
    
    def _extract_area_references_enhanced(self, prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Extract all area references with better context
        """
        
        result = {
            "primary": None,
            "all": [],
            "context": {
                "floors": [],
                "rooms": [],
                "grids": [],
                "areas": [],
                "systems": []
            }
        }
        
        # Use original method to get primary
        result["primary"] = self._extract_area_reference(prompt_lower)
        
        # Floor/Level extraction - ENHANCED
        floor_patterns = [
            (r'(?:level|floor|flr\.?)\s*(\d+)', 'single'),
            (r'(\d+)(?:st|nd|rd|th)\s*(?:floor|level)', 'single'),
            (r'(?:floors?|levels?)\s*(\d+)\s*(?:to|through|-)\s*(\d+)', 'range'),
            (r'(?:floors?|levels?)\s*(\d+)\s*(?:and|&)\s*(\d+)', 'multiple'),
            (r'L(\d+)', 'single'),
            (r'(\d+)F\b', 'single'),
            (r'(?:basement|cellar|ground|first|second|third)', 'named'),
            (r'(?:all|every|each)\s+(?:floor|level)', 'all')
        ]
        
        for pattern, ptype in floor_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                if ptype == 'single':
                    floor = f"floor_{match.group(1)}"
                    result["all"].append(floor)
                    result["context"]["floors"].append(int(match.group(1)))
                elif ptype == 'range':
                    start = int(match.group(1))
                    end = int(match.group(2))
                    for f in range(start, end + 1):
                        result["all"].append(f"floor_{f}")
                        result["context"]["floors"].append(f)
                elif ptype == 'multiple':
                    for i in [1, 2]:
                        floor = f"floor_{match.group(i)}"
                        result["all"].append(floor)
                        result["context"]["floors"].append(int(match.group(i)))
                elif ptype == 'named':
                    floor_name = match.group(0).lower()
                    result["all"].append(f"floor_{floor_name}")
                    result["context"]["floors"].append(floor_name)
                elif ptype == 'all':
                    result["all"].append("floor_all")
                    result["context"]["floors"].append("all")
        
        # Room extraction - ENHANCED
        room_patterns = [
            r'(?:in|at|for)\s+(?:room|rm\.?)\s*([A-Z0-9]+)',
            r'(?:room|rm\.?)\s*([A-Z0-9]+)',
            r'(?:all|every)\s+(\w+\s*\w*)\s*(?:room|space)s?',
            r'(\w+\s*\w*)\s+(?:room|space|area)s?',
        ]
        
        # Also check for specific room types
        room_types = [
            'conference', 'meeting', 'office', 'bathroom', 'restroom',
            'kitchen', 'break room', 'storage', 'mechanical', 'electrical',
            'janitor', 'closet', 'lobby', 'corridor', 'stairwell'
        ]
        
        for pattern in room_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                room = f"room_{match.group(1)}"
                result["all"].append(room)
                result["context"]["rooms"].append(match.group(1))
        
        for room_type in room_types:
            if room_type in prompt_lower:
                result["all"].append(f"room_type_{room_type.replace(' ', '_')}")
                result["context"]["rooms"].append(room_type)
        
        # Grid references
        grid_patterns = [
            r'(?:at|in|near)\s+grid\s*([A-Z]-?\d+)',
            r'grid\s*([A-Z]-?\d+)',
            r'(?:between\s+)?grids?\s*([A-Z]-?\d+)\s*(?:and|to)\s*([A-Z]-?\d+)'
        ]
        
        for pattern in grid_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                if match.lastindex == 1:
                    grid = f"grid_{match.group(1)}"
                    result["all"].append(grid)
                    result["context"]["grids"].append(match.group(1))
                else:
                    # Grid range
                    for i in [1, 2]:
                        grid = f"grid_{match.group(i)}"
                        result["all"].append(grid)
                        result["context"]["grids"].append(match.group(i))
        
        # Area descriptions
        area_keywords = [
            'north', 'south', 'east', 'west', 'central', 'perimeter',
            'interior', 'exterior', 'parking', 'garage', 'roof', 'basement'
        ]
        
        for keyword in area_keywords:
            if keyword in prompt_lower:
                result["all"].append(f"area_{keyword}")
                result["context"]["areas"].append(keyword)
        
        # System references
        system_keywords = [
            'emergency', 'backup', 'normal power', 'fire alarm',
            'sprinkler', 'hvac', 'plumbing', 'electrical'
        ]
        
        for keyword in system_keywords:
            if keyword in prompt_lower:
                result["all"].append(f"system_{keyword.replace(' ', '_')}")
                result["context"]["systems"].append(keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_all = []
        for item in result["all"]:
            if item not in seen:
                seen.add(item)
                unique_all.append(item)
        result["all"] = unique_all
        
        return result
    
    def _determine_scope(self, prompt_lower: str, analysis: Dict[str, Any]) -> str:
        """
        NEW METHOD - Determine the scope of the analysis needed
        Returns: 'specific', 'multiple', or 'comprehensive'
        """
        
        # Check for comprehensive indicators
        comprehensive_score = 0
        for indicator in self.scope_indicators["comprehensive"]:
            if indicator in prompt_lower:
                comprehensive_score += 1
        
        # Check for specific indicators
        specific_score = 0
        for indicator in self.scope_indicators["specific"]:
            if indicator in prompt_lower:
                specific_score += 1
        
        # Check for multiple indicators
        multiple_score = 0
        for indicator in self.scope_indicators["multiple"]:
            if indicator in prompt_lower:
                multiple_score += 1
        
        # Additional checks
        if analysis["wants_total"] or analysis.get("wants_comprehensive", False):
            return "comprehensive"
        
        if len(analysis.get("specific_areas", [])) > 1:
            multiple_score += 2
        
        if len(analysis.get("requested_pages", [])) > 1:
            multiple_score += 1
        
        if analysis.get("location_context", {}).get("floors") == ["all"]:
            return "comprehensive"
        
        if len(analysis.get("location_context", {}).get("floors", [])) > 2:
            multiple_score += 2
        
        # Determine scope based on scores
        if comprehensive_score >= 2 or (comprehensive_score > 0 and specific_score == 0):
            return "comprehensive"
        elif multiple_score >= 2 or len(analysis.get("specific_areas", [])) > 1:
            return "multiple"
        else:
            return "specific"
    
    def _is_comparison(self, prompt_lower: str) -> bool:
        """Check if this is a comparison request"""
        
        return any(pattern in prompt_lower for pattern in self.contextual_patterns["comparison"])
    
    def _extract_temporal_context(self, prompt_lower: str) -> Optional[str]:
        """Extract temporal context (current, proposed, existing, etc.)"""
        
        temporal_terms = {
            'existing': 'existing',
            'current': 'existing',
            'as-is': 'existing',
            'proposed': 'proposed',
            'new': 'proposed',
            'future': 'proposed',
            'planned': 'proposed',
            'to be': 'proposed',
            'as-built': 'as-built',
            'as built': 'as-built',
            'demo': 'demolition',
            'demolition': 'demolition',
            'remove': 'demolition',
            'revised': 'revised',
            'updated': 'revised'
        }
        
        for term, context in temporal_terms.items():
            if term in prompt_lower:
                return context
        
        return None
    
    def _add_enhanced_analysis(self, analysis: Dict[str, Any], prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Add enhanced analysis fields without breaking backward compatibility
        """
        
        # Get enhanced page references
        page_refs = self._extract_page_references_enhanced(prompt_lower)
        analysis["requested_pages"] = page_refs["multiple"]
        
        # Get enhanced area references
        area_refs = self._extract_area_references_enhanced(prompt_lower)
        analysis["specific_areas"] = area_refs["all"]
        analysis["location_context"] = area_refs["context"]
        
        # Get enhanced count scope
        count_scope = self._analyze_count_scope_enhanced(prompt_lower)
        analysis["wants_comprehensive"] = count_scope["wants_comprehensive"]
        
        # Determine scope
        analysis["scope"] = self._determine_scope(prompt_lower, analysis)
        
        # Add recommendation
        analysis["recommended_approach"] = self._get_recommended_approach(analysis)
        
        return analysis
    
    def _add_calculation_analysis(self, analysis: Dict[str, Any], prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Add calculation-specific analysis
        """
        
        # Determine calculation type
        calc_type = self._determine_calculation_type(prompt_lower)
        analysis["calculation_type"] = calc_type
        
        # Extract calculation parameters
        calc_params = self._extract_calculation_parameters(prompt_lower)
        analysis["calculation_parameters"] = calc_params
        
        # Determine if we need comprehensive data for calculation
        if calc_type in ["area", "cost", "load", "coverage"]:
            analysis["needs_comprehensive_data"] = True
            analysis["scope"] = "comprehensive"
        
        # Extract any numerical values or units
        analysis["extracted_values"] = self._extract_numerical_values(prompt_lower)
        
        return analysis
    
    def _determine_calculation_type(self, prompt_lower: str) -> str:
        """Determine specific calculation type - NEW METHOD"""
        
        # Priority order matters - check more specific patterns first
        if any(pattern in prompt_lower for pattern in ['cost', 'price', 'budget', '$', 'dollar', 'expense']):
            return "cost"
        
        elif any(pattern in prompt_lower for pattern in ['electrical load', 'power load', 'watts', 'amps', 'voltage']):
            return "electrical_load"
        
        elif any(pattern in prompt_lower for pattern in ['hvac load', 'cooling load', 'heating load', 'tons', 'btu', 'cfm']):
            return "hvac_load"
        
        elif any(pattern in prompt_lower for pattern in ['area', 'square footage', 'sq ft', 'square feet']):
            return "area"
        
        elif any(pattern in prompt_lower for pattern in ['spacing', 'distance', 'apart', 'between', 'separation']):
            return "spacing"
        
        elif any(pattern in prompt_lower for pattern in ['coverage', 'cover', 'protection']):
            return "coverage"
        
        elif any(pattern in prompt_lower for pattern in ['paint', 'drywall', 'material needed', 'quantity needed', 'how much']):
            return "material"
        
        elif any(pattern in prompt_lower for pattern in ['hours', 'days', 'duration', 'schedule', 'timeline']):
            return "time"
        
        elif any(pattern in prompt_lower for pattern in ['percentage', '%', 'ratio', 'proportion']):
            return "percentage"
        
        else:
            return "general"
    
    def _extract_calculation_parameters(self, prompt_lower: str) -> Dict[str, Any]:
        """Extract parameters needed for calculations - NEW METHOD"""
        
        params = {
            "has_constraints": False,
            "constraints": [],
            "assumptions_requested": False,
            "specific_conditions": []
        }
        
        # Check for constraints
        constraint_patterns = [
            (r'(?:max|maximum)\s+(\d+)', 'max'),
            (r'(?:min|minimum)\s+(\d+)', 'min'),
            (r'(?:budget|under|less than)\s*\$?\s*(\d+)', 'budget'),
            (r'(?:within|inside)\s+(\d+)', 'within')
        ]
        
        for pattern, constraint_type in constraint_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                params["has_constraints"] = True
                params["constraints"].append({
                    "type": constraint_type,
                    "value": match.group(1)
                })
        
        # Check for assumption requests
        if any(word in prompt_lower for word in ['assuming', 'assume', 'if', 'based on']):
            params["assumptions_requested"] = True
        
        # Check for specific conditions
        conditions = [
            'residential', 'commercial', 'industrial',
            'new construction', 'renovation', 'retrofit',
            'emergency', 'standard', 'premium',
            'code minimum', 'energy efficient'
        ]
        
        for condition in conditions:
            if condition in prompt_lower:
                params["specific_conditions"].append(condition)
        
        return params
    
    def _extract_numerical_values(self, prompt_lower: str) -> List[Dict[str, Any]]:
        """Extract numerical values and their units - NEW METHOD"""
        
        values = []
        
        # Pattern for number + unit
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:sq|square)\s*(?:ft|feet|foot)', 'area', 'sq ft'),
            (r'(\d+(?:\.\d+)?)\s*(?:gallons?|gal)', 'volume', 'gallons'),
            (r'(\d+(?:\.\d+)?)\s*(?:watts?|w)', 'power', 'watts'),
            (r'(\d+(?:\.\d+)?)\s*(?:amps?|a)\b', 'current', 'amps'),
            (r'(\d+(?:\.\d+)?)\s*(?:volts?|v)\b', 'voltage', 'volts'),
            (r'(\d+(?:\.\d+)?)\s*(?:tons?)\b', 'cooling', 'tons'),
            (r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', 'time', 'hours'),
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage', 'percent'),
            (r'\$\s*(\d+(?:\.\d+)?)', 'currency', 'dollars'),
        ]
        
        for pattern, value_type, unit in patterns:
            matches = re.finditer(pattern, prompt_lower)
            for match in matches:
                values.append({
                    "value": float(match.group(1)),
                    "type": value_type,
                    "unit": unit,
                    "context": prompt_lower[max(0, match.start()-20):match.end()+20]
                })
        
        return values
    
    def _get_recommended_approach(self, analysis: Dict[str, Any]) -> str:
        """
        NEW METHOD - Get recommended approach for page loading
        """
        
        if analysis.get("scope") == "comprehensive" or analysis["wants_total"]:
            return "load_all_thumbnails"
        elif analysis.get("scope") == "multiple":
            return "load_relevant_sections"
        else:
            return "load_specific_pages"
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and adjust analysis for consistency
        ENHANCED: Better multi-page handling and calculation support
        """
        
        # If asking for total, ensure it's a count question (unless it's a calculation)
        if analysis["wants_total"] and analysis["type"] not in [QuestionType.COUNT, QuestionType.ESTIMATE]:
            analysis["type"] = QuestionType.COUNT
        
        # If comprehensive scope, likely wants total
        if analysis.get("scope") == "comprehensive" and analysis["type"] == QuestionType.COUNT:
            analysis["wants_total"] = True
        
        # If specific page requested with total, adjust scope
        if analysis["requested_page"] and analysis["wants_total"]:
            # Could be asking for total on a specific page
            analysis["scope"] = "specific"
            analysis["wants_total"] = False
            analysis["confidence_in_analysis"] *= 0.9
        
        # Multiple areas but single page - adjust
        if len(analysis.get("specific_areas", [])) > 1 and analysis["requested_page"]:
            analysis["scope"] = "multiple"
        
        # Validate element focus
        if analysis["element_focus"] == "element":
            # Try to infer from question type and context
            if analysis["type"] == QuestionType.COMPLIANCE:
                # Compliance questions often about specific elements
                compliance_elements = ['outlet', 'door', 'stair', 'exit', 'sprinkler']
                for elem in compliance_elements:
                    if elem in analysis["normalized_prompt"]:
                        analysis["element_focus"] = elem
                        break
            elif analysis["type"] == QuestionType.ESTIMATE:
                # For calculations, might be about the building itself
                if analysis.get("calculation_type") in ["area", "cost"] and "building" in analysis["normalized_prompt"]:
                    analysis["element_focus"] = "building"
        
        # Add analysis metadata
        analysis["analysis_timestamp"] = datetime.utcnow().isoformat()
        analysis["analyzer_version"] = "3.0"  # Updated version
        analysis["unlimited_loading"] = CONFIG.get("unlimited_page_loading", True)
        
        return analysis
    
    def get_element_context(self, element_type: str) -> Dict[str, Any]:
        """Get context information for a specific element type"""
        
        context = {
            "element_type": element_type,
            "patterns": VISUAL_PATTERNS.get(element_type, VISUAL_PATTERNS.get("element", {})),
            "variations": [],
            "common_questions": [],
            "typical_locations": [],
            "multi_page_likelihood": "medium"  # NEW
        }
        
        # Find variations
        for variation, mapped_type in self.element_variations.items():
            if mapped_type == element_type:
                context["variations"].append(variation)
        
        # Element-specific context
        element_contexts = {
            "outlet": {
                "common_questions": [
                    "How many outlets are there?",
                    "Are outlets spaced to code?",
                    "Where are GFCI outlets located?",
                    "Count outlets on Level 2"
                ],
                "typical_locations": ["walls", "countertops", "floors"],
                "multi_page_likelihood": "high"
            },
            "door": {
                "common_questions": [
                    "How many doors are there?",
                    "What is the door width?",
                    "Are doors ADA compliant?",
                    "List all exit doors"
                ],
                "typical_locations": ["walls", "openings", "entries"],
                "multi_page_likelihood": "high"
            },
            "window": {
                "common_questions": [
                    "How many windows per floor?",
                    "What are the window sizes?",
                    "Which windows are operable?"
                ],
                "typical_locations": ["exterior walls", "facades"],
                "multi_page_likelihood": "high"
            },
            "sprinkler": {
                "common_questions": [
                    "Total sprinkler count?",
                    "Sprinkler coverage area?",
                    "Are all areas protected?"
                ],
                "typical_locations": ["ceilings", "storage areas"],
                "multi_page_likelihood": "very high"
            },
            "light fixture": {
                "common_questions": [
                    "How many light fixtures?",
                    "Emergency lighting locations?",
                    "Lighting levels adequate?"
                ],
                "typical_locations": ["ceilings", "walls"],
                "multi_page_likelihood": "very high"
            }
        }
        
        if element_type in element_contexts:
            context.update(element_contexts[element_type])
        
        return context
    
    def requires_multi_page_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        NEW METHOD - Determine if this query requires multi-page analysis
        Helps decide when to use all thumbnails
        """
        
        # Definitely needs multi-page
        if analysis.get("scope") == "comprehensive":
            return True
        
        if analysis["wants_total"] or analysis.get("wants_comprehensive", False):
            return True
        
        # Multiple locations specified
        if len(analysis.get("specific_areas", [])) > 1:
            return True
        
        if len(analysis.get("location_context", {}).get("floors", [])) > 1:
            return True
        
        # Element type that typically spans multiple pages
        element_context = self.get_element_context(analysis["element_focus"])
        if element_context.get("multi_page_likelihood") in ["high", "very high"]:
            if analysis["type"] == QuestionType.COUNT:
                return True
        
        # Keywords that suggest multi-page
        multi_page_keywords = [
            "all floors", "every floor", "each level", "throughout",
            "entire building", "all rooms", "comprehensive", "audit"
        ]
        
        if any(keyword in analysis["normalized_prompt"] for keyword in multi_page_keywords):
            return True
        
        return False
, 'expense', 'pricing',
                
                # Area/Square footage
                'area', 'square footage', 'sq ft', 'square feet', 'sqft',
                
                # Materials/Quantities
                'material', 'quantity needed', 'how much', 'how many gallons',
                'material needed', 'paint needed', 'drywall needed',
                
                # Loads (electrical, HVAC)
                'load', 'watts', 'amps', 'electrical load', 'hvac load',
                'cooling load', 'heating load', 'tonnage', 'cfm',
                
                # Spacing/Coverage
                'spacing', 'distance between', 'coverage', 'cover',
                
                # Ratios/Percentages
                'ratio', 'percentage', '%', 'percent', 'proportion',
                
                # Other indicators
                'rough estimate', 'approximate', 'ballpark'
            ]
        }
        
        # Drawing type patterns
        self.drawing_types = {
            'floor plan': ['floor plan', 'plan view', 'level', 'floor', 'flr'],
            'elevation': ['elevation', 'elev', 'facade', 'exterior view'],
            'section': ['section', 'cross section', 'sect', 'cut'],
            'detail': ['detail', 'enlarged', 'typical', 'dtl'],
            'schedule': ['schedule', 'table', 'legend', 'equipment list'],
            'site plan': ['site plan', 'site', 'civil', 'survey', 'plot plan'],
            'structural': ['structural', 'framing', 'foundation', 'struct'],
            'electrical': ['electrical', 'power', 'lighting', 'elec'],
            'plumbing': ['plumbing', 'piping', 'waste', 'water', 'plmb'],
            'mechanical': ['mechanical', 'hvac', 'ductwork', 'mech'],
            'fire': ['fire protection', 'fire alarm', 'sprinkler', 'fp']
        }
        
        # Use imported ELEMENT_VARIATIONS
        self.element_variations = ELEMENT_VARIATIONS
        
        # Contextual patterns - ENHANCED
        self.contextual_patterns = {
            "total_count": [
                "entire document", "whole file", "all pages", "total in",
                "throughout", "complete", "full set", "entire set",
                "across all", "in total", "whole building", "entire building",
                "all floors", "every floor", "all levels", "entire project",
                "complete count", "comprehensive count", "full count"
            ],
            "specific_area": [
                "in room", "on floor", "in area", "at grid", "near",
                "adjacent to", "next to", "around", "between",
                "within", "inside", "outside", "above", "below"
            ],
            "comparison": [
                "compare", "versus", "vs", "difference between",
                "more than", "less than", "same as", "match",
                "differ", "contrast"
            ],
            "multi_location": [
                "on floors", "in rooms", "across levels", "multiple areas",
                "various locations", "different floors", "several rooms",
                "floors \\d+ (?:to|through|-) \\d+", "levels \\d+ and \\d+"
            ],
            "calculation_context": [  # NEW
                "per square foot", "per unit", "total cost", "overall",
                "average", "mean", "typical", "standard", "based on",
                "assuming", "if", "with", "including", "excluding"
            ]
        }
        
        # Scope indicators - NEW
        self.scope_indicators = {
            "comprehensive": [
                "all", "every", "entire", "complete", "whole", "total",
                "throughout", "across", "comprehensive", "full"
            ],
            "specific": [
                "only", "just", "specific", "particular", "single",
                "one", "this", "that"
            ],
            "multiple": [
                "and", "&", "plus", "also", "as well as", "both",
                "multiple", "various", "several", "many"
            ]
        }
        
        # Calculation type patterns - NEW
        self.calculation_patterns = {
            "area": [
                r'\b(?:square|sq\.?)\s*(?:feet|ft|foot|footage)\b',
                r'\b(?:area|coverage|surface)\b',
                r'\bsqft\b', r'\bsf\b'
            ],
            "cost": [
                r'\$\s*\d+', r'\bdollar\b', r'\bcost\b', r'\bprice\b',
                r'\bbudget\b', r'\bexpense\b', r'\bquote\b'
            ],
            "load": [
                r'\b(?:electrical|power|hvac|cooling|heating)\s*load\b',
                r'\bwatts?\b', r'\bamps?\b', r'\bvoltage\b',
                r'\bbtu\b', r'\btonnage\b', r'\bcfm\b'
            ],
            "material": [
                r'\bhow\s+(?:much|many)\s+\w+\s+(?:do|will|needed)\b',
                r'\b(?:material|supplies?|quantity)\s+needed\b',
                r'\bgallons?\b', r'\bsheets?\b', r'\bunits?\b'
            ],
            "spacing": [
                r'\b(?:spacing|distance|apart|between)\b',
                r'\b(?:interval|gap|separation|clearance)\b'
            ],
            "time": [
                r'\b(?:hours?|days?|weeks?|months?)\b',
                r'\b(?:duration|timeline|schedule)\b',
                r'\bhow\s+long\b', r'\blabor\s+hours?\b'
            ],
            "percentage": [
                r'\d+\s*%', r'\bpercent(?:age)?\b',
                r'\bratio\b', r'\bproportion\b'
            ]
        }
    
    def set_vision_client(self, client):
        """Set the vision client for AI-based analysis"""
        self.vision_client = client
    
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Main method to analyze a question
        Returns comprehensive analysis of the user's intent
        
        ENHANCED: Better detection of queries needing multi-page analysis
        Enhanced calculation detection and categorization
        Backward compatible - all original fields preserved
        """
        
        logger.info(f"🤔 Analyzing question: '{prompt}'")
        
        prompt_lower = prompt.lower().strip()
        
        # Initialize analysis result - ALL ORIGINAL FIELDS PRESERVED
        analysis = {
            "original_prompt": prompt,
            "normalized_prompt": prompt_lower,
            "type": QuestionType.GENERAL,
            "element_focus": "element",
            "highlight_request": False,
            "requested_page": None,  # Original field preserved
            "drawing_type": None,
            "wants_total": False,
            "specific_area": None,  # Original field preserved
            "comparison_request": False,
            "temporal_context": None,
            "confidence_in_analysis": 0.9
        }
        
        # Check for highlight request
        if prompt_lower.startswith('highlight'):
            analysis["highlight_request"] = True
            prompt_lower = prompt_lower.replace('highlight', '', 1).strip()
        
        # Detect question type - ENHANCED FOR CALCULATIONS
        analysis["type"] = self._detect_question_type(prompt_lower)
        
        # Detect element focus (AI-first with fallback)
        analysis["element_focus"] = await self._detect_element_focus(prompt_lower)
        
        # Extract page references - USING ORIGINAL METHOD
        analysis["requested_page"] = self._extract_page_reference(prompt_lower)
        
        # Detect drawing type preference
        analysis["drawing_type"] = self._detect_drawing_type(prompt_lower)
        
        # Check for total count request - USING ORIGINAL METHOD
        analysis["wants_total"] = self._wants_total_count(prompt_lower)
        
        # Extract specific area references - USING ORIGINAL METHOD
        analysis["specific_area"] = self._extract_area_reference(prompt_lower)
        
        # Check for comparison requests
        analysis["comparison_request"] = self._is_comparison(prompt_lower)
        
        # Extract temporal context
        analysis["temporal_context"] = self._extract_temporal_context(prompt_lower)
        
        # NEW: Add enhanced analysis using new methods
        analysis = self._add_enhanced_analysis(analysis, prompt_lower)
        
        # NEW: Add calculation-specific analysis if ESTIMATE type
        if analysis["type"] == QuestionType.ESTIMATE:
            analysis = self._add_calculation_analysis(analysis, prompt_lower)
        
        # Validate and adjust analysis
        analysis = self._validate_analysis(analysis)
        
        logger.info(f"✅ Question analysis complete: Type={analysis['type'].value}, "
                   f"Element={analysis['element_focus']}, Scope={analysis.get('scope', 'specific')}, "
                   f"Pages={analysis.get('requested_pages', []) or analysis['requested_page'] or 'auto'}")
        
        return analysis
    
    def _detect_question_type(self, prompt_lower: str) -> QuestionType:
        """Detect the type of question being asked - ENHANCED FOR CALCULATIONS"""
        
        # Check each question type's keywords
        best_match = QuestionType.GENERAL
        best_score = 0
        
        for question_type, keywords in self.question_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > best_score:
                best_score = score
                best_match = question_type
        
        # Special cases and overrides
        if best_match == QuestionType.GENERAL:
            # Check for implicit patterns
            if any(word in prompt_lower for word in ['list', 'show me all', 'give me']):
                if 'where' in prompt_lower:
                    best_match = QuestionType.LOCATION
                else:
                    best_match = QuestionType.COUNT
            elif '?' in prompt_lower and any(word in prompt_lower for word in ['is', 'are']):
                best_match = QuestionType.IDENTIFY
        
        # Override for certain patterns
        if 'how many' in prompt_lower or 'count' in prompt_lower:
            # Check if it's actually a calculation
            if any(calc_word in prompt_lower for calc_word in ['cost', 'need', 'gallons', 'sheets']):
                best_match = QuestionType.ESTIMATE
            else:
                best_match = QuestionType.COUNT
        elif 'where' in prompt_lower and best_match != QuestionType.COUNT:
            best_match = QuestionType.LOCATION
        
        # Enhanced calculation detection
        if self._is_calculation_question(prompt_lower):
            best_match = QuestionType.ESTIMATE
        
        return best_match
    
    def _is_calculation_question(self, prompt_lower: str) -> bool:
        """Detect if this is a calculation question - NEW METHOD"""
        
        # Check for explicit calculation keywords
        calculation_indicators = [
            'calculate', 'compute', 'figure out', 'work out',
            'how much paint', 'how many gallons', 'how much material',
            'what is the cost', 'price', 'estimate',
            'square footage', 'sq ft', 'area',
            'electrical load', 'hvac load', 'cooling load',
            'spacing', 'distance between',
            'labor hours', 'man hours', 'duration',
            'percentage', 'ratio', 'proportion'
        ]
        
        for indicator in calculation_indicators:
            if indicator in prompt_lower:
                return True
        
        # Check for calculation patterns
        for calc_type, patterns in self.calculation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    return True
        
        # Check for unit mentions that suggest calculations
        unit_patterns = [
            r'\b\d+\s*(?:sq|square)\s*(?:ft|feet|foot)\b',
            r'\b\d+\s*(?:gallons?|gal)\b',
            r'\b\d+\s*(?:watts?|w)\b',
            r'\b\d+\s*(?:amps?|a)\b',
            r'\b\d+\s*(?:tons?)\b',
            r'\b\d+\s*(?:hours?|hrs?)\b',
            r'\b\d+\s*%'
        ]
        
        for pattern in unit_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        return False
    
    async def _detect_element_focus(self, prompt_lower: str) -> str:
        """
        Detect what element the user is asking about
        Uses AI first, then falls back to keyword matching
        """
        
        # Try AI detection first
        if self.vision_client and CONFIG.get("element_detection_confidence", 0.7) > 0:
            try:
                ai_element = await self._ai_detect_element(prompt_lower)
                if ai_element and ai_element != "element":
                    logger.info(f"🧠 AI detected element: '{ai_element}'")
                    return ai_element
            except Exception as e:
                logger.debug(f"AI element detection failed, using fallback: {e}")
        
        # Fallback to keyword detection
        return self._keyword_detect_element(prompt_lower)
    
    async def _ai_detect_element(self, prompt_lower: str) -> Optional[str]:
        """Use AI to understand what element type the user is asking about"""
        
        detection_prompt = f"""Analyze this construction/blueprint question and identify the PRIMARY element type being asked about.

QUESTION: {prompt_lower}

Consider:
- Complex phrases and technical jargon
- Abbreviations and shortcuts (e.g., "receps" for receptacles, "AHU" for air handling unit)
- Context clues about what they're looking for
- Industry-specific terminology
- Plural forms and variations
- If it's a calculation question, identify the element being calculated about

If the user is asking about multiple elements, identify the PRIMARY one.
If it's a general question not about a specific element, return "element".
If it's a calculation about the entire building/area, return "building".

Common construction elements include:
outlet, door, window, light, fixture, panel, breaker, sprinkler, column, beam, 
pipe, duct, equipment, diffuser, vav, pump, valve, sensor, detector, building, etc.

Return ONLY the element type in lowercase singular form (e.g., door not doors, outlet not outlets)

ELEMENT TYPE:"""

        try:
            response = await asyncio.wait_for(
                self.vision_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at understanding construction terminology and identifying element types from questions. You understand abbreviations, slang, and technical jargon."
                        },
                        {"role": "user", "content": detection_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.0
                ),
                timeout=CONFIG.get("element_detection_timeout", 10.0)
            )
            
            if response and response.choices:
                detected = response.choices[0].message.content.strip().lower()
                
                # Remove plural 's' if present
                if detected.endswith('s') and detected[:-1] in VISUAL_PATTERNS:
                    detected = detected[:-1]
                
                # Validate against known elements
                if detected in VISUAL_PATTERNS:
                    return detected
                
                # Check if it's a variation we know
                if detected in self.element_variations:
                    return self.element_variations[detected]
                
                # Check partial matches
                for known_element in VISUAL_PATTERNS.keys():
                    if known_element in detected or detected in known_element:
                        return known_element
                
                # Special case for building-wide calculations
                if detected in ["building", "project", "area", "space"]:
                    return "building"
                
                # If it seems valid but unknown, return it
                if detected and detected != "unknown" and detected != "element" and len(detected.split()) <= 2:
                    logger.info(f"🆕 New element type detected: '{detected}'")
                    return detected
                    
        except Exception as e:
            logger.debug(f"AI element detection error: {e}")
        
        return None
    
    def _keyword_detect_element(self, prompt_lower: str) -> str:
        """Keyword-based element detection"""
        
        # First check for exact matches
        for element_type in VISUAL_PATTERNS.keys():
            if element_type in prompt_lower:
                return element_type
            # Check plural
            if element_type + 's' in prompt_lower:
                return element_type
            # Check without spaces
            if element_type.replace(' ', '') in prompt_lower.replace(' ', ''):
                return element_type
        
        # Check variations and synonyms
        for variation, element_type in self.element_variations.items():
            if variation in prompt_lower:
                return element_type
        
        # Check for building-wide calculations
        building_indicators = [
            "building", "project", "total area", "entire", "whole",
            "square footage of", "paint for", "material for"
        ]
        for indicator in building_indicators:
            if indicator in prompt_lower:
                return "building"
        
        # Check for partial matches
        words = prompt_lower.split()
        for word in words:
            for element_type in VISUAL_PATTERNS.keys():
                if word in element_type or element_type in word:
                    return element_type
        
        return "element"
    
    def _extract_page_reference(self, prompt_lower: str) -> Optional[int]:
        """
        ORIGINAL METHOD - Extract specific page number from prompt
        Returns single page number for backward compatibility
        """
        
        # Page reference patterns
        page_patterns = [
            r'page\s+(\d+)',
            r'on\s+page\s+(\d+)',
            r'sheet\s+(\d+)',
            r'drawing\s+(\d+)',
            r'p\.?\s*(\d+)',
            r'pg\.?\s*(\d+)',
            r'page\s+number\s+(\d+)',
            r'look\s+at\s+page\s+(\d+)',
            r'check\s+page\s+(\d+)',
            r'see\s+page\s+(\d+)'
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                page_num = int(match.group(1))
                logger.info(f"📍 Detected page reference: {page_num}")
                return page_num
        
        return None
    
    def _extract_page_references_enhanced(self, prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Extract page references with support for multiple pages and ranges
        """
        
        result = {
            "single": None,
            "multiple": [],
            "ranges": []
        }
        
        # Get single page using original method
        result["single"] = self._extract_page_reference(prompt_lower)
        
        # Page reference patterns
        all_patterns = [
            (r'pages?\s+(\d+)\s*(?:to|through|-)\s*(\d+)', 'range'),
            (r'pages?\s+(\d+)\s*(?:and|&)\s*(\d+)', 'multiple'),
            (r'page\s+(\d+)', 'single'),
            (r'on\s+page\s+(\d+)', 'single'),
            (r'sheet\s+(\d+)', 'single'),
            (r'drawing\s+(\d+)', 'single'),
            (r'p\.?\s*(\d+)', 'single'),
            (r'pg\.?\s*(\d+)', 'single'),
            (r'pages?\s+(\d+(?:\s*,\s*\d+)+)', 'list'),
        ]
        
        all_pages = set()
        
        for pattern, ptype in all_patterns:
            matches = re.finditer(pattern, prompt_lower)
            for match in matches:
                if ptype == 'single':
                    page_num = int(match.group(1))
                    all_pages.add(page_num)
                elif ptype == 'range':
                    start = int(match.group(1))
                    end = int(match.group(2))
                    result["ranges"].append((start, end))
                    all_pages.update(range(start, end + 1))
                elif ptype == 'multiple':
                    page1 = int(match.group(1))
                    page2 = int(match.group(2))
                    all_pages.add(page1)
                    all_pages.add(page2)
                elif ptype == 'list':
                    page_list = match.group(1)
                    pages = [int(p.strip()) for p in page_list.split(',')]
                    all_pages.update(pages)
        
        result["multiple"] = sorted(list(all_pages))
        
        return result
    
    def _detect_drawing_type(self, prompt_lower: str) -> Optional[str]:
        """Detect if user is asking about a specific drawing type"""
        
        for dtype, terms in self.drawing_types.items():
            if any(term in prompt_lower for term in terms):
                return dtype
        
        return None
    
    def _wants_total_count(self, prompt_lower: str) -> bool:
        """
        ORIGINAL METHOD - Check if user wants total count across all pages
        Returns boolean for backward compatibility
        """
        
        total_indicators = [
            'total', 'all pages', 'whole file', 'entire',
            'document', 'complete set', 'throughout',
            'across all', 'in total', 'sum', 'aggregate'
        ]
        
        return any(indicator in prompt_lower for indicator in total_indicators)
    
    def _analyze_count_scope_enhanced(self, prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Enhanced detection of total/comprehensive count requests
        """
        
        result = {
            "wants_total": False,
            "wants_comprehensive": False,
            "confidence": 0.0
        }
        
        # Direct total indicators
        total_indicators = [
            'total', 'all pages', 'whole file', 'entire',
            'document', 'complete set', 'throughout',
            'across all', 'in total', 'sum', 'aggregate',
            'whole building', 'entire building', 'all floors',
            'every floor', 'all levels', 'entire project',
            'complete count', 'comprehensive', 'full count'
        ]
        
        # Check for total indicators
        total_score = sum(1 for indicator in total_indicators if indicator in prompt_lower)
        
        # Regex patterns for total count
        total_patterns = [
            r'how many.*?(?:total|all|entire)',
            r'total.*?(?:number|count|quantity)',
            r'count.*?(?:all|entire|whole)',
            r'(?:all|every|entire).*?(?:floor|level|page|room)',
            r'across\s+(?:all|the\s+entire)',
            r'throughout\s+(?:the\s+)?(?:building|document|project)'
        ]
        
        for pattern in total_patterns:
            if re.search(pattern, prompt_lower):
                total_score += 2
        
        # Comprehensive analysis indicators
        comprehensive_patterns = [
            'comprehensive', 'complete analysis', 'full review',
            'detailed count', 'thorough count', 'analyze all',
            'review all', 'check all', 'inspect all'
        ]
        
        comprehensive_score = sum(1 for pattern in comprehensive_patterns if pattern in prompt_lower)
        
        # Set results based on scores
        if total_score >= 2:
            result["wants_total"] = True
            result["confidence"] = min(1.0, total_score * 0.3)
        
        if comprehensive_score >= 1 or total_score >= 3:
            result["wants_comprehensive"] = True
            result["confidence"] = max(result["confidence"], 0.8)
        
        # Check for limiting words that negate total count
        limiting_words = ['only', 'just', 'specific', 'particular', 'single']
        if any(word in prompt_lower for word in limiting_words):
            result["wants_total"] = False
            result["wants_comprehensive"] = False
            result["confidence"] *= 0.5
        
        return result
    
    def _extract_area_reference(self, prompt_lower: str) -> Optional[str]:
        """
        ORIGINAL METHOD - Extract specific area references (room, grid, etc.)
        Returns single area reference for backward compatibility
        """
        
        # Room references
        room_match = re.search(r'(?:in|at|for)\s+(?:room|rm\.?)\s*([A-Z0-9]+)', prompt_lower, re.IGNORECASE)
        if room_match:
            return f"room_{room_match.group(1)}"
        
        # Grid references
        grid_match = re.search(r'(?:at|in|near)\s+grid\s*([A-Z]-?\d+)', prompt_lower, re.IGNORECASE)
        if grid_match:
            return f"grid_{grid_match.group(1)}"
        
        # Floor references
        floor_match = re.search(r'(?:on|at)\s+(?:floor|level)\s*(\d+|[A-Z]+)', prompt_lower, re.IGNORECASE)
        if floor_match:
            return f"floor_{floor_match.group(1)}"
        
        # Area descriptions
        for pattern in self.contextual_patterns["specific_area"]:
            if pattern in prompt_lower:
                # Extract the area description
                area_match = re.search(f'{pattern}\\s+([\\w\\s]+?)(?:\\?|$|,)', prompt_lower)
                if area_match:
                    return f"area_{area_match.group(1).strip()}"
        
        return None
    
    def _extract_area_references_enhanced(self, prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Extract all area references with better context
        """
        
        result = {
            "primary": None,
            "all": [],
            "context": {
                "floors": [],
                "rooms": [],
                "grids": [],
                "areas": [],
                "systems": []
            }
        }
        
        # Use original method to get primary
        result["primary"] = self._extract_area_reference(prompt_lower)
        
        # Floor/Level extraction - ENHANCED
        floor_patterns = [
            (r'(?:level|floor|flr\.?)\s*(\d+)', 'single'),
            (r'(\d+)(?:st|nd|rd|th)\s*(?:floor|level)', 'single'),
            (r'(?:floors?|levels?)\s*(\d+)\s*(?:to|through|-)\s*(\d+)', 'range'),
            (r'(?:floors?|levels?)\s*(\d+)\s*(?:and|&)\s*(\d+)', 'multiple'),
            (r'L(\d+)', 'single'),
            (r'(\d+)F\b', 'single'),
            (r'(?:basement|cellar|ground|first|second|third)', 'named'),
            (r'(?:all|every|each)\s+(?:floor|level)', 'all')
        ]
        
        for pattern, ptype in floor_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                if ptype == 'single':
                    floor = f"floor_{match.group(1)}"
                    result["all"].append(floor)
                    result["context"]["floors"].append(int(match.group(1)))
                elif ptype == 'range':
                    start = int(match.group(1))
                    end = int(match.group(2))
                    for f in range(start, end + 1):
                        result["all"].append(f"floor_{f}")
                        result["context"]["floors"].append(f)
                elif ptype == 'multiple':
                    for i in [1, 2]:
                        floor = f"floor_{match.group(i)}"
                        result["all"].append(floor)
                        result["context"]["floors"].append(int(match.group(i)))
                elif ptype == 'named':
                    floor_name = match.group(0).lower()
                    result["all"].append(f"floor_{floor_name}")
                    result["context"]["floors"].append(floor_name)
                elif ptype == 'all':
                    result["all"].append("floor_all")
                    result["context"]["floors"].append("all")
        
        # Room extraction - ENHANCED
        room_patterns = [
            r'(?:in|at|for)\s+(?:room|rm\.?)\s*([A-Z0-9]+)',
            r'(?:room|rm\.?)\s*([A-Z0-9]+)',
            r'(?:all|every)\s+(\w+\s*\w*)\s*(?:room|space)s?',
            r'(\w+\s*\w*)\s+(?:room|space|area)s?',
        ]
        
        # Also check for specific room types
        room_types = [
            'conference', 'meeting', 'office', 'bathroom', 'restroom',
            'kitchen', 'break room', 'storage', 'mechanical', 'electrical',
            'janitor', 'closet', 'lobby', 'corridor', 'stairwell'
        ]
        
        for pattern in room_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                room = f"room_{match.group(1)}"
                result["all"].append(room)
                result["context"]["rooms"].append(match.group(1))
        
        for room_type in room_types:
            if room_type in prompt_lower:
                result["all"].append(f"room_type_{room_type.replace(' ', '_')}")
                result["context"]["rooms"].append(room_type)
        
        # Grid references
        grid_patterns = [
            r'(?:at|in|near)\s+grid\s*([A-Z]-?\d+)',
            r'grid\s*([A-Z]-?\d+)',
            r'(?:between\s+)?grids?\s*([A-Z]-?\d+)\s*(?:and|to)\s*([A-Z]-?\d+)'
        ]
        
        for pattern in grid_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                if match.lastindex == 1:
                    grid = f"grid_{match.group(1)}"
                    result["all"].append(grid)
                    result["context"]["grids"].append(match.group(1))
                else:
                    # Grid range
                    for i in [1, 2]:
                        grid = f"grid_{match.group(i)}"
                        result["all"].append(grid)
                        result["context"]["grids"].append(match.group(i))
        
        # Area descriptions
        area_keywords = [
            'north', 'south', 'east', 'west', 'central', 'perimeter',
            'interior', 'exterior', 'parking', 'garage', 'roof', 'basement'
        ]
        
        for keyword in area_keywords:
            if keyword in prompt_lower:
                result["all"].append(f"area_{keyword}")
                result["context"]["areas"].append(keyword)
        
        # System references
        system_keywords = [
            'emergency', 'backup', 'normal power', 'fire alarm',
            'sprinkler', 'hvac', 'plumbing', 'electrical'
        ]
        
        for keyword in system_keywords:
            if keyword in prompt_lower:
                result["all"].append(f"system_{keyword.replace(' ', '_')}")
                result["context"]["systems"].append(keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_all = []
        for item in result["all"]:
            if item not in seen:
                seen.add(item)
                unique_all.append(item)
        result["all"] = unique_all
        
        return result
    
    def _determine_scope(self, prompt_lower: str, analysis: Dict[str, Any]) -> str:
        """
        NEW METHOD - Determine the scope of the analysis needed
        Returns: 'specific', 'multiple', or 'comprehensive'
        """
        
        # Check for comprehensive indicators
        comprehensive_score = 0
        for indicator in self.scope_indicators["comprehensive"]:
            if indicator in prompt_lower:
                comprehensive_score += 1
        
        # Check for specific indicators
        specific_score = 0
        for indicator in self.scope_indicators["specific"]:
            if indicator in prompt_lower:
                specific_score += 1
        
        # Check for multiple indicators
        multiple_score = 0
        for indicator in self.scope_indicators["multiple"]:
            if indicator in prompt_lower:
                multiple_score += 1
        
        # Additional checks
        if analysis["wants_total"] or analysis.get("wants_comprehensive", False):
            return "comprehensive"
        
        if len(analysis.get("specific_areas", [])) > 1:
            multiple_score += 2
        
        if len(analysis.get("requested_pages", [])) > 1:
            multiple_score += 1
        
        if analysis.get("location_context", {}).get("floors") == ["all"]:
            return "comprehensive"
        
        if len(analysis.get("location_context", {}).get("floors", [])) > 2:
            multiple_score += 2
        
        # For calculations, often need comprehensive data
        if analysis["type"] == QuestionType.ESTIMATE and analysis.get("calculation_type") in ["area", "cost", "load"]:
            return "comprehensive"
        
        # Determine scope based on scores
        if comprehensive_score >= 2 or (comprehensive_score > 0 and specific_score == 0):
            return "comprehensive"
        elif multiple_score >= 2 or len(analysis.get("specific_areas", [])) > 1:
            return "multiple"
        else:
            return "specific"
    
    def _is_comparison(self, prompt_lower: str) -> bool:
        """Check if this is a comparison request"""
        
        return any(pattern in prompt_lower for pattern in self.contextual_patterns["comparison"])
    
    def _extract_temporal_context(self, prompt_lower: str) -> Optional[str]:
        """Extract temporal context (current, proposed, existing, etc.)"""
        
        temporal_terms = {
            'existing': 'existing',
            'current': 'existing',
            'as-is': 'existing',
            'proposed': 'proposed',
            'new': 'proposed',
            'future': 'proposed',
            'planned': 'proposed',
            'to be': 'proposed',
            'as-built': 'as-built',
            'as built': 'as-built',
            'demo': 'demolition',
            'demolition': 'demolition',
            'remove': 'demolition',
            'revised': 'revised',
            'updated': 'revised'
        }
        
        for term, context in temporal_terms.items():
            if term in prompt_lower:
                return context
        
        return None
    
    def _add_enhanced_analysis(self, analysis: Dict[str, Any], prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Add enhanced analysis fields without breaking backward compatibility
        """
        
        # Get enhanced page references
        page_refs = self._extract_page_references_enhanced(prompt_lower)
        analysis["requested_pages"] = page_refs["multiple"]
        
        # Get enhanced area references
        area_refs = self._extract_area_references_enhanced(prompt_lower)
        analysis["specific_areas"] = area_refs["all"]
        analysis["location_context"] = area_refs["context"]
        
        # Get enhanced count scope
        count_scope = self._analyze_count_scope_enhanced(prompt_lower)
        analysis["wants_comprehensive"] = count_scope["wants_comprehensive"]
        
        # Determine scope
        analysis["scope"] = self._determine_scope(prompt_lower, analysis)
        
        # Add recommendation
        analysis["recommended_approach"] = self._get_recommended_approach(analysis)
        
        return analysis
    
    def _add_calculation_analysis(self, analysis: Dict[str, Any], prompt_lower: str) -> Dict[str, Any]:
        """
        NEW METHOD - Add calculation-specific analysis
        """
        
        # Determine calculation type
        calc_type = self._determine_calculation_type(prompt_lower)
        analysis["calculation_type"] = calc_type
        
        # Extract calculation parameters
        calc_params = self._extract_calculation_parameters(prompt_lower)
        analysis["calculation_parameters"] = calc_params
        
        # Determine if we need comprehensive data for calculation
        if calc_type in ["area", "cost", "load", "coverage"]:
            analysis["needs_comprehensive_data"] = True
            analysis["scope"] = "comprehensive"
        
        # Extract any numerical values or units
        analysis["extracted_values"] = self._extract_numerical_values(prompt_lower)
        
        return analysis
    
    def _determine_calculation_type(self, prompt_lower: str) -> str:
        """Determine specific calculation type - NEW METHOD"""
        
        # Priority order matters - check more specific patterns first
        if any(pattern in prompt_lower for pattern in ['cost', 'price', 'budget', '$', 'dollar', 'expense']):
            return "cost"
        
        elif any(pattern in prompt_lower for pattern in ['electrical load', 'power load', 'watts', 'amps', 'voltage']):
            return "electrical_load"
        
        elif any(pattern in prompt_lower for pattern in ['hvac load', 'cooling load', 'heating load', 'tons', 'btu', 'cfm']):
            return "hvac_load"
        
        elif any(pattern in prompt_lower for pattern in ['area', 'square footage', 'sq ft', 'square feet']):
            return "area"
        
        elif any(pattern in prompt_lower for pattern in ['spacing', 'distance', 'apart', 'between', 'separation']):
            return "spacing"
        
        elif any(pattern in prompt_lower for pattern in ['coverage', 'cover', 'protection']):
            return "coverage"
        
        elif any(pattern in prompt_lower for pattern in ['paint', 'drywall', 'material needed', 'quantity needed', 'how much']):
            return "material"
        
        elif any(pattern in prompt_lower for pattern in ['hours', 'days', 'duration', 'schedule', 'timeline']):
            return "time"
        
        elif any(pattern in prompt_lower for pattern in ['percentage', '%', 'ratio', 'proportion']):
            return "percentage"
        
        else:
            return "general"
    
    def _extract_calculation_parameters(self, prompt_lower: str) -> Dict[str, Any]:
        """Extract parameters needed for calculations - NEW METHOD"""
        
        params = {
            "has_constraints": False,
            "constraints": [],
            "assumptions_requested": False,
            "specific_conditions": []
        }
        
        # Check for constraints
        constraint_patterns = [
            (r'(?:max|maximum)\s+(\d+)', 'max'),
            (r'(?:min|minimum)\s+(\d+)', 'min'),
            (r'(?:budget|under|less than)\s*\$?\s*(\d+)', 'budget'),
            (r'(?:within|inside)\s+(\d+)', 'within')
        ]
        
        for pattern, constraint_type in constraint_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                params["has_constraints"] = True
                params["constraints"].append({
                    "type": constraint_type,
                    "value": match.group(1)
                })
        
        # Check for assumption requests
        if any(word in prompt_lower for word in ['assuming', 'assume', 'if', 'based on']):
            params["assumptions_requested"] = True
        
        # Check for specific conditions
        conditions = [
            'residential', 'commercial', 'industrial',
            'new construction', 'renovation', 'retrofit',
            'emergency', 'standard', 'premium',
            'code minimum', 'energy efficient'
        ]
        
        for condition in conditions:
            if condition in prompt_lower:
                params["specific_conditions"].append(condition)
        
        return params
    
    def _extract_numerical_values(self, prompt_lower: str) -> List[Dict[str, Any]]:
        """Extract numerical values and their units - NEW METHOD"""
        
        values = []
        
        # Pattern for number + unit
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:sq|square)\s*(?:ft|feet|foot)', 'area', 'sq ft'),
            (r'(\d+(?:\.\d+)?)\s*(?:gallons?|gal)', 'volume', 'gallons'),
            (r'(\d+(?:\.\d+)?)\s*(?:watts?|w)', 'power', 'watts'),
            (r'(\d+(?:\.\d+)?)\s*(?:amps?|a)\b', 'current', 'amps'),
            (r'(\d+(?:\.\d+)?)\s*(?:volts?|v)\b', 'voltage', 'volts'),
            (r'(\d+(?:\.\d+)?)\s*(?:tons?)\b', 'cooling', 'tons'),
            (r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', 'time', 'hours'),
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage', 'percent'),
            (r'\$\s*(\d+(?:\.\d+)?)', 'currency', 'dollars'),
        ]
        
        for pattern, value_type, unit in patterns:
            matches = re.finditer(pattern, prompt_lower)
            for match in matches:
                values.append({
                    "value": float(match.group(1)),
                    "type": value_type,
                    "unit": unit,
                    "context": prompt_lower[max(0, match.start()-20):match.end()+20]
                })
        
        return values
    
    def _get_recommended_approach(self, analysis: Dict[str, Any]) -> str:
        """
        NEW METHOD - Get recommended approach for page loading
        Enhanced for calculations
        """
        
        if analysis.get("scope") == "comprehensive" or analysis["wants_total"]:
            return "load_all_thumbnails"
        elif analysis.get("scope") == "multiple":
            return "load_relevant_sections"
        elif analysis["type"] == QuestionType.ESTIMATE and analysis.get("needs_comprehensive_data"):
            return "load_all_thumbnails"
        else:
            return "load_specific_pages"
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and adjust analysis for consistency
        ENHANCED: Better multi-page handling
        """
        
        # If asking for total, ensure it's a count question (unless it's an estimate)
        if analysis["wants_total"] and analysis["type"] not in [QuestionType.COUNT, QuestionType.ESTIMATE]:
            analysis["type"] = QuestionType.COUNT
        
        # If comprehensive scope, likely wants total
        if analysis.get("scope") == "comprehensive" and analysis["type"] == QuestionType.COUNT:
            analysis["wants_total"] = True
        
        # If specific page requested with total, adjust scope
        if analysis["requested_page"] and analysis["wants_total"]:
            # Could be asking for total on a specific page
            analysis["scope"] = "specific"
            analysis["wants_total"] = False
            analysis["confidence_in_analysis"] *= 0.9
        
        # Multiple areas but single page - adjust
        if len(analysis.get("specific_areas", [])) > 1 and analysis["requested_page"]:
            analysis["scope"] = "multiple"
        
        # Validate element focus
        if analysis["element_focus"] == "element":
            # Try to infer from question type and context
            if analysis["type"] == QuestionType.COMPLIANCE:
                # Compliance questions often about specific elements
                compliance_elements = ['outlet', 'door', 'stair', 'exit', 'sprinkler']
                for elem in compliance_elements:
                    if elem in analysis["normalized_prompt"]:
                        analysis["element_focus"] = elem
                        break
        
        # Add analysis metadata
        analysis["analysis_timestamp"] = datetime.utcnow().isoformat()
        analysis["analyzer_version"] = "3.0"  # Updated version
        analysis["unlimited_loading"] = CONFIG.get("unlimited_page_loading", True)
        
        return analysis
    
    def get_element_context(self, element_type: str) -> Dict[str, Any]:
        """Get context information for a specific element type"""
        
        context = {
            "element_type": element_type,
            "patterns": VISUAL_PATTERNS.get(element_type, VISUAL_PATTERNS.get("element", {})),
            "variations": [],
            "common_questions": [],
            "typical_locations": [],
            "multi_page_likelihood": "medium",  # NEW
            "calculable_properties": []  # NEW
        }
        
        # Find variations
        for variation, mapped_type in self.element_variations.items():
            if mapped_type == element_type:
                context["variations"].append(variation)
        
        # Element-specific context - ENHANCED WITH CALCULATIONS
        element_contexts = {
            "outlet": {
                "common_questions": [
                    "How many outlets are there?",
                    "Are outlets spaced to code?",
                    "Where are GFCI outlets located?",
                    "Count outlets on Level 2",
                    "Calculate electrical load for outlets",
                    "What's the outlet spacing?"
                ],
                "typical_locations": ["walls", "countertops", "floors"],
                "multi_page_likelihood": "high",
                "calculable_properties": ["spacing", "electrical_load", "cost"]
            },
            "door": {
                "common_questions": [
                    "How many doors are there?",
                    "What is the door width?",
                    "Are doors ADA compliant?",
                    "List all exit doors",
                    "Calculate door costs",
                    "How many fire-rated doors?"
                ],
                "typical_locations": ["walls", "openings", "entries"],
                "multi_page_likelihood": "high",
                "calculable_properties": ["cost", "fire_rating", "width"]
            },
            "window": {
                "common_questions": [
                    "How many windows per floor?",
                    "What are the window sizes?",
                    "Which windows are operable?",
                    "Calculate window-to-wall ratio",
                    "Estimate window costs"
                ],
                "typical_locations": ["exterior walls", "facades"],
                "multi_page_likelihood": "high",
                "calculable_properties": ["area", "cost", "ratio"]
            },
            "sprinkler": {
                "common_questions": [
                    "Total sprinkler count?",
                    "Sprinkler coverage area?",
                    "Are all areas protected?",
                    "Calculate sprinkler spacing",
                    "Is coverage to NFPA 13?"
                ],
                "typical_locations": ["ceilings", "storage areas"],
                "multi_page_likelihood": "very high",
                "calculable_properties": ["coverage", "spacing", "density"]
            },
            "light fixture": {
                "common_questions": [
                    "How many light fixtures?",
                    "Emergency lighting locations?",
                    "Lighting levels adequate?",
                    "Calculate lighting load",
                    "What's the watts per square foot?"
                ],
                "typical_locations": ["ceilings", "walls"],
                "multi_page_likelihood": "very high",
                "calculable_properties": ["electrical_load", "spacing", "coverage"]
            },
            "building": {
                "common_questions": [
                    "What's the total square footage?",
                    "Calculate project cost",
                    "How much paint is needed?",
                    "Calculate total electrical load",
                    "What's the building area?"
                ],
                "typical_locations": ["entire project"],
                "multi_page_likelihood": "very high",
                "calculable_properties": ["area", "cost", "materials", "loads"]
            }
        }
        
        if element_type in element_contexts:
            context.update(element_contexts[element_type])
        
        return context
    
    def requires_multi_page_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        NEW METHOD - Determine if this query requires multi-page analysis
        Helps decide when to use all thumbnails
        Enhanced for calculations
        """
        
        # Definitely needs multi-page
        if analysis.get("scope") == "comprehensive":
            return True
        
        if analysis["wants_total"] or analysis.get("wants_comprehensive", False):
            return True
        
        # Calculations often need comprehensive data
        if analysis["type"] == QuestionType.ESTIMATE:
            calc_type = analysis.get("calculation_type", "")
            if calc_type in ["area", "cost", "load", "coverage", "material"]:
                return True
        
        # Multiple locations specified
        if len(analysis.get("specific_areas", [])) > 1:
            return True
        
        if len(analysis.get("location_context", {}).get("floors", [])) > 1:
            return True
        
        # Element type that typically spans multiple pages
        element_context = self.get_element_context(analysis["element_focus"])
        if element_context.get("multi_page_likelihood") in ["high", "very high"]:
            if analysis["type"] in [QuestionType.COUNT, QuestionType.ESTIMATE]:
                return True
        
        # Keywords that suggest multi-page
        multi_page_keywords = [
            "all floors", "every floor", "each level", "throughout",
            "entire building", "all rooms", "comprehensive", "audit",
            "total area", "overall cost", "complete"
        ]
        
        if any(keyword in analysis["normalized_prompt"] for keyword in multi_page_keywords):
            return True
        
        return False