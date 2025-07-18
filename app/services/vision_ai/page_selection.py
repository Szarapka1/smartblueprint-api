# page_selection.py
import asyncio
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG

# Add model imports that might be needed
from app.models.schemas import (
    VisualIntelligenceResult,
    ElementGeometry,
    SemanticHighlight
)

logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Extracted context from the user's question"""
    element_type: str
    scope: str  # "specific", "multiple", "all", "range"
    locations: List[str]  # floors, rooms, areas, etc.
    location_types: List[str]  # "floor", "room", "area", "system"
    modifiers: List[str]  # "emergency", "GFCI", "accessible", etc.
    wants_total: bool
    specific_pages: List[int]
    
class PageSelector:
    """
    ULTRA-INTELLIGENT page selection using advanced NLP and vision understanding
    Handles ANY type of construction query with full context understanding
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None
        
        # Comprehensive patterns for location extraction
        self.location_patterns = {
            "floor": [
                r'level\s*(\d+)',
                r'floor\s*(\d+)',
                r'(\d+)(?:st|nd|rd|th)\s*floor',
                r'floors?\s*(\d+)\s*(?:to|through|-)\s*(\d+)',
                r'floors?\s*(\d+)\s*(?:and|&)\s*(\d+)',
                r'L(\d+)',  # L1, L2, etc.
                r'(\d+)F',  # 1F, 2F, etc.
            ],
            "room": [
                r'(?:in|within)\s+(?:the\s+)?(\w+\s*\w*)\s*(?:room|space|area)s?',
                r'(?:all|every)\s+(\w+\s*\w*)\s*(?:room|space|area)s?',
                r'(\w+\s*\w*)\s+(?:room|space|area)s?',
                r'(?:storage|mechanical|electrical|janitor|conference|bath)(?:room)?s?',
                r'(?:closet|office|lobby|corridor|hallway|stairwell)s?',
                r'(?:kitchen|breakroom|restroom|bathroom|toilet)s?',
            ],
            "area": [
                r'(?:north|south|east|west)\s*(?:wing|side|area|zone)',
                r'(?:main|central|perimeter|core)\s*(?:area|zone)?',
                r'(?:public|private|secure|restricted)\s*(?:area|zone)s?',
                r'(?:parking|garage|basement|roof|penthouse)',
                r'(?:interior|exterior|outdoor|indoor)\s*(?:area|space)?s?',
            ],
            "system": [
                r'(?:emergency|backup|normal)\s*(?:power|lighting|system)?',
                r'(?:GFCI|GFI)\s*(?:protected|circuit)?',
                r'(?:fire|life safety)\s*(?:system)?',
                r'(?:HVAC|mechanical|electrical|plumbing)\s*(?:system)?',
            ],
            "building": [
                r'(?:entire|whole|complete|all|full)\s*(?:building|project|document|set)?',
                r'(?:building|tower|phase)\s*([A-Z]|\d+)',
                r'(?:new|existing|proposed|future)\s*(?:building|construction)?',
            ]
        }
        
        # Room type keywords for semantic understanding
        self.room_types = {
            "storage": ["storage", "closet", "janitor", "custodial", "supply"],
            "bathroom": ["bathroom", "restroom", "toilet", "lavatory", "washroom"],
            "office": ["office", "workstation", "cubicle", "workspace"],
            "meeting": ["conference", "meeting", "boardroom", "training"],
            "utility": ["mechanical", "electrical", "telecom", "IT", "server"],
            "circulation": ["corridor", "hallway", "lobby", "vestibule", "stair"],
            "food": ["kitchen", "breakroom", "cafeteria", "dining", "pantry"],
            "parking": ["parking", "garage", "carport"],
            "special": ["lab", "clean room", "vault", "secure", "data center"]
        }
        
        # Drawing sheet patterns
        self.sheet_patterns = {
            "architectural": [r'A[-\.]?\d+', r'A\d+\.\d+', r'ARCH[-\s]?\d+'],
            "structural": [r'S[-\.]?\d+', r'S\d+\.\d+', r'STR[-\s]?\d+'],
            "mechanical": [r'M[-\.]?\d+', r'M\d+\.\d+', r'MECH[-\s]?\d+'],
            "electrical": [r'E[-\.]?\d+', r'E\d+\.\d+', r'ELEC[-\s]?\d+'],
            "plumbing": [r'P[-\.]?\d+', r'P\d+\.\d+', r'PLMB[-\s]?\d+'],
            "fire": [r'FP[-\.]?\d+', r'F\d+\.\d+', r'FIRE[-\s]?\d+'],
            "civil": [r'C[-\.]?\d+', r'C\d+\.\d+', r'CIVIL[-\s]?\d+']
        }
    
    def set_vision_client(self, client):
        """Set the vision client for API calls"""
        self.vision_client = client
    
    async def select_relevant_pages(
        self,
        thumbnails: List[Dict[str, Any]],
        question_analysis: Dict[str, Any],
        current_page: Optional[int]
    ) -> List[int]:
        """
        MASTER METHOD: Intelligently select pages based on ANY query type
        """
        
        if not thumbnails:
            logger.error("❌ No thumbnails provided!")
            return [1]
        
        # Extract comprehensive query context
        query_context = self._extract_query_context(question_analysis)
        
        logger.info(f"🧠 Query Context Analysis:")
        logger.info(f"  - Element: {query_context.element_type}")
        logger.info(f"  - Scope: {query_context.scope}")
        logger.info(f"  - Locations: {query_context.locations}")
        logger.info(f"  - Location Types: {query_context.location_types}")
        logger.info(f"  - Modifiers: {query_context.modifiers}")
        
        # Build intelligent selection prompt
        selection_prompt = self._build_intelligent_prompt(query_context, question_analysis)
        
        # Analyze thumbnails with our intelligent prompt
        selected_pages = await self._analyze_with_context(
            thumbnails, selection_prompt, query_context
        )
        
        # If no pages selected, use smart fallback
        if not selected_pages:
            logger.warning("⚠️ No pages selected, using intelligent fallback")
            selected_pages = await self._intelligent_fallback(
                thumbnails, query_context, question_analysis
            )
        
        # Apply intelligent limits
        selected_pages = self._apply_intelligent_limits(
            selected_pages, query_context, len(thumbnails)
        )
        
        logger.info(f"✅ Selected {len(selected_pages)} pages: {selected_pages}")
        return selected_pages
    
    def _extract_query_context(self, question_analysis: Dict[str, Any]) -> QueryContext:
        """Extract comprehensive context from the question"""
        
        prompt = question_analysis.get("original_prompt", "").lower()
        element_type = question_analysis.get("element_focus", "element")
        
        context = QueryContext(
            element_type=element_type,
            scope="specific",
            locations=[],
            location_types=[],
            modifiers=[],
            wants_total=question_analysis.get("wants_total", False),
            specific_pages=[]
        )
        
        # Extract all location references
        for loc_type, patterns in self.location_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, prompt, re.IGNORECASE)
                for match in matches:
                    context.location_types.append(loc_type)
                    
                    # Handle different match groups
                    if match.groups():
                        if len(match.groups()) == 2:  # Range or multiple
                            context.locations.extend([match.group(1), match.group(2)])
                            context.scope = "range" if "to" in prompt or "through" in prompt else "multiple"
                        else:
                            context.locations.append(match.group(1))
                    else:
                        context.locations.append(match.group(0))
        
        # Extract room types
        for room_category, keywords in self.room_types.items():
            for keyword in keywords:
                if keyword in prompt:
                    context.locations.append(keyword)
                    context.location_types.append("room")
        
        # Extract modifiers
        modifier_patterns = [
            "emergency", "backup", "gfci", "gfi", "accessible", "ada",
            "fire rated", "exterior", "interior", "new", "existing",
            "typical", "special", "dedicated", "shared"
        ]
        for modifier in modifier_patterns:
            if modifier in prompt:
                context.modifiers.append(modifier)
        
        # Determine scope
        if any(word in prompt for word in ["entire", "whole", "all", "complete", "total", "every"]):
            context.scope = "all"
        elif len(context.locations) > 1:
            context.scope = "multiple"
        elif "and" in prompt or "&" in prompt:
            context.scope = "multiple"
        
        # Extract specific page references
        page_patterns = [
            r'page\s*(\d+)',
            r'sheet\s*(\d+)',
            r'drawing\s*(\d+)',
            r'see\s*page\s*(\d+)'
        ]
        for pattern in page_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            context.specific_pages.extend([int(m) for m in matches])
        
        return context
    
    def _build_intelligent_prompt(self, context: QueryContext, question_analysis: Dict[str, Any]) -> str:
        """Build an intelligent prompt for page selection"""
        
        prompt_parts = []
        
        # Start with the original question
        original = question_analysis.get("original_prompt", "")
        prompt_parts.append(f"USER QUESTION: {original}")
        prompt_parts.append(f"\nLOOKING FOR: {context.element_type}s")
        
        # Add location context
        if context.locations:
            if context.scope == "all":
                prompt_parts.append("\nSCOPE: ENTIRE BUILDING/DOCUMENT - Select ALL relevant pages")
            elif context.scope == "multiple":
                prompt_parts.append(f"\nMULTIPLE LOCATIONS: {', '.join(context.locations)}")
            elif context.scope == "range":
                prompt_parts.append(f"\nRANGE: {context.locations[0]} through {context.locations[-1]}")
            else:
                prompt_parts.append(f"\nSPECIFIC LOCATION: {', '.join(context.locations)}")
        
        # Add modifier context
        if context.modifiers:
            prompt_parts.append(f"\nSPECIAL REQUIREMENTS: {', '.join(context.modifiers)}")
        
        # Add intelligent instructions
        prompt_parts.append("\n\nANALYZE EACH PAGE FOR:")
        
        # Location-specific instructions
        if "floor" in context.location_types or "level" in context.location_types:
            prompt_parts.append("- Floor plans showing the specified levels")
            prompt_parts.append("- Elevations or sections that include these floors")
            prompt_parts.append("- Schedules that break down by floor")
        
        if "room" in context.location_types:
            room_keywords = [loc for loc in context.locations if any(
                loc in room_list for room_list in self.room_types.values()
            )]
            if room_keywords:
                prompt_parts.append(f"- Rooms labeled as: {', '.join(room_keywords)}")
                prompt_parts.append("- Enlarged plans of these room types")
                prompt_parts.append("- Room schedules or finish schedules")
        
        if "area" in context.location_types:
            prompt_parts.append("- Specific building areas or zones mentioned")
            prompt_parts.append("- Site plans showing different areas")
            prompt_parts.append("- Zoning or area plans")
        
        if "system" in context.location_types:
            prompt_parts.append("- System diagrams or riser diagrams")
            prompt_parts.append("- Equipment schedules for these systems")
            prompt_parts.append("- One-line diagrams or schematics")
        
        # Element-specific instructions
        prompt_parts.append(f"\n- Any page showing {context.element_type} symbols")
        prompt_parts.append(f"- {context.element_type} schedules or legends")
        prompt_parts.append(f"- Details showing {context.element_type} installation")
        
        # Final instructions
        prompt_parts.append("\nBE COMPREHENSIVE - Include:")
        prompt_parts.append("- All pages that match the location criteria")
        prompt_parts.append("- Related schedules and details")
        prompt_parts.append("- Overview pages that show the full scope")
        
        return "\n".join(prompt_parts)
    
    async def _analyze_with_context(
        self,
        thumbnails: List[Dict[str, Any]],
        selection_prompt: str,
        context: QueryContext
    ) -> List[int]:
        """Analyze thumbnails with full context understanding"""
        
        if not self.vision_client:
            logger.error("❌ No vision client available!")
            return []
        
        # For specific pages, return them directly
        if context.specific_pages:
            logger.info(f"📄 Using specific pages: {context.specific_pages}")
            return context.specific_pages
        
        # Process all thumbnails for comprehensive analysis
        all_selected = []
        chunk_size = 10  # Smaller chunks for better accuracy
        
        for i in range(0, len(thumbnails), chunk_size):
            chunk = thumbnails[i:i+chunk_size]
            
            content = [
                {"type": "text", "text": f"ANALYZING PAGES {i+1} to {min(i+chunk_size, len(thumbnails))}:"}
            ]
            
            # Add thumbnails with clear page numbers
            for thumb in chunk:
                content.append({
                    "type": "text",
                    "text": f"\n=== PAGE {thumb['page']} ==="
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": thumb["url"], "detail": "low"}
                })
            
            content.append({"type": "text", "text": selection_prompt})
            content.append({
                "type": "text",
                "text": "\n\nRESPONSE FORMAT:\nRELEVANT_PAGES: [list ALL relevant page numbers]\nREASONING: [explain your selection]"
            })
            
            try:
                response = await self.vision_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert construction document analyst. 
You understand building floors, room types, systems, and how to find information across multiple pages.
Be thorough - missing relevant pages means missing important information."""
                        },
                        {"role": "user", "content": content}
                    ],
                    max_tokens=2000,
                    temperature=0.0
                )
                
                if response and response.choices:
                    result = response.choices[0].message.content or ""
                    pages = self._parse_comprehensive_selection(result, chunk)
                    all_selected.extend(pages)
                    
                    # Log reasoning for debugging
                    reasoning_match = re.search(r'REASONING:\s*(.+)', result, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        logger.debug(f"Selection reasoning: {reasoning_match.group(1)[:200]}...")
                    
            except Exception as e:
                logger.error(f"Vision analysis error: {e}")
        
        return sorted(list(set(all_selected)))
    
    def _parse_comprehensive_selection(self, result: str, chunk: List[Dict[str, Any]]) -> List[int]:
        """Parse selection with better understanding of responses"""
        
        pages = []
        
        # Standard format parsing
        pages_match = re.search(
            r'RELEVANT_PAGES:\s*\[?\s*([^\]]+)\s*\]?',
            result,
            re.IGNORECASE | re.DOTALL
        )
        
        if pages_match:
            page_str = pages_match.group(1)
            
            # Handle various formats
            # Numbers: 1, 2, 3
            numbers = re.findall(r'\b(\d+)\b', page_str)
            pages.extend([int(n) for n in numbers])
            
            # Ranges: 3-5, 3 to 5, 3 through 5
            ranges = re.findall(r'(\d+)\s*(?:-|to|through)\s*(\d+)', page_str)
            for start, end in ranges:
                pages.extend(range(int(start), int(end) + 1))
        
        # Also check the reasoning for page mentions
        page_mentions = re.findall(r'page\s+(\d+)', result, re.IGNORECASE)
        pages.extend([int(p) for p in page_mentions])
        
        # Check for "all pages" type responses
        if any(phrase in result.lower() for phrase in [
            "all pages", "every page", "all of them", "all relevant",
            "pages 1 through", "all sheets shown"
        ]):
            # Add all pages from this chunk
            pages.extend([thumb["page"] for thumb in chunk])
        
        # Clean up and validate
        valid_pages = sorted(list(set(p for p in pages if p > 0)))
        
        return valid_pages
    
    async def _intelligent_fallback(
        self,
        thumbnails: List[Dict[str, Any]],
        context: QueryContext,
        question_analysis: Dict[str, Any]
    ) -> List[int]:
        """Intelligent fallback when primary selection fails"""
        
        logger.info("🔄 Using intelligent fallback selection")
        
        pages = set()
        
        # For "all" scope, return everything
        if context.scope == "all" or context.wants_total:
            logger.info("📄 Scope is 'all' - selecting entire document")
            return [t["page"] for t in thumbnails]
        
        # For floor-specific queries
        if "floor" in context.location_types:
            # Common patterns: early pages have floor plans
            # Add pages in groups (floors often span 2-3 pages)
            if len(thumbnails) >= 10:
                pages.update([1, 2, 3, 4, 5])  # First floors
                pages.update([6, 7, 8, 9, 10])  # Additional floors
            else:
                pages.update(range(1, len(thumbnails) + 1))
        
        # For room-specific queries
        if "room" in context.location_types:
            # Detailed plans are often in the middle
            if len(thumbnails) > 5:
                mid = len(thumbnails) // 2
                pages.update(range(max(1, mid - 2), min(len(thumbnails) + 1, mid + 3)))
        
        # For system queries
        if "system" in context.location_types:
            # System drawings often after architectural
            if len(thumbnails) > 10:
                pages.update(range(len(thumbnails) // 2, len(thumbnails) + 1))
        
        # Always include some key pages
        if len(thumbnails) >= 3:
            pages.add(1)  # Usually has general info
            pages.add(2)  # Often first floor plan
            pages.add(len(thumbnails))  # Often has schedules
        
        # Element-specific fallbacks
        element_pages = {
            "outlet": [1, 2, 3, 4, 5],  # Electrical usually early
            "window": [1, 2, 3] + list(range(len(thumbnails) - 2, len(thumbnails) + 1)),
            "door": [1, 2, 3] + [len(thumbnails)],  # Plans + schedule
            "equipment": list(range(max(1, len(thumbnails) - 5), len(thumbnails) + 1))
        }
        
        if context.element_type in element_pages:
            pages.update([p for p in element_pages[context.element_type] if p <= len(thumbnails)])
        
        return sorted(list(pages))
    
    def _apply_intelligent_limits(
        self,
        pages: List[int],
        context: QueryContext,
        total_pages: int
    ) -> List[int]:
        """Apply intelligent limits based on context"""
        
        # No limits for "all" scope
        if context.scope == "all" or context.wants_total:
            logger.info("📊 No limits applied - comprehensive analysis needed")
            return pages
        
        # Generous limits for complex queries
        if context.scope in ["multiple", "range"]:
            max_pages = min(20, total_pages)
        else:
            max_pages = min(10, total_pages)
        
        if len(pages) <= max_pages:
            return pages
        
        logger.info(f"📉 Reducing {len(pages)} pages to {max_pages}")
        
        # Intelligent reduction - keep diversity
        if len(pages) > max_pages * 2:
            # Take every Nth page to maintain coverage
            step = len(pages) // max_pages
            selected = pages[::step][:max_pages]
        else:
            # Prioritize based on context
            prioritized = []
            
            # Always include first and last
            if 1 in pages:
                prioritized.append(1)
            if total_pages in pages:
                prioritized.append(total_pages)
            
            # Fill in the middle
            remaining = max_pages - len(prioritized)
            middle_pages = [p for p in pages if p not in prioritized]
            
            if middle_pages:
                # Evenly distribute
                step = max(1, len(middle_pages) // remaining)
                prioritized.extend(middle_pages[::step][:remaining])
            
            selected = sorted(prioritized)
        
        return selected
    
    def _wants_total_count(self, question_analysis: Dict[str, Any]) -> bool:
        """Enhanced check for total count requests"""
        
        prompt = question_analysis.get("original_prompt", "").lower()
        
        # Explicit total indicators
        total_phrases = [
            "total", "all", "entire", "whole", "complete",
            "how many.*in the.*document",
            "how many.*in the.*building",
            "how many.*in the.*project",
            "count.*all",
            "every.*in the"
        ]
        
        for phrase in total_phrases:
            if re.search(phrase, prompt):
                return True
        
        # Check scope from context
        if question_analysis.get("wants_total"):
            return True
        
        return False
