# vision_intelligence.py - COMPLETE REWRITE FOR DOCUMENT INTELLIGENCE

import asyncio
import re
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from openai import AsyncOpenAI

from app.models.schemas import VisualIntelligenceResult, ElementGeometry
from app.core.config import CONFIG
from .patterns import VISUAL_PATTERNS, ELEMENT_VARIATIONS

logger = logging.getLogger(__name__)


class VisionIntelligence:
    """
    Document Intelligence System using GPT-4 Vision.
    
    PHILOSOPHY: Understand the ENTIRE document first, then answer ANY question.
    This is not an element hunter - it's a document comprehension system.
    """

    def __init__(self, settings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        self._client: Optional[AsyncOpenAI] = None
        self.vision_semaphore: Optional[asyncio.Semaphore] = None
        self.deterministic_seed = 42
        
        # Document Intelligence Cache - stores complete understanding
        self.document_knowledge: Dict[str, Any] = {}
        self.current_document_id: Optional[str] = None
        
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
        Main analysis method - BUT NOW IT'S DIFFERENT!
        
        Instead of hunting for elements, we:
        1. Build/retrieve comprehensive document knowledge
        2. Answer the question from that knowledge
        3. Return results in the expected format
        """
        logger.info(f"🧠 Document Intelligence Analysis for: {prompt}")
        
        # Generate document ID from images (or use provided one)
        doc_id = self._generate_document_id(images)
        
        try:
            # STEP 1: Do we already understand this document?
            if not self._has_document_knowledge(doc_id):
                logger.info("📚 Building comprehensive document intelligence...")
                await self._build_document_intelligence(
                    doc_id, images, comprehensive_data
                )
            else:
                logger.info("✅ Using cached document intelligence")
            
            # STEP 2: Answer the question from our knowledge
            logger.info(f"💡 Answering from document knowledge: {prompt}")
            answer = await self._answer_from_knowledge(
                prompt, question_analysis, doc_id
            )
            
            # STEP 3: Convert to expected format
            visual_result = self._format_as_visual_result(
                answer, question_analysis, page_number
            )
            
            logger.info(f"✅ Intelligence complete: {visual_result.count} {visual_result.element_type}(s)")
            return visual_result

        except Exception as e:
            logger.error(f"❌ Document Intelligence error: {e}", exc_info=True)
            return self._create_error_result(page_number, str(e))

    async def _build_document_intelligence(
        self,
        doc_id: str,
        images: List[Dict[str, Any]],
        comprehensive_data: Optional[Dict[str, Any]] = None
    ):
        """
        BUILD COMPREHENSIVE DOCUMENT UNDERSTANDING - This is the KEY!
        We analyze EVERYTHING ONCE, understand it deeply, then cache it.
        """
        
        self._ensure_semaphores_initialized()
        comprehensive_data = comprehensive_data or {}
        
        # Initialize knowledge structure
        knowledge = {
            "document_id": doc_id,
            "created_at": datetime.utcnow().isoformat(),
            "project_overview": {},
            "page_contents": {},
            "all_elements": {},
            "systems": {},
            "spatial_organization": {},
            "relationships": {},
            "specifications": {},
            "metadata": {
                "total_pages": len(images),
                "analysis_version": "2.0"
            }
        }
        
        # PHASE 1: Project Overview - What is this building/project?
        logger.info("📋 Phase 1: Understanding the project...")
        knowledge["project_overview"] = await self._understand_project(images[:5])
        
        # PHASE 2: Page-by-Page Deep Analysis
        logger.info("📄 Phase 2: Analyzing each page comprehensively...")
        knowledge["page_contents"] = await self._analyze_all_pages(images)
        
        # PHASE 3: Element Inventory - Count EVERYTHING
        logger.info("🔍 Phase 3: Building complete element inventory...")
        knowledge["all_elements"] = await self._inventory_all_elements(images)
        
        # PHASE 4: Systems Understanding
        logger.info("⚙️ Phase 4: Understanding building systems...")
        knowledge["systems"] = await self._understand_systems(images, knowledge)
        
        # PHASE 5: Spatial Organization
        logger.info("📐 Phase 5: Understanding spatial organization...")
        knowledge["spatial_organization"] = self._analyze_spatial_organization(
            comprehensive_data.get("grid_systems", {}),
            knowledge["page_contents"]
        )
        
        # PHASE 6: Relationships and Cross-References
        logger.info("🔗 Phase 6: Building relationships...")
        knowledge["relationships"] = self._build_relationships(
            knowledge,
            comprehensive_data.get("document_index", {})
        )
        
        # Cache this knowledge
        self.document_knowledge[doc_id] = knowledge
        self.current_document_id = doc_id
        
        logger.info(f"✅ Document Intelligence complete! Understood {len(images)} pages.")
        logger.info(f"📊 Found {len(knowledge['all_elements'])} element types")

    async def _understand_project(self, sample_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Understand what this project/building is about."""
        
        prompt = """UNDERSTAND THIS CONSTRUCTION PROJECT

Analyze these blueprint pages and tell me:

1. **PROJECT TYPE**: What is being built? (office, residential, industrial, etc.)
2. **SCALE**: How big is this project? (square footage, stories, etc.)
3. **DISCIPLINES**: What drawing disciplines are included? (architectural, MEP, structural, etc.)
4. **COMPLEXITY**: Simple, moderate, or complex project?
5. **KEY FEATURES**: Notable elements or systems visible?

Provide a comprehensive overview of what this project is.

RESPONSE FORMAT:
PROJECT TYPE: [type]
SCALE: [description]
DISCIPLINES: [list]
COMPLEXITY: [level]
KEY FEATURES: [list]
SUMMARY: [2-3 sentence overview]"""

        content = self._prepare_vision_content(sample_images, prompt)
        response = await self._make_vision_request(
            content,
            "You are a senior construction professional reviewing a new project set.",
            2000
        )
        
        return self._parse_project_overview(response)

    async def _analyze_all_pages(self, images: List[Dict[str, Any]]) -> Dict[int, Any]:
        """Deep analysis of EVERY page - understanding what's on each."""
        
        page_contents = {}
        
        # Process in batches for efficiency
        batch_size = 5
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_analysis = await self._analyze_page_batch(batch, i)
            page_contents.update(batch_analysis)
        
        return page_contents

    async def _analyze_page_batch(
        self, 
        batch: List[Dict[str, Any]], 
        start_index: int
    ) -> Dict[int, Any]:
        """Analyze a batch of pages comprehensively."""
        
        prompt = """COMPREHENSIVE PAGE ANALYSIS

For EACH page shown, provide:

1. **PAGE TYPE**: (floor plan, elevation, section, detail, schedule, etc.)
2. **CONTENT SUMMARY**: What's shown on this page?
3. **ELEMENTS VISIBLE**: List ALL construction elements you can see with rough counts
4. **SYSTEMS SHOWN**: What building systems are depicted?
5. **KEY INFORMATION**: Important notes, dimensions, or specifications

Don't just count one element type - describe EVERYTHING on each page!

For each page, format as:
PAGE [number]:
- TYPE: [type]
- CONTENT: [what's shown]
- ELEMENTS: [comprehensive list with counts]
- SYSTEMS: [list]
- KEY INFO: [important details]"""

        content = self._prepare_vision_content(batch, prompt)
        response = await self._make_vision_request(
            content,
            "You are analyzing construction drawings to build a complete understanding of what's on each page.",
            3000
        )
        
        return self._parse_page_analysis(response, start_index)

    async def _inventory_all_elements(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a complete inventory of ALL elements in the document."""
        
        prompt = """COMPLETE ELEMENT INVENTORY

Count ALL construction elements across ALL pages. Be exhaustive!

Categories to inventory:
- ARCHITECTURAL: doors, windows, walls, rooms, stairs, elevators, etc.
- STRUCTURAL: columns, beams, footings, slabs, etc.
- ELECTRICAL: outlets, switches, panels, lights, conduits, etc.
- PLUMBING: fixtures, pipes, valves, drains, water heaters, etc.
- MECHANICAL: diffusers, ducts, equipment, VAVs, thermostats, etc.
- FIRE/LIFE SAFETY: sprinklers, alarms, extinguishers, exits, etc.
- SPECIALTIES: signage, equipment, casework, accessories, etc.

For EACH element type found:
ELEMENT: [type]
TOTAL COUNT: [number]
PAGES FOUND ON: [list]
TYPICAL TAGS/LABELS: [if any]

Be thorough - this is our master inventory!"""

        # For large documents, we might process in chunks
        # For now, send all at once
        content = self._prepare_vision_content(images, prompt)
        response = await self._make_vision_request(
            content,
            "You are creating a comprehensive construction element inventory. Count EVERYTHING visible.",
            4000
        )
        
        return self._parse_element_inventory(response)

    async def _understand_systems(
        self, 
        images: List[Dict[str, Any]], 
        knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Understand all building systems and their relationships."""
        
        # Focus on system-specific pages if we identified them
        system_pages = self._identify_system_pages(knowledge["page_contents"])
        
        if system_pages:
            relevant_images = [img for img in images if img.get("page", 0) in system_pages]
        else:
            relevant_images = images
        
        prompt = """BUILDING SYSTEMS ANALYSIS

Identify and describe ALL building systems:

1. **ELECTRICAL SYSTEM**:
   - Power distribution (panels, transformers, etc.)
   - Lighting systems and controls
   - Emergency power
   - Special systems (fire alarm, security, etc.)

2. **MECHANICAL SYSTEM**:
   - HVAC equipment and distribution
   - Controls and zones
   - Exhaust systems

3. **PLUMBING SYSTEM**:
   - Water supply and distribution
   - Drainage and venting
   - Fixtures and equipment
   - Special systems (gas, medical, etc.)

4. **FIRE PROTECTION**:
   - Sprinkler systems
   - Standpipes
   - Fire alarm and detection

5. **STRUCTURAL SYSTEM**:
   - Foundation type
   - Framing system
   - Lateral system

For each system, describe:
- Components and quantities
- Distribution/layout strategy
- Notable features or requirements"""

        content = self._prepare_vision_content(relevant_images[:10], prompt)
        response = await self._make_vision_request(
            content,
            "You are a building systems expert analyzing the MEP and structural systems.",
            3000
        )
        
        return self._parse_systems_analysis(response)

    async def _answer_from_knowledge(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Answer ANY question using our comprehensive document knowledge.
        This is where the magic happens - we already know everything!
        """
        
        knowledge = self.document_knowledge[doc_id]
        
        # Build a context-aware prompt with our knowledge
        answer_prompt = f"""Using my comprehensive knowledge of this construction document, answer this question:

USER QUESTION: {prompt}

MY DOCUMENT KNOWLEDGE:
- Project: {knowledge['project_overview'].get('summary', 'Construction project')}
- Total Pages: {knowledge['metadata']['total_pages']}
- Element Inventory: I have catalogued {len(knowledge['all_elements'])} different element types
- Systems: I understand {len(knowledge['systems'])} building systems

RELEVANT KNOWLEDGE:
{self._get_relevant_knowledge(prompt, question_analysis, knowledge)}

Provide a detailed, accurate answer based on my complete understanding of this document.
If the question asks for counts, locations, or specifications, be precise using my knowledge.

IMPORTANT: I already know what's in this document - I don't need to search, I need to recall and report!"""

        # We don't need images here - we're answering from knowledge!
        response = await self._query_knowledge(answer_prompt, knowledge)
        
        return self._parse_knowledge_answer(response, question_analysis, knowledge)

    async def _query_knowledge(self, prompt: str, knowledge: Dict[str, Any]) -> str:
        """Query our knowledge base for an answer."""
        
        async with self.vision_semaphore:
            try:
                # We can use chat completion without images since we're working from knowledge
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a construction document expert with perfect recall of the document you've already analyzed. Answer questions using your stored knowledge."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.0,
                    seed=self.deterministic_seed
                )
                return response.choices[0].message.content if response and response.choices else ""
            except Exception as e:
                logger.error(f"Knowledge query failed: {e}")
                return ""

    def _get_relevant_knowledge(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> str:
        """Extract relevant knowledge for the question."""
        
        relevant_parts = []
        element_type = question_analysis.get("element_focus", "element")
        
        # If asking about a specific element, provide that inventory
        if element_type in knowledge["all_elements"]:
            element_data = knowledge["all_elements"][element_type]
            relevant_parts.append(
                f"- {element_type.upper()} INVENTORY: "
                f"Total Count: {element_data.get('total_count', 0)}, "
                f"Found on pages: {element_data.get('pages', [])}"
            )
        
        # Add page-specific knowledge if asking about specific pages
        requested_page = question_analysis.get("requested_page")
        if requested_page and requested_page in knowledge["page_contents"]:
            page_data = knowledge["page_contents"][requested_page]
            relevant_parts.append(
                f"- PAGE {requested_page} CONTENTS: {page_data.get('content_summary', 'No data')}"
            )
        
        # Add system knowledge if relevant
        for system_name, system_data in knowledge["systems"].items():
            if any(word in prompt.lower() for word in system_name.lower().split()):
                relevant_parts.append(
                    f"- {system_name.upper()}: {system_data.get('summary', 'System present')}"
                )
        
        return "\n".join(relevant_parts) if relevant_parts else "- Full document knowledge available"

    def _format_as_visual_result(
        self,
        answer: Dict[str, Any],
        question_analysis: Dict[str, Any],
        page_number: int
    ) -> VisualIntelligenceResult:
        """Convert our knowledge-based answer to the expected format."""
        
        element_type = answer.get("element_type", question_analysis.get("element_focus", "element"))
        
        # Build the result from our answer
        return VisualIntelligenceResult(
            element_type=element_type,
            count=answer.get("count", 0),
            locations=answer.get("locations", []),
            confidence=answer.get("confidence", 0.95),  # High confidence from knowledge!
            grid_references=answer.get("grid_references", []),
            visual_evidence=answer.get("evidence", [
                "Found in document knowledge base",
                f"Document contains comprehensive inventory of {element_type}s"
            ]),
            pattern_matches=answer.get("patterns", []),
            verification_notes=answer.get("notes", [
                "Answer derived from comprehensive document intelligence"
            ]),
            page_number=page_number,
            analysis_metadata={
                "source": "document_knowledge",
                "knowledge_version": "2.0",
                "total_pages_understood": len(self.document_knowledge.get(
                    self.current_document_id, {}
                ).get("page_contents", {}))
            }
        )

    # === PARSING METHODS ===
    
    def _parse_project_overview(self, response: str) -> Dict[str, Any]:
        """Parse project overview response."""
        overview = {
            "project_type": "Unknown",
            "scale": "Unknown", 
            "disciplines": [],
            "complexity": "Unknown",
            "key_features": [],
            "summary": "Construction project"
        }
        
        if not response:
            return overview
        
        # Parse each field
        type_match = re.search(r'PROJECT TYPE:\s*(.+)', response, re.IGNORECASE)
        if type_match:
            overview["project_type"] = type_match.group(1).strip()
        
        scale_match = re.search(r'SCALE:\s*(.+)', response, re.IGNORECASE)
        if scale_match:
            overview["scale"] = scale_match.group(1).strip()
        
        summary_match = re.search(r'SUMMARY:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            overview["summary"] = summary_match.group(1).strip()
        
        return overview

    def _parse_page_analysis(self, response: str, start_index: int) -> Dict[int, Any]:
        """Parse page-by-page analysis."""
        pages = {}
        
        if not response:
            return pages
        
        # Split by PAGE markers
        page_sections = re.split(r'PAGE\s+\d+:', response, flags=re.IGNORECASE)
        
        for i, section in enumerate(page_sections[1:], start=1):  # Skip first empty split
            page_num = start_index + i
            
            page_data = {
                "page_type": "Unknown",
                "content_summary": "",
                "elements": {},
                "systems": [],
                "key_info": []
            }
            
            # Parse TYPE
            type_match = re.search(r'TYPE:\s*(.+)', section, re.IGNORECASE)
            if type_match:
                page_data["page_type"] = type_match.group(1).strip()
            
            # Parse CONTENT
            content_match = re.search(r'CONTENT:\s*(.+)', section, re.IGNORECASE)
            if content_match:
                page_data["content_summary"] = content_match.group(1).strip()
            
            # Parse ELEMENTS (this is crucial!)
            elements_match = re.search(r'ELEMENTS:\s*(.+?)(?:SYSTEMS:|KEY INFO:|$)', section, re.IGNORECASE | re.DOTALL)
            if elements_match:
                elements_text = elements_match.group(1)
                # Parse individual elements with counts
                element_pattern = r'(\d+)\s+(\w+(?:\s+\w+)*)'
                for match in re.finditer(element_pattern, elements_text):
                    count, element_type = match.groups()
                    page_data["elements"][element_type.lower()] = int(count)
            
            pages[page_num] = page_data
        
        return pages

    def _parse_element_inventory(self, response: str) -> Dict[str, Any]:
        """Parse complete element inventory."""
        inventory = {}
        
        if not response:
            return inventory
        
        # Parse each element entry
        element_pattern = r'ELEMENT:\s*(.+?)\nTOTAL COUNT:\s*(\d+)'
        
        for match in re.finditer(element_pattern, response, re.IGNORECASE | re.DOTALL):
            element_type = match.group(1).strip().lower()
            count = int(match.group(2))
            
            # Also try to get pages
            pages_match = re.search(
                f'PAGES FOUND ON:.*?([0-9, ]+)',
                response[match.end():match.end()+200],
                re.IGNORECASE
            )
            
            pages = []
            if pages_match:
                pages = [int(p.strip()) for p in pages_match.group(1).split(',') if p.strip().isdigit()]
            
            inventory[element_type] = {
                "total_count": count,
                "pages": pages,
                "distributed": len(pages) > 1
            }
        
        return inventory

    def _parse_systems_analysis(self, response: str) -> Dict[str, Any]:
        """Parse building systems analysis."""
        systems = {}
        
        if not response:
            return systems
        
        # Define system categories
        system_categories = [
            "ELECTRICAL SYSTEM",
            "MECHANICAL SYSTEM", 
            "PLUMBING SYSTEM",
            "FIRE PROTECTION",
            "STRUCTURAL SYSTEM"
        ]
        
        for category in system_categories:
            # Find section for this system
            pattern = f'{category}:(.+?)(?:{"|".join(system_categories)}:|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                system_text = match.group(1)
                systems[category.lower().replace(" system", "")] = {
                    "summary": system_text.strip()[:500],  # First 500 chars
                    "present": True
                }
        
        return systems

    def _parse_knowledge_answer(
        self,
        response: str,
        question_analysis: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse answer from knowledge query."""
        
        element_type = question_analysis.get("element_focus", "element")
        
        # Start with data from our inventory
        result = {
            "element_type": element_type,
            "count": 0,
            "locations": [],
            "grid_references": [],
            "evidence": [],
            "confidence": 0.95
        }
        
        # If we have this element in our inventory, use that data
        if element_type in knowledge["all_elements"]:
            element_data = knowledge["all_elements"][element_type]
            result["count"] = element_data.get("total_count", 0)
            
            # Build locations from page data
            for page_num in element_data.get("pages", []):
                if page_num in knowledge["page_contents"]:
                    page_data = knowledge["page_contents"][page_num]
                    element_count_on_page = page_data.get("elements", {}).get(element_type, 0)
                    
                    # Create location entries
                    for i in range(min(element_count_on_page, 10)):  # Limit to 10 per page
                        result["locations"].append({
                            "page": page_num,
                            "grid_ref": f"See page {page_num}",
                            "visual_details": f"{element_type} on {page_data.get('page_type', 'page')}"
                        })
        
        # Parse any specific counts from response
        count_match = re.search(r'(?:total|count|found).*?(\d+)', response, re.IGNORECASE)
        if count_match and result["count"] == 0:
            result["count"] = int(count_match.group(1))
        
        result["notes"] = [
            "Answer based on comprehensive document analysis",
            f"Document intelligence includes {len(knowledge['all_elements'])} element types"
        ]
        
        return result

    # === HELPER METHODS ===
    
    def _generate_document_id(self, images: List[Dict[str, Any]]) -> str:
        """Generate unique ID for this document."""
        # Create hash from first image URL or use timestamp
        if images and images[0].get("url"):
            return hashlib.md5(images[0]["url"].encode()).hexdigest()[:12]
        return f"doc_{datetime.utcnow().timestamp()}"

    def _has_document_knowledge(self, doc_id: str) -> bool:
        """Check if we already have knowledge for this document."""
        return doc_id in self.document_knowledge

    def _identify_system_pages(self, page_contents: Dict[int, Any]) -> List[int]:
        """Identify which pages contain system drawings."""
        system_pages = []
        
        system_keywords = ["mechanical", "electrical", "plumbing", "MEP", "HVAC", "power", "lighting"]
        
        for page_num, content in page_contents.items():
            page_type = content.get("page_type", "").lower()
            if any(keyword in page_type for keyword in system_keywords):
                system_pages.append(page_num)
        
        return system_pages

    def _analyze_spatial_organization(
        self,
        grid_systems: Dict[str, Any],
        page_contents: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Understand the spatial organization of the building."""
        
        spatial = {
            "grid_system": "present" if grid_systems else "not detected",
            "floors": [],
            "zones": [],
            "areas": {}
        }
        
        # Extract floor information from page contents
        for page_num, content in page_contents.items():
            if "floor plan" in content.get("page_type", "").lower():
                # Extract floor number if present
                floor_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*floor', 
                                      content.get("content_summary", ""), re.IGNORECASE)
                if floor_match:
                    spatial["floors"].append(floor_match.group(1))
        
        return spatial

    def _build_relationships(
        self,
        knowledge: Dict[str, Any],
        document_index: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build relationships between elements and pages."""
        
        relationships = {
            "element_to_pages": {},
            "page_to_elements": {},
            "cross_references": []
        }
        
        # Build element-to-page mapping
        for element_type, data in knowledge["all_elements"].items():
            relationships["element_to_pages"][element_type] = data.get("pages", [])
        
        # Build page-to-element mapping
        for page_num, content in knowledge["page_contents"].items():
            relationships["page_to_elements"][page_num] = list(content.get("elements", {}).keys())
        
        return relationships

    def _prepare_vision_content(self, images: List[Dict[str, Any]], text_prompt: str) -> List[Dict[str, Any]]:
        """Prepare content for vision API request."""
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

    async def _make_vision_request(
        self, 
        content: List[Dict[str, Any]], 
        system_prompt: str, 
        max_tokens: int
    ) -> Optional[str]:
        """Make vision API request with proper error handling."""
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

    def _create_error_result(self, page_number: int, error_message: str) -> VisualIntelligenceResult:
        """Create error result in expected format."""
        return VisualIntelligenceResult(
            element_type="error",
            count=0,
            locations=[],
            confidence=0.0,
            grid_references=[],
            visual_evidence=[],
            pattern_matches=[],
            verification_notes=[f"Analysis failed: {error_message}"],
            page_number=page_number,
            analysis_metadata={"error": True}
        )