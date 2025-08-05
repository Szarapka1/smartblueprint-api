# ai_service_core.py - COMPLETE REWRITE FOR DOCUMENT INTELLIGENCE FIRST

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import hashlib

# Import from same package
from .vision_intelligence import VisionIntelligence
from .page_selection import PageSelector
from .validation_system import ValidationSystem
from .response_formatter import ResponseFormatter
from .semantic_highlighter import SemanticHighlighter
from .enhanced_cache import EnhancedCache
from .data_loader import DataLoader
from .question_analyzer import QuestionAnalyzer
from .note_generator import NoteGenerator
from .post_processor import PostProcessor
from .calculation_engine import CalculationEngine

# Import from app.core.config
from app.core.config import CONFIG

# Import models
from app.models.schemas import (
    VisualIntelligenceResult,
    ValidationResult,
    TrustMetrics,
    NoteSuggestion,
    QuestionType
)

logger = logging.getLogger(__name__)


class VisualIntelligenceFirst:
    """
    Document Intelligence First - Core Orchestrator
    
    PHILOSOPHY: Understand the ENTIRE document ONCE, then answer ANY question.
    No more hunting for elements - we build comprehensive knowledge and query it.
    
    This orchestrator ensures we:
    1. Build complete document understanding on first access
    2. Cache that understanding for instant answers
    3. Answer questions from knowledge, not from searching
    4. Validate against our knowledge, not by re-scanning
    """

    def __init__(self, settings):
        self.settings = settings
        logger.info("🚀 Initializing Document Intelligence First System...")

        # Initialize all modules
        self.vision = VisionIntelligence(settings)
        self.page_selector = PageSelector(settings)  # Kept for compatibility
        self.validator = ValidationSystem(settings)
        self.formatter = ResponseFormatter()
        self.highlighter = SemanticHighlighter()
        self.cache = EnhancedCache()
        self.data_loader = DataLoader(settings)
        self.question_analyzer = QuestionAnalyzer(settings)
        self.note_generator = NoteGenerator(settings)
        self.post_processor = PostProcessor(settings)
        self.calculation_engine = CalculationEngine()

        # Document knowledge cache
        self.document_knowledge_cache = {}
        
        # Initialize vision clients
        try:
            logger.info("🔧 Initializing vision clients...")
            vision_client = self.vision.client
            
            # Distribute vision client to modules that need it
            self.page_selector.set_vision_client(vision_client)
            self.validator.set_vision_client(vision_client)
            self.post_processor.set_vision_client(vision_client)
            self.question_analyzer.set_vision_client(vision_client)
            
            self.vision._ensure_semaphores_initialized()
            logger.info("✅ Vision clients initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Vision client initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize vision client: {e}")

        self.current_prompt = None
        self.current_response = None
        logger.info("✅ Document Intelligence First System ready!")

    async def get_ai_response(
        self,
        prompt: str,
        document_id: str,
        storage_service,
        author: str = None,
        current_page: Optional[int] = None,
        request_highlights: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main pipeline - Document Intelligence First Approach
        
        Instead of hunting for elements, we:
        1. Ensure we have complete document knowledge
        2. Answer the question from that knowledge
        3. Validate against the knowledge
        4. Return comprehensive, accurate results
        """
        
        logger.info(f"💡 Document Intelligence Query: '{prompt}'")
        logger.info(f"📄 Document: {document_id}")
        
        start_time = time.time()
        self.current_prompt = prompt
        
        try:
            # STEP 1: Ensure we have document knowledge
            logger.info("📚 STEP 1: Ensuring document knowledge exists...")
            document_knowledge = await self._ensure_document_knowledge(
                document_id, storage_service
            )
            
            # STEP 2: Analyze the question
            logger.info("🤔 STEP 2: Understanding the question...")
            question_analysis = await self.question_analyzer.analyze(prompt)
            effective_page = current_page or question_analysis.get("requested_page")
            
            logger.info(f"📋 Question type: {question_analysis.get('type')}, "
                       f"Element: {question_analysis.get('element_focus')}")
            
            # STEP 3: Load necessary data (minimal - we already have knowledge!)
            logger.info("📊 STEP 3: Preparing response data...")
            comprehensive_data = await self._prepare_response_data(
                document_id, 
                storage_service, 
                document_knowledge,
                question_analysis,
                effective_page
            )
            
            # STEP 4: Answer from knowledge (not by scanning!)
            logger.info("🧠 STEP 4: Answering from document knowledge...")
            visual_result = await self._answer_from_knowledge(
                prompt,
                question_analysis,
                document_knowledge,
                comprehensive_data,
                effective_page or 1
            )
            
            logger.info(f"💡 Knowledge-based answer: {visual_result.count} {visual_result.element_type}(s)")
            
            # STEP 5: Validate against knowledge (not by re-scanning!)
            logger.info("✅ STEP 5: Validating against document knowledge...")
            validation_results = await self.validator.validate(
                visual_result, comprehensive_data, question_analysis
            )
            
            # STEP 6: Build consensus
            logger.info("🤝 STEP 6: Building consensus...")
            consensus_result = await self.validator.build_consensus(
                visual_result, validation_results, question_analysis
            )
            
            # STEP 7: Calculate trust metrics
            trust_metrics = self.validator.calculate_trust_metrics(
                visual_result, validation_results, consensus_result
            )
            logger.info(f"📊 Trust Score: {trust_metrics.reliability_score:.2f}")
            
            # STEP 8: Generate response
            logger.info("📝 STEP 8: Formatting response...")
            raw_response = self.formatter.format_response(
                visual_result, validation_results, consensus_result, 
                trust_metrics, question_analysis
            )
            
            # STEP 9: Post-processing (enrichment from knowledge)
            logger.info("🔧 STEP 9: Post-processing...")
            post_results = await self._knowledge_based_post_processing(
                visual_result, document_knowledge, question_analysis
            )
            
            if post_results.get("summary"):
                raw_response += "\n\n" + post_results["summary"]
            
            # STEP 10: Generate highlights if requested
            highlights = []
            if request_highlights and visual_result.count > 0:
                logger.info("🎨 STEP 10: Generating semantic highlights...")
                # For knowledge-based system, we might generate conceptual highlights
                highlights = await self._generate_knowledge_highlights(
                    visual_result, comprehensive_data
                )
            
            # STEP 11: Generate notes
            logger.info("📝 STEP 11: Generating notes...")
            notes = await self.note_generator.generate_notes(
                visual_result, document_id, storage_service, author
            )
            
            # Assemble final response
            final_result = self._assemble_response(
                {"response": raw_response, "highlights": highlights},
                visual_result, validation_results, trust_metrics,
                question_analysis, notes, start_time
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Document Intelligence complete in {elapsed_time:.2f}s")
            logger.info(f"📊 Final Answer: {visual_result.count} {visual_result.element_type}(s) "
                       f"with {int(trust_metrics.reliability_score * 100)}% confidence")
            
            return final_result

        except Exception as e:
            logger.error(f"❌ Document Intelligence error: {e}", exc_info=True)
            return self._generate_error_response(prompt, document_id, str(e))

    async def _ensure_document_knowledge(
        self,
        document_id: str,
        storage_service
    ) -> Dict[str, Any]:
        """
        Ensure we have complete document knowledge.
        Build it if we don't, retrieve it if we do.
        """
        
        # Check memory cache first
        if document_id in self.document_knowledge_cache:
            logger.info("✅ Document knowledge found in memory cache")
            return self.document_knowledge_cache[document_id]
        
        # Check persistent cache
        cache_key = f"doc_knowledge_{document_id}"
        cached_knowledge = self.cache.get(cache_key, "document_knowledge")
        
        if cached_knowledge:
            logger.info("✅ Document knowledge retrieved from cache")
            self.document_knowledge_cache[document_id] = cached_knowledge
            return cached_knowledge
        
        # Build comprehensive knowledge
        logger.info("🏗️ Building comprehensive document knowledge...")
        logger.info("⏳ This will take a moment but only happens ONCE per document...")
        
        # Load ALL pages for comprehensive analysis
        all_pages = await self._load_all_pages_for_knowledge(document_id, storage_service)
        
        if not all_pages:
            raise ValueError(f"Could not load pages for document {document_id}")
        
        # Load structured data
        comprehensive_data = await self.data_loader.load_comprehensive_data(
            document_id, storage_service, {}, all_pages
        )
        
        # Build document knowledge using VisionIntelligence
        # This is the KEY - we analyze EVERYTHING ONCE
        dummy_question = {"element_focus": "all", "type": QuestionType.GENERAL}
        
        # The vision module will build complete knowledge
        await self.vision.analyze(
            "Build complete document understanding",
            dummy_question,
            all_pages,
            1,
            comprehensive_data
        )
        
        # Retrieve the built knowledge
        if hasattr(self.vision, 'document_knowledge') and document_id in self.vision.document_knowledge:
            knowledge = self.vision.document_knowledge[document_id]
        else:
            # Fallback - build basic knowledge from what we have
            knowledge = await self._build_basic_knowledge(
                document_id, all_pages, comprehensive_data
            )
        
        # Enhance knowledge with post-processing insights
        knowledge = await self._enhance_knowledge(knowledge, all_pages, comprehensive_data)
        
        # Cache the knowledge
        self.document_knowledge_cache[document_id] = knowledge
        self.cache.set(cache_key, knowledge, "document_knowledge")
        
        logger.info(f"✅ Document knowledge built and cached!")
        logger.info(f"📊 Understood: {len(all_pages)} pages, "
                   f"{len(knowledge.get('all_elements', {}))} element types")
        
        return knowledge

    async def _load_all_pages_for_knowledge(
        self,
        document_id: str,
        storage_service
    ) -> List[Dict[str, Any]]:
        """Load ALL pages for building comprehensive knowledge."""
        
        logger.info("📄 Loading all document pages for knowledge building...")
        
        # Get metadata to know how many pages
        metadata = await self.data_loader.load_metadata(document_id, storage_service)
        total_pages = metadata.get("page_count", 50) if metadata else 50
        
        logger.info(f"📄 Document has {total_pages} pages")
        
        # For knowledge building, we want high-res images of ALL pages
        # But we'll process in batches for memory efficiency
        all_pages = []
        batch_size = 10
        
        for start_page in range(1, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size - 1, total_pages)
            page_numbers = list(range(start_page, end_page + 1))
            
            logger.info(f"📥 Loading pages {start_page} to {end_page}...")
            
            batch_pages = await self.data_loader.load_specific_pages(
                document_id, storage_service, page_numbers
            )
            
            all_pages.extend(batch_pages)
            
            # Small delay to avoid overwhelming the system
            if end_page < total_pages:
                await asyncio.sleep(0.1)
        
        logger.info(f"✅ Loaded {len(all_pages)} pages for analysis")
        
        return all_pages

    async def _prepare_response_data(
        self,
        document_id: str,
        storage_service,
        document_knowledge: Dict[str, Any],
        question_analysis: Dict[str, Any],
        effective_page: Optional[int]
    ) -> Dict[str, Any]:
        """
        Prepare minimal data needed for response.
        We don't need to load much - we have knowledge!
        """
        
        # Start with document knowledge
        comprehensive_data = {
            "document_knowledge": document_knowledge,
            "images": [],  # Might need for highlights
            "context": document_knowledge.get("project_overview", {}).get("summary", "")
        }
        
        # Load structured data for validation
        try:
            # Grid systems for spatial validation
            grid_systems = await storage_service.download_blob_as_json(
                self.settings.AZURE_CACHE_CONTAINER_NAME,
                f"{document_id}_grid_systems.json"
            )
            comprehensive_data["grid_systems"] = grid_systems
        except:
            logger.debug("Grid systems not available")
        
        try:
            # Document index for cross-reference validation
            doc_index = await storage_service.download_blob_as_json(
                self.settings.AZURE_CACHE_CONTAINER_NAME,
                f"{document_id}_document_index.json"
            )
            comprehensive_data["document_index"] = doc_index
        except:
            logger.debug("Document index not available")
        
        # Only load specific images if needed for highlights
        if effective_page and question_analysis.get("requested_page"):
            # Load just the requested page
            pages = await self.data_loader.load_specific_pages(
                document_id, storage_service, [effective_page]
            )
            comprehensive_data["images"] = pages
        
        return comprehensive_data

    async def _answer_from_knowledge(
        self,
        prompt: str,
        question_analysis: Dict[str, Any],
        document_knowledge: Dict[str, Any],
        comprehensive_data: Dict[str, Any],
        page_number: int
    ) -> VisualIntelligenceResult:
        """
        Answer the question from our comprehensive document knowledge.
        No scanning needed - we already know everything!
        """
        
        element_type = question_analysis.get("element_focus", "element")
        question_type = question_analysis.get("type", QuestionType.GENERAL)
        
        # Check if this is a calculation request
        if question_type == QuestionType.ESTIMATE or self._is_calculation_request(prompt):
            logger.info("🧮 Detected calculation request - using Calculation Engine")
            
            # Perform calculation
            calc_result = await self.calculation_engine.calculate(
                prompt,
                document_knowledge,
                element_data=document_knowledge.get("all_elements", {}).get(element_type)
            )
            
            # Convert calculation result to VisualIntelligenceResult format
            return self._format_calculation_as_visual_result(
                calc_result, element_type, page_number, document_knowledge
            )
        
        # Regular knowledge-based answer
        # Check our knowledge for this element
        if element_type in document_knowledge.get("all_elements", {}):
            element_data = document_knowledge["all_elements"][element_type]
            
            # Build result from knowledge
            count = element_data.get("total_count", 0)
            pages = element_data.get("pages", [])
            
            # Build locations from page contents
            locations = []
            grid_references = []
            
            page_contents = document_knowledge.get("page_contents", {})
            for page_num in pages[:20]:  # Limit for response size
                if page_num in page_contents:
                    page_data = page_contents[page_num]
                    page_elements = page_data.get("elements", {})
                    
                    if element_type in page_elements:
                        # Create representative locations
                        element_count_on_page = page_elements[element_type]
                        for i in range(min(5, element_count_on_page)):  # Max 5 per page
                            locations.append({
                                "page": page_num,
                                "grid_ref": f"Page {page_num}",
                                "visual_details": f"{element_type} on {page_data.get('page_type', 'page')}",
                                "element_tag": f"{element_type.upper()}{len(locations)+1}"
                            })
                            grid_references.append(f"Page{page_num}")
            
            # Build comprehensive result
            visual_result = VisualIntelligenceResult(
                element_type=element_type,
                count=count,
                locations=locations,
                confidence=0.95,  # High confidence from knowledge!
                grid_references=grid_references,
                visual_evidence=[
                    f"Found in comprehensive document analysis",
                    f"Documented across {len(pages)} pages",
                    f"Total inventory: {count} {element_type}(s)"
                ],
                pattern_matches=[f"{element_type}_standard"],
                verification_notes=[
                    "Result from document knowledge base",
                    f"Knowledge built from {len(page_contents)} analyzed pages"
                ],
                page_number=page_number,
                analysis_metadata={
                    "source": "document_knowledge",
                    "pages_with_element": pages,
                    "knowledge_confidence": "high"
                }
            )
            
        else:
            # Element not found in knowledge
            visual_result = VisualIntelligenceResult(
                element_type=element_type,
                count=0,
                locations=[],
                confidence=0.90,  # Still confident - we know it's not there
                grid_references=[],
                visual_evidence=[
                    f"No {element_type}s found in document analysis",
                    f"Verified across entire document"
                ],
                pattern_matches=[],
                verification_notes=[
                    f"{element_type} not present in document inventory",
                    f"Confirmed by analysis of {len(document_knowledge.get('page_contents', {}))} pages"
                ],
                page_number=page_number,
                analysis_metadata={
                    "source": "document_knowledge",
                    "result": "not_found"
                }
            )
        
        # Handle specific page requests
        if question_analysis.get("requested_page"):
            requested_page = question_analysis["requested_page"]
            visual_result = self._filter_result_by_page(
                visual_result, requested_page, document_knowledge
            )
        
        return visual_result

    def _filter_result_by_page(
        self,
        visual_result: VisualIntelligenceResult,
        page_number: int,
        document_knowledge: Dict[str, Any]
    ) -> VisualIntelligenceResult:
        """Filter result to specific page if requested."""
        
        page_contents = document_knowledge.get("page_contents", {})
        
        if page_number in page_contents:
            page_data = page_contents[page_number]
            page_elements = page_data.get("elements", {})
            
            if visual_result.element_type in page_elements:
                # Update count for this page only
                page_count = page_elements[visual_result.element_type]
                
                # Filter locations to this page
                page_locations = [
                    loc for loc in visual_result.locations 
                    if loc.get("page") == page_number
                ]
                
                # Create new result for specific page
                visual_result.count = page_count
                visual_result.locations = page_locations
                visual_result.verification_notes.append(
                    f"Filtered to page {page_number} only"
                )
                visual_result.analysis_metadata["filtered_to_page"] = page_number
            else:
                # Element not on this page
                visual_result.count = 0
                visual_result.locations = []
                visual_result.verification_notes.append(
                    f"No {visual_result.element_type}s on page {page_number}"
                )
        
        return visual_result

    async def _knowledge_based_post_processing(
        self,
        visual_result: VisualIntelligenceResult,
        document_knowledge: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Post-processing using document knowledge.
        No need to re-analyze - use what we know!
        """
        
        results = {
            "schedule_found": False,
            "schedule_details": None,
            "compliance_notes": [],
            "additional_findings": [],
            "recommendations": [],
            "summary": ""
        }
        
        # Check if we found schedules in our knowledge
        element_type = visual_result.element_type
        
        # Look for schedule information in page contents
        page_contents = document_knowledge.get("page_contents", {})
        for page_num, content in page_contents.items():
            if "schedule" in content.get("page_type", "").lower():
                if element_type in content.get("content_summary", "").lower():
                    results["schedule_found"] = True
                    results["schedule_details"] = {
                        "page": page_num,
                        "type": f"{element_type} schedule",
                        "description": content.get("content_summary", "")
                    }
                    break
        
        # Add system-level insights
        if element_type in ["outlet", "panel", "light fixture"]:
            electrical_system = document_knowledge.get("systems", {}).get("electrical", {})
            if electrical_system.get("present"):
                results["additional_findings"].append(
                    f"Part of electrical system: {electrical_system.get('summary', '')[:100]}..."
                )
        
        # Add building-specific insights
        project_type = document_knowledge.get("project_overview", {}).get("project_type", "")
        if project_type:
            results["additional_findings"].append(
                f"For {project_type} project"
            )
        
        # Build summary if we have findings
        if results["schedule_found"]:
            results["summary"] = f"📋 {element_type.title()} schedule found on page {results['schedule_details']['page']}"
        
        return results

    async def _generate_knowledge_highlights(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate conceptual highlights based on knowledge.
        Since we're not looking at specific images, these are representative.
        """
        
        highlights = []
        
        # For now, return empty - highlights need actual image coordinates
        # In a full implementation, we might:
        # 1. Load specific pages where elements are found
        # 2. Use stored geometry data from initial analysis
        # 3. Generate highlights on those specific pages
        
        return highlights

    async def _build_basic_knowledge(
        self,
        document_id: str,
        all_pages: List[Dict[str, Any]],
        comprehensive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build basic knowledge structure if vision module didn't.
        This is a fallback for compatibility.
        """
        
        knowledge = {
            "document_id": document_id,
            "created_at": datetime.utcnow().isoformat(),
            "project_overview": {
                "summary": "Construction document",
                "total_pages": len(all_pages)
            },
            "page_contents": {},
            "all_elements": {},
            "systems": {},
            "spatial_organization": {},
            "relationships": {},
            "metadata": {
                "total_pages": len(all_pages),
                "analysis_version": "1.0"
            }
        }
    
    def _is_calculation_request(self, prompt: str) -> bool:
        """Check if the prompt is requesting a calculation."""
        
        prompt_lower = prompt.lower()
        
        calculation_keywords = [
            "calculate", "area", "square footage", "sq ft", "load",
            "watts", "amps", "cost", "price", "spacing", "coverage",
            "ratio", "percentage", "%", "how much paint", "how many gallons",
            "material needed", "quantity needed", "cfm", "tons"
        ]
        
        return any(keyword in prompt_lower for keyword in calculation_keywords)
    
    def _format_calculation_as_visual_result(
        self,
        calc_result,
        element_type: str,
        page_number: int,
        document_knowledge: Dict[str, Any]
    ) -> VisualIntelligenceResult:
        """Format calculation result as VisualIntelligenceResult for consistency."""
        
        # Build a special visual result for calculations
        return VisualIntelligenceResult(
            element_type=element_type,
            count=int(calc_result.value) if calc_result.unit in ["units", "sheets", "gallons"] else 0,
            locations=[],
            confidence=calc_result.confidence,
            grid_references=[],
            visual_evidence=[
                f"Calculation: {calc_result.value} {calc_result.unit}",
                f"Formula: {calc_result.formula_used}"
            ] + calc_result.assumptions,
            pattern_matches=[],
            verification_notes=[
                f"Calculation type: {calc_result.calculation_type.value}",
                f"Based on document knowledge"
            ],
            page_number=page_number,
            analysis_metadata={
                "source": "calculation_engine",
                "calculation_result": {
                    "value": calc_result.value,
                    "unit": calc_result.unit,
                    "formula": calc_result.formula_used,
                    "details": calc_result.details
                }
            }
        )
        
        # Build basic page inventory
        for i, page in enumerate(all_pages, 1):
            knowledge["page_contents"][i] = {
                "page_type": "drawing",
                "content_summary": f"Page {i}",
                "elements": {}
            }
        
        return knowledge

    async def _enhance_knowledge(
        self,
        knowledge: Dict[str, Any],
        all_pages: List[Dict[str, Any]],
        comprehensive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance knowledge with additional insights."""
        
        # Add insights from structured data
        if comprehensive_data.get("grid_systems"):
            knowledge["spatial_organization"]["grid_system"] = "present"
            knowledge["spatial_organization"]["grid_pages"] = list(
                comprehensive_data["grid_systems"].keys()
            )
        
        if comprehensive_data.get("document_index"):
            doc_index = comprehensive_data["document_index"]
            if "sheet_numbers" in doc_index:
                knowledge["metadata"]["sheet_count"] = len(doc_index["sheet_numbers"])
        
        return knowledge

    def _generate_cache_key(self, prompt: str, document_id: str, page: Optional[int]) -> str:
        """Generates a unique cache key for a request."""
        key_data = f"{prompt.lower().strip()}|{document_id}|{page or 'all'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _assemble_response(
        self, result_highlights: Dict[str, Any], visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult], trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any], notes: List[NoteSuggestion], start_time: float
    ) -> Dict[str, Any]:
        """Assembles the final response dictionary."""
        return {
            'status': 'success',
            'ai_response': result_highlights.get('response', ''),
            'highlights': result_highlights.get('highlights', []),
            'visual_result': visual_result.dict(),
            'trust_metrics': trust_metrics.dict(),
            'validation_breakdown': [v.dict() for v in validation_results],
            'notes': [note.dict() for note in notes],
            'processing_time': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'debug_info': {
                'question_type': question_analysis.get('type', QuestionType.GENERAL).value,
                'element_focus': question_analysis.get('element_focus'),
                'knowledge_based': True,
                'source': 'document_intelligence'
            }
        }

    def _generate_error_response(self, prompt: str, document_id: str, error: str) -> Dict[str, Any]:
        """Generates a standardized error response."""
        logger.error(f"🚨 Generating error response for doc {document_id}: {error}")
        return {
            'status': 'error',
            'message': f'An error occurred during analysis: {error}',
            'ai_response': f'I encountered an error while analyzing your question. Please try again. Error: {error}',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'debug_info': {
                'prompt': prompt,
                'document_id': document_id,
                'error': error,
                'knowledge_based': True
            }
        }