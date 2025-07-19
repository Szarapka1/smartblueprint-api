# ai_service_core.py - REVISED FOR ACCURACY AND DATA INTEGRATION

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
    Visual Intelligence First Construction AI - Core Orchestrator.
    REVISED to integrate structured data (grid systems, document index)
    into the full analysis pipeline for enhanced accuracy.
    """

    def __init__(self, settings):
        self.settings = settings
        logger.info("🔧 Initializing VisualIntelligenceFirst modules...")

        # Initialize all modules
        self.vision = VisionIntelligence(settings)
        self.page_selector = PageSelector(settings)
        self.validator = ValidationSystem(settings)
        self.formatter = ResponseFormatter()
        self.highlighter = SemanticHighlighter()
        self.cache = EnhancedCache()
        self.data_loader = DataLoader(settings)
        self.question_analyzer = QuestionAnalyzer(settings)
        self.note_generator = NoteGenerator(settings)
        self.post_processor = PostProcessor(settings)

        try:
            logger.info("🔧 Initializing and distributing OpenAI vision client...")
            vision_client = self.vision.client  # Trigger lazy initialization
            self.page_selector.set_vision_client(vision_client)
            self.validator.set_vision_client(vision_client)
            self.post_processor.set_vision_client(vision_client)
            self.question_analyzer.set_vision_client(vision_client)
            self.vision._ensure_semaphores_initialized()
            logger.info("✅ Vision client initialized and distributed successfully.")
        except Exception as e:
            logger.error(f"❌ Vision client initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize vision client: {e}")

        self.current_prompt = None
        self.current_response = None
        logger.info("✅ VisualIntelligenceFirst initialized successfully.")

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
        Main pipeline following the vision-first approach.
        REVISED to load and utilize structured ground-truth data.
        """
        logger.info(f"🚀 Starting AI analysis for question: '{prompt}'")
        logger.info(f"📄 Document ID: {document_id}, Current page: {current_page}")

        start_time = time.time()
        self.current_prompt = prompt

        try:
            # STEP 1: Understand the question
            logger.info("🧠 STEP 1: Understanding the question")
            question_analysis = await self.question_analyzer.analyze(prompt)
            effective_page = current_page or question_analysis.get("requested_page")

            logger.info(f"📋 Question analysis: Type={question_analysis.get('type')}, Element={question_analysis.get('element_focus')}")

            # STEP 2: Load thumbnails and metadata
            logger.info("📸 STEP 2: Loading thumbnails and metadata")
            metadata = await self.data_loader.load_metadata(document_id, storage_service)
            total_pages = metadata.get("page_count") if metadata else 10 # Fallback
            logger.info(f"📄 Metadata indicates {total_pages} pages in document.")
            
            thumbnails = await self.data_loader.load_all_thumbnails(document_id, storage_service, total_pages)
            logger.info(f"✅ Loaded {len(thumbnails)} thumbnails for analysis.")

            # STEP 3: Select relevant pages using vision
            logger.info("🔍 STEP 3: Analyzing thumbnails to select pages")
            selected_pages = await self.page_selector.select_relevant_pages(thumbnails, question_analysis, current_page)
            logger.warning(f"📍 Selected pages for detailed analysis: {selected_pages}")

            if not selected_pages:
                logger.error("❌ CRITICAL: No pages were selected for analysis. Falling back to page 1.")
                selected_pages = [1]

            # STEP 4: Load high-res images for selected pages
            logger.info(f"📄 STEP 4: Loading {len(selected_pages)} high-res pages")
            images = await self.data_loader.load_specific_pages(document_id, storage_service, selected_pages)
            logger.warning(f"📸 Loaded {len(images)} high-res images")

            if not images:
                return self._generate_error_response(prompt, document_id, "No images could be loaded for analysis.")

            # STEP 5: Load COMPREHENSIVE data (including structured JSON files)
            logger.info("📊 STEP 5: Loading comprehensive data (Text, Grids, Index)")
            # This is the key change: comprehensive_data will now contain the contents of
            # _grid_systems.json and _document_index.json for the AI to use.
            comprehensive_data = await self.data_loader.load_comprehensive_data(
                document_id, storage_service, question_analysis, images
            )

            # New detailed logging to confirm structured data is loaded
            grid_count = len(comprehensive_data.get('grid_systems', {}))
            index_page_count = len(comprehensive_data.get('document_index', {}).get('page_index', {}))
            logger.info(f"📊 Comprehensive data loaded: {len(images)} images, "
                        f"{len(comprehensive_data.get('context', ''))} chars, "
                        f"{grid_count} grid systems, "
                        f"{index_page_count} indexed pages.")

            # STEP 6: Visual intelligence analysis (Now with structured data context)
            logger.info("👁️ STEP 6: Visual intelligence analysis (Scan phase)")
            logger.info(f"🎯 Analyzing for: {question_analysis.get('element_focus')} elements")
            visual_result = await self.vision.analyze(
                prompt, question_analysis, images, effective_page or 1, comprehensive_data
            )
            logger.info(f"👁️ Initial visual scan found: {visual_result.count} {visual_result.element_type}(s)")

            # STEP 7: 4x Validation (Now with structured data for verification)
            logger.info("🛡️ STEP 7: 4x Validation (Validate phase)")
            validation_results = await self.validator.validate(
                visual_result, comprehensive_data, question_analysis
            )
            high_conf_validations = sum(1 for v in validation_results if v.confidence >= 0.90)
            logger.info(f"✅ Validation complete: {high_conf_validations}/{len(validation_results)} high confidence checks.")

            # STEP 8: Build consensus from all findings
            logger.info("🤝 STEP 8: Building consensus")
            consensus_result = await self.validator.build_consensus(visual_result, validation_results, question_analysis)
            logger.info(f"🤝 Consensus result: {consensus_result.get('validation_agreement')}")

            # STEP 9: Calculate final trust metrics
            trust_metrics = self.validator.calculate_trust_metrics(visual_result, validation_results, consensus_result)
            logger.info(f"📊 Final Trust Metrics: Reliability={trust_metrics.reliability_score:.2f}, Perfect Accuracy={trust_metrics.perfect_accuracy_achieved}")

            # STEP 10: Generate response text
            logger.info("📝 STEP 10: Generating response")
            raw_response = self.formatter.format_response(
                visual_result, validation_results, consensus_result, trust_metrics, question_analysis
            )

            # STEP 11: Post-processing (e.g., schedule detection)
            logger.info("🔧 STEP 11: Post-processing")
            post_results = await self.post_processor.process(visual_result, images, comprehensive_data, question_analysis)
            if post_results.get("summary"):
                raw_response += "\n\n" + post_results["summary"]
                logger.info("✅ Added post-processing summary to response.")
            
            result_highlights = {"response": raw_response} # Placeholder for now

            # STEP 12: Note generation
            logger.info("📝 STEP 12: Generating notes")
            notes = await self.note_generator.generate_notes(visual_result, document_id, storage_service, author)
            logger.info(f"✅ Generated {len(notes)} notes.")

            # Assemble and return the final, comprehensive result
            final_result = self._assemble_response(
                result_highlights, visual_result, validation_results,
                trust_metrics, question_analysis, notes, start_time
            )

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Analysis complete in {elapsed_time:.2f}s. "
                        f"Final Count: {visual_result.count} {visual_result.element_type}(s) "
                        f"with {int(trust_metrics.reliability_score * 100)}% confidence.")

            return final_result

        except Exception as e:
            logger.error(f"❌ Unhandled pipeline error: {e}", exc_info=True)
            return self._generate_error_response(prompt, document_id, str(e))

    def _generate_cache_key(self, prompt: str, document_id: str, page: Optional[int]) -> str:
        """Generates a unique cache key for a request."""
        key_data = f"{prompt.lower().strip()}|{document_id}|{page or 'all'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _assemble_response(
        self, result_highlights: Dict[str, Any], visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult], trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any], notes: List[NoteSuggestion], start_time: float
    ) -> Dict[str, Any]:
        """Assembles the final, detailed response dictionary."""
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
                'question_type': getattr(question_analysis.get('type'), 'value', 'unknown'),
                'element_focus': question_analysis.get('element_focus'),
                'pages_analyzed': visual_result.analysis_metadata.get("analyzed_pages", [])
            }
        }

    def _generate_error_response(self, prompt: str, document_id: str, error: str) -> Dict[str, Any]:
        """Generates a standardized error response."""
        logger.error(f"🚨 Generating error response for doc {document_id}: {error}")
        return {
            'status': 'error',
            'message': f'An unexpected error occurred during analysis: {error}',
            'ai_response': f'I encountered an error while analyzing your question. Please try again or rephrase. Error: {error}',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'debug_info': {'prompt': prompt, 'document_id': document_id, 'error': error}
        }