# ai_service_core.py
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Import from same package (relative imports are correct)
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

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG

# Add model imports that might be needed
from app.models.schemas import (
    VisualIntelligenceResult,
    ValidationResult,
    TrustMetrics,
    NoteSuggestion,
    QuestionType
)

logger = logging.getLogger(__name__)

class VisualIntelligenceFirst:
    """Visual Intelligence First Construction AI - Core Orchestrator"""
    
    def __init__(self, settings):
        self.settings = settings
        
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
        
        # FIXED: Set vision client for all modules that need it
        vision_client = self.vision.client  # This will lazy-initialize the client
        self.page_selector.set_vision_client(vision_client)
        self.validator.set_vision_client(vision_client)
        self.post_processor.set_vision_client(vision_client)
        self.question_analyzer.set_vision_client(vision_client)
        
        self.current_prompt = None
        self.current_response = None
    
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
        Main pipeline following the vision-first approach:
        1. Understand the question
        2. Load thumbnails (with metadata check)
        3. Analyze thumbnails to select pages
        4. Load high-res pages
        5. Visual analysis with intelligence
        6. 4x validation
        7. Post-processing
        8. Highlights if requested
        9. Note generation
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
            
            logger.info(f"📋 Question analysis: Type={question_analysis.get('type')}, "
                       f"Element={question_analysis.get('element_focus')}, "
                       f"Wants total={question_analysis.get('wants_total')}")
            
            # Check cache
            cache_key = self._generate_cache_key(prompt, document_id, effective_page)
            if CONFIG["aggressive_caching"]:
                cached_result = self.cache.get(cache_key, "analysis")
                if cached_result:
                    logger.info("✅ Returning cached analysis response.")
                    return cached_result
            
            # STEP 2: Load and analyze thumbnails
            logger.info("📸 STEP 2: Loading thumbnails and metadata")
            metadata = await self.data_loader.load_metadata(document_id, storage_service)
            total_pages = metadata.get("page_count") if metadata else None
            
            if total_pages:
                logger.info(f"📄 Metadata indicates {total_pages} pages in document")
            else:
                logger.warning("⚠️ No metadata found - will probe for pages")
            
            thumbnails = await self.data_loader.load_all_thumbnails(
                document_id, storage_service, total_pages
            )
            
            logger.info(f"✅ Loaded {len(thumbnails)} thumbnails for analysis")
            
            # Debug: List first few thumbnails
            if thumbnails:
                logger.debug(f"📸 First 3 thumbnails: Pages {[t['page'] for t in thumbnails[:3]]}")
            
            # STEP 3: Select relevant pages using vision
            logger.info("🔍 STEP 3: Analyzing thumbnails to select pages")
            selected_pages = await self.page_selector.select_relevant_pages(
                thumbnails, question_analysis, current_page
            )
            
            logger.warning(f"📍 Selected pages: {selected_pages}")
            
            # Critical: Check if pages were selected
            if not selected_pages:
                logger.error("❌ No pages selected! Using all thumbnail pages")
                selected_pages = [thumb["page"] for thumb in thumbnails]
                logger.warning(f"📄 Using all {len(selected_pages)} pages: {selected_pages}")
            
            # Ensure we have valid page numbers
            if not selected_pages:
                logger.error("❌ CRITICAL: No pages available!")
                selected_pages = [1]  # Fallback to page 1
            
            # STEP 4: Load high-res pages
            logger.info(f"📄 STEP 4: Loading {len(selected_pages)} high-res pages")
            images = await self.data_loader.load_specific_pages(
                document_id, storage_service, selected_pages
            )
            
            logger.warning(f"📸 Loaded {len(images)} high-res images")
            
            # Check if we got the images we requested
            if len(images) < len(selected_pages):
                logger.warning(f"⚠️ Only loaded {len(images)} of {len(selected_pages)} requested pages")
                loaded_pages = [img["page"] for img in images]
                logger.warning(f"📄 Loaded pages: {loaded_pages}")
            
            # Critical: Ensure we have at least some images
            if not images:
                logger.error("❌ CRITICAL: No high-res images loaded!")
                # Try to use thumbnails as fallback
                if thumbnails:
                    logger.warning("🔄 Falling back to thumbnails for analysis")
                    images = thumbnails[:5]  # Use first 5 thumbnails
                else:
                    return self._generate_error_response(
                        prompt, document_id, "No images could be loaded for analysis"
                    )
            
            # STEP 5: Load comprehensive data
            logger.info("📊 STEP 5: Loading comprehensive data")
            comprehensive_data = await self.data_loader.load_comprehensive_data(
                document_id, storage_service, question_analysis, images
            )
            
            # Log comprehensive data summary
            logger.info(f"📊 Comprehensive data: {len(comprehensive_data.get('images', []))} images, "
                       f"Context length: {len(comprehensive_data.get('context', ''))} chars")
            
            # STEP 6: Visual intelligence analysis
            logger.info("👁️ STEP 6: Visual intelligence analysis")
            logger.info(f"🎯 Analyzing for: {question_analysis.get('element_focus')} elements")
            
            visual_result = await self.vision.analyze(
                prompt, question_analysis, images, 
                effective_page or 1, comprehensive_data
            )
            
            logger.info(f"👁️ Visual analysis found: {visual_result.count} {visual_result.element_type}(s)")
            
            # STEP 7: 4x Validation
            logger.info("🛡️ STEP 7: 4x Validation")
            validation_results = await self.validator.validate(
                visual_result, comprehensive_data, question_analysis
            )
            
            # Log validation summary
            high_conf_validations = sum(1 for v in validation_results if v.confidence >= 0.90)
            logger.info(f"✅ Validation complete: {high_conf_validations}/{len(validation_results)} high confidence")
            
            # STEP 8: Consensus
            logger.info("🤝 STEP 8: Building consensus")
            consensus_result = await self.validator.build_consensus(
                visual_result, validation_results, question_analysis
            )
            
            logger.info(f"🤝 Consensus: {consensus_result.get('validation_agreement')}")
            
            # STEP 9: Trust metrics
            trust_metrics = self.validator.calculate_trust_metrics(
                visual_result, validation_results, consensus_result
            )
            
            logger.info(f"📊 Trust metrics: Reliability={trust_metrics.reliability_score:.2f}, "
                       f"Perfect accuracy={trust_metrics.perfect_accuracy_achieved}")
            
            # STEP 10: Generate response
            logger.info("📝 STEP 10: Generating response")
            raw_response = self.formatter.format_response(
                visual_result, validation_results, consensus_result,
                trust_metrics, question_analysis
            )
            self.current_response = raw_response
            
            # STEP 11: Post-processing
            logger.info("🔧 STEP 11: Post-processing")
            post_results = await self.post_processor.process(
                visual_result, images, comprehensive_data, question_analysis
            )
            
            if post_results.get("summary"):
                raw_response += "\n\n" + post_results["summary"]
                logger.info("✅ Added post-processing summary")
            
            # STEP 12: Highlights (if requested)
            result_highlights = {"response": raw_response}
            if request_highlights and prompt.lower().startswith("highlight"):
                logger.info("🎨 STEP 12: Generating semantic highlights")
                highlights = await self.highlighter.generate_highlights(
                    visual_result, images, comprehensive_data
                )
                result_highlights["highlights"] = highlights
                logger.info(f"✅ Generated {len(highlights)} highlights")
            
            # STEP 13: Note generation (always)
            logger.info("📝 STEP 13: Generating notes")
            notes = await self.note_generator.generate_notes(
                visual_result, document_id, storage_service, author
            )
            logger.info(f"✅ Generated {len(notes)} notes")
            
            # Assemble final response
            final_result = self._assemble_response(
                result_highlights, visual_result, validation_results,
                trust_metrics, question_analysis, notes, start_time
            )
            
            # Cache if high confidence
            if trust_metrics.reliability_score > 0.85:
                self.cache.set(cache_key, final_result, "analysis")
                logger.info("💾 Cached high-confidence result")
            
            # Final summary log
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Analysis complete in {elapsed_time:.2f}s: "
                       f"Found {visual_result.count} {visual_result.element_type}(s) "
                       f"with {int(trust_metrics.reliability_score * 100)}% confidence")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            return self._generate_error_response(prompt, document_id, str(e))
    
    def _generate_cache_key(self, prompt: str, document_id: str, page: Optional[int]) -> str:
        import hashlib
        key_data = f"{prompt.lower().strip()}|{document_id}|{page or 'all'}"
        return hashlib.md5(key_data.encode()).hexdigest()[:32]
    
    def _assemble_response(
        self,
        result_highlights: Dict[str, Any],
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any],
        notes: List[NoteSuggestion],
        start_time: float
    ) -> Dict[str, Any]:
        """Assemble the final response"""
        return {
            'status': 'success',
            'ai_response': result_highlights.get('response', ''),
            'highlights': result_highlights.get('highlights', []),
            'visual_result': {
                'element_type': visual_result.element_type,
                'count': visual_result.count,
                'confidence': visual_result.confidence,
                'locations': visual_result.locations,
                'pages_analyzed': len(visual_result.locations) if hasattr(visual_result, 'locations') else 0
            },
            'trust_metrics': {
                'reliability_score': trust_metrics.reliability_score,
                'perfect_accuracy': trust_metrics.perfect_accuracy_achieved,
                'validation_consensus': trust_metrics.validation_consensus
            },
            'notes': [note.dict() for note in notes],
            'processing_time': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat(),
            'debug_info': {
                'question_type': question_analysis.get('type').value if question_analysis.get('type') else 'unknown',
                'element_focus': question_analysis.get('element_focus'),
                'pages_analyzed': visual_result.page_number if hasattr(visual_result, 'page_number') else 'multiple'
            }
        }
    
    def _generate_error_response(self, prompt: str, document_id: str, error: str) -> Dict[str, Any]:
        logger.error(f"🚨 Generating error response: {error}")
        return {
            'status': 'error',
            'message': f'⚠️ Analysis failed: {error}',
            'ai_response': f'I encountered an error while analyzing your question: {error}. Please try again or rephrase your question.',
            'timestamp': datetime.utcnow().isoformat(),
            'debug_info': {
                'prompt': prompt,
                'document_id': document_id,
                'error': error
            }
        }
    
    async def debug_analysis(self, document_id: str, storage_service):
        """Debug method to check document setup"""
        logger.warning("🔍 DEBUG: Checking document setup")
        
        # Check metadata
        metadata = await self.data_loader.load_metadata(document_id, storage_service)
        logger.warning(f"📋 Metadata: {metadata}")
        
        # List all files
        files = await self.data_loader.debug_list_all_document_files(document_id, storage_service)
        
        # Load thumbnails
        thumbnails = await self.data_loader.load_all_thumbnails(document_id, storage_service)
        logger.warning(f"📸 Loaded {len(thumbnails)} thumbnails")
        
        # Try to load first high-res page
        if thumbnails:
            first_page = await self.data_loader.load_specific_pages(
                document_id, storage_service, [1]
            )
            logger.warning(f"📄 First high-res page: {'Loaded' if first_page else 'Failed'}")
        
        return {
            "metadata": metadata,
            "file_count": len(files),
            "thumbnail_count": len(thumbnails),
            "files": files[:10]  # First 10 files
        }
