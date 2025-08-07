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
from .calculation_engine import CalculationEngine  # NOW IMPORTED!

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
    """
    Visual Intelligence First Construction AI - Core Orchestrator
    
    SIMPLIFIED PIPELINE:
    1. Understand question
    2. Select pages intelligently
    3. Triple verification visual analysis
    4. Run calculations if needed
    5. Format response
    6. Generate highlights/notes if requested
    """
    
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
        self.calculation_engine = CalculationEngine()  # NOW INITIALIZED!
        
        # Set vision client for modules that need it
        vision_client = self.vision.client
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
        Main pipeline - SIMPLIFIED to core steps only
        """
        
        logger.info(f"🚀 Starting AI analysis for: '{prompt}'")
        logger.info(f"📄 Document: {document_id}, Page: {current_page}")
        
        start_time = time.time()
        self.current_prompt = prompt
        
        try:
            # STEP 1: Understand the question
            logger.info("🧠 STEP 1: Understanding the question")
            question_analysis = await self.question_analyzer.analyze(prompt)
            effective_page = current_page or question_analysis.get("requested_page")
            
            # Check cache
            cache_key = self._generate_cache_key(prompt, document_id, effective_page)
            if CONFIG["aggressive_caching"]:
                cached_result = self.cache.get(cache_key, "analysis")
                if cached_result:
                    logger.info("✅ Returning cached analysis")
                    return cached_result
            
            # STEP 2: Load thumbnails and select pages
            logger.info("📸 STEP 2: Loading thumbnails")
            metadata = await self.data_loader.load_metadata(document_id, storage_service)
            total_pages = metadata.get("page_count") if metadata else None
            
            thumbnails = await self.data_loader.load_all_thumbnails(
                document_id, storage_service, total_pages
            )
            
            logger.info("🔍 STEP 3: Selecting relevant pages")
            selected_pages = await self.page_selector.select_relevant_pages(
                thumbnails, question_analysis, current_page
            )
            
            # STEP 4: Load high-res pages and comprehensive data
            logger.info(f"📄 STEP 4: Loading {len(selected_pages)} high-res pages")
            images = await self.data_loader.load_specific_pages(
                document_id, storage_service, selected_pages
            )
            
            comprehensive_data = await self.data_loader.load_comprehensive_data(
                document_id, storage_service, question_analysis, images
            )
            
            # STEP 5: Triple verification visual analysis
            logger.info("👁️ STEP 5: Triple verification visual analysis")
            visual_result = await self.vision.analyze(
                prompt, question_analysis, images, 
                effective_page or 1, comprehensive_data
            )
            
            # STEP 6: Validation and consensus building
            logger.info("🛡️ STEP 6: Building consensus from triple verification")
            validation_results = await self.validator.validate(
                visual_result, comprehensive_data, question_analysis
            )
            
            consensus_result = await self.validator.build_consensus(
                visual_result, validation_results, question_analysis
            )
            
            trust_metrics = self.validator.calculate_trust_metrics(
                visual_result, validation_results, consensus_result
            )
            
            # STEP 7: Run calculations if needed
            calculation_result = None
            if self._should_run_calculations(question_analysis, consensus_result):
                logger.info("🧮 STEP 7: Running calculations")
                calculation_result = await self.calculation_engine.calculate(
                    prompt,
                    comprehensive_data,  # Document knowledge
                    {"visual_result": visual_result, "consensus": consensus_result}
                )
            else:
                logger.info("🧮 STEP 7: Calculations not needed")
            
            # STEP 8: Generate response
            logger.info("📝 STEP 8: Generating response")
            raw_response = self.formatter.format_response(
                visual_result, validation_results, consensus_result,
                trust_metrics, question_analysis, calculation_result
            )
            self.current_response = raw_response
            
            # STEP 9: Optional post-processing for schedules/compliance
            if self._should_run_post_processing(question_analysis):
                logger.info("🔧 STEP 9: Post-processing")
                post_results = await self.post_processor.process(
                    visual_result, images, comprehensive_data, question_analysis
                )
                
                if post_results.get("summary"):
                    raw_response += "\n\n" + post_results["summary"]
            
            # STEP 10: Generate highlights if requested
            result_highlights = {"response": raw_response}
            if request_highlights and prompt.lower().startswith("highlight"):
                logger.info("🎨 STEP 10: Generating highlights")
                highlights = await self.highlighter.generate_highlights(
                    visual_result, images, comprehensive_data
                )
                result_highlights["highlights"] = highlights
            
            # STEP 11: Generate notes (always helpful)
            logger.info("📝 STEP 11: Generating notes")
            notes = await self.note_generator.generate_notes(
                visual_result, document_id, storage_service, author
            )
            
            # Assemble final response
            final_result = self._assemble_response(
                result_highlights, visual_result, validation_results,
                trust_metrics, question_analysis, notes, calculation_result,
                start_time
            )
            
            # Cache if high confidence
            if trust_metrics.reliability_score > 0.85:
                self.cache.set(cache_key, final_result, "analysis")
            
            logger.info(f"✅ Analysis complete in {time.time() - start_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            return self._generate_error_response(prompt, document_id, str(e))
    
    def _should_run_calculations(
        self,
        question_analysis: Dict[str, Any],
        consensus_result: Dict[str, Any]
    ) -> bool:
        """Determine if calculations are needed"""
        
        # Check if question explicitly asks for calculations
        prompt_lower = question_analysis.get("original_prompt", "").lower()
        calc_keywords = [
            'calculate', 'calculation', 'cost', 'price', 'load', 'electrical load',
            'area', 'square footage', 'sq ft', 'coverage', 'spacing',
            'how much', 'estimate', 'material', 'budget', 'quantity'
        ]
        
        if any(keyword in prompt_lower for keyword in calc_keywords):
            return True
        
        # Check if validation suggests calculations would help
        if consensus_result.get("calculation_recommended"):
            return True
        
        # Check question type
        if question_analysis.get("type") == QuestionType.ESTIMATE:
            return True
        
        return False
    
    def _should_run_post_processing(
        self,
        question_analysis: Dict[str, Any]
    ) -> bool:
        """Determine if post-processing is needed"""
        
        # Always run for compliance questions
        if question_analysis.get("type") == QuestionType.COMPLIANCE:
            return True
        
        # Run if specifically asking about schedules
        prompt_lower = question_analysis.get("original_prompt", "").lower()
        if any(word in prompt_lower for word in ['schedule', 'table', 'legend']):
            return True
        
        # Run for detailed analysis questions
        if question_analysis.get("type") == QuestionType.DETAILED:
            return True
        
        return False
    
    def _generate_cache_key(self, prompt: str, document_id: str, page: Optional[int]) -> str:
        """Generate cache key"""
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
        calculation_result: Optional[Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Assemble the final response"""
        
        response = {
            'status': 'success',
            'ai_response': result_highlights.get('response', ''),
            'highlights': result_highlights.get('highlights', []),
            'visual_result': {
                'element_type': visual_result.element_type,
                'count': visual_result.count,
                'confidence': visual_result.confidence,
                'locations': visual_result.locations,
                'verification_notes': visual_result.verification_notes
            },
            'trust_metrics': {
                'reliability_score': trust_metrics.reliability_score,
                'perfect_accuracy': trust_metrics.perfect_accuracy_achieved,
                'validation_consensus': trust_metrics.validation_consensus
            },
            'notes': [note.dict() for note in notes],
            'processing_time': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add calculation results if present
        if calculation_result:
            response['calculation'] = {
                'value': calculation_result.value,
                'unit': calculation_result.unit,
                'calculation_type': calculation_result.calculation_type.value,
                'formula_used': calculation_result.formula_used,
                'confidence': calculation_result.confidence,
                'details': calculation_result.details
            }
        
        # Add metadata
        response['metadata'] = {
            'question_type': question_analysis.get('type').value if question_analysis.get('type') else 'general',
            'element_focus': question_analysis.get('element_focus'),
            'triple_verification': True,
            'calculations_performed': calculation_result is not None
        }
        
        return response
    
    def _generate_error_response(self, prompt: str, document_id: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'status': 'error',
            'message': f'⚠️ Analysis failed: {error}',
            'ai_response': f'I encountered an error while analyzing your question: {error}\n\nPlease try again or rephrase your question.',
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': {
                'prompt': prompt,
                'document_id': document_id,
                'error_details': error
            }
        }