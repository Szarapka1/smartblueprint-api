# validation_system.py
import asyncio
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG

# Fix: Import from app.models.schemas instead of .models
from app.models.schemas import VisualIntelligenceResult, ValidationResult, TrustMetrics

logger = logging.getLogger(__name__)


class ValidationSystem:
    """
    Triple Verification Consensus Builder
    
    Takes the results from vision_intelligence.py's 3 passes and:
    1. Builds consensus
    2. Calculates trust metrics
    3. Identifies any discrepancies
    4. Determines if calculations are needed
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None  # Will be set by core
        
        # Confidence thresholds based on consensus
        self.confidence_thresholds = {
            "perfect": 0.95,      # All 3 passes agree exactly
            "excellent": 0.90,    # All 3 passes within 1
            "good": 0.85,         # 2 passes agree, 1 differs slightly
            "moderate": 0.75,     # Some agreement
            "low": 0.60,          # Significant disagreement
            "failed": 0.30        # Major discrepancies
        }
    
    def set_vision_client(self, client):
        """Set the vision client (for compatibility)"""
        self.vision_client = client
    
    async def validate(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> List[ValidationResult]:
        """
        Build validation results from triple verification
        
        Since vision_intelligence.py now does the 3 passes,
        we just need to package the results properly
        """
        
        logger.info(f"🛡️ Processing triple verification results for {visual_result.count} {visual_result.element_type}(s)")
        
        # Extract the 3 pass results from metadata
        metadata = visual_result.analysis_metadata or {}
        
        # Create validation results for each pass
        validation_results = []
        
        # Pass 1: Visual Count
        validation_results.append(self._create_pass_result(
            pass_number=1,
            methodology="visual_count",
            count=metadata.get("pass1_count", visual_result.count),
            confidence=0.90 if metadata.get("pass1_count") == visual_result.count else 0.70,
            visual_result=visual_result
        ))
        
        # Pass 2: Spatial Verification
        validation_results.append(self._create_pass_result(
            pass_number=2,
            methodology="spatial_verification",
            count=metadata.get("pass2_count", visual_result.count),
            confidence=0.95 if visual_result.locations else 0.70,
            visual_result=visual_result,
            has_locations=bool(visual_result.locations)
        ))
        
        # Pass 3: Text Review
        text_count = metadata.get("pass3_count")
        schedule_count = metadata.get("schedule_count")
        
        validation_results.append(self._create_pass_result(
            pass_number=3,
            methodology="text_review",
            count=text_count or schedule_count,
            confidence=0.95 if schedule_count is not None else (0.85 if text_count is not None else 0.60),
            visual_result=visual_result,
            schedule_found=schedule_count is not None,
            text_found=text_count is not None
        ))
        
        # Add a consensus validation result
        consensus_result = self._create_consensus_validation(
            validation_results, visual_result, question_analysis
        )
        validation_results.append(consensus_result)
        
        logger.info(f"✅ Validation complete with {len(validation_results)} results")
        return validation_results
    
    def _create_pass_result(
        self,
        pass_number: int,
        methodology: str,
        count: Optional[int],
        confidence: float,
        visual_result: VisualIntelligenceResult,
        **kwargs
    ) -> ValidationResult:
        """Create a validation result for a single pass"""
        
        findings = {
            "count": count,
            "methodology": methodology,
            "element_type": visual_result.element_type
        }
        
        # Add any additional kwargs to findings
        findings.update(kwargs)
        
        # Determine status
        if count is None:
            status = "no_data"
        elif count == visual_result.count:
            status = "confirmed"
        elif abs(count - visual_result.count) <= 1:
            status = "near_match"
        else:
            status = "discrepancy"
        
        findings["status"] = status
        
        # Create result
        result = ValidationResult(
            pass_number=pass_number,
            methodology=methodology,
            findings=findings,
            confidence=confidence,
            trust_score=confidence
        )
        
        # Add any discrepancies
        if status == "discrepancy" and count is not None:
            result.discrepancies.append(
                f"{methodology} found {count} but consensus is {visual_result.count}"
            )
        
        # Add cross-references for Pass 3
        if methodology == "text_review":
            if kwargs.get("schedule_found"):
                result.cross_references.append("Schedule verification completed")
            if kwargs.get("text_found"):
                result.cross_references.append("Text documentation reviewed")
        
        return result
    
    def _create_consensus_validation(
        self,
        validation_results: List[ValidationResult],
        visual_result: VisualIntelligenceResult,
        question_analysis: Dict[str, Any]
    ) -> ValidationResult:
        """Create the final consensus validation result"""
        
        # Extract counts from all passes
        counts = []
        for val in validation_results:
            if val.findings.get("count") is not None:
                counts.append(val.findings["count"])
        
        # Determine consensus level
        if len(counts) >= 2:
            unique_counts = set(counts)
            if len(unique_counts) == 1:
                consensus_level = "perfect"
                consensus_desc = "Perfect consensus - all passes agree"
            elif max(counts) - min(counts) <= 1:
                consensus_level = "excellent"
                consensus_desc = "Excellent consensus - minor variation"
            elif max(counts) - min(counts) <= 2:
                consensus_level = "good"
                consensus_desc = "Good consensus - acceptable variation"
            else:
                consensus_level = "moderate"
                consensus_desc = "Moderate consensus - some disagreement"
        else:
            consensus_level = "low"
            consensus_desc = "Limited data for consensus"
        
        # Calculate final confidence
        confidence = self.confidence_thresholds[consensus_level]
        
        # Check if calculations might help
        needs_calculation = self._check_if_calculation_helps(
            question_analysis, visual_result, validation_results
        )
        
        findings = {
            "consensus_level": consensus_level,
            "consensus_description": consensus_desc,
            "total_passes": len(validation_results),
            "counts_found": counts,
            "final_count": visual_result.count,
            "needs_calculation": needs_calculation,
            "status": "consensus"
        }
        
        result = ValidationResult(
            pass_number=4,  # Consensus is the 4th "validation"
            methodology="consensus_analysis",
            findings=findings,
            confidence=confidence,
            trust_score=confidence
        )
        
        # Note if calculation would add value
        if needs_calculation:
            result.cross_references.append(
                "Calculation analysis recommended for additional verification"
            )
        
        return result
    
    def _check_if_calculation_helps(
        self,
        question_analysis: Dict[str, Any],
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult]
    ) -> bool:
        """
        Determine if running calculations would provide additional validation
        
        Calculations help when:
        1. Question explicitly asks for calculations
        2. Element type benefits from calculation verification
        3. There's some uncertainty in counts
        """
        
        # Check if question asks for calculations
        prompt_lower = question_analysis.get("original_prompt", "").lower()
        calc_keywords = [
            'calculate', 'cost', 'load', 'area', 'coverage',
            'spacing', 'how much', 'estimate', 'total load',
            'square footage', 'material', 'budget'
        ]
        
        if any(keyword in prompt_lower for keyword in calc_keywords):
            return True
        
        # Check if element type benefits from calculations
        calculable_elements = [
            'outlet', 'panel', 'light fixture',  # Electrical load
            'sprinkler', 'diffuser',              # Coverage area
            'door', 'window',                     # Cost estimates
            'parking',                            # Spacing requirements
            'column', 'beam'                      # Structural calcs
        ]
        
        if visual_result.element_type in calculable_elements:
            # If there's any uncertainty, calculations might help
            confidences = [v.confidence for v in validation_results]
            if any(conf < 0.90 for conf in confidences):
                return True
        
        return False
    
    async def build_consensus(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build consensus from validation results
        This is now much simpler since we have triple verification
        """
        
        # Extract counts from each pass
        pass_counts = {}
        for val in validation_results:
            if val.methodology in ["visual_count", "spatial_verification", "text_review"]:
                count = val.findings.get("count")
                if count is not None:
                    pass_counts[val.methodology] = count
        
        # Get unique counts
        all_counts = list(pass_counts.values())
        unique_counts = list(set(all_counts))
        
        # Check for schedule
        schedule_count = None
        for val in validation_results:
            if val.methodology == "text_review" and val.findings.get("schedule_found"):
                schedule_count = val.findings.get("count")
                break
        
        # Determine agreement level
        if len(unique_counts) == 1 and len(all_counts) >= 2:
            agreement = "PERFECT_CONSENSUS"
            consensus_confidence = 0.99
        elif len(unique_counts) == 2 and max(unique_counts) - min(unique_counts) <= 1:
            agreement = "EXCELLENT_CONSENSUS"
            consensus_confidence = 0.95
        elif len(unique_counts) == 2 and max(unique_counts) - min(unique_counts) <= 2:
            agreement = "GOOD_CONSENSUS"
            consensus_confidence = 0.90
        elif schedule_count is not None and visual_result.count == 0:
            agreement = "SCHEDULE_VISUAL_MISMATCH"
            consensus_confidence = 0.50
        else:
            agreement = "MODERATE_CONSENSUS"
            consensus_confidence = 0.75
        
        # Build consensus dictionary
        consensus = {
            "visual_count": visual_result.count,
            "pass_counts": pass_counts,
            "unique_counts": unique_counts,
            "consensus_count": visual_result.count,
            "validation_agreement": agreement,
            "consensus_confidence": consensus_confidence,
            "schedule_found": schedule_count is not None,
            "schedule_count": schedule_count,
            "triple_verification_complete": True
        }
        
        # Add notes about agreement
        if agreement == "PERFECT_CONSENSUS":
            consensus["note"] = "All verification passes agree perfectly"
        elif agreement == "EXCELLENT_CONSENSUS":
            consensus["note"] = "Verification passes show excellent agreement"
        elif agreement == "SCHEDULE_VISUAL_MISMATCH" and schedule_count:
            consensus["discrepancy_note"] = f"Schedule shows {schedule_count} but none found visually"
        
        # Note if calculations recommended
        if any(v.findings.get("needs_calculation") for v in validation_results):
            consensus["calculation_recommended"] = True
            consensus["calculation_reason"] = "Additional verification through calculations"
        
        logger.info(f"🤝 Consensus: {agreement} (confidence: {consensus_confidence:.2f})")
        
        return consensus
    
    def calculate_trust_metrics(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any]
    ) -> TrustMetrics:
        """
        Calculate final trust metrics based on triple verification
        """
        
        metrics = TrustMetrics()
        
        # Get consensus info
        agreement = consensus_result.get("validation_agreement", "UNKNOWN")
        consensus_confidence = consensus_result.get("consensus_confidence", 0.75)
        
        # Visual intelligence score is the final confidence from vision_intelligence.py
        metrics.visual_intelligence_score = visual_result.confidence
        
        # Perfect accuracy achieved if perfect consensus
        if agreement == "PERFECT_CONSENSUS":
            metrics.perfect_accuracy_achieved = True
            metrics.reliability_score = 0.99
            metrics.confidence_basis = "Perfect triple verification consensus"
        elif agreement == "EXCELLENT_CONSENSUS":
            metrics.perfect_accuracy_achieved = False
            metrics.reliability_score = 0.95
            metrics.confidence_basis = "Excellent agreement in triple verification"
        elif agreement == "SCHEDULE_VISUAL_MISMATCH":
            metrics.perfect_accuracy_achieved = False
            metrics.reliability_score = 0.50
            metrics.confidence_basis = "Discrepancy between visual and documented counts"
            metrics.uncertainty_factors.append("Schedule shows different count than visual analysis")
        else:
            metrics.perfect_accuracy_achieved = False
            metrics.reliability_score = consensus_confidence
            metrics.confidence_basis = "Consensus from triple verification"
        
        # Validation consensus flag
        metrics.validation_consensus = agreement in [
            "PERFECT_CONSENSUS", "EXCELLENT_CONSENSUS", "GOOD_CONSENSUS"
        ]
        
        # Accuracy sources
        metrics.accuracy_sources = ["triple_verification"]
        for val in validation_results:
            if val.confidence >= 0.90:
                metrics.accuracy_sources.append(f"pass_{val.pass_number}_{val.methodology}")
        
        # Source quality scores
        for val in validation_results:
            metrics.source_quality_scores[val.methodology] = val.confidence
        
        # Add uncertainty factors
        if visual_result.count == 0:
            metrics.uncertainty_factors.append("Zero count requires manual verification")
        
        if consensus_result.get("calculation_recommended"):
            metrics.uncertainty_factors.append("Calculations recommended for additional verification")
        
        # Add any discrepancies
        for val in validation_results:
            if val.discrepancies:
                metrics.uncertainty_factors.extend(val.discrepancies[:1])  # Just first one
        
        # Ensure bounds
        metrics.reliability_score = max(0.0, min(1.0, metrics.reliability_score))
        
        logger.info(f"📊 Trust Metrics: Reliability={metrics.reliability_score:.2f}, "
                   f"Perfect={metrics.perfect_accuracy_achieved}, "
                   f"Consensus={metrics.validation_consensus}")
        
        return metrics