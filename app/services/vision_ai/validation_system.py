# validation_system.py - COMPLETE REWRITE FOR KNOWLEDGE-BASED VALIDATION

import asyncio
import re
import logging
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from app.core.config import CONFIG
from app.models.schemas import VisualIntelligenceResult, ValidationResult, TrustMetrics

logger = logging.getLogger(__name__)


class ValidationSystem:
    """
    Knowledge-Based Validation System - The Engineering Authority.
    
    PHILOSOPHY: Validate answers against our comprehensive document knowledge,
    not by re-scanning. We already KNOW what's in the document - now we verify
    the answer is consistent with that knowledge.
    """

    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None
        
        # Validation methods now check different aspects of knowledge consistency
        self.validation_methods = [
            "knowledge_consistency",      # Is answer consistent with our knowledge?
            "ground_truth_verification",   # Does it match the JSON ground truth?
            "spatial_logic_validation",    # Are locations spatially logical?
            "cross_reference_validation"   # Do references and relationships match?
        ]
        
        # Validation confidence thresholds
        self.thresholds = {
            "perfect_match": 1.0,
            "excellent_match": 0.95,
            "good_match": 0.85,
            "acceptable_match": 0.75,
            "poor_match": 0.50
        }

    def set_vision_client(self, client):
        """Set the vision client for validation checks."""
        self.vision_client = client

    async def validate(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> List[ValidationResult]:
        """
        Performs 4x validation against our document knowledge.
        
        No more re-scanning! We validate by checking:
        1. Is the answer consistent with our comprehensive knowledge?
        2. Does it match ground-truth data?
        3. Is it spatially/logically sound?
        4. Do all cross-references check out?
        """
        
        element_type = visual_result.element_type
        count = visual_result.count
        
        logger.info(f"🛡️ Knowledge-Based Validation for {count} {element_type}(s)")
        logger.info(f"📚 Validating against comprehensive document knowledge")
        
        # Extract document knowledge if available
        document_knowledge = self._extract_document_knowledge(comprehensive_data)
        
        # Run all validation methods
        validation_tasks = []
        for method in self.validation_methods:
            task = self._perform_knowledge_validation(
                method,
                visual_result,
                comprehensive_data,
                question_analysis,
                document_knowledge
            )
            validation_tasks.append(task)
        
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"Validation method '{self.validation_methods[i]}' failed: {result}")
                processed_results.append(self._create_failed_validation(i + 1, self.validation_methods[i]))
            else:
                processed_results.append(result)
        
        # Log validation summary
        high_confidence = sum(1 for r in processed_results if r.confidence >= 0.90)
        logger.info(f"✅ Validation complete: {high_confidence}/{len(processed_results)} high confidence")
        
        return processed_results

    async def _perform_knowledge_validation(
        self,
        methodology: str,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any],
        document_knowledge: Dict[str, Any]
    ) -> ValidationResult:
        """
        Perform a single validation check against document knowledge.
        Each method validates a different aspect of the answer.
        """
        
        logger.info(f"🔍 Validating: {methodology}")
        
        try:
            if methodology == "knowledge_consistency":
                return await self._validate_knowledge_consistency(
                    visual_result, document_knowledge, question_analysis
                )
            
            elif methodology == "ground_truth_verification":
                return await self._validate_ground_truth(
                    visual_result, comprehensive_data, document_knowledge
                )
            
            elif methodology == "spatial_logic_validation":
                return await self._validate_spatial_logic(
                    visual_result, comprehensive_data, document_knowledge
                )
            
            elif methodology == "cross_reference_validation":
                return await self._validate_cross_references(
                    visual_result, comprehensive_data, document_knowledge
                )
            
            else:
                return self._create_failed_validation(
                    self.validation_methods.index(methodology) + 1,
                    methodology
                )
                
        except Exception as e:
            logger.error(f"Validation error in {methodology}: {e}")
            return self._create_failed_validation(
                self.validation_methods.index(methodology) + 1,
                methodology
            )

    async def _validate_knowledge_consistency(
        self,
        visual_result: VisualIntelligenceResult,
        document_knowledge: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that the answer is consistent with our document knowledge.
        This is the PRIMARY validation - does the answer match what we know?
        """
        
        element_type = visual_result.element_type
        reported_count = visual_result.count
        
        findings = {
            "status": "CHECKING",
            "details": [],
            "discrepancies": []
        }
        
        # Check if we have this element in our knowledge
        if document_knowledge and "all_elements" in document_knowledge:
            if element_type in document_knowledge["all_elements"]:
                known_data = document_knowledge["all_elements"][element_type]
                known_count = known_data.get("total_count", 0)
                known_pages = known_data.get("pages", [])
                
                # Compare counts
                if reported_count == known_count:
                    findings["status"] = "PERFECT_MATCH"
                    findings["details"].append(
                        f"Count matches knowledge: {reported_count} = {known_count}"
                    )
                    confidence = 1.0
                elif abs(reported_count - known_count) <= 2:
                    findings["status"] = "CLOSE_MATCH"
                    findings["details"].append(
                        f"Count nearly matches: {reported_count} vs {known_count} known"
                    )
                    confidence = 0.90
                else:
                    findings["status"] = "MISMATCH"
                    findings["discrepancies"].append(
                        f"Count mismatch: reported {reported_count} vs {known_count} in knowledge"
                    )
                    confidence = 0.70
                
                # Validate page distribution
                if visual_result.locations:
                    reported_pages = set(loc.get("page", 0) for loc in visual_result.locations)
                    if reported_pages.issubset(set(known_pages)):
                        findings["details"].append("Page distribution consistent")
                    else:
                        findings["discrepancies"].append(
                            f"Pages don't match: reported {reported_pages} vs known {known_pages}"
                        )
                        confidence *= 0.9
                
            else:
                # Element not in knowledge base
                if reported_count == 0:
                    findings["status"] = "CONFIRMED_ABSENT"
                    findings["details"].append(f"Correctly identified {element_type} as not present")
                    confidence = 0.95
                else:
                    findings["status"] = "NOT_IN_KNOWLEDGE"
                    findings["discrepancies"].append(
                        f"Found {reported_count} {element_type}s but not in document knowledge"
                    )
                    confidence = 0.60
        else:
            # No knowledge base available - can't validate this way
            findings["status"] = "NO_KNOWLEDGE_BASE"
            findings["details"].append("Document knowledge not available for validation")
            confidence = 0.50
        
        # Calculate trust score
        trust_score = self._calculate_trust_score(findings["status"], confidence)
        
        return ValidationResult(
            pass_number=1,
            methodology="knowledge_consistency",
            findings=findings,
            confidence=confidence,
            trust_score=trust_score,
            discrepancies=findings["discrepancies"],
            cross_references=findings.get("details", [])
        )

    async def _validate_ground_truth(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        document_knowledge: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate against ground-truth data (grid_systems.json, document_index.json).
        This checks if our answer aligns with the structured data files.
        """
        
        findings = {
            "status": "CHECKING",
            "details": [],
            "discrepancies": []
        }
        
        confidence = 0.85  # Base confidence
        
        # Check against document index
        if comprehensive_data.get("document_index"):
            doc_index = comprehensive_data["document_index"]
            
            # Check if element appears in index
            element_type = visual_result.element_type
            if self._check_element_in_index(element_type, doc_index):
                findings["details"].append(f"{element_type} confirmed in document index")
                confidence += 0.05
            
            # Validate page references
            if "page_index" in doc_index:
                valid_pages = self._validate_page_references(
                    visual_result.locations,
                    doc_index["page_index"]
                )
                if valid_pages:
                    findings["details"].append("Page references validated against index")
                    confidence += 0.05
                else:
                    findings["discrepancies"].append("Some page references not in index")
                    confidence -= 0.10
        
        # Check against grid systems
        if comprehensive_data.get("grid_systems"):
            grid_systems = comprehensive_data["grid_systems"]
            
            # Validate grid references
            valid_grids = self._validate_grid_references(
                visual_result.grid_references,
                grid_systems
            )
            
            if valid_grids["all_valid"]:
                findings["details"].append("All grid references validated")
                findings["status"] = "GROUND_TRUTH_VERIFIED"
                confidence = min(0.98, confidence + 0.08)
            elif valid_grids["partial_valid"]:
                findings["details"].append(f"{valid_grids['valid_count']}/{valid_grids['total_count']} grid refs valid")
                findings["status"] = "PARTIAL_VERIFICATION"
                confidence *= 0.90
            else:
                findings["discrepancies"].append("Grid references could not be verified")
                findings["status"] = "VERIFICATION_FAILED"
                confidence *= 0.75
        else:
            findings["status"] = "NO_GROUND_TRUTH"
            findings["details"].append("No ground truth data available")
        
        trust_score = self._calculate_trust_score(findings["status"], confidence)
        
        return ValidationResult(
            pass_number=2,
            methodology="ground_truth_verification",
            findings=findings,
            confidence=confidence,
            trust_score=trust_score,
            discrepancies=findings["discrepancies"],
            cross_references=findings.get("details", [])
        )

    async def _validate_spatial_logic(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        document_knowledge: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that the spatial distribution and locations make logical sense.
        E.g., outlets should be distributed along walls, not clustered in one spot.
        """
        
        element_type = visual_result.element_type
        locations = visual_result.locations
        count = visual_result.count
        
        findings = {
            "status": "CHECKING",
            "details": [],
            "discrepancies": []
        }
        
        confidence = 0.85
        
        # Check distribution logic
        if count > 0 and locations:
            # Analyze distribution pattern
            distribution = self._analyze_distribution_pattern(locations, element_type)
            
            if distribution["is_logical"]:
                findings["status"] = "SPATIALLY_LOGICAL"
                findings["details"].append(f"Distribution pattern: {distribution['pattern']}")
                confidence = 0.95
            else:
                findings["status"] = "SPATIAL_CONCERNS"
                findings["discrepancies"].append(distribution["concern"])
                confidence = 0.70
            
            # Check for impossible locations
            impossible = self._check_impossible_locations(locations, element_type)
            if impossible:
                findings["discrepancies"].extend(impossible)
                confidence *= 0.80
            
            # Validate against known building layout
            if document_knowledge and "spatial_organization" in document_knowledge:
                spatial_org = document_knowledge["spatial_organization"]
                
                # Check if locations match known floors
                if spatial_org.get("floors"):
                    location_floors = self._extract_floors_from_locations(locations)
                    known_floors = spatial_org["floors"]
                    
                    if location_floors.issubset(set(known_floors)):
                        findings["details"].append("All locations on valid floors")
                    else:
                        unknown_floors = location_floors - set(known_floors)
                        findings["discrepancies"].append(
                            f"References unknown floors: {unknown_floors}"
                        )
                        confidence *= 0.85
        
        elif count == 0:
            # Zero count is spatially logical for absent elements
            findings["status"] = "SPATIALLY_LOGICAL"
            findings["details"].append(f"No {element_type}s is spatially valid")
            confidence = 0.90
        
        else:
            findings["status"] = "NO_SPATIAL_DATA"
            findings["details"].append("No location data to validate")
            confidence = 0.60
        
        trust_score = self._calculate_trust_score(findings["status"], confidence)
        
        return ValidationResult(
            pass_number=3,
            methodology="spatial_logic_validation",
            findings=findings,
            confidence=confidence,
            trust_score=trust_score,
            discrepancies=findings["discrepancies"],
            cross_references=findings.get("details", [])
        )

    async def _validate_cross_references(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        document_knowledge: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate cross-references and relationships.
        Check if element references, tags, and relationships are consistent.
        """
        
        findings = {
            "status": "CHECKING",
            "details": [],
            "discrepancies": []
        }
        
        confidence = 0.85
        
        # Check element tags if present
        if visual_result.locations:
            tags_found = []
            for loc in visual_result.locations:
                if loc.get("element_tag"):
                    tags_found.append(loc["element_tag"])
            
            if tags_found:
                # Validate tag format
                valid_tags = self._validate_tag_format(tags_found, visual_result.element_type)
                if valid_tags["all_valid"]:
                    findings["details"].append(f"All {len(tags_found)} tags properly formatted")
                    confidence += 0.05
                else:
                    findings["discrepancies"].append(
                        f"Invalid tag format: {valid_tags['invalid']}"
                    )
                    confidence -= 0.10
                
                # Check for duplicates
                if len(tags_found) != len(set(tags_found)):
                    findings["discrepancies"].append("Duplicate tags found")
                    confidence -= 0.15
        
        # Validate relationships if in knowledge base
        if document_knowledge and "relationships" in document_knowledge:
            relationships = document_knowledge["relationships"]
            
            # Check element-to-page mapping
            if visual_result.element_type in relationships.get("element_to_pages", {}):
                expected_pages = relationships["element_to_pages"][visual_result.element_type]
                actual_pages = [loc.get("page", 0) for loc in visual_result.locations]
                
                if set(actual_pages).issubset(set(expected_pages)):
                    findings["details"].append("Page relationships verified")
                    findings["status"] = "REFERENCES_VALIDATED"
                    confidence = min(0.95, confidence + 0.05)
                else:
                    findings["discrepancies"].append(
                        "Page relationships don't match knowledge"
                    )
                    findings["status"] = "REFERENCE_MISMATCH"
                    confidence *= 0.85
        
        # Check against schedules if mentioned
        if visual_result.analysis_metadata.get("schedule_found"):
            schedule_count = visual_result.analysis_metadata.get("schedule_count", 0)
            if abs(schedule_count - visual_result.count) > 2:
                findings["discrepancies"].append(
                    f"Count doesn't match schedule: {visual_result.count} vs {schedule_count}"
                )
                confidence *= 0.80
            else:
                findings["details"].append("Count matches schedule")
                confidence += 0.05
        
        # Final status
        if not findings["discrepancies"]:
            findings["status"] = "ALL_REFERENCES_VALID"
        elif len(findings["discrepancies"]) == 1:
            findings["status"] = "MINOR_ISSUES"
        else:
            findings["status"] = "MULTIPLE_ISSUES"
        
        trust_score = self._calculate_trust_score(findings["status"], confidence)
        
        return ValidationResult(
            pass_number=4,
            methodology="cross_reference_validation",
            findings=findings,
            confidence=confidence,
            trust_score=trust_score,
            discrepancies=findings["discrepancies"],
            cross_references=findings.get("details", [])
        )

    async def build_consensus(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build consensus from validation results.
        Now based on knowledge validation, not re-scanning.
        """
        
        logger.info("🤝 Building consensus from knowledge-based validations")
        
        consensus = {
            "validation_agreement": "UNKNOWN",
            "final_count": visual_result.count,
            "confidence_level": 0.0,
            "discrepancy_note": None,
            "validation_summary": []
        }
        
        if not validation_results:
            consensus["validation_agreement"] = "NO_VALIDATION"
            return consensus
        
        # Calculate agreement based on validation confidence
        high_confidence = [v for v in validation_results if v.confidence >= 0.90]
        medium_confidence = [v for v in validation_results if 0.75 <= v.confidence < 0.90]
        low_confidence = [v for v in validation_results if v.confidence < 0.75]
        
        # Determine consensus
        if len(high_confidence) >= 3:
            consensus["validation_agreement"] = "STRONG_CONSENSUS"
            consensus["confidence_level"] = sum(v.confidence for v in high_confidence) / len(high_confidence)
        elif len(high_confidence) >= 2:
            consensus["validation_agreement"] = "CONSENSUS_REACHED"
            consensus["confidence_level"] = sum(v.confidence for v in validation_results) / len(validation_results)
        elif len(medium_confidence) >= 2:
            consensus["validation_agreement"] = "MODERATE_CONSENSUS"
            consensus["confidence_level"] = sum(v.confidence for v in validation_results) / len(validation_results)
        else:
            consensus["validation_agreement"] = "WEAK_CONSENSUS"
            consensus["confidence_level"] = sum(v.confidence for v in validation_results) / len(validation_results)
        
        # Check for critical discrepancies
        all_discrepancies = []
        for v in validation_results:
            all_discrepancies.extend(v.discrepancies)
        
        if all_discrepancies:
            # Find most critical discrepancy
            critical_discrepancy = self._find_critical_discrepancy(all_discrepancies)
            if critical_discrepancy:
                consensus["discrepancy_note"] = critical_discrepancy
                consensus["validation_agreement"] = "CONSENSUS_WITH_CONCERNS"
        
        # Build validation summary
        for v in validation_results:
            status = v.findings.get("status", "UNKNOWN")
            consensus["validation_summary"].append(
                f"{v.methodology}: {status} (confidence: {v.confidence:.2f})"
            )
        
        # Adjust final count if knowledge strongly disagrees
        knowledge_validation = next(
            (v for v in validation_results if v.methodology == "knowledge_consistency"),
            None
        )
        
        if knowledge_validation and knowledge_validation.confidence >= 0.95:
            # Trust the knowledge base for high-confidence validations
            if "known_count" in str(knowledge_validation.findings):
                # Extract known count from findings
                match = re.search(r'(\d+) known', str(knowledge_validation.findings))
                if match:
                    known_count = int(match.group(1))
                    if known_count != visual_result.count:
                        consensus["final_count"] = known_count
                        consensus["discrepancy_note"] = (
                            f"Adjusted count from {visual_result.count} to {known_count} "
                            f"based on document knowledge"
                        )
        
        logger.info(f"✅ Consensus: {consensus['validation_agreement']} "
                   f"(confidence: {consensus['confidence_level']:.2f})")
        
        return consensus

    def calculate_trust_metrics(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any]
    ) -> TrustMetrics:
        """
        Calculate final trust metrics based on knowledge validation.
        Higher trust when answer matches our comprehensive knowledge.
        """
        
        logger.info("📊 Calculating trust metrics from knowledge validation")
        
        if not validation_results:
            return TrustMetrics(
                reliability_score=0.50,
                perfect_accuracy_achieved=False,
                validation_consensus=False,
                confidence_factors={
                    "no_validation": 0.50
                }
            )
        
        # Calculate component scores
        avg_confidence = sum(v.confidence for v in validation_results) / len(validation_results)
        avg_trust = sum(v.trust_score for v in validation_results) / len(validation_results)
        
        # Check for perfect accuracy (all validations > 0.95)
        perfect_accuracy = all(v.confidence >= 0.95 for v in validation_results)
        
        # Check for validation consensus
        validation_consensus = consensus_result.get("validation_agreement") in [
            "STRONG_CONSENSUS", "CONSENSUS_REACHED"
        ]
        
        # Build confidence factors
        confidence_factors = {}
        
        # Factor 1: Knowledge consistency
        knowledge_val = next(
            (v for v in validation_results if v.methodology == "knowledge_consistency"),
            None
        )
        if knowledge_val:
            confidence_factors["knowledge_match"] = knowledge_val.confidence
            if knowledge_val.confidence >= 0.95:
                confidence_factors["perfect_knowledge_match"] = 1.0
        
        # Factor 2: Ground truth verification
        ground_truth_val = next(
            (v for v in validation_results if v.methodology == "ground_truth_verification"),
            None
        )
        if ground_truth_val:
            confidence_factors["ground_truth"] = ground_truth_val.confidence
        
        # Factor 3: Spatial logic
        spatial_val = next(
            (v for v in validation_results if v.methodology == "spatial_logic_validation"),
            None
        )
        if spatial_val:
            confidence_factors["spatial_logic"] = spatial_val.confidence
        
        # Factor 4: Cross references
        cross_ref_val = next(
            (v for v in validation_results if v.methodology == "cross_reference_validation"),
            None
        )
        if cross_ref_val:
            confidence_factors["references"] = cross_ref_val.confidence
        
        # Calculate final reliability score
        if perfect_accuracy:
            reliability_score = 0.99
        elif validation_consensus:
            reliability_score = min(0.95, avg_trust * 1.05)  # Small boost for consensus
        else:
            reliability_score = avg_trust
        
        # Adjust for critical issues
        if consensus_result.get("discrepancy_note"):
            if "mismatch" in consensus_result["discrepancy_note"].lower():
                reliability_score *= 0.85
            else:
                reliability_score *= 0.95
        
        # Round to 2 decimal places
        reliability_score = round(min(0.99, max(0.01, reliability_score)), 2)
        
        logger.info(f"📈 Final Trust Metrics: Reliability={reliability_score:.2f}, "
                   f"Perfect={perfect_accuracy}, Consensus={validation_consensus}")
        
        return TrustMetrics(
            reliability_score=reliability_score,
            perfect_accuracy_achieved=perfect_accuracy,
            validation_consensus=validation_consensus,
            confidence_factors=confidence_factors,
            validation_summary={
                "methods_passed": len([v for v in validation_results if v.confidence >= 0.75]),
                "total_methods": len(validation_results),
                "average_confidence": round(avg_confidence, 3),
                "consensus_type": consensus_result.get("validation_agreement", "UNKNOWN")
            }
        )

    # ============= HELPER METHODS =============

    def _extract_document_knowledge(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document knowledge from comprehensive data if available."""
        
        # This would ideally get the knowledge from the VisionIntelligence cache
        # For now, we'll construct it from available data
        knowledge = {}
        
        # Try to extract from comprehensive data
        if comprehensive_data:
            # Look for cached knowledge
            if "document_knowledge" in comprehensive_data:
                return comprehensive_data["document_knowledge"]
            
            # Build minimal knowledge from available data
            if comprehensive_data.get("document_index"):
                knowledge["document_index"] = comprehensive_data["document_index"]
            
            if comprehensive_data.get("grid_systems"):
                knowledge["grid_systems"] = comprehensive_data["grid_systems"]
            
            # Extract any element counts from context
            if comprehensive_data.get("context"):
                knowledge["context"] = comprehensive_data["context"]
        
        return knowledge

    def _check_element_in_index(self, element_type: str, doc_index: Dict[str, Any]) -> bool:
        """Check if element type appears in document index."""
        
        # Search in page index
        if "page_index" in doc_index:
            for page_id, page_info in doc_index["page_index"].items():
                if element_type.lower() in str(page_info).lower():
                    return True
        
        # Search in sheet numbers
        if "sheet_numbers" in doc_index:
            for sheet, info in doc_index["sheet_numbers"].items():
                if element_type.lower() in str(info).lower():
                    return True
        
        return False

    def _validate_page_references(
        self,
        locations: List[Dict[str, Any]],
        page_index: Dict[str, Any]
    ) -> bool:
        """Validate that page references exist in the index."""
        
        if not locations:
            return True
        
        indexed_pages = set()
        for page_id in page_index.keys():
            # Extract page number from page_id (e.g., "page_1" -> 1)
            match = re.search(r'(\d+)', page_id)
            if match:
                indexed_pages.add(int(match.group(1)))
        
        location_pages = set(loc.get("page", 0) for loc in locations)
        
        # Check if all location pages are in the index
        return location_pages.issubset(indexed_pages)

    def _validate_grid_references(
        self,
        grid_refs: List[str],
        grid_systems: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate grid references against grid system data."""
        
        result = {
            "all_valid": False,
            "partial_valid": False,
            "valid_count": 0,
            "total_count": len(grid_refs)
        }
        
        if not grid_refs:
            result["all_valid"] = True
            return result
        
        # Extract all valid grids from grid_systems
        valid_grids = set()
        for page_id, grid_data in grid_systems.items():
            if isinstance(grid_data, dict):
                # Look for grid identifiers
                if "grids" in grid_data:
                    valid_grids.update(grid_data["grids"])
                # Also check for grid lines
                if "vertical_grids" in grid_data:
                    valid_grids.update(grid_data["vertical_grids"])
                if "horizontal_grids" in grid_data:
                    valid_grids.update(grid_data["horizontal_grids"])
        
        # If no grid system defined, can't validate
        if not valid_grids:
            return result
        
        # Count valid references
        for ref in grid_refs:
            if ref in valid_grids or self._is_valid_grid_format(ref):
                result["valid_count"] += 1
        
        result["all_valid"] = result["valid_count"] == result["total_count"]
        result["partial_valid"] = result["valid_count"] > 0
        
        return result

    def _is_valid_grid_format(self, grid_ref: str) -> bool:
        """Check if grid reference follows standard format (e.g., A-1, B-2)."""
        
        # Common grid formats
        patterns = [
            r'^[A-Z]-?\d+$',  # A-1, B2
            r'^\d+-?[A-Z]$',  # 1-A, 2B
            r'^[A-Z]\.\d+$',  # A.1
            r'^Grid\s+[A-Z]-?\d+$',  # Grid A-1
        ]
        
        for pattern in patterns:
            if re.match(pattern, grid_ref, re.IGNORECASE):
                return True
        
        return False

    def _analyze_distribution_pattern(
        self,
        locations: List[Dict[str, Any]],
        element_type: str
    ) -> Dict[str, Any]:
        """Analyze if the distribution pattern is logical for this element type."""
        
        result = {
            "is_logical": True,
            "pattern": "distributed",
            "concern": None
        }
        
        if not locations:
            return result
        
        # Get unique grid references
        grid_refs = [loc.get("grid_ref", "") for loc in locations]
        unique_grids = set(grid_refs)
        
        # Calculate distribution metrics
        distribution_ratio = len(unique_grids) / len(locations) if locations else 0
        
        # Element-specific distribution logic
        if element_type == "outlet":
            # Outlets should be well-distributed
            if distribution_ratio < 0.5:
                result["is_logical"] = False
                result["pattern"] = "clustered"
                result["concern"] = "Outlets appear clustered, should be distributed along walls"
        
        elif element_type == "door":
            # Doors should mostly be in different locations
            if distribution_ratio < 0.7:
                result["is_logical"] = False
                result["pattern"] = "clustered"
                result["concern"] = "Multiple doors in same location is unusual"
        
        elif element_type == "column":
            # Columns often follow a grid pattern
            result["pattern"] = "grid-aligned"
            # Check if they follow a pattern (simplified check)
            if len(unique_grids) < len(locations) * 0.8:
                result["concern"] = "Some columns share locations, verify structural grid"
        
        elif element_type == "sprinkler":
            # Sprinklers should have regular spacing
            if distribution_ratio < 0.6:
                result["is_logical"] = False
                result["pattern"] = "irregular"
                result["concern"] = "Sprinkler distribution appears irregular"
        
        # Determine pattern description
        if distribution_ratio > 0.8:
            result["pattern"] = "well-distributed"
        elif distribution_ratio > 0.5:
            result["pattern"] = "moderately-distributed"
        else:
            result["pattern"] = "clustered"
        
        return result

    def _check_impossible_locations(
        self,
        locations: List[Dict[str, Any]],
        element_type: str
    ) -> List[str]:
        """Check for logically impossible locations."""
        
        impossible = []
        
        for loc in locations:
            details = loc.get("visual_details", "").lower()
            grid_ref = loc.get("grid_ref", "").lower()
            
            # Element-specific impossible locations
            if element_type == "outlet":
                if "ceiling" in details or "roof" in details:
                    impossible.append("Outlet on ceiling/roof is unusual")
            
            elif element_type == "sprinkler":
                if "floor" in details and "floor drain" not in details:
                    impossible.append("Sprinkler on floor is unusual")
            
            elif element_type == "door":
                if "ceiling" in details or "roof" in details:
                    impossible.append("Door in ceiling/roof is impossible")
            
            elif element_type == "column":
                if "mid-air" in details or "floating" in details:
                    impossible.append("Floating column is structurally impossible")
        
        return impossible

    def _extract_floors_from_locations(self, locations: List[Dict[str, Any]]) -> set:
        """Extract floor numbers from location data."""
        
        floors = set()
        
        for loc in locations:
            # Check page number (often corresponds to floor)
            page = loc.get("page", 0)
            if page > 0:
                floors.add(str(page))
            
            # Check visual details for floor mentions
            details = loc.get("visual_details", "")
            floor_match = re.search(r'(?:floor|level)\s*(\d+)', details, re.IGNORECASE)
            if floor_match:
                floors.add(floor_match.group(1))
        
        return floors

    def _validate_tag_format(
        self,
        tags: List[str],
        element_type: str
    ) -> Dict[str, Any]:
        """Validate that element tags follow expected format."""
        
        result = {
            "all_valid": True,
            "invalid": []
        }
        
        # Define expected tag patterns for each element type
        tag_patterns = {
            "door": [r'^D\d+', r'^DOOR[-\s]?\d+'],
            "window": [r'^W\d+', r'^WIN[-\s]?\d+'],
            "outlet": [r'^[A-Z]+\d+', r'^R\d+'],  # R for receptacle
            "panel": [r'^P\d+', r'^EP\d+', r'^LP\d+'],  # Electrical/Lighting Panel
            "column": [r'^C\d+', r'^COL[-\s]?\d+'],
            "beam": [r'^B\d+', r'^BM[-\s]?\d+'],
            "equipment": [r'^[A-Z]+-\d+', r'^EQ[-\s]?\d+']
        }
        
        # Get patterns for this element type
        patterns = tag_patterns.get(element_type, [r'^[A-Z]+[-\s]?\d+'])
        
        for tag in tags:
            tag_valid = False
            for pattern in patterns:
                if re.match(pattern, tag.upper()):
                    tag_valid = True
                    break
            
            if not tag_valid:
                result["all_valid"] = False
                result["invalid"].append(tag)
        
        return result

    def _find_critical_discrepancy(self, discrepancies: List[str]) -> Optional[str]:
        """Find the most critical discrepancy from the list."""
        
        if not discrepancies:
            return None
        
        # Priority order for discrepancies
        priority_keywords = [
            "mismatch",
            "unknown floor",
            "impossible",
            "not in knowledge",
            "verification failed",
            "duplicate",
            "invalid"
        ]
        
        # Find highest priority discrepancy
        for keyword in priority_keywords:
            for discrepancy in discrepancies:
                if keyword in discrepancy.lower():
                    return discrepancy
        
        # Return first discrepancy if no priority match
        return discrepancies[0]

    def _calculate_trust_score(self, status: str, confidence: float) -> float:
        """Calculate trust score based on validation status and confidence."""
        
        # Status-based multipliers
        status_multipliers = {
            "PERFECT_MATCH": 1.0,
            "GROUND_TRUTH_VERIFIED": 1.0,
            "SPATIALLY_LOGICAL": 0.95,
            "ALL_REFERENCES_VALID": 0.95,
            "CLOSE_MATCH": 0.90,
            "CONFIRMED_ABSENT": 0.90,
            "MINOR_ISSUES": 0.85,
            "PARTIAL_VERIFICATION": 0.80,
            "SPATIAL_CONCERNS": 0.75,
            "MISMATCH": 0.70,
            "NOT_IN_KNOWLEDGE": 0.65,
            "VERIFICATION_FAILED": 0.60,
            "MULTIPLE_ISSUES": 0.55,
            "NO_KNOWLEDGE_BASE": 0.50,
            "NO_GROUND_TRUTH": 0.50,
            "NO_SPATIAL_DATA": 0.50
        }
        
        multiplier = status_multipliers.get(status, 0.50)
        
        # Calculate trust score
        trust_score = confidence * multiplier
        
        # Ensure within bounds
        return min(0.99, max(0.01, trust_score))

    def _create_failed_validation(self, pass_number: int, methodology: str) -> ValidationResult:
        """Create a standardized failed validation result."""
        
        return ValidationResult(
            pass_number=pass_number,
            methodology=methodology,
            findings={
                "status": "VALIDATION_FAILED",
                "details": ["Validation method encountered an error"],
                "discrepancies": ["Could not complete validation"]
            },
            confidence=0.0,
            trust_score=0.0,
            discrepancies=["Validation failed to execute"],
            cross_references=[]
        )