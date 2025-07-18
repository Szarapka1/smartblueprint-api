# validation_system.py
import asyncio
import re
import logging
import json
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
    4x Validation System - CRITICAL for construction accuracy
    Ensures perfect accuracy through multiple validation methods
    
    ENHANCED: Better support for multi-page document validation
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None  # Will be set by core
        
        # Validation methods in order of importance
        self.validation_methods = [
            ("visual_pattern_verification", "Verify each element matches expected visual patterns"),
            ("count_reconciliation", "Reconcile count with schedules and specifications"),
            ("spatial_consistency", "Verify spatial distribution and locations make sense"),
            ("cross_reference_validation", "Validate against drawing standards and text")
        ]
        
        # Confidence thresholds
        self.confidence_thresholds = {
            "perfect": 0.95,      # All validations pass perfectly
            "high": 0.90,         # Minor discrepancies only
            "medium": 0.80,       # Some validation concerns
            "low": 0.70,          # Major discrepancies
            "failed": 0.50        # Validation failure
        }
    
    def set_vision_client(self, client):
        """Set the vision client for validation checks"""
        self.vision_client = client
    
    async def validate(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> List[ValidationResult]:
        """
        Perform 4x validation of visual findings
        CRITICAL: This ensures construction accuracy
        
        ENHANCED: Better handling of multi-page results
        """
        
        pages_analyzed = visual_result.analysis_metadata.get("pages_analyzed", 1) if hasattr(visual_result, 'analysis_metadata') else 1
        page_context = f"across {pages_analyzed} pages" if pages_analyzed > 1 else "on 1 page"
        
        logger.info(f"🛡️ Starting 4x validation for {visual_result.count} {visual_result.element_type}(s) {page_context}")
        
        # Create validation tasks
        tasks = []
        for i, (method, description) in enumerate(self.validation_methods, 1):
            tasks.append(
                self._perform_single_validation(
                    i, method, description, visual_result, 
                    comprehensive_data, question_analysis
                )
            )
        
        # Execute all validations with timeout
        try:
            validation_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=CONFIG.get("validation_timeout", 60.0)
            )
        except asyncio.TimeoutError:
            logger.error("⚠️ Validation timeout - using partial results")
            validation_results = []
        
        # Process results
        processed_results = []
        for i, result in enumerate(validation_results, 1):
            if isinstance(result, Exception):
                logger.error(f"Validation {i} failed with error: {result}")
                processed_results.append(self._create_failed_validation(i, self.validation_methods[i-1][0]))
            elif isinstance(result, ValidationResult):
                processed_results.append(result)
            else:
                # Shouldn't happen but handle gracefully
                processed_results.append(self._create_failed_validation(i, "unknown"))
        
        # Ensure we have all 4 validations
        while len(processed_results) < 4:
            processed_results.append(
                self._create_failed_validation(len(processed_results) + 1, "missing")
            )
        
        logger.info(f"✅ Completed {len(processed_results)} validations")
        return processed_results
    
    async def _perform_single_validation(
        self,
        pass_number: int,
        methodology: str,
        description: str,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> ValidationResult:
        """Perform a single validation pass"""
        
        logger.info(f"🔍 Validation {pass_number}: {methodology}")
        
        try:
            # Build validation prompt based on methodology
            validation_prompt = self._build_validation_prompt(
                methodology, visual_result, comprehensive_data, question_analysis
            )
            
            # Use all loaded images for validation
            images = comprehensive_data.get("images", [])
            content = [
                {"type": "image_url", "image_url": {"url": image["url"], "detail": "high"}}
                for image in images
            ]
            content.append({"type": "text", "text": validation_prompt})
            
            # Perform validation
            response = await asyncio.wait_for(
                self.vision_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_validation_system_prompt(methodology)
                        },
                        {"role": "user", "content": content}
                    ],
                    max_tokens=3000,
                    temperature=0.0,
                    seed=42  # Deterministic
                ),
                timeout=CONFIG.get("validation_timeout", 60.0) / 2
            )
            
            if response and response.choices:
                validation_content = response.choices[0].message.content or ""
                return self._parse_validation_result(
                    pass_number, methodology, validation_content, visual_result
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Validation {methodology} timed out")
        except Exception as e:
            logger.error(f"Validation {methodology} error: {e}")
        
        return self._create_failed_validation(pass_number, methodology)
    
    def _build_validation_prompt(
        self,
        methodology: str,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> str:
        """
        Build specific validation prompt based on methodology
        ENHANCED: Better support for multi-page validation
        """
        
        element_type = visual_result.element_type
        count = visual_result.count
        locations = visual_result.grid_references[:10]  # First 10 for brevity
        extracted_text = comprehensive_data.get("context", "")[:1500]  # First 1500 chars
        
        # Get multi-page context
        pages_analyzed = visual_result.analysis_metadata.get("pages_analyzed", 1) if hasattr(visual_result, 'analysis_metadata') else 1
        page_counts = visual_result.analysis_metadata.get("page_counts", {}) if hasattr(visual_result, 'analysis_metadata') else {}
        analyzed_pages = visual_result.analysis_metadata.get("analyzed_pages", []) if hasattr(visual_result, 'analysis_metadata') else []
        
        # Multi-page context string
        multi_page_context = ""
        if pages_analyzed > 1:
            multi_page_context = f"""
MULTI-PAGE CONTEXT:
- Total Pages Analyzed: {pages_analyzed}
- Pages: {analyzed_pages}
- Per-Page Breakdown: {json.dumps(page_counts, indent=2) if page_counts else "Not available"}
"""
        
        base_prompt = f"""VALIDATION TASK: {methodology}

VISUAL INTELLIGENCE FINDINGS TO VALIDATE:
- Element Type: {element_type}
- Count Detected: {count}
- Sample Locations: {', '.join(locations)} {f'(+{len(visual_result.grid_references)-10} more)' if len(visual_result.grid_references) > 10 else ''}
- Total Pages Analyzed: {pages_analyzed}
{multi_page_context}

IMPORTANT: You are looking at ALL {len(comprehensive_data.get('images', []))} pages that were analyzed."""

        # Add extracted text context
        if extracted_text:
            base_prompt += f"""

EXTRACTED TEXT/CONTEXT:
{extracted_text}

USE THIS TEXT TO:
- Verify element tags and labels
- Check for elements mentioned but not visually found
- Cross-reference schedules and notes
- Validate against specifications"""

        # Add methodology-specific instructions
        if methodology == "visual_pattern_verification":
            base_prompt += self._get_visual_pattern_prompt(element_type, pages_analyzed, page_counts)
            
        elif methodology == "count_reconciliation":
            base_prompt += self._get_count_reconciliation_prompt(element_type, count, pages_analyzed, page_counts)
            
        elif methodology == "spatial_consistency":
            base_prompt += self._get_spatial_consistency_prompt(element_type, count, pages_analyzed, page_counts)
            
        elif methodology == "cross_reference_validation":
            base_prompt += self._get_cross_reference_prompt(element_type, pages_analyzed)

        return base_prompt
    
    def _get_visual_pattern_prompt(self, element_type: str, pages_analyzed: int, page_counts: Dict[int, int]) -> str:
        """Get visual pattern verification prompt - ENHANCED for multi-page"""
        prompt = f"""

VISUAL PATTERN VERIFICATION:
1. CHECK each detected {element_type} matches expected visual patterns
2. VERIFY symbols are correctly identified as {element_type}s
3. LOOK for any missed {element_type}s that match the pattern
4. CONFIRM no false positives (other elements mistaken for {element_type}s)"""

        if pages_analyzed > 1:
            prompt += f"""
5. VERIFY consistency of {element_type} symbols across different pages
6. CHECK for any page-specific variations in how {element_type}s are shown
7. CONFIRM no duplicate counting between pages"""

        prompt += """

PROVIDE:
- Verified Count: [number after verification]
- Pattern Match Quality: [EXCELLENT/GOOD/FAIR/POOR]
- Missed Elements: [list any found during verification]
- False Positives: [list any incorrect identifications]"""

        if pages_analyzed > 1:
            prompt += """
- Cross-Page Consistency: [CONSISTENT/VARIED/INCONSISTENT]
- Duplicate Risk: [NONE/LOW/MEDIUM/HIGH]"""

        prompt += """
- Confidence: [HIGH/MEDIUM/LOW]"""

        return prompt
    
    def _get_count_reconciliation_prompt(self, element_type: str, count: int, pages_analyzed: int, page_counts: Dict[int, int]) -> str:
        """Get count reconciliation prompt - ENHANCED for multi-page"""
        prompt = f"""

COUNT RECONCILIATION:
1. SEARCH for any {element_type} schedule, legend, or specification table
2. IF FOUND, count entries and compare to visual count of {count}
3. CHECK for any notes indicating quantity (e.g., "provide 12 outlets")
4. RECONCILE any differences between visual count and documented count"""

        if pages_analyzed > 1:
            prompt += f"""
5. VERIFY the sum of page counts equals the total: {sum(page_counts.values()) if page_counts else "N/A"} = {count}?
6. CHECK if schedule shows per-floor or per-area breakdowns
7. ENSURE no elements are counted on multiple pages"""

        prompt += """

PROVIDE:
- Schedule Found: [YES/NO]
- Schedule Count: [number if found]
- Visual Count: """ + str(count)

        if pages_analyzed > 1 and page_counts:
            prompt += f"""
- Page Count Sum: {sum(page_counts.values())}
- Discrepancy: [{sum(page_counts.values()) - count} if different]"""

        prompt += """
- Other Quantity References: [any notes about quantities]
- Match Status: [PERFECT_MATCH/CLOSE_MATCH/MISMATCH/NO_SCHEDULE]
- Reconciliation: [explain any differences]"""

        return prompt
    
    def _get_spatial_consistency_prompt(self, element_type: str, count: int, pages_analyzed: int, page_counts: Dict[int, int]) -> str:
        """Get spatial consistency prompt - ENHANCED for multi-page"""
        prompt = f"""

SPATIAL CONSISTENCY CHECK:
1. ANALYZE the distribution of {count} {element_type}s
2. CHECK if locations make engineering sense
3. VERIFY spacing and placement follows standards
4. IDENTIFY any illogical placements or gaps"""

        if pages_analyzed > 1:
            prompt += f"""
5. ANALYZE distribution across pages: {json.dumps(page_counts, indent=2) if page_counts else "See visual count"}
6. VERIFY each floor/area has appropriate coverage
7. CHECK for missing coverage on any page
8. ENSURE consistent spacing standards across pages"""

        prompt += f"""

QUESTIONS TO ANSWER:
- Are {element_type}s placed where expected? (e.g., outlets near walls, sprinklers in ceilings)
- Is the spacing consistent with standards?
- Are there obvious areas missing {element_type}s?
- Do the locations follow a logical pattern?"""

        if pages_analyzed > 1:
            prompt += """
- Is the distribution reasonable across different floors/areas?
- Are any pages suspiciously lacking elements?
- Do similar spaces have similar element densities?"""

        prompt += """

PROVIDE:
- Spatial Logic: [EXCELLENT/GOOD/QUESTIONABLE/POOR]
- Distribution Quality: [UNIFORM/CLUSTERED/IRREGULAR]
- Missing Areas: [list any obvious gaps]
- Placement Issues: [any illogical locations]"""

        if pages_analyzed > 1:
            prompt += """
- Cross-Page Distribution: [BALANCED/UNBALANCED/SUSPICIOUS]
- Coverage Completeness: [COMPLETE/PARTIAL/GAPS_FOUND]"""

        prompt += """
- Engineering Assessment: [makes sense/concerns noted]"""

        return prompt
    
    def _get_cross_reference_prompt(self, element_type: str, pages_analyzed: int) -> str:
        """Get cross-reference validation prompt - ENHANCED for multi-page"""
        prompt = f"""

CROSS-REFERENCE VALIDATION:
1. CHECK {element_type} symbols against drawing legend/standards
2. VERIFY all tagged elements (e.g., P1, D1) are accounted for
3. VALIDATE against any general notes or specifications
4. CONFIRM drawing scale and type appropriate for {element_type}s"""

        if pages_analyzed > 1:
            prompt += """
5. VERIFY symbol consistency across all pages
6. CHECK if different disciplines use different symbols
7. CONFIRM all referenced elements from notes are found
8. VALIDATE sheet references (e.g., "SEE SHEET E-2")"""

        prompt += f"""

VERIFY:
- Symbol Standards: Are {element_type} symbols per standard?
- Tag Verification: Are all tagged {element_type}s found?
- Notes Compliance: Do findings match general notes?
- Drawing Appropriateness: Is this the right drawing type?"""

        if pages_analyzed > 1:
            prompt += """
- Multi-Sheet Consistency: Are symbols consistent across sheets?
- Cross-References: Are sheet references valid?"""

        prompt += """

PROVIDE:
- Standards Compliance: [FULLY_COMPLIANT/MOSTLY_COMPLIANT/ISSUES_FOUND]
- Tag Accountability: [ALL_FOUND/SOME_MISSING/NOT_APPLICABLE]
- Note Alignment: [MATCHES/DISCREPANCIES/NO_NOTES]"""

        if pages_analyzed > 1:
            prompt += """
- Symbol Consistency: [CONSISTENT/MINOR_VARIATIONS/MAJOR_VARIATIONS]
- Sheet Reference Validity: [ALL_VALID/SOME_INVALID/NOT_APPLICABLE]"""

        prompt += """
- Overall Validation: [VALIDATED/CONCERNS/FAILED]"""

        return prompt
    
    def _get_validation_system_prompt(self, methodology: str) -> str:
        """
        Get system prompt for validation
        ENHANCED: Multi-page awareness
        """
        
        base_prompt = f"""You are a construction validation expert performing {methodology}.
Your role is CRITICAL for ensuring accuracy. You must:
1. Be thorough and systematic in your validation
2. Actually check the drawings, don't just agree with the findings
3. Report any discrepancies honestly
4. Provide specific examples when issues are found
5. Give accurate confidence assessments"""

        # Add multi-page specific instructions
        if "multi" in methodology.lower() or "page" in methodology.lower():
            base_prompt += """
6. Pay special attention to cross-page consistency
7. Check for duplicate counting between pages
8. Verify total counts match sum of page counts"""

        base_prompt += """

This validation ensures construction professionals can trust the analysis.
Be rigorous - construction safety and costs depend on accuracy."""

        return base_prompt
    
    def _parse_validation_result(
        self,
        pass_number: int,
        methodology: str,
        response: str,
        visual_result: VisualIntelligenceResult
    ) -> ValidationResult:
        """
        Parse validation response into structured result
        ENHANCED: Better handling of multi-page validation results
        """
        
        result = ValidationResult(
            pass_number=pass_number,
            methodology=methodology,
            findings={},
            confidence=0.85,  # Default
            trust_score=0.85
        )
        
        # Extract verified count
        count_match = re.search(r'Verified Count:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            result.findings["count"] = int(count_match.group(1))
        else:
            result.findings["count"] = visual_result.count
        
        # Check if this is multi-page
        pages_analyzed = visual_result.analysis_metadata.get("pages_analyzed", 1) if hasattr(visual_result, 'analysis_metadata') else 1
        result.findings["pages_analyzed"] = pages_analyzed
        
        # Methodology-specific parsing
        if methodology == "visual_pattern_verification":
            self._parse_visual_pattern_result(result, response, pages_analyzed)
        elif methodology == "count_reconciliation":
            self._parse_count_reconciliation_result(result, response, pages_analyzed)
        elif methodology == "spatial_consistency":
            self._parse_spatial_consistency_result(result, response, pages_analyzed)
        elif methodology == "cross_reference_validation":
            self._parse_cross_reference_result(result, response, pages_analyzed)
        
        # Calculate final confidence and trust score
        result.confidence, result.trust_score = self._calculate_validation_confidence(
            result, visual_result, methodology
        )
        
        logger.info(f"✅ {methodology}: Confidence={result.confidence:.2f}, Trust={result.trust_score:.2f}")
        
        return result
    
    def _parse_visual_pattern_result(self, result: ValidationResult, response: str, pages_analyzed: int):
        """
        Parse visual pattern verification results
        ENHANCED: Multi-page pattern checks
        """
        
        # Pattern match quality
        quality_match = re.search(r'Pattern Match Quality:\s*(\w+)', response, re.IGNORECASE)
        if quality_match:
            quality = quality_match.group(1).upper()
            result.findings["pattern_quality"] = quality
            result.pattern_matches_verified = ["EXCELLENT", "GOOD"].count(quality) > 0
        
        # Check for missed elements
        missed_match = re.search(r'Missed Elements:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if missed_match and "none" not in missed_match.group(1).lower():
            result.discrepancies.append(f"Missed elements: {missed_match.group(1)}")
            result.findings["has_missed_elements"] = True
        
        # Check for false positives
        false_match = re.search(r'False Positives:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if false_match and "none" not in false_match.group(1).lower():
            result.discrepancies.append(f"False positives: {false_match.group(1)}")
            result.findings["has_false_positives"] = True
        
        # Multi-page specific checks
        if pages_analyzed > 1:
            # Cross-page consistency
            consistency_match = re.search(r'Cross-Page Consistency:\s*(\w+)', response, re.IGNORECASE)
            if consistency_match:
                result.findings["cross_page_consistency"] = consistency_match.group(1).upper()
            
            # Duplicate risk
            duplicate_match = re.search(r'Duplicate Risk:\s*(\w+)', response, re.IGNORECASE)
            if duplicate_match:
                duplicate_risk = duplicate_match.group(1).upper()
                result.findings["duplicate_risk"] = duplicate_risk
                if duplicate_risk in ["MEDIUM", "HIGH"]:
                    result.discrepancies.append(f"Duplicate counting risk: {duplicate_risk}")
        
        # Set status
        if quality_match:
            quality = result.findings.get("pattern_quality", "UNKNOWN")
            if quality in ["EXCELLENT", "GOOD"]:
                result.findings["status"] = "confirmed"
            elif quality == "FAIR":
                result.findings["status"] = "partial"
            else:
                result.findings["status"] = "poor"
    
    def _parse_count_reconciliation_result(self, result: ValidationResult, response: str, pages_analyzed: int):
        """
        Parse count reconciliation results
        ENHANCED: Multi-page count validation
        """
        
        # Schedule found
        schedule_found = "Schedule Found: YES" in response or "SCHEDULE_FOUND: YES" in response
        result.findings["schedule_found"] = schedule_found
        
        # Schedule count
        schedule_count_match = re.search(r'Schedule Count:\s*(\d+)', response, re.IGNORECASE)
        if schedule_count_match:
            result.findings["schedule_count"] = int(schedule_count_match.group(1))
        
        # Visual count
        visual_count_match = re.search(r'Visual Count:\s*(\d+)', response, re.IGNORECASE)
        if visual_count_match:
            result.findings["visual_count"] = int(visual_count_match.group(1))
        
        # Multi-page specific
        if pages_analyzed > 1:
            # Page count sum
            sum_match = re.search(r'Page Count Sum:\s*(\d+)', response, re.IGNORECASE)
            if sum_match:
                page_sum = int(sum_match.group(1))
                result.findings["page_count_sum"] = page_sum
                
                # Check if sum matches total
                visual_count = result.findings.get("visual_count", result.findings.get("count", 0))
                if page_sum != visual_count:
                    result.discrepancies.append(f"Page sum ({page_sum}) ≠ total count ({visual_count})")
        
        # Match status
        match_status = "NO_SCHEDULE"
        if re.search(r'PERFECT_MATCH|MATCHES?', response, re.IGNORECASE):
            match_status = "PERFECT_MATCH"
            result.findings["status"] = "confirmed"
        elif re.search(r'CLOSE_MATCH|CLOSE', response, re.IGNORECASE):
            match_status = "CLOSE_MATCH"
            result.findings["status"] = "close"
        elif re.search(r'MISMATCH|DISCREPANCY', response, re.IGNORECASE):
            match_status = "MISMATCH"
            result.findings["status"] = "mismatch"
            result.discrepancies.append("Count mismatch with schedule")
        
        result.findings["match_status"] = match_status
        
        # Cross-references
        if schedule_found:
            result.cross_references.append(f"Schedule verification: {match_status}")
    
    def _parse_spatial_consistency_result(self, result: ValidationResult, response: str, pages_analyzed: int):
        """
        Parse spatial consistency results
        ENHANCED: Multi-page distribution checks
        """
        
        # Spatial logic assessment
        logic_match = re.search(r'Spatial Logic:\s*(\w+)', response, re.IGNORECASE)
        if logic_match:
            logic = logic_match.group(1).upper()
            result.findings["spatial_logic"] = logic
            
            if logic in ["EXCELLENT", "GOOD"]:
                result.findings["status"] = "confirmed"
            elif logic == "QUESTIONABLE":
                result.findings["status"] = "concerns"
                result.discrepancies.append("Questionable spatial distribution")
            else:
                result.findings["status"] = "poor"
                result.discrepancies.append("Poor spatial distribution")
        
        # Distribution quality
        dist_match = re.search(r'Distribution Quality:\s*(\w+)', response, re.IGNORECASE)
        if dist_match:
            result.findings["distribution"] = dist_match.group(1).upper()
        
        # Missing areas
        missing_match = re.search(r'Missing Areas:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if missing_match and "none" not in missing_match.group(1).lower():
            result.discrepancies.append(f"Missing areas: {missing_match.group(1)}")
            result.findings["has_gaps"] = True
        
        # Multi-page specific
        if pages_analyzed > 1:
            # Cross-page distribution
            cross_dist_match = re.search(r'Cross-Page Distribution:\s*(\w+)', response, re.IGNORECASE)
            if cross_dist_match:
                cross_dist = cross_dist_match.group(1).upper()
                result.findings["cross_page_distribution"] = cross_dist
                if cross_dist in ["UNBALANCED", "SUSPICIOUS"]:
                    result.discrepancies.append(f"Cross-page distribution: {cross_dist}")
            
            # Coverage completeness
            coverage_match = re.search(r'Coverage Completeness:\s*(\w+)', response, re.IGNORECASE)
            if coverage_match:
                coverage = coverage_match.group(1).upper()
                result.findings["coverage_completeness"] = coverage
                if coverage == "GAPS_FOUND":
                    result.findings["has_gaps"] = True
    
    def _parse_cross_reference_result(self, result: ValidationResult, response: str, pages_analyzed: int):
        """
        Parse cross-reference validation results
        ENHANCED: Multi-sheet consistency checks
        """
        
        # Standards compliance
        standards_match = re.search(r'Standards Compliance:\s*(\w+)', response, re.IGNORECASE)
        if standards_match:
            compliance = standards_match.group(1).upper()
            result.findings["standards_compliance"] = compliance
            
            if "FULLY_COMPLIANT" in compliance:
                result.findings["status"] = "confirmed"
            elif "MOSTLY_COMPLIANT" in compliance:
                result.findings["status"] = "mostly_confirmed"
            else:
                result.findings["status"] = "issues"
                result.discrepancies.append("Standards compliance issues")
        
        # Tag accountability
        tag_match = re.search(r'Tag Accountability:\s*(\w+)', response, re.IGNORECASE)
        if tag_match:
            tag_status = tag_match.group(1).upper()
            result.findings["tag_accountability"] = tag_status
            if "MISSING" in tag_status:
                result.discrepancies.append("Some tagged elements missing")
        
        # Multi-page specific
        if pages_analyzed > 1:
            # Symbol consistency
            symbol_match = re.search(r'Symbol Consistency:\s*(\w+)', response, re.IGNORECASE)
            if symbol_match:
                symbol_consistency = symbol_match.group(1).upper()
                result.findings["symbol_consistency"] = symbol_consistency
                if "MAJOR_VARIATIONS" in symbol_consistency:
                    result.discrepancies.append("Major symbol variations across sheets")
            
            # Sheet reference validity
            ref_match = re.search(r'Sheet Reference Validity:\s*(\w+)', response, re.IGNORECASE)
            if ref_match:
                ref_validity = ref_match.group(1).upper()
                result.findings["sheet_reference_validity"] = ref_validity
                if "INVALID" in ref_validity:
                    result.discrepancies.append("Invalid sheet references found")
        
        # Overall validation
        overall_match = re.search(r'Overall Validation:\s*(\w+)', response, re.IGNORECASE)
        if overall_match:
            overall = overall_match.group(1).upper()
            if overall == "VALIDATED":
                result.cross_references.append("Drawing standards validated")
            elif overall == "FAILED":
                result.findings["status"] = "failed"
                result.discrepancies.append("Overall validation failed")
    
    def _calculate_validation_confidence(
        self,
        result: ValidationResult,
        visual_result: VisualIntelligenceResult,
        methodology: str
    ) -> Tuple[float, float]:
        """
        Calculate confidence and trust scores for validation
        CRITICAL: This determines if we can trust the results
        
        ENHANCED: Consider multi-page factors
        """
        
        base_confidence = 0.7  # Start conservative
        trust_score = 0.7
        
        # Get multi-page context
        pages_analyzed = result.findings.get("pages_analyzed", 1)
        
        # Status-based scoring
        status = result.findings.get("status", "unknown")
        status_scores = {
            "confirmed": 0.95,
            "mostly_confirmed": 0.90,
            "close": 0.85,
            "partial": 0.80,
            "concerns": 0.70,
            "mismatch": 0.60,
            "poor": 0.50,
            "failed": 0.40
        }
        
        if status in status_scores:
            base_confidence = status_scores[status]
        
        # Methodology-specific adjustments
        if methodology == "visual_pattern_verification":
            # Visual pattern is most important
            if result.findings.get("pattern_quality") in ["EXCELLENT", "GOOD"]:
                trust_score = base_confidence
            else:
                trust_score = base_confidence * 0.9
            
            # Multi-page adjustments
            if pages_analyzed > 1:
                consistency = result.findings.get("cross_page_consistency", "UNKNOWN")
                if consistency == "CONSISTENT":
                    base_confidence *= 1.05  # Boost for consistency
                    trust_score *= 1.05
                elif consistency == "INCONSISTENT":
                    base_confidence *= 0.85
                    trust_score *= 0.85
                
                # Duplicate risk penalty
                dup_risk = result.findings.get("duplicate_risk", "UNKNOWN")
                if dup_risk in ["MEDIUM", "HIGH"]:
                    base_confidence *= 0.9
                    trust_score *= 0.9
            
            # Penalties for issues
            if result.findings.get("has_missed_elements"):
                base_confidence *= 0.85
                trust_score *= 0.85
            if result.findings.get("has_false_positives"):
                base_confidence *= 0.90
                trust_score *= 0.90
                
        elif methodology == "count_reconciliation":
            # Schedule match is very important
            if result.findings.get("schedule_found"):
                schedule_count = result.findings.get("schedule_count", 0)
                visual_count = result.findings.get("count", visual_result.count)
                
                if schedule_count == visual_count:
                    base_confidence = 0.99  # Perfect match
                    trust_score = 0.99
                elif abs(schedule_count - visual_count) <= 1:
                    base_confidence = 0.95  # Close match
                    trust_score = 0.95
                else:
                    # Major discrepancy
                    base_confidence = 0.50
                    trust_score = 0.50
            
            # Multi-page sum check
            if pages_analyzed > 1 and "page_count_sum" in result.findings:
                page_sum = result.findings["page_count_sum"]
                total_count = result.findings.get("count", visual_result.count)
                if page_sum != total_count:
                    base_confidence *= 0.85
                    trust_score *= 0.85
                    
        elif methodology == "spatial_consistency":
            # Spatial logic check
            spatial_logic = result.findings.get("spatial_logic", "UNKNOWN")
            if spatial_logic in ["EXCELLENT", "GOOD"]:
                trust_score = base_confidence
            else:
                trust_score = base_confidence * 0.85
            
            # Multi-page distribution
            if pages_analyzed > 1:
                cross_dist = result.findings.get("cross_page_distribution", "UNKNOWN")
                if cross_dist == "BALANCED":
                    base_confidence *= 1.05
                elif cross_dist in ["UNBALANCED", "SUSPICIOUS"]:
                    base_confidence *= 0.85
                    trust_score *= 0.85
            
            # Gaps are concerning
            if result.findings.get("has_gaps"):
                base_confidence *= 0.80
                trust_score *= 0.80
                
        elif methodology == "cross_reference_validation":
            # Standards compliance
            compliance = result.findings.get("standards_compliance", "UNKNOWN")
            if "FULLY_COMPLIANT" in compliance:
                trust_score = base_confidence
            else:
                trust_score = base_confidence * 0.90
            
            # Multi-page symbol consistency
            if pages_analyzed > 1:
                symbol_consistency = result.findings.get("symbol_consistency", "UNKNOWN")
                if symbol_consistency == "CONSISTENT":
                    base_confidence *= 1.05
                elif symbol_consistency == "MAJOR_VARIATIONS":
                    base_confidence *= 0.85
                    trust_score *= 0.85
        
        # Count consistency check
        if result.findings.get("count") != visual_result.count:
            difference = abs(result.findings.get("count", 0) - visual_result.count)
            if difference > 2:
                base_confidence *= 0.70
                trust_score *= 0.70
            elif difference > 0:
                base_confidence *= 0.90
                trust_score *= 0.90
        
        # Discrepancy penalties
        discrepancy_penalty = min(0.05 * len(result.discrepancies), 0.20)
        base_confidence -= discrepancy_penalty
        trust_score -= discrepancy_penalty
        
        # Multi-page complexity adjustment
        if pages_analyzed > 10:
            # Slightly lower confidence for very large documents
            complexity_factor = 0.95
            base_confidence *= complexity_factor
            trust_score *= complexity_factor
        
        # Ensure bounds
        confidence = max(0.0, min(1.0, base_confidence))
        trust = max(0.0, min(1.0, trust_score))
        
        return confidence, trust
    
    async def build_consensus(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build consensus from all validation results
        ENHANCED: Better handling of multi-page consensus
        """
        
        # Extract all counts
        all_counts = [visual_result.count]
        validation_counts = []
        
        for val_result in validation_results:
            if val_result.findings.get("count") is not None:
                validation_counts.append(val_result.findings["count"])
                all_counts.append(val_result.findings["count"])
        
        unique_counts = list(set(all_counts))
        
        # Check for schedule discrepancy
        schedule_found = any(
            v.findings.get("schedule_found") or v.findings.get("schedule_count") is not None
            for v in validation_results
        )
        
        schedule_count = None
        for v in validation_results:
            if v.findings.get("schedule_count") is not None:
                schedule_count = v.findings["schedule_count"]
                break
        
        # Check multi-page consistency
        pages_analyzed = visual_result.analysis_metadata.get("pages_analyzed", 1) if hasattr(visual_result, 'analysis_metadata') else 1
        page_count_issues = any(
            v.findings.get("page_count_sum", 0) != visual_result.count
            for v in validation_results
            if "page_count_sum" in v.findings
        )
        
        # Build consensus result
        consensus = {
            "visual_count": visual_result.count,
            "validation_counts": validation_counts,
            "all_counts": all_counts,
            "unique_counts": unique_counts,
            "consensus_count": visual_result.count,
            "validation_agreement": "PENDING",
            "schedule_found": schedule_found,
            "schedule_count": schedule_count,
            "pages_analyzed": pages_analyzed,  # NEW
            "page_count_consistent": not page_count_issues  # NEW
        }
        
        # Determine consensus level
        if schedule_found and schedule_count is not None and visual_result.count == 0:
            # CRITICAL: Schedule exists but no visual elements
            consensus["validation_agreement"] = "MAJOR_DISCREPANCY"
            consensus["consensus_confidence"] = 0.0
            consensus["discrepancy_note"] = f"Schedule shows {schedule_count} items but none found visually"
            
        elif page_count_issues and pages_analyzed > 1:
            # Multi-page count inconsistency
            consensus["validation_agreement"] = "PAGE_COUNT_MISMATCH"
            consensus["consensus_confidence"] = 0.70
            consensus["discrepancy_note"] = "Sum of page counts doesn't match total count"
            
        elif len(unique_counts) == 1:
            # Perfect consensus
            consensus["validation_agreement"] = "PERFECT_CONSENSUS"
            consensus["consensus_confidence"] = 1.0
            
        elif len(unique_counts) == 2 and max(unique_counts) - min(unique_counts) <= 1:
            # Near consensus (off by 1)
            consensus["validation_agreement"] = "NEAR_CONSENSUS"
            consensus["consensus_confidence"] = 0.95
            consensus["consensus_count"] = int(sum(all_counts) / len(all_counts) + 0.5)  # Round
            
        elif len(unique_counts) == 2 and max(unique_counts) - min(unique_counts) <= 2:
            # Acceptable variance
            consensus["validation_agreement"] = "ACCEPTABLE_VARIANCE"
            consensus["consensus_confidence"] = 0.85
            consensus["consensus_count"] = max(set(all_counts), key=all_counts.count)  # Mode
            
        else:
            # Requires reanalysis
            consensus["validation_agreement"] = "REQUIRES_REANALYSIS"
            consensus["consensus_confidence"] = 0.60
            consensus["discrepancy_note"] = f"Counts vary significantly: {unique_counts}"
        
        # Check validation quality
        high_confidence_validations = sum(
            1 for v in validation_results if v.confidence >= 0.90
        )
        
        if high_confidence_validations < 2:
            consensus["validation_quality"] = "LOW"
            consensus["consensus_confidence"] *= 0.8
        elif high_confidence_validations >= 3:
            consensus["validation_quality"] = "HIGH"
        else:
            consensus["validation_quality"] = "MEDIUM"
        
        # Multi-page specific quality checks
        if pages_analyzed > 1:
            # Check for consistent findings across validators
            cross_page_issues = sum(
                1 for v in validation_results
                if v.findings.get("cross_page_consistency") == "INCONSISTENT"
                or v.findings.get("duplicate_risk") in ["MEDIUM", "HIGH"]
                or v.findings.get("cross_page_distribution") in ["UNBALANCED", "SUSPICIOUS"]
            )
            
            if cross_page_issues >= 2:
                consensus["multi_page_concerns"] = True
                consensus["consensus_confidence"] *= 0.9
        
        logger.info(f"🤝 Consensus: {consensus['validation_agreement']} " +
                   f"(confidence: {consensus['consensus_confidence']:.2f})")
        
        return consensus
    
    def calculate_trust_metrics(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any]
    ) -> TrustMetrics:
        """
        Calculate final trust metrics
        CRITICAL: This determines if results can be trusted
        
        ENHANCED: Consider multi-page complexity
        """
        
        metrics = TrustMetrics()
        
        # Base scores
        metrics.visual_intelligence_score = visual_result.confidence
        
        # Calculate validation scores
        validation_scores = [v.trust_score for v in validation_results if v.trust_score > 0]
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.5
        
        # Multi-page complexity factor
        pages_analyzed = consensus_result.get("pages_analyzed", 1)
        if pages_analyzed > 10:
            complexity_factor = 0.95  # Slightly lower trust for very complex documents
        elif pages_analyzed > 5:
            complexity_factor = 0.98
        else:
            complexity_factor = 1.0
        
        # Perfect accuracy conditions
        perfect_conditions = [
            consensus_result["validation_agreement"] == "PERFECT_CONSENSUS",
            visual_result.confidence >= 0.95,
            avg_validation_score >= 0.90,
            all(v.confidence >= 0.85 for v in validation_results if v.findings.get("status") != "failed"),
            not consensus_result.get("discrepancy_note"),
            consensus_result.get("page_count_consistent", True)  # NEW condition
        ]
        
        if all(perfect_conditions):
            metrics.perfect_accuracy_achieved = True
            metrics.reliability_score = 0.99 * complexity_factor
            metrics.confidence_basis = "Perfect consensus with high confidence across all validations"
            
        elif consensus_result["validation_agreement"] == "NEAR_CONSENSUS" and avg_validation_score >= 0.85:
            metrics.perfect_accuracy_achieved = False
            metrics.reliability_score = min(0.95, (visual_result.confidence + avg_validation_score) / 2) * complexity_factor
            metrics.confidence_basis = "Near consensus with good validation scores"
            
        elif consensus_result["validation_agreement"] == "PAGE_COUNT_MISMATCH":
            metrics.perfect_accuracy_achieved = False
            metrics.reliability_score = 0.70 * complexity_factor
            metrics.confidence_basis = "Page count inconsistency detected"
            metrics.uncertainty_factors.append("Sum of page counts doesn't match total")
            
        elif consensus_result["validation_agreement"] == "MAJOR_DISCREPANCY":
            metrics.perfect_accuracy_achieved = False
            metrics.reliability_score = 0.30
            metrics.confidence_basis = "Major discrepancy detected between visual and documented counts"
            metrics.uncertainty_factors.append("Schedule/visual count mismatch")
            
        else:
            # Calculate weighted score
            weights = {
                "visual": 0.40,
                "validation": 0.35,
                "consensus": 0.25
            }
            
            metrics.reliability_score = (
                weights["visual"] * visual_result.confidence +
                weights["validation"] * avg_validation_score +
                weights["consensus"] * consensus_result.get("consensus_confidence", 0.5)
            ) * complexity_factor
            
            metrics.perfect_accuracy_achieved = False
            metrics.confidence_basis = "Weighted average of visual and validation scores"
        
        # Set validation consensus flag
        metrics.validation_consensus = consensus_result["validation_agreement"] in [
            "PERFECT_CONSENSUS", "NEAR_CONSENSUS"
        ]
        
        # Accuracy sources
        metrics.accuracy_sources = ["visual_intelligence", "4x_validation", "consensus_analysis"]
        
        # Add specific validation methods that passed
        for v in validation_results:
            if v.confidence >= 0.90:
                metrics.accuracy_sources.append(f"validation_{v.methodology}")
        
        # Multi-page specific sources
        if pages_analyzed > 1:
            if consensus_result.get("page_count_consistent"):
                metrics.accuracy_sources.append("multi_page_consistency")
        
        # Uncertainty factors
        if visual_result.count == 0:
            metrics.uncertainty_factors.append("Zero count requires extra verification")
        
        if pages_analyzed > 10:
            metrics.uncertainty_factors.append(f"Large document complexity ({pages_analyzed} pages)")
        
        if consensus_result.get("multi_page_concerns"):
            metrics.uncertainty_factors.append("Multi-page consistency concerns")
            
        for v in validation_results:
            if v.discrepancies:
                metrics.uncertainty_factors.extend(v.discrepancies[:2])  # First 2
        
        # Quality scores
        for v in validation_results:
            metrics.source_quality_scores[v.methodology] = v.confidence
        
        # Final adjustments
        if consensus_result.get("schedule_found") and consensus_result.get("schedule_count", 0) != visual_result.count:
            if abs(consensus_result.get("schedule_count", 0) - visual_result.count) > 2:
                metrics.reliability_score = min(metrics.reliability_score, 0.70)
                metrics.uncertainty_factors.append("Significant schedule/visual discrepancy")
        
        # Ensure bounds
        metrics.reliability_score = max(0.0, min(1.0, metrics.reliability_score))
        
        logger.info(f"📊 Trust Metrics: Reliability={metrics.reliability_score:.2f}, " +
                   f"Perfect={metrics.perfect_accuracy_achieved}, " +
                   f"Consensus={metrics.validation_consensus}")
        
        return metrics
    
    def _create_failed_validation(self, pass_number: int, methodology: str) -> ValidationResult:
        """Create a failed validation result"""
        
        return ValidationResult(
            pass_number=pass_number,
            methodology=methodology,
            findings={"status": "failed", "count": None},
            confidence=0.0,
            trust_score=0.0,
            discrepancies=["Validation failed to complete"]
        )
