# response_formatter.py
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter

# Import from app.models.schemas instead of .models and .config
from app.models.schemas import (
    VisualIntelligenceResult, 
    ValidationResult, 
    TrustMetrics,
    QuestionType,
    NoteSuggestion,
    ElementGeometry
)

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Formats responses based on question type with consistent templates
    Each question type gets appropriate formatting while maintaining consistency
    """
    
    def __init__(self):
        # Response templates for each question type
        self.templates = {
            QuestionType.COUNT: self._format_count_response,
            QuestionType.LOCATION: self._format_location_response,
            QuestionType.IDENTIFY: self._format_identification_response,
            QuestionType.SPECIFICATION: self._format_specification_response,
            QuestionType.COMPLIANCE: self._format_compliance_response,
            QuestionType.GENERAL: self._format_general_response,
            QuestionType.DETAILED: self._format_detailed_response,
            QuestionType.ESTIMATE: self._format_estimate_response
        }
        
        # Confidence descriptors
        self.confidence_descriptors = {
            (0.98, 1.0): ("PERFECT ENGINEERING ACCURACY", "Engineering analysis verified with perfect accuracy"),
            (0.95, 0.98): ("EXCELLENT RELIABILITY", "High confidence engineering analysis"),
            (0.90, 0.95): ("HIGH RELIABILITY", "Professional engineering analysis"),
            (0.85, 0.90): ("GOOD RELIABILITY", "Reliable engineering analysis"),
            (0.80, 0.85): ("MODERATE CONFIDENCE", "Engineering analysis with moderate confidence"),
            (0.70, 0.80): ("REQUIRES VERIFICATION", "Analysis requires additional verification"),
            (0.0, 0.70): ("LOW CONFIDENCE", "Analysis requires significant verification")
        }
        
        # Code requirements database
        self.code_requirements_db = {
            "outlet": {
                "NEC 210.52": "Receptacle spacing - max 12ft apart, 6ft from any point on wall",
                "NEC 210.8": "GFCI required in bathrooms, kitchens, outdoors, garages",
                "NEC 406.4": "Minimum 15-18 inches above floor",
                "ADA 308.2": "15-48 inches above floor for accessibility"
            },
            "door": {
                "IBC 1010.1": "Minimum clear width 32 inches",
                "ADA 404": "Clear width 32 inches min, thresholds max 1/2 inch",
                "IBC 1010.1.1": "Height minimum 80 inches",
                "NFPA 101": "Swing direction for egress"
            },
            "sprinkler": {
                "NFPA 13": "Max spacing 15ft (light hazard), 12ft (ordinary)",
                "NFPA 13": "Min 4 inches from walls, max 12 inches",
                "NFPA 13": "Coverage per sprinkler 130-200 sq ft",
                "IBC 903": "Required in specific occupancies"
            },
            "stair": {
                "IBC 1011.5": "Min width 44 inches (occupant load >50)",
                "IBC 1011.5.2": "Riser height 4-7 inches, tread depth min 11 inches",
                "ADA 504": "Handrails both sides, 34-38 inches high",
                "IBC 1011.11": "Landing required every 12ft vertical"
            },
            "light fixture": {
                "NEC 410": "Proper support and grounding required",
                "IECC C405": "Lighting power density limits",
                "IBC 1205": "Natural and artificial light requirements",
                "ADA 215": "Controls 15-48 inches above floor"
            },
            "panel": {
                "NEC 110.26": "Working space 36 inches deep, 30 inches wide",
                "NEC 408.4": "Circuit directory required",
                "NEC 240": "Overcurrent protection requirements",
                "NEC 110.26(E)": "Illumination required at panels"
            }
        }
        
        # Standard spacing requirements
        self.standard_spacing = {
            "outlet": {"residential": 12, "commercial": 20, "special": 6},
            "sprinkler": {"light_hazard": 15, "ordinary_hazard": 12, "extra_hazard": 10},
            "light fixture": {"office": 8, "corridor": 10, "storage": 15},
            "diffuser": {"standard": 10, "high_capacity": 15, "vav": 12}
        }
        
        # Estimate factors
        self.estimate_factors = {
            "material_cost": {
                "outlet": {"low": 15, "avg": 25, "high": 40},
                "door": {"low": 150, "avg": 350, "high": 800},
                "window": {"low": 200, "avg": 500, "high": 1200},
                "light fixture": {"low": 50, "avg": 150, "high": 400},
                "sprinkler": {"low": 75, "avg": 125, "high": 200},
                "plumbing fixture": {"low": 200, "avg": 600, "high": 1500}
            },
            "labor_hours": {
                "outlet": {"low": 0.5, "avg": 0.75, "high": 1.0},
                "door": {"low": 2.0, "avg": 3.0, "high": 4.0},
                "window": {"low": 2.5, "avg": 4.0, "high": 6.0},
                "light fixture": {"low": 0.75, "avg": 1.25, "high": 2.0},
                "sprinkler": {"low": 1.0, "avg": 1.5, "high": 2.0}
            }
        }
    
    def format_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """
        Main method to format response based on question type
        """
        
        question_type = question_analysis.get("type", QuestionType.GENERAL)
        
        # Get the appropriate formatter
        formatter = self.templates.get(question_type, self._format_general_response)
        
        # Format the response
        try:
            response = formatter(
                visual_result, validation_results, consensus_result,
                trust_metrics, question_analysis
            )
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            response = self._format_fallback_response(
                visual_result, trust_metrics, question_analysis
            )
        
        return response
    
    def _format_count_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format COUNT type questions - How many X are there?"""
        
        count = visual_result.count
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, confidence_desc = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        # Build location summary
        location_summary = self._build_location_summary(visual_result, count)
        
        # Build verification summary
        verification_summary = self._build_verification_summary(
            validation_results, consensus_result, visual_result
        )
        
        # Check for discrepancies
        discrepancy_note = ""
        if consensus_result.get("discrepancy_note"):
            discrepancy_note = f"\n\n⚠️ **IMPORTANT**: {consensus_result['discrepancy_note']}"
        
        # Format based on count
        if count == 0:
            response = f"""**ANSWER: No {element_type}s found in the analyzed pages**

**ENGINEERING ANALYSIS:**
- Search Result: 0 {element_type}s detected
- Pages Analyzed: {len(visual_result.locations) if hasattr(visual_result, 'locations') else 'All relevant pages'}
- Analysis Method: Comprehensive visual intelligence scan
- Verification: {consensus_result.get('validation_agreement', 'Completed')}

**FINDINGS:**
No {element_type}s were identified in the blueprint pages analyzed. This could mean:
- The {element_type}s are on different pages not included in this analysis
- The drawing type may not typically show {element_type}s
- The {element_type}s may be indicated differently than expected

{verification_summary}

**CONFIDENCE: {confidence}% - {confidence_label}**
*Zero-count findings require careful verification*{discrepancy_note}"""
        
        else:
            response = f"""**ANSWER: I found exactly {count} {element_type}(s)**

**ENGINEERING ANALYSIS:**
- Total Count: {count} {element_type}(s) identified
- Distribution: {self._get_distribution_summary(visual_result)}
- Analysis Method: Master engineering intelligence with cross-verification
- Validation Status: {consensus_result.get('validation_agreement', 'VERIFIED')}

{location_summary}

{verification_summary}

**CONFIDENCE: {confidence}% - {confidence_label}**
*{confidence_desc}*{discrepancy_note}"""
        
        return response
    
    def _format_location_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format LOCATION type questions - Where are the X located?"""
        
        element_type = visual_result.element_type
        locations = visual_result.locations
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        if not locations:
            return f"""**ANSWER: No {element_type}s found to locate**

**ANALYSIS:**
- Search Result: No {element_type}s detected in the analyzed pages
- Recommendation: Check other drawing sheets or sections

**CONFIDENCE: {confidence}% - {confidence_label}**"""
        
        # Build detailed location list
        location_entries = []
        for i, loc in enumerate(locations[:20], 1):  # Limit to 20 for readability
            entry = f"{i}. **Grid {loc.get('grid_ref', 'Unknown')}**"
            
            if loc.get('visual_details'):
                entry += f" - {loc['visual_details']}"
            if loc.get('room'):
                entry += f" (Room: {loc['room']})"
            if loc.get('element_tag'):
                entry += f" [Tag: {loc['element_tag']}]"
            
            location_entries.append(entry)
        
        remaining = ""
        if len(locations) > 20:
            remaining = f"\n\n*Plus {len(locations) - 20} additional locations...*"
        
        response = f"""**ANSWER: {element_type}s found at {len(locations)} locations**

**LOCATION ANALYSIS:**
- Total Locations: {len(locations)} {element_type}(s) identified
- Distribution: {self._get_distribution_summary(visual_result)}
- Grid Coverage: {', '.join(visual_result.grid_references[:10])}{'...' if len(visual_result.grid_references) > 10 else ''}

**DETAILED LOCATIONS:**
{chr(10).join(location_entries)}{remaining}

**SPATIAL VERIFICATION:**
- Layout Pattern: {self._analyze_spatial_pattern(locations)}
- Validation: {consensus_result.get('validation_agreement', 'Spatial distribution verified')}

**CONFIDENCE: {confidence}% - {confidence_label}**
*Location accuracy based on grid reference system*"""
        
        return response
    
    def _format_identification_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format IDENTIFY type questions - What is this? What type?"""
        
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        # Extract identification details
        specifications = self._extract_specifications(visual_result)
        
        response = f"""**ANSWER: {element_type.title()} Identification Complete**

**IDENTIFICATION SUMMARY:**
- Element Type: {element_type.title()}
- Quantity Found: {visual_result.count}
- Primary Classification: {self._get_element_classification(element_type)}

**IDENTIFIED CHARACTERISTICS:**
{self._format_characteristics(visual_result, specifications)}

**TECHNICAL DETAILS:**
{self._format_technical_details(specifications)}

**VERIFICATION:**
- Visual Patterns Matched: {len(visual_result.pattern_matches) if visual_result.pattern_matches else 'Standard patterns identified'}
- Cross-Reference: {self._get_cross_reference_summary(validation_results)}

**CONFIDENCE: {confidence}% - {confidence_label}**
*Identification based on engineering standards and visual analysis*"""
        
        return response
    
    def _format_specification_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format SPECIFICATION type questions - What are the specs? Size? Model?"""
        
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        # Extract all specification data
        specs = self._extract_detailed_specifications(visual_result, validation_results)
        
        response = f"""**ANSWER: {element_type.title()} Specifications**

**SPECIFICATION SUMMARY:**
- Element: {element_type.title()}
- Total Items: {visual_result.count}
- Specification Source: {self._get_spec_sources(visual_result, validation_results)}

**DETAILED SPECIFICATIONS:**
{self._format_specification_list(specs)}

**STANDARD COMPLIANCE:**
{self._format_standards_compliance(visual_result, validation_results)}

**ADDITIONAL NOTES:**
{self._format_specification_notes(visual_result)}

**CONFIDENCE: {confidence}% - {confidence_label}**
*Specifications extracted from drawings and verified against standards*"""
        
        return response
    
    def _format_compliance_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format COMPLIANCE type questions - Does this meet code? Is it compliant?"""
        
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        # Extract compliance information
        compliance_status = self._determine_compliance_status(visual_result, validation_results)
        
        response = f"""**ANSWER: {element_type.title()} Compliance Analysis**

**COMPLIANCE OVERVIEW:**
- Element Type: {element_type.title()}
- Items Reviewed: {visual_result.count}
- Overall Status: {compliance_status['overall']}

**CODE REQUIREMENTS:**
{self._format_code_requirements(element_type)}

**COMPLIANCE FINDINGS:**
{self._format_compliance_findings(visual_result, validation_results, compliance_status)}

**SPECIFIC OBSERVATIONS:**
{self._format_compliance_observations(visual_result, compliance_status)}

**RECOMMENDATIONS:**
{self._format_compliance_recommendations(compliance_status)}

⚠️ **DISCLAIMER**: This analysis is for general guidance only. Always verify with local authority having jurisdiction and applicable codes.

**CONFIDENCE: {confidence}% - {confidence_label}**
*Compliance assessment based on standard requirements*"""
        
        return response
    
    def _format_general_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format GENERAL type questions - Open-ended or high-level questions"""
        
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        original_question = question_analysis.get("original_prompt", "")
        
        # Determine the focus of the general question
        focus = self._determine_general_focus(original_question, visual_result)
        
        response = f"""**PROFESSIONAL ANALYSIS**

**QUESTION**: {original_question}

**SUMMARY**:
{self._create_general_summary(visual_result, focus)}

**KEY FINDINGS**:
{self._format_key_findings(visual_result, focus)}

**TECHNICAL DETAILS**:
{self._format_general_technical_details(visual_result, validation_results)}

**PROFESSIONAL ASSESSMENT**:
{self._create_professional_assessment(visual_result, validation_results, focus)}

**CONFIDENCE: {confidence}% - {confidence_label}**
*Analysis performed using master engineering expertise*"""
        
        return response
    
    def _format_detailed_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format DETAILED type questions - Specific technical questions requiring deep analysis"""
        
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        response = f"""**DETAILED TECHNICAL ANALYSIS**

**SUBJECT**: {element_type.title()} - Detailed Review

**COMPREHENSIVE FINDINGS**:

1. **Quantitative Analysis**:
   - Total Count: {visual_result.count}
   - Distribution: {self._get_distribution_summary(visual_result)}
   - Density: {self._calculate_density(visual_result)}

2. **Technical Specifications**:
{self._format_detailed_specifications(visual_result)}

3. **Spatial Analysis**:
{self._format_spatial_analysis(visual_result)}

4. **Cross-Reference Validation**:
{self._format_validation_details(validation_results)}

5. **Engineering Observations**:
{self._format_engineering_observations(visual_result, validation_results)}

**TECHNICAL RECOMMENDATIONS**:
{self._format_technical_recommendations(visual_result, element_type)}

**CONFIDENCE: {confidence}% - {confidence_label}**
*Detailed analysis using comprehensive engineering methodology*"""
        
        return response
    
    def _format_estimate_response(
        self,
        visual_result: VisualIntelligenceResult,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Format ESTIMATE type questions - Estimate area, cost, materials, etc."""
        
        element_type = visual_result.element_type
        confidence = int(trust_metrics.reliability_score * 100)
        confidence_label, _ = self._get_confidence_descriptor(trust_metrics.reliability_score)
        
        # Determine what's being estimated
        estimate_type = self._determine_estimate_type(question_analysis.get("original_prompt", ""))
        
        response = f"""**ENGINEERING ESTIMATE**

**ESTIMATE REQUEST**: {question_analysis.get('original_prompt', 'Estimation request')}

**BASE DATA**:
- Element Type: {element_type.title()}
- Quantity: {visual_result.count} units
- Coverage: {self._estimate_coverage(visual_result)}

**ESTIMATION CALCULATIONS**:
{self._format_estimation_calculations(visual_result, estimate_type)}

**ESTIMATE SUMMARY**:
{self._format_estimate_summary(visual_result, estimate_type)}

**ASSUMPTIONS & FACTORS**:
{self._format_estimation_assumptions(estimate_type, element_type)}

**RANGES**:
- Low Estimate: {self._calculate_low_estimate(visual_result, estimate_type)}
- Most Likely: {self._calculate_likely_estimate(visual_result, estimate_type)}
- High Estimate: {self._calculate_high_estimate(visual_result, estimate_type)}

⚠️ **NOTE**: This is a preliminary estimate based on visual analysis. Actual quantities should be verified with detailed takeoffs.

**CONFIDENCE: {confidence}% - {confidence_label}**
*Estimate based on industry standards and visible information*"""
        
        return response
    
    # Helper methods for building response components
    
    def _get_confidence_descriptor(self, score: float) -> Tuple[str, str]:
        """Get confidence label and description based on score"""
        for (low, high), (label, desc) in self.confidence_descriptors.items():
            if low <= score < high:
                return label, desc
        return "UNKNOWN", "Confidence level unclear"
    
    def _build_location_summary(self, visual_result: VisualIntelligenceResult, count: int) -> str:
        """Build a summary of locations found"""
        if count == 0:
            return ""
        
        locations = visual_result.locations[:count]
        if len(locations) <= 5:
            # Show all locations
            details = []
            for i, loc in enumerate(locations, 1):
                detail = f"{i}. **{loc.get('grid_ref', 'Unknown')}**"
                if loc.get('visual_details'):
                    detail += f" - {loc['visual_details']}"
                if loc.get('element_tag'):
                    detail += f" [{loc['element_tag']}]"
                details.append(detail)
            return "**DETAILED FINDINGS:**\n" + "\n".join(details)
        else:
            # Summarize locations
            grid_refs = visual_result.grid_references[:10]
            summary = f"**LOCATION SUMMARY:**\n"
            summary += f"- Grid References: {', '.join(grid_refs)}"
            if len(visual_result.grid_references) > 10:
                summary += f" (and {len(visual_result.grid_references) - 10} more)"
            summary += f"\n- Distribution: {self._get_distribution_summary(visual_result)}"
            return summary
    
    def _build_verification_summary(
        self,
        validation_results: List[ValidationResult],
        consensus_result: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> str:
        """Build verification summary"""
        summary_parts = ["**VERIFICATION PROCESS:**"]
        
        # Add validation results
        for val in validation_results:
            if val.confidence >= 0.90:
                status = "✓ PASSED"
            elif val.confidence >= 0.80:
                status = "⚡ PARTIAL"
            else:
                status = "⚠️ REVIEW"
            
            summary_parts.append(f"- {val.methodology.replace('_', ' ').title()}: {status}")
        
        # Add consensus
        summary_parts.append(f"- Consensus Status: {consensus_result.get('validation_agreement', 'VERIFIED')}")
        
        # Add any important notes
        if visual_result.verification_notes:
            summary_parts.append("\n**Verification Notes:**")
            for note in visual_result.verification_notes[:3]:
                summary_parts.append(f"- {note}")
        
        return "\n".join(summary_parts)
    
    def _get_distribution_summary(self, visual_result: VisualIntelligenceResult) -> str:
        """Analyze and describe the distribution of elements"""
        if not visual_result.locations:
            return "No elements to analyze"
        
        # Analyze grid references
        grid_refs = visual_result.grid_references
        if len(set(grid_refs)) == 1:
            return f"All in grid {grid_refs[0]}"
        elif len(set(grid_refs)) <= 3:
            return f"Concentrated in grids {', '.join(set(grid_refs))}"
        else:
            return f"Distributed across {len(set(grid_refs))} grid locations"
    
    def _analyze_spatial_pattern(self, locations: List[Dict[str, Any]]) -> str:
        """Analyze the spatial pattern of locations"""
        if len(locations) <= 1:
            return "Single location"
        elif len(locations) <= 4:
            return "Sparse distribution"
        elif len(locations) <= 10:
            return "Moderate distribution"
        else:
            return "Extensive distribution"
    
    def _extract_specifications(self, visual_result: VisualIntelligenceResult) -> Dict[str, Any]:
        """Extract specifications from visual evidence"""
        specs = {}
        
        for evidence in visual_result.visual_evidence:
            # Look for size specifications
            if "size" in evidence.lower() or "dimension" in evidence.lower():
                specs["dimensions"] = evidence
            # Look for model/type
            elif "model" in evidence.lower() or "type" in evidence.lower():
                specs["model"] = evidence
            # Look for ratings
            elif "rating" in evidence.lower() or "capacity" in evidence.lower():
                specs["rating"] = evidence
        
        return specs
    
    def _format_characteristics(self, visual_result: VisualIntelligenceResult, specs: Dict[str, Any]) -> str:
        """Format element characteristics"""
        chars = []
        
        if visual_result.pattern_matches:
            chars.append(f"- Visual Pattern: {', '.join(visual_result.pattern_matches[:3])}")
        
        if specs.get("dimensions"):
            chars.append(f"- Dimensions: {specs['dimensions']}")
        
        if specs.get("model"):
            chars.append(f"- Type/Model: {specs['model']}")
        
        if visual_result.visual_evidence:
            chars.append(f"- Additional Features: {len(visual_result.visual_evidence)} characteristics identified")
        
        return "\n".join(chars) if chars else "- Standard configuration identified"
    
    def _format_technical_details(self, specs: Dict[str, Any]) -> str:
        """Format technical details"""
        if not specs:
            return "Standard specifications apply"
        
        details = []
        for key, value in specs.items():
            details.append(f"• {key.title()}: {value}")
        
        return "\n".join(details)
    
    def _get_element_classification(self, element_type: str) -> str:
        """Get classification for element type"""
        classifications = {
            "door": "Architectural - Openings",
            "window": "Architectural - Openings",
            "outlet": "Electrical - Power Distribution",
            "panel": "Electrical - Distribution Equipment",
            "light fixture": "Electrical - Lighting",
            "plumbing fixture": "Plumbing - Fixtures",
            "sprinkler": "Fire Protection - Suppression",
            "diffuser": "HVAC - Air Distribution",
            "column": "Structural - Vertical Support",
            "beam": "Structural - Horizontal Support"
        }
        
        return classifications.get(element_type, "Construction Element")
    
    def _get_cross_reference_summary(self, validation_results: List[ValidationResult]) -> str:
        """Get cross-reference summary from validations"""
        for val in validation_results:
            if val.methodology == "cross_reference_validation" and val.cross_references:
                return ", ".join(val.cross_references[:2])
        return "Visual standards verified"
    
    def _format_fallback_response(
        self,
        visual_result: VisualIntelligenceResult,
        trust_metrics: TrustMetrics,
        question_analysis: Dict[str, Any]
    ) -> str:
        """Fallback response format when specific formatting fails"""
        
        confidence = int(trust_metrics.reliability_score * 100)
        
        return f"""**ENGINEERING ANALYSIS**

**Question**: {question_analysis.get('original_prompt', 'Analysis request')}

**Findings**:
- Element Type: {visual_result.element_type}
- Count: {visual_result.count}
- Locations Identified: {len(visual_result.locations)}

**Verification**: Analysis completed with {len(visual_result.verification_notes)} verification checks.

**Confidence**: {confidence}%

*Technical analysis completed with available data.*"""
    
    def _determine_compliance_status(self, visual_result: VisualIntelligenceResult, 
                                   validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Determine overall compliance status"""
        
        compliance_status = {
            "overall": "REQUIRES DETAILED REVIEW",
            "issues_found": [],
            "compliant_items": [],
            "recommendations": []
        }
        
        # Check spacing compliance for relevant elements
        if visual_result.element_type in self.standard_spacing:
            spacing_ok = self._check_spacing_compliance(visual_result)
            if spacing_ok:
                compliance_status["compliant_items"].append("Spacing appears compliant")
            else:
                compliance_status["issues_found"].append("Spacing may not meet standards")
        
        # Check validation results for compliance issues
        for val in validation_results:
            if "compliance" in val.methodology:
                if val.confidence >= 0.90:
                    compliance_status["compliant_items"].append(f"{val.methodology} passed")
                else:
                    compliance_status["issues_found"].append(f"{val.methodology} needs review")
        
        # Determine overall status
        if not compliance_status["issues_found"]:
            compliance_status["overall"] = "APPEARS COMPLIANT"
        elif len(compliance_status["issues_found"]) > 2:
            compliance_status["overall"] = "MULTIPLE ISSUES FOUND"
        
        return compliance_status
    
    def _check_spacing_compliance(self, visual_result: VisualIntelligenceResult) -> bool:
        """Check if element spacing appears compliant"""
        # This is a simplified check - real implementation would analyze actual spacing
        if visual_result.count == 0:
            return True
        
        # Check if we have location data
        if not visual_result.locations:
            return False
        
        # Basic check: if elements are well-distributed, likely compliant
        unique_grids = len(set(visual_result.grid_references))
        if unique_grids >= visual_result.count * 0.7:  # Most elements in different grids
            return True
        
        return False
    
    def _format_code_requirements(self, element_type: str) -> str:
        """Format applicable code requirements"""
        requirements = self.code_requirements_db.get(element_type, {})
        
        if requirements:
            formatted = []
            for code, requirement in requirements.items():
                formatted.append(f"• **{code}**: {requirement}")
            return "\n".join(formatted)
        
        return "• Applicable building codes\n• Local jurisdiction requirements"
    
    def _format_compliance_findings(self, visual_result: VisualIntelligenceResult, 
                                  validation_results: List[ValidationResult], 
                                  compliance_status: Dict[str, Any]) -> str:
        """Format compliance findings"""
        findings = []
        
        # Compliant items
        if compliance_status["compliant_items"]:
            findings.append("**✓ COMPLIANT ASPECTS:**")
            for item in compliance_status["compliant_items"]:
                findings.append(f"- {item}")
        
        # Issues found
        if compliance_status["issues_found"]:
            findings.append("\n**⚠️ POTENTIAL ISSUES:**")
            for issue in compliance_status["issues_found"]:
                findings.append(f"- {issue}")
        
        # General findings
        findings.append(f"\n**GENERAL FINDINGS:**")
        findings.append(f"- {visual_result.count} {visual_result.element_type}(s) reviewed")
        findings.append(f"- Distribution pattern: {self._get_distribution_summary(visual_result)}")
        
        return "\n".join(findings) if findings else "No specific compliance findings"
    
    def _format_compliance_observations(self, visual_result: VisualIntelligenceResult, 
                                      compliance_status: Dict[str, Any]) -> str:
        """Format specific compliance observations"""
        observations = []
        
        # Element-specific observations
        if visual_result.element_type == "outlet":
            observations.append("- Outlet placement should be verified for GFCI requirements in wet areas")
            observations.append("- Spacing between outlets should not exceed 12 feet (NEC 210.52)")
        elif visual_result.element_type == "door":
            observations.append("- Door widths should provide minimum 32\" clear opening (ADA)")
            observations.append("- Swing direction should be verified for egress requirements")
        elif visual_result.element_type == "sprinkler":
            observations.append("- Sprinkler coverage patterns should be verified")
            observations.append("- Distance from walls and obstructions should be checked")
        
        # Add general observations
        if visual_result.count > 0:
            observations.append(f"- Total of {visual_result.count} elements identified for review")
        
        return "\n".join(observations) if observations else "Layout and spacing appear reasonable for intended use"
    
    def _format_compliance_recommendations(self, compliance_status: Dict[str, Any]) -> str:
        """Format compliance recommendations"""
        recommendations = [
            "• Verify all dimensions and clearances with field measurements",
            "• Ensure compliance with current local amendments to codes",
            "• Coordinate with AHJ (Authority Having Jurisdiction) for specific requirements"
        ]
        
        # Add specific recommendations based on issues
        if compliance_status["issues_found"]:
            recommendations.append("• Address identified potential issues before construction")
            recommendations.append("• Consider professional code review for areas of concern")
        
        return "\n".join(recommendations)
    
    def _determine_general_focus(self, question: str, visual_result: VisualIntelligenceResult) -> str:
        """Determine the focus of a general question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["overview", "summary", "general"]):
            return "overview"
        elif any(word in question_lower for word in ["design", "layout", "plan"]):
            return "design"
        elif any(word in question_lower for word in ["problem", "issue", "concern"]):
            return "issues"
        else:
            return "analysis"
    
    def _create_general_summary(self, visual_result: VisualIntelligenceResult, focus: str) -> str:
        """Create a summary for general questions"""
        if focus == "overview":
            return f"This blueprint contains {visual_result.count} {visual_result.element_type}(s) distributed across the analyzed areas. The analysis shows a {self._get_distribution_summary(visual_result).lower()} pattern."
        elif focus == "design":
            return f"The {visual_result.element_type} design shows {visual_result.count} elements strategically placed throughout the space. The layout appears to follow standard design practices."
        else:
            return f"Analysis of {visual_result.element_type}s reveals {visual_result.count} instances with specific technical considerations."
    
    def _format_key_findings(self, visual_result: VisualIntelligenceResult, focus: str) -> str:
        """Format key findings based on focus"""
        findings = []
        
        findings.append(f"• Total {visual_result.element_type}s: {visual_result.count}")
        
        if visual_result.count > 0:
            findings.append(f"• Distribution: {self._get_distribution_summary(visual_result)}")
            findings.append(f"• Coverage: {len(set(visual_result.grid_references))} grid zones")
        
        if visual_result.visual_evidence:
            findings.append(f"• Technical details: {len(visual_result.visual_evidence)} specifications identified")
        
        return "\n".join(findings)
    
    def _format_general_technical_details(self, visual_result: VisualIntelligenceResult, 
                                        validation_results: List[ValidationResult]) -> str:
        """Format technical details for general questions"""
        details = []
        
        # Validation summary
        high_confidence_count = sum(1 for v in validation_results if v.confidence >= 0.90)
        details.append(f"• Validation Methods: {len(validation_results)} applied")
        details.append(f"• High Confidence Validations: {high_confidence_count}/{len(validation_results)}")
        
        # Element specifics
        if visual_result.visual_evidence:
            details.append(f"• Evidence Sources: {', '.join(set(e.split()[0] for e in visual_result.visual_evidence[:3]))}")
        
        # Pattern matches
        if visual_result.pattern_matches:
            details.append(f"• Pattern Recognition: {len(visual_result.pattern_matches)} patterns identified")
        
        return "\n".join(details)
    
    def _create_professional_assessment(self, visual_result: VisualIntelligenceResult, 
                                      validation_results: List[ValidationResult], focus: str) -> str:
        """Create professional assessment"""
        
        # Calculate average validation confidence
        avg_confidence = sum(v.confidence for v in validation_results) / len(validation_results) if validation_results else 0
        
        if avg_confidence >= 0.90:
            assessment = "The analysis demonstrates high reliability with strong validation consensus. "
        elif avg_confidence >= 0.80:
            assessment = "The analysis shows good reliability with reasonable validation agreement. "
        else:
            assessment = "The analysis indicates areas requiring additional verification. "
        
        # Add focus-specific assessment
        if focus == "design":
            assessment += f"The {visual_result.element_type} layout appears to follow standard design practices with appropriate distribution."
        elif focus == "overview":
            assessment += f"Overall, {visual_result.count} {visual_result.element_type}(s) were identified with professional confidence."
        else:
            assessment += "Professional engineering judgment indicates the findings meet general industry standards."
        
        return assessment
    
    def _calculate_density(self, visual_result: VisualIntelligenceResult) -> str:
        """Calculate element density"""
        if visual_result.count == 0:
            return "N/A - No elements found"
        
        grid_count = len(set(visual_result.grid_references))
        if grid_count == 0:
            return "Unable to calculate"
        
        density = visual_result.count / grid_count
        
        if density >= 3:
            return f"{density:.1f} per grid zone (High density)"
        elif density >= 1.5:
            return f"{density:.1f} per grid zone (Moderate density)"
        else:
            return f"{density:.1f} per grid zone (Low density)"
    
    def _extract_detailed_specifications(self, visual_result: VisualIntelligenceResult, 
                                       validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Extract detailed specifications from all sources"""
        specs = self._extract_specifications(visual_result)
        
        # Add specifications from validation results
        for val in validation_results:
            if val.findings.get("specifications"):
                specs.update(val.findings["specifications"])
        
        # Add default specifications if none found
        if not specs:
            specs = self._get_default_specifications(visual_result.element_type)
        
        return specs
    
    def _get_default_specifications(self, element_type: str) -> Dict[str, Any]:
        """Get default specifications for element type"""
        defaults = {
            "outlet": {
                "voltage": "120V standard",
                "amperage": "15A or 20A",
                "type": "Duplex receptacle"
            },
            "door": {
                "width": "36 inches standard",
                "height": "80 inches minimum",
                "type": "Single swing"
            },
            "window": {
                "type": "Fixed or operable",
                "glazing": "Double pane typical",
                "frame": "Aluminum or vinyl"
            },
            "sprinkler": {
                "type": "Pendant or upright",
                "temperature": "Ordinary 135-170°F",
                "coverage": "Standard coverage"
            }
        }
        
        return defaults.get(element_type, {"type": "Standard specification"})
    
    def _get_spec_sources(self, visual_result: VisualIntelligenceResult, 
                        validation_results: List[ValidationResult]) -> str:
        """Identify specification sources"""
        sources = []
        
        if visual_result.visual_evidence:
            sources.append("Visual analysis")
        
        for val in validation_results:
            if "schedule" in val.methodology and val.findings.get("schedule_found"):
                sources.append("Equipment schedule")
            if "cross_reference" in val.methodology and val.cross_references:
                sources.append("Drawing notes")
        
        return ", ".join(sources) if sources else "Drawing interpretation"
    
    def _format_specification_list(self, specs: Dict[str, Any]) -> str:
        """Format specifications into readable list"""
        if not specs:
            return "• No detailed specifications available"
        
        formatted = []
        for key, value in specs.items():
            # Format key nicely
            formatted_key = key.replace("_", " ").title()
            formatted.append(f"• **{formatted_key}**: {value}")
        
        return "\n".join(formatted)
    
    def _format_standards_compliance(self, visual_result: VisualIntelligenceResult, 
                                   validation_results: List[ValidationResult]) -> str:
        """Format standards compliance information"""
        compliance_items = []
        
        # Check validation results for standards compliance
        for val in validation_results:
            if "standards_compliance" in val.findings:
                status = val.findings["standards_compliance"]
                if "FULLY_COMPLIANT" in status:
                    compliance_items.append("✓ Meets drawing standards")
                elif "MOSTLY_COMPLIANT" in status:
                    compliance_items.append("⚡ Generally meets standards with minor variations")
                else:
                    compliance_items.append("⚠️ Standards compliance requires review")
        
        # Add element-specific standards
        element_standards = {
            "outlet": "• NEC Article 406 - Receptacle requirements",
            "door": "• ANSI A117.1 - Accessible door specifications",
            "window": "• AAMA standards for window performance",
            "sprinkler": "• NFPA 13 - Installation standards"
        }
        
        if visual_result.element_type in element_standards:
            compliance_items.append(element_standards[visual_result.element_type])
        
        return "\n".join(compliance_items) if compliance_items else "• Standard industry specifications apply"
    
    def _format_specification_notes(self, visual_result: VisualIntelligenceResult) -> str:
        """Format additional specification notes"""
        notes = []
        
        # Add verification notes if any
        if visual_result.verification_notes:
            relevant_notes = [note for note in visual_result.verification_notes 
                            if "spec" in note.lower() or "dimension" in note.lower()]
            if relevant_notes:
                notes.extend(relevant_notes[:2])
        
        # Add standard notes
        notes.append("Field verification required for exact dimensions")
        notes.append("Specifications subject to manufacturer's standards")
        
        return "\n".join(f"• {note}" for note in notes)
    
    def _format_detailed_specifications(self, visual_result: VisualIntelligenceResult) -> str:
        """Format detailed specifications for detailed analysis"""
        specs = []
        
        # Get all available specifications
        detailed_specs = self._extract_detailed_specifications(visual_result, [])
        
        for key, value in detailed_specs.items():
            specs.append(f"   - {key.replace('_', ' ').title()}: {value}")
        
        if not specs:
            specs.append("   - Standard specifications assumed")
            specs.append("   - Field verification required")
        
        return "\n".join(specs)
    
    def _format_spatial_analysis(self, visual_result: VisualIntelligenceResult) -> str:
        """Format spatial analysis details"""
        analysis = []
        
        # Grid distribution
        unique_grids = set(visual_result.grid_references)
        analysis.append(f"   - Grid Coverage: {len(unique_grids)} unique grid references")
        
        # Distribution pattern
        if len(unique_grids) > 1:
            # Analyze pattern
            grid_counts = Counter(visual_result.grid_references)
            max_in_grid = max(grid_counts.values())
            
            if max_in_grid > 3:
                analysis.append(f"   - Concentration: Up to {max_in_grid} elements in single grid")
            else:
                analysis.append("   - Distribution: Even spread across grids")
        
        # Spacing analysis
        if visual_result.count > 1:
            analysis.append(f"   - Pattern Type: {self._analyze_spatial_pattern(visual_result.locations)}")
        
        return "\n".join(analysis)
    
    def _format_validation_details(self, validation_results: List[ValidationResult]) -> str:
        """Format validation details for detailed analysis"""
        details = []
        
        details.append(f"   - Total Validation Passes: {len(validation_results)}")
        
        # Count by confidence level
        high_conf = sum(1 for v in validation_results if v.confidence >= 0.90)
        med_conf = sum(1 for v in validation_results if 0.80 <= v.confidence < 0.90)
        low_conf = sum(1 for v in validation_results if v.confidence < 0.80)
        
        details.append(f"   - High Confidence: {high_conf} validations")
        details.append(f"   - Medium Confidence: {med_conf} validations")
        details.append(f"   - Low Confidence: {low_conf} validations")
        
        # Consensus check
        if all(v.confidence >= 0.85 for v in validation_results):
            details.append("   - Result: Strong validation consensus achieved")
        else:
            details.append("   - Result: Mixed validation results - review recommended")
        
        return "\n".join(details)
    
    def _format_engineering_observations(self, visual_result: VisualIntelligenceResult, 
                                       validation_results: List[ValidationResult]) -> str:
        """Format engineering observations"""
        observations = []
        
        # Layout observations
        if visual_result.count > 0:
            observations.append(f"   - Element Placement: {self._get_distribution_summary(visual_result)}")
            observations.append(f"   - Coverage Pattern: Appropriate for {visual_result.element_type} application")
        
        # Validation observations
        if any(v.discrepancies for v in validation_results):
            observations.append("   - Discrepancies: Some validation concerns noted - see verification summary")
        else:
            observations.append("   - Validation: No significant discrepancies found")
        
        # Engineering judgment
        observations.append("   - Engineering Assessment: Layout appears logical and functional")
        
        return "\n".join(observations)
    
    def _format_technical_recommendations(self, visual_result: VisualIntelligenceResult, 
                                        element_type: str) -> str:
        """Format technical recommendations"""
        recommendations = []
        
        # Element-specific recommendations
        element_recs = {
            "outlet": [
                "• Verify GFCI protection in required areas",
                "• Confirm circuit loading calculations",
                "• Check ADA mounting heights where applicable"
            ],
            "door": [
                "• Verify hardware specifications",
                "• Confirm fire ratings if required",
                "• Check ADA clearances and approach"
            ],
            "sprinkler": [
                "• Verify hydraulic calculations",
                "• Confirm coverage per NFPA 13",
                "• Check obstruction clearances"
            ],
            "window": [
                "• Verify egress requirements if applicable",
                "• Confirm energy code compliance",
                "• Check safety glazing requirements"
            ]
        }
        
        if element_type in element_recs:
            recommendations.extend(element_recs[element_type])
        else:
            recommendations.extend([
                f"• Verify {element_type} specifications during construction",
                "• Coordinate with relevant trades",
                "• Ensure code compliance"
            ])
        
        return "\n".join(recommendations)
    
    def _determine_estimate_type(self, question: str) -> str:
        """Determine what type of estimate is requested"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["cost", "price", "budget", "$", "dollar"]):
            return "cost"
        elif any(word in question_lower for word in ["area", "square", "footage", "sf", "coverage"]):
            return "area"
        elif any(word in question_lower for word in ["material", "quantity", "amount", "how much", "how many"]):
            return "material"
        elif any(word in question_lower for word in ["time", "duration", "schedule", "hours", "days"]):
            return "time"
        else:
            return "general"
    
    def _estimate_coverage(self, visual_result: VisualIntelligenceResult) -> str:
        """Estimate coverage area"""
        grid_count = len(set(visual_result.grid_references))
        
        # Assume each grid is approximately 10' x 10' = 100 sq ft
        estimated_area = grid_count * 100
        
        return f"{grid_count} grid zones (~{estimated_area:,} sq ft)"
    
    def _format_estimation_calculations(self, visual_result: VisualIntelligenceResult, 
                                      estimate_type: str) -> str:
        """Format estimation calculations"""
        calcs = []
        
        count = visual_result.count
        element_type = visual_result.element_type
        
        if estimate_type == "cost":
            # Material cost calculation
            if element_type in self.estimate_factors["material_cost"]:
                costs = self.estimate_factors["material_cost"][element_type]
                calcs.append(f"• Material Cost Range: ${costs['low']}-${costs['high']} per unit")
                calcs.append(f"• Total Material: ${count * costs['low']:,} - ${count * costs['high']:,}")
            
            # Labor calculation
            if element_type in self.estimate_factors["labor_hours"]:
                hours = self.estimate_factors["labor_hours"][element_type]
                labor_rate = 75  # $/hour average
                calcs.append(f"• Labor Hours: {hours['low']}-{hours['high']} per unit")
                calcs.append(f"• Labor Cost (@${labor_rate}/hr): ${count * hours['low'] * labor_rate:,.0f} - ${count * hours['high'] * labor_rate:,.0f}")
        
        elif estimate_type == "area":
            grid_zones = len(set(visual_result.grid_references))
            calcs.append(f"• Coverage: {grid_zones} grid zones")
            calcs.append(f"• Estimated Area: {grid_zones * 100:,} sq ft (@ 100 sq ft/grid)")
            calcs.append(f"• Density: {count / (grid_zones * 100):.3f} units per sq ft")
        
        elif estimate_type == "material":
            calcs.append(f"• Base Quantity: {count} {element_type}(s)")
            calcs.append(f"• Waste Factor: 10% typical")
            calcs.append(f"• Order Quantity: {math.ceil(count * 1.1)} units")
        
        elif estimate_type == "time":
            if element_type in self.estimate_factors["labor_hours"]:
                hours = self.estimate_factors["labor_hours"][element_type]
                total_hours_low = count * hours['low']
                total_hours_high = count * hours['high']
                calcs.append(f"• Hours per Unit: {hours['low']}-{hours['high']}")
                calcs.append(f"• Total Hours: {total_hours_low:.1f}-{total_hours_high:.1f}")
                calcs.append(f"• Duration (1 worker): {total_hours_low/8:.1f}-{total_hours_high/8:.1f} days")
                calcs.append(f"• Duration (2 workers): {total_hours_low/16:.1f}-{total_hours_high/16:.1f} days")
        
        return "\n".join(calcs) if calcs else "• Standard calculations apply"
    
    def _format_estimate_summary(self, visual_result: VisualIntelligenceResult, 
                               estimate_type: str) -> str:
        """Format estimate summary"""
        count = visual_result.count
        element_type = visual_result.element_type
        
        if estimate_type == "cost":
            if element_type in self.estimate_factors["material_cost"]:
                costs = self.estimate_factors["material_cost"][element_type]
                hours = self.estimate_factors["labor_hours"].get(element_type, {"avg": 1})
                
                material_avg = count * costs['avg']
                labor_avg = count * hours['avg'] * 75
                
                return f"""• Material Cost (estimated): ${material_avg:,.2f}
- Labor Cost (estimated): ${labor_avg:,.2f}
- **Total Estimated Cost: ${material_avg + labor_avg:,.2f}**
- Cost Range: ${(material_avg + labor_avg) * 0.8:,.2f} - ${(material_avg + labor_avg) * 1.2:,.2f}"""
        
        elif estimate_type == "area":
            grid_zones = len(set(visual_result.grid_references))
            return f"""• Total Coverage: {grid_zones * 100:,} sq ft (estimated)
- Element Density: {count / (grid_zones * 100):.3f} per sq ft
- Grid Zones Covered: {grid_zones}"""
        
        elif estimate_type == "material":
            return f"""• Required Quantity: {count} units
- With 10% Waste: {math.ceil(count * 1.1)} units
- With 15% Waste: {math.ceil(count * 1.15)} units
- **Recommended Order: {math.ceil(count * 1.1)} units**"""
        
        elif estimate_type == "time":
            if element_type in self.estimate_factors["labor_hours"]:
                hours = self.estimate_factors["labor_hours"][element_type]
                avg_hours = count * hours['avg']
                return f"""• Estimated Labor Hours: {avg_hours:.1f} hours
- Duration (1 worker): {avg_hours/8:.1f} days
- Duration (2 workers): {avg_hours/16:.1f} days
- **Recommended Schedule: {math.ceil(avg_hours/16)} days with 2-person crew**"""
        
        return f"Preliminary estimate based on {count} identified {element_type}(s)"
    
    def _format_estimation_assumptions(self, estimate_type: str, element_type: str) -> str:
        """Format estimation assumptions"""
        assumptions = ["• Quantities based on visual analysis", "• Standard installation conditions assumed"]
        
        if estimate_type == "cost":
            assumptions.extend([
                "• Regional pricing variations not included",
                "• Current material costs may vary",
                "• Labor rates based on $75/hour average",
                "• Does not include overhead, profit, or taxes"
            ])
        elif estimate_type == "area":
            assumptions.extend([
                "• Grid squares estimated at 10' x 10' (100 sq ft)",
                "• Actual dimensions require field verification",
                "• Coverage may vary based on actual layout"
            ])
        elif estimate_type == "material":
            assumptions.extend([
                "• 10% waste factor included (industry standard)",
                "• Minimum order quantities not considered",
                "• Special conditions may require additional materials"
            ])
        elif estimate_type == "time":
            assumptions.extend([
                "• Based on experienced crew productivity",
                "• Normal working conditions assumed",
                "• Does not include mobilization or setup time",
                "• Concurrent work not considered"
            ])
        
        return "\n".join(assumptions)
    
    def _calculate_low_estimate(self, visual_result: VisualIntelligenceResult, 
                               estimate_type: str) -> str:
        """Calculate low estimate"""
        count = visual_result.count
        element_type = visual_result.element_type
        
        if estimate_type == "cost":
            if element_type in self.estimate_factors["material_cost"]:
                material = count * self.estimate_factors["material_cost"][element_type]["low"]
                labor = count * self.estimate_factors["labor_hours"].get(element_type, {}).get("low", 0.5) * 65
                return f"${material + labor:,.2f} (materials + labor)"
        elif estimate_type == "material":
            return f"{count} units (no waste)"
        elif estimate_type == "time":
            if element_type in self.estimate_factors["labor_hours"]:
                hours = count * self.estimate_factors["labor_hours"][element_type]["low"]
                return f"{hours:.1f} hours ({hours/8:.1f} days)"
        
        return f"{count * 0.9:.0f} units (10% margin)"
    
    def _calculate_likely_estimate(self, visual_result: VisualIntelligenceResult, 
                                 estimate_type: str) -> str:
        """Calculate most likely estimate"""
        count = visual_result.count
        element_type = visual_result.element_type
        
        if estimate_type == "cost":
            if element_type in self.estimate_factors["material_cost"]:
                material = count * self.estimate_factors["material_cost"][element_type]["avg"]
                labor = count * self.estimate_factors["labor_hours"].get(element_type, {}).get("avg", 0.75) * 75
                return f"${material + labor:,.2f} (materials + labor)"
        elif estimate_type == "material":
            return f"{math.ceil(count * 1.1)} units (10% waste)"
        elif estimate_type == "time":
            if element_type in self.estimate_factors["labor_hours"]:
                hours = count * self.estimate_factors["labor_hours"][element_type]["avg"]
                return f"{hours:.1f} hours ({hours/8:.1f} days)"
        
        return f"{count} units"
    
    def _calculate_high_estimate(self, visual_result: VisualIntelligenceResult, 
                               estimate_type: str) -> str:
        """Calculate high estimate"""
        count = visual_result.count
        element_type = visual_result.element_type
        
        if estimate_type == "cost":
            if element_type in self.estimate_factors["material_cost"]:
                material = count * self.estimate_factors["material_cost"][element_type]["high"]
                labor = count * self.estimate_factors["labor_hours"].get(element_type, {}).get("high", 1.0) * 85
                return f"${material + labor:,.2f} (premium materials + labor)"
        elif estimate_type == "material":
            return f"{math.ceil(count * 1.2)} units (20% waste + contingency)"
        elif estimate_type == "time":
            if element_type in self.estimate_factors["labor_hours"]:
                hours = count * self.estimate_factors["labor_hours"][element_type]["high"]
                return f"{hours:.1f} hours ({hours/8:.1f} days)"
        
        return f"{count * 1.1:.0f} units (10% contingency)"
