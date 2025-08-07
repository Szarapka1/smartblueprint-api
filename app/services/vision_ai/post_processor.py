# post_processor.py
import asyncio
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG

# Fix: Import from app.models.schemas instead of .models
from app.models.schemas import VisualIntelligenceResult

# Add additional model imports that might be needed
from app.models.schemas import (
    ValidationResult,
    TrustMetrics,
    NoteSuggestion,
    QuestionType
)

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Simplified Post-processing that adds value beyond triple verification
    Focuses on:
    1. Code compliance insights
    2. Coordination checks between trades
    3. Professional recommendations
    
    Removed schedule detection - that's now in Pass 3 (text review)
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None  # Will be set by core
        
        # Code requirements by element type (simplified)
        self.code_requirements = {
            "outlet": {
                "spacing": "NEC 210.52 - Max 12ft apart along walls",
                "gfci": "NEC 210.8 - GFCI required in wet locations",
                "height": "15-48 inches above floor (NEC/ADA)"
            },
            "door": {
                "width": "IBC 1010.1 - Min 32\" clear width",
                "ada": "ADA 404 - 32\" min clear, 1/2\" max threshold",
                "egress": "NFPA 101 - Swing in direction of egress"
            },
            "sprinkler": {
                "spacing": "NFPA 13 - Max 15ft (light), 12ft (ordinary)",
                "clearance": "NFPA 13 - Min 4\" from walls",
                "coverage": "NFPA 13 - 130-200 sq ft per head"
            },
            "stair": {
                "width": "IBC 1011.5 - Min 44\" (occupant load >50)",
                "rise_run": "IBC 1011.5.2 - Max 7\" rise, min 11\" tread",
                "handrails": "IBC 1014 - Both sides if >44\" wide"
            },
            "parking": {
                "ada": "ADA 502 - 1 accessible per 25 spaces",
                "size": "ADA 502.2 - 8' wide + 5' aisle (standard)",
                "van": "ADA 502.2 - 11' wide + 5' aisle (van)"
            }
        }
    
    def set_vision_client(self, client):
        """Set vision client for API calls"""
        self.vision_client = client
    
    async def process(
        self,
        visual_result: VisualIntelligenceResult,
        images: List[Dict[str, Any]],
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simplified post-processing
        Only runs what adds value beyond triple verification
        """
        
        # Only process if it adds value for the question type
        if not self._should_process(question_analysis, visual_result):
            return {"summary": "", "processed": False}
        
        logger.info("🔧 Running value-add post-processing")
        
        results = {
            "compliance_notes": [],
            "coordination_issues": [],
            "recommendations": [],
            "summary": "",
            "processed": True
        }
        
        try:
            # Run only the valuable analyses
            element_type = visual_result.element_type
            count = visual_result.count
            
            # 1. Code compliance insights (always valuable)
            if element_type in self.code_requirements:
                compliance_notes = self._generate_compliance_insights(
                    element_type, count, visual_result
                )
                results["compliance_notes"] = compliance_notes
            
            # 2. Coordination checks (valuable for MEP elements)
            if self._needs_coordination(element_type):
                coordination_issues = self._check_coordination(
                    element_type, count, visual_result
                )
                results["coordination_issues"] = coordination_issues
            
            # 3. Professional recommendations
            recommendations = self._generate_recommendations(
                element_type, count, visual_result, question_analysis
            )
            results["recommendations"] = recommendations
            
            # Build summary only if we have valuable insights
            if any([results["compliance_notes"], results["coordination_issues"], 
                   results["recommendations"]]):
                results["summary"] = self._build_summary(results)
            
        except Exception as e:
            logger.error(f"Post-process error: {e}")
        
        return results
    
    def _should_process(
        self,
        question_analysis: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> bool:
        """Determine if post-processing adds value"""
        
        # Always process for compliance questions
        if question_analysis.get("type") == QuestionType.COMPLIANCE:
            return True
        
        # Process for detailed analysis
        if question_analysis.get("type") == QuestionType.DETAILED:
            return True
        
        # Process if we found elements that have code requirements
        if visual_result.count > 0 and visual_result.element_type in self.code_requirements:
            return True
        
        # Process for MEP coordination
        mep_elements = ["outlet", "panel", "sprinkler", "diffuser", "plumbing fixture"]
        if visual_result.element_type in mep_elements:
            return True
        
        return False
    
    def _generate_compliance_insights(
        self,
        element_type: str,
        count: int,
        visual_result: VisualIntelligenceResult
    ) -> List[str]:
        """Generate valuable compliance insights"""
        
        insights = []
        requirements = self.code_requirements.get(element_type, {})
        
        if not requirements or count == 0:
            return insights
        
        # Element-specific insights
        if element_type == "outlet":
            insights.append(f"📏 Code Requirement: {requirements['spacing']}")
            if any("gfci" in str(loc.get("visual_details", "")).lower() 
                  for loc in visual_result.locations):
                insights.append("✅ GFCI outlets identified in required areas")
            else:
                insights.append(f"⚡ Verify: {requirements['gfci']}")
        
        elif element_type == "door":
            insights.append(f"📏 Code Requirement: {requirements['width']}")
            insights.append(f"♿ ADA Requirement: {requirements['ada']}")
            if count > 10:  # Multiple doors suggest egress importance
                insights.append(f"🚪 Egress: {requirements['egress']}")
        
        elif element_type == "sprinkler":
            grid_count = len(set(visual_result.grid_references))
            if grid_count > 0 and count > 0:
                coverage_per_head = (grid_count * 100) / count  # Rough estimate
                insights.append(f"💧 Estimated coverage: ~{int(coverage_per_head)} sq ft per head")
                insights.append(f"📏 Code Requirement: {requirements['coverage']}")
        
        elif element_type == "parking" and count > 0:
            required_accessible = max(1, count // 25)
            insights.append(f"♿ ADA Requirement: Minimum {required_accessible} accessible space(s)")
            if count >= 100:
                van_required = max(1, required_accessible // 6)
                insights.append(f"🚐 Van Accessible: Minimum {van_required} van space(s)")
        
        return insights[:5]  # Limit to 5 most valuable insights
    
    def _needs_coordination(self, element_type: str) -> bool:
        """Check if element type needs coordination checks"""
        coordination_elements = [
            "sprinkler", "diffuser", "light fixture",  # Ceiling coordination
            "outlet", "switch", "panel",                # Wall coordination
            "column", "beam",                           # Structural coordination
            "pipe", "duct"                              # Service coordination
        ]
        return element_type in coordination_elements
    
    def _check_coordination(
        self,
        element_type: str,
        count: int,
        visual_result: VisualIntelligenceResult
    ) -> List[str]:
        """Check for coordination issues"""
        
        issues = []
        
        # Ceiling elements need coordination
        if element_type in ["sprinkler", "diffuser", "light fixture"] and count > 10:
            issues.append("📐 High density ceiling elements - coordinate layout carefully")
        
        # Structural coordination
        if element_type in ["column", "beam"]:
            issues.append("🏗️ Coordinate MEP routing with structural elements")
        
        # Panel coordination
        if element_type == "panel":
            issues.append("⚡ Ensure code-required clearances around panels")
        
        return issues[:3]  # Limit to 3 most important
    
    def _generate_recommendations(
        self,
        element_type: str,
        count: int,
        visual_result: VisualIntelligenceResult,
        question_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate professional recommendations"""
        
        recommendations = []
        
        # Zero count recommendations
        if count == 0:
            recommendations.append(
                f"🔍 No {element_type}s found - verify if on different sheets"
            )
            return recommendations
        
        # High count recommendations
        if count > 50:
            recommendations.append(
                f"📊 {count} {element_type}s found - consider detailed quantity takeoff"
            )
        
        # Element-specific recommendations
        if element_type in ["outlet", "panel"] and count > 0:
            recommendations.append("⚡ Verify electrical load calculations")
        
        elif element_type == "sprinkler" and count > 0:
            recommendations.append("💧 Verify hydraulic calculations meet NFPA 13")
        
        elif element_type == "door" and count > 0:
            recommendations.append("🚪 Verify fire ratings and hardware schedules")
        
        # Compliance question recommendations
        if question_analysis.get("type") == QuestionType.COMPLIANCE:
            recommendations.append("📋 Perform detailed code review with current local amendments")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _build_summary(self, results: Dict[str, Any]) -> str:
        """Build a concise summary of post-processing findings"""
        
        summary_parts = []
        
        # Compliance insights
        if results["compliance_notes"]:
            summary_parts.append("**📋 CODE COMPLIANCE INSIGHTS:**")
            for note in results["compliance_notes"][:3]:
                summary_parts.append(note)
        
        # Coordination issues
        if results["coordination_issues"]:
            if summary_parts:
                summary_parts.append("")  # Blank line
            summary_parts.append("**🔧 COORDINATION NOTES:**")
            for issue in results["coordination_issues"]:
                summary_parts.append(issue)
        
        # Recommendations
        if results["recommendations"]:
            if summary_parts:
                summary_parts.append("")  # Blank line
            summary_parts.append("**💡 PROFESSIONAL RECOMMENDATIONS:**")
            for rec in results["recommendations"][:3]:
                summary_parts.append(rec)
        
        return "\n".join(summary_parts) if summary_parts else ""