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
    Post-processing analysis that runs after main visual intelligence
    Handles schedule detection, code compliance, and additional insights
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.vision_client = None  # Will be set by core
        
        # Code requirements by element type
        self.code_requirements = {
            "outlet": {
                "NEC 210.52": "Outlets required every 12 feet along walls",
                "NEC 210.52(C)": "Kitchen counters: outlets every 4 feet",
                "NEC 210.8": "GFCI required in wet locations",
                "NEC 406.4": "Minimum 15-18 inches above floor"
            },
            "door": {
                "IBC 1010.1.1": "32\" minimum clear width for accessible doors",
                "ADA 404.2.3": "Door hardware 34\"-48\" above floor",
                "IBC 1010.1.2": "Maximum door height 48\" for required width",
                "NFPA 80": "Fire door requirements and ratings"
            },
            "window": {
                "IBC 1405.13": "Safety glazing requirements",
                "IRC R308.4": "Hazardous location glazing",
                "Energy Code": "U-factor and SHGC requirements"
            },
            "stair": {
                "IBC 1011.5": "Minimum 44\" width for occupant load >50",
                "IBC 1011.5.2": "Minimum 7\" rise, 11\" tread",
                "IBC 1014": "Handrails required both sides if >44\" wide",
                "ADA 504": "Accessible route requirements"
            },
            "parking": {
                "ADA 502": "1 accessible space per 25 parking spaces",
                "ADA 502.2": "Accessible spaces 8' wide with 5' access aisle",
                "ADA 502.4": "Van spaces 11' wide with 5' aisle",
                "Local codes": "Check local requirements"
            },
            "sprinkler": {
                "NFPA 13": "Maximum 15' spacing for light hazard",
                "NFPA 13": "Within 12\" of ceiling",
                "NFPA 13": "Minimum 4\" from walls",
                "Local fire": "Verify local fire marshal requirements"
            },
            "plumbing fixture": {
                "IPC 405": "Fixture spacing and clearances",
                "ADA 604": "Accessible toilet requirements",
                "ADA 606": "Accessible sink requirements",
                "IPC 419": "Minimum fixture counts"
            },
            "panel": {
                "NEC 110.26": "3' clear working space in front",
                "NEC 110.26(F)": "Dedicated equipment space above",
                "NEC 408.4": "Circuit directory requirements",
                "NEC 230": "Service equipment requirements"
            }
        }
        
        # Common schedule types by discipline
        self.schedule_types = {
            "architectural": ["door schedule", "window schedule", "finish schedule", "room schedule"],
            "electrical": ["panel schedule", "lighting schedule", "fixture schedule", "load schedule"],
            "plumbing": ["fixture schedule", "equipment schedule", "pipe schedule"],
            "mechanical": ["equipment schedule", "diffuser schedule", "duct schedule"],
            "fire": ["sprinkler schedule", "device schedule", "extinguisher schedule"],
            "structural": ["column schedule", "beam schedule", "footing schedule"]
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
        Main post-processing method
        Always runs to provide additional insights
        """
        
        logger.info("🔧 Starting post-process analysis")
        
        results = {
            "schedule_found": False,
            "schedule_details": None,
            "compliance_notes": [],
            "additional_findings": [],
            "recommendations": [],
            "summary": "",
            "discrepancies": []
        }
        
        try:
            # Run analyses in parallel for efficiency
            schedule_task = asyncio.create_task(
                self._detect_and_analyze_schedule(images, visual_result.element_type)
            )
            
            compliance_task = asyncio.create_task(
                self._check_code_compliance(visual_result, comprehensive_data, question_analysis)
            )
            
            coordination_task = asyncio.create_task(
                self._check_coordination_issues(visual_result, comprehensive_data)
            )
            
            # Wait for all tasks with timeout
            tasks = [schedule_task, compliance_task, coordination_task]
            completed = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )
            
            # Process results
            schedule_info = completed[0] if not isinstance(completed[0], Exception) else {}
            compliance_findings = completed[1] if not isinstance(completed[1], Exception) else []
            coordination_issues = completed[2] if not isinstance(completed[2], Exception) else []
            
            # Update results
            if schedule_info.get("found"):
                results["schedule_found"] = True
                results["schedule_details"] = schedule_info
                
                # Check for discrepancies
                if schedule_info.get("entry_count") and visual_result.count == 0:
                    results["discrepancies"].append(
                        f"Schedule shows {schedule_info['entry_count']} items but visual analysis found 0"
                    )
                elif schedule_info.get("entry_count") and abs(schedule_info["entry_count"] - visual_result.count) > 2:
                    results["discrepancies"].append(
                        f"Schedule count ({schedule_info['entry_count']}) differs from visual count ({visual_result.count})"
                    )
            
            results["compliance_notes"] = compliance_findings
            results["additional_findings"].extend(coordination_issues)
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(
                visual_result, results["discrepancies"], compliance_findings
            )
            
            # Build summary
            results["summary"] = self._build_post_process_summary(results)
            
        except asyncio.TimeoutError:
            logger.warning("Post-process analysis timeout - using partial results")
        except Exception as e:
            logger.error(f"Post-process error: {e}")
        
        logger.info(f"✅ Post-process complete: Schedule={results['schedule_found']}, "
                   f"Compliance={len(results['compliance_notes'])} notes")
        
        return results
    
    async def _detect_and_analyze_schedule(
        self,
        images: List[Dict[str, Any]],
        element_type: str
    ) -> Dict[str, Any]:
        """Detect if drawing contains relevant schedule or tabular data"""
        
        schedule_info = {
            "found": False,
            "type": None,
            "location": None,
            "page": None,
            "description": None,
            "entry_count": None,
            "headers": [],
            "relevant_to_element": False
        }
        
        # Determine likely schedule types for this element
        likely_schedules = self._get_likely_schedules(element_type)
        
        try:
            detection_prompt = f"""SCHEDULE DETECTION TASK

Looking for schedules/tables in this drawing, particularly:
{', '.join(likely_schedules)}

Also check for any tables related to {element_type}s.

ANALYZE THE DRAWING FOR:
1. Tabular data with rows and columns
2. Schedule titles or headers
3. Lists of items with specifications
4. Legend tables

If you find ANY schedule or table:
1. What type of schedule is it?
2. Where is it located on the drawing?
3. How many entries/rows does it have?
4. What are the column headers?
5. Is it relevant to {element_type}s?

RESPONSE FORMAT:
SCHEDULE_FOUND: [YES/NO]
SCHEDULE_TYPE: [type if found]
LOCATION: [where on drawing]
ENTRY_COUNT: [number of rows/items]
HEADERS: [column headers]
RELEVANT: [YES/NO]
DESCRIPTION: [brief description]"""

            # Analyze first few pages for schedules
            pages_to_check = min(3, len(images))
            
            for i, image in enumerate(images[:pages_to_check]):
                content = [
                    {"type": "image_url", "image_url": {"url": image["url"], "detail": "high"}},
                    {"type": "text", "text": detection_prompt}
                ]
                
                response = await asyncio.wait_for(
                    self.vision_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert at detecting schedules and tables in construction drawings."
                            },
                            {"role": "user", "content": content}
                        ],
                        max_tokens=500,
                        temperature=0.0
                    ),
                    timeout=20.0
                )
                
                if response and response.choices:
                    result = response.choices[0].message.content or ""
                    
                    if "SCHEDULE_FOUND: YES" in result:
                        schedule_info["found"] = True
                        schedule_info["page"] = image.get("page", i + 1)
                        
                        # Parse details
                        type_match = re.search(r'SCHEDULE_TYPE:\s*([^\n]+)', result)
                        if type_match:
                            schedule_info["type"] = type_match.group(1).strip()
                        
                        location_match = re.search(r'LOCATION:\s*([^\n]+)', result)
                        if location_match:
                            schedule_info["location"] = location_match.group(1).strip()
                        
                        count_match = re.search(r'ENTRY_COUNT:\s*(\d+)', result)
                        if count_match:
                            schedule_info["entry_count"] = int(count_match.group(1))
                        
                        headers_match = re.search(r'HEADERS:\s*([^\n]+)', result)
                        if headers_match:
                            headers_text = headers_match.group(1).strip()
                            schedule_info["headers"] = [h.strip() for h in headers_text.split(',')]
                        
                        relevant_match = re.search(r'RELEVANT:\s*YES', result)
                        schedule_info["relevant_to_element"] = bool(relevant_match)
                        
                        desc_match = re.search(r'DESCRIPTION:\s*([^\n]+)', result)
                        if desc_match:
                            schedule_info["description"] = desc_match.group(1).strip()
                        else:
                            schedule_info["description"] = f"{schedule_info['type'] or 'Schedule'} detected"
                        
                        # If we found a relevant schedule, stop checking
                        if schedule_info["relevant_to_element"]:
                            break
                
        except Exception as e:
            logger.debug(f"Schedule detection error: {e}")
        
        return schedule_info
    
    async def _check_code_compliance(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any],
        question_analysis: Dict[str, Any]
    ) -> List[str]:
        """Check for code compliance issues based on findings"""
        
        compliance_notes = []
        element_type = visual_result.element_type
        
        try:
            # Get applicable codes
            applicable_codes = self.code_requirements.get(element_type, {})
            
            if applicable_codes:
                # Add general code requirements
                compliance_notes.append(f"**Applicable Codes for {element_type.title()}s:**")
                
                for code, requirement in applicable_codes.items():
                    compliance_notes.append(f"• {code}: {requirement}")
                
                # Element-specific compliance checks
                if element_type == "outlet" and visual_result.count > 0:
                    compliance_notes.extend(self._check_outlet_compliance(visual_result))
                
                elif element_type == "door" and visual_result.count > 0:
                    compliance_notes.extend(self._check_door_compliance(visual_result))
                
                elif element_type == "parking" and visual_result.count > 0:
                    compliance_notes.extend(self._check_parking_compliance(visual_result))
                
                elif element_type == "sprinkler" and visual_result.count > 0:
                    compliance_notes.extend(self._check_sprinkler_compliance(visual_result))
                
                elif element_type == "stair":
                    compliance_notes.extend(self._check_stair_compliance(visual_result))
                
                # Add general compliance note
                if question_analysis.get("type").value[0] == "compliance":
                    compliance_notes.append(
                        "\n⚠️ This analysis is for general guidance only. "
                        "Always verify with local authority having jurisdiction."
                    )
            
        except Exception as e:
            logger.debug(f"Compliance check error: {e}")
        
        return compliance_notes[:10]  # Limit to 10 notes
    
    async def _check_coordination_issues(
        self,
        visual_result: VisualIntelligenceResult,
        comprehensive_data: Dict[str, Any]
    ) -> List[str]:
        """Check for coordination issues between trades"""
        
        coordination_notes = []
        element_type = visual_result.element_type
        
        try:
            # Ceiling-mounted elements need coordination
            ceiling_elements = ["sprinkler", "diffuser", "light fixture", "smoke detector"]
            if element_type in ceiling_elements:
                coordination_notes.append(
                    "📐 Coordinate ceiling layout with all trades to avoid conflicts"
                )
                
                # Check if multiple ceiling elements detected
                if visual_result.count > 10:
                    coordination_notes.append(
                        "⚠️ High density of ceiling elements - careful coordination required"
                    )
            
            # Wall elements coordination
            wall_elements = ["outlet", "switch", "panel", "thermostat"]
            if element_type in wall_elements:
                coordination_notes.append(
                    "📏 Verify wall rough-in locations before finishing"
                )
            
            # Structure coordination
            if element_type in ["beam", "column"] and visual_result.count > 0:
                coordination_notes.append(
                    "🏗️ Coordinate with MEP trades for penetrations and attachments"
                )
            
            # Check drawing scale for coordination
            drawing_info = comprehensive_data.get("drawing_info", {})
            for page, info in drawing_info.items():
                if info.get("scale") and "1/8" in str(info["scale"]):
                    coordination_notes.append(
                        "📊 Small scale drawing - verify dimensions with larger scale details"
                    )
                    break
            
        except Exception as e:
            logger.debug(f"Coordination check error: {e}")
        
        return coordination_notes
    
    def _get_likely_schedules(self, element_type: str) -> List[str]:
        """Get likely schedule types for an element"""
        
        schedule_map = {
            "door": ["door schedule", "door types", "hardware schedule"],
            "window": ["window schedule", "window types", "glazing schedule"],
            "outlet": ["panel schedule", "electrical schedule", "circuit schedule"],
            "panel": ["panel schedule", "load schedule", "circuit directory"],
            "light fixture": ["lighting schedule", "fixture schedule", "lighting fixture types"],
            "plumbing fixture": ["plumbing fixture schedule", "fixture types"],
            "sprinkler": ["sprinkler schedule", "head schedule"],
            "diffuser": ["diffuser schedule", "air device schedule", "HVAC schedule"],
            "equipment": ["equipment schedule", "mechanical schedule"],
            "column": ["column schedule", "column types"],
            "beam": ["beam schedule", "framing schedule"]
        }
        
        return schedule_map.get(element_type, ["schedule", "table", "legend"])
    
    def _check_outlet_compliance(self, visual_result: VisualIntelligenceResult) -> List[str]:
        """Specific compliance checks for outlets"""
        
        notes = []
        
        # Check spacing (if we have location data)
        if len(visual_result.grid_references) > 1:
            notes.append(
                "📏 Verify outlet spacing meets NEC 210.52 (12 feet max along walls)"
            )
        
        # Check for GFCI requirements
        gfci_found = any(
            "gfci" in str(loc.get("visual_details", "")).lower()
            for loc in visual_result.locations
        )
        
        if not gfci_found:
            notes.append(
                "⚡ Verify GFCI protection in required locations (bathrooms, kitchens, exterior)"
            )
        
        return notes
    
    def _check_door_compliance(self, visual_result: VisualIntelligenceResult) -> List[str]:
        """Specific compliance checks for doors"""
        
        notes = []
        
        # Check for accessible doors
        notes.append(
            "♿ Verify at least one accessible route with 32\" clear door width"
        )
        
        # Fire doors
        fire_doors = sum(
            1 for loc in visual_result.locations
            if "fire" in str(loc.get("visual_details", "")).lower()
        )
        
        if fire_doors > 0:
            notes.append(
                f"🔥 {fire_doors} fire door(s) detected - verify ratings and hardware"
            )
        
        return notes
    
    def _check_parking_compliance(self, visual_result: VisualIntelligenceResult) -> List[str]:
        """Specific compliance checks for parking"""
        
        notes = []
        count = visual_result.count
        
        # ADA requirements
        required_accessible = max(1, count // 25)
        notes.append(
            f"♿ Minimum {required_accessible} accessible space(s) required for {count} total spaces"
        )
        
        if count >= 100:
            required_van = max(1, required_accessible // 6)
            notes.append(
                f"🚐 Minimum {required_van} van accessible space(s) required"
            )
        
        return notes
    
    def _check_sprinkler_compliance(self, visual_result: VisualIntelligenceResult) -> List[str]:
        """Specific compliance checks for sprinklers"""
        
        notes = []
        
        notes.append(
            "💧 Verify spacing meets NFPA 13 requirements for occupancy type"
        )
        
        if visual_result.count > 0:
            notes.append(
                "📏 Check coverage area calculations and hydraulic requirements"
            )
        
        return notes
    
    def _check_stair_compliance(self, visual_result: VisualIntelligenceResult) -> List[str]:
        """Specific compliance checks for stairs"""
        
        notes = []
        
        notes.append(
            "📐 Verify rise/run meets code (max 7\" rise, min 11\" tread)"
        )
        
        notes.append(
            "🏃 Check width requirements based on occupant load"
        )
        
        notes.append(
            "✋ Verify handrail requirements (both sides if >44\" wide)"
        )
        
        return notes
    
    def _generate_recommendations(
        self,
        visual_result: VisualIntelligenceResult,
        discrepancies: List[str],
        compliance_notes: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Discrepancy-based recommendations
        if discrepancies:
            recommendations.append(
                "🔍 Review and reconcile discrepancies between visual count and schedules"
            )
        
        # Zero-count recommendations
        if visual_result.count == 0:
            recommendations.append(
                f"📋 Verify if {visual_result.element_type}s are on different sheets or in specifications"
            )
        
        # High-count recommendations
        elif visual_result.count > 50:
            recommendations.append(
                f"📊 Consider detailed takeoff for {visual_result.count} {visual_result.element_type}s"
            )
        
        # Compliance recommendations
        if compliance_notes:
            recommendations.append(
                "📏 Perform detailed code compliance review with current local codes"
            )
        
        # Element-specific recommendations
        if visual_result.element_type in ["outlet", "panel", "light fixture"]:
            recommendations.append(
                "⚡ Verify electrical loads and circuit assignments"
            )
        elif visual_result.element_type in ["sprinkler", "fire alarm device"]:
            recommendations.append(
                "🚒 Coordinate with fire marshal for system review"
            )
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _build_post_process_summary(self, results: Dict[str, Any]) -> str:
        """Build a summary of post-processing findings"""
        
        summary_parts = []
        
        # Schedule findings
        if results["schedule_found"]:
            schedule = results["schedule_details"]
            summary_parts.append(
                f"📋 **SCHEDULE DETECTED**: {schedule['description']}"
            )
            if schedule.get("location"):
                summary_parts.append(f"   Location: {schedule['location']}")
            if schedule.get("entry_count"):
                summary_parts.append(f"   Entries: {schedule['entry_count']} items listed")
            if schedule.get("relevant_to_element"):
                summary_parts.append("   ✅ Directly relevant to analyzed elements")
        
        # Discrepancies
        if results["discrepancies"]:
            summary_parts.append("\n⚠️ **DISCREPANCIES FOUND**:")
            for disc in results["discrepancies"]:
                summary_parts.append(f"   • {disc}")
        
        # Compliance notes
        if results["compliance_notes"]:
            summary_parts.append("\n⚖️ **CODE COMPLIANCE NOTES**:")
            # Show first 3 compliance notes
            for note in results["compliance_notes"][:3]:
                if not note.startswith("**"):  # Skip headers
                    summary_parts.append(f"   {note}")
        
        # Recommendations
        if results["recommendations"]:
            summary_parts.append("\n💡 **RECOMMENDATIONS**:")
            for rec in results["recommendations"][:3]:
                summary_parts.append(f"   {rec}")
        
        return "\n".join(summary_parts) if summary_parts else ""
