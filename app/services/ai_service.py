# app/services/ai_service.py - ULTIMATE PROFESSIONAL VERSION (with DEBUG logging)

import asyncio
import base64
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Safety check for OpenAI import
try:
    from openai import OpenAI, APIError
    from openai.types.chat import ChatCompletionToolParam
    logger.info("✅ OpenAI SDK imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import OpenAI SDK: {e}")
    raise

# Internal config and services
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService

# Initialize OpenAI client
try:
    settings: AppSettings = get_settings()
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("🚀 OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    raise

# Professional data structures for enhanced type safety and validation
class OccupancyType(Enum):
    ASSEMBLY_CONCENTRATED = "assembly_concentrated"
    ASSEMBLY_UNCONCENTRATED = "assembly_unconcentrated" 
    BUSINESS = "business"
    EDUCATIONAL = "educational"
    FACTORY_INDUSTRIAL = "factory_industrial"
    INSTITUTIONAL = "institutional"
    MERCANTILE = "mercantile"
    RESIDENTIAL = "residential"
    STORAGE = "storage"
    UTILITY_MISCELLANEOUS = "utility_miscellaneous"

class ConstructionType(Enum):
    TYPE_I_A = "type_i_a"  # Fire resistive
    TYPE_I_B = "type_i_b"  # Fire resistive
    TYPE_II_A = "type_ii_a"  # Non-combustible
    TYPE_II_B = "type_ii_b"  # Non-combustible
    TYPE_III_A = "type_iii_a"  # Ordinary
    TYPE_III_B = "type_iii_b"  # Ordinary
    TYPE_IV = "type_iv"  # Heavy timber
    TYPE_V_A = "type_v_a"  # Wood frame
    TYPE_V_B = "type_v_b"  # Wood frame

@dataclass
class ComplianceResult:
    """Structured compliance analysis result"""
    compliant: bool
    issues: List[str]
    recommendations: List[str]
    code_references: List[str]
    severity: str  # "critical", "major", "minor", "warning"
    calculations: Dict[str, Any]
    confidence: float  # 0-1 confidence in analysis

@dataclass
class BuildingParameters:
    """Comprehensive building parameters for analysis"""
    area: float
    height: float
    occupancy_type: OccupancyType
    construction_type: ConstructionType
    sprinklered: bool = False
    stories: int = 1
    basement: bool = False
    allowable_area_increases: List[str] = None
    special_conditions: List[str] = None

class EnhancedBuildingCodeAnalyzer:
    """Professional-grade building code analyzer with comprehensive IBC/NFPA knowledge"""
    
    # Comprehensive occupancy load factors per IBC Table 1004.5
    OCCUPANCY_LOAD_FACTORS = {
        OccupancyType.ASSEMBLY_CONCENTRATED: {
            "gross": 7,  # sq ft per person
            "net": 7,
            "description": "Assembly with fixed seating, concentrated use"
        },
        OccupancyType.ASSEMBLY_UNCONCENTRATED: {
            "gross": 15,
            "net": 15, 
            "description": "Assembly without fixed seating"
        },
        OccupancyType.BUSINESS: {
            "gross": 150,
            "net": 100,
            "description": "Business occupancies"
        },
        OccupancyType.EDUCATIONAL: {
            "gross": 50,
            "net": 20,
            "description": "Educational occupancies"
        },
        OccupancyType.FACTORY_INDUSTRIAL: {
            "gross": 200,
            "net": 100,
            "description": "Factory and industrial"
        },
        OccupancyType.INSTITUTIONAL: {
            "gross": 240,
            "net": 120,
            "description": "Institutional occupancies"
        },
        OccupancyType.MERCANTILE: {
            "gross": 60,
            "net": 30,
            "description": "Mercantile occupancies"
        },
        OccupancyType.RESIDENTIAL: {
            "gross": 300,
            "net": 200,
            "description": "Residential occupancies"
        },
        OccupancyType.STORAGE: {
            "gross": 500,
            "net": 500,
            "description": "Storage occupancies"
        },
        OccupancyType.UTILITY_MISCELLANEOUS: {
            "gross": 300,
            "net": 300,
            "description": "Utility and miscellaneous"
        }
    }
    
    # Enhanced travel distance matrix per IBC Table 1017.1
    MAX_TRAVEL_DISTANCES = {
        # (occupancy, sprinklered): max_distance_feet
        (OccupancyType.ASSEMBLY_CONCENTRATED, True): 300,
        (OccupancyType.ASSEMBLY_CONCENTRATED, False): 250,
        (OccupancyType.ASSEMBLY_UNCONCENTRATED, True): 300,
        (OccupancyType.ASSEMBLY_UNCONCENTRATED, False): 250,
        (OccupancyType.BUSINESS, True): 300,
        (OccupancyType.BUSINESS, False): 200,
        (OccupancyType.EDUCATIONAL, True): 300,
        (OccupancyType.EDUCATIONAL, False): 250,
        (OccupancyType.FACTORY_INDUSTRIAL, True): 400,
        (OccupancyType.FACTORY_INDUSTRIAL, False): 200,
        (OccupancyType.INSTITUTIONAL, True): 300,
        (OccupancyType.INSTITUTIONAL, False): 200,
        (OccupancyType.MERCANTILE, True): 300,
        (OccupancyType.MERCANTILE, False): 200,
        (OccupancyType.RESIDENTIAL, True): 300,
        (OccupancyType.RESIDENTIAL, False): 250,
        (OccupancyType.STORAGE, True): 400,
        (OccupancyType.STORAGE, False): 300,
        (OccupancyType.UTILITY_MISCELLANEOUS, True): 300,
        (OccupancyType.UTILITY_MISCELLANEOUS, False): 250
    }
    
    @staticmethod
    def comprehensive_egress_analysis(building_params: BuildingParameters, 
                                      travel_distances: List[float] = None,
                                      exit_widths: List[float] = None,
                                      exit_count: int = None) -> ComplianceResult:
        """Comprehensive egress analysis with enhanced IBC compliance checking"""
        try:
            issues = []
            recommendations = []
            calculations = {}
            
            # 1. Occupant Load Calculation (IBC Section 1004)
            load_factor_data = EnhancedBuildingCodeAnalyzer.OCCUPANCY_LOAD_FACTORS[building_params.occupancy_type]
            gross_load_factor = load_factor_data["gross"]
            net_load_factor = load_factor_data["net"]
            
            # Use net area factor for more conservative calculation
            occupant_load = math.ceil(building_params.area / net_load_factor)
            calculations["occupant_load"] = {
                "area_sqft": building_params.area,
                "load_factor_net": net_load_factor,
                "load_factor_gross": gross_load_factor,
                "calculated_occupant_load": occupant_load,
                "occupancy_description": load_factor_data["description"]
            }
            
            # 2. Required Number of Exits (IBC Section 1006)
            if occupant_load <= 49:
                required_exits = 1
            elif occupant_load <= 500:
                required_exits = 2
            elif occupant_load <= 1000:
                required_exits = 3
            else:
                required_exits = 4
            
            # High-hazard occupancies require 2 exits regardless of occupant load
            if building_params.occupancy_type == OccupancyType.FACTORY_INDUSTRIAL and occupant_load > 10:
                required_exits = max(required_exits, 2)
            
            calculations["required_exits"] = {
                "occupant_load": occupant_load,
                "required_exits": required_exits,
                "provided_exits": exit_count or 0
            }
            
            if exit_count and exit_count < required_exits:
                issues.append(f"Building requires {required_exits} exits for {occupant_load} occupants, only {exit_count} provided")
                recommendations.append(f"Add {required_exits - exit_count} additional exit(s) per IBC Section 1006")
            
            # 3. Travel Distance Analysis (IBC Section 1017)
            max_allowed_distance = EnhancedBuildingCodeAnalyzer.MAX_TRAVEL_DISTANCES.get(
                (building_params.occupancy_type, building_params.sprinklered), 250
            )
            
            calculations["travel_distance"] = {
                "max_allowed_feet": max_allowed_distance,
                "sprinklered": building_params.sprinklered,
                "occupancy_type": building_params.occupancy_type.value
            }
            
            if travel_distances:
                max_actual_distance = max(travel_distances)
                calculations["travel_distance"]["max_actual_feet"] = max_actual_distance
                calculations["travel_distance"]["all_distances"] = travel_distances
                
                if max_actual_distance > max_allowed_distance:
                    issues.append(f"Maximum travel distance {max_actual_distance}' exceeds {max_allowed_distance}' limit")
                    if not building_params.sprinklered:
                        recommendations.append("Install automatic sprinkler system to increase allowable travel distance")
                    recommendations.append("Relocate exits or add additional exits to reduce travel distances")
            
            # 4. Exit Width Requirements (IBC Section 1005)
            # Stairs: 0.3" per occupant, Level exits: 0.2" per occupant
            required_stair_width = occupant_load * 0.3  # inches
            required_level_width = occupant_load * 0.2  # inches
            
            # Minimum widths per IBC
            min_door_width = 32  # inches clear width
            min_corridor_width = 44  # inches for occupant loads > 49
            
            calculations["exit_width"] = {
                "occupant_load": occupant_load,
                "required_stair_width_inches": required_stair_width,
                "required_level_width_inches": required_level_width,
                "minimum_door_width": min_door_width,
                "minimum_corridor_width": min_corridor_width
            }
            
            if exit_widths:
                total_provided_width = sum(exit_widths)
                calculations["exit_width"]["provided_widths"] = exit_widths
                calculations["exit_width"]["total_provided"] = total_provided_width
                
                if total_provided_width < required_level_width:
                    issues.append(f"Total exit width {total_provided_width:.1f}\" insufficient for {occupant_load} occupants")
                    recommendations.append(f"Increase total exit width to minimum {required_level_width:.1f}\"")
                
                # Check individual door widths
                narrow_doors = [w for w in exit_widths if w < min_door_width]
                if narrow_doors:
                    issues.append(f"Exit doors with width < {min_door_width}\" found: {narrow_doors}")
                    recommendations.append(f"All exit doors must have minimum {min_door_width}\" clear width")
            
            # 5. Special Conditions Analysis
            severity = "minor"
            if any("travel distance" in issue.lower() for issue in issues):
                severity = "major"
            if any("exits" in issue.lower() for issue in issues):
                severity = "critical"
            
            # 6. Code References
            code_references = [
                "IBC Section 1004 - Occupant Load",
                "IBC Section 1005 - Egress Width",
                "IBC Section 1006 - Number of Exits", 
                "IBC Section 1017 - Travel Distance",
                "IBC Table 1004.5 - Maximum Floor Area Allowances per Occupant",
                "IBC Table 1017.1 - Exit Access Travel Distance"
            ]
            
            return ComplianceResult(
                compliant=len(issues) == 0,
                issues=issues,
                recommendations=recommendations,
                code_references=code_references,
                severity=severity,
                calculations=calculations,
                confidence=0.95 if travel_distances and exit_widths else 0.85
            )
            
        except Exception as e:
            return ComplianceResult(
                compliant=False,
                issues=[f"Egress analysis error: {str(e)}"],
                recommendations=["Consult structural engineer for detailed egress analysis"],
                code_references=["IBC Chapter 10"],
                severity="critical",
                calculations={"error": str(e)},
                confidence=0.0
            )

    ### --- NEW CODE START --- ###
    # This is the new function that directly uses the vision model for OCR.
    @staticmethod
    async def analyze_blueprint_image_with_vision(image_url: str) -> Dict[str, Any]:
        """
        Analyzes a blueprint image using GPT-4 Vision to extract ALL visible text,
        dimensions, callouts, and symbols. This is a targeted OCR and data extraction tool.
        This function should be called when the initial document text seems insufficient
        or when visual context is critical for analysis.
        """
        logger.debug(f"🖼️ Calling Vision model for targeted data extraction from image.")
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o instead of deprecated gpt-4-vision-preview
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized Optical Character Recognition (OCR) and data extraction engine for construction blueprints. Your task is to meticulously extract all text, dimensions, callouts, and symbols you see in the provided image. Present the extracted data as a comprehensive block of text. Be precise and capture everything, including text in schedules and notes."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url, "detail": "high"}
                                },
                                {
                                    "type": "text",
                                    "text": "Extract all textual and dimensional information from this blueprint image. Be as complete and accurate as possible."
                                }
                            ]
                        }
                    ],
                    max_tokens=2500,
                    temperature=0.0
                )
            )
            extracted_text = response.choices[0].message.content
            logger.debug(f"✅ Vision model extracted text: {len(extracted_text)} chars")
            return {
                "extraction_successful": True,
                "extracted_text": extracted_text,
                "source": "gpt-4o"
            }
        except Exception as e:
            logger.error(f"❌ Vision-based text extraction failed: {e}")
            return {"extraction_successful": False, "error": str(e)}
    ### --- NEW CODE END --- ###

    @staticmethod
    def _analyze_ramp_accessibility(**kwargs) -> ComplianceResult:
        """Placeholder for ramp accessibility analysis"""
        return ComplianceResult(
            compliant=False,
            issues=["Ramp analysis not yet implemented"],
            recommendations=["Consult ADA guidelines for ramp requirements"],
            code_references=["ADA Standards Section 405"],
            severity="warning",
            calculations={},
            confidence=0.0
        )
    
    @staticmethod
    def _analyze_parking_accessibility(**kwargs) -> ComplianceResult:
        """Placeholder for parking accessibility analysis"""
        return ComplianceResult(
            compliant=False,
            issues=["Parking analysis not yet implemented"],
            recommendations=["Consult ADA guidelines for parking requirements"],
            code_references=["ADA Standards Section 502"],
            severity="warning",
            calculations={},
            confidence=0.0
        )
    
    @staticmethod
    def _analyze_restroom_accessibility(**kwargs) -> ComplianceResult:
        """Placeholder for restroom accessibility analysis"""
        return ComplianceResult(
            compliant=False,
            issues=["Restroom analysis not yet implemented"],
            recommendations=["Consult ADA guidelines for restroom requirements"],
            code_references=["ADA Standards Section 604"],
            severity="warning",
            calculations={},
            confidence=0.0
        )
    
    @staticmethod
    def _get_comprehensive_fire_ratings(building_params: BuildingParameters) -> Dict[str, Any]:
        """Placeholder for fire rating analysis"""
        return {
            "fire_separation": "Analysis pending",
            "structural_ratings": "Analysis pending",
            "wall_ratings": "Analysis pending"
        }

    
    @staticmethod
    def comprehensive_accessibility_analysis(element_type: str, **kwargs) -> ComplianceResult:
        """Enhanced ADA compliance analysis with detailed requirements"""
        try:
            if element_type.lower() == "door":
                return EnhancedBuildingCodeAnalyzer._analyze_door_accessibility(**kwargs)
            elif element_type.lower() == "ramp":
                return EnhancedBuildingCodeAnalyzer._analyze_ramp_accessibility(**kwargs)
            elif element_type.lower() == "parking":
                return EnhancedBuildingCodeAnalyzer._analyze_parking_accessibility(**kwargs)
            elif element_type.lower() == "restroom":
                return EnhancedBuildingCodeAnalyzer._analyze_restroom_accessibility(**kwargs)
            else:
                return ComplianceResult(
                    compliant=False,
                    issues=[f"Accessibility analysis not implemented for: {element_type}"],
                    recommendations=["Consult ADA guidelines for specific requirements"],
                    code_references=["ADA Standards"],
                    severity="warning",
                    calculations={},
                    confidence=0.0
                )
                
        except Exception as e:
            return ComplianceResult(
                compliant=False,
                issues=[f"Accessibility analysis error: {str(e)}"],
                recommendations=["Consult accessibility specialist"],
                code_references=["ADA Standards"],
                severity="major",
                calculations={"error": str(e)},
                confidence=0.0
            )
    
    @staticmethod
    def _analyze_door_accessibility(width: float = 32, threshold_height: float = 0.5,
                                    opening_force: float = 5, maneuvering_clearance: float = 18,
                                    closing_speed: float = 5, **kwargs) -> ComplianceResult:
        """Comprehensive door accessibility analysis"""
        issues = []
        recommendations = []
        calculations = {}
        
        # ADA Standards for door accessibility
        min_clear_width = 32  # inches
        max_threshold = 0.75  # inches
        max_opening_force = 5  # pounds
        min_maneuvering_clearance = 18  # inches pull side
        min_closing_time = 5  # seconds
        
        calculations.update({
            "provided_width": width,
            "provided_threshold": threshold_height,
            "provided_opening_force": opening_force,
            "provided_clearance": maneuvering_clearance,
            "provided_closing_speed": closing_speed,
            "requirements": {
                "min_width": min_clear_width,
                "max_threshold": max_threshold,
                "max_opening_force": max_opening_force,
                "min_clearance": min_maneuvering_clearance,
                "min_closing_time": min_closing_time
            }
        })
        
        # Check compliance
        if width < min_clear_width:
            issues.append(f"Door clear width {width}\" < {min_clear_width}\" minimum")
            recommendations.append(f"Increase door clear opening to minimum {min_clear_width}\"")
        
        if threshold_height > max_threshold:
            issues.append(f"Threshold height {threshold_height}\" > {max_threshold}\" maximum")
            recommendations.append(f"Reduce threshold to maximum {max_threshold}\" or provide beveled edge")
        
        if opening_force > max_opening_force:
            issues.append(f"Opening force {opening_force} lbf > {max_opening_force} lbf maximum")
            recommendations.append("Adjust door closer or install power-assist operator")
        
        if maneuvering_clearance < min_maneuvering_clearance:
            issues.append(f"Maneuvering clearance {maneuvering_clearance}\" < {min_maneuvering_clearance}\" minimum")
            recommendations.append(f"Provide minimum {min_maneuvering_clearance}\" clearance on pull side")
        
        severity = "critical" if any("width" in issue for issue in issues) else "minor"
        
        return ComplianceResult(
            compliant=len(issues) == 0,
            issues=issues,
            recommendations=recommendations,
            code_references=["ADA Standards Section 404", "IBC Section 1010"],
            severity=severity,
            calculations=calculations,
            confidence=0.98
        )
    
    @staticmethod
    def comprehensive_fire_safety_analysis(building_params: BuildingParameters) -> ComplianceResult:
        """Professional fire safety analysis per NFPA and IBC"""
        try:
            issues = []
            recommendations = []
            calculations = {}
            
            # 1. Sprinkler System Requirements (IBC Section 903)
            sprinkler_required = EnhancedBuildingCodeAnalyzer._check_sprinkler_requirements(building_params)
            
            calculations["sprinkler_analysis"] = {
                "required": sprinkler_required["required"],
                "reasons": sprinkler_required["reasons"],
                "provided": building_params.sprinklered,
                "area_sqft": building_params.area,
                "height_feet": building_params.height,
                "stories": building_params.stories
            }
            
            if sprinkler_required["required"] and not building_params.sprinklered:
                issues.append("Automatic sprinkler system required")
                for reason in sprinkler_required["reasons"]:
                    issues.append(f"  - {reason}")
                recommendations.append("Install NFPA 13 compliant automatic sprinkler system")
            
            # 2. Fire Alarm System Requirements (IBC Section 907)
            alarm_required = EnhancedBuildingCodeAnalyzer._check_alarm_requirements(building_params)
            
            calculations["alarm_analysis"] = {
                "required": alarm_required["required"],
                "reasons": alarm_required["reasons"],
                "area_sqft": building_params.area,
                "occupancy_type": building_params.occupancy_type.value
            }
            
            if alarm_required["required"]:
                recommendations.append("Install NFPA 72 compliant fire alarm system")
                recommendations.append("Provide manual pull stations at exits")
                recommendations.append("Install smoke detection in required areas")
            
            # 3. Fire Rating Requirements
            fire_ratings = EnhancedBuildingCodeAnalyzer._get_comprehensive_fire_ratings(building_params)
            calculations["fire_ratings"] = fire_ratings
            
            # 4. Emergency Egress Requirements
            emergency_lighting_required = building_params.occupancy_type in [
                OccupancyType.ASSEMBLY_CONCENTRATED, OccupancyType.ASSEMBLY_UNCONCENTRATED,
                OccupancyType.BUSINESS, OccupancyType.EDUCATIONAL, OccupancyType.INSTITUTIONAL
            ]
            
            if emergency_lighting_required:
                recommendations.append("Provide emergency egress lighting per IBC Section 1008")
                recommendations.append("Install illuminated exit signs at all exits")
            
            # 5. Determine severity
            severity = "critical" if sprinkler_required["required"] and not building_params.sprinklered else "minor"
            
            code_references = [
                "IBC Section 903 - Automatic Sprinkler Systems",
                "IBC Section 907 - Fire Alarm and Detection Systems",
                "IBC Section 1008 - Emergency Egress",
                "NFPA 13 - Installation of Sprinkler Systems",
                "NFPA 72 - National Fire Alarm Code"
            ]
            
            return ComplianceResult(
                compliant=len(issues) == 0,
                issues=issues,
                recommendations=recommendations,
                code_references=code_references,
                severity=severity,
                calculations=calculations,
                confidence=0.92
            )
            
        except Exception as e:
            return ComplianceResult(
                compliant=False,
                issues=[f"Fire safety analysis error: {str(e)}"],
                recommendations=["Consult fire protection engineer"],
                code_references=["IBC Chapter 9", "NFPA Standards"],
                severity="critical",
                calculations={"error": str(e)},
                confidence=0.0
            )
    
    @staticmethod
    def _check_sprinkler_requirements(building_params: BuildingParameters) -> Dict[str, Any]:
        """Comprehensive sprinkler requirement analysis"""
        required = False
        reasons = []
        
        # High-rise buildings (IBC Section 403)
        if building_params.height > 75:
            required = True
            reasons.append(f"High-rise building (>{75}' height)")
        
        # Large area requirements vary by occupancy
        area_thresholds = {
            OccupancyType.ASSEMBLY_CONCENTRATED: 5000,
            OccupancyType.ASSEMBLY_UNCONCENTRATED: 5000,
            OccupancyType.BUSINESS: 12000,
            OccupancyType.EDUCATIONAL: 20000,
            OccupancyType.FACTORY_INDUSTRIAL: 12000,
            OccupancyType.INSTITUTIONAL: 5000,
            OccupancyType.MERCANTILE: 12000,
            OccupancyType.RESIDENTIAL: 5000,
            OccupancyType.STORAGE: 12000,
            OccupancyType.UTILITY_MISCELLANEOUS: 12000
        }
        
        threshold = area_thresholds.get(building_params.occupancy_type, 12000)
        if building_params.area > threshold:
            required = True
            reasons.append(f"Area {building_params.area} sq ft exceeds {threshold} sq ft threshold")
        
        # Multi-story requirements
        if building_params.stories > 2 and building_params.occupancy_type in [
            OccupancyType.BUSINESS, OccupancyType.MERCANTILE, OccupancyType.FACTORY_INDUSTRIAL
        ]:
            required = True
            reasons.append(f"Multi-story ({building_params.stories} stories) {building_params.occupancy_type.value}")
        
        # Basement requirements
        if building_params.basement and building_params.area > 1500:
            required = True
            reasons.append("Basement area > 1500 sq ft")
        
        return {"required": required, "reasons": reasons}
    
    @staticmethod
    def _check_alarm_requirements(building_params: BuildingParameters) -> Dict[str, Any]:
        """Fire alarm system requirement analysis"""
        required = False
        reasons = []
        
        # Occupancy-based requirements
        alarm_required_occupancies = [
            OccupancyType.ASSEMBLY_CONCENTRATED,
            OccupancyType.ASSEMBLY_UNCONCENTRATED,
            OccupancyType.BUSINESS,
            OccupancyType.EDUCATIONAL,
            OccupancyType.INSTITUTIONAL,
            OccupancyType.MERCANTILE
        ]
        
        if building_params.occupancy_type in alarm_required_occupancies:
            required = True
            reasons.append(f"{building_params.occupancy_type.value} occupancy requires fire alarm")
        
        # Area-based requirements
        if building_params.area > 5000:
            required = True
            reasons.append(f"Building area {building_params.area} sq ft > 5000 sq ft")
        
        # Height-based requirements
        if building_params.height > 55:
            required = True
            reasons.append(f"Building height {building_params.height}' > 55'")
        
        return {"required": required, "reasons": reasons}


class UniversalConstructionAnalyzer:
    """Enhanced universal calculation and analysis tools with advanced capabilities"""
    
    @staticmethod
    async def calculate_by_area_advanced(total_area: float, coverage_per_unit: float, 
                                           unit_name: str = "items", area_unit: str = "sqft",
                                           efficiency_factor: float = 1.0,
                                           overlap_factor: float = 0.0) -> Dict[str, Any]:
        """Advanced area-based calculation with efficiency and overlap considerations"""
        try:
            # Account for efficiency and overlap
            effective_coverage = coverage_per_unit * efficiency_factor * (1 - overlap_factor)
            units_needed = math.ceil(total_area / effective_coverage)
            
            # Multiple waste factor scenarios
            waste_scenarios = {
                "minimal": 1.05,
                "standard": 1.10,
                "conservative": 1.15,
                "high_waste": 1.20
            }
            
            waste_calculations = {}
            for scenario, factor in waste_scenarios.items():
                waste_calculations[scenario] = {
                    "units": math.ceil(units_needed * factor),
                    "waste_percentage": (factor - 1) * 100,
                    "extra_units": math.ceil(units_needed * factor) - units_needed
                }
            
            # Cost optimization analysis
            bulk_breakpoints = [100, 250, 500, 1000]
            cost_analysis = {}
            for breakpoint in bulk_breakpoints:
                if units_needed >= breakpoint * 0.8:  # Within 80% of breakpoint
                    potential_savings = (breakpoint - units_needed) / units_needed * 100
                    cost_analysis[f"bulk_{breakpoint}"] = {
                        "additional_units": breakpoint - units_needed,
                        "potential_savings_percent": round(potential_savings, 2)
                    }
            
            return {
                "calculation_type": "advanced_area_based",
                "input_parameters": {
                    "total_area": total_area,
                    "area_unit": area_unit,
                    "coverage_per_unit": coverage_per_unit,
                    "efficiency_factor": efficiency_factor,
                    "overlap_factor": overlap_factor
                },
                "basic_calculation": {
                    "effective_coverage": effective_coverage,
                    "units_needed": units_needed,
                    "unit_type": unit_name
                },
                "waste_scenarios": waste_calculations,
                "cost_optimization": cost_analysis,
                "formula": f"{total_area} {area_unit} ÷ ({coverage_per_unit} × {efficiency_factor} × {1-overlap_factor}) = {units_needed} {unit_name}",
                "confidence_level": 0.95 if efficiency_factor == 1.0 else 0.85
            }
        except Exception as e:
            return {"error": f"Advanced area calculation failed: {str(e)}"}
    
    @staticmethod
    def extract_measurements_with_ai_validation(text: str) -> Dict[str, Any]:
        """Enhanced measurement extraction with AI validation and error detection"""
        try:
            # Enhanced patterns with validation
            enhanced_patterns = {
                # Architectural measurements
                'room_dimensions': r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(?:feet|ft|')",
                'door_sizes': r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(?:door|dr)",
                'window_sizes': r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(?:window|win)",
                'ceiling_heights': r"(\d+\.?\d*)\s*(?:feet|ft|')\s*(?:ceiling|clg|height)",
                
                # Structural measurements
                'beam_sizes': r"([wW]\d+[xX]\d+|[hH][sS]\d+[xX]\d+|[cC]\d+[xX]\d+)",
                'column_sizes': r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(?:column|col)",
                'slab_thickness': r"(\d+\.?\d*)\s*(?:inch|in|\")\s*(?:slab|deck)",
                'rebar_spacing': r"(\d+\.?\d*)\s*(?:oc|on\s*center)\s*(?:rebar|bar)",
                
                # MEP measurements
                'duct_sizes': r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(?:duct|rectangular)",
                'pipe_sizes': r"(\d+\.?\d*)\s*(?:inch|in|\")\s*(?:pipe|dia|diameter)",
                'conduit_sizes': r"(\d+\.?\d*)\s*(?:inch|in|\")\s*(?:conduit|emt|rig)",
                'wire_sizes': r"#(\d+)\s*(?:awg|wire|conductor)",
                
                # Loads and capacities
                'electrical_loads': r"(\d+\.?\d*)\s*(?:amp|amps|amperes?|a)(?!\w)",
                'hvac_loads': r"(\d+\.?\d*)\s*(?:cfm|tons?|btu)",
                'structural_loads': r"(\d+\.?\d*)\s*(?:psf|plf|kips?|lbs?)",
                
                # Fire safety
                'sprinkler_spacing': r"(\d+\.?\d*)\s*(?:feet|ft|')\s*(?:oc|spacing|centers?)",
                'exit_widths': r"(\d+\.?\d*)\s*(?:inches?|in|\")\s*(?:wide|width|clear)",
                'fire_ratings': r"(\d+\.?\d*)\s*(?:hour|hr|hours?)\s*(?:rated|rating|fire)",
                
                # Areas and quantities
                'areas_sqft': r"(\d+[\d,]*\.?\d*)\s*(?:sq\.?\s*ft|sqft|square\s*feet)",
                'areas_sqm': r"(\d+[\d,]*\.?\d*)\s*(?:sq\.?\s*m|sqm|square\s*met)",
                'quantities': r"(\d+)\s*(?:each|ea|pcs?|pieces?|units?|items?)",
                
                # Special conditions
                'percentages': r"(\d+(?:\.\d+)?)\s*%",
                'temperatures': r"(\d+\.?\d*)\s*(?:degrees?|°)\s*(?:f|fahrenheit|c|celsius)",
                'slopes_grades': r"(\d+(?:\.\d+)?)\s*%\s*(?:slope|grade|pitch)",
                'flow_rates': r"(\d+\.?\d*)\s*(?:gpm|gph|lpm|cfm|cms)",
                'pressures': r"(\d+\.?\d*)\s*(?:psi|bar|kpa|pa)",
                'velocities': r"(\d+\.?\d*)\s*(?:mph|fps|mps|fpm)"
            }
            
            extracted_data = {}
            validation_results = {}
            
            text_lower = text.lower()
            
            for category, pattern in enhanced_patterns.items():
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    # Validate extracted measurements
                    validated_matches = []
                    validation_issues = []
                    
                    for match in matches:
                        validation = UniversalConstructionAnalyzer._validate_measurement(category, match)
                        if validation["valid"]:
                            validated_matches.append(match)
                        else:
                            validation_issues.append(validation["issue"])
                    
                    extracted_data[category] = validated_matches
                    if validation_issues:
                        validation_results[category] = validation_issues
            
            # Cross-reference validation
            cross_validation = UniversalConstructionAnalyzer._cross_validate_measurements(extracted_data)
            
            # Calculate confidence score
            total_extractions = sum(len(matches) for matches in extracted_data.values())
            total_issues = sum(len(issues) for issues in validation_results.values())
            confidence = max(0.5, 1.0 - (total_issues / max(total_extractions, 1)) * 0.5)
            
            return {
                "extraction_successful": True,
                "extracted_measurements": extracted_data,
                "validation_results": validation_results,
                "cross_validation": cross_validation,
                "total_patterns_found": len([k for k, v in extracted_data.items() if v]),
                "confidence_score": round(confidence, 2),
                "quality_indicators": {
                    "measurement_density": total_extractions / max(len(text), 1) * 1000,
                    "validation_pass_rate": (total_extractions - total_issues) / max(total_extractions, 1),
                    "pattern_diversity": len(extracted_data) / len(enhanced_patterns)
                }
            }
            
        except Exception as e:
            return {"error": f"Enhanced measurement extraction failed: {str(e)}"}
    
    @staticmethod
    def _validate_measurement(category: str, measurement) -> Dict[str, Any]:
        """Validate individual measurements for reasonableness"""
        try:
            # Convert measurement to numeric if possible
            if isinstance(measurement, tuple):
                values = [float(x) for x in measurement if str(x).replace('.', '').isdigit()]
            else:
                values = [float(measurement)] if str(measurement).replace('.', '').isdigit() else []
            
            if not values:
                return {"valid": True, "issue": None}  # Non-numeric measurements pass through
            
            # Validation rules by category
            validation_rules = {
                'room_dimensions': {"min": 1, "max": 1000, "typical_range": (8, 100)},
                'door_sizes': {"min": 1, "max": 20, "typical_range": (2, 12)},
                'ceiling_heights': {"min": 6, "max": 50, "typical_range": (8, 20)},
                'beam_sizes': {"min": 4, "max": 48, "typical_range": (8, 24)},
                'pipe_sizes': {"min": 0.5, "max": 48, "typical_range": (0.75, 12)},
                'electrical_loads': {"min": 1, "max": 10000, "typical_range": (15, 400)},
                'areas_sqft': {"min": 1, "max": 1000000, "typical_range": (100, 50000)},
                'fire_ratings': {"min": 0.5, "max": 4, "typical_range": (1, 3)},
                'sprinkler_spacing': {"min": 6, "max": 20, "typical_range": (10, 15)},
                'percentages': {"min": 0, "max": 100, "typical_range": (1, 99)}
            }
            
            rules = validation_rules.get(category, {"min": 0, "max": float('inf'), "typical_range": (0, float('inf'))})
            
            for value in values:
                if value < rules["min"] or value > rules["max"]:
                    return {
                        "valid": False, 
                        "issue": f"{category}: {value} outside reasonable range ({rules['min']}-{rules['max']})"
                    }
                
                # Warning for atypical values
                typical_min, typical_max = rules["typical_range"]
                if not (typical_min <= value <= typical_max):
                    return {
                        "valid": True,
                        "issue": f"{category}: {value} outside typical range ({typical_min}-{typical_max}) - verify"
                    }
            
            return {"valid": True, "issue": None}
            
        except Exception as e:
            return {"valid": False, "issue": f"Validation error: {str(e)}"}
    
    @staticmethod
    def _cross_validate_measurements(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate measurements for consistency"""
        issues = []
        suggestions = []
        
        try:
            # Check room dimension consistency
            if 'room_dimensions' in extracted_data and 'areas_sqft' in extracted_data:
                room_dims = extracted_data['room_dimensions']
                areas = extracted_data['areas_sqft']
                
                if room_dims and areas:
                    # Calculate area from dimensions and compare
                    for dim_pair in room_dims:
                        if len(dim_pair) >= 2:
                            calculated_area = float(dim_pair[0]) * float(dim_pair[1])
                            for area in areas:
                                if isinstance(area, tuple):
                                    area_val = float(area[0])
                                else:
                                    area_val = float(area)
                                
                                # Check if areas are reasonably close (within 20%)
                                if abs(calculated_area - area_val) / area_val > 0.2:
                                    issues.append(f"Area mismatch: {dim_pair[0]}' x {dim_pair[1]}' = {calculated_area} sq ft vs listed {area_val} sq ft")
            
            # Check electrical load consistency
            if 'electrical_loads' in extracted_data:
                loads = [float(x[0]) if isinstance(x, tuple) else float(x) for x in extracted_data['electrical_loads']]
                if loads:
                    total_load = sum(loads)
                    if total_load > 2000:  # High total load
                        suggestions.append(f"High total electrical load ({total_load}A) - verify panel capacity")
            
            # Check HVAC sizing consistency
            if 'hvac_loads' in extracted_data and 'areas_sqft' in extracted_data:
                hvac_loads = extracted_data['hvac_loads']
                areas = extracted_data['areas_sqft']
                
                if hvac_loads and areas:
                    # Rough sizing check (400-600 sq ft per ton)
                    for area in areas:
                        area_val = float(area[0]) if isinstance(area, tuple) else float(area)
                        estimated_tons = area_val / 500  # Conservative estimate
                        
                        for load in hvac_loads:
                            load_val = float(load[0]) if isinstance(load, tuple) else float(load)
                            if 'ton' in str(load).lower() and abs(load_val - estimated_tons) / estimated_tons > 0.5:
                                suggestions.append(f"HVAC sizing check: {area_val} sq ft suggests ~{estimated_tons:.1f} tons vs {load_val} tons specified")
            
            return {
                "consistency_issues": issues,
                "sizing_suggestions": suggestions,
                "validation_passed": len(issues) == 0
            }
            
        except Exception as e:
            return {"error": f"Cross-validation failed: {str(e)}"}


class ProfessionalAIService:
    """Ultimate professional-grade AI service with advanced building analysis capabilities"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client with enhanced error handling
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Professional OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.analyzer = UniversalConstructionAnalyzer()
        self.code_analyzer = EnhancedBuildingCodeAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Professional-grade tool definitions with enhanced capabilities
        self.tools = [
            # Enhanced calculation tools
            {
                "type": "function",
                "function": {
                    "name": "calculate_by_area_advanced",
                    "description": "Advanced area-based calculations with efficiency factors, overlap considerations, and cost optimization",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "total_area": {"type": "number", "description": "Total area needing coverage"},
                            "coverage_per_unit": {"type": "number", "description": "Area covered by each unit"},
                            "unit_name": {"type": "string", "description": "Type of item"},
                            "area_unit": {"type": "string", "description": "Unit of area", "default": "sqft"},
                            "efficiency_factor": {"type": "number", "description": "Installation efficiency (0.8-1.0)", "default": 1.0},
                            "overlap_factor": {"type": "number", "description": "Overlap factor (0.0-0.2)", "default": 0.0}
                        },
                        "required": ["total_area", "coverage_per_unit", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_measurements_enhanced",
                    "description": "Enhanced measurement extraction from TEXT using Regex. Use this after extracting text from an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_text": {"type": "string", "description": "Text content from the document"}
                        },
                        "required": ["document_text"]
                    }
                }
            },
            ### --- NEW CODE START --- ###
            # This is the new tool definition that enables vision analysis.
            {
                "type": "function",
                "function": {
                    "name": "analyze_blueprint_image_with_vision",
                    "description": "Performs targeted visual analysis (OCR) on a blueprint image to extract all text, dimensions, and callouts. Use this tool FIRST if the user asks a question about the visual content of the blueprint and the initial text context seems insufficient.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string", "description": "The base64 encoded data URL of the blueprint image to analyze."}
                        },
                        "required": ["image_url"]
                    }
                }
            },
            ### --- NEW CODE END --- ###
            # Professional building code compliance tools
            {
                "type": "function",
                "function": {
                    "name": "comprehensive_egress_analysis",
                    "description": "Professional-grade egress analysis per IBC with detailed occupant load calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "number", "description": "Total area in square feet"},
                            "occupancy_type": {"type": "string", "enum": ["assembly_concentrated", "assembly_unconcentrated", "business", "educational", "factory_industrial", "institutional", "mercantile", "residential", "storage", "utility_miscellaneous"], "description": "IBC occupancy classification"},
                            "height": {"type": "number", "description": "Building height in feet"},
                            "construction_type": {"type": "string", "enum": ["type_i_a", "type_i_b", "type_ii_a", "type_ii_b", "type_iii_a", "type_iii_b", "type_iv", "type_v_a", "type_v_b"], "description": "IBC construction type"},
                            "sprinklered": {"type": "boolean", "description": "Whether building has sprinkler system"},
                            "stories": {"type": "number", "description": "Number of stories", "default": 1},
                            "basement": {"type": "boolean", "description": "Whether building has basement", "default": False},
                            "travel_distances": {"type": "array", "items": {"type": "number"}, "description": "Travel distances to exits in feet"},
                            "exit_widths": {"type": "array", "items": {"type": "number"}, "description": "Exit widths in inches"},
                            "exit_count": {"type": "number", "description": "Number of exits provided"}
                        },
                        "required": ["area", "occupancy_type", "construction_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "comprehensive_accessibility_analysis",
                    "description": "Detailed ADA compliance analysis for building elements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_type": {"type": "string", "enum": ["door", "ramp", "parking", "restroom"], "description": "Type of element to analyze"},
                            "width": {"type": "number", "description": "Width in inches", "default": 32},
                            "threshold_height": {"type": "number", "description": "Threshold height in inches", "default": 0.5},
                            "opening_force": {"type": "number", "description": "Opening force in pounds", "default": 5},
                            "maneuvering_clearance": {"type": "number", "description": "Maneuvering clearance in inches", "default": 18},
                            "closing_speed": {"type": "number", "description": "Closing speed in seconds", "default": 5},
                            "slope": {"type": "number", "description": "Slope percentage for ramps"},
                            "length": {"type": "number", "description": "Length in feet"},
                            "has_handrails": {"type": "boolean", "description": "Whether handrails are present"},
                            "has_landings": {"type": "boolean", "description": "Whether landings are present"}
                        },
                        "required": ["element_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "comprehensive_fire_safety_analysis",
                    "description": "Professional fire safety analysis per NFPA and IBC standards",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "number", "description": "Building area in square feet"},
                            "height": {"type": "number", "description": "Building height in feet"},
                            "occupancy_type": {"type": "string", "enum": ["assembly_concentrated", "assembly_unconcentrated", "business", "educational", "factory_industrial", "institutional", "mercantile", "residential", "storage", "utility_miscellaneous"], "description": "IBC occupancy classification"},
                            "construction_type": {"type": "string", "enum": ["type_i_a", "type_i_b", "type_ii_a", "type_ii_b", "type_iii_a", "type_iii_b", "type_iv", "type_v_a", "type_v_b"], "description": "IBC construction type"},
                            "sprinklered": {"type": "boolean", "description": "Whether building has sprinkler system", "default": False},
                            "stories": {"type": "number", "description": "Number of stories", "default": 1},
                            "basement": {"type": "boolean", "description": "Whether building has basement", "default": False}
                        },
                        "required": ["area", "occupancy_type", "construction_type"]
                    }
                }
            },
            # Existing tools (maintained for compatibility)
            {
                "type": "function",
                "function": {
                    "name": "calculate_by_perimeter",
                    "description": "Calculate quantities for perimeter-based items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "perimeter_length": {"type": "number"},
                            "spacing": {"type": "number"},
                            "unit_name": {"type": "string"},
                            "length_unit": {"type": "string", "default": "ft"}
                        },
                        "required": ["perimeter_length", "spacing", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_grid_pattern",
                    "description": "Calculate quantities for grid-pattern items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "length": {"type": "number"},
                            "width": {"type": "number"},
                            "spacing": {"type": "number"},
                            "unit_name": {"type": "string"},
                            "unit": {"type": "string", "default": "ft"}
                        },
                        "required": ["length", "width", "spacing", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_units_professional",
                    "description": "Professional unit conversion with uncertainty analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "from_unit": {"type": "string"},
                            "to_unit": {"type": "string"},
                            "precision_required": {"type": "boolean", "default": False}
                        },
                        "required": ["value", "from_unit", "to_unit"]
                    }
                }
            }
        ]
        
        logger.info("✅ Professional AI Service initialized with enhanced capabilities")
    
    async def _execute_tool_call_async(self, tool_call) -> str:
        """Asynchronous tool execution with enhanced error handling"""
        function_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
            
            # Route to appropriate analyzer method
            if function_name == "calculate_by_area_advanced":
                result = await self.analyzer.calculate_by_area_advanced(**arguments)
            elif function_name == "extract_measurements_enhanced":
                result = self.analyzer.extract_measurements_with_ai_validation(arguments["document_text"])
            
            ### --- NEW CODE START --- ###
            # This is the new logic to handle the vision tool call.
            elif function_name == "analyze_blueprint_image_with_vision":
                result = await self.code_analyzer.analyze_blueprint_image_with_vision(arguments["image_url"])
            ### --- NEW CODE END --- ###
            
            elif function_name == "comprehensive_egress_analysis":
                # Convert arguments to BuildingParameters
                building_params = BuildingParameters(
                    area=arguments["area"],
                    height=arguments.get("height", 20),
                    occupancy_type=OccupancyType(arguments["occupancy_type"]),
                    construction_type=ConstructionType(arguments["construction_type"]),
                    sprinklered=arguments.get("sprinklered", False),
                    stories=arguments.get("stories", 1),
                    basement=arguments.get("basement", False)
                )
                
                result = self.code_analyzer.comprehensive_egress_analysis(
                    building_params,
                    travel_distances=arguments.get("travel_distances"),
                    exit_widths=arguments.get("exit_widths"),
                    exit_count=arguments.get("exit_count")
                )
                # Convert ComplianceResult to dict
                result = result.__dict__
                
            elif function_name == "comprehensive_accessibility_analysis":
                result = self.code_analyzer.comprehensive_accessibility_analysis(
                    arguments["element_type"], **{k: v for k, v in arguments.items() if k != "element_type"}
                )
                result = result.__dict__
                
            elif function_name == "comprehensive_fire_safety_analysis":
                building_params = BuildingParameters(
                    area=arguments["area"],
                    height=arguments.get("height", 20),
                    occupancy_type=OccupancyType(arguments["occupancy_type"]),
                    construction_type=ConstructionType(arguments["construction_type"]),
                    sprinklered=arguments.get("sprinklered", False),
                    stories=arguments.get("stories", 1),
                    basement=arguments.get("basement", False)
                )
                
                result = self.code_analyzer.comprehensive_fire_safety_analysis(building_params)
                result = result.__dict__
                
            # Legacy tool support
            elif function_name == "calculate_by_perimeter":
                result = UniversalConstructionAnalyzer.calculate_by_perimeter(**arguments)
            elif function_name == "calculate_grid_pattern":
                result = UniversalConstructionAnalyzer.calculate_grid_pattern(**arguments)
            elif function_name == "convert_units_professional":
                result = UniversalConstructionAnalyzer.universal_unit_converter(
                    arguments["value"], arguments["from_unit"], arguments["to_unit"]
                )
                if arguments.get("precision_required"):
                    result["precision_note"] = "High precision conversion - verify for critical applications"
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Tool execution error in {function_name}: {e}")
            return json.dumps({
                "error": f"Tool execution failed: {str(e)}",
                "function": function_name,
                "timestamp": datetime.now().isoformat()
            })

    async def get_ai_response(self, prompt: str, document_id: str, storage_service: StorageService, author: str = None) -> str:
        """Enhanced AI response with professional-grade analysis"""
        try:
            logger.info(f"🤖 Processing professional AI request for document {document_id}")
            
            # Load document context with enhanced error handling
            document_text = ""
            image_urls = []
            
            try:
                # Parallel loading of document assets
                context_task = storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                
                image_task = storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_page_1.png"
                )
                
                # Execute tasks with timeout
                document_text = await asyncio.wait_for(context_task, timeout=30.0)
                logger.info(f"📄 Loaded document context: {len(document_text):,} characters")
                
                try:
                    page_1_bytes = await asyncio.wait_for(image_task, timeout=30.0)
                    page_1_b64 = base64.b64encode(page_1_bytes).decode('utf-8')
                    image_url = f"data:image/png;base64,{page_1_b64}"
                    image_urls.append(image_url)
                    logger.info(f"🖼️ Loaded page 1 image: {len(page_1_bytes):,} bytes")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Image loading timeout - proceeding without image")
                except Exception as img_error:
                    logger.warning(f"⚠️ Could not load page images: {img_error}")
                
            except asyncio.TimeoutError:
                logger.error("❌ Document loading timeout")
                return "I apologize, but the document loading timed out. Please try again."
            except Exception as e:
                logger.error(f"❌ Failed to load document data: {e}")
                return f"I couldn't access the document data for '{document_id}'. Please ensure the document is properly processed."
            
            # Process with enhanced professional analysis
            result = await self.process_query_with_professional_analysis(
                prompt=prompt,
                document_text=document_text,
                image_url=image_urls[0] if image_urls else None,
                document_id=document_id
            )
            
            if result["success"]:
                logger.info(f"✅ Professional AI response generated: {result['tools_used']} tools, confidence: {result.get('confidence', 'N/A')}")
                return result["ai_response"]
            else:
                logger.error(f"❌ Professional AI processing failed: {result.get('error', 'Unknown error')}")
                return "I encountered an error during analysis. Please try rephrasing your question or contact support."
                
        except Exception as e:
            logger.error(f"❌ Professional AI response failed: {e}")
            return f"I encountered a system error: {str(e)}"

    async def process_query_with_professional_analysis(self, prompt: str, document_text: str = "", 
                                                       image_url: str = None, document_id: str = None) -> Dict[str, Any]:
        """Professional-grade query processing with comprehensive analysis"""
        try:
            # Enhanced system message with professional expertise
            system_message = {
                "role": "system",
                "content": """You are a PROFESSIONAL CONSTRUCTION AI ENGINEER with comprehensive expertise in building codes, engineering calculations, and construction analysis.

🏗️ PROFESSIONAL QUALIFICATIONS:
- Licensed Professional Engineer (PE) level knowledge
- Certified Building Code Expert (IBC, NFPA, ADA, NEC, IMC, IPC)
- 20+ years equivalent construction industry experience
- Expert in all construction trades and disciplines

📋 COMPREHENSIVE CAPABILITIES:
- BUILDING CODE COMPLIANCE: IBC, NFPA, ADA, NEC, etc.
- ENGINEERING ANALYSIS: Structural, Electrical, Mechanical, Plumbing, Fire Protection.
- CONSTRUCTION MANAGEMENT: Sequencing, materials, quantities.

PROFESSIONAL METHODOLOGY:
1.  **Assess the Goal:** Understand the user's question and the available data (text and image).
2.  **Visual First (If Necessary):** If the user's question requires visual information (like counting items, reading a specific callout, or understanding layout) and the provided text seems insufficient, **your first step must be to use the `analyze_blueprint_image_with_vision` tool.** This gives you the necessary visual context.
3.  **Extract Data:** Use the text (either from the initial context or from the vision tool) to extract specific measurements and data using the `extract_measurements_enhanced` tool.
4.  **Analyze & Calculate:** Apply appropriate codes and standards using the `comprehensive_..._analysis` tools.
5.  **Synthesize:** Formulate a professional answer, citing code references, showing calculations, and providing actionable recommendations.
6.  **Disclose:** Always flag items requiring licensed professional review.

Always approach each analysis as a licensed professional engineer would, with attention to safety, code compliance, and professional responsibility."""
            }
            
            # Build enhanced message structure
            messages = [system_message]
            
            # Add user message with professional context
            user_message = {"role": "user", "content": []}
            
            if image_url:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"}
                })
            
            # Enhanced document context
            if document_text:
                context_summary = f"""
DOCUMENT ANALYSIS REQUEST:
Document ID: {document_id or 'Unknown'}
Content Length: {len(document_text):,} characters
Analysis Type: Professional Engineering Review

DOCUMENT CONTENT (Initial Text Extraction):
{document_text}

PROFESSIONAL ANALYSIS REQUEST:
{prompt}

Please provide a comprehensive professional analysis. Remember to use the `analyze_blueprint_image_with_vision` tool if the initial text is insufficient to answer the user's question about the visual layout or details.
"""
                user_message["content"].append({"type": "text", "text": context_summary})
            else:
                user_message["content"].append({"type": "text", "text": f"Professional Engineering Query: {prompt}"})
            
            messages.append(user_message)
            
            # Enhanced API call with professional parameters
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model="gpt-4o",  # Use gpt-4o instead of deprecated gpt-4-vision-preview
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        max_tokens=4000,
                        temperature=0.05,  # Very low temperature for professional accuracy
                        top_p=0.9,
                        frequency_penalty=0.1
                    )
                ),
                timeout=120.0  # Extended timeout for complex analysis
            )
            
            assistant_message = response.choices[0].message
            tools_used = 0
            
            # Enhanced tool execution with parallel processing
            if assistant_message.tool_calls:
                messages.append(assistant_message)
                tools_used = len(assistant_message.tool_calls)
                
                # Execute tool calls with enhanced error handling
                tool_tasks = [
                    self._execute_tool_call_async(tool_call)
                    for tool_call in assistant_message.tool_calls
                ]
                
                tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)
                
                # Add tool results to conversation
                for tool_call, result in zip(assistant_message.tool_calls, tool_results):
                    if isinstance(result, Exception):
                        result_content = json.dumps({
                            "error": f"Tool execution error: {str(result)}",
                            "tool": tool_call.function.name
                        })
                    else:
                        result_content = result
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content
                    })
                
                # Get enhanced final response
                final_response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=4000,
                            temperature=0.05
                        )
                    ),
                    timeout=60.0
                )
                
                final_content = final_response.choices[0].message.content
            else:
                final_content = assistant_message.content
            
            # Calculate confidence score based on multiple factors
            confidence_factors = {
                "tools_used": min(tools_used / 3, 1.0) * 0.3,  # More tools = higher confidence
                "document_length": min(len(document_text) / 10000, 1.0) * 0.2,  # More context = higher confidence
                "image_available": 0.2 if image_url else 0.0,  # Visual analysis adds confidence
                "response_length": min(len(final_content) / 2000, 1.0) * 0.3  # Detailed response = higher confidence
            }
            
            overall_confidence = sum(confidence_factors.values())
            
            return {
                "ai_response": final_content,
                "tools_used": tools_used,
                "confidence": round(overall_confidence, 2),
                "success": True,
                "analysis_metadata": {
                    "document_length": len(document_text),
                    "image_analyzed": image_url is not None,
                    "response_length": len(final_content),
                    "processing_level": "professional_grade"
                }
            }
            
        except asyncio.TimeoutError:
            logger.error("Professional analysis timeout")
            return {
                "ai_response": "Analysis timeout - please try a more focused question or contact support for complex analysis.",
                "success": False,
                "error": "Processing timeout"
            }
        except Exception as e:
            logger.error(f"Professional AI processing error: {e}")
            return {
                "ai_response": f"Professional analysis encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }

    async def comprehensive_blueprint_compliance_audit(self, document_id: str, storage_service: StorageService) -> Dict[str, Any]:
        """Comprehensive blueprint compliance audit with professional-grade analysis"""
        try:
            logger.info(f"🔍 Starting comprehensive professional compliance audit for {document_id}")
            
            # Load all document assets
            context_blob = f"{document_id}_context.txt"
            document_text = await storage_service.download_blob_as_text(
                container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                blob_name=context_blob
            )
            
            # Load multiple page images if available
            image_urls = []
            for page_num in range(1, 6):  # Check up to 5 pages
                try:
                    page_blob = f"{document_id}_page_{page_num}.png"
                    page_bytes = await storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=page_blob
                    )
                    page_b64 = base64.b64encode(page_bytes).decode('utf-8')
                    image_urls.append(f"data:image/png;base64,{page_b64}")
                except Exception:
                    break  # No more pages available
            
            # Professional audit prompt
            audit_prompt = """
            Conduct a COMPREHENSIVE PROFESSIONAL COMPLIANCE AUDIT of this construction document.
            
            As a Licensed Professional Engineer, provide a detailed analysis covering:

            🏗️ **DOCUMENT CLASSIFICATION & SCOPE**
            - Document type and professional discipline
            - Project scope and complexity assessment
            - Applicable codes and standards identification
            - Required professional reviews and approvals

            📐 **TECHNICAL ANALYSIS**
            - Dimensional verification and measurement validation
            - Engineering calculations review
            - Material specifications assessment
            - System design adequacy evaluation

            📋 **CODE COMPLIANCE VERIFICATION**
            - IBC compliance (occupancy, construction type, area/height limits)
            - NFPA fire protection requirements (sprinklers, alarms, egress)
            - ADA accessibility compliance (doors, ramps, clearances)
            - Energy code compliance (IECC requirements)
            - Trade-specific code compliance (NEC, IMC, IPC)

            🚨 **SAFETY & REGULATORY ANALYSIS**
            - Life safety system adequacy
            - Structural safety considerations
            - Fire protection system requirements
            - Emergency egress evaluation
            - Accessibility barrier identification

            ⚖️ **PROFESSIONAL LIABILITY ASSESSMENT**
            - Items requiring professional engineer review
            - Potential liability exposure areas
            - Recommended professional consultations
            - Standard of care compliance evaluation

            📊 **COMPLIANCE SCORING & RECOMMENDATIONS**
            - Overall compliance rating (0-100%)
            - Critical issues requiring immediate attention
            - Major issues requiring professional review
            - Minor issues and recommendations for improvement
            - Priority ranking for remediation

            Use all available calculation and compliance tools to provide quantitative analysis.
            Provide specific code references and industry standard citations.
            """
            
            # Execute comprehensive analysis
            audit_results = []
            
            # Process with each available image for complete analysis
            for i, image_url in enumerate(image_urls[:3]):  # Analyze up to 3 pages
                page_result = await self.process_query_with_professional_analysis(
                    prompt=f"{audit_prompt}\n\n--- ANALYZING PAGE {i+1} ---",
                    document_text=document_text if i == 0 else "",  # Full text only on first page
                    image_url=image_url,
                    document_id=document_id
                )
                audit_results.append({
                    "page": i + 1,
                    "analysis": page_result["ai_response"],
                    "tools_used": page_result["tools_used"],
                    "confidence": page_result.get("confidence", 0.8)
                })
            
            # If no images, do text-only analysis
            if not image_urls:
                text_result = await self.process_query_with_professional_analysis(
                    prompt=audit_prompt,
                    document_text=document_text,
                    document_id=document_id
                )
                audit_results.append({
                    "page": "text_only",
                    "analysis": text_result["ai_response"],
                    "tools_used": text_result["tools_used"],
                    "confidence": text_result.get("confidence", 0.7)
                })
            
            # Generate executive summary
            summary_prompt = f"""
            Based on the comprehensive compliance audit results, provide an EXECUTIVE SUMMARY:

            **OVERALL COMPLIANCE STATUS:** [Compliant/Issues Found/Major Concerns/Critical Issues]
            **COMPLIANCE SCORE:** [0-100%]
            **PROFESSIONAL REVIEW REQUIRED:** [Yes/No]

            **CRITICAL ISSUES:** (Immediate attention required)
            **MAJOR ISSUES:** (Professional review recommended)  
            **MINOR ISSUES:** (Improvement opportunities)

            **NEXT STEPS:**
            **ESTIMATED REMEDIATION EFFORT:**
            **PROFESSIONAL CONSULTATIONS RECOMMENDED:**

            Audit Results Data:
            {json.dumps([r["analysis"][:500] + "..." for r in audit_results], indent=2)}
            """
            
            summary_result = await self.process_query_with_professional_analysis(
                prompt=summary_prompt,
                document_text="",
                document_id=document_id
            )
            
            return {
                "document_id": document_id,
                "audit_complete": True,
                "audit_type": "professional_comprehensive",
                "pages_analyzed": len(audit_results),
                "total_tools_used": sum(r["tools_used"] for r in audit_results),
                "average_confidence": sum(r["confidence"] for r in audit_results) / len(audit_results),
                "detailed_analysis": audit_results,
                "executive_summary": summary_result["ai_response"],
                "audit_timestamp": datetime.now().isoformat(),
                "professional_grade": True,
                "liability_disclaimer": "This analysis is for informational purposes only. Professional engineer review required for final design approval."
            }
            
        except Exception as e:
            logger.error(f"❌ Professional compliance audit failed: {e}")
            return {
                "document_id": document_id,
                "audit_complete": False,
                "error": str(e),
                "audit_timestamp": datetime.now().isoformat(),
                "professional_grade": True
            }

    async def generate_professional_report(self, document_id: str, storage_service: StorageService, 
                                           report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate professional-grade compliance and analysis reports"""
        try:
            # Conduct full audit first
            audit_results = await self.comprehensive_blueprint_compliance_audit(document_id, storage_service)
            
            if not audit_results.get("audit_complete"):
                return {"error": "Unable to complete audit for report generation"}
            
            # Generate formal report
            report_prompt = f"""
            Generate a formal PROFESSIONAL ENGINEERING REPORT based on the audit results.
            
            Format as a professional consulting engineer's report with:

            **EXECUTIVE SUMMARY**
            **PROJECT INFORMATION**
            **SCOPE OF REVIEW**
            **METHODOLOGY**
            **FINDINGS AND ANALYSIS**
            **CODE COMPLIANCE STATUS**
            **RECOMMENDATIONS**
            **PROFESSIONAL OPINION**
            **LIMITATIONS AND DISCLAIMERS**

            Use formal engineering language and professional presentation standards.
            Include specific code references and industry standards.
            Provide clear action items and priority rankings.

            Base the report on this audit data:
            {json.dumps(audit_results, indent=2, default=str)[:5000]}...
            """
            
            report_result = await self.process_query_with_professional_analysis(
                prompt=report_prompt,
                document_text="",
                document_id=document_id
            )
            
            return {
                "document_id": document_id,
                "report_type": report_type,
                "report_content": report_result["ai_response"],
                "based_on_audit": audit_results["audit_timestamp"],
                "professional_grade": True,
                "report_timestamp": datetime.now().isoformat(),
                "disclaimer": "This report represents professional engineering analysis based on available information. Final design approval requires licensed professional engineer review and approval."
            }
            
        except Exception as e:
            logger.error(f"❌ Professional report generation failed: {e}")
            return {
                "document_id": document_id,
                "error": str(e),
                "report_timestamp": datetime.now().isoformat()
            }

    def get_professional_capabilities(self) -> Dict[str, Any]:
        """Return comprehensive list of professional capabilities"""
        return {
            "building_codes": [
                "IBC - International Building Code (All Chapters)",
                "NFPA 13 - Sprinkler Systems Design and Installation",
                "NFPA 72 - Fire Alarm and Detection Systems",
                "NFPA 101 - Life Safety Code",
                "ADA Standards - Accessibility Guidelines",
                "NEC - National Electrical Code",
                "IMC - International Mechanical Code",
                "IPC - International Plumbing Code",
                "IECC - International Energy Conservation Code",
                "AISC 360 - Steel Construction Specification",
                "ACI 318 - Concrete Design and Construction",
                "NDS - Wood Construction Standards",
                "ASCE 7 - Minimum Design Loads and Associated Criteria"
            ],
            "engineering_disciplines": [
                "Structural Engineering (Design and Analysis)",
                "Electrical Engineering (Power Systems and Controls)",
                "Mechanical Engineering (HVAC and Building Systems)",
                "Fire Protection Engineering (Life Safety and Systems)",
                "Civil Engineering (Site Development and Utilities)",
                "Architectural Engineering (Building Design Integration)"
            ],
            "professional_services": [
                "Code Compliance Verification",
                "Professional Engineering Review",
                "Constructability Analysis",
                "Value Engineering Assessment",
                "Risk Assessment and Mitigation",
                "Professional Liability Evaluation",
                "Peer Review and Quality Assurance",
                "Expert Witness Support"
            ],
            "analysis_types": [
                "Comprehensive Blueprint Review",
                "Code Compliance Audit",
                "Life Safety Analysis",
                "Accessibility Compliance Review",
                "Fire Protection System Design Review",
                "Structural Adequacy Assessment",
                "MEP Systems Analysis",
                "Energy Code Compliance Review"
            ],
            "deliverables": [
                "Professional Engineering Reports",
                "Code Compliance Matrices",
                "Calculation Packages",
                "Professional Recommendations",
                "Remediation Action Plans",
                "Professional Opinions and Certifications"
            ]
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Enhanced utility functions for professional operations
def calculate_professional_confidence(factors: Dict[str, float]) -> float:
    """Calculate professional confidence score with weighted factors"""
    weights = {
        "code_compliance": 0.4,
        "engineering_analysis": 0.3,
        "documentation_quality": 0.2,
        "professional_review": 0.1
    }
    
    weighted_score = sum(weights.get(factor, 0.1) * score for factor, score in factors.items())
    return min(1.0, max(0.0, weighted_score))

def generate_professional_disclaimer() -> str:
    """Generate professional engineering disclaimer"""
    return """
PROFESSIONAL ENGINEERING DISCLAIMER:

This analysis has been performed using professional engineering principles and industry standards. 
However, this review is based on the information provided and should not be considered a substitute 
for a comprehensive professional engineering review by a licensed professional engineer.

The analysis is provided for informational purposes only and should be verified by qualified 
professionals before implementation. No warranty is provided regarding the completeness or 
accuracy of the analysis.

Final design approval and code compliance verification must be performed by licensed professionals 
in accordance with applicable state and local regulations.
"""

# Alias for backward compatibility
AIService = ProfessionalAIService
