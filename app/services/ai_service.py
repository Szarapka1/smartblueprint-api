# app/services/ai_service.py - COMPLETE INTELLIGENT AI SERVICE

import asyncio
import base64
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# OpenAI imports
try:
    from openai import OpenAI, APIError
    logger.info("✅ OpenAI SDK imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import OpenAI SDK: {e}")
    raise

# Internal imports
from app.core.config import AppSettings, get_settings
from app.services.storage_service import StorageService

# Initialize settings and client
try:
    settings: AppSettings = get_settings()
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("🚀 OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    raise


class QuestionIntent:
    """Classifies user question intent"""
    
    GENERAL_OVERVIEW = "general_overview"  # "tell me about this document"
    SIMPLE_INFO = "simple_info"  # "what's the address?"
    COUNTING = "counting"  # "how many doors?"
    CALCULATION = "calculation"  # "calculate the electrical load"
    CODE_COMPLIANCE = "code_compliance"  # "does this meet code?"
    SPECIFIC_TECHNICAL = "specific_technical"  # "what size beam for..."
    COMPARISON = "comparison"  # "is X bigger than Y?"
    
    @staticmethod
    def classify(question: str) -> str:
        """Determine the intent of the question"""
        q_lower = question.lower()
        
        # General overview patterns
        if any(phrase in q_lower for phrase in [
            "tell me about", "what is this", "describe this", "what kind of",
            "what type of", "overview", "summary", "general"
        ]):
            return QuestionIntent.GENERAL_OVERVIEW
        
        # Simple info patterns
        elif any(phrase in q_lower for phrase in [
            "address", "project name", "architect", "owner", "date",
            "sheet number", "scale", "who", "where is the project"
        ]):
            return QuestionIntent.SIMPLE_INFO
        
        # Counting patterns
        elif any(phrase in q_lower for phrase in [
            "how many", "count", "number of", "total"
        ]):
            return QuestionIntent.COUNTING
        
        # Calculation patterns
        elif any(phrase in q_lower for phrase in [
            "calculate", "size", "what size", "capacity", "load",
            "tonnage", "cfm", "gallons", "watts"
        ]):
            return QuestionIntent.CALCULATION
        
        # Code compliance patterns
        elif any(phrase in q_lower for phrase in [
            "code", "compliant", "meet", "requirement", "violation",
            "ibc", "nfpa", "ada", "nec", "allowed"
        ]):
            return QuestionIntent.CODE_COMPLIANCE
        
        # Technical/specific patterns
        else:
            return QuestionIntent.SPECIFIC_TECHNICAL


class ComprehensiveBuildingKnowledge:
    """Complete building knowledge for all trades and codes"""
    
    # IBC Occupancy Classifications and Load Factors
    OCCUPANCY_DATA = {
        "assembly_concentrated": {"load_factor": 7, "group": "A-1", "description": "Fixed seating"},
        "assembly_unconcentrated": {"load_factor": 15, "group": "A-2/A-3", "description": "Without fixed seating"},
        "business": {"load_factor": 150, "group": "B", "description": "Office, professional"},
        "educational": {"load_factor": 20, "group": "E", "description": "Schools K-12"},
        "factory": {"load_factor": 100, "group": "F", "description": "Industrial"},
        "institutional": {"load_factor": 120, "group": "I", "description": "Hospitals, care facilities"},
        "mercantile": {"load_factor": 30, "group": "M", "description": "Retail, stores"},
        "residential": {"load_factor": 200, "group": "R", "description": "Dwelling units"},
        "storage": {"load_factor": 300, "group": "S", "description": "Warehouses"},
        "office": {"load_factor": 150, "group": "B", "description": "Business use"}
    }
    
    # NFPA 13 Sprinkler Requirements
    SPRINKLER_REQUIREMENTS = {
        "light_hazard": {
            "spacing": 15,  # feet
            "coverage": 225,  # sq ft per head max
            "typical": 200,  # sq ft conservative
            "density": 0.1,  # gpm/sq ft
            "examples": ["office", "residential", "educational", "religious"]
        },
        "ordinary_hazard_1": {
            "spacing": 15,
            "coverage": 130,
            "typical": 130,
            "density": 0.15,
            "examples": ["mercantile", "parking", "restaurant_seating"]
        },
        "ordinary_hazard_2": {
            "spacing": 15,
            "coverage": 130,
            "typical": 130,
            "density": 0.2,
            "examples": ["mechanical_rooms", "kitchen", "storage"]
        }
    }
    
    # Comprehensive Electrical Requirements
    ELECTRICAL_REQUIREMENTS = {
        "lighting_power_density": {  # Watts per sq ft (IECC Table C405.3.2)
            "office": 0.82,
            "retail": 1.26,
            "warehouse": 0.66,
            "dining": 0.89,
            "kitchen": 1.21,
            "restroom": 0.98,
            "corridor": 0.66,
            "parking": 0.15
        },
        "receptacle_loads": {  # Watts per sq ft
            "office": 1.5,
            "retail": 1.0,
            "restaurant": 2.0,
            "residential": 3.0,
            "warehouse": 0.5
        },
        "panel_sizes": [100, 150, 200, 225, 400, 600, 800, 1200, 1600, 2000, 2500, 3000],
        "voltage_systems": {
            "residential": "120/240V 1-phase",
            "commercial": "120/208V 3-phase",
            "industrial": "277/480V 3-phase"
        }
    }
    
    # HVAC Design Parameters
    HVAC_PARAMETERS = {
        "ventilation_rates": {  # ASHRAE 62.1
            "office": {"cfm_per_person": 20, "cfm_per_sqft": 0.06},
            "retail": {"cfm_per_person": 15, "cfm_per_sqft": 0.12},
            "restaurant": {"cfm_per_person": 20, "cfm_per_sqft": 0.18},
            "classroom": {"cfm_per_person": 10, "cfm_per_sqft": 0.12}
        },
        "cooling_loads": {  # Square feet per ton
            "office": 350,
            "retail": 250,
            "restaurant": 200,
            "warehouse": 500,
            "residential": 400
        },
        "duct_velocities": {  # FPM
            "main": 1000,
            "branch": 600,
            "residential": 700
        }
    }
    
    # Plumbing Fixture Requirements (IPC Table 403.1)
    PLUMBING_REQUIREMENTS = {
        "business": {
            "water_closets": {"male": 1.25, "female": 1.25},  # per 1000 sq ft
            "lavatories": 1.25,
            "drinking_fountains": 0.1
        },
        "mercantile": {
            "water_closets": {"male": 0.5, "female": 0.5},
            "lavatories": 0.5,
            "drinking_fountains": 0.1
        },
        "restaurant": {
            "water_closets": {"male": 2.0, "female": 2.0},
            "lavatories": 2.0,
            "drinking_fountains": 0.1
        }
    }
    
    # ADA/Accessibility Requirements
    ADA_REQUIREMENTS = {
        "doors": {
            "clear_width": 32,  # inches
            "threshold": 0.5,  # inches max
            "maneuvering_clearance": 18,  # inches pull side
            "handle_height": {"min": 34, "max": 48}
        },
        "corridors": {
            "width": 36,  # inches minimum
            "passing_space": 60,  # inches for wheelchairs
            "turn_around": 60  # inches diameter
        },
        "ramps": {
            "slope": 8.33,  # percent (1:12)
            "landing": 60,  # inches
            "handrail_height": {"min": 34, "max": 38}
        },
        "parking": {
            "space_width": 96,  # inches
            "aisle_width": 60,  # inches (96 for van)
            "ratio": {  # Spaces required
                "1-25": 1,
                "26-50": 2,
                "51-75": 3,
                "76-100": 4,
                "101-150": 5,
                "151-200": 6
            }
        }
    }
    
    # Structural Systems
    STRUCTURAL_DATA = {
        "steel": {
            "typical_bay": "25-30 ft",
            "max_bay": "45 ft",
            "floor_system": "metal deck with concrete",
            "fireproofing": "1-3 hour rating required"
        },
        "concrete": {
            "typical_bay": "20-25 ft",
            "max_bay": "35 ft",
            "slab_thickness": {"office": "8-10 in", "parking": "10-12 in"},
            "column_size": "18-36 in typical"
        },
        "wood": {
            "max_span": {"2x10": 16, "2x12": 20, "engineered": 24},
            "spacing": "16 or 24 in o.c.",
            "loads": {"residential": 40, "commercial": 50}  # psf live load
        }
    }


class DocumentIntelligence:
    """Intelligent document understanding and analysis"""
    
    def __init__(self):
        self.knowledge = ComprehensiveBuildingKnowledge()
        self.current_context = {
            "document_type": None,
            "area": None,
            "occupancy": None,
            "key_elements": {},
            "measurements": {},
            "visual_elements": []
        }
    
    async def analyze_blueprint_visually(self, image_url: str) -> Dict[str, Any]:
        """Comprehensive visual analysis of blueprint"""
        logger.info("🔍 Performing intelligent visual analysis")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are analyzing a construction blueprint. Extract ALL information comprehensively:

PROJECT INFORMATION:
- Project name, address, location
- Owner, architect, engineer, contractor
- Drawing date, sheet number, scale
- Drawing type (architectural, structural, MEP, etc.)

SPACES AND AREAS:
- List all rooms/spaces with names and square footage
- Overall building dimensions and total area
- Floor level information

BUILDING ELEMENTS (count everything):
- Doors: Count, sizes, types, labels
- Windows: Count, sizes, types
- Walls: Types, thickness
- Columns: Count, size, spacing
- Fixtures: All plumbing, electrical, mechanical

SYSTEMS:
- HVAC: Equipment, ductwork, diffusers
- Electrical: Panels, circuits, outlets, lights
- Plumbing: Fixtures, piping
- Fire Protection: Sprinklers, alarms, extinguishers

DIMENSIONS AND MEASUREMENTS:
- All dimensional callouts
- Grid lines and spacing
- Ceiling heights
- Clear widths

NOTES AND SPECIFICATIONS:
- All text notes and callouts
- Material specifications
- Code references
- Special requirements

Be extremely thorough. This information will be used for code compliance and calculations."""
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                                {"type": "text", "text": "Extract all information from this blueprint."}
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
            )
            
            extracted_text = response.choices[0].message.content
            
            # Parse and structure the extracted information
            self._update_context_from_visual(extracted_text)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "structured_data": self.current_context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_context_from_visual(self, extracted_text: str):
        """Update context with extracted visual information"""
        text_lower = extracted_text.lower()
        
        # Extract area
        area_pattern = r'(\d+[\d,]*\.?\d*)\s*(?:sq\.?\s*ft|square\s*feet)'
        areas = re.findall(area_pattern, text_lower)
        if areas:
            self.current_context["area"] = max(float(a.replace(',', '')) for a in areas)
        
        # Determine occupancy
        if "office" in text_lower:
            self.current_context["occupancy"] = "business"
        elif "retail" in text_lower or "store" in text_lower:
            self.current_context["occupancy"] = "mercantile"
        elif "restaurant" in text_lower:
            self.current_context["occupancy"] = "assembly"
        
        # Count elements
        element_patterns = {
            "doors": r'(\d+)\s*doors?|door\s*(?:d|dr)?(\d+)',
            "windows": r'(\d+)\s*windows?|window\s*(?:w)?(\d+)',
            "sprinklers": r'sprinkler|spklr',
            "lights": r'light|fixture|luminaire',
            "outlets": r'outlet|receptacle|duplex'
        }
        
        for element, pattern in element_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                self.current_context["key_elements"][element] = len(matches)
    
    def extract_measurements(self, text: str) -> Dict[str, List[Any]]:
        """Extract all measurements from text"""
        patterns = {
            # Dimensions
            'dimensions': r'(\d+)[\'\-]?\s*(\d+)?\"?\s*[xX]\s*(\d+)[\'\-]?\s*(\d+)?\"?',
            'areas': r'(\d+[\d,]*\.?\d*)\s*(?:sq\.?\s*ft|square\s*feet|sf)',
            'linear': r'(\d+)[\'\-]?\s*(\d+)?\"?\s*(?:lf|linear|lin)',
            
            # Electrical
            'electrical_loads': r'(\d+\.?\d*)\s*(?:kw|kva|amps?|a)',
            'voltage': r'(\d+)\s*(?:v|volts?)',
            'circuits': r'(\d+)\s*(?:circuit|ckt)',
            
            # HVAC
            'cfm': r'(\d+)\s*cfm',
            'tonnage': r'(\d+\.?\d*)\s*(?:tons?|ton)',
            'duct_size': r'(\d+)\s*[xX]\s*(\d+)\s*(?:duct|supply|return)',
            
            # Plumbing
            'pipe_size': r'(\d+\.?\d*)\s*(?:inch|in|\")\s*(?:pipe|p)',
            'fixture_units': r'(\d+\.?\d*)\s*(?:fu|fixture\s*units?)',
            
            # Structural
            'beam_size': r'([wW]\d+[xX]\d+)',
            'column_size': r'(\d+)\s*[xX]\s*(\d+)\s*(?:col|column)',
            
            # Fire Protection
            'fire_rating': r'(\d+\.?\d*)\s*(?:hr|hour)\s*(?:rated?|rating)'
        }
        
        results = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results[category] = matches
                
        return results
    
    def calculate_code_requirements(self, system: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate requirements for any building system"""
        
        if system == "sprinklers":
            return self._calculate_sprinkler_requirements(parameters)
        elif system == "electrical":
            return self._calculate_electrical_requirements(parameters)
        elif system == "hvac":
            return self._calculate_hvac_requirements(parameters)
        elif system == "plumbing":
            return self._calculate_plumbing_requirements(parameters)
        elif system == "egress":
            return self._calculate_egress_requirements(parameters)
        elif system == "parking":
            return self._calculate_parking_requirements(parameters)
        elif system == "accessibility":
            return self._check_ada_compliance(parameters)
        else:
            return {"error": f"Unknown system: {system}"}
    
    def _calculate_sprinkler_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """NFPA 13 sprinkler calculations"""
        area = params.get("area", 0)
        occupancy = params.get("occupancy", "office")
        
        # Determine hazard classification
        if occupancy in ["office", "residential", "educational"]:
            hazard = "light_hazard"
        elif occupancy in ["retail", "mercantile"]:
            hazard = "ordinary_hazard_1"
        else:
            hazard = "ordinary_hazard_2"
        
        req = self.knowledge.SPRINKLER_REQUIREMENTS[hazard]
        
        # Calculate heads needed
        heads_required = math.ceil(area / req["typical"])
        
        # Add 10% for coverage overlap and obstructions
        heads_recommended = math.ceil(heads_required * 1.1)
        
        return {
            "area": area,
            "occupancy": occupancy,
            "hazard_classification": hazard,
            "coverage_per_head": req["typical"],
            "max_spacing": req["spacing"],
            "heads_calculated": heads_required,
            "heads_recommended": heads_recommended,
            "density": req["density"],
            "code_reference": "NFPA 13",
            "notes": [
                f"Based on {hazard.replace('_', ' ')} occupancy",
                f"Maximum {req['coverage']} sq ft per head allowed",
                f"Typical design uses {req['typical']} sq ft per head",
                "Additional heads may be required for obstructions"
            ]
        }
    
    def _calculate_electrical_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """NEC electrical load calculations"""
        area = params.get("area", 0)
        occupancy = params.get("occupancy", "office")
        
        # Get load densities
        lighting = self.knowledge.ELECTRICAL_REQUIREMENTS["lighting_power_density"].get(occupancy, 1.0)
        receptacle = self.knowledge.ELECTRICAL_REQUIREMENTS["receptacle_loads"].get(occupancy, 1.5)
        
        # Calculate loads
        lighting_load = area * lighting
        receptacle_load = area * receptacle
        
        # Apply demand factors
        lighting_demand = lighting_load  # 100% for continuous
        receptacle_demand = receptacle_load * 0.5  # 50% demand factor
        
        total_demand = lighting_demand + receptacle_demand
        
        # Calculate amperage (assuming 208V 3-phase)
        voltage = 208
        amps = total_demand / (voltage * 1.732)  # 1.732 for 3-phase
        
        # Size panel (125% for continuous loads)
        required_amps = amps * 1.25
        panel_size = next((size for size in self.knowledge.ELECTRICAL_REQUIREMENTS["panel_sizes"] 
                          if size >= required_amps), 400)
        
        return {
            "area": area,
            "occupancy": occupancy,
            "lighting": {
                "watts_per_sqft": lighting,
                "total_watts": round(lighting_load),
                "demand_watts": round(lighting_demand)
            },
            "receptacles": {
                "watts_per_sqft": receptacle,
                "total_watts": round(receptacle_load),
                "demand_watts": round(receptacle_demand)
            },
            "total": {
                "connected_load": round(lighting_load + receptacle_load),
                "demand_load": round(total_demand),
                "calculated_amps": round(amps, 1),
                "required_amps": round(required_amps, 1),
                "panel_size": f"{panel_size}A"
            },
            "voltage_system": self.knowledge.ELECTRICAL_REQUIREMENTS["voltage_systems"]["commercial"],
            "code_references": [
                "NEC Article 220 - Load Calculations",
                "NEC Table 220.12 - Lighting Loads",
                "IECC Table C405.3.2 - LPD"
            ]
        }
    
    def _calculate_hvac_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """ASHRAE 62.1 ventilation and cooling calculations"""
        area = params.get("area", 0)
        occupancy = params.get("occupancy", "office")
        occupants = params.get("occupants", area / 150)  # Default occupant density
        
        # Get ventilation rates
        vent = self.knowledge.HVAC_PARAMETERS["ventilation_rates"].get(
            occupancy, {"cfm_per_person": 20, "cfm_per_sqft": 0.06}
        )
        
        # Calculate ventilation
        people_oa = occupants * vent["cfm_per_person"]
        area_oa = area * vent["cfm_per_sqft"]
        total_oa = people_oa + area_oa
        
        # Calculate cooling
        sqft_per_ton = self.knowledge.HVAC_PARAMETERS["cooling_loads"].get(occupancy, 350)
        tons_required = area / sqft_per_ton
        
        return {
            "area": area,
            "occupancy": occupancy,
            "occupants": round(occupants),
            "ventilation": {
                "people_cfm": round(people_oa),
                "area_cfm": round(area_oa),
                "total_cfm": round(total_oa),
                "cfm_per_person": vent["cfm_per_person"],
                "cfm_per_sqft": vent["cfm_per_sqft"]
            },
            "cooling": {
                "tons_required": round(tons_required, 1),
                "btuh": round(tons_required * 12000),
                "sqft_per_ton": sqft_per_ton
            },
            "code_references": [
                "ASHRAE 62.1 - Ventilation",
                "ASHRAE 90.1 - Energy Standard",
                "IMC Chapter 4 - Ventilation"
            ]
        }
    
    def _calculate_plumbing_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """IPC plumbing fixture calculations"""
        area = params.get("area", 0)
        occupancy = params.get("occupancy", "business")
        
        # Get fixture requirements
        fixtures = self.knowledge.PLUMBING_REQUIREMENTS.get(
            occupancy, self.knowledge.PLUMBING_REQUIREMENTS["business"]
        )
        
        # Calculate required fixtures
        wc_required = max(2, math.ceil(area * fixtures["water_closets"]["male"] / 1000))
        lav_required = max(2, math.ceil(area * fixtures["lavatories"] / 1000))
        df_required = max(1, math.ceil(area * fixtures["drinking_fountains"] / 1000))
        
        return {
            "area": area,
            "occupancy": occupancy,
            "fixtures_required": {
                "water_closets": wc_required,
                "lavatories": lav_required,
                "drinking_fountains": df_required
            },
            "code_reference": "IPC Table 403.1",
            "notes": [
                "Minimum 2 water closets required",
                "50% allocated to each gender typically",
                "Additional fixtures may be required for assembly occupancies"
            ]
        }
    
    def _calculate_egress_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """IBC egress calculations"""
        area = params.get("area", 0)
        occupancy = params.get("occupancy", "business")
        sprinklered = params.get("sprinklered", True)
        
        # Get occupant load factor
        occupancy_data = self.knowledge.OCCUPANCY_DATA.get(
            occupancy, self.knowledge.OCCUPANCY_DATA["business"]
        )
        
        # Calculate occupant load
        occupant_load = math.ceil(area / occupancy_data["load_factor"])
        
        # Determine required exits
        if occupant_load <= 49:
            required_exits = 1
        elif occupant_load <= 500:
            required_exits = 2
        elif occupant_load <= 1000:
            required_exits = 3
        else:
            required_exits = 4
        
        # Calculate exit width
        stair_width = occupant_load * 0.3  # inches
        level_width = occupant_load * 0.2  # inches
        
        # Travel distance
        group = occupancy_data["group"][0]  # First letter
        max_travel = self.knowledge.MAX_TRAVEL_DISTANCES[
            "sprinklered" if sprinklered else "non_sprinklered"
        ].get(group, 200)
        
        return {
            "area": area,
            "occupancy": occupancy,
            "occupancy_group": occupancy_data["group"],
            "load_factor": occupancy_data["load_factor"],
            "occupant_load": occupant_load,
            "required_exits": required_exits,
            "exit_width": {
                "stairs": round(stair_width),
                "level": round(level_width),
                "minimum_door": 32
            },
            "max_travel_distance": max_travel,
            "sprinklered": sprinklered,
            "code_references": [
                "IBC Section 1004 - Occupant Load",
                "IBC Section 1005 - Egress Width", 
                "IBC Section 1006 - Number of Exits",
                "IBC Table 1017.2 - Travel Distance"
            ]
        }
    
    def _calculate_parking_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parking calculations including ADA"""
        area = params.get("area", 0)
        occupancy = params.get("occupancy", "business")
        
        # Parking ratios (per 1000 sq ft)
        ratios = {
            "business": 3.3,  # 1 per 300 sq ft
            "retail": 5.0,    # 1 per 200 sq ft
            "restaurant": 10.0,  # 1 per 100 sq ft
            "warehouse": 0.5   # 1 per 2000 sq ft
        }
        
        ratio = ratios.get(occupancy, 4.0)
        required_spaces = math.ceil(area * ratio / 1000)
        
        # ADA spaces
        ada_required = 0
        for threshold, count in [(1, 1), (25, 1), (50, 2), (75, 3), (100, 4), 
                                 (150, 5), (200, 6), (300, 7), (400, 8), (500, 9)]:
            if required_spaces >= threshold:
                ada_required = count
        
        # Van accessible (1 per 6 ADA spaces)
        van_required = max(1, math.ceil(ada_required / 6))
        
        return {
            "area": area,
            "occupancy": occupancy,
            "parking_ratio": f"{ratio} per 1000 sq ft",
            "standard_spaces": required_spaces - ada_required,
            "ada_spaces": ada_required - van_required,
            "van_spaces": van_required,
            "total_spaces": required_spaces,
            "dimensions": {
                "standard": "9' x 18'",
                "ada": "8' x 18' + 5' aisle",
                "van": "8' x 18' + 8' aisle"
            },
            "code_reference": "IBC Chapter 11 / ADA Standards"
        }
    
    def _check_ada_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """ADA compliance checking"""
        element = params.get("element", "general")
        
        ada = self.knowledge.ADA_REQUIREMENTS
        
        if element == "door":
            return {
                "element": "door",
                "requirements": ada["doors"],
                "code_reference": "ADA Section 404"
            }
        elif element == "corridor":
            return {
                "element": "corridor", 
                "requirements": ada["corridors"],
                "code_reference": "ADA Section 403"
            }
        elif element == "ramp":
            return {
                "element": "ramp",
                "requirements": ada["ramps"],
                "code_reference": "ADA Section 405"
            }
        elif element == "parking":
            return {
                "element": "parking",
                "requirements": ada["parking"],
                "code_reference": "ADA Section 502"
            }
        else:
            return {
                "element": "general",
                "key_requirements": {
                    "clear_width": "36 inches minimum",
                    "turning_space": "60 inch diameter",
                    "reach_ranges": "15-48 inches",
                    "door_width": "32 inches clear"
                },
                "code_reference": "2010 ADA Standards"
            }


class IntelligentAIService:
    """AI Service with intelligent document understanding and natural responses"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Intelligent AI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.analyzer = DocumentIntelligence()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Tool definitions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_blueprint_visually",
                    "description": "Visually analyze blueprint to extract all information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string"}
                        },
                        "required": ["image_url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_measurements",
                    "description": "Extract measurements from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_requirements",
                    "description": "Calculate code requirements for any building system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system": {
                                "type": "string",
                                "enum": ["sprinklers", "electrical", "hvac", "plumbing", 
                                        "egress", "parking", "accessibility"]
                            },
                            "parameters": {"type": "object"}
                        },
                        "required": ["system", "parameters"]
                    }
                }
            }
        ]
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process queries with intelligent understanding"""
        try:
            logger.info(f"🤖 Processing query for document {document_id}: {prompt[:50]}...")
            
            # Classify question intent
            intent = QuestionIntent.classify(prompt)
            logger.info(f"📊 Question intent: {intent}")
            
            # Load document data
            document_text = ""
            image_url = None
            
            # Try to load text
            try:
                context_blob = f"{document_id}_context.txt"
                document_text = await storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=context_blob
                )
                logger.info(f"✅ Loaded text: {len(document_text)} chars")
            except Exception as e:
                logger.warning(f"Text not available: {e}")
            
            # Try to load image
            try:
                image_blob = f"{document_id}_page_1.png"
                image_bytes = await storage_service.download_blob_as_bytes(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=image_blob
                )
                image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                logger.info(f"✅ Loaded image: {len(image_bytes)} bytes")
            except Exception as e:
                logger.warning(f"Image not available: {e}")
            
            # Check if we have any data
            if not document_text and not image_url:
                return "I cannot access the document. Please ensure it has been uploaded and processed."
            
            # Process based on intent
            result = await self._process_with_intent(
                prompt=prompt,
                intent=intent,
                document_text=document_text,
                image_url=image_url,
                document_id=document_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}"
    
    async def _process_with_intent(self, prompt: str, intent: str, 
                                    document_text: str, image_url: str, 
                                    document_id: str) -> str:
        """Process based on question intent"""
        try:
            # Create system message based on intent
            if intent == QuestionIntent.GENERAL_OVERVIEW:
                system_content = """You are analyzing a construction document. When asked for an overview, use the analyze_blueprint_visually tool to examine the document, then provide a clear, conversational overview of what you see. Include:
- Document type and purpose
- Key spaces and their uses
- Overall size and scope
- Notable features or systems
Keep it informative but accessible."""
            
            elif intent == QuestionIntent.SIMPLE_INFO:
                system_content = """You are looking for specific project information. ALWAYS use the analyze_blueprint_visually tool FIRST to examine the document, then find and provide the requested information directly and concisely. Look for:
- Project addresses, names, dates
- People involved (owner, architect, etc.)
- Drawing numbers and details
Just give the specific answer requested."""
            
            elif intent == QuestionIntent.COUNTING:
                system_content = """You need to count specific items in the blueprint. ALWAYS use the analyze_blueprint_visually tool to examine the document, then:
- Identify and count the requested items
- Be precise with your count
- Note what you counted and where
Provide the exact count with brief explanation."""
            
            elif intent == QuestionIntent.CALCULATION:
                system_content = """You need to perform technical calculations. First use the analyze_blueprint_visually tool to get the document information, then use the calculate_requirements tool to:
- Calculate based on actual document data
- Apply relevant building codes
- Show your work briefly
- Provide practical recommendations"""
            
            elif intent == QuestionIntent.CODE_COMPLIANCE:
                system_content = """You need to check code compliance. First use the analyze_blueprint_visually tool to examine the document, then analyze against:
- Relevant building codes (IBC, NFPA, ADA, etc.)
- Specific requirements for the space type
- Safety and accessibility standards
Provide clear compliance status with specifics."""
            
            else:  # SPECIFIC_TECHNICAL
                system_content = """You are answering a specific technical question. Use the analyze_blueprint_visually tool if visual information would help, then provide:
- Direct answer to the question
- Relevant calculations or code references
- Practical considerations
Be thorough but focused on what was asked."""
            
            # Add base capabilities
            system_content += """

You have access to these tools:
- analyze_blueprint_visually: Use this to examine blueprints and extract all information
- extract_measurements: Extract measurements from text
- calculate_requirements: Calculate code requirements for building systems

CRITICAL INSTRUCTIONS:
1. For ANY question about the document content (address, what's shown, counting items, etc.), you MUST use the analyze_blueprint_visually tool FIRST
2. You have the image_url in the conversation - use it with the tool
3. NEVER say "I can't access the image" - you CAN access it through the analyze_blueprint_visually tool
4. This is about a SPECIFIC document the user uploaded - always analyze it

Remember: The user uploaded a blueprint and wants information about IT specifically."""
            
            messages = [{"role": "system", "content": system_content}]
            
            # Build user message
            user_message = {"role": "user", "content": []}
            
            if image_url:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"}
                })
            
            # Add context and question
            if image_url:
                context = f"""Document ID: {document_id}
{f'Document text (may be incomplete): {document_text[:1000]}...' if document_text else 'No text extraction available'}

The user has uploaded a blueprint image that you can analyze.

Question: {prompt}

IMPORTANT: Use the analyze_blueprint_visually tool with the provided image_url to examine this specific blueprint and answer the question based on what you find."""
            else:
                context = f"""Document ID: {document_id}
Document text: {document_text[:2000] if document_text else 'No text available'}...

Question: {prompt}

Note: No image is available, work with the text information provided."""
            
            user_message["content"].append({"type": "text", "text": context})
            messages.append(user_message)
            
            # Get response
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=2000,
                    temperature=0.3
                )
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls
            if assistant_message.tool_calls:
                messages.append(assistant_message)
                
                for tool_call in assistant_message.tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)
                    })
                
                # Get final response
                final_response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.3
                    )
                )
                
                return final_response.choices[0].message.content
            else:
                return assistant_message.content
                
        except Exception as e:
            logger.error(f"Intent processing error: {e}", exc_info=True)
            return f"Error processing {intent} question: {str(e)}"
    
    async def _execute_tool(self, tool_call) -> Dict[str, Any]:
        """Execute tool calls"""
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            logger.debug(f"Executing tool: {function_name}")
            
            if function_name == "analyze_blueprint_visually":
                return await self.analyzer.analyze_blueprint_visually(arguments["image_url"])
            elif function_name == "extract_measurements":
                return self.analyzer.extract_measurements(arguments["text"])
            elif function_name == "calculate_requirements":
                return self.analyzer.calculate_code_requirements(
                    arguments["system"],
                    arguments["parameters"]
                )
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Create aliases for backward compatibility
AIService = IntelligentAIService
ProfessionalAIService = IntelligentAIService
EnhancedAIService = IntelligentAIService
