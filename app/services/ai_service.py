# app/services/ai_service.py - COMPREHENSIVE MULTI-TRADE AI WITH DEEP UNDERSTANDING

import asyncio
import base64
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# OpenAI imports with safety check
try:
    from openai import OpenAI, APIError
    from openai.types.chat import ChatCompletionToolParam
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


class UniversalBuildingKnowledge:
    """Comprehensive knowledge base for ALL building trades and codes"""
    
    # ARCHITECTURAL KNOWLEDGE
    SPACE_TYPES = {
        "office": {"load_factor": 150, "min_ceiling": 9, "lighting_fc": 50},
        "conference": {"load_factor": 15, "min_ceiling": 9, "lighting_fc": 30},
        "restroom": {"load_factor": 50, "min_ceiling": 8, "lighting_fc": 20},
        "corridor": {"load_factor": 100, "min_ceiling": 8, "lighting_fc": 10},
        "storage": {"load_factor": 300, "min_ceiling": 8, "lighting_fc": 10},
        "mechanical": {"load_factor": 300, "min_ceiling": 8, "lighting_fc": 30},
        "retail": {"load_factor": 30, "min_ceiling": 10, "lighting_fc": 50},
        "restaurant": {"load_factor": 15, "min_ceiling": 10, "lighting_fc": 50},
        "kitchen": {"load_factor": 100, "min_ceiling": 10, "lighting_fc": 70},
        "lobby": {"load_factor": 40, "min_ceiling": 12, "lighting_fc": 20}
    }
    
    # STRUCTURAL KNOWLEDGE
    STRUCTURAL_SYSTEMS = {
        "steel_frame": {
            "typical_bay": "25-30 ft",
            "max_bay": "45 ft",
            "floor_depth": "24-36 in",
            "fireproofing": "1-3 hour rating"
        },
        "concrete_frame": {
            "typical_bay": "20-25 ft",
            "max_bay": "35 ft",
            "slab_thickness": "8-12 in",
            "column_size": "18-36 in"
        },
        "wood_frame": {
            "typical_span": "12-16 ft",
            "max_span": "24 ft",
            "joist_spacing": "16-24 in oc",
            "load_capacity": "40-50 psf"
        },
        "masonry": {
            "wall_thickness": "8-16 in",
            "max_height": "35 ft",
            "reinforcement": "#5 @ 32 in oc",
            "grouting": "partial or full"
        }
    }
    
    # MECHANICAL (HVAC) KNOWLEDGE
    HVAC_REQUIREMENTS = {
        "office": {
            "cfm_per_person": 20,
            "cfm_per_sqft": 0.06,
            "tons_per_sqft": 350,  # 1 ton per 350 sqft
            "duct_velocity": 1000,  # fpm
            "vav_min": 0.3  # 30% minimum
        },
        "retail": {
            "cfm_per_person": 15,
            "cfm_per_sqft": 0.12,
            "tons_per_sqft": 250,
            "duct_velocity": 1200,
            "exhaust_cfm": 0.5  # per sqft for general
        },
        "restaurant": {
            "cfm_per_person": 20,
            "cfm_per_sqft": 0.18,
            "tons_per_sqft": 200,
            "kitchen_exhaust": 100,  # cfm per linear ft of hood
            "makeup_air": 0.8  # 80% of exhaust
        },
        "restroom": {
            "exhaust_cfm": 75,  # per toilet/urinal
            "ach": 10,  # air changes per hour
            "negative_pressure": 0.05  # inches w.c.
        }
    }
    
    # ELECTRICAL KNOWLEDGE
    ELECTRICAL_LOADS = {
        "lighting": {
            "office": 1.3,  # watts/sqft
            "retail": 1.5,
            "warehouse": 0.6,
            "parking": 0.2,
            "emergency": 0.2  # additional
        },
        "receptacles": {
            "office": 1.5,  # watts/sqft
            "retail": 1.0,
            "industrial": 2.0,
            "spacing": 12,  # feet max
            "kitchen": 3.0
        },
        "equipment": {
            "elevator": "30-50 HP",
            "escalator": "15-25 HP",
            "hvac_rtu": "varies",
            "fire_pump": "50-150 HP"
        },
        "panel_sizing": {
            "standard": [100, 200, 400, 600, 800, 1200, 1600, 2000],
            "voltage": [120, 208, 277, 480],
            "phases": [1, 3]
        }
    }
    
    # PLUMBING KNOWLEDGE
    PLUMBING_FIXTURES = {
        "water_closets": {
            "office": 1.25,  # per 1000 sqft
            "retail": 0.5,
            "restaurant": 2.0,
            "assembly": 3.0,
            "gpf": 1.6  # gallons per flush
        },
        "lavatories": {
            "ratio_to_wc": 1.0,  # 1:1 with water closets
            "gpm": 0.5,  # gallons per minute
            "ada_height": 34  # inches
        },
        "drinking_fountains": {
            "per_floor": 1,
            "per_1000_sqft": 0.1,
            "ada_dual": True  # high/low required
        },
        "hot_water": {
            "office": 1.0,  # gallons per person per day
            "restaurant": 2.5,
            "recovery": 4.0,  # gallons per hour per fixture
            "temp": 120  # degrees F
        },
        "drainage": {
            "slope": 0.25,  # inches per foot min
            "vent_distance": 6,  # feet max
            "stack_size": "varies",
            "roof_drain": 1.5  # per 1000 sqft roof
        }
    }
    
    # FIRE PROTECTION KNOWLEDGE
    FIRE_PROTECTION = {
        "sprinklers": {
            "light_hazard": {"spacing": 225, "density": 0.1},
            "ordinary_1": {"spacing": 130, "density": 0.15},
            "ordinary_2": {"spacing": 130, "density": 0.2},
            "extra_hazard": {"spacing": 100, "density": 0.3}
        },
        "fire_alarm": {
            "smoke_detector": 30,  # feet spacing
            "pull_station": 200,  # feet travel
            "horn_strobe": 15,  # candela
            "battery_backup": 24  # hours
        },
        "extinguishers": {
            "class_a": 75,  # feet travel
            "class_b": 50,
            "class_c": 75,
            "class_k": 30,  # kitchen
            "size": "2A:10BC"  # typical
        },
        "fire_ratings": {
            "corridors": 1,  # hour
            "stairwells": 2,
            "electrical_rooms": 2,
            "mechanical_rooms": 1,
            "exterior_walls": "varies"
        }
    }
    
    # SITE/CIVIL KNOWLEDGE
    SITE_REQUIREMENTS = {
        "parking": {
            "standard": {"width": 9, "length": 18},
            "compact": {"width": 8, "length": 16},
            "ada": {"width": 8, "aisle": 5},
            "ratio": {
                "office": 4.0,  # per 1000 sqft
                "retail": 5.0,
                "restaurant": 10.0
            },
            "ada_count": {
                1: 1, 25: 1, 50: 2, 75: 3, 100: 4,
                150: 5, 200: 6, 300: 7, 400: 8, 500: 9
            }
        },
        "stormwater": {
            "retention": 1.0,  # inches typical
            "bio_swale": 0.5,  # cfs per acre
            "pervious": 0.3,  # ratio target
            "pipe_slope": 0.5  # percent min
        },
        "utilities": {
            "water": {"depth": 48, "pressure": "40-80 psi"},
            "sewer": {"depth": 60, "slope": 1.0},
            "gas": {"depth": 36, "pressure": "varies"},
            "electric": {"depth": 36, "primary": 480},
            "telecom": {"depth": 24, "conduits": 4}
        }
    }
    
    # ENERGY/SUSTAINABILITY
    ENERGY_REQUIREMENTS = {
        "envelope": {
            "wall_r": {"climate_1": 13, "climate_4": 19, "climate_6": 25},
            "roof_r": {"climate_1": 25, "climate_4": 30, "climate_6": 38},
            "window_u": {"climate_1": 0.50, "climate_4": 0.40, "climate_6": 0.35},
            "shgc": {"climate_1": 0.25, "climate_4": 0.40, "climate_6": 0.45}
        },
        "lighting_power": {
            "lpd": {  # watts per sqft
                "office": 0.82,
                "retail": 1.26,
                "warehouse": 0.66,
                "parking": 0.15
            },
            "controls": ["occupancy", "daylight", "dimming", "scheduling"]
        },
        "renewable": {
            "solar_pv": 5.0,  # watts per sqft roof
            "solar_thermal": 0.75,  # efficiency
            "geothermal": 4.0  # COP
        }
    }
    
    # ACCESSIBILITY (ADA)
    ADA_REQUIREMENTS = {
        "general": {
            "clear_width": 36,  # inches
            "door_width": 32,
            "turning_radius": 60,
            "reach_height": {"min": 15, "max": 48},
            "counter_height": 36
        },
        "ramps": {
            "slope": 8.33,  # percent (1:12)
            "landing": 60,  # inches
            "handrail": {"min": 34, "max": 38},
            "width": 36
        },
        "parking": {
            "width": 96,  # inches (8 feet)
            "aisle": 60,  # access aisle
            "van_aisle": 96,
            "slope": 2.0  # percent max
        },
        "restroom": {
            "stall": {"width": 60, "depth": 59},
            "grab_bars": {"side": 42, "rear": 36},
            "lavatory": {"height": 34, "knee": 27},
            "accessories": {"height": 48, "reach": 40}
        }
    }


class ComprehensiveDocumentAnalyzer:
    """Analyzes any construction document with deep multi-trade understanding"""
    
    def __init__(self):
        self.knowledge = UniversalBuildingKnowledge()
        self.understanding = {
            "document_type": None,
            "identified_elements": {},
            "measurements": {},
            "systems": set(),
            "spaces": [],
            "trade_specific_items": {},
            "code_items": []
        }
    
    async def deep_visual_analysis(self, image_url: str) -> Dict[str, Any]:
        """Comprehensive visual analysis for ANY trade or system"""
        logger.debug("🔍 Performing deep multi-trade visual analysis")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are analyzing a construction document. Extract EVERYTHING you see, including:

GENERAL:
- Document type, title, project name, sheet number
- Scale, north arrow, grid lines
- Overall dimensions and areas
- All rooms/spaces with names and dimensions

ARCHITECTURAL:
- Doors (count, sizes, types, swings)
- Windows (count, sizes, types)
- Walls (types, thickness)
- Finishes (floor, ceiling, wall)
- Furniture, fixtures, equipment
- Stairs, elevators, ramps

STRUCTURAL:
- Columns (size, spacing, grid)
- Beams (size, type)
- Slabs (thickness, type)
- Foundations (type, size)
- Reinforcement callouts
- Load information

MECHANICAL (HVAC):
- Ductwork (sizes, CFM)
- Diffusers/grilles (count, sizes)
- Equipment (RTUs, VAVs, fans)
- Piping (if shown)
- Thermostat locations

ELECTRICAL:
- Panels (location, size)
- Circuits (home runs)
- Receptacles (count, type)
- Lighting (fixtures, switches)
- Special systems (data, fire alarm)
- Conduit runs

PLUMBING:
- Fixtures (count, type)
- Piping (sizes, materials)
- Drains (floor, roof)
- Water heaters
- Cleanouts, vents

FIRE PROTECTION:
- Sprinkler heads (count, type)
- Fire alarm devices
- Extinguisher locations
- Exit signs
- Fire ratings

SITE/CIVIL:
- Parking (spaces, dimensions)
- Utilities (water, sewer, gas, electric)
- Grading/drainage
- Landscaping
- Sidewalks/paving

Count EVERYTHING. Note ALL text, dimensions, and callouts. Be extremely thorough."""
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                                {"type": "text", "text": "Extract all information from this construction document."}
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
            )
            
            return {
                "success": True,
                "extracted_data": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Visual analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_all_measurements(self, text: str) -> Dict[str, Any]:
        """Extract measurements for ALL trades"""
        patterns = {
            # Dimensions
            'room_dims': r'(\d+)[\'\-\s]*(\d+)?"?\s*[xX×]\s*(\d+)[\'\-\s]*(\d+)?"?',
            'linear_dims': r'(\d+)[\'\-\s]*(\d+)?"?\s*(?:LF|lf|linear)',
            'area_sqft': r'(\d+[\d,]*\.?\d*)\s*(?:SF|sq\.?\s*ft\.?|square feet)',
            'height': r'(\d+)[\'\-\s]*(\d+)?"?\s*(?:AFF|aff|high|height|clg)',
            
            # Structural
            'beam_size': r'([WwSsCc]\d+[xX×]\d+(?:\.\d+)?)',
            'column_size': r'(\d+)"\s*[xX×]\s*(\d+)"\s*(?:COL|col|column)',
            'slab_thick': r'(\d+)"\s*(?:SLAB|slab|thick)',
            'rebar': r'#(\d+)\s*@\s*(\d+)"\s*(?:OC|oc|o\.c\.)',
            
            # Mechanical
            'cfm': r'(\d+)\s*(?:CFM|cfm)',
            'duct_size': r'(\d+)"\s*[xX×]\s*(\d+)"\s*(?:DUCT|duct|supply|return)',
            'tons': r'(\d+\.?\d*)\s*(?:TON|ton|tons)',
            'pipe_size': r'(\d+\.?\d*)"\s*(?:PIPE|pipe|dia\.?)',
            
            # Electrical
            'voltage': r'(\d+)\s*(?:V|volt|volts)',
            'amperage': r'(\d+)\s*(?:A|amp|amps|ampere)',
            'kw': r'(\d+\.?\d*)\s*(?:KW|kw|kilowatt)',
            'circuit': r'(\d+)\s*(?:CKT|ckt|circuit)',
            'conduit': r'(\d+\.?\d*)"\s*(?:C|EMT|RGS|conduit)',
            
            # Plumbing
            'gpm': r'(\d+\.?\d*)\s*(?:GPM|gpm)',
            'fixture_units': r'(\d+\.?\d*)\s*(?:FU|fu|fixture units)',
            'drain_size': r'(\d+)"\s*(?:DRAIN|drain|waste)',
            
            # Fire Protection
            'sprinkler_spacing': r'(\d+)[\'\-\s]*(\d+)?"?\s*(?:OC|oc|o\.c\.)\s*(?:SPKLR|sprinkler)',
            'fire_rating': r'(\d+\.?\d*)\s*(?:HR|hr|hour)\s*(?:RATED|rated|rating)',
            
            # Counts
            'door_count': r'(?:DOOR|door|DR|dr)[\s\-]*(\d+)',
            'fixture_count': r'(\d+)\s*(?:WC|LAV|UR|DF|wc|lav)',
            'device_count': r'(\d+)\s*(?:devices?|heads?|units?)'
        }
        
        results = {}
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                results[pattern_name] = matches
        
        return results
    
    def analyze_any_system(self, system_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ANY building system with appropriate calculations"""
        
        if system_type == "hvac_sizing":
            return self._calculate_hvac_requirements(parameters)
        elif system_type == "electrical_load":
            return self._calculate_electrical_loads(parameters)
        elif system_type == "plumbing_fixtures":
            return self._calculate_plumbing_requirements(parameters)
        elif system_type == "structural_loads":
            return self._calculate_structural_requirements(parameters)
        elif system_type == "parking":
            return self._calculate_parking_requirements(parameters)
        elif system_type == "egress":
            return self._calculate_egress_requirements(parameters)
        elif system_type == "fire_protection":
            return self._calculate_fire_protection(parameters)
        elif system_type == "accessibility":
            return self._check_ada_compliance(parameters)
        elif system_type == "energy":
            return self._calculate_energy_requirements(parameters)
        else:
            return self._general_code_check(system_type, parameters)
    
    def _calculate_hvac_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate HVAC requirements for any space"""
        space_type = params.get("space_type", "office")
        area = params.get("area", 0)
        occupants = params.get("occupants", 0)
        
        reqs = self.knowledge.HVAC_REQUIREMENTS.get(space_type, self.knowledge.HVAC_REQUIREMENTS["office"])
        
        # Ventilation calculations
        cfm_people = occupants * reqs["cfm_per_person"]
        cfm_area = area * reqs["cfm_per_sqft"]
        total_cfm = cfm_people + cfm_area
        
        # Cooling calculations
        tons_required = area / reqs["tons_per_sqft"]
        
        # Duct sizing (simplified)
        main_duct_area = total_cfm / reqs["duct_velocity"]
        
        return {
            "space_type": space_type,
            "area": area,
            "ventilation": {
                "cfm_per_person": reqs["cfm_per_person"],
                "cfm_per_area": reqs["cfm_per_sqft"],
                "people_cfm": cfm_people,
                "area_cfm": cfm_area,
                "total_cfm": total_cfm,
                "code_ref": "ASHRAE 62.1"
            },
            "cooling": {
                "tons_required": round(tons_required, 1),
                "btuh": tons_required * 12000,
                "rule_of_thumb": f"{reqs['tons_per_sqft']} sqft/ton"
            },
            "duct_sizing": {
                "velocity_fpm": reqs["duct_velocity"],
                "main_duct_area_sqin": round(main_duct_area * 144),
                "typical_size": self._suggest_duct_size(main_duct_area)
            },
            "notes": [
                "Final design requires heat load calculations",
                "Consider diversity factors for VAV systems",
                "Account for building envelope and orientation"
            ]
        }
    
    def _calculate_electrical_loads(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate electrical loads for any space"""
        space_type = params.get("space_type", "office")
        area = params.get("area", 0)
        
        lighting = self.knowledge.ELECTRICAL_LOADS["lighting"].get(space_type, 1.3)
        receptacle = self.knowledge.ELECTRICAL_LOADS["receptacles"].get(space_type, 1.5)
        
        lighting_load = area * lighting
        receptacle_load = area * receptacle
        total_load = lighting_load + receptacle_load
        
        # Add demand factors per NEC
        lighting_demand = lighting_load * 1.0  # 100% for first 3000W
        receptacle_demand = receptacle_load * 0.5  # 50% for receptacles
        
        total_demand = lighting_demand + receptacle_demand
        
        # Size panel
        voltage = 208  # typical commercial
        phases = 3
        amps = total_demand / (voltage * 1.732)  # 1.732 for 3-phase
        
        panel_size = next((size for size in self.knowledge.ELECTRICAL_LOADS["panel_sizing"]["standard"] 
                          if size >= amps * 1.25), 400)
        
        return {
            "space_type": space_type,
            "area": area,
            "lighting": {
                "watts_per_sqft": lighting,
                "total_watts": lighting_load,
                "demand_watts": lighting_demand
            },
            "receptacles": {
                "watts_per_sqft": receptacle,
                "total_watts": receptacle_load,
                "demand_watts": receptacle_demand
            },
            "total": {
                "connected_load": total_load,
                "demand_load": total_demand,
                "amps": round(amps, 1),
                "panel_size": panel_size
            },
            "code_references": [
                "NEC Article 220 - Load Calculations",
                "NEC Table 220.12 - Lighting Load Demand",
                "NEC Section 220.44 - Receptacle Loads"
            ]
        }
    
    def _calculate_plumbing_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate plumbing fixture requirements"""
        space_type = params.get("space_type", "office")
        area = params.get("area", 0)
        occupants = params.get("occupants", 0)
        
        fixtures = self.knowledge.PLUMBING_FIXTURES
        
        # Calculate required fixtures
        wc_per_1000 = fixtures["water_closets"].get(space_type, 1.25)
        required_wc = max(2, math.ceil(area * wc_per_1000 / 1000))
        required_lav = required_wc  # 1:1 ratio typical
        
        # Drinking fountains
        df_required = max(1, math.ceil(area * fixtures["drinking_fountains"]["per_1000_sqft"] / 1000))
        
        # Hot water
        hot_water_gpd = occupants * fixtures["hot_water"].get(space_type, 1.0)
        
        return {
            "fixtures_required": {
                "water_closets": required_wc,
                "lavatories": required_lav,
                "drinking_fountains": df_required,
                "calculation": f"Based on {area} sqft and {space_type} use"
            },
            "hot_water": {
                "gallons_per_day": hot_water_gpd,
                "recovery_gph": required_wc * fixtures["hot_water"]["recovery"],
                "temperature": fixtures["hot_water"]["temp"]
            },
            "drainage": {
                "min_slope": f"{fixtures['drainage']['slope']} in/ft",
                "roof_drains": math.ceil(area * fixtures["drainage"]["roof_drain"] / 1000)
            },
            "code_references": [
                "IPC Table 403.1 - Minimum Fixtures",
                "IPC Section 604 - Water Distribution",
                "IPC Section 704 - Drainage Piping"
            ]
        }
    
    def _suggest_duct_size(self, area_sqft: float) -> str:
        """Suggest rectangular duct size based on area"""
        area_sqin = area_sqft * 144
        
        # Common aspect ratios
        if area_sqin < 144:
            return "12x12"
        elif area_sqin < 288:
            return "24x12"
        elif area_sqin < 432:
            return "24x18"
        elif area_sqin < 576:
            return "24x24"
        else:
            return "30x24 or larger"
    
    def _calculate_structural_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic structural calculations"""
        return {
            "loads": {
                "dead_load": "15-25 psf typical",
                "live_load": "50-100 psf (office/retail)",
                "total": "65-125 psf"
            },
            "note": "Requires structural engineer for final design"
        }
    
    def _calculate_parking_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate parking requirements"""
        space_type = params.get("space_type", "office")
        area = params.get("area", 0)
        
        parking = self.knowledge.SITE_REQUIREMENTS["parking"]
        ratio = parking["ratio"].get(space_type, 4.0)
        required_spaces = math.ceil(area * ratio / 1000)
        
        # ADA spaces
        ada_required = 0
        for threshold, count in parking["ada_count"].items():
            if required_spaces >= threshold:
                ada_required = count
        
        return {
            "required_spaces": required_spaces,
            "ada_spaces": ada_required,
            "van_accessible": max(1, ada_required // 8),
            "calculation": f"{ratio} spaces per 1000 sqft",
            "dimensions": parking["standard"]
        }
    
    def _calculate_egress_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate egress requirements - already comprehensive"""
        space_type = params.get("space_type", "office")
        area = params.get("area", 0)
        
        # Get occupant load factor
        load_factor = self.knowledge.SPACE_TYPES.get(space_type, {"load_factor": 150})["load_factor"]
        occupant_load = math.ceil(area / load_factor)
        
        # Required exits
        if occupant_load <= 49:
            required_exits = 1
        elif occupant_load <= 500:
            required_exits = 2
        else:
            required_exits = 3
        
        return {
            "occupant_load": occupant_load,
            "load_factor": load_factor,
            "required_exits": required_exits,
            "exit_width": occupant_load * 0.2,  # inches
            "travel_distance": "See IBC Table 1017.2"
        }
    
    def _calculate_fire_protection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fire protection requirements"""
        space_type = params.get("space_type", "office")
        area = params.get("area", 0)
        
        # Determine hazard
        if space_type in ["office", "residential"]:
            hazard = "light_hazard"
        else:
            hazard = "ordinary_1"
        
        sprinkler = self.knowledge.FIRE_PROTECTION["sprinklers"][hazard]
        heads_required = math.ceil(area / sprinkler["spacing"])
        
        # Fire alarm
        smoke_detectors = math.ceil(area / (30 * 30))  # 30ft spacing
        pull_stations = math.ceil(math.sqrt(area) * 4 / 200)  # perimeter / 200
        
        return {
            "sprinklers": {
                "hazard": hazard,
                "heads_required": heads_required,
                "spacing": sprinkler["spacing"],
                "density": sprinkler["density"]
            },
            "fire_alarm": {
                "smoke_detectors": smoke_detectors,
                "pull_stations": max(2, pull_stations),
                "notification": "horn/strobes per NFPA 72"
            },
            "extinguishers": math.ceil(area / (75 * 75))  # 75ft travel
        }
    
    def _check_ada_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check ADA compliance for various elements"""
        element = params.get("element", "general")
        
        ada = self.knowledge.ADA_REQUIREMENTS
        
        if element == "door":
            return {
                "min_clear_width": ada["general"]["clear_width"],
                "threshold": "1/2 inch max",
                "maneuvering_clearance": "18 inches pull side",
                "hardware_height": "34-48 inches AFF"
            }
        elif element == "ramp":
            return ada["ramps"]
        elif element == "parking":
            return ada["parking"]
        elif element == "restroom":
            return ada["restroom"]
        else:
            return ada["general"]
    
    def _calculate_energy_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate energy code requirements"""
        climate_zone = params.get("climate_zone", 4)
        area = params.get("area", 0)
        space_type = params.get("space_type", "office")
        
        energy = self.knowledge.ENERGY_REQUIREMENTS
        
        return {
            "envelope": {
                "wall_r_value": energy["envelope"]["wall_r"][f"climate_{climate_zone}"],
                "roof_r_value": energy["envelope"]["roof_r"][f"climate_{climate_zone}"],
                "window_u_factor": energy["envelope"]["window_u"][f"climate_{climate_zone}"]
            },
            "lighting": {
                "max_lpd": energy["lighting_power"]["lpd"].get(space_type, 1.0),
                "total_watts": area * energy["lighting_power"]["lpd"].get(space_type, 1.0),
                "controls_required": energy["lighting_power"]["controls"]
            },
            "code": "IECC/ASHRAE 90.1"
        }
    
    def _general_code_check(self, system: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """General code compliance check for any system"""
        return {
            "system": system,
            "parameters": params,
            "note": "Consult applicable codes for specific requirements",
            "common_codes": [
                "IBC - International Building Code",
                "NEC - National Electrical Code", 
                "IPC - International Plumbing Code",
                "IMC - International Mechanical Code",
                "NFPA - Fire Protection Standards",
                "ADA - Accessibility Guidelines"
            ]
        }


class EnhancedAIService:
    """AI Service with comprehensive multi-trade understanding"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ AI client initialized with comprehensive capabilities")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.analyzer = ComprehensiveDocumentAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Comprehensive tool definitions for ALL trades
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_document_visually",
                    "description": "Comprehensive visual analysis of construction document for ALL trades",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string", "description": "Document image URL"}
                        },
                        "required": ["image_url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_measurements",
                    "description": "Extract all measurements and quantities from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to analyze"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_building_system",
                    "description": "Analyze any building system (HVAC, electrical, plumbing, structural, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_type": {
                                "type": "string",
                                "enum": ["hvac_sizing", "electrical_load", "plumbing_fixtures", 
                                        "structural_loads", "parking", "egress", "fire_protection",
                                        "accessibility", "energy", "general"],
                                "description": "Type of system to analyze"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "System-specific parameters"
                            }
                        },
                        "required": ["system_type", "parameters"]
                    }
                }
            }
        ]
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process ANY construction-related query with deep understanding"""
        try:
            logger.info(f"🤖 Processing comprehensive query for document {document_id}")
            
            # Load document
            document_text = ""
            image_url = None
            
            try:
                context_task = storage_service.download_blob_as_text(
                    container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                    blob_name=f"{document_id}_context.txt"
                )
                document_text = await asyncio.wait_for(context_task, timeout=30.0)
                
                try:
                    image_task = storage_service.download_blob_as_bytes(
                        container_name=self.settings.AZURE_CACHE_CONTAINER_NAME,
                        blob_name=f"{document_id}_page_1.png"
                    )
                    page_bytes = await asyncio.wait_for(image_task, timeout=30.0)
                    page_b64 = base64.b64encode(page_bytes).decode('utf-8')
                    image_url = f"data:image/png;base64,{page_b64}"
                except:
                    logger.info("No image available")
                    
            except Exception as e:
                logger.error(f"Document loading error: {e}")
                return "I'm having trouble accessing the document. Please ensure it's been properly uploaded."
            
            # Process with comprehensive understanding
            result = await self._process_with_deep_understanding(
                prompt=prompt,
                document_text=document_text,
                image_url=image_url,
                document_id=document_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return "I encountered an error. Please try rephrasing your question."
    
    async def _process_with_deep_understanding(self, prompt: str, document_text: str, 
                                               image_url: str = None, document_id: str = None) -> str:
        """Process with comprehensive multi-trade understanding"""
        try:
            # Comprehensive system message
            system_message = {
                "role": "system",
                "content": """You are an AI with COMPREHENSIVE understanding of ALL construction trades and building systems. You have expert-level knowledge in:

TRADES & DISCIPLINES:
- Architecture & Space Planning
- Structural Engineering (steel, concrete, wood, masonry)
- Mechanical/HVAC (sizing, ductwork, equipment)
- Electrical (loads, panels, circuits, lighting)
- Plumbing (fixtures, piping, drainage)
- Fire Protection (sprinklers, alarms, ratings)
- Civil/Site (parking, utilities, drainage)
- Energy & Sustainability
- Accessibility (ADA compliance)

CODES & STANDARDS:
- IBC (International Building Code) - all chapters
- NEC (National Electrical Code)
- IPC (International Plumbing Code)
- IMC (International Mechanical Code)
- NFPA (all relevant standards)
- ADA/ANSI accessibility standards
- ASHRAE standards
- Local codes and amendments

CAPABILITIES:
- Read and interpret ANY construction drawing
- Perform calculations for ANY building system
- Count/quantify ANY element accurately
- Apply appropriate codes for ANY situation
- Provide practical construction guidance
- Identify potential issues or conflicts

APPROACH:
1. Understand exactly what's being asked - could be about any trade or system
2. Use visual analysis if it helps answer the question better
3. Apply relevant codes and standards
4. Perform appropriate calculations
5. Provide clear, helpful answers
6. Include code references when relevant
7. Note when professional review is needed

Answer naturally but thoroughly. You can handle questions about ANYTHING in construction."""
            }
            
            messages = [system_message]
            
            # Build user message
            user_message = {"role": "user", "content": []}
            
            if image_url:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"}
                })
            
            # Context with query
            context = f"""Document: {document_id}
Text from document:
{document_text[:3000]}...

Question: {prompt}

Provide a comprehensive answer using your deep knowledge of all construction trades and systems."""
            
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
                    max_tokens=2500,
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
                        "content": json.dumps(result)
                    })
                
                # Final response
                final_response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=2500,
                        temperature=0.3
                    )
                )
                
                return final_response.choices[0].message.content
            else:
                return assistant_message.content
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return f"Processing error: {str(e)}"
    
    async def _execute_tool(self, tool_call) -> Dict[str, Any]:
        """Execute tool calls for any trade or system"""
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "analyze_document_visually":
                return await self.analyzer.deep_visual_analysis(arguments["image_url"])
            elif function_name == "extract_measurements":
                return self.analyzer.extract_all_measurements(arguments["text"])
            elif function_name == "analyze_building_system":
                return self.analyzer.analyze_any_system(
                    arguments["system_type"],
                    arguments["parameters"]
                )
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            logger.error(f"Tool error: {e}")
            return {"error": str(e)}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Backward compatibility
AIService = EnhancedAIService
ProfessionalAIService = EnhancedAIService
