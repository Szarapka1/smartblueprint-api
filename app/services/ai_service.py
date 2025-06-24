# app/services/ai_service.py - COMPREHENSIVE BLUEPRINT ANALYSIS AI WITH HUMAN-LIKE UNDERSTANDING

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
import numpy as np
from collections import defaultdict

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


class UniversalBlueprintKnowledge:
    """Complete knowledge base for ALL construction disciplines and calculations"""
    
    # ============ COMMON SCALES ============
    COMMON_SCALES = {
        "1/16\" = 1'-0\"": 192,
        "1/8\" = 1'-0\"": 96,
        "3/16\" = 1'-0\"": 64,
        "1/4\" = 1'-0\"": 48,
        "3/8\" = 1'-0\"": 32,
        "1/2\" = 1'-0\"": 24,
        "3/4\" = 1'-0\"": 16,
        "1\" = 1'-0\"": 12,
        "1 1/2\" = 1'-0\"": 8,
        "3\" = 1'-0\"": 4,
        "1:100": 100,
        "1:50": 50,
        "1:20": 20
    }
    
    # ============ DRAWING RECOGNITION ============
    DRAWING_PATTERNS = {
        # Architectural
        "floor_plan": ["FLOOR PLAN", "LEVEL", "PLAN VIEW", "BUILDING PLAN"],
        "reflected_ceiling": ["RCP", "REFLECTED CEILING", "CEILING PLAN"],
        "elevation": ["ELEVATION", "BUILDING ELEVATION", "EXTERIOR VIEW"],
        "section": ["SECTION", "BUILDING SECTION", "CROSS SECTION"],
        "detail": ["DETAIL", "TYPICAL", "ENLARGED", "CONNECTION"],
        "schedule": ["SCHEDULE", "ROOM FINISH", "DOOR", "WINDOW", "PARTITION"],
        
        # Structural
        "foundation": ["FOUNDATION", "FOOTING", "PILE", "GRADE BEAM"],
        "framing": ["FRAMING", "BEAM", "JOIST", "DECK", "SLAB"],
        "reinforcement": ["REBAR", "REINFORCEMENT", "REINFORCING", "BARS"],
        
        # MEP
        "hvac": ["HVAC", "MECHANICAL", "DUCTWORK", "AIR HANDLING"],
        "plumbing": ["PLUMBING", "PIPING", "DRAINAGE", "WATER", "WASTE"],
        "electrical": ["ELECTRICAL", "POWER", "LIGHTING", "PANEL"],
        "fire_protection": ["SPRINKLER", "FIRE PROTECTION", "FP", "STANDPIPE"],
        
        # Civil
        "site": ["SITE PLAN", "PLOT PLAN", "SURVEY"],
        "grading": ["GRADING", "CONTOURS", "ELEVATIONS", "EARTHWORK"],
        "utility": ["UTILITY", "UNDERGROUND", "STORM", "SEWER"],
        
        # Specialty
        "kitchen": ["KITCHEN", "FOOD SERVICE", "EQUIPMENT"],
        "millwork": ["MILLWORK", "CASEWORK", "CABINET"],
        "landscape": ["LANDSCAPE", "PLANTING", "IRRIGATION"]
    }
    
    # ============ VISUAL PATTERN RECOGNITION ============
    VISUAL_SYMBOLS = {
        # Fire Protection
        "sprinkler_pendant": "○ with stem down",
        "sprinkler_upright": "○ with stem up", 
        "sprinkler_sidewall": "○ with stem to side",
        "fire_extinguisher": "FE in triangle",
        "fire_alarm_pull": "F with arrow",
        "smoke_detector": "S in circle",
        "horn_strobe": "H/S in box",
        
        # Plumbing
        "floor_drain": "FD in square",
        "catch_basin": "CB in square",
        "cleanout": "CO in circle",
        "water_closet": "elongated oval",
        "lavatory": "rectangle with bowl",
        "shower": "square with X",
        "water_heater": "WH in circle",
        
        # Electrical
        "duplex_outlet": "⊕ or circle with 2 lines",
        "switch": "S with line",
        "light_fixture": "○ or ⊗",
        "panel": "rectangle with lines",
        "junction_box": "J in square",
        "data_outlet": "D in square",
        
        # HVAC
        "supply_diffuser": "square with X pattern",
        "return_grille": "rectangle with lines",
        "exhaust_fan": "EF in circle",
        "thermostat": "T in square",
        "VAV_box": "rectangle labeled VAV",
        
        # Structural
        "column": "solid square/circle with grid",
        "beam": "dashed lines between columns",
        "shear_wall": "thick solid line",
        "brace": "diagonal line with notation"
    }
    
    # ============ SCHEDULING KNOWLEDGE ============
    SCHEDULE_TYPES = {
        "door_schedule": {
            "columns": ["Mark", "Width", "Height", "Type", "Material", "Fire Rating", "Hardware"],
            "calculations": ["count_by_type", "total_doors", "fire_rated_count"]
        },
        "window_schedule": {
            "columns": ["Mark", "Width", "Height", "Type", "Glazing", "Frame"],
            "calculations": ["total_glazing_area", "count_by_type"]
        },
        "room_finish_schedule": {
            "columns": ["Room #", "Room Name", "Floor", "Base", "Walls", "Ceiling", "Height"],
            "calculations": ["area_by_finish", "paint_quantity", "flooring_quantity"]
        },
        "equipment_schedule": {
            "columns": ["Tag", "Description", "Model", "Power", "Weight", "Mounting"],
            "calculations": ["total_load", "total_weight", "breaker_sizing"]
        },
        "beam_schedule": {
            "columns": ["Mark", "Size", "Length", "Camber", "Connection", "Paint"],
            "calculations": ["total_tonnage", "paint_area", "bolt_count"]
        },
        "column_schedule": {
            "columns": ["Mark", "Size", "Height", "Base Plate", "Fireproofing"],
            "calculations": ["total_tonnage", "fireproofing_area"]
        },
        "lighting_fixture_schedule": {
            "columns": ["Type", "Description", "Lamp", "Voltage", "Mounting"],
            "calculations": ["total_wattage", "circuit_count", "control_zones"]
        }
    }
    
    # ============ ADVANCED CALCULATIONS ============
    
    # Structural Calculations
    STRUCTURAL_CALCS = {
        "steel_weight": {
            "W_shapes": lambda d, bf, tw, tf, L: 3.4 * (d * tw + 2 * bf * tf) * L / 144,
            "HSS_rect": lambda H, B, t, L: 3.4 * 2 * t * (H + B - 2 * t) * L / 144,
            "HSS_round": lambda D, t, L: 3.4 * math.pi * (D - t) * t * L / 144
        },
        "concrete_volume": {
            "slab": lambda L, W, t: L * W * t / 27,  # cubic yards
            "beam": lambda b, h, L: b * h * L / 1728 / 27,
            "column": lambda b, h, H: b * h * H / 1728 / 27,
            "footing": lambda L, W, h: L * W * h / 27
        },
        "rebar_weight": {
            "#3": 0.376, "#4": 0.668, "#5": 1.043, "#6": 1.502,
            "#7": 2.044, "#8": 2.670, "#9": 3.400, "#10": 4.303,
            "#11": 5.313, "#14": 7.650, "#18": 13.600  # lbs per foot
        }
    }
    
    # MEP Calculations
    MEP_CALCULATIONS = {
        "duct_sizing": {
            "velocity_method": lambda CFM, V: math.sqrt(CFM / V * 144 / math.pi) * 2,
            "friction_method": lambda CFM: 1.3 * math.pow(CFM, 0.625),
            "aspect_ratio": lambda area: {
                "options": [(w, area/w) for w in range(6, 97, 2) if area/w <= 96]
            }
        },
        "pipe_sizing": {
            "water_velocity": lambda GPM, D: 0.408 * GPM / (D * D),
            "drainage_capacity": {  # Horizontal drain at 1/4" per foot
                "2": 21, "2.5": 44, "3": 75, "4": 180, "5": 311,
                "6": 538, "8": 1117, "10": 1942, "12": 3088
            },
            "fixture_units": {
                "water_closet": 4, "lavatory": 1, "shower": 2,
                "kitchen_sink": 2, "drinking_fountain": 1, "urinal": 4
            }
        },
        "electrical_load": {
            "voltage_drop": lambda I, L, V: (2 * I * L * 0.866) / (1000 * V) * 100,
            "breaker_size": lambda load: next(b for b in [15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,175,200,225,250,300,350,400] if b >= load * 1.25),
            "conduit_fill": {
                "THHN": {"14": 0.0097, "12": 0.0133, "10": 0.0211, "8": 0.0366, "6": 0.0507, "4": 0.0824, "2": 0.1158, "1": 0.1562, "1/0": 0.1855}
            }
        }
    }
    
    # Fire Protection Calculations  
    FIRE_PROTECTION_CALCS = {
        "sprinkler_density": {
            "light_hazard": {"density": 0.10, "area": 3000, "hose": 100},
            "ordinary_1": {"density": 0.15, "area": 3000, "hose": 250},
            "ordinary_2": {"density": 0.20, "area": 3000, "hose": 250},
            "extra_1": {"density": 0.30, "area": 2500, "hose": 500},
            "extra_2": {"density": 0.40, "area": 2500, "hose": 1000}
        },
        "hydraulic_calculation": {
            "k_factor": lambda Q, P: Q / math.sqrt(P),
            "pressure": lambda Q, K: math.pow(Q / K, 2),
            "friction_loss": lambda C, Q, D, L: 4.52 * math.pow(Q, 1.85) * L / (math.pow(C, 1.85) * math.pow(D, 4.87))
        },
        "water_supply": {
            "storage_tank": lambda area, density, duration: area * density * duration * 1.2,
            "pump_size": lambda Q, H: Q * H * 0.0632 / 0.7  # HP
        }
    }
    
    # ============ CODE REQUIREMENTS ============
    CODE_REQUIREMENTS = {
        "egress": {
            "corridor_width": {"min": 44, "accessible": 48, "hospital": 96},
            "stair_width": {"min": 44, "floor_occupant_load": {
                "0-50": 44, "51-100": 56, "101-200": 68, "201+": 80
            }},
            "door_width": {"min": 32, "accessible": 36, "pair": 48},
            "travel_distance": {
                "sprinklered": {"business": 300, "assembly": 250, "residential": 250},
                "non_sprinklered": {"business": 200, "assembly": 150, "residential": 200}
            }
        },
        "accessibility": {
            "parking": {
                "ratio": {1: 1, 25: 1, 50: 2, 75: 3, 100: 4, 150: 5, 200: 6, 300: 7, 400: 8, 500: 9, 1000: 20},
                "van_accessible": lambda total_accessible: max(1, total_accessible // 6)
            },
            "restroom": {
                "clearances": {"front": 48, "side": 18, "rear": 36},
                "grab_bars": {"side": 42, "rear": 36},
                "fixture_height": {"wc": [17, 19], "lav": [29, 34], "urinal": 17}
            },
            "ramp": {
                "max_slope": 8.33,  # 1:12
                "max_rise": 30,
                "landing": {"length": 60, "width": "ramp_width"}
            }
        },
        "fire_ratings": {
            "occupancy_separation": {
                "B_to_B": 0, "B_to_M": 2, "B_to_R": 1, "B_to_S": 2,
                "R_to_R": 0, "R_to_S": 3, "A_to_B": 2, "A_to_R": 2
            },
            "shaft_enclosures": {"1-3_stories": 1, "4+_stories": 2},
            "exit_enclosures": 2,
            "corridor_walls": {"I-2": 1, "others": 0.5}
        }
    }
    
    # ============ QUANTITY TAKEOFF PATTERNS ============
    TAKEOFF_PATTERNS = {
        "count_items": {
            "doors": ["D[0-9]+", "DR-", "DOOR"],
            "windows": ["W[0-9]+", "WIN-", "WINDOW"],
            "fixtures": ["P[0-9]+", "EF-", "WC", "LAV", "UR"],
            "equipment": ["EQ-", "UNIT-", "AHU", "RTU", "FCU"]
        },
        "measure_linear": {
            "walls": "sum of wall lengths",
            "piping": "centerline lengths",
            "conduit": "point to point + fittings",
            "base": "perimeter - door openings"
        },
        "measure_area": {
            "flooring": "room area",
            "painting": "wall area - openings",
            "insulation": "wall/ceiling area",
            "roofing": "roof area + parapet"
        },
        "measure_volume": {
            "concrete": "slab + beams + columns",
            "excavation": "footprint + overexcavation",
            "hvac": "room volume for air changes"
        }
    }


class HumanLikeBlueprintAnalyzer:
    """Analyzes blueprints with human-like visual understanding and reasoning"""
    
    def __init__(self):
        self.knowledge = UniversalBlueprintKnowledge()
        self.visual_memory = {}
        self.current_drawing = {
            "type": None,
            "scale": None,
            "title_block": {},
            "grid_system": {},
            "elements_found": defaultdict(list),
            "measurements": defaultdict(dict),
            "schedules": {},
            "notes": [],
            "symbols_legend": {},
            "cross_references": []
        }
    
    async def analyze_like_human(self, image_url: str) -> Dict[str, Any]:
        """Analyze blueprint with human-like visual processing"""
        logger.debug("👁️ Performing human-like visual analysis")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are looking at a blueprint EXACTLY as a human expert would. Follow this visual scanning pattern:

STEP 1 - ORIENT YOURSELF (Like opening a physical drawing):
- Look at bottom right corner for title block
- Read: Project name, drawing title, number, scale, date
- Check revision clouds and deltas for latest changes
- Note the drawing orientation (north arrow if present)

STEP 2 - UNDERSTAND THE DRAWING TYPE:
- What am I looking at? (plan, section, elevation, detail, schedule?)
- What discipline? (architectural, structural, MEP, etc.)
- What's the scale? This affects EVERYTHING

STEP 3 - SCAN THE DRAWING SYSTEMATICALLY:
Like reading a book, go left to right, top to bottom:
- Find the grid lines (1,2,3... and A,B,C...)
- Identify major elements (walls, columns, equipment)
- Look for dimension strings
- Find the legend/keynotes
- Spot any schedules or tables

STEP 4 - READ ALL TEXT AND ANNOTATIONS:
- General notes (usually left side or bottom)
- Specific callouts and labels
- Reference bubbles (detail/section marks)
- Equipment tags and room numbers

STEP 5 - COUNT AND MEASURE:
- Count repeated elements (doors, windows, fixtures, etc.)
- Note dimensions between grids
- Calculate areas from dimensions
- Identify typical vs. special conditions

STEP 6 - UNDERSTAND RELATIONSHIPS:
- How do elements connect?
- What references other drawings?
- What's typical vs. unique?
- Where are the critical dimensions?

STEP 7 - EXTRACT SCHEDULING INFO:
If schedules are present:
- Read column headers
- Count entries
- Note sizes, types, quantities
- Calculate totals where applicable

CRITICAL: Read the drawing like you're standing at a plan table:
- Start with the big picture
- Zoom in on details
- Cross-reference between areas
- Build a mental 3D model

Output EVERYTHING you see with specific locations using grid references."""
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                                {"type": "text", "text": "Analyze this blueprint as a human expert would, describing your visual scan pattern and findings."}
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
            )
            
            visual_analysis = response.choices[0].message.content
            
            # Process the visual understanding
            structured_data = self._process_visual_understanding(visual_analysis)
            
            return {
                "success": True,
                "visual_understanding": visual_analysis,
                "structured_data": structured_data,
                "drawing_type": structured_data.get("drawing_type"),
                "elements_counted": structured_data.get("counts"),
                "measurements_found": structured_data.get("measurements"),
                "schedules_extracted": structured_data.get("schedules"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Visual analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_visual_understanding(self, analysis_text: str) -> Dict[str, Any]:
        """Process human-like visual analysis into structured data"""
        structured = {
            "drawing_type": None,
            "project_info": {},
            "scale": None,
            "grid_system": {},
            "counts": defaultdict(int),
            "measurements": defaultdict(dict),
            "schedules": {},
            "areas": {},
            "notes": []
        }
        
        # Extract title block info
        title_patterns = {
            "project": r"(?:project|job)[\s:]+([^\n]+)",
            "drawing_title": r"(?:drawing title|sheet title)[\s:]+([^\n]+)",
            "sheet_number": r"(?:sheet|drawing)[\s#:]+([A-Z0-9\-\.]+)",
            "scale": r"(?:scale)[\s:]+([^\n,]+)",
            "date": r"(?:date)[\s:]+([^\n]+)"
        }
        
        for key, pattern in title_patterns.items():
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                structured["project_info"][key] = match.group(1).strip()
        
        # Identify drawing type
        for dtype, keywords in self.knowledge.DRAWING_PATTERNS.items():
            if any(keyword in analysis_text.upper() for keyword in keywords):
                structured["drawing_type"] = dtype
                break
        
        # Extract grid references
        grid_pattern = r"(?:grid|column line)s?\s+([A-Z0-9,\s\-]+)"
        grid_matches = re.findall(grid_pattern, analysis_text, re.IGNORECASE)
        if grid_matches:
            structured["grid_system"]["references"] = grid_matches
        
        # Extract counts of elements
        count_patterns = [
            (r"(\d+)\s*(?:sprinkler|head)", "sprinklers"),
            (r"(\d+)\s*(?:door)", "doors"),
            (r"(\d+)\s*(?:window)", "windows"),
            (r"(\d+)\s*(?:parking|stall|space)", "parking_spaces"),
            (r"(\d+)\s*(?:column)", "columns"),
            (r"(\d+)\s*(?:beam)", "beams"),
            (r"(\d+)\s*(?:light|fixture)", "light_fixtures"),
            (r"(\d+)\s*(?:outlet|receptacle)", "outlets"),
            (r"(\d+)\s*(?:diffuser|grille)", "air_devices"),
            (r"(\d+)\s*(?:plumbing fixture|WC|lav)", "plumbing_fixtures")
        ]
        
        for pattern, item_type in count_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            if matches:
                structured["counts"][item_type] = sum(int(m) for m in matches)
        
        # Extract areas
        area_patterns = [
            (r"(\d+[,\d]*\.?\d*)\s*(?:sq\.?\s*ft|square feet|SF)", "sqft"),
            (r"(\d+[,\d]*\.?\d*)\s*(?:sq\.?\s*m|square meters|m²|SQ\.?\s*MTS?)", "sqm")
        ]
        
        for pattern, unit in area_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            if matches:
                areas = [float(m.replace(',', '')) for m in matches]
                structured["areas"][unit] = {
                    "values": areas,
                    "total": max(areas) if areas else 0  # Largest is usually total
                }
        
        # Extract dimensions
        dim_pattern = r"(\d+)[\'\-\s]*(\d+)?(?:\s*(\d+\/\d+))?\"?"
        dim_matches = re.findall(dim_pattern, analysis_text)
        if dim_matches:
            structured["measurements"]["dimensions_found"] = len(dim_matches)
        
        # Look for schedules
        schedule_keywords = ["SCHEDULE", "TABLE", "LEGEND"]
        for keyword in schedule_keywords:
            if keyword in analysis_text.upper():
                # Try to extract schedule data
                table_pattern = r"(?:" + keyword + r")[^\n]+\n([^\n]+(?:\n[^\n]+)*)"
                table_match = re.search(table_pattern, analysis_text, re.IGNORECASE)
                if table_match:
                    structured["schedules"][keyword.lower()] = table_match.group(1)
        
        return structured
    
    def perform_calculation(self, calc_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform specific engineering calculations"""
        
        if calc_type == "sprinkler_hydraulics":
            return self._calculate_sprinkler_hydraulics(parameters)
        elif calc_type == "structural_loads":
            return self._calculate_structural_loads(parameters)
        elif calc_type == "electrical_demand":
            return self._calculate_electrical_demand(parameters)
        elif calc_type == "hvac_loads":
            return self._calculate_hvac_loads(parameters)
        elif calc_type == "plumbing_demand":
            return self._calculate_plumbing_demand(parameters)
        elif calc_type == "quantity_takeoff":
            return self._perform_quantity_takeoff(parameters)
        elif calc_type == "code_compliance":
            return self._check_code_compliance(parameters)
        else:
            return {"error": f"Unknown calculation type: {calc_type}"}
    
    def _calculate_sprinkler_hydraulics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed sprinkler hydraulic calculations"""
        area = params.get("area_sqft", 0)
        occupancy = params.get("occupancy", "ordinary_1")
        height = params.get("ceiling_height", 10)
        
        # Get hazard classification
        hazard_data = self.knowledge.FIRE_PROTECTION_CALCS["sprinkler_density"].get(
            occupancy, 
            self.knowledge.FIRE_PROTECTION_CALCS["sprinkler_density"]["ordinary_1"]
        )
        
        # Design area (most remote)
        design_area = min(area, hazard_data["area"])
        
        # Calculate flow
        density = hazard_data["density"]
        sprinkler_flow = design_area * density
        hose_stream = hazard_data["hose"]
        total_flow = sprinkler_flow + hose_stream
        
        # Estimate pressure (simplified)
        # Starting pressure at sprinkler
        min_pressure = 7  # psi minimum
        
        # Friction loss estimate (very simplified)
        pipe_length = math.sqrt(design_area) * 2  # Rough estimate
        friction_loss = self.knowledge.FIRE_PROTECTION_CALCS["hydraulic_calculation"]["friction_loss"](
            120,  # C factor
            sprinkler_flow / 10,  # Flow per line
            4,    # Pipe diameter
            pipe_length
        )
        
        # Elevation pressure
        elevation_pressure = height * 0.433
        
        # Total pressure
        total_pressure = min_pressure + friction_loss + elevation_pressure + 10  # Safety factor
        
        # Pump sizing
        pump_hp = self.knowledge.FIRE_PROTECTION_CALCS["water_supply"]["pump_size"](
            total_flow, total_pressure
        )
        
        return {
            "hazard_classification": occupancy,
            "design_density": density,
            "design_area": design_area,
            "calculations": {
                "sprinkler_flow_gpm": round(sprinkler_flow),
                "hose_stream_gpm": hose_stream,
                "total_flow_gpm": round(total_flow),
                "min_pressure_psi": round(total_pressure, 1),
                "pump_size_hp": round(pump_hp, 1)
            },
            "pipe_sizing": {
                "mains": "6-8 inch",
                "cross_mains": "4-6 inch", 
                "branch_lines": "1.5-2.5 inch",
                "note": "Final sizes require hydraulic calculation software"
            },
            "water_supply_requirements": {
                "flow": f"{round(total_flow)} gpm @ {round(total_pressure)} psi",
                "duration": "60-120 minutes depending on occupancy",
                "storage": f"{round(total_flow * 90)} gallons minimum"
            }
        }
    
    def _calculate_structural_loads(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate structural loads and member sizes"""
        span = params.get("span_ft", 30)
        spacing = params.get("spacing_ft", 10)
        occupancy = params.get("occupancy", "office")
        
        # Load determination
        dead_loads = {
            "steel_deck": 3,
            "concrete_on_deck": 47,  # 3.5" normal weight
            "mep": 5,
            "ceiling": 3,
            "misc": 2
        }
        total_dead = sum(dead_loads.values())
        
        # Live loads by occupancy
        live_loads = {
            "office": 50,
            "corridor": 100,
            "retail": 100,
            "parking": 40,
            "roof": 20
        }
        live_load = live_loads.get(occupancy, 50)
        
        # Total load
        total_load = total_dead + live_load
        
        # Beam sizing (simplified)
        tributary_width = spacing
        w = total_load * tributary_width / 1000  # kips/ft
        M = w * span * span / 8  # kip-ft
        
        # Required section modulus (A992 steel, Fy=50ksi)
        Fb = 0.66 * 50  # Allowable stress
        S_req = M * 12 / Fb  # in³
        
        # Select beam size (simplified table)
        beam_sizes = [
            ("W12x26", 33.4), ("W14x30", 42.0), ("W16x36", 56.5),
            ("W18x40", 68.4), ("W21x44", 81.6), ("W24x55", 114.0),
            ("W27x84", 213.0), ("W30x90", 245.0), ("W33x118", 359.0)
        ]
        
        selected_beam = next((size for size, S in beam_sizes if S >= S_req), beam_sizes[-1])
        
        # Column load
        column_load = total_load * spacing * span / 2000  # kips
        
        return {
            "loads_psf": {
                "dead": total_dead,
                "live": live_load, 
                "total": total_load
            },
            "beam_design": {
                "span": span,
                "spacing": spacing,
                "uniform_load": round(w, 2),
                "moment": round(M),
                "required_S": round(S_req, 1),
                "selected_size": selected_beam[0]
            },
            "column_load_kips": round(column_load),
            "deflection_limit": f"L/{360 if occupancy != 'parking' else 240}",
            "notes": [
                "Preliminary sizing only",
                "Consider lateral loads",
                "Check local buckling and deflections",
                "Verify with structural analysis software"
            ]
        }
    
    def _calculate_electrical_demand(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate electrical loads per NEC"""
        area_sqft = params.get("area_sqft", 10000)
        occupancy = params.get("occupancy", "office")
        
        # Lighting loads per NEC Table 220.12
        lighting_loads = {
            "office": 1.3,
            "retail": 1.5,
            "warehouse": 0.25,
            "parking": 0.2,
            "restaurant": 2.0,
            "school": 1.5
        }
        
        lighting_wpersqft = lighting_loads.get(occupancy, 1.3)
        lighting_load = area_sqft * lighting_wpersqft
        
        # Receptacle loads
        receptacle_load = area_sqft * 1.0  # 1W/sqft general
        
        # HVAC estimate
        hvac_tons = area_sqft / 400  # Rule of thumb
        hvac_load = hvac_tons * 3517  # Watts per ton
        
        # Apply demand factors
        lighting_demand = lighting_load  # 100% continuous
        receptacle_demand = 10000 + (receptacle_load - 10000) * 0.5  # NEC 220.44
        hvac_demand = hvac_load  # 100%
        
        # Total demand
        total_demand = lighting_demand + receptacle_demand + hvac_demand
        
        # Service sizing
        voltage = 480  # Commercial
        phases = 3
        current = total_demand / (voltage * math.sqrt(3))
        
        # Panel sizing
        panel_size = next(s for s in [400, 600, 800, 1000, 1200, 1600, 2000, 2500, 3000] 
                         if s >= current * 1.25)
        
        # Transformer sizing
        transformer_kva = total_demand / 1000 / 0.9  # 90% PF
        std_transformer = next(s for s in [75, 112.5, 150, 225, 300, 500, 750, 1000, 1500]
                              if s >= transformer_kva)
        
        return {
            "area_sqft": area_sqft,
            "load_summary_watts": {
                "lighting": round(lighting_load),
                "receptacles": round(receptacle_load),
                "hvac": round(hvac_load),
                "total_connected": round(lighting_load + receptacle_load + hvac_load)
            },
            "demand_loads_watts": {
                "lighting": round(lighting_demand),
                "receptacles": round(receptacle_demand),
                "hvac": round(hvac_demand),
                "total_demand": round(total_demand)
            },
            "service_requirements": {
                "voltage": f"{voltage}V, 3-phase",
                "current_amps": round(current),
                "panel_size_amps": panel_size,
                "transformer_kva": std_transformer
            },
            "circuit_quantities": {
                "20A_lighting": math.ceil(lighting_demand / (277 * 16)),
                "20A_receptacle": math.ceil(receptacle_demand / (120 * 16)),
                "hvac_circuits": "As required per equipment"
            },
            "code_references": [
                "NEC Article 220 - Load Calculations",
                "NEC Table 220.12 - Lighting Load Demand",
                "NEC 220.44 - Receptacle Loads"
            ]
        }
    
    def _calculate_hvac_loads(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate HVAC loads using simplified methods"""
        area_sqft = params.get("area_sqft", 10000)
        occupancy = params.get("occupancy", "office")
        occupants = params.get("occupants", area_sqft / 150)
        location = params.get("location", "moderate")
        
        # Rules of thumb by occupancy
        cooling_factors = {
            "office": 300,      # sqft/ton
            "retail": 250,
            "restaurant": 200,
            "data_center": 100,
            "warehouse": 500
        }
        
        heating_factors = {  # BTU/sqft
            "office": 35,
            "retail": 40,
            "restaurant": 50,
            "warehouse": 30
        }
        
        # Cooling load
        tons_required = area_sqft / cooling_factors.get(occupancy, 300)
        cooling_btuh = tons_required * 12000
        
        # Heating load
        heating_btuh = area_sqft * heating_factors.get(occupancy, 35)
        
        # Ventilation per ASHRAE 62.1
        cfm_per_person = 5
        cfm_per_sqft = 0.06
        ventilation_cfm = (occupants * cfm_per_person) + (area_sqft * cfm_per_sqft)
        
        # Supply air (simplified)
        supply_cfm = max(ventilation_cfm, cooling_btuh / (1.08 * 20))  # 20°F delta T
        
        # Duct sizing (main)
        main_duct_area = supply_cfm / 1200  # 1200 fpm velocity
        
        return {
            "load_summary": {
                "cooling_tons": round(tons_required, 1),
                "cooling_btuh": round(cooling_btuh),
                "heating_btuh": round(heating_btuh),
                "sensible_heat_ratio": 0.75
            },
            "airflow_requirements": {
                "ventilation_cfm": round(ventilation_cfm),
                "supply_air_cfm": round(supply_cfm),
                "return_air_cfm": round(supply_cfm * 0.9),
                "exhaust_cfm": round(ventilation_cfm)
            },
            "equipment_selection": {
                "rooftop_units": math.ceil(tons_required / 25),
                "unit_size_tons": 25,
                "heating_type": "Gas furnace",
                "efficiency": "16 SEER, 95% AFUE"
            },
            "duct_sizing": {
                "main_duct_sqft": round(main_duct_area, 1),
                "typical_size": f"{int(math.sqrt(main_duct_area) * 12)} x {int(math.sqrt(main_duct_area) * 12)}",
                "velocity_fpm": 1200,
                "static_pressure": "2.5 inches w.c."
            },
            "controls": {
                "zones": math.ceil(area_sqft / 5000),
                "thermostats": math.ceil(area_sqft / 2500),
                "type": "DDC with BMS integration"
            }
        }
    
    def _calculate_plumbing_demand(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate plumbing fixture requirements and demands"""
        area_sqft = params.get("area_sqft", 10000)
        occupancy = params.get("occupancy", "office")
        occupants = params.get("occupants", 50)
        
        # Fixture calculations per IPC
        fixture_ratios = {
            "office": {
                "water_closet": {"male": 1/50, "female": 1/50},
                "lavatory": {"ratio": 1/40},
                "drinking_fountain": {"ratio": 1/100}
            },
            "retail": {
                "water_closet": {"male": 1/500, "female": 1/500},
                "lavatory": {"ratio": 1/750},
                "drinking_fountain": {"ratio": 1/1000}
            },
            "restaurant": {
                "water_closet": {"male": 1/75, "female": 1/75},
                "lavatory": {"ratio": 1/75},
                "drinking_fountain": {"ratio": 1/100}
            }
        }
        
        ratios = fixture_ratios.get(occupancy, fixture_ratios["office"])
        
        # Calculate fixtures
        male_occupants = occupants * 0.5
        female_occupants = occupants * 0.5
        
        fixtures = {
            "water_closets_male": max(1, math.ceil(male_occupants * ratios["water_closet"]["male"])),
            "water_closets_female": max(1, math.ceil(female_occupants * ratios["water_closet"]["female"])),
            "urinals": max(0, math.ceil(male_occupants * ratios["water_closet"]["male"] * 0.5)),
            "lavatories": max(2, math.ceil(occupants * ratios["lavatory"]["ratio"])),
            "drinking_fountains": max(1, math.ceil(occupants * ratios["drinking_fountain"]["ratio"]))
        }
        
        # Calculate fixture units
        fu_values = self.knowledge.MEP_CALCULATIONS["pipe_sizing"]["fixture_units"]
        total_wfu = (fixtures["water_closets_male"] * fu_values["water_closet"] +
                    fixtures["water_closets_female"] * fu_values["water_closet"] +
                    fixtures["urinals"] * fu_values["urinal"] +
                    fixtures["lavatories"] * fu_values["lavatory"])
        
        total_dfu = total_wfu  # Simplified - same for drainage
        
        # Water demand
        gpm_demand = 0.25 * math.sqrt(total_wfu) * 30  # Hunter's curve approximation
        
        # Hot water
        hot_water_demand = occupants * 2  # gallons per person per day
        heater_size = hot_water_demand * 1.25  # 25% safety factor
        
        return {
            "fixture_counts": fixtures,
            "fixture_units": {
                "water_supply": total_wfu,
                "drainage": total_dfu
            },
            "water_demand": {
                "peak_flow_gpm": round(gpm_demand),
                "daily_usage_gallons": round(occupants * 20),
                "meter_size": "2 inch minimum"
            },
            "hot_water": {
                "daily_demand_gallons": hot_water_demand,
                "heater_size_gallons": round(heater_size),
                "recovery_gph": round(heater_size / 4)
            },
            "pipe_sizing": {
                "cold_water_main": "2 inch",
                "hot_water_main": "1.5 inch",
                "branch_sizing": "Per fixture requirements"
            },
            "drainage": {
                "building_drain": "4 inch minimum",
                "vent_stack": "3 inch minimum",
                "slope": "1/4 inch per foot"
            },
            "code_reference": "IPC Table 403.1"
        }
    
    def _perform_quantity_takeoff(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quantity takeoff from drawing data"""
        drawing_type = params.get("drawing_type", "unknown")
        elements = params.get("elements", {})
        scale = params.get("scale", "1/8\" = 1'-0\"")
        
        takeoff = {
            "quantities": {},
            "measurements": {},
            "areas": {},
            "notes": []
        }
        
        # Scale factor
        scale_factor = self.knowledge.COMMON_SCALES.get(scale, 96)
        
        # Process by element type
        for element_type, count in elements.items():
            if element_type == "doors":
                takeoff["quantities"]["doors"] = {
                    "count": count,
                    "types": "Refer to door schedule",
                    "hardware_sets": count,
                    "frames": count
                }
            elif element_type == "walls":
                # Estimate from perimeter
                if "perimeter" in params:
                    wall_length = params["perimeter"]
                    wall_height = params.get("height", 10)
                    takeoff["measurements"]["walls"] = {
                        "length_lf": wall_length,
                        "area_sqft": wall_length * wall_height,
                        "studs_required": math.ceil(wall_length / 1.33),  # 16" OC
                        "gwb_sheets": math.ceil(wall_length * wall_height / 32)  # 4x8 sheets
                    }
            elif element_type == "concrete":
                if "slab_area" in params:
                    slab_area = params["slab_area"]
                    slab_thickness = params.get("thickness", 4) / 12  # inches to feet
                    takeoff["quantities"]["concrete"] = {
                        "slab_sqft": slab_area,
                        "volume_cy": round(slab_area * slab_thickness / 27, 1),
                        "rebar_tons": round(slab_area * 0.003, 2),  # Estimate
                        "vapor_barrier_sqft": slab_area * 1.1
                    }
        
        return takeoff
    
    def _check_code_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check code compliance for various requirements"""
        check_type = params.get("check_type", "general")
        
        if check_type == "egress":
            return self._check_egress_compliance(params)
        elif check_type == "accessibility":
            return self._check_ada_compliance(params)
        elif check_type == "fire_separation":
            return self._check_fire_separation(params)
        else:
            return {"error": "Unknown compliance check type"}
    
    def _check_egress_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check egress code compliance"""
        occupant_load = params.get("occupant_load", 100)
        travel_distance = params.get("travel_distance", 200)
        exits_provided = params.get("exits", 2)
        sprinklered = params.get("sprinklered", True)
        occupancy = params.get("occupancy", "business")
        
        # Required exits
        if occupant_load <= 49:
            required_exits = 1
        elif occupant_load <= 500:
            required_exits = 2
        elif occupant_load <= 1000:
            required_exits = 3
        else:
            required_exits = 4
        
        # Exit width
        stair_width_per_person = 0.3  # inches
        door_width_per_person = 0.2
        required_stair_width = occupant_load * stair_width_per_person
        required_door_width = occupant_load * door_width_per_person
        
        # Travel distance
        max_travel = self.knowledge.CODE_REQUIREMENTS["egress"]["travel_distance"][
            "sprinklered" if sprinklered else "non_sprinklered"
        ].get(occupancy, 250)
        
        compliance = {
            "exits": {
                "required": required_exits,
                "provided": exits_provided,
                "compliant": exits_provided >= required_exits
            },
            "width": {
                "stair_width_required": round(required_stair_width),
                "door_width_required": round(required_door_width),
                "min_stair_width": 44,
                "min_door_width": 32
            },
            "travel_distance": {
                "maximum_allowed": max_travel,
                "provided": travel_distance,
                "compliant": travel_distance <= max_travel
            },
            "common_path": {
                "maximum": 75 if sprinklered else 75,
                "note": "Varies by occupancy"
            },
            "dead_end": {
                "maximum": 50 if sprinklered else 20
            },
            "references": [
                "IBC Chapter 10 - Means of Egress",
                "IBC Table 1006.2.1 - Exit Width",
                "IBC Table 1017.2 - Travel Distance"
            ]
        }
        
        return compliance
    
    def _check_ada_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check ADA compliance"""
        # Implementation would go here
        return {"status": "ADA compliance check not fully implemented"}
    
    def _check_fire_separation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check fire separation requirements"""
        # Implementation would go here
        return {"status": "Fire separation check not fully implemented"}
    
    def process_schedule_data(self, schedule_type: str, raw_data: str) -> Dict[str, Any]:
        """Process schedule data into structured format"""
        schedule_info = self.knowledge.SCHEDULE_TYPES.get(schedule_type, {})
        
        result = {
            "type": schedule_type,
            "entries": [],
            "totals": {},
            "calculations": {}
        }
        
        # Parse the raw schedule data
        lines = raw_data.strip().split('\n')
        if not lines:
            return result
        
        # Assume first line is headers
        headers = [h.strip() for h in lines[0].split('\t')]
        
        # Parse entries
        for line in lines[1:]:
            if line.strip():
                values = [v.strip() for v in line.split('\t')]
                entry = dict(zip(headers, values))
                result["entries"].append(entry)
        
        # Perform calculations based on schedule type
        if schedule_type == "door_schedule" and result["entries"]:
            result["totals"]["total_doors"] = len(result["entries"])
            result["totals"]["fire_rated"] = sum(1 for e in result["entries"] 
                                               if "RATED" in e.get("Fire Rating", "").upper())
        
        elif schedule_type == "beam_schedule" and result["entries"]:
            # Calculate steel tonnage
            total_weight = 0
            for entry in result["entries"]:
                if "Size" in entry and "Length" in entry:
                    # Extract weight per foot from size (e.g., W24x55 = 55 lb/ft)
                    size_match = re.search(r"W\d+x(\d+)", entry["Size"])
                    if size_match:
                        weight_per_ft = float(size_match.group(1))
                        length = float(re.search(r"(\d+)", entry["Length"]).group(1))
                        total_weight += weight_per_ft * length
            
            result["calculations"]["total_tonnage"] = round(total_weight / 2000, 2)
        
        return result


class ProfessionalAIService:
    """Main AI service with human-like blueprint understanding - includes all required methods"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.OPENAI_API_KEY
        
        if not self.openai_api_key:
            logger.error("❌ OpenAI API key not provided")
            raise
    
    async def get_ai_response(self, prompt: str, document_id: str, 
                              storage_service: StorageService, author: str = None) -> str:
        """Process blueprint queries with human understanding"""
        try:
            logger.info(f"👁️ Processing blueprint query with human understanding for {document_id}")
            
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
                return "I need to see the blueprint to provide accurate analysis. Please ensure the document is properly uploaded."
            
            # Process with human-like understanding
            result = await self._process_like_human_expert(
                prompt=prompt,
                document_text=document_text,
                image_url=image_url,
                document_id=document_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return "I encountered an error analyzing the blueprint. Please try again."
    
    async def _process_like_human_expert(self, prompt: str, document_text: str, 
                                         image_url: str = None, document_id: str = None) -> str:
        """Process queries with human-like expertise"""
        try:
            # Expert system message
            system_message = {
                "role": "system",
                "content": """You are a master blueprint reader with 30+ years of experience. You read blueprints EXACTLY as a human expert would:

VISUAL PROCESSING:
1. First, always identify what drawing you're looking at (title block)
2. Note the scale - this is CRITICAL for all measurements
3. Scan for the big picture before details
4. Build a mental 3D model as you read

EXPERT KNOWLEDGE:
- You know every symbol, abbreviation, and convention
- You understand how different drawings relate (architectural to structural to MEP)
- You can perform any calculation in your head or show the work
- You know all relevant codes (IBC, NFPA, NEC, etc.)

RESPONSE STYLE:
- Start with what drawing you're analyzing
- Be specific with numbers, counts, and measurements
- Show calculations when asked
- Reference grid lines and specific locations
- Acknowledge what's shown vs. what's required by code
- Point out what information would be on other drawings

CRITICAL BEHAVIORS:
- If it's a parking garage, use Ordinary Hazard Group 1 (130 sqft/sprinkler)
- If it's an office, use Light Hazard (200 sqft/sprinkler)
- Count actual symbols shown, don't guess
- Read dimensions and areas directly from the drawing
- Identify schedules and extract data from them
- Notice revision clouds and check revision block

Example response:
"Looking at this Level P3 parking garage floor plan (Sheet AW-1.05, Scale: 1/8" = 1'-0"), I can see this is part of the South Yards Phase 1B project in Burnaby, BC.

The drawing shows a total area of 2,720.2 square meters (29,277 sq ft) as marked on the plan. For sprinkler requirements in a parking garage (Ordinary Hazard Group 1 per NFPA 13):

- Maximum coverage: 130 sq ft per sprinkler
- Required sprinklers: 29,277 ÷ 130 = 225 heads minimum
- Maximum spacing: 15 feet between heads

I can see sprinkler symbols ('sp' in circles) distributed throughout the parking areas. There are also water curtain sprinklers ('wc') shown at specific locations..."

Be this thorough and specific in every response."""
            }
            
            messages = [system_message]
            
            # Build user message
            user_message = {"role": "user", "content": []}
            
            if image_url:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"}
                })
            
            # Add context
            context = f"""Blueprint: {document_id}

Question: {prompt}

Remember: Read this like you're standing at a plan table, using your decades of experience."""
            
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
                    max_tokens=3000,
                    temperature=0.2
                )
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls
            if assistant_message.tool_calls:
                messages.append(assistant_message)
                
                for tool_call in assistant_message.tool_calls:
                    result = await self._execute_expert_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Get final response
                final_response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=3000,
                        temperature=0.2
                    )
                )
                
                return final_response.choices[0].message.content
            else:
                return assistant_message.content
                
        except Exception as e:
            logger.error(f"Expert analysis error: {e}")
            return f"Analysis error: {str(e)}"
    
    async def _execute_expert_tool(self, tool_call) -> Dict[str, Any]:
        """Execute expert analysis tools"""
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "analyze_blueprint_visually":
                return await self.analyzer.analyze_like_human(arguments["image_url"])
            
            elif function_name == "perform_engineering_calculation":
                return self.analyzer.perform_calculation(
                    arguments["calc_type"],
                    arguments["parameters"]
                )
            
            elif function_name == "process_schedule":
                return self.analyzer.process_schedule_data(
                    arguments["schedule_type"],
                    arguments["raw_data"]
                )
            
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Export the comprehensive AI service with all aliases
# These ensure compatibility with different parts of the codebase
ExpertBlueprintAI = ProfessionalAIService
EnhancedAIService = ProfessionalAIService
AIService = ProfessionalAIService ValueError("OpenAI API key is required")
        
        try:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=60.0)
            logger.info("✅ Human-like Blueprint AI initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        self.analyzer = HumanLikeBlueprintAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Advanced analysis tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_blueprint_visually",
                    "description": "Analyze blueprint with human-like visual understanding",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string", "description": "Blueprint image URL"}
                        },
                        "required": ["image_url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "perform_engineering_calculation",
                    "description": "Perform any engineering calculation (structural, MEP, fire, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "calc_type": {
                                "type": "string",
                                "enum": ["sprinkler_hydraulics", "structural_loads", "electrical_demand", 
                                        "hvac_loads", "plumbing_demand", "quantity_takeoff", "code_compliance"],
                                "description": "Type of calculation"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Calculation parameters"
                            }
                        },
                        "required": ["calc_type", "parameters"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "process_schedule",
                    "description": "Process and analyze schedule data from drawings",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "schedule_type": {
                                "type": "string",
                                "description": "Type of schedule (door, window, beam, etc.)"
                            },
                            "raw_data": {
                                "type": "string",
                                "description": "Raw schedule data"
                            }
                        },
                        "required": ["schedule_type", "raw_data"]
                    }
                }
            }
        ]
    
    def _smart_page_selection(self, query: str, visual_summary: Dict[str, Any], 
                            structured_text: Dict[str, Any], max_pages: int = 3) -> List[int]:
        """
        Intelligently select the most relevant pages based on the query.
        This is the method that was missing and causing the error.
        
        Args:
            query: User's question
            visual_summary: Visual summary data from the blueprint
            structured_text: Structured text data from the blueprint
            max_pages: Maximum number of pages to select
            
        Returns:
            List of page numbers (1-indexed) most relevant to the query
        """
        try:
            # Get total pages from visual summary
            total_pages = visual_summary.get('page_count', 1)
            
            # If we have 3 or fewer pages, return all
            if total_pages <= max_pages:
                return list(range(1, total_pages + 1))
            
            # Use the analyzer's knowledge base for intelligent selection
            query_lower = query.lower()
            
            # Score pages based on relevance
            page_scores = {}
            
            # Check if structured_text has page-specific content
            if 'pages' in structured_text:
                for page_data in structured_text['pages']:
                    page_num = page_data.get('page_number', 1)
                    score = 0
                    page_text = page_data.get('text', '').lower()
                    
                    # Check against drawing patterns
                    for drawing_type, patterns in self.analyzer.knowledge.DRAWING_PATTERNS.items():
                        for pattern in patterns:
                            if pattern.lower() in page_text:
                                score += 10
                                
                                # Extra weight for matches with query
                                if any(keyword in query_lower for keyword in pattern.lower().split()):
                                    score += 5
                    
                    # Check query terms
                    query_terms = query_lower.split()
                    for term in query_terms:
                        if len(term) > 3 and term in page_text:
                            score += 2
                    
                    # Special handling for common queries
                    if "what is" in query_lower or "identify" in query_lower:
                        if page_num == 1:  # First page often has title/overview
                            score += 10
                    
                    if "detail" in query_lower:
                        if "detail" in page_text:
                            score += 8
                    
                    if "schedule" in query_lower:
                        if any(sched in page_text for sched in ["schedule", "table", "legend"]):
                            score += 10
                    
                    page_scores[page_num] = score
            else:
                # No page-specific data, use heuristics
                if "what is" in query_lower or "identify" in query_lower:
                    return [1]  # Title page
                elif "detail" in query_lower:
                    return list(range(max(1, total_pages - 2), total_pages + 1))  # Last pages often have details
                else:
                    # Default to first few pages
                    return list(range(1, min(max_pages + 1, total_pages + 1)))
            
            # Sort pages by score and return top pages
            if page_scores:
                sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
                selected_pages = [page for page, score in sorted_pages[:max_pages] if score > 0]
                
                if not selected_pages:
                    selected_pages = [1]  # Default to first page
                
                return sorted(selected_pages)  # Return in page order
            else:
                return list(range(1, min(max_pages + 1, total_pages + 1)))
                
        except Exception as e:
            logger.warning(f"Error in smart page selection: {str(e)}. Defaulting to first page.")
            return [1]
    
    async def analyze_blueprint(self, document_id: str, query: str, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze blueprint with professional GPT-4o capabilities
        This method is called from the routes and must include _smart_page_selection
        
        Args:
            document_id: Unique document identifier
            query: User's question about the blueprint
            context: Context containing text and visual data
            
        Returns:
            AI analysis response
        """
        try:
            intent_info = await self._understand_intent(query)
            required_components = intent_info.get('required_components', ['text'])
            
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(intent_info['intent'])
                }
            ]
            
            selected_pages = []
            
            # Handle visual components with smart page selection
            if 'visual' in required_components and context.get('visual_summary'):
                visual_summary = context.get('visual_summary', {})
                structured_text = context.get('structured_text', {})
                
                # Use smart page selection for visual analysis
                selected_pages = self._smart_page_selection(
                    query, visual_summary, structured_text
                )
                
                logger.info(f"Smart page selection for '{query}': pages {selected_pages}")
                
                # Build user message with selected pages
                user_content = self._build_visual_message(
                    query, context, selected_pages
                )
                messages.append({
                    "role": "user", 
                    "content": user_content
                })
            else:
                # Text-only analysis
                messages.append({
                    "role": "user",
                    "content": self._build_text_message(query, context)
                })
            
            # Get AI response
            response = await self._get_ai_response(messages)
            
            return {
                "answer": response,
                "intent": intent_info['intent'],
                "confidence": intent_info.get('confidence', 0.95),
                "components_used": required_components,
                "pages_analyzed": selected_pages if 'visual' in required_components else None
            }
            
        except Exception as e:
            logger.error(f"❌ AI analysis error: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "answer": "I encountered an error analyzing the blueprint. Please try rephrasing your question."
            }
    
    def _get_system_prompt(self, intent: str) -> str:
        """Get specialized system prompt based on intent"""
        base_prompt = """You are a master blueprint reader with 30+ years of experience. You read blueprints EXACTLY as a human expert would:

VISUAL PROCESSING:
1. First, always identify what drawing you're looking at (title block)
2. Note the scale - this is CRITICAL for all measurements
3. Scan for the big picture before details
4. Build a mental 3D model as you read

EXPERT KNOWLEDGE:
- You know every symbol, abbreviation, and convention
- You understand how different drawings relate (architectural to structural to MEP)
- You can perform any calculation in your head or show the work
- You know all relevant codes (IBC, NFPA, NEC, etc.)

RESPONSE STYLE:
- Start with what drawing you're analyzing
- Be specific with numbers, counts, and measurements
- Show calculations when asked
- Reference grid lines and specific locations
- Acknowledge what's shown vs. what's required by code
- Point out what information would be on other drawings"""
        
        intent_prompts = {
            "identification": """Focus on identifying the type of blueprint, project details, 
            and key components. Describe what the blueprint shows and its purpose.""",
            
            "measurement": """Extract and calculate dimensions, areas, and quantities accurately. 
            Always state units and show calculations when relevant.""",
            
            "code_compliance": """Analyze for building code compliance, safety requirements, 
            and regulatory standards. Reference specific codes when applicable.""",
            
            "technical_details": """Explain technical specifications, materials, systems, 
            and construction methods shown in the blueprint.""",
            
            "comparison": """Compare different elements, options, or versions shown. 
            Highlight differences and similarities clearly."""
        }
        
        specific_prompt = intent_prompts.get(intent, "")
        return f"{base_prompt}\n\n{specific_prompt}"
    
    def _build_visual_message(self, query: str, context: Dict[str, Any], 
                            selected_pages: List[int]) -> List[Dict[str, Any]]:
        """Build message content including visual elements"""
        content = [
            {
                "type": "text",
                "text": f"Question: {query}\n\nContext:\n{context.get('text', '')[:2000]}"
            }
        ]
        
        # Add selected page images
        for page_num in selected_pages:
            page_image = context.get('pages', {}).get(f'page_{page_num}')
            if page_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page_image}",
                        "detail": "high"
                    }
                })
        
        return content
    
    def _build_text_message(self, query: str, context: Dict[str, Any]) -> str:
        """Build text-only message"""
        return f"""Question: {query}

Blueprint Context:
{context.get('text', '')[:3000]}

Please provide a detailed and accurate response based on the blueprint information."""
    
    async def _understand_intent(self, query: str) -> Dict[str, Any]:
        """Understand user intent from query"""
        query_lower = query.lower()
        
        # Intent patterns
        patterns = {
            "identification": ["what is", "identify", "type of", "what kind", "describe"],
            "measurement": ["dimension", "size", "area", "how big", "how large", "measure"],
            "code_compliance": ["code", "comply", "regulation", "standard", "requirement"],
            "technical_details": ["detail", "specification", "material", "system", "how"],
            "comparison": ["compare", "difference", "between", "versus", "vs"]
        }
        
        # Determine intent
        intent = "general"
        confidence = 0.7
        
        for intent_type, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                intent = intent_type
                confidence = 0.9
                break
        
        # Determine required components
        visual_triggers = ["show", "identify", "what is", "locate", "where", "find", 
                         "diagram", "drawing", "image", "picture"]
        
        required_components = ["text"]
        if any(trigger in query_lower for trigger in visual_triggers):
            required_components.append("visual")
        
        return {
            "intent": intent,
            "confidence": confidence,
            "required_components": required_components
        }
    
    async def _get_ai_response(self, messages: List[Dict[str, Any]]) -> str:
        """Get response from OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ OpenAI error: {str(e)}")
            raise
