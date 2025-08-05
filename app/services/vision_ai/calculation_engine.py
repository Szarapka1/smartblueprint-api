# calculation_engine.py - Construction Calculations Engine

import math
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CalculationType(Enum):
    """Types of calculations the engine can perform"""
    AREA = "area"
    LENGTH = "length"
    LOAD = "load"
    COST = "cost"
    RATIO = "ratio"
    SPACING = "spacing"
    COVERAGE = "coverage"
    QUANTITY = "quantity"
    VOLUME = "volume"
    PERCENTAGE = "percentage"


@dataclass
class CalculationResult:
    """Result of a calculation with units and confidence"""
    value: float
    unit: str
    calculation_type: CalculationType
    formula_used: str
    assumptions: List[str]
    confidence: float
    details: Dict[str, Any]


class CalculationEngine:
    """
    Construction Calculation Engine
    
    Performs all types of construction calculations based on document knowledge.
    Integrates with the Document Intelligence system to provide instant calculations.
    """
    
    def __init__(self):
        # Standard construction values and codes
        self.standards = {
            "electrical": {
                "outlet_spacing_residential": 12,  # feet
                "outlet_spacing_commercial": 20,   # feet
                "outlet_load_standard": 180,       # watts
                "outlet_load_dedicated": 1500,     # watts
                "lighting_load_residential": 3,    # watts/sq ft
                "lighting_load_commercial": 1.5,   # watts/sq ft
                "panel_efficiency": 0.8,           # 80% derating
            },
            "plumbing": {
                "fixture_units": {
                    "toilet": 4,
                    "urinal": 4,
                    "lavatory": 2,
                    "sink": 2,
                    "shower": 3,
                    "tub": 4,
                    "drinking_fountain": 1,
                },
                "pipe_flow_rates": {
                    "1/2": 5,   # GPM
                    "3/4": 10,  # GPM
                    "1": 20,    # GPM
                    "1.5": 40,  # GPM
                    "2": 65,    # GPM
                }
            },
            "hvac": {
                "cooling_load_residential": 400,    # sq ft per ton
                "cooling_load_commercial": 300,     # sq ft per ton
                "cfm_per_ton": 400,                # CFM
                "diffuser_coverage": 100,           # sq ft per diffuser
                "duct_velocity_supply": 700,        # FPM
                "duct_velocity_return": 500,        # FPM
            },
            "fire_protection": {
                "sprinkler_coverage_light": 225,    # sq ft
                "sprinkler_coverage_ordinary": 130, # sq ft
                "sprinkler_coverage_extra": 100,    # sq ft
                "sprinkler_spacing_max": 15,        # feet
            },
            "structural": {
                "concrete_coverage": 150,           # sq ft per cubic yard
                "rebar_spacing_slab": 18,          # inches
                "steel_weight_per_foot": {         # lbs/ft
                    "W8": 24,
                    "W10": 33,
                    "W12": 40,
                    "W14": 48,
                    "W16": 57,
                    "W18": 65,
                }
            },
            "materials": {
                "drywall_sheet": 32,               # sq ft (4x8)
                "paint_coverage": 350,             # sq ft per gallon
                "carpet_roll_width": 12,           # feet
                "ceiling_tile": 4,                 # sq ft (2x2)
                "concrete_psi": 3000,              # standard PSI
            }
        }
        
        # Cost data (example values - would be updated regularly)
        self.cost_data = {
            "materials": {
                "outlet": 15,
                "switch": 12,
                "door_standard": 250,
                "door_fire_rated": 450,
                "window_standard": 400,
                "light_fixture_standard": 150,
                "sprinkler_head": 75,
                "diffuser": 85,
                "drywall_sheet": 12,
                "paint_gallon": 35,
            },
            "labor_hours": {
                "outlet": 0.5,
                "switch": 0.5,
                "door": 2.5,
                "window": 3.0,
                "light_fixture": 1.0,
                "sprinkler_head": 1.5,
                "diffuser": 1.0,
            },
            "labor_rate": 75  # $/hour average
        }
        
        logger.info("✅ Calculation Engine initialized")
    
    async def calculate(
        self,
        calculation_request: str,
        document_knowledge: Dict[str, Any],
        element_data: Optional[Dict[str, Any]] = None
    ) -> CalculationResult:
        """
        Main calculation method - interprets request and performs calculation.
        
        Args:
            calculation_request: Natural language calculation request
            document_knowledge: Complete document understanding
            element_data: Specific element data if needed
        """
        
        logger.info(f"🧮 Calculating: {calculation_request}")
        
        # Determine calculation type
        calc_type = self._determine_calculation_type(calculation_request)
        
        # Route to appropriate calculator
        if calc_type == CalculationType.AREA:
            return await self._calculate_area(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.LOAD:
            return await self._calculate_load(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.COST:
            return await self._calculate_cost(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.COVERAGE:
            return await self._calculate_coverage(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.SPACING:
            return await self._calculate_spacing(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.QUANTITY:
            return await self._calculate_quantity(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.RATIO:
            return await self._calculate_ratio(calculation_request, document_knowledge)
        
        elif calc_type == CalculationType.PERCENTAGE:
            return await self._calculate_percentage(calculation_request, document_knowledge)
        
        else:
            return self._create_error_result("Cannot determine calculation type")
    
    def _determine_calculation_type(self, request: str) -> CalculationType:
        """Determine what type of calculation is being requested."""
        
        request_lower = request.lower()
        
        # Check for calculation keywords
        if any(word in request_lower for word in ["area", "square footage", "sq ft", "square feet"]):
            return CalculationType.AREA
        
        elif any(word in request_lower for word in ["load", "watts", "amps", "voltage", "power"]):
            return CalculationType.LOAD
        
        elif any(word in request_lower for word in ["cost", "price", "budget", "$", "dollar"]):
            return CalculationType.COST
        
        elif any(word in request_lower for word in ["coverage", "cover", "protection"]):
            return CalculationType.COVERAGE
        
        elif any(word in request_lower for word in ["spacing", "distance", "apart", "between"]):
            return CalculationType.SPACING
        
        elif any(word in request_lower for word in ["how many", "quantity", "amount", "total"]):
            return CalculationType.QUANTITY
        
        elif any(word in request_lower for word in ["ratio", "proportion", "compare"]):
            return CalculationType.RATIO
        
        elif any(word in request_lower for word in ["percent", "%", "percentage"]):
            return CalculationType.PERCENTAGE
        
        else:
            return CalculationType.QUANTITY  # Default
    
    async def _calculate_area(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate area-based metrics."""
        
        request_lower = request.lower()
        
        # Extract building dimensions from knowledge
        project_info = knowledge.get("project_overview", {})
        scale_info = project_info.get("scale", "")
        
        # Try to extract square footage from project overview
        sqft_match = re.search(r'(\d+,?\d*)\s*(?:sq|square)\s*(?:ft|feet|foot)', scale_info)
        
        if sqft_match:
            total_sqft = float(sqft_match.group(1).replace(',', ''))
            
            # Check what specific area calculation is needed
            if "per floor" in request_lower or "each floor" in request_lower:
                floors = self._extract_floor_count(knowledge)
                if floors > 0:
                    sqft_per_floor = total_sqft / floors
                    
                    return CalculationResult(
                        value=round(sqft_per_floor, 0),
                        unit="sq ft per floor",
                        calculation_type=CalculationType.AREA,
                        formula_used=f"Total Area ({total_sqft:,.0f} sq ft) ÷ Floors ({floors})",
                        assumptions=[f"Assumed {floors} floors of equal size"],
                        confidence=0.85,
                        details={
                            "total_area": total_sqft,
                            "floors": floors,
                            "area_per_floor": sqft_per_floor
                        }
                    )
            
            # Return total area
            return CalculationResult(
                value=total_sqft,
                unit="sq ft",
                calculation_type=CalculationType.AREA,
                formula_used="Extracted from project overview",
                assumptions=["Based on project documentation"],
                confidence=0.95,
                details={"source": "project_overview"}
            )
        
        # Fallback: estimate from room count or other metrics
        return self._estimate_area_from_elements(knowledge)
    
    async def _calculate_load(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate electrical, HVAC, or structural loads."""
        
        request_lower = request.lower()
        
        if "electrical" in request_lower or "power" in request_lower:
            return self._calculate_electrical_load(request, knowledge)
        
        elif "cooling" in request_lower or "hvac" in request_lower:
            return self._calculate_hvac_load(request, knowledge)
        
        else:
            return self._calculate_electrical_load(request, knowledge)  # Default
    
    def _calculate_electrical_load(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate electrical load based on outlets, lights, and equipment."""
        
        all_elements = knowledge.get("all_elements", {})
        
        total_load = 0
        load_details = {}
        assumptions = []
        
        # Outlet load
        if "outlet" in all_elements:
            outlet_count = all_elements["outlet"].get("total_count", 0)
            outlet_load = outlet_count * self.standards["electrical"]["outlet_load_standard"]
            total_load += outlet_load
            load_details["outlets"] = {
                "count": outlet_count,
                "watts_each": self.standards["electrical"]["outlet_load_standard"],
                "total_watts": outlet_load
            }
            assumptions.append(f"Assumed {self.standards['electrical']['outlet_load_standard']}W per outlet")
        
        # Lighting load
        if "light fixture" in all_elements:
            light_count = all_elements["light fixture"].get("total_count", 0)
            light_load = light_count * 100  # Assume 100W average per fixture
            total_load += light_load
            load_details["lighting"] = {
                "count": light_count,
                "watts_each": 100,
                "total_watts": light_load
            }
            assumptions.append("Assumed 100W average per light fixture")
        
        # Add area-based lighting load if we have area
        area_result = self._estimate_area_from_elements(knowledge)
        if area_result.value > 0:
            area_lighting_load = area_result.value * self.standards["electrical"]["lighting_load_commercial"]
            total_load += area_lighting_load
            load_details["area_lighting"] = {
                "area_sqft": area_result.value,
                "watts_per_sqft": self.standards["electrical"]["lighting_load_commercial"],
                "total_watts": area_lighting_load
            }
            assumptions.append(f"Added {self.standards['electrical']['lighting_load_commercial']}W/sq ft for general lighting")
        
        # Convert to kW if large
        if total_load > 10000:
            value = total_load / 1000
            unit = "kW"
        else:
            value = total_load
            unit = "watts"
        
        return CalculationResult(
            value=round(value, 1),
            unit=unit,
            calculation_type=CalculationType.LOAD,
            formula_used="Sum of all electrical loads",
            assumptions=assumptions,
            confidence=0.85,
            details=load_details
        )
    
    def _calculate_hvac_load(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate HVAC cooling/heating load."""
        
        # Get building area
        area_result = self._estimate_area_from_elements(knowledge)
        building_area = area_result.value
        
        if building_area == 0:
            return self._create_error_result("Cannot determine building area for HVAC calculation")
        
        # Determine if residential or commercial
        project_type = knowledge.get("project_overview", {}).get("project_type", "").lower()
        
        if "residential" in project_type:
            sqft_per_ton = self.standards["hvac"]["cooling_load_residential"]
        else:
            sqft_per_ton = self.standards["hvac"]["cooling_load_commercial"]
        
        # Calculate tons of cooling needed
        tons_needed = building_area / sqft_per_ton
        
        # Calculate CFM needed
        cfm_needed = tons_needed * self.standards["hvac"]["cfm_per_ton"]
        
        return CalculationResult(
            value=round(tons_needed, 1),
            unit="tons of cooling",
            calculation_type=CalculationType.LOAD,
            formula_used=f"Building Area ({building_area:,.0f} sq ft) ÷ {sqft_per_ton} sq ft/ton",
            assumptions=[
                f"Used {'residential' if 'residential' in project_type else 'commercial'} standard",
                f"Assumed {sqft_per_ton} sq ft per ton"
            ],
            confidence=0.80,
            details={
                "building_area": building_area,
                "sqft_per_ton": sqft_per_ton,
                "tons": tons_needed,
                "cfm_required": cfm_needed
            }
        )
    
    async def _calculate_cost(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate costs for materials, labor, or total project."""
        
        all_elements = knowledge.get("all_elements", {})
        total_material_cost = 0
        total_labor_cost = 0
        cost_breakdown = {}
        
        # Calculate costs for each element type
        for element_type, element_data in all_elements.items():
            count = element_data.get("total_count", 0)
            
            if count > 0 and element_type in self.cost_data["materials"]:
                # Material cost
                material_unit_cost = self.cost_data["materials"].get(element_type, 0)
                material_total = count * material_unit_cost
                total_material_cost += material_total
                
                # Labor cost
                labor_hours = self.cost_data["labor_hours"].get(element_type, 0)
                labor_total = count * labor_hours * self.cost_data["labor_rate"]
                total_labor_cost += labor_total
                
                cost_breakdown[element_type] = {
                    "count": count,
                    "material_unit": material_unit_cost,
                    "material_total": material_total,
                    "labor_hours": labor_hours * count,
                    "labor_total": labor_total,
                    "total": material_total + labor_total
                }
        
        total_cost = total_material_cost + total_labor_cost
        
        return CalculationResult(
            value=round(total_cost, 2),
            unit="USD",
            calculation_type=CalculationType.COST,
            formula_used="Sum of (Material Costs + Labor Costs)",
            assumptions=[
                f"Labor rate: ${self.cost_data['labor_rate']}/hour",
                "Used standard material costs",
                "Does not include overhead or profit"
            ],
            confidence=0.75,
            details={
                "material_cost": round(total_material_cost, 2),
                "labor_cost": round(total_labor_cost, 2),
                "total_cost": round(total_cost, 2),
                "breakdown": cost_breakdown
            }
        )
    
    async def _calculate_coverage(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate coverage area for sprinklers, lighting, HVAC, etc."""
        
        request_lower = request.lower()
        
        if "sprinkler" in request_lower:
            return self._calculate_sprinkler_coverage(knowledge)
        elif "diffuser" in request_lower or "hvac" in request_lower:
            return self._calculate_hvac_coverage(knowledge)
        else:
            return self._calculate_sprinkler_coverage(knowledge)  # Default
    
    def _calculate_sprinkler_coverage(
        self,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate sprinkler coverage area."""
        
        all_elements = knowledge.get("all_elements", {})
        sprinkler_count = all_elements.get("sprinkler", {}).get("total_count", 0)
        
        if sprinkler_count == 0:
            return self._create_error_result("No sprinklers found in document")
        
        # Assume ordinary hazard occupancy
        coverage_per_head = self.standards["fire_protection"]["sprinkler_coverage_ordinary"]
        total_coverage = sprinkler_count * coverage_per_head
        
        # Get building area for comparison
        area_result = self._estimate_area_from_elements(knowledge)
        building_area = area_result.value
        
        coverage_percentage = (total_coverage / building_area * 100) if building_area > 0 else 0
        
        return CalculationResult(
            value=round(total_coverage, 0),
            unit="sq ft",
            calculation_type=CalculationType.COVERAGE,
            formula_used=f"{sprinkler_count} sprinklers × {coverage_per_head} sq ft/sprinkler",
            assumptions=[
                "Assumed ordinary hazard occupancy",
                f"Used {coverage_per_head} sq ft per sprinkler"
            ],
            confidence=0.85,
            details={
                "sprinkler_count": sprinkler_count,
                "coverage_per_head": coverage_per_head,
                "total_coverage": total_coverage,
                "building_area": building_area,
                "coverage_percentage": round(coverage_percentage, 1)
            }
        )
    
    def _calculate_hvac_coverage(
        self,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate HVAC diffuser coverage area."""
        
        all_elements = knowledge.get("all_elements", {})
        diffuser_count = all_elements.get("diffuser", {}).get("total_count", 0)
        
        if diffuser_count == 0:
            return self._create_error_result("No diffusers found in document")
        
        coverage_per_diffuser = self.standards["hvac"]["diffuser_coverage"]
        total_coverage = diffuser_count * coverage_per_diffuser
        
        return CalculationResult(
            value=round(total_coverage, 0),
            unit="sq ft",
            calculation_type=CalculationType.COVERAGE,
            formula_used=f"{diffuser_count} diffusers × {coverage_per_diffuser} sq ft/diffuser",
            assumptions=[f"Assumed {coverage_per_diffuser} sq ft coverage per diffuser"],
            confidence=0.80,
            details={
                "diffuser_count": diffuser_count,
                "coverage_per_diffuser": coverage_per_diffuser,
                "total_coverage": total_coverage
            }
        )
    
    async def _calculate_spacing(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate average spacing between elements."""
        
        # Extract element type from request
        element_type = self._extract_element_from_request(request)
        
        if not element_type:
            return self._create_error_result("Cannot determine element type for spacing calculation")
        
        all_elements = knowledge.get("all_elements", {})
        element_data = all_elements.get(element_type, {})
        count = element_data.get("total_count", 0)
        
        if count <= 1:
            return self._create_error_result(f"Need at least 2 {element_type}s for spacing calculation")
        
        # Get perimeter or area for spacing calculation
        area_result = self._estimate_area_from_elements(knowledge)
        building_area = area_result.value
        
        # Estimate perimeter (assume square building for simplicity)
        perimeter = 4 * math.sqrt(building_area) if building_area > 0 else 0
        
        # Calculate average spacing
        if element_type == "outlet":
            # Outlets are typically along walls
            avg_spacing = perimeter / count if count > 0 else 0
            
            # Check against code
            residential_max = self.standards["electrical"]["outlet_spacing_residential"]
            compliant = avg_spacing <= residential_max
            
            return CalculationResult(
                value=round(avg_spacing, 1),
                unit="feet",
                calculation_type=CalculationType.SPACING,
                formula_used=f"Perimeter ({perimeter:.0f} ft) ÷ {count} outlets",
                assumptions=[
                    "Assumed square building shape",
                    "Assumed outlets distributed along perimeter"
                ],
                confidence=0.70,
                details={
                    "count": count,
                    "perimeter": round(perimeter, 0),
                    "avg_spacing": round(avg_spacing, 1),
                    "code_compliant": compliant,
                    "code_max": residential_max
                }
            )
        
        else:
            # For other elements, use area-based spacing
            area_per_element = building_area / count if count > 0 else 0
            avg_spacing = math.sqrt(area_per_element)
            
            return CalculationResult(
                value=round(avg_spacing, 1),
                unit="feet",
                calculation_type=CalculationType.SPACING,
                formula_used=f"√(Area per {element_type})",
                assumptions=["Assumed uniform distribution"],
                confidence=0.65,
                details={
                    "count": count,
                    "building_area": building_area,
                    "area_per_element": round(area_per_element, 0),
                    "avg_spacing": round(avg_spacing, 1)
                }
            )
    
    async def _calculate_quantity(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate quantities of materials needed."""
        
        request_lower = request.lower()
        
        # Check what quantity is being requested
        if "paint" in request_lower:
            return self._calculate_paint_quantity(knowledge)
        elif "drywall" in request_lower:
            return self._calculate_drywall_quantity(knowledge)
        elif "conduit" in request_lower or "wire" in request_lower:
            return self._calculate_conduit_quantity(knowledge)
        else:
            return self._create_error_result("Cannot determine what quantity to calculate")
    
    def _calculate_paint_quantity(
        self,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate paint needed based on wall area."""
        
        # Estimate wall area from building area
        area_result = self._estimate_area_from_elements(knowledge)
        floor_area = area_result.value
        
        # Estimate wall area (assume 10ft ceilings, 4 walls)
        perimeter = 4 * math.sqrt(floor_area)
        wall_area = perimeter * 10  # 10 ft ceiling height
        
        # Add ceiling area
        total_paintable_area = wall_area + floor_area
        
        # Calculate gallons needed
        coverage_per_gallon = self.standards["materials"]["paint_coverage"]
        gallons_needed = total_paintable_area / coverage_per_gallon
        
        # Add 10% waste factor
        gallons_with_waste = gallons_needed * 1.1
        
        return CalculationResult(
            value=math.ceil(gallons_with_waste),
            unit="gallons",
            calculation_type=CalculationType.QUANTITY,
            formula_used=f"Paintable Area ({total_paintable_area:.0f} sq ft) ÷ {coverage_per_gallon} sq ft/gallon × 1.1",
            assumptions=[
                "Assumed 10 ft ceiling height",
                "Included walls and ceilings",
                "Added 10% waste factor"
            ],
            confidence=0.70,
            details={
                "floor_area": floor_area,
                "wall_area": round(wall_area, 0),
                "total_area": round(total_paintable_area, 0),
                "gallons_exact": round(gallons_needed, 1),
                "gallons_with_waste": round(gallons_with_waste, 1)
            }
        )
    
    def _calculate_drywall_quantity(
        self,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate drywall sheets needed."""
        
        # Get wall area similar to paint calculation
        area_result = self._estimate_area_from_elements(knowledge)
        floor_area = area_result.value
        
        perimeter = 4 * math.sqrt(floor_area)
        wall_area = perimeter * 10  # 10 ft ceiling height
        
        # Drywall for walls (both sides) and ceiling
        total_drywall_area = (wall_area * 2) + floor_area
        
        # Calculate sheets needed
        sheet_size = self.standards["materials"]["drywall_sheet"]
        sheets_needed = total_drywall_area / sheet_size
        
        # Add 15% waste factor
        sheets_with_waste = sheets_needed * 1.15
        
        return CalculationResult(
            value=math.ceil(sheets_with_waste),
            unit="sheets (4x8)",
            calculation_type=CalculationType.QUANTITY,
            formula_used=f"Drywall Area ({total_drywall_area:.0f} sq ft) ÷ {sheet_size} sq ft/sheet × 1.15",
            assumptions=[
                "Assumed 10 ft ceiling height",
                "Included both sides of walls",
                "Added 15% waste factor"
            ],
            confidence=0.70,
            details={
                "wall_area": round(wall_area, 0),
                "ceiling_area": floor_area,
                "total_area": round(total_drywall_area, 0),
                "sheets_exact": round(sheets_needed, 1),
                "sheets_with_waste": round(sheets_with_waste, 1)
            }
        )
    
    def _calculate_conduit_quantity(
        self,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate conduit length needed for electrical."""
        
        all_elements = knowledge.get("all_elements", {})
        outlet_count = all_elements.get("outlet", {}).get("total_count", 0)
        panel_count = all_elements.get("panel", {}).get("total_count", 0)
        
        if outlet_count == 0:
            return self._create_error_result("No outlets found for conduit calculation")
        
        # Estimate average run length from panel to outlet
        avg_run_length = 50  # feet, typical
        
        # Calculate total conduit needed
        total_conduit = outlet_count * avg_run_length
        
        # Add home runs (assume 1 circuit per 10 outlets)
        circuits = math.ceil(outlet_count / 10)
        home_run_length = circuits * 30  # 30 ft average home run
        
        total_conduit += home_run_length
        
        # Add 20% waste factor
        total_with_waste = total_conduit * 1.2
        
        return CalculationResult(
            value=round(total_with_waste, 0),
            unit="linear feet",
            calculation_type=CalculationType.QUANTITY,
            formula_used="(Outlet runs + Home runs) × 1.2 waste factor",
            assumptions=[
                f"Assumed {avg_run_length} ft average run per outlet",
                "Assumed 1 circuit per 10 outlets",
                "Added 20% waste factor"
            ],
            confidence=0.65,
            details={
                "outlet_count": outlet_count,
                "circuits": circuits,
                "outlet_runs": outlet_count * avg_run_length,
                "home_runs": home_run_length,
                "total_length": total_conduit,
                "with_waste": round(total_with_waste, 0)
            }
        )
    
    async def _calculate_ratio(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate ratios between different elements or metrics."""
        
        request_lower = request.lower()
        
        if "window" in request_lower and "wall" in request_lower:
            return self._calculate_window_wall_ratio(knowledge)
        else:
            return self._create_error_result("Cannot determine what ratio to calculate")
    
    def _calculate_window_wall_ratio(
        self,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate window-to-wall ratio for energy analysis."""
        
        all_elements = knowledge.get("all_elements", {})
        window_count = all_elements.get("window", {}).get("total_count", 0)
        
        if window_count == 0:
            return CalculationResult(
                value=0,
                unit="percent",
                calculation_type=CalculationType.RATIO,
                formula_used="No windows found",
                assumptions=[],
                confidence=0.95,
                details={"window_count": 0}
            )
        
        # Estimate window area (assume 15 sq ft average per window)
        avg_window_area = 15  # sq ft
        total_window_area = window_count * avg_window_area
        
        # Estimate wall area
        area_result = self._estimate_area_from_elements(knowledge)
        floor_area = area_result.value
        perimeter = 4 * math.sqrt(floor_area)
        wall_area = perimeter * 10  # 10 ft ceiling height
        
        # Calculate ratio
        window_wall_ratio = (total_window_area / wall_area) * 100 if wall_area > 0 else 0
        
        return CalculationResult(
            value=round(window_wall_ratio, 1),
            unit="percent",
            calculation_type=CalculationType.RATIO,
            formula_used=f"Window Area ({total_window_area} sq ft) ÷ Wall Area ({wall_area:.0f} sq ft) × 100",
            assumptions=[
                f"Assumed {avg_window_area} sq ft per window",
                "Assumed 10 ft ceiling height",
                "Exterior walls only"
            ],
            confidence=0.70,
            details={
                "window_count": window_count,
                "window_area": total_window_area,
                "wall_area": round(wall_area, 0),
                "ratio": round(window_wall_ratio, 1)
            }
        )
    
    async def _calculate_percentage(
        self,
        request: str,
        knowledge: Dict[str, Any]
    ) -> CalculationResult:
        """Calculate percentage of completion, coverage, etc."""
        
        # For now, redirect to coverage calculation
        return await self._calculate_coverage(request, knowledge)
    
    # Helper methods
    
    def _estimate_area_from_elements(self, knowledge: Dict[str, Any]) -> CalculationResult:
        """Estimate building area from element counts when not directly available."""
        
        all_elements = knowledge.get("all_elements", {})
        
        # Method 1: From room count
        room_count = all_elements.get("room", {}).get("total_count", 0)
        if room_count > 0:
            avg_room_size = 150  # sq ft average
            estimated_area = room_count * avg_room_size
            
            return CalculationResult(
                value=estimated_area,
                unit="sq ft",
                calculation_type=CalculationType.AREA,
                formula_used=f"{room_count} rooms × {avg_room_size} sq ft/room",
                assumptions=[f"Assumed {avg_room_size} sq ft average per room"],
                confidence=0.60,
                details={"room_count": room_count}
            )
        
        # Method 2: From door count
        door_count = all_elements.get("door", {}).get("total_count", 0)
        if door_count > 0:
            # Rough estimate: 1 door per 500 sq ft
            estimated_area = door_count * 500
            
            return CalculationResult(
                value=estimated_area,
                unit="sq ft",
                calculation_type=CalculationType.AREA,
                formula_used=f"{door_count} doors × 500 sq ft/door",
                assumptions=["Rough estimate: 1 door per 500 sq ft"],
                confidence=0.50,
                details={"door_count": door_count}
            )
        
        # No good estimate available
        return CalculationResult(
            value=0,
            unit="sq ft",
            calculation_type=CalculationType.AREA,
            formula_used="Could not estimate",
            assumptions=["Insufficient data for area estimation"],
            confidence=0.0,
            details={}
        )
    
    def _extract_floor_count(self, knowledge: Dict[str, Any]) -> int:
        """Extract number of floors from knowledge."""
        
        # Check spatial organization
        spatial_org = knowledge.get("spatial_organization", {})
        floors = spatial_org.get("floors", [])
        
        if floors:
            return len(floors)
        
        # Check project overview
        scale_info = knowledge.get("project_overview", {}).get("scale", "")
        
        # Look for floor mentions
        floor_match = re.search(r'(\d+)\s*(?:story|stories|floor|floors)', scale_info, re.IGNORECASE)
        if floor_match:
            return int(floor_match.group(1))
        
        # Check page contents for floor references
        page_contents = knowledge.get("page_contents", {})
        floor_numbers = set()
        
        for content in page_contents.values():
            if "floor" in content.get("page_type", "").lower():
                # Extract floor number
                floor_num_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*floor', 
                                          content.get("content_summary", ""), re.IGNORECASE)
                if floor_num_match:
                    floor_numbers.add(int(floor_num_match.group(1)))
        
        return len(floor_numbers) if floor_numbers else 1
    
    def _extract_element_from_request(self, request: str) -> Optional[str]:
        """Extract element type from calculation request."""
        
        request_lower = request.lower()
        
        # Common element types
        element_types = [
            "outlet", "door", "window", "panel", "light fixture",
            "sprinkler", "diffuser", "column", "beam", "pipe",
            "duct", "equipment", "plumbing fixture"
        ]
        
        for element in element_types:
            if element in request_lower:
                return element
        
        # Check plurals
        for element in element_types:
            if element + "s" in request_lower:
                return element
        
        return None
    
    def _create_error_result(self, error_message: str) -> CalculationResult:
        """Create an error calculation result."""
        
        return CalculationResult(
            value=0,
            unit="error",
            calculation_type=CalculationType.QUANTITY,
            formula_used="N/A",
            assumptions=[],
            confidence=0.0,
            details={"error": error_message}
        )