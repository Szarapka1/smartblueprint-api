# app/services/ai_service.py

import logging
import base64
import re
import json
from typing import Optional, List, Dict, Any, Tuple # Added Any, Tuple for tool outputs
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionToolParam # Import for tool definition
import math # Added for UniversalConstructionAnalyzer's math functions
from datetime import datetime # Added for UniversalConstructionAnalyzer's usage if needed

from app.core.config import AppSettings
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# --- UniversalConstructionAnalyzer (copied directly from your provided code) ---
class UniversalConstructionAnalyzer:
    """Universal calculation and analysis tools for ALL construction trades and document types"""
    
    @staticmethod
    def calculate_by_area(total_area: float, coverage_per_unit: float, unit_name: str = "items", area_unit: str = "sqft") -> Dict[str, Any]:
        """Universal area-based calculation for ANY trade item"""
        try:
            units_needed = math.ceil(total_area / coverage_per_unit)
            
            return {
                "calculation_type": "area_based",
                "total_area": total_area,
                "area_unit": area_unit,
                "coverage_per_unit": coverage_per_unit,
                "units_needed": units_needed,
                "unit_type": unit_name,
                "with_10_percent_overage": math.ceil(units_needed * 1.1),
                "with_15_percent_overage": math.ceil(units_needed * 1.15),
                "formula": f"{total_area} {area_unit} ÷ {coverage_per_unit} {area_unit} per {unit_name} = {units_needed} {unit_name}"
            }
        except Exception as e:
            return {"error": f"Area calculation failed: {str(e)}"}
    
    @staticmethod
    def calculate_by_perimeter(perimeter_length: float, spacing: float, unit_name: str = "items", length_unit: str = "ft") -> Dict[str, Any]:
        """Universal perimeter/linear calculation"""
        try:
            units_needed = math.ceil(perimeter_length / spacing) + 1  # +1 for closure
            
            return {
                "calculation_type": "perimeter_based",
                "perimeter_length": perimeter_length,
                "length_unit": length_unit,
                "spacing": spacing,
                "units_needed": units_needed,
                "unit_type": unit_name,
                "actual_spacing": perimeter_length / (units_needed - 1) if units_needed > 1 else perimeter_length,
                "formula": f"{perimeter_length} {length_unit} ÷ {spacing} {length_unit} spacing + 1 = {units_needed} {unit_name}"
            }
        except Exception as e:
            return {"error": f"Perimeter calculation failed: {str(e)}"}
    
    @staticmethod
    def calculate_grid_pattern(length: float, width: float, spacing: float, unit_name: str = "items", unit: str = "ft") -> Dict[str, Any]:
        """Universal grid pattern calculation for any regularly spaced items"""
        try:
            units_x = math.ceil(length / spacing)
            units_y = math.ceil(width / spacing)
            total_units = units_x * units_y
            
            return {
                "calculation_type": "grid_pattern",
                "dimensions": f"{length} x {width} {unit}",
                "spacing": f"{spacing} {unit}",
                "units_along_length": units_x,
                "units_along_width": units_y,
                "total_units": total_units,
                "unit_type": unit_name,
                "actual_spacing_x": length / (units_x - 1) if units_x > 1 else length,
                "actual_spacing_y": width / (units_y - 1) if units_y > 1 else width,
                "coverage_area": length * width,
                "formula": f"{units_x} × {units_y} = {total_units} {unit_name}"
            }
        except Exception as e:
            return {"error": f"Grid calculation failed: {str(e)}"}
    
    @staticmethod
    def calculate_by_capacity(total_demand: float, unit_capacity: float, unit_name: str = "units") -> Dict[str, Any]:
        """Universal capacity-based calculation (electrical loads, HVAC, structural, etc.)"""
        try:
            units_needed = math.ceil(total_demand / unit_capacity)
            total_capacity = units_needed * unit_capacity
            utilization = (total_demand / total_capacity) * 100
            
            return {
                "calculation_type": "capacity_based",
                "total_demand": total_demand,
                "unit_capacity": unit_capacity,
                "units_needed": units_needed,
                "unit_type": unit_name,
                "total_provided_capacity": total_capacity,
                "utilization_percentage": round(utilization, 1),
                "spare_capacity": total_capacity - total_demand,
                "formula": f"{total_demand} ÷ {unit_capacity} per {unit_name} = {units_needed} {unit_name}"
            }
        except Exception as e:
            return {"error": f"Capacity calculation failed: {str(e)}"}
    
    @staticmethod
    def calculate_material_quantity(coverage_area_or_length: float, unit_coverage: float, waste_factor: float = 0.10, unit_name: str = "units") -> Dict[str, Any]:
        """Universal material quantity calculation with waste factor"""
        try:
            base_quantity = math.ceil(coverage_area_or_length / unit_coverage)
            waste_quantity = math.ceil(base_quantity * waste_factor)
            total_quantity = base_quantity + waste_quantity
            
            return {
                "calculation_type": "material_quantity",
                "coverage_needed": coverage_area_or_length,
                "unit_coverage": unit_coverage,
                "base_quantity": base_quantity,
                "waste_percentage": waste_factor * 100,
                "waste_quantity": waste_quantity,
                "total_order_quantity": total_quantity,
                "unit_type": unit_name,
                "formula": f"Base: {base_quantity} + Waste({waste_factor*100}%): {waste_quantity} = {total_quantity} {unit_name}"
            }
        except Exception as e:
            return {"error": f"Material calculation failed: {str(e)}"}
    
    @staticmethod
    def universal_unit_converter(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Comprehensive unit conversion for all construction measurements"""
        try:
            # Comprehensive conversion factors
            conversions = {
                # Length
                ("mm", "m"): 0.001, ("m", "mm"): 1000,
                ("cm", "m"): 0.01, ("m", "cm"): 100,
                ("in", "ft"): 1/12, ("ft", "in"): 12,
                ("ft", "m"): 0.3048, ("m", "ft"): 3.28084,
                ("in", "mm"): 25.4, ("mm", "in"): 0.0394,
                ("in", "cm"): 2.54, ("cm", "in"): 0.394,
                
                # Area
                ("sqft", "sqm"): 0.092903, ("sqm", "sqft"): 10.7639,
                ("sqin", "sqft"): 1/144, ("sqft", "sqin"): 144,
                ("sqm", "sqcm"): 10000, ("sqcm", "sqm"): 0.0001,
                
                # Volume
                ("cuft", "cum"): 0.0283168, ("cum", "cuft"): 35.3147,
                ("gal", "l"): 3.78541, ("l", "gal"): 0.264172,
                ("cuft", "gal"): 7.48052, ("gal", "cuft"): 0.133681,
                
                # Weight/Mass
                ("lb", "kg"): 0.453592, ("kg", "lb"): 2.20462,
                ("oz", "g"): 28.3495, ("g", "oz"): 0.035274,
                ("ton", "kg"): 907.185, ("kg", "ton"): 0.00110231,
                
                # Pressure
                ("psi", "kpa"): 6.89476, ("kpa", "psi"): 0.145038,
                ("psi", "bar"): 0.0689476, ("bar", "psi"): 14.5038,
                
                # Temperature
                # Note: Temperature conversions need special handling
                
                # Flow
                ("gpm", "lpm"): 3.78541, ("lpm", "gpm"): 0.264172,
                ("cfm", "cms"): 0.000471947, ("cms", "cfm"): 2118.88,
                
                # Power/Energy
                ("hp", "kw"): 0.745699, ("kw", "hp"): 1.34102,
                ("btu", "kj"): 1.05506, ("kj", "btu"): 0.947817,
                ("w", "btu_hr"): 3.41214, ("btu_hr", "w"): 0.293071,
                
                # Electrical
                ("a", "ma"): 1000, ("ma", "a"): 0.001,
                ("kw", "w"): 1000, ("w", "kw"): 0.001,
                ("mw", "w"): 1000000, ("w", "mw"): 0.000001
            }
            
            # Normalize unit names
            from_key = from_unit.lower().replace(" ", "").replace(".", "")
            to_key = to_unit.lower().replace(" ", "").replace(".", "")
            
            conversion_factor = conversions.get((from_key, to_key))
            
            if conversion_factor is None:
                return {"error": f"Conversion from {from_unit} to {to_unit} not supported"}
            
            converted_value = value * conversion_factor
            
            return {
                "original_value": value,
                "original_unit": from_unit,
                "converted_value": round(converted_value, 6),
                "converted_unit": to_unit,
                "conversion_factor": conversion_factor,
                "formula": f"{value} {from_unit} × {conversion_factor} = {converted_value} {to_unit}"
            }
            
        except Exception as e:
            return {"error": f"Unit conversion failed: {str(e)}"}
    
    @staticmethod
    def extract_measurements_from_text(text: str) -> Dict[str, Any]:
        """Extract ALL types of measurements and specifications from ANY construction document"""
        try:
            # Universal measurement patterns
            patterns = {
                # Dimensions - All formats
                'feet_inches': r"(\d+)'\s*-?\s*(\d+(?:\s*\d+/\d+)?)\s*\"?",
                'decimal_feet': r"(\d+\.?\d*)\s*(?:feet|ft|')",
                'meters': r"(\d+\.?\d*)\s*(?:meters?|m)(?!\w)",
                'millimeters': r"(\d+\.?\d*)\s*(?:mm|millimeters?)",
                'inches': r"(\d+\.?\d*)\s*(?:inches?|in|\")",
                
                # Areas
                'area_sqft': r"(\d+[\d,]*\.?\d*)\s*(?:sq\.?\s*ft|sqft|square\s*feet)",
                'area_sqm': r"(\d+[\d,]*\.?\d*)\s*(?:sq\.?\s*m|sqm|square\s*meters?)",
                
                # Volumes
                'volume_cuft': r"(\d+[\d,]*\.?\d*)\s*(?:cu\.?\s*ft|cuft|cubic\s*feet)",
                'volume_cum': r"(\d+[\d,]*\.?\d*)\s*(?:cu\.?\s*m|cum|cubic\s*meters?)",
                
                # Quantities and Counts
                'quantities': r"(\d+)\s*(?:each|ea|pieces?|pcs?|units?|items?)",
                'stalls': r"(\d+)\s*(?:stalls?|spaces?|parking)",
                'rooms': r"(\d+)\s*(?:rooms?|bedrooms?|bathrooms?)",
                
                # Electrical
                'voltage': r"(\d+)\s*(?:volts?|v)(?!\w)",
                'amperage': r"(\d+\.?\d*)\s*(?:amps?|amperes?|a)(?!\w)",
                'wattage': r"(\d+\.?\d*)\s*(?:watts?|w|kw|kilowatts?)(?!\w)",
                'circuits': r"(\d+)\s*(?:circuits?|cct)",
                
                # Mechanical/HVAC
                'cfm': r"(\d+\.?\d*)\s*cfm",
                'gpm': r"(\d+\.?\d*)\s*(?:us\s*)?gpm",
                'psi': r"(\d+\.?\d*)\s*psi",
                'btu': r"(\d+\.?\d*)\s*btu",
                'tons_cooling': r"(\d+\.?\d*)\s*tons?\s*(?:cooling|ac)",
                
                # Fire/Life Safety
                'fire_rating_hours': r"(\d+(?:\.\d+)?)\s*hr?\s*(?:fire|rated|rating)",
                'fire_rating_minutes': r"(\d+)\s*min\.?\s*(?:fire|rated|rating)",
                'sprinkler_spacing': r"(\d+)\s*(?:oc|on\s*center)",
                'exit_width': r"(\d+)\s*(?:inch|in|\")\s*(?:wide|width|door)",
                
                # Structural
                'loads_psf': r"(\d+\.?\d*)\s*psf",
                'loads_lbs': r"(\d+\.?\d*)\s*(?:lbs?|pounds?)",
                'concrete_psi': r"(\d+)\s*psi\s*(?:concrete|conc)",
                'rebar_size': r"#(\d+)\s*(?:rebar|bar)",
                'steel_sections': r"([wW]\d+[xX]\d+|[hH][sS]\d+[xX]\d+|[cC]\d+[xX]\d+)",
                
                # Plumbing
                'pipe_sizes': r"(\d+(?:\.\d+)?)\s*(?:inch|in|\")\s*(?:pipe|dia)",
                'flow_rates': r"(\d+\.?\d*)\s*(?:gpm|lpm)",
                'pressure': r"(\d+\.?\d*)\s*(?:psi|bar|kpa)",
                
                # Spacing/Layout
                'on_center': r"(\d+(?:\.\d+)?)\s*(?:oc|on\s*center)",
                'spacing': r"(\d+(?:\.\d+)?)\s*(?:spacing|spaced)",
                
                # Elevations
                'elevations': r"el\.?\s*(\d+\.?\d*)",
                'heights': r"(\d+\.?\d*)\s*(?:high|height|tall)",
                
                # Percentages
                'percentages': r"(\d+(?:\.\d+)?)\s*%",
                
                # Generic numbers with units
                'numeric_values': r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)"
            }
            
            extracted_data = {}
            for category, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    extracted_data[category] = matches
            
            return {
                "extraction_successful": True,
                "extracted_measurements": extracted_data,
                "total_patterns_found": len([k for k, v in extracted_data.items() if v])
            }
            
        except Exception as e:
            return {"error": f"Measurement extraction failed: {str(e)}"}
    
    @staticmethod
    def analyze_document_type(text: str) -> Dict[str, Any]:
        """Automatically determine document type and relevant trades - Comprehensive Analysis"""
        try:
            # Comprehensive document type indicators - 8x more detailed
            doc_indicators = {
                'architectural': [
                    # Plans & Drawings
                    'floor plan', 'elevation', 'section', 'site plan', 'roof plan', 'ceiling plan',
                    'reflected ceiling plan', 'rcp', 'demo plan', 'demolition', 'existing conditions',
                    'proposed plan', 'enlarged plan', 'partial plan', 'typical plan',
                    # Building Elements
                    'room', 'door', 'window', 'wall', 'partition', 'corridor', 'hallway', 'lobby',
                    'stair', 'elevator', 'escalator', 'balcony', 'terrace', 'atrium', 'courtyard',
                    'vestibule', 'foyer', 'reception', 'office', 'conference', 'restroom', 'storage',
                    'closet', 'janitor', 'kitchen', 'cafeteria', 'dining', 'lounge', 'bedroom',
                    'bathroom', 'living room', 'family room', 'garage', 'basement', 'attic',
                    # Finishes & Materials
                    'finish schedule', 'door schedule', 'window schedule', 'room finish',
                    'flooring', 'carpet', 'tile', 'hardwood', 'vinyl', 'ceiling', 'paint',
                    'wallpaper', 'millwork', 'casework', 'cabinetry', 'countertop', 'trim',
                    # Measurements & Annotations
                    'square footage', 'occupancy', 'ada', 'accessibility', 'egress', 'means of egress',
                    'building code', 'zoning', 'setback', 'height restriction', 'building height'
                ],
                
                'structural': [
                    # Structural Elements
                    'beam', 'column', 'foundation', 'footing', 'slab', 'deck', 'joist', 'girder',
                    'truss', 'purlin', 'girt', 'lintel', 'header', 'post', 'pile', 'caisson',
                    'spread footing', 'strip footing', 'mat foundation', 'basement wall',
                    'retaining wall', 'shear wall', 'bearing wall', 'load bearing', 'non-bearing',
                    # Materials
                    'rebar', 'reinforcement', 'concrete', 'steel', 'wood', 'timber', 'masonry',
                    'cmu', 'block', 'brick', 'stone', 'precast', 'prestressed', 'post-tension',
                    'structural steel', 'wide flange', 'i-beam', 'hss', 'angle', 'channel',
                    'tube steel', 'bar joist', 'metal deck', 'composite deck', 'plywood', 'osb',
                    'glulam', 'lvl', 'engineered lumber', 'dimensional lumber',
                    # Loads & Analysis
                    'load', 'dead load', 'live load', 'wind load', 'seismic', 'lateral load',
                    'point load', 'distributed load', 'moment', 'shear', 'deflection', 'stress',
                    'compression', 'tension', 'bearing', 'uplift', 'overturning', 'sliding',
                    'psf', 'plf', 'kip', 'pound', 'ksi', 'psi', 'allowable stress', 'ultimate',
                    # Standards & Codes
                    'aisc', 'aci', 'ibc', 'asce', 'nds', 'seismic design', 'wind design',
                    'structural analysis', 'structural design', 'structural plan', 'framing plan'
                ],
                
                'mechanical': [
                    # HVAC Systems
                    'hvac', 'heating', 'ventilation', 'air conditioning', 'cooling', 'climate control',
                    'ahu', 'air handler', 'rtu', 'rooftop unit', 'split system', 'vrf', 'vav',
                    'cav', 'fcu', 'fan coil', 'unit heater', 'baseboard', 'radiator', 'boiler',
                    'chiller', 'cooling tower', 'heat pump', 'geothermal', 'radiant heating',
                    'radiant cooling', 'underfloor heating', 'forced air', 'natural ventilation',
                    # Ductwork & Distribution
                    'duct', 'ductwork', 'supply', 'return', 'exhaust', 'outside air', 'fresh air',
                    'makeup air', 'diffuser', 'grille', 'register', 'damper', 'vav box',
                    'terminal unit', 'flex duct', 'spiral duct', 'rectangular duct', 'round duct',
                    'duct sizing', 'cfm', 'air flow', 'velocity', 'static pressure', 'total pressure',
                    # Controls & Instruments
                    'thermostat', 'sensor', 'actuator', 'control valve', 'control panel', 'bms',
                    'building management', 'ddc', 'pneumatic', 'electric', 'modulating', 'two-position',
                    'economizer', 'energy recovery', 'heat recovery', 'enthalpy wheel', 'heat wheel',
                    # Performance & Efficiency
                    'btu', 'btuh', 'ton', 'tonnage', 'seer', 'eer', 'cop', 'efficiency', 'energy star',
                    'leed', 'ashrae', 'load calculation', 'heat load', 'cooling load', 'ventilation rate',
                    'indoor air quality', 'iaq', 'filtration', 'uv', 'humidity', 'dehumidification'
                ],
                
                'electrical': [
                    # Power Distribution
                    'electrical', 'power', 'circuit', 'panel', 'panelboard', 'switchboard', 'mcc',
                    'motor control center', 'transformer', 'disconnect', 'breaker', 'fuse',
                    'load center', 'distribution panel', 'sub panel', 'main panel', 'service entrance',
                    'meter', 'utility', 'generator', 'ups', 'emergency power', 'standby power',
                    # Wiring & Devices
                    'outlet', 'receptacle', 'gfci', 'afci', 'switch', 'dimmer', 'occupancy sensor',
                    'motion sensor', 'photocell', 'timer', 'relay', 'contactor', 'starter',
                    'wire', 'cable', 'conductor', 'conduit', 'raceway', 'tray', 'busway', 'wirenut',
                    'junction box', 'pull box', 'device box', 'panel box', 'weatherproof',
                    # Lighting Systems
                    'lighting', 'fixture', 'luminaire', 'lamp', 'bulb', 'led', 'fluorescent',
                    'incandescent', 'halogen', 'hid', 'ballast', 'driver', 'emergency lighting',
                    'exit sign', 'exit lighting', 'accent lighting', 'task lighting', 'ambient',
                    'foot candle', 'lux', 'lumens', 'lumen', 'watt', 'efficacy', 'color temperature',
                    # Specifications & Standards
                    'amp', 'ampere', 'volt', 'voltage', 'watt', 'wattage', 'kw', 'kilowatt', 'kva',
                    'phase', 'single phase', 'three phase', 'neutral', 'ground', 'grounding',
                    'bonding', 'nec', 'national electrical code', 'ul', 'listed', 'approved',
                    'electrical plan', 'power plan', 'lighting plan', 'electrical schedule'
                ],
                
                'plumbing': [
                    # Water Systems
                    'plumbing', 'water', 'potable water', 'hot water', 'cold water', 'domestic water',
                    'water heater', 'boiler', 'tankless', 'storage tank', 'expansion tank',
                    'pressure tank', 'well', 'pump', 'sump pump', 'ejector pump', 'booster pump',
                    'water softener', 'filtration', 'backflow preventer', 'rpz', 'pressure reducing valve',
                    # Drainage & Waste
                    'drain', 'waste', 'vent', 'dwv', 'sanitary', 'sewer', 'septic', 'grease trap',
                    'floor drain', 'area drain', 'roof drain', 'scupper', 'downspout', 'gutter',
                    'storm water', 'storm drain', 'catch basin', 'manhole', 'cleanout', 'trap',
                    'p-trap', 's-trap', 'vent stack', 'soil stack', 'waste stack', 'branch',
                    # Fixtures & Equipment
                    'fixture', 'sink', 'lavatory', 'toilet', 'urinal', 'shower', 'bathtub', 'tub',
                    'faucet', 'valve', 'mixing valve', 'shower valve', 'flush valve', 'angle stop',
                    'hose bib', 'spigot', 'drinking fountain', 'water cooler', 'ice maker',
                    'dishwasher', 'washing machine', 'laundry', 'utility sink', 'mop sink',
                    # Piping & Materials
                    'pipe', 'piping', 'copper', 'pvc', 'cpvc', 'pex', 'cast iron', 'ductile iron',
                    'galvanized', 'stainless steel', 'hdpe', 'abs', 'fitting', 'elbow', 'tee',
                    'coupling', 'reducer', 'cap', 'plug', 'union', 'flange', 'gasket', 'solder',
                    'glue', 'primer', 'thread', 'npt', 'compression', 'push fit', 'press fit',
                    # Flow & Pressure
                    'gpm', 'gallons per minute', 'flow rate', 'pressure', 'psi', 'head', 'velocity',
                    'friction loss', 'pipe sizing', 'fixture units', 'water demand', 'peak demand'
                ],
                
                'fire_safety': [
                    # Fire Protection Systems
                    'fire protection', 'sprinkler', 'sprinkler system', 'fire sprinkler', 'wet system',
                    'dry system', 'preaction', 'deluge', 'foam system', 'suppression', 'suppression system',
                    'clean agent', 'co2', 'fm200', 'novec', 'water mist', 'standpipe', 'fire pump',
                    'fire tank', 'jockey pump', 'pressure maintenance', 'backflow preventer',
                    # Detection & Alarm
                    'fire alarm', 'smoke detector', 'heat detector', 'flame detector', 'gas detector',
                    'carbon monoxide', 'manual pull station', 'horn', 'strobe', 'speaker',
                    'notification appliance', 'control panel', 'facp', 'annunciator', 'repeater',
                    'wireless', 'addressable', 'conventional', 'analog', 'photoelectric', 'ionization',
                    # Emergency Systems
                    'emergency', 'exit', 'egress', 'exit sign', 'emergency lighting', 'path of egress',
                    'exit discharge', 'exit access', 'exit width', 'occupant load', 'travel distance',
                    'fire door', 'fire damper', 'smoke damper', 'fire wall', 'fire barrier',
                    'fire partition', 'smoke barrier', 'smoke partition', 'fire rating', 'hourly rating',
                    # Safety Equipment
                    'extinguisher', 'fire extinguisher', 'hose', 'fire hose', 'hydrant', 'fire hydrant',
                    'siamese connection', 'fdc', 'fire department connection', 'fire lane', 'access',
                    'emergency vehicle access', 'knox box', 'key box', 'emergency responder',
                    # Codes & Standards
                    'nfpa', 'ifc', 'life safety', 'building code', 'fire code', 'authority having jurisdiction',
                    'ahj', 'inspection', 'testing', 'maintenance', 'itm', 'commissioning', 'fire marshal'
                ],
                
                'site': [
                    # Site Planning
                    'site plan', 'site', 'plot plan', 'survey', 'boundary', 'property line', 'setback',
                    'easement', 'right of way', 'zoning', 'variance', 'conditional use', 'master plan',
                    'development', 'subdivision', 'lot', 'parcel', 'tract', 'acreage', 'density',
                    # Civil Engineering
                    'civil', 'grading', 'earthwork', 'excavation', 'cut', 'fill', 'slope', 'grade',
                    'elevation', 'contour', 'topography', 'drainage', 'swale', 'retention', 'detention',
                    'bioswale', 'rain garden', 'permeable', 'impervious', 'runoff', 'watershed',
                    'erosion control', 'sedimentation', 'best management practices', 'bmp',
                    # Infrastructure
                    'utility', 'utilities', 'water main', 'sewer main', 'storm sewer', 'gas line',
                    'electric', 'power line', 'telephone', 'cable', 'fiber optic', 'broadband',
                    'transformer', 'utility pole', 'manholes', 'valve', 'meter', 'service',
                    'connection', 'tap', 'lateral', 'stub', 'easement', 'franchise', 'right of way',
                    # Transportation
                    'parking', 'parking lot', 'parking space', 'stall', 'drive aisle', 'circulation',
                    'driveway', 'access', 'ingress', 'egress', 'roadway', 'street', 'sidewalk',
                    'walkway', 'path', 'trail', 'curb', 'gutter', 'median', 'island', 'traffic',
                    'signage', 'striping', 'marking', 'handicap', 'ada', 'accessible',
                    # Landscaping
                    'landscape', 'landscaping', 'planting', 'tree', 'shrub', 'grass', 'turf', 'sod',
                    'seed', 'mulch', 'irrigation', 'sprinkler', 'drip', 'plant list', 'plant schedule',
                    'native plants', 'drought tolerant', 'xeriscaping', 'maintenance', 'pruning',
                    # Pavements & Surfaces
                    'pavement', 'asphalt', 'concrete', 'paving', 'base course', 'subgrade',
                    'compaction', 'aggregate', 'crushed stone', 'gravel', 'sand', 'geotextile',
                    'pervious concrete', 'porous asphalt', 'pavers', 'brick', 'stone', 'decorative'
                ],
                
                'specifications': [
                    # Specification Documents
                    'specification', 'spec', 'specs', 'technical specification', 'construction specification',
                    'project manual', 'division', 'section', 'masterformat', 'csi', 'construction specifications institute',
                    'uniformat', 'omniclass', 'part 1', 'part 2', 'part 3', 'general', 'products', 'execution',
                    # Standards & Codes
                    'standard', 'code', 'regulation', 'requirement', 'criteria', 'guideline', 'procedure',
                    'astm', 'ansi', 'iso', 'ul', 'nema', 'ieee', 'aci', 'aisc', 'awwa', 'asme',
                    'building code', 'ibc', 'irc', 'imc', 'ipc', 'iecc', 'nec', 'nfpa', 'ada', 'accessibility',
                    'osha', 'epa', 'doe', 'energy code', 'green building', 'leed', 'energy star',
                    # Quality & Testing
                    'quality control', 'quality assurance', 'testing', 'inspection', 'verification',
                    'certification', 'approval', 'submittal', 'shop drawing', 'product data',
                    'sample', 'mock up', 'field test', 'laboratory test', 'third party', 'independent',
                    'performance', 'warranty', 'guarantee', 'maintenance', 'operation', 'commissioning',
                    # Contract & Legal
                    'note', 'general note', 'drawing note', 'detail', 'typical', 'schedule', 'legend',
                    'abbreviation', 'symbol', 'reference', 'see detail', 'see drawing', 'see specification',
                    'contractor', 'subcontractor', 'supplier', 'manufacturer', 'vendor', 'owner',
                    'architect', 'engineer', 'consultant', 'authority having jurisdiction', 'building official',
                    'permit', 'approval', 'variance', 'exception', 'substitution', 'equivalent', 'or equal'
                ],
                
                'surveying': [
                    # Survey Types
                    'survey', 'boundary survey', 'topographic', 'topo', 'as-built', 'existing conditions',
                    'alta', 'title survey', 'mortgage survey', 'location survey', 'site survey',
                    'construction survey', 'layout', 'stakeout', 'grade stakes', 'offset stakes',
                    # Measurements & Coordinates
                    'bearing', 'azimuth', 'distance', 'angle', 'coordinate', 'latitude', 'longitude',
                    'northing', 'easting', 'elevation', 'benchmark', 'datum', 'grid', 'state plane',
                    'utm', 'gps', 'gnss', 'total station', 'theodolite', 'level', 'rod', 'prism',
                    # Property & Legal
                    'deed', 'plat', 'record', 'monument', 'marker', 'pin', 'iron pipe', 'concrete monument',
                    'found', 'set', 'corner', 'lot corner', 'section corner', 'quarter corner',
                    'metes and bounds', 'calls', 'course', 'legal description', 'tract', 'parcel'
                ],
                
                'geotechnical': [
                    # Soil & Foundation
                    'geotechnical', 'soils', 'soil investigation', 'boring', 'test pit', 'spt',
                    'standard penetration test', 'soil classification', 'unified soil classification',
                    'uscs', 'clay', 'silt', 'sand', 'gravel', 'rock', 'bedrock', 'groundwater',
                    'water table', 'bearing capacity', 'settlement', 'consolidation', 'compaction',
                    'density', 'moisture content', 'plasticity', 'liquid limit', 'plastic limit',
                    'foundation', 'shallow foundation', 'deep foundation', 'pile', 'drilled pier',
                    'caisson', 'mat foundation', 'raft foundation', 'grade beam', 'tie beam'
                ],
                
                'environmental': [
                    # Environmental Assessment
                    'environmental', 'phase i', 'phase ii', 'esa', 'environmental site assessment',
                    'contamination', 'hazardous materials', 'hazmat', 'asbestos', 'lead', 'radon',
                    'underground storage tank', 'ust', 'petroleum', 'chlorinated solvents', 'pcb',
                    'brownfield', 'remediation', 'cleanup', 'mitigation', 'monitoring', 'sampling',
                    # Sustainability
                    'sustainability', 'green building', 'leed', 'energy efficiency', 'renewable energy',
                    'solar', 'wind', 'geothermal', 'carbon footprint', 'life cycle assessment', 'lca',
                    'embodied energy', 'recycled content', 'regional materials', 'low voc', 'indoor air quality'
                ]
            }
            
            detected_types = {}
            text_lower = text.lower()
            
            for doc_type, keywords in doc_indicators.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                if matches > 0:
                    detected_types[doc_type] = matches
            
            # Determine primary document type
            primary_type = max(detected_types, key=detected_types.get) if detected_types else "unknown"
            
            return {
                "primary_document_type": primary_type,
                "all_detected_types": detected_types,
                "relevant_trades": list(detected_types.keys()),
                "confidence_score": max(detected_types.values()) if detected_types else 0
            }
            
        except Exception as e:
            return {"error": f"Document analysis failed: {str(e)}"}

class EnhancedUniversalAIService:
    """AI Service with comprehensive tools for ALL construction trades and document types"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.analyzer = UniversalConstructionAnalyzer()
        
        # Define universal tools that work for ANY construction document
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_by_area",
                    "description": "Calculate quantities for any area-based items (sprinklers, lights, outlets, flooring, paint, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "total_area": {"type": "number", "description": "Total area needing coverage"},
                            "coverage_per_unit": {"type": "number", "description": "Area covered by each unit"},
                            "unit_name": {"type": "string", "description": "Type of item (sprinklers, lights, etc.)"},
                            "area_unit": {"type": "string", "description": "Unit of area (sqft, sqm, etc.)", "default": "sqft"}
                        },
                        "required": ["total_area", "coverage_per_unit", "unit_name"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "calculate_by_perimeter",
                    "description": "Calculate quantities for perimeter-based items (wall outlets, perimeter lighting, handrails, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "perimeter_length": {"type": "number", "description": "Total perimeter length"},
                            "spacing": {"type": "number", "description": "Spacing between items"},
                            "unit_name": {"type": "string", "description": "Type of item"},
                            "length_unit": {"type": "string", "description": "Unit of length", "default": "ft"}
                        },
                        "required": ["perimeter_length", "spacing", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_grid_pattern", 
                    "description": "Calculate quantities for grid-pattern items (ceiling tiles, sprinklers, lights, supports, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "length": {"type": "number", "description": "Length dimension"},
                            "width": {"type": "number", "description": "Width dimension"}, 
                            "spacing": {"type": "number", "description": "Grid spacing"},
                            "unit_name": {"type": "string", "description": "Type of item"},
                            "unit": {"type": "string", "description": "Unit of measurement", "default": "ft"}
                        },
                        "required": ["length", "width", "spacing", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_by_capacity",
                    "description": "Calculate quantities based on load/capacity (electrical panels, HVAC units, structural members, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "total_demand": {"type": "number", "description": "Total demand/load"},
                            "unit_capacity": {"type": "number", "description": "Capacity per unit"},
                            "unit_name": {"type": "string", "description": "Type of unit"}
                        },
                        "required": ["total_demand", "unit_capacity", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_material_quantity",
                    "description": "Calculate material quantities with waste factor (lumber, concrete, wiring, piping, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coverage_needed": {"type": "number", "description": "Total coverage needed"},
                            "unit_coverage": {"type": "number", "description": "Coverage per unit"},
                            "waste_factor": {"type": "number", "description": "Waste factor (0.10 = 10%)", "default": 0.10},
                            "unit_name": {"type": "string", "description": "Type of material"}
                        },
                        "required": ["coverage_needed", "unit_coverage", "unit_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_units",
                    "description": "Convert between any construction measurement units",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number", "description": "Value to convert"},
                            "from_unit": {"type": "string", "description": "Original unit"},
                            "to_unit": {"type": "string", "description": "Target unit"}
                        },
                        "required": ["value", "from_unit", "to_unit"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_measurements",
                    "description": "Extract all measurements and specifications from construction documents",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "document_text": {"type": "string", "description": "Text content from the document"}
                        },
                        "required": ["document_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_document_type",
                    "description": "Determine document type and relevant construction trades",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_text": {"type": "string", "description": "Text content from the document"}
                        },
                        "required": ["document_text"]
                    }
                }
            }
        ]
    
    def _execute_tool_call(self, tool_call) -> str:
        """Execute any tool call and return formatted results"""
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Route to appropriate analyzer method
            if function_name == "calculate_by_area":
                result = self.analyzer.calculate_by_area(**arguments)
            elif function_name == "calculate_by_perimeter":
                result = self.analyzer.calculate_by_perimeter(**arguments)
            elif function_name == "calculate_grid_pattern":
                result = self.analyzer.calculate_grid_pattern(**arguments)
            elif function_name == "calculate_by_capacity":
                result = self.analyzer.calculate_by_capacity(**arguments)
            elif function_name == "calculate_material_quantity":
                result = self.analyzer.calculate_material_quantity(**arguments)
            elif function_name == "convert_units":
                result = self.analyzer.universal_unit_converter(**arguments)
            elif function_name == "extract_measurements":
                result = self.analyzer.extract_measurements_from_text(arguments["document_text"])
            elif function_name == "analyze_document_type":
                result = self.analyzer.analyze_document_type(arguments["document_text"])
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
    
    async def process_query_with_vision_and_tools(self, prompt: str, document_text: str = "", image_url: str = None) -> Dict[str, Any]:
        """Process any construction-related query with both vision and calculation capabilities"""
        try:
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": """You are a universal construction AI assistant with comprehensive calculation and analysis capabilities. 
                    
You can help with ANY construction trade including:
- Architecture (rooms, doors, windows, areas)
- Structural (beams, columns, loads, materials)
- Mechanical (HVAC, ductwork, equipment)
- Electrical (circuits, loads, panels, outlets)
- Plumbing (pipes, fixtures, flow rates)
- Fire Safety (sprinklers, exits, alarms)
- Site Work (parking, utilities, grading)

For ANY quantification question, use your calculation tools to provide exact numbers with proper formulas and methodology.

Always:
1. Analyze what type of calculation is needed
2. Extract relevant measurements from the document
3. Use appropriate calculation tools
4. Show your work with clear formulas
5. Provide practical recommendations

Be thorough and professional in your analysis."""
                }
            ]
            
            # Add user message with vision if image provided
            user_message = {"role": "user", "content": []}
            
            if image_url:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            if document_text:
                user_message["content"].append({
                    "type": "text", 
                    "text": f"Document content:\n{document_text}\n\nQuestion: {prompt}"
                })
            else:
                user_message["content"].append({
                    "type": "text",
                    "text": prompt
                })
            
            messages.append(user_message)
            
            # Make API call with tools
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.1
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls if present
            if assistant_message.tool_calls:
                # Add assistant message to conversation
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_result = self._execute_tool_call(tool_call)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Get final response with tool results
                final_response = openai.chat.completions.create(
                    model="gpt-4-vision-preview", 
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1
                )
                
                final_content = final_response.choices[0].message.content
            else:
                final_content = assistant_message.content
            
            return {
                "ai_response": final_content,
                "tools_used": len(assistant_message.tool_calls) if assistant_message.tool_calls else 0,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return {
                "ai_response": f"I encountered an error processing your request: {str(e)}",
                "success": False,
                "error": str(e)
            }
