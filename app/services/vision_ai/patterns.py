# app/services/vision_ai/patterns.py
"""
Visual patterns and constants for construction element detection.

This module contains the comprehensive pattern definitions used by the Visual Intelligence
system to detect and analyze construction elements in blueprints and technical drawings.
"""

# Comprehensive Visual Patterns for Construction Elements
# Designed to ENHANCE GPT-4 Vision's natural ability to SEE and UNDERSTAND
# These are GUIDES, not rules - GPT should trust what it SEES
VISUAL_PATTERNS = {
    "door": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Openings in walls where people pass through.
        
        VISUAL CLUES TO USE YOUR INTELLIGENCE ON:
        - Look for breaks in wall lines - that's where doors go
        - There's usually a curved line (arc) showing how the door swings
        - The door itself might be shown as a thin rectangle at an angle
        - Sometimes labeled with D1, D2, etc.
        - Use your intelligence - if it looks like a door opening, it probably is!
        
        TRUST YOUR VISION: You can recognize doors even if they're drawn differently
        than described. Use these hints but trust what you SEE.
        """,
        "typical_appearance": [
            "Break in wall with quarter-circle arc",
            "Thin rectangle showing door position",
            "May have swing direction indicated",
            "Often numbered or labeled"
        ],
        "common_variations": "Single, double, sliding, pocket, bi-fold, overhead",
        "highlight_geometry": "door_with_swing"
    },
    
    "window": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Openings in walls for light and views.
        
        USE YOUR VISUAL INTELLIGENCE:
        - Windows often have multiple parallel lines (showing glass/frames)
        - Look for rectangles in walls, especially exterior walls
        - May have a sill that sticks out past the wall
        - Sometimes show mullions or grid patterns
        - If it's in a wall and looks like glass, it's probably a window!
        
        YOU KNOW WHAT WINDOWS LOOK LIKE - trust your recognition abilities.
        """,
        "typical_appearance": [
            "Rectangle in wall with double/triple lines",
            "Sill projection on exterior side",
            "May show glass pattern or mullions",
            "Often labeled W1, W2, etc."
        ],
        "common_variations": "Fixed, casement, double-hung, awning, sliding",
        "highlight_geometry": "rectangular_with_sill"
    },
    
    "outlet": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Electrical outlets/receptacles in walls.
        
        USE YOUR RECOGNITION SKILLS:
        - Usually small circles on or near walls
        - Might have one or two lines (showing plug slots)
        - Could be labeled with circuit numbers
        - GFCI outlets might have 'G' or 'GFCI' notation
        - Trust your ability to spot electrical symbols!
        
        They're usually where you'd expect outlets - along walls, near counters, etc.
        """,
        "typical_appearance": [
            "Small circle with line(s)",
            "May have ground symbol",
            "Circuit number nearby",
            "Height notation (like 18\" AFF)"
        ],
        "common_variations": "Standard, GFCI, 220V, floor outlet",
        "highlight_geometry": "circular_symbol"
    },
    
    "panel": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Electrical panels/breaker boxes.
        
        YOUR INTELLIGENT RECOGNITION:
        - Rectangular boxes on walls
        - Usually labeled P1, P2, EP1, etc.
        - Might show circuit count or amperage
        - Often in utility areas, garages, electrical rooms
        - You can recognize these - they look like what they are!
        
        Use context - if it's a rectangle labeled "Panel" or "P", that's it!
        """,
        "typical_appearance": [
            "Rectangle on wall",
            "Panel designation (P1, P2)",
            "May list amperage/voltage",
            "Shows required clearances"
        ],
        "common_variations": "Main panel, subpanel, distribution panel",
        "highlight_geometry": "rectangular"
    },
    
    "light fixture": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Lights in ceilings or on walls.
        
        VISUAL INTELLIGENCE:
        - Various symbols on ceiling plans (circles, squares, rectangles)
        - Often have type labels (A, B, C or L1, L2, etc.)
        - May show connection to switches
        - Sometimes dimensions like 2x4 for recessed fixtures
        - If it provides light, you'll recognize it!
        """,
        "typical_appearance": [
            "Geometric shapes in ceiling",
            "Letter/number designation",
            "May show mounting type",
            "Switch connections shown"
        ],
        "common_variations": "Recessed, surface, pendant, track, emergency",
        "highlight_geometry": "symbol_based"
    },
    
    "plumbing fixture": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Toilets, sinks, showers, tubs in bathrooms/kitchens.
        
        USE YOUR KNOWLEDGE:
        - Toilets are usually oval or elongated shapes
        - Sinks are rectangles or circles with drain indicated
        - Showers are squares with X or drain symbol
        - Tubs are long rectangles
        - Trust your understanding of what these look like!
        """,
        "typical_appearance": [
            "Distinctive shapes for each fixture",
            "Usually in wet areas",
            "May show water/drain connections",
            "Often have labels"
        ],
        "common_variations": "Standard, ADA, wall-mounted, floor-mounted",
        "highlight_geometry": "symbol_based"
    },
    
    "sprinkler": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Fire sprinkler heads in ceilings.
        
        VISUAL INTELLIGENCE TIPS:
        - Small symbols in ceiling plans
        - Often circles with crosses or asterisk patterns
        - Usually in a regular grid pattern
        - Might show coverage areas
        - If it's in the ceiling and fights fires, you've found it!
        
        Trust your pattern recognition - sprinklers have distinctive symbols.
        """,
        "typical_appearance": [
            "Circle with cross inside",
            "Asterisk or star pattern",
            "Regular spacing pattern",
            "May show pipe connections"
        ],
        "common_variations": "Pendant, upright, sidewall, concealed",
        "highlight_geometry": "circular_symbol"
    },
    
    "diffuser": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Air vents/diffusers in ceilings.
        
        YOUR RECOGNITION:
        - Square or rectangular symbols with patterns inside
        - Often show air flow direction
        - May have size labels (24x24, etc.)
        - Usually connected to ductwork
        - If it moves air, you'll spot it!
        """,
        "typical_appearance": [
            "Square with X or lines pattern",
            "Size dimensions shown",
            "CFM (air flow) notation",
            "Duct connections"
        ],
        "common_variations": "Supply, return, linear, round",
        "highlight_geometry": "square_with_pattern"
    },
    
    "column": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Structural columns holding up the building.
        
        USE YOUR UNDERSTANDING:
        - Usually solid squares, rectangles, or circles
        - Often at grid intersections (like A-1, B-2)
        - Might be filled solid or have hatching
        - Labeled C1, C2, or similar
        - You know what holds up buildings - trust that knowledge!
        
        If it's at a grid intersection and looks structural, it's probably a column.
        """,
        "typical_appearance": [
            "Solid filled shape",
            "At grid intersections",
            "Column mark (C1, C2)",
            "May show size dimensions"
        ],
        "common_variations": "Steel, concrete, round, square, rectangular",
        "highlight_geometry": "rectangular"
    },
    
    "beam": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Horizontal structural members.
        
        VISUAL CLUES:
        - Dashed lines above (hidden from view)
        - Connect between columns
        - May have beam designation (B1, B2)
        - Show depth/size notation
        - If it spans between supports, it's likely a beam!
        """,
        "typical_appearance": [
            "Dashed parallel lines",
            "Between columns",
            "Size/depth notation",
            "May show connections"
        ],
        "common_variations": "Steel W-shapes, concrete, wood, composite",
        "highlight_geometry": "linear_path"
    },
    
    "pipe": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Plumbing or process piping.
        
        YOUR INTELLIGENCE:
        - Single or double lines with size notation
        - Connect to fixtures or equipment
        - May show flow direction arrows
        - Often labeled with pipe size
        - Trust your understanding of how fluids move!
        """,
        "typical_appearance": [
            "Lines with size labels",
            "Flow arrows",
            "Valve symbols",
            "Connection to fixtures"
        ],
        "common_variations": "Supply, waste, vent, process",
        "highlight_geometry": "linear_path"
    },
    
    "duct": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: HVAC ductwork for air distribution.
        
        VISUAL RECOGNITION:
        - Double lines or rectangles with dimensions
        - Show width x height (like 24x12)
        - Connect to diffusers and equipment
        - May show insulation notation
        - If it carries air, you'll see it!
        """,
        "typical_appearance": [
            "Double lines with dimensions",
            "Rectangular sections",
            "Size notation (WxH)",
            "Connected to air devices"
        ],
        "common_variations": "Supply, return, exhaust, round, rectangular",
        "highlight_geometry": "linear_path"
    },
    
    "equipment": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Mechanical/electrical equipment.
        
        USE YOUR KNOWLEDGE:
        - Large rectangles with labels
        - Equipment tags (AHU-1, RTU-1, etc.)
        - Show connections and clearances
        - Often in mechanical rooms or on roof
        - Trust your ability to recognize equipment!
        """,
        "typical_appearance": [
            "Large rectangles",
            "Equipment tags/labels",
            "Connection points",
            "Clearance zones shown"
        ],
        "common_variations": "HVAC units, pumps, generators, transformers",
        "highlight_geometry": "rectangular"
    },
    
    "stair": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Stairs between levels.
        
        VISUAL CLUES:
        - Series of parallel lines (treads)
        - Direction arrow showing UP or DN
        - Break line where stairs continue
        - Stair identification (ST1, ST2)
        - You know what stairs look like!
        """,
        "typical_appearance": [
            "Parallel lines for treads",
            "UP/DN directional arrow",
            "Break line symbol",
            "Handrails shown"
        ],
        "common_variations": "Straight, L-shaped, U-shaped, spiral",
        "highlight_geometry": "linear_path"
    },
    
    "parking": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Parking spaces/stalls.
        
        YOUR RECOGNITION:
        - Rectangular spaces in parking areas
        - Usually striped or outlined
        - May be numbered
        - Accessible spaces marked with symbols
        - Trust your understanding of parking layouts!
        """,
        "typical_appearance": [
            "Rectangular spaces",
            "Stripe lines",
            "Space numbers",
            "Accessible symbols"
        ],
        "common_variations": "Standard, compact, accessible, angled",
        "highlight_geometry": "rectangular"
    },
    
    "fire alarm device": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Fire alarm devices (pulls, horns, strobes).
        
        VISUAL INTELLIGENCE:
        - Small symbols on walls
        - Triangles, squares with letters
        - Near exits and in corridors
        - May show coverage patterns
        - If it alerts about fire, you'll find it!
        """,
        "typical_appearance": [
            "Triangle or square symbols",
            "Device type letters",
            "Mounting heights noted",
            "Near exits"
        ],
        "common_variations": "Pull stations, horn/strobe, smoke detectors",
        "highlight_geometry": "symbol_based"
    },
    
    "elevator": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Elevators/lifts.
        
        USE YOUR KNOWLEDGE:
        - Rectangle with X or diagonal lines
        - Shows car and shaft
        - Door opening indicated
        - May have equipment room above
        - You know what moves people vertically!
        """,
        "typical_appearance": [
            "Rectangle with X pattern",
            "Shaft walls shown",
            "Door location",
            "Machine room if traction"
        ],
        "common_variations": "Passenger, freight, service",
        "highlight_geometry": "rectangular"
    },
    
    "generic_element": {
        "vision_guidance": """
        WHAT YOU'RE LOOKING FOR: Any construction element.
        
        USE YOUR FULL VISUAL INTELLIGENCE:
        - Look at what the user is asking about
        - Use context clues from the drawing
        - Apply your construction knowledge
        - If it looks like what they're asking for, it probably is!
        - Don't be constrained by patterns - SEE what's actually there
        
        You have the intelligence to recognize construction elements even without
        specific patterns. Trust your visual understanding!
        """,
        "typical_appearance": [
            "Varies by element type",
            "Use drawing context",
            "Check labels and tags",
            "Apply common sense"
        ],
        "common_variations": "Infinite - construction has many elements",
        "highlight_geometry": "auto_detect"
    }
}

# Configuration that emphasizes intelligence over rigid rules
VISION_CONFIG = {
    "trust_gpt_vision": True,
    "allow_flexible_recognition": True,
    "use_context_clues": True,
    "apply_common_sense": True,
    "pattern_matching_weight": 0.3,  # Low weight - patterns are just hints
    "visual_intelligence_weight": 0.7,  # High weight - trust what GPT sees
}

# Philosophy for GPT-4V
VISION_PHILOSOPHY = """
You are GPT-4 Vision with incredible ability to SEE and UNDERSTAND construction drawings.

TRUST YOUR VISION:
- You can recognize elements even if they're drawn uniquely
- Use the patterns as hints, but trust what you actually SEE
- Apply your construction knowledge intelligently
- If something looks like what the user is asking for, it probably is
- You understand context - use it!

VISUAL INTELLIGENCE APPROACH:
1. LOOK at the drawing with your own eyes
2. UNDERSTAND what you're seeing using your knowledge
3. RECOGNIZE elements based on visual appearance and context
4. VERIFY using any text, labels, or schedules
5. TRUST your visual intelligence over rigid patterns

Remember: Construction drawings are made for humans to understand visually.
You have better vision than most humans - use it!
"""

# Element variations and synonyms for easier mapping
ELEMENT_VARIATIONS = {
    "receptacle": "outlet",
    "plug": "outlet",
    "power outlet": "outlet",
    "electrical outlet": "outlet",
    "duplex": "outlet",
    "lite": "light fixture",
    "lighting": "light fixture",
    "luminaire": "light fixture",
    "lamp": "light fixture",
    "electrical panel": "panel",
    "breaker panel": "panel",
    "panelboard": "panel",
    "distribution panel": "panel",
    "loadcenter": "panel",
    "toilet": "plumbing fixture",
    "wc": "plumbing fixture",
    "water closet": "plumbing fixture",
    "lavatory": "plumbing fixture",
    "lav": "plumbing fixture",
    "sink": "plumbing fixture",
    "shower": "plumbing fixture",
    "tub": "plumbing fixture",
    "bathtub": "plumbing fixture",
    "floor sink": "floor drain",
    "fd": "floor drain",
    "co": "cleanout",
    "sprinkler head": "sprinkler",
    "fire sprinkler": "sprinkler",
    "smoke alarm": "fire alarm device",
    "smoke detector": "fire alarm device",
    "pull station": "fire alarm device",
    "fire pull": "fire alarm device",
    "grille": "diffuser",
    "register": "diffuser",
    "air diffuser": "diffuser",
    "supply diffuser": "diffuser",
    "return air": "diffuser",
    "ductwork": "duct",
    "ducting": "duct",
    "hvac unit": "equipment",
    "rtu": "equipment",
    "ahu": "equipment",
    "post": "column",
    "pillar": "column",
    "pier": "column",
    "girder": "beam",
    "joist": "beam",
    "header": "beam",
    "lintel": "beam",
    "foundation": "footing",
    "footer": "footing",
    "concrete slab": "slab",
    "sog": "slab",
    "rebar": "rebar",
    "reinforcement": "rebar",
    "reinforcing": "rebar",
    "cb": "catch basin",
    "mh": "manhole",
    "parking space": "parking",
    "parking stall": "parking",
    "parking spot": "parking",
    "lift": "elevator",
    "stairway": "stair",
    "stairs": "stair",
    "staircase": "stair",
    "ej": "expansion joint",
    "exp joint": "expansion joint",
    "callout": "detail callout",
    "bubble": "detail callout",
    "section cut": "section marker",
    "section line": "section marker",
    "grid": "grid line",
    "gridline": "grid line",
    "column line": "grid line"
}

# Export all pattern-related constants
__all__ = [
    'VISUAL_PATTERNS',
    'VISION_CONFIG', 
    'VISION_PHILOSOPHY',
    'ELEMENT_VARIATIONS'
]
