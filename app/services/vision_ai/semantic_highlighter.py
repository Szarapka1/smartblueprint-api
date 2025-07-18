# semantic_highlighter.py
import math
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Fix: Import from app.models.schemas instead of .models
from app.models.schemas import (
    VisualIntelligenceResult, 
    ElementGeometry, 
    SemanticHighlight
)

# Fix: Import from app.core.config instead of .config
from .patterns import VISUAL_PATTERNS

logger = logging.getLogger(__name__)

class SemanticHighlighter:
    """
    Advanced highlighter using semantic spatial intelligence
    Generates precise highlights based on element geometry
    """
    
    def __init__(self):
        # Geometry pattern generators
        self.geometry_patterns = {
            "door_with_swing": self._generate_door_highlight,
            "linear_path": self._generate_linear_highlight,
            "circular_symbol": self._generate_circular_highlight,
            "rectangular": self._generate_rectangular_highlight,
            "rectangular_with_sill": self._generate_window_highlight,
            "curved_path": self._generate_curved_highlight,
            "symbol_based": self._generate_symbol_highlight,
            "square_with_pattern": self._generate_patterned_square_highlight,
            "triangular_symbol": self._generate_triangular_highlight,
            "composite": self._generate_composite_highlight,
            "auto_detect": self._auto_detect_geometry
        }
        
        # Default highlight styles
        self.default_styles = {
            "stroke_width": 3.0,
            "stroke_color": "#FFD700",  # Gold
            "fill_opacity": 0.2,
            "secondary_stroke_color": "#FF6B6B",  # Red for warnings
            "tertiary_stroke_color": "#4ECDC4"   # Teal for info
        }
        
        # Element-specific style overrides
        self.element_styles = {
            "door": {"stroke_color": "#FF6B6B", "stroke_width": 2.5},
            "window": {"stroke_color": "#4ECDC4", "stroke_width": 2.5},
            "outlet": {"stroke_color": "#FFE66D", "stroke_width": 2.0},
            "sprinkler": {"stroke_color": "#FF6B6B", "stroke_width": 2.0},
            "column": {"stroke_color": "#95E1D3", "fill_opacity": 0.3},
            "beam": {"stroke_color": "#95E1D3", "stroke_width": 4.0},
            "pipe": {"stroke_color": "#3D5A80", "stroke_width": 3.0}
        }
    
    async def generate_highlights(
        self,
        visual_result: VisualIntelligenceResult,
        images: List[Dict[str, Any]],
        comprehensive_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Main method to generate highlights for visual results
        Returns highlight data ready for frontend rendering
        """
        
        logger.info(f"🎨 Generating semantic highlights for {visual_result.count} {visual_result.element_type}(s)")
        
        # First, detect element geometries if not already done
        if not visual_result.element_geometries:
            element_geometries = await self._detect_element_geometries(
                visual_result, images, comprehensive_data
            )
            visual_result.element_geometries = element_geometries
        
        # Generate semantic highlights
        semantic_highlights = await self._generate_semantic_highlights(
            visual_result, visual_result.element_geometries
        )
        visual_result.semantic_highlights = semantic_highlights
        
        # Convert to frontend format
        highlights = self._convert_to_frontend_format(
            semantic_highlights, visual_result
        )
        
        logger.info(f"✅ Generated {len(highlights)} highlights")
        
        return highlights
    
    async def _detect_element_geometries(
        self,
        visual_result: VisualIntelligenceResult,
        images: List[Dict[str, Any]],
        comprehensive_data: Dict[str, Any]
    ) -> List[ElementGeometry]:
        """
        Detect precise element geometries from visual result
        This would normally call the vision API for geometry detection
        """
        
        # For now, generate geometries based on locations
        geometries = []
        
        for i, location in enumerate(visual_result.locations):
            geometry = self._create_geometry_from_location(
                location, visual_result.element_type, i
            )
            geometries.append(geometry)
        
        return geometries
    
    async def _generate_semantic_highlights(
        self,
        visual_result: VisualIntelligenceResult,
        element_geometries: List[ElementGeometry]
    ) -> List[SemanticHighlight]:
        """Generate semantic highlights for detected elements"""
        
        highlights = []
        
        for i, geometry in enumerate(element_geometries):
            if i >= len(visual_result.locations):
                break
            
            location = visual_result.locations[i]
            
            # Determine highlight generation method
            geometry_type = VISUAL_PATTERNS.get(
                visual_result.element_type, {}
            ).get("highlight_geometry", "auto_detect")
            
            # Get the appropriate generator
            generator = self.geometry_patterns.get(
                geometry_type,
                self._auto_detect_geometry
            )
            
            # Generate the highlight
            highlight = generator(geometry, location, visual_result)
            
            # Set unique ID and metadata
            highlight.element_id = f"{visual_result.element_type}_{i}_{uuid.uuid4().hex[:8]}"
            highlight.confidence = visual_result.confidence
            highlight.page = visual_result.page_number
            
            # Apply element-specific styling
            self._apply_element_styling(highlight, visual_result.element_type)
            
            highlights.append(highlight)
        
        return highlights
    
    def _generate_door_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate highlight for door with swing arc"""
        
        center = geometry.center_point
        width = geometry.dimensions.get("width", 36)  # Default 3' door
        thickness = 4  # Door thickness
        
        # Door opening (rectangular)
        opening_points = [
            {"x": center["x"] - width/2, "y": center["y"] - thickness/2},
            {"x": center["x"] + width/2, "y": center["y"] - thickness/2},
            {"x": center["x"] + width/2, "y": center["y"] + thickness/2},
            {"x": center["x"] - width/2, "y": center["y"] + thickness/2},
            {"x": center["x"] - width/2, "y": center["y"] - thickness/2}  # Close path
        ]
        
        # Swing arc (quarter circle)
        arc_points = []
        swing_radius = width
        swing_direction = geometry.special_features.get("swing_direction", "right")
        
        # Determine arc angles based on swing direction
        if swing_direction == "right":
            start_angle = 0
            end_angle = 90
        elif swing_direction == "left":
            start_angle = 90
            end_angle = 180
        elif swing_direction == "up":
            start_angle = 270
            end_angle = 360
        else:  # down
            start_angle = 180
            end_angle = 270
        
        # Generate arc points
        for angle in range(start_angle, end_angle + 1, 5):
            rad = math.radians(angle)
            arc_points.append({
                "x": center["x"] + swing_radius * math.cos(rad),
                "y": center["y"] + swing_radius * math.sin(rad)
            })
        
        # Combine paths with separator
        all_points = opening_points + [{"x": -1, "y": -1}] + arc_points
        
        return SemanticHighlight(
            element_id="",
            element_type="door",
            geometry_type="door_with_swing",
            path_points=all_points,
            path_type="composite",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_window_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate highlight for window with sill projection"""
        
        center = geometry.center_point
        width = geometry.dimensions.get("width", 48)  # Default 4' window
        height = geometry.dimensions.get("height", 36)  # Default 3' height
        sill_projection = 4  # Sill extends beyond wall
        
        # Window frame
        frame_points = [
            {"x": center["x"] - width/2, "y": center["y"] - height/2},
            {"x": center["x"] + width/2, "y": center["y"] - height/2},
            {"x": center["x"] + width/2, "y": center["y"] + height/2},
            {"x": center["x"] - width/2, "y": center["y"] + height/2},
            {"x": center["x"] - width/2, "y": center["y"] - height/2}
        ]
        
        # Sill line (extends beyond frame)
        sill_points = [
            {"x": center["x"] - width/2 - sill_projection, "y": center["y"] + height/2},
            {"x": center["x"] + width/2 + sill_projection, "y": center["y"] + height/2}
        ]
        
        # Combine with separator
        all_points = frame_points + [{"x": -1, "y": -1}] + sill_points
        
        return SemanticHighlight(
            element_id="",
            element_type="window",
            geometry_type="rectangular_with_sill",
            path_points=all_points,
            path_type="composite",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_linear_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate highlight for linear elements (pipes, ducts, conduits)"""
        
        # Use boundary points if available
        if geometry.boundary_points and len(geometry.boundary_points) >= 2:
            path_points = geometry.boundary_points
        else:
            # Generate simple line
            center = geometry.center_point
            length = geometry.dimensions.get("length", 100)
            angle = geometry.orientation
            
            rad = math.radians(angle)
            dx = length * math.cos(rad) / 2
            dy = length * math.sin(rad) / 2
            
            path_points = [
                {"x": center["x"] - dx, "y": center["y"] - dy},
                {"x": center["x"] + dx, "y": center["y"] + dy}
            ]
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="linear_path",
            path_points=path_points,
            path_type="polyline",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_circular_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate circular highlight for outlets, fixtures, etc."""
        
        center = geometry.center_point
        radius = geometry.dimensions.get("radius", 10)
        
        # Generate circle points
        path_points = []
        for angle in range(0, 361, 10):
            rad = math.radians(angle)
            path_points.append({
                "x": center["x"] + radius * math.cos(rad),
                "y": center["y"] + radius * math.sin(rad)
            })
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="circular_symbol",
            path_points=path_points,
            path_type="polygon",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_rectangular_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate rectangular highlight"""
        
        center = geometry.center_point
        width = geometry.dimensions.get("width", 50)
        height = geometry.dimensions.get("height", 30)
        rotation = geometry.orientation
        
        # Generate rectangle points (before rotation)
        half_width = width / 2
        half_height = height / 2
        
        corners = [
            {"x": -half_width, "y": -half_height},
            {"x": half_width, "y": -half_height},
            {"x": half_width, "y": half_height},
            {"x": -half_width, "y": half_height},
            {"x": -half_width, "y": -half_height}  # Close path
        ]
        
        # Apply rotation if needed
        if rotation != 0:
            rad = math.radians(rotation)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            
            path_points = []
            for corner in corners:
                rotated_x = corner["x"] * cos_r - corner["y"] * sin_r
                rotated_y = corner["x"] * sin_r + corner["y"] * cos_r
                path_points.append({
                    "x": center["x"] + rotated_x,
                    "y": center["y"] + rotated_y
                })
        else:
            # No rotation, just translate
            path_points = [
                {"x": center["x"] + corner["x"], "y": center["y"] + corner["y"]}
                for corner in corners
            ]
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="rectangular",
            path_points=path_points,
            path_type="polygon",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_curved_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate highlight for curved elements"""
        
        # Use boundary points if available
        if geometry.boundary_points and len(geometry.boundary_points) > 2:
            path_points = geometry.boundary_points
        else:
            # Generate default arc
            center = geometry.center_point
            radius = geometry.dimensions.get("radius", 50)
            start_angle = geometry.special_features.get("start_angle", 0)
            end_angle = geometry.special_features.get("end_angle", 90)
            
            path_points = []
            angle_step = 5 if abs(end_angle - start_angle) > 45 else 2
            
            current_angle = start_angle
            while current_angle <= end_angle:
                rad = math.radians(current_angle)
                path_points.append({
                    "x": center["x"] + radius * math.cos(rad),
                    "y": center["y"] + radius * math.sin(rad)
                })
                current_angle += angle_step
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="curved_path",
            path_points=path_points,
            path_type="polyline",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_symbol_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate highlight for symbol-based elements"""
        
        # Try to use boundary points for custom symbols
        if geometry.boundary_points and len(geometry.boundary_points) >= 3:
            path_points = geometry.boundary_points
        else:
            # Fall back to element-specific symbol
            return self._generate_element_specific_symbol(
                geometry, location, visual_result
            )
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="symbol_based",
            path_points=path_points,
            path_type="polygon",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_patterned_square_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate square with pattern (for diffusers, etc.)"""
        
        center = geometry.center_point
        size = geometry.dimensions.get("width", 24)  # Default 24" diffuser
        
        # Outer square
        half_size = size / 2
        outer_points = [
            {"x": center["x"] - half_size, "y": center["y"] - half_size},
            {"x": center["x"] + half_size, "y": center["y"] - half_size},
            {"x": center["x"] + half_size, "y": center["y"] + half_size},
            {"x": center["x"] - half_size, "y": center["y"] + half_size},
            {"x": center["x"] - half_size, "y": center["y"] - half_size}
        ]
        
        # Add pattern (cross for supply diffuser)
        pattern_points = []
        if visual_result.element_type == "diffuser":
            # Cross pattern
            cross_size = size * 0.6
            pattern_points = [
                {"x": -1, "y": -1},  # Separator
                {"x": center["x"] - cross_size/2, "y": center["y"]},
                {"x": center["x"] + cross_size/2, "y": center["y"]},
                {"x": -1, "y": -1},  # Separator
                {"x": center["x"], "y": center["y"] - cross_size/2},
                {"x": center["x"], "y": center["y"] + cross_size/2}
            ]
        
        all_points = outer_points + pattern_points
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="square_with_pattern",
            path_points=all_points,
            path_type="composite",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_triangular_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate triangular highlight (for fire extinguishers, etc.)"""
        
        center = geometry.center_point
        size = geometry.dimensions.get("width", 20)
        
        # Equilateral triangle
        height = size * math.sqrt(3) / 2
        
        path_points = [
            {"x": center["x"], "y": center["y"] - height * 2/3},  # Top
            {"x": center["x"] - size/2, "y": center["y"] + height * 1/3},  # Bottom left
            {"x": center["x"] + size/2, "y": center["y"] + height * 1/3},  # Bottom right
            {"x": center["x"], "y": center["y"] - height * 2/3}  # Close
        ]
        
        return SemanticHighlight(
            element_id="",
            element_type=visual_result.element_type,
            geometry_type="triangular_symbol",
            path_points=path_points,
            path_type="polygon",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_composite_highlight(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate composite highlight for complex elements"""
        
        # This would handle complex multi-part elements
        # For now, defer to auto-detect
        return self._auto_detect_geometry(geometry, location, visual_result)
    
    def _auto_detect_geometry(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Auto-detect and generate appropriate highlight"""
        
        # If we have boundary points, use them
        if geometry.boundary_points and len(geometry.boundary_points) >= 3:
            return SemanticHighlight(
                element_id="",
                element_type=visual_result.element_type,
                geometry_type="auto_detect",
                path_points=geometry.boundary_points,
                path_type="polygon",
                visual_description=location.get("visual_details", ""),
                page=1
            )
        
        # Otherwise, create a default highlight based on element type
        if visual_result.element_type in ["outlet", "switch", "sprinkler"]:
            return self._generate_circular_highlight(geometry, location, visual_result)
        elif visual_result.element_type in ["pipe", "duct", "conduit", "beam"]:
            return self._generate_linear_highlight(geometry, location, visual_result)
        else:
            return self._generate_rectangular_highlight(geometry, location, visual_result)
    
    def _generate_element_specific_symbol(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate element-specific symbols"""
        
        element_type = visual_result.element_type
        
        if element_type == "outlet":
            return self._generate_outlet_symbol(geometry, location, visual_result)
        elif element_type == "switch":
            return self._generate_switch_symbol(geometry, location, visual_result)
        elif element_type == "sprinkler":
            return self._generate_sprinkler_symbol(geometry, location, visual_result)
        else:
            # Default to circular
            return self._generate_circular_highlight(geometry, location, visual_result)
    
    def _generate_outlet_symbol(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate outlet symbol (circle with lines)"""
        
        center = geometry.center_point
        radius = 10
        
        # Circle
        circle_points = []
        for angle in range(0, 361, 30):
            rad = math.radians(angle)
            circle_points.append({
                "x": center["x"] + radius * math.cos(rad),
                "y": center["y"] + radius * math.sin(rad)
            })
        
        # Add outlet lines
        line_length = radius * 0.7
        outlet_lines = [
            {"x": -1, "y": -1},  # Separator
            {"x": center["x"] - line_length/2, "y": center["y"] - 3},
            {"x": center["x"] - line_length/2, "y": center["y"] + 3},
            {"x": -1, "y": -1},  # Separator
            {"x": center["x"] + line_length/2, "y": center["y"] - 3},
            {"x": center["x"] + line_length/2, "y": center["y"] + 3}
        ]
        
        all_points = circle_points + outlet_lines
        
        return SemanticHighlight(
            element_id="",
            element_type="outlet",
            geometry_type="symbol_based",
            path_points=all_points,
            path_type="composite",
            visual_description=location.get("visual_details", ""),
            page=1
        )
    
    def _generate_switch_symbol(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate switch symbol"""
        # Implementation would go here
        # For now, use circular
        return self._generate_circular_highlight(geometry, location, visual_result)
    
    def _generate_sprinkler_symbol(
        self,
        geometry: ElementGeometry,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> SemanticHighlight:
        """Generate sprinkler symbol"""
        # Implementation would go here
        # For now, use circular
        return self._generate_circular_highlight(geometry, location, visual_result)
    
    def _create_geometry_from_location(
        self,
        location: Dict[str, Any],
        element_type: str,
        index: int
    ) -> ElementGeometry:
        """Create basic geometry from location data"""
        
        # Generate a basic position
        base_x = 100 + (index % 5) * 200
        base_y = 100 + (index // 5) * 150
        
        geometry = ElementGeometry(
            element_type=element_type,
            geometry_type="auto_detect",
            center_point={"x": base_x, "y": base_y},
            boundary_points=[]
        )
        
        # Set element-specific dimensions
        dimension_map = {
            "door": {"width": 36, "height": 4},
            "window": {"width": 48, "height": 36},
            "outlet": {"radius": 10},
            "switch": {"radius": 10},
            "panel": {"width": 24, "height": 36},
            "light fixture": {"width": 48, "height": 24},
            "plumbing fixture": {"width": 30, "height": 20},
            "sprinkler": {"radius": 8},
            "diffuser": {"width": 24, "height": 24},
            "column": {"width": 24, "height": 24},
            "beam": {"length": 120, "width": 12}
        }
        
        if element_type in dimension_map:
            geometry.dimensions = dimension_map[element_type]
        else:
            geometry.dimensions = {"width": 40, "height": 30}
        
        return geometry
    
    def _apply_element_styling(self, highlight: SemanticHighlight, element_type: str):
        """Apply element-specific styling to highlight"""
        
        # Get element-specific styles
        styles = self.element_styles.get(element_type, {})
        
        # Apply styles
        if "stroke_color" in styles:
            highlight.stroke_color = styles["stroke_color"]
        if "stroke_width" in styles:
            highlight.stroke_width = styles["stroke_width"]
        if "fill_opacity" in styles:
            highlight.fill_opacity = styles["fill_opacity"]
    
    def _convert_to_frontend_format(
        self,
        semantic_highlights: List[SemanticHighlight],
        visual_result: VisualIntelligenceResult
    ) -> List[Dict[str, Any]]:
        """Convert semantic highlights to frontend-ready format"""
        
        highlights = []
        
        for highlight in semantic_highlights:
            frontend_highlight = {
                "id": f"hl_{highlight.element_id}",
                "type": "semantic_spatial",
                "element_type": highlight.element_type,
                "geometry_type": highlight.geometry_type,
                "path_data": self._generate_svg_path(highlight),
                "path_points": highlight.path_points,
                "path_type": highlight.path_type,
                "style": {
                    "stroke": highlight.stroke_color,
                    "strokeWidth": highlight.stroke_width,
                    "fillOpacity": highlight.fill_opacity,
                    "fill": highlight.stroke_color
                },
                "confidence": highlight.confidence,
                "visual_description": highlight.visual_description,
                "page": highlight.page,
                "timestamp": visual_result.page_number  # Will be set by caller
            }
            
            # Add interaction data
            frontend_highlight["interaction"] = {
                "hoverable": True,
                "clickable": True,
                "tooltip": self._generate_tooltip(highlight, visual_result)
            }
            
            highlights.append(frontend_highlight)
        
        return highlights
    
    def _generate_svg_path(self, highlight: SemanticHighlight) -> str:
        """Generate SVG path data from points"""
        
        if not highlight.path_points:
            return ""
        
        path_parts = []
        current_path = []
        
        for point in highlight.path_points:
            if point["x"] == -1 and point["y"] == -1:
                # Separator - finish current path
                if current_path:
                    path_parts.append(self._points_to_svg_path(current_path, highlight.path_type))
                    current_path = []
            else:
                current_path.append(point)
        
        # Add final path
        if current_path:
            path_parts.append(self._points_to_svg_path(current_path, highlight.path_type))
        
        return " ".join(path_parts)
    
    def _points_to_svg_path(self, points: List[Dict[str, float]], path_type: str) -> str:
        """Convert points to SVG path string"""
        
        if not points:
            return ""
        
        # Start with move to first point
        path = f"M {points[0]['x']} {points[0]['y']}"
        
        if path_type == "polygon" or path_type == "polyline":
            # Line to each subsequent point
            for point in points[1:]:
                path += f" L {point['x']} {point['y']}"
            
            if path_type == "polygon":
                path += " Z"  # Close path
        
        return path
    
    def _generate_tooltip(self, highlight: SemanticHighlight, visual_result: VisualIntelligenceResult) -> str:
        """Generate tooltip text for highlight"""
        
        tooltip_parts = [
            f"{highlight.element_type.title()}",
            f"Confidence: {int(highlight.confidence * 100)}%"
        ]
        
        if highlight.visual_description:
            tooltip_parts.append(highlight.visual_description)
        
        return " | ".join(tooltip_parts)
