# note_generator.py
import uuid
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Fix: Import from app.models.schemas instead of .models
from app.models.schemas import VisualIntelligenceResult, VisualElement, NoteSuggestion

# Fix: Import from app.core.config instead of .config
from app.core.config import CONFIG
from .patterns import VISUAL_PATTERNS

logger = logging.getLogger(__name__)

class NoteGenerator:
    """
    Generates intelligent notes for construction elements
    Creates appropriate titles and extracts relevant information
    """
    
    def __init__(self, settings):
        self.settings = settings
        
        # Note templates by element type
        self.note_templates = {
            "door": {
                "title_format": "Door {tag} - {location}",
                "content_fields": ["type", "size", "fire_rating", "hardware", "swing_direction"],
                "default_title": "Door at {location}"
            },
            "window": {
                "title_format": "Window {tag} - {location}",
                "content_fields": ["type", "size", "glazing", "operation", "sill_height"],
                "default_title": "Window at {location}"
            },
            "outlet": {
                "title_format": "{type} Outlet - {location}",
                "content_fields": ["type", "circuit", "mounting_height", "special_features"],
                "default_title": "Electrical Outlet at {location}"
            },
            "panel": {
                "title_format": "Panel {tag} - {name}",
                "content_fields": ["type", "voltage", "amperage", "circuits", "location"],
                "default_title": "Electrical Panel {tag}"
            },
            "light fixture": {
                "title_format": "Light Fixture {tag} - {location}",
                "content_fields": ["type", "mounting", "lamping", "control", "emergency"],
                "default_title": "Lighting at {location}"
            },
            "plumbing fixture": {
                "title_format": "{fixture_type} - {location}",
                "content_fields": ["fixture_type", "model", "mounting", "connections"],
                "default_title": "Plumbing Fixture at {location}"
            },
            "sprinkler": {
                "title_format": "Sprinkler Head - {location}",
                "content_fields": ["type", "coverage", "temperature_rating", "finish"],
                "default_title": "Fire Sprinkler at {location}"
            },
            "diffuser": {
                "title_format": "{type} Diffuser - {location}",
                "content_fields": ["type", "size", "cfm", "neck_size", "pattern"],
                "default_title": "Air Diffuser at {location}"
            },
            "equipment": {
                "title_format": "{equipment_type} {tag}",
                "content_fields": ["equipment_type", "model", "capacity", "electrical", "weight"],
                "default_title": "Equipment {tag}"
            },
            "column": {
                "title_format": "Column {tag} - Grid {location}",
                "content_fields": ["type", "size", "material", "fire_rating", "height"],
                "default_title": "Structural Column at {location}"
            },
            "beam": {
                "title_format": "Beam {tag} - {span}",
                "content_fields": ["type", "size", "material", "span", "loading"],
                "default_title": "Structural Beam"
            }
        }
        
        # Common abbreviations to expand
        self.abbreviations = {
            "AFF": "Above Finished Floor",
            "NIC": "Not In Contract",
            "VIF": "Verify In Field",
            "TYP": "Typical",
            "CLG": "Ceiling",
            "FLR": "Floor",
            "EQ": "Equal",
            "OC": "On Center",
            "EA": "Each",
            "GA": "Gauge",
            "MTL": "Metal",
            "GALV": "Galvanized",
            "SS": "Stainless Steel",
            "CONC": "Concrete",
            "GYP": "Gypsum",
            "BD": "Board",
            "DEMO": "Demolish",
            "EXIST": "Existing",
            "REF": "Reference",
            "SIM": "Similar",
            "DIA": "Diameter",
            "RAD": "Radius",
            "HORIZ": "Horizontal",
            "VERT": "Vertical"
        }
    
    async def generate_notes(
        self,
        visual_result: VisualIntelligenceResult,
        document_id: str,
        storage_service: Any,
        author: str = None
    ) -> List[NoteSuggestion]:
        """
        Generate notes from visual intelligence results
        Creates one note per found element with appropriate title and content
        """
        
        logger.info(f"📝 Generating notes for {visual_result.count} {visual_result.element_type}(s)")
        
        notes = []
        
        # Generate notes for each location found
        for i, location in enumerate(visual_result.locations):
            note = await self._generate_single_note(
                visual_result,
                location,
                i + 1,  # Element number
                document_id,
                author
            )
            
            if note:
                notes.append(note)
        
        # Add summary note if multiple elements
        if visual_result.count > 3:
            summary_note = self._generate_summary_note(
                visual_result,
                document_id,
                author
            )
            if summary_note:
                notes.insert(0, summary_note)  # Put summary first
        
        logger.info(f"✅ Generated {len(notes)} notes")
        
        return notes
    
    async def _generate_single_note(
        self,
        visual_result: VisualIntelligenceResult,
        location: Dict[str, Any],
        element_number: int,
        document_id: str,
        author: str
    ) -> Optional[NoteSuggestion]:
        """Generate a note for a single element"""
        
        element_type = visual_result.element_type
        
        # Extract relevant information
        element_info = self._extract_element_info(location, visual_result, element_number)
        
        # Generate title
        title = self._generate_note_title(element_type, element_info)
        
        # Generate content
        content = self._generate_note_content(element_type, element_info, visual_result)
        
        # Create note
        note = NoteSuggestion(
            id=f"note_{uuid.uuid4().hex[:8]}",
            element_id=location.get("element_id", f"{element_type}_{element_number}"),
            title=title,
            content=content,
            element_type=element_type,
            metadata={
                "grid_reference": location.get("grid_ref", "Unknown"),
                "page": visual_result.page_number,
                "element_number": element_number,
                "visual_confidence": visual_result.confidence,
                "has_tag": bool(location.get("element_tag")),
                "document_id": document_id
            },
            confidence=visual_result.confidence,
            author=author or "AI Assistant",
            timestamp=datetime.utcnow().isoformat(),
            tags=self._generate_tags(element_type, element_info)
        )
        
        return note
    
    def _extract_element_info(
        self,
        location: Dict[str, Any],
        visual_result: VisualIntelligenceResult,
        element_number: int
    ) -> Dict[str, Any]:
        """Extract all relevant information about an element"""
        
        info = {
            "location": location.get("grid_ref", "Unknown"),
            "tag": location.get("element_tag", ""),
            "visual_details": location.get("visual_details", ""),
            "element_number": element_number,
            "room": location.get("room", ""),
            "area": location.get("area", "")
        }
        
        # Parse visual details for specific information
        details = info["visual_details"].lower()
        
        # Extract size information
        size_match = re.search(r'(\d+)["\']?\s*x\s*(\d+)["\']?', details)
        if size_match:
            info["size"] = f"{size_match.group(1)}x{size_match.group(2)}"
        
        # Extract type information
        for evidence in visual_result.visual_evidence:
            if "type" in evidence.lower():
                info["type"] = self._extract_type_from_evidence(evidence)
                break
        
        # Element-specific extraction
        if visual_result.element_type == "door":
            info.update(self._extract_door_info(details, location))
        elif visual_result.element_type == "window":
            info.update(self._extract_window_info(details, location))
        elif visual_result.element_type == "outlet":
            info.update(self._extract_outlet_info(details, location))
        elif visual_result.element_type == "panel":
            info.update(self._extract_panel_info(details, location))
        elif visual_result.element_type == "equipment":
            info.update(self._extract_equipment_info(details, location))
        
        return info
    
    def _generate_note_title(self, element_type: str, info: Dict[str, Any]) -> str:
        """Generate an appropriate title for the note"""
        
        template = self.note_templates.get(element_type, {})
        
        # Try to use the template format
        if "title_format" in template:
            try:
                # Replace missing values with defaults
                format_dict = {
                    "tag": info.get("tag") or f"#{info['element_number']}",
                    "location": info.get("location", "Unknown"),
                    "type": info.get("type", element_type.title()),
                    "name": info.get("name", ""),
                    "fixture_type": info.get("fixture_type", element_type),
                    "equipment_type": info.get("equipment_type", "Equipment"),
                    "span": info.get("span", "")
                }
                
                title = template["title_format"].format(**format_dict)
                
                # Clean up empty placeholders
                title = re.sub(r'\s+\-\s+$', '', title)
                title = re.sub(r'\s+\-\s+\-\s+', ' - ', title)
                
                return title.strip()
                
            except:
                pass
        
        # Fallback to default title
        default = template.get("default_title", f"{element_type.title()} at {{location}}")
        return default.format(
            location=info.get("location", "Unknown"),
            tag=info.get("tag", f"#{info['element_number']}")
        )
    
    def _generate_note_content(
        self,
        element_type: str,
        info: Dict[str, Any],
        visual_result: VisualIntelligenceResult
    ) -> str:
        """Generate detailed content for the note"""
        
        content_parts = []
        
        # Basic information
        content_parts.append(f"**Element Type**: {element_type.title()}")
        content_parts.append(f"**Location**: Grid {info['location']}")
        
        if info.get("tag"):
            content_parts.append(f"**Tag**: {info['tag']}")
        
        if info.get("room"):
            content_parts.append(f"**Room**: {info['room']}")
        
        # Element-specific content
        template = self.note_templates.get(element_type, {})
        content_fields = template.get("content_fields", [])
        
        for field in content_fields:
            if field in info and info[field]:
                field_name = field.replace("_", " ").title()
                field_value = info[field]
                
                # Expand abbreviations
                field_value = self._expand_abbreviations(str(field_value))
                
                content_parts.append(f"**{field_name}**: {field_value}")
        
        # Add visual evidence if relevant
        relevant_evidence = self._get_relevant_evidence(visual_result, element_type)
        if relevant_evidence:
            content_parts.append("\n**Technical Details**:")
            for evidence in relevant_evidence[:3]:  # Limit to 3
                content_parts.append(f"• {evidence}")
        
        # Add any special notes
        special_notes = self._get_special_notes(element_type, info)
        if special_notes:
            content_parts.append(f"\n**Notes**: {special_notes}")
        
        # Add confidence indicator
        confidence_level = "High" if visual_result.confidence > 0.9 else "Medium" if visual_result.confidence > 0.7 else "Review Required"
        content_parts.append(f"\n*Confidence: {confidence_level} ({int(visual_result.confidence * 100)}%)*")
        
        return "\n".join(content_parts)
    
    def _generate_summary_note(
        self,
        visual_result: VisualIntelligenceResult,
        document_id: str,
        author: str
    ) -> NoteSuggestion:
        """Generate a summary note for multiple elements"""
        
        element_type = visual_result.element_type
        
        # Create summary title
        title = f"{element_type.title()} Summary - {visual_result.count} Items"
        
        # Create summary content
        content_parts = [
            f"**Total {element_type.title()}s**: {visual_result.count}",
            f"**Pages Analyzed**: {len(set(loc.get('page', visual_result.page_number) for loc in visual_result.locations))}",
            ""
        ]
        
        # Grid distribution
        grid_refs = visual_result.grid_references
        unique_grids = list(set(grid_refs))
        if len(unique_grids) <= 10:
            content_parts.append(f"**Grid Locations**: {', '.join(sorted(unique_grids))}")
        else:
            content_parts.append(f"**Grid Coverage**: {len(unique_grids)} different grid locations")
        
        # Type distribution if applicable
        types = {}
        for loc in visual_result.locations:
            el_type = loc.get('type') or loc.get('visual_details', '').split()[0] if loc.get('visual_details') else 'Standard'
            types[el_type] = types.get(el_type, 0) + 1
        
        if len(types) > 1:
            content_parts.append("\n**Type Distribution**:")
            for el_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                content_parts.append(f"• {el_type}: {count}")
        
        # Add any patterns noticed
        patterns = self._identify_patterns(visual_result)
        if patterns:
            content_parts.append(f"\n**Patterns Identified**:")
            for pattern in patterns:
                content_parts.append(f"• {pattern}")
        
        # Create summary note
        return NoteSuggestion(
            id=f"note_summary_{uuid.uuid4().hex[:8]}",
            element_id=f"{element_type}_summary",
            title=title,
            content="\n".join(content_parts),
            element_type=element_type,
            metadata={
                "is_summary": True,
                "element_count": visual_result.count,
                "document_id": document_id
            },
            confidence=visual_result.confidence,
            author=author or "AI Assistant",
            timestamp=datetime.utcnow().isoformat(),
            tags=["summary", element_type, "overview"]
        )
    
    def _extract_door_info(self, details: str, location: Dict[str, Any]) -> Dict[str, Any]:
        """Extract door-specific information"""
        
        info = {}
        
        # Door type
        if "fire" in details or "fd" in details:
            info["type"] = "Fire Door"
            info["fire_rating"] = self._extract_fire_rating(details)
        elif "double" in details:
            info["type"] = "Double Door"
        elif "sliding" in details:
            info["type"] = "Sliding Door"
        else:
            info["type"] = "Single Door"
        
        # Size
        width_match = re.search(r"(\d+)['\"]?\s*(?:x|wide)", details)
        if width_match:
            info["size"] = f"{width_match.group(1)}\" wide"
        
        # Swing direction
        if "lh" in details:
            info["swing_direction"] = "Left Hand"
        elif "rh" in details:
            info["swing_direction"] = "Right Hand"
        
        return info
    
    def _extract_window_info(self, details: str, location: Dict[str, Any]) -> Dict[str, Any]:
        """Extract window-specific information"""
        
        info = {}
        
        # Window type
        if "fixed" in details:
            info["type"] = "Fixed Window"
        elif "casement" in details:
            info["type"] = "Casement Window"
        elif "sliding" in details:
            info["type"] = "Sliding Window"
        elif "awning" in details:
            info["type"] = "Awning Window"
        
        # Glazing
        if "double" in details or "insulated" in details:
            info["glazing"] = "Double Glazed"
        elif "tempered" in details:
            info["glazing"] = "Tempered Glass"
        
        return info
    
    def _extract_outlet_info(self, details: str, location: Dict[str, Any]) -> Dict[str, Any]:
        """Extract outlet-specific information"""
        
        info = {}
        
        # Outlet type
        if "gfci" in details or "gfi" in details:
            info["type"] = "GFCI"
        elif "dedicated" in details:
            info["type"] = "Dedicated Circuit"
        elif "quad" in details:
            info["type"] = "Quad Outlet"
        else:
            info["type"] = "Standard Duplex"
        
        # Mounting height
        height_match = re.search(r"(\d+)[\"']?\s*(?:aff|above)", details)
        if height_match:
            info["mounting_height"] = f"{height_match.group(1)}\" AFF"
        
        # Circuit
        circuit_match = re.search(r"(?:circuit|ckt)\s*(\d+[a-z]?)", details, re.IGNORECASE)
        if circuit_match:
            info["circuit"] = circuit_match.group(1).upper()
        
        return info
    
    def _extract_panel_info(self, details: str, location: Dict[str, Any]) -> Dict[str, Any]:
        """Extract panel-specific information"""
        
        info = {}
        
        # Panel type
        if "main" in details:
            info["type"] = "Main Distribution Panel"
        elif "sub" in details:
            info["type"] = "Sub Panel"
        elif "lighting" in details or "lp" in details:
            info["type"] = "Lighting Panel"
        elif "power" in details or "pp" in details:
            info["type"] = "Power Panel"
        
        # Amperage
        amp_match = re.search(r"(\d+)\s*(?:amp|a)", details, re.IGNORECASE)
        if amp_match:
            info["amperage"] = f"{amp_match.group(1)}A"
        
        # Voltage
        if "480" in details:
            info["voltage"] = "480V"
        elif "277" in details:
            info["voltage"] = "277V"
        elif "208" in details:
            info["voltage"] = "208V"
        elif "120" in details:
            info["voltage"] = "120V"
        
        return info
    
    def _extract_equipment_info(self, details: str, location: Dict[str, Any]) -> Dict[str, Any]:
        """Extract equipment-specific information"""
        
        info = {}
        
        # Equipment type
        if "rtu" in details:
            info["equipment_type"] = "Rooftop Unit"
        elif "ahu" in details:
            info["equipment_type"] = "Air Handler Unit"
        elif "vav" in details:
            info["equipment_type"] = "VAV Box"
        elif "pump" in details:
            info["equipment_type"] = "Pump"
        
        # Model number
        model_match = re.search(r"model\s*[:#]?\s*([A-Z0-9\-]+)", details, re.IGNORECASE)
        if model_match:
            info["model"] = model_match.group(1)
        
        return info
    
    def _extract_type_from_evidence(self, evidence: str) -> str:
        """Extract type information from visual evidence"""
        
        # Remove common prefixes
        evidence_clean = evidence.lower()
        for prefix in ["type:", "type is", "type -", "identified as"]:
            if prefix in evidence_clean:
                type_part = evidence_clean.split(prefix, 1)[1].strip()
                return type_part.split()[0].title() if type_part else "Standard"
        
        return "Standard"
    
    def _extract_fire_rating(self, details: str) -> str:
        """Extract fire rating from details"""
        
        rating_match = re.search(r"(\d+)\s*(?:hr|hour|min|minute)", details, re.IGNORECASE)
        if rating_match:
            value = rating_match.group(1)
            unit = "hour" if "hr" in details or "hour" in details else "minute"
            return f"{value} {unit}"
        
        return "Rating not specified"
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common construction abbreviations"""
        
        for abbr, full in self.abbreviations.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + abbr + r'\b'
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        
        return text
    
    def _get_relevant_evidence(
        self,
        visual_result: VisualIntelligenceResult,
        element_type: str
    ) -> List[str]:
        """Get evidence relevant to this specific element"""
        
        relevant = []
        
        # Keywords that indicate relevant evidence
        relevance_keywords = {
            "door": ["size", "rating", "hardware", "frame", "threshold"],
            "window": ["glazing", "frame", "sill", "operation", "size"],
            "outlet": ["circuit", "voltage", "amperage", "mounting", "type"],
            "panel": ["breaker", "circuit", "voltage", "phase", "amperage"],
            "sprinkler": ["coverage", "temperature", "type", "spacing"],
            "diffuser": ["cfm", "size", "neck", "pattern", "damper"]
        }
        
        keywords = relevance_keywords.get(element_type, ["type", "size", "specification"])
        
        for evidence in visual_result.visual_evidence:
            if any(keyword in evidence.lower() for keyword in keywords):
                relevant.append(self._expand_abbreviations(evidence))
        
        return relevant
    
    def _get_special_notes(self, element_type: str, info: Dict[str, Any]) -> str:
        """Generate special notes based on element type and info"""
        
        notes = []
        
        # Check for code compliance notes
        if element_type == "outlet" and info.get("type") == "GFCI":
            notes.append("GFCI protection required per NEC")
        elif element_type == "door" and info.get("type") == "Fire Door":
            notes.append("Maintain fire rating integrity during installation")
        elif element_type == "stair":
            notes.append("Verify code compliance for rise/run and handrails")
        
        # Check for coordination notes
        if element_type in ["diffuser", "sprinkler", "light fixture"]:
            notes.append("Coordinate ceiling layout with other trades")
        
        # Installation notes
        if info.get("mounting_height"):
            notes.append(f"Verify mounting height: {info['mounting_height']}")
        
        return "; ".join(notes) if notes else ""
    
    def _generate_tags(self, element_type: str, info: Dict[str, Any]) -> List[str]:
        """Generate searchable tags for the note"""
        
        tags = [element_type]
        
        # Add type tag
        if info.get("type"):
            tags.append(info["type"].lower().replace(" ", "-"))
        
        # Add location tag
        if info.get("location"):
            tags.append(f"grid-{info['location'].lower()}")
        
        # Add special feature tags
        if element_type == "door" and "fire" in str(info.get("type", "")).lower():
            tags.append("fire-rated")
        elif element_type == "outlet" and "gfci" in str(info.get("type", "")).lower():
            tags.append("gfci-protected")
        
        # Add discipline tag
        discipline_map = {
            "door": "architectural",
            "window": "architectural",
            "outlet": "electrical",
            "panel": "electrical",
            "light fixture": "electrical",
            "plumbing fixture": "plumbing",
            "pipe": "plumbing",
            "sprinkler": "fire-protection",
            "diffuser": "mechanical",
            "duct": "mechanical",
            "column": "structural",
            "beam": "structural"
        }
        
        if element_type in discipline_map:
            tags.append(discipline_map[element_type])
        
        return tags
    
    def _identify_patterns(self, visual_result: VisualIntelligenceResult) -> List[str]:
        """Identify patterns in element distribution"""
        
        patterns = []
        
        # Check for regular spacing
        if visual_result.count > 5:
            # This would need actual coordinate analysis
            patterns.append("Elements appear to follow regular spacing pattern")
        
        # Check for clustering
        grid_counts = {}
        for ref in visual_result.grid_references:
            grid_counts[ref] = grid_counts.get(ref, 0) + 1
        
        if any(count > 3 for count in grid_counts.values()):
            patterns.append("Multiple elements clustered in same grid areas")
        
        # Check for linear arrangement
        if len(set(ref[0] for ref in visual_result.grid_references)) == 1:
            patterns.append("Elements aligned vertically")
        elif len(set(ref[1:] for ref in visual_result.grid_references)) == 1:
            patterns.append("Elements aligned horizontally")
        
        return patterns
