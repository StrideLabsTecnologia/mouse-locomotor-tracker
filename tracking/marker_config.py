"""
Marker Configuration Module

Defines marker sets for different mouse tracking views and provides
utilities for loading custom configurations from YAML files.

Author: Stride Labs
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarkerSet:
    """
    Configuration for a set of body part markers used in tracking.

    Attributes:
        name: Descriptive name for the marker set (e.g., 'mouse_ventral')
        markers: List of marker/body part names
        connections: List of tuples defining connections between markers for skeleton
        description: Optional description of the marker set
        reference_marker: Primary marker used for position reference (default: first marker)

    Example:
        >>> marker_set = MarkerSet(
        ...     name="simple",
        ...     markers=["head", "body", "tail"],
        ...     connections=[("head", "body"), ("body", "tail")]
        ... )
        >>> marker_set.validate()
        True
    """

    name: str
    markers: List[str]
    connections: List[Tuple[str, str]] = field(default_factory=list)
    description: str = ""
    reference_marker: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize reference marker if not provided."""
        if self.reference_marker is None and self.markers:
            self.reference_marker = self.markers[0]

    def validate(self) -> bool:
        """
        Validate the marker set configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.name:
            raise ValueError("MarkerSet name cannot be empty")

        if not self.markers:
            raise ValueError("MarkerSet must have at least one marker")

        # Check for duplicate markers
        if len(self.markers) != len(set(self.markers)):
            raise ValueError("MarkerSet contains duplicate markers")

        # Validate connections reference existing markers
        marker_set = set(self.markers)
        for conn in self.connections:
            if len(conn) != 2:
                raise ValueError(f"Connection must have exactly 2 markers: {conn}")
            if conn[0] not in marker_set:
                raise ValueError(f"Connection references unknown marker: {conn[0]}")
            if conn[1] not in marker_set:
                raise ValueError(f"Connection references unknown marker: {conn[1]}")

        # Validate reference marker
        if self.reference_marker and self.reference_marker not in marker_set:
            raise ValueError(f"Reference marker not in markers list: {self.reference_marker}")

        logger.debug(f"MarkerSet '{self.name}' validated successfully")
        return True

    def get_connections(self) -> List[Tuple[str, str]]:
        """
        Get the list of marker connections for skeleton visualization.

        Returns:
            List of tuples containing connected marker pairs
        """
        return self.connections.copy()

    def get_connection_indices(self) -> List[Tuple[int, int]]:
        """
        Get connections as marker index pairs.

        Returns:
            List of tuples containing connected marker indices
        """
        marker_to_idx = {marker: idx for idx, marker in enumerate(self.markers)}
        return [(marker_to_idx[c[0]], marker_to_idx[c[1]]) for c in self.connections]

    def get_marker_index(self, marker_name: str) -> int:
        """
        Get the index of a marker by name.

        Args:
            marker_name: Name of the marker

        Returns:
            Index of the marker in the markers list

        Raises:
            ValueError: If marker not found
        """
        try:
            return self.markers.index(marker_name)
        except ValueError:
            raise ValueError(f"Marker '{marker_name}' not found in marker set '{self.name}'")

    def subset(self, marker_names: List[str]) -> "MarkerSet":
        """
        Create a subset of this marker set with only specified markers.

        Args:
            marker_names: List of marker names to include

        Returns:
            New MarkerSet containing only specified markers
        """
        # Validate all markers exist
        marker_set = set(self.markers)
        for name in marker_names:
            if name not in marker_set:
                raise ValueError(f"Marker '{name}' not found in marker set")

        # Filter connections to only include those between subset markers
        subset_set = set(marker_names)
        filtered_connections = [
            conn for conn in self.connections
            if conn[0] in subset_set and conn[1] in subset_set
        ]

        # Determine new reference marker
        new_ref = self.reference_marker if self.reference_marker in subset_set else marker_names[0]

        return MarkerSet(
            name=f"{self.name}_subset",
            markers=marker_names,
            connections=filtered_connections,
            description=f"Subset of {self.name}",
            reference_marker=new_ref
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MarkerSet":
        """
        Load marker set configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            MarkerSet instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML content is invalid

        Expected YAML format:
            name: mouse_ventral
            description: Ventral view markers for mouse tracking
            markers:
              - snout
              - foreL
              - foreR
            connections:
              - [snout, foreL]
              - [snout, foreR]
            reference_marker: snout
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for loading YAML configurations. "
                            "Install with: pip install pyyaml")

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("YAML file must contain a dictionary")

        # Extract required fields
        if 'name' not in config:
            raise ValueError("YAML config must include 'name' field")
        if 'markers' not in config:
            raise ValueError("YAML config must include 'markers' field")

        # Parse connections (convert lists to tuples)
        connections = []
        if 'connections' in config:
            for conn in config['connections']:
                if isinstance(conn, (list, tuple)) and len(conn) == 2:
                    connections.append(tuple(conn))
                else:
                    raise ValueError(f"Invalid connection format: {conn}")

        marker_set = cls(
            name=config['name'],
            markers=config['markers'],
            connections=connections,
            description=config.get('description', ''),
            reference_marker=config.get('reference_marker')
        )

        marker_set.validate()
        logger.info(f"Loaded marker set '{marker_set.name}' with {len(marker_set.markers)} markers")

        return marker_set

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save marker set configuration to a YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for saving YAML configurations. "
                            "Install with: pip install pyyaml")

        config = {
            'name': self.name,
            'markers': self.markers,
            'connections': [list(conn) for conn in self.connections],
            'description': self.description,
            'reference_marker': self.reference_marker
        }

        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved marker set '{self.name}' to {yaml_path}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert marker set to dictionary representation.

        Returns:
            Dictionary containing marker set configuration
        """
        return {
            'name': self.name,
            'markers': self.markers,
            'connections': [list(conn) for conn in self.connections],
            'description': self.description,
            'reference_marker': self.reference_marker
        }

    def __len__(self) -> int:
        """Return number of markers."""
        return len(self.markers)

    def __contains__(self, marker: str) -> bool:
        """Check if marker is in the set."""
        return marker in self.markers

    def __repr__(self) -> str:
        return (f"MarkerSet(name='{self.name}', "
                f"markers={len(self.markers)}, "
                f"connections={len(self.connections)})")


# =============================================================================
# Pre-defined Marker Sets
# =============================================================================

# Ventral view markers (11 markers) - Standard for gait analysis
MOUSE_VENTRAL = MarkerSet(
    name="mouse_ventral",
    markers=[
        'snout',    # Nose tip
        'snoutL',   # Left side of snout
        'snoutR',   # Right side of snout
        'foreL',    # Left forepaw
        'foreR',    # Right forepaw
        'hindL',    # Left hindpaw
        'hindR',    # Right hindpaw
        'torso',    # Center of torso
        'torsoL',   # Left side of torso
        'torsoR',   # Right side of torso
        'tail',     # Tail base
    ],
    connections=[
        # Head connections
        ('snoutL', 'snout'),
        ('snout', 'snoutR'),
        # Body midline
        ('snout', 'torso'),
        ('torso', 'tail'),
        # Torso lateral
        ('torsoL', 'torso'),
        ('torso', 'torsoR'),
        # Forepaws to torso
        ('foreL', 'torsoL'),
        ('foreR', 'torsoR'),
        # Hindpaws to tail
        ('hindL', 'tail'),
        ('hindR', 'tail'),
        # Lateral connections
        ('snoutL', 'torsoL'),
        ('snoutR', 'torsoR'),
    ],
    description="Standard ventral (bottom) view marker set for mouse gait analysis. "
                "11 markers covering snout, paws, torso, and tail base.",
    reference_marker='torso'
)

# Lateral view markers (6 markers) - For kinematic analysis
MOUSE_LATERAL = MarkerSet(
    name="mouse_lateral",
    markers=[
        'crest',    # Iliac crest (hip bone landmark)
        'hip',      # Hip joint
        'knee',     # Knee joint
        'ankle',    # Ankle joint
        'foot',     # Metatarsal/foot
        'toe',      # Toe tip
    ],
    connections=[
        ('crest', 'hip'),
        ('hip', 'knee'),
        ('knee', 'ankle'),
        ('ankle', 'foot'),
        ('foot', 'toe'),
    ],
    description="Lateral (side) view marker set for hindlimb kinematic analysis. "
                "6 markers tracking the hindlimb from iliac crest to toe.",
    reference_marker='hip'
)


def get_preset_marker_set(name: str) -> MarkerSet:
    """
    Get a pre-defined marker set by name.

    Args:
        name: Name of the preset ('ventral', 'lateral', 'mouse_ventral', 'mouse_lateral')

    Returns:
        MarkerSet instance

    Raises:
        ValueError: If preset name not recognized
    """
    presets = {
        'ventral': MOUSE_VENTRAL,
        'mouse_ventral': MOUSE_VENTRAL,
        'lateral': MOUSE_LATERAL,
        'mouse_lateral': MOUSE_LATERAL,
    }

    name_lower = name.lower()
    if name_lower not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return presets[name_lower]
