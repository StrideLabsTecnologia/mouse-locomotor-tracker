"""
JSON Exporter - Mouse Locomotor Tracker
=======================================

Export analysis results to JSON format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict, is_dataclass
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


class JSONExporter:
    """
    Export locomotor analysis results to JSON format.

    Creates a comprehensive JSON file with all metrics,
    metadata, and raw data arrays.
    """

    def __init__(self, indent: int = 2, include_arrays: bool = True):
        """
        Initialize JSON exporter.

        Args:
            indent: JSON indentation level
            include_arrays: Whether to include raw data arrays
        """
        self.indent = indent
        self.include_arrays = include_arrays

    def export(
        self,
        results: Dict[str, Any],
        output_path: Path,
        video_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Export analysis results to JSON.

        Args:
            results: Dictionary containing analysis results
            output_path: Path for output JSON file
            video_name: Optional name for the video
            metadata: Optional metadata dictionary

        Returns:
            Path to the created JSON file
        """
        output_path = Path(output_path)

        # Build export structure
        export_data = {
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'video_name': video_name,
            'metadata': metadata or {},
            'summary': self._create_summary(results),
            'metrics': self._process_results(results)
        }

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, cls=NumpyEncoder, indent=self.indent)

        logger.info(f"Exported JSON to: {output_path}")
        return output_path

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary section with key metrics.

        Args:
            results: Analysis results

        Returns:
            Summary dictionary
        """
        summary = {}

        # Velocity summary
        if 'velocity' in results:
            v = results['velocity']
            if is_dataclass(v):
                v = asdict(v)
            summary['velocity'] = {
                'avg_speed_cm_s': v.get('avg_speed', 0),
                'peak_speed_cm_s': v.get('peak_speed', 0),
                'peak_acceleration_cm_s2': v.get('peak_acceleration', 0)
            }

        # Gait summary
        if 'gait' in results:
            g = results['gait']
            if is_dataclass(g):
                g = asdict(g)
            summary['gait'] = {
                'num_steps': g.get('num_steps', 0),
                'movement_duration_s': g.get('movement_duration', 0),
                'avg_cadence_hz': np.mean(list(g.get('cadence', {}).values())) if g.get('cadence') else 0,
                'avg_stride_length_cm': np.mean(list(g.get('stride_length', {}).values())) if g.get('stride_length') else 0
            }

        # Coordination summary
        if 'coordination' in results:
            coord = results['coordination']
            r_values = []
            for pair, metrics in coord.items():
                if is_dataclass(metrics):
                    metrics = asdict(metrics)
                r_values.append(metrics.get('r_value', 0))

            summary['coordination'] = {
                'mean_r_value': np.mean(r_values) if r_values else 0,
                'coordination_quality': 'good' if np.mean(r_values) > 0.7 else 'moderate' if np.mean(r_values) > 0.5 else 'poor'
            }

        return summary

    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process results for JSON export.

        Args:
            results: Raw results dictionary

        Returns:
            Processed results
        """
        processed = {}

        for key, value in results.items():
            if key == 'report':
                continue  # Skip report object

            if is_dataclass(value):
                value = asdict(value)

            if isinstance(value, dict):
                processed[key] = self._process_dict(value)
            elif isinstance(value, (list, np.ndarray)):
                if self.include_arrays:
                    processed[key] = value
            else:
                processed[key] = value

        return processed

    def _process_dict(self, d: Dict) -> Dict:
        """
        Recursively process dictionary.

        Args:
            d: Input dictionary

        Returns:
            Processed dictionary
        """
        result = {}

        for key, value in d.items():
            if is_dataclass(value):
                value = asdict(value)

            if isinstance(value, dict):
                result[key] = self._process_dict(value)
            elif isinstance(value, np.ndarray):
                if self.include_arrays:
                    result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
            else:
                result[key] = value

        return result


class JSONLExporter:
    """
    Export results to JSON Lines format (one JSON object per line).
    Useful for streaming and large datasets.
    """

    def export_batch(
        self,
        results_list: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Export multiple results to JSONL file.

        Args:
            results_list: List of result dictionaries
            output_path: Output file path

        Returns:
            Path to created file
        """
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            for results in results_list:
                if is_dataclass(results):
                    results = asdict(results)
                line = json.dumps(results, cls=NumpyEncoder)
                f.write(line + '\n')

        logger.info(f"Exported JSONL to: {output_path}")
        return output_path
