"""
CSV Exporter - Mouse Locomotor Tracker
======================================

Export analysis results to CSV format.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict, is_dataclass
import logging

logger = logging.getLogger(__name__)


class CSVExporter:
    """
    Export locomotor analysis results to CSV format.

    Creates a comprehensive statistics CSV with all metrics
    similar to Locomotor-Allodi2021 output format.
    """

    # Column definitions matching Locomotor-Allodi2021 format
    COLUMNS = [
        'name', 'bodyLen', 'duration', 'belt_speed', 'avg_speed',
        'loc_front', 'loc_rear', 'peak_acc', 'num_drag',
        'num_rec', 'count_ratio', 'dur_drag', 'dur_rec', 'mov_dur', 'num_steps',
        'LH_st_len', 'LF_st_len', 'RH_st_len', 'RF_st_len',
        'LH_st_frq', 'LF_st_frq', 'RH_st_frq', 'RF_st_frq',
        'LHRH_ang', 'LHLF_ang', 'RHRF_ang', 'LFRH_ang', 'RFLH_ang', 'LFRF_ang',
        'LHRH_rad', 'LHLF_rad', 'RHRF_rad', 'LFRH_rad', 'RFLH_rad', 'LFRF_rad',
        'LHRH_width', 'hip_ang', 'knee_ang', 'ankle_ang', 'foot_ang'
    ]

    def __init__(self, float_format: str = '%.4f'):
        """
        Initialize CSV exporter.

        Args:
            float_format: Format string for floating point numbers
        """
        self.float_format = float_format

    def export(
        self,
        results: Dict[str, Any],
        output_path: Path,
        video_name: Optional[str] = None
    ) -> Path:
        """
        Export analysis results to CSV.

        Args:
            results: Dictionary containing analysis results
            output_path: Path for output CSV file
            video_name: Optional name for the video

        Returns:
            Path to the created CSV file
        """
        output_path = Path(output_path)

        # Flatten results to single row
        row = self._flatten_results(results, video_name)

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerow(row)

        logger.info(f"Exported statistics to: {output_path}")
        return output_path

    def export_batch(
        self,
        results_list: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Export multiple analysis results to single CSV.

        Args:
            results_list: List of result dictionaries
            output_path: Path for output CSV file

        Returns:
            Path to the created CSV file
        """
        output_path = Path(output_path)

        rows = [self._flatten_results(r) for r in results_list]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported batch statistics to: {output_path}")
        return output_path

    def _flatten_results(
        self,
        results: Dict[str, Any],
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flatten nested results dictionary to single-level dict.

        Args:
            results: Nested results dictionary
            video_name: Optional video name

        Returns:
            Flattened dictionary matching CSV columns
        """
        row = {col: 0 for col in self.COLUMNS}  # Initialize with zeros

        row['name'] = video_name or results.get('name', 'unknown')

        # Velocity metrics
        if 'velocity' in results:
            v = results['velocity']
            if is_dataclass(v):
                v = asdict(v)
            row['avg_speed'] = self._format_float(v.get('avg_speed', 0))
            row['peak_acc'] = self._format_float(v.get('peak_acceleration', 0))
            row['duration'] = self._format_float(v.get('duration', 0))

        # Gait metrics
        if 'gait' in results:
            g = results['gait']
            if is_dataclass(g):
                g = asdict(g)

            row['mov_dur'] = self._format_float(g.get('movement_duration', 0))
            row['num_steps'] = g.get('num_steps', 0)

            # Stride lengths
            stride_len = g.get('stride_length', {})
            row['LH_st_len'] = self._format_float(stride_len.get('LH', 0))
            row['RH_st_len'] = self._format_float(stride_len.get('RH', 0))
            row['LF_st_len'] = self._format_float(stride_len.get('LF', 0))
            row['RF_st_len'] = self._format_float(stride_len.get('RF', 0))

            # Cadence
            cadence = g.get('cadence', {})
            row['LH_st_frq'] = self._format_float(cadence.get('LH', 0))
            row['RH_st_frq'] = self._format_float(cadence.get('RH', 0))
            row['LF_st_frq'] = self._format_float(cadence.get('LF', 0))
            row['RF_st_frq'] = self._format_float(cadence.get('RF', 0))

        # Coordination metrics
        if 'coordination' in results:
            coord = results['coordination']

            for pair in ['LHRH', 'LHLF', 'RHRF', 'LFRH', 'RFLH', 'LFRF']:
                if pair in coord:
                    c = coord[pair]
                    if is_dataclass(c):
                        c = asdict(c)
                    row[f'{pair}_ang'] = self._format_float(c.get('mean_phase', 0))
                    row[f'{pair}_rad'] = self._format_float(c.get('r_value', 0))

        # Kinematics metrics
        if 'kinematics' in results:
            k = results['kinematics']
            if is_dataclass(k):
                k = asdict(k)

            for joint in ['hip', 'knee', 'ankle', 'foot']:
                row[f'{joint}_ang'] = self._format_float(k.get(f'{joint}_range', 0))

        return row

    def _format_float(self, value: float) -> float:
        """Format float value."""
        try:
            return float(self.float_format % value)
        except (TypeError, ValueError):
            return 0.0
