"""
Export Module - Mouse Locomotor Tracker
========================================

Data export functionality for analysis results.

Supported formats:
- CSV: Simple tabular data
- JSON: Structured data with metadata
- HDF5: Efficient binary format for large datasets
- NWB: Neurodata Without Borders (neuroscience standard)
"""

from .csv_exporter import CSVExporter
from .json_exporter import JSONExporter
from .report_generator import ReportGenerator

# Scientific formats (optional dependencies)
try:
    from .scientific_exporter import (
        HDF5Exporter,
        NWBExporter,
        ExportMetadata,
        export_to_scientific_format,
    )
    SCIENTIFIC_EXPORT_AVAILABLE = True
except ImportError:
    SCIENTIFIC_EXPORT_AVAILABLE = False
    HDF5Exporter = None
    NWBExporter = None
    ExportMetadata = None
    export_to_scientific_format = None

__all__ = [
    'CSVExporter',
    'JSONExporter',
    'ReportGenerator',
    'HDF5Exporter',
    'NWBExporter',
    'ExportMetadata',
    'export_to_scientific_format',
    'SCIENTIFIC_EXPORT_AVAILABLE',
]
