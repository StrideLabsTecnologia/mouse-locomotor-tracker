"""
Scientific Data Export Module
=============================

Export tracking data to scientific formats:
- HDF5: Efficient binary format for large datasets
- NWB: Neurodata Without Borders (neuroscience standard)

Author: Stride Labs
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.behavior import (
        BehavioralTimeSeries,
        Position,
        SpatialSeries,
    )
    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False


@dataclass
class ExportMetadata:
    """Metadata for scientific export."""
    experiment_id: str
    subject_id: str
    session_date: datetime
    experimenter: str = "Unknown"
    lab: str = "Stride Labs"
    institution: str = ""
    description: str = "Mouse locomotion tracking data"
    keywords: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = ["locomotion", "mouse", "tracking", "biomechanics"]


class HDF5Exporter:
    """
    Export tracking data to HDF5 format.

    HDF5 is an efficient binary format ideal for large datasets.
    Supports compression, chunking, and hierarchical organization.

    Example:
        >>> exporter = HDF5Exporter()
        >>> exporter.export(
        ...     output_path="experiment.h5",
        ...     positions=positions,
        ...     velocities=velocities,
        ...     metadata=metadata
        ... )
    """

    def __init__(self, compression: str = "gzip", compression_opts: int = 4):
        """
        Initialize HDF5 exporter.

        Args:
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py required. Install with: pip install h5py")

        self.compression = compression
        self.compression_opts = compression_opts

    def export(
        self,
        output_path: Union[str, Path],
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        frame_rate: float = 30.0,
        metadata: Optional[ExportMetadata] = None,
        additional_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Path:
        """
        Export tracking data to HDF5.

        Args:
            output_path: Output file path
            positions: Position data (N, 2) array of (x, y) coordinates
            velocities: Velocity data (N,) array
            timestamps: Timestamp array (N,) in seconds
            frame_rate: Video frame rate
            metadata: Export metadata
            additional_data: Additional arrays to include

        Returns:
            Path to created file
        """
        output_path = Path(output_path)

        if timestamps is None:
            timestamps = np.arange(len(positions)) / frame_rate

        with h5py.File(output_path, 'w') as f:
            # Root attributes
            f.attrs['format'] = 'MLT-HDF5'
            f.attrs['version'] = '1.0'
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['frame_rate'] = frame_rate

            # Metadata group
            if metadata:
                meta = f.create_group('metadata')
                meta.attrs['experiment_id'] = metadata.experiment_id
                meta.attrs['subject_id'] = metadata.subject_id
                meta.attrs['session_date'] = metadata.session_date.isoformat()
                meta.attrs['experimenter'] = metadata.experimenter
                meta.attrs['lab'] = metadata.lab
                meta.attrs['institution'] = metadata.institution
                meta.attrs['description'] = metadata.description
                meta.attrs['keywords'] = ','.join(metadata.keywords)

            # Tracking data group
            tracking = f.create_group('tracking')

            # Timestamps
            tracking.create_dataset(
                'timestamps',
                data=timestamps,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            tracking['timestamps'].attrs['unit'] = 'seconds'

            # Positions
            tracking.create_dataset(
                'positions',
                data=positions,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            tracking['positions'].attrs['unit'] = 'pixels'
            tracking['positions'].attrs['columns'] = 'x,y'

            # Velocities
            if velocities is not None:
                tracking.create_dataset(
                    'velocities',
                    data=velocities,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                tracking['velocities'].attrs['unit'] = 'cm/s'

            # Additional data
            if additional_data:
                extra = f.create_group('additional')
                for name, data in additional_data.items():
                    extra.create_dataset(
                        name,
                        data=data,
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )

        return output_path

    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            Dictionary with loaded data
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py required. Install with: pip install h5py")

        data = {}
        with h5py.File(file_path, 'r') as f:
            data['format'] = f.attrs.get('format', 'unknown')
            data['version'] = f.attrs.get('version', 'unknown')
            data['frame_rate'] = f.attrs.get('frame_rate', 30.0)

            if 'tracking' in f:
                tracking = f['tracking']
                data['timestamps'] = tracking['timestamps'][:]
                data['positions'] = tracking['positions'][:]
                if 'velocities' in tracking:
                    data['velocities'] = tracking['velocities'][:]

            if 'metadata' in f:
                meta = f['metadata']
                data['metadata'] = {k: meta.attrs[k] for k in meta.attrs}

        return data


class NWBExporter:
    """
    Export tracking data to Neurodata Without Borders (NWB) format.

    NWB is the standard format for neuroscience data sharing.
    Compatible with DANDI archive and major analysis tools.

    Example:
        >>> exporter = NWBExporter()
        >>> exporter.export(
        ...     output_path="experiment.nwb",
        ...     positions=positions,
        ...     velocities=velocities,
        ...     metadata=metadata
        ... )
    """

    def __init__(self):
        """Initialize NWB exporter."""
        if not NWB_AVAILABLE:
            raise ImportError("pynwb required. Install with: pip install pynwb")

    def export(
        self,
        output_path: Union[str, Path],
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        frame_rate: float = 30.0,
        metadata: Optional[ExportMetadata] = None,
    ) -> Path:
        """
        Export tracking data to NWB format.

        Args:
            output_path: Output file path
            positions: Position data (N, 2) array
            velocities: Velocity data (N,) array
            timestamps: Timestamp array (N,)
            frame_rate: Video frame rate
            metadata: Export metadata

        Returns:
            Path to created file
        """
        output_path = Path(output_path)

        if timestamps is None:
            timestamps = np.arange(len(positions)) / frame_rate

        if metadata is None:
            metadata = ExportMetadata(
                experiment_id="unknown",
                subject_id="unknown",
                session_date=datetime.now()
            )

        # Create NWB file
        nwbfile = NWBFile(
            session_description=metadata.description,
            identifier=f"{metadata.experiment_id}_{metadata.session_date.strftime('%Y%m%d')}",
            session_start_time=metadata.session_date,
            experimenter=metadata.experimenter,
            lab=metadata.lab,
            institution=metadata.institution,
            keywords=metadata.keywords,
        )

        # Add behavior module
        behavior_module = nwbfile.create_processing_module(
            name="behavior",
            description="Mouse locomotion tracking data"
        )

        # Add position data
        position = Position(name="position")

        spatial_series = SpatialSeries(
            name="mouse_position",
            description="XY position of mouse center of mass",
            data=positions,
            timestamps=timestamps,
            reference_frame="video frame, origin top-left",
            unit="pixels",
        )
        position.add_spatial_series(spatial_series)
        behavior_module.add(position)

        # Add velocity data
        if velocities is not None:
            velocity_ts = BehavioralTimeSeries(name="velocity")
            from pynwb.base import TimeSeries
            vel_series = TimeSeries(
                name="instantaneous_velocity",
                description="Instantaneous velocity of mouse",
                data=velocities,
                timestamps=timestamps,
                unit="cm/s",
            )
            velocity_ts.add_timeseries(vel_series)
            behavior_module.add(velocity_ts)

        # Write to file
        with NWBHDF5IO(str(output_path), 'w') as io:
            io.write(nwbfile)

        return output_path

    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from NWB file.

        Args:
            file_path: Path to NWB file

        Returns:
            Dictionary with loaded data
        """
        if not NWB_AVAILABLE:
            raise ImportError("pynwb required. Install with: pip install pynwb")

        data = {}
        with NWBHDF5IO(str(file_path), 'r') as io:
            nwbfile = io.read()

            data['session_description'] = nwbfile.session_description
            data['identifier'] = nwbfile.identifier
            data['session_start_time'] = nwbfile.session_start_time

            if 'behavior' in nwbfile.processing:
                behavior = nwbfile.processing['behavior']

                if 'position' in behavior.data_interfaces:
                    position = behavior.data_interfaces['position']
                    for name, series in position.spatial_series.items():
                        data['positions'] = series.data[:]
                        data['timestamps'] = series.timestamps[:]

                if 'velocity' in behavior.data_interfaces:
                    velocity = behavior.data_interfaces['velocity']
                    for name, series in velocity.time_series.items():
                        data['velocities'] = series.data[:]

        return data


def export_to_scientific_format(
    output_path: Union[str, Path],
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    frame_rate: float = 30.0,
    metadata: Optional[ExportMetadata] = None,
    format: str = "auto",
) -> Path:
    """
    Export tracking data to scientific format.

    Automatically selects format based on file extension or format parameter.

    Args:
        output_path: Output file path
        positions: Position data
        velocities: Velocity data
        timestamps: Timestamps
        frame_rate: Frame rate
        metadata: Export metadata
        format: Format ('hdf5', 'nwb', or 'auto')

    Returns:
        Path to created file

    Example:
        >>> export_to_scientific_format(
        ...     "experiment.h5",
        ...     positions,
        ...     velocities,
        ...     format="hdf5"
        ... )
    """
    output_path = Path(output_path)

    if format == "auto":
        suffix = output_path.suffix.lower()
        if suffix in ['.h5', '.hdf5']:
            format = "hdf5"
        elif suffix == '.nwb':
            format = "nwb"
        else:
            format = "hdf5"  # Default to HDF5

    if format == "hdf5":
        exporter = HDF5Exporter()
    elif format == "nwb":
        exporter = NWBExporter()
    else:
        raise ValueError(f"Unknown format: {format}")

    return exporter.export(
        output_path=output_path,
        positions=positions,
        velocities=velocities,
        timestamps=timestamps,
        frame_rate=frame_rate,
        metadata=metadata,
    )
