"""
Interlimb Coordination Analysis Module
======================================

Implements circular statistics (Rayleigh test) for analyzing
interlimb coordination patterns in mouse locomotion.

Based on Locomotor-Allodi2021 methodology with enhancements.

Author: Stride Labs - Mouse Locomotor Tracker
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import warnings

from .metrics import (
    LimbPair,
    CircularStatistics,
    CoordinationMetrics,
)


# Standard phase relationships for normal quadruped gait patterns
NORMAL_PHASE_RELATIONSHIPS = {
    LimbPair.LHRH: 180.0,    # Left-right hindlimbs: alternating (180 deg)
    LimbPair.LFRF: 180.0,    # Left-right forelimbs: alternating (180 deg)
    LimbPair.LHLF: 90.0,     # Ipsilateral hind-fore: ~90 deg lag
    LimbPair.RHRF: 90.0,     # Ipsilateral hind-fore: ~90 deg lag
    LimbPair.LFRH: 90.0,     # Diagonal: variable
    LimbPair.RFLH: 90.0,     # Diagonal: variable
}


class CircularCoordinationAnalyzer:
    """
    Analyzes interlimb coordination using circular statistics.

    This class computes phase relationships between limb pairs and
    uses the Rayleigh test to assess the consistency of coordination.
    A high mean vector length (R value close to 1) indicates strong
    coordination, while R close to 0 indicates no consistent pattern.

    Circular Statistics Background:
    - Phase values wrap around at 360 degrees (or 2*pi radians)
    - Regular mean/std don't work for circular data
    - We use vector representation: convert angles to unit vectors,
      average the vectors, and get the mean angle from the result
    - R = |mean vector| measures concentration (0=uniform, 1=perfect alignment)
    - Rayleigh test: tests if distribution is significantly non-uniform

    Attributes:
        significance_level: P-value threshold for Rayleigh test (default: 0.05)
        min_samples: Minimum samples required for analysis (default: 10)
        frame_rate: Video frame rate for temporal calculations

    Example:
        >>> analyzer = CircularCoordinationAnalyzer()
        >>> stats = analyzer.analyze_pair(lh_phases, rh_phases)
        >>> print(f"R value: {stats.mean_vector_length:.3f}")
        >>> print(f"Significant: {stats.is_significant}")
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 10,
        frame_rate: float = 30.0,
    ):
        """
        Initialize the CircularCoordinationAnalyzer.

        Args:
            significance_level: P-value threshold for significance
            min_samples: Minimum number of samples for valid analysis
            frame_rate: Video frame rate in Hz
        """
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        if min_samples < 3:
            raise ValueError("min_samples must be at least 3")

        self.significance_level = significance_level
        self.min_samples = min_samples
        self.frame_rate = frame_rate

    @staticmethod
    def _normalize_phase(phase: np.ndarray) -> np.ndarray:
        """
        Normalize phase values to [0, 2*pi) range.

        Args:
            phase: Phase values in radians

        Returns:
            Normalized phase values
        """
        return np.mod(phase, 2 * np.pi)

    @staticmethod
    def _degrees_to_radians(degrees: np.ndarray) -> np.ndarray:
        """Convert degrees to radians."""
        return np.deg2rad(degrees)

    @staticmethod
    def _radians_to_degrees(radians: np.ndarray) -> np.ndarray:
        """Convert radians to degrees."""
        return np.rad2deg(radians)

    def circular_mean(
        self, phases: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute circular mean and mean vector length.

        The circular mean is computed by:
        1. Convert phases to unit vectors on the unit circle
        2. Average the X (cos) and Y (sin) components
        3. The mean angle is atan2(mean_Y, mean_X)
        4. R = sqrt(mean_X^2 + mean_Y^2) is the mean vector length

        Args:
            phases: Phase values in radians
            weights: Optional weights for weighted mean

        Returns:
            Tuple of (mean_angle, R, X_component, Y_component)
        """
        phases = np.asarray(phases, dtype=np.float64)
        phases = self._normalize_phase(phases)

        # Remove NaN values
        valid_mask = ~np.isnan(phases)
        phases = phases[valid_mask]

        if len(phases) == 0:
            return 0.0, 0.0, 0.0, 0.0

        if weights is not None:
            weights = np.asarray(weights)[valid_mask]
            weights = weights / np.sum(weights)  # Normalize
        else:
            weights = np.ones(len(phases)) / len(phases)

        # Convert to Cartesian coordinates on unit circle
        X = np.sum(weights * np.cos(phases))
        Y = np.sum(weights * np.sin(phases))

        # Mean vector length (concentration parameter)
        R = np.sqrt(X**2 + Y**2)

        # Mean angle
        mean_angle = np.arctan2(Y, X)
        if mean_angle < 0:
            mean_angle += 2 * np.pi

        return float(mean_angle), float(R), float(X), float(Y)

    def circular_variance(self, phases: np.ndarray) -> float:
        """
        Compute circular variance.

        Circular variance = 1 - R, where R is mean vector length.
        Range: [0, 1], where 0 = no variance, 1 = maximum variance.

        Args:
            phases: Phase values in radians

        Returns:
            Circular variance
        """
        _, R, _, _ = self.circular_mean(phases)
        return 1.0 - R

    def circular_std(self, phases: np.ndarray) -> float:
        """
        Compute circular standard deviation.

        Using the formula: std = sqrt(-2 * ln(R))

        Args:
            phases: Phase values in radians

        Returns:
            Circular standard deviation in radians
        """
        _, R, _, _ = self.circular_mean(phases)

        if R == 0:
            return np.inf

        # Avoid numerical issues
        R = min(R, 0.9999999)

        return np.sqrt(-2 * np.log(R))

    def rayleigh_test(
        self, phases: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Perform Rayleigh test for circular uniformity.

        Tests the null hypothesis that the phases are uniformly
        distributed around the circle. A significant result indicates
        a preferred direction (non-uniform distribution).

        The Rayleigh Z statistic is: Z = n * R^2
        where n is sample size and R is mean vector length.

        Args:
            phases: Phase values in radians

        Returns:
            Tuple of (Z_statistic, p_value, is_significant)
        """
        phases = np.asarray(phases, dtype=np.float64)
        phases = phases[~np.isnan(phases)]

        n = len(phases)
        if n < self.min_samples:
            warnings.warn(
                f"Sample size ({n}) below minimum ({self.min_samples}). "
                "Results may be unreliable."
            )

        _, R, _, _ = self.circular_mean(phases)

        # Rayleigh Z statistic
        Z = n * R**2

        # P-value approximation (Mardia & Jupp, 2000)
        # For large n, 2*Z approximately follows chi-squared with 2 df
        p_value = np.exp(-Z) * (
            1 + (2 * Z - Z**2) / (4 * n) -
            (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)
        )

        is_significant = p_value < self.significance_level

        return float(Z), float(p_value), is_significant

    def compute_phase_difference(
        self,
        phases_a: np.ndarray,
        phases_b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute phase difference between two limbs.

        The phase difference is computed as: diff = phases_b - phases_a
        and normalized to [-pi, pi].

        Args:
            phases_a: Phase values for limb A (radians)
            phases_b: Phase values for limb B (radians)

        Returns:
            Array of phase differences in radians
        """
        phases_a = np.asarray(phases_a, dtype=np.float64)
        phases_b = np.asarray(phases_b, dtype=np.float64)

        if len(phases_a) != len(phases_b):
            raise ValueError(
                f"Phase arrays must have same length: "
                f"{len(phases_a)} vs {len(phases_b)}"
            )

        diff = phases_b - phases_a

        # Normalize to [-pi, pi]
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi

        return diff

    def phase_from_cycle_fraction(
        self,
        cycle_times: np.ndarray,
        event_times: np.ndarray,
    ) -> np.ndarray:
        """
        Convert event times to phase values within gait cycles.

        Phase is computed as: phase = 2*pi * (event_time - cycle_start) / cycle_duration

        Args:
            cycle_times: Start times of each gait cycle
            event_times: Times of events (e.g., foot contacts)

        Returns:
            Phase values in radians [0, 2*pi)
        """
        cycle_times = np.asarray(cycle_times, dtype=np.float64)
        event_times = np.asarray(event_times, dtype=np.float64)

        phases = []

        for event_time in event_times:
            # Find which cycle this event belongs to
            cycle_idx = np.searchsorted(cycle_times, event_time) - 1

            if cycle_idx < 0 or cycle_idx >= len(cycle_times) - 1:
                phases.append(np.nan)
                continue

            cycle_start = cycle_times[cycle_idx]
            cycle_end = cycle_times[cycle_idx + 1]
            cycle_duration = cycle_end - cycle_start

            if cycle_duration <= 0:
                phases.append(np.nan)
                continue

            # Compute phase as fraction of cycle
            phase = 2 * np.pi * (event_time - cycle_start) / cycle_duration
            phases.append(phase)

        return np.array(phases)

    def analyze_pair(
        self,
        phases_a: np.ndarray,
        phases_b: np.ndarray,
        pair_type: Optional[LimbPair] = None,
    ) -> CircularStatistics:
        """
        Analyze coordination between a pair of limbs.

        Args:
            phases_a: Phase values for limb A (radians)
            phases_b: Phase values for limb B (radians)
            pair_type: Optional LimbPair identifier

        Returns:
            CircularStatistics for the limb pair
        """
        # Compute phase differences
        phase_diff = self.compute_phase_difference(phases_a, phases_b)

        # Remove NaN values
        valid_diff = phase_diff[~np.isnan(phase_diff)]
        n_samples = len(valid_diff)

        if n_samples < self.min_samples:
            # Return empty statistics for insufficient data
            return CircularStatistics(
                mean_angle=0.0,
                mean_angle_degrees=0.0,
                mean_vector_length=0.0,
                x_component=0.0,
                y_component=0.0,
                rayleigh_z=0.0,
                rayleigh_p=1.0,
                is_significant=False,
                sample_size=n_samples,
            )

        # Normalize to [0, 2*pi)
        valid_diff = self._normalize_phase(valid_diff)

        # Compute circular statistics
        mean_angle, R, X, Y = self.circular_mean(valid_diff)

        # Perform Rayleigh test
        Z, p_value, is_significant = self.rayleigh_test(valid_diff)

        return CircularStatistics(
            mean_angle=mean_angle,
            mean_angle_degrees=float(self._radians_to_degrees(mean_angle)),
            mean_vector_length=R,
            x_component=X,
            y_component=Y,
            rayleigh_z=Z,
            rayleigh_p=p_value,
            is_significant=is_significant,
            sample_size=n_samples,
        )

    def analyze_all_pairs(
        self,
        limb_phases: Dict[str, np.ndarray],
    ) -> CoordinationMetrics:
        """
        Analyze coordination for all 6 standard limb pairs.

        Expected keys in limb_phases:
        - 'LH': Left hindlimb
        - 'RH': Right hindlimb
        - 'LF': Left forelimb
        - 'RF': Right forelimb

        Args:
            limb_phases: Dictionary mapping limb names to phase arrays

        Returns:
            CoordinationMetrics containing all pair analyses
        """
        # Validate input
        required_limbs = {'LH', 'RH', 'LF', 'RF'}
        available_limbs = set(limb_phases.keys())

        if not required_limbs.issubset(available_limbs):
            missing = required_limbs - available_limbs
            raise ValueError(f"Missing limb phases: {missing}")

        # Define pair mappings
        pair_definitions = {
            LimbPair.LHRH: ('LH', 'RH'),   # Left Hind - Right Hind
            LimbPair.LHLF: ('LH', 'LF'),   # Left Hind - Left Fore
            LimbPair.RHRF: ('RH', 'RF'),   # Right Hind - Right Fore
            LimbPair.LFRH: ('LF', 'RH'),   # Left Fore - Right Hind
            LimbPair.RFLH: ('RF', 'LH'),   # Right Fore - Left Hind
            LimbPair.LFRF: ('LF', 'RF'),   # Left Fore - Right Fore
        }

        # Analyze each pair
        pair_statistics: Dict[LimbPair, CircularStatistics] = {}

        for pair, (limb_a, limb_b) in pair_definitions.items():
            stats = self.analyze_pair(
                limb_phases[limb_a],
                limb_phases[limb_b],
                pair_type=pair,
            )
            pair_statistics[pair] = stats

        # Compute aggregate metrics
        all_R = [s.mean_vector_length for s in pair_statistics.values()]
        overall_score = float(np.mean(all_R))

        # Ipsilateral (same side): LHLF, RHRF
        ipsilateral_R = [
            pair_statistics[LimbPair.LHLF].mean_vector_length,
            pair_statistics[LimbPair.RHRF].mean_vector_length,
        ]
        ipsilateral_coupling = float(np.mean(ipsilateral_R))

        # Contralateral (opposite side, same level): LHRH, LFRF
        contralateral_R = [
            pair_statistics[LimbPair.LHRH].mean_vector_length,
            pair_statistics[LimbPair.LFRF].mean_vector_length,
        ]
        contralateral_coupling = float(np.mean(contralateral_R))

        # Diagonal: LFRH, RFLH
        diagonal_R = [
            pair_statistics[LimbPair.LFRH].mean_vector_length,
            pair_statistics[LimbPair.RFLH].mean_vector_length,
        ]
        fore_hind_coupling = float(np.mean(diagonal_R))

        # Left-right: LHRH, LFRF
        left_right_coupling = contralateral_coupling  # Same as contralateral

        # Detect gait pattern
        pattern = self._detect_gait_pattern(pair_statistics)

        return CoordinationMetrics(
            pair_statistics=pair_statistics,
            overall_coordination_score=overall_score,
            ipsilateral_coupling=ipsilateral_coupling,
            contralateral_coupling=contralateral_coupling,
            fore_hind_coupling=fore_hind_coupling,
            left_right_coupling=left_right_coupling,
            coordination_pattern=pattern,
        )

    def _detect_gait_pattern(
        self, pair_stats: Dict[LimbPair, CircularStatistics]
    ) -> str:
        """
        Detect the gait pattern based on phase relationships.

        Gait patterns in quadrupeds:
        - Walk: Sequential limb movement, ~25% duty factor
        - Trot: Diagonal limbs move together (LHRH~180, LFRF~180)
        - Pace: Ipsilateral limbs move together
        - Gallop/Bound: Various asymmetric patterns

        Args:
            pair_stats: Statistics for each limb pair

        Returns:
            Name of detected gait pattern
        """
        # Get phase angles in degrees
        lhrh_angle = pair_stats[LimbPair.LHRH].mean_angle_degrees
        lfrf_angle = pair_stats[LimbPair.LFRF].mean_angle_degrees
        lhlf_angle = pair_stats[LimbPair.LHLF].mean_angle_degrees
        rhrf_angle = pair_stats[LimbPair.RHRF].mean_angle_degrees

        # Get R values for significance
        lhrh_R = pair_stats[LimbPair.LHRH].mean_vector_length
        lfrf_R = pair_stats[LimbPair.LFRF].mean_vector_length

        # Check for trot pattern (diagonal sync, 180 deg phase)
        if (lhrh_R > 0.7 and lfrf_R > 0.7 and
            self._angle_near(lhrh_angle, 180, tolerance=30) and
            self._angle_near(lfrf_angle, 180, tolerance=30)):
            return "trot"

        # Check for pace pattern (ipsilateral sync)
        if (lhrh_R > 0.7 and lfrf_R > 0.7 and
            self._angle_near(lhlf_angle, 0, tolerance=30) and
            self._angle_near(rhrf_angle, 0, tolerance=30)):
            return "pace"

        # Check for walk pattern (sequential, lower coordination)
        if lhrh_R > 0.5 and lfrf_R > 0.5:
            return "walk"

        # Check for bound (hindlimbs together, forelimbs together)
        if (self._angle_near(lhrh_angle, 0, tolerance=30) and
            self._angle_near(lfrf_angle, 0, tolerance=30)):
            return "bound"

        # Low coordination - possibly injured or irregular
        if lhrh_R < 0.3 or lfrf_R < 0.3:
            return "irregular"

        return "unknown"

    @staticmethod
    def _angle_near(
        angle: float, target: float, tolerance: float = 30
    ) -> bool:
        """
        Check if an angle is near a target value (circular comparison).

        Args:
            angle: Angle to check (degrees)
            target: Target angle (degrees)
            tolerance: Tolerance in degrees

        Returns:
            True if angle is within tolerance of target
        """
        diff = abs(angle - target)
        diff = min(diff, 360 - diff)  # Handle wrap-around
        return diff <= tolerance

    def compute_coupling_matrix(
        self, limb_phases: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute a coupling matrix showing R values between all limb pairs.

        Args:
            limb_phases: Dictionary mapping limb names to phase arrays

        Returns:
            4x4 matrix of R values (LH, RH, LF, RF order)
        """
        limbs = ['LH', 'RH', 'LF', 'RF']
        n_limbs = len(limbs)
        matrix = np.zeros((n_limbs, n_limbs))

        for i, limb_a in enumerate(limbs):
            for j, limb_b in enumerate(limbs):
                if i == j:
                    matrix[i, j] = 1.0  # Self-coupling is perfect
                elif j > i:
                    # Compute R value
                    if limb_a in limb_phases and limb_b in limb_phases:
                        stats = self.analyze_pair(
                            limb_phases[limb_a],
                            limb_phases[limb_b]
                        )
                        matrix[i, j] = stats.mean_vector_length
                        matrix[j, i] = stats.mean_vector_length  # Symmetric

        return matrix


def create_coordination_analyzer(
    config: Optional[Dict[str, Any]] = None
) -> CircularCoordinationAnalyzer:
    """
    Factory function to create a CircularCoordinationAnalyzer.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured CircularCoordinationAnalyzer instance
    """
    if config is None:
        config = {}

    return CircularCoordinationAnalyzer(
        significance_level=config.get('significance_level', 0.05),
        min_samples=config.get('min_samples', 10),
        frame_rate=config.get('frame_rate', 30.0),
    )


# Utility functions for phase extraction

def extract_phase_from_position(
    position: np.ndarray,
    method: str = 'hilbert',
) -> np.ndarray:
    """
    Extract instantaneous phase from a position signal.

    Args:
        position: 1D position signal
        method: Extraction method ('hilbert' or 'peaks')

    Returns:
        Phase values in radians
    """
    from scipy.signal import hilbert

    position = np.asarray(position, dtype=np.float64)

    # Remove mean
    position_centered = position - np.nanmean(position)

    if method == 'hilbert':
        # Hilbert transform for instantaneous phase
        analytic = hilbert(position_centered)
        phase = np.angle(analytic)
        # Convert to [0, 2*pi)
        phase = np.mod(phase, 2 * np.pi)
    else:
        raise ValueError(f"Unknown method: {method}")

    return phase


def compute_phase_coherence(
    phases_a: np.ndarray,
    phases_b: np.ndarray,
    n_bins: int = 36,
) -> float:
    """
    Compute phase coherence between two signals.

    Phase coherence measures how consistently two signals
    maintain their phase relationship over time.

    Args:
        phases_a: Phase values for signal A
        phases_b: Phase values for signal B
        n_bins: Number of bins for phase histogram

    Returns:
        Coherence value between 0 (no coherence) and 1 (perfect coherence)
    """
    analyzer = CircularCoordinationAnalyzer()
    phase_diff = analyzer.compute_phase_difference(phases_a, phases_b)
    valid_diff = phase_diff[~np.isnan(phase_diff)]

    if len(valid_diff) == 0:
        return 0.0

    # Compute R value of phase difference distribution
    _, R, _, _ = analyzer.circular_mean(valid_diff)

    return R
