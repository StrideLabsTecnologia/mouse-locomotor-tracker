#!/usr/bin/env python3
"""
Scientific Report Generator
===========================

Generates comprehensive scientific reports in multiple formats:
- LaTeX (for academic papers)
- Markdown (for documentation)
- HTML (for web viewing)
- JSON (for data interchange)

Follows standards for:
- Journal of Neurophysiology
- Nature Methods
- PNAS

Author: Stride Labs
Version: 1.0.0
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class ReportMetadata:
    """Metadata for scientific report."""
    experiment_id: str = "EXP001"
    subject_id: str = "MOUSE001"
    experimenter: str = "Unknown"
    institution: str = "Stride Labs"
    date: str = ""
    species: str = "Mus musculus"
    strain: str = "C57BL/6"
    sex: str = "Unknown"
    age_weeks: Optional[int] = None
    weight_g: Optional[float] = None
    apparatus: str = "Treadmill"
    software_version: str = "1.0.0"

    def __post_init__(self):
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d")


class ScientificReportGenerator:
    """
    Generate scientific reports for gait analysis.

    Example:
        >>> generator = ScientificReportGenerator()
        >>> generator.generate_latex_report(metrics, metadata, "report.tex")
        >>> generator.generate_markdown_report(metrics, metadata, "report.md")
    """

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate_latex_report(
        self,
        metrics: Dict,
        metadata: ReportMetadata,
        output_path: Union[str, Path],
        include_figures: bool = True,
    ) -> Path:
        """
        Generate LaTeX report suitable for academic publication.

        Args:
            metrics: Gait metrics dictionary
            metadata: Experiment metadata
            output_path: Output file path
            include_figures: Include figure placeholders
        """
        output_path = Path(output_path)

        # Extract metrics
        spatio = metrics.get('spatiotemporal', {})
        kin = metrics.get('kinematic', {})
        var = metrics.get('variability', {})
        dist = metrics.get('distance_duration', {})
        activity = metrics.get('activity', {})
        quality = metrics.get('quality', {})

        vel = spatio.get('velocity', {})
        stride = spatio.get('stride_length', {})
        cadence = spatio.get('cadence', {})

        latex_content = r"""
\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{siunitx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}

% Page setup
\geometry{margin=2.5cm}

% SI units setup
\sisetup{
    separate-uncertainty = true,
    multi-part-units = single
}

\title{Gait Analysis Report\\
\large """ + f"Experiment: {metadata.experiment_id}" + r"""}
\author{""" + metadata.experimenter + r"""\\
\small """ + metadata.institution + r"""}
\date{""" + metadata.date + r"""}

\begin{document}

\maketitle

% ============================================================================
\section{Experiment Information}
% ============================================================================

\begin{table}[h]
\centering
\caption{Experiment and subject details}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Experiment ID & """ + metadata.experiment_id + r""" \\
Subject ID & """ + metadata.subject_id + r""" \\
Species & """ + metadata.species + r""" \\
Strain & """ + metadata.strain + r""" \\
Sex & """ + metadata.sex + r""" \\
Age & """ + (f"{metadata.age_weeks} weeks" if metadata.age_weeks else "N/A") + r""" \\
Weight & """ + (f"{metadata.weight_g:.1f} g" if metadata.weight_g else "N/A") + r""" \\
Apparatus & """ + metadata.apparatus + r""" \\
Software Version & """ + metadata.software_version + r""" \\
Date & """ + metadata.date + r""" \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================================
\section{Methods}
% ============================================================================

Locomotion was recorded using a treadmill apparatus. Video was acquired at
\SI{30}{\hertz} and analyzed using Mouse Locomotor Tracker (v""" + metadata.software_version + r""").
Position tracking was performed using motion-based detection. Spatiotemporal
gait parameters were calculated following established protocols
\cite{catwalk2023, digigait2021}.

% ============================================================================
\section{Results}
% ============================================================================

\subsection{Spatiotemporal Parameters}

\begin{table}[h]
\centering
\caption{Spatiotemporal gait parameters. Values are mean $\pm$ SD.}
\label{tab:spatiotemporal}
\begin{tabular}{lcc}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Unit} \\
\midrule
Velocity & $""" + f"{vel.get('mean', 0):.2f} \\pm {vel.get('sd', 0):.2f}" + r"""$ & \si{\centi\meter\per\second} \\
Maximum Velocity & """ + f"{vel.get('max', 0):.2f}" + r""" & \si{\centi\meter\per\second} \\
Stride Length & $""" + f"{stride.get('mean', 0):.2f} \\pm {stride.get('sd', 0):.2f}" + r"""$ & \si{\centi\meter} \\
Cadence & $""" + f"{cadence.get('mean', 0):.2f} \\pm {cadence.get('sd', 0):.2f}" + r"""$ & steps/s \\
Step Cycle & $""" + f"{spatio.get('step_cycle', {}).get('mean', 0):.3f} \\pm {spatio.get('step_cycle', {}).get('sd', 0):.3f}" + r"""$ & \si{\second} \\
Swing Time & """ + f"{spatio.get('swing_time', {}).get('percent', 40):.0f}" + r"""\% & - \\
Stance Time & """ + f"{spatio.get('stance_time', {}).get('percent', 60):.0f}" + r"""\% & - \\
Duty Factor & """ + f"{spatio.get('duty_factor', 0.6):.2f}" + r""" & - \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Kinematic Parameters}

\begin{table}[h]
\centering
\caption{Kinematic parameters derived from velocity profile.}
\label{tab:kinematic}
\begin{tabular}{lcc}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Unit} \\
\midrule
Mean Acceleration & $""" + f"{kin.get('acceleration', {}).get('mean', 0):.2f} \\pm {kin.get('acceleration', {}).get('sd', 0):.2f}" + r"""$ & \si{\centi\meter\per\second\squared} \\
Peak Acceleration & """ + f"{kin.get('acceleration', {}).get('max', 0):.2f}" + r""" & \si{\centi\meter\per\second\squared} \\
Mean Jerk & $""" + f"{kin.get('jerk', {}).get('mean', 0):.2f} \\pm {kin.get('jerk', {}).get('sd', 0):.2f}" + r"""$ & \si{\centi\meter\per\second\cubed} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Activity Classification}

The subject exhibited the following activity distribution:
\begin{itemize}
    \item Resting: """ + f"{activity.get('time_resting_pct', 0):.1f}" + r"""\%
    \item Walking ($<$\SI{10}{\centi\meter\per\second}): """ + f"{activity.get('time_walking_pct', 0):.1f}" + r"""\%
    \item Trotting (\SI{10}{}-\SI{25}{\centi\meter\per\second}): """ + f"{activity.get('time_trotting_pct', 0):.1f}" + r"""\%
    \item Galloping ($>$\SI{25}{\centi\meter\per\second}): """ + f"{activity.get('time_galloping_pct', 0):.1f}" + r"""\%
\end{itemize}

Total distance traveled: \SI{""" + f"{dist.get('total_distance_cm', 0):.1f}" + r"""}{\centi\meter} over \SI{""" + f"{dist.get('total_duration_s', 0):.1f}" + r"""}{\second}.

\subsection{Gait Variability}

\begin{table}[h]
\centering
\caption{Gait variability and regularity metrics.}
\label{tab:variability}
\begin{tabular}{lc}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Regularity Index & """ + f"{var.get('regularity_index', 0):.1f}" + r"""\% \\
Stride Variability (CV) & """ + f"{var.get('stride_variability', 0):.1f}" + r"""\% \\
Velocity Stability & """ + f"{var.get('velocity_stability', 0):.1f}" + r"""\% \\
Velocity CV & """ + f"{vel.get('cv', 0):.1f}" + r"""\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Data Quality}

Tracking was successful for """ + f"{quality.get('tracking_rate', 0):.1f}" + r"""\% of frames
(""" + f"{quality.get('n_frames_tracked', 0)}" + r"""/""" + f"{quality.get('n_frames_total', 0)}" + r""" frames).
A total of """ + f"{quality.get('n_strides_detected', 0)}" + r""" complete strides were detected.
Overall data quality score: """ + f"{quality.get('data_quality_score', 0):.1f}" + r"""/100.

"""
        if include_figures:
            latex_content += r"""
% ============================================================================
\section{Figures}
% ============================================================================

\begin{figure}[h]
\centering
% \includegraphics[width=0.8\textwidth]{velocity_timeseries.pdf}
\caption{Velocity profile over time. Dashed line indicates mean velocity.
Background shading indicates activity classification (gray: rest, blue: walk,
yellow: trot, red: gallop).}
\label{fig:velocity}
\end{figure}

\begin{figure}[h]
\centering
% \includegraphics[width=0.8\textwidth]{gait_summary.pdf}
\caption{Comprehensive gait analysis summary showing (A) velocity statistics,
(B) stride metrics, (C) activity distribution, (D) gait phase composition,
(E) kinematic parameters, (F) regularity metrics, (G) distance and duration,
(H) data quality, and (I) summary table.}
\label{fig:summary}
\end{figure}

"""

        latex_content += r"""
% ============================================================================
\section{References}
% ============================================================================

\begin{thebibliography}{9}

\bibitem{catwalk2023}
Lakes, E.H., Allen, K.D.
\newblock CatWalk XT gait parameters: a review of reported parameters in
pre-clinical studies of multiple central nervous system and peripheral
nervous system disease models.
\newblock \textit{Front. Behav. Neurosci.} 17, 1147784 (2023).

\bibitem{digigait2021}
Mouse Specifics, Inc.
\newblock DigiGait Imaging System.
\newblock \url{https://mousespecifics.com/digigait/} (2021).

\bibitem{deeplabcut}
Mathis, A., et al.
\newblock DeepLabCut: markerless pose estimation of user-defined body parts
with deep learning.
\newblock \textit{Nat. Neurosci.} 21, 1281-1289 (2018).

\end{thebibliography}

\end{document}
"""

        with open(output_path, 'w') as f:
            f.write(latex_content)

        print(f"LaTeX report saved: {output_path}")
        return output_path

    def generate_markdown_report(
        self,
        metrics: Dict,
        metadata: ReportMetadata,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate Markdown report for documentation.

        Args:
            metrics: Gait metrics dictionary
            metadata: Experiment metadata
            output_path: Output file path
        """
        output_path = Path(output_path)

        # Extract metrics
        spatio = metrics.get('spatiotemporal', {})
        kin = metrics.get('kinematic', {})
        var = metrics.get('variability', {})
        dist = metrics.get('distance_duration', {})
        activity = metrics.get('activity', {})
        quality = metrics.get('quality', {})

        vel = spatio.get('velocity', {})
        stride = spatio.get('stride_length', {})
        cadence = spatio.get('cadence', {})

        md_content = f"""# Gait Analysis Report

**Experiment:** {metadata.experiment_id}
**Subject:** {metadata.subject_id}
**Date:** {metadata.date}
**Experimenter:** {metadata.experimenter}
**Institution:** {metadata.institution}

---

## Experiment Details

| Parameter | Value |
|-----------|-------|
| Species | {metadata.species} |
| Strain | {metadata.strain} |
| Sex | {metadata.sex} |
| Age | {f"{metadata.age_weeks} weeks" if metadata.age_weeks else "N/A"} |
| Weight | {f"{metadata.weight_g:.1f} g" if metadata.weight_g else "N/A"} |
| Apparatus | {metadata.apparatus} |
| Software | Mouse Locomotor Tracker v{metadata.software_version} |

---

## Results

### Spatiotemporal Parameters

| Parameter | Mean ± SD | Unit |
|-----------|-----------|------|
| Velocity | {vel.get('mean', 0):.2f} ± {vel.get('sd', 0):.2f} | cm/s |
| Max Velocity | {vel.get('max', 0):.2f} | cm/s |
| Stride Length | {stride.get('mean', 0):.2f} ± {stride.get('sd', 0):.2f} | cm |
| Cadence | {cadence.get('mean', 0):.2f} ± {cadence.get('sd', 0):.2f} | steps/s |
| Step Cycle | {spatio.get('step_cycle', {}).get('mean', 0):.3f} ± {spatio.get('step_cycle', {}).get('sd', 0):.3f} | s |
| Swing Phase | {spatio.get('swing_time', {}).get('percent', 40):.0f}% | - |
| Stance Phase | {spatio.get('stance_time', {}).get('percent', 60):.0f}% | - |
| Duty Factor | {spatio.get('duty_factor', 0.6):.2f} | - |

### Kinematic Parameters

| Parameter | Mean ± SD | Unit |
|-----------|-----------|------|
| Acceleration | {kin.get('acceleration', {}).get('mean', 0):.2f} ± {kin.get('acceleration', {}).get('sd', 0):.2f} | cm/s² |
| Peak Acceleration | {kin.get('acceleration', {}).get('max', 0):.2f} | cm/s² |
| Jerk | {kin.get('jerk', {}).get('mean', 0):.2f} ± {kin.get('jerk', {}).get('sd', 0):.2f} | cm/s³ |

### Activity Classification

| Activity | Percentage |
|----------|------------|
| Resting | {activity.get('time_resting_pct', 0):.1f}% |
| Walking | {activity.get('time_walking_pct', 0):.1f}% |
| Trotting | {activity.get('time_trotting_pct', 0):.1f}% |
| Galloping | {activity.get('time_galloping_pct', 0):.1f}% |

**Total Distance:** {dist.get('total_distance_cm', 0):.1f} cm
**Total Duration:** {dist.get('total_duration_s', 0):.1f} s
**Active Time:** {dist.get('active_duration_s', 0):.1f} s

### Gait Variability

| Metric | Value |
|--------|-------|
| Regularity Index | {var.get('regularity_index', 0):.1f}% |
| Stride Variability (CV) | {var.get('stride_variability', 0):.1f}% |
| Velocity Stability | {var.get('velocity_stability', 0):.1f}% |

### Data Quality

| Metric | Value |
|--------|-------|
| Tracking Rate | {quality.get('tracking_rate', 0):.1f}% |
| Frames Analyzed | {quality.get('n_frames_total', 0)} |
| Strides Detected | {quality.get('n_strides_detected', 0)} |
| Quality Score | {quality.get('data_quality_score', 0):.1f}/100 |

---

## Methods

Locomotion was recorded at 30 Hz using a treadmill apparatus. Video analysis
was performed using Mouse Locomotor Tracker (Stride Labs). Position tracking
utilized motion-based detection with ROI constraints. Gait parameters were
calculated following CatWalk XT and DigiGait standards.

---

*Report generated by Mouse Locomotor Tracker v{metadata.software_version}*
*{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        with open(output_path, 'w') as f:
            f.write(md_content)

        print(f"Markdown report saved: {output_path}")
        return output_path

    def generate_json_report(
        self,
        metrics: Dict,
        metadata: ReportMetadata,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate JSON report for data interchange.

        Args:
            metrics: Gait metrics dictionary
            metadata: Experiment metadata
            output_path: Output file path
        """
        output_path = Path(output_path)

        report = {
            "metadata": {
                "experiment_id": metadata.experiment_id,
                "subject_id": metadata.subject_id,
                "experimenter": metadata.experimenter,
                "institution": metadata.institution,
                "date": metadata.date,
                "species": metadata.species,
                "strain": metadata.strain,
                "sex": metadata.sex,
                "age_weeks": metadata.age_weeks,
                "weight_g": metadata.weight_g,
                "apparatus": metadata.apparatus,
                "software_version": metadata.software_version,
                "generated_at": datetime.now().isoformat(),
            },
            "metrics": metrics,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        print(f"JSON report saved: {output_path}")
        return output_path


def generate_scientific_reports(
    metrics: Dict,
    output_dir: Union[str, Path],
    experiment_id: str = "EXP001",
    subject_id: str = "MOUSE001",
    experimenter: str = "Unknown",
) -> Dict[str, Path]:
    """
    Generate all report formats at once.

    Args:
        metrics: Gait metrics dictionary
        output_dir: Output directory
        experiment_id: Experiment identifier
        subject_id: Subject identifier
        experimenter: Experimenter name

    Returns:
        Dictionary of output paths by format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = ReportMetadata(
        experiment_id=experiment_id,
        subject_id=subject_id,
        experimenter=experimenter,
    )

    generator = ScientificReportGenerator()

    paths = {}
    paths['latex'] = generator.generate_latex_report(
        metrics, metadata, output_dir / f"{experiment_id}_report.tex"
    )
    paths['markdown'] = generator.generate_markdown_report(
        metrics, metadata, output_dir / f"{experiment_id}_report.md"
    )
    paths['json'] = generator.generate_json_report(
        metrics, metadata, output_dir / f"{experiment_id}_report.json"
    )

    return paths


if __name__ == "__main__":
    print("Scientific Report Generator - Demo")
    print("=" * 50)

    # Demo metrics
    demo_metrics = {
        'spatiotemporal': {
            'velocity': {'mean': 12.5, 'sd': 3.2, 'max': 28.3},
            'stride_length': {'mean': 4.2, 'sd': 0.8},
            'cadence': {'mean': 3.1, 'sd': 0.5},
            'step_cycle': {'mean': 0.32, 'sd': 0.05},
            'swing_time': {'percent': 40},
            'stance_time': {'percent': 60},
            'duty_factor': 0.60,
        },
        'kinematic': {
            'acceleration': {'mean': 45.2, 'sd': 12.3, 'max': 98.5},
            'jerk': {'mean': 320.5, 'sd': 85.2},
        },
        'variability': {
            'regularity_index': 85.3,
            'stride_variability': 19.0,
            'velocity_stability': 74.4,
        },
        'distance_duration': {
            'total_distance_cm': 156.8,
            'total_duration_s': 12.5,
            'active_duration_s': 10.2,
        },
        'activity': {
            'time_resting_pct': 18.4,
            'time_walking_pct': 45.2,
            'time_trotting_pct': 28.6,
            'time_galloping_pct': 7.8,
        },
        'quality': {
            'tracking_rate': 98.5,
            'n_frames_total': 300,
            'n_strides_detected': 25,
            'data_quality_score': 92.3,
        }
    }

    paths = generate_scientific_reports(
        demo_metrics,
        output_dir="./output",
        experiment_id="DEMO001",
        subject_id="MOUSE_TEST",
        experimenter="Demo User",
    )

    print("\nGenerated reports:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")
