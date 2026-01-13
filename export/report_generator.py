"""
Report Generator - Mouse Locomotor Tracker
==========================================

Generate comprehensive PDF reports with analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import asdict, is_dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import reportlab, fall back to simple text report if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available. PDF reports will be limited.")


class ReportGenerator:
    """
    Generate comprehensive PDF reports for locomotor analysis.

    Creates professional reports with:
    - Executive summary
    - Velocity metrics
    - Gait analysis
    - Coordination analysis
    - Embedded plots
    """

    def __init__(self, page_size: str = 'letter'):
        """
        Initialize report generator.

        Args:
            page_size: Page size ('letter' or 'A4')
        """
        self.page_size = letter if page_size == 'letter' else A4
        self.styles = None

        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title_Custom',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e')
        ))

        self.styles.add(ParagraphStyle(
            name='Section',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e')
        ))

        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=5,
            spaceAfter=5
        ))

    def generate(
        self,
        results: Dict[str, Any],
        plots_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        video_name: Optional[str] = None
    ) -> Path:
        """
        Generate PDF report.

        Args:
            results: Analysis results dictionary
            plots_dir: Directory containing plot images
            output_path: Output path for PDF
            video_name: Name of analyzed video

        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_text_report(results, output_path, video_name)

        output_path = Path(output_path) if output_path else Path('report.pdf')

        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Build content
        story = []

        # Title
        story.append(Paragraph(
            "Mouse Locomotor Analysis Report",
            self.styles['Title_Custom']
        ))

        # Metadata
        story.append(Paragraph(
            f"Video: {video_name or 'Unknown'}<br/>"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            f"System: Mouse Locomotor Tracker v1.0.0",
            self.styles['Normal']
        ))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.extend(self._create_summary_section(results))

        # Velocity Section
        story.extend(self._create_velocity_section(results))

        # Gait Section
        story.extend(self._create_gait_section(results))

        # Coordination Section
        story.extend(self._create_coordination_section(results))

        # Plots
        if plots_dir:
            story.extend(self._add_plots(plots_dir))

        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF report: {output_path}")

        return output_path

    def _create_summary_section(self, results: Dict[str, Any]) -> List:
        """Create executive summary section."""
        content = []
        content.append(Paragraph("Executive Summary", self.styles['Section']))

        # Summary table
        data = [['Metric', 'Value', 'Status']]

        # Add velocity
        if 'velocity' in results:
            v = results['velocity']
            if is_dataclass(v):
                v = asdict(v)
            avg_speed = v.get('avg_speed', 0)
            status = '✓ Normal' if 5 < avg_speed < 30 else '⚠ Check'
            data.append(['Average Speed', f"{avg_speed:.2f} cm/s", status])

        # Add gait
        if 'gait' in results:
            g = results['gait']
            if is_dataclass(g):
                g = asdict(g)
            num_steps = g.get('num_steps', 0)
            data.append(['Total Steps', str(num_steps), '✓'])

        # Add coordination
        if 'coordination' in results:
            coord = results['coordination']
            r_values = []
            for pair, metrics in coord.items():
                if is_dataclass(metrics):
                    metrics = asdict(metrics)
                r_values.append(metrics.get('r_value', 0))
            mean_r = sum(r_values) / len(r_values) if r_values else 0
            status = '✓ Good' if mean_r > 0.7 else '⚠ Moderate' if mean_r > 0.5 else '✗ Poor'
            data.append(['Coordination (R)', f"{mean_r:.3f}", status])

        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        content.append(table)
        content.append(Spacer(1, 20))

        return content

    def _create_velocity_section(self, results: Dict[str, Any]) -> List:
        """Create velocity analysis section."""
        content = []
        content.append(Paragraph("Velocity Analysis", self.styles['Section']))

        if 'velocity' not in results:
            content.append(Paragraph("No velocity data available.", self.styles['Normal']))
            return content

        v = results['velocity']
        if is_dataclass(v):
            v = asdict(v)

        metrics = [
            f"<b>Average Speed:</b> {v.get('avg_speed', 0):.2f} cm/s",
            f"<b>Peak Speed:</b> {v.get('peak_speed', 0):.2f} cm/s",
            f"<b>Peak Acceleration:</b> {v.get('peak_acceleration', 0):.2f} cm/s²",
            f"<b>Drag Events:</b> {v.get('num_drag', 0)}",
            f"<b>Recovery Events:</b> {v.get('num_recovery', 0)}",
        ]

        for m in metrics:
            content.append(Paragraph(m, self.styles['Metric']))

        content.append(Spacer(1, 20))
        return content

    def _create_gait_section(self, results: Dict[str, Any]) -> List:
        """Create gait analysis section."""
        content = []
        content.append(Paragraph("Gait Analysis", self.styles['Section']))

        if 'gait' not in results:
            content.append(Paragraph("No gait data available.", self.styles['Normal']))
            return content

        g = results['gait']
        if is_dataclass(g):
            g = asdict(g)

        content.append(Paragraph(
            f"<b>Movement Duration:</b> {g.get('movement_duration', 0):.2f} s",
            self.styles['Metric']
        ))
        content.append(Paragraph(
            f"<b>Total Steps:</b> {g.get('num_steps', 0)}",
            self.styles['Metric']
        ))

        # Cadence table
        cadence = g.get('cadence', {})
        stride_len = g.get('stride_length', {})

        if cadence:
            data = [['Limb', 'Cadence (Hz)', 'Stride Length (cm)']]
            for limb in ['LH', 'RH', 'LF', 'RF']:
                cad = cadence.get(limb, 0)
                sl = stride_len.get(limb, 0)
                data.append([limb, f"{cad:.2f}", f"{sl:.2f}"])

            table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            content.append(Spacer(1, 10))
            content.append(table)

        content.append(Spacer(1, 20))
        return content

    def _create_coordination_section(self, results: Dict[str, Any]) -> List:
        """Create coordination analysis section."""
        content = []
        content.append(Paragraph("Limb Coordination (Circular Statistics)", self.styles['Section']))

        if 'coordination' not in results:
            content.append(Paragraph("No coordination data available.", self.styles['Normal']))
            return content

        coord = results['coordination']

        # Explanation
        content.append(Paragraph(
            "<i>R value indicates coordination strength (0=random, 1=perfect). "
            "Phase angle shows timing relationship between limbs.</i>",
            self.styles['Normal']
        ))
        content.append(Spacer(1, 10))

        # Coordination table
        data = [['Limb Pair', 'Description', 'Phase (°)', 'R Value', 'Quality']]

        pair_descriptions = {
            'LHRH': 'Hindlimb L-R',
            'LHLF': 'Homolateral L',
            'RHRF': 'Homolateral R',
            'LFRH': 'Diagonal 1',
            'RFLH': 'Diagonal 2',
            'LFRF': 'Forelimb L-R'
        }

        for pair, metrics in coord.items():
            if is_dataclass(metrics):
                metrics = asdict(metrics)

            r = metrics.get('r_value', 0)
            phase = metrics.get('mean_phase', 0)
            quality = 'Good' if r > 0.7 else 'Moderate' if r > 0.5 else 'Poor'

            data.append([
                pair,
                pair_descriptions.get(pair, ''),
                f"{phase:.1f}",
                f"{r:.3f}",
                quality
            ])

        table = Table(data, colWidths=[1*inch, 1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        content.append(table)

        content.append(Spacer(1, 20))
        return content

    def _add_plots(self, plots_dir: Path) -> List:
        """Add plot images to report."""
        content = []
        plots_dir = Path(plots_dir)

        # Look for plot files
        plot_files = list(plots_dir.glob('*.png')) + list(plots_dir.glob('*.pdf'))

        if not plot_files:
            return content

        content.append(PageBreak())
        content.append(Paragraph("Analysis Plots", self.styles['Section']))

        for plot_file in plot_files[:5]:  # Limit to 5 plots
            try:
                if plot_file.suffix == '.png':
                    img = Image(str(plot_file), width=5*inch, height=4*inch)
                    content.append(img)
                    content.append(Paragraph(
                        f"<i>{plot_file.stem}</i>",
                        self.styles['Normal']
                    ))
                    content.append(Spacer(1, 20))
            except Exception as e:
                logger.warning(f"Could not add plot {plot_file}: {e}")

        return content

    def _generate_text_report(
        self,
        results: Dict[str, Any],
        output_path: Path,
        video_name: Optional[str]
    ) -> Path:
        """Generate simple text report when reportlab not available."""
        output_path = Path(output_path).with_suffix('.txt')

        lines = [
            "=" * 60,
            "MOUSE LOCOMOTOR ANALYSIS REPORT",
            "=" * 60,
            f"Video: {video_name or 'Unknown'}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if 'velocity' in results:
            v = results['velocity']
            if is_dataclass(v):
                v = asdict(v)
            lines.extend([
                "VELOCITY ANALYSIS",
                "-" * 40,
                f"Average Speed: {v.get('avg_speed', 0):.2f} cm/s",
                f"Peak Speed: {v.get('peak_speed', 0):.2f} cm/s",
                ""
            ])

        if 'gait' in results:
            g = results['gait']
            if is_dataclass(g):
                g = asdict(g)
            lines.extend([
                "GAIT ANALYSIS",
                "-" * 40,
                f"Total Steps: {g.get('num_steps', 0)}",
                f"Movement Duration: {g.get('movement_duration', 0):.2f}s",
                ""
            ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Generated text report: {output_path}")
        return output_path
