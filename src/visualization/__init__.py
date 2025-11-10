"""
Visualization Module

Provides visualization tools for bibliometric analysis including:
- Geographic heatmaps
- Dynamic word clouds
- Timeline visualizations
- PDF export
- Term visualizations
- Network graphs
- Temporal analysis
"""

from .geographic_heatmap import GeographicHeatmap
from .dynamic_wordcloud import DynamicWordCloud
from .timeline_visualization import TimelineVisualization
from .pdf_exporter import PDFExporter

__all__ = [
    'GeographicHeatmap',
    'DynamicWordCloud',
    'TimelineVisualization',
    'PDFExporter',
]

__version__ = '1.0.0'
