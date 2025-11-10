"""
Timeline Visualization Module

Visualizes temporal evolution of scientific publications with comprehensive
statistical analysis and interactive visualizations.

Features:
- Temporal data extraction and validation
- Yearly statistics and growth analysis
- Professional timeline plots
- Stacked area charts
- Venue-based timeline analysis
- Interactive Plotly timelines
- Publication burst detection
- Statistical reports in Markdown
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import warnings
from loguru import logger
from collections import Counter, defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')


class TimelineVisualization:
    """
    Visualizes temporal evolution of publications.

    This class provides comprehensive temporal analysis and visualization
    capabilities for scientific production over time.
    """

    def __init__(self, unified_data_path: str):
        """
        Initialize timeline visualization.

        Args:
            unified_data_path: Path to unified CSV data file
        """
        self.unified_data_path = Path(unified_data_path)

        # Validate file exists
        if not self.unified_data_path.exists():
            raise FileNotFoundError(f"Unified data file not found: {unified_data_path}")

        # Load data
        logger.info(f"Loading data from: {unified_data_path}")
        self.df = pd.read_csv(self.unified_data_path, encoding='utf-8')
        logger.info(f"Loaded {len(self.df)} records")

        # Cache for temporal data
        self._temporal_df = None
        self._yearly_stats = None

        logger.success("TimelineVisualization initialized successfully")

    def extract_temporal_data(self) -> pd.DataFrame:
        """
        Extract and structure temporal data.

        Required fields:
        - year: Publication year
        - publication_type: Type (journal/conference)
        - journal_conference: Venue name
        - title, authors: Metadata

        Returns:
            Clean and validated DataFrame with temporal data
        """
        logger.info("Extracting temporal data")

        df = self.df.copy()

        # Ensure year column exists
        if 'year' not in df.columns:
            logger.error("Year column not found in data")
            raise ValueError("Data must contain 'year' column")

        # Convert year to numeric
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Remove invalid years
        original_count = len(df)
        df = df.dropna(subset=['year'])
        df = df[df['year'] > 1900]  # Reasonable year range
        df = df[df['year'] <= datetime.now().year + 1]  # Future threshold

        if len(df) < original_count:
            logger.warning(f"Removed {original_count - len(df)} records with invalid years")

        # Convert year to integer
        df['year'] = df['year'].astype(int)

        # Ensure other required fields exist
        if 'publication_type' not in df.columns:
            df['publication_type'] = 'unknown'

        if 'journal_conference' not in df.columns:
            df['journal_conference'] = 'Unknown Venue'

        # Clean venue names
        df['journal_conference'] = df['journal_conference'].fillna('Unknown Venue')
        df['journal_conference'] = df['journal_conference'].astype(str)

        # Normalize publication types
        df['publication_type'] = df['publication_type'].str.lower()
        df['publication_type'] = df['publication_type'].replace({
            'journal article': 'journal',
            'conference paper': 'conference',
            'proceedings': 'conference',
            '': 'unknown',
            'nan': 'unknown'
        })

        # Sort by year
        df = df.sort_values('year')

        logger.info(f"Extracted temporal data: {len(df)} records from "
                   f"{df['year'].min()} to {df['year'].max()}")

        # Cache result
        self._temporal_df = df

        return df

    def calculate_yearly_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive yearly statistics.

        Statistics:
        - Total publications per year
        - Breakdown by venue
        - Year-over-year growth rate
        - 3-year moving average
        - Future trend projection (linear regression)

        Args:
            df: Temporal DataFrame (uses cached if None)

        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating yearly statistics")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Publications per year
        yearly_counts = df.groupby('year').size().reset_index(name='count')

        # Breakdown by type
        type_yearly = df.groupby(['year', 'publication_type']).size().unstack(fill_value=0)

        # Breakdown by venue
        venue_yearly = df.groupby(['year', 'journal_conference']).size().unstack(fill_value=0)

        # Growth rate calculation
        yearly_counts['growth_rate'] = yearly_counts['count'].pct_change() * 100

        # 3-year moving average
        yearly_counts['moving_avg_3y'] = yearly_counts['count'].rolling(window=3, min_periods=1).mean()

        # Linear regression for trend projection
        from scipy.stats import linregress

        years = yearly_counts['year'].values
        counts = yearly_counts['count'].values

        if len(years) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(years, counts)

            # Project next 3 years
            future_years = np.arange(years[-1] + 1, years[-1] + 4)
            projected_counts = slope * future_years + intercept

            projection = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'future_years': future_years.tolist(),
                'projected_counts': projected_counts.tolist()
            }
        else:
            projection = None

        # Compile statistics
        stats = {
            'yearly_counts': yearly_counts.to_dict('records'),
            'type_breakdown': type_yearly.to_dict(),
            'venue_breakdown': venue_yearly.to_dict(),
            'summary': {
                'first_year': int(df['year'].min()),
                'last_year': int(df['year'].max()),
                'total_years': int(df['year'].nunique()),
                'total_publications': int(len(df)),
                'avg_per_year': float(len(df) / df['year'].nunique()),
                'most_productive_year': int(yearly_counts.loc[yearly_counts['count'].idxmax(), 'year']),
                'most_productive_year_count': int(yearly_counts['count'].max()),
                'avg_growth_rate': float(yearly_counts['growth_rate'].mean())
            },
            'projection': projection
        }

        # Cache result
        self._yearly_stats = stats

        logger.info(f"Calculated statistics for {stats['summary']['total_years']} years")

        return stats

    def create_timeline_plot(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = 'timeline.png',
        dpi: int = 300
    ):
        """
        Create main timeline plot.

        Features:
        - X-axis: Years
        - Y-axis: Number of publications
        - Main line: Total per year
        - Secondary lines: By publication type
        - Shaded area showing trend
        - Markers on data points
        - Annotations on peaks/valleys
        - Professional styling (300 DPI)

        Args:
            df: Temporal DataFrame
            output_path: Path to save PNG
            dpi: Resolution
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Matplotlib/Seaborn not installed")
            return

        logger.info("Creating timeline plot")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Calculate yearly counts
        yearly_total = df.groupby('year').size().reset_index(name='total')

        # By type
        type_counts = df.groupby(['year', 'publication_type']).size().unstack(fill_value=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)

        # Plot total (main line)
        ax.plot(
            yearly_total['year'],
            yearly_total['total'],
            marker='o',
            linewidth=2.5,
            markersize=8,
            color='#2E86AB',
            label='Total Publications',
            zorder=3
        )

        # Plot by type (secondary lines)
        colors = {'journal': '#A23B72', 'conference': '#F18F01', 'unknown': '#C73E1D'}
        for pub_type in type_counts.columns:
            if pub_type in colors:
                ax.plot(
                    type_counts.index,
                    type_counts[pub_type],
                    marker='s',
                    linewidth=1.5,
                    markersize=5,
                    color=colors.get(pub_type, '#666666'),
                    label=pub_type.capitalize(),
                    alpha=0.7,
                    linestyle='--',
                    zorder=2
                )

        # Add trend line (polynomial fit)
        if len(yearly_total) >= 3:
            z = np.polyfit(yearly_total['year'], yearly_total['total'], 2)
            p = np.poly1d(z)
            trend_line = p(yearly_total['year'])

            ax.fill_between(
                yearly_total['year'],
                trend_line - yearly_total['total'].std() * 0.5,
                trend_line + yearly_total['total'].std() * 0.5,
                alpha=0.2,
                color='#2E86AB',
                label='Trend Band',
                zorder=1
            )

        # Annotate peaks and valleys
        if len(yearly_total) > 0:
            max_idx = yearly_total['total'].idxmax()
            max_year = yearly_total.loc[max_idx, 'year']
            max_count = yearly_total.loc[max_idx, 'total']

            ax.annotate(
                f'Peak: {max_count}',
                xy=(max_year, max_count),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black')
            )

        # Styling
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Publications', fontsize=14, fontweight='bold')
        ax.set_title('Temporal Evolution of Scientific Publications', fontsize=16, fontweight='bold', pad=20)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

        # Tight layout
        plt.tight_layout()

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"Timeline plot saved to: {output_file}")

    def create_stacked_area_chart(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = 'stacked_area.png',
        group_by: str = 'publication_type',
        dpi: int = 300
    ):
        """
        Create stacked area chart showing composition over time.

        Shows how publication composition changes over time.

        Args:
            df: Temporal DataFrame
            output_path: Path to save PNG
            group_by: Field to group by ('publication_type', 'journal_conference')
            dpi: Resolution
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Matplotlib/Seaborn not installed")
            return

        logger.info(f"Creating stacked area chart grouped by {group_by}")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Group by year and specified field
        if group_by == 'journal_conference':
            # Take top 5 venues
            top_venues = df['journal_conference'].value_counts().head(5).index
            df_filtered = df[df['journal_conference'].isin(top_venues)]
            grouped = df_filtered.groupby(['year', group_by]).size().unstack(fill_value=0)
        else:
            grouped = df.groupby(['year', group_by]).size().unstack(fill_value=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)

        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(grouped.columns)))

        # Create stacked area
        ax.stackplot(
            grouped.index,
            *[grouped[col] for col in grouped.columns],
            labels=grouped.columns,
            colors=colors,
            alpha=0.8
        )

        # Styling
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Publications', fontsize=14, fontweight='bold')
        title = f'Publication Composition Over Time (by {group_by.replace("_", " ").title()})'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)

        # Legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # Tight layout
        plt.tight_layout()

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"Stacked area chart saved to: {output_file}")

    def create_venue_timeline(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = 'venue_timeline.png',
        top_n_venues: int = 10,
        visualization_type: str = 'heatmap',
        dpi: int = 300
    ):
        """
        Create timeline broken down by venue.

        Visualization options:
        1. 'heatmap': Years (X) vs Venues (Y), color = frequency
        2. 'lines': One line per venue
        3. 'small_multiples': Mini-chart per venue

        Args:
            df: Temporal DataFrame
            output_path: Path to save PNG
            top_n_venues: Number of top venues to show
            visualization_type: Type of visualization
            dpi: Resolution
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Matplotlib/Seaborn not installed")
            return

        logger.info(f"Creating venue timeline ({visualization_type})")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Get top N venues
        top_venues = df['journal_conference'].value_counts().head(top_n_venues).index
        df_filtered = df[df['journal_conference'].isin(top_venues)]

        # Create year-venue matrix
        venue_yearly = df_filtered.groupby(['year', 'journal_conference']).size().unstack(fill_value=0)

        if visualization_type == 'heatmap':
            # Heatmap visualization
            fig, ax = plt.subplots(figsize=(14, max(8, top_n_venues * 0.5)), dpi=dpi)

            sns.heatmap(
                venue_yearly.T,
                cmap='YlOrRd',
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Publications'},
                linewidths=0.5,
                ax=ax
            )

            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Venue', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n_venues} Venues: Publication Timeline', fontsize=14, fontweight='bold')

            plt.tight_layout()

        elif visualization_type == 'lines':
            # Multiple lines visualization
            fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)

            for venue in venue_yearly.columns:
                ax.plot(
                    venue_yearly.index,
                    venue_yearly[venue],
                    marker='o',
                    label=venue,
                    linewidth=2,
                    markersize=5
                )

            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Publications', fontsize=12, fontweight='bold')
            ax.set_title(f'Publication Trends: Top {top_n_venues} Venues', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

        elif visualization_type == 'small_multiples':
            # Small multiples visualization
            n_cols = 3
            n_rows = (top_n_venues + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3), dpi=dpi)
            axes = axes.flatten() if top_n_venues > 1 else [axes]

            for idx, venue in enumerate(venue_yearly.columns[:top_n_venues]):
                ax = axes[idx]
                ax.plot(
                    venue_yearly.index,
                    venue_yearly[venue],
                    marker='o',
                    color='#2E86AB',
                    linewidth=2
                )
                ax.fill_between(
                    venue_yearly.index,
                    venue_yearly[venue],
                    alpha=0.3,
                    color='#2E86AB'
                )
                ax.set_title(venue, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Year', fontsize=9)
                ax.set_ylabel('Pubs', fontsize=9)

            # Hide unused subplots
            for idx in range(top_n_venues, len(axes)):
                axes[idx].axis('off')

            fig.suptitle(f'Publication Trends by Venue', fontsize=14, fontweight='bold')
            plt.tight_layout()

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"Venue timeline saved to: {output_file}")

    def create_interactive_timeline(
        self,
        df: Optional[pd.DataFrame] = None,
        output_html: str = 'timeline_interactive.html'
    ):
        """
        Create interactive timeline with Plotly.

        Features:
        - Hover: Details of year (article titles)
        - Zoom temporal
        - Toggle series (show/hide venues)
        - Range slider
        - Filter buttons by type
        - Exportable to PNG from browser

        Args:
            df: Temporal DataFrame
            output_html: Path to save HTML file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed")
            return

        logger.info("Creating interactive timeline")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Calculate yearly totals
        yearly_total = df.groupby('year').size().reset_index(name='count')

        # By type
        type_counts = df.groupby(['year', 'publication_type']).size().reset_index(name='count')

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Publication Timeline', 'Publications by Type'),
            vertical_spacing=0.12
        )

        # Main timeline
        fig.add_trace(
            go.Scatter(
                x=yearly_total['year'],
                y=yearly_total['count'],
                mode='lines+markers',
                name='Total Publications',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Year %{x}</b><br>Publications: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # By type (stacked bar in subplot)
        for pub_type in df['publication_type'].unique():
            type_data = type_counts[type_counts['publication_type'] == pub_type]
            fig.add_trace(
                go.Bar(
                    x=type_data['year'],
                    y=type_data['count'],
                    name=pub_type.capitalize(),
                    hovertemplate=f'<b>{pub_type.capitalize()}</b><br>Year: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive Publication Timeline',
                font=dict(size=20, family='Arial, sans-serif')
            ),
            xaxis=dict(
                title='Year',
                rangeslider=dict(visible=True),
                type='linear'
            ),
            yaxis=dict(title='Publications'),
            xaxis2=dict(title='Year'),
            yaxis2=dict(title='Publications'),
            hovermode='x unified',
            height=800,
            showlegend=True,
            barmode='stack'
        )

        # Add range selector buttons
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            row=1, col=1
        )

        # Save
        output_file = Path(output_html)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        logger.success(f"Interactive timeline saved to: {output_file}")

    def create_publication_burst_analysis(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = 'burst_analysis.png',
        threshold_std: float = 1.5,
        dpi: int = 300
    ):
        """
        Identify and visualize publication bursts.

        Analysis:
        - Detects years with anomalous growth
        - Identifies emerging themes in those years
        - Marks burst periods

        Args:
            df: Temporal DataFrame
            output_path: Path to save PNG
            threshold_std: Standard deviations above mean to consider a burst
            dpi: Resolution
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Matplotlib/Seaborn not installed")
            return

        logger.info("Creating publication burst analysis")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Calculate yearly counts
        yearly_counts = df.groupby('year').size().reset_index(name='count')

        # Calculate statistics
        mean_count = yearly_counts['count'].mean()
        std_count = yearly_counts['count'].std()

        # Identify bursts (counts above threshold)
        threshold = mean_count + threshold_std * std_count
        bursts = yearly_counts[yearly_counts['count'] > threshold]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)

        # Plot timeline
        ax.plot(
            yearly_counts['year'],
            yearly_counts['count'],
            marker='o',
            linewidth=2,
            color='#2E86AB',
            label='Publications per Year'
        )

        # Plot mean line
        ax.axhline(
            y=mean_count,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Mean ({mean_count:.1f})'
        )

        # Plot threshold line
        ax.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Burst Threshold ({threshold:.1f})'
        )

        # Highlight burst years
        for _, burst in bursts.iterrows():
            ax.axvspan(
                burst['year'] - 0.4,
                burst['year'] + 0.4,
                alpha=0.3,
                color='orange',
                zorder=0
            )

            # Annotate burst
            ax.annotate(
                f"Burst\n{burst['count']}",
                xy=(burst['year'], burst['count']),
                xytext=(0, 15),
                textcoords='offset points',
                fontsize=9,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red')
            )

        # Styling
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Publications', fontsize=14, fontweight='bold')
        ax.set_title('Publication Burst Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"Burst analysis saved to: {output_file}")

        # Return burst years for further analysis
        return bursts['year'].tolist()

    def generate_temporal_statistics_report(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = 'temporal_statistics.md'
    ):
        """
        Generate statistical report in Markdown.

        Includes:
        - First/last publication year
        - Most productive year
        - Average annual growth rate
        - Most consistent venues
        - Future predictions
        - Detailed year-by-year table

        Args:
            df: Temporal DataFrame
            output_path: Path to save Markdown file
        """
        logger.info("Generating temporal statistics report")

        if df is None:
            if self._temporal_df is None:
                df = self.extract_temporal_data()
            else:
                df = self._temporal_df

        # Calculate statistics if not cached
        if self._yearly_stats is None:
            stats = self.calculate_yearly_statistics(df)
        else:
            stats = self._yearly_stats

        # Start building report
        report_lines = []

        # Header
        report_lines.append("# Temporal Statistics Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary
        report_lines.append("## Summary")
        report_lines.append("")
        summary = stats['summary']
        report_lines.append(f"- **First Publication:** {summary['first_year']}")
        report_lines.append(f"- **Last Publication:** {summary['last_year']}")
        report_lines.append(f"- **Time Span:** {summary['total_years']} years")
        report_lines.append(f"- **Total Publications:** {summary['total_publications']}")
        report_lines.append(f"- **Average per Year:** {summary['avg_per_year']:.2f}")
        report_lines.append(f"- **Most Productive Year:** {summary['most_productive_year']} ({summary['most_productive_year_count']} publications)")
        report_lines.append(f"- **Average Growth Rate:** {summary['avg_growth_rate']:.2f}% per year")
        report_lines.append("")

        # Top venues
        report_lines.append("## Top 10 Most Productive Venues")
        report_lines.append("")
        top_venues = df['journal_conference'].value_counts().head(10)
        report_lines.append("| Rank | Venue | Publications |")
        report_lines.append("|------|-------|--------------|")
        for idx, (venue, count) in enumerate(top_venues.items(), 1):
            report_lines.append(f"| {idx} | {venue} | {count} |")
        report_lines.append("")

        # Yearly breakdown
        report_lines.append("## Year-by-Year Breakdown")
        report_lines.append("")
        report_lines.append("| Year | Publications | Growth Rate | 3-Year Avg |")
        report_lines.append("|------|--------------|-------------|------------|")

        for record in stats['yearly_counts']:
            year = record['year']
            count = record['count']
            growth = record.get('growth_rate', 0)
            moving_avg = record.get('moving_avg_3y', count)

            growth_str = f"{growth:+.1f}%" if not np.isnan(growth) else "N/A"
            report_lines.append(f"| {year} | {count} | {growth_str} | {moving_avg:.1f} |")

        report_lines.append("")

        # Future projection
        if stats.get('projection'):
            proj = stats['projection']
            report_lines.append("## Future Projection")
            report_lines.append("")
            report_lines.append(f"Based on linear regression (R² = {proj['r_squared']:.4f}):")
            report_lines.append("")
            report_lines.append("| Year | Projected Publications |")
            report_lines.append("|------|------------------------|")

            for year, count in zip(proj['future_years'], proj['projected_counts']):
                report_lines.append(f"| {year} | {max(0, int(count))} |")

            report_lines.append("")
            report_lines.append(f"**Trend:** {proj['slope']:+.2f} publications per year")
            report_lines.append("")

        # Publication type distribution
        report_lines.append("## Publication Type Distribution")
        report_lines.append("")
        type_dist = df['publication_type'].value_counts()
        report_lines.append("| Type | Count | Percentage |")
        report_lines.append("|------|-------|------------|")
        total = len(df)
        for pub_type, count in type_dist.items():
            percentage = (count / total * 100)
            report_lines.append(f"| {pub_type.capitalize()} | {count} | {percentage:.1f}% |")
        report_lines.append("")

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.success(f"Temporal statistics report saved to: {output_file}")

        return str(output_file)


# Example usage
if __name__ == "__main__":
    logger.info("Timeline Visualization Module - Example Usage")

    # This would normally be called with real unified data
    # timeline = TimelineVisualization('data/processed/unified_data.csv')
    #
    # # Extract temporal data
    # df = timeline.extract_temporal_data()
    #
    # # Calculate statistics
    # stats = timeline.calculate_yearly_statistics(df)
    #
    # # Create visualizations
    # timeline.create_timeline_plot(df, 'output/timeline.png')
    # timeline.create_stacked_area_chart(df, 'output/stacked_area.png')
    # timeline.create_venue_timeline(df, 'output/venue_heatmap.png', visualization_type='heatmap')
    # timeline.create_interactive_timeline(df, 'output/timeline_interactive.html')
    # timeline.create_publication_burst_analysis(df, 'output/burst_analysis.png')
    #
    # # Generate report
    # timeline.generate_temporal_statistics_report(df, 'output/temporal_report.md')

    logger.info("Example complete!")
