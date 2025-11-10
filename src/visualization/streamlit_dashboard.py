"""
Interactive Streamlit Dashboard

Interactive web dashboard for exploring bibliometric visualizations.

Features:
- Overview page with KPIs
- Geographic analysis page
- Word cloud analysis page
- Timeline analysis page
- PDF export page
- Filters and interactive controls

Usage:
    streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import visualization modules
try:
    from .geographic_heatmap import GeographicHeatmap
    from .dynamic_wordcloud import DynamicWordCloud
    from .timeline_visualization import TimelineVisualization
    from .pdf_exporter import PDFExporter
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from visualization.geographic_heatmap import GeographicHeatmap
    from visualization.dynamic_wordcloud import DynamicWordCloud
    from visualization.timeline_visualization import TimelineVisualization
    from visualization.pdf_exporter import PDFExporter


class VisualizationDashboard:
    """
    Interactive dashboard for bibliometric visualizations.

    Provides a web-based interface for exploring geographic distribution,
    term analysis, temporal evolution, and generating PDF reports.
    """

    def __init__(self, data_path: str):
        """
        Initialize dashboard with data.

        Args:
            data_path: Path to unified CSV data file
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            st.error(f"Data file not found: {data_path}")
            st.stop()

        # Load data
        self._load_data()

        # Initialize visualization objects (with caching)
        self._init_visualizations()

    @st.cache_data
    def _load_data(_self):
        """Load and cache data."""
        return pd.read_csv(_self.data_path, encoding='utf-8')

    def _init_visualizations(self):
        """Initialize visualization objects."""
        try:
            self.geo_map = GeographicHeatmap(str(self.data_path))
            self.wordcloud = DynamicWordCloud(str(self.data_path))
            self.timeline = TimelineVisualization(str(self.data_path))
        except Exception as e:
            st.error(f"Error initializing visualizations: {e}")
            st.stop()

    def create_sidebar(self):
        """
        Create sidebar with controls and filters.
        """
        st.sidebar.title("📊 Controls")

        # Page selector
        page = st.sidebar.selectbox(
            "Select Page",
            ["📈 Overview", "🌍 Geographic", "☁️ Word Cloud", "📅 Timeline", "📄 Export PDF"]
        )

        st.sidebar.markdown("---")

        # Filters
        st.sidebar.subheader("🔍 Filters")

        # Load data for filters
        df = pd.read_csv(self.data_path)

        # Year filter
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])

            min_year = int(df['year'].min())
            max_year = int(df['year'].max())

            year_range = st.sidebar.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_range = None

        # Country filter (if available)
        countries = []
        if 'journal_conference' in df.columns:
            # Venue filter
            top_venues = df['journal_conference'].value_counts().head(10).index.tolist()
            selected_venues = st.sidebar.multiselect(
                "Venues",
                options=['All'] + top_venues,
                default=['All']
            )
        else:
            selected_venues = ['All']

        # Publication type filter
        if 'publication_type' in df.columns:
            pub_types = df['publication_type'].unique().tolist()
            selected_types = st.sidebar.multiselect(
                "Publication Type",
                options=['All'] + pub_types,
                default=['All']
            )
        else:
            selected_types = ['All']

        st.sidebar.markdown("---")

        # Action buttons
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()

        # Store filters in session state
        st.session_state['year_range'] = year_range
        st.session_state['selected_venues'] = selected_venues
        st.session_state['selected_types'] = selected_types

        return page

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply selected filters to dataframe."""
        # Year filter
        if 'year_range' in st.session_state and st.session_state['year_range']:
            df = df[
                (df['year'] >= st.session_state['year_range'][0]) &
                (df['year'] <= st.session_state['year_range'][1])
            ]

        # Venue filter
        if 'selected_venues' in st.session_state and 'All' not in st.session_state['selected_venues']:
            df = df[df['journal_conference'].isin(st.session_state['selected_venues'])]

        # Type filter
        if 'selected_types' in st.session_state and 'All' not in st.session_state['selected_types']:
            df = df[df['publication_type'].isin(st.session_state['selected_types'])]

        return df

    def show_overview_page(self):
        """
        Display overview page with KPIs.
        """
        st.title("📈 Bibliometric Analysis - Overview")

        # Load and filter data
        df = pd.read_csv(self.data_path)
        df = self._apply_filters(df)

        # Calculate KPIs
        total_pubs = len(df)

        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            years_covered = df['year'].nunique()
            year_range = f"{int(df['year'].min())} - {int(df['year'].max())}"

            # Calculate growth rate
            yearly_counts = df.groupby('year').size()
            if len(yearly_counts) > 1:
                growth_rate = yearly_counts.pct_change().mean() * 100
            else:
                growth_rate = 0
        else:
            years_covered = 0
            year_range = "N/A"
            growth_rate = 0

        # Unique countries (approximate from authors field)
        countries_count = 0  # Placeholder

        # Display KPIs in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📚 Total Publications",
                value=f"{total_pubs:,}",
                delta=f"+{growth_rate:.1f}% avg growth"
            )

        with col2:
            st.metric(
                label="🌍 Years Covered",
                value=years_covered,
                delta=year_range
            )

        with col3:
            venues_count = df['journal_conference'].nunique() if 'journal_conference' in df.columns else 0
            st.metric(
                label="📖 Unique Venues",
                value=venues_count
            )

        with col4:
            authors_count = df['authors'].nunique() if 'authors' in df.columns else 0
            st.metric(
                label="👥 Unique Authors",
                value=authors_count
            )

        st.markdown("---")

        # Summary visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Publications by Type")
            if 'publication_type' in df.columns:
                type_counts = df['publication_type'].value_counts()

                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Distribution by Publication Type",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Publication type data not available")

        with col2:
            st.subheader("Top 10 Venues")
            if 'journal_conference' in df.columns:
                venue_counts = df['journal_conference'].value_counts().head(10)

                fig = px.bar(
                    x=venue_counts.values,
                    y=venue_counts.index,
                    orientation='h',
                    title="Most Productive Venues",
                    labels={'x': 'Publications', 'y': 'Venue'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Venue data not available")

        # Yearly trend
        st.subheader("Publication Trend Over Time")
        if 'year' in df.columns:
            yearly_counts = df.groupby('year').size().reset_index(name='count')

            fig = px.line(
                yearly_counts,
                x='year',
                y='count',
                title="Publications per Year",
                markers=True
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Publications")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year data not available")

    def show_geographic_page(self):
        """
        Display geographic analysis page.
        """
        st.title("🌍 Geographic Distribution")

        # Extract affiliations
        with st.spinner("Analyzing geographic distribution..."):
            try:
                affiliations = self.geo_map.extract_author_affiliations()
                geo_data = self.geo_map.geocode_locations()
                stats = self.geo_map.generate_geographic_statistics()

                # Display map
                st.subheader("Global Distribution Map")

                # Create Plotly map
                df_plot = self.geo_map.calculate_publication_density(geo_data)

                fig = px.scatter_geo(
                    df_plot,
                    lat='lat',
                    lon='lon',
                    size='publications',
                    color='publications',
                    hover_name='country',
                    hover_data={'publications': True, 'percentage': ':.2f', 'lat': False, 'lon': False},
                    size_max=50,
                    color_continuous_scale='RdYlBu_r',
                    title='Geographic Distribution of Publications'
                )

                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top 10 Countries")
                    if 'top_10_countries' in stats:
                        top_df = pd.DataFrame(stats['top_10_countries'][:10])
                        st.dataframe(
                            top_df[['country', 'publications', 'percentage']],
                            use_container_width=True,
                            hide_index=True
                        )

                with col2:
                    st.subheader("Distribution by Continent")
                    if 'continent_distribution' in stats:
                        cont_df = pd.DataFrame(stats['continent_distribution'])

                        fig = px.bar(
                            cont_df,
                            x='continent',
                            y='publications',
                            title="Publications by Continent"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error in geographic analysis: {e}")

    def show_wordcloud_page(self):
        """
        Display word cloud analysis page.
        """
        st.title("☁️ Term Analysis")

        # Controls
        col1, col2 = st.columns([1, 3])

        with col1:
            max_terms = st.slider("Number of Terms", 20, 200, 100)
            weighting_method = st.selectbox(
                "Weighting Method",
                ["tfidf", "log_frequency", "frequency", "normalized"]
            )

        # Generate word cloud
        with st.spinner("Analyzing terms..."):
            try:
                terms = self.wordcloud.extract_and_process_terms(
                    sources=['abstract', 'keywords'],
                    max_terms=max_terms
                )

                weights = self.wordcloud.calculate_term_weights(
                    terms,
                    method=weighting_method
                )

                # Display interactive word cloud visualization
                st.subheader("Term Distribution")

                # Create scatter plot as word cloud alternative
                top_terms = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:max_terms]

                terms_list = [t[0] for t in top_terms]
                weights_list = [t[1] for t in top_terms]

                # Bar chart
                fig = px.bar(
                    x=weights_list[:20],
                    y=terms_list[:20],
                    orientation='h',
                    title=f"Top 20 Terms (Method: {weighting_method})",
                    labels={'x': 'Weight', 'y': 'Term'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Term table
                st.subheader("Term Frequencies")

                term_data = []
                for term, weight in top_terms[:50]:
                    term_data.append({
                        'Term': term,
                        'Frequency': terms[term],
                        'Weight': f"{weight:.4f}"
                    })

                st.dataframe(pd.DataFrame(term_data), use_container_width=True, hide_index=True)

                # Search functionality
                st.subheader("Search Terms")
                search_term = st.text_input("Enter term to search")

                if search_term:
                    if search_term.lower() in [t.lower() for t in terms.keys()]:
                        # Find exact match
                        matched_term = [t for t in terms.keys() if t.lower() == search_term.lower()][0]
                        st.success(f"**{matched_term}** found!")
                        st.write(f"Frequency: {terms[matched_term]}")
                        st.write(f"Weight: {weights.get(matched_term, 0):.4f}")
                    else:
                        st.warning(f"Term '{search_term}' not found in dataset")

            except Exception as e:
                st.error(f"Error in term analysis: {e}")

    def show_timeline_page(self):
        """
        Display timeline analysis page.
        """
        st.title("📅 Temporal Evolution")

        # Load and filter data
        df = self.timeline.extract_temporal_data()
        df = self._apply_filters(df)

        # Calculate statistics
        stats = self.timeline.calculate_yearly_statistics(df)

        # Controls
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Area", "Bar"]
        )

        # Main timeline
        st.subheader("Publication Timeline")

        yearly_counts = df.groupby('year').size().reset_index(name='count')

        if chart_type == "Line":
            fig = px.line(
                yearly_counts,
                x='year',
                y='count',
                title="Publications Over Time",
                markers=True
            )
        elif chart_type == "Area":
            fig = px.area(
                yearly_counts,
                x='year',
                y='count',
                title="Publications Over Time"
            )
        else:  # Bar
            fig = px.bar(
                yearly_counts,
                x='year',
                y='count',
                title="Publications Over Time"
            )

        fig.update_layout(xaxis_title="Year", yaxis_title="Publications", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)

        summary = stats['summary']

        with col1:
            st.metric("First Year", summary['first_year'])
            st.metric("Last Year", summary['last_year'])

        with col2:
            st.metric("Total Publications", f"{summary['total_publications']:,}")
            st.metric("Avg per Year", f"{summary['avg_per_year']:.1f}")

        with col3:
            st.metric("Most Productive Year", f"{summary['most_productive_year']} ({summary['most_productive_year_count']} pubs)")
            st.metric("Avg Growth Rate", f"{summary['avg_growth_rate']:.2f}%")

        # Yearly data table
        st.subheader("Year-by-Year Data")

        yearly_data = pd.DataFrame(stats['yearly_counts'])
        st.dataframe(yearly_data, use_container_width=True, hide_index=True)

        # Download button
        csv = yearly_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='yearly_data.csv',
            mime='text/csv'
        )

        # Future projection
        if stats.get('projection'):
            st.subheader("Future Projection")

            proj = stats['projection']
            st.write(f"**Trend**: {proj['slope']:+.2f} publications per year")
            st.write(f"**R² Score**: {proj['r_squared']:.4f}")

            proj_df = pd.DataFrame({
                'Year': proj['future_years'],
                'Projected Publications': [int(x) for x in proj['projected_counts']]
            })

            st.dataframe(proj_df, use_container_width=True, hide_index=True)

    def show_export_page(self):
        """
        Display PDF export page.
        """
        st.title("📄 Export PDF Report")

        st.markdown("""
        Generate a professional PDF report with all visualizations and statistics.
        """)

        # PDF Configuration
        st.subheader("Report Configuration")

        col1, col2 = st.columns(2)

        with col1:
            report_title = st.text_input("Report Title", "Bibliometric Analysis Report")
            report_subtitle = st.text_input("Subtitle", "Research Trends and Insights")

        with col2:
            author_name = st.text_input("Author/Analyst", "Research Team")
            institution = st.text_input("Institution", "")

        # Sections to include
        st.subheader("Sections to Include")

        col1, col2, col3 = st.columns(3)

        with col1:
            include_geo = st.checkbox("Geographic Analysis", value=True)

        with col2:
            include_wc = st.checkbox("Word Cloud Analysis", value=True)

        with col3:
            include_timeline = st.checkbox("Timeline Analysis", value=True)

        # Generate PDF button
        if st.button("🔄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report... This may take a moment."):
                try:
                    # Create temporary directory for images
                    temp_dir = Path('temp_pdf_images')
                    temp_dir.mkdir(exist_ok=True)

                    # Generate visualizations
                    geo_data = None
                    if include_geo:
                        try:
                            self.geo_map.extract_author_affiliations()
                            geo_data_dict = self.geo_map.geocode_locations()
                            geo_stats = self.geo_map.generate_geographic_statistics()

                            # Generate static map
                            map_path = temp_dir / 'geo_map.png'
                            self.geo_map.create_static_map(str(map_path), dpi=300)

                            geo_data = {
                                'map_image': str(map_path),
                                'statistics': geo_stats
                            }
                        except Exception as e:
                            st.warning(f"Could not generate geographic section: {e}")

                    wc_data = None
                    if include_wc:
                        try:
                            terms = self.wordcloud.extract_and_process_terms()
                            weights = self.wordcloud.calculate_term_weights(terms, method='tfidf')

                            # Generate word cloud
                            wc_path = temp_dir / 'wordcloud.png'
                            self.wordcloud.generate_wordcloud(weights, str(wc_path), style='scientific', dpi=300)

                            top_terms = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
                            wc_data = {
                                'image': str(wc_path),
                                'statistics': {
                                    'top_terms': [(term, terms[term], weight) for term, weight in top_terms]
                                }
                            }
                        except Exception as e:
                            st.warning(f"Could not generate word cloud section: {e}")

                    timeline_data = None
                    if include_timeline:
                        try:
                            df = self.timeline.extract_temporal_data()
                            temporal_stats = self.timeline.calculate_yearly_statistics(df)

                            # Generate timeline plot
                            timeline_path = temp_dir / 'timeline.png'
                            self.timeline.create_timeline_plot(df, str(timeline_path), dpi=300)

                            timeline_data = {
                                'images': [str(timeline_path)],
                                'statistics': temporal_stats
                            }
                        except Exception as e:
                            st.warning(f"Could not generate timeline section: {e}")

                    # Generate PDF
                    pdf_path = Path('output/dashboard_report.pdf')
                    pdf_path.parent.mkdir(exist_ok=True)

                    exporter = PDFExporter(str(pdf_path))

                    # Overall statistics
                    df = pd.read_csv(self.data_path)
                    overall_stats = {
                        'total_publications': len(df),
                        'years_covered': df['year'].nunique() if 'year' in df.columns else 0,
                        'unique_venues': df['journal_conference'].nunique() if 'journal_conference' in df.columns else 0
                    }

                    # Processing info
                    import platform
                    processing_info = {
                        'version': '1.0.0',
                        'python_version': platform.python_version(),
                        'sources': ['Research Databases']
                    }

                    # Generate PDF
                    exporter.generate_complete_pdf(
                        title=report_title,
                        subtitle=report_subtitle,
                        authors=[author_name] if author_name else None,
                        institution=institution,
                        geographic_data=geo_data,
                        wordcloud_data=wc_data,
                        timeline_data=timeline_data,
                        overall_stats=overall_stats,
                        processing_info=processing_info
                    )

                    st.success(f"✅ PDF generated successfully!")

                    # Offer download
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name='bibliometric_report.pdf',
                            mime='application/pdf'
                        )

                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    def run_dashboard(self):
        """
        Main method to run the dashboard.
        """
        # Page configuration
        st.set_page_config(
            page_title="Bibliometric Analysis Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create sidebar and get selected page
        page = self.create_sidebar()

        # Route to appropriate page
        if page == "📈 Overview":
            self.show_overview_page()
        elif page == "🌍 Geographic":
            self.show_geographic_page()
        elif page == "☁️ Word Cloud":
            self.show_wordcloud_page()
        elif page == "📅 Timeline":
            self.show_timeline_page()
        elif page == "📄 Export PDF":
            self.show_export_page()


# Main entry point
def main():
    """Main function to run dashboard."""
    # Default data path
    default_path = Path('data/sample/complete_sample_data.csv')

    if not default_path.exists():
        st.error(f"Data file not found: {default_path}")
        st.info("Please ensure you have sample data or provide a custom path")
        st.stop()

    # Create and run dashboard
    dashboard = VisualizationDashboard(str(default_path))
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
