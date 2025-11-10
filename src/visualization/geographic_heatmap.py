"""
Geographic Heatmap Visualization Module

Generates interactive and static geographic heatmaps showing the distribution
of scientific publications across countries and institutions.

Features:
- Extract author affiliations using NER
- Geocode locations to coordinates
- Create interactive maps with Folium
- Create interactive maps with Plotly
- Create static high-quality maps for PDF export
- Generate comprehensive geographic statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import re
from loguru import logger
from collections import Counter, defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class GeographicHeatmap:
    """
    Generates geographic heatmap with distribution of publications.

    This class provides comprehensive geographic analysis and visualization
    of scientific production, including interactive and static maps.
    """

    def __init__(self, unified_data_path: str):
        """
        Initialize geographic heatmap analyzer.

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

        # Initialize caches
        self._affiliation_cache = {}
        self._geocoding_cache = {}
        self._country_database = self._load_country_database()

        # Load spaCy model if available
        self._nlp = None
        self._load_spacy_model()

        logger.success("GeographicHeatmap initialized successfully")

    def _load_spacy_model(self):
        """Load spaCy model for Named Entity Recognition."""
        try:
            import spacy
            try:
                # Try loading English model
                self._nlp = spacy.load('en_core_web_sm')
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                logger.info("Will use pattern-based extraction as fallback")
        except ImportError:
            logger.warning("spaCy not installed. Will use pattern-based extraction only")

    def _load_country_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Load country database with coordinates and metadata.

        Returns:
            Dictionary mapping country names to their data
        """
        # Comprehensive country database with coordinates
        countries = {
            'USA': {'name': 'United States', 'lat': 37.0902, 'lon': -95.7129, 'continent': 'North America'},
            'United States': {'name': 'United States', 'lat': 37.0902, 'lon': -95.7129, 'continent': 'North America'},
            'UK': {'name': 'United Kingdom', 'lat': 55.3781, 'lon': -3.4360, 'continent': 'Europe'},
            'United Kingdom': {'name': 'United Kingdom', 'lat': 55.3781, 'lon': -3.4360, 'continent': 'Europe'},
            'China': {'name': 'China', 'lat': 35.8617, 'lon': 104.1954, 'continent': 'Asia'},
            'Germany': {'name': 'Germany', 'lat': 51.1657, 'lon': 10.4515, 'continent': 'Europe'},
            'France': {'name': 'France', 'lat': 46.2276, 'lon': 2.2137, 'continent': 'Europe'},
            'Japan': {'name': 'Japan', 'lat': 36.2048, 'lon': 138.2529, 'continent': 'Asia'},
            'Canada': {'name': 'Canada', 'lat': 56.1304, 'lon': -106.3468, 'continent': 'North America'},
            'Australia': {'name': 'Australia', 'lat': -25.2744, 'lon': 133.7751, 'continent': 'Oceania'},
            'India': {'name': 'India', 'lat': 20.5937, 'lon': 78.9629, 'continent': 'Asia'},
            'Brazil': {'name': 'Brazil', 'lat': -14.2350, 'lon': -51.9253, 'continent': 'South America'},
            'Spain': {'name': 'Spain', 'lat': 40.4637, 'lon': -3.7492, 'continent': 'Europe'},
            'Italy': {'name': 'Italy', 'lat': 41.8719, 'lon': 12.5674, 'continent': 'Europe'},
            'South Korea': {'name': 'South Korea', 'lat': 35.9078, 'lon': 127.7669, 'continent': 'Asia'},
            'Netherlands': {'name': 'Netherlands', 'lat': 52.1326, 'lon': 5.2913, 'continent': 'Europe'},
            'Switzerland': {'name': 'Switzerland', 'lat': 46.8182, 'lon': 8.2275, 'continent': 'Europe'},
            'Sweden': {'name': 'Sweden', 'lat': 60.1282, 'lon': 18.6435, 'continent': 'Europe'},
            'Russia': {'name': 'Russia', 'lat': 61.5240, 'lon': 105.3188, 'continent': 'Europe/Asia'},
            'Mexico': {'name': 'Mexico', 'lat': 23.6345, 'lon': -102.5528, 'continent': 'North America'},
            'Singapore': {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'continent': 'Asia'},
            'Israel': {'name': 'Israel', 'lat': 31.0461, 'lon': 34.8516, 'continent': 'Asia'},
            'Belgium': {'name': 'Belgium', 'lat': 50.5039, 'lon': 4.4699, 'continent': 'Europe'},
            'Denmark': {'name': 'Denmark', 'lat': 56.2639, 'lon': 9.5018, 'continent': 'Europe'},
            'Norway': {'name': 'Norway', 'lat': 60.4720, 'lon': 8.4689, 'continent': 'Europe'},
            'Finland': {'name': 'Finland', 'lat': 61.9241, 'lon': 25.7482, 'continent': 'Europe'},
            'Austria': {'name': 'Austria', 'lat': 47.5162, 'lon': 14.5501, 'continent': 'Europe'},
            'Poland': {'name': 'Poland', 'lat': 51.9194, 'lon': 19.1451, 'continent': 'Europe'},
            'Portugal': {'name': 'Portugal', 'lat': 39.3999, 'lon': -8.2245, 'continent': 'Europe'},
            'Greece': {'name': 'Greece', 'lat': 39.0742, 'lon': 21.8243, 'continent': 'Europe'},
            'Turkey': {'name': 'Turkey', 'lat': 38.9637, 'lon': 35.2433, 'continent': 'Europe/Asia'},
            'Argentina': {'name': 'Argentina', 'lat': -38.4161, 'lon': -63.6167, 'continent': 'South America'},
            'Chile': {'name': 'Chile', 'lat': -35.6751, 'lon': -71.5430, 'continent': 'South America'},
            'Colombia': {'name': 'Colombia', 'lat': 4.5709, 'lon': -74.2973, 'continent': 'South America'},
            'South Africa': {'name': 'South Africa', 'lat': -30.5595, 'lon': 22.9375, 'continent': 'Africa'},
            'Egypt': {'name': 'Egypt', 'lat': 26.8206, 'lon': 30.8025, 'continent': 'Africa'},
            'Saudi Arabia': {'name': 'Saudi Arabia', 'lat': 23.8859, 'lon': 45.0792, 'continent': 'Asia'},
            'Iran': {'name': 'Iran', 'lat': 32.4279, 'lon': 53.6880, 'continent': 'Asia'},
            'Pakistan': {'name': 'Pakistan', 'lat': 30.3753, 'lon': 69.3451, 'continent': 'Asia'},
            'Thailand': {'name': 'Thailand', 'lat': 15.8700, 'lon': 100.9925, 'continent': 'Asia'},
            'Malaysia': {'name': 'Malaysia', 'lat': 4.2105, 'lon': 101.9758, 'continent': 'Asia'},
            'Indonesia': {'name': 'Indonesia', 'lat': -0.7893, 'lon': 113.9213, 'continent': 'Asia'},
            'New Zealand': {'name': 'New Zealand', 'lat': -40.9006, 'lon': 174.8860, 'continent': 'Oceania'},
        }

        # Add university-to-country mapping
        self._university_to_country = {
            'MIT': 'USA',
            'Stanford': 'USA',
            'Harvard': 'USA',
            'Berkeley': 'USA',
            'Cambridge': 'UK',
            'Oxford': 'UK',
            'ETH': 'Switzerland',
            'Tsinghua': 'China',
            'Peking University': 'China',
            'University of Tokyo': 'Japan',
            'TU Munich': 'Germany',
            'Max Planck': 'Germany',
            'CNRS': 'France',
            'Sorbonne': 'France',
            'Toronto': 'Canada',
            'McGill': 'Canada',
            'Sydney': 'Australia',
            'Melbourne': 'Australia',
            'IIT': 'India',
            'NUS': 'Singapore',
            'Nanyang': 'Singapore',
            'KAIST': 'South Korea',
            'Seoul National': 'South Korea',
        }

        return countries

    def extract_author_affiliations(self, author_field: str = 'authors') -> Dict[str, Dict[str, str]]:
        """
        Extract country/institution from author information.

        Strategies:
        1. Pattern-based extraction (e.g., "John Doe (MIT, USA)")
        2. Named Entity Recognition with spaCy
        3. Institution-to-country mapping

        Args:
            author_field: Name of the column containing author information

        Returns:
            Dictionary mapping article IDs to affiliation data:
            {
                'article_id': {
                    'first_author': str,
                    'institution': str,
                    'city': str,
                    'country': str
                }
            }
        """
        logger.info("Extracting author affiliations")

        affiliations = {}

        for idx, row in self.df.iterrows():
            article_id = row.get('id', f'article_{idx}')
            authors_text = str(row.get(author_field, ''))

            if not authors_text or authors_text == 'nan':
                continue

            # Strategy 1: Pattern-based extraction
            affiliation = self._extract_affiliation_patterns(authors_text)

            # Strategy 2: NER if spaCy is available
            if not affiliation.get('country') and self._nlp:
                ner_result = self._extract_affiliation_ner(authors_text)
                affiliation.update({k: v for k, v in ner_result.items() if v})

            # Strategy 3: Institution mapping
            if not affiliation.get('country') and affiliation.get('institution'):
                country = self._map_institution_to_country(affiliation['institution'])
                if country:
                    affiliation['country'] = country

            if affiliation:
                affiliations[article_id] = affiliation

        logger.info(f"Extracted affiliations for {len(affiliations)} articles")
        self._affiliation_cache = affiliations

        return affiliations

    def _extract_affiliation_patterns(self, text: str) -> Dict[str, str]:
        """
        Extract affiliation using regex patterns.

        Patterns recognized:
        - "Author (Institution, Country)"
        - "Author, Institution, Country"
        - "Author [Institution] Country"
        """
        result = {
            'first_author': '',
            'institution': '',
            'city': '',
            'country': ''
        }

        # Extract first author (before first comma, semicolon, or parenthesis)
        author_match = re.match(r'^([^,;(\[]+)', text)
        if author_match:
            result['first_author'] = author_match.group(1).strip()

        # Pattern 1: (Institution, Country) or (Institution, City, Country)
        paren_pattern = r'\(([^)]+)\)'
        paren_matches = re.findall(paren_pattern, text)

        if paren_matches:
            affil_text = paren_matches[0]
            parts = [p.strip() for p in affil_text.split(',')]

            if len(parts) >= 2:
                # Check if last part is a country
                potential_country = parts[-1]
                if self._is_country(potential_country):
                    result['country'] = self._normalize_country_name(potential_country)
                    result['institution'] = parts[0]
                    if len(parts) == 3:
                        result['city'] = parts[1]

        # Pattern 2: Comma-separated (Author, Institution, Country)
        if not result['country']:
            parts = [p.strip() for p in text.split(',')]
            if len(parts) >= 3:
                potential_country = parts[-1]
                if self._is_country(potential_country):
                    result['country'] = self._normalize_country_name(potential_country)
                    result['institution'] = parts[1]

        # Pattern 3: Semicolon-separated authors (take first)
        if not result['country'] and ';' in text:
            first_author_text = text.split(';')[0]
            return self._extract_affiliation_patterns(first_author_text)

        return result

    def _extract_affiliation_ner(self, text: str) -> Dict[str, str]:
        """Extract affiliation using spaCy NER."""
        result = {
            'first_author': '',
            'institution': '',
            'city': '',
            'country': ''
        }

        if not self._nlp:
            return result

        doc = self._nlp(text)

        # Extract entities
        for ent in doc.ents:
            if ent.label_ == 'GPE':  # Geo-Political Entity
                # Could be country or city
                if self._is_country(ent.text):
                    if not result['country']:
                        result['country'] = self._normalize_country_name(ent.text)
                else:
                    if not result['city']:
                        result['city'] = ent.text

            elif ent.label_ == 'ORG':  # Organization
                if not result['institution']:
                    result['institution'] = ent.text

            elif ent.label_ == 'PERSON':  # Person
                if not result['first_author']:
                    result['first_author'] = ent.text

        return result

    def _is_country(self, text: str) -> bool:
        """Check if text matches a known country."""
        text_normalized = text.strip().upper()

        # Check country database
        for country_key, country_data in self._country_database.items():
            if (text_normalized == country_key.upper() or
                text_normalized == country_data['name'].upper()):
                return True

        # Check common country codes
        country_codes = ['USA', 'UK', 'EU', 'UAE', 'PRC']
        if text_normalized in country_codes:
            return True

        return False

    def _normalize_country_name(self, country: str) -> str:
        """Normalize country name to standard form."""
        country_normalized = country.strip()

        # Check if it's in the database
        if country_normalized in self._country_database:
            return self._country_database[country_normalized]['name']

        # Check uppercase match
        for key, data in self._country_database.items():
            if country_normalized.upper() == key.upper():
                return data['name']
            if country_normalized.upper() == data['name'].upper():
                return data['name']

        return country_normalized

    def _map_institution_to_country(self, institution: str) -> Optional[str]:
        """Map institution name to country."""
        for inst_key, country in self._university_to_country.items():
            if inst_key.lower() in institution.lower():
                return self._country_database[country]['name']

        return None

    def geocode_locations(self, countries: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Geocode countries to coordinates.

        Args:
            countries: Optional list of countries to geocode.
                      If None, extracts from affiliations.

        Returns:
            Dictionary mapping country names to their data:
            {
                'country_name': {
                    'lat': float,
                    'lon': float,
                    'count': int,
                    'continent': str
                }
            }
        """
        logger.info("Geocoding locations")

        # If no affiliations extracted yet, extract them
        if not self._affiliation_cache:
            self.extract_author_affiliations()

        # Count publications per country
        country_counts = Counter()

        for affiliation in self._affiliation_cache.values():
            country = affiliation.get('country', '')
            if country:
                country_counts[country] += 1

        # Geocode each country
        geo_data = {}

        for country, count in country_counts.items():
            # Look up in database
            coords = self._get_country_coordinates(country)

            if coords:
                geo_data[country] = {
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'count': count,
                    'continent': coords.get('continent', 'Unknown')
                }

        logger.info(f"Geocoded {len(geo_data)} countries")
        self._geocoding_cache = geo_data

        return geo_data

    def _get_country_coordinates(self, country: str) -> Optional[Dict[str, Any]]:
        """Get coordinates for a country from database."""
        # Direct lookup
        if country in self._country_database:
            return self._country_database[country]

        # Case-insensitive search
        for key, data in self._country_database.items():
            if country.lower() == key.lower() or country.lower() == data['name'].lower():
                return data

        return None

    def calculate_publication_density(self, geo_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate density of publications by country.

        Args:
            geo_data: Optional geocoded data. If None, uses cached data.

        Returns:
            DataFrame with publication density metrics
        """
        logger.info("Calculating publication density")

        if geo_data is None:
            if not self._geocoding_cache:
                self.geocode_locations()
            geo_data = self._geocoding_cache

        # Calculate metrics
        total_pubs = sum(data['count'] for data in geo_data.values())

        density_data = []
        for country, data in geo_data.items():
            density_data.append({
                'country': country,
                'publications': data['count'],
                'percentage': (data['count'] / total_pubs * 100) if total_pubs > 0 else 0,
                'lat': data['lat'],
                'lon': data['lon'],
                'continent': data['continent']
            })

        # Create DataFrame and sort by publications
        df_density = pd.DataFrame(density_data)
        df_density = df_density.sort_values('publications', ascending=False)
        df_density = df_density.reset_index(drop=True)

        logger.info(f"Calculated density for {len(df_density)} countries")

        return df_density

    def create_interactive_map(self, geo_data: Optional[Dict] = None,
                              output_html: str = 'geographic_heatmap.html'):
        """
        Generate interactive map with Folium.

        Features:
        - OpenStreetMap base layer
        - Circle markers proportional to publications
        - Tooltips with country stats
        - Heatmap layer
        - Marker clustering
        - Layer controls

        Args:
            geo_data: Optional geocoded data
            output_html: Path to save HTML file
        """
        try:
            import folium
            from folium.plugins import HeatMap, MarkerCluster
        except ImportError:
            logger.error("Folium not installed. Install with: pip install folium")
            return

        logger.info("Creating interactive Folium map")

        if geo_data is None:
            if not self._geocoding_cache:
                self.geocode_locations()
            geo_data = self._geocoding_cache

        if not geo_data:
            logger.warning("No geographic data available")
            return

        # Create base map centered on world view
        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles='OpenStreetMap'
        )

        # Add alternative tile layers
        folium.TileLayer('CartoDB positron', name='CartoDB').add_to(m)

        # Prepare data for heatmap
        heat_data = [[data['lat'], data['lon'], data['count']]
                     for data in geo_data.values()]

        # Add heatmap layer
        HeatMap(
            heat_data,
            name='Heatmap',
            min_opacity=0.2,
            max_zoom=10,
            radius=25,
            blur=35,
            gradient={0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)

        # Add marker cluster
        marker_cluster = MarkerCluster(name='Markers').add_to(m)

        # Calculate max count for scaling
        max_count = max(data['count'] for data in geo_data.values())

        # Add markers
        for country, data in geo_data.items():
            # Calculate marker size (proportional to publications)
            radius = 5 + (data['count'] / max_count * 30)

            # Create tooltip
            tooltip_text = f"""
            <b>{country}</b><br>
            Publications: {data['count']}<br>
            Continent: {data['continent']}
            """

            # Color based on publication count
            if data['count'] >= max_count * 0.7:
                color = 'red'
            elif data['count'] >= max_count * 0.4:
                color = 'orange'
            elif data['count'] >= max_count * 0.2:
                color = 'blue'
            else:
                color = 'lightblue'

            # Add circle marker
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=radius,
                popup=tooltip_text,
                tooltip=country,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(marker_cluster)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        output_path = Path(output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))

        logger.success(f"Interactive Folium map saved to: {output_path}")

        return m

    def create_plotly_map(self, geo_data: Optional[Dict] = None,
                         output_html: str = 'geographic_plotly.html'):
        """
        Generate interactive map with Plotly.

        Advantages:
        - Better for static image export
        - Smooth animations
        - Dashboard integration

        Args:
            geo_data: Optional geocoded data
            output_html: Path to save HTML file
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Install with: pip install plotly")
            return

        logger.info("Creating interactive Plotly map")

        if geo_data is None:
            if not self._geocoding_cache:
                self.geocode_locations()
            geo_data = self._geocoding_cache

        if not geo_data:
            logger.warning("No geographic data available")
            return

        # Prepare data for Plotly
        df_plot = self.calculate_publication_density(geo_data)

        # Create scatter geo plot
        fig = px.scatter_geo(
            df_plot,
            lat='lat',
            lon='lon',
            size='publications',
            color='publications',
            hover_name='country',
            hover_data={
                'publications': True,
                'percentage': ':.2f',
                'continent': True,
                'lat': False,
                'lon': False
            },
            size_max=50,
            color_continuous_scale='RdYlBu_r',
            title='Geographic Distribution of Scientific Publications',
            labels={'publications': 'Publications'}
        )

        # Update layout
        fig.update_layout(
            geo=dict(
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                projection_type='natural earth',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                showcountries=True,
                countrycolor='rgb(204, 204, 204)'
            ),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        # Save figure
        output_path = Path(output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

        logger.success(f"Interactive Plotly map saved to: {output_path}")

        return fig

    def create_static_map(self, geo_data: Optional[Dict] = None,
                         output_png: str = 'geographic_map.png',
                         dpi: int = 300):
        """
        Generate static high-quality map for PDF export.

        Uses matplotlib with cartopy for professional cartographic visualization.

        Args:
            geo_data: Optional geocoded data
            output_png: Path to save PNG file
            dpi: Resolution (default 300 DPI for print quality)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.error("Matplotlib not installed")
            return

        logger.info("Creating static map")

        if geo_data is None:
            if not self._geocoding_cache:
                self.geocode_locations()
            geo_data = self._geocoding_cache

        if not geo_data:
            logger.warning("No geographic data available")
            return

        # Prepare data
        df_plot = self.calculate_publication_density(geo_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)

        # Try to use cartopy if available
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = plt.axes(projection=ccrs.Robinson())

            # Add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

            ax.set_global()

            # Plot points
            max_count = df_plot['publications'].max()

            for _, row in df_plot.iterrows():
                size = 100 + (row['publications'] / max_count * 1000)

                # Color based on count
                if row['publications'] >= max_count * 0.7:
                    color = '#d62728'  # Red
                elif row['publications'] >= max_count * 0.4:
                    color = '#ff7f0e'  # Orange
                elif row['publications'] >= max_count * 0.2:
                    color = '#1f77b4'  # Blue
                else:
                    color = '#7fcdbb'  # Light blue

                ax.scatter(
                    row['lon'], row['lat'],
                    s=size,
                    c=color,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )

        except ImportError:
            logger.warning("Cartopy not available, creating basic plot")

            # Fallback to simple matplotlib
            max_count = df_plot['publications'].max()

            for _, row in df_plot.iterrows():
                size = 100 + (row['publications'] / max_count * 1000)

                # Color based on count
                if row['publications'] >= max_count * 0.7:
                    color = '#d62728'
                elif row['publications'] >= max_count * 0.4:
                    color = '#ff7f0e'
                elif row['publications'] >= max_count * 0.2:
                    color = '#1f77b4'
                else:
                    color = '#7fcdbb'

                ax.scatter(
                    row['lon'], row['lat'],
                    s=size,
                    c=color,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5
                )

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.grid(True, alpha=0.3)

        # Add title
        plt.title('Geographic Distribution of Scientific Publications',
                 fontsize=16, fontweight='bold', pad=20)

        # Create legend
        legend_elements = [
            mpatches.Patch(color='#d62728', label=f'High (≥{int(max_count*0.7)})'),
            mpatches.Patch(color='#ff7f0e', label=f'Medium-High ({int(max_count*0.4)}-{int(max_count*0.7)})'),
            mpatches.Patch(color='#1f77b4', label=f'Medium ({int(max_count*0.2)}-{int(max_count*0.4)})'),
            mpatches.Patch(color='#7fcdbb', label=f'Low (<{int(max_count*0.2)})')
        ]

        ax.legend(handles=legend_elements, loc='lower left',
                 title='Publications', fontsize=10, framealpha=0.9)

        # Add note
        fig.text(0.99, 0.01,
                f'Total countries: {len(df_plot)} | Total publications: {df_plot["publications"].sum()}',
                ha='right', va='bottom', fontsize=10, style='italic')

        plt.tight_layout()

        # Save figure
        output_path = Path(output_png)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.success(f"Static map saved to: {output_path}")

    def generate_geographic_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive geographic statistics.

        Returns:
            Dictionary with all geographic statistics including:
            - Top countries
            - Distribution by continent
            - Temporal evolution by region
        """
        logger.info("Generating geographic statistics")

        # Ensure data is extracted
        if not self._affiliation_cache:
            self.extract_author_affiliations()

        if not self._geocoding_cache:
            self.geocode_locations()

        # Calculate density
        df_density = self.calculate_publication_density()

        # Top 10 countries
        top_countries = df_density.head(10).to_dict('records')

        # Distribution by continent
        continent_dist = df_density.groupby('continent').agg({
            'publications': 'sum',
            'country': 'count'
        }).reset_index()
        continent_dist.columns = ['continent', 'publications', 'num_countries']
        continent_dist = continent_dist.sort_values('publications', ascending=False)

        # Calculate percentages
        total_pubs = df_density['publications'].sum()

        stats = {
            'summary': {
                'total_countries': len(df_density),
                'total_publications': int(total_pubs),
                'total_continents': len(continent_dist)
            },
            'top_10_countries': top_countries,
            'continent_distribution': continent_dist.to_dict('records'),
            'coverage': {
                'countries_with_data': len(self._geocoding_cache),
                'articles_with_location': len(self._affiliation_cache),
                'total_articles': len(self.df),
                'coverage_percentage': len(self._affiliation_cache) / len(self.df) * 100
            }
        }

        logger.success("Geographic statistics generated")

        return stats

    def save_statistics_report(self, output_path: str = 'geographic_stats.json'):
        """
        Save geographic statistics to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        stats = self.generate_geographic_statistics()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.success(f"Statistics report saved to: {output_file}")

        return stats


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    logger.info("Geographic Heatmap Module - Example Usage")

    # This would normally be called with real unified data
    # geo_map = GeographicHeatmap('data/processed/unified_data.csv')
    #
    # # Extract affiliations
    # affiliations = geo_map.extract_author_affiliations()
    #
    # # Geocode locations
    # geo_data = geo_map.geocode_locations()
    #
    # # Calculate density
    # density_df = geo_map.calculate_publication_density()
    # print(density_df.head(10))
    #
    # # Create interactive maps
    # geo_map.create_interactive_map(output_html='output/geographic_folium.html')
    # geo_map.create_plotly_map(output_html='output/geographic_plotly.html')
    #
    # # Create static map for PDF
    # geo_map.create_static_map(output_png='output/geographic_map.png', dpi=300)
    #
    # # Generate and save statistics
    # stats = geo_map.generate_geographic_statistics()
    # geo_map.save_statistics_report('output/geographic_stats.json')

    logger.info("Example complete!")
