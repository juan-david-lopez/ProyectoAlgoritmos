"""
Tests for Geographic Heatmap Visualization Module

This test suite validates the GeographicHeatmap class functionality.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.geographic_heatmap import GeographicHeatmap


class TestGeographicHeatmap:
    """Test suite for GeographicHeatmap class."""

    @pytest.fixture
    def sample_data_file(self):
        """Create a temporary CSV file with sample data."""
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        # Sample data
        sample_data = pd.DataFrame([
            {
                'id': 'pub_001',
                'title': 'AI Research',
                'authors': 'John Doe (MIT, USA)',
                'year': 2023,
                'abstract': 'Study on AI...',
                'keywords': 'AI;ML',
                'doi': '10.1234/001',
                'source': 'IEEE'
            },
            {
                'id': 'pub_002',
                'title': 'ML Study',
                'authors': 'Jane Smith, Stanford University, USA',
                'year': 2023,
                'abstract': 'ML research...',
                'keywords': 'ML;DL',
                'doi': '10.1234/002',
                'source': 'ACM'
            },
            {
                'id': 'pub_003',
                'title': 'NLP Paper',
                'authors': 'Maria Garcia (Barcelona, Spain)',
                'year': 2023,
                'abstract': 'NLP study...',
                'keywords': 'NLP',
                'doi': '10.1234/003',
                'source': 'ScienceDirect'
            }
        ])

        # Save to CSV
        csv_path = temp_dir / 'test_data.csv'
        sample_data.to_csv(csv_path, index=False, encoding='utf-8')

        yield csv_path

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_initialization(self, sample_data_file):
        """Test GeographicHeatmap initialization."""
        geo_map = GeographicHeatmap(str(sample_data_file))

        assert geo_map is not None
        assert len(geo_map.df) == 3
        assert geo_map.unified_data_path.exists()

    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            GeographicHeatmap('non_existent_file.csv')

    def test_extract_affiliations(self, sample_data_file):
        """Test affiliation extraction."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        affiliations = geo_map.extract_author_affiliations()

        assert isinstance(affiliations, dict)
        assert len(affiliations) > 0

        # Check structure
        for article_id, affiliation in affiliations.items():
            assert 'first_author' in affiliation
            assert 'institution' in affiliation
            assert 'country' in affiliation

    def test_pattern_extraction_parentheses(self, sample_data_file):
        """Test pattern extraction with parentheses format."""
        geo_map = GeographicHeatmap(str(sample_data_file))

        # Test pattern: "Author (Institution, Country)"
        result = geo_map._extract_affiliation_patterns('John Doe (MIT, USA)')

        assert result['first_author'] == 'John Doe'
        assert result['country'] == 'United States'

    def test_pattern_extraction_comma_separated(self, sample_data_file):
        """Test pattern extraction with comma-separated format."""
        geo_map = GeographicHeatmap(str(sample_data_file))

        # Test pattern: "Author, Institution, Country"
        result = geo_map._extract_affiliation_patterns(
            'Jane Smith, Stanford University, USA'
        )

        assert result['first_author'] == 'Jane Smith'
        # Note: Country detection depends on pattern matching

    def test_country_recognition(self, sample_data_file):
        """Test country name recognition."""
        geo_map = GeographicHeatmap(str(sample_data_file))

        # Test various country formats
        assert geo_map._is_country('USA')
        assert geo_map._is_country('United States')
        assert geo_map._is_country('UK')
        assert geo_map._is_country('China')
        assert not geo_map._is_country('Random Text')

    def test_country_normalization(self, sample_data_file):
        """Test country name normalization."""
        geo_map = GeographicHeatmap(str(sample_data_file))

        # Test normalization
        assert geo_map._normalize_country_name('USA') == 'United States'
        assert geo_map._normalize_country_name('UK') == 'United Kingdom'

    def test_geocode_locations(self, sample_data_file):
        """Test geocoding functionality."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()

        geo_data = geo_map.geocode_locations()

        assert isinstance(geo_data, dict)

        # Check structure
        for country, data in geo_data.items():
            assert 'lat' in data
            assert 'lon' in data
            assert 'count' in data
            assert 'continent' in data
            assert isinstance(data['lat'], float)
            assert isinstance(data['lon'], float)
            assert isinstance(data['count'], int)

    def test_calculate_density(self, sample_data_file):
        """Test publication density calculation."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()

        density_df = geo_map.calculate_publication_density()

        assert isinstance(density_df, pd.DataFrame)
        assert not density_df.empty

        # Check columns
        expected_columns = ['country', 'publications', 'percentage', 'lat', 'lon', 'continent']
        for col in expected_columns:
            assert col in density_df.columns

        # Check percentages sum to ~100%
        total_percentage = density_df['percentage'].sum()
        assert 99.9 <= total_percentage <= 100.1

    def test_generate_statistics(self, sample_data_file):
        """Test statistics generation."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()

        stats = geo_map.generate_geographic_statistics()

        assert isinstance(stats, dict)

        # Check summary
        assert 'summary' in stats
        assert 'total_countries' in stats['summary']
        assert 'total_publications' in stats['summary']

        # Check top countries
        assert 'top_10_countries' in stats
        assert isinstance(stats['top_10_countries'], list)

        # Check continent distribution
        assert 'continent_distribution' in stats

        # Check coverage
        assert 'coverage' in stats
        assert 'coverage_percentage' in stats['coverage']

    def test_create_interactive_map(self, sample_data_file):
        """Test Folium map creation."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()

        # Create temporary output file
        temp_dir = Path(tempfile.mkdtemp())
        output_html = temp_dir / 'test_map.html'

        try:
            geo_map.create_interactive_map(output_html=str(output_html))

            # Check file was created
            assert output_html.exists()
            assert output_html.stat().st_size > 0

            # Check it's valid HTML
            with open(output_html, 'r', encoding='utf-8') as f:
                content = f.read()
                assert '<html>' in content or '<!DOCTYPE html>' in content

        except ImportError:
            pytest.skip("Folium not installed")
        finally:
            shutil.rmtree(temp_dir)

    def test_create_plotly_map(self, sample_data_file):
        """Test Plotly map creation."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()

        # Create temporary output file
        temp_dir = Path(tempfile.mkdtemp())
        output_html = temp_dir / 'test_plotly.html'

        try:
            geo_map.create_plotly_map(output_html=str(output_html))

            # Check file was created
            assert output_html.exists()
            assert output_html.stat().st_size > 0

        except ImportError:
            pytest.skip("Plotly not installed")
        finally:
            shutil.rmtree(temp_dir)

    def test_create_static_map(self, sample_data_file):
        """Test static map creation."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()

        # Create temporary output file
        temp_dir = Path(tempfile.mkdtemp())
        output_png = temp_dir / 'test_map.png'

        try:
            geo_map.create_static_map(output_png=str(output_png), dpi=100)

            # Check file was created
            assert output_png.exists()
            assert output_png.stat().st_size > 0

        finally:
            shutil.rmtree(temp_dir)

    def test_save_statistics_report(self, sample_data_file):
        """Test statistics report saving."""
        geo_map = GeographicHeatmap(str(sample_data_file))
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()

        # Create temporary output file
        temp_dir = Path(tempfile.mkdtemp())
        output_json = temp_dir / 'stats.json'

        try:
            geo_map.save_statistics_report(str(output_json))

            # Check file was created
            assert output_json.exists()
            assert output_json.stat().st_size > 0

            # Check it's valid JSON
            import json
            with open(output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert 'summary' in data

        finally:
            shutil.rmtree(temp_dir)

    def test_institution_mapping(self, sample_data_file):
        """Test institution to country mapping."""
        geo_map = GeographicHeatmap(str(sample_data_file))

        # Test known institutions
        assert geo_map._map_institution_to_country('MIT') == 'United States'
        assert geo_map._map_institution_to_country('Cambridge') == 'United Kingdom'
        assert geo_map._map_institution_to_country('ETH Zurich') == 'Switzerland'

        # Test unknown institution
        assert geo_map._map_institution_to_country('Unknown University') is None

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Create empty CSV
        temp_dir = Path(tempfile.mkdtemp())
        empty_csv = temp_dir / 'empty.csv'

        pd.DataFrame(columns=['id', 'title', 'authors', 'year']).to_csv(
            empty_csv, index=False
        )

        try:
            geo_map = GeographicHeatmap(str(empty_csv))
            affiliations = geo_map.extract_author_affiliations()

            assert isinstance(affiliations, dict)
            assert len(affiliations) == 0

        finally:
            shutil.rmtree(temp_dir)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
