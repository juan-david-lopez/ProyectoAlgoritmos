"""
Tests for Visualization Module

Tests all visualization components including:
- Geographic Heatmap
- Dynamic Word Cloud
- Timeline Visualization
- PDF Exporter
- Visualization Pipeline

Author: Bibliometric Analysis System
Date: 2024
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.geographic_heatmap import GeographicHeatmap
from src.visualization.dynamic_wordcloud import DynamicWordCloud
from src.visualization.timeline_visualization import TimelineVisualization
from src.visualization.pdf_exporter import PDFExporter
from src.visualization.visualization_pipeline import VisualizationPipeline


# Fixtures

@pytest.fixture(scope="module")
def sample_data_path(tmp_path_factory):
    """Create sample data for testing."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    data_path = tmp_dir / "test_data.csv"

    # Create sample data
    np.random.seed(42)
    data = []

    for i in range(50):
        data.append({
            'id': f'test_{i:04d}',
            'title': f'Research Study {i}',
            'authors': f'Author {i} (MIT, USA)',
            'year': 2020 + (i % 4),
            'abstract': 'This study explores deep learning, neural networks, and artificial intelligence.',
            'keywords': 'deep learning; neural networks; AI',
            'doi': f'10.1234/test.{i}',
            'source': 'Test Database',
            'publication_type': 'journal' if i % 2 == 0 else 'conference',
            'journal_conference': 'Test Journal' if i % 2 == 0 else 'Test Conference'
        })

    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False, encoding='utf-8')

    return str(data_path)


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    """Create temporary output directory."""
    return tmp_path_factory.mktemp("test_output")


# Geographic Heatmap Tests

class TestGeographicHeatmap:
    """Tests for GeographicHeatmap class."""

    def test_initialization(self, sample_data_path):
        """Test GeographicHeatmap initialization."""
        geo_map = GeographicHeatmap(sample_data_path)
        assert geo_map is not None
        assert geo_map.data_path == Path(sample_data_path)

    def test_extract_affiliations(self, sample_data_path):
        """Test affiliation extraction."""
        geo_map = GeographicHeatmap(sample_data_path)
        affiliations = geo_map.extract_author_affiliations()

        assert isinstance(affiliations, dict)
        assert len(affiliations) > 0

    def test_geocoding(self, sample_data_path):
        """Test geocoding of locations."""
        geo_map = GeographicHeatmap(sample_data_path)
        geo_map.extract_author_affiliations()
        geo_data = geo_map.geocode_locations()

        assert isinstance(geo_data, dict)
        assert 'USA' in geo_data
        assert 'lat' in geo_data['USA']
        assert 'lon' in geo_data['USA']

    def test_statistics_generation(self, sample_data_path):
        """Test geographic statistics generation."""
        geo_map = GeographicHeatmap(sample_data_path)
        geo_map.extract_author_affiliations()
        geo_map.geocode_locations()
        stats = geo_map.generate_geographic_statistics()

        assert isinstance(stats, dict)
        assert 'top_10_countries' in stats
        assert 'total_countries' in stats


# Word Cloud Tests

class TestDynamicWordCloud:
    """Tests for DynamicWordCloud class."""

    def test_initialization(self, sample_data_path):
        """Test DynamicWordCloud initialization."""
        wc = DynamicWordCloud(sample_data_path)
        assert wc is not None
        assert wc.data_path == Path(sample_data_path)

    def test_term_extraction(self, sample_data_path):
        """Test term extraction."""
        wc = DynamicWordCloud(sample_data_path)
        terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])

        assert isinstance(terms, dict)
        assert len(terms) > 0
        # Should contain common terms from our sample data
        assert any('learning' in term.lower() for term in terms.keys())

    def test_weight_calculation(self, sample_data_path):
        """Test weight calculation."""
        wc = DynamicWordCloud(sample_data_path)
        terms = wc.extract_and_process_terms(sources=['abstract'])
        weights = wc.calculate_term_weights(terms, method='tfidf')

        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(w, (int, float)) for w in weights.values())

    def test_different_weighting_methods(self, sample_data_path):
        """Test different weighting methods."""
        wc = DynamicWordCloud(sample_data_path)
        terms = wc.extract_and_process_terms(sources=['abstract'])

        methods = ['frequency', 'log_frequency', 'normalized', 'tfidf']
        for method in methods:
            weights = wc.calculate_term_weights(terms, method=method)
            assert isinstance(weights, dict)
            assert len(weights) > 0


# Timeline Tests

class TestTimelineVisualization:
    """Tests for TimelineVisualization class."""

    def test_initialization(self, sample_data_path):
        """Test TimelineVisualization initialization."""
        timeline = TimelineVisualization(sample_data_path)
        assert timeline is not None
        assert timeline.data_path == Path(sample_data_path)

    def test_temporal_data_extraction(self, sample_data_path):
        """Test temporal data extraction."""
        timeline = TimelineVisualization(sample_data_path)
        df = timeline.extract_temporal_data()

        assert isinstance(df, pd.DataFrame)
        assert 'year' in df.columns
        assert len(df) > 0

    def test_yearly_statistics(self, sample_data_path):
        """Test yearly statistics calculation."""
        timeline = TimelineVisualization(sample_data_path)
        df = timeline.extract_temporal_data()
        stats = timeline.calculate_yearly_statistics(df)

        assert isinstance(stats, dict)
        assert 'summary' in stats
        assert 'yearly_counts' in stats
        assert 'total_publications' in stats['summary']

    def test_projection(self, sample_data_path):
        """Test future projection."""
        timeline = TimelineVisualization(sample_data_path)
        df = timeline.extract_temporal_data()
        stats = timeline.calculate_yearly_statistics(df)

        if 'projection' in stats:
            proj = stats['projection']
            assert 'slope' in proj
            assert 'future_years' in proj
            assert 'projected_counts' in proj


# PDF Exporter Tests

class TestPDFExporter:
    """Tests for PDFExporter class."""

    def test_initialization(self, output_dir):
        """Test PDFExporter initialization."""
        pdf_path = output_dir / "test_report.pdf"
        exporter = PDFExporter(str(pdf_path))
        assert exporter is not None
        assert exporter.output_pdf_path == pdf_path

    def test_cover_page_creation(self, output_dir):
        """Test cover page creation."""
        pdf_path = output_dir / "test_cover.pdf"
        exporter = PDFExporter(str(pdf_path))

        # Should not raise an error
        exporter.create_cover_page(
            title="Test Report",
            subtitle="Test Subtitle",
            authors=["Test Author"]
        )
        assert len(exporter.story) > 0


# Visualization Pipeline Tests

class TestVisualizationPipeline:
    """Tests for VisualizationPipeline class."""

    def test_initialization(self, sample_data_path, output_dir):
        """Test VisualizationPipeline initialization."""
        pipeline = VisualizationPipeline(
            unified_data_path=sample_data_path,
            output_dir=str(output_dir / "pipeline_test")
        )
        assert pipeline is not None
        assert pipeline.data_path == Path(sample_data_path)
        assert len(pipeline.df) == 50

    def test_data_validation(self, sample_data_path, output_dir):
        """Test data validation."""
        pipeline = VisualizationPipeline(
            unified_data_path=sample_data_path,
            output_dir=str(output_dir / "pipeline_validation")
        )
        is_valid, issues = pipeline.validate_data()

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_output_structure_creation(self, sample_data_path, output_dir):
        """Test output directory structure creation."""
        test_output = output_dir / "pipeline_structure"
        pipeline = VisualizationPipeline(
            unified_data_path=sample_data_path,
            output_dir=str(test_output)
        )

        # Check that directories were created
        assert (test_output / 'geographic').exists()
        assert (test_output / 'wordclouds').exists()
        assert (test_output / 'timeline').exists()
        assert (test_output / 'reports').exists()
        assert (test_output / 'dashboard').exists()

    def test_execution_report_generation(self, sample_data_path, output_dir):
        """Test execution report generation."""
        pipeline = VisualizationPipeline(
            unified_data_path=sample_data_path,
            output_dir=str(output_dir / "pipeline_report")
        )

        # Set some dummy results
        pipeline.execution_start = pd.Timestamp.now()
        pipeline.execution_end = pd.Timestamp.now()
        pipeline.results['successful'] = ['geographic', 'wordcloud']
        pipeline.results['timings'] = {'validation': 1.2, 'geographic': 3.5}

        report_path = pipeline.generate_execution_report(
            str(output_dir / "test_execution_report.md")
        )

        assert Path(report_path).exists()
        assert Path(report_path).with_suffix('.json').exists()

    @pytest.mark.slow
    def test_full_pipeline_execution(self, sample_data_path, output_dir):
        """Test full pipeline execution (slow test)."""
        pipeline = VisualizationPipeline(
            unified_data_path=sample_data_path,
            output_dir=str(output_dir / "pipeline_full")
        )

        results = pipeline.run_all_visualizations()

        assert results['status'] == 'success'
        assert len(results['results']['successful']) > 0


# Integration Tests

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.slow
    def test_geographic_to_pdf_workflow(self, sample_data_path, output_dir):
        """Test workflow from geographic analysis to PDF."""
        # Generate geographic visualization
        geo_map = GeographicHeatmap(sample_data_path)
        geo_map.extract_author_affiliations()
        geo_data = geo_map.geocode_locations()
        stats = geo_map.generate_geographic_statistics()

        # Create static map
        map_path = output_dir / "test_map.png"
        try:
            geo_map.create_static_map(str(map_path), dpi=150)
            map_created = True
        except:
            map_created = False  # Cartopy might not be available

        # Generate PDF
        pdf_path = output_dir / "test_workflow.pdf"
        exporter = PDFExporter(str(pdf_path))

        if map_created:
            geographic_data = {
                'map_image': str(map_path),
                'statistics': stats
            }
        else:
            geographic_data = None

        # Should complete without errors
        result = exporter.generate_complete_pdf(
            title="Test Workflow",
            subtitle="Integration Test",
            geographic_data=geographic_data,
            overall_stats={'total_publications': 50}
        )

        assert Path(result).exists()


# Utility function for running tests
def run_tests():
    """Run all tests."""
    import subprocess

    print("Running visualization tests...")
    result = subprocess.run(
        ['pytest', __file__, '-v', '--tb=short'],
        capture_output=False
    )
    return result.returncode


if __name__ == "__main__":
    # Run tests if executed directly
    exit_code = run_tests()
    sys.exit(exit_code)
