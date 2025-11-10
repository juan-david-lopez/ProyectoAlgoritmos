"""
Complete Visualization System - Integrated Demo

This script demonstrates the complete bibliometric visualization system,
integrating all 4 parts:
1. Geographic Heatmap
2. Dynamic Word Cloud
3. Timeline Visualization
4. PDF Export

Generates a complete professional PDF report with all visualizations.

Author: Bibliometric Analysis System
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    GeographicHeatmap,
    DynamicWordCloud,
    TimelineVisualization,
    PDFExporter
)
from loguru import logger


def create_comprehensive_sample_data():
    """Create realistic sample data for all visualizations."""
    logger.info("Creating comprehensive sample data")

    np.random.seed(42)

    # Sample data for years 2018-2023
    data = []
    article_id = 1

    years = range(2018, 2024)
    venues = [
        'IEEE Transactions on AI',
        'ACM Computing Surveys',
        'Nature Machine Intelligence',
        'ICML Conference',
        'NeurIPS Conference'
    ]

    countries = ['USA', 'UK', 'China', 'Germany', 'Canada']
    institutions = ['MIT', 'Stanford', 'Oxford', 'Tsinghua', 'Toronto']

    for year in years:
        n_pubs = 15 + year - 2018 + np.random.randint(5, 15)  # Growing trend

        for _ in range(n_pubs):
            country_idx = np.random.choice(len(countries))

            data.append({
                'id': f'pub_{article_id:04d}',
                'title': f'Research on AI and Machine Learning - Study {article_id}',
                'authors': f'Author {article_id} ({institutions[country_idx]}, {countries[country_idx]})',
                'year': year,
                'abstract': 'This study explores deep learning, neural networks, artificial intelligence, '
                           'machine learning algorithms, natural language processing, computer vision, '
                           'reinforcement learning, and generative models for various applications.',
                'keywords': 'deep learning; neural networks; AI; machine learning; transformers',
                'doi': f'10.1234/example.{article_id}',
                'source': 'Research Database',
                'publication_type': np.random.choice(['journal', 'conference']),
                'journal_conference': np.random.choice(venues)
            })
            article_id += 1

    df = pd.DataFrame(data)

    # Save
    output_dir = Path('data/sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'complete_sample_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.success(f"Sample data created: {output_path} ({len(df)} records)")
    return output_path


def generate_complete_report():
    """Generate complete PDF report with all visualizations."""
    logger.info("="  * 70)
    logger.info("COMPLETE VISUALIZATION SYSTEM DEMO")
    logger.info("=" * 70)

    # Create sample data
    data_path = create_comprehensive_sample_data()

    # Create output directories
    output_dir = Path('output/complete_report')
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    # ===== PART 1: Geographic Analysis =====
    logger.info("\n[Part 1/4] Generating geographic visualizations...")

    try:
        geo_map = GeographicHeatmap(str(data_path))
        geo_map.extract_author_affiliations()
        geo_data = geo_map.geocode_locations()

        # Generate static map for PDF
        geo_map_path = viz_dir / 'geographic_map.png'
        geo_map.create_static_map(str(geo_map_path), dpi=300)

        # Get statistics
        geo_stats = geo_map.generate_geographic_statistics()

        logger.success(f"Geographic analysis complete: {len(geo_data)} countries")

    except Exception as e:
        logger.error(f"Geographic analysis failed: {e}")
        geo_map_path = None
        geo_stats = {}

    # ===== PART 2: Word Cloud Analysis =====
    logger.info("\n[Part 2/4] Generating word cloud visualizations...")

    try:
        wc = DynamicWordCloud(str(data_path))
        terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])
        weights = wc.calculate_term_weights(terms, method='tfidf')

        # Generate word cloud for PDF
        wc_path = viz_dir / 'wordcloud.png'
        wc.generate_wordcloud(weights, str(wc_path), style='scientific', dpi=300)

        # Prepare term statistics
        top_terms = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
        term_stats = {
            'top_terms': [(term, terms[term], weight) for term, weight in top_terms]
        }

        logger.success(f"Word cloud complete: {len(terms)} unique terms")

    except Exception as e:
        logger.error(f"Word cloud analysis failed: {e}")
        wc_path = None
        term_stats = {}

    # ===== PART 3: Timeline Analysis =====
    logger.info("\n[Part 3/4] Generating timeline visualizations...")

    try:
        timeline = TimelineVisualization(str(data_path))
        df = timeline.extract_temporal_data()
        temporal_stats = timeline.calculate_yearly_statistics(df)

        # Generate timeline plot for PDF
        timeline_path = viz_dir / 'timeline.png'
        timeline.create_timeline_plot(df, str(timeline_path), dpi=300)

        # Generate stacked area chart
        stacked_path = viz_dir / 'stacked_area.png'
        timeline.create_stacked_area_chart(df, str(stacked_path), dpi=300)

        logger.success(f"Timeline analysis complete: {temporal_stats['summary']['total_years']} years")

    except Exception as e:
        logger.error(f"Timeline analysis failed: {e}")
        timeline_path = None
        stacked_path = None
        temporal_stats = {}

    # ===== PART 4: Generate PDF Report =====
    logger.info("\n[Part 4/4] Generating PDF report...")

    try:
        pdf_path = output_dir / 'bibliometric_report.pdf'
        exporter = PDFExporter(str(pdf_path))

        # Prepare overall statistics
        overall_stats = {
            'total_publications': len(pd.read_csv(data_path)),
            'countries_analyzed': len(geo_data) if geo_data else 0,
            'unique_terms': len(terms) if terms else 0,
            'years_covered': temporal_stats.get('summary', {}).get('total_years', 0),
            'avg_publications_per_year': temporal_stats.get('summary', {}).get('avg_per_year', 0),
            'top_venue': pd.read_csv(data_path)['journal_conference'].value_counts().index[0]
        }

        # Processing info
        import platform
        processing_info = {
            'version': '1.0.0',
            'python_version': platform.python_version(),
            'sources': ['IEEE Xplore', 'ACM Digital Library', 'Scopus', 'Web of Science'],
            'references': 'Bibliometric Analysis System v1.0'
        }

        # Generate complete PDF
        pdf_output = exporter.generate_complete_pdf(
            title="Bibliometric Analysis Report",
            subtitle="Artificial Intelligence and Machine Learning Research",
            date_range=f"{temporal_stats.get('summary', {}).get('first_year', 'N/A')} - "
                      f"{temporal_stats.get('summary', {}).get('last_year', 'N/A')}",
            authors=["Research Analysis Team"],
            institution="Advanced Bibliometrics Institute",
            geographic_data={
                'map_image': str(geo_map_path) if geo_map_path else None,
                'statistics': geo_stats
            },
            wordcloud_data={
                'image': str(wc_path) if wc_path else None,
                'statistics': term_stats
            },
            timeline_data={
                'images': [str(timeline_path), str(stacked_path)] if timeline_path else [],
                'statistics': temporal_stats
            },
            overall_stats=overall_stats,
            processing_info=processing_info
        )

        logger.success(f"\n{'='*70}")
        logger.success("COMPLETE REPORT GENERATED SUCCESSFULLY!")
        logger.success(f"{'='*70}")
        logger.success(f"\nPDF Report: {pdf_output}")
        logger.success(f"Visualizations: {viz_dir}")

        logger.info("\nGenerated content:")
        logger.info("  ✓ Cover page with title and metadata")
        logger.info("  ✓ Table of contents")
        logger.info("  ✓ Executive summary")
        logger.info("  ✓ Geographic distribution analysis")
        logger.info("  ✓ Term frequency analysis (word cloud)")
        logger.info("  ✓ Temporal evolution analysis")
        logger.info("  ✓ Summary statistics")
        logger.info("  ✓ Metadata and methods")

        return pdf_output

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run complete demo."""
    try:
        pdf_path = generate_complete_report()

        if pdf_path:
            logger.info("\n" + "=" * 70)
            logger.info("SUCCESS! Complete bibliometric report generated")
            logger.info("=" * 70)
            logger.info(f"\nOpen the PDF to view the complete report: {pdf_path}")
        else:
            logger.error("Report generation failed. Check logs for details.")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
