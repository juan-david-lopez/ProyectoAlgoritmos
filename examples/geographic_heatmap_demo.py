"""
Geographic Heatmap Visualization - Example Usage

This script demonstrates how to use the GeographicHeatmap class to:
1. Extract author affiliations from publication data
2. Geocode locations to coordinates
3. Create interactive maps (Folium and Plotly)
4. Create static high-quality maps for PDF export
5. Generate comprehensive geographic statistics

Author: Bibliometric Analysis System
Date: 2024
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.geographic_heatmap import GeographicHeatmap
from loguru import logger
import pandas as pd
import json


def create_sample_data():
    """
    Create sample unified data for demonstration.

    In production, this data would come from the actual scraping
    and unification process.
    """
    logger.info("Creating sample data for demonstration")

    sample_data = [
        {
            'id': 'pub_001',
            'title': 'Deep Learning for Computer Vision',
            'authors': 'John Doe (MIT, USA); Jane Smith (Stanford, USA)',
            'year': 2023,
            'abstract': 'Study on deep learning applications...',
            'keywords': 'deep learning; computer vision; neural networks',
            'doi': '10.1234/example.001',
            'source': 'IEEE',
            'publication_type': 'article',
            'journal_conference': 'IEEE CVPR',
            'url': 'https://example.com/001'
        },
        {
            'id': 'pub_002',
            'title': 'Natural Language Processing Advances',
            'authors': 'Maria Garcia, University of Barcelona, Spain',
            'year': 2023,
            'abstract': 'Recent advances in NLP...',
            'keywords': 'NLP; transformers; BERT',
            'doi': '10.1234/example.002',
            'source': 'ACM',
            'publication_type': 'article',
            'journal_conference': 'ACL',
            'url': 'https://example.com/002'
        },
        {
            'id': 'pub_003',
            'title': 'Quantum Computing Applications',
            'authors': 'Wei Chen (Tsinghua University, China)',
            'year': 2023,
            'abstract': 'Quantum computing research...',
            'keywords': 'quantum computing; qubits',
            'doi': '10.1234/example.003',
            'source': 'Nature',
            'publication_type': 'article',
            'journal_conference': 'Nature',
            'url': 'https://example.com/003'
        },
        {
            'id': 'pub_004',
            'title': 'Machine Learning in Healthcare',
            'authors': 'Hans Mueller (ETH Zurich, Switzerland); Anna Schmidt (Germany)',
            'year': 2023,
            'abstract': 'ML applications in healthcare...',
            'keywords': 'machine learning; healthcare; diagnosis',
            'doi': '10.1234/example.004',
            'source': 'ScienceDirect',
            'publication_type': 'article',
            'journal_conference': 'Medical Informatics',
            'url': 'https://example.com/004'
        },
        {
            'id': 'pub_005',
            'title': 'Artificial Intelligence Ethics',
            'authors': 'Yuki Tanaka, University of Tokyo, Japan',
            'year': 2023,
            'abstract': 'Ethics in AI systems...',
            'keywords': 'AI; ethics; fairness',
            'doi': '10.1234/example.005',
            'source': 'IEEE',
            'publication_type': 'article',
            'journal_conference': 'IEEE AI Ethics',
            'url': 'https://example.com/005'
        },
        {
            'id': 'pub_006',
            'title': 'Robotics and Automation',
            'authors': 'Pierre Dubois (CNRS, France)',
            'year': 2023,
            'abstract': 'Advances in robotics...',
            'keywords': 'robotics; automation; control',
            'doi': '10.1234/example.006',
            'source': 'Scopus',
            'publication_type': 'article',
            'journal_conference': 'Robotics Journal',
            'url': 'https://example.com/006'
        },
        {
            'id': 'pub_007',
            'title': 'Blockchain Technology',
            'authors': 'Sarah Johnson (Cambridge, UK)',
            'year': 2023,
            'abstract': 'Blockchain applications...',
            'keywords': 'blockchain; cryptocurrency; security',
            'doi': '10.1234/example.007',
            'source': 'ACM',
            'publication_type': 'article',
            'journal_conference': 'ACM Blockchain',
            'url': 'https://example.com/007'
        },
        {
            'id': 'pub_008',
            'title': 'Data Science Best Practices',
            'authors': 'Raj Patel (IIT Bombay, India); Priya Sharma (India)',
            'year': 2023,
            'abstract': 'Data science methodologies...',
            'keywords': 'data science; analytics; visualization',
            'doi': '10.1234/example.008',
            'source': 'IEEE',
            'publication_type': 'article',
            'journal_conference': 'Data Science Review',
            'url': 'https://example.com/008'
        },
        {
            'id': 'pub_009',
            'title': 'Cloud Computing Security',
            'authors': 'Michael Brown (Microsoft Research, USA)',
            'year': 2023,
            'abstract': 'Security in cloud systems...',
            'keywords': 'cloud computing; security; encryption',
            'doi': '10.1234/example.009',
            'source': 'IEEE',
            'publication_type': 'article',
            'journal_conference': 'Cloud Security',
            'url': 'https://example.com/009'
        },
        {
            'id': 'pub_010',
            'title': 'Internet of Things Development',
            'authors': 'Kim Min-Jun (KAIST, South Korea)',
            'year': 2023,
            'abstract': 'IoT development practices...',
            'keywords': 'IoT; sensors; connectivity',
            'doi': '10.1234/example.010',
            'source': 'ScienceDirect',
            'publication_type': 'article',
            'journal_conference': 'IoT Journal',
            'url': 'https://example.com/010'
        },
        {
            'id': 'pub_011',
            'title': 'Augmented Reality Applications',
            'authors': 'Lucas Silva (University of Sao Paulo, Brazil)',
            'year': 2023,
            'abstract': 'AR applications study...',
            'keywords': 'AR; virtual reality; visualization',
            'doi': '10.1234/example.011',
            'source': 'ACM',
            'publication_type': 'article',
            'journal_conference': 'AR/VR Conference',
            'url': 'https://example.com/011'
        },
        {
            'id': 'pub_012',
            'title': 'Cybersecurity Frameworks',
            'authors': 'Emma Wilson (Oxford University, UK)',
            'year': 2023,
            'abstract': 'Cybersecurity best practices...',
            'keywords': 'cybersecurity; frameworks; protection',
            'doi': '10.1234/example.012',
            'source': 'IEEE',
            'publication_type': 'article',
            'journal_conference': 'IEEE Security',
            'url': 'https://example.com/012'
        },
    ]

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # Save to CSV
    output_dir = Path('data/sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'sample_unified_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.success(f"Sample data created: {output_path}")
    return output_path


def example_basic_usage():
    """Example 1: Basic usage - Extract and analyze geographic data."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Basic Geographic Analysis")
    logger.info("=" * 70)

    # Create sample data
    data_path = create_sample_data()

    # Initialize GeographicHeatmap
    geo_map = GeographicHeatmap(str(data_path))

    # Extract affiliations
    logger.info("Extracting author affiliations...")
    affiliations = geo_map.extract_author_affiliations()

    logger.info(f"Found {len(affiliations)} articles with affiliations")
    logger.info("\nSample affiliations:")
    for idx, (article_id, affiliation) in enumerate(list(affiliations.items())[:3]):
        logger.info(f"  {article_id}:")
        logger.info(f"    Author: {affiliation.get('first_author', 'N/A')}")
        logger.info(f"    Institution: {affiliation.get('institution', 'N/A')}")
        logger.info(f"    Country: {affiliation.get('country', 'N/A')}")

    # Geocode locations
    logger.info("\nGeocoding locations...")
    geo_data = geo_map.geocode_locations()

    logger.info(f"Geocoded {len(geo_data)} countries")
    logger.info("\nCountries with publications:")
    for country, data in list(geo_data.items())[:5]:
        logger.info(f"  {country}: {data['count']} publications")


def example_publication_density():
    """Example 2: Calculate and display publication density."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Publication Density Analysis")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_unified_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    geo_map = GeographicHeatmap(str(data_path))
    geo_map.extract_author_affiliations()
    geo_map.geocode_locations()

    # Calculate density
    logger.info("Calculating publication density...")
    density_df = geo_map.calculate_publication_density()

    logger.info("\nTop 10 Countries by Publications:")
    logger.info("-" * 70)
    logger.info(f"{'Rank':<6} {'Country':<20} {'Publications':<15} {'Percentage':<12} {'Continent'}")
    logger.info("-" * 70)

    for idx, row in density_df.head(10).iterrows():
        logger.info(
            f"{idx+1:<6} "
            f"{row['country']:<20} "
            f"{row['publications']:<15} "
            f"{row['percentage']:.2f}%{'':<8} "
            f"{row['continent']}"
        )


def example_interactive_maps():
    """Example 3: Create interactive maps."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Interactive Map Generation")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_unified_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    geo_map = GeographicHeatmap(str(data_path))
    geo_map.extract_author_affiliations()
    geo_map.geocode_locations()

    # Create output directory
    output_dir = Path('output/geographic_maps')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Folium interactive map
    logger.info("Creating Folium interactive map...")
    folium_path = output_dir / 'geographic_folium.html'
    geo_map.create_interactive_map(output_html=str(folium_path))
    logger.info(f"Folium map saved to: {folium_path}")

    # Create Plotly interactive map
    logger.info("Creating Plotly interactive map...")
    plotly_path = output_dir / 'geographic_plotly.html'
    geo_map.create_plotly_map(output_html=str(plotly_path))
    logger.info(f"Plotly map saved to: {plotly_path}")

    logger.info("\nOpen the HTML files in a web browser to view interactive maps!")


def example_static_map():
    """Example 4: Create static high-quality map for PDF export."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Static Map for PDF Export")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_unified_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    geo_map = GeographicHeatmap(str(data_path))
    geo_map.extract_author_affiliations()
    geo_map.geocode_locations()

    # Create output directory
    output_dir = Path('output/geographic_maps')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create static map
    logger.info("Creating static map (300 DPI for print quality)...")
    static_path = output_dir / 'geographic_map.png'
    geo_map.create_static_map(output_png=str(static_path), dpi=300)
    logger.info(f"Static map saved to: {static_path}")

    logger.info("\nThis high-resolution image can be embedded in PDF reports!")


def example_statistics():
    """Example 5: Generate comprehensive geographic statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Geographic Statistics")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_unified_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    geo_map = GeographicHeatmap(str(data_path))
    geo_map.extract_author_affiliations()
    geo_map.geocode_locations()

    # Generate statistics
    logger.info("Generating geographic statistics...")
    stats = geo_map.generate_geographic_statistics()

    # Display summary
    logger.info("\nSUMMARY:")
    logger.info("-" * 70)
    logger.info(f"Total countries: {stats['summary']['total_countries']}")
    logger.info(f"Total publications: {stats['summary']['total_publications']}")
    logger.info(f"Total continents: {stats['summary']['total_continents']}")

    # Display top countries
    logger.info("\nTOP 5 COUNTRIES:")
    logger.info("-" * 70)
    for country in stats['top_10_countries'][:5]:
        logger.info(f"  {country['country']}: {country['publications']} publications ({country['percentage']:.2f}%)")

    # Display continent distribution
    logger.info("\nCONTINENT DISTRIBUTION:")
    logger.info("-" * 70)
    for continent in stats['continent_distribution']:
        logger.info(f"  {continent['continent']}: {continent['publications']} publications from {continent['num_countries']} countries")

    # Display coverage
    logger.info("\nCOVERAGE:")
    logger.info("-" * 70)
    logger.info(f"Articles with location data: {stats['coverage']['articles_with_location']} / {stats['coverage']['total_articles']}")
    logger.info(f"Coverage: {stats['coverage']['coverage_percentage']:.2f}%")

    # Save statistics to JSON
    output_dir = Path('output/geographic_maps')
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / 'geographic_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.success(f"\nStatistics saved to: {stats_path}")


def example_complete_workflow():
    """Example 6: Complete workflow - All visualizations and statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 6: Complete Geographic Visualization Workflow")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_unified_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    # Initialize
    logger.info("Initializing GeographicHeatmap...")
    geo_map = GeographicHeatmap(str(data_path))

    # Step 1: Extract affiliations
    logger.info("\n[Step 1/6] Extracting author affiliations...")
    affiliations = geo_map.extract_author_affiliations()
    logger.info(f"  Extracted {len(affiliations)} affiliations")

    # Step 2: Geocode locations
    logger.info("\n[Step 2/6] Geocoding locations...")
    geo_data = geo_map.geocode_locations()
    logger.info(f"  Geocoded {len(geo_data)} countries")

    # Step 3: Calculate density
    logger.info("\n[Step 3/6] Calculating publication density...")
    density_df = geo_map.calculate_publication_density()
    logger.info(f"  Calculated density for {len(density_df)} countries")

    # Create output directory
    output_dir = Path('output/geographic_complete')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Create interactive maps
    logger.info("\n[Step 4/6] Creating interactive maps...")
    geo_map.create_interactive_map(output_html=str(output_dir / 'folium_map.html'))
    geo_map.create_plotly_map(output_html=str(output_dir / 'plotly_map.html'))
    logger.info("  Interactive maps created")

    # Step 5: Create static map
    logger.info("\n[Step 5/6] Creating static map...")
    geo_map.create_static_map(output_png=str(output_dir / 'static_map.png'), dpi=300)
    logger.info("  Static map created")

    # Step 6: Generate statistics
    logger.info("\n[Step 6/6] Generating statistics...")
    geo_map.save_statistics_report(str(output_dir / 'statistics.json'))
    logger.info("  Statistics generated")

    logger.success(f"\n{' COMPLETE ':=^70}")
    logger.success(f"All outputs saved to: {output_dir}")
    logger.success("=" * 70)


def main():
    """Run all examples."""
    logger.info("Geographic Heatmap Visualization - Comprehensive Demo")
    logger.info("=" * 70)

    try:
        # Run examples
        example_basic_usage()
        example_publication_density()
        example_interactive_maps()
        example_static_map()
        example_statistics()
        example_complete_workflow()

        logger.success("\n" + "=" * 70)
        logger.success("All examples completed successfully!")
        logger.success("=" * 70)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
