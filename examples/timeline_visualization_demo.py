"""
Timeline Visualization - Example Usage

This script demonstrates how to use the TimelineVisualization class to:
1. Extract and validate temporal data
2. Calculate yearly statistics and growth rates
3. Create professional timeline plots
4. Generate stacked area charts
5. Analyze venue-based timelines
6. Create interactive Plotly timelines
7. Detect and visualize publication bursts
8. Generate comprehensive statistical reports

Author: Bibliometric Analysis System
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.timeline_visualization import TimelineVisualization
from loguru import logger


def create_sample_data():
    """
    Create sample temporal data for demonstration.
    """
    logger.info("Creating sample temporal data")

    np.random.seed(42)

    # Generate realistic publication data from 2018-2023
    years = range(2018, 2024)
    venues = [
        'IEEE Transactions on AI',
        'ACM Computing Surveys',
        'Nature Machine Intelligence',
        'ICML Conference',
        'NeurIPS Conference',
        'CVPR Conference',
        'ICLR Conference',
        'AAAI Conference'
    ]

    publication_types = ['journal', 'journal', 'journal', 'conference',
                        'conference', 'conference', 'conference', 'conference']

    data = []
    article_id = 1

    # Base publications with growth trend
    base_pubs_per_year = {
        2018: 20,
        2019: 25,
        2020: 35,  # Burst year
        2021: 30,
        2022: 40,
        2023: 50   # Burst year
    }

    for year in years:
        n_pubs = base_pubs_per_year[year]

        for _ in range(n_pubs):
            venue_idx = np.random.choice(len(venues))
            venue = venues[venue_idx]
            pub_type = publication_types[venue_idx]

            data.append({
                'id': f'pub_{article_id:04d}',
                'title': f'Research Study {article_id} on AI and Machine Learning',
                'authors': f'Author {article_id}',
                'year': year,
                'publication_type': pub_type,
                'journal_conference': venue,
                'abstract': f'This study explores advanced topics in artificial intelligence...',
                'keywords': 'AI; machine learning; deep learning',
                'doi': f'10.1234/example.{article_id}'
            })
            article_id += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_dir = Path('data/sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'sample_timeline_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.success(f"Sample temporal data created: {output_path}")
    logger.info(f"  Years: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"  Total publications: {len(df)}")
    logger.info(f"  Venues: {df['journal_conference'].nunique()}")

    return output_path


def example_basic_usage():
    """Example 1: Basic temporal data extraction and statistics."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Basic Temporal Analysis")
    logger.info("=" * 70)

    # Create sample data
    data_path = create_sample_data()

    # Initialize TimelineVisualization
    timeline = TimelineVisualization(str(data_path))

    # Extract temporal data
    logger.info("Extracting temporal data...")
    df = timeline.extract_temporal_data()

    logger.info(f"Temporal data extracted:")
    logger.info(f"  Records: {len(df)}")
    logger.info(f"  Years: {df['year'].min()} to {df['year'].max()}")
    logger.info(f"  Time span: {df['year'].nunique()} years")

    # Calculate yearly statistics
    logger.info("\nCalculating yearly statistics...")
    stats = timeline.calculate_yearly_statistics(df)

    # Display summary
    summary = stats['summary']
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total publications: {summary['total_publications']}")
    logger.info(f"  Average per year: {summary['avg_per_year']:.2f}")
    logger.info(f"  Most productive year: {summary['most_productive_year']} ({summary['most_productive_year_count']} pubs)")
    logger.info(f"  Average growth rate: {summary['avg_growth_rate']:.2f}% per year")

    # Display yearly breakdown
    logger.info("\nYearly Breakdown:")
    for record in stats['yearly_counts'][:5]:  # Show first 5 years
        year = record['year']
        count = record['count']
        growth = record.get('growth_rate', 0)
        logger.info(f"  {year}: {count} publications (growth: {growth:+.1f}%)")


def example_timeline_plot():
    """Example 2: Create professional timeline plot."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Timeline Plot")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    timeline = TimelineVisualization(str(data_path))
    df = timeline.extract_temporal_data()

    output_dir = Path('output/timeline_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating timeline plot...")
    timeline.create_timeline_plot(
        df,
        output_path=str(output_dir / 'timeline.png'),
        dpi=300
    )

    logger.info(f"Timeline plot saved to: {output_dir / 'timeline.png'}")


def example_stacked_area_chart():
    """Example 3: Create stacked area chart."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Stacked Area Chart")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    timeline = TimelineVisualization(str(data_path))
    df = timeline.extract_temporal_data()

    output_dir = Path('output/timeline_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    # By publication type
    logger.info("Creating stacked area chart (by publication type)...")
    timeline.create_stacked_area_chart(
        df,
        output_path=str(output_dir / 'stacked_area_type.png'),
        group_by='publication_type',
        dpi=300
    )

    # By venue (top 5)
    logger.info("Creating stacked area chart (by top 5 venues)...")
    timeline.create_stacked_area_chart(
        df,
        output_path=str(output_dir / 'stacked_area_venue.png'),
        group_by='journal_conference',
        dpi=300
    )

    logger.info(f"Stacked area charts saved to: {output_dir}")


def example_venue_timeline():
    """Example 4: Create venue-based timeline visualizations."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Venue Timeline Visualizations")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    timeline = TimelineVisualization(str(data_path))
    df = timeline.extract_temporal_data()

    output_dir = Path('output/timeline_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap
    logger.info("Creating venue timeline (heatmap)...")
    timeline.create_venue_timeline(
        df,
        output_path=str(output_dir / 'venue_heatmap.png'),
        top_n_venues=8,
        visualization_type='heatmap',
        dpi=300
    )

    # Multiple lines
    logger.info("Creating venue timeline (multiple lines)...")
    timeline.create_venue_timeline(
        df,
        output_path=str(output_dir / 'venue_lines.png'),
        top_n_venues=8,
        visualization_type='lines',
        dpi=300
    )

    # Small multiples
    logger.info("Creating venue timeline (small multiples)...")
    timeline.create_venue_timeline(
        df,
        output_path=str(output_dir / 'venue_small_multiples.png'),
        top_n_venues=8,
        visualization_type='small_multiples',
        dpi=300
    )

    logger.info(f"Venue timelines saved to: {output_dir}")


def example_interactive_timeline():
    """Example 5: Create interactive Plotly timeline."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Interactive Timeline")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    timeline = TimelineVisualization(str(data_path))
    df = timeline.extract_temporal_data()

    output_dir = Path('output/timeline_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating interactive timeline...")
    timeline.create_interactive_timeline(
        df,
        output_html=str(output_dir / 'timeline_interactive.html')
    )

    logger.info(f"Interactive timeline saved to: {output_dir / 'timeline_interactive.html'}")
    logger.info("Open in browser to interact with the visualization!")


def example_burst_analysis():
    """Example 6: Detect and visualize publication bursts."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 6: Publication Burst Analysis")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    timeline = TimelineVisualization(str(data_path))
    df = timeline.extract_temporal_data()

    output_dir = Path('output/timeline_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Analyzing publication bursts...")
    burst_years = timeline.create_publication_burst_analysis(
        df,
        output_path=str(output_dir / 'burst_analysis.png'),
        threshold_std=1.0,  # Lower threshold for demo data
        dpi=300
    )

    logger.info(f"Burst analysis saved to: {output_dir / 'burst_analysis.png'}")

    if burst_years:
        logger.info(f"\nDetected {len(burst_years)} burst year(s):")
        for year in burst_years:
            year_df = df[df['year'] == year]
            logger.info(f"  {year}: {len(year_df)} publications")
    else:
        logger.info("No publication bursts detected")


def example_statistics_report():
    """Example 7: Generate comprehensive statistical report."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 7: Temporal Statistics Report")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    timeline = TimelineVisualization(str(data_path))
    df = timeline.extract_temporal_data()

    output_dir = Path('output/timeline_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating temporal statistics report...")
    report_path = timeline.generate_temporal_statistics_report(
        df,
        output_path=str(output_dir / 'temporal_report.md')
    )

    logger.info(f"Statistical report saved to: {report_path}")

    # Display a preview
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:20]  # First 20 lines

    logger.info("\nReport Preview:")
    for line in lines:
        logger.info(line.rstrip())


def example_complete_workflow():
    """Example 8: Complete temporal analysis workflow."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 8: Complete Temporal Analysis Workflow")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_timeline_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    # Initialize
    logger.info("Initializing TimelineVisualization...")
    timeline = TimelineVisualization(str(data_path))

    # Create output directory
    output_dir = Path('output/timeline_complete')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract data
    logger.info("\n[Step 1/7] Extracting temporal data...")
    df = timeline.extract_temporal_data()
    logger.info(f"  Extracted {len(df)} records from {df['year'].min()} to {df['year'].max()}")

    # Step 2: Calculate statistics
    logger.info("\n[Step 2/7] Calculating yearly statistics...")
    stats = timeline.calculate_yearly_statistics(df)
    logger.info(f"  Calculated statistics for {stats['summary']['total_years']} years")

    # Step 3: Create timeline plot
    logger.info("\n[Step 3/7] Creating timeline plot...")
    timeline.create_timeline_plot(
        df,
        output_path=str(output_dir / 'timeline.png'),
        dpi=300
    )

    # Step 4: Create stacked area charts
    logger.info("\n[Step 4/7] Creating stacked area charts...")
    timeline.create_stacked_area_chart(
        df,
        output_path=str(output_dir / 'stacked_area.png'),
        group_by='publication_type',
        dpi=300
    )

    # Step 5: Create venue timelines
    logger.info("\n[Step 5/7] Creating venue timeline visualizations...")
    timeline.create_venue_timeline(
        df,
        output_path=str(output_dir / 'venue_heatmap.png'),
        top_n_venues=8,
        visualization_type='heatmap',
        dpi=300
    )

    # Step 6: Create interactive timeline
    logger.info("\n[Step 6/7] Creating interactive timeline...")
    timeline.create_interactive_timeline(
        df,
        output_html=str(output_dir / 'timeline_interactive.html')
    )

    # Step 7: Generate report
    logger.info("\n[Step 7/7] Generating statistical report...")
    timeline.generate_temporal_statistics_report(
        df,
        output_path=str(output_dir / 'temporal_report.md')
    )

    # Step 8: Burst analysis
    logger.info("\n[Step 8/7] Analyzing publication bursts...")
    timeline.create_publication_burst_analysis(
        df,
        output_path=str(output_dir / 'burst_analysis.png'),
        threshold_std=1.0
    )

    logger.success(f"\n{' COMPLETE ':=^70}")
    logger.success(f"All outputs saved to: {output_dir}")
    logger.success("=" * 70)

    # Summary
    logger.info("\nGenerated files:")
    logger.info(f"  - timeline.png: Main timeline plot")
    logger.info(f"  - stacked_area.png: Composition over time")
    logger.info(f"  - venue_heatmap.png: Venue-based heatmap")
    logger.info(f"  - timeline_interactive.html: Interactive Plotly timeline")
    logger.info(f"  - burst_analysis.png: Publication burst detection")
    logger.info(f"  - temporal_report.md: Statistical report")


def main():
    """Run all examples."""
    logger.info("Timeline Visualization - Comprehensive Demo")
    logger.info("=" * 70)

    try:
        # Run examples
        example_basic_usage()
        example_timeline_plot()
        example_stacked_area_chart()
        example_venue_timeline()
        example_interactive_timeline()
        example_burst_analysis()
        example_statistics_report()
        example_complete_workflow()

        logger.success("\n" + "=" * 70)
        logger.success("All examples completed successfully!")
        logger.success("=" * 70)

        logger.info("\nGenerated outputs:")
        logger.info("  - output/timeline_demo/        : Individual visualizations")
        logger.info("  - output/timeline_complete/    : Complete workflow outputs")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
