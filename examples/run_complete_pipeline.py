"""
Complete Visualization Pipeline - Demonstration

This example demonstrates the complete visualization pipeline that:
1. Validates data
2. Generates all visualizations (geographic, word cloud, timeline)
3. Creates professional PDF report
4. Prepares dashboard data
5. Generates execution reports

Author: Bibliometric Analysis System
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.visualization_pipeline import VisualizationPipeline
from loguru import logger


def create_sample_data():
    """Create sample bibliometric data for demonstration."""
    logger.info("Creating sample bibliometric data...")

    np.random.seed(42)

    # Sample data for 2018-2023
    data = []
    article_id = 1

    years = range(2018, 2024)
    venues = [
        'IEEE Transactions on Pattern Analysis and Machine Intelligence',
        'Nature Machine Intelligence',
        'ACM Computing Surveys',
        'Neural Information Processing Systems (NeurIPS)',
        'International Conference on Machine Learning (ICML)',
        'Journal of Machine Learning Research',
        'Artificial Intelligence Review',
        'IEEE Transactions on Neural Networks and Learning Systems'
    ]

    countries = ['USA', 'UK', 'China', 'Germany', 'Canada', 'France', 'Australia', 'Japan', 'Singapore', 'Switzerland']
    institutions = [
        'MIT', 'Stanford', 'Oxford', 'Tsinghua', 'Toronto',
        'ETH Zurich', 'Cambridge', 'Berkeley', 'NUS', 'CMU'
    ]

    research_topics = [
        'deep learning', 'neural networks', 'artificial intelligence',
        'machine learning', 'natural language processing', 'computer vision',
        'reinforcement learning', 'generative models', 'transformers',
        'convolutional neural networks', 'recurrent neural networks',
        'attention mechanisms', 'transfer learning', 'few-shot learning',
        'meta-learning', 'self-supervised learning', 'graph neural networks'
    ]

    for year in years:
        # More publications in recent years
        n_pubs = 20 + (year - 2018) * 5 + np.random.randint(5, 15)

        for _ in range(n_pubs):
            country_idx = np.random.choice(len(countries), p=[0.25, 0.10, 0.20, 0.08, 0.07, 0.05, 0.05, 0.08, 0.06, 0.06])

            # Select random topics
            selected_topics = np.random.choice(research_topics, size=np.random.randint(3, 6), replace=False)

            abstract = (
                f"This study investigates {selected_topics[0]} and {selected_topics[1]} "
                f"in the context of {selected_topics[2]}. We propose a novel approach "
                f"combining {' and '.join(selected_topics[3:])} to address key challenges. "
                f"Our experimental results on benchmark datasets demonstrate significant "
                f"improvements in accuracy and efficiency. The proposed method achieves "
                f"state-of-the-art performance while reducing computational complexity."
            )

            keywords = "; ".join(selected_topics[:4])

            data.append({
                'id': f'pub_{article_id:04d}',
                'title': f'Advances in {selected_topics[0].title()}: A {selected_topics[1].title()} Approach',
                'authors': f'Researcher {article_id} et al. ({institutions[country_idx]}, {countries[country_idx]})',
                'year': year,
                'abstract': abstract,
                'keywords': keywords,
                'doi': f'10.1234/ai.{year}.{article_id}',
                'source': 'Research Database',
                'publication_type': np.random.choice(['journal', 'conference'], p=[0.4, 0.6]),
                'journal_conference': np.random.choice(venues)
            })
            article_id += 1

    df = pd.DataFrame(data)

    # Save
    output_dir = Path('data/sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'pipeline_demo_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.success(f"Sample data created: {output_path} ({len(df)} records)")
    return output_path


def run_pipeline_demo():
    """Run complete pipeline demonstration."""
    logger.info("=" * 80)
    logger.info("COMPLETE VISUALIZATION PIPELINE DEMONSTRATION")
    logger.info("=" * 80)

    # Create sample data
    data_path = create_sample_data()

    # Initialize pipeline
    logger.info("\nInitializing visualization pipeline...")
    pipeline = VisualizationPipeline(
        unified_data_path=str(data_path),
        output_dir='output/complete_pipeline'
    )

    logger.info(f"Pipeline initialized with {len(pipeline.df)} records")
    logger.info(f"Output directory: {pipeline.output_dir}")

    # Run complete pipeline
    logger.info("\nExecuting complete visualization pipeline...")
    results = pipeline.run_all_visualizations()

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION RESULTS")
    logger.info("=" * 80)

    if results['status'] == 'success':
        logger.success("\n✓ Pipeline completed successfully!")

        # Show successful visualizations
        logger.info(f"\n✓ Successful visualizations ({len(results['results']['successful'])}):")
        for viz in results['results']['successful']:
            logger.info(f"  - {viz.title()}")

        # Show failed visualizations (if any)
        if results['results']['failed']:
            logger.warning(f"\n✗ Failed visualizations ({len(results['results']['failed'])}):")
            for viz in results['results']['failed']:
                logger.warning(f"  - {viz.title()}")

        # Show warnings (if any)
        if results['results']['warnings']:
            logger.warning(f"\n⚠ Warnings ({len(results['results']['warnings'])}):")
            for warning in results['results']['warnings']:
                logger.warning(f"  - {warning}")

        # Show outputs
        logger.info("\n📁 Generated outputs:")
        for key, path in results['results']['outputs'].items():
            logger.info(f"  - {key}: {path}")

        # Show timings
        logger.info("\n⏱ Performance metrics:")
        for module, time_taken in results['results']['timings'].items():
            logger.info(f"  - {module.title()}: {time_taken:.2f}s")

        logger.info(f"\n📋 Execution report: {results['report_path']}")

    else:
        logger.error(f"\n✗ Pipeline failed: {results.get('reason', 'Unknown error')}")

    return results


def demonstrate_incremental_update():
    """Demonstrate incremental update functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("INCREMENTAL UPDATE DEMONSTRATION")
    logger.info("=" * 80)

    # Create new data
    logger.info("\nCreating new studies to add...")
    np.random.seed(123)

    new_data = []
    for i in range(20):
        new_data.append({
            'id': f'pub_new_{i:04d}',
            'title': f'New Research on AI and ML - Study {i}',
            'authors': f'New Author {i} (MIT, USA)',
            'year': 2024,
            'abstract': 'This new study explores cutting-edge topics in artificial intelligence, '
                       'machine learning, deep learning, and neural networks with novel approaches.',
            'keywords': 'AI; machine learning; deep learning; transformers',
            'doi': f'10.1234/new.{i}',
            'source': 'Research Database',
            'publication_type': 'journal',
            'journal_conference': 'Nature Machine Intelligence'
        })

    new_df = pd.DataFrame(new_data)
    new_data_path = Path('data/sample/new_studies.csv')
    new_df.to_csv(new_data_path, index=False, encoding='utf-8')

    logger.success(f"Created {len(new_df)} new studies at {new_data_path}")

    # Load existing pipeline
    logger.info("\nLoading existing pipeline...")
    pipeline = VisualizationPipeline(
        unified_data_path='data/sample/pipeline_demo_data.csv',
        output_dir='output/complete_pipeline'
    )

    # Run incremental update
    logger.info("\nRunning incremental update...")
    results = pipeline.update_visualizations_incremental(str(new_data_path))

    logger.info("\n" + "=" * 80)
    logger.success("✓ Incremental update completed!")
    logger.info("=" * 80)

    return results


def main():
    """Main demonstration function."""
    try:
        # 1. Run complete pipeline
        logger.info("\n" + "🚀 " * 20)
        logger.info("DEMO 1: Complete Pipeline Execution")
        logger.info("🚀 " * 20)

        results1 = run_pipeline_demo()

        # 2. Demonstrate incremental update
        logger.info("\n\n" + "🔄 " * 20)
        logger.info("DEMO 2: Incremental Update")
        logger.info("🔄 " * 20)

        results2 = demonstrate_incremental_update()

        # Final summary
        logger.info("\n\n" + "=" * 80)
        logger.info("DEMONSTRATION COMPLETE")
        logger.info("=" * 80)

        logger.info("\n📊 All visualizations have been generated!")
        logger.info("\n📁 Check the following directories:")
        logger.info("  - output/complete_pipeline/geographic/")
        logger.info("  - output/complete_pipeline/wordclouds/")
        logger.info("  - output/complete_pipeline/timeline/")
        logger.info("  - output/complete_pipeline/reports/")
        logger.info("  - output/complete_pipeline/dashboard/")

        logger.info("\n📄 View the PDF report:")
        logger.info("  - output/complete_pipeline/reports/full_report.pdf")

        logger.info("\n📋 View execution reports:")
        logger.info("  - output/complete_pipeline/reports/execution_report.md")
        logger.info("  - output/complete_pipeline/reports/execution_report.json")

        logger.info("\n🎯 Next steps:")
        logger.info("  1. Open the PDF report to view all visualizations")
        logger.info("  2. Check the interactive HTML files (maps, timelines)")
        logger.info("  3. Review the execution report for performance metrics")
        logger.info("  4. Use the dashboard data for Streamlit visualization")

        logger.success("\n✅ All demonstrations completed successfully!")

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
