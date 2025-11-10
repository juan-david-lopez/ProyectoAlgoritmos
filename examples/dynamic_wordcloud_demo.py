"""
Dynamic Word Cloud Visualization - Example Usage

This script demonstrates how to use the DynamicWordCloud class to:
1. Extract and process terms from scientific publications
2. Calculate term weights using different methods
3. Generate static word clouds with various styles
4. Create interactive word clouds
5. Generate comparative visualizations
6. Perform incremental updates (dynamic)
7. Analyze temporal evolution

Author: Bibliometric Analysis System
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.dynamic_wordcloud import DynamicWordCloud
from loguru import logger


def create_sample_data():
    """
    Create sample unified data for demonstration.
    """
    logger.info("Creating sample data for demonstration")

    sample_data = [
        {
            'id': 'pub_001',
            'title': 'Deep Learning for Computer Vision',
            'abstract': 'This study presents a novel deep learning approach for computer vision tasks. '
                       'We propose a convolutional neural network architecture that achieves state-of-the-art '
                       'performance on image classification and object detection benchmarks.',
            'keywords': 'deep learning; computer vision; neural networks; CNN; image classification',
            'year': 2021
        },
        {
            'id': 'pub_002',
            'title': 'Natural Language Processing with Transformers',
            'abstract': 'Natural language processing has been revolutionized by transformer models. '
                       'This paper explores the application of transformers for text analysis, sentiment '
                       'classification, and machine translation tasks.',
            'keywords': 'NLP; transformers; BERT; attention mechanism; text processing',
            'year': 2021
        },
        {
            'id': 'pub_003',
            'title': 'Quantum Computing Applications',
            'abstract': 'Quantum computing offers unprecedented computational power for specific problems. '
                       'We demonstrate quantum algorithms for optimization and cryptography applications.',
            'keywords': 'quantum computing; quantum algorithms; qubits; optimization',
            'year': 2022
        },
        {
            'id': 'pub_004',
            'title': 'Machine Learning in Healthcare',
            'abstract': 'Machine learning techniques are transforming healthcare diagnostics. This research '
                       'applies supervised learning methods for disease prediction and medical image analysis.',
            'keywords': 'machine learning; healthcare; diagnosis; medical imaging; supervised learning',
            'year': 2022
        },
        {
            'id': 'pub_005',
            'title': 'Artificial Intelligence Ethics and Fairness',
            'abstract': 'As artificial intelligence systems become ubiquitous, addressing ethical concerns '
                       'and ensuring fairness is critical. This paper discusses bias detection and mitigation '
                       'strategies in AI systems.',
            'keywords': 'AI ethics; fairness; bias detection; responsible AI',
            'year': 2022
        },
        {
            'id': 'pub_006',
            'title': 'Robotics and Autonomous Systems',
            'abstract': 'Autonomous robotics systems combine perception, planning, and control. This work '
                       'presents a framework for multi-robot coordination in dynamic environments.',
            'keywords': 'robotics; autonomous systems; robot coordination; control systems',
            'year': 2023
        },
        {
            'id': 'pub_007',
            'title': 'Blockchain Technology for Supply Chain',
            'abstract': 'Blockchain technology provides transparency and security for supply chain management. '
                       'We implement a distributed ledger system for tracking products through the supply chain.',
            'keywords': 'blockchain; distributed ledger; supply chain; smart contracts',
            'year': 2023
        },
        {
            'id': 'pub_008',
            'title': 'Advanced Data Science Techniques',
            'abstract': 'Data science combines statistical analysis, machine learning, and visualization. '
                       'This paper surveys advanced techniques for big data analytics and predictive modeling.',
            'keywords': 'data science; big data; analytics; predictive modeling; statistics',
            'year': 2023
        },
        {
            'id': 'pub_009',
            'title': 'Cloud Computing Security',
            'abstract': 'Cloud computing security requires comprehensive strategies for data protection and '
                       'access control. We propose a multi-layer security framework for cloud infrastructure.',
            'keywords': 'cloud computing; security; encryption; access control; data protection',
            'year': 2023
        },
        {
            'id': 'pub_010',
            'title': 'Internet of Things Development',
            'abstract': 'Internet of Things connects billions of devices for data collection and automation. '
                       'This research explores IoT architectures for smart cities and industrial applications.',
            'keywords': 'IoT; sensors; connectivity; smart cities; industrial IoT',
            'year': 2023
        },
    ]

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # Save to CSV
    output_dir = Path('data/sample')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'sample_wordcloud_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.success(f"Sample data created: {output_path}")
    return output_path


def example_basic_usage():
    """Example 1: Basic term extraction and word cloud generation."""
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Basic Word Cloud Generation")
    logger.info("=" * 70)

    # Create sample data
    data_path = create_sample_data()

    # Initialize DynamicWordCloud
    wc = DynamicWordCloud(str(data_path))

    # Extract and process terms
    logger.info("Extracting and processing terms...")
    terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])

    logger.info(f"Extracted {len(terms)} unique terms")
    logger.info("\nTop 10 terms:")
    top_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:10]
    for term, freq in top_terms:
        logger.info(f"  {term}: {freq}")

    # Calculate weights
    logger.info("\nCalculating term weights...")
    weights = wc.calculate_term_weights(terms, method='log_frequency')

    # Generate basic word cloud
    output_dir = Path('output/wordcloud_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating word cloud...")
    wc.generate_wordcloud(
        weights,
        output_path=str(output_dir / 'wordcloud_basic.png'),
        style='scientific'
    )


def example_different_styles():
    """Example 2: Generate word clouds with different styles."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Different Visual Styles")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    wc = DynamicWordCloud(str(data_path))
    terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])
    weights = wc.calculate_term_weights(terms, method='log_frequency')

    output_dir = Path('output/wordcloud_styles')
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = ['scientific', 'colorful', 'academic', 'tech']

    for style in styles:
        logger.info(f"Generating {style} style word cloud...")
        wc.generate_wordcloud(
            weights,
            output_path=str(output_dir / f'wordcloud_{style}.png'),
            style=style,
            max_words=100
        )

    logger.info(f"\nAll styles saved to: {output_dir}")


def example_weighting_methods():
    """Example 3: Compare different term weighting methods."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Different Weighting Methods")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    wc = DynamicWordCloud(str(data_path))
    terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])

    output_dir = Path('output/wordcloud_weights')
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['frequency', 'log_frequency', 'normalized', 'tfidf']

    for method in methods:
        logger.info(f"Calculating weights with method: {method}")
        weights = wc.calculate_term_weights(terms, method=method)

        logger.info(f"Generating word cloud with {method} weights...")
        wc.generate_wordcloud(
            weights,
            output_path=str(output_dir / f'wordcloud_{method}.png'),
            style='scientific',
            max_words=100
        )

    logger.info(f"\nAll weight methods saved to: {output_dir}")


def example_interactive_wordcloud():
    """Example 4: Generate interactive word cloud."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Interactive Word Cloud")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    wc = DynamicWordCloud(str(data_path))
    terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])
    weights = wc.calculate_term_weights(terms, method='tfidf')

    output_dir = Path('output/wordcloud_interactive')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating interactive word cloud...")
    wc.generate_interactive_wordcloud(
        weights,
        output_html=str(output_dir / 'wordcloud_interactive.html'),
        max_words=80
    )

    logger.info(f"\nInteractive word cloud saved to: {output_dir}")
    logger.info("Open the HTML file in a web browser to interact with it!")


def example_comparative_wordclouds():
    """Example 5: Create comparative word clouds."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Comparative Word Clouds")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    wc = DynamicWordCloud(str(data_path))

    output_dir = Path('output/wordcloud_comparative')

    logger.info("Creating comparative word clouds...")
    logger.info("  - Abstracts only")
    logger.info("  - Keywords only")
    logger.info("  - Combined")
    logger.info("  - By year")

    wc.create_comparative_wordclouds(
        output_dir=str(output_dir),
        style='scientific',
        dpi=300
    )

    logger.info(f"\nComparative word clouds saved to: {output_dir}")


def example_incremental_update():
    """Example 6: Demonstrate incremental update (dynamic feature)."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 6: Incremental Update (Dynamic Word Cloud)")
    logger.info("=" * 70)

    # Create initial data
    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    # Generate initial word cloud
    logger.info("Step 1: Generating initial word cloud...")
    wc = DynamicWordCloud(str(data_path))
    terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])
    weights = wc.calculate_term_weights(terms, method='log_frequency')

    output_dir = Path('output/wordcloud_dynamic')
    output_dir.mkdir(parents=True, exist_ok=True)

    wc.generate_wordcloud(
        weights,
        output_path=str(output_dir / 'wordcloud_initial.png'),
        style='scientific'
    )

    # Save weights for later
    weights_path = output_dir / 'term_weights_initial.pkl'
    wc.save_term_weights(str(weights_path), weights)
    logger.info(f"Initial weights saved to: {weights_path}")

    # Create "new" data (in real scenario, this would be new publications)
    logger.info("\nStep 2: Creating new data...")
    new_data = pd.DataFrame([
        {
            'id': 'pub_011',
            'title': 'Reinforcement Learning for Robotics',
            'abstract': 'Reinforcement learning enables robots to learn from interaction. This work '
                       'demonstrates deep reinforcement learning for robotic manipulation tasks.',
            'keywords': 'reinforcement learning; robotics; deep RL; robotic manipulation',
            'year': 2024
        },
        {
            'id': 'pub_012',
            'title': 'Edge Computing for IoT',
            'abstract': 'Edge computing brings computation closer to IoT devices. We propose an edge '
                       'computing framework for real-time data processing in IoT networks.',
            'keywords': 'edge computing; IoT; real-time processing; distributed computing',
            'year': 2024
        }
    ])

    new_data_path = output_dir / 'new_data.csv'
    new_data.to_csv(new_data_path, index=False, encoding='utf-8')

    # Update word cloud incrementally
    logger.info("\nStep 3: Updating word cloud with new data...")
    updated_weights = wc.update_wordcloud_incremental(
        new_data_path=str(new_data_path),
        previous_weights_path=str(weights_path),
        output_path=str(output_dir / 'wordcloud_updated.png'),
        style='scientific'
    )

    logger.info(f"\nDynamic update complete! Compare:")
    logger.info(f"  Initial: {output_dir / 'wordcloud_initial.png'}")
    logger.info(f"  Updated: {output_dir / 'wordcloud_updated.png'}")


def example_temporal_evolution():
    """Example 7: Analyze temporal evolution of terms."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 7: Temporal Evolution Analysis")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    wc = DynamicWordCloud(str(data_path))

    output_dir = Path('output/wordcloud_evolution')

    logger.info("Analyzing temporal evolution of terms...")
    logger.info("  - Creating word clouds for each year")
    logger.info("  - Identifying emerging terms")
    logger.info("  - Identifying declining terms")
    logger.info("  - Creating GIF animation")

    wc.generate_wordcloud_evolution(
        output_dir=str(output_dir),
        create_animation=True,
        style='scientific'
    )

    # Load and display trends
    trends_file = output_dir / 'term_trends.json'
    if trends_file.exists():
        import json
        with open(trends_file, 'r', encoding='utf-8') as f:
            trends = json.load(f)

        logger.info(f"\nPeriod: {trends['period']}")

        logger.info("\nTop 5 Emerging Terms:")
        for item in trends['emerging_terms'][:5]:
            logger.info(f"  {item['term']}: growth = {item['growth']:.2f}")

        logger.info("\nTop 5 Declining Terms:")
        for item in trends['declining_terms'][:5]:
            logger.info(f"  {item['term']}: decline = {item['decline']:.2f}")

    logger.info(f"\nEvolution analysis saved to: {output_dir}")


def example_complete_workflow():
    """Example 8: Complete workflow demonstration."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 8: Complete Word Cloud Workflow")
    logger.info("=" * 70)

    data_path = Path('data/sample/sample_wordcloud_data.csv')
    if not data_path.exists():
        data_path = create_sample_data()

    # Initialize
    logger.info("Initializing DynamicWordCloud...")
    wc = DynamicWordCloud(str(data_path))

    # Create output directory
    output_dir = Path('output/wordcloud_complete')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract terms
    logger.info("\n[Step 1/6] Extracting and processing terms...")
    terms = wc.extract_and_process_terms(
        sources=['abstract', 'keywords'],
        ngram_range=(1, 3),
        max_terms=200
    )
    logger.info(f"  Extracted {len(terms)} unique terms")

    # Step 2: Calculate weights
    logger.info("\n[Step 2/6] Calculating term weights (TF-IDF)...")
    weights = wc.calculate_term_weights(terms, method='tfidf')

    # Step 3: Generate static word clouds
    logger.info("\n[Step 3/6] Generating static word clouds...")
    for style in ['scientific', 'colorful']:
        wc.generate_wordcloud(
            weights,
            output_path=str(output_dir / f'wordcloud_{style}.png'),
            style=style,
            dpi=300
        )

    # Step 4: Generate interactive word cloud
    logger.info("\n[Step 4/6] Generating interactive word cloud...")
    wc.generate_interactive_wordcloud(
        weights,
        output_html=str(output_dir / 'wordcloud_interactive.html')
    )

    # Step 5: Create comparative visualizations
    logger.info("\n[Step 5/6] Creating comparative word clouds...")
    comp_dir = output_dir / 'comparative'
    wc.create_comparative_wordclouds(
        output_dir=str(comp_dir),
        style='scientific'
    )

    # Step 6: Analyze temporal evolution
    logger.info("\n[Step 6/6] Analyzing temporal evolution...")
    evol_dir = output_dir / 'evolution'
    wc.generate_wordcloud_evolution(
        output_dir=str(evol_dir),
        create_animation=True
    )

    logger.success(f"\n{' COMPLETE ':=^70}")
    logger.success(f"All outputs saved to: {output_dir}")
    logger.success("=" * 70)


def main():
    """Run all examples."""
    logger.info("Dynamic Word Cloud Visualization - Comprehensive Demo")
    logger.info("=" * 70)

    try:
        # Run examples
        example_basic_usage()
        example_different_styles()
        example_weighting_methods()
        example_interactive_wordcloud()
        example_comparative_wordclouds()
        example_incremental_update()
        example_temporal_evolution()
        example_complete_workflow()

        logger.success("\n" + "=" * 70)
        logger.success("All examples completed successfully!")
        logger.success("=" * 70)

        logger.info("\nGenerated outputs:")
        logger.info("  - output/wordcloud_demo/        : Basic word clouds")
        logger.info("  - output/wordcloud_styles/      : Different visual styles")
        logger.info("  - output/wordcloud_weights/     : Different weighting methods")
        logger.info("  - output/wordcloud_interactive/ : Interactive HTML word cloud")
        logger.info("  - output/wordcloud_comparative/ : Comparative analysis")
        logger.info("  - output/wordcloud_dynamic/     : Incremental updates")
        logger.info("  - output/wordcloud_evolution/   : Temporal evolution + GIF")
        logger.info("  - output/wordcloud_complete/    : Complete workflow")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
