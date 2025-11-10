"""
Main entry point for the Bibliometric Analysis project

This module provides a CLI interface for executing the bibliometric analysis pipeline.
It supports multiple execution modes and provides comprehensive logging.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger, get_logger
from src.utils.file_handler import FileHandler


def print_banner():
    """Print application banner"""
    banner = """
    ===================================================================
    
              BIBLIOMETRIC ANALYSIS - GENERATIVE AI
    
             Analisis de Publicaciones Cientificas v1.0.0
    
    ===================================================================
    """
    print(banner)


def print_menu():
    """Print interactive menu"""
    menu = """
    -----------------------------------------------------------------
                            EXECUTION MODES
    -----------------------------------------------------------------
      1. scrape      - Download data from academic databases
      2. deduplicate - Detect and remove duplicates
      3. preprocess  - Clean and preprocess data
      4. cluster     - Perform clustering analysis
      5. visualize   - Generate visualizations
      6. report      - Create PDF report
      7. full        - Execute complete pipeline
      8. info        - Show project information
      9. exit        - Exit application
    -----------------------------------------------------------------
    """
    print(menu)


def print_info(config):
    """Print project information"""
    info = f"""
    ===================================================================
                          PROJECT INFORMATION
    ===================================================================

    Project Name: {config.get('project.name', 'Bibliometric Analysis')}
    Description:  {config.get('project.description', 'N/A')}
    Version:      {config.get('project.version', '1.0.0')}
    Author:       {config.get('project.author', 'Research Team')}

    Search Query:
       Keywords:     {config.get('query.keywords', 'N/A')}
       Language:     {config.get('query.language', 'N/A')}
       Date Range:   {config.get('query.date_range.start', 'N/A')} to {config.get('query.date_range.end', 'current')}

    Data Sources:
       IEEE:         {'[X] Enabled' if config.is_source_enabled('ieee') else '[ ] Disabled'}
       Scopus:       {'[X] Enabled' if config.is_source_enabled('scopus') else '[ ] Disabled'}
       Web of Science: {'[X] Enabled' if config.is_source_enabled('web_of_science') else '[ ] Disabled'}

    Clustering Algorithms:
       K-Means:      {'[X] Enabled' if config.get('clustering.algorithms.kmeans.enabled') else '[ ] Disabled'}
       DBSCAN:       {'[X] Enabled' if config.get('clustering.algorithms.dbscan.enabled') else '[ ] Disabled'}
       Hierarchical: {'[X] Enabled' if config.get('clustering.algorithms.hierarchical.enabled') else '[ ] Disabled'}

    Output Paths:
       Raw Data:     {config.get('paths.raw_data', 'data/raw')}
       Processed:    {config.get('paths.processed_data', 'data/processed')}
       Outputs:      {config.get('paths.outputs', 'outputs')}

    ===================================================================
    """
    print(info)


def run_scraping(config, logger, file_handler, sources: Optional[List[str]] = None):
    """
    Execute web scraping module

    Args:
        config: Configuration object
        logger: Logger instance
        file_handler: File handler instance
        sources: List of sources to scrape (None = all enabled)
    """
    logger.info("=" * 70)
    logger.info("STEP 1: WEB SCRAPING - Downloading Publications")
    logger.info("=" * 70)

    try:
        # Import scraper modules
        from src.scrapers.ieee_scraper import IEEEScraper
        from src.scrapers.scopus_scraper import ScopusScraper
        from src.scrapers.wos_scraper import WOSScraper

        # Determine which sources to scrape
        if sources is None:
            sources = []
            if config.is_source_enabled('ieee'):
                sources.append('ieee')
            if config.is_source_enabled('scopus'):
                sources.append('scopus')
            if config.is_source_enabled('web_of_science'):
                sources.append('web_of_science')

        logger.info(f"Scraping from sources: {', '.join(sources)}")

        results = {}

        # IEEE Scraper
        if 'ieee' in sources:
            logger.info("\nScraping IEEE Xplore...")
            ieee_scraper = IEEEScraper(config)
            ieee_results = ieee_scraper.scrape()
            results['ieee'] = ieee_results
            logger.info(f"[OK] IEEE: {len(ieee_results)} publications downloaded")

        # Scopus Scraper
        if 'scopus' in sources:
            logger.info("\nScraping Scopus...")
            scopus_scraper = ScopusScraper(config)
            scopus_results = scopus_scraper.scrape()
            results['scopus'] = scopus_results
            logger.info(f"[OK] Scopus: {len(scopus_results)} publications downloaded")

        # Web of Science Scraper
        if 'web_of_science' in sources:
            logger.info("\nScraping Web of Science...")
            wos_scraper = WOSScraper(config)
            wos_results = wos_scraper.scrape()
            results['web_of_science'] = wos_results
            logger.info(f"[OK] WOS: {len(wos_results)} publications downloaded")

        # Save results
        total_publications = sum(len(r) for r in results.values())
        logger.info(f"\n[OK] Total publications downloaded: {total_publications}")

        logger.info("=" * 70)
        logger.info("SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return results

    except ImportError as e:
        logger.warning(f"WARNING: Scraper modules not yet implemented: {e}")
        logger.info("Placeholder: Scraping functionality will be implemented")
        return {}
    except Exception as e:
        logger.error(f"ERROR: Error during scraping: {str(e)}", exc_info=True)
        raise


def run_deduplication(config, logger, file_handler):
    """Execute deduplication module"""
    logger.info("=" * 70)
    logger.info("STEP 2: DEDUPLICATION - Removing Duplicate Publications")
    logger.info("=" * 70)

    try:
        from src.preprocessing.deduplicator import Deduplicator

        deduplicator = Deduplicator(config)
        results = deduplicator.deduplicate()

        logger.info(f"[OK] Original records: {results.get('original_count', 0)}")
        logger.info(f"[OK] Duplicates found: {results.get('duplicates_count', 0)}")
        logger.info(f"[OK] Clean records: {results.get('clean_count', 0)}")

        logger.info("=" * 70)
        logger.info("DEDUPLICATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return results

    except ImportError as e:
        logger.warning(f"WARNING: Deduplication module not yet implemented: {e}")
        logger.info("Placeholder: Deduplication functionality will be implemented")
        return {}
    except Exception as e:
        logger.error(f"ERROR: Error during deduplication: {str(e)}", exc_info=True)
        raise


def run_preprocessing(config, logger, file_handler):
    """Execute preprocessing module"""
    logger.info("=" * 70)
    logger.info("STEP 3: PREPROCESSING - Cleaning and Normalizing Data")
    logger.info("=" * 70)

    try:
        from src.preprocessing.text_processor import TextProcessor

        processor = TextProcessor(config)
        results = processor.process()

        logger.info(f"[OK] Records processed: {results.get('processed_count', 0)}")

        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return results

    except ImportError as e:
        logger.warning(f"⚠️  Preprocessing module not yet implemented: {e}")
        logger.info("Placeholder: Preprocessing functionality will be implemented")
        return {}
    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {str(e)}", exc_info=True)
        raise


def run_clustering(config, logger, file_handler):
    """Execute clustering module"""
    logger.info("=" * 70)
    logger.info("STEP 4: CLUSTERING - Thematic Analysis")
    logger.info("=" * 70)

    try:
        from src.clustering.kmeans_clustering import KMeansClustering
        from src.clustering.dbscan_clustering import DBSCANClustering
        from src.clustering.hierarchical_clustering import HierarchicalClustering

        results = {}

        # K-Means
        if config.get('clustering.algorithms.kmeans.enabled'):
            logger.info("\n📊 Running K-Means clustering...")
            kmeans = KMeansClustering(config)
            results['kmeans'] = kmeans.cluster()
            logger.info(f"✓ K-Means completed")

        # DBSCAN
        if config.get('clustering.algorithms.dbscan.enabled'):
            logger.info("\n Running DBSCAN clustering...")
            dbscan = DBSCANClustering(config)
            results['dbscan'] = dbscan.cluster()
            logger.info(f"✓ DBSCAN completed")

        # Hierarchical
        if config.get('clustering.algorithms.hierarchical.enabled'):
            logger.info("\n Running Hierarchical clustering...")
            hierarchical = HierarchicalClustering(config)
            results['hierarchical'] = hierarchical.cluster()
            logger.info(f"✓ Hierarchical clustering completed")

        logger.info("=" * 70)
        logger.info("CLUSTERING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return results

    except ImportError as e:
        logger.warning(f"⚠️  Clustering modules not yet implemented: {e}")
        logger.info("Placeholder: Clustering functionality will be implemented")
        return {}
    except Exception as e:
        logger.error(f"❌ Error during clustering: {str(e)}", exc_info=True)
        raise


def run_visualization(config, logger, file_handler):
    """Execute visualization module"""
    logger.info("=" * 70)
    logger.info("STEP 5: VISUALIZATION - Generating Charts and Graphs")
    logger.info("=" * 70)

    try:
        from src.visualization.temporal_plots import TemporalPlots
        from src.visualization.geographic_maps import GeographicMaps
        from src.visualization.network_graphs import NetworkGraphs
        from src.visualization.cluster_plots import ClusterPlots

        visualizations = []

        # Temporal plots
        if config.get('visualization.charts.temporal_trends.enabled'):
            logger.info("\n📈 Generating temporal trends...")
            temporal = TemporalPlots(config)
            temporal.generate()
            visualizations.append('temporal_trends')

        # Geographic maps
        if config.get('visualization.charts.country_distribution.enabled'):
            logger.info("\n🌍 Generating geographic distribution...")
            geographic = GeographicMaps(config)
            geographic.generate()
            visualizations.append('geographic_maps')

        # Network graphs
        if config.get('visualization.charts.coauthorship_network.enabled'):
            logger.info("\n🕸️  Generating coauthorship network...")
            network = NetworkGraphs(config)
            network.generate()
            visualizations.append('network_graphs')

        # Cluster visualization
        if config.get('visualization.charts.cluster_visualization.enabled'):
            logger.info("\n📍 Generating cluster visualization...")
            clusters = ClusterPlots(config)
            clusters.generate()
            visualizations.append('cluster_plots')

        logger.info(f"\n✓ Generated {len(visualizations)} visualizations")

        logger.info("=" * 70)
        logger.info("VISUALIZATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return visualizations

    except ImportError as e:
        logger.warning(f"⚠️  Visualization modules not yet implemented: {e}")
        logger.info("Placeholder: Visualization functionality will be implemented")
        return []
    except Exception as e:
        logger.error(f"❌ Error during visualization: {str(e)}", exc_info=True)
        raise


def run_report(config, logger, file_handler):
    """Execute report generation module"""
    logger.info("=" * 70)
    logger.info("STEP 6: REPORT GENERATION - Creating PDF Report")
    logger.info("=" * 70)

    try:
        from src.visualization.report_generator import ReportGenerator

        generator = ReportGenerator(config)
        report_path = generator.generate()

        logger.info(f"✓ Report generated: {report_path}")

        logger.info("=" * 70)
        logger.info("REPORT GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return report_path

    except ImportError as e:
        logger.warning(f"⚠️  Report generation module not yet implemented: {e}")
        logger.info("Placeholder: Report generation functionality will be implemented")
        return None
    except Exception as e:
        logger.error(f"❌ Error during report generation: {str(e)}", exc_info=True)
        raise


def run_pipeline(config, logger, file_handler, mode: str, **kwargs):
    """
    Execute the bibliometric analysis pipeline

    Args:
        config: Configuration object
        logger: Logger instance
        file_handler: File handler instance
        mode: Execution mode
        **kwargs: Additional arguments
    """
    start_time = datetime.now()

    try:
        if mode in ['scrape', 'full']:
            run_scraping(config, logger, file_handler, sources=kwargs.get('sources'))

        if mode in ['deduplicate', 'full']:
            run_deduplication(config, logger, file_handler)

        if mode in ['preprocess', 'full']:
            run_preprocessing(config, logger, file_handler)

        if mode in ['cluster', 'full']:
            run_clustering(config, logger, file_handler)

        if mode in ['visualize', 'full']:
            run_visualization(config, logger, file_handler)

        if mode in ['report', 'full']:
            run_report(config, logger, file_handler)

        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "=" * 70)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
        logger.info(f"⏱️  Total execution time: {duration}")
        logger.info("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        raise


def interactive_mode(config, logger, file_handler):
    """Run in interactive CLI menu mode"""
    while True:
        print_menu()
        choice = input("\n👉 Select an option (1-9): ").strip()

        mode_map = {
            '1': 'scrape',
            '2': 'deduplicate',
            '3': 'preprocess',
            '4': 'cluster',
            '5': 'visualize',
            '6': 'report',
            '7': 'full',
            '8': 'info',
            '9': 'exit'
        }

        if choice == '8':
            print_info(config)
            continue

        if choice == '9':
            print("\n👋 Goodbye! Thanks for using Bibliometric Analysis.\n")
            break

        if choice not in mode_map:
            print("\n❌ Invalid option. Please select 1-9.\n")
            continue

        mode = mode_map[choice]

        print(f"\n🚀 Starting {mode} mode...\n")

        try:
            run_pipeline(config, logger, file_handler, mode)
            input("\n✓ Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user.\n")
            input("Press Enter to continue...")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            input("Press Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Bibliometric Analysis - Inteligencia Artificial Generativa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py --mode full              # Run complete pipeline
  python main.py --mode scrape            # Only download data
  python main.py --mode cluster           # Only clustering
  python main.py --mode scrape --sources ieee,scopus  # Specific sources
  python main.py --config custom.yaml     # Custom config file
        """
    )

    parser.add_argument(
        '--mode',
        choices=['scrape', 'deduplicate', 'preprocess', 'cluster', 'visualize', 'report', 'full', 'interactive'],
        default='interactive',
        help='Execution mode (default: interactive)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )

    parser.add_argument(
        '--sources',
        type=str,
        help='Comma-separated list of sources to scrape (e.g., ieee,scopus,web_of_science)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    try:
        # Load configuration
        config = get_config(args.config)

        # Setup logging
        log_level = 'DEBUG' if args.debug else ('INFO' if args.verbose else config.get('logging.level', 'INFO'))
        setup_logger(level=log_level, config=config)
        logger = get_logger(__name__)

        # Initialize file handler
        file_handler = FileHandler(config)

        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Log level: {log_level}")

        # Parse sources if provided
        sources = None
        if args.sources:
            sources = [s.strip() for s in args.sources.split(',')]
            logger.info(f"Selected sources: {sources}")

        # Execute based on mode
        if args.mode == 'interactive':
            interactive_mode(config, logger, file_handler)
        else:
            run_pipeline(config, logger, file_handler, args.mode, sources=sources)

        return 0

    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Please ensure the configuration file exists.")
        return 1
    except KeyboardInterrupt:
        print("\n  Operation cancelled by user. Goodbye!\n")
        return 130
    except Exception as e:
        print(f"\n Fatal error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
