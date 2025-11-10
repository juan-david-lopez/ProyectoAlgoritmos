"""
Automation Pipeline
Complete automated workflow for downloading and unifying bibliographic data
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.scrapers.acm_scraper import ACMScraper
from src.scrapers.sciencedirect_scraper import ScienceDirectScraper
from src.scrapers.semantic_scholar_api import SemanticScholarAPI
from src.preprocessing.data_unifier import DataUnifier


class AutomationPipeline:
    """
    Automated pipeline for bibliographic data collection and unification

    Workflow:
    1. Execute ACM scraper
    2. Execute ScienceDirect scraper
    3. Unify results
    4. Detect and handle duplicates
    5. Generate execution report
    """

    def __init__(self, config, query: str, max_results_per_source: int = 100):
        """
        Initialize automation pipeline

        Args:
            config: Configuration object
            query: Search query
            max_results_per_source: Maximum results per source
        """
        self.config = config
        self.query = query
        self.max_results = max_results_per_source

        # Initialize components
        self.acm_scraper = None
        self.sd_scraper = None
        self.semantic_scraper = None
        self.unifier = DataUnifier(config)

        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'acm_records': 0,
            'sciencedirect_records': 0,
            'total_downloaded': 0,
            'duplicates_found': 0,
            'final_unique_records': 0,
            'errors': [],
            'warnings': []
        }

        logger.info("Automation Pipeline initialized")
        logger.info(f"Query: '{query}'")
        logger.info(f"Max results per source: {max_results_per_source}")

    def setup_logging(self):
        """Configure logging for pipeline"""
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Add file handler with rotation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"automation_pipeline_{timestamp}.log"

        logger.add(
            log_file,
            rotation="10 MB",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )

        logger.info("Logging configured")

    def run_acm_scraper(self) -> List[Dict[str, Any]]:
        """
        Execute ACM Digital Library scraper

        Returns:
            List of records from ACM
        """
        logger.info("=" * 70)
        logger.info("STEP 1: ACM Digital Library Scraping")
        logger.info("=" * 70)

        try:
            # Initialize scraper
            self.acm_scraper = ACMScraper(self.config, headless=True)

            # Execute scraping
            records = self.acm_scraper.scrape(
                query=self.query,
                max_results=self.max_results
            )

            self.stats['acm_records'] = len(records)

            logger.success(f"ACM scraping complete: {len(records)} records")
            return records

        except Exception as e:
            error_msg = f"ACM scraping failed: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return []

    def run_sciencedirect_scraper(self) -> List[Dict[str, Any]]:
        """
        Execute ScienceDirect scraper

        Returns:
            List of records from ScienceDirect
        """
        logger.info("=" * 70)
        logger.info("STEP 2: ScienceDirect Scraping")
        logger.info("=" * 70)

        try:
            # Initialize scraper
            self.sd_scraper = ScienceDirectScraper(self.config, headless=True)

            # Execute scraping
            records = self.sd_scraper.scrape(
                query=self.query,
                max_results=self.max_results
            )

            self.stats['sciencedirect_records'] = len(records)

            logger.success(f"ScienceDirect scraping complete: {len(records)} records")
            return records

        except Exception as e:
            error_msg = f"ScienceDirect scraping failed: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return []

    def run_semantic_scholar_scraper(self) -> List[Dict[str, Any]]:
        """
        Execute Semantic Scholar API scraper (fallback method)

        Returns:
            List of records from Semantic Scholar
        """
        logger.info("=" * 70)
        logger.info("STEP 3 (Fallback): Semantic Scholar API Scraping")
        logger.info("=" * 70)

        try:
            # Initialize scraper
            self.semantic_scraper = SemanticScholarAPI(self.config)

            # Execute scraping
            records = self.semantic_scraper.scrape(
                query=self.query,
                max_results=self.max_results * 2  # Get more since this is the only source
            )

            self.stats['semantic_scholar_records'] = len(records)

            logger.success(f"Semantic Scholar scraping complete: {len(records)} records")
            return records

        except Exception as e:
            error_msg = f"Semantic Scholar scraping failed: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return []

    def unify_data(self, records_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Unify data from all sources

        Args:
            records_list: List of record lists from different sources

        Returns:
            Unification statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 3: Data Unification")
        logger.info("=" * 70)

        try:
            # Generate output filenames with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unified_filename = f'unified_data_{timestamp}.csv'
            duplicates_filename = f'duplicates_log_{timestamp}.csv'

            # Execute unification
            unify_stats = self.unifier.unify(
                records_list,
                output_filename=unified_filename,
                duplicates_filename=duplicates_filename
            )

            # Update statistics
            self.stats['total_downloaded'] = unify_stats['original_count']
            self.stats['duplicates_found'] = unify_stats['duplicates_count']
            self.stats['final_unique_records'] = unify_stats['clean_count']
            self.stats['unified_file'] = unify_stats['unified_file']
            self.stats['duplicates_file'] = unify_stats['duplicates_file']

            logger.success("Data unification complete")
            return unify_stats

        except Exception as e:
            error_msg = f"Data unification failed: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise

    def generate_report(self) -> str:
        """
        Generate execution report

        Returns:
            Report as formatted string
        """
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        duration_minutes = duration / 60

        report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║              AUTOMATION PIPELINE EXECUTION REPORT                 ║
╚═══════════════════════════════════════════════════════════════════╝

📋 Query Information:
   Search Query: "{self.query}"
   Max Results per Source: {self.max_results}

⏱️  Execution Time:
   Start: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
   End:   {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
   Duration: {duration:.2f}s ({duration_minutes:.2f} minutes)

📊 Download Statistics:
   ACM Digital Library: {self.stats['acm_records']} records
   ScienceDirect:       {self.stats['sciencedirect_records']} records
   Semantic Scholar:    {self.stats.get('semantic_scholar_records', 0)} records
   Total Downloaded:    {self.stats['total_downloaded']} records

🔄 Deduplication Results:
   Duplicates Found:    {self.stats['duplicates_found']} records
   Final Unique Records: {self.stats['final_unique_records']} records
   Deduplication Rate:  {(self.stats['duplicates_found']/self.stats['total_downloaded']*100) if self.stats['total_downloaded'] > 0 else 0:.2f}%

📁 Output Files:
   Unified Data: {self.stats.get('unified_file', 'N/A')}
   Duplicates Log: {self.stats.get('duplicates_file', 'N/A')}

{'⚠️  Warnings: ' + str(len(self.stats['warnings'])) if self.stats['warnings'] else '✓ No warnings'}
{'❌ Errors: ' + str(len(self.stats['errors'])) if self.stats['errors'] else '✓ No errors'}

{'═' * 70}
Status: {'✓ SUCCESS' if not self.stats['errors'] else '✗ COMPLETED WITH ERRORS'}
{'═' * 70}
"""

        # Add error details if any
        if self.stats['errors']:
            report += "\n\nError Details:\n"
            for i, error in enumerate(self.stats['errors'], 1):
                report += f"  {i}. {error}\n"

        return report

    def save_report(self, report: str):
        """
        Save execution report to file

        Args:
            report: Report string
        """
        # Create reports directory
        reports_dir = Path('outputs/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f'automation_report_{timestamp}.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Report saved to: {report_file}")

    def run(self) -> Dict[str, Any]:
        """
        Execute complete automation pipeline

        Returns:
            Statistics dictionary
        """
        self.stats['start_time'] = datetime.now()

        logger.info("╔═══════════════════════════════════════════════════════════════════╗")
        logger.info("║        STARTING AUTOMATION PIPELINE                               ║")
        logger.info("╚═══════════════════════════════════════════════════════════════════╝")

        try:
            # Step 1: ACM Scraping
            acm_records = self.run_acm_scraper()

            # Step 2: ScienceDirect Scraping
            sd_records = self.run_sciencedirect_scraper()

            # Check if we have any records
            if not acm_records and not sd_records:
                logger.warning("Web scrapers failed, using Semantic Scholar API as fallback...")
                semantic_records = self.run_semantic_scholar_scraper()
                
                if not semantic_records:
                    logger.error("All data sources failed - no data collected")
                    raise Exception("All data sources failed - no data collected")
                
                # Use only Semantic Scholar records
                records_list = [semantic_records]
            else:
                # Step 3: Unification
                records_list = []
                if acm_records:
                    records_list.append(acm_records)
                if sd_records:
                    records_list.append(sd_records)

            unify_stats = self.unify_data(records_list)

            # Mark completion time
            self.stats['end_time'] = datetime.now()

            # Generate and display report
            report = self.generate_report()
            logger.info("\n" + report)

            # Save report
            self.save_report(report)

            logger.success("╔═══════════════════════════════════════════════════════════════════╗")
            logger.success("║        AUTOMATION PIPELINE COMPLETED SUCCESSFULLY                 ║")
            logger.success("╚═══════════════════════════════════════════════════════════════════╝")

            return self.stats

        except Exception as e:
            self.stats['end_time'] = datetime.now()
            logger.error(f"Pipeline failed: {e}", exc_info=True)

            # Still generate report
            report = self.generate_report()
            logger.info("\n" + report)
            self.save_report(report)

            raise


def main():
    """Main entry point for automation pipeline"""
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Automated Bibliographic Data Collection and Unification Pipeline'
    )

    parser.add_argument(
        '--query',
        type=str,
        default='generative artificial intelligence',
        help='Search query (default: "generative artificial intelligence")'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=100,
        help='Maximum results per source (default: 100)'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browsers in headless mode'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Create pipeline
    pipeline = AutomationPipeline(
        config=config,
        query=args.query,
        max_results_per_source=args.max_results
    )

    # Setup logging
    pipeline.setup_logging()

    # Run pipeline
    try:
        stats = pipeline.run()
        return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
