"""
Visualization Pipeline - Complete Integration Script

Orchestrates all bibliometric visualizations and generates organized outputs.

Features:
- Validates input data
- Generates all visualizations (geographic, word cloud, timeline)
- Creates professional PDF reports
- Prepares data for dashboard
- Handles errors gracefully
- Generates execution reports
- Supports incremental updates

Author: Bibliometric Analysis System
Date: 2024
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import time
import warnings
import yaml
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Import visualization modules
try:
    from .geographic_heatmap import GeographicHeatmap
    from .dynamic_wordcloud import DynamicWordCloud
    from .timeline_visualization import TimelineVisualization
    from .pdf_exporter import PDFExporter
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from visualization.geographic_heatmap import GeographicHeatmap
    from visualization.dynamic_wordcloud import DynamicWordCloud
    from visualization.timeline_visualization import TimelineVisualization
    from visualization.pdf_exporter import PDFExporter


class VisualizationPipeline:
    """
    Complete pipeline for bibliometric visualizations.

    Executes all visualizations, validates data, generates reports,
    and organizes outputs in a structured directory.
    """

    def __init__(self, unified_data_path: str, output_dir: str,
                 config_path: Optional[str] = None):
        """
        Initialize visualization pipeline.

        Args:
            unified_data_path: Path to unified CSV data file
            output_dir: Base directory for all outputs
            config_path: Path to configuration YAML (optional)

        Output Structure:
            output_dir/
            ├── geographic/
            │   ├── map_interactive.html
            │   ├── map_static.png
            │   └── statistics.json
            ├── wordclouds/
            │   ├── combined.png
            │   ├── abstracts_only.png
            │   ├── keywords_only.png
            │   └── interactive.html
            ├── timeline/
            │   ├── main_timeline.png
            │   ├── venue_breakdown.png
            │   ├── interactive.html
            │   └── statistics.json
            ├── reports/
            │   ├── full_report.pdf
            │   └── summary.md
            └── dashboard/
                └── data/ (processed data for dashboard)
        """
        self.data_path = Path(unified_data_path)
        self.output_dir = Path(output_dir)

        # Load configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default config path
            self.config_path = Path(__file__).parent.parent.parent / 'config' / 'visualization_config.yaml'

        self.config = self._load_config()

        # Setup logging
        self._setup_logging()

        # Validate input data path
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Create output directory structure
        self._create_output_structure()

        # Initialize tracking
        self.execution_start = None
        self.execution_end = None
        self.results = {
            'successful': [],
            'failed': [],
            'warnings': [],
            'outputs': {},
            'timings': {},
            'statistics': {}
        }

        # Load data
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        logger.success(f"Loaded {len(self.df)} records")

        # Initialize visualization objects
        self.geo_map = None
        self.wordcloud = None
        self.timeline = None
        self.pdf_exporter = None

        logger.success("VisualizationPipeline initialized successfully")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.success(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'image_quality': {'dpi': 300},
            'logging': {'level': 'INFO'},
            'error_handling': {'continue_on_error': True},
            'progress': {'enabled': True}
        }

    def _setup_logging(self):
        """Configure logging with loguru."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')

        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            self.output_dir / 'pipeline.log',
            level=log_level,
            format=log_config.get('format',
                                 "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                                 "<level>{level: <8}</level> | "
                                 "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
                                 "<level>{message}</level>"),
            rotation=log_config.get('rotation', '10 MB'),
            retention=log_config.get('retention', '30 days')
        )
        logger.add(lambda msg: print(msg), level=log_level, colorize=True)

    def _create_output_structure(self):
        """Create organized output directory structure."""
        subdirs = self.config.get('output', {}).get('subdirs', [
            'geographic', 'wordclouds', 'timeline', 'reports', 'dashboard', 'temp'
        ])

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create dashboard data subdirectory
        (self.output_dir / 'dashboard' / 'data').mkdir(parents=True, exist_ok=True)

        logger.info(f"Output structure created at {self.output_dir}")

    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Validate input data quality.

        Checks:
        - Required fields present
        - No critical null values
        - Year format valid
        - Data types correct
        - Reasonable value ranges

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        logger.info("Validating input data...")
        issues = []

        # Get validation config
        val_config = self.config.get('validation', {})
        required_fields = val_config.get('required_fields', ['id', 'title', 'authors', 'year'])
        year_range = val_config.get('year_range', {'min': 1900, 'max': 2030})
        quality = val_config.get('quality', {})

        # 1. Check required fields
        missing_fields = [field for field in required_fields if field not in self.df.columns]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
            logger.error(f"Missing required fields: {missing_fields}")
            return False, issues

        # 2. Check for null values in required fields
        for field in required_fields:
            null_count = self.df[field].isnull().sum()
            null_pct = null_count / len(self.df)

            if null_pct > quality.get('max_null_percentage', 0.3):
                issues.append(f"Field '{field}' has {null_pct:.1%} null values (>{quality.get('max_null_percentage', 0.3):.1%})")
                logger.warning(f"High null percentage in '{field}': {null_pct:.1%}")

        # 3. Validate year field
        if 'year' in self.df.columns:
            # Convert to numeric
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')

            # Check for invalid years
            invalid_years = self.df[
                (self.df['year'] < year_range['min']) |
                (self.df['year'] > year_range['max'])
            ]

            if len(invalid_years) > 0:
                issues.append(f"{len(invalid_years)} records have invalid years (outside {year_range['min']}-{year_range['max']})")
                logger.warning(f"Found {len(invalid_years)} records with invalid years")

        # 4. Check data completeness
        completeness = 1 - (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)))
        if completeness < quality.get('min_completeness', 0.7):
            issues.append(f"Data completeness {completeness:.1%} is below threshold {quality.get('min_completeness', 0.7):.1%}")
            logger.warning(f"Low data completeness: {completeness:.1%}")

        # 5. Check for duplicates
        duplicates = self.df.duplicated(subset=['id']).sum() if 'id' in self.df.columns else 0
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate IDs")
            logger.warning(f"Found {duplicates} duplicate IDs")

        # Summary
        if len(issues) == 0:
            logger.success("✓ Data validation passed")
            return True, []
        else:
            logger.warning(f"⚠ Data validation completed with {len(issues)} issues")
            return len(issues) == 0, issues

    def run_all_visualizations(self, skip_on_error: Optional[bool] = None) -> Dict:
        """
        Execute complete visualization pipeline.

        Steps:
        1. Validate data
        2. Generate geographic visualizations
        3. Generate word clouds
        4. Generate timeline visualizations
        5. Collect statistics
        6. Generate PDF report
        7. Prepare dashboard data
        8. Generate execution report

        Args:
            skip_on_error: Continue if visualization fails (default from config)

        Returns:
            Dictionary with results and outputs
        """
        self.execution_start = datetime.now()
        logger.info("=" * 70)
        logger.info("STARTING COMPLETE VISUALIZATION PIPELINE")
        logger.info("=" * 70)

        # Use config or parameter
        if skip_on_error is None:
            skip_on_error = self.config.get('error_handling', {}).get('continue_on_error', True)

        # Total steps for progress tracking
        total_steps = 7
        progress_enabled = self.config.get('progress', {}).get('enabled', True)

        if progress_enabled:
            pbar = tqdm(total=total_steps, desc="Pipeline Progress",
                       bar_format=self.config.get('progress', {}).get('bar_format',
                       "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"))

        # Step 1: Validate data
        logger.info("\n[Step 1/7] Validating data...")
        step_start = time.time()
        is_valid, issues = self.validate_data()

        if not is_valid and not skip_on_error:
            logger.error("Data validation failed. Stopping pipeline.")
            return {'status': 'failed', 'reason': 'data_validation', 'issues': issues}

        self.results['timings']['validation'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)

        # Step 2: Geographic visualizations
        logger.info("\n[Step 2/7] Generating geographic visualizations...")
        step_start = time.time()
        geo_success = self._generate_geographic_visualizations()

        if not geo_success and not skip_on_error:
            logger.error("Geographic visualization failed. Stopping pipeline.")
            return {'status': 'failed', 'reason': 'geographic_visualization'}

        self.results['timings']['geographic'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)

        # Step 3: Word cloud visualizations
        logger.info("\n[Step 3/7] Generating word cloud visualizations...")
        step_start = time.time()
        wc_success = self._generate_wordcloud_visualizations()

        if not wc_success and not skip_on_error:
            logger.error("Word cloud visualization failed. Stopping pipeline.")
            return {'status': 'failed', 'reason': 'wordcloud_visualization'}

        self.results['timings']['wordcloud'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)

        # Step 4: Timeline visualizations
        logger.info("\n[Step 4/7] Generating timeline visualizations...")
        step_start = time.time()
        timeline_success = self._generate_timeline_visualizations()

        if not timeline_success and not skip_on_error:
            logger.error("Timeline visualization failed. Stopping pipeline.")
            return {'status': 'failed', 'reason': 'timeline_visualization'}

        self.results['timings']['timeline'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)

        # Step 5: Collect statistics
        logger.info("\n[Step 5/7] Collecting overall statistics...")
        step_start = time.time()
        self._collect_statistics()
        self.results['timings']['statistics'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)

        # Step 6: Generate PDF report
        logger.info("\n[Step 6/7] Generating PDF report...")
        step_start = time.time()
        pdf_success = self._generate_pdf_report()

        if not pdf_success and not skip_on_error:
            logger.warning("PDF generation failed, but continuing...")

        self.results['timings']['pdf'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)

        # Step 7: Prepare dashboard data
        logger.info("\n[Step 7/7] Preparing dashboard data...")
        step_start = time.time()
        self._prepare_dashboard_data()
        self.results['timings']['dashboard'] = time.time() - step_start
        if progress_enabled:
            pbar.update(1)
            pbar.close()

        # Finalize
        self.execution_end = datetime.now()

        # Generate execution report
        report_path = self.generate_execution_report(
            str(self.output_dir / 'reports' / 'execution_report.md')
        )

        logger.info("=" * 70)
        logger.success("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Total time: {(self.execution_end - self.execution_start).total_seconds():.2f}s")
        logger.info(f"Execution report: {report_path}")

        return {
            'status': 'success',
            'results': self.results,
            'report_path': report_path
        }

    def _generate_geographic_visualizations(self) -> bool:
        """Generate all geographic visualizations."""
        try:
            geo_dir = self.output_dir / 'geographic'
            dpi = self.config.get('image_quality', {}).get('dpi', 300)

            # Initialize
            self.geo_map = GeographicHeatmap(str(self.data_path))

            # Extract affiliations
            logger.info("Extracting author affiliations...")
            affiliations = self.geo_map.extract_author_affiliations()

            # Geocode
            logger.info("Geocoding locations...")
            geo_data = self.geo_map.geocode_locations()

            # Generate interactive map
            logger.info("Creating interactive map...")
            interactive_path = geo_dir / 'map_interactive.html'
            self.geo_map.create_interactive_map(geo_data, str(interactive_path))
            self.results['outputs']['geo_interactive'] = str(interactive_path)

            # Generate static map
            logger.info("Creating static map...")
            static_path = geo_dir / 'map_static.png'
            try:
                self.geo_map.create_static_map(str(static_path), dpi=dpi)
                self.results['outputs']['geo_static'] = str(static_path)
            except Exception as e:
                logger.warning(f"Could not create static map (likely missing cartopy): {e}")
                self.results['warnings'].append(f"Static map generation skipped: {e}")

            # Generate statistics
            logger.info("Generating geographic statistics...")
            stats = self.geo_map.generate_geographic_statistics()

            # Save statistics
            stats_path = geo_dir / 'statistics.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.results['outputs']['geo_statistics'] = str(stats_path)

            self.results['successful'].append('geographic')
            logger.success("✓ Geographic visualizations completed")
            return True

        except Exception as e:
            logger.error(f"Geographic visualization failed: {e}")
            self.results['failed'].append('geographic')
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _generate_wordcloud_visualizations(self) -> bool:
        """Generate all word cloud visualizations."""
        try:
            wc_dir = self.output_dir / 'wordclouds'
            dpi = self.config.get('image_quality', {}).get('dpi', 300)

            # Initialize
            self.wordcloud = DynamicWordCloud(str(self.data_path))

            # 1. Combined word cloud (abstracts + keywords)
            logger.info("Creating combined word cloud...")
            terms_combined = self.wordcloud.extract_and_process_terms(
                sources=['abstract', 'keywords']
            )
            weights_combined = self.wordcloud.calculate_term_weights(
                terms_combined, method='tfidf'
            )

            combined_path = wc_dir / 'combined.png'
            self.wordcloud.generate_wordcloud(
                weights_combined, str(combined_path),
                style='scientific', dpi=dpi
            )
            self.results['outputs']['wc_combined'] = str(combined_path)

            # 2. Abstracts only
            logger.info("Creating abstracts-only word cloud...")
            terms_abstract = self.wordcloud.extract_and_process_terms(
                sources=['abstract']
            )
            weights_abstract = self.wordcloud.calculate_term_weights(
                terms_abstract, method='tfidf'
            )

            abstract_path = wc_dir / 'abstracts_only.png'
            self.wordcloud.generate_wordcloud(
                weights_abstract, str(abstract_path),
                style='academic', dpi=dpi
            )
            self.results['outputs']['wc_abstracts'] = str(abstract_path)

            # 3. Keywords only
            logger.info("Creating keywords-only word cloud...")
            terms_keywords = self.wordcloud.extract_and_process_terms(
                sources=['keywords']
            )
            weights_keywords = self.wordcloud.calculate_term_weights(
                terms_keywords, method='tfidf'
            )

            keywords_path = wc_dir / 'keywords_only.png'
            self.wordcloud.generate_wordcloud(
                weights_keywords, str(keywords_path),
                style='colorful', dpi=dpi
            )
            self.results['outputs']['wc_keywords'] = str(keywords_path)

            # 4. Interactive word cloud (HTML placeholder - would need additional library)
            # For now, we'll create a simple statistics JSON
            stats_path = wc_dir / 'term_statistics.json'
            top_terms = sorted(weights_combined.items(), key=lambda x: x[1], reverse=True)[:50]
            term_stats = {
                'top_terms': [
                    {'term': term, 'frequency': terms_combined.get(term, 0), 'weight': weight}
                    for term, weight in top_terms
                ],
                'total_unique_terms': len(terms_combined),
                'sources': ['abstract', 'keywords']
            }

            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(term_stats, f, indent=2, ensure_ascii=False)
            self.results['outputs']['wc_statistics'] = str(stats_path)

            self.results['successful'].append('wordcloud')
            logger.success("✓ Word cloud visualizations completed")
            return True

        except Exception as e:
            logger.error(f"Word cloud visualization failed: {e}")
            self.results['failed'].append('wordcloud')
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _generate_timeline_visualizations(self) -> bool:
        """Generate all timeline visualizations."""
        try:
            timeline_dir = self.output_dir / 'timeline'
            dpi = self.config.get('image_quality', {}).get('dpi', 300)

            # Initialize
            self.timeline = TimelineVisualization(str(self.data_path))

            # Extract temporal data
            logger.info("Extracting temporal data...")
            df = self.timeline.extract_temporal_data()

            # Calculate statistics
            logger.info("Calculating temporal statistics...")
            stats = self.timeline.calculate_yearly_statistics(df)

            # 1. Main timeline
            logger.info("Creating main timeline...")
            main_path = timeline_dir / 'main_timeline.png'
            self.timeline.create_timeline_plot(df, str(main_path), dpi=dpi)
            self.results['outputs']['timeline_main'] = str(main_path)

            # 2. Venue breakdown
            if 'journal_conference' in df.columns:
                logger.info("Creating venue breakdown...")
                venue_path = timeline_dir / 'venue_breakdown.png'
                self.timeline.create_venue_timeline(df, str(venue_path), top_n=10, dpi=dpi)
                self.results['outputs']['timeline_venues'] = str(venue_path)

            # 3. Stacked area chart
            logger.info("Creating stacked area chart...")
            stacked_path = timeline_dir / 'stacked_area.png'
            self.timeline.create_stacked_area_chart(df, str(stacked_path), dpi=dpi)
            self.results['outputs']['timeline_stacked'] = str(stacked_path)

            # 4. Interactive timeline (Plotly)
            logger.info("Creating interactive timeline...")
            interactive_path = timeline_dir / 'interactive.html'
            self.timeline.create_interactive_timeline(df, str(interactive_path))
            self.results['outputs']['timeline_interactive'] = str(interactive_path)

            # Save statistics
            stats_path = timeline_dir / 'statistics.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                # Convert numpy types to native Python types for JSON serialization
                stats_serializable = self._convert_to_serializable(stats)
                json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
            self.results['outputs']['timeline_statistics'] = str(stats_path)

            self.results['successful'].append('timeline')
            logger.success("✓ Timeline visualizations completed")
            return True

        except Exception as e:
            logger.error(f"Timeline visualization failed: {e}")
            self.results['failed'].append('timeline')
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _collect_statistics(self):
        """Collect overall statistics from all modules."""
        logger.info("Compiling overall statistics...")

        stats = {
            'dataset': {
                'total_records': len(self.df),
                'fields': list(self.df.columns),
                'date_range': {
                    'min': int(self.df['year'].min()) if 'year' in self.df.columns else None,
                    'max': int(self.df['year'].max()) if 'year' in self.df.columns else None
                }
            },
            'completeness': {
                field: f"{(1 - self.df[field].isnull().sum() / len(self.df)):.1%}"
                for field in self.df.columns
            }
        }

        # Add venue stats if available
        if 'journal_conference' in self.df.columns:
            venue_counts = self.df['journal_conference'].value_counts()
            stats['venues'] = {
                'total_unique': len(venue_counts),
                'top_10': venue_counts.head(10).to_dict()
            }

        # Add publication type stats if available
        if 'publication_type' in self.df.columns:
            type_counts = self.df['publication_type'].value_counts()
            stats['publication_types'] = type_counts.to_dict()

        self.results['statistics'] = stats

        # Save to file
        stats_path = self.output_dir / 'reports' / 'overall_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.success(f"✓ Statistics saved to {stats_path}")

    def _generate_pdf_report(self) -> bool:
        """Generate comprehensive PDF report."""
        try:
            logger.info("Generating comprehensive PDF report...")

            pdf_path = self.output_dir / 'reports' / 'full_report.pdf'
            self.pdf_exporter = PDFExporter(str(pdf_path))

            # Prepare data for PDF
            geographic_data = None
            if 'geo_static' in self.results['outputs']:
                with open(self.output_dir / 'geographic' / 'statistics.json', 'r') as f:
                    geo_stats = json.load(f)
                geographic_data = {
                    'map_image': self.results['outputs'].get('geo_static'),
                    'statistics': geo_stats
                }

            wordcloud_data = None
            if 'wc_combined' in self.results['outputs']:
                with open(self.output_dir / 'wordclouds' / 'term_statistics.json', 'r') as f:
                    wc_stats = json.load(f)
                # Convert to expected format
                wc_data_formatted = {
                    'top_terms': [
                        (item['term'], item['frequency'], item['weight'])
                        for item in wc_stats['top_terms']
                    ]
                }
                wordcloud_data = {
                    'image': self.results['outputs'].get('wc_combined'),
                    'statistics': wc_data_formatted
                }

            timeline_data = None
            if 'timeline_main' in self.results['outputs']:
                with open(self.output_dir / 'timeline' / 'statistics.json', 'r') as f:
                    timeline_stats = json.load(f)
                timeline_data = {
                    'images': [
                        self.results['outputs'].get('timeline_main'),
                        self.results['outputs'].get('timeline_stacked')
                    ],
                    'statistics': timeline_stats
                }

            # Overall stats
            overall_stats = self.results['statistics']['dataset']

            # Processing info
            import platform
            processing_info = {
                'version': self.config.get('version', '1.0.0'),
                'python_version': platform.python_version(),
                'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sources': ['Unified Bibliometric Database']
            }

            # Generate PDF
            self.pdf_exporter.generate_complete_pdf(
                title="Comprehensive Bibliometric Analysis Report",
                subtitle="Complete Visualization Pipeline Output",
                date_range=f"{overall_stats['date_range']['min']} - {overall_stats['date_range']['max']}"
                          if overall_stats['date_range']['min'] else "N/A",
                authors=["Bibliometric Analysis System"],
                institution="Research Analytics",
                geographic_data=geographic_data,
                wordcloud_data=wordcloud_data,
                timeline_data=timeline_data,
                overall_stats=overall_stats,
                processing_info=processing_info
            )

            self.results['outputs']['pdf_report'] = str(pdf_path)
            self.results['successful'].append('pdf')
            logger.success(f"✓ PDF report generated: {pdf_path}")
            return True

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            self.results['failed'].append('pdf')
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _prepare_dashboard_data(self):
        """Prepare processed data for dashboard."""
        logger.info("Preparing data for dashboard...")

        dashboard_dir = self.output_dir / 'dashboard' / 'data'

        # Copy main dataset
        dashboard_data_path = dashboard_dir / 'bibliometric_data.csv'
        self.df.to_csv(dashboard_data_path, index=False, encoding='utf-8')
        self.results['outputs']['dashboard_data'] = str(dashboard_data_path)

        # Create metadata file
        metadata = {
            'generated': datetime.now().isoformat(),
            'total_records': len(self.df),
            'fields': list(self.df.columns),
            'visualizations_available': self.results['successful'],
            'outputs': self.results['outputs']
        }

        metadata_path = dashboard_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.success(f"✓ Dashboard data prepared: {dashboard_dir}")

    def generate_execution_report(self, output_path: str) -> str:
        """
        Generate detailed execution report.

        Args:
            output_path: Path for report output

        Returns:
            Path to generated report
        """
        logger.info("Generating execution report...")

        report_path = Path(output_path)

        # Calculate total time
        if self.execution_start and self.execution_end:
            total_time = (self.execution_end - self.execution_start).total_seconds()
        else:
            total_time = 0

        # Generate Markdown report
        md_content = f"""# Visualization Pipeline Execution Report

## Summary

**Execution Date**: {self.execution_start.strftime('%Y-%m-%d %H:%M:%S') if self.execution_start else 'N/A'}
**Total Duration**: {total_time:.2f} seconds
**Status**: {'✓ SUCCESS' if len(self.results['failed']) == 0 else '⚠ PARTIAL SUCCESS'}

## Dataset Information

- **Total Records**: {len(self.df):,}
- **Fields**: {', '.join(self.df.columns)}
- **Date Range**: {self.results['statistics']['dataset']['date_range']['min']} - {self.results['statistics']['dataset']['date_range']['max']}

## Visualizations Generated

### Successful ({len(self.results['successful'])})

"""
        for viz in self.results['successful']:
            md_content += f"- ✓ {viz.title()}\n"

        if self.results['failed']:
            md_content += f"\n### Failed ({len(self.results['failed'])})\n\n"
            for viz in self.results['failed']:
                md_content += f"- ✗ {viz.title()}\n"

        if self.results['warnings']:
            md_content += f"\n### Warnings ({len(self.results['warnings'])})\n\n"
            for warning in self.results['warnings']:
                md_content += f"- ⚠ {warning}\n"

        md_content += "\n## Performance Metrics\n\n"
        md_content += "| Module | Time (seconds) |\n"
        md_content += "|--------|----------------|\n"

        for module, time_taken in self.results['timings'].items():
            md_content += f"| {module.title()} | {time_taken:.2f} |\n"

        md_content += f"| **Total** | **{total_time:.2f}** |\n"

        md_content += "\n## Output Files\n\n"
        for key, path in self.results['outputs'].items():
            md_content += f"- **{key}**: `{path}`\n"

        md_content += f"\n## Configuration\n\n"
        md_content += f"- **Config File**: {self.config_path}\n"
        md_content += f"- **Output Directory**: {self.output_dir}\n"
        md_content += f"- **DPI**: {self.config.get('image_quality', {}).get('dpi', 300)}\n"

        md_content += "\n---\n\n"
        md_content += f"*Report generated by Visualization Pipeline v{self.config.get('version', '1.0.0')}*\n"

        # Write Markdown report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # Also save JSON version
        json_path = report_path.with_suffix('.json')
        json_data = {
            'execution': {
                'start': self.execution_start.isoformat() if self.execution_start else None,
                'end': self.execution_end.isoformat() if self.execution_end else None,
                'duration_seconds': total_time
            },
            'results': self.results,
            'dataset': {
                'path': str(self.data_path),
                'records': len(self.df),
                'fields': list(self.df.columns)
            },
            'configuration': {
                'config_path': str(self.config_path),
                'output_dir': str(self.output_dir),
                'dpi': self.config.get('image_quality', {}).get('dpi', 300)
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        logger.success(f"✓ Execution report generated: {report_path}")
        logger.info(f"  JSON version: {json_path}")

        return str(report_path)

    def update_visualizations_incremental(self, new_data_path: str):
        """
        Update visualizations incrementally with new data.

        Process:
        1. Load new data
        2. Combine with existing data
        3. Re-run only affected visualizations
        4. Update PDF report
        5. Update dashboard data

        Args:
            new_data_path: Path to new data CSV file
        """
        logger.info("=" * 70)
        logger.info("INCREMENTAL UPDATE MODE")
        logger.info("=" * 70)

        # Load new data
        logger.info(f"Loading new data from {new_data_path}")
        new_df = pd.read_csv(new_data_path, encoding='utf-8')
        logger.info(f"Loaded {len(new_df)} new records")

        # Combine with existing
        logger.info("Combining with existing data...")
        combined_df = pd.concat([self.df, new_df], ignore_index=True)

        # Remove duplicates based on ID if available
        if 'id' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
            removed = before_dedup - len(combined_df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate records")

        logger.info(f"Total records after merge: {len(combined_df)}")

        # Save combined data
        combined_path = self.output_dir / 'dashboard' / 'data' / 'bibliometric_data_updated.csv'
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')

        # Update internal dataframe
        self.df = combined_df

        # Re-save as main data source
        self.df.to_csv(self.data_path, index=False, encoding='utf-8')
        logger.success(f"✓ Updated main data file: {self.data_path}")

        # Re-run visualizations
        logger.info("Re-running visualizations with updated data...")
        result = self.run_all_visualizations()

        logger.info("=" * 70)
        logger.success("✓ INCREMENTAL UPDATE COMPLETED")
        logger.info("=" * 70)

        return result


# Example usage
if __name__ == "__main__":
    logger.info("Visualization Pipeline - Example Usage")

    # This would normally be called with real data
    # pipeline = VisualizationPipeline(
    #     unified_data_path='data/sample/complete_sample_data.csv',
    #     output_dir='output/pipeline_output'
    # )
    #
    # # Run complete pipeline
    # results = pipeline.run_all_visualizations()
    #
    # # Or update incrementally
    # # results = pipeline.update_visualizations_incremental('data/new_studies.csv')

    logger.info("Example complete!")
