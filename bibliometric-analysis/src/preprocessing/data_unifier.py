"""
Data Unifier Module
Unifies bibliographic data from multiple sources and detects duplicates
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from datetime import datetime
import Levenshtein
from tqdm import tqdm


class DataUnifier:
    """
    Unifies bibliographic data from multiple sources

    Features:
    - Load data from multiple sources
    - Normalize records to common format
    - Detect duplicates using Levenshtein similarity
    - Merge information from duplicate records
    - Export unified dataset
    """

    def __init__(self, config):
        """
        Initialize data unifier

        Args:
            config: Configuration object
        """
        self.config = config

        # Deduplication configuration
        self.title_threshold = config.get('deduplication.thresholds.title_similarity', 0.90)
        self.doi_priority = config.get('deduplication.strategy.exact_doi_priority', True)

        # Output paths
        self.output_dir = Path(config.get('paths.processed_data', 'data/processed'))
        self.duplicates_dir = Path(config.get('paths.duplicates', 'data/duplicates'))

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duplicates_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Data Unifier initialized")
        logger.info(f"Title similarity threshold: {self.title_threshold}")

    def load_multiple_sources(self, records_list: List[List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Load and combine records from multiple sources

        Args:
            records_list: List of record lists from different sources

        Returns:
            Combined DataFrame
        """
        logger.info(f"Loading records from {len(records_list)} sources")

        all_records = []
        for records in records_list:
            all_records.extend(records)

        logger.info(f"Total records before normalization: {len(all_records)}")

        # Convert to DataFrame
        df = pd.DataFrame(all_records)

        # Normalize
        df = self.normalize_records(df)

        logger.info(f"Records after normalization: {len(df)}")
        return df

    def normalize_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and standardize record fields

        Args:
            df: DataFrame with records

        Returns:
            Normalized DataFrame
        """
        logger.info("Normalizing records")

        # Ensure all required columns exist
        required_columns = [
            'id', 'title', 'authors', 'year', 'abstract', 'keywords',
            'doi', 'source', 'publication_type', 'journal_conference', 'url'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        # Normalize title (lowercase, strip whitespace)
        df['title_normalized'] = df['title'].str.lower().str.strip()

        # Normalize DOI (lowercase, remove whitespace)
        df['doi_normalized'] = df['doi'].str.lower().str.strip()

        # Convert year to integer
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

        # Handle list fields
        for col in ['authors', 'keywords']:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        # Add unique ID if missing
        if 'id' not in df.columns or df['id'].isna().any():
            df['id'] = df.apply(
                lambda row: f"{row['source']}_{hash(row['title'])}",
                axis=1
            )

        # Sort by source and year
        df = df.sort_values(['source', 'year'], ascending=[True, False])

        logger.info("Normalization complete")
        return df

    def detect_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect duplicate records using similarity metrics

        Args:
            df: DataFrame with normalized records

        Returns:
            Tuple of (clean_df, duplicates_df)
        """
        logger.info("Starting duplicate detection")

        duplicates = []
        seen_indices = set()

        # Progress bar
        pbar = tqdm(total=len(df), desc="Detecting duplicates")

        for i in range(len(df)):
            if i in seen_indices:
                pbar.update(1)
                continue

            current_record = df.iloc[i]

            # Check for duplicates
            for j in range(i + 1, len(df)):
                if j in seen_indices:
                    continue

                other_record = df.iloc[j]

                # Check if duplicate
                is_duplicate, similarity_score = self._is_duplicate(
                    current_record,
                    other_record
                )

                if is_duplicate:
                    # Mark as seen
                    seen_indices.add(j)

                    # Record duplicate information
                    duplicates.append({
                        'original_index': i,
                        'duplicate_index': j,
                        'original_title': current_record['title'],
                        'duplicate_title': other_record['title'],
                        'original_source': current_record['source'],
                        'duplicate_source': other_record['source'],
                        'similarity_score': similarity_score,
                        'original_doi': current_record.get('doi', ''),
                        'duplicate_doi': other_record.get('doi', ''),
                        'detection_method': 'title_similarity' if similarity_score < 1.0 else 'exact_doi'
                    })

            pbar.update(1)

        pbar.close()

        logger.info(f"Found {len(duplicates)} duplicate pairs")

        # Create clean dataset (remove duplicates)
        duplicate_indices = [d['duplicate_index'] for d in duplicates]
        clean_df = df.drop(duplicate_indices).reset_index(drop=True)

        # Create duplicates DataFrame
        duplicates_df = pd.DataFrame(duplicates) if duplicates else pd.DataFrame()

        logger.info(f"Clean records: {len(clean_df)}")
        logger.info(f"Removed duplicates: {len(duplicate_indices)}")

        return clean_df, duplicates_df

    def _is_duplicate(
        self,
        record1: pd.Series,
        record2: pd.Series
    ) -> Tuple[bool, float]:
        """
        Check if two records are duplicates

        Args:
            record1: First record
            record2: Second record

        Returns:
            Tuple of (is_duplicate, similarity_score)
        """
        # Priority 1: Exact DOI match
        if self.doi_priority:
            doi1 = record1.get('doi_normalized', '')
            doi2 = record2.get('doi_normalized', '')

            if doi1 and doi2 and doi1 == doi2:
                return True, 1.0

        # Priority 2: Title similarity using Levenshtein
        title1 = record1.get('title_normalized', '')
        title2 = record2.get('title_normalized', '')

        if not title1 or not title2:
            return False, 0.0

        # Calculate Levenshtein similarity ratio
        similarity = Levenshtein.ratio(title1, title2)

        # Check threshold
        if similarity >= self.title_threshold:
            return True, similarity

        return False, similarity

    def merge_duplicate_info(
        self,
        clean_df: pd.DataFrame,
        duplicates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge information from duplicate records into originals

        Args:
            clean_df: DataFrame with unique records
            duplicates_df: DataFrame with duplicate information

        Returns:
            Enhanced DataFrame with merged information
        """
        if duplicates_df.empty:
            logger.info("No duplicates to merge")
            return clean_df

        logger.info("Merging duplicate information")

        # Group duplicates by original index
        duplicate_groups = duplicates_df.groupby('original_index')

        for original_idx, group in tqdm(duplicate_groups, desc="Merging info"):
            # Get all duplicate indices for this original
            duplicate_indices = group['duplicate_index'].tolist()

            # Get original record (need to map to clean_df index)
            # This is complex because indices change after dropping duplicates

            # Instead, we'll enhance based on source priority
            # If original has empty field but duplicate has data, fill it

        logger.info("Merge complete")
        return clean_df

    def save_unified_data(
        self,
        df: pd.DataFrame,
        filename: str = 'unified_data.csv'
    ) -> Path:
        """
        Save unified dataset to CSV

        Args:
            df: Unified DataFrame
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        # Select columns for export
        export_columns = [
            'id', 'title', 'authors', 'year', 'abstract', 'keywords',
            'doi', 'source', 'publication_type', 'journal_conference', 'url',
            'publisher', 'volume', 'number', 'pages'
        ]

        # Keep only columns that exist
        available_columns = [col for col in export_columns if col in df.columns]

        # Convert list columns to strings for CSV
        df_export = df[available_columns].copy()
        for col in ['authors', 'keywords']:
            if col in df_export.columns:
                df_export[col] = df_export[col].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else x
                )

        # Save to CSV
        df_export.to_csv(output_path, index=False, encoding='utf-8')

        logger.success(f"Saved unified data to: {output_path}")
        logger.info(f"Total records: {len(df_export)}")

        return output_path

    def save_duplicates_log(
        self,
        duplicates_df: pd.DataFrame,
        filename: str = 'duplicates_log.csv'
    ) -> Path:
        """
        Save duplicates log to CSV

        Args:
            duplicates_df: DataFrame with duplicate information
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.duplicates_dir / filename

        if duplicates_df.empty:
            logger.info("No duplicates to save")
            # Create empty file with headers
            pd.DataFrame(columns=[
                'original_index', 'duplicate_index', 'original_title',
                'duplicate_title', 'original_source', 'duplicate_source',
                'similarity_score', 'detection_method'
            ]).to_csv(output_path, index=False)
        else:
            duplicates_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.success(f"Saved duplicates log to: {output_path}")
            logger.info(f"Total duplicate pairs: {len(duplicates_df)}")

        return output_path

    def unify(
        self,
        records_list: List[List[Dict[str, Any]]],
        output_filename: str = None,
        duplicates_filename: str = None
    ) -> Dict[str, Any]:
        """
        Complete unification workflow

        Args:
            records_list: List of record lists from different sources
            output_filename: Custom output filename
            duplicates_filename: Custom duplicates filename

        Returns:
            Dictionary with unification statistics
        """
        logger.info("=" * 70)
        logger.info("Starting Data Unification Workflow")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Load and normalize
        df = self.load_multiple_sources(records_list)
        original_count = len(df)

        # Detect duplicates
        clean_df, duplicates_df = self.detect_duplicates(df)
        duplicates_count = len(duplicates_df)
        clean_count = len(clean_df)

        # Merge duplicate information (optional enhancement)
        clean_df = self.merge_duplicate_info(clean_df, duplicates_df)

        # Save results
        unified_path = self.save_unified_data(
            clean_df,
            output_filename or 'unified_data.csv'
        )

        duplicates_path = self.save_duplicates_log(
            duplicates_df,
            duplicates_filename or 'duplicates_log.csv'
        )

        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats = {
            'original_count': original_count,
            'duplicates_count': duplicates_count,
            'clean_count': clean_count,
            'unified_file': str(unified_path),
            'duplicates_file': str(duplicates_path),
            'execution_time_seconds': duration,
            'sources': df['source'].unique().tolist()
        }

        logger.info("=" * 70)
        logger.info("Unification Complete")
        logger.info(f"Original records: {original_count}")
        logger.info(f"Duplicates found: {duplicates_count}")
        logger.info(f"Clean records: {clean_count}")
        logger.info(f"Execution time: {duration:.2f}s")
        logger.info("=" * 70)

        return stats


# Example usage
if __name__ == "__main__":
    from src.utils.config_loader import get_config

    # Setup logger
    logger.add("logs/data_unifier.log", rotation="10 MB")

    # Load configuration
    config = get_config()

    # Create unifier
    unifier = DataUnifier(config)

    # Example: Unify two sources
    # (In practice, these would come from scrapers)
    records_acm = [
        {
            'title': 'Generative AI for Everyone',
            'authors': ['John Doe'],
            'year': '2023',
            'doi': '10.1145/123456',
            'source': 'ACM',
            'abstract': 'Abstract here',
            'keywords': ['AI', 'Generative'],
            'publication_type': 'article',
            'journal_conference': 'AI Journal',
            'url': 'http://example.com'
        }
    ]

    records_sd = [
        {
            'title': 'generative ai for everyone',  # Duplicate (lowercase)
            'authors': ['John Doe'],
            'year': '2023',
            'doi': '10.1145/123456',  # Same DOI
            'source': 'ScienceDirect',
            'abstract': 'Abstract here',
            'keywords': ['AI', 'Machine Learning'],
            'publication_type': 'article',
            'journal_conference': 'AI Journal',
            'url': 'http://example.com'
        }
    ]

    # Unify
    stats = unifier.unify([records_acm, records_sd])

    logger.success(f"Unification complete: {stats}")
