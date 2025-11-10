"""
File Handler utility module
Provides file I/O operations for the application
"""

import os
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import shutil


class FileHandler:
    """
    Centralized file handling for the bibliometric analysis project

    Handles:
    - Reading/writing CSV files
    - Reading/writing JSON files
    - Reading/writing Excel files
    - Directory management
    - File validation
    """

    def __init__(self, config: Optional[object] = None):
        """
        Initialize file handler

        Args:
            config: Configuration object
        """
        self.config = config

        # Get paths from config
        if config:
            self.raw_data_path = Path(config.get('paths.raw_data', 'data/raw'))
            self.processed_data_path = Path(config.get('paths.processed_data', 'data/processed'))
            self.duplicates_path = Path(config.get('paths.duplicates', 'data/duplicates'))
            self.outputs_path = Path(config.get('paths.outputs', 'outputs'))
            self.logs_path = Path(config.get('paths.logs', 'logs'))
        else:
            self.raw_data_path = Path('data/raw')
            self.processed_data_path = Path('data/processed')
            self.duplicates_path = Path('data/duplicates')
            self.outputs_path = Path('outputs')
            self.logs_path = Path('logs')

        # Create directories if they don't exist
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        for path in [
            self.raw_data_path,
            self.processed_data_path,
            self.duplicates_path,
            self.outputs_path,
            self.logs_path
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # CSV operations
    def read_csv(
        self,
        filename: Union[str, Path],
        directory: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read CSV file into pandas DataFrame

        Args:
            filename: Name of the CSV file
            directory: Directory containing the file (default: processed_data_path)
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with CSV contents
        """
        if directory is None:
            directory = self.processed_data_path

        filepath = Path(directory) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        return pd.read_csv(filepath, **kwargs)

    def write_csv(
        self,
        data: pd.DataFrame,
        filename: Union[str, Path],
        directory: Optional[Union[str, Path]] = None,
        add_timestamp: bool = False,
        **kwargs
    ) -> Path:
        """
        Write DataFrame to CSV file

        Args:
            data: DataFrame to write
            filename: Name of the CSV file
            directory: Directory to write to (default: processed_data_path)
            add_timestamp: Add timestamp to filename
            **kwargs: Additional arguments for pd.to_csv

        Returns:
            Path to the written file
        """
        if directory is None:
            directory = self.processed_data_path

        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = Path(filename).stem
            ext = Path(filename).suffix
            filename = f"{name}_{timestamp}{ext}"

        filepath = Path(directory) / filename

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        data.to_csv(filepath, index=False, **kwargs)

        return filepath

    # JSON operations
    def read_json(
        self,
        filename: Union[str, Path],
        directory: Optional[Union[str, Path]] = None
    ) -> Union[Dict, List]:
        """
        Read JSON file

        Args:
            filename: Name of the JSON file
            directory: Directory containing the file

        Returns:
            Parsed JSON data
        """
        if directory is None:
            directory = self.processed_data_path

        filepath = Path(directory) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def write_json(
        self,
        data: Union[Dict, List],
        filename: Union[str, Path],
        directory: Optional[Union[str, Path]] = None,
        add_timestamp: bool = False,
        indent: int = 2
    ) -> Path:
        """
        Write data to JSON file

        Args:
            data: Data to write
            filename: Name of the JSON file
            directory: Directory to write to
            add_timestamp: Add timestamp to filename
            indent: JSON indentation

        Returns:
            Path to the written file
        """
        if directory is None:
            directory = self.processed_data_path

        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = Path(filename).stem
            ext = Path(filename).suffix
            filename = f"{name}_{timestamp}{ext}"

        filepath = Path(directory) / filename

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return filepath

    # Excel operations
    def read_excel(
        self,
        filename: Union[str, Path],
        directory: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read Excel file into pandas DataFrame

        Args:
            filename: Name of the Excel file
            directory: Directory containing the file
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            DataFrame with Excel contents
        """
        if directory is None:
            directory = self.raw_data_path

        filepath = Path(directory) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Excel file not found: {filepath}")

        return pd.read_excel(filepath, **kwargs)

    def write_excel(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: Union[str, Path],
        directory: Optional[Union[str, Path]] = None,
        add_timestamp: bool = False,
        **kwargs
    ) -> Path:
        """
        Write DataFrame(s) to Excel file

        Args:
            data: DataFrame or dict of DataFrames (for multiple sheets)
            filename: Name of the Excel file
            directory: Directory to write to
            add_timestamp: Add timestamp to filename
            **kwargs: Additional arguments for pd.to_excel

        Returns:
            Path to the written file
        """
        if directory is None:
            directory = self.outputs_path

        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = Path(filename).stem
            ext = Path(filename).suffix
            filename = f"{name}_{timestamp}{ext}"

        filepath = Path(directory) / filename

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, dict):
            # Multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)
        else:
            # Single sheet
            data.to_excel(filepath, index=False, **kwargs)

        return filepath

    # File utilities
    def list_files(
        self,
        directory: Union[str, Path],
        pattern: str = '*',
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in a directory

        Args:
            directory: Directory to search
            pattern: Glob pattern (e.g., '*.csv')
            recursive: Search recursively

        Returns:
            List of file paths
        """
        directory = Path(directory)

        if not directory.exists():
            return []

        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))

    def file_exists(self, filepath: Union[str, Path]) -> bool:
        """Check if file exists"""
        return Path(filepath).exists()

    def delete_file(self, filepath: Union[str, Path]) -> bool:
        """
        Delete a file

        Args:
            filepath: Path to file

        Returns:
            True if deleted successfully
        """
        filepath = Path(filepath)

        if filepath.exists():
            filepath.unlink()
            return True

        return False

    def copy_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path]
    ) -> Path:
        """
        Copy a file

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Path to the copied file
        """
        source = Path(source)
        destination = Path(destination)

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, destination)

        return destination

    def move_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path]
    ) -> Path:
        """
        Move a file

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Path to the moved file
        """
        source = Path(source)
        destination = Path(destination)

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(source), str(destination))

        return destination

    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        stat = filepath.stat()

        return {
            'name': filepath.name,
            'path': str(filepath.absolute()),
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': filepath.is_file(),
            'is_dir': filepath.is_dir(),
            'extension': filepath.suffix
        }

    def combine_csv_files(
        self,
        directory: Union[str, Path],
        pattern: str = '*.csv',
        output_filename: str = 'combined.csv',
        output_directory: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Combine multiple CSV files into one

        Args:
            directory: Directory containing CSV files
            pattern: Glob pattern for CSV files
            output_filename: Name of combined file
            output_directory: Directory for output file

        Returns:
            Path to combined file
        """
        csv_files = self.list_files(directory, pattern)

        if not csv_files:
            raise ValueError(f"No CSV files found in {directory} matching pattern {pattern}")

        # Read and combine all CSVs
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['source_file'] = csv_file.name  # Add source file column
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Write combined file
        return self.write_csv(
            combined_df,
            output_filename,
            directory=output_directory or self.processed_data_path
        )


# Example usage
if __name__ == "__main__":
    from src.utils.config_loader import get_config

    # Load config
    config = get_config()

    # Initialize file handler
    fh = FileHandler(config)

    # Example: List CSV files
    csv_files = fh.list_files(fh.raw_data_path, '*.csv')
    print(f"Found {len(csv_files)} CSV files")

    # Example: File info
    if csv_files:
        info = fh.get_file_info(csv_files[0])
        print(f"File info: {json.dumps(info, indent=2, default=str)}")
