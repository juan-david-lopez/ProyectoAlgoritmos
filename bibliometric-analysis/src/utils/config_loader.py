"""
Configuration Loader Utility
Loads configuration from YAML files and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """
    Configuration loader for the bibliometric analysis project

    Features:
    - Load YAML configuration files
    - Load environment variables from .env
    - Support for nested configuration
    - Environment variable interpolation
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration loader

        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.env_loaded = False

        # Load environment variables
        self._load_env()

        # Load YAML configuration
        if self.config_path.exists():
            self._load_yaml()
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def _load_env(self):
        """Load environment variables from .env file"""
        env_paths = [
            Path("config/.env"),
            Path(".env"),
            Path("../.env")
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                self.env_loaded = True
                break

    def _load_yaml(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation

        Args:
            key: Configuration key in dot notation (e.g., 'sources.ieee.enabled')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = ConfigLoader()
            >>> config.get('sources.ieee.enabled')
            True
            >>> config.get('sources.ieee.max_results')
            1000
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a specific service

        Args:
            service: Service name (e.g., 'scopus', 'wos', 'ieee')

        Returns:
            API key or None
        """
        env_var_map = {
            'scopus': 'SCOPUS_API_KEY',
            'wos': 'WOS_API_KEY',
            'web_of_science': 'WOS_API_KEY',
            'ieee': 'IEEE_API_KEY'
        }

        env_var = env_var_map.get(service.lower())
        if env_var:
            return self.get_env(env_var)

        return None

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration from environment variables

        Returns:
            Dictionary with database configuration
        """
        return {
            'type': self.get_env('DB_TYPE', 'sqlite'),
            'host': self.get_env('DB_HOST', 'localhost'),
            'port': int(self.get_env('DB_PORT', '5432')),
            'name': self.get_env('DB_NAME', 'bibliometric_db'),
            'user': self.get_env('DB_USER', ''),
            'password': self.get_env('DB_PASSWORD', ''),
            'path': self.get_env('DB_PATH', 'data/bibliometric.db')
        }

    def get_paths(self) -> Dict[str, Path]:
        """
        Get all configured paths as Path objects

        Returns:
            Dictionary of paths
        """
        paths_config = self.get('paths', {})
        return {
            key: Path(value)
            for key, value in paths_config.items()
        }

    def get_source_config(self, source: str) -> Dict[str, Any]:
        """
        Get configuration for a specific data source

        Args:
            source: Source name (e.g., 'ieee', 'scopus', 'web_of_science')

        Returns:
            Source configuration dictionary
        """
        config = self.get(f'sources.{source.lower()}', {})

        # Add API key if available
        api_key = self.get_api_key(source)
        if api_key:
            if 'api' not in config:
                config['api'] = {}
            config['api']['key'] = api_key

        return config

    def get_clustering_config(self, algorithm: str = None) -> Dict[str, Any]:
        """
        Get clustering configuration

        Args:
            algorithm: Specific algorithm name (e.g., 'kmeans', 'dbscan', 'hierarchical')

        Returns:
            Clustering configuration
        """
        if algorithm:
            return self.get(f'clustering.algorithms.{algorithm}', {})
        return self.get('clustering', {})

    def get_visualization_config(self, chart_type: str = None) -> Dict[str, Any]:
        """
        Get visualization configuration

        Args:
            chart_type: Specific chart type (e.g., 'temporal_trends', 'country_distribution')

        Returns:
            Visualization configuration
        """
        if chart_type:
            return self.get(f'visualization.charts.{chart_type}', {})
        return self.get('visualization', {})

    def is_source_enabled(self, source: str) -> bool:
        """
        Check if a data source is enabled

        Args:
            source: Source name

        Returns:
            True if enabled, False otherwise
        """
        return self.get(f'sources.{source}.enabled', False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation"""
        return f"ConfigLoader(config_path='{self.config_path}', env_loaded={self.env_loaded})"

    def __str__(self) -> str:
        """String representation"""
        return f"Configuration loaded from {self.config_path}"


# Singleton instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get or create the global configuration instance

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)

    return _config_instance


def reload_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Reload configuration (force recreation of singleton)

    Args:
        config_path: Path to configuration file

    Returns:
        New ConfigLoader instance
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = get_config()

    # Access configuration values
    print(f"Project name: {config.get('project.name')}")
    print(f"Query keywords: {config.get('query.keywords')}")

    # Check sources
    print(f"\nEnabled sources:")
    for source in ['ieee', 'scopus', 'web_of_science']:
        enabled = config.is_source_enabled(source)
        print(f"  {source}: {enabled}")

    # Get API keys
    print(f"\nAPI Keys:")
    for service in ['scopus', 'wos', 'ieee']:
        api_key = config.get_api_key(service)
        if api_key:
            print(f"  {service}: {api_key[:10]}...")
        else:
            print(f"  {service}: Not configured")

    # Get paths
    print(f"\nConfigured paths:")
    paths = config.get_paths()
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Get clustering config
    print(f"\nK-Means configuration:")
    kmeans_config = config.get_clustering_config('kmeans')
    print(f"  Enabled: {kmeans_config.get('enabled')}")
    print(f"  Clusters: {kmeans_config.get('n_clusters')}")
    print(f"  Max iterations: {kmeans_config.get('max_iter')}")
