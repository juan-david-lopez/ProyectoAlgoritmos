"""
Logging utility module
Provides centralized logging configuration for the application
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Colors.RESET}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return result


def setup_logger(
    name: Optional[str] = None,
    level: str = 'INFO',
    config: Optional[object] = None
) -> logging.Logger:
    """
    Setup and configure logger

    Args:
        name: Logger name (None for root logger)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Configuration object

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Get configuration
    if config:
        log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        date_format = config.get('logging.date_format', '%Y-%m-%d %H:%M:%S')
        console_enabled = config.get('logging.console.enabled', True)
        console_level = config.get('logging.console.level', 'INFO')
        file_enabled = config.get('logging.files.main') is not None
        colored = config.get('logging.console.colored', True)
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        console_enabled = True
        console_level = 'INFO'
        file_enabled = False
        colored = True

    # Console handler with UTF-8 encoding support
    if console_enabled:
        # Create a UTF-8 stream wrapper for stdout to handle Unicode characters
        import io
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace',  # Replace characters that can't be encoded
            line_buffering=True
        )
        
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))

        if colored and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_enabled and config:
        log_dir = Path(config.get('paths.logs', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / config.get('logging.files.main', 'bibliometric_analysis.log')

        # Rotating file handler
        if config.get('logging.rotation.enabled', True):
            max_bytes = config.get('logging.rotation.max_bytes', 10485760)  # 10 MB
            backup_count = config.get('logging.rotation.backup_count', 5)

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')

        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def create_module_logger(module_name: str, config: Optional[object] = None) -> logging.Logger:
    """
    Create a logger for a specific module with optional dedicated log file

    Args:
        module_name: Name of the module (e.g., 'scraper', 'clustering')
        config: Configuration object

    Returns:
        Logger instance
    """
    logger = logging.getLogger(module_name)

    # If module-specific log file is configured
    if config and config.get(f'logging.files.{module_name}'):
        log_dir = Path(config.get('paths.logs', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / config.get(f'logging.files.{module_name}')

        # Create dedicated file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.get('logging.rotation.max_bytes', 10485760),
            backupCount=config.get('logging.rotation.backup_count', 5),
            encoding='utf-8'
        )

        log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        date_format = config.get('logging.date_format', '%Y-%m-%d %H:%M:%S')

        formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


class LoggerContext:
    """Context manager for temporary logger configuration"""

    def __init__(self, logger: logging.Logger, level: Optional[int] = None):
        self.logger = logger
        self.original_level = logger.level
        self.new_level = level

    def __enter__(self):
        if self.new_level:
            self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


# Example usage
if __name__ == "__main__":
    # Setup basic logger
    logger = setup_logger(level='DEBUG')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Get module logger
    module_logger = get_logger('test_module')
    module_logger.info("Message from test module")

    # Use context manager
    with LoggerContext(logger, logging.ERROR):
        logger.info("This won't be shown")
        logger.error("This will be shown")

    logger.info("Back to normal level")
