"""
Base Scraper Module
Abstract base class for academic database scrapers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from loguru import logger
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Import new WebDriverManager
from src.utils.webdriver_manager import WebDriverManager as WDManager


class BaseScraper(ABC):
    """
    Abstract base class for academic database scrapers

    Provides common functionality for:
    - Browser automation with Selenium
    - Authentication handling
    - Error handling and retries
    - File download management
    - Progress tracking
    """

    def __init__(self, config, headless: bool = True):
        """
        Initialize base scraper

        Args:
            config: Configuration object
            headless: Run browser in headless mode
        """
        self.config = config
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.webdriver_manager: Optional[WDManager] = None

        # Retry configuration
        self.max_retries = config.get('scraping.retry.max_attempts', 3)
        self.backoff_factor = config.get('scraping.retry.backoff_factor', 2)

        # Timeout configuration
        self.page_load_timeout = config.get('scraping.timeouts.page_load', 30)
        self.element_wait_timeout = config.get('scraping.timeouts.element_wait', 10)

        # Download directory
        self.download_dir = Path(config.get('paths.raw_data', 'data/raw'))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {self.__class__.__name__}")

    def _setup_driver(self) -> webdriver.Chrome:
        """
        Setup Chrome WebDriver using WebDriverManager with anti-detection

        Returns:
            Configured Chrome WebDriver instance
        """
        logger.debug("Setting up Chrome WebDriver with WebDriverManager")

        try:
            # Initialize WebDriverManager with configuration
            self.webdriver_manager = WDManager(
                headless=self.headless,
                download_dir=str(self.download_dir.absolute())
            )
            
            # Create driver with all anti-detection features
            driver = self.webdriver_manager.create_driver()
            
            # Set page load timeout
            driver.set_page_load_timeout(self.page_load_timeout)

            logger.info("Chrome WebDriver setup complete with WebDriverManager and anti-detection")
            return driver

        except Exception as e:
            logger.error(f"Failed to setup Chrome WebDriver with WebDriverManager: {e}")
            raise

    def start_session(self):
        """Start browser session"""
        if self.driver is None:
            self.driver = self._setup_driver()
            self.wait = WebDriverWait(self.driver, self.element_wait_timeout)
            logger.info("Browser session started")

    def close_session(self):
        """Close browser session and cleanup WebDriverManager"""
        if self.driver:
            try:
                # Close WebDriverManager (handles driver.quit() internally)
                if hasattr(self, 'webdriver_manager') and self.webdriver_manager:
                    self.webdriver_manager.close()
                else:
                    # Fallback if manager not available
                    self.driver.quit()
                logger.info("Browser session closed")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            finally:
                self.driver = None
                self.wait = None
                if hasattr(self, 'webdriver_manager'):
                    self.webdriver_manager = None

    def retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} for {func.__name__}")
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")

        raise last_exception

    def safe_find_element(self, by: By, value: str, timeout: Optional[int] = None):
        """
        Safely find element with wait

        Args:
            by: Selenium By locator
            value: Locator value
            timeout: Custom timeout (uses default if None)

        Returns:
            WebElement or None
        """
        try:
            if timeout:
                wait = WebDriverWait(self.driver, timeout)
                return wait.until(EC.presence_of_element_located((by, value)))
            else:
                return self.wait.until(EC.presence_of_element_located((by, value)))
        except TimeoutException:
            logger.warning(f"Element not found: {by}={value}")
            return None

    def safe_click(self, by: By, value: str, timeout: Optional[int] = None):
        """
        Safely click element with wait

        Args:
            by: Selenium By locator
            value: Locator value
            timeout: Custom timeout

        Returns:
            True if clicked successfully
        """
        try:
            if timeout:
                wait = WebDriverWait(self.driver, timeout)
                element = wait.until(EC.element_to_be_clickable((by, value)))
            else:
                element = self.wait.until(EC.element_to_be_clickable((by, value)))

            element.click()
            logger.debug(f"Clicked element: {by}={value}")
            return True

        except Exception as e:
            logger.warning(f"Failed to click element {by}={value}: {e}")
            return False

    def wait_for_download(self, filename_pattern: str, timeout: int = 60) -> Optional[Path]:
        """
        Wait for file download to complete

        Args:
            filename_pattern: Pattern to match filename
            timeout: Maximum wait time in seconds

        Returns:
            Path to downloaded file or None
        """
        logger.debug(f"Waiting for download: {filename_pattern}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check for downloaded files
            files = list(self.download_dir.glob(filename_pattern))

            # Filter out .crdownload (Chrome) and .part (Firefox) files
            complete_files = [
                f for f in files
                if not f.name.endswith(('.crdownload', '.part', '.tmp'))
            ]

            if complete_files:
                downloaded_file = complete_files[0]
                logger.info(f"Download complete: {downloaded_file.name}")
                return downloaded_file

            time.sleep(0.5)

        logger.error(f"Download timeout after {timeout}s")
        return None

    def human_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0):
        """
        Random delay to simulate human behavior
        
        Args:
            min_seconds: Minimum delay
            max_seconds: Maximum delay
        """
        import random
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        
    def scroll_page(self):
        """Scroll page to simulate human reading behavior"""
        try:
            # Scroll down slowly
            self.driver.execute_script("""
                window.scrollTo({
                    top: document.body.scrollHeight / 2,
                    behavior: 'smooth'
                });
            """)
            self.human_delay(0.5, 1.5)
            
            # Scroll back up a bit
            self.driver.execute_script("""
                window.scrollTo({
                    top: document.body.scrollHeight / 4,
                    behavior: 'smooth'
                });
            """)
            self.human_delay(0.3, 0.8)
        except Exception as e:
            logger.debug(f"Scroll simulation failed: {e}")

    # Abstract methods that must be implemented by subclasses

    @abstractmethod
    def login(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Perform login to the academic database

        Args:
            username: Username or email
            password: Password

        Returns:
            True if login successful
        """
        pass

    @abstractmethod
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute search query

        Args:
            query: Search query string
            filters: Optional search filters (year range, document type, etc.)

        Returns:
            Number of results found
        """
        pass

    @abstractmethod
    def download_results(self, format: str = 'bibtex', max_results: Optional[int] = None) -> Path:
        """
        Download search results

        Args:
            format: Export format (bibtex, ris, csv, etc.)
            max_results: Maximum number of results to download

        Returns:
            Path to downloaded file
        """
        pass

    @abstractmethod
    def parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Parse downloaded file and extract records

        Args:
            filepath: Path to downloaded file

        Returns:
            List of publication records
        """
        pass

    def scrape(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Complete scraping workflow

        Args:
            query: Search query
            max_results: Maximum results to download

        Returns:
            List of publication records
        """
        logger.info(f"Starting scrape workflow for query: '{query}'")

        try:
            # Start browser session
            self.start_session()

            # Login if needed
            login_success = self.retry_with_backoff(self.login)
            if not login_success:
                logger.warning("Login failed or not required, continuing anyway")

            # Execute search
            num_results = self.retry_with_backoff(self.search, query)
            logger.info(f"Found {num_results} results")

            # Download results
            downloaded_file = self.retry_with_backoff(
                self.download_results,
                max_results=max_results
            )

            # Parse results
            records = self.parse_file(downloaded_file)
            logger.info(f"Parsed {len(records)} records")

            return records

        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=True)
            raise

        finally:
            # Always close browser
            self.close_session()

    def __enter__(self):
        """Context manager entry"""
        self.start_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_session()
