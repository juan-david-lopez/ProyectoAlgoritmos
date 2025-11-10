"""
ScienceDirect Scraper
Automates data collection from ScienceDirect/Elsevier
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import rispy
from datetime import datetime

from src.scrapers.base_scraper import BaseScraper


class ScienceDirectScraper(BaseScraper):
    """
    Scraper for ScienceDirect/Elsevier

    Features:
    - Automated search
    - RIS export
    - Institutional authentication support
    - Robust error handling
    """

    BASE_URL = "https://www.sciencedirect.com"
    SEARCH_URL = f"{BASE_URL}/search"

    def __init__(self, config, headless: bool = True):
        """
        Initialize ScienceDirect scraper

        Args:
            config: Configuration object
            headless: Run in headless mode
        """
        super().__init__(config, headless)

        # ScienceDirect-specific configuration
        self.sd_config = config.get_source_config('sciencedirect')
        self.max_results = self.sd_config.get('max_results', 1000)

        # Credentials (if using institutional login)
        self.username = config.get_env('SCIENCEDIRECT_USERNAME')
        self.password = config.get_env('SCIENCEDIRECT_PASSWORD')

        logger.info("ScienceDirect scraper initialized")

    def login(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Perform institutional login to ScienceDirect

        Args:
            username: Username/email (uses config if None)
            password: Password (uses config if None)

        Returns:
            True if login successful or not required
        """
        username = username or self.username
        password = password or self.password

        # If no credentials, assume open access or institutional IP
        if not username or not password:
            logger.info("No ScienceDirect credentials provided, assuming open/institutional access")
            return True

        try:
            logger.info("Attempting ScienceDirect institutional login")

            # Navigate to login page
            self.driver.get(f"{self.BASE_URL}/user/login")
            time.sleep(2)

            # Find and fill username
            username_field = self.safe_find_element(By.ID, "username")
            if not username_field:
                username_field = self.safe_find_element(By.NAME, "username")

            if username_field:
                username_field.clear()
                username_field.send_keys(username)

            # Find and fill password
            password_field = self.safe_find_element(By.ID, "password")
            if not password_field:
                password_field = self.safe_find_element(By.NAME, "password")

            if password_field:
                password_field.clear()
                password_field.send_keys(password)

            # Submit form
            submit_button = self.safe_find_element(By.CSS_SELECTOR, "button[type='submit']")
            if submit_button:
                submit_button.click()
                time.sleep(3)

            # Verify login success
            if "user-info" in self.driver.page_source.lower():
                logger.info("ScienceDirect login successful")
                return True
            else:
                logger.warning("ScienceDirect login may have failed")
                return False

        except Exception as e:
            logger.error(f"ScienceDirect login error: {e}")
            return False

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute search on ScienceDirect

        Args:
            query: Search query (e.g., "generative artificial intelligence")
            filters: Optional filters (year_start, year_end, content_type)

        Returns:
            Number of results found
        """
        try:
            logger.info(f"Searching ScienceDirect for: '{query}'")

            # Build search URL with query - use more compatible format
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"{self.SEARCH_URL}?qs={encoded_query}"
            
            logger.debug(f"Navigating to: {search_url}")
            self.driver.get(search_url)
            
            # Human-like behavior
            self.human_delay(2, 4)  # Wait like a human would
            self.scroll_page()  # Simulate reading

            # Apply filters if provided
            if filters:
                self._apply_filters(filters)

            # Get number of results
            num_results = self._get_result_count()
            logger.info(f"ScienceDirect search found {num_results} results")

            return num_results

        except Exception as e:
            logger.error(f"ScienceDirect search error: {e}")
            raise

    def _apply_filters(self, filters: Dict[str, Any]):
        """
        Apply search filters

        Args:
            filters: Dictionary of filters
        """
        logger.debug(f"Applying filters: {filters}")

        # Year range filter
        if 'year_start' in filters or 'year_end' in filters:
            # Click on date filter
            date_filter = self.safe_find_element(
                By.CSS_SELECTOR,
                "button[aria-label*='Date'], .date-range-filter"
            )
            if date_filter:
                date_filter.click()
                time.sleep(1)

            if 'year_start' in filters:
                year_start_field = self.safe_find_element(By.NAME, "yearFrom")
                if year_start_field:
                    year_start_field.clear()
                    year_start_field.send_keys(str(filters['year_start']))

            if 'year_end' in filters:
                year_end_field = self.safe_find_element(By.NAME, "yearTo")
                if year_end_field:
                    year_end_field.clear()
                    year_end_field.send_keys(str(filters['year_end']))

            # Apply date filter
            apply_button = self.safe_find_element(
                By.CSS_SELECTOR,
                "button[aria-label*='Apply']"
            )
            if apply_button:
                apply_button.click()
                time.sleep(2)

        # Content type filter
        if 'content_type' in filters:
            content_type = filters['content_type']
            checkbox = self.safe_find_element(
                By.CSS_SELECTOR,
                f"input[value='{content_type}']"
            )
            if checkbox and not checkbox.is_selected():
                checkbox.click()
                time.sleep(1)

    def _get_result_count(self) -> int:
        """
        Extract number of search results

        Returns:
            Number of results
        """
        try:
            # ScienceDirect shows results like "1,543 results"
            # Try multiple selectors
            result_text_element = self.safe_find_element(
                By.CSS_SELECTOR,
                ".search-result-count, .results-number, .search-body__info, span.result-count"
            )
            
            # Try by xpath if CSS selector fails
            if not result_text_element:
                try:
                    result_text_element = self.driver.find_element(By.XPATH, "//*[contains(text(), 'result') or contains(text(), 'Result')]")
                except:
                    pass

            if result_text_element:
                text = result_text_element.text
                logger.debug(f"Result text: {text}")

                # Extract number with multiple patterns
                import re
                
                # Pattern 1: "X,XXX results"
                match = re.search(r'([\d,]+)\s*results?', text, re.IGNORECASE)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)
                
                # Pattern 2: "X,XXX Results found"
                match = re.search(r'([\d,]+)\s*Results?\s+found', text, re.IGNORECASE)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)

            logger.warning("Could not parse result count, returning 0")
            return 0

        except Exception as e:
            logger.warning(f"Error getting result count: {e}")
            return 0

    def download_results(self, format: str = 'ris', max_results: Optional[int] = None) -> Path:
        """
        Download search results in RIS format

        Args:
            format: Export format (ris is default for ScienceDirect)
            max_results: Maximum number of results to download

        Returns:
            Path to downloaded RIS file
        """
        try:
            max_results = max_results or self.max_results
            logger.info(f"Downloading up to {max_results} results in {format} format")

            # Wait for page to fully load
            time.sleep(3)

            # Select results - Try multiple selectors
            select_all_link = self.safe_find_element(
                By.CSS_SELECTOR,
                "a[title*='Select all'], .select-all-results, button.select-all, input.select-all-checkbox"
            )
            
            # Try by text
            if not select_all_link:
                try:
                    select_all_link = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Select all')] | //button[contains(text(), 'Select all')]")
                except:
                    pass

            if select_all_link:
                try:
                    # Scroll into view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", select_all_link)
                    time.sleep(1)
                    
                    # Click with JavaScript
                    self.driver.execute_script("arguments[0].click();", select_all_link)
                    time.sleep(2)
                    logger.debug("Selected all results on page")
                except Exception as e:
                    logger.warning(f"Could not select all: {e}")

            # Click export button - Try multiple selectors
            export_button = self.safe_find_element(
                By.CSS_SELECTOR,
                "button[aria-label*='Export'], .export-results, button.export, a.export-link, button[title*='Export']"
            )
            
            # Try by text
            if not export_button:
                try:
                    export_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Export')] | //a[contains(text(), 'Export')] | //button[contains(@aria-label, 'export')]")
                except:
                    pass

            if export_button:
                try:
                    # Scroll into view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", export_button)
                    time.sleep(1)
                    
                    # Click with JavaScript
                    self.driver.execute_script("arguments[0].click();", export_button)
                    time.sleep(2)
                    logger.debug("Clicked export button")
                except Exception as e:
                    logger.warning(f"Could not click export: {e}")
                    raise Exception("Could not click export button")
            else:
                raise Exception("Could not find export button")

            # Select RIS format
            ris_option = self.safe_find_element(
                By.CSS_SELECTOR,
                "input[value='RIS'], label:contains('RIS')"
            )

            if ris_option:
                ris_option.click()
                time.sleep(1)

            # Click download button
            download_button = self.safe_find_element(
                By.CSS_SELECTOR,
                "button.export-download, button[aria-label*='Export']"
            )

            if download_button:
                download_button.click()
                logger.debug("Initiated download")

            # Wait for download to complete
            downloaded_file = self.wait_for_download("*.ris", timeout=60)

            if not downloaded_file:
                raise Exception("Download timeout - RIS file not found")

            # Rename file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_name = f"sciencedirect_results_{timestamp}.ris"
            renamed_file = downloaded_file.parent / new_name
            downloaded_file.rename(renamed_file)

            logger.info(f"Downloaded and saved as: {renamed_file.name}")
            return renamed_file

        except Exception as e:
            logger.error(f"Error downloading ScienceDirect results: {e}")
            raise

    def parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Parse RIS file and extract publication records

        Args:
            filepath: Path to RIS file

        Returns:
            List of normalized publication records
        """
        try:
            logger.info(f"Parsing RIS file: {filepath.name}")

            with open(filepath, 'r', encoding='utf-8') as ris_file:
                entries = rispy.load(ris_file)

            records = []

            for entry in entries:
                record = self._normalize_ris_entry(entry)
                records.append(record)

            logger.info(f"Parsed {len(records)} records from RIS file")
            return records

        except Exception as e:
            logger.error(f"Error parsing RIS file: {e}")
            raise

    def _normalize_ris_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize RIS entry to standard format

        Args:
            entry: RIS entry dictionary

        Returns:
            Normalized publication record
        """
        # Extract authors
        authors = entry.get('authors', [])
        if isinstance(authors, str):
            authors = [authors]

        # Extract keywords
        keywords = entry.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]

        # Determine publication type
        pub_type = entry.get('type_of_reference', 'JOUR')  # JOUR = Journal Article

        # Extract venue
        venue = entry.get('journal_name') or entry.get('secondary_title', '')

        # Normalize record
        record = {
            'id': entry.get('id', ''),
            'title': entry.get('title', '') or entry.get('primary_title', ''),
            'authors': authors,
            'year': entry.get('year', ''),
            'abstract': entry.get('abstract', ''),
            'keywords': keywords,
            'doi': entry.get('doi', ''),
            'source': 'ScienceDirect',
            'publication_type': pub_type,
            'journal_conference': venue,
            'url': entry.get('url', ''),
            'publisher': entry.get('publisher', 'Elsevier'),
            'pages': entry.get('start_page', '') + '-' + entry.get('end_page', '') if entry.get('start_page') else '',
            'volume': entry.get('volume', ''),
            'number': entry.get('number', ''),
            'issn': entry.get('issn', ''),
            'raw_ris': entry
        }

        return record


# Example usage
if __name__ == "__main__":
    from src.utils.config_loader import get_config

    # Setup logger
    logger.add("logs/sciencedirect_scraper.log", rotation="10 MB")

    # Load configuration
    config = get_config()

    # Create scraper
    scraper = ScienceDirectScraper(config, headless=False)

    try:
        # Execute search and download
        query = "generative artificial intelligence"
        records = scraper.scrape(query, max_results=50)

        logger.success(f"Successfully scraped {len(records)} records from ScienceDirect")

        # Print first record as example
        if records:
            logger.info(f"Example record: {records[0]['title']}")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
