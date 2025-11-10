"""
ACM Digital Library Scraper
Automates data collection from ACM Digital Library
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import bibtexparser
from datetime import datetime

from src.scrapers.base_scraper import BaseScraper


class ACMScraper(BaseScraper):
    """
    Scraper for ACM Digital Library

    Features:
    - Automated search
    - BibTeX export
    - Institutional authentication support
    - Robust error handling
    """

    BASE_URL = "https://dl.acm.org"
    SEARCH_URL = f"{BASE_URL}/search/advanced"

    def __init__(self, config, headless: bool = True):
        """
        Initialize ACM scraper

        Args:
            config: Configuration object
            headless: Run in headless mode
        """
        super().__init__(config, headless)

        # ACM-specific configuration
        self.acm_config = config.get_source_config('acm')
        self.max_results = self.acm_config.get('max_results', 1000)

        # Credentials (if using institutional login)
        self.username = config.get_env('ACM_USERNAME')
        self.password = config.get_env('ACM_PASSWORD')

        logger.info("ACM Digital Library scraper initialized")

    def login(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Perform institutional login to ACM

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
            logger.info("No ACM credentials provided, assuming open/institutional access")
            return True

        try:
            logger.info("Attempting ACM institutional login")

            # Navigate to login page
            self.driver.get(f"{self.BASE_URL}/action/showLogin")
            time.sleep(2)

            # Find and fill username
            username_field = self.safe_find_element(By.ID, "username")
            if username_field:
                username_field.clear()
                username_field.send_keys(username)

            # Find and fill password
            password_field = self.safe_find_element(By.ID, "password")
            if password_field:
                password_field.clear()
                password_field.send_keys(password)

            # Submit form
            submit_button = self.safe_find_element(By.CSS_SELECTOR, "button[type='submit']")
            if submit_button:
                submit_button.click()
                time.sleep(3)

            # Verify login success
            if "my-account" in self.driver.current_url.lower():
                logger.info("ACM login successful")
                return True
            else:
                logger.warning("ACM login may have failed")
                return False

        except Exception as e:
            logger.error(f"ACM login error: {e}")
            return False

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute search on ACM Digital Library

        Args:
            query: Search query (e.g., "generative artificial intelligence")
            filters: Optional filters (year_start, year_end, content_type)

        Returns:
            Number of results found
        """
        try:
            logger.info(f"Searching ACM for: '{query}'")

            # Navigate directly to search results with query parameter
            # This bypasses the search form interaction issues
            from urllib.parse import quote
            encoded_query = quote(query)
            search_results_url = f"https://dl.acm.org/action/doSearch?AllField={encoded_query}"
            
            logger.debug(f"Navigating to: {search_results_url}")
            self.driver.get(search_results_url)
            
            # Human-like behavior
            self.human_delay(2, 4)  # Wait like a human would
            self.scroll_page()  # Simulate reading
            
            # Apply filters if provided
            if filters:
                self._apply_filters(filters)

            # Get number of results
            num_results = self._get_result_count()
            logger.info(f"ACM search found {num_results} results")

            return num_results

        except Exception as e:
            logger.error(f"ACM search error: {e}")
            raise

    def _apply_filters(self, filters: Dict[str, Any]):
        """
        Apply search filters

        Args:
            filters: Dictionary of filters
        """
        logger.debug(f"Applying filters: {filters}")

        # Year range filter
        if 'year_start' in filters:
            year_start_field = self.safe_find_element(By.NAME, "startPage")
            if year_start_field:
                year_start_field.clear()
                year_start_field.send_keys(str(filters['year_start']))

        if 'year_end' in filters:
            year_end_field = self.safe_find_element(By.NAME, "endPage")
            if year_end_field:
                year_end_field.clear()
                year_end_field.send_keys(str(filters['year_end']))

        # Content type filter (research articles, conference papers, etc.)
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
            from selenium.webdriver.common.by import By
            
            # ACM shows results like "1-20 of 1,543 results"
            # Try multiple selectors for result count
            result_text_element = self.safe_find_element(
                By.CSS_SELECTOR,
                ".result__count, .hitsLength, span.result-count, .search-result-count"
            )
            
            # Also try by text content
            if not result_text_element:
                try:
                    # Look for any element containing "results"
                    elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'results') or contains(text(), 'Results')]")
                    if elements:
                        result_text_element = elements[0]
                except:
                    pass

            if result_text_element:
                text = result_text_element.text
                logger.debug(f"Result text: {text}")

                # Extract number (e.g., "1,543" from "1-20 of 1,543 results" or "1,543 Results")
                import re
                
                # Try "X of Y results" pattern
                match = re.search(r'of\s+([\d,]+)', text)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)
                
                # Try "Y results" pattern
                match = re.search(r'([\d,]+)\s+[Rr]esults?', text)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)

            logger.warning("Could not parse result count, returning 0")
            return 0

        except Exception as e:
            logger.warning(f"Error getting result count: {e}")
            return 0

    def download_results(self, format: str = 'bibtex', max_results: Optional[int] = None) -> Path:
        """
        Download search results in BibTeX format

        Args:
            format: Export format (bibtex is default for ACM)
            max_results: Maximum number of results to download

        Returns:
            Path to downloaded BibTeX file
        """
        try:
            max_results = max_results or self.max_results
            logger.info(f"Downloading up to {max_results} results in {format} format")

            # Try multiple methods to select results
            time.sleep(2)  # Wait for page to fully load
            
            # Method 1: Try to find and click "Select All" checkbox
            select_all_checkbox = self.safe_find_element(
                By.CSS_SELECTOR,
                "input[type='checkbox'][name='selectAll'], .select-all, input.select-all-checkbox"
            )

            if select_all_checkbox:
                try:
                    # Scroll into view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", select_all_checkbox)
                    time.sleep(1)
                    
                    if not select_all_checkbox.is_selected():
                        # Try clicking with JavaScript if regular click fails
                        try:
                            select_all_checkbox.click()
                        except:
                            self.driver.execute_script("arguments[0].click();", select_all_checkbox)
                        time.sleep(1)
                        logger.debug("Selected all results on page")
                except Exception as e:
                    logger.warning(f"Could not click select all checkbox: {e}")

            # Method 2: Click export/download button with multiple attempts
            export_button = self.safe_find_element(
                By.CSS_SELECTOR,
                "a[title*='Export'], button[title*='Export'], .export-citations, button.export-button, a.export-link"
            )
            
            # Also try by text
            if not export_button:
                try:
                    export_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Export')] | //a[contains(text(), 'Export')]")
                except:
                    pass

            if export_button:
                try:
                    # Scroll into view
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", export_button)
                    time.sleep(1)
                    
                    # Try clicking with JavaScript
                    self.driver.execute_script("arguments[0].click();", export_button)
                    time.sleep(2)
                    logger.debug("Clicked export button")
                except Exception as e:
                    logger.warning(f"Error clicking export button: {e}")
                    raise Exception("Could not click export button")
            else:
                raise Exception("Could not find export button")

            # Select BibTeX format with multiple selectors
            bibtex_option = self.safe_find_element(
                By.CSS_SELECTOR,
                "input[value='bibtex'], label[for='bibtex'], input[id='bibtex']"
            )
            
            # Try by text
            if not bibtex_option:
                try:
                    bibtex_option = self.driver.find_element(By.XPATH, "//label[contains(text(), 'BibTeX')] | //input[@type='radio' and contains(@id, 'bibtex')]")
                except:
                    pass

            if bibtex_option:
                try:
                    self.driver.execute_script("arguments[0].click();", bibtex_option)
                    time.sleep(1)
                    logger.debug("Selected BibTeX format")
                except Exception as e:
                    logger.warning(f"Could not select BibTeX format: {e}")

            # Click download/export button
            download_button = self.safe_find_element(
                By.CSS_SELECTOR,
                "button[type='submit'], input[type='submit'], .export-submit, button.download-button"
            )
            
            # Try by text
            if not download_button:
                try:
                    download_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Download')] | //button[contains(text(), 'Export')]")
                except:
                    pass

            if download_button:
                try:
                    self.driver.execute_script("arguments[0].click();", download_button)
                    logger.debug("Initiated download")
                except Exception as e:
                    logger.warning(f"Could not click download button: {e}")

            # Wait for download to complete
            downloaded_file = self.wait_for_download("*.bib", timeout=60)

            if not downloaded_file:
                raise Exception("Download timeout - file not found")

            # Rename file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_name = f"acm_results_{timestamp}.bib"
            renamed_file = downloaded_file.parent / new_name
            downloaded_file.rename(renamed_file)

            logger.info(f"Downloaded and saved as: {renamed_file.name}")
            return renamed_file

        except Exception as e:
            logger.error(f"Error downloading ACM results: {e}")
            raise

    def parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Parse BibTeX file and extract publication records

        Args:
            filepath: Path to BibTeX file

        Returns:
            List of normalized publication records
        """
        try:
            logger.info(f"Parsing BibTeX file: {filepath.name}")

            with open(filepath, 'r', encoding='utf-8') as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)

            records = []

            for entry in bib_database.entries:
                record = self._normalize_bibtex_entry(entry)
                records.append(record)

            logger.info(f"Parsed {len(records)} records from BibTeX file")
            return records

        except Exception as e:
            logger.error(f"Error parsing BibTeX file: {e}")
            raise

    def _normalize_bibtex_entry(self, entry: Dict[str, str]) -> Dict[str, Any]:
        """
        Normalize BibTeX entry to standard format

        Args:
            entry: BibTeX entry dictionary

        Returns:
            Normalized publication record
        """
        # Extract authors (handle various formats)
        authors_str = entry.get('author', '')
        authors = [a.strip() for a in authors_str.split(' and ')] if authors_str else []

        # Extract keywords
        keywords_str = entry.get('keywords', '')
        keywords = [k.strip() for k in keywords_str.split(',')] if keywords_str else []

        # Determine publication type
        pub_type = entry.get('ENTRYTYPE', 'article')

        # Extract journal or conference
        venue = entry.get('journal') or entry.get('booktitle') or entry.get('series', '')

        # Normalize record
        record = {
            'id': entry.get('ID', ''),
            'title': entry.get('title', '').replace('{', '').replace('}', ''),
            'authors': authors,
            'year': entry.get('year', ''),
            'abstract': entry.get('abstract', ''),
            'keywords': keywords,
            'doi': entry.get('doi', ''),
            'source': 'ACM Digital Library',
            'publication_type': pub_type,
            'journal_conference': venue,
            'url': entry.get('url', ''),
            'publisher': entry.get('publisher', 'ACM'),
            'pages': entry.get('pages', ''),
            'volume': entry.get('volume', ''),
            'number': entry.get('number', ''),
            'isbn': entry.get('isbn', ''),
            'raw_bibtex': entry
        }

        return record


# Example usage
if __name__ == "__main__":
    from src.utils.config_loader import get_config

    # Setup logger
    logger.add("logs/acm_scraper.log", rotation="10 MB")

    # Load configuration
    config = get_config()

    # Create scraper
    scraper = ACMScraper(config, headless=False)

    try:
        # Execute search and download
        query = "generative artificial intelligence"
        records = scraper.scrape(query, max_results=50)

        logger.success(f"Successfully scraped {len(records)} records from ACM")

        # Print first record as example
        if records:
            logger.info(f"Example record: {records[0]['title']}")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
