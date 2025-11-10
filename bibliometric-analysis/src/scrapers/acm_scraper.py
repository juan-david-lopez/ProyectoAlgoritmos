"""
ACM Digital Library Scraper - Versión Actualizada (Oct 2025)
Scraping directo de HTML con selectores validados

Selectores CSS validados por inspect_acm.py:
- Contador: span.hitsLength
- Artículos: li.search__item  
- Título: h3
- Autores: [class*='author']
- DOI: a[href*='/doi/']
"""

import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime

from src.scrapers.base_scraper import BaseScraper


class ACMScraper(BaseScraper):
    """
    Scraper para ACM Digital Library con scraping directo de HTML.
    
    Estrategia: Extrae metadata directamente del HTML en lugar de usar botones export.
    Selectores validados: span.hitsLength, li.search__item, h3, [class*='author']
    """

    BASE_URL = "https://dl.acm.org"
    
    # Selectores CSS validados (octubre 2025)
    RESULT_COUNT_SELECTOR = "span.hitsLength"
    ARTICLE_SELECTOR = "li.search__item"
    TITLE_SELECTOR = "h3"
    AUTHORS_SELECTOR = "[class*='author']"
    DOI_LINK_SELECTOR = "a[href*='/doi/']"
    NEXT_PAGE_SELECTOR = "a.pagination__btn--next"

    def __init__(self, config, headless: bool = True):
        """
        Initialize ACM scraper

        Args:
            config: Configuration object
            headless: Run in headless mode
        """
        super().__init__(config, headless)

        # ACM-specific configuration
        self.acm_config = config.get_source_config('acm') if hasattr(config, 'get_source_config') else {}
        
        # Set max_results from config or use default
        if self.acm_config:
            self.max_results = self.acm_config.get('max_results', 100)
        else:
            # Fallback to scraping config
            scraping_config = config.get('scraping', {}) if hasattr(config, 'get') else {}
            self.max_results = scraping_config.get('max_results', 100)

        # Credentials (if using institutional login)
        if hasattr(config, 'get_env'):
            self.username = config.get_env('ACM_USERNAME')
            self.password = config.get_env('ACM_PASSWORD')
        else:
            self.username = None
            self.password = None
        
        # Collected articles storage
        self.collected_articles = []
        
        # Initialize WebDriver and WebDriverWait
        self.driver = self._setup_driver()
        self.wait = WebDriverWait(self.driver, self.element_wait_timeout)

        logger.info("ACM scraper initialized (Direct HTML scraping v2025)")

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

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
               max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute search on ACM and extract article metadata.
        Uses direct HTML scraping instead of export buttons.

        Args:
            query: Search query (e.g., "generative artificial intelligence")
            filters: Optional filters (not implemented in this version)
            max_results: Maximum number of results to collect (overrides config)

        Returns:
            List of article dictionaries with metadata
        """
        try:
            # Use provided max_results or fall back to instance default
            max_results = max_results or self.max_results
            
            logger.info(f"Searching ACM for: '{query}' (max: {max_results})")

            # Navigate to search results
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"{self.BASE_URL}/action/doSearch?AllField={encoded_query}"
            
            logger.debug(f"Navigating to: {search_url}")
            self.driver.get(search_url)
            
            # Check for Cloudflare challenge
            if self._is_cloudflare_challenge():
                logger.warning("Cloudflare challenge detected, waiting for bypass...")
                self._wait_for_cloudflare_bypass()
            
            # Human-like behavior
            self.human_delay(2, 4)
            self.scroll_page()
            
            # Handle cookie banner if present
            self._handle_cookie_banner()

            # Get number of results
            total_results = self._get_result_count()
            logger.info(f"ACM search found {total_results:,} results")

            if total_results == 0:
                logger.warning("No results found")
                return []

            # Calculate pages to scrape (20 articles per page)
            articles_per_page = 20
            pages_needed = min((max_results // articles_per_page) + 1, 10)
            logger.info(f"Will scrape {pages_needed} pages to collect ~{max_results} articles")

            # Scrape multiple pages
            self.collected_articles = []
            for page_num in range(1, pages_needed + 1):
                logger.info(f"Scraping page {page_num}/{pages_needed}")
                
                page_articles = self._scrape_current_page()
                self.collected_articles.extend(page_articles)
                
                logger.info(f"Collected {len(page_articles)} articles from page {page_num}")
                logger.info(f"Total articles so far: {len(self.collected_articles)}")
                
                # Stop if reached max_results
                if len(self.collected_articles) >= max_results:
                    logger.info(f"Reached max_results limit ({max_results})")
                    break
                
                # Navigate to next page
                if page_num < pages_needed:
                    if not self._go_to_next_page(page_num):
                        logger.warning("Could not navigate to next page, stopping")
                        break
                    self.human_delay(2, 3)

            # Limit to max_results
            self.collected_articles = self.collected_articles[:max_results]
            logger.success(f"Collection complete: {len(self.collected_articles)} articles")
            
            return self.collected_articles

        except Exception as e:
            logger.error(f"ACM search error: {e}")
            if self.webdriver_manager:
                self.webdriver_manager.take_screenshot('acm_search_error.png')
            raise

    def _handle_cookie_banner(self):
        """Handle cookie consent banner if it appears"""
        try:
            cookie_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.CybotCookiebotDialogBodyButton"))
            )
            cookie_btn.click()
            logger.info("Cookie banner handled")
            time.sleep(1)
        except TimeoutException:
            logger.debug("No cookie banner found")
    
    def _is_cloudflare_challenge(self) -> bool:
        """Detect if Cloudflare challenge page is shown"""
        try:
            page_text = self.driver.page_source.lower()
            page_body = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            
            # Cloudflare indicators
            cf_indicators = [
                "cloudflare",
                "checking your browser",
                "verifique que usted es un ser humano",
                "verify you are human",
                "ray id:",
                "challenge-form",
                "cf-browser-verification"
            ]
            
            for indicator in cf_indicators:
                if indicator in page_text or indicator in page_body:
                    logger.warning(f"Cloudflare challenge detected: '{indicator}'")
                    return True
            
            return False
        except Exception as e:
            logger.debug(f"Error checking for Cloudflare: {e}")
            return False
    
    def _wait_for_cloudflare_bypass(self, max_wait: int = 30):
        """Wait for Cloudflare challenge to be bypassed"""
        logger.info(f"Waiting up to {max_wait}s for Cloudflare bypass...")
        
        start_time = time.time()
        while (time.time() - start_time) < max_wait:
            # Check if we're past Cloudflare
            if not self._is_cloudflare_challenge():
                logger.success("Cloudflare challenge bypassed!")
                time.sleep(2)  # Extra wait for page stability
                return True
            
            # Human-like random delay
            time.sleep(1 + (time.time() % 1))  # 1-2 seconds random
            
            # Scroll a bit (mimics human behavior)
            try:
                self.driver.execute_script("window.scrollTo(0, 100);")
            except:
                pass
        
        logger.error(f"Cloudflare challenge not bypassed after {max_wait}s")
        # Save screenshot for debugging
        if self.webdriver_manager:
            self.webdriver_manager.take_screenshot('cloudflare_challenge.png')
        return False

    def _get_result_count(self) -> int:
        """
        Get result count from page (tries multiple methods)
        
        Returns:
            Number of results found
        """
        # Method 1: Try validated selector first
        try:
            count_elem = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.RESULT_COUNT_SELECTOR))
            )
            text = count_elem.text.strip()
            count = int(text.replace(',', '').replace(' ', ''))
            logger.info(f"Found result count via selector: {count:,}")
            return count
        except (TimeoutException, ValueError) as e:
            logger.debug(f"Selector method failed: {e}")
        
        # Method 2: Try alternative selectors
        alternative_selectors = [
            ".result-count",
            "span.result__count",
            "div.search-result-count",
            "[class*='result'][class*='count']",
        ]
        
        for selector in alternative_selectors:
            try:
                elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                text = elem.text.strip()
                count = int(re.sub(r'[^\d]', '', text))
                logger.info(f"Found result count via {selector}: {count:,}")
                return count
            except:
                continue
        
        # Method 3: Extract from page text (most reliable)
        return self._get_result_count_from_text()

    def _get_result_count_from_text(self) -> int:
        """Fallback: extract count from page text using regex"""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Try multiple patterns
            patterns = [
                r'(\d{1,3}(?:,\d{3})*)\s*results?',  # "146,823 results"
                r'(\d{1,3}(?:,\d{3})*)\s*artículos?',  # Spanish
                r'results?[:\s]+(\d{1,3}(?:,\d{3})*)',  # "results: 146823"
                r'found\s+(\d{1,3}(?:,\d{3})*)',  # "found 146823"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    count = int(match.group(1).replace(',', ''))
                    logger.info(f"Found result count in text: {count:,}")
                    return count
            
            logger.warning("Could not parse result count from text, returning 0")
            return 0
        except Exception as e:
            logger.error(f"Error parsing result count: {e}")
            return 0

    def _scrape_current_page(self) -> List[Dict[str, Any]]:
        """
        Extract article metadata from current page using validated selectors.
        
        Returns:
            List of article dictionaries
        """
        articles = []
        article_elements = []
        
        # Try multiple selectors for article containers
        article_selectors = [
            self.ARTICLE_SELECTOR,  # li.search__item
            "li.search-result",
            "div.issue-item",
            "article",
            "div[class*='search'][class*='item']",
            "li[class*='result']",
        ]
        
        for selector in article_selectors:
            try:
                article_elements = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                )
                if article_elements:
                    logger.info(f"Found {len(article_elements)} article elements using selector: {selector}")
                    break
            except TimeoutException:
                continue
        
        if not article_elements:
            logger.error("Could not find article elements with any selector")
            if self.webdriver_manager:
                self.webdriver_manager.take_screenshot('acm_no_articles.png')
            return []
        
        # Extract metadata from each article
        for idx, elem in enumerate(article_elements, 1):
            try:
                article_data = self._extract_article_metadata(elem)
                if article_data:
                    articles.append(article_data)
                    logger.debug(f"[{idx}] Extracted: {article_data.get('title', '')[:50]}...")
            except Exception as e:
                logger.warning(f"Error extracting article {idx}: {e}")
                continue
        
        return articles

    def _extract_article_metadata(self, element) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from article element using validated selectors.
        
        Args:
            element: Selenium WebElement (li.search__item)
            
        Returns:
            Dictionary with article metadata or None
        """
        try:
            article = {}
            
            # Title (selector: h3)
            try:
                title_elem = element.find_element(By.CSS_SELECTOR, self.TITLE_SELECTOR)
                article['title'] = title_elem.text.strip()
                if not article['title']:
                    return None
            except NoSuchElementException:
                logger.debug("Could not find title")
                return None
            
            # Authors (selector: [class*='author'])
            try:
                authors_elem = element.find_element(By.CSS_SELECTOR, self.AUTHORS_SELECTOR)
                article['authors'] = authors_elem.text.strip()
            except NoSuchElementException:
                article['authors'] = "Unknown"
            
            # DOI and URL (selector: a[href*='/doi/'])
            try:
                doi_link = element.find_element(By.CSS_SELECTOR, self.DOI_LINK_SELECTOR)
                href = doi_link.get_attribute('href')
                doi_match = re.search(r'/doi/(10\.\d+/[^\s?]+)', href)
                if doi_match:
                    article['doi'] = doi_match.group(1)
                    article['url'] = href
            except NoSuchElementException:
                article['doi'] = None
                article['url'] = None
            
            # Year/Date
            try:
                date_selectors = [".dot-separator span", ".bookPubDate", "[class*='date']"]
                for selector in date_selectors:
                    try:
                        date_elem = element.find_element(By.CSS_SELECTOR, selector)
                        date_text = date_elem.text
                        year_match = re.search(r'\d{4}', date_text)
                        if year_match:
                            article['year'] = year_match.group()
                            break
                    except:
                        continue
                
                if 'year' not in article:
                    article['year'] = str(datetime.now().year)
            except:
                article['year'] = str(datetime.now().year)
            
            # Additional metadata
            article['source'] = 'acm'
            article['scraped_at'] = datetime.now().isoformat()
            
            return article
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
            return None

    def _go_to_next_page(self, current_page: int) -> bool:
        """
        Navigate to next page using validated selector.
        
        Args:
            current_page: Current page number
            
        Returns:
            bool: True if navigation successful
        """
        try:
            next_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, self.NEXT_PAGE_SELECTOR))
            )
            
            # Scroll to button
            self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            self.human_delay(0.5, 1)
            
            next_button.click()
            logger.info(f"Navigated to page {current_page + 1}")
            return True
            
        except (TimeoutException, NoSuchElementException) as e:
            logger.warning(f"Could not find next page button: {e}")
            return False

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
        Generate output file from collected article data.
        No longer uses export buttons - generates file directly from scraped data.

        Args:
            format: Export format ('bibtex', 'json', or 'csv')
            max_results: Maximum number of results (uses collected articles)

        Returns:
            Path to generated file
        """
        if not self.collected_articles:
            logger.error("No articles collected. Run search() first.")
            raise Exception("No articles to download. Call search() before download_results()")
        
        logger.info(f"Generating {format} file from {len(self.collected_articles)} articles")
        
        output_dir = self.download_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'bibtex':
            output_file = output_dir / f"acm_results_{timestamp}.bib"
            content = self._convert_to_bibtex(self.collected_articles)
        elif format.lower() == 'json':
            output_file = output_dir / f"acm_results_{timestamp}.json"
            import json
            content = json.dumps(self.collected_articles, indent=2, ensure_ascii=False)
        elif format.lower() == 'csv':
            output_file = output_dir / f"acm_results_{timestamp}.csv"
            content = self._convert_to_csv(self.collected_articles)
        else:
            logger.error(f"Unsupported format: {format}")
            raise ValueError(f"Unsupported format: {format}")
        
        # Write file
        output_file.write_text(content, encoding='utf-8')
        logger.success(f"Results saved to: {output_file}")
        
        return output_file

    def _convert_to_bibtex(self, articles: List[Dict[str, Any]]) -> str:
        """Convert articles to BibTeX format"""
        bibtex_entries = []
        
        for art in articles:
            # Create clean citation key
            title_words = art.get('title', 'unknown')[:30].split()
            clean_words = [w for w in title_words[:3] if w.isalnum()]
            key = f"acm_{art.get('year', '2024')}_{'_'.join(clean_words)}"
            key = re.sub(r'[^\w_]', '', key)
            
            entry = f"""@article{{{key},
  title = {{{art.get('title', 'Unknown')}}},
  author = {{{art.get('authors', 'Unknown')}}},
  year = {{{art.get('year', '2024')}}},
  doi = {{{art.get('doi', 'N/A')}}},
  url = {{{art.get('url', 'N/A')}}},
  publisher = {{ACM}},
  source = {{ACM Digital Library}}
}}
"""
            bibtex_entries.append(entry)
        
        header = f"""% BibTeX Export from ACM Digital Library
% Generated: {datetime.now().isoformat()}
% Total entries: {len(articles)}
% Scraper: ACM Direct HTML Scraping v2025

"""
        return header + "\n".join(bibtex_entries)

    def _convert_to_csv(self, articles: List[Dict[str, Any]]) -> str:
        """Convert articles to CSV format"""
        import csv
        from io import StringIO
        
        output = StringIO()
        fieldnames = ['title', 'authors', 'year', 'doi', 'url', 'source']
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        
        writer.writeheader()
        for article in articles:
            # Clean any newlines in fields
            cleaned = {k: str(v).replace('\n', ' ').replace('\r', ' ') 
                      for k, v in article.items() if k in fieldnames}
            writer.writerow(cleaned)
        
        return output.getvalue()
    
    def scrape(self, query: str, max_results: Optional[int] = None, format: str = 'bibtex') -> Path:
        """
        Convenience method: search + download in one call.
        
        Args:
            query: Search query string
            max_results: Maximum results to collect
            format: Output format ('bibtex', 'json', or 'csv')
        
        Returns:
            Path to generated output file
        """
        logger.info(f"Starting complete scrape: '{query}' (max: {max_results or self.max_results})")
        
        # Search and collect articles
        articles = self.search(query, max_results=max_results)
        
        if not articles:
            logger.warning("No articles found for query")
            return None
        
        # Generate output file
        output_file = self.download_results(format=format)
        
        logger.success(f"Scrape complete: {len(articles)} articles → {output_file.name}")
        return output_file

    def parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Parse generated file and return normalized records.
        Supports BibTeX, JSON, and CSV formats.

        Args:
            filepath: Path to generated file

        Returns:
            List of normalized publication records
        """
        try:
            logger.info(f"Parsing file: {filepath.name}")
            
            # Determine format from extension
            if filepath.suffix == '.bib':
                records = self._parse_bibtex_file(filepath)
            elif filepath.suffix == '.json':
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                records = [self._normalize_scraped_article(art) for art in articles]
            elif filepath.suffix == '.csv':
                import csv
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    articles = list(reader)
                records = [self._normalize_scraped_article(art) for art in articles]
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            logger.info(f"Parsed {len(records)} records from {filepath.suffix} file")
            return records

        except Exception as e:
            logger.error(f"Error parsing file: {e}")
            raise

    def _parse_bibtex_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Simple BibTeX parser without external dependencies"""
        records = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by @article entries
        entries = re.findall(r'@article\{([^}]+)\s*,([^@]+)', content, re.DOTALL)
        
        for entry_id, entry_content in entries:
            # Extract fields
            fields = {}
            field_matches = re.findall(r'(\w+)\s*=\s*\{([^}]+)\}', entry_content)
            for field_name, field_value in field_matches:
                fields[field_name.lower()] = field_value.strip()
            
            # Normalize to standard format
            record = self._normalize_scraped_article(fields)
            record['id'] = entry_id.strip()
            records.append(record)
        
        return records

    def _normalize_scraped_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize scraped article to standard publication record format.

        Args:
            article: Scraped article dictionary

        Returns:
            Normalized publication record
        """
        # Handle authors (could be string or list)
        authors = article.get('authors', '')
        if isinstance(authors, str):
            # Split by common separators
            authors = [a.strip() for a in re.split(r'[,;]|\band\b', authors) if a.strip()]
        
        # Extract year
        year = article.get('year', '')
        if not year and 'date' in article:
            # Try to extract year from date string
            year_match = re.search(r'20\d{2}', str(article.get('date', '')))
            if year_match:
                year = year_match.group()
        
        # Normalize record to standard format
        record = {
            'id': article.get('doi', article.get('id', '')),
            'title': article.get('title', 'Unknown Title'),
            'authors': authors if isinstance(authors, list) else [authors],
            'year': year,
            'abstract': article.get('abstract', ''),
            'keywords': article.get('keywords', []),
            'doi': article.get('doi', ''),
            'source': article.get('source', 'ACM Digital Library'),
            'publication_type': article.get('publication_type', 'article'),
            'journal_conference': article.get('journal_conference', ''),
            'url': article.get('url', ''),
            'publisher': article.get('publisher', 'ACM'),
            'pages': article.get('pages', ''),
            'volume': article.get('volume', ''),
            'number': article.get('number', ''),
            'isbn': article.get('isbn', ''),
        }
        
        return record
    
    def close(self):
        """Close browser and cleanup resources"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
                logger.info("ACM scraper closed successfully")
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")


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
