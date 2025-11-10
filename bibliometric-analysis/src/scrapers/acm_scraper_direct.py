"""
ACM Digital Library Direct Scraper
Extracts data directly from HTML without using export buttons
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
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper


class ACMScraperDirect(BaseScraper):
    """
    ACM Digital Library direct scraper
    Extracts article data directly from search results HTML
    """

    BASE_URL = "https://dl.acm.org"
    SEARCH_URL = f"{BASE_URL}/action/doSearch"

    def __init__(self, config, headless: bool = True):
        """Initialize ACM scraper"""
        super().__init__(config, headless)
        self.max_results = config.get('scraping.max_results_per_source', 50)
        logger.info("ACM Digital Library direct scraper initialized")

    def login(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """ACM doesn't require login for basic searches"""
        logger.info("No ACM credentials provided, assuming open/institutional access")
        return True

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute search on ACM Digital Library

        Args:
            query: Search query
            filters: Optional filters

        Returns:
            Number of results found
        """
        try:
            logger.info(f"Searching ACM for: '{query}'")

            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"{self.SEARCH_URL}?AllField={encoded_query}"
            
            logger.debug(f"Navigating to: {search_url}")
            self.driver.get(search_url)
            
            # Human-like behavior
            self.human_delay(3, 5)
            self.scroll_page()
            
            # Get number of results
            num_results = self._get_result_count()
            logger.info(f"ACM search found {num_results} results")

            return num_results

        except Exception as e:
            logger.error(f"ACM search error: {e}")
            raise

    def _get_result_count(self) -> int:
        """Extract number of search results"""
        try:
            from selenium.webdriver.common.by import By
            
            # Wait for results to load
            time.sleep(2)
            
            # Try to find result count in page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Look for result count patterns
            patterns = [
                r'of\s+([\d,]+)\s+results',
                r'([\d,]+)\s+results',
                r'showing\s+\d+-\d+\s+of\s+([\d,]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, page_source, re.IGNORECASE)
                if match:
                    count_str = match.group(1).replace(',', '')
                    return int(count_str)
            
            # If no count found, count visible articles
            articles = soup.find_all(['li', 'div'], class_=re.compile(r'search.*result|result.*item|article.*item', re.I))
            if articles:
                logger.info(f"Found {len(articles)} articles on current page")
                return len(articles)
            
            return 0

        except Exception as e:
            logger.warning(f"Error getting result count: {e}")
            return 0

    def download_results(self, format: str = 'bibtex', max_results: Optional[int] = None) -> Path:
        """
        Extract article data directly from HTML

        Args:
            format: Format (not used, always extracts HTML data)
            max_results: Maximum results to extract

        Returns:
            Path to output file
        """
        max_results = max_results or self.max_results
        logger.info(f"Extracting up to {max_results} articles from HTML")

        all_articles = []
        pages_scraped = 0
        
        try:
            while len(all_articles) < max_results:
                # Extract articles from current page
                articles = self._extract_articles_from_page()
                
                if not articles:
                    logger.warning("No articles found on current page")
                    break
                
                all_articles.extend(articles)
                pages_scraped += 1
                
                logger.info(f"Extracted {len(articles)} articles from page {pages_scraped} (total: {len(all_articles)})")
                
                # Stop if we have enough
                if len(all_articles) >= max_results:
                    break
                
                # Try to go to next page
                if not self._go_to_next_page():
                    logger.info("No more pages available")
                    break
                
                self.human_delay(2, 4)
            
            # Trim to max_results
            all_articles = all_articles[:max_results]
            
            # Save to file
            output_path = self._save_articles(all_articles)
            logger.success(f"Extracted {len(all_articles)} articles successfully")
            
            return output_path

        except Exception as e:
            logger.error(f"Error extracting articles: {e}")
            raise

    def _extract_articles_from_page(self) -> List[Dict[str, Any]]:
        """Extract article metadata from current page"""
        articles = []
        
        try:
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Find all article containers (try multiple patterns)
            article_containers = []
            
            # Pattern 1: li with search-result classes
            article_containers.extend(soup.find_all('li', class_=re.compile(r'search.*result', re.I)))
            
            # Pattern 2: div with result/article classes
            if not article_containers:
                article_containers.extend(soup.find_all('div', class_=re.compile(r'result.*item|article.*card', re.I)))
            
            # Pattern 3: Look for DOI links (most reliable for ACM)
            if not article_containers:
                doi_links = soup.find_all('a', href=re.compile(r'/doi/(10\.\d+/.+)'))
                for link in doi_links:
                    parent = link.find_parent(['li', 'div', 'article'])
                    if parent and parent not in article_containers:
                        article_containers.append(parent)
            
            logger.debug(f"Found {len(article_containers)} article containers")
            
            for container in article_containers:
                article = self._extract_article_data(container)
                if article:
                    articles.append(article)
            
            return articles

        except Exception as e:
            logger.error(f"Error extracting articles from page: {e}")
            return []

    def _extract_article_data(self, container) -> Optional[Dict[str, Any]]:
        """Extract metadata from single article container"""
        try:
            article = {}
            
            # Title
            title_elem = container.find(['h5', 'h4', 'h3', 'a'], class_=re.compile(r'title|hlFld-Title', re.I))
            if not title_elem:
                title_elem = container.find('a', href=re.compile(r'/doi/'))
            article['title'] = title_elem.get_text(strip=True) if title_elem else 'Unknown Title'
            
            # Authors
            authors_elem = container.find(['span', 'div'], class_=re.compile(r'author|hlFld-ContribAuthor', re.I))
            if authors_elem:
                # Extract all author names
                author_links = authors_elem.find_all('a')
                if author_links:
                    article['authors'] = [a.get_text(strip=True) for a in author_links]
                else:
                    article['authors'] = [authors_elem.get_text(strip=True)]
            else:
                article['authors'] = []
            
            # Year
            year_elem = container.find(['span', 'div'], class_=re.compile(r'date|year|publicationDate', re.I))
            if year_elem:
                year_text = year_elem.get_text(strip=True)
                year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                article['year'] = year_match.group(0) if year_match else year_text
            else:
                article['year'] = None
            
            # DOI
            doi_link = container.find('a', href=re.compile(r'/doi/(10\.\d+/.+)'))
            if doi_link:
                doi_match = re.search(r'10\.\d+/.+', doi_link['href'])
                article['doi'] = doi_match.group(0) if doi_match else None
                article['url'] = f"https://doi.org/{article['doi']}" if article['doi'] else None
            else:
                article['doi'] = None
                article['url'] = None
            
            # Abstract
            abstract_elem = container.find(['div', 'p'], class_=re.compile(r'abstract|snippet', re.I))
            article['abstract'] = abstract_elem.get_text(strip=True) if abstract_elem else ''
            
            # Citation count
            citation_elem = container.find(['span', 'div'], class_=re.compile(r'citation|metric', re.I))
            if citation_elem:
                citation_text = citation_elem.get_text(strip=True)
                citation_match = re.search(r'\d+', citation_text)
                article['citation_count'] = int(citation_match.group(0)) if citation_match else 0
            else:
                article['citation_count'] = 0
            
            # Source
            article['source'] = 'ACM Digital Library'
            article['database'] = 'acm'
            
            return article if article.get('title') and article['title'] != 'Unknown Title' else None

        except Exception as e:
            logger.debug(f"Error extracting article data: {e}")
            return None

    def _go_to_next_page(self) -> bool:
        """Navigate to next page of results"""
        try:
            # Look for next page button
            next_button = self.safe_find_element(
                By.CSS_SELECTOR,
                "a.pagination__btn--next, button.next-page, a[aria-label='Next page']"
            )
            
            if not next_button:
                # Try finding by text
                try:
                    next_button = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Next')] | //button[contains(text(), 'Next')]")
                except:
                    pass
            
            if next_button and next_button.is_enabled():
                self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                self.human_delay(0.5, 1)
                self.driver.execute_script("arguments[0].click();", next_button)
                self.human_delay(2, 4)
                return True
            
            return False

        except Exception as e:
            logger.debug(f"Could not navigate to next page: {e}")
            return False

    def _save_articles(self, articles: List[Dict[str, Any]]) -> Path:
        """Save extracted articles to JSON file"""
        import json
        
        output_file = self.download_dir / f"acm_articles_direct_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {output_file}")
        return output_file

    def scrape(self, query: str, filters: Optional[Dict[str, Any]] = None,
               format: str = 'bibtex', max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Complete scraping workflow

        Args:
            query: Search query
            filters: Optional filters
            format: Export format (not used in direct scraping)
            max_results: Maximum results

        Returns:
            List of article dictionaries
        """
        try:
            logger.info(f"Starting scrape workflow for query: '{query}'")
            
            # Start browser
            self.start_session()
            
            # Login (not required for ACM)
            self.login()
            
            # Search
            num_results = self.search(query, filters)
            
            if num_results == 0:
                logger.warning("No results found")
                return []
            
            # Extract articles
            output_file = self.download_results(format, max_results)
            
            # Load and return articles
            import json
            with open(output_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            logger.success(f"Scraping complete: {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            raise
        finally:
            self.close_session()
