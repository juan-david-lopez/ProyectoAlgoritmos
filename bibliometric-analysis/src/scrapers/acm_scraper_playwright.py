"""
ACM Digital Library Scraper - Playwright Version
Uses institutional access via Uniquindío library to bypass Cloudflare
"""

import os
import sys
import json
import csv
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.playwright_manager import PlaywrightManager


class ACMScraperPlaywright:
    """
    ACM Digital Library scraper using Playwright with institutional access.
    Bypasses Cloudflare by accessing through university library portal.
    """
    
    # ACM URLs
    LIBRARY_URL = "https://library.uniquindio.edu.co/databases"
    ACM_BASE_URL = "https://dl.acm.org"
    ACM_SEARCH_URL = f"{ACM_BASE_URL}/action/doSearch"
    
    # Selectors (validated from inspector)
    RESULT_COUNT_SELECTOR = "span.hitsLength"
    ARTICLE_SELECTOR = "li.search__item"
    TITLE_SELECTOR = "h5.issue-item__title"
    AUTHORS_SELECTOR = "ul.rlist--inline"
    DOI_LINK_SELECTOR = "a[href*='/doi/']"
    YEAR_SELECTOR = "span.dot-separator"
    NEXT_PAGE_SELECTOR = "a.pagination__btn--next"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, headless: bool = False):
        """
        Initialize ACM scraper with Playwright.
        
        Args:
            config: Configuration dictionary (optional)
            headless: Run browser in headless mode
        """
        self.config = config or {}
        self.headless = headless
        self.playwright_manager = None
        self.page = None
        self.institutional_access_established = False
        
        logger.info("ACM Playwright Scraper initialized")
    
    def _establish_institutional_access(self):
        """
        Establish institutional access by visiting university library first.
        This sets up cookies/session that allow access to ACM without Cloudflare.
        """
        if self.institutional_access_established:
            logger.debug("Institutional access already established")
            return
        
        try:
            logger.info("🏛️ Establishing institutional access via Uniquindío library...")
            self.page.goto(self.LIBRARY_URL, wait_until="domcontentloaded")
            logger.success("✅ Library portal loaded")
            
            # Wait for session establishment
            logger.info("⏳ Waiting 3s for institutional session...")
            self.page.wait_for_timeout(3000)
            
            self.institutional_access_established = True
            logger.success("✅ Institutional access established")
            
        except Exception as e:
            logger.error(f"Failed to establish institutional access: {e}")
            raise
    
    def search(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search ACM Digital Library and extract article metadata.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of article dictionaries with metadata
        """
        try:
            # Initialize Playwright if not already done
            if not self.playwright_manager:
                logger.info("Initializing Playwright...")
                self.playwright_manager = PlaywrightManager(
                    headless=self.headless,
                    timeout=60000
                )
                self.page = self.playwright_manager.start()
            
            # Establish institutional access
            self._establish_institutional_access()
            
            # Build search URL
            search_url = f"{self.ACM_SEARCH_URL}?fillQuickSearch=false&target=advanced&AllField={query}"
            
            logger.info(f"🔍 Searching ACM: '{query}' (max: {max_results})")
            self.page.goto(search_url, wait_until="domcontentloaded")
            
            # Wait for results to load
            logger.info("⏳ Waiting for search results...")
            self.page.wait_for_timeout(3000)
            
            # DEBUG: Save screenshot and HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"logs/acm_search_{timestamp}.png"
            html_path = f"logs/acm_search_{timestamp}.html"
            
            os.makedirs("logs", exist_ok=True)
            self.page.screenshot(path=screenshot_path, full_page=True)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self.page.content())
            
            logger.info(f"📸 Screenshot saved: {screenshot_path}")
            logger.info(f"📄 HTML saved: {html_path}")
            
            # Get result count
            total_results = self._get_result_count()
            logger.info(f"📊 Found {total_results} total results")
            
            # Extract articles from multiple pages
            articles = []
            page_num = 1
            max_pages = (max_results // 20) + 1  # ACM shows ~20 results per page
            
            while len(articles) < max_results and page_num <= max_pages:
                logger.info(f"📄 Scraping page {page_num}...")
                
                page_articles = self._scrape_current_page()
                articles.extend(page_articles)
                
                logger.info(f"✅ Extracted {len(page_articles)} articles from page {page_num}")
                
                # Try to go to next page
                if len(articles) < max_results:
                    if not self._go_to_next_page():
                        logger.info("No more pages available")
                        break
                    page_num += 1
                else:
                    break
            
            # Trim to max_results
            articles = articles[:max_results]
            logger.success(f"✅ Scraping complete: {len(articles)} articles extracted")
            
            return articles
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _get_result_count(self) -> int:
        """Extract the total number of search results."""
        try:
            # Try primary selector
            count_elem = self.page.query_selector(self.RESULT_COUNT_SELECTOR)
            if count_elem:
                count_text = count_elem.inner_text()
                count = int(count_text.replace(',', ''))
                return count
            
            # Fallback: search in page text
            content = self.page.content()
            patterns = [
                r'(\d+(?:,\d+)*)\s+results?',
                r'results?\s+\((\d+(?:,\d+)*)\)',
                r'showing\s+\d+\s*-\s*\d+\s+of\s+(\d+(?:,\d+)*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    count = int(match.group(1).replace(',', ''))
                    logger.info(f"Result count extracted via regex: {count}")
                    return count
            
            logger.warning("Could not parse result count, returning 0")
            return 0
            
        except Exception as e:
            logger.error(f"Error getting result count: {e}")
            return 0
    
    def _scrape_current_page(self) -> List[Dict[str, Any]]:
        """Extract article metadata from current results page."""
        articles = []
        
        try:
            # Wait for articles to load
            self.page.wait_for_timeout(2000)
            
            # Find all article containers
            article_elements = self.page.query_selector_all(self.ARTICLE_SELECTOR)
            
            if not article_elements:
                logger.warning("No articles found with primary selector")
                # Try alternative selectors
                alt_selectors = [
                    "li.search-result",
                    "div.issue-item",
                    "article",
                    "div[class*='search']"
                ]
                
                for selector in alt_selectors:
                    article_elements = self.page.query_selector_all(selector)
                    if article_elements:
                        logger.info(f"Found articles with selector: {selector}")
                        break
            
            logger.debug(f"Found {len(article_elements)} article elements")
            
            # Extract metadata from each article
            for elem in article_elements:
                try:
                    article_data = self._extract_article_metadata(elem)
                    if article_data:
                        articles.append(article_data)
                except Exception as e:
                    logger.debug(f"Error extracting article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping page: {e}")
            return articles
    
    def _extract_article_metadata(self, element) -> Optional[Dict[str, Any]]:
        """Extract metadata from a single article element."""
        try:
            article = {}
            
            # Title
            title_elem = element.query_selector(self.TITLE_SELECTOR)
            if not title_elem:
                title_elem = element.query_selector("h3, h4, h5")
            if title_elem:
                article['title'] = title_elem.inner_text().strip()
            
            # Authors
            authors_elem = element.query_selector(self.AUTHORS_SELECTOR)
            if not authors_elem:
                authors_elem = element.query_selector("[class*='author']")
            if authors_elem:
                authors_text = authors_elem.inner_text()
                # Split by common separators
                authors = [a.strip() for a in re.split(r'[,;]|\sand\s', authors_text) if a.strip()]
                article['authors'] = authors
            
            # DOI/URL
            doi_elem = element.query_selector(self.DOI_LINK_SELECTOR)
            if doi_elem:
                doi_url = doi_elem.get_attribute('href')
                if doi_url:
                    if doi_url.startswith('/'):
                        doi_url = self.ACM_BASE_URL + doi_url
                    article['url'] = doi_url
                    
                    # Extract DOI from URL
                    doi_match = re.search(r'/doi/(?:abs/)?(\d+\.\d+/[\d.]+)', doi_url)
                    if doi_match:
                        article['doi'] = doi_match.group(1)
            
            # Year
            year_elem = element.query_selector(self.YEAR_SELECTOR)
            if year_elem:
                year_text = year_elem.inner_text()
                year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                if year_match:
                    article['year'] = int(year_match.group(0))
            
            # Source
            article['source'] = 'ACM Digital Library'
            article['scraped_at'] = datetime.now().isoformat()
            
            return article if 'title' in article else None
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
            return None
    
    def _go_to_next_page(self) -> bool:
        """Navigate to next page of results."""
        try:
            next_button = self.page.query_selector(self.NEXT_PAGE_SELECTOR)
            
            if not next_button:
                return False
            
            # Check if button is disabled
            if next_button.get_attribute('disabled'):
                return False
            
            # Click and wait
            next_button.click()
            self.page.wait_for_timeout(2000)
            
            return True
            
        except Exception as e:
            logger.debug(f"Error navigating to next page: {e}")
            return False
    
    def download_results(self, articles: List[Dict[str, Any]], 
                        output_dir: str = "output",
                        formats: List[str] = ["json", "bibtex", "csv"]) -> Dict[str, str]:
        """
        Save scraped articles to files in multiple formats.
        
        Args:
            articles: List of article dictionaries
            output_dir: Output directory path
            formats: List of formats to generate (json, bibtex, csv)
            
        Returns:
            Dictionary mapping format to file path
        """
        output_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # JSON format
            if "json" in formats:
                json_path = os.path.join(output_dir, f"acm_results_{timestamp}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(articles, f, indent=2, ensure_ascii=False)
                output_paths['json'] = json_path
                logger.info(f"✅ JSON saved: {json_path}")
            
            # BibTeX format
            if "bibtex" in formats:
                bibtex_path = os.path.join(output_dir, f"acm_results_{timestamp}.bib")
                self._convert_to_bibtex(articles, bibtex_path)
                output_paths['bibtex'] = bibtex_path
                logger.info(f"✅ BibTeX saved: {bibtex_path}")
            
            # CSV format
            if "csv" in formats:
                csv_path = os.path.join(output_dir, f"acm_results_{timestamp}.csv")
                self._convert_to_csv(articles, csv_path)
                output_paths['csv'] = csv_path
                logger.info(f"✅ CSV saved: {csv_path}")
            
            logger.success(f"✅ Results saved in {len(formats)} formats")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return output_paths
    
    def _convert_to_bibtex(self, articles: List[Dict[str, Any]], output_path: str):
        """Convert articles to BibTeX format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, article in enumerate(articles, 1):
                # Generate citation key
                first_author = article.get('authors', ['Unknown'])[0].split()[-1]
                year = article.get('year', 'YEAR')
                key = f"{first_author}{year}_{i}"
                
                # Write BibTeX entry
                f.write(f"@article{{{key},\n")
                f.write(f"  title = {{{article.get('title', 'Unknown')}}},\n")
                
                if 'authors' in article:
                    authors = ' and '.join(article['authors'])
                    f.write(f"  author = {{{authors}}},\n")
                
                if 'year' in article:
                    f.write(f"  year = {{{article['year']}}},\n")
                
                if 'doi' in article:
                    f.write(f"  doi = {{{article['doi']}}},\n")
                
                if 'url' in article:
                    f.write(f"  url = {{{article['url']}}},\n")
                
                f.write(f"  note = {{Source: ACM Digital Library}}\n")
                f.write("}\n\n")
    
    def _convert_to_csv(self, articles: List[Dict[str, Any]], output_path: str):
        """Convert articles to CSV format."""
        if not articles:
            return
        
        # Get all unique keys
        all_keys = set()
        for article in articles:
            all_keys.update(article.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for article in articles:
                # Convert lists to strings
                row = {}
                for key, value in article.items():
                    if isinstance(value, list):
                        row[key] = '; '.join(map(str, value))
                    else:
                        row[key] = value
                writer.writerow(row)
    
    def close(self):
        """Close Playwright browser."""
        if self.playwright_manager:
            logger.info("Closing Playwright browser...")
            self.playwright_manager.close()
            self.playwright_manager = None
            self.page = None
            self.institutional_access_established = False
            logger.info("✅ Browser closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Quick test
    logger.info("Testing ACM Playwright Scraper...")
    
    scraper = ACMScraperPlaywright(headless=False)
    try:
        results = scraper.search("artificial intelligence", max_results=5)
        print(f"\n✅ Found {len(results)} articles\n")
        
        for i, article in enumerate(results, 1):
            print(f"{i}. {article.get('title', 'No title')}")
            print(f"   Authors: {', '.join(article.get('authors', ['Unknown']))}")
            print(f"   Year: {article.get('year', 'N/A')}")
            print(f"   DOI: {article.get('doi', 'N/A')}\n")
        
        # Save results
        scraper.download_results(results, output_dir="output")
        
    finally:
        scraper.close()
