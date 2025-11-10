"""
Semantic Scholar API Scraper
Alternative to web scraping - uses official API
More reliable and doesn't break with website updates
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime


class SemanticScholarAPI:
    """
    Scraper using Semantic Scholar API
    
    Features:
    - Reliable API-based data collection
    - No web scraping issues
    - Rich metadata including citations, authors, venues
    - Rate limiting handled automatically
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SEARCH_URL = f"{BASE_URL}/paper/search"
    
    def __init__(self, config=None, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar API scraper
        
        Args:
            config: Configuration object (optional)
            api_key: API key for higher rate limits (optional)
        """
        self.config = config
        
        # Try to get API key from multiple sources
        if api_key is None:
            # Try environment variable
            api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
            
            # Try config if available
            if api_key is None and config:
                api_key = config.get_env('SEMANTIC_SCHOLAR_API_KEY')
        
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if api_key:
            self.session.headers['x-api-key'] = api_key
            logger.info("Semantic Scholar API scraper initialized with API key")
        else:
            logger.info("Semantic Scholar API scraper initialized (no API key - rate limited)")
            logger.warning("Consider getting an API key for better performance. See SEMANTIC_SCHOLAR_API_KEY.md")
    
    def search(self, query: str, max_results: int = 100, 
               fields: Optional[List[str]] = None,
               year_start: Optional[int] = None,
               year_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar API
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            fields: Fields to retrieve (default: comprehensive set)
            year_start: Start year filter
            year_end: End year filter
            
        Returns:
            List of paper metadata dictionaries
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors',
                'venue', 'citationCount', 'referenceCount',
                'fieldsOfStudy', 'publicationDate', 'journal'
            ]
        
        papers = []
        offset = 0
        limit = 100  # Max per request
        
        logger.info(f"Searching Semantic Scholar for: '{query}'")
        logger.info(f"Target results: {max_results}")
        
        while len(papers) < max_results:
            try:
                # Build request parameters
                params = {
                    'query': query,
                    'offset': offset,
                    'limit': min(limit, max_results - len(papers)),
                    'fields': ','.join(fields)
                }
                
                # Add year filters if provided
                # Note: Semantic Scholar API uses 'year' parameter with format like '2020-2023'
                # But needs both start and end, so we'll handle it differently
                if year_start or year_end:
                    # If only start provided, set end to current year
                    if year_start and not year_end:
                        from datetime import datetime
                        year_end = datetime.now().year
                    # If only end provided, set start to a reasonable past year
                    elif year_end and not year_start:
                        year_start = year_end - 10
                    
                    # Now set the year range
                    if year_start and year_end:
                        params['year'] = f"{year_start}-{year_end}"
                
                # Make request
                logger.debug(f"Request: offset={offset}, limit={params['limit']}")
                response = self.session.get(self.SEARCH_URL, params=params)
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Extract papers
                batch = data.get('data', [])
                if not batch:
                    logger.info("No more results available")
                    break
                
                papers.extend(batch)
                offset += len(batch)
                
                logger.info(f"Retrieved {len(papers)}/{max_results} papers")
                
                # Check if we have all available results
                total = data.get('total', 0)
                if offset >= total:
                    logger.info(f"Retrieved all {total} available results")
                    break
                
                # Rate limiting: be nice to the API
                # Wait longer to avoid hitting rate limits
                time.sleep(3)
                
            except requests.RequestException as e:
                logger.error(f"API request failed: {e}")
                if len(papers) > 0:
                    logger.warning(f"Returning {len(papers)} papers collected so far")
                    break
                else:
                    raise
        
        logger.info(f"Search complete: {len(papers)} papers retrieved")
        return papers[:max_results]
    
    def get_paper_details(self, paper_id: str, 
                         fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific paper
        
        Args:
            paper_id: Semantic Scholar paper ID
            fields: Fields to retrieve
            
        Returns:
            Paper metadata dictionary
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors',
                'venue', 'citationCount', 'referenceCount',
                'fieldsOfStudy', 'publicationDate', 'journal'
            ]
        
        try:
            url = f"{self.BASE_URL}/paper/{paper_id}"
            params = {'fields': ','.join(fields)}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"Failed to get paper details: {e}")
            return None
    
    def convert_to_unified_format(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Semantic Scholar format to unified format
        
        Args:
            papers: List of papers from API
            
        Returns:
            List of papers in unified format
        """
        unified_papers = []
        
        for paper in papers:
            try:
                # Extract author information
                authors = []
                author_ids = []
                for author in paper.get('authors', []):
                    name = author.get('name', 'Unknown')
                    authors.append(name)
                    if 'authorId' in author:
                        author_ids.append(author['authorId'])
                
                # Extract journal name
                journal_info = paper.get('journal', {})
                journal_name = ''
                if isinstance(journal_info, dict):
                    journal_name = journal_info.get('name', paper.get('venue', ''))
                else:
                    journal_name = paper.get('venue', '')
                
                # Build unified record
                unified = {
                    'id': paper.get('paperId', ''),
                    'title': paper.get('title', ''),
                    'authors': authors,
                    'author_ids': author_ids,
                    'year': paper.get('year'),
                    'venue': paper.get('venue', ''),
                    'journal': journal_name,
                    'abstract': paper.get('abstract', ''),
                    'doi': '',  # Not available in basic fields
                    'url': '',  # Not available in basic fields
                    'citation_count': paper.get('citationCount', 0),
                    'reference_count': paper.get('referenceCount', 0),
                    'fields_of_study': paper.get('fieldsOfStudy', []),
                    'publication_types': [],  # Not available in basic fields
                    'publication_date': paper.get('publicationDate', ''),
                    'external_ids': {},  # Not available in basic fields
                    'source': 'semantic_scholar'
                }
                
                unified_papers.append(unified)
                
            except Exception as e:
                logger.warning(f"Error converting paper: {e}")
                continue
        
        return unified_papers
    
    def save_to_json(self, papers: List[Dict[str, Any]], 
                     output_path: Path) -> bool:
        """
        Save papers to JSON file
        
        Args:
            papers: List of papers
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(papers)} papers to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save papers: {e}")
            return False
    
    def scrape(self, query: str, max_results: int = 100,
               year_start: Optional[int] = None,
               year_end: Optional[int] = None,
               output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Complete scraping workflow
        
        Args:
            query: Search query
            max_results: Maximum results to retrieve
            year_start: Start year filter
            year_end: End year filter
            output_path: Optional output file path
            
        Returns:
            List of papers in unified format
        """
        logger.info(f"Starting Semantic Scholar scrape for: '{query}'")
        
        # Search for papers
        papers = self.search(query, max_results, year_start=year_start, year_end=year_end)
        
        if not papers:
            logger.warning("No papers found")
            return []
        
        # Convert to unified format
        unified_papers = self.convert_to_unified_format(papers)
        
        # Save if output path provided
        if output_path:
            self.save_to_json(unified_papers, output_path)
        
        logger.info(f"Scraping complete: {len(unified_papers)} papers")
        return unified_papers


def main():
    """Demo usage"""
    from loguru import logger
    
    # Initialize scraper
    scraper = SemanticScholarAPI()
    
    # Search for papers
    query = "generative artificial intelligence"
    papers = scraper.scrape(
        query=query,
        max_results=50,
        year_start=2020,
        output_path=Path("data/semantic_scholar_results.json")
    )
    
    # Display results
    logger.info(f"\nRetrieved {len(papers)} papers")
    for i, paper in enumerate(papers[:5], 1):
        logger.info(f"\n{i}. {paper['title']}")
        logger.info(f"   Authors: {', '.join(paper['authors'][:3])}")
        logger.info(f"   Year: {paper['year']}")
        logger.info(f"   Citations: {paper['citation_count']}")


if __name__ == "__main__":
    main()
