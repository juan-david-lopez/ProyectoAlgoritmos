"""
Test script for ACM Direct Scraper
Tests HTML extraction instead of export buttons
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'bibliometric-analysis'))

from src.config.config_manager import ConfigManager
from src.scrapers.acm_scraper_direct import ACMScraperDirect
from loguru import logger

def main():
    print("="*70)
    print("Testing ACM Direct Scraper (HTML Extraction)")
    print("="*70)
    
    try:
        # Initialize
        config = ConfigManager()
        scraper = ACMScraperDirect(config, headless=False)  # Visible browser
        
        print("\n1. Starting browser session...")
        scraper.start_session()
        
        print("2. Searching for 'machine learning'...")
        num_results = scraper.search("machine learning")
        print(f"   Found: {num_results} results")
        
        print("\n3. Extracting articles from HTML...")
        articles = []
        for i in range(2):  # Extract from 2 pages
            page_articles = scraper._extract_articles_from_page()
            articles.extend(page_articles)
            print(f"   Page {i+1}: {len(page_articles)} articles")
            
            if not scraper._go_to_next_page():
                break
        
        print(f"\n4. Total extracted: {len(articles)} articles")
        
        if articles:
            print("\n5. Sample article:")
            sample = articles[0]
            print(f"   Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"   Authors: {', '.join(sample.get('authors', [])[:3])}")
            print(f"   Year: {sample.get('year', 'N/A')}")
            print(f"   DOI: {sample.get('doi', 'N/A')}")
            print(f"   Citations: {sample.get('citation_count', 0)}")
        
        print("\n" + "="*70)
        print("TEST SUCCESSFUL!")
        print("="*70)
        
        input("\nPress Enter to close browser...")
        scraper.close_session()
        
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        if 'scraper' in locals() and scraper.driver:
            input("\nPress Enter to close browser...")
            scraper.close_session()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
