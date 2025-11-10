"""
Simple test for ACM scraper - tests just the core functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# Setup logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_simple():
    """Simple test without full config"""
    
    logger.info("=" * 80)
    logger.info("Simple ACM Scraper Test")
    logger.info("=" * 80)
    
    # Create minimal config dict
    config = {
        'scraping': {
            'max_results': 5,
            'headless': False,
            'download_dir': str(Path('outputs/downloads').absolute())
        }
    }
    
    # Import after setting up path
    from src.scrapers.acm_scraper import ACMScraper
    
    try:
        # Create scraper
        logger.info("Creating ACM scraper...")
        scraper = ACMScraper(config, headless=False)
        
        # Test search
        query = "artificial intelligence"
        logger.info(f"Searching for: '{query}' (max 5 results)")
        
        articles = scraper.search(query, max_results=5)
        
        if not articles:
            logger.error("❌ No articles found!")
            scraper.close()
            return False
        
        logger.success(f"✅ Found {len(articles)} articles")
        
        # Show first article
        if articles:
            logger.info("\n📄 First article:")
            first = articles[0]
            logger.info(f"  Title: {first.get('title', 'N/A')[:80]}")
            logger.info(f"  Authors: {first.get('authors', 'N/A')}")
            logger.info(f"  Year: {first.get('year', 'N/A')}")
            logger.info(f"  DOI: {first.get('doi', 'N/A')}")
        
        # Test file generation
        logger.info("\n📝 Generating BibTeX file...")
        bibtex_file = scraper.download_results(format='bibtex')
        
        if bibtex_file and bibtex_file.exists():
            logger.success(f"✅ BibTeX file: {bibtex_file.name}")
            
            # Show first 10 lines
            content = bibtex_file.read_text(encoding='utf-8')
            lines = content.split('\n')[:10]
            logger.info("\n📖 BibTeX preview:")
            for line in lines:
                logger.info(f"   {line}")
        else:
            logger.error("❌ BibTeX file generation failed!")
        
        scraper.close()
        logger.success("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.exception("Full error:")
        return False

if __name__ == "__main__":
    success = test_simple()
    sys.exit(0 if success else 1)
