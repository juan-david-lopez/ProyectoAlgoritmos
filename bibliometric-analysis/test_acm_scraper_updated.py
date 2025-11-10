"""
Test script for updated ACM scraper with direct HTML scraping.
Tests the complete flow: search → extract → download → parse
"""

import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.scrapers.acm_scraper import ACMScraper
from src.utils.config_loader import get_config

# Setup logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/acm_test.log", rotation="10 MB")

def test_acm_scraper():
    """Test complete ACM scraper functionality"""
    
    logger.info("=" * 80)
    logger.info("Testing Updated ACM Scraper - Direct HTML Scraping")
    logger.info("=" * 80)
    
    try:
        # Load config
        config = get_config()
        
        # Create scraper (visible browser for testing)
        logger.info("Initializing ACM scraper (visible browser)...")
        scraper = ACMScraper(config, headless=False)
        
        # Test 1: Small query
        query = "artificial intelligence"
        max_results = 5
        
        logger.info(f"\n📋 TEST 1: Search and collect {max_results} articles")
        logger.info(f"Query: '{query}'")
        
        articles = scraper.search(query, max_results=max_results)
        
        if not articles:
            logger.error("❌ No articles found!")
            return False
        
        logger.success(f"✅ Found {len(articles)} articles")
        
        # Display first article
        if articles:
            logger.info("\n📄 First article details:")
            first = articles[0]
            logger.info(f"  Title: {first.get('title', 'N/A')}")
            logger.info(f"  Authors: {first.get('authors', 'N/A')}")
            logger.info(f"  Year: {first.get('year', 'N/A')}")
            logger.info(f"  DOI: {first.get('doi', 'N/A')}")
            logger.info(f"  URL: {first.get('url', 'N/A')}")
        
        # Test 2: Generate BibTeX file
        logger.info(f"\n📝 TEST 2: Generate BibTeX file")
        
        bibtex_file = scraper.download_results(format='bibtex')
        
        if not bibtex_file or not bibtex_file.exists():
            logger.error("❌ BibTeX file not generated!")
            return False
        
        logger.success(f"✅ BibTeX file created: {bibtex_file.name}")
        logger.info(f"   Path: {bibtex_file}")
        logger.info(f"   Size: {bibtex_file.stat().st_size} bytes")
        
        # Show first few lines
        content = bibtex_file.read_text(encoding='utf-8')
        lines = content.split('\n')[:15]
        logger.info("\n📖 First 15 lines of BibTeX file:")
        for line in lines:
            logger.info(f"   {line}")
        
        # Test 3: Generate JSON file
        logger.info(f"\n📝 TEST 3: Generate JSON file")
        
        json_file = scraper.download_results(format='json')
        
        if not json_file or not json_file.exists():
            logger.error("❌ JSON file not generated!")
            return False
        
        logger.success(f"✅ JSON file created: {json_file.name}")
        logger.info(f"   Path: {json_file}")
        logger.info(f"   Size: {json_file.stat().st_size} bytes")
        
        # Test 4: Generate CSV file
        logger.info(f"\n📝 TEST 4: Generate CSV file")
        
        csv_file = scraper.download_results(format='csv')
        
        if not csv_file or not csv_file.exists():
            logger.error("❌ CSV file not generated!")
            return False
        
        logger.success(f"✅ CSV file created: {csv_file.name}")
        logger.info(f"   Path: {csv_file}")
        logger.info(f"   Size: {csv_file.stat().st_size} bytes")
        
        # Test 5: Parse BibTeX file
        logger.info(f"\n📚 TEST 5: Parse BibTeX file")
        
        records = scraper.parse_file(bibtex_file)
        
        if not records:
            logger.error("❌ No records parsed!")
            return False
        
        logger.success(f"✅ Parsed {len(records)} records from BibTeX")
        
        if records:
            logger.info("\n📄 First parsed record:")
            first_record = records[0]
            logger.info(f"  ID: {first_record.get('id', 'N/A')}")
            logger.info(f"  Title: {first_record.get('title', 'N/A')[:60]}...")
            logger.info(f"  Authors: {first_record.get('authors', [])}")
            logger.info(f"  Year: {first_record.get('year', 'N/A')}")
            logger.info(f"  DOI: {first_record.get('doi', 'N/A')}")
        
        # Test 6: Convenience method scrape()
        logger.info(f"\n🚀 TEST 6: Test convenience method scrape()")
        
        # Create new scraper instance for clean test
        scraper2 = ACMScraper(config, headless=False)
        
        output_file = scraper2.scrape(
            query="machine learning",
            max_results=3,
            format='bibtex'
        )
        
        if not output_file or not output_file.exists():
            logger.error("❌ scrape() method failed!")
            return False
        
        logger.success(f"✅ scrape() method successful: {output_file.name}")
        
        # Cleanup
        logger.info("\n🧹 Closing browser...")
        scraper.close()
        scraper2.close()
        
        logger.info("\n" + "=" * 80)
        logger.success("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("\n📊 Summary:")
        logger.info(f"  ✅ Search & extraction working")
        logger.info(f"  ✅ BibTeX generation working")
        logger.info(f"  ✅ JSON generation working")
        logger.info(f"  ✅ CSV generation working")
        logger.info(f"  ✅ File parsing working")
        logger.info(f"  ✅ Convenience method working")
        logger.info("\n🎉 ACM scraper is fully operational!")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = test_acm_scraper()
    sys.exit(0 if success else 1)
