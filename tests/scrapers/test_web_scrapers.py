"""
Test de los scrapers web actualizados (ACM y ScienceDirect)
"""

import sys
from pathlib import Path

# Add bibliometric-analysis to path
sys.path.insert(0, str(Path(__file__).parent / 'bibliometric-analysis'))

from src.utils.config_loader import get_config
from src.scrapers.acm_scraper import ACMScraper
from src.scrapers.sciencedirect_scraper import ScienceDirectScraper
from loguru import logger

def test_acm_scraper():
    """Test ACM scraper con selectores actualizados"""
    logger.info("=" * 70)
    logger.info("Testing ACM Digital Library Scraper")
    logger.info("=" * 70)
    
    scraper = None
    try:
        # Load config
        import os
        os.chdir(Path(__file__).parent / 'bibliometric-analysis')
        config = get_config()
        
        # Initialize scraper
        scraper = ACMScraper(config, headless=False)  # No headless para ver qué pasa
        
        # Start browser session
        scraper.start_session()
        
        # Test search
        query = "generative artificial intelligence"
        logger.info(f"Searching for: '{query}'")
        
        num_results = scraper.search(query)
        
        if num_results > 0:
            logger.success(f"✓ ACM Search successful: {num_results} results found")
            return True
        else:
            logger.warning(f"⚠ ACM Search returned 0 results")
            return False
            
    except Exception as e:
        logger.error(f"✗ ACM Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if scraper:
            try:
                scraper.close_session()
            except:
                pass

def test_sciencedirect_scraper():
    """Test ScienceDirect scraper con selectores actualizados"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing ScienceDirect Scraper")
    logger.info("=" * 70)
    
    scraper = None
    try:
        # Load config
        import os
        os.chdir(Path(__file__).parent / 'bibliometric-analysis')
        config = get_config()
        
        # Initialize scraper
        scraper = ScienceDirectScraper(config, headless=False)  # No headless para ver qué pasa
        
        # Start browser session
        scraper.start_session()
        
        # Test search
        query = "generative artificial intelligence"
        logger.info(f"Searching for: '{query}'")
        
        num_results = scraper.search(query)
        
        if num_results > 0:
            logger.success(f"✓ ScienceDirect Search successful: {num_results} results found")
            return True
        else:
            logger.warning(f"⚠ ScienceDirect Search returned 0 results")
            return False
            
    except Exception as e:
        logger.error(f"✗ ScienceDirect Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if scraper:
            try:
                scraper.close_session()
            except:
                pass

if __name__ == "__main__":
    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info("║        TESTING UPDATED WEB SCRAPERS                               ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝\n")
    
    results = {
        'acm': False,
        'sciencedirect': False
    }
    
    # Test ACM
    results['acm'] = test_acm_scraper()
    
    # Test ScienceDirect
    results['sciencedirect'] = test_sciencedirect_scraper()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    if results['acm']:
        logger.success("✓ ACM Scraper: WORKING")
    else:
        logger.error("✗ ACM Scraper: FAILED")
    
    if results['sciencedirect']:
        logger.success("✓ ScienceDirect Scraper: WORKING")
    else:
        logger.error("✗ ScienceDirect Scraper: FAILED")
    
    logger.info("=" * 70)
    
    if all(results.values()):
        logger.success("\n🎉 ALL SCRAPERS WORKING!")
    elif any(results.values()):
        logger.warning("\n⚠️  SOME SCRAPERS WORKING (Semantic Scholar will be used as fallback for failed ones)")
    else:
        logger.error("\n❌ ALL SCRAPERS FAILED (Semantic Scholar API will be used as fallback)")
    
    logger.info("\nNote: Even if scrapers fail, the system will use Semantic Scholar API as backup.")
