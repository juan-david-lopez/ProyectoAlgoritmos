"""
Test del nuevo scraper de Semantic Scholar
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'bibliometric-analysis'))

from src.scrapers.semantic_scholar_api import SemanticScholarAPI
from loguru import logger

def test_semantic_scholar():
    """Test básico del scraper"""
    
    logger.info("=" * 70)
    logger.info("Testing Semantic Scholar API Scraper")
    logger.info("=" * 70)
    
    # Initialize scraper
    scraper = SemanticScholarAPI()
    
    # Test search
    query = "generative artificial intelligence"
    logger.info(f"Testing search for: '{query}'")
    
    try:
        papers = scraper.scrape(
            query=query,
            max_results=10  # Small test
        )
        
        logger.info(f"\n✓ Success! Retrieved {len(papers)} papers")
        
        if papers:
            logger.info("\n" + "=" * 70)
            logger.info("Sample Results:")
            logger.info("=" * 70)
            
            for i, paper in enumerate(papers[:3], 1):
                logger.info(f"\n{i}. {paper['title']}")
                logger.info(f"   Authors: {', '.join(paper['authors'][:3])}")
                logger.info(f"   Year: {paper['year']}")
                logger.info(f"   Citations: {paper['citation_count']}")
                logger.info(f"   Venue: {paper['venue']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_semantic_scholar()
    
    if success:
        logger.success("\n" + "=" * 70)
        logger.success("✓ ALL TESTS PASSED")
        logger.success("=" * 70)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("✗ TESTS FAILED")
        logger.error("=" * 70)
