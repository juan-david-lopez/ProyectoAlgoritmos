"""
Test script for improved WebDriver anti-detection
Tests the enhanced configuration with ACM and ScienceDirect
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent / 'bibliometric-analysis'
sys.path.insert(0, str(project_root))

# Change to project directory
os.chdir(str(project_root))

from src.config.config_manager import ConfigManager
from src.scrapers.acm_scraper import ACMScraper
from src.scrapers.sciencedirect_scraper import ScienceDirectScraper
from loguru import logger


def test_detection():
    """Test if WebDriver is detected"""
    print("="*70)
    print("TEST 1: WebDriver Detection Check")
    print("="*70)
    
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Remove webdriver flag
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    # Test detection
    driver.get("https://bot.sannysoft.com/")
    
    input("\nPress Enter to close browser and continue...")
    driver.quit()
    
    print("✓ Detection test complete - check if 'webdriver' was detected\n")


def test_acm_improved():
    """Test improved ACM scraper"""
    print("="*70)
    print("TEST 2: ACM Digital Library with Improved WebDriver")
    print("="*70)
    
    try:
        config = ConfigManager()
        scraper = ACMScraper(config, headless=False)  # Visible for debugging
        
        print("\n1. Starting browser session...")
        scraper.start_session()
        
        print("2. Searching ACM...")
        num_results = scraper.search("machine learning", filters=None)
        
        print(f"\n✓ Search successful: {num_results} results found")
        
        input("\nPress Enter to close browser...")
        scraper.close_session()
        
        return True
        
    except Exception as e:
        print(f"\n✗ ACM test failed: {e}")
        if scraper and scraper.driver:
            scraper.close_session()
        return False


def test_sciencedirect_improved():
    """Test improved ScienceDirect scraper"""
    print("\n" + "="*70)
    print("TEST 3: ScienceDirect with Improved WebDriver")
    print("="*70)
    
    try:
        config = ConfigManager()
        scraper = ScienceDirectScraper(config, headless=False)  # Visible for debugging
        
        print("\n1. Starting browser session...")
        scraper.start_session()
        
        print("2. Searching ScienceDirect...")
        num_results = scraper.search("artificial intelligence", filters=None)
        
        print(f"\n✓ Search successful: {num_results} results found")
        
        input("\nPress Enter to close browser...")
        scraper.close_session()
        
        return True
        
    except Exception as e:
        print(f"\n✗ ScienceDirect test failed: {e}")
        if scraper and scraper.driver:
            scraper.close_session()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("IMPROVED WEBDRIVER ANTI-DETECTION TESTS")
    print("="*70)
    print("\nThese tests will:")
    print("1. Check if WebDriver is detected by bot detection sites")
    print("2. Test ACM scraper with improved configuration")
    print("3. Test ScienceDirect scraper with improved configuration")
    print("\nAll browsers will be VISIBLE so you can see what's happening.")
    print("="*70)
    
    input("\nPress Enter to start testing...")
    
    # Test 1: Detection check
    test_detection()
    
    # Test 2: ACM
    acm_success = test_acm_improved()
    
    # Test 3: ScienceDirect
    sd_success = test_sciencedirect_improved()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"ACM Scraper: {'✓ PASSED' if acm_success else '✗ FAILED'}")
    print(f"ScienceDirect Scraper: {'✓ PASSED' if sd_success else '✗ FAILED'}")
    print("="*70)
    
    if acm_success and sd_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The improved WebDriver configuration is working correctly.")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("The web scrapers may still be detected or blocked.")
        print("Consider using the Semantic Scholar API as primary source.")
    
    return 0 if (acm_success and sd_success) else 1


if __name__ == "__main__":
    sys.exit(main())
