"""
Playwright WebDriver Manager with Stealth and Anti-Detection
Provides better Cloudflare bypass capabilities than Selenium
"""

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from loguru import logger
import time
import random
from typing import Optional, Dict, Any


class PlaywrightManager:
    """
    Manages Playwright browser instances with stealth mode and anti-detection.
    Designed to bypass Cloudflare and other anti-bot protections.
    """
    
    def __init__(self, headless: bool = False, timeout: int = 30000):
        """
        Initialize Playwright manager.
        
        Args:
            headless: Run browser in headless mode
            timeout: Default timeout in milliseconds (default: 30s)
        """
        self.headless = headless
        self.timeout = timeout
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        logger.info(f"PlaywrightManager initialized (headless={headless})")
    
    def start(self) -> Page:
        """
        Start Playwright browser with stealth mode.
        
        Returns:
            Page: Playwright page object
        """
        try:
            logger.info("Starting Playwright browser...")
            
            # Launch Playwright
            self.playwright = sync_playwright().start()
            
            # Browser launch arguments for better anti-detection
            launch_args = [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--allow-running-insecure-content',
                '--disable-blink-features=AutomationControlled',
                '--disable-ipc-flooding-protection',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-background-timer-throttling',
                '--disable-hang-monitor',
                '--disable-client-side-phishing-detection',
                '--disable-popup-blocking',
                '--disable-prompt-on-repost',
                '--disable-sync',
                '--metrics-recording-only',
                '--no-first-run',
                '--safebrowsing-disable-auto-update',
                '--disable-component-update',
                '--ignore-certificate-errors'
            ]
            
            # Launch browser
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=launch_args
            )
            
            # Create context with realistic viewport and user agent
            self.context = self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},  # New York
                color_scheme='light',
                accept_downloads=True,
                ignore_https_errors=True
            )
            
            # Create page
            self.page = self.context.new_page()
            
            # Set default timeout
            self.page.set_default_timeout(self.timeout)
            
            # Apply stealth mode with custom evasion scripts
            logger.info("Applying anti-detection scripts...")
            self._inject_evasion_scripts()
            
            logger.success("Playwright browser started with stealth mode")
            return self.page
            
        except Exception as e:
            logger.error(f"Failed to start Playwright: {e}")
            raise
    
    def _inject_evasion_scripts(self):
        """Inject JavaScript to enhance evasion capabilities."""
        try:
            # Override navigator properties
            self.page.add_init_script("""
                // Override navigator.webdriver
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Override chrome property
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
            """)
            
            logger.debug("Evasion scripts injected")
            
        except Exception as e:
            logger.warning(f"Could not inject evasion scripts: {e}")
    
    def goto(self, url: str, wait_until: str = 'networkidle') -> None:
        """
        Navigate to URL with realistic behavior.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation succeeded
                       ('load', 'domcontentloaded', 'networkidle')
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        try:
            logger.info(f"Navigating to: {url}")
            
            # Navigate with realistic wait
            self.page.goto(url, wait_until=wait_until, timeout=self.timeout)
            
            # Add random human-like delay
            time.sleep(random.uniform(2, 4))
            
            # Scroll a bit to mimic human
            self._human_scroll()
            
            logger.success(f"Navigation complete: {url}")
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise
    
    def _human_scroll(self):
        """Perform human-like scrolling behavior."""
        try:
            # Random small scroll
            scroll_amount = random.randint(100, 500)
            self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            time.sleep(random.uniform(0.5, 1.5))
            
            # Scroll back
            self.page.evaluate(f"window.scrollBy(0, -{scroll_amount})")
            time.sleep(random.uniform(0.3, 0.8))
            
        except Exception as e:
            logger.debug(f"Scroll behavior failed: {e}")
    
    def wait_for_cloudflare(self, max_wait: int = 30) -> bool:
        """
        Wait for Cloudflare challenge to complete.
        
        Args:
            max_wait: Maximum seconds to wait
            
        Returns:
            True if bypassed, False if still challenged
        """
        if not self.page:
            return False
        
        logger.info(f"Checking for Cloudflare challenge (max {max_wait}s)...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Get page content
                content = self.page.content().lower()
                
                # Check for Cloudflare indicators
                cloudflare_indicators = [
                    'cloudflare',
                    'checking your browser',
                    'verifique que usted es un ser humano',
                    'ray id:',
                    'cf-browser-verification',
                    'challenge-form'
                ]
                
                has_challenge = any(indicator in content for indicator in cloudflare_indicators)
                
                if not has_challenge:
                    logger.success("Cloudflare challenge bypassed!")
                    return True
                
                # Log progress
                elapsed = int(time.time() - start_time)
                if elapsed % 5 == 0:  # Log every 5 seconds
                    logger.info(f"Still waiting for Cloudflare bypass... ({elapsed}s/{max_wait}s)")
                
                # Human-like behavior while waiting
                self._human_scroll()
                
                # Wait before checking again
                time.sleep(random.uniform(1.5, 2.5))
                
            except Exception as e:
                logger.debug(f"Error checking Cloudflare: {e}")
                time.sleep(2)
        
        logger.error(f"Cloudflare challenge not bypassed after {max_wait}s")
        
        # Save screenshot for debugging
        try:
            screenshot_path = "logs/cloudflare_playwright.png"
            self.page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"Screenshot saved: {screenshot_path}")
        except Exception as e:
            logger.warning(f"Could not save screenshot: {e}")
        
        return False
    
    def click(self, selector: str, timeout: Optional[int] = None) -> None:
        """Click element with human-like behavior."""
        if not self.page:
            raise RuntimeError("Browser not started")
        
        try:
            # Wait for element
            self.page.wait_for_selector(selector, timeout=timeout or self.timeout)
            
            # Random delay before click
            time.sleep(random.uniform(0.3, 0.8))
            
            # Click
            self.page.click(selector)
            
            # Random delay after click
            time.sleep(random.uniform(0.5, 1.2))
            
        except Exception as e:
            logger.error(f"Click failed on {selector}: {e}")
            raise
    
    def fill(self, selector: str, text: str, timeout: Optional[int] = None) -> None:
        """Fill input with human-like typing."""
        if not self.page:
            raise RuntimeError("Browser not started")
        
        try:
            # Wait for element
            self.page.wait_for_selector(selector, timeout=timeout or self.timeout)
            
            # Random delay before typing
            time.sleep(random.uniform(0.3, 0.8))
            
            # Type with delay between characters
            self.page.fill(selector, text)
            
            # Random delay after typing
            time.sleep(random.uniform(0.5, 1.2))
            
        except Exception as e:
            logger.error(f"Fill failed on {selector}: {e}")
            raise
    
    def get_content(self) -> str:
        """Get current page HTML content."""
        if not self.page:
            raise RuntimeError("Browser not started")
        return self.page.content()
    
    def screenshot(self, path: str, full_page: bool = False) -> None:
        """Take screenshot."""
        if not self.page:
            raise RuntimeError("Browser not started")
        
        try:
            self.page.screenshot(path=path, full_page=full_page)
            logger.info(f"Screenshot saved: {path}")
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
    
    def close(self):
        """Close browser and cleanup."""
        try:
            if self.page:
                self.page.close()
                self.page = None
            
            if self.context:
                self.context.close()
                self.context = None
            
            if self.browser:
                self.browser.close()
                self.browser = None
            
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
            
            logger.info("Playwright browser closed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
