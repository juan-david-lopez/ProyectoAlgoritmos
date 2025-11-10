"""
Simple test runner for PlaywrightManager.
This script starts Playwright, navigates to the ACM Digital Library home, waits for any Cloudflare challenge to be bypassed, and saves diagnostic outputs.

Usage (after installing dependencies):
    pip install -r requirements.txt
    playwright install
    python src/scrapers/playwright_test.py
"""

import os
import sys
import time
from loguru import logger

# Ensure project root (bibliometric-analysis) is on sys.path so imports work when script is
# executed directly from the project root or from inside the workspace.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scrapers.playwright_manager import PlaywrightManager


def main():
    logger.info("=" * 80)
    logger.info("🧪 PRUEBA PLAYWRIGHT: Biblioteca Uniquindío → ACM")
    logger.info("=" * 80)
    
    pm = PlaywrightManager(headless=False, timeout=60000)
    try:
        page = pm.start()
        
        # PASO 1: Primero visitar la biblioteca institucional
        logger.info("📚 PASO 1: Navegando a biblioteca Uniquindío...")
        library_url = "https://library.uniquindio.edu.co/databases"
        pm.goto(library_url, wait_until='networkidle')
        logger.success("✅ Página de biblioteca cargada")
        
        # Esperar para establecer cookies institucionales
        logger.info("⏳ Esperando 5s para establecer sesión institucional...")
        page.wait_for_timeout(5000)
        
        # PASO 2: Ahora navegar a ACM con contexto institucional
        logger.info(" PASO 2: Navegando a ACM Digital Library...")
        url = "https://dl.acm.org/"
        pm.goto(url, wait_until='networkidle')
        logger.success("Página ACM cargada")

        # Wait up to 60s for Cloudflare challenge (if present)
        logger.info("Verificando Cloudflare...")
        bypassed = pm.wait_for_cloudflare(max_wait=60)
        if bypassed:
            logger.success("Bypass successful — saving diagnostic outputs")
        else:
            logger.warning("Bypass not successful — saving diagnostic outputs")

        # Save page HTML and screenshot for inspection
        timestamp = int(time.time())
        html_path = f"logs/playwright_acm_{timestamp}.html"
        png_path = f"logs/playwright_acm_{timestamp}.png"

        try:
            html = pm.get_content()
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Saved HTML to: {html_path}")
        except Exception as e:
            logger.error(f"Failed to save HTML: {e}")

        try:
            pm.screenshot(png_path, full_page=True)
            logger.info(f"Saved screenshot to: {png_path}")
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        pm.close()
        logger.info("PlaywrightManager test finished")


if __name__ == '__main__':
    main()
