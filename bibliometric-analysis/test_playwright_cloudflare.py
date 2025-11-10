"""
Quick test: Can Playwright bypass Cloudflare on ACM?
"""

import sys
sys.path.insert(0, 'src')

from scrapers.playwright_manager import PlaywrightManager
from loguru import logger
import time

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
    level="INFO"
)

def test_cloudflare_bypass():
    """Test if Playwright can bypass Cloudflare on ACM."""
    
    print("="*80)
    print("🧪 PRUEBA: Playwright vs Cloudflare en ACM Digital Library")
    print("="*80)
    print()
    
    manager = None
    
    try:
        # Initialize Playwright (visible browser to see what's happening)
        logger.info("Inicializando Playwright (navegador visible)...")
        manager = PlaywrightManager(headless=False, timeout=30000)
        page = manager.start()
        
        logger.success("Playwright inicializado con stealth mode")
        print()
        
        # Navigate to ACM search
        logger.info("Navegando a ACM Digital Library...")
        url = "https://dl.acm.org/action/doSearch?fillQuickSearch=false&target=advanced&AllField=artificial+intelligence"
        manager.goto(url, wait_until='domcontentloaded')
        
        print()
        logger.info("Página cargada, verificando si hay Cloudflare...")
        time.sleep(2)
        
        # Check for Cloudflare
        content = page.content().lower()
        
        if 'cloudflare' in content or 'verifique que usted' in content or 'checking your browser' in content:
            logger.warning("⚠️  Cloudflare challenge detectado")
            logger.info("Esperando bypass automático (máximo 45 segundos)...")
            print()
            
            # Wait for bypass
            if manager.wait_for_cloudflare(max_wait=45):
                logger.success("✅ ¡Cloudflare bypassed exitosamente!")
                print()
                
                # Try to find result count
                logger.info("Buscando contador de resultados...")
                time.sleep(2)
                
                try:
                    # Check if we have results
                    final_content = page.content()
                    
                    if 'hitsLength' in final_content or 'search__item' in final_content:
                        logger.success("✅ ¡Página de resultados alcanzada!")
                        
                        # Try to extract result count
                        result_count = page.locator('span.hitsLength').first.text_content(timeout=5000)
                        logger.success(f"✅ Resultados encontrados: {result_count}")
                        
                        # Count articles
                        articles = page.locator('li.search__item').all()
                        logger.success(f"✅ Artículos en página: {len(articles)}")
                        
                        if len(articles) > 0:
                            # Extract first article title
                            first_title = articles[0].locator('h3').text_content(timeout=5000)
                            logger.info(f"Primer artículo: {first_title[:80]}...")
                        
                        print()
                        print("="*80)
                        print("✅ ÉXITO: Playwright puede bypassear Cloudflare en ACM")
                        print("="*80)
                        
                        # Take success screenshot
                        manager.screenshot("logs/acm_playwright_success.png", full_page=True)
                        logger.info("Captura guardada: logs/acm_playwright_success.png")
                        
                        return True
                        
                    else:
                        logger.warning("Página alcanzada pero sin resultados esperados")
                        manager.screenshot("logs/acm_playwright_unexpected.png", full_page=True)
                        return False
                        
                except Exception as e:
                    logger.error(f"Error extrayendo resultados: {e}")
                    manager.screenshot("logs/acm_playwright_error.png", full_page=True)
                    return False
                
            else:
                logger.error("❌ Cloudflare NO pudo ser bypassed")
                print()
                print("="*80)
                print("❌ FALLO: Cloudflare bloqueó el acceso después de 45s")
                print("="*80)
                return False
        
        else:
            logger.success("✅ ¡No hay Cloudflare challenge!")
            logger.info("Verificando resultados...")
            
            try:
                # Check for results
                result_count = page.locator('span.hitsLength').first.text_content(timeout=5000)
                logger.success(f"✅ Resultados: {result_count}")
                
                articles = page.locator('li.search__item').all()
                logger.success(f"✅ Artículos: {len(articles)}")
                
                print()
                print("="*80)
                print("✅ ÉXITO: Sin Cloudflare, acceso directo")
                print("="*80)
                
                return True
                
            except Exception as e:
                logger.error(f"Error: {e}")
                manager.screenshot("logs/acm_playwright_no_results.png", full_page=True)
                return False
    
    except Exception as e:
        logger.error(f"❌ Error en la prueba: {e}")
        if manager and manager.page:
            manager.screenshot("logs/acm_playwright_crash.png", full_page=True)
        return False
    
    finally:
        # Keep browser open for 5 seconds to see result
        if manager:
            logger.info("Esperando 5s antes de cerrar el navegador...")
            time.sleep(5)
            manager.close()

if __name__ == "__main__":
    success = test_cloudflare_bypass()
    sys.exit(0 if success else 1)
