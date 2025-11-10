"""
Script de depuración para verificar selectores en página ACM real
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from src.utils.webdriver_manager import WebDriverManager as WDManager

logger.remove()
logger.add(sys.stderr, level="INFO")

def debug_acm_page():
    """Debug ACM page selectors"""
    
    print("\n🔍 DEPURACIÓN DE SELECTORES ACM\n")
    
    # Inicializar driver
    wdm = WDManager(headless=False, download_dir="./outputs/downloads")
    driver = wdm.create_driver()
    
    try:
        # Navegar a búsqueda
        query = "artificial intelligence"
        from urllib.parse import quote
        encoded_query = quote(query)
        search_url = f"https://dl.acm.org/action/doSearch?AllField={encoded_query}"
        
        print(f"📍 Navegando a: {search_url}")
        driver.get(search_url)
        
        # Esperar carga
        time.sleep(5)
        
        # Manejar cookies
        try:
            cookie_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.CybotCookiebotDialogBodyButton"))
            )
            cookie_btn.click()
            print("✅ Cookie banner manejado")
            time.sleep(2)
        except:
            print("ℹ️ Sin cookie banner")
        
        # Probar selectores de contador
        print("\n🔍 Probando selectores de contador de resultados:\n")
        
        selectors = [
            "span.hitsLength",
            ".hitsLength",
            "span.result__count",
            ".result__count",
            "span[class*='hits']",
            "span[class*='result']",
            "span[class*='count']",
        ]
        
        for selector in selectors:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                print(f"✅ {selector:30s} → '{elem.text}'")
            except:
                print(f"❌ {selector:30s} → No encontrado")
        
        # Buscar cualquier span con texto de números
        print("\n🔍 Buscando spans con números:\n")
        spans = driver.find_elements(By.TAG_NAME, "span")
        for span in spans[:50]:  # Primeros 50
            text = span.text.strip()
            if text and any(c.isdigit() for c in text):
                classes = span.get_attribute("class") or "sin-clase"
                print(f"   <span class='{classes[:40]}'>  →  '{text[:50]}'")
        
        # Buscar en texto completo
        print("\n🔍 Buscando patrón de resultados en texto:\n")
        import re
        body_text = driver.find_element(By.TAG_NAME, "body").text
        matches = re.findall(r'(\d{1,3}(?:,\d{3})*)\s*(results?|artículos?)', body_text, re.IGNORECASE)
        for match in matches[:5]:
            print(f"   Encontrado: '{match[0]} {match[1]}'")
        
        # Buscar artículos
        print("\n🔍 Probando selectores de artículos:\n")
        
        article_selectors = [
            "li.search__item",
            ".search__item",
            "li[class*='search']",
            "li[class*='item']",
            "article",
            ".result-item",
        ]
        
        for selector in article_selectors:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, selector)
                print(f"✅ {selector:30s} → {len(elems)} elementos")
            except:
                print(f"❌ {selector:30s} → No encontrado")
        
        # Guardar HTML
        html_file = Path("logs/debug_acm_page.html")
        html_file.parent.mkdir(exist_ok=True)
        html_file.write_text(driver.page_source, encoding='utf-8')
        print(f"\n💾 HTML guardado en: {html_file}")
        
        # Guardar screenshot
        screenshot_file = Path("logs/debug_acm_page.png")
        driver.save_screenshot(str(screenshot_file))
        print(f"📸 Screenshot guardado en: {screenshot_file}")
        
        # Esperar para inspección manual
        print("\n⏳ Esperando 10 segundos para inspección manual...")
        time.sleep(10)
        
    finally:
        driver.quit()
        print("\n✅ Depuración completada\n")

if __name__ == "__main__":
    debug_acm_page()
