"""
Script para inspeccionar la estructura actual de ACM Digital Library
y encontrar los selectores CSS correctos.

Uso:
    python -m src.scrapers.inspect_acm
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.utils.webdriver_manager import WebDriverManager
from loguru import logger
import time


def inspect_acm_page():
    """
    Inspecciona la página de ACM y muestra todos los elementos relevantes.
    """
    query = "generative artificial intelligence"
    
    logger.info("="*70)
    logger.info("INSPECTOR DE ACM DIGITAL LIBRARY")
    logger.info("="*70)
    
    manager = WebDriverManager(headless=False)  # headless=False para ver el navegador
    
    try:
        driver = manager.create_driver()
        
        # Navegar a ACM
        search_url = f"https://dl.acm.org/action/doSearch?AllField={query.replace(' ', '+')}"
        logger.info(f"Navegando a: {search_url}")
        driver.get(search_url)
        
        # Esperar a que cargue
        logger.info("Esperando 5 segundos para carga completa...")
        time.sleep(5)
        
        # Guardar HTML para inspección
        html = driver.page_source
        html_path = Path('logs/acm_page.html')
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.success(f"HTML guardado en {html_path}")
        
        # Tomar screenshot
        manager.take_screenshot('acm_search_page.png')
        logger.success("Screenshot guardado en logs/acm_search_page.png")
        
        # Buscar contador de resultados
        logger.info("\n" + "="*70)
        logger.info("BUSCANDO CONTADOR DE RESULTADOS")
        logger.info("="*70)
        
        result_selectors = [
            "span.hitsLength",
            "span.result__count",
            "div.search-result__count",
            "[class*='result-count']",
            "[class*='hits']",
            "span[class*='count']",
            ".search-result__count-value",
            ".result-count",
            "[data-test='search-result-count']",
            "h1 span",
            ".search-heading span"
        ]
        
        found_count = False
        for selector in result_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.success(f"✓ Encontrado con '{selector}': {len(elements)} elementos")
                    for i, elem in enumerate(elements[:3]):
                        if elem.is_displayed():
                            text = elem.text.strip()
                            if text:
                                logger.info(f"  [{i+1}] Texto: '{text}'")
                                logger.info(f"      Class: {elem.get_attribute('class')}")
                                logger.info(f"      HTML: {elem.get_attribute('outerHTML')[:120]}...")
                                found_count = True
            except Exception as e:
                logger.debug(f"Error con selector {selector}: {e}")
        
        if not found_count:
            logger.warning("⚠ No se encontró contador con selectores predefinidos")
        
        # Buscar en el texto de la página
        logger.info("\n--- Buscando en texto de página ---")
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            patterns = [
                r'(\d{1,3}(?:,\d{3})*)\s*(?:results?|items?)',
                r'showing.*?(\d{1,3}(?:,\d{3})*)',
                r'found\s*(\d{1,3}(?:,\d{3})*)'
            ]
            for pattern in patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    logger.success(f"✓ Pattern '{pattern}' encontró: {matches[:3]}")
        except Exception as e:
            logger.error(f"Error buscando en texto: {e}")
        
        # Buscar checkbox "Select All"
        logger.info("\n" + "="*70)
        logger.info("BUSCANDO SELECT ALL CHECKBOX")
        logger.info("="*70)
        
        checkbox_selectors = [
            "input[type='checkbox'][name='selectAll']",
            "input[type='checkbox'][id*='select']",
            "input.select-all",
            "label[for*='select-all']",
            "[class*='select-all']",
            "input[aria-label*='Select all']",
            "button[class*='select-all']",
            "input[type='checkbox'][class*='bulk']",
            ".bulk-select input",
            "[data-test='select-all']"
        ]
        
        found_checkbox = False
        for selector in checkbox_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.success(f"✓ Encontrado con '{selector}': {len(elements)} elementos")
                    for i, elem in enumerate(elements[:3]):
                        logger.info(f"  [{i+1}] Type: {elem.get_attribute('type')}")
                        logger.info(f"      ID: {elem.get_attribute('id')}")
                        logger.info(f"      Name: {elem.get_attribute('name')}")
                        logger.info(f"      Class: {elem.get_attribute('class')}")
                        logger.info(f"      HTML: {elem.get_attribute('outerHTML')[:120]}...")
                        found_checkbox = True
            except Exception as e:
                logger.debug(f"Error con selector {selector}: {e}")
        
        if not found_checkbox:
            logger.warning("⚠ No se encontró checkbox con selectores predefinidos")
        
        # Buscar botón Export
        logger.info("\n" + "="*70)
        logger.info("BUSCANDO BOTÓN EXPORT")
        logger.info("="*70)
        
        export_selectors = [
            "a[title*='Export']",
            "button[title*='Export']",
            "a[href*='export']",
            "button[class*='export']",
            "[aria-label*='Export']",
            "a.citation__export",
            "button.citation-export",
            "div[class*='export'] button",
            "li[class*='export'] a",
            "[data-test='export-button']",
            "button[data-action='export']"
        ]
        
        found_export = False
        for selector in export_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.success(f"✓ Encontrado con '{selector}': {len(elements)} elementos")
                    for i, elem in enumerate(elements[:3]):
                        if elem.is_displayed():
                            logger.info(f"  [{i+1}] Texto: '{elem.text}'")
                            logger.info(f"      Href: {elem.get_attribute('href')}")
                            logger.info(f"      Class: {elem.get_attribute('class')}")
                            logger.info(f"      HTML: {elem.get_attribute('outerHTML')[:120]}...")
                            found_export = True
            except Exception as e:
                logger.debug(f"Error con selector {selector}: {e}")
        
        if not found_export:
            logger.warning("⚠ No se encontró botón export con selectores predefinidos")
        
        # Listar todos los botones visibles
        logger.info("\n" + "="*70)
        logger.info("TODOS LOS BOTONES VISIBLES EN LA PÁGINA")
        logger.info("="*70)
        
        all_buttons = driver.find_elements(By.TAG_NAME, "button")
        visible_buttons = [btn for btn in all_buttons if btn.is_displayed()]
        logger.info(f"Total botones visibles: {len(visible_buttons)}")
        
        for i, btn in enumerate(visible_buttons[:15]):
            text = btn.text.strip()
            if text or btn.get_attribute('aria-label'):
                logger.info(f"[{i+1}] '{text or btn.get_attribute('aria-label')}'")
                logger.info(f"    Class: {btn.get_attribute('class')}")
                logger.info(f"    Type: {btn.get_attribute('type')}")
        
        # Listar todos los links relevantes
        logger.info("\n" + "="*70)
        logger.info("LINKS RELEVANTES (con 'export', 'download', 'citation')")
        logger.info("="*70)
        
        all_links = driver.find_elements(By.TAG_NAME, "a")
        relevant_keywords = ['export', 'download', 'citation', 'cite', 'bibtex']
        
        for link in all_links:
            href = link.get_attribute('href') or ""
            text = link.text.strip()
            classes = link.get_attribute('class') or ""
            
            if any(keyword in href.lower() or keyword in text.lower() or keyword in classes.lower() 
                   for keyword in relevant_keywords):
                if link.is_displayed():
                    logger.info(f"Link: '{text}'")
                    logger.info(f"  Href: {href[:80]}")
                    logger.info(f"  Class: {classes}")
        
        # Buscar elementos de resultado individuales
        logger.info("\n" + "="*70)
        logger.info("ELEMENTOS DE RESULTADO (artículos individuales)")
        logger.info("="*70)
        
        result_item_selectors = [
            "li.search__item",
            "div.issue-item",
            "article.search-result",
            "[class*='search-result']",
            "li[class*='result']",
            ".search-result-item",
            "[data-test='search-result']"
        ]
        
        for selector in result_item_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.success(f"✓ Encontrado con '{selector}': {len(elements)} elementos")
                    # Mostrar estructura del primer elemento
                    if len(elements) > 0:
                        first_elem = elements[0]
                        logger.info("  Estructura del primer elemento:")
                        logger.info(f"    Class: {first_elem.get_attribute('class')}")
                        
                        # Buscar título dentro
                        title_selectors = ["h5", "h3", ".title", "[class*='title']"]
                        for ts in title_selectors:
                            try:
                                title = first_elem.find_element(By.CSS_SELECTOR, ts)
                                logger.info(f"    Título ({ts}): {title.text[:60]}...")
                                break
                            except:
                                pass
                        
                        # Buscar autores
                        author_selectors = [".author", ".authors", "[class*='author']", ".contributors"]
                        for aus in author_selectors:
                            try:
                                authors = first_elem.find_element(By.CSS_SELECTOR, aus)
                                logger.info(f"    Autores ({aus}): {authors.text[:60]}...")
                                break
                            except:
                                pass
            except Exception as e:
                logger.debug(f"Error con selector {selector}: {e}")
        
        # Buscar DOIs/IDs
        logger.info("\n" + "="*70)
        logger.info("BUSCANDO DOIs EN LA PÁGINA")
        logger.info("="*70)
        
        doi_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/doi/']")
        logger.info(f"Total links con '/doi/': {len(doi_links)}")
        
        dois_found = []
        import re
        for link in doi_links[:10]:
            href = link.get_attribute('href')
            match = re.search(r'/doi/(10\.\d+/[^\s?]+)', href)
            if match:
                doi = match.group(1)
                dois_found.append(doi)
                logger.info(f"  DOI: {doi}")
        
        logger.success(f"Total DOIs únicos encontrados: {len(set(dois_found))}")
        
        # Esperar para inspección manual
        logger.info("\n" + "="*70)
        logger.info("INSPECCIÓN MANUAL")
        logger.info("="*70)
        logger.info("El navegador permanecerá abierto por 30 segundos.")
        logger.info("Usa las DevTools de Chrome (F12) para inspeccionar elementos.")
        logger.info("Presiona Ctrl+C para cerrar antes.")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Inspección interrumpida por el usuario")
        
        logger.success("\n" + "="*70)
        logger.success("INSPECCIÓN COMPLETADA")
        logger.success("="*70)
        logger.success("Archivos generados:")
        logger.success(f"  - logs/acm_page.html (código fuente)")
        logger.success(f"  - logs/acm_search_page.png (captura de pantalla)")
        
    except Exception as e:
        logger.error(f"Error durante inspección: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            manager.close()
            logger.info("WebDriver cerrado")
        except:
            pass


if __name__ == "__main__":
    inspect_acm_page()
