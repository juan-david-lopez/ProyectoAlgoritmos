"""
Universidad del Quindío - Portal Institucional Scraper AUTOMÁTICO
Acceso a bases de datos académicas a través del portal institucional
https://library.uniquindio.edu.co/databases

Basado en el patrón de scraper_sage.py
"""

import os
import time
import re
import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

# Configurar carpeta de descargas
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data", "raw", "uniquindio")
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Configurar opciones de Chrome
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": os.path.abspath(DOWNLOAD_FOLDER),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Comentar la siguiente línea para VER el navegador
# chrome_options.add_argument("--headless")


def scrape_acm_via_uniquindio(query="generative artificial intelligence", max_results=50):
    """
    Extrae artículos de ACM Digital Library a través del portal de Uniquindío
    
    Args:
        query: Término de búsqueda
        max_results: Número máximo de resultados a extraer
    """
    
    PORTAL_URL = "https://library.uniquindio.edu.co/databases"
    
    driver = webdriver.Chrome(options=chrome_options)
    all_articles = []
    
    try:
        print("\n" + "="*80)
        print("🎓 SCRAPER AUTOMÁTICO - PORTAL UNIQUINDÍO")
        print("="*80)
        print(f"🔍 Query: {query}")
        print(f"📊 Máximo de resultados: {max_results}")
        print("="*80 + "\n")
        
        # ====================
        # PASO 1: Acceder al portal
        # ====================
        print("📡 Paso 1: Accediendo al portal institucional...")
        driver.get(PORTAL_URL)
        time.sleep(5)
        
        # Scroll para cargar contenido
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(2)
        
        print("✅ Portal cargado exitosamente")
        
        # ====================
        # PASO 2: Buscar ACM Digital Library
        # ====================
        print("\n📚 Paso 2: Buscando ACM Digital Library...")
        
        acm_link = None
        search_patterns = [
            "ACM Digital Library",
            "ACM",
            "Association for Computing Machinery"
        ]
        
        for pattern in search_patterns:
            try:
                links = driver.find_elements(By.XPATH, 
                    f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{pattern.lower()}')]"
                )
                
                if links:
                    acm_link = links[0]
                    acm_url = acm_link.get_attribute("href")
                    print(f"✅ ACM encontrada: {pattern}")
                    print(f"   URL: {acm_url[:80]}...")
                    break
            except:
                continue
        
        if not acm_link:
            print("❌ No se pudo encontrar ACM Digital Library en el portal")
            return []
        
        # ====================
        # PASO 3: Acceder a ACM
        # ====================
        print("\n🌐 Paso 3: Accediendo a ACM Digital Library...")
        
        # Abrir en nueva pestaña
        main_window = driver.current_window_handle
        driver.execute_script(f"window.open('{acm_url}', '_blank');")
        time.sleep(3)
        
        # Cambiar a la nueva pestaña
        for handle in driver.window_handles:
            if handle != main_window:
                driver.switch_to.window(handle)
                break
        
        time.sleep(5)
        print(f"✅ ACM cargada: {driver.current_url[:80]}...")
        
        # ====================
        # PASO 4: Realizar búsqueda
        # ====================
        print(f"\n🔍 Paso 4: Buscando '{query}'...")
        
        # Buscar campo de búsqueda
        search_box = None
        search_selectors = [
            (By.NAME, "AllField"),
            (By.ID, "AllField"),
            (By.CSS_SELECTOR, "input[name='AllField']"),
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.XPATH, "//input[contains(@placeholder, 'Search')]")
        ]
        
        for by, selector in search_selectors:
            try:
                search_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de búsqueda encontrado")
                break
            except:
                continue
        
        if not search_box:
            print("❌ No se pudo encontrar el campo de búsqueda")
            return []
        
        # Escribir query
        search_box.clear()
        time.sleep(0.5)
        search_box.send_keys(query)
        print(f"✅ Query ingresada: '{query}'")
        time.sleep(1)
        
        # Presionar Enter o buscar botón
        try:
            search_button = driver.find_element(By.CSS_SELECTOR, "button.search__submit")
            driver.execute_script("arguments[0].click();", search_button)
            print("✅ Botón de búsqueda clickeado")
        except:
            search_box.send_keys(Keys.RETURN)
            print("✅ Enter presionado")
        
        time.sleep(5)
        
        # ====================
        # PASO 5: Extraer resultados
        # ====================
        print("\n📖 Paso 5: Extrayendo artículos...")
        
        # Scroll para cargar resultados
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(2)
        
        # Buscar elementos de artículos
        article_elements = []
        article_selectors = [
            "li.search__item",
            ".search-result-item",
            "[class*='search-item']"
        ]
        
        for selector in article_selectors:
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if article_elements:
                    print(f"✅ Encontrados {len(article_elements)} artículos (selector: {selector})")
                    break
            except:
                continue
        
        if not article_elements:
            print("⚠️ No se encontraron artículos con selectores estándar")
            # Intentar guardar screenshot para debug
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "debug_no_articles.png"))
            print(f"📸 Screenshot guardada en: {DOWNLOAD_FOLDER}/debug_no_articles.png")
            return []
        
        # Procesar artículos
        print(f"\n📚 Procesando {min(len(article_elements), max_results)} artículos...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                # Scroll hasta el artículo
                driver.execute_script("arguments[0].scrollIntoView(true);", article)
                time.sleep(0.3)
                
                # Extraer título
                title = ""
                title_selectors = ["h3", "h2", "[class*='title']", "a.search__item__title"]
                for sel in title_selectors:
                    try:
                        title_elem = article.find_element(By.CSS_SELECTOR, sel)
                        title = title_elem.text.strip()
                        if title:
                            break
                    except:
                        continue
                
                if not title:
                    continue
                
                # Extraer autores
                authors = []
                try:
                    author_elements = article.find_elements(By.CSS_SELECTOR, "[class*='author']")
                    for author in author_elements:
                        author_text = author.text.strip()
                        if author_text and len(author_text) > 2:
                            authors.append(author_text)
                except:
                    pass
                
                # Extraer DOI
                doi = ""
                try:
                    doi_link = article.find_element(By.CSS_SELECTOR, "a[href*='/doi/']")
                    doi_href = doi_link.get_attribute("href")
                    doi = doi_href.split("/doi/")[-1] if doi_href else ""
                except:
                    pass
                
                # Extraer año
                year = ""
                try:
                    date_selectors = ["[class*='date']", "[class*='year']", "time"]
                    for sel in date_selectors:
                        try:
                            date_elem = article.find_element(By.CSS_SELECTOR, sel)
                            year_match = re.search(r'\b(19|20)\d{2}\b', date_elem.text)
                            if year_match:
                                year = year_match.group(0)
                                break
                        except:
                            continue
                except:
                    pass
                
                # Extraer abstract
                abstract = ""
                try:
                    abstract_selectors = ["[class*='abstract']", ".snippet", ".description"]
                    for sel in abstract_selectors:
                        try:
                            abstract_elem = article.find_element(By.CSS_SELECTOR, sel)
                            abstract = abstract_elem.text.strip()
                            if abstract:
                                break
                        except:
                            continue
                except:
                    pass
                
                # Crear registro
                article_data = {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "year": year,
                    "source": "ACM Digital Library (Uniquindio)",
                    "abstract": abstract,
                    "keywords": [],
                    "url": f"https://dl.acm.org/doi/{doi}" if doi else "",
                    "extracted_date": datetime.now().isoformat()
                }
                
                all_articles.append(article_data)
                print(f"  {i}. {title[:70]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error en artículo {i}: {e}")
                continue
        
        print(f"\n✅ Total de artículos extraídos: {len(all_articles)}")
        
    except Exception as e:
        print(f"\n❌ Error general: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔚 Cerrando navegador...")
        driver.quit()
    
    return all_articles


def save_results(articles, output_filename="uniquindio_acm_results.json"):
    """Guarda los resultados en formato JSON"""
    
    if not articles:
        print("\n⚠️ No hay artículos para guardar")
        return None
    
    output_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    
    results = {
        "query": "generative artificial intelligence",
        "source": "ACM Digital Library via Uniquindio Portal",
        "extraction_date": datetime.now().isoformat(),
        "total_articles": len(articles),
        "articles": articles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Resultados guardados en: {output_path}")
    print(f"📊 Total de artículos: {len(articles)}")
    
    return output_path


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎓 SCRAPER AUTOMÁTICO - PORTAL UNIQUINDÍO")
    print("   ACM Digital Library")
    print("="*80)
    
    # Ejecutar scraping
    articles = scrape_acm_via_uniquindio(
        query="generative artificial intelligence",
        max_results=50
    )
    
    # Guardar resultados
    if articles:
        save_results(articles)
        
        print("\n" + "="*80)
        print("✅ EXTRACCIÓN COMPLETADA EXITOSAMENTE")
        print("="*80)
        print(f"\n📁 Archivos guardados en: {DOWNLOAD_FOLDER}")
        print("\n💡 Próximos pasos:")
        print("  1. Revisa los resultados en data/raw/uniquindio/")
        print("  2. Ejecuta el pipeline de unificación")
        print("  3. Realiza análisis de similitud")
    else:
        print("\n" + "="*80)
        print("⚠️ NO SE EXTRAJERON ARTÍCULOS")
        print("="*80)
        print("\n💡 Posibles causas:")
        print("  1. No estás en la red de Uniquindío (conecta VPN)")
        print("  2. ACM requiere autenticación adicional")
        print("  3. Los selectores CSS cambiaron")
        print("\n🔧 Soluciones:")
        print("  • Ejecuta sin --headless para ver qué pasa")
        print("  • Revisa el screenshot en data/raw/uniquindio/")
        print("  • Verifica tu acceso al portal manualmente")
