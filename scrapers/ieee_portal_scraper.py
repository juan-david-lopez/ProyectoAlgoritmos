"""
IEEE Xplore Scraper - Portal Institucional Universidad del Quindío
Extrae artículos de IEEE sin necesidad de API key usando acceso institucional
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from dotenv import load_dotenv

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Cargar variables de entorno
load_dotenv()

# Configuración
EMAIL = os.getenv("UNIQUINDIO_EMAIL", "juand.lopezh@uqvirtual.edu.co")
PASSWORD = os.getenv("UNIQUINDIO_PASSWORD", "")

DOWNLOAD_FOLDER = Path(__file__).parent / "data" / "raw" / "ieee"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Configurar Chrome
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": str(DOWNLOAD_FOLDER.absolute()),
    "download.prompt_for_download": False,
})
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)
# Comentar para ver el navegador
# chrome_options.add_argument("--headless")


def wait_for_page_load(driver, timeout=10):
    """Espera a que la página se cargue completamente"""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(2)
    except TimeoutException:
        print("⚠️ Timeout esperando carga de página")


def google_login(driver, email, password):
    """
    Realiza login con Google (usado por el portal institucional)
    """
    print("\n🔐 Iniciando login con Google...")
    
    try:
        # Buscar botón de Google
        google_button_selectors = [
            (By.ID, "btn-google"),
            (By.CLASS_NAME, "btn-google"),
            (By.XPATH, "//button[contains(@class, 'google')]"),
            (By.XPATH, "//button[contains(text(), 'Google')]"),
        ]
        
        google_button = None
        for by, selector in google_button_selectors:
            try:
                google_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((by, selector))
                )
                print(f"✅ Botón de Google encontrado: {selector}")
                break
            except:
                continue
        
        if not google_button:
            print("❌ No se encontró botón de Google")
            return False
        
        # Guardar ventana principal
        main_window = driver.current_window_handle
        
        # Click en botón de Google
        google_button.click()
        print("⏳ Esperando ventana de login de Google...")
        time.sleep(3)
        
        # Cambiar a ventana de Google
        for handle in driver.window_handles:
            if handle != main_window:
                driver.switch_to.window(handle)
                break
        
        print(f"📍 URL del login: {driver.current_url}")
        time.sleep(2)
        
        # Ingresar email
        print("📧 Ingresando email...")
        email_selectors = [
            (By.ID, "identifierId"),
            (By.NAME, "identifier"),
            (By.XPATH, "//input[@type='email']"),
        ]
        
        email_input = None
        for by, selector in email_selectors:
            try:
                email_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                break
            except:
                continue
        
        if not email_input:
            print("❌ No se encontró campo de email")
            return False
        
        email_input.clear()
        email_input.send_keys(email)
        email_input.send_keys(Keys.RETURN)
        print(f"✅ Email ingresado: {email}")
        
        # Esperar y ingresar contraseña
        print("⏳ Esperando campo de contraseña...")
        time.sleep(5)
        
        password_input = None
        password_selectors = [
            (By.NAME, "Passwd"),
            (By.NAME, "password"),
            (By.XPATH, "//input[@type='password']"),
        ]
        
        for by, selector in password_selectors:
            try:
                password_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                break
            except:
                continue
        
        if not password_input:
            print("❌ No se encontró campo de contraseña")
            return False
        
        password_input.clear()
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)
        print("✅ Contraseña ingresada")
        
        # Esperar cierre de ventana de login
        print("⏳ Esperando completar login...")
        time.sleep(5)
        
        # Verificar si la ventana se cerró
        for i in range(30):
            if len(driver.window_handles) == 1:
                print(f"✅ Login completado (ventana cerrada después de {i+1}s)")
                driver.switch_to.window(main_window)
                return True
            time.sleep(1)
        
        # Si no se cerró, cambiar manualmente
        driver.switch_to.window(main_window)
        print("✅ Login completado (cambiado manualmente a ventana principal)")
        return True
        
    except Exception as e:
        print(f"❌ Error en login: {str(e)}")
        return False


def scrape_ieee_portal(query="generative artificial intelligence", max_results=50):
    """
    Scraper de IEEE Xplore usando portal institucional
    """
    print("="*80)
    print("🎓 IEEE XPLORE SCRAPER - PORTAL UNIQUINDÍO")
    print("="*80)
    print(f"🔍 Query: {query}")
    print(f"📊 Máx. resultados: {max_results}")
    print("="*80)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    articles = []
    
    try:
        # Paso 1: Acceder al portal institucional de IEEE
        print("\n📡 Paso 1: Accediendo al portal institucional...")
        ieee_portal_url = "https://ieeexplore-ieee-org.crai.referencistas.com/"
        driver.get(ieee_portal_url)
        wait_for_page_load(driver)
        print(f"✅ Portal cargado")
        print(f"📍 URL actual: {driver.current_url}")
        
        # Verificar si necesita autenticación
        if "login" in driver.current_url.lower() or "intelproxy" in driver.current_url:
            print("\n🔐 Se requiere autenticación...")
            
            if not PASSWORD:
                print("⚠️ No se encontró contraseña en variables de entorno")
                print("💡 Por favor, inicia sesión manualmente en el navegador")
                print("⏳ Esperando 60 segundos...")
                time.sleep(60)
            else:
                login_success = google_login(driver, EMAIL, PASSWORD)
                if login_success:
                    print("✅ Login completado, esperando redirección...")
                    time.sleep(5)
                else:
                    print("⚠️ Login falló, intenta manualmente")
                    time.sleep(30)
        
        # Paso 2: Buscar en IEEE
        print(f"\n🔍 Paso 2: Buscando '{query}' en IEEE Xplore...")
        print(f"📍 URL actual: {driver.current_url}")
        
        # Buscar campo de búsqueda
        search_selectors = [
            (By.ID, "xplGlobalSearchInput"),
            (By.NAME, "queryText"),
            (By.CSS_SELECTOR, "input[placeholder*='Search']"),
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.XPATH, "//input[@aria-label='Search IEEE Xplore']"),
        ]
        
        search_input = None
        for by, selector in search_selectors:
            try:
                search_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de búsqueda encontrado: {selector}")
                break
            except:
                continue
        
        if not search_input:
            print("❌ No se encontró campo de búsqueda")
            print("📸 Guardando captura...")
            driver.save_screenshot(str(DOWNLOAD_FOLDER / "ieee_no_search_field.png"))
            return []
        
        # Ejecutar búsqueda
        search_input.clear()
        search_input.send_keys(query)
        search_input.send_keys(Keys.RETURN)
        print("✅ Búsqueda ejecutada")
        
        # Esperar resultados
        print("⏳ Esperando resultados...")
        time.sleep(5)
        print(f"📍 URL de resultados: {driver.current_url}")
        
        # Paso 3: Extraer artículos
        print("\n📖 Paso 3: Extrayendo artículos...")
        
        # Scroll para cargar más resultados
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(2)
        
        # Buscar elementos de resultados
        result_selectors = [
            "xpl-results-item",
            ".List-results-items .result-item",
            "[class*='result-item']",
            ".document-type-article",
            "div[class*='search-result']",
        ]
        
        article_elements = []
        for selector in result_selectors:
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if article_elements:
                    print(f"✅ Encontrados {len(article_elements)} resultados con selector: {selector}")
                    break
            except:
                continue
        
        if not article_elements:
            print("⚠️ No se encontraron artículos")
            print("📸 Guardando captura para análisis...")
            driver.save_screenshot(str(DOWNLOAD_FOLDER / "ieee_no_results.png"))
            
            # Guardar HTML para análisis
            with open(DOWNLOAD_FOLDER / "ieee_page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("💾 HTML guardado para análisis")
            return []
        
        # Procesar artículos
        print(f"\n📚 Procesando {min(len(article_elements), max_results)} artículos...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", article)
                time.sleep(0.3)
                
                # Título
                title = ""
                title_selectors = [
                    "h3.result-item-title",
                    "h2[class*='title']",
                    "a.document-title",
                    "[class*='article-title']",
                ]
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
                
                # Autores
                authors = []
                try:
                    author_elems = article.find_elements(By.CSS_SELECTOR, "[class*='author']")
                    authors = [a.text.strip() for a in author_elems if a.text.strip()]
                except:
                    pass
                
                # Año
                year = ""
                try:
                    import re
                    year_elem = article.find_element(By.CSS_SELECTOR, "[class*='date'], [class*='year']")
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_elem.text)
                    if year_match:
                        year = year_match.group(0)
                except:
                    pass
                
                # DOI/URL
                doi = ""
                url = ""
                try:
                    link_elem = article.find_element(By.CSS_SELECTOR, "a[href*='/document/']")
                    url = link_elem.get_attribute("href")
                    if "/document/" in url:
                        doc_id = url.split("/document/")[-1].split("/")[0]
                        doi = f"10.1109/{doc_id}"
                except:
                    pass
                
                # Abstract
                abstract = ""
                try:
                    abstract_elem = article.find_element(By.CSS_SELECTOR, "[class*='abstract'], [class*='description']")
                    abstract = abstract_elem.text.strip()
                except:
                    pass
                
                # Venue (conference/journal)
                venue = ""
                try:
                    venue_elem = article.find_element(By.CSS_SELECTOR, "[class*='publication'], [class*='venue']")
                    venue = venue_elem.text.strip()
                except:
                    pass
                
                article_data = {
                    "id": f"ieee_{i}",
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "url": url,
                    "abstract": abstract,
                    "source": "ieee",
                    "publication_venue": venue,
                    "keywords": [],
                    "extracted_date": datetime.now().isoformat()
                }
                
                articles.append(article_data)
                print(f"  {i}. {title[:65]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error procesando artículo {i}: {str(e)}")
                continue
        
        print(f"\n✅ Total extraído: {len(articles)} artículos")
        
    except Exception as e:
        print(f"❌ Error general: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n⏳ Cerrando navegador en 5 segundos...")
        time.sleep(5)
        driver.quit()
    
    # Guardar resultados
    if articles:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = DOWNLOAD_FOLDER / f"ieee_portal_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados: {output_file}")
        print(f"📊 Total de artículos: {len(articles)}")
        print(f"📁 Ruta: {output_file.absolute()}")
    
    return articles


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 INICIANDO SCRAPER DE IEEE XPLORE")
    print("="*80)
    
    articles = scrape_ieee_portal(
        query="generative artificial intelligence",
        max_results=50
    )
    
    if articles:
        print("\n" + "="*80)
        print("✅ SCRAPING COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"\n📊 Resumen:")
        print(f"   • Total de artículos: {len(articles)}")
        print(f"   • Con abstract: {sum(1 for a in articles if a.get('abstract'))}")
        print(f"   • Con DOI: {sum(1 for a in articles if a.get('doi'))}")
        print(f"   • Con autores: {sum(1 for a in articles if a.get('authors'))}")
    else:
        print("\n" + "="*80)
        print("⚠️ NO SE PUDIERON EXTRAER ARTÍCULOS")
        print("="*80)
        print("\n💡 Posibles soluciones:")
        print("   1. Verifica las credenciales en el archivo .env")
        print("   2. Intenta ejecutar con el navegador visible (comenta headless)")
        print("   3. Revisa las capturas guardadas en data/raw/ieee/")
        print("   4. Considera usar la API de IEEE con API key gratuita")
