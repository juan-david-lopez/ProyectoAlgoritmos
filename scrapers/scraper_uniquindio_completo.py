"""
Scraper Automático - Portal Institucional Universidad del Quindío
Extrae artículos de ACM Digital Library, IEEE Xplore y ScienceDirect
a través del portal institucional

Basado en los scrapers funcionales de IEEE y ScienceDirect
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# Configurar carpeta de descargas
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data", "raw", "uniquindio")
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Configurar Chrome
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

# COMENTAR ESTA LÍNEA PARA VER EL NAVEGADOR
# chrome_options.add_argument("--headless")


def wait_for_page_load(driver, timeout=10):
    """Espera a que la página se cargue completamente"""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(2)
    except TimeoutException:
        print("Timeout esperando carga de página")


def remove_overlays(driver):
    """Elimina modales y overlays que puedan interferir"""
    scripts = [
        "document.querySelectorAll('.modal, .overlay, .modal-backdrop, [role=\"dialog\"]').forEach(e=>e.remove());",
        "document.querySelectorAll('[style*=\"z-index\"]').forEach(e=>{ if(parseInt(window.getComputedStyle(e).zIndex) > 1000) e.style.display='none'; });",
    ]
    for script in scripts:
        try:
            driver.execute_script(script)
        except:
            pass


def click_element_safely(driver, elem, description="element"):
    """Intenta varias formas de hacer click en un elemento"""
    try:
        elem.click()
        return True, "click()"
    except:
        try:
            ActionChains(driver).move_to_element(elem).click().perform()
            return True, "ActionChains"
        except:
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", elem)
                return True, "JS click"
            except Exception as e:
                return False, f"failed: {e}"


def login_with_google(driver, email=None, password=None):
    """
    Login automático con Google (basado en scrapers de IEEE y ScienceDirect)
    
    Args:
        driver: WebDriver instance
        email: Email institucional (o usa variable de entorno EMAIL)
        password: Contraseña (o usa variable de entorno PASSWORD)
    
    Returns:
        bool: True si login exitoso, False si falla
    """
    
    # Obtener credenciales
    EMAIL = email or os.getenv("EMAIL")
    PASSWORD = password or os.getenv("PASSWORD")
    
    if not EMAIL or not PASSWORD:
        print("Error: Credenciales no configuradas")
        print("Configura EMAIL y PASSWORD en .env o pásalas como parámetros")
        return False
    
    try:
        print("\nIniciando login con Google...")
        main_window = driver.current_window_handle
        
        # Buscar botón de login con Google con más selectores
        google_login_selectors = [
            (By.ID, "btn-google"),
            (By.XPATH, "//button[contains(text(), 'Google')]"),
            (By.XPATH, "//a[contains(text(), 'Google')]"),
            (By.XPATH, "//button[contains(@class, 'google')]"),
            (By.XPATH, "//a[contains(@class, 'google')]"),
            (By.CSS_SELECTOR, "button[id*='google']"),
            (By.CSS_SELECTOR, "a[id*='google']"),
            (By.CSS_SELECTOR, "button[class*='google']"),
            (By.CSS_SELECTOR, "a[class*='google']"),
        ]
        
        google_button = None
        for by, selector in google_login_selectors:
            try:
                google_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((by, selector))
                )
                print(f"Botón de Google encontrado: {selector}")
                break
            except:
                continue
        
        if not google_button:
            print("No se encontró botón de login con Google")
            return False
        
        # Click en botón de Google con múltiples estrategias
        try:
            google_button.click()
        except:
            try:
                driver.execute_script("arguments[0].click();", google_button)
            except:
                print("No se pudo hacer click en botón de Google")
                return False
        
        print("Esperando ventana de login de Google...")
        time.sleep(5)
        
        # Cambiar a ventana de Google si hay popup
        if len(driver.window_handles) > 1:
            for handle in driver.window_handles:
                if handle != main_window:
                    driver.switch_to.window(handle)
                    print("Cambiado a ventana de Google")
                    break
        
        # Captura de pantalla para debug
        print(f"URL del login: {driver.current_url}")
        driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "login_page_debug.png"))
        print("📸 Captura guardada: login_page_debug.png")
        
        # Ingresar email
        print("📧 Buscando campo 'Correo electrónico o teléfono'...")
        email_input = None
        email_selectors = [
            # Primero buscar específicamente por el texto "Correo electrónico o teléfono"
            (By.XPATH, "//input[@aria-label='Correo electrónico o teléfono']"),
            (By.XPATH, "//input[contains(@placeholder, 'Correo electrónico')]"),
            (By.XPATH, "//input[contains(@aria-label, 'Correo')]"),
            (By.XPATH, "//input[contains(@aria-label, 'correo')]"),
            # Luego los selectores estándar de Google
            (By.ID, "identifierId"),
            (By.NAME, "identifier"),
            (By.CSS_SELECTOR, "input[type='email']"),
            (By.CSS_SELECTOR, "input[type='text']"),
            (By.XPATH, "//input[@type='email']"),
            (By.XPATH, "//input[@type='text' and @name='identifier']"),
        ]
        
        for by, selector in email_selectors:
            try:
                email_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de email encontrado: {selector}")
                break
            except:
                continue
        
        # Si no se encontró con selectores específicos, buscar cualquier input visible
        if not email_input:
            print("⚠️ Intentando buscar cualquier campo de input visible...")
            try:
                all_inputs = driver.find_elements(By.TAG_NAME, "input")
                for inp in all_inputs:
                    if inp.is_displayed() and inp.is_enabled():
                        input_type = inp.get_attribute("type")
                        if input_type in ["text", "email", ""]:
                            email_input = inp
                            print(f"✅ Campo de input encontrado (type={input_type})")
                            break
            except:
                pass
        
        if not email_input:
            print("❌ No se encontró campo de email")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "login_no_email_field.png"))
            return False
        
        # Hacer scroll al campo y hacer visible
        driver.execute_script("arguments[0].scrollIntoView(true);", email_input)
        time.sleep(0.5)
        
        email_input.clear()
        email_input.send_keys(EMAIL)
        print(f"✅ Email ingresado: {EMAIL}")
        time.sleep(2)
        
        # Click en siguiente o enviar con múltiples estrategias
        print("🔍 Buscando botón 'Siguiente'...")
        next_clicked = False
        next_button_selectors = [
            (By.ID, "identifierNext"),
            (By.XPATH, "//button[contains(text(), 'Siguiente')]"),
            (By.XPATH, "//button[contains(text(), 'Next')]"),
            (By.CSS_SELECTOR, "button[type='submit']"),
        ]
        
        for by, selector in next_button_selectors:
            try:
                next_button = driver.find_element(by, selector)
                next_button.click()
                print("✅ Click en botón 'Siguiente'")
                next_clicked = True
                break
            except:
                continue
        
        if not next_clicked:
            print("⚠️ No se encontró botón, enviando con Enter")
            email_input.send_keys(Keys.RETURN)
        
        print("⏳ Esperando campo de contraseña...")
        time.sleep(5)
        
        # Ingresar contraseña
        print("🔑 Buscando campo de contraseña...")
        password_input = None
        password_selectors = [
            (By.NAME, "Passwd"),
            (By.NAME, "password"),
            (By.ID, "password"),
            (By.CSS_SELECTOR, "input[type='password']"),
        ]
        
        for by, selector in password_selectors:
            try:
                password_input = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de contraseña encontrado: {selector}")
                break
            except:
                continue
        
        # Si no se encontró, buscar cualquier input de tipo password visible
        if not password_input:
            print("⚠️ Intentando buscar cualquier campo de password visible...")
            try:
                all_inputs = driver.find_elements(By.TAG_NAME, "input")
                for inp in all_inputs:
                    if inp.is_displayed() and inp.is_enabled():
                        input_type = inp.get_attribute("type")
                        if input_type == "password":
                            password_input = inp
                            print(f"✅ Campo de password encontrado")
                            break
            except:
                pass
        
        if not password_input:
            print("❌ No se encontró campo de contraseña")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "login_no_password_field.png"))
            return False
        
        # Hacer scroll al campo
        driver.execute_script("arguments[0].scrollIntoView(true);", password_input)
        time.sleep(0.5)
        
        password_input.clear()
        password_input.send_keys(PASSWORD)
        print("✅ Contraseña ingresada")
        time.sleep(2)
        
        # Click en siguiente o enviar con múltiples estrategias
        print("🔍 Buscando botón para enviar contraseña...")
        submit_clicked = False
        submit_button_selectors = [
            (By.ID, "passwordNext"),
            (By.XPATH, "//button[contains(text(), 'Siguiente')]"),
            (By.XPATH, "//button[contains(text(), 'Next')]"),
            (By.XPATH, "//button[@type='submit']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
        ]
        
        for by, selector in submit_button_selectors:
            try:
                submit_button = driver.find_element(by, selector)
                submit_button.click()
                print("✅ Click en botón de envío")
                submit_clicked = True
                break
            except:
                continue
        
        if not submit_clicked:
            print("⚠️ No se encontró botón, enviando con Enter")
            password_input.send_keys(Keys.RETURN)
        
        print("⏳ Esperando a que complete el login...")
        time.sleep(5)
        
        # Esperar a que el login se complete - puede ser popup o misma ventana
        # Estrategia 1: Si es popup, esperar a que se cierre (max 30 segundos)
        max_wait = 30
        login_completed = False
        
        print("🔍 Esperando que el login se complete...")
        for i in range(max_wait):
            # Verificar si la ventana popup se cerró
            if len(driver.window_handles) == 1:
                print(f"✅ Ventana de login cerrada automáticamente (después de {i+1}s)")
                login_completed = True
                break
            
            # O verificar si ya no estamos en la página de login de Google
            try:
                current_url = driver.current_url.lower()
                if "accounts.google.com" not in current_url and "intelproxy" not in current_url:
                    print(f"✅ Redirigido fuera de Google login (después de {i+1}s)")
                    login_completed = True
                    break
            except:
                pass
            
            time.sleep(1)
        
        if not login_completed:
            print(f"⚠️ Timeout esperando completar login después de {max_wait}s")
        
        # Volver a ventana principal si aún hay múltiples ventanas
        if len(driver.window_handles) > 1:
            print("🔄 Cerrando ventanas adicionales y volviendo a principal...")
            windows = driver.window_handles
            for handle in windows:
                if handle != main_window:
                    try:
                        driver.switch_to.window(handle)
                        driver.close()
                    except:
                        pass
            driver.switch_to.window(main_window)
            print("✅ Vuelto a ventana principal")
        
        # Esperar que la página cargue después del login
        print("⏳ Esperando carga de página después del login...")
        time.sleep(5)
        wait_for_page_load(driver, 15)
        print(f"📍 URL actual después del login: {driver.current_url[:100]}...")
        
        print("✅ Login completado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error durante login: {e}")
        # Intentar volver a ventana principal
        try:
            driver.switch_to.window(main_window)
        except:
            pass
        return False


def scrape_acm_uniquindio(query="generative artificial intelligence", max_results=50, email=None, password=None):
    """
    Extrae artículos de ACM Digital Library vía portal Uniquindío
    """
    
    print("\n" + "="*80)
    print("🎓 SCRAPER ACM - PORTAL UNIQUINDÍO")
    print("="*80)
    print(f"🔍 Query: {query}")
    print(f"📊 Máx. resultados: {max_results}")
    print("="*80 + "\n")
    
    PORTAL_URL = "https://library.uniquindio.edu.co/databases"
    driver = webdriver.Chrome(options=chrome_options)
    all_articles = []
    
    try:
        # PASO 1: Acceder al portal
        print("📡 Paso 1: Accediendo al portal institucional...")
        driver.get(PORTAL_URL)
        wait_for_page_load(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(2)
        print("✅ Portal cargado")
        
        # PASO 2: Buscar ACM (búsqueda exhaustiva)
        print("\n📚 Paso 2: Buscando ACM Digital Library...")
        acm_found = False
        
        # Expandir todas las categorías primero
        try:
            print("   📂 Expandiendo categorías...")
            categories = driver.find_elements(By.TAG_NAME, "h2") + driver.find_elements(By.TAG_NAME, "h3")
            for cat in categories[:10]:  # Limitar a primeras 10
                try:
                    if "Ingeniería" in cat.text or "Ciencias Básicas" in cat.text or "Tecnológicas" in cat.text:
                        driver.execute_script("arguments[0].scrollIntoView(true);", cat)
                        time.sleep(0.5)
                        driver.execute_script("arguments[0].click();", cat)
                        time.sleep(1)
                        print(f"   ✅ Categoría expandida: {cat.text[:50]}")
                except:
                    pass
        except:
            pass
        
        # Buscar ACM con múltiples estrategias
        all_links = driver.find_elements(By.TAG_NAME, "a")
        print(f"   🔍 Analizando {len(all_links)} enlaces...")
        
        acm_keywords = ["acm", "computing machinery", "dl.acm"]
        
        for link in all_links:
            try:
                text = link.text.strip().lower()
                href = link.get_attribute("href") or ""
                
                if any(kw in text or kw in href.lower() for kw in acm_keywords):
                    if href and "http" in href:
                        acm_url = href
                        print(f"✅ ACM encontrada: {link.text.strip()}")
                        print(f"   URL: {acm_url[:80]}...")
                        
                        # Ir a la URL directamente en la misma ventana
                        driver.get(acm_url)
                        time.sleep(5)
                        
                        acm_found = True
                        break
            except:
                continue
        
        # Si no se encontró, intentar ir directamente a ACM
        if not acm_found:
            print("   ⚠️ No se encontró ACM en el portal, intentando acceso directo...")
            try:
                acm_url = "https://dl.acm.org"
                driver.get(acm_url)
                time.sleep(5)
                
                print(f"✅ Acceso directo a ACM: {driver.current_url[:80]}...")
                acm_found = True
            except:
                print("❌ No se pudo acceder a ACM")
                return []
        
        # PASO 2.5: Login si es necesario
        wait_for_page_load(driver, 15)
        current_url = driver.current_url.lower()
        
        if ("login" in current_url or "signin" in current_url or "auth" in current_url) and email and password:
            print("\n🔐 Paso 2.5: Autenticación detectada, iniciando sesión con Google...")
            try:
                login_with_google(driver, email, password)
                print("✅ Login completado exitosamente")
                wait_for_page_load(driver, 10)
            except Exception as e:
                print(f"⚠️ Error en login: {str(e)}")
                print("   Continuando sin autenticación...")
        
        # PASO 3: Buscar en ACM
        print(f"\n🔍 Paso 3: Buscando '{query}' en ACM...")
        
        # Buscar campo de búsqueda
        search_box = None
        search_selectors = [
            (By.NAME, "AllField"),
            (By.ID, "AllField"),
            (By.CSS_SELECTOR, "input[name='AllField']"),
            (By.CSS_SELECTOR, "input[type='search']"),
        ]
        
        for by, selector in search_selectors:
            try:
                search_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((by, selector))
                )
                break
            except:
                continue
        
        if not search_box:
            print("❌ No se encontró campo de búsqueda")
            return []
        
        # Escribir y buscar
        search_box.clear()
        search_box.send_keys(query)
        time.sleep(1)
        
        try:
            search_btn = driver.find_element(By.CSS_SELECTOR, "button.search__submit")
            driver.execute_script("arguments[0].click();", search_btn)
        except:
            search_box.send_keys(Keys.RETURN)
        
        print("✅ Búsqueda ejecutada")
        wait_for_page_load(driver, 10)
        
        # PASO 4: Extraer artículos
        print("\n📖 Paso 4: Extrayendo artículos...")
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(2)
        
        article_elements = []
        for selector in ["li.search__item", ".search-result-item", "[class*='search-item']"]:
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if article_elements:
                    print(f"✅ Encontrados {len(article_elements)} resultados")
                    break
            except:
                continue
        
        if not article_elements:
            print("⚠️ No se encontraron artículos")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "acm_no_results.png"))
            return []
        
        # Procesar artículos
        print(f"\n📚 Procesando {min(len(article_elements), max_results)} artículos...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", article)
                time.sleep(0.3)
                
                # Título
                title = ""
                for sel in ["h3", "h2", "[class*='title']"]:
                    try:
                        title = article.find_element(By.CSS_SELECTOR, sel).text.strip()
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
                    authors = [a.text.strip() for a in author_elems if a.text.strip() and len(a.text.strip()) > 2]
                except:
                    pass
                
                # DOI
                doi = ""
                try:
                    doi_link = article.find_element(By.CSS_SELECTOR, "a[href*='/doi/']")
                    doi_href = doi_link.get_attribute("href")
                    doi = doi_href.split("/doi/")[-1] if doi_href else ""
                except:
                    pass
                
                # Año
                year = ""
                try:
                    import re
                    for sel in ["[class*='date']", "[class*='year']", "time"]:
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
                
                # Abstract
                abstract = ""
                try:
                    for sel in ["[class*='abstract']", ".snippet", ".description"]:
                        try:
                            abstract = article.find_element(By.CSS_SELECTOR, sel).text.strip()
                            if abstract:
                                break
                        except:
                            continue
                except:
                    pass
                
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
        
        print(f"\n✅ Total extraído: {len(all_articles)} artículos")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return all_articles


def save_results(articles, database_name="ACM"):
    """Guarda resultados en JSON"""
    if not articles:
        print("\n⚠️ No hay artículos para guardar")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"uniquindio_{database_name.lower().replace(' ', '_')}_{timestamp}.json"
    output_path = os.path.join(DOWNLOAD_FOLDER, filename)
    
    results = {
        "database": database_name,
        "source": f"{database_name} via Uniquindio Portal",
        "query": "generative artificial intelligence",
        "extraction_date": datetime.now().isoformat(),
        "total_articles": len(articles),
        "articles": articles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados: {filename}")
    print(f"📊 Total de artículos: {len(articles)}")
    print(f"📁 Ruta: {output_path}")
    
    return output_path


def scrape_ieee_uniquindio(query="generative artificial intelligence", max_results=50, email=None, password=None):
    """
    Extrae artículos de IEEE Xplore vía portal Uniquindío
    Basado en el scraper funcional de IEEE
    """
    
    print("\n" + "="*80)
    print("🎓 SCRAPER IEEE - PORTAL UNIQUINDÍO")
    print("="*80)
    print(f"🔍 Query: {query}")
    print(f"📊 Máx. resultados: {max_results}")
    print("="*80 + "\n")
    
    PORTAL_URL = "https://library.uniquindio.edu.co/databases"
    driver = webdriver.Chrome(options=chrome_options)
    all_articles = []
    
    try:
        # PASO 1: Acceder al portal
        print("📡 Paso 1: Accediendo al portal institucional...")
        driver.get(PORTAL_URL)
        wait_for_page_load(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(2)
        print("✅ Portal cargado")
        
        # PASO 2: Buscar IEEE
        print("\n📚 Paso 2: Buscando IEEE Xplore...")
        ieee_found = False
        
        all_links = driver.find_elements(By.TAG_NAME, "a")
        ieee_keywords = ["ieee", "xplore", "ieeexplore"]
        
        for link in all_links:
            try:
                text = link.text.strip().lower()
                href = link.get_attribute("href") or ""
                
                if any(kw in text or kw in href.lower() for kw in ieee_keywords):
                    if href and "http" in href:
                        ieee_url = href
                        print(f"✅ IEEE encontrada: {link.text.strip()}")
                        print(f"   URL: {ieee_url[:80]}...")
                        
                        # Ir a la URL directamente en la misma ventana
                        driver.get(ieee_url)
                        time.sleep(5)
                        
                        ieee_found = True
                        break
            except:
                continue
        
        if not ieee_found:
            print("   ⚠️ No se encontró IEEE en el portal, intentando acceso directo...")
            ieee_url = "https://ieeexplore.ieee.org"
            driver.get(ieee_url)
            time.sleep(5)
            
            print(f"✅ Acceso directo a IEEE: {driver.current_url[:80]}...")
            ieee_found = True
        
        if not ieee_found:
            print("❌ No se pudo acceder a IEEE")
            return []
        
        # PASO 2.5: Login si es necesario
        wait_for_page_load(driver, 15)
        current_url = driver.current_url.lower()
        
        if ("login" in current_url or "signin" in current_url or "auth" in current_url or "crai" in current_url) and email and password:
            print("\n🔐 Paso 2.5: Autenticación detectada, iniciando sesión con Google...")
            try:
                login_success = login_with_google(driver, email, password)
                if login_success:
                    print("✅ Login completado, esperando redirección a IEEE...")
                    time.sleep(10)  # Esperar más tiempo después del login
                    wait_for_page_load(driver, 15)
                    
                    # Verificar si estamos en IEEE ahora
                    print(f"📍 URL actual: {driver.current_url[:100]}...")
                else:
                    print("⚠️ Login falló, continuando sin autenticación...")
            except Exception as e:
                print(f"⚠️ Error en login: {str(e)}")
                print("   Continuando sin autenticación...")
        
        # PASO 3: Buscar en IEEE
        print(f"\n🔍 Paso 3: Buscando '{query}' en IEEE...")
        print(f"📍 Verificando URL actual: {driver.current_url[:100]}...")
        
        # Esperar que la página cargue completamente
        wait_for_page_load(driver, 15)
        time.sleep(5)  # Espera adicional para asegurar que el JS termine de cargar
        
        # Hacer scroll para asegurar que el campo de búsqueda esté visible
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Buscar campo de búsqueda con selectores actualizados de IEEE 2024
        search_box = None
        search_selectors = [
            # Selector principal de IEEE Xplore (2024)
            (By.CSS_SELECTOR, "input.Typeahead-input"),
            (By.CSS_SELECTOR, "input[aria-label*='search']"),
            (By.CSS_SELECTOR, "input[placeholder*='Search']"),
            # Selectores generales de IEEE
            (By.NAME, "queryText"),
            (By.ID, "queryText"),
            (By.ID, "xplGlobalSearchInput"),
            (By.CSS_SELECTOR, "input[name='queryText']"),
            (By.CSS_SELECTOR, "input.search-input-main"),
            (By.CSS_SELECTOR, "input.searchInput"),
            # Campo en el header
            (By.XPATH, "//input[@type='text' and contains(@class, 'search')]"),
            (By.XPATH, "//input[@type='text' and @placeholder]"),
            # Campo al lado de la lupa
            (By.XPATH, "//input[following-sibling::button[contains(@class, 'search')]]"),
            (By.XPATH, "//input[following-sibling::button[contains(@class, 'fa-search')]]"),
            (By.XPATH, "//input[@type='text' and following-sibling::button]"),
            # Otros selectores comunes
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.CSS_SELECTOR, "input[aria-label*='Search']"),
            (By.XPATH, "//input[contains(@placeholder, 'Search')]"),
            # Cualquier input visible cerca de la lupa
            (By.XPATH, "//button[contains(@class, 'search')]/..//input"),
            (By.XPATH, "//form[contains(@class, 'search')]//input"),
        ]
        
        print("🔍 Buscando campo de búsqueda (al lado de la lupa)...")
        for by, selector in search_selectors:
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de búsqueda encontrado: {selector}")
                break
            except:
                continue
        
        # Si no encontró, buscar cualquier input visible de tipo texto
        if not search_box:
            print("⚠️ Buscando cualquier campo de texto visible...")
            try:
                all_inputs = driver.find_elements(By.TAG_NAME, "input")
                for inp in all_inputs:
                    if inp.is_displayed() and inp.is_enabled():
                        input_type = inp.get_attribute("type")
                        placeholder = inp.get_attribute("placeholder") or ""
                        if input_type in ["text", "search", ""] and ("search" in placeholder.lower() or not placeholder):
                            search_box = inp
                            print(f"✅ Campo encontrado por visibilidad (placeholder: {placeholder})")
                            break
            except:
                pass
        
        if not search_box:
            print("❌ No se encontró campo de búsqueda")
            print("📸 Guardando captura de pantalla...")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "ieee_no_search_box.png"))
            print(f"💡 Página actual: {driver.title}")
            return []
        
        # Hacer scroll al campo y hacerlo visible
        driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
        time.sleep(0.5)
        
        # Escribir y buscar con múltiples estrategias
        print("✏️ Ingresando query...")
        try:
            # PRIMERO: Hacer click en el campo para activarlo
            print("🖱️ Haciendo click en el campo de búsqueda...")
            try:
                search_box.click()
            except:
                # Si falla el click normal, usar ActionChains
                try:
                    ActionChains(driver).move_to_element(search_box).click().perform()
                except:
                    # Último recurso: JavaScript click
                    driver.execute_script("arguments[0].click();", search_box)
            
            time.sleep(0.5)
            print("✅ Campo activado")
            
            # Intentar limpiar el campo
            try:
                search_box.clear()
            except:
                # Si no se puede limpiar, usar JavaScript
                driver.execute_script("arguments[0].value = '';", search_box)
            
            # Intentar ingresar texto normalmente
            try:
                search_box.send_keys(query)
            except:
                # Si falla, usar JavaScript
                driver.execute_script(f"arguments[0].value = '{query}';", search_box)
            
            print(f"✅ Query ingresada: {query}")
            time.sleep(1)
            
            # Intentar enviar con Enter o buscar botón de búsqueda
            try:
                search_box.send_keys(Keys.RETURN)
            except:
                # Buscar botón de búsqueda (lupa)
                try:
                    search_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                    search_btn.click()
                except:
                    try:
                        search_btn = driver.find_element(By.XPATH, "//button[contains(@class, 'search')]")
                        search_btn.click()
                    except:
                        # Último recurso: submit del form
                        driver.execute_script("arguments[0].form.submit();", search_box)
            
        except Exception as e:
            print(f"⚠️ Error al ingresar búsqueda: {e}")
            print("📸 Guardando captura...")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "ieee_search_error.png"))
            return []
        
        print("✅ Búsqueda ejecutada")
        wait_for_page_load(driver, 10)
        
        # Aceptar cookies si aparecen
        try:
            cookie_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.osano-cm-accept-all"))
            )
            cookie_btn.click()
            time.sleep(1)
        except:
            pass
        
        # PASO 4: Extraer artículos
        print("\n📖 Paso 4: Extrayendo artículos...")
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(3)
        
        article_elements = []
        for selector in [".List-results-items", ".result-item", "xpl-results-item"]:
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if article_elements:
                    print(f"✅ Encontrados {len(article_elements)} resultados")
                    break
            except:
                continue
        
        if not article_elements:
            print("⚠️ No se encontraron artículos")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "ieee_no_results.png"))
            return []
        
        # Procesar artículos
        print(f"\n📚 Procesando {min(len(article_elements), max_results)} artículos...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", article)
                time.sleep(0.3)
                
                # Título
                title = ""
                for sel in ["h3", "h2", ".result-item-title", "[class*='title']"]:
                    try:
                        title = article.find_element(By.CSS_SELECTOR, sel).text.strip()
                        if title:
                            break
                    except:
                        continue
                
                if not title:
                    continue
                
                # Autores
                authors = []
                try:
                    author_elems = article.find_elements(By.CSS_SELECTOR, "[class*='author'], .author-name")
                    authors = [a.text.strip() for a in author_elems if a.text.strip() and len(a.text.strip()) > 2]
                except:
                    pass
                
                # DOI
                doi = ""
                try:
                    doi_link = article.find_element(By.CSS_SELECTOR, "a[href*='document']")
                    doi_href = doi_link.get_attribute("href")
                    if doi_href:
                        import re
                        doi_match = re.search(r'document/(\d+)', doi_href)
                        if doi_match:
                            doi = doi_match.group(1)
                except:
                    pass
                
                # Año
                year = ""
                try:
                    import re
                    date_elem = article.find_element(By.CSS_SELECTOR, "[class*='date'], [class*='year']")
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_elem.text)
                    if year_match:
                        year = year_match.group(0)
                except:
                    pass
                
                # Abstract
                abstract = ""
                try:
                    abstract = article.find_element(By.CSS_SELECTOR, "[class*='abstract'], .description").text.strip()
                except:
                    pass
                
                article_data = {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "year": year,
                    "source": "IEEE Xplore (Uniquindio)",
                    "abstract": abstract,
                    "keywords": [],
                    "url": f"https://ieeexplore.ieee.org/document/{doi}" if doi else "",
                    "extracted_date": datetime.now().isoformat()
                }
                
                all_articles.append(article_data)
                print(f"  {i}. {title[:70]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error en artículo {i}: {e}")
                continue
        
        print(f"\n✅ Total extraído: {len(all_articles)} artículos")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return all_articles


def scrape_sciencedirect_uniquindio(query="generative artificial intelligence", max_results=50, email=None, password=None):
    """
    Extrae artículos de ScienceDirect vía portal Uniquindío
    Basado en el scraper funcional de ScienceDirect
    """
    
    print("\n" + "="*80)
    print("🎓 SCRAPER SCIENCEDIRECT - PORTAL UNIQUINDÍO")
    print("="*80)
    print(f"🔍 Query: {query}")
    print(f"📊 Máx. resultados: {max_results}")
    print("="*80 + "\n")
    
    PORTAL_URL = "https://library.uniquindio.edu.co/databases"
    driver = webdriver.Chrome(options=chrome_options)
    all_articles = []
    
    try:
        # PASO 1: Acceder al portal
        print("📡 Paso 1: Accediendo al portal institucional...")
        driver.get(PORTAL_URL)
        wait_for_page_load(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(2)
        print("✅ Portal cargado")
        
        # PASO 2: Buscar ScienceDirect
        print("\n📚 Paso 2: Buscando ScienceDirect...")
        sd_found = False
        
        all_links = driver.find_elements(By.TAG_NAME, "a")
        sd_keywords = ["sciencedirect", "science direct", "elsevier"]
        
        for link in all_links:
            try:
                text = link.text.strip().lower()
                href = link.get_attribute("href") or ""
                
                if any(kw in text or kw in href.lower() for kw in sd_keywords):
                    if href and "http" in href:
                        sd_url = href
                        print(f"✅ ScienceDirect encontrada: {link.text.strip()}")
                        print(f"   URL: {sd_url[:80]}...")
                        
                        # Ir a la URL directamente en la misma ventana
                        driver.get(sd_url)
                        time.sleep(5)
                        
                        sd_found = True
                        break
            except:
                continue
        
        if not sd_found:
            print("   ⚠️ No se encontró ScienceDirect en el portal, intentando acceso directo...")
            sd_url = "https://www.sciencedirect.com"
            driver.get(sd_url)
            time.sleep(5)
            
            print(f"✅ Acceso directo a ScienceDirect: {driver.current_url[:80]}...")
            sd_found = True
        
        if not sd_found:
            print("❌ No se pudo acceder a ScienceDirect")
            return []
        
        # PASO 2.5: Login si es necesario
        wait_for_page_load(driver, 15)
        current_url = driver.current_url.lower()
        
        if ("login" in current_url or "signin" in current_url or "auth" in current_url or "crai" in current_url) and email and password:
            print("\n🔐 Paso 2.5: Autenticación detectada, iniciando sesión con Google...")
            try:
                login_success = login_with_google(driver, email, password)
                if login_success:
                    print("✅ Login completado, esperando redirección a ScienceDirect...")
                    time.sleep(10)  # Esperar más tiempo después del login
                    wait_for_page_load(driver, 15)
                    
                    # Verificar si estamos en ScienceDirect ahora
                    print(f"📍 URL actual: {driver.current_url[:100]}...")
                else:
                    print("⚠️ Login falló, continuando sin autenticación...")
            except Exception as e:
                print(f"⚠️ Error en login: {str(e)}")
                print("   Continuando sin autenticación...")
        
        # PASO 3: Buscar en ScienceDirect
        print(f"\n🔍 Paso 3: Buscando '{query}' en ScienceDirect...")
        print(f"📍 Verificando URL actual: {driver.current_url[:100]}...")
        
        # Esperar que la página cargue completamente
        wait_for_page_load(driver, 10)
        time.sleep(3)
        
        # Buscar campo de búsqueda con más selectores y más tiempo
        search_box = None
        search_selectors = [
            (By.NAME, "qs"),
            (By.ID, "qs"),
            (By.CSS_SELECTOR, "input[name='qs']"),
            (By.CSS_SELECTOR, "input.search-input"),
            (By.CSS_SELECTOR, "input[type='text'][placeholder*='Search']"),
            (By.CSS_SELECTOR, "input[aria-label*='search']"),
            (By.XPATH, "//input[@name='qs']"),
            (By.XPATH, "//input[contains(@class, 'search')]"),
        ]
        
        print("🔍 Buscando campo de búsqueda...")
        for by, selector in search_selectors:
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de búsqueda encontrado: {selector}")
                break
            except:
                continue
        
        if not search_box:
            print("❌ No se encontró campo de búsqueda en ScienceDirect")
            print("📸 Guardando captura de pantalla...")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "sciencedirect_no_search_box.png"))
            print(f"💡 Página actual: {driver.title}")
            return []
        
        # Escribir y buscar
        search_box.clear()
        search_box.send_keys(query)
        time.sleep(1)
        search_box.send_keys(Keys.RETURN)
        
        print("✅ Búsqueda ejecutada")
        wait_for_page_load(driver, 10)
        
        # PASO 4: Extraer artículos
        print("\n📖 Paso 4: Extrayendo artículos...")
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(3)
        
        article_elements = []
        for selector in [".result-item", ".ResultItem", "[class*='result']"]:
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if article_elements:
                    print(f"✅ Encontrados {len(article_elements)} resultados")
                    break
            except:
                continue
        
        if not article_elements:
            print("⚠️ No se encontraron artículos")
            driver.save_screenshot(os.path.join(DOWNLOAD_FOLDER, "sd_no_results.png"))
            return []
        
        # Procesar artículos
        print(f"\n📚 Procesando {min(len(article_elements), max_results)} artículos...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", article)
                time.sleep(0.3)
                
                # Título
                title = ""
                for sel in ["h3", "h2", ".result-item-title", "[class*='title']"]:
                    try:
                        title = article.find_element(By.CSS_SELECTOR, sel).text.strip()
                        if title:
                            break
                    except:
                        continue
                
                if not title:
                    continue
                
                # Autores
                authors = []
                try:
                    author_elems = article.find_elements(By.CSS_SELECTOR, "[class*='author'], .author-name")
                    authors = [a.text.strip() for a in author_elems if a.text.strip() and len(a.text.strip()) > 2]
                except:
                    pass
                
                # DOI
                doi = ""
                try:
                    doi_link = article.find_element(By.CSS_SELECTOR, "a[href*='pii']")
                    doi_href = doi_link.get_attribute("href")
                    if doi_href:
                        import re
                        doi_match = re.search(r'pii/([A-Z0-9]+)', doi_href)
                        if doi_match:
                            doi = doi_match.group(1)
                except:
                    pass
                
                # Año
                year = ""
                try:
                    import re
                    date_elem = article.find_element(By.CSS_SELECTOR, "[class*='date'], [class*='year']")
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_elem.text)
                    if year_match:
                        year = year_match.group(0)
                except:
                    pass
                
                # Abstract
                abstract = ""
                try:
                    abstract = article.find_element(By.CSS_SELECTOR, "[class*='abstract'], .description").text.strip()
                except:
                    pass
                
                article_data = {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "year": year,
                    "source": "ScienceDirect (Uniquindio)",
                    "abstract": abstract,
                    "keywords": [],
                    "url": f"https://www.sciencedirect.com/science/article/pii/{doi}" if doi else "",
                    "extracted_date": datetime.now().isoformat()
                }
                
                all_articles.append(article_data)
                print(f"  {i}. {title[:70]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error en artículo {i}: {e}")
                continue
        
        print(f"\n✅ Total extraído: {len(all_articles)} artículos")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return all_articles


def scrape_springer_uniquindio(query, max_results=50, email=None, password=None):
    """
    Extrae artículos de Springer a través del portal de Uniquindío
    """
    print("\n" + "="*80)
    print("📚 SPRINGER - EXTRACCIÓN DE ARTÍCULOS")
    print("="*80)
    
    all_articles = []
    driver = None
    
    try:
        # PASO 1: Inicializar navegador
        print("\n🌐 Paso 1: Inicializando navegador...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://library.uniquindio.edu.co/databases")
        time.sleep(2)
        print("✅ Portal cargado")
        
        # PASO 2: Buscar Springer
        print("\n📚 Paso 2: Buscando Springer...")
        springer_found = False
        
        all_links = driver.find_elements(By.TAG_NAME, "a")
        springer_keywords = ["springer"]
        
        for link in all_links:
            try:
                text = link.text.strip().lower()
                href = link.get_attribute("href") or ""
                
                if any(kw in text or kw in href.lower() for kw in springer_keywords):
                    if href and "http" in href and "springer" in href.lower():
                        springer_url = href
                        print(f"✅ Springer encontrado: {link.text.strip()}")
                        print(f"   URL: {springer_url[:80]}...")
                        
                        # Ir a la URL directamente en la misma ventana
                        driver.get(springer_url)
                        time.sleep(5)
                        
                        springer_found = True
                        break
            except:
                continue
        
        if not springer_found:
            print("   ⚠️ No se encontró Springer en el portal, intentando acceso directo...")
            springer_url = "https://link-springer-com.crai.referencistas.com/"
            driver.get(springer_url)
            time.sleep(5)
            
            print(f"✅ Acceso directo a Springer: {driver.current_url[:80]}...")
            springer_found = True
        
        if not springer_found:
            print("❌ No se pudo acceder a Springer")
            return []
        
        # PASO 2.5: Login si es necesario
        wait_for_page_load(driver, 15)
        current_url = driver.current_url.lower()
        
        if ("login" in current_url or "signin" in current_url or "auth" in current_url or "crai" in current_url) and email and password:
            print("\n🔐 Paso 2.5: Autenticación detectada, iniciando sesión con Google...")
            try:
                login_success = login_with_google(driver, email, password)
                if login_success:
                    print("✅ Login exitoso, esperando carga de Springer...")
                    time.sleep(5)
                    print(f"   URL actual: {driver.current_url[:80]}...")
                else:
                    print("⚠️ Login no completado, pero continuando con la extracción...")
            except Exception as e:
                print(f"⚠️ Error en login: {e}")
                print("   Continuando sin autenticación...")
        
        # PASO 3: Buscar en Springer (usando el mismo patrón exitoso de ScienceDirect)
        print(f"\n🔍 Paso 3: Buscando '{query}' en Springer...")
        print(f"📍 Verificando URL actual: {driver.current_url[:100]}...")
        
        # Esperar que la página cargue completamente
        wait_for_page_load(driver, 10)
        time.sleep(3)
        
        # Buscar campo de búsqueda con múltiples selectores (patrón de ScienceDirect)
        search_box = None
        search_selectors = [
            (By.NAME, "query"),
            (By.ID, "query"),
            (By.CSS_SELECTOR, "input[name='query']"),
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.CSS_SELECTOR, "input.search-input"),
            (By.CSS_SELECTOR, "input[placeholder*='Search']"),
            (By.CSS_SELECTOR, "input[aria-label*='search']"),
            (By.XPATH, "//input[@name='query']"),
            (By.XPATH, "//input[contains(@class, 'search')]"),
            (By.CSS_SELECTOR, "#search-springerlink"),
        ]
        
        print("🔍 Buscando campo de búsqueda...")
        for by, selector in search_selectors:
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de búsqueda encontrado: {selector}")
                break
            except:
                continue
        
        if not search_box:
            print("❌ No se encontró campo de búsqueda en Springer")
            print("📸 Guardando captura de pantalla...")
            driver.save_screenshot("springer_no_search_box.png")
            print(f"💡 Página actual: {driver.title}")
            return []
        
        # Escribir y buscar (método simple que funciona en ScienceDirect)
        try:
            search_box.clear()
            search_box.send_keys(query)
            time.sleep(1)
            search_box.send_keys(Keys.RETURN)
            print("✅ Búsqueda ejecutada")
        except Exception as e:
            # Si falla el método directo, intentar con JavaScript
            print(f"⚠️ Método directo falló: {e}")
            print("🔄 Intentando con JavaScript...")
            try:
                driver.execute_script(f"arguments[0].value = '{query}';", search_box)
                driver.execute_script("arguments[0].form.submit();", search_box)
                print("✅ Búsqueda ejecutada con JavaScript")
            except Exception as e2:
                print(f"❌ Error con JavaScript: {e2}")
                return []
        
        wait_for_page_load(driver, 10)
        
        # PASO 4: Extraer resultados (patrón de ScienceDirect)
        print(f"\n📊 Paso 4: Extrayendo hasta {max_results} resultados...")
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(3)
        
        # Selectores para artículos de Springer
        article_elements = []
        article_selectors = [
            "li[data-test='result-item']",
            "article.c-listing",
            ".result-item",
            "li.app-card-open",
            "ol.c-list-group li",
            "[class*='result']"
        ]
        
        for selector in article_selectors:
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if article_elements:
                    print(f"✅ Encontrados {len(article_elements)} resultados")
                    break
            except:
                continue
        
        if not article_elements:
            print("⚠️ No se encontraron artículos")
            driver.save_screenshot("springer_no_results.png")
            return []
        
        print(f"   Extrayendo los primeros {min(len(article_elements), max_results)}...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                # Título
                title = ""
                try:
                    title_elem = article.find_element(By.CSS_SELECTOR, "h3 a, .title a, a.title")
                    title = title_elem.text.strip()
                except:
                    pass
                
                if not title:
                    continue
                
                # Autores
                authors = []
                try:
                    author_elems = article.find_elements(By.CSS_SELECTOR, ".c-author-list a, .authors a")
                    authors = [a.text.strip() for a in author_elems if a.text.strip()]
                except:
                    pass
                
                # DOI/URL
                url = ""
                doi = ""
                try:
                    link = article.find_element(By.CSS_SELECTOR, "h3 a, .title a")
                    url = link.get_attribute("href")
                    if url and "doi.org" in url:
                        import re
                        doi_match = re.search(r'10\.\d+/[^\s]+', url)
                        if doi_match:
                            doi = doi_match.group(0)
                except:
                    pass
                
                # Año
                year = ""
                try:
                    import re
                    year_elem = article.find_element(By.CSS_SELECTOR, ".c-meta__item, .publication-date")
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_elem.text)
                    if year_match:
                        year = year_match.group(0)
                except:
                    pass
                
                # Abstract
                abstract = ""
                try:
                    abstract = article.find_element(By.CSS_SELECTOR, ".c-card__summary, .snippet").text.strip()
                except:
                    pass
                
                article_data = {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "year": year,
                    "source": "Springer (Uniquindio)",
                    "abstract": abstract,
                    "keywords": [],
                    "url": url,
                    "extracted_date": datetime.now().isoformat()
                }
                
                all_articles.append(article_data)
                print(f"  {i}. {title[:70]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error en artículo {i}: {e}")
                continue
        
        print(f"\n✅ Total extraído: {len(all_articles)} artículos")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return all_articles


def main():
    """Función principal - Extrae de las 3 bases de datos principales"""
    print("\n" + "="*80)
    print("🎓 SCRAPER AUTOMÁTICO - PORTAL UNIQUINDÍO")
    print("   Universidad del Quindío - Biblioteca")
    print("="*80)
    print("\n💡 Este scraper extrae artículos automáticamente de:")
    print("   • IEEE Xplore")
    print("   • ScienceDirect")
    print("   • Springer")
    print("\n🔧 Configuración:")
    print(f"   • Query: 'generative artificial intelligence'")
    print(f"   • Resultados por base: 50")
    print(f"   • Carpeta de salida: {DOWNLOAD_FOLDER}")
    
    # Obtener credenciales de variables de entorno
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    
    if email and password:
        print(f"   • Autenticación: Habilitada ({email})")
    else:
        print(f"   • Autenticación: Deshabilitada (crea .env con EMAIL y PASSWORD)")
    
    print("\n" + "="*80 + "\n")
    
    time.sleep(2)
    
    query = "generative artificial intelligence"
    max_results = 50
    
    all_results = {}
    
    # 1. Extraer de IEEE Xplore
    print("🚀 1/3 Extrayendo de IEEE Xplore...\n")
    ieee_articles = scrape_ieee_uniquindio(query, max_results, email, password)
    if ieee_articles:
        save_results(ieee_articles, "IEEE Xplore")
        all_results["IEEE"] = len(ieee_articles)
        print(f"✅ IEEE completado: {len(ieee_articles)} artículos extraídos\n")
    else:
        print("⚠️  IEEE: No se encontraron artículos\n")
        all_results["IEEE"] = 0
    
    time.sleep(3)
    
    # 2. Extraer de ScienceDirect
    print("\n🚀 2/3 Extrayendo de ScienceDirect...\n")
    sd_articles = scrape_sciencedirect_uniquindio(query, max_results, email, password)
    if sd_articles:
        save_results(sd_articles, "ScienceDirect")
        all_results["ScienceDirect"] = len(sd_articles)
        print(f"✅ ScienceDirect completado: {len(sd_articles)} artículos extraídos\n")
    else:
        print("⚠️  ScienceDirect: No se encontraron artículos\n")
        all_results["ScienceDirect"] = 0
    
    time.sleep(3)
    
    # 3. Extraer de Springer
    print("\n🚀 3/3 Extrayendo de Springer...\n")
    springer_articles = scrape_springer_uniquindio(query, max_results, email, password)
    if springer_articles:
        save_results(springer_articles, "Springer")
        all_results["Springer"] = len(springer_articles)
        print(f"✅ Springer completado: {len(springer_articles)} artículos extraídos\n")
    else:
        print("⚠️  Springer: No se encontraron artículos\n")
        all_results["Springer"] = 0
    
    # Resumen final
    total_articles = sum(all_results.values())
    
    print("\n" + "="*80)
    print("✅ EXTRACCIÓN COMPLETADA")
    print("="*80)
    print(f"\n📊 Resumen:")
    for db, count in all_results.items():
        print(f"   • {db}: {count} artículos")
    print(f"   • Total: {total_articles} artículos")
    print(f"\n📁 Archivos guardados en: {DOWNLOAD_FOLDER}")
    print("\n💡 Próximos pasos:")
    print("   1. Revisa los archivos JSON generados")
    print("   2. Ejecuta el pipeline de unificación:")
    print("      cd bibliometric-analysis")
    print("      python automation_pipeline.py")
    print("   3. Realiza análisis de similitud y clustering")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
