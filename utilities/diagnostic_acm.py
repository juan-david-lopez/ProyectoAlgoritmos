"""
Script de diagnóstico para ACM Digital Library
Muestra qué elementos encuentra en la página para ajustar los selectores
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configurar Chrome
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
# NO headless para ver el navegador
# chrome_options.add_argument("--headless")

print("="*70)
print("🔍 DIAGNÓSTICO ACM DIGITAL LIBRARY")
print("="*70)

driver = webdriver.Chrome(options=chrome_options)

try:
    # Ir directamente a ACM
    print("\n📡 Paso 1: Accediendo a ACM Digital Library...")
    driver.get("https://dl.acm.org/")
    time.sleep(5)
    
    print("✅ Página cargada")
    print(f"📍 URL: {driver.current_url}")
    
    # Buscar campo de búsqueda
    print("\n🔍 Paso 2: Buscando campo de búsqueda...")
    
    search_selectors = [
        (By.ID, "search"),
        (By.NAME, "AllField"),
        (By.CSS_SELECTOR, "input[type='search']"),
        (By.CSS_SELECTOR, "input[placeholder*='Search']"),
        (By.CSS_SELECTOR, "input[class*='search']"),
        (By.XPATH, "//input[@type='search' or @type='text']")
    ]
    
    search_input = None
    for by, selector in search_selectors:
        try:
            search_input = driver.find_element(by, selector)
            print(f"✅ Campo de búsqueda encontrado: {selector}")
            print(f"   Atributos: id='{search_input.get_attribute('id')}', name='{search_input.get_attribute('name')}', class='{search_input.get_attribute('class')}'")
            break
        except:
            continue
    
    if not search_input:
        print("❌ No se encontró campo de búsqueda")
        print("\n📋 Todos los inputs en la página:")
        all_inputs = driver.find_elements(By.TAG_NAME, "input")
        for inp in all_inputs[:10]:  # Mostrar primeros 10
            print(f"   - type='{inp.get_attribute('type')}', id='{inp.get_attribute('id')}', name='{inp.get_attribute('name')}', placeholder='{inp.get_attribute('placeholder')}'")
        driver.quit()
        exit(1)
    
    # Realizar búsqueda
    print("\n🔍 Paso 3: Realizando búsqueda...")
    query = "generative artificial intelligence"
    search_input.clear()
    search_input.send_keys(query)
    search_input.send_keys(Keys.RETURN)
    
    print("⏳ Esperando resultados...")
    time.sleep(5)
    
    print(f"✅ Búsqueda ejecutada")
    print(f"📍 URL actual: {driver.current_url}")
    
    # Buscar resultados
    print("\n📊 Paso 4: Analizando resultados...")
    
    # Intentar diferentes selectores
    result_selectors = [
        "li.search__item",
        ".search-result-item",
        "[class*='search-item']",
        "[class*='result']",
        "li[class*='item']",
        "article",
        ".issue-item",
        "[data-title]"
    ]
    
    found_any = False
    for selector in result_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                print(f"✅ Selector '{selector}' encontró {len(elements)} elementos")
                
                # Analizar el primer resultado
                if len(elements) > 0:
                    elem = elements[0]
                    print(f"\n   📝 Análisis del primer elemento:")
                    print(f"   - Tag: {elem.tag_name}")
                    print(f"   - Classes: {elem.get_attribute('class')}")
                    
                    # Buscar título
                    for title_sel in ["h3", "h2", "h4", "[class*='title']", "a.issue-item__title"]:
                        try:
                            title_elem = elem.find_element(By.CSS_SELECTOR, title_sel)
                            print(f"   - Título (selector {title_sel}): {title_elem.text[:60]}...")
                            break
                        except:
                            pass
                    
                    # Buscar autores
                    for auth_sel in ["[class*='author']", ".contrib-author", ".loa__item"]:
                        try:
                            author_elems = elem.find_elements(By.CSS_SELECTOR, auth_sel)
                            if author_elems:
                                print(f"   - Autores (selector {auth_sel}): {len(author_elems)} encontrados")
                                break
                        except:
                            pass
                    
                    # Buscar DOI
                    try:
                        doi_link = elem.find_element(By.CSS_SELECTOR, "a[href*='/doi/']")
                        print(f"   - DOI encontrado: {doi_link.get_attribute('href')}")
                    except:
                        print(f"   - DOI: No encontrado")
                    
                    found_any = True
                    print()
        except Exception as e:
            pass
    
    if not found_any:
        print("❌ No se encontraron resultados con ningún selector")
        print("\n📸 Guardando captura de pantalla...")
        driver.save_screenshot("acm_diagnostic.png")
        print("✅ Captura guardada como 'acm_diagnostic.png'")
        
        print("\n📋 HTML de la página (primeros 2000 caracteres):")
        print(driver.page_source[:2000])
    
    print("\n✅ Diagnóstico completado")
    print("⏳ El navegador permanecerá abierto por 30 segundos para inspección manual...")
    time.sleep(30)
    
finally:
    driver.quit()
    print("\n👋 Navegador cerrado")
