"""
ACM Scraper con pausa manual para CAPTCHA
El navegador se mantiene abierto para que resuelvas el CAPTCHA manualmente
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from datetime import datetime
from pathlib import Path

def scrape_acm_with_manual_captcha(query="generative artificial intelligence", max_results=50):
    """
    Scraper de ACM que permite resolver CAPTCHA manualmente
    """
    
    print("="*70)
    print("🎓 ACM DIGITAL LIBRARY SCRAPER (Con resolución manual de CAPTCHA)")
    print("="*70)
    print(f"🔍 Query: {query}")
    print(f"📊 Máx. resultados: {max_results}")
    print("="*70)
    
    # Configurar Chrome (SIN headless para ver el navegador)
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Agregar user agent real
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    
    try:
        # Paso 1: Ir a ACM
        print("\n📡 Paso 1: Accediendo a ACM Digital Library...")
        driver.get("https://dl.acm.org/")
        time.sleep(3)
        
        print("✅ Página cargada")
        print(f"📍 URL: {driver.current_url}")
        
        # Verificar si hay CAPTCHA
        if "captcha" in driver.page_source.lower() or "recaptcha" in driver.page_source.lower():
            print("\n" + "="*70)
            print("⚠️  CAPTCHA DETECTADO")
            print("="*70)
            print("\n🤖 Por favor, resuelve el CAPTCHA manualmente en el navegador.")
            print("👉 Una vez resuelto, presiona ENTER en esta terminal para continuar...")
            print("\n⏳ Esperando...")
            input()
            print("✅ Continuando...")
        
        # Paso 2: Buscar
        print("\n🔍 Paso 2: Buscando campo de búsqueda...")
        
        search_input = None
        search_selectors = [
            (By.ID, "search"),
            (By.NAME, "AllField"),
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.CSS_SELECTOR, "input[placeholder*='Search']"),
            (By.XPATH, "//input[@type='search' or @name='AllField']")
        ]
        
        for by, selector in search_selectors:
            try:
                search_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((by, selector))
                )
                print(f"✅ Campo de búsqueda encontrado: {selector}")
                break
            except:
                continue
        
        if not search_input:
            print("❌ No se encontró campo de búsqueda")
            print("\n📸 Tomando captura...")
            driver.save_screenshot("acm_no_search.png")
            return []
        
        # Realizar búsqueda
        print(f"\n🔍 Paso 3: Buscando '{query}'...")
        search_input.clear()
        search_input.send_keys(query)
        search_input.send_keys(Keys.RETURN)
        
        print("⏳ Esperando resultados...")
        time.sleep(5)
        
        # Verificar CAPTCHA nuevamente después de búsqueda
        if "captcha" in driver.page_source.lower() or "recaptcha" in driver.page_source.lower():
            print("\n" + "="*70)
            print("⚠️  CAPTCHA DETECTADO DESPUÉS DE BÚSQUEDA")
            print("="*70)
            print("\n🤖 Por favor, resuelve el CAPTCHA manualmente.")
            print("👉 Presiona ENTER cuando esté resuelto...")
            input()
            time.sleep(3)
        
        print(f"✅ Búsqueda ejecutada")
        print(f"📍 URL actual: {driver.current_url}")
        
        # Paso 3: Extraer artículos
        print("\n📖 Paso 4: Extrayendo artículos...")
        driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(2)
        
        # Probar múltiples selectores
        article_elements = []
        result_selectors = [
            "li.issue-item",
            "li.search__item",
            ".search-result-item",
            "[class*='issue-item']",
            "div[class*='search-result']",
            "article"
        ]
        
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
            print("\n📸 Tomando captura...")
            driver.save_screenshot("acm_no_results.png")
            
            print("\n🔍 ¿Quieres intentar extraer manualmente?")
            print("👉 Navega a los resultados en el navegador y presiona ENTER...")
            input()
            
            # Reintentar
            for selector in result_selectors:
                try:
                    article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if article_elements:
                        print(f"✅ Ahora encontrados {len(article_elements)} resultados")
                        break
                except:
                    continue
        
        if not article_elements:
            print("❌ No se pudieron extraer artículos")
            return []
        
        # Procesar artículos
        articles = []
        print(f"\n📚 Procesando {min(len(article_elements), max_results)} artículos...\n")
        
        for i, article in enumerate(article_elements[:max_results], 1):
            try:
                # Scroll al elemento
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", article)
                time.sleep(0.3)
                
                # Extraer título
                title = ""
                title_selectors = ["h3.issue-item__title", "h3", "h2", "[class*='title'] a", "a.issue-item__title"]
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
                    author_list = article.find_element(By.CSS_SELECTOR, "ul.loa, ul.rlist--inline")
                    author_elems = author_list.find_elements(By.TAG_NAME, "li")
                    authors = [a.text.strip() for a in author_elems if a.text.strip()]
                except:
                    pass
                
                # Extraer DOI/URL
                doi = ""
                url = ""
                try:
                    doi_link = article.find_element(By.CSS_SELECTOR, "a[href*='/doi/']")
                    href = doi_link.get_attribute("href")
                    url = href if href.startswith("http") else f"https://dl.acm.org{href}"
                    doi = href.split("/doi/")[-1] if "/doi/" in href else ""
                except:
                    pass
                
                # Extraer año
                year = ""
                try:
                    import re
                    date_elems = article.find_elements(By.CSS_SELECTOR, "[class*='date'], [class*='year'], time")
                    for elem in date_elems:
                        year_match = re.search(r'\b(19|20)\d{2}\b', elem.text)
                        if year_match:
                            year = year_match.group(0)
                            break
                except:
                    pass
                
                # Extraer abstract
                abstract = ""
                try:
                    abstract_elem = article.find_element(By.CSS_SELECTOR, "[class*='abstract'], .issue-item__abstract")
                    abstract = abstract_elem.text.strip()
                except:
                    pass
                
                article_data = {
                    "id": f"acm_{i}",
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "url": url,
                    "abstract": abstract,
                    "source": "acm",
                    "keywords": [],
                    "extracted_date": datetime.now().isoformat()
                }
                
                articles.append(article_data)
                print(f"  {i}. {title[:65]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error en artículo {i}: {str(e)}")
                continue
        
        print(f"\n✅ Total extraído: {len(articles)} artículos")
        
        # Guardar resultados
        if articles:
            output_dir = Path("scrapers/data/raw/acm")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"acm_manual_{timestamp}.json"
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Resultados guardados: {output_file}")
            print(f"📊 Total de artículos: {len(articles)}")
        
        print("\n⏳ El navegador permanecerá abierto 10 segundos...")
        time.sleep(10)
        
        return articles
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        driver.save_screenshot("acm_error.png")
        return []
        
    finally:
        driver.quit()
        print("\n👋 Navegador cerrado")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 INICIANDO SCRAPER DE ACM CON RESOLUCIÓN MANUAL DE CAPTCHA")
    print("="*70)
    print("\n💡 INSTRUCCIONES:")
    print("   1. El navegador se abrirá automáticamente")
    print("   2. Si aparece un CAPTCHA, resuélvelo manualmente")
    print("   3. Presiona ENTER en esta terminal para continuar")
    print("   4. El scraper extraerá los artículos automáticamente")
    print("\n" + "="*70)
    input("\n👉 Presiona ENTER para comenzar...")
    
    articles = scrape_acm_with_manual_captcha()
    
    if articles:
        print(f"\n✅ ¡Scraping completado exitosamente!")
        print(f"📊 Total de artículos extraídos: {len(articles)}")
    else:
        print(f"\n⚠️ No se pudieron extraer artículos de ACM")
        print("\n💡 ALTERNATIVA: Usa los datos de Semantic Scholar, ScienceDirect y Springer")
        print("   Ya tienes 50 + 25 + 20 = 95 artículos descargados!")
