"""
ACM Digital Library Scraper usando requests (más confiable que Selenium)
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

def scrape_acm_api(query="generative artificial intelligence", max_results=50):
    """
    Scraper de ACM usando búsqueda directa (sin API key requerida)
    """
    
    print("="*70)
    print("🎓 ACM DIGITAL LIBRARY SCRAPER (Método directo)")
    print("="*70)
    print(f"🔍 Query: {query}")
    print(f"📊 Máx. resultados: {max_results}")
    print("="*70)
    
    # URL de búsqueda de ACM
    base_url = "https://dl.acm.org/action/doSearch"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    # Parámetros de búsqueda
    params = {
        "AllField": query,
        "pageSize": min(max_results, 50),  # ACM max 50 por página
        "startPage": 0
    }
    
    articles = []
    
    try:
        print(f"\n📡 Realizando búsqueda...")
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ Respuesta recibida (status {response.status_code})")
            
            # ACM devuelve HTML, necesitamos parsearlo
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscar resultados
            # ACM usa diferentes estructuras, probar varias
            result_selectors = [
                {"class": "issue-item"},
                {"class": "search__item"},
                {"class": "search-result"},
            ]
            
            results = []
            for selector in result_selectors:
                results = soup.find_all("li", selector)
                if results:
                    print(f"✅ Encontrados {len(results)} resultados con selector {selector}")
                    break
            
            if not results:
                # Intentar con divs
                results = soup.find_all("div", class_=lambda x: x and "search" in x.lower())
                if results:
                    print(f"✅ Encontrados {len(results)} resultados (divs)")
            
            if not results:
                print("⚠️ No se encontraron resultados en el HTML")
                # Guardar HTML para análisis
                with open("acm_response.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("💾 HTML guardado en 'acm_response.html' para análisis")
                return []
            
            print(f"\n📚 Procesando {len(results)} artículos...\n")
            
            for i, result in enumerate(results[:max_results], 1):
                try:
                    # Extraer título
                    title_elem = result.find(["h3", "h2", "h4"], class_=lambda x: x and "title" in x.lower())
                    if not title_elem:
                        title_elem = result.find("a", class_="issue-item__title")
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extraer autores
                    authors = []
                    author_section = result.find("ul", class_="loa")
                    if author_section:
                        author_elems = author_section.find_all("li")
                        authors = [a.get_text(strip=True) for a in author_elems]
                    
                    # Extraer DOI/URL
                    doi = ""
                    url = ""
                    doi_link = result.find("a", href=lambda x: x and "/doi/" in x)
                    if doi_link:
                        url = "https://dl.acm.org" + doi_link['href']
                        doi = doi_link['href'].split("/doi/")[-1] if "/doi/" in doi_link['href'] else ""
                    
                    # Extraer año
                    year = ""
                    import re
                    date_elem = result.find(class_=lambda x: x and ("date" in x.lower() or "year" in x.lower()))
                    if date_elem:
                        year_match = re.search(r'\b(19|20)\d{2}\b', date_elem.get_text())
                        if year_match:
                            year = year_match.group(0)
                    
                    # Extraer abstract/snippet
                    abstract = ""
                    abstract_elem = result.find(class_=lambda x: x and ("abstract" in x.lower() or "snippet" in x.lower()))
                    if abstract_elem:
                        abstract = abstract_elem.get_text(strip=True)
                    
                    article = {
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
                    
                    articles.append(article)
                    print(f"  {i}. {title[:60]}...")
                    
                except Exception as e:
                    print(f"  ⚠️ Error procesando artículo {i}: {str(e)}")
                    continue
            
            print(f"\n✅ Total extraído: {len(articles)} artículos")
            
        else:
            print(f"❌ Error HTTP: {response.status_code}")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    
    # Guardar resultados
    if articles:
        output_dir = Path("scrapers/data/raw/acm")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"acm_articles_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados: {output_file}")
        print(f"📊 Total de artículos: {len(articles)}")
    
    return articles


if __name__ == "__main__":
    # Instalar beautifulsoup4 si no está instalado
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("⚠️ Instalando beautifulsoup4...")
        import subprocess
        subprocess.check_call(["pip", "install", "beautifulsoup4"])
        from bs4 import BeautifulSoup
    
    articles = scrape_acm_api()
    
    if articles:
        print(f"\n✅ Scraping completado exitosamente")
    else:
        print(f"\n⚠️ No se pudieron extraer artículos")
        print("\n💡 ALTERNATIVAS:")
        print("   1. Usar Semantic Scholar (ya funcionando)")
        print("   2. Descargar manualmente de ACM y importar BibTeX/RIS")
        print("   3. Registrarse para API key de ACM")
