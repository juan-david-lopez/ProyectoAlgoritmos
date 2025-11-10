"""
Universidad del Quindío - Portal Institucional Scraper
Acceso a bases de datos académicas a través del portal institucional
https://library.uniquindio.edu.co/databases

Este scraper:
1. Se conecta al portal de la biblioteca
2. Accede a las bases de datos disponibles (ACM, IEEE, Scopus, etc.)
3. Extrae artículos usando las credenciales institucionales
4. Unifica los resultados en formato estándar
"""

import os
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime


class UniquindioPortalScraper:

    """
    Scraper para acceder a bases de datos a través del portal institucional
    de la Universidad del Quindío
    
    Portal: https://library.uniquindio.edu.co/databases
    Acceso: Red institucional o VPN
    """

    PORTAL_URL = "https://library.uniquindio.edu.co/databases"
    
    # Categorías relevantes para ingeniería/tecnología
    RELEVANT_CATEGORIES = [
        "Fac. Ingeniería",
        "Fac. Ciencias Básicas y Tecnológicas"
    ]
    
    # Bases de datos objetivo
    TARGET_DATABASES = [
        "ACM Digital Library",
        "IEEE Xplore",
        "Scopus",
        "Web of Science",
        "ScienceDirect"
    ]

    def __init__(self, config=None, headless: bool = True):
        """
        Initialize Uniquindio portal scraper

        Args:
            config: Configuration object (opcional)
            headless: Run browser in headless mode
        """
        self.config = config
        self.headless = headless
        self.driver = None
        self.available_databases = []
        
        # Configurar carpeta de descargas
        self.download_folder = Path("data/raw")
        self.download_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info("Uniquindio Portal Scraper initialized")
    
    def _setup_driver(self):
        """Configura el driver de Chrome con opciones optimizadas"""
        chrome_options = Options()
        
        # Configurar descargas
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_folder.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        
        # Opciones adicionales
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.maximize_window()
        logger.info("Chrome driver configurado exitosamente")
    
    def close(self):
        """Cierra el navegador"""
        if self.driver:
            self.driver.quit()
            logger.info("Navegador cerrado")

    def discover_databases(self) -> List[Dict[str, str]]:
        """
        Descubre las bases de datos disponibles en el portal de forma automática
        
        Returns:
            Lista de bases de datos con nombre y URL de acceso
        """
        logger.info("🔍 Explorando bases de datos disponibles en el portal...")
        
        try:
            self.driver.get(self.PORTAL_URL)
            time.sleep(5)
            
            # Scroll para cargar contenido dinámico
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(2)
            
            databases = []
            
            # Estrategia 1: Expandir todas las categorías automáticamente
            logger.info("📂 Expandiendo todas las categorías...")
            try:
                # Buscar y hacer clic en "Expandir todo" si existe
                expand_buttons = self.driver.find_elements(By.XPATH, 
                    "//button[contains(text(), 'Expandir') or contains(@class, 'expand')]"
                )
                for btn in expand_buttons:
                    try:
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.5)
                    except:
                        pass
            except:
                pass
            
            # Estrategia 2: Buscar todas las categorías y expandirlas
            for category in self.RELEVANT_CATEGORIES:
                try:
                    # Buscar encabezados de categoría (h2, h3, h4)
                    selectors = [
                        f"//h2[contains(text(), '{category}')]",
                        f"//h3[contains(text(), '{category}')]",
                        f"//h4[contains(text(), '{category}')]",
                        f"//*[contains(@class, 'heading') and contains(text(), '{category}')]"
                    ]
                    
                    category_element = None
                    for selector in selectors:
                        elements = self.driver.find_elements(By.XPATH, selector)
                        if elements:
                            category_element = elements[0]
                            break
                    
                    if category_element:
                        logger.info(f"✅ Categoría encontrada: {category}")
                        
                        # Scroll hasta el elemento
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", category_element)
                        time.sleep(1)
                        
                        # Intentar expandir si está colapsado
                        try:
                            self.driver.execute_script("arguments[0].click();", category_element)
                            time.sleep(1)
                        except:
                            pass
                        
                        # Buscar bases de datos en el contenedor padre
                        parent = category_element.find_element(By.XPATH, "./ancestor::div[contains(@class, 'section') or contains(@class, 'category')][1]")
                        
                        # Buscar todos los enlaces dentro de esta sección
                        db_links = parent.find_elements(By.TAG_NAME, "a")
                        
                        for link in db_links:
                            try:
                                db_name = link.text.strip()
                                db_url = link.get_attribute("href")
                                
                                # Filtrar enlaces válidos
                                if (db_name and db_url and 
                                    len(db_name) > 3 and 
                                    'http' in db_url and
                                    db_url not in [d["url"] for d in databases]):
                                    
                                    databases.append({
                                        "name": db_name,
                                        "url": db_url,
                                        "category": category
                                    })
                                    logger.info(f"  📚 {db_name}")
                            except:
                                continue
                
                except Exception as e:
                    logger.debug(f"Error explorando categoría {category}: {e}")
            
            # Estrategia 3: Búsqueda directa de bases de datos conocidas
            logger.info("🔍 Buscando bases de datos específicas...")
            for target_db in self.TARGET_DATABASES:
                try:
                    # Múltiples selectores para encontrar la base de datos
                    selectors = [
                        f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_db.lower()}')]",
                        f"//a[contains(@title, '{target_db}')]",
                        f"//a[contains(@aria-label, '{target_db}')]",
                        f"//*[contains(text(), '{target_db}')]/ancestor::a[1]"
                    ]
                    
                    for selector in selectors:
                        db_elements = self.driver.find_elements(By.XPATH, selector)
                        
                        for element in db_elements:
                            try:
                                db_url = element.get_attribute("href")
                                if db_url and 'http' in db_url and db_url not in [d["url"] for d in databases]:
                                    databases.append({
                                        "name": target_db,
                                        "url": db_url,
                                        "category": "Búsqueda directa"
                                    })
                                    logger.info(f"✅ {target_db} encontrada: {db_url[:60]}...")
                                    break
                            except:
                                continue
                        
                        if any(d["name"] == target_db for d in databases):
                            break
                
                except Exception as e:
                    logger.debug(f"Base de datos {target_db} no encontrada: {e}")
            
            # Estrategia 4: Búsqueda en todo el contenido de la página
            if not databases:
                logger.warning("⚠️ No se encontraron bases de datos con métodos anteriores")
                logger.info("🔍 Buscando en todo el contenido de la página...")
                
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                keywords = ["acm", "ieee", "scopus", "science", "web of science", "elsevier", 
                           "springer", "wiley", "jstor", "ebsco"]
                
                for link in all_links:
                    try:
                        text = link.text.strip().lower()
                        href = link.get_attribute("href")
                        
                        if href and any(kw in text or kw in href.lower() for kw in keywords):
                            if href not in [d["url"] for d in databases]:
                                databases.append({
                                    "name": link.text.strip() or "Base de datos",
                                    "url": href,
                                    "category": "Detección automática"
                                })
                                logger.info(f"  📚 {link.text.strip()}")
                    except:
                        continue
            
            self.available_databases = databases
            logger.info(f"✅ Total de bases de datos encontradas: {len(databases)}")
            
            return databases
        
        except Exception as e:
            logger.error(f"❌ Error descubriendo bases de datos: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def access_database(self, database_name: str) -> bool:
        """
        Accede a una base de datos específica a través del portal de forma automática
        
        Args:
            database_name: Nombre de la base de datos
            
        Returns:
            True si el acceso fue exitoso
        """
        logger.info(f"🔐 Intentando acceder a: {database_name}")
        
        # Buscar la base de datos en la lista
        db_info = next(
            (db for db in self.available_databases if database_name.lower() in db["name"].lower()),
            None
        )
        
        if not db_info:
            logger.error(f"❌ Base de datos '{database_name}' no encontrada en el portal")
            return False
        
        try:
            logger.info(f"🌐 Navegando a: {db_info['url'][:80]}...")
            
            # Abrir en una nueva pestaña para mantener la sesión del portal
            original_window = self.driver.current_window_handle
            self.driver.execute_script(f"window.open('{db_info['url']}', '_blank');")
            
            # Cambiar a la nueva pestaña
            time.sleep(2)
            windows = self.driver.window_handles
            self.driver.switch_to.window(windows[-1])
            
            # Esperar a que la página cargue
            time.sleep(5)
            
            # Verificar URL actual
            current_url = self.driver.current_url
            logger.info(f"📍 URL actual: {current_url[:80]}...")
            
            # Detectar y manejar autenticación automáticamente
            if "login" in current_url.lower() or "auth" in current_url.lower() or "signin" in current_url.lower():
                logger.warning("⚠️ Página de autenticación detectada")
                
                # Intentar encontrar enlaces de acceso institucional
                institutional_links = [
                    "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'institution')]",
                    "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'shibboleth')]",
                    "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'institutional')]",
                    "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'institution')]"
                ]
                
                for selector in institutional_links:
                    try:
                        elements = self.driver.find_elements(By.XPATH, selector)
                        if elements:
                            logger.info("✅ Encontrado enlace de acceso institucional")
                            self.driver.execute_script("arguments[0].click();", elements[0])
                            time.sleep(3)
                            break
                    except:
                        continue
                
                # Verificar si seguimos en página de login
                current_url = self.driver.current_url
                if "login" in current_url.lower():
                    logger.warning("⚠️ Aún en página de login - puede requerir credenciales")
                    logger.info("💡 Continuando de todas formas...")
            
            # Verificar si llegamos a la base de datos
            success_indicators = [
                database_name.lower().replace(" ", ""),
                "search", "explore", "browse", "database"
            ]
            
            page_content = self.driver.page_source.lower()
            if any(indicator in current_url.lower() or indicator in page_content for indicator in success_indicators):
                logger.info(f"✅ Acceso exitoso a {database_name}")
                return True
            else:
                logger.warning(f"⚠️ Acceso incierto a {database_name}, continuando...")
                return True  # Intentar de todas formas
        
        except Exception as e:
            logger.error(f"❌ Error accediendo a {database_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def search_in_database(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda en la base de datos actual
        
        Args:
            query: Término de búsqueda
            max_results: Número máximo de resultados
            
        Returns:
            Lista de artículos encontrados
        """
        logger.info(f"🔍 Buscando: '{query}'")
        
        try:
            # Detectar tipo de base de datos actual
            current_url = self.driver.current_url.lower()
            
            if "acm.org" in current_url:
                return self._search_acm(query, max_results)
            elif "ieeexplore.ieee.org" in current_url:
                return self._search_ieee(query, max_results)
            elif "sciencedirect.com" in current_url:
                return self._search_sciencedirect(query, max_results)
            elif "scopus.com" in current_url:
                return self._search_scopus(query, max_results)
            else:
                logger.warning(f"⚠️ Base de datos no reconocida: {current_url}")
                return self._search_generic(query, max_results)
        
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}")
            return []

    def _search_acm(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Búsqueda específica en ACM Digital Library con automatización completa"""
        logger.info("📚 Ejecutando búsqueda automática en ACM Digital Library")
        
        try:
            # Esperar a que la página cargue completamente
            time.sleep(3)
            
            # Buscar el campo de búsqueda con múltiples selectores
            search_box = None
            search_selectors = [
                (By.NAME, "AllField"),
                (By.ID, "AllField"),
                (By.CSS_SELECTOR, "input[name='AllField']"),
                (By.CSS_SELECTOR, "input[type='search']"),
                (By.CSS_SELECTOR, ".search-input"),
                (By.XPATH, "//input[contains(@placeholder, 'Search')]")
            ]
            
            logger.info("🔍 Buscando campo de búsqueda...")
            for by, selector in search_selectors:
                try:
                    search_box = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    logger.info(f"✅ Campo de búsqueda encontrado: {selector}")
                    break
                except:
                    continue
            
            if not search_box:
                logger.error("❌ No se pudo encontrar el campo de búsqueda")
                return []
            
            # Limpiar y escribir query
            search_box.clear()
            time.sleep(0.5)
            search_box.send_keys(query)
            logger.info(f"✅ Query ingresada: '{query}'")
            time.sleep(1)
            
            # Buscar y hacer clic en el botón de búsqueda
            search_button = None
            button_selectors = [
                (By.CSS_SELECTOR, "button.search__submit"),
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.XPATH, "//button[contains(@class, 'search')]"),
                (By.XPATH, "//button[contains(text(), 'Search')]"),
                (By.CSS_SELECTOR, ".search-button")
            ]
            
            for by, selector in button_selectors:
                try:
                    search_button = self.driver.find_element(by, selector)
                    break
                except:
                    continue
            
            if search_button:
                self.driver.execute_script("arguments[0].click();", search_button)
                logger.info("✅ Botón de búsqueda clickeado")
            else:
                # Intentar presionar Enter
                from selenium.webdriver.common.keys import Keys
                search_box.send_keys(Keys.RETURN)
                logger.info("✅ Enter presionado en campo de búsqueda")
            
            # Esperar resultados
            time.sleep(5)
            
            # Scroll para cargar resultados
            self.driver.execute_script("window.scrollTo(0, 500);")
            time.sleep(2)
            
            # Buscar contador de resultados
            try:
                result_count_selectors = [
                    "span.hitsLength",
                    ".result-count",
                    ".search-result__count",
                    "[class*='result-count']"
                ]
                
                for selector in result_count_selectors:
                    try:
                        count_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                        count_text = count_elem.text
                        logger.info(f"📊 Resultados encontrados: {count_text}")
                        break
                    except:
                        continue
            except:
                logger.debug("No se pudo obtener contador de resultados")
            
            # Extraer artículos con múltiples selectores
            articles = []
            article_selectors = [
                "li.search__item",
                ".search-result-item",
                "[class*='search-item']",
                "article.search-result"
            ]
            
            article_elements = []
            for selector in article_selectors:
                try:
                    article_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if article_elements:
                        logger.info(f"✅ Elementos de artículos encontrados: {len(article_elements)} con selector: {selector}")
                        break
                except:
                    continue
            
            if not article_elements:
                logger.warning("⚠️ No se encontraron elementos de artículos con selectores estándar")
                logger.info("🔍 Intentando selector genérico...")
                # Intento genérico
                article_elements = self.driver.find_elements(By.TAG_NAME, "article")
            
            logger.info(f"📚 Procesando {min(len(article_elements), max_results)} artículos...")
            
            for i, article in enumerate(article_elements[:max_results], 1):
                try:
                    # Scroll hasta el artículo
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", article)
                    time.sleep(0.2)
                    
                    # Extraer título
                    title = ""
                    title_selectors = ["h3", "h2", "h4", "[class*='title']", "a.search__item__title"]
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
                    author_selectors = [
                        "[class*='author']",
                        "[class*='contributor']",
                        ".author-name"
                    ]
                    for sel in author_selectors:
                        try:
                            author_elements = article.find_elements(By.CSS_SELECTOR, sel)
                            for author in author_elements:
                                author_text = author.text.strip()
                                if author_text and len(author_text) > 2:
                                    authors.append(author_text)
                            if authors:
                                break
                        except:
                            continue
                    
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
                        year_selectors = [
                            "[class*='date']",
                            "[class*='year']",
                            "time"
                        ]
                        for sel in year_selectors:
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
                    
                    # Extraer abstract (si está visible)
                    abstract = ""
                    try:
                        abstract_selectors = [
                            "[class*='abstract']",
                            ".snippet",
                            ".description"
                        ]
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
                    
                    # Crear entrada de artículo
                    article_data = {
                        "title": title,
                        "authors": authors,
                        "doi": doi,
                        "year": year,
                        "source": "ACM Digital Library (Uniquindio)",
                        "abstract": abstract,
                        "keywords": [],
                        "url": f"https://dl.acm.org/doi/{doi}" if doi else ""
                    }
                    
                    articles.append(article_data)
                    logger.info(f"  {i}. {title[:60]}...")
                
                except Exception as e:
                    logger.debug(f"Error extrayendo artículo {i}: {e}")
                    continue
            
            logger.info(f"✅ Total de artículos extraídos de ACM: {len(articles)}")
            return articles
        
        except Exception as e:
            logger.error(f"❌ Error en búsqueda ACM: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def _search_ieee(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Búsqueda específica en IEEE Xplore"""
        logger.info("📚 Ejecutando búsqueda en IEEE Xplore")
        # TODO: Implementar búsqueda en IEEE
        logger.warning("⚠️ Búsqueda en IEEE aún no implementada")
        return []

    def _search_sciencedirect(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Búsqueda específica en ScienceDirect"""
        logger.info("📚 Ejecutando búsqueda en ScienceDirect")
        # TODO: Implementar búsqueda en ScienceDirect
        logger.warning("⚠️ Búsqueda en ScienceDirect aún no implementada")
        return []

    def _search_scopus(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Búsqueda específica en Scopus"""
        logger.info("📚 Ejecutando búsqueda en Scopus")
        # TODO: Implementar búsqueda en Scopus
        logger.warning("⚠️ Búsqueda en Scopus aún no implementada")
        return []

    def _search_generic(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Búsqueda genérica para bases de datos no reconocidas"""
        logger.warning("⚠️ Usando búsqueda genérica")
        return []

    def scrape(self, query: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Método principal de scraping
        
        Args:
            query: Término de búsqueda
            max_results: Número máximo de resultados
            
        Returns:
            Diccionario con resultados de todas las bases de datos
        """
        logger.info("=" * 80)
        logger.info("🎓 SCRAPING A TRAVÉS DEL PORTAL INSTITUCIONAL UNIQUINDÍO")
        logger.info("=" * 80)
        
        all_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "portal": self.PORTAL_URL,
            "databases": {},
            "total_records": 0,
            "status": "success"
        }
        
        try:
            # Paso 1: Descubrir bases de datos disponibles
            databases = self.discover_databases()
            
            if not databases:
                logger.error("❌ No se encontraron bases de datos disponibles")
                all_results["status"] = "error"
                all_results["error"] = "No databases found"
                return all_results
            
            # Paso 2: Acceder y buscar en cada base de datos objetivo
            for db_name in self.TARGET_DATABASES:
                logger.info(f"\n{'='*60}")
                logger.info(f"Procesando: {db_name}")
                logger.info(f"{'='*60}")
                
                if self.access_database(db_name):
                    articles = self.search_in_database(query, max_results)
                    
                    if articles:
                        all_results["databases"][db_name] = {
                            "count": len(articles),
                            "articles": articles
                        }
                        all_results["total_records"] += len(articles)
                        logger.info(f"✅ {db_name}: {len(articles)} artículos extraídos")
                    else:
                        logger.warning(f"⚠️ {db_name}: Sin resultados")
                else:
                    logger.error(f"❌ No se pudo acceder a {db_name}")
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ SCRAPING COMPLETADO")
            logger.info(f"Total de artículos: {all_results['total_records']}")
            logger.info(f"Bases de datos procesadas: {len(all_results['databases'])}")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"❌ Error en scraping: {e}")
            all_results["status"] = "error"
            all_results["error"] = str(e)
        
        finally:
            self.close()
        
        return all_results


if __name__ == "__main__":
    # Test del scraper
    from src.utils.config_loader import get_config
    
    config = get_config()
    scraper = UniquindioPortalScraper(config, headless=False)
    
    # Realizar scraping
    results = scraper.scrape(
        query="generative artificial intelligence",
        max_results=20
    )
    
    print(f"\n✅ Total de artículos: {results['total_records']}")
    for db_name, db_data in results.get('databases', {}).items():
        print(f"  - {db_name}: {db_data['count']} artículos")
