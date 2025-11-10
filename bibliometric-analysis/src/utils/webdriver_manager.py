"""
Gestor centralizado de Chrome WebDriver con configuración automática.
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from typing import Optional
import platform
from pathlib import Path
from loguru import logger as log


class WebDriverManager:
    """
    Gestor de Chrome WebDriver con configuración automática y multiplataforma.
    """
    
    def __init__(self, headless: bool = True, download_dir: Optional[str] = None):
        """
        Inicializa el gestor de WebDriver.
        
        Args:
            headless: Si True, ejecuta Chrome en modo headless (sin interfaz gráfica)
            download_dir: Directorio para descargas automáticas
        """
        self.headless = headless
        self.download_dir = download_dir or str(Path.cwd() / "data" / "raw")
        self.driver = None
        
    def get_chrome_options(self) -> Options:
        """
        Configura opciones optimizadas de Chrome.
        
        Configuraciones incluidas:
        - Headless mode (opcional)
        - Desactivar notificaciones
        - Desactivar GPU (para headless)
        - Maximizar ventana
        - Directorio de descarga personalizado
        - User agent realista
        - Desactivar automatización detectada
        
        Returns:
            Options: Objeto con configuraciones de Chrome
        """
        chrome_options = Options()
        
        # Modo headless
        if self.headless:
            chrome_options.add_argument('--headless=new')  # Nuevo modo headless
            chrome_options.add_argument('--disable-gpu')
        
        # Configuraciones generales
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--start-maximized')
        
        # Anti-detección mejorada
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        chrome_options.add_argument('--disable-site-isolation-trials')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-popup-blocking')
        chrome_options.add_argument('--disable-infobars')
        
        # Desactivar notificaciones y pop-ups
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-save-password-bubble')
        
        # Render improvements
        chrome_options.add_argument('--enable-features=NetworkService,NetworkServiceInProcess')
        chrome_options.add_argument('--disable-background-timer-throttling')
        chrome_options.add_argument('--disable-backgrounding-occluded-windows')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        
        # User agent realista (simular navegador normal)
        user_agent = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/142.0.0.0 Safari/537.36'
        )
        chrome_options.add_argument(f'user-agent={user_agent}')
        
        # Configurar descargas automáticas
        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        prefs = {
            'download.default_directory': str(Path(self.download_dir).absolute()),
            'download.prompt_for_download': False,
            'download.directory_upgrade': True,
            'safebrowsing.enabled': True,
            'profile.default_content_settings.popups': 0,
            'profile.content_settings.exceptions.automatic_downloads.*.setting': 1
        }
        chrome_options.add_experimental_option('prefs', prefs)
        
        # Excluir logs de automatización
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        log.info("Chrome options configuradas correctamente")
        return chrome_options
    
    def create_driver(self) -> webdriver.Chrome:
        """
        Crea y configura una instancia de Chrome WebDriver.
        
        Utiliza webdriver_manager para instalación automática del driver correcto.
        
        Returns:
            webdriver.Chrome: Instancia configurada de Chrome WebDriver
        """
        try:
            log.info("Inicializando Chrome WebDriver...")
            log.info(f"Sistema operativo: {platform.system()} {platform.release()}")
            
            # Instalar/obtener ChromeDriver automáticamente
            service = Service(ChromeDriverManager().install())
            
            # Crear driver con opciones configuradas
            chrome_options = self.get_chrome_options()
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Configurar timeouts
            driver.set_page_load_timeout(60)  # 60 segundos para cargar página
            driver.implicitly_wait(10)  # 10 segundos de espera implícita
            
            # Obtener user agent de las opciones
            user_agent = None
            for arg in chrome_options.arguments:
                if arg.startswith('user-agent='):
                    user_agent = arg.replace('user-agent=', '')
                    break
            
            # Ejecutar scripts para evitar detección de automatización
            if user_agent:
                driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                    "userAgent": user_agent
                })
            
            # Remove navigator.webdriver flag
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Modify Chrome object
            driver.execute_script("""
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en', 'es']
                });
            """)
            
            # Set realistic viewport
            driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
                'width': 1920,
                'height': 1080,
                'deviceScaleFactor': 1,
                'mobile': False
            })
            
            log.success("Chrome WebDriver inicializado exitosamente con anti-detección")
            self.driver = driver
            return driver
            
        except Exception as e:
            log.error(f"Error al crear Chrome WebDriver: {str(e)}")
            raise
    
    def wait_for_element(self, by: By, value: str, timeout: int = 20) -> bool:
        """
        Espera a que un elemento esté presente en la página.
        
        Args:
            by: Tipo de selector (By.ID, By.XPATH, By.CSS_SELECTOR, etc.)
            value: Valor del selector
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            bool: True si el elemento se encuentra, False si timeout
        """
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            log.debug(f"Elemento encontrado: {value}")
            return True
        except Exception as e:
            log.warning(f"Timeout esperando elemento: {value}")
            return False
    
    def wait_for_clickable(self, by: By, value: str, timeout: int = 20) -> bool:
        """
        Espera a que un elemento sea clickeable.
        
        Args:
            by: Tipo de selector
            value: Valor del selector
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            bool: True si el elemento es clickeable, False si timeout
        """
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            log.debug(f"Elemento clickeable: {value}")
            return True
        except Exception as e:
            log.warning(f"Timeout esperando elemento clickeable: {value}")
            return False
    
    def wait_for_download_complete(self, timeout: int = 300) -> bool:
        """
        Espera a que se complete una descarga.
        
        Args:
            timeout: Tiempo máximo de espera en segundos (default 5 minutos)
            
        Returns:
            bool: True si descarga completada, False si timeout
        """
        import time
        download_path = Path(self.download_dir)
        
        log.info("Esperando a que se complete la descarga...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Buscar archivos .crdownload (Chrome downloading)
            downloading_files = list(download_path.glob("*.crdownload"))
            
            if not downloading_files:
                # Verificar si hay archivos recién descargados
                files = list(download_path.glob("*"))
                recent_files = [f for f in files if f.is_file() and 
                              time.time() - f.stat().st_mtime < 5]
                
                if recent_files:
                    log.success(f"Descarga completada: {recent_files[0].name}")
                    return True
            
            time.sleep(1)
        
        log.error("Timeout esperando descarga")
        return False
    
    def human_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0):
        """
        Delay aleatorio para simular comportamiento humano.
        
        Args:
            min_seconds: Mínimo de segundos
            max_seconds: Máximo de segundos
        """
        import random
        import time
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def scroll_page(self):
        """Simula scroll de página para parecer más humano."""
        try:
            # Scroll down slowly
            self.driver.execute_script("""
                window.scrollTo({
                    top: document.body.scrollHeight / 2,
                    behavior: 'smooth'
                });
            """)
            self.human_delay(0.5, 1.5)
            
            # Scroll back up a bit
            self.driver.execute_script("""
                window.scrollTo({
                    top: document.body.scrollHeight / 4,
                    behavior: 'smooth'
                });
            """)
            self.human_delay(0.3, 0.8)
        except Exception as e:
            log.debug(f"Scroll simulation failed: {e}")
    
    def take_screenshot(self, filename: str = "screenshot.png"):
        """
        Captura screenshot de la página actual (útil para debugging).
        
        Args:
            filename: Nombre del archivo de screenshot
        """
        if self.driver:
            screenshot_path = Path("logs") / filename
            screenshot_path.parent.mkdir(exist_ok=True, parents=True)
            self.driver.save_screenshot(str(screenshot_path))
            log.info(f"Screenshot guardado: {screenshot_path}")
    
    def close(self):
        """Cierra el navegador y limpia recursos."""
        if self.driver:
            try:
                self.driver.quit()
                log.info("Chrome WebDriver cerrado correctamente")
            except Exception as e:
                log.warning(f"Error al cerrar WebDriver: {str(e)}")
            finally:
                self.driver = None
    
    def __enter__(self):
        """Context manager entry."""
        self.create_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False  # Don't suppress exceptions


class WebDriverPool:
    """
    Pool de WebDrivers para procesamiento paralelo (opcional, avanzado).
    """
    
    def __init__(self, pool_size: int = 3, headless: bool = True):
        """
        Inicializa pool de WebDrivers.
        
        Args:
            pool_size: Número de drivers en el pool
            headless: Modo headless para todos los drivers
        """
        self.pool_size = pool_size
        self.headless = headless
        self.drivers = []
        self.managers = []
        
    def initialize_pool(self):
        """Inicializa todos los drivers del pool."""
        log.info(f"Inicializando pool de {self.pool_size} WebDrivers...")
        
        for i in range(self.pool_size):
            try:
                manager = WebDriverManager(headless=self.headless)
                driver = manager.create_driver()
                self.drivers.append(driver)
                self.managers.append(manager)
                log.info(f"Driver {i+1}/{self.pool_size} inicializado")
            except Exception as e:
                log.error(f"Error inicializando driver {i+1}: {str(e)}")
        
        log.success(f"Pool de {len(self.drivers)} drivers listo")
    
    def get_driver(self) -> Optional[webdriver.Chrome]:
        """Obtiene un driver disponible del pool."""
        return self.drivers.pop() if self.drivers else None
    
    def return_driver(self, driver: webdriver.Chrome):
        """Devuelve un driver al pool."""
        self.drivers.append(driver)
    
    def close_all(self):
        """Cierra todos los drivers del pool."""
        log.info("Cerrando todos los drivers del pool...")
        for manager in self.managers:
            try:
                manager.close()
            except Exception as e:
                log.warning(f"Error cerrando driver: {str(e)}")
        self.drivers.clear()
        self.managers.clear()
        log.success("Todos los drivers cerrados")
