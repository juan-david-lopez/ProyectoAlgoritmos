# Configuración de Chrome WebDriver

## Descripción
El sistema utiliza Selenium con Chrome WebDriver para automatizar la descarga de artículos desde bases de datos académicas (ACM Digital Library, ScienceDirect, etc.).

## Instalación Automática
El sistema usa `webdriver-manager` para instalar automáticamente la versión correcta de ChromeDriver según tu versión de Chrome instalada.

**No necesitas descargar ChromeDriver manualmente** - se descarga automáticamente en la primera ejecución.

## Requisitos

### Software Necesario
- **Google Chrome** o **Chromium** instalado
- **Python 3.9+**
- Paquetes Python:
  - `selenium==4.15.2`
  - `webdriver-manager==4.0.1`

### Instalación de Dependencias
```bash
# Desde el directorio bibliometric-analysis
pip install -r requirements.txt
```

### Verificar Instalación de Chrome
```bash
# Linux/macOS
google-chrome --version
# o
chromium --version

# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --version
```

## Uso Básico

### Context Manager (Recomendado)
```python
from src.utils.webdriver_manager import WebDriverManager

# Uso con context manager (cierra automáticamente)
with WebDriverManager(headless=True) as manager:
    driver = manager.driver
    driver.get("https://example.com")
    # Tu código aquí...
# Se cierra automáticamente al salir del bloque
```

### Uso Manual
```python
from src.utils.webdriver_manager import WebDriverManager

# Crear manager
manager = WebDriverManager(headless=False, download_dir="./downloads")
driver = manager.create_driver()

try:
    driver.get("https://example.com")
    # ... operaciones ...
finally:
    manager.close()  # Importante: siempre cerrar
```

## Modos de Operación

### Modo Headless (Sin Interfaz Gráfica)
```python
manager = WebDriverManager(headless=True)
```

**Ventajas:**
- Más rápido (sin renderizado visual)
- Menos consumo de recursos
- Funciona en servidores sin display
- Ideal para producción

**Desventajas:**
- No puedes ver qué está pasando
- Más difícil de debuggear

### Modo Con Interfaz Gráfica
```python
manager = WebDriverManager(headless=False)
```

**Ventajas:**
- Ver el proceso en tiempo real
- Debugging visual
- Identificar problemas fácilmente
- Ideal para desarrollo

**Desventajas:**
- Más lento
- Más recursos
- Requiere display gráfico

## Características Avanzadas

### Anti-Detección
El WebDriver incluye configuración anti-detección para evitar bloqueos:

```python
# Automáticamente configurado
- navigator.webdriver = undefined
- User-Agent realista
- Propiedades de navegador reales (plugins, languages)
- Viewport realista
```

### Comportamiento Humano
```python
with WebDriverManager() as manager:
    driver = manager.driver
    driver.get("https://example.com")
    
    # Delay aleatorio (simula lectura)
    manager.human_delay(2, 4)  # Entre 2-4 segundos
    
    # Scroll de página (simula navegación)
    manager.scroll_page()
```

### Espera de Elementos
```python
from selenium.webdriver.common.by import By

with WebDriverManager() as manager:
    driver = manager.driver
    driver.get("https://example.com")
    
    # Esperar a que elemento exista
    found = manager.wait_for_element(By.ID, "search-box", timeout=20)
    
    # Esperar a que elemento sea clickeable
    clickable = manager.wait_for_clickable(By.CSS_SELECTOR, ".btn-submit", timeout=15)
```

### Screenshots para Debugging
```python
with WebDriverManager() as manager:
    driver = manager.driver
    driver.get("https://example.com")
    
    # Tomar screenshot
    manager.take_screenshot("debug_screenshot.png")
    # Guardado en: logs/debug_screenshot.png
```

### Esperar Descargas
```python
with WebDriverManager(download_dir="./downloads") as manager:
    driver = manager.driver
    driver.get("https://example.com")
    
    # Hacer clic en botón de descarga
    download_btn = driver.find_element(By.ID, "download")
    download_btn.click()
    
    # Esperar a que se complete (hasta 5 minutos)
    completed = manager.wait_for_download_complete(timeout=300)
    
    if completed:
        print("Descarga exitosa")
```

## Pool de WebDrivers (Avanzado)

Para procesamiento paralelo:

```python
from src.utils.webdriver_manager import WebDriverPool

# Crear pool de 3 drivers
pool = WebDriverPool(pool_size=3, headless=True)
pool.initialize_pool()

# Obtener driver del pool
driver = pool.get_driver()

try:
    # Usar driver...
    driver.get("https://example.com")
finally:
    # Devolver al pool
    pool.return_driver(driver)

# Cerrar todos cuando termines
pool.close_all()
```

## Troubleshooting

### Error: "ChromeDriver not found"
**Causa**: Primera ejecución o ChromeDriver no descargado.

**Solución**: webdriver-manager lo descargará automáticamente. Asegúrate de tener conexión a internet en la primera ejecución.

```python
# Se descarga automáticamente
service = Service(ChromeDriverManager().install())
```

### Error: "Chrome version mismatch"
**Causa**: ChromeDriver no coincide con tu versión de Chrome.

**Solución 1 - Actualizar Chrome**:
```bash
# Linux
sudo apt update && sudo apt upgrade google-chrome-stable

# Windows - Descarga desde google.com/chrome
```

**Solución 2 - Limpiar caché de webdriver-manager**:
```bash
# Linux/macOS
rm -rf ~/.wdm

# Windows
rmdir /s %USERPROFILE%\.wdm
```

### Error: "Session not created"
**Causa**: Chrome no está instalado o no se encuentra.

**Solución**:
```bash
# Verificar instalación
google-chrome --version

# Linux - Instalar Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb

# O instalar Chromium
sudo apt install chromium-browser
```

### Chrome no se cierra correctamente
**Causa**: No se llamó a `.close()` o hubo una excepción.

**Solución**: Siempre usar context manager o try-finally:
```python
# Opción 1: Context manager (recomendado)
with WebDriverManager() as manager:
    # tu código

# Opción 2: Try-finally
manager = WebDriverManager()
try:
    driver = manager.create_driver()
    # tu código
finally:
    manager.close()
```

### Error de permisos en Linux
**Causa**: ChromeDriver no tiene permisos de ejecución.

**Solución**:
```bash
# Dar permisos de ejecución
chmod +x ~/.wdm/drivers/chromedriver/linux64/*/chromedriver
```

### Timeout loading page
**Causa**: Página muy lenta o problemas de red.

**Solución**: Aumentar timeout:
```python
manager = WebDriverManager()
driver = manager.create_driver()
driver.set_page_load_timeout(120)  # 2 minutos
```

### "Element not interactable"
**Causa**: Elemento no visible o cubierto por otro.

**Solución**:
```python
# Hacer scroll al elemento
element = driver.find_element(By.ID, "button")
driver.execute_script("arguments[0].scrollIntoView(true);", element)

# Esperar un poco
manager.human_delay(1, 2)

# Clickear con JavaScript
driver.execute_script("arguments[0].click();", element)
```

## Configuraciones Avanzadas

### Cambiar User Agent
```python
# Ya está incluido en get_chrome_options()
# Para personalizar, modifica src/utils/webdriver_manager.py
user_agent = "Tu user agent personalizado"
chrome_options.add_argument(f'user-agent={user_agent}')
```

### Usar Proxy
```python
# En get_chrome_options()
chrome_options.add_argument('--proxy-server=http://proxy.example.com:8080')
```

### Desactivar Imágenes (Más Rápido)
```python
# En get_chrome_options()
prefs = {
    "profile.managed_default_content_settings.images": 2,
    # ... otras preferencias
}
chrome_options.add_experimental_option("prefs", prefs)
```

### Configurar Tamaño de Ventana
```python
# Ya configurado en 1920x1080
# Para cambiar:
chrome_options.add_argument('--window-size=1366,768')
```

## Docker

### Configuración para Contenedores

El Dockerfile debe incluir instalación de Chrome:

```dockerfile
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip

# Agregar repositorio de Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list

# Instalar Chrome
RUN apt-get update && apt-get install -y \
    google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# ChromeDriver se instala automáticamente via webdriver-manager
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tu código...
```

### Variables de Entorno para Docker
```dockerfile
# Desactivar sandbox (necesario en Docker)
ENV CHROME_NO_SANDBOX=1
```

## Testing

### Ejecutar Suite de Tests
```bash
python test_webdriver_complete.py
```

### Tests Incluidos
1. Inicialización de WebDriver
2. Navegación web
3. Espera de elementos
4. Anti-detección
5. Acceso a ACM
6. Captura de screenshots
7. Comportamiento humano
8. Directorio de descargas

## Best Practices

### 1. Siempre usa Context Manager
```python
# ✓ Correcto
with WebDriverManager() as manager:
    # código

# ✗ Incorrecto
manager = WebDriverManager()
driver = manager.create_driver()
# ... (si hay error, no se cierra)
```

### 2. Maneja Excepciones
```python
try:
    with WebDriverManager() as manager:
        driver = manager.driver
        driver.get("https://example.com")
        # operaciones...
except Exception as e:
    logger.error(f"Error: {e}")
    if manager:
        manager.take_screenshot("error.png")
```

### 3. Usa Esperas Explícitas
```python
# ✓ Correcto - espera explícita
manager.wait_for_element(By.ID, "element", timeout=20)

# ✗ Incorrecto - sleep fijo
import time
time.sleep(5)  # Muy lento o muy rápido
```

### 4. Simula Comportamiento Humano
```python
# Agrega delays aleatorios
manager.human_delay(1, 3)

# Simula scroll
manager.scroll_page()
```

### 5. Limita Recursos en Producción
```python
# Headless en producción
manager = WebDriverManager(headless=True)

# Timeout razonable
driver.set_page_load_timeout(60)
```

## Ejemplos Completos

### Ejemplo 1: Scraping Básico
```python
from src.utils.webdriver_manager import WebDriverManager
from selenium.webdriver.common.by import By

def scrape_example():
    with WebDriverManager(headless=True) as manager:
        driver = manager.driver
        
        # Navegar
        driver.get("https://example.com")
        
        # Esperar carga
        manager.wait_for_element(By.CLASS_NAME, "content")
        
        # Extraer datos
        title = driver.find_element(By.TAG_NAME, "h1").text
        
        return {"title": title}
```

### Ejemplo 2: Con Descarga
```python
def download_file():
    download_dir = "./downloads"
    
    with WebDriverManager(headless=True, download_dir=download_dir) as manager:
        driver = manager.driver
        driver.get("https://example.com/download")
        
        # Click download
        btn = driver.find_element(By.ID, "download-btn")
        btn.click()
        
        # Esperar descarga
        if manager.wait_for_download_complete(timeout=300):
            print("Descarga exitosa!")
```

## Recursos Adicionales

- [Documentación Selenium](https://www.selenium.dev/documentation/)
- [WebDriver Manager](https://github.com/SergeyPirogov/webdriver_manager)
- [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)

## Soporte

Si encuentras problemas:
1. Revisa esta documentación
2. Ejecuta `python test_webdriver_complete.py`
3. Verifica logs en `logs/`
4. Captura screenshots con `.take_screenshot()`
