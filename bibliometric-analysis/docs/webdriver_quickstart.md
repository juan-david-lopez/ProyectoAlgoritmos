# Chrome WebDriver - Guía de Inicio Rápido

## Instalación (5 minutos)

### 1. Instalar Google Chrome
```bash
# Ya lo tienes? Verificar:
google-chrome --version
```

Si no lo tienes:
- **Windows**: Descarga desde https://www.google.com/chrome/
- **Linux**: `sudo apt install google-chrome-stable`
- **macOS**: Descarga desde https://www.google.com/chrome/

### 2. Instalar Dependencias Python
```bash
cd bibliometric-analysis
pip install -r requirements.txt
```

### 3. Probar Instalación
```bash
cd ..
python test_webdriver_complete.py
```

## Uso en 3 Líneas

```python
from src.utils.webdriver_manager import WebDriverManager

with WebDriverManager(headless=True) as manager:
    manager.driver.get("https://example.com")
    # ¡Listo!
```

## Ejemplos Comunes

### Ejemplo 1: Navegar y Extraer Texto
```python
from src.utils.webdriver_manager import WebDriverManager
from selenium.webdriver.common.by import By

with WebDriverManager() as manager:
    driver = manager.driver
    driver.get("https://example.com")
    
    # Extraer título
    title = driver.find_element(By.TAG_NAME, "h1").text
    print(f"Título: {title}")
```

### Ejemplo 2: Buscar en Google
```python
from src.utils.webdriver_manager import WebDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

with WebDriverManager(headless=False) as manager:
    driver = manager.driver
    
    # Ir a Google
    driver.get("https://www.google.com")
    
    # Buscar
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys("machine learning")
    search_box.send_keys(Keys.RETURN)
    
    # Esperar resultados
    manager.wait_for_element(By.ID, "search")
    print("Búsqueda completada!")
```

### Ejemplo 3: Descargar Archivo
```python
from src.utils.webdriver_manager import WebDriverManager
from selenium.webdriver.common.by import By

with WebDriverManager(download_dir="./downloads") as manager:
    driver = manager.driver
    driver.get("https://example.com/files")
    
    # Click en botón de descarga
    btn = driver.find_element(By.CLASS_NAME, "download-btn")
    btn.click()
    
    # Esperar descarga
    if manager.wait_for_download_complete():
        print("Archivo descargado!")
```

## Modos

### Desarrollo (ver navegador)
```python
manager = WebDriverManager(headless=False)
```

### Producción (invisible)
```python
manager = WebDriverManager(headless=True)
```

## Debugging

### Tomar Screenshot
```python
with WebDriverManager() as manager:
    driver = manager.driver
    driver.get("https://example.com")
    manager.take_screenshot("debug.png")
    # Ver en: logs/debug.png
```

### Ver Logs
```python
from loguru import logger
logger.info("Tu mensaje")
```

## Problemas Comunes

### "ChromeDriver not found"
**Solución**: Se descarga automáticamente en primera ejecución. Asegura conexión a internet.

### "Chrome version mismatch"
**Solución**: 
```bash
# Actualizar Chrome
sudo apt upgrade google-chrome-stable  # Linux
# O descargar última versión manualmente
```

### Navegador no cierra
**Solución**: Usa `with` statement (context manager)

## Siguiente Paso

Lee la documentación completa en `docs/webdriver_setup.md`
