"""
Script simple para capturar el HTML real de ACM y ver qué contiene
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from pathlib import Path

# Crear driver básico
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    # Navegar
    url = "https://dl.acm.org/action/doSearch?AllField=artificial%20intelligence"
    print(f"Navegando a: {url}")
    driver.get(url)
    
    # Esperar 10 segundos
    print("Esperando 10 segundos para que cargue...")
    time.sleep(10)
    
    # Guardar HTML
    html_file = Path("logs/acm_raw.html")
    html_file.parent.mkdir(exist_ok=True)
    html_file.write_text(driver.page_source, encoding='utf-8')
    print(f"✅ HTML guardado en: {html_file}")
    
    # Guardar screenshot
    screenshot_file = Path("logs/acm_raw.png")
    driver.save_screenshot(str(screenshot_file))
    print(f"✅ Screenshot guardado en: {screenshot_file}")
    
    # Mostrar primeras 100 líneas
    lines = driver.page_source.split('\n')[:100]
    print("\n📄 Primeras 100 líneas del HTML:\n")
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line[:120]}")
    
    # Buscar "result" en el texto visible
    body_text = driver.find_element(By.TAG_NAME, "body").text
    print(f"\n📝 Texto visible de la página (primeros 2000 caracteres):\n")
    print(body_text[:2000])
    
    print("\n⏳ Manteniendo navegador abierto 30 segundos para inspección...")
    time.sleep(30)
    
finally:
    driver.quit()
    print("\n✅ Completado")
