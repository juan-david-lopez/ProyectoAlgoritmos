"""
Test simple del WebDriver Manager
"""
import sys
import os

# Cambiar al directorio del proyecto
os.chdir('bibliometric-analysis')
sys.path.insert(0, os.getcwd())

from src.utils.webdriver_manager import WebDriverManager
from selenium.webdriver.common.by import By

print("="*70)
print("TEST RÁPIDO - CHROME WEBDRIVER")
print("="*70)

# Test 1: Inicialización
print("\n1. Probando inicialización...")
try:
    with WebDriverManager(headless=True) as manager:
        print("   ✓ WebDriver inicializado correctamente")
        print(f"   ✓ Navegador: {manager.driver.name}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Navegación
print("\n2. Probando navegación a Google...")
try:
    with WebDriverManager(headless=True) as manager:
        manager.driver.get("https://www.google.com")
        title = manager.driver.title
        print(f"   ✓ Navegación exitosa")
        print(f"   ✓ Título: {title}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Anti-detección
print("\n3. Probando anti-detección...")
try:
    with WebDriverManager(headless=True) as manager:
        manager.driver.get("https://www.google.com")
        
        webdriver_detected = manager.driver.execute_script(
            "return navigator.webdriver"
        )
        
        print(f"   ✓ navigator.webdriver: {webdriver_detected}")
        
        if webdriver_detected is None or webdriver_detected is False:
            print("   ✓ Anti-detección funcionando")
        else:
            print("   ⚠ WebDriver detectable")
            
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Comportamiento humano
print("\n4. Probando comportamiento humano...")
try:
    import time
    with WebDriverManager(headless=True) as manager:
        manager.driver.get("https://www.google.com")
        
        start = time.time()
        manager.human_delay(0.5, 1.0)
        elapsed = time.time() - start
        
        print(f"   ✓ Delay aleatorio: {elapsed:.2f}s")
        
        manager.scroll_page()
        print("   ✓ Scroll simulado")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✓ TODOS LOS TESTS PASARON!")
print("="*70)
print("\nEl WebDriver está configurado correctamente y listo para usar.")
print("Ver documentación en: docs/webdriver_setup.md")
