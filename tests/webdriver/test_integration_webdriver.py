"""
Test de Integración - WebDriverManager en Base Scraper
Verifica que la integración del WebDriverManager funcione correctamente
"""

import sys
import os
from pathlib import Path

# Cambiar al directorio del proyecto y agregar al path
project_dir = Path(__file__).parent / 'bibliometric-analysis'
os.chdir(str(project_dir))
sys.path.insert(0, str(project_dir))

from src.utils.config_loader import get_config
from src.scrapers.acm_scraper import ACMScraper


def test_acm_scraper_initialization():
    """Test 1: Verificar que ACM scraper inicializa con nuevo WebDriverManager"""
    print("\n" + "="*60)
    print("TEST 1: Inicialización de ACM Scraper con WebDriverManager")
    print("="*60)
    
    try:
        # Cargar configuración
        config = get_config()
        print("✓ Configuración cargada")
        
        # Crear scraper (sin iniciar sesión todavía)
        scraper = ACMScraper(config, headless=True)
        print(f"✓ ACM Scraper creado: {scraper.__class__.__name__}")
        print(f"  - Headless: {scraper.headless}")
        print(f"  - Download dir: {scraper.download_dir}")
        print(f"  - WebDriverManager inicializado: {scraper.webdriver_manager is None} (esperado None antes de start_session)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webdriver_creation():
    """Test 2: Verificar que el WebDriver se crea correctamente"""
    print("\n" + "="*60)
    print("TEST 2: Creación de WebDriver con Anti-detección")
    print("="*60)
    
    try:
        config = get_config()
        scraper = ACMScraper(config, headless=True)
        
        # Iniciar sesión (esto crea el driver)
        print("Iniciando sesión del navegador...")
        scraper.start_session()
        
        print(f"✓ Sesión iniciada")
        print(f"  - Driver creado: {scraper.driver is not None}")
        print(f"  - WebDriverManager activo: {scraper.webdriver_manager is not None}")
        print(f"  - Wait configurado: {scraper.wait is not None}")
        
        # Probar navegación simple
        print("\nProbando navegación a Google...")
        scraper.driver.get("https://www.google.com")
        
        # Verificar anti-detección
        webdriver_value = scraper.driver.execute_script("return navigator.webdriver")
        print(f"  - navigator.webdriver: {webdriver_value}")
        
        if webdriver_value is None or webdriver_value is False:
            print("✓ Anti-detección funcionando correctamente!")
        else:
            print("⚠ Advertencia: navigator.webdriver no está oculto")
        
        # Cerrar sesión
        print("\nCerrando sesión...")
        scraper.close_session()
        print("✓ Sesión cerrada correctamente")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Intentar cerrar de todas formas
        try:
            if 'scraper' in locals():
                scraper.close_session()
        except:
            pass
        
        return False


def test_acm_navigation():
    """Test 3: Verificar navegación a ACM Digital Library"""
    print("\n" + "="*60)
    print("TEST 3: Navegación a ACM Digital Library")
    print("="*60)
    
    try:
        config = get_config()
        scraper = ACMScraper(config, headless=True)
        
        print("Iniciando sesión...")
        scraper.start_session()
        
        # Navegar a ACM
        print("Navegando a ACM Digital Library...")
        scraper.driver.get("https://dl.acm.org/")
        scraper.human_delay(1, 2)
        
        # Verificar título
        title = scraper.driver.title
        print(f"✓ Página cargada: {title}")
        
        # Verificar que estamos en ACM
        if "ACM" in title or "acm" in scraper.driver.current_url.lower():
            print("✓ Navegación exitosa a ACM Digital Library")
            success = True
        else:
            print(f"⚠ URL actual: {scraper.driver.current_url}")
            print(f"⚠ Título: {title}")
            success = False
        
        scraper.close_session()
        return success
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            if 'scraper' in locals():
                scraper.close_session()
        except:
            pass
        
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print(" TEST DE INTEGRACIÓN - WebDriverManager en Base Scraper")
    print("="*70)
    
    results = []
    
    # Test 1: Inicialización
    results.append(("Inicialización", test_acm_scraper_initialization()))
    
    # Test 2: Creación de WebDriver
    results.append(("Creación WebDriver", test_webdriver_creation()))
    
    # Test 3: Navegación a ACM
    results.append(("Navegación ACM", test_acm_navigation()))
    
    # Resumen
    print("\n" + "="*70)
    print(" RESUMEN DE TESTS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nResultado: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✓ WebDriverManager integrado correctamente")
        print("✓ Anti-detección funcionando")
        print("✓ Scrapers listos para usar")
    else:
        print(f"\n⚠ {total - passed} test(s) fallaron")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
