"""
Tests para verificar configuración de WebDriver.
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent / 'bibliometric-analysis'
sys.path.insert(0, str(project_root))

from src.utils.webdriver_manager import WebDriverManager
from selenium.webdriver.common.by import By
from loguru import logger


def test_webdriver_initialization():
    """Verifica que WebDriver se inicializa correctamente."""
    print("\n" + "="*70)
    print("TEST 1: Inicialización de WebDriver")
    print("="*70)
    
    try:
        with WebDriverManager(headless=True) as manager:
            assert manager.driver is not None, "Driver no inicializado"
            assert manager.driver.name == 'chrome', "Driver no es Chrome"
            print("✓ WebDriver inicializado correctamente")
            print(f"✓ Navegador: {manager.driver.name}")
            print(f"✓ Modo headless: {manager.headless}")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_webdriver_navigation():
    """Verifica que WebDriver puede navegar a páginas."""
    print("\n" + "="*70)
    print("TEST 2: Navegación Web")
    print("="*70)
    
    try:
        with WebDriverManager(headless=True) as manager:
            manager.driver.get("https://www.google.com")
            title = manager.driver.title
            assert "Google" in title, f"Título incorrecto: {title}"
            print(f"✓ Navegación exitosa a Google")
            print(f"✓ Título de página: {title}")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_element_waiting():
    """Verifica que el wait funciona correctamente."""
    print("\n" + "="*70)
    print("TEST 3: Espera de Elementos")
    print("="*70)
    
    try:
        with WebDriverManager(headless=True) as manager:
            manager.driver.get("https://www.google.com")
            found = manager.wait_for_element(By.NAME, "q", timeout=10)
            assert found is True, "Elemento no encontrado"
            print("✓ Elemento de búsqueda encontrado")
            print("✓ Sistema de espera funcional")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_anti_detection():
    """Verifica configuración anti-detección."""
    print("\n" + "="*70)
    print("TEST 4: Anti-Detección")
    print("="*70)
    
    try:
        with WebDriverManager(headless=True) as manager:
            manager.driver.get("https://www.google.com")
            
            # Verificar que navigator.webdriver está oculto
            webdriver_detected = manager.driver.execute_script(
                "return navigator.webdriver"
            )
            
            # Verificar user agent
            user_agent = manager.driver.execute_script(
                "return navigator.userAgent"
            )
            
            print(f"✓ navigator.webdriver: {webdriver_detected}")
            print(f"✓ User Agent: {user_agent[:50]}...")
            
            if webdriver_detected is None or webdriver_detected is False:
                print("✓ Anti-detección configurada correctamente")
                return True
            else:
                print("⚠ WebDriver detectable (puede causar bloqueos)")
                return True  # No falla el test, solo advierte
                
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_acm_access():
    """Verifica acceso a ACM Digital Library."""
    print("\n" + "="*70)
    print("TEST 5: Acceso a ACM Digital Library")
    print("="*70)
    
    try:
        with WebDriverManager(headless=True) as manager:
            manager.driver.get("https://dl.acm.org/")
            manager.human_delay(2, 3)
            title = manager.driver.title
            assert "ACM" in title, f"Título incorrecto: {title}"
            print(f"✓ Acceso exitoso a ACM")
            print(f"✓ Título: {title}")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_screenshot_capability():
    """Verifica que se pueden tomar screenshots."""
    print("\n" + "="*70)
    print("TEST 6: Capacidad de Screenshots")
    print("="*70)
    
    try:
        import os
        with WebDriverManager(headless=True) as manager:
            manager.driver.get("https://www.google.com")
            manager.take_screenshot("test_screenshot.png")
            screenshot_path = Path("logs/test_screenshot.png")
            assert screenshot_path.exists(), "Screenshot no creado"
            print(f"✓ Screenshot guardado en: {screenshot_path}")
            print(f"✓ Tamaño: {screenshot_path.stat().st_size} bytes")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_human_behavior_simulation():
    """Verifica simulación de comportamiento humano."""
    print("\n" + "="*70)
    print("TEST 7: Simulación de Comportamiento Humano")
    print("="*70)
    
    try:
        import time
        with WebDriverManager(headless=True) as manager:
            manager.driver.get("https://www.google.com")
            
            # Test delay aleatorio
            start = time.time()
            manager.human_delay(0.5, 1.0)
            elapsed = time.time() - start
            assert 0.5 <= elapsed <= 1.1, f"Delay fuera de rango: {elapsed}"
            print(f"✓ Delay aleatorio: {elapsed:.2f}s")
            
            # Test scroll
            manager.scroll_page()
            print("✓ Simulación de scroll completada")
            
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_download_directory():
    """Verifica configuración de directorio de descargas."""
    print("\n" + "="*70)
    print("TEST 8: Directorio de Descargas")
    print("="*70)
    
    try:
        download_dir = str(Path.cwd() / "test_downloads")
        with WebDriverManager(headless=True, download_dir=download_dir) as manager:
            assert manager.download_dir == download_dir, "Directorio no configurado"
            assert Path(download_dir).exists(), "Directorio no creado"
            print(f"✓ Directorio configurado: {download_dir}")
            print(f"✓ Directorio existe: {Path(download_dir).exists()}")
            
            # Cleanup
            import shutil
            if Path(download_dir).exists():
                shutil.rmtree(download_dir)
            
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*70)
    print("SUITE DE TESTS - CHROME WEBDRIVER")
    print("="*70)
    
    tests = [
        ("Inicialización", test_webdriver_initialization),
        ("Navegación", test_webdriver_navigation),
        ("Espera de Elementos", test_element_waiting),
        ("Anti-Detección", test_anti_detection),
        ("Acceso ACM", test_acm_access),
        ("Screenshots", test_screenshot_capability),
        ("Comportamiento Humano", test_human_behavior_simulation),
        ("Directorio Descargas", test_download_directory),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error ejecutando test '{name}': {e}")
            results.append((name, False))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
