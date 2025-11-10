"""
🔍 Verificador de Configuración - Scraper Uniquindío
Este script verifica que todo está listo para ejecutar el scraper
"""

import os
import sys
from pathlib import Path

def check_module(module_name, package_name=None):
    """Verifica si un módulo está instalado"""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"  ✅ {package_name}")
        return True
    except ImportError:
        print(f"  ❌ {package_name} - FALTA")
        return False

def main():
    print("\n" + "="*70)
    print("🔍 VERIFICADOR DE CONFIGURACIÓN")
    print("="*70 + "\n")
    
    all_ok = True
    
    # 1. Verificar Python
    print("1️⃣ Versión de Python:")
    python_version = sys.version_info
    print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("  ⚠️ Se recomienda Python 3.8 o superior")
        all_ok = False
    print()
    
    # 2. Verificar dependencias
    print("2️⃣ Dependencias instaladas:")
    modules_to_check = [
        ("selenium", "selenium"),
        ("dotenv", "python-dotenv"),
    ]
    
    for module, package in modules_to_check:
        if not check_module(module, package):
            all_ok = False
    print()
    
    # 3. Verificar archivo .env
    print("3️⃣ Archivo de credenciales (.env):")
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_example_path.exists():
        print("  ⚠️ .env.example no encontrado")
        all_ok = False
    else:
        print("  ✅ .env.example existe")
    
    if not env_path.exists():
        print("  ❌ .env NO EXISTE")
        print("     💡 Solución:")
        print("        Copy-Item .env.example .env")
        print("        notepad .env")
        all_ok = False
    else:
        print("  ✅ .env existe")
        
        # Verificar si tiene contenido
        from dotenv import load_dotenv
        load_dotenv()
        
        email = os.getenv("EMAIL")
        password = os.getenv("PASSWORD")
        
        if not email or "tu_correo" in email:
            print("  ⚠️ EMAIL no configurado o es plantilla")
            print(f"     Valor actual: {email or 'vacío'}")
            all_ok = False
        else:
            print(f"  ✅ EMAIL configurado: {email}")
        
        if not password or "tu_contraseña" in password:
            print("  ⚠️ PASSWORD no configurado o es plantilla")
            all_ok = False
        else:
            print(f"  ✅ PASSWORD configurado: {'*' * len(password)}")
    print()
    
    # 4. Verificar carpetas de salida
    print("4️⃣ Carpetas de salida:")
    output_dir = Path("data/raw/uniquindio")
    
    if not output_dir.exists():
        print(f"  ⚠️ {output_dir} no existe (se creará automáticamente)")
    else:
        print(f"  ✅ {output_dir} existe")
        
        # Ver archivos existentes
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            print(f"     📁 {len(json_files)} archivos JSON existentes")
            for f in json_files[-3:]:  # Mostrar últimos 3
                size_kb = f.stat().st_size / 1024
                print(f"        • {f.name} ({size_kb:.1f} KB)")
    print()
    
    # 5. Verificar ChromeDriver
    print("5️⃣ ChromeDriver:")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        print("  🔍 Intentando iniciar Chrome...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.quit()
        print("  ✅ ChromeDriver funciona correctamente")
    except Exception as e:
        print(f"  ❌ Error con ChromeDriver: {str(e)[:100]}")
        print("     💡 Solución:")
        print("        1. Asegúrate de tener Chrome instalado")
        print("        2. Selenium descarga ChromeDriver automáticamente")
        print("        3. Si falla, instala manualmente desde:")
        print("           https://chromedriver.chromium.org/")
        all_ok = False
    print()
    
    # 6. Verificar scraper
    print("6️⃣ Script principal:")
    scraper_path = Path("scraper_uniquindio_completo.py")
    
    if not scraper_path.exists():
        print("  ❌ scraper_uniquindio_completo.py NO ENCONTRADO")
        all_ok = False
    else:
        size_kb = scraper_path.stat().st_size / 1024
        print(f"  ✅ scraper_uniquindio_completo.py ({size_kb:.1f} KB)")
    print()
    
    # Resultado final
    print("="*70)
    if all_ok:
        print("✅ TODO LISTO PARA EJECUTAR")
        print("="*70)
        print("\n💡 Ejecuta el scraper con:")
        print("   python scraper_uniquindio_completo.py")
        print()
    else:
        print("⚠️ HAY PROBLEMAS QUE RESOLVER")
        print("="*70)
        print("\n💡 Sigue las soluciones indicadas arriba")
        print("\n📚 Documentación:")
        print("   • COMO_USAR_SCRAPER.md - Guía rápida")
        print("   • GUIA_SCRAPER_AUTO.md - Guía detallada")
        print("   • IMPLEMENTACION_FINAL_SCRAPER.md - Detalles técnicos")
        print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
