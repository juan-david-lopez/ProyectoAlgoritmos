"""
Test de Acceso - Portal Institucional Uniquindío
Verifica el acceso al portal y lista las bases de datos disponibles

Este script:
1. Intenta acceder al portal de la biblioteca
2. Lista todas las bases de datos disponibles
3. Verifica si requiere autenticación
4. NO realiza scraping completo (solo exploración)

Uso:
    python test_uniquindio_access.py
"""

import sys
from pathlib import Path
from loguru import logger

# Configurar logging simple
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

print("\n" + "="*80)
print("🎓 TEST DE ACCESO - PORTAL UNIQUINDÍO")
print("="*80)
print("\nVerificando acceso al portal institucional...")
print("URL: https://library.uniquindio.edu.co/databases\n")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    
    # Configurar Chrome
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Comentado para ver el navegador
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    logger.info("Iniciando navegador Chrome...")
    driver = webdriver.Chrome(options=chrome_options)
    driver.maximize_window()
    
    # Intentar acceder al portal
    logger.info("Accediendo al portal...")
    driver.get("https://library.uniquindio.edu.co/databases")
    time.sleep(5)
    
    # Verificar título de la página
    page_title = driver.title
    logger.info(f"Título de página: {page_title}")
    
    # Verificar URL actual
    current_url = driver.current_url
    logger.info(f"URL actual: {current_url}")
    
    # Verificar si requiere autenticación
    if "login" in current_url.lower() or "auth" in current_url.lower():
        logger.warning("⚠️ REQUIERE AUTENTICACIÓN")
        print("\n" + "="*80)
        print("⚠️ El portal requiere autenticación institucional")
        print("="*80)
        print("\n💡 Opciones:")
        print("  1. Conéctate a la red WiFi de Uniquindío")
        print("  2. Configura la VPN institucional")
        print("  3. Inicia sesión con tus credenciales @uniquindio.edu.co")
        print("\n⏳ Esperando 30 segundos para que inicies sesión manualmente...")
        print("   (Si no necesitas autenticarte, cierra este mensaje)")
        time.sleep(30)
        
        # Verificar si se autenticó
        current_url = driver.current_url
        if "login" not in current_url.lower():
            logger.success("✅ Autenticación exitosa")
    else:
        logger.success("✅ Acceso directo sin autenticación")
    
    # Buscar categorías de bases de datos
    logger.info("Buscando bases de datos disponibles...")
    
    print("\n" + "="*80)
    print("📚 BASES DE DATOS DISPONIBLES")
    print("="*80 + "\n")
    
    # Buscar por categorías específicas
    categories_found = []
    
    # Método 1: Buscar encabezados H2/H3
    try:
        headers = driver.find_elements(By.TAG_NAME, "h2") + driver.find_elements(By.TAG_NAME, "h3")
        for header in headers:
            text = header.text.strip()
            if text and len(text) > 5:
                categories_found.append(text)
        
        if categories_found:
            print("📂 Categorías encontradas:")
            for i, cat in enumerate(set(categories_found), 1):
                print(f"  {i}. {cat}")
    except Exception as e:
        logger.debug(f"Error buscando headers: {e}")
    
    # Método 2: Buscar enlaces con texto relevante
    try:
        print("\n🔗 Enlaces a bases de datos:")
        links = driver.find_elements(By.TAG_NAME, "a")
        db_links = []
        
        keywords = ["ACM", "IEEE", "Scopus", "Science", "Web of Science", "Engineering"]
        
        for link in links[:100]:  # Limitar a primeros 100 enlaces
            text = link.text.strip()
            href = link.get_attribute("href")
            
            if text and href and any(kw.lower() in text.lower() for kw in keywords):
                if text not in db_links:
                    db_links.append(text)
                    print(f"  • {text}")
                    logger.debug(f"    URL: {href}")
        
        if not db_links:
            logger.warning("No se encontraron enlaces obvios a bases de datos")
            print("\n⚠️ No se detectaron bases de datos de forma automática")
            print("💡 Posibles causas:")
            print("  - La página requiere interacción adicional")
            print("  - Las bases de datos están en un portal diferente")
            print("  - Necesitas navegar manualmente a la sección correcta")
    
    except Exception as e:
        logger.error(f"Error buscando enlaces: {e}")
    
    # Método 3: Buscar contenido específico del portal
    try:
        page_content = driver.page_source
        
        # Buscar términos clave
        if "Fac. Ingeniería" in page_content:
            logger.success("✅ Sección 'Fac. Ingeniería' encontrada")
        if "Fac. Ciencias Básicas" in page_content:
            logger.success("✅ Sección 'Fac. Ciencias Básicas' encontrada")
        if "ACM" in page_content:
            logger.success("✅ ACM Digital Library disponible")
        if "IEEE" in page_content:
            logger.success("✅ IEEE Xplore disponible")
    
    except Exception as e:
        logger.debug(f"Error analizando contenido: {e}")
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN DEL TEST")
    print("="*80)
    print(f"✅ Acceso al portal: {'EXITOSO' if driver.current_url else 'FALLIDO'}")
    print(f"📍 URL: {driver.current_url}")
    print(f"📄 Título: {driver.title}")
    print(f"🔐 Autenticación: {'Requerida' if 'login' in current_url.lower() else 'No requerida'}")
    print("="*80)
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("  1. Si viste bases de datos listadas arriba: ✅ Listo para scraping")
    print("  2. Si necesitas autenticación: Conéctate a la red institucional")
    print("  3. Ejecuta el scraper completo con: python run_uniquindio_portal.py")
    print()
    
    # Preguntar si desea continuar explorando
    print("\n⏳ El navegador quedará abierto por 60 segundos para que explores...")
    print("   Presiona Ctrl+C para cerrar antes")
    
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Cerrando navegador...")
    
    driver.quit()
    logger.success("✅ Test completado exitosamente")

except ImportError as e:
    print("\n❌ ERROR: Falta instalar dependencias")
    print(f"   {e}")
    print("\n💡 Solución:")
    print("   pip install selenium")
    print("   Descarga ChromeDriver: https://chromedriver.chromium.org/")

except Exception as e:
    logger.error(f"❌ Error durante el test: {e}")
    print(f"\n❌ Error: {e}")
    print("\n💡 Posibles soluciones:")
    print("  • Verifica que ChromeDriver esté instalado")
    print("  • Verifica tu conexión a internet")
    print("  • Intenta conectarte a la red institucional primero")
    
    import traceback
    print("\n🔍 Detalles del error:")
    traceback.print_exc()

finally:
    print("\n" + "="*80)
    print("Test finalizado")
    print("="*80 + "\n")
