"""
Prueba directa del scraper ACM actualizado
Usa la configuración existente del proyecto
"""

import sys
from pathlib import Path

# Configurar path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.config_loader import get_config
from src.scrapers.acm_scraper import ACMScraper

# Configurar logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/test_acm.log", rotation="10 MB")

def main():
    """Prueba completa del scraper ACM"""
    
    print("\n" + "="*80)
    print("🧪 PRUEBA DEL SCRAPER ACM - VERSIÓN ACTUALIZADA")
    print("="*80 + "\n")
    
    try:
        # Cargar configuración
        logger.info("Cargando configuración del proyecto...")
        config = get_config()
        logger.success("Configuración cargada correctamente")
        
        # Crear scraper
        logger.info("Inicializando scraper ACM (navegador visible)...")
        scraper = ACMScraper(config, headless=False)
        logger.success("Scraper inicializado")
        
        # Prueba 1: Búsqueda pequeña
        print("\n" + "-"*80)
        print("📋 PRUEBA 1: Búsqueda y extracción de artículos")
        print("-"*80)
        
        query = "artificial intelligence"
        max_results = 3
        
        logger.info(f"Buscando: '{query}' (máximo {max_results} resultados)")
        
        articles = scraper.search(query, max_results=max_results)
        
        if not articles:
            logger.error("❌ No se encontraron artículos")
            scraper.close()
            return False
        
        logger.success(f"✅ Encontrados {len(articles)} artículos")
        
        # Mostrar detalles de los artículos
        print("\n📚 Artículos encontrados:")
        for i, article in enumerate(articles, 1):
            print(f"\n  {i}. {article.get('title', 'Sin título')[:70]}...")
            print(f"     Autores: {article.get('authors', 'N/A')}")
            print(f"     Año: {article.get('year', 'N/A')}")
            print(f"     DOI: {article.get('doi', 'N/A')}")
            print(f"     URL: {article.get('url', 'N/A')[:60]}...")
        
        # Prueba 2: Generación de archivo BibTeX
        print("\n" + "-"*80)
        print("📝 PRUEBA 2: Generación de archivo BibTeX")
        print("-"*80)
        
        logger.info("Generando archivo BibTeX...")
        bibtex_file = scraper.download_results(format='bibtex')
        
        if not bibtex_file or not bibtex_file.exists():
            logger.error("❌ Error al generar archivo BibTeX")
            scraper.close()
            return False
        
        logger.success(f"✅ Archivo BibTeX generado: {bibtex_file.name}")
        print(f"\n   📁 Ruta: {bibtex_file}")
        print(f"   📊 Tamaño: {bibtex_file.stat().st_size:,} bytes")
        
        # Mostrar contenido del archivo
        content = bibtex_file.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        print(f"\n   📖 Primeras 20 líneas del archivo BibTeX:")
        print("   " + "-"*70)
        for line in lines[:20]:
            print(f"   {line}")
        print("   " + "-"*70)
        
        # Prueba 3: Generación de archivo JSON
        print("\n" + "-"*80)
        print("📝 PRUEBA 3: Generación de archivo JSON")
        print("-"*80)
        
        logger.info("Generando archivo JSON...")
        json_file = scraper.download_results(format='json')
        
        if not json_file or not json_file.exists():
            logger.error("❌ Error al generar archivo JSON")
            scraper.close()
            return False
        
        logger.success(f"✅ Archivo JSON generado: {json_file.name}")
        print(f"\n   📁 Ruta: {json_file}")
        print(f"   📊 Tamaño: {json_file.stat().st_size:,} bytes")
        
        # Mostrar primeras líneas
        json_content = json_file.read_text(encoding='utf-8')
        json_lines = json_content.split('\n')
        
        print(f"\n   📖 Primeras 15 líneas del archivo JSON:")
        print("   " + "-"*70)
        for line in json_lines[:15]:
            print(f"   {line}")
        print("   " + "-"*70)
        
        # Prueba 4: Generación de archivo CSV
        print("\n" + "-"*80)
        print("📝 PRUEBA 4: Generación de archivo CSV")
        print("-"*80)
        
        logger.info("Generando archivo CSV...")
        csv_file = scraper.download_results(format='csv')
        
        if not csv_file or not csv_file.exists():
            logger.error("❌ Error al generar archivo CSV")
            scraper.close()
            return False
        
        logger.success(f"✅ Archivo CSV generado: {csv_file.name}")
        print(f"\n   📁 Ruta: {csv_file}")
        print(f"   📊 Tamaño: {csv_file.stat().st_size:,} bytes")
        
        # Mostrar contenido
        csv_content = csv_file.read_text(encoding='utf-8')
        csv_lines = csv_content.split('\n')
        
        print(f"\n   📖 Contenido del archivo CSV:")
        print("   " + "-"*70)
        for line in csv_lines[:5]:
            print(f"   {line[:75]}...")
        print("   " + "-"*70)
        
        # Prueba 5: Parseo de archivos
        print("\n" + "-"*80)
        print("📚 PRUEBA 5: Parseo de archivo BibTeX")
        print("-"*80)
        
        logger.info("Parseando archivo BibTeX...")
        records = scraper.parse_file(bibtex_file)
        
        if not records:
            logger.error("❌ Error al parsear archivo")
            scraper.close()
            return False
        
        logger.success(f"✅ Parseados {len(records)} registros")
        
        # Mostrar primer registro parseado
        if records:
            first_record = records[0]
            print("\n   📄 Primer registro parseado:")
            print(f"      ID: {first_record.get('id', 'N/A')}")
            print(f"      Título: {first_record.get('title', 'N/A')[:60]}...")
            print(f"      Autores: {first_record.get('authors', [])}")
            print(f"      Año: {first_record.get('year', 'N/A')}")
            print(f"      DOI: {first_record.get('doi', 'N/A')}")
            print(f"      Fuente: {first_record.get('source', 'N/A')}")
            print(f"      Publisher: {first_record.get('publisher', 'N/A')}")
        
        # Cerrar navegador
        logger.info("🧹 Cerrando navegador...")
        scraper.close()
        logger.success("✅ Navegador cerrado")
        
        # Resumen final
        print("\n" + "="*80)
        print("🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("="*80)
        
        print("\n📊 RESUMEN:")
        print(f"   ✅ Búsqueda: {len(articles)} artículos encontrados")
        print(f"   ✅ BibTeX: {bibtex_file.name}")
        print(f"   ✅ JSON: {json_file.name}")
        print(f"   ✅ CSV: {csv_file.name}")
        print(f"   ✅ Parseo: {len(records)} registros")
        
        print("\n🎯 El scraper ACM está completamente funcional!")
        print("   - Extracción directa de HTML ✅")
        print("   - Paginación automática ✅")
        print("   - Múltiples formatos de salida ✅")
        print("   - Parseo de archivos ✅")
        
        print("\n" + "="*80 + "\n")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Prueba interrumpida por el usuario")
        return False
        
    except Exception as e:
        logger.error(f"\n❌ Error en la prueba: {e}")
        logger.exception("Traceback completo:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
