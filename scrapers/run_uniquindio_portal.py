"""
Script de Ejecución - Portal Institucional Uniquindío
Acceso automatizado a bases de datos académicas

Uso:
    python run_uniquindio_portal.py
    
    O con parámetros:
    python run_uniquindio_portal.py --query "generative AI" --max-results 50
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# Agregar directorio bibliometric-analysis al path
sys.path.insert(0, str(Path(__file__).parent / "bibliometric-analysis"))

from src.utils.config_loader import get_config
from src.scrapers.uniquindio_portal_scraper import UniquindioPortalScraper


def setup_logging():
    """Configurar logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"uniquindio_portal_{timestamp}.log"
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )


def save_results(results: dict, output_dir: Path):
    """
    Guardar resultados en archivos JSON
    
    Args:
        results: Diccionario con resultados del scraping
        output_dir: Directorio de salida
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar resultados completos
    full_results_file = output_dir / f"uniquindio_full_results_{timestamp}.json"
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Resultados completos guardados en: {full_results_file}")
    
    # Guardar artículos por base de datos
    for db_name, db_data in results.get('databases', {}).items():
        db_file = output_dir / f"uniquindio_{db_name.lower().replace(' ', '_')}_{timestamp}.json"
        
        db_export = {
            "database": db_name,
            "query": results['query'],
            "timestamp": results['timestamp'],
            "count": db_data['count'],
            "articles": db_data['articles']
        }
        
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(db_export, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  📄 {db_name}: {db_file.name}")
    
    # Crear resumen
    summary = {
        "execution_date": datetime.now().isoformat(),
        "query": results['query'],
        "portal": results['portal'],
        "status": results['status'],
        "statistics": {
            "total_articles": results['total_records'],
            "databases_processed": len(results.get('databases', {})),
            "databases": {
                db_name: db_data['count']
                for db_name, db_data in results.get('databases', {}).items()
            }
        }
    }
    
    summary_file = output_dir / f"uniquindio_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Resumen guardado en: {summary_file}")
    
    return full_results_file


def print_summary(results: dict):
    """Imprimir resumen de resultados"""
    print("\n" + "="*80)
    print("📊 RESUMEN DE EXTRACCIÓN")
    print("="*80)
    print(f"🔍 Query: {results['query']}")
    print(f"🌐 Portal: {results['portal']}")
    print(f"📅 Fecha: {results['timestamp']}")
    print(f"📈 Total de artículos: {results['total_records']}")
    print(f"🎯 Estado: {results['status']}")
    print("\n📚 Artículos por base de datos:")
    
    for db_name, db_data in results.get('databases', {}).items():
        print(f"  • {db_name}: {db_data['count']} artículos")
    
    print("="*80)


def main():
    """Función principal"""
    import argparse
    import os
    
    # Cambiar directorio de trabajo a bibliometric-analysis
    os.chdir(Path(__file__).parent / "bibliometric-analysis")
    
    parser = argparse.ArgumentParser(
        description="Scraper del portal institucional Uniquindío"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="generative artificial intelligence",
        help="Término de búsqueda (default: 'generative artificial intelligence')"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Número máximo de resultados por base de datos (default: 50)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecutar navegador en modo headless (sin interfaz gráfica)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directorio de salida (default: 'data/raw')"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    
    print("\n" + "="*80)
    print("🎓 PORTAL INSTITUCIONAL - UNIVERSIDAD DEL QUINDÍO")
    print("="*80)
    print(f"🔍 Query: {args.query}")
    print(f"📊 Resultados máximos por BD: {args.max_results}")
    print(f"🌐 Portal: https://library.uniquindio.edu.co/databases")
    print("="*80 + "\n")
    
    # Cargar configuración
    logger.info("Cargando configuración...")
    config = get_config()
    
    # Inicializar scraper
    logger.info("Inicializando scraper del portal institucional...")
    scraper = UniquindioPortalScraper(config, headless=args.headless)
    
    try:
        # Ejecutar scraping
        logger.info("🚀 Iniciando extracción de datos...")
        results = scraper.scrape(
            query=args.query,
            max_results=args.max_results
        )
        
        # Verificar si hay resultados
        if results['total_records'] == 0:
            logger.warning("⚠️ No se obtuvieron resultados")
            print("\n⚠️ No se encontraron artículos.")
            print("\n💡 Posibles razones:")
            print("  1. No estás conectado a la red institucional de Uniquindío")
            print("  2. Necesitas configurar VPN institucional")
            print("  3. Las bases de datos requieren autenticación manual")
            print("\n📌 Recomendaciones:")
            print("  • Conéctate a la red de la universidad")
            print("  • O configura la VPN institucional")
            print("  • Ejecuta sin --headless para autenticarte manualmente")
            return
        
        # Guardar resultados
        output_dir = Path(args.output_dir)
        results_file = save_results(results, output_dir)
        
        # Mostrar resumen
        print_summary(results)
        
        print(f"\n✅ Extracción completada exitosamente")
        print(f"📁 Resultados guardados en: {output_dir}")
        print(f"📄 Archivo principal: {results_file.name}")
        
        # Siguiente paso
        print("\n" + "="*80)
        print("📌 PRÓXIMO PASO: UNIFICACIÓN DE DATOS")
        print("="*80)
        print("\nPara unificar los datos descargados, ejecuta:")
        print("  python automation_pipeline.py")
        print("\nO usa el menú interactivo:")
        print("  python menu_interactivo.py")
        print("="*80 + "\n")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Ejecución interrumpida por el usuario")
        print("\n\n⚠️ Proceso cancelado por el usuario")
    
    except Exception as e:
        logger.error(f"❌ Error en la ejecución: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Sugerencias:")
        print("  • Verifica tu conexión a la red institucional")
        print("  • Asegúrate de tener ChromeDriver instalado")
        print("  • Revisa los logs en la carpeta 'logs/'")
    
    finally:
        logger.info("Finalizando scraper...")


if __name__ == "__main__":
    main()
