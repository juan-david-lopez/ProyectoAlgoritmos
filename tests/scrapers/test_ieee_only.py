"""
Script de prueba para IEEE únicamente
"""
import sys
import os
from dotenv import load_dotenv

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Cargar variables de entorno
load_dotenv()

# Importar la función de IEEE del scraper principal
from scraper_uniquindio_completo import scrape_ieee_uniquindio, save_results

def main():
    """Probar solo IEEE"""
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    
    print("🧪 PRUEBA RÁPIDA - Solo IEEE Xplore\n")
    
    query = "generative artificial intelligence"
    max_results = 50
    
    ieee_articles = scrape_ieee_uniquindio(query, max_results, email, password)
    
    if ieee_articles:
        print(f"\n✅ Éxito: {len(ieee_articles)} artículos extraídos")
        save_results(ieee_articles, "IEEE Xplore")
    else:
        print("\n❌ No se extrajeron artículos de IEEE")

if __name__ == "__main__":
    main()
