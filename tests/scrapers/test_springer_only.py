"""
Script de prueba para Springer únicamente
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from scraper_uniquindio_completo import scrape_springer_uniquindio, save_results
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

email = os.getenv("EMAIL")
password = os.getenv("PASSWORD")

print("\n" + "="*80)
print("🧪 TEST: SPRINGER ÚNICAMENTE")
print("="*80)
print(f"\n📧 Email: {email}")
print(f"🔑 Password: {'*' * len(password) if password else 'No configurado'}")
print("\n" + "="*80 + "\n")

# Extraer de Springer
articles = scrape_springer_uniquindio(
    query="generative artificial intelligence",
    max_results=10,  # Solo 10 para prueba rápida
    email=email,
    password=password
)

if articles:
    print(f"\n✅ Extracción exitosa: {len(articles)} artículos")
    save_results(articles, "Springer_TEST")
    print(f"\n📄 Archivo guardado en: data/raw/uniquindio/")
else:
    print("\n❌ No se extrajeron artículos")
