"""
Script para unificar los artículos extraídos de las 3 bases de datos
"""
import json
import os
from datetime import datetime

# Archivos más recientes de cada base de datos
archivos = [
    "data/raw/uniquindio/uniquindio_acm_digital_library_20251030_131341.json",
    "data/raw/uniquindio/uniquindio_sciencedirect_20251030_131543.json",
    "data/raw/uniquindio/uniquindio_springer_20251030_131705.json"
]

print("\n" + "="*80)
print("📚 UNIFICACIÓN DE ARTÍCULOS")
print("="*80 + "\n")

todos_articulos = []
estadisticas = {}

for archivo in archivos:
    if os.path.exists(archivo):
        print(f"📖 Leyendo: {os.path.basename(archivo)}")
        
        with open(archivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        articulos = data.get('articles', [])
        database = data.get('database', 'Unknown')
        
        print(f"   • Base de datos: {database}")
        print(f"   • Artículos: {len(articulos)}")
        
        estadisticas[database] = len(articulos)
        todos_articulos.extend(articulos)
        print()
    else:
        print(f"⚠️  Archivo no encontrado: {archivo}\n")

# Eliminar duplicados por título
print("🔍 Eliminando duplicados...")
articulos_unicos = []
titulos_vistos = set()

for articulo in todos_articulos:
    titulo = articulo.get('title', '').strip().lower()
    if titulo and titulo not in titulos_vistos:
        titulos_vistos.add(titulo)
        articulos_unicos.append(articulo)

duplicados = len(todos_articulos) - len(articulos_unicos)
print(f"   • Total artículos: {len(todos_articulos)}")
print(f"   • Duplicados eliminados: {duplicados}")
print(f"   • Artículos únicos: {len(articulos_unicos)}\n")

# Crear archivo unificado
output_file = "data/unified_articles.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

resultado = {
    "metadata": {
        "unified_date": datetime.now().isoformat(),
        "query": "generative artificial intelligence",
        "total_articles": len(articulos_unicos),
        "duplicates_removed": duplicados,
        "databases": estadisticas
    },
    "articles": articulos_unicos
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

print("="*80)
print("✅ UNIFICACIÓN COMPLETADA")
print("="*80 + "\n")

print("📊 Resumen por base de datos:")
for db, count in estadisticas.items():
    print(f"   • {db}: {count} artículos")

print(f"\n📁 Archivo unificado guardado en: {output_file}")
print(f"📚 Total de artículos únicos: {len(articulos_unicos)}")
print("\n" + "="*80 + "\n")
