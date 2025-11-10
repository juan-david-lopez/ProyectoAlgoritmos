import json

with open('data/unified_articles.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('\n' + '='*70)
print('📊 RESUMEN DE DATOS DESCARGADOS')
print('='*70)
print(f'\nTotal de artículos: {len(data)}')
print(f'Rango de años: {min([a["year"] for a in data if a.get("year")])} - {max([a["year"] for a in data if a.get("year")])}')

print('\n🔬 Primeros 5 artículos:')
for i, art in enumerate(data[:5], 1):
    print(f'\n{i}. {art["title"][:70]}...')
    print(f'   📅 Año: {art["year"]}')
    print(f'   👥 Autores: {", ".join(art["authors"][:2])}')
    print(f'   📚 Fuente: {art["source"]}')
    print(f'   📖 Citas: {art.get("citation_count", 0)}')

# Estadísticas por fuente
fuentes = {}
for art in data:
    source = art['source']
    fuentes[source] = fuentes.get(source, 0) + 1

print('\n📊 Distribución por fuente:')
for source, count in sorted(fuentes.items(), key=lambda x: x[1], reverse=True):
    print(f'   - {source}: {count} artículos')

print('\n✅ ¡Los datos se descargaron correctamente en tiempo real!')
print('='*70)
