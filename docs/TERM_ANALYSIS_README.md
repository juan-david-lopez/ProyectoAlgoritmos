# Sistema de Análisis de Frecuencia de Términos

**Módulo:** `src/preprocessing/term_analysis/`
**Fecha:** 2025-10-27

---

## 📋 Descripción

Sistema completo para analizar frecuencia de términos predefinidos en abstracts científicos, con foco en **Concepts of Generative AI in Education**.

### Características Principales

✅ **Búsqueda flexible** con variantes (singular/plural, guiones, espacios)
✅ **Análisis de co-ocurrencia** entre términos
✅ **Estadísticas descriptivas** completas
✅ **3 tipos de visualizaciones** automáticas
✅ **Reportes detallados** en Markdown

---

## 🎯 Términos Predefinidos

El sistema analiza estos 15 términos relacionados con IA Generativa en Educación:

1. Generative models
2. Prompting
3. Machine learning
4. Multimodality
5. Fine-tuning
6. Training data
7. Algorithmic bias
8. Explainability
9. Transparency
10. Ethics
11. Privacy
12. Personalization
13. Human-AI interaction
14. AI literacy
15. Co-creation

---

## 🔧 Componentes Implementados

### 1. PredefinedTermsAnalyzer

Clase principal que implementa todo el análisis.

**Ubicación:** `src/preprocessing/term_analysis/predefined_terms_analyzer.py`

#### Métodos Principales

##### `__init__(unified_data_path)`
Inicializa el analizador cargando abstracts desde JSON.

```python
from src.preprocessing.term_analysis import PredefinedTermsAnalyzer

analyzer = PredefinedTermsAnalyzer('data/unified_articles.json')
```

##### `preprocess_text(text)`
Preprocesamiento suave que preserva términos compuestos:
- Lowercase para búsqueda case-insensitive
- Normalización de espacios
- Mantiene guiones y caracteres especiales

**Ejemplo:**
```python
text = "Machine   Learning and Fine-Tuning"
processed = analyzer.preprocess_text(text)
# Output: "machine learning and fine-tuning"
```

##### `find_term_variants(term)`
Genera variantes del término para búsqueda flexible.

**Estrategias:**
1. **Singular/Plural:** "models" → ["model", "models"]
2. **Guiones:** "Fine-tuning" → ["fine-tuning", "fine tuning", "finetuning"]
3. **Formas verbales:** "Fine-tuning" → ["finetune", "finetuned"]

**Ejemplo:**
```python
variants = analyzer.find_term_variants("Fine-tuning")
# Output: [
#     "fine-tuning",
#     "fine tuning",
#     "finetuning",
#     "finetune",
#     "finetuned"
# ]
```

##### `calculate_frequencies(abstracts)`
Calcula frecuencias de todos los términos predefinidos.

**Retorna:**
```python
{
    'Generative models': {
        'total_count': 45,           # Total de ocurrencias
        'documents_count': 23,       # Documentos que lo contienen
        'avg_per_document': 1.96,    # Promedio por documento
        'document_frequency': 0.23,  # % de documentos
        'variants_found': {
            'generative model': 30,
            'generative models': 15
        }
    },
    ...
}
```

**Ejemplo:**
```python
frequencies = analyzer.calculate_frequencies()

# Acceder a datos de un término
ml_stats = frequencies['Machine learning']
print(f"Ocurrencias: {ml_stats['total_count']}")
print(f"En {ml_stats['documents_count']} documentos")
```

##### `calculate_cooccurrence_matrix(abstracts)`
Calcula matriz de co-ocurrencia entre términos.

**Retorna:** DataFrame (términos × términos) con conteos.

**Ejemplo:**
```python
cooccurrence = analyzer.calculate_cooccurrence_matrix()

# Cuántas veces "Machine learning" y "Ethics" aparecen juntos
count = cooccurrence.loc['Machine learning', 'Ethics']
print(f"Co-ocurrencia: {count} documentos")
```

##### `generate_statistics_report(frequencies)`
Genera DataFrame con estadísticas descriptivas.

**Columnas:**
- Rank
- Term
- Total Count
- Documents
- Avg per Doc
- Doc Frequency (%)
- Variants Used

**Ejemplo:**
```python
stats_df = analyzer.generate_statistics_report(frequencies)
print(stats_df.head())
```

##### `visualize_frequencies(frequencies, output_dir)`
Genera 3 visualizaciones:

1. **Gráfico de barras horizontal**
   - Frecuencia total por término
   - Colores según magnitud
   - Valores anotados

2. **Heatmap de co-ocurrencia**
   - Matriz simétrica
   - Términos que aparecen juntos
   - Anotaciones con conteos

3. **Distribución estadística**
   - Histograma de frecuencias
   - Box plot de document frequency
   - Estadísticas descriptivas

**Ejemplo:**
```python
analyzer.visualize_frequencies(
    frequencies,
    'output/term_analysis'
)
# Genera:
#   - term_frequencies_bar.png
#   - term_cooccurrence_heatmap.png
#   - term_distribution_stats.png
```

##### `generate_detailed_report(frequencies, output_path)`
Genera reporte Markdown completo.

**Contenido:**
- Resumen ejecutivo
- Tabla de estadísticas
- Detalles de variantes por término
- Insights automáticos

**Ejemplo:**
```python
analyzer.generate_detailed_report(
    frequencies,
    'output/term_analysis/report.md'
)
```

---

## 🚀 Uso Rápido

### Instalación

```bash
pip install numpy pandas matplotlib seaborn scipy tabulate
```

### Demo Completo

```bash
python examples/term_analysis_demo.py
```

### Uso Programático

```python
from src.preprocessing.term_analysis import PredefinedTermsAnalyzer

# 1. Inicializar
analyzer = PredefinedTermsAnalyzer('data/unified_articles.json')

# 2. Calcular frecuencias
frequencies = analyzer.calculate_frequencies()

# 3. Generar estadísticas
stats_df = analyzer.generate_statistics_report(frequencies)
print(stats_df)

# 4. Visualizaciones
analyzer.visualize_frequencies(frequencies, 'output/term_analysis')

# 5. Reporte detallado
analyzer.generate_detailed_report(frequencies, 'output/report.md')
```

---

## 📊 Ejemplo de Resultados

### Estadísticas (con datos de ejemplo)

```
Rank  Term                    Total Count  Documents  Avg per Doc  Doc Frequency (%)
----  ----------------------  -----------  ---------  -----------  -----------------
   1  Machine learning               42         35         1.20              70.0
   2  Ethics                         28         22         0.80              44.0
   3  Generative models              25         18         0.71              36.0
   4  Privacy                        20         15         0.57              30.0
   5  Transparency                   18         14         0.51              28.0
   6  Fine-tuning                    15         12         0.43              24.0
   7  Training data                  12         10         0.34              20.0
   8  Personalization                10          8         0.29              16.0
   9  Explainability                  8          7         0.23              14.0
  10  AI literacy                     6          5         0.17              10.0
  11  Human-AI interaction            5          4         0.14               8.0
  12  Multimodality                   4          3         0.11               6.0
  13  Prompting                       3          3         0.09               6.0
  14  Algorithmic bias                2          2         0.06               4.0
  15  Co-creation                     1          1         0.03               2.0
```

### Variantes Detectadas

**Machine learning:**
- `machine learning`: 30 ocurrencias
- `ml`: 8 ocurrencias
- `machine learned`: 4 ocurrencias

**Fine-tuning:**
- `fine-tuning`: 8 ocurrencias
- `finetuning`: 4 ocurrencias
- `fine tuning`: 3 ocurrencias

### Co-ocurrencia (Top pares)

1. `Machine learning` + `Ethics`: 15 documentos
2. `Privacy` + `Ethics`: 12 documentos
3. `Machine learning` + `Generative models`: 10 documentos
4. `Transparency` + `Explainability`: 8 documentos
5. `Training data` + `Machine learning`: 7 documentos

---

## 🔍 Detalles de Implementación

### Búsqueda con Word Boundaries

Para evitar matches parciales, se usa regex con word boundaries:

```python
# Evita matchear "model" en "remodel"
pattern = r'\b' + re.escape(variant) + r'\b'
matches = re.findall(pattern, abstract)
```

### Generación de Variantes

Algoritmo inteligente que genera variantes relevantes:

1. **Guiones:** Sustituir por espacio o eliminar
2. **Singular/Plural:** Reglas heurísticas
3. **Términos compuestos:** Variantes de cada palabra
4. **Formas verbales:** Para términos como "tuning"

### Caché de Variantes

Para optimizar, las variantes se calculan una vez y se cachean:

```python
self._variant_cache = {}

def find_term_variants(self, term):
    if term in self._variant_cache:
        return self._variant_cache[term]
    # ... calcular ...
    self._variant_cache[term] = variants
    return variants
```

---

## 📈 Visualizaciones Generadas

### 1. Gráfico de Barras

![Term Frequencies Bar](../output/term_analysis/term_frequencies_bar.png)

**Características:**
- Barras horizontales ordenadas por frecuencia
- Colores graduados (viridis colormap)
- Valores anotados al final de cada barra
- Grid para mejor lectura

### 2. Heatmap de Co-ocurrencia

![Co-occurrence Heatmap](../output/term_analysis/term_cooccurrence_heatmap.png)

**Características:**
- Matriz simétrica
- Valores anotados en cada celda
- Escala de colores (YlOrRd)
- Diagonal muestra auto-ocurrencia

### 3. Distribución Estadística

![Distribution Stats](../output/term_analysis/term_distribution_stats.png)

**Características:**
- Subplot 1: Histograma con media/mediana
- Subplot 2: Box plot con estadísticas
- Anotaciones informativas

---

## 🧪 Testing

### Casos de Prueba

El sistema maneja correctamente:

✅ **Textos vacíos**
```python
analyzer.calculate_frequencies([])  # Retorna diccionario vacío
```

✅ **Términos no encontrados**
```python
# Término con 0 ocurrencias tiene total_count=0
```

✅ **Variantes múltiples**
```python
# Detecta todas las formas del término
```

✅ **Case insensitivity**
```python
# "Machine Learning" == "machine learning"
```

### Ejecutar Demo

```bash
python examples/term_analysis_demo.py
```

**Salida esperada:**
- Log detallado del proceso
- 3 visualizaciones en PNG
- Reporte Markdown
- Estadísticas en consola

---

## 💡 Extensibilidad

### Agregar Nuevos Términos

Modificar la lista en la clase:

```python
class PredefinedTermsAnalyzer:
    PREDEFINED_TERMS = [
        "Generative models",
        ...
        "Tu nuevo término"  # Agregar aquí
    ]
```

### Personalizar Variantes

Sobrescribir `find_term_variants()`:

```python
class CustomAnalyzer(PredefinedTermsAnalyzer):
    def find_term_variants(self, term):
        variants = super().find_term_variants(term)
        # Agregar lógica personalizada
        variants.extend(['variant_custom1', 'variant_custom2'])
        return variants
```

### Filtrar Abstracts

Analizar solo un subconjunto:

```python
# Filtrar por año
filtered_abstracts = [
    art['abstract']
    for art in analyzer.unified_data
    if art.get('year') == 2023
]

frequencies = analyzer.calculate_frequencies(filtered_abstracts)
```

---

## 📝 Outputs Generados

### Archivos de Visualización

```
output/term_analysis/
├── term_frequencies_bar.png        # 12x10 inches, 300 DPI
├── term_cooccurrence_heatmap.png   # 14x12 inches, 300 DPI
└── term_distribution_stats.png     # 14x6 inches, 300 DPI
```

### Reporte Markdown

```
output/term_analysis/
└── predefined_terms_report.md      # Reporte completo
```

**Contenido del reporte:**
- Resumen ejecutivo
- Tabla de estadísticas
- Variantes detectadas por término
- Insights automáticos

### Logs

```
term_analysis_demo.log              # Log de ejecución
```

---

## ⚡ Optimizaciones Implementadas

### 1. Preprocesamiento único
```python
# Preprocesar todos los abstracts una vez
preprocessed = [preprocess_text(abs) for abs in abstracts]

# Usar versiones preprocesadas en loops
for abstract in preprocessed:
    # ...
```

### 2. Caché de variantes
```python
# Evitar recalcular variantes
self._variant_cache[term] = variants
```

### 3. Regex compilado implícitamente
```python
# re.findall() cachea patterns internamente
pattern = r'\b' + re.escape(variant) + r'\b'
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tabulate'"

**Solución:**
```bash
pip install tabulate
```

### Error: "FileNotFoundError: data/unified_articles.json"

**Solución:**
- Verificar que el archivo existe
- Ajustar la ruta si es necesario

### Visualizaciones no se generan

**Causa:** Directorio de salida no existe

**Solución:**
```python
Path(output_dir).mkdir(parents=True, exist_ok=True)
```

### Términos no detectados

**Causa:** Variantes no incluidas

**Solución:**
- Verificar con `find_term_variants()` qué variantes se buscan
- Agregar variantes manualmente si es necesario

---

## 📚 Referencias

### Papers Relacionados

- Salton, G., & McGill, M. J. (1983). *Introduction to modern information retrieval.* McGraw-Hill.
- Manning, C. D., et al. (2008). *Introduction to Information Retrieval.* Cambridge University Press.

### Herramientas Utilizadas

- **NumPy:** Operaciones numéricas
- **Pandas:** DataFrames y estadísticas
- **Matplotlib:** Visualizaciones base
- **Seaborn:** Visualizaciones avanzadas (heatmaps)
- **Scipy:** Estadísticas adicionales
- **Tabulate:** Formateo de tablas Markdown

---

## 🎓 Conclusión

Este sistema proporciona un análisis exhaustivo de términos predefinidos con:
- ✅ Búsqueda flexible y robusta
- ✅ Estadísticas descriptivas completas
- ✅ Visualizaciones profesionales
- ✅ Reportes automáticos
- ✅ Código documentado y extensible

**Estado:** ✅ **COMPLETO Y FUNCIONAL**

---

**Última actualización:** 2025-10-27
