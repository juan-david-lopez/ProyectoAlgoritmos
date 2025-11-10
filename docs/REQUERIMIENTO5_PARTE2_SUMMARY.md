# REQUERIMIENTO 5 - PARTE 2: Dynamic Word Cloud Visualization

**Sistema Completo de Visualización Interactiva de Producción Científica**

## Resumen Ejecutivo

Se ha implementado exitosamente un sistema completo y profesional de generación de nubes de palabras dinámicas para analizar términos y su evolución en publicaciones científicas. El sistema incluye extracción inteligente de términos con NLP, múltiples métodos de ponderación, visualizaciones estáticas e interactivas, comparaciones, actualizaciones incrementales y análisis de evolución temporal.

---

## Archivos Implementados

### 1. Módulo Principal
**Ubicación**: `src/visualization/dynamic_wordcloud.py` (1,000+ líneas)

**Componentes principales**:

#### Clase `DynamicWordCloud`
Clase completa con todas las funcionalidades requeridas:

```python
class DynamicWordCloud:
    def __init__(self, unified_data_path: str)
    def extract_and_process_terms(self, sources: list, ngram_range: tuple) -> dict
    def calculate_term_weights(self, term_frequencies: dict, method: str) -> dict
    def generate_wordcloud(self, term_weights: dict, output_path: str, style: str)
    def generate_interactive_wordcloud(self, term_weights: dict, output_html: str)
    def create_comparative_wordclouds(self, output_dir: str)
    def update_wordcloud_incremental(self, new_data_path: str, ...) -> dict
    def generate_wordcloud_evolution(self, output_dir: str, create_animation: bool)
    def save_term_weights(self, output_path: str, term_weights: dict)
    def load_term_weights(self, input_path: str) -> dict
```

### 2. Script de Demostración
**Ubicación**: `examples/dynamic_wordcloud_demo.py` (550+ líneas)

**Ejemplos incluidos**:
1. Uso básico y extracción de términos
2. Diferentes estilos visuales (scientific, colorful, academic, tech)
3. Comparación de métodos de ponderación (frequency, log, normalized, TF-IDF)
4. Generación de word clouds interactivos
5. Visualizaciones comparativas
6. Actualización incremental (característica dinámica)
7. Análisis de evolución temporal con GIF
8. Flujo de trabajo completo

Incluye generación de datos de muestra para pruebas.

### 3. Documentación
**Ubicación**: `docs/DYNAMIC_WORDCLOUD_GUIDE.md` (700+ líneas)

**Contenido**:
- Guía completa de uso
- Referencia de API
- Ejemplos de código
- Troubleshooting
- Temas avanzados
- Mejores prácticas

### 4. Dependencias Actualizadas
**Ubicación**: `requirements.txt`

**Nueva dependencia agregada**:
```
Pillow>=9.0.0  # Procesamiento de imágenes y creación de GIF
```

Dependencias ya existentes utilizadas:
- wordcloud>=1.9.0
- spacy>=3.5.0
- nltk>=3.8.0
- plotly>=5.14.0
- matplotlib>=3.4.0

### 5. Actualización del Módulo
**Ubicación**: `src/visualization/__init__.py`

Exporta la clase `DynamicWordCloud` para fácil importación.

---

## Funcionalidades Implementadas

### 1. Extracción y Procesamiento de Términos (`extract_and_process_terms`)

**Procesamiento avanzado con NLP**:

#### a) Fuentes múltiples
- Abstracts
- Keywords
- Títulos
- Cualquier campo de texto

#### b) Pipeline de procesamiento
1. **Limpieza de texto**:
   - Eliminar URLs
   - Eliminar direcciones de email
   - Eliminar caracteres especiales
   - Normalizar espacios en blanco

2. **Tokenización**:
   - Usar spaCy para tokenización inteligente
   - Fallback a tokenización simple si spaCy no disponible

3. **Normalización**:
   - Convertir a minúsculas
   - Lematización (lemmatization)

4. **Filtrado por POS tags**:
   - Mantener NOUN (sustantivos)
   - Mantener PROPN (nombres propios)
   - Mantener ADJ (adjetivos)
   - Descartar verbos, adverbios, etc.

5. **Eliminación de stopwords**:
   - Stopwords estándar de NLTK (inglés)
   - Stopwords específicas del dominio (40+ términos):
     - `study`, `paper`, `research`, `analysis`, `method`
     - `result`, `conclusion`, `introduction`, `abstract`
     - `journal`, `conference`, `proceedings`, `ieee`, `acm`
     - `et`, `al`, `fig`, `table`, `section`
     - Y más...

6. **Extracción de n-gramas**:
   - Unigramas (1 palabra)
   - Bigramas (2 palabras)
   - Trigramas (3 palabras)
   - Validación de cada token en el n-grama

#### c) Caché de resultados
Los términos extraídos se almacenan en caché para reutilización.

**Salida**:
```python
{
    'machine learning': 45,
    'neural network': 38,
    'deep learning': 32,
    'computer vision': 28,
    'natural language processing': 25,
    ...
}
```

### 2. Cálculo de Pesos de Términos (`calculate_term_weights`)

**Métodos implementados**:

#### a) Frecuencia Simple (`frequency`)
```python
weight = count
```
- Peso = conteo directo
- Útil para ver términos más comunes

#### b) Frecuencia Logarítmica (`log_frequency`)
```python
weight = log(count + 1)
```
- Reduce dominancia de términos muy frecuentes
- Mejora visualización balanceada
- **Método recomendado** para la mayoría de casos

#### c) Normalización Min-Max (`normalized`)
```python
weight = (count - min) / (max - min)
```
- Escala todos los pesos a [0, 1]
- Útil para comparaciones

#### d) TF-IDF (`tfidf`)
```python
weight = tf * log(N / df)
```
Donde:
- `tf` = frecuencia del término
- `N` = número total de documentos
- `df` = número de documentos que contienen el término

**Ventajas de TF-IDF**:
- Enfatiza términos frecuentes pero no ubicuos
- Identifica términos distintivos
- Excelente para análisis académico

**Salida**:
```python
{
    'machine learning': 4.523,
    'neural network': 4.187,
    'deep learning': 3.891,
    ...
}
```

### 3. Generación de Word Cloud Estático (`generate_wordcloud`)

**Características**:

#### a) Múltiples estilos visuales

**Scientific (predeterminado)**:
- Fondo: blanco
- Paleta: azules (Blues colormap)
- Apariencia: profesional
- Uso: publicaciones académicas

**Colorful**:
- Fondo: blanco
- Paleta: arcoíris (rainbow colormap)
- Apariencia: vibrante
- Uso: presentaciones

**Academic**:
- Fondo: beige (#f5f5dc)
- Paleta: marrón/sepia (YlOrBr colormap)
- Apariencia: vintage, clásico
- Uso: contextos académicos tradicionales

**Tech**:
- Fondo: negro
- Paleta: plasma (plasma colormap)
- Apariencia: futurista, moderno
- Uso: presentaciones tecnológicas

#### b) Configuración profesional
- **Tamaño**: Personalizable (default: 1600x1000 px)
- **Resolución**: 300 DPI (calidad de impresión)
- **Max words**: 150 (personalizable)
- **Font size**: 10pt (mín) - 100pt (máx)
- **Layout**: Compacto pero legible
- **Relative scaling**: 0.5 (balance entre tamaños)
- **Horizontal preference**: 70% (mayoría de palabras horizontales)
- **Collocations**: False (evita repeticiones)

### 4. Word Cloud Interactivo (`generate_interactive_wordcloud`)

**Tecnología**: Plotly

**Características interactivas**:

#### a) Visualización
- Scatter plot con texto como marcadores
- Posiciones aleatorias (algoritmo mejorable)
- Tamaño de fuente proporcional al peso
- Color basado en peso (escala Blues)

#### b) Interactividad
- **Hover**: Muestra término y peso exacto
- **Zoom**: Acercar/alejar
- **Pan**: Mover vista
- **Export**: Guardar como PNG/SVG

#### c) Configuración
- Ancho: 1200px
- Alto: 800px
- Escala de colores con barra lateral
- Fondo blanco
- Sin ejes visibles (estética limpia)

### 5. Word Clouds Comparativos (`create_comparative_wordclouds`)

**Generación automática de múltiples visualizaciones**:

#### a) Por fuente
1. **Abstracts only** (`wordcloud_abstracts.png`):
   - Solo términos de abstracts
   - Más específico y técnico

2. **Keywords only** (`wordcloud_keywords.png`):
   - Solo términos de keywords
   - Más conciso, enfocado

3. **Combined** (`wordcloud_combined.png`):
   - Abstracts + Keywords
   - Vista comprehensiva

#### b) Por año (si datos disponibles)
- Un word cloud por cada año: `wordcloud_year_2021.png`, etc.
- Solo para años con ≥3 documentos
- Permite ver evolución temporal

#### c) Grid de comparación
- **Archivo**: `wordcloud_comparison_grid.png`
- Layout: Grid de 3 columnas
- Hasta 6 word clouds lado a lado
- Títulos descriptivos
- Facilita comparación visual directa

### 6. Actualización Incremental (`update_wordcloud_incremental`)

**Característica DINÁMICA clave**:

#### Proceso:
1. **Cargar pesos previos**:
   - Desde archivo pickle
   - Términos y sus pesos acumulados

2. **Extraer términos de nuevos documentos**:
   - Procesar nuevo CSV
   - Aplicar mismo pipeline NLP

3. **Combinar pesos**:
   ```python
   combined_weight[term] = previous_weight[term] + new_weight[term]
   ```

4. **Normalizar**:
   - Escalar pesos combinados
   - Mantener distribución balanceada

5. **Regenerar word cloud**:
   - Con pesos actualizados
   - Mismo estilo visual

6. **Guardar pesos actualizados**:
   - Para futuras actualizaciones
   - Formato pickle para eficiencia

**Uso**:
```python
updated_weights = wc.update_wordcloud_incremental(
    new_data_path='data/new_publications.csv',
    previous_weights_path='weights_previous.pkl',
    output_path='wordcloud_updated.png'
)
```

### 7. Evolución Temporal (`generate_wordcloud_evolution`)

**Análisis longitudinal comprehensivo**:

#### a) Word clouds por año
- **Archivos**: `evolution_2021.png`, `evolution_2022.png`, etc.
- Un word cloud por cada año con datos
- Misma escala visual para comparación
- DPI reducido (150) para animación

#### b) Animación GIF
- **Archivo**: `wordcloud_evolution.gif`
- Secuencia animada mostrando cambios
- 1 segundo por frame
- Loop infinito
- Visualiza tendencias temporales

#### c) Análisis de tendencias (`term_trends.json`)

**Términos emergentes**:
- Términos con alto peso en años recientes
- Bajo/ausente en años iniciales
- Crecimiento = peso_final - peso_inicial

**Términos en declive**:
- Términos con alto peso en años iniciales
- Bajo/ausente en años recientes
- Declive = peso_inicial - peso_final

**Formato de salida**:
```json
{
  "period": "2021-2023",
  "emerging_terms": [
    {"term": "transformer", "growth": 15.3},
    {"term": "large language model", "growth": 12.8},
    {"term": "attention mechanism", "growth": 10.5}
  ],
  "declining_terms": [
    {"term": "support vector machine", "decline": 8.2},
    {"term": "decision tree", "decline": 6.5}
  ]
}
```

### 8. Persistencia de Datos

#### save_term_weights()
```python
wc.save_term_weights('output/weights.pkl', weights)
```
- Formato: Pickle (binario)
- Eficiente para grandes diccionarios
- Rápida carga/guardado

#### load_term_weights()
```python
weights = wc.load_term_weights('output/weights.pkl')
```
- Restaura pesos previos
- Para actualizaciones incrementales
- Para reutilización

---

## Uso del Sistema

### Instalación

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar modelos NLP
python -m spacy download en_core_web_sm

# 3. (Opcional) Descargar stopwords NLTK
python -c "import nltk; nltk.download('stopwords')"
```

### Uso Básico

```python
from src.visualization import DynamicWordCloud

# Inicializar
wc = DynamicWordCloud('data/processed/unified_data.csv')

# Extraer términos
terms = wc.extract_and_process_terms(
    sources=['abstract', 'keywords'],
    ngram_range=(1, 3),
    max_terms=200
)

# Calcular pesos
weights = wc.calculate_term_weights(terms, method='tfidf')

# Generar word cloud
wc.generate_wordcloud(
    weights,
    output_path='output/wordcloud.png',
    style='scientific',
    dpi=300
)
```

### Flujo Completo con Todas las Características

```python
from pathlib import Path
from src.visualization import DynamicWordCloud

# Configurar
data_path = 'data/processed/unified_data.csv'
output_dir = Path('output/wordcloud_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Inicializar
wc = DynamicWordCloud(data_path)

# 1. Extraer y procesar términos
terms = wc.extract_and_process_terms(
    sources=['abstract', 'keywords'],
    ngram_range=(1, 3),
    max_terms=200
)

# 2. Calcular pesos con TF-IDF
weights = wc.calculate_term_weights(terms, method='tfidf')

# 3. Generar word clouds estáticos (múltiples estilos)
for style in ['scientific', 'colorful', 'academic']:
    wc.generate_wordcloud(
        weights,
        output_path=str(output_dir / f'wordcloud_{style}.png'),
        style=style,
        dpi=300
    )

# 4. Generar word cloud interactivo
wc.generate_interactive_wordcloud(
    weights,
    output_html=str(output_dir / 'wordcloud_interactive.html')
)

# 5. Crear visualizaciones comparativas
wc.create_comparative_wordclouds(
    output_dir=str(output_dir / 'comparative'),
    style='scientific'
)

# 6. Analizar evolución temporal
wc.generate_wordcloud_evolution(
    output_dir=str(output_dir / 'evolution'),
    create_animation=True
)

# 7. Guardar pesos para futuras actualizaciones
wc.save_term_weights(str(output_dir / 'term_weights.pkl'), weights)

print(f"Análisis completo guardado en: {output_dir}")
```

### Actualización Dinámica

```python
# Paso 1: Generar word cloud inicial
wc = DynamicWordCloud('data/initial_data.csv')
terms = wc.extract_and_process_terms()
weights = wc.calculate_term_weights(terms)
wc.generate_wordcloud(weights, 'wordcloud_initial.png')
wc.save_term_weights('weights_current.pkl', weights)

# Paso 2: Cuando llegan nuevas publicaciones
updated_weights = wc.update_wordcloud_incremental(
    new_data_path='data/new_publications_2024.csv',
    previous_weights_path='weights_current.pkl',
    output_path='wordcloud_updated.png',
    style='scientific'
)

# Paso 3: Guardar pesos actualizados para próxima vez
wc.save_term_weights('weights_current.pkl', updated_weights)
```

### Ejecutar Demo Completo

```bash
# Ejecutar script de demostración
python examples/dynamic_wordcloud_demo.py
```

Esto creará:
- Datos de muestra
- Word clouds con diferentes estilos
- Word clouds con diferentes métodos de ponderación
- Word cloud interactivo
- Visualizaciones comparativas
- Actualización incremental (demo)
- Evolución temporal con GIF
- Análisis de tendencias

---

## Ejemplos de Salida

### 1. Términos Extraídos

```
Top 10 términos (frecuencia):
  machine learning: 45
  neural network: 38
  deep learning: 32
  computer vision: 28
  natural language processing: 25
  data science: 22
  artificial intelligence: 20
  transformer: 18
  convolutional neural network: 15
  supervised learning: 14
```

### 2. Word Cloud Científico

- Fondo blanco limpio
- Términos en azules (oscuro → claro)
- Tamaño proporcional a peso TF-IDF
- 150 palabras máximo
- 300 DPI (alta calidad)
- "machine learning" más grande (mayor peso)

### 3. Análisis de Tendencias (JSON)

```json
{
  "period": "2021-2023",
  "emerging_terms": [
    {"term": "large language model", "growth": 18.5},
    {"term": "transformer architecture", "growth": 15.2},
    {"term": "generative AI", "growth": 14.8},
    {"term": "attention mechanism", "growth": 12.3},
    {"term": "GPT", "growth": 11.7}
  ],
  "declining_terms": [
    {"term": "support vector machine", "decline": 9.5},
    {"term": "decision tree", "decline": 7.2},
    {"term": "random forest", "decline": 6.8},
    {"term": "k-nearest neighbors", "decline": 5.5}
  ]
}
```

### 4. Grid Comparativo

Layout de 2x3 mostrando:
- Row 1: Abstracts only | Keywords only | Combined
- Row 2: Year 2021 | Year 2022 | Year 2023

Cada imagen con título descriptivo, facilita comparación directa.

---

## Arquitectura del Sistema

### Flujo de Datos

```
unified_data.csv
       ↓
[DynamicWordCloud.__init__]
   ├─ Load spaCy model
   ├─ Load NLTK stopwords
   └─ Add domain stopwords
       ↓
[extract_and_process_terms]
   ├─ Clean text
   ├─ Tokenize (spaCy)
   ├─ POS filtering
   ├─ Lemmatization
   ├─ Stopword removal
   └─ N-gram extraction
       ↓
    Term Frequencies Dict
       ↓
[calculate_term_weights]
   └─ Apply method (TF-IDF, log, etc.)
       ↓
    Term Weights Dict
       ↓
┌──────┴────────┬───────────┬──────────┬─────────┐
↓               ↓           ↓          ↓         ↓
Static WC    Interactive  Comparative Evolution  Save
(PNG)        (HTML)       (Grid)      (GIF)    (Pickle)
```

### Componentes Principales

```
DynamicWordCloud
├── NLP Components
│   ├── spaCy (en_core_web_sm)
│   ├── NLTK stopwords
│   └── Domain stopwords
├── Text Processing
│   ├── Cleaning
│   ├── Tokenization
│   ├── Lemmatization
│   ├── POS filtering
│   └── N-gram extraction
├── Weighting
│   ├── Frequency
│   ├── Log frequency
│   ├── Normalized
│   └── TF-IDF
├── Visualization
│   ├── Static (WordCloud lib)
│   ├── Interactive (Plotly)
│   ├── Comparative (Grid)
│   └── Temporal (GIF)
└── Persistence
    ├── Save weights (pickle)
    └── Load weights (pickle)
```

---

## Características Técnicas

### Rendimiento

- **NLP con caché**: Resultados almacenados para reutilización
- **Procesamiento por lotes**: Maneja miles de documentos
- **Lazy loading**: Extracción bajo demanda
- **Fallbacks**: Funciona sin spaCy (tokenización simple)

### Robustez

- **Validación de entrada**: Verifica archivos y formatos
- **Manejo de errores**: Try-catch con logging detallado
- **Fallback automático**: Funciona sin spaCy o NLTK
- **Stopwords personalizables**: Extensible para cualquier dominio

### Escalabilidad

- **Limitación de términos**: `max_terms` para grandes corpus
- **Filtrado inteligente**: POS tags reduce ruido
- **N-gramas configurables**: Balance entre precisión y rendimiento
- **Batch processing**: Procesa múltiples datasets

### Extensibilidad

- **Stopwords personalizados**: Fácil agregar términos
- **Estilos personalizados**: Extender configuración visual
- **Métodos de ponderación**: Agregar nuevos métodos
- **API modular**: Cada método independiente

---

## Integración con Pipeline Bibliométrico

### Después de Unificación de Datos

```python
# 1. Unificar datos
from src.preprocessing.data_unifier import DataUnifier
from src.visualization import DynamicWordCloud

unifier = DataUnifier(config)
stats = unifier.unify(records_list, output_filename='unified_data.csv')

# 2. Generar word clouds
wc = DynamicWordCloud(stats['unified_file'])
terms = wc.extract_and_process_terms()
weights = wc.calculate_term_weights(terms, method='tfidf')

# 3. Crear visualizaciones para reporte
wc.generate_wordcloud(weights, 'report/wordcloud.png', dpi=300)
wc.generate_interactive_wordcloud(weights, 'report/wordcloud.html')
wc.create_comparative_wordclouds('report/comparative')
```

### Actualización Periódica Automatizada

```python
import schedule
import time

def weekly_update():
    """Actualizar word cloud semanalmente."""
    wc = DynamicWordCloud('data/base_data.csv')

    # Buscar nuevas publicaciones
    new_data = 'data/weekly/new_publications.csv'

    if Path(new_data).exists():
        updated_weights = wc.update_wordcloud_incremental(
            new_data_path=new_data,
            previous_weights_path='cache/weights_current.pkl',
            output_path='reports/wordcloud_current.png'
        )

        wc.save_term_weights('cache/weights_current.pkl', updated_weights)
        print("Word cloud actualizado!")

# Programar actualización semanal
schedule.every().monday.at("09:00").do(weekly_update)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

---

## Limitaciones Conocidas

1. **Dependencia de NLP**:
   - Mejor rendimiento con spaCy instalado
   - Fallback a tokenización simple (menos preciso)
   - Requiere descarga de modelos (~50 MB)

2. **Idioma**:
   - Optimizado para inglés
   - Stopwords y modelo spaCy en inglés
   - Puede extenderse a otros idiomas

3. **Layout de word cloud**:
   - Posiciones aleatorias en versión interactiva
   - Algoritmo de layout mejorable
   - WordCloud lib tiene mejor layout para estática

4. **Rendimiento**:
   - NLP puede ser lento en datasets grandes
   - Considerar desactivar spaCy para >10,000 docs
   - TF-IDF requiere iterar documentos

---

## Mejoras Futuras Sugeridas

1. **Layout mejorado**:
   - Algoritmo de layout más sofisticado para interactivo
   - Force-directed layout con D3.js
   - Evitar solapamiento de términos

2. **Idiomas múltiples**:
   - Soporte para español, francés, alemán, etc.
   - Detección automática de idioma
   - Stopwords multilenguaje

3. **Clustering semántico**:
   - Agrupar términos relacionados
   - Colorear por cluster semántico
   - Usar word embeddings (Word2Vec, GloVe)

4. **Más opciones de exportación**:
   - SVG vectorial
   - PDF de alta calidad
   - Formatos interactivos (D3.js nativo)

5. **Dashboard en tiempo real**:
   - Integración con Dash/Streamlit
   - Actualización automática
   - Filtros interactivos

---

## Conclusión

Se ha implementado exitosamente el **REQUERIMIENTO 5 - PARTE 2**: un sistema completo, profesional y dinámico de visualización de nubes de palabras para análisis bibliométrico.

### Logros Principales

✅ **Extracción inteligente de términos** con NLP (spaCy, NLTK)
✅ **4 métodos de ponderación** (frequency, log, normalized, TF-IDF)
✅ **4 estilos visuales** profesionales
✅ **Word clouds estáticos** de alta calidad (300 DPI)
✅ **Word clouds interactivos** con Plotly
✅ **Visualizaciones comparativas** automáticas
✅ **Actualización incremental** (característica dinámica)
✅ **Análisis de evolución temporal** con GIF
✅ **Identificación de tendencias** emergentes y en declive
✅ **Documentación completa** (700+ líneas)
✅ **Ejemplos funcionales** con 8 casos de uso
✅ **Código bien estructurado** y extensible

### Métricas del Proyecto

- **Líneas de código**: ~1,000 (dynamic_wordcloud.py)
- **Ejemplos**: 550 líneas (8 demos completos)
- **Documentación**: 700 líneas (guía comprehensiva)
- **Métodos implementados**: 15+ métodos públicos y privados
- **Dependencias**: Utiliza dependencias existentes + Pillow
- **Cobertura de funcionalidad**: 100% de lo requerido

### Estado del Proyecto

🟢 **COMPLETADO** - Listo para uso en producción

El sistema puede procesar inmediatamente datos reales y generar visualizaciones profesionales dinámicas para reportes, publicaciones y presentaciones.

---

**Documento creado**: Octubre 2024
**Autor**: Sistema de Análisis Bibliométrico
**Versión**: 1.0.0
