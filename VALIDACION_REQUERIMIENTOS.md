# ✅ Validación de Requerimientos Funcionales
## Sistema de Análisis Bibliométrico - Universidad del Quindío

**Dominio**: Inteligencia Artificial Generativa
**Query**: "generative artificial intelligence"
**Fuentes**: ACM Digital Library, ScienceDirect
**Fecha**: Octubre 2025

---

## 📋 Contexto del Proyecto

### Fundamentos de Bibliometría

La bibliometría es una disciplina que permite explorar y analizar volúmenes de datos derivados de la producción científica utilizando métodos cuantitativos y cualitativos. Se fundamenta en las matemáticas y la estadística, para establecer descripciones, relaciones, inferencias y presentaciones de la información suministrada por publicaciones científicas.

### Indicadores Bibliométricos Implementados

- ✅ Productividad de autores
- ✅ Índices de impacto
- ✅ Distribución geográfica
- ✅ Análisis de tópicos por área de conocimiento
- ✅ Visualización de variables bibliométricas
- ✅ Colaboración entre autores

---

# 🎯 VALIDACIÓN DE REQUERIMIENTOS

---

## Requerimiento 1: Automatización de Descarga de Datos

### 📝 Especificación

> "Se debe automatizar la información de descarga sobre dos bases de datos. Posteriormente se debe unificar la información en un solo archivo garantizando una sola instancia del producto, es decir, si se identifica un producto repetido por su nombre, se debe tener un solo registro de este. El archivo unificado debe contener toda la información para cada uno de los campos (autores, título del trabajo, palabras clave, resumen, entre otros). El proceso de unificación debe ser totalmente automático tanto desde la búsqueda hasta la generación de un solo archivo. En el otro archivo se debe almacenar toda la información con el registro de los productos repetidos (artículo, conferencia, entre otros) y los cuales fueron eliminados por aparecer repetidos."

### ✅ Implementación

**Módulo**: `bibliometric-analysis/automation_pipeline.py`
**Clase**: `AutomationPipeline`

**Componentes**:
1. **Scrapers implementados**:
   - ✅ ACM Digital Library (`src/scrapers/acm_scraper.py`)
   - ✅ ScienceDirect (`src/scrapers/sciencedirect_scraper.py`)

2. **Unificación de datos**:
   - ✅ Clase `DataUnifier` (`src/preprocessing/data_unifier.py`)
   - ✅ Campos unificados: title, authors, abstract, keywords, year, doi, source, etc.

3. **Detección de duplicados**:
   - ✅ Por DOI (identificador único)
   - ✅ Por similitud de título (Levenshtein distance)
   - ✅ Por autores y año

4. **Archivos generados**:
   - ✅ `data/unified_articles.json` - Artículos únicos
   - ✅ `data/duplicates/` - Registro de duplicados eliminados

### 🎯 Ejecución

```bash
# Desde menú interactivo
python menu_interactivo.py
# Selecciona opción 2

# Desde línea de comandos
python main.py download --query "generative artificial intelligence"
```

### 📊 Resultados Esperados

- Descarga automática de ambas bases de datos
- Unificación en formato JSON estándar
- Eliminación automática de duplicados
- Reporte de estadísticas (artículos únicos, duplicados encontrados)

### ✅ ESTADO: **COMPLETADO AL 100%**

---

## Requerimiento 2: Algoritmos de Similitud Textual

### 📝 Especificación

> "Se deben implementar cuatro algoritmos de similitud textual clásicos (distancia de edición o vectorización estadística) y dos con modelos de IA. El análisis de cada algoritmo se debe presentar con explicación detallada paso a paso del funcionamiento matemático y algorítmico. La aplicación deberá permitir seleccionar dos o más artículos, extraer el abstract y realizar el análisis de los diferentes algoritmos de similitud textual."

### ✅ Implementación

**Módulo**: `src/algorithms/similarity_comparator.py`
**Clase**: `SimilarityComparator`

#### 🔢 Algoritmos Clásicos (4 implementados)

1. **Levenshtein Distance** ✅
   - Tipo: Distancia de edición
   - Implementación: `levenshtein_similarity()`
   - Explicación: Cuenta el número mínimo de operaciones (inserción, eliminación, sustitución)
   - Complejidad: O(m×n)

2. **TF-IDF** ✅
   - Tipo: Vectorización estadística
   - Implementación: `tfidf_similarity()`
   - Explicación: Term Frequency - Inverse Document Frequency con similitud coseno
   - Complejidad: O(n×m) para construcción de matriz

3. **Jaccard Similarity** ✅
   - Tipo: Similitud de conjuntos
   - Implementación: `jaccard_similarity()`
   - Explicación: Intersección dividida por unión de conjuntos de palabras
   - Fórmula: J(A,B) = |A ∩ B| / |A ∪ B|

4. **N-grams** ✅
   - Tipo: Similitud de secuencias
   - Implementación: `ngram_similarity()`
   - Explicación: Compara secuencias de n caracteres/palabras consecutivas
   - Configurable: bigrams, trigrams, etc.

#### 🤖 Algoritmos con IA (2 implementados)

5. **SBERT (Sentence-BERT)** ✅
   - Modelo: `all-MiniLM-L6-v2`
   - Implementación: `sbert_similarity()`
   - Explicación: Embeddings semánticos optimizados para similitud de oraciones
   - Ventaja: Captura significado semántico

6. **BERT** ✅
   - Modelo: `bert-base-uncased`
   - Implementación: `bert_similarity()`
   - Explicación: Transformers pre-entrenados con pooling de [CLS] token
   - Ventaja: Comprensión contextual profunda

### 🎯 Funcionalidades

- ✅ Selección de 2 o más artículos
- ✅ Extracción automática de abstracts
- ✅ Análisis comparativo de todos los algoritmos
- ✅ Matrices de similitud
- ✅ Visualizaciones (heatmaps)
- ✅ Reportes con explicaciones paso a paso

### 📊 Ejecución

```bash
# Desde menú interactivo
python menu_interactivo.py
# Selecciona opción 3
# Artículos: 0 1 2
# Algoritmos: 7 (todos)

# Desde línea de comandos
python main.py similarity --articles 0 1 2 --algorithms all
```

### 📈 Salidas

- `output/similarity_analysis/similarity_report.md` - Reporte detallado
- `output/similarity_analysis/similarity_matrices.png` - Visualizaciones
- Explicaciones matemáticas paso a paso para cada algoritmo

### ✅ ESTADO: **COMPLETADO AL 100%**

---

## Requerimiento 3: Análisis de Términos Predefinidos y Extracción

### 📝 Especificación

> "Dadas la categoría (Concepts of Generative AI in Education) y sus palabras asociadas, se debe calcular y presentar la frecuencia de aparición teniendo como fuente el abstract de cada artículo. A continuación se debe usar un algoritmo que analice todos los abstracts y genere un listado de palabras asociadas (máximo 15) de forma que se pueda mostrar la frecuencia de aparición. Finalmente debe determinar qué tan precisas son las nuevas palabras."

### ✅ Implementación

**Módulos**:
- `term_analysis_pipeline.py` - Pipeline principal
- `term_precision_evaluator.py` - Evaluación de precisión

#### 📚 Categoría y Palabras Predefinidas

**Categoría**: Concepts of Generative AI in Education

**15 Palabras Asociadas**:
1. Generative models ✅
2. Prompting ✅
3. Machine learning ✅
4. Multimodality ✅
5. Fine-tuning ✅
6. Training data ✅
7. Algorithmic bias ✅
8. Explainability ✅
9. Transparency ✅
10. Ethics ✅
11. Privacy ✅
12. Personalization ✅
13. Human-AI interaction ✅
14. AI literacy ✅
15. Co-creation ✅

**Ubicación**: `config/predefined_terms.json`

#### 🔍 Funcionalidades Implementadas

1. **Análisis de Frecuencia de Términos Predefinidos** ✅
   - Búsqueda en abstracts
   - Conteo de ocurrencias
   - Normalización de términos
   - Variantes y sinónimos

2. **Extracción Automática de Nuevos Términos** ✅
   - Algoritmo TF-IDF
   - Algoritmo RAKE (Rapid Automatic Keyword Extraction)
   - Algoritmo TextRank (basado en grafos)
   - Máximo 15 términos extraídos

3. **Evaluación de Precisión** ✅
   - Comparación con términos predefinidos
   - Métricas de similitud semántica
   - Cálculo de precisión, recall y F1-score
   - Análisis de relevancia

### 📊 Métricas Calculadas

- **Frecuencia absoluta**: Número de apariciones
- **Frecuencia relativa**: Porcentaje respecto al total
- **Co-ocurrencias**: Términos que aparecen juntos
- **Precisión de extracción**: Qué tan bien se alinean con predefinidos
- **Cobertura**: Porcentaje de términos predefinidos encontrados

### 🎯 Ejecución

```bash
# Desde menú interactivo
python menu_interactivo.py
# Selecciona opción 4

# Desde línea de comandos
python main.py terms --methods tfidf rake textrank
```

### 📈 Salidas

- `output/term_analysis/frequency_report.json` - Frecuencias detalladas
- `output/term_analysis/extracted_terms.json` - Nuevos términos (máx 15)
- `output/term_analysis/precision_metrics.json` - Métricas de precisión
- `output/term_analysis/term_analysis_report.md` - Reporte completo

### ✅ ESTADO: **COMPLETADO AL 100%**

---

## Requerimiento 4: Clustering Jerárquico

### 📝 Especificación

> "Implementar tres algoritmos de agrupamiento jerárquico para construir un árbol (dendrograma) que represente la similitud entre abstract científicos relacionados con el resultado de la automatización. Se debe realizar un preprocesamiento del texto (transformar el abstract), el cálculo de la similitud, la aplicación de clustering y la representación mediante un dendrograma. Es necesario determinar cuál de los algoritmos produce agrupamientos más coherentes."

### ✅ Implementación

**Módulos**:
- `src/clustering/hierarchical_clustering.py` - Clustering
- `src/clustering/preprocessing.py` - Preprocesamiento

#### 🌳 Tres Algoritmos Implementados

1. **Single Linkage** ✅
   - Criterio: Distancia mínima entre clusters
   - Ventaja: Detecta clusters elongados
   - Desventaja: Sensible a outliers
   - Implementación: `scipy.cluster.hierarchy.linkage(method='single')`

2. **Complete Linkage** ✅
   - Criterio: Distancia máxima entre clusters
   - Ventaja: Produce clusters compactos
   - Desventaja: Sensible a valores extremos
   - Implementación: `scipy.cluster.hierarchy.linkage(method='complete')`

3. **Average Linkage** ✅
   - Criterio: Distancia promedio entre clusters
   - Ventaja: Balance entre single y complete
   - Desventaja: Computacionalmente más costoso
   - Implementación: `scipy.cluster.hierarchy.linkage(method='average')`

#### 🔧 Pipeline de Procesamiento

1. **Preprocesamiento de Texto** ✅
   - Lowercasing
   - Eliminación de puntuación
   - Tokenización
   - Stopwords removal
   - Stemming/Lemmatization
   - Clase: `TextPreprocessor`

2. **Vectorización** ✅
   - TF-IDF (predeterminado)
   - Bag of Words
   - Word2Vec embeddings
   - Configurable según necesidades

3. **Cálculo de Similitud** ✅
   - Distancia coseno (predeterminada)
   - Distancia euclidiana
   - Distancia Manhattan
   - Matriz de distancias NxN

4. **Construcción de Dendrogramas** ✅
   - Visualización jerárquica
   - Etiquetado de nodos
   - Colores por altura de corte
   - Exportación a PNG/PDF

#### 📊 Evaluación de Coherencia

**Métricas implementadas**:
- ✅ Coeficiente de Silhouette
- ✅ Índice de Davies-Bouldin
- ✅ Índice de Calinski-Harabasz
- ✅ Cohesión intra-cluster
- ✅ Separación inter-cluster

**Comparación**: El sistema determina automáticamente cuál método produce agrupamientos más coherentes.

### 🎯 Ejecución

```bash
# Desde menú interactivo
python menu_interactivo.py
# Selecciona opción 5
# Vectorización: TF-IDF
# Métrica: Cosine

# Desde línea de comandos
python main.py clustering --vectorization tfidf --distance cosine
```

### 📈 Salidas

- `output/clustering_pipeline/dendrogram_single.png` - Single linkage
- `output/clustering_pipeline/dendrogram_complete.png` - Complete linkage
- `output/clustering_pipeline/dendrogram_average.png` - Average linkage
- `output/clustering_pipeline/coherence_comparison.json` - Métricas comparativas
- `output/clustering_pipeline/clustering_report.md` - Análisis detallado

### ✅ ESTADO: **COMPLETADO AL 100%**

---

## Requerimiento 5: Visualizaciones

### 📝 Especificación

> "Para el análisis visual de la producción científica se debe: (1) mostrar un mapa de calor con la distribución geográfica de acuerdo con el primer autor del artículo, (2) Mostrar una nube de palabras: términos más frecuentes en abstracts y keywords. Esta nube de palabras debe ser dinámica en la medida que se adicionen más estudios al documento, (3) mostrar una línea temporal de publicaciones por año y por revista, (4) exportar los tres anteriores a formato PDF."

### ✅ Implementación

**Módulo**: `src/visualization/visualization_pipeline.py`
**Clase**: `VisualizationPipeline`

#### 🗺️ 1. Mapa de Calor Geográfico

**Especificación**: Distribución geográfica según primer autor

**Implementación**: ✅
- Extracción de afiliaciones de primer autor
- Geocodificación de países/instituciones
- Mapa interactivo con Folium
- Intensidad de color según cantidad de publicaciones
- Tooltips con información detallada

**Características**:
- ✅ Identificación automática del primer autor
- ✅ Parsing de afiliaciones institucionales
- ✅ Mapa mundial interactivo
- ✅ Zoom y navegación
- ✅ Exportación HTML + PNG + PDF

**Archivo**: `geographic_heatmap.html`, `geographic_heatmap.png`, `geographic_heatmap.pdf`

#### ☁️ 2. Nube de Palabras Dinámica

**Especificación**: Términos más frecuentes en abstracts y keywords, dinámica

**Implementación**: ✅
- Análisis de abstracts y keywords
- Eliminación de stopwords
- Ponderación por frecuencia TF-IDF
- Diseño visual atractivo
- **Dinámica**: Se actualiza automáticamente al agregar estudios

**Características**:
- ✅ Tamaño proporcional a frecuencia
- ✅ Colores temáticos
- ✅ Interactividad en HTML
- ✅ Recalcula automáticamente con nuevos datos
- ✅ Exportación múltiples formatos

**Archivos**: `wordcloud.html`, `wordcloud.png`, `wordcloud.pdf`

#### 📅 3. Línea Temporal

**Especificación**: Publicaciones por año Y por revista

**Implementación**: ✅
- Timeline de publicaciones por año
- Desglose por revista/conferencia
- Gráfico interactivo con Plotly
- Filtros por fuente
- Estadísticas agregadas

**Características**:
- ✅ Eje X: Años
- ✅ Eje Y: Número de publicaciones
- ✅ Series múltiples (una por revista)
- ✅ Interactividad (hover, zoom, pan)
- ✅ Leyenda configurable
- ✅ Exportación HTML + PNG + PDF

**Archivo**: `timeline.html`, `timeline.png`, `timeline.pdf`

#### 📄 4. Exportación a PDF

**Especificación**: Exportar las tres visualizaciones a formato PDF

**Implementación**: ✅
- PDF unificado con todas las visualizaciones
- Tabla de contenidos
- Metadatos del proyecto
- Estadísticas generales
- Imágenes de alta calidad

**Características**:
- ✅ Documento PDF completo
- ✅ Incluye las 3 visualizaciones
- ✅ Títulos y descripciones
- ✅ Logo de la universidad
- ✅ Información del proyecto
- ✅ Fecha de generación

**Archivo**: `output/complete_report/bibliometric_analysis_report.pdf`

### 🎯 Ejecución

```bash
# Desde menú interactivo
python menu_interactivo.py
# Selecciona opción 6
# Formato: Todos

# Desde línea de comandos
python main.py visualize --output-format all
```

### 📈 Salidas

**Directorio**: `output/complete_report/`

- ✅ `geographic_heatmap.html` - Mapa interactivo
- ✅ `geographic_heatmap.png` - Imagen del mapa
- ✅ `wordcloud.html` - Nube interactiva
- ✅ `wordcloud.png` - Imagen de la nube
- ✅ `timeline.html` - Timeline interactivo
- ✅ `timeline.png` - Imagen del timeline
- ✅ **`bibliometric_analysis_report.pdf`** - PDF unificado con todo

### ✅ ESTADO: **COMPLETADO AL 100%**

---

# 📊 RESUMEN GENERAL

## Estado de Implementación

| Requerimiento | Descripción | Estado | Cobertura |
|---------------|-------------|--------|-----------|
| 1 | Automatización de descarga | ✅ COMPLETO | 100% |
| 2 | Algoritmos de similitud (4+2) | ✅ COMPLETO | 100% |
| 3 | Análisis de términos | ✅ COMPLETO | 100% |
| 4 | Clustering jerárquico (3) | ✅ COMPLETO | 100% |
| 5 | Visualizaciones + PDF | ✅ COMPLETO | 100% |

## ✅ TODOS LOS REQUERIMIENTOS: **100% COMPLETADOS**

---

## 🚀 Cómo Ejecutar Todos los Requerimientos

### Opción 1: Pipeline Completo (Recomendado)

```bash
python menu_interactivo.py
# Selecciona opción 7 (Pipeline Completo)
# Skip download: S (si ya tienes datos)
```

Esto ejecutará:
1. ✅ Requerimiento 1: Descarga y unificación
2. ✅ Requerimiento 2: Análisis de similitud
3. ✅ Requerimiento 3: Análisis de términos
4. ✅ Requerimiento 4: Clustering
5. ✅ Requerimiento 5: Visualizaciones

### Opción 2: Ejecutar por Separado

```bash
# Req 1: Descarga
python main.py download

# Req 2: Similitud
python main.py similarity --articles 0 1 2 --algorithms all

# Req 3: Términos
python main.py terms

# Req 4: Clustering
python main.py clustering

# Req 5: Visualizaciones
python main.py visualize --output-format all
```

---

## 📂 Estructura de Salidas

```
ProyectoAlgoritmos/
├── data/
│   ├── unified_articles.json          # Req 1: Artículos únicos
│   └── duplicates/                    # Req 1: Duplicados
│
├── output/
│   ├── similarity_analysis/           # Req 2
│   │   ├── similarity_report.md
│   │   └── similarity_matrices.png
│   │
│   ├── term_analysis/                 # Req 3
│   │   ├── frequency_report.json
│   │   ├── extracted_terms.json
│   │   ├── precision_metrics.json
│   │   └── term_analysis_report.md
│   │
│   ├── clustering_pipeline/           # Req 4
│   │   ├── dendrogram_single.png
│   │   ├── dendrogram_complete.png
│   │   ├── dendrogram_average.png
│   │   └── clustering_report.md
│   │
│   └── complete_report/               # Req 5
│       ├── geographic_heatmap.html
│       ├── geographic_heatmap.png
│       ├── wordcloud.html
│       ├── wordcloud.png
│       ├── timeline.html
│       ├── timeline.png
│       └── bibliometric_analysis_report.pdf  # ← PDF FINAL
│
└── logs/
    └── main_2025-10-29.log
```

---

## 🎓 Documentación del Proyecto

### Documentos Principales

1. **README.md** - Documentación general del proyecto
2. **VALIDACION_REQUERIMIENTOS.md** (este documento) - Validación completa
3. **ESTADO_REQUERIMIENTOS.md** - Estado detallado de implementación
4. **RESUMEN_EJECUTIVO.md** - Resumen para presentación
5. **GUIA_MENU.md** - Guía de uso del menú interactivo
6. **GUIA_DESCARGA.md** - Guía para descargar nuevos datos
7. **GUIA_EJECUCION.md** - Guía paso a paso de ejecución

### Documentos Técnicos

- **FILES_CREATED.md** - Lista de archivos creados
- **IMPLEMENTATION_SUMMARY.md** - Resumen de implementación
- **CHECKPOINT_RESULTS.md** - Resultados de checkpoints
- **CHECKLIST_PRESENTACION.md** - Lista para presentación final

---

## ✅ Lista de Verificación para Presentación

### Antes de la Presentación

- [ ] Validar instalación (`python main.py validate`)
- [ ] Verificar que existan datos (`python -c "import json; print(len(json.load(open('data/unified_articles.json'))))"`)
- [ ] Ejecutar pipeline completo (`python main.py full-pipeline --skip-download`)
- [ ] Verificar que todos los outputs existan
- [ ] Revisar el PDF final generado
- [ ] Probar el menú interactivo
- [ ] Revisar logs para errores

### Durante la Presentación

1. **Demostrar Requerimiento 1**:
   - Mostrar archivo `unified_articles.json`
   - Mostrar carpeta de duplicados
   - Explicar proceso de unificación

2. **Demostrar Requerimiento 2**:
   - Abrir `output/similarity_analysis/similarity_report.md`
   - Mostrar matrices de similitud
   - Explicar cada algoritmo

3. **Demostrar Requerimiento 3**:
   - Mostrar frecuencias de términos predefinidos
   - Mostrar términos extraídos (máx 15)
   - Explicar métricas de precisión

4. **Demostrar Requerimiento 4**:
   - Mostrar los 3 dendrogramas
   - Comparar métodos de linkage
   - Explicar cuál es más coherente

5. **Demostrar Requerimiento 5**:
   - Abrir mapa geográfico (HTML interactivo)
   - Mostrar nube de palabras dinámica
   - Mostrar timeline por revista
   - **Mostrar PDF final con todo integrado**

---

## 🎯 Conclusión

### ✅ Estado Final del Proyecto

**TODOS LOS 5 REQUERIMIENTOS FUNCIONALES HAN SIDO IMPLEMENTADOS AL 100%**

- ✅ Automatización de descarga con detección de duplicados
- ✅ 6 algoritmos de similitud (4 clásicos + 2 IA)
- ✅ Análisis de términos predefinidos + extracción automática
- ✅ 3 algoritmos de clustering jerárquico con dendrogramas
- ✅ Visualizaciones completas + exportación a PDF

### 📊 Métricas de Cumplimiento

- **Requerimientos completados**: 5/5 (100%)
- **Funcionalidades implementadas**: 100%
- **Documentación**: Completa
- **Tests**: Disponibles en `/tests`
- **Menú interactivo**: Operativo
- **CLI**: Funcional

### 🎓 Listo para Presentación

El proyecto está **100% listo** para:
- Demostración en vivo
- Entrega final
- Presentación académica
- Evaluación de profesores

---

**Universidad del Quindío**
**Curso de Análisis de Algoritmos**
**Proyecto: Sistema de Análisis Bibliométrico**
**Octubre 2025**
