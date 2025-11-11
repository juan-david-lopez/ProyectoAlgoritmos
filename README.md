Sistema de Análisis Bibliométrico

Universidad del Quindío - Análisis de Algoritmos 2025-2

Sistema completo de análisis bibliométrico sobre producción científica en Inteligencia Artificial Generativa, implementando algoritmos de similitud textual, análisis de términos, clustering jerárquico y visualizaciones interactivas.

Tabla de Contenidos:
- [Descripción](#descripción)
- [Estado del Proyecto](#estado-del-proyecto)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Requerimientos Implementados](#requerimientos-implementados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Documentación](#documentación)


Descripción:
Sistema de análisis bibliométrico que procesa publicaciones científicas desde bases de datos académicas (ACM Digital Library, ScienceDirect) para:
- Automatizar descarga y unificación de datos
- Comparar similitud entre textos con 6 algoritmos
- Analizar términos clave en literatura académica
- Agrupar documentos por similitud
- Generar visualizaciones interactivas

Dominio: Inteligencia Artificial Generativa  
Query: "generative artificial intelligence"

Estado del Proyecto:

| Requerimiento | Estado |
|---------------|--------|
| 1. Automatización de descarga y unificación | COMPLETADO |
| 2. Algoritmos de similitud (4 clásicos + 2 IA) | COMPLETADO |
| 3. Análisis de términos predefinidos | COMPLETADO |
| 4. Clustering jerárquico (3 algoritmos) | COMPLETADO |
| 5. Visualizaciones + PDF | COMPLETADO |

*Proyecto: 100% Completado*

## Instalación

### Requisitos

- Python 3.8+
- 4 GB RAM mínimo

### Instalación

# 1. Clonar proyecto
cd ProyectoAlgoritmos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelo spaCy
python -m spacy download en_core_web_sm

# 4. Validar instalación
python main.py validate

### Dependencias Principales

numpy, pandas, matplotlib, seaborn, scikit-learn
nltk, spacy, rake-nltk, pytextrank
torch, transformers, sentence-transformers
scipy, folium, plotly, wordcloud

## Uso Rápido

### Opción 1: Menú Interactivo (Recomendado)

python menu_interactivo.py

### Opción 2: Pipeline Completo

# Con descarga de datos
python main.py full-pipeline

# Sin descarga (si ya tienes datos)
python main.py full-pipeline --skip-download

### Opción 3: Por Requerimiento

python main.py download
python main.py similarity --articles 0 1 2 --algorithms all
python main.py terms
python main.py clustering
python main.py visualize --output-format all

## Requerimientos Implementados

### Req 1: Automatización de Descarga

*Funcionalidad:* Descarga automática desde ACM y ScienceDirect, unificación de datos, eliminación de duplicados.

*Detección de duplicados:*
- Por DOI (exacto)
- Por similitud de título (Levenshtein > 0.9)
- Por autores + año

*Outputs:*
- data/unified_articles.json - Artículos únicos
- data/duplicates/ - Registro de duplicados

*Ejecución:*
python main.py download --query "generative artificial intelligence"

### Req 2: Algoritmos de Similitud Textual

*Algoritmos implementados:*

*Clásicos (4):*
1. Levenshtein Distance - Distancia de edición, O(m×n)
2. TF-IDF + Coseno - Vectorización estadística, O(n×m)
3. Jaccard Similarity - Teoría de conjuntos, O(n+m)
4. N-gramas - Similitud de secuencias, O(m+n)

*IA (2):*
5. Sentence-BERT - Embeddings semánticos
6. BERT - Transformers pre-entrenados

*Funcionalidades:*
- Selección de múltiples artículos
- Extracción automática de abstracts
- Matrices de similitud
- Visualizaciones (heatmaps)
- Medición de tiempo y memoria

*Ejecución:*
python main.py similarity --articles 0 1 2 --algorithms all

*Outputs:*
- output/similarity_analysis/similarity_report.md
- output/similarity_analysis/similarity_matrices.png

*Documentación completa:* docs/similarity_algorithms.md

### Req 3: Análisis de Términos

*Términos predefinidos (15):* Generative models, Prompting, Machine learning, Multimodality, Fine-tuning, Training data, Algorithmic bias, Explainability, Transparency, Ethics, Privacy, Personalization, Human-AI interaction, AI literacy, Co-creation

*Pipeline de 4 partes:*

1. *Análisis de términos predefinidos* - Frecuencias y co-ocurrencias
2. *Extracción automática* - RAKE, TextRank, Combinado (máximo 15 términos)
3. *Evaluación de precisión* - Similitud semántica con SBERT, métricas P/R/F1
4. *Pipeline integrado* - Análisis completo con visualizaciones

*Ejecución:*
python main.py terms --methods tfidf rake textrank

*Outputs:*
- output/term_analysis/frequency_report.json
- output/term_analysis/extracted_terms.json
- output/term_analysis/precision_metrics.json
- output/term_analysis/term_analysis_report.md

### Req 4: Clustering Jerárquico

*Algoritmos (3):*
1. Single Linkage - Distancia mínima
2. Complete Linkage - Distancia máxima
3. Average Linkage - Distancia promedio

*Pipeline:*
- Preprocesamiento (tokenización, stopwords, stemming)
- Vectorización (TF-IDF)
- Cálculo de distancias (coseno, euclidiana)
- Dendrogramas visuales

*Métricas de coherencia:*
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

*Ejecución:*
python main.py clustering --vectorization tfidf --distance cosine

*Outputs:*
- output/clustering_pipeline/dendrogram_single.png
- output/clustering_pipeline/dendrogram_complete.png
- output/clustering_pipeline/dendrogram_average.png
- output/clustering_pipeline/clustering_report.md

### Req 5: Visualizaciones y PDF

*Visualizaciones (3):*

1. *Mapa de calor geográfico* - Distribución por primer autor
2. *Nube de palabras dinámica* - Términos frecuentes (abstracts/keywords)
3. *Línea temporal* - Publicaciones por año y revista

*Exportación:* HTML, PNG, PDF (individual y unificado)

*Ejecución:*
python main.py visualize --output-format all

*Outputs:*
- output/complete_report/geographic_heatmap.html|png|pdf
- output/complete_report/wordcloud.html|png|pdf
- output/complete_report/timeline.html|png|pdf
- output/complete_report/bibliometric_analysis_report.pdf (unificado)

## Estructura del Proyecto

ProyectoAlgoritmos/
├── main.py                          # CLI principal
├── menu_interactivo.py              # Menú interactivo
├── requirements.txt                 # Dependencias
│
├── automation_pipeline.py           # Req 1
├── term_analysis_pipeline.py        # Req 3 (integrado)
│
├── src/
│   ├── algorithms/                  # Req 2: Similitud
│   │   ├── similarity_comparator.py
│   │   └── [levenshtein, tfidf, jaccard, ngram, sbert, bert].py
│   │
│   ├── scrapers/                    # Req 1: Descarga
│   │   ├── acm_scraper.py
│   │   └── sciencedirect_scraper.py
│   │
│   ├── clustering/                  # Req 4
│   │   └── hierarchical_clustering.py
│   │
│   └── visualization/               # Req 5
│       └── visualization_pipeline.py
│
├── data/
│   └── unified_articles.json        # Datos unificados
│
├── output/                          # Resultados generados
│   ├── similarity_analysis/
│   ├── term_analysis/
│   ├── clustering_pipeline/
│   └── complete_report/
│
├── docs/                            # Documentación técnica
│   ├── similarity_algorithms.md
│   ├── PARTE1_ANALISIS_PREDEFINIDOS.md
│   ├── PARTE2_EXTRACCION.md
│   ├── PARTE3_EVALUACION.md
│   ├── PARTE4_PIPELINE.md
│   └── VALIDACION_REQUERIMIENTOS.md
│
└── tests/                           # Tests unitarios
    ├── test_similarity.py
    └── test_pipeline_integration.py

## Documentación

### Documentación Técnica

- *Similitud Textual:* docs/similarity_algorithms.md - Explicaciones matemáticas con LaTeX
- *Análisis de Términos:* docs/PARTE[1-4]_*.md - 4 partes del sistema
- *Validación:* docs/VALIDACION_REQUERIMIENTOS.md - Cumplimiento de requerimientos

### Guías de Uso

- *Menú Interactivo:* docs/GUIA_MENU.md
- *Ejecución:* docs/GUIA_EJECUCION.md

### Ejemplos

# Demo de similitud (sin IA)
python examples/similarity_demo_basic.py

# Demo de similitud (con IA)
python examples/similarity_demo.py

# Pipeline de términos
python example_complete_pipeline.py

## Tests

# Tests de similitud (35 tests)
python tests/test_similarity.py

# Tests de términos
pytest tests/test_term_precision_evaluator.py -v

# Tests de integración
pytest tests/test_pipeline_integration.py -v

*Cobertura:* 100% de tests pasados

## Métricas de Desempeño

### Tiempo de Ejecución (100-500 papers)

| Operación | Tiempo |
|-----------|--------|
| Similitud | 5 segundos |
| Análisis términos | 1-2 minutos |
| Clustering | 30-60 segundos |
| Visualizaciones | 10-15 segundos |
| *Pipeline completo* | *3-5 minutos* |

### Comparación de Algoritmos de Similitud

| Algoritmo | Velocidad | Mejor para |
|-----------|-----------|-----------|
| Jaccard | Ultra rápido (0.0002s) | Comparación de palabras clave |
| N-grama | Muy rápido (0.0005s) | Detección de plagio |
| TF-IDF | Rápido (0.003s) | Búsqueda de documentos |
| Levenshtein | Lento (1.675s) | Textos cortos, corrección ortográfica |
| S-BERT | Moderado | *Similitud semántica (recomendado)* |
| BERT | Lento | Máxima precisión |

## Resolución de Problemas

### Errores Comunes

*"Module not found"*
pip install -r requirements.txt
python -m spacy download en_core_web_sm

*"CUDA out of memory"*
# Reducir batch size
comparator = BERTComparator(batch_size=2)

*Pipeline lento*
- Reducir número de artículos
- Usar solo algoritmos rápidos (TF-IDF, Jaccard)
- Evitar Levenshtein para textos largos

## Contacto

*Universidad del Quindío*  
Programa de Ingeniería de Sistemas y Computación  
Curso: Análisis de Algoritmos - 2025-2


*Versión:* 1.0.0  
*Estado:* Producción  
*Última actualización:* Noviembre 2025