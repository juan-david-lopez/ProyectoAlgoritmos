# 📊 Bibliometric Analysis - Inteligencia Artificial Generativa

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📖 Descripción del Proyecto

Sistema completo de **análisis bibliométrico** diseñado para el estudio sistemático de publicaciones científicas sobre **"inteligencia artificial generativa"**.

El proyecto implementa un pipeline end-to-end que incluye:

- 🔍 **Descarga automatizada** de datos desde bases académicas (IEEE Xplore, Scopus, Web of Science)
- 🔄 **Detección inteligente de duplicados** usando múltiples algoritmos de similitud
- 🧹 **Preprocesamiento avanzado** con NLP en español e inglés
- 📈 **Clustering temático** con K-means, DBSCAN y clustering jerárquico
- 📊 **Visualización interactiva** de tendencias, redes y distribuciones
- 📄 **Generación automática de reportes** en formato PDF con estadísticas y gráficos

### 🎯 Objetivos

1. Automatizar la recolección de datos bibliográficos de múltiples fuentes
2. Identificar y eliminar publicaciones duplicadas con alta precisión
3. Analizar tendencias temporales y geográficas en investigación de IA generativa
4. Descubrir agrupaciones temáticas mediante técnicas de machine learning
5. Generar visualizaciones y reportes profesionales para análisis académico

## 📁 Estructura del Proyecto

```
bibliometric-analysis/
├── 📂 config/                  # Configuración
│   ├── config.yaml            # Parámetros principales (640+ líneas)
│   └── .env.example           # Template de variables de entorno
│
├── 📂 data/                    # Datos (no versionados)
│   ├── raw/                   # Datos descargados originales
│   ├── processed/             # Datos procesados y limpios
│   └── duplicates/            # Registros duplicados identificados
│
├── 📂 src/                     # Código fuente
│   ├── scrapers/              # Módulos de descarga
│   │   ├── ieee_scraper.py
│   │   ├── scopus_scraper.py
│   │   └── wos_scraper.py
│   │
│   ├── algorithms/            # Algoritmos de similitud
│   │   ├── levenshtein.py
│   │   ├── jaro_winkler.py
│   │   └── jaccard.py
│   │
│   ├── preprocessing/         # Limpieza y preprocesamiento
│   │   ├── data_cleaner.py
│   │   ├── deduplicator.py
│   │   └── text_processor.py
│   │
│   ├── clustering/            # Algoritmos de clustering
│   │   ├── kmeans_clustering.py
│   │   ├── dbscan_clustering.py
│   │   └── hierarchical_clustering.py
│   │
│   ├── visualization/         # Generación de gráficos
│   │   ├── temporal_plots.py
│   │   ├── geographic_maps.py
│   │   ├── network_graphs.py
│   │   └── cluster_plots.py
│   │
│   └── utils/                 # Utilidades generales
│       ├── config_loader.py   # Carga de configuración
│       ├── logger.py          # Sistema de logging
│       └── file_handler.py    # Manejo de archivos
│
├── 📂 scripts/                 # Scripts de utilidad
│   ├── verify_installation.py
│   └── download_nlp_models.py
│
├── 📂 docs/                    # Documentación
│   └── SETUP.md               # Guía de instalación detallada
│
├── 📂 tests/                   # Tests unitarios
│   ├── test_scrapers.py
│   ├── test_algorithms.py
│   └── test_clustering.py
│
├── 📂 outputs/                 # Resultados (no versionados)
│   ├── reports/               # Reportes PDF generados
│   └── visualizations/        # Gráficos e imágenes
│
├── 📂 notebooks/               # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
│
├── 📂 logs/                    # Logs de ejecución
│
├── main.py                    # Punto de entrada principal
├── requirements.txt           # Dependencias Python (50+ paquetes)
├── README.md                  # Este archivo
└── .gitignore                 # Archivos excluidos de git
```

## 🎯 Requerimientos del Proyecto

### 1️⃣ Descarga de Datos (Web Scraping)

**Descripción**: Automatización de la descarga de publicaciones científicas desde múltiples bases de datos académicas.

**Fuentes de datos**:
- 📚 **IEEE Xplore**: Publicaciones de ingeniería y tecnología
- 📚 **Scopus**: Base de datos multidisciplinaria de Elsevier
- 📚 **Web of Science**: Índice de citas de Clarivate

**Características**:
- Query: "inteligencia artificial generativa"
- Soporte para API y web scraping
- Rate limiting automático
- Manejo de errores y reintentos
- Extracción de campos: título, autores, abstract, DOI, año, keywords, citas

**Formato de salida**: CSV con campos estandarizados

---

### 2️⃣ Detección de Duplicados

**Descripción**: Identificación y eliminación de publicaciones duplicadas usando múltiples algoritmos de similitud de texto.

**Algoritmos implementados**:
- 🔤 **Levenshtein Distance**: Distancia de edición entre cadenas
- 🔤 **Jaro-Winkler**: Similitud de cadenas con énfasis en prefijos
- 🔤 **Jaccard Index**: Similitud basada en conjuntos de palabras

**Campos analizados**:
- Título (weight: 0.4)
- Abstract (weight: 0.3)
- DOI (exact match)
- Autores (weight: 0.3)

**Thresholds**:
- Similitud de título: ≥ 85%
- Similitud de abstract: ≥ 80%
- Similitud combinada: ≥ 75%

**Salida**: Archivo de duplicados con ID de grupo y métricas de similitud

---

### 3️⃣ Preprocesamiento de Datos

**Descripción**: Limpieza, normalización y transformación de datos bibliográficos.

**Operaciones**:
- ✅ Normalización de texto (lowercase, whitespace)
- ✅ Eliminación de HTML tags, URLs, emails
- ✅ Tokenización y lemmatización (spaCy)
- ✅ Eliminación de stop words (español/inglés)
- ✅ Validación de campos requeridos
- ✅ Estandarización de formatos de fecha
- ✅ Parsing de listas de autores

**Lenguajes soportados**: Español (primario), Inglés (secundario)

---

### 4️⃣ Clustering Temático

**Descripción**: Agrupación automática de publicaciones por similitud temática usando técnicas de machine learning.

**Algoritmos**:

1. **K-Means**
   - Número de clusters: 5 (configurable)
   - Feature extraction: TF-IDF o Sentence Transformers
   - Optimización automática con método del codo

2. **DBSCAN**
   - Epsilon: 0.5 (auto-tuning disponible)
   - Min samples: 5
   - Detección automática de outliers

3. **Clustering Jerárquico**
   - Linkage: Ward
   - Generación de dendrogramas
   - Corte adaptativo

**Features utilizadas**:
- Título + Abstract + Keywords
- Vectorización con TF-IDF (1000 features)
- Reducción dimensional: PCA/t-SNE/UMAP

**Evaluación**:
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

---

### 5️⃣ Visualización de Resultados

**Descripción**: Generación de gráficos interactivos y estáticos para análisis visual de resultados.

**Visualizaciones implementadas**:

📈 **Temporal**:
- Tendencia de publicaciones por año
- Tasa de crecimiento

🌍 **Geográfica**:
- Mapa coroplético de distribución por país
- Top 20 países productores

📊 **Distribución**:
- Publicaciones por fuente (IEEE, Scopus, WOS)
- Top journals y conferencias

🕸️ **Redes**:
- Red de coautoría (NetworkX)
- Comunidades de investigación

☁️ **Análisis de texto**:
- Word cloud de keywords
- Frecuencia de términos

📍 **Clustering**:
- Scatter plot 2D/3D de clusters
- Heatmap de similitud

**Formatos de salida**: PNG (300 DPI), SVG, HTML interactivo (Plotly)

---

### 6️⃣ Reporte Automatizado

**Descripción**: Generación automática de reportes profesionales en formato PDF con análisis completo.

**Secciones del reporte**:
1. 📄 Portada con metadata
2. 📑 Tabla de contenidos
3. 📝 Resumen ejecutivo
4. 🔬 Metodología
5. 📊 Resultados y estadísticas
6. 📈 Visualizaciones
7. 💡 Conclusiones
8. 📚 Referencias

**Estadísticas incluidas**:
- Total de publicaciones
- Distribución temporal
- Top autores (más productivos, más citados)
- Top países e instituciones
- Top fuentes de publicación
- Métricas de citas
- Análisis de keywords
- Resumen de clusters

**Formato**: PDF con tipografía profesional (Times New Roman + Arial)

## ⚙️ Instalación

### 🚀 Instalación Rápida

```bash
# 1. Navegar al directorio del proyecto
cd bibliometric-analysis

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 4. Actualizar pip
python -m pip install --upgrade pip

# 5. Instalar dependencias (10-15 minutos)
pip install -r requirements.txt

# 6. Descargar modelos NLP
python scripts/download_nlp_models.py

# 7. Verificar instalación
python scripts/verify_installation.py

# 8. Configurar variables de entorno
cp config/.env.example config/.env
# Editar config/.env con tus credenciales de API (opcional)
```

### 📋 Requisitos del Sistema

- **Python**: 3.8 o superior
- **RAM**: Mínimo 8 GB (recomendado 16 GB)
- **Espacio**: 10 GB libres
- **SO**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)

### 🔑 Configuración de API Keys (Opcional)

Para usar las APIs oficiales (en lugar de web scraping):

1. **Scopus API**: Registrarse en https://dev.elsevier.com/
2. **Web of Science API**: Registrarse en https://developer.clarivate.com/
3. **IEEE Xplore API**: Registrarse en https://developer.ieee.org/

Agregar las keys en `config/.env`:
```bash
SCOPUS_API_KEY=tu_clave_aqui
WOS_API_KEY=tu_clave_aqui
IEEE_API_KEY=tu_clave_aqui
```

### 📖 Instalación Detallada

Ver [docs/SETUP.md](docs/SETUP.md) para:
- Instrucciones paso a paso
- Troubleshooting
- Configuración avanzada
- Instalación de wkhtmltopdf para PDFs

---

## 🚀 Cómo Usar

### Ejecución Completa (Pipeline End-to-End)

```bash
# Ejecutar todos los requerimientos en secuencia
python main.py --mode full
```

### Ejecución por Módulos

#### 1️⃣ Descarga de Datos

```bash
# Ejecutar todos los scrapers
python main.py --mode scrape

# O ejecutar scrapers individuales
python -m src.scrapers.ieee_scraper
python -m src.scrapers.scopus_scraper
python -m src.scrapers.wos_scraper
```

**Salida**: `data/raw/publications_{source}_{timestamp}.csv`

---

#### 2️⃣ Detección y Eliminación de Duplicados

```bash
# Ejecutar deduplicación
python main.py --mode preprocess

# O ejecutar directamente
python -m src.preprocessing.deduplicator \
    --input data/raw/ \
    --output data/processed/publications_clean.csv \
    --duplicates data/duplicates/duplicates.csv
```

**Parámetros configurables** (en `config/config.yaml`):
- Thresholds de similitud
- Algoritmos a usar
- Campos a comparar

**Salida**:
- `data/processed/publications_clean.csv` - Datos sin duplicados
- `data/duplicates/duplicates.csv` - Registros duplicados con métricas

---

#### 3️⃣ Preprocesamiento de Texto

```bash
# Preprocesamiento automático
python -m src.preprocessing.text_processor \
    --input data/processed/publications_clean.csv \
    --output data/processed/publications_preprocessed.csv
```

**Operaciones realizadas**:
- Limpieza de texto
- Tokenización
- Lemmatización (spaCy)
- Eliminación de stop words
- Validación de datos

---

#### 4️⃣ Clustering Temático

```bash
# Ejecutar todos los algoritmos
python main.py --mode cluster

# O ejecutar algoritmos individuales
python -m src.clustering.kmeans_clustering
python -m src.clustering.dbscan_clustering
python -m src.clustering.hierarchical_clustering
```

**Configuración** (`config/config.yaml`):
```yaml
clustering:
  algorithms:
    kmeans:
      n_clusters: 5
    dbscan:
      eps: 0.5
      min_samples: 5
```

**Salida**:
- `data/processed/publications_clustered.csv` - Datos con labels de cluster
- `outputs/clustering_metrics.json` - Métricas de evaluación

---

#### 5️⃣ Visualización

```bash
# Generar todas las visualizaciones
python main.py --mode visualize

# O generar visualizaciones específicas
python -m src.visualization.temporal_plots
python -m src.visualization.geographic_maps
python -m src.visualization.network_graphs
python -m src.visualization.cluster_plots
```

**Salida**: `outputs/visualizations/`
- `temporal_trends.png`
- `country_distribution.png`
- `coauthorship_network.png`
- `keyword_cloud.png`
- `cluster_visualization.png`
- `*.html` - Versiones interactivas (Plotly)

---

#### 6️⃣ Generación de Reporte

```bash
# Generar reporte completo en PDF
python main.py --mode report

# O ejecutar directamente
python -m src.visualization.report_generator \
    --output outputs/reports/bibliometric_analysis_report.pdf
```

**Salida**: `outputs/reports/bibliometric_analysis_report_{timestamp}.pdf`

---

### Ejemplos de Uso Avanzado

```bash
# Ejecutar solo IEEE y Scopus
python main.py --mode scrape --sources ieee,scopus

# Clustering con K-Means solamente
python -m src.clustering.kmeans_clustering --n-clusters 7

# Generar solo visualizaciones temporales
python -m src.visualization.temporal_plots --years 2018-2024

# Reporte en HTML en lugar de PDF
python main.py --mode report --format html
```

### Interfaz Web Interactiva (Streamlit)

```bash
# Lanzar dashboard interactivo
streamlit run app.py

# Acceder en el navegador
# http://localhost:8501
```

### Jupyter Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir notebooks en notebooks/
# - exploratory_analysis.ipynb
# - results_visualization.ipynb
```

---

## 🛠️ Tecnologías y Herramientas

### Core Technologies
- ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white) **Python 3.8+**
- **Data Processing**: Pandas, NumPy, OpenPyXL
- **Web Scraping**: Selenium, BeautifulSoup4, Requests, WebDriver Manager

### NLP & Machine Learning
- **NLP**: NLTK, spaCy, python-Levenshtein, Jellyfish
- **ML/Clustering**: Scikit-learn, SciPy
- **Deep Learning**: PyTorch (CPU), Transformers, Sentence-Transformers

### Visualization & Reporting
- **Charts**: Matplotlib, Seaborn, Plotly, WordCloud
- **Maps**: Folium (mapas interactivos)
- **Networks**: NetworkX
- **Reports**: ReportLab, FPDF, PDFKit

### Additional Tools
- **Web App**: Streamlit
- **Bibliographic Parsing**: python-RISparser, bibtexparser
- **Utilities**: PyYAML, python-dotenv, tqdm, loguru

**Total**: ~50 paquetes de Python

---

## 📊 Configuración

Todo el proyecto es configurable mediante archivos:

### `config/config.yaml` (640+ líneas)
Configuración completa de:
- Queries y fuentes de datos
- Parámetros de scraping y rate limiting
- Thresholds de deduplicación
- Configuración de clustering (K-means, DBSCAN, Hierarchical)
- Estilos de visualización
- Formato de reportes

### `config/.env`
Variables de entorno sensibles:
- API keys (Scopus, WOS, IEEE)
- Credenciales de base de datos
- Configuración de scraping

### Ejemplo de configuración:

```yaml
# config/config.yaml
query:
  keywords: "inteligencia artificial generativa"
  date_range:
    start: "2018-01-01"
    end: null

clustering:
  algorithms:
    kmeans:
      n_clusters: 5
    dbscan:
      eps: 0.5
```

---

## 📈 Estado del Proyecto

### Infraestructura ✅
- [x] Estructura de carpetas completa
- [x] Configuración exhaustiva (YAML + .env)
- [x] Sistema de logging
- [x] Utilidad de carga de configuración
- [x] Scripts de instalación y verificación
- [x] .gitignore completo (330+ líneas)

### Requerimientos del Proyecto
- [ ] 1️⃣ Módulos de descarga (IEEE, Scopus, WOS)
- [ ] 2️⃣ Sistema de deduplicación (Levenshtein, Jaro-Winkler, Jaccard)
- [ ] 3️⃣ Preprocesamiento de datos
- [ ] 4️⃣ Clustering (K-means, DBSCAN, Jerárquico)
- [ ] 5️⃣ Visualización (9 tipos de gráficos)
- [ ] 6️⃣ Generación de reportes PDF

### Próximos Pasos
1. Implementar scrapers para las 3 fuentes de datos
2. Desarrollar sistema de deduplicación
3. Crear pipeline de preprocesamiento
4. Implementar algoritmos de clustering
5. Generar visualizaciones
6. Crear generador de reportes

---

## 🤝 Contribuir

### Estructura de Commits
```bash
git commit -m "feat: Add IEEE scraper implementation"
git commit -m "fix: Resolve duplicate detection threshold issue"
git commit -m "docs: Update README with usage examples"
```

### Testing
```bash
# Ejecutar tests
pytest tests/

# Con coverage
pytest --cov=src tests/

# Test específico
pytest tests/test_scrapers.py::test_ieee_scraper
```

### Code Quality
```bash
# Format code
black src/

# Linting
flake8 src/

# Type checking (opcional)
mypy src/
```

---

## 📚 Documentación Adicional

- 📖 [Guía de Instalación Detallada](docs/SETUP.md)
- 📊 [Configuración de config.yaml](config/config.yaml)
- 🔑 [Variables de Entorno](config/.env.example)
- 🧪 [Jupyter Notebooks](notebooks/)

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## ✨ Características Destacadas

- 🚀 **Pipeline automatizado end-to-end**
- 🔧 **Altamente configurable** (640+ líneas de config)
- 🌍 **Soporte multilenguaje** (Español/Inglés)
- 📊 **Múltiples algoritmos de clustering**
- 🎨 **Visualizaciones interactivas** (Plotly)
- 📄 **Reportes profesionales** en PDF
- 🧪 **Testing completo**
- 📚 **Documentación exhaustiva**

---

## 👥 Autores

Proyecto de Análisis Bibliométrico - 2025

**Equipo de Investigación**

---

## 🙏 Agradecimientos

Este proyecto utiliza datos de:
- IEEE Xplore Digital Library
- Elsevier Scopus
- Clarivate Web of Science

---

## 📧 Contacto

Para preguntas, sugerencias o reportar issues:
- 📬 Email: [tu-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/tu-usuario/bibliometric-analysis/issues)

---

**⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub!**
