# Proyecto: Sistema de Análisis de Términos en Literatura Académica

## 🎯 Descripción General

Sistema completo para análisis y evaluación de términos en corpus académico, integrando análisis de frecuencias, extracción automática y evaluación de precisión con similitud semántica.

---

## 📦 Componentes del Sistema

### Arquitectura Modular

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED DATA INPUT                          │
│               (unified_abstracts.json)                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────┐
    │  PARTE 1: Análisis de Términos Predefinidos          │
    │  • predefined_term_analyzer.py                        │
    │  • Frecuencias y co-ocurrencias                       │
    │  • Identificación de términos clave                   │
    └───────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────┐
    │  PARTE 2: Extracción Automática de Términos          │
    │  • auto_term_extractor.py                             │
    │  • RAKE (Rapid Automatic Keyword Extraction)          │
    │  • TextRank (Graph-based ranking)                     │
    │  • Método combinado                                   │
    └───────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────┐
    │  PARTE 3: Evaluación de Precisión                    │
    │  • term_precision_evaluator.py                        │
    │  • Similitud semántica con SBERT                      │
    │  • Métricas: Precision, Recall, F1-Score              │
    │  • Análisis de términos nuevos                        │
    └───────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────┐
    │  PARTE 4: Pipeline Completo                          │
    │  • term_analysis_pipeline.py                          │
    │  • Integración de todas las partes                    │
    │  • Visualizaciones comparativas                       │
    │  • Reporte maestro consolidado                        │
    └───────────────────────────────────────────────────────┘
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación Completa

```bash
# 1. Clonar o descargar el proyecto
cd ProyectoAlgoritmos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelo spaCy (para TextRank)
python -m spacy download en_core_web_sm

# 4. (Opcional) Verificar instalación
python -c "import nltk, spacy, sentence_transformers; print('✓ Todo instalado')"
```

### Dependencias Principales

```
# Análisis y visualización
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0

# Extracción de términos
nltk>=3.8.0
rake-nltk>=1.0.6
spacy>=3.5.0
pytextrank>=3.2.0

# Evaluación semántica
sentence-transformers>=2.2.0
matplotlib-venn>=0.11.9
```

---

## 📖 Documentación por Componente

### Parte 1: Análisis de Términos Predefinidos

**Archivo**: `predefined_term_analyzer.py`

**Funcionalidad**:
- Cálculo de frecuencias de términos
- Análisis de co-ocurrencias
- Identificación de términos más relevantes
- Generación de reportes con visualizaciones

**Documentación**: `PARTE1_ANALISIS_PREDEFINIDOS.md`

**Uso rápido**:
```python
from predefined_term_analyzer import PredefinedTermAnalyzer

analyzer = PredefinedTermAnalyzer(abstracts, predefined_terms)
frequencies = analyzer.calculate_term_frequencies()
analyzer.generate_report('report.md')
```

---

### Parte 2: Extracción Automática

**Archivo**: `auto_term_extractor.py`

**Funcionalidad**:
- RAKE: Extracción basada en co-ocurrencias
- TextRank: Ranking basado en grafos
- Método combinado con fusión de scores
- Comparación de métodos

**Documentación**: `PARTE2_EXTRACCION.md`

**Uso rápido**:
```python
from auto_term_extractor import AutoTermExtractor

extractor = AutoTermExtractor(abstracts)
extractor.extract_with_rake()
extractor.extract_with_textrank()
terms = extractor.get_combined_top_terms(n=50)
```

---

### Parte 3: Evaluación de Precisión

**Archivo**: `term_precision_evaluator.py`

**Funcionalidad**:
- Similitud semántica con SBERT
- Identificación de matches (exactos, parciales, nuevos)
- Métricas: Precision, Recall, F1-Score, Coverage
- Análisis contextual de términos nuevos

**Documentación**: `PARTE3_EVALUACION.md` y `README_PARTE3.md`

**Uso rápido**:
```python
from term_precision_evaluator import TermPrecisionEvaluator

evaluator = TermPrecisionEvaluator(predefined_terms, extracted_terms)
metrics = evaluator.calculate_metrics()
evaluator.generate_evaluation_report('report.md', abstracts)
```

---

### Parte 4: Pipeline Completo

**Archivo**: `term_analysis_pipeline.py`

**Funcionalidad**:
- Integración de todas las partes
- Workflow automatizado completo
- Visualizaciones comparativas
- Reporte maestro consolidado

**Documentación**: `PARTE4_PIPELINE.md` y `README_PARTE4.md`

**Uso rápido**:
```python
from term_analysis_pipeline import run_complete_analysis

pipeline = run_complete_analysis(
    'unified_abstracts.json',
    'analysis_output'
)
```

---

## 🎮 Ejemplos Ejecutables

### Ejemplo 1: Pipeline Completo (MÁS SIMPLE)

```bash
python example_complete_pipeline.py
```

Ejecuta el pipeline completo con datos de muestra y genera todos los reportes.

### Ejemplo 2: Análisis por Partes

```python
# Parte 1
from predefined_term_analyzer import PredefinedTermAnalyzer
analyzer = PredefinedTermAnalyzer(abstracts, predefined_terms)
analyzer.generate_report('part1_report.md')

# Parte 2
from auto_term_extractor import AutoTermExtractor
extractor = AutoTermExtractor(abstracts)
extractor.extract_with_rake()
extractor.extract_with_textrank()

# Parte 3
from term_precision_evaluator import TermPrecisionEvaluator
evaluator = TermPrecisionEvaluator(
    predefined_terms,
    extractor.get_combined_top_terms(50)
)
evaluator.generate_evaluation_report('part3_report.md', abstracts)
```

### Ejemplo 3: Workflow Completo con Datos Reales

```python
# 1. Obtener unified_abstracts.json del buscador académico

# 2. Ejecutar pipeline
from term_analysis_pipeline import run_complete_analysis

pipeline = run_complete_analysis(
    'unified_abstracts.json',
    'results'
)

# 3. Analizar resultados
best_method = max(
    pipeline.evaluation_results.keys(),
    key=lambda m: pipeline.evaluation_results[m]['metrics']['f1_score']
)

print(f"Mejor método: {best_method}")
print(f"F1-Score: {pipeline.evaluation_results[best_method]['metrics']['f1_score']:.2%}")

# 4. Revisar reporte
print("Reporte: results/reports/term_analysis_report.md")
```

---

## 🧪 Tests

### Tests Unitarios

```bash
# Parte 1 (si se implementan)
pytest test_predefined_term_analyzer.py -v

# Parte 2 (si se implementan)
pytest test_auto_term_extractor.py -v

# Parte 3
pytest test_term_precision_evaluator.py -v

# Parte 4 - Integración
pytest test_pipeline_integration.py -v
```

### Ejecutar Todos los Tests

```bash
pytest . -v --tb=short
```

---

## 📊 Outputs del Sistema

### Estructura de Archivos Generados

```
output_dir/
│
├── data/
│   ├── predefined_terms_frequencies.csv
│   ├── extracted_terms_all_methods.csv
│   └── evaluation_metrics.json
│
├── reports/
│   ├── term_analysis_report.md           ⭐ REPORTE MAESTRO
│   ├── predefined_terms_report.md
│   ├── extracted_terms_report.md
│   ├── evaluation_rake.md
│   ├── evaluation_textrank.md
│   └── evaluation_combined.md
│
└── visualizations/
    ├── metrics_comparison.png
    ├── frequency_distribution.png
    ├── methods_overlap.png
    ├── top_terms_comparison.png
    └── evaluation_*/
        ├── venn_diagram.png
        └── similarity_heatmap.png
```

---

## 📈 Métricas y Evaluación

### Métricas Calculadas

| Métrica | Descripción | Rango |
|---------|-------------|-------|
| **Precision** | Proporción de términos extraídos relevantes | 0-100% |
| **Recall** | Proporción de términos predefinidos encontrados | 0-100% |
| **F1-Score** | Media armónica de P y R | 0-100% |
| **Coverage** | Porcentaje de términos predefinidos cubiertos | 0-100% |

### Interpretación de Resultados

- **F1 ≥ 70%**: Excelente desempeño
- **F1 60-69%**: Buen desempeño
- **F1 50-59%**: Aceptable, puede mejorar
- **F1 < 50%**: Requiere ajustes

---

## 🎨 Visualizaciones Generadas

### 1. Comparación de Métricas
Gráfico de barras comparando Precision, Recall y F1 entre métodos.

### 2. Distribución de Frecuencias
- Top 15 términos predefinidos
- Histograma de distribución

### 3. Overlap entre Métodos
Diagrama de Venn (3 conjuntos) mostrando términos compartidos.

### 4. Top Términos por Método
Tablas visuales de top 10 términos.

### 5. Similitud Semántica
Heatmaps de similitud entre términos predefinidos y extraídos.

---

## ⚙️ Configuración y Personalización

### Ajustar Parámetros de Extracción

```python
# En auto_term_extractor.py

# RAKE
extractor.extract_with_rake(
    min_phrase_length=1,
    max_phrase_length=4,
    min_keyword_frequency=2
)

# TextRank
extractor.extract_with_textrank(
    limit_phrases=50,
    limit_ratio=0.25
)
```

### Ajustar Threshold de Similitud

```python
# En term_precision_evaluator.py
matches = evaluator.identify_matches(threshold=0.75)  # Default: 0.70
```

### Personalizar Visualizaciones

Modificar métodos `_create_*_chart()` en cada componente para ajustar:
- Colores
- Tamaños de figura
- Fuentes
- Estilos

---

## 🔧 Troubleshooting

### Problema: Módulos no encontrados

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Problema: Error con SBERT

```bash
# Reinstalar
pip uninstall sentence-transformers
pip install sentence-transformers
```

### Problema: Pipeline muy lento

**Soluciones**:
1. Reducir número de términos extraídos
2. Usar GPU si está disponible
3. Procesar en batches más pequeños
4. Limitar tamaño del corpus

### Problema: Métricas muy bajas

**Causas**:
- Términos predefinidos no representativos
- Corpus muy ruidoso

**Soluciones**:
1. Revisar calidad de términos predefinidos
2. Filtrar abstracts cortos o de baja calidad
3. Ajustar parámetros de extracción

---

## 📚 Documentación Adicional

### Por Componente
- `PARTE1_ANALISIS_PREDEFINIDOS.md`
- `PARTE2_EXTRACCION.md`
- `PARTE3_EVALUACION.md` / `README_PARTE3.md`
- `PARTE4_PIPELINE.md` / `README_PARTE4.md`

### Ejemplos
- `example_complete_pipeline.py`: Ejemplos interactivos del pipeline
- `example_precision_evaluation.py`: Ejemplos de evaluación

### Tests
- `test_term_precision_evaluator.py`: Tests unitarios Parte 3
- `test_pipeline_integration.py`: Tests de integración completos

---

## 🎯 Casos de Uso

### 1. Análisis de Literatura Científica

```python
# Cargar papers de un dominio específico
pipeline = run_complete_analysis(
    'deep_learning_papers.json',
    'dl_analysis'
)

# Identificar términos emergentes
novel_terms = pipeline.evaluation_results['Combined']['matches']['novel_terms']
```

### 2. Validación de Taxonomía

```python
# Verificar si taxonomía cubre el dominio
evaluator = TermPrecisionEvaluator(taxonomy_terms, extracted_terms)
metrics = evaluator.calculate_metrics()

if metrics['coverage'] < 70:
    print("⚠️ Taxonomía incompleta")
```

### 3. Comparación de Métodos de Extracción

```python
# Evaluar múltiples métodos
pipeline = run_complete_analysis('data.json', 'output')

# Ver comparación en reporte maestro
# o acceder programáticamente:
for method in ['RAKE', 'TextRank', 'Combined']:
    f1 = pipeline.evaluation_results[method]['metrics']['f1_score']
    print(f"{method}: {f1:.2%}")
```

### 4. Actualización de Glosario

```python
# Identificar términos para agregar al glosario
pipeline = run_complete_analysis('papers.json', 'output')

novel_explanations = evaluator.explain_novel_terms(
    matches['novel_terms'],
    abstracts
)

# Filtrar por alta relevancia
candidates = {
    term: info
    for term, info in novel_explanations.items()
    if info['relevance_score'] > 10
}
```

---

## 🔄 Workflow Recomendado

```
1. Preparar Datos
   ├─ Ejecutar buscador académico
   └─ Generar unified_abstracts.json

2. Análisis Inicial
   ├─ Ejecutar pipeline completo
   └─ Revisar reporte maestro

3. Evaluación
   ├─ Analizar métricas
   ├─ Identificar mejor método
   └─ Revisar términos nuevos

4. Ajuste (si es necesario)
   ├─ Modificar parámetros
   ├─ Actualizar términos predefinidos
   └─ Re-ejecutar pipeline

5. Iteración
   └─ Repetir hasta obtener resultados satisfactorios
```

---

## 🌟 Mejores Prácticas

### 1. Calidad de Datos

```python
# Verificar abstracts antes de analizar
pipeline.load_data()
short_abstracts = [a for a in pipeline.abstracts if len(a) < 100]
print(f"⚠️ {len(short_abstracts)} abstracts cortos")
```

### 2. Términos Predefinidos

- Usar términos de glosarios reconocidos
- Incluir variaciones (singular/plural, abreviaturas)
- Mantener granularidad consistente
- Actualizar basándose en términos nuevos descubiertos

### 3. Análisis Iterativo

```python
# Primera iteración
pipeline1 = run_complete_analysis('data.json', 'iter1')

# Ajustar basándose en resultados
# Segunda iteración
pipeline2 = run_complete_analysis('data.json', 'iter2')

# Comparar mejoras
```

### 4. Documentación

- Guardar parámetros usados
- Documentar decisiones de diseño
- Mantener log de cambios en términos predefinidos

---

## 📊 Performance

### Tiempo de Ejecución Estimado

| Tamaño Corpus | Tiempo Total |
|---------------|--------------|
| < 50 papers | 30-60 seg |
| 50-100 papers | 1-2 min |
| 100-500 papers | 2-5 min |
| > 500 papers | 5-15 min |

*Tiempos en CPU moderna. GPU acelera significativamente SBERT.*

### Uso de Memoria

| Tamaño Corpus | Memoria |
|---------------|---------|
| < 100 papers | ~500 MB |
| 100-500 papers | ~1-2 GB |
| > 500 papers | ~2-4 GB |

---

## 🤝 Contribuciones

### Estructura del Código

```
ProyectoAlgoritmos/
├── predefined_term_analyzer.py       # Parte 1
├── auto_term_extractor.py            # Parte 2
├── term_precision_evaluator.py       # Parte 3
├── term_analysis_pipeline.py         # Parte 4
├── example_*.py                      # Ejemplos
├── test_*.py                         # Tests
├── requirements.txt                  # Dependencias
└── *.md                              # Documentación
```

### Agregar Nuevas Funcionalidades

1. Crear nueva rama
2. Implementar funcionalidad
3. Agregar tests
4. Actualizar documentación
5. Hacer pull request

---

## 📝 Notas de Versión

### Versión 1.0 (Actual)

**Implementado**:
- ✅ Parte 1: Análisis de términos predefinidos
- ✅ Parte 2: Extracción automática (RAKE + TextRank)
- ✅ Parte 3: Evaluación con similitud semántica
- ✅ Parte 4: Pipeline completo integrado
- ✅ Visualizaciones comparativas
- ✅ Reportes consolidados
- ✅ Tests de integración

**Pendiente para versiones futuras**:
- Análisis temporal de términos
- Clustering de términos similares
- Exportación a bases de datos
- Interfaz gráfica (GUI)
- API REST

---

## 📧 Soporte y Contacto

Para preguntas o problemas:

1. **Revisar documentación**: Cada parte tiene su README detallado
2. **Ejecutar ejemplos**: `example_complete_pipeline.py`
3. **Verificar tests**: `pytest test_pipeline_integration.py -v`
4. **Revisar logs**: Los mensajes de consola son detallados

---

## 📜 Licencia

[Especificar licencia del proyecto]

---

## 🎓 Referencias

### Papers y Recursos

- **RAKE**: Rose, S., et al. "Automatic keyword extraction from individual documents"
- **TextRank**: Mihalcea, R., & Tarau, P. "TextRank: Bringing order into texts"
- **SBERT**: Reimers, N., & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

### Bibliotecas Utilizadas

- **NLTK**: Natural Language Toolkit
- **spaCy**: Industrial-strength NLP
- **Sentence-Transformers**: State-of-the-art sentence embeddings
- **scikit-learn**: Machine learning tools
- **matplotlib/seaborn**: Visualización de datos

---

## ✅ Checklist de Implementación Completa

### Código Core
- ✅ Parte 1: PredefinedTermAnalyzer
- ✅ Parte 2: AutoTermExtractor
- ✅ Parte 3: TermPrecisionEvaluator
- ✅ Parte 4: TermAnalysisPipeline

### Funcionalidades
- ✅ Análisis de frecuencias
- ✅ Co-ocurrencias
- ✅ RAKE extraction
- ✅ TextRank extraction
- ✅ Similitud semántica (SBERT)
- ✅ Métricas P/R/F1
- ✅ Análisis de términos nuevos
- ✅ Pipeline integrado

### Visualizaciones
- ✅ Gráficos de frecuencias
- ✅ Matrices de co-ocurrencia
- ✅ Comparación de métodos
- ✅ Diagramas de Venn
- ✅ Heatmaps de similitud

### Reportes
- ✅ Reportes por componente
- ✅ Reporte maestro consolidado
- ✅ Formato Markdown
- ✅ Visualizaciones embebidas

### Tests
- ✅ Tests unitarios (Parte 3)
- ✅ Tests de integración (Pipeline)
- ✅ Casos edge
- ✅ Validación end-to-end

### Documentación
- ✅ READMEs por componente
- ✅ Documentación técnica detallada
- ✅ Ejemplos ejecutables
- ✅ README general del proyecto

### Extras
- ✅ requirements.txt completo
- ✅ Ejemplos interactivos
- ✅ Troubleshooting guide
- ✅ Mejores prácticas

---

**Fecha**: 2025
**Versión**: 1.0
**Estado**: ✅ **PROYECTO COMPLETO**

---

*Sistema de Análisis de Términos en Literatura Académica - Proyecto completo e integrado*
