# REQUERIMIENTO 5 - PARTE 5: Dashboard Interactivo con Streamlit

## Resumen Ejecutivo

Se ha implementado exitosamente el **Dashboard Interactivo** utilizando **Streamlit**, completando la Parte 5 del Requerimiento 5. Este dashboard proporciona una interfaz web interactiva que integra todas las visualizaciones bibliométricas desarrolladas en las partes anteriores (Geographic Heatmap, Dynamic Word Cloud, Timeline Visualization) y permite exportar reportes profesionales en PDF.

## Implementación Completada

### 1. Archivo Principal
- **Archivo**: `src/visualization/streamlit_dashboard.py` (~750 líneas)
- **Clase principal**: `VisualizationDashboard`
- **Tipo**: Dashboard web interactivo

### 2. Características Implementadas

#### A. Clase `VisualizationDashboard`

**Inicialización**:
```python
def __init__(self, data_path: str):
    """
    Inicializa el dashboard con datos.

    - Carga archivo CSV de publicaciones
    - Inicializa objetos de visualización (GeographicHeatmap, DynamicWordCloud, TimelineVisualization)
    - Configura caché de Streamlit para mejor rendimiento
    """
```

**Características**:
- Carga y validación automática de datos
- Manejo de errores robusto
- Inicialización de todos los módulos de visualización
- Sistema de caché para mejorar rendimiento

#### B. Sistema de Navegación

**Método `create_sidebar()`**:
```python
def create_sidebar(self):
    """
    Crea barra lateral con controles y filtros.

    Elementos:
    - Selector de página (5 páginas disponibles)
    - Filtros interactivos:
      * Rango de años (slider)
      * Venues/conferencias (multiselect)
      * Tipo de publicación (multiselect)
    - Botón de actualización de datos
    """
```

**Páginas disponibles**:
1. 📈 Overview - Vista general con KPIs
2. 🌍 Geographic - Análisis geográfico
3. ☁️ Word Cloud - Análisis de términos
4. 📅 Timeline - Evolución temporal
5. 📄 Export PDF - Generación de reportes

#### C. Páginas del Dashboard

##### 1. Overview Page (`show_overview_page()`)

**Métricas KPI** (4 columnas):
- 📚 **Total Publications**: Total de publicaciones con tasa de crecimiento promedio
- 🌍 **Years Covered**: Años cubiertos con rango temporal
- 📖 **Unique Venues**: Número de venues/conferencias únicas
- 👥 **Unique Authors**: Número de autores únicos

**Visualizaciones**:
1. **Gráfico de Pastel**: Distribución por tipo de publicación (journal/conference)
2. **Gráfico de Barras Horizontal**: Top 10 venues más productivos
3. **Gráfico de Líneas**: Tendencia de publicaciones por año

**Características**:
- Actualización automática basada en filtros
- Diseño responsivo con columnas
- Métricas con deltas e indicadores de cambio

##### 2. Geographic Page (`show_geographic_page()`)

**Análisis geográfico completo**:
```python
def show_geographic_page(self):
    """
    Página de análisis geográfico.

    Componentes:
    - Mapa mundial interactivo (Plotly scatter_geo)
    - Tamaño de puntos proporcional a publicaciones
    - Escala de colores para densidad
    - Tabla de top 10 países
    - Gráfico de barras por continente
    """
```

**Visualizaciones**:
1. **Mapa Global Interactivo**:
   - Scatter geo con Plotly
   - Puntos escalados por número de publicaciones
   - Hover data con información detallada
   - Escala de colores RdYlBu_r

2. **Top 10 Países**:
   - Tabla con ranking, país, publicaciones, porcentaje
   - Datos filtrados en tiempo real

3. **Distribución por Continente**:
   - Gráfico de barras interactivo
   - Agrupación automática por continente

##### 3. Word Cloud Page (`show_wordcloud_page()`)

**Análisis de términos**:
```python
def show_wordcloud_page(self):
    """
    Página de análisis de términos.

    Controles:
    - Slider: Número de términos a mostrar (20-200)
    - Selector: Método de ponderación (tfidf, log_frequency, frequency, normalized)

    Visualizaciones:
    - Gráfico de barras de top 20 términos
    - Tabla de 50 términos principales
    - Búsqueda de términos específicos
    """
```

**Características**:
1. **Control de parámetros**:
   - Slider para max_terms (20-200)
   - Selectbox para método de ponderación

2. **Gráfico de Barras Horizontal**:
   - Top 20 términos con ponderación
   - Ordenado por valor descendente
   - Alturas ajustables

3. **Tabla de Términos**:
   - Top 50 términos con frecuencia y peso
   - Formato con 4 decimales para pesos
   - Scrollable para fácil navegación

4. **Búsqueda de Términos**:
   - Input de texto para buscar términos específicos
   - Búsqueda case-insensitive
   - Muestra frecuencia y peso del término encontrado

##### 4. Timeline Page (`show_timeline_page()`)

**Evolución temporal**:
```python
def show_timeline_page(self):
    """
    Página de evolución temporal.

    Controles:
    - Selector de tipo de gráfico (Line/Area/Bar)

    Visualizaciones:
    - Gráfico principal interactivo
    - Estadísticas en 3 columnas
    - Tabla de datos año por año
    - Proyección futura si disponible
    """
```

**Características**:
1. **Gráfico Principal**:
   - Tres tipos: Línea, Área, Barras
   - Interactivo con Plotly
   - Zoom, pan, hover

2. **Estadísticas** (3 columnas):
   - Primer/último año
   - Total de publicaciones y promedio por año
   - Año más productivo
   - Tasa de crecimiento promedio

3. **Tabla de Datos Anuales**:
   - Año por año con todas las métricas
   - Botón de descarga CSV
   - Formato limpio y legible

4. **Proyección Futura**:
   - Basada en regresión lineal
   - Muestra tendencia (slope)
   - R² score para confiabilidad
   - Predicciones para 3 años futuros

##### 5. Export PDF Page (`show_export_page()`)

**Generación de reportes profesionales**:
```python
def show_export_page(self):
    """
    Página de exportación a PDF.

    Configuración:
    - Título del reporte
    - Subtítulo
    - Autor/analista
    - Institución

    Secciones:
    - Geographic Analysis (checkbox)
    - Word Cloud Analysis (checkbox)
    - Timeline Analysis (checkbox)

    Generación:
    - Botón de generación
    - Spinner de progreso
    - Botón de descarga del PDF
    """
```

**Proceso de generación**:
1. **Configuración del Reporte**:
   - Campos de texto para metadatos
   - Checkboxes para seleccionar secciones

2. **Generación de Visualizaciones**:
   - Crea imágenes estáticas de alta calidad (300 DPI)
   - Extrae estadísticas de cada módulo
   - Maneja errores por sección

3. **Compilación del PDF**:
   - Usa PDFExporter para crear reporte profesional
   - Incluye todas las secciones seleccionadas
   - Formato A4 con márgenes adecuados

4. **Descarga**:
   - Botón de descarga automático
   - Archivo PDF completo y profesional

#### D. Método Principal

**`run_dashboard()`**:
```python
def run_dashboard(self):
    """
    Método principal para ejecutar el dashboard.

    Configuración de página:
    - Título: "Bibliometric Analysis Dashboard"
    - Icono: 📊
    - Layout: wide (ancho completo)
    - Sidebar: expandido por defecto

    Routing:
    - Crea sidebar y obtiene página seleccionada
    - Enruta a la página correspondiente
    """
```

**Características**:
- Configuración de página con `st.set_page_config()`
- CSS personalizado para métricas
- Sistema de routing basado en selección
- Layout optimizado para visualizaciones

### 3. Sistema de Filtros

**Filtros implementados**:
1. **Year Range** (Slider):
   - Rango dinámico basado en datos
   - Filtra todas las visualizaciones
   - Almacenado en `st.session_state['year_range']`

2. **Venues** (Multiselect):
   - Top 10 venues más comunes
   - Opción "All" para mostrar todos
   - Almacenado en `st.session_state['selected_venues']`

3. **Publication Type** (Multiselect):
   - Journal, Conference, etc.
   - Opción "All" para mostrar todos
   - Almacenado en `st.session_state['selected_types']`

**Aplicación de filtros**:
```python
def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros seleccionados al DataFrame.

    - Lee filtros de session_state
    - Filtra por año si no es "All"
    - Filtra por venue si no es "All"
    - Filtra por tipo si no es "All"
    - Retorna DataFrame filtrado
    """
```

### 4. Optimizaciones de Rendimiento

**Caché de datos**:
```python
@st.cache_data
def _load_data(_self):
    """Carga y cachea datos para evitar lecturas repetidas."""
    return pd.read_csv(_self.data_path, encoding='utf-8')
```

**Ventajas**:
- Reduce tiempo de carga
- Evita lecturas repetidas de archivos
- Mejora experiencia del usuario
- Actualizable con botón "Refresh Data"

### 5. Manejo de Errores

**Validaciones**:
- Verificación de existencia de archivo de datos
- Try-catch en cada sección de visualización
- Mensajes de error informativos con `st.error()`
- Warnings para secciones que no se pueden generar
- Fallback graceful si faltan dependencias

**Ejemplo**:
```python
try:
    affiliations = self.geo_map.extract_author_affiliations()
    geo_data = self.geo_map.geocode_locations()
    # ... visualización
except Exception as e:
    st.error(f"Error in geographic analysis: {e}")
```

## Integración con Módulos Existentes

### 1. GeographicHeatmap
- Extracción de afiliaciones
- Geocodificación
- Generación de mapas estáticos para PDF
- Estadísticas geográficas

### 2. DynamicWordCloud
- Extracción de términos
- Cálculo de ponderaciones
- Generación de word clouds estáticos para PDF
- Estadísticas de términos

### 3. TimelineVisualization
- Extracción de datos temporales
- Cálculo de estadísticas anuales
- Generación de gráficos de línea de tiempo
- Proyecciones futuras

### 4. PDFExporter
- Generación de reportes completos
- Inclusión de todas las visualizaciones
- Formato profesional A4
- Optimización de imágenes a 300 DPI

## Dependencias Actualizadas

### requirements.txt
Se agregó:
```txt
streamlit>=1.28.0             # Dashboard interactivo web
```

**Dependencias existentes utilizadas**:
- pandas: Manipulación de datos
- numpy: Operaciones numéricas
- plotly: Visualizaciones interactivas
- matplotlib: Visualizaciones estáticas (para PDF)
- loguru: Logging

## Uso del Dashboard

### 1. Ejecución Básica

```bash
# Desde el directorio raíz del proyecto
streamlit run src/visualization/streamlit_dashboard.py
```

### 2. Con Datos Personalizados

**Modificar la ruta en el código**:
```python
# En streamlit_dashboard.py, línea 736
default_path = Path('ruta/a/tus/datos.csv')
```

### 3. Estructura de Datos Requerida

El archivo CSV debe contener las siguientes columnas:
- `id`: Identificador único
- `title`: Título de la publicación
- `authors`: Autores (con afiliaciones entre paréntesis)
- `year`: Año de publicación
- `abstract`: Resumen
- `keywords`: Palabras clave
- `journal_conference`: Nombre del venue
- `publication_type`: journal o conference

**Ejemplo**:
```csv
id,title,authors,year,abstract,keywords,journal_conference,publication_type
pub_0001,Research on AI,John Doe (MIT USA),2023,This study...,AI; ML,IEEE Trans,journal
```

## Características Destacadas

### 1. Interfaz Intuitiva
- ✅ Navegación clara con iconos
- ✅ Diseño responsivo
- ✅ Métricas visuales atractivas
- ✅ Gráficos interactivos

### 2. Interactividad
- ✅ Filtros en tiempo real
- ✅ Zoom y pan en gráficos
- ✅ Hover para información detallada
- ✅ Búsqueda de términos

### 3. Exportación
- ✅ Generación de PDF profesional
- ✅ Descarga directa desde navegador
- ✅ Imágenes de alta calidad (300 DPI)
- ✅ Secciones configurables

### 4. Rendimiento
- ✅ Caché de datos
- ✅ Carga rápida
- ✅ Actualización selectiva
- ✅ Manejo eficiente de errores

### 5. Análisis Completo
- ✅ Vista general con KPIs
- ✅ Análisis geográfico detallado
- ✅ Análisis de términos frecuentes
- ✅ Evolución temporal
- ✅ Proyecciones futuras

## Casos de Uso

### Caso 1: Exploración Rápida
1. Ejecutar dashboard
2. Ver página Overview para métricas generales
3. Explorar diferentes períodos con slider de años
4. Identificar tendencias principales

### Caso 2: Análisis Geográfico
1. Ir a página Geographic
2. Explorar mapa interactivo
3. Identificar países líderes
4. Analizar distribución por continente

### Caso 3: Análisis de Términos
1. Ir a página Word Cloud
2. Ajustar número de términos
3. Cambiar método de ponderación
4. Buscar términos específicos
5. Exportar datos de términos

### Caso 4: Análisis Temporal
1. Ir a página Timeline
2. Cambiar tipo de gráfico (Line/Area/Bar)
3. Revisar estadísticas anuales
4. Ver proyecciones futuras
5. Descargar datos en CSV

### Caso 5: Generación de Reporte
1. Ir a página Export PDF
2. Configurar título, autor, institución
3. Seleccionar secciones a incluir
4. Generar PDF
5. Descargar reporte completo

## Ventajas del Dashboard

### 1. Accesibilidad
- No requiere conocimientos de programación
- Interfaz web familiar
- Accesible desde cualquier navegador
- Compartible vía URL (si se despliega en servidor)

### 2. Flexibilidad
- Filtros ajustables
- Visualizaciones personalizables
- Secciones modulares
- Exportación configurable

### 3. Integración
- Usa todos los módulos desarrollados
- Aprovecha funcionalidad completa
- Consistencia en análisis
- Flujo de trabajo integrado

### 4. Profesionalismo
- Diseño limpio y moderno
- Visualizaciones de alta calidad
- Reportes profesionales
- Métricas bien presentadas

## Ejemplo de Flujo de Trabajo

```
1. Inicio del Dashboard
   └─> Carga automática de datos
   └─> Inicialización de módulos

2. Exploración Inicial (Overview)
   └─> Ver KPIs generales
   └─> Identificar períodos de interés
   └─> Ajustar filtros

3. Análisis Detallado
   ├─> Geographic: Identificar países líderes
   ├─> Word Cloud: Encontrar términos principales
   └─> Timeline: Analizar tendencias

4. Generación de Reporte
   └─> Configurar metadatos
   └─> Seleccionar secciones
   └─> Generar y descargar PDF

5. Compartir Resultados
   └─> PDF profesional listo para presentación
```

## Mejoras Futuras Posibles

### 1. Funcionalidades Adicionales
- [ ] Análisis de co-autoría (redes)
- [ ] Comparación de múltiples datasets
- [ ] Exportación a otros formatos (Excel, PowerPoint)
- [ ] Análisis de citaciones

### 2. Visualizaciones Adicionales
- [ ] Gráficos de red de colaboración
- [ ] Mapas de calor de correlación
- [ ] Análisis de sentimiento en abstracts
- [ ] Clustering de publicaciones

### 3. Interactividad Mejorada
- [ ] Anotaciones personalizadas
- [ ] Guardado de configuraciones
- [ ] Historial de consultas
- [ ] Comparación lado a lado

### 4. Despliegue
- [ ] Dockerización
- [ ] Despliegue en Streamlit Cloud
- [ ] Autenticación de usuarios
- [ ] Base de datos para persistencia

## Resumen de Archivos

### Archivos Creados
1. **src/visualization/streamlit_dashboard.py** (~750 líneas)
   - Clase VisualizationDashboard
   - 8 métodos principales + helpers
   - Manejo completo de errores
   - Sistema de caché

### Archivos Modificados
1. **requirements.txt**
   - Agregado: `streamlit>=1.28.0`

2. **src/visualization/pdf_exporter.py**
   - Corregido typo: `self.colors.white night` → `self.colors.whitesmoke`

## Conclusión

El **Dashboard Interactivo con Streamlit** completa exitosamente la Parte 5 del Requerimiento 5, proporcionando:

✅ **Interfaz web completa** con 5 páginas especializadas
✅ **Sistema de filtros** en tiempo real
✅ **Visualizaciones interactivas** con Plotly
✅ **Integración perfecta** con todos los módulos anteriores
✅ **Exportación a PDF** profesional
✅ **Rendimiento optimizado** con caché
✅ **Manejo robusto de errores**
✅ **Diseño profesional** y responsive

El dashboard es **listo para producción** y puede usarse inmediatamente para análisis bibliométricos interactivos y generación de reportes profesionales.

---

**Total de Implementación**:
- 750 líneas de código Python
- 8 métodos principales
- 5 páginas interactivas
- 3 tipos de filtros
- 15+ visualizaciones diferentes
- Integración completa con 4 módulos

**Estado**: ✅ COMPLETADO

**Fecha**: 2024
