"""
Interfaz Web Local para el Sistema de Análisis Bibliométrico
Usa Streamlit para crear una interfaz con botones que ejecutan las opciones del main.py
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
import time
import json

# Configurar la página
st.set_page_config(
    page_title="Análisis Bibliométrico - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        margin: 5px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 ANÁLISIS BIBLIOMÉTRICO - DASHBOARD</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar con información
with st.sidebar:
    st.header("📈 Estadísticas")
    
    # Intentar leer artículos
    try:
        data_file = Path(__file__).parent.parent / "data" / "unified_articles.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            st.metric("Total de Artículos", len(articles))
            
            # Estadísticas por fuente
            sources = {}
            for art in articles:
                source = art.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            st.write("**Por Fuente:**")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {source}: {count}")
        else:
            st.info("No hay datos descargados aún")
    except Exception as e:
        st.warning("No se pudieron cargar las estadísticas")
    


# Función para ejecutar comandos del main.py
def run_main_option(option_number):
    """
    Ejecuta una opción del main.py de forma programática
    """
    try:
        # Cambiar al directorio bibliometric-analysis
        main_dir = Path(__file__).parent / "bibliometric-analysis"
        
        # Mapear número de opción a modo
        mode_map = {
            1: "scrape",
            2: "deduplicate",
            3: "preprocess",
            4: "cluster",
            5: "visualize",
            6: "report",
            7: "full"
        }
        
        if option_number not in mode_map:
            return None, "Opción no válida"
        
        # Construir el comando usando powershell para Windows
        if sys.platform == 'win32':
            # Usar powershell para ejecutar correctamente
            cmd = f'cd "{main_dir}" ; python main.py --mode {mode_map[option_number]}'
            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutos timeout
                encoding='utf-8',
                errors='replace'
            )
        else:
            cmd = ["python3", "main.py", "--mode", mode_map[option_number]]
            result = subprocess.run(
                cmd,
                cwd=str(main_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
        
        return result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return None, "⚠️ Timeout: La operación tomó más de 5 minutos"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def run_scraper():
    """
    Ejecuta el scraper completo (UniQuindío: IEEE, ScienceDirect, Springer)
    """
    try:
        scraper_path = Path(__file__).parent / "scrapers" / "scraper_uniquindio_completo.py"
        
        result = subprocess.Popen(
            ["python", str(scraper_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        return True, "✅ Scraper ejecutándose en nueva ventana"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# Tabs principales
tab1, tab2, tab3 = st.tabs(["🚀 Operaciones Principales", "📊 Visualización de Datos", "⚙️ Configuración"])

with tab1:
    st.header("Operaciones del Pipeline")
    st.write("Haz clic en los botones para ejecutar cada operación del sistema")
    
    # Layout en 3 columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1️⃣ Descarga de Datos")
        if st.button("🔍 SCRAPE - Descargar Artículos", key="btn_scrape", help="Descarga artículos de IEEE, ScienceDirect y Springer"):
            with st.spinner("Ejecutando scraper..."):
                success, message = run_scraper()
                if success:
                    st.success(message)
                    st.info("💡 El scraper se está ejecutando en una ventana separada. Esto puede tomar 5-10 minutos.")
                else:
                    st.error(message)
        
        st.markdown("---")
        
        st.subheader("2️⃣ Limpieza")
        if st.button("🔄 DEDUPLICATE - Eliminar Duplicados", key="btn_dedup", help="Detecta y elimina artículos duplicados"):
            with st.spinner("Ejecutando deduplicación..."):
                stdout, stderr = run_main_option(2)
                if stderr and "Error" in stderr:
                    st.error(f"Error: {stderr}")
                else:
                    st.success("✅ Deduplicación completada")
                    if stdout:
                        with st.expander("Ver detalles"):
                            st.code(stdout)
        
        st.markdown("---")
        
        st.subheader("3️⃣ Preprocesamiento")
        if st.button("🧹 PREPROCESS - Limpiar Datos", key="btn_preprocess", help="Limpia y normaliza los datos"):
            with st.spinner("Ejecutando preprocesamiento..."):
                stdout, stderr = run_main_option(3)
                if stderr and "Error" in stderr:
                    st.error(f"Error: {stderr}")
                else:
                    st.success("✅ Preprocesamiento completado")
                    if stdout:
                        with st.expander("Ver detalles"):
                            st.code(stdout)
    
    with col2:
        st.subheader("4️⃣ Clustering")
        if st.button("📈 CLUSTER - Análisis Temático", key="btn_cluster", help="Agrupa artículos por similitud temática"):
            with st.spinner("Ejecutando clustering..."):
                stdout, stderr = run_main_option(4)
                if stderr and "Error" in stderr:
                    st.error(f"Error: {stderr}")
                else:
                    st.success("✅ Clustering completado")
                    if stdout:
                        with st.expander("Ver detalles"):
                            st.code(stdout)
        
        st.markdown("---")
        
        st.subheader("5️⃣ Visualización")
        if st.button("📊 VISUALIZE - Generar Gráficos", key="btn_visualize", help="Genera visualizaciones y gráficos"):
            with st.spinner("Generando visualizaciones..."):
                stdout, stderr = run_main_option(5)
                if stderr and "Error" in stderr:
                    st.error(f"Error: {stderr}")
                else:
                    st.success("✅ Visualizaciones generadas")
                    st.info("💡 Revisa la carpeta output/visualizations/")
                    if stdout:
                        with st.expander("Ver detalles"):
                            st.code(stdout)
        
        st.markdown("---")
        
        st.subheader("6️⃣ Reportes")
        if st.button("📄 REPORT - Generar Reporte PDF", key="btn_report", help="Crea un reporte PDF completo"):
            with st.spinner("Generando reporte..."):
                stdout, stderr = run_main_option(6)
                if stderr and "Error" in stderr:
                    st.error(f"Error: {stderr}")
                else:
                    st.success("✅ Reporte generado")
                    st.info("💡 Revisa la carpeta output/reports/")
                    if stdout:
                        with st.expander("Ver detalles"):
                            st.code(stdout)
    
    with col3:
        st.subheader("7️⃣ Pipeline Completo")
        if st.button("🚀 FULL - Ejecutar Todo", key="btn_full", help="Ejecuta el pipeline completo de principio a fin"):
            with st.spinner("Ejecutando pipeline completo... Esto puede tomar varios minutos"):
                stdout, stderr = run_main_option(7)
                if stderr and "Error" in stderr:
                    st.error(f"Error: {stderr}")
                else:
                    st.success("✅ Pipeline completo ejecutado")
                    if stdout:
                        with st.expander("Ver detalles"):
                            st.code(stdout)
        


with tab2:
    st.header("📊 Visualización de Datos Actuales")
    
    try:
        data_file = Path(__file__).parent / "data" / "unified_articles.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            st.success(f"✅ {len(articles)} artículos cargados")
            
            # Gráfico de distribución por fuente
            st.subheader("Distribución por Fuente")
            sources = {}
            for art in articles:
                source = art.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            st.bar_chart(sources)
            
            # Gráfico de distribución por año
            st.subheader("Distribución por Año")
            years = {}
            for art in articles:
                year = art.get('year', 'unknown')
                if year and year != 'unknown':
                    years[str(year)] = years.get(str(year), 0) + 1
            
            st.line_chart(years)
            
            # Tabla de primeros 10 artículos
            st.subheader("Primeros 10 Artículos")
            for i, art in enumerate(articles[:10], 1):
                with st.expander(f"{i}. {art.get('title', 'Sin título')[:80]}..."):
                    st.write(f"**Autores:** {', '.join(art.get('authors', ['N/A'])[:3])}")
                    st.write(f"**Año:** {art.get('year', 'N/A')}")
                    st.write(f"**Fuente:** {art.get('source', 'N/A')}")
                    st.write(f"**DOI:** {art.get('doi', 'N/A')}")
                    if art.get('abstract'):
                        st.write(f"**Abstract:** {art.get('abstract')[:200]}...")
        else:
            st.info("📂 No hay datos disponibles. Ejecuta el scraper primero.")
            
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")

with tab3:
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("📁 Rutas del Proyecto")
    project_root = Path(__file__).parent
    st.text(f"Raíz del proyecto: {project_root}")
    st.text(f"Datos: {project_root / 'data'}")
    st.text(f"Salidas: {project_root / 'output'}")
    st.text(f"Scrapers: {project_root / 'scrapers'}")
    
    st.markdown("---")
    
    st.subheader("🔧 Acceso Rápido")
    
    if st.button("📂 Abrir carpeta de datos"):
        data_folder = project_root / "data"
        if sys.platform == 'win32':
            os.startfile(str(data_folder))
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(data_folder)])
    
    if st.button("📂 Abrir carpeta de outputs"):
        output_folder = project_root / "output"
        if sys.platform == 'win32':
            os.startfile(str(output_folder))
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(output_folder)])

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Sistema de Análisis Bibliométrico - Universidad del Quindío</p>
    </div>
""", unsafe_allow_html=True)
