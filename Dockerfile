# ============================================================================
# DOCKERFILE OPTIMIZADO PARA RENDER
# Tamaño estimado: ~1 GB (dentro del límite de 2 GB)
# ============================================================================

# Stage 1: Builder - Instalar dependencias
FROM python:3.11-slim as builder

WORKDIR /build

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY bibliometric-analysis/requirements.txt ./bibliometric-requirements.txt
COPY requirements.txt ./project-requirements.txt

# Instalar todas las dependencias en el usuario
RUN pip install --no-cache-dir --user -r bibliometric-requirements.txt && \
    pip install --no-cache-dir --user -r project-requirements.txt

# Stage 2: Runtime - Imagen final optimizada
FROM python:3.11-slim

WORKDIR /app

# Instalar solo dependencias de runtime necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias Python desde builder
COPY --from=builder /root/.local /root/.local

# Copiar código fuente (sin data pesada ni outputs)
COPY bibliometric-analysis/ ./bibliometric-analysis/
COPY src/ ./src/
COPY scrapers/*.py ./scrapers/
COPY pipelines/ ./pipelines/
COPY utilities/ ./utilities/
COPY config/ ./config/
COPY dashboard.py .
COPY mostrar_datos.py .

# Copiar solo datos de muestra (no raw data grande)
COPY data/sample/ ./data/sample/
COPY data/unified_articles.json ./data/

# Crear directorios necesarios
RUN mkdir -p \
    data/raw \
    data/processed \
    data/duplicates \
    output \
    logs \
    scrapers/data/raw \
    scrapers/data/processed

# Variables de entorno
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app:/app/bibliometric-analysis \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Exponer puerto de Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Comando por defecto: ejecutar dashboard
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
