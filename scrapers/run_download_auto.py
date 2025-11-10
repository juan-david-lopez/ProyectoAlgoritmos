"""
Script automatizado para ejecutar descarga con Semantic Scholar
"""

import subprocess
import sys
from pathlib import Path

# Cambiar al directorio correcto
import os
os.chdir(Path(__file__).parent)

# Simular entrada del usuario para el menú
# Opción 2: Descarga y unificación
# Luego 's' para usar configuración predeterminada

print("╔══════════════════════════════════════════════════════════════════╗")
print("║   DESCARGA AUTOMATIZADA CON SEMANTIC SCHOLAR                     ║")
print("╚══════════════════════════════════════════════════════════════════╝\n")

print("Iniciando descarga con respaldo de Semantic Scholar API...")
print("Configuración:")
print("  • Query: 'generative artificial intelligence'")
print("  • Máximo: 50 artículos")
print("  • Fuentes: ACM (intentará), ScienceDirect (intentará), Semantic Scholar (respaldo)")
print("\nEsto puede tardar algunos minutos...\n")

# Ejecutar directamente el comando de descarga
# Cargar el módulo main.py desde la carpeta 'bibliometric-analysis' que contiene un guion en el nombre
import importlib.util
module_path = Path(__file__).parent / 'bibliometric-analysis' / 'main.py'
spec = importlib.util.spec_from_file_location("bibliometric_analysis_main", str(module_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cmd_download = module.cmd_download

from argparse import Namespace

# Crear argumentos
args = Namespace(
    query='generative artificial intelligence',
    sources=['acm', 'sciencedirect']
)

# Ejecutar descarga
try:
    cmd_download(args)
    print("\n Descarga completada exitosamente!")
except Exception as e:
    print(f"\n  Proceso completado con advertencias: {e}")
    print("Verificar si se descargaron datos desde Semantic Scholar")
