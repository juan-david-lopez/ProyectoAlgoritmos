"""
Script de demostración básico (sin modelos de IA) para pruebas rápidas.

Este script ejecuta solo los 4 algoritmos básicos (Levenshtein, TF-IDF, Jaccard, N-gram)
para una demostración rápida sin necesidad de descargar modelos grandes.
"""

import sys
import time
import logging
from pathlib import Path
import json
import numpy as np

# Configurar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.levenshtein import LevenshteinComparator
from src.algorithms.tfidf_cosine import TFIDFCosineComparator
from src.algorithms.jaccard import JaccardComparator
from src.algorithms.ngram import NGramComparator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('similarity_demo_basic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta demostración básica."""
    logger.info("=" * 80)
    logger.info("DEMOSTRACIÓN BÁSICA DE ALGORITMOS DE SIMILITUD")
    logger.info("=" * 80)

    start_time_total = time.perf_counter()

    # ========================================================================
    # PASO 1: CARGAR DATOS
    # ========================================================================
    logger.info("\nPASO 1: Cargando datos...")

    data_path = 'data/unified_articles.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    logger.info(f"✓ {len(articles)} artículos cargados")

    # ========================================================================
    # PASO 2: SELECCIONAR ARTÍCULOS
    # ========================================================================
    logger.info("\nPASO 2: Seleccionando 3 artículos...")

    selected = articles[:3]
    abstracts = [art['abstract'] for art in selected]

    for idx, art in enumerate(selected, 1):
        logger.info(f"\nArtículo {idx}: {art['title']}")
        logger.info(f"  ID: {art['id']}")
        logger.info(f"  Abstract: {art['abstract'][:100]}...")
        logger.info(f"  Longitud: {len(art['abstract'])} caracteres")

    # ========================================================================
    # PASO 3: COMPARAR CON ALGORITMOS BÁSICOS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PASO 3: Comparando con algoritmos básicos")
    logger.info("=" * 80)

    algorithms = {
        'Levenshtein': LevenshteinComparator(),
        'TF-IDF + Coseno': TFIDFCosineComparator(),
        'Jaccard': JaccardComparator(),
        'N-grama (n=3)': NGramComparator(n=3, method='dice')
    }

    results = {}

    for name, comparator in algorithms.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Ejecutando: {name}")
        logger.info(f"{'='*60}")

        start_time = time.perf_counter()
        matrix = comparator.compare_multiple(abstracts)
        elapsed = time.perf_counter() - start_time

        results[name] = {
            'matrix': matrix,
            'time': elapsed
        }

        logger.info(f"✓ Completado en {elapsed:.4f}s")

        # Calcular estadísticas
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        sim_values = matrix[mask]

        logger.info(f"\nEstadísticas:")
        logger.info(f"  Similitud media: {np.mean(sim_values):.4f}")
        logger.info(f"  Similitud máxima: {np.max(sim_values):.4f}")
        logger.info(f"  Similitud mínima: {np.min(sim_values):.4f}")

        # Verificar rango [0, 1]
        if np.min(matrix) < 0 or np.max(matrix) > 1:
            logger.warning(f"  ⚠️ ADVERTENCIA: Valores fuera del rango [0,1]!")
        else:
            logger.info(f"  ✓ Valores en rango correcto [0,1]")

        # Mostrar matriz
        logger.info(f"\nMatriz de similitud:")
        for i in range(len(abstracts)):
            row_str = "  "
            for j in range(len(abstracts)):
                row_str += f"{matrix[i][j]:.3f}  "
            logger.info(row_str)

    # ========================================================================
    # PASO 4: ANÁLISIS COMPARATIVO
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS COMPARATIVO")
    logger.info("=" * 80)

    # Comparar tiempos
    logger.info("\nTiempos de ejecución:")
    for name, data in results.items():
        logger.info(f"  {name:20s}: {data['time']:.4f}s")

    fastest = min(results.items(), key=lambda x: x[1]['time'])
    logger.info(f"\n🏆 Algoritmo más rápido: {fastest[0]} ({fastest[1]['time']:.4f}s)")

    # Comparar similitudes promedio
    logger.info("\nSimilitudes promedio entre artículos:")
    for name, data in results.items():
        matrix = data['matrix']
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        avg_sim = np.mean(matrix[mask])
        logger.info(f"  {name:20s}: {avg_sim:.4f}")

    # ========================================================================
    # PASO 5: ANÁLISIS DE PARES ESPECÍFICOS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE PARES ESPECÍFICOS")
    logger.info("=" * 80)

    logger.info("\nArtículo 1 vs Artículo 2:")
    logger.info("(Ambos sobre NLP/ML)")
    for name, data in results.items():
        sim = data['matrix'][0][1]
        logger.info(f"  {name:20s}: {sim:.4f}")

    logger.info("\nArtículo 1 vs Artículo 3:")
    logger.info("(NLP vs Computer Vision)")
    for name, data in results.items():
        sim = data['matrix'][0][2]
        logger.info(f"  {name:20s}: {sim:.4f}")

    # ========================================================================
    # EXPLICACIÓN: TF-IDF VS OTROS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("¿POR QUÉ DIFERENTES RESULTADOS?")
    logger.info("=" * 80)

    logger.info("\n📊 Análisis de diferencias:")
    logger.info("\n1. LEVENSHTEIN:")
    logger.info("   - Compara carácter por carácter")
    logger.info("   - Muy estricto, detecta mínimas diferencias")
    logger.info("   - Valor bajo esperado para textos largos diferentes")

    logger.info("\n2. TF-IDF + COSENO:")
    logger.info("   - Compara importancia de palabras")
    logger.info("   - Detecta vocabulario compartido")
    logger.info("   - Ignora palabras comunes (stopwords)")

    logger.info("\n3. JACCARD:")
    logger.info("   - Compara conjunto de palabras únicas")
    logger.info("   - No considera frecuencias")
    logger.info("   - Simple pero efectivo")

    logger.info("\n4. N-GRAMA:")
    logger.info("   - Compara subcadenas de caracteres")
    logger.info("   - Robusto a errores")
    logger.info("   - Captura similitud local")

    # ========================================================================
    # RECOMENDACIONES
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RECOMENDACIONES PARA ABSTRACTS CIENTÍFICOS")
    logger.info("=" * 80)

    logger.info("\n🎯 Para este tipo de textos:")
    logger.info("  1. TF-IDF: Excelente para documentos largos")
    logger.info("  2. N-grama: Bueno para detectar paráfrasis")
    logger.info("  3. Jaccard: Rápido pero menos preciso")
    logger.info("  4. Levenshtein: No recomendado para textos largos")

    logger.info("\n💡 Para mejor precisión semántica:")
    logger.info("  - Usar S-BERT o BERT (requiere modelos de IA)")
    logger.info("  - Estos capturan significado, no solo palabras")
    logger.info("  - Ver similarity_demo.py para versión completa")

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    elapsed_total = time.perf_counter() - start_time_total

    logger.info("\n" + "=" * 80)
    logger.info("DEMOSTRACIÓN COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"Tiempo total: {elapsed_total:.4f}s")
    logger.info(f"Artículos analizados: {len(selected)}")
    logger.info(f"Algoritmos ejecutados: {len(algorithms)}")
    logger.info(f"\nResultados guardados en: similarity_demo_basic.log")


if __name__ == "__main__":
    main()
