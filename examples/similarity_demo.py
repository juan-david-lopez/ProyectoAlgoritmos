"""
Script de demostración completo del sistema de comparación de similitud.

Este script demuestra el flujo completo:
1. Cargar datos unificados
2. Seleccionar 3-5 artículos de ejemplo
3. Comparar con todos los 6 algoritmos
4. Generar visualizaciones
5. Crear reporte detallado

Autor: Sistema de Análisis de Similitud
Fecha: 2025-10-27
"""

import sys
import time
import logging
from pathlib import Path

# Configurar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.similarity_comparator import SimilarityComparator


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('similarity_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Función principal que ejecuta la demostración completa.
    """
    logger.info("=" * 80)
    logger.info("INICIANDO DEMOSTRACIÓN DE COMPARACIÓN DE SIMILITUD")
    logger.info("=" * 80)

    # Medir tiempo total de ejecución
    start_time_total = time.perf_counter()

    try:
        # ========================================================================
        # PASO 1: CARGAR DATOS
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 1: CARGANDO DATOS")
        logger.info("=" * 80)

        data_path = 'data/unified_articles.json'
        logger.info(f"Ruta de datos: {data_path}")

        start_time = time.perf_counter()
        comparator = SimilarityComparator(data_path)
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ Datos cargados exitosamente en {elapsed:.3f} segundos")

        # ========================================================================
        # PASO 2: SELECCIONAR ARTÍCULOS
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 2: SELECCIONANDO ARTÍCULOS DE EJEMPLO")
        logger.info("=" * 80)

        # Opción 1: Seleccionar por IDs específicos (si conoces los IDs)
        # article_ids = ['article_1', 'article_2', 'article_3']

        # Opción 2: Seleccionar los primeros N artículos
        num_articles = 3
        article_ids = [article['id'] for article in comparator.unified_data[:num_articles]]

        logger.info(f"Artículos seleccionados: {article_ids}")

        start_time = time.perf_counter()
        selected_articles = comparator.select_articles(article_ids)
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ {len(selected_articles)} artículos seleccionados en {elapsed:.3f} segundos")

        # Mostrar información de artículos seleccionados
        for idx, article in enumerate(selected_articles, 1):
            logger.info(f"\n  Artículo {idx}:")
            logger.info(f"    ID: {article['id']}")
            logger.info(f"    Título: {article['title'][:80]}...")
            logger.info(f"    Abstract: {article['abstract'][:150]}...")
            logger.info(f"    Longitud: {len(article['abstract'])} caracteres")

        # Extraer abstracts
        abstracts = [article['abstract'] for article in selected_articles]

        # ========================================================================
        # PASO 3: COMPARAR CON TODOS LOS ALGORITMOS
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 3: COMPARANDO CON TODOS LOS ALGORITMOS")
        logger.info("=" * 80)

        start_time = time.perf_counter()
        results = comparator.compare_all_algorithms(abstracts)
        elapsed = time.perf_counter() - start_time

        logger.info(f"\n✓ Comparación completada en {elapsed:.3f} segundos")

        # Mostrar resumen de resultados
        logger.info("\n" + "-" * 80)
        logger.info("RESUMEN DE RESULTADOS:")
        logger.info("-" * 80)

        algorithms = ['levenshtein', 'tfidf_cosine', 'jaccard', 'ngram', 'sbert', 'bert']

        for algo_name in algorithms:
            exec_time = results['execution_times'].get(algo_name)
            mem_usage = results['memory_usage'].get(algo_name)
            sim_matrix = results['similarities'].get(algo_name)

            if exec_time is not None and sim_matrix is not None:
                # Calcular estadísticas
                import numpy as np
                mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
                sim_values = sim_matrix[mask]

                logger.info(f"\n{algo_name.upper()}:")
                logger.info(f"  Tiempo: {exec_time:.3f}s")
                logger.info(f"  Memoria: {mem_usage:.2f} MB")
                logger.info(f"  Similitud media: {np.mean(sim_values):.3f}")
                logger.info(f"  Similitud máxima: {np.max(sim_values):.3f}")
                logger.info(f"  Similitud mínima: {np.min(sim_values):.3f}")

                # Verificar que los valores estén en [0, 1]
                if np.min(sim_matrix) < 0 or np.max(sim_matrix) > 1:
                    logger.warning(f"  ⚠️ ADVERTENCIA: Valores fuera del rango [0,1]!")
                else:
                    logger.info(f"  ✓ Valores en rango correcto [0,1]")

        # ========================================================================
        # PASO 4: GENERAR VISUALIZACIONES
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 4: GENERANDO VISUALIZACIONES")
        logger.info("=" * 80)

        output_dir = 'output/visualizations'
        logger.info(f"Directorio de salida: {output_dir}")

        start_time = time.perf_counter()
        comparator.visualize_results(results, output_dir)
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ Visualizaciones generadas en {elapsed:.3f} segundos")
        logger.info(f"  - Heatmaps de similitud: {output_dir}/similarity_heatmaps.png")
        logger.info(f"  - Tiempos de ejecución: {output_dir}/execution_times.png")
        logger.info(f"  - Uso de memoria: {output_dir}/memory_usage.png")
        logger.info(f"  - Tabla comparativa: {output_dir}/comparison_table.png")

        # ========================================================================
        # PASO 5: CREAR REPORTE DETALLADO
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 5: GENERANDO REPORTE DETALLADO")
        logger.info("=" * 80)

        report_path = 'output/similarity_report.md'
        logger.info(f"Ruta del reporte: {report_path}")

        start_time = time.perf_counter()
        comparator.generate_detailed_report(results, report_path, selected_articles)
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ Reporte generado en {elapsed:.3f} segundos")

        # ========================================================================
        # RESUMEN FINAL
        # ========================================================================
        elapsed_total = time.perf_counter() - start_time_total

        logger.info("\n" + "=" * 80)
        logger.info("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"Tiempo total de ejecución: {elapsed_total:.3f} segundos")
        logger.info(f"Artículos analizados: {len(selected_articles)}")
        logger.info(f"Algoritmos ejecutados: {len([a for a in algorithms if results['execution_times'].get(a) is not None])}")
        logger.info(f"\nResultados guardados en:")
        logger.info(f"  - Visualizaciones: {output_dir}/")
        logger.info(f"  - Reporte: {report_path}")
        logger.info(f"  - Log: similarity_demo.log")

        # ========================================================================
        # ANÁLISIS COMPARATIVO
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("ANÁLISIS COMPARATIVO")
        logger.info("=" * 80)

        # Algoritmo más rápido
        fastest = min(
            [(name, time) for name, time in results['execution_times'].items() if time is not None],
            key=lambda x: x[1]
        )
        logger.info(f"\n🏆 Algoritmo más rápido: {fastest[0].upper()} ({fastest[1]:.3f}s)")

        # Algoritmo con menor memoria
        lowest_mem = min(
            [(name, mem) for name, mem in results['memory_usage'].items() if mem is not None],
            key=lambda x: x[1]
        )
        logger.info(f"💾 Menor uso de memoria: {lowest_mem[0].upper()} ({lowest_mem[1]:.2f} MB)")

        # Algoritmo más lento
        slowest = max(
            [(name, time) for name, time in results['execution_times'].items() if time is not None],
            key=lambda x: x[1]
        )
        logger.info(f"🐌 Algoritmo más lento: {slowest[0].upper()} ({slowest[1]:.3f}s)")

        # Comparar S-BERT vs TF-IDF
        logger.info("\n" + "-" * 80)
        logger.info("COMPARACIÓN: S-BERT vs TF-IDF")
        logger.info("-" * 80)

        import numpy as np

        sbert_matrix = results['similarities'].get('sbert')
        tfidf_matrix = results['similarities'].get('tfidf_cosine')

        if sbert_matrix is not None and tfidf_matrix is not None:
            mask = ~np.eye(sbert_matrix.shape[0], dtype=bool)

            sbert_avg = np.mean(sbert_matrix[mask])
            tfidf_avg = np.mean(tfidf_matrix[mask])

            logger.info(f"S-BERT - Similitud promedio: {sbert_avg:.3f}")
            logger.info(f"TF-IDF - Similitud promedio: {tfidf_avg:.3f}")
            logger.info(f"Diferencia: {abs(sbert_avg - tfidf_avg):.3f}")

            logger.info("\n📊 ¿Por qué S-BERT da resultados diferentes?")
            logger.info("  • TF-IDF: Compara palabras exactas (léxico)")
            logger.info("  • S-BERT: Compara significado (semántica)")
            logger.info("  • S-BERT captura sinónimos y contexto")
            logger.info("  • TF-IDF solo coincidencias de términos")

            if sbert_avg > tfidf_avg:
                logger.info("\n  → S-BERT detecta más similitud semántica")
                logger.info("    (los textos hablan de temas relacionados)")
            else:
                logger.info("\n  → TF-IDF detecta más coincidencias léxicas")
                logger.info("    (los textos usan palabras similares)")

        # ========================================================================
        # RECOMENDACIONES
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("RECOMENDACIONES PARA ABSTRACTS CIENTÍFICOS")
        logger.info("=" * 80)

        logger.info("\n🎯 Recomendación principal: S-BERT")
        logger.info("\nRazones:")
        logger.info("  1. Captura similitud semántica (conceptos relacionados)")
        logger.info("  2. Robusto a diferentes formulaciones del mismo concepto")
        logger.info("  3. Buen balance velocidad/precisión")
        logger.info("  4. Entrenado en textos científicos")

        logger.info("\n📌 Alternativas según contexto:")
        logger.info("  • Tiempo real / Alto volumen → TF-IDF")
        logger.info("  • Máxima precisión / Dataset pequeño → BERT")
        logger.info("  • Recursos limitados → Jaccard o N-gram")
        logger.info("  • Detección de plagio → Levenshtein + N-gram")

        logger.info("\n" + "=" * 80)

    except FileNotFoundError as e:
        logger.error(f"❌ Error: Archivo no encontrado - {e}")
        logger.error("Asegúrate de que 'data/unified_articles.json' existe")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
