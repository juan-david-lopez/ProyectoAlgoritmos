"""
Demo del sistema de análisis de términos predefinidos.

Este script demuestra cómo usar el PredefinedTermsAnalyzer para:
1. Analizar frecuencia de términos predefinidos
2. Generar estadísticas descriptivas
3. Crear visualizaciones
4. Generar reportes detallados
"""

import sys
import logging
from pathlib import Path

# Configurar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('term_analysis_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta demostración completa del análisis de términos."""
    logger.info("="*80)
    logger.info("DEMOSTRACIÓN DE ANÁLISIS DE TÉRMINOS PREDEFINIDOS")
    logger.info("="*80)

    try:
        # ====================================================================
        # PASO 1: INICIALIZAR ANALIZADOR
        # ====================================================================
        logger.info("\nPASO 1: Inicializando analizador...")

        analyzer = PredefinedTermsAnalyzer('data/unified_articles.json')

        logger.info(f"\nTérminos predefinidos a analizar:")
        for i, term in enumerate(analyzer.PREDEFINED_TERMS, 1):
            logger.info(f"  {i:2d}. {term}")

        # ====================================================================
        # PASO 2: CALCULAR FRECUENCIAS
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 2: Calculando frecuencias de términos")
        logger.info("="*80)

        frequencies = analyzer.calculate_frequencies()

        # Mostrar resumen
        logger.info("\nResumen de resultados:")
        total_occurrences = sum(f['total_count'] for f in frequencies.values())
        terms_found = sum(1 for f in frequencies.values() if f['total_count'] > 0)

        logger.info(f"  Total de ocurrencias: {total_occurrences}")
        logger.info(f"  Términos encontrados: {terms_found}/{len(analyzer.PREDEFINED_TERMS)}")

        # Top 5 términos
        sorted_terms = sorted(
            frequencies.items(),
            key=lambda x: x[1]['total_count'],
            reverse=True
        )

        logger.info("\nTop 5 términos más frecuentes:")
        for i, (term, stats) in enumerate(sorted_terms[:5], 1):
            logger.info(f"  {i}. {term}: {stats['total_count']} ocurrencias "
                       f"({stats['documents_count']} documentos)")

        # Términos menos frecuentes
        logger.info("\nTérminos menos frecuentes:")
        for i, (term, stats) in enumerate(sorted_terms[-3:], 1):
            if stats['total_count'] == 0:
                logger.info(f"  {i}. {term}: No encontrado")
            else:
                logger.info(f"  {i}. {term}: {stats['total_count']} ocurrencias")

        # ====================================================================
        # PASO 3: DETALLES DE VARIANTES
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 3: Análisis de variantes detectadas")
        logger.info("="*80)

        # Mostrar variantes de los términos más frecuentes
        logger.info("\nVariantes detectadas (Top 3 términos):")
        for term, stats in sorted_terms[:3]:
            if stats['variants_found']:
                logger.info(f"\n  {term}:")
                sorted_variants = sorted(
                    stats['variants_found'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for variant, count in sorted_variants[:5]:  # Top 5 variantes
                    logger.info(f"    - '{variant}': {count} veces")

        # ====================================================================
        # PASO 4: GENERAR ESTADÍSTICAS
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 4: Generando reporte estadístico")
        logger.info("="*80)

        stats_df = analyzer.generate_statistics_report(frequencies)

        logger.info("\nTabla de estadísticas:")
        logger.info("\n" + stats_df.to_string(index=False))

        # Estadísticas descriptivas
        import numpy as np
        counts = [f['total_count'] for f in frequencies.values()]

        logger.info("\nEstadísticas descriptivas:")
        logger.info(f"  Media: {np.mean(counts):.2f}")
        logger.info(f"  Mediana: {np.median(counts):.2f}")
        logger.info(f"  Desviación estándar: {np.std(counts):.2f}")
        logger.info(f"  Mínimo: {np.min(counts)}")
        logger.info(f"  Máximo: {np.max(counts)}")

        # ====================================================================
        # PASO 5: ANÁLISIS DE CO-OCURRENCIA
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 5: Análisis de co-ocurrencia")
        logger.info("="*80)

        cooccurrence = analyzer.calculate_cooccurrence_matrix()

        logger.info("\nPares de términos que más co-ocurren:")

        # Encontrar top pares (excluyendo diagonal)
        pairs = []
        for i in range(len(analyzer.PREDEFINED_TERMS)):
            for j in range(i+1, len(analyzer.PREDEFINED_TERMS)):
                count = cooccurrence.iloc[i, j]
                if count > 0:
                    pairs.append((
                        analyzer.PREDEFINED_TERMS[i],
                        analyzer.PREDEFINED_TERMS[j],
                        count
                    ))

        pairs.sort(key=lambda x: x[2], reverse=True)

        for i, (term1, term2, count) in enumerate(pairs[:5], 1):
            logger.info(f"  {i}. '{term1}' + '{term2}': {count} documentos")

        # ====================================================================
        # PASO 6: GENERAR VISUALIZACIONES
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 6: Generando visualizaciones")
        logger.info("="*80)

        output_dir = 'output/term_analysis'
        analyzer.visualize_frequencies(frequencies, output_dir)

        logger.info(f"\nVisualizaciones guardadas en '{output_dir}':")
        logger.info("  1. term_frequencies_bar.png - Gráfico de barras")
        logger.info("  2. term_cooccurrence_heatmap.png - Heatmap de co-ocurrencia")
        logger.info("  3. term_distribution_stats.png - Distribución estadística")

        # ====================================================================
        # PASO 7: GENERAR REPORTE DETALLADO
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 7: Generando reporte detallado")
        logger.info("="*80)

        report_path = 'output/term_analysis/predefined_terms_report.md'
        analyzer.generate_detailed_report(frequencies, report_path)

        logger.info(f"\nReporte guardado en: {report_path}")

        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("DEMOSTRACIÓN COMPLETADA")
        logger.info("="*80)

        logger.info("\nArchivos generados:")
        logger.info(f"  - Visualizaciones: output/term_analysis/")
        logger.info(f"  - Reporte: output/term_analysis/predefined_terms_report.md")
        logger.info(f"  - Log: term_analysis_demo.log")

        logger.info("\nInsights clave:")
        logger.info(f"  - {terms_found} de {len(analyzer.PREDEFINED_TERMS)} términos encontrados")
        logger.info(f"  - Término más frecuente: '{sorted_terms[0][0]}' ({sorted_terms[0][1]['total_count']} ocurrencias)")
        logger.info(f"  - Promedio de ocurrencias por término: {np.mean(counts):.1f}")

        if pairs:
            logger.info(f"  - Par más co-ocurrente: '{pairs[0][0]}' + '{pairs[0][1]}' ({pairs[0][2]} documentos)")

        logger.info("\n✅ Análisis completado exitosamente")

    except FileNotFoundError as e:
        logger.error(f"❌ Error: Archivo no encontrado - {e}")
        logger.error("Asegúrate de que 'data/unified_articles.json' existe")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
