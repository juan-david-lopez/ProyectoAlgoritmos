"""
Demo completo del evaluador de precisión de términos.

Este script demuestra:
1. Extracción automática de términos
2. Comparación con términos predefinidos usando similitud semántica (SBERT)
3. Identificación de matches exactos, parciales y términos nuevos
4. Cálculo de métricas (Precision, Recall, F1, Coverage)
5. Explicación de términos nuevos con contextos
6. Visualizaciones (matriz de similitud, Venn diagram)
7. Generación de reporte completo
"""

import sys
import logging
from pathlib import Path

# Configurar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer
from src.preprocessing.term_analysis.automatic_term_extractor import AutomaticTermExtractor
from src.preprocessing.term_analysis.term_precision_evaluator import TermPrecisionEvaluator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precision_evaluation_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta demostración completa del evaluador de precisión."""
    logger.info("=" * 80)
    logger.info("DEMOSTRACIÓN DE EVALUACIÓN DE PRECISIÓN DE TÉRMINOS")
    logger.info("=" * 80)

    try:
        # ====================================================================
        # PASO 1: CARGAR DATOS
        # ====================================================================
        logger.info("\nPASO 1: Cargando abstracts...")

        import json
        with open('data/unified_articles.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)

        abstracts = [art['abstract'] for art in articles if art.get('abstract')]

        logger.info(f"✓ {len(abstracts)} abstracts cargados")

        # ====================================================================
        # PASO 2: EXTRAER TÉRMINOS AUTOMÁTICAMENTE
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 2: Extrayendo términos automáticamente")
        logger.info("=" * 80)

        extractor = AutomaticTermExtractor(abstracts, max_terms=15)

        # Usar método ensemble (combinado)
        combined_terms = extractor.extract_combined(15)

        # Preparar lista de términos extraídos (solo nombres)
        extracted_terms_list = [term for term, _, _ in combined_terms]

        logger.info(f"\n✓ Extraídos {len(extracted_terms_list)} términos")
        logger.info("\nTop 10 términos extraídos:")
        for i, (term, score, scores_dict) in enumerate(combined_terms[:10], 1):
            logger.info(f"  {i}. {term} (score={score:.4f}, methods={scores_dict['methods_count']})")

        # ====================================================================
        # PASO 3: OBTENER TÉRMINOS PREDEFINIDOS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 3: Obteniendo términos predefinidos")
        logger.info("=" * 80)

        predefined_terms_list = PredefinedTermsAnalyzer.PREDEFINED_TERMS

        logger.info(f"\n✓ {len(predefined_terms_list)} términos predefinidos")
        logger.info("\nTérminos predefinidos:")
        for i, term in enumerate(predefined_terms_list, 1):
            logger.info(f"  {i}. {term}")

        # ====================================================================
        # PASO 4: INICIALIZAR EVALUADOR DE PRECISIÓN
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 4: Inicializando evaluador de precisión")
        logger.info("=" * 80)

        evaluator = TermPrecisionEvaluator(
            predefined_terms=predefined_terms_list,
            extracted_terms=extracted_terms_list
        )

        # ====================================================================
        # PASO 5: CALCULAR MATRIZ DE SIMILITUD
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 5: Calculando matriz de similitud")
        logger.info("=" * 80)

        similarity_matrix = evaluator.calculate_similarity_matrix()

        logger.info(f"\nEstadísticas de similitud:")
        logger.info(f"  Promedio: {similarity_matrix.mean():.3f}")
        logger.info(f"  Mínimo: {similarity_matrix.min():.3f}")
        logger.info(f"  Máximo: {similarity_matrix.max():.3f}")
        logger.info(f"  Desviación estándar: {similarity_matrix.std():.3f}")

        # Mostrar top 5 similitudes
        import numpy as np
        flat_indices = np.argsort(similarity_matrix.ravel())[::-1]
        top_n = 5

        logger.info(f"\nTop {top_n} similitudes más altas:")
        for idx in flat_indices[:top_n]:
            i, j = np.unravel_index(idx, similarity_matrix.shape)
            pred_term = predefined_terms_list[i]
            ext_term = extracted_terms_list[j]
            sim = similarity_matrix[i, j]
            logger.info(f"  {pred_term} <-> {ext_term}: {sim:.3f}")

        # ====================================================================
        # PASO 6: IDENTIFICAR MATCHES
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 6: Identificando matches con threshold=0.70")
        logger.info("=" * 80)

        matches = evaluator.identify_matches(threshold=0.70)

        # Mostrar matches exactos
        if matches['exact_matches']:
            logger.info(f"\n✓ Exact Matches ({len(matches['exact_matches'])}):")
            for pred, ext, sim in matches['exact_matches']:
                logger.info(f"  - '{pred}' <-> '{ext}' (similarity={sim:.3f})")
        else:
            logger.info("\n  No se encontraron matches exactos")

        # Mostrar matches parciales
        if matches['partial_matches']:
            logger.info(f"\n✓ Partial Matches ({len(matches['partial_matches'])}):")
            for pred, ext, sim in matches['partial_matches']:
                logger.info(f"  - '{pred}' <-> '{ext}' (similarity={sim:.3f})")
        else:
            logger.info("\n  No se encontraron matches parciales")

        # Mostrar términos nuevos
        if matches['novel_terms']:
            logger.info(f"\n✓ Novel Terms ({len(matches['novel_terms'])}):")
            for term in matches['novel_terms'][:5]:
                logger.info(f"  - {term}")
            if len(matches['novel_terms']) > 5:
                logger.info(f"  ... y {len(matches['novel_terms']) - 5} más")
        else:
            logger.info("\n  No se encontraron términos nuevos")

        # Mostrar términos no encontrados
        if matches['predefined_not_found']:
            logger.info(f"\n⚠️ Predefined Terms Not Found ({len(matches['predefined_not_found'])}):")
            for term in matches['predefined_not_found'][:5]:
                logger.info(f"  - {term}")
            if len(matches['predefined_not_found']) > 5:
                logger.info(f"  ... y {len(matches['predefined_not_found']) - 5} más")
        else:
            logger.info("\n  ✓ Todos los términos predefinidos fueron encontrados")

        # ====================================================================
        # PASO 7: CALCULAR MÉTRICAS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 7: Calculando métricas de evaluación")
        logger.info("=" * 80)

        metrics = evaluator.calculate_metrics(matches)

        logger.info(f"\nMétricas de evaluación:")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"  Coverage: {metrics['coverage']:.1%}")
        logger.info(f"\nDesglose:")
        logger.info(f"  Exact matches: {metrics['exact_match_count']}")
        logger.info(f"  Partial matches: {metrics['partial_match_count']}")
        logger.info(f"  Novel terms: {metrics['novel_terms_count']}")
        logger.info(f"  Not found: {metrics['predefined_not_found_count']}")

        # Interpretación
        if metrics['f1_score'] >= 0.7:
            assessment = "✅ EXCELENTE - Alta concordancia"
        elif metrics['f1_score'] >= 0.5:
            assessment = "✓ BUENO - Concordancia moderada"
        else:
            assessment = "⚠️ REQUIERE MEJORA - Baja concordancia"

        logger.info(f"\nEvaluación general: {assessment}")

        # ====================================================================
        # PASO 8: EXPLICAR TÉRMINOS NUEVOS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 8: Explicando términos nuevos")
        logger.info("=" * 80)

        if matches['novel_terms']:
            novel_explanations = evaluator.explain_novel_terms(
                matches['novel_terms'],
                abstracts
            )

            logger.info(f"\nAnálisis de términos nuevos:")
            for term in list(matches['novel_terms'])[:3]:  # Top 3
                if term in novel_explanations:
                    expl = novel_explanations[term]
                    logger.info(f"\n  Término: '{term}'")
                    logger.info(f"    Frecuencia: {expl['frequency']}")
                    logger.info(f"    Documentos: {expl['document_frequency']}")
                    logger.info(f"    Relevance score: {expl['relevance_score']:.3f}")
                    logger.info(f"    Interpretación: {expl['interpretation'][:100]}...")

                    if expl['example_contexts']:
                        logger.info(f"    Contexto ejemplo: {expl['example_contexts'][0][:80]}...")

            if len(matches['novel_terms']) > 3:
                logger.info(f"\n  ... y {len(matches['novel_terms']) - 3} términos más")
                logger.info("  (ver reporte completo para detalles)")

        else:
            novel_explanations = {}
            logger.info("\n  No hay términos nuevos para explicar")

        # ====================================================================
        # PASO 9: GENERAR VISUALIZACIONES
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 9: Generando visualizaciones")
        logger.info("=" * 80)

        output_dir = 'output/term_analysis'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Matriz de similitud
        evaluator.visualize_similarity_matrix(
            f'{output_dir}/precision_similarity_matrix.png'
        )

        # Venn diagram
        evaluator.visualize_venn_diagram(
            matches,
            f'{output_dir}/precision_venn_diagram.png'
        )

        logger.info(f"\n✓ Visualizaciones guardadas en '{output_dir}'")

        # ====================================================================
        # PASO 10: GENERAR REPORTE COMPLETO
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 10: Generando reporte completo")
        logger.info("=" * 80)

        evaluator.generate_evaluation_report(
            matches=matches,
            metrics=metrics,
            novel_explanations=novel_explanations,
            output_path=f'{output_dir}/precision_evaluation_report.md'
        )

        logger.info(f"\n✓ Reporte generado")

        # ====================================================================
        # PASO 11: CREAR TABLA COMPARATIVA DETALLADA
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 11: Creando tabla comparativa detallada")
        logger.info("=" * 80)

        import pandas as pd

        # Crear DataFrame con todos los términos predefinidos y sus matches
        comparison_data = []

        for i, pred_term in enumerate(predefined_terms_list):
            # Buscar mejor match para este término predefinido
            best_match_idx = similarity_matrix[i, :].argmax()
            best_similarity = similarity_matrix[i, best_match_idx]
            best_match_term = extracted_terms_list[best_match_idx]

            # Determinar categoría
            if best_similarity >= 0.70:
                category = "Exact Match"
            elif best_similarity >= 0.50:
                category = "Partial Match"
            else:
                category = "Not Found"

            comparison_data.append({
                'Predefined Term': pred_term,
                'Best Match': best_match_term if best_similarity >= 0.50 else '-',
                'Similarity': f'{best_similarity:.3f}',
                'Category': category
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Guardar como CSV
        comparison_df.to_csv(f'{output_dir}/precision_comparison.csv', index=False)
        logger.info(f"\n✓ Tabla comparativa guardada en CSV")

        # Mostrar resumen
        logger.info(f"\nResumen por categoría:")
        category_counts = comparison_df['Category'].value_counts()
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count}")

        # ====================================================================
        # PASO 12: ANÁLISIS DE THRESHOLD
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PASO 12: Analizando impacto de diferentes thresholds")
        logger.info("=" * 80)

        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
        threshold_analysis = []

        for threshold in thresholds:
            matches_t = evaluator.identify_matches(threshold=threshold)
            metrics_t = evaluator.calculate_metrics(matches_t)

            threshold_analysis.append({
                'Threshold': threshold,
                'Exact Matches': len(matches_t['exact_matches']),
                'Partial Matches': len(matches_t['partial_matches']),
                'Precision': f"{metrics_t['precision']:.3f}",
                'Recall': f"{metrics_t['recall']:.3f}",
                'F1-Score': f"{metrics_t['f1_score']:.3f}"
            })

        threshold_df = pd.DataFrame(threshold_analysis)

        logger.info("\nAnálisis de thresholds:")
        logger.info("\n" + threshold_df.to_string(index=False))

        # Guardar análisis
        threshold_df.to_csv(f'{output_dir}/threshold_analysis.csv', index=False)

        # Encontrar mejor threshold
        best_threshold_idx = threshold_df['F1-Score'].astype(float).idxmax()
        best_threshold = threshold_df.loc[best_threshold_idx]

        logger.info(f"\nMejor threshold: {best_threshold['Threshold']:.2f}")
        logger.info(f"  F1-Score: {best_threshold['F1-Score']}")
        logger.info(f"  Exact matches: {best_threshold['Exact Matches']}")

        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("DEMOSTRACIÓN COMPLETADA")
        logger.info("=" * 80)

        logger.info("\nArchivos generados:")
        logger.info(f"  - Matriz de similitud: {output_dir}/precision_similarity_matrix.png")
        logger.info(f"  - Venn diagram: {output_dir}/precision_venn_diagram.png")
        logger.info(f"  - Reporte completo: {output_dir}/precision_evaluation_report.md")
        logger.info(f"  - Tabla comparativa: {output_dir}/precision_comparison.csv")
        logger.info(f"  - Análisis de thresholds: {output_dir}/threshold_analysis.csv")
        logger.info(f"  - Log: precision_evaluation_demo.log")

        logger.info("\nResultados clave:")
        logger.info(f"  - Precision: {metrics['precision']:.3f}")
        logger.info(f"  - Recall: {metrics['recall']:.3f}")
        logger.info(f"  - F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"  - Coverage: {metrics['coverage']:.1%}")
        logger.info(f"  - Exact matches: {metrics['exact_match_count']}/{len(predefined_terms_list)}")
        logger.info(f"  - Novel terms: {metrics['novel_terms_count']}")

        logger.info(f"\n{assessment}")

        logger.info("\n✅ Evaluación de precisión completada exitosamente")

    except FileNotFoundError as e:
        logger.error(f"❌ Error: Archivo no encontrado - {e}")
        logger.error("Asegúrate de que 'data/unified_articles.json' existe")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
