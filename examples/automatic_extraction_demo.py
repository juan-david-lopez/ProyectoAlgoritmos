"""
Demo completo del sistema de extracción automática de términos.

Este script demuestra:
1. Extracción automática con 3 métodos (TF-IDF, RAKE, TextRank)
2. Método ensemble (combinado)
3. Evaluación vs términos predefinidos
4. Comparación de métodos
5. Visualizaciones y reportes
"""

import sys
import logging
from pathlib import Path

# Configurar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.term_analysis.automatic_term_extractor import AutomaticTermExtractor
from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer
from src.preprocessing.term_analysis.term_evaluator import TermEvaluator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automatic_extraction_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta demostración completa del sistema de extracción automática."""
    logger.info("="*80)
    logger.info("DEMOSTRACIÓN DE EXTRACCIÓN AUTOMÁTICA DE TÉRMINOS")
    logger.info("="*80)

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
        # PASO 2: INICIALIZAR EXTRACTOR
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 2: Inicializando extractor automático")
        logger.info("="*80)

        extractor = AutomaticTermExtractor(abstracts, max_terms=15)

        # ====================================================================
        # PASO 3: EXTRAER CON CADA MÉTODO
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 3: Extrayendo términos con cada método")
        logger.info("="*80)

        # TF-IDF
        tfidf_terms = extractor.extract_with_tfidf(15)
        logger.info(f"\nTF-IDF - Top 5:")
        for term, score in tfidf_terms[:5]:
            logger.info(f"  {term}: {score:.4f}")

        # RAKE
        rake_terms = extractor.extract_with_rake(15)
        logger.info(f"\nRAKE - Top 5:")
        for term, score in rake_terms[:5]:
            logger.info(f"  {term}: {score:.4f}")

        # TextRank
        textrank_terms = extractor.extract_with_textrank(15)
        logger.info(f"\nTextRank - Top 5:")
        for term, score in textrank_terms[:5]:
            logger.info(f"  {term}: {score:.4f}")

        # ====================================================================
        # PASO 4: MÉTODO ENSEMBLE (COMBINADO)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 4: Extracción con método ensemble")
        logger.info("="*80)

        combined_terms = extractor.extract_combined(15)

        logger.info(f"\nCombined (Ensemble) - Top 10:")
        for term, score, scores_dict in combined_terms[:10]:
            methods_used = scores_dict['methods_count']
            logger.info(f"  {term}: {score:.4f} (detectado por {methods_used} métodos)")

        # Mostrar desglose de un término
        if combined_terms:
            top_term, top_score, top_scores = combined_terms[0]
            logger.info(f"\nDesglose del término top '{top_term}':")
            logger.info(f"  TF-IDF:   {top_scores['tfidf']:.4f}")
            logger.info(f"  RAKE:     {top_scores['rake']:.4f}")
            logger.info(f"  TextRank: {top_scores['textrank']:.4f}")
            logger.info(f"  Combined: {top_score:.4f}")

        # ====================================================================
        # PASO 5: COMPARAR CON TÉRMINOS PREDEFINIDOS
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 5: Evaluando vs términos predefinidos")
        logger.info("="*80)

        # Términos predefinidos como ground truth
        ground_truth = PredefinedTermsAnalyzer.PREDEFINED_TERMS

        logger.info(f"\nUsando {len(ground_truth)} términos predefinidos como ground truth")

        # Preparar resultados para evaluación
        methods_results = {
            'TF-IDF': [term for term, _ in tfidf_terms],
            'RAKE': [term for term, _ in rake_terms],
            'TextRank': [term for term, _ in textrank_terms],
            'Combined': [term for term, _, _ in combined_terms]
        }

        # Crear evaluador
        evaluator = TermEvaluator(ground_truth)

        # Comparar métodos
        comparison = evaluator.compare_methods(methods_results)

        logger.info(f"\nComparación de métodos:")
        logger.info("\n" + comparison.to_string(index=False))

        # ====================================================================
        # PASO 6: ANÁLISIS DETALLADO DEL MEJOR MÉTODO
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 6: Análisis del mejor método")
        logger.info("="*80)

        best_method = comparison.iloc[0]
        logger.info(f"\nMejor método: {best_method['Method']}")
        logger.info(f"  Precision: {best_method['Precision']:.3f}")
        logger.info(f"  Recall: {best_method['Recall']:.3f}")
        logger.info(f"  F1-Score: {best_method['F1-Score']:.3f}")
        logger.info(f"  True Positives: {best_method['TP']}")
        logger.info(f"  False Positives: {best_method['FP']}")
        logger.info(f"  False Negatives: {best_method['FN']}")

        # Mostrar términos correctos e incorrectos del mejor método
        best_method_name = best_method['Method']
        best_extracted = methods_results[best_method_name]

        # Normalizar para comparación
        best_normalized = set(t.lower() for t in best_extracted)
        gt_normalized = set(t.lower() for t in ground_truth)

        correct_terms = best_normalized & gt_normalized
        incorrect_terms = best_normalized - gt_normalized
        missed_terms = gt_normalized - best_normalized

        logger.info(f"\n✓ Términos correctos ({len(correct_terms)}):")
        for term in sorted(correct_terms)[:5]:
            logger.info(f"  - {term}")
        if len(correct_terms) > 5:
            logger.info(f"  ... y {len(correct_terms) - 5} más")

        logger.info(f"\n✗ Términos incorrectos ({len(incorrect_terms)}):")
        for term in sorted(incorrect_terms)[:5]:
            logger.info(f"  - {term}")
        if len(incorrect_terms) > 5:
            logger.info(f"  ... y {len(incorrect_terms) - 5} más")

        logger.info(f"\n⚠️ Términos no detectados ({len(missed_terms)}):")
        for term in sorted(missed_terms)[:5]:
            logger.info(f"  - {term}")
        if len(missed_terms) > 5:
            logger.info(f"  ... y {len(missed_terms) - 5} más")

        # ====================================================================
        # PASO 7: GENERAR VISUALIZACIONES
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 7: Generando visualizaciones")
        logger.info("="*80)

        output_dir = 'output/term_analysis'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Visualización de comparación
        evaluator.visualize_comparison(
            comparison,
            f'{output_dir}/automatic_extraction_comparison.png'
        )

        logger.info(f"\n✓ Visualización guardada")

        # ====================================================================
        # PASO 8: GENERAR REPORTES
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 8: Generando reportes")
        logger.info("="*80)

        # Reporte de evaluación
        evaluator.generate_evaluation_report(
            comparison,
            methods_results,
            f'{output_dir}/automatic_extraction_evaluation.md'
        )

        # Reporte de términos extraídos
        import pandas as pd

        # Crear DataFrame con todos los términos
        all_terms_data = []
        for term, score, scores_dict in combined_terms:
            all_terms_data.append({
                'Term': term,
                'Combined Score': f'{score:.4f}',
                'TF-IDF': f'{scores_dict["tfidf"]:.4f}',
                'RAKE': f'{scores_dict["rake"]:.4f}',
                'TextRank': f'{scores_dict["textrank"]:.4f}',
                'Methods': scores_dict['methods_count'],
                'In Ground Truth': 'Yes' if term.lower() in gt_normalized else 'No'
            })

        terms_df = pd.DataFrame(all_terms_data)

        # Guardar como CSV
        terms_df.to_csv(f'{output_dir}/extracted_terms.csv', index=False)
        logger.info(f"✓ Términos extraídos guardados en CSV")

        logger.info(f"\n✓ Reportes generados en '{output_dir}'")

        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("DEMOSTRACIÓN COMPLETADA")
        logger.info("="*80)

        logger.info("\nArchivos generados:")
        logger.info(f"  - Visualización: {output_dir}/automatic_extraction_comparison.png")
        logger.info(f"  - Reporte de evaluación: {output_dir}/automatic_extraction_evaluation.md")
        logger.info(f"  - Términos extraídos (CSV): {output_dir}/extracted_terms.csv")
        logger.info(f"  - Log: automatic_extraction_demo.log")

        logger.info("\nInsights clave:")
        logger.info(f"  - Mejor método: {best_method['Method']} (F1={best_method['F1-Score']:.3f})")
        logger.info(f"  - Términos correctos: {best_method['TP']}/{best_method['Total Extracted']}")
        logger.info(f"  - Cobertura ground truth: {best_method['TP']}/{len(ground_truth)}")

        # Calcular overlap entre métodos
        tfidf_set = set(t.lower() for t, _ in tfidf_terms)
        rake_set = set(t.lower() for t, _ in rake_terms)
        textrank_set = set(t.lower() for t, _ in textrank_terms)

        overlap_all = len(tfidf_set & rake_set & textrank_set)
        logger.info(f"  - Términos detectados por los 3 métodos: {overlap_all}")

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
