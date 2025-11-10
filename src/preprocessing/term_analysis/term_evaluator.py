"""
Herramientas de evaluación y comparación de métodos de extracción de términos.

Este módulo permite:
    - Comparar términos extraídos con términos predefinidos (ground truth)
    - Calcular métricas de precisión (Precision, Recall, F1)
    - Visualizar comparaciones entre métodos
    - Generar reportes de evaluación
"""

import logging
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TermEvaluator:
    """
    Evalúa y compara métodos de extracción de términos.

    Métricas implementadas:
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
        - Overlap: Proporción de términos en común

    Donde:
        - TP (True Positives): Términos extraídos que están en ground truth
        - FP (False Positives): Términos extraídos que NO están en ground truth
        - FN (False Negatives): Términos en ground truth NO extraídos
    """

    def __init__(self, ground_truth_terms: List[str]):
        """
        Inicializa evaluador con términos de referencia (ground truth).

        Args:
            ground_truth_terms: Lista de términos esperados/correctos
        """
        logger.info(f"Inicializando TermEvaluator...")
        logger.info(f"  Ground truth: {len(ground_truth_terms)} términos")

        # Normalizar términos de referencia (lowercase)
        self.ground_truth = set(term.lower() for term in ground_truth_terms)

        logger.info("✓ TermEvaluator inicializado")

    def normalize_term(self, term: str) -> str:
        """
        Normaliza un término para comparación.

        Args:
            term: Término a normalizar

        Returns:
            Término normalizado (lowercase, stripped)
        """
        return term.lower().strip()

    def calculate_metrics(self, extracted_terms: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de evaluación para términos extraídos.

        Métricas:
            Precision = TP / (TP + FP)
                = Proporción de términos extraídos que son correctos

            Recall = TP / (TP + FN)
                = Proporción de términos correctos que fueron extraídos

            F1-Score = 2 × (P × R) / (P + R)
                = Media armónica de Precision y Recall

        Ejemplo:
            Ground truth: {A, B, C, D, E}  (5 términos)
            Extracted: {A, B, F, G}        (4 términos)

            TP: {A, B} = 2
            FP: {F, G} = 2
            FN: {C, D, E} = 3

            Precision: 2 / (2+2) = 0.50  (50% de extraídos son correctos)
            Recall: 2 / (2+3) = 0.40     (40% de correctos fueron extraídos)
            F1: 2×(0.50×0.40)/(0.50+0.40) = 0.44

        Args:
            extracted_terms: Lista de términos extraídos

        Returns:
            Dict con métricas: precision, recall, f1_score, tp, fp, fn
        """
        # Normalizar términos extraídos
        extracted_set = set(self.normalize_term(term) for term in extracted_terms)

        # Calcular TP, FP, FN
        tp = len(extracted_set & self.ground_truth)  # Intersección
        fp = len(extracted_set - self.ground_truth)  # Extraídos pero no en GT
        fn = len(self.ground_truth - extracted_set)  # En GT pero no extraídos

        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_extracted': len(extracted_set),
            'total_ground_truth': len(self.ground_truth)
        }

    def compare_methods(self, methods_results: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compara múltiples métodos de extracción.

        Args:
            methods_results: Dict con nombre_método -> lista_términos_extraídos
                Ejemplo: {
                    'TF-IDF': ['machine learning', 'neural networks', ...],
                    'RAKE': ['deep learning', 'natural language', ...],
                    'Combined': [...]
                }

        Returns:
            DataFrame con métricas por método
        """
        logger.info(f"\nComparando {len(methods_results)} métodos...")

        results = []
        for method_name, extracted_terms in methods_results.items():
            logger.info(f"  Evaluando: {method_name}")
            metrics = self.calculate_metrics(extracted_terms)

            results.append({
                'Method': method_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'TP': metrics['true_positives'],
                'FP': metrics['false_positives'],
                'FN': metrics['false_negatives'],
                'Total Extracted': metrics['total_extracted']
            })

        # Crear DataFrame
        df = pd.DataFrame(results)

        # Ordenar por F1-Score descendente
        df = df.sort_values('F1-Score', ascending=False)

        logger.info(f"\n✓ Comparación completada")

        return df

    def visualize_comparison(self, comparison_df: pd.DataFrame, output_path: str = None):
        """
        Visualiza comparación de métodos.

        Genera 2 gráficos:
            1. Barras agrupadas: Precision, Recall, F1-Score
            2. Confusion matrix style: TP, FP, FN

        Args:
            comparison_df: DataFrame de compare_methods()
            output_path: Ruta para guardar figura (None = mostrar)
        """
        logger.info("\nGenerando visualización de comparación...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ====================================================================
        # Subplot 1: Métricas principales (Precision, Recall, F1)
        # ====================================================================

        methods = comparison_df['Method'].tolist()
        metrics_data = comparison_df[['Precision', 'Recall', 'F1-Score']].values

        x = np.arange(len(methods))
        width = 0.25

        # Barras
        ax1.bar(x - width, metrics_data[:, 0], width, label='Precision', color='steelblue')
        ax1.bar(x, metrics_data[:, 1], width, label='Recall', color='lightcoral')
        ax1.bar(x + width, metrics_data[:, 2], width, label='F1-Score', color='mediumseagreen')

        # Configurar
        ax1.set_xlabel('Method', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax1.set_title('Comparison of Extraction Methods', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Agregar valores sobre las barras
        for i, method_idx in enumerate(x):
            for j, metric_val in enumerate(metrics_data[i]):
                x_pos = method_idx + (j - 1) * width
                ax1.text(x_pos, metric_val + 0.02, f'{metric_val:.2f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        # ====================================================================
        # Subplot 2: Desglose TP/FP/FN
        # ====================================================================

        tp_fp_fn_data = comparison_df[['TP', 'FP', 'FN']].values

        # Barras apiladas
        ax2.bar(methods, tp_fp_fn_data[:, 0], label='True Positives (TP)', color='#2ecc71')
        ax2.bar(methods, tp_fp_fn_data[:, 1], bottom=tp_fp_fn_data[:, 0],
               label='False Positives (FP)', color='#e74c3c')
        ax2.bar(methods, tp_fp_fn_data[:, 2],
               bottom=tp_fp_fn_data[:, 0] + tp_fp_fn_data[:, 1],
               label='False Negatives (FN)', color='#f39c12')

        # Configurar
        ax2.set_xlabel('Method', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Count', fontweight='bold', fontsize=12)
        ax2.set_title('True/False Positives and False Negatives', fontweight='bold', fontsize=14)
        ax2.set_xticklabels(methods, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Visualización guardada en: {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_evaluation_report(self, comparison_df: pd.DataFrame,
                                   methods_results: Dict[str, List[str]],
                                   output_path: str):
        """
        Genera reporte detallado de evaluación en Markdown.

        Args:
            comparison_df: DataFrame de comparación
            methods_results: Resultados originales de métodos
            output_path: Ruta donde guardar el reporte
        """
        logger.info(f"\nGenerando reporte de evaluación en '{output_path}'...")

        report = []
        report.append("# Reporte de Evaluación de Extracción de Términos\n")
        report.append(f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Ground Truth:** {len(self.ground_truth)} términos\n")
        report.append("\n---\n")

        # Resumen ejecutivo
        report.append("## Resumen Ejecutivo\n")
        best_method = comparison_df.iloc[0]
        report.append(f"- **Mejor método:** {best_method['Method']} (F1={best_method['F1-Score']:.3f})\n")
        report.append(f"- **Mayor Precision:** {comparison_df.loc[comparison_df['Precision'].idxmax()]['Method']}\n")
        report.append(f"- **Mayor Recall:** {comparison_df.loc[comparison_df['Recall'].idxmax()]['Method']}\n")
        report.append("\n---\n")

        # Tabla de comparación
        report.append("## Comparación de Métodos\n")
        report.append(comparison_df.to_markdown(index=False))
        report.append("\n\n---\n")

        # Términos por método
        report.append("## Términos Extraídos por Método\n")
        for method_name, extracted_terms in methods_results.items():
            report.append(f"\n### {method_name}\n")
            report.append(f"**Total:** {len(extracted_terms)} términos\n\n")

            # Clasificar términos
            normalized = set(self.normalize_term(t) for t in extracted_terms)
            tp_terms = normalized & self.ground_truth
            fp_terms = normalized - self.ground_truth

            if tp_terms:
                report.append("**✓ True Positives (correctos):**\n")
                for term in sorted(tp_terms):
                    report.append(f"- `{term}`\n")
                report.append("\n")

            if fp_terms:
                report.append("**✗ False Positives (incorrectos):**\n")
                for term in sorted(fp_terms):
                    report.append(f"- `{term}`\n")
                report.append("\n")

        # Términos no detectados
        report.append("## Términos No Detectados (False Negatives)\n")
        all_extracted = set()
        for extracted in methods_results.values():
            all_extracted.update(self.normalize_term(t) for t in extracted)

        fn_terms = self.ground_truth - all_extracted
        if fn_terms:
            report.append(f"**{len(fn_terms)} términos** del ground truth no fueron detectados por ningún método:\n\n")
            for term in sorted(fn_terms):
                report.append(f"- `{term}`\n")
        else:
            report.append("✓ Todos los términos del ground truth fueron detectados por al menos un método.\n")

        report.append("\n---\n")

        # Recomendaciones
        report.append("## Recomendaciones\n")
        best_f1 = comparison_df.iloc[0]

        if best_f1['F1-Score'] >= 0.7:
            report.append(f"✅ El método **{best_f1['Method']}** muestra excelente desempeño (F1 ≥ 0.7).\n")
        elif best_f1['F1-Score'] >= 0.5:
            report.append(f"✓ El método **{best_f1['Method']}** muestra buen desempeño (F1 ≥ 0.5).\n")
        else:
            report.append(f"⚠️ El mejor método ({best_f1['Method']}) tiene F1 < 0.5. Considerar:\n")
            report.append("- Ajustar parámetros de extracción\n")
            report.append("- Expandir corpus de entrenamiento\n")
            report.append("- Revisar ground truth\n")

        # Guardar reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))

        logger.info(f"✓ Reporte guardado en '{output_path}'")


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Ground truth (términos esperados)
    ground_truth = [
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing",
        "artificial intelligence"
    ]

    # Resultados de diferentes métodos (simulados)
    methods_results = {
        'TF-IDF': ['machine learning', 'deep learning', 'algorithms', 'data'],
        'RAKE': ['machine learning', 'neural networks', 'training data'],
        'TextRank': ['deep learning', 'artificial intelligence', 'models'],
        'Combined': ['machine learning', 'deep learning', 'neural networks', 'nlp']
    }

    # Crear evaluador
    evaluator = TermEvaluator(ground_truth)

    # Comparar métodos
    comparison = evaluator.compare_methods(methods_results)
    print("\nComparación de métodos:")
    print(comparison)

    # Visualizar
    evaluator.visualize_comparison(comparison, 'output/term_analysis/method_comparison.png')

    # Generar reporte
    evaluator.generate_evaluation_report(
        comparison,
        methods_results,
        'output/term_analysis/evaluation_report.md'
    )
