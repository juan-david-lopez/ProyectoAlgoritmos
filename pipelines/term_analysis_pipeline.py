"""
Term Analysis Pipeline - Análisis Completo Integrado
Integra las 3 partes del proyecto:
- Parte 1: Análisis de términos predefinidos
- Parte 2: Extracción automática de términos
- Parte 3: Evaluación de precisión
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importar componentes del proyecto
from predefined_term_analyzer import PredefinedTermAnalyzer
from auto_term_extractor import AutoTermExtractor
from term_precision_evaluator import TermPrecisionEvaluator


class TermAnalysisPipeline:
    """
    Pipeline completo para análisis de términos en corpus académico.
    Integra análisis de términos predefinidos, extracción automática y evaluación.
    """

    def __init__(self, unified_data_path: str, output_dir: str):
        """
        Inicializa el pipeline.

        Args:
            unified_data_path: Ruta al archivo unified_abstracts.json
            output_dir: Directorio para guardar todos los outputs
        """
        self.unified_data_path = unified_data_path
        self.output_dir = output_dir

        # Crear directorio de salida
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Subdirectorios
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.data_dir = os.path.join(output_dir, 'data')
        self.reports_dir = os.path.join(output_dir, 'reports')

        for dir_path in [self.viz_dir, self.data_dir, self.reports_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Datos y resultados
        self.data = None
        self.abstracts = None
        self.predefined_terms = None
        self.predefined_results = None
        self.extracted_results = None
        self.evaluation_results = None

    def load_data(self) -> Dict:
        """
        Carga datos del archivo unified_abstracts.json.

        Returns:
            Diccionario con datos del unified file
        """
        print("\n" + "="*70)
        print("CARGANDO DATOS")
        print("="*70)

        with open(self.unified_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Extraer abstracts
        self.abstracts = [
            paper['abstract']
            for paper in self.data['papers']
            if paper.get('abstract')
        ]

        # Extraer términos predefinidos
        self.predefined_terms = self.data.get('predefined_terms', [])

        print(f"✓ Total papers: {len(self.data['papers'])}")
        print(f"✓ Abstracts disponibles: {len(self.abstracts)}")
        print(f"✓ Términos predefinidos: {len(self.predefined_terms)}")
        print(f"✓ Rango de fechas: {self.data['metadata']['date_range']['start']} - "
              f"{self.data['metadata']['date_range']['end']}")

        return self.data

    def analyze_predefined_terms(self) -> Dict:
        """
        Ejecuta análisis de términos predefinidos (Parte 1).

        Returns:
            Resultados del análisis de términos predefinidos
        """
        print("\n" + "="*70)
        print("PARTE 1: ANÁLISIS DE TÉRMINOS PREDEFINIDOS")
        print("="*70)

        analyzer = PredefinedTermAnalyzer(
            abstracts=self.abstracts,
            predefined_terms=self.predefined_terms
        )

        # Calcular frecuencias
        print("\n[1/3] Calculando frecuencias de términos...")
        frequencies = analyzer.calculate_term_frequencies()

        # Calcular co-ocurrencias
        print("[2/3] Calculando co-ocurrencias...")
        co_occurrences = analyzer.calculate_co_occurrences()

        # Generar reporte
        print("[3/3] Generando reporte de términos predefinidos...")
        report_path = os.path.join(self.reports_dir, 'predefined_terms_report.md')
        analyzer.generate_report(report_path)

        # Guardar frecuencias en CSV
        freq_df = pd.DataFrame([
            {'term': term, 'frequency': freq}
            for term, freq in frequencies.items()
        ]).sort_values('frequency', ascending=False)

        freq_csv_path = os.path.join(self.data_dir, 'predefined_terms_frequencies.csv')
        freq_df.to_csv(freq_csv_path, index=False)

        self.predefined_results = {
            'frequencies': frequencies,
            'co_occurrences': co_occurrences,
            'report_path': report_path,
            'freq_csv_path': freq_csv_path,
            'top_10_terms': freq_df.head(10)['term'].tolist()
        }

        print(f"\n✓ Reporte generado: {report_path}")
        print(f"✓ Frecuencias guardadas: {freq_csv_path}")

        return self.predefined_results

    def extract_terms_automatically(self) -> Dict:
        """
        Ejecuta extracción automática de términos (Parte 2).

        Returns:
            Resultados de extracción automática
        """
        print("\n" + "="*70)
        print("PARTE 2: EXTRACCIÓN AUTOMÁTICA DE TÉRMINOS")
        print("="*70)

        extractor = AutoTermExtractor(self.abstracts)

        # Extraer con RAKE
        print("\n[1/4] Extrayendo términos con RAKE...")
        extractor.extract_with_rake()
        rake_terms = extractor.rake_terms[:50]
        print(f"  ✓ RAKE extrajo {len(extractor.rake_terms)} términos (top 50 seleccionados)")

        # Extraer con TextRank
        print("[2/4] Extrayendo términos con TextRank...")
        extractor.extract_with_textrank()
        textrank_terms = extractor.textrank_terms[:50]
        print(f"  ✓ TextRank extrajo {len(extractor.textrank_terms)} términos (top 50 seleccionados)")

        # Combinar términos
        print("[3/4] Combinando términos de ambos métodos...")
        combined_terms = extractor.get_combined_top_terms(n=50)
        print(f"  ✓ Términos combinados: {len(combined_terms)}")

        # Generar reporte
        print("[4/4] Generando reporte de extracción...")
        extraction_report_path = os.path.join(
            self.reports_dir,
            'extracted_terms_report.md'
        )
        extractor.generate_extraction_report(extraction_report_path)

        # Guardar términos extraídos en CSV
        extracted_df = pd.DataFrame([
            {
                'term': term,
                'method': method,
                'score': score
            }
            for term, method, score in [
                *[(t[0], 'RAKE', t[1]) for t in rake_terms[:50]],
                *[(t[0], 'TextRank', t[1]) for t in textrank_terms[:50]],
                *[(t, 'Combined', 1.0) for t in combined_terms]
            ]
        ])

        extracted_csv_path = os.path.join(
            self.data_dir,
            'extracted_terms_all_methods.csv'
        )
        extracted_df.to_csv(extracted_csv_path, index=False)

        self.extracted_results = {
            'rake_terms': [t[0] for t in rake_terms],
            'textrank_terms': [t[0] for t in textrank_terms],
            'combined_terms': combined_terms,
            'report_path': extraction_report_path,
            'csv_path': extracted_csv_path,
            'extractor': extractor
        }

        print(f"\n✓ Reporte generado: {extraction_report_path}")
        print(f"✓ Términos guardados: {extracted_csv_path}")

        return self.extracted_results

    def evaluate_precision(self) -> Dict:
        """
        Evalúa precisión de términos extraídos (Parte 3).

        Returns:
            Resultados de evaluación de precisión
        """
        print("\n" + "="*70)
        print("PARTE 3: EVALUACIÓN DE PRECISIÓN")
        print("="*70)

        results = {}

        # Evaluar cada método
        methods = [
            ('RAKE', self.extracted_results['rake_terms']),
            ('TextRank', self.extracted_results['textrank_terms']),
            ('Combined', self.extracted_results['combined_terms'])
        ]

        for method_name, extracted_terms in methods:
            print(f"\n[Evaluando {method_name}]")

            evaluator = TermPrecisionEvaluator(
                self.predefined_terms,
                extracted_terms
            )

            # Calcular métricas
            evaluator.calculate_similarity_matrix()
            matches = evaluator.identify_matches(threshold=0.70)
            metrics = evaluator.calculate_metrics()

            # Generar reporte
            report_path = os.path.join(
                self.reports_dir,
                f'evaluation_{method_name.lower()}.md'
            )
            evaluator.generate_evaluation_report(report_path, self.abstracts)

            results[method_name] = {
                'metrics': metrics,
                'matches': matches,
                'report_path': report_path,
                'evaluator': evaluator
            }

            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall:    {metrics['recall']:.2%}")
            print(f"  F1-Score:  {metrics['f1_score']:.2%}")

        self.evaluation_results = results

        # Guardar métricas en JSON
        metrics_json_path = os.path.join(self.data_dir, 'evaluation_metrics.json')

        metrics_data = {
            method: {
                'precision': results[method]['metrics']['precision'],
                'recall': results[method]['metrics']['recall'],
                'f1_score': results[method]['metrics']['f1_score'],
                'coverage': results[method]['metrics']['coverage'],
                'n_exact_matches': results[method]['metrics']['n_exact_matches'],
                'n_partial_matches': results[method]['metrics']['n_partial_matches'],
                'n_novel_terms': results[method]['metrics']['n_novel_terms']
            }
            for method in results.keys()
        }

        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"\n✓ Métricas guardadas: {metrics_json_path}")

        return results

    def create_comparative_visualizations(self):
        """
        Crea visualizaciones comparativas entre métodos.
        """
        print("\n" + "="*70)
        print("GENERANDO VISUALIZACIONES COMPARATIVAS")
        print("="*70)

        # 1. Comparación de métricas entre métodos
        self._create_metrics_comparison_chart()

        # 2. Distribución de frecuencias de términos predefinidos
        self._create_frequency_distribution_chart()

        # 3. Overlap entre métodos (Venn de 3 conjuntos)
        self._create_methods_overlap_chart()

        # 4. Top términos por método
        self._create_top_terms_comparison()

        print("\n✓ Todas las visualizaciones generadas")

    def _create_metrics_comparison_chart(self):
        """Gráfico comparativo de métricas P/R/F1 entre métodos."""
        methods = list(self.evaluation_results.keys())
        metrics_names = ['Precision', 'Recall', 'F1-Score']

        data = {
            metric: [
                self.evaluation_results[method]['metrics'][metric.lower().replace('-', '_')]
                for method in methods
            ]
            for metric in metrics_names
        }

        # Crear gráfico de barras agrupadas
        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics_names):
            offset = width * (i - 1)
            bars = ax.bar(
                x + offset,
                data[metric],
                width,
                label=metric,
                alpha=0.8
            )

            # Añadir valores encima de las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2%}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        ax.set_xlabel('Método de Extracción', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Métricas entre Métodos de Extracción',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.viz_dir, 'metrics_comparison.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Comparación de métricas: {path}")

    def _create_frequency_distribution_chart(self):
        """Distribución de frecuencias de términos predefinidos."""
        frequencies = list(self.predefined_results['frequencies'].values())
        frequencies.sort(reverse=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico 1: Top 15 términos
        top_15 = sorted(
            self.predefined_results['frequencies'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        terms_15 = [t[0] for t in top_15]
        freqs_15 = [t[1] for t in top_15]

        bars = ax1.barh(range(len(terms_15)), freqs_15, color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(terms_15)))
        ax1.set_yticklabels(terms_15, fontsize=9)
        ax1.set_xlabel('Frecuencia', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Términos Predefinidos', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Añadir valores
        for i, (bar, freq) in enumerate(zip(bars, freqs_15)):
            ax1.text(freq, i, f' {freq}', va='center', fontsize=8)

        # Gráfico 2: Distribución general
        ax2.hist(frequencies, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Frecuencia', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Número de Términos', fontsize=11, fontweight='bold')
        ax2.set_title('Distribución de Frecuencias', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.viz_dir, 'frequency_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Distribución de frecuencias: {path}")

    def _create_methods_overlap_chart(self):
        """Diagrama de overlap entre métodos de extracción."""
        from matplotlib_venn import venn3

        rake_set = set(self.extracted_results['rake_terms'])
        textrank_set = set(self.extracted_results['textrank_terms'])
        predefined_set = set(self.predefined_terms)

        fig, ax = plt.subplots(figsize=(10, 8))

        venn = venn3(
            [rake_set, textrank_set, predefined_set],
            set_labels=('RAKE', 'TextRank', 'Predefinidos'),
            ax=ax
        )

        # Personalizar colores
        if venn.get_patch_by_id('100'):
            venn.get_patch_by_id('100').set_color('#ff9999')
        if venn.get_patch_by_id('010'):
            venn.get_patch_by_id('010').set_color('#99ccff')
        if venn.get_patch_by_id('001'):
            venn.get_patch_by_id('001').set_color('#99ff99')

        plt.title('Overlap entre Métodos de Extracción y Términos Predefinidos',
                  fontsize=14, fontweight='bold', pad=20)

        path = os.path.join(self.viz_dir, 'methods_overlap.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Overlap entre métodos: {path}")

    def _create_top_terms_comparison(self):
        """Tabla visual de top términos por método."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))

        methods = [
            ('RAKE', self.extracted_results['rake_terms'][:10]),
            ('TextRank', self.extracted_results['textrank_terms'][:10]),
            ('Predefinidos\n(Top)', self.predefined_results['top_10_terms'])
        ]

        for ax, (method_name, terms) in zip(axes, methods):
            # Crear tabla
            table_data = [[f"{i+1}. {term}"] for i, term in enumerate(terms)]

            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=table_data,
                cellLoc='left',
                loc='center',
                colWidths=[0.9]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Estilizar celdas
            for i in range(len(table_data)):
                cell = table[(i, 0)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('#cccccc')

            ax.set_title(f'Top 10 - {method_name}',
                        fontsize=12, fontweight='bold', pad=10)

        plt.suptitle('Comparación de Top 10 Términos por Método',
                     fontsize=14, fontweight='bold', y=0.98)

        path = os.path.join(self.viz_dir, 'top_terms_comparison.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Comparación de top términos: {path}")

    def generate_master_report(self):
        """
        Genera reporte maestro consolidando todos los análisis.
        """
        print("\n" + "="*70)
        print("GENERANDO REPORTE MAESTRO")
        print("="*70)

        report_path = os.path.join(self.reports_dir, 'term_analysis_report.md')

        content = self._build_master_report_content()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n✓ Reporte maestro generado: {report_path}")

        return report_path

    def _build_master_report_content(self) -> str:
        """Construye contenido del reporte maestro."""

        # Metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# Reporte de Análisis Completo de Términos

**Fecha de generación**: {timestamp}
**Corpus**: {self.data['metadata']['query']}
**Rango temporal**: {self.data['metadata']['date_range']['start']} - {self.data['metadata']['date_range']['end']}

---

## Resumen Ejecutivo

### Estadísticas del Corpus

| Métrica | Valor |
|---------|-------|
| Total de Papers | {len(self.data['papers'])} |
| Abstracts Analizados | {len(self.abstracts)} |
| Términos Predefinidos | {len(self.predefined_terms)} |
| Periodo | {self.data['metadata']['date_range']['start']} - {self.data['metadata']['date_range']['end']} |

### Comparación de Métodos de Extracción

"""

        # Tabla comparativa de métricas
        content += "| Método | Precision | Recall | F1-Score | Coverage |\n"
        content += "|--------|-----------|--------|----------|----------|\n"

        for method in ['RAKE', 'TextRank', 'Combined']:
            metrics = self.evaluation_results[method]['metrics']
            content += (f"| **{method}** | {metrics['precision']:.2%} | "
                       f"{metrics['recall']:.2%} | {metrics['f1_score']:.2%} | "
                       f"{metrics['coverage']:.1f}% |\n")

        # Mejor método
        best_method = max(
            self.evaluation_results.keys(),
            key=lambda m: self.evaluation_results[m]['metrics']['f1_score']
        )
        best_f1 = self.evaluation_results[best_method]['metrics']['f1_score']

        content += f"\n**Mejor método**: {best_method} (F1-Score: {best_f1:.2%})\n"

        content += "\n---\n\n## Visualizaciones\n\n"
        content += "### Comparación de Métricas\n"
        content += "![Comparación de Métricas](../visualizations/metrics_comparison.png)\n\n"

        content += "### Distribución de Frecuencias\n"
        content += "![Distribución de Frecuencias](../visualizations/frequency_distribution.png)\n\n"

        content += "### Overlap entre Métodos\n"
        content += "![Overlap entre Métodos](../visualizations/methods_overlap.png)\n\n"

        content += "### Top Términos por Método\n"
        content += "![Top Términos](../visualizations/top_terms_comparison.png)\n\n"

        content += "---\n\n## Parte 1: Análisis de Términos Predefinidos\n\n"

        # Top términos
        content += "### Top 10 Términos Más Frecuentes\n\n"
        content += "| Posición | Término | Frecuencia |\n"
        content += "|----------|---------|------------|\n"

        top_10 = sorted(
            self.predefined_results['frequencies'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for i, (term, freq) in enumerate(top_10, 1):
            content += f"| {i} | {term} | {freq} |\n"

        content += f"\n📄 [Ver reporte completo](predefined_terms_report.md)\n"

        content += "\n---\n\n## Parte 2: Extracción Automática\n\n"

        # Términos extraídos por método
        for method in ['RAKE', 'TextRank', 'Combined']:
            content += f"### Método: {method}\n\n"

            if method == 'RAKE':
                terms = self.extracted_results['rake_terms'][:5]
            elif method == 'TextRank':
                terms = self.extracted_results['textrank_terms'][:5]
            else:
                terms = self.extracted_results['combined_terms'][:5]

            content += "**Top 5 términos**:\n"
            for i, term in enumerate(terms, 1):
                content += f"{i}. {term}\n"
            content += "\n"

        content += f"📄 [Ver reporte completo de extracción](extracted_terms_report.md)\n"

        content += "\n---\n\n## Parte 3: Evaluación de Precisión\n\n"

        # Detalles por método
        for method in ['RAKE', 'TextRank', 'Combined']:
            metrics = self.evaluation_results[method]['metrics']
            matches = self.evaluation_results[method]['matches']

            content += f"### Evaluación: {method}\n\n"
            content += "| Métrica | Valor |\n"
            content += "|---------|-------|\n"
            content += f"| Precision | {metrics['precision']:.2%} |\n"
            content += f"| Recall | {metrics['recall']:.2%} |\n"
            content += f"| F1-Score | {metrics['f1_score']:.2%} |\n"
            content += f"| Exact Matches | {metrics['n_exact_matches']} |\n"
            content += f"| Partial Matches | {metrics['n_partial_matches']} |\n"
            content += f"| Novel Terms | {metrics['n_novel_terms']} |\n"
            content += "\n"

            content += f"📄 [Ver reporte completo de evaluación](evaluation_{method.lower()}.md)\n\n"

        content += "---\n\n## Conclusiones y Recomendaciones\n\n"

        content += self._generate_conclusions()

        content += "\n---\n\n## Archivos Generados\n\n"
        content += "### Datos\n"
        content += "- `data/predefined_terms_frequencies.csv`: Frecuencias de términos predefinidos\n"
        content += "- `data/extracted_terms_all_methods.csv`: Términos extraídos por todos los métodos\n"
        content += "- `data/evaluation_metrics.json`: Métricas de evaluación en formato JSON\n\n"

        content += "### Reportes\n"
        content += "- `reports/predefined_terms_report.md`: Análisis detallado de términos predefinidos\n"
        content += "- `reports/extracted_terms_report.md`: Reporte de extracción automática\n"
        content += "- `reports/evaluation_rake.md`: Evaluación del método RAKE\n"
        content += "- `reports/evaluation_textrank.md`: Evaluación del método TextRank\n"
        content += "- `reports/evaluation_combined.md`: Evaluación del método combinado\n\n"

        content += "### Visualizaciones\n"
        content += "- `visualizations/metrics_comparison.png`: Comparación de métricas entre métodos\n"
        content += "- `visualizations/frequency_distribution.png`: Distribución de frecuencias\n"
        content += "- `visualizations/methods_overlap.png`: Diagrama de Venn de overlap\n"
        content += "- `visualizations/top_terms_comparison.png`: Comparación de top términos\n"

        content += "\n---\n\n*Reporte generado automáticamente por TermAnalysisPipeline*\n"

        return content

    def _generate_conclusions(self) -> str:
        """Genera conclusiones automáticas basadas en resultados."""

        conclusions = "### Hallazgos Principales\n\n"

        # Mejor método
        best_method = max(
            self.evaluation_results.keys(),
            key=lambda m: self.evaluation_results[m]['metrics']['f1_score']
        )
        best_metrics = self.evaluation_results[best_method]['metrics']

        conclusions += f"1. **Mejor Método**: {best_method} demostró el mejor desempeño general "
        conclusions += f"con F1-Score de {best_metrics['f1_score']:.2%}.\n\n"

        # Análisis de precision/recall
        for method in self.evaluation_results.keys():
            metrics = self.evaluation_results[method]['metrics']

            if metrics['precision'] > 0.7 and metrics['recall'] > 0.7:
                conclusions += f"2. **{method}**: Balance excelente entre precision y recall, "
                conclusions += "adecuado para uso en producción.\n\n"
            elif metrics['precision'] > metrics['recall']:
                conclusions += f"2. **{method}**: Alta precision ({metrics['precision']:.2%}) "
                conclusions += "pero recall moderado, extrae pocos términos pero de alta calidad.\n\n"
            elif metrics['recall'] > metrics['precision']:
                conclusions += f"2. **{method}**: Alto recall ({metrics['recall']:.2%}) "
                conclusions += "pero precision moderada, captura muchos conceptos con algo de ruido.\n\n"

        # Novel terms
        total_novel = sum(
            self.evaluation_results[m]['metrics']['n_novel_terms']
            for m in self.evaluation_results.keys()
        )

        if total_novel > 20:
            conclusions += f"3. **Descubrimiento**: Se identificaron {total_novel} términos nuevos "
            conclusions += "que no estaban en los términos predefinidos, sugiriendo conceptos emergentes.\n\n"

        conclusions += "\n### Recomendaciones\n\n"

        if best_metrics['f1_score'] >= 0.7:
            conclusions += f"- Usar **{best_method}** como método principal de extracción.\n"
        else:
            conclusions += "- Considerar ajustar parámetros o combinar múltiples métodos.\n"

        conclusions += "- Revisar términos nuevos de alta frecuencia para actualizar términos predefinidos.\n"
        conclusions += "- Analizar términos predefinidos no encontrados para entender por qué no aparecen.\n"

        return conclusions


def run_complete_analysis(unified_data_path: str, output_dir: str):
    """
    Ejecuta análisis completo de términos.

    Args:
        unified_data_path: Ruta al archivo unified_abstracts.json
        output_dir: Directorio para guardar outputs

    Returns:
        Instancia de TermAnalysisPipeline con todos los resultados
    """
    print("\n" + "="*70)
    print(" PIPELINE COMPLETO DE ANÁLISIS DE TÉRMINOS")
    print("="*70)
    print(f"Input:  {unified_data_path}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Crear pipeline
    pipeline = TermAnalysisPipeline(unified_data_path, output_dir)

    # 1. Cargar datos
    pipeline.load_data()

    # 2. Analizar términos predefinidos
    pipeline.analyze_predefined_terms()

    # 3. Extraer términos automáticamente
    pipeline.extract_terms_automatically()

    # 4. Evaluar precisión
    pipeline.evaluate_precision()

    # 5. Crear visualizaciones comparativas
    pipeline.create_comparative_visualizations()

    # 6. Generar reporte maestro
    pipeline.generate_master_report()

    print("\n" + "="*70)
    print(" ANÁLISIS COMPLETO FINALIZADO")
    print("="*70)
    print(f"\n✓ Todos los resultados guardados en: {output_dir}")
    print(f"\n📊 Archivos generados:")
    print(f"  • Reporte maestro: {output_dir}/reports/term_analysis_report.md")
    print(f"  • Datos CSV: {output_dir}/data/")
    print(f"  • Visualizaciones: {output_dir}/visualizations/")
    print(f"  • Reportes detallados: {output_dir}/reports/")

    return pipeline


def main():
    """Ejemplo de uso del pipeline."""

    # Configuración
    unified_data_path = 'unified_abstracts.json'
    output_dir = 'term_analysis_output'

    # Verificar que existe el archivo
    if not os.path.exists(unified_data_path):
        print(f"\n❌ Error: No se encontró el archivo {unified_data_path}")
        print("\nAsegúrate de:")
        print("  1. Haber ejecutado el buscador académico")
        print("  2. Tener el archivo unified_abstracts.json en el directorio actual")
        return

    # Ejecutar análisis completo
    pipeline = run_complete_analysis(unified_data_path, output_dir)

    print("\n✓ Pipeline ejecutado exitosamente!")


if __name__ == "__main__":
    main()
