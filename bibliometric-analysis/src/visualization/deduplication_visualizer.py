"""
Generador de Visualizaciones para Análisis de Deduplicación
Genera gráficas estadísticas de duplicados detectados
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class DeduplicationVisualizer:
    """Genera gráficas de estadísticas de deduplicación"""
    
    def __init__(self, data_dir='data/duplicates', output_dir='output/deduplication'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_duplicates_csv(self):
        """Carga el CSV de duplicados más reciente"""
        csv_files = list(self.data_dir.glob('duplicates_log_*.csv'))
        if not csv_files:
            print("⚠ No se encontraron archivos CSV de duplicados")
            return None
        
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"📄 Cargando: {latest_csv.name}")
        df = pd.read_csv(latest_csv)
        
        if df.empty:
            print("⚠ El archivo CSV está vacío")
            return None
            
        return df
    
    def load_report_json(self):
        """Carga el reporte JSON si existe"""
        report_path = self.data_dir / 'duplicates_report.json'
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def create_sample_data(self):
        """Crea datos de ejemplo para demostración"""
        print("📊 Generando datos de ejemplo...")
        return {
            "summary": {
                "original_count": 1523,
                "duplicates_count": 247,
                "clean_count": 1276,
                "duplicate_rate": 16.22,
                "processing_time": "00:02:34"
            },
            "by_source": {
                "ACM Digital Library": 598,
                "ScienceDirect": 678,
                "Duplicados entre fuentes": 56
            },
            "by_detection_method": {
                "DOI Exacto": 89,
                "Similitud de Título": 102,
                "Autores + Año": 56
            },
            "algorithms": {
                "Levenshtein": {
                    "threshold": 0.85,
                    "duplicates_found": 102,
                    "avg_similarity": 0.91
                },
                "Jaro-Winkler": {
                    "threshold": 0.90,
                    "duplicates_found": 89,
                    "avg_similarity": 0.94
                },
                "Jaccard": {
                    "threshold": 0.80,
                    "duplicates_found": 56,
                    "avg_similarity": 0.87
                }
            }
        }
    
    def plot_summary_statistics(self, report):
        """Gráfica 1: Resumen Original vs Duplicados vs Únicos"""
        if not report:
            return
        
        summary = report['summary']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Registros\nOriginales', 'Duplicados\nDetectados', 'Registros\nÚnicos']
        values = [
            summary['original_count'],
            summary['duplicates_count'],
            summary['clean_count']
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Añadir valores encima de las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                   f'{value:,}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Número de Artículos', fontsize=13, fontweight='bold')
        ax.set_title('Resumen de Deduplicación de Artículos', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir información adicional
        plt.text(0.98, 0.02, f"Tasa de duplicados: {summary['duplicate_rate']:.2f}%",
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / '1_summary_statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def plot_duplicate_rate_pie(self, report):
        """Gráfica 2: Gráfica de pastel - Tasa de duplicados"""
        if not report:
            return
        
        summary = report['summary']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        sizes = [summary['clean_count'], summary['duplicates_count']]
        labels = [
            f"Registros Únicos\n{summary['clean_count']:,} artículos\n({100-summary['duplicate_rate']:.1f}%)",
            f"Duplicados\n{summary['duplicates_count']:,} artículos\n({summary['duplicate_rate']:.1f}%)"
        ]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.15)
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=90,
            textprops={'fontsize': 13, 'fontweight': 'bold'}
        )
        
        # Mejorar el estilo de los porcentajes
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(16)
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribución de Artículos Duplicados', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / '2_duplicate_rate_pie.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def plot_duplicates_by_source(self, report):
        """Gráfica 3: Duplicados por fuente"""
        if not report or 'by_source' not in report:
            return
        
        by_source = report['by_source']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sources = list(by_source.keys())
        values = list(by_source.values())
        colors = ['#3498db', '#e67e22', '#9b59b6']
        
        bars = ax.barh(sources, values, color=colors, alpha=0.85, 
                      edgecolor='black', linewidth=1.5)
        
        # Añadir valores al final de las barras
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 15, bar.get_y() + bar.get_height()/2.,
                   f'{value:,} artículos',
                   ha='left', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Número de Artículos', fontsize=13, fontweight='bold')
        ax.set_title('Distribución de Artículos por Fuente de Datos', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, max(values) * 1.2)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '3_duplicates_by_source.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def plot_detection_methods(self, report):
        """Gráfica 4: Duplicados por método de detección"""
        if not report or 'by_detection_method' not in report:
            return
        
        methods = report['by_detection_method']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        labels = list(methods.keys())
        values = list(methods.values())
        colors = ['#e74c3c', '#f39c12', '#3498db']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=1.5)
        
        # Añadir valores encima
        for bar, value in zip(bars, values):
            height = bar.get_height()
            percentage = value / sum(values) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value:,}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Duplicados Detectados', fontsize=13, fontweight='bold')
        ax.set_title('Duplicados por Método de Detección', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '4_detection_methods.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def plot_algorithm_performance(self, report):
        """Gráfica 5: Performance de algoritmos (dos subgráficas)"""
        if not report or 'algorithms' not in report:
            return
        
        algorithms = report['algorithms']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Gráfica 1: Duplicados encontrados
        names = list(algorithms.keys())
        duplicates = [algo['duplicates_found'] for algo in algorithms.values()]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        bars1 = ax1.bar(names, duplicates, color=colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars1, duplicates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value:,}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Duplicados Detectados', fontsize=12, fontweight='bold')
        ax1.set_title('Duplicados por Algoritmo', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, max(duplicates) * 1.2)
        ax1.grid(axis='y', alpha=0.3)
        
        # Gráfica 2: Similitud promedio
        avg_similarities = [algo['avg_similarity'] for algo in algorithms.values()]
        
        bars2 = ax2.bar(names, avg_similarities, color=colors, alpha=0.85,
                       edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars2, avg_similarities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Similitud Promedio', fontsize=12, fontweight='bold')
        ax2.set_title('Similitud Promedio por Algoritmo', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comparativa de Algoritmos de Similitud', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / '5_algorithm_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def plot_algorithm_thresholds(self, report):
        """Gráfica 6: Thresholds de algoritmos"""
        if not report or 'algorithms' not in report:
            return
        
        algorithms = report['algorithms']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(algorithms.keys())
        thresholds = [algo['threshold'] for algo in algorithms.values()]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        bars = ax.bar(names, thresholds, color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, thresholds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Threshold de Similitud', fontsize=13, fontweight='bold')
        ax.set_title('Umbrales de Detección por Algoritmo', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Threshold mínimo (0.80)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '6_algorithm_thresholds.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def generate_summary_report(self, report):
        """Genera reporte de texto con estadísticas"""
        if not report:
            return
        
        summary = report['summary']
        
        report_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║              REPORTE DE DEDUPLICACIÓN DE ARTÍCULOS               ║
╚══════════════════════════════════════════════════════════════════╝

📊 RESUMEN GENERAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Registros originales:      {summary['original_count']:,} artículos
  • Duplicados detectados:      {summary['duplicates_count']:,} artículos
  • Registros únicos:           {summary['clean_count']:,} artículos
  • Tasa de duplicación:        {summary['duplicate_rate']:.2f}%
  • Tiempo de procesamiento:    {summary.get('processing_time', 'N/A')}

"""
        
        if 'by_source' in report:
            report_text += """
📁 DISTRIBUCIÓN POR FUENTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            for source, count in report['by_source'].items():
                report_text += f"  • {source:30} {count:,} artículos\n"
        
        if 'by_detection_method' in report:
            report_text += """
🔍 MÉTODOS DE DETECCIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            for method, count in report['by_detection_method'].items():
                percentage = count / summary['duplicates_count'] * 100
                report_text += f"  • {method:30} {count:,} duplicados ({percentage:.1f}%)\n"
        
        if 'algorithms' in report:
            report_text += """
⚙️  ALGORITMOS DE SIMILITUD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            for algo_name, algo_data in report['algorithms'].items():
                report_text += f"""  {algo_name}:
    - Threshold:          {algo_data['threshold']:.2f}
    - Duplicados:         {algo_data['duplicates_found']:,}
    - Similitud promedio: {algo_data['avg_similarity']:.3f}

"""
        
        report_text += """
╚══════════════════════════════════════════════════════════════════╝
"""
        
        output_path = self.output_dir / 'deduplication_summary.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Reporte de texto guardado: {output_path}")
        print(report_text)
    
    def generate_all_visualizations(self):
        """Genera todas las visualizaciones"""
        print("\n" + "="*70)
        print("  GENERANDO VISUALIZACIONES DE DEDUPLICACIÓN")
        print("="*70 + "\n")
        
        # Intentar cargar datos reales
        report = self.load_report_json()
        
        # Si no hay datos reales, usar datos de ejemplo
        if not report:
            print("⚠ No se encontró reporte JSON real.")
            print("📊 Generando visualizaciones con datos de ejemplo...\n")
            report = self.create_sample_data()
        
        # Generar todas las gráficas
        self.plot_summary_statistics(report)
        self.plot_duplicate_rate_pie(report)
        self.plot_duplicates_by_source(report)
        self.plot_detection_methods(report)
        self.plot_algorithm_performance(report)
        self.plot_algorithm_thresholds(report)
        self.generate_summary_report(report)
        
        print("\n" + "="*70)
        print(f"✅ Todas las visualizaciones guardadas en:")
        print(f"   {self.output_dir.absolute()}")
        print("="*70 + "\n")
        
        print("📊 Archivos generados:")
        print("  1. 1_summary_statistics.png      - Resumen general")
        print("  2. 2_duplicate_rate_pie.png      - Gráfica de pastel")
        print("  3. 3_duplicates_by_source.png    - Por fuente de datos")
        print("  4. 4_detection_methods.png       - Por método de detección")
        print("  5. 5_algorithm_performance.png   - Comparativa de algoritmos")
        print("  6. 6_algorithm_thresholds.png    - Umbrales de detección")
        print("  7. deduplication_summary.txt     - Reporte en texto\n")


if __name__ == "__main__":
    visualizer = DeduplicationVisualizer()
    visualizer.generate_all_visualizations()
