"""
Módulo comparador principal que integra todos los algoritmos de similitud.
"""

import json
import time
import psutil
import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .levenshtein import LevenshteinComparator
from .tfidf_cosine import TFIDFCosineComparator
from .jaccard import JaccardComparator
from .ngram import NGramComparator
from .sbert import SBERTComparator
from .bert import BERTComparator


class SimilarityComparator:
    """
    Clase principal para comparar abstracts con todos los algoritmos.
    """

    def __init__(self, unified_data_path: str):
        """
        Carga datos y inicializa modelos de IA (caché).

        Args:
            unified_data_path: Ruta al archivo JSON con datos unificados
        """
        print("🔧 Inicializando SimilarityComparator...")

        # Cargar datos
        with open(unified_data_path, 'r', encoding='utf-8') as f:
            self.unified_data = json.load(f)

        print(f"✓ Datos cargados: {len(self.unified_data)} artículos")

        # Inicializar algoritmos básicos
        self.levenshtein = LevenshteinComparator()
        self.tfidf_cosine = TFIDFCosineComparator()
        self.jaccard = JaccardComparator()
        self.ngram = NGramComparator(n=3)

        # Inicializar modelos de IA con caché
        print("🤖 Inicializando modelos de IA...")
        self.sbert = SBERTComparator(cache_dir='./cache/sbert')
        self.bert = BERTComparator(cache_dir='./cache/bert')

        print("✓ SimilarityComparator inicializado correctamente")

    def select_articles(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Selecciona artículos por ID y extrae abstracts.

        Args:
            article_ids: Lista de IDs de artículos a seleccionar

        Returns:
            Lista de diccionarios con información de artículos seleccionados
        """
        selected_articles = []

        for article_id in article_ids:
            # Buscar artículo por ID
            article = next((a for a in self.unified_data if a['id'] == article_id), None)

            if article:
                selected_articles.append({
                    'id': article['id'],
                    'title': article['title'],
                    'abstract': article['abstract'],
                    'source': article.get('source', 'unknown')
                })
            else:
                print(f"⚠️ Advertencia: Artículo '{article_id}' no encontrado")

        return selected_articles

    def compare_all_algorithms(self, abstracts: List[str]) -> Dict[str, Any]:
        """
        Ejecuta los 6 algoritmos y retorna resultados.

        Args:
            abstracts: Lista de abstracts a comparar

        Returns:
            Diccionario con resultados de todos los algoritmos
        """
        print(f"\n🔬 Comparando {len(abstracts)} abstracts con 6 algoritmos...")

        results = {
            'execution_times': {},
            'memory_usage': {},
            'similarities': {}
        }

        # Obtener uso de memoria inicial
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        algorithms = [
            ('levenshtein', self.levenshtein),
            ('tfidf_cosine', self.tfidf_cosine),
            ('jaccard', self.jaccard),
            ('ngram', self.ngram),
            ('sbert', self.sbert),
            ('bert', self.bert)
        ]

        for algo_name, algo_instance in algorithms:
            print(f"\n⚙️  Ejecutando {algo_name.upper()}...")

            # Medir tiempo de ejecución
            start_time = time.time()

            try:
                similarity_matrix = algo_instance.compare_multiple(abstracts)
                execution_time = time.time() - start_time

                # Medir uso de memoria
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory

                # Guardar resultados
                results['similarities'][algo_name] = similarity_matrix
                results['execution_times'][algo_name] = execution_time
                results['memory_usage'][algo_name] = memory_used

                print(f"   ✓ Completado en {execution_time:.3f}s")
                print(f"   📊 Memoria usada: {memory_used:.2f} MB")

            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
                results['similarities'][algo_name] = None
                results['execution_times'][algo_name] = None
                results['memory_usage'][algo_name] = None

        print("\n✅ Comparación completada")
        return results

    def visualize_results(self, results: Dict[str, Any], output_dir: str):
        """
        Genera visualizaciones de los resultados.

        Args:
            results: Diccionario con resultados de compare_all_algorithms
            output_dir: Directorio donde guardar las visualizaciones
        """
        print(f"\n📊 Generando visualizaciones en '{output_dir}'...")

        # Crear directorio de salida
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Heatmaps de similitud para cada algoritmo
        print("   📈 Generando heatmaps de similitud...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Matrices de Similitud por Algoritmo', fontsize=16, fontweight='bold')

        algorithms = ['levenshtein', 'tfidf_cosine', 'jaccard', 'ngram', 'sbert', 'bert']

        for idx, algo_name in enumerate(algorithms):
            ax = axes[idx // 3, idx % 3]

            similarity_matrix = results['similarities'].get(algo_name)

            if similarity_matrix is not None:
                sns.heatmap(
                    similarity_matrix,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    vmin=0,
                    vmax=1,
                    ax=ax,
                    cbar_kws={'label': 'Similitud'}
                )
                ax.set_title(f'{algo_name.upper()}', fontweight='bold')
                ax.set_xlabel('Artículo')
                ax.set_ylabel('Artículo')
            else:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{algo_name.upper()} (Error)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Gráfico comparativo de tiempos de ejecución
        print("   ⏱️  Generando gráfico de tiempos...")
        fig, ax = plt.subplots(figsize=(12, 6))

        algo_names = []
        times = []

        for algo_name in algorithms:
            exec_time = results['execution_times'].get(algo_name)
            if exec_time is not None:
                algo_names.append(algo_name.upper())
                times.append(exec_time)

        bars = ax.bar(algo_names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        ax.set_ylabel('Tiempo (segundos)', fontweight='bold')
        ax.set_title('Tiempo de Ejecución por Algoritmo', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}s',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/execution_times.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Gráfico de uso de memoria
        print("   💾 Generando gráfico de memoria...")
        fig, ax = plt.subplots(figsize=(12, 6))

        algo_names = []
        memory = []

        for algo_name in algorithms:
            mem_usage = results['memory_usage'].get(algo_name)
            if mem_usage is not None:
                algo_names.append(algo_name.upper())
                memory.append(mem_usage)

        bars = ax.bar(algo_names, memory, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'])
        ax.set_ylabel('Memoria (MB)', fontweight='bold')
        ax.set_title('Uso de Memoria por Algoritmo', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f} MB',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Tabla comparativa de resultados
        print("   📋 Generando tabla comparativa...")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        # Preparar datos de la tabla
        table_data = []
        table_data.append(['Algoritmo', 'Tiempo (s)', 'Memoria (MB)', 'Similitud Media', 'Similitud Máx', 'Similitud Mín'])

        for algo_name in algorithms:
            exec_time = results['execution_times'].get(algo_name)
            mem_usage = results['memory_usage'].get(algo_name)
            sim_matrix = results['similarities'].get(algo_name)

            if exec_time is not None and sim_matrix is not None:
                # Calcular estadísticas (excluyendo diagonal)
                mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
                sim_values = sim_matrix[mask]

                table_data.append([
                    algo_name.upper(),
                    f'{exec_time:.3f}',
                    f'{mem_usage:.2f}',
                    f'{np.mean(sim_values):.3f}',
                    f'{np.max(sim_values):.3f}',
                    f'{np.min(sim_values):.3f}'
                ])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Estilo para encabezado
        for i in range(6):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Colores alternos para filas
        for i in range(1, len(table_data)):
            color = '#ECF0F1' if i % 2 == 0 else 'white'
            for j in range(6):
                table[(i, j)].set_facecolor(color)

        plt.title('Tabla Comparativa de Resultados', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{output_dir}/comparison_table.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualizaciones guardadas en '{output_dir}'")

    def generate_detailed_report(self, results: Dict[str, Any], output_path: str,
                                 article_info: List[Dict[str, Any]] = None):
        """
        Genera reporte Markdown con análisis detallado.

        Args:
            results: Diccionario con resultados de compare_all_algorithms
            output_path: Ruta donde guardar el reporte
            article_info: Información de los artículos comparados (opcional)
        """
        print(f"\n📝 Generando reporte detallado en '{output_path}'...")

        report = []
        report.append("# Reporte de Comparación de Algoritmos de Similitud\n")
        report.append(f"**Fecha de generación:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")

        # Información de artículos
        if article_info:
            report.append("## 📚 Artículos Analizados\n")
            for idx, article in enumerate(article_info, 1):
                report.append(f"### {idx}. {article['title']}\n")
                report.append(f"- **ID:** `{article['id']}`\n")
                report.append(f"- **Fuente:** {article.get('source', 'N/A')}\n")
                report.append(f"- **Abstract:** {article['abstract'][:200]}...\n")
            report.append("\n---\n")

        # Resumen de resultados
        report.append("## 📊 Resumen de Resultados\n")
        report.append("| Algoritmo | Tiempo (s) | Memoria (MB) | Similitud Media | Similitud Máx | Similitud Mín |\n")
        report.append("|-----------|------------|--------------|-----------------|---------------|---------------|\n")

        algorithms = ['levenshtein', 'tfidf_cosine', 'jaccard', 'ngram', 'sbert', 'bert']

        for algo_name in algorithms:
            exec_time = results['execution_times'].get(algo_name)
            mem_usage = results['memory_usage'].get(algo_name)
            sim_matrix = results['similarities'].get(algo_name)

            if exec_time is not None and sim_matrix is not None:
                mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
                sim_values = sim_matrix[mask]

                report.append(
                    f"| **{algo_name.upper()}** | "
                    f"{exec_time:.3f} | "
                    f"{mem_usage:.2f} | "
                    f"{np.mean(sim_values):.3f} | "
                    f"{np.max(sim_values):.3f} | "
                    f"{np.min(sim_values):.3f} |\n"
                )

        report.append("\n---\n")

        # Descripción de cada algoritmo
        report.append("## 🔬 Descripción de Algoritmos\n")

        algo_descriptions = {
            'levenshtein': {
                'nombre': 'Distancia de Levenshtein',
                'descripcion': 'Mide el número mínimo de operaciones de edición (inserción, eliminación, sustitución) necesarias para transformar un texto en otro.',
                'ventajas': ['Fácil de entender', 'No requiere preprocesamiento', 'Útil para textos cortos'],
                'desventajas': ['Muy lento para textos largos', 'No considera semántica', 'Sensible a orden de palabras'],
                'uso_recomendado': 'Detección de errores tipográficos, corrección de ortografía, textos muy cortos'
            },
            'tfidf_cosine': {
                'nombre': 'TF-IDF + Similitud del Coseno',
                'descripcion': 'Convierte textos en vectores usando TF-IDF (Term Frequency-Inverse Document Frequency) y calcula similitud mediante ángulo entre vectores.',
                'ventajas': ['Rápido y escalable', 'Considera importancia de términos', 'Independiente de longitud'],
                'desventajas': ['No captura semántica', 'Ignora orden de palabras', 'Requiere corpus representativo'],
                'uso_recomendado': 'Búsqueda de documentos, clasificación de textos, recuperación de información'
            },
            'jaccard': {
                'nombre': 'Índice de Jaccard',
                'descripcion': 'Mide similitud como la razón entre la intersección y la unión de conjuntos de palabras.',
                'ventajas': ['Simple y rápido', 'Intuitivo', 'Útil para conjuntos'],
                'desventajas': ['No considera frecuencia', 'Ignora importancia de términos', 'No captura semántica'],
                'uso_recomendado': 'Comparación de etiquetas, categorías, palabras clave'
            },
            'ngram': {
                'nombre': 'Similitud de N-gramas',
                'descripcion': 'Compara secuencias de N caracteres consecutivos, capturando patrones locales en el texto.',
                'ventajas': ['Robusto a errores', 'Captura patrones locales', 'Útil para diferentes idiomas'],
                'desventajas': ['Sensible a valor de N', 'No captura semántica', 'Puede ser lento'],
                'uso_recomendado': 'Detección de plagio, comparación multilingüe, textos con errores'
            },
            'sbert': {
                'nombre': 'Sentence-BERT (S-BERT)',
                'descripcion': 'Modelo transformer optimizado para generar embeddings de oraciones semánticamente significativos.',
                'ventajas': ['Captura semántica profunda', 'Rápido en inferencia', 'Multilingüe'],
                'desventajas': ['Requiere GPU (recomendado)', 'Mayor uso de memoria', 'Modelo grande'],
                'uso_recomendado': 'Búsqueda semántica, agrupación de documentos, recomendación de contenido'
            },
            'bert': {
                'nombre': 'BERT (Bidirectional Encoder)',
                'descripcion': 'Modelo transformer que analiza texto bidireccionalmente, capturando contexto completo.',
                'ventajas': ['Máxima comprensión semántica', 'Estado del arte', 'Contexto bidireccional'],
                'desventajas': ['Muy lento', 'Alto uso de memoria', 'Requiere GPU'],
                'uso_recomendado': 'Análisis profundo de similitud, cuando precisión es crítica, datasets pequeños'
            }
        }

        for algo_name in algorithms:
            info = algo_descriptions[algo_name]
            report.append(f"### {info['nombre']}\n")
            report.append(f"**Descripción:** {info['descripcion']}\n\n")
            report.append("**Ventajas:**\n")
            for ventaja in info['ventajas']:
                report.append(f"- ✓ {ventaja}\n")
            report.append("\n**Desventajas:**\n")
            for desventaja in info['desventajas']:
                report.append(f"- ✗ {desventaja}\n")
            report.append(f"\n**Uso recomendado:** {info['uso_recomendado']}\n\n")

        report.append("---\n")

        # Matrices de similitud
        report.append("## 📈 Matrices de Similitud\n")

        for algo_name in algorithms:
            sim_matrix = results['similarities'].get(algo_name)

            if sim_matrix is not None:
                report.append(f"### {algo_name.upper()}\n")
                report.append("```\n")

                # Crear encabezado
                n = sim_matrix.shape[0]
                header = "     " + "".join([f"  Art{i+1} " for i in range(n)])
                report.append(header + "\n")

                # Crear filas
                for i in range(n):
                    row = f"Art{i+1} "
                    for j in range(n):
                        row += f" {sim_matrix[i, j]:.3f} "
                    report.append(row + "\n")

                report.append("```\n\n")

        report.append("---\n")

        # Análisis y recomendaciones
        report.append("## 💡 Análisis y Recomendaciones\n")

        # Encontrar algoritmo más rápido
        fastest_algo = min(results['execution_times'].items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
        report.append(f"- **Algoritmo más rápido:** {fastest_algo[0].upper()} ({fastest_algo[1]:.3f}s)\n")

        # Encontrar algoritmo con menor uso de memoria
        lowest_memory = min(results['memory_usage'].items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
        report.append(f"- **Menor uso de memoria:** {lowest_memory[0].upper()} ({lowest_memory[1]:.2f} MB)\n")

        # Calcular similitud promedio
        avg_similarities = {}
        for algo_name in algorithms:
            sim_matrix = results['similarities'].get(algo_name)
            if sim_matrix is not None:
                mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
                avg_similarities[algo_name] = np.mean(sim_matrix[mask])

        if avg_similarities:
            highest_sim = max(avg_similarities.items(), key=lambda x: x[1])
            report.append(f"- **Mayor similitud promedio:** {highest_sim[0].upper()} ({highest_sim[1]:.3f})\n")

        report.append("\n### Recomendaciones Generales\n")
        report.append("1. **Para aplicaciones en tiempo real:** Use TF-IDF o Jaccard\n")
        report.append("2. **Para máxima precisión semántica:** Use S-BERT o BERT\n")
        report.append("3. **Para recursos limitados:** Use Jaccard o N-grams\n")
        report.append("4. **Para textos cortos:** Use Levenshtein o Jaccard\n")
        report.append("5. **Para textos largos:** Use TF-IDF o S-BERT\n")

        # Guardar reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))

        print(f"✓ Reporte guardado en '{output_path}'")


if __name__ == "__main__":
    # Ejemplo de uso
    comparator = SimilarityComparator('data/unified_articles.json')

    # Seleccionar artículos (ejemplo con IDs)
    selected = comparator.select_articles(['article_1', 'article_2', 'article_3'])
    abstracts = [art['abstract'] for art in selected]

    # Comparar con todos los algoritmos
    results = comparator.compare_all_algorithms(abstracts)

    # Generar visualizaciones
    comparator.visualize_results(results, 'output/visualizations')

    # Generar reporte
    comparator.generate_detailed_report(results, 'output/report.md', selected)
