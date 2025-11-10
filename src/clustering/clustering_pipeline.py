"""
Clustering Pipeline Module
Pipeline completo de clustering jerárquico para análisis bibliométrico.

Este módulo integra todos los componentes del análisis de clustering:
- Preprocesamiento de datos
- Vectorización (TF-IDF, Word2Vec, SBERT)
- Cálculo de matrices de distancia
- Clustering jerárquico con múltiples métodos
- Visualización de dendrogramas
- Evaluación y comparación de métodos
- Exportación de resultados

Características:
- Pipeline automatizado end-to-end
- Soporte para múltiples métodos de vectorización
- Evaluación exhaustiva de calidad
- Recomendación automática del mejor método
- Generación de reportes completos
- Exportación de clusters en múltiples formatos

Uso típico:
===========
pipeline = ClusteringPipeline('data/unified_data.csv', 'output/clustering')
results = pipeline.run_complete_analysis(vectorization_method='tfidf')
pipeline.export_clusters(results['best_linkage'], n_clusters=5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Importar componentes del proyecto
from distance_calculator import DistanceCalculator
from hierarchical_clustering import HierarchicalClustering
from dendrogram_visualizer import DendrogramVisualizer
from clustering_evaluator import ClusteringEvaluator

# Sklearn para vectorización
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """
    Pipeline completo de clustering jerárquico para análisis bibliométrico.

    Este pipeline integra todos los pasos del análisis de clustering,
    desde la carga de datos hasta la exportación de resultados, proporcionando
    un flujo automatizado y reproducible.

    Attributes:
        unified_data_path: Ruta al archivo CSV con datos unificados
        output_dir: Directorio base para outputs
        data: DataFrame con los datos cargados
        abstracts: Lista de abstracts procesados
        titles: Lista de títulos de artículos
        vectors: Matriz de vectores (features)
        distance_matrix: Matriz de distancias
        linkage_matrices: Diccionario con matrices de linkage por método
        evaluation_results: Resultados de evaluación
    """

    def __init__(self,
                 unified_data_path: str,
                 output_dir: str = 'output/clustering'):
        """
        Inicializa el pipeline de clustering.

        Args:
            unified_data_path: Ruta al CSV con datos unificados
                             Debe contener columnas: 'Title', 'Abstract'
            output_dir: Directorio para guardar todos los outputs

        Example:
            >>> pipeline = ClusteringPipeline(
            ...     'data/unified_data.csv',
            ...     'output/clustering'
            ... )
        """
        self.unified_data_path = unified_data_path
        self.output_dir = Path(output_dir)

        # Crear directorios de output
        self._create_output_directories()

        # Atributos a inicializar
        self.data = None
        self.abstracts = None
        self.titles = None
        self.vectors = None
        self.distance_matrix = None
        self.linkage_matrices = {}
        self.evaluation_results = None

        logger.info("="*70)
        logger.info("CLUSTERING PIPELINE INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Datos: {unified_data_path}")
        logger.info(f"Output: {output_dir}")

    def run_complete_analysis(self,
                             vectorization_method: str = 'tfidf',
                             distance_metric: str = 'cosine',
                             max_features: int = 500,
                             min_df: int = 2,
                             max_df: float = 0.8) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo de clustering jerárquico.

        Este método ejecuta todo el pipeline desde la carga de datos
        hasta la evaluación y recomendación del mejor método.

        Pipeline completo:
        =================
        1. Carga y preprocesamiento de datos
        2. Vectorización de abstracts (TF-IDF u otros)
        3. Cálculo de matriz de distancia
        4. Clustering jerárquico con 3 métodos (single, complete, average)
        5. Generación de dendrogramas (estáticos e interactivos)
        6. Evaluación exhaustiva de todos los métodos
        7. Recomendación del mejor método
        8. Exportación de resultados y reportes

        Args:
            vectorization_method: Método de vectorización ('tfidf', 'binary', 'count')
            distance_metric: Métrica de distancia ('cosine', 'euclidean', 'manhattan')
            max_features: Número máximo de features para TF-IDF
            min_df: Frecuencia mínima de documento para términos
            max_df: Frecuencia máxima de documento (filtrar stopwords)

        Returns:
            Diccionario con todos los resultados:
            {
                'vectors': matriz de vectores,
                'distance_matrix': matriz de distancia,
                'linkage_matrices': matrices de linkage por método,
                'evaluation': resultados de evaluación,
                'recommendation': recomendación del mejor método,
                'execution_time': tiempo total de ejecución
            }

        Example:
            >>> results = pipeline.run_complete_analysis(
            ...     vectorization_method='tfidf',
            ...     distance_metric='cosine'
            ... )
            >>> print(f"Mejor método: {results['recommendation']['recommended_method']}")
        """
        start_time = time.time()

        logger.info("\n" + "="*70)
        logger.info("INICIANDO ANÁLISIS COMPLETO DE CLUSTERING")
        logger.info("="*70)
        logger.info(f"Vectorización: {vectorization_method}")
        logger.info(f"Distancia: {distance_metric}")

        # 1. Carga y preprocesamiento
        logger.info("\n--- PASO 1: Carga de datos ---")
        self._load_data()

        # 2. Vectorización
        logger.info("\n--- PASO 2: Vectorización ---")
        self._vectorize_abstracts(
            method=vectorization_method,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df
        )

        # 3. Cálculo de distancias
        logger.info("\n--- PASO 3: Cálculo de matriz de distancia ---")
        self._calculate_distances(metric=distance_metric)

        # 4. Clustering jerárquico
        logger.info("\n--- PASO 4: Clustering jerárquico ---")
        self._apply_hierarchical_clustering()

        # 5. Generación de dendrogramas
        logger.info("\n--- PASO 5: Generación de dendrogramas ---")
        self._generate_dendrograms()

        # 6. Evaluación
        logger.info("\n--- PASO 6: Evaluación de métodos ---")
        self._evaluate_methods()

        # 7. Recomendación
        logger.info("\n--- PASO 7: Recomendación del mejor método ---")
        recommendation = self._recommend_best_method()

        # 8. Exportación de resultados
        logger.info("\n--- PASO 8: Exportación de resultados ---")
        self._export_summary_report(recommendation)

        elapsed_time = time.time() - start_time

        logger.info("\n" + "="*70)
        logger.info("ANÁLISIS COMPLETO FINALIZADO")
        logger.info("="*70)
        logger.info(f"Tiempo total: {elapsed_time:.2f} segundos")

        return {
            'vectors': self.vectors,
            'distance_matrix': self.distance_matrix,
            'linkage_matrices': self.linkage_matrices,
            'evaluation': self.evaluation_results,
            'recommendation': recommendation,
            'execution_time': elapsed_time
        }

    def analyze_optimal_clusters(self,
                                 linkage_matrix: np.ndarray,
                                 k_range: range = range(2, 11),
                                 method_name: str = 'best') -> Dict[str, Any]:
        """
        Determina el número óptimo de clusters usando múltiples heurísticas.

        Este método aplica varias técnicas para sugerir el número óptimo
        de clusters:
        1. Elbow method (análisis de distancias de fusión)
        2. Silhouette analysis (maximizar separación)
        3. Gap statistics (mayor salto entre fusiones)

        Args:
            linkage_matrix: Matriz de linkage del método a analizar
            k_range: Rango de números de clusters a evaluar
            method_name: Nombre del método (para labels)

        Returns:
            Diccionario con análisis:
            {
                'optimal_k_silhouette': int,
                'optimal_k_elbow': int,
                'optimal_k_gap': int,
                'silhouette_scores': dict,
                'merge_distances': array,
                'recommendation': int  # Consenso de los métodos
            }

        Example:
            >>> analysis = pipeline.analyze_optimal_clusters(
            ...     Z_average,
            ...     range(2, 11),
            ...     'average'
            ... )
            >>> print(f"Número óptimo sugerido: {analysis['recommendation']}")
        """
        logger.info("\n" + "="*70)
        logger.info(f"ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS - {method_name.upper()}")
        logger.info("="*70)
        logger.info(f"Evaluando rango: {k_range.start} - {k_range.stop-1}")

        results = {}

        # 1. Silhouette Analysis
        logger.info("\n1. Análisis de Silhouette...")
        from sklearn.metrics import silhouette_score
        from scipy.cluster.hierarchy import fcluster

        silhouette_scores = {}
        best_sil_k = None
        best_sil_score = -1

        for k in k_range:
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
            sil_score = silhouette_score(
                self.distance_matrix,
                cluster_labels,
                metric='precomputed'
            )
            silhouette_scores[k] = sil_score

            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_sil_k = k

            logger.info(f"  k={k}: Silhouette = {sil_score:.4f}")

        results['optimal_k_silhouette'] = best_sil_k
        results['silhouette_scores'] = silhouette_scores
        logger.info(f"\nÓptimo según Silhouette: k={best_sil_k} (score={best_sil_score:.4f})")

        # 2. Elbow Method (usando distancias de fusión)
        logger.info("\n2. Elbow Method...")
        merge_distances = linkage_matrix[:, 2]

        # Calcular segunda derivada para encontrar el codo
        last_merges = merge_distances[-(k_range.stop-1):]
        first_diff = np.diff(last_merges)
        second_diff = np.diff(first_diff)

        elbow_idx = np.argmax(np.abs(second_diff)) + 1
        optimal_k_elbow = elbow_idx + k_range.start

        results['optimal_k_elbow'] = optimal_k_elbow
        results['merge_distances'] = merge_distances
        logger.info(f"Óptimo según Elbow: k={optimal_k_elbow}")

        # 3. Gap Statistics (mayor salto entre fusiones)
        logger.info("\n3. Gap Statistics...")
        gaps = np.diff(last_merges)
        max_gap_idx = np.argmax(gaps)
        optimal_k_gap = max_gap_idx + k_range.start + 1

        results['optimal_k_gap'] = optimal_k_gap
        logger.info(f"Óptimo según Gap: k={optimal_k_gap}")

        # Consenso (usar el más común o promedio)
        suggestions = [best_sil_k, optimal_k_elbow, optimal_k_gap]
        # Usar mediana como consenso
        recommended_k = int(np.median(suggestions))

        results['recommendation'] = recommended_k

        logger.info("\n" + "="*70)
        logger.info("RECOMENDACIÓN FINAL")
        logger.info("="*70)
        logger.info(f"Silhouette sugiere: k={best_sil_k}")
        logger.info(f"Elbow sugiere: k={optimal_k_elbow}")
        logger.info(f"Gap sugiere: k={optimal_k_gap}")
        logger.info(f"\nRecomendación (consenso): k={recommended_k}")
        logger.info("="*70)

        # Generar gráfico de análisis
        self._plot_optimal_k_analysis(results, k_range, method_name)

        return results

    def export_clusters(self,
                       linkage_matrix: np.ndarray,
                       n_clusters: int,
                       output_path: Optional[str] = None,
                       include_terms: bool = True) -> pd.DataFrame:
        """
        Exporta la asignación de documentos a clusters en formato CSV.

        El CSV generado incluye:
        - document_id: Índice del documento
        - title: Título del artículo
        - cluster_id: ID del cluster asignado
        - cluster_label: Etiqueta temática del cluster (términos principales)
        - representative_terms: Top términos del cluster

        Args:
            linkage_matrix: Matriz de linkage del método seleccionado
            n_clusters: Número de clusters
            output_path: Ruta del CSV (si None, usa default)
            include_terms: Si True, incluye términos representativos

        Returns:
            DataFrame con asignaciones de clusters

        Example:
            >>> df = pipeline.export_clusters(
            ...     results['linkage_matrices']['average'],
            ...     n_clusters=5,
            ...     output_path='output/clusters.csv'
            ... )
        """
        logger.info("\n" + "="*70)
        logger.info("EXPORTANDO ASIGNACIÓN DE CLUSTERS")
        logger.info("="*70)
        logger.info(f"Número de clusters: {n_clusters}")

        from scipy.cluster.hierarchy import fcluster

        # Obtener asignaciones
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Crear DataFrame
        df = pd.DataFrame({
            'document_id': range(len(self.titles)),
            'title': self.titles,
            'cluster_id': cluster_labels
        })

        # Añadir términos representativos si se solicita
        if include_terms and self.abstracts is not None:
            logger.info("Extrayendo términos representativos por cluster...")

            # Vectorizar abstracts
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                max_df=0.8,
                min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform(self.abstracts)
            feature_names = vectorizer.get_feature_names_out()

            # Extraer términos por cluster
            cluster_terms = {}

            for cluster_id in range(1, n_clusters + 1):
                # Índices de documentos en este cluster
                cluster_indices = np.where(cluster_labels == cluster_id)[0]

                if len(cluster_indices) == 0:
                    cluster_terms[cluster_id] = []
                    continue

                # TF-IDF promedio del cluster
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
                top_indices = cluster_tfidf.argsort()[-5:][::-1]
                top_terms = [feature_names[i] for i in top_indices]

                cluster_terms[cluster_id] = top_terms

                logger.info(f"Cluster {cluster_id}: {', '.join(top_terms)}")

            # Añadir al DataFrame
            df['cluster_label'] = df['cluster_id'].map(
                lambda cid: ', '.join(cluster_terms.get(cid, [])[:3])
            )
            df['representative_terms'] = df['cluster_id'].map(
                lambda cid: '; '.join(cluster_terms.get(cid, []))
            )

        # Ordenar por cluster_id
        df = df.sort_values(['cluster_id', 'document_id'])

        # Guardar
        if output_path is None:
            output_path = self.output_dir / 'results' / f'cluster_assignments_k{n_clusters}.csv'

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"\nAsignaciones exportadas a: {output_path}")
        logger.info(f"Total de documentos: {len(df)}")

        # Mostrar distribución
        cluster_counts = df['cluster_id'].value_counts().sort_index()
        logger.info("\nDistribución de documentos por cluster:")
        for cid, count in cluster_counts.items():
            logger.info(f"  Cluster {cid}: {count} documentos")

        logger.info("="*70)

        return df

    def _create_output_directories(self) -> None:
        """Crea estructura de directorios para outputs."""
        subdirs = [
            'dendrograms',
            'evaluation',
            'results',
            'reports'
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Directorios de output creados en: {self.output_dir}")

    def _load_data(self) -> None:
        """Carga datos del CSV unificado."""
        logger.info(f"Cargando datos desde: {self.unified_data_path}")

        self.data = pd.read_csv(self.unified_data_path)

        # Verificar columnas requeridas
        required_cols = ['Title', 'Abstract']
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")

        # Limpiar valores nulos
        self.data = self.data.dropna(subset=['Abstract'])

        # Extraer abstracts y títulos
        self.abstracts = self.data['Abstract'].tolist()
        self.titles = self.data['Title'].tolist()

        logger.info(f"Datos cargados: {len(self.abstracts)} documentos")

    def _vectorize_abstracts(self,
                            method: str = 'tfidf',
                            max_features: int = 500,
                            min_df: int = 2,
                            max_df: float = 0.8) -> None:
        """Vectoriza abstracts usando el método especificado."""
        logger.info(f"Vectorizando abstracts con método: {method}")

        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 2)  # Incluir bigramas
            )
            self.vectors = vectorizer.fit_transform(self.abstracts).toarray()

        elif method == 'binary':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=min_df,
                max_df=max_df,
                binary=True
            )
            self.vectors = vectorizer.fit_transform(self.abstracts).toarray()

        elif method == 'count':
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=min_df,
                max_df=max_df
            )
            self.vectors = vectorizer.fit_transform(self.abstracts).toarray()

        else:
            raise ValueError(f"Método de vectorización desconocido: {method}")

        # Normalizar vectores
        self.vectors = normalize(self.vectors, norm='l2')

        logger.info(f"Vectores creados: {self.vectors.shape}")

    def _calculate_distances(self, metric: str = 'cosine') -> None:
        """Calcula matriz de distancia."""
        logger.info(f"Calculando matriz de distancia ({metric})...")

        calc = DistanceCalculator()
        self.distance_matrix = calc.calculate_distance_matrix(self.vectors, metric=metric)

        logger.info(f"Matriz de distancia: {self.distance_matrix.shape}")

    def _apply_hierarchical_clustering(self) -> None:
        """Aplica clustering jerárquico con múltiples métodos."""
        logger.info("Aplicando clustering jerárquico...")

        hc = HierarchicalClustering(self.distance_matrix, self.titles)

        # Ejecutar los 3 métodos principales
        self.linkage_matrices['single'] = hc.single_linkage()
        self.linkage_matrices['complete'] = hc.complete_linkage()
        self.linkage_matrices['average'] = hc.average_linkage()

        logger.info(f"Métodos ejecutados: {list(self.linkage_matrices.keys())}")

    def _generate_dendrograms(self) -> None:
        """Genera dendrogramas para todos los métodos."""
        logger.info("Generando dendrogramas...")

        # Dendrogramas individuales
        for method_name, linkage_matrix in self.linkage_matrices.items():
            viz = DendrogramVisualizer(linkage_matrix, self.titles, truncate_length=60)

            # Dendrograma estático
            static_path = self.output_dir / 'dendrograms' / f'{method_name}_linkage.png'
            viz.plot_static_dendrogram(str(static_path), f'{method_name.title()} Linkage')

            # Dendrograma interactivo
            interactive_path = self.output_dir / 'dendrograms' / f'{method_name}_interactive.html'
            viz.plot_interactive_dendrogram(str(interactive_path), f'{method_name.title()} Linkage')

        # Comparación lado a lado
        viz_compare = DendrogramVisualizer(
            self.linkage_matrices['average'],
            self.titles,
            truncate_length=40
        )

        comparison_path = self.output_dir / 'dendrograms' / 'comparison.png'
        viz_compare.plot_comparison_dendrograms(
            {k.title(): v for k, v in self.linkage_matrices.items()},
            str(comparison_path)
        )

        logger.info(f"Dendrogramas guardados en: {self.output_dir / 'dendrograms'}")

    def _evaluate_methods(self) -> None:
        """Evalúa todos los métodos de clustering."""
        logger.info("Evaluando métodos de clustering...")

        evaluator = ClusteringEvaluator(
            self.distance_matrix,
            self.linkage_matrices,
            self.abstracts
        )

        # Comparación completa
        self.evaluation_results = evaluator.compare_all_methods([3, 5, 7])

        # Guardar tabla de comparación
        csv_path = self.output_dir / 'evaluation' / 'metrics_comparison.csv'
        self.evaluation_results.to_csv(csv_path, index=False)
        logger.info(f"Tabla de comparación guardada: {csv_path}")

        # Gráfico de Silhouette
        sil_path = self.output_dir / 'evaluation' / 'silhouette_analysis.png'
        evaluator.plot_silhouette_comparison(range(2, 11), str(sil_path))

    def _recommend_best_method(self) -> Dict[str, Any]:
        """Recomienda el mejor método basado en evaluación."""
        logger.info("Determinando mejor método...")

        evaluator = ClusteringEvaluator(
            self.distance_matrix,
            self.linkage_matrices,
            self.abstracts
        )

        recommendation = evaluator.recommend_best_method([3, 5, 7])

        # Guardar recomendación
        json_path = self.output_dir / 'results' / 'best_method_recommendation.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convertir DataFrame a dict para JSON
            rec_copy = recommendation.copy()
            if 'comparison_table' in rec_copy:
                rec_copy['comparison_table'] = rec_copy['comparison_table'].to_dict()
            json.dump(rec_copy, f, indent=2)

        logger.info(f"Recomendación guardada: {json_path}")

        return recommendation

    def _export_summary_report(self, recommendation: Dict[str, Any]) -> None:
        """Genera reporte resumen en Markdown."""
        logger.info("Generando reporte resumen...")

        report_path = self.output_dir / 'reports' / 'clustering_summary.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Clustering Jerárquico\n\n")

            # Información general
            f.write("## Información General\n\n")
            f.write(f"- **Número de documentos**: {len(self.titles)}\n")
            f.write(f"- **Dimensión de vectores**: {self.vectors.shape[1]}\n")
            f.write(f"- **Métodos evaluados**: {', '.join(self.linkage_matrices.keys())}\n\n")

            # Método recomendado
            f.write("## Método Recomendado\n\n")
            f.write(f"**{recommendation['recommended_method'].upper()}**\n\n")
            f.write(f"**Número sugerido de clusters**: {recommendation['suggested_n_clusters']}\n\n")
            f.write("### Justificación\n\n")
            f.write(recommendation['justification'] + "\n\n")

            # Métricas
            f.write("## Métricas del Mejor Método\n\n")
            for metric, value in recommendation['metrics'].items():
                f.write(f"- **{metric}**: {value:.4f}\n")

            f.write("\n")

            # Tabla comparativa
            f.write("## Comparación de Métodos\n\n")
            if self.evaluation_results is not None:
                f.write(self.evaluation_results.to_markdown(index=False))
                f.write("\n\n")

            # Archivos generados
            f.write("## Archivos Generados\n\n")
            f.write("### Dendrogramas\n")
            for method in self.linkage_matrices.keys():
                f.write(f"- `dendrograms/{method}_linkage.png`\n")
                f.write(f"- `dendrograms/{method}_interactive.html`\n")
            f.write("- `dendrograms/comparison.png`\n\n")

            f.write("### Evaluación\n")
            f.write("- `evaluation/metrics_comparison.csv`\n")
            f.write("- `evaluation/silhouette_analysis.png`\n\n")

            f.write("### Resultados\n")
            f.write("- `results/best_method_recommendation.json`\n")

        logger.info(f"Reporte guardado: {report_path}")

    def _plot_optimal_k_analysis(self,
                                results: Dict[str, Any],
                                k_range: range,
                                method_name: str) -> None:
        """Genera gráfico de análisis de k óptimo."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # 1. Silhouette scores
        k_values = list(results['silhouette_scores'].keys())
        sil_values = list(results['silhouette_scores'].values())

        ax1.plot(k_values, sil_values, marker='o', linewidth=2, markersize=8)
        ax1.axvline(
            results['optimal_k_silhouette'],
            color='red',
            linestyle='--',
            label=f"Óptimo: k={results['optimal_k_silhouette']}"
        )
        ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax1.set_ylabel('Silhouette Score', fontsize=12)
        ax1.set_title(f'Análisis de Silhouette - {method_name.title()}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Merge distances (Elbow)
        merge_distances = results['merge_distances']
        n_merges = len(merge_distances)

        # Últimas fusiones (correspondientes a k_range)
        n_to_plot = k_range.stop - k_range.start + 1
        relevant_merges = merge_distances[-n_to_plot:]
        k_vals = list(range(k_range.stop, k_range.start - 1, -1))

        # Asegurar que tengan la misma longitud
        min_len = min(len(k_vals), len(relevant_merges))
        k_vals = k_vals[:min_len]
        relevant_merges = relevant_merges[:min_len]

        ax2.plot(k_vals, relevant_merges, marker='o', linewidth=2, markersize=8)
        ax2.axvline(
            results['optimal_k_elbow'],
            color='red',
            linestyle='--',
            label=f"Elbow: k={results['optimal_k_elbow']}"
        )
        ax2.set_xlabel('Número de Clusters (k)', fontsize=12)
        ax2.set_ylabel('Distancia de Fusión', fontsize=12)
        ax2.set_title(f'Elbow Method - {method_name.title()}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Guardar
        output_path = self.output_dir / 'evaluation' / f'optimal_k_analysis_{method_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráfico de análisis guardado: {output_path}")


def main():
    """Ejemplo de uso del pipeline completo."""

    print("\n" + "="*70)
    print(" EJEMPLO: Clustering Pipeline")
    print("="*70)

    # Crear datos de ejemplo (simular CSV unificado)
    print("\nCreando datos de ejemplo...")

    # Simular dataset
    example_data = {
        'Title': [
            'Deep Learning for Natural Language Processing',
            'Convolutional Neural Networks in Computer Vision',
            'Recurrent Networks for Sequence Modeling',
            'Attention Mechanisms and Transformers',
            'Reinforcement Learning: Theory and Applications',
            'Transfer Learning in Deep Neural Networks',
            'Graph Neural Networks for Relational Data',
            'Generative Adversarial Networks for Image Synthesis',
            'Meta-Learning and Few-Shot Learning',
            'Neural Architecture Search and AutoML',
            'Federated Learning for Privacy-Preserving ML',
            'Explainable AI and Model Interpretability',
            'Self-Supervised Learning Methods',
            'Multi-Task Learning with Shared Representations',
            'Knowledge Distillation and Model Compression'
        ],
        'Abstract': [
            'deep learning natural language processing nlp text classification sentiment analysis',
            'convolutional neural networks cnn computer vision image recognition object detection',
            'recurrent neural networks rnn sequence modeling time series lstm gru',
            'attention mechanisms transformer models bert gpt language understanding',
            'reinforcement learning policy gradient q learning reward optimization',
            'transfer learning domain adaptation pretrained models fine tuning',
            'graph neural networks gnn relational data knowledge graphs',
            'generative adversarial networks gan image synthesis deep generative models',
            'meta learning few shot learning learning to learn optimization',
            'neural architecture search nas automl automatic machine learning',
            'federated learning distributed learning privacy preserving machine learning',
            'explainable ai interpretability model understanding decision making',
            'self supervised learning representation learning unsupervised learning',
            'multi task learning shared representations joint training',
            'knowledge distillation model compression teacher student learning'
        ]
    }

    df = pd.DataFrame(example_data)

    # Guardar como CSV temporal
    temp_csv = 'output/temp_unified_data.csv'
    Path(temp_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temp_csv, index=False)

    print(f"Datos de ejemplo creados: {len(df)} documentos")

    # Crear y ejecutar pipeline
    print("\n" + "="*70)
    print("EJECUTANDO PIPELINE COMPLETO")
    print("="*70)

    pipeline = ClusteringPipeline(
        temp_csv,
        'output/clustering_pipeline'
    )

    # Ejecutar análisis completo
    results = pipeline.run_complete_analysis(
        vectorization_method='tfidf',
        distance_metric='cosine',
        max_features=100
    )

    # Análisis de k óptimo
    print("\n" + "="*70)
    print("ANÁLISIS DE NÚMERO ÓPTIMO DE CLUSTERS")
    print("="*70)

    best_method = results['recommendation']['recommended_method']
    best_linkage = results['linkage_matrices'][best_method]

    optimal_k_analysis = pipeline.analyze_optimal_clusters(
        best_linkage,
        range(2, 8),
        best_method
    )

    # Exportar clusters
    print("\n" + "="*70)
    print("EXPORTANDO CLUSTERS")
    print("="*70)

    clusters_df = pipeline.export_clusters(
        best_linkage,
        n_clusters=optimal_k_analysis['recommendation'],
        include_terms=True
    )

    print("\n" + "="*70)
    print(" PIPELINE COMPLETADO")
    print("="*70)
    print(f"\nMétodo recomendado: {best_method.upper()}")
    print(f"Número óptimo de clusters: {optimal_k_analysis['recommendation']}")
    print(f"Tiempo de ejecución: {results['execution_time']:.2f} segundos")
    print(f"\nResultados guardados en: output/clustering_pipeline/")


if __name__ == "__main__":
    main()
