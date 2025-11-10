"""
Clustering Evaluator Module
Evaluación y comparación de la calidad de diferentes métodos de clustering jerárquico.

Características:
- Cophenetic Correlation Coefficient (CPCC)
- Silhouette Score analysis
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Análisis de coherencia temática
- Comparación completa de métodos
- Recomendación automática del mejor método
- Generación de reportes detallados

Métricas implementadas:
=======================
1. CPCC: Mide qué tan bien el dendrograma preserva distancias originales
2. Silhouette: Evalúa separación y cohesión de clusters
3. Davies-Bouldin: Mide compacidad y separación (menor es mejor)
4. Calinski-Harabasz: Ratio de dispersión inter/intra cluster (mayor es mejor)
5. Coherencia temática: Análisis de contenido usando TF-IDF

Uso típico:
===========
1. Ejecutar múltiples métodos de clustering (single, complete, average)
2. Crear evaluador con matrices de linkage
3. Comparar métricas de calidad
4. Obtener recomendación del mejor método
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import time
from tqdm import tqdm

# Scipy para clustering y estadísticas
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import squareform

# Sklearn para métricas de clustering
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Evaluador de calidad de clustering jerárquico.

    Esta clase proporciona un conjunto completo de métricas para evaluar
    y comparar diferentes métodos de clustering jerárquico, facilitando
    la selección del método más apropiado para un dataset específico.

    Attributes:
        distance_matrix: Matriz de distancias original (n_samples, n_samples)
        linkage_matrices: Diccionario con matrices de linkage para cada método
        abstracts: Textos de los documentos para análisis de coherencia temática
        condensed_distances: Forma condensada de la matriz de distancia
        n_samples: Número de muestras
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 linkage_matrices: Dict[str, np.ndarray],
                 abstracts: Optional[List[str]] = None):
        """
        Inicializa el evaluador de clustering.

        Args:
            distance_matrix: Matriz cuadrada simétrica de distancias (n_samples, n_samples)
            linkage_matrices: Diccionario con matrices de linkage por método
                            {'single': Z1, 'complete': Z2, 'average': Z3, ...}
            abstracts: Lista opcional de abstracts/textos de documentos
                      para análisis de coherencia temática

        Example:
            >>> evaluator = ClusteringEvaluator(
            ...     distance_matrix,
            ...     {'single': Z_single, 'complete': Z_complete, 'average': Z_average},
            ...     abstracts=doc_abstracts
            ... )
        """
        self.distance_matrix = distance_matrix
        self.linkage_matrices = linkage_matrices
        self.abstracts = abstracts
        self.n_samples = distance_matrix.shape[0]

        # Convertir a forma condensada para scipy
        self.condensed_distances = squareform(distance_matrix, checks=False)

        # Cache de resultados
        self._cache = {}

        logger.info("="*70)
        logger.info("CLUSTERING EVALUATOR INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Número de muestras: {self.n_samples}")
        logger.info(f"Métodos a evaluar: {list(linkage_matrices.keys())}")
        logger.info(f"Abstracts disponibles: {'Sí' if abstracts else 'No'}")

    def cophenetic_correlation(self, linkage_matrix: np.ndarray) -> float:
        """
        Calcula el Cophenetic Correlation Coefficient (CPCC).

        Explicación matemática:
        ======================
        El CPCC mide qué tan bien el dendrograma preserva las distancias
        originales entre puntos.

        Para cada par de puntos (i, j):
        - d_original(i,j) = distancia original entre i y j
        - d_cophenetic(i,j) = altura en el dendrograma donde i y j se fusionan

        CPCC = correlación de Pearson entre d_original y d_cophenetic

        Interpretación:
        ==============
        - CPCC ≈ 1.0: El dendrograma preserva perfectamente las distancias
        - CPCC > 0.9: Excelente preservación
        - CPCC > 0.8: Buena preservación
        - CPCC > 0.7: Preservación moderada
        - CPCC < 0.7: Pobre preservación

        Ventajas:
        - Métrica intrínseca (no requiere clusters definidos)
        - Útil para comparar métodos de linkage
        - Independiente del número de clusters

        Args:
            linkage_matrix: Matriz de linkage de scipy

        Returns:
            CPCC score en rango [-1, 1]

        Example:
            >>> cpcc = evaluator.cophenetic_correlation(Z_average)
            >>> print(f"CPCC: {cpcc:.4f}")
        """
        logger.info("Calculando Cophenetic Correlation...")

        # Calcular correlación cofenética
        c, coph_dists = cophenet(linkage_matrix, self.condensed_distances)

        logger.info(f"CPCC calculado: {c:.4f}")

        # Interpretación
        if c > 0.9:
            interpretation = "Excelente"
        elif c > 0.8:
            interpretation = "Buena"
        elif c > 0.7:
            interpretation = "Moderada"
        else:
            interpretation = "Pobre"

        logger.info(f"Interpretación: {interpretation}")

        return float(c)

    def silhouette_analysis(self,
                           linkage_matrix: np.ndarray,
                           n_clusters_range: range = range(2, 11)) -> Dict[int, Dict[str, Any]]:
        """
        Análisis de Silhouette Score para diferentes números de clusters.

        Explicación matemática:
        ======================
        El Silhouette Score mide qué tan bien un punto está asignado a su cluster.

        Para cada punto i:
        - a(i) = distancia promedio de i a otros puntos en su mismo cluster
        - b(i) = distancia promedio de i a puntos del cluster más cercano

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

        Silhouette promedio = media de s(i) sobre todos los puntos

        Interpretación:
        ==============
        - s(i) ≈ 1: Punto bien clasificado, lejos de clusters vecinos
        - s(i) ≈ 0: Punto en la frontera entre dos clusters
        - s(i) < 0: Punto probablemente mal clasificado

        Rangos generales:
        - 0.71-1.0: Estructura fuerte
        - 0.51-0.70: Estructura razonable
        - 0.26-0.50: Estructura débil
        - < 0.25: Sin estructura sustancial

        Args:
            linkage_matrix: Matriz de linkage
            n_clusters_range: Rango de números de clusters a evaluar

        Returns:
            Diccionario con resultados por número de clusters:
            {
                n_clusters: {
                    'silhouette_avg': float,  # Promedio general
                    'silhouette_per_cluster': list,  # Por cluster
                    'silhouette_samples': array  # Por muestra
                }
            }

        Example:
            >>> results = evaluator.silhouette_analysis(Z_average, range(2, 8))
            >>> for n, metrics in results.items():
            ...     print(f"{n} clusters: {metrics['silhouette_avg']:.3f}")
        """
        logger.info("\n" + "="*70)
        logger.info("ANÁLISIS DE SILHOUETTE")
        logger.info("="*70)
        logger.info(f"Rango de clusters: {n_clusters_range.start} - {n_clusters_range.stop-1}")

        results = {}

        for n_clusters in tqdm(n_clusters_range, desc="Análisis Silhouette", unit="k"):
            # Obtener asignaciones de clusters
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

            # Calcular Silhouette score promedio
            silhouette_avg = silhouette_score(
                self.distance_matrix,
                cluster_labels,
                metric='precomputed'
            )

            # Calcular Silhouette por muestra
            silhouette_vals = silhouette_samples(
                self.distance_matrix,
                cluster_labels,
                metric='precomputed'
            )

            # Calcular Silhouette promedio por cluster
            silhouette_per_cluster = []
            for cluster_id in range(1, n_clusters + 1):
                cluster_silhouette = silhouette_vals[cluster_labels == cluster_id]
                silhouette_per_cluster.append(float(cluster_silhouette.mean()))

            results[n_clusters] = {
                'silhouette_avg': float(silhouette_avg),
                'silhouette_per_cluster': silhouette_per_cluster,
                'silhouette_samples': silhouette_vals
            }

            logger.info(f"n={n_clusters}: Silhouette avg = {silhouette_avg:.4f}")

        # Encontrar mejor k
        best_k = max(results.items(), key=lambda x: x[1]['silhouette_avg'])[0]
        logger.info(f"\nMejor número de clusters: {best_k} (Silhouette = {results[best_k]['silhouette_avg']:.4f})")

        logger.info("="*70)

        return results

    def davies_bouldin_index(self,
                            linkage_matrix: np.ndarray,
                            n_clusters: int) -> float:
        """
        Calcula el Davies-Bouldin Index (DBI).

        Explicación matemática:
        ======================
        El DBI mide la separación entre clusters. Valores más bajos indican
        mejor separación y compacidad de clusters.

        Para cada cluster i:
        - σ_i = dispersión promedio dentro del cluster (promedio de distancias al centroide)

        Para cada par de clusters (i, j):
        - d(c_i, c_j) = distancia entre centroides

        R_i,j = (σ_i + σ_j) / d(c_i, c_j)

        Para cada cluster i:
        - R_i = max(R_i,j) sobre todos j ≠ i

        DB = (1/k) × Σ R_i

        Interpretación:
        ==============
        - Valores cercanos a 0: Clusters bien separados y compactos
        - Valores altos: Clusters dispersos o mal separados
        - No tiene límite superior definido
        - Menor es mejor

        Ventajas:
        - Considera tanto compacidad como separación
        - No requiere conocimiento de la estructura real

        Desventajas:
        - Favorece clusters esféricos
        - Sensible a outliers

        Args:
            linkage_matrix: Matriz de linkage
            n_clusters: Número de clusters

        Returns:
            Davies-Bouldin Index (menor es mejor)

        Example:
            >>> db = evaluator.davies_bouldin_index(Z_average, n_clusters=5)
            >>> print(f"Davies-Bouldin Index: {db:.4f}")
        """
        logger.info(f"Calculando Davies-Bouldin Index para {n_clusters} clusters...")

        # Obtener asignaciones
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # NOTA: davies_bouldin_score de sklearn requiere las features originales,
        # no la matriz de distancia. Como trabajamos con matriz de distancia
        # precomputada, necesitamos usar una implementación alternativa o
        # usar MDS para obtener coordenadas.

        # Para simplificar, usamos MDS para obtener coordenadas en 2D
        from sklearn.manifold import MDS

        # MDS para obtener coordenadas que preserven distancias
        mds = MDS(n_components=min(2, self.n_samples - 1),
                  dissimilarity='precomputed',
                  random_state=42)
        coords = mds.fit_transform(self.distance_matrix)

        # Calcular Davies-Bouldin
        db_score = davies_bouldin_score(coords, cluster_labels)

        logger.info(f"Davies-Bouldin Index: {db_score:.4f}")

        return float(db_score)

    def calinski_harabasz_score_eval(self,
                                     linkage_matrix: np.ndarray,
                                     n_clusters: int) -> float:
        """
        Calcula el Calinski-Harabasz Score (Variance Ratio Criterion).

        Explicación matemática:
        ======================
        También conocido como Variance Ratio Criterion, mide el ratio entre
        la dispersión inter-cluster e intra-cluster.

        CH = (SS_B / (k-1)) / (SS_W / (n-k))

        donde:
        - SS_B = suma de cuadrados entre clusters (dispersión inter-cluster)
        - SS_W = suma de cuadrados dentro de clusters (dispersión intra-cluster)
        - k = número de clusters
        - n = número de muestras

        Interpretación:
        ==============
        - Valores altos: Clusters densos y bien separados
        - Mayor es mejor
        - No tiene límite superior definido

        Args:
            linkage_matrix: Matriz de linkage
            n_clusters: Número de clusters

        Returns:
            Calinski-Harabasz Score (mayor es mejor)

        Example:
            >>> ch = evaluator.calinski_harabasz_score_eval(Z_average, n_clusters=5)
            >>> print(f"Calinski-Harabasz Score: {ch:.4f}")
        """
        logger.info(f"Calculando Calinski-Harabasz Score para {n_clusters} clusters...")

        # Obtener asignaciones
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Usar MDS para obtener coordenadas
        from sklearn.manifold import MDS
        mds = MDS(n_components=min(2, self.n_samples - 1),
                  dissimilarity='precomputed',
                  random_state=42)
        coords = mds.fit_transform(self.distance_matrix)

        # Calcular Calinski-Harabasz
        ch_score = calinski_harabasz_score(coords, cluster_labels)

        logger.info(f"Calinski-Harabasz Score: {ch_score:.4f}")

        return float(ch_score)

    def calculate_cluster_coherence(self,
                                   linkage_matrix: np.ndarray,
                                   n_clusters: int,
                                   top_n_terms: int = 10) -> Dict[int, Dict[str, Any]]:
        """
        Evalúa coherencia temática de clusters usando TF-IDF.

        Este método analiza el contenido textual de los documentos en cada
        cluster para evaluar la coherencia temática.

        Métricas calculadas:
        ===================
        1. Top términos TF-IDF por cluster
        2. Similitud intra-cluster (cohesión)
        3. Similitud inter-cluster (separación)
        4. Coherence score = cohesión / separación

        Args:
            linkage_matrix: Matriz de linkage
            n_clusters: Número de clusters
            top_n_terms: Número de términos principales a extraer

        Returns:
            Diccionario con información por cluster:
            {
                cluster_id: {
                    'size': int,
                    'top_terms': list,
                    'intra_similarity': float,
                    'inter_similarity': float,
                    'coherence_score': float
                }
            }

        Example:
            >>> coherence = evaluator.calculate_cluster_coherence(Z_average, n_clusters=5)
            >>> for cid, info in coherence.items():
            ...     print(f"Cluster {cid}: {info['top_terms'][:3]}")
        """
        if self.abstracts is None:
            logger.warning("No hay abstracts disponibles para análisis de coherencia")
            return {}

        logger.info("\n" + "="*70)
        logger.info("ANÁLISIS DE COHERENCIA TEMÁTICA")
        logger.info("="*70)
        logger.info(f"Número de clusters: {n_clusters}")
        logger.info(f"Top términos: {top_n_terms}")

        # Obtener asignaciones
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Vectorizar abstracts con TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(self.abstracts)
        feature_names = vectorizer.get_feature_names_out()

        results = {}

        for cluster_id in range(1, n_clusters + 1):
            # Índices de documentos en este cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            # TF-IDF del cluster
            cluster_tfidf = tfidf_matrix[cluster_indices]

            # Calcular términos principales (promedio de TF-IDF en el cluster)
            tfidf_mean = cluster_tfidf.mean(axis=0).A1
            top_indices = tfidf_mean.argsort()[-top_n_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]

            # Similitud intra-cluster (cohesión)
            # Promedio de distancias entre documentos del mismo cluster
            if len(cluster_indices) > 1:
                intra_distances = []
                for i, idx_i in enumerate(cluster_indices):
                    for idx_j in cluster_indices[i+1:]:
                        intra_distances.append(self.distance_matrix[idx_i, idx_j])
                intra_similarity = 1 - np.mean(intra_distances) if intra_distances else 1.0
            else:
                intra_similarity = 1.0

            # Similitud inter-cluster (separación)
            # Promedio de distancias a documentos de otros clusters
            other_indices = np.where(cluster_labels != cluster_id)[0]
            if len(other_indices) > 0:
                inter_distances = []
                for idx_i in cluster_indices:
                    for idx_j in other_indices:
                        inter_distances.append(self.distance_matrix[idx_i, idx_j])
                inter_similarity = 1 - np.mean(inter_distances)
            else:
                inter_similarity = 0.0

            # Coherence score: queremos alta cohesión (intra alta) y baja separación (inter baja)
            # Score = intra / (inter + epsilon) o intra - inter
            if inter_similarity > 0:
                coherence_score = intra_similarity / (inter_similarity + 1e-6)
            else:
                coherence_score = intra_similarity

            results[cluster_id] = {
                'size': len(cluster_indices),
                'top_terms': top_terms,
                'intra_similarity': float(intra_similarity),
                'inter_similarity': float(inter_similarity),
                'coherence_score': float(coherence_score)
            }

            logger.info(f"\nCluster {cluster_id}:")
            logger.info(f"  Tamaño: {len(cluster_indices)}")
            logger.info(f"  Top términos: {top_terms[:5]}")
            logger.info(f"  Cohesión (intra): {intra_similarity:.4f}")
            logger.info(f"  Separación (inter): {inter_similarity:.4f}")
            logger.info(f"  Coherencia: {coherence_score:.4f}")

        logger.info("="*70)

        return results

    def compare_all_methods(self,
                           n_clusters_eval: List[int] = [3, 5, 7]) -> pd.DataFrame:
        """
        Genera tabla comparativa completa de todos los métodos.

        Métricas incluidas:
        ==================
        1. Cophenetic Correlation (CPCC)
        2. Silhouette Score promedio para diferentes k
        3. Davies-Bouldin Index para diferentes k
        4. Calinski-Harabasz Score para diferentes k
        5. Coherencia temática promedio (si hay abstracts)
        6. Tiempo de evaluación

        Args:
            n_clusters_eval: Lista de números de clusters para evaluar

        Returns:
            DataFrame con resultados comparativos

        Example:
            >>> df = evaluator.compare_all_methods([3, 5, 7])
            >>> print(df.to_string())
        """
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN COMPLETA DE MÉTODOS")
        logger.info("="*70)
        logger.info(f"Métodos: {list(self.linkage_matrices.keys())}")
        logger.info(f"Evaluando para k = {n_clusters_eval}")

        results = []

        for method_name, linkage_matrix in tqdm(self.linkage_matrices.items(),
                                                 desc="Comparando métodos",
                                                 unit="método"):
            logger.info(f"\n--- Evaluando {method_name.upper()} ---")

            start_time = time.time()

            row = {'Method': method_name}

            # 1. Cophenetic Correlation
            try:
                cpcc = self.cophenetic_correlation(linkage_matrix)
                row['CPCC'] = cpcc
            except Exception as e:
                logger.error(f"Error calculando CPCC: {e}")
                row['CPCC'] = np.nan

            # 2-4. Métricas para diferentes k
            for k in n_clusters_eval:
                try:
                    # Silhouette
                    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
                    sil = silhouette_score(self.distance_matrix, cluster_labels, metric='precomputed')
                    row[f'Silhouette_k{k}'] = sil

                    # Davies-Bouldin
                    db = self.davies_bouldin_index(linkage_matrix, k)
                    row[f'DB_k{k}'] = db

                    # Calinski-Harabasz
                    ch = self.calinski_harabasz_score_eval(linkage_matrix, k)
                    row[f'CH_k{k}'] = ch

                except Exception as e:
                    logger.error(f"Error evaluando k={k}: {e}")
                    row[f'Silhouette_k{k}'] = np.nan
                    row[f'DB_k{k}'] = np.nan
                    row[f'CH_k{k}'] = np.nan

            # 5. Coherencia temática (si disponible)
            if self.abstracts:
                try:
                    coherence_results = self.calculate_cluster_coherence(
                        linkage_matrix,
                        n_clusters=n_clusters_eval[0]  # Usar primer k
                    )
                    if coherence_results:
                        avg_coherence = np.mean([
                            info['coherence_score']
                            for info in coherence_results.values()
                        ])
                        row['Avg_Coherence'] = avg_coherence
                except Exception as e:
                    logger.error(f"Error calculando coherencia: {e}")
                    row['Avg_Coherence'] = np.nan

            # 6. Tiempo de evaluación
            elapsed_time = time.time() - start_time
            row['Time_sec'] = elapsed_time

            results.append(row)

        # Crear DataFrame
        df = pd.DataFrame(results)

        # Ordenar columnas
        cols = ['Method', 'CPCC']
        for k in n_clusters_eval:
            cols.extend([f'Silhouette_k{k}', f'DB_k{k}', f'CH_k{k}'])
        if 'Avg_Coherence' in df.columns:
            cols.append('Avg_Coherence')
        cols.append('Time_sec')

        df = df[cols]

        logger.info("\n" + "="*70)
        logger.info("TABLA COMPARATIVA")
        logger.info("="*70)
        logger.info(f"\n{df.to_string(index=False)}")
        logger.info("="*70)

        return df

    def recommend_best_method(self,
                             n_clusters_eval: List[int] = [3, 5, 7]) -> Dict[str, Any]:
        """
        Determina cuál método produce mejores agrupamientos.

        Criterios de evaluación (por prioridad):
        ========================================
        1. CPCC > 0.80 (preservación de distancias) - ALTA PRIORIDAD
        2. Silhouette score alto (separación clara) - ALTA PRIORIDAD
        3. Davies-Bouldin bajo (compacidad y separación) - MEDIA PRIORIDAD
        4. Calinski-Harabasz alto (ratio de varianza) - MEDIA PRIORIDAD
        5. Coherencia temática alta (si disponible) - BAJA PRIORIDAD

        Sistema de puntuación:
        =====================
        - Cada métrica se normaliza a [0, 1]
        - Se aplican pesos según prioridad
        - Se calcula score total

        Args:
            n_clusters_eval: Lista de k para evaluar

        Returns:
            Diccionario con recomendación:
            {
                'recommended_method': str,
                'justification': str,
                'metrics': dict,
                'suggested_n_clusters': int,
                'scores': dict  # Scores normalizados por método
            }

        Example:
            >>> recommendation = evaluator.recommend_best_method([3, 5, 7])
            >>> print(f"Método recomendado: {recommendation['recommended_method']}")
            >>> print(f"Razón: {recommendation['justification']}")
        """
        logger.info("\n" + "="*70)
        logger.info("RECOMENDACIÓN DEL MEJOR MÉTODO")
        logger.info("="*70)

        # Obtener comparación completa
        df = self.compare_all_methods(n_clusters_eval)

        # Sistema de puntuación
        scores = {}
        metrics_by_method = {}

        for idx, row in df.iterrows():
            method = row['Method']
            score = 0.0
            metrics = {}

            # 1. CPCC (peso: 3.0)
            if not np.isnan(row['CPCC']):
                cpcc_score = row['CPCC']  # Ya está en [0, 1]
                score += cpcc_score * 3.0
                metrics['CPCC'] = cpcc_score

            # 2. Silhouette promedio (peso: 2.5)
            sil_scores = [row[f'Silhouette_k{k}'] for k in n_clusters_eval if f'Silhouette_k{k}' in row]
            sil_scores = [s for s in sil_scores if not np.isnan(s)]
            if sil_scores:
                avg_sil = np.mean(sil_scores)
                # Normalizar de [-1, 1] a [0, 1]
                sil_normalized = (avg_sil + 1) / 2
                score += sil_normalized * 2.5
                metrics['Avg_Silhouette'] = avg_sil

            # 3. Davies-Bouldin promedio (peso: 1.5, invertido porque menor es mejor)
            db_scores = [row[f'DB_k{k}'] for k in n_clusters_eval if f'DB_k{k}' in row]
            db_scores = [s for s in db_scores if not np.isnan(s)]
            if db_scores:
                avg_db = np.mean(db_scores)
                # Normalizar e invertir (asumiendo DB típicamente < 5)
                db_normalized = max(0, 1 - avg_db / 5.0)
                score += db_normalized * 1.5
                metrics['Avg_DB'] = avg_db

            # 4. Calinski-Harabasz promedio (peso: 1.0)
            ch_scores = [row[f'CH_k{k}'] for k in n_clusters_eval if f'CH_k{k}' in row]
            ch_scores = [s for s in ch_scores if not np.isnan(s)]
            if ch_scores:
                avg_ch = np.mean(ch_scores)
                # Normalizar (asumiendo CH típicamente < 1000)
                ch_normalized = min(1.0, avg_ch / 1000.0)
                score += ch_normalized * 1.0
                metrics['Avg_CH'] = avg_ch

            # 5. Coherencia temática (peso: 1.0)
            if 'Avg_Coherence' in row and not np.isnan(row['Avg_Coherence']):
                coherence = row['Avg_Coherence']
                # Normalizar (asumiendo coherence típicamente < 10)
                coh_normalized = min(1.0, coherence / 10.0)
                score += coh_normalized * 1.0
                metrics['Coherence'] = coherence

            scores[method] = score
            metrics_by_method[method] = metrics

        # Encontrar mejor método
        best_method = max(scores.items(), key=lambda x: x[1])[0]
        best_score = scores[best_method]
        best_metrics = metrics_by_method[best_method]

        logger.info("\nPuntuaciones por método:")
        for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {method}: {score:.4f}")

        logger.info(f"\nMejor método: {best_method.upper()} (score: {best_score:.4f})")

        # Sugerir número óptimo de clusters basado en Silhouette
        best_k = None
        best_k_sil = -1
        linkage_matrix = self.linkage_matrices[best_method]

        for k in n_clusters_eval:
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
            try:
                sil = silhouette_score(self.distance_matrix, cluster_labels, metric='precomputed')
                if sil > best_k_sil:
                    best_k_sil = sil
                    best_k = k
            except:
                pass

        # Generar justificación
        justification = f"El método {best_method} fue seleccionado basado en:\n"

        if 'CPCC' in best_metrics:
            cpcc = best_metrics['CPCC']
            justification += f"- CPCC = {cpcc:.4f} "
            justification += ("(excelente)" if cpcc > 0.9 else "(bueno)" if cpcc > 0.8 else "(moderado)") + "\n"

        if 'Avg_Silhouette' in best_metrics:
            sil = best_metrics['Avg_Silhouette']
            justification += f"- Silhouette promedio = {sil:.4f} "
            justification += ("(estructura fuerte)" if sil > 0.7 else "(estructura razonable)" if sil > 0.5 else "(estructura débil)") + "\n"

        if 'Avg_DB' in best_metrics:
            db = best_metrics['Avg_DB']
            justification += f"- Davies-Bouldin promedio = {db:.4f} "
            justification += ("(excelente separación)" if db < 1.0 else "(buena separación)" if db < 1.5 else "(separación moderada)") + "\n"

        logger.info(f"\nJustificación:\n{justification}")

        if best_k:
            logger.info(f"Número sugerido de clusters: {best_k}")

        logger.info("="*70)

        return {
            'recommended_method': best_method,
            'justification': justification,
            'metrics': best_metrics,
            'suggested_n_clusters': best_k,
            'scores': scores,
            'comparison_table': df
        }

    def plot_silhouette_comparison(self,
                                  n_clusters_range: range = range(2, 11),
                                  output_path: str = 'output/silhouette_comparison.png',
                                  figsize: Tuple[int, int] = (12, 6),
                                  dpi: int = 300) -> None:
        """
        Genera gráfico comparativo de Silhouette Score vs número de clusters.

        Args:
            n_clusters_range: Rango de k a evaluar
            output_path: Ruta para guardar la imagen
            figsize: Tamaño de la figura
            dpi: Resolución

        Example:
            >>> evaluator.plot_silhouette_comparison(range(2, 11))
        """
        logger.info("Generando gráfico de Silhouette Score...")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for method_name, linkage_matrix in self.linkage_matrices.items():
            silhouette_scores = []
            valid_k = []

            for k in n_clusters_range:
                try:
                    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
                    # Verificar que hay al menos 2 clusters
                    n_unique_labels = len(np.unique(cluster_labels))
                    if n_unique_labels < 2:
                        logger.warning(f"Solo {n_unique_labels} cluster(s) para k={k}, omitiendo...")
                        continue

                    sil = silhouette_score(self.distance_matrix, cluster_labels, metric='precomputed')
                    silhouette_scores.append(sil)
                    valid_k.append(k)
                except Exception as e:
                    logger.warning(f"Error calculando Silhouette para k={k}: {e}")
                    continue

            if len(silhouette_scores) > 0:
                ax.plot(valid_k, silhouette_scores,
                       marker='o', label=method_name, linewidth=2)

        ax.set_xlabel('Número de Clusters', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Comparación de Silhouette Score\npor Método de Linkage',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráfico guardado en: {output_path}")


def main():
    """Ejemplo de uso del evaluador de clustering."""

    print("\n" + "="*70)
    print(" EJEMPLO: Clustering Evaluator")
    print("="*70)

    # Crear datos de ejemplo
    from distance_calculator import DistanceCalculator
    from hierarchical_clustering import HierarchicalClustering

    np.random.seed(42)

    # Simular 20 documentos
    n_docs = 20
    n_features = 30
    vectors = np.random.rand(n_docs, n_features)

    # Normalizar
    from sklearn.preprocessing import normalize
    vectors = normalize(vectors, norm='l2')

    # Abstracts de ejemplo
    abstracts = [
        "neural networks deep learning classification",
        "machine learning algorithms supervised learning",
        "natural language processing text mining",
        "computer vision image recognition",
        "reinforcement learning policy optimization",
        "transfer learning domain adaptation",
        "generative adversarial networks synthesis",
        "recurrent neural networks sequence modeling",
        "convolutional neural networks feature extraction",
        "attention mechanisms transformer models",
        "graph neural networks relational data",
        "meta learning few shot learning",
        "federated learning privacy preserving",
        "explainable ai interpretability",
        "adversarial robustness defense mechanisms",
        "neural architecture search optimization",
        "self supervised learning representation",
        "knowledge distillation model compression",
        "multi task learning shared representations",
        "online learning streaming data"
    ]

    print(f"\nDocumentos: {n_docs}")
    print(f"Features: {n_features}")

    # Calcular distancias
    dist_calc = DistanceCalculator()
    distance_matrix = dist_calc.cosine_distance(vectors)

    # Clustering con diferentes métodos
    hc = HierarchicalClustering(distance_matrix)

    Z_single = hc.single_linkage()
    Z_complete = hc.complete_linkage()
    Z_average = hc.average_linkage()

    linkage_matrices = {
        'single': Z_single,
        'complete': Z_complete,
        'average': Z_average
    }

    # Crear evaluador
    print("\n" + "="*70)
    print("EVALUACIÓN DE CLUSTERING")
    print("="*70)

    evaluator = ClusteringEvaluator(
        distance_matrix,
        linkage_matrices,
        abstracts=abstracts
    )

    # 1. Comparar todos los métodos
    print("\n1. Comparación completa...")
    df = evaluator.compare_all_methods(n_clusters_eval=[3, 5, 7])

    # 2. Recomendación
    print("\n2. Recomendación del mejor método...")
    recommendation = evaluator.recommend_best_method([3, 5, 7])

    print(f"\nMétodo recomendado: {recommendation['recommended_method'].upper()}")
    print(f"Número de clusters sugerido: {recommendation['suggested_n_clusters']}")
    print(f"\nJustificación:\n{recommendation['justification']}")

    # 3. Gráfico de Silhouette
    print("\n3. Generando gráfico de Silhouette...")
    evaluator.plot_silhouette_comparison(range(2, 11))

    print("\n" + "="*70)
    print(" EJEMPLO COMPLETADO")
    print("="*70)
    print("\nArchivos generados:")
    print("  - output/silhouette_comparison.png")


if __name__ == "__main__":
    main()
