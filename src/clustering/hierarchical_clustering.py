"""
Hierarchical Clustering Module
Implementación de clustering jerárquico aglomerativo con múltiples métodos de linkage.

Características:
- Single Linkage (Nearest Neighbor)
- Complete Linkage (Farthest Neighbor)
- Average Linkage (UPGMA)
- Ward's Method (minimiza varianza)
- Logging detallado de pasos del algoritmo
- Validación robusta de inputs
- Soporte para labels personalizados
- Análisis de dendrogramas

Métodos de Linkage:
==================
1. Single: d(C1, C2) = min{d(x,y) : x∈C1, y∈C2}
   - Distancia entre clusters = puntos más cercanos
   - Tiende a crear cadenas largas
   - Sensible a outliers y ruido

2. Complete: d(C1, C2) = max{d(x,y) : x∈C1, y∈C2}
   - Distancia entre clusters = puntos más lejanos
   - Produce clusters compactos
   - Robusto a outliers

3. Average: d(C1, C2) = promedio de todas las distancias entre pares
   - Balance entre single y complete
   - Buen rendimiento general
   - Menos sensible a outliers

4. Ward: minimiza incremento de varianza intra-cluster
   - Uno de los mejores métodos
   - Requiere distancia euclidiana
   - Produce clusters balanceados
"""

import numpy as np
import logging
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Scipy para clustering jerárquico
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster,
    inconsistent,
    cophenet
)
from scipy.spatial.distance import squareform

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchicalClustering:
    """
    Clustering jerárquico aglomerativo con múltiples métodos de linkage.

    Esta clase implementa algoritmos de clustering jerárquico que construyen
    una jerarquía de clusters (dendrograma) fusionando iterativamente los
    clusters más cercanos según diferentes criterios de distancia.

    Flujo del algoritmo:
    ===================
    1. Iniciar con N clusters (cada punto es un cluster)
    2. Calcular distancias entre todos los pares de clusters
    3. Fusionar los dos clusters más cercanos
    4. Actualizar matriz de distancias
    5. Repetir 2-4 hasta tener un solo cluster

    Attributes:
        distance_matrix: Matriz de distancias (n_samples, n_samples)
        labels: Etiquetas opcionales para los documentos/puntos
        n_samples: Número de muestras
        linkage_matrix: Matriz de linkage resultante (creada al ejecutar un método)
    """

    SUPPORTED_METHODS = ['single', 'complete', 'average', 'ward']

    def __init__(self,
                 distance_matrix: np.ndarray,
                 labels: Optional[List[str]] = None):
        """
        Inicializa el clustering jerárquico con una matriz de distancia.

        Args:
            distance_matrix: Matriz cuadrada simétrica de distancias (n_samples, n_samples)
            labels: Lista opcional de etiquetas para los puntos (ej: títulos de artículos)

        Raises:
            ValueError: Si la matriz de distancia es inválida

        Example:
            >>> dist_matrix = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
            >>> labels = ['Doc1', 'Doc2', 'Doc3']
            >>> hc = HierarchicalClustering(dist_matrix, labels)
        """
        # Validar matriz de distancia
        self._validate_distance_matrix(distance_matrix)

        self.distance_matrix = distance_matrix
        self.n_samples = distance_matrix.shape[0]

        # Validar y asignar labels
        if labels is None:
            self.labels = [f"Sample_{i}" for i in range(self.n_samples)]
            logger.info(f"No se proporcionaron labels, usando nombres por defecto")
        else:
            if len(labels) != self.n_samples:
                raise ValueError(
                    f"Número de labels ({len(labels)}) no coincide con "
                    f"número de muestras ({self.n_samples})"
                )
            self.labels = labels

        # Convertir matriz a forma condensada para scipy
        # scipy.cluster.hierarchy.linkage espera forma condensada (1D array)
        # que contiene solo el triángulo superior de la matriz de distancias
        self.condensed_distances = squareform(distance_matrix, checks=False)

        # Matriz de linkage (se creará al ejecutar un método)
        self.linkage_matrix = None
        self.method_used = None

        logger.info("="*70)
        logger.info("HIERARCHICAL CLUSTERING INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Número de muestras: {self.n_samples}")
        logger.info(f"Labels: {self.labels[:5]}{'...' if len(self.labels) > 5 else ''}")
        logger.info(f"Distancia mínima: {self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)].min():.4f}")
        logger.info(f"Distancia máxima: {self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)].max():.4f}")
        logger.info(f"Distancia promedio: {self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)].mean():.4f}")

    def single_linkage(self) -> np.ndarray:
        """
        Single Linkage (Nearest Neighbor / Minimum Method).

        Explicación matemática:
        =====================
        d(C1, C2) = min{d(x, y) : x ∈ C1, y ∈ C2}

        La distancia entre dos clusters C1 y C2 se define como la distancia
        MÍNIMA entre cualquier par de puntos donde un punto está en C1 y
        el otro en C2.

        Algoritmo:
        =========
        1. Iniciar con N clusters singleton (cada punto es un cluster)
        2. Encontrar el par de clusters (Ci, Cj) con distancia mínima
        3. Fusionar Ci y Cj en un nuevo cluster Ck
        4. Actualizar distancias:
           Para todo cluster Cl:
           d(Ck, Cl) = min(d(Ci, Cl), d(Cj, Cl))
        5. Repetir pasos 2-4 hasta tener un solo cluster

        Características:
        ==============
        Ventajas:
        - Puede detectar clusters de formas no convexas (alargadas, irregulares)
        - Bueno para datos con clusters naturalmente "encadenados"
        - Computacionalmente eficiente

        Desventajas:
        - Efecto "chaining": tiende a crear clusters largos y delgados
        - MUY sensible a outliers y ruido
        - Puede conectar clusters que deberían estar separados

        Cuándo usar:
        - Cuando esperas clusters de formas irregulares
        - Cuando los datos tienen baja dimensionalidad
        - NO usar si hay ruido o outliers

        Returns:
            Linkage matrix (n_samples-1, 4) donde cada fila [i, j, dist, n_items] representa:
            - i, j: índices de clusters fusionados
            - dist: distancia a la que se fusionaron
            - n_items: número de elementos en el nuevo cluster

        Example:
            >>> hc = HierarchicalClustering(distance_matrix)
            >>> Z = hc.single_linkage()
            >>> # Z contiene la jerarquía completa de fusiones
        """
        logger.info("\n" + "="*70)
        logger.info("EJECUTANDO SINGLE LINKAGE")
        logger.info("="*70)
        logger.info("Método: Nearest Neighbor")
        logger.info("Criterio: d(C1,C2) = min{d(x,y) : x∈C1, y∈C2}")

        # Ejecutar linkage con scipy
        # method='single' implementa el algoritmo de nearest neighbor
        self.linkage_matrix = linkage(
            self.condensed_distances,
            method='single'
        )
        self.method_used = 'single'

        # Logging de resultados
        self._log_linkage_results()

        logger.info("="*70)
        logger.info("SINGLE LINKAGE COMPLETADO")
        logger.info("="*70)

        return self.linkage_matrix

    def complete_linkage(self) -> np.ndarray:
        """
        Complete Linkage (Farthest Neighbor / Maximum Method).

        Explicación matemática:
        =====================
        d(C1, C2) = max{d(x, y) : x ∈ C1, y ∈ C2}

        La distancia entre dos clusters C1 y C2 se define como la distancia
        MÁXIMA entre cualquier par de puntos donde un punto está en C1 y
        el otro en C2.

        Algoritmo:
        =========
        1. Iniciar con N clusters singleton
        2. Encontrar el par de clusters (Ci, Cj) con distancia mínima
        3. Fusionar Ci y Cj en un nuevo cluster Ck
        4. Actualizar distancias:
           Para todo cluster Cl:
           d(Ck, Cl) = max(d(Ci, Cl), d(Cj, Cl))
        5. Repetir hasta tener un solo cluster

        Características:
        ==============
        Ventajas:
        - Produce clusters COMPACTOS y bien separados
        - Robusto a outliers (no los encadena fácilmente)
        - Tiende a crear clusters de tamaños similares
        - Buena separación entre clusters

        Desventajas:
        - Puede romper clusters naturales alargados
        - Tiende a clusters esféricos
        - Sensible a la presencia de puntos puente entre clusters

        Cuándo usar:
        - Cuando quieres clusters compactos y bien definidos
        - Cuando hay presencia de outliers
        - Datos con clusters esféricos o convexos

        Comparación con Single:
        =====================
        Single Linkage → Clusters alargados, sensible a ruido
        Complete Linkage → Clusters compactos, robusto a ruido

        Returns:
            Linkage matrix (n_samples-1, 4)

        Example:
            >>> hc = HierarchicalClustering(distance_matrix)
            >>> Z = hc.complete_linkage()
        """
        logger.info("\n" + "="*70)
        logger.info("EJECUTANDO COMPLETE LINKAGE")
        logger.info("="*70)
        logger.info("Método: Farthest Neighbor")
        logger.info("Criterio: d(C1,C2) = max{d(x,y) : x∈C1, y∈C2}")

        # Ejecutar linkage con scipy
        self.linkage_matrix = linkage(
            self.condensed_distances,
            method='complete'
        )
        self.method_used = 'complete'

        # Logging de resultados
        self._log_linkage_results()

        logger.info("="*70)
        logger.info("COMPLETE LINKAGE COMPLETADO")
        logger.info("="*70)

        return self.linkage_matrix

    def average_linkage(self) -> np.ndarray:
        """
        Average Linkage (UPGMA - Unweighted Pair Group Method with Arithmetic Mean).

        Explicación matemática:
        =====================
        d(C1, C2) = (1/(|C1| × |C2|)) × Σ Σ d(x, y)
                                         x∈C1 y∈C2

        La distancia entre dos clusters es el PROMEDIO de todas las distancias
        entre pares de puntos, uno de cada cluster.

        Fórmula recursiva:
        Para fusionar Ci y Cj en Ck, la distancia a otro cluster Cl es:

        d(Ck, Cl) = (|Ci| × d(Ci, Cl) + |Cj| × d(Cj, Cl)) / (|Ci| + |Cj|)

        donde |C| denota el número de elementos en el cluster C.

        Algoritmo:
        =========
        1. Iniciar con N clusters singleton
        2. Encontrar par de clusters con distancia promedio mínima
        3. Fusionar ese par
        4. Actualizar distancias usando promedio ponderado
        5. Repetir hasta tener un solo cluster

        Características:
        ==============
        Ventajas:
        - Balance óptimo entre single y complete linkage
        - Menos sensible a outliers que single
        - Más flexible que complete (permite formas no esféricas)
        - Buen rendimiento general en muchos datasets
        - Interpretación intuitiva (distancia promedio)

        Desventajas:
        - Computacionalmente más costoso que single/complete
        - Puede ser influenciado por clusters de tamaños muy diferentes

        Cuándo usar:
        - Como método por defecto cuando no conoces la estructura de datos
        - Cuando quieres balance entre robustez y flexibilidad
        - Datasets con clusters de diferentes formas y tamaños

        Comparación:
        ===========
        Single    → d(C1,C2) = MIN(todas las distancias)
        Complete  → d(C1,C2) = MAX(todas las distancias)
        Average   → d(C1,C2) = PROMEDIO(todas las distancias)

        Returns:
            Linkage matrix (n_samples-1, 4)

        Example:
            >>> hc = HierarchicalClustering(distance_matrix)
            >>> Z = hc.average_linkage()
        """
        logger.info("\n" + "="*70)
        logger.info("EJECUTANDO AVERAGE LINKAGE (UPGMA)")
        logger.info("="*70)
        logger.info("Método: Average Linkage")
        logger.info("Criterio: d(C1,C2) = promedio de todas las distancias entre pares")

        # Ejecutar linkage con scipy
        # method='average' implementa UPGMA
        self.linkage_matrix = linkage(
            self.condensed_distances,
            method='average'
        )
        self.method_used = 'average'

        # Logging de resultados
        self._log_linkage_results()

        logger.info("="*70)
        logger.info("AVERAGE LINKAGE (UPGMA) COMPLETADO")
        logger.info("="*70)

        return self.linkage_matrix

    def ward_linkage(self) -> np.ndarray:
        """
        Ward's Method (Minimum Variance Method) - BONUS.

        Explicación matemática:
        =====================
        Ward's method minimiza el incremento en la suma de cuadrados
        dentro de clusters (varianza intra-cluster) al fusionar.

        Para cada fusión posible de clusters Ci y Cj, calcula:

        ΔSS(Ci, Cj) = SS(Ci ∪ Cj) - SS(Ci) - SS(Cj)

        donde SS(C) = Σ ||x - μ_C||² es la suma de cuadrados dentro del cluster
        y μ_C es el centroide del cluster C.

        Fórmula de Lance-Williams:
        =========================
        d_ward(Ck, Cl) = √(((|Ci|+|Cl|)×d²(Ci,Cl) + (|Cj|+|Cl|)×d²(Cj,Cl) - |Cl|×d²(Ci,Cj)) / (|Ci|+|Cj|+|Cl|))

        Criterio de fusión:
        ==================
        En cada paso, fusiona el par de clusters que produce el MÍNIMO
        incremento en la varianza total intra-cluster.

        Características:
        ==============
        Ventajas:
        - Uno de los MEJORES métodos de clustering jerárquico
        - Produce clusters muy balanceados en tamaño
        - Minimiza la varianza dentro de clusters
        - Maximiza la separación entre clusters
        - Excelente para datos con clusters esféricos

        Desventajas:
        - REQUIERE distancia euclidiana (no funciona con otras métricas)
        - Sesgo hacia clusters de tamaño similar
        - Sensible a outliers

        Cuándo usar:
        - Cuando usas distancia euclidiana
        - Cuando quieres clusters balanceados y bien separados
        - Como primera opción para la mayoría de aplicaciones
        - NO usar con distancia coseno (usar average en su lugar)

        Relación con K-means:
        ===================
        Ward's method optimiza el mismo criterio que K-means (minimizar
        varianza intra-cluster), pero de forma jerárquica. De hecho,
        cortar un dendrograma de Ward en k clusters tiende a dar resultados
        similares a K-means con k clusters.

        IMPORTANTE:
        ==========
        Para usar Ward con matriz de distancias precomputadas, scipy requiere
        que las distancias sean euclidianas. Si usaste otra métrica (ej: coseno),
        los resultados pueden ser incorrectos o dar error.

        Returns:
            Linkage matrix (n_samples-1, 4)

        Raises:
            Warning: Si la matriz de distancia no es euclidiana

        Example:
            >>> # Con distancia euclidiana
            >>> dist_calc = DistanceCalculator()
            >>> dist_matrix = dist_calc.euclidean_distance(vectors)
            >>> hc = HierarchicalClustering(dist_matrix)
            >>> Z = hc.ward_linkage()
        """
        logger.info("\n" + "="*70)
        logger.info("EJECUTANDO WARD'S METHOD (BONUS)")
        logger.info("="*70)
        logger.info("Método: Minimum Variance")
        logger.info("Criterio: Minimizar incremento de varianza intra-cluster")

        logger.warning(
            "⚠️  IMPORTANTE: Ward's method requiere distancia EUCLIDIANA. "
            "Si usaste otra métrica (ej: coseno), los resultados pueden ser incorrectos."
        )

        # Ejecutar linkage con scipy
        # method='ward' implementa minimum variance
        # NOTA: scipy implementa Ward de forma especial para matrices de distancia
        try:
            self.linkage_matrix = linkage(
                self.condensed_distances,
                method='ward'
            )
            self.method_used = 'ward'

            # Logging de resultados
            self._log_linkage_results()

            logger.info("="*70)
            logger.info("WARD'S METHOD COMPLETADO")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"Error en Ward's method: {e}")
            logger.error(
                "Esto puede ocurrir si la matriz de distancia no es euclidiana. "
                "Intenta usar average_linkage() en su lugar."
            )
            raise

        return self.linkage_matrix

    def get_clusters(self, n_clusters: int, method: str = 'average') -> np.ndarray:
        """
        Obtiene asignación de clusters cortando el dendrograma.

        Args:
            n_clusters: Número de clusters deseado
            method: Método de linkage a usar si no se ha ejecutado ninguno

        Returns:
            Array de asignaciones de cluster para cada muestra

        Example:
            >>> hc = HierarchicalClustering(dist_matrix)
            >>> hc.average_linkage()
            >>> clusters = hc.get_clusters(n_clusters=3)
            >>> print(clusters)  # [1, 1, 2, 3, 2, ...]
        """
        # Si no se ha ejecutado ningún método, ejecutar el especificado
        if self.linkage_matrix is None:
            logger.info(f"No se ha ejecutado clustering, usando {method} linkage")

            if method == 'single':
                self.single_linkage()
            elif method == 'complete':
                self.complete_linkage()
            elif method == 'average':
                self.average_linkage()
            elif method == 'ward':
                self.ward_linkage()
            else:
                raise ValueError(f"Método desconocido: {method}")

        # Cortar dendrograma para obtener clusters
        cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')

        logger.info(f"Dendrograma cortado en {n_clusters} clusters")
        logger.info(f"Distribución de clusters: {np.bincount(cluster_labels)}")

        return cluster_labels

    def compare_methods(self) -> Dict[str, np.ndarray]:
        """
        Ejecuta y compara todos los métodos de linkage disponibles.

        Calcula el coeficiente de correlación cofenética para cada método,
        que mide qué tan bien el dendrograma preserva las distancias originales.

        Coeficiente cofenético:
        ======================
        Correlación entre:
        - Distancias originales entre puntos
        - Distancias cofenéticas (altura en dendrograma donde se fusionan)

        Rango: [-1, 1]
        - Valores cercanos a 1 → Dendrograma preserva bien las distancias
        - Valores bajos → Dendrograma distorsiona las distancias

        Returns:
            Diccionario con linkage matrices y métricas de calidad

        Example:
            >>> hc = HierarchicalClustering(dist_matrix)
            >>> comparison = hc.compare_methods()
            >>> for method, metrics in comparison['metrics'].items():
            ...     print(f"{method}: cophenetic={metrics['cophenetic']:.3f}")
        """
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN DE MÉTODOS DE LINKAGE")
        logger.info("="*70)

        results = {}
        metrics = {}

        # Ejecutar cada método
        for method in ['single', 'complete', 'average']:
            logger.info(f"\n--- Evaluando {method.upper()} ---")

            try:
                # Ejecutar método
                if method == 'single':
                    Z = self.single_linkage()
                elif method == 'complete':
                    Z = self.complete_linkage()
                elif method == 'average':
                    Z = self.average_linkage()

                results[method] = Z

                # Calcular coeficiente cofenético
                c, coph_dists = cophenet(Z, self.condensed_distances)

                metrics[method] = {
                    'cophenetic_correlation': float(c),
                    'min_distance': float(Z[:, 2].min()),
                    'max_distance': float(Z[:, 2].max()),
                    'mean_distance': float(Z[:, 2].mean())
                }

                logger.info(f"Correlación cofenética: {c:.4f}")

            except Exception as e:
                logger.error(f"Error en {method}: {e}")
                results[method] = None
                metrics[method] = None

        # Resumen
        logger.info("\n" + "="*70)
        logger.info("RESUMEN DE COMPARACIÓN")
        logger.info("="*70)

        for method, metric in metrics.items():
            if metric:
                logger.info(f"\n{method.upper()}:")
                logger.info(f"  Correlación cofenética: {metric['cophenetic_correlation']:.4f}")
                logger.info(f"  Distancia de fusión mín: {metric['min_distance']:.4f}")
                logger.info(f"  Distancia de fusión máx: {metric['max_distance']:.4f}")
                logger.info(f"  Distancia de fusión media: {metric['mean_distance']:.4f}")

        # Mejor método según correlación cofenética
        best_method = max(metrics.items(), key=lambda x: x[1]['cophenetic_correlation'] if x[1] else -1)
        logger.info(f"\nMejor método (mayor correlación cofenética): {best_method[0].upper()}")

        return {
            'linkage_matrices': results,
            'metrics': metrics,
            'best_method': best_method[0]
        }

    def _log_linkage_results(self) -> None:
        """
        Registra información detallada sobre los resultados del linkage.

        Internal method.
        """
        if self.linkage_matrix is None:
            return

        # Información sobre fusiones
        logger.info(f"\nNúmero de fusiones: {len(self.linkage_matrix)}")
        logger.info(f"Forma de linkage matrix: {self.linkage_matrix.shape}")

        # Distancias de fusión
        merge_distances = self.linkage_matrix[:, 2]
        logger.info(f"\nDistancias de fusión:")
        logger.info(f"  Primera fusión: {merge_distances[0]:.4f}")
        logger.info(f"  Última fusión: {merge_distances[-1]:.4f}")
        logger.info(f"  Distancia promedio: {merge_distances.mean():.4f}")
        logger.info(f"  Distancia mediana: {np.median(merge_distances):.4f}")

        # Tamaños de clusters
        cluster_sizes = self.linkage_matrix[:, 3]
        logger.info(f"\nTamaños finales de clusters:")
        logger.info(f"  Mínimo: {int(cluster_sizes.min())}")
        logger.info(f"  Máximo: {int(cluster_sizes.max())}")
        logger.info(f"  Promedio: {cluster_sizes.mean():.2f}")

        # Calcular coeficiente cofenético
        try:
            c, _ = cophenet(self.linkage_matrix, self.condensed_distances)
            logger.info(f"\nCorrelación cofenética: {c:.4f}")
            logger.info(f"  Interpretación: {'Excelente' if c > 0.9 else 'Buena' if c > 0.8 else 'Moderada' if c > 0.7 else 'Baja'}")
        except:
            pass

    def _validate_distance_matrix(self, distance_matrix: np.ndarray) -> None:
        """
        Valida que la matriz de distancia sea válida.

        Args:
            distance_matrix: Matriz a validar

        Raises:
            ValueError: Si la matriz es inválida
        """
        if not isinstance(distance_matrix, np.ndarray):
            raise ValueError("La matriz de distancia debe ser un numpy.ndarray")

        if distance_matrix.ndim != 2:
            raise ValueError(
                f"La matriz debe ser 2D, recibido: {distance_matrix.ndim}D"
            )

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"La matriz debe ser cuadrada, recibido: {distance_matrix.shape}"
            )

        if distance_matrix.shape[0] < 2:
            raise ValueError(
                f"Se necesitan al menos 2 muestras, recibido: {distance_matrix.shape[0]}"
            )

        if not np.allclose(distance_matrix, distance_matrix.T):
            raise ValueError("La matriz de distancia debe ser simétrica")

        if not np.allclose(np.diag(distance_matrix), 0):
            raise ValueError("La diagonal de la matriz debe ser 0")

        if np.any(distance_matrix < 0):
            raise ValueError("La matriz no puede contener distancias negativas")

        if not np.isfinite(distance_matrix).all():
            raise ValueError("La matriz contiene valores infinitos o NaN")

        logger.debug(f"Matriz de distancia validada: {distance_matrix.shape}")


def main():
    """Ejemplo de uso del clustering jerárquico."""

    print("\n" + "="*70)
    print(" EJEMPLO: Hierarchical Clustering")
    print("="*70)

    # Crear matriz de distancia de ejemplo
    np.random.seed(42)

    # Simular 8 documentos
    n_docs = 8
    labels = [f"Doc{i+1}" for i in range(n_docs)]

    # Crear matriz de distancia sintética con estructura de clusters
    # Cluster 1: Doc1, Doc2, Doc3 (similares entre sí)
    # Cluster 2: Doc4, Doc5 (similares entre sí)
    # Cluster 3: Doc6, Doc7, Doc8 (similares entre sí)

    distance_matrix = np.array([
        [0.0, 0.1, 0.15, 0.8, 0.9, 0.95, 0.85, 0.92],  # Doc1
        [0.1, 0.0, 0.12, 0.82, 0.88, 0.90, 0.87, 0.91],  # Doc2
        [0.15, 0.12, 0.0, 0.79, 0.85, 0.93, 0.84, 0.89],  # Doc3
        [0.8, 0.82, 0.79, 0.0, 0.2, 0.75, 0.78, 0.81],   # Doc4
        [0.9, 0.88, 0.85, 0.2, 0.0, 0.77, 0.80, 0.83],   # Doc5
        [0.95, 0.90, 0.93, 0.75, 0.77, 0.0, 0.18, 0.16], # Doc6
        [0.85, 0.87, 0.84, 0.78, 0.80, 0.18, 0.0, 0.14], # Doc7
        [0.92, 0.91, 0.89, 0.81, 0.83, 0.16, 0.14, 0.0]  # Doc8
    ])

    print(f"\nMatriz de distancia: {distance_matrix.shape}")
    print(f"Labels: {labels}")

    # Crear clustering jerárquico
    hc = HierarchicalClustering(distance_matrix, labels)

    # 1. Single Linkage
    print("\n" + "="*70)
    print("1. SINGLE LINKAGE")
    print("="*70)
    Z_single = hc.single_linkage()
    print(f"\nPrimeras fusiones (single):")
    print(f"Clusters {int(Z_single[0, 0])} y {int(Z_single[0, 1])} a distancia {Z_single[0, 2]:.4f}")

    # 2. Complete Linkage
    print("\n" + "="*70)
    print("2. COMPLETE LINKAGE")
    print("="*70)
    Z_complete = hc.complete_linkage()
    print(f"\nPrimeras fusiones (complete):")
    print(f"Clusters {int(Z_complete[0, 0])} y {int(Z_complete[0, 1])} a distancia {Z_complete[0, 2]:.4f}")

    # 3. Average Linkage
    print("\n" + "="*70)
    print("3. AVERAGE LINKAGE")
    print("="*70)
    Z_average = hc.average_linkage()
    print(f"\nPrimeras fusiones (average):")
    print(f"Clusters {int(Z_average[0, 0])} y {int(Z_average[0, 1])} a distancia {Z_average[0, 2]:.4f}")

    # 4. Obtener clusters
    print("\n" + "="*70)
    print("4. OBTENER CLUSTERS (k=3)")
    print("="*70)

    clusters = hc.get_clusters(n_clusters=3)
    print(f"\nAsignación de clusters:")
    for label, cluster_id in zip(labels, clusters):
        print(f"  {label}: Cluster {cluster_id}")

    # 5. Comparar métodos
    print("\n" + "="*70)
    print("5. COMPARACIÓN DE MÉTODOS")
    print("="*70)

    # Recrear objeto para comparación limpia
    hc2 = HierarchicalClustering(distance_matrix, labels)
    comparison = hc2.compare_methods()

    print(f"\nMejor método: {comparison['best_method'].upper()}")

    print("\n" + "="*70)
    print(" EJEMPLO COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
