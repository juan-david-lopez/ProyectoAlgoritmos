"""
Distance Calculator Module
Cálculo de matrices de distancia y similitud para clustering jerárquico.

Características:
- Múltiples métricas de distancia: cosine, euclidean, manhattan, correlation
- Implementación eficiente con sklearn y scipy
- Validación de inputs y manejo de errores
- Matrices simétricas optimizadas
- Logging detallado
- Explicaciones matemáticas claras

Métricas disponibles:
- cosine: Distancia coseno (1 - similitud coseno)
- euclidean: Distancia euclidiana (L2)
- manhattan: Distancia Manhattan (L1)
- correlation: Distancia de correlación (1 - correlación de Pearson)
"""

import numpy as np
import logging
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn para cálculos eficientes de distancia
from sklearn.metrics.pairwise import (
    cosine_similarity,
    cosine_distances,
    euclidean_distances,
    manhattan_distances
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistanceCalculator:
    """
    Calculador de matrices de distancia/similitud para clustering.

    Proporciona múltiples métricas de distancia optimizadas para análisis
    de similitud entre documentos vectorizados.

    Métricas soportadas:
    - cosine: Mejor para vectores TF-IDF y embeddings normalizados
    - euclidean: Distancia geométrica clásica
    - manhattan: Suma de diferencias absolutas
    - correlation: Basada en correlación de Pearson
    """

    SUPPORTED_METRICS = ['cosine', 'euclidean', 'manhattan', 'correlation']

    def __init__(self):
        """Inicializa el calculador de distancias."""
        logger.info("DistanceCalculator inicializado")

    def cosine_distance(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancia coseno entre todos los pares de vectores.

        Explicación matemática:
        =====================
        La distancia coseno mide el ángulo entre dos vectores:

        cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
                                = Σ(ai × bi) / (√Σ(ai²) × √Σ(bi²))

        cosine_distance(A, B) = 1 - cosine_similarity(A, B)

        Rango: [0, 2]
        - 0: vectores idénticos (ángulo 0°)
        - 1: vectores ortogonales (ángulo 90°)
        - 2: vectores opuestos (ángulo 180°)

        Propiedades:
        - Invariante a escala (solo importa dirección, no magnitud)
        - Ideal para vectores TF-IDF y embeddings
        - Matriz simétrica con diagonal = 0

        Args:
            vectors: Matriz de vectores (n_samples, n_features)

        Returns:
            Matriz de distancia coseno (n_samples, n_samples)

        Raises:
            ValueError: Si la matriz está vacía o tiene dimensiones inválidas

        Example:
            >>> vectors = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
            >>> dist_calc = DistanceCalculator()
            >>> distances = dist_calc.cosine_distance(vectors)
            >>> print(distances.shape)
            (3, 3)
        """
        # Validar input
        self._validate_vectors(vectors)

        n_samples = vectors.shape[0]
        logger.info(f"Calculando distancia coseno para {n_samples} muestras")

        # Usar sklearn para cálculo eficiente
        # cosine_distances ya calcula 1 - cosine_similarity
        distance_matrix = cosine_distances(vectors)

        # Verificar que es simétrica
        assert np.allclose(distance_matrix, distance_matrix.T), "Matriz no simétrica"

        # Verificar diagonal es 0 (distancia consigo mismo)
        assert np.allclose(np.diag(distance_matrix), 0), "Diagonal no es cero"

        logger.info(f"Distancia coseno calculada: {distance_matrix.shape}")
        logger.info(f"  Min: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].min():.4f}")
        logger.info(f"  Max: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].max():.4f}")
        logger.info(f"  Media: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.4f}")

        return distance_matrix

    def euclidean_distance(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancia euclidiana entre todos los pares de vectores.

        Explicación matemática:
        =====================
        La distancia euclidiana es la distancia geométrica en el espacio:

        d(A, B) = ||A - B|| = √(Σ(ai - bi)²)

        Equivalente a la longitud del segmento que une A y B.

        Rango: [0, +∞)
        - 0: vectores idénticos
        - ∞: vectores muy diferentes

        Propiedades:
        - Sensible a magnitud y dirección
        - Métrica estándar del espacio euclidiano
        - Matriz simétrica con diagonal = 0

        Nota sobre normalización:
        ========================
        Para vectores TF-IDF normalizados (||v|| = 1), la distancia euclidiana
        está relacionada con la distancia coseno:

        d_euclidean² = 2 × (1 - cosine_similarity)
        d_euclidean = √(2 × d_cosine)

        En este caso, ambas métricas producen rankings similares, pero
        cosine_distance es más interpretable.

        Args:
            vectors: Matriz de vectores (n_samples, n_features)

        Returns:
            Matriz de distancia euclidiana (n_samples, n_samples)

        Raises:
            ValueError: Si la matriz está vacía o tiene dimensiones inválidas

        Example:
            >>> vectors = np.array([[1, 0], [0, 1], [1, 1]])
            >>> dist_calc = DistanceCalculator()
            >>> distances = dist_calc.euclidean_distance(vectors)
            >>> print(distances[0, 1])  # Distancia entre [1,0] y [0,1]
            1.414...
        """
        # Validar input
        self._validate_vectors(vectors)

        n_samples = vectors.shape[0]
        logger.info(f"Calculando distancia euclidiana para {n_samples} muestras")

        # Usar sklearn para cálculo eficiente
        distance_matrix = euclidean_distances(vectors)

        # Verificar que es simétrica
        assert np.allclose(distance_matrix, distance_matrix.T), "Matriz no simétrica"

        # Verificar diagonal es 0
        assert np.allclose(np.diag(distance_matrix), 0), "Diagonal no es cero"

        logger.info(f"Distancia euclidiana calculada: {distance_matrix.shape}")
        logger.info(f"  Min: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].min():.4f}")
        logger.info(f"  Max: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].max():.4f}")
        logger.info(f"  Media: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.4f}")

        return distance_matrix

    def manhattan_distance(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancia Manhattan (L1) entre todos los pares.

        Explicación matemática:
        =====================
        La distancia Manhattan (también llamada "city block" o "taxicab"):

        d(A, B) = Σ|ai - bi|

        Es la suma de las diferencias absolutas en cada dimensión.
        Equivale a la distancia recorrida en una cuadrícula (como calles de ciudad).

        Rango: [0, +∞)

        Propiedades:
        - Menos sensible a outliers que euclidiana
        - Útil en espacios de alta dimensionalidad
        - Computacionalmente eficiente

        Args:
            vectors: Matriz de vectores (n_samples, n_features)

        Returns:
            Matriz de distancia Manhattan (n_samples, n_samples)

        Raises:
            ValueError: Si la matriz está vacía o tiene dimensiones inválidas
        """
        # Validar input
        self._validate_vectors(vectors)

        n_samples = vectors.shape[0]
        logger.info(f"Calculando distancia Manhattan para {n_samples} muestras")

        # Usar sklearn para cálculo eficiente
        distance_matrix = manhattan_distances(vectors)

        # Verificar simetría
        assert np.allclose(distance_matrix, distance_matrix.T), "Matriz no simétrica"

        logger.info(f"Distancia Manhattan calculada: {distance_matrix.shape}")
        logger.info(f"  Min: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].min():.4f}")
        logger.info(f"  Max: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].max():.4f}")
        logger.info(f"  Media: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.4f}")

        return distance_matrix

    def correlation_distance(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de distancia de correlación entre todos los pares.

        Explicación matemática:
        =====================
        La distancia de correlación se basa en la correlación de Pearson:

        correlation(A, B) = cov(A, B) / (σ_A × σ_B)

        donde:
        - cov(A, B) = Σ((ai - mean(A)) × (bi - mean(B))) / n
        - σ_A = desviación estándar de A

        d_correlation(A, B) = 1 - correlation(A, B)

        Rango: [0, 2]
        - 0: correlación perfecta positiva
        - 1: sin correlación
        - 2: correlación perfecta negativa

        Propiedades:
        - Invariante a transformaciones afines (escala y traslación)
        - Captura relaciones lineales
        - Similar a cosine pero centra los vectores

        Args:
            vectors: Matriz de vectores (n_samples, n_features)

        Returns:
            Matriz de distancia de correlación (n_samples, n_samples)

        Raises:
            ValueError: Si la matriz está vacía o tiene dimensiones inválidas
        """
        # Validar input
        self._validate_vectors(vectors)

        n_samples = vectors.shape[0]
        logger.info(f"Calculando distancia de correlación para {n_samples} muestras")

        # Usar scipy.spatial.distance.pdist con métrica 'correlation'
        # pdist calcula solo la parte superior triangular (más eficiente)
        condensed_distances = pdist(vectors, metric='correlation')

        # Convertir a matriz cuadrada
        distance_matrix = squareform(condensed_distances)

        # Verificar simetría
        assert np.allclose(distance_matrix, distance_matrix.T), "Matriz no simétrica"

        # Manejar NaNs que pueden aparecer con vectores constantes
        if np.any(np.isnan(distance_matrix)):
            logger.warning("Valores NaN detectados en matriz de correlación (vectores constantes?)")
            # Reemplazar NaN con 0 (sin distancia consigo mismo)
            np.fill_diagonal(distance_matrix, 0)
            # Reemplazar otros NaN con distancia máxima
            distance_matrix = np.nan_to_num(distance_matrix, nan=2.0)

        logger.info(f"Distancia de correlación calculada: {distance_matrix.shape}")
        logger.info(f"  Min: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].min():.4f}")
        logger.info(f"  Max: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].max():.4f}")
        logger.info(f"  Media: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.4f}")

        return distance_matrix

    def calculate_distance_matrix(self,
                                  vectors: np.ndarray,
                                  metric: str = 'cosine') -> np.ndarray:
        """
        Calcula matriz de distancia completa con la métrica especificada.

        Este es el método principal y más flexible del calculador.

        Métricas disponibles:
        ====================
        1. 'cosine': Distancia coseno (recomendada para TF-IDF y embeddings)
           - Rango: [0, 2]
           - Invariante a escala
           - Mide similitud angular

        2. 'euclidean': Distancia euclidiana (L2)
           - Rango: [0, +∞)
           - Distancia geométrica estándar
           - Sensible a magnitud

        3. 'manhattan': Distancia Manhattan (L1)
           - Rango: [0, +∞)
           - Suma de diferencias absolutas
           - Robusta a outliers

        4. 'correlation': Distancia de correlación
           - Rango: [0, 2]
           - Basada en correlación de Pearson
           - Invariante a transformaciones afines

        Recomendaciones por tipo de datos:
        =================================
        - TF-IDF: cosine
        - Word2Vec/SBERT: cosine o euclidean
        - Datos con outliers: manhattan
        - Series temporales: correlation

        Args:
            vectors: Matriz de vectores (n_samples, n_features)
            metric: Métrica de distancia ('cosine', 'euclidean', 'manhattan', 'correlation')

        Returns:
            Matriz de distancia simétrica (n_samples, n_samples)

        Raises:
            ValueError: Si la métrica no es soportada o la matriz es inválida

        Example:
            >>> vectors = np.random.rand(100, 50)
            >>> dist_calc = DistanceCalculator()
            >>> distances = dist_calc.calculate_distance_matrix(vectors, metric='cosine')
            >>> print(distances.shape)
            (100, 100)
        """
        # Validar métrica
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Métrica '{metric}' no soportada. "
                f"Usar una de: {', '.join(self.SUPPORTED_METRICS)}"
            )

        # Validar vectores
        self._validate_vectors(vectors)

        logger.info("="*70)
        logger.info("CALCULANDO MATRIZ DE DISTANCIA")
        logger.info("="*70)
        logger.info(f"Métrica: {metric}")
        logger.info(f"Vectores: {vectors.shape}")

        # Calcular según métrica
        if metric == 'cosine':
            distance_matrix = self.cosine_distance(vectors)

        elif metric == 'euclidean':
            distance_matrix = self.euclidean_distance(vectors)

        elif metric == 'manhattan':
            distance_matrix = self.manhattan_distance(vectors)

        elif metric == 'correlation':
            distance_matrix = self.correlation_distance(vectors)

        logger.info("="*70)
        logger.info("MATRIZ DE DISTANCIA COMPLETADA")
        logger.info("="*70)

        return distance_matrix

    def calculate_similarity_matrix(self,
                                   vectors: np.ndarray,
                                   metric: str = 'cosine') -> np.ndarray:
        """
        Calcula matriz de similitud (inverso de distancia).

        Convierte distancias a similitudes:
        - Para cosine y correlation: similarity = 1 - distance
        - Para euclidean y manhattan: similarity = 1 / (1 + distance)

        Args:
            vectors: Matriz de vectores (n_samples, n_features)
            metric: Métrica de distancia

        Returns:
            Matriz de similitud (n_samples, n_samples)

        Example:
            >>> vectors = np.random.rand(10, 20)
            >>> dist_calc = DistanceCalculator()
            >>> similarities = dist_calc.calculate_similarity_matrix(vectors)
            >>> # Valores altos = más similares
        """
        logger.info(f"Calculando matriz de similitud con métrica {metric}")

        # Calcular distancias
        distance_matrix = self.calculate_distance_matrix(vectors, metric=metric)

        # Convertir a similitud según métrica
        if metric in ['cosine', 'correlation']:
            # Para métricas acotadas: similarity = 1 - distance
            similarity_matrix = 1 - distance_matrix
        else:
            # Para métricas no acotadas: similarity = 1 / (1 + distance)
            similarity_matrix = 1 / (1 + distance_matrix)

        logger.info(f"Matriz de similitud calculada: {similarity_matrix.shape}")
        logger.info(f"  Min: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].min():.4f}")
        logger.info(f"  Max: {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].max():.4f}")

        return similarity_matrix

    def compare_metrics(self, vectors: np.ndarray) -> dict:
        """
        Compara todas las métricas de distancia disponibles.

        Útil para explorar qué métrica es más apropiada para tus datos.

        Args:
            vectors: Matriz de vectores (n_samples, n_features)

        Returns:
            Diccionario con matrices de distancia para cada métrica:
            {
                'cosine': matriz_cosine,
                'euclidean': matriz_euclidean,
                'manhattan': matriz_manhattan,
                'correlation': matriz_correlation,
                'statistics': estadísticas comparativas
            }

        Example:
            >>> vectors = np.random.rand(50, 30)
            >>> dist_calc = DistanceCalculator()
            >>> comparison = dist_calc.compare_metrics(vectors)
            >>> for metric, stats in comparison['statistics'].items():
            ...     print(f"{metric}: mean={stats['mean']:.4f}")
        """
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN DE MÉTRICAS")
        logger.info("="*70)

        results = {}
        statistics = {}

        for metric in self.SUPPORTED_METRICS:
            logger.info(f"\nCalculando {metric}...")

            try:
                distance_matrix = self.calculate_distance_matrix(vectors, metric=metric)
                results[metric] = distance_matrix

                # Calcular estadísticas (solo triángulo superior, sin diagonal)
                upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

                statistics[metric] = {
                    'min': float(upper_triangle.min()),
                    'max': float(upper_triangle.max()),
                    'mean': float(upper_triangle.mean()),
                    'std': float(upper_triangle.std()),
                    'median': float(np.median(upper_triangle))
                }

            except Exception as e:
                logger.error(f"Error calculando {metric}: {e}")
                results[metric] = None
                statistics[metric] = None

        results['statistics'] = statistics

        # Imprimir resumen
        logger.info("\n" + "="*70)
        logger.info("RESUMEN DE MÉTRICAS")
        logger.info("="*70)

        for metric, stats in statistics.items():
            if stats:
                logger.info(f"\n{metric.upper()}:")
                logger.info(f"  Min:    {stats['min']:.4f}")
                logger.info(f"  Max:    {stats['max']:.4f}")
                logger.info(f"  Media:  {stats['mean']:.4f}")
                logger.info(f"  Std:    {stats['std']:.4f}")
                logger.info(f"  Mediana: {stats['median']:.4f}")

        return results

    def _validate_vectors(self, vectors: np.ndarray) -> None:
        """
        Valida que la matriz de vectores sea válida.

        Args:
            vectors: Matriz a validar

        Raises:
            ValueError: Si la matriz es inválida
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Los vectores deben ser un numpy.ndarray")

        if vectors.ndim != 2:
            raise ValueError(
                f"Los vectores deben ser 2D (n_samples, n_features), "
                f"recibido: {vectors.ndim}D"
            )

        if vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError(
                f"Matriz vacía recibida: {vectors.shape}"
            )

        if vectors.shape[0] < 2:
            raise ValueError(
                f"Se necesitan al menos 2 muestras para calcular distancias, "
                f"recibido: {vectors.shape[0]}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "La matriz contiene valores infinitos o NaN"
            )

        logger.debug(f"Vectores validados: {vectors.shape}")


def main():
    """Ejemplo de uso del calculador de distancias."""

    print("\n" + "="*70)
    print(" EJEMPLO: Distance Calculator")
    print("="*70)

    # Crear vectores de ejemplo (documentos en espacio TF-IDF simulado)
    np.random.seed(42)

    # 5 documentos, 20 features
    vectors = np.random.rand(5, 20)

    # Normalizar vectores (simulando TF-IDF normalizado)
    from sklearn.preprocessing import normalize
    vectors = normalize(vectors, norm='l2')

    print(f"\nVectores: {vectors.shape}")
    print(f"Norma del primer vector: {np.linalg.norm(vectors[0]):.4f}")

    # Crear calculador
    calc = DistanceCalculator()

    # 1. Distancia coseno
    print("\n" + "="*70)
    print("1. DISTANCIA COSENO")
    print("="*70)

    dist_cosine = calc.cosine_distance(vectors)
    print(f"\nMatriz de distancia coseno:")
    print(dist_cosine.round(4))

    # 2. Distancia euclidiana
    print("\n" + "="*70)
    print("2. DISTANCIA EUCLIDIANA")
    print("="*70)

    dist_euclidean = calc.euclidean_distance(vectors)
    print(f"\nMatriz de distancia euclidiana:")
    print(dist_euclidean.round(4))

    # 3. Comparar todas las métricas
    print("\n" + "="*70)
    print("3. COMPARACIÓN DE MÉTRICAS")
    print("="*70)

    comparison = calc.compare_metrics(vectors)

    # 4. Matriz de similitud
    print("\n" + "="*70)
    print("4. MATRIZ DE SIMILITUD")
    print("="*70)

    similarity = calc.calculate_similarity_matrix(vectors, metric='cosine')
    print(f"\nMatriz de similitud coseno:")
    print(similarity.round(4))

    # 5. Encontrar documentos más similares
    print("\n" + "="*70)
    print("5. DOCUMENTOS MÁS SIMILARES")
    print("="*70)

    # Para cada documento, encontrar el más similar (excluyéndose a sí mismo)
    for i in range(len(vectors)):
        # Distancias desde documento i
        distances_from_i = dist_cosine[i].copy()
        distances_from_i[i] = np.inf  # Ignorar distancia consigo mismo

        # Documento más cercano
        closest_idx = np.argmin(distances_from_i)
        closest_dist = distances_from_i[closest_idx]

        print(f"\nDocumento {i} más similar a Documento {closest_idx} (distancia: {closest_dist:.4f})")

    print("\n" + "="*70)
    print(" ✓ EJEMPLO COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
