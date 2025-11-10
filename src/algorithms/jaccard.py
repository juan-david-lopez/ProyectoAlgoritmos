"""
Algoritmo de Índice de Jaccard para comparación de similitud de textos.

Matemática:
===========
El Índice de Jaccard mide la similitud entre dos conjuntos finitos A y B,
definido como el tamaño de la intersección dividido por el tamaño de la unión.

Definición formal:
    J(A, B) = |A ∩ B| / |A ∪ B|

    donde:
    - |A ∩ B| = número de elementos en común
    - |A ∪ B| = número de elementos únicos totales

Propiedades:
    - Rango: [0, 1]
    - J(A, A) = 1 (conjunto consigo mismo)
    - J(A, B) = J(B, A) (simétrico)
    - J(A, ∅) = 0 (con conjunto vacío)
    - Si A = B = ∅, entonces J(A, B) = 1 por convención

Ejemplo con palabras:
    A = {"the", "cat", "sat", "on", "mat"}
    B = {"the", "dog", "sat", "on", "log"}

    A ∩ B = {"the", "sat", "on"}  →  |A ∩ B| = 3
    A ∪ B = {"the", "cat", "sat", "on", "mat", "dog", "log"}  →  |A ∪ B| = 7

    J(A, B) = 3/7 ≈ 0.429

Distancia de Jaccard:
    d(A, B) = 1 - J(A, B)

Complejidad:
    Tiempo: O(n + m) usando conjuntos hash
    Espacio: O(n + m) para almacenar conjuntos

Referencias:
    - Jaccard, P. (1901). Étude comparative de la distribution florale dans
      une portion des Alpes et des Jura. Bulletin de la Société Vaudoise des
      Sciences Naturelles, 37, 547-579.
"""

import numpy as np
import logging
import time
from typing import List, Set

logger = logging.getLogger(__name__)


class JaccardComparator:
    """
    Implementa comparación de similitud usando el Índice de Jaccard.

    El Índice de Jaccard trata textos como conjuntos de palabras (o tokens),
    midiendo la proporción de elementos compartidos.

    Características:
        - Simple e intuitivo
        - Rápido (tiempo lineal)
        - No considera frecuencias
        - Ignora orden de palabras
        - Útil para conjuntos de palabras clave
    """

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Inicializa el comparador de Jaccard.

        Args:
            lowercase: Convertir a minúsculas antes de comparar
            remove_punctuation: Eliminar puntuación
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        logger.info(f"Inicializando JaccardComparator (lowercase={lowercase}, remove_punctuation={remove_punctuation})")

    def tokenize(self, text: str) -> Set[str]:
        """
        Tokeniza texto en conjunto de palabras.

        Pasos:
            1. Convertir a minúsculas (si lowercase=True)
            2. Eliminar puntuación (si remove_punctuation=True)
            3. Dividir por espacios
            4. Crear conjunto (elimina duplicados automáticamente)

        Ejemplo:
            Input:  "The cat, the CAT!"
            Step 1: "the cat, the cat!"  (lowercase)
            Step 2: "the cat the cat"    (remove punctuation)
            Step 3: ["the", "cat", "the", "cat"]  (split)
            Step 4: {"the", "cat"}       (set)

        Args:
            text: Texto a tokenizar

        Returns:
            Conjunto de palabras únicas
        """
        # Convertir a minúsculas
        if self.lowercase:
            text = text.lower()

        # Eliminar puntuación
        if self.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenizar y crear conjunto
        tokens = set(text.split())

        return tokens

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calcula similitud de Jaccard entre dos conjuntos.

        Fórmula:
            J(A, B) = |A ∩ B| / |A ∪ B|

        Implementación eficiente usando operaciones de conjuntos en Python:
            - intersection: A & B  →  O(min(|A|, |B|))
            - union: A | B  →  O(|A| + |B|)

        Caso especial:
            Si ambos conjuntos están vacíos, retorna 1.0 (similitud perfecta)

        Args:
            set1: Primer conjunto
            set2: Segundo conjunto

        Returns:
            Similitud de Jaccard en [0, 1]
        """
        # Caso especial: ambos vacíos
        if len(set1) == 0 and len(set2) == 0:
            return 1.0

        # Caso especial: uno vacío
        if len(set1) == 0 or len(set2) == 0:
            return 0.0

        # Calcular intersección y unión
        intersection = set1 & set2  # Elementos en común
        union = set1 | set2          # Todos los elementos únicos

        # Calcular similitud
        similarity = len(intersection) / len(union)

        return float(similarity)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud de Jaccard entre dos textos.

        Proceso completo:
            1. Tokenizar ambos textos
            2. Calcular similitud entre conjuntos

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Similitud de Jaccard en [0, 1]
        """
        set1 = self.tokenize(text1)
        set2 = self.tokenize(text2)
        return self.jaccard_similarity(set1, set2)

    def compare_multiple(self, texts: List[str]) -> np.ndarray:
        """
        Compara múltiples textos y genera matriz de similitud.

        Para cada par de textos (i, j):
            1. Tokenizar texto i → conjunto A
            2. Tokenizar texto j → conjunto B
            3. Calcular J(A, B)
            4. Almacenar en matrix[i][j] y matrix[j][i]

        Ejemplo con 3 textos:
            T1 = "cat dog"
            T2 = "cat bird"
            T3 = "fish"

            Sets:
            S1 = {cat, dog}
            S2 = {cat, bird}
            S3 = {fish}

            Similitudes:
            J(S1, S2) = |{cat}| / |{cat,dog,bird}| = 1/3 ≈ 0.333
            J(S1, S3) = |{}| / |{cat,dog,fish}| = 0/3 = 0.0
            J(S2, S3) = |{}| / |{cat,bird,fish}| = 0/3 = 0.0

            Matriz:
                    T1    T2    T3
            T1    1.0   0.33  0.0
            T2    0.33  1.0   0.0
            T3    0.0   0.0   1.0

        Args:
            texts: Lista de textos a comparar

        Returns:
            Matriz de similitud de forma (n, n)
        """
        start_time = time.perf_counter()
        n = len(texts)

        logger.info(f"Comparando {n} textos con Jaccard...")

        # Manejar casos especiales
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([[1.0]])

        # Tokenizar todos los textos primero (optimización)
        tokenized = [self.tokenize(text) for text in texts]

        # Crear matriz de similitud
        matrix = np.zeros((n, n))

        # Calcular similitud para cada par
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self.jaccard_similarity(tokenized[i], tokenized[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim  # Simetría

        elapsed = time.perf_counter() - start_time
        logger.info(f"Jaccard completado en {elapsed:.3f}s")

        return matrix

    def get_common_words(self, text1: str, text2: str) -> Set[str]:
        """
        Obtiene palabras en común entre dos textos.

        Returns:
            Conjunto de palabras compartidas
        """
        set1 = self.tokenize(text1)
        set2 = self.tokenize(text2)
        return set1 & set2

    def get_unique_words(self, text1: str, text2: str) -> tuple:
        """
        Obtiene palabras únicas de cada texto.

        Returns:
            Tupla (únicos_text1, únicos_text2)
        """
        set1 = self.tokenize(text1)
        set2 = self.tokenize(text2)
        return (set1 - set2, set2 - set1)


# Ejemplo de uso
if __name__ == "__main__":
    comparator = JaccardComparator()

    # Ejemplo 1: Similitud básica
    text1 = "The cat sat on the mat"
    text2 = "The dog sat on the log"
    sim = comparator.similarity(text1, text2)
    print(f"Similitud: {sim:.3f}")

    # Mostrar análisis detallado
    common = comparator.get_common_words(text1, text2)
    unique1, unique2 = comparator.get_unique_words(text1, text2)
    print(f"Palabras comunes: {common}")
    print(f"Únicas en texto 1: {unique1}")
    print(f"Únicas en texto 2: {unique2}")

    # Ejemplo 2: Comparación múltiple
    texts = [
        "cat dog bird",
        "cat fish",
        "dog bird"
    ]
    matrix = comparator.compare_multiple(texts)
    print("\nMatriz de similitud:")
    print(matrix)
