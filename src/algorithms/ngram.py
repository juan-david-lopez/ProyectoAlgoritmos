"""
Algoritmo de Similitud de N-gramas para comparación de textos.

Matemática:
===========
Un n-grama es una secuencia contigua de n caracteres de un texto.

Ejemplos con n=3 (trigramas):
    "hello" → {"hel", "ell", "llo"}
    "world" → {"wor", "orl", "rld"}

La similitud se calcula usando el coeficiente de Dice o Jaccard sobre
el conjunto de n-gramas:

Similitud de Dice:
    sim(A, B) = 2 × |A ∩ B| / (|A| + |B|)

Similitud de Jaccard:
    sim(A, B) = |A ∩ B| / |A ∪ B|

Ventajas sobre palabras completas:
    - Robusto a errores tipográficos
    - Funciona con diferentes idiomas
    - Captura similitud de subcadenas
    - No requiere tokenización por espacios

Ejemplo detallado (n=2, bigramas):
    s1 = "abc"  →  {"ab", "bc"}
    s2 = "bcd"  →  {"bc", "cd"}

    Intersección: {"bc"}  →  |A ∩ B| = 1
    Unión: {"ab", "bc", "cd"}  →  |A ∪ B| = 3
    Suma: |A| + |B| = 2 + 2 = 4

    Jaccard: 1/3 ≈ 0.333
    Dice: 2×1/4 = 0.500

Elección de n:
    - n=2 (bigramas): Muy permisivo, detecta similitud débil
    - n=3 (trigramas): Balance óptimo para la mayoría de casos
    - n=4+ : Más estricto, requiere subcadenas más largas

Complejidad:
    Tiempo: O(m + n) donde m, n son longitudes de textos
    Espacio: O(m + n) para almacenar n-gramas

Referencias:
    - Kondrak, G. (2005). N-Gram Similarity and Distance. String Processing
      and Information Retrieval, 115-126.
"""

import numpy as np
import logging
import time
from typing import List, Set

logger = logging.getLogger(__name__)


class NGramComparator:
    """
    Implementa comparación de similitud usando n-gramas.

    Los n-gramas capturan similitud a nivel de subcadenas, siendo robustos
    a errores y útiles para diferentes idiomas.

    Características:
        - Robusto a errores tipográficos
        - Independiente del idioma
        - Captura patrones locales
        - No requiere espacios para tokenizar
    """

    def __init__(self, n: int = 3, method: str = 'dice'):
        """
        Inicializa el comparador de n-gramas.

        Args:
            n: Longitud de los n-gramas (típicamente 2-4)
            method: Método de similitud ('dice' o 'jaccard')
        """
        self.n = n
        self.method = method
        logger.info(f"Inicializando NGramComparator (n={n}, method={method})")

    def extract_ngrams(self, text: str) -> Set[str]:
        """
        Extrae n-gramas de un texto.

        Proceso paso a paso para "hello" con n=3:
            1. Posición 0: "hel"
            2. Posición 1: "ell"
            3. Posición 2: "llo"
            4. Resultado: {"hel", "ell", "llo"}

        Casos especiales:
            - Si len(text) < n: retorna {text}
            - Texto vacío: retorna set()

        Args:
            text: Texto de entrada

        Returns:
            Conjunto de n-gramas
        """
        # Caso especial: texto muy corto
        if len(text) < self.n:
            return {text} if text else set()

        # Extraer n-gramas con ventana deslizante
        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            ngrams.add(ngram)

        return ngrams

    def dice_coefficient(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calcula coeficiente de Dice entre dos conjuntos.

        Fórmula:
            Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)

        El coeficiente de Dice da más peso a elementos comunes que Jaccard.

        Propiedades:
            - Rango: [0, 1]
            - Dice ≥ Jaccard siempre
            - Más tolerante a diferencias

        Args:
            set1: Primer conjunto
            set2: Segundo conjunto

        Returns:
            Coeficiente de Dice en [0, 1]
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        if len(set1) == 0 or len(set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        total = len(set1) + len(set2)

        return (2.0 * intersection) / total

    def jaccard_coefficient(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calcula coeficiente de Jaccard entre dos conjuntos.

        Fórmula:
            Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Args:
            set1: Primer conjunto
            set2: Segundo conjunto

        Returns:
            Coeficiente de Jaccard en [0, 1]
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        if len(set1) == 0 or len(set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud entre dos textos usando n-gramas.

        Ejemplo completo con n=2:
            text1 = "night"
            text2 = "nacht"

            N-gramas:
            text1: {"ni", "ig", "gh", "ht"}  (4 bigramas)
            text2: {"na", "ac", "ch", "ht"}  (4 bigramas)

            Intersección: {"ht"}  (1 bigrama común)

            Dice: 2×1 / (4+4) = 2/8 = 0.25
            Jaccard: 1 / 7 ≈ 0.14

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Similitud en [0, 1]
        """
        # Extraer n-gramas
        ngrams1 = self.extract_ngrams(text1)
        ngrams2 = self.extract_ngrams(text2)

        # Calcular similitud según método
        if self.method == 'dice':
            return self.dice_coefficient(ngrams1, ngrams2)
        elif self.method == 'jaccard':
            return self.jaccard_coefficient(ngrams1, ngrams2)
        else:
            raise ValueError(f"Método desconocido: {self.method}")

    def compare_multiple(self, texts: List[str]) -> np.ndarray:
        """
        Compara múltiples textos usando n-gramas.

        Args:
            texts: Lista de textos a comparar

        Returns:
            Matriz de similitud de forma (n, n)
        """
        start_time = time.perf_counter()
        n = len(texts)

        logger.info(f"Comparando {n} textos con {self.n}-gramas ({self.method})...")

        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([[1.0]])

        # Extraer n-gramas de todos los textos (optimización)
        ngrams_list = [self.extract_ngrams(text) for text in texts]

        # Crear matriz de similitud
        matrix = np.zeros((n, n))

        # Calcular similitud para cada par
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    if self.method == 'dice':
                        sim = self.dice_coefficient(ngrams_list[i], ngrams_list[j])
                    else:
                        sim = self.jaccard_coefficient(ngrams_list[i], ngrams_list[j])

                    matrix[i][j] = sim
                    matrix[j][i] = sim

        elapsed = time.perf_counter() - start_time
        logger.info(f"N-grama completado en {elapsed:.3f}s")

        return matrix


# Ejemplo de uso
if __name__ == "__main__":
    # Comparar con diferentes valores de n
    for n in [2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"N-gramas con n={n}")
        print('='*50)

        comparator = NGramComparator(n=n, method='dice')

        text1 = "kitten"
        text2 = "sitting"

        ngrams1 = comparator.extract_ngrams(text1)
        ngrams2 = comparator.extract_ngrams(text2)

        print(f"Text1 '{text1}': {ngrams1}")
        print(f"Text2 '{text2}': {ngrams2}")
        print(f"Común: {ngrams1 & ngrams2}")
        print(f"Similitud: {comparator.similarity(text1, text2):.3f}")

    # Comparación múltiple
    print(f"\n{'='*50}")
    print("Comparación múltiple")
    print('='*50)

    comparator = NGramComparator(n=3)
    texts = ["hello", "hallo", "hola"]
    matrix = comparator.compare_multiple(texts)
    print(matrix)
