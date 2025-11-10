"""
Algoritmo de Distancia de Levenshtein para comparación de similitud de textos.

Matemática:
===========
La distancia de Levenshtein (LD) entre dos cadenas s₁ y s₂ se define como el número
mínimo de operaciones de edición (inserción, eliminación, sustitución) necesarias
para transformar s₁ en s₂.

Definición recursiva:
    LD(i, j) = max(i, j)                                    si min(i,j) = 0
    LD(i, j) = min {
                    LD(i-1, j) + 1           (eliminación)
                    LD(i, j-1) + 1           (inserción)
                    LD(i-1, j-1) + cost      (sustitución)
               }                                            si min(i,j) > 0

    donde cost = 0 si s₁[i] = s₂[j], sino cost = 1

La similitud normalizada se calcula como:
    Similitud = 1 - (LD / max(len(s₁), len(s₂)))

Esto garantiza que el resultado esté en el rango [0, 1], donde:
    - 1 = textos idénticos
    - 0 = textos completamente diferentes

Complejidad:
    Tiempo: O(m × n) donde m, n son longitudes de las cadenas
    Espacio: O(m × n) con matriz completa, O(min(m,n)) con optimización

Referencias:
    - Levenshtein, V. I. (1966). Binary codes capable of correcting deletions,
      insertions, and reversals. Soviet Physics Doklady, 10(8), 707-710.
"""

import numpy as np
import logging
import time
from typing import List

logger = logging.getLogger(__name__)


class LevenshteinComparator:
    """
    Implementa comparación de similitud usando distancia de Levenshtein.

    La distancia de Levenshtein (o distancia de edición) es una métrica de
    similitud que cuenta el número mínimo de operaciones necesarias para
    transformar una cadena en otra.

    Características:
        - No requiere preprocesamiento
        - Sensible al orden de caracteres
        - Útil para textos cortos
        - Complejidad cuadrática O(n²)
    """

    def __init__(self):
        """Inicializa el comparador de Levenshtein."""
        logger.info("Inicializando LevenshteinComparator")

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calcula la distancia de Levenshtein entre dos cadenas.

        Implementación usando programación dinámica con matriz completa.

        Ejemplo paso a paso:
            s1 = "kitten"
            s2 = "sitting"

            Matriz DP:
                    ""  s  i  t  t  i  n  g
                ""   0  1  2  3  4  5  6  7
                k    1  1  2  3  4  5  6  7
                i    2  2  1  2  3  4  5  6
                t    3  3  2  1  2  3  4  5
                t    4  4  3  2  1  2  3  4
                e    5  5  4  3  2  2  3  4
                n    6  6  5  4  3  3  2  3

            Distancia = 3 (última celda)
            Operaciones: k→s, e→i, insertar g

        Args:
            s1: Primera cadena
            s2: Segunda cadena

        Returns:
            Distancia de Levenshtein (número de operaciones)
        """
        # Casos base
        if s1 == s2:
            return 0
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)

        # Crear matriz de programación dinámica
        # dp[i][j] = distancia entre s1[0:i] y s2[0:j]
        m, n = len(s1), len(s2)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        # Inicializar primera fila y columna
        # (distancia a cadena vacía = longitud)
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Llenar matriz usando la recurrencia
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Costo de sustitución: 0 si caracteres iguales, 1 si diferentes
                cost = 0 if s1[i - 1] == s2[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Eliminación
                    dp[i][j - 1] + 1,      # Inserción
                    dp[i - 1][j - 1] + cost  # Sustitución
                )

        return int(dp[m][n])

    def similarity(self, s1: str, s2: str) -> float:
        """
        Calcula similitud normalizada entre dos cadenas.

        La similitud se normaliza dividiendo por la longitud máxima:
            sim = 1 - (distance / max(len(s1), len(s2)))

        Esto garantiza:
            - Resultado en [0, 1]
            - 1 para textos idénticos
            - 0 para máxima diferencia

        Args:
            s1: Primera cadena
            s2: Segunda cadena

        Returns:
            Similitud normalizada en [0, 1]
        """
        # Manejar casos especiales
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Calcular distancia
        distance = self.levenshtein_distance(s1, s2)

        # Normalizar por longitud máxima
        max_len = max(len(s1), len(s2))
        similarity = 1.0 - (distance / max_len)

        return float(similarity)

    def compare_multiple(self, texts: List[str]) -> np.ndarray:
        """
        Compara múltiples textos y genera matriz de similitud.

        Calcula similitud por pares para todos los textos, generando una
        matriz simétrica donde:
            - Diagonal = 1.0 (texto consigo mismo)
            - matrix[i][j] = similitud entre textos i y j

        Args:
            texts: Lista de textos a comparar

        Returns:
            Matriz de similitud de forma (n, n)
        """
        start_time = time.perf_counter()
        n = len(texts)
        matrix = np.zeros((n, n))

        logger.info(f"Comparando {n} textos con Levenshtein...")

        # Calcular similitud para cada par
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self.similarity(texts[i], texts[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim  # Simetría

        elapsed = time.perf_counter() - start_time
        logger.info(f"Levenshtein completado en {elapsed:.3f}s")

        return matrix


# Ejemplo de uso
if __name__ == "__main__":
    comparator = LevenshteinComparator()

    # Ejemplo 1: Similitud básica
    s1 = "kitten"
    s2 = "sitting"
    print(f"Similitud entre '{s1}' y '{s2}': {comparator.similarity(s1, s2):.3f}")

    # Ejemplo 2: Comparación múltiple
    texts = [
        "The quick brown fox",
        "The quick brown dog",
        "A fast brown fox"
    ]
    matrix = comparator.compare_multiple(texts)
    print("\nMatriz de similitud:")
    print(matrix)
