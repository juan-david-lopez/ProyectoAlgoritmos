"""
Algoritmo TF-IDF con Similitud del Coseno para comparación de textos.

Matemática:
===========
TF-IDF (Term Frequency-Inverse Document Frequency) es una medida estadística
que evalúa la importancia de una palabra en un documento dentro de una colección.

1. Term Frequency (TF):
   TF(t, d) = (Número de veces que t aparece en d) / (Total de términos en d)

   Variante log-normalizada:
   TF(t, d) = 1 + log(freq(t, d))  si freq(t, d) > 0
            = 0                     si freq(t, d) = 0

2. Inverse Document Frequency (IDF):
   IDF(t, D) = log(N / df(t))

   donde:
   - N = número total de documentos
   - df(t) = número de documentos que contienen t

3. TF-IDF:
   TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

4. Similitud del Coseno:
   La similitud entre dos vectores TF-IDF v₁ y v₂ se calcula como:

   sim(v₁, v₂) = cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)
                        = Σᵢ(v₁ᵢ × v₂ᵢ) / (√Σᵢv₁ᵢ² × √Σᵢv₂ᵢ²)

   Propiedades:
   - Rango: [0, 1] (considerando solo valores no negativos)
   - 1 = vectores idénticos (misma dirección)
   - 0 = vectores ortogonales (sin términos comunes)

Complejidad:
    Tiempo: O(n × m) donde n = documentos, m = términos únicos
    Espacio: O(n × m) matriz dispersa (sparse)

Referencias:
    - Salton, G., & McGill, M. J. (1983). Introduction to modern information
      retrieval. McGraw-Hill.
"""

import numpy as np
import logging
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TFIDFCosineComparator:
    """
    Implementa comparación de similitud usando TF-IDF y similitud del coseno.

    TF-IDF pondera términos por su importancia relativa en la colección,
    y la similitud del coseno mide el ángulo entre vectores de documentos.

    Características:
        - Rápido y escalable
        - Independiente de longitud del documento
        - Considera importancia de términos
        - Ignora orden de palabras
    """

    def __init__(self, max_features: int = None, ngram_range: tuple = (1, 1)):
        """
        Inicializa el comparador TF-IDF.

        Args:
            max_features: Número máximo de características (vocabulario limitado)
            ngram_range: Rango de n-gramas (1,1) = unigramas, (1,2) = uni+bigramas
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        logger.info(f"Inicializando TFIDFCosineComparator (max_features={max_features}, ngram_range={ngram_range})")

    def compare_multiple(self, texts: List[str]) -> np.ndarray:
        """
        Compara múltiples textos usando TF-IDF y similitud del coseno.

        Proceso paso a paso:
            1. Tokenización: "The cat sat" → ["the", "cat", "sat"]
            2. Construcción de vocabulario: {the: 0, cat: 1, sat: 2, ...}
            3. Cálculo TF-IDF:
               - Para cada documento, calcular TF para cada término
               - Calcular IDF global
               - Multiplicar TF × IDF
            4. Normalización L2: dividir cada vector por su norma
            5. Similitud del coseno: producto punto entre vectores normalizados

        Ejemplo con 2 documentos:
            D1 = "cat sat mat"
            D2 = "cat sat hat"

            Vocabulario: {cat, sat, mat, hat}

            Matriz TF:
                     cat  sat  mat  hat
            D1:     0.33 0.33 0.33  0
            D2:     0.33 0.33  0   0.33

            IDF (N=2):
            - cat: log(2/2) = 0
            - sat: log(2/2) = 0
            - mat: log(2/1) = 0.693
            - hat: log(2/1) = 0.693

            TF-IDF:
                     cat  sat   mat   hat
            D1:      0    0    0.231   0
            D2:      0    0     0    0.231

            Similitud del coseno = 0.0 (términos únicos diferentes)

        Args:
            texts: Lista de textos a comparar

        Returns:
            Matriz de similitud de forma (n, n)
        """
        start_time = time.perf_counter()
        n = len(texts)

        logger.info(f"Comparando {n} textos con TF-IDF + Coseno...")

        # Manejar casos especiales
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([[1.0]])

        # Crear vectorizador TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            stop_words='english',  # Eliminar palabras comunes
            use_idf=True,          # Usar IDF
            smooth_idf=True,       # Suavizar IDF: log((N+1)/(df+1)) + 1
            sublinear_tf=True      # Usar escala log para TF: 1 + log(tf)
        )

        # Transformar textos a matriz TF-IDF
        # Resultado: matriz dispersa (n_documentos, n_términos)
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        logger.debug(f"Matriz TF-IDF: {tfidf_matrix.shape}")
        logger.debug(f"Vocabulario: {len(self.vectorizer.vocabulary_)} términos")

        # Calcular similitud del coseno entre todos los pares
        # cosine_similarity usa producto punto optimizado
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Asegurar que diagonal = 1.0 (similitud consigo mismo)
        np.fill_diagonal(similarity_matrix, 1.0)

        # Clipar valores al rango [0, 1] (por precisión numérica)
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        elapsed = time.perf_counter() - start_time
        logger.info(f"TF-IDF + Coseno completado en {elapsed:.3f}s")

        return similarity_matrix

    def get_feature_names(self) -> List[str]:
        """
        Obtiene los términos del vocabulario.

        Returns:
            Lista de términos en el vocabulario
        """
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_terms(self, text: str, n: int = 10) -> List[tuple]:
        """
        Obtiene los términos más importantes de un texto según TF-IDF.

        Args:
            text: Texto a analizar
            n: Número de términos a retornar

        Returns:
            Lista de tuplas (término, score TF-IDF)
        """
        if self.vectorizer is None:
            logger.warning("Vectorizador no entrenado. Llame a compare_multiple primero.")
            return []

        # Transformar texto
        tfidf_vector = self.vectorizer.transform([text])

        # Obtener términos y scores
        feature_names = self.get_feature_names()
        scores = tfidf_vector.toarray()[0]

        # Ordenar por score descendente
        top_indices = np.argsort(scores)[::-1][:n]
        top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

        return top_terms


# Ejemplo de uso
if __name__ == "__main__":
    comparator = TFIDFCosineComparator()

    # Ejemplo de comparación
    texts = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are animals"
    ]

    matrix = comparator.compare_multiple(texts)

    print("Matriz de similitud:")
    print(matrix)

    print("\nVocabulario:")
    print(comparator.get_feature_names())

    print("\nTérminos más importantes del primer texto:")
    print(comparator.get_top_terms(texts[0]))
