"""
Algoritmo S-BERT (Sentence-BERT) para similitud semántica de textos.

Matemática y Arquitectura:
===========================
S-BERT es una modificación de BERT optimizada para generar embeddings de
oraciones que sean semánticamente significativos.

1. Arquitectura Transformer:
   Basado en el modelo transformer con atención multi-cabeza:

   Attention(Q, K, V) = softmax((Q·K^T) / √d_k) · V

   donde:
   - Q, K, V = matrices de query, key, value
   - d_k = dimensión de las claves
   - Atención permite al modelo enfocarse en partes relevantes del texto

2. Pooling Strategies:
   Para obtener un embedding de oración fijo desde tokens variables:

   a) Mean pooling (usado por defecto):
      e_sentence = (Σᵢ eᵢ) / n
      Promedio de todos los embeddings de tokens

   b) CLS pooling:
      e_sentence = e_[CLS]
      Usar solo el embedding del token [CLS]

   c) Max pooling:
      e_sentence[j] = max_i(eᵢ[j])
      Máximo por dimensión

3. Similitud del Coseno:
   sim(u, v) = (u · v) / (||u|| × ||v||)
            = Σᵢ(uᵢ × vᵢ) / (√Σᵢuᵢ² × √Σᵢvᵢ²)

   Rango: [-1, 1], pero S-BERT típicamente genera valores en [0, 1]
   después de normalización.

Ventajas sobre BERT estándar:
    - 2000x más rápido para comparaciones por pares
    - Genera embeddings comparables directamente
    - Mantiene calidad semántica de BERT

Proceso de entrenamiento (Siamese Network):
    1. Dos oraciones s₁, s₂ pasan por BERT compartido
    2. Genera embeddings u, v
    3. Objetivo: |u - v| pequeño si similares, grande si diferentes
    4. Función de pérdida: softmax contrastiva o triplet loss

Complejidad:
    Encoding: O(n × d²) donde n = tokens, d = dimensión modelo
    Comparación: O(d) con embeddings precalculados
    Memoria: ~500MB para modelo base

Referencias:
    - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
      using Siamese BERT-Networks. EMNLP 2019.
"""

import numpy as np
import logging
import time
from typing import List
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)


class SBERTComparator:
    """
    Implementa comparación de similitud usando Sentence-BERT.

    S-BERT genera embeddings de oraciones que capturan significado semántico,
    permitiendo comparaciones rápidas y precisas.

    Características:
        - Captura semántica profunda
        - Rápido en inferencia (comparado con BERT)
        - Multilingüe (según modelo)
        - Embeddings de dimensión fija
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None,
                 device: str = None):
        """
        Inicializa el comparador S-BERT.

        Args:
            model_name: Nombre del modelo a usar
                - 'all-MiniLM-L6-v2': Rápido, inglés (default)
                - 'all-mpnet-base-v2': Mejor calidad, inglés
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingüe
            cache_dir: Directorio para caché de modelos
            device: 'cuda', 'cpu', o None (auto-detectar)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Detectar dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Inicializando SBERTComparator")
        logger.info(f"  Modelo: {model_name}")
        logger.info(f"  Dispositivo: {self.device}")
        logger.info(f"  Cache: {cache_dir}")

        # Crear directorio de caché si no existe
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Cargar modelo
        logger.info("Cargando modelo S-BERT...")
        start_time = time.perf_counter()

        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=self.device
        )

        elapsed = time.perf_counter() - start_time
        logger.info(f"✓ Modelo cargado en {elapsed:.3f}s")

        # Obtener información del modelo
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"  Dimensión de embeddings: {self.embedding_dim}")

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """
        Codifica textos en embeddings.

        Proceso interno:
            1. Tokenización: "Hello world" → [101, 7592, 2088, 102]
            2. BERT encoding: tokens → hidden states [batch, seq_len, 384]
            3. Mean pooling: hidden states → sentence embedding [batch, 384]
            4. Normalización: embedding → unit vector

        Args:
            texts: Lista de textos a codificar
            batch_size: Tamaño de lote para procesamiento
            show_progress: Mostrar barra de progreso

        Returns:
            Matriz de embeddings (n_texts, embedding_dim)
        """
        logger.debug(f"Codificando {len(texts)} textos (batch_size={batch_size})...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar a vectores unitarios
        )

        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud semántica entre dos textos.

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Similitud en [0, 1]
        """
        # Codificar textos
        embeddings = self.encode([text1, text2])

        # Calcular similitud del coseno
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        # Mapear de [-1, 1] a [0, 1]
        sim = (sim + 1) / 2

        return float(sim)

    def compare_multiple(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Compara múltiples textos usando S-BERT.

        Proceso optimizado:
            1. Codificar todos los textos en un solo pase (batching)
            2. Calcular matriz de similitud con operación vectorizada
            3. Normalizar resultados

        Ventaja: Encoding batch >> encoding individual
        Ejemplo: 100 textos
            - Individual: 100 llamadas al modelo
            - Batch: 4 llamadas (batch_size=32)

        Args:
            texts: Lista de textos a comparar
            batch_size: Tamaño de lote para encoding

        Returns:
            Matriz de similitud de forma (n, n)
        """
        start_time = time.perf_counter()
        n = len(texts)

        logger.info(f"Comparando {n} textos con S-BERT...")

        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([[1.0]])

        # Codificar todos los textos en batches
        embeddings = self.encode(texts, batch_size=batch_size)

        logger.debug(f"Embeddings shape: {embeddings.shape}")

        # Calcular matriz de similitud del coseno
        # Optimizado con operaciones matriciales
        similarity_matrix = cosine_similarity(embeddings)

        # Mapear de [-1, 1] a [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2

        # Asegurar diagonal = 1.0
        np.fill_diagonal(similarity_matrix, 1.0)

        # Clipar al rango [0, 1]
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        elapsed = time.perf_counter() - start_time
        logger.info(f"S-BERT completado en {elapsed:.3f}s")

        return similarity_matrix

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Obtiene embedding de un texto individual.

        Útil para análisis o almacenamiento de embeddings.

        Args:
            text: Texto a codificar

        Returns:
            Vector embedding (embedding_dim,)
        """
        return self.encode([text])[0]

    def find_most_similar(self, query: str, candidates: List[str],
                          top_k: int = 5) -> List[tuple]:
        """
        Encuentra los textos más similares a una consulta.

        Args:
            query: Texto de consulta
            candidates: Lista de textos candidatos
            top_k: Número de resultados a retornar

        Returns:
            Lista de tuplas (índice, similitud) ordenadas por similitud
        """
        # Codificar query y candidatos
        all_texts = [query] + candidates
        embeddings = self.encode(all_texts)

        # Calcular similitudes
        query_emb = embeddings[0:1]
        cand_embs = embeddings[1:]

        similarities = cosine_similarity(query_emb, cand_embs)[0]
        similarities = (similarities + 1) / 2  # Mapear a [0, 1]

        # Ordenar por similitud descendente
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    comparator = SBERTComparator()

    # Ejemplo 1: Similitud semántica
    text1 = "The cat sits on the mat"
    text2 = "A feline rests on the rug"
    print(f"Similitud semántica: {comparator.similarity(text1, text2):.3f}")
    print("(Note: alta similitud aunque palabras diferentes)")

    # Ejemplo 2: Búsqueda
    query = "machine learning algorithms"
    candidates = [
        "deep learning neural networks",
        "cooking recipes for pasta",
        "artificial intelligence methods",
        "car maintenance tips"
    ]

    results = comparator.find_most_similar(query, candidates, top_k=2)
    print("\nTextos más similares:")
    for idx, sim in results:
        print(f"  {candidates[idx]}: {sim:.3f}")
