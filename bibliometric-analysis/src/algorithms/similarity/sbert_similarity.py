"""
Sentence-BERT (SBERT) Similarity Algorithm

MATHEMATICAL EXPLANATION:
========================

Sentence-BERT is a modification of BERT that uses siamese and triplet network structures
to derive semantically meaningful sentence embeddings that can be compared using cosine
similarity. Unlike traditional BERT, SBERT is specifically optimized for semantic similarity tasks.

ARCHITECTURE OVERVIEW:
----------------------

Traditional BERT comparison requires:
    - Concatenate sentences A and B
    - Feed through BERT → O(n²) complexity for n sentences
    - Not suitable for similarity search

SBERT improvements:
    - Encode sentences independently
    - Generate fixed-size embeddings
    - Compare using cosine similarity → O(n) complexity
    - Enables efficient semantic search

EMBEDDING GENERATION:
---------------------

For a sentence S with tokens [t₁, t₂, ..., tₙ]:

1. Token Embeddings:
   BERT(S) → [h₁, h₂, ..., hₙ]
   where hᵢ ∈ ℝᵈ (typically d=768 or d=384)

2. Pooling Strategy (MEAN pooling - most common):

           1   n
   e(S) = ─── Σ hᵢ
           n  i=1

   Alternative strategies:
   - MAX pooling: e(S) = max(h₁, h₂, ..., hₙ) element-wise
   - CLS pooling: e(S) = h₁ (use [CLS] token embedding)

3. Normalization (L2 norm):

           e(S)
   ê(S) = ──────
          ||e(S)||₂

   where ||e(S)||₂ = √(Σᵢ eᵢ²)

COSINE SIMILARITY:
------------------

For normalized embeddings ê(A) and ê(B):

                    ê(A) · ê(B)
   cos_sim(A, B) = ─────────────
                   ||ê(A)|| × ||ê(B)||

Since embeddings are already normalized:

   cos_sim(A, B) = ê(A) · ê(B) = Σᵢ (êₐᵢ × êᵦᵢ)

Properties:
   - cos_sim ∈ [-1, 1]
   - cos_sim = 1  → identical semantic meaning
   - cos_sim = 0  → orthogonal (unrelated)
   - cos_sim = -1 → opposite meanings

SBERT MODELS:
-------------

Popular pre-trained models:

1. **all-MiniLM-L6-v2** (default choice):
   - Embedding dimension: 384
   - Fast inference (~5x faster than base BERT)
   - Good performance on semantic similarity
   - Model size: ~80 MB

2. **all-mpnet-base-v2**:
   - Embedding dimension: 768
   - Best overall performance
   - Slower but more accurate

3. **multi-qa-MiniLM-L6-cos-v1**:
   - Optimized for question-answering
   - Good for asymmetric similarity

TRAINING OBJECTIVE:
-------------------

SBERT is trained using contrastive learning:

1. Siamese Network:
   - Positive pairs: similar sentences
   - Negative pairs: dissimilar sentences

2. Loss Function (Cosine Similarity Loss):

   L = max(0, ε - cos_sim(A, B))  for negative pairs
   L = max(0, cos_sim(A, B) - ε)  for positive pairs

   where ε is a margin parameter

EXAMPLE CALCULATION:
====================

Sentences:
   S1: "The cat sits on the mat"
   S2: "A feline rests on a rug"
   S3: "Python is a programming language"

Step 1: Encode sentences using SBERT
   e(S1) = [0.23, -0.15, 0.08, ..., 0.42]  (384 dimensions)
   e(S2) = [0.21, -0.12, 0.11, ..., 0.39]
   e(S3) = [-0.31, 0.52, -0.18, ..., -0.07]

Step 2: Normalize embeddings (L2 norm)
   ||e(S1)|| = 1.0 (already normalized by model)

Step 3: Compute cosine similarities
   cos_sim(S1, S2) = e(S1) · e(S2) = 0.87  (high - semantically similar)
   cos_sim(S1, S3) = e(S1) · e(S3) = 0.12  (low - different topics)
   cos_sim(S2, S3) = e(S2) · e(S3) = 0.09  (low - different topics)

ADVANTAGES:
-----------
1. Captures semantic meaning (not just lexical overlap)
2. Handles paraphrases effectively
3. Language understanding through pre-training
4. Fast inference (single pass per sentence)
5. Fixed-size embeddings regardless of sentence length

COMPLEXITY:
-----------
Encoding:  O(n × L) where n = batch size, L = sequence length
Similarity: O(d) where d = embedding dimension
Space:     O(d) per sentence embedding
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger
import time
import torch

# Global model cache to avoid reloading
_SBERT_MODEL_CACHE = {}


class SBERTSimilarity:
    """
    Sentence-BERT Similarity calculator

    Uses pre-trained Sentence-BERT models to generate semantic embeddings
    and compute cosine similarity between texts.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize SBERT Similarity calculator

        Args:
            model_name: Name of pre-trained SBERT model
                       Options: 'all-MiniLM-L6-v2' (default, fast),
                               'all-mpnet-base-v2' (accurate),
                               'multi-qa-MiniLM-L6-cos-v1' (QA)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            cache_folder: Directory to cache downloaded models
            debug: If True, prints intermediate calculation steps

        Example:
            >>> # Default model (MiniLM)
            >>> calc = SBERTSimilarity()
            >>>
            >>> # High-accuracy model
            >>> calc = SBERTSimilarity(model_name='all-mpnet-base-v2')
        """
        self.model_name = model_name
        self.debug = debug

        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Set cache folder
        if cache_folder:
            cache_path = Path(cache_folder)
            cache_path.mkdir(parents=True, exist_ok=True)
        else:
            cache_folder = None

        logger.info(f"Initializing SBERT model: {model_name}")
        logger.info(f"Device: {device}")

        start_time = time.perf_counter()

        # Load model
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder
            )
            load_time = time.perf_counter() - start_time

            # Get model info
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.success(f"Model loaded in {load_time:.2f}s")
            logger.info(f"Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            raise

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings to unit length

        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)

        Example:
            >>> calc = SBERTSimilarity()
            >>> texts = ["Hello world", "Goodbye world"]
            >>> embeddings = calc.encode(texts)
            >>> print(embeddings.shape)
            (2, 384)
        """
        start_time = time.perf_counter()

        logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})")

        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nEncoding Results:")
            logger.debug(f"  Input texts: {len(texts)}")
            logger.debug(f"  Embedding shape: {embeddings.shape}")
            logger.debug(f"  Embedding dtype: {embeddings.dtype}")
            logger.debug(f"  Encoding time: {elapsed_time:.3f}s")
            logger.debug(f"  Time per text: {elapsed_time/len(texts)*1000:.2f}ms")

            # Show sample embedding statistics
            logger.debug(f"\n  Embedding statistics:")
            logger.debug(f"    Mean: {embeddings.mean():.6f}")
            logger.debug(f"    Std:  {embeddings.std():.6f}")
            logger.debug(f"    Min:  {embeddings.min():.6f}")
            logger.debug(f"    Max:  {embeddings.max():.6f}")

        logger.info(f"Encoded {len(texts)} texts in {elapsed_time:.2f}s "
                   f"({len(texts)/elapsed_time:.1f} texts/sec)")

        return embeddings

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute SBERT cosine similarity between two texts

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Cosine similarity in range [-1.0, 1.0]
            (typically [0.0, 1.0] for most texts)
            - 1.0 = semantically identical
            - 0.0 = unrelated
            - -1.0 = opposite meanings (rare)

        Example:
            >>> calc = SBERTSimilarity()
            >>> sim = calc.compute_similarity(
            ...     "The cat sits on the mat",
            ...     "A feline rests on a rug"
            ... )
            >>> print(f"{sim:.3f}")
            0.756
        """
        start_time = time.perf_counter()

        # Encode both texts
        embeddings = self.encode([text1, text2], show_progress=False)

        # Extract individual embeddings
        emb1 = embeddings[0]
        emb2 = embeddings[1]

        # Compute cosine similarity (embeddings are already normalized)
        similarity = float(np.dot(emb1, emb2))

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nSBERT Similarity Calculation:")
            logger.debug(f"  Text 1: '{text1}'")
            logger.debug(f"  Text 2: '{text2}'")
            logger.debug(f"\n  Embedding 1 shape: {emb1.shape}")
            logger.debug(f"  Embedding 2 shape: {emb2.shape}")
            logger.debug(f"\n  Dot product: {similarity:.6f}")
            logger.debug(f"  (Embeddings pre-normalized by model)")

        logger.info(f"SBERT similarity: {similarity:.4f} (computed in {elapsed_time*1000:.2f}ms)")

        return similarity

    def compute_similarity_matrix(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for multiple texts

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Similarity matrix of shape (n_texts, n_texts)
            Element [i, j] is similarity between texts[i] and texts[j]

        Example:
            >>> calc = SBERTSimilarity()
            >>> texts = ["doc1", "doc2", "doc3"]
            >>> matrix = calc.compute_similarity_matrix(texts)
            >>> print(matrix.shape)
            (3, 3)
        """
        start_time = time.perf_counter()

        logger.info(f"Computing similarity matrix for {len(texts)} texts")

        # Encode all texts
        embeddings = self.encode(texts, batch_size=batch_size, show_progress=True)

        # Compute pairwise cosine similarities
        # Since embeddings are normalized, cosine = dot product
        similarity_matrix = np.dot(embeddings, embeddings.T)

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nSimilarity Matrix:")
            logger.debug(f"  Shape: {similarity_matrix.shape}")
            logger.debug(f"  Diagonal (self-similarity): {np.diag(similarity_matrix)}")
            logger.debug(f"  Mean similarity: {similarity_matrix.mean():.4f}")
            logger.debug(f"  Min similarity: {similarity_matrix.min():.4f}")
            logger.debug(f"  Max similarity (off-diagonal): "
                        f"{np.max(similarity_matrix - np.eye(len(texts))):.4f}")

        logger.info(f"Similarity matrix computed in {elapsed_time:.2f}s")

        return similarity_matrix

    def find_most_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find most similar texts in corpus to query

        Args:
            query: Query text
            corpus: List of corpus texts to search
            top_k: Number of top results to return

        Returns:
            List of (index, text, similarity) tuples, sorted by similarity descending

        Example:
            >>> calc = SBERTSimilarity()
            >>> query = "machine learning"
            >>> corpus = ["deep learning", "cooking recipes", "neural networks"]
            >>> results = calc.find_most_similar(query, corpus, top_k=2)
            >>> for idx, text, sim in results:
            ...     print(f"{text}: {sim:.3f}")
            deep learning: 0.756
            neural networks: 0.712
        """
        logger.info(f"Finding top {top_k} similar texts to query in corpus of {len(corpus)}")

        start_time = time.perf_counter()

        # Encode query and corpus
        query_embedding = self.encode([query], show_progress=False)[0]
        corpus_embeddings = self.encode(corpus, show_progress=True)

        # Compute similarities
        similarities = np.dot(corpus_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Create results
        results = [
            (int(idx), corpus[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        elapsed_time = time.perf_counter() - start_time

        logger.info(f"Found {len(results)} results in {elapsed_time:.2f}s")

        if self.debug:
            logger.debug(f"\nTop {top_k} Results:")
            for rank, (idx, text, sim) in enumerate(results, 1):
                logger.debug(f"  {rank}. [{idx}] {text[:50]}... (similarity: {sim:.4f})")

        return results

    def semantic_search(
        self,
        queries: List[str],
        corpus: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[int, str, float]]]:
        """
        Perform semantic search for multiple queries

        Args:
            queries: List of query texts
            corpus: List of corpus texts to search
            top_k: Number of top results per query

        Returns:
            List of result lists (one per query)
            Each result list contains (index, text, similarity) tuples

        Example:
            >>> calc = SBERTSimilarity()
            >>> queries = ["AI research", "cooking"]
            >>> corpus = ["machine learning", "deep learning", "pasta recipe"]
            >>> all_results = calc.semantic_search(queries, corpus, top_k=2)
            >>> for q_idx, results in enumerate(all_results):
            ...     print(f"Query {q_idx}: {len(results)} results")
        """
        logger.info(f"Semantic search: {len(queries)} queries, {len(corpus)} corpus texts")

        start_time = time.perf_counter()

        # Encode queries and corpus
        query_embeddings = self.encode(queries, show_progress=False)
        corpus_embeddings = self.encode(corpus, show_progress=True)

        # Compute similarity matrix (queries × corpus)
        similarity_matrix = np.dot(query_embeddings, corpus_embeddings.T)

        # Get top-k for each query
        all_results = []
        for q_idx in range(len(queries)):
            similarities = similarity_matrix[q_idx]

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Create results for this query
            results = [
                (int(idx), corpus[idx], float(similarities[idx]))
                for idx in top_indices
            ]

            all_results.append(results)

        elapsed_time = time.perf_counter() - start_time

        logger.info(f"Semantic search completed in {elapsed_time:.2f}s")

        return all_results


# ============================================================================
# STANDALONE FUNCTION FOR SIMPLE USAGE
# ============================================================================

def sbert_similarity(
    text1: str,
    text2: str,
    model_name: str = 'all-MiniLM-L6-v2',
    use_cache: bool = True
) -> float:
    """
    Similitud usando Sentence-BERT embeddings.

    Función standalone simplificada para uso rápido sin instanciar clases.

    Explicación técnica:
    --------------------

    Sentence-BERT genera embeddings densos (vectores) de 384 dimensiones que
    capturan el significado semántico de las oraciones completas.

    **Ventajas sobre métodos clásicos:**
    - ✅ Captura semántica, no solo léxica
    - ✅ "car" y "automobile" tendrán alta similitud
    - ✅ Robusto a paráfrasis
    - ✅ Entiende sinónimos y conceptos relacionados

    **Arquitectura SBERT:**
    1. Input sentence → BERT encoder
    2. Token embeddings [h₁, h₂, ..., hₙ]
    3. Mean pooling: e(S) = (1/n) Σᵢ hᵢ
    4. L2 normalization: ê(S) = e(S) / ||e(S)||
    5. Output: 384-dimensional embedding

    **Similitud coseno:**
    cos(θ) = ê(A) · ê(B)  (dot product, since normalized)

    Pasos del algoritmo:
    --------------------
    1. Cargar modelo pre-entrenado (con caché)
    2. Generar embeddings para ambos textos
    3. Calcular cosine similarity entre embeddings
    4. Retornar score [0, 1]

    Args:
        text1: Primer texto a comparar
        text2: Segundo texto a comparar
        model_name: Nombre del modelo SBERT pre-entrenado
                   'all-MiniLM-L6-v2' (default): Rápido, 384 dim, bueno para similitud
                   'all-mpnet-base-v2': Más preciso, 768 dim, más lento
        use_cache: Si True, cachea el modelo para llamadas futuras

    Returns:
        Similitud semántica en rango [0.0, 1.0]
        - 1.0 = semánticamente idénticos
        - 0.0 = sin relación semántica

    Example:
        >>> # Paráfrasis - alta similitud
        >>> sim = sbert_similarity(
        ...     "The cat sits on the mat",
        ...     "A feline rests on a rug"
        ... )
        >>> print(f"Paraphrase similarity: {sim:.3f}")
        Paraphrase similarity: 0.756

        >>> # Conceptos relacionados
        >>> sim = sbert_similarity(
        ...     "machine learning algorithms",
        ...     "artificial intelligence techniques"
        ... )
        >>> print(f"Related concepts: {sim:.3f}")
        Related concepts: 0.682

        >>> # Tópicos diferentes - baja similitud
        >>> sim = sbert_similarity(
        ...     "machine learning",
        ...     "cooking recipes"
        ... )
        >>> print(f"Different topics: {sim:.3f}")
        Different topics: 0.103

    Modelo usado:
    -------------
    all-MiniLM-L6-v2:
    - Tamaño: ~80 MB
    - Dimensión embeddings: 384
    - Velocidad: ~5x más rápido que BERT-base
    - Performance: Excelente para similitud semántica
    - Entrenamiento: Fine-tuned en 1B+ pares de oraciones

    Note:
        - Primera llamada descarga el modelo (~80 MB)
        - Llamadas subsecuentes usan modelo cacheado (rápido)
        - Para GPU: instalar torch con CUDA support
        - Requiere: pip install sentence-transformers
    """
    global _SBERT_MODEL_CACHE

    start_time = time.perf_counter()

    # Load model from cache or download
    if use_cache and model_name in _SBERT_MODEL_CACHE:
        model = _SBERT_MODEL_CACHE[model_name]
        logger.debug(f"Using cached SBERT model: {model_name}")
    else:
        logger.info(f"Loading SBERT model: {model_name}")
        model = SentenceTransformer(model_name)

        if use_cache:
            _SBERT_MODEL_CACHE[model_name] = model
            logger.debug(f"Cached model: {model_name}")

    # Generate embeddings for both texts
    # normalize_embeddings=True ensures L2 normalization
    embeddings = model.encode(
        [text1, text2],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Extract individual embeddings
    emb1 = embeddings[0]  # Shape: (384,)
    emb2 = embeddings[1]  # Shape: (384,)

    # Compute cosine similarity
    # Since embeddings are normalized, cosine = dot product
    similarity = float(np.dot(emb1, emb2))

    elapsed_time = time.perf_counter() - start_time

    logger.debug(f"SBERT similarity computed in {elapsed_time*1000:.2f}ms: {similarity:.4f}")

    return similarity


# Example usage and demonstration
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/sbert_similarity.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("=" * 70)
    print("SENTENCE-BERT (SBERT) SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Basic semantic similarity
    print("\n1. SEMANTIC SIMILARITY (Paraphrases)")
    print("-" * 70)

    calc = SBERTSimilarity(debug=True)

    text1 = "The cat sits on the mat"
    text2 = "A feline rests on a rug"
    text3 = "Python is a programming language"

    sim_12 = calc.compute_similarity(text1, text2)
    sim_13 = calc.compute_similarity(text1, text3)

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity (1, 2): {sim_12:.4f} (paraphrases - high)")
    print(f"Similarity (1, 3): {sim_13:.4f} (different topics - low)")

    # Example 2: Lexical vs Semantic
    print("\n2. LEXICAL OVERLAP vs SEMANTIC MEANING")
    print("-" * 70)

    calc_compare = SBERTSimilarity(debug=False)

    # High lexical overlap, different meanings
    text1 = "The bank is near the river"
    text2 = "I need to bank this check"

    # Low lexical overlap, similar meanings
    text3 = "The financial institution is by the water"

    sim_12 = calc_compare.compute_similarity(text1, text2)
    sim_13 = calc_compare.compute_similarity(text1, text3)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}' (same word 'bank', different meaning)")
    print(f"Text 3: '{text3}' (different words, same meaning)")
    print(f"\nSimilarity (1, 2): {sim_12:.4f}")
    print(f"Similarity (1, 3): {sim_13:.4f}")
    print("\nNote: SBERT captures semantic meaning, not just word overlap")

    # Example 3: Similarity matrix
    print("\n3. SIMILARITY MATRIX")
    print("-" * 70)

    calc_matrix = SBERTSimilarity(debug=False)

    texts = [
        "machine learning algorithms",
        "deep neural networks",
        "cooking pasta recipes",
        "artificial intelligence systems"
    ]

    matrix = calc_matrix.compute_similarity_matrix(texts)

    print("\nTexts:")
    for i, text in enumerate(texts):
        print(f"  {i}: '{text}'")

    print("\nSimilarity Matrix:")
    print(matrix.round(3))

    # Example 4: Semantic search
    print("\n4. SEMANTIC SEARCH")
    print("-" * 70)

    calc_search = SBERTSimilarity(debug=False)

    query = "artificial intelligence and machine learning"

    corpus = [
        "Deep learning is a subset of machine learning",
        "Python is a popular programming language",
        "Neural networks are inspired by the human brain",
        "Cooking Italian food requires fresh ingredients",
        "Generative AI models can create new content",
        "The weather is nice today"
    ]

    results = calc_search.find_most_similar(query, corpus, top_k=3)

    print(f"\nQuery: '{query}'")
    print(f"Corpus: {len(corpus)} texts")
    print("\nTop 3 Most Similar:")
    for rank, (idx, text, sim) in enumerate(results, 1):
        print(f"  {rank}. [{idx}] '{text}'")
        print(f"      Similarity: {sim:.4f}")

    # Example 5: Multi-query semantic search
    print("\n5. BATCH SEMANTIC SEARCH")
    print("-" * 70)

    queries = [
        "machine learning research",
        "cooking and recipes",
        "weather conditions"
    ]

    all_results = calc_search.semantic_search(queries, corpus, top_k=2)

    for q_idx, (query, results) in enumerate(zip(queries, all_results)):
        print(f"\nQuery {q_idx + 1}: '{query}'")
        print(f"  Top 2 results:")
        for rank, (idx, text, sim) in enumerate(results, 1):
            print(f"    {rank}. '{text[:40]}...' (sim: {sim:.3f})")

    # Example 6: Embedding visualization
    print("\n6. EMBEDDING ANALYSIS")
    print("-" * 70)

    texts_to_analyze = [
        "generative artificial intelligence",
        "deep learning models"
    ]

    embeddings = calc.encode(texts_to_analyze)

    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Number of texts: {embeddings.shape[0]}")

    for i, text in enumerate(texts_to_analyze):
        emb = embeddings[i]
        print(f"\nText: '{text}'")
        print(f"  Embedding stats:")
        print(f"    Mean: {emb.mean():.6f}")
        print(f"    Std:  {emb.std():.6f}")
        print(f"    L2 norm: {np.linalg.norm(emb):.6f}")
        print(f"    First 5 values: {emb[:5].round(4)}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
