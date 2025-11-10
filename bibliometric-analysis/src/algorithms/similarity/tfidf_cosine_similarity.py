"""
TF-IDF Cosine Similarity Algorithm

MATHEMATICAL EXPLANATION:
========================

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects
how important a word is to a document in a collection. Combined with cosine similarity,
it provides a powerful method for comparing text documents.

PART 1: TF-IDF CALCULATION
---------------------------

1. TERM FREQUENCY (TF):
   Measures how frequently a term occurs in a document.

   TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)

   Alternative formulation (log normalization):
   TF(t, d) = 1 + log(freq(t, d))   if freq(t, d) > 0
            = 0                     otherwise

2. INVERSE DOCUMENT FREQUENCY (IDF):
   Measures how important a term is across all documents.

   IDF(t, D) = log(N / |{d ∈ D : t ∈ d}|)

   where:
   - N = total number of documents
   - |{d ∈ D : t ∈ d}| = number of documents containing term t

3. TF-IDF WEIGHT:
   TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

PART 2: COSINE SIMILARITY
--------------------------

Given two document vectors A and B in TF-IDF space:

                      A · B
   cos(θ) = ─────────────────────
            ||A|| × ||B||

           Σᵢ (Aᵢ × Bᵢ)
         = ──────────────────────
           √(Σᵢ Aᵢ²) × √(Σᵢ Bᵢ²)

where:
   - A · B = dot product of vectors A and B
   - ||A|| = Euclidean norm (magnitude) of vector A
   - θ = angle between vectors A and B

Properties:
   - cos(θ) ∈ [-1, 1] for general vectors
   - cos(θ) ∈ [0, 1] for TF-IDF vectors (non-negative values)
   - cos(0°) = 1.0 → identical direction (very similar)
   - cos(90°) = 0.0 → orthogonal (no similarity)

EXAMPLE CALCULATION:
====================

Documents:
   D1: "the cat sat on the mat"
   D2: "the dog sat on the log"

Step 1: Calculate TF for each term
   D1 TF: {the: 2/6, cat: 1/6, sat: 1/6, on: 1/6, mat: 1/6, dog: 0, log: 0}
   D2 TF: {the: 2/6, cat: 0, sat: 1/6, on: 1/6, mat: 0, dog: 1/6, log: 1/6}

Step 2: Calculate IDF (N=2 documents)
   IDF: {the: log(2/2)=0, cat: log(2/1)=0.693, sat: log(2/2)=0,
         on: log(2/2)=0, mat: log(2/1)=0.693, dog: log(2/1)=0.693, log: log(2/1)=0.693}

Step 3: Calculate TF-IDF vectors
   D1: [0, 0.116, 0, 0, 0.116, 0, 0]
   D2: [0, 0, 0, 0, 0, 0.116, 0.116]

Step 4: Cosine similarity
   cos(D1, D2) = (D1 · D2) / (||D1|| × ||D2||)
               = 0 / (0.164 × 0.164)
               = 0.0

COMPLEXITY:
-----------
Time:  O(n × m) where n = number of documents, m = vocabulary size
Space: O(n × m) for TF-IDF matrix
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from typing import List, Tuple, Optional
from loguru import logger
import time


class TfidfCosineSimilarity:
    """
    TF-IDF Cosine Similarity calculator

    Uses scikit-learn's TfidfVectorizer for efficient TF-IDF computation
    and computes cosine similarity between document vectors.
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        debug: bool = False
    ):
        """
        Initialize TF-IDF Cosine Similarity calculator

        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to extract (min_n, max_n)
            min_df: Minimum document frequency (ignore terms appearing in fewer docs)
            max_df: Maximum document frequency (ignore terms appearing in more docs)
            debug: If True, prints intermediate calculation steps

        Example:
            >>> calc = TfidfCosineSimilarity(ngram_range=(1, 2))
            >>> # Will extract both unigrams and bigrams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.debug = debug

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            strip_accents='unicode',
            stop_words=None  # Can be set to 'english' if needed
        )

        self.fitted = False
        self.vocabulary_ = None

        logger.info(f"TF-IDF Cosine Similarity initialized (ngram_range={ngram_range})")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer on texts and transform to TF-IDF matrix

        Args:
            texts: List of text documents

        Returns:
            TF-IDF matrix (n_documents × n_features)

        Example:
            >>> calc = TfidfCosineSimilarity()
            >>> texts = ["hello world", "hello python"]
            >>> matrix = calc.fit_transform(texts)
            >>> print(matrix.shape)
            (2, n_features)
        """
        start_time = time.perf_counter()

        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")

        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        self.vocabulary_ = self.vectorizer.vocabulary_

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nTF-IDF Vectorization:")
            logger.debug(f"  Documents: {len(texts)}")
            logger.debug(f"  Vocabulary size: {len(self.vocabulary_)}")
            logger.debug(f"  Matrix shape: {tfidf_matrix.shape}")
            logger.debug(f"  Matrix density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")

            # Show top features
            feature_names = self.vectorizer.get_feature_names_out()
            logger.debug(f"\n  Sample features (first 10): {list(feature_names[:10])}")

        logger.info(f"TF-IDF matrix computed in {elapsed_time*1000:.2f}ms")

        return tfidf_matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted vectorizer

        Args:
            texts: List of text documents

        Returns:
            TF-IDF matrix

        Raises:
            ValueError: If vectorizer not fitted yet
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")

        start_time = time.perf_counter()
        tfidf_matrix = self.vectorizer.transform(texts)
        elapsed_time = time.perf_counter() - start_time

        logger.info(f"Transformed {len(texts)} documents in {elapsed_time*1000:.2f}ms")

        return tfidf_matrix.toarray()

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute TF-IDF cosine similarity between two texts

        Args:
            text1: First text document
            text2: Second text document

        Returns:
            Cosine similarity score in range [0.0, 1.0]
            - 1.0 = identical semantic content
            - 0.0 = no common terms

        Example:
            >>> calc = TfidfCosineSimilarity()
            >>> sim = calc.compute_similarity(
            ...     "machine learning algorithms",
            ...     "machine learning techniques"
            ... )
            >>> print(f"{sim:.3f}")
            0.667
        """
        start_time = time.perf_counter()

        # Create corpus with both texts
        texts = [text1, text2]

        # Compute TF-IDF vectors
        tfidf_matrix = self.fit_transform(texts)

        # Extract vectors
        vector1 = tfidf_matrix[0]
        vector2 = tfidf_matrix[1]

        # Compute cosine similarity manually for debugging
        if self.debug:
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)

            logger.debug(f"\nCosine Similarity Calculation:")
            logger.debug(f"  Vector 1 shape: {vector1.shape}")
            logger.debug(f"  Vector 2 shape: {vector2.shape}")
            logger.debug(f"  Dot product (A · B): {dot_product:.6f}")
            logger.debug(f"  Norm ||A||: {norm1:.6f}")
            logger.debug(f"  Norm ||B||: {norm2:.6f}")

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                logger.debug(f"  Formula: {dot_product:.6f} / ({norm1:.6f} × {norm2:.6f})")
                logger.debug(f"  Result: {similarity:.6f}")
            else:
                similarity = 0.0
                logger.debug(f"  Result: 0.0 (zero vector)")

        else:
            # Use sklearn's optimized implementation
            similarity = sklearn_cosine([vector1], [vector2])[0][0]

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"TF-IDF cosine similarity: {similarity:.4f} (computed in {elapsed_time*1000:.2f}ms)")

        return float(similarity)

    def compute_similarity_matrix(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for multiple texts

        Args:
            texts: List of text documents

        Returns:
            Similarity matrix (n × n) where element [i,j] is similarity between texts[i] and texts[j]

        Example:
            >>> calc = TfidfCosineSimilarity()
            >>> texts = ["doc1 content", "doc2 content", "doc3 content"]
            >>> matrix = calc.compute_similarity_matrix(texts)
            >>> print(matrix.shape)
            (3, 3)
        """
        start_time = time.perf_counter()

        logger.info(f"Computing similarity matrix for {len(texts)} documents")

        # Compute TF-IDF vectors
        tfidf_matrix = self.fit_transform(texts)

        # Compute pairwise cosine similarities
        similarity_matrix = sklearn_cosine(tfidf_matrix)

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nSimilarity Matrix:")
            logger.debug(f"  Shape: {similarity_matrix.shape}")
            logger.debug(f"  Diagonal (self-similarity): {np.diag(similarity_matrix)}")
            logger.debug(f"  Mean similarity: {similarity_matrix.mean():.4f}")
            logger.debug(f"  Min similarity: {similarity_matrix.min():.4f}")
            logger.debug(f"  Max similarity (off-diagonal): {np.max(similarity_matrix - np.eye(len(texts))):.4f}")

        logger.info(f"Similarity matrix computed in {elapsed_time*1000:.2f}ms")

        return similarity_matrix

    def get_top_features(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF weighted features from a text

        Args:
            text: Text document
            top_n: Number of top features to return

        Returns:
            List of (feature, tfidf_score) tuples, sorted by score descending

        Example:
            >>> calc = TfidfCosineSimilarity()
            >>> calc.fit_transform(["sample corpus text"])
            >>> features = calc.get_top_features("machine learning algorithms")
            >>> print(features[:3])
            [('learning', 0.58), ('machine', 0.58), ('algorithms', 0.58)]
        """
        if not self.fitted:
            # Fit on single document for analysis
            self.fit_transform([text])

        # Transform text
        tfidf_vector = self.transform([text])[0]

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Create (feature, score) pairs
        feature_scores = [(feature_names[i], tfidf_vector[i])
                         for i in range(len(feature_names))
                         if tfidf_vector[i] > 0]

        # Sort by score descending
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        return feature_scores[:top_n]

    def explain_similarity(
        self,
        text1: str,
        text2: str,
        top_n: int = 5
    ) -> dict:
        """
        Explain similarity by showing top contributing features

        Args:
            text1: First text
            text2: Second text
            top_n: Number of top features to show

        Returns:
            Dictionary with similarity score and top contributing features
        """
        # Compute TF-IDF vectors
        tfidf_matrix = self.fit_transform([text1, text2])
        vector1 = tfidf_matrix[0]
        vector2 = tfidf_matrix[1]

        # Compute similarity
        similarity = sklearn_cosine([vector1], [vector2])[0][0]

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Compute contribution of each feature
        contributions = []
        for i in range(len(feature_names)):
            if vector1[i] > 0 and vector2[i] > 0:
                contrib = vector1[i] * vector2[i]
                contributions.append((feature_names[i], contrib, vector1[i], vector2[i]))

        # Sort by contribution
        contributions.sort(key=lambda x: x[1], reverse=True)

        result = {
            'similarity': similarity,
            'top_contributing_features': [
                {
                    'term': term,
                    'contribution': contrib,
                    'tfidf_text1': tfidf1,
                    'tfidf_text2': tfidf2
                }
                for term, contrib, tfidf1, tfidf2 in contributions[:top_n]
            ],
            'total_features': len(feature_names),
            'common_features': len(contributions)
        }

        return result


# Example usage and demonstration
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/tfidf_cosine_similarity.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("=" * 70)
    print("TF-IDF COSINE SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Basic usage
    print("\n1. BASIC EXAMPLE")
    print("-" * 70)

    calc = TfidfCosineSimilarity(debug=True)

    text1 = "machine learning algorithms for classification"
    text2 = "machine learning techniques for prediction"

    similarity = calc.compute_similarity(text1, text2)
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity: {similarity:.4f}")

    # Example 2: Identical texts
    print("\n2. IDENTICAL TEXTS")
    print("-" * 70)

    text = "artificial intelligence"
    similarity = calc.compute_similarity(text, text)
    print(f"Text: '{text}'")
    print(f"Self-similarity: {similarity:.4f}")

    # Example 3: Completely different texts
    print("\n3. COMPLETELY DIFFERENT TEXTS")
    print("-" * 70)

    text1 = "python programming language"
    text2 = "cooking recipes dinner"

    similarity = calc.compute_similarity(text1, text2)
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity: {similarity:.4f}")

    # Example 4: Similarity matrix
    print("\n4. SIMILARITY MATRIX")
    print("-" * 70)

    calc_matrix = TfidfCosineSimilarity(debug=False)
    texts = [
        "machine learning and artificial intelligence",
        "deep learning neural networks",
        "natural language processing",
        "machine learning algorithms"
    ]

    matrix = calc_matrix.compute_similarity_matrix(texts)

    print("\nTexts:")
    for i, text in enumerate(texts):
        print(f"  {i}: '{text}'")

    print("\nSimilarity Matrix:")
    print(matrix.round(3))

    # Example 5: Feature explanation
    print("\n5. SIMILARITY EXPLANATION")
    print("-" * 70)

    calc_explain = TfidfCosineSimilarity()
    text1 = "generative artificial intelligence models"
    text2 = "generative AI language models"

    explanation = calc_explain.explain_similarity(text1, text2, top_n=5)

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"\nSimilarity: {explanation['similarity']:.4f}")
    print(f"Common features: {explanation['common_features']} / {explanation['total_features']}")
    print("\nTop contributing features:")
    for feat in explanation['top_contributing_features']:
        print(f"  - {feat['term']:20s}: contribution={feat['contribution']:.4f} "
              f"(TF-IDF: {feat['tfidf_text1']:.3f}, {feat['tfidf_text2']:.3f})")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
