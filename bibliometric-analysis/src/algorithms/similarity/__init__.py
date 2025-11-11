"""
Similarity Algorithms Module

This module provides 6 different text similarity algorithms:

Classical Algorithms:
1. Levenshtein Similarity - Edit distance based
2. TF-IDF Cosine Similarity - Term frequency based
3. Jaccard Similarity - Set intersection based
4. Character N-gram Similarity - Fuzzy matching based

AI-Based Algorithms:
5. Word Embeddings (Word2Vec/GloVe) - Static word embeddings
6. SBERT Similarity - Sentence-BERT embeddings (contextualized)
7. BERT Transformer Similarity - BERT contextual embeddings

Note: Total of 7 algorithms (4 classical + 3 AI-based)

Usage (Class-based):
    from src.algorithms.similarity import LevenshteinSimilarity, WordEmbeddingSimilarity

    calc = LevenshteinSimilarity()
    similarity = calc.compute_similarity("text1", "text2")

Usage (Standalone functions):
    from src.algorithms.similarity import word_embedding_similarity, sbert_similarity

    # Quick similarity calculation
    sim = word_embedding_similarity("machine learning", "artificial intelligence")
    print(f"Semantic similarity: {sim:.3f}")
"""

# Import classes
from .levenshtein_similarity import LevenshteinSimilarity
from .tfidf_cosine_similarity import TfidfCosineSimilarity
from .jaccard_similarity import JaccardSimilarity
from .ngram_similarity import CharacterNgramSimilarity
from .word_embedding_similarity import WordEmbeddingSimilarity, word_embedding_similarity
from .sbert_similarity import SBERTSimilarity, sbert_similarity
from .transformer_similarity import TransformerSimilarity, bert_similarity
from .similarity_comparator import SimilarityComparator

__all__ = [
    # Classes
    'LevenshteinSimilarity',
    'TfidfCosineSimilarity',
    'JaccardSimilarity',
    'CharacterNgramSimilarity',
    'WordEmbeddingSimilarity',
    'SBERTSimilarity',
    'TransformerSimilarity',
    'SimilarityComparator',
    # Standalone functions
    'word_embedding_similarity',
    'sbert_similarity',
    'bert_similarity',
]

__version__ = '1.0.1'
