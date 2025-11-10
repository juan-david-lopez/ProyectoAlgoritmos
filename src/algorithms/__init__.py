"""
Módulo de algoritmos de similitud.

Contiene implementaciones de 6 algoritmos diferentes para comparar similitud entre textos.
"""

from .levenshtein import LevenshteinComparator
from .tfidf_cosine import TFIDFCosineComparator
from .jaccard import JaccardComparator
from .ngram import NGramComparator

# Los modelos de IA son opcionales (requieren dependencias adicionales)
try:
    from .sbert import SBERTComparator
    from .bert import BERTComparator
    _AI_MODELS_AVAILABLE = True
except ImportError:
    _AI_MODELS_AVAILABLE = False
    SBERTComparator = None
    BERTComparator = None

from .similarity_comparator import SimilarityComparator

__all__ = [
    "LevenshteinComparator",
    "TFIDFCosineComparator",
    "JaccardComparator",
    "NGramComparator",
    "SBERTComparator",
    "BERTComparator",
    "SimilarityComparator",
]


def get_available_algorithms():
    """
    Retorna lista de algoritmos disponibles en el sistema actual.

    Returns:
        dict: Diccionario con disponibilidad de cada algoritmo
    """
    return {
        'levenshtein': True,
        'tfidf_cosine': True,
        'jaccard': True,
        'ngram': True,
        'sbert': _AI_MODELS_AVAILABLE,
        'bert': _AI_MODELS_AVAILABLE,
    }
