"""
Módulo de análisis de términos.

Contiene herramientas para análisis de frecuencia de términos predefinidos,
extracción automática de términos clave, y evaluación de precisión.
"""

from .predefined_terms_analyzer import PredefinedTermsAnalyzer
from .automatic_term_extractor import AutomaticTermExtractor
from .term_evaluator import TermEvaluator
from .term_precision_evaluator import TermPrecisionEvaluator

__all__ = [
    'PredefinedTermsAnalyzer',
    'AutomaticTermExtractor',
    'TermEvaluator',
    'TermPrecisionEvaluator',
]
