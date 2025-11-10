"""
Tests para Term Analysis - Módulo de Visualizaciones y Análisis

Tests completos para TermNormalizer, DomainStopwords y TermVisualizer.
Incluye corpus de ejemplo y validación de todas las funcionalidades.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no-GUI para tests
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

# Añadir src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.term_viz import (
    TermNormalizer,
    DomainStopwords,
    TermVisualizer
)


# ============================================================================
# FIXTURES Y DATOS DE PRUEBA
# ============================================================================

@pytest.fixture
def sample_corpus():
    """Corpus de ejemplo con abstracts académicos."""
    return [
        "Deep learning and neural networks revolutionize computer vision applications.",
        "Machine learning models demonstrate impressive results in image classification.",
        "Natural language processing uses recurrent neural networks for text analysis.",
        "Convolutional neural networks excel at computer vision tasks.",
        "Deep learning techniques advance artificial intelligence research."
    ]


@pytest.fixture
def sample_terms():
    """Términos de ejemplo para testing."""
    return [
        "deep learning",
        "neural networks",
        "machine learning",
        "computer vision",
        "natural language processing",
        "convolutional networks",
        "image classification"
    ]


@pytest.fixture
def sample_terms_with_plurals():
    """Términos con variaciones de plural/singular."""
    return [
        "neural network",
        "neural networks",
        "deep learning model",
        "deep learning models",
        "classification task",
        "classification tasks"
    ]


@pytest.fixture
def sample_frequencies():
    """Frecuencias de términos de ejemplo."""
    return {
        'deep learning': 45,
        'neural networks': 38,
        'machine learning': 52,
        'computer vision': 28,
        'natural language processing': 25,
        'convolutional networks': 20,
        'paper': 15,  # Stopword
        'result': 10   # Stopword
    }


@pytest.fixture
def sample_cooccurrences():
    """Co-ocurrencias de términos de ejemplo."""
    return {
        ('deep learning', 'neural networks'): 15,
        ('deep learning', 'machine learning'): 20,
        ('neural networks', 'computer vision'): 12,
        ('machine learning', 'computer vision'): 10,
        ('natural language processing', 'neural networks'): 8,
        ('deep learning', 'convolutional networks'): 6
    }


# ============================================================================
# TESTS: TermNormalizer
# ============================================================================

class TestTermNormalizer:
    """Tests para el normalizador de términos."""

    def test_initialization(self):
        """Test: Inicialización correcta del normalizador."""
        normalizer = TermNormalizer()
        assert normalizer.nlp is not None
        assert 'en_core_web_sm' in normalizer.nlp.meta['name'] or 'en_core' in normalizer.nlp.meta['name']

    def test_normalize_single_term(self):
        """Test: Normalización de un término individual."""
        normalizer = TermNormalizer()

        # Singular -> Lema
        normalized = normalizer.normalize_term("neural networks")
        assert "network" in normalized.lower()

        # Lowercase
        normalized = normalizer.normalize_term("Deep Learning")
        assert normalized == normalized.lower()

    def test_normalize_plural_to_singular(self, sample_terms_with_plurals):
        """Test: Conversión de plurales a singular."""
        normalizer = TermNormalizer()

        mapping = normalizer.normalize_with_mapping(sample_terms_with_plurals)

        # Verificar que plurales se normalizan a singular
        assert mapping['neural networks'] == mapping['neural network']
        assert mapping['classification tasks'] == mapping['classification task']

    def test_normalize_multiple_terms(self, sample_terms):
        """Test: Normalización de múltiples términos."""
        normalizer = TermNormalizer()

        normalized = normalizer.normalize_terms(sample_terms, show_progress=False)

        assert len(normalized) == len(sample_terms)
        assert all(isinstance(t, str) for t in normalized)
        assert all(t == t.lower() for t in normalized)

    def test_normalize_with_mapping(self, sample_terms):
        """Test: Normalización con mapeo."""
        normalizer = TermNormalizer()

        mapping = normalizer.normalize_with_mapping(sample_terms)

        assert len(mapping) == len(sample_terms)
        assert all(original in mapping for original in sample_terms)
        assert all(isinstance(v, str) for v in mapping.values())

    def test_normalize_empty_string(self):
        """Test: Normalización de string vacío."""
        normalizer = TermNormalizer()

        normalized = normalizer.normalize_term("")
        assert normalized == ""

    def test_normalize_special_characters(self):
        """Test: Manejo de caracteres especiales."""
        normalizer = TermNormalizer()

        term = "machine-learning (ML)"
        normalized = normalizer.normalize_term(term)

        # Debe contener los tokens principales
        assert "machine" in normalized or "learning" in normalized


# ============================================================================
# TESTS: DomainStopwords
# ============================================================================

class TestDomainStopwords:
    """Tests para gestor de stopwords del dominio."""

    def test_initialization(self):
        """Test: Inicialización de stopwords."""
        stopwords = DomainStopwords()

        assert len(stopwords.stopwords) > 0
        assert 'paper' in stopwords.stopwords
        assert 'research' in stopwords.stopwords

    def test_initialization_with_additional(self):
        """Test: Inicialización con stopwords adicionales."""
        additional = {'custom', 'test', 'example'}
        stopwords = DomainStopwords(additional_stopwords=additional)

        assert 'custom' in stopwords.stopwords
        assert 'test' in stopwords.stopwords
        assert 'paper' in stopwords.stopwords  # También las predefinidas

    def test_is_stopword(self):
        """Test: Verificación de stopwords."""
        stopwords = DomainStopwords()

        # Stopwords académicas
        assert stopwords.is_stopword('paper')
        assert stopwords.is_stopword('research')
        assert stopwords.is_stopword('result')

        # Términos técnicos NO son stopwords
        assert not stopwords.is_stopword('neural networks')
        assert not stopwords.is_stopword('deep learning')

    def test_is_stopword_case_insensitive(self):
        """Test: Verificación case-insensitive."""
        stopwords = DomainStopwords()

        assert stopwords.is_stopword('PAPER')
        assert stopwords.is_stopword('Paper')
        assert stopwords.is_stopword('paper')

    def test_filter_terms(self, sample_terms):
        """Test: Filtrado de términos."""
        stopwords = DomainStopwords()

        # Añadir algunos términos que no son stopwords
        terms_with_stopwords = sample_terms + ['paper', 'research', 'result']

        filtered = stopwords.filter_terms(terms_with_stopwords)

        assert len(filtered) <= len(terms_with_stopwords)
        assert 'paper' not in filtered
        assert 'deep learning' in filtered

    def test_add_stopwords(self):
        """Test: Añadir stopwords dinámicamente."""
        stopwords = DomainStopwords()

        initial_count = len(stopwords.stopwords)

        stopwords.add_stopwords({'new_stopword', 'another_one'})

        assert len(stopwords.stopwords) == initial_count + 2
        assert 'new_stopword' in stopwords.stopwords


# ============================================================================
# TESTS: TermVisualizer
# ============================================================================

class TestTermVisualizer:
    """Tests para generador de visualizaciones."""

    def test_initialization(self):
        """Test: Inicialización del visualizador."""
        visualizer = TermVisualizer()

        assert visualizer.normalizer is not None
        assert visualizer.stopwords is not None

    def test_initialization_with_custom_components(self):
        """Test: Inicialización con componentes personalizados."""
        normalizer = TermNormalizer()
        stopwords = DomainStopwords()

        visualizer = TermVisualizer(normalizer=normalizer, stopwords=stopwords)

        assert visualizer.normalizer is normalizer
        assert visualizer.stopwords is stopwords

    def test_plot_term_frequencies(self, sample_frequencies):
        """Test: Gráfico de frecuencias."""
        visualizer = TermVisualizer()

        fig = visualizer.plot_term_frequencies(
            sample_frequencies,
            top_n=5,
            normalize=False
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_frequencies_with_normalization(self, sample_frequencies):
        """Test: Gráfico con normalización."""
        visualizer = TermVisualizer()

        fig = visualizer.plot_term_frequencies(
            sample_frequencies,
            top_n=5,
            normalize=True
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_frequencies_filters_stopwords(self, sample_frequencies):
        """Test: El gráfico filtra stopwords."""
        visualizer = TermVisualizer()

        # sample_frequencies incluye 'paper' y 'result' como stopwords
        fig = visualizer.plot_term_frequencies(
            sample_frequencies,
            top_n=20,
            normalize=False
        )

        # El gráfico debe existir incluso después de filtrar
        assert fig is not None
        plt.close(fig)

    def test_plot_cooccurrence_heatmap(self, sample_cooccurrences):
        """Test: Heatmap de co-ocurrencia."""
        visualizer = TermVisualizer()

        fig = visualizer.plot_cooccurrence_heatmap(
            sample_cooccurrences,
            top_n=5
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_venn_diagram_2_sets(self):
        """Test: Diagrama de Venn de 2 conjuntos."""
        visualizer = TermVisualizer()

        set1 = {'a', 'b', 'c'}
        set2 = {'b', 'c', 'd'}

        fig = visualizer.plot_venn_diagram(
            set1, set2,
            labels=('Set 1', 'Set 2')
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_venn_diagram_3_sets(self):
        """Test: Diagrama de Venn de 3 conjuntos."""
        visualizer = TermVisualizer()

        set1 = {'a', 'b', 'c'}
        set2 = {'b', 'c', 'd'}
        set3 = {'c', 'd', 'e'}

        fig = visualizer.plot_venn_diagram(
            set1, set2, set3,
            labels=('Set 1', 'Set 2', 'Set 3')
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_wordcloud(self, sample_frequencies):
        """Test: Word cloud."""
        visualizer = TermVisualizer()

        fig = visualizer.plot_wordcloud(
            sample_frequencies,
            max_words=50
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_similarity_matrix(self, sample_terms):
        """Test: Matriz de similitud."""
        visualizer = TermVisualizer()

        # Crear matriz de similitud simulada
        n = len(sample_terms)
        similarity_matrix = np.random.rand(n, n)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1.0)

        fig = visualizer.plot_similarity_matrix(
            sample_terms,
            similarity_matrix,
            max_terms=5
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_method_comparison(self):
        """Test: Comparación de métodos."""
        visualizer = TermVisualizer()

        methods_data = {
            'TF-IDF': {'precision': 0.75, 'recall': 0.68, 'f1_score': 0.71},
            'RAKE': {'precision': 0.65, 'recall': 0.72, 'f1_score': 0.68},
            'TextRank': {'precision': 0.70, 'recall': 0.75, 'f1_score': 0.72}
        }

        fig = visualizer.plot_method_comparison(methods_data)

        assert fig is not None
        plt.close(fig)

    def test_generate_all_visualizations(self, sample_frequencies, sample_cooccurrences, tmp_path):
        """Test: Generación de todas las visualizaciones."""
        visualizer = TermVisualizer()

        # Preparar datos
        data = {
            'frequencies': sample_frequencies,
            'cooccurrences': sample_cooccurrences,
            'sets': [
                {'deep learning', 'neural networks'},
                {'machine learning', 'neural networks'}
            ],
            'set_labels': ('AI', 'ML'),
            'term_scores': sample_frequencies,
            'terms': list(sample_frequencies.keys())[:5],
            'similarity_matrix': np.random.rand(5, 5),
            'methods_comparison': {
                'TF-IDF': {'precision': 0.75, 'recall': 0.68, 'f1_score': 0.71},
                'RAKE': {'precision': 0.65, 'recall': 0.72, 'f1_score': 0.68}
            }
        }

        output_dir = str(tmp_path / "viz_output")

        outputs = visualizer.generate_all_visualizations(data, output_dir)

        # Verificar que se generaron archivos
        assert len(outputs) > 0

        # Verificar que los archivos existen
        for key, path in outputs.items():
            assert os.path.exists(path)


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Tests de integración del sistema completo."""

    def test_full_workflow(self, sample_corpus, sample_terms, tmp_path):
        """Test: Workflow completo de análisis."""

        # 1. Normalizar términos
        normalizer = TermNormalizer()
        normalized_terms = normalizer.normalize_terms(sample_terms, show_progress=False)

        assert len(normalized_terms) == len(sample_terms)

        # 2. Filtrar stopwords
        stopwords = DomainStopwords()
        filtered_terms = stopwords.filter_terms(normalized_terms)

        assert len(filtered_terms) <= len(normalized_terms)

        # 3. Crear visualizaciones
        visualizer = TermVisualizer(normalizer=normalizer, stopwords=stopwords)

        frequencies = {term: i+1 for i, term in enumerate(filtered_terms)}

        fig = visualizer.plot_term_frequencies(frequencies, top_n=5, normalize=False)

        assert fig is not None

        # Guardar
        output_path = str(tmp_path / "test_plot.png")
        plt.savefig(output_path)
        plt.close(fig)

        assert os.path.exists(output_path)

    def test_corpus_processing(self, sample_corpus):
        """Test: Procesamiento de corpus completo."""
        normalizer = TermNormalizer()
        stopwords = DomainStopwords()

        # Extraer todos los términos del corpus
        all_terms = []
        for text in sample_corpus:
            terms = text.lower().split()
            all_terms.extend(terms)

        # Normalizar
        normalized = normalizer.normalize_terms(all_terms, show_progress=False)

        # Filtrar stopwords
        filtered = stopwords.filter_terms(normalized)

        # Contar frecuencias
        freq = Counter(filtered)

        assert len(freq) > 0
        assert all(isinstance(k, str) for k in freq.keys())
        assert all(isinstance(v, int) for v in freq.values())


# ============================================================================
# TESTS DE CASOS EDGE
# ============================================================================

class TestEdgeCases:
    """Tests de casos límite y situaciones especiales."""

    def test_empty_frequencies(self):
        """Test: Gráfico con frecuencias vacías."""
        visualizer = TermVisualizer()

        fig = visualizer.plot_term_frequencies({}, top_n=10)

        # Debe retornar None o manejar el caso
        assert fig is None

    def test_single_term_frequency(self):
        """Test: Gráfico con un solo término."""
        visualizer = TermVisualizer()

        fig = visualizer.plot_term_frequencies({'term': 5}, top_n=10, normalize=False)

        assert fig is not None
        plt.close(fig)

    def test_all_stopwords(self):
        """Test: Todas las frecuencias son stopwords."""
        visualizer = TermVisualizer()

        freq = {'paper': 10, 'research': 5, 'result': 3}

        fig = visualizer.plot_term_frequencies(freq, top_n=10, normalize=False)

        # Debe retornar None después del filtrado
        assert fig is None

    def test_very_long_term_names(self):
        """Test: Términos con nombres muy largos."""
        visualizer = TermVisualizer()

        freq = {
            'this is a very long term name that should be handled properly': 10,
            'another extremely long technical term': 5
        }

        fig = visualizer.plot_term_frequencies(freq, top_n=10, normalize=False)

        assert fig is not None
        plt.close(fig)

    def test_similarity_matrix_truncation(self):
        """Test: Truncamiento de matriz de similitud grande."""
        visualizer = TermVisualizer()

        # Crear matriz grande
        n = 50
        terms = [f"term_{i}" for i in range(n)]
        matrix = np.random.rand(n, n)

        fig = visualizer.plot_similarity_matrix(terms, matrix, max_terms=10)

        assert fig is not None
        plt.close(fig)


# ============================================================================
# TESTS DE RENDIMIENTO
# ============================================================================

class TestPerformance:
    """Tests de rendimiento con datasets grandes."""

    def test_normalize_large_term_list(self):
        """Test: Normalización de lista grande de términos."""
        normalizer = TermNormalizer()

        # Crear lista grande
        large_list = [f"term_{i}" for i in range(1000)]

        # Debe completarse sin errores
        normalized = normalizer.normalize_terms(large_list, show_progress=False)

        assert len(normalized) == len(large_list)

    def test_filter_large_stopword_list(self):
        """Test: Filtrado de lista grande."""
        stopwords = DomainStopwords()

        large_list = ['paper'] * 500 + ['deep learning'] * 500

        filtered = stopwords.filter_terms(large_list)

        # Debe filtrar todas las stopwords
        assert len(filtered) == 500
        assert all(t == 'deep learning' for t in filtered)


def run_all_tests():
    """Ejecuta todos los tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" EJECUTANDO TESTS: Term Analysis Module")
    print("="*70 + "\n")

    run_all_tests()
