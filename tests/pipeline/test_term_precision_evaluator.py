"""
Tests unitarios para Term Precision Evaluator
"""

import pytest
import numpy as np
from term_precision_evaluator import TermPrecisionEvaluator
import os
import tempfile


class TestTermPrecisionEvaluator:
    """Suite de tests para TermPrecisionEvaluator"""

    @pytest.fixture
    def sample_terms(self):
        """Fixture con términos de ejemplo"""
        predefined = [
            "machine learning",
            "neural networks",
            "deep learning",
            "natural language processing",
            "computer vision"
        ]

        extracted = [
            "machine learning",  # Match exacto
            "deep neural networks",  # Match parcial con neural networks
            "convolutional networks",  # Nuevo término
            "nlp techniques",  # Match parcial con NLP
            "image recognition",  # Nuevo término
            "deep learning algorithms"  # Match parcial con deep learning
        ]

        return predefined, extracted

    @pytest.fixture
    def sample_abstracts(self):
        """Fixture con abstracts de ejemplo"""
        return [
            "Machine learning and deep neural networks are used in image recognition.",
            "Convolutional networks excel at computer vision tasks.",
            "NLP techniques enable natural language processing applications.",
            "Deep learning algorithms require large datasets for training."
        ]

    @pytest.fixture
    def evaluator(self, sample_terms):
        """Fixture que crea un evaluador"""
        predefined, extracted = sample_terms
        return TermPrecisionEvaluator(predefined, extracted)

    def test_initialization(self, sample_terms):
        """Test: Inicialización correcta del evaluador"""
        predefined, extracted = sample_terms
        evaluator = TermPrecisionEvaluator(predefined, extracted)

        assert evaluator.predefined_terms == predefined
        assert evaluator.extracted_terms == extracted
        assert evaluator.model is not None
        assert evaluator.predefined_embeddings is not None
        assert evaluator.extracted_embeddings is not None

    def test_embeddings_shape(self, evaluator, sample_terms):
        """Test: Shape correcto de embeddings"""
        predefined, extracted = sample_terms

        # SBERT all-MiniLM-L6-v2 produce embeddings de dimensión 384
        assert evaluator.predefined_embeddings.shape == (len(predefined), 384)
        assert evaluator.extracted_embeddings.shape == (len(extracted), 384)

    def test_calculate_similarity_matrix(self, evaluator, sample_terms):
        """Test: Cálculo de matriz de similitud"""
        predefined, extracted = sample_terms

        matrix = evaluator.calculate_similarity_matrix()

        # Shape correcto
        assert matrix.shape == (len(predefined), len(extracted))

        # Valores en rango [0, 1]
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)

        # Matriz no vacía
        assert not np.all(matrix == 0)

    def test_identify_matches_structure(self, evaluator):
        """Test: Estructura correcta del resultado de identify_matches"""
        matches = evaluator.identify_matches(threshold=0.70)

        # Verificar que contiene todas las claves
        assert 'exact_matches' in matches
        assert 'partial_matches' in matches
        assert 'novel_terms' in matches
        assert 'predefined_not_found' in matches

        # Verificar que son listas
        assert isinstance(matches['exact_matches'], list)
        assert isinstance(matches['partial_matches'], list)
        assert isinstance(matches['novel_terms'], list)
        assert isinstance(matches['predefined_not_found'], list)

    def test_identify_matches_exact(self, evaluator):
        """Test: Identificación de matches exactos"""
        matches = evaluator.identify_matches(threshold=0.70)

        # Debe haber al menos un match exacto (machine learning = machine learning)
        assert len(matches['exact_matches']) >= 1

        # Verificar estructura de match exacto
        if matches['exact_matches']:
            match = matches['exact_matches'][0]
            assert 'predefined' in match
            assert 'extracted' in match
            assert 'similarity' in match
            assert match['similarity'] >= 0.70

    def test_identify_matches_threshold_impact(self, evaluator):
        """Test: Impacto del threshold en matches"""
        # Threshold alto: menos matches
        matches_strict = evaluator.identify_matches(threshold=0.90)

        # Threshold bajo: más matches
        matches_lenient = evaluator.identify_matches(threshold=0.50)

        # Con threshold más bajo debe haber más exact matches
        assert (len(matches_lenient['exact_matches']) >=
                len(matches_strict['exact_matches']))

    def test_calculate_metrics_structure(self, evaluator):
        """Test: Estructura de métricas calculadas"""
        evaluator.identify_matches()
        metrics = evaluator.calculate_metrics()

        # Verificar claves
        required_keys = [
            'precision', 'recall', 'f1_score', 'coverage',
            'n_predefined', 'n_extracted',
            'n_exact_matches', 'n_partial_matches',
            'n_novel_terms', 'n_predefined_not_found'
        ]

        for key in required_keys:
            assert key in metrics

    def test_calculate_metrics_ranges(self, evaluator):
        """Test: Rangos correctos de métricas"""
        metrics = evaluator.calculate_metrics()

        # Precision, recall, f1 deben estar en [0, 1]
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1

        # Coverage en [0, 100]
        assert 0 <= metrics['coverage'] <= 100

        # Conteos no negativos
        assert metrics['n_predefined'] >= 0
        assert metrics['n_extracted'] >= 0
        assert metrics['n_exact_matches'] >= 0
        assert metrics['n_partial_matches'] >= 0
        assert metrics['n_novel_terms'] >= 0
        assert metrics['n_predefined_not_found'] >= 0

    def test_calculate_metrics_consistency(self, evaluator):
        """Test: Consistencia en conteos de métricas"""
        metrics = evaluator.calculate_metrics()

        # Total de extracted debe ser suma de matched + novel
        total_matched = (metrics['n_exact_matches'] +
                        metrics['n_partial_matches'] +
                        metrics['n_novel_terms'])

        assert total_matched == metrics['n_extracted']

        # Total de predefined debe ser suma de found + not_found
        total_predefined = (metrics['n_exact_matches'] +
                           metrics['n_partial_matches'] +
                           metrics['n_predefined_not_found'])

        assert total_predefined == metrics['n_predefined']

    def test_explain_novel_terms_structure(self, evaluator, sample_abstracts):
        """Test: Estructura de explicación de términos nuevos"""
        matches = evaluator.identify_matches()

        if matches['novel_terms']:
            explanations = evaluator.explain_novel_terms(
                matches['novel_terms'],
                sample_abstracts
            )

            # Debe retornar diccionario
            assert isinstance(explanations, dict)

            # Cada término debe tener estructura correcta
            for term, info in explanations.items():
                assert 'frequency' in info
                assert 'example_contexts' in info
                assert 'relevance_score' in info
                assert 'interpretation' in info

                assert isinstance(info['frequency'], int)
                assert isinstance(info['example_contexts'], list)
                assert isinstance(info['relevance_score'], (int, float))
                assert isinstance(info['interpretation'], str)

    def test_find_contexts(self, evaluator, sample_abstracts):
        """Test: Búsqueda de contextos"""
        term = "machine learning"
        contexts = evaluator._find_contexts(term, sample_abstracts)

        # Debe encontrar al menos un contexto
        assert len(contexts) >= 1

        # Contexto debe contener el término
        assert any(term.lower() in ctx.lower() for ctx in contexts)

    def test_find_contexts_window(self, evaluator):
        """Test: Window parameter en find_contexts"""
        abstracts = [
            "This is a very long abstract with machine learning in the middle " * 10
        ]

        contexts_small = evaluator._find_contexts("machine learning", abstracts, window=20)
        contexts_large = evaluator._find_contexts("machine learning", abstracts, window=100)

        # Window más grande debe producir contextos más largos
        if contexts_small and contexts_large:
            assert len(contexts_large[0]) > len(contexts_small[0])

    def test_interpret_term(self, evaluator):
        """Test: Interpretación de términos"""
        # Contexto con keywords de método
        contexts = ["This method uses a novel approach for classification"]
        interpretation = evaluator._interpret_term("novel approach", contexts)

        assert isinstance(interpretation, str)
        assert len(interpretation) > 0

    def test_generate_report_creates_files(self, evaluator, sample_abstracts):
        """Test: Generación de reporte crea archivos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'test_report.md')

            evaluator.generate_evaluation_report(report_path, sample_abstracts)

            # Verificar que el reporte fue creado
            assert os.path.exists(report_path)

            # Verificar que tiene contenido
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
                assert 'Reporte de Evaluación' in content
                assert 'Precision' in content
                assert 'Recall' in content

            # Verificar que se creó directorio de visualizaciones
            viz_dir = report_path.replace('.md', '_visualizations')
            assert os.path.exists(viz_dir)

    def test_generate_report_visualizations(self, evaluator, sample_abstracts):
        """Test: Generación de visualizaciones"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'test_report.md')

            evaluator.generate_evaluation_report(report_path, sample_abstracts)

            viz_dir = report_path.replace('.md', '_visualizations')

            # Verificar que existen las visualizaciones
            venn_path = os.path.join(viz_dir, 'venn_diagram.png')
            heatmap_path = os.path.join(viz_dir, 'similarity_heatmap.png')

            assert os.path.exists(venn_path)
            assert os.path.exists(heatmap_path)

            # Verificar que tienen contenido (tamaño > 0)
            assert os.path.getsize(venn_path) > 0
            assert os.path.getsize(heatmap_path) > 0

    def test_empty_inputs(self):
        """Test: Manejo de entradas vacías"""
        # Listas vacías no deberían causar crash
        evaluator = TermPrecisionEvaluator([], [])

        # Embeddings deberían tener shape (0, 384)
        assert evaluator.predefined_embeddings.shape[0] == 0
        assert evaluator.extracted_embeddings.shape[0] == 0

    def test_single_term(self):
        """Test: Funcionamiento con un solo término"""
        evaluator = TermPrecisionEvaluator(
            ["machine learning"],
            ["machine learning"]
        )

        metrics = evaluator.calculate_metrics()

        # Debe tener precision y recall perfectos
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_no_matches(self):
        """Test: Caso sin matches"""
        evaluator = TermPrecisionEvaluator(
            ["machine learning", "neural networks"],
            ["cooking recipes", "music theory"]
        )

        matches = evaluator.identify_matches(threshold=0.70)
        metrics = evaluator.calculate_metrics()

        # No debe haber matches exactos
        assert len(matches['exact_matches']) == 0

        # Todos los extraídos deben ser novel
        assert len(matches['novel_terms']) == 2

        # Recall debe ser 0
        assert metrics['recall'] == 0.0

    def test_perfect_match(self):
        """Test: Caso de match perfecto"""
        terms = ["machine learning", "neural networks", "deep learning"]
        evaluator = TermPrecisionEvaluator(terms, terms)

        metrics = evaluator.calculate_metrics()

        # Precision, recall y f1 deben ser perfectos
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['coverage'] == 100.0

    def test_case_insensitivity(self, evaluator, sample_abstracts):
        """Test: Búsqueda case-insensitive"""
        term = "Machine Learning"
        contexts = evaluator._find_contexts(term, sample_abstracts)

        # Debe encontrar "machine learning" en lowercase
        assert len(contexts) > 0

    def test_metrics_without_matches(self, evaluator):
        """Test: Calcular métricas sin llamar identify_matches"""
        # Debe llamar identify_matches automáticamente
        metrics = evaluator.calculate_metrics()

        assert metrics is not None
        assert 'precision' in metrics

    def test_report_without_abstracts(self, evaluator):
        """Test: Reporte sin abstracts (opcional)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'test_report.md')

            # No pasar abstracts
            evaluator.generate_evaluation_report(report_path)

            # Debe crear el reporte de todos modos
            assert os.path.exists(report_path)

    def test_similarity_symmetry(self, evaluator):
        """Test: Similitud debe ser simétrica conceptualmente"""
        # Crear evaluador inverso
        evaluator_inverse = TermPrecisionEvaluator(
            evaluator.extracted_terms,
            evaluator.predefined_terms
        )

        matrix1 = evaluator.calculate_similarity_matrix()
        matrix2 = evaluator_inverse.calculate_similarity_matrix()

        # Las matrices deben ser transpuestas una de otra
        np.testing.assert_array_almost_equal(matrix1, matrix2.T, decimal=5)

    def test_relevance_score_calculation(self, evaluator, sample_abstracts):
        """Test: Cálculo de relevance score"""
        matches = evaluator.identify_matches()

        if matches['novel_terms']:
            explanations = evaluator.explain_novel_terms(
                matches['novel_terms'],
                sample_abstracts
            )

            for term, info in explanations.items():
                # Score debe ser positivo si hay frecuencia
                if info['frequency'] > 0:
                    assert info['relevance_score'] > 0

                # Score debe correlacionar con frecuencia
                assert info['relevance_score'] >= info['frequency']


class TestIntegration:
    """Tests de integración"""

    def test_full_pipeline(self):
        """Test: Pipeline completo de evaluación"""
        # Datos de entrada
        predefined = [
            "machine learning",
            "neural networks",
            "deep learning"
        ]

        extracted = [
            "machine learning",
            "deep neural networks",
            "reinforcement learning"
        ]

        abstracts = [
            "Machine learning is a subset of AI.",
            "Deep neural networks excel at pattern recognition.",
            "Reinforcement learning uses rewards for training."
        ]

        # Ejecutar pipeline completo
        evaluator = TermPrecisionEvaluator(predefined, extracted)
        evaluator.calculate_similarity_matrix()
        matches = evaluator.identify_matches(threshold=0.70)
        metrics = evaluator.calculate_metrics()

        novel_explanations = evaluator.explain_novel_terms(
            matches['novel_terms'],
            abstracts
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'integration_report.md')
            evaluator.generate_evaluation_report(report_path, abstracts)

            # Verificaciones finales
            assert os.path.exists(report_path)
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert isinstance(novel_explanations, dict)

        print("\n✓ Pipeline completo ejecutado exitosamente")


def run_tests():
    """Ejecuta todos los tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
