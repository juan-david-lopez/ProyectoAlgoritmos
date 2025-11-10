"""
Tests unitarios para todos los algoritmos de similitud.

Este módulo contiene tests exhaustivos para verificar:
    1. Rango de valores [0, 1]
    2. Casos extremos (textos vacíos, idénticos)
    3. Propiedades matemáticas (simetría, reflexividad)
    4. Casos conocidos con resultados esperados
    5. Robustez a errores
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.levenshtein import LevenshteinComparator
from src.algorithms.tfidf_cosine import TFIDFCosineComparator
from src.algorithms.jaccard import JaccardComparator
from src.algorithms.ngram import NGramComparator


class TestLevenshtein(unittest.TestCase):
    """Tests para el algoritmo de Levenshtein."""

    def setUp(self):
        """Inicializar comparador antes de cada test."""
        self.comparator = LevenshteinComparator()

    def test_identical_strings(self):
        """Textos idénticos deben tener similitud 1.0."""
        text = "hello world"
        sim = self.comparator.similarity(text, text)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_completely_different(self):
        """Textos completamente diferentes deben tener similitud baja."""
        sim = self.comparator.similarity("abc", "xyz")
        self.assertLess(sim, 0.5)

    def test_empty_strings(self):
        """Manejar cadenas vacías correctamente."""
        # Ambas vacías: similitud 1.0
        self.assertEqual(self.comparator.similarity("", ""), 1.0)
        # Una vacía: similitud 0.0
        self.assertEqual(self.comparator.similarity("", "hello"), 0.0)
        self.assertEqual(self.comparator.similarity("hello", ""), 0.0)

    def test_range_01(self):
        """Verificar que resultados están en [0, 1]."""
        texts = ["hello", "hallo", "world", ""]
        for i, text1 in enumerate(texts):
            for text2 in texts[i:]:
                sim = self.comparator.similarity(text1, text2)
                self.assertGreaterEqual(sim, 0.0, f"Similitud < 0 para '{text1}' y '{text2}'")
                self.assertLessEqual(sim, 1.0, f"Similitud > 1 para '{text1}' y '{text2}'")

    def test_known_case_kitten_sitting(self):
        """Caso conocido: 'kitten' vs 'sitting'."""
        # Distancia de Levenshtein = 3
        # Similitud = 1 - (3 / 7) ≈ 0.571
        sim = self.comparator.similarity("kitten", "sitting")
        self.assertAlmostEqual(sim, 0.571, places=2)

    def test_symmetry(self):
        """Verificar que sim(A, B) = sim(B, A)."""
        text1 = "the cat"
        text2 = "the dog"
        sim1 = self.comparator.similarity(text1, text2)
        sim2 = self.comparator.similarity(text2, text1)
        self.assertAlmostEqual(sim1, sim2, places=5)

    def test_compare_multiple(self):
        """Test de comparación múltiple."""
        texts = ["cat", "dog", "bird"]
        matrix = self.comparator.compare_multiple(texts)

        # Verificar forma
        self.assertEqual(matrix.shape, (3, 3))

        # Verificar diagonal = 1.0
        np.testing.assert_array_almost_equal(np.diag(matrix), [1.0, 1.0, 1.0])

        # Verificar simetría
        np.testing.assert_array_almost_equal(matrix, matrix.T)

        # Verificar rango
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0))


class TestTFIDFCosine(unittest.TestCase):
    """Tests para TF-IDF + Similitud del Coseno."""

    def setUp(self):
        """Inicializar comparador antes de cada test."""
        self.comparator = TFIDFCosineComparator()

    def test_identical_texts(self):
        """Textos idénticos deben tener similitud 1.0."""
        text = "the cat sits on the mat"
        matrix = self.comparator.compare_multiple([text, text])
        self.assertAlmostEqual(matrix[0][1], 1.0, places=5)

    def test_range_01(self):
        """Verificar que resultados están en [0, 1]."""
        texts = [
            "machine learning algorithms",
            "deep learning networks",
            "cooking pasta recipes",
            "car maintenance"
        ]
        matrix = self.comparator.compare_multiple(texts)
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0))

    def test_symmetry(self):
        """Verificar simetría de la matriz."""
        texts = ["text one", "text two", "text three"]
        matrix = self.comparator.compare_multiple(texts)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_diagonal_ones(self):
        """Verificar que la diagonal contiene 1.0."""
        texts = ["first", "second", "third"]
        matrix = self.comparator.compare_multiple(texts)
        np.testing.assert_array_almost_equal(np.diag(matrix), [1.0, 1.0, 1.0])

    def test_no_common_words(self):
        """Textos sin palabras comunes deben tener similitud 0.0."""
        texts = ["cat dog", "bird fish"]
        matrix = self.comparator.compare_multiple(texts)
        # Puede no ser exactamente 0 debido a preprocessing
        self.assertLess(matrix[0][1], 0.3)

    def test_single_text(self):
        """Manejar un solo texto correctamente."""
        matrix = self.comparator.compare_multiple(["hello"])
        self.assertEqual(matrix.shape, (1, 1))
        self.assertEqual(matrix[0][0], 1.0)

    def test_empty_list(self):
        """Manejar lista vacía."""
        matrix = self.comparator.compare_multiple([])
        self.assertEqual(matrix.shape, (0,))


class TestJaccard(unittest.TestCase):
    """Tests para el Índice de Jaccard."""

    def setUp(self):
        """Inicializar comparador antes de cada test."""
        self.comparator = JaccardComparator()

    def test_identical_sets(self):
        """Conjuntos idénticos deben tener similitud 1.0."""
        text = "cat dog bird"
        sim = self.comparator.similarity(text, text)
        self.assertEqual(sim, 1.0)

    def test_disjoint_sets(self):
        """Conjuntos disjuntos deben tener similitud 0.0."""
        sim = self.comparator.similarity("cat dog", "bird fish")
        self.assertEqual(sim, 0.0)

    def test_known_case(self):
        """Caso conocido: intersección/unión."""
        # A = {cat, dog, bird}
        # B = {cat, fish, bird}
        # Intersección = {cat, bird} = 2
        # Unión = {cat, dog, bird, fish} = 4
        # Jaccard = 2/4 = 0.5
        sim = self.comparator.similarity("cat dog bird", "cat fish bird")
        self.assertAlmostEqual(sim, 0.5, places=5)

    def test_empty_strings(self):
        """Manejar cadenas vacías."""
        self.assertEqual(self.comparator.similarity("", ""), 1.0)
        self.assertEqual(self.comparator.similarity("", "hello"), 0.0)

    def test_case_sensitivity(self):
        """Verificar que lowercase funciona."""
        comparator_case = JaccardComparator(lowercase=True)
        comparator_nocase = JaccardComparator(lowercase=False)

        sim_case = comparator_case.similarity("Cat", "cat")
        sim_nocase = comparator_nocase.similarity("Cat", "cat")

        self.assertEqual(sim_case, 1.0)  # Debe ser igual con lowercase
        self.assertEqual(sim_nocase, 0.0)  # Debe ser diferente sin lowercase

    def test_range_01(self):
        """Verificar rango [0, 1]."""
        texts = ["cat dog", "dog bird", "fish", "cat fish bird"]
        matrix = self.comparator.compare_multiple(texts)
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0))

    def test_common_words_extraction(self):
        """Test de extracción de palabras comunes."""
        common = self.comparator.get_common_words("cat dog bird", "cat fish bird")
        self.assertEqual(common, {"cat", "bird"})

    def test_unique_words_extraction(self):
        """Test de extracción de palabras únicas."""
        unique1, unique2 = self.comparator.get_unique_words("cat dog", "cat fish")
        self.assertEqual(unique1, {"dog"})
        self.assertEqual(unique2, {"fish"})


class TestNGram(unittest.TestCase):
    """Tests para similitud de N-gramas."""

    def setUp(self):
        """Inicializar comparador antes de cada test."""
        self.comparator = NGramComparator(n=3, method='dice')

    def test_identical_texts(self):
        """Textos idénticos deben tener similitud 1.0."""
        text = "hello"
        sim = self.comparator.similarity(text, text)
        self.assertEqual(sim, 1.0)

    def test_completely_different(self):
        """Textos completamente diferentes deben tener similitud baja."""
        sim = self.comparator.similarity("abc", "xyz")
        self.assertEqual(sim, 0.0)

    def test_known_case_bigrams(self):
        """Caso conocido con bigramas."""
        comparator = NGramComparator(n=2, method='jaccard')

        # "abc" → {ab, bc}
        # "bcd" → {bc, cd}
        # Intersección = {bc} = 1
        # Unión = {ab, bc, cd} = 3
        # Jaccard = 1/3 ≈ 0.333

        sim = comparator.similarity("abc", "bcd")
        self.assertAlmostEqual(sim, 0.333, places=2)

    def test_dice_vs_jaccard(self):
        """Verificar que Dice ≥ Jaccard."""
        comparator_dice = NGramComparator(n=3, method='dice')
        comparator_jaccard = NGramComparator(n=3, method='jaccard')

        text1 = "hello"
        text2 = "hallo"

        dice = comparator_dice.similarity(text1, text2)
        jaccard = comparator_jaccard.similarity(text1, text2)

        self.assertGreaterEqual(dice, jaccard)

    def test_short_texts(self):
        """Manejar textos más cortos que n."""
        comparator = NGramComparator(n=5)
        sim = comparator.similarity("cat", "cat")
        self.assertEqual(sim, 1.0)

    def test_empty_strings(self):
        """Manejar cadenas vacías."""
        self.assertEqual(self.comparator.similarity("", ""), 1.0)
        self.assertEqual(self.comparator.similarity("", "hello"), 0.0)

    def test_range_01(self):
        """Verificar rango [0, 1]."""
        texts = ["hello", "hallo", "hola", "bonjour"]
        matrix = self.comparator.compare_multiple(texts)
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0))

    def test_ngram_extraction(self):
        """Test de extracción de n-gramas."""
        ngrams = self.comparator.extract_ngrams("hello")
        expected = {"hel", "ell", "llo"}
        self.assertEqual(ngrams, expected)


class TestMatrixProperties(unittest.TestCase):
    """Tests de propiedades matemáticas de matrices de similitud."""

    def test_all_algorithms_range(self):
        """Verificar que todos los algoritmos retornan valores en [0, 1]."""
        texts = [
            "The cat sits on the mat",
            "A dog runs in the park",
            "Birds fly in the sky"
        ]

        algorithms = [
            LevenshteinComparator(),
            TFIDFCosineComparator(),
            JaccardComparator(),
            NGramComparator(n=3)
        ]

        for algo in algorithms:
            with self.subTest(algorithm=algo.__class__.__name__):
                matrix = algo.compare_multiple(texts)

                # Verificar rango
                self.assertTrue(np.all(matrix >= 0.0),
                               f"{algo.__class__.__name__}: valores < 0")
                self.assertTrue(np.all(matrix <= 1.0),
                               f"{algo.__class__.__name__}: valores > 1")

                # Verificar diagonal
                np.testing.assert_array_almost_equal(
                    np.diag(matrix),
                    np.ones(len(texts)),
                    decimal=5,
                    err_msg=f"{algo.__class__.__name__}: diagonal != 1"
                )

                # Verificar simetría
                np.testing.assert_array_almost_equal(
                    matrix,
                    matrix.T,
                    decimal=5,
                    err_msg=f"{algo.__class__.__name__}: matriz no simétrica"
                )


class TestEdgeCases(unittest.TestCase):
    """Tests de casos extremos y robustez."""

    def test_very_long_texts(self):
        """Manejar textos muy largos."""
        text = "word " * 1000  # 1000 palabras
        comparator = JaccardComparator()
        sim = comparator.similarity(text, text)
        self.assertEqual(sim, 1.0)

    def test_special_characters(self):
        """Manejar caracteres especiales."""
        comparator = JaccardComparator()
        text1 = "hello @world #test"
        text2 = "hello @world #test"
        sim = comparator.similarity(text1, text2)
        self.assertGreater(sim, 0.0)

    def test_unicode_characters(self):
        """Manejar caracteres Unicode."""
        comparator = LevenshteinComparator()
        text1 = "café résumé"
        text2 = "café résumé"
        sim = comparator.similarity(text1, text2)
        self.assertEqual(sim, 1.0)

    def test_numbers(self):
        """Manejar números."""
        comparator = TFIDFCosineComparator()
        texts = ["123 456", "456 789"]
        matrix = comparator.compare_multiple(texts)
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0))


def run_tests():
    """Ejecutar todos los tests."""
    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar todos los tests
    suite.addTests(loader.loadTestsFromTestCase(TestLevenshtein))
    suite.addTests(loader.loadTestsFromTestCase(TestTFIDFCosine))
    suite.addTests(loader.loadTestsFromTestCase(TestJaccard))
    suite.addTests(loader.loadTestsFromTestCase(TestNGram))
    suite.addTests(loader.loadTestsFromTestCase(TestMatrixProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Imprimir resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallidos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
