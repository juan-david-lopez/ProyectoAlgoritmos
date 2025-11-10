"""
Comprehensive tests for ClusteringPreprocessor module.

Tests cover:
- Initialization and configuration
- Deep text cleaning
- Advanced tokenization
- Stopwords removal
- Lemmatization
- Vectorization methods (TF-IDF, Word2Vec, SBERT)
- Full preprocessing pipeline
- Edge cases and error handling
"""

import unittest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from clustering.clustering_preprocessing import (
    ClusteringPreprocessor,
    GENSIM_AVAILABLE,
    SBERT_AVAILABLE
)

# Suprimir warnings durante tests
warnings.filterwarnings('ignore')


class TestClusteringPreprocessorInitialization(unittest.TestCase):
    """Tests para inicialización del preprocesador."""

    def setUp(self):
        """Configurar abstracts de prueba."""
        self.sample_abstracts = [
            "Deep learning models are used for image classification.",
            "Natural language processing with transformer architectures.",
            "Reinforcement learning agents learn through rewards."
        ]

    def test_initialization_basic(self):
        """Test inicialización básica."""
        preprocessor = ClusteringPreprocessor(self.sample_abstracts)

        self.assertEqual(preprocessor.n_abstracts, 3)
        self.assertEqual(len(preprocessor.abstracts), 3)
        self.assertIsNotNone(preprocessor.nlp)

    def test_initialization_stores_abstracts(self):
        """Test que los abstracts se almacenan correctamente."""
        preprocessor = ClusteringPreprocessor(self.sample_abstracts)

        self.assertEqual(preprocessor.abstracts, self.sample_abstracts)

    def test_initialization_compiles_regex_patterns(self):
        """Test que los patrones regex se compilan."""
        preprocessor = ClusteringPreprocessor(self.sample_abstracts)

        self.assertIsNotNone(preprocessor.url_pattern)
        self.assertIsNotNone(preprocessor.email_pattern)
        self.assertIsNotNone(preprocessor.doi_pattern)
        self.assertIsNotNone(preprocessor.number_pattern)
        self.assertIsNotNone(preprocessor.special_chars_pattern)
        self.assertIsNotNone(preprocessor.whitespace_pattern)

    def test_initialization_empty_abstracts(self):
        """Test inicialización con lista vacía."""
        preprocessor = ClusteringPreprocessor([])

        self.assertEqual(preprocessor.n_abstracts, 0)

    def test_academic_stopwords_exist(self):
        """Test que las stopwords académicas están definidas."""
        self.assertIsInstance(ClusteringPreprocessor.ACADEMIC_STOPWORDS, set)
        self.assertGreater(len(ClusteringPreprocessor.ACADEMIC_STOPWORDS), 30)

        # Verificar algunas stopwords clave
        expected_stopwords = {'paper', 'study', 'research', 'author', 'result'}
        self.assertTrue(expected_stopwords.issubset(ClusteringPreprocessor.ACADEMIC_STOPWORDS))


class TestDeepCleaning(unittest.TestCase):
    """Tests para limpieza profunda de textos."""

    def setUp(self):
        """Configurar preprocesador."""
        self.preprocessor = ClusteringPreprocessor(["dummy"])

    def test_lowercase_conversion(self):
        """Test conversión a minúsculas."""
        text = "Deep Learning And NLP"
        result = self.preprocessor.deep_clean(text)

        self.assertEqual(result, "deep learning and nlp")

    def test_url_removal(self):
        """Test eliminación de URLs."""
        text = "Check https://example.com or www.test.org for details"
        result = self.preprocessor.deep_clean(text)

        self.assertNotIn("https://example.com", result)
        self.assertNotIn("www.test.org", result)

    def test_email_removal(self):
        """Test eliminación de emails."""
        text = "Contact author@university.edu for more info"
        result = self.preprocessor.deep_clean(text)

        self.assertNotIn("author@university.edu", result)

    def test_doi_removal(self):
        """Test eliminación de DOIs."""
        text = "See DOI: 10.1234/example.2023 for reference"
        result = self.preprocessor.deep_clean(text)

        self.assertNotIn("10.1234", result)

    def test_number_normalization(self):
        """Test normalización de números a TOKEN_NUM."""
        text = "We tested 100 samples with accuracy of 95.6%"
        result = self.preprocessor.deep_clean(text)

        self.assertIn("token_num", result)
        self.assertNotIn("100", result)
        self.assertNotIn("95.6", result)

    def test_hyphen_preservation_in_compounds(self):
        """Test preservación de guiones en términos compuestos."""
        text = "Multi-task learning and end-to-end training"
        result = self.preprocessor.deep_clean(text)

        # Guiones entre letras deberían preservarse
        self.assertIn("multi-task", result)
        self.assertIn("end-to-end", result)

    def test_special_chars_removal(self):
        """Test eliminación de caracteres especiales."""
        text = "Model (accuracy: 95%) performs well!"
        result = self.preprocessor.deep_clean(text)

        self.assertNotIn("(", result)
        self.assertNotIn(")", result)
        self.assertNotIn(":", result)
        self.assertNotIn("!", result)

    def test_whitespace_normalization(self):
        """Test normalización de espacios en blanco."""
        text = "Multiple    spaces   and\n\nnewlines"
        result = self.preprocessor.deep_clean(text)

        self.assertNotIn("  ", result)  # No dobles espacios
        self.assertNotIn("\n", result)  # No newlines

    def test_empty_string_handling(self):
        """Test manejo de strings vacíos."""
        result = self.preprocessor.deep_clean("")
        self.assertEqual(result, "")

    def test_none_handling(self):
        """Test manejo de None."""
        result = self.preprocessor.deep_clean(None)
        self.assertEqual(result, "")

    def test_non_string_handling(self):
        """Test manejo de entrada no-string."""
        result = self.preprocessor.deep_clean(12345)
        self.assertEqual(result, "")


class TestAdvancedTokenization(unittest.TestCase):
    """Tests para tokenización avanzada con POS tagging."""

    def setUp(self):
        """Configurar preprocesador."""
        self.preprocessor = ClusteringPreprocessor(["dummy"])

    def test_basic_tokenization(self):
        """Test tokenización básica."""
        text = "deep learning model"
        tokens = self.preprocessor.advanced_tokenization(text)

        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_pos_filtering_keeps_nouns(self):
        """Test que se preservan sustantivos."""
        text = "neural network model"
        tokens = self.preprocessor.advanced_tokenization(text)

        # Debe contener estos sustantivos
        self.assertTrue(any(t in tokens for t in ['neural', 'network', 'model']))

    def test_pos_filtering_keeps_adjectives(self):
        """Test que se preservan adjetivos."""
        text = "deep convolutional architecture"
        tokens = self.preprocessor.advanced_tokenization(text)

        # Debe contener estos adjetivos
        self.assertTrue(any(t in tokens for t in ['deep', 'convolutional']))

    def test_removes_articles_and_prepositions(self):
        """Test que se eliminan artículos y preposiciones."""
        text = "the model is trained on the dataset"
        tokens = self.preprocessor.advanced_tokenization(text)

        # No debe contener artículos/preposiciones
        self.assertNotIn('the', tokens)
        self.assertNotIn('is', tokens)
        self.assertNotIn('on', tokens)

    def test_important_verbs_preserved(self):
        """Test que verbos importantes se preservan."""
        text = "we train and evaluate the model to classify images"
        tokens = self.preprocessor.advanced_tokenization(text)

        # Verbos importantes deben estar presentes
        important = ['train', 'evaluate', 'classify']
        self.assertTrue(any(verb in tokens for verb in important))

    def test_short_tokens_removed(self):
        """Test que tokens cortos (<2 chars) se eliminan."""
        text = "a neural network is used"
        tokens = self.preprocessor.advanced_tokenization(text)

        # No debe contener tokens de 1 carácter
        self.assertTrue(all(len(t) >= 2 for t in tokens))

    def test_punctuation_removed(self):
        """Test que puntuación se elimina."""
        text = "model, network, architecture!"
        tokens = self.preprocessor.advanced_tokenization(text)

        # No debe contener puntuación
        self.assertNotIn(',', tokens)
        self.assertNotIn('!', tokens)

    def test_empty_text_handling(self):
        """Test manejo de texto vacío."""
        tokens = self.preprocessor.advanced_tokenization("")
        self.assertEqual(tokens, [])

    def test_lowercase_tokens(self):
        """Test que tokens son lowercase."""
        text = "Deep Learning Network"
        tokens = self.preprocessor.advanced_tokenization(text)

        self.assertTrue(all(t.islower() for t in tokens))


class TestStopwordsRemoval(unittest.TestCase):
    """Tests para eliminación inteligente de stopwords."""

    def setUp(self):
        """Configurar preprocesador."""
        self.preprocessor = ClusteringPreprocessor(["dummy"])

    def test_academic_stopwords_removed(self):
        """Test que stopwords académicas se eliminan."""
        tokens = ['paper', 'study', 'neural', 'network', 'research', 'model']
        filtered = self.preprocessor.remove_stopwords(tokens)

        # Stopwords académicas eliminadas
        self.assertNotIn('paper', filtered)
        self.assertNotIn('study', filtered)
        self.assertNotIn('research', filtered)

        # Términos técnicos preservados
        self.assertIn('neural', filtered)
        self.assertIn('network', filtered)
        self.assertIn('model', filtered)

    def test_technical_terms_preserved(self):
        """Test que términos técnicos se preservan."""
        technical = ['neural', 'network', 'learning', 'deep', 'transformer', 'attention']
        filtered = self.preprocessor.remove_stopwords(technical)

        # Todos los términos técnicos deben preservarse
        self.assertEqual(set(filtered), set(technical))

    def test_ml_architecture_terms_preserved(self):
        """Test que arquitecturas ML se preservan."""
        architectures = ['cnn', 'rnn', 'lstm', 'gru', 'bert', 'gpt', 'resnet']
        filtered = self.preprocessor.remove_stopwords(architectures)

        # Todas las arquitecturas deben preservarse
        self.assertEqual(set(filtered), set(architectures))

    def test_custom_stopwords_added(self):
        """Test añadir stopwords personalizadas."""
        tokens = ['neural', 'network', 'custom_stop']
        custom = {'custom_stop'}

        filtered = self.preprocessor.remove_stopwords(tokens, custom_stopwords=custom)

        self.assertNotIn('custom_stop', filtered)
        self.assertIn('neural', filtered)
        self.assertIn('network', filtered)

    def test_empty_tokens_handling(self):
        """Test manejo de lista vacía."""
        filtered = self.preprocessor.remove_stopwords([])
        self.assertEqual(filtered, [])

    def test_preserves_domain_terms(self):
        """Test preservación de términos de dominio."""
        domain_terms = ['classification', 'regression', 'clustering', 'segmentation',
                       'detection', 'recognition', 'nlp', 'vision']
        filtered = self.preprocessor.remove_stopwords(domain_terms)

        # Todos los términos de dominio preservados
        self.assertEqual(set(filtered), set(domain_terms))


class TestLemmatization(unittest.TestCase):
    """Tests para lematización."""

    def setUp(self):
        """Configurar preprocesador."""
        self.preprocessor = ClusteringPreprocessor(["dummy"])

    def test_plural_to_singular(self):
        """Test conversión plural a singular."""
        tokens = ['models', 'networks', 'algorithms']
        lemmas = self.preprocessor.lemmatize(tokens)

        self.assertIn('model', lemmas)
        self.assertIn('network', lemmas)
        self.assertIn('algorithm', lemmas)

    def test_verb_normalization(self):
        """Test normalización de verbos."""
        tokens = ['training', 'trained', 'generated', 'generating']
        lemmas = self.preprocessor.lemmatize(tokens)

        # Verbos deben normalizarse a forma base
        self.assertTrue(any(l in ['train', 'training'] for l in lemmas))
        self.assertTrue(any(l in ['generate', 'generating'] for l in lemmas))

    def test_empty_tokens_handling(self):
        """Test manejo de lista vacía."""
        lemmas = self.preprocessor.lemmatize([])
        self.assertEqual(lemmas, [])

    def test_pronouns_removed(self):
        """Test que pronombres como -PRON- se eliminan."""
        tokens = ['we', 'they', 'it', 'model']
        lemmas = self.preprocessor.lemmatize(tokens)

        # No debe contener -PRON-
        self.assertNotIn('-pron-', lemmas)

    def test_preserves_technical_terms(self):
        """Test preservación de términos técnicos."""
        tokens = ['neural', 'learning', 'deep', 'attention']
        lemmas = self.preprocessor.lemmatize(tokens)

        # Términos técnicos deben estar presentes
        self.assertTrue(len(lemmas) > 0)


class TestTFIDFVectorization(unittest.TestCase):
    """Tests para vectorización TF-IDF."""

    def setUp(self):
        """Configurar abstracts de prueba."""
        self.abstracts = [
            "deep learning neural network model",
            "natural language processing transformer",
            "convolutional neural network image classification"
        ]
        self.preprocessor = ClusteringPreprocessor(self.abstracts)

        # Preprocesar textos
        cleaned = [self.preprocessor.deep_clean(t) for t in self.abstracts]
        tokenized = [self.preprocessor.advanced_tokenization(t) for t in cleaned]
        filtered = [self.preprocessor.remove_stopwords(t) for t in tokenized]
        lemmatized = [self.preprocessor.lemmatize(t) for t in filtered]
        self.processed_texts = [' '.join(tokens) for tokens in lemmatized]

    def test_tfidf_returns_matrix_and_vectorizer(self):
        """Test que TF-IDF retorna matriz y vectorizador."""
        matrix, vectorizer = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='tfidf'
        )

        self.assertIsInstance(matrix, np.ndarray)
        self.assertIsNotNone(vectorizer)

    def test_tfidf_matrix_shape(self):
        """Test dimensiones de matriz TF-IDF."""
        matrix, _ = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='tfidf'
        )

        # Filas = número de documentos
        self.assertEqual(matrix.shape[0], len(self.processed_texts))
        # Columnas > 0
        self.assertGreater(matrix.shape[1], 0)

    def test_tfidf_custom_params(self):
        """Test parámetros personalizados para TF-IDF."""
        matrix, vectorizer = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='tfidf',
            max_features=10,
            ngram_range=(1, 2)
        )

        # Max features respetado
        self.assertLessEqual(matrix.shape[1], 10)

    def test_tfidf_matrix_numeric(self):
        """Test que matriz contiene valores numéricos."""
        matrix, _ = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='tfidf'
        )

        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrix >= 0).all())  # TF-IDF no negativo


class TestWord2VecVectorization(unittest.TestCase):
    """Tests para vectorización Word2Vec."""

    def setUp(self):
        """Configurar abstracts de prueba."""
        self.abstracts = [
            "deep learning neural network model architecture",
            "natural language processing transformer attention mechanism",
            "convolutional neural network image classification task",
            "reinforcement learning agent policy gradient method",
            "generative adversarial network image generation"
        ]
        self.preprocessor = ClusteringPreprocessor(self.abstracts)

        # Preprocesar textos
        cleaned = [self.preprocessor.deep_clean(t) for t in self.abstracts]
        tokenized = [self.preprocessor.advanced_tokenization(t) for t in cleaned]
        filtered = [self.preprocessor.remove_stopwords(t) for t in tokenized]
        lemmatized = [self.preprocessor.lemmatize(t) for t in filtered]
        self.processed_texts = [' '.join(tokens) for tokens in lemmatized]

    @unittest.skipIf(not GENSIM_AVAILABLE, "Gensim not available")
    def test_word2vec_returns_matrix_and_model(self):
        """Test que Word2Vec retorna matriz y modelo."""
        matrix, model = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='word2vec'
        )

        self.assertIsInstance(matrix, np.ndarray)
        self.assertIsNotNone(model)

    @unittest.skipIf(not GENSIM_AVAILABLE, "Gensim not available")
    def test_word2vec_matrix_shape(self):
        """Test dimensiones de matriz Word2Vec."""
        matrix, _ = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='word2vec',
            vector_size=50
        )

        # Filas = número de documentos
        self.assertEqual(matrix.shape[0], len(self.processed_texts))
        # Columnas = vector_size
        self.assertEqual(matrix.shape[1], 50)

    @unittest.skipIf(not GENSIM_AVAILABLE, "Gensim not available")
    def test_word2vec_custom_params(self):
        """Test parámetros personalizados."""
        matrix, model = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='word2vec',
            vector_size=100,
            window=3,
            min_count=1
        )

        self.assertEqual(matrix.shape[1], 100)

    @unittest.skipIf(not GENSIM_AVAILABLE, "Gensim not available")
    def test_word2vec_matrix_numeric(self):
        """Test que matriz contiene valores numéricos válidos."""
        matrix, _ = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='word2vec'
        )

        self.assertTrue(np.isfinite(matrix).all())


class TestSBERTVectorization(unittest.TestCase):
    """Tests para vectorización SBERT."""

    def setUp(self):
        """Configurar abstracts de prueba."""
        self.abstracts = [
            "deep learning neural network",
            "natural language processing",
            "computer vision models"
        ]
        self.preprocessor = ClusteringPreprocessor(self.abstracts)

        # Preprocesar textos
        cleaned = [self.preprocessor.deep_clean(t) for t in self.abstracts]
        tokenized = [self.preprocessor.advanced_tokenization(t) for t in cleaned]
        filtered = [self.preprocessor.remove_stopwords(t) for t in tokenized]
        lemmatized = [self.preprocessor.lemmatize(t) for t in filtered]
        self.processed_texts = [' '.join(tokens) for tokens in lemmatized]

    @unittest.skipIf(not SBERT_AVAILABLE, "Sentence-Transformers not available")
    def test_sbert_returns_matrix_and_model(self):
        """Test que SBERT retorna matriz y modelo."""
        matrix, model = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='sbert'
        )

        self.assertIsInstance(matrix, np.ndarray)
        self.assertIsNotNone(model)

    @unittest.skipIf(not SBERT_AVAILABLE, "Sentence-Transformers not available")
    def test_sbert_matrix_shape(self):
        """Test dimensiones de matriz SBERT."""
        matrix, _ = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='sbert'
        )

        # Filas = número de documentos
        self.assertEqual(matrix.shape[0], len(self.processed_texts))
        # Columnas > 0 (depende del modelo)
        self.assertGreater(matrix.shape[1], 0)

    @unittest.skipIf(not SBERT_AVAILABLE, "Sentence-Transformers not available")
    def test_sbert_matrix_numeric(self):
        """Test que matriz contiene valores numéricos válidos."""
        matrix, _ = self.preprocessor.vectorize_texts(
            self.processed_texts,
            method='sbert'
        )

        self.assertTrue(np.isfinite(matrix).all())


class TestVectorizationErrors(unittest.TestCase):
    """Tests para manejo de errores en vectorización."""

    def setUp(self):
        """Configurar preprocesador."""
        self.preprocessor = ClusteringPreprocessor(["test text"])
        self.processed_texts = ["test"]

    def test_invalid_method_raises_error(self):
        """Test que método inválido lanza error."""
        with self.assertRaises(ValueError):
            self.preprocessor.vectorize_texts(
                self.processed_texts,
                method='invalid_method'
            )

    @unittest.skipIf(GENSIM_AVAILABLE, "Test only when Gensim not available")
    def test_word2vec_without_gensim_raises_error(self):
        """Test que Word2Vec sin gensim lanza error."""
        with self.assertRaises(ImportError):
            self.preprocessor.vectorize_texts(
                self.processed_texts,
                method='word2vec'
            )

    @unittest.skipIf(SBERT_AVAILABLE, "Test only when SBERT not available")
    def test_sbert_without_transformers_raises_error(self):
        """Test que SBERT sin sentence-transformers lanza error."""
        with self.assertRaises(ImportError):
            self.preprocessor.vectorize_texts(
                self.processed_texts,
                method='sbert'
            )


class TestFullPipeline(unittest.TestCase):
    """Tests para pipeline completo end-to-end."""

    def setUp(self):
        """Configurar abstracts de prueba."""
        self.abstracts = [
            """Deep learning and convolutional neural networks have revolutionized
            computer vision. We propose a new architecture for image classification.""",

            """Natural language processing uses transformer models and attention mechanisms.
            BERT and GPT demonstrate state-of-the-art results in text understanding.""",

            """Reinforcement learning agents learn through interaction with environments.
            Q-learning and policy gradients are fundamental algorithms in RL."""
        ]
        self.preprocessor = ClusteringPreprocessor(self.abstracts)

    def test_pipeline_tfidf_completes(self):
        """Test que pipeline con TF-IDF completa exitosamente."""
        result = self.preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertIn('feature_matrix', result)
        self.assertIn('vectorizer', result)
        self.assertIn('processed_texts', result)
        self.assertIn('n_documents', result)
        self.assertIn('n_features', result)
        self.assertIn('method', result)

    def test_pipeline_returns_correct_documents(self):
        """Test que pipeline procesa todos los documentos."""
        result = self.preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], len(self.abstracts))
        self.assertEqual(len(result['processed_texts']), len(self.abstracts))

    def test_pipeline_feature_matrix_shape(self):
        """Test dimensiones de matriz de features."""
        result = self.preprocessor.full_preprocessing_pipeline(method='tfidf')

        matrix = result['feature_matrix']
        self.assertEqual(matrix.shape[0], len(self.abstracts))
        self.assertEqual(matrix.shape[1], result['n_features'])

    def test_pipeline_with_intermediate_results(self):
        """Test pipeline con resultados intermedios."""
        result = self.preprocessor.full_preprocessing_pipeline(
            method='tfidf',
            return_intermediate=True
        )

        self.assertIn('cleaned_texts', result)
        self.assertIn('tokenized_texts', result)
        self.assertIn('lemmatized_texts', result)
        self.assertIn('vocabulary_size', result)

    def test_pipeline_intermediate_text_counts(self):
        """Test conteo de textos en resultados intermedios."""
        result = self.preprocessor.full_preprocessing_pipeline(
            method='tfidf',
            return_intermediate=True
        )

        n_docs = len(self.abstracts)
        self.assertEqual(len(result['cleaned_texts']), n_docs)
        self.assertEqual(len(result['tokenized_texts']), n_docs)
        self.assertEqual(len(result['lemmatized_texts']), n_docs)

    def test_pipeline_custom_vectorization_params(self):
        """Test pipeline con parámetros personalizados."""
        result = self.preprocessor.full_preprocessing_pipeline(
            method='tfidf',
            max_features=50,
            ngram_range=(1, 2)
        )

        # Verificar que max_features se respeta
        self.assertLessEqual(result['n_features'], 50)

    @unittest.skipIf(not GENSIM_AVAILABLE, "Gensim not available")
    def test_pipeline_word2vec_completes(self):
        """Test pipeline con Word2Vec."""
        result = self.preprocessor.full_preprocessing_pipeline(
            method='word2vec',
            vector_size=50
        )

        self.assertEqual(result['method'], 'word2vec')
        self.assertEqual(result['feature_matrix'].shape[1], 50)

    @unittest.skipIf(not SBERT_AVAILABLE, "Sentence-Transformers not available")
    def test_pipeline_sbert_completes(self):
        """Test pipeline con SBERT."""
        result = self.preprocessor.full_preprocessing_pipeline(method='sbert')

        self.assertEqual(result['method'], 'sbert')
        self.assertGreater(result['feature_matrix'].shape[1], 0)


class TestEdgeCases(unittest.TestCase):
    """Tests para casos extremos y manejo de errores."""

    def test_single_abstract(self):
        """Test con un solo abstract."""
        preprocessor = ClusteringPreprocessor(["Deep learning model"])
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], 1)

    def test_very_short_abstracts(self):
        """Test con abstracts muy cortos."""
        short_abstracts = ["AI", "ML", "DL"]
        preprocessor = ClusteringPreprocessor(short_abstracts)
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], 3)

    def test_abstracts_with_special_unicode(self):
        """Test con caracteres Unicode especiales."""
        unicode_abstracts = [
            "Deep learning with α=0.001 and β=0.9",
            "Résumé of neural networks",
            "Machine learning — state of the art"
        ]
        preprocessor = ClusteringPreprocessor(unicode_abstracts)
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], 3)

    def test_abstracts_with_excessive_punctuation(self):
        """Test con puntuación excesiva."""
        punctuated = [
            "Model!!! Training??? Testing...",
            "Architecture: CNN, RNN, LSTM!!!",
            "Results (95% accuracy) are significant!!!"
        ]
        preprocessor = ClusteringPreprocessor(punctuated)
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], 3)

    def test_duplicate_abstracts(self):
        """Test con abstracts duplicados."""
        duplicates = [
            "Deep learning model",
            "Deep learning model",
            "Deep learning model"
        ]
        preprocessor = ClusteringPreprocessor(duplicates)
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], 3)

    def test_mixed_empty_and_valid_abstracts(self):
        """Test con mezcla de abstracts vacíos y válidos."""
        mixed = [
            "Deep learning model",
            "",
            "Natural language processing",
            "   ",
            "Computer vision"
        ]
        preprocessor = ClusteringPreprocessor(mixed)
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertEqual(result['n_documents'], 5)


class TestStatisticsAndLogging(unittest.TestCase):
    """Tests para estadísticas y logging."""

    def setUp(self):
        """Configurar abstracts."""
        self.abstracts = [
            "Deep learning neural network model for classification",
            "Transformer architecture with attention mechanism",
            "Convolutional neural network for image processing"
        ]

    def test_preprocessor_stores_intermediate_results(self):
        """Test que resultados intermedios se almacenan."""
        preprocessor = ClusteringPreprocessor(self.abstracts)
        preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertGreater(len(preprocessor.cleaned_texts), 0)
        self.assertGreater(len(preprocessor.tokenized_texts), 0)
        self.assertGreater(len(preprocessor.processed_texts), 0)

    def test_vectorizer_is_stored(self):
        """Test que vectorizador se almacena."""
        preprocessor = ClusteringPreprocessor(self.abstracts)
        preprocessor.full_preprocessing_pipeline(method='tfidf')

        self.assertIsNotNone(preprocessor.vectorizer)
        self.assertIsNotNone(preprocessor.feature_matrix)


class TestIntegrationScenarios(unittest.TestCase):
    """Tests de integración con escenarios realistas."""

    def test_scientific_abstract_processing(self):
        """Test con abstract científico realista."""
        abstract = """
        In this paper, we propose a novel deep learning architecture for image classification.
        Our method combines convolutional neural networks with attention mechanisms to achieve
        state-of-the-art results on ImageNet dataset. We demonstrate that our approach
        outperforms existing methods by 5% in terms of accuracy. The proposed model consists
        of 50 layers with residual connections. Experiments show significant improvements
        in both training speed and generalization performance. Contact: author@university.edu
        DOI: 10.1234/example.2023
        """

        preprocessor = ClusteringPreprocessor([abstract])
        result = preprocessor.full_preprocessing_pipeline(
            method='tfidf',
            return_intermediate=True
        )

        # Verificar que se procesó correctamente
        self.assertEqual(result['n_documents'], 1)

        # Verificar que se eliminaron URLs, DOIs, emails
        cleaned = result['cleaned_texts'][0]
        self.assertNotIn('author@university.edu', cleaned)
        self.assertNotIn('10.1234', cleaned)

        # Verificar que se normalizaron números
        self.assertIn('token_num', cleaned)

        # Verificar que hay vocabulario técnico
        processed = result['processed_texts'][0]
        self.assertIn('learning', processed.lower())

    def test_multiple_domain_abstracts(self):
        """Test con abstracts de múltiples dominios."""
        abstracts = [
            # Computer Vision
            "Convolutional neural networks for object detection in images using deep learning",

            # NLP
            "Transformer models with self-attention for natural language understanding tasks",

            # Reinforcement Learning
            "Policy gradient methods for training agents in complex environments",

            # Generative Models
            "Generative adversarial networks for high-quality image synthesis"
        ]

        preprocessor = ClusteringPreprocessor(abstracts)
        result = preprocessor.full_preprocessing_pipeline(method='tfidf')

        # Todos los abstracts procesados
        self.assertEqual(result['n_documents'], 4)

        # Matriz tiene dimensiones correctas
        self.assertEqual(result['feature_matrix'].shape[0], 4)

        # Hay features suficientes para diferenciar
        self.assertGreater(result['n_features'], 5)


def run_tests():
    """Ejecutar todos los tests."""
    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Añadir todos los test cases
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringPreprocessorInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepCleaning))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedTokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestStopwordsRemoval))
    suite.addTests(loader.loadTestsFromTestCase(TestLemmatization))
    suite.addTests(loader.loadTestsFromTestCase(TestTFIDFVectorization))
    suite.addTests(loader.loadTestsFromTestCase(TestWord2VecVectorization))
    suite.addTests(loader.loadTestsFromTestCase(TestSBERTVectorization))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorizationErrors))
    suite.addTests(loader.loadTestsFromTestCase(TestFullPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticsAndLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))

    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_tests()
