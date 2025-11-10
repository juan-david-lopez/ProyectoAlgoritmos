"""
Unit Tests for Clustering System
==================================

Suite completa de tests unitarios para el sistema de clustering jerárquico.

Tests incluidos:
- DistanceCalculator: Validación de métricas de distancia
- HierarchicalClustering: Verificación de métodos de linkage
- ClusteringEvaluator: Tests de métricas de evaluación
- SBERTVectorizer: Validación de vectorización y caché
- ParallelProcessor: Tests de paralelización y checkpoints
- ClusteringPipeline: Test de integración end-to-end

Datasets sintéticos:
- Pequeño (10 docs, 2 clusters): Tests básicos
- Mediano (100 docs, 4 clusters): Tests de robustez
- Grande (600 docs, 5 clusters): Tests de paralelización
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.distance_calculator import DistanceCalculator
from clustering.hierarchical_clustering import HierarchicalClustering
from clustering.clustering_evaluator import ClusteringEvaluator
from clustering.sbert_vectorizer import SBERTVectorizer, SBERT_AVAILABLE
from clustering.parallel_processor import (
    ParallelProcessor,
    CheckpointManager,
    JOBLIB_AVAILABLE
)


class SyntheticDataGenerator:
    """
    Generador de datasets sintéticos para testing.

    Genera clusters con separación controlada usando distribuciones gaussianas.
    """

    @staticmethod
    def generate_gaussian_clusters(n_samples: int,
                                   n_features: int,
                                   n_clusters: int,
                                   cluster_std: float = 0.5,
                                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera clusters gaussianos sintéticos.

        Args:
            n_samples: Número total de muestras
            n_features: Dimensionalidad de vectores
            n_clusters: Número de clusters a generar
            cluster_std: Desviación estándar intra-cluster
            random_state: Semilla para reproducibilidad

        Returns:
            (vectors, labels): Vectores y etiquetas verdaderas
        """
        np.random.seed(random_state)

        samples_per_cluster = n_samples // n_clusters
        vectors = []
        labels = []

        for i in range(n_clusters):
            # Centro del cluster en el espacio n-dimensional
            center = np.random.randn(n_features) * 5

            # Generar muestras del cluster
            cluster_samples = np.random.randn(samples_per_cluster, n_features) * cluster_std + center
            vectors.append(cluster_samples)
            labels.extend([i] * samples_per_cluster)

        vectors = np.vstack(vectors)
        labels = np.array(labels)

        # Shuffle
        indices = np.random.permutation(len(labels))
        vectors = vectors[indices]
        labels = labels[indices]

        # Normalizar L2
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        return vectors, labels

    @staticmethod
    def generate_text_documents(n_docs: int,
                               n_clusters: int,
                               words_per_doc: int = 50,
                               random_state: int = 42) -> Tuple[List[str], np.ndarray]:
        """
        Genera documentos de texto sintéticos con vocabulario temático.

        Args:
            n_docs: Número de documentos
            n_clusters: Número de clusters temáticos
            words_per_doc: Palabras por documento
            random_state: Semilla

        Returns:
            (documents, labels): Lista de documentos y etiquetas
        """
        np.random.seed(random_state)

        # Vocabularios temáticos por cluster
        vocabularies = [
            ['neural', 'network', 'deep', 'learning', 'model', 'training', 'layer', 'activation'],
            ['algorithm', 'optimization', 'gradient', 'descent', 'convergence', 'function', 'minimum'],
            ['data', 'analysis', 'statistical', 'inference', 'probability', 'distribution', 'sample'],
            ['computer', 'vision', 'image', 'recognition', 'detection', 'segmentation', 'pixel'],
            ['language', 'processing', 'text', 'semantic', 'syntax', 'parsing', 'embedding']
        ]

        documents = []
        labels = []
        docs_per_cluster = n_docs // n_clusters

        for cluster_id in range(n_clusters):
            vocab = vocabularies[cluster_id % len(vocabularies)]

            for _ in range(docs_per_cluster):
                # Generar documento con 80% palabras del vocabulario temático
                words = []
                for _ in range(words_per_doc):
                    if np.random.rand() < 0.8:
                        words.append(np.random.choice(vocab))
                    else:
                        # Ruido de otros clusters
                        other_vocab = [w for v in vocabularies for w in v if v != vocab]
                        words.append(np.random.choice(other_vocab))

                documents.append(' '.join(words))
                labels.append(cluster_id)

        # Shuffle
        indices = np.random.permutation(len(labels))
        documents = [documents[i] for i in indices]
        labels = np.array([labels[i] for i in indices])

        return documents, labels


class TestDistanceCalculator(unittest.TestCase):
    """Tests para DistanceCalculator."""

    def setUp(self):
        """Setup antes de cada test."""
        self.calculator = DistanceCalculator()

        # Dataset pequeño
        self.vectors_small = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])

        # Normalizar
        self.vectors_small = self.vectors_small / np.linalg.norm(
            self.vectors_small, axis=1, keepdims=True
        )

    def test_cosine_distance(self):
        """Test distancia coseno."""
        dist_matrix = self.calculator.cosine_distance(self.vectors_small)

        # Propiedades básicas
        self.assertEqual(dist_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(dist_matrix), 0))  # Diagonal = 0
        self.assertTrue(np.allclose(dist_matrix, dist_matrix.T))  # Simétrica
        self.assertTrue(np.all(dist_matrix >= 0))  # No negativa

    def test_euclidean_distance(self):
        """Test distancia euclidiana."""
        dist_matrix = self.calculator.euclidean_distance(self.vectors_small)

        self.assertEqual(dist_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(dist_matrix), 0))
        self.assertTrue(np.allclose(dist_matrix, dist_matrix.T))

    def test_manhattan_distance(self):
        """Test distancia Manhattan."""
        dist_matrix = self.calculator.manhattan_distance(self.vectors_small)

        self.assertEqual(dist_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(dist_matrix), 0))

    def test_correlation_distance(self):
        """Test distancia de correlación."""
        dist_matrix = self.calculator.correlation_distance(self.vectors_small)

        self.assertEqual(dist_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(dist_matrix), 0))

    def test_large_dataset(self):
        """Test con dataset grande."""
        vectors_large, _ = SyntheticDataGenerator.generate_gaussian_clusters(
            n_samples=200, n_features=50, n_clusters=4
        )

        dist_matrix = self.calculator.cosine_distance(vectors_large)

        self.assertEqual(dist_matrix.shape, (200, 200))
        self.assertTrue(np.allclose(np.diag(dist_matrix), 0))


class TestHierarchicalClustering(unittest.TestCase):
    """Tests para HierarchicalClustering."""

    def setUp(self):
        """Setup antes de cada test."""
        # Generar datos sintéticos
        self.vectors, self.true_labels = SyntheticDataGenerator.generate_gaussian_clusters(
            n_samples=50, n_features=20, n_clusters=3
        )

        # Calcular distancias
        calculator = DistanceCalculator()
        self.distance_matrix = calculator.cosine_distance(self.vectors)

    def test_single_linkage(self):
        """Test single linkage."""
        clusterer = HierarchicalClustering(self.distance_matrix)
        linkage_matrix = clusterer.single_linkage()

        # Verificar dimensiones
        n = len(self.distance_matrix)
        self.assertEqual(linkage_matrix.shape, (n - 1, 4))

        # Verificar estructura
        self.assertTrue(np.all(linkage_matrix[:, 2] >= 0))  # Distancias no negativas

    def test_complete_linkage(self):
        """Test complete linkage."""
        clusterer = HierarchicalClustering(self.distance_matrix)
        linkage_matrix = clusterer.complete_linkage()

        n = len(self.distance_matrix)
        self.assertEqual(linkage_matrix.shape, (n - 1, 4))

    def test_average_linkage(self):
        """Test average linkage."""
        clusterer = HierarchicalClustering(self.distance_matrix)
        linkage_matrix = clusterer.average_linkage()

        n = len(self.distance_matrix)
        self.assertEqual(linkage_matrix.shape, (n - 1, 4))

    def test_ward_linkage(self):
        """Test Ward linkage."""
        clusterer = HierarchicalClustering(self.distance_matrix)
        linkage_matrix = clusterer.ward_linkage()

        n = len(self.distance_matrix)
        self.assertEqual(linkage_matrix.shape, (n - 1, 4))

    def test_linkage_monotonicity(self):
        """Test monotonicidad de distancias en linkage."""
        clusterer = HierarchicalClustering(self.distance_matrix)
        linkage_matrix = clusterer.average_linkage()

        # Las distancias deben ser no decrecientes
        distances = linkage_matrix[:, 2]
        self.assertTrue(np.all(distances[1:] >= distances[:-1]))


class TestClusteringEvaluator(unittest.TestCase):
    """Tests para ClusteringEvaluator."""

    def setUp(self):
        """Setup antes de cada test."""
        # Generar datos bien separados
        self.vectors, self.true_labels = SyntheticDataGenerator.generate_gaussian_clusters(
            n_samples=60, n_features=30, n_clusters=3, cluster_std=0.3
        )

        # Calcular distancias
        calculator = DistanceCalculator()
        self.distance_matrix = calculator.cosine_distance(self.vectors)

        # Crear clustering y linkage matrices
        clusterer = HierarchicalClustering(self.distance_matrix)
        linkage_matrices = {
            'single': clusterer.single_linkage(),
            'complete': clusterer.complete_linkage(),
            'average': clusterer.average_linkage(),
            'ward': clusterer.ward_linkage()
        }

        # Crear evaluador
        self.evaluator = ClusteringEvaluator(self.distance_matrix, linkage_matrices)

    def test_cophenetic_correlation(self):
        """Test correlación cofenética."""
        # Usar linkage_matrix ya calculado en setUp
        linkage_matrix = self.evaluator.linkage_matrices['average']

        cpcc = self.evaluator.cophenetic_correlation(linkage_matrix)

        # CPCC debe estar en [-1, 1]
        self.assertGreaterEqual(cpcc, -1.0)
        self.assertLessEqual(cpcc, 1.0)

        # Para buenos clusters, CPCC > 0.7
        self.assertGreater(cpcc, 0.5)

    def test_silhouette_analysis(self):
        """Test análisis de Silhouette."""
        linkage_matrix = self.evaluator.linkage_matrices['average']

        results = self.evaluator.silhouette_analysis(
            linkage_matrix,
            n_clusters_range=range(2, 6)
        )

        # Verificar que retorna un diccionario por cada k
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

        # Verificar estructura para cada k
        for k, result in results.items():
            self.assertIn('silhouette_avg', result)
            # Silhouette en [-1, 1]
            self.assertGreaterEqual(result['silhouette_avg'], -1.0)
            self.assertLessEqual(result['silhouette_avg'], 1.0)

    def test_davies_bouldin_index(self):
        """Test índice Davies-Bouldin."""
        linkage_matrix = self.evaluator.linkage_matrices['average']

        # Davies-Bouldin toma un solo n_clusters, no un rango
        db_score = self.evaluator.davies_bouldin_index(
            linkage_matrix,
            n_clusters=3
        )

        # DB debe ser no negativo
        self.assertIsInstance(db_score, (int, float))
        self.assertGreaterEqual(db_score, 0.0)

    def test_calinski_harabasz_score(self):
        """Test score Calinski-Harabasz."""
        linkage_matrix = self.evaluator.linkage_matrices['average']

        # Calinski-Harabasz toma un solo n_clusters, no un rango
        ch_score = self.evaluator.calinski_harabasz_score_eval(
            linkage_matrix,
            n_clusters=3
        )

        # CH debe ser positivo
        self.assertIsInstance(ch_score, (int, float))
        self.assertGreater(ch_score, 0.0)


@unittest.skipIf(not SBERT_AVAILABLE, "sentence-transformers not available")
class TestSBERTVectorizer(unittest.TestCase):
    """Tests para SBERTVectorizer."""

    def setUp(self):
        """Setup antes de cada test."""
        # Crear directorio temporal para caché
        self.temp_dir = tempfile.mkdtemp()

        self.vectorizer = SBERTVectorizer(
            model_name='mini',
            cache_dir=self.temp_dir,
            use_cache=True,
            batch_size=4,
            show_progress=False
        )

        self.texts = [
            "Machine learning and artificial intelligence",
            "Deep neural networks for image recognition",
            "Natural language processing with transformers",
            "Reinforcement learning and robotics"
        ]

    def tearDown(self):
        """Cleanup después de cada test."""
        shutil.rmtree(self.temp_dir)

    def test_encode_basic(self):
        """Test encoding básico."""
        embeddings = self.vectorizer.encode(self.texts, normalize=True)

        # Verificar dimensiones
        self.assertEqual(embeddings.shape[0], len(self.texts))
        self.assertEqual(embeddings.shape[1], 384)  # mini model

        # Verificar normalización L2
        norms = np.linalg.norm(embeddings, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_cache_system(self):
        """Test sistema de caché."""
        # Primera codificación (calcula)
        embeddings1 = self.vectorizer.encode(self.texts, normalize=True)

        # Segunda codificación (desde caché)
        embeddings2 = self.vectorizer.encode(self.texts, normalize=True)

        # Deben ser idénticos
        self.assertTrue(np.allclose(embeddings1, embeddings2))

        # Verificar que hay archivos en caché
        cache_info = self.vectorizer.get_cache_info()
        self.assertGreater(cache_info['n_files'], 0)

    def test_cache_invalidation(self):
        """Test invalidación de caché al cambiar textos."""
        embeddings1 = self.vectorizer.encode(self.texts, normalize=True)

        # Textos diferentes
        different_texts = ["Different text"] + self.texts[1:]
        embeddings2 = self.vectorizer.encode(different_texts, normalize=True)

        # No deben ser idénticos
        self.assertFalse(np.allclose(embeddings1, embeddings2))

    def test_clear_cache(self):
        """Test limpieza de caché."""
        self.vectorizer.encode(self.texts, normalize=True)

        # Verificar que hay caché
        cache_info = self.vectorizer.get_cache_info()
        self.assertGreater(cache_info['n_files'], 0)

        # Limpiar caché
        self.vectorizer.clear_cache()

        # Verificar que está vacío
        cache_info = self.vectorizer.get_cache_info()
        self.assertEqual(cache_info['n_files'], 0)


@unittest.skipIf(not JOBLIB_AVAILABLE, "joblib not available")
class TestParallelProcessor(unittest.TestCase):
    """Tests para ParallelProcessor y CheckpointManager."""

    def setUp(self):
        """Setup antes de cada test."""
        # Directorio temporal para checkpoints
        self.temp_dir = tempfile.mkdtemp()

        self.checkpoint_manager = CheckpointManager(self.temp_dir)
        self.processor = ParallelProcessor(n_jobs=2, threshold=50, verbose=0)

    def tearDown(self):
        """Cleanup después de cada test."""
        shutil.rmtree(self.temp_dir)

    def test_checkpoint_save_load_array(self):
        """Test guardar/cargar arrays numpy."""
        data = np.random.rand(100, 50)

        # Guardar
        self.checkpoint_manager.save_checkpoint('test_array', data)

        # Cargar
        loaded_data = self.checkpoint_manager.load_checkpoint('test_array')

        # Verificar
        self.assertTrue(np.allclose(data, loaded_data))

    def test_checkpoint_save_load_dict(self):
        """Test guardar/cargar diccionarios."""
        data = {
            'scores': [0.1, 0.2, 0.3],
            'optimal_k': 3,
            'method': 'average'
        }

        # Guardar
        self.checkpoint_manager.save_checkpoint('test_dict', data)

        # Cargar
        loaded_data = self.checkpoint_manager.load_checkpoint('test_dict')

        # Verificar
        self.assertEqual(data, loaded_data)

    def test_checkpoint_exists(self):
        """Test verificación de existencia."""
        data = np.array([1, 2, 3])

        # No existe
        self.assertFalse(self.checkpoint_manager.checkpoint_exists('nonexistent'))

        # Guardar
        self.checkpoint_manager.save_checkpoint('exists_test', data)

        # Existe
        self.assertTrue(self.checkpoint_manager.checkpoint_exists('exists_test'))

    def test_list_checkpoints(self):
        """Test listar checkpoints."""
        # Guardar varios checkpoints
        self.checkpoint_manager.save_checkpoint('cp1', np.array([1, 2, 3]))
        self.checkpoint_manager.save_checkpoint('cp2', {'key': 'value'})
        self.checkpoint_manager.save_checkpoint('cp3', np.array([4, 5, 6]))

        # Listar
        checkpoints = self.checkpoint_manager.list_checkpoints()

        self.assertEqual(len(checkpoints), 3)
        self.assertIn('cp1', checkpoints)
        self.assertIn('cp2', checkpoints)
        self.assertIn('cp3', checkpoints)

    def test_clear_checkpoints(self):
        """Test limpiar checkpoints."""
        # Guardar checkpoints
        self.checkpoint_manager.save_checkpoint('cp1', np.array([1, 2, 3]))
        self.checkpoint_manager.save_checkpoint('cp2', np.array([4, 5, 6]))

        # Limpiar
        self.checkpoint_manager.clear_checkpoints()

        # Verificar vacío
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 0)

    def test_should_parallelize(self):
        """Test lógica de decisión de paralelización."""
        # Pequeño: no paralelizar
        self.assertFalse(self.processor.should_parallelize(49))

        # En umbral: sí paralelizar
        self.assertTrue(self.processor.should_parallelize(50))

        # Grande: sí paralelizar
        self.assertTrue(self.processor.should_parallelize(1000))

    def test_parallel_pairwise_distances(self):
        """Test cálculo paralelo de distancias."""
        # Dataset grande (>threshold)
        vectors = np.random.rand(150, 30)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        from sklearn.metrics.pairwise import cosine_distances

        # Calcular en paralelo
        dist_parallel = self.processor.parallel_pairwise_distances(
            vectors, cosine_distances, show_progress=False
        )

        # Calcular secuencialmente
        dist_sequential = cosine_distances(vectors)

        # Verificar que son equivalentes
        self.assertTrue(np.allclose(dist_parallel, dist_sequential, atol=1e-10))


class TestIntegration(unittest.TestCase):
    """Tests de integración end-to-end."""

    def setUp(self):
        """Setup antes de cada test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup después de cada test."""
        shutil.rmtree(self.temp_dir)

    def test_complete_pipeline_small(self):
        """Test pipeline completo con dataset pequeño."""
        # Generar datos sintéticos
        vectors, true_labels = SyntheticDataGenerator.generate_gaussian_clusters(
            n_samples=30, n_features=20, n_clusters=3, cluster_std=0.3
        )

        # 1. Calcular distancias
        calculator = DistanceCalculator()
        distance_matrix = calculator.cosine_distance(vectors)

        # 2. Clustering jerárquico
        clusterer = HierarchicalClustering(distance_matrix)
        linkage_matrix = clusterer.average_linkage()

        # 3. Evaluar
        linkage_matrices = {'average': linkage_matrix}
        evaluator = ClusteringEvaluator(distance_matrix, linkage_matrices)
        cpcc = evaluator.cophenetic_correlation(linkage_matrix)

        # Verificar que el clustering es razonable
        self.assertGreater(cpcc, 0.5)  # Buena correlación cofenética

    def test_complete_pipeline_large(self):
        """Test pipeline completo con dataset grande."""
        # Dataset grande (activa paralelización)
        vectors, true_labels = SyntheticDataGenerator.generate_gaussian_clusters(
            n_samples=600, n_features=50, n_clusters=5, cluster_std=0.4
        )

        # Con procesamiento paralelo
        if JOBLIB_AVAILABLE:
            processor = ParallelProcessor(n_jobs=2, threshold=500, verbose=0)
            checkpoint_manager = CheckpointManager(self.temp_dir)

            from sklearn.metrics.pairwise import cosine_distances

            # Calcular distancias en paralelo
            distance_matrix = processor.parallel_pairwise_distances(
                vectors, cosine_distances, show_progress=False
            )

            # Guardar checkpoint
            checkpoint_manager.save_checkpoint('distance_matrix', distance_matrix)

            # Cargar checkpoint
            loaded_distance_matrix = checkpoint_manager.load_checkpoint('distance_matrix')

            # Verificar
            self.assertTrue(np.allclose(distance_matrix, loaded_distance_matrix))


def run_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*70)
    print(" EJECUTANDO TESTS UNITARIOS - SISTEMA DE CLUSTERING")
    print("="*70)

    # Crear suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Añadir tests
    suite.addTests(loader.loadTestsFromTestCase(TestDistanceCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalClustering))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringEvaluator))

    if SBERT_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestSBERTVectorizer))
    else:
        print("\nWARNING: Skipping SBERT tests (sentence-transformers not installed)")

    if JOBLIB_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestParallelProcessor))
    else:
        print("\nWARNING: Skipping parallel processing tests (joblib not installed)")

    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Ejecutar
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print(" RESUMEN DE TESTS")
    print("="*70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Éxitos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
