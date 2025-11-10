"""
Parallel Processor Module
Sistema de paralelización y checkpoints para clustering de datasets grandes.

Este módulo proporciona herramientas para procesar datasets grandes de forma
eficiente mediante:
- Paralelización de cálculos costosos (>500 documentos)
- Sistema de checkpoints para recuperación de fallos
- Progress bars para operaciones largas
- Procesamiento por batches

Paralelización:
==============
Usa joblib para paralelizar operaciones independientes:
- Cálculo de distancias por pares
- Evaluación de métricas de clustering
- Cálculo de Silhouette por muestra

Sistema de Checkpoints:
======================
Guarda estados intermedios:
- Matrices de distancia
- Matrices TF-IDF/SBERT
- Matrices de linkage
- Resultados de evaluación

Permite reanudar procesamiento sin recalcular todo.
"""

import numpy as np
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from tqdm import tqdm
import time
import json
from datetime import datetime

try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning("joblib not available. Install with: pip install joblib")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Gestor de checkpoints para guardar y recuperar estados intermedios.

    Los checkpoints son útiles para:
    - Evitar recalcular operaciones costosas si falla el proceso
    - Permitir iteración incremental
    - Debugging y análisis de pasos intermedios

    Estructura de checkpoint:
    ========================
    checkpoint_dir/
        metadata.json          # Información del checkpoint
        distance_matrix.npy    # Matriz de distancias
        vectors.npy           # Vectores TF-IDF/SBERT
        linkage_single.npy    # Linkage matrix (single)
        linkage_complete.npy  # Linkage matrix (complete)
        linkage_average.npy   # Linkage matrix (average)
        evaluation.pkl        # Resultados de evaluación
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """
        Inicializa el gestor de checkpoints.

        Args:
            checkpoint_dir: Directorio para guardar checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.checkpoint_dir / 'metadata.json'

        logger.info(f"CheckpointManager inicializado: {checkpoint_dir}")

    def save_checkpoint(self,
                       name: str,
                       data: Any,
                       metadata: Optional[Dict] = None) -> None:
        """
        Guarda un checkpoint con metadata.

        Args:
            name: Nombre del checkpoint (ej: 'distance_matrix', 'vectors')
            data: Datos a guardar (numpy array, dict, etc.)
            metadata: Metadata adicional del checkpoint

        Example:
            >>> manager.save_checkpoint('distance_matrix', dist_matrix, {'metric': 'cosine'})
        """
        logger.info(f"Guardando checkpoint: {name}")

        # Determinar formato según tipo de dato
        if isinstance(data, np.ndarray):
            # NumPy array -> .npy
            file_path = self.checkpoint_dir / f"{name}.npy"
            np.save(file_path, data)

        elif isinstance(data, (dict, list)):
            # Dict/List -> pickle
            file_path = self.checkpoint_dir / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # Otros -> pickle
            file_path = self.checkpoint_dir / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Actualizar metadata
        self._update_metadata(name, metadata)

        logger.info(f"Checkpoint guardado: {file_path}")

    def load_checkpoint(self, name: str) -> Optional[Any]:
        """
        Carga un checkpoint.

        Args:
            name: Nombre del checkpoint

        Returns:
            Datos del checkpoint o None si no existe

        Example:
            >>> dist_matrix = manager.load_checkpoint('distance_matrix')
        """
        # Buscar archivo .npy primero
        npy_path = self.checkpoint_dir / f"{name}.npy"
        if npy_path.exists():
            logger.info(f"Cargando checkpoint: {npy_path}")
            return np.load(npy_path)

        # Buscar archivo .pkl
        pkl_path = self.checkpoint_dir / f"{name}.pkl"
        if pkl_path.exists():
            logger.info(f"Cargando checkpoint: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

        logger.warning(f"Checkpoint no encontrado: {name}")
        return None

    def checkpoint_exists(self, name: str) -> bool:
        """Verifica si existe un checkpoint."""
        npy_path = self.checkpoint_dir / f"{name}.npy"
        pkl_path = self.checkpoint_dir / f"{name}.pkl"
        return npy_path.exists() or pkl_path.exists()

    def list_checkpoints(self) -> List[str]:
        """Lista todos los checkpoints disponibles."""
        checkpoints = []

        for file_path in self.checkpoint_dir.iterdir():
            if file_path.suffix in ['.npy', '.pkl']:
                checkpoints.append(file_path.stem)

        return sorted(checkpoints)

    def clear_checkpoints(self) -> None:
        """Elimina todos los checkpoints."""
        logger.info("Limpiando checkpoints...")

        for file_path in self.checkpoint_dir.iterdir():
            if file_path.suffix in ['.npy', '.pkl', '.json']:
                file_path.unlink()

        logger.info("Checkpoints eliminados")

    def _update_metadata(self, name: str, metadata: Optional[Dict]) -> None:
        """Actualiza metadata del checkpoint."""
        # Cargar metadata existente
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}

        # Actualizar metadata del checkpoint
        checkpoint_metadata = {
            'timestamp': datetime.now().isoformat(),
            'size_bytes': self._get_checkpoint_size(name)
        }

        if metadata:
            checkpoint_metadata.update(metadata)

        all_metadata[name] = checkpoint_metadata

        # Guardar metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    def _get_checkpoint_size(self, name: str) -> int:
        """Obtiene tamaño del checkpoint en bytes."""
        npy_path = self.checkpoint_dir / f"{name}.npy"
        pkl_path = self.checkpoint_dir / f"{name}.pkl"

        if npy_path.exists():
            return npy_path.stat().st_size
        elif pkl_path.exists():
            return pkl_path.stat().st_size
        return 0


class ParallelProcessor:
    """
    Procesador paralelo para operaciones de clustering en datasets grandes.

    Paraleliza operaciones costosas cuando el dataset supera un umbral
    (por defecto 500 documentos).

    Operaciones paralelizables:
    ==========================
    1. Cálculo de distancias por pares
    2. Evaluación de métricas (Silhouette, Davies-Bouldin)
    3. Validación cruzada
    4. Bootstrapping

    Usa joblib con backend 'loky' para paralelización robusta.
    """

    def __init__(self,
                 n_jobs: int = -1,
                 threshold: int = 500,
                 verbose: int = 1):
        """
        Inicializa el procesador paralelo.

        Args:
            n_jobs: Número de workers (-1 usa todos los CPUs)
            threshold: Umbral de documentos para activar paralelización
            verbose: Nivel de verbosidad (0=silencioso, 10=muy verbose)

        Example:
            >>> processor = ParallelProcessor(n_jobs=-1, threshold=500)
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError(
                "joblib not installed. "
                "Install with: pip install joblib"
            )

        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.threshold = threshold
        self.verbose = verbose

        logger.info("="*70)
        logger.info("PARALLEL PROCESSOR INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Workers: {self.n_jobs}")
        logger.info(f"Umbral paralelización: {threshold} documentos")
        logger.info("="*70)

    def should_parallelize(self, n_samples: int) -> bool:
        """
        Determina si debe usar paralelización.

        Args:
            n_samples: Número de muestras

        Returns:
            True si debe paralelizar
        """
        return n_samples >= self.threshold

    def parallel_pairwise_distances(self,
                                   vectors: np.ndarray,
                                   metric: Callable,
                                   show_progress: bool = True) -> np.ndarray:
        """
        Calcula distancias por pares en paralelo.

        Divide la matriz de distancias en bloques y procesa cada bloque
        en un worker diferente.

        Estrategia:
        ==========
        Para matriz de distancia n×n:
        1. Dividir en bloques de filas
        2. Cada worker calcula distancias para su bloque
        3. Combinar resultados

        Args:
            vectors: Matriz de vectores (n_samples, n_features)
            metric: Función de distancia (ej: cosine_distances)
            show_progress: Si mostrar barra de progreso

        Returns:
            Matriz de distancias (n_samples, n_samples)
        """
        n_samples = vectors.shape[0]

        logger.info(f"Calculando distancias en paralelo ({self.n_jobs} workers)...")

        if not self.should_parallelize(n_samples):
            logger.info("Dataset pequeño, usando procesamiento secuencial")
            return metric(vectors)

        # Dividir en chunks
        chunk_size = max(1, n_samples // self.n_jobs)
        chunks = []

        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunks.append((i, end))

        # Función para procesar un chunk
        def process_chunk(start, end):
            return metric(vectors[start:end], vectors)

        # Procesar chunks en paralelo
        if show_progress:
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(process_chunk)(start, end)
                for start, end in tqdm(chunks, desc="Calculando distancias")
            )
        else:
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(process_chunk)(start, end)
                for start, end in chunks
            )

        # Combinar resultados
        distance_matrix = np.vstack(results)

        logger.info(f"Matriz de distancias calculada: {distance_matrix.shape}")

        return distance_matrix

    def parallel_silhouette_samples(self,
                                   distance_matrix: np.ndarray,
                                   labels: np.ndarray,
                                   show_progress: bool = True) -> np.ndarray:
        """
        Calcula Silhouette score por muestra en paralelo.

        Silhouette para muestra i:
        ==========================
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

        donde:
        - a(i) = distancia promedio intra-cluster
        - b(i) = distancia promedio al cluster más cercano

        Args:
            distance_matrix: Matriz de distancias precomputada
            labels: Etiquetas de cluster
            show_progress: Si mostrar progreso

        Returns:
            Array de scores Silhouette por muestra
        """
        from sklearn.metrics import silhouette_samples

        n_samples = len(labels)

        if not self.should_parallelize(n_samples):
            logger.info("Calculando Silhouette (secuencial)...")
            return silhouette_samples(distance_matrix, labels, metric='precomputed')

        logger.info(f"Calculando Silhouette en paralelo ({self.n_jobs} workers)...")

        # sklearn ya paraleliza internamente, pero podemos dividir por clusters
        unique_labels = np.unique(labels)

        def compute_for_cluster(cluster_label):
            mask = labels == cluster_label
            indices = np.where(mask)[0]

            # Silhouette para muestras de este cluster
            scores = silhouette_samples(
                distance_matrix[np.ix_(indices, np.arange(n_samples))],
                labels,
                metric='precomputed'
            )
            return indices, scores

        # Procesar en paralelo
        if show_progress:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_for_cluster)(label)
                for label in tqdm(unique_labels, desc="Calculando Silhouette")
            )
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_for_cluster)(label)
                for label in unique_labels
            )

        # Combinar resultados
        silhouette_scores = np.zeros(n_samples)
        for indices, scores in results:
            silhouette_scores[indices] = scores

        return silhouette_scores


def main():
    """Ejemplo de uso del sistema de checkpoints y paralelización."""

    print("\n" + "="*70)
    print(" EJEMPLO: Parallel Processor & Checkpoints")
    print("="*70)

    # 1. Sistema de Checkpoints
    print("\n" + "="*70)
    print("1. SISTEMA DE CHECKPOINTS")
    print("="*70)

    manager = CheckpointManager('checkpoints_example')

    # Simular datos
    distance_matrix = np.random.rand(100, 100)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Simétrica
    np.fill_diagonal(distance_matrix, 0)

    vectors = np.random.rand(100, 50)

    # Guardar checkpoints
    print("\nGuardando checkpoints...")
    manager.save_checkpoint('distance_matrix', distance_matrix, {'metric': 'cosine'})
    manager.save_checkpoint('vectors', vectors, {'method': 'tfidf'})

    # Listar checkpoints
    print("\nCheckpoints disponibles:")
    for cp in manager.list_checkpoints():
        print(f"  - {cp}")

    # Cargar checkpoint
    print("\nCargando checkpoint...")
    loaded_matrix = manager.load_checkpoint('distance_matrix')
    print(f"Matriz cargada: {loaded_matrix.shape}")
    assert np.allclose(distance_matrix, loaded_matrix)
    print("✓ Checkpoint cargado correctamente")

    # 2. Procesamiento Paralelo
    if JOBLIB_AVAILABLE:
        print("\n" + "="*70)
        print("2. PROCESAMIENTO PARALELO")
        print("="*70)

        processor = ParallelProcessor(n_jobs=2, threshold=50)

        # Simular cálculo de distancias
        print("\nCalculando distancias en paralelo...")
        vectors_large = np.random.rand(200, 50)

        from sklearn.metrics.pairwise import cosine_distances

        start_time = time.time()
        dist_parallel = processor.parallel_pairwise_distances(
            vectors_large,
            cosine_distances,
            show_progress=True
        )
        parallel_time = time.time() - start_time

        print(f"\nTiempo (paralelo): {parallel_time:.2f} segundos")
        print(f"Matriz resultante: {dist_parallel.shape}")

    # 3. Limpiar
    print("\n" + "="*70)
    print("3. LIMPIEZA")
    print("="*70)

    manager.clear_checkpoints()
    print("Checkpoints limpiados")

    print("\n" + "="*70)
    print(" EJEMPLO COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
