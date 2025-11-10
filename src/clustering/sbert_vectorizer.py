"""
SBERT Vectorizer Module
Vectorización de textos usando Sentence-BERT (SBERT) con caché de modelos.

SBERT (Sentence-BERT):
=====================
Extensión de BERT que genera embeddings semánticos de oraciones de alta calidad.
Utiliza arquitectura siamesa/triple para aprender representaciones que capturan
similitud semántica.

Ventajas sobre TF-IDF:
- Captura semántica profunda (no solo co-ocurrencia)
- Embeddings densos de dimensionalidad fija
- Mejor para similitud semántica
- Pre-entrenado en grandes corpus

Modelos disponibles:
- all-MiniLM-L6-v2: Rápido, ligero (384 dims)
- all-mpnet-base-v2: Mejor calidad (768 dims)
- paraphrase-multilingual: Soporte multilenguaje

Sistema de caché:
- Guarda embeddings calculados en disco
- Evita recalcular para mismo dataset
- Usa hash del contenido para invalidación
"""

import numpy as np
import pickle
import hashlib
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SBERTVectorizer:
    """
    Vectorizador de textos usando Sentence-BERT con sistema de caché.

    Este vectorizador utiliza modelos pre-entrenados de Sentence-BERT
    para generar embeddings semánticos densos de alta calidad.

    Características:
    - Caché automático de embeddings en disco
    - Progress bars para operaciones largas
    - Batch processing para eficiencia
    - Normalización L2 de embeddings
    - Soporte para múltiples modelos

    Attributes:
        model_name: Nombre del modelo SBERT
        model: Instancia del modelo cargado
        cache_dir: Directorio para caché de embeddings
        use_cache: Si usar sistema de caché
        batch_size: Tamaño de batch para encoding
    """

    # Modelos recomendados
    MODELS = {
        'mini': 'all-MiniLM-L6-v2',  # Rápido, ligero (384 dims)
        'base': 'all-mpnet-base-v2',  # Mejor calidad (768 dims)
        'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2'  # Multilenguaje
    }

    def __init__(self,
                 model_name: str = 'mini',
                 cache_dir: str = 'cache/sbert',
                 use_cache: bool = True,
                 batch_size: int = 32,
                 show_progress: bool = True):
        """
        Inicializa el vectorizador SBERT.

        Args:
            model_name: Nombre del modelo ('mini', 'base', 'multilingual') o nombre completo
            cache_dir: Directorio para guardar caché de embeddings
            use_cache: Si usar caché de embeddings
            batch_size: Tamaño de batch para procesamiento
            show_progress: Si mostrar barra de progreso

        Example:
            >>> vectorizer = SBERTVectorizer(model_name='mini', use_cache=True)
            >>> embeddings = vectorizer.encode(abstracts)
        """
        if not SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Resolver nombre del modelo
        if model_name in self.MODELS:
            self.model_name = self.MODELS[model_name]
        else:
            self.model_name = model_name

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_cache = use_cache
        self.batch_size = batch_size
        self.show_progress = show_progress

        # Cargar modelo
        logger.info("="*70)
        logger.info("SBERT VECTORIZER INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Modelo: {self.model_name}")
        logger.info(f"Caché: {'Habilitado' if use_cache else 'Deshabilitado'}")
        logger.info(f"Batch size: {batch_size}")

        logger.info("Cargando modelo SBERT...")
        self.model = SentenceTransformer(self.model_name)

        # Información del modelo
        embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Dimensión de embeddings: {embedding_dim}")
        logger.info("="*70)

    def encode(self,
               texts: List[str],
               normalize: bool = True) -> np.ndarray:
        """
        Codifica textos a embeddings SBERT.

        Proceso:
        =======
        1. Verifica si hay embeddings en caché
        2. Si no, calcula embeddings usando el modelo
        3. Normaliza embeddings (L2 norm) si se solicita
        4. Guarda en caché para uso futuro
        5. Retorna matriz de embeddings

        Normalización L2:
        ================
        Para embedding e, normalizado e' = e / ||e||_2

        Ventajas:
        - Convierte distancia euclidiana en similitud coseno
        - Mejora rendimiento de clustering
        - Rango acotado [-1, 1] para similitud

        Args:
            texts: Lista de textos a codificar
            normalize: Si normalizar embeddings con L2

        Returns:
            Matriz de embeddings (n_texts, embedding_dim)

        Example:
            >>> embeddings = vectorizer.encode(['text 1', 'text 2'])
            >>> print(embeddings.shape)
            (2, 384)
        """
        logger.info(f"\nCodificando {len(texts)} textos...")

        # Calcular hash del contenido para caché
        cache_key = None
        if self.use_cache:
            cache_key = self._compute_cache_key(texts, normalize)
            cache_path = self.cache_dir / f"{cache_key}.pkl"

            # Verificar caché
            if cache_path.exists():
                logger.info(f"Cargando embeddings desde caché: {cache_path}")
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Embeddings cargados desde caché: {embeddings.shape}")
                return embeddings

        # Codificar con modelo
        logger.info("Calculando embeddings con SBERT...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        logger.info(f"Embeddings calculados: {embeddings.shape}")

        # Guardar en caché
        if self.use_cache and cache_key:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            logger.info(f"Guardando embeddings en caché: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

        return embeddings

    def _compute_cache_key(self, texts: List[str], normalize: bool) -> str:
        """
        Calcula clave de caché basada en contenido y parámetros.

        Usa hash SHA256 de:
        - Nombre del modelo
        - Textos concatenados
        - Flag de normalización

        Returns:
            Hash hexadecimal de 16 caracteres
        """
        # Concatenar todos los textos
        content = '|'.join(texts)

        # Añadir parámetros
        params = f"{self.model_name}|{normalize}"

        # Hash SHA256
        hash_obj = hashlib.sha256((content + params).encode('utf-8'))
        cache_key = hash_obj.hexdigest()[:16]

        return cache_key

    def clear_cache(self) -> None:
        """Limpia todos los archivos de caché."""
        logger.info("Limpiando caché de embeddings...")

        cache_files = list(self.cache_dir.glob("*.pkl"))
        for file_path in cache_files:
            file_path.unlink()

        logger.info(f"{len(cache_files)} archivos de caché eliminados")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el caché.

        Returns:
            Diccionario con estadísticas del caché
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)

        return {
            'n_files': len(cache_files),
            'total_size_mb': total_size_mb,
            'cache_dir': str(self.cache_dir)
        }


def main():
    """Ejemplo de uso del vectorizador SBERT."""

    print("\n" + "="*70)
    print(" EJEMPLO: SBERT Vectorizer")
    print("="*70)

    if not SBERT_AVAILABLE:
        print("\nERROR: sentence-transformers no está instalado")
        print("Instalar con: pip install sentence-transformers")
        return

    # Textos de ejemplo
    texts = [
        "Deep learning and neural networks for image classification",
        "Natural language processing with transformers",
        "Convolutional neural networks for computer vision",
        "Machine learning algorithms and supervised learning",
        "Reinforcement learning and policy optimization"
    ]

    print(f"\nTextos de ejemplo: {len(texts)}")

    # 1. Vectorizador con caché
    print("\n" + "="*70)
    print("1. VECTORIZACIÓN CON CACHÉ")
    print("="*70)

    vectorizer = SBERTVectorizer(
        model_name='mini',
        use_cache=True,
        batch_size=2,
        show_progress=True
    )

    # Primera codificación (calcula)
    print("\nPrimera codificación (calculando)...")
    embeddings1 = vectorizer.encode(texts, normalize=True)
    print(f"Embeddings shape: {embeddings1.shape}")
    print(f"Norma del primer embedding: {np.linalg.norm(embeddings1[0]):.4f}")

    # Segunda codificación (desde caché)
    print("\nSegunda codificación (desde caché)...")
    embeddings2 = vectorizer.encode(texts, normalize=True)
    print(f"Embeddings shape: {embeddings2.shape}")

    # Verificar que son idénticos
    assert np.allclose(embeddings1, embeddings2)
    print("✓ Embeddings desde caché son idénticos")

    # 2. Información de caché
    print("\n" + "="*70)
    print("2. INFORMACIÓN DE CACHÉ")
    print("="*70)

    cache_info = vectorizer.get_cache_info()
    print(f"Archivos en caché: {cache_info['n_files']}")
    print(f"Tamaño total: {cache_info['total_size_mb']:.2f} MB")
    print(f"Directorio: {cache_info['cache_dir']}")

    # 3. Similitud semántica
    print("\n" + "="*70)
    print("3. SIMILITUD SEMÁNTICA")
    print("="*70)

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings1)

    print("\nMatriz de similitud coseno:")
    print(similarities.round(3))

    # Pares más similares
    print("\nPares de textos más similares:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = similarities[i, j]
            if sim > 0.5:  # Threshold
                print(f"\nSimilitud: {sim:.3f}")
                print(f"  [{i}] {texts[i][:50]}...")
                print(f"  [{j}] {texts[j][:50]}...")

    # 4. Limpiar caché
    print("\n" + "="*70)
    print("4. LIMPIEZA DE CACHÉ")
    print("="*70)

    vectorizer.clear_cache()

    print("\n" + "="*70)
    print(" EJEMPLO COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
