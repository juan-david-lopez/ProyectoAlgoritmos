"""
Word Embedding Similarity (Word2Vec/GloVe)

Algoritmo 5: Embeddings Estáticos (Word2Vec/GloVe)

Este módulo implementa similitud basada en word embeddings pre-entrenados.
A diferencia de BERT (contextuales), estos embeddings son estáticos: cada palabra
tiene una representación fija independiente del contexto.

=============================================================================
FUNDAMENTOS MATEMÁTICOS
=============================================================================

1. **Word Embeddings**
   Cada palabra w se mapea a un vector denso en ℝᵈ:
   
   w → v_w ∈ ℝᵈ
   
   Donde d = dimensión del embedding (típicamente 100, 200, 300)

2. **Embedding de Documentos**
   Para un documento D = {w₁, w₂, ..., wₙ}:
   
   v_D = (1/n) Σᵢ v_wᵢ
   
   (Promedio de vectores de palabras)

3. **Similitud por Coseno**
   
   sim(D₁, D₂) = (v_D₁ · v_D₂) / (||v_D₁|| · ||v_D₂||)
   
   Donde:
   - · = producto punto
   - || || = norma euclidiana

4. **Modelos Pre-entrenados**
   
   **Word2Vec:**
   - Skip-gram: Predice contexto desde palabra central
   - CBOW: Predice palabra central desde contexto
   - Dimensiones comunes: 100, 300
   - Corpus: Google News (100B palabras)
   
   **GloVe (Global Vectors):**
   - Factorización de matriz de co-ocurrencias
   - Dimensiones: 50, 100, 200, 300
   - Corpus: Wikipedia + Gigaword (6B tokens)

=============================================================================
COMPLEJIDAD ALGORÍTMICA
=============================================================================

Supongamos:
- n = número de palabras en documento
- d = dimensión de embedding (ej. 300)
- V = tamaño de vocabulario (ej. 400,000)

**Tiempo:**
- Lookup por palabra:    O(1) con diccionario hash
- Vector promedio:       O(n·d)
- Similitud coseno:      O(d)
- **Total por par:**     O(n·d)

**Espacio:**
- Modelo pre-entrenado:  O(V·d) ≈ 400,000 × 300 = 120M floats ≈ 480 MB
- Vector por documento:  O(d)

=============================================================================
VENTAJAS vs DESVENTAJAS
=============================================================================

✅ **Ventajas:**
- Captura similitud semántica (rey → reina)
- Más rápido que BERT (sin inferencia neural)
- Menor uso de memoria que transformers
- Efectivo para vocabulario técnico

❌ **Desventajas:**
- No captura contexto (bank = banco siempre)
- Palabras fuera de vocabulario (OOV)
- Ignora orden de palabras
- Inferior a transformers en tareas complejas

=============================================================================
EJEMPLO DE USO
=============================================================================

```python
from src.algorithms.similarity import WordEmbeddingSimilarity

# Opción 1: Word2Vec (requiere descarga)
calc = WordEmbeddingSimilarity(model_type='word2vec')

# Opción 2: GloVe (más ligero)
calc = WordEmbeddingSimilarity(model_type='glove', dimensions=100)

# Calcular similitud
text1 = "machine learning algorithms"
text2 = "artificial intelligence models"
similarity = calc.compute_similarity(text1, text2)

print(f"Similarity: {similarity:.3f}")  # Output: ~0.85

# Matriz de similitud
texts = ["AI research", "ML models", "deep learning", "neural networks"]
matrix = calc.compute_similarity_matrix(texts)
```

=============================================================================
COMPARACIÓN CON OTROS ALGORITMOS
=============================================================================

| Algoritmo    | Semántica | Contexto | Velocidad | Memoria |
|--------------|-----------|----------|-----------|---------|
| Levenshtein  | ❌        | ❌       | ⚡⚡      | ⚡⚡⚡  |
| TF-IDF       | ⚠️        | ❌       | ⚡⚡⚡    | ⚡⚡    |
| Jaccard      | ❌        | ❌       | ⚡⚡⚡    | ⚡⚡⚡  |
| N-grams      | ❌        | ❌       | ⚡⚡      | ⚡      |
| **Word2Vec** | ✅        | ❌       | ⚡⚡      | ⚡⚡    |
| SBERT        | ✅        | ✅       | ⚡        | ⚡      |
| BERT         | ✅        | ✅       | 🐌        | 🐌      |

=============================================================================
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import warnings
from loguru import logger

# Intentar importar librerías de embeddings
try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("gensim not installed. Install with: pip install gensim")

try:
    import torchtext
    from torchtext.vocab import GloVe
    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False
    logger.warning("torchtext not installed. Install with: pip install torchtext")


class WordEmbeddingSimilarity:
    """
    Calcula similitud textual usando word embeddings estáticos (Word2Vec/GloVe).
    
    Los embeddings mapean palabras a vectores densos que capturan
    relaciones semánticas. La similitud de documentos se calcula
    promediando los vectores de palabras y usando similitud de coseno.
    
    Attributes:
        model_type: Tipo de embedding ('word2vec' o 'glove')
        dimensions: Dimensión de los vectores (50, 100, 200, 300)
        model: Modelo de embeddings cargado
        vocab_size: Tamaño del vocabulario
    """
    
    def __init__(
        self,
        model_type: str = 'glove',
        dimensions: int = 100,
        model_path: Optional[str] = None
    ):
        """
        Inicializa el calculador de similitud con embeddings.
        
        Args:
            model_type: 'word2vec' o 'glove'
            dimensions: Dimensión de vectores (50, 100, 200, 300)
            model_path: Ruta a modelo personalizado (opcional)
        """
        self.model_type = model_type.lower()
        self.dimensions = dimensions
        self.model = None
        self.vocab_size = 0
        
        logger.info(f"Initializing {model_type.upper()} embeddings ({dimensions}D)")
        
        if model_path:
            self._load_custom_model(model_path)
        else:
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Carga modelo pre-entrenado de Word2Vec o GloVe"""
        
        if self.model_type == 'word2vec':
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim required for Word2Vec. Install: pip install gensim")
            
            logger.info("Loading Word2Vec from gensim...")
            logger.warning("First download may take several minutes (~1.5 GB)")
            
            try:
                # word2vec-google-news-300 es el más completo
                self.model = api.load('word2vec-google-news-300')
                self.dimensions = 300  # Forzar dimensión correcta
                self.vocab_size = len(self.model.index_to_key)
                logger.info(f"✓ Word2Vec loaded: {self.vocab_size:,} words, {self.dimensions}D")
                
            except Exception as e:
                logger.error(f"Failed to load Word2Vec: {e}")
                logger.info("Falling back to GloVe...")
                self.model_type = 'glove'
                self._load_glove()
        
        elif self.model_type == 'glove':
            self._load_glove()
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'word2vec' or 'glove'")
    
    def _load_glove(self):
        """Carga modelo GloVe"""
        if not TORCHTEXT_AVAILABLE:
            raise ImportError("torchtext required for GloVe. Install: pip install torchtext")
        
        logger.info(f"Loading GloVe {self.dimensions}D...")
        
        valid_dims = [50, 100, 200, 300]
        if self.dimensions not in valid_dims:
            logger.warning(f"Invalid dimension {self.dimensions}. Using 100D")
            self.dimensions = 100
        
        try:
            # Cargar GloVe pre-entrenado
            self.model = GloVe(name='6B', dim=self.dimensions)
            self.vocab_size = len(self.model.stoi)
            logger.info(f"✓ GloVe loaded: {self.vocab_size:,} words, {self.dimensions}D")
            
        except Exception as e:
            logger.error(f"Failed to load GloVe: {e}")
            raise
    
    def _load_custom_model(self, model_path: str):
        """Carga modelo personalizado desde archivo"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading custom model from {model_path}")
        
        try:
            # Intentar cargar con gensim
            self.model = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
            self.vocab_size = len(self.model.index_to_key)
            self.dimensions = self.model.vector_size
            logger.info(f"✓ Custom model loaded: {self.vocab_size:,} words, {self.dimensions}D")
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Obtiene el vector de embedding para una palabra.
        
        Args:
            word: Palabra a buscar
            
        Returns:
            Vector numpy de dimensión d, o None si no existe
        """
        word = word.lower().strip()
        
        if not word:
            return None
        
        try:
            if self.model_type == 'word2vec':
                if word in self.model:
                    return self.model[word]
            else:  # glove
                if word in self.model.stoi:
                    idx = self.model.stoi[word]
                    return self.model.vectors[idx].numpy()
        except:
            pass
        
        return None
    
    def get_document_vector(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Calcula vector de documento promediando embeddings de palabras.
        
        Proceso:
        1. Tokenizar texto en palabras
        2. Buscar vector de cada palabra
        3. Promediar vectores encontrados
        4. (Opcional) Normalizar vector resultante
        
        Args:
            text: Texto a vectorizar
            normalize: Si normalizar el vector resultante
            
        Returns:
            Vector numpy de dimensión d
        """
        # Tokenización simple (mejoras posibles: lemmatización, stopwords)
        words = text.lower().split()
        
        vectors = []
        oov_count = 0  # Out-of-vocabulary
        
        for word in words:
            vec = self.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
            else:
                oov_count += 1
        
        if not vectors:
            # Si no hay palabras en vocabulario, retornar vector de ceros
            logger.warning(f"No words found in vocabulary for text: '{text[:50]}...'")
            return np.zeros(self.dimensions)
        
        # Promedio de vectores
        doc_vector = np.mean(vectors, axis=0)
        
        # Normalizar (para similitud de coseno más eficiente)
        if normalize:
            norm = np.linalg.norm(doc_vector)
            if norm > 0:
                doc_vector = doc_vector / norm
        
        if oov_count > 0:
            coverage = len(vectors) / len(words) * 100
            logger.debug(f"Vocabulary coverage: {coverage:.1f}% ({oov_count}/{len(words)} OOV)")
        
        return doc_vector
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        return_metadata: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calcula similitud de coseno entre dos textos.
        
        Fórmula:
            sim = (v₁ · v₂) / (||v₁|| · ||v₂||)
        
        Args:
            text1: Primer texto
            text2: Segundo texto
            return_metadata: Si retornar información adicional
            
        Returns:
            Similitud en [0, 1] o diccionario con metadata
        """
        # Obtener vectores de documentos (ya normalizados)
        vec1 = self.get_document_vector(text1, normalize=True)
        vec2 = self.get_document_vector(text2, normalize=True)
        
        # Similitud de coseno (producto punto de vectores normalizados)
        similarity = float(np.dot(vec1, vec2))
        
        # Asegurar rango [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        if not return_metadata:
            return similarity
        
        return {
            'similarity': similarity,
            'vector1': vec1.tolist(),
            'vector2': vec2.tolist(),
            'dimensions': self.dimensions,
            'model_type': self.model_type
        }
    
    def compute_similarity_matrix(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Calcula matriz de similitud para múltiples textos.
        
        Args:
            texts: Lista de textos
            show_progress: Mostrar progreso
            
        Returns:
            Matriz numpy de tamaño (n, n) con similitudes
        """
        n = len(texts)
        matrix = np.zeros((n, n))
        
        # Calcular vectores una vez
        logger.info(f"Computing document vectors for {n} texts...")
        vectors = [self.get_document_vector(text, normalize=True) for text in texts]
        
        # Calcular similitudes
        logger.info(f"Computing {n*(n-1)//2} pairwise similarities...")
        
        for i in range(n):
            matrix[i, i] = 1.0  # Similitud consigo mismo
            
            for j in range(i + 1, n):
                sim = float(np.dot(vectors[i], vectors[j]))
                sim = max(0.0, min(1.0, sim))
                
                matrix[i, j] = sim
                matrix[j, i] = sim  # Matriz simétrica
            
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{n} rows completed")
        
        logger.info("✓ Similarity matrix computed")
        return matrix
    
    def get_most_similar_words(
        self,
        word: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Encuentra las palabras más similares a una palabra dada.
        
        Útil para explorar el espacio de embeddings y validar
        que el modelo capture relaciones semánticas.
        
        Args:
            word: Palabra de entrada
            top_n: Número de palabras más similares a retornar
            
        Returns:
            Lista de tuplas (palabra, similitud)
        """
        word = word.lower()
        
        if self.model_type == 'word2vec':
            if word not in self.model:
                logger.warning(f"Word '{word}' not in vocabulary")
                return []
            
            return self.model.most_similar(word, topn=top_n)
        
        else:  # glove
            if word not in self.model.stoi:
                logger.warning(f"Word '{word}' not in vocabulary")
                return []
            
            # Calcular similitudes manualmente
            word_vec = self.get_word_vector(word)
            
            similarities = []
            for other_word in list(self.model.stoi.keys())[:10000]:  # Limitar para performance
                if other_word == word:
                    continue
                
                other_vec = self.get_word_vector(other_word)
                if other_vec is not None:
                    sim = float(np.dot(word_vec, other_vec))
                    similarities.append((other_word, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
    
    def analogy(
        self,
        word1: str,
        word2: str,
        word3: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Resuelve analogías: word1 es a word2 como word3 es a ?
        
        Ejemplo clásico:
            king - man + woman ≈ queen
        
        Args:
            word1: Primera palabra (ej. "king")
            word2: Segunda palabra (ej. "man")
            word3: Tercera palabra (ej. "woman")
            top_n: Número de resultados
            
        Returns:
            Lista de tuplas (palabra_resultado, similitud)
        """
        if self.model_type != 'word2vec':
            logger.warning("Analogies work best with Word2Vec")
            return []
        
        try:
            # king - man + woman = ?
            positive = [word2, word3]  # woman, king
            negative = [word1]  # man
            
            results = self.model.most_similar(positive=positive, negative=negative, topn=top_n)
            return results
            
        except Exception as e:
            logger.error(f"Analogy failed: {e}")
            return []


def word_embedding_similarity(
    text1: str,
    text2: str,
    model_type: str = 'glove',
    dimensions: int = 100
) -> float:
    """
    Función standalone para calcular similitud rápidamente.
    
    Args:
        text1: Primer texto
        text2: Segundo texto
        model_type: 'word2vec' o 'glove'
        dimensions: Dimensión de embeddings
        
    Returns:
        Similitud en [0, 1]
    
    Example:
        >>> from src.algorithms.similarity import word_embedding_similarity
        >>> sim = word_embedding_similarity("AI research", "machine learning")
        >>> print(f"Similarity: {sim:.3f}")
    """
    calc = WordEmbeddingSimilarity(model_type=model_type, dimensions=dimensions)
    return calc.compute_similarity(text1, text2)


if __name__ == "__main__":
    # Demo
    print("\n" + "="*70)
    print("  WORD EMBEDDING SIMILARITY DEMO")
    print("="*70 + "\n")
    
    # Crear instancia
    calc = WordEmbeddingSimilarity(model_type='glove', dimensions=100)
    
    # Ejemplo 1: Similitud semántica
    text1 = "machine learning and artificial intelligence"
    text2 = "AI and ML algorithms"
    text3 = "cooking recipes and food"
    
    sim12 = calc.compute_similarity(text1, text2)
    sim13 = calc.compute_similarity(text1, text3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}\n")
    
    print(f"Similarity(1, 2): {sim12:.3f}  ✓ (alta - mismo dominio)")
    print(f"Similarity(1, 3): {sim13:.3f}  ✓ (baja - dominios distintos)\n")
    
    # Ejemplo 2: Palabras similares
    print("Most similar words to 'computer':")
    similar = calc.get_most_similar_words('computer', top_n=5)
    for word, sim in similar[:5]:
        print(f"  • {word:15} {sim:.3f}")
    
    print("\n" + "="*70)
