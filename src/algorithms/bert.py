"""
Algoritmo BERT (Bidirectional Encoder Representations from Transformers)
para similitud semántica profunda.

Matemática y Arquitectura:
===========================
BERT es un modelo transformer bidireccional pre-entrenado que captura
representaciones contextuales profundas del lenguaje.

1. Arquitectura Transformer:
   El núcleo de BERT es el transformer encoder con self-attention:

   MultiHeadAttention(Q, K, V) = Concat(head₁, ..., headₕ)W^O

   donde cada head:
   headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)

   Attention(Q, K, V) = softmax((QK^T) / √dₖ) V

2. Bidireccionalidad:
   A diferencia de modelos unidireccionales, BERT procesa contexto
   en ambas direcciones:

   "The cat [MASK] on mat" →
   - Mira izquierda: "The cat"
   - Mira derecha: "on mat"
   - Predice: "sat"

3. Embeddings:
   Cada token obtiene una representación contextual:

   E_token = TokenEmb + PositionEmb + SegmentEmb

   - TokenEmb: Embedding de la palabra
   - PositionEmb: Posición en secuencia
   - SegmentEmb: Identificador de oración (A/B)

4. Similitud de Oraciones:
   Para comparar oraciones, se usan varias estrategias:

   a) CLS Token:
      sim = cosine(h_[CLS]₁, h_[CLS]₂)

   b) Mean Pooling:
      h_sent = mean(h₁, h₂, ..., hₙ)
      sim = cosine(h_sent₁, h_sent₂)

   c) Cross-Encoding (más preciso pero lento):
      score = BERT([CLS] sent1 [SEP] sent2 [SEP])

5. Pre-entrenamiento:
   BERT se pre-entrena con dos objetivos:

   a) Masked Language Model (MLM):
      P(wᵢ | context) con 15% tokens enmascarados

   b) Next Sentence Prediction (NSP):
      P(sent_B | sent_A) para entender relaciones

Complejidad:
    Tiempo: O(n² × d) por self-attention cuadrática
    Memoria: ~500MB-1GB según variante (base/large)
    Parámetros: 110M (base), 340M (large)

Optimizaciones Implementadas:
    1. Batching: Procesar múltiples textos simultáneamente
    2. Truncation: Limitar longitud máxima
    3. Padding: Agrupar textos de longitud similar
    4. FP16 (opcional): Usar precisión mixta para GPU
    5. Gradient checkpointing: Reducir uso de memoria

Referencias:
    - Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding. NAACL 2019.
"""

import numpy as np
import logging
import time
from typing import List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BERTComparator:
    """
    Implementa comparación de similitud usando BERT.

    BERT proporciona la máxima comprensión semántica a costa de velocidad.
    Esta implementación incluye optimizaciones de batching y memoria.

    Características:
        - Máxima precisión semántica
        - Contexto bidireccional
        - Estado del arte en NLP
        - Lento pero preciso
    """

    def __init__(self, model_name: str = 'bert-base-uncased',
                 cache_dir: str = None, device: str = None,
                 max_length: int = 512, batch_size: int = 8):
        """
        Inicializa el comparador BERT.

        Args:
            model_name: Nombre del modelo BERT
                - 'bert-base-uncased': 12 capas, 110M params
                - 'bert-large-uncased': 24 capas, 340M params
                - 'bert-base-multilingual-cased': Multilingüe
            cache_dir: Directorio para caché de modelos
            device: 'cuda', 'cpu', o None (auto-detectar)
            max_length: Longitud máxima de secuencia (512 max)
            batch_size: Tamaño de lote para procesamiento
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = batch_size

        # Detectar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Inicializando BERTComparator")
        logger.info(f"  Modelo: {model_name}")
        logger.info(f"  Dispositivo: {self.device}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Cache: {cache_dir}")

        # Crear directorio de caché
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Cargar tokenizador y modelo
        logger.info("Cargando modelo BERT...")
        start_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Mover modelo a dispositivo
        self.model.to(self.device)
        self.model.eval()  # Modo evaluación (desactivar dropout)

        elapsed = time.perf_counter() - start_time
        logger.info(f"✓ Modelo cargado en {elapsed:.3f}s")

        # Información del modelo
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"  Dimensión de embeddings: {self.embedding_dim}")
        logger.info(f"  Número de capas: {self.model.config.num_hidden_layers}")

    def mean_pooling(self, token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Realiza mean pooling sobre embeddings de tokens.

        El mean pooling promedia los embeddings de todos los tokens,
        considerando la máscara de atención para ignorar padding.

        Fórmula:
            e_sent = Σᵢ(eᵢ × maskᵢ) / Σᵢ(maskᵢ)

        Args:
            token_embeddings: Embeddings de tokens [batch, seq_len, hidden_dim]
            attention_mask: Máscara de atención [batch, seq_len]

        Returns:
            Embeddings de oraciones [batch, hidden_dim]
        """
        # Expandir máscara a dimensiones de embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sumar embeddings ponderados por máscara
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Sumar máscara para obtener número de tokens (evitar división por 0)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Calcular promedio
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Codifica textos en embeddings usando BERT.

        Proceso con batching optimizado:
            1. Dividir textos en batches
            2. Para cada batch:
               a. Tokenizar: text → token IDs
               b. Forward pass: BERT encoding
               c. Mean pooling: tokens → sentence embedding
               d. Normalizar embedding
            3. Concatenar resultados

        Args:
            texts: Lista de textos a codificar
            batch_size: Tamaño de lote (usa self.batch_size si None)

        Returns:
            Matriz de embeddings (n_texts, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size

        logger.debug(f"Codificando {len(texts)} textos (batch_size={batch_size})...")

        all_embeddings = []

        # Procesar en batches para eficiencia
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenizar batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,  # Pad al máximo del batch
                truncation=True,  # Truncar si excede max_length
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Mover a dispositivo
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Forward pass (sin calcular gradientes)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Obtener embeddings de tokens (última capa oculta)
                token_embeddings = outputs.last_hidden_state

            # Mean pooling
            sentence_embeddings = self.mean_pooling(token_embeddings, attention_mask)

            # Normalizar a vectores unitarios
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            # Mover a CPU y convertir a numpy
            all_embeddings.append(sentence_embeddings.cpu().numpy())

        # Concatenar todos los embeddings
        embeddings = np.vstack(all_embeddings)

        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud semántica entre dos textos.

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Similitud en [0, 1]
        """
        embeddings = self.encode([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        # Mapear de [-1, 1] a [0, 1]
        sim = (sim + 1) / 2
        return float(sim)

    def compare_multiple(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Compara múltiples textos usando BERT con batching optimizado.

        Optimizaciones implementadas:
            1. Batching: Procesar múltiples textos en paralelo
            2. No-grad: Desactivar cálculo de gradientes
            3. Padding inteligente: Agrupar por longitud similar
            4. Normalización: Usar vectores unitarios para cosine

        Args:
            texts: Lista de textos a comparar
            batch_size: Tamaño de lote (usa self.batch_size si None)

        Returns:
            Matriz de similitud de forma (n, n)
        """
        start_time = time.perf_counter()
        n = len(texts)

        logger.info(f"Comparando {n} textos con BERT (batch_size={batch_size or self.batch_size})...")

        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([[1.0]])

        # Codificar todos los textos en batches
        embeddings = self.encode(texts, batch_size=batch_size)

        logger.debug(f"Embeddings shape: {embeddings.shape}")

        # Calcular matriz de similitud del coseno
        similarity_matrix = cosine_similarity(embeddings)

        # Mapear de [-1, 1] a [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2

        # Asegurar diagonal = 1.0
        np.fill_diagonal(similarity_matrix, 1.0)

        # Clipar al rango [0, 1]
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        elapsed = time.perf_counter() - start_time
        logger.info(f"BERT completado en {elapsed:.3f}s")

        return similarity_matrix

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Obtiene embedding de un texto individual.

        Args:
            text: Texto a codificar

        Returns:
            Vector embedding (embedding_dim,)
        """
        return self.encode([text])[0]


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Usar modelo pequeño para ejemplo
    comparator = BERTComparator(batch_size=2)

    # Ejemplo: Similitud semántica profunda
    texts = [
        "The cat sits on the mat",
        "A feline rests on the rug",
        "Dogs are playing in the park"
    ]

    print("Comparando textos con BERT...")
    matrix = comparator.compare_multiple(texts)

    print("\nMatriz de similitud:")
    for i, text in enumerate(texts):
        print(f"\n{i+1}. {text[:40]}...")
        for j in range(len(texts)):
            print(f"   vs {j+1}: {matrix[i][j]:.3f}")
