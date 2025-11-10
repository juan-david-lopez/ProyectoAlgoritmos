"""
BERT Transformer Similarity Algorithm

MATHEMATICAL EXPLANATION:
========================

BERT (Bidirectional Encoder Representations from Transformers) is a deep learning model
that produces contextualized word embeddings. Unlike SBERT which is optimized for sentence
similarity, this implementation uses raw BERT with mean pooling for similarity computation.

BERT ARCHITECTURE:
------------------

BERT consists of stacked Transformer encoder layers:

Input: Token sequence [CLS] t₁ t₂ ... tₙ [SEP]

1. **Embedding Layer**:
   - Token embeddings (vocabulary-based)
   - Position embeddings (positional encoding)
   - Segment embeddings (for sentence pairs)

   E = TokenEmb + PositionEmb + SegmentEmb

2. **Transformer Layers** (12 or 24 layers):
   Each layer applies:

   a) Multi-Head Self-Attention:
      For each head h:

      Q = XWᵠʰ    (Query)
      K = XWᵏʰ    (Key)
      V = XWᵛʰ    (Value)

      Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V

      where:
      - dₖ = dimension of key vectors
      - √dₖ = scaling factor to prevent vanishing gradients

   b) Multi-Head Concatenation:
      MultiHead(X) = Concat(head₁, ..., headₕ)W⁰

   c) Add & Norm (Residual Connection + Layer Normalization):
      X' = LayerNorm(X + MultiHead(X))

   d) Feed-Forward Network:
      FFN(X') = max(0, X'W₁ + b₁)W₂ + b₂

   e) Add & Norm:
      Output = LayerNorm(X' + FFN(X'))

3. **Output**:
   Contextual embeddings: H = [h₁, h₂, ..., hₙ]
   where each hᵢ ∈ ℝ⁷⁶⁸ (for BERT-base)

ATTENTION MECHANISM EXPLAINED:
-------------------------------

The attention score between token i and token j is:

           exp(qᵢ · kⱼ / √dₖ)
αᵢⱼ = ─────────────────────────
      Σₜ exp(qᵢ · kₜ / √dₖ)

Output for token i:
hᵢ = Σⱼ αᵢⱼvⱼ

This allows each token to attend to all other tokens bidirectionally.

POOLING STRATEGIES:
-------------------

To get a single sentence embedding from token embeddings H = [h₁, ..., hₙ]:

1. **MEAN POOLING** (used in this implementation):

           1   n
   e(S) = ─── Σ hᵢ
           n  i=1

   Advantages:
   - Uses all token information
   - Robust to sequence length
   - Simple and effective

2. **CLS POOLING**:
   e(S) = h₁  (use [CLS] token)

   Advantages:
   - Pre-trained for sentence-level tasks
   - Single token, fast

   Disadvantages:
   - May lose token-level information

3. **MAX POOLING**:
   e(S)ᵢ = max(h₁ᵢ, h₂ᵢ, ..., hₙᵢ)  for each dimension i

   Advantages:
   - Captures most prominent features

   Disadvantages:
   - Can be noisy

SIMILARITY COMPUTATION:
-----------------------

For two sentences A and B:

1. Encode with BERT:
   H_A = BERT(A) → [h₁ᴬ, h₂ᴬ, ..., hₙᴬ]
   H_B = BERT(B) → [h₁ᴮ, h₂ᴮ, ..., hₘᴮ]

2. Apply mean pooling:
   e(A) = mean(H_A)
   e(B) = mean(H_B)

3. Normalize (L2 norm):
   ê(A) = e(A) / ||e(A)||₂
   ê(B) = e(B) / ||e(B)||₂

4. Cosine similarity:
   sim(A, B) = ê(A) · ê(B)

BERT vs SBERT:
--------------

**BERT (this implementation)**:
- Uses base BERT model with pooling
- Not specifically optimized for similarity
- Good for transfer learning
- Requires mean pooling for sentence embeddings

**SBERT**:
- Fine-tuned on similarity tasks
- Optimized pooling strategy
- Better performance on semantic similarity
- Faster inference (optimized architecture)

EXAMPLE CALCULATION:
====================

Sentences:
   S1: "The weather is nice"
   S2: "Today is a beautiful day"

Step 1: Tokenize and add special tokens
   S1: [CLS] the weather is nice [SEP]
   S2: [CLS] today is a beautiful day [SEP]

Step 2: BERT encoding (simplified)
   After 12 transformer layers:
   H_S1 = [[0.23, -0.15, ..., 0.42],  # [CLS]
           [0.11, 0.08, ..., -0.19],   # the
           [-0.05, 0.33, ..., 0.27],   # weather
           [0.18, -0.22, ..., 0.14],   # is
           [0.31, 0.09, ..., -0.08],   # nice
           [0.07, -0.11, ..., 0.19]]   # [SEP]

   Each vector has 768 dimensions.

Step 3: Mean pooling (average all token embeddings)
   e(S1) = mean(H_S1) = [0.14, 0.00, ..., 0.13]

Step 4: Normalize to unit length
   ê(S1) = e(S1) / ||e(S1)||₂

Step 5: Compute cosine similarity
   sim(S1, S2) = ê(S1) · ê(S2) = 0.68

ATTENTION VISUALIZATION:
------------------------

For "The cat sat":
Attention weights might look like:

      The   cat   sat
The  [0.4   0.3   0.3]   # "The" attends mostly to itself
cat  [0.2   0.5   0.3]   # "cat" attends mostly to itself
sat  [0.1   0.3   0.6]   # "sat" attends mostly to itself

This shows which words the model considers related.

COMPLEXITY:
-----------
Encoding:  O(n² × d) per layer, due to self-attention
           Total: O(L × n² × d) where L = number of layers
Similarity: O(d) where d = embedding dimension (768)
Space:     O(d) per sentence embedding
"""

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger
import time

# Global model cache to avoid reloading
_BERT_MODEL_CACHE = {}
_BERT_TOKENIZER_CACHE = {}


class TransformerSimilarity:
    """
    BERT Transformer Similarity calculator

    Uses pre-trained BERT models with mean pooling to generate
    contextual embeddings and compute similarity.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize BERT Transformer Similarity calculator

        Args:
            model_name: Name of pre-trained BERT model
                       Options: 'bert-base-uncased' (default, 12 layers, 768 dim),
                               'bert-large-uncased' (24 layers, 1024 dim),
                               'distilbert-base-uncased' (6 layers, faster)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache downloaded models
            debug: If True, prints intermediate calculation steps

        Example:
            >>> # Default BERT-base
            >>> calc = TransformerSimilarity()
            >>>
            >>> # Faster DistilBERT
            >>> calc = TransformerSimilarity(model_name='distilbert-base-uncased')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.debug = debug

        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        logger.info(f"Initializing BERT Transformer: {model_name}")
        logger.info(f"Device: {device}")

        start_time = time.perf_counter()

        try:
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            # Load model
            self.model = BertModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            # Move model to device
            self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            load_time = time.perf_counter() - start_time

            # Get model info
            self.embedding_dim = self.model.config.hidden_size
            self.num_layers = self.model.config.num_hidden_layers

            logger.success(f"Model loaded in {load_time:.2f}s")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            logger.info(f"Number of layers: {self.num_layers}")

        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings

        Args:
            token_embeddings: Token embeddings from BERT (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Pooled embeddings (batch_size, hidden_dim)

        Note:
            We use attention mask to ignore padding tokens in the mean calculation.
        """
        # Expand attention mask to match embedding dimensions
        # Shape: (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings (masking padding tokens)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Sum attention mask (to get actual sequence lengths)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Mean pooling
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings using BERT

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings to unit length

        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)

        Example:
            >>> calc = TransformerSimilarity()
            >>> texts = ["Hello world", "Goodbye world"]
            >>> embeddings = calc.encode(texts)
            >>> print(embeddings.shape)
            (2, 768)
        """
        start_time = time.perf_counter()

        logger.info(f"Encoding {len(texts)} texts with BERT (batch_size={batch_size})")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Get token embeddings (last hidden state)
            token_embeddings = outputs.last_hidden_state

            # Apply mean pooling
            batch_embeddings = self._mean_pooling(
                token_embeddings,
                encoded['attention_mask']
            )

            # Normalize if requested
            if normalize:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            # Move to CPU and convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()

            all_embeddings.append(batch_embeddings)

            if self.debug and i == 0:
                logger.debug(f"\nFirst batch encoding:")
                logger.debug(f"  Batch size: {len(batch_texts)}")
                logger.debug(f"  Token embeddings shape: {token_embeddings.shape}")
                logger.debug(f"  Pooled embeddings shape: {batch_embeddings.shape}")

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nEncoding Results:")
            logger.debug(f"  Input texts: {len(texts)}")
            logger.debug(f"  Final embedding shape: {embeddings.shape}")
            logger.debug(f"  Encoding time: {elapsed_time:.3f}s")
            logger.debug(f"  Time per text: {elapsed_time/len(texts)*1000:.2f}ms")

            # Show statistics
            logger.debug(f"\n  Embedding statistics:")
            logger.debug(f"    Mean: {embeddings.mean():.6f}")
            logger.debug(f"    Std:  {embeddings.std():.6f}")
            logger.debug(f"    Min:  {embeddings.min():.6f}")
            logger.debug(f"    Max:  {embeddings.max():.6f}")

        logger.info(f"Encoded {len(texts)} texts in {elapsed_time:.2f}s "
                   f"({len(texts)/elapsed_time:.1f} texts/sec)")

        return embeddings

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute BERT cosine similarity between two texts

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Cosine similarity in range [-1.0, 1.0]
            - 1.0 = semantically identical
            - 0.0 = unrelated
            - -1.0 = opposite meanings (rare)

        Example:
            >>> calc = TransformerSimilarity()
            >>> sim = calc.compute_similarity(
            ...     "The cat sits on the mat",
            ...     "A feline rests on a rug"
            ... )
            >>> print(f"{sim:.3f}")
            0.712
        """
        start_time = time.perf_counter()

        # Encode both texts
        embeddings = self.encode([text1, text2])

        # Extract individual embeddings
        emb1 = embeddings[0]
        emb2 = embeddings[1]

        # Compute cosine similarity (embeddings are normalized)
        similarity = float(np.dot(emb1, emb2))

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nBERT Transformer Similarity:")
            logger.debug(f"  Text 1: '{text1}'")
            logger.debug(f"  Text 2: '{text2}'")
            logger.debug(f"\n  Embedding 1 shape: {emb1.shape}")
            logger.debug(f"  Embedding 2 shape: {emb2.shape}")
            logger.debug(f"\n  Cosine similarity: {similarity:.6f}")

        logger.info(f"BERT similarity: {similarity:.4f} (computed in {elapsed_time*1000:.2f}ms)")

        return similarity

    def compute_similarity_matrix(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for multiple texts

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Similarity matrix of shape (n_texts, n_texts)

        Example:
            >>> calc = TransformerSimilarity()
            >>> texts = ["doc1", "doc2", "doc3"]
            >>> matrix = calc.compute_similarity_matrix(texts)
            >>> print(matrix.shape)
            (3, 3)
        """
        start_time = time.perf_counter()

        logger.info(f"Computing BERT similarity matrix for {len(texts)} texts")

        # Encode all texts
        embeddings = self.encode(texts, batch_size=batch_size)

        # Compute pairwise cosine similarities
        similarity_matrix = np.dot(embeddings, embeddings.T)

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nSimilarity Matrix:")
            logger.debug(f"  Shape: {similarity_matrix.shape}")
            logger.debug(f"  Mean similarity: {similarity_matrix.mean():.4f}")
            logger.debug(f"  Min similarity: {similarity_matrix.min():.4f}")
            logger.debug(f"  Max similarity (off-diagonal): "
                        f"{np.max(similarity_matrix - np.eye(len(texts))):.4f}")

        logger.info(f"Similarity matrix computed in {elapsed_time:.2f}s")

        return similarity_matrix

    def get_attention_weights(
        self,
        text: str
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get attention weights for a text (for visualization)

        Args:
            text: Input text

        Returns:
            Tuple of (tokens, attention_weights)
            - tokens: List of token strings
            - attention_weights: Attention matrix from last layer
                                Shape: (num_heads, seq_len, seq_len)

        Example:
            >>> calc = TransformerSimilarity()
            >>> tokens, attn = calc.get_attention_weights("The cat sat")
            >>> print(f"Tokens: {tokens}")
            >>> print(f"Attention shape: {attn.shape}")
        """
        logger.info(f"Computing attention weights for: '{text}'")

        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**encoded, output_attentions=True)

        # Get attention from last layer
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        last_layer_attention = outputs.attentions[-1]

        # Remove batch dimension and move to CPU
        attention_weights = last_layer_attention[0].cpu().numpy()

        # Get tokens
        token_ids = encoded['input_ids'][0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        logger.info(f"Attention weights extracted: {attention_weights.shape}")

        return tokens, attention_weights


# ============================================================================
# STANDALONE FUNCTION FOR SIMPLE USAGE
# ============================================================================

def bert_similarity(
    text1: str,
    text2: str,
    model_name: str = 'bert-base-uncased',
    use_cache: bool = True
) -> float:
    """
    Similitud usando embeddings de BERT.

    Función standalone simplificada para uso rápido sin instanciar clases.

    Explicación técnica:
    --------------------

    BERT (Bidirectional Encoder Representations from Transformers) genera
    embeddings contextuales donde cada token considera todo el contexto
    de la oración bidireccional mente.

    **Diferencia con SBERT:**
    - ❌ BERT no está optimizado específicamente para similitud de frases
    - ✅ BERT captura mejor el contexto de palabras ambiguas
    - ❌ Requiere mean/max pooling manual de tokens
    - ✅ SBERT está fine-tuned específicamente para similitud (más rápido)
    - ⚠️  Para similitud de frases, SBERT suele ser mejor opción

    **Cuándo usar BERT en lugar de SBERT:**
    - Cuando necesitas comprender contexto de palabras polisémicas
    - Ejemplo: "bank" (financial) vs "bank" (river)
    - Para fine-tuning en tareas específicas
    - Cuando ya tienes modelo BERT disponible

    **Arquitectura BERT:**
    1. Input: [CLS] token₁ token₂ ... tokenₙ [SEP]
    2. Token + Position + Segment Embeddings
    3. 12 Transformer Layers (BERT-base):
       - Multi-Head Self-Attention
       - Feed-Forward Networks
       - Residual connections + Layer Normalization
    4. Output: Contextual embeddings [h₁, h₂, ..., hₙ]
    5. Mean Pooling: e(S) = (1/n) Σᵢ hᵢ
    6. L2 Normalization
    7. Cosine similarity

    **Self-Attention (núcleo de BERT):**
    Attention(Q, K, V) = softmax(QKᵀ/√dₖ) V

    Esto permite que cada token "atienda" a todos los otros tokens,
    capturando dependencias bidireccionales.

    Pasos del algoritmo:
    --------------------
    1. Tokenizar textos con BertTokenizer
    2. Obtener outputs del modelo BERT (última capa oculta)
    3. Aplicar mean pooling sobre tokens
    4. Normalizar embeddings (L2 norm)
    5. Calcular similitud coseno

    Args:
        text1: Primer texto a comparar
        text2: Segundo texto a comparar
        model_name: Nombre del modelo BERT pre-entrenado
                   'bert-base-uncased' (default): 12 layers, 768 dim
                   'bert-large-uncased': 24 layers, 1024 dim (más preciso, más lento)
                   'distilbert-base-uncased': 6 layers (más rápido)
        use_cache: Si True, cachea modelo y tokenizer

    Returns:
        Similitud contextual en rango [0.0, 1.0]
        - 1.0 = contexto semántico idéntico
        - 0.0 = sin relación contextual

    Example:
        >>> # Comprensión contextual de "bank"
        >>> sim1 = bert_similarity(
        ...     "I went to the bank to deposit money",      # financial bank
        ...     "The financial institution is closed"       # same meaning
        ... )
        >>> print(f"Same context: {sim1:.3f}")
        Same context: 0.680

        >>> sim2 = bert_similarity(
        ...     "I went to the bank to deposit money",      # financial bank
        ...     "We sat by the river bank"                  # river bank
        ... )
        >>> print(f"Different context: {sim2:.3f}")
        Different context: 0.420

        >>> # BERT entiende que "bank" tiene diferentes significados!

    Modelo usado:
    -------------
    bert-base-uncased:
    - Tamaño: ~440 MB
    - Dimensión embeddings: 768
    - Capas: 12 transformer layers
    - Attention heads: 12
    - Parámetros: 110M
    - Vocabulario: 30,522 tokens
    - Entrenamiento: BooksCorpus (800M words) + English Wikipedia (2,500M words)

    Performance:
    ------------
    - Velocidad: ~10-20 textos/seg (CPU)
    - Velocidad: ~100-200 textos/seg (GPU)
    - Memoria: ~1.5 GB (modelo + activaciones)

    Note:
        - Primera llamada descarga el modelo (~440 MB)
        - Llamadas subsecuentes usan modelo cacheado
        - Para GPU: instalar torch con CUDA support
        - Requiere: pip install transformers torch
        - Para similitud semántica pura, considerar sbert_similarity()
    """
    global _BERT_MODEL_CACHE, _BERT_TOKENIZER_CACHE

    start_time = time.perf_counter()

    # Load tokenizer from cache or download
    if use_cache and model_name in _BERT_TOKENIZER_CACHE:
        tokenizer = _BERT_TOKENIZER_CACHE[model_name]
        logger.debug(f"Using cached BERT tokenizer: {model_name}")
    else:
        logger.info(f"Loading BERT tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_cache:
            _BERT_TOKENIZER_CACHE[model_name] = tokenizer
            logger.debug(f"Cached tokenizer: {model_name}")

    # Load model from cache or download
    if use_cache and model_name in _BERT_MODEL_CACHE:
        model = _BERT_MODEL_CACHE[model_name]
        logger.debug(f"Using cached BERT model: {model_name}")
    else:
        logger.info(f"Loading BERT model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode

        if use_cache:
            _BERT_MODEL_CACHE[model_name] = model
            logger.debug(f"Cached model: {model_name}")

    # Tokenize both texts
    encoded = tokenizer(
        [text1, text2],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Forward pass through BERT
    with torch.no_grad():
        outputs = model(**encoded)

    # Get last hidden state (token embeddings)
    # Shape: (2, seq_len, 768)
    token_embeddings = outputs.last_hidden_state

    # Apply mean pooling
    # We need to mask padding tokens
    attention_mask = encoded['attention_mask']

    # Expand mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum embeddings (masking padding)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

    # Sum mask (to get actual lengths)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Mean pooling
    mean_embeddings = sum_embeddings / sum_mask

    # L2 normalization
    mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

    # Extract embeddings
    emb1 = mean_embeddings[0].numpy()  # Shape: (768,)
    emb2 = mean_embeddings[1].numpy()  # Shape: (768,)

    # Compute cosine similarity (embeddings are normalized, so just dot product)
    similarity = float(np.dot(emb1, emb2))

    elapsed_time = time.perf_counter() - start_time

    logger.debug(f"BERT similarity computed in {elapsed_time*1000:.2f}ms: {similarity:.4f}")

    return similarity


# Example usage and demonstration
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/transformer_similarity.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("=" * 70)
    print("BERT TRANSFORMER SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Basic similarity
    print("\n1. BASIC BERT SIMILARITY")
    print("-" * 70)

    calc = TransformerSimilarity(debug=True)

    text1 = "The cat sits on the mat"
    text2 = "A feline rests on a rug"
    text3 = "Python programming is fun"

    sim_12 = calc.compute_similarity(text1, text2)
    sim_13 = calc.compute_similarity(text1, text3)

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity (1, 2): {sim_12:.4f} (similar meaning)")
    print(f"Similarity (1, 3): {sim_13:.4f} (different topics)")

    # Example 2: Contextual understanding
    print("\n2. CONTEXTUAL WORD UNDERSTANDING")
    print("-" * 70)

    calc_context = TransformerSimilarity(debug=False)

    # "bank" in different contexts
    text1 = "I went to the bank to deposit money"
    text2 = "We sat by the river bank"
    text3 = "The financial institution is closed"

    sim_12 = calc_context.compute_similarity(text1, text2)
    sim_13 = calc_context.compute_similarity(text1, text3)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity (1, 2): {sim_12:.4f} (same word, different meaning)")
    print(f"Similarity (1, 3): {sim_13:.4f} (different words, same meaning)")
    print("\nNote: BERT understands context!")

    # Example 3: Similarity matrix
    print("\n3. SIMILARITY MATRIX")
    print("-" * 70)

    calc_matrix = TransformerSimilarity(debug=False)

    texts = [
        "machine learning algorithms",
        "deep neural networks",
        "artificial intelligence",
        "cooking pasta recipes"
    ]

    matrix = calc_matrix.compute_similarity_matrix(texts)

    print("\nTexts:")
    for i, text in enumerate(texts):
        print(f"  {i}: '{text}'")

    print("\nSimilarity Matrix:")
    print(matrix.round(3))

    # Example 4: Attention visualization
    print("\n4. ATTENTION WEIGHTS VISUALIZATION")
    print("-" * 70)

    calc_attn = TransformerSimilarity()

    text = "The cat sat on the mat"
    tokens, attention = calc_attn.get_attention_weights(text)

    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Attention shape: {attention.shape}")
    print(f"  (num_heads={attention.shape[0]}, seq_len={attention.shape[1]})")

    # Show average attention across all heads
    avg_attention = attention.mean(axis=0)
    print(f"\nAverage Attention Matrix (first 5x5):")
    print(avg_attention[:5, :5].round(3))

    # Example 5: Batch encoding performance
    print("\n5. BATCH ENCODING PERFORMANCE")
    print("-" * 70)

    calc_perf = TransformerSimilarity(debug=False)

    texts = [
        f"This is test sentence number {i}"
        for i in range(20)
    ]

    start = time.perf_counter()
    embeddings = calc_perf.encode(texts, batch_size=8)
    duration = time.perf_counter() - start

    print(f"\nEncoded {len(texts)} texts")
    print(f"Total time: {duration:.2f}s")
    print(f"Per text: {duration/len(texts)*1000:.1f}ms")
    print(f"Throughput: {len(texts)/duration:.1f} texts/sec")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
