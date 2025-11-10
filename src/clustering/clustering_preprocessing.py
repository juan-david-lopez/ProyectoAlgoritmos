"""
Clustering Preprocessing Module
Preprocesamiento especializado y optimizado para clustering de textos científicos.

Características:
- Limpieza profunda de textos académicos
- Tokenización avanzada con spaCy y POS tagging
- Stopwords inteligentes del dominio
- Lematización robusta
- 3 métodos de vectorización: TF-IDF, Word2Vec, SBERT
- Pipeline completo end-to-end
- Logging detallado y progress bars
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Set, Optional, Dict, Any
from collections import Counter
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn para TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Gensim para Word2Vec
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Gensim no disponible. Word2Vec no estará disponible.")

# Sentence-Transformers para SBERT
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    warnings.warn("Sentence-Transformers no disponible. SBERT no estará disponible.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusteringPreprocessor:
    """
    Preprocesamiento optimizado para clustering de textos científicos.

    Proporciona limpieza profunda, tokenización avanzada, eliminación de stopwords,
    lematización y múltiples métodos de vectorización.
    """

    # Stopwords académicas del dominio
    ACADEMIC_STOPWORDS = {
        # Palabras estructurales
        'paper', 'study', 'research', 'article', 'work', 'author', 'authors',
        'propose', 'present', 'presented', 'show', 'shows', 'shown',
        'result', 'results', 'finding', 'findings', 'conclusion', 'conclusions',

        # Metodología genérica
        'method', 'methods', 'approach', 'approaches', 'technique', 'techniques',
        'analysis', 'analyses', 'evaluation', 'evaluations',

        # Descriptores genéricos
        'new', 'novel', 'effective', 'efficient', 'important', 'significant',
        'based', 'using', 'used', 'propose', 'proposed', 'provide', 'provides',

        # Secciones de paper
        'introduction', 'background', 'related', 'section', 'sections',
        'figure', 'figures', 'table', 'tables', 'appendix',

        # Términos de investigación
        'investigate', 'investigated', 'examine', 'examined', 'explore', 'explored',
        'discuss', 'discussed', 'describe', 'described', 'demonstrate', 'demonstrated',

        # Referencias
        'et', 'al', 'etc', 'i.e', 'e.g', 'cf', 'vs', 'viz'
    }

    def __init__(self, abstracts: List[str], spacy_model: str = 'en_core_web_sm'):
        """
        Inicializa el preprocesador.

        Args:
            abstracts: Lista de abstracts a procesar
            spacy_model: Nombre del modelo spaCy a usar
        """
        self.abstracts = abstracts
        self.n_abstracts = len(abstracts)

        logger.info(f"Inicializando ClusteringPreprocessor con {self.n_abstracts} abstracts")

        # Cargar modelo spaCy
        logger.info(f"Cargando modelo spaCy: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.error(f"Modelo {spacy_model} no encontrado. Descargarlo con: python -m spacy download {spacy_model}")
            raise

        # Optimizar spaCy (desactivar componentes innecesarios para velocidad)
        if 'ner' in self.nlp.pipe_names:
            self.nlp.disable_pipes(['ner'])

        # Compilar patrones regex para eficiencia
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.doi_pattern = re.compile(r'doi:\s*\S+|10\.\d{4,}/\S+', re.IGNORECASE)
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.special_chars_pattern = re.compile(r'[^\w\s\-]')
        self.whitespace_pattern = re.compile(r'\s+')

        # Variables para almacenar resultados
        self.cleaned_texts = []
        self.tokenized_texts = []
        self.processed_texts = []
        self.vectorizer = None
        self.feature_matrix = None

        logger.info("ClusteringPreprocessor inicializado exitosamente")

    def deep_clean(self, text: str) -> str:
        """
        Limpieza profunda de texto científico.

        Pasos:
        1. Lowercase
        2. Eliminar URLs, emails, DOIs
        3. Normalizar números → TOKEN_NUM
        4. Eliminar puntuación (preservar guiones en términos compuestos)
        5. Normalizar espacios en blanco
        6. Eliminar caracteres especiales

        Args:
            text: Texto a limpiar

        Returns:
            Texto limpio
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Eliminar URLs, emails, DOIs
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.doi_pattern.sub(' ', text)

        # 3. Normalizar números
        text = self.number_pattern.sub(' TOKEN_NUM ', text)

        # 4. Eliminar puntuación (preservar guiones en términos compuestos)
        # Reemplazar guiones por espacios solo si no están entre letras
        text = re.sub(r'(?<![a-z])-(?![a-z])', ' ', text)

        # Eliminar otros caracteres especiales
        text = self.special_chars_pattern.sub(' ', text)

        # 5 y 6. Normalizar espacios en blanco
        text = self.whitespace_pattern.sub(' ', text).strip()

        return text

    def advanced_tokenization(self, text: str) -> List[str]:
        """
        Tokenización avanzada con spaCy y POS tagging.

        Pasos:
        1. Tokenización con spaCy
        2. Filtrar por POS: mantener NOUN, ADJ, VERB
        3. Eliminar pronombres, artículos, preposiciones
        4. Mantener términos importantes (verbos clave como generate, train, etc.)

        Args:
            text: Texto a tokenizar

        Returns:
            Lista de tokens relevantes
        """
        if not text:
            return []

        # Procesar con spaCy
        doc = self.nlp(text)

        # POS tags a mantener
        keep_pos = {'NOUN', 'ADJ', 'VERB', 'PROPN'}  # Añadir nombres propios

        # Verbos importantes a preservar
        important_verbs = {
            'generate', 'train', 'learn', 'classify', 'predict', 'detect',
            'recognize', 'extract', 'identify', 'analyze', 'compute', 'process',
            'optimize', 'evaluate', 'measure', 'compare', 'improve', 'develop'
        }

        tokens = []

        for token in doc:
            # Saltar si es stopword de spaCy (pero preservaremos los importantes después)
            if token.is_stop and token.lemma_.lower() not in important_verbs:
                continue

            # Saltar puntuación y espacios
            if token.is_punct or token.is_space:
                continue

            # Saltar tokens muy cortos (< 2 caracteres)
            if len(token.text) < 2:
                continue

            # Filtrar por POS
            if token.pos_ in keep_pos:
                tokens.append(token.text.lower())
            # Preservar verbos importantes incluso si no están en keep_pos
            elif token.lemma_.lower() in important_verbs:
                tokens.append(token.text.lower())

        return tokens

    def remove_stopwords(self, tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
        """
        Eliminación inteligente de stopwords.

        Combina:
        - Stopwords académicas predefinidas
        - Custom stopwords del usuario
        - Preserva términos técnicos de IA/ML

        Args:
            tokens: Lista de tokens
            custom_stopwords: Stopwords adicionales opcionales

        Returns:
            Tokens filtrados
        """
        # Combinar stopwords
        stopwords = self.ACADEMIC_STOPWORDS.copy()

        if custom_stopwords:
            stopwords.update(custom_stopwords)

        # Términos técnicos a preservar (nunca eliminar)
        technical_terms = {
            # ML/AI
            'neural', 'network', 'learning', 'deep', 'machine', 'model', 'algorithm',
            'training', 'test', 'validation', 'accuracy', 'loss', 'optimization',

            # Arquitecturas
            'cnn', 'rnn', 'lstm', 'gru', 'transformer', 'attention', 'bert', 'gpt',
            'resnet', 'vgg', 'inception', 'mobilenet', 'efficientnet',

            # Técnicas
            'classification', 'regression', 'clustering', 'segmentation', 'detection',
            'recognition', 'prediction', 'generation', 'embedding', 'encoding',

            # Dominios
            'vision', 'nlp', 'speech', 'audio', 'image', 'text', 'language',
            'computer', 'artificial', 'intelligence', 'data', 'feature'
        }

        # Filtrar tokens
        filtered = []
        for token in tokens:
            # Preservar términos técnicos
            if token in technical_terms:
                filtered.append(token)
            # Eliminar stopwords
            elif token not in stopwords:
                filtered.append(token)

        return filtered

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lematización con spaCy para normalizar variantes.

        Ejemplos:
        - 'models' → 'model'
        - 'generated' → 'generate'
        - 'training' → 'train'

        Args:
            tokens: Lista de tokens

        Returns:
            Tokens lematizados
        """
        if not tokens:
            return []

        # Procesar en batch para eficiencia
        # Crear documento temporal
        text = ' '.join(tokens)
        doc = self.nlp(text)

        # Lematizar manteniendo solo tokens relevantes
        lemmas = []
        for token in doc:
            if not token.is_punct and not token.is_space:
                lemma = token.lemma_.lower()
                # Evitar lemmas como '-PRON-'
                if lemma not in {'-pron-', '-'}:
                    lemmas.append(lemma)

        return lemmas

    def vectorize_texts(self,
                       processed_texts: List[str],
                       method: str = 'tfidf',
                       **kwargs) -> Tuple[np.ndarray, Any]:
        """
        Convierte textos procesados a vectores numéricos.

        Métodos disponibles:
        1. 'tfidf': TF-IDF clásico (sklearn)
        2. 'word2vec': Promedio de embeddings Word2Vec (gensim)
        3. 'sbert': Sentence-BERT embeddings (sentence-transformers)

        Args:
            processed_texts: Lista de textos preprocesados
            method: Método de vectorización
            **kwargs: Parámetros adicionales para el método

        Returns:
            (feature_matrix, vectorizer_object)
        """
        logger.info(f"Vectorizando con método: {method}")

        if method == 'tfidf':
            return self._vectorize_tfidf(processed_texts, **kwargs)

        elif method == 'word2vec':
            if not GENSIM_AVAILABLE:
                raise ImportError("Gensim no está instalado. Instalar con: pip install gensim")
            return self._vectorize_word2vec(processed_texts, **kwargs)

        elif method == 'sbert':
            if not SBERT_AVAILABLE:
                raise ImportError("Sentence-Transformers no está instalado. Instalar con: pip install sentence-transformers")
            return self._vectorize_sbert(processed_texts, **kwargs)

        else:
            raise ValueError(f"Método '{method}' no reconocido. Usar: 'tfidf', 'word2vec', o 'sbert'")

    def _vectorize_tfidf(self, texts: List[str], **kwargs) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Vectorización con TF-IDF.

        Parámetros optimizados para clustering:
        - max_features: limitar dimensionalidad
        - ngram_range: capturar frases
        - min_df: frecuencia mínima
        - max_df: frecuencia máxima
        - sublinear_tf: escala logarítmica
        """
        # Parámetros por defecto optimizados
        params = {
            'max_features': kwargs.get('max_features', 1000),
            'ngram_range': kwargs.get('ngram_range', (1, 3)),
            'min_df': kwargs.get('min_df', 2),
            'max_df': kwargs.get('max_df', 0.85),
            'sublinear_tf': kwargs.get('sublinear_tf', True),
            'use_idf': kwargs.get('use_idf', True),
            'smooth_idf': kwargs.get('smooth_idf', True)
        }

        logger.info(f"TF-IDF params: {params}")

        # Crear y ajustar vectorizador
        vectorizer = TfidfVectorizer(**params)

        try:
            feature_matrix = vectorizer.fit_transform(texts)
            logger.info(f"Matriz TF-IDF: {feature_matrix.shape} ({feature_matrix.nnz} valores no-cero)")

            return feature_matrix.toarray(), vectorizer

        except ValueError as e:
            logger.error(f"Error en TF-IDF: {e}")
            raise

    def _vectorize_word2vec(self, texts: List[str], **kwargs) -> Tuple[np.ndarray, Word2Vec]:
        """
        Vectorización con Word2Vec.

        Vector de documento = promedio de vectores de palabras.
        """
        # Tokenizar textos para Word2Vec
        tokenized = [text.split() for text in texts]

        # Parámetros Word2Vec
        params = {
            'vector_size': kwargs.get('vector_size', 100),
            'window': kwargs.get('window', 5),
            'min_count': kwargs.get('min_count', 2),
            'workers': kwargs.get('workers', 4),
            'epochs': kwargs.get('epochs', 10),
            'sg': kwargs.get('sg', 1)  # 1=skip-gram, 0=CBOW
        }

        logger.info(f"Word2Vec params: {params}")

        # Entrenar modelo
        logger.info("Entrenando modelo Word2Vec...")
        model = Word2Vec(sentences=tokenized, **params)

        # Convertir documentos a vectores
        logger.info("Generando vectores de documentos...")
        doc_vectors = []

        for tokens in tqdm(tokenized, desc="Vectorizando"):
            # Obtener vectores de palabras
            word_vectors = [
                model.wv[word]
                for word in tokens
                if word in model.wv
            ]

            if word_vectors:
                # Promedio de vectores
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Vector cero si no hay palabras en vocabulario
                doc_vector = np.zeros(params['vector_size'])

            doc_vectors.append(doc_vector)

        feature_matrix = np.array(doc_vectors)

        logger.info(f"Matriz Word2Vec: {feature_matrix.shape}")

        return feature_matrix, model

    def _vectorize_sbert(self, texts: List[str], **kwargs) -> Tuple[np.ndarray, SentenceTransformer]:
        """
        Vectorización con Sentence-BERT.

        Usa modelos pre-entrenados para embeddings densos.
        """
        # Modelo SBERT a usar
        model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
        batch_size = kwargs.get('batch_size', 32)

        logger.info(f"Cargando modelo SBERT: {model_name}")
        model = SentenceTransformer(model_name)

        # Generar embeddings
        logger.info("Generando embeddings SBERT...")
        feature_matrix = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Matriz SBERT: {feature_matrix.shape}")

        return feature_matrix, model

    def full_preprocessing_pipeline(self,
                                   method: str = 'tfidf',
                                   return_intermediate: bool = False,
                                   **vectorization_kwargs) -> Dict[str, Any]:
        """
        Pipeline completo end-to-end.

        Pasos:
        1. Limpieza profunda
        2. Tokenización avanzada
        3. Stopwords removal
        4. Lematización
        5. Vectorización

        Args:
            method: Método de vectorización ('tfidf', 'word2vec', 'sbert')
            return_intermediate: Si True, retorna resultados intermedios
            **vectorization_kwargs: Parámetros para vectorización

        Returns:
            Dict con:
            - feature_matrix: Matriz de features
            - vectorizer: Objeto vectorizador
            - processed_texts: Textos procesados
            - [opcional] cleaned_texts, tokenized_texts, etc.
        """
        logger.info("="*70)
        logger.info("INICIANDO PIPELINE COMPLETO DE PREPROCESAMIENTO")
        logger.info("="*70)
        logger.info(f"Abstracts: {self.n_abstracts}")
        logger.info(f"Método de vectorización: {method}")

        # 1. Limpieza profunda
        logger.info("\n[1/5] Limpieza profunda...")
        self.cleaned_texts = [
            self.deep_clean(text)
            for text in tqdm(self.abstracts, desc="Limpiando")
        ]
        logger.info(f"✓ {len(self.cleaned_texts)} textos limpiados")

        # 2. Tokenización avanzada
        logger.info("\n[2/5] Tokenización avanzada...")
        self.tokenized_texts = [
            self.advanced_tokenization(text)
            for text in tqdm(self.cleaned_texts, desc="Tokenizando")
        ]

        # Estadísticas de tokens
        total_tokens = sum(len(tokens) for tokens in self.tokenized_texts)
        avg_tokens = total_tokens / len(self.tokenized_texts) if self.tokenized_texts else 0
        logger.info(f"✓ Total tokens: {total_tokens}, Promedio: {avg_tokens:.1f} tokens/doc")

        # 3. Stopwords removal
        logger.info("\n[3/5] Eliminación de stopwords...")
        filtered_texts = [
            self.remove_stopwords(tokens)
            for tokens in tqdm(self.tokenized_texts, desc="Filtrando stopwords")
        ]

        # Estadísticas después de filtrado
        total_after = sum(len(tokens) for tokens in filtered_texts)
        logger.info(f"✓ Tokens después de filtrado: {total_after} ({total_after/total_tokens*100:.1f}%)")

        # 4. Lematización
        logger.info("\n[4/5] Lematización...")
        lemmatized_texts = [
            self.lemmatize(tokens)
            for tokens in tqdm(filtered_texts, desc="Lematizando")
        ]

        # Convertir a strings para vectorización
        self.processed_texts = [
            ' '.join(tokens)
            for tokens in lemmatized_texts
        ]

        # Estadísticas finales
        vocab_size = len(set(token for tokens in lemmatized_texts for token in tokens))
        logger.info(f"✓ Vocabulario único: {vocab_size} términos")

        # 5. Vectorización
        logger.info(f"\n[5/5] Vectorización con {method}...")
        self.feature_matrix, self.vectorizer = self.vectorize_texts(
            self.processed_texts,
            method=method,
            **vectorization_kwargs
        )

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("="*70)
        logger.info(f"Matriz final: {self.feature_matrix.shape}")
        logger.info(f"Densidad: {np.count_nonzero(self.feature_matrix) / self.feature_matrix.size * 100:.2f}%")

        # Preparar resultado
        result = {
            'feature_matrix': self.feature_matrix,
            'vectorizer': self.vectorizer,
            'processed_texts': self.processed_texts,
            'n_documents': self.n_abstracts,
            'n_features': self.feature_matrix.shape[1],
            'method': method
        }

        # Añadir resultados intermedios si se solicita
        if return_intermediate:
            result['cleaned_texts'] = self.cleaned_texts
            result['tokenized_texts'] = self.tokenized_texts
            result['lemmatized_texts'] = lemmatized_texts
            result['vocabulary_size'] = vocab_size

        return result


def main():
    """Ejemplo de uso del preprocesador."""

    print("\n" + "="*70)
    print(" EJEMPLO: Clustering Preprocessing")
    print("="*70)

    # Abstracts de ejemplo
    abstracts = [
        """Deep learning and convolutional neural networks have revolutionized
        computer vision. We propose a new architecture for image classification.""",

        """Natural language processing uses transformer models and attention mechanisms.
        BERT and GPT demonstrate state-of-the-art results in text understanding.""",

        """Reinforcement learning agents learn through interaction with environments.
        Q-learning and policy gradients are fundamental algorithms in RL.""",

        """Generative adversarial networks can generate realistic images.
        The generator and discriminator are trained in an adversarial manner.""",

        """Convolutional neural networks extract hierarchical features from images.
        ResNet and VGG are popular architectures for computer vision tasks."""
    ]

    print(f"\nCorpus: {len(abstracts)} abstracts\n")

    # Crear preprocesador
    preprocessor = ClusteringPreprocessor(abstracts)

    # Ejecutar pipeline completo con TF-IDF
    print("\n" + "="*70)
    print("Pipeline con TF-IDF")
    print("="*70)

    result_tfidf = preprocessor.full_preprocessing_pipeline(
        method='tfidf',
        return_intermediate=True,
        max_features=100
    )

    print(f"\nMatriz TF-IDF shape: {result_tfidf['feature_matrix'].shape}")
    print(f"Vocabulario: {result_tfidf['vocabulary_size']} términos únicos")

    # Mostrar algunos textos procesados
    print("\n" + "-"*70)
    print("Ejemplos de textos procesados:")
    print("-"*70)

    for i, text in enumerate(result_tfidf['processed_texts'][:2]):
        print(f"\n[Doc {i+1}]: {text[:100]}...")

    print("\n" + "="*70)
    print(" ✓ EJEMPLO COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
