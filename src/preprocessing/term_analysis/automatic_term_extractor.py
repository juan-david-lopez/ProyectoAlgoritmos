"""
Extractor automático de términos clave de abstracts científicos.

Este módulo implementa 3 métodos de extracción automática:
    1. TF-IDF: Basado en frecuencia estadística
    2. RAKE: Rapid Automatic Keyword Extraction
    3. TextRank: Basado en algoritmo PageRank

También incluye un método ensemble que combina los 3 para mejor precisión.

Referencias:
    - Rose, S., et al. (2010). Automatic Keyword Extraction from Individual Documents.
    - Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into Texts.
"""

import re
import logging
from typing import List, Tuple, Dict
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# RAKE
try:
    from rake_nltk import Rake
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False

# TextRank
try:
    import pytextrank
    import spacy
    TEXTRANK_AVAILABLE = True
except ImportError:
    TEXTRANK_AVAILABLE = False

logger = logging.getLogger(__name__)


class AutomaticTermExtractor:
    """
    Extrae automáticamente los términos más relevantes de abstracts.

    Esta clase implementa múltiples algoritmos de extracción de keywords
    y los combina en un método ensemble para máxima precisión.

    Métodos disponibles:
        - TF-IDF: Estadístico, rápido
        - RAKE: Frases candidatas
        - TextRank: Basado en grafo
        - Combined: Ensemble de los 3

    Atributos:
        abstracts: Lista de abstracts a analizar
        max_terms: Número máximo de términos a extraer
    """

    # Stopwords personalizadas para texto técnico
    CUSTOM_STOPWORDS = {
        'paper', 'study', 'research', 'article', 'work', 'approach',
        'method', 'result', 'conclusion', 'introduction', 'abstract',
        'using', 'based', 'propose', 'present', 'show', 'demonstrate',
        'however', 'therefore', 'moreover', 'furthermore', 'additionally',
        'also', 'although', 'thus', 'hence', 'whereas', 'within'
    }

    def __init__(self, abstracts: List[str], max_terms: int = 15):
        """
        Inicializa el extractor con corpus de abstracts.

        Args:
            abstracts: Lista de abstracts científicos
            max_terms: Número máximo de términos a extraer por método
        """
        logger.info(f"Inicializando AutomaticTermExtractor...")
        logger.info(f"  Abstracts: {len(abstracts)}")
        logger.info(f"  Max términos: {max_terms}")

        self.abstracts = abstracts
        self.max_terms = max_terms

        # Verificar disponibilidad de bibliotecas
        self._check_dependencies()

        # Inicializar componentes NLP
        if NLTK_AVAILABLE:
            self._init_nltk()

        logger.info("✓ AutomaticTermExtractor inicializado")

    def _check_dependencies(self):
        """Verifica qué bibliotecas están disponibles."""
        logger.info("\nVerificando dependencias:")
        logger.info(f"  NLTK: {'✓' if NLTK_AVAILABLE else '✗ (pip install nltk)'}")
        logger.info(f"  RAKE: {'✓' if RAKE_AVAILABLE else '✗ (pip install rake-nltk)'}")
        logger.info(f"  TextRank: {'✓' if TEXTRANK_AVAILABLE else '✗ (pip install pytextrank spacy)'}")

        if not NLTK_AVAILABLE:
            logger.warning("⚠️ NLTK no disponible. Funcionalidad limitada.")

    def _init_nltk(self):
        """Inicializa componentes de NLTK (descarga recursos si es necesario)."""
        try:
            # Descargar recursos necesarios si no existen
            resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']

            for resource in resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else
                                  f'corpora/{resource}' if resource in ['stopwords', 'wordnet', 'omw-1.4'] else
                                  f'taggers/{resource}')
                except LookupError:
                    logger.info(f"  Descargando recurso NLTK: {resource}")
                    nltk.download(resource, quiet=True)

            # Inicializar lematizador
            self.lemmatizer = WordNetLemmatizer()

            # Cargar stopwords
            self.stopwords = set(stopwords.words('english'))
            self.stopwords.update(self.CUSTOM_STOPWORDS)

            logger.debug("✓ NLTK inicializado correctamente")

        except Exception as e:
            logger.warning(f"⚠️ Error inicializando NLTK: {e}")

    def preprocess_for_extraction(self, text: str) -> List[str]:
        """
        Preprocesamiento completo para extracción de términos.

        Pasos:
            1. Tokenización con NLTK
            2. Lowercase
            3. Eliminación de stopwords (inglés + técnico)
            4. Lematización (agrupar variantes)
            5. Filtrado por POS tags (sustantivos/adjetivos)

        POS Tags relevantes:
            - NN, NNS, NNP, NNPS: Sustantivos (singular, plural, propios)
            - JJ, JJR, JJS: Adjetivos (base, comparativo, superlativo)
            - VBG: Gerundios (ej: "learning", "training")

        Ejemplo:
            Input:  "Machine Learning algorithms are improving rapidly"
            Output: ["machine", "learning", "algorithm", "improve"]

        Args:
            text: Texto a preprocesar

        Returns:
            Lista de tokens limpios y lematizados
        """
        if not NLTK_AVAILABLE:
            # Fallback simple sin NLTK
            tokens = text.lower().split()
            tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]
            return tokens

        # 1. Tokenización
        tokens = word_tokenize(text.lower())

        # 2. Filtrar por longitud y tipo (solo palabras)
        tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]

        # 3. POS tagging
        tagged = pos_tag(tokens)

        # 4. Filtrar por POS tags relevantes
        relevant_pos = {'NN', 'NNS', 'NNP', 'NNPS',  # Sustantivos
                       'JJ', 'JJR', 'JJS',            # Adjetivos
                       'VBG'}                         # Gerundios

        tokens_filtered = [word for word, pos in tagged if pos in relevant_pos]

        # 5. Lematización
        tokens_lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens_filtered]

        # 6. Eliminar stopwords
        tokens_clean = [t for t in tokens_lemmatized if t not in self.stopwords]

        # 7. Filtrar tokens muy cortos después del procesamiento
        tokens_final = [t for t in tokens_clean if len(t) > 2]

        return tokens_final

    def extract_with_tfidf(self, n_terms: int = None) -> List[Tuple[str, float]]:
        """
        Extracción usando TF-IDF (Term Frequency-Inverse Document Frequency).

        Método Estadístico:
            TF-IDF = TF(t,d) × IDF(t,D)

            donde:
            - TF(t,d) = Frecuencia del término t en documento d
            - IDF(t,D) = log(N / df(t))
            - N = número total de documentos
            - df(t) = documentos que contienen t

        Ventajas:
            - Penaliza términos muy comunes
            - Resalta términos distintivos
            - Rápido y escalable

        Configuración del Vectorizer:
            - max_features: 500 (limitar vocabulario)
            - ngram_range: (1,3) (unigramas, bigramas, trigramas)
            - min_df: 2 (mínimo 2 documentos)
            - max_df: 0.8 (máximo 80% de documentos)
            - sublinear_tf: True (escala logarítmica)

        Args:
            n_terms: Número de términos a extraer (usa self.max_terms si None)

        Returns:
            Lista de tuplas (término, score TF-IDF) ordenadas por score
        """
        if n_terms is None:
            n_terms = self.max_terms

        logger.info(f"\n🔍 Extrayendo términos con TF-IDF (top {n_terms})...")

        try:
            # Configurar vectorizador TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),  # Captura términos compuestos
                min_df=2,            # Aparecer en al menos 2 docs
                max_df=0.8,          # No más del 80% de docs
                sublinear_tf=True,   # Escala logarítmica para TF
                stop_words='english'
            )

            # Fit y transform
            tfidf_matrix = vectorizer.fit_transform(self.abstracts)

            # Obtener nombres de características
            feature_names = vectorizer.get_feature_names_out()

            # Sumar scores TF-IDF por término (across all documents)
            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

            # Crear lista de (término, score)
            term_scores = list(zip(feature_names, tfidf_scores))

            # Ordenar por score descendente
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Retornar top n
            top_terms = term_scores[:n_terms]

            logger.info(f"✓ TF-IDF extrajo {len(top_terms)} términos")

            return top_terms

        except Exception as e:
            logger.error(f"✗ Error en TF-IDF extraction: {e}")
            return []

    def extract_with_rake(self, n_terms: int = None) -> List[Tuple[str, float]]:
        """
        Extracción usando RAKE (Rapid Automatic Keyword Extraction).

        Algoritmo RAKE:
            1. Identificar frases candidatas (secuencias sin stopwords)
            2. Para cada palabra en frases candidatas:
               - deg(w) = grado (número de co-ocurrencias)
               - freq(w) = frecuencia
            3. Score palabra = deg(w) / freq(w)
            4. Score frase = suma de scores de palabras

        Ventajas:
            - No requiere corpus de entrenamiento
            - Bueno para términos compuestos (ej: "machine learning")
            - Considera co-ocurrencia de palabras

        Ejemplo:
            Frase: "machine learning algorithms for text processing"
            Candidatas: ["machine learning algorithms", "text processing"]
            Scores basados en co-ocurrencia de palabras

        Args:
            n_terms: Número de términos a extraer

        Returns:
            Lista de tuplas (término, score RAKE) ordenadas
        """
        if n_terms is None:
            n_terms = self.max_terms

        if not RAKE_AVAILABLE:
            logger.warning("⚠️ RAKE no disponible. Instalar: pip install rake-nltk")
            return []

        logger.info(f"\n🔍 Extrayendo términos con RAKE (top {n_terms})...")

        try:
            # Inicializar RAKE
            rake = Rake(
                stopwords=self.stopwords if NLTK_AVAILABLE else None,
                max_length=3  # Máximo 3 palabras por frase
            )

            # Concatenar todos los abstracts
            full_text = " ".join(self.abstracts)

            # Extraer keywords
            rake.extract_keywords_from_text(full_text)

            # Obtener keywords rankeadas con scores
            keywords_with_scores = rake.get_ranked_phrases_with_scores()

            # Invertir para tener (phrase, score)
            term_scores = [(phrase, score) for score, phrase in keywords_with_scores]

            # Ordenar por score descendente
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Retornar top n
            top_terms = term_scores[:n_terms]

            logger.info(f"✓ RAKE extrajo {len(top_terms)} términos")

            return top_terms

        except Exception as e:
            logger.error(f"✗ Error en RAKE extraction: {e}")
            return []

    def extract_with_textrank(self, n_terms: int = None) -> List[Tuple[str, float]]:
        """
        Extracción usando TextRank (basado en PageRank de Google).

        Algoritmo TextRank:
            1. Construir grafo de co-ocurrencia:
               - Nodos = palabras
               - Aristas = co-ocurrencia en ventana de N palabras
            2. Aplicar PageRank iterativo:
               WS(V_i) = (1-d) + d × Σ_{V_j ∈ In(V_i)} (WS(V_j) / |Out(V_j)|)

               donde:
               - d = damping factor (típicamente 0.85)
               - In(V_i) = nodos apuntando a V_i
               - Out(V_j) = nodos desde V_j
            3. Palabras con mayor score = más centrales

        Ventajas:
            - Captura importancia estructural
            - No requiere entrenamiento
            - Considera contexto global

        Analogía:
            Como PageRank para páginas web, pero con palabras.
            Palabras "centrales" en el grafo de co-ocurrencia son más importantes.

        Args:
            n_terms: Número de términos a extraer

        Returns:
            Lista de tuplas (término, score PageRank) ordenadas
        """
        if n_terms is None:
            n_terms = self.max_terms

        if not TEXTRANK_AVAILABLE:
            logger.warning("⚠️ TextRank no disponible. Instalar: pip install pytextrank spacy python -m spacy download en_core_web_sm")
            return self._fallback_textrank(n_terms)

        logger.info(f"\n🔍 Extrayendo términos con TextRank (top {n_terms})...")

        try:
            # Cargar modelo spaCy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Modelo spaCy no encontrado. Intentando descarga...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                nlp = spacy.load("en_core_web_sm")

            # Agregar pytextrank al pipeline
            nlp.add_pipe("textrank")

            # Concatenar abstracts
            full_text = " ".join(self.abstracts)

            # Procesar con spaCy + TextRank
            doc = nlp(full_text)

            # Extraer frases rankeadas
            term_scores = []
            for phrase in doc._.phrases[:n_terms * 2]:  # Obtener más para filtrar
                # phrase.text = texto de la frase
                # phrase.rank = score PageRank
                term_scores.append((phrase.text.lower(), phrase.rank))

            # Ordenar por score
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Retornar top n
            top_terms = term_scores[:n_terms]

            logger.info(f"✓ TextRank extrajo {len(top_terms)} términos")

            return top_terms

        except Exception as e:
            logger.error(f"✗ Error en TextRank extraction: {e}")
            return self._fallback_textrank(n_terms)

    def _fallback_textrank(self, n_terms: int) -> List[Tuple[str, float]]:
        """
        Implementación fallback de TextRank si bibliotecas no disponibles.

        Simplificación:
            1. Construir grafo de co-ocurrencia con ventana de 5 palabras
            2. Calcular grado de cada palabra (número de conexiones)
            3. Usar grado como aproximación de PageRank

        Args:
            n_terms: Número de términos

        Returns:
            Lista de términos con scores aproximados
        """
        logger.info("  Usando implementación fallback de TextRank...")

        # Preprocesar todos los abstracts
        all_tokens = []
        for abstract in self.abstracts:
            tokens = self.preprocess_for_extraction(abstract)
            all_tokens.extend(tokens)

        # Construir grafo de co-ocurrencia
        window_size = 5
        graph = defaultdict(set)

        for i in range(len(all_tokens) - window_size + 1):
            window = all_tokens[i:i + window_size]
            for word in window:
                # Conectar con todas las otras palabras en la ventana
                for other_word in window:
                    if word != other_word:
                        graph[word].add(other_word)

        # Calcular "grado" (número de conexiones) como proxy de PageRank
        degrees = {word: len(connections) for word, connections in graph.items()}

        # Ordenar por grado
        sorted_terms = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Normalizar scores a [0, 1]
        if sorted_terms:
            max_degree = sorted_terms[0][1]
            normalized = [(term, degree / max_degree) for term, degree in sorted_terms]
            return normalized[:n_terms]

        return []

    def extract_combined(self, n_terms: int = None) -> List[Tuple[str, float, Dict]]:
        """
        Método ensemble: combina resultados de los 3 métodos.

        Estrategia de Combinación:
            1. Ejecutar TF-IDF, RAKE, TextRank
            2. Normalizar scores individuales a [0, 1]
            3. Para cada término único:
               score_combined = (w1×TF-IDF + w2×RAKE + w3×TextRank) / sum(w)
            4. Ordenar por score combinado

        Pesos por defecto:
            - TF-IDF: 0.4 (estadístico, confiable)
            - RAKE: 0.3 (frases, co-ocurrencia)
            - TextRank: 0.3 (estructural, grafo)

        Ventajas del Ensemble:
            - Más robusto que métodos individuales
            - Combina fortalezas de cada enfoque
            - Reduce sesgos de métodos individuales

        Args:
            n_terms: Número de términos finales

        Returns:
            Lista de tuplas (término, score_combinado, scores_dict)
            donde scores_dict = {'tfidf': x, 'rake': y, 'textrank': z}
        """
        if n_terms is None:
            n_terms = self.max_terms

        logger.info(f"\n🎯 Extracción combinada (ensemble) - top {n_terms}...")

        # Pesos para cada método
        weights = {
            'tfidf': 0.4,
            'rake': 0.3,
            'textrank': 0.3
        }

        # 1. Ejecutar los 3 métodos
        tfidf_results = self.extract_with_tfidf(n_terms * 2)
        rake_results = self.extract_with_rake(n_terms * 2)
        textrank_results = self.extract_with_textrank(n_terms * 2)

        # 2. Normalizar scores a [0, 1]
        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            range_score = max_score - min_score if max_score != min_score else 1.0

            return {
                term: (score - min_score) / range_score
                for term, score in results
            }

        tfidf_norm = normalize_scores(tfidf_results)
        rake_norm = normalize_scores(rake_results)
        textrank_norm = normalize_scores(textrank_results)

        logger.debug(f"  Términos normalizados: TF-IDF={len(tfidf_norm)}, RAKE={len(rake_norm)}, TextRank={len(textrank_norm)}")

        # 3. Combinar scores
        all_terms = set(tfidf_norm.keys()) | set(rake_norm.keys()) | set(textrank_norm.keys())

        combined_results = []
        for term in all_terms:
            # Obtener scores (0 si no está presente)
            score_tfidf = tfidf_norm.get(term, 0.0)
            score_rake = rake_norm.get(term, 0.0)
            score_textrank = textrank_norm.get(term, 0.0)

            # Calcular score combinado (promedio ponderado)
            # Solo considerar métodos que detectaron el término
            active_weights = []
            active_scores = []

            if score_tfidf > 0:
                active_weights.append(weights['tfidf'])
                active_scores.append(score_tfidf)
            if score_rake > 0:
                active_weights.append(weights['rake'])
                active_scores.append(score_rake)
            if score_textrank > 0:
                active_weights.append(weights['textrank'])
                active_scores.append(score_textrank)

            if active_weights:
                combined_score = sum(w * s for w, s in zip(active_weights, active_scores)) / sum(active_weights)
            else:
                combined_score = 0.0

            # Guardar scores individuales
            scores_dict = {
                'tfidf': score_tfidf,
                'rake': score_rake,
                'textrank': score_textrank,
                'methods_count': sum(1 for s in [score_tfidf, score_rake, score_textrank] if s > 0)
            }

            combined_results.append((term, combined_score, scores_dict))

        # 4. Ordenar por score combinado
        combined_results.sort(key=lambda x: (x[1], x[2]['methods_count']), reverse=True)

        # 5. Retornar top n
        top_terms = combined_results[:n_terms]

        logger.info(f"✓ Ensemble combinó {len(all_terms)} términos únicos")
        logger.info(f"  Top {n_terms} seleccionados")

        # Mostrar breakdown de métodos
        methods_count = Counter(t[2]['methods_count'] for t in top_terms)
        logger.info(f"  Términos por número de métodos: {dict(methods_count)}")

        return top_terms


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Datos de ejemplo
    abstracts = [
        "Machine learning algorithms have revolutionized natural language processing.",
        "Deep learning techniques enable advanced text understanding and generation.",
        "Neural networks can learn complex patterns from training data."
    ]

    # Crear extractor
    extractor = AutomaticTermExtractor(abstracts, max_terms=10)

    # Extraer con cada método
    print("\n" + "="*80)
    print("TF-IDF:")
    tfidf_terms = extractor.extract_with_tfidf()
    for term, score in tfidf_terms[:5]:
        print(f"  {term}: {score:.3f}")

    print("\n" + "="*80)
    print("RAKE:")
    rake_terms = extractor.extract_with_rake()
    for term, score in rake_terms[:5]:
        print(f"  {term}: {score:.3f}")

    print("\n" + "="*80)
    print("TextRank:")
    textrank_terms = extractor.extract_with_textrank()
    for term, score in textrank_terms[:5]:
        print(f"  {term}: {score:.3f}")

    print("\n" + "="*80)
    print("COMBINED (Ensemble):")
    combined_terms = extractor.extract_combined()
    for term, score, scores_dict in combined_terms[:5]:
        print(f"  {term}: {score:.3f} (métodos: {scores_dict['methods_count']})")
