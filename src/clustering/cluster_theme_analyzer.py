"""
Cluster Theme Analyzer Module
Análisis e interpretación temática de clusters para análisis bibliométrico.

Este módulo proporciona herramientas avanzadas para analizar y visualizar
los temas principales en cada cluster, facilitando la interpretación de
los resultados del clustering jerárquico.

Características:
- Extracción de tópicos con TF-IDF, LDA y RAKE
- Generación de resúmenes interpretativos de clusters
- Word clouds temáticos por cluster
- Redes de co-ocurrencia de términos
- Visualización 2D con t-SNE y UMAP
- Análisis temporal de clusters
- Identificación de documentos representativos
- Heatmaps de similitud intra/inter-cluster

Métodos de extracción de tópicos:
=================================
1. TF-IDF: Términos con mayor importancia relativa en el cluster
2. LDA: Modelos de tópicos probabilísticos (para clusters grandes)
3. RAKE: Extracción de frases clave multi-palabra
4. Co-ocurrencia: Análisis de términos que aparecen juntos

Uso típico:
===========
analyzer = ClusterThemeAnalyzer(abstracts, cluster_labels)
summary = analyzer.generate_cluster_summary(cluster_id=1, docs_data=data)
analyzer.visualize_cluster_themes(output_dir='output/themes')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from tqdm import tqdm
import json

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Dimensionality reduction
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

# Word clouds
from wordcloud import WordCloud

# RAKE for keyword extraction
try:
    from rake_nltk import Rake
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False
    logging.warning("RAKE not available. Install with: pip install rake-nltk")

# NetworkX for co-occurrence networks
import networkx as nx

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ClusterThemeAnalyzer:
    """
    Analizador de temas para clusters jerárquicos.

    Esta clase proporciona un conjunto completo de herramientas para analizar
    e interpretar los temas principales en cada cluster, facilitando la
    comprensión de los resultados del clustering.

    Attributes:
        abstracts: Lista de abstracts/textos de documentos
        cluster_labels: Array de asignaciones de cluster para cada documento
        n_clusters: Número total de clusters
        tfidf_vectorizer: Vectorizador TF-IDF ajustado a los datos
        tfidf_matrix: Matriz TF-IDF de todos los documentos
    """

    def __init__(self,
                 abstracts: List[str],
                 cluster_labels: np.ndarray,
                 max_features: int = 500):
        """
        Inicializa el analizador de temas de clusters.

        Args:
            abstracts: Lista de abstracts/textos de documentos
            cluster_labels: Array de asignaciones de cluster (1-indexed)
            max_features: Número máximo de features para TF-IDF

        Example:
            >>> analyzer = ClusterThemeAnalyzer(abstracts, cluster_labels)
            >>> themes = analyzer.extract_cluster_topics(cluster_id=1)
        """
        self.abstracts = abstracts
        self.cluster_labels = cluster_labels
        self.n_clusters = len(np.unique(cluster_labels))
        self.max_features = max_features

        # Vectorizar todos los documentos con TF-IDF
        logger.info("Vectorizando documentos con TF-IDF...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            max_df=0.8,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(abstracts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        logger.info("="*70)
        logger.info("CLUSTER THEME ANALYZER INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Documentos: {len(abstracts)}")
        logger.info(f"Clusters: {self.n_clusters}")
        logger.info(f"Features (términos): {len(self.feature_names)}")

    def extract_cluster_topics(self,
                               cluster_id: int,
                               n_topics: int = 10,
                               method: str = 'tfidf') -> List[Tuple[str, float]]:
        """
        Extrae los tópicos principales de un cluster específico.

        Métodos disponibles:
        ===================
        1. 'tfidf': TF-IDF promedio en el cluster
           - Rápido y efectivo
           - Identifica términos distintivos del cluster

        2. 'lda': Latent Dirichlet Allocation
           - Modelos de tópicos probabilísticos
           - Mejor para clusters grandes (>20 docs)
           - Identifica tópicos latentes

        3. 'rake': Rapid Automatic Keyword Extraction
           - Extrae frases clave multi-palabra
           - Captura conceptos compuestos
           - Requiere rake-nltk

        Args:
            cluster_id: ID del cluster (1-indexed)
            n_topics: Número de tópicos/términos a extraer
            method: Método de extracción ('tfidf', 'lda', 'rake')

        Returns:
            Lista de tuplas (término, score) ordenadas por relevancia

        Example:
            >>> topics = analyzer.extract_cluster_topics(cluster_id=1, n_topics=5)
            >>> for term, score in topics:
            ...     print(f"{term}: {score:.3f}")
        """
        logger.info(f"\nExtrayendo tópicos del cluster {cluster_id} con método '{method}'...")

        # Obtener documentos del cluster
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            logger.warning(f"Cluster {cluster_id} está vacío")
            return []

        cluster_abstracts = [self.abstracts[i] for i in cluster_indices]

        if method == 'tfidf':
            return self._extract_tfidf_topics(cluster_indices, n_topics)

        elif method == 'lda':
            return self._extract_lda_topics(cluster_abstracts, n_topics)

        elif method == 'rake':
            if not RAKE_AVAILABLE:
                logger.warning("RAKE no disponible, usando TF-IDF")
                return self._extract_tfidf_topics(cluster_indices, n_topics)
            return self._extract_rake_topics(cluster_abstracts, n_topics)

        else:
            raise ValueError(f"Método desconocido: {method}")

    def _extract_tfidf_topics(self,
                             cluster_indices: np.ndarray,
                             n_topics: int) -> List[Tuple[str, float]]:
        """
        Extrae tópicos usando TF-IDF promedio del cluster.

        Metodología:
        ===========
        1. Calcula TF-IDF promedio de todos los documentos en el cluster
        2. Identifica términos con mayor TF-IDF promedio
        3. Estos términos son distintivos y frecuentes en el cluster

        Args:
            cluster_indices: Índices de documentos en el cluster
            n_topics: Número de términos a extraer

        Returns:
            Lista de (término, tfidf_score)
        """
        # TF-IDF del cluster
        cluster_tfidf = self.tfidf_matrix[cluster_indices]

        # Promedio de TF-IDF en el cluster
        tfidf_mean = cluster_tfidf.mean(axis=0).A1

        # Top términos
        top_indices = tfidf_mean.argsort()[-n_topics:][::-1]
        topics = [
            (self.feature_names[idx], tfidf_mean[idx])
            for idx in top_indices
        ]

        logger.info(f"Top {n_topics} términos TF-IDF:")
        for term, score in topics[:5]:
            logger.info(f"  {term}: {score:.4f}")

        return topics

    def _extract_lda_topics(self,
                           cluster_abstracts: List[str],
                           n_topics: int) -> List[Tuple[str, float]]:
        """
        Extrae tópicos usando Latent Dirichlet Allocation.

        LDA (Latent Dirichlet Allocation):
        ==================================
        Modelo probabilístico generativo que asume:
        - Cada documento es una mezcla de tópicos
        - Cada tópico es una distribución sobre palabras

        Para un documento d y tópico z:
        P(w|d) = Σ P(w|z) × P(z|d)

        Parámetros del modelo:
        - α: parámetro de concentración Dirichlet para distribución de tópicos
        - β: parámetro de concentración Dirichlet para distribución de palabras

        Args:
            cluster_abstracts: Textos del cluster
            n_topics: Número de tópicos a extraer

        Returns:
            Lista de (término, probabilidad)
        """
        if len(cluster_abstracts) < 5:
            logger.warning("Cluster muy pequeño para LDA, usando TF-IDF")
            cluster_indices = np.where(self.cluster_labels ==
                                      self.cluster_labels[np.where(
                                          [a in cluster_abstracts for a in self.abstracts]
                                      )[0][0]])[0]
            return self._extract_tfidf_topics(cluster_indices, n_topics)

        # Vectorizar con CountVectorizer (LDA usa conteos, no TF-IDF)
        count_vectorizer = CountVectorizer(
            max_features=200,
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        doc_term_matrix = count_vectorizer.fit_transform(cluster_abstracts)
        feature_names = count_vectorizer.get_feature_names_out()

        # LDA con 1 tópico (resumen del cluster)
        lda = LatentDirichletAllocation(
            n_components=1,
            max_iter=50,
            learning_method='batch',
            random_state=42
        )
        lda.fit(doc_term_matrix)

        # Obtener distribución de palabras del tópico
        topic_word_dist = lda.components_[0]
        top_indices = topic_word_dist.argsort()[-n_topics:][::-1]

        topics = [
            (feature_names[idx], topic_word_dist[idx])
            for idx in top_indices
        ]

        logger.info(f"Top {n_topics} términos LDA:")
        for term, score in topics[:5]:
            logger.info(f"  {term}: {score:.4f}")

        return topics

    def _extract_rake_topics(self,
                            cluster_abstracts: List[str],
                            n_topics: int) -> List[Tuple[str, float]]:
        """
        Extrae frases clave usando RAKE (Rapid Automatic Keyword Extraction).

        RAKE Algorithm:
        ==============
        1. Divide texto en candidatos (secuencias de palabras contenido)
        2. Para cada palabra, calcula:
           - freq(w): frecuencia de la palabra
           - deg(w): grado (co-ocurrencia con otras palabras)
        3. Score de palabra: deg(w) / freq(w)
        4. Score de frase: suma de scores de sus palabras

        Ventajas:
        - Extrae frases multi-palabra (ej: "deep neural network")
        - No requiere corpus de entrenamiento
        - Rápido y efectivo

        Args:
            cluster_abstracts: Textos del cluster
            n_topics: Número de frases clave a extraer

        Returns:
            Lista de (frase, score)
        """
        # Concatenar todos los abstracts del cluster
        cluster_text = ' '.join(cluster_abstracts)

        # RAKE
        rake = Rake()
        rake.extract_keywords_from_text(cluster_text)

        # Obtener frases con scores
        phrases_with_scores = rake.get_ranked_phrases_with_scores()

        # Ordenar por score descendente
        topics = sorted(phrases_with_scores, key=lambda x: x[0], reverse=True)[:n_topics]

        logger.info(f"Top {n_topics} frases clave RAKE:")
        for score, phrase in topics[:5]:
            logger.info(f"  {phrase}: {score:.4f}")

        # Invertir orden para consistencia con otros métodos
        return [(phrase, score) for score, phrase in topics]

    def generate_cluster_summary(self,
                                 cluster_id: int,
                                 docs_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Genera un resumen interpretable completo del cluster.

        El resumen incluye:
        =================
        1. Estadísticas básicas:
           - Número de documentos
           - Porcentaje del total

        2. Análisis temático:
           - Tema principal inferido
           - Top palabras clave (TF-IDF)
           - Frases clave (RAKE, si disponible)

        3. Documentos representativos:
           - Documentos más cercanos al centroide del cluster
           - Títulos más representativos

        4. Análisis temporal (si docs_data incluye 'Year'):
           - Rango de años
           - Distribución temporal
           - Año promedio

        5. Análisis de autores (si docs_data incluye 'Authors'):
           - Autores más frecuentes
           - Número de autores únicos

        Args:
            cluster_id: ID del cluster
            docs_data: DataFrame con metadata de documentos
                      Columnas opcionales: 'Title', 'Year', 'Authors', etc.

        Returns:
            Diccionario con información del cluster:
            {
                'cluster_id': int,
                'size': int,
                'percentage': float,
                'main_theme': str,
                'keywords': list,
                'key_phrases': list,
                'representative_docs': list,
                'temporal_info': dict (si disponible),
                'author_info': dict (si disponible)
            }

        Example:
            >>> summary = analyzer.generate_cluster_summary(1, docs_df)
            >>> print(f"Tema: {summary['main_theme']}")
            >>> print(f"Keywords: {summary['keywords'][:5]}")
        """
        logger.info("\n" + "="*70)
        logger.info(f"GENERANDO RESUMEN DEL CLUSTER {cluster_id}")
        logger.info("="*70)

        # Índices de documentos en el cluster
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            logger.warning(f"Cluster {cluster_id} está vacío")
            return {}

        summary = {
            'cluster_id': cluster_id,
            'size': len(cluster_indices),
            'percentage': (len(cluster_indices) / len(self.abstracts)) * 100
        }

        logger.info(f"Tamaño: {summary['size']} documentos ({summary['percentage']:.1f}%)")

        # 1. Análisis temático con TF-IDF
        tfidf_topics = self.extract_cluster_topics(cluster_id, n_topics=10, method='tfidf')
        summary['keywords'] = [term for term, _ in tfidf_topics]

        # Inferir tema principal (primeros 3 términos)
        main_theme = ', '.join(summary['keywords'][:3])
        summary['main_theme'] = main_theme.title()
        logger.info(f"Tema principal: {summary['main_theme']}")

        # 2. Frases clave con RAKE (si disponible)
        if RAKE_AVAILABLE:
            rake_phrases = self.extract_cluster_topics(cluster_id, n_topics=5, method='rake')
            summary['key_phrases'] = [phrase for phrase, _ in rake_phrases]
            logger.info(f"Frases clave: {summary['key_phrases'][:3]}")

        # 3. Documentos representativos (más cercanos al centroide)
        centroid = self.tfidf_matrix[cluster_indices].mean(axis=0).A1

        # Calcular distancia de cada documento al centroide
        from sklearn.metrics.pairwise import cosine_similarity
        distances = []
        for idx in cluster_indices:
            doc_vec = self.tfidf_matrix[idx].toarray()[0]
            similarity = cosine_similarity([doc_vec], [centroid])[0][0]
            distances.append((idx, similarity))

        # Ordenar por similitud descendente
        distances.sort(key=lambda x: x[1], reverse=True)

        # Top 5 documentos más representativos
        representative_indices = [idx for idx, _ in distances[:5]]
        summary['representative_doc_indices'] = representative_indices

        if docs_data is not None and 'Title' in docs_data.columns:
            representative_titles = [
                docs_data.iloc[idx]['Title'] for idx in representative_indices
            ]
            summary['representative_docs'] = representative_titles
            logger.info(f"\nDocumentos representativos:")
            for i, title in enumerate(representative_titles[:3], 1):
                logger.info(f"  {i}. {title}")

        # 4. Análisis temporal
        if docs_data is not None and 'Year' in docs_data.columns:
            cluster_years = docs_data.iloc[cluster_indices]['Year'].dropna()

            if len(cluster_years) > 0:
                summary['temporal_info'] = {
                    'year_range': (int(cluster_years.min()), int(cluster_years.max())),
                    'mean_year': float(cluster_years.mean()),
                    'year_distribution': cluster_years.value_counts().to_dict()
                }
                logger.info(f"\nRango temporal: {summary['temporal_info']['year_range'][0]} - {summary['temporal_info']['year_range'][1]}")
                logger.info(f"Año promedio: {summary['temporal_info']['mean_year']:.1f}")

        # 5. Análisis de autores
        if docs_data is not None and 'Authors' in docs_data.columns:
            cluster_authors = docs_data.iloc[cluster_indices]['Authors'].dropna()

            # Extraer todos los autores (asumiendo formato "Author1; Author2; ...")
            all_authors = []
            for authors_str in cluster_authors:
                if isinstance(authors_str, str):
                    authors = [a.strip() for a in authors_str.split(';')]
                    all_authors.extend(authors)

            if all_authors:
                author_counts = Counter(all_authors)
                top_authors = author_counts.most_common(5)

                summary['author_info'] = {
                    'n_unique_authors': len(author_counts),
                    'top_authors': dict(top_authors)
                }
                logger.info(f"\nAutores únicos: {summary['author_info']['n_unique_authors']}")
                logger.info(f"Top autores: {list(summary['author_info']['top_authors'].keys())[:3]}")

        logger.info("="*70)

        return summary

    def visualize_cluster_themes(self,
                                output_dir: str,
                                docs_data: Optional[pd.DataFrame] = None,
                                dim_reduction: str = 'tsne') -> None:
        """
        Genera visualizaciones temáticas de todos los clusters.

        Visualizaciones generadas:
        =========================
        1. Word clouds por cluster
           - Tamaño de palabra proporcional a TF-IDF
           - Colores distintivos por cluster

        2. Red de co-ocurrencia de términos
           - Nodos = términos frecuentes
           - Aristas = co-ocurrencia en documentos
           - Peso de arista = frecuencia de co-ocurrencia

        3. Distribución temporal de clusters (si hay datos de año)
           - Línea temporal por cluster
           - Identifica evolución de temas

        4. Mapa 2D de documentos
           - t-SNE o UMAP para reducción de dimensionalidad
           - Colores por cluster
           - Visualiza separación y cohesión

        5. Heatmap de similitud intra/inter-cluster
           - Similitud promedio dentro de cada cluster
           - Similitud entre clusters
           - Identifica clusters relacionados

        Args:
            output_dir: Directorio para guardar visualizaciones
            docs_data: DataFrame con metadata (opcional)
            dim_reduction: Método de reducción ('tsne' o 'umap')

        Example:
            >>> analyzer.visualize_cluster_themes(
            ...     'output/themes',
            ...     docs_df,
            ...     dim_reduction='tsne'
            ... )
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*70)
        logger.info("GENERANDO VISUALIZACIONES TEMÁTICAS")
        logger.info("="*70)
        logger.info(f"Output: {output_dir}")

        # 1. Word clouds por cluster
        logger.info("\n1. Generando word clouds...")
        self._plot_wordclouds(output_path)

        # 2. Red de co-ocurrencia
        logger.info("\n2. Generando red de co-ocurrencia...")
        self._plot_cooccurrence_network(output_path)

        # 3. Distribución temporal (si disponible)
        if docs_data is not None and 'Year' in docs_data.columns:
            logger.info("\n3. Generando distribución temporal...")
            self._plot_temporal_distribution(output_path, docs_data)

        # 4. Mapa 2D
        logger.info("\n4. Generando mapa 2D de documentos...")
        self._plot_2d_map(output_path, method=dim_reduction)

        # 5. Heatmap de similitud
        logger.info("\n5. Generando heatmap de similitud...")
        self._plot_similarity_heatmap(output_path)

        logger.info("\n" + "="*70)
        logger.info("VISUALIZACIONES COMPLETADAS")
        logger.info("="*70)

    def _plot_wordclouds(self, output_path: Path) -> None:
        """Genera word clouds para cada cluster."""
        n_cols = min(3, self.n_clusters)
        n_rows = int(np.ceil(self.n_clusters / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=150)

        if self.n_clusters == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for cluster_id in range(1, self.n_clusters + 1):
            ax = axes[cluster_id - 1]

            # Obtener términos del cluster
            topics = self.extract_cluster_topics(cluster_id, n_topics=50, method='tfidf')

            if len(topics) == 0:
                ax.axis('off')
                continue

            # Crear diccionario de frecuencias
            word_freq = {term: score for term, score in topics}

            # Word cloud
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color='white',
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(word_freq)

            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Cluster {cluster_id}\n({len(np.where(self.cluster_labels == cluster_id)[0])} docs)',
                        fontsize=14, fontweight='bold')
            ax.axis('off')

        # Ocultar axes sobrantes
        for idx in range(self.n_clusters, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_path / 'cluster_wordclouds.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Word clouds guardados: {output_path / 'cluster_wordclouds.png'}")

    def _plot_cooccurrence_network(self, output_path: Path, top_n: int = 30) -> None:
        """Genera red de co-ocurrencia de términos."""
        # Ajustar top_n al número disponible de features
        top_n = min(top_n, len(self.feature_names))

        # Obtener top términos globales
        global_tfidf = self.tfidf_matrix.mean(axis=0).A1
        top_indices = global_tfidf.argsort()[-top_n:][::-1]
        top_terms = [self.feature_names[idx] for idx in top_indices]

        # Calcular co-ocurrencia
        # Dos términos co-ocurren si aparecen en el mismo documento
        cooccurrence = np.zeros((top_n, top_n))

        for doc_idx in range(len(self.abstracts)):
            doc_vec = self.tfidf_matrix[doc_idx].toarray()[0]

            # Términos presentes en este documento
            present_terms = []
            for i, term_idx in enumerate(top_indices):
                if doc_vec[term_idx] > 0:
                    present_terms.append(i)

            # Incrementar co-ocurrencia
            for i in present_terms:
                for j in present_terms:
                    if i != j:
                        cooccurrence[i, j] += 1

        # Crear grafo
        G = nx.Graph()

        # Añadir nodos
        for term in top_terms:
            G.add_node(term)

        # Añadir aristas (solo si co-ocurrencia > threshold)
        threshold = np.percentile(cooccurrence[cooccurrence > 0], 50) if np.any(cooccurrence > 0) else 0

        for i in range(top_n):
            for j in range(i+1, top_n):
                if cooccurrence[i, j] > threshold:
                    G.add_edge(top_terms[i], top_terms[j], weight=cooccurrence[i, j])

        # Visualizar
        plt.figure(figsize=(16, 12), dpi=150)

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Dibujar nodos
        node_sizes = [global_tfidf[top_indices[i]] * 5000 for i in range(top_n)]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                              alpha=0.8, edgecolors='black', linewidths=1.5)

        # Dibujar aristas
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        if weights:
            max_weight = max(weights)
            widths = [w / max_weight * 3 for w in weights]
            nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray')

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        plt.title('Red de Co-ocurrencia de Términos', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / 'cooccurrence_network.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Red de co-ocurrencia guardada: {output_path / 'cooccurrence_network.png'}")

    def _plot_temporal_distribution(self, output_path: Path, docs_data: pd.DataFrame) -> None:
        """Genera distribución temporal de clusters."""
        if 'Year' not in docs_data.columns:
            logger.warning("Columna 'Year' no encontrada")
            return

        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

        for cluster_id in range(1, self.n_clusters + 1):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_years = docs_data.iloc[cluster_indices]['Year'].dropna()

            if len(cluster_years) > 0:
                year_counts = cluster_years.value_counts().sort_index()

                ax.plot(year_counts.index, year_counts.values,
                       marker='o', linewidth=2, markersize=6,
                       label=f'Cluster {cluster_id}', alpha=0.7)

        ax.set_xlabel('Año', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Documentos', fontsize=12, fontweight='bold')
        ax.set_title('Distribución Temporal de Clusters', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'temporal_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Distribución temporal guardada: {output_path / 'temporal_distribution.png'}")

    def _plot_2d_map(self, output_path: Path, method: str = 'tsne') -> None:
        """Genera mapa 2D de documentos con reducción de dimensionalidad."""
        logger.info(f"Reduciendo dimensionalidad con {method.upper()}...")

        # Convertir sparse matrix a dense
        X_dense = self.tfidf_matrix.toarray()

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_dense)-1))
            X_2d = reducer.fit_transform(X_dense)

        elif method == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP no disponible, usando t-SNE")
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_dense)-1))
                X_2d = reducer.fit_transform(X_dense)
            else:
                reducer = umap.UMAP(n_components=2, random_state=42)
                X_2d = reducer.fit_transform(X_dense)

        else:
            raise ValueError(f"Método desconocido: {method}")

        # Visualizar
        plt.figure(figsize=(12, 10), dpi=150)

        # Scatter plot por cluster
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = self.cluster_labels == cluster_id
            plt.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
                       label=f'Cluster {cluster_id}',
                       s=80, alpha=0.7, edgecolors='black', linewidths=0.5)

        plt.xlabel(f'{method.upper()} Component 1', fontsize=12, fontweight='bold')
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12, fontweight='bold')
        plt.title(f'Mapa 2D de Documentos ({method.upper()})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'2d_map_{method}.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Mapa 2D guardado: {output_path / f'2d_map_{method}.png'}")

    def _plot_similarity_heatmap(self, output_path: Path) -> None:
        """Genera heatmap de similitud intra/inter-cluster."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Calcular similitud promedio intra e inter-cluster
        similarity_matrix = np.zeros((self.n_clusters, self.n_clusters))

        for i in range(1, self.n_clusters + 1):
            cluster_i_indices = np.where(self.cluster_labels == i)[0]
            cluster_i_vecs = self.tfidf_matrix[cluster_i_indices]

            for j in range(1, self.n_clusters + 1):
                cluster_j_indices = np.where(self.cluster_labels == j)[0]
                cluster_j_vecs = self.tfidf_matrix[cluster_j_indices]

                # Similitud promedio entre todos los pares
                if len(cluster_i_indices) > 0 and len(cluster_j_indices) > 0:
                    sims = cosine_similarity(cluster_i_vecs, cluster_j_vecs)
                    similarity_matrix[i-1, j-1] = sims.mean()

        # Visualizar
        plt.figure(figsize=(10, 8), dpi=150)

        sns.heatmap(similarity_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='YlOrRd',
                   square=True,
                   xticklabels=[f'C{i}' for i in range(1, self.n_clusters + 1)],
                   yticklabels=[f'C{i}' for i in range(1, self.n_clusters + 1)],
                   cbar_kws={'label': 'Similitud Coseno Promedio'})

        plt.title('Heatmap de Similitud Intra/Inter-Cluster', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Cluster', fontsize=12, fontweight='bold')
        plt.ylabel('Cluster', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'similarity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Heatmap de similitud guardado: {output_path / 'similarity_heatmap.png'}")


def main():
    """Ejemplo de uso del analizador de temas de clusters."""

    print("\n" + "="*70)
    print(" EJEMPLO: Cluster Theme Analyzer")
    print("="*70)

    # Datos de ejemplo
    np.random.seed(42)

    abstracts = [
        "deep learning neural networks image classification computer vision",
        "convolutional neural networks object detection visual recognition",
        "recurrent neural networks sequence modeling time series",
        "natural language processing text classification sentiment analysis",
        "machine learning algorithms supervised learning classification",
        "reinforcement learning policy gradient deep reinforcement learning",
        "transfer learning domain adaptation pretrained models",
        "attention mechanisms transformer models bert gpt",
        "graph neural networks relational data knowledge graphs",
        "generative adversarial networks image synthesis generation",
    ]

    # Asignaciones de cluster (simuladas)
    cluster_labels = np.array([1, 1, 2, 2, 3, 3, 1, 2, 1, 1])

    # DataFrame con metadata
    docs_data = pd.DataFrame({
        'Title': [
            'Deep Learning for Image Classification',
            'CNNs for Object Detection',
            'RNNs for Sequence Modeling',
            'NLP and Text Classification',
            'Supervised Machine Learning',
            'Deep Reinforcement Learning',
            'Transfer Learning Methods',
            'Attention and Transformers',
            'Graph Neural Networks',
            'GANs for Image Generation'
        ],
        'Year': [2018, 2019, 2017, 2020, 2016, 2019, 2020, 2021, 2020, 2019],
        'Authors': [
            'Smith, J.; Doe, A.',
            'Johnson, M.',
            'Williams, R.; Brown, T.',
            'Davis, L.',
            'Miller, K.; Wilson, S.',
            'Moore, J.',
            'Taylor, A.; Anderson, B.',
            'Thomas, M.',
            'Jackson, W.',
            'White, H.; Harris, C.'
        ]
    })

    print(f"\nDocumentos: {len(abstracts)}")
    print(f"Clusters: {len(np.unique(cluster_labels))}")

    # Crear analizador
    analyzer = ClusterThemeAnalyzer(abstracts, cluster_labels)

    # 1. Extraer tópicos
    print("\n" + "="*70)
    print("1. EXTRACCIÓN DE TÓPICOS")
    print("="*70)

    for cluster_id in range(1, len(np.unique(cluster_labels)) + 1):
        print(f"\nCluster {cluster_id}:")
        topics = analyzer.extract_cluster_topics(cluster_id, n_topics=5, method='tfidf')
        for term, score in topics:
            print(f"  - {term}: {score:.3f}")

    # 2. Generar resumen
    print("\n" + "="*70)
    print("2. RESUMEN DE CLUSTERS")
    print("="*70)

    for cluster_id in range(1, len(np.unique(cluster_labels)) + 1):
        summary = analyzer.generate_cluster_summary(cluster_id, docs_data)

        print(f"\nCluster {cluster_id}: {summary['main_theme']}")
        print(f"  Documentos: {summary['size']}")
        print(f"  Keywords: {', '.join(summary['keywords'][:5])}")
        if 'temporal_info' in summary:
            print(f"  Años: {summary['temporal_info']['year_range']}")

    # 3. Visualizaciones
    print("\n" + "="*70)
    print("3. VISUALIZACIONES TEMÁTICAS")
    print("="*70)

    output_dir = 'output/cluster_themes'
    analyzer.visualize_cluster_themes(output_dir, docs_data, dim_reduction='tsne')

    print("\n" + "="*70)
    print(" EJEMPLO COMPLETADO")
    print("="*70)
    print(f"\nArchivos generados en: {output_dir}/")


if __name__ == "__main__":
    main()
