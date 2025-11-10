"""
Term Visualization Module
Genera visualizaciones avanzadas para análisis de términos con normalización robusta.

Características:
- 6 tipos de visualizaciones especializadas
- Normalización robusta con spaCy
- Manejo de plurales/singulares
- Stopwords personalizadas del dominio
- Logging detallado
- Progress bars
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import spacy
from tqdm import tqdm
from wordcloud import WordCloud
from matplotlib_venn import venn2, venn3
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TermNormalizer:
    """
    Normaliza términos compuestos de manera robusta.
    Maneja plurales, singulares, lematización y variaciones.
    """

    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Inicializa el normalizador con spaCy.

        Args:
            model_name: Nombre del modelo spaCy a usar
        """
        logger.info(f"Cargando modelo spaCy: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Modelo {model_name} no encontrado. Descargarlo con: python -m spacy download {model_name}")
            raise

        # Configurar pipeline para mejor rendimiento
        if 'parser' in self.nlp.pipe_names:
            self.nlp.disable_pipes(['parser'])
        if 'ner' in self.nlp.pipe_names:
            self.nlp.disable_pipes(['ner'])

        logger.info("TermNormalizer inicializado exitosamente")

    def normalize_term(self, term: str) -> str:
        """
        Normaliza un término individual.

        Args:
            term: Término a normalizar

        Returns:
            Término normalizado (lematizado, lowercase)
        """
        doc = self.nlp(term.lower())

        # Lematizar cada token
        lemmas = [token.lemma_ for token in doc if not token.is_punct]

        normalized = ' '.join(lemmas)

        logger.debug(f"Normalizado: '{term}' -> '{normalized}'")

        return normalized

    def normalize_terms(self, terms: List[str], show_progress: bool = True) -> List[str]:
        """
        Normaliza múltiples términos.

        Args:
            terms: Lista de términos
            show_progress: Mostrar progress bar

        Returns:
            Lista de términos normalizados
        """
        logger.info(f"Normalizando {len(terms)} términos...")

        normalized = []

        iterator = tqdm(terms, desc="Normalizando") if show_progress else terms

        for term in iterator:
            normalized.append(self.normalize_term(term))

        logger.info(f"Normalización completada: {len(normalized)} términos")

        return normalized

    def normalize_with_mapping(self, terms: List[str]) -> Dict[str, str]:
        """
        Normaliza términos y retorna mapeo original -> normalizado.

        Args:
            terms: Lista de términos

        Returns:
            Diccionario {término_original: término_normalizado}
        """
        mapping = {}

        for term in tqdm(terms, desc="Normalizando con mapeo"):
            normalized = self.normalize_term(term)
            mapping[term] = normalized

        return mapping


class DomainStopwords:
    """
    Gestiona stopwords personalizadas del dominio académico.
    """

    # Stopwords académicas generales
    ACADEMIC_STOPWORDS = {
        'paper', 'study', 'research', 'article', 'approach', 'method',
        'result', 'results', 'finding', 'findings', 'analysis', 'analyses',
        'data', 'show', 'shows', 'presented', 'present', 'propose',
        'proposed', 'work', 'works', 'based', 'using', 'used',
        'application', 'applications', 'system', 'systems',
        'problem', 'problems', 'solution', 'solutions',
        'model', 'models', 'technique', 'techniques',
        'novel', 'new', 'presented', 'discuss', 'discussed',
        'provide', 'provides', 'demonstrated', 'demonstrate',
        'review', 'overview', 'survey', 'introduction',
        'conclusion', 'conclusions', 'summary', 'abstract',
        'section', 'sections', 'figure', 'figures', 'table', 'tables',
        'example', 'examples', 'case', 'cases', 'experiment', 'experiments'
    }

    def __init__(self, additional_stopwords: Optional[Set[str]] = None):
        """
        Inicializa stopwords.

        Args:
            additional_stopwords: Stopwords adicionales específicas
        """
        self.stopwords = self.ACADEMIC_STOPWORDS.copy()

        if additional_stopwords:
            self.stopwords.update(additional_stopwords)

        logger.info(f"Stopwords inicializadas: {len(self.stopwords)} palabras")

    def is_stopword(self, term: str) -> bool:
        """
        Verifica si un término es stopword.

        Args:
            term: Término a verificar

        Returns:
            True si es stopword
        """
        # Normalizar para comparación
        term_lower = term.lower().strip()

        # Verificar término completo
        if term_lower in self.stopwords:
            return True

        # Verificar cada palabra del término
        words = term_lower.split()
        if len(words) == 1 and words[0] in self.stopwords:
            return True

        return False

    def filter_terms(self, terms: List[str]) -> List[str]:
        """
        Filtra stopwords de una lista de términos.

        Args:
            terms: Lista de términos

        Returns:
            Lista filtrada
        """
        filtered = [t for t in terms if not self.is_stopword(t)]

        logger.info(f"Filtrados: {len(terms)} -> {len(filtered)} términos")

        return filtered

    def add_stopwords(self, words: Set[str]):
        """Añade stopwords adicionales."""
        self.stopwords.update(words)
        logger.info(f"Añadidas {len(words)} stopwords. Total: {len(self.stopwords)}")


class TermVisualizer:
    """
    Genera visualizaciones avanzadas para análisis de términos.
    """

    def __init__(self,
                 normalizer: Optional[TermNormalizer] = None,
                 stopwords: Optional[DomainStopwords] = None):
        """
        Inicializa el visualizador.

        Args:
            normalizer: Normalizador de términos (se crea uno si no se proporciona)
            stopwords: Gestor de stopwords (se crea uno si no se proporciona)
        """
        self.normalizer = normalizer or TermNormalizer()
        self.stopwords = stopwords or DomainStopwords()

        # Configurar estilo de visualizaciones
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        logger.info("TermVisualizer inicializado")

    def plot_term_frequencies(self,
                             frequencies: Dict[str, int],
                             top_n: int = 20,
                             title: str = "Frecuencias de Términos",
                             output_path: Optional[str] = None,
                             normalize: bool = True) -> plt.Figure:
        """
        1. Gráfico de barras de frecuencias de términos.

        Args:
            frequencies: Diccionario {término: frecuencia}
            top_n: Número de términos top a mostrar
            title: Título del gráfico
            output_path: Ruta para guardar (opcional)
            normalize: Normalizar términos antes de graficar

        Returns:
            Figura de matplotlib
        """
        logger.info(f"Generando gráfico de frecuencias (top {top_n})...")

        # Normalizar si se solicita
        if normalize:
            logger.info("Normalizando términos...")
            freq_normalized = {}
            for term, freq in tqdm(frequencies.items(), desc="Normalizando"):
                norm_term = self.normalizer.normalize_term(term)
                freq_normalized[norm_term] = freq_normalized.get(norm_term, 0) + freq
            frequencies = freq_normalized

        # Filtrar stopwords
        frequencies = {
            term: freq
            for term, freq in frequencies.items()
            if not self.stopwords.is_stopword(term)
        }

        # Obtener top N
        top_terms = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if not top_terms:
            logger.warning("No hay términos para graficar después del filtrado")
            return None

        terms, freqs = zip(*top_terms)

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 8))

        # Gráfico de barras horizontales
        y_pos = np.arange(len(terms))
        bars = ax.barh(y_pos, freqs, color='steelblue', alpha=0.8, edgecolor='navy')

        # Añadir valores en las barras
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            ax.text(freq, i, f' {freq}', va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {output_path}")

        return fig

    def plot_cooccurrence_heatmap(self,
                                 cooccurrences: Dict[Tuple[str, str], int],
                                 top_n: int = 15,
                                 title: str = "Matriz de Co-ocurrencia de Términos",
                                 output_path: Optional[str] = None) -> plt.Figure:
        """
        2. Heatmap de co-ocurrencia de términos.

        Args:
            cooccurrences: Diccionario {(term1, term2): count}
            top_n: Número de términos top a incluir
            title: Título del gráfico
            output_path: Ruta para guardar

        Returns:
            Figura de matplotlib
        """
        logger.info(f"Generando heatmap de co-ocurrencia (top {top_n})...")

        # Obtener términos más frecuentes
        term_counts = Counter()
        for (t1, t2), count in cooccurrences.items():
            term_counts[t1] += count
            term_counts[t2] += count

        top_terms = [term for term, _ in term_counts.most_common(top_n)]

        # Crear matriz de co-ocurrencia
        matrix = np.zeros((len(top_terms), len(top_terms)))

        for i, term1 in enumerate(top_terms):
            for j, term2 in enumerate(top_terms):
                if i == j:
                    continue

                # Buscar co-ocurrencia (en ambas direcciones)
                count = cooccurrences.get((term1, term2), 0)
                count += cooccurrences.get((term2, term1), 0)

                matrix[i, j] = count

        # Crear heatmap
        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(
            matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Co-ocurrencias'},
            ax=ax,
            square=True
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap guardado en: {output_path}")

        return fig

    def plot_venn_diagram(self,
                         set1: Set[str],
                         set2: Set[str],
                         set3: Optional[Set[str]] = None,
                         labels: Tuple[str, ...] = ('Set 1', 'Set 2', 'Set 3'),
                         title: str = "Overlap de Términos",
                         output_path: Optional[str] = None) -> plt.Figure:
        """
        3. Diagrama de Venn para visualizar overlap entre conjuntos.

        Args:
            set1: Primer conjunto de términos
            set2: Segundo conjunto de términos
            set3: Tercer conjunto opcional
            labels: Etiquetas para cada conjunto
            title: Título del gráfico
            output_path: Ruta para guardar

        Returns:
            Figura de matplotlib
        """
        logger.info("Generando diagrama de Venn...")

        fig, ax = plt.subplots(figsize=(10, 8))

        if set3 is None:
            # Venn de 2 conjuntos
            venn2([set1, set2], set_labels=labels[:2], ax=ax)
        else:
            # Venn de 3 conjuntos
            venn3([set1, set2, set3], set_labels=labels, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagrama de Venn guardado en: {output_path}")

        return fig

    def plot_wordcloud(self,
                      term_scores: Dict[str, float],
                      title: str = "Word Cloud de Términos",
                      output_path: Optional[str] = None,
                      max_words: int = 100,
                      background_color: str = 'white',
                      colormap: str = 'viridis') -> plt.Figure:
        """
        4. Word cloud de términos (tamaño proporcional a score).

        Args:
            term_scores: Diccionario {término: score}
            title: Título del gráfico
            output_path: Ruta para guardar
            max_words: Número máximo de palabras
            background_color: Color de fondo
            colormap: Mapa de colores

        Returns:
            Figura de matplotlib
        """
        logger.info(f"Generando word cloud (max {max_words} palabras)...")

        # Filtrar stopwords
        term_scores_filtered = {
            term: score
            for term, score in term_scores.items()
            if not self.stopwords.is_stopword(term)
        }

        if not term_scores_filtered:
            logger.warning("No hay términos para el word cloud")
            return None

        # Crear word cloud
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(term_scores_filtered)

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 9))

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Word cloud guardado en: {output_path}")

        return fig

    def plot_similarity_matrix(self,
                              terms: List[str],
                              similarity_matrix: np.ndarray,
                              title: str = "Matriz de Similitud Semántica",
                              output_path: Optional[str] = None,
                              max_terms: int = 20) -> plt.Figure:
        """
        5. Matriz de similitud semántica entre términos.

        Args:
            terms: Lista de términos
            similarity_matrix: Matriz de similitud (terms x terms)
            title: Título del gráfico
            output_path: Ruta para guardar
            max_terms: Número máximo de términos a mostrar

        Returns:
            Figura de matplotlib
        """
        logger.info(f"Generando matriz de similitud (max {max_terms} términos)...")

        # Limitar términos si es necesario
        if len(terms) > max_terms:
            terms = terms[:max_terms]
            similarity_matrix = similarity_matrix[:max_terms, :max_terms]

        # Acortar nombres de términos para visualización
        term_labels = [t[:30] + '...' if len(t) > 30 else t for t in terms]

        # Crear heatmap
        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(
            similarity_matrix,
            xticklabels=term_labels,
            yticklabels=term_labels,
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Similitud'},
            ax=ax,
            square=True
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de similitud guardada en: {output_path}")

        return fig

    def plot_method_comparison(self,
                              methods_data: Dict[str, Dict[str, float]],
                              title: str = "Comparación de Métodos de Extracción",
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        6. Gráfico de comparación entre métodos de extracción.

        Args:
            methods_data: Dict {method_name: {metric: value}}
                Ejemplo: {'TF-IDF': {'precision': 0.8, 'recall': 0.7, ...}}
            title: Título del gráfico
            output_path: Ruta para guardar

        Returns:
            Figura de matplotlib
        """
        logger.info("Generando comparación de métodos...")

        # Convertir a DataFrame para facilitar plotting
        df = pd.DataFrame(methods_data).T

        # Crear figura con subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico 1: Métricas principales (barras agrupadas)
        main_metrics = ['precision', 'recall', 'f1_score']
        main_metrics = [m for m in main_metrics if m in df.columns]

        if main_metrics:
            df_main = df[main_metrics]
            df_main.plot(kind='bar', ax=axes[0], width=0.8, alpha=0.8)
            axes[0].set_title('Métricas de Evaluación', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Método', fontsize=11)
            axes[0].set_ylabel('Score', fontsize=11)
            axes[0].set_ylim(0, 1)
            axes[0].legend(title='Métrica')
            axes[0].grid(axis='y', alpha=0.3)
            axes[0].set_xticklabels(df_main.index, rotation=45, ha='right')

        # Gráfico 2: Radar chart o métricas adicionales
        other_metrics = [col for col in df.columns if col not in main_metrics]

        if other_metrics:
            df_other = df[other_metrics]
            df_other.plot(kind='bar', ax=axes[1], width=0.8, alpha=0.8)
            axes[1].set_title('Otras Métricas', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Método', fontsize=11)
            axes[1].set_ylabel('Valor', fontsize=11)
            axes[1].legend(title='Métrica')
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].set_xticklabels(df_other.index, rotation=45, ha='right')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparación de métodos guardada en: {output_path}")

        return fig

    def generate_all_visualizations(self,
                                   data: Dict,
                                   output_dir: str = './visualizations') -> Dict[str, str]:
        """
        Genera todas las visualizaciones disponibles.

        Args:
            data: Diccionario con todos los datos necesarios
            output_dir: Directorio donde guardar las visualizaciones

        Returns:
            Diccionario con rutas de archivos generados
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Generando todas las visualizaciones en: {output_dir}")

        outputs = {}

        # 1. Frecuencias
        if 'frequencies' in data:
            path = os.path.join(output_dir, 'term_frequencies.png')
            self.plot_term_frequencies(data['frequencies'], output_path=path)
            outputs['frequencies'] = path

        # 2. Co-ocurrencia
        if 'cooccurrences' in data:
            path = os.path.join(output_dir, 'cooccurrence_heatmap.png')
            self.plot_cooccurrence_heatmap(data['cooccurrences'], output_path=path)
            outputs['cooccurrence'] = path

        # 3. Venn diagram
        if 'sets' in data and len(data['sets']) >= 2:
            path = os.path.join(output_dir, 'venn_diagram.png')
            sets = data['sets']
            if len(sets) == 2:
                self.plot_venn_diagram(
                    sets[0], sets[1],
                    labels=data.get('set_labels', ('Set 1', 'Set 2')),
                    output_path=path
                )
            else:
                self.plot_venn_diagram(
                    sets[0], sets[1], sets[2],
                    labels=data.get('set_labels', ('Set 1', 'Set 2', 'Set 3')),
                    output_path=path
                )
            outputs['venn'] = path

        # 4. Word cloud
        if 'term_scores' in data:
            path = os.path.join(output_dir, 'wordcloud.png')
            self.plot_wordcloud(data['term_scores'], output_path=path)
            outputs['wordcloud'] = path

        # 5. Similarity matrix
        if 'similarity_matrix' in data and 'terms' in data:
            path = os.path.join(output_dir, 'similarity_matrix.png')
            self.plot_similarity_matrix(
                data['terms'],
                data['similarity_matrix'],
                output_path=path
            )
            outputs['similarity'] = path

        # 6. Method comparison
        if 'methods_comparison' in data:
            path = os.path.join(output_dir, 'methods_comparison.png')
            self.plot_method_comparison(data['methods_comparison'], output_path=path)
            outputs['comparison'] = path

        logger.info(f"✓ Generadas {len(outputs)} visualizaciones")

        return outputs


def main():
    """Ejemplo de uso del módulo de visualizaciones."""

    print("\n" + "="*70)
    print(" EJEMPLO: Term Visualization Module")
    print("="*70)

    # Datos de ejemplo
    frequencies = {
        'deep learning': 45,
        'neural networks': 38,
        'machine learning': 52,
        'computer vision': 28,
        'natural language processing': 25,
        'convolutional networks': 20,
        'paper': 15,  # Será filtrado como stopword
        'result': 10   # Será filtrado como stopword
    }

    cooccurrences = {
        ('deep learning', 'neural networks'): 15,
        ('deep learning', 'machine learning'): 20,
        ('neural networks', 'computer vision'): 12,
        ('machine learning', 'computer vision'): 10,
        ('natural language processing', 'neural networks'): 8
    }

    # Crear visualizador
    visualizer = TermVisualizer()

    # 1. Gráfico de frecuencias
    print("\n[1/6] Generando gráfico de frecuencias...")
    visualizer.plot_term_frequencies(frequencies, top_n=10, normalize=True)
    plt.show()

    # 2. Heatmap de co-ocurrencia
    print("[2/6] Generando heatmap de co-ocurrencia...")
    visualizer.plot_cooccurrence_heatmap(cooccurrences, top_n=5)
    plt.show()

    # 3. Diagrama de Venn
    print("[3/6] Generando diagrama de Venn...")
    set1 = {'deep learning', 'neural networks', 'machine learning'}
    set2 = {'machine learning', 'computer vision', 'image recognition'}
    set3 = {'natural language processing', 'machine learning', 'deep learning'}

    visualizer.plot_venn_diagram(
        set1, set2, set3,
        labels=('AI Methods', 'Vision', 'NLP')
    )
    plt.show()

    # 4. Word cloud
    print("[4/6] Generando word cloud...")
    term_scores = frequencies  # Usar frecuencias como scores
    visualizer.plot_wordcloud(term_scores, max_words=50)
    plt.show()

    # 5. Matriz de similitud (ejemplo simulado)
    print("[5/6] Generando matriz de similitud...")
    terms = list(frequencies.keys())[:5]
    similarity_matrix = np.random.rand(5, 5)
    # Hacer simétrica
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    np.fill_diagonal(similarity_matrix, 1.0)

    visualizer.plot_similarity_matrix(terms, similarity_matrix)
    plt.show()

    # 6. Comparación de métodos
    print("[6/6] Generando comparación de métodos...")
    methods_data = {
        'TF-IDF': {'precision': 0.75, 'recall': 0.68, 'f1_score': 0.71},
        'RAKE': {'precision': 0.65, 'recall': 0.72, 'f1_score': 0.68},
        'TextRank': {'precision': 0.70, 'recall': 0.75, 'f1_score': 0.72}
    }

    visualizer.plot_method_comparison(methods_data)
    plt.show()

    print("\n" + "="*70)
    print(" ✓ TODAS LAS VISUALIZACIONES GENERADAS")
    print("="*70)


if __name__ == "__main__":
    main()
