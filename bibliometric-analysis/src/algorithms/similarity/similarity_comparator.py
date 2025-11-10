"""
Similarity Comparator Module

Compares all 6 similarity algorithms on the same dataset and generates
comprehensive analysis including:
- Performance metrics (execution time, memory usage)
- Similarity matrices for each algorithm
- Comparative visualizations
- Correlation analysis between algorithms
- Detailed Markdown reports
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import time
from datetime import datetime
import tracemalloc

# Import all similarity algorithms
from src.algorithms.similarity.levenshtein_similarity import LevenshteinSimilarity
from src.algorithms.similarity.tfidf_cosine_similarity import TfidfCosineSimilarity
from src.algorithms.similarity.jaccard_similarity import JaccardSimilarity
from src.algorithms.similarity.ngram_similarity import CharacterNgramSimilarity
from src.algorithms.similarity.sbert_similarity import SBERTSimilarity
from src.algorithms.similarity.transformer_similarity import TransformerSimilarity


class SimilarityComparator:
    """
    Comprehensive similarity algorithm comparator

    Compares 6 different similarity algorithms:
    1. Levenshtein Distance/Similarity (edit distance)
    2. TF-IDF Cosine Similarity (term frequency)
    3. Jaccard Similarity (set-based)
    4. Character N-gram Similarity (fuzzy matching)
    5. SBERT Similarity (sentence embeddings)
    6. BERT Transformer Similarity (contextual embeddings)
    """

    def __init__(
        self,
        unified_data_path: Optional[str] = None,
        output_dir: str = 'outputs/similarity_analysis',
        enable_ai_models: bool = True,
        debug: bool = False
    ):
        """
        Initialize Similarity Comparator

        Carga datos unificados y inicializa modelos de IA con caché.

        Args:
            unified_data_path: Path to unified CSV data file (optional)
                              If provided, loads data for article selection
            output_dir: Directory for output files (reports, visualizations)
            enable_ai_models: Whether to include AI models (SBERT, BERT)
                             Set to False for faster testing with classical algorithms only
            debug: If True, enables debug logging

        Example:
            >>> # With data file
            >>> comparator = SimilarityComparator(
            ...     unified_data_path='data/processed/unified_data.csv'
            ... )
            >>> articles = comparator.select_articles([0, 1, 2, 3, 4])
            >>> results = comparator.compare_all_algorithms(articles)

            >>> # Without data file (for custom texts)
            >>> comparator = SimilarityComparator()
            >>> results = comparator.compare_all_algorithms(['text1', 'text2'])
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_ai_models = enable_ai_models
        self.debug = debug

        # Load unified data if provided
        self.unified_data = None
        self.unified_data_path = unified_data_path

        if unified_data_path:
            logger.info(f"Loading unified data from: {unified_data_path}")
            try:
                self.unified_data = pd.read_csv(unified_data_path)
                logger.success(f"Loaded {len(self.unified_data)} articles from unified data")
            except Exception as e:
                logger.error(f"Failed to load unified data: {e}")
                self.unified_data = None

        # Initialize algorithms
        logger.info("Initializing similarity algorithms...")

        self.algorithms = {}

        # Classical algorithms (always enabled)
        self.algorithms['levenshtein'] = LevenshteinSimilarity(debug=debug)
        self.algorithms['tfidf_cosine'] = TfidfCosineSimilarity(debug=debug)
        self.algorithms['jaccard'] = JaccardSimilarity(n=1, char_level=False, debug=debug)
        self.algorithms['ngram'] = CharacterNgramSimilarity(n=3, debug=debug)

        # AI-based algorithms (optional) - with caching
        if enable_ai_models:
            try:
                self.algorithms['sbert'] = SBERTSimilarity(debug=debug)
                logger.success("SBERT model loaded (cached for future use)")
            except Exception as e:
                logger.warning(f"Failed to load SBERT: {e}")

            try:
                self.algorithms['bert'] = TransformerSimilarity(debug=debug)
                logger.success("BERT model loaded (cached for future use)")
            except Exception as e:
                logger.warning(f"Failed to load BERT: {e}")

        self.results = {}

        logger.success(f"Initialized {len(self.algorithms)} similarity algorithms")

    def load_unified_data(
        self,
        filepath: str,
        text_column: str = 'abstract',
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load unified bibliographic data

        Args:
            filepath: Path to unified CSV file
            text_column: Column containing text to compare
            max_samples: Maximum number of samples to load (for testing)

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from: {filepath}")

        df = pd.read_csv(filepath)

        # Filter rows with non-empty text
        df = df[df[text_column].notna() & (df[text_column] != '')]

        if max_samples:
            df = df.head(max_samples)

        logger.info(f"Loaded {len(df)} records")
        return df

    def select_articles(
        self,
        article_ids: List[int],
        text_column: str = 'abstract'
    ) -> List[str]:
        """
        Selecciona artículos por ID y extrae abstracts

        Args:
            article_ids: Lista de índices o IDs de artículos a seleccionar
            text_column: Nombre de la columna con el texto (default: 'abstract')

        Returns:
            Lista de textos (abstracts) seleccionados

        Raises:
            ValueError: Si no hay datos cargados o los IDs son inválidos

        Example:
            >>> comparator = SimilarityComparator(
            ...     unified_data_path='data/processed/unified_data.csv'
            ... )
            >>> # Seleccionar los primeros 5 artículos
            >>> abstracts = comparator.select_articles([0, 1, 2, 3, 4])
            >>> print(f"Selected {len(abstracts)} abstracts")
            Selected 5 abstracts

            >>> # Seleccionar artículos específicos
            >>> abstracts = comparator.select_articles([10, 25, 42, 100])
        """
        if self.unified_data is None:
            raise ValueError(
                "No unified data loaded. Initialize with unified_data_path or call load_unified_data()"
            )

        logger.info(f"Selecting {len(article_ids)} articles from unified data")

        # Validate indices
        max_index = len(self.unified_data) - 1
        invalid_ids = [idx for idx in article_ids if idx < 0 or idx > max_index]

        if invalid_ids:
            raise ValueError(
                f"Invalid article IDs: {invalid_ids}. "
                f"Valid range: [0, {max_index}]"
            )

        # Check if text column exists
        if text_column not in self.unified_data.columns:
            raise ValueError(
                f"Column '{text_column}' not found in data. "
                f"Available columns: {list(self.unified_data.columns)}"
            )

        # Extract texts
        selected_texts = []
        for idx in article_ids:
            text = self.unified_data.iloc[idx][text_column]

            # Handle NaN or empty strings
            if pd.isna(text) or text == '':
                logger.warning(f"Article {idx} has empty {text_column}, using placeholder")
                text = f"[Empty {text_column} for article {idx}]"

            selected_texts.append(str(text))

        logger.success(f"Selected {len(selected_texts)} texts from column '{text_column}'")

        # Log sample info
        if self.debug:
            logger.debug(f"\nSelected articles:")
            for idx, text in zip(article_ids, selected_texts):
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.debug(f"  Article {idx}: {preview}")

        return selected_texts

    def compute_similarity_matrix(
        self,
        algorithm_name: str,
        texts: List[str]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute similarity matrix using specified algorithm

        Args:
            algorithm_name: Name of algorithm
            texts: List of text strings

        Returns:
            Tuple of (similarity_matrix, execution_time, memory_used)
        """
        logger.info(f"Computing similarity matrix with {algorithm_name}...")

        algorithm = self.algorithms[algorithm_name]

        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        # Compute matrix based on algorithm type
        if algorithm_name in ['sbert', 'bert', 'tfidf']:
            # These algorithms have built-in matrix computation
            similarity_matrix = algorithm.compute_similarity_matrix(texts)

        else:
            # Pairwise computation for other algorithms
            n = len(texts)
            similarity_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        sim = algorithm.compute_similarity(texts[i], texts[j])
                        similarity_matrix[i][j] = sim
                        similarity_matrix[j][i] = sim

        # Measure performance
        execution_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_used = peak / 1024 / 1024  # Convert to MB

        logger.success(f"{algorithm_name}: {execution_time:.2f}s, {memory_used:.2f}MB")

        return similarity_matrix, execution_time, memory_used

    def compare_all_algorithms(
        self,
        abstracts: List[str],
        article_ids: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta los 6 algoritmos y retorna resultados estructurados.

        Este método compara todos los abstracts usando los 6 algoritmos de similitud,
        mide tiempo de ejecución y uso de memoria, y retorna resultados en formato
        estructurado para fácil análisis y visualización.

        Args:
            abstracts: Lista de textos (abstracts) a comparar
            article_ids: IDs opcionales de los artículos (para tracking)

        Returns:
            Diccionario con estructura:
            {
                'levenshtein': np.ndarray,        # Matriz de similitud
                'tfidf_cosine': np.ndarray,       # Matriz de similitud
                'jaccard': np.ndarray,            # Matriz de similitud
                'ngram': np.ndarray,              # Matriz de similitud
                'sbert': np.ndarray,              # Matriz de similitud (si disponible)
                'bert': np.ndarray,               # Matriz de similitud (si disponible)
                'execution_times': {              # Tiempos de ejecución
                    'levenshtein': float (seconds),
                    'tfidf_cosine': float,
                    'jaccard': float,
                    'ngram': float,
                    'sbert': float,
                    'bert': float
                },
                'memory_usage': {                 # Uso de memoria
                    'levenshtein': float (MB),
                    'tfidf_cosine': float,
                    'jaccard': float,
                    'ngram': float,
                    'sbert': float,
                    'bert': float
                },
                'metadata': {                     # Metadata adicional
                    'n_texts': int,
                    'article_ids': list,
                    'timestamp': str,
                    'algorithms_used': list
                },
                'statistics': {                   # Estadísticas por algoritmo
                    'levenshtein': {'mean': float, 'std': float, ...},
                    ...
                }
            }

        Example:
            >>> comparator = SimilarityComparator(
            ...     unified_data_path='data/processed/unified_data.csv',
            ...     enable_ai_models=True
            ... )
            >>> # Seleccionar artículos
            >>> abstracts = comparator.select_articles([0, 1, 2, 3, 4])
            >>> # Comparar con todos los algoritmos
            >>> results = comparator.compare_all_algorithms(abstracts)
            >>> # Acceder a resultados
            >>> print(f"SBERT similarity matrix shape: {results['sbert'].shape}")
            >>> print(f"TF-IDF execution time: {results['execution_times']['tfidf_cosine']:.2f}s")
            >>> print(f"Algorithms used: {results['metadata']['algorithms_used']}")
        """
        logger.info("=" * 70)
        logger.info(f"COMPARING ALL ALGORITHMS ON {len(abstracts)} ABSTRACTS")
        logger.info("=" * 70)

        if article_ids is None:
            article_ids = list(range(len(abstracts)))

        # Initialize result structure
        results = {
            'execution_times': {},
            'memory_usage': {},
            'metadata': {
                'n_texts': len(abstracts),
                'article_ids': article_ids,
                'timestamp': datetime.now().isoformat(),
                'algorithms_used': list(self.algorithms.keys())
            },
            'statistics': {}
        }

        # Compute similarity matrices for each algorithm
        for algo_name in self.algorithms.keys():
            logger.info(f"Running {algo_name.upper()} algorithm...")

            matrix, exec_time, memory = self.compute_similarity_matrix(
                algo_name,
                abstracts
            )

            # Store matrix directly with algorithm name as key
            results[algo_name] = matrix

            # Store execution time
            results['execution_times'][algo_name] = exec_time

            # Store memory usage
            results['memory_usage'][algo_name] = memory

            # Compute and store statistics
            results['statistics'][algo_name] = {
                'mean_similarity': float(matrix.mean()),
                'std_similarity': float(matrix.std()),
                'min_similarity': float(matrix.min()),
                'max_similarity': float(np.max(matrix - np.eye(len(abstracts)))),  # Max off-diagonal
                'median_similarity': float(np.median(matrix))
            }

            logger.success(
                f"{algo_name}: {exec_time:.2f}s, {memory:.2f}MB, "
                f"mean_sim={results['statistics'][algo_name]['mean_similarity']:.4f}"
            )

        # Store in instance for later use
        self.results = results

        logger.success("=" * 70)
        logger.success(f"All {len(self.algorithms)} algorithms compared successfully!")
        logger.success("=" * 70)

        return results

    def visualize_results(
        self,
        results: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Genera visualizaciones comprehensivas de los resultados.

        Este método crea 4 tipos de visualizaciones principales:
        1. Heatmaps de similitud para cada algoritmo
        2. Gráfico comparativo de tiempos de ejecución
        3. Gráfico de uso de memoria
        4. Tabla comparativa de estadísticas

        Args:
            results: Diccionario de resultados de compare_all_algorithms()
            output_dir: Directorio para guardar visualizaciones (opcional)
                       Si None, usa self.output_dir

        Returns:
            Dict[str, plt.Figure]: Diccionario con objetos Figure de matplotlib
            {
                'similarity_matrices': Figure,  # Heatmaps de todos los algoritmos
                'performance': Figure,          # Tiempos y memoria
                'correlation': Figure,          # Correlación entre algoritmos
                'distributions': Figure         # Distribuciones de similitud
            }

        Example:
            >>> comparator = SimilarityComparator(
            ...     unified_data_path='data/processed/unified_data.csv'
            ... )
            >>> abstracts = comparator.select_articles([0, 1, 2, 3, 4])
            >>> results = comparator.compare_all_algorithms(abstracts)
            >>> # Generar visualizaciones
            >>> figures = comparator.visualize_results(
            ...     results,
            ...     output_dir='outputs/my_analysis'
            ... )
            >>> # Mostrar una figura específica
            >>> figures['similarity_matrices'].show()
        """
        logger.info("Generating visualizations from results...")

        # Use provided output_dir or instance's output_dir
        if output_dir:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir

        # Set results for internal plotting methods
        old_results = self.results
        self.results = results

        try:
            figures = {}

            # 1. Similarity matrices heatmaps
            logger.info("Creating similarity matrices heatmaps...")
            fig1 = self._plot_similarity_matrices_v2(results)
            figures['similarity_matrices'] = fig1

            # 2. Performance comparison (execution times + memory)
            logger.info("Creating performance comparison...")
            fig2 = self._plot_performance_v2(results)
            figures['performance'] = fig2

            # 3. Algorithm correlation
            logger.info("Creating algorithm correlation matrix...")
            fig3 = self._plot_correlation_v2(results)
            figures['correlation'] = fig3

            # 4. Distributions
            logger.info("Creating similarity distributions...")
            fig4 = self._plot_distributions_v2(results)
            figures['distributions'] = fig4

            # Save all figures
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for name, fig in figures.items():
                filepath = save_dir / f'{name}_{timestamp}.png'
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.success(f"Saved: {filepath}")

            logger.success(f"Generated {len(figures)} visualizations")
            return figures

        finally:
            # Restore original results
            self.results = old_results

    def _plot_similarity_matrices_v2(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot heatmaps of similarity matrices (version 2 for new results format)"""
        # Get algorithm names (exclude metadata keys)
        algo_names = [k for k in results.keys()
                     if k not in ['execution_times', 'memory_usage', 'metadata', 'statistics']]

        n_algorithms = len(algo_names)
        n_cols = 3
        n_rows = (n_algorithms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_algorithms > 1 else [axes]

        for idx, algo_name in enumerate(algo_names):
            matrix = results[algo_name]
            stats = results['statistics'][algo_name]

            sns.heatmap(
                matrix,
                ax=axes[idx],
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Similarity'},
                annot=len(matrix) <= 10  # Only annotate if small matrix
            )

            axes[idx].set_title(
                f'{algo_name.upper()}\n'
                f'Mean: {stats["mean_similarity"]:.3f}, '
                f'Time: {results["execution_times"][algo_name]:.2f}s'
            )

        # Hide extra subplots
        for idx in range(n_algorithms, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def _plot_performance_v2(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot performance metrics (version 2)"""
        algo_names = list(results['execution_times'].keys())
        exec_times = [results['execution_times'][name] for name in algo_names]
        memories = [results['memory_usage'][name] for name in algo_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Execution time
        bars1 = ax1.barh(algo_names, exec_times, color='steelblue')
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_title('Algorithm Execution Time Comparison')
        ax1.grid(axis='x', alpha=0.3)

        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}s',
                    ha='left', va='center', fontsize=9)

        # Memory usage
        bars2 = ax2.barh(algo_names, memories, color='coral')
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_title('Algorithm Memory Usage Comparison')
        ax2.grid(axis='x', alpha=0.3)

        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}MB',
                    ha='left', va='center', fontsize=9)

        plt.tight_layout()
        return fig

    def _plot_correlation_v2(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot correlation between algorithms (version 2)"""
        algo_names = [k for k in results.keys()
                     if k not in ['execution_times', 'memory_usage', 'metadata', 'statistics']]
        n_algos = len(algo_names)

        # Create correlation matrix
        correlation_matrix = np.zeros((n_algos, n_algos))

        for i, algo1 in enumerate(algo_names):
            for j, algo2 in enumerate(algo_names):
                matrix1 = results[algo1]
                matrix2 = results[algo2]

                flat1 = matrix1.flatten()
                flat2 = matrix2.flatten()

                correlation_matrix[i][j] = np.corrcoef(flat1, flat2)[0, 1]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            ax=ax,
            xticklabels=algo_names,
            yticklabels=algo_names,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Pearson Correlation'}
        )

        ax.set_title('Algorithm Correlation Matrix\n(How similarly do algorithms rank text pairs?)')

        plt.tight_layout()
        return fig

    def _plot_distributions_v2(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot similarity score distributions (version 2)"""
        algo_names = [k for k in results.keys()
                     if k not in ['execution_times', 'memory_usage', 'metadata', 'statistics']]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, algo_name in enumerate(algo_names):
            if idx >= 6:  # Max 6 subplots
                break

            matrix = results[algo_name]
            stats = results['statistics'][algo_name]

            # Get off-diagonal values
            n = matrix.shape[0]
            off_diagonal = matrix[~np.eye(n, dtype=bool)]

            axes[idx].hist(off_diagonal, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Similarity Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(
                f'{algo_name.upper()}\n'
                f'Mean: {stats["mean_similarity"]:.3f}, Std: {stats["std_similarity"]:.3f}'
            )
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide extra subplots
        for idx in range(len(algo_names), 6):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def generate_visualizations(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate comparison visualizations

        Args:
            save: Whether to save figures to disk

        Returns:
            Dictionary of figure objects
        """
        logger.info("Generating visualizations...")

        if not self.results:
            raise ValueError("No results available. Run compare_all_algorithms() first.")

        figures = {}

        # 1. Similarity matrices heatmaps
        fig1 = self._plot_similarity_matrices()
        figures['similarity_matrices'] = fig1

        # 2. Performance comparison
        fig2 = self._plot_performance_comparison()
        figures['performance'] = fig2

        # 3. Algorithm correlation
        fig3 = self._plot_algorithm_correlation()
        figures['correlation'] = fig3

        # 4. Distribution comparison
        fig4 = self._plot_similarity_distributions()
        figures['distributions'] = fig4

        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for name, fig in figures.items():
                filepath = self.output_dir / f'{name}_{timestamp}.png'
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved: {filepath}")

        logger.success(f"Generated {len(figures)} visualizations")
        return figures

    def _plot_similarity_matrices(self) -> plt.Figure:
        """Plot heatmaps of similarity matrices for all algorithms"""

        n_algorithms = len(self.results['algorithms'])
        n_cols = 3
        n_rows = (n_algorithms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_algorithms > 1 else [axes]

        for idx, (algo_name, algo_data) in enumerate(self.results['algorithms'].items()):
            matrix = algo_data['similarity_matrix']

            sns.heatmap(
                matrix,
                ax=axes[idx],
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Similarity'},
                annot=len(matrix) <= 10  # Only annotate if small matrix
            )

            axes[idx].set_title(f'{algo_name.upper()}\n'
                               f'Mean: {algo_data["mean_similarity"]:.3f}, '
                               f'Time: {algo_data["execution_time"]:.2f}s')

        # Hide extra subplots
        for idx in range(n_algorithms, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def _plot_performance_comparison(self) -> plt.Figure:
        """Plot performance metrics comparison"""

        algo_names = list(self.results['algorithms'].keys())
        exec_times = [self.results['algorithms'][name]['execution_time'] for name in algo_names]
        memories = [self.results['algorithms'][name]['memory_mb'] for name in algo_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Execution time
        bars1 = ax1.barh(algo_names, exec_times, color='steelblue')
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_title('Algorithm Execution Time')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}s',
                    ha='left', va='center', fontsize=9)

        # Memory usage
        bars2 = ax2.barh(algo_names, memories, color='coral')
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_title('Algorithm Memory Usage')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}MB',
                    ha='left', va='center', fontsize=9)

        plt.tight_layout()
        return fig

    def _plot_algorithm_correlation(self) -> plt.Figure:
        """Plot correlation between algorithms"""

        algo_names = list(self.results['algorithms'].keys())
        n_algos = len(algo_names)

        # Create correlation matrix
        correlation_matrix = np.zeros((n_algos, n_algos))

        for i, algo1 in enumerate(algo_names):
            for j, algo2 in enumerate(algo_names):
                matrix1 = self.results['algorithms'][algo1]['similarity_matrix']
                matrix2 = self.results['algorithms'][algo2]['similarity_matrix']

                # Flatten and compute correlation
                flat1 = matrix1.flatten()
                flat2 = matrix2.flatten()

                correlation_matrix[i][j] = np.corrcoef(flat1, flat2)[0, 1]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            ax=ax,
            xticklabels=algo_names,
            yticklabels=algo_names,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Pearson Correlation'}
        )

        ax.set_title('Algorithm Correlation Matrix\n(How similarly do algorithms rank text pairs?)')

        plt.tight_layout()
        return fig

    def _plot_similarity_distributions(self) -> plt.Figure:
        """Plot similarity score distributions"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (algo_name, algo_data) in enumerate(self.results['algorithms'].items()):
            matrix = algo_data['similarity_matrix']

            # Get off-diagonal values (actual comparisons, not self-similarity)
            n = matrix.shape[0]
            off_diagonal = matrix[~np.eye(n, dtype=bool)]

            axes[idx].hist(off_diagonal, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Similarity Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{algo_name.upper()}\n'
                               f'Mean: {off_diagonal.mean():.3f}, Std: {off_diagonal.std():.3f}')
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide extra subplot if needed
        if len(self.results['algorithms']) < 6:
            for idx in range(len(self.results['algorithms']), 6):
                axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def generate_report(self, format: str = 'markdown') -> str:
        """
        Generate comprehensive comparison report

        Args:
            format: Report format ('markdown', 'txt', or 'html')

        Returns:
            Report string
        """
        logger.info(f"Generating {format} report...")

        if not self.results:
            raise ValueError("No results available. Run compare_all_algorithms() first.")

        if format == 'markdown':
            report = self._generate_markdown_report()
        elif format == 'txt':
            report = self._generate_text_report()
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f'similarity_comparison_report_{timestamp}.{format[:2]}'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.success(f"Report saved: {filepath}")

        return report

    def _generate_markdown_report(self) -> str:
        """Generate detailed Markdown report"""

        report = f"""# Similarity Algorithms Comparison Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: {len(self.results['texts'])} texts
**Algorithms**: {len(self.results['algorithms'])}

---

## 📊 Executive Summary

This report compares {len(self.results['algorithms'])} different text similarity algorithms
on a dataset of {len(self.results['texts'])} texts.

### Algorithms Compared:

1. **Levenshtein Similarity** - Edit distance (character-level)
2. **TF-IDF Cosine Similarity** - Term frequency-inverse document frequency
3. **Jaccard Similarity** - Set-based comparison (word-level)
4. **Character N-gram Similarity** - Fuzzy matching (character trigrams)
"""

        if 'sbert' in self.results['algorithms']:
            report += "5. **SBERT Similarity** - Sentence-BERT embeddings (semantic)\n"

        if 'bert' in self.results['algorithms']:
            report += "6. **BERT Similarity** - BERT transformer embeddings (contextual)\n"

        report += "\n---\n\n## ⚡ Performance Metrics\n\n"
        report += "| Algorithm | Execution Time | Memory Usage | Mean Similarity | Std Dev |\n"
        report += "|-----------|----------------|--------------|-----------------|----------|\n"

        for algo_name, algo_data in self.results['algorithms'].items():
            report += f"| {algo_name.upper():15s} | "
            report += f"{algo_data['execution_time']:>13.3f}s | "
            report += f"{algo_data['memory_mb']:>11.2f}MB | "
            report += f"{algo_data['mean_similarity']:>14.4f} | "
            report += f"{algo_data['std_similarity']:>7.4f} |\n"

        report += "\n---\n\n## 📈 Statistical Analysis\n\n"

        for algo_name, algo_data in self.results['algorithms'].items():
            report += f"### {algo_name.upper()}\n\n"
            report += f"- **Mean Similarity**: {algo_data['mean_similarity']:.4f}\n"
            report += f"- **Std Deviation**: {algo_data['std_similarity']:.4f}\n"
            report += f"- **Min Similarity**: {algo_data['min_similarity']:.4f}\n"
            report += f"- **Max Similarity**: {algo_data['max_similarity']:.4f}\n"
            report += f"- **Execution Time**: {algo_data['execution_time']:.3f} seconds\n"
            report += f"- **Memory Used**: {algo_data['memory_mb']:.2f} MB\n"
            report += "\n"

        report += "---\n\n## 🔗 Algorithm Correlations\n\n"
        report += "Pearson correlation between algorithms:\n\n"

        algo_names = list(self.results['algorithms'].keys())
        report += "| | " + " | ".join([f"{a.upper()}" for a in algo_names]) + " |\n"
        report += "|" + "|".join(["-" * 10 for _ in range(len(algo_names) + 1)]) + "|\n"

        for i, algo1 in enumerate(algo_names):
            row = f"| {algo1.upper():8s} |"
            for j, algo2 in enumerate(algo_names):
                matrix1 = self.results['algorithms'][algo1]['similarity_matrix']
                matrix2 = self.results['algorithms'][algo2]['similarity_matrix']
                corr = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
                row += f" {corr:>7.3f} |"
            report += row + "\n"

        report += "\n---\n\n## 💡 Recommendations\n\n"

        # Find best algorithm by different criteria
        fastest = min(self.results['algorithms'].items(), key=lambda x: x[1]['execution_time'])
        lowest_memory = min(self.results['algorithms'].items(), key=lambda x: x[1]['memory_mb'])

        report += f"- **Fastest**: {fastest[0].upper()} ({fastest[1]['execution_time']:.3f}s)\n"
        report += f"- **Lowest Memory**: {lowest_memory[0].upper()} ({lowest_memory[1]['memory_mb']:.2f}MB)\n"

        report += "\n### Use Case Recommendations:\n\n"
        report += "- **Typo Detection**: Levenshtein or Character N-gram\n"
        report += "- **Semantic Similarity**: SBERT or BERT (if available)\n"
        report += "- **Fast Lexical Matching**: Jaccard or TF-IDF\n"
        report += "- **Balanced Performance**: TF-IDF or Jaccard\n"

        report += "\n---\n\n## 📝 Methodology\n\n"
        report += "Each algorithm computed pairwise similarity scores for all text pairs.\n"
        report += "Performance measured using:\n"
        report += "- **Execution Time**: Python `time.perf_counter()`\n"
        report += "- **Memory Usage**: Python `tracemalloc` module\n"
        report += "- **Similarity Metrics**: Algorithm-specific scores normalized to [0, 1]\n"

        report += "\n---\n\n*End of Report*\n"

        return report

    def _generate_text_report(self) -> str:
        """Generate plain text report"""
        # Similar to markdown but without markdown formatting
        return self._generate_markdown_report().replace('#', '').replace('*', '').replace('|', ' ')


# Example usage
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/similarity_comparator.log",
        rotation="10 MB",
        level="INFO"
    )

    print("=" * 70)
    print("SIMILARITY COMPARATOR DEMONSTRATION")
    print("=" * 70)

    # Create comparator
    comparator = SimilarityComparator(
        output_dir='outputs/similarity_analysis',
        enable_ai_models=True,  # Set to False for faster testing
        debug=False
    )

    # Example texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand human language",
        "Computer vision allows machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties"
    ]

    print(f"\nComparing {len(texts)} texts across {len(comparator.algorithms)} algorithms...")

    # Compare all algorithms
    results = comparator.compare_all_algorithms(texts)

    # Generate visualizations
    print("\nGenerating visualizations...")
    figures = comparator.generate_visualizations(save=True)

    # Generate report
    print("\nGenerating comparison report...")
    report = comparator.generate_report(format='markdown')

    print("\n" + "=" * 70)
    print(f"Analysis complete! Check: {comparator.output_dir}")
    print("=" * 70)
