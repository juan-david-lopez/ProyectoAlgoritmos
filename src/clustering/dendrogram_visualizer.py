"""
Dendrogram Visualizer Module
Generación de dendrogramas estáticos e interactivos para clustering jerárquico.

Características:
- Dendrogramas estáticos de alta calidad con matplotlib
- Dendrogramas interactivos con Plotly (zoom, pan, tooltips)
- Comparación visual de múltiples métodos de linkage
- Anotación automática de clusters
- Truncamiento inteligente de etiquetas largas
- Sugerencia automática de corte óptimo
- Colores distintivos por cluster
- Exportación a PNG, PDF, SVG, HTML

Uso típico:
===========
1. Ejecutar clustering jerárquico
2. Crear visualizador con linkage matrix y labels
3. Generar dendrogramas estáticos y/o interactivos
4. Comparar diferentes métodos visualmente
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Scipy para dendrogramas
from scipy.cluster.hierarchy import dendrogram, fcluster

# Plotly para visualizaciones interactivas
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')


class DendrogramVisualizer:
    """
    Visualizador de dendrogramas para clustering jerárquico.

    Genera visualizaciones estáticas e interactivas de la jerarquía de clusters,
    facilitando el análisis y la interpretación de los resultados.

    Attributes:
        linkage_matrix: Matriz de linkage de scipy (n_samples-1, 4)
        labels: Etiquetas de los documentos/puntos
        n_samples: Número de muestras
        truncate_length: Longitud máxima para etiquetas (default: 50)
    """

    def __init__(self,
                 linkage_matrix: np.ndarray,
                 labels: List[str],
                 truncate_length: int = 50):
        """
        Inicializa el visualizador con matriz de linkage y etiquetas.

        Args:
            linkage_matrix: Matriz de linkage de scipy (n_samples-1, 4)
            labels: Lista de etiquetas para los puntos (ej: títulos de artículos)
            truncate_length: Longitud máxima para truncar etiquetas largas

        Raises:
            ValueError: Si los parámetros son inválidos

        Example:
            >>> from hierarchical_clustering import HierarchicalClustering
            >>> hc = HierarchicalClustering(distance_matrix, labels)
            >>> Z = hc.average_linkage()
            >>> viz = DendrogramVisualizer(Z, labels)
            >>> viz.plot_static_dendrogram('dendrogram.png', 'Average Linkage')
        """
        # Validar linkage matrix
        self._validate_linkage_matrix(linkage_matrix)
        self.linkage_matrix = linkage_matrix

        # Número de muestras = filas en linkage + 1
        self.n_samples = len(linkage_matrix) + 1

        # Validar labels
        if len(labels) != self.n_samples:
            raise ValueError(
                f"Número de labels ({len(labels)}) no coincide con "
                f"número de muestras ({self.n_samples})"
            )
        self.labels = labels
        self.truncate_length = truncate_length

        # Truncar labels largos
        self.truncated_labels = [
            self._truncate_label(label) for label in labels
        ]

        logger.info("="*70)
        logger.info("DENDROGRAM VISUALIZER INICIALIZADO")
        logger.info("="*70)
        logger.info(f"Número de muestras: {self.n_samples}")
        logger.info(f"Labels truncados a: {truncate_length} caracteres")
        logger.info(f"Altura mínima: {linkage_matrix[:, 2].min():.4f}")
        logger.info(f"Altura máxima: {linkage_matrix[:, 2].max():.4f}")

    def plot_static_dendrogram(self,
                               output_path: str,
                               method_name: str = 'Hierarchical Clustering',
                               figsize: Tuple[int, int] = (20, 10),
                               color_threshold: Optional[float] = None,
                               n_clusters_suggest: Optional[int] = None,
                               dpi: int = 300) -> None:
        """
        Genera dendrograma estático de alta calidad con matplotlib.

        Características del dendrograma:
        ================================
        - Colores diferentes para clusters principales
        - Etiquetas truncadas y legibles
        - Línea horizontal para corte sugerido (si se especifica n_clusters)
        - Grid para facilitar lectura de alturas
        - Alta resolución para publicación
        - Título informativo
        - Leyenda de colores

        Args:
            output_path: Ruta donde guardar la imagen (PNG, PDF, SVG soportados)
            method_name: Nombre del método ('Single Linkage', 'Average Linkage', etc.)
            figsize: Tamaño de la figura (ancho, alto) en pulgadas
            color_threshold: Umbral de altura para colorear clusters.
                           Si None, se calcula automáticamente como 70% de altura máxima
            n_clusters_suggest: Si se especifica, dibuja línea horizontal en la altura
                              que produce ese número de clusters
            dpi: Resolución en DPI (300 recomendado para publicación)

        Example:
            >>> viz.plot_static_dendrogram(
            ...     'output/dendrogram.png',
            ...     'Average Linkage',
            ...     n_clusters_suggest=5
            ... )
        """
        logger.info("\n" + "="*70)
        logger.info("GENERANDO DENDROGRAMA ESTÁTICO")
        logger.info("="*70)
        logger.info(f"Método: {method_name}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Tamaño: {figsize}")
        logger.info(f"DPI: {dpi}")

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Determinar color threshold
        if color_threshold is None:
            # 70% de altura máxima (regla heurística común)
            color_threshold = 0.7 * self.linkage_matrix[:, 2].max()

        logger.info(f"Color threshold: {color_threshold:.4f}")

        # Generar dendrograma
        dendro = dendrogram(
            self.linkage_matrix,
            labels=self.truncated_labels,
            ax=ax,
            color_threshold=color_threshold,
            above_threshold_color='gray',
            leaf_rotation=90,
            leaf_font_size=8
        )

        # Título
        ax.set_title(
            f'Dendrograma - {method_name}\n'
            f'({self.n_samples} documentos)',
            fontsize=16,
            fontweight='bold',
            pad=20
        )

        # Labels
        ax.set_xlabel('Documentos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distancia / Altura', fontsize=12, fontweight='bold')

        # Grid para facilitar lectura
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # Línea horizontal para corte sugerido
        if n_clusters_suggest is not None:
            # Calcular altura que produce n_clusters
            cut_height = self._calculate_cut_height(n_clusters_suggest)

            ax.axhline(
                y=cut_height,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Corte sugerido: {n_clusters_suggest} clusters'
            )

            logger.info(f"Corte sugerido para {n_clusters_suggest} clusters: altura={cut_height:.4f}")

            # Añadir anotación
            ax.text(
                0.02, cut_height,
                f' {n_clusters_suggest} clusters',
                transform=ax.get_yaxis_transform(),
                fontsize=10,
                color='red',
                fontweight='bold',
                verticalalignment='bottom'
            )

        # Leyenda
        ax.legend(loc='upper right', fontsize=10)

        # Ajustar layout para evitar cortes
        plt.tight_layout()

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Dendrograma guardado en: {output_path}")

        plt.close()

        logger.info("="*70)
        logger.info("DENDROGRAMA ESTÁTICO COMPLETADO")
        logger.info("="*70)

    def plot_interactive_dendrogram(self,
                                   output_path: str,
                                   method_name: str = 'Hierarchical Clustering',
                                   height: int = 800) -> None:
        """
        Genera dendrograma interactivo con Plotly.

        Características interactivas:
        ============================
        - Zoom y pan para explorar detalles
        - Hover tooltips con información del documento
        - Selección y descarga de imagen
        - Responsive (se adapta al tamaño de ventana)
        - Colores distintivos por cluster
        - Se guarda como HTML standalone (se puede abrir en navegador)

        Args:
            output_path: Ruta donde guardar el HTML
            method_name: Nombre del método de linkage
            height: Altura de la figura en píxeles

        Example:
            >>> viz.plot_interactive_dendrogram(
            ...     'output/dendrogram_interactive.html',
            ...     'Average Linkage'
            ... )
        """
        logger.info("\n" + "="*70)
        logger.info("GENERANDO DENDROGRAMA INTERACTIVO (PLOTLY)")
        logger.info("="*70)
        logger.info(f"Método: {method_name}")
        logger.info(f"Output: {output_path}")

        # Crear dendrograma con plotly figure_factory
        fig = ff.create_dendrogram(
            self.linkage_matrix,
            orientation='bottom',
            labels=self.truncated_labels,
            linkagefun=lambda x: self.linkage_matrix  # Ya tenemos linkage matrix
        )

        # Actualizar layout
        fig.update_layout(
            title={
                'text': f'Dendrograma Interactivo - {method_name}<br>'
                       f'<sub>({self.n_samples} documentos)</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis={
                'title': 'Documentos',
                'tickangle': -90,
                'tickfont': {'size': 8}
            },
            yaxis={
                'title': 'Distancia / Altura',
                'gridcolor': 'lightgray'
            },
            height=height,
            hovermode='closest',
            plot_bgcolor='white',
            showlegend=False
        )

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar como HTML
        fig.write_html(output_path)
        logger.info(f"Dendrograma interactivo guardado en: {output_path}")

        logger.info("="*70)
        logger.info("DENDROGRAMA INTERACTIVO COMPLETADO")
        logger.info("="*70)

    def plot_comparison_dendrograms(self,
                                   linkage_matrices: Dict[str, np.ndarray],
                                   output_path: str,
                                   figsize: Tuple[int, int] = (24, 8),
                                   dpi: int = 300) -> None:
        """
        Genera figura con múltiples dendrogramas lado a lado para comparación.

        Útil para comparar visualmente diferentes métodos de linkage
        (Single, Complete, Average) en el mismo dataset.

        Args:
            linkage_matrices: Diccionario con matrices de linkage
                            {'single': Z1, 'complete': Z2, 'average': Z3}
            output_path: Ruta donde guardar la imagen
            figsize: Tamaño de la figura (ancho, alto)
            dpi: Resolución

        Example:
            >>> # Asumiendo que tenemos las matrices
            >>> linkage_mats = {
            ...     'Single': Z_single,
            ...     'Complete': Z_complete,
            ...     'Average': Z_average
            ... }
            >>> viz.plot_comparison_dendrograms(
            ...     linkage_mats,
            ...     'output/comparison.png'
            ... )
        """
        logger.info("\n" + "="*70)
        logger.info("GENERANDO COMPARACIÓN DE DENDROGRAMAS")
        logger.info("="*70)
        logger.info(f"Métodos a comparar: {list(linkage_matrices.keys())}")
        logger.info(f"Output: {output_path}")

        n_methods = len(linkage_matrices)

        # Crear subplots
        fig, axes = plt.subplots(1, n_methods, figsize=figsize, dpi=dpi)

        # Si solo hay un método, axes no es array
        if n_methods == 1:
            axes = [axes]

        # Generar dendrograma para cada método
        for idx, (method_name, Z) in enumerate(linkage_matrices.items()):
            ax = axes[idx]

            # Color threshold automático
            color_threshold = 0.7 * Z[:, 2].max()

            # Dendrograma
            dendrogram(
                Z,
                labels=self.truncated_labels,
                ax=ax,
                color_threshold=color_threshold,
                above_threshold_color='gray',
                leaf_rotation=90,
                leaf_font_size=6
            )

            # Título
            ax.set_title(
                f'{method_name}\n({self.n_samples} docs)',
                fontsize=14,
                fontweight='bold'
            )

            # Labels
            ax.set_xlabel('Documentos', fontsize=10)
            if idx == 0:
                ax.set_ylabel('Distancia / Altura', fontsize=10)
            else:
                ax.set_ylabel('')

            # Grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)

        # Título general
        fig.suptitle(
            'Comparación de Métodos de Linkage',
            fontsize=18,
            fontweight='bold',
            y=1.02
        )

        # Ajustar layout
        plt.tight_layout()

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Comparación guardada en: {output_path}")

        plt.close()

        logger.info("="*70)
        logger.info("COMPARACIÓN COMPLETADA")
        logger.info("="*70)

    def annotate_clusters(self,
                         n_clusters: int,
                         include_labels: bool = True) -> Dict[int, Dict]:
        """
        Identifica y anota clusters principales cortando el dendrograma.

        Útil para:
        - Obtener asignación de documentos a clusters
        - Analizar composición de cada cluster
        - Generar reportes de clusters

        Args:
            n_clusters: Número de clusters deseado
            include_labels: Si True, incluye las etiquetas completas en el resultado

        Returns:
            Diccionario con información de cada cluster:
            {
                1: {
                    'size': 15,
                    'indices': [0, 2, 5, ...],
                    'labels': ['Doc1', 'Doc3', ...] (si include_labels=True)
                },
                2: {...},
                ...
            }

        Example:
            >>> clusters_info = viz.annotate_clusters(n_clusters=5)
            >>> for cluster_id, info in clusters_info.items():
            ...     print(f"Cluster {cluster_id}: {info['size']} documentos")
            ...     print(f"  Documentos: {info['labels'][:3]}...")
        """
        logger.info("\n" + "="*70)
        logger.info("ANOTANDO CLUSTERS")
        logger.info("="*70)
        logger.info(f"Número de clusters: {n_clusters}")

        # Obtener asignaciones de clusters
        cluster_assignments = fcluster(
            self.linkage_matrix,
            n_clusters,
            criterion='maxclust'
        )

        logger.info(f"Asignaciones calculadas: {len(cluster_assignments)}")

        # Organizar por cluster
        clusters_info = {}

        for cluster_id in range(1, n_clusters + 1):
            # Índices de documentos en este cluster
            indices = np.where(cluster_assignments == cluster_id)[0].tolist()

            # Información del cluster
            cluster_info = {
                'size': len(indices),
                'indices': indices
            }

            # Incluir labels si se solicita
            if include_labels:
                cluster_info['labels'] = [self.labels[i] for i in indices]

            clusters_info[cluster_id] = cluster_info

            logger.info(f"Cluster {cluster_id}: {len(indices)} documentos")

        logger.info("="*70)
        logger.info("ANOTACIÓN COMPLETADA")
        logger.info("="*70)

        return clusters_info

    def plot_cluster_sizes(self,
                          n_clusters: int,
                          output_path: str,
                          figsize: Tuple[int, int] = (10, 6),
                          dpi: int = 300) -> None:
        """
        Genera gráfico de barras con el tamaño de cada cluster.

        Args:
            n_clusters: Número de clusters
            output_path: Ruta donde guardar la imagen
            figsize: Tamaño de la figura
            dpi: Resolución

        Example:
            >>> viz.plot_cluster_sizes(5, 'output/cluster_sizes.png')
        """
        logger.info("\n" + "="*70)
        logger.info("GENERANDO GRÁFICO DE TAMAÑOS DE CLUSTERS")
        logger.info("="*70)

        # Obtener información de clusters
        clusters_info = self.annotate_clusters(n_clusters, include_labels=False)

        # Extraer tamaños
        cluster_ids = sorted(clusters_info.keys())
        sizes = [clusters_info[cid]['size'] for cid in cluster_ids]

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Gráfico de barras
        bars = ax.bar(cluster_ids, sizes, color='steelblue', alpha=0.8, edgecolor='black')

        # Añadir valores encima de las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        # Títulos y labels
        ax.set_title(
            f'Distribución de Documentos por Cluster\n({n_clusters} clusters)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Número de Documentos', fontsize=12)

        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # Ajustar eje x
        ax.set_xticks(cluster_ids)

        # Layout
        plt.tight_layout()

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Gráfico guardado en: {output_path}")

        plt.close()

        logger.info("="*70)
        logger.info("GRÁFICO DE TAMAÑOS COMPLETADO")
        logger.info("="*70)

    def suggest_n_clusters(self,
                          max_clusters: int = 10,
                          method: str = 'elbow') -> int:
        """
        Sugiere número óptimo de clusters usando heurísticas.

        Métodos disponibles:
        ===================
        - 'elbow': Busca el "codo" en la curva de distancias de fusión
        - 'gap': Busca el mayor gap entre fusiones consecutivas

        Args:
            max_clusters: Número máximo de clusters a considerar
            method: Método heurístico ('elbow' o 'gap')

        Returns:
            Número sugerido de clusters

        Example:
            >>> n_optimal = viz.suggest_n_clusters(max_clusters=10)
            >>> print(f"Número sugerido de clusters: {n_optimal}")
        """
        logger.info("\n" + "="*70)
        logger.info("SUGIRIENDO NÚMERO ÓPTIMO DE CLUSTERS")
        logger.info("="*70)
        logger.info(f"Método: {method}")
        logger.info(f"Max clusters: {max_clusters}")

        # Distancias de fusión (ordenadas de menor a mayor)
        merge_distances = self.linkage_matrix[:, 2]

        if method == 'gap':
            # Método del gap: buscar el mayor salto entre fusiones
            # Últimas fusiones tienen distancias más grandes
            last_merges = merge_distances[-(max_clusters-1):]

            # Calcular gaps
            gaps = np.diff(last_merges)

            # Índice del mayor gap
            max_gap_idx = np.argmax(gaps)

            # Número de clusters = número de pasos antes del mayor gap + 2
            n_suggested = max_gap_idx + 2

            logger.info(f"Mayor gap encontrado entre fusión {max_gap_idx} y {max_gap_idx+1}")
            logger.info(f"Tamaño del gap: {gaps[max_gap_idx]:.4f}")

        elif method == 'elbow':
            # Método del codo: usar regla heurística
            # Típicamente, buscar donde la tasa de cambio disminuye significativamente
            last_merges = merge_distances[-(max_clusters-1):]

            # Calcular segunda derivada (curvatura)
            first_diff = np.diff(last_merges)
            second_diff = np.diff(first_diff)

            # El codo está donde la curvatura es máxima
            elbow_idx = np.argmax(np.abs(second_diff)) + 1
            n_suggested = elbow_idx + 2

            logger.info(f"Codo encontrado en índice {elbow_idx}")

        else:
            raise ValueError(f"Método desconocido: {method}")

        # Limitar al rango válido
        n_suggested = max(2, min(n_suggested, max_clusters))

        logger.info(f"Número sugerido de clusters: {n_suggested}")
        logger.info("="*70)

        return n_suggested

    def _calculate_cut_height(self, n_clusters: int) -> float:
        """
        Calcula la altura a la que cortar el dendrograma para obtener n_clusters.

        Args:
            n_clusters: Número deseado de clusters

        Returns:
            Altura de corte

        Internal method.
        """
        # Las distancias de fusión están en orden creciente
        # Para obtener k clusters, cortamos en la distancia de la fusión (n-k)
        # donde n es el número de muestras

        if n_clusters >= self.n_samples:
            return 0.0

        if n_clusters <= 1:
            return self.linkage_matrix[-1, 2] * 1.1

        # Índice de la fusión que produce n_clusters
        fusion_idx = self.n_samples - n_clusters - 1

        # Altura de esa fusión
        cut_height = self.linkage_matrix[fusion_idx, 2]

        # Altura de la siguiente fusión
        if fusion_idx < len(self.linkage_matrix) - 1:
            next_height = self.linkage_matrix[fusion_idx + 1, 2]
            # Cortar en el punto medio
            cut_height = (cut_height + next_height) / 2
        else:
            # Si es la última fusión, añadir un pequeño margen
            cut_height = cut_height * 1.01

        return cut_height

    def _truncate_label(self, label: str) -> str:
        """
        Trunca etiquetas largas para mejorar legibilidad.

        Args:
            label: Etiqueta original

        Returns:
            Etiqueta truncada

        Internal method.
        """
        if len(label) <= self.truncate_length:
            return label

        # Truncar y añadir '...'
        return label[:self.truncate_length-3] + '...'

    def _validate_linkage_matrix(self, linkage_matrix: np.ndarray) -> None:
        """
        Valida que la matriz de linkage sea válida.

        Args:
            linkage_matrix: Matriz a validar

        Raises:
            ValueError: Si la matriz es inválida

        Internal method.
        """
        if not isinstance(linkage_matrix, np.ndarray):
            raise ValueError("linkage_matrix debe ser un numpy.ndarray")

        if linkage_matrix.ndim != 2:
            raise ValueError(
                f"linkage_matrix debe ser 2D, recibido: {linkage_matrix.ndim}D"
            )

        if linkage_matrix.shape[1] != 4:
            raise ValueError(
                f"linkage_matrix debe tener 4 columnas, recibido: {linkage_matrix.shape[1]}"
            )

        if linkage_matrix.shape[0] == 0:
            raise ValueError("linkage_matrix está vacía")

        if not np.isfinite(linkage_matrix).all():
            raise ValueError("linkage_matrix contiene valores infinitos o NaN")

        logger.debug(f"Linkage matrix validada: {linkage_matrix.shape}")


def main():
    """Ejemplo de uso del visualizador de dendrogramas."""

    print("\n" + "="*70)
    print(" EJEMPLO: Dendrogram Visualizer")
    print("="*70)

    # Primero necesitamos crear un clustering jerárquico
    from hierarchical_clustering import HierarchicalClustering
    from distance_calculator import DistanceCalculator

    # Crear datos de ejemplo
    np.random.seed(42)

    # Simular vectores TF-IDF para 10 documentos
    n_docs = 10
    n_features = 20
    vectors = np.random.rand(n_docs, n_features)

    # Normalizar
    from sklearn.preprocessing import normalize
    vectors = normalize(vectors, norm='l2')

    # Labels de ejemplo
    labels = [
        "Deep Learning for NLP: A Comprehensive Survey",
        "Neural Networks and Natural Language Processing",
        "Machine Learning Fundamentals and Applications",
        "Sentiment Analysis using LSTM Networks",
        "Text Classification with Transformers",
        "Graph Neural Networks: A Review",
        "Convolutional Networks for Image Recognition",
        "Transfer Learning in Computer Vision",
        "Reinforcement Learning: Theory and Practice",
        "Federated Learning for Privacy-Preserving ML"
    ]

    print(f"\nDocumentos de ejemplo: {n_docs}")
    print(f"Features: {n_features}")

    # Calcular matriz de distancia
    dist_calc = DistanceCalculator()
    distance_matrix = dist_calc.cosine_distance(vectors)

    print(f"\nMatriz de distancia calculada: {distance_matrix.shape}")

    # Clustering jerárquico
    hc = HierarchicalClustering(distance_matrix, labels)

    # Ejecutar diferentes métodos
    print("\n" + "="*70)
    print("EJECUTANDO CLUSTERING")
    print("="*70)

    Z_single = hc.single_linkage()
    Z_complete = hc.complete_linkage()
    Z_average = hc.average_linkage()

    # Crear visualizador
    print("\n" + "="*70)
    print("CREANDO VISUALIZACIONES")
    print("="*70)

    viz = DendrogramVisualizer(Z_average, labels, truncate_length=40)

    # 1. Dendrograma estático
    print("\n1. Dendrograma estático con corte sugerido...")
    viz.plot_static_dendrogram(
        'output/dendrogram_static.png',
        'Average Linkage',
        n_clusters_suggest=3
    )

    # 2. Dendrograma interactivo
    print("\n2. Dendrograma interactivo...")
    viz.plot_interactive_dendrogram(
        'output/dendrogram_interactive.html',
        'Average Linkage'
    )

    # 3. Comparación de métodos
    print("\n3. Comparación de métodos...")
    linkage_matrices = {
        'Single': Z_single,
        'Complete': Z_complete,
        'Average': Z_average
    }

    viz.plot_comparison_dendrograms(
        linkage_matrices,
        'output/comparison_dendrograms.png'
    )

    # 4. Anotar clusters
    print("\n4. Anotando clusters...")
    clusters_info = viz.annotate_clusters(n_clusters=3)

    for cluster_id, info in clusters_info.items():
        print(f"\nCluster {cluster_id}: {info['size']} documentos")
        for label in info['labels'][:3]:  # Mostrar primeros 3
            print(f"  - {label}")
        if info['size'] > 3:
            print(f"  ... y {info['size'] - 3} más")

    # 5. Tamaños de clusters
    print("\n5. Gráfico de tamaños de clusters...")
    viz.plot_cluster_sizes(3, 'output/cluster_sizes.png')

    # 6. Sugerir número óptimo de clusters
    print("\n6. Sugiriendo número óptimo de clusters...")
    n_optimal = viz.suggest_n_clusters(max_clusters=5, method='gap')
    print(f"\nNúmero sugerido de clusters: {n_optimal}")

    print("\n" + "="*70)
    print(" EJEMPLO COMPLETADO")
    print("="*70)
    print("\nArchivos generados:")
    print("  - output/dendrogram_static.png")
    print("  - output/dendrogram_interactive.html")
    print("  - output/comparison_dendrograms.png")
    print("  - output/cluster_sizes.png")


if __name__ == "__main__":
    main()
