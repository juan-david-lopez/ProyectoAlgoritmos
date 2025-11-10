"""
Analizador de frecuencia de términos predefinidos.

Este módulo analiza la frecuencia de términos específicos relacionados con
'Concepts of Generative AI in Education' en abstracts científicos.

Características:
    - Búsqueda flexible con variantes (singular/plural, guiones)
    - Análisis de co-ocurrencia entre términos
    - Estadísticas descriptivas completas
    - Visualizaciones múltiples (barras, heatmap, distribución)
"""

import json
import re
import logging
from typing import List, Dict, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

logger = logging.getLogger(__name__)


class PredefinedTermsAnalyzer:
    """
    Analiza frecuencia de términos específicos de 'Concepts of Generative AI in Education'.

    Esta clase implementa un análisis exhaustivo de términos predefinidos,
    incluyendo:
        - Detección flexible con variantes
        - Estadísticas de frecuencia
        - Análisis de co-ocurrencia
        - Visualizaciones múltiples
    """

    # Términos predefinidos del dominio de Generative AI in Education
    PREDEFINED_TERMS = [
        "Generative models",
        "Prompting",
        "Machine learning",
        "Multimodality",
        "Fine-tuning",
        "Training data",
        "Algorithmic bias",
        "Explainability",
        "Transparency",
        "Ethics",
        "Privacy",
        "Personalization",
        "Human-AI interaction",
        "AI literacy",
        "Co-creation"
    ]

    def __init__(self, unified_data_path: str):
        """
        Inicializa el analizador con datos unificados.

        Args:
            unified_data_path: Ruta al archivo JSON con abstracts unificados
        """
        logger.info(f"Inicializando PredefinedTermsAnalyzer...")
        logger.info(f"Cargando datos desde: {unified_data_path}")

        # Cargar datos
        with open(unified_data_path, 'r', encoding='utf-8') as f:
            self.unified_data = json.load(f)

        logger.info(f"✓ Datos cargados: {len(self.unified_data)} artículos")
        logger.info(f"✓ Términos predefinidos: {len(self.PREDEFINED_TERMS)}")

        # Extraer abstracts
        self.abstracts = [art['abstract'] for art in self.unified_data if art.get('abstract')]
        logger.info(f"✓ Abstracts disponibles: {len(self.abstracts)}")

        # Caché de variantes
        self._variant_cache = {}

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesamiento suave para búsqueda flexible.

        Aplica transformaciones mínimas para preservar términos compuestos:
            - Lowercase para búsqueda case-insensitive
            - Normalización de espacios múltiples
            - Mantiene guiones y caracteres especiales

        Ejemplo:
            Input:  "Machine   Learning and Fine-Tuning"
            Output: "machine learning and fine-tuning"

        Args:
            text: Texto a preprocesar

        Returns:
            Texto preprocesado en minúsculas con espacios normalizados
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Normalizar espacios múltiples a uno solo
        text = re.sub(r'\s+', ' ', text)

        # Trim
        text = text.strip()

        return text

    def find_term_variants(self, term: str) -> List[str]:
        """
        Genera variantes del término para búsqueda flexible.

        Estrategias de variación:
            1. Singular/Plural:
               - "models" → ["model", "models"]
               - "ethics" → ["ethic", "ethics"]

            2. Guiones/Espacios:
               - "Fine-tuning" → ["fine-tuning", "fine tuning", "finetuning"]
               - "Human-AI" → ["human-ai", "human ai"]

            3. Case variations:
               - Todas en lowercase para búsqueda

        Ejemplo:
            Input:  "Fine-tuning"
            Output: [
                "fine-tuning",
                "fine tuning",
                "finetuning",
                "finetune",
                "finetuned",
                "finetuning"
            ]

        Args:
            term: Término original

        Returns:
            Lista de variantes únicas del término
        """
        # Check caché
        if term in self._variant_cache:
            return self._variant_cache[term]

        variants = set()
        term_lower = term.lower()

        # Variante original
        variants.add(term_lower)

        # 1. Variantes de guiones
        if '-' in term_lower:
            # Con espacio: "fine-tuning" → "fine tuning"
            variants.add(term_lower.replace('-', ' '))
            # Sin separador: "fine-tuning" → "finetuning"
            variants.add(term_lower.replace('-', ''))

        # 2. Variantes singular/plural
        # Simple: agregar/quitar 's' al final
        if term_lower.endswith('s') and len(term_lower) > 2:
            # Plural → Singular
            singular = term_lower[:-1]
            variants.add(singular)

            # Casos especiales de plural
            if term_lower.endswith('ies'):
                # "ethics" → "ethic"
                variants.add(term_lower[:-1])
            elif term_lower.endswith('es'):
                # "biases" → "bias"
                variants.add(term_lower[:-2])
        else:
            # Singular → Plural
            variants.add(term_lower + 's')

            # Plural con 'es'
            if term_lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
                variants.add(term_lower + 'es')

        # 3. Variantes específicas para términos compuestos
        words = term_lower.split()
        if len(words) > 1:
            # Versión sin espacios
            variants.add(''.join(words))

            # Variantes de cada palabra
            for i, word in enumerate(words):
                # Singular/plural de cada palabra
                if word.endswith('s'):
                    new_words = words.copy()
                    new_words[i] = word[:-1]
                    variants.add(' '.join(new_words))
                else:
                    new_words = words.copy()
                    new_words[i] = word + 's'
                    variants.add(' '.join(new_words))

        # 4. Variantes verbales (para términos como "Fine-tuning")
        if 'tuning' in term_lower:
            base = term_lower.replace('tuning', 'tune')
            variants.add(base)
            variants.add(base.replace('-', ' '))
            variants.add(base.replace('-', ''))
            variants.add(term_lower.replace('tuning', 'tuned'))

        # Convertir a lista ordenada
        variants_list = sorted(list(variants))

        # Guardar en caché
        self._variant_cache[term] = variants_list

        logger.debug(f"Variantes de '{term}': {variants_list}")

        return variants_list

    def calculate_frequencies(self, abstracts: List[str] = None) -> Dict:
        """
        Calcula frecuencia de cada término predefinido en los abstracts.

        Proceso:
            1. Para cada término:
               a. Generar variantes (singular/plural, guiones)
               b. Buscar cada variante en textos preprocesados
               c. Contar ocurrencias totales y documentos

            2. Calcular estadísticas:
               - Total de ocurrencias
               - Documentos que contienen el término
               - Promedio por documento
               - Desglose por variante

        Ejemplo de resultado:
            {
                'Generative models': {
                    'total_count': 45,
                    'documents_count': 23,
                    'avg_per_document': 1.96,
                    'document_frequency': 0.23,  # % de documentos
                    'variants_found': {
                        'generative model': 30,
                        'generative models': 15
                    }
                },
                ...
            }

        Args:
            abstracts: Lista de abstracts (usa self.abstracts si None)

        Returns:
            Diccionario con frecuencias y estadísticas por término
        """
        if abstracts is None:
            abstracts = self.abstracts

        logger.info(f"\nCalculando frecuencias de {len(self.PREDEFINED_TERMS)} términos en {len(abstracts)} abstracts...")

        frequencies = {}
        total_docs = len(abstracts)

        # Preprocesar todos los abstracts una vez
        preprocessed_abstracts = [self.preprocess_text(abs) for abs in abstracts]

        for term in self.PREDEFINED_TERMS:
            logger.debug(f"  Analizando término: '{term}'")

            # Generar variantes
            variants = self.find_term_variants(term)

            # Contadores
            total_count = 0
            documents_with_term = 0
            variants_counts = defaultdict(int)

            # Buscar en cada abstract
            for abstract in preprocessed_abstracts:
                found_in_doc = False

                for variant in variants:
                    # Búsqueda con word boundaries para evitar matches parciales
                    # Ejemplo: "model" no debe matchear en "remodel"
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    matches = re.findall(pattern, abstract)

                    if matches:
                        count = len(matches)
                        total_count += count
                        variants_counts[variant] += count
                        found_in_doc = True

                if found_in_doc:
                    documents_with_term += 1

            # Calcular estadísticas
            avg_per_document = total_count / total_docs if total_docs > 0 else 0
            document_frequency = documents_with_term / total_docs if total_docs > 0 else 0

            frequencies[term] = {
                'total_count': total_count,
                'documents_count': documents_with_term,
                'avg_per_document': avg_per_document,
                'document_frequency': document_frequency,
                'variants_found': dict(variants_counts)
            }

            logger.debug(f"    Total: {total_count}, Docs: {documents_with_term}, Avg: {avg_per_document:.2f}")

        logger.info(f"✓ Frecuencias calculadas")

        return frequencies

    def calculate_cooccurrence_matrix(self, abstracts: List[str] = None) -> pd.DataFrame:
        """
        Calcula matriz de co-ocurrencia entre términos.

        La co-ocurrencia mide cuántas veces dos términos aparecen juntos
        en el mismo documento (abstract).

        Matriz simétrica donde:
            - Diagonal: frecuencia del término consigo mismo
            - [i,j]: número de documentos donde términos i y j aparecen juntos

        Args:
            abstracts: Lista de abstracts (usa self.abstracts si None)

        Returns:
            DataFrame con matriz de co-ocurrencia (términos x términos)
        """
        if abstracts is None:
            abstracts = self.abstracts

        logger.info("Calculando matriz de co-ocurrencia...")

        n_terms = len(self.PREDEFINED_TERMS)
        cooccurrence = np.zeros((n_terms, n_terms), dtype=int)

        # Preprocesar abstracts
        preprocessed_abstracts = [self.preprocess_text(abs) for abs in abstracts]

        # Para cada abstract
        for abstract in preprocessed_abstracts:
            # Detectar qué términos aparecen
            terms_present = []

            for i, term in enumerate(self.PREDEFINED_TERMS):
                variants = self.find_term_variants(term)

                # Verificar si alguna variante aparece
                found = False
                for variant in variants:
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    if re.search(pattern, abstract):
                        found = True
                        break

                if found:
                    terms_present.append(i)

            # Incrementar co-ocurrencias
            for i in terms_present:
                for j in terms_present:
                    cooccurrence[i, j] += 1

        # Crear DataFrame
        df = pd.DataFrame(
            cooccurrence,
            index=self.PREDEFINED_TERMS,
            columns=self.PREDEFINED_TERMS
        )

        logger.info("✓ Matriz de co-ocurrencia calculada")

        return df

    def generate_statistics_report(self, frequencies: Dict) -> pd.DataFrame:
        """
        Genera DataFrame con estadísticas descriptivas.

        Incluye:
            - Frecuencias totales y por documento
            - Número de documentos
            - Frecuencia en corpus (%)
            - Rankings

        Args:
            frequencies: Diccionario de frecuencias de calculate_frequencies()

        Returns:
            DataFrame con estadísticas ordenadas por frecuencia total
        """
        logger.info("Generando reporte estadístico...")

        # Extraer datos
        data = []
        for term, stats in frequencies.items():
            data.append({
                'Term': term,
                'Total Count': stats['total_count'],
                'Documents': stats['documents_count'],
                'Avg per Doc': stats['avg_per_document'],
                'Doc Frequency (%)': stats['document_frequency'] * 100,
                'Variants Used': len([v for v, c in stats['variants_found'].items() if c > 0])
            })

        # Crear DataFrame
        df = pd.DataFrame(data)

        # Ordenar por frecuencia total descendente
        df = df.sort_values('Total Count', ascending=False)

        # Agregar ranking
        df['Rank'] = range(1, len(df) + 1)

        # Reordenar columnas
        df = df[['Rank', 'Term', 'Total Count', 'Documents', 'Avg per Doc', 'Doc Frequency (%)', 'Variants Used']]

        logger.info("✓ Reporte estadístico generado")

        return df

    def visualize_frequencies(self, frequencies: Dict, output_dir: str):
        """
        Genera visualizaciones de frecuencias.

        Crea 3 tipos de visualizaciones:
            1. Gráfico de barras horizontal: Frecuencia total por término
            2. Heatmap de co-ocurrencia: Términos que aparecen juntos
            3. Distribución estadística: Histograma de frecuencias

        Args:
            frequencies: Diccionario de frecuencias
            output_dir: Directorio donde guardar las visualizaciones
        """
        logger.info(f"\nGenerando visualizaciones en '{output_dir}'...")

        # Crear directorio
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # ====================================================================
        # 1. GRÁFICO DE BARRAS HORIZONTAL - Frecuencias totales
        # ====================================================================
        logger.info("  Generando gráfico de barras...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Preparar datos
        terms = list(frequencies.keys())
        counts = [frequencies[t]['total_count'] for t in terms]

        # Ordenar por frecuencia
        sorted_data = sorted(zip(terms, counts), key=lambda x: x[1])
        terms_sorted, counts_sorted = zip(*sorted_data)

        # Crear barras horizontales
        bars = ax.barh(range(len(terms_sorted)), counts_sorted)

        # Colorear barras según frecuencia
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Configurar ejes
        ax.set_yticks(range(len(terms_sorted)))
        ax.set_yticklabels(terms_sorted, fontsize=10)
        ax.set_xlabel('Total Count', fontweight='bold', fontsize=12)
        ax.set_title('Frequency of Predefined Terms in Abstracts',
                    fontweight='bold', fontsize=14, pad=20)

        # Agregar valores en las barras
        for i, (count, bar) in enumerate(zip(counts_sorted, bars)):
            ax.text(count + max(counts_sorted)*0.01, i, f'{count}',
                   va='center', fontweight='bold', fontsize=9)

        # Grid
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/term_frequencies_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ====================================================================
        # 2. HEATMAP DE CO-OCURRENCIA
        # ====================================================================
        logger.info("  Generando heatmap de co-ocurrencia...")

        # Calcular matriz de co-ocurrencia
        cooccurrence = self.calculate_cooccurrence_matrix()

        fig, ax = plt.subplots(figsize=(14, 12))

        # Crear heatmap
        sns.heatmap(
            cooccurrence,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Co-occurrence Count'},
            ax=ax
        )

        # Configurar
        ax.set_title('Term Co-occurrence Matrix\n(Number of documents where terms appear together)',
                    fontweight='bold', fontsize=14, pad=20)

        # Rotar etiquetas para mejor legibilidad
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/term_cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ====================================================================
        # 3. DISTRIBUCIÓN ESTADÍSTICA - Histograma y Box Plot
        # ====================================================================
        logger.info("  Generando distribución estadística...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Datos
        counts = [frequencies[t]['total_count'] for t in frequencies.keys()]
        doc_freqs = [frequencies[t]['document_frequency'] * 100 for t in frequencies.keys()]

        # Subplot 1: Histograma de frecuencias totales
        ax1.hist(counts, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts):.1f}')
        ax1.axvline(np.median(counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(counts):.1f}')
        ax1.set_xlabel('Total Count', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Number of Terms', fontweight='bold', fontsize=11)
        ax1.set_title('Distribution of Term Frequencies', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Subplot 2: Box plot de document frequency
        box = ax2.boxplot([doc_freqs], vert=True, patch_artist=True, widths=0.5)
        box['boxes'][0].set_facecolor('lightcoral')
        box['boxes'][0].set_alpha(0.7)

        ax2.set_ylabel('Document Frequency (%)', fontweight='bold', fontsize=11)
        ax2.set_title('Distribution of Document Frequency', fontweight='bold', fontsize=12)
        ax2.set_xticklabels(['All Terms'])
        ax2.grid(axis='y', alpha=0.3)

        # Agregar estadísticas
        stats_text = f'Mean: {np.mean(doc_freqs):.1f}%\nMedian: {np.median(doc_freqs):.1f}%\nStd: {np.std(doc_freqs):.1f}%'
        ax2.text(1.15, np.mean(doc_freqs), stats_text, fontsize=9, va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/term_distribution_stats.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Visualizaciones guardadas en '{output_dir}':")
        logger.info(f"  - term_frequencies_bar.png")
        logger.info(f"  - term_cooccurrence_heatmap.png")
        logger.info(f"  - term_distribution_stats.png")

    def generate_detailed_report(self, frequencies: Dict, output_path: str):
        """
        Genera reporte detallado en formato Markdown.

        Args:
            frequencies: Diccionario de frecuencias
            output_path: Ruta donde guardar el reporte
        """
        logger.info(f"\nGenerando reporte detallado en '{output_path}'...")

        # Generar estadísticas
        stats_df = self.generate_statistics_report(frequencies)

        # Crear reporte
        report = []
        report.append("# Reporte de Análisis de Términos Predefinidos\n")
        report.append(f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Documentos analizados:** {len(self.abstracts)}\n")
        report.append(f"**Términos predefinidos:** {len(self.PREDEFINED_TERMS)}\n")
        report.append("\n---\n")

        # Resumen ejecutivo
        report.append("## Resumen Ejecutivo\n")
        report.append(f"- **Total de ocurrencias:** {sum(f['total_count'] for f in frequencies.values())}\n")
        report.append(f"- **Término más frecuente:** {stats_df.iloc[0]['Term']} ({stats_df.iloc[0]['Total Count']} ocurrencias)\n")
        report.append(f"- **Término menos frecuente:** {stats_df.iloc[-1]['Term']} ({stats_df.iloc[-1]['Total Count']} ocurrencias)\n")
        report.append(f"- **Promedio por término:** {np.mean([f['total_count'] for f in frequencies.values()]):.2f}\n")
        report.append("\n---\n")

        # Tabla de estadísticas
        report.append("## Estadísticas por Término\n")
        report.append(stats_df.to_markdown(index=False))
        report.append("\n\n---\n")

        # Detalles de variantes
        report.append("## Variantes Detectadas por Término\n")
        for term in self.PREDEFINED_TERMS:
            if frequencies[term]['variants_found']:
                report.append(f"\n### {term}\n")
                variants = frequencies[term]['variants_found']
                sorted_variants = sorted(variants.items(), key=lambda x: x[1], reverse=True)
                for variant, count in sorted_variants:
                    report.append(f"- `{variant}`: {count} ocurrencias\n")

        # Guardar reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))

        logger.info(f"✓ Reporte guardado en '{output_path}'")


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Crear analizador
    analyzer = PredefinedTermsAnalyzer('data/unified_articles.json')

    # Calcular frecuencias
    frequencies = analyzer.calculate_frequencies()

    # Generar reporte estadístico
    stats_df = analyzer.generate_statistics_report(frequencies)
    print("\n" + "="*80)
    print("ESTADÍSTICAS DE TÉRMINOS")
    print("="*80)
    print(stats_df.to_string(index=False))

    # Generar visualizaciones
    analyzer.visualize_frequencies(frequencies, 'output/term_analysis')

    # Generar reporte detallado
    analyzer.generate_detailed_report(frequencies, 'output/term_analysis/predefined_terms_report.md')

    print("\n✓ Análisis completado")
