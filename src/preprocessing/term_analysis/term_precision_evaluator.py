"""
Evaluador de precisión de términos extraídos vs predefinidos.

Este módulo evalúa la calidad de la extracción automática comparando términos
extraídos con términos predefinidos usando similitud semántica (SBERT).

Características:
    - Similitud semántica con SBERT (embeddings)
    - Fallback a similitud léxica si SBERT no disponible
    - Identificación de matches exactos, parciales y términos nuevos
    - Métricas: Precision, Recall, F1, Coverage
    - Explicación de términos nuevos con contextos
    - Visualizaciones: matriz de similitud, Venn diagram
    - Reportes detallados en Markdown
"""

import logging
import re
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# SBERT (opcional, fallback a similitud léxica)
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# Similitud léxica (fallback)
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class TermPrecisionEvaluator:
    """
    Evalúa precisión de términos extraídos automáticamente comparados con predefinidos.

    Usa similitud semántica (SBERT) para identificar matches más allá de coincidencias
    léxicas exactas. Por ejemplo:
        - "machine learning" y "ML algorithms" tienen alta similitud semántica
        - "fine-tuning" y "model tuning" son conceptos relacionados

    Si SBERT no está disponible, usa similitud léxica (SequenceMatcher).

    Atributos:
        predefined_terms: Lista de términos de referencia
        extracted_terms: Lista de términos extraídos automáticamente
        model: Modelo SBERT (si disponible)
        similarity_matrix: Matriz de similitudes calculada
    """

    def __init__(self, predefined_terms: List[str], extracted_terms: List[str]):
        """
        Inicializa evaluador con términos predefinidos y extraídos.

        Args:
            predefined_terms: Lista de términos predefinidos (ground truth)
            extracted_terms: Lista de términos extraídos automáticamente
        """
        logger.info("Inicializando TermPrecisionEvaluator...")
        logger.info(f"  Predefined terms: {len(predefined_terms)}")
        logger.info(f"  Extracted terms: {len(extracted_terms)}")

        self.predefined_terms = predefined_terms
        self.extracted_terms = extracted_terms

        # Normalizar términos (lowercase, strip)
        self.predefined_normalized = [self._normalize_term(t) for t in predefined_terms]
        self.extracted_normalized = [self._normalize_term(t) for t in extracted_terms]

        # Inicializar modelo SBERT si disponible
        self.model = None
        self.use_sbert = False

        if SBERT_AVAILABLE:
            try:
                logger.info("  Cargando modelo SBERT...")
                # Modelo pequeño y rápido para similitud semántica
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_sbert = True
                logger.info("  ✓ SBERT cargado correctamente")
            except Exception as e:
                logger.warning(f"  ⚠️ Error cargando SBERT: {e}")
                logger.warning("  Usando similitud léxica como fallback")
        else:
            logger.warning("  ⚠️ SBERT no disponible (instalar: pip install sentence-transformers)")
            logger.warning("  Usando similitud léxica como fallback")

        # Matriz de similitud (se calcula bajo demanda)
        self.similarity_matrix = None

        logger.info("✓ TermPrecisionEvaluator inicializado")

    def _normalize_term(self, term: str) -> str:
        """
        Normaliza término para comparación.

        Args:
            term: Término a normalizar

        Returns:
            Término normalizado (lowercase, stripped)
        """
        return term.lower().strip()

    def _lexical_similarity(self, term1: str, term2: str) -> float:
        """
        Calcula similitud léxica entre dos términos (fallback).

        Usa SequenceMatcher de difflib para similitud de caracteres.

        Args:
            term1: Primer término
            term2: Segundo término

        Returns:
            Similitud en [0, 1]
        """
        return SequenceMatcher(None, term1, term2).ratio()

    def calculate_similarity_matrix(self) -> np.ndarray:
        """
        Calcula matriz de similitud entre términos predefinidos y extraídos.

        Método:
            1. Si SBERT disponible: Calcula embeddings y similitud coseno
            2. Si no: Usa similitud léxica (SequenceMatcher)

        Matriz resultante: shape (len(predefined), len(extracted))
            - Fila i: similitudes del término predefinido i con todos los extraídos
            - Columna j: similitudes del término extraído j con todos los predefinidos
            - Valor [i,j]: similitud entre predefined[i] y extracted[j]

        Returns:
            Matriz numpy de similitudes en [0, 1]
        """
        logger.info("\nCalculando matriz de similitud...")

        if self.use_sbert and self.model is not None:
            # ================================================================
            # MÉTODO 1: SBERT (Sentence-BERT embeddings + cosine similarity)
            # ================================================================
            logger.info("  Método: SBERT (similitud semántica)")

            # Generar embeddings
            logger.info("  Generando embeddings...")
            predefined_embeddings = self.model.encode(
                self.predefined_normalized,
                show_progress_bar=False
            )
            extracted_embeddings = self.model.encode(
                self.extracted_normalized,
                show_progress_bar=False
            )

            # Calcular similitud coseno
            # similarity[i,j] = cos(predefined[i], extracted[j])
            logger.info("  Calculando similitud coseno...")
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(
                predefined_embeddings,
                extracted_embeddings
            )

        else:
            # ================================================================
            # MÉTODO 2: Similitud léxica (fallback)
            # ================================================================
            logger.info("  Método: Similitud léxica (fallback)")

            n_predefined = len(self.predefined_normalized)
            n_extracted = len(self.extracted_normalized)

            similarity_matrix = np.zeros((n_predefined, n_extracted))

            for i, pred_term in enumerate(self.predefined_normalized):
                for j, ext_term in enumerate(self.extracted_normalized):
                    similarity_matrix[i, j] = self._lexical_similarity(pred_term, ext_term)

        self.similarity_matrix = similarity_matrix

        logger.info(f"  ✓ Matriz calculada: {similarity_matrix.shape}")
        logger.info(f"  Similitud promedio: {similarity_matrix.mean():.3f}")
        logger.info(f"  Similitud máxima: {similarity_matrix.max():.3f}")

        return similarity_matrix

    def identify_matches(self, threshold: float = 0.70) -> Dict[str, List]:
        """
        Identifica matches entre términos predefinidos y extraídos.

        Clasificación:
            - Exact matches (>= threshold): Alta similitud semántica
            - Partial matches (0.5 - threshold): Similitud moderada
            - Novel terms: Términos extraídos sin match con predefinidos
            - Predefined not found: Términos predefinidos no extraídos

        Args:
            threshold: Umbral para considerar match exacto (default: 0.70)

        Returns:
            Dict con 4 listas:
                {
                    'exact_matches': [(pred_term, ext_term, similarity), ...],
                    'partial_matches': [(pred_term, ext_term, similarity), ...],
                    'novel_terms': [ext_term, ...],
                    'predefined_not_found': [pred_term, ...]
                }
        """
        logger.info(f"\nIdentificando matches (threshold={threshold})...")

        # Calcular matriz si no existe
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        exact_matches = []
        partial_matches = []

        # Rastrear cuáles términos ya tienen match
        matched_predefined = set()
        matched_extracted = set()

        # Para cada término predefinido, buscar mejor match
        for i, pred_term in enumerate(self.predefined_terms):
            best_match_idx = np.argmax(self.similarity_matrix[i, :])
            best_similarity = self.similarity_matrix[i, best_match_idx]

            if best_similarity >= threshold:
                # Match exacto
                ext_term = self.extracted_terms[best_match_idx]
                exact_matches.append((pred_term, ext_term, best_similarity))
                matched_predefined.add(i)
                matched_extracted.add(best_match_idx)

            elif best_similarity >= 0.5:
                # Match parcial
                ext_term = self.extracted_terms[best_match_idx]
                partial_matches.append((pred_term, ext_term, best_similarity))
                matched_predefined.add(i)
                matched_extracted.add(best_match_idx)

        # Términos predefinidos no encontrados
        predefined_not_found = [
            self.predefined_terms[i]
            for i in range(len(self.predefined_terms))
            if i not in matched_predefined
        ]

        # Términos nuevos (extraídos sin match)
        novel_terms = [
            self.extracted_terms[i]
            for i in range(len(self.extracted_terms))
            if i not in matched_extracted
        ]

        # Log resultados
        logger.info(f"  ✓ Exact matches: {len(exact_matches)}")
        logger.info(f"  ✓ Partial matches: {len(partial_matches)}")
        logger.info(f"  ✓ Novel terms: {len(novel_terms)}")
        logger.info(f"  ✓ Predefined not found: {len(predefined_not_found)}")

        return {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'novel_terms': novel_terms,
            'predefined_not_found': predefined_not_found
        }

    def calculate_metrics(self, matches: Dict[str, List]) -> Dict[str, float]:
        """
        Calcula métricas de evaluación de precisión.

        Métricas:
            Precision = términos extraídos relevantes / total extraídos
                Relevantes = exact_matches + partial_matches

            Recall = términos predefinidos encontrados / total predefinidos
                Encontrados = exact_matches + partial_matches

            F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

            Coverage = porcentaje de términos predefinidos cubiertos
                Coverage = (exact + partial) / total predefinidos

        Args:
            matches: Dict retornado por identify_matches()

        Returns:
            Dict con métricas:
                {
                    'precision': float,
                    'recall': float,
                    'f1_score': float,
                    'coverage': float,
                    'exact_match_count': int,
                    'partial_match_count': int,
                    'novel_terms_count': int
                }
        """
        logger.info("\nCalculando métricas de evaluación...")

        # Contar matches
        n_exact = len(matches['exact_matches'])
        n_partial = len(matches['partial_matches'])
        n_novel = len(matches['novel_terms'])
        n_not_found = len(matches['predefined_not_found'])

        # Total de términos
        n_extracted = len(self.extracted_terms)
        n_predefined = len(self.predefined_terms)

        # Términos relevantes (exact + partial)
        n_relevant = n_exact + n_partial

        # Precision: proporción de extraídos que son relevantes
        precision = n_relevant / n_extracted if n_extracted > 0 else 0.0

        # Recall: proporción de predefinidos que fueron encontrados
        recall = n_relevant / n_predefined if n_predefined > 0 else 0.0

        # F1-Score: media armónica
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Coverage: porcentaje de predefinidos cubiertos
        coverage = n_relevant / n_predefined if n_predefined > 0 else 0.0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'coverage': coverage,
            'exact_match_count': n_exact,
            'partial_match_count': n_partial,
            'novel_terms_count': n_novel,
            'predefined_not_found_count': n_not_found
        }

        # Log métricas
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1_score:.3f}")
        logger.info(f"  Coverage: {coverage:.3f} ({n_relevant}/{n_predefined})")

        return metrics

    def explain_novel_terms(self, novel_terms: List[str], abstracts: List[str]) -> Dict[str, Dict]:
        """
        Explica por qué términos nuevos son relevantes.

        Para cada término nuevo:
            1. Buscar contextos donde aparece (frases)
            2. Calcular frecuencia de aparición
            3. Calcular score de relevancia (basado en frecuencia)
            4. Proveer interpretación del término

        Args:
            novel_terms: Lista de términos nuevos (sin match con predefinidos)
            abstracts: Lista de abstracts donde buscar contextos

        Returns:
            Dict con explicación de cada término:
                {
                    término: {
                        'frequency': int,
                        'document_frequency': int,
                        'example_contexts': [frase1, frase2, ...],
                        'relevance_score': float,
                        'interpretation': str
                    }
                }
        """
        logger.info(f"\nExplicando {len(novel_terms)} términos nuevos...")

        explanations = {}

        for term in novel_terms:
            term_lower = term.lower()

            # Contar frecuencias
            total_count = 0
            doc_count = 0
            contexts = []

            for abstract in abstracts:
                abstract_lower = abstract.lower()

                # Buscar término en abstract
                # Usar regex con word boundaries para evitar partial matches
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                matches = list(re.finditer(pattern, abstract_lower))

                if matches:
                    total_count += len(matches)
                    doc_count += 1

                    # Extraer contexto (ventana de ±50 caracteres)
                    for match in matches[:2]:  # Máximo 2 contextos por documento
                        start = max(0, match.start() - 50)
                        end = min(len(abstract), match.end() + 50)
                        context = abstract[start:end].strip()

                        # Limpiar contexto
                        context = re.sub(r'\s+', ' ', context)
                        if not context.startswith('...'):
                            context = '...' + context
                        if not context.endswith('...'):
                            context = context + '...'

                        contexts.append(context)

            # Calcular relevance score
            # Basado en frecuencia y document frequency
            # Score alto = aparece frecuentemente en múltiples documentos
            doc_freq_ratio = doc_count / len(abstracts) if abstracts else 0
            avg_freq = total_count / len(abstracts) if abstracts else 0
            relevance_score = (doc_freq_ratio + avg_freq) / 2

            # Interpretación automática (básica)
            interpretation = self._interpret_term(term, total_count, doc_count)

            explanations[term] = {
                'frequency': total_count,
                'document_frequency': doc_count,
                'example_contexts': contexts[:3],  # Top 3 contextos
                'relevance_score': relevance_score,
                'interpretation': interpretation
            }

            logger.info(f"  {term}: freq={total_count}, docs={doc_count}, score={relevance_score:.3f}")

        logger.info("✓ Explicaciones generadas")

        return explanations

    def _interpret_term(self, term: str, frequency: int, doc_frequency: int) -> str:
        """
        Genera interpretación básica de un término nuevo.

        Args:
            term: Término a interpretar
            frequency: Frecuencia total
            doc_frequency: Número de documentos donde aparece

        Returns:
            String con interpretación
        """
        # Clasificación básica por frecuencia
        if doc_frequency >= 3:
            prominence = "highly prominent"
        elif doc_frequency >= 2:
            prominence = "moderately prominent"
        else:
            prominence = "appears in limited contexts"

        # Tipo de término (heurística simple)
        if len(term.split()) >= 3:
            term_type = "multi-word phrase"
        elif len(term.split()) == 2:
            term_type = "compound term"
        else:
            term_type = "single term"

        interpretation = (
            f"This {term_type} is {prominence} across the corpus "
            f"(appears {frequency} times in {doc_frequency} documents). "
            f"It represents a concept not captured by predefined terms."
        )

        return interpretation

    def visualize_similarity_matrix(self, output_path: str = None):
        """
        Visualiza matriz de similitud como heatmap.

        Args:
            output_path: Ruta para guardar figura (None = mostrar)
        """
        logger.info("\nGenerando visualización de matriz de similitud...")

        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Heatmap
        sns.heatmap(
            self.similarity_matrix,
            xticklabels=self.extracted_terms,
            yticklabels=self.predefined_terms,
            cmap='RdYlGn',
            annot=True if len(self.predefined_terms) <= 15 and len(self.extracted_terms) <= 15 else False,
            fmt='.2f',
            cbar_kws={'label': 'Similarity Score'},
            ax=ax
        )

        ax.set_xlabel('Extracted Terms', fontweight='bold', fontsize=12)
        ax.set_ylabel('Predefined Terms', fontweight='bold', fontsize=12)
        ax.set_title('Similarity Matrix: Predefined vs Extracted Terms',
                     fontweight='bold', fontsize=14)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Visualización guardada: {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_venn_diagram(self, matches: Dict[str, List], output_path: str = None):
        """
        Visualiza overlap entre términos predefinidos y extraídos con Venn diagram.

        Args:
            matches: Dict de identify_matches()
            output_path: Ruta para guardar figura (None = mostrar)
        """
        logger.info("\nGenerando Venn diagram...")

        # Contar categorías
        n_exact = len(matches['exact_matches'])
        n_partial = len(matches['partial_matches'])
        n_novel = len(matches['novel_terms'])
        n_not_found = len(matches['predefined_not_found'])

        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ====================================================================
        # Subplot 1: Categorías de términos (barras)
        # ====================================================================

        categories = ['Exact\nMatches', 'Partial\nMatches', 'Novel\nTerms', 'Not\nFound']
        counts = [n_exact, n_partial, n_novel, n_not_found]
        colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']

        bars = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Agregar valores sobre barras
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax1.set_ylabel('Count', fontweight='bold', fontsize=12)
        ax1.set_title('Term Categories', fontweight='bold', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)

        # ====================================================================
        # Subplot 2: Proporción de overlap (pie chart)
        # ====================================================================

        # Combinar exact y partial como "Matched"
        n_matched = n_exact + n_partial

        sizes = [n_matched, n_novel, n_not_found]
        labels = [
            f'Matched\n({n_matched})',
            f'Novel Terms\n({n_novel})',
            f'Not Found\n({n_not_found})'
        ]
        colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
        explode = (0.05, 0, 0)

        ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               explode=explode, shadow=True, startangle=90,
               textprops={'fontweight': 'bold', 'fontsize': 11})

        ax2.set_title('Term Overlap Distribution', fontweight='bold', fontsize=14)

        plt.suptitle('Predefined vs Extracted Terms Analysis',
                     fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Venn diagram guardado: {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_evaluation_report(self, matches: Dict[str, List],
                                   metrics: Dict[str, float],
                                   novel_explanations: Dict[str, Dict],
                                   output_path: str):
        """
        Genera reporte Markdown completo de evaluación.

        Contenido:
            1. Resumen ejecutivo con métricas
            2. Tabla comparativa términos predefinidos vs extraídos
            3. Análisis detallado de matches
            4. Análisis de términos nuevos con explicaciones
            5. Términos predefinidos no encontrados
            6. Recomendaciones

        Args:
            matches: Dict de identify_matches()
            metrics: Dict de calculate_metrics()
            novel_explanations: Dict de explain_novel_terms()
            output_path: Ruta donde guardar reporte
        """
        logger.info(f"\nGenerando reporte de evaluación en '{output_path}'...")

        report = []

        # ====================================================================
        # ENCABEZADO
        # ====================================================================
        report.append("# Reporte de Evaluación de Precisión de Términos\n")
        report.append(f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Método de similitud:** {'SBERT (semántica)' if self.use_sbert else 'Léxica (fallback)'}\n")
        report.append("\n---\n")

        # ====================================================================
        # RESUMEN EJECUTIVO
        # ====================================================================
        report.append("## Resumen Ejecutivo\n")
        report.append(f"- **Términos predefinidos:** {len(self.predefined_terms)}\n")
        report.append(f"- **Términos extraídos:** {len(self.extracted_terms)}\n")
        report.append(f"- **Precision:** {metrics['precision']:.3f}\n")
        report.append(f"- **Recall:** {metrics['recall']:.3f}\n")
        report.append(f"- **F1-Score:** {metrics['f1_score']:.3f}\n")
        report.append(f"- **Coverage:** {metrics['coverage']:.1%} ({metrics['exact_match_count'] + metrics['partial_match_count']}/{len(self.predefined_terms)})\n")
        report.append("\n")

        # Interpretación de métricas
        if metrics['f1_score'] >= 0.7:
            assessment = "✅ **Excelente**: Alta concordancia entre términos predefinidos y extraídos"
        elif metrics['f1_score'] >= 0.5:
            assessment = "✓ **Bueno**: Concordancia moderada con margen de mejora"
        else:
            assessment = "⚠️ **Requiere revisión**: Baja concordancia, considerar ajustar extracción"

        report.append(f"**Evaluación general:** {assessment}\n")
        report.append("\n---\n")

        # ====================================================================
        # MÉTRICAS DETALLADAS
        # ====================================================================
        report.append("## Métricas Detalladas\n")

        metrics_table = pd.DataFrame([{
            'Metric': 'Precision',
            'Value': f"{metrics['precision']:.3f}",
            'Interpretation': 'Proporción de términos extraídos que son relevantes'
        }, {
            'Metric': 'Recall',
            'Value': f"{metrics['recall']:.3f}",
            'Interpretation': 'Proporción de términos predefinidos encontrados'
        }, {
            'Metric': 'F1-Score',
            'Value': f"{metrics['f1_score']:.3f}",
            'Interpretation': 'Media armónica de Precision y Recall'
        }, {
            'Metric': 'Coverage',
            'Value': f"{metrics['coverage']:.1%}",
            'Interpretation': 'Porcentaje de términos predefinidos cubiertos'
        }])

        report.append(metrics_table.to_markdown(index=False))
        report.append("\n\n---\n")

        # ====================================================================
        # MATCHES EXACTOS
        # ====================================================================
        report.append("## Matches Exactos\n")
        report.append(f"**Total:** {len(matches['exact_matches'])} matches con similitud ≥ 0.70\n\n")

        if matches['exact_matches']:
            exact_df = pd.DataFrame([
                {
                    'Predefined Term': pred,
                    'Extracted Term': ext,
                    'Similarity': f"{sim:.3f}"
                }
                for pred, ext, sim in sorted(matches['exact_matches'], key=lambda x: -x[2])
            ])

            report.append(exact_df.to_markdown(index=False))
            report.append("\n")
        else:
            report.append("*No se encontraron matches exactos.*\n")

        report.append("\n---\n")

        # ====================================================================
        # MATCHES PARCIALES
        # ====================================================================
        report.append("## Matches Parciales\n")
        report.append(f"**Total:** {len(matches['partial_matches'])} matches con similitud 0.50-0.69\n\n")

        if matches['partial_matches']:
            partial_df = pd.DataFrame([
                {
                    'Predefined Term': pred,
                    'Extracted Term': ext,
                    'Similarity': f"{sim:.3f}"
                }
                for pred, ext, sim in sorted(matches['partial_matches'], key=lambda x: -x[2])
            ])

            report.append(partial_df.to_markdown(index=False))
            report.append("\n")
        else:
            report.append("*No se encontraron matches parciales.*\n")

        report.append("\n---\n")

        # ====================================================================
        # TÉRMINOS NUEVOS (NOVEL TERMS)
        # ====================================================================
        report.append("## Términos Nuevos Descubiertos\n")
        report.append(f"**Total:** {len(matches['novel_terms'])} términos sin match con predefinidos\n\n")
        report.append("Estos términos representan conceptos relevantes no capturados por la lista predefinida.\n\n")

        if matches['novel_terms']:
            for term in matches['novel_terms']:
                report.append(f"### {term}\n")

                if term in novel_explanations:
                    expl = novel_explanations[term]

                    report.append(f"- **Frecuencia total:** {expl['frequency']}\n")
                    report.append(f"- **Documentos:** {expl['document_frequency']}\n")
                    report.append(f"- **Relevance score:** {expl['relevance_score']:.3f}\n")
                    report.append(f"- **Interpretación:** {expl['interpretation']}\n")

                    if expl['example_contexts']:
                        report.append("\n**Contextos de ejemplo:**\n")
                        for i, ctx in enumerate(expl['example_contexts'][:2], 1):
                            report.append(f"{i}. *\"{ctx}\"*\n")

                    report.append("\n")
        else:
            report.append("*Todos los términos extraídos tienen match con predefinidos.*\n")

        report.append("\n---\n")

        # ====================================================================
        # TÉRMINOS PREDEFINIDOS NO ENCONTRADOS
        # ====================================================================
        report.append("## Términos Predefinidos No Encontrados\n")
        report.append(f"**Total:** {len(matches['predefined_not_found'])}\n\n")

        if matches['predefined_not_found']:
            report.append("Los siguientes términos predefinidos no fueron detectados por la extracción automática:\n\n")
            for term in sorted(matches['predefined_not_found']):
                report.append(f"- `{term}`\n")

            report.append("\n**Posibles causas:**\n")
            report.append("- Términos no aparecen con suficiente frecuencia en el corpus\n")
            report.append("- Términos expresados con diferente vocabulario\n")
            report.append("- Parámetros de extracción (min_df, max_df) filtran estos términos\n")
            report.append("- Términos demasiado generales para ser detectados como keywords\n")
        else:
            report.append("✓ Todos los términos predefinidos fueron encontrados.\n")

        report.append("\n---\n")

        # ====================================================================
        # RECOMENDACIONES
        # ====================================================================
        report.append("## Recomendaciones\n")

        # Recomendaciones basadas en métricas
        if metrics['precision'] < 0.6:
            report.append("### Mejorar Precision\n")
            report.append("- Aumentar `min_df` en TF-IDF para filtrar términos raros\n")
            report.append("- Ajustar threshold de similitud (aumentar para matches más estrictos)\n")
            report.append("- Expandir lista de stopwords para eliminar términos genéricos\n")
            report.append("\n")

        if metrics['recall'] < 0.6:
            report.append("### Mejorar Recall\n")
            report.append("- Extraer más términos (aumentar `n_terms`)\n")
            report.append("- Reducir `min_df` para capturar términos menos frecuentes\n")
            report.append("- Verificar que términos predefinidos estén en el corpus\n")
            report.append("- Considerar variantes de términos (singular/plural, sinónimos)\n")
            report.append("\n")

        if len(matches['novel_terms']) > len(self.predefined_terms):
            report.append("### Alto Número de Términos Nuevos\n")
            report.append("- Revisar términos nuevos relevantes para añadir a lista predefinida\n")
            report.append("- Verificar si extracción está capturando ruido (términos irrelevantes)\n")
            report.append("- Considerar usar ensemble method para mejor precisión\n")
            report.append("\n")

        if metrics['f1_score'] >= 0.7:
            report.append("### Sistema Funcionando Bien\n")
            report.append("✅ La extracción automática tiene alta concordancia con términos predefinidos.\n")
            report.append("- Considerar usar términos nuevos relevantes para enriquecer análisis\n")
            report.append("- Documentar términos nuevos importantes para futuros estudios\n")

        report.append("\n---\n")

        # ====================================================================
        # TABLA RESUMEN COMPARATIVA
        # ====================================================================
        report.append("## Tabla Resumen Comparativa\n")

        summary_data = {
            'Categoría': [
                'Exact Matches',
                'Partial Matches',
                'Novel Terms',
                'Predefined Not Found',
                'Total Predefined',
                'Total Extracted'
            ],
            'Cantidad': [
                metrics['exact_match_count'],
                metrics['partial_match_count'],
                metrics['novel_terms_count'],
                metrics['predefined_not_found_count'],
                len(self.predefined_terms),
                len(self.extracted_terms)
            ],
            'Porcentaje': [
                f"{metrics['exact_match_count']/len(self.predefined_terms)*100:.1f}%",
                f"{metrics['partial_match_count']/len(self.predefined_terms)*100:.1f}%",
                f"{metrics['novel_terms_count']/len(self.extracted_terms)*100:.1f}%",
                f"{metrics['predefined_not_found_count']/len(self.predefined_terms)*100:.1f}%",
                "100.0%",
                "100.0%"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        report.append(summary_df.to_markdown(index=False))

        report.append("\n\n---\n")
        report.append("\n*Fin del reporte*\n")

        # Guardar reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))

        logger.info(f"✓ Reporte guardado en '{output_path}'")


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Términos predefinidos
    predefined = [
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing",
        "artificial intelligence"
    ]

    # Términos extraídos (simulados)
    extracted = [
        "machine learning",  # Match exacto
        "ML algorithms",     # Similar semánticamente a "machine learning"
        "deep neural nets",  # Similar a "deep learning" + "neural networks"
        "NLP techniques",    # Similar a "natural language processing"
        "computer vision",   # Término nuevo
        "data mining"        # Término nuevo
    ]

    # Abstracts de ejemplo
    abstracts = [
        "Machine learning algorithms are transforming computer vision applications.",
        "Deep neural nets achieve state-of-the-art results in NLP techniques.",
        "Data mining reveals patterns in large datasets using ML algorithms."
    ]

    # Crear evaluador
    evaluator = TermPrecisionEvaluator(predefined, extracted)

    # Calcular similitud
    similarity_matrix = evaluator.calculate_similarity_matrix()
    print("\nSimilarity Matrix:")
    print(similarity_matrix)

    # Identificar matches
    matches = evaluator.identify_matches(threshold=0.70)

    # Calcular métricas
    metrics = evaluator.calculate_metrics(matches)
    print(f"\nMetrics: {metrics}")

    # Explicar términos nuevos
    novel_explanations = evaluator.explain_novel_terms(matches['novel_terms'], abstracts)

    # Generar visualizaciones
    Path('output/term_analysis').mkdir(parents=True, exist_ok=True)
    evaluator.visualize_similarity_matrix('output/term_analysis/similarity_matrix.png')
    evaluator.visualize_venn_diagram(matches, 'output/term_analysis/venn_diagram.png')

    # Generar reporte
    evaluator.generate_evaluation_report(
        matches,
        metrics,
        novel_explanations,
        'output/term_analysis/precision_evaluation_report.md'
    )
