"""
Term Precision Evaluator
Evalúa la precisión de términos extraídos automáticamente comparándolos
con términos predefinidos usando similitud semántica.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


class TermPrecisionEvaluator:
    """
    Evalúa qué tan precisos son los términos extraídos automáticamente
    comparados con los predefinidos.
    """

    def __init__(self, predefined_terms: List[str], extracted_terms: List[str]):
        """
        Inicializa con ambas listas de términos.

        Args:
            predefined_terms: Lista de términos predefinidos manualmente
            extracted_terms: Lista de términos extraídos automáticamente
        """
        self.predefined_terms = predefined_terms
        self.extracted_terms = extracted_terms

        # Modelo SBERT para similitud semántica
        print("Cargando modelo SBERT para similitud semántica...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Computar embeddings
        print("Generando embeddings de términos...")
        self.predefined_embeddings = self.model.encode(predefined_terms)
        self.extracted_embeddings = self.model.encode(extracted_terms)

        # Matrices y resultados
        self.similarity_matrix = None
        self.matches = None
        self.metrics = None

    def calculate_similarity_matrix(self) -> np.ndarray:
        """
        Calcula similitud entre términos predefinidos y extraídos.
        Usa SBERT para similitud semántica (no solo léxica).

        Returns:
            Matriz len(predefined) x len(extracted) con similitudes
        """
        print("Calculando matriz de similitud semántica...")
        self.similarity_matrix = cosine_similarity(
            self.predefined_embeddings,
            self.extracted_embeddings
        )
        return self.similarity_matrix

    def identify_matches(self, threshold: float = 0.70) -> Dict:
        """
        Identifica matches considerando similitud semántica.

        Un match es válido si similarity >= threshold.

        Args:
            threshold: Umbral mínimo de similitud para considerar match

        Returns:
            Diccionario con diferentes tipos de matches
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        exact_matches = []
        partial_matches = []
        novel_terms = []
        predefined_not_found = []

        # Rastrear qué términos extraídos ya fueron emparejados
        matched_extracted_indices = set()

        # Para cada término predefinido, buscar mejor match
        for i, pred_term in enumerate(self.predefined_terms):
            # Obtener similitudes para este término predefinido
            similarities = self.similarity_matrix[i]
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]

            if max_sim >= threshold:
                # Match exacto/fuerte
                exact_matches.append({
                    'predefined': pred_term,
                    'extracted': self.extracted_terms[max_sim_idx],
                    'similarity': float(max_sim)
                })
                matched_extracted_indices.add(max_sim_idx)

            elif max_sim >= 0.5:
                # Match parcial
                partial_matches.append({
                    'predefined': pred_term,
                    'extracted': self.extracted_terms[max_sim_idx],
                    'similarity': float(max_sim)
                })
                matched_extracted_indices.add(max_sim_idx)

            else:
                # Término predefinido no encontrado
                predefined_not_found.append({
                    'predefined': pred_term,
                    'best_match': self.extracted_terms[max_sim_idx],
                    'similarity': float(max_sim)
                })

        # Identificar términos nuevos (no emparejados)
        for j, ext_term in enumerate(self.extracted_terms):
            if j not in matched_extracted_indices:
                # Calcular mejor similitud con predefinidos
                max_sim = float(np.max(self.similarity_matrix[:, j]))
                novel_terms.append({
                    'extracted': ext_term,
                    'max_similarity_to_predefined': max_sim
                })

        self.matches = {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'novel_terms': novel_terms,
            'predefined_not_found': predefined_not_found
        }

        return self.matches

    def calculate_metrics(self, matches: Dict = None) -> Dict:
        """
        Calcula métricas de evaluación.

        Precision = términos extraídos relevantes / total extraídos
        Recall = términos predefinidos encontrados / total predefinidos
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Args:
            matches: Diccionario de matches (usa self.matches si no se proporciona)

        Returns:
            Diccionario con métricas de evaluación
        """
        if matches is None:
            if self.matches is None:
                self.identify_matches()
            matches = self.matches

        n_predefined = len(self.predefined_terms)
        n_extracted = len(self.extracted_terms)

        # Términos relevantes = exact + partial matches
        n_relevant = len(matches['exact_matches']) + len(matches['partial_matches'])

        # Términos predefinidos encontrados
        n_found = len(matches['exact_matches']) + len(matches['partial_matches'])

        # Calcular métricas
        precision = n_relevant / n_extracted if n_extracted > 0 else 0
        recall = n_found / n_predefined if n_predefined > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        coverage = (n_found / n_predefined * 100) if n_predefined > 0 else 0

        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'coverage': coverage,
            'n_predefined': n_predefined,
            'n_extracted': n_extracted,
            'n_exact_matches': len(matches['exact_matches']),
            'n_partial_matches': len(matches['partial_matches']),
            'n_novel_terms': len(matches['novel_terms']),
            'n_predefined_not_found': len(matches['predefined_not_found'])
        }

        return self.metrics

    def explain_novel_terms(self, novel_terms: List[Dict], abstracts: List[str]) -> Dict:
        """
        Explica por qué los términos nuevos son relevantes.

        Para cada término nuevo:
        1. Encontrar contextos donde aparece (frases)
        2. Calcular score de relevancia (frecuencia, TF-IDF)
        3. Identificar aspecto que representa

        Args:
            novel_terms: Lista de términos nuevos del identify_matches
            abstracts: Lista de abstracts para buscar contextos

        Returns:
            Diccionario con análisis de cada término nuevo
        """
        explanations = {}

        # Combinar todos los abstracts en un corpus
        full_corpus = " ".join(abstracts)

        for term_info in novel_terms:
            term = term_info['extracted']

            # Encontrar contextos (frases que contienen el término)
            contexts = self._find_contexts(term, abstracts)

            # Calcular frecuencia
            frequency = full_corpus.lower().count(term.lower())

            # Score de relevancia basado en frecuencia y longitud del término
            # Términos más largos y específicos obtienen mayor score
            relevance_score = frequency * (1 + len(term.split()) * 0.2)

            explanations[term] = {
                'frequency': frequency,
                'example_contexts': contexts[:3],  # Top 3 contextos
                'relevance_score': relevance_score,
                'interpretation': self._interpret_term(term, contexts)
            }

        return explanations

    def _find_contexts(self, term: str, abstracts: List[str], window: int = 100) -> List[str]:
        """
        Encuentra contextos (fragmentos de texto) donde aparece el término.

        Args:
            term: Término a buscar
            abstracts: Lista de abstracts
            window: Caracteres antes/después del término

        Returns:
            Lista de contextos
        """
        contexts = []
        term_pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)

        for abstract in abstracts:
            for match in term_pattern.finditer(abstract):
                start = max(0, match.start() - window)
                end = min(len(abstract), match.end() + window)
                context = abstract[start:end].strip()

                # Limpiar el contexto
                context = re.sub(r'\s+', ' ', context)
                if context not in contexts:
                    contexts.append(f"...{context}...")

                if len(contexts) >= 5:
                    break

            if len(contexts) >= 5:
                break

        return contexts

    def _interpret_term(self, term: str, contexts: List[str]) -> str:
        """
        Genera una interpretación del término basada en sus contextos.

        Args:
            term: Término a interpretar
            contexts: Contextos donde aparece

        Returns:
            Interpretación textual
        """
        if not contexts:
            return "Término sin contexto suficiente para interpretación."

        # Análisis simple basado en palabras clave comunes
        context_text = " ".join(contexts).lower()

        interpretations = []

        if any(word in context_text for word in ['method', 'approach', 'technique', 'algorithm']):
            interpretations.append("método o técnica")

        if any(word in context_text for word in ['system', 'framework', 'architecture']):
            interpretations.append("sistema o framework")

        if any(word in context_text for word in ['challenge', 'problem', 'issue', 'limitation']):
            interpretations.append("desafío o problema")

        if any(word in context_text for word in ['application', 'use case', 'domain']):
            interpretations.append("dominio de aplicación")

        if interpretations:
            return f"Posible {', '.join(interpretations)}"
        else:
            return "Concepto específico del dominio"

    def generate_evaluation_report(self, output_path: str, abstracts: List[str] = None):
        """
        Genera reporte Markdown completo.

        Args:
            output_path: Ruta donde guardar el reporte
            abstracts: Lista de abstracts (para análisis de términos nuevos)
        """
        if self.matches is None:
            self.identify_matches()

        if self.metrics is None:
            self.calculate_metrics()

        # Generar visualizaciones
        viz_dir = output_path.replace('.md', '_visualizations')
        import os
        os.makedirs(viz_dir, exist_ok=True)

        venn_path = self._create_venn_diagram(viz_dir)
        matrix_path = self._create_similarity_heatmap(viz_dir)

        # Explicar términos nuevos si hay abstracts disponibles
        novel_explanations = {}
        if abstracts and self.matches['novel_terms']:
            novel_explanations = self.explain_novel_terms(
                self.matches['novel_terms'],
                abstracts
            )

        # Generar contenido del reporte
        report_content = self._build_report_content(
            venn_path, matrix_path, novel_explanations
        )

        # Guardar reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n✓ Reporte generado: {output_path}")
        print(f"✓ Visualizaciones guardadas en: {viz_dir}")

    def _create_venn_diagram(self, viz_dir: str) -> str:
        """Crea diagrama de Venn mostrando overlap de términos."""
        from matplotlib_venn import venn2

        fig, ax = plt.subplots(figsize=(10, 8))

        # Conjuntos
        n_predefined = self.metrics['n_predefined']
        n_extracted = self.metrics['n_extracted']
        n_overlap = self.metrics['n_exact_matches'] + self.metrics['n_partial_matches']

        # Elementos únicos
        predefined_only = n_predefined - n_overlap
        extracted_only = n_extracted - n_overlap

        venn = venn2(
            subsets=(predefined_only, extracted_only, n_overlap),
            set_labels=('Términos\nPredefinidos', 'Términos\nExtraídos'),
            ax=ax
        )

        # Personalizar colores
        venn.get_patch_by_id('10').set_color('#ff9999')
        venn.get_patch_by_id('01').set_color('#99ccff')
        venn.get_patch_by_id('11').set_color('#99ff99')

        plt.title('Overlap entre Términos Predefinidos y Extraídos',
                  fontsize=14, fontweight='bold', pad=20)

        path = os.path.join(viz_dir, 'venn_diagram.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path

    def _create_similarity_heatmap(self, viz_dir: str) -> str:
        """Crea heatmap de matriz de similitud."""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        # Limitar tamaño para visualización
        max_terms = 20
        matrix = self.similarity_matrix[:max_terms, :max_terms]
        pred_labels = [t[:30] for t in self.predefined_terms[:max_terms]]
        ext_labels = [t[:30] for t in self.extracted_terms[:max_terms]]

        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(
            matrix,
            xticklabels=ext_labels,
            yticklabels=pred_labels,
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Similitud Semántica'},
            ax=ax
        )

        plt.title('Matriz de Similitud Semántica\n(Predefinidos vs Extraídos)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Términos Extraídos', fontsize=12, fontweight='bold')
        plt.ylabel('Términos Predefinidos', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        path = os.path.join(viz_dir, 'similarity_heatmap.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path

    def _build_report_content(self, venn_path: str, matrix_path: str,
                             novel_explanations: Dict) -> str:
        """Construye el contenido completo del reporte Markdown."""

        content = f"""# Reporte de Evaluación de Precisión de Términos

## 1. Resumen Ejecutivo

### Métricas Principales

| Métrica | Valor |
|---------|-------|
| **Precision** | {self.metrics['precision']:.2%} |
| **Recall** | {self.metrics['recall']:.2%} |
| **F1-Score** | {self.metrics['f1_score']:.2%} |
| **Coverage** | {self.metrics['coverage']:.1f}% |

### Estadísticas de Términos

| Categoría | Cantidad |
|-----------|----------|
| Términos Predefinidos | {self.metrics['n_predefined']} |
| Términos Extraídos | {self.metrics['n_extracted']} |
| Matches Exactos | {self.metrics['n_exact_matches']} |
| Matches Parciales | {self.metrics['n_partial_matches']} |
| Términos Nuevos | {self.metrics['n_novel_terms']} |
| Predefinidos No Encontrados | {self.metrics['n_predefined_not_found']} |

---

## 2. Visualizaciones

### Diagrama de Venn
![Venn Diagram]({os.path.basename(venn_path)})

### Matriz de Similitud Semántica
![Similarity Heatmap]({os.path.basename(matrix_path)})

---

## 3. Análisis Detallado de Matches

### 3.1 Matches Exactos (Similitud ≥ 0.70)

"""

        # Tabla de matches exactos
        if self.matches['exact_matches']:
            content += "| Término Predefinido | Término Extraído | Similitud |\n"
            content += "|---------------------|------------------|----------|\n"

            for match in sorted(self.matches['exact_matches'],
                              key=lambda x: x['similarity'], reverse=True):
                content += f"| {match['predefined']} | {match['extracted']} | {match['similarity']:.3f} |\n"
        else:
            content += "*No se encontraron matches exactos.*\n"

        content += "\n### 3.2 Matches Parciales (Similitud 0.50-0.69)\n\n"

        if self.matches['partial_matches']:
            content += "| Término Predefinido | Término Extraído | Similitud |\n"
            content += "|---------------------|------------------|----------|\n"

            for match in sorted(self.matches['partial_matches'],
                              key=lambda x: x['similarity'], reverse=True):
                content += f"| {match['predefined']} | {match['extracted']} | {match['similarity']:.3f} |\n"
        else:
            content += "*No se encontraron matches parciales.*\n"

        content += "\n### 3.3 Términos Predefinidos No Encontrados\n\n"

        if self.matches['predefined_not_found']:
            content += "| Término Predefinido | Mejor Match | Similitud |\n"
            content += "|---------------------|-------------|----------|\n"

            for item in sorted(self.matches['predefined_not_found'],
                             key=lambda x: x['similarity'], reverse=True):
                content += f"| {item['predefined']} | {item['best_match']} | {item['similarity']:.3f} |\n"
        else:
            content += "*Todos los términos predefinidos fueron encontrados.*\n"

        content += "\n---\n\n## 4. Términos Nuevos Descubiertos\n\n"

        if self.matches['novel_terms']:
            content += f"Se descubrieron **{len(self.matches['novel_terms'])} términos nuevos** "
            content += "que no tienen correspondencia con los términos predefinidos.\n\n"

            if novel_explanations:
                # Ordenar por relevancia
                sorted_terms = sorted(
                    novel_explanations.items(),
                    key=lambda x: x[1]['relevance_score'],
                    reverse=True
                )

                for term, info in sorted_terms[:10]:  # Top 10
                    content += f"### `{term}`\n\n"
                    content += f"- **Frecuencia**: {info['frequency']} ocurrencias\n"
                    content += f"- **Score de Relevancia**: {info['relevance_score']:.2f}\n"
                    content += f"- **Interpretación**: {info['interpretation']}\n"

                    if info['example_contexts']:
                        content += f"\n**Ejemplos de contexto**:\n\n"
                        for ctx in info['example_contexts'][:2]:
                            content += f"> {ctx}\n\n"

                    content += "---\n\n"
            else:
                content += "| Término Nuevo | Similitud Máxima |\n"
                content += "|---------------|------------------|\n"
                for item in sorted(self.matches['novel_terms'],
                                 key=lambda x: x['max_similarity_to_predefined'],
                                 reverse=True)[:15]:
                    content += f"| {item['extracted']} | {item['max_similarity_to_predefined']:.3f} |\n"
        else:
            content += "*No se encontraron términos nuevos.*\n"

        content += "\n---\n\n## 5. Interpretación de Resultados\n\n"

        # Interpretación automática basada en métricas
        content += self._generate_interpretation()

        content += "\n---\n\n## 6. Recomendaciones\n\n"
        content += self._generate_recommendations()

        content += "\n---\n\n*Reporte generado automáticamente por TermPrecisionEvaluator*\n"

        return content

    def _generate_interpretation(self) -> str:
        """Genera interpretación automática de las métricas."""
        interpretation = ""

        precision = self.metrics['precision']
        recall = self.metrics['recall']
        f1 = self.metrics['f1_score']

        # Precision
        if precision >= 0.8:
            interpretation += "- **Precision Alta**: La mayoría de los términos extraídos son relevantes.\n"
        elif precision >= 0.6:
            interpretation += "- **Precision Moderada**: Hay una cantidad razonable de términos relevantes extraídos.\n"
        else:
            interpretation += "- **Precision Baja**: Muchos términos extraídos no son relevantes o no coinciden con predefinidos.\n"

        # Recall
        if recall >= 0.8:
            interpretation += "- **Recall Alto**: Se capturó la mayoría de los términos predefinidos.\n"
        elif recall >= 0.6:
            interpretation += "- **Recall Moderado**: Se capturó una parte razonable de términos predefinidos.\n"
        else:
            interpretation += "- **Recall Bajo**: Muchos términos predefinidos no fueron detectados.\n"

        # F1
        if f1 >= 0.7:
            interpretation += "- **Balance Bueno**: El sistema logra un buen equilibrio entre precision y recall.\n"
        else:
            interpretation += "- **Balance Mejorable**: Hay espacio para mejorar el equilibrio entre precision y recall.\n"

        # Novel terms
        novel_ratio = self.metrics['n_novel_terms'] / self.metrics['n_extracted'] if self.metrics['n_extracted'] > 0 else 0
        if novel_ratio > 0.3:
            interpretation += f"- **Descubrimiento**: {novel_ratio:.0%} de términos extraídos son nuevos, "
            interpretation += "indicando potencial descubrimiento de conceptos no considerados inicialmente.\n"

        return interpretation

    def _generate_recommendations(self) -> str:
        """Genera recomendaciones basadas en las métricas."""
        recommendations = ""

        if self.metrics['recall'] < 0.6:
            recommendations += "1. **Mejorar Recall**: Considerar ampliar la extracción de términos o "
            recommendations += "ajustar parámetros para capturar más términos predefinidos.\n\n"

        if self.metrics['precision'] < 0.6:
            recommendations += "2. **Mejorar Precision**: Aplicar filtros más estrictos o ajustar "
            recommendations += "umbrales de TF-IDF para reducir ruido.\n\n"

        if self.metrics['n_novel_terms'] > 10:
            recommendations += "3. **Revisar Términos Nuevos**: Evaluar términos nuevos para "
            recommendations += "potencialmente actualizar el conjunto de términos predefinidos.\n\n"

        if self.metrics['n_predefined_not_found'] > 0:
            recommendations += "4. **Analizar Términos Faltantes**: Investigar por qué algunos términos "
            recommendations += "predefinidos no fueron detectados (frecuencia baja, sinónimos, etc.).\n\n"

        if not recommendations:
            recommendations = "El sistema muestra un desempeño sólido. Continuar monitoreando métricas.\n"

        return recommendations


def main():
    """Función de prueba del evaluador."""

    # Ejemplo de uso
    predefined = [
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision"
    ]

    extracted = [
        "machine learning",
        "deep neural networks",
        "convolutional networks",
        "nlp techniques",
        "image recognition",
        "transformer models",
        "reinforcement learning"
    ]

    abstracts = [
        "Machine learning and deep neural networks have revolutionized NLP techniques.",
        "Computer vision tasks use convolutional networks for image recognition.",
        "Transformer models achieve state-of-the-art results in natural language processing."
    ]

    # Crear evaluador
    evaluator = TermPrecisionEvaluator(predefined, extracted)

    # Calcular similitud
    evaluator.calculate_similarity_matrix()

    # Identificar matches
    matches = evaluator.identify_matches(threshold=0.7)

    # Calcular métricas
    metrics = evaluator.calculate_metrics()

    print("\n" + "="*60)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*60)
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-Score:  {metrics['f1_score']:.2%}")
    print(f"Coverage:  {metrics['coverage']:.1f}%")
    print("="*60)

    # Generar reporte
    evaluator.generate_evaluation_report('evaluation_report.md', abstracts)


if __name__ == "__main__":
    main()
