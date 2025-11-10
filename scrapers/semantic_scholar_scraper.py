#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Scholar API Scraper
Descarga artículos académicos reales usando la API gratuita de Semantic Scholar
No requiere autenticación ni API key
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict

class SemanticScholarScraper:
    """Scraper para Semantic Scholar API"""

    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {
            "User-Agent": "BiblioMetric-Analysis-Bot (Universidad del Quindío)"
        }

    def search_papers(self, query: str, limit: int = 50, retry_delay: int = 5) -> List[Dict]:
        """
        Busca papers en Semantic Scholar con manejo de rate limiting

        Args:
            query: Término de búsqueda
            limit: Número máximo de resultados
            retry_delay: Segundos de espera entre reintentos

        Returns:
            Lista de artículos encontrados
        """
        print(f"\n[BUSQUEDA] Buscando articulos sobre: '{query}'")
        print(f"[INFO] Limite: {limit} articulos\n")

        # Delay inicial para evitar rate limiting
        print(f"[INFO] Esperando {retry_delay} segundos antes de consultar la API...")
        time.sleep(retry_delay)

        # Endpoint de búsqueda
        url = f"{self.base_url}/paper/search"

        # Parámetros de búsqueda - reducir limit para evitar rate limiting
        batch_size = min(limit, 100)  # API permite max 100 por request
        params = {
            "query": query,
            "limit": batch_size,
            "fields": "title,abstract,year,authors,citationCount,referenceCount,publicationTypes,externalIds,venue,publicationDate"
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Intento {attempt + 1}/{max_retries}...")
                response = requests.get(url, params=params, headers=self.headers, timeout=30)

                if response.status_code == 429:
                    wait_time = retry_delay * (attempt + 1) * 2
                    print(f"[WARN] Rate limit alcanzado. Esperando {wait_time} segundos...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                papers = data.get("data", [])
                print(f"[OK] Se encontraron {len(papers)} articulos")

                return papers

            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"[WARN] Error HTTP: {e}. Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Error en la busqueda despues de {max_retries} intentos: {e}")
                    return []
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Error en la busqueda: {e}")
                return []

        return []

    def extract_keywords(self, title: str, abstract: str) -> List[str]:
        """
        Extrae keywords relevantes del título y abstract

        Args:
            title: Título del artículo
            abstract: Abstract del artículo

        Returns:
            Lista de keywords extraídas
        """
        # Términos clave relacionados con IA generativa
        ai_terms = [
            "generative AI", "artificial intelligence", "machine learning",
            "deep learning", "neural networks", "transformers", "GPT",
            "large language models", "LLM", "generative models",
            "diffusion models", "GANs", "variational autoencoders",
            "natural language processing", "NLP", "computer vision",
            "text generation", "image generation", "multimodal"
        ]

        text = f"{title} {abstract}".lower()
        keywords = []

        for term in ai_terms:
            if term.lower() in text:
                keywords.append(term)

        return keywords[:10]  # Máximo 10 keywords

    def format_paper(self, paper: Dict, index: int) -> Dict:
        """
        Formatea un paper al formato esperado por el sistema

        Args:
            paper: Paper crudo de la API
            index: Índice del artículo

        Returns:
            Paper formateado
        """
        # Extraer información básica
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        year = paper.get("year", 2024)

        # Si no hay abstract, usar información adicional
        if not abstract or len(abstract) < 50:
            venue = paper.get("venue", "")
            citation_count = paper.get("citationCount", 0)
            abstract = f"This paper discusses {title}. Published in {venue}. Citations: {citation_count}."

        # Formatear autores
        authors = []
        for author in paper.get("authors", [])[:5]:  # Máximo 5 autores
            author_name = author.get("name", "Unknown")
            authors.append(author_name)

        if not authors:
            authors = ["Unknown Author"]

        # Extraer keywords
        keywords = self.extract_keywords(title, abstract)

        # Obtener DOI o Paper ID
        external_ids = paper.get("externalIds", {})
        doi = external_ids.get("DOI", "")
        paper_id = paper.get("paperId", f"ss_{index}")

        # Determinar fuente
        venue = paper.get("venue", "").lower()
        if "arxiv" in venue:
            source = "arxiv"
        elif "ieee" in venue:
            source = "ieee"
        elif "acm" in venue:
            source = "acm"
        elif "springer" in venue:
            source = "springer"
        else:
            source = "semantic_scholar"

        # Formato unificado
        formatted = {
            "id": f"article_{index}",
            "title": title,
            "abstract": abstract,
            "source": source,
            "year": year if year else 2024,
            "authors": authors,
            "keywords": keywords if keywords else ["artificial intelligence"],
            "doi": doi,
            "paper_id": paper_id,
            "citation_count": paper.get("citationCount", 0)
        }

        return formatted

    def download_papers(self, query: str = "generative artificial intelligence",
                       limit: int = 50,
                       output_file: str = "data/unified_articles.json") -> List[Dict]:
        """
        Descarga y guarda papers

        Args:
            query: Término de búsqueda
            limit: Número de artículos a descargar
            output_file: Ruta del archivo de salida

        Returns:
            Lista de artículos descargados
        """
        print("="*70)
        print("SEMANTIC SCHOLAR API SCRAPER")
        print("="*70)

        # Buscar papers
        raw_papers = self.search_papers(query, limit)

        if not raw_papers:
            print("\n[ERROR] No se encontraron articulos")
            return []

        # Formatear papers
        print("\n[PROCESO] Formateando articulos...")
        formatted_papers = []

        for i, paper in enumerate(raw_papers, 1):
            try:
                formatted = self.format_paper(paper, i)
                formatted_papers.append(formatted)

                # Progreso
                if i % 10 == 0:
                    print(f"   Procesados: {i}/{len(raw_papers)}")

            except Exception as e:
                print(f"[WARN] Error al formatear articulo {i}: {e}")
                continue

        print(f"[OK] Total formateados: {len(formatted_papers)} articulos")

        # Guardar archivo
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_papers, f, indent=2, ensure_ascii=False)

        print(f"\n[GUARDADO] Articulos guardados en: {output_path.absolute()}")
        print("\n" + "="*70)
        print("[OK] DESCARGA COMPLETADA")
        print("="*70)

        # Mostrar estadísticas
        print("\n[ESTADISTICAS]:")
        print(f"   - Total de articulos: {len(formatted_papers)}")

        years = [p['year'] for p in formatted_papers if p.get('year')]
        if years:
            print(f"   - Rango de años: {min(years)} - {max(years)}")

        sources = {}
        for p in formatted_papers:
            source = p.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        print(f"   - Fuentes:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"      - {source}: {count} articulos")

        total_citations = sum(p.get('citation_count', 0) for p in formatted_papers)
        print(f"   - Total de citaciones: {total_citations:,}")

        return formatted_papers


def main():
    """Función principal"""
    scraper = SemanticScholarScraper()

    # Configuración
    query = "generative artificial intelligence"
    limit = 50  # Puedes ajustar este número
    output_file = "data/unified_articles.json"

    # Descargar
    papers = scraper.download_papers(query, limit, output_file)

    if papers:
        print(f"\n[OK] Proceso completado exitosamente")
        print(f"\n[SIGUIENTE PASO]:")
        print(f"   Ejecuta el menú interactivo:")
        print(f"   python menu_interactivo.py")
        print(f"\n   O ejecuta el pipeline completo:")
        print(f"   python main.py similarity")
        print(f"   python main.py extract-terms")
        print(f"   python main.py cluster")
        print(f"   python main.py visualize")
    else:
        print(f"\n[ERROR] No se descargaron articulos")


if __name__ == "__main__":
    main()
