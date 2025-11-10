"""
Tests de Integración para Term Analysis Pipeline
Tests end-to-end del flujo completo de análisis.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path

from term_analysis_pipeline import (
    TermAnalysisPipeline,
    run_complete_analysis
)


@pytest.fixture
def sample_unified_data():
    """Crea datos de muestra unificados."""
    return {
        "metadata": {
            "query": "deep learning",
            "total_results": 5,
            "date_range": {
                "start": "2023-01-01",
                "end": "2024-12-31"
            },
            "search_engines": ["arXiv"]
        },
        "predefined_terms": [
            "deep learning",
            "neural networks",
            "machine learning",
            "computer vision",
            "natural language processing"
        ],
        "papers": [
            {
                "title": "Deep Learning for Computer Vision",
                "abstract": "Deep learning and neural networks revolutionize computer vision. Machine learning techniques enable image understanding through convolutional networks.",
                "year": 2023,
                "authors": ["Smith, J."],
                "doi": "10.1000/test1"
            },
            {
                "title": "Natural Language Processing with Neural Networks",
                "abstract": "Natural language processing benefits from deep learning. Neural networks and machine learning advance text understanding capabilities.",
                "year": 2023,
                "authors": ["Doe, A."],
                "doi": "10.1000/test2"
            },
            {
                "title": "Machine Learning Applications",
                "abstract": "Machine learning and deep learning transform various domains. Computer vision and natural language processing demonstrate significant advances.",
                "year": 2024,
                "authors": ["Johnson, M."],
                "doi": "10.1000/test3"
            },
            {
                "title": "Advances in Neural Networks",
                "abstract": "Neural networks continue evolving with new architectures. Deep learning methods improve computer vision and NLP performance.",
                "year": 2024,
                "authors": ["Lee, K."],
                "doi": "10.1000/test4"
            },
            {
                "title": "Deep Learning Survey",
                "abstract": "Comprehensive survey of deep learning techniques. Machine learning, neural networks, computer vision, and natural language processing covered.",
                "year": 2024,
                "authors": ["Garcia, R."],
                "doi": "10.1000/test5"
            }
        ]
    }


@pytest.fixture
def temp_unified_file(sample_unified_data):
    """Crea archivo temporal con datos unificados."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False,
        encoding='utf-8'
    ) as f:
        json.dump(sample_unified_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Crea directorio temporal para outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestPipelineInitialization:
    """Tests de inicialización del pipeline."""

    def test_pipeline_creation(self, temp_unified_file, temp_output_dir):
        """Test: Creación básica del pipeline."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)

        assert pipeline.unified_data_path == temp_unified_file
        assert pipeline.output_dir == temp_output_dir
        assert pipeline.data is None  # No cargado aún

    def test_directory_structure_creation(self, temp_unified_file, temp_output_dir):
        """Test: Creación de estructura de directorios."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)

        # Verificar que se crearon los subdirectorios
        assert os.path.exists(pipeline.viz_dir)
        assert os.path.exists(pipeline.data_dir)
        assert os.path.exists(pipeline.reports_dir)


class TestDataLoading:
    """Tests de carga de datos."""

    def test_load_data_success(self, temp_unified_file, temp_output_dir):
        """Test: Carga exitosa de datos."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        data = pipeline.load_data()

        assert data is not None
        assert 'metadata' in data
        assert 'papers' in data
        assert 'predefined_terms' in data

    def test_abstracts_extraction(self, temp_unified_file, temp_output_dir):
        """Test: Extracción de abstracts."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()

        assert pipeline.abstracts is not None
        assert len(pipeline.abstracts) == 5
        assert all(isinstance(a, str) for a in pipeline.abstracts)

    def test_predefined_terms_extraction(self, temp_unified_file, temp_output_dir):
        """Test: Extracción de términos predefinidos."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()

        assert pipeline.predefined_terms is not None
        assert len(pipeline.predefined_terms) == 5
        assert "deep learning" in pipeline.predefined_terms


class TestPart1Integration:
    """Tests de integración con Parte 1."""

    def test_analyze_predefined_terms(self, temp_unified_file, temp_output_dir):
        """Test: Análisis de términos predefinidos."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        results = pipeline.analyze_predefined_terms()

        # Verificar estructura de resultados
        assert 'frequencies' in results
        assert 'co_occurrences' in results
        assert 'report_path' in results
        assert 'freq_csv_path' in results
        assert 'top_10_terms' in results

    def test_frequencies_calculation(self, temp_unified_file, temp_output_dir):
        """Test: Cálculo correcto de frecuencias."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        results = pipeline.analyze_predefined_terms()

        frequencies = results['frequencies']

        # Verificar que hay frecuencias para todos los términos
        assert len(frequencies) > 0

        # Términos frecuentes deben tener conteos > 0
        assert frequencies.get('deep learning', 0) > 0
        assert frequencies.get('neural networks', 0) > 0

    def test_report_generation_part1(self, temp_unified_file, temp_output_dir):
        """Test: Generación de reporte Parte 1."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        results = pipeline.analyze_predefined_terms()

        report_path = results['report_path']

        # Verificar que el reporte existe
        assert os.path.exists(report_path)

        # Verificar contenido
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0
            assert 'Análisis de Términos Predefinidos' in content


class TestPart2Integration:
    """Tests de integración con Parte 2."""

    def test_extract_terms_automatically(self, temp_unified_file, temp_output_dir):
        """Test: Extracción automática de términos."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        results = pipeline.extract_terms_automatically()

        # Verificar estructura
        assert 'rake_terms' in results
        assert 'textrank_terms' in results
        assert 'combined_terms' in results
        assert 'report_path' in results
        assert 'csv_path' in results

    def test_extraction_methods(self, temp_unified_file, temp_output_dir):
        """Test: Ambos métodos extraen términos."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        results = pipeline.extract_terms_automatically()

        # Verificar que ambos métodos extrajeron términos
        assert len(results['rake_terms']) > 0
        assert len(results['textrank_terms']) > 0
        assert len(results['combined_terms']) > 0

    def test_csv_export_part2(self, temp_unified_file, temp_output_dir):
        """Test: Exportación a CSV de términos extraídos."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        results = pipeline.extract_terms_automatically()

        csv_path = results['csv_path']

        # Verificar que existe
        assert os.path.exists(csv_path)

        # Verificar contenido
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        assert 'term' in df.columns
        assert 'method' in df.columns


class TestPart3Integration:
    """Tests de integración con Parte 3."""

    def test_evaluate_precision(self, temp_unified_file, temp_output_dir):
        """Test: Evaluación de precisión."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        pipeline.analyze_predefined_terms()
        pipeline.extract_terms_automatically()
        results = pipeline.evaluate_precision()

        # Verificar que evalúa los 3 métodos
        assert 'RAKE' in results
        assert 'TextRank' in results
        assert 'Combined' in results

    def test_metrics_structure(self, temp_unified_file, temp_output_dir):
        """Test: Estructura de métricas."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        pipeline.analyze_predefined_terms()
        pipeline.extract_terms_automatically()
        results = pipeline.evaluate_precision()

        for method in ['RAKE', 'TextRank', 'Combined']:
            assert 'metrics' in results[method]
            metrics = results[method]['metrics']

            # Verificar métricas estándar
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'coverage' in metrics

    def test_metrics_ranges(self, temp_unified_file, temp_output_dir):
        """Test: Rangos válidos de métricas."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        pipeline.analyze_predefined_terms()
        pipeline.extract_terms_automatically()
        results = pipeline.evaluate_precision()

        for method in results.values():
            metrics = method['metrics']

            # Métricas en [0, 1]
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1

            # Coverage en [0, 100]
            assert 0 <= metrics['coverage'] <= 100


class TestVisualizations:
    """Tests de visualizaciones."""

    def test_create_visualizations(self, temp_unified_file, temp_output_dir):
        """Test: Creación de visualizaciones."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        pipeline.analyze_predefined_terms()
        pipeline.extract_terms_automatically()
        pipeline.evaluate_precision()
        pipeline.create_comparative_visualizations()

        # Verificar que se crearon las visualizaciones
        viz_files = [
            'metrics_comparison.png',
            'frequency_distribution.png',
            'methods_overlap.png',
            'top_terms_comparison.png'
        ]

        for viz_file in viz_files:
            path = os.path.join(pipeline.viz_dir, viz_file)
            assert os.path.exists(path), f"Falta: {viz_file}"
            assert os.path.getsize(path) > 0, f"Vacío: {viz_file}"


class TestMasterReport:
    """Tests del reporte maestro."""

    def test_generate_master_report(self, temp_unified_file, temp_output_dir):
        """Test: Generación de reporte maestro."""
        pipeline = TermAnalysisPipeline(temp_unified_file, temp_output_dir)
        pipeline.load_data()
        pipeline.analyze_predefined_terms()
        pipeline.extract_terms_automatically()
        pipeline.evaluate_precision()
        report_path = pipeline.generate_master_report()

        # Verificar que existe
        assert os.path.exists(report_path)

        # Verificar contenido
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Verificar secciones clave
            assert 'Reporte de Análisis Completo' in content
            assert 'Resumen Ejecutivo' in content
            assert 'RAKE' in content
            assert 'TextRank' in content
            assert 'Combined' in content
            assert 'Conclusiones' in content


class TestCompleteAnalysis:
    """Tests de función run_complete_analysis."""

    def test_run_complete_analysis_success(self, temp_unified_file, temp_output_dir):
        """Test: Ejecución completa exitosa."""
        pipeline = run_complete_analysis(temp_unified_file, temp_output_dir)

        # Verificar que pipeline tiene todos los resultados
        assert pipeline.data is not None
        assert pipeline.predefined_results is not None
        assert pipeline.extracted_results is not None
        assert pipeline.evaluation_results is not None

    def test_all_files_generated(self, temp_unified_file, temp_output_dir):
        """Test: Todos los archivos esperados se generan."""
        pipeline = run_complete_analysis(temp_unified_file, temp_output_dir)

        # Data files
        data_files = [
            'predefined_terms_frequencies.csv',
            'extracted_terms_all_methods.csv',
            'evaluation_metrics.json'
        ]

        for data_file in data_files:
            path = os.path.join(pipeline.data_dir, data_file)
            assert os.path.exists(path), f"Falta: {data_file}"

        # Reports
        report_files = [
            'term_analysis_report.md',
            'predefined_terms_report.md',
            'extracted_terms_report.md',
            'evaluation_rake.md',
            'evaluation_textrank.md',
            'evaluation_combined.md'
        ]

        for report_file in report_files:
            path = os.path.join(pipeline.reports_dir, report_file)
            assert os.path.exists(path), f"Falta: {report_file}"

        # Visualizations
        viz_files = [
            'metrics_comparison.png',
            'frequency_distribution.png',
            'methods_overlap.png',
            'top_terms_comparison.png'
        ]

        for viz_file in viz_files:
            path = os.path.join(pipeline.viz_dir, viz_file)
            assert os.path.exists(path), f"Falta: {viz_file}"


class TestEndToEnd:
    """Tests end-to-end completos."""

    def test_full_workflow(self, temp_unified_file, temp_output_dir):
        """Test: Workflow completo de principio a fin."""

        # 1. Ejecutar análisis completo
        pipeline = run_complete_analysis(temp_unified_file, temp_output_dir)

        # 2. Verificar que todos los componentes funcionaron
        assert len(pipeline.abstracts) > 0
        assert len(pipeline.predefined_terms) > 0

        # 3. Verificar resultados de Parte 1
        assert pipeline.predefined_results is not None
        assert len(pipeline.predefined_results['frequencies']) > 0

        # 4. Verificar resultados de Parte 2
        assert pipeline.extracted_results is not None
        assert len(pipeline.extracted_results['combined_terms']) > 0

        # 5. Verificar resultados de Parte 3
        assert pipeline.evaluation_results is not None

        for method in ['RAKE', 'TextRank', 'Combined']:
            assert method in pipeline.evaluation_results
            metrics = pipeline.evaluation_results[method]['metrics']
            assert 0 <= metrics['f1_score'] <= 1

        # 6. Verificar que se puede identificar mejor método
        best_method = max(
            pipeline.evaluation_results.keys(),
            key=lambda m: pipeline.evaluation_results[m]['metrics']['f1_score']
        )
        assert best_method in ['RAKE', 'TextRank', 'Combined']

        print(f"\n✓ Workflow completo ejecutado exitosamente")
        print(f"✓ Mejor método: {best_method}")

    def test_results_consistency(self, temp_unified_file, temp_output_dir):
        """Test: Consistencia entre resultados."""
        pipeline = run_complete_analysis(temp_unified_file, temp_output_dir)

        # Verificar que términos predefinidos coinciden
        assert (set(pipeline.predefined_terms) ==
                set(pipeline.predefined_results['frequencies'].keys()))

        # Verificar que número de términos es correcto
        n_rake = len(pipeline.extracted_results['rake_terms'])
        n_textrank = len(pipeline.extracted_results['textrank_terms'])
        n_combined = len(pipeline.extracted_results['combined_terms'])

        assert n_rake > 0
        assert n_textrank > 0
        assert n_combined > 0

        # Verificar métricas JSON vs pipeline
        metrics_json_path = os.path.join(
            pipeline.data_dir,
            'evaluation_metrics.json'
        )

        with open(metrics_json_path, 'r') as f:
            metrics_json = json.load(f)

        for method in ['RAKE', 'TextRank', 'Combined']:
            json_f1 = metrics_json[method]['f1_score']
            pipeline_f1 = pipeline.evaluation_results[method]['metrics']['f1_score']
            assert abs(json_f1 - pipeline_f1) < 0.001  # Tolerancia numérica


class TestErrorHandling:
    """Tests de manejo de errores."""

    def test_invalid_input_file(self, temp_output_dir):
        """Test: Archivo de entrada inválido."""
        with pytest.raises(FileNotFoundError):
            pipeline = TermAnalysisPipeline(
                'nonexistent_file.json',
                temp_output_dir
            )
            pipeline.load_data()

    def test_empty_abstracts(self, temp_output_dir):
        """Test: Manejo de abstracts vacíos."""
        # Crear archivo con abstracts vacíos
        data = {
            "metadata": {"query": "test", "total_results": 1},
            "predefined_terms": ["test"],
            "papers": [{"title": "Test", "abstract": "", "year": 2024}]
        }

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(data, f)
            temp_file = f.name

        try:
            pipeline = TermAnalysisPipeline(temp_file, temp_output_dir)
            pipeline.load_data()

            # Debe filtrar abstracts vacíos
            assert len(pipeline.abstracts) == 0

        finally:
            os.unlink(temp_file)


def run_integration_tests():
    """Ejecuta todos los tests de integración."""
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-k', 'Test'
    ])


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TESTS DE INTEGRACIÓN - PIPELINE COMPLETO")
    print("="*70 + "\n")

    run_integration_tests()
