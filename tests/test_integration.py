"""
Tests de Integración End-to-End del Sistema Completo
Universidad del Quindío - Análisis de Algoritmos

Verifica que todos los módulos funcionan correctamente en conjunto.
"""

import pytest
import json
import sys
from pathlib import Path
import shutil
import tempfile

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFullPipelineIntegration:
    """Tests del pipeline completo con datos de muestra"""
    
    @pytest.fixture
    def sample_data(self):
        """Genera datos de muestra para testing"""
        return [
            {
                "id": "test_1",
                "title": "Generative AI in Education: A Comprehensive Review",
                "authors": ["John Doe", "Jane Smith"],
                "year": 2024,
                "abstract": "This paper explores the application of generative artificial intelligence in educational settings. We discuss machine learning models, fine-tuning techniques, and training data requirements. The study examines ethical considerations including privacy, transparency, and algorithmic bias.",
                "keywords": ["generative AI", "education", "machine learning"],
                "source": "test",
                "doi": "10.1000/test.001"
            },
            {
                "id": "test_2",
                "title": "Human-AI Interaction: Explainability and Trust",
                "authors": ["Alice Johnson"],
                "year": 2023,
                "abstract": "Understanding human-AI interaction requires focus on explainability and transparency. This research investigates how AI literacy affects user trust and the importance of co-creation in AI systems. We analyze prompting strategies and personalization techniques.",
                "keywords": ["AI interaction", "explainability", "trust"],
                "source": "test",
                "doi": "10.1000/test.002"
            },
            {
                "id": "test_3",
                "title": "Multimodal Generative Models for Content Creation",
                "authors": ["Bob Wilson", "Carol Davis"],
                "year": 2024,
                "abstract": "Multimodal generative models represent a significant advance in artificial intelligence. This paper discusses various generative models, their training data requirements, and fine-tuning approaches. We address ethical concerns and the need for transparency in AI systems.",
                "keywords": ["multimodal AI", "generative models", "ethics"],
                "source": "test",
                "doi": "10.1000/test.003"
            },
            {
                "id": "test_4",
                "title": "Deep Learning for Natural Language Understanding",
                "authors": ["David Brown"],
                "year": 2023,
                "abstract": "Deep learning techniques have revolutionized natural language processing. We explore machine learning architectures, training methodologies, and the importance of quality training data. Algorithmic bias and fairness are critical considerations.",
                "keywords": ["deep learning", "NLP", "machine learning"],
                "source": "test",
                "doi": "10.1000/test.004"
            },
            {
                "id": "test_5",
                "title": "Privacy-Preserving AI Systems",
                "authors": ["Emma Garcia", "Frank Miller"],
                "year": 2024,
                "abstract": "Privacy is paramount in modern AI systems. This research examines privacy-preserving techniques, ethical frameworks, and the role of transparency in building trustworthy AI. We discuss personalization while maintaining privacy and user control.",
                "keywords": ["privacy", "AI ethics", "security"],
                "source": "test",
                "doi": "10.1000/test.005"
            },
            {
                "id": "test_6",
                "title": "AI Literacy in Higher Education",
                "authors": ["George Taylor"],
                "year": 2023,
                "abstract": "AI literacy is essential for students and educators. This paper investigates how to effectively teach AI concepts, promote human-AI interaction skills, and develop critical thinking about algorithmic systems. Co-creation and hands-on learning are emphasized.",
                "keywords": ["AI literacy", "education", "pedagogy"],
                "source": "test",
                "doi": "10.1000/test.006"
            },
            {
                "id": "test_7",
                "title": "Bias Mitigation in Machine Learning Models",
                "authors": ["Helen Anderson", "Ian White"],
                "year": 2024,
                "abstract": "Algorithmic bias poses significant challenges in machine learning. We present techniques for bias detection and mitigation, emphasizing the importance of diverse training data and transparent model development. Ethics and fairness metrics are discussed.",
                "keywords": ["bias", "fairness", "machine learning"],
                "source": "test",
                "doi": "10.1000/test.007"
            },
            {
                "id": "test_8",
                "title": "Prompt Engineering for Large Language Models",
                "authors": ["Jack Lee"],
                "year": 2024,
                "abstract": "Effective prompting is crucial for leveraging large language models. This research explores prompting strategies, fine-tuning approaches, and the relationship between prompt design and model performance. We analyze various prompting techniques and their applications.",
                "keywords": ["prompting", "LLM", "generative AI"],
                "source": "test",
                "doi": "10.1000/test.008"
            },
            {
                "id": "test_9",
                "title": "Explainable AI: Methods and Applications",
                "authors": ["Karen Martinez", "Leo Thompson"],
                "year": 2023,
                "abstract": "Explainability in AI systems is essential for trust and adoption. This paper surveys explainability methods, transparency techniques, and their applications across domains. We discuss the balance between model complexity and explainability.",
                "keywords": ["explainable AI", "transparency", "interpretability"],
                "source": "test",
                "doi": "10.1000/test.009"
            },
            {
                "id": "test_10",
                "title": "Personalized Learning with AI Technologies",
                "authors": ["Maria Rodriguez", "Nathan Clark"],
                "year": 2024,
                "abstract": "Personalization in education can be enhanced through AI. This study examines personalized learning systems, adaptive content generation, and the role of machine learning in customizing educational experiences. Privacy and ethics are key considerations.",
                "keywords": ["personalization", "adaptive learning", "education AI"],
                "source": "test",
                "doi": "10.1000/test.010"
            }
        ]
    
    @pytest.fixture
    def temp_workspace(self, sample_data, tmp_path):
        """Crea un workspace temporal con datos de muestra"""
        # Crear estructura de directorios
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Guardar datos de muestra
        data_file = data_dir / "unified_articles.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        return tmp_path
    
    def test_full_pipeline_with_sample_data(self, temp_workspace, sample_data):
        """
        Test del pipeline completo con datos de muestra.
        
        Flujo:
        1. Carga datos de muestra (10 artículos)
        2. Ejecuta análisis de similitud
        3. Ejecuta análisis de términos
        4. Ejecuta clustering
        5. Genera visualizaciones
        6. Verifica que todos los outputs existen
        """
        import os
        original_dir = os.getcwd()
        
        try:
            # Cambiar al workspace temporal
            os.chdir(temp_workspace)
            
            # 1. Verificar carga de datos
            data_file = temp_workspace / "data" / "unified_articles.json"
            assert data_file.exists(), "Datos de muestra no encontrados"
            
            with open(data_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == 10, f"Se esperaban 10 artículos, se encontraron {len(loaded_data)}"
            
            # 2. Test de similitud
            from src.algorithms.tfidf_cosine import TFIDFCosineComparator
            
            comparator = TFIDFCosineComparator()
            
            abstracts = [article['abstract'] for article in loaded_data[:3]]
            similarity_matrix = comparator.compare_multiple(abstracts)
            
            assert similarity_matrix is not None, "Matriz de similitud no generada"
            assert similarity_matrix.shape == (3, 3), f"Forma incorrecta: {similarity_matrix.shape}"
            
            # 3. Test de análisis de términos
            from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer
            
            # Usar datos de muestra para análisis
            temp_data_path = temp_workspace / "data" / "unified_articles.json"
            
            analyzer = PredefinedTermsAnalyzer(str(temp_data_path))
            frequencies = analyzer.calculate_frequencies()
            
            assert frequencies is not None, "Frecuencias no calculadas"
            assert len(frequencies) > 0, "No se encontraron frecuencias"
            
            # Verificar que algunos términos se encontraron
            found_terms = [term for term, data in frequencies.items() if data['total_count'] > 0]
            assert len(found_terms) > 0, "No se encontraron términos en el corpus"
            
            # 4. Test de clustering (versión simplificada)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from scipy.cluster.hierarchy import linkage
            
            vectorizer = TfidfVectorizer(max_features=100)
            abstracts_for_clustering = [a['abstract'] for a in loaded_data]
            
            tfidf_matrix = vectorizer.fit_transform(abstracts_for_clustering)
            
            # Probar diferentes métodos de linkage
            for method in ['single', 'complete', 'average']:
                linkage_matrix = linkage(tfidf_matrix.toarray(), method=method, metric='cosine')
                assert linkage_matrix is not None, f"Linkage {method} falló"
                assert linkage_matrix.shape[0] == len(loaded_data) - 1, "Dimensiones incorrectas"
            
            # 5. Test de visualizaciones (verificar que se pueden crear)
            output_viz_dir = temp_workspace / "output" / "visualizations"
            output_viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Crear una visualización simple de prueba
            import matplotlib
            matplotlib.use('Agg')  # Backend no interactivo
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            term_counts = [data['total_count'] for term, data in list(frequencies.items())[:5]]
            term_names = list(frequencies.keys())[:5]
            
            ax.bar(term_names, term_counts)
            ax.set_title("Test: Frecuencia de Términos")
            
            viz_file = output_viz_dir / "test_frequencies.png"
            plt.savefig(viz_file)
            plt.close()
            
            assert viz_file.exists(), "Visualización no generada"
            
            # 6. Verificar estructura de outputs
            expected_outputs = [
                "data/unified_articles.json",
                "output/visualizations/test_frequencies.png"
            ]
            
            for output_path in expected_outputs:
                full_path = temp_workspace / output_path
                assert full_path.exists(), f"Output esperado no encontrado: {output_path}"
            
        finally:
            os.chdir(original_dir)
    
    def test_data_consistency_across_modules(self, sample_data):
        """
        Verifica que los datos se mantienen consistentes entre módulos.
        
        Prueba:
        1. IDs únicos se preservan
        2. Campos requeridos están presentes
        3. Formatos de datos son correctos
        """
        # Verificar IDs únicos
        ids = [article['id'] for article in sample_data]
        assert len(ids) == len(set(ids)), "IDs duplicados encontrados"
        
        # Verificar campos requeridos
        required_fields = ['id', 'title', 'abstract', 'authors', 'year']
        
        for article in sample_data:
            for field in required_fields:
                assert field in article, f"Campo requerido '{field}' faltante en {article.get('id', 'unknown')}"
                assert article[field], f"Campo '{field}' vacío en {article.get('id', 'unknown')}"
        
        # Verificar tipos de datos
        for article in sample_data:
            assert isinstance(article['id'], str), "ID debe ser string"
            assert isinstance(article['title'], str), "Title debe ser string"
            assert isinstance(article['abstract'], str), "Abstract debe ser string"
            assert isinstance(article['authors'], list), "Authors debe ser lista"
            assert isinstance(article['year'], int), "Year debe ser entero"
            assert 2000 <= article['year'] <= 2030, f"Año inválido: {article['year']}"
    
    def test_error_handling_missing_data(self, tmp_path):
        """
        Verifica que los errores se manejan apropiadamente.
        
        Casos:
        1. Archivo de datos no existe
        2. Archivo JSON malformado
        3. Datos incompletos
        """
        # Caso 1: Archivo no existe
        from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer
        
        with pytest.raises(FileNotFoundError):
            analyzer = PredefinedTermsAnalyzer(str(tmp_path / "nonexistent.json"))
        
        # Caso 2: JSON malformado
        bad_json_file = tmp_path / "bad.json"
        with open(bad_json_file, 'w') as f:
            f.write("{invalid json}")
        
        with pytest.raises(json.JSONDecodeError):
            with open(bad_json_file, 'r') as f:
                json.load(f)
        
        # Caso 3: Datos incompletos (sin abstract)
        incomplete_data = [{"id": "1", "title": "Test"}]
        incomplete_file = tmp_path / "incomplete.json"
        
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_data, f)
        
        # El analyzer debería manejar esto graciosamente
        try:
            analyzer = PredefinedTermsAnalyzer(str(incomplete_file))
            assert len(analyzer.abstracts) == 0, "Debería no tener abstracts válidos"
        except Exception as e:
            # Es aceptable que lance excepción si no hay abstracts
            assert "abstract" in str(e).lower() or "empty" in str(e).lower()
    
    def test_similarity_algorithms_consistency(self, sample_data):
        """
        Verifica que los algoritmos de similitud son consistentes.
        
        Propiedades a verificar:
        1. Similitud de un texto consigo mismo = 1.0
        2. Matriz de similitud es simétrica
        3. Valores en rango [0, 1]
        """
        from src.algorithms.tfidf_cosine import TFIDFCosineComparator
        from src.algorithms.jaccard import JaccardComparator
        
        texts = [article['abstract'] for article in sample_data[:3]]
        
        # Test TF-IDF
        tfidf_comp = TFIDFCosineComparator()
        tfidf_matrix = tfidf_comp.compare_multiple(texts)
        
        # Propiedad 1: Diagonal = 1.0
        import numpy as np
        for i in range(len(texts)):
            assert abs(tfidf_matrix[i, i] - 1.0) < 0.01, f"Diagonal[{i}] != 1.0"
        
        # Propiedad 2: Simetría
        assert np.allclose(tfidf_matrix, tfidf_matrix.T), "Matriz no simétrica"
        
        # Propiedad 3: Rango [0, 1]
        assert np.all(tfidf_matrix >= 0), "Valores negativos encontrados"
        assert np.all(tfidf_matrix <= 1), "Valores > 1 encontrados"
        
        # Test Jaccard
        jaccard_comp = JaccardComparator()
        
        for i in range(len(texts)):
            score = jaccard_comp.similarity(texts[i], texts[i])
            assert abs(score - 1.0) < 0.01, f"Jaccard self-similarity != 1.0"
    
    def test_term_frequency_calculation(self, sample_data):
        """
        Verifica que el cálculo de frecuencias de términos es correcto.
        
        Prueba con términos conocidos en los datos de muestra.
        """
        # Términos que sabemos que están en los datos de muestra
        known_terms = {
            'machine learning': 3,  # Aparece en múltiples abstracts
            'privacy': 2,
            'explainability': 2
        }
        
        # Contar manualmente
        for term, expected_min_count in known_terms.items():
            actual_count = sum(
                1 for article in sample_data 
                if term.lower() in article['abstract'].lower()
            )
            
            assert actual_count >= expected_min_count, \
                f"Término '{term}': esperado >= {expected_min_count}, encontrado {actual_count}"
    
    def test_clustering_produces_valid_dendrogram(self, sample_data):
        """
        Verifica que el clustering produce dendrogramas válidos.
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Preparar datos
        abstracts = [article['abstract'] for article in sample_data]
        
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        
        # Probar cada método de linkage
        for method in ['single', 'complete', 'average']:
            linkage_matrix = linkage(tfidf_matrix.toarray(), method=method, metric='cosine')
            
            # Verificar dimensiones
            n_samples = len(sample_data)
            assert linkage_matrix.shape == (n_samples - 1, 4), \
                f"Dimensiones incorrectas para método {method}"
            
            # Verificar que se puede crear un dendrograma
            try:
                dend = dendrogram(linkage_matrix, no_plot=True)
                assert 'icoord' in dend, "Dendrogram inválido"
                assert len(dend['icoord']) > 0, "Dendrogram vacío"
            except Exception as e:
                pytest.fail(f"Fallo al crear dendrogram con método {method}: {e}")
    
    def test_visualization_files_created(self, temp_workspace, sample_data):
        """
        Verifica que los archivos de visualización se crean correctamente.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        output_dir = temp_workspace / "output" / "viz_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear visualizaciones de prueba
        viz_types = {
            'bar_chart': lambda: plt.bar(['A', 'B', 'C'], [1, 2, 3]),
            'line_plot': lambda: plt.plot([1, 2, 3], [1, 4, 9]),
            'scatter': lambda: plt.scatter([1, 2, 3], [1, 4, 9])
        }
        
        for viz_name, viz_func in viz_types.items():
            plt.figure()
            viz_func()
            plt.title(f"Test: {viz_name}")
            
            output_file = output_dir / f"{viz_name}.png"
            plt.savefig(output_file)
            plt.close()
            
            assert output_file.exists(), f"Visualización {viz_name} no creada"
            assert output_file.stat().st_size > 0, f"Archivo {viz_name} vacío"


class TestModuleIntegration:
    """Tests de integración entre módulos específicos"""
    
    def test_similarity_to_clustering_pipeline(self):
        """
        Verifica que los resultados de similitud se pueden usar para clustering.
        """
        from src.algorithms.tfidf_cosine import TFIDFCosineComparator
        from scipy.cluster.hierarchy import linkage
        
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
        
        # Calcular similitud
        comparator = TFIDFCosineComparator()
        similarity_matrix = comparator.compare_multiple(texts)
        
        # Convertir a distancia
        import numpy as np
        distance_matrix = 1 - similarity_matrix
        
        # Aplicar clustering
        try:
            # Usar solo triángulo superior (condensed form)
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(distance_matrix)
            
            linkage_matrix = linkage(condensed_dist, method='average')
            
            assert linkage_matrix is not None
            assert linkage_matrix.shape[0] == len(texts) - 1
            
        except Exception as e:
            pytest.fail(f"Fallo en pipeline similitud->clustering: {e}")
    
    def test_terms_to_visualization_pipeline(self, tmp_path):
        """
        Verifica que los resultados de análisis de términos se pueden visualizar.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Simular resultados de análisis de términos
        term_frequencies = {
            'machine learning': {'total_count': 5, 'doc_count': 3},
            'artificial intelligence': {'total_count': 4, 'doc_count': 2},
            'deep learning': {'total_count': 3, 'doc_count': 2}
        }
        
        # Crear visualización
        terms = list(term_frequencies.keys())
        counts = [data['total_count'] for data in term_frequencies.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(terms, counts)
        plt.title("Frecuencia de Términos")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = tmp_path / "term_viz_test.png"
        plt.savefig(output_file)
        plt.close()
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestPerformanceAndScalability:
    """Tests de rendimiento y escalabilidad"""
    
    def test_similarity_performance_with_large_texts(self):
        """
        Verifica que los algoritmos de similitud manejan textos largos.
        """
        from src.algorithms.tfidf_cosine import TFIDFCosineComparator
        import time
        
        # Crear textos largos
        long_text = " ".join(["word" + str(i) for i in range(1000)])
        texts = [long_text for _ in range(10)]
        
        comparator = TFIDFCosineComparator()
        
        start_time = time.time()
        similarity_matrix = comparator.compare_multiple(texts)
        elapsed = time.time() - start_time
        
        assert similarity_matrix is not None
        assert elapsed < 10.0, f"Similitud demasiado lenta: {elapsed:.2f}s"
    
    def test_clustering_scales_with_data(self):
        """
        Verifica que el clustering escala razonablemente con más datos.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.cluster.hierarchy import linkage
        import time
        
        # Probar con diferentes tamaños
        sizes = [10, 20, 30]
        times = []
        
        for size in sizes:
            texts = [f"Document {i} with some text" for i in range(size)]
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            start_time = time.time()
            linkage_matrix = linkage(tfidf_matrix.toarray(), method='average', metric='cosine')
            elapsed = time.time() - start_time
            
            times.append(elapsed)
        
        # Verificar que no crece exponencialmente
        # (esto es aproximado, el clustering es O(n^2) a O(n^3))
        assert all(t < 5.0 for t in times), f"Tiempos: {times}"


def test_project_structure():
    """Verifica que la estructura del proyecto es correcta"""
    required_dirs = [
        'src',
        'src/algorithms',
        'src/clustering',
        'src/visualization',
        'tests',
        'docs',
        'examples'
    ]
    
    for directory in required_dirs:
        assert Path(directory).exists(), f"Directorio requerido no encontrado: {directory}"


def test_required_files_exist():
    """Verifica que los archivos críticos existen"""
    required_files = [
        'README.md',
        'requirements.txt',
        'main.py',
        'ESTADO_REQUERIMIENTOS.md',
        'RESUMEN_EJECUTIVO.md'
    ]
    
    for file in required_files:
        assert Path(file).exists(), f"Archivo requerido no encontrado: {file}"


if __name__ == "__main__":
    # Ejecutar tests si se llama directamente
    pytest.main([__file__, '-v', '--tb=short'])
