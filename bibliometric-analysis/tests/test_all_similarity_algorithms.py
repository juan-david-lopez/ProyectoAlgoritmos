"""
Script de prueba para verificar los 6 algoritmos de similitud
"""

import sys
from pathlib import Path

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.algorithms.similarity import (
        LevenshteinSimilarity,
        TfidfCosineSimilarity,
        JaccardSimilarity,
        CharacterNgramSimilarity,
        WordEmbeddingSimilarity,
        SBERTSimilarity,
        TransformerSimilarity
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def test_all_algorithms():
    """Prueba todos los 6 algoritmos"""
    
    print("\n" + "="*80)
    print("  VERIFICACIÓN DE 6 ALGORITMOS DE SIMILITUD TEXTUAL")
    print("="*80 + "\n")
    
    # Textos de prueba
    text1 = "machine learning algorithms for data analysis"
    text2 = "data mining techniques and machine learning"
    text3 = "cooking recipes for italian food"
    
    print(f"📝 Textos de prueba:\n")
    print(f"  Text 1: {text1}")
    print(f"  Text 2: {text2}")
    print(f"  Text 3: {text3}\n")
    
    results = {}
    
    # =========================================================================
    # ALGORITMOS CLÁSICOS
    # =========================================================================
    
    print("━" * 80)
    print("  ALGORITMOS CLÁSICOS (4)")
    print("━" * 80 + "\n")
    
    # 1. Levenshtein
    print("1️⃣  Algoritmo 1: Distancia de Levenshtein")
    print("   Complejidad: O(m×n)")
    print("   Tipo: Edit distance\n")
    try:
        lev = LevenshteinSimilarity()
        sim12 = lev.compute_similarity(text1, text2)
        sim13 = lev.compute_similarity(text1, text3)
        results['Levenshtein'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 2. TF-IDF + Coseno
    print("2️⃣  Algoritmo 2: Similitud de Coseno con TF-IDF")
    print("   Complejidad: O(n×m)")
    print("   Tipo: Term frequency vectorization\n")
    try:
        tfidf = TfidfCosineSimilarity()
        tfidf.fit([text1, text2, text3])
        sim12 = tfidf.compute_similarity(text1, text2)
        sim13 = tfidf.compute_similarity(text1, text3)
        results['TF-IDF'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 3. Jaccard
    print("3️⃣  Algoritmo 3: Distancia de Jaccard")
    print("   Complejidad: O(n+m)")
    print("   Tipo: Set intersection\n")
    try:
        jac = JaccardSimilarity()
        sim12 = jac.compute_similarity(text1, text2)
        sim13 = jac.compute_similarity(text1, text3)
        results['Jaccard'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # 4. N-gramas
    print("4️⃣  Algoritmo 4: N-gramas de Caracteres")
    print("   Complejidad: O(m+n)")
    print("   Tipo: Character sequence matching\n")
    try:
        ngram = CharacterNgramSimilarity(n=3)
        sim12 = ngram.compute_similarity(text1, text2)
        sim13 = ngram.compute_similarity(text1, text3)
        results['N-grams'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # =========================================================================
    # ALGORITMOS CON IA
    # =========================================================================
    
    print("━" * 80)
    print("  ALGORITMOS CON IA (3)")
    print("━" * 80 + "\n")
    
    # 5. Word2Vec/GloVe
    print("5️⃣  Algoritmo 5: Embeddings (Word2Vec/GloVe)")
    print("   Complejidad: O(n×d)")
    print("   Tipo: Static word embeddings\n")
    try:
        emb = WordEmbeddingSimilarity(model_type='glove', dimensions=50)
        sim12 = emb.compute_similarity(text1, text2)
        sim13 = emb.compute_similarity(text1, text3)
        results['Word2Vec/GloVe'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ⚠️  Requiere instalación: pip install gensim torchtext")
        print(f"   ⚠️  Error: {e}\n")
    
    # 6. S-BERT
    print("6️⃣  Algoritmo 6: Sentence-BERT (SBERT)")
    print("   Complejidad: O(n²×d)")
    print("   Tipo: Sentence embeddings (fine-tuned for similarity)\n")
    try:
        sbert = SBERTSimilarity()
        sim12 = sbert.compute_similarity(text1, text2)
        sim13 = sbert.compute_similarity(text1, text3)
        results['SBERT'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ⚠️  Requiere instalación: pip install sentence-transformers")
        print(f"   ⚠️  Error: {e}\n")
    
    # 7. BERT Transformers
    print("7️⃣  Algoritmo 7: BERT Transformers")
    print("   Complejidad: O(n²×d×L)")
    print("   Tipo: Contextualized embeddings\n")
    try:
        bert = TransformerSimilarity(model_name='bert-base-uncased')
        sim12 = bert.compute_similarity(text1, text2)
        sim13 = bert.compute_similarity(text1, text3)
        results['BERT'] = (sim12, sim13)
        print(f"   ✓ Similarity(1,2): {sim12:.4f}")
        print(f"   ✓ Similarity(1,3): {sim13:.4f}\n")
    except Exception as e:
        print(f"   ⚠️  Requiere instalación: pip install transformers torch")
        print(f"   ⚠️  Error: {e}\n")
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    
    print("="*80)
    print("  RESUMEN DE RESULTADOS")
    print("="*80 + "\n")
    
    print(f"{'Algoritmo':<20} | {'Sim(1,2)':<12} | {'Sim(1,3)':<12} | {'Diferencia':<12}")
    print("-"*80)
    
    for algo, (sim12, sim13) in results.items():
        diff = sim12 - sim13
        print(f"{algo:<20} | {sim12:>10.4f}   | {sim13:>10.4f}   | {diff:>10.4f}")
    
    print("\n" + "="*80)
    print("  ANÁLISIS")
    print("="*80 + "\n")
    
    print("✅ Algoritmos funcionando correctamente si:")
    print("   • Similarity(1,2) > Similarity(1,3)")
    print("   • Text 1 y 2 son sobre ML/AI (similar)")
    print("   • Text 1 y 3 son sobre dominios diferentes (no similar)\n")
    
    # Verificar
    all_correct = all(sim12 > sim13 for sim12, sim13 in results.values())
    
    if all_correct:
        print("🎉 ¡TODOS LOS ALGORITMOS FUNCIONAN CORRECTAMENTE!")
    else:
        print("⚠️  Algunos algoritmos pueden necesitar ajustes")
    
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    results = test_all_algorithms()
