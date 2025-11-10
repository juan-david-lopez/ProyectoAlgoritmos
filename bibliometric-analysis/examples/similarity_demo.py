"""
Similarity Algorithms Demonstration Script

This script demonstrates all 6 similarity algorithms with comprehensive examples,
showing their strengths, weaknesses, and appropriate use cases.

Usage:
    python examples/similarity_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.similarity import (
    LevenshteinSimilarity,
    TfidfCosineSimilarity,
    JaccardSimilarity,
    CharacterNgramSimilarity,
    SBERTSimilarity,
    TransformerSimilarity,
    SimilarityComparator
)
from loguru import logger
import numpy as np


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subheader(title: str):
    """Print formatted subsection header"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80 + "\n")


def demo_levenshtein():
    """Demonstrate Levenshtein similarity"""
    print_header("1. LEVENSHTEIN SIMILARITY - Edit Distance")

    calc = LevenshteinSimilarity(debug=False)

    examples = [
        ("kitten", "sitting", "Classic example - multiple edits"),
        ("machine", "machien", "Typo - transposed letters"),
        ("hello", "hello", "Identical strings"),
        ("abc", "xyz", "Completely different"),
        ("Machine Learning", "machine learning", "Case difference"),
    ]

    print("Levenshtein measures minimum edits (insertions, deletions, substitutions):")
    print()

    for text1, text2, description in examples:
        distance, _ = calc.compute_distance(text1, text2, case_sensitive=False)
        similarity = calc.compute_similarity(text1, text2, case_sensitive=False)

        print(f"'{text1}' vs '{text2}'")
        print(f"  → Distance: {distance}, Similarity: {similarity:.4f} - {description}")
        print()

    print("✓ Best for: Typo detection, spell checking, fuzzy string matching")
    print("✗ Limitation: Character-level only, no semantic understanding")


def demo_tfidf():
    """Demonstrate TF-IDF cosine similarity"""
    print_header("2. TF-IDF COSINE SIMILARITY - Term Frequency")

    calc = TfidfCosineSimilarity(ngram_range=(1, 2), debug=False)

    examples = [
        ("machine learning algorithms", "machine learning techniques", "Similar topic, different word"),
        ("deep neural networks", "shallow decision trees", "Same domain, different concepts"),
        ("python programming", "cooking recipes", "Completely different domains"),
    ]

    print("TF-IDF weights terms by frequency and importance across documents:")
    print()

    for text1, text2, description in examples:
        similarity = calc.compute_similarity(text1, text2)

        # Get explanation
        explanation = calc.explain_similarity(text1, text2, top_n=3)

        print(f"'{text1}' vs '{text2}'")
        print(f"  → Similarity: {similarity:.4f} - {description}")
        print(f"  Common terms ({explanation['common_features']} total):")

        for feat in explanation['top_contributing_features'][:3]:
            print(f"    - '{feat['term']}': contribution={feat['contribution']:.4f}")
        print()

    print("✓ Best for: Document similarity, keyword matching, information retrieval")
    print("✗ Limitation: No word order, no synonyms, requires corpus context")


def demo_jaccard():
    """Demonstrate Jaccard similarity"""
    print_header("3. JACCARD SIMILARITY - Set-Based Comparison")

    calc_word = JaccardSimilarity(n=1, char_level=False, debug=False)
    calc_bigram = JaccardSimilarity(n=2, char_level=False, debug=False)

    text1 = "the cat sat on the mat"
    text2 = "the dog sat on the log"

    print(f"Comparing: '{text1}' vs '{text2}'")
    print()

    # Word unigrams
    sim_word = calc_word.compute_similarity(text1, text2)
    common_word = calc_word.get_common_ngrams(text1, text2)

    print(f"Word Unigrams (n=1):")
    print(f"  → Similarity: {sim_word:.4f}")
    print(f"  → Common words: {common_word}")
    print()

    # Word bigrams
    sim_bigram = calc_bigram.compute_similarity(text1, text2)
    common_bigram = calc_bigram.get_common_ngrams(text1, text2)

    print(f"Word Bigrams (n=2):")
    print(f"  → Similarity: {sim_bigram:.4f}")
    print(f"  → Common bigrams: {common_bigram}")
    print()

    print("✓ Best for: Deduplication, quick similarity checks, sparse data")
    print("✗ Limitation: Sensitive to vocabulary size, no semantic understanding")


def demo_character_ngram():
    """Demonstrate character n-gram similarity"""
    print_header("4. CHARACTER N-GRAM SIMILARITY - Fuzzy Matching")

    calc = CharacterNgramSimilarity(n=3, remove_spaces=True, debug=False)

    typo_examples = [
        ("neural", "nerual", "Transposed letters"),
        ("machine learning", "machine learnig", "Missing letter"),
        ("artificial intelligence", "artificial inteligence", "Typo"),
    ]

    print("Character n-grams capture substring overlap (good for typos):")
    print()

    for text1, text2, description in typo_examples:
        similarity = calc.compute_similarity(text1, text2)
        common = calc.get_common_ngrams(text1, text2)

        print(f"'{text1}' vs '{text2}'")
        print(f"  → Similarity: {similarity:.4f} - {description}")
        print(f"  → Common trigrams: {len(common)} total")
        print()

    # Typo detection demo
    print("\nTYPO CORRECTION DEMO:")
    dictionary = ["machine", "learning", "artificial", "intelligence", "neural", "network"]
    typos = ["machien", "learnig", "artifical", "inteligence", "nerual"]

    for typo in typos:
        matches = calc.find_typos(typo, dictionary, threshold=0.3, top_n=1)
        if matches:
            print(f"  '{typo:15s}' → '{matches[0][0]:15s}' (confidence: {matches[0][1]:.3f})")

    print()
    print("✓ Best for: Typo correction, fuzzy search, partial name matching")
    print("✗ Limitation: Computationally expensive for large datasets")


def demo_sbert():
    """Demonstrate SBERT similarity"""
    print_header("5. SBERT SIMILARITY - Semantic Embeddings")

    try:
        calc = SBERTSimilarity(debug=False)

        print("SBERT captures semantic meaning using pre-trained sentence embeddings:")
        print()

        # Paraphrase detection
        print("PARAPHRASE DETECTION:")
        text1 = "The cat sits on the mat"
        text2 = "A feline rests on a rug"
        text3 = "Python is a programming language"

        sim_12 = calc.compute_similarity(text1, text2)
        sim_13 = calc.compute_similarity(text1, text3)

        print(f"  '{text1}'")
        print(f"  vs '{text2}'")
        print(f"  → Similarity: {sim_12:.4f} (paraphrase - HIGH)")
        print()
        print(f"  '{text1}'")
        print(f"  vs '{text3}'")
        print(f"  → Similarity: {sim_13:.4f} (different topic - LOW)")
        print()

        # Semantic search
        print("SEMANTIC SEARCH DEMO:")
        query = "machine learning research"
        corpus = [
            "Deep learning is a subset of machine learning",
            "Python is a popular programming language",
            "Neural networks are inspired by the brain",
            "Cooking Italian food requires fresh ingredients",
            "Generative AI models can create new content"
        ]

        results = calc.find_most_similar(query, corpus, top_k=3)

        print(f"  Query: '{query}'")
        print(f"  Top 3 results:")
        for rank, (idx, text, sim) in enumerate(results, 1):
            print(f"    {rank}. [{sim:.3f}] {text}")

        print()
        print("✓ Best for: Semantic search, paraphrase detection, question answering")
        print("✗ Limitation: Requires GPU for speed, model download ~80MB")

    except Exception as e:
        print(f"SBERT not available: {e}")
        print("Install with: pip install sentence-transformers")


def demo_bert():
    """Demonstrate BERT similarity"""
    print_header("6. BERT TRANSFORMER SIMILARITY - Contextual Embeddings")

    try:
        calc = TransformerSimilarity(debug=False)

        print("BERT provides contextual word embeddings using transformer architecture:")
        print()

        # Context-aware understanding
        print("CONTEXTUAL UNDERSTANDING:")
        text1 = "I went to the bank to deposit money"
        text2 = "We sat by the river bank"
        text3 = "The financial institution is closed today"

        sim_12 = calc.compute_similarity(text1, text2)
        sim_13 = calc.compute_similarity(text1, text3)

        print(f"  '{text1}'")
        print(f"  vs '{text2}'")
        print(f"  → Similarity: {sim_12:.4f} (same word 'bank', different meaning)")
        print()
        print(f"  '{text1}'")
        print(f"  vs '{text3}'")
        print(f"  → Similarity: {sim_13:.4f} (different words, same meaning)")
        print()

        print("Note: BERT understands that 'bank' has different meanings in different contexts!")
        print()

        print("✓ Best for: Context-aware similarity, transfer learning, fine-tuning")
        print("✗ Limitation: Slower than SBERT, requires more memory (~400MB)")

    except Exception as e:
        print(f"BERT not available: {e}")
        print("Install with: pip install transformers torch")


def demo_comparison():
    """Demonstrate comparison of all algorithms"""
    print_header("7. ALGORITHM COMPARISON")

    # Create comparator (AI models optional)
    print("Initializing similarity comparator...")
    comparator = SimilarityComparator(
        output_dir='outputs/similarity_demo',
        enable_ai_models=True,  # Set to False for faster demo
        debug=False
    )

    # Test texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
        "The weather is nice and sunny today"
    ]

    print(f"\nComparing {len(texts)} texts using {len(comparator.algorithms)} algorithms...")
    print()

    # Compare all algorithms
    results = comparator.compare_all_algorithms(texts)

    # Display results
    print("\nPERFORMANCE SUMMARY:")
    print(f"{'Algorithm':<20s} {'Time (s)':<12s} {'Memory (MB)':<15s} {'Mean Sim':<12s}")
    print("-" * 65)

    for algo_name, algo_data in results['algorithms'].items():
        print(f"{algo_name.upper():<20s} "
              f"{algo_data['execution_time']:<12.3f} "
              f"{algo_data['memory_mb']:<15.2f} "
              f"{algo_data['mean_similarity']:<12.4f}")

    print()

    # Generate visualizations and report
    print("\nGenerating visualizations and report...")
    comparator.generate_visualizations(save=True)
    comparator.generate_report(format='markdown')

    print(f"\n✓ Analysis complete! Check: {comparator.output_dir}")


def main():
    """Main demonstration function"""

    # Setup logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/similarity_demo.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "SIMILARITY ALGORITHMS DEMONSTRATION" + " " * 23 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Run demos for each algorithm
    demo_levenshtein()
    demo_tfidf()
    demo_jaccard()
    demo_character_ngram()
    demo_sbert()
    demo_bert()
    demo_comparison()

    # Final summary
    print_header("DEMONSTRATION COMPLETE")

    print("Summary of Similarity Algorithms:")
    print()
    print("1. Levenshtein     - Edit distance (character-level)")
    print("2. TF-IDF          - Term frequency weighting")
    print("3. Jaccard         - Set intersection")
    print("4. Character N-gram - Substring matching")
    print("5. SBERT           - Semantic embeddings (AI)")
    print("6. BERT            - Contextual embeddings (AI)")
    print()
    print("Choose the right algorithm for your use case:")
    print("  • Typos/Fuzzy matching → Levenshtein, Character N-gram")
    print("  • Document similarity → TF-IDF, Jaccard")
    print("  • Semantic search → SBERT, BERT")
    print("  • Fast performance → TF-IDF, Jaccard")
    print("  • Highest accuracy → SBERT (semantic understanding)")
    print()
    print("For bibliometric analysis, we recommend:")
    print("  • Abstract similarity: SBERT (semantic)")
    print("  • Title matching: TF-IDF or Jaccard (fast)")
    print("  • Duplicate detection: Levenshtein + TF-IDF (combined)")
    print()

    print("=" * 80)
    print("Done! Check 'outputs/similarity_demo/' for detailed reports and visualizations.")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
