"""
Demonstration of Standalone Similarity Functions

This script shows how to use the simplified standalone functions
for quick similarity calculations without instantiating classes.

Usage:
    python examples/standalone_functions_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.similarity import sbert_similarity, bert_similarity
from loguru import logger


def demo_sbert_standalone():
    """Demonstrate SBERT standalone function"""
    print("\n" + "=" * 80)
    print("  SBERT STANDALONE FUNCTION DEMO")
    print("=" * 80 + "\n")

    # Example 1: Paraphrase detection
    print("1. PARAPHRASE DETECTION")
    print("-" * 80)

    text1 = "The cat sits on the mat"
    text2 = "A feline rests on a rug"

    sim = sbert_similarity(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"→ Similarity: {sim:.4f} (HIGH - paraphrase detected!)")
    print()

    # Example 2: Related concepts
    print("2. RELATED CONCEPTS")
    print("-" * 80)

    text1 = "machine learning algorithms"
    text2 = "artificial intelligence techniques"

    sim = sbert_similarity(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"→ Similarity: {sim:.4f} (MODERATE - related concepts)")
    print()

    # Example 3: Different topics
    print("3. DIFFERENT TOPICS")
    print("-" * 80)

    text1 = "machine learning research"
    text2 = "cooking Italian food"

    sim = sbert_similarity(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"→ Similarity: {sim:.4f} (LOW - different topics)")
    print()

    # Example 4: Batch comparison
    print("4. BATCH COMPARISON")
    print("-" * 80)

    reference = "generative artificial intelligence"
    candidates = [
        "generative AI models",
        "deep learning networks",
        "cooking pasta recipes",
        "artificial intelligence systems"
    ]

    print(f"Reference: '{reference}'")
    print("\nComparing against candidates:")

    for candidate in candidates:
        sim = sbert_similarity(reference, candidate)
        print(f"  [{sim:.3f}] {candidate}")

    print()


def demo_bert_standalone():
    """Demonstrate BERT standalone function"""
    print("\n" + "=" * 80)
    print("  BERT STANDALONE FUNCTION DEMO")
    print("=" * 80 + "\n")

    # Example 1: Contextual understanding
    print("1. CONTEXTUAL UNDERSTANDING OF 'BANK'")
    print("-" * 80)

    text1 = "I went to the bank to deposit money"
    text2 = "We sat by the river bank"
    text3 = "The financial institution is closed"

    sim_12 = bert_similarity(text1, text2)
    sim_13 = bert_similarity(text1, text3)

    print(f"Text 1: '{text1}' (financial bank)")
    print(f"Text 2: '{text2}' (river bank)")
    print(f"Text 3: '{text3}' (financial institution)")
    print()
    print(f"Similarity (1 vs 2): {sim_12:.4f} (different meaning of 'bank')")
    print(f"Similarity (1 vs 3): {sim_13:.4f} (same meaning, different words)")
    print()
    print("✓ BERT understands context and word sense!")
    print()

    # Example 2: Semantic similarity
    print("2. SEMANTIC SIMILARITY")
    print("-" * 80)

    text1 = "The weather is nice today"
    text2 = "Today is a beautiful day"

    sim = bert_similarity(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"→ Similarity: {sim:.4f} (similar meaning)")
    print()


def demo_comparison():
    """Compare SBERT vs BERT"""
    print("\n" + "=" * 80)
    print("  SBERT vs BERT COMPARISON")
    print("=" * 80 + "\n")

    text1 = "machine learning algorithms"
    text2 = "deep learning techniques"

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print()

    # SBERT (optimized for similarity)
    sbert_sim = sbert_similarity(text1, text2)
    print(f"SBERT similarity: {sbert_sim:.4f}")

    # BERT (contextual embeddings)
    bert_sim = bert_similarity(text1, text2)
    print(f"BERT similarity:  {bert_sim:.4f}")
    print()

    print("Note:")
    print("  - SBERT: Optimized for semantic similarity (faster)")
    print("  - BERT: Better for contextual understanding (more detailed)")
    print()


def main():
    """Main demonstration function"""

    # Setup logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "STANDALONE SIMILARITY FUNCTIONS DEMO" + " " * 27 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Run demos
    try:
        demo_sbert_standalone()
    except Exception as e:
        print(f"\n⚠️  SBERT demo skipped: {e}")
        print("Install with: pip install sentence-transformers")

    try:
        demo_bert_standalone()
    except Exception as e:
        print(f"\n⚠️  BERT demo skipped: {e}")
        print("Install with: pip install transformers torch")

    try:
        demo_comparison()
    except Exception as e:
        print(f"\n⚠️  Comparison demo skipped: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80 + "\n")

    print("Standalone Functions Usage:")
    print()
    print("  # Quick SBERT similarity")
    print("  from src.algorithms.similarity import sbert_similarity")
    print("  sim = sbert_similarity('text1', 'text2')")
    print()
    print("  # Quick BERT similarity")
    print("  from src.algorithms.similarity import bert_similarity")
    print("  sim = bert_similarity('text1', 'text2')")
    print()
    print("Advantages:")
    print("  ✓ No need to instantiate classes")
    print("  ✓ Model caching for repeated calls")
    print("  ✓ Simple API for quick calculations")
    print("  ✓ Perfect for scripting and notebooks")
    print()
    print("When to use classes instead:")
    print("  → Need advanced features (batch processing, matrices)")
    print("  → Want more control over model loading")
    print("  → Building larger applications")
    print()

    print("=" * 80)
    print("Done!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
