"""
Jaccard Similarity Algorithm

MATHEMATICAL EXPLANATION:
========================

The Jaccard similarity coefficient (also known as Jaccard index or Intersection over Union)
measures similarity between finite sets by comparing the size of their intersection
to the size of their union.

FORMAL DEFINITION:
------------------

For two sets A and B, the Jaccard similarity coefficient is:

                |A ∩ B|
    J(A, B) = ─────────
              |A ∪ B|

where:
    - A ∩ B = intersection of A and B (elements in both sets)
    - A ∪ B = union of A and B (all unique elements in either set)
    - |X| = cardinality (size) of set X

Alternative formulation:

                |A ∩ B|
    J(A, B) = ─────────────────
              |A| + |B| - |A ∩ B|

Properties:
    - J(A, B) ∈ [0, 1]
    - J(A, B) = 1 iff A = B (identical sets)
    - J(A, B) = 0 iff A ∩ B = ∅ (disjoint sets)
    - J(A, B) = J(B, A) (symmetric)

JACCARD DISTANCE:
-----------------
The Jaccard distance is the complement of Jaccard similarity:

    d(A, B) = 1 - J(A, B)

TEXT SIMILARITY WITH N-GRAMS:
------------------------------

For text comparison, we convert strings to sets of n-grams:
- Unigrams (n=1): individual words
- Bigrams (n=2): consecutive word pairs
- Trigrams (n=3): consecutive word triplets

EXAMPLE CALCULATION:
====================

Text 1: "the cat sat on the mat"
Text 2: "the dog sat on the log"

Using word-level unigrams (n=1):

Step 1: Create word sets
    A = {the, cat, sat, on, mat}
    B = {the, dog, sat, on, log}

Step 2: Compute intersection
    A ∩ B = {the, sat, on}
    |A ∩ B| = 3

Step 3: Compute union
    A ∪ B = {the, cat, sat, on, mat, dog, log}
    |A ∪ B| = 7

Step 4: Calculate Jaccard similarity
    J(A, B) = 3 / 7 = 0.4286

Using word-level bigrams (n=2):

Step 1: Create bigram sets
    A = {(the,cat), (cat,sat), (sat,on), (on,the), (the,mat)}
    B = {(the,dog), (dog,sat), (sat,on), (on,the), (the,log)}

Step 2: Compute intersection
    A ∩ B = {(sat,on), (on,the)}
    |A ∩ B| = 2

Step 3: Compute union
    |A ∪ B| = 8

Step 4: Calculate Jaccard similarity
    J(A, B) = 2 / 8 = 0.25

COMPLEXITY:
-----------
Time:  O(|A| + |B|) for set operations
Space: O(|A| + |B|) for storing sets
"""

from typing import Set, List, Tuple
from loguru import logger
import time
import re


class JaccardSimilarity:
    """
    Jaccard Similarity calculator for text comparison

    Uses n-gram tokenization to convert texts into sets and computes
    Jaccard similarity coefficient.
    """

    def __init__(
        self,
        n: int = 1,
        char_level: bool = False,
        case_sensitive: bool = False,
        debug: bool = False
    ):
        """
        Initialize Jaccard Similarity calculator

        Args:
            n: N-gram size (1=unigrams, 2=bigrams, 3=trigrams, etc.)
            char_level: If True, use character-level n-grams; if False, use word-level
            case_sensitive: Whether to consider case in comparison
            debug: If True, prints intermediate calculation steps

        Example:
            >>> # Word-level bigrams
            >>> calc = JaccardSimilarity(n=2, char_level=False)
            >>>
            >>> # Character-level trigrams
            >>> calc = JaccardSimilarity(n=3, char_level=True)
        """
        self.n = n
        self.char_level = char_level
        self.case_sensitive = case_sensitive
        self.debug = debug

        logger.info(f"Jaccard Similarity initialized (n={n}, char_level={char_level})")

    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        # Normalize case if needed
        if not self.case_sensitive:
            text = text.lower()

        # Simple word tokenization (split on whitespace and punctuation)
        words = re.findall(r'\b\w+\b', text)

        return words

    def _create_ngrams(self, items: List, n: int) -> Set[Tuple]:
        """
        Create n-grams from a list of items

        Args:
            items: List of items (words or characters)
            n: N-gram size

        Returns:
            Set of n-gram tuples

        Example:
            >>> calc = JaccardSimilarity()
            >>> items = ['the', 'cat', 'sat']
            >>> ngrams = calc._create_ngrams(items, 2)
            >>> print(ngrams)
            {('the', 'cat'), ('cat', 'sat')}
        """
        if len(items) < n:
            # If text is shorter than n, return the whole sequence as one n-gram
            return {tuple(items)} if items else set()

        ngrams = set()
        for i in range(len(items) - n + 1):
            ngram = tuple(items[i:i + n])
            ngrams.add(ngram)

        return ngrams

    def _text_to_ngrams(self, text: str) -> Set[Tuple]:
        """
        Convert text to set of n-grams

        Args:
            text: Input text

        Returns:
            Set of n-gram tuples
        """
        if self.char_level:
            # Character-level n-grams
            if not self.case_sensitive:
                text = text.lower()

            # Remove whitespace for character-level
            chars = [c for c in text if not c.isspace()]
            return self._create_ngrams(chars, self.n)
        else:
            # Word-level n-grams
            words = self._tokenize_words(text)
            return self._create_ngrams(words, self.n)

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute Jaccard similarity between two texts

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Jaccard similarity coefficient in range [0.0, 1.0]
            - 1.0 = identical n-gram sets
            - 0.0 = no common n-grams

        Example:
            >>> calc = JaccardSimilarity(n=1)
            >>> similarity = calc.compute_similarity(
            ...     "the cat sat on the mat",
            ...     "the dog sat on the log"
            ... )
            >>> print(f"{similarity:.3f}")
            0.429
        """
        start_time = time.perf_counter()

        # Convert texts to n-gram sets
        set_a = self._text_to_ngrams(text1)
        set_b = self._text_to_ngrams(text2)

        if self.debug:
            logger.debug(f"\nJaccard Similarity Calculation:")
            logger.debug(f"  Text 1: '{text1}'")
            logger.debug(f"  Text 2: '{text2}'")
            logger.debug(f"  N-gram size: {self.n}")
            logger.debug(f"  Level: {'character' if self.char_level else 'word'}")
            logger.debug(f"\n  Set A (|A|={len(set_a)}): {set_a}")
            logger.debug(f"  Set B (|B|={len(set_b)}): {set_b}")

        # Handle empty sets
        if len(set_a) == 0 and len(set_b) == 0:
            similarity = 1.0
            if self.debug:
                logger.debug(f"\n  Both sets empty → similarity = 1.0")
        elif len(set_a) == 0 or len(set_b) == 0:
            similarity = 0.0
            if self.debug:
                logger.debug(f"\n  One set empty → similarity = 0.0")
        else:
            # Compute intersection and union
            intersection = set_a & set_b
            union = set_a | set_b

            # Calculate Jaccard coefficient
            similarity = len(intersection) / len(union)

            if self.debug:
                logger.debug(f"\n  Intersection (A ∩ B): {intersection}")
                logger.debug(f"  |A ∩ B| = {len(intersection)}")
                logger.debug(f"\n  Union (A ∪ B): {union}")
                logger.debug(f"  |A ∪ B| = {len(union)}")
                logger.debug(f"\n  Formula: J(A,B) = |A ∩ B| / |A ∪ B|")
                logger.debug(f"         = {len(intersection)} / {len(union)}")
                logger.debug(f"         = {similarity:.6f}")

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Jaccard similarity: {similarity:.4f} (computed in {elapsed_time*1000:.2f}ms)")

        return similarity

    def compute_distance(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute Jaccard distance between two texts

        Distance = 1 - Similarity

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Jaccard distance in range [0.0, 1.0]
        """
        similarity = self.compute_similarity(text1, text2)
        distance = 1.0 - similarity

        if self.debug:
            logger.debug(f"\n  Jaccard Distance: d(A,B) = 1 - J(A,B)")
            logger.debug(f"                  = 1 - {similarity:.6f}")
            logger.debug(f"                  = {distance:.6f}")

        return distance

    def compute_batch_similarity(
        self,
        texts: List[str],
        reference_text: str
    ) -> List[float]:
        """
        Compute Jaccard similarity of multiple texts against a reference

        Args:
            texts: List of texts to compare
            reference_text: Reference text to compare against

        Returns:
            List of similarity scores

        Example:
            >>> calc = JaccardSimilarity(n=2)
            >>> texts = ["hello world", "hello python", "goodbye world"]
            >>> scores = calc.compute_batch_similarity(texts, "hello world")
            >>> print([f"{s:.3f}" for s in scores])
            ['1.000', '0.333', '0.333']
        """
        logger.info(f"Computing batch Jaccard similarity for {len(texts)} texts")
        start_time = time.perf_counter()

        # Convert reference to n-grams once
        ref_ngrams = self._text_to_ngrams(reference_text)

        similarities = []
        for text in texts:
            text_ngrams = self._text_to_ngrams(text)

            # Compute Jaccard
            if len(text_ngrams) == 0 and len(ref_ngrams) == 0:
                similarity = 1.0
            elif len(text_ngrams) == 0 or len(ref_ngrams) == 0:
                similarity = 0.0
            else:
                intersection = text_ngrams & ref_ngrams
                union = text_ngrams | ref_ngrams
                similarity = len(intersection) / len(union)

            similarities.append(similarity)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Batch similarity computed in {elapsed_time*1000:.2f}ms")

        return similarities

    def get_common_ngrams(
        self,
        text1: str,
        text2: str
    ) -> Set[Tuple]:
        """
        Get common n-grams between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Set of common n-grams (intersection)
        """
        set_a = self._text_to_ngrams(text1)
        set_b = self._text_to_ngrams(text2)

        return set_a & set_b

    def get_unique_ngrams(
        self,
        text1: str,
        text2: str
    ) -> Tuple[Set[Tuple], Set[Tuple]]:
        """
        Get unique n-grams for each text (elements not in intersection)

        Args:
            text1: First text
            text2: Second text

        Returns:
            Tuple of (unique_to_text1, unique_to_text2)
        """
        set_a = self._text_to_ngrams(text1)
        set_b = self._text_to_ngrams(text2)

        unique_a = set_a - set_b
        unique_b = set_b - set_a

        return unique_a, unique_b


# Example usage and demonstration
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/jaccard_similarity.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("=" * 70)
    print("JACCARD SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Word-level unigrams
    print("\n1. WORD-LEVEL UNIGRAMS (n=1)")
    print("-" * 70)

    calc_unigram = JaccardSimilarity(n=1, char_level=False, debug=True)

    text1 = "the cat sat on the mat"
    text2 = "the dog sat on the log"

    similarity = calc_unigram.compute_similarity(text1, text2)
    print(f"\nJaccard Similarity (unigrams): {similarity:.4f}")

    # Example 2: Word-level bigrams
    print("\n2. WORD-LEVEL BIGRAMS (n=2)")
    print("-" * 70)

    calc_bigram = JaccardSimilarity(n=2, char_level=False, debug=True)

    similarity = calc_bigram.compute_similarity(text1, text2)
    print(f"\nJaccard Similarity (bigrams): {similarity:.4f}")

    # Example 3: Character-level trigrams
    print("\n3. CHARACTER-LEVEL TRIGRAMS (n=3)")
    print("-" * 70)

    calc_char = JaccardSimilarity(n=3, char_level=True, debug=True)

    text1 = "machine learning"
    text2 = "machine learnig"  # typo

    similarity = calc_char.compute_similarity(text1, text2)
    print(f"\nJaccard Similarity (char trigrams): {similarity:.4f}")

    # Example 4: Identical texts
    print("\n4. IDENTICAL TEXTS")
    print("-" * 70)

    calc = JaccardSimilarity(n=1, debug=False)
    text = "artificial intelligence"

    similarity = calc.compute_similarity(text, text)
    print(f"Text: '{text}'")
    print(f"Self-similarity: {similarity:.4f}")

    # Example 5: Completely different texts
    print("\n5. COMPLETELY DIFFERENT TEXTS")
    print("-" * 70)

    text1 = "apple banana cherry"
    text2 = "dog elephant fox"

    similarity = calc.compute_similarity(text1, text2)
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity: {similarity:.4f}")

    # Example 6: Common and unique n-grams
    print("\n6. ANALYZING N-GRAMS")
    print("-" * 70)

    calc_analysis = JaccardSimilarity(n=1, char_level=False)

    text1 = "machine learning algorithms"
    text2 = "machine learning techniques"

    common = calc_analysis.get_common_ngrams(text1, text2)
    unique_a, unique_b = calc_analysis.get_unique_ngrams(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"\nCommon n-grams: {common}")
    print(f"Unique to text 1: {unique_a}")
    print(f"Unique to text 2: {unique_b}")

    # Example 7: Batch processing
    print("\n7. BATCH PROCESSING")
    print("-" * 70)

    calc_batch = JaccardSimilarity(n=2, char_level=False, debug=False)

    texts = [
        "generative artificial intelligence",
        "generative AI models",
        "deep learning networks",
        "artificial intelligence systems"
    ]
    reference = "generative artificial intelligence"

    similarities = calc_batch.compute_batch_similarity(texts, reference)

    print(f"\nReference: '{reference}'")
    for text, sim in zip(texts, similarities):
        print(f"  '{text:35s}' → {sim:.4f}")

    # Example 8: Effect of n-gram size
    print("\n8. EFFECT OF N-GRAM SIZE")
    print("-" * 70)

    text1 = "the quick brown fox jumps over the lazy dog"
    text2 = "the quick brown fox jumps over a lazy cat"

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print("\nSimilarity by n-gram size:")

    for n in [1, 2, 3]:
        calc_n = JaccardSimilarity(n=n, char_level=False, debug=False)
        sim = calc_n.compute_similarity(text1, text2)
        print(f"  n={n}: {sim:.4f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
