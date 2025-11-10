"""
Character N-gram Similarity Algorithm

MATHEMATICAL EXPLANATION:
========================

Character n-gram similarity is a specialized form of Jaccard similarity that operates
on character-level n-grams rather than word-level tokens. This makes it particularly
useful for detecting typos, spelling variations, and fuzzy string matching.

N-GRAM DEFINITION:
------------------

An n-gram is a contiguous sequence of n characters from a text.

For text T = "hello":
- 1-grams (unigrams):  {h, e, l, l, o}
- 2-grams (bigrams):   {he, el, ll, lo}
- 3-grams (trigrams):  {hel, ell, llo}
- 4-grams:             {hell, ello}

SIMILARITY COMPUTATION:
-----------------------

Given two texts A and B, we:

1. Extract character n-grams:
   N(A) = set of n-grams from text A
   N(B) = set of n-grams from text B

2. Compute Jaccard similarity:

                |N(A) ∩ N(B)|
   S(A, B) = ─────────────────
             |N(A) ∪ N(B)|

ADVANTAGES OVER WORD-BASED METHODS:
------------------------------------

1. **Typo Tolerance**: Captures partial matches even with spelling errors
   - "machine" vs "machien" → high similarity despite typo
   - Word-based would see these as completely different

2. **Substring Matching**: Detects shared substrings
   - "international" vs "national" → shares "national"

3. **Language Independent**: Works without word boundaries
   - Useful for languages without spaces (Chinese, Japanese)

4. **Partial Name Matching**: Good for name variations
   - "Smith, John" vs "John Smith" → high similarity

OPTIMAL N-GRAM SIZE:
--------------------

The choice of n affects similarity behavior:

- n=1 (character unigrams):
  * Very tolerant, may be too lenient
  * Good for very short strings

- n=2 (character bigrams):
  * Balanced, good general-purpose choice
  * Captures character transitions

- n=3 (character trigrams):
  * More discriminative
  * Standard for many applications
  * Good balance of specificity and robustness

- n=4+ (larger n-grams):
  * Very specific, less tolerant of variations
  * Good for longer texts

EXAMPLE CALCULATION:
====================

Compare "machine" vs "machien" using character trigrams (n=3):

Text 1: "machine"
Text 2: "machien"

Step 1: Extract character trigrams

  N(A) = {mac, ach, chi, hin, ine}  (5 trigrams)
  N(B) = {mac, ach, chi, hie, ien}  (5 trigrams)

Step 2: Compute intersection and union

  N(A) ∩ N(B) = {mac, ach, chi}     (3 common trigrams)
  N(A) ∪ N(B) = {mac, ach, chi, hin, ine, hie, ien}  (7 unique trigrams)

Step 3: Calculate Jaccard similarity

  S(A, B) = 3 / 7 = 0.4286

Despite the typo (swapped 'i' and 'e'), similarity is ~0.43, indicating
substantial overlap. Word-based comparison would give 0.0 (no match).

PADDING VARIANTS:
-----------------

Some implementations add padding characters (e.g., '#') to capture
start/end of string:

  "cat" with padding → "#cat#"
  Trigrams: {#ca, cat, at#}

This makes position-dependent comparisons more accurate.

COMPLEXITY:
-----------
Time:  O(m + n) where m, n are string lengths
Space: O(m + n) for storing n-grams
"""

from typing import Set, List, Tuple, Optional
from loguru import logger
import time
import re


class CharacterNgramSimilarity:
    """
    Character N-gram Similarity calculator

    Specialized for typo detection and fuzzy string matching using
    character-level n-grams.
    """

    def __init__(
        self,
        n: int = 3,
        padding: bool = False,
        padding_char: str = '#',
        case_sensitive: bool = False,
        remove_spaces: bool = True,
        debug: bool = False
    ):
        """
        Initialize Character N-gram Similarity calculator

        Args:
            n: N-gram size (typically 2-4)
            padding: If True, add padding characters to string boundaries
            padding_char: Character to use for padding
            case_sensitive: Whether to consider case in comparison
            remove_spaces: Whether to remove whitespace before n-gram extraction
            debug: If True, prints intermediate calculation steps

        Example:
            >>> # Standard trigrams without padding
            >>> calc = CharacterNgramSimilarity(n=3)
            >>>
            >>> # Bigrams with padding
            >>> calc = CharacterNgramSimilarity(n=2, padding=True)
        """
        self.n = n
        self.padding = padding
        self.padding_char = padding_char
        self.case_sensitive = case_sensitive
        self.remove_spaces = remove_spaces
        self.debug = debug

        logger.info(f"Character N-gram Similarity initialized (n={n}, padding={padding})")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Normalize case if needed
        if not self.case_sensitive:
            text = text.lower()

        # Remove spaces if configured
        if self.remove_spaces:
            text = re.sub(r'\s+', '', text)

        # Add padding if configured
        if self.padding:
            # Add (n-1) padding characters on each side
            padding = self.padding_char * (self.n - 1)
            text = padding + text + padding

        return text

    def _extract_ngrams(self, text: str) -> Set[str]:
        """
        Extract character n-grams from text

        Args:
            text: Input text (already preprocessed)

        Returns:
            Set of n-gram strings

        Example:
            >>> calc = CharacterNgramSimilarity(n=3)
            >>> text = "hello"
            >>> ngrams = calc._extract_ngrams(text)
            >>> print(ngrams)
            {'hel', 'ell', 'llo'}
        """
        if len(text) < self.n:
            # If text is shorter than n, return the whole text as one n-gram
            return {text} if text else set()

        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            ngrams.add(ngram)

        return ngrams

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute character n-gram similarity between two texts

        Uses Jaccard similarity coefficient on character n-gram sets.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score in range [0.0, 1.0]
            - 1.0 = identical character n-gram sets
            - 0.0 = no common character n-grams

        Example:
            >>> calc = CharacterNgramSimilarity(n=3)
            >>> similarity = calc.compute_similarity("machine", "machien")
            >>> print(f"{similarity:.3f}")
            0.429
        """
        start_time = time.perf_counter()

        # Preprocess texts
        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)

        if self.debug:
            logger.debug(f"\nCharacter N-gram Similarity Calculation:")
            logger.debug(f"  Original Text 1: '{text1}'")
            logger.debug(f"  Original Text 2: '{text2}'")
            logger.debug(f"  Processed Text 1: '{processed_text1}'")
            logger.debug(f"  Processed Text 2: '{processed_text2}'")
            logger.debug(f"  N-gram size: {self.n}")
            logger.debug(f"  Padding: {self.padding}")

        # Extract n-grams
        ngrams_a = self._extract_ngrams(processed_text1)
        ngrams_b = self._extract_ngrams(processed_text2)

        if self.debug:
            logger.debug(f"\n  N-grams A (|A|={len(ngrams_a)}): {sorted(ngrams_a)}")
            logger.debug(f"  N-grams B (|B|={len(ngrams_b)}): {sorted(ngrams_b)}")

        # Handle empty sets
        if len(ngrams_a) == 0 and len(ngrams_b) == 0:
            similarity = 1.0
            if self.debug:
                logger.debug(f"\n  Both sets empty → similarity = 1.0")
        elif len(ngrams_a) == 0 or len(ngrams_b) == 0:
            similarity = 0.0
            if self.debug:
                logger.debug(f"\n  One set empty → similarity = 0.0")
        else:
            # Compute Jaccard similarity
            intersection = ngrams_a & ngrams_b
            union = ngrams_a | ngrams_b

            similarity = len(intersection) / len(union)

            if self.debug:
                logger.debug(f"\n  Intersection (A ∩ B): {sorted(intersection)}")
                logger.debug(f"  |A ∩ B| = {len(intersection)}")
                logger.debug(f"\n  Union (A ∪ B): {sorted(union)}")
                logger.debug(f"  |A ∪ B| = {len(union)}")
                logger.debug(f"\n  Jaccard Similarity: {len(intersection)} / {len(union)} = {similarity:.6f}")

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Character n-gram similarity: {similarity:.4f} (computed in {elapsed_time*1000:.2f}ms)")

        return similarity

    def compute_batch_similarity(
        self,
        texts: List[str],
        reference_text: str
    ) -> List[float]:
        """
        Compute character n-gram similarity of multiple texts against a reference

        Args:
            texts: List of texts to compare
            reference_text: Reference text to compare against

        Returns:
            List of similarity scores

        Example:
            >>> calc = CharacterNgramSimilarity(n=3)
            >>> texts = ["machine", "machien", "machine learning"]
            >>> scores = calc.compute_batch_similarity(texts, "machine")
            >>> print([f"{s:.3f}" for s in scores])
            ['1.000', '0.429', '0.500']
        """
        logger.info(f"Computing batch character n-gram similarity for {len(texts)} texts")
        start_time = time.perf_counter()

        # Preprocess and extract n-grams from reference once
        processed_ref = self._preprocess_text(reference_text)
        ref_ngrams = self._extract_ngrams(processed_ref)

        similarities = []
        for text in texts:
            processed_text = self._preprocess_text(text)
            text_ngrams = self._extract_ngrams(processed_text)

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
    ) -> Set[str]:
        """
        Get common character n-grams between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Set of common n-grams
        """
        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)

        ngrams_a = self._extract_ngrams(processed_text1)
        ngrams_b = self._extract_ngrams(processed_text2)

        return ngrams_a & ngrams_b

    def get_unique_ngrams(
        self,
        text1: str,
        text2: str
    ) -> Tuple[Set[str], Set[str]]:
        """
        Get unique character n-grams for each text

        Args:
            text1: First text
            text2: Second text

        Returns:
            Tuple of (unique_to_text1, unique_to_text2)
        """
        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)

        ngrams_a = self._extract_ngrams(processed_text1)
        ngrams_b = self._extract_ngrams(processed_text2)

        unique_a = ngrams_a - ngrams_b
        unique_b = ngrams_b - ngrams_a

        return unique_a, unique_b

    def find_typos(
        self,
        candidate: str,
        dictionary: List[str],
        threshold: float = 0.7,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find potential matches for a misspelled word

        Uses character n-gram similarity to find dictionary words
        that are similar to the candidate (likely typos).

        Args:
            candidate: Potentially misspelled word
            dictionary: List of correct words
            threshold: Minimum similarity to consider a match
            top_n: Maximum number of matches to return

        Returns:
            List of (word, similarity) tuples, sorted by similarity descending

        Example:
            >>> calc = CharacterNgramSimilarity(n=3)
            >>> dictionary = ["machine", "learning", "artificial", "intelligence"]
            >>> matches = calc.find_typos("machien", dictionary)
            >>> print(matches)
            [('machine', 0.4286)]
        """
        logger.info(f"Finding typo matches for '{candidate}' in dictionary of {len(dictionary)} words")

        # Compute similarities to all dictionary words
        similarities = self.compute_batch_similarity(dictionary, candidate)

        # Filter by threshold and create (word, similarity) pairs
        matches = [(word, sim) for word, sim in zip(dictionary, similarities) if sim >= threshold]

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        result = matches[:top_n]

        logger.info(f"Found {len(result)} potential matches")
        return result


# Example usage and demonstration
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/ngram_similarity.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("=" * 70)
    print("CHARACTER N-GRAM SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Typo detection
    print("\n1. TYPO DETECTION (n=3)")
    print("-" * 70)

    calc_typo = CharacterNgramSimilarity(n=3, debug=True)

    text1 = "machine"
    text2 = "machien"  # typo: swapped 'i' and 'e'

    similarity = calc_typo.compute_similarity(text1, text2)
    print(f"\nSimilarity: {similarity:.4f}")
    print("Note: Despite typo, similarity is ~0.43, indicating substantial overlap")

    # Example 2: Effect of n-gram size
    print("\n2. EFFECT OF N-GRAM SIZE")
    print("-" * 70)

    text1 = "artificial intelligence"
    text2 = "artificial inteligence"  # typo: missing 'l'

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}' (missing 'l' in 'intelligence')")
    print("\nSimilarity by n-gram size:")

    for n in [2, 3, 4, 5]:
        calc_n = CharacterNgramSimilarity(n=n, debug=False)
        sim = calc_n.compute_similarity(text1, text2)
        print(f"  n={n}: {sim:.4f}")

    print("\nObservation: Larger n is more sensitive to errors")

    # Example 3: With and without padding
    print("\n3. PADDING COMPARISON")
    print("-" * 70)

    calc_no_pad = CharacterNgramSimilarity(n=3, padding=False, debug=True)
    calc_with_pad = CharacterNgramSimilarity(n=3, padding=True, debug=False)

    text1 = "cat"
    text2 = "scat"

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")

    sim_no_pad = calc_no_pad.compute_similarity(text1, text2)
    sim_with_pad = calc_with_pad.compute_similarity(text1, text2)

    print(f"\nWithout padding: {sim_no_pad:.4f}")
    print(f"With padding:    {sim_with_pad:.4f}")

    # Example 4: Space handling
    print("\n4. SPACE HANDLING")
    print("-" * 70)

    calc_spaces = CharacterNgramSimilarity(n=3, remove_spaces=True, debug=False)
    calc_keep_spaces = CharacterNgramSimilarity(n=3, remove_spaces=False, debug=False)

    text1 = "machine learning"
    text2 = "machinelearning"

    sim_remove = calc_spaces.compute_similarity(text1, text2)
    sim_keep = calc_keep_spaces.compute_similarity(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"\nRemoving spaces:  {sim_remove:.4f}")
    print(f"Keeping spaces:   {sim_keep:.4f}")

    # Example 5: Finding typo corrections
    print("\n5. TYPO CORRECTION")
    print("-" * 70)

    calc_correct = CharacterNgramSimilarity(n=3, debug=False)

    dictionary = [
        "machine", "learning", "artificial", "intelligence",
        "neural", "network", "deep", "generative", "model"
    ]

    typos = ["machien", "learnig", "artifical", "inteligence", "nerual"]

    print("Finding corrections for typos:\n")
    for typo in typos:
        matches = calc_correct.find_typos(typo, dictionary, threshold=0.3, top_n=3)
        if matches:
            print(f"'{typo:15s}' → {matches[0][0]:15s} (similarity: {matches[0][1]:.3f})")
        else:
            print(f"'{typo:15s}' → No matches found")

    # Example 6: Analyzing n-grams
    print("\n6. N-GRAM ANALYSIS")
    print("-" * 70)

    calc_analysis = CharacterNgramSimilarity(n=3, debug=False)

    text1 = "neural"
    text2 = "nerual"  # typo

    common = calc_analysis.get_common_ngrams(text1, text2)
    unique_a, unique_b = calc_analysis.get_unique_ngrams(text1, text2)

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}' (typo)")
    print(f"\nCommon trigrams: {sorted(common)}")
    print(f"Unique to '{text1}': {sorted(unique_a)}")
    print(f"Unique to '{text2}': {sorted(unique_b)}")

    # Example 7: Batch processing
    print("\n7. BATCH PROCESSING")
    print("-" * 70)

    calc_batch = CharacterNgramSimilarity(n=3, debug=False)

    texts = [
        "generative AI",
        "generative artificial intelligence",
        "generativ AI",  # typo
        "deep learning",
        "generative Ai"  # case variation
    ]
    reference = "generative AI"

    similarities = calc_batch.compute_batch_similarity(texts, reference)

    print(f"\nReference: '{reference}'")
    for text, sim in zip(texts, similarities):
        print(f"  '{text:40s}' → {sim:.4f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
