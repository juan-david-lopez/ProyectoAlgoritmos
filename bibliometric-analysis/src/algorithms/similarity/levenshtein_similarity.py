"""
Levenshtein Distance/Similarity Algorithm

MATHEMATICAL EXPLANATION:
========================

The Levenshtein distance (also called edit distance) is the minimum number of single-character
edits (insertions, deletions, or substitutions) required to change one string into another.

FORMAL DEFINITION:
------------------
For two strings A and B of length m and n respectively, the Levenshtein distance D(i,j)
is computed using dynamic programming:

                    ⎧ max(i, j)                                    if min(i, j) = 0
        D(i,j) =    ⎨
                    ⎩ min { D(i-1, j) + 1,                         otherwise
                          { D(i, j-1) + 1,
                          { D(i-1, j-1) + cost

where:
    - D(i, j) = distance between first i characters of A and first j characters of B
    - cost = 0 if A[i] == B[j], otherwise cost = 1
    - D(i-1, j) + 1 = deletion from A
    - D(i, j-1) + 1 = insertion into A
    - D(i-1, j-1) + cost = substitution (or match if cost=0)

BASE CASES:
-----------
    D(0, 0) = 0     (empty strings have distance 0)
    D(i, 0) = i     (delete all i characters from A)
    D(0, j) = j     (insert all j characters into A)

SIMILARITY CONVERSION:
----------------------
The normalized similarity score is:

    similarity = 1 - (distance / max(len(A), len(B)))

This produces a value in [0, 1] where:
    - 1.0 = identical strings
    - 0.0 = completely different strings

EXAMPLE CALCULATION:
====================
Compare "kitten" vs "sitting":

    k i t t e n
  0 1 2 3 4 5 6
s 1 1 2 3 4 5 6
i 2 2 1 2 3 4 5
t 3 3 2 1 2 3 4
t 4 4 3 2 1 2 3
i 5 5 4 3 2 2 3
n 6 6 5 4 3 3 2
g 7 7 6 5 4 4 3

Final distance: D(7,6) = 3
Operations: substitute k→s, substitute e→i, insert g
Similarity: 1 - (3 / max(6,7)) = 1 - (3/7) = 0.571

COMPLEXITY:
-----------
Time:  O(m × n) where m, n are string lengths
Space: O(m × n) for full matrix, O(min(m,n)) with optimization
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger
import time


class LevenshteinSimilarity:
    """
    Levenshtein distance and similarity calculator

    Implements the classic dynamic programming algorithm for computing
    edit distance between two strings.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize Levenshtein calculator

        Args:
            debug: If True, prints intermediate calculation steps
        """
        self.debug = debug
        logger.info("Levenshtein Similarity calculator initialized")

    def compute_distance(
        self,
        text1: str,
        text2: str,
        case_sensitive: bool = False
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Compute Levenshtein distance between two texts

        Args:
            text1: First text string
            text2: Second text string
            case_sensitive: Whether to consider case in comparison

        Returns:
            Tuple of (distance, matrix)
            - distance: Edit distance (integer)
            - matrix: Dynamic programming matrix (if debug=True)

        Example:
            >>> calc = LevenshteinSimilarity()
            >>> distance, _ = calc.compute_distance("kitten", "sitting")
            >>> print(distance)
            3
        """
        start_time = time.perf_counter()

        # Normalize text if case-insensitive
        if not case_sensitive:
            text1 = text1.lower()
            text2 = text2.lower()

        m, n = len(text1), len(text2)

        if self.debug:
            logger.debug(f"Computing Levenshtein distance:")
            logger.debug(f"  Text 1: '{text1}' (length={m})")
            logger.debug(f"  Text 2: '{text2}' (length={n})")

        # Initialize DP matrix
        # D[i][j] = distance between text1[0..i-1] and text2[0..j-1]
        D = np.zeros((m + 1, n + 1), dtype=int)

        # BASE CASES:
        # D[i][0] = i (delete all i characters from text1)
        for i in range(m + 1):
            D[i][0] = i

        # D[0][j] = j (insert all j characters into text1)
        for j in range(n + 1):
            D[0][j] = j

        if self.debug:
            logger.debug("\nInitialized DP matrix (base cases):")
            self._print_matrix(D, text1, text2)

        # DYNAMIC PROGRAMMING RECURRENCE:
        # Fill the matrix row by row
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate cost: 0 if characters match, 1 if different
                cost = 0 if text1[i-1] == text2[j-1] else 1

                # Compute minimum of three operations:
                deletion = D[i-1][j] + 1      # Delete from text1
                insertion = D[i][j-1] + 1     # Insert into text1
                substitution = D[i-1][j-1] + cost  # Substitute or match

                D[i][j] = min(deletion, insertion, substitution)

                if self.debug and i <= 3 and j <= 3:
                    logger.debug(f"\nStep (i={i}, j={j}):")
                    logger.debug(f"  Comparing: '{text1[i-1]}' vs '{text2[j-1]}'")
                    logger.debug(f"  Cost: {cost}")
                    logger.debug(f"  Deletion:     D[{i-1}][{j}] + 1 = {deletion}")
                    logger.debug(f"  Insertion:    D[{i}][{j-1}] + 1 = {insertion}")
                    logger.debug(f"  Substitution: D[{i-1}][{j-1}] + {cost} = {substitution}")
                    logger.debug(f"  Minimum: {D[i][j]}")

        distance = int(D[m][n])

        if self.debug:
            logger.debug("\nFinal DP matrix:")
            self._print_matrix(D, text1, text2)
            logger.debug(f"\nFinal Levenshtein distance: {distance}")

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Levenshtein distance computed in {elapsed_time*1000:.2f}ms: {distance}")

        return distance, D if self.debug else None

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        case_sensitive: bool = False
    ) -> float:
        """
        Compute normalized Levenshtein similarity between two texts

        Similarity = 1 - (distance / max(len(text1), len(text2)))

        Args:
            text1: First text string
            text2: Second text string
            case_sensitive: Whether to consider case in comparison

        Returns:
            Similarity score in range [0.0, 1.0]
            - 1.0 = identical strings
            - 0.0 = completely different

        Example:
            >>> calc = LevenshteinSimilarity()
            >>> similarity = calc.compute_similarity("kitten", "sitting")
            >>> print(f"{similarity:.3f}")
            0.571
        """
        start_time = time.perf_counter()

        # Handle edge cases
        if not text1 and not text2:
            return 1.0  # Both empty
        if not text1 or not text2:
            return 0.0  # One empty

        # Compute distance
        distance, _ = self.compute_distance(text1, text2, case_sensitive)

        # Normalize to [0, 1]
        max_len = max(len(text1), len(text2))
        similarity = 1.0 - (distance / max_len)

        elapsed_time = time.perf_counter() - start_time

        if self.debug:
            logger.debug(f"\nSimilarity Calculation:")
            logger.debug(f"  Distance: {distance}")
            logger.debug(f"  Max length: {max_len}")
            logger.debug(f"  Formula: 1 - ({distance} / {max_len}) = {similarity:.4f}")

        logger.info(f"Levenshtein similarity: {similarity:.4f} (computed in {elapsed_time*1000:.2f}ms)")

        return similarity

    def compute_batch_similarity(
        self,
        texts: list,
        reference_text: str,
        case_sensitive: bool = False
    ) -> list:
        """
        Compute similarity of multiple texts against a reference

        Args:
            texts: List of texts to compare
            reference_text: Reference text to compare against
            case_sensitive: Whether to consider case in comparison

        Returns:
            List of similarity scores

        Example:
            >>> calc = LevenshteinSimilarity()
            >>> texts = ["kitten", "sitting", "kitchen"]
            >>> scores = calc.compute_batch_similarity(texts, "kitten")
            >>> print([f"{s:.3f}" for s in scores])
            ['1.000', '0.571', '0.714']
        """
        logger.info(f"Computing batch similarity for {len(texts)} texts")
        start_time = time.perf_counter()

        similarities = []
        for text in texts:
            similarity = self.compute_similarity(text, reference_text, case_sensitive)
            similarities.append(similarity)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Batch similarity computed in {elapsed_time*1000:.2f}ms")

        return similarities

    def _print_matrix(self, matrix: np.ndarray, text1: str, text2: str):
        """
        Pretty-print the DP matrix for debugging

        Args:
            matrix: DP matrix to print
            text1: First text (row labels)
            text2: Second text (column labels)
        """
        m, n = matrix.shape

        # Header row
        header = "      " + "  ".join([f"{c:>2}" for c in text2])
        logger.debug(header)

        # Separator
        logger.debug("  " + "-" * (len(header) - 2))

        # Matrix rows
        for i in range(m):
            if i == 0:
                row_label = "  "
            else:
                row_label = f"{text1[i-1]:>2}"

            row_values = "  ".join([f"{matrix[i][j]:>2}" for j in range(n)])
            logger.debug(f"{row_label} | {row_values}")


# Example usage and demonstration
if __name__ == "__main__":
    # Setup logger
    logger.add(
        "logs/levenshtein_similarity.log",
        rotation="10 MB",
        level="DEBUG"
    )

    print("=" * 70)
    print("LEVENSHTEIN SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Basic usage
    print("\n1. BASIC EXAMPLE: 'kitten' vs 'sitting'")
    print("-" * 70)

    calc = LevenshteinSimilarity(debug=True)
    distance, matrix = calc.compute_distance("kitten", "sitting")
    similarity = calc.compute_similarity("kitten", "sitting")

    print(f"\nDistance: {distance}")
    print(f"Similarity: {similarity:.4f}")

    # Example 2: Identical strings
    print("\n2. IDENTICAL STRINGS: 'hello' vs 'hello'")
    print("-" * 70)

    similarity = calc.compute_similarity("hello", "hello")
    print(f"Similarity: {similarity:.4f}")

    # Example 3: Completely different
    print("\n3. COMPLETELY DIFFERENT: 'abc' vs 'xyz'")
    print("-" * 70)

    similarity = calc.compute_similarity("abc", "xyz")
    print(f"Similarity: {similarity:.4f}")

    # Example 4: Case sensitivity
    print("\n4. CASE SENSITIVITY: 'Hello' vs 'hello'")
    print("-" * 70)

    sim_sensitive = calc.compute_similarity("Hello", "hello", case_sensitive=True)
    sim_insensitive = calc.compute_similarity("Hello", "hello", case_sensitive=False)

    print(f"Case-sensitive:   {sim_sensitive:.4f}")
    print(f"Case-insensitive: {sim_insensitive:.4f}")

    # Example 5: Batch processing
    print("\n5. BATCH PROCESSING")
    print("-" * 70)

    calc_batch = LevenshteinSimilarity(debug=False)
    texts = [
        "machine learning",
        "machine learnig",  # typo
        "deep learning",
        "artificial intelligence"
    ]
    reference = "machine learning"

    similarities = calc_batch.compute_batch_similarity(texts, reference)

    print(f"\nReference: '{reference}'")
    for text, sim in zip(texts, similarities):
        print(f"  '{text:30s}' → {sim:.4f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
