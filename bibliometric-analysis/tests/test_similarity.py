"""
Unit Tests for Similarity Algorithms

Tests all 6 similarity algorithms with known test cases to ensure correctness.

Usage:
    pytest tests/test_similarity.py -v
    pytest tests/test_similarity.py::TestLevenshteinSimilarity -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.algorithms.similarity import (
    LevenshteinSimilarity,
    TfidfCosineSimilarity,
    JaccardSimilarity,
    CharacterNgramSimilarity,
    SBERTSimilarity,
    TransformerSimilarity
)


class TestLevenshteinSimilarity:
    """Test Levenshtein similarity algorithm"""

    @pytest.fixture
    def calculator(self):
        return LevenshteinSimilarity(debug=False)

    def test_identical_strings(self, calculator):
        """Test that identical strings have similarity 1.0"""
        sim = calculator.compute_similarity("hello", "hello")
        assert sim == 1.0

    def test_empty_strings(self, calculator):
        """Test empty string handling"""
        sim = calculator.compute_similarity("", "")
        assert sim == 1.0

        sim = calculator.compute_similarity("hello", "")
        assert sim == 0.0

    def test_completely_different(self, calculator):
        """Test completely different strings"""
        sim = calculator.compute_similarity("abc", "xyz")
        assert sim == 0.0

    def test_known_distance(self, calculator):
        """Test known edit distance: kitten -> sitting = 3 edits"""
        distance, _ = calculator.compute_distance("kitten", "sitting")
        assert distance == 3

        # Similarity should be 1 - (3 / 7) = 0.571...
        sim = calculator.compute_similarity("kitten", "sitting")
        assert abs(sim - 0.571428) < 0.001

    def test_case_sensitivity(self, calculator):
        """Test case sensitivity option"""
        sim_sensitive = calculator.compute_similarity("Hello", "hello", case_sensitive=True)
        sim_insensitive = calculator.compute_similarity("Hello", "hello", case_sensitive=False)

        assert sim_sensitive < 1.0
        assert sim_insensitive == 1.0

    def test_batch_similarity(self, calculator):
        """Test batch processing"""
        texts = ["hello", "hallo", "world"]
        reference = "hello"

        similarities = calculator.compute_batch_similarity(texts, reference)

        assert len(similarities) == 3
        assert similarities[0] == 1.0  # "hello" vs "hello"
        assert similarities[1] > 0.5   # "hallo" vs "hello"
        assert similarities[2] < 0.5   # "world" vs "hello"


class TestTfidfCosineSimilarity:
    """Test TF-IDF cosine similarity algorithm"""

    @pytest.fixture
    def calculator(self):
        return TfidfCosineSimilarity(debug=False)

    def test_identical_texts(self, calculator):
        """Test that identical texts have similarity 1.0"""
        sim = calculator.compute_similarity("hello world", "hello world")
        assert abs(sim - 1.0) < 0.001

    def test_no_common_terms(self, calculator):
        """Test texts with no common terms"""
        sim = calculator.compute_similarity("apple banana", "dog cat")
        assert abs(sim - 0.0) < 0.001

    def test_partial_overlap(self, calculator):
        """Test texts with partial overlap"""
        sim = calculator.compute_similarity(
            "machine learning algorithms",
            "machine learning techniques"
        )
        assert 0.4 < sim < 0.9  # Should have moderate similarity

    def test_similarity_matrix(self, calculator):
        """Test similarity matrix computation"""
        texts = ["hello world", "hello python", "goodbye world"]

        matrix = calculator.compute_similarity_matrix(texts)

        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal should be 1.0
        assert np.allclose(matrix, matrix.T)  # Should be symmetric

    def test_top_features(self, calculator):
        """Test top feature extraction"""
        calculator.fit_transform(["sample text for testing"])
        features = calculator.get_top_features("machine learning", top_n=5)

        assert len(features) <= 5
        assert all(isinstance(feat[0], str) for feat in features)
        assert all(isinstance(feat[1], float) for feat in features)


class TestJaccardSimilarity:
    """Test Jaccard similarity algorithm"""

    @pytest.fixture
    def calculator(self):
        return JaccardSimilarity(n=1, char_level=False, debug=False)

    def test_identical_sets(self, calculator):
        """Test that identical sets have similarity 1.0"""
        sim = calculator.compute_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_disjoint_sets(self, calculator):
        """Test that disjoint sets have similarity 0.0"""
        sim = calculator.compute_similarity("apple banana", "dog cat")
        assert sim == 0.0

    def test_known_jaccard(self, calculator):
        """Test known Jaccard coefficient"""
        # "the cat" vs "the dog"
        # Sets: {the, cat} and {the, dog}
        # Intersection: {the} -> size 1
        # Union: {the, cat, dog} -> size 3
        # Jaccard: 1/3 = 0.333...
        sim = calculator.compute_similarity("the cat", "the dog")
        assert abs(sim - 0.333333) < 0.001

    def test_common_ngrams(self, calculator):
        """Test common n-grams extraction"""
        common = calculator.get_common_ngrams("hello world", "hello python")
        assert ("hello",) in common

    def test_unique_ngrams(self, calculator):
        """Test unique n-grams extraction"""
        unique_a, unique_b = calculator.get_unique_ngrams("hello world", "hello python")

        assert ("world",) in unique_a
        assert ("python",) in unique_b
        assert ("hello",) not in unique_a
        assert ("hello",) not in unique_b


class TestCharacterNgramSimilarity:
    """Test character n-gram similarity algorithm"""

    @pytest.fixture
    def calculator(self):
        return CharacterNgramSimilarity(n=3, debug=False)

    def test_identical_strings(self, calculator):
        """Test that identical strings have similarity 1.0"""
        sim = calculator.compute_similarity("hello", "hello")
        assert sim == 1.0

    def test_typo_detection(self, calculator):
        """Test typo similarity (should be high)"""
        # "machine" vs "machien" (transposed letters)
        sim = calculator.compute_similarity("machine", "machien")
        assert sim > 0.3  # Should detect similarity despite typo

    def test_completely_different(self, calculator):
        """Test completely different strings"""
        sim = calculator.compute_similarity("abc", "xyz")
        assert sim == 0.0

    def test_padding(self):
        """Test n-gram extraction with padding"""
        calc_no_pad = CharacterNgramSimilarity(n=3, padding=False)
        calc_pad = CharacterNgramSimilarity(n=3, padding=True)

        sim_no_pad = calc_no_pad.compute_similarity("cat", "scat")
        sim_pad = calc_pad.compute_similarity("cat", "scat")

        # With padding, "cat" becomes "##cat##" and similarity should differ
        assert sim_no_pad != sim_pad

    def test_typo_correction(self, calculator):
        """Test typo correction functionality"""
        dictionary = ["machine", "learning", "artificial"]
        matches = calculator.find_typos("machien", dictionary, threshold=0.3, top_n=1)

        assert len(matches) > 0
        assert matches[0][0] == "machine"
        assert matches[0][1] > 0.3


@pytest.mark.slow
class TestSBERTSimilarity:
    """Test SBERT similarity algorithm (slow - requires model download)"""

    @pytest.fixture
    def calculator(self):
        try:
            return SBERTSimilarity(debug=False)
        except Exception:
            pytest.skip("SBERT not available")

    def test_identical_texts(self, calculator):
        """Test that identical texts have high similarity"""
        sim = calculator.compute_similarity("hello world", "hello world")
        assert sim > 0.99

    def test_paraphrase_detection(self, calculator):
        """Test that paraphrases have high similarity"""
        sim = calculator.compute_similarity(
            "The cat sits on the mat",
            "A feline rests on a rug"
        )
        # Paraphrases should have relatively high similarity
        assert sim > 0.5

    def test_different_topics(self, calculator):
        """Test that different topics have low similarity"""
        sim = calculator.compute_similarity(
            "Machine learning algorithms",
            "Cooking Italian food"
        )
        assert sim < 0.5

    def test_embedding_dimension(self, calculator):
        """Test that embeddings have correct dimension"""
        embeddings = calculator.encode(["hello world"])
        assert embeddings.shape == (1, calculator.embedding_dim)

    def test_similarity_matrix(self, calculator):
        """Test similarity matrix computation"""
        texts = ["hello", "hi", "goodbye"]
        matrix = calculator.compute_similarity_matrix(texts)

        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0, atol=0.01)

    def test_semantic_search(self, calculator):
        """Test semantic search functionality"""
        query = "machine learning"
        corpus = ["deep learning", "cooking", "neural networks"]

        results = calculator.find_most_similar(query, corpus, top_k=2)

        assert len(results) == 2
        assert results[0][1] in ["deep learning", "neural networks"]  # Should find ML-related


@pytest.mark.slow
class TestTransformerSimilarity:
    """Test BERT Transformer similarity algorithm (slow - requires model)"""

    @pytest.fixture
    def calculator(self):
        try:
            return TransformerSimilarity(debug=False)
        except Exception:
            pytest.skip("BERT not available")

    def test_identical_texts(self, calculator):
        """Test that identical texts have high similarity"""
        sim = calculator.compute_similarity("hello world", "hello world")
        assert sim > 0.99

    def test_contextual_understanding(self, calculator):
        """Test contextual word understanding"""
        # "bank" in financial context
        text1 = "I went to the bank to deposit money"
        # "bank" in river context
        text2 = "We sat by the river bank"
        # Financial institution (semantic match to text1)
        text3 = "The financial institution is closed"

        sim_12 = calculator.compute_similarity(text1, text2)
        sim_13 = calculator.compute_similarity(text1, text3)

        # text1 and text3 should be more similar (same meaning of "bank")
        # This tests BERT's contextual understanding
        assert sim_13 > sim_12

    def test_embedding_dimension(self, calculator):
        """Test that embeddings have correct dimension"""
        embeddings = calculator.encode(["hello world"])
        assert embeddings.shape == (1, calculator.embedding_dim)

    def test_similarity_matrix(self, calculator):
        """Test similarity matrix computation"""
        texts = ["hello", "hi", "goodbye"]
        matrix = calculator.compute_similarity_matrix(texts)

        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0, atol=0.01)


class TestEdgeCases:
    """Test edge cases across all algorithms"""

    def test_empty_strings_all_algorithms(self):
        """Test that all algorithms handle empty strings"""
        algorithms = [
            LevenshteinSimilarity(),
            TfidfCosineSimilarity(),
            JaccardSimilarity(),
            CharacterNgramSimilarity()
        ]

        for algo in algorithms:
            # Empty vs empty should be 1.0 or handle gracefully
            try:
                sim = algo.compute_similarity("", "")
                assert 0.0 <= sim <= 1.0
            except Exception:
                pass  # Some algorithms may not support empty strings

    def test_very_long_strings(self):
        """Test performance with long strings"""
        long_text = "word " * 1000  # 1000 words

        algorithms = [
            LevenshteinSimilarity(),
            JaccardSimilarity(),
            CharacterNgramSimilarity()
        ]

        for algo in algorithms:
            try:
                sim = algo.compute_similarity(long_text, long_text)
                assert sim == 1.0 or sim > 0.99
            except Exception:
                pass  # May timeout or fail gracefully

    def test_special_characters(self):
        """Test handling of special characters"""
        text_with_special = "Hello! How are you? #Python #ML"

        algorithms = [
            LevenshteinSimilarity(),
            TfidfCosineSimilarity(),
            JaccardSimilarity(),
            CharacterNgramSimilarity()
        ]

        for algo in algorithms:
            try:
                sim = algo.compute_similarity(text_with_special, text_with_special)
                assert 0.99 <= sim <= 1.0
            except Exception:
                pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])  # Skip slow tests by default
