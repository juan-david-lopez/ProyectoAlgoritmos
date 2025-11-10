# Automatic Term Extraction - Documentation

## Overview

This module provides automatic extraction of key terms from text documents using multiple algorithms and ensemble methods. It's designed to discover important terms without requiring a predefined list.

**Location**: `src/preprocessing/term_analysis/automatic_term_extractor.py`

**Related Components**:
- `term_evaluator.py`: Evaluation and comparison tools
- `examples/automatic_extraction_demo.py`: Complete demonstration

---

## Table of Contents

1. [Core Components](#core-components)
2. [Extraction Methods](#extraction-methods)
3. [Evaluation System](#evaluation-system)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

---

## Core Components

### AutomaticTermExtractor

Main class for automatic keyword extraction with 3 algorithms + ensemble method.

**Constructor**:
```python
AutomaticTermExtractor(
    abstracts: List[str],
    max_terms: int = 15,
    language: str = 'english'
)
```

**Parameters**:
- `abstracts`: List of text documents to analyze
- `max_terms`: Default number of terms to extract per method
- `language`: Language for stopwords (default: 'english')

**Initialization**:
- Automatically downloads required NLTK resources
- Checks for optional dependencies (rake-nltk, spacy, pytextrank)
- Preprocesses all documents for analysis

### TermEvaluator

Evaluation system for comparing extraction methods against ground truth.

**Constructor**:
```python
TermEvaluator(ground_truth_terms: List[str])
```

**Parameters**:
- `ground_truth_terms`: Reference terms for evaluation (e.g., expert-defined keywords)

---

## Extraction Methods

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Mathematical Basis**:
```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)

Where:
- TF(t,d) = frequency of term t in document d
- IDF(t,D) = log(N / df_t)
  - N = total number of documents
  - df_t = number of documents containing term t
```

**What it finds**: Terms that are frequent in specific documents but rare across the corpus.

**Usage**:
```python
extractor = AutomaticTermExtractor(abstracts)
tfidf_terms = extractor.extract_with_tfidf(n_terms=15)

# Returns: [(term, score), ...]
# Example: [('machine learning', 0.8543), ('neural networks', 0.7621), ...]
```

**Configuration**:
```python
TfidfVectorizer(
    max_features=500,       # Top 500 candidates
    ngram_range=(1, 3),    # Unigrams, bigrams, trigrams
    min_df=2,              # Must appear in ≥2 documents
    max_df=0.8,            # Must appear in <80% of documents
    sublinear_tf=True      # Use log-scaling for term frequency
)
```

**Best for**: Finding distinctive technical terms and multi-word phrases

**Strengths**:
- ✅ Excellent for multi-word phrases (n-grams)
- ✅ Filters out overly common terms (max_df)
- ✅ Filters out rare noise (min_df)
- ✅ Fast and scalable

**Limitations**:
- ❌ May miss important common terms
- ❌ Requires multiple documents to work well

---

### 2. RAKE (Rapid Automatic Keyword Extraction)

**Mathematical Basis**:
```
RAKE Score = degree(word) / frequency(word)

Where:
- degree(word) = number of co-occurrences with other words
- frequency(word) = total occurrences

Phrases are scored as: sum(word_scores) for words in phrase
```

**What it finds**: Multi-word phrases with high co-occurrence ratios.

**Usage**:
```python
rake_terms = extractor.extract_with_rake(n_terms=15)

# Returns: [(term, score), ...]
# Example: [('generative ai models', 12.5), ('prompt engineering', 8.3), ...]
```

**Best for**: Extracting descriptive multi-word phrases

**Strengths**:
- ✅ Excellent for domain-specific phrases
- ✅ Captures semantic units naturally
- ✅ No training data required
- ✅ Works with single documents

**Limitations**:
- ❌ May extract overly specific phrases
- ❌ Scores not normalized (harder to interpret)

---

### 3. TextRank (Graph-based)

**Mathematical Basis**:
```
TextRank uses PageRank algorithm on word co-occurrence graph:

TR(V_i) = (1-d) + d × Σ(w_ji / Σw_jk × TR(V_j))
                      j∈In(V_i)

Where:
- d = damping factor (0.85)
- w_ji = edge weight between words j and i
- In(V_i) = vertices pointing to V_i
```

**What it finds**: Central terms in the semantic network of the text.

**Usage**:
```python
textrank_terms = extractor.extract_with_textrank(n_terms=15)

# Returns: [(term, score), ...]
# Example: [('learning', 0.0523), ('ai', 0.0498), ...]
```

**Fallback Mode**: If spaCy/pytextrank unavailable, uses simplified co-occurrence graph with degree centrality.

**Best for**: Finding core concepts that connect ideas

**Strengths**:
- ✅ Captures semantic importance
- ✅ Unsupervised (no training needed)
- ✅ Good at finding topical terms

**Limitations**:
- ❌ Requires spaCy + pytextrank (or uses fallback)
- ❌ May favor high-frequency terms
- ❌ More computationally intensive

---

### 4. Combined Ensemble Method

**Mathematical Basis**:
```
Combined_Score(t) = Σ(w_i × normalize(Score_i(t)))
                    i∈methods

Normalization: Score'(t) = (Score(t) - min) / (max - min)

Default weights:
- w_tfidf = 0.4
- w_rake = 0.3
- w_textrank = 0.3
```

**What it finds**: Terms that perform well across multiple algorithms.

**Usage**:
```python
combined_terms = extractor.extract_combined(n_terms=15)

# Returns: [(term, combined_score, scores_dict), ...]
# scores_dict contains individual method scores
```

**Example Result**:
```python
[
    ('machine learning', 0.8234, {
        'tfidf': 0.8543,
        'rake': 0.7201,
        'textrank': 0.8891,
        'methods_count': 3  # Detected by all 3 methods
    }),
    ...
]
```

**Best for**: Most robust and reliable extraction

**Strengths**:
- ✅ Combines strengths of all methods
- ✅ More robust than single methods
- ✅ Provides transparency (individual scores)
- ✅ Terms detected by multiple methods are prioritized

**Configuration**:
```python
# Modify weights in source code:
weights = {
    'tfidf': 0.4,    # Increase for better phrase detection
    'rake': 0.3,     # Increase for more descriptive phrases
    'textrank': 0.3  # Increase for topical coherence
}
```

---

## Evaluation System

### Metrics

**Precision**: Proportion of extracted terms that are correct
```
Precision = TP / (TP + FP)
```
- High precision = few false positives (extracted terms are accurate)

**Recall**: Proportion of correct terms that were extracted
```
Recall = TP / (TP + FN)
```
- High recall = few false negatives (found most relevant terms)

**F1-Score**: Harmonic mean of Precision and Recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balances precision and recall

### Example Evaluation

```python
from src.preprocessing.term_analysis.term_evaluator import TermEvaluator

# Ground truth (expert-defined terms)
ground_truth = [
    "machine learning",
    "deep learning",
    "neural networks",
    "natural language processing",
    "artificial intelligence"
]

# Create evaluator
evaluator = TermEvaluator(ground_truth)

# Prepare results from different methods
methods_results = {
    'TF-IDF': [term for term, _ in tfidf_terms],
    'RAKE': [term for term, _ in rake_terms],
    'TextRank': [term for term, _ in textrank_terms],
    'Combined': [term for term, _, _ in combined_terms]
}

# Compare methods
comparison = evaluator.compare_methods(methods_results)
print(comparison)
```

**Output**:
```
   Method  Precision  Recall  F1-Score  TP  FP  FN  Total Extracted
0  Combined     0.667   0.400     0.500   2   1   3               3
1  TF-IDF       0.500   0.400     0.444   2   2   3               4
2  RAKE         0.333   0.200     0.250   1   2   4               3
3  TextRank     0.333   0.200     0.250   1   2   4               3
```

**Interpretation**:
- **Combined** has best F1-Score (0.500)
- Found 2 correct terms (TP=2)
- 1 incorrect term (FP=1)
- Missed 3 ground truth terms (FN=3)

---

## Quick Start

### Basic Extraction

```python
import json
from src.preprocessing.term_analysis.automatic_term_extractor import AutomaticTermExtractor

# Load documents
with open('data/unified_articles.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

abstracts = [art['abstract'] for art in articles if art.get('abstract')]

# Initialize extractor
extractor = AutomaticTermExtractor(abstracts, max_terms=15)

# Extract with combined method (recommended)
terms = extractor.extract_combined(15)

# Display results
print("\nTop 10 extracted terms:")
for term, score, scores_dict in terms[:10]:
    methods_count = scores_dict['methods_count']
    print(f"  {term}: {score:.4f} (detected by {methods_count} methods)")
```

### With Evaluation

```python
from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer
from src.preprocessing.term_analysis.term_evaluator import TermEvaluator

# Use predefined terms as ground truth
ground_truth = PredefinedTermsAnalyzer.PREDEFINED_TERMS

# Extract with all methods
tfidf_terms = extractor.extract_with_tfidf(15)
rake_terms = extractor.extract_with_rake(15)
textrank_terms = extractor.extract_with_textrank(15)
combined_terms = extractor.extract_combined(15)

# Prepare for evaluation
methods_results = {
    'TF-IDF': [term for term, _ in tfidf_terms],
    'RAKE': [term for term, _ in rake_terms],
    'TextRank': [term for term, _ in textrank_terms],
    'Combined': [term for term, _, _ in combined_terms]
}

# Evaluate
evaluator = TermEvaluator(ground_truth)
comparison = evaluator.compare_methods(methods_results)

print("\nMethod Comparison:")
print(comparison.to_string(index=False))

# Visualize
evaluator.visualize_comparison(
    comparison,
    'output/term_analysis/method_comparison.png'
)

# Generate report
evaluator.generate_evaluation_report(
    comparison,
    methods_results,
    'output/term_analysis/evaluation_report.md'
)
```

---

## Detailed Usage

### Preprocessing Details

The extractor performs sophisticated NLP preprocessing:

```python
def preprocess_for_extraction(self, text: str) -> List[str]:
    """
    Steps:
    1. Tokenization with NLTK word_tokenize
    2. Lowercase conversion
    3. POS tagging (Part-of-Speech)
    4. Filter by POS: NN (nouns), JJ (adjectives), VBG (gerunds)
    5. Lemmatization (convert to base form)
    6. Stopword removal
    """
```

**Example**:
```
Input: "The generative models are learning quickly"

After tokenization: ['The', 'generative', 'models', 'are', 'learning', 'quickly']
After POS filtering: ['generative', 'models', 'learning']  # JJ, NN, VBG
After lemmatization: ['generative', 'model', 'learn']
After stopwords: ['generative', 'model', 'learn']
```

### Comparing Individual Methods

```python
# Extract with each method
tfidf_terms = extractor.extract_with_tfidf(15)
rake_terms = extractor.extract_with_rake(15)
textrank_terms = extractor.extract_with_textrank(15)

# Compare overlap
tfidf_set = set(t.lower() for t, _ in tfidf_terms)
rake_set = set(t.lower() for t, _ in rake_terms)
textrank_set = set(t.lower() for t, _ in textrank_terms)

# Terms found by all 3 methods
consensus_terms = tfidf_set & rake_set & textrank_set
print(f"Terms detected by all 3 methods: {len(consensus_terms)}")
print(consensus_terms)

# Terms unique to each method
unique_tfidf = tfidf_set - rake_set - textrank_set
unique_rake = rake_set - tfidf_set - textrank_set
unique_textrank = textrank_set - tfidf_set - rake_set
```

### Generating Reports

```python
import pandas as pd

# Create detailed DataFrame
all_terms_data = []
for term, score, scores_dict in combined_terms:
    all_terms_data.append({
        'Term': term,
        'Combined Score': f'{score:.4f}',
        'TF-IDF': f'{scores_dict["tfidf"]:.4f}',
        'RAKE': f'{scores_dict["rake"]:.4f}',
        'TextRank': f'{scores_dict["textrank"]:.4f}',
        'Methods Count': scores_dict['methods_count']
    })

terms_df = pd.DataFrame(all_terms_data)

# Save as CSV
terms_df.to_csv('output/extracted_terms.csv', index=False)

# Display in console
print(terms_df.to_string(index=False))
```

---

## Understanding Results

### Interpreting Scores

**TF-IDF Scores** (typically 0.1 - 1.0):
- 0.8+ = Highly distinctive term
- 0.5-0.8 = Moderately distinctive
- <0.5 = Common term

**RAKE Scores** (typically 1.0 - 20.0):
- 15+ = Highly co-occurring phrase
- 5-15 = Moderate co-occurrence
- <5 = Low co-occurrence

**TextRank Scores** (typically 0.01 - 0.10):
- 0.05+ = Central concept
- 0.02-0.05 = Moderate importance
- <0.02 = Peripheral term

**Combined Scores** (normalized 0.0 - 1.0):
- 0.7+ = Strong consensus across methods
- 0.4-0.7 = Moderate agreement
- <0.4 = Weak agreement

### Methods Count

The `methods_count` field shows how many methods detected the term:
- **3**: Detected by all methods (highest confidence)
- **2**: Detected by two methods (good confidence)
- **1**: Detected by one method (method-specific)

**Recommendation**: Prioritize terms with `methods_count >= 2`

### Evaluation Metrics

**Good Performance**:
- Precision ≥ 0.7 (70% of extracted terms are correct)
- Recall ≥ 0.5 (50% of relevant terms found)
- F1-Score ≥ 0.6

**Acceptable Performance**:
- Precision ≥ 0.5
- Recall ≥ 0.3
- F1-Score ≥ 0.4

**Poor Performance** (needs tuning):
- F1-Score < 0.4

---

## Troubleshooting

### Issue: ImportError for rake-nltk

**Error**: `ImportError: No module named 'rake_nltk'`

**Solution**:
```bash
pip install rake-nltk
```

If RAKE unavailable, only TF-IDF and TextRank will run.

---

### Issue: ImportError for spacy/pytextrank

**Error**: `ImportError: No module named 'spacy'` or `'pytextrank'`

**Solution**:
```bash
pip install spacy pytextrank
python -m spacy download en_core_web_sm
```

If unavailable, TextRank will use fallback mode (co-occurrence graph with degree centrality).

---

### Issue: NLTK Resource Not Found

**Error**: `LookupError: Resource punkt not found`

**Solution**: The extractor automatically downloads required resources on first run. If this fails:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

---

### Issue: Too Many Generic Terms

**Problem**: Extracting terms like "study", "paper", "research"

**Solutions**:

1. **Expand stopwords**:
```python
# In automatic_term_extractor.py
self.stopwords = set(stopwords.words(language))
self.stopwords.update(['study', 'paper', 'research', 'article', 'analysis'])
```

2. **Increase min_df in TF-IDF**:
```python
TfidfVectorizer(
    min_df=3,  # Must appear in ≥3 documents (instead of 2)
    ...
)
```

3. **Filter by length**:
```python
# Keep only terms with 2+ words or longer single words
filtered_terms = [
    (term, score) for term, score in terms
    if len(term.split()) >= 2 or len(term) >= 8
]
```

---

### Issue: Missing Multi-word Phrases

**Problem**: Only extracting single words

**Solutions**:

1. **Increase ngram_range in TF-IDF**:
```python
TfidfVectorizer(
    ngram_range=(2, 4),  # Only bigrams, trigrams, 4-grams
    ...
)
```

2. **Increase RAKE weight in ensemble**:
```python
weights = {
    'tfidf': 0.3,
    'rake': 0.5,    # RAKE is best for phrases
    'textrank': 0.2
}
```

---

### Issue: Low Recall (Missing Important Terms)

**Problem**: F1-Score is low due to poor recall

**Solutions**:

1. **Increase n_terms**:
```python
terms = extractor.extract_combined(n_terms=30)  # Extract more candidates
```

2. **Lower min_df in TF-IDF**:
```python
TfidfVectorizer(
    min_df=1,  # Allow terms appearing in 1+ documents
    ...
)
```

3. **Check if terms are in stopwords**:
```python
print(extractor.stopwords)  # Review stopword list
```

---

### Issue: Low Precision (Too Many Incorrect Terms)

**Problem**: F1-Score is low due to poor precision

**Solutions**:

1. **Increase min_df**:
```python
TfidfVectorizer(
    min_df=3,  # Require terms in 3+ documents
    ...
)
```

2. **Decrease max_df**:
```python
TfidfVectorizer(
    max_df=0.6,  # Exclude terms in >60% of documents
    ...
)
```

3. **Filter by methods_count**:
```python
# Keep only terms detected by 2+ methods
filtered_terms = [
    (term, score, scores_dict) for term, score, scores_dict in combined_terms
    if scores_dict['methods_count'] >= 2
]
```

---

### Issue: Inconsistent Results Across Runs

**Problem**: Different terms extracted each time

**Cause**: This is expected behavior - the algorithms are deterministic but may vary slightly due to:
- Document order in corpus
- Tie-breaking in ranking

**Solution**: If you need reproducibility, set random seeds and use fixed document order.

---

## Best Practices

### 1. Start with Combined Method

The ensemble method (`extract_combined()`) is the most robust:
```python
terms = extractor.extract_combined(15)
```

### 2. Evaluate Against Ground Truth

Always validate extraction quality:
```python
evaluator = TermEvaluator(ground_truth_terms)
comparison = evaluator.compare_methods(methods_results)
```

### 3. Prioritize Consensus Terms

Focus on terms detected by multiple methods:
```python
high_confidence = [
    (term, score) for term, score, scores_dict in combined_terms
    if scores_dict['methods_count'] >= 2
]
```

### 4. Visualize and Report

Generate visualizations for better understanding:
```python
evaluator.visualize_comparison(comparison, 'output/comparison.png')
evaluator.generate_evaluation_report(comparison, methods_results, 'output/report.md')
```

### 5. Iterative Refinement

1. Extract terms
2. Evaluate metrics
3. Adjust parameters (weights, min_df, stopwords)
4. Re-extract and compare
5. Repeat until satisfactory

---

## Advanced Usage

### Custom Weighting in Ensemble

```python
# Modify in automatic_term_extractor.py

def extract_combined(self, n_terms: int = 15, custom_weights: dict = None):
    weights = custom_weights or {
        'tfidf': 0.4,
        'rake': 0.3,
        'textrank': 0.3
    }
    # ... rest of method
```

**Usage**:
```python
# Emphasize TF-IDF for technical domains
terms = extractor.extract_combined(15, custom_weights={
    'tfidf': 0.6,
    'rake': 0.2,
    'textrank': 0.2
})
```

### Domain-Specific Stopwords

```python
# Add domain-specific stopwords
extractor.stopwords.update([
    'study', 'research', 'paper', 'article',
    'method', 'approach', 'result', 'conclusion'
])

# Re-extract
terms = extractor.extract_combined(15)
```

### Export for External Tools

```python
import json

# Export as JSON
export_data = {
    'method': 'combined',
    'timestamp': pd.Timestamp.now().isoformat(),
    'terms': [
        {
            'term': term,
            'score': float(score),
            'tfidf': float(scores_dict['tfidf']),
            'rake': float(scores_dict['rake']),
            'textrank': float(scores_dict['textrank']),
            'methods_count': scores_dict['methods_count']
        }
        for term, score, scores_dict in combined_terms
    ]
}

with open('output/extracted_terms.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)
```

---

## References

### Academic Papers

1. **TF-IDF**: Salton & Buckley (1988). "Term-weighting approaches in automatic text retrieval"
2. **RAKE**: Rose et al. (2010). "Automatic Keyword Extraction from Individual Documents"
3. **TextRank**: Mihalcea & Tarau (2004). "TextRank: Bringing Order into Text"

### Libraries

- **scikit-learn**: TF-IDF implementation
- **rake-nltk**: RAKE implementation
- **spaCy + pytextrank**: TextRank implementation
- **NLTK**: Preprocessing tools

---

## Summary

**Key Takeaways**:

1. **Three extraction methods** with different strengths:
   - TF-IDF: Statistical, great for phrases
   - RAKE: Co-occurrence, descriptive phrases
   - TextRank: Graph-based, topical concepts

2. **Combined ensemble method** is most robust

3. **Evaluation system** provides Precision/Recall/F1 metrics

4. **Automatic preprocessing** handles tokenization, lemmatization, POS tagging

5. **Comprehensive reporting** with visualizations and Markdown reports

**Recommended Workflow**:
```
1. Load documents → AutomaticTermExtractor
2. Extract with combined method
3. Evaluate against ground truth → TermEvaluator
4. Generate visualizations and reports
5. Refine parameters if needed
6. Use extracted terms for analysis
```

---

For more information, see:
- `examples/automatic_extraction_demo.py` - Complete working example
- `REQUERIMIENTO_3_SUMMARY.md` - Full system overview
- Source code documentation in module files
