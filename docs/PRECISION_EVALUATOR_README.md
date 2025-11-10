# Term Precision Evaluator - Documentation

## Overview

The Term Precision Evaluator compares automatically extracted terms with predefined terms using **semantic similarity** to evaluate extraction quality. Unlike simple string matching, it uses AI-powered embeddings (SBERT) to identify matches even when terms are expressed differently.

**Key Innovation**: Recognizes that "machine learning" and "ML algorithms" are semantically similar, enabling more accurate evaluation than lexical matching alone.

**Location**: `src/preprocessing/term_analysis/term_precision_evaluator.py`

**Related Components**:
- `automatic_term_extractor.py`: Automatic term extraction
- `predefined_terms_analyzer.py`: Predefined terms reference
- `examples/precision_evaluation_demo.py`: Complete demonstration

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Quick Start](#quick-start)
3. [Similarity Methods](#similarity-methods)
4. [Match Categories](#match-categories)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Detailed Usage](#detailed-usage)
7. [Visualizations](#visualizations)
8. [Understanding Results](#understanding-results)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Core Concepts

### Semantic vs Lexical Similarity

**Lexical Similarity** (string-based):
```python
"machine learning" vs "ML algorithms" → Low similarity (~0.2)
"fine-tuning" vs "fine tuning" → High similarity (~0.9)
```

**Semantic Similarity** (meaning-based with SBERT):
```python
"machine learning" vs "ML algorithms" → High similarity (~0.75)
"fine-tuning" vs "fine tuning" → Very high similarity (~0.95)
"neural networks" vs "deep neural nets" → High similarity (~0.80)
```

### Why Semantic Similarity?

Automatically extracted terms often use different vocabulary than predefined terms, even when referring to the same concept:

| Predefined Term | Extracted Term | Lexical Similarity | Semantic Similarity |
|----------------|----------------|-------------------|---------------------|
| "Natural language processing" | "NLP techniques" | 0.15 | 0.78 |
| "Machine learning" | "ML algorithms" | 0.25 | 0.82 |
| "Fine-tuning" | "Model tuning" | 0.35 | 0.71 |
| "Ethics" | "Ethical considerations" | 0.45 | 0.88 |

Semantic similarity provides more accurate evaluation by understanding meaning, not just matching characters.

---

## Quick Start

### Basic Usage

```python
from src.preprocessing.term_analysis.term_precision_evaluator import TermPrecisionEvaluator

# Predefined terms (ground truth)
predefined_terms = [
    "machine learning",
    "deep learning",
    "neural networks",
    "natural language processing",
    "artificial intelligence"
]

# Extracted terms (from automatic extraction)
extracted_terms = [
    "machine learning",
    "ML algorithms",
    "deep neural nets",
    "NLP techniques",
    "computer vision"
]

# Create evaluator
evaluator = TermPrecisionEvaluator(predefined_terms, extracted_terms)

# Calculate similarity matrix
similarity_matrix = evaluator.calculate_similarity_matrix()

# Identify matches
matches = evaluator.identify_matches(threshold=0.70)

# Calculate metrics
metrics = evaluator.calculate_metrics(matches)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
```

### Complete Workflow

```python
# 1. Extract terms automatically
from src.preprocessing.term_analysis.automatic_term_extractor import AutomaticTermExtractor
from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer

extractor = AutomaticTermExtractor(abstracts)
combined_terms = extractor.extract_combined(15)
extracted_list = [term for term, _, _ in combined_terms]

# 2. Get predefined terms
predefined_list = PredefinedTermsAnalyzer.PREDEFINED_TERMS

# 3. Evaluate precision
evaluator = TermPrecisionEvaluator(predefined_list, extracted_list)
matches = evaluator.identify_matches(threshold=0.70)
metrics = evaluator.calculate_metrics(matches)

# 4. Explain novel terms
novel_explanations = evaluator.explain_novel_terms(
    matches['novel_terms'],
    abstracts
)

# 5. Generate visualizations and report
evaluator.visualize_similarity_matrix('output/similarity_matrix.png')
evaluator.visualize_venn_diagram(matches, 'output/venn_diagram.png')
evaluator.generate_evaluation_report(
    matches,
    metrics,
    novel_explanations,
    'output/evaluation_report.md'
)
```

---

## Similarity Methods

### Method 1: SBERT (Sentence-BERT) - Recommended

**How it works**:
1. Converts each term to a 384-dimensional embedding vector
2. Calculates cosine similarity between vectors
3. Similarity score ranges from 0 (completely different) to 1 (identical)

**Model**: `all-MiniLM-L6-v2`
- Size: ~90MB
- Speed: Fast (processes hundreds of terms per second)
- Quality: State-of-the-art semantic similarity

**Installation**:
```bash
pip install sentence-transformers
```

**Example Embeddings**:
```python
# "machine learning" → [0.12, -0.45, 0.78, ..., 0.34]  (384 dimensions)
# "ML algorithms"   → [0.15, -0.42, 0.81, ..., 0.31]  (similar vector!)

# Cosine similarity = 0.82 (high similarity)
```

**When SBERT is active**:
```
Inicializando TermPrecisionEvaluator...
  Cargando modelo SBERT...
  ✓ SBERT cargado correctamente
```

### Method 2: Lexical Similarity (Fallback)

**How it works**:
1. Uses Python's `difflib.SequenceMatcher`
2. Compares character sequences
3. Score based on longest common substring ratio

**When it's used**:
- SBERT not installed
- SBERT loading fails
- Explicitly disabled

**Example**:
```python
lexical_similarity("machine learning", "ML algorithms")
# → 0.25 (low - different characters)

lexical_similarity("fine-tuning", "fine tuning")
# → 0.91 (high - similar characters)
```

**When fallback is active**:
```
⚠️ SBERT no disponible (instalar: pip install sentence-transformers)
Usando similitud léxica como fallback
```

### Comparison

| Aspect | SBERT | Lexical |
|--------|-------|---------|
| **Accuracy** | High (understands meaning) | Low (only character matching) |
| **Speed** | Fast (~1000 terms/sec) | Very fast (~10000 terms/sec) |
| **Memory** | ~200MB | Minimal |
| **Setup** | Requires installation | Built-in |
| **Use case** | Production, accurate evaluation | Quick tests, no dependencies |

**Recommendation**: Always use SBERT for production evaluation.

---

## Match Categories

The evaluator classifies term pairs into 4 categories:

### 1. Exact Matches (similarity ≥ threshold)

**Definition**: Predefined terms with high semantic similarity to extracted terms.

**Examples** (threshold=0.70):
```python
[
    ("machine learning", "machine learning", 1.000),  # Perfect match
    ("deep learning", "deep neural learning", 0.89),  # Very similar
    ("ethics", "ethical considerations", 0.75),       # Related concept
]
```

**Interpretation**: These are successfully extracted concepts, even if wording differs.

### 2. Partial Matches (0.50 ≤ similarity < threshold)

**Definition**: Terms with moderate similarity - related but not identical concepts.

**Examples** (threshold=0.70):
```python
[
    ("neural networks", "network architectures", 0.65),
    ("training data", "data preprocessing", 0.58),
    ("personalization", "adaptive learning", 0.52),
]
```

**Interpretation**: Extraction captured related concepts but not exact matches. May indicate:
- Different aspect of same topic
- Broader or narrower concept
- Complementary terms

### 3. Novel Terms (no match)

**Definition**: Extracted terms with no similar predefined term (all similarities < 0.50).

**Examples**:
```python
[
    "computer vision",
    "reinforcement learning",
    "transfer learning",
]
```

**Interpretation**: New relevant concepts discovered by extraction that weren't in predefined list. These can be:
- **True discoveries**: Important terms missed by predefined list
- **False positives**: Irrelevant terms incorrectly extracted
- **Domain expansion**: Related but out-of-scope concepts

### 4. Predefined Not Found (no match)

**Definition**: Predefined terms with no similar extracted term (all similarities < 0.50).

**Examples**:
```python
[
    "algorithmic bias",
    "transparency",
    "co-creation",
]
```

**Interpretation**: Expected terms that extraction failed to capture. Possible reasons:
- Terms don't appear frequently enough in corpus
- Filtered by extraction parameters (min_df, max_df)
- Terms too general to rank highly
- Vocabulary mismatch (different wording in corpus)

---

## Evaluation Metrics

### Precision

**Formula**:
```
Precision = (Exact Matches + Partial Matches) / Total Extracted Terms
```

**What it measures**: Proportion of extracted terms that are relevant.

**Example**:
```
Extracted: 10 terms
Exact matches: 4
Partial matches: 2
Novel terms: 4

Precision = (4 + 2) / 10 = 0.60 (60% of extracted terms are relevant)
```

**High precision** (≥0.7): Extraction is accurate, few false positives.

**Low precision** (<0.5): Extraction produces many irrelevant terms.

### Recall

**Formula**:
```
Recall = (Exact Matches + Partial Matches) / Total Predefined Terms
```

**What it measures**: Proportion of expected terms that were found.

**Example**:
```
Predefined: 15 terms
Exact matches: 4
Partial matches: 2
Not found: 9

Recall = (4 + 2) / 15 = 0.40 (40% of expected terms were found)
```

**High recall** (≥0.7): Extraction captures most relevant terms.

**Low recall** (<0.5): Extraction misses many relevant terms.

### F1-Score

**Formula**:
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**What it measures**: Harmonic mean of precision and recall - balanced metric.

**Example**:
```
Precision = 0.60
Recall = 0.40

F1 = 2 × (0.60 × 0.40) / (0.60 + 0.40) = 0.48
```

**Interpretation**:
- **F1 ≥ 0.70**: Excellent extraction quality
- **F1 0.50-0.69**: Good extraction with room for improvement
- **F1 < 0.50**: Poor extraction, needs tuning

**Why F1?**: Balances precision and recall. High F1 means both are good.

### Coverage

**Formula**:
```
Coverage = (Exact Matches + Partial Matches) / Total Predefined Terms
```

**What it measures**: Same as recall, presented as percentage.

**Example**:
```
Coverage = 6 / 15 = 0.40 = 40%
```

**Interpretation**: Percentage of predefined terms covered by extraction.

### Detailed Example

```
Scenario:
- 15 predefined terms
- 12 extracted terms

Results:
- 5 exact matches
- 3 partial matches
- 4 novel terms
- 7 predefined not found

Metrics:
- Precision = (5+3) / 12 = 0.667
- Recall = (5+3) / 15 = 0.533
- F1-Score = 2×(0.667×0.533)/(0.667+0.533) = 0.593
- Coverage = 8/15 = 53.3%

Assessment: Good precision (67%), moderate recall (53%), overall F1 = 0.59
```

---

## Detailed Usage

### Calculating Similarity Matrix

```python
evaluator = TermPrecisionEvaluator(predefined_terms, extracted_terms)

# Calculate matrix (automatically chooses SBERT or lexical)
similarity_matrix = evaluator.calculate_similarity_matrix()

# Shape: (len(predefined), len(extracted))
print(similarity_matrix.shape)  # e.g., (15, 12)

# Access specific similarity
pred_idx = 0
ext_idx = 3
sim = similarity_matrix[pred_idx, ext_idx]
print(f"{predefined_terms[pred_idx]} <-> {extracted_terms[ext_idx]}: {sim:.3f}")

# Find best match for a predefined term
best_ext_idx = similarity_matrix[0, :].argmax()
best_similarity = similarity_matrix[0, best_ext_idx]
print(f"Best match: {extracted_terms[best_ext_idx]} (sim={best_similarity:.3f})")

# Statistics
print(f"Average similarity: {similarity_matrix.mean():.3f}")
print(f"Max similarity: {similarity_matrix.max():.3f}")
print(f"Min similarity: {similarity_matrix.min():.3f}")
```

### Identifying Matches with Different Thresholds

```python
# Strict threshold (high precision, low recall)
matches_strict = evaluator.identify_matches(threshold=0.80)

# Moderate threshold (balanced)
matches_moderate = evaluator.identify_matches(threshold=0.70)

# Lenient threshold (low precision, high recall)
matches_lenient = evaluator.identify_matches(threshold=0.60)

# Compare
for name, matches in [("Strict", matches_strict),
                       ("Moderate", matches_moderate),
                       ("Lenient", matches_lenient)]:
    metrics = evaluator.calculate_metrics(matches)
    print(f"{name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1_score']:.2f}")
```

### Explaining Novel Terms

```python
matches = evaluator.identify_matches(threshold=0.70)

if matches['novel_terms']:
    explanations = evaluator.explain_novel_terms(
        matches['novel_terms'],
        abstracts
    )

    for term, expl in explanations.items():
        print(f"\nTerm: {term}")
        print(f"  Frequency: {expl['frequency']}")
        print(f"  Documents: {expl['document_frequency']}")
        print(f"  Relevance: {expl['relevance_score']:.3f}")
        print(f"  Interpretation: {expl['interpretation']}")

        if expl['example_contexts']:
            print(f"  Example: {expl['example_contexts'][0]}")
```

### Threshold Analysis

Find optimal threshold by testing multiple values:

```python
import pandas as pd

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
results = []

for threshold in thresholds:
    matches = evaluator.identify_matches(threshold=threshold)
    metrics = evaluator.calculate_metrics(matches)

    results.append({
        'Threshold': threshold,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'Exact': len(matches['exact_matches']),
        'Partial': len(matches['partial_matches'])
    })

df = pd.DataFrame(results)
print(df)

# Find best F1
best_idx = df['F1-Score'].idxmax()
best_threshold = df.loc[best_idx, 'Threshold']
print(f"\nBest threshold: {best_threshold} (F1={df.loc[best_idx, 'F1-Score']:.3f})")
```

---

## Visualizations

### 1. Similarity Matrix Heatmap

**What it shows**: Similarity scores between all predefined and extracted terms.

**How to generate**:
```python
evaluator.visualize_similarity_matrix('output/similarity_matrix.png')
```

**Reading the heatmap**:
- **Rows**: Predefined terms
- **Columns**: Extracted terms
- **Color**: Green (high similarity) → Yellow → Red (low similarity)
- **Values**: Similarity scores (0.00 - 1.00)

**Use cases**:
- Identify which extracted terms match which predefined terms
- Spot patterns in vocabulary differences
- Find semantic clusters

### 2. Venn Diagram / Category Distribution

**What it shows**: Distribution of terms across categories.

**How to generate**:
```python
matches = evaluator.identify_matches(threshold=0.70)
evaluator.visualize_venn_diagram(matches, 'output/venn_diagram.png')
```

**Components**:

**Subplot 1 - Bar Chart**:
- Exact Matches (green)
- Partial Matches (orange)
- Novel Terms (blue)
- Not Found (red)

**Subplot 2 - Pie Chart**:
- Matched (exact + partial)
- Novel Terms
- Not Found

**Use cases**:
- Quick overview of evaluation results
- Presentation-ready visualizations
- Identify balance between categories

---

## Understanding Results

### Good Results Example

```
Metrics:
  Precision: 0.750
  Recall: 0.667
  F1-Score: 0.706
  Coverage: 66.7%

Breakdown:
  Exact matches: 8
  Partial matches: 2
  Novel terms: 3
  Not found: 5

Assessment: ✅ EXCELLENT - High concordance
```

**Interpretation**:
- 75% of extracted terms are relevant (high precision)
- 67% of predefined terms found (good recall)
- F1=0.706 indicates excellent overall performance
- Few novel terms suggests good alignment with expectations
- Some terms not found is normal (may not be in corpus)

### Moderate Results Example

```
Metrics:
  Precision: 0.550
  Recall: 0.467
  F1-Score: 0.505
  Coverage: 46.7%

Breakdown:
  Exact matches: 4
  Partial matches: 3
  Novel terms: 6
  Not found: 8

Assessment: ✓ GOOD - Concordance moderada con margen de mejora
```

**Interpretation**:
- 55% of extracted terms relevant (moderate precision)
- 47% of predefined terms found (moderate recall)
- F1=0.505 indicates acceptable but improvable performance
- Many novel terms suggests extraction finding new concepts
- Many not found suggests corpus vocabulary mismatch

**Actions**:
- Review novel terms - are they relevant? Add good ones to predefined list
- Check predefined terms not found - do they appear in corpus?
- Adjust extraction parameters (increase n_terms for better recall)

### Poor Results Example

```
Metrics:
  Precision: 0.300
  Recall: 0.267
  F1-Score: 0.282
  Coverage: 26.7%

Breakdown:
  Exact matches: 2
  Partial matches: 2
  Novel terms: 10
  Not found: 11

Assessment: ⚠️ REQUIRES REVIEW - Low concordance
```

**Interpretation**:
- 30% of extracted terms relevant (low precision - many false positives)
- 27% of predefined terms found (low recall - missing expected terms)
- F1=0.282 indicates poor performance
- Many novel terms suggests extraction off-target
- Many not found suggests fundamental mismatch

**Actions**:
- **Review extraction method**: Is combined/ensemble method being used?
- **Check corpus**: Does it actually contain predefined terms?
- **Adjust parameters**:
  - Increase min_df to reduce noise
  - Increase n_terms to capture more candidates
  - Review stopwords (may be filtering important terms)
- **Review predefined terms**: Are they appropriate for this corpus?

---

## Troubleshooting

### Issue: SBERT Not Available

**Symptoms**:
```
⚠️ SBERT no disponible (instalar: pip install sentence-transformers)
Usando similitud léxica como fallback
```

**Impact**: Lower accuracy - lexical similarity misses semantic matches.

**Solution**:
```bash
pip install sentence-transformers
```

This will install ~500MB of dependencies (PyTorch + transformers).

**Alternative**: If installation fails or disk space limited, lexical similarity still provides basic functionality.

---

### Issue: Low Precision (Many Novel Terms)

**Symptoms**:
- Precision < 0.5
- Many novel terms (>50% of extracted)
- F1-Score low

**Possible Causes**:
1. Extraction capturing irrelevant terms
2. Predefined list too narrow
3. Extraction parameters too lenient

**Solutions**:

**1. Increase extraction selectivity**:
```python
# In automatic_term_extractor.py
TfidfVectorizer(
    min_df=3,      # Increase from 2 (filter rare terms)
    max_df=0.7,    # Decrease from 0.8 (filter common terms)
    ...
)
```

**2. Use ensemble method** (most robust):
```python
combined_terms = extractor.extract_combined(15)  # Not individual methods
```

**3. Filter by methods_count**:
```python
# Keep only terms detected by 2+ methods
filtered = [
    term for term, _, scores_dict in combined_terms
    if scores_dict['methods_count'] >= 2
]
```

**4. Review novel terms** - add relevant ones to predefined list:
```python
# Check explanations
for term in matches['novel_terms']:
    if term in novel_explanations:
        print(f"{term}: relevance={novel_explanations[term]['relevance_score']:.3f}")
```

---

### Issue: Low Recall (Many Not Found)

**Symptoms**:
- Recall < 0.5
- Many predefined terms not found
- F1-Score low

**Possible Causes**:
1. Terms not in corpus
2. Terms filtered by extraction parameters
3. Not extracting enough terms (n_terms too low)

**Solutions**:

**1. Increase n_terms**:
```python
combined_terms = extractor.extract_combined(30)  # Increase from 15
```

**2. Lower min_df**:
```python
TfidfVectorizer(
    min_df=1,  # Allow terms appearing in 1+ documents
    ...
)
```

**3. Check if terms actually appear**:
```python
from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer

analyzer = PredefinedTermsAnalyzer()
frequencies = analyzer.calculate_frequencies(abstracts)

for term in matches['predefined_not_found']:
    if term in frequencies:
        print(f"{term}: found {frequencies[term]['total_count']} times")
    else:
        print(f"{term}: NOT IN CORPUS")
```

**4. Check stopwords**:
```python
# See if terms are being filtered
print(extractor.stopwords)

# If necessary, remove terms from stopwords
extractor.stopwords.discard('ethics')  # Example
```

---

### Issue: All Similarities Near 0.5

**Symptoms**:
- Most similarities in 0.4-0.6 range
- Few exact matches even with low threshold
- Unclear categorization

**Cause**: Using lexical similarity (SBERT not available) with terms that are semantically related but lexically different.

**Example**:
```
"machine learning" vs "ML algorithms"
Lexical: 0.25
SBERT: 0.82
```

**Solution**: Install SBERT for semantic similarity:
```bash
pip install sentence-transformers
```

---

### Issue: Threshold Selection

**Question**: What threshold should I use?

**Answer**: Depends on use case and similarity method.

**For SBERT (semantic)**:
- **0.80-1.00**: Very strict - only very similar terms
- **0.70-0.80**: Recommended - balanced
- **0.60-0.70**: Lenient - captures related concepts
- **0.50-0.60**: Very lenient - may include tangentially related

**For Lexical**:
- **0.90-1.00**: Very strict - nearly identical strings
- **0.80-0.90**: Recommended - similar spelling
- **0.70-0.80**: Lenient - some differences allowed
- **<0.70**: Very lenient - many character differences

**How to choose**:
1. Run threshold analysis (test 0.50, 0.60, 0.70, 0.80, 0.90)
2. Plot F1-Score vs threshold
3. Choose threshold with best F1
4. Manually review matches at that threshold
5. Adjust based on false positives/negatives

**Example**:
```python
# Test multiple thresholds
for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
    matches = evaluator.identify_matches(threshold)
    metrics = evaluator.calculate_metrics(matches)
    print(f"{threshold}: F1={metrics['f1_score']:.3f}")

# Output:
# 0.60: F1=0.642
# 0.65: F1=0.678
# 0.70: F1=0.706  ← Best
# 0.75: F1=0.650
# 0.80: F1=0.583
```

---

## Advanced Usage

### Custom Similarity Function

Override similarity calculation:

```python
class CustomEvaluator(TermPrecisionEvaluator):
    def calculate_similarity_matrix(self):
        # Custom similarity logic
        # Example: combine SBERT + domain-specific rules

        # First get SBERT similarities
        sbert_matrix = super().calculate_similarity_matrix()

        # Apply custom adjustments
        for i, pred_term in enumerate(self.predefined_terms):
            for j, ext_term in enumerate(self.extracted_terms):
                # Boost similarity if both contain same acronym
                if self._same_acronym(pred_term, ext_term):
                    sbert_matrix[i, j] = min(1.0, sbert_matrix[i, j] + 0.1)

        return sbert_matrix

    def _same_acronym(self, term1, term2):
        acronyms1 = re.findall(r'\b[A-Z]{2,}\b', term1)
        acronyms2 = re.findall(r'\b[A-Z]{2,}\b', term2)
        return bool(set(acronyms1) & set(acronyms2))
```

### Batch Evaluation

Evaluate multiple extraction configurations:

```python
from src.preprocessing.term_analysis.automatic_term_extractor import AutomaticTermExtractor
from src.preprocessing.term_analysis.predefined_terms_analyzer import PredefinedTermsAnalyzer

predefined = PredefinedTermsAnalyzer.PREDEFINED_TERMS

# Test different extraction methods
results = []

for method_name, extract_func in [
    ('TF-IDF', lambda ext: ext.extract_with_tfidf(15)),
    ('RAKE', lambda ext: ext.extract_with_rake(15)),
    ('TextRank', lambda ext: ext.extract_with_textrank(15)),
    ('Combined', lambda ext: ext.extract_combined(15))
]:
    extractor = AutomaticTermExtractor(abstracts)
    terms = extract_func(extractor)

    # Convert to list
    if isinstance(terms[0], tuple) and len(terms[0]) == 3:
        extracted = [t for t, _, _ in terms]  # Combined
    else:
        extracted = [t for t, _ in terms]  # Others

    # Evaluate
    evaluator = TermPrecisionEvaluator(predefined, extracted)
    matches = evaluator.identify_matches(threshold=0.70)
    metrics = evaluator.calculate_metrics(matches)

    results.append({
        'Method': method_name,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score']
    })

df = pd.DataFrame(results)
print(df.sort_values('F1-Score', ascending=False))
```

### Export Results

```python
import json

# Export matches and metrics
export_data = {
    'metadata': {
        'timestamp': pd.Timestamp.now().isoformat(),
        'similarity_method': 'SBERT' if evaluator.use_sbert else 'Lexical',
        'threshold': 0.70
    },
    'metrics': metrics,
    'matches': {
        'exact': [
            {'predefined': p, 'extracted': e, 'similarity': float(s)}
            for p, e, s in matches['exact_matches']
        ],
        'partial': [
            {'predefined': p, 'extracted': e, 'similarity': float(s)}
            for p, e, s in matches['partial_matches']
        ],
        'novel_terms': matches['novel_terms'],
        'not_found': matches['predefined_not_found']
    }
}

with open('output/precision_results.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)
```

---

## Best Practices

### 1. Always Use SBERT for Production

Lexical similarity is a fallback, but SBERT provides much better accuracy:

```python
# Check if SBERT is being used
if evaluator.use_sbert:
    print("✓ Using SBERT (semantic similarity)")
else:
    print("⚠️ Using lexical similarity - install sentence-transformers for better results")
```

### 2. Start with Threshold=0.70

This is a balanced default for SBERT:

```python
matches = evaluator.identify_matches(threshold=0.70)
```

Adjust based on results:
- Too many false positives in exact matches? Increase to 0.75-0.80
- Too few matches? Decrease to 0.65-0.70

### 3. Review Novel Terms

Novel terms aren't always bad - they may reveal important concepts:

```python
explanations = evaluator.explain_novel_terms(matches['novel_terms'], abstracts)

high_relevance = [
    term for term, expl in explanations.items()
    if expl['relevance_score'] > 0.3 and expl['document_frequency'] >= 2
]

print(f"High-relevance novel terms to consider adding to predefined list:")
for term in high_relevance:
    print(f"  - {term}")
```

### 4. Generate Complete Reports

Always generate visualizations and reports for documentation:

```python
evaluator.visualize_similarity_matrix('output/similarity_matrix.png')
evaluator.visualize_venn_diagram(matches, 'output/venn_diagram.png')
evaluator.generate_evaluation_report(
    matches, metrics, novel_explanations,
    'output/evaluation_report.md'
)
```

### 5. Iterate and Refine

Use evaluation results to improve extraction:

1. **Run evaluation** → Identify issues (low precision/recall)
2. **Adjust extraction** → Change parameters, methods
3. **Re-evaluate** → Check if metrics improved
4. **Repeat** → Until satisfactory

---

## Summary

**Key Takeaways**:

1. **Semantic similarity (SBERT)** provides more accurate evaluation than lexical matching

2. **Four match categories**:
   - Exact matches: Successfully extracted
   - Partial matches: Related concepts
   - Novel terms: New discoveries (review for relevance)
   - Not found: Missing expected terms

3. **Metrics**:
   - Precision: Accuracy of extracted terms
   - Recall: Coverage of expected terms
   - F1-Score: Overall quality (aim for ≥0.70)

4. **Threshold selection**: Start with 0.70, adjust based on F1-Score analysis

5. **Comprehensive evaluation**: Use visualizations + reports for complete understanding

**Recommended Workflow**:
```
1. Extract terms → AutomaticTermExtractor
2. Create evaluator → TermPrecisionEvaluator
3. Calculate similarity matrix
4. Identify matches (threshold=0.70)
5. Calculate metrics
6. Explain novel terms
7. Generate visualizations and report
8. Review and iterate
```

---

For more information, see:
- `examples/precision_evaluation_demo.py` - Complete working example
- `src/preprocessing/term_analysis/term_precision_evaluator.py` - Source code
- `REQUERIMIENTO_3_SUMMARY.md` - Full system overview
