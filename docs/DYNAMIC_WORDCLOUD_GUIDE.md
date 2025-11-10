# Dynamic Word Cloud Visualization Guide

Comprehensive guide for using the Dynamic Word Cloud visualization system to analyze term frequency and evolution in scientific publications.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

The Dynamic Word Cloud module provides professional, automatically updatable word cloud visualizations for analyzing scientific production. It extracts terms from abstracts and keywords, applies sophisticated weighting schemes, and generates both static and interactive visualizations.

### Key Capabilities

- **Intelligent Term Extraction**: Multi-source extraction with NLP processing
- **Multiple Weighting Methods**: Frequency, TF-IDF, logarithmic normalization
- **Professional Styling**: Scientific, colorful, academic, and tech styles
- **Interactive Visualizations**: Plotly-based interactive word clouds
- **Comparative Analysis**: Side-by-side comparisons across sources and time
- **Dynamic Updates**: Incremental updates with new publications
- **Temporal Evolution**: Analyze term trends over time with animations

---

## Features

### 1. Term Extraction and Processing

**Multi-source extraction**:
- Abstracts
- Keywords
- Titles
- Any text field

**Advanced NLP processing**:
- **Tokenization**: Break text into terms
- **Normalization**: Lowercase, lemmatization
- **Stopword removal**: Standard + domain-specific stopwords
- **POS filtering**: Keep only nouns and adjectives
- **N-gram extraction**: 1-3 word phrases

### 2. Term Weighting Methods

**Available methods**:

1. **Frequency**: Simple occurrence count
   - `weight = count`

2. **Log Frequency**: Logarithmic normalization
   - `weight = log(count + 1)`
   - Reduces dominance of very frequent terms

3. **Normalized**: Min-max normalization
   - `weight = (count - min) / (max - min)`
   - Scales all weights to [0, 1]

4. **TF-IDF**: Term Frequency-Inverse Document Frequency
   - `weight = tf * log(N / df)`
   - Emphasizes terms that are frequent but not ubiquitous

### 3. Visual Styles

**Scientific** (default):
- White background
- Blue color palette
- Professional appearance
- Suitable for academic publications

**Colorful**:
- Rainbow color scheme
- Vibrant and eye-catching
- Good for presentations

**Academic**:
- Beige background
- Sepia/brown tones
- Classic, vintage feel

**Tech**:
- Black background
- Plasma colors (purple/pink/yellow)
- Modern, futuristic look

### 4. Visualization Types

#### Static Word Clouds
- High-resolution PNG (300+ DPI)
- Customizable size and style
- Print-ready quality

#### Interactive Word Clouds
- Plotly-based HTML
- Hover to show exact weights
- Zoom and pan
- Exportable to image

#### Comparative Visualizations
- Multiple word clouds side-by-side
- Abstracts vs Keywords
- Year-by-year comparison
- Cluster-based comparison

#### Temporal Evolution
- Sequence of word clouds over time
- GIF animation
- Emerging vs declining terms analysis

---

## Installation

### Basic Requirements

```bash
# Core dependencies
pip install pandas numpy matplotlib

# NLP libraries
pip install nltk spacy
python -m spacy download en_core_web_sm

# Word cloud generation
pip install wordcloud

# Interactive visualizations
pip install plotly

# Image processing (for GIF animation)
pip install Pillow
```

### Installation from requirements.txt

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### NLTK Data

On first run, NLTK stopwords will be automatically downloaded. If needed, download manually:

```python
import nltk
nltk.download('stopwords')
```

---

## Quick Start

### Basic Usage

```python
from src.visualization import DynamicWordCloud

# Initialize with unified data
wc = DynamicWordCloud('data/processed/unified_data.csv')

# Extract and process terms
terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])

# Calculate weights
weights = wc.calculate_term_weights(terms, method='tfidf')

# Generate word cloud
wc.generate_wordcloud(
    weights,
    output_path='output/wordcloud.png',
    style='scientific'
)
```

### Complete Workflow

```python
from src.visualization import DynamicWordCloud
from pathlib import Path

# Initialize
wc = DynamicWordCloud('data/processed/unified_data.csv')

# Extract terms
terms = wc.extract_and_process_terms(
    sources=['abstract', 'keywords'],
    ngram_range=(1, 3),
    max_terms=200
)

# Calculate weights
weights = wc.calculate_term_weights(terms, method='tfidf')

# Create output directory
output_dir = Path('output/wordclouds')
output_dir.mkdir(parents=True, exist_ok=True)

# Generate static word cloud
wc.generate_wordcloud(
    weights,
    output_path=str(output_dir / 'wordcloud_scientific.png'),
    style='scientific',
    dpi=300
)

# Generate interactive word cloud
wc.generate_interactive_wordcloud(
    weights,
    output_html=str(output_dir / 'wordcloud_interactive.html')
)

# Create comparative word clouds
wc.create_comparative_wordclouds(
    output_dir=str(output_dir / 'comparative')
)
```

---

## Detailed Usage

### Step 1: Data Preparation

The DynamicWordCloud expects a CSV file with:

- `abstract`: Publication abstracts
- `keywords`: Publication keywords
- `year`: Publication year (optional, for temporal analysis)
- Other metadata fields

Example data format:

```csv
id,title,abstract,keywords,year
pub_001,"AI Study","This paper presents...",deep learning; AI; neural networks",2023
pub_002,"ML Paper","Machine learning has...",machine learning; classification,2023
```

### Step 2: Term Extraction

```python
# Extract from multiple sources
terms = wc.extract_and_process_terms(
    sources=['abstract', 'keywords'],  # Which fields to extract from
    ngram_range=(1, 3),                # Extract 1-3 word phrases
    min_term_length=3,                 # Minimum characters per term
    max_terms=200                      # Limit total terms
)

# View extracted terms
print(f"Extracted {len(terms)} unique terms")
top_10 = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:10]
for term, freq in top_10:
    print(f"{term}: {freq}")
```

**Extraction process**:
1. Combines text from specified sources
2. Cleans text (removes URLs, special characters)
3. Tokenizes using spaCy (or fallback)
4. Filters by POS tags (nouns, adjectives)
5. Removes stopwords
6. Extracts n-grams
7. Counts frequencies

### Step 3: Weight Calculation

```python
# Try different weighting methods
freq_weights = wc.calculate_term_weights(terms, method='frequency')
log_weights = wc.calculate_term_weights(terms, method='log_frequency')
norm_weights = wc.calculate_term_weights(terms, method='normalized')
tfidf_weights = wc.calculate_term_weights(terms, method='tfidf')

# Compare top terms
print("Top 5 by TF-IDF:")
top_tfidf = sorted(tfidf_weights.items(), key=lambda x: x[1], reverse=True)[:5]
for term, weight in top_tfidf:
    print(f"{term}: {weight:.4f}")
```

### Step 4: Generate Visualizations

#### Static Word Cloud

```python
wc.generate_wordcloud(
    term_weights=weights,
    output_path='output/wordcloud.png',
    style='scientific',      # Visual style
    max_words=150,          # Maximum words to display
    width=1600,             # Image width (pixels)
    height=1000,            # Image height (pixels)
    dpi=300                 # Resolution (for print)
)
```

#### Interactive Word Cloud

```python
wc.generate_interactive_wordcloud(
    term_weights=weights,
    output_html='output/wordcloud_interactive.html',
    max_words=100
)
```

Features:
- Hover to see exact weight
- Zoom and pan
- Color scale shows weight
- Exportable to PNG

#### Comparative Word Clouds

```python
wc.create_comparative_wordclouds(
    output_dir='output/comparative',
    style='scientific',
    dpi=300
)
```

Creates:
- `wordcloud_abstracts.png`: From abstracts only
- `wordcloud_keywords.png`: From keywords only
- `wordcloud_combined.png`: From both sources
- `wordcloud_year_YYYY.png`: One per year (if data available)
- `wordcloud_comparison_grid.png`: Grid view of all

### Step 5: Dynamic Updates

```python
# Generate initial word cloud
terms = wc.extract_and_process_terms()
weights = wc.calculate_term_weights(terms)
wc.generate_wordcloud(weights, 'output/wordcloud_initial.png')

# Save weights for later
wc.save_term_weights('output/weights_initial.pkl', weights)

# Later, when new data arrives...
updated_weights = wc.update_wordcloud_incremental(
    new_data_path='data/new_publications.csv',
    previous_weights_path='output/weights_initial.pkl',
    output_path='output/wordcloud_updated.png'
)
```

### Step 6: Temporal Evolution

```python
wc.generate_wordcloud_evolution(
    output_dir='output/evolution',
    create_animation=True,
    style='scientific'
)
```

Creates:
- `evolution_YYYY.png`: One word cloud per year
- `wordcloud_evolution.gif`: Animated sequence
- `term_trends.json`: Emerging and declining terms

---

## API Reference

### DynamicWordCloud Class

#### Constructor

```python
DynamicWordCloud(unified_data_path: str)
```

**Parameters:**
- `unified_data_path`: Path to unified CSV data file

**Raises:**
- `FileNotFoundError`: If data file doesn't exist

#### Methods

##### extract_and_process_terms()

```python
extract_and_process_terms(
    sources: List[str] = ['abstract', 'keywords'],
    ngram_range: Tuple[int, int] = (1, 3),
    min_term_length: int = 3,
    max_terms: Optional[int] = None
) -> Dict[str, int]
```

Extract and process terms from text.

**Parameters:**
- `sources`: List of column names to extract from
- `ngram_range`: (min_n, max_n) for n-gram extraction
- `min_term_length`: Minimum character length
- `max_terms`: Maximum number of terms to return

**Returns:**
Dictionary mapping terms to frequency counts

##### calculate_term_weights()

```python
calculate_term_weights(
    term_frequencies: Optional[Dict[str, int]] = None,
    method: str = 'log_frequency'
) -> Dict[str, float]
```

Calculate term weights for visualization.

**Parameters:**
- `term_frequencies`: Dictionary of term frequencies
- `method`: One of 'frequency', 'log_frequency', 'normalized', 'tfidf'

**Returns:**
Dictionary mapping terms to weights

##### generate_wordcloud()

```python
generate_wordcloud(
    term_weights: Optional[Dict[str, float]] = None,
    output_path: str = 'wordcloud.png',
    style: str = 'scientific',
    max_words: int = 150,
    width: int = 1600,
    height: int = 1000,
    dpi: int = 300
)
```

Generate static word cloud.

**Parameters:**
- `term_weights`: Dictionary of term weights
- `output_path`: Path to save PNG
- `style`: One of 'scientific', 'colorful', 'academic', 'tech'
- `max_words`: Maximum words to display
- `width`, `height`: Image dimensions in pixels
- `dpi`: Resolution for saving

##### generate_interactive_wordcloud()

```python
generate_interactive_wordcloud(
    term_weights: Optional[Dict[str, float]] = None,
    output_html: str = 'wordcloud_interactive.html',
    max_words: int = 100
)
```

Generate interactive Plotly word cloud.

##### create_comparative_wordclouds()

```python
create_comparative_wordclouds(
    output_dir: str,
    style: str = 'scientific',
    dpi: int = 300
)
```

Generate multiple word clouds for comparison.

##### update_wordcloud_incremental()

```python
update_wordcloud_incremental(
    new_data_path: str,
    previous_weights_path: Optional[str] = None,
    output_path: str = 'wordcloud_updated.png',
    style: str = 'scientific'
) -> Dict[str, float]
```

Update word cloud with new data (dynamic feature).

**Returns:**
Combined term weights

##### generate_wordcloud_evolution()

```python
generate_wordcloud_evolution(
    output_dir: str,
    create_animation: bool = True,
    style: str = 'scientific'
)
```

Generate temporal evolution analysis with GIF animation.

##### save_term_weights() / load_term_weights()

```python
save_term_weights(output_path: str, term_weights: Optional[Dict] = None)
load_term_weights(input_path: str) -> Dict[str, float]
```

Save and load term weights for reuse.

---

## Examples

### Example 1: Basic Word Cloud

```python
from src.visualization import DynamicWordCloud

wc = DynamicWordCloud('data/processed/unified_data.csv')
terms = wc.extract_and_process_terms()
weights = wc.calculate_term_weights(terms)
wc.generate_wordcloud(weights, 'output/basic.png')
```

### Example 2: Compare Weighting Methods

```python
wc = DynamicWordCloud('data/processed/unified_data.csv')
terms = wc.extract_and_process_terms()

methods = ['frequency', 'log_frequency', 'normalized', 'tfidf']
for method in methods:
    weights = wc.calculate_term_weights(terms, method=method)
    wc.generate_wordcloud(
        weights,
        output_path=f'output/wordcloud_{method}.png',
        style='scientific'
    )
```

### Example 3: Interactive Analysis

```python
wc = DynamicWordCloud('data/processed/unified_data.csv')
terms = wc.extract_and_process_terms(max_terms=100)
weights = wc.calculate_term_weights(terms, method='tfidf')

wc.generate_interactive_wordcloud(
    weights,
    output_html='output/interactive.html'
)
# Open in browser for interactive exploration
```

### Example 4: Temporal Evolution

```python
wc = DynamicWordCloud('data/processed/unified_data.csv')

wc.generate_wordcloud_evolution(
    output_dir='output/evolution',
    create_animation=True,
    style='scientific'
)

# View emerging and declining terms
import json
with open('output/evolution/term_trends.json') as f:
    trends = json.load(f)

print("Emerging terms:")
for item in trends['emerging_terms'][:10]:
    print(f"  {item['term']}: +{item['growth']:.2f}")
```

---

## Troubleshooting

### Issue: spaCy model not found

**Error:**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: NLTK stopwords not found

**Error:**
```
LookupError: Resource stopwords not found
```

**Solution:**
The module auto-downloads stopwords. If it fails:
```python
import nltk
nltk.download('stopwords')
```

### Issue: Word cloud is too crowded

**Solutions:**

1. Reduce max_words:
   ```python
   wc.generate_wordcloud(weights, max_words=75)
   ```

2. Use TF-IDF weighting to emphasize distinctive terms:
   ```python
   weights = wc.calculate_term_weights(terms, method='tfidf')
   ```

3. Filter by minimum weight:
   ```python
   filtered_weights = {t: w for t, w in weights.items() if w > threshold}
   ```

### Issue: Too many generic terms

**Solutions:**

1. Add domain-specific stopwords:
   ```python
   wc._domain_stopwords.update(['term1', 'term2', 'term3'])
   ```

2. Use TF-IDF instead of frequency:
   ```python
   weights = wc.calculate_term_weights(terms, method='tfidf')
   ```

3. Extract only from abstracts (more specific than keywords):
   ```python
   terms = wc.extract_and_process_terms(sources=['abstract'])
   ```

---

## Advanced Topics

### Custom Stopwords

```python
# Add custom stopwords
wc = DynamicWordCloud('data/processed/unified_data.csv')
wc._domain_stopwords.update([
    'custom_term1',
    'custom_term2',
    'generic_word'
])

# Re-load stopwords
wc._stopwords.update(wc._domain_stopwords)
```

### Custom Color Schemes

```python
# Modify style configuration
def custom_style_config(style):
    if style == 'custom':
        return {
            'background': '#f0f0f0',
            'colormap': 'viridis',
            'mode': 'RGB'
        }
    return wc._get_style_config(style)

# Apply custom style
wc._get_style_config = custom_style_config
```

### N-gram Analysis

```python
# Extract only bigrams and trigrams
terms = wc.extract_and_process_terms(
    sources=['abstract'],
    ngram_range=(2, 3),  # Only 2-3 word phrases
    min_term_length=10   # Longer minimum
)

# Filter out single words
multi_word_terms = {t: f for t, f in terms.items() if ' ' in t}
```

### Batch Processing

```python
# Process multiple datasets
datasets = [
    'data/2021_publications.csv',
    'data/2022_publications.csv',
    'data/2023_publications.csv'
]

for i, dataset in enumerate(datasets):
    wc = DynamicWordCloud(dataset)
    terms = wc.extract_and_process_terms()
    weights = wc.calculate_term_weights(terms)
    wc.generate_wordcloud(
        weights,
        output_path=f'output/wordcloud_{i+1}.png'
    )
```

### Incremental Update Pipeline

```python
# Initial setup
wc = DynamicWordCloud('data/initial_data.csv')
terms = wc.extract_and_process_terms()
weights = wc.calculate_term_weights(terms)
wc.save_term_weights('weights_current.pkl', weights)
wc.generate_wordcloud(weights, 'wordcloud_current.png')

# Weekly update function
def update_weekly(new_data_csv):
    wc = DynamicWordCloud('data/initial_data.csv')
    updated_weights = wc.update_wordcloud_incremental(
        new_data_path=new_data_csv,
        previous_weights_path='weights_current.pkl',
        output_path='wordcloud_current.png'
    )
    wc.save_term_weights('weights_current.pkl', updated_weights)
    print("Word cloud updated!")

# Use in automated pipeline
# update_weekly('data/this_week_publications.csv')
```

### Integration with Reports

```python
from pathlib import Path

def generate_wordcloud_report(data_path, output_dir):
    """Generate complete word cloud report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wc = DynamicWordCloud(data_path)

    # Extract and analyze
    terms = wc.extract_and_process_terms(max_terms=200)
    weights = wc.calculate_term_weights(terms, method='tfidf')

    # Generate visualizations
    wc.generate_wordcloud(
        weights,
        output_path=str(output_dir / 'wordcloud_main.png'),
        style='scientific',
        dpi=300
    )

    wc.generate_interactive_wordcloud(
        weights,
        output_html=str(output_dir / 'wordcloud_interactive.html')
    )

    wc.create_comparative_wordclouds(
        output_dir=str(output_dir / 'comparative')
    )

    if 'year' in wc.df.columns:
        wc.generate_wordcloud_evolution(
            output_dir=str(output_dir / 'evolution'),
            create_animation=True
        )

    print(f"Report generated in: {output_dir}")

# Use in pipeline
generate_wordcloud_report(
    'data/processed/unified_data.csv',
    'output/reports/wordcloud'
)
```

---

## Best Practices

1. **Data Quality**: Clean and preprocess data before analysis
2. **Weighting**: Use TF-IDF for most analyses to emphasize distinctive terms
3. **Stopwords**: Customize stopword list for your domain
4. **Resolution**: Use 300 DPI for print, 150 DPI for web
5. **Max Words**: 100-150 words for readability
6. **N-grams**: Include bigrams and trigrams for context
7. **Updates**: Save weights for incremental updates
8. **Validation**: Review top terms to ensure they make sense

---

## Performance Tips

1. **Limit terms**: Use `max_terms` parameter for large datasets
2. **Disable NER**: Set `wc._nlp = None` for faster processing
3. **Cache results**: Save and reuse term weights
4. **Batch processing**: Process multiple datasets in parallel
5. **Lower DPI**: Use 150 DPI for draft versions

---

## Citation

If you use this module in academic work, please cite:

```
Bibliometric Analysis System - Dynamic Word Cloud Module
Author: [Your Name/Organization]
Year: 2024
URL: [Repository URL]
```

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [Repository Issues]
- Documentation: [Full Documentation]
- Examples: See `examples/dynamic_wordcloud_demo.py`

---

## License

[Your License Here]

---

**Last Updated**: October 2024
**Version**: 1.0.0
