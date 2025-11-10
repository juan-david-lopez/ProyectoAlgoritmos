"""
Dynamic Word Cloud Visualization Module

Generates professional, dynamic, and automatically updatable word clouds
for analyzing scientific production.

Features:
- Extract and process terms from abstracts and keywords
- Multiple weighting schemes (frequency, TF-IDF, logarithmic)
- Static word clouds with professional styling
- Interactive word clouds with Plotly
- Comparative visualizations
- Incremental updates (dynamic)
- Temporal evolution analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import re
from collections import Counter, defaultdict
import warnings
from loguru import logger
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')


class DynamicWordCloud:
    """
    Generates dynamic and automatically updatable word clouds.

    This class provides comprehensive word cloud visualization capabilities
    including static, interactive, comparative, and temporal analysis.
    """

    def __init__(self, unified_data_path: str):
        """
        Initialize dynamic word cloud generator.

        Args:
            unified_data_path: Path to unified CSV data file
        """
        self.unified_data_path = Path(unified_data_path)

        # Validate file exists
        if not self.unified_data_path.exists():
            raise FileNotFoundError(f"Unified data file not found: {unified_data_path}")

        # Load data
        logger.info(f"Loading data from: {unified_data_path}")
        self.df = pd.read_csv(self.unified_data_path, encoding='utf-8')
        logger.info(f"Loaded {len(self.df)} records")

        # Initialize NLP components
        self._nlp = None
        self._stopwords = set()
        self._load_nlp_components()

        # Cache for term frequencies
        self._term_cache = {}

        # Domain-specific stopwords
        self._domain_stopwords = {
            'study', 'paper', 'research', 'analysis', 'approach', 'method',
            'result', 'conclusion', 'introduction', 'abstract', 'article',
            'journal', 'conference', 'proceedings', 'ieee', 'acm', 'publisher',
            'et', 'al', 'fig', 'table', 'section', 'chapter', 'volume',
            'issue', 'pp', 'doi', 'isbn', 'issn', 'www', 'http', 'https',
            'based', 'using', 'used', 'propose', 'proposed', 'present',
            'presented', 'show', 'showed', 'shown', 'provide', 'provided'
        }

        logger.success("DynamicWordCloud initialized successfully")

    def _load_nlp_components(self):
        """Load NLP components (spaCy, NLTK)."""
        # Try loading spaCy
        try:
            import spacy
            try:
                self._nlp = spacy.load('en_core_web_sm')
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not installed")

        # Load NLTK stopwords
        try:
            import nltk
            from nltk.corpus import stopwords

            try:
                self._stopwords = set(stopwords.words('english'))
                logger.info("Loaded NLTK stopwords")
            except LookupError:
                logger.warning("NLTK stopwords not found. Downloading...")
                try:
                    nltk.download('stopwords', quiet=True)
                    self._stopwords = set(stopwords.words('english'))
                    logger.info("Downloaded and loaded NLTK stopwords")
                except:
                    logger.warning("Could not download NLTK stopwords")
                    self._stopwords = set()
        except ImportError:
            logger.warning("NLTK not installed")

        # Add domain-specific stopwords
        self._stopwords.update(self._domain_stopwords)

    def extract_and_process_terms(
        self,
        sources: List[str] = ['abstract', 'keywords'],
        ngram_range: Tuple[int, int] = (1, 3),
        min_term_length: int = 3,
        max_terms: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Extract and process terms from multiple sources.

        Preprocessing steps:
        1. Combine abstracts and keywords
        2. Tokenization
        3. Normalization (lowercase, lemmatization)
        4. Remove stopwords (standard + domain-specific)
        5. Filter by POS tags (keep NOUN, ADJ)
        6. Extract n-grams (1-3 words)

        Args:
            sources: List of column names to extract from
            ngram_range: Tuple of (min_n, max_n) for n-grams
            min_term_length: Minimum character length for terms
            max_terms: Maximum number of terms to return

        Returns:
            Dictionary mapping terms to frequency counts
        """
        logger.info("Extracting and processing terms")

        all_terms = []

        # Process each document
        for idx, row in self.df.iterrows():
            # Combine text from sources
            text_parts = []

            for source in sources:
                if source in row and pd.notna(row[source]):
                    text = str(row[source])
                    text_parts.append(text)

            if not text_parts:
                continue

            combined_text = ' '.join(text_parts)

            # Extract terms from this document
            terms = self._extract_terms_from_text(
                combined_text,
                ngram_range=ngram_range,
                min_term_length=min_term_length
            )

            all_terms.extend(terms)

        # Count frequencies
        term_frequencies = Counter(all_terms)

        # Limit to max_terms if specified
        if max_terms:
            term_frequencies = dict(term_frequencies.most_common(max_terms))
        else:
            term_frequencies = dict(term_frequencies)

        logger.info(f"Extracted {len(term_frequencies)} unique terms")

        # Cache results
        self._term_cache = term_frequencies

        return term_frequencies

    def _extract_terms_from_text(
        self,
        text: str,
        ngram_range: Tuple[int, int] = (1, 3),
        min_term_length: int = 3
    ) -> List[str]:
        """
        Extract terms from a single text.

        Args:
            text: Input text
            ngram_range: N-gram range
            min_term_length: Minimum term length

        Returns:
            List of extracted terms
        """
        terms = []

        # Clean text
        text = self._clean_text(text)

        # Use spaCy if available for better processing
        if self._nlp:
            doc = self._nlp(text)

            # Extract unigrams with POS filtering
            for token in doc:
                if self._is_valid_token(token):
                    lemma = token.lemma_.lower()
                    if len(lemma) >= min_term_length and lemma not in self._stopwords:
                        terms.append(lemma)

            # Extract n-grams (2-3 words)
            for n in range(2, ngram_range[1] + 1):
                for i in range(len(doc) - n + 1):
                    ngram_tokens = doc[i:i+n]

                    # Check if all tokens are valid
                    if all(self._is_valid_token(t) for t in ngram_tokens):
                        ngram = ' '.join(t.lemma_.lower() for t in ngram_tokens)

                        # Filter stopwords from n-gram
                        ngram_words = ngram.split()
                        if not any(w in self._stopwords for w in ngram_words):
                            if len(ngram) >= min_term_length:
                                terms.append(ngram)

        else:
            # Fallback to simple tokenization
            terms = self._simple_tokenization(text, min_term_length)

        return terms

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _is_valid_token(self, token) -> bool:
        """Check if token is valid for extraction."""
        # Keep nouns and adjectives
        if token.pos_ not in ['NOUN', 'PROPN', 'ADJ']:
            return False

        # Skip stopwords
        if token.is_stop or token.lemma_.lower() in self._stopwords:
            return False

        # Skip punctuation
        if token.is_punct:
            return False

        # Skip very short tokens
        if len(token.text) < 2:
            return False

        return True

    def _simple_tokenization(self, text: str, min_term_length: int = 3) -> List[str]:
        """Simple tokenization fallback."""
        # Lowercase and split
        words = text.lower().split()

        # Filter
        terms = [
            w for w in words
            if len(w) >= min_term_length and w not in self._stopwords
        ]

        return terms

    def calculate_term_weights(
        self,
        term_frequencies: Optional[Dict[str, int]] = None,
        method: str = 'log_frequency'
    ) -> Dict[str, float]:
        """
        Calculate weights for visualization.

        Methods:
        1. 'frequency': Simple frequency count
        2. 'tfidf': TF-IDF weighting (if corpus available)
        3. 'log_frequency': Logarithmic normalization: weight = log(freq + 1)
        4. 'normalized': Min-max normalization to [0, 1]

        Args:
            term_frequencies: Dictionary of term frequencies
            method: Weighting method

        Returns:
            Dictionary mapping terms to normalized weights
        """
        logger.info(f"Calculating term weights using method: {method}")

        if term_frequencies is None:
            if not self._term_cache:
                term_frequencies = self.extract_and_process_terms()
            else:
                term_frequencies = self._term_cache

        if method == 'frequency':
            weights = term_frequencies.copy()

        elif method == 'log_frequency':
            weights = {
                term: np.log(freq + 1)
                for term, freq in term_frequencies.items()
            }

        elif method == 'normalized':
            freqs = np.array(list(term_frequencies.values()))
            min_freq, max_freq = freqs.min(), freqs.max()

            weights = {
                term: (freq - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0.5
                for term, freq in term_frequencies.items()
            }

        elif method == 'tfidf':
            weights = self._calculate_tfidf_weights(term_frequencies)

        else:
            logger.warning(f"Unknown method '{method}', using 'log_frequency'")
            weights = {
                term: np.log(freq + 1)
                for term, freq in term_frequencies.items()
            }

        logger.info(f"Calculated weights for {len(weights)} terms")

        return weights

    def _calculate_tfidf_weights(self, term_frequencies: Dict[str, int]) -> Dict[str, float]:
        """Calculate TF-IDF weights."""
        # Calculate document frequencies
        doc_frequencies = defaultdict(int)

        for idx, row in self.df.iterrows():
            # Get document text
            text = ''
            for col in ['abstract', 'keywords', 'title']:
                if col in row and pd.notna(row[col]):
                    text += ' ' + str(row[col])

            text = text.lower()

            # Count which terms appear in this document
            for term in term_frequencies.keys():
                if term in text:
                    doc_frequencies[term] += 1

        # Calculate TF-IDF
        num_docs = len(self.df)
        tfidf_weights = {}

        for term, tf in term_frequencies.items():
            df = doc_frequencies.get(term, 1)
            idf = np.log(num_docs / df)
            tfidf_weights[term] = tf * idf

        return tfidf_weights

    def generate_wordcloud(
        self,
        term_weights: Optional[Dict[str, float]] = None,
        output_path: str = 'wordcloud.png',
        style: str = 'scientific',
        max_words: int = 150,
        width: int = 1600,
        height: int = 1000,
        dpi: int = 300
    ):
        """
        Generate styled word cloud.

        Styles:
        - 'scientific': Professional white background, blue/grey palette
        - 'colorful': Vibrant color palette
        - 'academic': Sepia/vintage style
        - 'tech': Neon/futuristic style

        Args:
            term_weights: Dictionary of term weights
            output_path: Path to save image
            style: Visual style
            max_words: Maximum number of words
            width: Image width in pixels
            height: Image height in pixels
            dpi: Resolution for saving
        """
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("wordcloud package not installed. Install with: pip install wordcloud")
            return

        logger.info(f"Generating word cloud with style: {style}")

        if term_weights is None:
            if not self._term_cache:
                self.extract_and_process_terms()
            term_weights = self.calculate_term_weights()

        # Configure style
        wc_config = self._get_style_config(style)

        # Create word cloud
        wc = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color=wc_config['background'],
            colormap=wc_config['colormap'],
            font_path=wc_config.get('font_path'),
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=100,
            prefer_horizontal=0.7,
            collocations=False,
            mode=wc_config.get('mode', 'RGB')
        )

        # Generate from frequencies
        wc.generate_from_frequencies(term_weights)

        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=dpi)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_file), dpi=dpi, bbox_inches='tight', facecolor=wc_config['background'])
        plt.close()

        logger.success(f"Word cloud saved to: {output_file}")

    def _get_style_config(self, style: str) -> Dict[str, Any]:
        """Get configuration for word cloud style."""
        styles = {
            'scientific': {
                'background': 'white',
                'colormap': 'Blues',
                'mode': 'RGB'
            },
            'colorful': {
                'background': 'white',
                'colormap': 'rainbow',
                'mode': 'RGB'
            },
            'academic': {
                'background': '#f5f5dc',  # Beige
                'colormap': 'YlOrBr',
                'mode': 'RGB'
            },
            'tech': {
                'background': 'black',
                'colormap': 'plasma',
                'mode': 'RGB'
            }
        }

        return styles.get(style, styles['scientific'])

    def generate_interactive_wordcloud(
        self,
        term_weights: Optional[Dict[str, float]] = None,
        output_html: str = 'wordcloud_interactive.html',
        max_words: int = 100
    ):
        """
        Generate interactive word cloud with Plotly.

        Features:
        - Hover to show exact frequency
        - Click for filtering (future enhancement)
        - Zoom and pan
        - Exportable to image

        Args:
            term_weights: Dictionary of term weights
            output_html: Path to save HTML file
            max_words: Maximum number of words to display
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Install with: pip install plotly")
            return

        logger.info("Generating interactive word cloud")

        if term_weights is None:
            if not self._term_cache:
                self.extract_and_process_terms()
            term_weights = self.calculate_term_weights()

        # Sort by weight and take top N
        sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:max_words]

        # Create scatter plot with text
        terms = [t[0] for t in sorted_terms]
        weights = [t[1] for t in sorted_terms]

        # Normalize weights for size
        min_weight, max_weight = min(weights), max(weights)
        sizes = [
            10 + (w - min_weight) / (max_weight - min_weight) * 90
            for w in weights
        ]

        # Generate random positions (can be improved with better layout algorithm)
        np.random.seed(42)
        n = len(terms)
        x = np.random.randn(n) * 50
        y = np.random.randn(n) * 50

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='text',
            text=terms,
            textfont=dict(
                size=sizes,
                color=weights,
                colorscale='Blues'
            ),
            hovertemplate='<b>%{text}</b><br>Weight: %{marker.color:.2f}<extra></extra>',
            marker=dict(
                color=weights,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title='Weight')
            )
        ))

        fig.update_layout(
            title='Interactive Word Cloud',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            hovermode='closest',
            plot_bgcolor='white',
            width=1200,
            height=800
        )

        # Save
        output_file = Path(output_html)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        logger.success(f"Interactive word cloud saved to: {output_file}")

    def create_comparative_wordclouds(
        self,
        output_dir: str,
        style: str = 'scientific',
        dpi: int = 300
    ):
        """
        Generate multiple word clouds for comparison.

        Creates:
        1. Abstracts only
        2. Keywords only
        3. Combined
        4. By year (if sufficient data)
        5. By cluster (if available)

        Args:
            output_dir: Directory to save outputs
            style: Visual style for all word clouds
            dpi: Resolution
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not installed")
            return

        logger.info("Creating comparative word clouds")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Abstracts only
        logger.info("Generating word cloud from abstracts")
        abstract_terms = self.extract_and_process_terms(sources=['abstract'])
        abstract_weights = self.calculate_term_weights(abstract_terms, method='log_frequency')
        self.generate_wordcloud(
            abstract_weights,
            output_path=str(output_path / 'wordcloud_abstracts.png'),
            style=style,
            dpi=dpi
        )

        # 2. Keywords only
        logger.info("Generating word cloud from keywords")
        keyword_terms = self.extract_and_process_terms(sources=['keywords'])
        keyword_weights = self.calculate_term_weights(keyword_terms, method='log_frequency')
        self.generate_wordcloud(
            keyword_weights,
            output_path=str(output_path / 'wordcloud_keywords.png'),
            style=style,
            dpi=dpi
        )

        # 3. Combined
        logger.info("Generating combined word cloud")
        combined_terms = self.extract_and_process_terms(sources=['abstract', 'keywords'])
        combined_weights = self.calculate_term_weights(combined_terms, method='log_frequency')
        self.generate_wordcloud(
            combined_weights,
            output_path=str(output_path / 'wordcloud_combined.png'),
            style=style,
            dpi=dpi
        )

        # 4. By year (if year column exists)
        if 'year' in self.df.columns:
            self._create_wordclouds_by_year(output_path, style, dpi)

        # 5. Create comparison grid
        self._create_comparison_grid(output_path)

        logger.success(f"Comparative word clouds saved to: {output_path}")

    def _create_wordclouds_by_year(self, output_dir: Path, style: str, dpi: int):
        """Create word clouds grouped by year."""
        years = self.df['year'].dropna().unique()

        # Only create if we have multiple years
        if len(years) < 2:
            logger.info("Not enough years for temporal comparison")
            return

        logger.info(f"Creating word clouds for {len(years)} years")

        for year in sorted(years):
            # Filter data for this year
            year_df = self.df[self.df['year'] == year]

            if len(year_df) < 3:  # Skip years with very few documents
                continue

            # Temporarily replace dataframe
            original_df = self.df
            self.df = year_df

            # Extract terms
            terms = self.extract_and_process_terms(sources=['abstract', 'keywords'])
            weights = self.calculate_term_weights(terms, method='log_frequency')

            # Generate word cloud
            self.generate_wordcloud(
                weights,
                output_path=str(output_dir / f'wordcloud_year_{int(year)}.png'),
                style=style,
                dpi=dpi
            )

            # Restore dataframe
            self.df = original_df

    def _create_comparison_grid(self, output_dir: Path):
        """Create a grid comparison of multiple word clouds."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.image import imread
        except ImportError:
            return

        # Find all word cloud images
        wc_files = sorted(output_dir.glob('wordcloud_*.png'))

        if len(wc_files) < 2:
            return

        # Create grid
        n_images = min(len(wc_files), 6)  # Max 6 for readability
        n_cols = 3
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_images > 1 else [axes]

        for idx, wc_file in enumerate(wc_files[:n_images]):
            img = imread(wc_file)
            axes[idx].imshow(img)
            axes[idx].set_title(wc_file.stem.replace('wordcloud_', '').replace('_', ' ').title())
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(str(output_dir / 'wordcloud_comparison_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logger.success("Comparison grid created")

    def update_wordcloud_incremental(
        self,
        new_data_path: str,
        previous_weights_path: Optional[str] = None,
        output_path: str = 'wordcloud_updated.png',
        style: str = 'scientific'
    ):
        """
        Update word cloud with new studies (dynamic update).

        Process:
        1. Load previous term weights
        2. Extract terms from new documents
        3. Combine and recalculate weights
        4. Regenerate word cloud

        Args:
            new_data_path: Path to new data CSV
            previous_weights_path: Path to saved previous weights (pickle)
            output_path: Path to save updated word cloud
            style: Visual style
        """
        logger.info("Updating word cloud incrementally")

        # Load previous weights if available
        previous_weights = {}
        if previous_weights_path and Path(previous_weights_path).exists():
            try:
                with open(previous_weights_path, 'rb') as f:
                    previous_weights = pickle.load(f)
                logger.info(f"Loaded {len(previous_weights)} previous term weights")
            except Exception as e:
                logger.warning(f"Could not load previous weights: {e}")

        # Load new data
        new_df = pd.read_csv(new_data_path, encoding='utf-8')
        logger.info(f"Loaded {len(new_df)} new records")

        # Temporarily replace dataframe
        original_df = self.df
        self.df = new_df

        # Extract terms from new data
        new_terms = self.extract_and_process_terms(sources=['abstract', 'keywords'])
        new_weights = self.calculate_term_weights(new_terms, method='log_frequency')

        # Restore original dataframe
        self.df = original_df

        # Combine weights
        combined_weights = previous_weights.copy()
        for term, weight in new_weights.items():
            combined_weights[term] = combined_weights.get(term, 0) + weight

        # Normalize combined weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {
                term: weight / total_weight * 100
                for term, weight in combined_weights.items()
            }

        # Generate updated word cloud
        self.generate_wordcloud(
            combined_weights,
            output_path=output_path,
            style=style
        )

        # Save updated weights
        weights_save_path = Path(output_path).parent / 'term_weights_updated.pkl'
        with open(weights_save_path, 'wb') as f:
            pickle.dump(combined_weights, f)

        logger.success(f"Updated word cloud saved to: {output_path}")
        logger.info(f"Updated weights saved to: {weights_save_path}")

        return combined_weights

    def generate_wordcloud_evolution(
        self,
        output_dir: str,
        create_animation: bool = True,
        style: str = 'scientific'
    ):
        """
        Generate sequence showing temporal evolution of word clouds.

        Creates:
        - One word cloud per year
        - GIF animation showing evolution
        - Identification of emerging vs declining terms

        Args:
            output_dir: Directory to save outputs
            create_animation: Whether to create GIF animation
            style: Visual style
        """
        logger.info("Generating word cloud evolution")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if year data exists
        if 'year' not in self.df.columns:
            logger.warning("Year column not found in data")
            return

        years = sorted(self.df['year'].dropna().unique())

        if len(years) < 2:
            logger.warning("Not enough years for evolution analysis")
            return

        logger.info(f"Creating evolution across {len(years)} years")

        # Store term weights by year
        yearly_weights = {}
        wordcloud_files = []

        # Generate word cloud for each year
        for year in years:
            year_df = self.df[self.df['year'] == year]

            if len(year_df) < 3:  # Skip years with few documents
                continue

            # Temporarily replace dataframe
            original_df = self.df
            self.df = year_df

            # Extract and process
            terms = self.extract_and_process_terms(sources=['abstract', 'keywords'])
            weights = self.calculate_term_weights(terms, method='log_frequency')

            yearly_weights[int(year)] = weights

            # Generate word cloud
            wc_file = output_path / f'evolution_{int(year)}.png'
            self.generate_wordcloud(
                weights,
                output_path=str(wc_file),
                style=style,
                dpi=150  # Lower DPI for animation
            )

            wordcloud_files.append(wc_file)

            # Restore dataframe
            self.df = original_df

        # Analyze emerging and declining terms
        self._analyze_term_trends(yearly_weights, output_path)

        # Create animation if requested
        if create_animation and len(wordcloud_files) > 1:
            self._create_evolution_animation(wordcloud_files, output_path)

        logger.success(f"Evolution analysis saved to: {output_path}")

    def _analyze_term_trends(self, yearly_weights: Dict[int, Dict[str, float]], output_dir: Path):
        """Analyze emerging and declining terms."""
        if len(yearly_weights) < 2:
            return

        years = sorted(yearly_weights.keys())
        first_year = years[0]
        last_year = years[-1]

        first_weights = yearly_weights[first_year]
        last_weights = yearly_weights[last_year]

        # Identify emerging terms (high in last year, low/absent in first year)
        emerging = {}
        for term, weight in last_weights.items():
            old_weight = first_weights.get(term, 0)
            if weight > old_weight:
                emerging[term] = weight - old_weight

        # Identify declining terms (high in first year, low/absent in last year)
        declining = {}
        for term, weight in first_weights.items():
            new_weight = last_weights.get(term, 0)
            if weight > new_weight:
                declining[term] = weight - new_weight

        # Sort and save top terms
        emerging_top = sorted(emerging.items(), key=lambda x: x[1], reverse=True)[:20]
        declining_top = sorted(declining.items(), key=lambda x: x[1], reverse=True)[:20]

        # Create report
        report = {
            'period': f'{first_year}-{last_year}',
            'emerging_terms': [{'term': t, 'growth': float(g)} for t, g in emerging_top],
            'declining_terms': [{'term': t, 'decline': float(d)} for t, d in declining_top]
        }

        # Save report
        report_path = output_dir / 'term_trends.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Term trends saved to: {report_path}")

    def _create_evolution_animation(self, image_files: List[Path], output_dir: Path):
        """Create GIF animation from word cloud sequence."""
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not installed. Cannot create animation")
            return

        logger.info("Creating evolution animation")

        # Load images
        images = []
        for img_file in image_files:
            img = Image.open(img_file)
            images.append(img)

        # Save as GIF
        gif_path = output_dir / 'wordcloud_evolution.gif'
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # 1 second per frame
            loop=0
        )

        logger.success(f"Animation saved to: {gif_path}")

    def save_term_weights(self, output_path: str, term_weights: Optional[Dict[str, float]] = None):
        """
        Save term weights for later use.

        Args:
            output_path: Path to save weights (pickle format)
            term_weights: Weights to save (uses cached if None)
        """
        if term_weights is None:
            term_weights = self._term_cache

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(term_weights, f)

        logger.success(f"Term weights saved to: {output_file}")

    def load_term_weights(self, input_path: str) -> Dict[str, float]:
        """
        Load previously saved term weights.

        Args:
            input_path: Path to saved weights

        Returns:
            Dictionary of term weights
        """
        with open(input_path, 'rb') as f:
            weights = pickle.load(f)

        logger.info(f"Loaded {len(weights)} term weights from: {input_path}")

        return weights


# Example usage
if __name__ == "__main__":
    logger.info("Dynamic Word Cloud Module - Example Usage")

    # This would normally be called with real unified data
    # wc = DynamicWordCloud('data/processed/unified_data.csv')
    #
    # # Extract and process terms
    # terms = wc.extract_and_process_terms(sources=['abstract', 'keywords'])
    # print(f"Extracted {len(terms)} unique terms")
    #
    # # Calculate weights
    # weights = wc.calculate_term_weights(terms, method='log_frequency')
    #
    # # Generate word clouds
    # wc.generate_wordcloud(weights, output_path='output/wordcloud_scientific.png', style='scientific')
    # wc.generate_wordcloud(weights, output_path='output/wordcloud_colorful.png', style='colorful')
    #
    # # Generate interactive word cloud
    # wc.generate_interactive_wordcloud(weights, output_html='output/wordcloud_interactive.html')
    #
    # # Create comparative word clouds
    # wc.create_comparative_wordclouds(output_dir='output/comparative')
    #
    # # Generate evolution analysis
    # wc.generate_wordcloud_evolution(output_dir='output/evolution', create_animation=True)

    logger.info("Example complete!")
