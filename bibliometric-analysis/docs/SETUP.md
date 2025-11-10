# Setup Instructions - Bibliometric Analysis

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional)

## Installation Steps

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd bibliometric-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note**: Installation may take 10-15 minutes due to deep learning libraries (PyTorch, Transformers).

### 3. Download NLP Models

After installing dependencies, download required NLP models:

```python
# Run this Python script
python -c "
import nltk
import spacy

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

print('NLTK downloads complete!')
print('Next, install spaCy language model:')
print('python -m spacy download es_core_news_sm')
print('python -m spacy download en_core_web_sm')
"

# Download Spanish spaCy model
python -m spacy download es_core_news_sm

# Download English spaCy model (optional)
python -m spacy download en_core_web_sm
```

### 4. Install wkhtmltopdf (for PDF export)

**pdfkit** requires wkhtmltopdf to be installed separately:

**Windows:**
- Download from: https://wkhtmltopdf.org/downloads.html
- Install and add to PATH

**macOS:**
```bash
brew install wkhtmltopdf
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install wkhtmltopdf
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template
cp config/.env.example config/.env

# Edit with your credentials
nano config/.env
```

Example `.env` file:
```
# API Keys (if available)
SCOPUS_API_KEY=your_scopus_api_key_here
WOS_API_KEY=your_wos_api_key_here

# Database credentials (if needed)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bibliometric_db
DB_USER=your_username
DB_PASSWORD=your_password

# Scraping settings
DOWNLOAD_DELAY=2
MAX_RETRIES=3
TIMEOUT=30
```

### 6. Verify Installation

```bash
# Run test script
python -c "
import pandas as pd
import numpy as np
import sklearn
import transformers
import streamlit
import plotly

print('✓ All core packages installed successfully!')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```

## Dependency Categories

### Core Dependencies (Required)
- **Data**: pandas, numpy, openpyxl
- **ML**: scikit-learn, scipy
- **NLP**: nltk, spacy, python-Levenshtein
- **Visualization**: matplotlib, seaborn, plotly

### Advanced Dependencies (Optional)
- **Deep Learning**: torch, transformers, sentence-transformers
- **Web Scraping**: selenium, webdriver-manager
- **Web App**: streamlit
- **Report Generation**: reportlab, pdfkit

## Troubleshooting

### Issue: PyTorch installation fails

**Solution**: Install CPU-only version explicitly:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Selenium WebDriver errors

**Solution**: Use webdriver-manager for automatic driver management:
```python
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())
```

### Issue: spaCy model not found

**Solution**: Download language model:
```bash
python -m spacy download es_core_news_sm
```

### Issue: Memory errors during installation

**Solution**: Install packages in batches:
```bash
# Install basic packages first
pip install pandas numpy scikit-learn matplotlib

# Then install heavy packages
pip install torch transformers sentence-transformers

# Finally install remaining packages
pip install -r requirements.txt
```

### Issue: Encoding errors on Windows

**Solution**: Set UTF-8 encoding:
```bash
# Add to environment variables
set PYTHONIOENCODING=utf-8
```

## System Requirements

### Minimum:
- RAM: 8 GB
- Storage: 5 GB free space
- CPU: 2 cores

### Recommended:
- RAM: 16 GB
- Storage: 10 GB free space
- CPU: 4+ cores
- GPU: Not required (CPU-only PyTorch)

## Next Steps

After successful installation:

1. Review `config/config.yaml` and adjust settings
2. Run the main script: `python main.py --help`
3. Check the documentation: `docs/`
4. Explore example notebooks: `notebooks/`

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs in `bibliometric_analysis.log`
3. Check package versions: `pip list`
4. Verify Python version: `python --version`

## License

See LICENSE file for details.
