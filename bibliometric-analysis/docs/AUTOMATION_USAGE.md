# 🤖 Automation Pipeline - Usage Guide

## Overview

The Automation Pipeline provides a complete automated workflow for downloading and unifying bibliographic data from multiple academic databases.

## Features

✅ **Automated Web Scraping**
- ACM Digital Library (BibTeX format)
- ScienceDirect/Elsevier (RIS format)
- Selenium-based browser automation
- Robust error handling with exponential backoff

✅ **Intelligent Deduplication**
- Levenshtein similarity matching (threshold: 90%)
- DOI-based exact matching
- Cross-source duplicate detection

✅ **Data Unification**
- Normalized output format
- Combined CSV with all unique records
- Duplicate tracking log

✅ **Comprehensive Reporting**
- Execution statistics
- Download metrics
- Error logging

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Automation Pipeline

```bash
# Basic usage (default query: "generative artificial intelligence")
python automation_pipeline.py

# Custom query
python automation_pipeline.py --query "machine learning healthcare"

# Limit results per source
python automation_pipeline.py --max-results 50

# Run in headless mode (no browser UI)
python automation_pipeline.py --headless
```

---

## Output Files

### Unified Data (`data/processed/unified_data_YYYYMMDD_HHMMSS.csv`)

**Columns:**
- `id` - Unique identifier
- `title` - Publication title
- `authors` - Authors (semicolon-separated)
- `year` - Publication year
- `abstract` - Abstract text
- `keywords` - Keywords (semicolon-separated)
- `doi` - Digital Object Identifier
- `source` - Data source (ACM/ScienceDirect)
- `publication_type` - Type of publication
- `journal_conference` - Venue name
- `url` - Publication URL
- `publisher` - Publisher name
- `volume` - Volume number
- `number` - Issue number
- `pages` - Page range

### Duplicates Log (`data/duplicates/duplicates_log_YYYYMMDD_HHMMSS.csv`)

**Columns:**
- `original_index` - Index of original record
- `duplicate_index` - Index of duplicate record
- `original_title` - Title of original
- `duplicate_title` - Title of duplicate
- `original_source` - Source of original
- `duplicate_source` - Source of duplicate
- `similarity_score` - Similarity score (0.0-1.0)
- `detection_method` - Method used (title_similarity/exact_doi)

### Execution Report (`outputs/reports/automation_report_YYYYMMDD_HHMMSS.txt`)

Contains:
- Query information
- Execution time
- Download statistics
- Deduplication results
- Error/warning details

---

## Module Details

### 1. Base Scraper (`src/scrapers/base_scraper.py`)

Abstract base class providing:
- Selenium WebDriver management
- Retry logic with exponential backoff
- Download handling
- Authentication support

**Key Methods:**
```python
login(username, password) -> bool
search(query, filters) -> int
download_results(format, max_results) -> Path
parse_file(filepath) -> List[Dict]
```

### 2. ACM Scraper (`src/scrapers/acm_scraper.py`)

**Features:**
- ACM Digital Library integration
- BibTeX export
- Institutional authentication
- Advanced search filters

**Configuration:**
```yaml
# config/config.yaml
sources:
  acm:
    enabled: true
    max_results: 1000
    base_url: "https://dl.acm.org"
```

**Environment Variables:**
```bash
# config/.env
ACM_USERNAME=your_username
ACM_PASSWORD=your_password
```

### 3. ScienceDirect Scraper (`src/scrapers/sciencedirect_scraper.py`)

**Features:**
- ScienceDirect/Elsevier integration
- RIS export
- Date range filters
- Content type filtering

**Configuration:**
```yaml
# config/config.yaml
sources:
  sciencedirect:
    enabled: true
    max_results: 1000
    base_url: "https://www.sciencedirect.com"
```

**Environment Variables:**
```bash
# config/.env
SCIENCEDIRECT_USERNAME=your_username
SCIENCEDIRECT_PASSWORD=your_password
```

### 4. Data Unifier (`src/preprocessing/data_unifier.py`)

**Features:**
- Multi-source data loading
- Field normalization
- Levenshtein-based duplicate detection
- Information merging from duplicates

**Configuration:**
```yaml
# config/config.yaml
deduplication:
  enabled: true
  thresholds:
    title_similarity: 0.90
  strategy:
    exact_doi_priority: true
```

**Usage:**
```python
from src.preprocessing.data_unifier import DataUnifier

unifier = DataUnifier(config)
stats = unifier.unify([acm_records, sd_records])
```

---

## Advanced Usage

### Custom Filters

```python
from src.scrapers.acm_scraper import ACMScraper

scraper = ACMScraper(config)

# Search with filters
filters = {
    'year_start': 2020,
    'year_end': 2024,
    'content_type': 'research-article'
}

scraper.search("generative AI", filters=filters)
```

### Context Manager Pattern

```python
with ACMScraper(config) as scraper:
    records = scraper.scrape("machine learning")
    # Browser automatically closed when done
```

### Manual Workflow

```python
from src.scrapers.acm_scraper import ACMScraper
from src.scrapers.sciencedirect_scraper import ScienceDirectScraper
from src.preprocessing.data_unifier import DataUnifier

# 1. Scrape ACM
acm_scraper = ACMScraper(config)
acm_records = acm_scraper.scrape("AI ethics")

# 2. Scrape ScienceDirect
sd_scraper = ScienceDirectScraper(config)
sd_records = sd_scraper.scrape("AI ethics")

# 3. Unify
unifier = DataUnifier(config)
stats = unifier.unify([acm_records, sd_records])

print(f"Unified {stats['clean_count']} unique records")
```

---

## Troubleshooting

### ChromeDriver Issues

**Error:** "ChromeDriver not found"

**Solution:**
```bash
pip install --upgrade webdriver-manager
```

### Download Timeouts

**Error:** "Download timeout - file not found"

**Solution:**
Increase timeout in configuration:
```yaml
scraping:
  timeouts:
    download: 120  # Increase to 120 seconds
```

### Login Failures

**Error:** "Login failed"

**Solution:**
1. Verify credentials in `.env`
2. Check institutional access
3. Try without login (if open access available)

### Memory Issues

**Error:** "MemoryError during unification"

**Solution:**
Process in smaller batches:
```bash
python automation_pipeline.py --max-results 50
```

---

## Logging

Logs are saved to:
- `logs/automation_pipeline_YYYYMMDD_HHMMSS.log` - Main pipeline log
- `logs/acm_scraper.log` - ACM scraper log
- `logs/sciencedirect_scraper.log` - ScienceDirect scraper log
- `logs/data_unifier.log` - Unification log

**Log Levels:**
- `DEBUG` - Detailed information
- `INFO` - General progress
- `WARNING` - Non-critical issues
- `ERROR` - Critical failures

**View logs:**
```bash
tail -f logs/automation_pipeline_*.log
```

---

## Performance Tips

1. **Use Headless Mode**
   ```bash
   python automation_pipeline.py --headless
   ```

2. **Limit Results**
   ```bash
   python automation_pipeline.py --max-results 100
   ```

3. **Run Scrapers Separately**
   ```python
   # Run ACM only
   acm_records = pipeline.run_acm_scraper()
   ```

4. **Disable Unnecessary Sources**
   ```yaml
   # config/config.yaml
   sources:
     acm:
       enabled: true
     sciencedirect:
       enabled: false  # Disable if not needed
   ```

---

## Example Output

```
╔═══════════════════════════════════════════════════════════════════╗
║              AUTOMATION PIPELINE EXECUTION REPORT                 ║
╚═══════════════════════════════════════════════════════════════════╝

📋 Query Information:
   Search Query: "generative artificial intelligence"
   Max Results per Source: 100

⏱️  Execution Time:
   Start: 2025-01-15 10:30:00
   End:   2025-01-15 10:45:30
   Duration: 930.00s (15.50 minutes)

📊 Download Statistics:
   ACM Digital Library: 98 records
   ScienceDirect:       102 records
   Total Downloaded:    200 records

🔄 Deduplication Results:
   Duplicates Found:    15 records
   Final Unique Records: 185 records
   Deduplication Rate:  7.50%

📁 Output Files:
   Unified Data: data/processed/unified_data_20250115_104530.csv
   Duplicates Log: data/duplicates/duplicates_log_20250115_104530.csv

✓ No warnings
✓ No errors

═══════════════════════════════════════════════════════════════════
Status: ✓ SUCCESS
═══════════════════════════════════════════════════════════════════
```

---

## Next Steps

After running the automation pipeline:

1. **Review Unified Data**
   ```bash
   head data/processed/unified_data_*.csv
   ```

2. **Check Duplicates**
   ```bash
   cat data/duplicates/duplicates_log_*.csv
   ```

3. **Proceed to Analysis**
   - Text preprocessing
   - Clustering
   - Visualization

---

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review `docs/SETUP.md` for installation help
- Consult `README.md` for general documentation
