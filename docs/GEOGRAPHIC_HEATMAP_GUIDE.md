# Geographic Heatmap Visualization Guide

Comprehensive guide for using the Geographic Heatmap visualization system to analyze the geographic distribution of scientific publications.

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

The Geographic Heatmap module provides professional, dynamic, and exportable visualizations to analyze scientific production from geographic perspectives. It extracts author affiliations, geocodes locations, and creates multiple types of visualizations suitable for reports, presentations, and publications.

### Key Capabilities

- **Affiliation Extraction**: Automatically extract country and institution information from author data using pattern matching and Named Entity Recognition (NER)
- **Geocoding**: Convert country names to geographic coordinates
- **Interactive Maps**: Generate interactive HTML maps using Folium and Plotly
- **Static Maps**: Create high-resolution static maps for PDF export
- **Statistics**: Comprehensive geographic statistics and analytics

---

## Features

### 1. Author Affiliation Extraction

Multiple strategies for robust extraction:

- **Pattern-based extraction**: Recognizes common patterns like "Author (Institution, Country)"
- **Named Entity Recognition**: Uses spaCy NLP for intelligent entity detection
- **Institution mapping**: Maps well-known universities to countries
- **Fallback mechanisms**: Ensures maximum data capture

### 2. Geographic Visualization Types

#### Interactive Maps (Folium)
- OpenStreetMap and CartoDB base layers
- Proportional circle markers
- Heatmap overlay
- Marker clustering for better visualization
- Interactive tooltips and popups
- Layer controls

#### Interactive Maps (Plotly)
- Scatter geo plots
- Color-coded by publication count
- Smooth animations
- Easy export to static images
- Dashboard integration ready

#### Static Maps
- High-resolution (300+ DPI) for print quality
- Professional cartographic projection (Robinson/Mercator)
- Suitable for PDF reports and publications
- Clean, publication-ready aesthetics

### 3. Statistical Analysis

- Top N countries by publication count
- Distribution by continent
- Publication density metrics
- Coverage analysis
- Temporal evolution (when time series data available)

---

## Installation

### Basic Requirements

```bash
# Install core dependencies
pip install pandas numpy matplotlib

# Install visualization libraries
pip install folium plotly

# Install NLP libraries (for affiliation extraction)
pip install spacy
python -m spacy download en_core_web_sm

# Install geographic libraries
pip install geopy pycountry

# Optional: For advanced cartographic maps
pip install cartopy

# Optional: For Plotly image export
pip install kaleido
```

### Installation from requirements.txt

```bash
pip install -r requirements.txt
```

### System Dependencies (for Cartopy)

Cartopy requires additional system libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get install libgeos-dev libproj-dev
```

**macOS (with Homebrew):**
```bash
brew install geos proj
```

**Windows:**
- Use conda: `conda install -c conda-forge cartopy`
- Or use pre-built wheels from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/)

---

## Quick Start

### Basic Usage

```python
from src.visualization.geographic_heatmap import GeographicHeatmap

# Initialize with unified data
geo_map = GeographicHeatmap('data/processed/unified_data.csv')

# Extract affiliations and geocode
geo_map.extract_author_affiliations()
geo_map.geocode_locations()

# Create interactive map
geo_map.create_interactive_map(output_html='output/map.html')

# Generate statistics
stats = geo_map.generate_geographic_statistics()
print(f"Publications from {stats['summary']['total_countries']} countries")
```

### Complete Workflow

```python
from src.visualization.geographic_heatmap import GeographicHeatmap
from pathlib import Path

# Initialize
geo_map = GeographicHeatmap('data/processed/unified_data.csv')

# Extract and analyze
geo_map.extract_author_affiliations()
geo_map.geocode_locations()

# Create all visualizations
output_dir = Path('output/geographic')
output_dir.mkdir(parents=True, exist_ok=True)

# Interactive maps
geo_map.create_interactive_map(
    output_html=str(output_dir / 'interactive_folium.html')
)
geo_map.create_plotly_map(
    output_html=str(output_dir / 'interactive_plotly.html')
)

# Static map for PDF
geo_map.create_static_map(
    output_png=str(output_dir / 'publication_map.png'),
    dpi=300
)

# Statistics
geo_map.save_statistics_report(
    str(output_dir / 'geographic_stats.json')
)
```

---

## Detailed Usage

### Step 1: Data Preparation

The GeographicHeatmap expects a CSV file with unified publication data. The file should contain at least:

- `id`: Unique article identifier
- `authors`: Author information (can include affiliations)
- `title`: Publication title
- Other metadata fields

Example data format:

```csv
id,title,authors,year,abstract,keywords,doi,source
pub_001,"AI Study","John Doe (MIT, USA); Jane Smith (Stanford, USA)",2023,"...","AI;ML","10.1234/1","IEEE"
pub_002,"ML Paper","Maria Garcia, University of Barcelona, Spain",2023,"...","ML;DL","10.1234/2","ACM"
```

### Step 2: Extract Affiliations

```python
# Extract affiliations from authors field
affiliations = geo_map.extract_author_affiliations()

# View extracted data
for article_id, affiliation in list(affiliations.items())[:5]:
    print(f"{article_id}:")
    print(f"  Author: {affiliation['first_author']}")
    print(f"  Institution: {affiliation['institution']}")
    print(f"  Country: {affiliation['country']}")
```

The extraction recognizes patterns like:
- `"Author (Institution, Country)"`
- `"Author, Institution, City, Country"`
- `"Author; Another Author"`

### Step 3: Geocode Locations

```python
# Geocode extracted countries
geo_data = geo_map.geocode_locations()

# View geocoded data
for country, data in list(geo_data.items())[:5]:
    print(f"{country}:")
    print(f"  Latitude: {data['lat']}")
    print(f"  Longitude: {data['lon']}")
    print(f"  Publications: {data['count']}")
    print(f"  Continent: {data['continent']}")
```

### Step 4: Calculate Metrics

```python
# Calculate publication density
density_df = geo_map.calculate_publication_density()

# Display top countries
print("\nTop 10 Countries:")
print(density_df.head(10))

# Filter by continent
europe = density_df[density_df['continent'] == 'Europe']
print(f"\nEuropean countries: {len(europe)}")
```

### Step 5: Create Visualizations

#### Folium Interactive Map

```python
geo_map.create_interactive_map(
    output_html='output/folium_map.html'
)
```

Features:
- Hover tooltips showing country and publication count
- Clickable markers with detailed information
- Heatmap layer (toggle on/off)
- Marker clustering for dense areas
- Multiple base map options

#### Plotly Interactive Map

```python
geo_map.create_plotly_map(
    output_html='output/plotly_map.html'
)
```

Features:
- Smooth zoom and pan
- Color scale legend
- Detailed hover information
- Easy export to PNG/SVG
- Responsive design

#### Static Map for PDF

```python
geo_map.create_static_map(
    output_png='output/publication_map.png',
    dpi=300  # High resolution
)
```

Features:
- 300+ DPI for print quality
- Professional cartographic projection
- Clean legend and annotations
- Suitable for academic publications

### Step 6: Generate Statistics

```python
stats = geo_map.generate_geographic_statistics()

# Access statistics
print(f"Total countries: {stats['summary']['total_countries']}")
print(f"Total publications: {stats['summary']['total_publications']}")

# Top countries
for country in stats['top_10_countries']:
    print(f"{country['country']}: {country['publications']} ({country['percentage']:.1f}%)")

# Continent distribution
for continent in stats['continent_distribution']:
    print(f"{continent['continent']}: {continent['publications']} publications")

# Save to file
geo_map.save_statistics_report('output/stats.json')
```

---

## API Reference

### GeographicHeatmap Class

#### Constructor

```python
GeographicHeatmap(unified_data_path: str)
```

**Parameters:**
- `unified_data_path`: Path to unified CSV data file

**Raises:**
- `FileNotFoundError`: If data file doesn't exist

#### Methods

##### extract_author_affiliations()

```python
extract_author_affiliations(author_field: str = 'authors') -> Dict[str, Dict[str, str]]
```

Extract country/institution from author information.

**Parameters:**
- `author_field`: Column name containing author data (default: 'authors')

**Returns:**
Dictionary mapping article IDs to affiliation data:
```python
{
    'article_id': {
        'first_author': str,
        'institution': str,
        'city': str,
        'country': str
    }
}
```

##### geocode_locations()

```python
geocode_locations(countries: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]
```

Geocode countries to coordinates.

**Parameters:**
- `countries`: Optional list of countries to geocode

**Returns:**
```python
{
    'country_name': {
        'lat': float,
        'lon': float,
        'count': int,
        'continent': str
    }
}
```

##### calculate_publication_density()

```python
calculate_publication_density(geo_data: Optional[Dict] = None) -> pd.DataFrame
```

Calculate publication density metrics.

**Returns:**
DataFrame with columns: country, publications, percentage, lat, lon, continent

##### create_interactive_map()

```python
create_interactive_map(
    geo_data: Optional[Dict] = None,
    output_html: str = 'geographic_heatmap.html'
)
```

Generate interactive Folium map.

**Parameters:**
- `geo_data`: Optional geocoded data
- `output_html`: Path to save HTML file

##### create_plotly_map()

```python
create_plotly_map(
    geo_data: Optional[Dict] = None,
    output_html: str = 'geographic_plotly.html'
)
```

Generate interactive Plotly map.

##### create_static_map()

```python
create_static_map(
    geo_data: Optional[Dict] = None,
    output_png: str = 'geographic_map.png',
    dpi: int = 300
)
```

Generate static high-quality map.

**Parameters:**
- `dpi`: Resolution (default: 300 for print quality)

##### generate_geographic_statistics()

```python
generate_geographic_statistics() -> Dict[str, Any]
```

Generate comprehensive statistics.

**Returns:**
```python
{
    'summary': {...},
    'top_10_countries': [...],
    'continent_distribution': [...],
    'coverage': {...}
}
```

---

## Examples

### Example 1: Basic Analysis

```python
from src.visualization.geographic_heatmap import GeographicHeatmap

geo_map = GeographicHeatmap('data/processed/unified_data.csv')
affiliations = geo_map.extract_author_affiliations()

print(f"Extracted {len(affiliations)} affiliations")
```

### Example 2: Top Countries Report

```python
geo_map = GeographicHeatmap('data/processed/unified_data.csv')
geo_map.extract_author_affiliations()
geo_map.geocode_locations()

density = geo_map.calculate_publication_density()

print("Top 10 Countries by Publications:")
for idx, row in density.head(10).iterrows():
    print(f"{idx+1}. {row['country']}: {row['publications']} ({row['percentage']:.1f}%)")
```

### Example 3: Multi-format Visualization

```python
geo_map = GeographicHeatmap('data/processed/unified_data.csv')
geo_map.extract_author_affiliations()
geo_map.geocode_locations()

# Create all formats
geo_map.create_interactive_map(output_html='output/web_map.html')
geo_map.create_plotly_map(output_html='output/plotly_map.html')
geo_map.create_static_map(output_png='output/print_map.png', dpi=300)
```

### Example 4: Filtering by Region

```python
geo_map = GeographicHeatmap('data/processed/unified_data.csv')
geo_map.extract_author_affiliations()
geo_data = geo_map.geocode_locations()

# Filter European countries
europe_data = {
    country: data for country, data in geo_data.items()
    if data['continent'] == 'Europe'
}

# Create Europe-only map
geo_map.create_interactive_map(geo_data=europe_data, output_html='output/europe_map.html')
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

### Issue: Cartopy installation fails

**Error:**
```
ERROR: Failed building wheel for cartopy
```

**Solutions:**

1. Use conda:
   ```bash
   conda install -c conda-forge cartopy
   ```

2. Install system dependencies first (Linux):
   ```bash
   sudo apt-get install libgeos-dev libproj-dev
   pip install cartopy
   ```

3. Skip cartopy (use basic matplotlib):
   - The module will automatically fall back to basic matplotlib if cartopy is not available

### Issue: No affiliations extracted

**Possible causes:**
- Author field name is different (not 'authors')
- Author data format is non-standard
- Missing or empty author information

**Solution:**
```python
# Specify custom author field
affiliations = geo_map.extract_author_affiliations(author_field='author_names')

# Check data format
print(geo_map.df['authors'].head())
```

### Issue: Low geocoding coverage

**Possible causes:**
- Country names in non-standard format
- Affiliations not properly extracted
- Missing country information in author data

**Solution:**
- Add custom country mappings to the database
- Enhance pattern matching rules
- Manually review and clean affiliation data

---

## Advanced Topics

### Custom Country Database

Add custom countries or update coordinates:

```python
geo_map = GeographicHeatmap('data/processed/unified_data.csv')

# Add custom country
geo_map._country_database['Custom Country'] = {
    'name': 'Custom Country',
    'lat': 45.0,
    'lon': 10.0,
    'continent': 'Europe'
}
```

### Custom Institution Mapping

Map institutions to countries:

```python
geo_map._university_to_country.update({
    'My University': 'USA',
    'Research Institute': 'Germany'
})
```

### Custom Affiliation Patterns

Extend pattern matching for non-standard formats:

```python
# Override or extend _extract_affiliation_patterns method
# Add custom regex patterns for your specific data format
```

### Performance Optimization

For large datasets:

```python
# 1. Cache results
affiliations = geo_map.extract_author_affiliations()
# Save to pickle for reuse
import pickle
with open('affiliations_cache.pkl', 'wb') as f:
    pickle.dump(affiliations, f)

# 2. Process in chunks
# For very large datasets, process in batches

# 3. Disable NER for speed
geo_map._nlp = None  # Use pattern-based extraction only
```

### Integration with Reports

```python
from pathlib import Path

def generate_geographic_report(data_path, output_dir):
    """Generate complete geographic analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    geo_map = GeographicHeatmap(data_path)
    geo_map.extract_author_affiliations()
    geo_map.geocode_locations()

    # Generate all outputs
    geo_map.create_interactive_map(
        output_html=str(output_dir / 'interactive_map.html')
    )
    geo_map.create_static_map(
        output_png=str(output_dir / 'geographic_distribution.png'),
        dpi=300
    )
    geo_map.save_statistics_report(
        str(output_dir / 'geographic_statistics.json')
    )

    print(f"Report generated in: {output_dir}")

# Use in pipeline
generate_geographic_report(
    'data/processed/unified_data.csv',
    'output/reports/geographic'
)
```

---

## Best Practices

1. **Data Quality**: Ensure author affiliations are well-formatted in source data
2. **Validation**: Review extracted affiliations for accuracy
3. **Coverage**: Document geocoding coverage percentage
4. **Formats**: Generate multiple visualization formats for different use cases
5. **Resolution**: Use 300+ DPI for print publications
6. **Caching**: Cache geocoding results for large datasets
7. **Documentation**: Include methodology notes in reports

---

## Citation

If you use this module in academic work, please cite:

```
Bibliometric Analysis System - Geographic Heatmap Module
Author: [Your Name/Organization]
Year: 2024
URL: [Repository URL]
```

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [Repository Issues]
- Documentation: [Full Documentation]
- Examples: See `examples/geographic_heatmap_demo.py`

---

## License

[Your License Here]

---

**Last Updated**: October 2024
**Version**: 1.0.0
