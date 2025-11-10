"""
Installation Verification Script
Checks if all required packages are installed and working correctly
"""

import sys
from typing import Dict, List, Tuple


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and importable

    Args:
        package_name: Display name of the package
        import_name: Actual import name (if different from package_name)

    Returns:
        Tuple of (success, version_or_error)
    """
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main verification function"""

    print("=" * 70)
    print("BIBLIOMETRIC ANALYSIS - Installation Verification")
    print("=" * 70)
    print()

    # Define packages to check
    packages: Dict[str, List[Tuple[str, str]]] = {
        "Data Processing": [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("openpyxl", "openpyxl"),
        ],

        "Web Scraping": [
            ("requests", "requests"),
            ("beautifulsoup4", "bs4"),
            ("selenium", "selenium"),
            ("webdriver-manager", "webdriver_manager"),
        ],

        "Bibliographic Parsing": [
            ("python-RISparser", "rispy"),
            ("bibtexparser", "bibtexparser"),
        ],

        "Natural Language Processing": [
            ("nltk", "nltk"),
            ("spacy", "spacy"),
            ("python-Levenshtein", "Levenshtein"),
        ],

        "Machine Learning": [
            ("scikit-learn", "sklearn"),
            ("scipy", "scipy"),
        ],

        "Deep Learning & Transformers": [
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("sentence-transformers", "sentence_transformers"),
        ],

        "Visualization": [
            ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn"),
            ("plotly", "plotly"),
            ("wordcloud", "wordcloud"),
            ("folium", "folium"),
            ("networkx", "networkx"),
        ],

        "Report Generation": [
            ("reportlab", "reportlab"),
            ("fpdf", "fpdf"),
            ("pdfkit", "pdfkit"),
            ("Pillow", "PIL"),
        ],

        "Web Application": [
            ("streamlit", "streamlit"),
        ],

        "Utilities": [
            ("pyyaml", "yaml"),
            ("python-dotenv", "dotenv"),
            ("tqdm", "tqdm"),
            ("loguru", "loguru"),
        ],
    }

    all_success = True
    results = {}

    for category, package_list in packages.items():
        print(f"\n{category}")
        print("-" * 70)

        category_results = []
        for display_name, import_name in package_list:
            success, version = check_package(display_name, import_name)
            category_results.append((display_name, success, version))

            status = "✓" if success else "✗"
            color = "\033[92m" if success else "\033[91m"
            reset = "\033[0m"

            if success:
                print(f"  {color}{status}{reset} {display_name:30s} v{version}")
            else:
                print(f"  {color}{status}{reset} {display_name:30s} NOT INSTALLED")
                all_success = False

        results[category] = category_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_packages = sum(len(pkgs) for pkgs in packages.values())
    installed_packages = sum(
        sum(1 for _, success, _ in category_results if success)
        for category_results in results.values()
    )

    print(f"\nInstalled: {installed_packages}/{total_packages} packages")

    if all_success:
        print("\n✓ All packages installed successfully!")
        print("\nNext steps:")
        print("  1. Download NLP models: python scripts/download_nlp_models.py")
        print("  2. Configure environment: cp config/.env.example config/.env")
        print("  3. Run the application: python main.py")
    else:
        print("\n✗ Some packages are missing. Please install them:")
        print("  pip install -r requirements.txt")
        return 1

    # Additional checks
    print("\n" + "=" * 70)
    print("ADDITIONAL CHECKS")
    print("=" * 70)

    # Check Python version
    py_version = sys.version_info
    print(f"\nPython Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("  ✗ Python 3.8+ required")
        all_success = False
    else:
        print("  ✓ Python version OK")

    # Check NLTK data
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("  ✓ NLTK punkt tokenizer installed")
        except LookupError:
            print("  ✗ NLTK punkt tokenizer not found - run: python scripts/download_nlp_models.py")
            all_success = False
    except ImportError:
        pass

    # Check spaCy models
    try:
        import spacy
        try:
            spacy.load('es_core_news_sm')
            print("  ✓ spaCy Spanish model installed")
        except OSError:
            print("  ✗ spaCy Spanish model not found - run: python -m spacy download es_core_news_sm")
            all_success = False
    except ImportError:
        pass

    # Check PyTorch device
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"  ✓ PyTorch device: {device}")
    except ImportError:
        pass

    print("\n" + "=" * 70)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
