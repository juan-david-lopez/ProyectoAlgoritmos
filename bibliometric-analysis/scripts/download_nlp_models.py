"""
Download Required NLP Models
Downloads and installs NLTK data and spaCy language models
"""

import sys
import subprocess


def download_nltk_data():
    """Download required NLTK datasets"""
    print("=" * 70)
    print("Downloading NLTK Data")
    print("=" * 70)

    try:
        import nltk
    except ImportError:
        print("✗ NLTK not installed. Please run: pip install nltk")
        return False

    # List of required NLTK datasets
    nltk_datasets = [
        'punkt',              # Sentence tokenizer
        'stopwords',          # Stop words
        'wordnet',            # WordNet lexical database
        'averaged_perceptron_tagger',  # POS tagger
        'maxent_ne_chunker',  # Named entity chunker
        'words',              # Word lists
        'omw-1.4',            # Open Multilingual Wordnet
    ]

    print("\nDownloading NLTK datasets...")
    all_success = True

    for dataset in nltk_datasets:
        try:
            print(f"  Downloading {dataset}...", end=" ")
            nltk.download(dataset, quiet=True)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            all_success = False

    if all_success:
        print("\n✓ All NLTK datasets downloaded successfully!")
    else:
        print("\n✗ Some NLTK datasets failed to download")

    return all_success


def download_spacy_models():
    """Download required spaCy language models"""
    print("\n" + "=" * 70)
    print("Downloading spaCy Models")
    print("=" * 70)

    try:
        import spacy
    except ImportError:
        print("✗ spaCy not installed. Please run: pip install spacy")
        return False

    # List of spaCy models to download
    spacy_models = [
        ('es_core_news_sm', 'Spanish'),    # Spanish small model
        ('en_core_web_sm', 'English'),     # English small model
    ]

    print("\nDownloading spaCy models...")
    all_success = True

    for model_name, language in spacy_models:
        try:
            # Check if model is already installed
            try:
                spacy.load(model_name)
                print(f"  ✓ {language} model ({model_name}) already installed")
                continue
            except OSError:
                pass

            print(f"  Downloading {language} model ({model_name})...", end=" ")

            # Download using subprocess
            result = subprocess.run(
                [sys.executable, '-m', 'spacy', 'download', model_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✓")
            else:
                print(f"✗ Error: {result.stderr}")
                all_success = False

        except Exception as e:
            print(f"✗ Error: {e}")
            all_success = False

    if all_success:
        print("\n✓ All spaCy models downloaded successfully!")
    else:
        print("\n✗ Some spaCy models failed to download")

    return all_success


def verify_installations():
    """Verify that all models are installed correctly"""
    print("\n" + "=" * 70)
    print("Verifying Installations")
    print("=" * 70)

    all_ok = True

    # Verify NLTK
    try:
        import nltk
        print("\nNLTK Datasets:")
        datasets_to_check = ['punkt', 'stopwords', 'wordnet']

        for dataset in datasets_to_check:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
                print(f"  ✓ {dataset}")
            except LookupError:
                print(f"  ✗ {dataset} not found")
                all_ok = False
    except ImportError:
        print("  ✗ NLTK not installed")
        all_ok = False

    # Verify spaCy
    try:
        import spacy
        print("\nspaCy Models:")
        models_to_check = [
            ('es_core_news_sm', 'Spanish'),
            ('en_core_web_sm', 'English'),
        ]

        for model_name, language in models_to_check:
            try:
                nlp = spacy.load(model_name)
                print(f"  ✓ {language} ({model_name}) - {len(nlp.pipe_names)} components")
            except OSError:
                print(f"  ✗ {language} ({model_name}) not found")
                all_ok = False
    except ImportError:
        print("  ✗ spaCy not installed")
        all_ok = False

    return all_ok


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("NLP MODELS DOWNLOAD SCRIPT")
    print("=" * 70)

    # Download NLTK data
    nltk_success = download_nltk_data()

    # Download spaCy models
    spacy_success = download_spacy_models()

    # Verify installations
    verify_success = verify_installations()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if nltk_success and spacy_success and verify_success:
        print("\n✓ All NLP models downloaded and verified successfully!")
        print("\nYou can now run the application:")
        print("  python main.py")
        return 0
    else:
        print("\n✗ Some downloads failed. Please check the errors above.")
        print("\nYou can manually install models:")
        print("  NLTK: python -c \"import nltk; nltk.download('all')\"")
        print("  spaCy ES: python -m spacy download es_core_news_sm")
        print("  spaCy EN: python -m spacy download en_core_web_sm")
        return 1


if __name__ == "__main__":
    sys.exit(main())
