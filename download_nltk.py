# download_nltk.py
# This is a one-time setup script to download necessary NLTK data packages.

import nltk
import ssl

def download_nltk_data():
    """
    Downloads the NLTK data packages required for the plagiarism checker project.
    Includes a workaround for SSL certificate verification issues on some systems.
    """
    packages = ['punkt', 'stopwords', 'wordnet']
    
    print("Starting download of NLTK data packages...")
    
    try:
        # This is a workaround for a common SSL certificate verification error.
        # It creates an unverified SSL context for the downloader.
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # If the attribute doesn't exist, it means we're on a Python version
        # where this is not an issue, or it has been handled differently.
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for package_id in packages:
        try:
            # The find() method will raise a LookupError if the resource is not found.
            # We construct the resource path based on common NLTK structures.
            # Note: The exact path might vary slightly depending on the package type.
            # This is a general approach to check for common package types.
            if package_id == 'punkt':
                nltk.data.find(f'tokenizers/{package_id}')
            elif package_id == 'stopwords':
                nltk.data.find(f'corpora/{package_id}')
            elif package_id == 'wordnet':
                nltk.data.find(f'corpora/{package_id}')
            
            print(f"-> NLTK package '{package_id}' is already downloaded.")
        except LookupError:
            print(f"-> NLTK package '{package_id}' not found. Downloading...")
            nltk.download(package_id)
            print(f"   Downloaded '{package_id}' successfully.")

    print("\nAll necessary NLTK data packages are installed and ready.")

if __name__ == "__main__":
    download_nltk_data()
