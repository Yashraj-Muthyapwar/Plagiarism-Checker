# download_nltk.py
import nltk
import ssl

def download_nltk_data():
    """
    Downloads the NLTK data packages required for the plagiarism checker project.
    """
    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    
    print("Starting download of NLTK data packages...")
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for package_id in packages:
        try:
            # Check different possible locations
            if package_id in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{package_id}')
            elif package_id in ['stopwords', 'wordnet', 'omw-1.4']:
                nltk.data.find(f'corpora/{package_id}')
            
            print(f"-> NLTK package '{package_id}' is already downloaded.")
        except LookupError:
            print(f"-> NLTK package '{package_id}' not found. Downloading...")
            try:
                nltk.download(package_id, quiet=False)
                print(f"   Downloaded '{package_id}' successfully.")
            except Exception as e:
                print(f"   Error downloading '{package_id}': {e}")

    print("\nAll necessary NLTK data packages are installed and ready.")

if __name__ == "__main__":
    download_nltk_data()
