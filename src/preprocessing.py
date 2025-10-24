# src/preprocessing.py
"""
Core Text Preprocessing Module for the Plagiarism Checker.

This module is responsible for taking raw text and transforming it into a clean,
standardized format suitable for NLP analysis. The process involves a pipeline
of several steps:
1.  Tokenization: Breaking text into individual words (tokens).
2.  Lowercasing: Converting all tokens to lowercase.
3.  Punctuation Removal: Removing all punctuation characters.
4.  Stopword Filtering: Removing common words that don't add semantic value.
5.  Lemmatization: Reducing words to their base or dictionary form.

By centralizing these functions here, we create a reusable and testable
component that forms the core of the plagiarism detection engine.
"""

import string
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes a string of text into a list of words and punctuation.

    This function serves as the first step in the NLP preprocessing pipeline.
    It uses NLTK's `word_tokenize`, which is more sophisticated than a simple
    string split. It relies on the 'punkt' NLTK data package.

    Args:
        text (str): The raw input string to be tokenized.

    Returns:
        List[str]: A list of tokens (words and punctuation). Returns an empty
                   list if the input is not a string or is empty.
    """
    if not isinstance(text, str):
        print("Error: Input for tokenization must be a string.")
        return []
    tokens = word_tokenize(text)
    return tokens

def normalize_case(tokens: List[str]) -> List[str]:
    """
    Converts a list of tokens to lowercase.

    This step, known as case normalization, is crucial for treating words like
    'The' and 'the' as identical, reducing the vocabulary size and improving
    the accuracy of the similarity analysis.

    Args:
        tokens (List[str]): A list of string tokens.

    Returns:
        List[str]: A new list containing the lowercase version of each token.
    """
    lower_tokens = [token.lower() for token in tokens]
    return lower_tokens

def remove_punctuation(tokens: List[str]) -> List[str]:
    """
    Removes punctuation tokens from a list of tokens.

    This function filters a list of tokens, removing any token that is found
    in Python's standard `string.punctuation` set. This helps to clean the
    data by focusing only on word tokens.

    Args:
        tokens (List[str]): A list of lowercase string tokens.

    Returns:
        List[str]: A new list of tokens with all punctuation tokens removed.
    """
    cleaned_tokens = [token for token in tokens if token not in string.punctuation]
    return cleaned_tokens

def filter_stopwords(tokens: List[str]) -> List[str]:
    """
    Removes common English stopwords from a list of tokens.

    This function uses NLTK's pre-compiled list of English stopwords. For
    efficiency, the stopword list is converted to a set to allow for
    fast membership checking.

    Args:
        tokens (List[str]): A list of clean, lowercase word tokens.

    Returns:
        List[str]: A new list of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Reduces each word in a list of tokens to its dictionary form (lemma).

    This function uses NLTK's WordNetLemmatizer. It is a crucial step for
    grouping different inflected forms of a word into a single conceptual
    item (e.g., 'cars' becomes 'car').

    Note: The default lemmatizer assumes every word is a noun. For more
    accuracy, part-of-speech (POS) tagging would be needed as a prior step.

    Args:
        tokens (List[str]): A list of filtered, clean, lowercase word tokens.

    Returns:
        List[str]: A new list containing the lemmatized version of each token.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# --- NEW CODE STARTS HERE ---

def preprocess_text(text: str) -> List[str]:
    """
    A complete text preprocessing pipeline.

    This function takes raw text and applies a sequence of cleaning and
    normalization steps:
    1. Tokenization
    2. Case normalization (lowercasing)
    3. Punctuation removal
    4. Stopword filtering
    5. Lemmatization

    It serves as the main entry point for text preprocessing in the application.

    Args:
        text (str): The raw text string to be processed.

    Returns:
        List[str]: A list of clean, lemmatized tokens ready for analysis.
    """
    # Step 1: Tokenize the raw text.
    tokens = tokenize_text(text)
    
    # Step 2: Convert all tokens to lowercase.
    lower_tokens = normalize_case(tokens)
    
    # Step 3: Remove all punctuation tokens.
    no_punc_tokens = remove_punctuation(lower_tokens)
    
    # Step 4: Filter out all stopwords.
    filtered_tokens = filter_stopwords(no_punc_tokens)
    
    # Step 5: Lemmatize the remaining tokens.
    final_tokens = lemmatize_tokens(filtered_tokens)
    
    return final_tokens

# --- NEW CODE ENDS HERE ---
