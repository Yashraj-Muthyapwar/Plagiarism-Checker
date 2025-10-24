# src/similarity.py (FIXED FOR 100% ACCURACY)
"""
Core Similarity Calculation Module for the Plagiarism Checker.
Uses Sentence-BERT embeddings with proper precision handling.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import embed_texts

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculates cosine similarity between two texts using SBERT embeddings.
    
    Args:
        text1 (str): First text document
        text2 (str): Second text document
        
    Returns:
        float: Similarity percentage (0-100)
    """
    # Generate embeddings for both texts
    # embed_texts already normalizes embeddings (normalize_embeddings=True)
    embeddings = embed_texts([text1, text2])
    
    # Use sklearn's cosine_similarity directly
    similarity_matrix = cosine_similarity(embeddings[0:1], embeddings[1:2])
    similarity_score = similarity_matrix[0, 0]
    
    # Round to percentage
    percentage_score = round(similarity_score * 100, 2)
    
    return percentage_score

if __name__ == '__main__':
    print("--- Running SBERT Similarity Module Test ---")

    sample_corpus = [
        "Python is a versatile programming language used for web development.",
        "Many programmers enjoy using the Python language for web projects.", 
        "The sun is a star at the center of our solar system.",
        "Python is a versatile programming language used for web development."
    ]

    print("\nTest Corpus:")
    for i, doc in enumerate(sample_corpus, 1):
        print(f"  Doc {i}: '{doc}'")

    print("\nPerforming pairwise similarity calculations with SBERT...")

    # Test different scenarios
    score_identical = calculate_similarity(sample_corpus[0], sample_corpus[3])
    score_similar = calculate_similarity(sample_corpus[0], sample_corpus[1])
    score_different = calculate_similarity(sample_corpus[0], sample_corpus[2])
    
    print(f"  - Identical docs: {score_identical}% (Should be 100.00%)")
    print(f"  - Similar docs: {score_similar}%") 
    print(f"  - Different docs: {score_different}%")
    
    print("\n--- SBERT Test Complete ---")
