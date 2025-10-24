# src/embedder.py (ADDITIONAL FIX FOR COSINE SIMILARITY PRECISION)
"""
Sentence-BERT Embedder Module for the Plagiarism Checker Project.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from typing import List, Union
import torch

# Load a pre-trained Sentence-BERT model
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generates Sentence-BERT embeddings for text(s).
    """
    if isinstance(texts, str):
        texts = [texts]
    
    print(f"Generating SBERT embeddings for {len(texts)} text(s)...")
    embeddings = EMBEDDER.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    print("Embedding generation complete.")
    
    # FIX: Ensure consistent return type - always return 2D array
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    return embeddings

def embed_corpus(corpus: List[str]) -> np.ndarray:
    """
    Generates SBERT embeddings for an entire corpus.
    """
    return embed_texts(corpus)

def safe_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity with safety checks for floating-point precision.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        np.ndarray: Similarity scores clamped to [0.0, 1.0]
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Calculate raw cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings2)
    
    # Clamp values to handle floating-point precision issues
    similarity = np.clip(similarity, 0.0, 1.0)
    
    return similarity

def save_embedder(model: SentenceTransformer, file_path: str):
    """
    Saves the SentenceTransformer model.
    """
    try:
        print(f"Saving SBERT model to {file_path}...")
        model.save(file_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_embedder(file_path: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model.
    """
    try:
        print(f"Loading SBERT model from {file_path}...")
        model = SentenceTransformer(file_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Test the embedder
    sample_texts = [
        "Python is a programming language",
        "Java is also a programming language", 
        "The weather is nice today"
    ]
    
    embeddings = embed_texts(sample_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print("SBERT embedder test completed successfully.")
