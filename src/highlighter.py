# src/highlighter.py (UPDATED WITH FIXES)
"""
This module contains functions related to generating a detailed plagiarism report,
including sentence-level analysis and HTML highlighting of similar sentences.
"""

import nltk
from src.embedder import embed_texts  # Changed from vectorizer to embedder
import numpy as np
import html
from sklearn.metrics.pairwise import cosine_similarity

def split_into_sentences(text):
    """
    Splits a given block of text into a list of sentences.
    """
    if not text:
        return []
    
    sentences = nltk.sent_tokenize(text)
    return sentences

def vectorize_sentences(source_sentences, suspect_sentences):
    """
    Creates SBERT embeddings for two lists of sentences.
    
    Args:
        source_sentences (list[str]): Sentences from source document
        suspect_sentences (list[str]): Sentences from suspect document
        
    Returns:
        tuple: (source_embeddings, suspect_embeddings) as numpy arrays
    """
    if not source_sentences and not suspect_sentences:
        return (None, None)

    all_sentences = source_sentences + suspect_sentences
    all_embeddings = embed_texts(all_sentences)

    num_source_sentences = len(source_sentences)
    source_embeddings = all_embeddings[:num_source_sentences]
    suspect_embeddings = all_embeddings[num_source_sentences:]
    
    return source_embeddings, suspect_embeddings

def calculate_sentence_similarity_matrix(source_embeddings, suspect_embeddings):
    """
    Calculates cosine similarity between sentence embeddings.
    """
    if source_embeddings is None or suspect_embeddings is None:
        return np.array([])
    if len(source_embeddings) == 0 or len(suspect_embeddings) == 0:
        return np.array([])
    
    # FIX: Ensure both arrays are 2D for cosine_similarity
    if source_embeddings.ndim == 1:
        source_embeddings = source_embeddings.reshape(1, -1)
    if suspect_embeddings.ndim == 1:
        suspect_embeddings = suspect_embeddings.reshape(1, -1)
        
    similarity_matrix = cosine_similarity(suspect_embeddings, source_embeddings)
    return similarity_matrix

def identify_matching_sentences(similarity_matrix, threshold=0.8):
    """
    Identifies plagiarized sentences based on similarity threshold.
    """
    if similarity_matrix.size == 0:
        return set()
        
    matching_pairs = np.argwhere(similarity_matrix >= threshold)
    plagiarized_suspect_indices = [match[0] for match in matching_pairs]

    return set(plagiarized_suspect_indices)

def generate_html_report(suspect_sentences, plagiarized_indices):
    """
    Generates HTML with plagiarized sentences highlighted.
    """
    highlighted_html_parts = []
    
    for i, sentence in enumerate(suspect_sentences):
        safe_sentence = html.escape(sentence)
        
        if i in plagiarized_indices:
            highlighted_sentence = f'<mark class="highlight">{safe_sentence}</mark>'
            highlighted_html_parts.append(highlighted_sentence)
        else:
            highlighted_html_parts.append(safe_sentence)
            
    return " ".join(highlighted_html_parts)
