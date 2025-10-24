# app.py
"""
Main Streamlit Web Application for the Plagiarism Checker.
Enhanced minimalistic UI with better visual hierarchy.
"""

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import pickle
import numpy as np

sys.path.append('src')

try:
    from utils import (
        read_uploaded_file, get_all_documents_from_db, 
        validate_document_text
    )
    from similarity import calculate_similarity
    from highlighter import (
        split_into_sentences,
        vectorize_sentences,
        calculate_sentence_similarity_matrix,
        identify_matching_sentences,
        generate_html_report
    )
    from embedder import embed_texts

except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Fixed threshold - optimized for academic plagiarism detection
DEFAULT_THRESHOLD = 0.8

@st.cache_resource
def load_sbert_model():
    """Loads the Sentence-BERT model (cached for performance)."""
    try:
        from embedder import EMBEDDER
        return EMBEDDER
    except Exception as e:
        st.error(f"Error loading SBERT model: {e}")
        return None

def load_css():
    """Defines the custom CSS for highlighting."""
    st.markdown("""
        <style>
            .highlight {
                background-color: #f4cccc;
                color: #7a1c1c;
                padding: 2px 4px;
                border-radius: 4px;
                border: 1px solid #e6b8b8;
            }
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .section-header {
                font-size: 1.5rem;
                font-weight: 600;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        </style>
    """, unsafe_allow_html=True)

def safe_progress_value(value: float) -> float:
    """Ensures progress value is within valid range [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))

def flexible_validate_document_text(text: str, filename: str) -> tuple:
    """
    More flexible validation that allows smaller documents for testing.
    """
    if not text:
        return False, f"❌ No text could be extracted from '{filename}'. The file may be empty or in an unsupported format."
    
    words = text.split()
    word_count = len(words)
    
    # Reduced minimum word count for testing
    if word_count < 10:
        return False, f"❌ Very limited text in '{filename}' - only {word_count} words extracted. Please use a document with more content."
    
    # Check text quality metrics
    valid_words = [word for word in words if len(word) > 1]
    valid_ratio = len(valid_words) / len(words) if words else 0
    
    if valid_ratio < 0.5:  # Reduced from 0.7
        return False, f"❌ Poor text quality in '{filename}' - only {valid_ratio:.1%} of words appear valid."
    
    return True, ""

def get_similarity_interpretation(score: float):
    """Returns interpretation and styling for similarity score."""
    if score > 80:
        return {
            'label': 'High Similarity',
            'emoji': '🔴',
            'description': 'Substantial overlap detected - requires review'
        }
    elif score > 50:
        return {
            'label': 'Moderate Similarity', 
            'emoji': '🟡',
            'description': 'Some overlap detected - manual review recommended'
        }
    else:
        return {
            'label': 'Low Similarity',
            'emoji': '🟢',
            'description': 'Minimal overlap - likely original work'
        }

def main():
    st.set_page_config(page_title="Plagiarism Checker Pro", page_icon="📚", layout="wide")
    
    # Load model
    sbert_model = load_sbert_model()
    if sbert_model is None:
        st.stop()

    load_css()
    
    # Header
    st.title("📚 Plagiarism Checker Pro")
    st.markdown("""
    Welcome to **Plagiarism Checker Pro**! This tool helps you compare text documents 
    to detect potential plagiarism. Choose your comparison mode below.
    """)
 
    
    # Mode selection
    mode = st.radio(
        "Select Comparison Mode",
        ("Compare two documents", "Compare against corpus"),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    # ========== TWO DOCUMENT COMPARISON MODE ==========
    if mode == "Compare two documents":
        st.subheader("Mode: Compare Two Files")
        st.info("Upload two documents below to calculate the similarity between them.")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file1 = st.file_uploader(
                "Source Document",
                type=['txt', 'pdf', 'docx'],
                key="file1",
                label_visibility="collapsed"
            )
        with col2:
            uploaded_file2 = st.file_uploader(
                "Document to Check", 
                type=['txt', 'pdf', 'docx'],
                key="file2",
                label_visibility="collapsed"
            )

        # Analysis button
        if uploaded_file1 and uploaded_file2:
            if st.button("**🔍 Analyze Documents**", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    try:
                        # Read files
                        source_text = read_uploaded_file(uploaded_file1)
                        suspect_text = read_uploaded_file(uploaded_file2)
                        
                        # Use flexible validation
                        is_source_valid, source_error = flexible_validate_document_text(source_text, uploaded_file1.name)
                        is_suspect_valid, suspect_error = flexible_validate_document_text(suspect_text, uploaded_file2.name)
                        
                        if not is_source_valid:
                            st.error(source_error)
                            return
                        if not is_suspect_valid:
                            st.error(suspect_error)
                            return
                        
                        # Overall similarity
                        similarity_score = calculate_similarity(source_text, suspect_text)
                        interp = get_similarity_interpretation(similarity_score)
                        
                        
                        st.subheader("Overall Similarity Score")
                        st.progress(safe_progress_value(similarity_score / 100))
                        st.metric("Document Similarity", f"{similarity_score:.1f}%")
                        st.divider()
                        
                        # Detailed analysis for documents with sufficient content
                        if len(suspect_text.split()) >= 20:
                            st.subheader("Detailed Plagiarism Report")
                            # Sentence-level analysis with fixed threshold
                            source_sentences = split_into_sentences(source_text)
                            suspect_sentences = split_into_sentences(suspect_text)
                            
                            if len(source_sentences) > 0 and len(suspect_sentences) > 0:
                                source_embeddings, suspect_embeddings = vectorize_sentences(
                                    source_sentences, suspect_sentences
                                )
                                
                                sim_matrix = calculate_sentence_similarity_matrix(
                                    source_embeddings, suspect_embeddings
                                )
                                
                                # Use fixed threshold
                                plagiarized_indices = identify_matching_sentences(sim_matrix, threshold=DEFAULT_THRESHOLD)
                                percentage = (len(plagiarized_indices) / len(suspect_sentences)) * 100 if suspect_sentences else 0
                                
                                # Statistics in cards
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    with st.container():
                                        st.metric("Total Sentences", len(suspect_sentences))
                                with col2:
                                    with st.container():
                                        st.metric("Similar Sentences", len(plagiarized_indices))
                                with col3:
                                    with st.container():
                                        st.metric("Similarity Ratio", f"{percentage:.1f}%")
                                
                                # Highlighted report
                                if plagiarized_indices:
                                    st.markdown("---")
                                    st.write("**📝 Document with Highlighted Similar Content**")
                                    html_report = generate_html_report(suspect_sentences, plagiarized_indices)
                                    st.markdown(html_report, unsafe_allow_html=True)
                                else:
                                    st.success("✅ No similar sentences detected above the threshold")
                            else:
                                st.warning("⚠️ Could not extract enough sentences for detailed analysis")
                        else:
                            st.info("📄 For detailed sentence-level analysis, use documents with 20+ words")

                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
        else:
           st.warning("Please upload both documents to generate a report.")

    # ========== CORPUS COMPARISON MODE ==========
    else:
        st.subheader("Mode: Compare Against Corpus")
        st.info("Upload a single document to check it against all documents in the corpus database.")
        uploaded_file1 = st.file_uploader(
            "Document to check against corpus",
            type=['txt', 'pdf', 'docx'],
            key="corpus_check_file",
            label_visibility="collapsed"
        )   
    
        if uploaded_file1:
            if st.button("**🔬 Check Against Corpus**", type="primary", use_container_width=True):
                with st.spinner("Searching corpus..."):
                    try:
                        suspect_text = read_uploaded_file(uploaded_file1)
                        
                        # Use flexible validation for corpus mode too
                        is_valid, error_msg = flexible_validate_document_text(suspect_text, uploaded_file1.name)
                        if not is_valid:
                            st.error(error_msg)
                            return
                        
                        suspect_embedding = embed_texts(suspect_text)
                        corpus_docs = get_all_documents_from_db()
                        
                        if not corpus_docs:
                            st.error("❌ Corpus is empty. Run ingest.py first.")
                            st.stop()
                   
                        corpus_filenames = [doc[0] for doc in corpus_docs]
                        corpus_embeddings_list = [pickle.loads(doc[1]) for doc in corpus_docs]
                        corpus_embeddings_matrix = np.vstack(corpus_embeddings_list)

                        if suspect_embedding.ndim == 1:
                            suspect_embedding = suspect_embedding.reshape(1, -1)
                        if corpus_embeddings_matrix.ndim == 1:
                            corpus_embeddings_matrix = corpus_embeddings_matrix.reshape(1, -1)

                        similarity_scores = cosine_similarity(
                            suspect_embedding, corpus_embeddings_matrix
                        ).flatten()

                        results = list(zip(corpus_filenames, similarity_scores))
                        sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
                        
                        st.markdown('<div class="section-header">📈 Corpus Comparison Results</div>', unsafe_allow_html=True)
                        st.write(f"**Found {len(sorted_results)} documents in corpus**")
                        
                        top_matches = sorted_results[:5]

                        for i, (filename, score) in enumerate(top_matches, 1):
                            percentage = score * 100
                            interp = get_similarity_interpretation(percentage)
                            
                            # Create a clean match card
                            col1, col2, col3 = st.columns([0.3, 4, 1.5])
                            with col1:
                                st.write(f"**#{i}**")
                            with col2:
                                st.write(f"**{filename}**")
                                st.progress(safe_progress_value(score))
                            with col3:
                                st.write(f"**{percentage:.1f}%**")
                                st.caption(f"{interp['emoji']} {interp['label']}")
                            
                            if i < len(top_matches):
                                st.markdown("---")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
