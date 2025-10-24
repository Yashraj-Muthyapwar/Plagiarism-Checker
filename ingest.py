# ingest.py (FIXED FOR 100% CORPUS CONSISTENCY)
"""
Updated ingestion script to use Sentence-BERT embeddings.
Fixed to ensure embeddings match exactly with app.py queries.
"""

import os
import sys
import sqlite3
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils import read_txt_file, read_docx_file, read_pdf_file
    from embedder import embed_texts, save_embedder, EMBEDDER
except ImportError:
    print("Error: Could not import utility functions from 'src'.")
    sys.exit(1)

DATABASE_FILE = "corpus.db"
CORPUS_DIRECTORY = os.path.join("data", "corpus_files")
MODELS_DIR = "models"
SBERT_MODEL_PATH = os.path.join(MODELS_DIR, "sbert_model")

def read_file_content(file_path):
    """Helper to dispatch to correct file reader."""
    _, extension = os.path.splitext(file_path)
    if extension == '.txt':
        return read_txt_file(file_path)
    elif extension == '.docx':
        return read_docx_file(file_path)
    elif extension == '.pdf':
        return read_pdf_file(file_path)
    else:
        return None

def create_database_and_table():
    """Creates database and documents table."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL UNIQUE,
        text_content TEXT NOT NULL,
        document_vector BLOB,
        upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Database '{DATABASE_FILE}' is ready.")
        return conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

def ingest_documents(conn):
    """Ingests documents with SBERT embeddings - ONE AT A TIME for consistency."""
    print(f"\nStarting ingestion from: '{CORPUS_DIRECTORY}'")
    
    if not os.path.isdir(CORPUS_DIRECTORY):
        print(f"Error: Corpus directory not found at '{CORPUS_DIRECTORY}'")
        return

    # Read all document filenames first
    print("\n--- Processing documents individually ---")
    cursor = conn.cursor()
    inserted_count = 0
    skipped_count = 0
    
    for filename in os.listdir(CORPUS_DIRECTORY):
        file_path = os.path.join(CORPUS_DIRECTORY, filename)
        if os.path.isfile(file_path):
            print(f"  - Processing '{filename}'...")
            
            content = read_file_content(file_path)
            if not content:
                print(f"    -> Skipped (could not read content).")
                skipped_count += 1
                continue
            
            # CRITICAL FIX: Generate embedding individually (same as app.py does)
            # This ensures exact same process as when checking against corpus
            embedding = embed_texts(content)  # Single text, not batch
            
            # embedding is already 2D from embed_texts(), shape: (1, 384)
            # Store the entire array (not just first row)
            serialized_embedding = pickle.dumps(embedding)
            
            insert_sql = """
            INSERT INTO documents (filename, text_content, document_vector) 
            VALUES (?, ?, ?)
            """
            
            try:
                cursor.execute(insert_sql, (filename, content, serialized_embedding))
                inserted_count += 1
                print(f"    -> Successfully inserted.")
            except sqlite3.IntegrityError:
                print(f"    -> Skipped (already exists).")
                skipped_count += 1
            except sqlite3.Error as e:
                print(f"    -> Error: {e}")
                skipped_count += 1

    conn.commit()
    
    # Save the SBERT model (optional)
    print(f"\nSaving SBERT model to '{SBERT_MODEL_PATH}'...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_embedder(EMBEDDER, SBERT_MODEL_PATH)
    print("SBERT model saved.")
    
    print("\nIngestion complete.")
    print(f"  {inserted_count} new documents inserted.")
    print(f"  {skipped_count} documents skipped.")

def main():
    """Main ingestion function."""
    conn = create_database_and_table()
    if conn:
        try:
            ingest_documents(conn)
        finally:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()
