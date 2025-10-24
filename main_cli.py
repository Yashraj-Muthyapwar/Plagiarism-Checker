# main_cli.py (UPDATED)
"""
Main CLI for Plagiarism Checker - Updated for Sentence-BERT.
"""

import argparse
import os
import sys

from src.utils import read_txt_file, read_docx_file, read_pdf_file
from src.similarity import calculate_similarity  # Now uses SBERT

def read_file_content(file_path: str) -> str | None:
    """Reads file content based on extension."""
    try:
        _, extension = os.path.splitext(file_path)
        if extension == '.txt':
            return read_txt_file(file_path)
        elif extension == '.docx':
            return read_docx_file(file_path)
        elif extension == '.pdf':
            return read_pdf_file(file_path)
        else:
            print(f"Error: Unsupported file type '{extension}'.", file=sys.stderr)
            return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading '{file_path}': {e}", file=sys.stderr)
        return None

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Compare two text files using SBERT.")
    parser.add_argument("file1", help="Path to first file")
    parser.add_argument("file2", help="Path to second file")
    
    args = parser.parse_args()

    print(f"Comparing '{args.file1}' and '{args.file2}' with Sentence-BERT...")
    print("-" * 50)

    # Read files
    text1 = read_file_content(args.file1)
    text2 = read_file_content(args.file2)

    if text1 is None or text2 is None:
        print("\nFile reading failed. Check paths and formats.")
        sys.exit(1)

    print("Files read successfully.")
    print("Calculating semantic similarity with SBERT...")
    
    try:
        similarity_score = calculate_similarity(text1, text2)
        print("Similarity calculation complete.")
    except Exception as e:
        print(f"Error during similarity calculation: {e}", file=sys.stderr)
        sys.exit(1)

    # Present results
    print("\n" + "="*50)
    print("        SBERT Plagiarism Check Result")
    print("="*50)

    file1_name = os.path.basename(args.file1)
    file2_name = os.path.basename(args.file2)

    print(f"Semantic similarity between '{file1_name}' and '{file2_name}': {similarity_score}%")
    print("="*50)

if __name__ == "__main__":
    main()
