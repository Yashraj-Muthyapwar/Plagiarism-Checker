# src/pdf_enhanced.py (NEW - ROBUST PDF PROCESSING)
"""
Enhanced PDF processing with multiple fallback methods for academic papers.
"""

import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
import io
import re
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AcademicPDFProcessor:
    """
    Robust PDF processor with multiple extraction methods and fallbacks.
    """
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pymupdf,
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2
        ]
    
    def extract_text(self, file_input, max_pages: int = 50) -> Optional[str]:
        """
        Extract text from PDF using multiple methods with fallbacks.
        
        Args:
            file_input: File path or BytesIO object
            max_pages: Maximum pages to process (for performance)
            
        Returns:
            Extracted text or None if all methods fail
        """
        methods_results = []
        
        for method in self.extraction_methods:
            try:
                text = method(file_input, max_pages)
                if text and self._is_meaningful_text(text):
                    methods_results.append((method.__name__, text, len(text.split())))
                    logger.info(f"Method {method.__name__} extracted {len(text.split())} words")
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        if not methods_results:
            return None
        
        # Choose the method that extracted the most text
        best_method = max(methods_results, key=lambda x: x[2])
        logger.info(f"Selected {best_method[0]} with {best_method[2]} words")
        
        return self._clean_academic_text(best_method[1])
    
    def _extract_with_pymupdf(self, file_input, max_pages: int) -> Optional[str]:
        """Extract using PyMuPDF (most robust for academic papers)."""
        try:
            if isinstance(file_input, str):
                doc = fitz.open(file_input)
            else:
                file_input.seek(0)
                doc = fitz.open(stream=file_input.read(), filetype="pdf")
            
            text_parts = []
            for page_num in range(min(len(doc), max_pages)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(self._clean_page_text(text))
            
            doc.close()
            return " ".join(text_parts) if text_parts else None
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return None
    
    def _extract_with_pdfplumber(self, file_input, max_pages: int) -> Optional[str]:
        """Extract using pdfplumber (good for structured content)."""
        try:
            if isinstance(file_input, str):
                pdf = pdfplumber.open(file_input)
            else:
                file_input.seek(0)
                pdf = pdfplumber.open(io.BytesIO(file_input.read()))
            
            text_parts = []
            for page_num in range(min(len(pdf.pages), max_pages)):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(self._clean_page_text(text))
            
            pdf.close()
            return " ".join(text_parts) if text_parts else None
            
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            return None
    
    def _extract_with_pypdf2(self, file_input, max_pages: int) -> Optional[str]:
        """Extract using PyPDF2 (basic fallback)."""
        try:
            if isinstance(file_input, str):
                reader = PdfReader(file_input)
            else:
                file_input.seek(0)
                reader = PdfReader(io.BytesIO(file_input.read()))
            
            text_parts = []
            for page_num in range(min(len(reader.pages), max_pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(self._clean_page_text(text))
            
            return " ".join(text_parts) if text_parts else None
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return None
    
    def _clean_page_text(self, text: str) -> str:
        """Clean text from a single page."""
        # Remove common academic paper noise
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip headers, footers, page numbers
            if (self._is_header_footer(line) or 
                self._is_page_number(line) or
                len(line) < 5):  # Very short lines
                continue
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _is_header_footer(self, line: str) -> bool:
        """Check if line is likely a header or footer."""
        header_indicators = [
            r'^arxiv:\d+\.\d+', r'proceedings of', r'journal of', r'conference on',
            r'vol\.', r'no\.', r'pp\.', r'©', r'http://', r'https://',
            r'^[A-Z][A-Z\s]{10,}$'  # ALL CAPS lines
        ]
        
        line_lower = line.lower()
        return any(re.search(pattern, line_lower) for pattern in header_indicators)
    
    def _is_page_number(self, line: str) -> bool:
        """Check if line is likely a page number."""
        return bool(re.match(r'^\d+$', line.strip()))
    
    def _is_meaningful_text(self, text: str, min_words: int = 50) -> bool:
        """Check if extracted text is meaningful."""
        words = text.split()
        if len(words) < min_words:
            return False
        
        # Check for reasonable word length distribution
        avg_word_len = sum(len(word) for word in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 15:
            return False
            
        return True
    
    def _clean_academic_text(self, text: str) -> str:
        """Final cleaning of academic text."""
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

# Global processor instance
PDF_PROCESSOR = AcademicPDFProcessor()
