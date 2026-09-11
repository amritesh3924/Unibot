"""
PDF processor for UniBot
Extracts text content from PDF files found on college websites
Supports both text-based and scanned PDFs using Tesseract OCR
"""
from typing import Optional, Dict
import requests
from io import BytesIO
import PyPDF2
import pdfplumber
import os
import re
import warnings
import sys
from contextlib import contextmanager

# Try to import Tesseract OCR (optional dependency)
try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# Try to import pdf2image for converting PDF pages to images
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Suppress PDF parsing warnings (invalid color values, etc.)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*gray.*color.*')
warnings.filterwarnings('ignore', message='.*invalid float value.*')

# Context manager to suppress stderr for PDF processing
@contextmanager
def suppress_stderr():
    """Suppress stderr output temporarily"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old_stderr

class PDFProcessor:
    """Process PDF files and extract text content"""
    
    @staticmethod
    def is_pdf_url(url: str) -> bool:
        """Check if URL points to a PDF file"""
        url_lower = url.lower()
        return url_lower.endswith('.pdf') or 'application/pdf' in url_lower
    
    @staticmethod
    def download_pdf(url: str, timeout: int = 15) -> tuple[Optional[BytesIO], Optional[str]]:
        """
        Download PDF from URL with timeout.

        Returns:
            (BytesIO_or_None, failure_reason_or_None). failure_reason is only
            set when the first element is None, and is one of:
            'timeout', 'http_error_<code>', 'download_error',
            'not_pdf_content_type', 'invalid_pdf_magic_bytes'.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)

            if response.status_code != 200:
                return None, f'http_error_{response.status_code}'

            # Check if it's actually a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                return None, 'not_pdf_content_type'

            # Quick validation - check if it starts with PDF magic bytes
            content = response.content
            if len(content) < 4 or not content[:4].startswith(b'%PDF'):
                return None, 'invalid_pdf_magic_bytes'

            return BytesIO(content), None
        except requests.Timeout:
            return None, 'timeout'
        except Exception:
            return None, 'download_error'
    
    @staticmethod
    def extract_text_pypdf2(pdf_bytes: BytesIO) -> str:
        """Extract text using PyPDF2"""
        try:
            pdf_bytes.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception:
            # Silently fail - errors are handled at higher level
            return ""
    
    @staticmethod
    def extract_text_pdfplumber(pdf_bytes: BytesIO, max_pages: int = 100) -> str:
        """Extract text using pdfplumber (better for complex PDFs) with page limit"""
        try:
            pdf_bytes.seek(0)
            text = ""
            # Suppress warnings and stderr during PDF processing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with suppress_stderr():
                    with pdfplumber.open(pdf_bytes) as pdf:
                        # Limit pages to prevent hanging on huge PDFs
                        pages_to_process = min(len(pdf.pages), max_pages)
                        for i, page in enumerate(pdf.pages[:pages_to_process]):
                            try:
                                # Try extracting text with layout preservation
                                with suppress_stderr():
                                    page_text = page.extract_text(layout=True)
                                if not page_text:
                                    # Fallback to simple extraction
                                    with suppress_stderr():
                                        page_text = page.extract_text()
                                
                                if page_text:
                                    # Clean up the text
                                    page_text = PDFProcessor._clean_text(page_text)
                                    if page_text:
                                        text += page_text + "\n"
                            except Exception:
                                # Skip problematic pages
                                continue
            return text.strip()
        except Exception:
            # Silently fail - errors are handled at higher level
            return ""
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text to improve quality"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove single character lines that are likely artifacts
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep lines that are meaningful (more than 2 chars, or contain numbers/letters)
            if len(line) > 2 or (line and any(c.isalnum() for c in line)):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    @staticmethod
    def _is_garbled_text(text: str) -> bool:
        """
        Detect if text is garbled (character-by-character extraction or poor quality)
        
        Returns True if text appears to be garbled/low quality
        """
        if not text or len(text) < 50:
            return True
        
        # Check for excessive single-character words (sign of garbled extraction)
        words = text.split()
        if len(words) < 10:
            return True
        
        single_char_words = sum(1 for word in words if len(word) == 1)
        if len(words) > 0 and single_char_words / len(words) > 0.3:  # More than 30% single chars
            return True
        
        # Check for excessive whitespace between characters (another sign of garbled text)
        if re.search(r'\s+[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]', text):
            # Pattern like "a b c d" suggests character-by-character extraction
            if text.count(' ') > len(text) * 0.4:  # More than 40% spaces
                return True
        
        # Check for very low word-to-character ratio (suggests poor extraction)
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 2.0:  # Average word less than 2 chars
                return True
        
        # Check for readable text patterns (good sign)
        # Look for common words or patterns
        common_patterns = [
            r'\bthe\b', r'\band\b', r'\bis\b', r'\bto\b', r'\bof\b',
            r'\b[a-z]{3,}\s+[a-z]{3,}',  # Two words of 3+ chars
            r'\d+',  # Numbers
        ]
        
        pattern_matches = sum(1 for pattern in common_patterns if re.search(pattern, text.lower()))
        if pattern_matches < 2:  # Very few common patterns found
            return True
        
        return False
    
    @staticmethod
    def _calculate_text_quality_score(text: str) -> float:
        """
        Calculate a quality score for extracted text (0.0 to 1.0)
        Higher score = better quality
        """
        if not text or len(text) < 50:
            return 0.0
        
        score = 1.0
        
        # Penalize excessive single-character words
        words = text.split()
        if len(words) > 0:
            single_char_ratio = sum(1 for word in words if len(word) == 1) / len(words)
            score -= single_char_ratio * 0.5  # Penalty up to 0.5
        
        # Penalize excessive whitespace
        space_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
        if space_ratio > 0.3:
            score -= (space_ratio - 0.3) * 0.5  # Penalty for excessive spaces
        
        # Reward for common words and patterns
        common_words = ['the', 'and', 'is', 'to', 'of', 'in', 'for', 'a', 'an']
        text_lower = text.lower()
        found_common = sum(1 for word in common_words if word in text_lower)
        score += min(found_common / 10.0, 0.2)  # Bonus up to 0.2
        
        # Reward for longer words (indicates better extraction)
        if len(words) > 0:
            avg_word_len = sum(len(word) for word in words) / len(words)
            if avg_word_len > 4:
                score += 0.1  # Bonus for longer average words
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    @staticmethod
    def extract_text(url: str) -> Dict[str, object]:
        """
        Extract text from PDF URL with quality checks

        Always returns a dict (never None) so callers - and the scrape
        tracker - can tell success from failure, and *why* it failed:

            {'success': True, 'content': ..., 'title': ..., 'type': 'pdf',
             'quality_score': ..., 'page_count': ..., 'failure_reason': None}

            {'success': False, 'content': '', 'title': ..., 'type': 'pdf',
             'quality_score': None, 'page_count': 0,
             'failure_reason': '<see taxonomy below>'}

        failure_reason values: 'timeout', 'http_error_<code>',
        'download_error', 'not_pdf_content_type', 'invalid_pdf_magic_bytes',
        'no_text_extracted' (nothing readable even after OCR fallback),
        'low_quality_score' (extracted but scored < 0.3, likely garbled).
        """
        title = os.path.basename(url).replace('.pdf', '').replace('_', ' ').replace('-', ' ')

        def _failure(reason: str) -> Dict[str, object]:
            return {
                'url': url,
                'content': '',
                'title': title,
                'type': 'pdf',
                'quality_score': None,
                'page_count': 0,
                'success': False,
                'failure_reason': reason
            }

        pdf_bytes, download_failure_reason = PDFProcessor.download_pdf(url)
        if not pdf_bytes:
            return _failure(download_failure_reason or 'download_error')
        
        # Try pdfplumber first (better quality)
        text = PDFProcessor.extract_text_pdfplumber(pdf_bytes)
        used_pdfplumber = True
        page_count = 0
        
        # Get page count
        try:
            pdf_bytes.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with suppress_stderr():
                    with pdfplumber.open(pdf_bytes) as pdf:
                        page_count = len(pdf.pages)
        except Exception:
            pass
        
        # Fallback to PyPDF2 if pdfplumber fails or produces poor results
        if not text or len(text) < 50:
            pdf_bytes.seek(0)
            text = PDFProcessor.extract_text_pypdf2(pdf_bytes)
            used_pdfplumber = False
            if not page_count:
                try:
                    pdf_bytes.seek(0)
                    pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                    page_count = len(pdf_reader.pages)
                except Exception:
                    pass
        
        # If still no text, try OCR for scanned PDFs
        if (not text or len(text) < 50) and TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE:
            try:
                pdf_bytes.seek(0)
                ocr_text = PDFProcessor.extract_text_ocr(pdf_bytes, max_pages=10)  # Limit OCR to first 10 pages
                if ocr_text and len(ocr_text) > 50:
                    text = ocr_text
                    print(f"  ✓ Used OCR for scanned PDF")
            except Exception as e:
                print(f"  ⚠ OCR failed: {e}")
        
        if not text or len(text) < 50:
            return _failure('no_text_extracted')
        
        # Clean the text
        text = PDFProcessor._clean_text(text)
        
        # Check if text is garbled - try alternative method if needed
        if PDFProcessor._is_garbled_text(text) and used_pdfplumber:
            # Try PyPDF2 as alternative if pdfplumber gave garbled results
            pdf_bytes.seek(0)
            alt_text = PDFProcessor.extract_text_pypdf2(pdf_bytes)
            alt_text = PDFProcessor._clean_text(alt_text)
            if alt_text and len(alt_text) >= 50 and not PDFProcessor._is_garbled_text(alt_text):
                text = alt_text
                used_pdfplumber = False
        
        # Calculate quality score
        quality_score = PDFProcessor._calculate_text_quality_score(text)
        
        # Filter out very low quality extractions (quality score < 0.3)
        if quality_score < 0.3:
            return _failure('low_quality_score')
        
        return {
            'url': url,
            'content': text,
            'title': title,
            'type': 'pdf',
            'quality_score': quality_score,
            'page_count': page_count,
            'success': True,
            'failure_reason': None
        }
    
    @staticmethod
    def extract_text_ocr(pdf_bytes: BytesIO, max_pages: int = 10) -> str:
        """
        Extract text from scanned PDF using Tesseract OCR
        
        Args:
            pdf_bytes: PDF file as BytesIO
            max_pages: Maximum number of pages to process (OCR is slow)
            
        Returns:
            Extracted text string
        """
        if not TESSERACT_AVAILABLE or not PDF2IMAGE_AVAILABLE:
            return ""
        
        try:
            pdf_bytes.seek(0)
            # Convert PDF pages to images
            images = convert_from_bytes(pdf_bytes.read(), first_page=1, last_page=max_pages, dpi=300)
            
            text_parts = []
            for i, image in enumerate(images):
                try:
                    # Perform OCR on the image
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"  ⚠ OCR error on page {i+1}: {e}")
                    continue
            
            return '\n'.join(text_parts).strip()
        except Exception as e:
            print(f"  ⚠ OCR processing error: {e}")
            return ""
    
    @staticmethod
    def process_pdf_url(url: str) -> Dict[str, object]:
        """
        Process PDF URL (alias for extract_text for compatibility)
        
        Returns:
            Dictionary with 'success' bool and either the extracted content
            or a 'failure_reason' - see extract_text() docstring.
        """
        return PDFProcessor.extract_text(url)