"""
OCR module for extracting medicine names from medicine cover images.
Uses EasyOCR for text extraction with image preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress PyTorch warnings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
try:
    import torch
    torch.set_warn_always(False)
    # Suppress the specific cadam32bit_grad_fp32 warning
    import logging
    logging.getLogger('torch').setLevel(logging.ERROR)
except:
    pass

# Lazy loading - only import when actually needed
USE_EASYOCR = False
USE_TESSERACT = False

def _try_import_easyocr():
    """Try to import EasyOCR, return True if successful."""
    global USE_EASYOCR
    try:
        import easyocr
        USE_EASYOCR = True
        return True
    except (ImportError, Exception):
        USE_EASYOCR = False
        return False

def _try_import_tesseract():
    """Try to import Tesseract, return True if successful."""
    global USE_TESSERACT
    try:
        import pytesseract
        USE_TESSERACT = True
        return True
    except (ImportError, Exception):
        USE_TESSERACT = False
        return False

class MedicineNameExtractor:
    """Extract medicine names from medicine cover images using OCR."""
    
    def __init__(self):
        """Initialize OCR reader."""
        self.reader = None
        
        # Try to import EasyOCR first
        if _try_import_easyocr():
            try:
                import easyocr
                print("Initializing EasyOCR...")
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("EasyOCR initialized successfully")
                return
            except Exception as e:
                print(f"Warning: EasyOCR initialization failed: {e}")
                self.reader = None
        
        # Try Tesseract as fallback
        if _try_import_tesseract():
            try:
                import pytesseract
                print("Using Tesseract OCR")
                self.reader = 'tesseract'
                return
            except Exception as e:
                print(f"Warning: Tesseract initialization failed: {e}")
        
        # No OCR available
        raise RuntimeError(
            "No OCR engine available. Please install EasyOCR: pip install easyocr\n"
            "Note: If you have NumPy 2.0, you may need to downgrade: pip install 'numpy<2.0'"
        )
    
    def preprocess_image(self, image_path, method='auto'):
        """
        Preprocess image for better OCR accuracy with multiple methods.
        
        Args:
            image_path: Path to image file
            method: 'auto', 'original', 'processed', or 'both'
        
        Returns:
            Preprocessed image(s) - dict with 'original' and/or 'processed'
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        result = {}
        
        if method in ['auto', 'original', 'both']:
            # Keep original for EasyOCR (works better with original)
            result['original'] = img
        
        if method in ['auto', 'processed', 'both']:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize if image is too small (helps OCR)
            h, w = gray.shape
            if h < 300 or w < 300:
                scale = max(300 / h, 300 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Denoise (try multiple methods)
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Enhance contrast (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Adaptive threshold (better for varying lighting)
            binary_adaptive = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Also try Otsu threshold
            _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Deskew (rotate image if needed)
            coords = np.column_stack(np.where(binary_otsu > 0))
            if len(coords) > 0:
                try:
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                    
                    # Rotate image if angle is significant
                    if abs(angle) > 0.5:
                        (h, w) = binary_otsu.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        binary_otsu = cv2.warpAffine(
                            binary_otsu, M, (w, h), 
                            flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        binary_adaptive = cv2.warpAffine(
                            binary_adaptive, M, (w, h), 
                            flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_REPLICATE
                        )
                except:
                    pass  # Skip deskew if it fails
            
            # Return the better-looking binary (adaptive usually works better)
            result['processed'] = binary_adaptive
        
        return result if len(result) > 1 else (result['processed'] if 'processed' in result else result['original'])
    
    def correct_ocr_errors(self, text):
        """
        Correct common OCR errors in medicine names.
        
        Args:
            text: Text string from OCR
        
        Returns:
            Corrected text string
        """
        if not text:
            return text
        
        # Common OCR error corrections
        corrections = {
            # Common character misreadings
            'Tablel': 'Tablet',
            'Tablets': 'Tablets',
            'Tabl': 'Tablet',
            'Tab1et': 'Tablet',
            'Tab1ets': 'Tablets',
            'Capsu1e': 'Capsule',
            'Capsu1es': 'Capsules',
            'Injection': 'Injection',
            'Injectab1e': 'Injectable',
            # Common word separations
            'D A P A G L I F L O Z I N': 'Dapagliflozin',
            'DAPAGLIFLOZIN': 'Dapagliflozin',
            # Common lowercase/uppercase issues
            'ixio': 'Ixio',
            'IXIO': 'Ixio',
        }
        
        # Apply corrections (case-insensitive)
        corrected = text
        for error, correction in corrections.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(error), re.IGNORECASE)
            corrected = pattern.sub(correction, corrected)
        
        # Fix common patterns: "Tablel" -> "Tablet" (if followed by s, make it Tablets)
        corrected = re.sub(r'Tablel(?=\s|$)', 'Tablet', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'Tablets', 'Tablets', corrected, flags=re.IGNORECASE)
        
        # Fix spacing issues (multiple spaces, weird characters)
        corrected = re.sub(r'\s+', ' ', corrected)  # Multiple spaces to single
        corrected = re.sub(r'[^\w\s,.-]', '', corrected)  # Remove weird characters except common ones
        corrected = corrected.strip()
        
        return corrected
    
    def extract_text_easyocr(self, image_path):
        """Extract text using EasyOCR with improved preprocessing."""
        import easyocr
        
        # EasyOCR works better with original image, so use original for reading
        results = self.reader.readtext(image_path, paragraph=False, detail=1)
        
        # Also try with preprocessed image for comparison
        preprocessed = self.preprocess_image(image_path, method='processed')
        if isinstance(preprocessed, dict):
            preprocessed = preprocessed.get('processed', None)
        
        if preprocessed is not None:
            # Convert binary to 3-channel for EasyOCR
            preprocessed_3ch = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            results_processed = self.reader.readtext(preprocessed_3ch, paragraph=False, detail=1)
            
            # Merge results, preferring higher confidence
            all_results = {}
            for (bbox, text, confidence) in results:
                key = text.lower().strip()
                if key not in all_results or confidence > all_results[key]['confidence']:
                    all_results[key] = {'bbox': bbox, 'text': text, 'confidence': confidence}
            
            for (bbox, text, confidence) in results_processed:
                key = text.lower().strip()
                if key not in all_results or confidence > all_results[key]['confidence']:
                    all_results[key] = {'bbox': bbox, 'text': text, 'confidence': confidence}
            
            results = [(v['bbox'], v['text'], v['confidence']) for v in all_results.values()]
        
        # Extract all text with confidence scores
        extracted_texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Increased threshold for better quality
                # Correct common OCR errors
                corrected_text = self.correct_ocr_errors(text.strip())
                if corrected_text and len(corrected_text) > 1:
                    extracted_texts.append({
                        'text': corrected_text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'original': text.strip()  # Keep original for debugging
                    })
        
        return extracted_texts
    
    def extract_text_tesseract(self, image_path):
        """Extract text using Tesseract OCR with improved preprocessing."""
        import pytesseract
        
        # Preprocess image
        processed = self.preprocess_image(image_path, method='processed')
        if isinstance(processed, dict):
            processed = processed.get('processed', None)
        if processed is None:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed = gray
        
        # Try multiple PSM modes for better results
        psm_modes = [6, 11, 7]  # Single uniform block, Sparse text, Single text line
        best_text = ""
        best_confidence = 0.0
        
        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-')
                if text and len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue
        
        # Split into lines and extract with confidence
        lines = best_text.split('\n')
        extracted_texts = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:
                # Correct OCR errors
                corrected = self.correct_ocr_errors(line)
                if corrected:
                    extracted_texts.append({
                        'text': corrected,
                        'confidence': 0.75,  # Tesseract confidence estimation
                        'bbox': None,
                        'original': line
                    })
        
        return extracted_texts
    
    def extract_all_text(self, image_path):
        """
        Extract all text from image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of extracted text segments with confidence scores
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if self.reader is None:
            raise RuntimeError("No OCR engine available. Please install EasyOCR or Tesseract.")
        
        try:
            # Check if reader is EasyOCR Reader instance
            try:
                import easyocr
                if isinstance(self.reader, easyocr.Reader):
                    return self.extract_text_easyocr(image_path)
            except:
                pass
            
            # Check if reader is Tesseract
            if self.reader == 'tesseract':
                return self.extract_text_tesseract(image_path)
            
            raise RuntimeError("OCR engine not properly initialized")
        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {str(e)}")
    
    def identify_medicine_name(self, extracted_texts):
        """
        Identify medicine name from extracted text with improved logic.
        
        Args:
            extracted_texts: List of extracted text segments
        
        Returns:
            Most likely medicine name (string)
        """
        if not extracted_texts:
            return None
        
        # Combine all text
        all_text = ' '.join([item['text'] for item in extracted_texts])
        
        # Common medicine label patterns (improved)
        patterns = [
            r'(?:Generic\s+Name|Active\s+Ingredient|Medicine\s+Name|Drug\s+Name)[:\s]*([A-Za-z][A-Za-z\s,]+?)(?:\n|$|mg|ml|%)',
            r'(?:Name|Ingredient)[:\s]*([A-Za-z][A-Za-z\s,]+?)(?:\n|$|mg|ml|%)',
            r'^([A-Z][A-Za-z\s,]+?)(?:\s*\d+|\s*mg|\s*ml|\s*%|Tablet|Tablets|Capsule|Capsules)',
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*(?:Tablet|Tablets|Capsule|Capsules|Injection|Injectable)',
        ]
        
        # Try to find medicine name using patterns
        for pattern in patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Clean and return the first match
                name = matches[0].strip()
                # Remove trailing dosage form words if they were captured
                name = re.sub(r'\s+(Tablet|Tablets|Capsule|Capsules|Injection|Injectable)$', '', name, flags=re.IGNORECASE)
                if len(name) > 3 and len(name) < 100:  # Reasonable length
                    return name
        
        # Fallback: Use the most prominent text (highest confidence, largest font)
        # Sort by confidence and position (top-left is usually most prominent)
        sorted_texts = sorted(extracted_texts, 
                             key=lambda x: (
                                 x.get('confidence', 0.5), 
                                 -x.get('bbox', [(0, 0)])[0][1] if x.get('bbox') else 0,
                                 len(x.get('text', ''))  # Longer text often more important
                             ), 
                             reverse=True)
        
        # Filter out common label text (not medicine names)
        common_label_words = ['generic', 'name', 'active', 'ingredient', 'medicine', 
                             'drug', 'mg', 'ml', 'strength', 'dosage', 'manufacturer', 
                             'expiry', 'batch', 'lot', 'exp', 'mfg', 'manufactured']
        
        # Medicine name patterns (words that suggest it's a medicine name)
        medicine_indicators = ['tablet', 'tablets', 'capsule', 'capsules', 'injection', 
                              'injectable', 'suspension', 'syrup', 'cream', 'ointment']
        
        for text_item in sorted_texts:
            text = text_item['text'].strip()
            
            # Skip if it's clearly just a label word
            text_lower = text.lower()
            if any(word == text_lower for word in common_label_words):
                continue
            
            # If it contains medicine indicators, extract the medicine name part
            if any(indicator in text_lower for indicator in medicine_indicators):
                # Extract the part before the indicator
                for indicator in medicine_indicators:
                    if indicator in text_lower:
                        parts = re.split(re.escape(indicator), text_lower, flags=re.IGNORECASE, maxsplit=1)
                        if parts[0].strip():
                            candidate = parts[0].strip()
                            # Title case it
                            candidate = ' '.join(word.capitalize() for word in candidate.split())
                            if 3 < len(candidate) < 100:
                                return candidate
            
            # Check if it looks like a medicine name (contains letters, reasonable length)
            # Medicine names typically have capital letters
            if re.search(r'[A-Z][a-z]+', text) and 3 < len(text) < 100:
                # Check if it's not just common words
                words = text.split()
                if len(words) >= 1 and any(len(w) > 2 for w in words):
                    return text
        
        # Last resort: return the highest confidence text, cleaned
        if sorted_texts:
            text = sorted_texts[0]['text'].strip()
            # Remove common suffixes/prefixes
            text = re.sub(r'^(Generic|Active|Ingredient|Name|Drug)\s*:?\s*', '', text, flags=re.IGNORECASE)
            text = text.strip()
            if text and 3 < len(text) < 100:
                return text
        
        return None
    
    def extract_medicine_name(self, image_path):
        """
        Main method: Extract medicine name from image.
        
        Args:
            image_path: Path to medicine cover image
        
        Returns:
            Dictionary with:
            - 'medicine_name': Extracted medicine name (string)
            - 'confidence': Confidence score (float)
            - 'all_text': All extracted text (list)
            - 'success': Whether extraction was successful (bool)
            - 'message': Status message (string)
        """
        try:
            # Extract all text
            extracted_texts = self.extract_all_text(image_path)
            
            if not extracted_texts:
                return {
                    'medicine_name': None,
                    'confidence': 0.0,
                    'all_text': [],
                    'success': False,
                    'message': 'No text found in image. Please ensure the image is clear and contains readable text.'
                }
            
            # Identify medicine name
            medicine_name = self.identify_medicine_name(extracted_texts)
            
            if medicine_name:
                # Calculate average confidence
                avg_confidence = np.mean([item.get('confidence', 0.5) for item in extracted_texts])
                
                return {
                    'medicine_name': medicine_name,
                    'confidence': float(avg_confidence),
                    'all_text': [item['text'] for item in extracted_texts],
                    'success': True,
                    'message': f'Successfully extracted medicine name: {medicine_name}'
                }
            else:
                return {
                    'medicine_name': None,
                    'confidence': 0.0,
                    'all_text': [item['text'] for item in extracted_texts],
                    'success': False,
                    'message': 'Could not identify medicine name from extracted text. Please verify the image contains a medicine name.'
                }
        
        except Exception as e:
            return {
                'medicine_name': None,
                'confidence': 0.0,
                'all_text': [],
                'success': False,
                'message': f'OCR extraction error: {str(e)}'
            }

def extract_medicine_name_from_image(image_path):
    """
    Convenience function to extract medicine name from image.
    
    Args:
        image_path: Path to medicine cover image
    
    Returns:
        Extracted medicine name (string) or None if extraction failed
    """
    extractor = MedicineNameExtractor()
    result = extractor.extract_medicine_name(image_path)
    
    if result['success']:
        return result['medicine_name']
    else:
        print(f"Warning: {result['message']}")
        return None

