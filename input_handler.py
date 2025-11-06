"""
Input handler for detecting and routing different input types (text, image).
Handles validation and error cases gracefully.
"""

import os
import re
from PIL import Image
import numpy as np

def is_image_file(file_path):
    """
    Check if file is a valid image file.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file is a valid image, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        # Try to open as image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def is_valid_text(text):
    """
    Check if text is a valid medicine name.
    
    Args:
        text: Input text string
    
    Returns:
        True if text looks like a valid medicine name, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Check length (medicine names are typically 3-100 characters)
    if len(text) < 3 or len(text) > 100:
        return False
    
    # Check if it contains at least some letters (not just numbers/symbols)
    if not re.search(r'[A-Za-z]', text):
        return False
    
    # Check if it's not just common words
    common_words = ['image', 'photo', 'picture', 'file', 'upload', 'select', 'choose']
    if text.lower() in common_words:
        return False
    
    return True

def detect_input_type(input_data):
    """
    Detect whether input is text, image path, or invalid.
    
    Args:
        input_data: Input can be:
            - String (text medicine name)
            - String (image file path)
            - Image file object
    
    Returns:
        Dictionary with:
        - 'type': 'text', 'image', or 'invalid'
        - 'data': Processed input data
        - 'message': Status message
    """
    # Handle None or empty input
    if input_data is None:
        return {
            'type': 'invalid',
            'data': None,
            'message': 'Input is None or empty'
        }
    
    # Handle string input
    if isinstance(input_data, str):
        input_data = input_data.strip()
        
        # Check if it's a file path
        if os.path.exists(input_data):
            # Check if it's an image file
            if is_image_file(input_data):
                return {
                    'type': 'image',
                    'data': input_data,
                    'message': f'Detected image file: {input_data}'
                }
            else:
                return {
                    'type': 'invalid',
                    'data': None,
                    'message': f'File exists but is not a valid image: {input_data}'
                }
        
        # Check if it's a valid text (medicine name)
        if is_valid_text(input_data):
            return {
                'type': 'text',
                'data': input_data,
                'message': f'Detected text input: {input_data}'
            }
        else:
            return {
                'type': 'invalid',
                'data': None,
                'message': f'Text input does not appear to be a valid medicine name: {input_data}'
            }
    
    # Handle PIL Image object
    if isinstance(input_data, Image.Image):
        return {
            'type': 'image',
            'data': input_data,
            'message': 'Detected PIL Image object'
        }
    
    # Handle numpy array (image)
    if isinstance(input_data, np.ndarray):
        try:
            # Try to convert to PIL Image
            img = Image.fromarray(input_data)
            return {
                'type': 'image',
                'data': img,
                'message': 'Detected numpy array image'
            }
        except Exception:
            return {
                'type': 'invalid',
                'data': None,
                'message': 'Numpy array could not be converted to image'
            }
    
    # Unknown input type
    return {
        'type': 'invalid',
        'data': None,
        'message': f'Unknown input type: {type(input_data)}'
    }

def process_input(input_data, ocr_extractor=None):
    """
    Process input (text or image) and extract medicine name.
    
    Args:
        input_data: Input (text string or image path/file)
        ocr_extractor: MedicineNameExtractor instance (optional, will create if needed)
    
    Returns:
        Dictionary with:
        - 'medicine_name': Extracted or validated medicine name (string)
        - 'input_type': 'text' or 'image'
        - 'success': Whether processing was successful (bool)
        - 'message': Status message (string)
        - 'confidence': Confidence score if from OCR (float)
    """
    # Detect input type
    detection = detect_input_type(input_data)
    
    if detection['type'] == 'invalid':
        return {
            'medicine_name': None,
            'input_type': 'invalid',
            'success': False,
            'message': detection['message'],
            'confidence': 0.0
        }
    
    # Handle text input
    if detection['type'] == 'text':
        return {
            'medicine_name': detection['data'],
            'input_type': 'text',
            'success': True,
            'message': f'Using text input: {detection["data"]}',
            'confidence': 1.0  # Text input is 100% confident
        }
    
    # Handle image input
    if detection['type'] == 'image':
        if ocr_extractor is None:
            from ocr_extractor import MedicineNameExtractor
            ocr_extractor = MedicineNameExtractor()
        
        # Extract medicine name from image
        image_path = detection['data']
        if isinstance(image_path, Image.Image):
            # Save PIL Image temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image_path.save(temp_file.name)
            image_path = temp_file.name
            # Note: temp file will be cleaned up by system
        
        ocr_result = ocr_extractor.extract_medicine_name(image_path)
        
        if ocr_result['success']:
            # Normalize the extracted medicine name
            medicine_name = ocr_result['medicine_name']
            normalized = normalize_medicine_name(medicine_name)
            
            return {
                'medicine_name': normalized,  # Use normalized version
                'input_type': 'image',
                'success': True,
                'message': f"Successfully extracted and normalized: {normalized} (from: {medicine_name})",
                'confidence': ocr_result['confidence'],
                'original_extracted': medicine_name  # Keep original for reference
            }
        else:
            return {
                'medicine_name': None,
                'input_type': 'image',
                'success': False,
                'message': ocr_result['message'],
                'confidence': ocr_result['confidence']
            }
    
    # Should not reach here
    return {
        'medicine_name': None,
        'input_type': 'invalid',
        'success': False,
        'message': 'Unknown error in input processing',
        'confidence': 0.0
    }

def normalize_medicine_name(medicine_name):
    """
    Normalize medicine name: fix common OCR errors, clean up text.
    
    Args:
        medicine_name: Medicine name string
    
    Returns:
        Normalized medicine name string
    """
    if not medicine_name:
        return medicine_name
    
    # Common OCR error corrections
    corrections = {
        'Tablel': 'Tablet',
        'Tabl': 'Tablet',
        'Tab1et': 'Tablet',
        'Tab1ets': 'Tablets',
        'Capsu1e': 'Capsule',
        'Capsu1es': 'Capsules',
        'Injectab1e': 'Injectable',
    }
    
    normalized = medicine_name.strip()
    
    # Apply corrections (case-insensitive)
    for error, correction in corrections.items():
        pattern = re.compile(re.escape(error), re.IGNORECASE)
        normalized = pattern.sub(correction, normalized)
    
    # Fix "Tablel" -> "Tablet" pattern
    normalized = re.sub(r'Tablel(?=\s|$)', 'Tablet', normalized, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove trailing dosage form words if they're duplicated
    normalized = re.sub(r'\s+(Tablet|Tablets|Capsule|Capsules|Injection|Injectable)\s+(Tablet|Tablets|Capsule|Capsules|Injection|Injectable)', r' \1', normalized, flags=re.IGNORECASE)
    
    # Clean up weird characters (keep letters, numbers, spaces, commas, hyphens, periods)
    normalized = re.sub(r'[^\w\s,.-]', '', normalized)
    
    normalized = normalized.strip()
    
    return normalized

def validate_medicine_name(medicine_name):
    """
    Validate and normalize extracted medicine name before using in prediction.
    
    Args:
        medicine_name: Extracted medicine name
    
    Returns:
        Dictionary with validation result and normalized name
    """
    if not medicine_name:
        return {
            'valid': False,
            'message': 'Medicine name is empty',
            'normalized_name': None
        }
    
    # Normalize the medicine name first
    normalized = normalize_medicine_name(medicine_name)
    
    if not is_valid_text(normalized):
        return {
            'valid': False,
            'message': f'Medicine name does not appear valid: {normalized}',
            'normalized_name': normalized
        }
    
    # Additional checks
    # Check for common OCR errors
    if len(normalized) < 3:
        return {
            'valid': False,
            'message': 'Medicine name is too short (likely OCR error)',
            'normalized_name': normalized
        }
    
    if len(normalized) > 100:
        return {
            'valid': False,
            'message': 'Medicine name is too long (likely extracted too much text)',
            'normalized_name': normalized
        }
    
    # Check if it's mostly special characters
    if not re.search(r'[A-Za-z]{3,}', normalized):
        return {
            'valid': False,
            'message': 'Medicine name does not contain enough letters',
            'normalized_name': normalized
        }
    
    return {
        'valid': True,
        'message': 'Medicine name is valid',
        'normalized_name': normalized
    }


