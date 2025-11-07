"""
Main prediction interface that combines OCR → Prediction → Analysis.
Handles both text and image inputs gracefully.
"""

import sys
import os

# Import modules with error handling
try:
    from input_handler import process_input, validate_medicine_name
    from analysis_generator import generate_analysis, generate_short_summary, generate_json_analysis
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    # Raise ImportError instead of exiting the interpreter so callers can handle the failure.
    raise ImportError(f"Error importing core modules: {e}")

# Import OCR extractor (optional - may fail if OCR dependencies not available)
try:
    from ocr_extractor import MedicineNameExtractor
    OCR_AVAILABLE = True
except (ImportError, Exception) as e:
    OCR_AVAILABLE = False
    MedicineNameExtractor = None
    # Continue without OCR - text input will still work

# Import prediction functions
try:
    from test import load_models, predict_all
except (ImportError, Exception) as e:
    error_msg = str(e)
    if 'numpy' in error_msg.lower() or 'ComplexWarning' in error_msg:
        print(f"\n❌ NumPy compatibility issue detected!")
        print(f"Error: {error_msg}")
        print("\nTo fix this, run:")
        print("  python fix_numpy.py")
        print("  or")
        print("  pip install 'numpy<2.0' --upgrade")
        print("\nThen try again.")
    else:
        print(f"Error: Could not import prediction functions from test.py")
        print(f"Details: {e}")
        print("Please ensure test.py exists and models are trained.")
    # Raise ImportError so callers (for example api.py) can detect and handle missing prediction functionality
    raise ImportError(f"Error importing prediction functions: {e}")

def predict_from_input(input_data, output_format='full'):
    """
    Complete prediction pipeline: OCR → Prediction → Analysis.
    
    Args:
        input_data: Input can be:
            - String (text medicine name)
            - String (image file path)
            - Image file object
        output_format: 'full', 'summary', or 'json'
    
    Returns:
        Dictionary with:
        - 'success': Whether the entire pipeline succeeded
        - 'medicine_name': Extracted or input medicine name
        - 'input_type': 'text' or 'image'
        - 'predictions': Model predictions
        - 'analysis': Generated analysis
        - 'messages': List of status messages
        - 'errors': List of errors (if any)
    """
    result = {
        'success': False,
        'medicine_name': None,
        'input_type': 'unknown',
        'predictions': None,
        'analysis': None,
        'messages': [],
        'errors': []
    }
    
    # Step 1: Process input (OCR if image, validate if text)
    try:
        # Initialize OCR extractor only if OCR is available
        ocr_extractor = None
        if OCR_AVAILABLE and MedicineNameExtractor:
            try:
                ocr_extractor = MedicineNameExtractor()
            except Exception as e:
                # OCR initialization failed, check if input is image
                from input_handler import detect_input_type
                detection = detect_input_type(input_data)
                if detection['type'] == 'image':
                    result['errors'].append(f"OCR initialization failed: {str(e)}")
                    result['errors'].append("Please install compatible dependencies: pip install 'numpy<2.0' easyocr")
                    result['input_type'] = 'image'
                    return result
                # If text input, continue without OCR
                result['messages'].append("OCR not available, but text input will work")
        else:
            # Check if input is an image when OCR not available
            from input_handler import detect_input_type
            detection = detect_input_type(input_data)
            if detection['type'] == 'image':
                result['errors'].append("Image input detected but OCR is not available.")
                result['errors'].append("Please install EasyOCR: pip install easyocr")
                result['errors'].append("Or downgrade NumPy: pip install 'numpy<2.0'")
                result['input_type'] = 'image'
                return result
        
        input_result = process_input(input_data, ocr_extractor=ocr_extractor)
        
        result['input_type'] = input_result['input_type']
        result['messages'].append(input_result['message'])
        
        if not input_result['success']:
            result['errors'].append(f"Input processing failed: {input_result['message']}")
            return result
        
        medicine_name = input_result['medicine_name']
        
        # Validate and normalize medicine name
        validation = validate_medicine_name(medicine_name)
        if not validation['valid']:
            result['errors'].append(f"Medicine name validation failed: {validation['message']}")
            result['messages'].append(f"Extracted name: {medicine_name}")
            if validation.get('normalized_name'):
                result['messages'].append(f"Normalized name: {validation['normalized_name']}")
            return result
        
        # Use normalized name
        medicine_name = validation.get('normalized_name', medicine_name)
        result['medicine_name'] = medicine_name
        if validation.get('normalized_name') and validation['normalized_name'] != input_result.get('medicine_name'):
            result['messages'].append(f"Normalized medicine name: {medicine_name} (from: {input_result.get('medicine_name', 'N/A')})")
        else:
            result['messages'].append(f"Validated medicine name: {medicine_name}")
        
    except Exception as e:
        result['errors'].append(f"Input processing error: {str(e)}")
        return result
    
    # Step 2: Load models and make predictions
    try:
        models = load_models()
        predictions = predict_all(models, medicine_name)
        
        result['predictions'] = predictions
        result['messages'].append("Predictions generated successfully")
        
    except Exception as e:
        result['errors'].append(f"Prediction error: {str(e)}")
        return result
    
    # Step 3: Generate analysis
    try:
        if output_format == 'summary':
            analysis = generate_short_summary(medicine_name, predictions)
        elif output_format == 'json':
            analysis = generate_json_analysis(medicine_name, predictions)
        else:  # 'full'
            analysis = generate_analysis(medicine_name, predictions)
        
        result['analysis'] = analysis
        result['messages'].append("Analysis generated successfully")
        result['success'] = True
        
    except Exception as e:
        result['errors'].append(f"Analysis generation error: {str(e)}")
        return result
    
    # Sanitize result so it contains only JSON-serializable native Python types
    try:
        sanitized = _sanitize_for_serialization(result)
        return sanitized
    except Exception:
        return result


def _sanitize_for_serialization(obj):
    """Recursively convert numpy types and other non-serializable objects to native Python types."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    # Scalars
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()

    # numpy arrays
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()

    # dict
    if isinstance(obj, dict):
        return { _sanitize_for_serialization(k): _sanitize_for_serialization(v) for k,v in obj.items() }

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [ _sanitize_for_serialization(v) for v in obj ]

    # bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            return str(obj)

    # other objects - fallback to str for unknown non-serializable
    try:
        # builtin types (int, float, str, bool, None) are returned unchanged
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
    except Exception:
        pass

    # Fallback: try to convert to primitive via __str__
    try:
        return str(obj)
    except Exception:
        return None

def display_result(result):
    """Display prediction result in a user-friendly format."""
    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    
    if result['success']:
        print(f"\n✓ Successfully processed {result['input_type']} input")
        print(f"  Medicine Name: {result['medicine_name']}")
        
        # Display predictions
        if result['predictions']:
            print("\n" + "-" * 80)
            print("PREDICTIONS:")
            print("-" * 80)
            
            # Dosage Form
            if result['predictions'].get('dosage_form'):
                print("\n📋 Dosage Form (Top 3):")
                for i, pred in enumerate(result['predictions']['dosage_form'][:3], 1):
                    print(f"  {i}. {pred['value'].title():<50} ({pred['confidence']:.2%})")
            
            # Manufacturer
            if result['predictions'].get('manufacturer'):
                print("\n🏭 Manufacturer (Top 3):")
                for i, pred in enumerate(result['predictions']['manufacturer'][:3], 1):
                    print(f"  {i}. {pred['value'].title():<50} ({pred['confidence']:.2%})")
            
            # Disposal Category
            if result['predictions'].get('disposal_category'):
                cat = result['predictions']['disposal_category']
                print(f"\n🗑️  Disposal Category: {cat['value'].title()} ({cat['confidence']:.2%})")
        
        # Display analysis
        if result['analysis']:
            print("\n" + "-" * 80)
            print("COMPLETE ANALYSIS:")
            print("-" * 80)
            print(result['analysis'])
    else:
        print("\n✗ Processing failed")
        if result['errors']:
            print("\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        if result['messages']:
            print("\nMessages:")
            for msg in result['messages']:
                print(f"  - {msg}")
    
    print("\n" + "=" * 80)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict medicine disposal information from text or image input'
    )
    parser.add_argument(
        'input',
        help='Input: medicine name (text) or image file path'
    )
    parser.add_argument(
        '--format',
        choices=['full', 'summary', 'json'],
        default='full',
        help='Output format: full (default), summary, or json'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (shortcut for --format json)'
    )
    
    args = parser.parse_args()
    
    # Handle JSON flag
    if args.json:
        args.format = 'json'
    
    # Process input
    result = predict_from_input(args.input, output_format=args.format)
    
    # Display or output result
    if args.format == 'json':
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        display_result(result)

if __name__ == '__main__':
    main()

