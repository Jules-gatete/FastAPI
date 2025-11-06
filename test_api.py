#!/usr/bin/env python3
"""
Test script for the Medicine Disposal Prediction API.
Run this after starting the API server to test endpoints.
"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("\n" + "=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("   Make sure the server is running: python run_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_text_prediction():
    """Test text prediction endpoint."""
    print("\n" + "=" * 60)
    print("Testing Text Prediction Endpoint")
    print("=" * 60)
    
    test_cases = [
        {"medicine_name": "Paracetamol", "output_format": "full"},
        {"medicine_name": "Dapagliflozin", "output_format": "summary"},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['medicine_name']} ---")
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/text",
                json=test_case
            )
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success: {result.get('success')}")
                print(f"  Medicine Name: {result.get('medicine_name')}")
                print(f"  Input Type: {result.get('input_type')}")
                
                if result.get('predictions'):
                    preds = result['predictions']
                    if 'dosage_form' in preds:
                        print(f"  Dosage Form: {preds['dosage_form'][0]['value'] if preds['dosage_form'] else 'N/A'}")
                    if 'disposal_category' in preds:
                        print(f"  Disposal Category: {preds['disposal_category'].get('value', 'N/A')}")
                
                if result.get('errors'):
                    print(f"  ⚠ Errors: {result['errors']}")
            else:
                print(f"❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_image_prediction():
    """Test image prediction endpoint."""
    print("\n" + "=" * 60)
    print("Testing Image Prediction Endpoint")
    print("=" * 60)
    
    # Check if test image exists
    test_image = "medicine_image.jpeg"
    import os
    if not os.path.exists(test_image):
        print(f"⚠ Skipping: Test image '{test_image}' not found")
        return
    
    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image, f, "image/jpeg")}
            data = {"output_format": "full"}
            
            response = requests.post(
                f"{API_BASE_URL}/predict/image",
                files=files,
                data=data
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success: {result.get('success')}")
                print(f"  Medicine Name: {result.get('medicine_name')}")
                print(f"  Input Type: {result.get('input_type')}")
                
                if result.get('predictions'):
                    preds = result['predictions']
                    if 'dosage_form' in preds:
                        print(f"  Dosage Form: {preds['dosage_form'][0]['value'] if preds['dosage_form'] else 'N/A'}")
                    if 'disposal_category' in preds:
                        print(f"  Disposal Category: {preds['disposal_category'].get('value', 'N/A')}")
                
                if result.get('errors'):
                    print(f"  ⚠ Errors: {result['errors']}")
            else:
                print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Medicine Disposal Prediction API - Test Suite")
    print("=" * 60)
    print(f"\nTesting API at: {API_BASE_URL}")
    print("Make sure the API server is running: python run_api.py")
    
    # Test health check
    if not test_health():
        print("\n❌ Health check failed. Please start the API server first.")
        sys.exit(1)
    
    # Test text prediction
    test_text_prediction()
    
    # Test image prediction
    test_image_prediction()
    
    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)
    print("\nFor interactive API documentation, visit:")
    print(f"  - Swagger UI: {API_BASE_URL}/docs")
    print(f"  - ReDoc: {API_BASE_URL}/redoc")
    print()

if __name__ == "__main__":
    main()


