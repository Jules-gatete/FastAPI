#!/usr/bin/env python3
"""
Run the FastAPI server for Medicine Disposal Prediction API.
"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print("=" * 60)
    print("Medicine Disposal Prediction API")
    print("=" * 60)
    print(f"Starting server on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Alternative docs: http://{host}:{port}/redoc")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    # Run the server
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )


