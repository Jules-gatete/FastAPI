"""
FastAPI application for Medicine Disposal Prediction System.
Provides REST API endpoints for text and image-based medicine predictions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
from pathlib import Path

# Import prediction functions
try:
    from predict import predict_from_input
except ImportError as e:
    print(f"Warning: Could not import predict module: {e}")
    predict_from_input = None

# Initialize FastAPI app
app = FastAPI(
    title="Medicine Disposal Prediction API",
    description="API for predicting medicine disposal information from generic names or images",
    version="1.0.0"
)

# CORS middleware - allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TextPredictionRequest(BaseModel):
    """Request model for text-based predictions."""
    medicine_name: str = Field(..., description="Medicine generic name", min_length=3, max_length=100)
    output_format: Optional[str] = Field("full", description="Output format: 'full', 'summary', or 'json'")

    class Config:
        schema_extra = {
            "example": {
                "medicine_name": "Paracetamol",
                "output_format": "full"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    medicine_name: Optional[str] = None
    input_type: Optional[str] = None
    predictions: Optional[Dict[str, Any]] = None
    analysis: Optional[str] = None
    messages: List[str] = []
    errors: List[str] = []

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "medicine_name": "Paracetamol",
                "input_type": "text",
                "predictions": {
                    "dosage_form": [{"value": "Tablets", "confidence": 0.85}],
                    "manufacturer": [{"value": "ABC Pharma", "confidence": 0.72}],
                    "disposal_category": {"value": "Solids", "confidence": 0.90}
                },
                "analysis": "Complete analysis text...",
                "messages": ["Successfully processed"],
                "errors": []
            }
        }

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    version: str = "1.0.0"

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Medicine Disposal Prediction API",
        "version": "1.0.0",
        "description": "API for predicting medicine disposal information",
        "endpoints": {
            "health": "/health",
            "text_prediction": "/predict/text",
            "image_prediction": "/predict/image",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check if models are available
        models_dir = Path("models")
        if not models_dir.exists():
            return HealthResponse(
                status="error",
                message="Models directory not found. Please run train.py first."
            )
        
        # Check if prediction function is available
        if predict_from_input is None:
            return HealthResponse(
                status="error",
                message="Prediction module not available"
            )
        
        return HealthResponse(
            status="healthy",
            message="API is operational and ready to process requests"
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            message=f"Health check failed: {str(e)}"
        )

@app.post("/predict/text", response_model=PredictionResponse, tags=["Predictions"])
async def predict_from_text(request: TextPredictionRequest):
    """
    Predict medicine disposal information from text input (generic name).
    
    Args:
        request: TextPredictionRequest with medicine_name and optional output_format
    
    Returns:
        PredictionResponse with predictions and analysis
    """
    if predict_from_input is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not available. Please ensure models are trained."
        )
    
    try:
        # Validate output format
        valid_formats = ["full", "summary", "json"]
        output_format = request.output_format.lower() if request.output_format else "full"
        if output_format not in valid_formats:
            output_format = "full"
        
        # Make prediction
        result = predict_from_input(
            input_data=request.medicine_name,
            output_format=output_format
        )

        # Ensure result is JSON-serializable (convert numpy types etc.)
        try:
            result = jsonable_encoder(result)
        except Exception:
            pass
        
        # Convert to response model
        return PredictionResponse(
            success=result.get("success", False),
            medicine_name=result.get("medicine_name"),
            input_type=result.get("input_type"),
            predictions=result.get("predictions"),
            analysis=result.get("analysis"),
            messages=result.get("messages", []),
            errors=result.get("errors", [])
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/image", response_model=PredictionResponse, tags=["Predictions"])
async def predict_from_image(
    file: UploadFile = File(..., description="Medicine image file (JPEG, PNG, etc.)"),
    output_format: str = Form("full", description="Output format: 'full', 'summary', or 'json'")
):
    """
    Predict medicine disposal information from image input (medicine cover image).
    
    Args:
        file: Uploaded image file
        output_format: Output format ('full', 'summary', or 'json')
    
    Returns:
        PredictionResponse with predictions and analysis
    """
    if predict_from_input is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not available. Please ensure models are trained."
        )
    
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    file_extension = Path(file.filename).suffix.lower() if file.filename else ""
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}"
        )
    
    # Validate output format
    valid_formats = ["full", "summary", "json"]
    output_format = output_format.lower() if output_format else "full"
    if output_format not in valid_formats:
        output_format = "full"
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        suffix = file_extension or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Make prediction
        result = predict_from_input(
            input_data=temp_path,
            output_format=output_format
        )

        # Ensure result is JSON-serializable (convert numpy types etc.)
        try:
            result = jsonable_encoder(result)
        except Exception:
            pass
        
        # Convert to response model
        response = PredictionResponse(
            success=result.get("success", False),
            medicine_name=result.get("medicine_name"),
            input_type=result.get("input_type"),
            predictions=result.get("predictions"),
            analysis=result.get("analysis"),
            messages=result.get("messages", []),
            errors=result.get("errors", [])
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.post("/predict", tags=["Predictions"])
async def predict_flexible(
    medicine_name: Optional[str] = Form(None, description="Medicine name (text input)"),
    file: Optional[UploadFile] = File(None, description="Medicine image file"),
    output_format: str = Form("full", description="Output format")
):
    """
    Flexible prediction endpoint that accepts either text or image input.
    
    Args:
        medicine_name: Text input (medicine generic name)
        file: Image file input
        output_format: Output format ('full', 'summary', or 'json')
    
    Returns:
        PredictionResponse with predictions and analysis
    
    Note: Either medicine_name OR file must be provided, not both.
    """
    if predict_from_input is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not available. Please ensure models are trained."
        )
    
    # Validate that exactly one input is provided
    if not medicine_name and not file:
        raise HTTPException(
            status_code=400,
            detail="Either 'medicine_name' (text) or 'file' (image) must be provided"
        )
    
    if medicine_name and file:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'medicine_name' OR 'file', not both"
        )
    
    # Validate output format
    valid_formats = ["full", "summary", "json"]
    output_format = output_format.lower() if output_format else "full"
    if output_format not in valid_formats:
        output_format = "full"
    
    try:
        input_data = None
        temp_file = None
        
        if medicine_name:
            # Text input
            input_data = medicine_name
        else:
            # Image input
            file_extension = Path(file.filename).suffix.lower() if file.filename else ""
            allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}"
                )
            
            # Save uploaded file temporarily
            suffix = file_extension or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                input_data = temp_file.name
        
        # Make prediction
        result = predict_from_input(
            input_data=input_data,
            output_format=output_format
        )

        # Ensure result is JSON-serializable (convert numpy types etc.)
        try:
            result = jsonable_encoder(result)
        except Exception:
            pass
        
        # Convert to response model
        response = PredictionResponse(
            success=result.get("success", False),
            medicine_name=result.get("medicine_name"),
            input_type=result.get("input_type"),
            predictions=result.get("predictions"),
            analysis=result.get("analysis"),
            messages=result.get("messages", []),
            errors=result.get("errors", [])
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    # Use port from environment (DigitalOcean / other platforms set $PORT). Fall back to 8000 for local dev.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


