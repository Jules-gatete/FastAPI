# Medicine Disposal Prediction API

FastAPI REST API for the Medicine Disposal Prediction System. Provides endpoints for predicting medicine disposal information from generic names or images.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure models are trained:
```bash
python train.py
```

## Running the API

### Option 1: Using the run script
```bash
python run_api.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Using environment variables
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
python run_api.py
```

## API Endpoints

### 1. Health Check
- **GET** `/health`
- Check API health status
- Returns: `{"status": "healthy", "message": "...", "version": "1.0.0"}`

### 2. Text Prediction
- **POST** `/predict/text`
- Predict from text input (medicine generic name)
- Request body:
  ```json
  {
    "medicine_name": "Paracetamol",
    "output_format": "full"  // Optional: "full", "summary", or "json"
  }
  ```
- Returns: `PredictionResponse` with predictions and analysis

### 3. Image Prediction
- **POST** `/predict/image`
- Predict from image file (medicine cover image)
- Form data:
  - `file`: Image file (JPEG, PNG, BMP, TIFF, WEBP)
  - `output_format`: Optional ("full", "summary", or "json")
- Returns: `PredictionResponse` with predictions and analysis

### 4. Flexible Prediction
- **POST** `/predict`
- Accepts either text or image input
- Form data:
  - `medicine_name`: Text input (optional)
  - `file`: Image file (optional)
  - `output_format`: Optional ("full", "summary", or "json")
- Note: Provide either `medicine_name` OR `file`, not both

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage

### Using cURL

#### Text Prediction
```bash
curl -X POST "http://localhost:8000/predict/text" \
  -H "Content-Type: application/json" \
  -d '{
    "medicine_name": "Paracetamol",
    "output_format": "full"
  }'
```

#### Image Prediction
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@medicine_image.jpeg" \
  -F "output_format=full"
```

### Using Python Requests

```python
import requests

# Text prediction
response = requests.post(
    "http://localhost:8000/predict/text",
    json={
        "medicine_name": "Paracetamol",
        "output_format": "full"
    }
)
result = response.json()
print(result)

# Image prediction
with open("medicine_image.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": f},
        data={"output_format": "full"}
    )
result = response.json()
print(result)
```

### Using JavaScript/Fetch

```javascript
// Text prediction
fetch('http://localhost:8000/predict/text', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    medicine_name: 'Paracetamol',
    output_format: 'full'
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Image prediction
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('output_format', 'full');

fetch('http://localhost:8000/predict/image', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Format

All prediction endpoints return a `PredictionResponse` object:

```json
{
  "success": true,
  "medicine_name": "Paracetamol",
  "input_type": "text",
  "predictions": {
    "dosage_form": [
      {"value": "Tablets", "confidence": 0.85}
    ],
    "manufacturer": [
      {"value": "ABC Pharma", "confidence": 0.72}
    ],
    "disposal_category": {
      "value": "Solids, Semisolids, Powders",
      "confidence": 0.90
    },
    "method_of_disposal": [
      {"value": "Landfill", "confidence": 0.75}
    ],
    "handling_method": "...",
    "disposal_remarks": "..."
  },
  "analysis": "Complete analysis text...",
  "messages": ["Successfully processed"],
  "errors": []
}
```

## Output Formats

- **`full`**: Complete detailed analysis with all sections
- **`summary`**: Short summary of key predictions
- **`json`**: Structured JSON output with all predictions

## Error Handling

The API returns appropriate HTTP status codes:
- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error
- **503**: Service Unavailable (models not loaded)

## CORS

CORS is enabled for all origins by default. For production, update the `allow_origins` in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify allowed origins
    ...
)
```

## Production Deployment

For production deployment:

1. Disable auto-reload:
```python
uvicorn.run("api:app", reload=False)
```

2. Use a production ASGI server (e.g., Gunicorn with Uvicorn workers):
```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

3. Set up proper logging and monitoring

4. Use environment variables for configuration

5. Set up HTTPS/SSL certificates

6. Configure rate limiting

7. Update CORS settings for security

## Troubleshooting

### Models not found
- Ensure you've run `python train.py` to generate models
- Check that the `models/` directory exists

### OCR not working
- Install EasyOCR: `pip install easyocr`
- Ensure NumPy version is < 2.0: `pip install 'numpy<2.0'`

### Import errors
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+)

## License

Same as the main project.


