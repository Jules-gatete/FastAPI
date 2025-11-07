# ===== Base Image =====
FROM python:3.11-slim

# ===== Set Working Directory =====
WORKDIR /app

# ===== System Dependencies =====
# Install essential packages for OpenCV, torch, and other ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ===== Copy Project Files =====
COPY . .

# ===== Install Python Dependencies =====
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ===== Expose the port (DigitalOcean uses 8080 by default) =====
EXPOSE 8080

# ===== Start FastAPI =====
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
