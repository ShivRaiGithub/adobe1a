FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR / app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training data and scripts
COPY training_pdfs/ ./training_pdfs/
COPY training_jsons/ ./training_jsons/
COPY train.py .
COPY test2.py .

# Train the XGBoost model during build
RUN python train.py

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the default command to run inference
CMD ["python", "test2.py"]