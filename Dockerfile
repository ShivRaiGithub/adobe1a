FROM --platform=linux/amd64 python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to use Docker cache efficiently
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and training data
COPY . .

# Ensure expected folders exist (even though COPY brings them in)
RUN mkdir -p input output model

# Train the model during build
RUN python train.py

# Set default command when container runs
CMD ["python", "test.py"]
