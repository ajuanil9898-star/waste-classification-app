# Use official TensorFlow image (matches Keras properly)
FROM tensorflow/tensorflow:2.20.0

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install additional dependencies (FastAPI etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD uvicorn app.app:app --host 0.0.0.0 --port 8000