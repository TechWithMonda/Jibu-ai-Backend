FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set default port if not provided
ENV PORT=8000

# Use shell form to ensure variable substitution
CMD gunicorn jibu_backend.wsgi:application --bind 0.0.0.0:$PORT