# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Tesseract)
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for pytesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Command to run the application
CMD ["gunicorn", "jibu_backend.wsgi:application"]