FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Tesseract path
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Use PORT environment variable (Railway provides this automatically)
EXPOSE $PORT

# Correct CMD instruction
CMD ["sh", "-c", "gunicorn jibu_backend.wsgi:application --bind 0.0.0.0:${PORT}"]