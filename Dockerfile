FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Compiler and build tools
    gcc \
    python3-dev \
    # Pillow dependencies
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    libwebp-dev \
    # Tesseract OCR
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    poppler-utils \
    # Other utilities
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - IMPORTANT ORDER
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pillow==9.5.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Copy the start script and give it permission
COPY start.sh .
RUN chmod +x start.sh

# Use entrypoint
ENTRYPOINT ["./start.sh"]