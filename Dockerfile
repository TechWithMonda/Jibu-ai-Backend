# Use Python 3.9 slim-buster for better compatibility with system packages
FROM python:3.9-slim-buster

# Install system dependencies including Tesseract
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-swa \
    # Additional dependencies often needed for Python image processing
    libgl1 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Collect static files (if using Django)
RUN python manage.py collectstatic --noinput

# Run Gunicorn
CMD ["gunicorn", "jibu_backend.wsgi:application", "--bind", "0.0.0.0:8000"]