FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Copy the start script and give it permission
COPY start.sh .
RUN chmod +x start.sh

# Use entrypoint
ENTRYPOINT ["./start.sh"]
