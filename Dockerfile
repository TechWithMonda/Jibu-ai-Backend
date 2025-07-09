FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Set default port
ENV PORT=8000

# Install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Solution 1 (choose one):
CMD ["/bin/sh", "-c", "gunicorn jibu_backend.wsgi:application --bind 0.0.0.0:$PORT"]

# OR Solution 2:
# COPY start.sh .
# RUN chmod +x start.sh
# CMD ["./start.sh"]