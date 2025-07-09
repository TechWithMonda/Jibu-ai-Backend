# Add this to your Dockerfile before installing Python dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*