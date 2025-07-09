
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \  # For English (add others like `tesseract-ocr-fra` for French)
    && rm -rf /var/lib/apt/lists/*