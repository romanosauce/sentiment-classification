# Sentiment Classifier Docker Image
# Build: docker build -t ml-app:v1 .
# Run:   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ml-app:v1 --input_path /app/input/data.csv --output_path /app/output/preds.csv

FROM python:3.11-slim

LABEL maintainer="sentiment-classifier"
LABEL description="Sentiment classification inference container"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only PyTorch (much smaller than CUDA version)
RUN pip install --no-cache-dir \
    torch && \
    pip install --no-cache-dir \
    numpy==1.26.2 \
    pandas==2.1.3 \
    tqdm==4.66.1

# Copy source code
COPY src/ ./src/

# Copy trained model
COPY models/sentiment_model/ ./models/sentiment_model/

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]

# Default arguments (can be overridden)
CMD ["--help"]
