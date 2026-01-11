# TorchServe Dockerfile for Sentiment Classifier
#
# Build:
#   docker build -t sentiment-serve:v1 -f serve/Dockerfile .
#
# Run:
#   docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 --name sentiment-server sentiment-serve:v1
#
# Test:
#   curl -X POST http://localhost:8080/predictions/sentiment \
#     -H "Content-Type: application/json" \
#     -d '{"text": "This movie was fantastic!"}'

FROM pytorch/torchserve:latest-cpu

LABEL maintainer="sentiment-classifier"
LABEL description="TorchServe container for sentiment classification"

# Switch to root to install dependencies
USER root

# Create model-store directory
RUN mkdir -p /home/model-server/model-store

# Copy the .mar archive
COPY serve/model-store/sentiment.mar /home/model-server/model-store/

# Copy config
COPY serve/config.properties /home/model-server/config.properties

# Set permissions
RUN chown -R model-server:model-server /home/model-server

# Switch back to model-server user
USER model-server

# Expose ports
# 8080 - Inference API
# 8081 - Management API
# 8082 - Metrics API
EXPOSE 8080 8081 8082

# Start TorchServe with config
CMD ["torchserve", \
     "--start", \
     "--model-store", "/home/model-server/model-store", \
     "--ts-config", "/home/model-server/config.properties", \
     "--foreground"]
