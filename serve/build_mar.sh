#!/bin/bash
# Build .mar archive for TorchServe
#
# Prerequisites:
#   pip install torch-model-archiver
#
# Usage:
#   ./serve/build_mar.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Building TorchServe Model Archive ==="

# Step 1: Export model to TorchScript
echo ""
echo "Step 1: Exporting model to TorchScript..."
python serve/export_model.py \
    --model_path models/sentiment_model \
    --output_path serve/model.pt

# Step 2: Create model-store directory
echo ""
echo "Step 2: Creating model-store directory..."
mkdir -p serve/model-store

# Step 3: Create .mar archive
echo ""
echo "Step 3: Creating .mar archive..."
torch-model-archiver \
    --model-name sentiment \
    --version 1.0 \
    --serialized-file serve/model.pt \
    --handler serve/handler.py \
    --extra-files "serve/preprocessor.json,serve/index_to_name.json" \
    --export-path serve/model-store \
    --force

echo ""
echo "=== Build Complete ==="
echo "Model archive: serve/model-store/sentiment.mar"
ls -lh serve/model-store/sentiment.mar

echo ""
echo "Next steps:"
echo "  1. Build Docker image:"
echo "     docker build -t sentiment-serve:v1 -f serve/Dockerfile ."
echo ""
echo "  2. Run container:"
echo "     docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 --name sentiment-server sentiment-serve:v1"
echo ""
echo "  3. Test prediction:"
echo "     curl -X POST http://localhost:8080/predictions/sentiment -H 'Content-Type: application/json' -d '{\"text\": \"Great movie!\"}'"
