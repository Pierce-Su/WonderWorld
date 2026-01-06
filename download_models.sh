#!/bin/bash

# Script to download all required HuggingFace models for WonderWorld
# This helps avoid authentication and network issues during runtime

set -e

echo "=========================================="
echo "WonderWorld Model Download Script"
echo "=========================================="
echo ""

# Check if huggingface-hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

# Set cache directory (optional, defaults to ~/.cache/huggingface)
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
echo "Using cache directory: $CACHE_DIR"
echo ""

# Function to download a model
download_model() {
    local model_id=$1
    local model_name=$2
    
    echo "----------------------------------------"
    echo "Downloading: $model_name"
    echo "Model ID: $model_id"
    echo "----------------------------------------"
    
    huggingface-cli download "$model_id" \
        --local-dir "$CACHE_DIR/hub/models--$(echo $model_id | tr '/' '--')" \
        --local-dir-use-symlinks False || {
        echo "Warning: Failed to download $model_id"
        echo "Trying alternative method..."
        python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('$model_id', cache_dir='$CACHE_DIR')
    print('Downloaded successfully!')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
    }
    
    echo "✓ Completed: $model_name"
    echo ""
}

# Download all models
echo "Starting model downloads..."
echo "This may take 20-30 minutes depending on your connection speed."
echo ""

# 1. Stable Diffusion 2 Inpainting
download_model "sd2-community/stable-diffusion-2-inpainting" "Stable Diffusion 2 Inpainting"

# 2. OneFormer
download_model "shi-labs/oneformer_ade20k_swin_large" "OneFormer (Segmentation)"

# 3. Marigold Depth
download_model "prs-eth/marigold-depth-v1-0" "Marigold Depth"

# 4. Marigold Normals
download_model "prs-eth/marigold-normals-v0-1" "Marigold Normals"

echo "=========================================="
echo "All models downloaded successfully!"
echo "=========================================="
echo ""
echo "Models are cached in: $CACHE_DIR"
echo "You can now run the server without download delays."
echo ""

