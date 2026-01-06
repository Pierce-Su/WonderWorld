#!/usr/bin/env python3
"""
Script to download all required HuggingFace models for WonderWorld.
This helps avoid authentication and network issues during runtime.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface-hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
    from huggingface_hub import snapshot_download

# Models to download
MODELS = [
    ("sd2-community/stable-diffusion-2-inpainting", "Stable Diffusion 2 Inpainting"),
    ("shi-labs/oneformer_ade20k_swin_large", "OneFormer (Segmentation)"),
    ("prs-eth/marigold-depth-v1-0", "Marigold Depth"),
    ("prs-eth/marigold-normals-v0-1", "Marigold Normals"),
]

def download_model(model_id, model_name, cache_dir=None):
    """Download a HuggingFace model."""
    print("=" * 50)
    print(f"Downloading: {model_name}")
    print(f"Model ID: {model_id}")
    print("=" * 50)
    
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
        )
        print(f"✓ Successfully downloaded: {model_name}\n")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}\n")
        return False

def main():
    print("=" * 50)
    print("WonderWorld Model Download Script")
    print("=" * 50)
    print()
    
    # Get cache directory
    cache_dir = os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface"))
    print(f"Using cache directory: {cache_dir}")
    print()
    print("Starting model downloads...")
    print("This may take 20-30 minutes depending on your connection speed.")
    print()
    
    # Download all models
    success_count = 0
    for model_id, model_name in MODELS:
        if download_model(model_id, model_name, cache_dir):
            success_count += 1
    
    print("=" * 50)
    if success_count == len(MODELS):
        print("✓ All models downloaded successfully!")
    else:
        print(f"⚠ Downloaded {success_count}/{len(MODELS)} models")
        print("Some models failed to download. Check errors above.")
    print("=" * 50)
    print()
    print(f"Models are cached in: {cache_dir}")
    print("You can now run the server without download delays.")
    print()

if __name__ == "__main__":
    main()

