# Model Weights Guide

This document explains how to load and manage model weights for WonderWorld.

## Overview

WonderWorld uses several pre-trained models that are loaded automatically or manually:

1. **RepViT SAM** - Manual download required
2. **Stable Diffusion** - Auto-downloaded from HuggingFace (configurable)
3. **OneFormer** - Auto-downloaded from HuggingFace
4. **Marigold Depth** - Auto-downloaded from HuggingFace
5. **Marigold Normals** - Auto-downloaded from HuggingFace

## Manual Model Downloads

### RepViT SAM Model

The RepViT SAM model must be downloaded manually and placed in the project root directory.

**Download command:**
```bash
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt
```

**Location:** Place `repvit_sam.pt` in the root directory (`/workspace/WonderWorld/`)

**Usage:** The model is loaded automatically by `util/segment_utils.py` when creating the mask generator.

## Auto-Downloaded Models (HuggingFace)

Most models are automatically downloaded from HuggingFace when you first run the code. They are cached locally for future use.

### Model Cache Location

By default, HuggingFace models are cached in:
- **Linux/Mac:** `~/.cache/huggingface/`
- **Windows:** `C:\Users\<username>\.cache\huggingface\`

You can change the cache location by setting the `HF_HOME` environment variable:
```bash
export HF_HOME=/path/to/your/cache
```

### Models Used

1. **Stable Diffusion Inpainting**
   - **Model ID:** `sd2-community/stable-diffusion-2-inpainting` (mirror of deprecated `stabilityai/stable-diffusion-2-inpainting`)
   - **Config Location:** `config/base-config.yaml` (line 24)
   - **Configurable:** Yes, via `stable_diffusion_checkpoint` in config
   - **Size:** ~5GB
   - **Note:** The original `stabilityai/stable-diffusion-2-inpainting` repository is deprecated. Use `sd2-community/stable-diffusion-2-inpainting` instead.

2. **OneFormer (Segmentation)**
   - **Model ID:** `shi-labs/oneformer_ade20k_swin_large`
   - **Location in code:** `run.py` (line 119-120)
   - **Configurable:** No (hardcoded)
   - **Size:** ~1GB

3. **Marigold Depth**
   - **Model ID:** `prs-eth/marigold-depth-v1-0`
   - **Location in code:** `run.py` (line 137)
   - **Configurable:** No (hardcoded)
   - **Size:** ~5GB

4. **Marigold Normals**
   - **Model ID:** `prs-eth/marigold-normals-v0-1`
   - **Location in code:** `run.py` (line 141)
   - **Configurable:** No (hardcoded)
   - **Size:** ~5GB

## Using Custom/Local Model Weights

### Option 1: Use Local HuggingFace Model

If you have a local copy of a HuggingFace model, you can use it by:

1. **For Stable Diffusion** (configurable):
   ```yaml
   # In config/base-config.yaml or your example config
   stable_diffusion_checkpoint: "/path/to/local/model"
   ```

2. **For other models** (requires code modification):
   - Modify `run.py` to use a local path instead of the HuggingFace model ID
   - Example for Marigold:
     ```python
     # Change line 137 in run.py from:
     depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", ...)
     # To:
     depth_model = MarigoldPipeline.from_pretrained("/path/to/local/marigold", ...)
     ```

### Option 2: Pre-download Models (Recommended)

**Easiest method - Use the provided script:**
```bash
# Run the automated download script
bash download_models.sh
```

**Manual method - Using HuggingFace CLI:**
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download models individually
huggingface-cli download sd2-community/stable-diffusion-2-inpainting
huggingface-cli download shi-labs/oneformer_ade20k_swin_large
huggingface-cli download prs-eth/marigold-depth-v1-0
huggingface-cli download prs-eth/marigold-normals-v0-1
```

**Note:** Pre-downloading models helps avoid authentication errors and network issues during runtime.

### Option 3: Use cache_dir Parameter

You can specify a custom cache directory when loading models by modifying the code:

```python
from_pretrained("model-id", cache_dir="/custom/path/to/cache")
```

## Generated Scene Models

WonderWorld also saves/loads generated 3D Gaussian Splatting models:

**Save location:** `examples/sky_images/{example_name}/`

**Files saved:**
- `finished_3dgs.ply` - Main 3D Gaussian model
- `finished_3dgs_sky_tanh.ply` - Sky model
- `visibility_filter_all.pth` - Visibility filters
- `is_sky_filter.pth` - Sky filters
- `delete_mask_all.pth` - Delete masks
- `{example_name}_finished_3dgs.splat` - Splat format

**Loading:** Set `load_gen: True` in your config file to load previously generated scenes.

## Model Size Summary

| Model | Size | Download Type |
|-------|------|---------------|
| RepViT SAM | ~100MB | Manual |
| Stable Diffusion 2 Inpainting | ~5GB | Auto (HuggingFace) |
| OneFormer | ~1GB | Auto (HuggingFace) |
| Marigold Depth | ~5GB | Auto (HuggingFace) |
| Marigold Normals | ~5GB | Auto (HuggingFace) |
| **Total** | **~16GB** | |

## Important Update: Stable Diffusion Model Path Changed

**⚠️ The Stable Diffusion 2 Inpainting model has moved!**

The original `stabilityai/stable-diffusion-2-inpainting` repository is now deprecated. The model is now available at:
- **New location:** `sd2-community/stable-diffusion-2-inpainting`
- **Reference:** [HuggingFace Model Page](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting)

All configuration files and scripts have been updated to use the new path. If you're using an older version of the codebase, update your `config/base-config.yaml`:

```yaml
# Old (deprecated):
stable_diffusion_checkpoint: "stabilityai/stable-diffusion-2-inpainting"

# New (current):
stable_diffusion_checkpoint: "sd2-community/stable-diffusion-2-inpainting"
```

## Troubleshooting

### 401 Authentication Error (Most Common Issue)

If you see errors like:
```
Couldn't connect to the Hub: 401 Client Error
Repository Not Found for url: https://huggingface.co/api/models/...
```

**Solution 1: Pre-download models using the provided script**
```bash
# Run the download script (recommended)
bash download_models.sh
```

**Solution 2: Authenticate with HuggingFace (if required)**
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Login to HuggingFace (creates token if needed)
huggingface-cli login
# Follow prompts to enter your token (get one from https://huggingface.co/settings/tokens)
```

**Solution 3: Download models manually**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Download each model individually
huggingface-cli download sd2-community/stable-diffusion-2-inpainting
huggingface-cli download shi-labs/oneformer_ade20k_swin_large
huggingface-cli download prs-eth/marigold-depth-v1-0
huggingface-cli download prs-eth/marigold-normals-v0-1
```

**Solution 4: Use environment variable for offline mode (if models are already cached)**
```bash
# If models are already downloaded but you're getting errors
export HF_HUB_OFFLINE=1
python run.py --example_config config/example.yaml --port 7777
```

### Model Download Issues

If models fail to download:
1. Check internet connection
2. Verify HuggingFace access (some models may require authentication)
3. Check disk space (models require ~16GB total)
4. Try downloading manually using `huggingface-cli download` or the `download_models.sh` script
5. Check if you're behind a firewall/proxy that blocks HuggingFace

### RepViT Model Not Found

If you get an error about `repvit_sam.pt`:
```bash
# Download and place in root directory
cd /workspace/WonderWorld
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt
```

### Custom Model Paths

To use custom model paths, you'll need to modify the code in `run.py` for non-configurable models, or update your config file for Stable Diffusion.

### Network/Proxy Issues

If you're behind a corporate firewall or proxy:
```bash
# Set proxy environment variables
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# Or configure huggingface-hub to use proxy
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## First Run

On the first run, the program will:
1. Download all HuggingFace models automatically (this may take 20-30 minutes depending on connection)
2. Cache them locally for future use
3. Generate sky panorama images (takes ~20 minutes on A6000 GPU)

Subsequent runs will use the cached models and skip sky generation if files already exist.

