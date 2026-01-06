# Automated Generation Guide

The `run_automated.py` script allows you to generate WonderWorld scenes automatically without the interactive web interface. This is useful for batch generation, scripting, or when you want to generate scenes using predefined camera paths and prompts.

## Features

- **No Interactive Interface**: Runs entirely from command line
- **Predefined Camera Paths**: Uses cameras from `rotation_path` in config
- **Flexible Prompting**: Supports GPT-based prompts, predefined prompts, or rotation-based generation
- **Batch Processing**: Generate multiple scenes sequentially

## Usage

### Basic Usage (Rotation-Based Generation)

Generate scenes using the default rotation path and GPT-based prompts:

```bash
python run_automated.py --example_config config/example.yaml
```

This will:
- Use all cameras from `rotation_path` (up to `num_scenes` in config)
- Generate prompts using GPT based on the example's style and content
- Generate scenes sequentially without user interaction

### Using Predefined Prompts

Generate scenes with specific prompts:

```bash
python run_automated.py --example_config config/example.yaml --prompts "campus,library,quad,dormitory"
```

Each comma-separated prompt will be used for the corresponding scene.

### Using Specific Camera Indices

Generate scenes using only specific cameras:

```bash
# Use cameras 1, 3, 5, 7
python run_automated.py --example_config config/example.yaml --camera-indices "1,3,5,7"

# Use all cameras
python run_automated.py --example_config config/example.yaml --camera-indices "all"
```

### Using GPT for Prompts

Force GPT-based prompt generation (even if prompts are provided):

```bash
python run_automated.py --example_config config/example.yaml --use-gpt
```

## Command-Line Arguments

- `--example-config` (required): Path to example config YAML file
- `--base-config`: Path to base config YAML (default: `./config/base-config.yaml`)
- `--camera-indices`: Comma-separated camera indices or "all" (default: uses all cameras from rotation_path)
- `--prompts`: Comma-separated scene prompts (default: uses GPT or rotation-based generation)
- `--use-gpt`: Force GPT-based prompt generation

## How It Works

1. **Initialization**: Loads models, generates sky, and creates initial scene (same as `run.py`)
2. **Camera Selection**: Uses cameras from `rotation_path` or specified indices
3. **Prompt Generation**: 
   - If `--prompts` provided: Uses predefined prompts
   - If `--use-gpt`: Uses GPT to generate prompts
   - Otherwise: Uses rotation-based generation (default)
4. **Scene Generation**: For each camera:
   - Converts camera to view matrix
   - Generates scene using inpainting
   - Trains 3DGS
   - Moves to next camera
5. **Output**: Saves all generated scenes to `runs_dir` (from config)

## Example Workflows

### Generate 16 scenes with GPT prompts:
```bash
python run_automated.py --example_config config/example.yaml --use-gpt
```

### Generate specific scenes with custom prompts:
```bash
python run_automated.py --example_config config/example.yaml \
  --camera-indices "1,2,3,4" \
  --prompts "campus entrance,library building,student quad,dormitory"
```

### Generate all scenes using rotation path:
```bash
python run_automated.py --example_config config/example.yaml --camera-indices "all"
```

## Differences from Interactive Mode

- **No Web Interface**: No Flask/SocketIO server, no HTML interface
- **No User Input**: All prompts and cameras are predetermined
- **Sequential Processing**: Scenes are generated one after another
- **No Real-time Rendering**: No live preview during generation
- **Faster**: No network overhead or UI updates

## Tips

- **Monitor Progress**: The script uses `tqdm` progress bars to show generation status
- **GPU Memory**: Ensure you have enough GPU memory for batch processing
- **Interrupt Handling**: You can interrupt with Ctrl+C, but partial scenes may be saved
- **Config Settings**: Make sure `num_scenes` in your config matches your expectations

## Troubleshooting

- **Out of Memory**: Reduce `num_scenes` or process cameras in smaller batches
- **Missing Cameras**: Ensure `rotation_path` has enough cameras for your needs
- **GPT Errors**: Check your `OPENAI_API_KEY` environment variable if using GPT
- **Model Loading**: First run will download models (same as `run.py`)

