# CLI Reference

This guide provides detailed information about all command-line interface (CLI) commands available in Segmented Creator, including their parameters and usage examples.

## Command Structure

All commands follow the pattern:
```bash
uv run python -m vivadatacreator.<step_name> [OPTIONS]
```

## Common Parameters

Most steps share these common parameters:

- `--root PATH`: Path to the video file (required)
- `--fac INT`: Scaling factor for image resizing (default: 2)
- `--sam2-chkpt PATH`: Path to SAM2 checkpoint file
- `--model-cfg PATH`: Path to SAM2 model configuration file
- `--n-imgs INT`: Number of images to process per batch (default: 200)
- `--n-obj INT`: Number of objects to process simultaneously (default: 20)
- `--img-size-sahi INT`: Image size for SAHI processing (default: 512)
- `--overlap-sahi FLOAT`: Overlap ratio for SAHI (default: 0.2)

## Step-by-Step Commands

### Step 1: Frame Extraction

Extracts and aligns video frames to correct camera vibrations.

```bash
uv run python -m vivadatacreator.first_step --root /path/to/video.mp4
```

**Parameters:**
- `--root PATH`: Input video file path (required)

**Outputs:**
- `imgsA/` folder with extracted frames
- `video_alineado.mp4` aligned video

### Step 2: Interactive Initial Segmentation

Interactively segment objects in the first frame using SAM2.

```bash
uv run python -m vivadatacreator.second_step \
  --root /path/to/video.mp4 \
  --sam2-chkpt checkpoints/sam2.1_hiera_large.pt \
  --model-cfg /path/to/sam2/config.yaml
```

**Parameters:**
- `--sam2-chkpt PATH`: SAM2 model checkpoint (required)
- `--model-cfg PATH`: SAM2 configuration file (required)

**Outputs:**
- `mask_prompts.csv` with segmentation prompts

### Step 3: Automatic Mask Propagation

Propagates initial masks throughout the video using SAM2 tracking.

```bash
uv run python -m vivadatacreator.third_step \
  --root /path/to/video.mp4 \
  --sam2-chkpt checkpoints/sam2.1_hiera_large.pt \
  --model-cfg /path/to/sam2/config.yaml \
  --n-imgs 200
```

**Parameters:**
- `--n-imgs INT`: Batch size for processing

**Outputs:**
- `masks/` folder with individual mask files
- `segmentation/` folder with combined masks per frame

### Step 4: Object Detection and Tracking

Detects and tracks objects using YOLO and DeepSort.

```bash
uv run python -m vivadatacreator.fourth_step \
  --root /path/to/video.mp4 \
  --img-size-sahi 512 \
  --overlap-sahi 0.2
```

**Optimized version with resource management:**
```bash
uv run python -m vivadatacreator.fourth_step_optimized \
  --root /path/to/video.mp4 \
  --auto-tune \
  --mask-cache auto \
  --ram-budget 0.6 \
  --device auto
```

**Parameters:**
- `--device {auto|cuda|cpu}`: Execution device
- `--batch-size INT`: Detection batch size
- `--img-size-sahi INT`: SAHI slice size
- `--overlap-sahi FLOAT`: SAHI overlap ratio
- `--ram-budget FLOAT`: RAM usage limit (0.1-0.9)
- `--mask-cache {auto|memory|disk}`: Caching strategy
- `--yolo-weights PATH`: Custom YOLO weights
- `--confidence-threshold FLOAT`: Detection confidence

**Outputs:**
- `track_dic.csv` with detected object tracks

### Step 5: Interactive Mask Refinement

Refines detected objects with interactive SAM2 segmentation.

```bash
uv run python -m vivadatacreator.fifth_step \
  --root /path/to/video.mp4 \
  --sam2-chkpt checkpoints/sam2.1_hiera_large.pt \
  --model-cfg /path/to/sam2/config.yaml
```

**Outputs:**
- `traked/` folder with refined masks
- `mask_list.csv` with refined object list

### Step 6: Enhanced Mask Propagation

Propagates refined masks forward and backward through the video.

```bash
uv run python -m vivadatacreator.sixth_step \
  --root /path/to/video.mp4 \
  --sam2-chkpt checkpoints/sam2.1_hiera_large.pt \
  --model-cfg /path/to/sam2/config.yaml \
  --n-imgs 200
```

**Outputs:**
- Updated `masks/` folder with enhanced segmentations

### Step 7: Color-Based Semantic Segmentation

Creates color-coded semantic segmentation maps.

```bash
uv run python -m vivadatacreator.seventh_step --root /path/to/video.mp4
```

**Outputs:**
- `semantic/` folder with colored segmentation images

### Step 8: Final Dataset Creation

Combines images with semantic masks to create the final dataset.

```bash
uv run python -m vivadatacreator.eighth_step --root /path/to/video.mp4
```

**Outputs:**
- `dataset/` folder with final dataset images

## Workflow Combinations

### Basic SAM2-Only Workflow

For simple segmentation without YOLO detection:

```bash
# Steps 1, 2, 3, 7, 8
uv run python -m vivadatacreator.first_step --root video.mp4
uv run python -m vivadatacreator.second_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.third_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.seventh_step --root video.mp4
uv run python -m vivadatacreator.eighth_step --root video.mp4
```

### Full Pipeline with Refinement

Complete workflow including YOLO detection and refinement:

```bash
# Steps 1-8
uv run python -m vivadatacreator.first_step --root video.mp4
uv run python -m vivadatacreator.second_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.third_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.fourth_step --root video.mp4
uv run python -m vivadatacreator.fifth_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.sixth_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.seventh_step --root video.mp4
uv run python -m vivadatacreator.eighth_step --root video.mp4
```

### Quick Segmentation (Skip Detection)

For when you only need basic segmentation:

```bash
# Steps 1, 2, 3, 7
uv run python -m vivadatacreator.first_step --root video.mp4
uv run python -m vivadatacreator.second_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.third_step --root video.mp4 --sam2-chkpt checkpoints/sam2.1_hiera_large.pt --model-cfg config.yaml
uv run python -m vivadatacreator.seventh_step --root video.mp4
```

## Configuration Management

Commands automatically save settings to `config.yaml`. You can also manually manage configuration:

```bash
# Download checkpoints manually
uv run python -m vivadatacreator.download_checkpoints

# Install additional dependencies
uv run python -m vivadatacreator.install
```

## Performance Tuning

### Memory Management

For large videos, adjust these parameters:

```bash
# Reduce batch size for lower memory usage
--n-imgs 50 --n-obj 10

# Use disk caching for large datasets
--mask-cache disk --ram-budget 0.3
```

### GPU Optimization

```bash
# Force GPU usage
--device cuda

# Use optimized settings
uv run python -m vivadatacreator.fourth_step_optimized --auto-tune
```

## Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch sizes
--n-imgs 50 --batch-size 4 --ram-budget 0.5
```

**Slow processing:**
```bash
# Use optimized pipeline
uv run python -m vivadatacreator.fourth_step_optimized --auto-tune
```

**Missing checkpoints:**
```bash
# Download manually
uv run python -m vivadatacreator.download_checkpoints
```