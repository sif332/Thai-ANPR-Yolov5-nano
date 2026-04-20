# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thai Automatic Number Plate Recognition (ANPR) system built on YOLOv5-nano. Four models work in sequence to extract license plate text from images of Thai vehicles.

Model weights (`.pt` files) are **not included** in the repository. Download the full package (weights + pre-created output folders) from Google Drive: `https://drive.google.com/file/d/1wI20-h89p8Tx0FKqaBJtHugBIT8ZlAUr/view?usp=share_link`

## Setup

```bash
pip install -r requirements.txt
```

Pre-commit hooks (optional, for code quality):
```bash
pip install pre-commit
pre-commit run --all-files
```

## Running the Pipeline

The primary interface is Jupyter notebooks, not command-line scripts:

| Notebook | Purpose |
|---|---|
| `run_model.ipynb` | **Start here** — full pipeline with analysis tools and detailed comments |
| `run_input_from_folder.ipynb` | Batch process images from `test_folder/` |
| `run_input_from_capture_screen.ipynb` | Live screen capture input |
| `run_fast_from_folder_without_image_processing.ipynb` | Faster inference, slight accuracy trade-off |
| `run_fast_capture_screen_without_image_processing.ipynb` | Faster screen capture variant |

**Default folder conventions:**
- Input images: `test_folder/`
- Output images: `my_output/`
- Model weights: `my_weight/`

> OpenCV will error if an output directory doesn't exist — always create output folders before running.

## CLI Scripts (from upstream YOLOv5)

```bash
# Training
python train.py --data coco128.yaml --weights yolov5n.pt --img 640

# Multi-GPU (DDP)
python -m torch.distributed.run --nproc_per_node 4 train.py --data coco128.yaml --weights yolov5n.pt --device 0,1,2,3

# Inference
python detect.py --weights my_weight/<model>.pt --source test_folder/

# Validation
python val.py --weights my_weight/<model>.pt --data <dataset>.yaml --img 640

# Export (ONNX, TensorRT, CoreML, etc.)
python export.py --weights my_weight/<model>.pt --include onnx
```

## Code Quality

Linting and formatting are configured in `setup.cfg` (max line length: 120):

```bash
flake8 .              # PEP8 lint
isort .               # sort imports
yapf -r -i .          # format code
```

Tests use `pytest` with `--doctest-modules` (doctests inside source files are the primary test surface). Run:
```bash
pytest
```

## Four-Model Pipeline Architecture

The ANPR pipeline chains four independent YOLOv5 models:

```
Full Image (640×640)
    │
    ▼ Model 1: Car detection (1 class)
Car crop
    │
    ▼ Model 2: License plate detection (1 class)
Plate crop
    │
    ▼ Model 3: Character detection (2 classes: letter, province)
                Sort detections by x-coordinate (left-to-right)
Character crops (224×224 each)
    │
    ▼ Model 4: Character classification (48 classes: 38 Thai alphabets + 10 digits)
License plate string
```

Each model is a separate `.pt` file loaded independently. The classification model (Model 4) requires manual resizing to `(1, 3, 224, 224)` tensors before inference — YOLOv5 classification has no built-in auto-resize.

## Model Architecture

All detection models use **YOLOv5-nano** (`models/yolov5n.yaml`):
- `depth_multiple: 0.33`, `width_multiple: 0.25` — 30%/25% of standard YOLOv5 channels/depth
- Backbone: `Conv → C3 → SPPF` (CSP architecture)
- FPN head with 3 detection scales: P3/8, P4/16, P5/32
- `nc` (number of classes) is overridden per model at load time

Key modules in `models/common.py`: `Conv` (Conv+BN+SiLU), `C3` (CSP bottleneck), `SPPF` (Spatial Pyramid Pooling Fast), `DetectMultiBackend` (multi-framework inference wrapper).

## Key Utility Locations

| Utility | File |
|---|---|
| Image letterboxing / NMS / scaling | `utils/general.py` |
| Dataset loaders (`LoadImages`, `LoadStreams`) | `utils/dataloaders.py` |
| Loss (`ComputeLoss`) | `utils/loss.py` |
| Metrics (`ap_per_class`, `ConfusionMatrix`) | `utils/metrics.py` |
| Device selection, EMA, DDP helpers | `utils/torch_utils.py` |
| Bounding box drawing, `Annotator` | `utils/plots.py` |
| TensorBoard / W&B / Comet logging | `utils/loggers/` |

## Dataset & Hyperparameter Configs

- Dataset definitions: `data/*.yaml` (paths, class names, train/val splits)
- Hyperparameter profiles: `data/hyps/hyp.scratch-{low,med,high}.yaml`

When training a custom model, copy the appropriate `data/hyps/` file and adjust, then pass it via `--hyp`.
