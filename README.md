# TinyBuildingDetection

Tiny object detection in video/images using a YOLO-based architecture enhanced with a **Super-Resolution branch** (SuperYOLO-style). The model is designed to detect **tiny buildings** in high-resolution aerial/drone imagery, where targets occupy only a few pixels and are easily missed by standard detectors.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Experiment Tracking (W&B)](#experiment-tracking-wb)
- [Checkpoints](#checkpoints)
- [Roadmap](#roadmap)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [Author](#author)

---

## Overview

Detecting tiny objects (e.g., small buildings in drone/satellite images) is challenging because:

- Tiny objects lose nearly all of their visual features after repeated downsampling in deep backbones.
- Standard detection heads operate on low-resolution feature maps where a tiny object may collapse into a single activation — or disappear entirely.

This project tackles the problem with a **multi-task architecture** inspired by [SuperYOLO](https://arxiv.org/abs/2209.09751): a YOLOv12n detection network combined with an **auxiliary Super-Resolution (SR) branch** that reconstructs high-resolution detail from low-resolution inputs, forcing the backbone to retain fine-grained spatial information that benefits tiny-object detection.

The input image is downsampled by a factor of 2 before entering the network; the SR branch (EDSR-based decoder) reconstructs the original resolution while the detection head predicts buildings at three feature scales.

## Key Features

- 🔍 **Tiny object detection** tailored for single-class building detection in aerial imagery
- 🏗️ **SuperYOLO-style multi-task architecture**: YOLOv12n backbone + EDSR super-resolution branch
- 🎯 **Rich detection loss**: Objectness (BCE) + Classification (BCE) + Box regression (**CIoU** + **DFL** — Distribution Focal Loss) + SR reconstruction (L1)
- 📊 **Weights & Biases integration**: live loss curves, gradient/parameter watching, train/val comparison plots
- 💾 **Checkpointing & resume**: per-epoch checkpoints with the option to resume training from the latest epoch
- 🖼️ **Letterbox preprocessing**: aspect-ratio-preserving resize with padding (YOLO-style)
- ⚙️ **JSON-driven configuration**: epochs, batch size, image size, device, paths, learning rate

## Architecture

```
Input Image (1024×1024)
        │
        ▼  downsample ×2
┌─────────────────────┐
│  BackBone           │  YOLOv12n layers 0–8
│  (pretrained)       │  → multi-scale features (p2, p3, p4, p5)
└──────────┬──────────┘
           │
     ┌─────┴──────────────┐
     ▼                    ▼
┌──────────────┐   ┌────────────────────┐
│ SuperRes     │   │ Head               │  YOLOv12n layers 9–21
│ Decoder+EDSR │   │ (FPN/PAN + Detect) │  patched to 1 class
│ (p2, p3)     │   │ (p3, p4, p5)       │
└──────┬───────┘   └─────────┬──────────┘
       │                     │
       ▼                     ▼
 SR output (×2)      Detections @ 3 scales
                       (strides 8/16/32)
```

**Components:**

| Module | File | Description |
|---|---|---|
| `BackBone` | `tinynetwork/yoloo.py` | YOLOv12n layers 0–8; returns multi-scale features `(p2, p3, p4, p5)` |
| `Head` | `tinynetwork/yoloo.py` | YOLOv12n layers 9–21; Detect head patched to a single output class |
| `SuperYoloo` | `tinynetwork/yoloo.py` | Full model: backbone + SR branch + head |
| `SuperResolution` | `tinynetwork/supreres.py` | Feature-fusion `Decoder` + `EDSR` upsampler (PixelShuffle, residual blocks) |
| Losses | `tinyloss/lossfunctions.py` | CIoU, DFL, objectness/classification BCE, target building |
| Train/val loops | `utils/util.py` | One-epoch train and validation routines, checkpoint I/O, logging |

**Total loss:**

```
L = L_obj + L_cls + L_box (CIoU + DFL) + 0.1 · L1(SR_output, HR_image)
```

## Project Structure

```
TinyBuildingDetection/
├── main.py                     # Entry point: training driver (wandb init, dataloaders, loops)
├── conf/
│   └── config.json             # Training configuration
├── tinydata/
│   └── tinydataset.py          # TinyData dataset (YOLO-format labels, letterbox resize)
├── tinyloss/
│   └── lossfunctions.py        # CIoU / DFL / obj / cls losses, target building
├── tinynetwork/
│   ├── yoloo.py                # BackBone, Head, SuperYoloo (main model)
│   └── supreres.py             # Upsampler, ResBlock, Decoder, EDSR, SuperResolution
├── utils/
│   └── util.py                 # Logger, checkpoint save/load, train/val loops, config loader
└── README.md
```

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU recommended (CPU is supported but slow)

### Setup

```bash
# Clone the repository
git clone https://github.com/Abolmw4/TinyBuildingDetection.git
cd TinyBuildingDetection

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision ultralytics opencv-python numpy tqdm wandb
```

You also need the pretrained **YOLOv12n** weights (`yolo12n.pt`), which are loaded automatically through the `ultralytics` package.

## Dataset Preparation

The dataset must follow the YOLO layout with **two subdirectories** under the root `dataset_dir`:

```
<dataset_dir>/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/
    ├── img_001.txt
    ├── img_002.txt
    └── ...
```

**Label format** (YOLO normalized, one object per line):

```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image width/height. Although the label files contain a class id, the detection head is configured for **single-class** output (building).

Preprocessing is handled automatically by `TinyData`:

- Aspect-ratio-preserving **letterbox resize** to `image_size` (padding color `114,114,114`)
- BGR → RGB, HWC → CHW conversion

## Configuration

All training settings live in [`conf/config.json`](conf/config.json):

| Key | Type | Default | Description |
|---|---|---|---|
| `number_of_epoch` | int | `2` | Number of training epochs |
| `batch_size` | int | `2` | Training batch size |
| `optimizer` | str | `"Adam"` | Optimizer (Adam) |
| `device` | str | `"cuda"` | `cuda` or `cpu` (falls back to CPU if CUDA unavailable) |
| `dataset_dir` | path | — | Root of the dataset (must contain `images/` and `labels/`) |
| `number_of_workers` | int | `1` | DataLoader worker processes |
| `image_size` | int | `1024` | Input image size (letterboxed) |
| `world_size` | int | `1` | Distributed world size (currently single-process) |
| `log_file` | path | — | Directory for training logs |
| `learning_rate` | float | `0.001` | Initial learning rate |
| `weights` | path | — | Directory where checkpoints are saved/loaded |

> ⚠️ **Note:** The default configuration and some modules currently use absolute paths (e.g. `/home/my_proj/...`). Update `conf/config.json` (and the pretrained-weights path in `tinynetwork/yoloo.py`) to match your machine before training.

## Training

```bash
python main.py --train 1
```

(or equivalently `python main.py -t 1`)

On startup you will be prompted:

```
Should we start training from the beginning (1) or from the end (2)?:
```

- **`1`** — fresh training from the pretrained YOLOv12n backbone
- **`2`** — resume from the latest checkpoint found in the `weights` directory

The training loop then:

1. Builds the `TinyData` dataset and a `DataLoader` with a custom `collate_fn` (image indices are injected into the label tensor).
2. Downsamples each input by ×2 and runs a forward pass through `SuperYoloo` (detection + super-resolution).
3. Computes the combined loss (obj + cls + CIoU/DFL box + `0.1 × L1` SR loss) across strides **8, 16, 32**.
4. Runs a validation epoch after each training epoch.
5. Saves a checkpoint (`checkpoint<epoch>.pth`) after every epoch.

## Experiment Tracking (W&B)

Training is instrumented with [Weights & Biases](https://wandb.ai):

- `wandb.login()` runs at startup (make sure you are authenticated: `wandb login`).
- A run is created in the project **`Tiny Building Detection`** with the full hyperparameter config.
- All model gradients/parameters are watched (`log="all"`, every 100 steps).
- Per-epoch train/validation losses are logged, with `val_obj_loss` tracked by its minimum.
- A final train-vs-validation loss line-series plot is logged at the end of training.

To run offline, set `WANDB_MODE=offline` before launching.

## Checkpoints

Checkpoints are written to the `weights` directory as `checkpoint<epoch>.pth` and contain:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "total_train_loss": float,
}
```

Load them with `utils.util.load_model(checkpoint, model, optimizer)` — this is done automatically when you choose resume (`2`) at the training prompt.

## Roadmap

- [ ] Inference / detection script for images and video (`main.py --train 0`)
- [ ] Evaluation metrics (mAP, precision/recall, IoU)
- [ ] `requirements.txt`
- [ ] Remove hard-coded absolute paths; full path configurability
- [ ] Multi-class support
- [ ] Pretrained SuperYoloo weights release

## Known Limitations

- **Inference mode is not yet implemented** — `python main.py -t 0` is currently a no-op.
- **No metric computation** (mAP/IoU) yet; only loss curves are tracked during validation.
- Configuration and model files reference **absolute paths** that must be adapted to your environment.
- Single-class (building) detection only.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Feature: my feature"`)
4. Push and open a Pull Request

## Author

**Abolfazl Asghari**

If you find this project useful, please consider giving it a ⭐ on GitHub.
