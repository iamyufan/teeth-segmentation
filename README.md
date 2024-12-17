# Teeth Segmentation with Two-Step Model

This repository contains code for a teeth segmentation model built using a two-step process. The first step employs **AnchorDETR** for object detection, while the second step utilizes **Segment Anything Model (SAM)** for precise segmentation. The entire model is designed using **PyTorch Lightning**, which streamlines the management of the training process, including optimizers, schedulers, and checkpoints.

The codebase also includes a custom **PyTorch DataModule** for loading datasets and a **PyTorch Lightning Module** for defining the model, allowing for easy configuration and experimentation.

## Overview

The model is designed for teeth segmentation, where the two-step process involves:

1. **AnchorDETR (Step 1)**: An object detection model that detects bounding boxes for potential objects (teeth in this case).
2. **Segment Anything Model (SAM) (Step 2)**: Once the bounding boxes are predicted, SAM is used to segment the objects within those boxes.

The code is structured using **PyTorch Lightning** to enhance modularity and allow easy experimentation with hyperparameters, training loops, and model checkpoints.

### Key Features:
- **AnchorDETR** for bounding box prediction.
- **Segment Anything Model (SAM)** for precise mask generation.
- **PyTorch Lightning** integration for streamlined training and evaluation.
- Easy to adapt for other segmentation tasks with minimal changes.

## Dependencies

Before running the code, you need to install a few libraries:

### Required Libraries:
- `torch` and `torchvision` (for PyTorch and related functionalities).
- `segment_anything` (for SAM).
- `anchordetr` (for AnchorDETR).
- `matplotlib` (for plotting).
- `wandb` (for experiment tracking, optional).

You can install the required dependencies using the following command:

```bash
pip install torch torchvision matplotlib wandb
```

To install **Segment Anything** and **AnchorDETR**:

1. **Segment Anything**:
   - Clone the repository from [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything) or install it via pip:
   ```bash
   pip install segment-anything
   ```

2. **AnchorDETR**:
   - Clone the repository from [AnchorDETR GitHub](https://github.com/Anchordetr/anchor-detr) or install it via pip:
   ```bash
   pip install anchordetr
   ```

## Repository Structure

Here is a breakdown of the repository structure:

```
teeth-segmentation/
├── datasets/                    # Dataset-related code and configurations
│   ├── __init__.py
│   └── tdd.py
├── models/                   # Model-related code
│   ├── __init__.py
│   ├── two_step_model.py
│   └── maskrcnn.py
├── outputs/                  # Directory to store training logs etc.
├── checkpoints/              # Directory for saving model checkpoints
└── README.md                 # Project overview and documentation
```

## Usage

### Training the Model

To train the model, run the `train.py` script. This will start the two-step training process:

1. Train the **AnchorDETR** model (for object detection).
2. Train the **SAM Neck** for fine-tuning mask segmentation based on the predicted bounding boxes.

```bash
python scripts/train.py --train_data_path <path_to_training_data> --num_epochs_det 200 --num_epochs_sam 50 --batch_size 8
```

## Training Procedure

The training process is divided into two stages:

1. **Stage 1 (Object Detection - AnchorDETR)**: 
   - The model learns to predict bounding boxes for each object (teeth).
   - Training is done for a specified number of epochs (`num_epochs_det`).
   
2. **Stage 2 (Segmentation - SAM)**: 
   - The trained object detection model is used to generate bounding boxes, and SAM is then fine-tuned to predict segmentation masks inside those boxes.
   - Training is done for a specified number of epochs (`num_epochs_sam`).

Both stages can be trained independently or together depending on your experimental setup.