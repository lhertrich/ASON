# ASON - Analysis of Spatial Organization of Nuclei

Repository for a university research project about the analysis of spatial organization of nuclei in human breast epithelium.

Author: Levin Hertrich

Supervised by: Daniel Sage and Pranay Dey

EPFL, January 2026

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Repository Structure](#repository-structure)
- [Demo](#demo)
- [Tissue Segmentation](#tissue-segmentation)
- [Development](#development)
- [AI Disclaimer](#ai-disclaimer)

## Overview

This project provides tools for analyzing the spatial organization of nuclei in histopathology images of human breast epithelium. The pipeline includes:

- **Stain normalization** using Reinhard normalization
- **Tissue segmentation** with deep learning models (U-Net, DeepLabV3, etc.)
- **Nuclei detection** using StarDist
- **Graph-based analysis** of nuclei spatial relationships
- **Layer detection** and classification of organized vs. unorganized tissue structures
- **Interactive GUI** for visualization and analysis

## Setup

### Prerequisites

- **Conda** (Miniconda or Anaconda)
  - If you don't have Conda installed, download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
  - Follow the installation instructions for your operating system

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd ASON
```

2. **Create the Conda environment:**

```bash
conda env create -f environment.yml
```

This will create a new environment called `research_project` with all required dependencies.

3. **Activate the environment:**

```bash
conda activate research_project
```

4. **Verify installation:**

```bash
python -c "import torch; import segmentation_models_pytorch; print('Installation successful!')"
```

### Important Note

**The pre-trained model checkpoints are not included in this repository due to file size limitations on GitHub.** You must train the models before using the demo or running the pipeline. See the [Tissue Segmentation](#tissue-segmentation) section for training instructions.

## Repository Structure

```
ASON/
├── configs/                 # Configuration files
│   ├── config.yaml          # Main pipeline configuration
│   └── model/               # Model-specific configs
│       ├── unet_2.yaml
│       ├── deeplabv3_2.yaml
│       └── ...
│
├── data/                   # Data directory where files should be put in
│   ├── images/             # Follow this structure
│   ├── masks/              
│   ├── normalized_images/  
│   └── cnn_training/
│
├── exploration/            # Jupyter notebooks for exploration
│   ├── graph_layer_detection.ipynb
│   ├── segmentation.ipynb  # Viz are commented out to be within GitHub file size restrictions
│   └── ...
│
├── gui_example_images/    # Example images for GUI demo
│
├── src/                   # Source code
│   ├── pipeline/          # Main pipeline modules
│   │   ├── segment.py     # Segmentation module
│   │   ├── layer_detection.py  # Layer detection
│   │   ├── plot.py        # Visualization functions
│   │   └── image_analysis_gui.py  # GUI application
│   ├── models/            # Model architectures
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   ├── pipeline_cli       # Started integration into qupath, not working
│   ├── qupath_integration # Started integration into qupath, not working
│   └── evaluate.py        # Evaluation script
│
├── run_gui.py             # GUI launcher
├── environment.yml        # Conda environment specification
└── README.md              # This file
```

## Demo

### Prerequisites

**Before running the demo, you must train a tissue segmentation model** as the pre-trained checkpoints are not included in the repository (too large for GitHub). Follow the instructions in the [Tissue Segmentation](#tissue-segmentation) section to train a model first.

### Running the Interactive GUI

Once you have trained a model, you can explore the pipeline through the interactive GUI:

```bash
conda activate research_project
python run_gui.py
```

### GUI Features

The GUI provides a step-by-step interface for the analysis pipeline:

1. **Load Image**: Browse for an image or use example images
   - Single layer example: `E2+DHT_1_M13_3L_0002.tif`
   - Multi-layer example: `E2+DHT_1_M13_3L_0008.tif`

2. **Normalize Image**: Apply Reinhard stain normalization

3. **Segment Tissue**: Detect tissue regions using deep learning

4. **Segment Nuclei**: Detect individual nuclei using StarDist

5. **Build Graph**: Construct spatial relationship graph

### Visualizations Available

- **Original Image**: Raw input image
- **Normalized Image**: Stain-normalized version
- **Tissue Mask**: Segmented tissue regions
- **Nuclei**: Detected nuclei centroids
- **Filtered Graph**: Spatial relationship graph (filtered by alignment/angle)
- **Top-2 Graph**: Graph with only top 2 neighbors per nucleus
- **Nuclei Axes**: Orientation axes of nuclei
- **Similarity Map**: Heatmap of nuclei organization similarity
- **Classification**: Organized vs. unorganized nuclei

### Quick Start Example

```bash
# 1. Activate environment
conda activate research_project

# 2. Train a segmentation model first (see Tissue Segmentation section)
python src/train.py --config configs/model/unet_2.yaml

# 3. Run GUI after training is complete
python run_demo.py

# In the GUI:
# 4. Click "Load Single Layer Example"
# 5. Click "Run Full Pipeline"
# 6. Explore different visualizations using the buttons
```

## Tissue Segmentation

### Training a Segmentation Model

**Note:** Training a segmentation model is **required** before using the demo or pipeline, as pre-trained checkpoints are not included in the repository due to GitHub file size limitations.

The project supports multiple segmentation architectures (U-Net, U-Net++, DeepLabV3) with various backbones.

#### 1. Prepare Training Data

Ensure your data is organized as follows (can be done with the `data_preparation.ipynb` notebook in `src/data`):

```
data/
├── cnn_training/
│   ├── resized_images/       # Training images
│   ├── resized_masks/        # Training masks
│   ├── resized_images_test/  # Test images
│   └── resized_masks_test/   # Test masks
```

#### 2. Configure Training

Edit or create a model configuration file in `configs/model/`. Example configurations:

- `unet_2.yaml` - U-Net with 2 classes (background, tissue)
- `unet_3.yaml` - U-Net with 3 classes (background, epithelium, stroma)
- `deeplabv3_2.yaml` - DeepLabV3 with 2 classes

Key configuration parameters:

```yaml
model:
  architecture: "unet"  # or "unet++", "deeplabv3"
  encoder: "resnet34"   # backbone architecture
  num_classes: 2        # number of output classes
  
training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 50
  
data:
  image_size: [512, 512]
```

#### 3. Train the Model

```bash
conda activate research_project

# Train with specific config
python src/train.py --config configs/model/unet_2.yaml

# Monitor training with Weights & Biases (optional)
# Training logs will be saved to wandb/
```

#### 4. Evaluate the Model

```bash
# Evaluate trained model
python src/evaluate.py --checkpoint checkpoints/your_model.pth

# Or use the evaluation notebook
jupyter notebook src/evaluate.ipynb
```

The trained models are saved in a `/checkpoints` directory. These checkpoints are used by the GUI and pipeline for tissue segmentation.

### Model Checkpoints

Due to GitHub file size limitations, pre-trained model checkpoints are **not included** in this repository. You must train your own models using the instructions above. Expected checkpoint names:

- `unet_resnet34_imagenet_2c.pth` - U-Net with ResNet34 backbone (2 classes)
- `deeplabv3_resnet34_imagenet_2c.pth` - DeepLabV3 with ResNet34 (2 classes)
- `unet++_resnet34_imagenet_2c.pth` - U-Net++ with ResNet34 (2 classes)

The GUI and pipeline will automatically load the appropriate checkpoint from the `checkpoints/` directory.

## Development

### Running Exploration Notebooks

```bash
conda activate research_project
jupyter notebook

# Navigate to exploration/ directory
# Open any notebook (e.g., graph_layer_detection.ipynb)
```

### Project Dependencies

Key libraries used:

- **PyTorch** - Deep learning framework
- **segmentation-models-pytorch** - Segmentation architectures
- **StarDist** - Nuclei detection
- **NetworkX** - Graph analysis
- **scikit-learn** - Machine learning utilities
- **OpenCV** - Image processing
- **matplotlib** - Visualization
- **tifffile** - TIFF image handling

### Adding New Models

1. Add model configuration to `configs/model/`
2. Update `src/models/` if custom architecture needed
3. Train using `src/train.py`
4. Add checkpoint to `checkpoints/`

## AI Disclaimer

Generative AI tools have been used in this project. Claude-4.5 has been used for ideation, code debugging and the generation of documentation. Furthermore, it has been used to generate the demo GUI, as well as this README file.
