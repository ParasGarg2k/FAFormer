# FACT: Feature-Action Coupling in Time

A PyTorch implementation of temporal action segmentation model that couples frame features with action tokens through multi-stage attention mechanisms.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Results](#results)

## Overview

FACT is a novel architecture for temporal action segmentation that:
- Processes video features and action tokens in parallel
- Uses multi-stage attention for feature-action coupling
- Employs temporal downsampling and upsampling for efficient processing
- Supports multiple backbone networks (MSTCN, MSTCN++, Transformer)

## Model Architecture

The model consists of three main types of blocks:

1. **Input Block (i)**:
   - Initial processing of frame features
   - Action token initialization
   - Basic feature-action coupling

2. **Update Block (u)**:
   - Cross-attention between frames and actions
   - Self-attention for action refinement
   - Feature updates through attention

3. **Update Block with TDU (U)**:
   - Temporal downsampling of frame features
   - Segment-level attention
   - Temporal upsampling for final predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fact.git
cd fact

# Create a conda environment (recommended)
conda create -n fact python=3.8
conda activate fact

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

## Project Structure

```
fact/
├── configs/              # Configuration files
│   ├── default.py       # Default configuration
│   └── gtea3.yaml       # Dataset specific config
├── models/              # Model implementation
│   ├── __init__.py
│   ├── attention.py     # Attention mechanisms
│   ├── base_block.py    # Base block definition
│   ├── fact.py         # Main FACT model
│   └── layers.py       # Neural network layers
├── utils/              # Utility functions
│   ├── dataset.py      # Data loading and processing
│   ├── evaluate.py     # Evaluation metrics
│   └── train_tools.py  # Training helpers
├── train_gtea.ipynb    # Training notebook
└── visualize_model.py  # Model visualization
```

## Usage

### Data Preparation

1. Prepare your video features:
```bash
data_i3d/
├── gtea/
│   ├── features/       # I3D features
│   ├── groundTruth/    # Frame-level labels
│   ├── mapping.txt     # Action label mapping
│   └── splits/         # Train/test splits
```

2. Update the config file:
- Modify `configs/gtea3.yaml` for dataset paths
- Adjust model parameters as needed

### Training

```python
# Using the training notebook
jupyter notebook train_gtea.ipynb

# Or run from command line
python train.py --config configs/gtea3.yaml
```

Training parameters can be modified through:
- Config files
- Command line arguments
- Notebook cells

## Configuration

Key configuration options in `configs/gtea3.yaml`:

```yaml
# Dataset
dataset: "gtea"
split: "split2"

# Model Architecture
FACT:
  block: "iuU"        # Block sequence
  ntoken: 64          # Number of action tokens
  trans: true         # Use transformer
  fpos: true         # Frame positional encoding

# Training
batch_size: 1
epoch: 400
lr: 0.0001
optimizer: "Adam"
```

## Results

Model performance on benchmark datasets:

| Dataset | F1@0.50 | Edit | Acc |
|---------|---------|------|-----|
| GTEA    | XX.X    | XX.X | XX.X|

## Visualization

Generate model architecture visualization:
```bash
python visualize_model.py
```
This will create a detailed flowchart showing the model's architecture and data flow.

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{fact2023,
  title={FACT: Feature-Action Coupling in Time},
  author={Your Name},
  booktitle={Conference},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Implementation based on the FACT paper
- I3D features extracted using [I3D model](https://github.com/deepmind/kinetics-i3d)
- MSTCN++ implementation referenced from [MSTCN++](https://github.com/yiskw713/asrf)
