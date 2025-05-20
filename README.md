# AIMI-project: ULS23 Segmentation Project

This repository contains the codebase for training and evaluating models for the ULS23 medical image segmentation challenge. All source code is organized within the ```src/``` folder and is modularly split into components related to data processing, model architecture, training, metrics, loss functions, and visualization.

# Installation Instructions in Cluster

After login to cluster, add this to your .profile file:

```
# disable pip caching downloads
export PIP_NO_CACHE_DIR=off
```

This disables pip caching, which would quickly fill up the 5GB of space we have on our home folders.

After cloning repo, cd to it and execute:
1. scripts/prepare_cluster.sh first. This creates symlinks
to data and logs paths which are on /vol/csedu-nobackup/course/IMC037_aimi/group08/ so that they don't occupy
our limited home folder storage. 
2. scripts/setup_virtual_environment.sh second, which creates a virtual environment for the project on your scratch folder where you can store a larger amount of temporal files. 
3. Finally, to sync virtual environments between nodes execute scripts/sync_csedu.sh. 

To install new packages, add them to requirements.txt and execute:

```
pip install -r requirements.txt
```

Then, run scripts/sync_csedu.sh again. Installing package directly with pip doesn't work with the syncing script, apparently.

More details on: https://gitlab.science.ru.nl/das-dl/cluster-skeleton-python/-/tree/main?ref_type=heads


# Project Structure
```
src/
├── seg_dataset.py            # Data loading and preprocessing
├── training.py               # PyTorch Lightning training pipeline
│
├── arch/                     # Model architectures
│   ├── vit.py                # Vision Transformer-based segmentation model
│   └── simple_unet.py        # Basic U-Net implementation
│
├── utils/                    # Utility scripts
│   ├── competition_metric.py # ULS competition metric (partial implementation)
│   ├── loss_fn.py            # Custom loss functions (e.g., Dice + BCE)
│   └── visualizations.py     # 2D/3D VOI visualization tools
```

# Code Modules

## ```seg_dataset.py```
This script is responsible for constructing data loaders using the dataset provided by the ULS23 challenge. It includes functionality for preprocessing, augmentation, and efficient batching, ensuring compatibility with PyTorch Lightning.

## ```training.py```
Handles the model training loop using PyTorch Lightning. It integrates data loading, model instantiation, loss functions, and metric evaluation into a clean training framework that supports GPU acceleration and reproducibility.

## ```arch/```
This folder contains the model architectures under evaluation for the challenge:

- ```vit.py```: A segmentation model based on the Vision Transformer (ViT) architecture.

- ```simple_unet.py```: A lightweight, baseline U-Net model for rapid prototyping.

## ```utils/```
```competition_metric.py```
Implements the evaluation metric used in the ULS23 competition. Most of the official metric is replicated, except for the final segment, which is not feasible to compute due to missing evaluation logic or components. It was obtained from the official [ULS23 Segmentation Project](https://github.com/DIAGNijmegen/ULS23) github repository.

```loss_fn.py```
Contains specialized loss functions tailored for segmentation, including. It was obtained from the official [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet) github repository:

- Dice Loss

- Dice + Binary Cross Entropy (BCE) Loss

These are designed to optimize for overlap-based performance metrics common in medical image segmentation.

```visualizations.py```
Includes tools to visualize the segmented Volumes of Interest (VOIs). It supports:

- Full 3D rendering of volumes

- Per-slice 2D visual inspection

## Getting Started

Clone the repository and explore the `src/` directory to begin. All training and evaluation pipelines are modular, making it easy to plug in new architectures, datasets, or metrics.

```bash
git clone https://github.com/your-username/uls23-segmentation.git
cd uls23-segmentation

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## Configuration
Update the paths to your dataset in the relevant files inside ```src/``` (such as ```seg_dataset.py```) so that the code can correctly locate and load your data.

## Running Code
To execute a script, use Python’s -m flag from the root directory of the repository. For example:
```
python -m src.training
```
This ensures all relative imports work properly.

## Logging & Visualization

When training a model, PyTorch Lightning will automatically create experiment logs inside the ```lightning_logs/``` folder in the root directory.

To visualize these logs using TensorBoard, run:
```
tensorboard --logdir_spec=./lightning_logs
```

Then open the provided URL (usually http://localhost:6006) in your browser to view training progress, metrics, and more.