# AIMI-project: ULS23 Segmentation Project

This repository contains the codebase for training and evaluating models for the ULS23 medical image segmentation challenge. All source code is organized within the ```src/``` folder and is modularly split into components related to data processing, model architecture, training, metrics, loss functions, and visualizations.

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

Make two different venv folder and:
  
  - To install new packages for nnunet, add them to requirements_nnunet.txt and execute:
  
  ```
  pip install -r requirements_nnunet.txt
  ```
  
 - To install packages for SegFormer-3D, add them to src/segformer3duls/requirements_segformer.txt:
  
  ```
  pip install -r src/segformer3duls/requirements_segformer.txt
  ```

Then, run scripts/sync_csedu.sh again. Installing package directly with pip doesn't work with the syncing script, apparently.

More details on: https://gitlab.science.ru.nl/das-dl/cluster-skeleton-python/-/tree/main?ref_type=heads


## Project Structure

```bash
├── experiments/              # Training logs, configs, results
├── scripts/                  # Shell scripts for setup and execution
├── src/                      # Core codebase
│   ├── arch/                 # Model architecture building blocks
│   ├── custom_unet/          # Modified UNet implementation
│   ├── evaluation_and_tta/   # Evaluation & test-time augmentation
│   ├── nnunet_extensions/    # Extensions and helpers for nnUNet
│   ├── segformer3duls/       # SegFormer-3D variant adapted for 3D MRI
│   └── utils/                # Metrics, logging, data utils
├── prediction_label_viz.ipynb   # Visualize predictions and labels
├── test_time_aug.ipynb          # TTA experiments and debugging
├── visualize_data.ipynb         # Dataset visualization notebook
├── .gitignore
├── README.md
└── requirements_nnunet.txt      # Python package requirements
```

# Notable Features
- Adapted SegFormer3D Architecture: Modified to handle arbitrarily sized 3D inputs, overcoming the original limitation of requiring equal dimensions.

- Efficient Loss Function: Implements a soft-Dice loss function based on nnUNet's efficient and parallelizable implementation.

- Lesion-Size-Aware Oversampling: Addresses class imbalance by oversampling underrepresented lesion sizes during training.

- Test-Time Augmentation: Incorporates morphological closing operations to enhance boundary segmentation quality.

# Running the Full Pipeline: Important Instructions
To successfully run the entire pipeline—from preprocessing and training to evaluation and visualization—please ensure you follow the setup and usage instructions associated with both the nnUNet and SegFormer frameworks. This project includes custom extensions and adaptations of both, so understanding their original architecture and environment requirements is important.

*Note: You will need to use two environments. Some components (e.g., nnUNet) depend on MONAI, PyTorch Lightning, or specific data preprocessing pipelines that differ from the transformer-based SegFormer-3D-ULS, which uses a different model backbone and uses wanbd for training.*

## Recommended Setup Guidelines
### nnUNet-related scripts and models:

Should be run in an environment that matches the original nnUNet dependencies (e.g., PyTorch ≥ 1.10, MONAI, SimpleITK).

Use the requirements_nnunet.txt file to install relevant packages.

### SegFormer-3D-ULS components:

Require custom adaptations for handling arbitrarily sized 3D patches.

The transformer model has its own architecture defined under src/segformer3duls/, and performance depends on proper data normalization (e.g., MinMax scaling). Adaptations 
to custom data shapes require setting the correct strides for the patch embedding layers
on model_parameters -> patch_stride_xy and model_parametes -> patch_stride_z on the experiment configuration files. For new experiments we recommend copying the template experiment on src -> segformer3duls -> uls_2023 -> template_experiment and adapting config and experiment runner script (run_experiment.py) as needed.

### Jupyter Notebooks:

Should be run in environments in appropriate enviroments, if you run either nnUNet or SegFormer. Some notebooks require adjusting the path of configs, data or models.

Notebooks like prediction_label_viz.ipynb and test_time_aug.ipynb assume that pretrained models have been saved to the appropriate experiments/ subdirectory.

### Logging:

SegFormer3D is set-up to log results to Weights & Biases, adjust config files with 
correct project, group and experiment parameters. To turn-off logging, set wandb_parameters -> mode to "offline".

The UNet baseline training script is implemented with PL and uses a tensorflow logger, use this tool to visualize its results.

# IMPORTANT:
Prepare datasets in the format expected by nnUNet and SegFormer (NIfTI or .npz format depending on pipeline and respecting their respective file structure standards). The raw data thus has to be preprocessed either with scripts we wrote or with scripts from the respective models.

Most vizualizations are present in notebooks in this repository.

Always verify CUDA compatibility for the installed PyTorch version in your environment.

If running on Slurm cluster, helpful scripts can be found on scripts/ to prepare virtual environments and sync them to different nodes. All credit for these scripts go to the Science cluster administrator of the Radboud Faculty of Science.

# License
This project is licensed under the MIT License.
