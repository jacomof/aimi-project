#!/bin/bash

# ADAPT TO YOUR USE CASE IF YOU ARE USING SLURM
# ---------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/run_nnUNet-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/run_nnUNet-%J.err
#SBATCH --job-name="nnUNet training with all training data"
#SBATCH --mem-per-gpu=64G
#SBATCH --exclude=gwn03,gwn02
# ---------------------------------------------

### notes
# this experiment launches training on all the dataset without
# ensembling or resampling.
# It assumes preprocessed data is already available in the
# preprocessed_complete directory.

# execute train CLI
# assumes current directory is aimi-project
export nnUNet_raw="/d/hpc/home/jf73497/projects/aimi-project-data/raw/"
export nnUNet_preprocessed="/d/hpc/home/jf73497/projects/aimi-project-data/preprocessed_complete/"
export nnUNet_results="/d/hpc/home/jf73497/projects/aimi-project-data/complete_data/"
source venv/bin/activate
echo "Environment activated"
echo "Launching training"
nnUNetv2_train 001 3d_fullres all --npz --c -tr nnUNetTrainer_ULS_800_QuarterLR -p nnUNetResEncUNetMPlans