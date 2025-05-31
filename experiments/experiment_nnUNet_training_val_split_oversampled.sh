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
# This experiment launches training on all the training split
# by merging validation and training data.
# It uses no ensembling or oversampling. 
# It assumes preprocessed data is already available in the
# preprocessed directory.
# Assumes current directory is aimi-project.

# assumes current directory is aimi-project
export nnUNet_raw="../aimi-project-data/raw/"
export nnUNet_preprocessed="/d/hpc/home/jf73497/projects/aimi-project-data/preprocessed_oversampled/nnUNet_preprocessed/"
export nnUNet_results="../aimi-project-data/"
source venv/bin/activate
echo "Environment activated"
echo "Launching training"
nnUNetv2_plan_and_preprocess 002 3d_fullres 0 --npz --c -tr nnUNetTrainer_ULS_400_QuarterLR -pl nnUNetResEncUNetMPlans