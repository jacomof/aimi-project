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
# this experiment launches training on all the dataset with
# resampling and no ensembling.
# It assumes preprocessed data is already available in the
# preprocessed_oversampled_complete directory.
# assumes current directory is aimi-project.

# raw_complete should have all data in imagesTr and labelsTr
# and no data in imagesTs and labelsTs
export nnUNet_raw="../aimi-project-data/raw/"
export nnUNet_preprocessed="../aimi-project-data/preprocessed_oversampled_complete/"
export nnUNet_results="../aimi-project-data/complete_data/"
source venv/bin/activate
echo "Environment activated"
echo "Launching training"
nnUNetv2_train 002 3d_fullres all --npz --c -tr nnUNetTrainer_ULS_800_QuarterLR -p nnUNetResEncUNetMPlans