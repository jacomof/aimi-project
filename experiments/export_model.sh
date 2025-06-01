#!/bin/bash

# ADAPT TO YOUR USE CASE IF YOU ARE USING SLURM
# ---------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/export_model-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/export_model-%J.err
#SBATCH --job-name="export nnUNet"
#SBATCH --mem-per-gpu=32G
#SBATCH --exclude=gwn03,gwn02,wn207,wn219
# ---------------------------------------------

# This is a utility script to export an nnUNet model in a SLURM cluster.
# It assumes the current directory is aimi-project.

export nnUNet_raw="../aimi-project-data/raw/"
export nnUNet_preprocessed="../aimi-project-data/preprocessed_complete/"
export nnUNet_results="../aimi-project-data/complete_data/"

source venv/bin/activate
echo "Environment activated!"
nnUNetv2_export_model_to_zip -d 001 -o ../aimi-project-data/nnunet_800.zip \
 -tr nnUNetTrainer_ULS_800_QuarterLR -p nnUNetResEncUNetMPlans \
 -c 3d_fullres -f all --not_strict -chk checkpoint_best.pth
