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
# Runs preprocessing for nnUNet on the complete dataset.
# Run before running experiment_nnUNet_complete_data.sh.
# It assumes the raw data is already available in the
# raw_complete directory. This directory should contain
# all images inimagesTr and labelsTr for training, 
# and imagesTs and labelsTs should be left empty. 
# The preprocessed data will be saved in the
# preprocessed_complete directory.


# execute train CLI
# assumes current directory is aimi-project

# raw_complete should have all data in imagesTr and labelsTr
# and no data in imagesTs and labelsTs
export nnUNet_raw="/d/hpc/home/jf73497/projects/aimi-project-data/raw_complete/"
export nnUNet_preprocessed="/d/hpc/home/jf73497/projects/aimi-project-data/preprocessed_complete/"
export nnUNet_results="/d/hpc/home/jf73497/projects/aimi-project-data/"
source venv/bin/activate
echo "Environment activated"
echo "Launching training"
nnUNetv2_plan_and_preprocess -d 001 -c 3d_fullres -pl nnUNetPlannerResEncM
