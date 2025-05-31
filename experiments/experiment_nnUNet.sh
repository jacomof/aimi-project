#!/bin/bash

# ADAPT TO YOUR USE CASE IF YOU ARE USING SLURM
# ---------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/run_nnUNet-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/run_nnUNet-%J.err
#SBATCH --job-name="nnUNet training"
#SBATCH --mem-per-gpu=32G
# ---------------------------------------------

### notes
# This is a utility script that performs all steps necessary 
# to train a nnUNet model in a single run. 
# data_training.py takes care of the preprocessing and training.
# It trains multiple folds for a 5-fold ensemble sequencially.
# It assumes raw data is already available in the
# raw directory.
# Trainer and plans are set in the python script.


# assumes current directory is aimi-project

export nnUNet_raw="/d/hpc/home/jf73497/projects/aimi-project-data/raw/"
export nnUNet_preprocessed="/d/hpc/home/jf73497/projects/aimi-project-data/preprocessed/"
export nnUNet_results="/d/hpc/home/jf73497/projects/aimi-project-data/"

source venv/bin/activate
echo "Environment activated"
echo "Launching training"
python data_training.py