#!/bin/bash

# ADAPT TO YOUR USE CASE IF YOU ARE USING SLURM
# ---------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/launch_jupyter-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/launch_jupyter-%J.err
#SBATCH --job-name="nnUNet training"
#SBATCH --mem-per-gpu=32G
#SBATCH --exclude=gwn03,gwn02
# ---------------------------------------------

# This is a utility script to launch a Jupyter Lab server with 
# the SegFormer3DULS environment for interactive work.
# Very useful for debugging and development!
# It assumes the current directory is aimi-project.


source src/segformer3duls/venv_segformer/bin/activate
echo "Environment activated"
echo "Launching Jupyter Lab"
jupyter lab --ip=0.0.0.0 --port 1200