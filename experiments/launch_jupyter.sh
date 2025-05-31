#!/bin/bash
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


# execute train CLI
# assumes current directory is aimi-project
#hostname -I
source SegFormer3D-main/venv_segformer/bin/activate
jupyter lab --ip=0.0.0.0 --port 1200