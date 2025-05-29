#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/run_nnUNet-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/run_nnUNet-%J.err
#SBATCH --job-name="nnUNet training with all training data with resampling."
#SBATCH --mem-per-gpu=64G



# execute train CLI
# assumes current directory is aimi-project
export nnUNet_raw="/d/hpc/home/jf73497/projects/aimi-project-data/raw/"
export nnUNet_preprocessed="/d/hpc/home/jf73497/projects/aimi-project-data/preprocessed_oversampled/nnUNet_preprocessed/"
export nnUNet_results="/d/hpc/home/jf73497/projects/aimi-project-data/"
source venv/bin/activate
echo "Environment activated"
echo "Launching training"
python data_training_all_resampled.py