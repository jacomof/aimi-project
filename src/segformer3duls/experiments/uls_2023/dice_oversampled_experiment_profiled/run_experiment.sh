#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/segformer_oversampled_profiled-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/segformer_oversampled_profiled-%J.err
#SBATCH --job-name="segformer preprocessing"
#SBATCH --mem-per-gpu=64G
#SBATCH --exclude=gwn03,gwn02


# execute train CLI
# assumes current directory is aimi-project

source ../../../venv_segformer/bin/activate

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# checking if cuda is available
python -c "import torch; print(torch.cuda.is_available())"
accelerate launch --config_file ./gpu_accelerate.yaml run_experiment.py 