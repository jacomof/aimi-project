#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/segformer_oversampled-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/segformer_oversampled-%J.err
#SBATCH --job-name="segformer preprocessing"
#SBATCH --mem-per-gpu=64G


# execute train CLI
# assumes current directory is aimi-project

# checking cuda visible devices
echo $CUDA_VISIBLE_DEVICES
# checking gpu availability
nvidia-smi

source ../../../venv_segformer/bin/activate

# checking if cuda is available
python -c "import torch; print(torch.cuda.is_available())"

echo "Environment activated!"
echo "Running segformer training..."
accelerate launch --config_file ./gpu_accelerate.yaml run_experiment.py 