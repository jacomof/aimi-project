#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/segformer-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/segformer-%J.err
#SBATCH --job-name="segformer preprocessing"
#SBATCH --mem-per-gpu=64G
#SBATCH --exclude=gwn03,gwn02,gwn08,wn205

# notes:
# Assumes current directory is aimi-project.


# Activating the virtual environment
source ../../../venv_segformer/bin/activate
echo "Environment activated!"

# Checking if cuda is available
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); torch.__version__"

# Checking state of the graphics card
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi


echo "Running segformer training..."
accelerate launch --config_file ./gpu_accelerate.yaml run_experiment.py 