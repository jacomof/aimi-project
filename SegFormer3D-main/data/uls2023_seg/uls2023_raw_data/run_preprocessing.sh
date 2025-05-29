#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/d/hpc/home/jf73497/logs/preprocess_segformer-%J.out
#SBATCH --error=/d/hpc/home/jf73497/logs/preprocess_segformer-%J.err
#SBATCH --job-name="segformer preprocessing"

### notes
# this experiment is meant to try out ResNet 18, 34, 50 and 101 on cifar10


# execute train CLI
# assumes current directory is aimi-project
source /d/hpc/home/jf73497/projects/aimi-project/SegFormer3D-main/venv_segformer/bin/activate
echo "environment activated, running script..."
python uls2023_seg_preprocess.py --cpu_count 16