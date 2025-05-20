#!/usr/bin/env bash
#SBATCH --partition=csedu-prio,csedu
#SBATCH --account=cseduimc037
#SBATCH --qos=csedu-preempt
#SBATCH --mem=15G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/jfigueira/aimi-project/logs/experiment_nnUNet_%j_%a.out
#SBATCH --error=/home/jfigueira/aimi-project/logs/experiment_nnUNet_%j_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

### notes
# this experiment is meant to try out ResNet 18, 34, 50 and 101 on cifar10


# execute train CLI
# assumes current directory is aimi-project
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_fullres 4 --npz -tr nnUNetTrainer_ULS_200_QuarterLR