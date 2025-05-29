#!/usr/bin/env bash
#SBATCH --partition=csedu-prio,csedu
#SBATCH --account=cseduimc037
#SBATCH --qos=csedu-preempt
#SBATCH --mem=15G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/jfigueira/aimi-project/logs/segformer_dice_oversampled_profiled_%j_%a.out
#SBATCH --error=/home/jfigueira/aimi-project/logs/segformer_dice_oversampled_profiled_%j_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL


# execute train CLI
# assumes current directory is aimi-project
source ../../../venv_segformer/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 run_experiment.py