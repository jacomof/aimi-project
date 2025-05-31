#!/usr/bin/env bash

# ADAPT TO YOUR USE CASE IF YOU ARE USING SLURM
# ---------------------------------------------
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
# ---------------------------------------------


# execute train CLI
# assumes current directory is aimi-project

# Select correct preprocessed data directory and nnUNet results
# for your use case.
export nnUNet_preprocessed="/d/hpc/home/jf73497/projects/aimi-project-data/preprocessed/"
export nnUNet_results="/d/hpc/home/jf73497/projects/aimi-project-data/"

source venv/bin/activate
nnUNetv2_find_best_configuration 001 -c 2d 3d_fullres -tr nnUNetTrainer_ULS_200_QuarterLR