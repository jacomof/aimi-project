#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --qos=csedu-small
#SBATCH --mem=15G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/jfigueira/aimi-project/logs/experiment_nnUNet_%j_%a.out
#SBATCH --error=/home/jfigueira/aimi-project/logs/experiment_nnUNet_%j_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL


# execute train CLI
# assumes current directory is aimi-project
#hostname -I
source venv/bin/activate
jupyter lab --ip=0.0.0.0 --port 1200