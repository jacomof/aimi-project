#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduproject
#SBATCH --qos=csedu-normal
#SBATCH --array=0-3%4
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=./logs/experiment3_%j_%a.out
#SBATCH --error=./logs/experiment3_%j_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

### notes
# this experiment is meant to try out ResNet 18, 34, 50 and 101 on cifar10

# location of repository and data
project_dir=. # assume sbatch is called from root project dir
cifar10_folder=$project_dir/data/cifar10

# training hyperparameters
num_epochs=30
num_gpus=1
num_workers=5 # should be at most (cpus-per-task-1)

# optimization hyperparameters
learning_rate=3e-3
batch_size=64

# network hyperparameters
NETWORK_ARRAY=(resnet18 resnet34 resnet50 resnet101 resnet152)
NETWORK_INDEX=$(( $SLURM_ARRAY_TASK_ID % 5))

network=${NETWORK_ARRAY["$NETWORK_INDEX"]}

# execute train CLI
source "$project_dir"/venv/bin/activate
python "$project_dir"/run_resnet.py \
  --data-folder "$cifar10_folder" \
  --max-epochs $num_epochs \
  --gpus $num_gpus \
  --num-workers $num_workers \
  --learning-rate $learning_rate \
  --batch-size $batch_size \
  --network "$network"