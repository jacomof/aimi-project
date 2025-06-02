#!/bin/bash


model_dir=$1
folds=$2


docker run \
  --gpus all \
   -v ./process_flexible.py:/opt/app/process_flexible.py \
   -v ./architecture/nnUNet_results/:/opt/ml/model/ \
  -it --entrypoint python3.10 uls2023-nnunet3d-800 \
  -m process_flexible \
  --model_dir Dataset001_MIX/nnUNetTrainer_ULS_800_QuarterLR__nnUNetResEncUNetMPlans__3d_fullres