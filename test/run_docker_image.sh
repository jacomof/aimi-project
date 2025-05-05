docker run --rm -it \
    -v ./ct-binary-uls:/output/images/ct-binary-uls/ \
    -v ./architecture/nnUNet_results/:/opt/ml/model/Dataset601_Full_128_64/nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlans_shallow__3d_fullres_resenc/ \
    -v ./architecture/input/:/input/ \
    -v ./process.py:/opt/app/process.py \
    --gpus all \
    uls23
