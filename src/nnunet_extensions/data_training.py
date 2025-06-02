import os
import shutil
import random
import subprocess
import argparse
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def convert_to_mixed_dataset(source_root: str, target_root: str, split_percent: float = 0.2):
    """ Convert multiple datasets of the ULS2023 challenge into a single mixed dataset.
    This function merges multiple datasets from the source root into a single dataset in the target root.

    Args:
        source_root (str): Path to the root directory containing the source datasets.
        target_root (str): Path to the root directory where the mixed dataset will be saved.
        split_percent (float): Percentage of data to be used for testing. Default is 0.2 (20%).
    """
    
    target_dataset_name = "Dataset001_MIX"
    target_dataset_folder = os.path.join(target_root, target_dataset_name)
    tgt_imagesTr = os.path.join(target_dataset_folder, "imagesTr")
    tgt_labelsTr = os.path.join(target_dataset_folder, "labelsTr")
    tgt_imagesTs = os.path.join(target_dataset_folder, "imagesTs")
    tgt_labelsTs = os.path.join(target_dataset_folder, "labelsTs")

    maybe_mkdir_p(tgt_imagesTr)
    maybe_mkdir_p(tgt_labelsTr)
    maybe_mkdir_p(tgt_imagesTs)
    maybe_mkdir_p(tgt_labelsTs)

    datasets = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    print("Found datasets:", datasets)

    temp_pairs = []
    for dataset in datasets:
        dataset_path = os.path.join(source_root, dataset)
        subdatasets = [d for d in os.listdir(dataset_path)
                       if os.path.isdir(os.path.join(dataset_path, d)) and d not in ["imagesTr", "labelsTr"]]

        if subdatasets:
            for subdataset in subdatasets:
                src_imagesTr_sub = os.path.join(dataset_path, subdataset, "imagesTr")
                src_labelsTr_sub = os.path.join(dataset_path, subdataset, "labelsTr")

                image_files_sub = sorted(glob(os.path.join(src_imagesTr_sub, "*.nii.gz")))
                for img_path in image_files_sub:
                    label_path = os.path.join(src_labelsTr_sub, os.path.basename(img_path))
                    assert os.path.exists(label_path), f"Missing label for {img_path}"
                    temp_pairs.append((img_path, label_path))
        else:
            src_imagesTr = os.path.join(dataset_path, "imagesTr")
            src_labelsTr = os.path.join(dataset_path, "labelsTr")

            image_files = sorted(glob(os.path.join(src_imagesTr, "*.nii.gz")))
            for img_path in image_files:
                label_path = os.path.join(src_labelsTr, os.path.basename(img_path))
                assert os.path.exists(label_path), f"Missing label for {img_path}"
                temp_pairs.append((img_path, label_path))

    random.shuffle(temp_pairs)

    total = len(temp_pairs)
    num_test = int(total * split_percent)
    num_train = total - num_test

    for idx, (img_path, lbl_path) in enumerate(temp_pairs):
        base_name = f"MIX_{idx:05d}"
        if idx < num_train:
            tgt_img = os.path.join(tgt_imagesTr, base_name + "_0000.nii.gz")
            tgt_lbl = os.path.join(tgt_labelsTr, base_name + ".nii.gz")
        else:
            tgt_img = os.path.join(tgt_imagesTs, base_name + "_0000.nii.gz")
            tgt_lbl = os.path.join(tgt_labelsTs, base_name + ".nii.gz")

        shutil.copy(img_path, tgt_img)
        shutil.copy(lbl_path, tgt_lbl)

    channel_names = {0: "CT"}
    labels = {"background": 0, "tumor": 1}

    generate_dataset_json(
        output_folder=target_dataset_folder,
        channel_names=channel_names,
        labels=labels,
        num_training_cases=num_train,
        file_ending=".nii.gz",
        dataset_name=target_dataset_name,
        description="Merged and shuffled dataset from all available sources with split",
        converted_by="Mauro",
    )

def train_on_single_gpu(dataset_id: str, trainer_name: str, folds: int = 5):
    """ Train the nnUNet model on a single GPU.

    This function sets up the training environment and runs the training command for the specified dataset and trainer.
    It can train multiple folds of the dataset for the specified modality by changing
    the .
    
    Args:
        dataset_id (str): The ID of the dataset to train on.
        trainer_name (str): The name of the trainer to use for training.
    """
    modalities = ["3d_fullres"]
    print("Starting training... ")
    for modality in modalities:
        for fold in range(folds):
            cmd = [
                "CUDA_VISIBLE_DEVICES=0",
                "nnUNetv2_train",
                dataset_id,
                modality,
                str(fold),
                "--npz",
                "--c",
                "-tr",
                trainer_name,
                "-p",
                "nnUNetResEncUNetMPlans"
            ]
            full_cmd = " ".join(cmd)
            print("Running:", full_cmd)
            subprocess.run(full_cmd, shell=True, check=True)

    print("Finding best configuration...")
    subprocess.run(["nnUNetv2_find_best_configuration", dataset_id, "-c"] + modalities, check=True)


if __name__ == "__main__":
    trainer = "nnUNetTrainer_ULS_400_QuarterLR"
    source_root = "../../../aimi-project-data/data/fully_annotated_data_copy"
    target_root = "../../../aimi-project-data/data/raw_oversampled_mauro/"
    dataset_id = "002"

    convert_to_mixed_dataset(source_root, target_root, split_percent=0.0)

    print("Running nnUNetv2_plan_and_preprocess...")
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "-c", "3d_fullres", "-pl", "nnUNetPlannerResEncM"], check=True)
    print("Running nnUNetv2 training...")
    train_on_single_gpu(dataset_id, trainer)

