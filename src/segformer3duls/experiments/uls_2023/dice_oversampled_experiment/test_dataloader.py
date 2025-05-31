import sys
sys.path.append("../../../")
import os
import torch
print("Loading libraries...")
from dataloaders.build_dataset import build_dataset, build_dataloader

print("Loading dataset...")
dataset = build_dataset(
    dataset_type="uls2023_seg",
    dataset_args={
        "root": "../../../data/uls2023_seg",
        "train": True,
        "fold_id": None,
    },
)
print("Dataset loaded successfully.")
# print("Loading dataloader...")
# train_dataloader_args ={
#     "batch_size": 8, 
#     "shuffle": True,
#     "num_workers": 1,
#     "drop_last": True
# }
# loader = build_dataloader(dataset, train_dataloader_args)

base_path = "../../../data/uls2023_seg/ULS2023_Training_Data/"

import time 

print("Base path: ", base_path)
print("Starting to load data...")

sys.stdout.flush()
for f in os.listdir(base_path):
    print("Current folder: ", f)
    sys.stdout.flush()
    for file in os.listdir(os.path.join(base_path, f)):
        print("Current folder: ", f)
        if file.endswith("_im.pt"):
            print("Current file: ", file)
            sys.stdout.flush()
            image_path = os.path.join(base_path, f, file)
            label_path = os.path.join(base_path, f, file.replace("_im.pt", "_label.pt"))
            print(f"Image path: {image_path}, Label path: {label_path}")
            image = torch.load(image_path)
            label = torch.load(label_path)
            data = {"image": image, "label": label}
            print(f"File: {f}, Data shape: {data['image'].shape}, Label shape: {data['label'].shape}")


print("Data loaded successfully.")
# for i, data in enumerate(loader):
#     print(i)
#     print(data["image"].shape)
#     print(data["label"].shape)

print("All data loaded successfully.")
sys.stdout.flush()