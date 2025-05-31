import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

DATASET_CONSTANTS_FROM_NNUNET = {
    "max": 3115.0,
    "mean": 24.188657449271258,
    "median": 40.0,
    "min": -1411.0,
    "percentile_00_5": -931.0,
    "percentile_99_5": 1974.0,
    "std": 535.6477268829442,
}


class SegDatasetCreator:
    def __init__(
        self,
        base_path,
        target_spacing=None,
        subsample=None,
    ):
        self.datasets = sorted(os.listdir(base_path))
        self.image_label_pairs = self.find_image_label_pairs(base_path)

        self.data = pd.DataFrame.from_records(
            self.image_label_pairs, columns=["image_path", "label_path"]
        )
        # Label each pair with their respective dataset for stratification
        self.data["dataset"] = self.data["image_path"].apply(
            lambda path: next(d for d in self.datasets if d in path)
        )

        self.target_spacing = target_spacing

        self.subsample = subsample
        if subsample is not None:
            self.data = self._subsample_data(self.data, subsample)

    def find_image_label_pairs(self, root_dir):
        image_label_pairs = []

        for dirpath, dirnames, filenames in sorted(os.walk(root_dir)):
            if os.path.basename(dirpath) == "imagesTr":
                labels_dir = os.path.join(os.path.dirname(dirpath), "labelsTr")
                for image_filename in sorted(os.listdir(dirpath)):
                    image_path = os.path.join(dirpath, image_filename)
                    label_path = os.path.join(labels_dir, image_filename)

                    image_label_pairs.append((image_path, label_path))

        return image_label_pairs

    def _subsample_data(self, data, subsample, groupby="dataset"):
        """Subsample the dataframe either by a fraction or a fixed number per dataset."""
        grouped = data.groupby(groupby)
        if isinstance(subsample, float):
            data = grouped.sample(frac=subsample, random_state=42)
        elif isinstance(subsample, int):
            data = grouped.sample(n=min(subsample, len(grouped)), random_state=42)
        return data.reset_index(drop=True)

    def get_train_val_datasets(
        self,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=True,
        transform=None,
    ):
        # Stratify by dataset
        if stratify:
            stratify = self.data["dataset"]

        X_train, X_val, y_train, y_val = train_test_split(
            self.data["image_path"],
            self.data["label_path"],
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        train_dataset = SegDataset(
            X_train, y_train, target_spacing=self.target_spacing, transform=transform
        )
        val_dataset = SegDataset(
            X_val, y_val, target_spacing=self.target_spacing, transform=transform
        )

        return train_dataset, val_dataset

    def get_train_val_dataloaders(
        self,
        train_dataset,
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Validation is typically not shuffled
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        return train_loader, val_loader

    def create_train_val(
        self,
        test_size=0.2,
        random_state=42,
        shuffle_datasets=True,
        stratify=True,
        transform=None,
        batch_size=32,
        shuffle_dataloaders=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    ):
        train_dataset, val_dataset = self.get_train_val_datasets(
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle_datasets,
            stratify=stratify,
            transform=transform,
        )
        train_loader, val_loader = self.get_train_val_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_dataloaders,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        return train_loader, val_loader

    def get_kfold_datasets(
        self,
        n_splits=5,
        random_state=42,
        shuffle=True,
        stratify=True,
        transform=None,
    ):
        """
        Returns a list of (train_dataset, val_dataset) pairs for k-fold cross-validation.
        """
        if stratify:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
            stratify_labels = self.data["dataset"]
        else:
            splitter = KFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
            stratify_labels = None

        image_paths = self.data["image_path"]
        label_paths = self.data["label_path"]

        if stratify:
            splits = splitter.split(image_paths, stratify_labels)
        else:
            splits = splitter.split(image_paths)

        dataset_folds = []
        for train_idx, val_idx in splits:
            X_train, X_val = image_paths[train_idx], image_paths[val_idx]
            y_train, y_val = label_paths[train_idx], label_paths[val_idx]

            train_dataset = SegDataset(
                X_train,
                y_train,
                target_spacing=self.target_spacing,
                transform=transform,
            )
            val_dataset = SegDataset(
                X_val, y_val, target_spacing=self.target_spacing, transform=transform
            )

            dataset_folds.append((train_dataset, val_dataset))

        return dataset_folds

    def get_kfold_dataloaders(
        self,
        dataset_folds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    ):
        dataloader_folds = []

        for train_dataset, val_dataset in dataset_folds:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,  # Validation is typically not shuffled
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            dataloader_folds.append((train_loader, val_loader))

        return dataloader_folds

    def create_kfold(
        self,
        n_splits=5,
        random_state=42,
        shuffle_datasets=True,
        stratify=True,
        transform=None,
        batch_size=32,
        shuffle_dataloaders=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    ):
        dataset_folds = self.get_kfold_datasets(
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle_datasets,
            stratify=stratify,
            transform=transform,
        )

        dataloader_folds = self.get_kfold_dataloaders(
            dataset_folds,
            batch_size=batch_size,
            shuffle=shuffle_dataloaders,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        return dataloader_folds


class SegDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        target_spacing=None,
        transform=None,
    ):
        super().__init__()

        assert len(image_paths) == len(label_paths), (
            "Image and label DataFrames must be the same length"
        )

        self.image_paths = image_paths.reset_index(drop=True)
        self.label_paths = label_paths.reset_index(drop=True)

        self.target_spacing = target_spacing

        # Optional transform for data augmentation
        self.transform = transform

    def load_image(
        self,
        image_path,
        is_label,
    ):
        sitk_image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(sitk_image)
        voxel_spacing = sitk_image.GetSpacing()

        if self.target_spacing:
            image_array = self.resample_image(sitk_image, is_label=is_label)

        return image_array, voxel_spacing

    def resample_image(self, image, is_label=False):
        # We dont use this because apparently neither did the baseline training
        original_size = image.GetSize()

        # Define resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(original_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

        # Use nearest neighbor for labels, linear for images
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        resampled_image = resampler.Execute(image)
        return sitk.GetArrayFromImage(resampled_image)

    def normalize_ct(self, image):
        image = np.clip(
            image,
            DATASET_CONSTANTS_FROM_NNUNET["percentile_00_5"],
            DATASET_CONSTANTS_FROM_NNUNET["percentile_99_5"],
        )
        image = (
            image - DATASET_CONSTANTS_FROM_NNUNET["mean"]
        ) / DATASET_CONSTANTS_FROM_NNUNET["std"]
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths.iloc[idx]
        label_path = self.label_paths.iloc[idx]

        image, image_spacing = self.load_image(image_path, is_label=False)
        label, label_spacing = self.load_image(label_path, is_label=True)

        image = self.normalize_ct(image)

        # Add channel dimension because we are working with 3d images
        image_tensor = torch.from_numpy(image[None, :])
        label_tensor = torch.from_numpy(label[None, :])

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor.float(), label_tensor.long()

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    test = SegDatasetCreator(r"..\oncology-ULS-fast-for-challenge\nnUNet_raw")
    train_dataset, val_dataset = test.get_train_val_datasets()
    loaders = test.get_train_val_dataloaders(train_dataset, val_dataset)

    folds = test.get_kfold_datasets()
    folds = test.get_kfold_dataloaders(folds)

    a = test.create_train_val()
    b = test.create_kfold()

    print(a)
    print(b)

    for data, label in train_dataset:
        print(data.shape, label.shape)
    print(train_dataset[0][0].shape)

    # This was used to test if the data was corrupt
    # from tqdm import tqdm

    # u = 0
    # for path_image, path_label in tqdm(test.image_label_pairs):
    #     image = sitk.ReadImage(path_image)
    #     label = sitk.ReadImage(path_label)

    #     if image.GetSpacing() != label.GetSpacing():
    #         print(f"bad: {path_image}")
    #         u += 1

    # print("total corrupted: " + str(u))
