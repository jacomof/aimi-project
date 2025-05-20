import torch.nn.functional as F
from matplotlib import pyplot as plt
from src.seg_dataset import SegDatasetCreator


def _downsample_mask(mask_tensor, scale=0.25):
    """
    Downsamples the volume using trilinear interpolation.
    """
    mask_down = F.interpolate(
        mask_tensor.unsqueeze(0).float(),
        scale_factor=scale,
        mode="trilinear",
        align_corners=False,
    ).bool()
    return mask_down.squeeze(0) > 0.5


def visualize_3d_voi(mask_tensor, save_path="./my_fig.png"):
    """
    Visualizes a binary 3D volume using matplotlib with styled axis and background.
    """
    assert mask_tensor.shape[0] == 1, "Expected shape (1, D, H, W)"

    small_mask = _downsample_mask(mask_tensor, scale=0.25)

    volume = small_mask.squeeze(0).cpu().numpy() > 0.5

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")  # Axis plane color

    ax.voxels(volume, facecolors="white", edgecolor="white", alpha=1.0)

    # Axis labels and tick color (black)
    ax.set_xlabel("X", color="black")
    ax.set_ylabel("Y", color="black")
    ax.set_zlabel("Z", color="black")
    ax.tick_params(colors="black")

    # Set white axis lines
    ax.xaxis.line.set_color("black")
    ax.yaxis.line.set_color("black")
    ax.zaxis.line.set_color("black")

    # Set the background planes of each axis to black manually
    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))

    # # Remove gridlines for cleaner look
    ax.grid()

    plt.title("3D Segmentation Volume", color="black")
    plt.savefig(save_path)
    plt.close()


def visualize_3d_voi_comparison(
    gt_tensor, pred_tensor, save_path="./comparison_fig.png"
):
    """
    Visualizes ground truth and predicted binary 3D volumes side by side.

    Args:
        gt_tensor (torch.Tensor): Ground truth tensor of shape (1, D, H, W).
        pred_tensor (torch.Tensor): Prediction tensor of shape (1, D, H, W).
        save_path (str): Where to save the output image.
    """
    assert gt_tensor.shape[0] == 1 and pred_tensor.shape[0] == 1, (
        "Expected shape (1, D, H, W) for both tensors"
    )

    def preprocess(tensor):
        tensor = _downsample_mask(tensor, scale=0.25)
        return tensor.squeeze(0).cpu().numpy() > 0.5

    gt_volume = preprocess(gt_tensor)
    pred_volume = preprocess(pred_tensor)

    fig = plt.figure(figsize=(16, 8), facecolor="white")

    # Ground Truth subplot
    ax1 = fig.add_subplot(121, projection="3d", facecolor="white")
    ax1.voxels(gt_volume, facecolors="white", edgecolor="white", alpha=1.0)
    ax1.set_title("Ground Truth", color="black")
    for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
        axis.set_pane_color((0, 0, 0, 1))
        axis.line.set_color("black")
    ax1.set_xlabel("X", color="black")
    ax1.set_ylabel("Y", color="black")
    ax1.set_zlabel("Z", color="black")
    ax1.tick_params(colors="black")
    ax1.grid()

    # Prediction subplot
    ax2 = fig.add_subplot(122, projection="3d", facecolor="white")
    ax2.voxels(pred_volume, facecolors="blue", edgecolor="white", alpha=1.0)
    ax2.set_title("Prediction", color="black")
    for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
        axis.set_pane_color((0, 0, 0, 1))
        axis.line.set_color("black")
    ax2.set_xlabel("X", color="black")
    ax2.set_ylabel("Y", color="black")
    ax2.set_zlabel("Z", color="black")
    ax2.tick_params(colors="black")
    ax2.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_slice_voi(mask_tensor, slice_idx):
    """
    Visualize a single slice from a 3D binary mask tensor.

    Args:
        mask_tensor (Tensor): Binary mask tensor of shape (1, 64, 128, 128).
        slice_idx (int): Index of the slice to visualize.
    """
    assert 0 <= slice_idx < mask_tensor.shape[1], (
        f"slice_idx should be in [0, {mask_tensor.shape[1] - 1}]"
    )

    slice_img = mask_tensor[0, slice_idx]  # Shape: (128, 128)

    plt.figure(figsize=(4, 4))
    plt.imshow(slice_img.cpu(), cmap="gray")
    plt.title(f"Slice {slice_idx}")
    plt.axis("off")
    plt.show()
    plt.close()


if __name__ == "__main__":
    creator = SegDatasetCreator(base_path=r"..\oncology-ULS-fast-for-challenge\nnUNet_raw", subsample=0.05)
    dataset = creator.get_train_val_datasets()

    mask = dataset[0][3][1]

    # Visualize all slices
    visualize_3d_voi(mask)

    # Visualize one slice (e.g., slice 30)
    visualize_slice_voi(mask, slice_idx=30)
