import torch
# import time
import numpy as np
# import torch.nn as nn
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure, binary_dilation, binary_erosion
# import matplotlib.pyplot as plt

# AUGMENTATIONS 

# flip_axes = [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]
# shifts = [
#     (0, 0, 0),
#     (10, 0, 0), (-10, 0, 0),
#     (0, 10, 0), (0, -10, 0),
#     (0, 0, 10), (0, 0, -10),
# ]

# flip_axes = [[]]
# shifts = [
#     (0, 0, 0),
# ]

def test_time_dilation(input_tensor):
    # Convert to NumPy and squeeze batch and channel dimensions
    volume = input_tensor.squeeze().cpu().numpy()
    
    # Apply binary dilation
    dilated = binary_dilation(volume, structure=np.ones((3, 3, 3)))

    return dilated.astype(np.uint8)

def test_time_2xdilation(input_tensor):
    volume = input_tensor.squeeze().cpu().numpy()
    
    # Apply binary dilation twice
    dilated = binary_dilation(volume, structure=np.ones((3, 3, 3)))
    dilated = binary_dilation(dilated, structure=np.ones((3, 3, 3)))

    return dilated.astype(np.uint8)

def test_time_opening(input_tensor):
    volume = input_tensor.squeeze().cpu().numpy()
    
    # Apply binary opening: erosion followed by dilation
    opened = binary_erosion(volume, structure=np.ones((3, 3, 3)))
    opened = binary_dilation(opened, structure=np.ones((3, 3, 3)))

    return opened.astype(np.uint8)

def test_time_closing(input_tensor):
    volume = input_tensor.squeeze().cpu().numpy()
    
    # Apply binary closing: dilation followed by erosion
    closed = binary_dilation(volume, structure=np.ones((3, 3, 3)))
    closed = binary_erosion(closed, structure=np.ones((3, 3, 3)))

    return closed.astype(np.uint8)

def test_time_shift(model, input_tensor, threshold=0.5):
    predictions = []

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    shifts = [
        (0, 0, 0),
        (5, 0, 0), 
        # (-5, 0, 0),
        # (0, 5, 0), 
        (0, -5, 0),
        # (0, 0, 5), 
        (0, 0, -5),
    ]

    with torch.no_grad():
        for dz, dx, dy in shifts:
            augmented = input_tensor.clone()

            # Shift
            augmented = torch.roll(augmented, shifts=(dz, dx, dy), dims=(2, 3, 4))

            # Model inference
            logits = model.forward(augmented)
            logits = logits[:, 1:, ...]
            pred = torch.sigmoid(logits[:, :, ...])
            pred = pred > 0.5

            # Undo shift
            pred = torch.roll(pred, shifts=(-dz, -dx, -dy), dims=(2, 3, 4))
            predictions.append(pred.float())

    avg_prediction = torch.stack(predictions).mean(dim=0, keepdims=True)

    # --- Morphological smoothing ---
    # Convert to NumPy for processing
    pred_np = avg_prediction.squeeze().cpu().numpy()

    # Binarize
    binary_pred = pred_np > threshold
    return binary_pred

# def test_time_augmentation(model, input_tensor, threshold=0.5, morph_op='opening'):
#     """
#     Perform Test-Time Augmentation (TTA) with flip, circular shift, and postprocessing smoothing.

#     Args:
#         model (torch.nn.Module): The model to evaluate.
#         input_tensor (torch.Tensor): A tensor of shape (1, 1, Z, X, Y)
#         threshold (float): Threshold to binarize output before morphological operation
#         morph_op (str): 'closing' or 'opening'

#     Returns:
#         torch.Tensor: Postprocessed prediction after TTA.
#     """
#     model.eval()
#     device = next(model.parameters()).device
#     input_tensor = input_tensor.to(device)

#     predictions = []

#     with torch.no_grad():
#         for flip in flip_axes:
#             for dz, dx, dy in shifts:
#                 augmented = input_tensor.clone()

#                 # Flip
#                 if flip:
#                     augmented = torch.flip(augmented, dims=flip)

#                 # Shift
#                 augmented = torch.roll(augmented, shifts=(dz, dx, dy), dims=(2, 3, 4))

#                 # Model inference
#                 logits = model.forward(augmented)
#                 logits = logits[:, 1:, ...]
#                 pred = torch.sigmoid(logits[:, :, ...])
#                 pred = pred > 0.5

#                 # Undo shift
#                 pred = torch.roll(pred, shifts=(-dz, -dx, -dy), dims=(2, 3, 4))

#                 # Undo flip
#                 if flip:
#                     pred = torch.flip(pred, dims=flip)

#                 predictions.append(pred.float())

#     avg_prediction = torch.stack(predictions).mean(dim=0, keepdims=True)

#     # --- Morphological smoothing ---
#     # Convert to NumPy for processing
#     pred_np = avg_prediction.squeeze().cpu().numpy()

#     # Binarize
#     binary_pred = pred_np > threshold

#     # Define 3D structuring element (connectivity-1, i.e., 6-connected)
#     structure = generate_binary_structure(rank=3, connectivity=1)

#     if morph_op == 'closing':
#         smoothed = binary_closing(binary_pred, structure=structure)
#     elif morph_op == 'opening':
#         # smoothed = binary_opening(binary_pred, structure=structure)
        
#         smoothed = binary_dilation(binary_pred, structure=structure)
#         # smoothed = binary_dilation(smoothed, structure=structure)
#         # smoothed = binary_dilation(smoothed, structure=structure)

#     else:
#         raise ValueError("morph_op must be 'closing' or 'opening'")

#     # Convert back to tensor (as float)
#     smoothed_tensor = torch.tensor(smoothed, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
#     return smoothed_tensor