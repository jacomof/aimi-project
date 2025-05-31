import torch


path = "/home/jfigueira/aimi-project/SegFormer3D-main/data/uls2023_oversampled_seg/model_checkpoints/best_dice_checkpoint_continued/last_epoch_model/trainer_state_dict.pkl"
d = torch.load(path)
print(d)