import torch
import os
import sys

sys.path.append("../../../")

def save_checkpoint(filename: str) -> None:
    """_summary_

    Args:
        filename (str): _description_
    """
    # saves the ema model checkpoint if availabale
    # TODO: ema saving untested (deprecated)
    # if self.ema_enabled and self.val_ema_model:
    #     checkpoint = {
    #         "state_dict": self.val_ema_model.state_dict(),
    #         "optimizer": self.optimizer.state_dict(),
    #     }
    #     torch.save(checkpoint, f"{os.path.dirname(filename)}/ema_model_ckpt.pth")
    #     self.val_ema_model = (
    #         None  # set ema model to None to avoid duplicate model saving
    #     )

    # standard model checkpoint
    # self.accelerator.save_state(filename, safe_serialization=False)
    trainer_state_dict = {
        "current_epoch": 59,
        "best_val_ema_dice": 0,
        "best_val_loss": 0,
        "best_val_dice": 0,
        "best_val_uls_metric": 0,
        "best_train_loss": 0,
    }

    # save the model state dict
    trainer_state_dict_path = os.path.join(filename, "trainer_state_dict.pkl")
    torch.save(trainer_state_dict, trainer_state_dict_path)

save_checkpoint("/home/jfigueira/aimi-project/data/segformer3d_checkpoints/best_dice_checkpoint/")
