import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from src.arch.simple_unet import UNet3D
from src.seg_dataset import SegDatasetCreator
from src.utils.competition_metric import ULS23_evaluator
from src.utils.loss_fn import (
    DC_and_BCE_loss,
    DC_and_CE_loss,
    MemoryEfficientSoftDiceLoss,
)
from src.utils.visualizations import visualize_3d_voi_comparison

DEBUG = False


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn=DC_and_BCE_loss,
        # Optimizer
        lr=1e-3,
        optimizer=torch.optim.AdamW,
        # Scheduler
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        # For inference
        threshold=0.5,
    ):
        super().__init__()
        self.model = model
        # DC_and_BCE_loss
        self.loss_fn = loss_fn({}, {})
        # nn.BCEWithLogitsLoss
        # self.loss_fn = loss_fn(reduction="none")

        # Optimizer
        self.lr = lr
        self.optimizer = optimizer
        # Scheduler
        self.lr_scheduler = lr_scheduler
        # ULS competition evaluator
        self.evaluator = ULS23_evaluator()
        # For inference
        self.threshold = threshold

    def forward(self, x):
        y = self.model(x)
        return y
    
    def predict_from_logits(self, logits):
        return F.sigmoid(logits) > self.threshold

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Loss
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Prediction
        prediction = self.predict_from_logits(logits)
        
        if DEBUG:
            visualize_3d_voi_comparison(y[0], prediction[0], save_path="./my_fig.png")

        # Score
        score = self.evaluator.ULS_score_metric(prediction, y)
        self.log("ULS_score", score, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": self.lr_scheduler(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            ),
            "monitor": "val_loss",  # The metric to monitor
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    # Change for training
    torch.set_float32_matmul_precision("medium")

    # Init your custom dataset loader
    # SET THE PATH TO THE CORRECT PATH PLEASEEEEEE (if you are in the cluster, needed)
    creator = SegDatasetCreator(
        base_path=r"..\oncology-ULS-fast-for-challenge\nnUNet_raw",
        subsample=0.55,
        target_spacing=None,
    )

    train_loader, val_loader = creator.create_train_val(
        test_size=0.2,
        random_state=42,
        shuffle_datasets=True,
        stratify=True,
        transform=None,
        batch_size=1,
        shuffle_dataloaders=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512])
    lightning_model = SegmentationModel(
        model,
        lr=3e-5,
    )

    # Callbacks
    # Save best ULS model
    checkpoint_cb = ModelCheckpoint(monitor="ULS_score", save_top_k=1, mode="max")
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=100,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_monitor],
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
