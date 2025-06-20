import os
import torch
import evaluate
from tqdm import tqdm
from typing import Dict
from copy import deepcopy
from termcolor import colored
from torch.utils.data import DataLoader
import monai
from metrics.segmentation_metrics import SlidingWindowInference
import kornia
import sys


#################################################################################################
class Segmentation_Trainer:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator=None,
    ) -> None:
        """classification trainer class init function

        Args:
            config (Dict): _description_
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            criterion (torch.nn.Module): _description_
            train_dataloader (DataLoader): _description_
            val_dataloader (DataLoader): _description_
            warmup_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            training_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            accelerator (_type_, optional): _description_. Defaults to None.
        """
        # config
        self.config = config
        self._configure_trainer()

        # model components
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # accelerate object
        self.accelerator = accelerator

        # get wandb object
        self.wandb_tracker = accelerator.get_tracker("wandb")

        # metrics
        self.current_epoch = 0  # current epoch
        self.epoch_train_loss = 0.0  # epoch train loss
        self.best_train_loss = 100.0  # best train loss
        self.epoch_val_loss = 0.0  # epoch validation loss
        self.best_val_loss = 100.0  # best validation loss
        self.epoch_val_dice = 0.0  # epoch validation accuracy
        self.best_val_dice = 0.0  # best validation accuracy
        self.best_val_uls_metric = 0.0  # best validation accuracy

        # external metric functions we can add
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
        )

        # training scheduler
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        # temp ema model copy
        self.val_ema_model = None
        self.ema_model = self._create_ema_model() if self.ema_enabled else None
        self.epoch_val_ema_dice = 0.0
        self.best_val_ema_dice = 0.0
        self.accelerator.print("[info] -- Trainer initialized!")
        
        # model saving parameters

    def _configure_trainer(self) -> None:
        """
        Configures useful config variables
        """
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.val_ema_every = self.config["ema"]["val_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"][
            "checkpoint_save_dir"
        ]
        self.checkpoint_save_frequency = self.config["training_parameters"]["checkpoint_save_frequency"]

    def _load_checkpoint(self):
        raise NotImplementedError

    def _create_ema_model(self) -> torch.nn.Module:
        self.accelerator.print(f"[info] -- creating ema model")
        ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            device=self.accelerator.device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config["ema"]["ema_decay"]
            ),
        )
        return ema_model

    def _train_step(self) -> float:
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        # set epoch to shift data order each epoch
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        
        for index, raw_data in enumerate(self.train_dataloader):
            
            # add in gradient accumulation
            # TODO: test gradient accumulation
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # print("data ", data.shape, "label ", labels.shape)

                # zero out existing gradients
                self.optimizer.zero_grad()

                # forward pass
                predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, labels)

                # backward pass
                self.accelerator.backward(loss)

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model)

                # update loss
                epoch_avg_loss += loss.item()

                if self.print_every:
                    if index % self.print_every == 0:
                        self.accelerator.print(
                            f"epoch: {str(self.current_epoch).zfill(4)} -- "
                            f"train loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )

        epoch_avg_loss = epoch_avg_loss / (index + 1)

        return epoch_avg_loss
    
    def _train_step_profiled(self) -> float:
        import time
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        # set epoch to shift data order each epoch
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        batch_start_time = time.time()
        first = True
        for index, raw_data in enumerate(tqdm(self.train_dataloader)):

            batch_time = time.time() - batch_start_time

            # add in gradient accumulation
            # TODO: test gradient accumulation
            with self.accelerator.accumulate(self.model):
                print(
                    f"[info] -- gpu id: {self.accelerator.device} -- "
                    f"batch {index} time: {batch_time:.4f}s"
                )
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # print("data ", data.shape, "label ", labels.shape)

                # zero out existing gradients
                self.optimizer.zero_grad()

                # forward pass
                forward_start = time.time()
                predicted = self.model.forward(data)
                forward_time = time.time() - forward_start

                print(
                    f"[info] -- gpu id: {self.accelerator.device} -- "
                    f"batch {index} forward time: {forward_time:.4f}s"
                )

                # calculate loss
                loss_start = time.time()
                print(f"predicted shape: {predicted.shape}, labels shape: {labels.shape}")
                loss = self.criterion(predicted, labels)
                loss_time = time.time() - loss_start

                # backward pass
                backward_start = time.time()
                self.accelerator.backward(loss)
                backward_time = time.time() - backward_start

                print(
                    f"[info] -- gpu id: {self.accelerator.device} -- "
                    f"batch {index} times: "
                    f"data: {batch_time:.4f}s | "
                    f"forward: {forward_time:.4f}s | "
                    f"loss: {loss_time:.4f}s | "
                    f"backward: {backward_time:.4f}s"
                )

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model)

                # update loss
                epoch_avg_loss += loss.item()

                if self.print_every:
                    if index % self.print_every == 0:
                        self.accelerator.print(
                            f"epoch: {str(self.current_epoch).zfill(4)} -- "
                            f"train loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )
                print(
                    f"[info] -- batch {index} times: "
                    f"get_batch: {batch_time:.4f}s | "
                    f"forward: {forward_time:.4f}s | "
                    f"loss: {loss_time:.4f}s | "
                    f"backward: {backward_time:.4f}s"
                )
                sys.stdout.flush()
            batch_start_time = time.time()  # reset batch start time

        epoch_avg_loss = epoch_avg_loss / (index + 1)

        return epoch_avg_loss

    def _val_step(self, use_ema: bool = False) -> float:
        """_summary_

        Args:
            use_ema (bool, optional): if use_ema runs validation with ema_model. Defaults to False.

        Returns:
            float: _description_
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_dice = 0.0
        total_uls_metric = 0.0

        # set model to train mode
        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        # set epoch to shift data order each epoch
        # self.val_dataloader.sampler.set_epoch(self.current_epoch)
        with torch.no_grad():
            for index, (raw_data) in enumerate(self.val_dataloader):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # forward pass
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, labels)

                # calculate metrics
                if self.calculate_metrics:
                    mean_dice, mean_uls_metric = self._calc_dice_metric(data, labels, use_ema)
                    # keep track of number of total correct
                    total_dice += mean_dice
                    total_uls_metric += mean_uls_metric

                # update loss for the current batch
                epoch_avg_loss += loss.item()

        if use_ema:
            self.epoch_val_ema_dice = total_dice / float(index + 1)
        else:
            self.epoch_val_dice = total_dice / float(index + 1)
            self.epoch_val_uls_metric = total_uls_metric / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)

        return epoch_avg_loss
    
    def _val_step_profiled(self, use_ema: bool = False) -> float:
        import time
        epoch_avg_loss = 0.0
        total_dice = 0.0
        total_uls_metric = 0.0

        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        with torch.no_grad():
            batch_start_time = time.time()
            first = True
            for index, raw_data in enumerate(tqdm(self.val_dataloader)):
                if not first:
                    batch_start_time = time.time()
                else:
                    first = False

                batch_time = time.time() - batch_start_time
                print(
                    f"[info] -- gpu id: {self.accelerator.device} -- "
                    f"val batch {index} time: {batch_time:.4f}s"
                )

                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )

                forward_start = time.time()
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)
                forward_time = time.time() - forward_start

                loss_start = time.time()
                loss = self.criterion(predicted, labels)
                loss_time = time.time() - loss_start

                if self.calculate_metrics:
                    metric_start = time.time()
                    mean_dice, mean_uls_metric = self._calc_dice_metric(data, labels, use_ema)
                    metric_time = time.time() - metric_start
                    total_dice += mean_dice
                    total_uls_metric += mean_uls_metric
                else:
                    metric_time = 0.0

                epoch_avg_loss += loss.item()

                if self.print_every:
                    if index % self.print_every == 0:
                        self.accelerator.print(
                            f"epoch: {str(self.current_epoch).zfill(4)} -- "
                            f"val loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )
                print(
                    f"[info] -- val batch {index} times: "
                    f"get_batch: {batch_time:.4f}s | "
                    f"forward: {forward_time:.4f}s | "
                    f"loss: {loss_time:.4f}s | "
                    f"metrics: {metric_time:.4f}s"
                )
                sys.stdout.flush()

        if use_ema:
            self.epoch_val_ema_dice = total_dice / float(index + 1)
        else:
            self.epoch_val_dice = total_dice / float(index + 1)
            self.epoch_val_uls_metric = total_uls_metric / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)

        return epoch_avg_loss

    def _calc_dice_metric(self, data, labels, use_ema: bool) -> float:
        """_summary_

        Args:
            predicted (_type_): _description_
            labels (_type_): _description_

        Returns:
            float: _description_
        """
        if use_ema:
            avg_dice_score = self.sliding_window_inference(
                data,
                labels,
                self.ema_model,
            )
        else:
            avg_dice_score, uls_metric = self.sliding_window_inference(
                data,
                labels,
                self.model,
            )
        return avg_dice_score, uls_metric

    def _run_train_val(self) -> None:
        """_summary_"""
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )
        print("[info] -- Starting training and validation")
        # Run Training and Validation
        for epoch in tqdm(range(self.current_epoch, self.num_epochs)):
            # update epoch
            self.current_epoch = epoch
            self._update_scheduler()

            # run a single training step
            train_loss = self._train_step()
            self.epoch_train_loss = train_loss

            # run a single validation step
            val_loss = self._val_step(use_ema=False)
            self.epoch_val_loss = val_loss


            if self.current_epoch % self.checkpoint_save_frequency == 0 and self.current_epoch != 0:  
                # if enabled run ema every x steps
                self._val_ema_model()

                # update metrics
                self._update_metrics()

                # log metrics
                self._log_metrics()

                # save and print
                self._save_and_print()

            # update schduler
            self.scheduler.step()

    def _run_train_val_profiled(self) -> None:
        """_summary_"""

        import time
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )
        print("[info] -- Starting training and validation with profiling")
        # Run Training and Validation
        for epoch in tqdm(range(self.current_epoch, self.num_epochs)):
            # update epoch
            self.current_epoch = epoch
            self._update_scheduler()

            # run a single training step
            train_step_time = time.time()
            train_loss = self._train_step_profiled()
            self.epoch_train_loss = train_loss
            train_step_time = time.time() - train_step_time

            validate_step_time = time.time()
            # run a single validation step
            val_loss = self._val_step_profiled(use_ema=False)
            self.epoch_val_loss = val_loss
            validate_step_time = time.time() - validate_step_time

            if self.current_epoch % self.checkpoint_save_frequency == 0 and self.current_epoch != 0:  
                # if enabled run ema every x steps
                self._val_ema_model()

                # update metrics
                self._update_metrics()

                # log metrics
                self._log_metrics()

                # save and print
                self._save_and_print()

            # update schduler
            self.scheduler.step()

    def _run_eval(self, use_ema=False) -> None:
        """_summary_"""
        # Tell wandb to watch the model and optimizer values

        self.accelerator.print("[info] -- Starting model evaluation")

        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_dice = 0.0
        total_uls_metric = 0.0

        my_total_uls_metric = 0.0

        # set model to train mode
        self.model.eval()
        print("YESS")

        # set epoch to shift data order each epoch
        # self.val_dataloader.sampler.set_epoch(self.current_epoch)
        with torch.no_grad():
            for index, (raw_data) in enumerate(tqdm(self.val_dataloader)):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],  
                )
                # forward pass
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                torch.save(data, f'/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/model_predictions_best_diceCE_checkpoint/input_tensor_idx{index}.pt')
                torch.save(predicted, f'/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/model_predictions_best_diceCE_checkpoint/pred_tensor_idx{index}.pt')
                torch.save(labels, f'/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/model_predictions_best_diceCE_checkpoint/label_tensor_idx{index}.pt')

                # # calculate loss
                # loss = self.criterion(predicted, labels)

                # y_pred = test_time_augmentation(self.model, data, morph_op='opening')
                # y_true = labels[:, 1:, ...]

                # my_uls_metric = self.sliding_window_inference.evaluator.ULS_score_metric(y_pred, y_true)
                # # print(f"ULS: {my_uls_metric}")
                


                # # calculate metrics
                # if self.calculate_metrics:
                #     mean_dice, mean_uls_metric = self._calc_dice_metric(data, labels, use_ema)
                #     # keep track of number of total correct
                #     total_dice += mean_dice
                #     total_uls_metric += mean_uls_metric

                #     my_total_uls_metric += my_uls_metric

                # update loss for the current batch
                # epoch_avg_loss += loss.item()

        if use_ema:
            self.epoch_val_ema_dice = total_dice / float(index + 1)
        else:
            self.epoch_val_dice = total_dice / float(index + 1)
            self.epoch_val_uls_metric = total_uls_metric / float(index + 1)
            self.my_epoch_val_uls_metric = my_total_uls_metric / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)


        self.epoch_val_loss = epoch_avg_loss

        self._update_metrics()

        self.accelerator.print(
            f"eval loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
            f"eval mean_uls_metric -- {colored(f'{self.epoch_val_uls_metric:.5f}', color='green')} -- saved"
            f"eval my_mean_uls_metric -- {colored(f'{self.my_epoch_val_uls_metric:.5f}', color='green')} -- saved"
            f"eval mean_dice -- {colored(f'{self.best_val_dice:.5f}', color='green')} -- saved"
        )



    def _update_scheduler(self) -> None:
        """_summary_"""
        if self.warmup_enabled:
            if self.current_epoch == 0:
                self.accelerator.print(
                    colored(f"\n[info] -- warming up learning rate \n", color="red")
                )
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.accelerator.print(
                    colored(
                        f"\n[info] -- switching to learning rate decay schedule \n",
                        color="red",
                    )
                )
                self.scheduler = self.training_scheduler
        elif self.current_epoch == 0:
            self.accelerator.print(
                colored(
                    f"\n[info] -- setting learning rate decay schedule \n",
                    color="red",
                )
            )
            self.scheduler = self.training_scheduler

    def _update_metrics(self) -> None:
        """_summary_"""
        # update training loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # update validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.calculate_metrics:
            if self.epoch_val_dice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice
            if self.epoch_val_uls_metric >= self.best_val_uls_metric:
                self.best_val_uls_metric = self.epoch_val_uls_metric

    def _log_metrics(self) -> None:
        """_summary_"""
        # data to be logged
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
            "mean_uls_metric": self.epoch_val_uls_metric,
        }
        # log the data
        self.accelerator.log(log_data)

    def _save_and_print(self) -> None:
        """_summary_"""
        # print only on the first gpu
        if self.epoch_val_uls_metric >= self.best_val_uls_metric:
            self.accelerator.print(
                f"[info] -- best model updated at epoch {self.current_epoch}")
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                save_path = self.checkpoint_save_dir
            else:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_dice_model_post_cutoff",
                )

            # save checkpoint and log
            self._save_checkpoint(save_path)

            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"val mean_uls_metric -- {colored(f'{self.best_val_uls_metric:.5f}', color='green')} -- saved"
                f"val mean_dice -- {colored(f'{self.best_val_dice:.5f}', color='green')} -- saved"
            )
        else:
            self.accelerator.print(f"[info] -- best model not updated at epoch {self.current_epoch}")

            self.accelerator.print(
                f"epoch -- {str(self.current_epoch).zfill(4)} || "
                f"train loss -- {self.epoch_train_loss:.5f} || "
                f"val loss -- {self.epoch_val_loss:.5f} || "
                f"lr -- {self.scheduler.get_last_lr()[0]:.8f} || "
                f"val mean_dice -- {self.epoch_val_dice:.5f}"
            )

        # Save the last epoch model for further training
        print(f"[info] -- update last model this epoch according to frequency: {self.current_epoch%self.checkpoint_save_frequency == 0}")
        self.accelerator.print("[info] -- saving last epoch model")
        save_path = os.path.join(
            self.checkpoint_save_dir,
            "last_epoch_model",
        )
        self._save_checkpoint(save_path)

    def _save_checkpoint(self, filename: str) -> None:
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
        self.accelerator.save_state(filename, safe_serialization=False)
        trainer_state_dict = {
            "current_epoch": self.current_epoch,
            "best_val_ema_dice": self.best_val_ema_dice,
            "best_val_loss": self.best_val_loss,
            "best_val_dice": self.best_val_dice,
            "best_val_uls_metric": self.best_val_uls_metric,
            "best_train_loss": self.best_train_loss,
        }

        # save the model state dict
        trainer_state_dict_path = os.path.join(filename, "trainer_state_dict.pkl")
        torch.save(trainer_state_dict, trainer_state_dict_path)




    def _val_ema_model(self):
        if self.ema_enabled and (self.current_epoch % self.val_ema_every == 0):
            self.val_ema_model = self._update_ema_bn(duplicate_model=False)
            _ = self._val_step(use_ema=True)
            self.accelerator.print(
                f"[info] -- gpu id: {self.accelerator.device} -- "
                f"ema val dice: {colored(f'{self.epoch_val_ema_dice:.5f}', color='red')}"
            )

        if self.epoch_val_ema_dice > self.best_val_ema_dice:
            torch.save(self.val_ema_model.module, "best_ema_model_ckpt.pth")
            self.best_val_ema_dice = self.epoch_val_ema_dice

    def _update_ema_bn(self, duplicate_model: bool = True):
        """
        updates the batch norm stats for the ema model
        if duplicate_model is true, a copy of the model is made and
        the batch norm stats are updated for the copy. This is used
        for intermediate ema model saving and validation purpose
        if duplicate model is false, then the original ema model is used
        for the batch norm updates and will be saved as the final
        ema model.
        Args:
            duplicate_model (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # update batch norm stats for ema model after training
        # TODO: test ema functionality
        self.accelerator.print(
            colored("[info] -- updating ema batch norm stats", color="red")
        )
        if duplicate_model:
            temp_ema_model = deepcopy(self.ema_model).to(
                self.accelerator.device
            )  # make temp copy
            torch.optim.swa_utils.update_bn(
                self.train_dataloader,
                temp_ema_model,
                device=self.accelerator.device,
            )
            return temp_ema_model
        else:
            torch.optim.swa_utils.update_bn(
                self.train_dataloader,
                self.ema_model,
                device=self.accelerator.device,
            )
            return None

    def train(self) -> None:
        """
        Runs a full training and validation of the dataset.
        """
        self.accelerator.print(colored("[info] -- starting training", color="red"))
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        self.accelerator.print(
            colored("[info] -- starting evaluation", color="red")
        )
        self._run_eval()
        self.accelerator.end_training()

    def profile(self) -> None:
        """
        Runs a full training and validation of the dataset with profiling.
        """
        self.accelerator.print(
            colored("[info] -- starting training with profiling", color="red")
        )
        self._run_train_val_profiled()
        self.accelerator.end_training()


#################################################################################################
class AutoEncoder_Trainer:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator=None,
    ) -> None:
        """classification trainer class init function

        Args:
            config (Dict): _description_
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            criterion (torch.nn.Module): _description_
            train_dataloader (DataLoader): _description_
            val_dataloader (DataLoader): _description_
            warmup_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            training_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            accelerator (_type_, optional): _description_. Defaults to None.
        """
        # config
        self.config = config
        self._configure_trainer()

        # model components
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # accelerate object
        self.accelerator = accelerator

        # get wandb object
        self.wandb_tracker = accelerator.get_tracker("wandb")

        # metrics
        self.current_epoch = 0  # current epoch
        self.epoch_train_loss = 0.0  # epoch train loss
        self.best_train_loss = 100.0  # best train loss
        self.epoch_val_loss = 0.0  # epoch validation loss
        self.best_val_loss = 100.0  # best validation loss
        self.epoch_val_iou = 0.0  # epoch validation accuracy
        self.best_val_iou = 0.0  # best validation accuracy
        self.ema_val_acc = 0.0  # best ema validation accuracy

        # external metric functions we can add
        # self.metric = evaluate.load("mean_iou")
        # self.metric = compute_iou()

        # training scheduler
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        # temp ema model copy
        self.val_ema_model = None

    def _configure_trainer(self) -> None:
        """
        Configures useful config variables
        """
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.print_ema_every = self.config["ema"]["print_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"][
            "checkpoint_save_dir"
        ]

    def _load_checkpoint(self):
        raise NotImplementedError

    def _create_ema_model(self, gpu_id: int) -> torch.nn.Module:
        self.accelerator.print(f"[info] -- creating ema model")
        ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            device=gpu_id,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config["ema"]["ema_decay"]
            ),
        )
        return ema_model

    def _train_step(self) -> float:
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        # set epoch to shift data order each epoch
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        for index, raw_data in enumerate(self.train_dataloader):
            # add in gradient accumulation
            # TODO: test gradient accumulation
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, _)
                data, _ = (
                    raw_data["image"],
                    raw_data["label"],
                )
                data = data[:, 0, :, :, :].unsqueeze(1)

                # zero out existing gradients
                self.optimizer.zero_grad()

                # forward pass
                predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, data)

                # backward pass
                self.accelerator.backward(loss)

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model.module)

                # update loss
                epoch_avg_loss += loss.item()

                if self.print_every:
                    if index % self.print_every == 0:
                        self.accelerator.print(
                            f"epoch: {str(self.current_epoch).zfill(4)} -- "
                            f"train loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )

        epoch_avg_loss = epoch_avg_loss / (index + 1)

        return epoch_avg_loss

    def _val_step(self, use_ema: bool = False) -> float:
        """_summary_

        Args:
            use_ema (bool, optional): if use_ema runs validation with ema_model. Defaults to False.

        Returns:
            float: _description_
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_iou = 0.0

        # set model to train mode
        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        # set epoch to shift data order each epoch
        # self.val_dataloader.sampler.set_epoch(self.current_epoch)
        with torch.no_grad():
            for index, (raw_data) in enumerate(self.val_dataloader):
                # get data ex: (data, _)
                data, _ = (
                    raw_data["image"],
                    raw_data["label"],
                )
                data = data[:, 0, :, :, :].unsqueeze(1)

                # forward pass
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)

                # calculate loss
                loss = self.criterion(predicted, data)

                if self.calculate_metrics:
                    mean_iou = self._calc_mean_ssim(predicted, data)
                    # keep track of number of total correct
                    total_iou += mean_iou

                # update loss for the current batch
                epoch_avg_loss += loss.item()

        if use_ema:
            self.epoch_val_iou = total_iou / float(index + 1)
        else:
            self.epoch_val_iou = total_iou / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)

        return epoch_avg_loss

    def _calc_mean_ssim(self, predicted, ground_truth) -> float:
        predictions, ground_truth = self.accelerator.gather_for_metrics(
            (predicted, ground_truth)
        )
        ssim_map = kornia.metrics.ssim3d(predictions, ground_truth, window_size=5)
        ssim_map = ssim_map.mean()

        return ssim_map.item()

    def _run_train_val(self) -> None:
        """_summary_"""
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )

        # Run Training and Validation
        for epoch in tqdm(range(self.num_epochs)):
            # update epoch
            self.current_epoch = epoch
            if self.warmup_enabled or self.current_epoch == 0:
                self._update_scheduler()

            # run a single training step
            train_loss = self._train_step()
            self.epoch_train_loss = train_loss

            # run a single validation step
            val_loss = self._val_step(use_ema=False)
            self.epoch_val_loss = val_loss

            # update metrics
            self._update_metrics()

            # log metrics
            self._log_metrics()

            # save and print
            self._save_and_print()

            # update schduler
            self.scheduler.step()

    def _update_scheduler(self) -> None:
        """_summary_"""
        if self.warmup_enabled:
            if self.current_epoch == 0:
                self.accelerator.print(
                    colored(f"\n[info] -- warming up learning rate \n", color="red")
                )
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.accelerator.print(
                    colored(
                        f"\n[info] -- switching to learning rate decay schedule \n",
                        color="red",
                    )
                )
                self.scheduler = self.training_scheduler
        else:
            self.accelerator.print(
                colored(
                    f"\n[info] -- setting learning rate decay schedule \n",
                    color="red",
                )
            )
            self.scheduler = self.training_scheduler

    def _update_metrics(self) -> None:
        """_summary_"""
        # update training loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # update validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.calculate_metrics:
            if self.epoch_val_iou >= self.best_val_iou:
                self.best_val_iou = self.epoch_val_iou

    def _log_metrics(self) -> None:
        """_summary_"""
        # data to be logged
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_iou": self.epoch_val_iou,
        }
        # log the data
        self.accelerator.log(log_data)

    def _save_and_print(self) -> None:
        """_summary_"""
        # print only on the first gpu
        if self.epoch_val_iou >= self.best_val_iou:
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_iou_state",
                )
            else:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_iou_state_post_cutoff.pth",
                )

            # save checkpoint and log
            self._save_checkpoint(save_path)

            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"val mean_ssim -- {colored(f'{self.best_val_iou:.5f}', color='green')} -- saved"
            )
        else:
            self.accelerator.print(
                f"epoch -- {str(self.current_epoch).zfill(4)} || "
                f"train loss -- {self.epoch_train_loss:.5f} || "
                f"val loss -- {self.epoch_val_loss:.5f} || "
                f"lr -- {self.scheduler.get_last_lr()[0]:.8f} || "
                f"val mean_ssim -- {self.epoch_val_iou:.5f}"
            )

    def _save_checkpoint(self, filename: str) -> None:
        """_summary_

        Args:
            filename (str): _description_
        """
        # saves the ema model checkpoint if availabale
        # TODO: ema saving untested
        if self.ema_enabled and self.val_ema_model:
            checkpoint = {
                "state_dict": self.val_ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{os.path.dirname(filename)}/ema_model_ckpt.pth")
            self.val_ema_model = (
                None  # set ema model to None to avoid duplicate model saving
            )

        # standard model checkpoint
        self.accelerator.save_state(filename, safe_serialization=False)

    def _update_ema_bn(self, duplicate_model: bool = True):
        """
        updates the batch norm stats for the ema model
        if duplicate_model is true, a copy of the model is made and
        the batch norm stats are updated for the copy. This is used
        for intermediate ema model saving and validation purpose
        if duplicate model is false, then the original ema model is used
        for the batch norm updates and will be saved as the final
        ema model.
        Args:
            duplicate_model (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # update batch norm stats for ema model after training
        # TODO: test ema functionality
        print(colored(f"[info] -- updating ema batch norm stats", color="red"))
        if duplicate_model:
            temp_ema_model = deepcopy(self.ema_model).to(self.gpu_id)  # make temp copy
            torch.optim.swa_utils.update_bn(
                self.train_dataloader, temp_ema_model, device=self.gpu_id
            )
            return temp_ema_model
        else:
            torch.optim.swa_utils.update_bn(
                self.train_dataloader, self.ema_model, device=self.gpu_id
            )
            return None

    def train(self) -> None:
        """
        Runs a full training and validation of the dataset.
        """
        self.accelerator.print("[info] -- Starting training!!")
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        pass
