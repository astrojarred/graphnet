"""Suggested Model subclass that enables simple user syntax."""

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Type

import numpy as np
import torch
from lightning_fabric.plugins.precision.amp import _optimizer_handles_unscaling
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import GradClipAlgorithmType
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data
import pandas as pd
from pytorch_lightning.loggers import Logger as LightningLogger

from graphnet.training.callbacks import ProgressBar
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask


def assert_exact_event_ids(
    df: pd.DataFrame, column: str = "event_id"
) -> None:
    """Raise if `column` in `df` holds floating-point event identifiers.

    Integer event identifiers that have passed through a floating-point
    representation are exact only up to 2**53 (float64) or 2**24 (float32).
    Prediction artifacts produced by older versions of
    `predict_as_dataframe`/`get_predictions` concatenated identifiers with
    float predictions and therefore stored corrupted identifiers. Use this
    guard to detect such legacy artifacts before relying on the identifiers.

    Args:
        df: DataFrame to check (e.g. loaded prediction artifact).
        column: Name of the event-identifier column. Defaults to
            "event_id".

    Raises:
        KeyError: If `column` is not present in `df`.
        TypeError: If the column has a floating-point dtype, meaning the
            identifiers cannot be trusted to be exact.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame "
            f"(columns: {list(df.columns)})."
        )
    if pd.api.types.is_float_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' has floating-point dtype "
            f"'{df[column].dtype}'. Integer event identifiers stored as "
            "floats are exact only up to 2**53 (float64) / 2**24 (float32) "
            "and may be corrupted. This is characteristic of legacy "
            "prediction artifacts that concatenated identifiers with float "
            "predictions."
        )


class EasySyntax(Model):
    """A suggested Model class that comes with simple user syntax.

    This class delivers simple user syntax for training and prediction,
    while imposing minimal constraints on structure.
    """

    def __init__(
        self,
        *,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

        self.validate_tasks()

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        raise NotImplementedError

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        raise NotImplementedError

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        raise NotImplementedError

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        raise NotImplementedError

    @staticmethod
    def _construct_trainer(
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> Trainer:
        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = 1

        trainer_init: Dict[str, Any] = {
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": max_epochs,
            "callbacks": callbacks,
            "log_every_n_steps": log_every_n_steps,
            "logger": logger,
            "gradient_clip_val": gradient_clip_val,
        }
        # PL 2.x rejects ``strategy=None``; omit so Lightning picks a default.
        if distribution_strategy is not None:
            trainer_init["strategy"] = distribution_strategy

        trainer = Trainer(**trainer_init, **trainer_kwargs)

        return trainer

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        *,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `StandardModel` using `pytorch_lightning.Trainer`."""
        # Checks
        if callbacks is None:
            # We create the bare-minimum callbacks for you.
            callbacks = self._create_default_callbacks(
                val_dataloader=val_dataloader,
                early_stopping_patience=early_stopping_patience,
            )
            self.debug("No Callbacks specified. Default callbacks added.")
        else:
            # You are on your own!
            self.debug("Initializing training with user-provided callbacks.")
            pass
        self._print_callbacks(callbacks)
        has_early_stopping = self._contains_callback(callbacks, EarlyStopping)
        has_model_checkpoint = self._contains_callback(
            callbacks, ModelCheckpoint
        )

        if (has_early_stopping) & (has_model_checkpoint is False):
            self.warning(
                "No ModelCheckpoint found in callbacks. Best-fit model will"
                " not automatically be loaded after training!"
                ""
            )

        self.train(mode=True)
        trainer = self._construct_trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )

        try:
            trainer.fit(
                self, train_dataloader, val_dataloader, ckpt_path=ckpt_path
            )
        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exiting gracefully.")
            pass

        # Load weights from best-fit model after training if possible
        if has_early_stopping & has_model_checkpoint:
            checkpoint_callback: Optional[ModelCheckpoint] = None
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
                    break
            best_path = (
                checkpoint_callback.best_model_path
                if checkpoint_callback is not None
                else ""
            )
            if best_path and os.path.isfile(best_path):
                self.load_state_dict(
                    torch.load(best_path, weights_only=False)["state_dict"]
                )
                self.info("Best-fit weights from EarlyStopping loaded.")
            else:
                self.warning_once(
                    "No checkpoint file to reload (e.g. ``fast_dev_run`` "
                    "suppresses checkpointing). Using in-memory weights."
                )

    def _print_callbacks(self, callbacks: List[Callback]) -> None:
        callback_names = []
        for cbck in callbacks:
            callback_names.append(cbck.__class__.__name__)
        self.info(
            f"Training initiated with callbacks: {', '.join(callback_names)}"
        )

    def _contains_callback(
        self, callbacks: List[Callback], callback: Callback
    ) -> bool:
        """Check if `callback` is in `callbacks`."""
        for cbck in callbacks:
            if isinstance(cbck, callback):
                return True
        return False

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Clip optimizer gradients (same timing/params as Lightning; see Notes).

        PyTorch's fused ``clip_grad_norm_`` path (``foreach=True`` default on CUDA)
        can stall under mixed precision + DDP when parameters mix dtypes (e.g.
        bf16 weights with fp32 task scalars). Using ``foreach=False`` avoids that.

        Fused optimizers that manage AMP unscaling themselves still use the
        default Lightning path (which may raise instead of clipping).
        """
        if gradient_clip_val is None:
            gradient_clip_val = self.trainer.gradient_clip_val
        if gradient_clip_val is None or float(gradient_clip_val) <= 0:
            return

        if gradient_clip_algorithm is None:
            algorithm = self.trainer.gradient_clip_algorithm
        elif isinstance(gradient_clip_algorithm, GradClipAlgorithmType):
            algorithm = gradient_clip_algorithm
        else:
            algorithm = GradClipAlgorithmType(str(gradient_clip_algorithm).lower())

        if algorithm == GradClipAlgorithmType.NORM and _optimizer_handles_unscaling(
            optimizer
        ):
            super().configure_gradient_clipping(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
            return

        parameters = list(self.trainer.precision_plugin.main_params(optimizer))
        if algorithm == GradClipAlgorithmType.VALUE:
            torch.nn.utils.clip_grad_value_(
                parameters, clip_value=float(gradient_clip_val)
            )
            return
        if algorithm == GradClipAlgorithmType.NORM:
            try:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    float(gradient_clip_val),
                    foreach=False,
                )
            except TypeError:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(gradient_clip_val)
                )
            return

        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform training step."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        loss = self.shared_step(train_batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        loss = self.shared_step(val_batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        self.train(mode=False)

        callbacks = self._create_default_callbacks(
            val_dataloader=None,
        )

        inference_trainer = self._construct_trainer(
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            callbacks=callbacks,
            **trainer_kwargs,
        )

        predictions_list = inference_trainer.predict(self, dataloader)
        assert len(predictions_list), "Got no predictions"

        nb_outputs = len(predictions_list[0])
        predictions: List[Tensor] = [
            torch.cat([preds[ix] for preds in predictions_list], dim=0)
            for ix in range(nb_outputs)
        ]
        return predictions

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels

        if additional_attributes is None:
            additional_attributes = []
        assert isinstance(additional_attributes, list)

        if (
            not isinstance(dataloader.sampler, SequentialSampler)
            and additional_attributes
        ):
            print(dataloader.sampler)
            raise UserWarning(
                "DataLoader has a `sampler` that is not `SequentialSampler`, "
                "indicating that shuffling is enabled. Using "
                "`predict_as_dataframe` with `additional_attributes` assumes "
                "that the sequence of batches in `dataloader` are "
                "deterministic. Either call this method a `dataloader` which "
                "doesn't resample batches; or do not request "
                "`additional_attributes`."
            )
        self.info(f"Column names for predictions are: \n {prediction_columns}")
        predictions_torch = self.predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )
        predictions = (
            torch.cat(predictions_torch, dim=1).detach().cpu().numpy()
        )
        assert len(prediction_columns) == predictions.shape[1], (
            f"Number of provided column names ({len(prediction_columns)}) and "
            f"number of output columns ({predictions.shape[1]}) don't match."
        )

        # Check if predictions are on event- or pulse-level
        pulse_level_predictions = len(predictions) > len(dataloader.dataset)

        # Get additional attributes
        attributes: Dict[str, List[np.ndarray]] = OrderedDict(
            [(attr, []) for attr in additional_attributes]
        )
        for batch in dataloader:
            for attr in attributes:
                attribute = batch[attr]
                if isinstance(attribute, torch.Tensor):
                    attribute = attribute.detach().cpu().numpy()

                # Check if node level predictions
                # If true, additional attributes are repeated
                # to make dimensions fit
                if pulse_level_predictions:
                    if len(attribute) < np.sum(
                        batch.n_pulses.detach().cpu().numpy()
                    ):
                        attribute = np.repeat(
                            attribute, batch.n_pulses.detach().cpu().numpy()
                        )
                attributes[attr].extend(attribute)

        # Confirm that attributes match length of predictions
        skip_attributes = []
        for attr in attributes.keys():
            try:
                assert len(attributes[attr]) == len(predictions)
            except AssertionError:
                self.warning_once(
                    "Could not automatically adjust length"
                    f" of additional attribute '{attr}' to match length of"
                    f" predictions.This error can be caused by heavy"
                    " disagreement between number of examples in the"
                    " dataset vs. actual events in the dataloader, e.g. "
                    " heavy filtering of events in `collate_fn` passed to"
                    " `dataloader`. This can also be caused by requesting"
                    " pulse-level attributes for `Task`s that produce"
                    " event-level predictions. Attribute skipped."
                )
                skip_attributes.append(attr)

        # Remove bad attributes
        for attr in skip_attributes:
            attributes.pop(attr)
            additional_attributes.remove(attr)

        # Build prediction columns from the float matrix, then attach each
        # additional attribute as its own pandas Series with native dtype.
        # NOTE: Identifiers (e.g. `event_id`/`event_no`) must never be
        # concatenated with the float32 prediction matrix: int64 values are
        # only exactly representable up to 2**24 in float32 (2**53 in
        # float64), so concatenation silently corrupts large identifiers.
        results = pd.DataFrame(predictions, columns=prediction_columns)
        for attr, values in attributes.items():
            results[attr] = pd.Series(np.asarray(values), index=results.index)
        return results

    def _create_default_callbacks(
        self,
        val_dataloader: DataLoader,
        early_stopping_patience: Optional[int] = None,
    ) -> List:
        """Create default callbacks.

        Used in cases where no callbacks are specified by the user in
        .fit
        """
        callbacks = [ProgressBar()]
        if val_dataloader is not None:
            assert early_stopping_patience is not None
            # Add Early Stopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                )
            )
            # Add Model Check Point
            callbacks.append(
                ModelCheckpoint(
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    filename=f"{self.backbone.__class__.__name__}"
                    + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                )
            )
            self.info(
                "EarlyStopping has been added"
                f" with a patience of {early_stopping_patience}."
            )
        return callbacks

    def _add_early_stopping(
        self, val_dataloader: DataLoader, callbacks: List
    ) -> List:
        if val_dataloader is None:
            return callbacks
        has_early_stopping = False
        assert isinstance(callbacks, list)
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                has_early_stopping = True

        if not has_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                )
            )
            self.warning_once(
                "Got validation dataloader but no EarlyStopping callback. An "
                "EarlyStopping callback has been added automatically with "
                "patience=5 and monitor = 'val_loss'."
            )
        return callbacks
