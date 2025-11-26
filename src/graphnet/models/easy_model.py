"""Suggested Model subclass that enables simple user syntax."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Type, cast

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data, Batch
import pandas as pd
from pytorch_lightning.loggers import Logger as LightningLogger

from graphnet.training.callbacks import ProgressBar
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask


class EasySyntax(Model):
    """A suggested Model class that comes with simple user syntax.

    This class delivers simple user syntax for training and prediction, while
    imposing minimal constraints on structure.
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

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
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

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            gradient_clip_val=gradient_clip_val,
            strategy=distribution_strategy,
            **trainer_kwargs,
        )

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
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
            self.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"]
            )
            self.info("Best-fit weights from EarlyStopping loaded.")

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
    ) -> List[Tensor]:
        """Return predictions for `dataloader`.

        Falls back to a fault-tolerant path that skips over bad samples if the
        standard prediction loop encounters DataLoader worker errors.
        """
        self.inference()
        self.train(mode=False)

        callbacks = self._create_default_callbacks(
            val_dataloader=None,
        )

        inference_trainer = self._construct_trainer(
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            callbacks=callbacks,
        )

        try:
            predictions_list = inference_trainer.predict(self, dataloader)
            assert len(predictions_list), "Got no predictions"

            nb_outputs = len(predictions_list[0])
            predictions: List[Tensor] = [
                torch.cat([preds[ix] for preds in predictions_list], dim=0)
                for ix in range(nb_outputs)
            ]
            return predictions
        except Exception as e:
            # Fallback: skip bad samples and continue using a resilient iterator
            self.warning_once(
                f"Standard predict failed with {e.__class__.__name__}: {e}. "
                "Falling back to safe prediction that skips bad samples."
            )
            try:
                return self._safe_predict_skip_errors(dataloader=dataloader, gpus=gpus)
            except Exception as e2:
                # As a last resort, iterate dataset indices directly and skip per-item failures
                self.warning_once(
                    f"Safe prediction path also failed with {e2.__class__.__name__}: {e2}. "
                    "Attempting ultra-safe streaming over dataset indices."
                )
                return self._ultra_safe_predict(dataloader=dataloader, gpus=gpus)

    def _ultra_safe_predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
    ) -> List[Tensor]:
        """Predict by iterating dataset indices with per-item try/except and building batches.

        This avoids DataLoader internals entirely after construction-time and tolerates
        sporadic per-index failures.
        """
        # Resolve device from `gpus` argument
        if gpus:
            device_index = gpus[0] if isinstance(gpus, list) else gpus
            device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        self.to(device)
        self.eval()

        dataset = dataloader.dataset
        batch_size = cast(int, dataloader.batch_size) if dataloader.batch_size else 1

        buffered_graphs: List[Data] = []
        per_output_batches: List[List[Tensor]] = []

        def flush_buffer() -> None:
            nonlocal buffered_graphs, per_output_batches
            if not buffered_graphs:
                return
            batch = Batch.from_data_list(buffered_graphs).to(device)
            with torch.no_grad():
                outputs = self(batch)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                if not per_output_batches:
                    per_output_batches = [[] for _ in range(len(outputs))]
                for ix, out in enumerate(outputs):
                    per_output_batches[ix].append(out.detach().to("cpu"))
            buffered_graphs = []

        # Iterate dataset indices and build batches manually
        for idx in range(len(dataset)):
            try:
                g = dataset[idx]
                n_pulses = getattr(g, "n_pulses", None)
                if n_pulses is None or int(n_pulses) <= 1:
                    continue
                buffered_graphs.append(g)
                if len(buffered_graphs) >= batch_size:
                    flush_buffer()
            except Exception:
                # skip bad sample
                continue

        # Flush any remaining graphs
        flush_buffer()

        assert per_output_batches, "Got no predictions in ultra-safe path"
        predictions: List[Tensor] = [torch.cat(chunks, dim=0) for chunks in per_output_batches]
        return predictions

    def _safe_predict_skip_errors(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
    ) -> List[Tensor]:
        """Predict by iterating the underlying dataset and skipping bad samples.

        This path avoids DataLoader worker crashes by fetching items directly
        from the dataset and building batches manually. It respects the original
        batch size and filters events with too few pulses to mimic the default
        collate behavior.
        """
        # no-op

        # Resolve device from `gpus` argument
        if gpus:
            if isinstance(gpus, list):
                device_index = gpus[0]
            else:
                device_index = gpus
            device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Move model to device
        self.to(device)
        self.eval()

        dataset = dataloader.dataset
        assert hasattr(dataloader, "batch_size"), "Dataloader must expose batch_size"
        batch_size = cast(int, dataloader.batch_size) if dataloader.batch_size else 1

        buffered_graphs: List[Data] = []
        # We accumulate outputs per-task across batches
        per_output_batches: List[List[Tensor]] = []

        def flush_buffer() -> None:
            nonlocal buffered_graphs, per_output_batches
            if not buffered_graphs:
                return
            batch = Batch.from_data_list(buffered_graphs).to(device)
            with torch.no_grad():
                outputs = self(batch)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                # Initialize collectors on first use
                if not per_output_batches:
                    per_output_batches = [[] for _ in range(len(outputs))]
                for ix, out in enumerate(outputs):
                    per_output_batches[ix].append(out.detach().to("cpu"))
            buffered_graphs = []

        # Iterate dataset indices and build batches manually
        for idx in range(len(dataset)):
            try:
                g = dataset[idx]
                # Mimic default collate_fn filtering: require >1 pulses
                n_pulses = getattr(g, "n_pulses", None)
                if n_pulses is None or int(n_pulses) <= 1:
                    continue
                buffered_graphs.append(g)
                if len(buffered_graphs) >= batch_size:
                    flush_buffer()
            except Exception as err:  # skip bad sample
                self.warning_once(f"Skipping sample at index {idx} due to {err}")
                continue

        # Flush any remaining graphs
        flush_buffer()

        assert per_output_batches, "Got no predictions in safe path"
        predictions: List[Tensor] = [torch.cat(chunks, dim=0) for chunks in per_output_batches]
        return predictions

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
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
        )
        predictions = (
            torch.cat(predictions_torch, dim=1).detach().cpu().numpy()
        )
        assert len(prediction_columns) == predictions.shape[1], (
            f"Number of provided column names ({len(prediction_columns)}) and "
            f"number of output columns ({predictions.shape[1]}) don't match."
        )

        # Fast-path: if no additional attributes requested, avoid iterating the
        # dataloader again (which could re-trigger data loading errors).
        if not additional_attributes:
            return pd.DataFrame(predictions, columns=prediction_columns)

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

                # special magic attributes
                # phi and theta are lists all with the same length, so we only need to get the first value for each sample of the batch
                if attr in ["signal"]:
                    # calculate n_pulses and start_indices
                    n_pulses = batch['n_pulses'].detach().cpu().numpy()
                    start_indices = np.concatenate([[0], np.cumsum(n_pulses[:-1])])
                    # extract the first value for each sample of the batch

                if attr == "signal":
                    end_indices = np.cumsum(n_pulses)
                    # Sum all signal values for each graph using explicit slicing
                    summed = [attribute[start:end].sum() for start, end in zip(start_indices, end_indices)]
                    attribute = summed
                elif attr == "tel_id":
                    # return the answer to len(tel_id.unique()) == 2 for each sample of the batch
                    attribute = [bool(len(np.unique(attribute[start:end])) == 2) for start, end in zip(start_indices, end_indices)]


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
                self.warning_once(f"Attribute {attr} has length {len(attributes[attr])}")
                self.warning_once(f"Predictions have length {len(predictions)}")
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

        data = np.concatenate(
            [predictions]
            + [
                np.asarray(values)[:, np.newaxis]
                for values in attributes.values()
            ],
            axis=1,
        )

        results = pd.DataFrame(
            data, columns=prediction_columns + additional_attributes
        )
        return results

    def _create_default_callbacks(
        self,
        val_dataloader: DataLoader,
        early_stopping_patience: Optional[int] = None,
    ) -> List:
        """Create default callbacks.

        Used in cases where no callbacks are specified by the user in .fit
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
