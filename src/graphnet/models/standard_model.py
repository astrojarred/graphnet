"""Standard model class(es)."""

from typing import Dict, List, Optional, Union, Type
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch.optim import Adam

from graphnet.models.gnn.gnn import GNN
from graphnet.models import Model
from .easy_model import EasySyntax
from graphnet.models.task import StandardLearnedTask
from graphnet.models.data_representation import (
    GraphDefinition,
    DataRepresentation,
)


def _task_loss_slug(task: StandardLearnedTask) -> str:
    """Short snake_case name for logging (e.g. ``SourcePositionTask`` → ``source_position``)."""
    name = task.__class__.__name__
    if name.endswith("Task"):
        name = name[:-4]
    parts: List[str] = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0 and name[i - 1].islower():
            parts.append("_")
        parts.append(c.lower())
    return "".join(parts)


class StandardModel(EasySyntax):
    """A Standard way of combining model components in GraphNeT.

    This model is compatible with the vast majority of supervised
    learning tasks such as regression, binary and multi-label
    classification.

    Capable of producing both event-level and pulse-level predictions.
    """

    def __init__(
        self,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        data_representation: Optional[DataRepresentation] = None,
        graph_definition: Optional[GraphDefinition] = None,
        backbone: Optional[Model] = None,
        gnn: Optional[GNN] = None,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        uncertainty_weighting: bool = False,
    ) -> None:
        """Construct `StandardModel`.

        Args:
            uncertainty_weighting: If True, combine per-task losses with
                homoscedastic uncertainty (Kendall et al., 2018): for task
                scalars ``L_i`` and learnable ``log_var_i = log(σ_i²)``,
                ``Σ_i exp(-log_var_i) * L_i + log_var_i``. At init (zeros)
                this equals ``Σ_i L_i``. Differs from the paper by constant
                factors commonly absorbed into the learning rate.
        """
        # Base class constructor
        super().__init__(
            tasks=tasks,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )
        # DEPRECATION ARG GRAPH_DEFINITION: REMOVE AT 2.0 LAUNCH
        # See https://github.com/graphnet-team/graphnet/issues/647

        if (data_representation is None) & (graph_definition is not None):
            data_representation = graph_definition
            # Code continues after warning
            self.warning(
                "DeprecationWarning: Argument `graph_definition` will be"
                " deprecated in GraphNeT 2.0. Please use `data_representation`"
                " instead."
                ""
            )
        elif (data_representation is None) & (graph_definition is None):
            # Code stops
            raise TypeError(
                "__init__() missing 1 required keyword argument:"
                "'data_representation'"
            )

        # deprecation warnings
        if (backbone is None) & (gnn is not None):
            backbone = gnn
            # Code continues after warning
            self.warning(
                "DeprecationWarning: Argument `gnn` will be deprecated in"
                " GraphNeT 2.0. Please use `backbone` instead."
                ""
            )
        elif (backbone is None) & (gnn is None):
            # Code stops
            raise TypeError(
                "__init__() missing 1 required keyword argument:'backbone'"
            )

        # Checks
        assert isinstance(backbone, Model)
        assert isinstance(data_representation, DataRepresentation)

        # Member variable(s)
        self._data_representation = data_representation
        self.backbone = backbone
        self._uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting:
            self._task_log_vars = nn.Parameter(
                torch.zeros(len(self._tasks), dtype=torch.float32)
            )
        else:
            self.register_parameter("_task_log_vars", None)

    def _combine_task_losses(self, task_losses: List[Tensor]) -> Tensor:
        """Sum task losses, optionally with Kendall et al. uncertainty weighting."""
        if not self._uncertainty_weighting:
            return torch.sum(torch.stack(task_losses))
        assert self._task_log_vars is not None
        total = task_losses[0].new_zeros(())
        for i, loss in enumerate(task_losses):
            log_v = self._task_log_vars[i]
            precision = torch.exp(-log_v)
            total = total + precision * loss + log_v
        return total

    def _merge_batch_targets(self, data: List[Data]) -> Dict[str, Tensor]:
        """Stack per-graph targets (and loss weights) for batched loss computation."""
        data_merged: Dict[str, Tensor] = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )
        return data_merged

    def _per_task_losses(
        self, preds: List[Tensor], data_merged: Dict[str, Tensor], verbose: bool = False
    ) -> List[Tensor]:
        """Scalar loss tensor for each task (same order as ``self._tasks``)."""
        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return losses

    def compute_loss(
        self, preds: List[Tensor], data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = self._merge_batch_targets(data)
        losses = self._per_task_losses(preds, data_merged, verbose=verbose)
        return self._combine_task_losses(losses)

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]
        x_list = []
        for d in data:
            x = self.backbone(d)
            x_list.append(x)
        x = torch.cat(x_list, dim=0)

        preds = [task(x) for task in self._tasks]
        return preds

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        preds = self(batch)
        loss = self.compute_loss(preds, batch)
        return loss

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Like :meth:`EasySyntax.training_step` but log each task loss for loggers (e.g. W&B)."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        batch_size = self._get_batch_size(train_batch)
        preds = self(train_batch)
        data_merged = self._merge_batch_targets(train_batch)
        task_losses = self._per_task_losses(preds, data_merged)
        loss = self._combine_task_losses(task_losses)

        for task, task_loss in zip(self._tasks, task_losses):
            slug = _task_loss_slug(task)
            self.log(
                f"train/loss_{slug}",
                task_loss,
                batch_size=batch_size,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
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
        """Like :meth:`EasySyntax.validation_step` but log each task loss for loggers (e.g. W&B)."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        batch_size = self._get_batch_size(val_batch)
        preds = self(val_batch)
        data_merged = self._merge_batch_targets(val_batch)
        task_losses = self._per_task_losses(preds, data_merged)
        loss = self._combine_task_losses(task_losses)

        for task, task_loss in zip(self._tasks, task_losses):
            slug = _task_loss_slug(task)
            self.log(
                f"val/loss_{slug}",
                task_loss,
                batch_size=batch_size,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        accepted_tasks = StandardLearnedTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    # DEPRECATION ARG GRAPH_DEFINITION: REMOVE AT 2.0 LAUNCH
    # See https://github.com/graphnet-team/graphnet/issues/647
    @property
    def _graph_definition(self) -> DataRepresentation:
        """Return the graph definition."""
        self.warning(
            "DeprecationWarning: `_graph_definition` will be deprecated in"
            " GraphNeT 2.0. Please use `_data_representation` instead."
        )
        return self._data_representation
