from __future__ import annotations

from typing import List, Optional, Tuple, ClassVar, Union, Callable

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from pytorch_lightning.callbacks import Callback

from graphnet.models.graphs.nodes import NodeDefinition
from graphnet.training.loss_functions import LossFunction
from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like

# from graphnet.models.task.task import LearnedTask
from graphnet.training.parquet_weight_fitting import MAGICZenithEnergyWeighter

__all__ = [
    "MAGCIEnergyNodes",
    "MAGICEnergyReconstruction",
]


class MAGCIEnergyNodes(NodeDefinition):
    """MAGIC node definition with pulse sampling to limit memory usage.

    This class samples pulses from MAGIC events to keep the number below
    a maximum threshold, preventing memory issues with large events.
    Can prioritize pulses by signal strength, time, or use random sampling.
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_pulses: int = 512,
        sampling_strategy: str = "signal",  # "signal", "time", "random", "minmax"
        signal_name: str = "signal",
        time_name: str = "t",
    ) -> None:
        """Construct `MAGICLimitedNodes`.

        Args:
            input_feature_names: Column names for input features. Default
                MAGIC features will be used if None.
            max_pulses: Maximum number of pulses to keep per event.
            sampling_strategy: How to sample pulses if over max_pulses:
                - "signal": Keep pulses with highest signal
                - "time": Keep earliest pulses
                - "random": Random sampling
                - "minmax": Keep pulses with highest and lowest signal: 90% highest and 10% lowest
            signal_name: Name of the signal column for signal-based sampling.
            time_name: Name of the time column for time-based sampling.
        """
        if input_feature_names is None:
            input_feature_names = [
                "x_cam",
                "y_cam",
                "tel_id",
                "t",
                "signal",
                "telescope_phi",
                "telescope_theta",
            ]

        super().__init__(input_feature_names=input_feature_names)

        self.max_pulses = max_pulses
        self.sampling_strategy = sampling_strategy
        self.signal_name = signal_name
        self.time_name = time_name

        # Get feature indices for sampling
        try:
            self.signal_idx = input_feature_names.index(signal_name)
        except ValueError:
            if sampling_strategy == "signal":
                raise ValueError(f"Signal column '{signal_name}' not found in features")
            self.signal_idx = None

        try:
            self.time_idx = input_feature_names.index(time_name)
        except ValueError:
            if sampling_strategy == "time":
                raise ValueError(f"Time column '{time_name}' not found in features")
            self.time_idx = None

    def _define_output_feature_names(self, input_feature_names: List[str]) -> List[str]:
        """Define the output feature names (same as input for MAGIC)."""
        return input_feature_names

    def _sample_pulses(self, x: torch.Tensor) -> torch.Tensor:
        """Sample pulses based on the configured strategy."""
        n_pulses = x.shape[0]

        if n_pulses <= self.max_pulses:
            return torch.arange(n_pulses)

        if self.sampling_strategy == "signal":
            # Keep pulses with highest signal
            _, indices = torch.topk(
                x[:, self.signal_idx], self.max_pulses, largest=True
            )
            return indices.sort().values

        elif self.sampling_strategy == "time":
            # Keep earliest pulses
            _, indices = torch.topk(x[:, self.time_idx], self.max_pulses, largest=False)
            return indices.sort().values

        elif self.sampling_strategy == "minmax":
            # Keep pulses with highest and lowest signal: 90% highest and 10% lowest
            # Ensure total always equals max_pulses
            n_high = int(np.floor(0.9 * self.max_pulses))
            n_low = self.max_pulses - n_high  # Ensures exact sum equals max_pulses
            _, indices = torch.topk(x[:, self.signal_idx], n_high, largest=True)
            _, indices_min = torch.topk(x[:, self.signal_idx], n_low, largest=False)
            return torch.cat([indices, indices_min]).sort().values

        else:  # random
            indices = torch.randperm(n_pulses)[: self.max_pulses]
            return indices.sort().values

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
        """Construct nodes with pulse sampling."""
        # Sample pulses
        selected_indices = self._sample_pulses(x)

        # Extract selected pulses
        sampled_x = x[selected_indices]

        return Data(x=sampled_x), self.input_feature_names


class MAGICEnergyReconstruction(StandardLearnedTask):
    """MAGIC energy reconstruction with integrated systematic weight correction.
    
    The task initializes a systematic weighter from parquet training data and
    applies weight corrections during training based on energy.
    
    Target format:
    - true_energy: [batch_size] (energy)
    - energy_pred: [batch_size] (predicted energy)
    """

    default_target_labels: ClassVar[List[str]] = ["true_energy"]
    default_prediction_labels: ClassVar[List[str]] = ["energy_pred"]
    nb_inputs: ClassVar[int] = 1

    def __init__(
        self, 
        hidden_size: int,
        target_labels: Union[str, List[str]] = ["true_energy"],
        prediction_labels: Optional[List[str]] = None,
        loss_function: Optional[LossFunction] = None,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
        coord_range: float = 1.5,
        # Systematic weighting parameters
        use_systematic_weights: bool = True,
        train_parquet_path: Optional[str] = None,
        zenith_key: str = "telescope_theta",
        energy_key: str = "true_energy",
        zenith_bins: int = 60,
        energy_bins: int = 35,
    ):
        """Initialize camera plane reconstruction task with systematic weighting.
        
        Args:
            coord_range: Maximum coordinate value for coordinate range info
            use_systematic_weights: Whether to apply systematic weight corrections
            train_parquet_path: Path to training parquet file for weight fitting
            zenith_key: Column name for zenith angle (radians)
            energy_key: Column name for true energy
            zenith_bins: Number of bins for zenith weighting
            energy_bins: Number of bins for energy weighting
        """
        super().__init__(
            hidden_size=hidden_size,
            target_labels=target_labels,
            prediction_labels=prediction_labels,
            loss_function=loss_function,
            transform_prediction_and_target=transform_prediction_and_target,
            transform_target=transform_target,
            transform_inference=transform_inference,
            additional_batch_keys=[
                zenith_key,
                energy_key,
            ],
        )
        self.coord_range = coord_range
        self.use_systematic_weights = use_systematic_weights
        self.zenith_key = zenith_key
        self.energy_key = energy_key
        
        # Initialize systematic weighter if requested
        self.systematic_weighter = None
        self._fitted_weight_params = None
        if use_systematic_weights and train_parquet_path is not None:
            print(f"Initializing systematic weighter from {train_parquet_path}")
            self.systematic_weighter = MAGICZenithEnergyWeighter(
                zenith_key=zenith_key,
                energy_key=energy_key,
                zenith_bins=zenith_bins,
                energy_bins=energy_bins,
            )
            print("Systematic weighter initialized successfully!")
            # Fit weights immediately to populate on-the-fly parameters
            self._fitted_weight_params = self.systematic_weighter.fit_systematic_weights_from_parquet(train_parquet_path)
        elif use_systematic_weights:
            print("Warning: use_systematic_weights=True but no train_parquet_path provided")
            print("Systematic weighting will be disabled")

    def _compute_systematic_weights(self, data: Data) -> Optional[Tensor]:
        if self.systematic_weighter is None:
            return None
        return self.systematic_weighter.get_weights_for_batch(data)

    def _forward(self, x: Tensor) -> Tensor:
        # Map latent activations to strictly positive energies
        energy = torch.nn.functional.softplus(x, beta=0.05) + eps_like(x)
        # Train in log-energy space for numerical stability
        return torch.log10(torch.clamp_min(energy, 1e-9))

    def compute_loss(self, pred: Tensor, data: Data) -> Tensor:
        """Compute loss with systematic weighting."""
        # Get predictions and targets
        energy_pred = pred
        target = data[self.default_target_labels[0]].view(-1, 1)
        target = self._transform_target(target)
        
        # Start with existing GraphNeT loss weights
        weights = data[self._loss_weight] if self._loss_weight is not None else None
        
        # Add systematic weights if enabled
        if self.use_systematic_weights and self.systematic_weighter is not None:
            systematic_weights = self._compute_systematic_weights(data)
            if systematic_weights is not None:
                if weights is not None:
                    weights = weights * systematic_weights
                else:
                    weights = systematic_weights
        
        # Compute per-event loss (energy)
        per_event_loss = self._loss_function(energy_pred, target, weights=None)
        
        # Apply systematic weights to final loss
        if weights is not None:
            weighted_loss = (per_event_loss * weights).mean()
        else:
            weighted_loss = per_event_loss.mean()
        
        # Add regularization
        total_loss = weighted_loss + self._regularisation_loss
        
        return total_loss

class MAGICEnergyRecoValidationLogger(Callback):
    """Log validation metrics for energy reconstruction with simple IRF stats."""

    def __init__(
        self,
        log_every_n_batches: int = 1,
        accumulate_over_epoch: bool = True,
        energy_key: str = "true_energy",
        prediction_key: str = "energy_pred",
        energy_bins: Optional[List[float]] = None,
        min_energy: float = 0.03,
        max_energy: float = 30.0,
        n_energy_bins: int = 8,
    ) -> None:
        """
        Args:
            log_every_n_batches: Frequency for per-batch logging when not accumulating.
            accumulate_over_epoch: If True, only log metrics at epoch end.
            energy_key: Key used to read true energies from the batch.
            prediction_key: Key used to identify prediction tensors from the model.
            energy_bins: Optional explicit bin edges (in TeV) for IRF-style stats.
            min_energy: Minimum energy used when auto-building bins (TeV).
            max_energy: Maximum energy used when auto-building bins (TeV).
            n_energy_bins: Number of bins when auto-building (log-spaced).
        """
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.accumulate_over_epoch = accumulate_over_epoch
        self.energy_key = energy_key
        self.prediction_key = prediction_key
        if energy_bins is not None and len(energy_bins) >= 2:
            self.energy_bins = sorted(energy_bins)
        else:
            self.energy_bins = list(
                np.geomspace(min_energy, max_energy, num=n_energy_bins)
            )
        self._errors: List[Tensor] = []
        self._preds: List[Tensor] = []
        self._targets: List[Tensor] = []

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _reset_epoch_storage(self) -> None:
        self._errors = []
        self._preds = []
        self._targets = []

    @staticmethod
    def _as_tensor(value: Optional[Tensor]) -> Optional[Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        try:
            return torch.as_tensor(value)
        except Exception:
            return None

    def _get_batch_value(self, batch, key: str) -> Optional[Tensor]:
        if key is None or batch is None:
            return None

        if isinstance(batch, (list, tuple)):
            tensors = [self._get_batch_value(item, key) for item in batch]
            tensors = [t for t in tensors if t is not None]
            if not tensors:
                return None
            return torch.cat(tensors, dim=0)

        if isinstance(batch, dict):
            if key in batch:
                return self._align_to_event(self._as_tensor(batch[key]), batch)
            for value in batch.values():
                candidate = self._get_batch_value(value, key)
                if candidate is not None:
                    return candidate
            return None

        if hasattr(batch, key):
            return self._align_to_event(self._as_tensor(getattr(batch, key)), batch)

        try:
            return self._align_to_event(self._as_tensor(batch[key]), batch)  # type: ignore[index]
        except Exception:
            return None

    @staticmethod
    def _align_to_event(values: Optional[Tensor], batch) -> Optional[Tensor]:
        if values is None:
            return None
        if values.dim() == 0:
            return values.view(1)
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is not None and values.shape[0] == num_graphs:
            return values
        batch_index = getattr(batch, "batch", None)
        if batch_index is not None and values.shape[0] == batch_index.shape[0]:
            return scatter_mean(values, batch_index, dim=0)
        return values

    def _extract_predictions(self, outputs, batch, pl_module) -> Optional[Tensor]:
        preds = None
        if isinstance(outputs, dict):
            if "preds" in outputs:
                preds = outputs["preds"]
            elif self.prediction_key in outputs:
                preds = outputs[self.prediction_key]
        elif isinstance(outputs, (list, tuple)):
            preds = outputs[0]
        else:
            preds = outputs

        tensor_like = self._as_tensor(preds) if preds is not None else None
        if tensor_like is not None and tensor_like.dim() == 0:
            tensor_like = None

        if preds is None or tensor_like is None:
            try:
                preds = pl_module(batch)
            except Exception:
                return None
        else:
            preds = tensor_like

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = self._as_tensor(preds)
        if preds is None:
            return None

        return preds.squeeze(-1)

    @staticmethod
    def _batch_size_from_tensor(tensor: Tensor) -> int:
        if tensor is None:
            return 1
        return tensor.shape[0]

    @staticmethod
    def _format_bin_tag(low: float, high: float) -> str:
        def fmt(x: float) -> str:
            text = np.format_float_positional(x, trim="-")
            return text.replace(".", "p")

        return f"{fmt(low)}_{fmt(high)}"

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._reset_epoch_storage()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        with torch.no_grad():
            preds = self._extract_predictions(outputs, batch, pl_module)
            targets = self._get_batch_value(batch, self.energy_key)

            if preds is None or targets is None:
                return

            preds = preds.view(-1)
            targets = targets.view(-1)

            if preds.shape[0] != targets.shape[0]:
                min_len = min(preds.shape[0], targets.shape[0])
                print(
                    f"[MAGICEnergyRecoValidationLogger] Length mismatch detected: "
                    f"preds={preds.shape[0]}, targets={targets.shape[0]}. "
                    f"Truncating both to {min_len} for metric computation."
                )
                preds = preds[:min_len]
                targets = targets[:min_len]

            errors = torch.abs(preds - targets)

            if self.accumulate_over_epoch:
                self._preds.append(preds.detach().cpu())
                self._targets.append(targets.detach().cpu())
                self._errors.append(errors.detach().cpu())
            elif batch_idx % self.log_every_n_batches == 0:
                metrics = self._compute_metrics(preds, targets, errors)
                pl_module.log_dict(
                    metrics,
                    batch_size=self._batch_size_from_tensor(targets),
                    sync_dist=True,
                )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.accumulate_over_epoch:
            return

        if len(self._errors) == 0:
            return

        preds = torch.cat(self._preds, dim=0)
        targets = torch.cat(self._targets, dim=0)
        errors = torch.cat(self._errors, dim=0)

        metrics = self._compute_metrics(preds, targets, errors)
        bin_metrics = self._compute_energy_bin_metrics(preds, targets, errors)
        metrics.update(bin_metrics)

        pl_module.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self._reset_epoch_storage()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _compute_metrics(
        self, preds: Tensor, targets: Tensor, errors: Tensor
    ) -> dict:
        diff = preds - targets
        rel_errors = errors / (targets.abs() + 1e-6)
        metrics = {
            "val_energy_mae": errors.mean().item(),
            "val_energy_median_error": errors.median().item(),
            "val_energy_68pct": torch.quantile(errors, 0.68).item(),
            "val_energy_95pct": torch.quantile(errors, 0.95).item(),
            "val_energy_bias": diff.mean().item(),
            "val_energy_rel_mae": rel_errors.mean().item(),
        }
        return metrics

    def _compute_energy_bin_metrics(
        self, preds: Tensor, targets: Tensor, errors: Tensor
    ) -> dict:
        metrics = {}
        diff = preds - targets
        bin_edges = torch.tensor(self.energy_bins, dtype=targets.dtype, device=targets.device)

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (targets >= low) & (targets < high)
            if not torch.any(mask):
                continue

            tag = self._format_bin_tag(low.item(), high.item())
            bin_errors = errors[mask]
            bin_diff = diff[mask]

            metrics[f"val_energy_mae_{tag}"] = bin_errors.mean().item()
            metrics[f"val_energy_bias_{tag}"] = bin_diff.mean().item()
            metrics[f"val_energy_count_{tag}"] = mask.sum().item()

        return metrics
    