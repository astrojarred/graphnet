from __future__ import annotations

from typing import List, cast, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import normalize as _normalize
from torch.nn.functional import softplus
from torch_geometric.data import Data
from graphnet.training.labels import Label
from graphnet.models.graphs.nodes import NodeDefinition

from graphnet.training.loss_functions import (
    EnsembleLoss,
    LossFunction,
    VonMisesFisher3DLoss,
)
from graphnet.utilities.maths import eps_like

"""Improved direction–reconstruction utilities for MAGIC telescopes.

This module keeps the original :class:`AngularOffsetLabel` convenience label
and introduces a *light-weight* alternative to the more complex FoV-aware loss
implemented in :pymod:`graphnet.models.task.magic_direction`.

Key differences
---------------
1.  **Task** – We reuse the *standard* :class:`~graphnet.models.task.reconstruction.DirectionReconstructionWithKappa` task.  Nothing new is
    defined here on the task side.
2.  **Loss** – An :class:`FieldOfViewPenaltyLoss` is added that *only* penalises
    predictions that fall outside a configurable field-of-view (FoV) cone.  It
    is **zero inside the FoV** and rises smoothly outside using a selectable
    penalty shape (``"logcosh"`` by default).
3.  **Ensemble loss** – :class:`MAGICEnsembleLoss` combines the canonical
    :class:`~graphnet.training.loss_functions.VonMisesFisher3DLoss` with the new
    FoV penalty by means of the generic
    :class:`~graphnet.training.loss_functions.EnsembleLoss` wrapper.

The intent is to achieve numerically stable training that discourages blatant
FoV violations without forcing the network to predict angular offsets
directly.
"""

__all__: List[str] = [
    "TrueTelescopeLabel",
    "DirectionWithAxisLabel",
    "FieldOfViewPenaltyLoss",
    "MAGICEnsembleLoss",
    "MAGICLimitedNodes",
]

class TrueTelescopeLabel(Label):
    """
    Produces a 2-component target tensor:
        [tel_theta, tel_phi]
    """
    def __init__(self, telescope_phi_key: str = "telescope_phi", telescope_theta_key: str = "telescope_theta", key: str = "true_telescope"):
        super().__init__(key=key)
        self.telescope_phi_key = telescope_phi_key
        self.telescope_theta_key = telescope_theta_key
    
    def __call__(self, row):

        # if true_ + label already exists, return it
        if "true_" + self.telescope_theta_key in row:
            theta = row["true_" + self.telescope_theta_key]
        else:
            theta = row[self.telescope_theta_key][0]

        if "true_" + self.telescope_phi_key in row:
            print(f"true_{self.telescope_phi_key} in row")
            phi = row["true_" + self.telescope_phi_key]
        else:
            phi = row[self.telescope_phi_key][0]

        return torch.tensor([theta, phi])

# graphnet/training/labels.py  (or any labels module)
class DirectionWithAxisLabel(Label):
    """
    Produces a 6-component target tensor:
        [dir_x, dir_y, dir_z, tel_x, tel_y, tel_z]
    """
    def __init__(
        self,
        dir_key: str = "direction",
        tel_key: str = "true_telescope",
        key: str = "direction_and_axis",
    ):
        super().__init__(key=key)
        self.dir_key = dir_key
        self.tel_key = tel_key

    def __call__(self, row):
        # True direction (already a tensor of shape (3,))
        dir_vec: Tensor = row[self.dir_key]

        # Telescope axis → Cartesian (3,)
        theta, phi = row[self.tel_key]
        tel_vec = torch.tensor(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dtype=dir_vec.dtype,
            device=dir_vec.device,
        ).reshape(-1, 3)

        return torch.cat([dir_vec, tel_vec]).flatten()

class FieldOfViewPenaltyLoss(LossFunction):
    """Penalty that activates *only* when a prediction leaves the FoV cone.

    The predicted direction is assumed to be given by the first three columns
    of ``prediction``—as produced by
    :class:`~graphnet.models.task.reconstruction.DirectionReconstructionWithKappa`.
    The FoV is modelled as a cone around the telescope pointing direction
    (z-axis).  Inside that cone the loss is strictly **zero**; outside it grows
    smoothly with the angular excess to avoid exploding gradients.

    Three different shapes are offered via ``loss_mode``:

    ``"logcosh"`` (default)
        Scales ~\|x\| for large deviations but quadratically for small ones;
        numerically robust.
    ``"mse"``
        Pure quadratic penalty *(x²)*.
    ``"rmse"``
        Square-root of the quadratic penalty *(sqrt(x² + ε))*—gives more weight
        to small deviations.
    """

    _VALID_MODES = {"logcosh", "mse", "rmse"}

    def __init__(
        self,
        fov_radius_deg: float = 2.5,
        loss_mode: str = "logcosh",
    ) -> None:
        super().__init__()
        if loss_mode not in self._VALID_MODES:
            raise ValueError(
                f"loss_mode must be one of {sorted(self._VALID_MODES)} (got {loss_mode})"
            )
        self._loss_mode = loss_mode
        self._fov_radius_rad: float = float(np.deg2rad(fov_radius_deg))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _log_cosh(x: Tensor) -> Tensor:
        """Numerically stable log-cosh implementation."""
        # identical to graphnet.training.loss_functions.LogCoshLoss._log_cosh
        return x + softplus(-2.0 * x) - np.log(2.0)

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:  # noqa: D401
        """Compute FoV penalty (element-wise).

        The penalty is proportional to the angular *excess* beyond
        ``fov_radius_deg`` between the predicted direction and *reference*
        direction:

        • If ``target`` has shape ``[N, 3]`` we fall back to using the ground-truth
            direction as reference (useful when a dedicated telescope-axis vector
            is not available).
        • If ``target`` has shape ``[N, 6]`` the **last three columns** are
            interpreted as the telescope pointing axis and used instead.  This
            allows strict physical FoV penalties without changing the core
            GraphNeT API.
        """
        if prediction.size(1) < 3:
            raise ValueError(
                "Prediction tensor must contain at least three columns (x,y,z)."
            )

        # Normalise predicted vector
        pred_dir: Tensor = _normalize(prediction[:, :3], dim=1)

        if target.size(1) >= 6:
            ref_dir = _normalize(target[:, 3:6], dim=1)
        else:
            ref_dir = _normalize(target[:, :3], dim=1)

        # Angular separation (0 … π)
        cos_sep = torch.sum(pred_dir * ref_dir, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angular_dist = torch.acos(cos_sep)

        # Excess outside FoV cone
        excess = torch.relu(angular_dist - self._fov_radius_rad)

        # Convert to desired loss shape (element-wise)
        if self._loss_mode == "mse":
            elements = excess ** 2
        elif self._loss_mode == "rmse":
            elements = torch.sqrt(excess ** 2 + eps_like(excess))
        else:  # "logcosh"
            elements = self._log_cosh(excess)

        return elements


class MAGICEnsembleLoss(EnsembleLoss):
    """Combine vMF likelihood with an FoV penalty term.

    The relative importance of the two constituents can be tuned via the
    ``vmf_weight`` and ``fov_weight`` parameters.
    """

    def __init__(
        self,
        vmf_weight: float = 1.0,
        fov_weight: float = 50.0,
        fov_radius_deg: float = 2.5,
        fov_loss_mode: str = "logcosh",
    ) -> None:
        vmf_loss = VonMisesFisher3DLoss()
        fov_loss = FieldOfViewPenaltyLoss(
            fov_radius_deg=fov_radius_deg, loss_mode=fov_loss_mode
        )
        super().__init__(
            loss_functions=cast(List[LossFunction], [vmf_loss, fov_loss]),
            loss_factors=[vmf_weight, fov_weight],        # 2 factors
            prediction_keys=[[0, 1, 2, 3], [0, 1, 2]],  # 2 keys
        )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        """Override forward to handle target reshaping before assertion check."""
        # Handle target tensor - it might have various problematic shapes
        if target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)  # [N, 1, 6] -> [N, 6]
        elif target.dim() == 2 and target.size(1) == 1:
            # DirectionWithAxisLabel produces flattened output that gets stacked as [N*6, 1]
            # We need to reshape it to [N, 6]
            batch_size = prediction.size(0)
            target = target.view(batch_size, -1)  # [N*6, 1] -> [N, 6]
        
        # Now call our custom _forward with properly shaped target
        elements = self._forward(prediction, target)
        
        if weights is not None:
            elements = elements * weights
        
        # Check that elements have correct batch size
        assert elements.size(dim=0) == target.size(dim=0), \
            "`_forward` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Custom forward that handles target slicing for different loss functions.
        
        VonMisesFisher3DLoss expects target[:, :3] (direction only)
        FieldOfViewPenaltyLoss can handle target with 6 columns (direction + telescope axis)
        """
        if self._prediction_keys is None:
            prediction_keys = [list(range(prediction.size(1)))] * len(self._loss_functions)
        else:
            prediction_keys = self._prediction_keys
            
        # Define target slices for each loss function
        # Index 0: VonMisesFisher3DLoss -> use only direction (first 3 columns)
        # Index 1: FieldOfViewPenaltyLoss -> use full target (handles both cases internally)
        target_slices = [
            target[:, :3],  # Direction only for vMF loss -> [N, 3]
            target,         # Full target for FoV loss -> [N, 6]
        ]
        
        for k, (loss_function, prediction_key, target_slice) in enumerate(
            zip(self._loss_functions, prediction_keys, target_slices)
        ):
            if k == 0:
                elements = self._factors[k] * loss_function._forward(
                    prediction=prediction[:, prediction_key], target=target_slice
                )
            else:
                elements += self._factors[k] * loss_function._forward(
                    prediction=prediction[:, prediction_key], target=target_slice
                )
        
        return elements

# NEW: MAGIC-specific node definition with pulse sampling
class MAGICLimitedNodes(NodeDefinition):
    """MAGIC node definition with pulse sampling to limit memory usage.
    
    This class samples pulses from MAGIC events to keep the number below
    a maximum threshold, preventing memory issues with large events.
    Can prioritize pulses by signal strength, time, or use random sampling.
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_pulses: int = 512,
        sampling_strategy: str = "signal",  # "signal", "time", "random"
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

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Define the output feature names (same as input for MAGIC)."""
        return input_feature_names

    def _sample_pulses(self, x: torch.Tensor) -> torch.Tensor:
        """Sample pulses based on the configured strategy."""
        n_pulses = x.shape[0]
        
        if n_pulses <= self.max_pulses:
            return torch.arange(n_pulses)
            
        if self.sampling_strategy == "signal":
            # Keep pulses with highest signal
            _, indices = torch.topk(x[:, self.signal_idx], self.max_pulses, largest=True)
            return indices.sort().values
            
        elif self.sampling_strategy == "time":
            # Keep earliest pulses
            _, indices = torch.topk(x[:, self.time_idx], self.max_pulses, largest=False)
            return indices.sort().values
            
        else:  # random
            indices = torch.randperm(n_pulses)[:self.max_pulses]
            return indices.sort().values

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
        """Construct nodes with pulse sampling."""
        # Sample pulses
        selected_indices = self._sample_pulses(x)
        
        # Extract selected pulses
        sampled_x = x[selected_indices]
        
        return Data(x=sampled_x), self.input_feature_names
