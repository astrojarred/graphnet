from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor
from torch.nn.functional import normalize as _normalize  # FIXED: Local import

from graphnet.training.loss_functions import LossFunction, VonMisesFisher3DLoss

from .magic_direction_improved import FieldOfViewPenaltyLoss

__all__ = ["PaperEnsembleLoss", "SmallAngleMSE"]


class SmallAngleMSE(LossFunction):
    """Quadratic loss for small angular errors. Returns element-wise loss."""
    def __init__(self, angle_thresh_deg: float = 0.5):
        super().__init__()
        self.threshold = torch.deg2rad(torch.tensor(angle_thresh_deg))

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_direction = _normalize(prediction[:, :3], dim=1)
        true_direction = _normalize(target[:, :3], dim=1)
        # FIXED: Tighter clamping to prevent NaN
        cos_angle = torch.sum(pred_direction * true_direction, dim=1).clamp(-1 + 1e-9, 1 - 1e-9)
        angle = torch.acos(cos_angle)
        return (angle / self.threshold.to(angle.device)) ** 2


class PaperEnsembleLoss(LossFunction):
    """
    Flexible ensemble loss implementing all our strategies.
    Returns element-wise loss as required by GraphNeT framework.
    """
    def __init__(
        self,
        angle_weight: float = 1.0,
        vmf_weight: float = 0.05,
        fov_weight: float = 0.0,
        fov_radius_deg: float = 2.5,
        fov_loss_mode: str = "mse",
        small_angle_weight: float = 0.0,
        small_angle_thresh_deg: float = 0.5,
    ):
        super().__init__()
        self.weights = {
            "angle": angle_weight,
            "vmf": vmf_weight,
            "fov": fov_weight,
            "small_angle": small_angle_weight,
        }

        # Initialize loss components only if needed
        if self.weights["vmf"] > 0:
            self.vmf_loss = VonMisesFisher3DLoss()
        if self.weights["fov"] > 0:
            # FIXED: Correct parameter name is 'fov_radius_deg' not 'radius'
            self.fov_loss = FieldOfViewPenaltyLoss(
                fov_radius_deg=fov_radius_deg, 
                loss_mode=fov_loss_mode
            )
        if self.weights["small_angle"] > 0:
            self.small_angle_loss = SmallAngleMSE(
                angle_thresh_deg=small_angle_thresh_deg
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
            if target.numel() == batch_size * 6:
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
        # Handle target format [N, 6] -> direction[:3], telescope_axis[3:6]
        true_direction = target[:, :3]
        
        # Initialize element-wise loss tensor
        total_loss = torch.zeros(prediction.size(0), device=prediction.device)

        # Competition metric: arccos(|dot|) - element-wise
        if self.weights["angle"] > 0:
            pred_dir = _normalize(prediction[:, :3], dim=1)
            true_dir = _normalize(true_direction, dim=1)
            # FIXED: Tighter clamping for stability
            cos_angle = torch.sum(pred_dir * true_dir, dim=1).clamp(-1 + 1e-9, 1 - 1e-9)
            angle_loss = torch.acos(cos_angle)  # Element-wise
            total_loss += self.weights["angle"] * angle_loss

        # VMF loss - already returns element-wise
        if self.weights["vmf"] > 0:
            vmf_elements = self.vmf_loss(prediction, true_direction)
            total_loss += self.weights["vmf"] * vmf_elements

        # FoV loss - use full target (includes telescope axis)
        if self.weights["fov"] > 0:
            fov_elements = self.fov_loss(prediction, target)
            total_loss += self.weights["fov"] * fov_elements

        # Small angle loss - element-wise
        if self.weights["small_angle"] > 0:
            small_elements = self.small_angle_loss(prediction, target)
            total_loss += self.weights["small_angle"] * small_elements
            
        return total_loss
