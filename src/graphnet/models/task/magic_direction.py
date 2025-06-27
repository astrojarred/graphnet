"""Angular offset reconstruction task for MAGIC telescopes.

This module implements direction reconstruction as angular offset from telescope
pointing, designed specifically for MAGIC's 3.5° field of view constraints.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union, Optional

from graphnet.models.task import StandardLearnedTask
from graphnet.training.loss_functions import LossFunction
from graphnet.training.labels import Label
from graphnet.utilities.maths import eps_like


class AngularOffsetLabel(Label):
    """Label for angular offset from telescope pointing.
    
    Converts telescope pointing and true direction into angular offset
    represented as a unit vector in telescope-centered coordinates.
    """

    def __init__(
        self,
        key: str = "angular_offset",
        telescope_phi_key: str = "telescope_phi",
        telescope_theta_key: str = "telescope_theta",
        true_phi_key: str = "true_phi", 
        true_theta_key: str = "true_theta",
    ):
        """Initialize AngularOffsetLabel.
        
        Args:
            telescope_phi_key: Column name for telescope azimuth (radians)
            telescope_theta_key: Column name for telescope zenith (radians)
            true_phi_key: Column name for true event azimuth (radians)
            true_theta_key: Column name for true event zenith (radians)
        """
        super().__init__(key=key)

        self.telescope_phi_key = telescope_phi_key
        self.telescope_theta_key = telescope_theta_key
        self.true_phi_key = true_phi_key
        self.true_theta_key = true_theta_key

    def __call__(self, row) -> torch.Tensor:
        """Convert telescope pointing and true direction to angular offset vector.
        
        Args:
            row: Data row containing telescope and true direction information
            
        Returns:
            Angular offset as unit vector [x, y, z] in telescope-centered coordinates
        """
        # Extract angles
        tel_phi = row[self.telescope_phi_key]
        tel_theta = row[self.telescope_theta_key] 
        true_phi = row[self.true_phi_key]
        true_theta = row[self.true_theta_key]
        
        # Convert to Cartesian coordinates on unit sphere
        # Telescope pointing direction (reference)
        tel_x = torch.sin(tel_theta) * torch.cos(tel_phi)
        tel_y = torch.sin(tel_theta) * torch.sin(tel_phi)
        tel_z = torch.cos(tel_theta)
        
        # True event direction
        true_x = torch.sin(true_theta) * torch.cos(true_phi)
        true_y = torch.sin(true_theta) * torch.sin(true_phi) 
        true_z = torch.cos(true_theta)
        
        # Calculate angular offset using rotation to telescope frame
        # Create rotation matrix to align telescope pointing with z-axis
        tel_dir = torch.stack([tel_x, tel_y, tel_z], dim=-1)  # [batch_size, 3]
        true_dir = torch.stack([true_x, true_y, true_z], dim=-1)  # [batch_size, 3]
        
        # Calculate offset using spherical coordinates relative to telescope
        # Dot product gives cosine of angular separation
        cos_sep = torch.sum(tel_dir * true_dir, dim=-1)  # Element-wise multiply then sum
        cos_sep = torch.clamp(cos_sep, -1.0, 1.0)  # Numerical stability
        
        # Angular separation
        angular_sep = torch.acos(cos_sep)
        
        # Calculate offset direction in telescope frame
        batch_size = angular_sep.shape[0] if angular_sep.dim() > 0 else 1
        
        # Handle batch dimension properly
        if angular_sep.dim() == 0:
            angular_sep = angular_sep.unsqueeze(0)
            tel_dir = tel_dir.unsqueeze(0)
            true_dir = true_dir.unsqueeze(0)
            cos_sep = cos_sep.unsqueeze(0)
        
        # Initialize output tensor
        result = torch.zeros(batch_size, 3, dtype=tel_dir.dtype, device=tel_dir.device)
        
        # Find events that are essentially on-axis
        on_axis_mask = angular_sep < 1e-6
        
        # Cross product gives perpendicular direction
        cross_prod = torch.cross(tel_dir, true_dir, dim=-1)
        cross_norm = torch.linalg.norm(cross_prod, dim=-1)
        
        # Find parallel or anti-parallel cases
        parallel_mask = cross_norm < 1e-6
        same_dir_mask = parallel_mask & (cos_sep > 0)
        opposite_dir_mask = parallel_mask & (cos_sep <= 0)
        
        # Set on-axis events
        result[on_axis_mask] = torch.tensor([0.0, 0.0, 1.0], dtype=result.dtype, device=result.device)
        
        # Set same direction events
        result[same_dir_mask] = torch.tensor([0.0, 0.0, 1.0], dtype=result.dtype, device=result.device)
        
        # Set opposite direction events
        result[opposite_dir_mask] = torch.tensor([0.0, 0.0, -1.0], dtype=result.dtype, device=result.device)
        
        # Handle general case
        general_mask = ~(on_axis_mask | parallel_mask)
        if general_mask.any():
            # Perpendicular unit vector
            perp_dir = cross_prod[general_mask] / cross_norm[general_mask].unsqueeze(-1)
            
            # Offset vector in telescope frame (polar coordinates)
            offset_x = torch.sin(angular_sep[general_mask]) * perp_dir[:, 0]
            offset_y = torch.sin(angular_sep[general_mask]) * perp_dir[:, 1]
            offset_z = torch.cos(angular_sep[general_mask])
            
            result[general_mask] = torch.stack([offset_x, offset_y, offset_z], dim=-1)
        
        # Always return a 2-D tensor of shape [batch, 3].
        # Keeping the leading batch dimension (even if batch_size == 1)
        return result


class MAGICFieldOfViewLoss(LossFunction):
    """Von Mises-Fisher loss with MAGIC field-of-view constraints.
    
    Combines VMF loss for directional prediction with strong penalties
    for predictions outside MAGIC's 3.5° field of view.
    """
    
    def __init__(
        self,
        fov_radius_deg: float = 1.75,  # MAGIC FoV radius in degrees
        fov_penalty_weight: float = 100.0,  # Penalty weight for FoV violations
        uncertainty_regularization: float = 0.01,  # Kappa regularization weight
    ):
        """Initialize MAGIC FoV loss.
        
        Args:
            fov_radius_deg: Field of view radius in degrees (default 1.75° for 3.5° diameter)
            fov_penalty_weight: Weight for penalty outside FoV
            uncertainty_regularization: Weight for kappa regularization
        """
        super().__init__()
        self.fov_radius_rad = np.deg2rad(fov_radius_deg)
        self.fov_penalty_weight = fov_penalty_weight
        self.uncertainty_reg = uncertainty_regularization
        
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate loss with FoV constraints.
        
        Args:
            prediction: Model output [N, 4] = [x_offset, y_offset, z_offset, kappa]
            target: Target offset vectors [N, 3] = [x_target, y_target, z_target]
            
        Returns:
            Loss tensor [N,]
        """
        # Extract predictions
        pred_direction = prediction[:, :3]  # [N, 3]
        kappa = prediction[:, 3]  # [N,]
        
        # Ensure target is properly shaped
        if target.dim() == 1:
            target = target.unsqueeze(0)
        elif target.dim() == 3:
            # Handle case where target has shape [batch_size, 1, 3] from torch.stack
            target = target.squeeze(1)
        target = target.reshape(-1, 3)
        
        # Normalize predicted directions
        pred_norm = torch.linalg.norm(pred_direction, dim=1, keepdim=True)
        pred_direction_norm = pred_direction / (pred_norm + eps_like(pred_direction))
        
        # Calculate Von Mises-Fisher loss
        # VMF loss: -kappa * dot(pred, target) + log(I0(kappa))
        dot_product = torch.sum(pred_direction_norm * target, dim=1)
        
        # Approximate log(I0(kappa)) for numerical stability
        # For large kappa: log(I0(kappa)) ≈ kappa - 0.5*log(2*pi*kappa)
        # For small kappa: log(I0(kappa)) ≈ log(1 + kappa^2/4)
        log_i0_approx = torch.where(
            kappa > 10,
            kappa - 0.5 * torch.log(2 * np.pi * kappa + eps_like(kappa)),
            torch.log(1 + kappa**2 / 4 + eps_like(kappa))
        )
        
        vmf_loss = -kappa * dot_product + log_i0_approx
        
        # Calculate field-of-view penalty
        # Convert offset vectors to angular separations
        pred_angular_sep = torch.acos(torch.clamp(pred_direction_norm[:, 2], -1.0, 1.0))
        target_angular_sep = torch.acos(torch.clamp(target[:, 2], -1.0, 1.0))
        
        # Penalty for predictions outside FoV
        fov_violation_pred = torch.relu(pred_angular_sep - self.fov_radius_rad)
        fov_violation_target = torch.relu(target_angular_sep - self.fov_radius_rad)
        
        # Strong penalty for predicting outside FoV
        fov_penalty = self.fov_penalty_weight * (fov_violation_pred**2)
        
        # Additional penalty if target is outside FoV (data quality issue)
        target_penalty = self.fov_penalty_weight * 0.1 * (fov_violation_target**2)
        
        # Kappa regularization to prevent over-confidence
        kappa_reg = self.uncertainty_reg * torch.relu(kappa - 100)**2
        
        # Total loss
        total_loss = vmf_loss + fov_penalty + target_penalty + kappa_reg
        
        return total_loss


class AngularOffsetReconstructionWithKappa(StandardLearnedTask):
    """Task for reconstructing angular offset from telescope pointing.
    
    Predicts the direction of detected events as angular offset from the
    telescope pointing direction, with uncertainty quantification via kappa.
    Specifically designed for MAGIC telescope constraints.
    """
    
    default_target_labels = ["angular_offset"]
    default_prediction_labels = [
        "x_offset_pred", 
        "y_offset_pred", 
        "z_offset_pred", 
        "offset_kappa"
    ]
    
    def __init__(
        self,
        hidden_size: int,
        target_labels: Union[str, List[str]] = "angular_offset",
        prediction_labels: Optional[List[str]] = None,
        loss_function: Optional[LossFunction] = None,
        transform_prediction_and_target: Optional[callable] = None,
        transform_target: Optional[callable] = None,
        transform_inference: Optional[callable] = None,
    ):
        """Initialize angular offset reconstruction task.
        
        Args:
            hidden_size: Size of the hidden layer from backbone
            target_labels: Target label name(s)
            prediction_labels: Prediction label names
            loss_function: Loss function (defaults to MAGICFieldOfViewLoss)
            transform_prediction_and_target: Optional transformation function
            transform_target: Optional target transformation function  
            transform_inference: Optional inference transformation function
        """
        # Store hidden_size BEFORE calling parent constructor (needed for nb_inputs property)
        self.hidden_size = hidden_size
        
        # Set default loss function
        if loss_function is None:
            loss_function = MAGICFieldOfViewLoss()
            
        # Set default prediction labels
        if prediction_labels is None:
            prediction_labels = self.default_prediction_labels
            
        super().__init__(
            hidden_size=hidden_size,
            target_labels=target_labels,
            prediction_labels=prediction_labels,
            loss_function=loss_function,
            transform_prediction_and_target=transform_prediction_and_target,
            transform_target=transform_target,
            transform_inference=transform_inference,
        )
        
        # Neural network layers for offset prediction
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),  # GELU works better than ReLU for transformers
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 3)  # x, y, z offset components
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights with small values for stability."""
        for module in self.offset_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
                
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass for angular offset prediction.
        
        Args:
            x: Input tensor from backbone [batch_size, hidden_size]
            
        Returns:
            Prediction tensor [batch_size, 4] = [x_offset, y_offset, z_offset, kappa]
        """
        # Predict offset vector
        offset_vector = self.offset_head(x)  # [batch_size, 3]
        
        # Calculate concentration parameter (kappa) from vector magnitude
        # This encodes prediction confidence - longer vectors = higher confidence
        vector_magnitude = torch.linalg.norm(offset_vector, dim=1)
        
        # Apply bounds to kappa for numerical stability
        kappa = torch.clamp(vector_magnitude, min=0.1, max=50.0) + eps_like(x)
        
        # Normalize direction vector
        direction = offset_vector / (vector_magnitude.unsqueeze(1) + eps_like(x))
        
        # Ensure z-component is positive (pointing forward from telescope)
        # Flip if pointing backward to keep within reasonable bounds
        z_component = direction[:, 2]
        flip_mask = z_component < -0.5  # If pointing significantly backward
        direction[flip_mask] = -direction[flip_mask]
        
        # Combine direction and concentration
        prediction = torch.cat([direction, kappa.unsqueeze(1)], dim=1)
        
        return prediction
    
    def shared_step(self, batch, batch_idx: int):
        """Shared step for training/validation with additional metrics."""
        # Standard shared step
        outputs = super().shared_step(batch, batch_idx)
        
        # Add custom metrics for monitoring
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            predictions = outputs.prediction
            targets = outputs.target
            
            # Calculate angular resolution metrics
            with torch.no_grad():
                pred_directions = predictions[:, :3]
                target_directions = targets.reshape(-1, 3)
                
                # Angular separation in degrees
                dot_products = torch.sum(pred_directions * target_directions, dim=1)
                dot_products = torch.clamp(dot_products, -1.0, 1.0)
                angular_separations = torch.acos(dot_products)
                angular_sep_deg = torch.rad2deg(angular_separations)
                
                # Predicted confidence (kappa)
                predicted_kappa = predictions[:, 3]
                
                # Check field-of-view violations
                pred_angular_from_center = torch.acos(torch.clamp(pred_directions[:, 2], -1.0, 1.0))
                fov_violations = (pred_angular_from_center > np.deg2rad(1.75)).float()
                
                # Calculate metrics
                metrics = {
                    "angular_resolution_median": torch.median(angular_sep_deg),
                    "angular_resolution_68": torch.quantile(angular_sep_deg, 0.68),
                    "angular_resolution_95": torch.quantile(angular_sep_deg, 0.95),
                    "mean_kappa": torch.mean(predicted_kappa),
                    "fov_violation_rate": torch.mean(fov_violations),
                }
                
                # Log metrics to trainer logger (for W&B)
                if hasattr(self, 'trainer') and self.trainer is not None and self.trainer.logger is not None:
                    # Determine if we're in training or validation mode
                    mode = "train" if self.training else "val"
                    logged_metrics = {f"{mode}_{k}": v.item() for k, v in metrics.items()}
                    self.trainer.logger.log_metrics(logged_metrics, step=self.trainer.global_step)
                
                # Also store on outputs for backward compatibility
                for k, v in metrics.items():
                    setattr(outputs, k, v)
                
        return outputs

    @property
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""
        return self.hidden_size


# Example usage and configuration
def create_angular_offset_model(backbone, detector):
    """Create a complete model with angular offset reconstruction.
    
    Args:
        backbone: Trained backbone model (e.g., DeepIce)
        detector: Detector configuration (e.g., MAGICDetector)
        
    Returns:
        Configured StandardModel for angular offset reconstruction
    """
    from graphnet.models import StandardModel
    from graphnet.models.graphs import KNNGraph
    from graphnet.models.graphs.nodes import NodesAsPulses
    
    # Graph definition
    graph_definition = KNNGraph(
        detector=detector,
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,  # Proven optimal from IceCube competition
    )
    
    # Angular offset task with MAGIC-specific loss
    task = AngularOffsetReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels="angular_offset",
        loss_function=MAGICFieldOfViewLoss(
            fov_radius_deg=1.75,  # MAGIC 3.5° FoV
            fov_penalty_weight=100.0,
            uncertainty_regularization=0.01
        )
    )
    
    # Complete model
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
    )
    
    return model


# Labels for data loading
def get_angular_offset_labels():
    """Get label configuration for angular offset training."""
    return {
        "angular_offset": AngularOffsetLabel(
            telescope_phi_key="telescope_phi",
            telescope_theta_key="telescope_theta", 
            true_phi_key="true_phi",
            true_theta_key="true_theta"
        )
    }
