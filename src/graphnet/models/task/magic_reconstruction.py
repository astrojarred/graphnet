"""MAGIC-specific reconstruction task classes.

Direction reconstruction tasks optimized for MAGIC telescope data,
based on IceCube Kaggle competition winning strategies.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like


def angles_to_direction_vector(theta: Tensor, phi: Tensor) -> Tensor:
    """Convert spherical coordinates to 3D direction vector.
    
    Args:
        theta: Zenith angles in radians [batch]
        phi: Azimuth angles in radians [batch]
        
    Returns:
        Direction vectors [batch, 3] with (x, y, z)
    """
    # Convert spherical to Cartesian coordinates
    # x = sin(theta) * cos(phi)
    # y = sin(theta) * sin(phi)  
    # z = cos(theta)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    return torch.stack([x, y, z], dim=1)


def direction_vector_to_angles(direction: Tensor) -> Tensor:
    """Convert 3D direction vector to spherical coordinates.
    
    Args:
        direction: Direction vectors [batch, 3] with (x, y, z)
        
    Returns:
        Angles [batch, 2] with (theta, phi) in radians
    """
    # Normalize to ensure unit vectors
    direction = F.normalize(direction, p=2, dim=1)
    
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    
    # Convert Cartesian to spherical coordinates
    # theta = arccos(z)  (zenith angle)
    # phi = atan2(y, x)  (azimuth angle)
    theta = torch.acos(torch.clamp(z, -1.0 + 1e-7, 1.0 - 1e-7))  # Clamp to avoid numerical issues
    phi = torch.atan2(y, x)
    
    return torch.stack([theta, phi], dim=1)


class MAGICDirectionReconstructionVMF(StandardLearnedTask):
    """Direction reconstruction using Von Mises-Fisher distribution.
    
    Based on IceCube winning solutions using VMF loss for directional data.
    This approach treats direction as a point on the sphere and uses the
    Von Mises-Fisher distribution for uncertainty modeling.
    """
    
    default_target_labels = ["true_theta", "true_phi"]  # MAGIC dataset columns
    default_prediction_labels = ["dir_x", "dir_y", "dir_z", "kappa"]
    nb_inputs = 4  # 3D direction + concentration
    
    def __init__(self, **kwargs):
        """Initialize with automatic transforms for spherical to Cartesian."""
        # Set transforms to convert between (theta, phi) and direction vectors
        if 'transform_target' not in kwargs:
            kwargs['transform_target'] = self._transform_angles_to_direction
        if 'transform_inference' not in kwargs:
            kwargs['transform_inference'] = self._transform_direction_to_angles
        super().__init__(**kwargs)
    
    def _transform_angles_to_direction(self, target: Tensor) -> Tensor:
        """Transform target from (theta, phi) to (dir_x, dir_y, dir_z).
        
        Args:
            target: [batch, 2] containing (theta, phi) in radians
            
        Returns:
            Direction vectors [batch, 3] containing (x, y, z)
        """
        theta = target[:, 0]  # Zenith angle
        phi = target[:, 1]    # Azimuth angle
        return angles_to_direction_vector(theta, phi)
    
    def _transform_direction_to_angles(self, prediction: Tensor) -> Tensor:
        """Transform prediction from direction vectors + kappa to (theta, phi, kappa).
        
        Args:
            prediction: [batch, 4] containing (dir_x, dir_y, dir_z, kappa)
            
        Returns:
            Angles [batch, 4] containing (theta, phi, dir_z, kappa) - keeping original format
        """
        direction = prediction[:, :3]
        kappa = prediction[:, 3:4]
        
        # Convert direction to angles
        angles = direction_vector_to_angles(direction)
        
        # Return angles + kappa, but keep the dir_z for compatibility
        return torch.cat([angles, prediction[:, 2:3], kappa], dim=1)
    
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass normalizing direction and ensuring positive kappa."""
        # Normalize direction vector
        direction = x[:, :3]
        direction = F.normalize(direction, p=2, dim=1)
        
        # Ensure positive concentration
        kappa = F.softplus(x[:, 3:4]) + eps_like(x)
        
        return torch.cat([direction, kappa], dim=1)


class MAGICDirectionClassification(StandardLearnedTask):
    """Direction classification using fine angular bins.
    
    Based on IceCube strategies of discretizing the sphere into angular bins
    and treating direction as a classification problem.
    """
    
    default_target_labels = ["true_theta", "true_phi"]  # MAGIC dataset columns
    default_prediction_labels = ["angular_class"]
    
    def __init__(self, num_bins: int = 64, **kwargs):
        self.num_bins = num_bins
        # Set transforms to convert (theta, phi) to bin index and back
        if 'transform_target' not in kwargs:
            kwargs['transform_target'] = self._transform_angles_to_bins
        if 'transform_inference' not in kwargs:
            kwargs['transform_inference'] = self._transform_bins_to_angles
        super().__init__(**kwargs)
    
    def _transform_angles_to_bins(self, target: Tensor) -> Tensor:
        """Transform target from (theta, phi) to bin index.
        
        Args:
            target: [batch, 2] containing (theta, phi) in radians
            
        Returns:
            Bin indices [batch, 1] containing integer class indices
        """
        theta = target[:, 0]  # Zenith angle [0, pi]
        phi = target[:, 1]    # Azimuth angle [0, 2pi]
        
        # Simple binning strategy: divide sphere into equal-area bins
        # For simplicity, use rectangular binning (can be improved with HEALPix)
        theta_bins = int(self.num_bins ** 0.5)  # Square root for roughly square bins
        phi_bins = int(self.num_bins / theta_bins)
        
        # Normalize angles to [0, 1]
        theta_norm = theta / torch.pi
        phi_norm = phi / (2 * torch.pi)
        
        # Convert to bin indices
        theta_bin_idx = torch.clamp((theta_norm * theta_bins).long(), 0, theta_bins - 1)
        phi_bin_idx = torch.clamp((phi_norm * phi_bins).long(), 0, phi_bins - 1)
        
        # Combined bin index
        bin_idx = theta_bin_idx * phi_bins + phi_bin_idx
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        
        return bin_idx.unsqueeze(1).float()  # [batch, 1]
    
    def _transform_bins_to_angles(self, prediction: Tensor) -> Tensor:
        """Transform prediction from bin logits to (theta, phi).
        
        Args:
            prediction: [batch, num_bins] containing log-probabilities for each bin
            
        Returns:
            Angles [batch, 2] containing (theta, phi) in radians
        """
        # Get the most likely bin
        bin_idx = torch.argmax(prediction, dim=1)  # [batch]
        
        # Convert back to angles
        theta_bins = int(self.num_bins ** 0.5)
        phi_bins = int(self.num_bins / theta_bins)
        
        theta_bin_idx = bin_idx // phi_bins
        phi_bin_idx = bin_idx % phi_bins
        
        # Convert bin indices back to continuous angles
        theta = (theta_bin_idx.float() + 0.5) / theta_bins * torch.pi
        phi = (phi_bin_idx.float() + 0.5) / phi_bins * 2 * torch.pi
        
        return torch.stack([theta, phi], dim=1)
    
    @property
    def nb_inputs(self) -> int:
        return self.num_bins
    
    def _forward(self, x: Tensor) -> Tensor:
        """Apply softmax for classification."""
        return F.log_softmax(x, dim=1)


class MAGICHybridDirectionTask(StandardLearnedTask):
    """Hybrid task combining VMF regression and classification.
    
    This task handles the output from the MAGICHybridModel which produces
    both VMF parameters and classification logits for ensemble learning.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_classification_bins: int = 136,
        **kwargs
    ):
        """Initialize hybrid direction task.
        
        Args:
            hidden_size: Size of hidden layer.
            num_classification_bins: Number of classification bins.
            **kwargs: Additional arguments passed to parent class.
        """
        self.num_classification_bins = num_classification_bins
        # Set transforms for hybrid outputs
        if 'transform_target' not in kwargs:
            kwargs['transform_target'] = self._transform_angles_to_direction
        if 'transform_inference' not in kwargs:
            kwargs['transform_inference'] = self._transform_hybrid_to_angles
        # Total outputs: 3 (direction) + 1 (kappa) + num_bins (classification logits)
        total_outputs = 4 + num_classification_bins
        super().__init__(hidden_size, **kwargs)
    
    default_target_labels = ["true_theta", "true_phi"]  # MAGIC dataset columns
    default_prediction_labels = [
        "dir_x", "dir_y", "dir_z", "kappa", "bin_logits"
    ]
    
    def _transform_angles_to_direction(self, target: Tensor) -> Tensor:
        """Transform target from (theta, phi) to direction vector."""
        theta = target[:, 0]
        phi = target[:, 1]
        return angles_to_direction_vector(theta, phi)
    
    def _transform_hybrid_to_angles(self, prediction: Tensor) -> Tensor:
        """Transform hybrid prediction to angles."""
        # Extract direction vector from prediction
        direction = prediction[:, :3]
        return direction_vector_to_angles(direction)
    
    @property 
    def nb_inputs(self) -> int:
        """Return number of input features."""
        return 4 + self.num_classification_bins
    
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass handling both VMF and classification outputs."""
        # Split outputs
        direction = x[:, :3]
        kappa = x[:, 3:4]
        classification_logits = x[:, 4:4+self.num_classification_bins]
        
        # Normalize direction vector
        direction = F.normalize(direction, p=2, dim=1)
        
        # Ensure positive concentration
        kappa = F.softplus(kappa) + eps_like(x)
        
        # Return combined predictions
        return torch.cat([direction, kappa, classification_logits], dim=1)


class MAGICAngularResolution(StandardLearnedTask):
    """Task for predicting angular resolution/uncertainty.
    
    Predicts the expected angular resolution for each event,
    useful for event selection and weighting in analyses.
    """
    
    default_target_labels = ["true_theta", "true_phi"]  # MAGIC dataset columns
    default_prediction_labels = ["predicted_angular_error"]
    nb_inputs = 1
    
    def __init__(self, **kwargs):
        """Initialize with transforms to compute angular error from true angles."""
        if 'transform_target' not in kwargs:
            kwargs['transform_target'] = self._transform_angles_to_error
        if 'transform_inference' not in kwargs:
            kwargs['transform_inference'] = self._identity_transform
        super().__init__(**kwargs)
    
    def _transform_angles_to_error(self, target: Tensor) -> Tensor:
        """Transform target angles to dummy angular error (for compatibility)."""
        # Return dummy error for training (actual error computed differently)
        batch_size = target.shape[0]
        return torch.ones(batch_size, 1, device=target.device) * 0.1  # 0.1 degree default
    
    def _identity_transform(self, prediction: Tensor) -> Tensor:
        """Identity transform for inference."""
        return prediction
    
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass ensuring positive angular error."""
        # Ensure positive angular error (in degrees)
        angular_error = F.softplus(x) + eps_like(x)
        return angular_error 
