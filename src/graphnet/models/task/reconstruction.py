"""Reconstruction-specific `Model` class(es)."""

import numpy as np
import torch
from torch import Tensor

from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like


class AzimuthReconstructionWithKappa(StandardLearnedTask):
    """Reconstructs azimuthal angle and associated kappa (1/var)."""

    # Requires two features: untransformed points in (x,y)-space.
    default_target_labels = ["azimuth"]
    default_prediction_labels = ["azimuth_pred", "azimuth_kappa"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        angle = torch.atan2(x[:, 1], x[:, 0])
        angle = torch.where(
            angle < 0, angle + 2 * np.pi, angle
        )  # atan(y,x) -> [-pi, pi]
        return torch.stack((angle, kappa), dim=1)


class AzimuthReconstruction(AzimuthReconstructionWithKappa):
    """Reconstructs azimuthal angle."""

    # Requires two features: untransformed points in (x,y)-space.
    default_target_labels = ["azimuth"]
    default_prediction_labels = ["azimuth_pred"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        res = super()._forward(x)
        angle = res[:, 0].unsqueeze(1)
        kappa = res[:, 1]
        sigma = torch.sqrt(1.0 / kappa)
        beta = 1e-3
        kl_loss = torch.mean(sigma**2 - torch.log(sigma) - 1)
        self._regularisation_loss += beta * kl_loss
        return angle


class DirectionReconstructionWithKappa(StandardLearnedTask):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    default_target_labels = [
        "direction"
    ]  # contains dir_x, dir_y, dir_z see https://github.com/graphnet-team/graphnet/blob/95309556cfd46a4046bc4bd7609888aab649e295/src/graphnet/training/labels.py#L29
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)


class ZenithReconstruction(StandardLearnedTask):
    """Reconstructs zenith angle."""

    # Requires two features: zenith angle itself.
    default_target_labels = ["zenith"]
    default_prediction_labels = ["zenith_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        return torch.sigmoid(x[:, :1]) * np.pi


class ZenithReconstructionWithKappa(ZenithReconstruction):
    """Reconstructs zenith angle and associated kappa (1/var)."""

    # Requires one feature in addition to `ZenithReconstruction`: kappa (unceratinty; 1/variance).
    default_target_labels = ["zenith"]
    default_prediction_labels = ["zenith_pred", "zenith_kappa"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:, :1]).squeeze(1)
        kappa = torch.abs(x[:, 1]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)


class EnergyReconstruction(StandardLearnedTask):
    """Reconstructs energy using stable method."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        return torch.nn.functional.softplus(x, beta=0.05) + eps_like(x)


class EnergyReconstructionWithPower(StandardLearnedTask):
    """Reconstructs energy."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        return torch.pow(10, x[:, 0] + 1.0).unsqueeze(1)


class EnergyTCReconstruction(StandardLearnedTask):
    """Reconstructs track and cascade energies using stable method."""

    # Requires two features: untransformed energy for track and cascade
    default_target_labels = ["energy_track", "energy_cascade"]
    default_prediction_labels = ["energy_track_pred", "energy_cascade_pred"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        x[:, 0] = torch.nn.functional.softplus(
            x[:, 0].clone(), beta=0.05
        ) + eps_like(x[:, 0].clone())
        x[:, 1] = torch.nn.functional.softplus(
            x[:, 1].clone(), beta=0.05
        ) + eps_like(x[:, 1].clone())
        return x


class EnergyReconstructionWithUncertainty(EnergyReconstruction):
    """Reconstructs energy and associated uncertainty (log(var))."""

    # Requires one feature in addition to `EnergyReconstruction`: log-variance (uncertainty).
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred", "energy_sigma"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        energy = super()._forward(x[:, :1]).squeeze(1)
        log_var = x[:, 1]
        pred = torch.stack((energy, log_var), dim=1)
        return pred


class VertexReconstruction(StandardLearnedTask):
    """Reconstructs vertex position and time."""

    # Requires four features, x, y, z, and t.
    default_target_labels = ["vertex"]
    default_prediction_labels = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
        "interaction_time_pred",
    ]
    nb_inputs = 4

    def _forward(self, x: Tensor) -> Tensor:
        # Scale xyz to roughly the right order of magnitude, leave time
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class PositionReconstruction(StandardLearnedTask):
    """Reconstructs vertex position."""

    # Requires three features, x, y, and z.
    default_target_labels = ["position"]
    default_prediction_labels = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class TimeReconstruction(StandardLearnedTask):
    """Reconstructs time."""

    # Requires one feature, time.
    default_target_labels = ["interaction_time"]
    default_prediction_labels = ["interaction_time_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Leave as it is
        return x


class InelasticityReconstruction(StandardLearnedTask):
    """Reconstructs interaction inelasticity.

    That is, 1-(track energy / hadronic energy).
    """

    # Requires one features: inelasticity itself
    default_target_labels = ["inelasticity"]
    default_prediction_labels = ["inelasticity_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform output to unit range
        return torch.sigmoid(x)


class HybridDirectionTask(StandardLearnedTask):
    """Hybrid direction reconstruction combining VMF regression with angular binning classification.
    
    This task implements a multi-head approach that outputs both:
    1. Von Mises-Fisher (VMF) parameters for continuous direction prediction
    2. Classification probabilities for fine-grained angular bins within region of interest (ROI)
    
    The approach is inspired by IceCube Kaggle competition winners who combined
    continuous regression with discrete classification for improved precision.
    """
    
    # Default to 68 total outputs (4 VMF + 64 bins + 1 outside ROI)
    nb_inputs = 69
    default_target_labels = ["direction"]
    default_prediction_labels = [
        "dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa",
        "bin_probs"
    ]

    def __init__(
        self,
        hidden_size: int,
        num_fine_bins: int = 64,
        roi_radius_deg: float = 0.5,
        **kwargs,
    ):
        """Initialize hybrid direction task.
        
        Args:
            hidden_size: Size of backbone output
            num_fine_bins: Number of fine angular bins within ROI
            roi_radius_deg: Radius of region of interest in degrees
            **kwargs: Additional arguments for StandardLearnedTask
        """
        self.num_fine_bins = num_fine_bins
        self.roi_radius_deg = roi_radius_deg
        self.roi_radius_rad = np.deg2rad(roi_radius_deg)
        
        # Override nb_inputs based on actual number of bins
        self.nb_inputs = 4 + num_fine_bins + 1  # VMF + fine bins + outside ROI
        
        super().__init__(hidden_size=hidden_size, **kwargs)
        
        # Create bin centers for fine bins within ROI
        self.bin_centers = self._create_bin_centers()
    
    def _create_bin_centers(self):
        """Create bin centers for fine angular bins within ROI.
        
        Returns:
            Tensor: Bin centers as (theta, phi) pairs in radians
        """
        # Create fine grid within ROI circle
        # Use approximately square grid with slight over-sampling
        n_per_side = int(np.ceil(np.sqrt(self.num_fine_bins)))
        
        # Create grid in [-roi_radius, +roi_radius]
        theta_vals = np.linspace(-self.roi_radius_rad, self.roi_radius_rad, n_per_side)
        phi_vals = np.linspace(-self.roi_radius_rad, self.roi_radius_rad, n_per_side)
        
        # Create meshgrid and select points within circle
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        
        # Keep only points within ROI circle
        distances = np.sqrt(theta_flat**2 + phi_flat**2)
        within_roi = distances <= self.roi_radius_rad
        
        theta_roi = theta_flat[within_roi]
        phi_roi = phi_flat[within_roi]
        
        # Take first num_fine_bins points
        if len(theta_roi) > self.num_fine_bins:
            theta_roi = theta_roi[:self.num_fine_bins]
            phi_roi = phi_roi[:self.num_fine_bins]
        
        # Convert to tensor
        bin_centers = torch.tensor(np.column_stack((theta_roi, phi_roi)), dtype=torch.float32)
        return bin_centers
    
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass combining VMF regression and classification.
        
        Args:
            x: Input tensor of shape (batch_size, nb_inputs)
            
        Returns:
            Tensor: Combined predictions [VMF params, bin probabilities]
        """
        # Split outputs
        vmf_raw = x[:, :4]  # First 4 outputs for VMF
        classification_raw = x[:, 4:]  # Remaining outputs for classification
        
        # Process VMF part (same as DirectionReconstructionWithKappa)
        kappa = torch.linalg.vector_norm(vmf_raw[:, :3], dim=1) + eps_like(vmf_raw)
        vec_x = vmf_raw[:, 0] / kappa
        vec_y = vmf_raw[:, 1] / kappa
        vec_z = vmf_raw[:, 2] / kappa
        vmf_pred = torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)
        
        # Process classification part
        bin_probs = torch.softmax(classification_raw, dim=-1)
        
        # Combine outputs
        return torch.cat([vmf_pred, bin_probs], dim=1)

    def assign_direction_bins(self, directions: Tensor, telescope_pointing: Tensor) -> Tensor:
        """Assign direction vectors to angular bins.
        
        Args:
            directions: True direction vectors (batch_size, 3)
            telescope_pointing: Telescope pointing directions (batch_size, 3)
            
        Returns:
            Tensor: Bin assignments (batch_size,) with values 0 to num_fine_bins
                   (num_fine_bins = outside ROI)
        """
        batch_size = directions.shape[0]
        
        # Convert 3D directions to angular offsets from telescope pointing
        # Compute angular separation using dot product
        cos_sep = torch.sum(directions * telescope_pointing, dim=1)
        angular_sep = torch.acos(torch.clamp(cos_sep, -1.0, 1.0))
        
        # For simplicity, assign to nearest bin center
        # This is a placeholder - in practice you'd want more sophisticated assignment
        bin_assignments = torch.full((batch_size,), self.num_fine_bins, dtype=torch.long)
        
        # Find events within ROI
        within_roi = angular_sep <= self.roi_radius_rad
        
        # For events within ROI, assign to closest bin
        if torch.any(within_roi):
            # Convert to theta/phi offsets (simplified approach)
            roi_events = directions[within_roi]
            roi_pointing = telescope_pointing[within_roi]
            
            # Compute angular distances to all bin centers
            # This is a simplified implementation - would need proper spherical geometry
            if len(self.bin_centers) > 0:
                # Assign to bin 0 for now (would need proper spherical distance calculation)
                bin_assignments[within_roi] = 0
        
        return bin_assignments


