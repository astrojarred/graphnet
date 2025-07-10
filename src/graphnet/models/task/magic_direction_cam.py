from __future__ import annotations

from typing import List, Union, Optional, Callable, ClassVar

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data

from graphnet.training.loss_functions import LossFunction
from graphnet.models.task import StandardLearnedTask
# from graphnet.models.task.task import LearnedTask
from graphnet.utilities.maths import eps_like
from graphnet.training.labels import Label

__all__ = [
    "MAGICEuclideanDistanceLoss", 
    "EnergyWeightedEuclideanDistanceLoss", 
    "TrueSourceCameraPosition",
    "CameraPlaneReconstructionWithUncertainty",
    "CameraPlaneReconstruction",
    "sky_to_camera_wrapper",
    "camera_to_sky_wrapper",
]

class TrueSourceCameraPosition(Label):
    """Produces a 2-component target tensor in relative camera plane coordinates:
        [camera_x, camera_y]

    Args:
        telescope_phi_key: Key for telescope phi
        telescope_theta_key: Key for telescope theta
        true_phi_key: Key for true phi
        true_theta_key: Key for true theta
        is_mc: Whether the data is Monte Carlo
        max_angle_deg: Maximum angle in degrees
        dist_cam: Distance between camera and mirror plane [m]

    Default dist_cam is based on the maximum FoV diameter of max_angle_deg = 2.5 degrees, 
        calculated like so: 1 / torch.tan(torch.deg2rad(max_angle_deg))
    """
    def __init__(
        self, 
        telescope_theta_key: str = "telescope_theta", 
        telescope_phi_key: str = "telescope_phi", 
        true_theta_key: str = "true_theta",
        true_phi_key: str = "true_phi",
        is_mc: bool = True,
        max_angle_deg: int | float | None = 2.5,
        dist_cam: int | float | torch.Tensor | None = None,
        key: str = "true_source_camera_position"
    ):
        super().__init__(key=key)
        self.telescope_theta_key = telescope_theta_key
        self.telescope_phi_key = telescope_phi_key
        self.true_theta_key = true_theta_key
        self.true_phi_key = true_phi_key

        if not dist_cam:
            max_angle_rad = torch.deg2rad(torch.tensor(max_angle_deg))
            max_XC = torch.tan(max_angle_rad)  # Maximum XC value (dimensionless)
            dist_cam = 1 / max_XC
        
        self.is_mc = is_mc
        self.max_angle_deg = max_angle_deg
        self.dist_cam = dist_cam

    def __call__(self, graph: Data) -> torch.Tensor:
        telescope_theta = graph[self.telescope_theta_key]
        telescope_phi = graph[self.telescope_phi_key]
        true_theta = graph[self.true_theta_key]
        true_phi = graph[self.true_phi_key]

        dist_cam = torch.as_tensor(
            self.dist_cam,
            device=telescope_phi.device,
            dtype=telescope_phi.dtype,
        )

        source_cam_x, source_cam_y = sky_to_camera_wrapper(
            telescope_theta, telescope_phi,
            true_theta, true_phi,
            use_monte_carlo=self.is_mc,  # Set based on your data type
            dist_cam=dist_cam
        )
        
        return torch.stack((source_cam_x, source_cam_y), dim=1)

class MAGICEuclideanDistanceLoss(LossFunction):
    """Euclidean distance loss."""
    def __init__(self):
        super().__init__()
    
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.linalg.vector_norm(prediction - target, dim=1)

class EnergyWeightedEuclideanDistanceLoss(LossFunction):
    """Energy-weighted Euclidean distance loss.

    Default parameters give weights from 0.5 (10 GeV) to 2.5 (30 TeV).

    prediction: [batch_size, 2] (x, y)
    target: [batch_size, 3] (x, y, energy_gev)
    """
    def __init__(self, energy_scale=100, energy_weight_scale=0.575, energy_weight_offset=1.075):
        super().__init__()
        self.energy_scale = energy_scale
        self.energy_weight_scale = energy_weight_scale
        self.energy_weight_offset = energy_weight_offset
    
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        target_coords = target[:, :2]
        target_energy = target[:, 2]

        # (log10(30000/100) * 0.575) + 1.075
        energy_weight = (torch.log10(torch.clamp(target_energy, min=1e-3) / self.energy_scale) * self.energy_weight_scale) + self.energy_weight_offset

        # Clamp to prevent NaN
        energy_weight = torch.clamp(energy_weight, 0.1, 10.0)

        return torch.linalg.vector_norm(prediction - target_coords, dim=1) * energy_weight

class CameraPlaneReconstructionWithUncertainty(StandardLearnedTask):
    """Camera plane direction reconstruction with uncertainty estimates.
    
    Works with TrueSourceCameraPosition label that produces a 2-component tensor.
    
    Reconstructs source position in camera plane coordinates with:
    - Coordinates clamped to [-1.5, 1.5] range (provides buffer beyond ±2.5° FoV)
    - Position uncertainty estimates for quality assessment
    - Camera center at (0, 0) with ±1.0 corresponding to ±2.5°
    
    Target format:
    - true_source_camera_position: [batch_size, 2] (x, y)
    - true_energy: [batch_size, 1] (true_energy)
    - Combined target passed to loss: [batch_size, 3] (x, y, true_energy)
    
    Designed for use with EnergyWeightedEuclideanDistanceLoss:
    - prediction: [batch_size, 2] (x, y) - only coordinates used in loss
    - target: [batch_size, 3] (x, y, true_energy) - automatically concatenated by GraphNeT
    """

    # Target uses the new 2-component tensor format
    default_target_labels: ClassVar[List[str]] = ["true_source_camera_position", "true_energy"]
    default_prediction_labels: ClassVar[List[str]] = [
        "camera_x_pred", 
        "camera_y_pred",
        "camera_x_sigma",
        "camera_y_sigma"
    ]
    nb_inputs: ClassVar[int] = 4  # x, y, sigma_x, sigma_y

    def __init__(
        self, 
        hidden_size: int,
        target_labels: Union[str, List[str]] = ["true_source_camera_position", "true_energy"],
        prediction_labels: Optional[List[str]] = None,
        loss_function: Optional[LossFunction] = None,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
        coord_range: float = 1.5, 
        uncertainty_scale: float = 0.1
    ):
        """Initialize camera plane reconstruction task.
        
        Args:
            coord_range: Maximum coordinate value (coordinates clamped to [-coord_range, +coord_range])
            uncertainty_scale: Scaling factor for uncertainty estimates
        """
        super().__init__(
            hidden_size=hidden_size,
            target_labels=target_labels,
            prediction_labels=prediction_labels,
            loss_function=loss_function,
            transform_prediction_and_target=transform_prediction_and_target,
            transform_target=transform_target,
            transform_inference=transform_inference,
        )
        self.coord_range = coord_range
        self.uncertainty_scale = uncertainty_scale

    def _forward(self, x: Tensor) -> Tensor:
        """Transform network outputs to camera coordinates with uncertainties.
        
        Args:
            x: Raw network outputs [batch_size, 4] (x, y, sigma_x, sigma_y)
            
        Returns:
            [batch_size, 4] (x_clamped, y_clamped, sigma_x, sigma_y)
            
        Note: 
            Loss function will only use first 2 columns (x, y).
            GraphNeT automatically concatenates target labels into [batch_size, 3] format:
            [camera_x, camera_y, true_energy] for the loss function.
        """
        # Extract coordinates and uncertainties
        raw_x = x[:, 0]
        raw_y = x[:, 1] 
        raw_sigma_x = x[:, 2]
        raw_sigma_y = x[:, 3]
        
        # Smoothly squash coordinates into the valid camera range instead of hard‐clamping.
        # Using tanh keeps non-zero gradients even when the raw network output is far
        # outside the valid interval, preventing the model from getting “stuck” with
        # saturated outputs.  We rescale tanh() so that ±1 maps exactly to
        # ±self.coord_range (i.e. ±1.5 by default).
        camera_x = torch.tanh(raw_x / self.coord_range) * self.coord_range
        camera_y = torch.tanh(raw_y / self.coord_range) * self.coord_range
        
        # Transform uncertainties to positive values
        # Use softplus for smooth, positive transformation
        sigma_x = F.softplus(raw_sigma_x) * self.uncertainty_scale + eps_like(raw_sigma_x)
        sigma_y = F.softplus(raw_sigma_y) * self.uncertainty_scale + eps_like(raw_sigma_y)
        
        return torch.stack((camera_x, camera_y, sigma_x, sigma_y), dim=1)

    def get_coordinates_only(self, x: Tensor) -> Tensor:
        """Extract only the coordinate predictions for loss computation.
        
        Args:
            x: Network outputs [batch_size, 4]
            
        Returns:
            [batch_size, 2] camera coordinates only
        """
        full_output = self._forward(x)
        return full_output[:, :2]  # Return only x, y coordinates

    def get_uncertainties_only(self, x: Tensor) -> Tensor:
        """Extract only the uncertainty estimates.
        
        Args:
            x: Network outputs [batch_size, 4]
            
        Returns:
            [batch_size, 2] uncertainty estimates (sigma_x, sigma_y)
        """
        full_output = self._forward(x)
        return full_output[:, 2:]  # Return only sigma_x, sigma_y

    def coordinate_range_info(self) -> dict:
        """Return information about the coordinate system."""
        return {
            "coord_range": f"[-{self.coord_range}, +{self.coord_range}]",
            "nominal_fov": "±1.0 corresponds to ±2.5°",
            "extended_range": f"±{self.coord_range} corresponds to ±{2.5 * self.coord_range}°",
            "center": "(0, 0) = camera center",
            "uncertainty_scale": self.uncertainty_scale
        }

    def compute_loss(self, pred: Tensor, data: Data) -> Tensor:
        # Build coordinates-only prediction
        coords_pred = self.get_coordinates_only(pred)
        # Extract 2D target and energy, ensure energy is [batch,1]
        target_coords = data[self.default_target_labels[0]]
        energy = data[self.default_target_labels[1]]
        if energy.dim() == 1:
            energy = energy.unsqueeze(1)
        # Concatenate into [batch,3]
        target = torch.cat([target_coords, energy], dim=1)
        # Handle optional loss weight
        weights = data[self._loss_weight] if self._loss_weight is not None else None
        # Compute and return loss
        loss = self._loss_function(coords_pred, target, weights=weights) + self._regularisation_loss
        return loss


class CameraPlaneReconstruction(StandardLearnedTask):
    """Simplified camera plane reconstruction without uncertainties.
    
    Works with TrueSourceCameraPosition label that produces a 2-component tensor.
    
    Use this if you want to test performance without uncertainty estimation.
    
    Target format:
    - true_source_camera_position: [batch_size, 2] (x, y)
    - true_energy: [batch_size, 1] (true_energy)
    - Combined target passed to loss: [batch_size, 3] (x, y, true_energy)
    """

    # Updated to use the new 2-component tensor format
    default_target_labels: ClassVar[List[str]] = ["true_source_camera_position", "true_energy"]
    default_prediction_labels: ClassVar[List[str]] = ["camera_x_pred", "camera_y_pred"]
    nb_inputs: ClassVar[int] = 2

    def __init__(
        self, 
        hidden_size: int,
        target_labels: Union[str, List[str]] = ["true_source_camera_position", "true_energy"],
        prediction_labels: Optional[List[str]] = None,
        loss_function: Optional[LossFunction] = None,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
        coord_range: float = 1.5
    ):
        """Initialize camera plane reconstruction task.
        
        Args:
            coord_range: Maximum coordinate value (coordinates clamped to [-coord_range, +coord_range])
        """
        super().__init__(
            hidden_size=hidden_size,
            target_labels=target_labels,
            prediction_labels=prediction_labels,
            loss_function=loss_function,
            transform_prediction_and_target=transform_prediction_and_target,
            transform_target=transform_target,
            transform_inference=transform_inference,
        )
        self.coord_range = coord_range

    def _forward(self, x: Tensor) -> Tensor:
        """Transform to clamped camera coordinates.
        
        Args:
            x: Raw network outputs [batch_size, 2] (x, y)
            
        Returns:
            [batch_size, 2] clamped camera coordinates
            
        Note:
            GraphNeT automatically concatenates target labels into [batch_size, 3] format:
            [camera_x, camera_y, true_energy] for the loss function.
        """
        # Extract coordinates and uncertainties
        raw_x = x[:, 0]
        raw_y = x[:, 1]

        # Same smooth squashing as above to retain gradients for out-of-range values
        camera_x = torch.tanh(raw_x / self.coord_range) * self.coord_range
        camera_y = torch.tanh(raw_y / self.coord_range) * self.coord_range
        
        return torch.stack((camera_x, camera_y), dim=1)

    def coordinate_range_info(self) -> dict:
        """Return information about the coordinate system."""
        return {
            "coord_range": f"[-{self.coord_range}, +{self.coord_range}]",
            "nominal_fov": "±1.0 corresponds to ±2.5°",
            "extended_range": f"±{self.coord_range} corresponds to ±{2.5 * self.coord_range}°",
            "center": "(0, 0) = camera center",
        }

    def compute_loss(self, pred: Tensor, data: Data) -> Tensor:
        # Prediction is already [batch,2]
        coords_pred = pred
        # Extract 2D target and energy, ensure energy is [batch,1]
        target_coords = data[self.default_target_labels[0]]
        energy = data[self.default_target_labels[1]]
        if energy.dim() == 1:
            energy = energy.unsqueeze(1)
        # Concatenate into [batch,3]
        target = torch.cat([target_coords, energy], dim=1)
        # Handle optional loss weight
        weights = data[self._loss_weight] if self._loss_weight is not None else None
        # Compute and return loss
        loss = self._loss_function(coords_pred, target, weights=weights) + self._regularisation_loss
        return loss


# coordinate transformation functions
def sky_to_camera_coordinates_magic(
    telescope_theta_deg: torch.Tensor,
    telescope_phi_deg: torch.Tensor,
    source_theta_deg: torch.Tensor,
    source_phi_deg: torch.Tensor,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sky coordinates to camera coordinates using MAGIC's exact implementation.

    Based on MStarCamTrans::Loc0LocToCam from MAGIC analysis pipeline.

    Parameters:
    -----------
    telescope_theta_deg : float or array
        Telescope pointing zenith distance [degrees]
    telescope_phi_deg : float or array
        Telescope pointing azimuth [degrees]
    source_theta_deg : float or array
        Source zenith distance [degrees]
    source_phi_deg : float or array
        Source azimuth [degrees]
    dist_cam : float
        Camera distance/focal length scaling factor (default=1.0 for angular units)

    Returns:
    --------
    x_cam, y_cam : float or array
        Camera coordinates in units determined by dist_cam

    Notes:
    ------
    - Exact implementation of MAGIC's MStarCamTrans::Loc0LocToCam
    - Uses MAGIC's coordinate conventions and sign conventions
    - For Monte Carlo data, phi angles are transformed as (180° - phi)
    """

    # Convert to radians (MAGIC code divides by kRad2Deg which is 180/π)
    theta0_rad = torch.deg2rad(telescope_theta_deg)
    phi0_rad = torch.deg2rad(telescope_phi_deg)
    theta_rad = torch.deg2rad(source_theta_deg)
    phi_rad = torch.deg2rad(source_phi_deg)

    # Calculate trigonometric functions
    sintheta0 = torch.sin(theta0_rad)
    costheta0 = torch.cos(theta0_rad)
    sintheta = torch.sin(theta_rad)
    costheta = torch.cos(theta_rad)

    # Angular difference in azimuth
    phi_diff = phi_rad - phi0_rad

    # MAGIC's exact formulation from MStarCamTrans::Loc0LocToCam
    denominator = costheta0 * costheta + sintheta0 * sintheta * torch.cos(phi_diff)

    # Check for division by zero (source behind camera)
    if torch.any(denominator <= 0):
        raise ValueError("Source is behind the camera (denominator <= 0)")

    XC = -sintheta * torch.sin(phi_diff) / denominator
    YC = (-sintheta0 * costheta + costheta0 * sintheta * torch.cos(phi_diff)) / denominator

    # Apply camera distance scaling
    X = XC * dist_cam
    Y = YC * dist_cam

    return X, Y


def sky_to_camera_coordinates_magic_mc(
    telescope_theta_deg: torch.Tensor,
    telescope_phi_deg: torch.Tensor,
    source_theta_deg: torch.Tensor,
    source_phi_deg: torch.Tensor,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sky coordinates to camera coordinates for Monte Carlo data.

    Based on the Monte Carlo branch in MSrcPosCalc::Process() which applies
    the 180° - phi transformation.

    Parameters:
    -----------
    telescope_theta_deg : float or array
        Telescope pointing zenith distance [degrees]
    telescope_phi_deg : float or array
        Telescope pointing azimuth [degrees]
    source_theta_deg : float or array
        Source zenith distance [degrees]
    source_phi_deg : float or array
        Source azimuth [degrees]
    dist_cam : float
        Camera distance/focal length scaling factor

    Returns:
    --------
    x_cam, y_cam : float or array
        Camera coordinates
    """

    # Apply Monte Carlo coordinate transformation (180° - phi)
    telescope_phi_transformed = 180.0 - telescope_phi_deg
    source_phi_transformed = 180.0 - source_phi_deg

    return sky_to_camera_coordinates_magic(
        telescope_theta_deg,
        telescope_phi_transformed,
        source_theta_deg,
        source_phi_transformed,
        dist_cam,
    )


def sky_to_camera_coordinates_magic_alt(
    telescope_theta_deg: torch.Tensor,
    telescope_phi_deg: torch.Tensor,
    source_theta_deg: torch.Tensor,
    source_phi_deg: torch.Tensor,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Alternative MAGIC implementation using MSrcPosCalc::CalcXYinCamera formulation.

    This uses a different mathematical approach but should give equivalent results.

    Parameters:
    -----------
    telescope_theta_deg : float or array
        Telescope pointing zenith distance [degrees]
    telescope_phi_deg : float or array
        Telescope pointing azimuth [degrees]
    source_theta_deg : float or array
        Source zenith distance [degrees]
    source_phi_deg : float or array
        Source azimuth [degrees]
    dist_cam : float
        Camera distance/focal length scaling factor

    Returns:
    --------
    x_cam, y_cam : float or array
        Camera coordinates
    """

    # Convert to radians
    theta0 = torch.deg2rad(telescope_theta_deg)
    phi0 = torch.deg2rad(telescope_phi_deg)
    theta = torch.deg2rad(source_theta_deg)
    phi = torch.deg2rad(source_phi_deg)

    # MSrcPosCalc::CalcXYinCamera implementation
    phi_diff = phi - phi0

    # Y coordinate calculation
    YC0 = torch.cos(theta0) * torch.tan(theta) * torch.cos(phi_diff) - torch.sin(theta0)
    YC1 = torch.cos(theta0) + torch.sin(theta0) * torch.tan(theta)

    # Check for division by zero
    if torch.any(YC1 == 0):
        raise ValueError("Division by zero in Y coordinate calculation")

    YC = YC0 / YC1

    # X coordinate calculation
    XC0 = torch.cos(theta0) - YC * torch.sin(theta0)
    XC = -torch.sin(phi_diff) * torch.tan(theta) * XC0

    # Apply camera distance scaling
    X = XC * dist_cam
    Y = YC * dist_cam

    return X, Y


def convert_radians_to_magic_input(
    telescope_theta_rad: torch.Tensor,
    telescope_phi_rad: torch.Tensor,
    source_theta_rad: torch.Tensor,
    source_phi_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert input from radians (as in your original question) to the format
    expected by MAGIC functions.

    Parameters:
    -----------
    telescope_theta_rad : float or array
        Telescope pointing zenith distance [radians]
    telescope_phi_rad : float or array
        Telescope pointing azimuth [radians]
    source_theta_rad : float or array
        Source zenith distance [radians]
    source_phi_rad : float or array
        Source azimuth [radians]

    Returns:
    --------
    telescope_theta_deg, telescope_phi_deg, source_theta_deg, source_phi_deg
        All converted to degrees for MAGIC functions
    """

    return (
        torch.rad2deg(telescope_theta_rad),
        torch.rad2deg(telescope_phi_rad),
        torch.rad2deg(source_theta_rad),
        torch.rad2deg(source_phi_rad),
    )


# Wrapper function to match your original question format
def sky_to_camera_wrapper(
    telescope_theta_rad: torch.Tensor,
    telescope_phi_rad: torch.Tensor,
    source_theta_rad: torch.Tensor,
    source_phi_rad: torch.Tensor,
    use_monte_carlo: bool = False,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function matching your original parameter format.

    Parameters:
    -----------
    telescope_theta_rad : float or array
        Telescope pointing zenith distance [radians]
    telescope_phi_rad : float or array
        Telescope pointing azimuth [radians]
    source_theta_rad : float or array
        Source zenith distance [radians]
    source_phi_rad : float or array
        Source azimuth [radians]
    use_monte_carlo : bool
        Whether to apply Monte Carlo coordinate transformation
    dist_cam : float
        Camera scaling factor

    Returns:
    --------
    x_cam, y_cam : float or array
        Camera coordinates
    """

    # Convert to degrees
    tel_theta_deg, tel_phi_deg, src_theta_deg, src_phi_deg = (
        convert_radians_to_magic_input(
            telescope_theta_rad, telescope_phi_rad, source_theta_rad, source_phi_rad
        )
    )

    # Choose appropriate function
    if use_monte_carlo:
        return sky_to_camera_coordinates_magic_mc(
            tel_theta_deg, tel_phi_deg, src_theta_deg, src_phi_deg, dist_cam
        )
    else:
        return sky_to_camera_coordinates_magic(
            tel_theta_deg, tel_phi_deg, src_theta_deg, src_phi_deg, dist_cam
        )


def camera_to_sky_coordinates_magic(
    telescope_theta_deg: torch.Tensor,
    telescope_phi_deg: torch.Tensor,
    x_cam: torch.Tensor,
    y_cam: torch.Tensor,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert camera coordinates back to sky coordinates (inverse of MAGIC transformation).

    Uses spherical trigonometry approach for robust inversion.

    Parameters:
    -----------
    telescope_theta_deg : float or array
        Telescope pointing zenith distance [degrees]
    telescope_phi_deg : float or array
        Telescope pointing azimuth [degrees]
    x_cam : float or array
        Camera X coordinate (in units determined by dist_cam)
    y_cam : float or array
        Camera Y coordinate (in units determined by dist_cam)
    dist_cam : float
        Camera distance/focal length scaling factor (same as used in forward transform)

    Returns:
    --------
    source_theta_deg, source_phi_deg : float or array
        Source zenith distance and azimuth [degrees]
    """

    # Convert camera coordinates back to angular coordinates
    XC = x_cam / dist_cam
    YC = y_cam / dist_cam

    # Convert telescope pointing to radians
    theta0_rad = torch.deg2rad(telescope_theta_deg)
    phi0_rad = torch.deg2rad(telescope_phi_deg)

    # Calculate angular distance from camera center
    r_rad = torch.sqrt(XC**2 + YC**2)

    # Handle case where source is exactly at telescope pointing
    if torch.any(r_rad == 0):
        # Source is at telescope pointing
        return telescope_theta_deg, telescope_phi_deg

    # Calculate position angle in camera (measured from +X axis, counterclockwise)
    # In MAGIC convention, +X is typically East, +Y is North
    # PA_cam = torch.arctan2(YC, XC)

    # For spherical trigonometry, we need to relate camera coordinates to sky coordinates
    # The camera coordinate system needs to be related to the sky coordinate system

    # Use the gnomonic projection inverse formula
    # In gnomonic projection: XC = tan(offset) * sin(PA), YC = tan(offset) * cos(PA)
    # where offset is angular distance from pointing center, PA is position angle

    # But MAGIC uses a specific coordinate system, so let's solve it iteratively
    # using Newton-Raphson to find theta and phi that produce the given XC, YC

    # Initial guess: assume small angle approximation
    # For small angles: XC ≈ -(phi - phi0) * sin(theta0), YC ≈ theta - theta0
    initial_theta = theta0_rad + YC
    initial_phi = torch.where(
        torch.sin(theta0_rad) > 1e-10, phi0_rad - XC / torch.sin(theta0_rad), phi0_rad
    )

    # Newton-Raphson iteration
    theta = initial_theta
    phi = initial_phi

    for _ in range(10):  # Usually converges in 2-3 iterations
        # Calculate forward transformation with current guess
        sintheta0 = torch.sin(theta0_rad)
        costheta0 = torch.cos(theta0_rad)
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        phi_diff = phi - phi0_rad

        denominator = costheta0 * costheta + sintheta0 * sintheta * torch.cos(phi_diff)

        # Avoid division by zero
        denominator = torch.where(torch.abs(denominator) < 1e-15, 1e-15, denominator)

        XC_calc = -sintheta * torch.sin(phi_diff) / denominator
        YC_calc = (
            -sintheta0 * costheta + costheta0 * sintheta * torch.cos(phi_diff)
        ) / denominator

        # Calculate residuals
        f_theta = YC_calc - YC
        f_phi = XC_calc - XC

        # Check convergence
        if torch.all(torch.abs(f_theta) < 1e-10) and torch.all(torch.abs(f_phi) < 1e-10):
            break

        # Calculate Jacobian (partial derivatives)
        # df_theta/d_theta and df_phi/d_phi
        cos_phi_diff = torch.cos(phi_diff)
        sin_phi_diff = torch.sin(phi_diff)

        # Partial derivatives (simplified for numerical stability)
        # denom_sq = denominator**2

        # ∂YC/∂θ
        dYC_dtheta = (
            sintheta0 * sintheta + costheta0 * costheta * cos_phi_diff
        ) / denominator + YC_calc * (
            costheta0 * sintheta - sintheta0 * costheta * cos_phi_diff
        ) / denominator

        # ∂XC/∂φ
        dXC_dphi = (
            -sintheta * cos_phi_diff / denominator
            + XC_calc * sintheta0 * sintheta * sin_phi_diff / denominator
        )

        # ∂YC/∂φ
        dYC_dphi = (
            costheta0 * sintheta * sin_phi_diff / denominator
            + YC_calc * sintheta0 * sintheta * sin_phi_diff / denominator
        )

        # ∂XC/∂θ
        dXC_dtheta = (
            -costheta * sin_phi_diff / denominator
            + XC_calc
            * (costheta0 * sintheta - sintheta0 * costheta * cos_phi_diff)
            / denominator
        )

        # Jacobian matrix determinant
        det = dYC_dtheta * dXC_dphi - dYC_dphi * dXC_dtheta
        det = torch.where(torch.abs(det) < 1e-15, 1e-15, det)

        # Newton-Raphson update
        d_theta = (dXC_dphi * f_theta - dYC_dphi * f_phi) / det
        d_phi = (dYC_dtheta * f_phi - dXC_dtheta * f_theta) / det

        # Update with damping for stability
        damping = 0.5
        theta = theta - damping * d_theta
        phi = phi - damping * d_phi

        # Keep theta in valid range [0, π]
        theta = torch.clip(theta, 1e-10, torch.pi - 1e-10)

    # Ensure phi is in [0, 2π] range
    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)
    phi = torch.where(phi >= 2 * torch.pi, phi - 2 * torch.pi, phi)

    # Convert back to degrees
    source_theta_deg = torch.rad2deg(theta)
    source_phi_deg = torch.rad2deg(phi)

    return source_theta_deg, source_phi_deg


def camera_to_sky_coordinates_magic_mc(
    telescope_theta_deg: torch.Tensor,
    telescope_phi_deg: torch.Tensor,
    x_cam: torch.Tensor,
    y_cam: torch.Tensor,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert camera coordinates back to sky coordinates for Monte Carlo data.
    Applies the inverse of the (180° - phi) transformation.
    """

    # The forward MC transformation applies (180° - phi) to both telescope and source
    # So we need to use the transformed telescope pointing for the inverse
    telescope_phi_transformed = 180.0 - telescope_phi_deg

    # Apply the regular inverse transformation with transformed telescope pointing
    source_theta_transformed, source_phi_transformed = camera_to_sky_coordinates_magic(
        telescope_theta_deg, telescope_phi_transformed, x_cam, y_cam, dist_cam
    )

    # Apply inverse of the (180° - phi) transformation to get original source coordinates
    source_phi_deg = 180.0 - source_phi_transformed
    source_theta_deg = source_theta_transformed  # theta is not transformed

    # Ensure phi is in [0, 360] range
    source_phi_deg = torch.where(
        source_phi_deg < 0, source_phi_deg + 360.0, source_phi_deg
    )
    source_phi_deg = torch.where(
        source_phi_deg >= 360.0, source_phi_deg - 360.0, source_phi_deg
    )

    return source_theta_deg, source_phi_deg


def camera_to_sky_wrapper(
    telescope_theta_rad: torch.Tensor,
    telescope_phi_rad: torch.Tensor,
    x_cam: torch.Tensor,
    y_cam: torch.Tensor,
    use_monte_carlo: bool = False,
    dist_cam: torch.Tensor = torch.tensor(1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for camera-to-sky conversion matching your radian parameter format.

    Parameters:
    -----------
    telescope_theta_rad : float or array
        Telescope pointing zenith distance [radians]
    telescope_phi_rad : float or array
        Telescope pointing azimuth [radians]
    x_cam : float or array
        Camera X coordinate
    y_cam : float or array
        Camera Y coordinate
    use_monte_carlo : bool
        Whether to apply Monte Carlo coordinate transformation
    dist_cam : float
        Camera scaling factor (same as used in forward transform)

    Returns:
    --------
    source_theta_rad, source_phi_rad : float or array
        Source zenith distance and azimuth [radians]
    """

    # Convert telescope pointing to degrees
    tel_theta_deg = torch.rad2deg(telescope_theta_rad)
    tel_phi_deg = torch.rad2deg(telescope_phi_rad)

    # Apply appropriate inverse transformation
    if use_monte_carlo:
        src_theta_deg, src_phi_deg = camera_to_sky_coordinates_magic_mc(
            tel_theta_deg, tel_phi_deg, x_cam, y_cam, dist_cam
        )
    else:
        src_theta_deg, src_phi_deg = camera_to_sky_coordinates_magic(
            tel_theta_deg, tel_phi_deg, x_cam, y_cam, dist_cam
        )

    # Convert back to radians
    source_phi_rad = torch.deg2rad(src_phi_deg)
    source_theta_rad = torch.deg2rad(src_theta_deg)

    return source_theta_rad, source_phi_rad
