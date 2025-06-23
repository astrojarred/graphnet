"""MAGIC-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector

MAGIC_GEOMETRY_PATH = (
    "/home/iwsatlas1/jgreen/Documents/graphnet/notebooks/magic_geometry.parquet"
)


class MAGICDetector(Detector):
    """`Detector` class for the MAGIC telescopes."""

    geometry_table_path = MAGIC_GEOMETRY_PATH

    xy = ["x_cam", "y_cam"]
    xyz = ["x_cam", "y_cam", "t"]
    telescope_id_column = "tel_id"
    global_features = ["telescope_phi", "telescope_theta"]

    # placeholder names
    string_id_column = "tel_id"
    sensor_id_column = "pixel_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "x_cam": self._xy,
            "y_cam": self._xy,
            "t": self._t,
            "tel_id": self._identity,
            "signal": self._signal,
            "telescope_phi": self._telescope_phi,
            "telescope_theta": self._telescope_theta,
        }

        return feature_map

    def _xy(self, x: torch.tensor) -> torch.tensor:
        return x / 28.5

    def _t(self, x: torch.tensor) -> torch.tensor:
        """Can adjust the time scaling factor to determine how bound space-time become"""
        t_min = -30
        t_max = 60
        factor = 0.11
        return ((x - t_min) / (t_max - t_min)) / factor
    
    def _signal(self, x: torch.tensor) -> torch.tensor:
        asinh_scale = 0.1
        signal_mean = 0.025
        signal_var = 2.376
        signal_asinh = torch.asinh(x / asinh_scale)
        return (signal_asinh - signal_mean) / signal_var

    def _telescope_phi(self, x: torch.tensor) -> torch.tensor:
        return 1 - torch.cos(x)
    
    def _telescope_theta(self, x: torch.tensor) -> torch.tensor:
        return x / (2 * torch.pi)


class MAGICDetectorFixed(Detector):
    """`Detector` class for MAGIC telescopes with IceCube-inspired preprocessing.
    
    Based on winning IceCube Kaggle solutions. Uses simplified preprocessing:
    - Time: (t - t_min) / (t_max - t_min) with factor = 1.0
    - Signal: log10(signal + 1e-6) instead of asinh transform
    - Simplified telescope angle handling
    """

    geometry_table_path = MAGIC_GEOMETRY_PATH

    xy = ["x_cam", "y_cam"]
    xyz = ["x_cam", "y_cam", "t"]
    telescope_id_column = "tel_id"
    global_features = ["telescope_phi", "telescope_theta"]

    # placeholder names
    string_id_column = "tel_id"
    sensor_id_column = "pixel_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "x_cam": self._xy,
            "y_cam": self._xy,
            "t": self._t_fixed,
            "tel_id": self._identity,
            "signal": self._signal_fixed,
            "telescope_phi": self._telescope_phi_fixed,
            "telescope_theta": self._telescope_theta_fixed,
        }
        return feature_map

    def _xy(self, x: torch.tensor) -> torch.tensor:
        """Keep same spatial normalization as original."""
        return x / 28.5

    def _t_fixed(self, x: torch.tensor) -> torch.tensor:
        """Simplified time scaling based on IceCube winners.
        
        Uses factor = 1.0 to put time on equal footing with spatial dimensions.
        This is critical for proper graph construction and attention mechanisms.
        """
        t_min = -30
        t_max = 60
        # factor = 1.0 instead of 0.11 - this is the key change!
        return (x - t_min) / (t_max - t_min)
    
    def _signal_fixed(self, x: torch.tensor) -> torch.tensor:
        """Simplified signal preprocessing based on IceCube winners.
        
        Uses log10 transform instead of asinh, which is simpler and proven
        to work well for Cherenkov light signals.
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        return torch.log10(x + epsilon)

    def _telescope_phi_fixed(self, x: torch.tensor) -> torch.tensor:
        """Simplified telescope phi preprocessing."""
        # Direct normalization instead of 1 - cos(x) transform
        return x / (2 * torch.pi)
    
    def _telescope_theta_fixed(self, x: torch.tensor) -> torch.tensor:
        """Simplified telescope theta preprocessing."""
        # Same as original but explicit about the normalization
        return x / (2 * torch.pi)
