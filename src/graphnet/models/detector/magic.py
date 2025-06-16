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
