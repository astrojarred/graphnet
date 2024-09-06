"""MAGIC-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector

MAGIC_GEOMETRY_PATH = "/home/iwsatlas1/jgreen/Documents/graphnet/notebooks/magic_geometry.parquet"

class MAGICDetector(Detector):
    """`Detector` class for the MAGIC telescopes."""

    geometry_table_path = MAGIC_GEOMETRY_PATH

    xyz = ["pixel_pos_x", "pixel_pos_y", "telescope_id"]
    string_id_column = "telescope_id"
    sensor_id_column = "pixel_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "pixel_pos_x": self._pixel_pos_xy,
            "pixel_pos_y": self._pixel_pos_xy,
            "telescope_number": self._telescope_number,
            "time_cal": self._time_cal,
            "ffcalib": self._ffcalib,
            "pedestal_cal": self._pedestal_cal,
            "signal": self._signal,
            "telescope_phi": self._telescope_phi,
            "telescope_theta": self._telescope_theta,
        }

        return feature_map

    def _pixel_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 30
    
    def _telescope_number(self, x: torch.tensor) -> torch.tensor:
        # Convert 1 to -sqrt(3) and +sqrt(3)
        if x == 1:
            return -0.05773503
        else:
            return 0.05773503
    
    def _time_cal(self, x: torch.tensor) -> torch.tensor:
        # range [0, 80] -> [-1, 1]
        return ( x - 40) / 40
    
    def _ffcalib(self, x: torch.tensor) -> torch.tensor:
        # range [0, 0.02] -> [-1, 1]
        return (x - 0.01) / 0.01
    
    def _pedestal_cal(self, x: torch.tensor) -> torch.tensor:
        # range [9985, 10000] -> [-1, 1]
        x_min = 9985
        x_max = 10000
        return 2 * ((x - x_min) / (x_max - x_min)) - 1
    
    def _signal(self, x: torch.tensor) -> torch.tensor:
        # range [0, 25000] -> [-1, 1]
        return (x - 12500) / 12500
    
    def _telescope_phi(self, x: torch.tensor) -> torch.tensor:
        # range [0, 2*pi] -> [-1, 1]
        pi = 3.1415926
        return (x - pi) / pi
    
    def _telescope_theta(self, x: torch.tensor) -> torch.tensor:
        # range [0, pi] -> [-1, 1]
        pi = 3.1415926
        return (x - pi/2) / (pi/2)
            