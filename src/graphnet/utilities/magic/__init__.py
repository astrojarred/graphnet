"""Utilities for MAGIC telescope coordinate transforms.

See :mod:`graphnet.utilities.magic.coordinates` for the public API.
"""

from .coordinates import (
    StarCamTrans,
    angular_separation_deg,
    camera_prediction_to_radec,
    cel0_cam_to_cel,
    cel0_cel_to_cam,
    cel_to_loc,
    loc0_cam_to_loc,
    loc0_loc_to_cam,
    loc_to_cel,
    normalize_hour_angle,
    ra_hours_from_lst_and_ha,
    hour_angle_hours_from_lst_and_ra,
    wrap24,
    wrap360,
)
from .metrics import (
    camera_pair_angular_separation_deg,
    containment_summary,
    decode_clear_rain_norm_to_mm,
    decode_v9_norm_to_mm,
    local_angular_separation_deg,
)

__all__ = [
    "StarCamTrans",
    "angular_separation_deg",
    "camera_pair_angular_separation_deg",
    "camera_prediction_to_radec",
    "cel0_cam_to_cel",
    "cel0_cel_to_cam",
    "cel_to_loc",
    "containment_summary",
    "decode_clear_rain_norm_to_mm",
    "decode_v9_norm_to_mm",
    "loc0_cam_to_loc",
    "loc0_loc_to_cam",
    "local_angular_separation_deg",
    "loc_to_cel",
    "normalize_hour_angle",
    "ra_hours_from_lst_and_ha",
    "hour_angle_hours_from_lst_and_ra",
    "wrap24",
    "wrap360",
]
