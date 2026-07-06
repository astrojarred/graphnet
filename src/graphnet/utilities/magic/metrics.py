"""Physical angular metrics for MAGIC camera-plane predictions.

All focal-distance / plate-scale conversions require an explicit
``camera_dist_mm`` (or ``camera_unit_mm`` for v9-style normalisation).
No default focal length is assumed anywhere in this module.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray

from graphnet.utilities.magic.coordinates import (
    FloatOrArray,
    angular_separation_deg,
    loc0_cam_to_loc,
)

__all__ = [
    "local_angular_separation_deg",
    "camera_pair_angular_separation_deg",
    "decode_v9_norm_to_mm",
    "decode_clear_rain_norm_to_mm",
    "containment_summary",
]


def local_angular_separation_deg(
    zd1_deg: FloatOrArray,
    az1_deg: FloatOrArray,
    zd2_deg: FloatOrArray,
    az2_deg: FloatOrArray,
) -> FloatOrArray:
    """Great-circle separation between two local horizon directions [deg].

    Local zenith distance ``zd`` and azimuth ``az`` (MARS convention:
    ``az = 0`` at north, ``90`` at east) are mapped to a spherical
    coordinate pair ``(az, 90 - zd)`` and passed to
    :func:`angular_separation_deg` as ``(longitude, latitude)`` in degrees.

    Args:
        zd1_deg: Zenith distance of direction 1, in degrees.
        az1_deg: Azimuth of direction 1, in degrees.
        zd2_deg: Zenith distance of direction 2, in degrees.
        az2_deg: Azimuth of direction 2, in degrees.

    Returns:
        Angular separation in degrees, vectorized over NumPy arrays.
    """
    elev1 = 90.0 - np.asarray(zd1_deg, dtype=float)
    elev2 = 90.0 - np.asarray(zd2_deg, dtype=float)
    return angular_separation_deg(
        np.asarray(az1_deg, dtype=float),
        elev1,
        np.asarray(az2_deg, dtype=float),
        elev2,
        ra_unit="deg",
    )


def camera_pair_angular_separation_deg(
    x1_mm: FloatOrArray,
    y1_mm: FloatOrArray,
    x2_mm: FloatOrArray,
    y2_mm: FloatOrArray,
    pointing_zd_deg: FloatOrArray,
    pointing_az_deg: FloatOrArray,
    camera_dist_mm: float,
) -> FloatOrArray:
    """Angular separation between two camera-plane points about a common pointing.

    Each ``(x, y)`` pair in mm is inverse-projected through
    :func:`loc0_cam_to_loc` about ``(pointing_zd_deg, pointing_az_deg)``,
    then the great-circle separation between the two recovered local
    directions is returned.

    Args:
        x1_mm: Camera X of point 1, in mm.
        y1_mm: Camera Y of point 1, in mm.
        x2_mm: Camera X of point 2, in mm.
        y2_mm: Camera Y of point 2, in mm.
        pointing_zd_deg: Telescope pointing zenith distance, in degrees.
        pointing_az_deg: Telescope pointing azimuth, in degrees.
        camera_dist_mm: Camera-to-reflector distance in mm (required, no default).

    Returns:
        Angular separation in degrees, vectorized over NumPy arrays.

    Raises:
        TypeError: If ``camera_dist_mm`` is ``None``.
    """
    if camera_dist_mm is None:
        raise TypeError(
            "camera_dist_mm is required; pass the MARS-exported focal distance "
            "explicitly (e.g. 17000 mm for MAGIC II)."
        )
    dist = np.asarray(camera_dist_mm, dtype=float)
    zd1, az1 = loc0_cam_to_loc(
        pointing_zd_deg, pointing_az_deg, x1_mm, y1_mm, dist
    )
    zd2, az2 = loc0_cam_to_loc(
        pointing_zd_deg, pointing_az_deg, x2_mm, y2_mm, dist
    )
    return local_angular_separation_deg(zd1, az1, zd2, az2)


def decode_v9_norm_to_mm(
    xy_norm: FloatOrArray,
    camera_unit_mm: float | None = None,
) -> FloatOrArray:
    """Decode v5/v8/v9 camera normalisation ``camera_mm / camera_unit_mm``.

    DeepIce v9 exports and targets use ``CAMERA_UNIT_MM = 400`` mm per
    normalised unit. The caller must pass ``400.0`` explicitly.

    Args:
        xy_norm: Normalised camera coordinate(s).
        camera_unit_mm: Millimetres per normalised unit (pass ``400.0`` for v9).

    Returns:
        Camera coordinate(s) in mm.

    Raises:
        ValueError: If ``camera_unit_mm`` is ``None``.
    """
    if camera_unit_mm is None:
        raise ValueError(
            "camera_unit_mm is required; pass 400.0 explicitly for v9-style "
            "normalisation (camera_mm / 400)."
        )
    return np.asarray(xy_norm, dtype=float) * float(camera_unit_mm)


def decode_clear_rain_norm_to_mm(
    xy_norm: FloatOrArray,
    camera_dist_mm: float,
    max_angle_deg: float,
) -> FloatOrArray:
    """Decode clear-rain-48 camera normalisation to focal-plane mm.

    The clear-rain direction head stores camera offsets as

    .. code-block:: text

        norm = tan(theta_offset) / tan(max_angle_deg)

    per axis in the tangent plane (``max_angle_deg = 2.5`` in the champion
    run). The deleted ``TrueSourceCameraPosition`` label used
    ``dist_cam = 1 / tan(radians(2.5))`` so that unit norm at the edge of
    the training disk corresponds to ``max_angle_deg`` off axis. Inverting,

    .. code-block:: text

        theta_offset = arctan(norm * tan(max_angle_deg))
        x_mm = tan(theta_x) * camera_dist_mm = norm * tan(max_angle_deg) * camera_dist_mm

    (and likewise for ``y``).

    Args:
        xy_norm: Normalised clear-rain camera coordinate(s).
        camera_dist_mm: Camera-to-reflector distance in mm (required).
        max_angle_deg: Half-field angle used during clear-rain training (2.5).

    Returns:
        Camera coordinate(s) in mm.

    Raises:
        TypeError: If ``camera_dist_mm`` is ``None``.
    """
    if camera_dist_mm is None:
        raise TypeError("camera_dist_mm is required for clear-rain decoding.")
    dist = np.asarray(camera_dist_mm, dtype=float)
    scale = np.tan(np.deg2rad(float(max_angle_deg))) * dist
    return np.asarray(xy_norm, dtype=float) * scale


def containment_summary(errors_deg: Union[FloatOrArray, NDArray[np.floating]]) -> Dict[str, Any]:
    """Summary statistics for an angular-error sample [deg].

    Args:
        errors_deg: Per-event angular errors in degrees (finite values only
            contribute to the statistics).

    Returns:
        Dict with keys ``mean``, ``median``, ``p68``, ``p95``, and ``n``.
    """
    arr = np.asarray(errors_deg, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p68": float("nan"),
            "p95": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p68": float(np.quantile(finite, 0.68)),
        "p95": float(np.quantile(finite, 0.95)),
        "n": int(finite.size),
    }
