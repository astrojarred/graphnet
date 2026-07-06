"""Unit tests for graphnet.utilities.magic.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from graphnet.utilities.magic import coordinates as gc
from graphnet.utilities.magic import metrics as mm

CAMERA_DIST_MM = 17000.0
CAMERA_UNIT_MM = 400.0
CLEAR_RAIN_MAX_ANGLE_DEG = 2.5


def test_camera_pair_matches_loc0_loc_to_cam_offset() -> None:
    """0.4 deg offset from pointing round-trips through camera_pair."""
    pointing_zd = 30.0
    pointing_az = 180.0
    offset_deg = 0.4
    src_zd = pointing_zd + offset_deg
    src_az = pointing_az

    x_mm, y_mm = gc.loc0_loc_to_cam(
        pointing_zd,
        pointing_az,
        src_zd,
        src_az,
        CAMERA_DIST_MM,
    )
    sep = mm.camera_pair_angular_separation_deg(
        x_mm,
        y_mm,
        0.0,
        0.0,
        pointing_zd,
        pointing_az,
        CAMERA_DIST_MM,
    )
    assert sep == pytest.approx(offset_deg, abs=1e-9)


def test_camera_pair_vectorized() -> None:
    pointing_zd = np.array([20.0, 30.0, 40.0])
    pointing_az = np.array([90.0, 180.0, 270.0])
    offsets = np.array([0.2, 0.4, 0.6])

    x_mm = np.zeros(3)
    y_mm = np.zeros(3)
    for i in range(3):
        x_mm[i], y_mm[i] = gc.loc0_loc_to_cam(
            pointing_zd[i],
            pointing_az[i],
            pointing_zd[i] + offsets[i],
            pointing_az[i],
            CAMERA_DIST_MM,
        )

    sep = mm.camera_pair_angular_separation_deg(
        x_mm,
        y_mm,
        0.0,
        0.0,
        pointing_zd,
        pointing_az,
        CAMERA_DIST_MM,
    )
    np.testing.assert_allclose(sep, offsets, atol=1e-8)


def test_decode_v9_norm_to_mm_roundtrip() -> None:
    xy_mm = np.array([120.0, -80.0])
    xy_norm = xy_mm / CAMERA_UNIT_MM
    decoded = mm.decode_v9_norm_to_mm(xy_norm, camera_unit_mm=CAMERA_UNIT_MM)
    np.testing.assert_allclose(decoded, xy_mm)


def test_decode_v9_norm_requires_camera_unit() -> None:
    with pytest.raises(ValueError, match="camera_unit_mm"):
        mm.decode_v9_norm_to_mm(1.0, camera_unit_mm=None)


def test_decode_clear_rain_norm_to_mm_roundtrip() -> None:
    xy_norm = np.array([0.5, -0.25])
    xy_mm = mm.decode_clear_rain_norm_to_mm(
        xy_norm, CAMERA_DIST_MM, CLEAR_RAIN_MAX_ANGLE_DEG
    )
    expected = (
        xy_norm * np.tan(np.deg2rad(CLEAR_RAIN_MAX_ANGLE_DEG)) * CAMERA_DIST_MM
    )
    np.testing.assert_allclose(xy_mm, expected)


def test_decode_clear_rain_requires_camera_dist() -> None:
    with pytest.raises(TypeError, match="camera_dist_mm"):
        mm.decode_clear_rain_norm_to_mm(1.0, None, CLEAR_RAIN_MAX_ANGLE_DEG)


def test_camera_pair_requires_camera_dist() -> None:
    with pytest.raises(TypeError, match="camera_dist_mm"):
        mm.camera_pair_angular_separation_deg(0, 0, 1, 1, 30, 180, None)


def test_containment_summary_known_array() -> None:
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    summary = mm.containment_summary(errors)
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["median"] == pytest.approx(0.3)
    assert summary["p68"] == pytest.approx(np.quantile(errors, 0.68))
    assert summary["p95"] == pytest.approx(np.quantile(errors, 0.95))


def test_containment_summary_empty() -> None:
    summary = mm.containment_summary(np.array([np.nan, np.inf]))
    assert summary["n"] == 0
    assert np.isnan(summary["mean"])
