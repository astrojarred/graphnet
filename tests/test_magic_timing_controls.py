"""Tests for the orthogonal MAGIC timing controls.

Covers the three settings threaded from ``MAGICParquetReader`` through
``clean_magic_event`` / ``process_telescope`` down to the timecal expansion:

- ``real_timecal_centering`` ("none" | "per_telescope_mean", default "none")
- ``real_timeslice_duration`` (default 1.0)
- ``mc_graft_timeslice_duration`` (default 0.6; deprecated alias
  ``graft_timeslice_ns``)

All data are synthetic; no real files or LMDBs are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graphnet.data.extractors.magic.calibration import (
    center_timecal_per_pixel,
    graft_mc_telescope_signal,
    interp1d_shared_xp_batch,
)
from graphnet.data.extractors.magic.cleaning import (
    clean_magic_event,
    resolve_mc_graft_duration,
)
from graphnet.data.readers.magic_parquet_reader import MAGICParquetReader

N_PIX = 16
N_TS = 10
PER = N_PIX * N_TS


class FakeLookup:
    """Minimal stand-in for TimecalLookup: __getitem__ -> (timecal_M1, timecal_M2)."""

    def __init__(self, m1: np.ndarray, m2: np.ndarray) -> None:
        self._m1 = np.asarray(m1, dtype=np.float32)
        self._m2 = np.asarray(m2, dtype=np.float32)

    def __getitem__(self, index: int):
        return self._m1, self._m2


def _make_tcal(seed: int, n_pix: int = N_PIX) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n_pix) * 2.0 + 30.0).astype(np.float32)


def _make_row(
    tcal1: np.ndarray,
    tcal2: np.ndarray,
    seed: int = 0,
    **extra,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    wf = rng.random(2 * PER).astype(np.float32)
    data = {
        "event_id": 7,
        "n_pixels": N_PIX,
        "n_timeslices": N_TS,
        "waveforms": wf,
        "timecal_M1": tcal1,
        "timecal_M2": tcal2,
    }
    data.update(extra)
    return pd.Series(data)


def _clean(row: pd.Series, **kwargs) -> dict:
    px = np.arange(PER, dtype=np.float32)
    py = np.arange(PER, dtype=np.float32) * 2.0
    return clean_magic_event(
        row,
        px=px,
        py=py,
        truth_columns=[],
        global_params=[],
        **kwargs,
    )


def _legacy_real_time(tcal: np.ndarray, duration: float = 1.0) -> np.ndarray:
    """Pre-change real-path formula: repeat(tcal) + tile(arange*duration)."""
    tc = np.asarray(tcal, dtype=np.float32).ravel()
    off = np.arange(N_TS, dtype=np.float32) * np.float32(duration)
    return np.repeat(tc, N_TS) + np.tile(off, N_PIX)


def _legacy_graft_reference(
    sig2d: np.ndarray, tcal: np.ndarray, timeslice_ns: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-change MC graft formula, reproduced verbatim."""
    sig = np.asarray(sig2d, dtype=np.float32)
    n_pix, n_ts = sig.shape
    tc = np.asarray(tcal, dtype=np.float32).ravel()
    time_offsets = np.arange(n_ts, dtype=np.float32) * np.float32(timeslice_ns)
    mean = float(np.mean(tc))
    shifted = (tc[:, None] - np.float32(mean)) + time_offsets
    out = interp1d_shared_xp_batch(
        time_offsets.astype(np.float64),
        sig.astype(np.float64),
        shifted.astype(np.float64),
        left=0.0,
        right=0.0,
    )
    return out, shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Exact legacy reproduction with defaults
# ---------------------------------------------------------------------------


def test_defaults_reproduce_legacy_real_path_exactly() -> None:
    t1, t2 = _make_tcal(1), _make_tcal(2)
    out = _clean(_make_row(t1, t2))
    expected = np.concatenate([_legacy_real_time(t1), _legacy_real_time(t2)])
    np.testing.assert_array_equal(out["time"], expected)
    # Sanity: uncentered (mean of times reflects raw tcal mean, not ~0).
    assert abs(float(np.mean(out["time"]))) > 10.0


def test_defaults_reproduce_legacy_mc_graft_exactly() -> None:
    rng = np.random.default_rng(3)
    t1, t2 = _make_tcal(4), _make_tcal(5)
    row = _make_row(t1, t2, seed=3)
    out = _clean(row, graft_lookup=FakeLookup(t1, t2))

    wf = np.asarray(row["waveforms"], dtype=np.float32)
    expected_time, expected_sig = [], []
    for tel, tc in enumerate([t1, t2]):
        sig2d = wf[tel * PER : (tel + 1) * PER].reshape(N_PIX, N_TS)
        ref_sig, ref_shift = _legacy_graft_reference(sig2d, tc, 0.6)
        expected_sig.append(ref_sig.reshape(-1))
        expected_time.append(ref_shift.reshape(-1))
    np.testing.assert_array_equal(out["time"], np.concatenate(expected_time))
    np.testing.assert_array_equal(out["signal"], np.concatenate(expected_sig))


def test_graft_function_matches_legacy_formula_bit_exact() -> None:
    rng = np.random.default_rng(6)
    sig = rng.random((N_PIX, N_TS)).astype(np.float32)
    tc = _make_tcal(7)
    out, shifted = graft_mc_telescope_signal(sig, tc, timeslice_ns=0.6)
    ref_out, ref_shifted = _legacy_graft_reference(sig, tc, 0.6)
    np.testing.assert_array_equal(shifted, ref_shifted)
    np.testing.assert_array_equal(out, ref_out)


# ---------------------------------------------------------------------------
# 2. Origin-only control
# ---------------------------------------------------------------------------


def test_origin_only_centering_shifts_times_by_per_telescope_mean() -> None:
    t1, t2 = _make_tcal(8), _make_tcal(9)
    row = _make_row(t1, t2, seed=8)
    legacy = _clean(row)
    centered = _clean(row, real_timecal_centering="per_telescope_mean")

    # Exact formula: centered tcal expanded with unit stride.
    expected = np.concatenate(
        [
            _legacy_real_time(t1 - np.float32(float(np.mean(t1)))),
            _legacy_real_time(t2 - np.float32(float(np.mean(t2)))),
        ]
    )
    np.testing.assert_array_equal(centered["time"], expected)

    # Difference from legacy is a constant per telescope (the telescope mean).
    diff = legacy["time"] - centered["time"]
    d1, d2 = diff[:PER], diff[PER:]
    np.testing.assert_allclose(d1, float(np.mean(t1)), rtol=0, atol=1e-4)
    np.testing.assert_allclose(d2, float(np.mean(t2)), rtol=0, atol=1e-4)

    # Signals / positions / masks unchanged by the origin shift.
    np.testing.assert_array_equal(legacy["signal"], centered["signal"])
    np.testing.assert_array_equal(legacy["x_cam"], centered["x_cam"])
    np.testing.assert_array_equal(legacy["tel_id"], centered["tel_id"])


# ---------------------------------------------------------------------------
# 3. Stride-only control
# ---------------------------------------------------------------------------


def test_stride_only_rescales_sample_offsets() -> None:
    t1, t2 = _make_tcal(10), _make_tcal(11)
    row = _make_row(t1, t2, seed=10)
    out = _clean(row, real_timeslice_duration=0.6)
    expected = np.concatenate(
        [_legacy_real_time(t1, 0.6), _legacy_real_time(t2, 0.6)]
    )
    np.testing.assert_array_equal(out["time"], expected)

    # Signals untouched.
    legacy = _clean(row)
    np.testing.assert_array_equal(legacy["signal"], out["signal"])


# ---------------------------------------------------------------------------
# 4. Canonical settings: centered + 0.6, real path == graft grid
# ---------------------------------------------------------------------------


def test_canonical_real_path_matches_mc_graft_time_grid() -> None:
    t1, t2 = _make_tcal(12), _make_tcal(13)
    row = _make_row(t1, t2, seed=12)
    real = _clean(
        row,
        real_timecal_centering="per_telescope_mean",
        real_timeslice_duration=0.6,
    )
    rng = np.random.default_rng(0)
    sig = rng.random((N_PIX, N_TS)).astype(np.float32)
    _, shifted1 = graft_mc_telescope_signal(sig, t1, timeslice_ns=0.6)
    _, shifted2 = graft_mc_telescope_signal(sig, t2, timeslice_ns=0.6)
    expected = np.concatenate([shifted1.reshape(-1), shifted2.reshape(-1)])
    np.testing.assert_array_equal(real["time"], expected)


# ---------------------------------------------------------------------------
# 5. M1/M2 independence
# ---------------------------------------------------------------------------


def test_telescopes_centered_by_their_own_means() -> None:
    t1 = np.full(N_PIX, 5.0, dtype=np.float32)
    t2 = np.full(N_PIX, 105.0, dtype=np.float32)
    out = _clean(_make_row(t1, t2), real_timecal_centering="per_telescope_mean")
    m1_times, m2_times = out["time"][:PER], out["time"][PER:]
    # Constant per-pixel tcal centers to zero for each telescope independently;
    # only the sample offsets remain (identical for both telescopes).
    base = np.tile(np.arange(N_TS, dtype=np.float32), N_PIX)
    np.testing.assert_array_equal(m1_times, base)
    np.testing.assert_array_equal(m2_times, base)


# ---------------------------------------------------------------------------
# 6. Mean computed before masking
# ---------------------------------------------------------------------------


def test_centering_mean_includes_masked_pixels() -> None:
    # Pixel 0 has an extreme timecal and is flagged invalid; the mean must
    # still include it (mean over ALL pixels, before cleaning/masking).
    t1 = np.zeros(N_PIX, dtype=np.float32)
    t1[0] = 100.0
    t2 = np.zeros(N_PIX, dtype=np.float32)
    pv1 = np.ones(N_PIX, dtype=bool)
    pv1[0] = False
    row = _make_row(t1, t2, pixel_valid_M1=pv1, pixel_valid_M2=np.ones(N_PIX, bool))
    out = _clean(row, real_timecal_centering="per_telescope_mean")

    m1_times = out["time"][out["tel_id"] == 0]
    assert m1_times.size == (N_PIX - 1) * N_TS  # pixel 0 masked out

    full_mean = np.float32(float(np.mean(t1)))  # includes masked pixel: 6.25
    masked_mean = np.float32(float(np.mean(t1[1:])))  # would be 0.0
    assert float(full_mean) != float(masked_mean)

    # Kept pixels all have tcal == 0, so times == -full_mean + offsets.
    expected = _legacy_real_time(
        np.full(N_PIX, -full_mean, dtype=np.float32)
    )[N_TS:]  # drop pixel 0's samples
    np.testing.assert_array_equal(m1_times, expected)

    # And NOT the masked-mean variant.
    wrong = _legacy_real_time(np.full(N_PIX, -masked_mean, dtype=np.float32))[N_TS:]
    assert not np.array_equal(m1_times, wrong)


# ---------------------------------------------------------------------------
# 7. Malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("centering", ["none", "per_telescope_mean"])
def test_real_path_rejects_wrong_size_timecal(centering: str) -> None:
    t1 = _make_tcal(14)[: N_PIX - 1]  # wrong length
    t2 = _make_tcal(15)
    with pytest.raises(ValueError):
        _clean(_make_row(t1, t2), real_timecal_centering=centering)


@pytest.mark.parametrize("centering", ["none", "per_telescope_mean"])
def test_real_path_rejects_non_finite_timecal(centering: str) -> None:
    t1 = _make_tcal(16)
    t1[3] = np.nan
    t2 = _make_tcal(17)
    with pytest.raises(ValueError):
        _clean(_make_row(t1, t2), real_timecal_centering=centering)


def test_graft_path_rejects_wrong_size_and_non_finite() -> None:
    rng = np.random.default_rng(18)
    sig = rng.random((N_PIX, N_TS)).astype(np.float32)
    with pytest.raises(ValueError):
        graft_mc_telescope_signal(sig, np.zeros(N_PIX - 1, dtype=np.float32))
    bad = np.zeros(N_PIX, dtype=np.float32)
    bad[0] = np.inf
    with pytest.raises(ValueError):
        graft_mc_telescope_signal(sig, bad)


def test_invalid_centering_value_rejected() -> None:
    t1, t2 = _make_tcal(19), _make_tcal(20)
    with pytest.raises(ValueError):
        _clean(_make_row(t1, t2), real_timecal_centering="bogus")
    with pytest.raises(ValueError):
        center_timecal_per_pixel(t1, n_pixels=N_PIX, centering="bogus")
    with pytest.raises(ValueError):
        MAGICParquetReader(real_timecal_centering="bogus")


# ---------------------------------------------------------------------------
# 8. Deprecated alias graft_timeslice_ns
# ---------------------------------------------------------------------------


def test_deprecated_alias_resolves_and_warns() -> None:
    with pytest.warns(DeprecationWarning):
        assert resolve_mc_graft_duration(None, 0.5) == 0.5
    # No alias: new name or default.
    assert resolve_mc_graft_duration(None, None) == 0.6
    assert resolve_mc_graft_duration(0.7, None) == 0.7
    # Both with same value: allowed (warns).
    with pytest.warns(DeprecationWarning):
        assert resolve_mc_graft_duration(0.5, 0.5) == 0.5
    # Conflict: raises.
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            resolve_mc_graft_duration(0.7, 0.5)


def test_deprecated_alias_in_clean_magic_event() -> None:
    t1, t2 = _make_tcal(21), _make_tcal(22)
    row = _make_row(t1, t2, seed=21)
    lookup = FakeLookup(t1, t2)
    with pytest.warns(DeprecationWarning):
        aliased = _clean(row, graft_lookup=lookup, graft_timeslice_ns=0.5)
    modern = _clean(row, graft_lookup=lookup, mc_graft_timeslice_duration=0.5)
    np.testing.assert_array_equal(aliased["time"], modern["time"])
    np.testing.assert_array_equal(aliased["signal"], modern["signal"])
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            _clean(
                row,
                graft_lookup=lookup,
                mc_graft_timeslice_duration=0.7,
                graft_timeslice_ns=0.5,
            )


def test_deprecated_alias_in_reader() -> None:
    with pytest.warns(DeprecationWarning):
        reader = MAGICParquetReader(graft_timeslice_ns=0.5)
    assert reader.timing_settings["mc_graft_timeslice_duration"] == 0.5
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            MAGICParquetReader(
                mc_graft_timeslice_duration=0.7, graft_timeslice_ns=0.5
            )


# ---------------------------------------------------------------------------
# Timing-settings exposure (for the future conversion manifest)
# ---------------------------------------------------------------------------


def test_timing_settings_exposed() -> None:
    reader = MAGICParquetReader(
        real_timecal_centering="per_telescope_mean",
        real_timeslice_duration=0.6,
        mc_graft_timeslice_duration=0.6,
    )
    assert reader.timing_settings == {
        "real_timecal_centering": "per_telescope_mean",
        "real_timeslice_duration": 0.6,
        "mc_graft_timeslice_duration": 0.6,
    }

    t1, t2 = _make_tcal(23), _make_tcal(24)
    out = _clean(_make_row(t1, t2))
    assert out["timing_settings"] == {
        "real_timecal_centering": "none",
        "real_timeslice_duration": 1.0,
        "mc_graft_timeslice_duration": 0.6,
        "grafting": False,
    }
