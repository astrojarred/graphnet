"""Cleaning utilities for MAGIC MC parquet events."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from graphnet.data.extractors.magic.calibration import (
    VALID_TIMECAL_CENTERING,
    TimecalLookup,
    center_timecal_per_pixel,
    graft_mc_telescope_signal,
)

# Effective default MC-graft timeslice duration (ns) when nothing is passed.
MC_GRAFT_TIMESLICE_DURATION_DEFAULT = 0.6


def resolve_mc_graft_duration(
    mc_graft_timeslice_duration: Optional[float] = None,
    graft_timeslice_ns: Optional[float] = None,
) -> float:
    """Resolve the MC-graft timeslice duration, honoring the deprecated alias.

    ``graft_timeslice_ns`` is the deprecated name for
    ``mc_graft_timeslice_duration``. Passing it emits a ``DeprecationWarning`` and
    maps onto the new name. Passing BOTH with different values raises
    ``ValueError``. When neither is given, the effective default
    (:data:`MC_GRAFT_TIMESLICE_DURATION_DEFAULT`, 0.6) is returned.
    """
    if graft_timeslice_ns is not None:
        warnings.warn(
            "graft_timeslice_ns is deprecated; use mc_graft_timeslice_duration "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if (
            mc_graft_timeslice_duration is not None
            and float(mc_graft_timeslice_duration) != float(graft_timeslice_ns)
        ):
            raise ValueError(
                "Conflicting timeslice-duration arguments: "
                f"mc_graft_timeslice_duration={mc_graft_timeslice_duration} vs "
                f"deprecated graft_timeslice_ns={graft_timeslice_ns}; pass only one."
            )
        return float(graft_timeslice_ns)
    if mc_graft_timeslice_duration is None:
        return MC_GRAFT_TIMESLICE_DURATION_DEFAULT
    return float(mc_graft_timeslice_duration)


DEFAULT_PIXEL_CACHE_PATH = (
    Path(__file__).resolve().parent / "magic_default_pixel_coordinates.json"
)
DEFAULT_GEOMETRY_PATH = Path(__file__).resolve().parent / "geometry.json"

DEFAULT_N_PIXELS = 1039
DEFAULT_N_TIMESLICES = 50

def threshold_cleaning(
    signal: np.ndarray,
    med: float = 0,
    mad: float = 0.16,
    n_low: float = 6.0,
) -> np.ndarray:
    """Clean signal using a lower threshold (p.e. scale for calibrated waveforms)."""
    pe_threshold = med + n_low * mad
    return signal > pe_threshold


def _expand_timecal_1d(
    tcal_per_pixel: np.ndarray,
    n_timeslices: int,
    timeslice_duration: float = 1.0,
) -> np.ndarray:
    """Expand per-pixel start times to one value per sample (pixel-major, time fastest)."""
    tcal = np.asarray(tcal_per_pixel, dtype=np.float32).ravel()
    n_pixels = len(tcal)
    time_offsets = np.arange(n_timeslices, dtype=np.float32) * np.float32(
        timeslice_duration
    )
    return np.repeat(tcal, n_timeslices) + np.tile(time_offsets, n_pixels)


def _legacy_stereo_time_flat(
    timecal_flat: np.ndarray,
    tel_idx: int,
    n_pixels: int,
    n_timeslices: int,
) -> np.ndarray:
    """Match legacy (2, n_ts, n_pix)[tel].T flattened order."""
    tc = np.asarray(timecal_flat, dtype=np.float32).reshape(
        2, n_timeslices, n_pixels
    )
    return tc[tel_idx].T.reshape(-1)


def _broadcast_pixel_valid(
    pixel_valid: np.ndarray, n_pixels: int, n_timeslices: int
) -> np.ndarray:
    """Broadcast per-pixel flags to per-sample mask matching waveform flatten order."""
    pv = np.asarray(pixel_valid, dtype=bool).ravel()
    if pv.size == n_pixels:
        return np.repeat(pv, n_timeslices)
    if pv.size == n_pixels * n_timeslices:
        return pv
    raise ValueError(
        f"pixel_valid has length {pv.size}, expected {n_pixels} or "
        f"{n_pixels * n_timeslices}"
    )


def _split_stereo_waveforms(
    waveforms: np.ndarray, n_pixels: int, n_timeslices: int
) -> List[np.ndarray]:
    """Split flat stereo buffer [M1 || M2] into two per-telescope arrays."""
    per = n_pixels * n_timeslices
    wf = np.asarray(waveforms, dtype=np.float32).ravel()
    if wf.size != 2 * per:
        raise ValueError(
            f"Expected stereo waveforms length {2 * per}, got {wf.size}"
        )
    return [wf[:per], wf[per:]]


def _telescope_id_from_number(telescope_number: Any) -> int:
    """Map MAGIC telescope_number (1/2) to internal tel_id (0/1)."""
    try:
        n = int(telescope_number)
    except (TypeError, ValueError):
        return 0
    if n == 2:
        return 1
    return 0


def _time_for_stereo_row(
    row: pd.Series,
    n_pixels: int,
    n_timeslices: int,
    real_timecal_centering: str = "none",
    real_timeslice_duration: float = 1.0,
) -> List[np.ndarray]:
    """Build per-telescope time arrays (length ``n_pixels * n_timeslices`` each).

    Handles v5 exports:
    - MC: no timecal (constant per MC); use zero baseline expansion.
    - Real: ``timecal_M1`` / ``timecal_M2`` length ``n_pixels``.
    - Legacy: full ``timecal`` length ``2 * n_ts * n_pix`` stereo layout.

    ``real_timecal_centering`` and ``real_timeslice_duration`` are the orthogonal
    timing-origin and timing-stride controls; they apply to the real
    ``timecal_M1`` / ``timecal_M2`` path only. The legacy ``timecal`` and MC
    placeholder paths keep their historical (uncentered, unit-stride) expansion.
    """
    zeros = np.zeros(n_pixels, dtype=np.float32)

    if "timecal_M1" in row.index and row.get("timecal_M1") is not None:
        t1 = center_timecal_per_pixel(
            row["timecal_M1"],
            n_pixels=n_pixels,
            centering=real_timecal_centering,
        )
        t2 = (
            center_timecal_per_pixel(
                row["timecal_M2"],
                n_pixels=n_pixels,
                centering=real_timecal_centering,
            )
            if row.get("timecal_M2") is not None
            else zeros
        )
        return [
            _expand_timecal_1d(t1, n_timeslices, real_timeslice_duration),
            _expand_timecal_1d(t2, n_timeslices, real_timeslice_duration),
        ]

    if "timecal" in row.index and row.get("timecal") is not None:
        tc = np.asarray(row["timecal"], dtype=np.float32).ravel()
        if tc.size == 2 * n_timeslices * n_pixels:
            return [
                _legacy_stereo_time_flat(tc, 0, n_pixels, n_timeslices),
                _legacy_stereo_time_flat(tc, 1, n_pixels, n_timeslices),
            ]
        if tc.size == 2 * n_pixels:
            return [
                _expand_timecal_1d(tc[:n_pixels], n_timeslices),
                _expand_timecal_1d(tc[n_pixels:], n_timeslices),
            ]
        if tc.size == n_pixels:
            return [
                _expand_timecal_1d(tc, n_timeslices),
                _expand_timecal_1d(zeros, n_timeslices),
            ]

    # MC or missing timecal: placeholder times (same expansion as dl0 for uniform MC)
    zt = _expand_timecal_1d(zeros, n_timeslices)
    return [zt, zt]


def _pixel_valid_stereo(
    row: pd.Series,
    n_pixels: int,
    n_timeslices: int,
) -> List[Optional[np.ndarray]]:
    """Per-telescope pixel_valid broadcast, or split stereo ``pixel_valid``."""
    if row.get("pixel_valid_M1") is not None or row.get("pixel_valid_M2") is not None:
        pvm: List[Optional[np.ndarray]] = [None, None]
        if row.get("pixel_valid_M1") is not None:
            pvm[0] = _broadcast_pixel_valid(
                row["pixel_valid_M1"], n_pixels, n_timeslices
            )
        if row.get("pixel_valid_M2") is not None:
            pvm[1] = _broadcast_pixel_valid(
                row["pixel_valid_M2"], n_pixels, n_timeslices
            )
        return pvm

    pv = row.get("pixel_valid")
    if pv is None:
        return [None, None]
    arr = np.asarray(pv, dtype=np.float32).ravel()
    per = n_pixels * n_timeslices
    if arr.size == 2 * per:
        return [
            _broadcast_pixel_valid(arr[:per], n_pixels, n_timeslices),
            _broadcast_pixel_valid(arr[per:], n_pixels, n_timeslices),
        ]
    if arr.size == 2 * n_pixels:
        return [
            _broadcast_pixel_valid(arr[:n_pixels], n_pixels, n_timeslices),
            _broadcast_pixel_valid(arr[n_pixels:], n_pixels, n_timeslices),
        ]
    if arr.size == n_pixels:
        b = _broadcast_pixel_valid(arr, n_pixels, n_timeslices)
        return [b, b]
    if arr.size == per:
        return [_broadcast_pixel_valid(arr, n_pixels, n_timeslices), None]

    raise ValueError(f"Unsupported pixel_valid length {arr.size}")


def _build_default_px_py(
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
    n_pixels: int = DEFAULT_N_PIXELS,
    n_timeslices: int = DEFAULT_N_TIMESLICES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build camera x/y coordinates for all pixel-time bins."""
    with geometry_path.open("r", encoding="utf-8") as f:
        geometry = json.load(f)

    pixel_coords = np.array(
        [[geometry[str(p)]["x"], geometry[str(p)]["y"]] for p in range(n_pixels)],
        dtype=np.float32,
    )
    px = np.repeat(pixel_coords[:, 0], n_timeslices).reshape(
        n_pixels, n_timeslices
    ).flatten()
    py = np.repeat(pixel_coords[:, 1], n_timeslices).reshape(
        n_pixels, n_timeslices
    ).flatten()
    return px, py


def _save_default_px_py_json(
    px: np.ndarray,
    py: np.ndarray,
    path: Path = DEFAULT_PIXEL_CACHE_PATH,
) -> None:
    payload = {
        "px": px.astype(float).tolist(),
        "py": py.astype(float).tolist(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_or_build_default_px_py(
    path: Path = DEFAULT_PIXEL_CACHE_PATH,
    n_pixels: int = DEFAULT_N_PIXELS,
    n_timeslices: int = DEFAULT_N_TIMESLICES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load default px/py from JSON cache, or build and save."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        px = np.asarray(payload["px"], dtype=np.float32)
        py = np.asarray(payload["py"], dtype=np.float32)
        expected = n_pixels * n_timeslices
        if px.size == expected:
            return px, py

    px, py = _build_default_px_py(
        n_pixels=n_pixels, n_timeslices=n_timeslices
    )
    _save_default_px_py_json(px=px, py=py, path=path)
    return px.astype(np.float32), py.astype(np.float32)


def log_size_clipped_from_row(row: pd.Series) -> float:
    """Return log10(sum_i max(p_i, 0)) over the raw pulse train, or nan if unavailable.

    Used to drop corrupt real-data flashes (typically ``log_size_clipped > 4.75``).
    Prefers column ``waveforms`` if present, otherwise ``waveforms_pe``.
    """
    wf_name = "waveforms" if "waveforms" in row.index else "waveforms_pe"
    if wf_name not in row.index:
        return float("nan")
    wf = np.asarray(row[wf_name], dtype=np.float64).ravel()
    if wf.size == 0:
        return float("nan")
    s_clipped = float(np.clip(wf, 0.0, None).sum())
    if s_clipped <= 0.0:
        return float("nan")
    return float(np.log10(s_clipped))


def clean_magic_event(
    row: pd.Series,
    apply_cleaning: bool = False,
    cleaning_n_low: float | None = None,
    px: Optional[np.ndarray] = None,
    py: Optional[np.ndarray] = None,
    index_column: Optional[str] = "event_id",
    global_params: Optional[List[str]] = None,
    truth_columns: Optional[List[str]] = None,
    graft_lookup: Optional[TimecalLookup] = None,
    mc_graft_timeslice_duration: Optional[float] = None,
    real_timecal_centering: str = "none",
    real_timeslice_duration: float = 1.0,
    graft_log_interpolation: bool = False,
    allow_missing_truth_global_columns: bool = False,
    graft_timeslice_ns: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert one MAGIC parquet row into a cleaned event dictionary.

    Three orthogonal timing controls are threaded through this function:

    - ``real_timecal_centering`` (``"none"`` | ``"per_telescope_mean"``, default
      ``"none"``): timing ORIGIN for the real ``timecal_M1`` / ``timecal_M2``
      path. ``"per_telescope_mean"`` subtracts the per-telescope arithmetic mean
      over all pixels (computed BEFORE cleaning/masking).
    - ``real_timeslice_duration`` (default ``1.0``): timing STRIDE (ns) between
      consecutive samples on the real path.
    - ``mc_graft_timeslice_duration`` (default ``0.6``): timing STRIDE (ns) used
      when grafting. ``graft_timeslice_ns`` is a deprecated alias for it.

    The defaults reproduce the legacy behaviour bit-for-bit. The MC graft is
    always mean-centered (canonical) regardless of ``real_timecal_centering``,
    which controls the real path only.

    When ``graft_lookup`` is set, real per-pixel timecal from the LMDB is grafted
    onto each telescope waveform (:func:`graft_mc_telescope_signal`) **before**
    threshold cleaning and ``pixel_valid`` masking. The stored ``time`` channel
    is the shifted graft grid, not parquet ``timecal``.

    The effective timing settings are returned under the ``"timing_settings"``
    key so a later conversion-manifest feature can record them.

    When ``allow_missing_truth_global_columns`` is True, any name in
    ``truth_columns`` or ``global_params`` that is absent from ``row`` is skipped
    and a warning is logged (useful for real data with optional or varying
    exports). When False (default), a missing name raises ``KeyError``.
    """

    if real_timecal_centering not in VALID_TIMECAL_CENTERING:
        raise ValueError(
            f"unknown real_timecal_centering {real_timecal_centering!r}; "
            f"expected one of {VALID_TIMECAL_CENTERING}"
        )
    mc_graft_duration = resolve_mc_graft_duration(
        mc_graft_timeslice_duration, graft_timeslice_ns
    )
    timing_settings = {
        "real_timecal_centering": real_timecal_centering,
        "real_timeslice_duration": float(real_timeslice_duration),
        "mc_graft_timeslice_duration": float(mc_graft_duration),
        "grafting": graft_lookup is not None,
    }

    n_pixels = int(row["n_pixels"]) if "n_pixels" in row.index and pd.notna(row.get("n_pixels")) else DEFAULT_N_PIXELS
    n_timeslices = (
        int(row["n_timeslices"])
        if "n_timeslices" in row.index and pd.notna(row.get("n_timeslices"))
        else DEFAULT_N_TIMESLICES
    )
    per_telescope = n_pixels * n_timeslices

    if px is None or py is None:
        default_px, default_py = load_or_build_default_px_py(
            n_pixels=n_pixels, n_timeslices=n_timeslices
        )
        px = default_px if px is None else px
        py = default_py if py is None else py

    wf = np.asarray(row["waveforms"], dtype=np.float32).ravel()

    signal_parts: List[np.ndarray] = []
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    tel_parts: List[np.ndarray] = []

    truth: Dict[str, object] = {}
    if truth_columns is not None:
        for truth_key in truth_columns:
            if truth_key not in row:
                if allow_missing_truth_global_columns:
                    _logger.warning(
                        "MAGIC clean_magic_event: truth column %r missing from row; "
                        "skipping.",
                        truth_key,
                    )
                    continue
                raise KeyError(f"Column '{truth_key}' was not found in input row.")
            truth[truth_key] = row[truth_key]

    globals_dict: Dict[str, object] = {}
    if global_params is not None:
        for param in global_params:
            if param not in row:
                if allow_missing_truth_global_columns:
                    _logger.warning(
                        "MAGIC clean_magic_event: global_params column %r missing from "
                        "row; skipping.",
                        param,
                    )
                    continue
                raise KeyError(f"Column '{param}' was not found in input row.")
            globals_dict[param] = row[param]

    event_id: object = -1
    if index_column is not None and index_column in row:
        event_id = row[index_column]

    graft_packed: tuple[np.ndarray, np.ndarray] | None = None
    if graft_lookup is not None:
        graft_packed = graft_lookup[int(event_id)]
        if graft_packed is None:
            raise ValueError(
                f"timecal graft: no LMDB entry for event_id={event_id!r} "
                f"(after lookup modulo / mod_shift)."
            )

    n_low = 6.0 if cleaning_n_low is None else cleaning_n_low

    def process_telescope(
        signal: np.ndarray,
        time: np.ndarray,
        tel_idx: int,
        pixel_valid_sample_mask: Optional[np.ndarray],
        graft_real_per_pixel: Optional[np.ndarray] = None,
    ) -> None:
        signal_2d = signal.reshape(n_pixels, n_timeslices)
        if graft_real_per_pixel is not None:
            signal_2d, shifted_time_2d = graft_mc_telescope_signal(
                signal_2d,
                graft_real_per_pixel,
                timeslice_ns=mc_graft_duration,
                log_interpolation=graft_log_interpolation,
            )
            signal = signal_2d.reshape(-1)
            time = shifted_time_2d.reshape(-1)
        else:
            signal = signal_2d.reshape(-1)
            time = time.reshape(-1)
        if signal.shape[0] != time.shape[0]:
            raise ValueError(
                f"signal/time length mismatch: {signal.shape[0]} vs {time.shape[0]}"
            )
        if apply_cleaning:
            keep = threshold_cleaning(signal, n_low=n_low)
        else:
            keep = np.ones(len(signal), dtype=bool)
        if pixel_valid_sample_mask is not None:
            pv = np.asarray(pixel_valid_sample_mask, dtype=bool).ravel()
            if pv.shape[0] != keep.shape[0]:
                raise ValueError(
                    f"pixel_valid length {pv.shape[0]} != signal length {keep.shape[0]}"
                )
            keep = keep & pv

        signal_parts.append(signal[keep])
        x_parts.append(px[keep])
        y_parts.append(py[keep])
        time_parts.append(time[keep])
        tel_parts.append(np.repeat(tel_idx, int(np.sum(keep))))
        globals_dict[f"size_M{tel_idx + 1}"] = float(np.sum(signal))

    if wf.size == per_telescope:
        tel_num = row.get("telescope_number")
        tel_idx = _telescope_id_from_number(tel_num)
        if "timecal_M1" in row.index and row.get("timecal_M1") is not None:
            tcal_1d = center_timecal_per_pixel(
                row["timecal_M1"],
                n_pixels=n_pixels,
                centering=real_timecal_centering,
            )
            single_tel_duration = real_timeslice_duration
        elif "timecal" in row.index and row.get("timecal") is not None:
            # Legacy timecal export: unchanged (uncentered, unit-stride).
            tcal_1d = np.asarray(row["timecal"], dtype=np.float32).ravel()
            single_tel_duration = 1.0
        else:
            tcal_1d = np.zeros(n_pixels, dtype=np.float32)
            single_tel_duration = 1.0
        time_flat = _expand_timecal_1d(tcal_1d, n_timeslices, single_tel_duration)
        pv_tel: Optional[np.ndarray] = None
        if row.get("pixel_valid_M1") is not None:
            pv_tel = _broadcast_pixel_valid(
                row["pixel_valid_M1"], n_pixels, n_timeslices
            )
        elif row.get("pixel_valid") is not None:
            pv_tel = _broadcast_pixel_valid(
                row["pixel_valid"], n_pixels, n_timeslices
            )
        graft_tcal = (
            graft_packed[tel_idx] if graft_packed is not None else None
        )
        process_telescope(wf, time_flat, tel_idx, pv_tel, graft_tcal)

    elif wf.size == 2 * per_telescope:
        sig_tels = _split_stereo_waveforms(wf, n_pixels, n_timeslices)
        time_tels = _time_for_stereo_row(
            row,
            n_pixels,
            n_timeslices,
            real_timecal_centering=real_timecal_centering,
            real_timeslice_duration=real_timeslice_duration,
        )
        pv_tels = _pixel_valid_stereo(row, n_pixels, n_timeslices)
        for tel_i in range(2):
            graft_tcal = (
                graft_packed[tel_i] if graft_packed is not None else None
            )
            process_telescope(
                sig_tels[tel_i],
                time_tels[tel_i],
                tel_i,
                pv_tels[tel_i],
                graft_tcal,
            )

    else:
        raise ValueError(
            f"Unsupported waveforms length {wf.size} "
            f"(expected {per_telescope} or {2 * per_telescope})"
        )

    if signal_parts:
        signal_clean = np.concatenate(signal_parts)
        x_clean = np.concatenate(x_parts)
        y_clean = np.concatenate(y_parts)
        time_clean = np.concatenate(time_parts)
        tel_clean = np.concatenate(tel_parts)
    else:
        signal_clean = np.array([], dtype=np.float32)
        x_clean = np.array([], dtype=np.float32)
        y_clean = np.array([], dtype=np.float32)
        time_clean = np.array([], dtype=np.float32)
        tel_clean = np.array([], dtype=np.float32)

    return {
        "signal": signal_clean,
        "x_cam": x_clean,
        "y_cam": y_clean,
        "time": time_clean,
        "tel_id": tel_clean,
        "cleaning_n_low": cleaning_n_low,
        "event_id": event_id,
        "truth": truth,
        "global_params": globals_dict,
        "timing_settings": timing_settings,
    }
