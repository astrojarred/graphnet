"""Cleaning utilities for MAGIC MC parquet events."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


DEFAULT_PIXEL_CACHE_PATH = (
    Path(__file__).resolve().parent / "magic_default_pixel_coordinates.json"
)
DEFAULT_GEOMETRY_PATH = Path(__file__).resolve().parent / "geometry.json"


def simple_cleaning(
    signal: np.ndarray,
    n_nodes: int = 1024,
    frac_lowest: float = 0.1,
) -> np.ndarray:
    """Keep brightest nodes plus a small dimmest tail."""
    num_bright = n_nodes - int(n_nodes * frac_lowest)
    num_dim = int(n_nodes * frac_lowest)
    order = np.argsort(signal)[::-1]
    keep_indices = np.concatenate([order[:num_bright], order[-num_dim:]])
    keep = np.zeros(len(signal), dtype=bool)
    keep[keep_indices] = True
    return keep


def _build_default_px_py(
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build camera x/y coordinates for all pixel-time bins."""
    with geometry_path.open("r", encoding="utf-8") as f:
        geometry = json.load(f)

    pixel_coords = np.array(
        [[geometry[str(p)]["x"], geometry[str(p)]["y"]] for p in range(1039)],
        dtype=np.float32,
    )
    px = np.repeat(pixel_coords[:, 0], 50).reshape(1039, 50).flatten()
    py = np.repeat(pixel_coords[:, 1], 50).reshape(1039, 50).flatten()
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Load default px/py from JSON cache, or build and save."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return (
            np.asarray(payload["px"], dtype=np.float32),
            np.asarray(payload["py"], dtype=np.float32),
        )

    px, py = _build_default_px_py()
    _save_default_px_py_json(px=px, py=py, path=path)
    return px.astype(np.float32), py.astype(np.float32)


def clean_magic_event(
    row: pd.Series,
    n_nodes: int = 1024,
    frac_lowest: float = 0.1,
    apply_cleaning: bool = True,
    px: Optional[np.ndarray] = None,
    py: Optional[np.ndarray] = None,
    index_column: Optional[str] = "event_id",
    global_params: Optional[List[str]] = None,
    truth_columns: Optional[List[str]] = None,
    is_mc: bool = True,
) -> Dict[str, Any]:
    """Convert one MAGIC parquet row into a cleaned event dictionary."""
    del is_mc

    if px is None or py is None:
        default_px, default_py = load_or_build_default_px_py()
        px = default_px if px is None else px
        py = default_py if py is None else py

    signal_parts: List[np.ndarray] = []
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    tel_parts: List[np.ndarray] = []

    truth: Dict[str, object] = {}
    if truth_columns is not None:
        for truth_key in truth_columns:
            if truth_key not in row:
                raise KeyError(f"Column '{truth_key}' was not found in input row.")
            truth[truth_key] = row[truth_key]

    globals_dict: Dict[str, object] = {}
    if global_params is not None:
        for param in global_params:
            if param not in row:
                raise KeyError(f"Column '{param}' was not found in input row.")
            globals_dict[param] = row[param]

    event_id: object = -1
    if index_column is not None and index_column in row:
        event_id = row[index_column]

    for tel_idx in range(2):
        signal = row.waveforms.reshape(2, 51950)[tel_idx].reshape(1039, 50).reshape(-1)
        time = row.timecal.reshape(2, 50, 1039)[tel_idx].T.reshape(-1)
        if apply_cleaning:
            keep = simple_cleaning(signal, n_nodes=n_nodes, frac_lowest=frac_lowest)
        else:
            keep = np.ones(len(signal), dtype=bool)

        signal_parts.append(signal[keep])
        x_parts.append(px[keep])
        y_parts.append(py[keep])
        time_parts.append(time[keep])
        tel_parts.append(np.repeat(tel_idx, np.sum(keep)))
        globals_dict[f"size_M{tel_idx + 1}"] = np.sum(signal)

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
        "n_nodes": n_nodes,
        "frac_lowest": frac_lowest,
        "event_id": event_id,
        "truth": truth,
        "global_params": globals_dict,
    }
