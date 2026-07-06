"""MAGIC pipeline audit: LMDB provenance, features, graphs, and integrity checks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch

from graphnet.data.dataset.dataset import Dataset
from graphnet.data.dataset.lmdb.magic_lmdb_dataset import MAGICLMDBDataset
from graphnet.data.utilities.magic_manifest import (
    manifest_paths_for_dataset,
    read_conversion_manifest,
)
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.models.detector.magic import MAGIC
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.utilities.config import DatasetConfig
from graphnet.utilities.config.graph_resolution import (
    resolve_dataset_graph_from_model,
)
from graphnet.utilities.magic.coordinates import camera_prediction_to_radec

AUDIT_VERSION = 1
MAGIC_NODE_FEATURES = ["signal", "x_cam", "y_cam", "time", "tel_id"]
QUANTILE_LEVELS = (0.01, 0.25, 0.5, 0.75, 0.99)
QUANTILE_LABELS = ("p01", "p25", "p50", "p75", "p99")
# La Palma observatory latitude (degrees); matches coordinate unit tests.
MAGIC_LATITUDE_DEG = 28.76
COS_LAT = float(np.cos(np.deg2rad(MAGIC_LATITUDE_DEG)))
SIN_LAT = float(np.sin(np.deg2rad(MAGIC_LATITUDE_DEG)))
V6_POINTING_COLUMNS = (
    "pointing_corr_dec_deg",
    "pointing_corr_ha_hours",
    "local_sidereal_time_hours",
    "camera_dist_mm",
)
PLOT_SEED = 42
PLOT_FIGSIZE = (8.0, 5.0)
SKY_SAMPLE_EVENTS = 5

__all__ = ["run_audit", "main"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            .stdout.strip()
        )
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _default_magic_graph() -> KNNGraph:
    return KNNGraph(
        detector=MAGIC(use_signal_epsilon=True),
        node_definition=NodesAsPulses(),
        input_feature_names=list(MAGIC_NODE_FEATURES),
        dtype=torch.float32,
        nb_nearest_neighbours=4,
        columns=[1, 2, 4, 3],
    )


def _coerce_dataset(
    loaded: Union[Dataset, Any],
) -> Tuple[Dataset, Optional[str]]:
    """Return a single `Dataset` and optional note when routing was needed."""
    from graphnet.data.dataset.dataset import EnsembleDataset

    if isinstance(loaded, dict):
        values = list(loaded.values())
        if len(values) == 1:
            return values[0], "used sole entry from selection dict"
        merged = Dataset.concatenate(values)
        return merged, f"concatenated {len(values)} selection splits"
    if isinstance(loaded, EnsembleDataset):
        return loaded, "using EnsembleDataset as loaded"
    return loaded, None


def _load_dataset(
    *,
    dataset_config: Optional[str],
    lmdb_path: Optional[str],
) -> Tuple[Dataset, Optional[DatasetConfig], str, Dict[str, Any]]:
    """Load a MAGIC LMDB dataset from config and/or direct path."""
    meta: Dict[str, Any] = {"load_mode": None}
    config: Optional[DatasetConfig] = None

    if dataset_config is not None:
        meta["load_mode"] = "dataset_config"
        config = DatasetConfig.load(dataset_config)
        loaded = Dataset.from_config(config)
        dataset, note = _coerce_dataset(loaded)
        if note:
            meta["routing_note"] = note
        path = config.path
        if isinstance(path, list):
            path = path[0]
        assert isinstance(path, str)
        return dataset, config, path, meta

    if lmdb_path is None:
        raise ValueError("Either dataset_config or lmdb_path is required.")

    meta["load_mode"] = "lmdb_path_direct"
    meta["routing_note"] = (
        "MAGICLMDBDataset with default pulsemap/features/graph; "
        "no DatasetConfig supplied"
    )
    graph = _default_magic_graph()
    dataset = MAGICLMDBDataset(
        path=lmdb_path,
        pulsemaps="MAGICPixels",
        features=list(MAGIC_NODE_FEATURES),
        truth=["event_no", "event_id"],
        index_column="event_no",
        truth_table="truth",
        data_representation=graph,
    )
    return dataset, config, lmdb_path, meta


def _sample_indices(
    n_total: int, n_sample: int, seed: int
) -> np.ndarray:
    n = min(int(n_sample), int(n_total))
    if n <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if n >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def _graph_dict_from_config(config: Optional[DatasetConfig]) -> Optional[dict]:
    if config is None:
        return None
    cfg = config.dict()
    graph = cfg.get("data_representation") or cfg.get("graph_definition")
    return graph if isinstance(graph, dict) else None


def _summarize_graph_dict(graph: Optional[dict]) -> Dict[str, Any]:
    if not graph:
        return {"status": "unavailable: no graph definition in config"}
    args = graph.get("arguments") or {}
    det = args.get("detector") or {}
    node = args.get("node_definition") or {}
    return {
        "class_name": graph.get("class_name"),
        "columns": args.get("columns"),
        "nb_nearest_neighbours": args.get("nb_nearest_neighbours"),
        "input_feature_names": args.get("input_feature_names"),
        "detector_class": det.get("class_name") if isinstance(det, dict) else None,
        "node_definition_class": (
            node.get("class_name") if isinstance(node, dict) else None
        ),
        "dtype": args.get("dtype"),
    }


def _audit_manifest(lmdb_path: str) -> Dict[str, Any]:
    manifest = read_conversion_manifest(lmdb_path)
    candidates = [str(p) for p in manifest_paths_for_dataset(lmdb_path)]
    if manifest is None:
        return {
            "present": False,
            "manifest_paths_checked": candidates,
            "status": "unavailable: no conversion manifest (legacy dataset)",
        }
    timing = manifest.get("timing_settings")
    return {
        "present": True,
        "manifest_paths_checked": candidates,
        "timing_settings": timing,
        "is_mc": manifest.get("is_mc"),
        "parquet_schema_version": manifest.get("parquet_schema_version"),
        "manifest_version": manifest.get("manifest_version"),
        "conversion_timestamp_utc": manifest.get("conversion_timestamp_utc"),
    }


def _scale_feature(
    detector: MAGIC, name: str, values: np.ndarray
) -> np.ndarray:
    fmap = detector.feature_map()
    fn = fmap.get(name)
    if fn is None:
        return values.astype(float)
    with torch.no_grad():
        scaled = fn(torch.as_tensor(values, dtype=torch.float32))
    return scaled.detach().cpu().numpy().astype(float)


def _collect_raw_features(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """Concatenate raw node features across sampled events."""
    feature_names = list(getattr(dataset, "_features", MAGIC_NODE_FEATURES))
    chunks: Dict[str, List[np.ndarray]] = {n: [] for n in feature_names}
    tel_idx = feature_names.index("tel_id") if "tel_id" in feature_names else None

    for idx in indices:
        try:
            features, _, _, _ = dataset._query(int(idx))  # type: ignore[attr-defined]
        except Exception:
            continue
        if features.ndim != 2:
            continue
        for col, name in enumerate(feature_names):
            if col >= features.shape[1]:
                continue
            chunks[name].append(np.asarray(features[:, col]))

    out: Dict[str, np.ndarray] = {}
    for name, parts in chunks.items():
        if parts:
            out[name] = np.concatenate(parts)
        else:
            out[name] = np.array([], dtype=float)
    out["_tel_index"] = tel_idx  # type: ignore[assignment]
    return out


def _quantile_row(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {label: float("nan") for label in QUANTILE_LABELS}
    qs = np.quantile(values.astype(float), QUANTILE_LEVELS)
    return {
        label: float(q)
        for label, q in zip(QUANTILE_LABELS, qs, strict=True)
    }


def _audit_feature_quantiles(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        raw = _collect_raw_features(dataset, indices)
        detector = MAGIC(use_signal_epsilon=True)
        tel = raw.get("tel_id", np.array([]))
        if tel.size == 0:
            return {"status": "unavailable: no pulse features in sample"}

        tables: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {"per_telescope": {}, "features": MAGIC_NODE_FEATURES}

        for tel_id in (0, 1):
            mask = np.isclose(tel, float(tel_id))
            tel_key = f"tel_{tel_id}"
            summary["per_telescope"][tel_key] = {}
            for fname in MAGIC_NODE_FEATURES:
                values = raw.get(fname, np.array([]))
                if values.size == 0:
                    continue
                sub = values[mask] if mask.any() else np.array([])
                raw_q = _quantile_row(sub)
                scaled_q = _quantile_row(_scale_feature(detector, fname, sub))
                summary["per_telescope"][tel_key][fname] = {
                    "raw": raw_q,
                    "detector_scaled": scaled_q,
                    "n_nodes": int(sub.size),
                }
                tables.append(
                    {
                        "tel_id": tel_id,
                        "feature": fname,
                        "kind": "raw",
                        **raw_q,
                        "n_nodes": int(sub.size),
                    }
                )
                tables.append(
                    {
                        "tel_id": tel_id,
                        "feature": fname,
                        "kind": "detector_scaled",
                        **scaled_q,
                        "n_nodes": int(sub.size),
                    }
                )

        return {
            "summary": summary,
            "source_table": tables,
            "status": "ok",
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _infer_stride_ns(time_values: np.ndarray) -> Optional[float]:
    """Most common positive gap among sorted unique per-telescope times."""
    uniq = np.unique(np.asarray(time_values, dtype=float))
    if uniq.size < 2:
        return None
    gaps = np.diff(np.sort(uniq))
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return None
    counts = Counter(np.round(positive, decimals=6))
    mode_gap, _ = counts.most_common(1)[0]
    return float(mode_gap)


def _audit_node_time_distributions(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        feature_names = list(getattr(dataset, "_features", MAGIC_NODE_FEATURES))
        if "time" not in feature_names or "tel_id" not in feature_names:
            return {"status": "unavailable: time/tel_id columns missing"}
        t_idx = feature_names.index("time")
        tel_idx = feature_names.index("tel_id")

        per_event: List[Dict[str, Any]] = []
        stride_gaps: List[float] = []

        for idx in indices:
            try:
                features, _, _, _ = dataset._query(int(idx))  # type: ignore[attr-defined]
            except Exception:
                continue
            if features.shape[0] == 0:
                continue
            times = features[:, t_idx]
            tels = features[:, tel_idx]
            for tel_id in (0.0, 1.0):
                mask = np.isclose(tels, tel_id)
                if not mask.any():
                    continue
                t_sub = times[mask]
                stride = _infer_stride_ns(t_sub)
                if stride is not None:
                    stride_gaps.append(stride)
                per_event.append(
                    {
                        "sequential_index": int(idx),
                        "tel_id": int(tel_id),
                        "time_min": float(np.min(t_sub)),
                        "time_max": float(np.max(t_sub)),
                        "time_mean": float(np.mean(t_sub)),
                        "time_std": float(np.std(t_sub)),
                        "inferred_stride_ns": stride,
                    }
                )

        inferred: Any
        if stride_gaps:
            inferred = float(Counter(stride_gaps).most_common(1)[0][0])
        else:
            inferred = None

        return {
            "status": "ok",
            "heuristic": (
                "inferred_stride_ns = mode of positive gaps between sorted "
                "unique per-event per-telescope time values (ns); labels "
                "legacy exports with non-uniform sampling as null per event"
            ),
            "aggregate_inferred_stride_ns": inferred,
            "per_event": per_event,
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _audit_pulse_image_distributions(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        signals: List[float] = []
        n_pulses: List[int] = []
        n_per_tel: List[int] = []

        feature_names = list(getattr(dataset, "_features", MAGIC_NODE_FEATURES))
        sig_idx = (
            feature_names.index("signal") if "signal" in feature_names else None
        )
        tel_idx = (
            feature_names.index("tel_id") if "tel_id" in feature_names else None
        )

        for idx in indices:
            try:
                features, _, _, _ = dataset._query(int(idx))  # type: ignore[attr-defined]
            except Exception:
                continue
            n = int(features.shape[0])
            n_pulses.append(n)
            if sig_idx is not None and n > 0:
                signals.extend(features[:, sig_idx].astype(float).tolist())
            if tel_idx is not None and n > 0:
                for tel_id in (0.0, 1.0):
                    c = int(np.sum(np.isclose(features[:, tel_idx], tel_id)))
                    if c > 0:
                        n_per_tel.append(c)

        def _dist(vals: Sequence[float]) -> Dict[str, Any]:
            if not vals:
                return {"status": "empty"}
            arr = np.asarray(vals, dtype=float)
            return {"quantiles": _quantile_row(arr), "n": int(arr.size)}

        return {
            "status": "ok",
            "signal": _dist(signals),
            "n_pulses": _dist([float(x) for x in n_pulses]),
            "image_size_per_telescope": _dist([float(x) for x in n_per_tel]),
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _audit_graph_definition(
    config: Optional[DatasetConfig],
    checkpoint: Optional[str],
    lmdb_path: str,
) -> Dict[str, Any]:
    requested = _summarize_graph_dict(_graph_dict_from_config(config))
    result: Dict[str, Any] = {
        "requested": requested,
        "effective_from_checkpoint": None,
        "status": "ok",
    }

    if checkpoint is None:
        result["effective_from_checkpoint"] = {
            "status": "unavailable: no --checkpoint supplied"
        }
        return result

    ckpt_path = Path(checkpoint)
    model_config_path = ckpt_path.parent / "model_config.yml"
    if not model_config_path.is_file():
        result["effective_from_checkpoint"] = {
            "status": (
                f"unavailable: model_config.yml not found next to checkpoint "
                f"({model_config_path})"
            )
        }
        return result

    if config is None:
        result["effective_from_checkpoint"] = {
            "status": (
                "unavailable: graph resolution requires a dataset config; "
                "use --mc-dataset-config or --real-dataset-config"
            )
        }
        return result

    try:
        resolved = resolve_dataset_graph_from_model(
            config, str(model_config_path), strict=False
        )
        effective = _summarize_graph_dict(_graph_dict_from_config(resolved))
        result["effective_from_checkpoint"] = effective
    except Exception as exc:
        result["effective_from_checkpoint"] = {
            "status": f"unavailable: {exc}"
        }
    return result


def _knn_columns(dataset: Dataset) -> List[int]:
    rep = getattr(dataset, "_data_representation", None)
    if rep is None:
        return [1, 2, 4, 3]
    edge = getattr(rep, "_edge_definition", None)
    if edge is not None and hasattr(edge, "_columns"):
        return list(edge._columns)
    return [1, 2, 4, 3]


def _audit_edge_statistics(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        feature_names = list(getattr(dataset, "_features", MAGIC_NODE_FEATURES))
        tel_idx = (
            feature_names.index("tel_id") if "tel_id" in feature_names else None
        )
        knn_cols = _knn_columns(dataset)
        per_event: List[Dict[str, Any]] = []
        all_dists: List[float] = []

        for idx in indices:
            try:
                data = dataset[int(idx)]
            except Exception:
                continue
            edge_index = getattr(data, "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                per_event.append(
                    {
                        "sequential_index": int(idx),
                        "n_edges": 0,
                        "mean_edge_distance": float("nan"),
                        "std_edge_distance": float("nan"),
                        "cross_telescope_fraction": float("nan"),
                    }
                )
                continue

            x = data.x.detach().cpu()
            pos = x[:, knn_cols].numpy()
            src = edge_index[0].cpu().numpy()
            tgt = edge_index[1].cpu().numpy()
            diff = pos[src] - pos[tgt]
            dists = np.linalg.norm(diff, axis=1)
            all_dists.extend(dists.tolist())

            cross_frac = float("nan")
            if tel_idx is not None and x.shape[1] > tel_idx:
                tel = x[:, tel_idx].numpy()
                tel_src = tel[src]
                tel_tgt = tel[tgt]
                cross = np.not_equal(tel_src, tel_tgt)
                cross_frac = float(np.mean(cross)) if cross.size else float("nan")

            per_event.append(
                {
                    "sequential_index": int(idx),
                    "n_edges": int(edge_index.shape[1]),
                    "mean_edge_distance": float(np.mean(dists)),
                    "std_edge_distance": float(np.std(dists)),
                    "cross_telescope_fraction": cross_frac,
                }
            )

        agg = {}
        if all_dists:
            arr = np.asarray(all_dists, dtype=float)
            agg = {
                "mean_edge_distance": float(np.mean(arr)),
                "std_edge_distance": float(np.std(arr)),
                "n_edges_total": int(arr.size),
            }

        return {
            "status": "ok",
            "knn_columns": knn_cols,
            "aggregate": agg,
            "per_event": per_event,
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _read_global_field(
    dataset: Dataset, sequential_index: int, field: str
) -> Optional[float]:
    try:
        dataset._update_cache(int(sequential_index))  # type: ignore[attr-defined]
        cached = getattr(dataset, "_cached_data", {})
        if not isinstance(cached, dict):
            return None
        for table in ("global", "truth"):
            tbl = cached.get(table)
            if not isinstance(tbl, dict) or field not in tbl:
                continue
            val = tbl[field]
            if isinstance(val, (list, np.ndarray)):
                val = val[0] if len(val) else None
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            return float(val)
    except Exception:
        return None
    return None


def _audit_source_reference_by_run(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        rows: List[Dict[str, Any]] = []
        nan_x = nan_y = 0
        for idx in indices:
            x_mm = _read_global_field(dataset, int(idx), "src_cam_x_mm")
            y_mm = _read_global_field(dataset, int(idx), "src_cam_y_mm")
            if x_mm is None and y_mm is None:
                x_mm = _read_global_field(dataset, int(idx), "src_cam_x_mm")
                y_mm = _read_global_field(dataset, int(idx), "src_cam_y_mm")
            run = _read_global_field(dataset, int(idx), "run_number")
            if x_mm is None or (isinstance(x_mm, float) and np.isnan(x_mm)):
                nan_x += 1
            if y_mm is None or (isinstance(y_mm, float) and np.isnan(y_mm)):
                nan_y += 1
            if x_mm is None and y_mm is None:
                continue
            rows.append(
                {
                    "sequential_index": int(idx),
                    "run_number": int(run) if run is not None else None,
                    "src_cam_x_mm": x_mm,
                    "src_cam_y_mm": y_mm,
                }
            )

        if not rows:
            return {
                "status": (
                    "unavailable: src_cam_x_mm/src_cam_y_mm not present "
                    "(expected for real data; MC may omit)"
                ),
                "nan_counts": {"src_cam_x_mm": nan_x, "src_cam_y_mm": nan_y},
            }

        df = pd.DataFrame(rows)
        by_run = (
            df.groupby("run_number", dropna=False)[["src_cam_x_mm", "src_cam_y_mm"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        by_run.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in by_run.columns
        ]
        return {
            "status": "ok",
            "nan_counts": {"src_cam_x_mm": nan_x, "src_cam_y_mm": nan_y},
            "by_run": by_run.to_dict(orient="records"),
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _audit_prediction_position(
    checkpoint: Optional[str],
) -> Dict[str, Any]:
    if checkpoint is None:
        return {
            "status": "predictions: skipped (no checkpoint / GPU job)",
        }
    # Full MAGIC DeepIce inference is not attempted in the audit CLI (CPU-only).
    return {
        "status": "predictions: skipped (no checkpoint / GPU job)",
        "note": (
            "Model forward pass omitted: checkpoint present but audit runs "
            "CPU-only without GPU inference job"
        ),
    }


def _find_classifier_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "particle_signal",
        "gamma_score",
        "classifier",
        "pred_particle_signal",
        "particle_signal_pred",
    ]
    for name in candidates:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in ("event_id", "event_no", "run_number")
    ]
    for c in numeric:
        if "signal" in c.lower() or "gamma" in c.lower() or "class" in c.lower():
            return c
    return numeric[0] if len(numeric) == 1 else None


def _survival_fractions(series: pd.Series, thresholds: Sequence[float]) -> Dict[str, float]:
    arr = series.astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {str(t): float("nan") for t in thresholds}
    return {str(t): float(np.mean(finite >= t)) for t in thresholds}


def _audit_classifier_survival(
    predictions_path: Optional[str],
) -> Dict[str, Any]:
    if predictions_path is None:
        return {"status": "unavailable: no --predictions parquet supplied"}
    try:
        path = Path(predictions_path)
        if not path.is_file():
            return {"status": f"unavailable: file not found ({path})"}
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        col = _find_classifier_column(df)
        if col is None:
            return {
                "status": "unavailable: could not identify classifier column",
                "columns": list(df.columns),
            }
        thresholds = (0.5, 0.9, 0.99)
        overall = _survival_fractions(df[col], thresholds)
        per_run: Any = "unavailable: run_number column missing"
        if "run_number" in df.columns:
            per_run = {}
            for run, grp in df.groupby("run_number"):
                per_run[str(run)] = _survival_fractions(grp[col], thresholds)
        return {
            "status": "ok",
            "classifier_column": col,
            "thresholds": list(thresholds),
            "overall": overall,
            "per_run": per_run,
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _audit_sky_coordinates(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        sample_idx = indices[:SKY_SAMPLE_EVENTS]
        missing_cols: set[str] = set()
        samples: List[Dict[str, Any]] = []

        for idx in sample_idx:
            row_missing = [
                c
                for c in V6_POINTING_COLUMNS
                if _read_global_field(dataset, int(idx), c) is None
            ]
            if row_missing:
                missing_cols.update(row_missing)
                samples.append(
                    {
                        "sequential_index": int(idx),
                        "status": f"missing columns: {sorted(row_missing)}",
                    }
                )
                continue

            x_mm = _read_global_field(dataset, int(idx), "src_cam_x_mm")
            y_mm = _read_global_field(dataset, int(idx), "src_cam_y_mm")
            if x_mm is None or y_mm is None:
                samples.append(
                    {
                        "sequential_index": int(idx),
                        "status": "missing src_cam_x_mm/src_cam_y_mm for sky transform",
                    }
                )
                continue

            dec = _read_global_field(dataset, int(idx), "pointing_corr_dec_deg")
            ha = _read_global_field(dataset, int(idx), "pointing_corr_ha_hours")
            lst = _read_global_field(
                dataset, int(idx), "local_sidereal_time_hours"
            )
            dist = _read_global_field(dataset, int(idx), "camera_dist_mm")
            assert dec is not None and ha is not None and lst is not None
            assert dist is not None

            ra_exact, dec_exact = camera_prediction_to_radec(
                x_mm,
                y_mm,
                camera_dist_mm=dist,
                cos_lat=COS_LAT,
                sin_lat=SIN_LAT,
                lst_hours=lst,
                corr_dec_deg=dec,
                corr_ha_hours=ha,
                mode="exact",
            )

            nominal_dec = _read_global_field(
                dataset, int(idx), "pointing_dec_deg"
            )
            nominal_ha = _read_global_field(
                dataset, int(idx), "pointing_ha_hours"
            )
            approx: Any = "unavailable: nominal pointing columns missing"
            if nominal_dec is not None and nominal_ha is not None:
                ra_approx, dec_approx = camera_prediction_to_radec(
                    x_mm,
                    y_mm,
                    camera_dist_mm=dist,
                    cos_lat=COS_LAT,
                    sin_lat=SIN_LAT,
                    lst_hours=lst,
                    mode="approx_nominal",
                    nominal_dec_deg=nominal_dec,
                    nominal_ha_hours=nominal_ha,
                )
                approx = {
                    "ra_hours": float(ra_approx),
                    "dec_deg": float(dec_approx),
                }

            samples.append(
                {
                    "sequential_index": int(idx),
                    "exact": {
                        "ra_hours": float(ra_exact),
                        "dec_deg": float(dec_exact),
                    },
                    "approx_nominal": approx,
                }
            )

        if missing_cols and all("missing" in s.get("status", "") for s in samples):
            return {
                "status": (
                    "unavailable: legacy file without v6 corrected pointing "
                    f"(missing: {sorted(missing_cols)})"
                ),
                "samples": samples,
            }

        return {"status": "ok", "samples": samples}
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _event_id_column(dataset: Dataset) -> str:
    return str(getattr(dataset, "_index_column", "event_no"))


def _audit_event_ids(
    dataset: Dataset, indices: np.ndarray
) -> Dict[str, Any]:
    try:
        col = _event_id_column(dataset)
        alt = "event_id" if col == "event_no" else "event_no"
        ids: List[int] = []

        for idx in indices:
            try:
                truth = dataset.query_table_as_mapping(  # type: ignore[attr-defined]
                    getattr(dataset, "_truth_table", "truth"),
                    [col, alt],
                    int(idx),
                )
            except Exception:
                try:
                    truth = dataset.query_table_as_mapping(
                        getattr(dataset, "_truth_table", "truth"),
                        [col],
                        int(idx),
                    )
                except Exception as exc:
                    return {"status": f"unavailable: {exc}"}

            for key in (col, alt):
                if key in truth:
                    arr = truth[key]
                    if arr.size:
                        ids.append(int(arr[0]))
                        break

        if not ids:
            return {"status": "unavailable: no event identifiers in sample"}

        arr = np.asarray(ids, dtype=np.int64)
        dtype_name = str(arr.dtype)
        n_unique = int(np.unique(arr).size)
        n = int(arr.size)
        pass_exact = dtype_name == "int64" and n_unique == n
        return {
            "status": "PASS" if pass_exact else "FAIL",
            "column": col,
            "dtype": dtype_name,
            "n": n,
            "n_unique": n_unique,
            "int64_exact_unique": pass_exact,
        }
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}


def _write_parquet_tables(
    output_dir: Path,
    label: str,
    feature_section: Dict[str, Any],
    edge_section: Dict[str, Any],
) -> List[str]:
    written: List[str] = []
    fq = feature_section.get("source_table")
    if isinstance(fq, list) and fq:
        path = output_dir / f"{label}_feature_quantiles.parquet"
        pd.DataFrame(fq).to_parquet(path, index=False)
        written.append(str(path))

    pe = edge_section.get("per_event")
    if isinstance(pe, list) and pe:
        path = output_dir / f"{label}_edge_stats.parquet"
        pd.DataFrame(pe).to_parquet(path, index=False)
        written.append(str(path))
    return written


def _write_plots(
    output_dir: Path,
    label: str,
    dataset: Dataset,
    indices: np.ndarray,
    feature_section: Dict[str, Any],
    edge_section: Dict[str, Any],
    source_section: Dict[str, Any],
) -> List[str]:
    np.random.seed(PLOT_SEED)
    written: List[str] = []

    try:
        raw = _collect_raw_features(dataset, indices)
        tel = raw.get("tel_id", np.array([]))
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
        axes = axes.ravel()
        for ax, fname in zip(axes, MAGIC_NODE_FEATURES, strict=True):
            vals = raw.get(fname, np.array([]))
            if vals.size == 0:
                ax.set_title(f"{fname} (empty)")
                continue
            for tel_id, color in ((0, "C0"), (1, "C1")):
                mask = np.isclose(tel, float(tel_id)) if tel.size else np.array([])
                sub = vals[mask] if mask.any() else np.array([])
                if sub.size:
                    ax.hist(sub, bins=30, alpha=0.55, label=f"tel {tel_id}", color=color)
            ax.set_title(fname)
            ax.legend(fontsize=8)
        fig.suptitle(f"{label}: raw feature histograms")
        fig.tight_layout()
        path = output_dir / f"{label}_feature_histograms.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(str(path))
    except Exception:
        plt.close("all")

    try:
        time_rows = feature_section.get("per_event") if False else None
        _ = time_rows
        feature_names = list(getattr(dataset, "_features", MAGIC_NODE_FEATURES))
        t_idx = feature_names.index("time")
        tel_idx = feature_names.index("tel_id")
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for idx in indices[: min(50, len(indices))]:
            features, _, _, _ = dataset._query(int(idx))  # type: ignore[attr-defined]
            if features.shape[0] == 0:
                continue
            for tel_id in (0.0, 1.0):
                mask = np.isclose(features[:, tel_idx], tel_id)
                if mask.any():
                    ax.scatter(
                        features[mask, t_idx],
                        np.full(mask.sum(), tel_id),
                        s=4,
                        alpha=0.4,
                    )
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("tel_id")
        ax.set_title(f"{label}: per-node time distribution (subset)")
        fig.tight_layout()
        path = output_dir / f"{label}_time_distribution.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(str(path))
    except Exception:
        plt.close("all")

    try:
        dists: List[float] = []
        knn_cols = edge_section.get("knn_columns", _knn_columns(dataset))
        for row in edge_section.get("per_event", [])[:200]:
            idx = row.get("sequential_index")
            if idx is None:
                continue
            data = dataset[int(idx)]
            ei = data.edge_index
            if ei is None or ei.numel() == 0:
                continue
            pos = data.x[:, knn_cols].detach().cpu().numpy()
            src, tgt = ei[0].cpu().numpy(), ei[1].cpu().numpy()
            diff = pos[src] - pos[tgt]
            dists.extend(np.linalg.norm(diff, axis=1).tolist())
        if dists:
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            ax.hist(dists, bins=40, color="steelblue", alpha=0.85)
            ax.set_xlabel("KNN coordinate-space edge distance")
            ax.set_title(f"{label}: edge-distance histogram")
            fig.tight_layout()
            path = output_dir / f"{label}_edge_distance_histogram.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)
            written.append(str(path))
    except Exception:
        plt.close("all")

    try:
        by_run = source_section.get("by_run")
        if isinstance(by_run, list) and by_run:
            df = pd.DataFrame(by_run)
            xcol = "src_cam_x_mm_mean"
            ycol = "src_cam_y_mm_mean"
            if xcol in df.columns and ycol in df.columns:
                fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
                ax.scatter(df[xcol], df[ycol], c=df.get("run_number", 0))
                ax.set_xlabel("mean src_cam_x_mm")
                ax.set_ylabel("mean src_cam_y_mm")
                ax.set_title(f"{label}: source reference by run")
                fig.tight_layout()
                path = output_dir / f"{label}_source_reference_by_run.png"
                fig.savefig(path, dpi=100)
                plt.close(fig)
                written.append(str(path))
    except Exception:
        plt.close("all")

    return written


def _audit_single_dataset(
    label: str,
    dataset: Dataset,
    config: Optional[DatasetConfig],
    lmdb_path: str,
    *,
    checkpoint: Optional[str],
    sample_events: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    indices = _sample_indices(len(dataset), sample_events, seed)
    load_note: Dict[str, Any] = {"n_sampled": int(indices.size), "n_total": len(dataset)}

    manifest = _audit_manifest(lmdb_path)
    features = _audit_feature_quantiles(dataset, indices)
    times = _audit_node_time_distributions(dataset, indices)
    pulses = _audit_pulse_image_distributions(dataset, indices)
    graph_def = _audit_graph_definition(config, checkpoint, lmdb_path)
    edges = _audit_edge_statistics(dataset, indices)
    source_ref = _audit_source_reference_by_run(dataset, indices)
    predictions = _audit_prediction_position(checkpoint)
    sky = _audit_sky_coordinates(dataset, indices)
    event_ids = _audit_event_ids(dataset, indices)

    section = {
        "label": label,
        "lmdb_path": lmdb_path,
        "load": load_note,
        "manifest": manifest,
        "feature_quantiles": features.get("summary", features),
        "node_time_distributions": times,
        "pulse_image_distributions": pulses,
        "graph_definition": graph_def,
        "edge_statistics": edges,
        "source_reference_by_run": source_ref,
        "prediction_position_by_run": predictions,
        "sky_coordinates": sky,
        "event_ids": event_ids,
    }

    _write_parquet_tables(output_dir, label, features, edges)
    plots = _write_plots(
        output_dir, label, dataset, indices, features, edges, source_ref
    )
    section["artifacts"] = {"plots": plots}
    return section


def run_audit(
    output_dir: Union[str, Path],
    *,
    mc_dataset_config: Optional[str] = None,
    real_dataset_config: Optional[str] = None,
    lmdb_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    predictions: Optional[str] = None,
    sample_events: int = 1000,
    seed: int = 42,
    argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the MAGIC pipeline audit and write artifacts under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not mc_dataset_config and not real_dataset_config and not lmdb_path:
        raise ValueError(
            "At least one of mc_dataset_config, real_dataset_config, or "
            "lmdb_path is required."
        )

    report: Dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "sample_events": sample_events,
        "seed": seed,
        "datasets": {},
    }

    configs_used: Dict[str, Any] = {}

    if mc_dataset_config:
        ds, cfg, path, meta = _load_dataset(
            dataset_config=mc_dataset_config, lmdb_path=None
        )
        report["datasets"]["mc"] = _audit_single_dataset(
            "mc",
            ds,
            cfg,
            path,
            checkpoint=checkpoint,
            sample_events=sample_events,
            seed=seed,
            output_dir=out,
        )
        report["datasets"]["mc"]["load"].update(meta)
        configs_used["mc_dataset_config"] = mc_dataset_config

    if real_dataset_config:
        ds, cfg, path, meta = _load_dataset(
            dataset_config=real_dataset_config, lmdb_path=None
        )
        report["datasets"]["real"] = _audit_single_dataset(
            "real",
            ds,
            cfg,
            path,
            checkpoint=checkpoint,
            sample_events=sample_events,
            seed=seed + 1,
            output_dir=out,
        )
        report["datasets"]["real"]["load"].update(meta)
        configs_used["real_dataset_config"] = real_dataset_config

    if lmdb_path and not mc_dataset_config and not real_dataset_config:
        ds, cfg, path, meta = _load_dataset(
            dataset_config=None, lmdb_path=lmdb_path
        )
        report["datasets"]["direct"] = _audit_single_dataset(
            "direct",
            ds,
            cfg,
            path,
            checkpoint=checkpoint,
            sample_events=sample_events,
            seed=seed,
            output_dir=out,
        )
        report["datasets"]["direct"]["load"].update(meta)
        configs_used["lmdb_path"] = lmdb_path

    report["classifier_survival"] = _audit_classifier_survival(predictions)

    audit_path = out / "audit.json"
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, sort_keys=True)
        fh.write("\n")

    snapshot = {
        "argv": argv if argv is not None else sys.argv,
        "configs": configs_used,
        "checkpoint": checkpoint,
        "predictions": predictions,
        "sample_events": sample_events,
        "seed": seed,
        "graphnet_commit": _safe_git_commit(),
        "generated_at_utc": report["generated_at_utc"],
    }
    snap_path = out / "command_snapshot.json"
    with snap_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(snapshot), fh, indent=2, sort_keys=True)
        fh.write("\n")

    report["artifacts"] = {
        "audit_json": str(audit_path),
        "command_snapshot_json": str(snap_path),
    }
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a MAGIC LMDB pipeline export (features, graph, integrity)."
    )
    parser.add_argument("--mc-dataset-config", type=str, default=None)
    parser.add_argument("--real-dataset-config", type=str, default=None)
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default=None,
        help=(
            "Direct LMDB path when DatasetConfig is unavailable; uses "
            "MAGICLMDBDataset defaults"
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--sample-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m graphnet.utilities.magic.audit``."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if (
        not args.mc_dataset_config
        and not args.real_dataset_config
        and not args.lmdb_path
    ):
        parser.error(
            "At least one of --mc-dataset-config, --real-dataset-config, "
            "or --lmdb-path is required."
        )

    run_audit(
        args.output_dir,
        mc_dataset_config=args.mc_dataset_config,
        real_dataset_config=args.real_dataset_config,
        lmdb_path=args.lmdb_path,
        checkpoint=args.checkpoint,
        predictions=args.predictions,
        sample_events=args.sample_events,
        seed=args.seed,
        argv=sys.argv if argv is None else ["audit", *argv],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
