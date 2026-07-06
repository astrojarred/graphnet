"""MAGIC LMDB conversion manifest read/write utilities.

Each converted MAGIC LMDB dataset directory should have a sibling
``magic_conversion_manifest.json`` recording conversion provenance so loaders
can warn on legacy datasets and optionally enforce timing settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from graphnet.constants import MAGIC_GEOMETRY_TABLE_DIR

MANIFEST_FILENAME = "magic_conversion_manifest.json"
MANIFEST_VERSION = 1
_FINGERPRINT_CHUNK_BYTES = 64 * 1024

_logger = logging.getLogger("graphnet")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_git_info() -> Dict[str, Optional[str]]:
    """Return graphnet commit hash and dirty flag; never raises."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        commit = None
    dirty: Optional[bool] = None
    if commit is not None:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
            dirty = bool(status.strip())
        except Exception:
            dirty = None
    return {"graphnet_commit": commit, "graphnet_commit_dirty": dirty}


def _file_fingerprint(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Fingerprint a file using size, mtime, and sha256 of the first 64 KiB.

    For large parquet inputs this avoids reading the full file while still
    detecting common changes (size/mtime) and many content edits (prefix hash).
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        stat = p.stat()
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            digest.update(fh.read(_FINGERPRINT_CHUNK_BYTES))
        return {
            "path": str(p.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256_first_64kb": digest.hexdigest(),
            "fingerprint_scheme": "size+mtime_ns+sha256(first_64kb)",
        }
    except Exception:
        return None


def _geometry_table_hash() -> Optional[str]:
    """Sha256 of the bundled MAGIC geometry parquet, if present."""
    try:
        geom_path = Path(MAGIC_GEOMETRY_TABLE_DIR) / "magic.parquet"
        if not geom_path.is_file():
            return None
        digest = hashlib.sha256()
        with geom_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _read_parquet_schema_version(sources: List[str]) -> Optional[int]:
    """Read ``parquet_schema_version`` from the first readable source file."""
    for src in sources:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(src, columns=["parquet_schema_version"])
            if table.num_rows == 0:
                continue
            col = table.column("parquet_schema_version")
            val = col[0].as_py()
            if val is not None:
                return int(val)
        except Exception:
            continue
    return None


def _reader_manifest_fields(reader: Any) -> Dict[str, Any]:
    """Best-effort manifest fields from a :class:`MAGICParquetReader`."""
    fields: Dict[str, Any] = {}
    try:
        timing = getattr(reader, "timing_settings", None)
        if isinstance(timing, dict):
            fields["timing_settings"] = dict(timing)
    except Exception:
        fields["timing_settings"] = None

    try:
        fields["truth_columns"] = list(getattr(reader, "_truth_columns", []) or [])
    except Exception:
        fields["truth_columns"] = None

    try:
        fields["global_columns"] = list(getattr(reader, "_global_params", []) or [])
    except Exception:
        fields["global_columns"] = None

    try:
        fields["cleaning"] = {
            "apply_cleaning": getattr(reader, "_apply_cleaning", None),
            "cleaning_n_low": getattr(reader, "_cleaning_n_low", None),
            "max_log_size_clipped": getattr(reader, "_max_log_size_clipped", None),
        }
    except Exception:
        fields["cleaning"] = None

    try:
        from graphnet.data.extractors.magic.cleaning import (
            DEFAULT_N_PIXELS,
            DEFAULT_N_TIMESLICES,
        )

        fields["pixel_count"] = DEFAULT_N_PIXELS
        fields["timeslice_count_default"] = DEFAULT_N_TIMESLICES
    except Exception:
        fields["pixel_count"] = None
        fields["timeslice_count_default"] = None

    try:
        fields["node_feature_order"] = [
            "signal",
            "x_cam",
            "y_cam",
            "time",
            "tel_id",
        ]
    except Exception:
        fields["node_feature_order"] = None

    return fields


def manifest_paths_for_dataset(dataset_dir: Union[str, Path]) -> List[Path]:
    """Return candidate manifest locations for an LMDB dataset path."""
    p = Path(dataset_dir)
    candidates: List[Path] = []
    if p.name.endswith(".lmdb"):
        candidates.append(p.parent / MANIFEST_FILENAME)
    candidates.append(p / MANIFEST_FILENAME)
    seen: set[Path] = set()
    ordered: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def write_conversion_manifest(
    output_dir: Union[str, Path],
    *,
    sources: Optional[List[str]] = None,
    reader: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``magic_conversion_manifest.json`` under ``output_dir``.

    All fields are best-effort; failures to populate individual fields are
    logged and recorded as ``null`` rather than aborting conversion.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / MANIFEST_FILENAME

    sources = list(sources or [])
    extra = dict(extra or {})
    payload: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "conversion_timestamp_utc": _utc_now_iso(),
        "command_line": list(sys.argv),
    }

    try:
        payload["sources"] = [
            fp for fp in (_file_fingerprint(s) for s in sources) if fp is not None
        ]
    except Exception as exc:
        _logger.warning("MAGIC manifest: could not fingerprint sources: %s", exc)
        payload["sources"] = None

    try:
        payload["parquet_schema_version"] = (
            _read_parquet_schema_version(sources) if sources else None
        )
    except Exception:
        payload["parquet_schema_version"] = None

    try:
        payload.update(_safe_git_info())
    except Exception:
        payload["graphnet_commit"] = None
        payload["graphnet_commit_dirty"] = None

    try:
        payload["geometry_table_sha256"] = _geometry_table_hash()
    except Exception:
        payload["geometry_table_sha256"] = None

    if reader is not None:
        try:
            payload.update(_reader_manifest_fields(reader))
        except Exception as exc:
            _logger.warning("MAGIC manifest: reader fields failed: %s", exc)

    for key in ("pixel_count", "timeslice_count_default", "timing_settings"):
        if key not in payload:
            payload[key] = None

    if extra.get("is_mc") is not None:
        payload["is_mc"] = extra.get("is_mc")
    else:
        payload["is_mc"] = None

    for key in (
        "cleaning",
        "node_feature_order",
        "truth_columns",
        "global_columns",
    ):
        if key in extra and payload.get(key) is None:
            payload[key] = extra.get(key)

    try:
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception as exc:
        _logger.warning("MAGIC manifest: failed to write %s: %s", dest, exc)
        raise

    return dest


def read_conversion_manifest(dataset_dir: Union[str, Path]) -> Optional[dict]:
    """Load the conversion manifest for ``dataset_dir``, if present."""
    for candidate in manifest_paths_for_dataset(dataset_dir):
        if candidate.is_file():
            try:
                with candidate.open(encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                _logger.warning(
                    "MAGIC manifest: could not read %s: %s", candidate, exc
                )
                return None
    return None


def _flatten_settings(manifest: dict) -> Dict[str, Any]:
    """Merge timing_settings and top-level keys for comparison."""
    flat: Dict[str, Any] = {}
    timing = manifest.get("timing_settings")
    if isinstance(timing, dict):
        flat.update(timing)
    cleaning = manifest.get("cleaning")
    if isinstance(cleaning, dict):
        for k, v in cleaning.items():
            flat[f"cleaning.{k}"] = v
    return flat


def check_manifest(
    dataset_dir: Union[str, Path],
    required: dict,
    *,
    strict: bool = False,
) -> List[str]:
    """Compare ``required`` settings against the dataset conversion manifest.

    Returns a list of human-readable mismatch strings. When ``strict`` is
    ``True`` and mismatches exist (or the manifest is absent), raises
    ``ValueError``. When the manifest is absent and ``strict`` is ``False``,
    emits a warning with ``timing_provenance=unknown/legacy`` wording.
    """
    manifest = read_conversion_manifest(dataset_dir)
    mismatches: List[str] = []

    if manifest is None:
        msg = (
            "MAGIC conversion manifest not found; "
            "timing_provenance=unknown/legacy"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        _logger.warning(msg)
        return mismatches

    flat = _flatten_settings(manifest)
    for key, expected in required.items():
        actual = flat.get(key, manifest.get(key))
        if actual != expected:
            mismatches.append(
                f"{key}: manifest={actual!r}, required={expected!r}"
            )

    if mismatches and strict:
        raise ValueError("; ".join(mismatches))

    return mismatches
