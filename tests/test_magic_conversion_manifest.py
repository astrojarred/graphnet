"""Tests for MAGIC LMDB conversion manifest utilities and loader hooks."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import pytest

from graphnet.data.constants import FEATURES
from graphnet.data.dataset.lmdb import LMDBDataset
from graphnet.data.readers.magic_parquet_reader import MAGICParquetReader
from graphnet.data.utilities.magic_manifest import (
    MANIFEST_FILENAME,
    check_manifest,
    read_conversion_manifest,
    write_conversion_manifest,
)
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses

PULSEMAP = "pulses"
TRUTH_TABLE = "truth"
INDEX_COLUMN = "event_id"
FEATURE_NAMES = FEATURES.DEEPCORE
TRUTH_COLUMNS = ["energy_gev"]
N_PULSES = 2
N_EVENTS = 3
EVENT_IDS = np.array([100, 101, 102], dtype=np.int64)

CANONICAL_TIMING = {
    "real_timecal_centering": "per_telescope_mean",
    "real_timeslice_duration": 1.25,
    "mc_graft_timeslice_duration": 0.6,
    "allow_placeholder_real_time": False,
}


def _make_reader(**timing_overrides: Any) -> MAGICParquetReader:
    kwargs = dict(
        apply_cleaning=True,
        cleaning_n_low=6.0,
        **CANONICAL_TIMING,
    )
    kwargs.update(timing_overrides)
    return MAGICParquetReader(**kwargs)


def _write_fake_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "parquet_schema_version": [5],
            "event_id": [1],
            "dummy": [0],
        }
    )
    df.to_parquet(path, index=False)


def _write_test_lmdb(path: str, event_ids: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    env = lmdb.open(path, map_size=1 << 24, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__meta_serialization__", b"pickle")
            for i, event_id in enumerate(event_ids):
                pulses = {
                    name: rng.normal(size=N_PULSES).tolist()
                    for name in FEATURE_NAMES
                }
                truth = {
                    INDEX_COLUMN: int(event_id),
                    "energy_gev": float(10.0 + i),
                }
                entry = {PULSEMAP: pulses, TRUTH_TABLE: truth}
                txn.put(
                    str(int(event_id)).encode("utf-8"),
                    pickle.dumps(entry),
                )
    finally:
        env.close()


def _make_graph_definition() -> KNNGraph:
    return KNNGraph(
        detector=IceCubeDeepCore(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=2,
        input_feature_names=list(FEATURE_NAMES),
    )


def test_write_conversion_manifest_records_reader_settings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "waveforms.parquet"
    _write_fake_parquet(source)
    reader = _make_reader()
    outdir = tmp_path / "converted"
    manifest_path = write_conversion_manifest(
        outdir,
        sources=[str(source)],
        reader=reader,
        extra={"is_mc": True},
    )
    assert manifest_path.name == MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["manifest_version"] == 1
    assert data["is_mc"] is True
    assert data["timing_settings"] == CANONICAL_TIMING
    assert data["graphnet_commit"] is not None
    assert isinstance(data["sources"], list) and len(data["sources"]) == 1
    fp = data["sources"][0]
    assert fp["size_bytes"] > 0
    assert fp["sha256_first_64kb"]
    assert fp["fingerprint_scheme"] == "size+mtime_ns+sha256(first_64kb)"
    assert data["parquet_schema_version"] == 5
    assert data["truth_columns"]
    assert data["global_columns"]
    assert data["pixel_count"] == 1039
    assert data["command_line"]


def test_read_and_check_manifest_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "src.parquet"
    _write_fake_parquet(source)
    reader = _make_reader()
    outdir = tmp_path / "dataset"
    write_conversion_manifest(
        outdir,
        sources=[str(source)],
        reader=reader,
        extra={"is_mc": False},
    )

    loaded = read_conversion_manifest(outdir)
    assert loaded is not None
    assert loaded["timing_settings"]["real_timecal_centering"] == (
        "per_telescope_mean"
    )

    assert (
        check_manifest(
            outdir,
            {"real_timecal_centering": "per_telescope_mean"},
            strict=False,
        )
        == []
    )

    with pytest.raises(ValueError, match="real_timecal_centering"):
        check_manifest(
            outdir,
            {"real_timecal_centering": "none"},
            strict=True,
        )

    mismatches = check_manifest(
        outdir,
        {"real_timecal_centering": "none"},
        strict=False,
    )
    assert len(mismatches) == 1


def test_absent_manifest_check_warns_or_raises(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="timing_provenance=unknown/legacy"):
        assert (
            check_manifest(
                tmp_path,
                {"real_timecal_centering": "none"},
                strict=False,
            )
            == []
        )

    with pytest.raises(ValueError, match="timing_provenance=unknown/legacy"):
        check_manifest(
            tmp_path,
            {"real_timecal_centering": "none"},
            strict=True,
        )


def test_lmdb_dataset_warns_without_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lmdb_path = str(tmp_path / "test.lmdb")
    _write_test_lmdb(lmdb_path, EVENT_IDS)

    caplog.set_level(logging.WARNING, logger="graphnet")
    LMDBDataset(
        path=lmdb_path,
        pulsemaps=PULSEMAP,
        features=list(FEATURE_NAMES),
        truth=list(TRUTH_COLUMNS),
        index_column=INDEX_COLUMN,
        truth_table=TRUTH_TABLE,
        data_representation=_make_graph_definition(),
    )
    assert any(
        "timing_provenance=unknown/legacy" in r.message for r in caplog.records
    )


def test_lmdb_dataset_logs_manifest_and_enforces_required_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lmdb_path = tmp_path / "merged.lmdb"
    _write_test_lmdb(str(lmdb_path), EVENT_IDS)

    source = tmp_path / "waveforms.parquet"
    _write_fake_parquet(source)
    write_conversion_manifest(
        tmp_path,
        sources=[str(source)],
        reader=_make_reader(real_timecal_centering="none"),
        extra={"is_mc": True},
    )

    caplog.set_level(logging.INFO, logger="graphnet")
    LMDBDataset(
        path=str(lmdb_path),
        pulsemaps=PULSEMAP,
        features=list(FEATURE_NAMES),
        truth=list(TRUTH_COLUMNS),
        index_column=INDEX_COLUMN,
        truth_table=TRUTH_TABLE,
        data_representation=_make_graph_definition(),
    )
    assert any(
        MANIFEST_FILENAME in r.message for r in caplog.records
    )

    with pytest.raises(ValueError, match="real_timecal_centering"):
        LMDBDataset(
            path=str(lmdb_path),
            pulsemaps=PULSEMAP,
            features=list(FEATURE_NAMES),
            truth=list(TRUTH_COLUMNS),
            index_column=INDEX_COLUMN,
            truth_table=TRUTH_TABLE,
            data_representation=_make_graph_definition(),
            required_manifest_settings={
                "real_timecal_centering": "per_telescope_mean",
            },
        )
