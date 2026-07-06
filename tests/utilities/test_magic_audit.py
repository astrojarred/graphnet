"""End-to-end tests for ``graphnet.utilities.magic.audit``."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pytest
import torch

from graphnet.data.readers.magic_parquet_reader import MAGICParquetReader
from graphnet.data.utilities.magic_manifest import write_conversion_manifest
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.models.detector.magic import MAGIC
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.utilities.magic.audit import main, run_audit

PULSEMAP = "MAGICPixels"
TRUTH_TABLE = "truth"
GLOBAL_TABLE = "global"
INDEX_COLUMN = "event_no"
FEATURE_NAMES = ["signal", "x_cam", "y_cam", "time", "tel_id"]
N_EVENTS = 12
BASE_ID = 2**53 + 17
EVENT_IDS = BASE_ID + np.arange(N_EVENTS, dtype=np.int64) * 7919 + 1
NODES_PER_TEL = 16
TIME_STRIDE_NS = 1.25


def _make_graph() -> KNNGraph:
    return KNNGraph(
        detector=MAGIC(use_signal_epsilon=True),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=4,
        input_feature_names=list(FEATURE_NAMES),
        dtype=torch.float32,
        columns=[1, 2, 4, 3],
    )


def _write_magic_lmdb(
    path: Path,
    event_ids: np.ndarray,
    *,
    include_global: bool = True,
    include_pointing: bool = False,
) -> None:
    rng = np.random.default_rng(7)
    env = lmdb.open(str(path), map_size=1 << 28, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__meta_serialization__", b"pickle")
            for i, event_id in enumerate(event_ids):
                rows_signal: list[float] = []
                rows_x: list[float] = []
                rows_y: list[float] = []
                rows_t: list[float] = []
                rows_tel: list[float] = []
                for tel_id in (0.0, 1.0):
                    for j in range(NODES_PER_TEL):
                        rows_signal.append(float(rng.uniform(1.0, 250.0)))
                        rows_x.append(float(rng.uniform(-80.0, 80.0)))
                        rows_y.append(float(rng.uniform(-80.0, 80.0)))
                        rows_t.append(
                            float(j * TIME_STRIDE_NS + rng.uniform(-2.0, 2.0))
                        )
                        rows_tel.append(tel_id)
                pulses = {
                    "signal": rows_signal,
                    "x_cam": rows_x,
                    "y_cam": rows_y,
                    "time": rows_t,
                    "tel_id": rows_tel,
                }
                truth = {
                    INDEX_COLUMN: int(event_id),
                    "event_id": int(event_id),
                    "run_number": int(1000 + (i % 3)),
                }
                entry: dict = {PULSEMAP: pulses, TRUTH_TABLE: truth}
                if include_global:
                    global_row = {
                        "run_number": int(1000 + (i % 3)),
                        "src_cam_x_mm": float(rng.uniform(-50.0, 50.0)),
                        "src_cam_y_mm": float(rng.uniform(-50.0, 50.0)),
                    }
                    if include_pointing:
                        global_row.update(
                            {
                                "pointing_corr_dec_deg": 38.2 + 0.01 * i,
                                "pointing_corr_ha_hours": 0.8 + 0.001 * i,
                                "local_sidereal_time_hours": 12.5,
                                "camera_dist_mm": 17000.0,
                                "pointing_dec_deg": 38.15,
                                "pointing_ha_hours": 0.81,
                            }
                        )
                    entry[GLOBAL_TABLE] = global_row
                txn.put(
                    str(int(event_id)).encode("utf-8"),
                    pickle.dumps(entry),
                )
    finally:
        env.close()


def _fake_parquet(path: Path) -> None:
    pd.DataFrame(
        {"parquet_schema_version": [6], "event_id": [1], "dummy": [0]}
    ).to_parquet(path, index=False)


@pytest.fixture()
def audit_dirs(tmp_path: Path) -> dict[str, Path]:
    with_manifest = tmp_path / "with_manifest"
    without_manifest = tmp_path / "without_manifest"
    with_manifest.mkdir()
    without_manifest.mkdir()

    lmdb_with = with_manifest / "data.lmdb"
    lmdb_without = without_manifest / "data.lmdb"
    _write_magic_lmdb(lmdb_with, EVENT_IDS, include_pointing=True)
    _write_magic_lmdb(lmdb_without, EVENT_IDS[:6], include_global=False)

    source = with_manifest / "src.parquet"
    _fake_parquet(source)
    reader = MAGICParquetReader(
        apply_cleaning=True,
        cleaning_n_low=6.0,
        real_timecal_centering="per_telescope_mean",
        real_timeslice_duration=1.25,
        mc_graft_timeslice_duration=0.6,
    )
    write_conversion_manifest(
        with_manifest,
        sources=[str(source)],
        reader=reader,
        extra={"is_mc": False},
    )

    out_with = tmp_path / "audit_with"
    out_without = tmp_path / "audit_without"
    return {
        "lmdb_with": lmdb_with,
        "lmdb_without": lmdb_without,
        "out_with": out_with,
        "out_without": out_without,
    }


def test_audit_end_to_end_with_and_without_manifest(
    audit_dirs: dict[str, Path],
) -> None:
    report_with = run_audit(
        audit_dirs["out_with"],
        lmdb_path=str(audit_dirs["lmdb_with"]),
        sample_events=8,
        seed=0,
    )
    audit_json = audit_dirs["out_with"] / "audit.json"
    assert audit_json.is_file()
    loaded = json.loads(audit_json.read_text(encoding="utf-8"))
    assert loaded["audit_version"] == 1
    assert "datasets" in loaded
    assert "direct" in loaded["datasets"]

    ds = loaded["datasets"]["direct"]
    for key in (
        "manifest",
        "feature_quantiles",
        "node_time_distributions",
        "pulse_image_distributions",
        "graph_definition",
        "edge_statistics",
        "source_reference_by_run",
        "prediction_position_by_run",
        "sky_coordinates",
        "event_ids",
    ):
        assert key in ds

    manifest = ds["manifest"]
    assert manifest["present"] is True
    assert manifest["is_mc"] is False
    assert manifest["parquet_schema_version"] == 6

    fq = ds["feature_quantiles"]
    assert "per_telescope" in fq
    assert fq["per_telescope"]["tel_0"]["signal"]["raw"]["p50"] == fq["per_telescope"]["tel_0"]["signal"]["raw"]["p50"]
    assert fq["per_telescope"]["tel_0"]["signal"]["n_nodes"] > 0

    assert ds["event_ids"]["status"] == "PASS"
    assert ds["event_ids"]["dtype"] == "int64"

    assert (audit_dirs["out_with"] / "direct_feature_quantiles.parquet").is_file()
    assert (audit_dirs["out_with"] / "direct_edge_stats.parquet").is_file()
    assert (audit_dirs["out_with"] / "command_snapshot.json").is_file()

    report_without = run_audit(
        audit_dirs["out_without"],
        lmdb_path=str(audit_dirs["lmdb_without"]),
        sample_events=4,
        seed=1,
    )
    _ = report_without
    loaded_no = json.loads(
        (audit_dirs["out_without"] / "audit.json").read_text(encoding="utf-8")
    )
    assert loaded_no["datasets"]["direct"]["manifest"]["present"] is False
    assert "legacy" in loaded_no["datasets"]["direct"]["manifest"]["status"]


def test_main_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
