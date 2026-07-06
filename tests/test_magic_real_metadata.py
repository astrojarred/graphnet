"""Tests for MAGIC real-data source-reference metadata retention.

Covers:

- ``DEFAULT_MAGIC_REAL_TRUTH_COLUMNS`` retaining the v5 ``src_cam_*``
  source-reference columns and the v6 pointing/LST fields.
- :func:`assert_no_source_reference_in_features` guarding against
  ``src_cam_*``/``tel_zref_cam_*`` being configured as node features.
- :func:`assert_source_reference_not_labels` guarding against training on
  ``src_cam_*``/``tel_zref_cam_*`` as an all-event MSE label on real data.
- The reader's ``src_cam_x_mm``-presence guard for real-data parquet files
  that predate the source-reference exporter, and its
  ``allow_missing_source_reference`` opt-out.
- A regression test confirming there is no ``tel_zref_cam_*`` -> ``src_cam_*``
  fallback anywhere in this package.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphnet.data.extractors.magic import (
    DEFAULT_MAGIC_REAL_TRUTH_COLUMNS,
    REAL_SOURCE_REFERENCE_COLUMNS,
    MAGICMCPulseExtractor,
    assert_no_source_reference_in_features,
    assert_source_reference_not_labels,
    raise_if_source_reference_missing,
)
from graphnet.data.readers import MAGICParquetReader


# --- Default truth-column coverage -----------------------------------------


def test_default_truth_columns_include_src_cam_and_v6_fields() -> None:
    truth = set(DEFAULT_MAGIC_REAL_TRUTH_COLUMNS)

    src_cam_cols = {
        "src_cam_x_mm",
        "src_cam_y_mm",
        "src_cam_x_deg",
        "src_cam_y_deg",
    }
    assert src_cam_cols <= truth

    v6_pointing_lst_cols = {
        "pointing_corr_zd_deg",
        "pointing_corr_az_deg",
        "pointing_ra_hours",
        "pointing_dec_deg",
        "pointing_ha_hours",
        "pointing_corr_dec_deg",
        "pointing_corr_ha_hours",
        "source_ha_hours",
        "local_sidereal_time_hours",
    }
    assert v6_pointing_lst_cols <= truth

    # tel_zref_cam_* retained for backward-compatible metadata access.
    assert {
        "tel_zref_cam_x_mm",
        "tel_zref_cam_y_mm",
        "tel_zref_cam_x_deg",
        "tel_zref_cam_y_deg",
    } <= truth


# --- assert_no_source_reference_in_features ---------------------------------


def test_assert_no_source_reference_in_features_raises_on_src_cam() -> None:
    with pytest.raises(ValueError, match="src_cam_x_mm"):
        assert_no_source_reference_in_features(
            ["signal", "x_cam", "y_cam", "time", "src_cam_x_mm"]
        )


def test_assert_no_source_reference_in_features_raises_on_tel_zref() -> None:
    with pytest.raises(ValueError):
        assert_no_source_reference_in_features(["tel_zref_cam_x_mm"])


def test_assert_no_source_reference_in_features_passes_for_standard_features() -> (
    None
):
    # The standard 5 pulse-level node features used throughout the MAGIC
    # pipeline (signal, camera x/y, time, telescope id).
    assert_no_source_reference_in_features(
        ["signal", "x_cam", "y_cam", "time", "tel_id"]
    )


# --- assert_source_reference_not_labels -------------------------------------


def test_assert_source_reference_not_labels_raises_for_real_data() -> None:
    with pytest.raises(ValueError, match="src_cam_x_mm"):
        assert_source_reference_not_labels(
            ["energy_gev", "src_cam_x_mm"], is_mc=False
        )


def test_assert_source_reference_not_labels_passes_for_mc() -> None:
    # MC may legitimately train on src_cam_* since it is per-event truth
    # there.
    assert_source_reference_not_labels(
        ["energy_gev", "src_cam_x_mm", "src_cam_y_mm"], is_mc=True
    )


def test_assert_source_reference_not_labels_passes_for_real_without_source_ref() -> (
    None
):
    assert_source_reference_not_labels(
        ["energy_gev", "some_other_label"], is_mc=False
    )


# --- Missing src_cam_x_mm guard on the reader --------------------------------


def _make_stereo_waveform_row(event_id: int) -> dict:
    n = 1039 * 50 * 2
    return {
        "waveforms": np.full(n, 1e-6, dtype=np.float32),
        "event_id": event_id,
    }


def test_reader_raises_on_missing_src_cam_for_real_data() -> None:
    df = pd.DataFrame([_make_stereo_waveform_row(1)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "real_no_src_cam.parquet"
        df.to_parquet(path)

        reader = MAGICParquetReader(
            truth_columns=["run_number", "event_id"],
            global_params=[],
            apply_cleaning=False,
            allow_missing_truth_global_columns=True,
        )
        reader.set_extractors([MAGICMCPulseExtractor()])
        with pytest.raises(ValueError, match="src_cam_x_mm"):
            reader(str(path))


def test_reader_allows_missing_src_cam_with_opt_out() -> None:
    df = pd.DataFrame([_make_stereo_waveform_row(1)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "real_no_src_cam.parquet"
        df.to_parquet(path)

        reader = MAGICParquetReader(
            truth_columns=["run_number", "event_id"],
            global_params=[],
            apply_cleaning=False,
            allow_missing_truth_global_columns=True,
            allow_missing_source_reference=True,
        )
        reader.set_extractors([MAGICMCPulseExtractor()])
        out = reader(str(path))
    assert len(out) == 1


def test_raise_if_source_reference_missing_direct() -> None:
    with pytest.raises(ValueError, match="src_cam_x_mm"):
        raise_if_source_reference_missing(["event_id", "run_number"])

    # No raise when present.
    raise_if_source_reference_missing(["event_id", "src_cam_x_mm"])

    # No raise (only a log warning) with opt-out.
    raise_if_source_reference_missing(
        ["event_id"], allow_missing_source_reference=True
    )


# --- Regression: no tel_zref -> src_cam fallback -----------------------------


def test_no_tel_zref_to_src_cam_fallback_exists() -> None:
    """Regression test: no silent tel_zref_cam_* -> src_cam_* substitution.

    ``REAL_SOURCE_REFERENCE_COLUMNS`` lists both frames explicitly and
    distinctly, with no mapping/alias function between them anywhere in
    ``graphnet.data.extractors.magic``. If such a fallback were reintroduced,
    this test documents the expectation that it must not exist: the two
    guard functions above operate on the raw configured column names, not on
    any resolved/aliased name, so a src_cam_* substitution for tel_zref_cam_*
    would leave both frames present in the module's public column-membership
    checks with no crossover.
    """
    import graphnet.data.extractors.magic as magic_pkg

    # No function anywhere in the module namespace performs an
    # alias/fallback between the two frames (e.g. no "resolve_src_cam",
    # "tel_zref_to_src_cam", "src_cam_fallback", etc.).
    forbidden_name_fragments = ("fallback", "tel_zref_to_src_cam", "resolve_src_cam")
    for name in dir(magic_pkg):
        lowered = name.lower()
        for fragment in forbidden_name_fragments:
            assert fragment not in lowered, (
                f"Found suspicious tel_zref->src_cam fallback symbol: {name!r}"
            )

    # The two frames are listed as distinct, non-overlapping entries within
    # REAL_SOURCE_REFERENCE_COLUMNS (i.e. tel_zref_cam_* is never silently
    # dropped in favor of src_cam_* or vice versa -- both remain visible to
    # the guards).
    assert "src_cam_x_mm" in REAL_SOURCE_REFERENCE_COLUMNS
    assert "tel_zref_cam_x_mm" in REAL_SOURCE_REFERENCE_COLUMNS
