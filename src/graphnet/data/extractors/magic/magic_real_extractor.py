"""Extractors and column defaults for MAGIC real (non-MC) parquet events.

Two families of per-event fields carry camera-plane source information and
must never be confused with each other, nor with per-event *training truth*:

- ``tel_zref_cam_x/y_mm/deg``: the source position projected into a
  **zenith-aligned auxiliary camera frame** (MARS ``TelZref`` convention).
  This is an auxiliary geometric reference frame only, NOT a source-truth
  label, and NOT the same frame as ``src_cam_x/y``.
- ``src_cam_x/y_mm/deg`` (v5+ real exports): the catalogue source position
  projected into the **actual event camera frame** at the time of the event.
  On real data this is a **source reference** derived from the known,
  catalogue source position and the telescope's pointing/tracking model --
  it is used for wobble-offset validation, on/off region definition, and
  theta-squared analyses. It is emphatically **not** a per-event
  reconstruction truth label: real data has no per-event ground truth
  direction. (On MC, the analogous ``src_cam_x/y_mm`` *is* per-event truth,
  since the simulated shower's true direction is known.)

Because of this real/MC asymmetry, any code path that would use
``src_cam_*``/``tel_zref_cam_*`` as an all-event MSE training target, or as a
network input feature (which would leak the answer for on-source studies),
is a bug. See :func:`assert_no_source_reference_in_features` and
:func:`assert_source_reference_not_labels` below, which guard against exactly
these two mistakes, and
:func:`raise_if_source_reference_missing` / ``allow_missing_source_reference``
on :class:`~graphnet.data.readers.magic_parquet_reader.MAGICParquetReader`,
which guards against silently processing pre-source-reference-export real
files.

No fallback from ``tel_zref_cam_*`` to ``src_cam_*`` (or vice versa) is
implemented anywhere in this package: they are different reference frames
and substituting one for the other silently would corrupt downstream wobble
/ theta-squared / on-off analyses. If ``src_cam_*`` is unavailable, callers
must either supply it (re-run the MARS exporter) or explicitly opt out via
``allow_missing_source_reference=True`` (which only downgrades the missing
export to a logged warning; it never substitutes ``tel_zref_cam_*``).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from .magic_extractor import MAGICExtractor
from .magic_mc_extractor import (
    MAGICMCGlobalExtractor,
    MAGICMCPulseExtractor,
    MAGICMCTruthExtractor,
)

_logger = logging.getLogger(__name__)


#: Camera-plane source-reference columns. On real data these are a *source
#: reference* (known catalogue position projected via the pointing model),
#: never a per-event reconstruction truth label; see the module docstring.
#: ``tel_zref_cam_*`` is included because it is the same kind of
#: pointing/geometry-derived camera-plane quantity and must be subject to the
#: same "not a feature, not a real-data label" guards.
REAL_SOURCE_REFERENCE_COLUMNS: tuple[str, ...] = (
    "src_cam_x_mm",
    "src_cam_y_mm",
    "src_cam_x_deg",
    "src_cam_y_deg",
    "tel_zref_cam_x_mm",
    "tel_zref_cam_y_mm",
    "tel_zref_cam_x_deg",
    "tel_zref_cam_y_deg",
)


def assert_no_source_reference_in_features(
    feature_columns: Iterable[str],
) -> None:
    """Raise if any source-reference column is configured as a node feature.

    ``src_cam_*``/``tel_zref_cam_*`` encode the (known, catalogue) source
    position. Using them as pulse-level node features would leak the
    on-source direction directly into the model input -- this function is a
    guard against that configuration mistake.

    Args:
        feature_columns: Configured node/pulse feature column names.

    Raises:
        ValueError: If any name in ``feature_columns`` is a member of
            :data:`REAL_SOURCE_REFERENCE_COLUMNS`.
    """
    offending = sorted(
        set(feature_columns) & set(REAL_SOURCE_REFERENCE_COLUMNS)
    )
    if offending:
        raise ValueError(
            "Source-reference column(s) configured as node feature(s): "
            f"{offending}. src_cam_*/tel_zref_cam_* encode the known "
            "catalogue source position (a reference for wobble/theta-squared "
            "analyses, not a per-event pulse-level observable) and must "
            "never be used as model input features."
        )


def assert_source_reference_not_labels(
    truth_label_names: Iterable[str],
    is_mc: bool,
) -> None:
    """Raise if a real-data label configuration trains on source-reference truth.

    On MC, ``src_cam_x/y_mm`` (and ``_deg``) are legitimate per-event truth
    (the simulated shower's true direction is known), so they may be used as
    training labels. On REAL data there is no per-event reconstruction truth:
    ``src_cam_*``/``tel_zref_cam_*`` are only a *source reference* (the known
    catalogue source position projected through the pointing model), used for
    wobble-offset validation, on/off region definitions, and theta-squared
    analyses -- never an all-event MSE regression target.

    Args:
        truth_label_names: Configured training-label column names.
        is_mc: Whether the labels are being configured for an MC dataset.
            When ``False`` (real data), no name in
            :data:`REAL_SOURCE_REFERENCE_COLUMNS` may appear.

    Raises:
        ValueError: If ``is_mc`` is ``False`` and any name in
            ``truth_label_names`` is a member of
            :data:`REAL_SOURCE_REFERENCE_COLUMNS`.
    """
    if is_mc:
        return
    offending = sorted(
        set(truth_label_names) & set(REAL_SOURCE_REFERENCE_COLUMNS)
    )
    if offending:
        raise ValueError(
            "Source-reference column(s) configured as training label(s) on "
            f"REAL data: {offending}. On real data, src_cam_*/tel_zref_cam_* "
            "are a source REFERENCE (known catalogue position via the "
            "pointing model; used for wobble validation, theta-squared, and "
            "on/off region definitions), not per-event reconstruction truth. "
            "They must never be used as an all-event MSE training label on "
            "real data. (MC is exempt: pass is_mc=True there, since the "
            "simulated direction is per-event truth.)"
        )


def raise_if_source_reference_missing(
    available_columns: Sequence[str],
    allow_missing_source_reference: bool = False,
    source_description: str = "input parquet file",
) -> None:
    """Guard against real-data files exported before source-reference support.

    Real MAGIC parquet files must carry ``src_cam_x_mm`` (schema v5+): the
    catalogue source position projected into the event camera frame via the
    pointing model. Older exports predate this and must be reconverted with a
    current MARS exporter; silently proceeding without it would make later
    wobble/theta-squared analyses impossible.

    Args:
        available_columns: Column names available in the real-data file
            (e.g. ``df.columns``).
        allow_missing_source_reference: If ``True``, downgrade the missing
            column to a logged warning instead of raising (only intended for
            legacy/test fixtures that predate the exporter).
        source_description: Human-readable description of the file/row used
            in the error/warning message.

    Raises:
        ValueError: If ``"src_cam_x_mm"`` is absent from ``available_columns``
            and ``allow_missing_source_reference`` is ``False``.
    """
    if "src_cam_x_mm" in available_columns:
        return
    message = (
        f"{source_description} lacks 'src_cam_x_mm': it predates the MARS "
        "exporter version that writes real-data source-reference metadata "
        "(schema v5+) and must be reconverted. Pass "
        "allow_missing_source_reference=True to downgrade this to a warning "
        "(only intended for legacy test fixtures)."
    )
    if allow_missing_source_reference:
        _logger.warning(message)
        return
    raise ValueError(message)


# v5 MExportParquet: event / DAQ / source / telescope metadata (real stereo).
# v6 adds real-data source-reference (src_cam_*) and pointing/LST columns.
DEFAULT_MAGIC_REAL_TRUTH_COLUMNS = [
    "event_id",
    "stream_event_id_M1",
    "stream_event_index_M1",
    "daq_event_number_M1",
    "trigger_pattern_unprescaled_M1",
    "is_l3_event_M1",
    "stream_event_id_M2",
    "stream_event_index_M2",
    "daq_event_number_M2",
    "trigger_pattern_unprescaled_M2",
    "is_l3_event_M2",
    "local_sidereal_time_hours",
    "particle_id",
    "pointing_corr_az_deg",
    "pointing_corr_dec_deg",
    "pointing_corr_ha_hours",
    "pointing_corr_zd_deg",
    "pointing_dec_deg",
    "pointing_ha_hours",
    "pointing_ra_hours",
    "run_number",
    "source_dec_arcsec",
    "source_dec_deg",
    "source_ha_hours",
    "source_ra_hours",
    "source_ra_timelike_sec",
    "src_cam_x_deg",
    "src_cam_x_mm",
    "src_cam_y_deg",
    "src_cam_y_mm",
    "stereo_evt_number",
    "subrun_index",
    # Zenith-aligned auxiliary frame (MARS TelZref) -- NOT source truth, and
    # NOT the same frame as src_cam_*. Kept for backward-compatible metadata
    # access only; see the module docstring.
    "tel_zref_cam_x_deg",
    "tel_zref_cam_x_mm",
    "tel_zref_cam_y_deg",
    "tel_zref_cam_y_mm",
    "telescope_dec_arcsec",
    "telescope_dec_deg",
    "telescope_ra_hours",
    "telescope_ra_timelike_sec",
]

# Pointing, drive, and schema / time context shared with MC naming where
# applicable. ``camera_dist_mm``/``camera_mm2deg`` are per-event exported
# fields (constant per telescope/epoch, but written on every event row by the
# v6 exporter) describing the camera geometry used to derive src_cam_* /
# tel_zref_cam_* from celestial coordinates; they are read from the same
# per-row parquet table as the other global/pointing context, so they belong
# here alongside pointing_az_deg/pointing_zd_deg rather than in the
# event/DAQ/source truth-columns list above.
DEFAULT_MAGIC_REAL_GLOBAL_COLUMNS = [
    "camera_dist_mm",
    "camera_mm2deg",
    "drive_current_az_deg",
    "drive_current_zd_deg",
    "drive_dec_deg",
    "drive_ha_hours",
    "drive_mjd",
    "drive_nominal_az_deg",
    "drive_nominal_zd_deg",
    "drive_ra_hours",
    "mjd",
    "parquet_schema_version",
    "pointing_az_deg",
    "pointing_zd_deg",
]


def default_magic_real_extractors() -> List[MAGICExtractor]:
    """Default extractor set for MAGIC real-data conversion.

    Use with :class:`~graphnet.data.readers.magic_parquet_reader.MAGICParquetReader`
    configured as::

        MAGICParquetReader(
            truth_columns=DEFAULT_MAGIC_REAL_TRUTH_COLUMNS,
            global_params=DEFAULT_MAGIC_REAL_GLOBAL_COLUMNS,
            allow_missing_truth_global_columns=True,
        )
    """
    return [
        MAGICMCPulseExtractor(extractor_name="MAGICPixels"),
        MAGICMCTruthExtractor(extractor_name="truth"),
        MAGICMCGlobalExtractor(extractor_name="global"),
    ]
