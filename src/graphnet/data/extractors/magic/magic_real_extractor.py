"""Extractors and column defaults for MAGIC real (non-MC) parquet events."""

from __future__ import annotations

from typing import List

from .magic_extractor import MAGICExtractor
from .magic_mc_extractor import (
    MAGICMCGlobalExtractor,
    MAGICMCPulseExtractor,
    MAGICMCTruthExtractor,
)


# v5 MExportParquet: event / DAQ / source / telescope metadata (real stereo).
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
    "particle_id",
    "run_number",
    "source_dec_arcsec",
    "source_dec_deg",
    "source_ra_hours",
    "source_ra_timelike_sec",
    "stereo_evt_number",
    "subrun_index",
    "tel_zref_cam_x_deg",
    "tel_zref_cam_x_mm",
    "tel_zref_cam_y_deg",
    "tel_zref_cam_y_mm",
    "telescope_dec_arcsec",
    "telescope_dec_deg",
    "telescope_ra_hours",
    "telescope_ra_timelike_sec",
]

# Pointing, drive, and schema / time context shared with MC naming where applicable.
DEFAULT_MAGIC_REAL_GLOBAL_COLUMNS = [
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
