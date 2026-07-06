"""Extractors for MAGIC data."""

from .magic_extractor import MAGICExtractor
from .magic_mc_extractor import (
    MAGICMCPulseExtractor,
    MAGICMCTruthExtractor,
    MAGICMCGlobalExtractor,
    default_magic_mc_extractors,
)
from .magic_real_extractor import (
    DEFAULT_MAGIC_REAL_GLOBAL_COLUMNS,
    DEFAULT_MAGIC_REAL_TRUTH_COLUMNS,
    REAL_SOURCE_REFERENCE_COLUMNS,
    assert_no_source_reference_in_features,
    assert_source_reference_not_labels,
    default_magic_real_extractors,
    raise_if_source_reference_missing,
)
from .cleaning import (
    clean_magic_event,
    load_or_build_default_px_py,
    log_size_clipped_from_row,
)
from .calibration import (
    TIMECAL_HEADER,
    TimecalLookup,
    build_magic_timecal_lmdb,
    decode_timecal_row,
    encode_timecal_row,
    expand_parquet_sources,
    graft_mc_telescope_signal,
    interp1d_shared_xp_batch,
    shift_signal_graft,
    timecal_concat,
)
