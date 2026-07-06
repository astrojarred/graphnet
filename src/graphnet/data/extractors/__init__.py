"""Module containing data-specific extractor modules."""

from .extractor import Extractor
from .combine_extractors import CombinedExtractor
from .internal import ParquetExtractor, SQLiteExtractor
from .magic import (
    MAGICExtractor,
    MAGICMCPulseExtractor,
    MAGICMCTruthExtractor,
    MAGICMCGlobalExtractor,
    DEFAULT_MAGIC_REAL_GLOBAL_COLUMNS,
    DEFAULT_MAGIC_REAL_TRUTH_COLUMNS,
    REAL_SOURCE_REFERENCE_COLUMNS,
    assert_no_source_reference_in_features,
    assert_source_reference_not_labels,
    default_magic_mc_extractors,
    default_magic_real_extractors,
    raise_if_source_reference_missing,
)
