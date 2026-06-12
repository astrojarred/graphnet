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
    default_magic_mc_extractors,
    default_magic_real_extractors,
)
