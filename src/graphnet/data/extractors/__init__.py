"""Module containing data-specific extractor modules."""

from .extractor import Extractor
from .combine_extractors import CombinedExtractor
from .internal import ParquetExtractor, SQLiteExtractor
from .magic import (
    MAGICExtractor,
    MAGICMCPulseExtractor,
    MAGICMCTruthExtractor,
    MAGICMCGlobalExtractor,
    default_magic_mc_extractors,
)
