"""Reader for raw MAGIC MC parquet files."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from graphnet.data.extractors.magic import (
    MAGICExtractor,
    clean_magic_event,
    load_or_build_default_px_py,
)
from .graphnet_file_reader import GraphNeTFileReader


DEFAULT_MC_TRUTH_COLUMNS = [
    "particle_id",
    "energy",
    "theta",
    "phi",
    "z_first_interaction",
    "impact_M1",
    "impact_M2",
]

DEFAULT_GLOBAL_COLUMNS = [
    "telescope_theta",
    "telescope_phi",
]


class MAGICParquetReader(GraphNeTFileReader):
    """Reader for raw MAGIC MC parquet files exported by MARS."""

    _accepted_file_extensions = [".parquet"]
    _accepted_extractors = [MAGICExtractor]

    def __init__(
        self,
        index_column: Optional[str] = "event_id",
        apply_cleaning: bool = False,
        cleaning_n_low: float | None = None,
        global_params: Optional[List[str]] = None,
        truth_columns: Optional[List[str]] = None,
        px: Optional[Any] = None,
        py: Optional[Any] = None,
    ) -> None:
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._index_column = index_column
        self._apply_cleaning = apply_cleaning
        self._cleaning_n_low = cleaning_n_low
        self._global_params = (
            global_params if global_params is not None else DEFAULT_GLOBAL_COLUMNS
        )
        self._truth_columns = (
            truth_columns if truth_columns is not None else DEFAULT_MC_TRUTH_COLUMNS
        )

        default_px, default_py = load_or_build_default_px_py()
        self._px = default_px if px is None else px
        self._py = default_py if py is None else py

    def __call__(
        self,
        file_path: str,
    ) -> List[OrderedDict[str, Dict[str, Any]]]:
        """Read one MAGIC parquet file and apply configured extractors."""
        df = pd.read_parquet(file_path)
        outputs: List[OrderedDict[str, Dict[str, Any]]] = []

        for _, row in df.iterrows():
            cleaned = clean_magic_event(
                row=row,
                apply_cleaning=self._apply_cleaning,
                cleaning_n_low=self._cleaning_n_low,
                px=self._px,
                py=self._py,
                index_column=self._index_column,
                global_params=self._global_params,
                truth_columns=self._truth_columns,
            )
            event_output: OrderedDict[str, Dict[str, Any]] = OrderedDict()
            for extractor in self._extractors:
                extracted = extractor(cleaned)
                if extracted is not None:
                    event_output[extractor.name] = extracted
            outputs.append(event_output)
        return outputs

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search recursively for parquet files under the given path(s).

        The path can be a directory or a .parquet dataset directory.
        Finds all parquet files under the path(s) passed, not the parent.
        """
        found: List[Path] = []
        paths = [Path(path)] if isinstance(path, str) else [Path(p) for p in path]

        for p in paths:
            p = p.resolve()
            if p.is_file():
                if p.suffix == ".parquet":
                    found.append(p)
            elif p.is_dir():
                found.extend(p.rglob("*.parquet"))

        file_strs = sorted(str(f) for f in set(found))
        self.validate_files(file_strs)
        return file_strs
