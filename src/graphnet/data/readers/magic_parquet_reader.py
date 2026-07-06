"""Reader for raw MAGIC MC parquet files."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from graphnet.data.extractors.magic import (
    MAGICExtractor,
    clean_magic_event,
    load_or_build_default_px_py,
    log_size_clipped_from_row,
    raise_if_source_reference_missing,
)
from graphnet.data.extractors.magic.calibration import (
    VALID_TIMECAL_CENTERING,
    TimecalLookup,
)
from graphnet.data.extractors.magic.cleaning import resolve_mc_graft_duration
from .graphnet_file_reader import GraphNeTFileReader


# v5 MExportParquet names on merged stereo waveform rows (MC and real share these).
DEFAULT_MC_TRUTH_COLUMNS = [
    "particle_id",
    "energy_gev",
    "mc_shower_theta_rad",
    "mc_shower_phi_rad",
    "z_first_interaction_cm",
    "src_cam_x_mm",
    "src_cam_y_mm",
    "core_x_cm",
    "core_y_cm",
]

DEFAULT_GLOBAL_COLUMNS = [
    "mc_telescope_theta_rad",
    "mc_telescope_phi_rad",
    "pointing_zd_deg",
    "pointing_az_deg",
    "mjd",
]


class MAGICParquetReader(GraphNeTFileReader):
    """Reader for raw MAGIC MC parquet files exported by MARS.

    Optional ``timecal_graft_lmdb`` opens a :class:`~graphnet.data.extractors.magic.calibration.TimecalLookup`
    LMDB; each event's MC waveforms are grafted with real timecal (see
    :func:`~graphnet.data.extractors.magic.calibration.graft_mc_telescope_signal`)
    inside :func:`~graphnet.data.extractors.magic.cleaning.clean_magic_event` before
    cleaning. Call :meth:`close` when done to release the LMDB environment.

    The LMDB is opened **lazily** on first :meth:`__call__` in each process so the
    reader stays picklable for :class:`~graphnet.data.dataconverter.DataConverter`
    multiprocessing workers.

    Set ``allow_missing_truth_global_columns=True`` when using broad real-data
    column lists so optional or version-specific parquet fields can be absent
    without failing the whole file (missing names are skipped with a log
    warning).

    Three orthogonal timing controls are threaded down into
    :func:`~graphnet.data.extractors.magic.cleaning.clean_magic_event`:

    - ``real_timecal_centering`` (``"none"`` | ``"per_telescope_mean"``, default
      ``"none"``): timing origin of the real ``timecal_M1``/``timecal_M2`` path.
    - ``real_timeslice_duration`` (default ``1.0``): sample stride (ns) on the
      real path.
    - ``mc_graft_timeslice_duration`` (default ``0.6``): sample stride (ns) for
      MC timecal grafting. ``graft_timeslice_ns`` is a deprecated alias.

    Defaults reproduce the legacy behavior exactly. The effective values are
    exposed on the instance as ``self.timing_settings`` (a plain dict) so a
    conversion manifest can record them.
    """

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
        max_log_size_clipped: Optional[float] = 4.75,
        timecal_graft_lmdb: Optional[Union[str, Path]] = None,
        mc_graft_timeslice_duration: Optional[float] = None,
        real_timecal_centering: str = "none",
        real_timeslice_duration: float = 1.0,
        graft_log_interpolation: bool = False,
        graft_mod_shift: int = 0,
        graft_map_size_gb: float = 8.0,
        allow_missing_truth_global_columns: bool = False,
        graft_timeslice_ns: Optional[float] = None,
        is_mc: Optional[bool] = None,
        allow_missing_source_reference: bool = False,
    ) -> None:
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._index_column = index_column
        self._apply_cleaning = apply_cleaning
        self._cleaning_n_low = cleaning_n_low
        self._max_log_size_clipped = max_log_size_clipped
        self._allow_missing_truth_global_columns = allow_missing_truth_global_columns
        self._global_params = (
            global_params if global_params is not None else DEFAULT_GLOBAL_COLUMNS
        )
        self._truth_columns = (
            truth_columns if truth_columns is not None else DEFAULT_MC_TRUTH_COLUMNS
        )
        # Whether this reader is configured for MC data. Used only to gate
        # the real-data source-reference guard below (see
        # ``allow_missing_source_reference``); it does not otherwise change
        # reader behavior. When not given explicitly, conservatively defaults
        # to MC (no guard) UNLESS a real-only marker column ("run_number",
        # present in ``DEFAULT_MAGIC_REAL_TRUTH_COLUMNS`` but not in
        # ``DEFAULT_MC_TRUTH_COLUMNS``) is among the configured truth columns
        # -- this avoids false positives on ad hoc/minimal truth-column
        # configurations (e.g. ``truth_columns=[]`` in unit tests unrelated
        # to real-data source-reference handling).
        self._is_mc = (
            is_mc if is_mc is not None else ("run_number" not in self._truth_columns)
        )
        self._allow_missing_source_reference = allow_missing_source_reference

        default_px, default_py = load_or_build_default_px_py()
        self._px = default_px if px is None else px
        self._py = default_py if py is None else py

        # Orthogonal timing controls (see class docstring). ``graft_timeslice_ns``
        # is a deprecated alias for ``mc_graft_timeslice_duration``.
        if real_timecal_centering not in VALID_TIMECAL_CENTERING:
            raise ValueError(
                f"unknown real_timecal_centering {real_timecal_centering!r}; "
                f"expected one of {VALID_TIMECAL_CENTERING}"
            )
        self._real_timecal_centering = real_timecal_centering
        self._real_timeslice_duration = float(real_timeslice_duration)
        self._mc_graft_timeslice_duration = resolve_mc_graft_duration(
            mc_graft_timeslice_duration, graft_timeslice_ns
        )
        # Effective timing settings, exposed for a later conversion manifest.
        self.timing_settings: Dict[str, Any] = {
            "real_timecal_centering": self._real_timecal_centering,
            "real_timeslice_duration": self._real_timeslice_duration,
            "mc_graft_timeslice_duration": self._mc_graft_timeslice_duration,
        }
        self._graft_log_interpolation = graft_log_interpolation
        self._graft_map_size_gb = graft_map_size_gb
        self._graft_mod_shift = graft_mod_shift
        self._timecal_graft_lmdb: Optional[str] = None
        self._graft_timecal_lookup: Optional[TimecalLookup] = None
        if timecal_graft_lmdb is not None:
            self._timecal_graft_lmdb = str(
                Path(timecal_graft_lmdb).expanduser().resolve()
            )

    def _graft_lookup_lazy(self) -> Optional[TimecalLookup]:
        """Return open :class:`TimecalLookup`, opening it once per process if needed."""
        if self._timecal_graft_lmdb is None:
            return None
        if self._graft_timecal_lookup is None:
            self._graft_timecal_lookup = TimecalLookup(
                self._timecal_graft_lmdb,
                map_size_gb=self._graft_map_size_gb,
                mod_shift=self._graft_mod_shift,
            )
        return self._graft_timecal_lookup

    def close(self) -> None:
        """Close the optional timecal graft LMDB handle."""
        if self._graft_timecal_lookup is not None:
            self._graft_timecal_lookup.close()
            self._graft_timecal_lookup = None

    def __getstate__(self) -> Dict[str, Any]:
        """Drop unpickleable LMDB handle so workers can unpickle and reopen lazily."""
        state = self.__dict__.copy()
        state["_graft_timecal_lookup"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __call__(
        self,
        file_path: str,
    ) -> List[OrderedDict[str, Dict[str, Any]]]:
        """Read one MAGIC parquet file and apply configured extractors."""
        df = pd.read_parquet(file_path)
        if not self._is_mc:
            raise_if_source_reference_missing(
                df.columns,
                allow_missing_source_reference=self._allow_missing_source_reference,
                source_description=f"real-data parquet file {file_path!r}",
            )
        outputs: List[OrderedDict[str, Dict[str, Any]]] = []

        for _, row in df.iterrows():
            if self._max_log_size_clipped is not None:
                lsc = log_size_clipped_from_row(row)
                if np.isfinite(lsc) and lsc > self._max_log_size_clipped:
                    continue
            cleaned = clean_magic_event(
                row=row,
                apply_cleaning=self._apply_cleaning,
                cleaning_n_low=self._cleaning_n_low,
                px=self._px,
                py=self._py,
                index_column=self._index_column,
                global_params=self._global_params,
                truth_columns=self._truth_columns,
                graft_lookup=self._graft_lookup_lazy(),
                mc_graft_timeslice_duration=self._mc_graft_timeslice_duration,
                real_timecal_centering=self._real_timecal_centering,
                real_timeslice_duration=self._real_timeslice_duration,
                graft_log_interpolation=self._graft_log_interpolation,
                allow_missing_truth_global_columns=self._allow_missing_truth_global_columns,
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
