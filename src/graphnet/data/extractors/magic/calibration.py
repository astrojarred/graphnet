"""LMDB-backed time calibration (``timecal_M1`` / ``timecal_M2``) lookups for MAGIC parquet data.

Call :func:`build_magic_timecal_lmdb` to read timecal columns from parquet shards (or a single file,
glob, dataset directory, or nested list of those) and fill an LMDB database in one step — row-group
at a time, no full-table load.
"""

from __future__ import annotations

import json
import shutil
import struct
from glob import glob
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

__all__ = [
    "TIMECAL_HEADER",
    "TimecalLookup",
    "build_magic_timecal_lmdb",
    "decode_timecal_row",
    "encode_timecal_row",
    "expand_parquet_sources",
    "timecal_concat",
]

TIMECAL_HEADER = struct.Struct("<II")  # n_M1, n_M2 as uint32 (little-endian)


def encode_timecal_row(m1: Any, m2: Any) -> bytes:
    """Pack M1/M2 float32 ravel payloads with an 8-byte length header."""
    a1 = np.asarray(m1, dtype=np.float32).ravel()
    a2 = np.asarray(m2, dtype=np.float32).ravel()
    return TIMECAL_HEADER.pack(a1.size, a2.size) + a1.tobytes() + a2.tobytes()


def decode_timecal_row(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode payload from :func:`encode_timecal_row`; returns (timecal_M1, timecal_M2) float32 1-D."""
    n1, n2 = TIMECAL_HEADER.unpack_from(buf, 0)
    off = TIMECAL_HEADER.size
    mv = memoryview(buf)
    a1 = np.frombuffer(mv[off : off + 4 * n1], dtype=np.float32).copy()
    a2 = np.frombuffer(mv[off + 4 * n1 : off + 4 * (n1 + n2)], dtype=np.float32).copy()
    return a1, a2


def timecal_concat(m1: Any, m2: Any) -> np.ndarray:
    """Concatenate M1 and M2 as a single float32 vector (same order as the LMDB payload body)."""
    a1 = np.asarray(m1, dtype=np.float32).ravel()
    a2 = np.asarray(m2, dtype=np.float32).ravel()
    return np.concatenate([a1, a2])


def expand_parquet_sources(
    sources: str | Path | list[str] | tuple[str, ...],
) -> list[str]:
    """Resolve a path, glob, dataset directory, or shard list to concrete ``*.parquet`` paths.

    A path may be a single parquet file, or a directory (including names ending in ``.parquet``)
    containing one or more shard files (discovered with :func:`Path.rglob` ``*.parquet``).
    """
    if isinstance(sources, (list, tuple)):
        chunks: list[str] = []
        for item in sources:
            chunks.extend(expand_parquet_sources(item))
        if not chunks:
            raise ValueError("sources list is empty")
        return chunks
    s = str(sources)
    if any(ch in s for ch in "*?["):
        matched = sorted(glob(s))
        if not matched:
            raise FileNotFoundError(f"No parquet matched glob: {s!r}")
        return matched
    p = Path(s).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.is_dir():
        parts = sorted(p.rglob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No *.parquet files under directory: {p}")
        return [str(x.resolve()) for x in parts]
    if p.is_file():
        return [str(p)]
    raise FileNotFoundError(f"Not a file or directory: {p}")


def build_magic_timecal_lmdb(
    sources: str | Path | list[str] | tuple[str, ...],
    lmdb_dir: str | Path,
    *,
    id_column: str = "event_id",
    store_event_id_keys: bool = True,
    map_size_gb: float = 256.0,
    timecal_m1: str = "timecal_M1",
    timecal_m2: str = "timecal_M2",
) -> dict[str, Any]:
    """Build an LMDB of ``timecal_M1`` / ``timecal_M2`` from parquet inputs (streaming, row-group).

    ``sources`` is the same shape as :func:`expand_parquet_sources`: a file path, a directory of
    shards (including names ending in ``.parquet``), a glob, or a list/tuple of those (nested lists
    are flattened).

    Row keys ``i:{012d}`` follow global order: expanded path order, then row groups, then rows.
    Optional ``e:{{event_id}}`` keys duplicate the value when ``id_column`` exists in the schema
    and ``store_event_id_keys`` is true.

    ``map_size_gb`` is the LMDB map size limit; increase if you hit ``MapFullError``.
    """
    expanded = expand_parquet_sources(sources)
    paths = [Path(p).resolve() for p in expanded]

    def _field_names(parquet_path: Path) -> set[str]:
        return {f.name for f in pq.ParquetFile(parquet_path).schema_arrow}

    names0 = _field_names(paths[0])
    for col in (timecal_m1, timecal_m2):
        if col not in names0:
            raise ValueError(
                f"Missing column {col!r} in first shard {expanded[0]!r}"
            )
    id_for_lmdb: str | None = (
        id_column if (id_column in names0 and store_event_id_keys) else None
    )
    use_id = bool(id_for_lmdb)
    if use_id:
        for p in paths[1:]:
            if id_for_lmdb not in _field_names(p):
                raise ValueError(
                    f"Shard {p} missing {id_for_lmdb!r} (required for id lookup)"
                )

    lmdb_path = Path(lmdb_dir)
    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)
    lmdb_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(lmdb_path),
        map_size=int(map_size_gb * (1024**3)),
        subdir=True,
        max_dbs=0,
        readahead=False,
    )

    row_idx = 0
    dup_event_keys = 0
    seen_event_keys: set[bytes] = set()

    with env.begin(write=True) as txn:
        for path in paths:
            pf = pq.ParquetFile(path)
            cols = [timecal_m1, timecal_m2]
            if use_id:
                cols.append(id_for_lmdb)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=cols)
                df = pl.from_arrow(tbl)
                for row in df.iter_rows(named=True):
                    payload = encode_timecal_row(row[timecal_m1], row[timecal_m2])
                    txn.put(f"i:{row_idx:012d}".encode(), payload)
                    if use_id:
                        eid = row[id_for_lmdb]
                        if eid is not None:
                            ek = f"e:{eid}".encode()
                            if ek in seen_event_keys:
                                dup_event_keys += 1
                            else:
                                seen_event_keys.add(ek)
                                txn.put(ek, payload)
                    row_idx += 1

        meta = {
            "n_rows": row_idx,
            "id_lookup": use_id,
            "id_column": id_for_lmdb if use_id else None,
            "duplicate_event_keys": dup_event_keys,
            "parquet_inputs": [str(p) for p in paths],
            "map_size_gb": map_size_gb,
        }
        txn.put(b"__meta__", json.dumps(meta).encode())

    env.sync()
    env.close()
    return meta


class TimecalLookup:
    """Read-only LMDB access to packed ``timecal_M1`` / ``timecal_M2`` values.

    **Storage indices:** canonical row ids are ``0 .. n_rows-1`` (keys ``i:…``). :meth:`by_row` and
    :meth:`by_index` use that index only. Bracket :meth:`__getitem__` applies ``mod_shift`` then
    ``% n_rows`` (convenience only).

    After :meth:`close`, reads raise :class:`ValueError` until :meth:`reopen` (or construct a new
    instance). Use ``with TimecalLookup(...) as lookup:`` for deterministic cleanup.
    """

    def __init__(
        self,
        lmdb_dir: str | Path,
        *,
        map_size_gb: float = 8.0,
        mod_shift: int = 0,
    ) -> None:
        self._root = Path(lmdb_dir)
        self._map_size_gb = map_size_gb
        self._mod_shift = mod_shift
        self._env: lmdb.Environment | None = None
        self._closed = True
        self._open()

    def _open(self) -> None:
        root = self._root
        data_mdb = root / "data.mdb"
        floor = int(self._map_size_gb * (1024**3))
        map_size = max(floor, data_mdb.stat().st_size) if data_mdb.is_file() else floor
        self._env = lmdb.open(
            str(root),
            map_size=map_size,
            subdir=True,
            readonly=True,
            readahead=False,
            lock=False,
        )
        self._closed = False

    def _require_open(self) -> lmdb.Environment:
        if self._closed or self._env is None:
            raise ValueError(
                "TimecalLookup is closed; call reopen() or use a new instance"
            )
        return self._env

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        if self._env is not None:
            self._env.close()
            self._env = None
        self._closed = True

    def reopen(self) -> None:
        """Open the LMDB again after :meth:`close` (same path and ``map_size_gb`` as at init)."""
        if not self._closed:
            return
        self._open()

    def __enter__(self) -> TimecalLookup:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def meta(self) -> dict[str, Any]:
        with self._require_open().begin() as txn:
            raw = txn.get(b"__meta__")
        return json.loads(raw.decode()) if raw else {}

    def by_row(self, row: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Fetch by canonical 0-based storage row (LMDB ``i:{row}``)."""
        with self._require_open().begin() as txn:
            buf = txn.get(f"i:{row:012d}".encode())
        return decode_timecal_row(buf) if buf is not None else None

    def by_index(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Alias for :meth:`by_row`."""
        return self.by_row(index)

    def _bracket_to_storage_row(self, k: int) -> int:
        n = len(self)
        if n == 0:
            raise IndexError("empty lookup")
        return (k + self._mod_shift) % n

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not isinstance(index, int):
            raise TypeError("only int indices are supported")
        return self.by_row(self._bracket_to_storage_row(index))

    def __len__(self) -> int:
        return int(self.meta().get("n_rows", 0))

    def by_event_id(self, event_id: Any) -> tuple[np.ndarray, np.ndarray] | None:
        with self._require_open().begin() as txn:
            buf = txn.get(f"e:{event_id}".encode())
        return decode_timecal_row(buf) if buf is not None else None

    @staticmethod
    def concat(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(m1, dtype=np.float32).ravel(),
                np.asarray(m2, dtype=np.float32).ravel(),
            ]
        )
