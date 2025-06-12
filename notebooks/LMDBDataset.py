# dl0/LMDBDataset.py
import io
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import lmdb
import torch
from torch import Tensor
from torch_geometric.data.database import (  # Assuming this is the intended base
    Database,
    Schema,
)


class LMDBDataset(Database):
    r"""An index-based key/value database based on LMDB.
    Modified for lazy opening to support PyTorch DataLoader with num_workers > 0.
    """

    def __init__(
        self,
        path: str | Path,
        schema: Schema = object,  # Keep if truly inheriting from PyG Database
        map_size: int = int(1e12),
        readonly: bool = False,
    ) -> None:
        super().__init__(schema)  # Keep if truly inheriting

        self.path_str = str(Path(path).absolute())  # Store path as string
        self.map_size = map_size
        self.readonly = readonly

        self._env: Optional[lmdb.Environment] = None

        # Ensure parent directory exists for LMDB creation if subdir=True
        # LMDB itself will create the specific directory (e.g., path_str itself if it's the db dir)
        if not readonly:
            # If self.path_str is "/A/B/C.lmdb", Path(self.path_str).parent is "/A/B"
            # If lmdb.open uses subdir=True, it means self.path_str is the directory.
            # So, we ensure self.path_str's parent exists, and lmdb.open(self.path_str, subdir=True)
            # will create self.path_str if it doesn't exist.
            db_dir = Path(self.path_str)
            if not db_dir.parent.exists():
                db_dir.parent.mkdir(parents=True, exist_ok=True)
            # If db_dir itself needs to be created by lmdb.open, this is fine.

    def _lazy_connect(self) -> None:
        """Connect to the LMDB database if not already connected."""
        if self._env is None:
            # If readonly and path doesn't exist, it's an error.
            # If not readonly, lmdb.open will create it.
            if self.readonly and not Path(self.path_str).exists():
                raise FileNotFoundError(
                    f"LMDB database not found at {self.path_str} (readonly mode)"
                )

            self._env = lmdb.open(
                self.path_str,  # path_str is the directory for LMDB files
                map_size=self.map_size,
                readonly=self.readonly,
                subdir=True,  # Assumes data.mdb and lock.mdb are in self.path_str
                lock=False if self.readonly else True,  # Disable lock for readonly
                readahead=False,  # Often better for random access patterns
                metasync=not self.readonly,  # Ensure metadata is synced for writes
                sync=not self.readonly,  # Ensure data is synced for writes
            )

    def close(self) -> None:
        """Close the connection to the database."""
        if self._env is not None:
            self._env.close()
            self._env = None

    # Custom pickling methods to handle multiprocessing
    def __getstate__(self):
        """Prepare the object for pickling. Don't pickle the LMDB environment."""
        state = self.__dict__.copy()
        state["_env"] = None  # LMDB environment is not picklable
        return state

    def __setstate__(self, state):
        """Restore the object after unpickling."""
        self.__dict__.update(state)
        self._env = None  # Ensure _env is None, will be re-established by _lazy_connect

    @staticmethod
    def to_key(index: int) -> bytes:
        return str(index).encode("utf-8")

    def insert(self, index: int, data: Any) -> None:
        if self.readonly:
            raise RuntimeError("Cannot insert into readonly database")
        self._lazy_connect()
        with self._env.begin(write=True) as txn:
            serialized_data = self._serialize(data)
            txn.put(self.to_key(index), serialized_data)

    def get(self, index: int) -> Any:
        self._lazy_connect()
        with self._env.begin(buffers=True) as txn:
            serialized_data = txn.get(self.to_key(index))
            if serialized_data is None:
                raise IndexError(
                    f"Index {index} not found in database at {self.path_str}"
                )
            return self._deserialize(serialized_data)

    def _multi_get(self, indices: Union[Sequence[int], Tensor]) -> List[Any]:
        self._lazy_connect()
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        data_list = []
        with self._env.begin(buffers=True) as txn:
            for index in indices:
                serialized_data = txn.get(self.to_key(index))
                if serialized_data is None:
                    raise IndexError(
                        f"Index {index} not found in database at {self.path_str}"
                    )
                data_list.append(self._deserialize(serialized_data))
        return data_list

    def _multi_insert(
        self,
        indices: Union[Sequence[int], Tensor],
        data_list: Sequence[Any],
        # Add batch_size and log for compatibility with user's original add_parquet_files
        batch_size: int = 1000,
        log: bool = False,
    ) -> None:
        if self.readonly:
            raise RuntimeError("Cannot insert into readonly database")
        self._lazy_connect()
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        # For large inserts, commit periodically or use a single large transaction
        # The original code implies batch_size is for the calling function, not LMDB batching
        with self._env.begin(write=True) as txn:
            for index, data in zip(indices, data_list):
                serialized_data = self._serialize(data)
                txn.put(self.to_key(index), serialized_data)

    def __len__(self) -> int:
        self._lazy_connect()
        with self._env.begin() as txn:  # Read-only transaction for stat
            length = txn.stat().get("entries")
            return length if length is not None else 0

    def keys(self) -> List[int]:
        self._lazy_connect()
        keys_list = []
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                # Assuming keys are simple integers encoded as utf-8 strings
                try:
                    keys_list.append(int(key_bytes.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    # Skip keys that are not simple integer strings (e.g., metadata)
                    continue
        return sorted(keys_list)

    def exists(self, index: int) -> bool:
        self._lazy_connect()
        with self._env.begin() as txn:
            return txn.get(self.to_key(index)) is not None

    def delete(self, index: int) -> bool:
        if self.readonly:
            raise RuntimeError("Cannot delete from readonly database")
        self._lazy_connect()
        with self._env.begin(write=True) as txn:
            return txn.delete(self.to_key(index))

    def _serialize(self, row: Any) -> bytes:
        if isinstance(row, Tensor):
            row = row.clone()
        buffer = io.BytesIO()
        torch.save(row, buffer)
        return buffer.getvalue()

    def _deserialize(self, row: bytes) -> Any:
        return torch.load(io.BytesIO(row), weights_only=False)

    def __enter__(self):
        # _lazy_connect will be called by operations, no need to connect here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        # Try to get length without failing if DB not accessible yet in main process
        length_str = "unknown (connect to get length)"
        if self._env is not None:  # Or try: Path(self.path_str).exists():
            try:
                length_str = str(len(self))
            except Exception as e:  # pylint: disable=bare-except
                print(f"Warning: Error while getting length: {e}")  # pragma: no cover
                pass
        return (
            f"{self.__class__.__name__}("
            f"path={self.path_str}, "
            f"readonly={self.readonly}, "
            f"length={length_str})"
        )
