"""`Dataset` class(es) for reading data from LMDB databases."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.data.dataset.dataset import Dataset, ColumnMissingException


class LMDBDataset(Dataset):
    """LMDB dataset for pre-stored PyTorch Geometric Data objects.

    This dataset is designed for cases where Data objects are already
    stored in LMDB format, such as the MAGIC telescope data.
    It bypasses the table-based structure and reads Data objects directly.
    """

    def __init__(
        self,
        path: Union[str, Path],
        features: Optional[List[str]] = None,
        truth: Optional[List[str]] = None,
        *,
        pulsemaps: Union[str, List[str]] = ["data"],  # Default to placeholder value
        graph_definition: Optional[object] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        transform: Optional[object] = None,
        **kwargs,
    ):
        """Construct MAGICLMDBDataset.

        Args:
            path: Path to the LMDB database.
            features: List of node feature names (optional for pre-structured data).
            truth: List of truth variable names (optional for pre-structured data).
            pulsemaps: Not used for pre-structured data, but kept for compatibility.
            graph_definition: Not used for pre-structured data, but kept for compatibility.
            selection: Event selection criteria (string query, list of indices, etc.).
            transform: Optional transform to apply to each Data object (e.g., UnpackGlobalFeatures).
            **kwargs: Additional arguments passed to base class.
        """

        if not Path(path).exists():
            raise FileNotFoundError(f"LMDB database not found at {path}")

        # Convert Path object to string for DatasetConfig validation
        path = str(path)

        # pulsemaps already has a default value of ["data"] in the signature
        if features is None:
            features = []  # Will be determined from actual data
        if truth is None:
            truth = []  # Will be determined from actual data

        # Create a dummy graph definition if none provided
        # Since we work with pre-built Data objects, this won't actually be used
        if graph_definition is None:
            raise ValueError("graph_definition is required for LMDBDataset")

        # Store transform for later use
        self._transform = transform

        # Call parent constructor explicitly
        Dataset.__init__(
            self,
            path=path,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            graph_definition=graph_definition,
            selection=selection,
            **kwargs,
        )

        self._event_id_to_idx = {}
        self._idx_to_event_id = {}
        self._all_indices = self._get_all_indices()
        self._selection = selection

    @classmethod
    def _resolve_graphnet_paths(
        cls, path: Union[str, Path, List[Union[str, Path]]]
    ) -> Union[str, List[str]]:
        """Resolve GraphNeT path references, handling Path objects."""
        if isinstance(path, list):
            return [cls._resolve_graphnet_paths(p) for p in path]

        # Convert Path to string if needed
        if isinstance(path, Path):
            path = str(path)

        # Call parent implementation with string path
        return Dataset._resolve_graphnet_paths(path)

    def _init(self) -> None:
        # Check(s)
        assert isinstance(self._path, (str, Path)), (
            f"MAGIC LMDB dataset requires a single path, got {type(self._path)}"
        )

        # Convert to string for LMDB
        if isinstance(self._path, Path):
            self._path = str(self._path)

        # LMDB connection parameters
        self._map_size = int(1e12)  # 1TB max size
        self._readonly = True  # Dataset is read-only

        # LMDB environment (will be lazily initialized)
        self._env: Optional[lmdb.Environment] = None

    def _post_init(self) -> None:
        # Override to skip column checking since we work with pre-built Data objects
        self._missing_variables: Dict[str, List[str]] = {}
        self._close_connection()

    def _lazy_connect(self) -> None:
        """Connect to the LMDB database if not already connected."""
        if self._env is None:
            path_obj = Path(self._path)
            if not path_obj.exists():
                raise FileNotFoundError(f"LMDB database not found at {self._path}")

            self._env = lmdb.open(
                self._path,
                map_size=self._map_size,
                readonly=self._readonly,
                subdir=True,  # Treat path as directory containing data.mdb
                lock=False,  # Disable lock for readonly access
                readahead=False,  # Better for random access patterns
                metasync=False,  # Not needed for readonly
                sync=False,  # Not needed for readonly
            )

    def _close_connection(self) -> None:
        """Close the LMDB environment."""
        if not hasattr(self, "_env"):
            self._env = None

        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                print(f"Error closing LMDB environment: {e}")
                pass  # Ignore errors during cleanup
            self._env = None

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from LMDB storage."""
        return torch.load(io.BytesIO(data), weights_only=False)

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        """Query table - simplified for pre-stored Data objects."""

        if isinstance(columns, str):
            columns = [columns]

        if sequential_index is None:
            indices = np.arange(0, len(self) if max_length is None else max_length, 1)
        else:
            indices = [sequential_index]

        arrays = []
        for idx in indices:
            array = self._query_table(table, columns, idx, selection)
            arrays.append(array)

        return np.concatenate(arrays, axis=0)

    def _query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
        _ignore_selection: bool = False,
    ) -> np.ndarray:
        """Query table - simplified for pre-stored Data objects."""

        if sequential_index is None:
            return None

        if isinstance(columns, str):
            columns = [columns]

        if self._selection is not None and not _ignore_selection:
            pass

        data = self.get_data(sequential_index, _ignore_selection=_ignore_selection)

        for col in columns:
            if not hasattr(data, col):
                raise ColumnMissingException(f"Column {col} not found in data")

        # Extract the attributes and convert to numpy arrays
        # max_length = max(len(getattr(data, col)) for col in columns)
        max_length = 1
        for col in columns:
            if hasattr(getattr(data, col), 'shape'):
                max_length = max(max_length, getattr(data, col).shape[0])

        column_data = []

        for col in columns:
            attr_data = getattr(data, col)
            if isinstance(attr_data, int) or isinstance(attr_data, float):
                attr_data = torch.tensor([attr_data])

            if len(attr_data) == 1:
                attr_data = torch.repeat_interleave(attr_data, max_length)
            # Convert PyTorch tensors to numpy arrays if needed
            if hasattr(attr_data, 'numpy'):
                attr_data = attr_data.detach().cpu().numpy()
            elif hasattr(attr_data, 'cpu'):
                attr_data = attr_data.cpu().numpy()
            elif not isinstance(attr_data, np.ndarray):
                attr_data = np.array(attr_data)
            
            # Ensure numeric types for columns that should be numeric (like event_id)
            if col == self._index_column and attr_data.dtype == object:
                attr_data = attr_data.astype(np.int64)

            column_data.append(attr_data)
        
        # Use appropriate dtype inference
        result = np.column_stack(column_data) if column_data else np.array([]).reshape(0, len(columns))
        return result

    def _query(
        self, sequential_index: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """Query file for event features and truth information.

        The returned lists have lengths corresponding to the number of pulses
        in the event. Their constituent tuples have lengths corresponding to
        the number of features/attributes in each output

        Args:
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`.

        Returns:
            Tuple containing pulse-level event features; event-level truth
                information; pulse-level truth information; and event-level
                loss weights, respectively.
        """

        features = self.query_table("", self._features, sequential_index)
        truth = self.query_table("", self._truth, sequential_index)

        if self._node_truth:
            node_truth = self.query_table("", self._node_truth, sequential_index)
        else:
            node_truth = None

        if self._loss_weight_column is not None:
            loss_weight = self.query_table(
                "", self._loss_weight_column, sequential_index
            )
        else:
            loss_weight = None

        return (features, truth, node_truth, loss_weight)

    def _get_all_indices(self) -> List[int]:
        """Return a list of all unique event indices in the database."""

        # first check if database as a map
        map_path = Path(self._path) / "idx_eventid_map.pkl"
        if map_path.exists():
            import pickle
            with open(map_path, "rb") as f:
                mapping = pickle.load(f)
            if "event_id_to_idx" in mapping:
                self.debug(f"Using cached indices from {map_path}")
                self._event_id_to_idx = mapping["event_id_to_idx"]
                self._idx_to_event_id = mapping["idx_to_event_id"]
                return list(mapping["event_id_to_idx"].keys())
            else:
                self.warning("Mapping file does not contain event_id_to_idx... recreating")

        self._lazy_connect()

        indices = []
        event_id_to_idx = {}
        idx_to_event_id = {}

        try:
            with self._env.begin() as txn:
                cursor = txn.cursor()
                for key_bytes, _ in cursor:
                    try:
                        # Assume keys are simple integers
                        # get the value of the index_column
                        index = int(key_bytes.decode("utf-8"))
                        event_id = self._query_table(
                            table="",
                            columns=[self._index_column],
                            sequential_index=index,
                            _ignore_selection=True,
                        )
                        event_id = event_id.flatten()[0]
                        indices.append(event_id)
                        event_id_to_idx[event_id] = index
                        idx_to_event_id[index] = event_id
                    except (ValueError, UnicodeDecodeError):
                        # Skip malformed keys
                        continue

        except lmdb.Error as e:
            raise RuntimeError(f"LMDB error getting indices: {e}")

        self._event_id_to_idx = event_id_to_idx
        self._idx_to_event_id = idx_to_event_id

        return indices

    def _get_event_index(self, sequential_index: int) -> int:
        """Return the event index corresponding to a sequential index."""
        # For this dataset, we assume indices are the actual keys
        return self._idx_to_event_id[sequential_index]

    def get_data(self, sequential_index: int, _ignore_selection: bool = False) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)) and not _ignore_selection:
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )

        # Resolve LMDB key robustly: map dataset position -> event_id -> LMDB key
        # Prefer the event_id mapping if available; fall back to raw index otherwise.
        try:
            event_id = self._indices[sequential_index]
        except Exception:
            raise IndexError(f"Index {sequential_index} not in range [0, {len(self) - 1}]")

        mapped_key = self._event_id_to_idx.get(event_id, None)
        # Start with mapped key if present; otherwise use sequential index as fallback
        effective_sequential_index = mapped_key if mapped_key is not None else sequential_index

        # Connect to database
        self._lazy_connect()

        # Create key for this event
        key = str(effective_sequential_index).encode("utf-8")

        try:
            with self._env.begin() as txn:
                serialized_data = txn.get(key)
                if serialized_data is None and mapped_key is not None:
                    # Fallback: try raw sequential index key in case mapping is stale
                    fallback_key = str(sequential_index).encode("utf-8")
                    serialized_data = txn.get(fallback_key)
                    if serialized_data is None:
                        raise IndexError(f"No data found for event_id {event_id}")
                elif serialized_data is None:
                    raise IndexError(f"No data found for index {effective_sequential_index}")

                # Deserialize the Data object
                data = self._deserialize(serialized_data)

                if not isinstance(data, Data):
                    raise ValueError(
                        f"Expected PyTorch Geometric Data object, got {type(data)}"
                    )

                # Create a copy to avoid potential multiprocessing issues
                # This ensures each process gets its own copy of the data
                data = data.clone() if hasattr(data, "clone") else data

                # Apply any necessary filtering based on features/truth
                if hasattr(self, "_features") and self._features:
                    # Filter node features if specified
                    available_features = []
                    for feat in self._features:
                        if hasattr(data, feat):
                            available_features.append(feat)

                if self._transform is not None:
                    data = self._transform(data)

                # The data is already a complete Data object with proper node/graph structure
                return data

        except lmdb.Error as e:
            raise RuntimeError(f"LMDB error reading event {key}: {e}")

    def _create_graph(
        self,
        features: np.ndarray,
        truth: np.ndarray,
        node_truth: Optional[np.ndarray] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the
                loss.

        Returns:
            Graph object.
        """
        # Convert truth to dict
        if len(truth.shape) == 1:
            truth = truth.reshape(1, -1)
        truth_dict = {key: truth[:, index] for index, key in enumerate(self._truth)}
        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            assert self._node_truth is not None
            node_truth_dict = {
                key: node_truth[:, index] for index, key in enumerate(self._node_truth)
            }

        # Create list of truth dicts with labels
        truth_dicts = [labels_dict, truth_dict]
        if node_truth is not None:
            truth_dicts.append(node_truth_dict)

        # Catch cases with no reconstructed pulses
        if len(features):
            node_features = features
        else:
            node_features = np.array([]).reshape((0, len(self._features)))

        assert isinstance(features, np.ndarray)
        # Construct graph data object
        assert self._graph_definition is not None
        graph = self._graph_definition(
            input_features=node_features,
            input_feature_names=self._features,
            truth_dicts=truth_dicts,
            custom_label_functions=self._label_fns,
            loss_weight_column=self._loss_weight_column,
            loss_weight=loss_weight,
            loss_weight_default_value=self._loss_weight_default_value,
            data_path=self._path,
        )
        return graph

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )

        if self._node_truth_table is not None:
            assert isinstance(self._node_truth, (list, str))  # mypy..
            node_truth = self.query_table(
                table=self._node_truth_table,
                columns=self._node_truth,
                sequential_index=sequential_index,
            )
        else:
            node_truth = None

        if self._loss_weight_table is not None:
            assert isinstance(self._loss_weight_column, str)
            loss_weight = self.query_table(
                table=self._loss_weight_table,
                columns=self._loss_weight_column,
                sequential_index=sequential_index,
            )
        else:
            loss_weight = None
        
        features = self.query_table("", self._features, sequential_index)
        truth = self.query_table("", self._truth, sequential_index)

        graph = self._create_graph(
            features=features,
            truth=truth,
            node_truth=node_truth,
            loss_weight=loss_weight,
        )
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        labels_dict = {
            self._index_column: truth_dict[self._index_column],
        }
        return labels_dict

    def __len__(self) -> int:
        if not hasattr(self, "_indices"):
            self._indices = self._get_all_indices()
        return len(self._indices)

    # Custom pickling methods to handle multiprocessing
    def __getstate__(self):
        """Prepare the object for pickling. Don't pickle the LMDB environment."""
        # Close connection before pickling
        self._close_connection()
        state = self.__dict__.copy()
        state["_env"] = None  # LMDB environment is not picklable
        # Keep cached indices - they're picklable and valuable to preserve
        return state

    def __setstate__(self, state):
        """Restore the object after unpickling."""
        self.__dict__.update(state)
        self._env = None  # Ensure _env is None, will be re-established by _lazy_connect
        # Don't open connection here - let it be lazy
        # _cached_indices will be restored from state if it was previously cached

    def __del__(self):
        """Cleanup when object is destroyed."""
        self._close_connection()
