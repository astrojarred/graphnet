"""Tests for bit-exact integer event-identifier transport.

MAGIC packed `event_id` values exceed 2**53 and are therefore not exactly
representable as float64 (and only exact to 2**24 as float32). These tests
verify that identifiers survive, bit-exact, through:

1. The LMDB dataset read path (mapping-based truth transport),
2. Truth attachment on `Data` objects (`torch.int64`), and
3. Prediction DataFrame construction (`predict_as_dataframe` and the
   deprecated `get_predictions`).
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import lmdb
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from graphnet.data.constants import FEATURES
from graphnet.data.dataset.lmdb import LMDBDataset
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.easy_model import assert_exact_event_ids
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.task import IdentityTask
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.utils import collate_fn, get_predictions

# Number of synthetic events; identifiers all above 2**53 with odd offsets,
# so any float64 round-trip (spacing 2.0 at 2**53) corrupts them.
N_EVENTS = 10_000
BASE_ID = 2**53
EVENT_IDS = BASE_ID + np.arange(N_EVENTS, dtype=np.int64) * 7919 + 1

PULSEMAP = "pulses"
TRUTH_TABLE = "truth"
INDEX_COLUMN = "event_id"
FEATURE_NAMES = FEATURES.DEEPCORE  # 7 features
TRUTH_COLUMNS = ["energy_gev", "is_gamma"]
N_PULSES = 4


def _write_test_lmdb(path: str, event_ids: np.ndarray) -> None:
    """Write a minimal LMDB in the dataset's expected serialization format.

    Each entry is a pickled dict of tables:
    ``{pulsemap: {feature: [values]}, truth: {column: value}}`` stored under
    the key ``str(event_id)``, plus the ``__meta_serialization__`` metadata
    entry declaring pickle serialization.
    """
    rng = np.random.default_rng(42)
    env = lmdb.open(path, map_size=1 << 30, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__meta_serialization__", b"pickle")
            for i, event_id in enumerate(event_ids):
                pulses = {
                    name: rng.normal(size=N_PULSES).tolist()
                    for name in FEATURE_NAMES
                }
                truth = {
                    INDEX_COLUMN: int(event_id),
                    "energy_gev": float(10.0 + i * 0.001),
                    "is_gamma": bool(i % 2 == 0),
                }
                entry = {PULSEMAP: pulses, TRUTH_TABLE: truth}
                txn.put(
                    str(int(event_id)).encode("utf-8"),
                    pickle.dumps(entry),
                )
    finally:
        env.close()


def _make_graph_definition(dtype: torch.dtype = torch.float32) -> KNNGraph:
    return KNNGraph(
        detector=IceCubeDeepCore(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=2,
        input_feature_names=FEATURE_NAMES,
        dtype=dtype,
    )


@pytest.fixture(scope="module")
def lmdb_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("lmdb") / "test_event_ids.lmdb")
    _write_test_lmdb(path, EVENT_IDS)
    return path


@pytest.fixture(scope="module")
def dataset(lmdb_path: str) -> LMDBDataset:
    return LMDBDataset(
        path=lmdb_path,
        pulsemaps=PULSEMAP,
        features=list(FEATURE_NAMES),
        truth=list(TRUTH_COLUMNS),
        index_column=INDEX_COLUMN,
        truth_table=TRUTH_TABLE,
        data_representation=_make_graph_definition(),
    )


def test_round_trip_event_ids_above_2_53(dataset: LMDBDataset) -> None:
    """Read via the dataset class: event_id must be torch.int64, bit-exact."""
    assert len(dataset) == N_EVENTS
    # LMDB keys are sorted; our ids are generated in increasing order.
    for sequential_index in (0, 1, 4999, 9999):
        data = dataset[sequential_index]
        expected = int(EVENT_IDS[sequential_index])
        assert data.event_id.dtype == torch.int64
        assert int(data.event_id) == expected
        # Odd values above 2**53 are not float64-representable (spacing is
        # 2.0 there), so the same value through float64 would NOT be exact:
        if expected % 2 == 1:
            assert int(np.float64(expected)) != expected


def test_predict_as_dataframe_preserves_event_ids(
    dataset: LMDBDataset,
) -> None:
    """Trivial model prediction: event_id column is int64 and bit-exact."""
    graph_definition = _make_graph_definition()
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["mean"],
    )
    task = IdentityTask(
        nb_outputs=1,
        target_labels="energy_gev",
        hidden_size=backbone.nb_outputs,
        loss_function=LogCoshLoss(),
    )
    model = StandardModel(
        data_representation=graph_definition,
        backbone=backbone,
        tasks=[task],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    df = model.predict_as_dataframe(
        dataloader,
        additional_attributes=[INDEX_COLUMN, "energy_gev"],
        gpus=None,
    )
    assert len(df) == N_EVENTS
    # Identifier column: native integer dtype, bit-exact, all unique.
    assert df[INDEX_COLUMN].dtype == np.int64
    np.testing.assert_array_equal(
        df[INDEX_COLUMN].to_numpy(), EVENT_IDS
    )
    assert df[INDEX_COLUMN].nunique() == N_EVENTS
    # Prediction column remains floating point.
    pred_col = task.default_prediction_labels[0]
    assert np.issubdtype(df[pred_col].dtype, np.floating)
    # The guard accepts this DataFrame.
    assert_exact_event_ids(df, column=INDEX_COLUMN)


def test_mixed_float_int_bool_truth_dtypes(dataset: LMDBDataset) -> None:
    """Mixed truth columns keep their dtypes through the mapping path."""
    data = dataset[0]
    assert data.event_id.dtype == torch.int64
    assert data.energy_gev.dtype == torch.float32  # configured float dtype
    assert data.is_gamma.dtype == torch.bool
    assert float(data.energy_gev) == pytest.approx(10.0)
    assert bool(data.is_gamma) is True


def test_query_returns_mapping_with_native_dtypes(
    dataset: LMDBDataset,
) -> None:
    """The LMDB `_query` truth payload is `{column: 1-D native array}`."""
    _, truth, _, _ = dataset._query(0)
    assert isinstance(truth, dict)
    assert set(truth.keys()) == {INDEX_COLUMN, *TRUTH_COLUMNS}
    assert truth[INDEX_COLUMN].dtype == np.int64
    assert truth[INDEX_COLUMN].ndim == 1
    assert truth["energy_gev"].dtype.kind == "f"
    assert truth["is_gamma"].dtype == np.bool_
    assert int(truth[INDEX_COLUMN][0]) == int(EVENT_IDS[0])


def test_legacy_matrix_truth_still_works(dataset: LMDBDataset) -> None:
    """`_create_graph` still accepts the legacy homogeneous matrix form."""
    features = np.random.default_rng(0).normal(
        size=(N_PULSES, len(FEATURE_NAMES))
    )
    # Homogeneous float64 matrix ordered as dataset._truth
    # ([event_id, energy_gev, is_gamma]); ids small enough to stay exact.
    truth_matrix = np.array([[12345.0, 10.5, 1.0]], dtype=np.float64)
    data = dataset._create_graph(features, truth_matrix)
    assert isinstance(data, Data)
    # Legacy behavior: values arrive as floats and stay floating point
    # (in the configured float dtype).
    assert data.event_id.dtype == torch.float32
    assert float(data.event_id) == 12345.0
    assert float(data.energy_gev) == pytest.approx(10.5)


def test_add_truth_tensor_dtypes() -> None:
    """int64 stays int64, bool stays bool, floats get configured dtype."""
    for float_dtype in (torch.float32, torch.float64):
        rep = _make_graph_definition(dtype=float_dtype)
        features = np.random.default_rng(1).normal(
            size=(N_PULSES, len(FEATURE_NAMES))
        )
        data = rep(
            input_features=features,
            input_feature_names=list(FEATURE_NAMES),
        )
        big_id = 2**53 + 12345
        truth_dicts: List[Dict[str, Any]] = [
            {
                "event_id": np.array([big_id], dtype=np.int64),
                "flag": np.array([True]),
                "target": np.array([1.5], dtype=np.float64),
                "py_int": 7,
            }
        ]
        data = rep._add_truth(data=data, truth_dicts=truth_dicts)
        assert data.event_id.dtype == torch.int64
        assert int(data.event_id) == big_id
        assert data.flag.dtype == torch.bool
        assert bool(data.flag) is True
        assert data.target.dtype == float_dtype
        assert data.py_int.dtype == torch.int64
        assert int(data.py_int) == 7


def test_deprecated_get_predictions_preserves_event_no() -> None:
    """The deprecated `get_predictions` path keeps int64 identifiers."""
    big_ids = (2**53 + 1 + np.arange(8, dtype=np.int64) * 101).astype(
        np.int64
    )
    graphs = []
    for event_no in big_ids:
        d = Data(x=torch.rand(3, 2))
        d.event_no = torch.tensor([int(event_no)], dtype=torch.int64)
        d.n_pulses = torch.tensor(3, dtype=torch.int32)
        graphs.append(d)
    batches = [
        Batch.from_data_list(graphs[:4]),
        Batch.from_data_list(graphs[4:]),
    ]

    class _StubTrainer:
        def predict(self, model: Any, dataloader: Any) -> List[Any]:
            return [(torch.rand(4, 1),), (torch.rand(4, 1),)]

    class _StubModel:
        def inference(self) -> None:
            pass

    df = get_predictions(
        trainer=_StubTrainer(),
        model=_StubModel(),
        dataloader=batches,
        prediction_columns=["pred"],
        additional_attributes=["event_no"],
    )
    assert df["event_no"].dtype == np.int64
    np.testing.assert_array_equal(df["event_no"].to_numpy(), big_ids)
    assert np.issubdtype(df["pred"].dtype, np.floating)


def test_assert_exact_event_ids_guard() -> None:
    """Guard raises on float identifier columns and missing columns."""
    good = pd.DataFrame(
        {"event_id": np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)}
    )
    assert_exact_event_ids(good)  # no raise

    legacy = pd.DataFrame(
        {"event_id": np.array([2**53 + 1, 2**53 + 3], dtype=np.float64)}
    )
    with pytest.raises(TypeError, match="floating-point"):
        assert_exact_event_ids(legacy)

    with pytest.raises(KeyError):
        assert_exact_event_ids(good, column="event_no")

    # float32 (exact only to 2**24) must also be rejected.
    legacy32 = pd.DataFrame(
        {"event_no": np.array([1.0, 2.0], dtype=np.float32)}
    )
    with pytest.raises(TypeError):
        assert_exact_event_ids(legacy32, column="event_no")
