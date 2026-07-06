"""Unit tests for checkpoint-authoritative graph resolution.

Covers `graphnet.utilities.config.graph_resolution`, which makes a trained
checkpoint's graph definition authoritative over the dataset config's when
constructing inference-time datasets. KNN `columns` only affect edge
construction, so `data.x` must be unchanged while `edge_index` follows the
checkpoint.
"""

import os
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from torch_geometric.nn import knn_graph

from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.data.dataset.dataset import parse_data_representation
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.models.data_representation.graphs.nodes import NodesAsPulses
from graphnet.models.detector.magic import MAGIC
from graphnet.utilities.config import (
    DatasetConfig,
    GraphCompatibilityError,
    ModelConfig,
    resolve_dataset_graph_from_model,
)

# Input feature schema used by MAGIC training/inference (v9).
FEATURES = ["signal", "x_cam", "y_cam", "time", "tel_id"]
COLUMNS_TRAINING = [1, 2, 4, 3]  # x_cam, y_cam, tel_id, time (v9 checkpoint)
COLUMNS_INFERENCE = [1, 2]  # x_cam, y_cam (stale real-data YAMLs)
NB_NEIGHBOURS = 8

DOWNLOADS_DIR = os.path.join(GRAPHNET_ROOT_DIR, "downloads")
V9C_MODEL_CONFIG = os.path.join(
    DOWNLOADS_DIR,
    "train-v9-xl/outputs/run-v9c-multitask-push/model_config.yml",
)
MRK421_CONFIG_DIR = os.path.join(DOWNLOADS_DIR, "configs/mrk421")


def _synthetic_event(n_per_telescope: int = 8) -> np.ndarray:
    """Synthetic MAGIC-like event: two telescopes sharing (x, y) positions.

    Each telescope observes near-identical camera positions, so KNN over
    (x_cam, y_cam) alone connects across telescopes, whereas including
    tel_id/time in the distance changes the neighbourhoods.
    """
    rng = np.random.default_rng(42)
    base_xy = rng.uniform(-100.0, 100.0, size=(n_per_telescope, 2))
    rows = []
    for tel_id in (0.0, 1.0):
        jitter = rng.normal(0.0, 0.5, size=base_xy.shape)
        xy = base_xy + jitter
        signal = rng.uniform(1.0, 300.0, size=n_per_telescope)
        time = rng.uniform(-10.0, 40.0, size=n_per_telescope)
        for i in range(n_per_telescope):
            rows.append(
                [signal[i], xy[i, 0], xy[i, 1], time[i], tel_id]
            )
    return np.asarray(rows, dtype=np.float64)


def _make_graph(columns: List[int]) -> KNNGraph:
    """KNNGraph mirroring the MAGIC training/inference definitions."""
    return KNNGraph(
        detector=MAGIC(use_signal_epsilon=True),
        node_definition=NodesAsPulses(),
        input_feature_names=list(FEATURES),
        dtype=torch.float32,
        nb_nearest_neighbours=NB_NEIGHBOURS,
        columns=list(columns),
    )


def _sorted_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Canonicalise an edge_index for comparison."""
    order = np.lexsort(
        (edge_index[0].numpy(), edge_index[1].numpy())
    )
    return edge_index[:, order]


def _graph_definition_dict(columns: List[int]) -> Dict[str, Any]:
    """Dataset-config style graph-definition dict (as in the real YAMLs)."""
    return {
        "class_name": "KNNGraph",
        "arguments": {
            "columns": list(columns),
            "detector": {"class_name": "MAGIC", "arguments": {}},
            "dtype": "torch.float32",
            "nb_nearest_neighbours": NB_NEIGHBOURS,
            "node_definition": {
                "class_name": "NodesAsPulses",
                "arguments": {},
            },
            "input_feature_names": list(FEATURES),
        },
    }


def _dataset_config(
    columns: List[int],
    features: Optional[List[str]] = None,
    input_feature_names: Optional[List[str]] = None,
) -> DatasetConfig:
    """Synthetic dataset config mirroring the mrk421 real-data YAMLs."""
    graph = _graph_definition_dict(columns)
    if input_feature_names is not None:
        graph["arguments"]["input_feature_names"] = list(input_feature_names)
    return DatasetConfig(
        path="/does/not/exist/merged.lmdb",
        pulsemaps="MAGICPixels",
        features=list(features or FEATURES),
        truth=["event_no"],
        index_column="event_no",
        truth_table="truth",
        seed=42,
        use_magic_lmdb=True,
        max_nodes=4096,
        max_nodes_seed=42,
        selection={"test": "event_no >= 0"},
        graph_definition=graph,
    )


def _model_config(columns: List[int]) -> ModelConfig:
    """Synthetic model config with the graph def nested under arguments.

    Mirrors the layout of the saved v9c `model_config.yml`, where
    `graph_definition` is an entry of the top-level model's `arguments`.
    """
    graph_config = ModelConfig(
        class_name="KNNGraph",
        arguments={
            "columns": list(columns),
            "detector": {
                "ModelConfig": {
                    "class_name": "MAGIC",
                    "arguments": {"use_signal_epsilon": True},
                }
            },
            "distance_as_edge_feature": False,
            "dtype": "torch.float32",
            "input_feature_names": list(FEATURES),
            "nb_nearest_neighbours": NB_NEIGHBOURS,
            "node_definition": {
                "ModelConfig": {
                    "class_name": "NodesAsPulses",
                    "arguments": {"input_feature_names": None},
                }
            },
            "perturbation_dict": None,
            "seed": None,
        },
    )
    return ModelConfig(
        class_name="MagicDeepIceV9Model",
        arguments={"graph_definition": graph_config, "decoder_hidden": 256},
    )


def _build_from_dataset_graph_dict(graph_dict: Dict[str, Any]) -> KNNGraph:
    """Instantiate a graph def the same way `Dataset.from_config` does."""
    return parse_data_representation(deepcopy(graph_dict))


def test_all_features_present_when_knn_columns_change() -> None:
    """`data.x` keeps all 5 features regardless of KNN columns."""
    event = _synthetic_event()
    graph_inference = _make_graph(COLUMNS_INFERENCE)
    graph_training = _make_graph(COLUMNS_TRAINING)

    data_inference = graph_inference(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )
    data_training = graph_training(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )

    assert data_inference.x.shape[1] == len(FEATURES)
    assert data_training.x.shape[1] == len(FEATURES)
    assert torch.equal(data_inference.x, data_training.x)


def test_only_edge_index_differs_between_graph_definitions() -> None:
    """Changing KNN columns changes `edge_index` only."""
    event = _synthetic_event()
    data_inference = _make_graph(COLUMNS_INFERENCE)(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )
    data_training = _make_graph(COLUMNS_TRAINING)(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )

    # Node features identical ...
    assert torch.equal(data_inference.x, data_training.x)
    # ... but edges genuinely differ on this synthetic event.
    edges_inference = _sorted_edges(data_inference.edge_index)
    edges_training = _sorted_edges(data_training.edge_index)
    assert not torch.equal(edges_inference, edges_training)

    # And the training-side edges match a direct knn_graph over the
    # checkpoint's columns.
    expected = knn_graph(
        data_training.x[:, COLUMNS_TRAINING], NB_NEIGHBOURS
    )
    assert torch.equal(
        _sorted_edges(data_training.edge_index), _sorted_edges(expected)
    )


def test_resolver_overrides_columns_from_checkpoint() -> None:
    """Resolver installs checkpoint columns; resolved graph == training."""
    dataset_config = _dataset_config(COLUMNS_INFERENCE)
    model_config = _model_config(COLUMNS_TRAINING)

    resolved = resolve_dataset_graph_from_model(
        dataset_config, model_config, strict=True
    )

    # Inputs not mutated.
    assert (
        dataset_config.graph_definition["arguments"]["columns"]
        == COLUMNS_INFERENCE
    )
    # Resolved config carries the checkpoint's edge definition.
    resolved_args = resolved.graph_definition["arguments"]
    assert resolved_args["columns"] == COLUMNS_TRAINING
    assert resolved_args["nb_nearest_neighbours"] == NB_NEIGHBOURS
    assert resolved_args["input_feature_names"] == FEATURES

    # Building the graph from the resolved config reproduces the
    # training-side edges on a shared synthetic event.
    event = _synthetic_event()
    graph_resolved = _build_from_dataset_graph_dict(
        resolved.graph_definition
    )
    graph_training = _make_graph(COLUMNS_TRAINING)

    data_resolved = graph_resolved(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )
    data_training = graph_training(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )

    assert torch.equal(data_resolved.x, data_training.x)
    assert torch.equal(
        _sorted_edges(data_resolved.edge_index),
        _sorted_edges(data_training.edge_index),
    )


def test_node_capping_rebuild_uses_checkpoint_columns() -> None:
    """`_rebuild_knn_edges` uses the resolved (checkpoint) columns and k."""
    pytest.importorskip("lmdb")
    from graphnet.data.dataset.lmdb.magic_lmdb_dataset import (
        MAGICLMDBDataset,
    )

    resolved = resolve_dataset_graph_from_model(
        _dataset_config(COLUMNS_INFERENCE),
        _model_config(COLUMNS_TRAINING),
        strict=True,
    )
    graph_resolved = _build_from_dataset_graph_dict(
        resolved.graph_definition
    )

    # The rebuild logic reads columns/k from the dataset's data
    # representation, so a resolved graph definition fixes capping too.
    edge_def = graph_resolved._edge_definition
    assert list(edge_def._columns) == COLUMNS_TRAINING
    assert int(edge_def._nb_nearest_neighbours) == NB_NEIGHBOURS

    event = _synthetic_event()
    data = graph_resolved(
        input_features=deepcopy(event), input_feature_names=list(FEATURES)
    )
    # Simulate node capping: drop some nodes, then rebuild edges the way
    # MAGICLMDBDataset does after subsampling.
    keep = torch.arange(0, data.x.shape[0], 2)
    data.x = data.x[keep]
    data.edge_index = None

    dummy_self = SimpleNamespace(_data_representation=graph_resolved)
    MAGICLMDBDataset._rebuild_knn_edges(dummy_self, data)

    expected = knn_graph(
        data.x[:, COLUMNS_TRAINING],
        min(NB_NEIGHBOURS, data.x.shape[0] - 1),
    )
    assert torch.equal(
        _sorted_edges(data.edge_index), _sorted_edges(expected)
    )


def test_incompatible_feature_schema_raises_when_strict() -> None:
    """Different feature names/order raises with strict=True."""
    scrambled = ["x_cam", "y_cam", "signal", "time", "tel_id"]
    dataset_config = _dataset_config(
        COLUMNS_INFERENCE,
        features=scrambled,
        input_feature_names=scrambled,
    )
    model_config = _model_config(COLUMNS_TRAINING)

    with pytest.raises(GraphCompatibilityError):
        resolve_dataset_graph_from_model(
            dataset_config, model_config, strict=True
        )

    # strict=False downgrades to a warning but still overrides.
    resolved = resolve_dataset_graph_from_model(
        dataset_config, model_config, strict=False
    )
    assert (
        resolved.graph_definition["arguments"]["columns"]
        == COLUMNS_TRAINING
    )
    assert (
        resolved.graph_definition["arguments"]["input_feature_names"]
        == FEATURES
    )


@pytest.mark.skipif(
    not os.path.isfile(V9C_MODEL_CONFIG)
    or not os.path.isdir(MRK421_CONFIG_DIR),
    reason="downloads/ working data not present",
)
def test_integration_real_v9c_checkpoint_config() -> None:
    """Real mrk421 dataset YAMLs + real v9c model config resolve to [1,2,4,3]."""
    yaml_files = [
        os.path.join(MRK421_CONFIG_DIR, f)
        for f in sorted(os.listdir(MRK421_CONFIG_DIR))
        if f.endswith(".yml")
    ]
    assert yaml_files, "No mrk421 YAMLs found"

    for yaml_file in yaml_files:
        resolved = resolve_dataset_graph_from_model(
            yaml_file, V9C_MODEL_CONFIG, strict=True
        )
        args = resolved.graph_definition["arguments"]
        assert args["columns"] == COLUMNS_TRAINING, yaml_file
        assert args["nb_nearest_neighbours"] == NB_NEIGHBOURS, yaml_file
        assert args["input_feature_names"] == FEATURES, yaml_file
