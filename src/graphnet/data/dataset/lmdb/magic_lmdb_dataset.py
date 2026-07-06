"""LMDBDataset subclass that merges global table fields onto Data objects.

In raw mode, LMDBDataset reads MAGICPixels + truth tables and builds graphs
via KNNGraph + MAGIC detector. However, the `global` table (telescope
pointing, sizes, etc.) is not accessed by the base class.

This subclass overrides __getitem__ to:
1. Build the graph via the parent (raw mode)
2. Read the `global` table from the LMDB cache
3. Attach global fields + derived labels to the Data object
"""

import math
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import BaseTransform

from graphnet.data.dataset.lmdb.lmdb_dataset import LMDBDataset


# CORSIKA particle codes
GAMMA_PARTICLE_ID = 1
PROTON_PARTICLE_ID = 14

# Do not copy these from LMDB `global` onto Data. They are derived from the
# uncapped node count, and `n_pulses` in globals would clobber that metadata.
# `event_id`/`event_no` must stay excluded even though truth-table transport
# now preserves integer dtypes (via `LMDBDataset.query_table_as_mapping`):
# the global-copy loop below casts values through `float(...)`, which would
# corrupt packed identifiers above 2**53.
_EXCLUDE_GLOBAL_COPY = frozenset(
    ("event_id", "event_no", "n_pulses", "global_n_pulses_log10")
)

# Real (on-sky) LMDB exports use drive/pointing zenith distance [deg], not
# ``mc_telescope_theta_rad``. Same normalization as MC: angle_rad / pi.
_MAGIC_REAL_ZD_GLOBAL_KEYS = (
    "pointing_zd_deg",
    "drive_nominal_zd_deg",
    "drive_current_zd_deg",
)


class MAGICLMDBDataset(LMDBDataset):
    """LMDBDataset with global table merge and MAGIC-specific labels."""

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        graph_definition: Any = None,
        data_representation: Any = None,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: Any = None,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
        labels: Optional[Dict[str, Any]] = None,
        src_cam_radius_mm: float = 400.0,
        global_fields: Optional[List[str]] = None,
        sim_weights_pkl: Optional[str] = None,
        sim_weights_key: str = "proton_weights",
        graph_transform: Optional[BaseTransform] = None,
        max_nodes: Optional[int] = None,
        max_nodes_seed: Optional[int] = None,
        max_nodes_strategy: str = "random",
        pre_computed_representation: Optional[str] = None,
        repeat_labels_by: Optional[int] = None,
    ):
        """Construct MAGICLMDBDataset."""
        super().__init__(
            path=path,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            graph_definition=graph_definition,
            data_representation=data_representation,
            node_truth=node_truth,
            index_column=index_column,
            truth_table=truth_table,
            node_truth_table=node_truth_table,
            string_selection=string_selection,
            selection=selection,
            dtype=dtype,
            loss_weight_table=loss_weight_table,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
            seed=seed,
            labels=labels,
            pre_computed_representation=pre_computed_representation,
            repeat_labels_by=repeat_labels_by,
        )
        self._src_cam_radius_mm = src_cam_radius_mm
        self._global_fields = global_fields
        self._graph_transform = graph_transform

        self._sim_weights_proton: Optional[Dict[int, float]] = None
        self._sim_weights_gamma: Optional[Dict[int, float]] = None
        self._sim_weights_single: Optional[Dict[int, float]] = None
        if sim_weights_pkl is not None:
            with open(sim_weights_pkl, "rb") as f:
                weights_data = pickle.load(f)
            pw = weights_data.get("proton_weights")
            gw = weights_data.get("gamma_weights")
            if isinstance(pw, dict) and isinstance(gw, dict):
                self._sim_weights_proton = pw
                self._sim_weights_gamma = gw
            elif isinstance(weights_data.get(sim_weights_key), dict):
                self._sim_weights_single = weights_data[sim_weights_key]
            else:
                raise KeyError(
                    "sim_weights_pkl must contain 'proton_weights' and "
                    "'gamma_weights' dicts (from fit_simulation_weights.py), "
                    f"or a single dict under key {sim_weights_key!r}."
                )

        self._max_nodes = max_nodes
        self._max_nodes_seed = max_nodes_seed
        self._max_nodes_strategy = max_nodes_strategy
        self._signal_feature_index = (
            features.index("signal") if "signal" in features else None
        )

    def _set_n_pulses_metadata(self, data: Data, n_full: int) -> None:
        """Store the uncapped pulse count on the graph."""
        data.n_pulses = torch.tensor(int(n_full), dtype=torch.int32)
        data.global_n_pulses_log10 = torch.tensor(
            math.log10(float(max(n_full, 1))),
            dtype=self._dtype,
        )

    def _get_subsample_indices(
        self,
        n_rows: int,
        sequential_index: int,
        *,
        features: Optional[np.ndarray] = None,
        data_x: Optional[torch.Tensor] = None,
    ) -> Optional[np.ndarray]:
        """Return deterministic row indices for the configured node cap."""
        cap = self._max_nodes
        if cap is None or cap <= 0 or n_rows <= cap:
            return None

        strategy = self._max_nodes_strategy
        if strategy in {"signal", "high_signal"}:
            scores = self._node_signal_scores(features=features, data_x=data_x)
            if scores is not None:
                return np.argsort(-scores, kind="stable")[:cap]

        if strategy == "minmax_signal":
            scores = self._node_signal_scores(features=features, data_x=data_x)
            if scores is not None:
                order = np.argsort(scores, kind="stable")
                n_low = cap // 2
                low = order[:n_low]
                high = order[-(cap - n_low) :]
                return np.concatenate([high[::-1], low])

        if strategy != "random":
            raise ValueError(
                "max_nodes_strategy must be one of 'random', 'signal', "
                f"'high_signal', or 'minmax_signal', got {strategy!r}."
            )

        gen = torch.Generator(device="cpu")
        if self._max_nodes_seed is not None:
            gen.manual_seed(int(self._max_nodes_seed) + int(sequential_index))
        perm = torch.randperm(n_rows, generator=gen)
        return perm[:cap].numpy()

    def _node_signal_scores(
        self,
        *,
        features: Optional[np.ndarray],
        data_x: Optional[torch.Tensor],
    ) -> Optional[np.ndarray]:
        """Return per-node signal scores before the v8 feature permutation."""
        idx = self._signal_feature_index
        if idx is None:
            return None
        if features is not None:
            return np.asarray(features[:, idx], dtype=float)
        if data_x is not None:
            return data_x[:, idx].detach().cpu().numpy().astype(float)
        return None

    def _apply_n_pulses_and_optional_cap(
        self, data: Data, sequential_index: int
    ) -> Data:
        """Set uncapped pulse metadata and optionally cap precomputed graphs."""
        n_full = int(data.x.size(0))
        self._set_n_pulses_metadata(data, n_full)

        idx_np = self._get_subsample_indices(
            n_full, sequential_index, data_x=data.x
        )
        if idx_np is None:
            return data

        idx = torch.as_tensor(idx_np, device=data.x.device, dtype=torch.long)

        data.x = data.x[idx]
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = data.pos[idx]

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = None

        self._rebuild_knn_edges(data)
        return data

    def _cap_features_before_graph_build(
        self,
        features: np.ndarray,
        node_truth: Optional[np.ndarray],
        sequential_index: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray], int]:
        """Cap raw feature rows before graph construction."""
        n_full = int(features.shape[0])
        idx_np = self._get_subsample_indices(
            n_full, sequential_index, features=features
        )
        if idx_np is None:
            return features, node_truth, n_full

        if node_truth is not None and int(node_truth.shape[0]) != n_full:
            raise ValueError(
                "Node truth and feature row counts diverged before MAGIC "
                "node capping."
            )

        features = features[idx_np]
        if node_truth is not None:
            node_truth = node_truth[idx_np]
        return features, node_truth, n_full

    def _rebuild_knn_edges(self, data: Data) -> None:
        """Rebuild KNN edges after node subsampling (MAGIC 5-column x, KNN columns)."""
        device = data.x.device
        num_nodes = int(data.x.size(0))

        rep = getattr(self, "_data_representation", None)
        edge_def = getattr(rep, "_edge_definition", None) if rep is not None else None
        if edge_def is None:
            data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            return

        k = int(edge_def._nb_nearest_neighbours)
        cols = list(edge_def._columns)
        pos = data.x[:, cols]
        batch = getattr(data, "batch", None)

        if num_nodes <= 0:
            data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            return

        if num_nodes == 1:
            data.edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            return

        k_eff = min(k, num_nodes - 1)
        if k_eff < 1:
            data.edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            return

        data.edge_index = knn_graph(
            pos,
            k_eff,
            batch,
        ).to(device=device)

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph Data with global fields and derived labels."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )

        if self._pre_computed_representation is None:
            features, truth, node_truth, loss_weight = self._query(
                sequential_index
            )
            features, node_truth, n_full = self._cap_features_before_graph_build(
                features, node_truth, sequential_index
            )
            data = self._create_graph(features, truth, node_truth, loss_weight)
            self._set_n_pulses_metadata(data, n_full)
        else:
            data = super().__getitem__(sequential_index)
            data = self._apply_n_pulses_and_optional_cap(data, sequential_index)

        cached = self._cached_data

        if isinstance(cached, dict) and "global" in cached:
            global_table = cached["global"]
            fields_to_add = (
                self._global_fields
                if self._global_fields is not None
                else [
                    k
                    for k in global_table.keys()
                    if k not in _EXCLUDE_GLOBAL_COPY
                ]
            )
            for field in fields_to_add:
                if field in _EXCLUDE_GLOBAL_COPY:
                    continue
                if field in global_table:
                    val = global_table[field]
                    if isinstance(val, list):
                        val = val[0]
                    data[field] = torch.tensor(
                        float(val), dtype=self._dtype
                    )

            if "mc_telescope_theta_rad" in global_table:
                theta = global_table["mc_telescope_theta_rad"]
                if isinstance(theta, list):
                    theta = theta[0]
                data.global_zenith_norm = torch.tensor(
                    float(theta) / math.pi, dtype=self._dtype
                )
            else:
                zd_deg: Optional[float] = None
                for key in _MAGIC_REAL_ZD_GLOBAL_KEYS:
                    if key not in global_table:
                        continue
                    raw = global_table[key]
                    if isinstance(raw, list):
                        raw = raw[0]
                    zd_deg = float(raw)
                    break
                if zd_deg is not None:
                    data.global_zenith_norm = torch.tensor(
                        math.radians(zd_deg) / math.pi,
                        dtype=self._dtype,
                    )

        if not hasattr(data, "global_zenith_norm"):
            data.global_zenith_norm = torch.tensor(0.0, dtype=self._dtype)

        self._add_derived_labels(data)

        if hasattr(data, "event_no"):
            event_no = int(data.event_no)
            weight: Optional[float] = None
            if self._sim_weights_proton is not None and self._sim_weights_gamma is not None:
                if hasattr(data, "particle_id"):
                    pid = int(data.particle_id)
                    table = (
                        self._sim_weights_gamma
                        if pid == GAMMA_PARTICLE_ID
                        else self._sim_weights_proton
                    )
                    weight = table.get(event_no, 1.0)
                else:
                    weight = 1.0
            elif self._sim_weights_single is not None:
                weight = self._sim_weights_single.get(event_no, 1.0)
            if weight is not None:
                data.loss_weight = torch.tensor(weight, dtype=self._dtype)

        if hasattr(data, "particle_signal"):
            if hasattr(data, "loss_weight"):
                data.regression_loss_weight = (
                    data.loss_weight * data.particle_signal
                )
            else:
                data.regression_loss_weight = data.particle_signal
        else:
            data.regression_loss_weight = torch.tensor(
                0.0, dtype=self._dtype
            )

        if self._graph_transform is not None:
            data = self._graph_transform(data)

        return data

    def _add_derived_labels(self, data: Data) -> None:
        """Add MAGIC-specific derived labels to the Data object."""
        if hasattr(data, "energy_gev"):
            energy = float(data.energy_gev)
            if energy > 0:
                data.log_energy = torch.tensor(
                    math.log10(energy), dtype=self._dtype
                )

        if hasattr(data, "particle_id"):
            pid = int(data.particle_id)
            data.particle_signal = torch.tensor(
                1.0 if pid == GAMMA_PARTICLE_ID else 0.0,
                dtype=self._dtype,
            )

        if hasattr(data, "src_cam_x_mm") and hasattr(data, "src_cam_y_mm"):
            data.src_cam_x_norm = torch.tensor(
                float(data.src_cam_x_mm) / self._src_cam_radius_mm,
                dtype=self._dtype,
            )
            data.src_cam_y_norm = torch.tensor(
                float(data.src_cam_y_mm) / self._src_cam_radius_mm,
                dtype=self._dtype,
            )
