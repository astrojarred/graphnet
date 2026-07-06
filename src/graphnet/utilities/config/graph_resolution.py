"""Checkpoint-authoritative graph-resolution utility.

A trained checkpoint's ``ModelConfig`` records the exact graph definition
(data representation) the model was trained with — including the KNN
``columns`` and ``nb_nearest_neighbours`` used for edge construction. If an
inference-time ``DatasetConfig`` carries a *different* graph definition, the
dataset silently builds different edges than the ones the checkpoint was
trained on (node features ``data.x`` are unaffected; ``columns`` only enters
edge construction).

``resolve_dataset_graph_from_model`` makes the checkpoint authoritative: it
extracts the graph definition from the model config, validates that it is
*compatible* with the dataset config (same input feature schema, detector,
node definition and dtype), and returns a **new** dataset config whose graph
definition is replaced by the checkpoint's. Differences that only affect edge
construction (KNN columns, number of neighbours, edge/graph class, etc.) are
logged and overridden — they are precisely what this utility exists to fix.
"""

import inspect
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from graphnet.utilities.config.dataset_config import DatasetConfig
from graphnet.utilities.config.model_config import ModelConfig
from graphnet.utilities.logging import Logger

__all__ = [
    "GraphCompatibilityError",
    "resolve_dataset_graph_from_model",
]

# ModelConfig argument keys under which a data representation may be stored.
_GRAPH_KEYS = ("graph_definition", "data_representation")

# Graph-definition constructor arguments that only affect *edge*
# construction. Differences in these are logged and overridden, never raised.
_EDGE_ONLY_ARGS = {
    "columns",
    "nb_nearest_neighbours",
    "distance_as_edge_feature",
    "edge_definition",
    "walk_length",
}


class GraphCompatibilityError(ValueError):
    """Dataset and checkpoint graph definitions are incompatible."""


def _load_class(class_name: str) -> Optional[type]:
    """Resolve a graphnet class by name; return None if lookup fails."""
    try:
        import graphnet.data
        import graphnet.models
        import graphnet.training
        from graphnet.utilities.config.parsing import (
            get_all_grapnet_classes,
        )

        return get_all_grapnet_classes(
            graphnet.data, graphnet.models, graphnet.training
        )[class_name]
    except Exception:  # pragma: no cover - lookup is best-effort
        return None


def _is_data_representation_class(class_name: str) -> bool:
    """Check whether `class_name` is a `DataRepresentation` subclass."""
    cls = _load_class(class_name)
    if cls is None:
        return False
    try:
        from graphnet.models.data_representation import DataRepresentation

        return issubclass(cls, DataRepresentation)
    except Exception:  # pragma: no cover
        return False


def _find_graph_definition(
    model_config: ModelConfig,
) -> Optional[ModelConfig]:
    """Locate the data-representation `ModelConfig` in `model_config`.

    Searches the model config's arguments (breadth-first, recursing into
    nested `ModelConfig` arguments) for a `graph_definition` /
    `data_representation` entry. Also handles the case where
    `model_config` itself *is* a data representation.
    """
    if _is_data_representation_class(model_config.class_name):
        return model_config

    queue: List[ModelConfig] = [model_config]
    while queue:
        current = queue.pop(0)
        # Direct hit on a known key.
        for key in _GRAPH_KEYS:
            value = current.arguments.get(key)
            if isinstance(value, ModelConfig):
                return value
        # Recurse into nested model configs.
        for value in current.arguments.values():
            if isinstance(value, ModelConfig):
                queue.append(value)
            elif isinstance(value, (list, tuple)):
                queue.extend(
                    v for v in value if isinstance(v, ModelConfig)
                )
    return None


def _serialise_value(value: Any) -> Any:
    """Convert `ModelConfig` graph arguments to dataset-config format."""
    import torch

    if isinstance(value, ModelConfig):
        return {
            "class_name": value.class_name,
            "arguments": {
                k: _serialise_value(v) for k, v in value.arguments.items()
            },
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialise_value(v) for v in value]
    return deepcopy(value)


def _model_config_to_dataset_graph_dict(
    graph_config: ModelConfig,
) -> Dict[str, Any]:
    """Serialise a graph-definition `ModelConfig` to dataset-config form."""
    return {
        "class_name": graph_config.class_name,
        "arguments": {
            k: _serialise_value(v)
            for k, v in graph_config.arguments.items()
        },
    }


def _dtype_str(value: Any) -> Optional[str]:
    """Normalise a dtype spec (str or `torch.dtype`) to a string."""
    if value is None:
        return None
    return str(value)


def _normalised_ctor_args(
    class_name: Optional[str], args: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Fill in constructor defaults so `{}` == explicit defaults.

    E.g. a `MAGIC` detector configured with `{}` is normalised to
    `{"use_signal_epsilon": True}` and therefore compares equal to a
    checkpoint that recorded the default explicitly.
    """
    args = dict(args or {})
    if class_name is None:
        return args
    cls = _load_class(class_name)
    if cls is None:
        return args
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover
        return args
    normalised: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        if param.default is not inspect.Parameter.empty:
            normalised[name] = param.default
    normalised.update(args)
    return normalised


def _sub_config(
    graph_dict: Optional[Dict[str, Any]], key: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return (class_name, arguments) of nested entry `key` (e.g. detector)."""
    if not graph_dict:
        return None, None
    entry = (graph_dict.get("arguments") or {}).get(key)
    if not isinstance(entry, dict):
        return None, None
    return entry.get("class_name"), entry.get("arguments") or {}


def _format_graph(graph_dict: Optional[Dict[str, Any]]) -> str:
    """Human-readable one-line summary of a graph-definition dict."""
    if not graph_dict:
        return "<none>"
    args = graph_dict.get("arguments") or {}
    det_cls, _ = _sub_config(graph_dict, "detector")
    node_cls, _ = _sub_config(graph_dict, "node_definition")
    return (
        f"{graph_dict.get('class_name')}("
        f"columns={args.get('columns')}, "
        f"nb_nearest_neighbours={args.get('nb_nearest_neighbours')}, "
        f"input_feature_names={args.get('input_feature_names')}, "
        f"detector={det_cls}, node_definition={node_cls}, "
        f"dtype={_dtype_str(args.get('dtype'))})"
    )


def resolve_dataset_graph_from_model(
    dataset_config: Union[DatasetConfig, str],
    model_config: Union[ModelConfig, str],
    strict: bool = True,
) -> DatasetConfig:
    """Return a dataset config using the checkpoint's graph definition.

    The graph definition (data representation) recorded in the trained
    model's `ModelConfig` is authoritative: the returned dataset config is a
    deep copy of `dataset_config` whose graph definition is replaced by the
    checkpoint's. Edge-construction differences (KNN ``columns``,
    ``nb_nearest_neighbours``, edge/graph class) are logged and overridden.
    Schema-level incompatibilities raise `GraphCompatibilityError` when
    `strict=True` (downgraded to warnings when `strict=False`):

    - input feature names/order mismatch (graph definition or dataset
      ``features``),
    - detector class or constructor-argument mismatch,
    - node-definition class mismatch,
    - dtype mismatch.

    Args:
        dataset_config: `DatasetConfig` instance or path to its YAML file.
        model_config: `ModelConfig` instance or path to its YAML file
            (e.g. the ``model_config.yml`` saved next to a checkpoint).
        strict: If True, raise on schema incompatibilities; if False, only
            warn (the checkpoint's graph definition is installed either way).

    Returns:
        A new `DatasetConfig`; the inputs are never mutated.

    Raises:
        GraphCompatibilityError: On schema incompatibility with strict=True.
        ValueError: If the model config contains no graph definition.
    """
    logger = Logger(log_folder=None)

    # Load configs if paths were given. Never mutate the inputs.
    if isinstance(dataset_config, str):
        dataset_config = DatasetConfig.load(dataset_config)
    if isinstance(model_config, str):
        model_config = ModelConfig.load(model_config)
    assert isinstance(dataset_config, DatasetConfig)
    assert isinstance(model_config, ModelConfig)

    # Extract the checkpoint's graph definition.
    graph_config = _find_graph_definition(model_config)
    if graph_config is None:
        raise ValueError(
            "Could not find a `graph_definition`/`data_representation` "
            f"entry in ModelConfig for `{model_config.class_name}`. "
            "Cannot resolve the dataset graph from this checkpoint."
        )
    model_graph = _model_config_to_dataset_graph_dict(graph_config)

    # Extract the dataset's requested graph definition (either field).
    dataset_field = (
        "data_representation"
        if getattr(dataset_config, "data_representation", None) is not None
        else "graph_definition"
    )
    dataset_graph = getattr(dataset_config, dataset_field, None)
    if dataset_graph is not None and not isinstance(dataset_graph, dict):
        raise GraphCompatibilityError(
            f"Dataset config `{dataset_field}` is not a plain mapping "
            f"(got {type(dataset_graph)}); cannot resolve."
        )

    logger.info(
        "Graph resolution (checkpoint-authoritative):\n"
        f"  requested (dataset config): {_format_graph(dataset_graph)}\n"
        f"  effective (checkpoint):     {_format_graph(model_graph)}"
    )

    # Compare and collect differences.
    problems: List[str] = []  # schema incompatibilities (raise if strict)
    overrides: List[str] = []  # edge-level differences (log + override)

    model_args = model_graph.get("arguments") or {}
    model_features = model_args.get("input_feature_names")

    if dataset_graph is not None:
        dataset_args = dataset_graph.get("arguments") or {}

        # 1) Input feature names/order.
        dataset_features = dataset_args.get("input_feature_names")
        if (
            dataset_features is not None
            and model_features is not None
            and list(dataset_features) != list(model_features)
        ):
            problems.append(
                "input_feature_names mismatch: dataset graph has "
                f"{list(dataset_features)}, checkpoint has "
                f"{list(model_features)}"
            )

        # 2) Detector class + constructor arguments.
        ds_det_cls, ds_det_args = _sub_config(dataset_graph, "detector")
        mc_det_cls, mc_det_args = _sub_config(model_graph, "detector")
        if ds_det_cls != mc_det_cls:
            problems.append(
                f"detector class mismatch: dataset has {ds_det_cls}, "
                f"checkpoint has {mc_det_cls}"
            )
        else:
            ds_norm = _normalised_ctor_args(ds_det_cls, ds_det_args)
            mc_norm = _normalised_ctor_args(mc_det_cls, mc_det_args)
            if ds_norm != mc_norm:
                problems.append(
                    f"detector constructor-argument mismatch for "
                    f"{ds_det_cls}: dataset has {ds_norm}, checkpoint has "
                    f"{mc_norm}"
                )

        # 3) Node-definition class.
        ds_node_cls, ds_node_args = _sub_config(
            dataset_graph, "node_definition"
        )
        mc_node_cls, mc_node_args = _sub_config(
            model_graph, "node_definition"
        )
        if ds_node_cls != mc_node_cls:
            problems.append(
                f"node-definition class mismatch: dataset has "
                f"{ds_node_cls}, checkpoint has {mc_node_cls}"
            )
        else:
            ds_norm = _normalised_ctor_args(ds_node_cls, ds_node_args)
            mc_norm = _normalised_ctor_args(mc_node_cls, mc_node_args)
            if ds_norm != mc_norm:
                overrides.append(
                    f"node-definition arguments differ for {ds_node_cls}: "
                    f"dataset has {ds_norm}, checkpoint has {mc_norm}"
                )

        # 4) dtype.
        ds_dtype = _dtype_str(dataset_args.get("dtype"))
        mc_dtype = _dtype_str(model_args.get("dtype"))
        if (
            ds_dtype is not None
            and mc_dtype is not None
            and ds_dtype != mc_dtype
        ):
            problems.append(
                f"dtype mismatch: dataset has {ds_dtype}, checkpoint has "
                f"{mc_dtype}"
            )

        # 5) Graph class (edge machinery) — override, not raise.
        if dataset_graph.get("class_name") != model_graph.get("class_name"):
            overrides.append(
                "graph class differs: dataset has "
                f"{dataset_graph.get('class_name')}, checkpoint has "
                f"{model_graph.get('class_name')}"
            )

        # 6) Remaining argument-level differences (edge-only args and any
        #    args not covered above) — override, not raise.
        handled = {
            "input_feature_names",
            "detector",
            "node_definition",
            "dtype",
        }
        for key in sorted(
            (set(dataset_args) | set(model_args)) - handled
        ):
            ds_val = dataset_args.get(key)
            mc_val = model_args.get(key)
            if ds_val != mc_val:
                overrides.append(
                    f"`{key}` differs: dataset has {ds_val!r}, checkpoint "
                    f"has {mc_val!r}"
                )

    # Dataset-level `features` must match the checkpoint's input schema.
    if (
        model_features is not None
        and dataset_config.features is not None
        and list(dataset_config.features) != list(model_features)
    ):
        problems.append(
            "dataset `features` mismatch with checkpoint "
            f"input_feature_names: dataset has "
            f"{list(dataset_config.features)}, checkpoint has "
            f"{list(model_features)}"
        )

    # Report.
    for message in overrides:
        logger.warning(f"Graph resolution override: {message}")
    if problems:
        joined = "\n  - ".join(problems)
        message = (
            "Dataset config graph definition is incompatible with the "
            f"checkpoint's:\n  - {joined}"
        )
        if strict:
            raise GraphCompatibilityError(message)
        logger.warning(message + "\n(strict=False: continuing anyway)")
    if not problems and not overrides and dataset_graph is not None:
        logger.info(
            "Graph resolution: dataset graph definition already matches "
            "the checkpoint; installing the checkpoint's definition anyway."
        )

    # Build the resolved config (deep copy; never mutate the input).
    try:
        resolved = dataset_config.model_copy(deep=True)  # pydantic v2
    except AttributeError:  # pragma: no cover - pydantic v1
        resolved = dataset_config.copy(deep=True)
    setattr(resolved, dataset_field, deepcopy(model_graph))

    logger.info(
        f"Graph resolution: dataset `{dataset_field}` replaced by the "
        f"checkpoint's graph definition: {_format_graph(model_graph)}"
    )
    return resolved
