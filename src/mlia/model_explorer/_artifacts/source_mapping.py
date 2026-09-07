# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Canonical source-operator mappings for Model Explorer graphs."""

from __future__ import annotations

import importlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mlia.core import output_schema as schema
from mlia.core.errors import UnsupportedConfigurationError

from .models import _NodeRef


def _source_operator_node_ids(
    model_path: Path,
    graph_collection: dict[str, Any],
) -> dict[str, list[_NodeRef]]:
    """Construct canonical source-operator identities from native adapter metadata.

    This is the format boundary: downstream code treats the resulting entity IDs
    as opaque values and never parses them. ONNX identities use the zero-based
    position in the top-level ``ModelProto.graph.node`` list and are correlated
    to actual adapter-emitted nodes through native name, tensor-tag, and operator
    label evidence. TFLite's direct adapter preserves operator indices as numeric
    node IDs and graph order as flatbuffer subgraph order. The VGF adapter prefixes
    SPIR-V node IDs with the VGF segment index and exposes the instruction result
    ID in the native ``spv id`` attribute.
    """
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return _onnx_source_operator_node_ids(model_path, graph_collection)
    if suffix == ".pt2":
        return _pt2_source_operator_node_ids(model_path, graph_collection)
    if suffix == ".tflite":
        return _tflite_source_operator_node_ids(graph_collection)
    if suffix == ".spv":
        return _spirv_source_operator_node_ids(graph_collection)
    if suffix == ".vgf":
        return _vgf_source_operator_node_ids(graph_collection)
    raise UnsupportedConfigurationError(
        "Model Explorer does not support source-operator mapping for model "
        f"format '{suffix or '<none>'}'."
    )


_ONNX_AUXILIARY_NODE_LABELS = frozenset(
    {"Input", "Output", "Initializer", "InitializedInput"}
)


def _load_onnx_model(model_path: Path) -> Any:
    """Load the original ONNX model once for source-operator correlation."""
    onnx = importlib.import_module("onnx")
    return onnx.load(str(model_path), load_external_data=False)


def _onnx_source_operator_node_ids(
    model_path: Path,
    graph_collection: dict[str, Any],
) -> dict[str, list[_NodeRef]]:
    """Map top-level ONNX nodes to actual nodes emitted by the ONNX adapter."""
    model = _load_onnx_model(model_path)
    source_nodes = list(model.graph.node)
    _reject_unsupported_onnx_graphs(model, source_nodes)

    graphs = list(graph_collection.get("graphs", []) or [])
    if len(graphs) != 1 or not isinstance(graphs[0], dict):
        raise RuntimeError(
            "ONNX source-operator mapping requires exactly one adapter graph for "
            "the top-level ModelProto.graph."
        )
    graph = graphs[0]
    expected_graph_id = str(model.graph.name or "<main>")
    graph_id = graph.get("id")
    if graph_id != expected_graph_id:
        raise RuntimeError(
            "ONNX source-operator mapping graph mismatch: expected adapter graph "
            f"{expected_graph_id!r}, got {graph_id!r}."
        )

    adapter_nodes = _onnx_adapter_nodes(graph)
    operation_nodes = [
        node
        for node in adapter_nodes
        if node["label"] not in _ONNX_AUXILIARY_NODE_LABELS
    ]
    source_name_counts = Counter(
        str(getattr(source_node, "name", "") or "")
        for source_node in source_nodes
        if str(getattr(source_node, "name", "") or "")
    )
    collection_label = str(graph_collection["label"])
    mappings: dict[str, list[_NodeRef]] = {}
    used_node_ids: set[str] = set()
    for node_index, source_node in enumerate(source_nodes):
        source_name = str(getattr(source_node, "name", "") or "")
        adapter_node = _match_onnx_adapter_node(
            node_index,
            source_node,
            operation_nodes,
            source_name_is_unique=bool(
                source_name and source_name_counts[source_name] == 1
            ),
        )
        adapter_node_id = str(adapter_node["id"])
        if adapter_node_id in used_node_ids:
            raise RuntimeError(
                "ONNX source-operator mapping is not one-to-one: adapter node "
                f"{adapter_node_id!r} matched multiple source nodes."
            )
        used_node_ids.add(adapter_node_id)
        mappings[schema.onnx_source_operator_id(node_index)] = [
            _NodeRef(collection_label, str(graph_id), adapter_node_id)
        ]

    operation_node_ids = {str(node["id"]) for node in operation_nodes}
    if used_node_ids != operation_node_ids or len(mappings) != len(source_nodes):
        raise RuntimeError(
            "ONNX source-operator mapping is incomplete: every source and adapter "
            "operation node must participate in exactly one mapping."
        )
    return mappings


def _reject_unsupported_onnx_graphs(model: Any, source_nodes: list[Any]) -> None:
    """Reject ONNX structures that need a future graph-qualified identity."""
    if list(getattr(model, "functions", []) or []):
        raise UnsupportedConfigurationError(
            "Model Explorer ONNX source-operator mapping does not support ONNX "
            "functions; a future graph-qualified identity format is required."
        )
    for node_index, node in enumerate(source_nodes):
        for attribute in list(getattr(node, "attribute", []) or []):
            has_graph = False
            has_field = getattr(attribute, "HasField", None)
            if callable(has_field):
                try:
                    has_graph = bool(has_field("g"))
                except ValueError:
                    has_graph = False
            else:
                has_graph = getattr(attribute, "g", None) is not None
            if has_graph or list(getattr(attribute, "graphs", []) or []):
                raise UnsupportedConfigurationError(
                    "Model Explorer ONNX source-operator mapping does not support "
                    f"subgraphs (top-level node index {node_index}); a future "
                    "graph-qualified identity format is required."
                )


def _onnx_adapter_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return adapter nodes while rejecting invalid or duplicate public IDs."""
    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in list(graph.get("nodes", []) or []):
        if not isinstance(node, dict):
            raise RuntimeError("ONNX adapter graph contains an invalid node record.")
        node_id = node.get("id")
        label = node.get("label")
        if not isinstance(node_id, str) or not node_id:
            raise RuntimeError("ONNX adapter graph contains a node without an ID.")
        if not isinstance(label, str) or not label:
            raise RuntimeError(
                f"ONNX adapter node {node_id!r} does not have an operation label."
            )
        if node_id in seen_ids:
            raise RuntimeError(
                f"ONNX adapter graph contains duplicate node ID {node_id!r}."
            )
        seen_ids.add(node_id)
        nodes.append(node)
    return nodes


def _match_onnx_adapter_node(
    node_index: int,
    source_node: Any,
    adapter_nodes: list[dict[str, Any]],
    *,
    source_name_is_unique: bool,
) -> dict[str, Any]:
    """Require all authoritative native evidence to identify one adapter node."""
    source_name = str(getattr(source_node, "name", "") or "")
    source_outputs = {
        str(output_name)
        for output_name in list(getattr(source_node, "output", []) or [])
        if str(output_name)
    }
    name_candidates = (
        [node for node in adapter_nodes if node["id"] == source_name]
        if source_name_is_unique
        else []
    )
    output_candidates = (
        [
            node
            for node in adapter_nodes
            if source_outputs <= _onnx_output_tensor_tags(node)
        ]
        if source_outputs
        else []
    )

    if source_name_is_unique:
        _require_single_onnx_evidence_candidate(
            node_index, "node-name", source_name, name_candidates
        )
    if source_outputs:
        _require_single_onnx_evidence_candidate(
            node_index,
            "output tensor",
            ", ".join(sorted(source_outputs)),
            output_candidates,
        )
    if (
        name_candidates
        and output_candidates
        and name_candidates[0] is not output_candidates[0]
    ):
        raise RuntimeError(
            "ONNX source-operator mapping evidence disagrees for top-level node "
            f"index {node_index}: name selects {name_candidates[0]['id']!r} but "
            f"outputs select {output_candidates[0]['id']!r}."
        )

    candidates = name_candidates or output_candidates
    if not candidates:
        raise RuntimeError(
            "ONNX source-operator mapping found zero candidates for top-level node "
            f"index {node_index}; the node has no usable name or tagged output."
        )
    candidate = candidates[0]
    expected_label = _onnx_operation_label(source_node)
    if candidate["label"] != expected_label:
        raise RuntimeError(
            "ONNX source-operator mapping operation label mismatch for top-level "
            f"node index {node_index}: expected {expected_label!r}, got "
            f"{candidate['label']!r} on adapter node {candidate['id']!r}."
        )
    return candidate


def _require_single_onnx_evidence_candidate(
    node_index: int,
    evidence_kind: str,
    evidence: str,
    candidates: list[dict[str, Any]],
) -> None:
    """Require one unambiguous adapter candidate for one evidence channel."""
    if not candidates:
        raise RuntimeError(
            "ONNX source-operator mapping found zero candidates for top-level node "
            f"index {node_index} using {evidence_kind} evidence {evidence!r}."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "ONNX source-operator mapping found multiple candidates for top-level "
            f"node index {node_index} using {evidence_kind} evidence {evidence!r}: "
            f"{[node['id'] for node in candidates]!r}."
        )


def _onnx_operation_label(source_node: Any) -> str:
    """Return the domain-qualified label emitted by the ONNX adapter."""
    domain = str(getattr(source_node, "domain", "") or "")
    op_type = str(getattr(source_node, "op_type", "") or "")
    return op_type if domain in {"", "ai.onnx"} else f"{domain}::{op_type}"


def _onnx_output_tensor_tags(node: dict[str, Any]) -> set[str]:
    """Return non-empty ``__tensor_tag`` values from adapter output metadata."""
    metadata = node.get("outputsMetadata", []) or []
    items = metadata.values() if isinstance(metadata, dict) else metadata
    tags: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attrs", []) or []
        attr_items = attrs.values() if isinstance(attrs, dict) else attrs
        for attr in attr_items:
            if (
                isinstance(attr, dict)
                and attr.get("key") == "__tensor_tag"
                and isinstance(attr.get("value"), str)
                and attr["value"]
            ):
                tags.add(attr["value"])
    return tags


_PT2_OPERATION_NODE_OPS = frozenset({"call_function", "call_method", "call_module"})


def _load_pt2_exported_program(model_path: Path) -> Any:
    """Load a PT2 ExportedProgram for canonical source-node discovery."""
    torch = importlib.import_module("torch")
    return torch.export.load(model_path)


def _pt2_source_operator_node_ids(
    model_path: Path,
    graph_collection: dict[str, Any],
) -> dict[str, list[_NodeRef]]:
    """Map top-level PT2 operation names to built-in adapter node IDs."""
    exported_program = _load_pt2_exported_program(model_path)
    graph_module = getattr(exported_program, "graph_module", None)
    graph = getattr(graph_module, "graph", None)
    source_nodes = list(getattr(graph, "nodes", []) or [])
    if graph is None:
        raise RuntimeError(
            "PT2 source-operator mapping requires an ExportedProgram graph module."
        )

    graphs = list(graph_collection.get("graphs", []) or [])
    if len(graphs) != 1 or not isinstance(graphs[0], dict):
        raise RuntimeError(
            "PT2 source-operator mapping requires exactly one top-level adapter graph."
        )
    adapter_graph = graphs[0]
    graph_id = adapter_graph.get("id")
    if not isinstance(graph_id, str) or not graph_id:
        raise RuntimeError("PT2 adapter graph does not have a valid graph ID.")

    adapter_node_ids: set[str] = set()
    for adapter_node in list(adapter_graph.get("nodes", []) or []):
        if not isinstance(adapter_node, dict):
            raise RuntimeError("PT2 adapter graph contains an invalid node record.")
        adapter_node_id = adapter_node.get("id")
        if not isinstance(adapter_node_id, str) or not adapter_node_id:
            raise RuntimeError("PT2 adapter graph contains a node without an ID.")
        if adapter_node_id in adapter_node_ids:
            raise RuntimeError(
                f"PT2 adapter graph contains duplicate node ID {adapter_node_id!r}."
            )
        adapter_node_ids.add(adapter_node_id)

    source_node_names = [str(node.name) for node in source_nodes]
    if len(source_node_names) != len(set(source_node_names)):
        raise RuntimeError("PT2 ExportedProgram graph contains duplicate node names.")
    if set(source_node_names) != adapter_node_ids:
        missing = sorted(set(source_node_names) - adapter_node_ids)
        unexpected = sorted(adapter_node_ids - set(source_node_names))
        raise RuntimeError(
            "PT2 source-operator mapping does not match the built-in adapter graph: "
            f"missing node IDs {missing!r}; unexpected node IDs {unexpected!r}."
        )

    collection_label = str(graph_collection["label"])
    return {
        schema.pt2_source_operator_id(str(node.name)): [
            _NodeRef(collection_label, graph_id, str(node.name))
        ]
        for node in source_nodes
        if getattr(node, "op", None) in _PT2_OPERATION_NODE_OPS
    }


def _tflite_source_operator_node_ids(
    graph_collection: dict[str, Any],
) -> dict[str, list[_NodeRef]]:
    mappings: dict[str, list[_NodeRef]] = {}
    auxiliary_labels = {"GraphInputs", "GraphOutputs", "pseudo_const"}
    collection_label = graph_collection["label"]
    for subgraph_index, graph in enumerate(graph_collection.get("graphs", []) or []):
        if not isinstance(graph, dict):
            continue
        graph_id = graph.get("id")
        if not isinstance(graph_id, str) or not graph_id:
            continue
        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if (
                not isinstance(node_id, str)
                or not node_id.isdecimal()
                or node.get("label") in auxiliary_labels
            ):
                continue
            source_reference = (
                f"operator/{node_id}"
                if subgraph_index == 0
                else f"subgraph/{subgraph_index}/operator/{node_id}"
            )
            mappings[f"source_operator/{source_reference}"] = [
                _NodeRef(collection_label, graph_id, node_id)
            ]
    return mappings


def _spirv_source_operator_node_ids(
    graph_collection: dict[str, Any],
) -> dict[str, list[_NodeRef]]:
    """Map standalone SPIR-V result IDs to segment-zero source operators."""
    mappings: dict[str, list[_NodeRef]] = {}
    collection_label = graph_collection["label"]
    for graph in graph_collection.get("graphs", []) or []:
        if not isinstance(graph, dict):
            continue
        graph_id = graph.get("id")
        if not isinstance(graph_id, str) or not graph_id:
            continue
        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            match = re.fullmatch(r"op_(\d+)", node_id)
            if match is None:
                continue
            node_result_id = match.group(1)
            result_id = _node_string_attr(node, "spv id")
            if result_id is None:
                continue
            result_id = result_id.removeprefix("%").strip()
            if result_id != node_result_id:
                continue
            canonical_id = schema.vgf_source_operator_id(0, int(result_id))
            mappings.setdefault(canonical_id, []).append(
                _NodeRef(collection_label, graph_id, node_id)
            )
    return mappings


def _vgf_source_operator_node_ids(
    graph_collection: dict[str, Any],
) -> dict[str, list[_NodeRef]]:
    mappings: dict[str, list[_NodeRef]] = {}
    collection_label = graph_collection["label"]
    for graph in graph_collection.get("graphs", []) or []:
        if not isinstance(graph, dict):
            continue
        graph_id = graph.get("id")
        if not isinstance(graph_id, str) or not graph_id:
            continue
        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            match = re.fullmatch(r"(\d+)__op_(\d+)", node_id)
            if match is None:
                continue
            segment_index, node_result_id = match.groups()
            result_id = _node_string_attr(node, "spv id")
            if result_id is None:
                continue
            result_id = result_id.removeprefix("%").strip()
            if result_id != node_result_id:
                continue
            canonical_id = schema.vgf_source_operator_id(
                int(segment_index), int(result_id)
            )
            mappings.setdefault(canonical_id, []).append(
                _NodeRef(collection_label, graph_id, node_id)
            )
    return mappings


def _node_string_attr(node: dict[str, Any], key: str) -> str | None:
    for attr in node.get("attrs", []) or []:
        if (
            isinstance(attr, dict)
            and attr.get("key") == key
            and isinstance(attr.get("value"), str)
        ):
            return attr["value"]
    return None

