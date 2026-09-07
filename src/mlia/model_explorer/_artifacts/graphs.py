# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Entity hierarchy discovery and graph generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mlia.core import output_schema as schema

from mlia.model_explorer.graph_data import (
    model_explorer_namespace_depth,
    model_explorer_namespace_prefixes,
    model_explorer_normalized_namespace,
)

from .models import (
    _BUILT_IN_ENTITY_KINDS,
    _ONLY_SHOW_STATS_BEARING_ENTITY_NAMESPACES,
    _SOURCE_OPERATOR_KIND,
    _EntityKindHierarchy,
    _GeneratedEntityHierarchy,
    _GeneratedHierarchyProjection,
    _GeneratedModelArtifact,
    _GraphGroup,
    _GraphProjection,
    _HierarchyGraphCollection,
    _HierarchyPathProjection,
    _NodeRef,
)
from .serialization import _safe_filename, _write_json
from .source_model import _jsonable, _without_none


def _generated_hierarchy_graph_id(graph_name: str, source_graph_id: str) -> str:
    """Return the concrete graph ID for one generated hierarchy clone."""
    return f"{graph_name}-{source_graph_id}"


def _generated_hierarchy_graph_ids(
    source_graph_collection: dict[str, Any] | None,
    graph_name: str,
) -> dict[str, str]:
    if source_graph_collection is None:
        return {}
    return {
        graph_id: _generated_hierarchy_graph_id(graph_name, graph_id)
        for graph_id in _graphs_by_id(source_graph_collection, "source")
    }


def _namespace_group_node_id(namespace: str) -> str:
    """Return the Model Explorer artificial group node id for a namespace."""
    normalized = model_explorer_normalized_namespace(namespace)
    return f"{normalized}___group___"


def _result_entity_node_ids(
    result: dict[str, Any],
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
) -> dict[str, list[_NodeRef]]:
    """Resolve entities to source nodes by monotone propagation to a fixed point."""
    entities_by_id = _entities_by_id(result)
    children_by_parent = _entity_children_by_parent(entities_by_id)
    node_ids_by_entity = {
        entity_id: (
            list(source_operator_to_node_ids.get(entity_id, []))
            if _entity_kind(entity) == _SOURCE_OPERATOR_KIND
            else []
        )
        for entity_id, entity in entities_by_id.items()
    }

    changed = True
    while changed:
        changed = False
        for entity_id in entities_by_id:
            target = node_ids_by_entity[entity_id]
            previous_size = len(target)
            for child_id in children_by_parent.get(entity_id, []):
                _append_unique_items(target, node_ids_by_entity.get(child_id, []))
            changed = changed or len(target) != previous_size

    return node_ids_by_entity


def _append_unique_items(target: list[_NodeRef], items: list[_NodeRef]) -> None:
    """Append node references while preserving first-seen order."""
    for item in items:
        if item not in target:
            target.append(item)


def _graphs_by_id(
    graph_collection: dict[str, Any], description: str
) -> dict[str, dict[str, Any]]:
    """Return valid graphs in serialized order while rejecting ambiguous IDs."""
    graphs: dict[str, dict[str, Any]] = {}
    for graph in graph_collection.get("graphs", []) or []:
        if not isinstance(graph, dict):
            continue
        graph_id = graph.get("id")
        if not isinstance(graph_id, str) or not graph_id:
            continue
        if graph_id in graphs:
            raise RuntimeError(
                f"Duplicate graph ID {graph_id!r} in {description} graphs."
            )
        graphs[graph_id] = graph
    return graphs


def _graph_projection(
    origin: tuple[str, str],
    collection_label: str,
    graph: dict[str, Any],
    canonical_ref_by_node_ref: dict[_NodeRef, _NodeRef] | None = None,
    terminal_complete_path_group_refs: frozenset[_NodeRef] = frozenset(),
) -> _GraphProjection:
    """Derive source identity or validate an explicit generated projection."""
    graph_id = graph.get("id")
    if not isinstance(graph_id, str) or not graph_id:
        raise RuntimeError("Graph projection requires a public graph ID.")

    ordered_leaf_refs: list[_NodeRef] = []
    ordered_leaf_node_ids: list[str] = []
    leaf_ids_by_canonical_ref: dict[_NodeRef, list[str]] = {}
    serialized_nodes: list[tuple[dict[str, Any], _NodeRef, str]] = []
    serialized_node_ids: set[str] = set()
    runtime_group_ids: set[str] = set()
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if node_id in serialized_node_ids:
            raise RuntimeError(
                f"Duplicate node ID {node_id!r} in graph projection "
                f"{collection_label!r}/{graph_id!r}."
            )
        serialized_node_ids.add(node_id)
        node_ref = _NodeRef(collection_label, graph_id, node_id)
        if canonical_ref_by_node_ref is None:
            canonical_ref = _NodeRef(origin[0], origin[1], node_id)
        else:
            mapped_canonical_ref = canonical_ref_by_node_ref.get(node_ref)
            if mapped_canonical_ref is None:
                raise RuntimeError(
                    "Generated graph canonical mapping is incomplete for serialized "
                    f"node {node_ref!r}; generated node IDs are never treated as "
                    "canonical source identities."
                )
            canonical_ref = mapped_canonical_ref
            if (canonical_ref.collection_label, canonical_ref.graph_id) != origin:
                raise RuntimeError(
                    "Generated graph canonical mapping points outside its source "
                    f"projection: {node_ref!r} maps to {canonical_ref!r}, expected "
                    f"origin {origin!r}."
                )
        ordered_leaf_refs.append(canonical_ref)
        ordered_leaf_node_ids.append(node_id)
        leaf_ids_by_canonical_ref.setdefault(canonical_ref, []).append(node_id)
        serialized_nodes.append((node, canonical_ref, node_id))

        namespace = node.get("namespace")
        if isinstance(namespace, str) and namespace:
            runtime_group_ids.update(
                _namespace_group_node_id(prefix)
                for prefix in model_explorer_namespace_prefixes(namespace)
            )

    if canonical_ref_by_node_ref is not None:
        unexpected_mapping_refs = sorted(
            (
                node_ref
                for node_ref in canonical_ref_by_node_ref
                if node_ref.collection_label == collection_label
                and node_ref.graph_id == graph_id
                and node_ref.node_id not in serialized_node_ids
            ),
            key=str,
        )
        if unexpected_mapping_refs:
            raise RuntimeError(
                "Generated graph canonical mapping contains unexpected nodes for "
                f"{collection_label!r}/{graph_id!r}: {unexpected_mapping_refs!r}."
            )

    colliding_node_ids = sorted(serialized_node_ids & runtime_group_ids)
    if colliding_node_ids:
        raise RuntimeError(
            "Serialized graph node IDs collide with Model Explorer runtime-created "
            f"namespace group IDs {colliding_node_ids!r} in graph projection "
            f"{collection_label!r}/{graph_id!r}; choose different serialized node "
            "IDs or namespaces so the runtime can address every leaf."
        )

    group_leaves: dict[str, list[tuple[_NodeRef, str]]] = {}
    for node, canonical_ref, node_id in serialized_nodes:
        namespace = node.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            continue
        for namespace_prefix in model_explorer_namespace_prefixes(namespace):
            group_leaves.setdefault(namespace_prefix, []).append(
                (canonical_ref, node_id)
            )

    groups = tuple(
        _GraphGroup(
            node_ref=_NodeRef(
                collection_label,
                graph_id,
                _namespace_group_node_id(namespace),
            ),
            namespace_depth=model_explorer_namespace_depth(namespace),
            ordered_leaf_ids=tuple(dict.fromkeys(node_id for _, node_id in leaves)),
            canonical_leaf_refs=frozenset(canonical_ref for canonical_ref, _ in leaves),
        )
        for namespace, leaves in group_leaves.items()
    )
    return _GraphProjection(
        origin=origin,
        collection_label=collection_label,
        graph_id=graph_id,
        ordered_leaf_refs=tuple(ordered_leaf_refs),
        ordered_leaf_node_ids=tuple(ordered_leaf_node_ids),
        leaf_ids_by_canonical_ref={
            canonical_ref: tuple(node_ids)
            for canonical_ref, node_ids in leaf_ids_by_canonical_ref.items()
        },
        groups=groups,
        terminal_complete_path_group_ids=frozenset(
            node_ref.node_id
            for node_ref in terminal_complete_path_group_refs
            if node_ref.collection_label == collection_label
            and node_ref.graph_id == graph_id
        ),
    )


def _projection_identity(projection: _GraphProjection) -> tuple[str, str]:
    return (projection.collection_label, projection.graph_id)


def _graph_mapping_target_ids(
    source_group: _GraphGroup,
    target_projection: _GraphProjection,
) -> list[str]:
    """Choose every deepest exact group, otherwise all corresponding leaves."""
    exact_groups = [
        group
        for group in target_projection.groups
        if group.canonical_leaf_refs
        and group.canonical_leaf_refs == source_group.canonical_leaf_refs
    ]
    if exact_groups:
        terminal_groups = [
            group
            for group in exact_groups
            if group.node_ref.node_id
            in target_projection.terminal_complete_path_group_ids
        ]
        if terminal_groups:
            return [group.node_ref.node_id for group in terminal_groups]
        deepest_depth = max(group.namespace_depth for group in exact_groups)
        return [
            group.node_ref.node_id
            for group in exact_groups
            if group.namespace_depth == deepest_depth
        ]

    missing_leaf_refs = (
        source_group.canonical_leaf_refs
        - target_projection.leaf_ids_by_canonical_ref.keys()
    )
    if missing_leaf_refs:
        missing_node_ids = sorted(node_ref.node_id for node_ref in missing_leaf_refs)
        raise RuntimeError(
            "Corresponding graph projection is missing canonical target leaves "
            f"{missing_node_ids!r} in {target_projection.collection_label!r}/"
            f"{target_projection.graph_id!r}."
        )
    return list(
        dict.fromkeys(
            node_id
            for canonical_ref in target_projection.ordered_leaf_refs
            if canonical_ref in source_group.canonical_leaf_refs
            for node_id in target_projection.leaf_ids_by_canonical_ref[canonical_ref]
        )
    )


def _entity_children_by_parent(
    entities_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    children_by_parent: dict[str, list[str]] = {}
    for entity_id, entity in entities_by_id.items():
        for parent_id in _entity_parent_ids(entity):
            children_by_parent.setdefault(parent_id, []).append(entity_id)
    for entity_id, entity in entities_by_id.items():
        for child_id in _entity_child_ids(entity):
            child = entities_by_id.get(child_id)
            if child is not None and not _entity_parent_ids(child):
                children_by_parent.setdefault(entity_id, []).append(child_id)
    return {
        parent_id: list(dict.fromkeys(child_ids))
        for parent_id, child_ids in children_by_parent.items()
    }


def _write_entity_subgraph_model_file(
    model_path: Path | None,
    root: Path,
    output: dict[str, Any],
    source_graph_collection: dict[str, Any] | None,
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
) -> list[_GeneratedModelArtifact]:
    """Write and retain one graph collection per entity-kind hierarchy."""
    if source_graph_collection is None:
        return []

    generated_hierarchies = _generated_entity_hierarchies(output)
    if not generated_hierarchies:
        return []

    results = output.get("results", [])
    if not isinstance(results, list):
        return []

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_stem = _safe_filename(model_path.stem) if model_path is not None else "model"
    artifacts: list[_GeneratedModelArtifact] = []
    for generated in generated_hierarchies:
        result = results[generated.result_index]
        if not isinstance(result, dict):
            continue
        path = model_dir / (
            f"{_entity_hierarchy_model_stem(model_stem, generated.graph_name)}.json"
        )
        generated_collection = _source_model_with_entity_namespaces(
            result,
            source_graph_collection,
            generated.hierarchy,
            generated.graph_name,
            generated.collection_label,
            source_operator_to_node_ids,
        )
        _write_json(path, generated_collection.graph_collection)
        artifacts.append(
            _GeneratedModelArtifact(
                path,
                generated,
                generated_collection.graph_collection,
                generated_collection.projection,
            )
        )
    return artifacts


def _generated_entity_hierarchies(
    output: dict[str, Any],
) -> list[_GeneratedEntityHierarchy]:
    """Describe hierarchy collections with globally collision-free identities."""
    results = output.get("results", [])
    if not isinstance(results, list):
        return []

    candidates: list[tuple[int, str, _EntityKindHierarchy]] = []
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        result_kind = str(result.get("kind") or "result")
        for hierarchy in _result_entity_kind_hierarchies(result):
            candidates.append((result_index, result_kind, hierarchy))

    base_counts = Counter(
        hierarchy.graph_name.casefold() for _, _, hierarchy in candidates
    )
    result_local_counts = Counter(
        (hierarchy.graph_name.casefold(), result_index)
        for result_index, _, hierarchy in candidates
    )
    result_local_ordinals: Counter[tuple[str, int]] = Counter()
    generated: list[_GeneratedEntityHierarchy] = []
    for result_index, result_kind, hierarchy in candidates:
        base_name = hierarchy.graph_name
        base_key = base_name.casefold()
        graph_name = base_name
        collection_label = base_name
        if base_counts[base_key] > 1:
            result_key = (base_key, result_index)
            result_local_ordinals[result_key] += 1
            graph_name = f"{base_name}-result-{result_index + 1}"
            label_qualifier = f"{result_kind} result {result_index + 1}"
            if result_local_counts[result_key] > 1:
                hierarchy_ordinal = result_local_ordinals[result_key]
                graph_name = f"{graph_name}-hierarchy-{hierarchy_ordinal}"
                readable_path = (
                    " > ".join(hierarchy.backend_kinds) or "entity hierarchy"
                )
                label_qualifier = (
                    f"{label_qualifier}, hierarchy {hierarchy_ordinal}: {readable_path}"
                )
            collection_label = f"{base_name} [{label_qualifier}]"
        generated.append(
            _GeneratedEntityHierarchy(
                result_index,
                result_kind,
                hierarchy,
                graph_name,
                collection_label,
            )
        )
    return generated


def _generated_model_artifacts_by_result(
    generated_artifacts: list[_GeneratedModelArtifact],
) -> dict[int, list[_GeneratedModelArtifact]]:
    """Return generated hierarchy artifacts grouped by result index."""
    by_result: dict[int, list[_GeneratedModelArtifact]] = {}
    for artifact in generated_artifacts:
        by_result.setdefault(artifact.descriptor.result_index, []).append(artifact)
    return by_result


_ENTITY_HIERARCHY_SUFFIX = "-hierarchy-"


def _entity_hierarchy_model_stem(model_stem: str, graph_name: str) -> str:
    return f"{model_stem}{_ENTITY_HIERARCHY_SUFFIX}{graph_name}"


def _entity_hierarchy_graph_name(backend_kinds: tuple[str, ...]) -> str:
    """Return the generated graph name for a backend hierarchy path."""
    if not backend_kinds:
        return "entity_hierarchy"
    return _safe_filename("_".join(backend_kinds)).replace("-", "_")


def _entity_hierarchy_graph(
    graph: dict[str, Any],
    generated_graph_ids: dict[str, str],
    source_operator_paths_by_node_id: dict[str, tuple[str, ...]],
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    """Clone one graph, splitting multi-path source nodes into sub-operators."""
    source_graph_id = graph.get("id")
    if not isinstance(source_graph_id, str) or not source_graph_id:
        raise RuntimeError("Generated hierarchy source graph requires a public ID.")
    generated_graph_id = generated_graph_ids.get(source_graph_id)
    if generated_graph_id is None:
        raise RuntimeError(
            "Generated hierarchy graph ID allocation is missing source graph "
            f"{source_graph_id!r}."
        )

    source_nodes = [
        node
        for node in (graph.get("nodes", []) or [])
        if isinstance(node, dict)
    ]
    source_node_ids: set[str] = set()
    for node in source_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if node_id in source_node_ids:
            raise RuntimeError(
                f"Duplicate node ID {node_id!r} in source graph {source_graph_id!r}."
            )
        source_node_ids.add(node_id)

    generated_node_ids: set[str] = set()
    generated_ids_by_source_id: dict[str, tuple[str, ...]] = {}
    for node in source_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        paths = tuple(
            sorted(dict.fromkeys(source_operator_paths_by_node_id.get(node_id, ())))
        )
        if len(paths) > 1:
            incoming_edges = node.get("incomingEdges", []) or []
            if any(
                isinstance(edge, dict) and edge.get("sourceNodeId") == node_id
                for edge in incoming_edges
            ):
                raise RuntimeError(
                    "Cannot split self-referential source node "
                    f"{node_id!r} in graph {source_graph_id!r} across multiple "
                    "complete hierarchy paths; a linear sub-operator chain cannot "
                    "preserve a self-feedback edge without introducing a cycle."
                )
            allocated_ids: list[str] = []
            for path in paths:
                generated_node_id = _allocate_sub_operator_node_id(
                    source_graph_id,
                    node_id,
                    path,
                    source_node_ids | generated_node_ids | set(allocated_ids),
                )
                allocated_ids.append(generated_node_id)
            generated_ids = tuple(allocated_ids)
        else:
            generated_ids = (node_id,)
        generated_node_ids.update(generated_ids)
        generated_ids_by_source_id[node_id] = generated_ids

    output_ports_by_node_id = _graph_output_ports(source_nodes)
    cloned_nodes: list[dict[str, Any]] = []
    for node in source_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            cloned_nodes.append(_without_none(_jsonable(node)))
            continue
        paths = tuple(
            sorted(dict.fromkeys(source_operator_paths_by_node_id.get(node_id, ())))
        )
        generated_ids = generated_ids_by_source_id[node_id]
        for index, generated_node_id in enumerate(generated_ids):
            cloned_node = _without_none(_jsonable(node))
            cloned_node["id"] = generated_node_id
            cloned_node["namespace"] = (
                paths[index] if len(paths) > 1 else (paths[0] if paths else "")
            )
            if len(paths) > 1:
                _number_sub_operator_display_label(cloned_node, index + 1)
            if index == 0:
                cloned_node["incomingEdges"] = _redirect_incoming_edges(
                    cloned_node.get("incomingEdges", []) or [],
                    generated_ids_by_source_id,
                )
            else:
                output_ports = output_ports_by_node_id.get(node_id, (None,))
                cloned_node["incomingEdges"] = _sub_operator_chain_edges(
                    generated_ids[index - 1], output_ports
                )
                cloned_node["inputsMetadata"] = [
                    {"id": output_port, "attrs": []}
                    for output_port in output_ports
                    if output_port is not None
                ]
            cloned_nodes.append(cloned_node)

    cloned = _without_none(_jsonable(graph))
    cloned["id"] = generated_graph_id
    cloned["nodes"] = cloned_nodes
    for node in cloned_nodes:
        if "subgraphIds" not in node:
            continue
        subgraph_ids = node.get("subgraphIds")
        if not isinstance(subgraph_ids, list):
            raise RuntimeError(
                f"Node {node.get('id')!r} in source graph {source_graph_id!r} has "
                "non-list subgraphIds."
            )
        rewritten_subgraph_ids: list[str] = []
        for subgraph_id in subgraph_ids:
            if not isinstance(subgraph_id, str) or not subgraph_id:
                raise RuntimeError(
                    f"Node {node.get('id')!r} in source graph "
                    f"{source_graph_id!r} has an invalid subgraph ID."
                )
            generated_subgraph_id = generated_graph_ids.get(subgraph_id)
            if generated_subgraph_id is None:
                raise RuntimeError(
                    f"Node {node.get('id')!r} in source graph "
                    f"{source_graph_id!r} references missing subgraph "
                    f"{subgraph_id!r}."
                )
            rewritten_subgraph_ids.append(generated_subgraph_id)
        node["subgraphIds"] = rewritten_subgraph_ids
    return cloned, generated_ids_by_source_id


def _number_sub_operator_display_label(node: dict[str, Any], position: int) -> None:
    """Suffix a usable cloned label while preserving existing fallback behavior.

    Model Explorer currently consumes the serialized ``label`` field and handles
    missing, empty, or non-string labels itself. Generated hierarchy nodes retain
    those values unchanged rather than introducing an ID-derived label that is not
    present on the canonical source node.
    """
    label = node.get("label")
    if isinstance(label, str) and label.strip():
        node["label"] = f"{label} (#{position})"


_MAX_GENERATED_NODE_ID_LENGTH = 160
_SUB_OPERATOR_DIGEST_LENGTH = 40
_SUB_OPERATOR_ID_MARKER = "__mlia_sub_operator__"


def _sub_operator_node_id(
    source_graph_id: str,
    canonical_node_id: str,
    complete_path: str,
    allocation_attempt: int = 0,
) -> str:
    """Return one bounded stable candidate ID for a visual sub-operator."""
    identity = json.dumps(
        [source_graph_id, canonical_node_id, complete_path, allocation_attempt],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:_SUB_OPERATOR_DIGEST_LENGTH]
    suffix = f"{_SUB_OPERATOR_ID_MARKER}{digest}"
    readable_length = max(1, _MAX_GENERATED_NODE_ID_LENGTH - len(suffix))
    return f"{canonical_node_id[:readable_length]}{suffix}"


def _allocate_sub_operator_node_id(
    source_graph_id: str,
    canonical_node_id: str,
    complete_path: str,
    occupied_node_ids: set[str],
) -> str:
    """Allocate a deterministic bounded ID without colliding with serialized IDs."""
    allocation_attempt = 0
    while True:
        candidate = _sub_operator_node_id(
            source_graph_id,
            canonical_node_id,
            complete_path,
            allocation_attempt,
        )
        if candidate not in occupied_node_ids:
            return candidate
        allocation_attempt += 1


def _graph_output_ports(
    nodes: list[dict[str, Any]],
) -> dict[str, tuple[str | None, ...]]:
    """Return every declared or consumed output port for each source node."""
    ports_by_node_id: dict[str, list[str | None]] = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        metadata = node.get("outputsMetadata", []) or []
        metadata_items = metadata.values() if isinstance(metadata, dict) else metadata
        for item in metadata_items:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            port = str(item["id"])
            if port not in ports_by_node_id.setdefault(node_id, []):
                ports_by_node_id[node_id].append(port)
    for consumer in nodes:
        for edge in consumer.get("incomingEdges", []) or []:
            if not isinstance(edge, dict):
                continue
            source_node_id = edge.get("sourceNodeId")
            if not isinstance(source_node_id, str) or not source_node_id:
                continue
            port_value = edge.get("sourceNodeOutputId")
            consumed_port = str(port_value) if port_value is not None else None
            if consumed_port not in ports_by_node_id.setdefault(source_node_id, []):
                ports_by_node_id[source_node_id].append(consumed_port)
    return {
        node_id: tuple(ports_by_node_id.get(node_id, []) or [None])
        for node_id in {
            str(node["id"])
            for node in nodes
            if isinstance(node.get("id"), str) and node.get("id")
        }
    }


def _redirect_incoming_edges(
    incoming_edges: list[Any],
    generated_ids_by_source_id: dict[str, tuple[str, ...]],
) -> list[Any]:
    """Redirect each original edge once to the final upstream sub-operator."""
    redirected: list[Any] = []
    for edge in incoming_edges:
        cloned_edge = _without_none(_jsonable(edge))
        if isinstance(cloned_edge, dict):
            source_node_id = cloned_edge.get("sourceNodeId")
            if isinstance(source_node_id, str):
                generated_ids = generated_ids_by_source_id.get(source_node_id)
                if generated_ids:
                    cloned_edge["sourceNodeId"] = generated_ids[-1]
        redirected.append(cloned_edge)
    return redirected


def _sub_operator_chain_edges(
    previous_node_id: str,
    output_ports: tuple[str | None, ...],
) -> list[dict[str, str]]:
    """Pass every original output port to the next sub-operator in the chain."""
    edges: list[dict[str, str]] = []
    for output_port in output_ports:
        edge = {"sourceNodeId": previous_node_id}
        if output_port is not None:
            edge["sourceNodeOutputId"] = output_port
            edge["targetNodeInputId"] = output_port
        edges.append(edge)
    return edges


def _source_model_with_entity_namespaces(
    result: dict[str, Any],
    source_graph_collection: dict[str, Any],
    hierarchy: _EntityKindHierarchy,
    graph_name: str,
    collection_label: str,
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
) -> _HierarchyGraphCollection:
    """Return one result's hierarchy graph and explicit canonical correspondence."""
    graph_collection = _without_none(_jsonable(source_graph_collection))
    source_collection_label = str(source_graph_collection["label"])
    source_graph_node_ids = _source_graph_node_ids(source_graph_collection)
    stats_bearing_entity_ids = _stats_bearing_entity_ids([result])
    entities = _entities_by_id(result)
    entity_node_ids = _result_entity_node_ids(result, source_operator_to_node_ids)
    hierarchy_paths = _entity_hierarchy_complete_paths(
        entities,
        hierarchy,
        stats_bearing_entity_ids,
    )

    source_operator_by_ref: dict[_NodeRef, str] = {}
    for source_operator_id, node_refs in source_operator_to_node_ids.items():
        if _entity_kind(entities.get(source_operator_id)) != _SOURCE_OPERATOR_KIND:
            continue
        for node_ref in node_refs:
            if node_ref not in source_graph_node_ids:
                continue
            previous = source_operator_by_ref.get(node_ref)
            if previous is not None and previous != source_operator_id:
                raise RuntimeError(
                    "A source graph node maps to multiple canonical source operators: "
                    f"{node_ref!r} maps to {previous!r} and {source_operator_id!r}."
                )
            source_operator_by_ref[node_ref] = source_operator_id

    graph_collection["label"] = collection_label
    generated_graph_ids = _generated_hierarchy_graph_ids(graph_collection, graph_name)
    generated_refs_by_canonical_ref: dict[_NodeRef, tuple[_NodeRef, ...]] = {}
    canonical_ref_by_generated_ref: dict[_NodeRef, _NodeRef] = {}
    source_operator_by_generated_ref: dict[_NodeRef, str] = {}
    terminal_complete_path_group_refs: set[_NodeRef] = set()
    generated_graphs: list[dict[str, Any]] = []
    for graph in graph_collection.get("graphs", []) or []:
        if not isinstance(graph, dict):
            continue
        source_graph_id = graph.get("id")
        if not isinstance(source_graph_id, str) or not source_graph_id:
            continue
        source_paths_by_node_id = {
            node_ref.node_id: hierarchy_paths.source_operator_paths_by_id.get(
                source_operator_id, ()
            )
            for node_ref, source_operator_id in source_operator_by_ref.items()
            if node_ref.graph_id == source_graph_id
        }
        generated_graph, generated_ids_by_source_id = _entity_hierarchy_graph(
            graph,
            generated_graph_ids,
            source_paths_by_node_id,
        )
        generated_graphs.append(generated_graph)
        generated_graph_id = str(generated_graph["id"])
        for source_node_id, generated_node_ids in generated_ids_by_source_id.items():
            canonical_ref = _NodeRef(
                source_collection_label, source_graph_id, source_node_id
            )
            generated_refs = tuple(
                _NodeRef(collection_label, generated_graph_id, generated_node_id)
                for generated_node_id in generated_node_ids
            )
            generated_refs_by_canonical_ref[canonical_ref] = generated_refs
            for generated_ref in generated_refs:
                canonical_ref_by_generated_ref[generated_ref] = canonical_ref
                mapped_source_operator_id = source_operator_by_ref.get(canonical_ref)
                if mapped_source_operator_id is not None:
                    source_operator_by_generated_ref[generated_ref] = (
                        mapped_source_operator_id
                    )
            complete_paths = source_paths_by_node_id.get(source_node_id, ())
            terminal_complete_path_group_refs.update(
                _NodeRef(
                    collection_label,
                    generated_graph_id,
                    _namespace_group_node_id(complete_path),
                )
                for complete_path in complete_paths
            )
    graph_collection["graphs"] = generated_graphs

    group_refs_by_entity_id: dict[str, list[_NodeRef]] = {}
    generated_graphs_by_id = _graphs_by_id(
        graph_collection, f"generated collection {collection_label!r}"
    )
    for source_graph_id, generated_graph_id in generated_graph_ids.items():
        projected_graph = generated_graphs_by_id.get(generated_graph_id)
        if projected_graph is None:
            continue
        runtime_group_ids = {
            group.node_ref.node_id
            for group in _graph_projection(
                (source_collection_label, source_graph_id),
                collection_label,
                projected_graph,
                canonical_ref_by_generated_ref,
            ).groups
        }
        for entity_id, complete_paths in hierarchy_paths.entity_paths_by_id.items():
            if not any(
                node_ref.collection_label == source_collection_label
                and node_ref.graph_id == source_graph_id
                for node_ref in entity_node_ids.get(entity_id, [])
            ):
                continue
            for complete_path in complete_paths:
                group_node_id = _namespace_group_node_id(complete_path)
                if group_node_id in runtime_group_ids:
                    group_refs_by_entity_id.setdefault(entity_id, []).append(
                        _NodeRef(collection_label, generated_graph_id, group_node_id)
                    )

    projection = _GeneratedHierarchyProjection(
        generated_refs_by_canonical_ref=generated_refs_by_canonical_ref,
        canonical_ref_by_generated_ref=canonical_ref_by_generated_ref,
        source_operator_by_generated_ref=source_operator_by_generated_ref,
        group_refs_by_entity_id={
            entity_id: tuple(dict.fromkeys(node_refs))
            for entity_id, node_refs in group_refs_by_entity_id.items()
        },
        terminal_complete_path_group_refs=frozenset(
            terminal_complete_path_group_refs
        ),
    )
    return _HierarchyGraphCollection(graph_collection, projection)


def _strip_visible_ancestor_name_prefix(
    value: str, visible_ancestor_names: list[str]
) -> str:
    """Strip an already-visible ancestor prefix from an entity display name."""
    for ancestor_name in sorted(visible_ancestor_names, key=len, reverse=True):
        prefix = f"{ancestor_name}/"
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _entity_namespace_component_with_id(
    display_name: str,
    entity_id: str,
    *,
    attempt: int = 1,
    preserve_hierarchy: bool = False,
) -> str:
    """Return a display component disambiguated by a namespace-safe entity id."""
    safe_entity_id = quote(entity_id, safe="-._~")
    if attempt >= 3:
        # This injective fallback has no Model Explorer namespace delimiters. It is
        # used only after readable ID suffixes themselves collide with another
        # entity name, and guarantees that another allocation pass can converge.
        return f"MLIA entity {safe_entity_id}"

    safe_display_name = (
        display_name.replace(";", "%3B") if preserve_hierarchy else display_name
    )
    suffix = (
        f" [{safe_entity_id}]"
        if attempt == 1
        else f" [{safe_entity_id} entity-id]"
    )
    return _escape_entity_namespace_component(f"{safe_display_name}{suffix}")


def _escape_entity_namespace_component(value: str) -> str:
    """Escape literal slashes in one Model Explorer namespace component."""
    # Entity names are raw labels, not preformatted Model Explorer namespaces.
    # Escape literal slashes so a name like 'cascade/0' remains one namespace
    # component. Hierarchy is introduced only when complete paths join components
    # with an unescaped '/'.
    return value.replace("/", r"\/")


def _entity_display_name(entity: dict[str, Any]) -> str:
    """Return the raw display name for an entity."""
    return str(entity.get("name") or entity.get("id") or "")


def _source_graph_node_ids(source_graph_collection: dict[str, Any]) -> set[_NodeRef]:
    collection_label = source_graph_collection["label"]
    return {
        _NodeRef(collection_label, graph_id, node_id)
        for graph in source_graph_collection.get("graphs", [])
        if isinstance(graph, dict)
        for graph_id in [graph.get("id")]
        if isinstance(graph_id, str) and graph_id
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
        for node_id in [node.get("id")]
        if isinstance(node_id, str)
    }


def _entities_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return entity records keyed by id."""
    return {
        str(entity["id"]): entity
        for entity in result.get("entities", []) or []
        if isinstance(entity, dict) and entity.get("id") is not None
    }


def _entity_id(entity: dict[str, Any]) -> str:
    """Return an entity id as text."""
    return str(entity.get("id") or "")


def _entity_kind(entity: Any) -> str | None:
    """Return an entity kind when the record has a valid schema-1.2 kind."""
    if not isinstance(entity, dict):
        return None
    kind = entity.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        return None
    return kind


def _result_entity_kind_hierarchies(
    result: dict[str, Any],
) -> list[_EntityKindHierarchy]:
    """Return source-operator hierarchies represented by the result's entities."""
    metadata = _entity_kind_metadata_by_id(result.get("entity_kinds", []))
    present_kinds = {
        kind
        for entity in (result.get("entities", []) or [])
        for kind in [_entity_kind(entity)]
        if kind is not None
    }
    hierarchies: dict[tuple[str, ...], _EntityKindHierarchy] = {}
    for upward_path, recursive_kinds in _source_operator_hierarchy_paths(
        metadata, present_kinds
    ):
        path = tuple(reversed(upward_path))
        hierarchies[path] = _EntityKindHierarchy(path, recursive_kinds)
    return _maximal_entity_kind_hierarchies(
        [hierarchies[path] for path in sorted(hierarchies)]
    )


def _maximal_entity_kind_hierarchies(
    hierarchies: list[_EntityKindHierarchy],
) -> list[_EntityKindHierarchy]:
    """Discard coarse hierarchy prefixes already represented by finer paths."""
    return [
        hierarchy
        for hierarchy in hierarchies
        if not any(
            len(other.backend_kinds) > len(hierarchy.backend_kinds)
            and other.backend_kinds[: len(hierarchy.backend_kinds)]
            == hierarchy.backend_kinds
            for other in hierarchies
        )
    ]


def _entity_kind_metadata_by_id(value: Any) -> dict[str, dict[str, Any]]:
    """Return entity kind metadata keyed by kind id."""
    metadata = {
        kind_id: {
            "id": definition.id,
            "parent_kinds": list(definition.parent_kinds),
            "child_kinds": list(definition.child_kinds),
        }
        for kind_id, definition in schema.WELL_KNOWN_ENTITY_KIND_DEFINITIONS.items()
    }
    for item in value or []:
        metadata[item["id"]] = item
    return metadata


def _entity_kind_parents_by_child(
    metadata: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Return declared parent kinds for each metadata kind and built-in child."""
    parents_by_child: dict[str, list[str]] = {}
    for kind_id, item in metadata.items():
        for parent_kind in item.get("parent_kinds", []):
            parents_by_child.setdefault(kind_id, []).append(parent_kind)
        for child_kind in item.get("child_kinds", []):
            parents_by_child.setdefault(child_kind, []).append(kind_id)
    return {
        child: list(dict.fromkeys(parents))
        for child, parents in parents_by_child.items()
    }


def _source_operator_parent_kinds(
    metadata: dict[str, dict[str, Any]], present_kinds: set[str]
) -> set[str]:
    """Return present backend kinds that directly contain source operators."""
    parents_by_child = _entity_kind_parents_by_child(metadata)
    return {
        kind
        for kind in parents_by_child.get(_SOURCE_OPERATOR_KIND, [])
        if kind in present_kinds and kind not in _BUILT_IN_ENTITY_KINDS
    }


def _source_operator_hierarchy_paths(
    metadata: dict[str, dict[str, Any]], present_kinds: set[str]
) -> list[tuple[tuple[str, ...], frozenset[str]]]:
    """Trace hierarchies whose source-adjacent backend kind is present."""
    parents_by_child = _entity_kind_parents_by_child(metadata)
    paths: dict[tuple[str, ...], frozenset[str]] = {}

    def record_path(path: tuple[str, ...], recursive_kinds: frozenset[str]) -> None:
        paths[path] = paths.get(path, frozenset()) | recursive_kinds

    pending: list[tuple[str, tuple[str, ...], frozenset[str]]] = [
        (parent_kind, (_SOURCE_OPERATOR_KIND, parent_kind), frozenset())
        for parent_kind in sorted(
            _source_operator_parent_kinds(metadata, present_kinds), reverse=True
        )
    ]
    while pending:
        kind, path, recursive_kinds = pending.pop()
        parents = parents_by_child.get(kind, [])
        if kind in parents:
            recursive_kinds |= frozenset({kind})
        if not parents:
            record_path(path, recursive_kinds)
            continue
        for parent in sorted(parents, reverse=True):
            if parent in path:
                record_path(path, recursive_kinds | frozenset({parent}))
                continue
            pending.append((parent, (*path, parent), recursive_kinds))
    return [(path, recursive_kinds) for path, recursive_kinds in sorted(paths.items())]


def _stats_bearing_entity_ids(results: list[Any]) -> set[str]:
    """Return entity IDs that have any metric-bearing breakdown in any result."""
    return {
        str(entity_id)
        for result in results
        if isinstance(result, dict)
        for breakdown in (result.get("breakdowns", []) or [])
        if isinstance(breakdown, dict)
        for entity_id in [breakdown.get("entity_id")]
        if entity_id is not None and _breakdown_has_metric_value(breakdown)
    }


def _breakdown_has_metric_value(breakdown: dict[str, Any]) -> bool:
    """Return whether a breakdown has any concrete metric value."""
    metrics = breakdown.get("metrics", []) or []
    return any(
        isinstance(metric, dict) and metric.get("value") is not None
        for metric in metrics
    )


def _entity_hierarchy_complete_paths(
    entities_by_id: dict[str, dict[str, Any]],
    hierarchy: _EntityKindHierarchy,
    stats_bearing_entity_ids: set[str],
) -> _HierarchyPathProjection:
    """Enumerate and render every complete ancestry path in one hierarchy."""
    lineages_by_entity_id = {
        entity_id: _entity_complete_lineages(
            entity_id, entities_by_id, hierarchy, frozenset()
        )
        for entity_id, entity in sorted(entities_by_id.items())
        if _entity_kind(entity) == _SOURCE_OPERATOR_KIND
        or _entity_belongs_to_hierarchy(entity, hierarchy)
    }

    component_display_by_prefix: dict[tuple[str, ...], str] = {}
    component_entity_id_by_prefix: dict[tuple[str, ...], str] = {}
    component_prefixes_by_lineage: dict[
        tuple[str, ...], tuple[tuple[str, ...], ...]
    ] = {}
    for lineages in lineages_by_entity_id.values():
        for lineage in lineages:
            visible_ancestor_names: list[str] = []
            hidden_ancestor_names: list[str] = []
            component_prefixes: list[tuple[str, ...]] = []
            for index, current_id in enumerate(lineage):
                current_entity = entities_by_id[current_id]
                if not _entity_belongs_to_hierarchy(current_entity, hierarchy):
                    continue
                current_name = _entity_display_name(current_entity)
                if not _include_entity_namespace(
                    current_id, stats_bearing_entity_ids
                ):
                    hidden_ancestor_names.append(current_name)
                    continue
                prefix = lineage[: index + 1]
                display_name = _strip_visible_ancestor_name_prefix(
                    current_name, visible_ancestor_names
                )
                component_display_by_prefix.setdefault(prefix, display_name)
                component_entity_id_by_prefix[prefix] = current_id
                component_prefixes.append(prefix)
                visible_ancestor_names.append(current_name)
                visible_ancestor_names.extend(
                    hidden_name
                    for hidden_name in hidden_ancestor_names
                    if current_name.startswith(f"{hidden_name}/")
                )
            component_prefixes_by_lineage[lineage] = tuple(component_prefixes)

    attempts = dict.fromkeys(component_display_by_prefix, 0)
    while True:
        component_by_prefix = {
            prefix: (
                _escape_entity_namespace_component(
                    component_display_by_prefix[prefix]
                )
                if attempts[prefix] == 0
                else _entity_namespace_component_with_id(
                    component_display_by_prefix[prefix],
                    component_entity_id_by_prefix[prefix],
                    attempt=attempts[prefix],
                    preserve_hierarchy=len(prefix) > 1,
                )
            )
            for prefix in component_display_by_prefix
        }
        namespace_by_prefix = {
            terminal_prefix: "/".join(
                component_by_prefix[prefix]
                for prefix in component_prefixes_by_lineage[lineage]
                if len(prefix) <= len(terminal_prefix)
            )
            for lineage, component_prefixes in component_prefixes_by_lineage.items()
            for terminal_prefix in component_prefixes
        }
        normalized_to_prefixes: dict[str, list[tuple[str, ...]]] = {}
        for prefix, namespace in namespace_by_prefix.items():
            normalized_to_prefixes.setdefault(
                model_explorer_normalized_namespace(namespace), []
            ).append(prefix)
        conflict_groups = [
            sorted(set(prefixes))
            for normalized, prefixes in normalized_to_prefixes.items()
            if not normalized or len(set(prefixes)) > 1
        ]
        if not conflict_groups:
            break

        allocation_prefixes: set[tuple[str, ...]] = set()
        for conflict_group in conflict_groups:
            component_sequences = [
                component_prefixes_by_lineage[prefix]
                for prefix in conflict_group
            ]
            max_common = min(len(sequence) for sequence in component_sequences)
            allocation_index = max_common - 1
            for candidate_index in range(max_common - 1, -1, -1):
                if len(
                    {
                        sequence[candidate_index][-1]
                        for sequence in component_sequences
                    }
                ) > 1:
                    allocation_index = candidate_index
                    break
            for sequence in component_sequences:
                allocation_prefixes.add(
                    sequence[min(allocation_index, len(sequence) - 1)]
                )

        if all(attempts[prefix] >= 3 for prefix in allocation_prefixes):
            conflicts = {
                prefix: namespace_by_prefix.get(prefix, "")
                for prefix in sorted(allocation_prefixes)
            }
            raise RuntimeError(
                "Could not allocate globally unique complete Model Explorer "
                f"hierarchy paths after canonical entity-ID encoding: {conflicts!r}."
            )
        for prefix in allocation_prefixes:
            attempts[prefix] = min(attempts[prefix] + 1, 3)

    def lineage_namespace(lineage: tuple[str, ...]) -> str:
        prefixes = component_prefixes_by_lineage.get(lineage, ())
        if not prefixes:
            return ""
        return namespace_by_prefix[prefixes[-1]]

    entity_paths_by_id: dict[str, tuple[str, ...]] = {}
    source_operator_paths_by_id: dict[str, tuple[str, ...]] = {}
    for entity_id, lineages in lineages_by_entity_id.items():
        entity = entities_by_id[entity_id]
        paths = tuple(
            sorted(
                {
                    namespace
                    for lineage in lineages
                    for namespace in [lineage_namespace(lineage)]
                    if namespace
                }
            )
        )
        if _entity_kind(entity) == _SOURCE_OPERATOR_KIND:
            source_operator_paths_by_id[entity_id] = paths
        elif _include_entity_namespace(entity_id, stats_bearing_entity_ids):
            entity_paths_by_id[entity_id] = paths
    return _HierarchyPathProjection(
        entity_paths_by_id=entity_paths_by_id,
        source_operator_paths_by_id=source_operator_paths_by_id,
    )


def _entity_complete_lineages(
    entity_id: str,
    entities_by_id: dict[str, dict[str, Any]],
    hierarchy: _EntityKindHierarchy,
    seen: frozenset[str],
) -> tuple[tuple[str, ...], ...]:
    """Return every finite root-to-entity lineage using all valid parents."""
    if entity_id in seen:
        return ()
    entity = entities_by_id.get(entity_id)
    if entity is None:
        return ()
    next_seen = seen | {entity_id}
    parents = _hierarchy_parent_ids(entity_id, entities_by_id, hierarchy, next_seen)
    lineages = {
        (*parent_lineage, entity_id)
        for parent_id in parents
        for parent_lineage in _entity_complete_lineages(
            parent_id, entities_by_id, hierarchy, next_seen
        )
    }
    if lineages:
        return tuple(sorted(lineages))
    return ((entity_id,),)


def _hierarchy_parent_ids(
    entity_id: str,
    entities_by_id: dict[str, dict[str, Any]],
    hierarchy: _EntityKindHierarchy,
    seen: frozenset[str],
) -> tuple[str, ...]:
    """Return every declared parent whose kind is a valid hierarchy ancestor."""
    entity = entities_by_id[entity_id]
    parent_ids = _entity_parent_ids(entity)
    if not parent_ids:
        parent_ids = _inferred_parent_ids_from_child_ids(entity_id, entities_by_id)
    child_kind = _entity_kind(entity)
    return tuple(
        sorted(
            {
                parent_id
                for parent_id in parent_ids
                if parent_id not in seen
                and _is_valid_hierarchy_parent(
                    child_kind,
                    _entity_kind(entities_by_id.get(parent_id)),
                    hierarchy,
                )
            }
        )
    )


def _entity_belongs_to_hierarchy(entity: Any, hierarchy: _EntityKindHierarchy) -> bool:
    kind = _entity_kind(entity)
    return kind is not None and kind in hierarchy.backend_kinds


def _entity_parent_ids(entity: dict[str, Any]) -> list[str]:
    """Return parent ids from schema-1.2 parent_ids."""
    parent_ids = entity.get("parent_ids")
    if isinstance(parent_ids, list):
        return [item for item in parent_ids if isinstance(item, str) and item]
    return []


def _entity_child_ids(entity: dict[str, Any]) -> list[str]:
    """Return child ids from schema-1.2 child_ids."""
    child_ids = entity.get("child_ids")
    if isinstance(child_ids, list):
        return [item for item in child_ids if isinstance(item, str) and item]
    return []


def _inferred_parent_ids_from_child_ids(
    entity_id: str,
    entities_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        candidate_id
        for candidate_id, candidate in entities_by_id.items()
        if entity_id in _entity_child_ids(candidate)
    ]


def _is_valid_hierarchy_parent(
    child_kind: str | None,
    parent_kind: str | None,
    hierarchy: _EntityKindHierarchy,
) -> bool:
    """Return whether a parent kind can precede the child in this hierarchy."""
    if child_kind is None or parent_kind is None:
        return False
    try:
        child_index = hierarchy.path.index(child_kind)
        parent_index = hierarchy.path.index(parent_kind)
    except ValueError:
        return False
    if parent_index < child_index:
        return True
    return parent_kind == child_kind and child_kind in hierarchy.recursive_kinds


def _include_entity_namespace(
    entity_id: str,
    stats_bearing_entity_ids: set[str],
) -> bool:
    return (
        not _ONLY_SHOW_STATS_BEARING_ENTITY_NAMESPACES
        or entity_id in stats_bearing_entity_ids
    )
