# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cross-graph entity mappings and synchronized navigation."""

from __future__ import annotations

from typing import Any

from mlia.model_explorer.graph_data import model_explorer_namespace_prefixes

from .graphs import (
    _append_unique_items,
    _entities_by_id,
    _entity_children_by_parent,
    _entity_kind,
    _generated_hierarchy_graph_id,
    _graph_mapping_target_ids,
    _graph_projection,
    _graphs_by_id,
    _namespace_group_node_id,
    _projection_identity,
    _source_graph_node_ids,
)
from .models import (
    _SOURCE_OPERATOR_KIND,
    _GeneratedModelArtifact,
    _GraphProjection,
    _NodeRef,
)


def _entity_node_mappings(
    source_graph_collection: dict[str, Any] | None,
    output: dict[str, Any],
    generated_model_artifacts: list[_GeneratedModelArtifact],
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
    additional_entity_refs: (dict[tuple[int, str], tuple[_NodeRef, ...]] | None) = None,
) -> dict[str, Any]:
    """Return result-scoped pairs of visualizer nodes and canonical entities."""
    source_node_refs = (
        _source_graph_node_ids(source_graph_collection)
        if source_graph_collection is not None
        else set()
    )
    results = output.get("results", [])
    if not isinstance(results, list):
        results = []

    mapped_results: list[dict[str, Any]] = []
    additional_entity_refs = additional_entity_refs or {}
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        entities_by_id = _entities_by_id(result)
        mappings: list[tuple[_NodeRef, str]] = []
        for entity_id, entity in entities_by_id.items():
            if _entity_kind(entity) != _SOURCE_OPERATOR_KIND:
                continue
            mappings.extend(
                (node_ref, entity_id)
                for node_ref in source_operator_to_node_ids.get(entity_id, [])
                if node_ref in source_node_refs
            )

        for artifact in generated_model_artifacts:
            if artifact.descriptor.result_index != result_index:
                continue
            mappings.extend(
                (node_ref, source_operator_id)
                for node_ref, source_operator_id in (
                    artifact.projection.source_operator_by_generated_ref.items()
                )
                if source_operator_id in entities_by_id
            )
            mappings.extend(
                (node_ref, entity_id)
                for entity_id, node_refs in (
                    artifact.projection.group_refs_by_entity_id.items()
                )
                if entity_id in entities_by_id
                for node_ref in node_refs
            )

        mappings.extend(
            (node_ref, entity_id)
            for (mapping_result_index, entity_id), node_refs in (
                additional_entity_refs.items()
            )
            if mapping_result_index == result_index and entity_id in entities_by_id
            for node_ref in node_refs
        )

        mapped_results.append(
            {
                "resultIndex": result_index,
                "resultKind": str(result.get("kind") or "result"),
                "mappings": [
                    {"node": node_ref.to_json(), "entityId": entity_id}
                    for node_ref, entity_id in sorted(
                        set(mappings),
                        key=lambda item: (*_node_ref_sort_key(item[0]), item[1]),
                    )
                ],
            }
        )
    return {"results": mapped_results}


def _node_ref_sort_key(node_ref: _NodeRef) -> tuple[str, str, str]:
    return (node_ref.collection_label, node_ref.graph_id, node_ref.node_id)


def _sync_navigation_data(
    source_graph_collection: dict[str, Any] | None,
    generated_model_artifacts: list[_GeneratedModelArtifact],
    additional_entity_refs: (dict[tuple[int, str], tuple[_NodeRef, ...]] | None) = None,
    additional_graph_collections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return graph-qualified mappings between corresponding graph projections."""
    families = _graph_projection_families(
        source_graph_collection, generated_model_artifacts
    )
    all_projections: list[_GraphProjection] = []
    projection_identities: set[tuple[str, str]] = set()
    for family in families:
        for projection in family:
            identity = _projection_identity(projection)
            if identity in projection_identities:
                raise RuntimeError(
                    "Duplicate sync-navigation graph projection "
                    f"{projection.collection_label!r}/{projection.graph_id!r}."
                )
            projection_identities.add(identity)
            all_projections.append(projection)

    graph_mappings: list[dict[str, Any]] = []
    for family in families:
        for source_projection in family:
            for group in source_projection.groups:
                destinations: dict[str, dict[str, list[str]]] = {}
                for target_projection in all_projections:
                    if _projection_identity(target_projection) == _projection_identity(
                        source_projection
                    ):
                        continue
                    target_ids = (
                        _graph_mapping_target_ids(group, target_projection)
                        if target_projection.origin == source_projection.origin
                        else []
                    )
                    _add_sync_navigation_destination(
                        destinations, target_projection, target_ids
                    )
                if destinations:
                    graph_mappings.append(
                        {
                            "from": group.node_ref.to_json(),
                            "to": destinations,
                        }
                    )

            for canonical_ref, source_node_id in zip(
                source_projection.ordered_leaf_refs,
                source_projection.ordered_leaf_node_ids,
            ):
                destinations = {}
                for target_projection in all_projections:
                    if _projection_identity(target_projection) == _projection_identity(
                        source_projection
                    ):
                        continue
                    target_ids = (
                        list(
                            target_projection.leaf_ids_by_canonical_ref.get(
                                canonical_ref, ()
                            )
                        )
                        if target_projection.origin == source_projection.origin
                        else []
                    )
                    _add_sync_navigation_destination(
                        destinations, target_projection, target_ids
                    )
                if destinations:
                    graph_mappings.append(
                        {
                            "from": _NodeRef(
                                source_projection.collection_label,
                                source_projection.graph_id,
                                source_node_id,
                            ).to_json(),
                            "to": destinations,
                        }
                    )

    if not additional_entity_refs and not additional_graph_collections:
        return {"type": "sync_navigation", "graphMappings": graph_mappings}

    additional_refs_by_entity = {
        key: list(node_refs)
        for key, node_refs in (additional_entity_refs or {}).items()
    }
    generated_refs_by_entity: dict[tuple[int, str], list[_NodeRef]] = {}
    for artifact in generated_model_artifacts:
        result_index = artifact.descriptor.result_index
        for entity_id, node_refs in artifact.projection.group_refs_by_entity_id.items():
            _append_unique_items(
                generated_refs_by_entity.setdefault((result_index, entity_id), []),
                list(node_refs),
            )
        for key, canonical_refs in additional_refs_by_entity.items():
            if key[0] != result_index:
                continue
            for canonical_ref in canonical_refs:
                _append_unique_items(
                    generated_refs_by_entity.setdefault(key, []),
                    list(
                        artifact.projection.generated_refs_by_canonical_ref.get(
                            canonical_ref, ()
                        )
                    ),
                )
    for key, additional_refs in additional_refs_by_entity.items():
        generated_refs = generated_refs_by_entity.get(key, [])
        all_refs = [*additional_refs, *generated_refs]
        if len(all_refs) < 2:
            continue
        for source_ref in additional_refs:
            destinations = {}
            for target_ref in all_refs:
                if target_ref == source_ref:
                    continue
                graph_routes = destinations.setdefault(target_ref.collection_label, {})
                target_ids = graph_routes.setdefault(target_ref.graph_id, [])
                _append_unique_items(target_ids, [target_ref.node_id])
            graph_mappings.append({"from": source_ref.to_json(), "to": destinations})
        for source_ref in generated_refs:
            destinations = {}
            for target_ref in additional_refs:
                graph_routes = destinations.setdefault(target_ref.collection_label, {})
                target_ids = graph_routes.setdefault(target_ref.graph_id, [])
                _append_unique_items(target_ids, [target_ref.node_id])
            if destinations:
                graph_mappings.append(
                    {"from": source_ref.to_json(), "to": destinations}
                )
    graph_identities = set(projection_identities)
    graph_identities.update(
        (ref.collection_label, ref.graph_id)
        for refs in additional_refs_by_entity.values()
        for ref in refs
    )
    collections = list(additional_graph_collections or [])
    if source_graph_collection is not None:
        collections.append(source_graph_collection)
    return {
        "type": "sync_navigation",
        "graphMappings": _complete_sync_navigation_mappings(
            graph_mappings, graph_identities, collections
        ),
    }


def _complete_sync_navigation_mappings(
    graph_mappings: list[dict[str, Any]],
    graph_identities: set[tuple[str, str]],
    graph_collections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Declare empty routes to prevent unrelated graphs' same-ID fallback."""
    graph_identities = set(graph_identities)
    merged_mappings = _merge_sync_navigation_mappings(graph_mappings)
    mapped_sources = {
        _NodeRef(
            mapping["from"]["collectionLabel"],
            mapping["from"]["graphId"],
            mapping["from"]["nodeId"],
        )
        for mapping in merged_mappings
    }
    for collection in graph_collections:
        collection_label = str(collection["label"])
        for graph in collection.get("graphs", []):
            graph_id = str(graph["id"])
            graph_identities.add((collection_label, graph_id))
            for node in graph.get("nodes", []):
                node_ids = [node["id"]]
                node_ids.extend(
                    _namespace_group_node_id(prefix)
                    for prefix in model_explorer_namespace_prefixes(
                        node.get("namespace", "")
                    )
                )
                for node_id in node_ids:
                    source_ref = _NodeRef(collection_label, graph_id, node_id)
                    if source_ref not in mapped_sources:
                        mapped_sources.add(source_ref)
                        merged_mappings.append({"from": source_ref.to_json(), "to": {}})
    ordered_graph_identities = sorted(graph_identities)
    for mapping in merged_mappings:
        source = mapping["from"]
        for collection_label, graph_id in ordered_graph_identities:
            if (collection_label, graph_id) != (
                source["collectionLabel"],
                source["graphId"],
            ):
                mapping["to"].setdefault(collection_label, {}).setdefault(graph_id, [])
    return merged_mappings


def _add_sync_navigation_destination(
    destinations: dict[str, dict[str, list[str]]],
    target_projection: _GraphProjection,
    target_ids: list[str],
) -> None:
    """Add one graph-qualified target route, rejecting duplicate projections."""
    collection_routes = destinations.setdefault(target_projection.collection_label, {})
    if target_projection.graph_id in collection_routes:
        raise RuntimeError(
            "Duplicate sync-navigation target projection "
            f"{target_projection.collection_label!r}/"
            f"{target_projection.graph_id!r}."
        )
    collection_routes[target_projection.graph_id] = target_ids


def _graph_projection_families(
    source_graph_collection: dict[str, Any] | None,
    generated_model_artifacts: list[_GeneratedModelArtifact],
) -> list[list[_GraphProjection]]:
    """Group source and generated projections by originating source graph."""
    if source_graph_collection is None:
        return []

    source_collection_label = source_graph_collection.get("label")
    if not isinstance(source_collection_label, str) or not source_collection_label:
        raise RuntimeError("Source graph collection requires a public label.")

    source_graphs = _graphs_by_id(source_graph_collection, "source")
    source_node_refs = _source_graph_node_ids(source_graph_collection)
    generated_graphs_by_artifact = []
    for artifact in generated_model_artifacts:
        _validate_generated_hierarchy_projection(artifact, source_node_refs)
        generated_graphs_by_artifact.append(
            (
                artifact,
                _graphs_by_id(
                    artifact.graph_collection,
                    f"generated collection {artifact.graph_collection.get('label')!r}",
                ),
            )
        )

    families: list[list[_GraphProjection]] = []
    for source_graph_id, source_graph in source_graphs.items():
        origin = (source_collection_label, source_graph_id)
        projections = [
            _graph_projection(
                origin,
                source_collection_label,
                source_graph,
            )
        ]
        for artifact, generated_graphs in generated_graphs_by_artifact:
            generated_collection_label = artifact.graph_collection.get("label")
            if (
                not isinstance(generated_collection_label, str)
                or not generated_collection_label
            ):
                raise RuntimeError(
                    "Generated graph collection requires a public label."
                )
            generated_graph_id = _generated_hierarchy_graph_id(
                artifact.descriptor.graph_name, source_graph_id
            )
            generated_graph = generated_graphs.get(generated_graph_id)
            if generated_graph is None:
                raise RuntimeError(
                    "Generated hierarchy collection is missing the projection for "
                    f"source graph {source_graph_id!r}: expected graph "
                    f"{generated_graph_id!r} in collection "
                    f"{generated_collection_label!r}."
                )
            projections.append(
                _graph_projection(
                    origin,
                    generated_collection_label,
                    generated_graph,
                    artifact.projection.canonical_ref_by_generated_ref,
                    artifact.projection.terminal_complete_path_group_refs,
                )
            )
        families.append(projections)
    return families


def _validate_generated_hierarchy_projection(
    artifact: _GeneratedModelArtifact,
    source_node_refs: set[_NodeRef],
) -> None:
    """Fail closed unless generated serialization and explicit mappings agree."""
    collection_label = artifact.graph_collection.get("label")
    if not isinstance(collection_label, str) or not collection_label:
        raise RuntimeError("Generated graph collection requires a public label.")

    serialized_refs: set[_NodeRef] = set()
    runtime_group_refs: set[_NodeRef] = set()
    for graph_id, graph in _graphs_by_id(
        artifact.graph_collection,
        f"generated collection {collection_label!r}",
    ).items():
        graph_node_ids: set[str] = set()
        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id:
                continue
            if node_id in graph_node_ids:
                raise RuntimeError(
                    f"Duplicate node ID {node_id!r} in generated graph "
                    f"{collection_label!r}/{graph_id!r}."
                )
            graph_node_ids.add(node_id)
            serialized_refs.add(_NodeRef(collection_label, graph_id, node_id))
            namespace = node.get("namespace")
            if isinstance(namespace, str) and namespace:
                runtime_group_refs.update(
                    _NodeRef(
                        collection_label,
                        graph_id,
                        _namespace_group_node_id(prefix),
                    )
                    for prefix in model_explorer_namespace_prefixes(namespace)
                )

    reverse_mapping = artifact.projection.canonical_ref_by_generated_ref
    mapped_refs = set(reverse_mapping)
    missing_refs = sorted(serialized_refs - mapped_refs, key=str)
    unexpected_refs = sorted(mapped_refs - serialized_refs, key=str)
    if missing_refs or unexpected_refs:
        raise RuntimeError(
            "Generated graph canonical mapping must cover every serialized node "
            "exactly and contain no unexpected nodes; "
            f"missing={missing_refs!r}, unexpected={unexpected_refs!r} in "
            f"collection {collection_label!r}."
        )

    invalid_canonical_refs = sorted(
        {
            canonical_ref
            for canonical_ref in reverse_mapping.values()
            if canonical_ref not in source_node_refs
        },
        key=str,
    )
    if invalid_canonical_refs:
        raise RuntimeError(
            "Generated graph canonical mapping references nodes outside the source "
            f"graph collection: {invalid_canonical_refs!r}."
        )

    forward_mapping: dict[_NodeRef, _NodeRef] = {}
    duplicate_forward_refs: set[_NodeRef] = set()
    for (
        canonical_ref,
        generated_refs,
    ) in artifact.projection.generated_refs_by_canonical_ref.items():
        for generated_ref in generated_refs:
            if generated_ref in forward_mapping:
                duplicate_forward_refs.add(generated_ref)
            forward_mapping[generated_ref] = canonical_ref
    if duplicate_forward_refs:
        raise RuntimeError(
            "Generated graph canonical mapping contains duplicate generated-node "
            f"references: {sorted(duplicate_forward_refs, key=str)!r}."
        )
    if set(forward_mapping) != serialized_refs:
        raise RuntimeError(
            "Generated graph forward canonical mapping does not exactly cover the "
            f"serialized nodes in collection {collection_label!r}."
        )
    inconsistent_refs = sorted(
        (
            generated_ref
            for generated_ref, canonical_ref in forward_mapping.items()
            if reverse_mapping.get(generated_ref) != canonical_ref
        ),
        key=str,
    )
    if inconsistent_refs:
        raise RuntimeError(
            "Generated graph forward and reverse canonical mappings disagree for "
            f"nodes {inconsistent_refs!r}."
        )

    unexpected_source_operator_refs = sorted(
        set(artifact.projection.source_operator_by_generated_ref) - serialized_refs,
        key=str,
    )
    if unexpected_source_operator_refs:
        raise RuntimeError(
            "Generated source-operator mapping contains unexpected nodes: "
            f"{unexpected_source_operator_refs!r}."
        )

    declared_group_refs = {
        node_ref
        for node_refs in artifact.projection.group_refs_by_entity_id.values()
        for node_ref in node_refs
    }
    invalid_group_refs = sorted(
        (
            declared_group_refs
            | set(artifact.projection.terminal_complete_path_group_refs)
        )
        - runtime_group_refs,
        key=str,
    )
    if invalid_group_refs:
        raise RuntimeError(
            "Generated hierarchy projection references namespace groups that are not "
            f"created by the serialized graph: {invalid_group_refs!r}."
        )


def _merge_sync_navigation_mappings(
    graph_mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge routes with the same source node into one navigation record."""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for mapping in graph_mappings:
        source = mapping["from"]
        key = (
            source["collectionLabel"],
            source["graphId"],
            source["nodeId"],
        )
        target = merged.setdefault(
            key,
            {"from": source, "to": {}},
        )["to"]
        for collection_label, graph_routes in mapping["to"].items():
            target_graph_routes = target.setdefault(collection_label, {})
            for graph_id, node_ids in graph_routes.items():
                target_node_ids = target_graph_routes.setdefault(graph_id, [])
                _append_unique_items(target_node_ids, node_ids)
    return list(merged.values())


def _merge_entity_ref_maps(
    *mappings: dict[tuple[int, str], tuple[_NodeRef, ...]],
) -> dict[tuple[int, str], tuple[_NodeRef, ...]]:
    """Merge result-scoped entity references while preserving their order."""
    merged: dict[tuple[int, str], list[_NodeRef]] = {}
    for mapping in mappings:
        for key, node_refs in mapping.items():
            _append_unique_items(merged.setdefault(key, []), list(node_refs))
    return {key: tuple(node_refs) for key, node_refs in merged.items()}


def _namespace_entity_refs_as_group_nodes(
    refs_by_entity: dict[tuple[int, str], tuple[_NodeRef, ...]],
) -> dict[tuple[int, str], tuple[_NodeRef, ...]]:
    """Convert namespace overlay keys to artificial group-node references."""
    return {
        key: tuple(
            _NodeRef(
                node_ref.collection_label,
                node_ref.graph_id,
                f"{node_ref.node_id}___group___",
            )
            for node_ref in node_refs
        )
        for key, node_refs in refs_by_entity.items()
    }


def _related_entity_source_refs(
    output: dict[str, Any],
    source_operator_refs: dict[str, list[_NodeRef]],
    entity_kind: str,
) -> dict[tuple[int, str], tuple[_NodeRef, ...]]:
    """Project explicit relationships for a requested entity kind into source refs."""
    refs: dict[tuple[int, str], tuple[_NodeRef, ...]] = {}
    for result_index, result in enumerate(output.get("results", []) or []):
        if not isinstance(result, dict):
            continue
        entities = _entities_by_id(result)
        children_by_parent = _entity_children_by_parent(entities)
        for entity_id, entity in entities.items():
            if _entity_kind(entity) != entity_kind:
                continue
            related_refs: list[_NodeRef] = []
            for source_id in children_by_parent.get(entity_id, []):
                if (
                    source_id in entities
                    and _entity_kind(entities[source_id]) == _SOURCE_OPERATOR_KIND
                ):
                    _append_unique_items(
                        related_refs, source_operator_refs.get(source_id, [])
                    )
            if related_refs:
                refs[(result_index, entity_id)] = tuple(related_refs)
    return refs
