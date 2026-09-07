# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model Explorer node-data overlays and presentation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mlia.core import output_schema as schema
from mlia.core.output_projection import project_entity_breakdowns

from mlia.model_explorer.graph_data import model_explorer_namespace_prefixes

from .graphs import (
    _append_unique_items,
    _entities_by_id,
    _entity_id,
    _entity_kind,
    _namespace_group_node_id,
    _result_entity_node_ids,
)
from .models import (
    _MISSING_QUALIFIERS,
    _MODEL_EXPLORER_NAMESPACE_KIND,
    _MODEL_KIND,
    _SOURCE_OPERATOR_KIND,
    _CanonicalJsonValue,
    _CanonicalQualifiers,
    _GeneratedModelArtifact,
    _LogicalOverlay,
    _NodeAnnotation,
    _NodeDataArtifact,
    _NodeRef,
    _OverlayKey,
)
from .serialization import (
    _NODE_DATA_FILENAME_EXTENSION,
    _allocate_safe_filename,
    _bounded_text_with_digest,
    _write_json,
)


def _write_node_data_files(
    output: dict[str, Any],
    node_data_dir: Path,
    directory_names_by_collection: dict[str, str],
    generated_artifacts_by_result: dict[int, list[_GeneratedModelArtifact]],
    source_graph_collection: dict[str, Any] | None,
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
) -> list[_NodeDataArtifact]:
    results = output.get("results", [])
    if not isinstance(results, list):
        results = []

    artifacts: list[_NodeDataArtifact] = []
    collection_directories: dict[str, Path] = {}
    allocated_filenames_by_collection: dict[str, set[str]] = {}
    logical_overlays = _node_data_overlays(
        results,
        generated_artifacts_by_result,
        source_graph_collection,
        source_operator_to_node_ids,
    )
    for overlay in logical_overlays:
        graph_data_by_collection = _collection_scoped_node_data(
            overlay.display_name, overlay.node_data
        )
        for collection_label, graph_data in graph_data_by_collection.items():
            if collection_label not in collection_directories:
                collection_directory = (
                    node_data_dir / directory_names_by_collection[collection_label]
                )
                collection_directory.mkdir(parents=True, exist_ok=True)
                collection_directories[collection_label] = collection_directory
            filename = _allocate_safe_filename(
                overlay.display_name,
                allocated_filenames_by_collection.setdefault(collection_label, set()),
            )
            path = collection_directories[collection_label] / (
                f"{filename}{_NODE_DATA_FILENAME_EXTENSION}"
            )
            _write_json(path, graph_data)
            artifacts.append(
                _NodeDataArtifact(
                    path,
                    graph_collection_label=collection_label,
                    overlay_key=overlay.key,
                    display_name=overlay.display_name,
                )
            )

    return artifacts


def _collection_scoped_node_data(
    name: str,
    graph_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Group exact logical node references by collection and graph."""
    results: dict[_NodeRef, dict[str, Any]] = graph_data["results"]
    graphs_data_by_collection: dict[str, dict[str, dict[str, Any]]] = {}
    for node_ref, value in results.items():
        graphs_data = graphs_data_by_collection.setdefault(
            node_ref.collection_label, {}
        )
        graph_data_for_id = graphs_data.setdefault(
            node_ref.graph_id,
            {**graph_data, "name": name, "results": {}},
        )
        graph_data_for_id["results"][node_ref.node_id] = value
    return graphs_data_by_collection


def _source_namespace_projection(
    result: dict[str, Any],
    source_graph_collection: dict[str, Any] | None,
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[_NodeRef]],
]:
    """Project source-operator metrics onto faithful source namespace entities."""
    if source_graph_collection is None:
        return [], {}, {}

    entities_by_id = _entities_by_id(result)
    source_entity_ids = {
        entity_id
        for entity_id, entity in entities_by_id.items()
        if _entity_kind(entity) == _SOURCE_OPERATOR_KIND
    }
    if not source_entity_ids:
        return [], {}, {}

    source_entity_ids_by_node_ref: dict[_NodeRef, list[str]] = {}
    for entity_id in sorted(source_entity_ids):
        for node_ref in source_operator_to_node_ids.get(entity_id, []):
            source_entity_ids_by_node_ref.setdefault(node_ref, []).append(entity_id)

    collection_label = source_graph_collection.get("label")
    if not isinstance(collection_label, str) or not collection_label:
        return [], {}, {}

    pseudo_entities_by_id: dict[str, dict[str, Any]] = {}
    pseudo_refs_by_entity_id: dict[str, list[_NodeRef]] = {}
    pseudo_id_by_graph_namespace: dict[tuple[str, str], str] = {}

    def pseudo_id(graph_id: str, namespace: str) -> str:
        key = (graph_id, namespace)
        existing = pseudo_id_by_graph_namespace.get(key)
        if existing is not None:
            return existing
        entity_id = "/".join(
            (
                _MODEL_EXPLORER_NAMESPACE_KIND,
                quote(collection_label, safe="-._~"),
                quote(graph_id, safe="-._~"),
                quote(namespace, safe="-._~"),
            )
        )
        pseudo_id_by_graph_namespace[key] = entity_id
        pseudo_entities_by_id[entity_id] = {
            "id": entity_id,
            "kind": _MODEL_EXPLORER_NAMESPACE_KIND,
            "name": namespace,
            "child_ids": [],
        }
        pseudo_refs_by_entity_id[entity_id] = [
            _NodeRef(
                collection_label,
                graph_id,
                _namespace_group_node_id(namespace),
            )
        ]
        return entity_id

    for graph in source_graph_collection.get("graphs", []) or []:
        if not isinstance(graph, dict):
            continue
        graph_id = graph.get("id")
        if not isinstance(graph_id, str) or not graph_id:
            continue
        for node in graph.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            namespace = node.get("namespace")
            if not isinstance(node_id, str) or not isinstance(namespace, str):
                continue
            prefixes = model_explorer_namespace_prefixes(namespace)
            if not prefixes:
                continue
            node_ref = _NodeRef(collection_label, graph_id, node_id)
            mapped_entity_ids = source_entity_ids_by_node_ref.get(node_ref, [])
            if not mapped_entity_ids:
                continue

            prefix_ids = [pseudo_id(graph_id, prefix) for prefix in prefixes]
            for parent_id, child_id in zip(prefix_ids, prefix_ids[1:]):
                parent_children = pseudo_entities_by_id[parent_id]["child_ids"]
                if child_id not in parent_children:
                    parent_children.append(child_id)
            deepest_children = pseudo_entities_by_id[prefix_ids[-1]]["child_ids"]
            for mapped_entity_id in mapped_entity_ids:
                if mapped_entity_id not in deepest_children:
                    deepest_children.append(mapped_entity_id)

    if not pseudo_entities_by_id:
        return [], {}, {}

    projection_entities = [
        {
            **entity,
            "name": str(entity.get("name") or entity_id),
        }
        for entity_id, entity in sorted(entities_by_id.items())
    ]
    authoritative_breakdowns = [
        {
            **breakdown,
            "metrics": [
                {**metric, "unit": str(metric.get("unit") or "")}
                for metric in (breakdown.get("metrics", []) or [])
                if isinstance(metric, dict)
            ],
        }
        for breakdown in (result.get("breakdowns", []) or [])
        if isinstance(breakdown, dict)
    ]
    entity_kinds = [
        entity_kind
        for entity_kind in (result.get("entity_kinds", []) or [])
        if isinstance(entity_kind, dict)
    ]
    projection_input = schema.Result.from_dict(
        {
            "kind": "performance",
            "status": "ok",
            "producer": "mlia-model-explorer",
            "entity_kinds": [
                *entity_kinds,
                {
                    "id": _MODEL_EXPLORER_NAMESPACE_KIND,
                    "parent_kinds": [_MODEL_EXPLORER_NAMESPACE_KIND],
                    "child_kinds": [
                        _MODEL_EXPLORER_NAMESPACE_KIND,
                        _SOURCE_OPERATOR_KIND,
                    ],
                },
            ],
            "entities": [*projection_entities, *pseudo_entities_by_id.values()],
            "breakdowns": authoritative_breakdowns,
        }
    )
    projected = project_entity_breakdowns(projection_input)
    projected_breakdowns = [
        breakdown.to_dict()
        for breakdown in projected.breakdowns
        if breakdown.entity_id in pseudo_entities_by_id
    ]
    return projected_breakdowns, pseudo_entities_by_id, pseudo_refs_by_entity_id


def _node_data_overlays(
    results: list[Any],
    generated_artifacts_by_result: dict[int, list[_GeneratedModelArtifact]],
    source_graph_collection: dict[str, Any] | None,
    source_operator_to_node_ids: dict[str, list[_NodeRef]],
) -> list[_LogicalOverlay]:
    overlays: dict[_OverlayKey, dict[_NodeRef, dict[str, Any]]] = {}
    text_overlays: dict[_OverlayKey, dict[_NodeRef, _NodeAnnotation]] = {}
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        result_kind = str(result.get("kind") or "result")
        # Model Explorer node data is attached to source graph nodes. Qualitative
        # facts such as placement, status, checks, and advice can sensibly be
        # repeated on each source node associated with a higher-level entity: the
        # repeated value means "this node participates in a group with this
        # property". Quantitative metric breakdowns are different: repeating a
        # model or group total on each associated node makes it look like a
        # per-node value and can cause misleading duplication. For that reason,
        # entity/check/advice overlays accept source operators and declared
        # backend hierarchy kinds. Metric overlays use canonical source nodes and
        # unsplit hierarchy clones, but not synthetic multi-path sub-operators.
        # Projected source namespaces and backend hierarchy entities remain
        # attached to exact namespace group rows.
        entities_by_id = _entities_by_id(result)
        source_node_ids_by_entity = _result_entity_node_ids(
            result, source_operator_to_node_ids
        )
        generated_artifacts = generated_artifacts_by_result.get(result_index, [])
        overlay_node_ids_by_entity = _result_overlay_node_ids(
            generated_artifacts, source_node_ids_by_entity
        )
        metric_node_ids_by_entity = _result_metric_node_ids(
            generated_artifacts,
            source_node_ids_by_entity,
            entities_by_id,
        )
        entities = [
            entity
            for entity in entities_by_id.values()
            if _is_node_overlay_entity(entity)
        ]
        namespace_refs_by_entity_id = _result_entity_namespace_refs(
            generated_artifacts
        )
        (
            source_namespace_breakdowns,
            source_namespace_entities,
            source_namespace_refs,
        ) = _source_namespace_projection(
            result,
            source_graph_collection,
            source_operator_to_node_ids,
        )
        metric_entities_by_id = {**entities_by_id, **source_namespace_entities}
        metric_namespace_refs_by_entity_id = {
            **namespace_refs_by_entity_id,
            **source_namespace_refs,
        }

        _add_entity_overlays(
            overlays,
            result_index,
            result_kind,
            entities,
            overlay_node_ids_by_entity,
        )
        _add_breakdown_metric_overlays(
            overlays,
            result_index,
            result_kind,
            [
                *(result.get("breakdowns", []) or []),
                *source_namespace_breakdowns,
            ],
            metric_entities_by_id,
            metric_namespace_refs_by_entity_id,
            metric_node_ids_by_entity,
        )
        _add_check_overlay(
            text_overlays,
            result_index,
            result_kind,
            result.get("checks", []) or [],
            entities_by_id,
            overlay_node_ids_by_entity,
        )
        _add_advice_overlay(
            text_overlays,
            result_index,
            result_kind,
            result.get("advice", []) or [],
            entities_by_id,
            overlay_node_ids_by_entity,
        )

    for overlay_key, entries in text_overlays.items():
        if entries:
            overlays[overlay_key] = {
                key: value.to_node_data_result()
                for key, value in sorted(entries.items(), key=lambda item: str(item[0]))
            }

    ordered_keys = sorted(
        (overlay_key for overlay_key, entries in overlays.items() if entries),
        key=_overlay_sort_key,
    )
    display_names = _overlay_display_names(ordered_keys)
    return [
        _LogicalOverlay(
            overlay_key,
            display_names[overlay_key],
            _graph_node_data(overlay_key, overlays[overlay_key]),
        )
        for overlay_key in ordered_keys
    ]


def _result_overlay_node_ids(
    generated_artifacts: list[_GeneratedModelArtifact],
    source_node_ids_by_entity: dict[str, list[_NodeRef]],
) -> dict[str, list[_NodeRef]]:
    """Return exact source, sub-operator, and generated node refs by entity."""
    overlay_node_ids_by_entity: dict[str, list[_NodeRef]] = {}
    for entity_id, source_node_refs in source_node_ids_by_entity.items():
        overlay_node_refs = list(source_node_refs)
        for artifact in generated_artifacts:
            for source_node_ref in source_node_refs:
                _append_unique_items(
                    overlay_node_refs,
                    list(
                        artifact.projection.generated_refs_by_canonical_ref.get(
                            source_node_ref, ()
                        )
                    ),
                )
        overlay_node_ids_by_entity[entity_id] = overlay_node_refs
    return overlay_node_ids_by_entity


def _result_metric_node_ids(
    generated_artifacts: list[_GeneratedModelArtifact],
    source_node_ids_by_entity: dict[str, list[_NodeRef]],
    entities_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[_NodeRef]]:
    """Return source refs and only unsplit generated clones for metric overlays."""
    metric_node_ids_by_entity = {
        entity_id: list(source_node_refs)
        for entity_id, source_node_refs in source_node_ids_by_entity.items()
    }
    for entity_id, entity in entities_by_id.items():
        if _entity_kind(entity) != _SOURCE_OPERATOR_KIND:
            continue
        metric_node_refs = metric_node_ids_by_entity.setdefault(entity_id, [])
        for artifact in generated_artifacts:
            for source_node_ref in source_node_ids_by_entity.get(entity_id, []):
                generated_refs = (
                    artifact.projection.generated_refs_by_canonical_ref.get(
                        source_node_ref, ()
                    )
                )
                if len(generated_refs) == 1:
                    _append_unique_items(metric_node_refs, list(generated_refs))
    return metric_node_ids_by_entity


def _result_entity_namespace_refs(
    generated_artifacts: list[_GeneratedModelArtifact],
) -> dict[str, list[_NodeRef]]:
    """Return only actual runtime hierarchy group references by entity."""
    refs_by_entity_id: dict[str, list[_NodeRef]] = {}
    for artifact in generated_artifacts:
        for entity_id, node_refs in artifact.projection.group_refs_by_entity_id.items():
            _append_unique_items(
                refs_by_entity_id.setdefault(entity_id, []),
                [
                    _NodeRef(
                        node_ref.collection_label,
                        node_ref.graph_id,
                        node_ref.node_id.removesuffix("___group___"),
                    )
                    for node_ref in node_refs
                ],
            )
    return refs_by_entity_id


def _is_node_overlay_entity(item: Any) -> bool:
    kind = _entity_kind(item)
    return kind is not None and kind != _MODEL_KIND


def _add_entity_overlays(
    overlays: dict[_OverlayKey, dict[_NodeRef, dict[str, Any]]],
    result_index: int,
    result_kind: str,
    entities: list[dict[str, Any]],
    entity_node_ids: dict[str, list[_NodeRef]],
) -> None:
    placement_key = _OverlayKey(result_index, result_kind, "placement", "placement")
    status_key = _OverlayKey(result_index, result_kind, "status", "status")
    for entity in entities:
        placement = entity.get("placement")
        if placement is not None:
            value = (
                str(placement).upper()
                if str(placement).lower() in _PLACEMENT_COLORS
                else placement
            )
            _add_raw_node_data_result(
                overlays.setdefault(placement_key, {}),
                entity_node_ids.get(_entity_id(entity), []),
                value,
                bg_color=_placement_color(str(placement)),
            )

        status = entity.get("status")
        if status is not None:
            _add_raw_node_data_result(
                overlays.setdefault(status_key, {}),
                entity_node_ids.get(_entity_id(entity), []),
                status,
            )


def _add_breakdown_metric_overlays(
    overlays: dict[_OverlayKey, dict[_NodeRef, dict[str, Any]]],
    result_index: int,
    result_kind: str,
    breakdowns: list[Any],
    entities_by_id: dict[str, dict[str, Any]],
    namespace_refs_by_entity_id: dict[str, list[_NodeRef]],
    entity_node_ids: dict[str, list[_NodeRef]],
) -> None:
    for breakdown in breakdowns:
        if not isinstance(breakdown, dict):
            continue
        entity_id = breakdown.get("entity_id")
        entity = entities_by_id.get(str(entity_id)) if entity_id is not None else None
        if not isinstance(entity, dict) or not _is_node_overlay_entity(entity):
            continue
        breakdown_qualifiers = _canonical_qualifiers(
            breakdown.get("qualifiers", _MISSING_QUALIFIERS)
        )
        if breakdown_qualifiers is None:
            # A partially retained qualifier object could merge this whole breakdown
            # with a valid identity, so malformed breakdown qualifiers reject it.
            continue
        for metric in breakdown.get("metrics", []) or []:
            if not isinstance(metric, dict) or "value" not in metric:
                continue
            metric_qualifiers = _canonical_qualifiers(
                metric.get("qualifiers", _MISSING_QUALIFIERS)
            )
            if metric_qualifiers is None:
                # Reject only the affected metric; valid siblings remain useful.
                continue
            metric_name = str(metric.get("name") or "metric")
            unit = str(metric.get("unit") or "")
            overlay_key = _OverlayKey(
                result_index,
                result_kind,
                "metric",
                metric_name,
                unit,
                breakdown_qualifiers=breakdown_qualifiers,
                metric_qualifiers=metric_qualifiers,
            )
            _add_entity_node_data_result(
                overlays.setdefault(overlay_key, {}),
                entity,
                metric.get("value"),
                namespace_refs_by_entity_id=namespace_refs_by_entity_id,
                node_ids=entity_node_ids.get(_entity_id(entity), []),
            )


def _add_check_overlay(
    overlays: dict[_OverlayKey, dict[_NodeRef, _NodeAnnotation]],
    result_index: int,
    result_kind: str,
    checks: list[Any],
    entities_by_id: dict[str, dict[str, Any]],
    entity_node_ids: dict[str, list[_NodeRef]],
) -> None:
    overlay_key = _OverlayKey(result_index, result_kind, "checks", "checks")
    entries = overlays.setdefault(overlay_key, {})
    for check in checks:
        if not isinstance(check, dict):
            continue
        messages = _check_reason_messages(check)
        if not messages:
            continue
        entity_id = check.get("entity_id")
        entity = entities_by_id.get(str(entity_id)) if entity_id is not None else None
        if not isinstance(entity, dict) or not _is_node_overlay_entity(entity):
            continue
        value = "; ".join(messages)
        bg_color = _check_status_color(str(check.get("status") or ""))
        _add_text_node_data_result(
            entries,
            entity_node_ids.get(_entity_id(entity), []),
            value,
            bg_color=bg_color,
        )


def _add_advice_overlay(
    overlays: dict[_OverlayKey, dict[_NodeRef, _NodeAnnotation]],
    result_index: int,
    result_kind: str,
    advice_items: list[Any],
    entities_by_id: dict[str, dict[str, Any]],
    entity_node_ids: dict[str, list[_NodeRef]],
) -> None:
    overlay_key = _OverlayKey(result_index, result_kind, "advice", "advice")
    entries = overlays.setdefault(overlay_key, {})
    for advice in advice_items:
        if not isinstance(advice, dict):
            continue
        message = advice.get("message")
        if not message:
            continue
        bg_color = _severity_color(str(advice.get("severity") or "info"))
        for entity_id in advice.get("affected_entity_ids", []) or []:
            entity = entities_by_id.get(str(entity_id))
            if isinstance(entity, dict) and _is_node_overlay_entity(entity):
                _add_text_node_data_result(
                    entries,
                    entity_node_ids.get(_entity_id(entity), []),
                    message,
                    bg_color=bg_color,
                )


def _check_reason_messages(check: dict[str, Any]) -> list[str]:
    details = check.get("details")
    if not isinstance(details, dict):
        return []
    reasons = details.get("reasons", []) or []
    messages: list[str] = []
    for reason in reasons:
        if not isinstance(reason, dict):
            continue
        message = reason.get("description") or reason.get("detail")
        if message:
            message_str = str(message)
            if message_str not in messages:
                messages.append(message_str)
    return messages


def _graph_node_data(
    overlay_key: _OverlayKey,
    entries: dict[_NodeRef, dict[str, Any]],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "results": {
            key: value
            for key, value in sorted(entries.items(), key=lambda item: str(item[0]))
        },
        "showExpandedSummaryOnGroupNode": True,
        "showLabelCountColumnsInChildrenStatsTable": True,
    }
    if overlay_key.category == "metric":
        data["gradient"] = [
            {"stop": 0, "bgColor": "#2e7d32"},
            {"stop": 1, "bgColor": "#c62828"},
        ]
    return data


def _add_text_node_data_result(
    entries: dict[_NodeRef, _NodeAnnotation],
    node_ids: list[_NodeRef],
    value: object,
    *,
    bg_color: str | None = None,
    text_color: str | None = None,
) -> None:
    for node_ref in node_ids:
        entries.setdefault(node_ref, _NodeAnnotation()).add(
            value,
            bg_color=bg_color,
            text_color=text_color,
        )


def _add_raw_node_data_result(
    entries: dict[_NodeRef, dict[str, Any]],
    node_ids: list[_NodeRef],
    value: object,
    *,
    bg_color: str | None = None,
    text_color: str | None = None,
) -> None:
    for node_ref in node_ids:
        _add_raw_node_data_value(
            entries,
            node_ref,
            value,
            bg_color=bg_color,
            text_color=text_color,
        )


def _add_entity_node_data_result(
    entries: dict[_NodeRef, dict[str, Any]],
    entity: dict[str, Any],
    value: object,
    *,
    namespace_refs_by_entity_id: dict[str, list[_NodeRef]],
    node_ids: list[_NodeRef],
) -> None:
    entity_id = entity.get("id")
    namespace_refs = (
        namespace_refs_by_entity_id.get(entity_id, [])
        if isinstance(entity_id, str)
        else []
    )
    if namespace_refs:
        for namespace_ref in namespace_refs:
            _add_raw_node_data_value(entries, namespace_ref, value)
        return
    _add_raw_node_data_result(entries, node_ids, value)


def _add_raw_node_data_value(
    entries: dict[_NodeRef, dict[str, Any]],
    key: _NodeRef,
    value: object,
    *,
    bg_color: str | None = None,
    text_color: str | None = None,
) -> None:
    if key in entries:
        result = entries[key]
        result["value"] = _merge_raw_node_data_values(result["value"], value)
    else:
        result = {"value": value}
        entries[key] = result
    if bg_color and not result.get("bgColor"):
        result["bgColor"] = bg_color
    if text_color and not result.get("textColor"):
        result["textColor"] = text_color


def _merge_raw_node_data_values(current: object, new: object) -> object:
    if isinstance(current, bool) or isinstance(new, bool):
        return _merge_text_values(current, new)
    if isinstance(current, int | float) and isinstance(new, int | float):
        return current + new
    return _merge_text_values(current, new)


def _merge_text_values(current: object, new: object) -> str:
    current_str = _stringify_value(current)
    new_str = _stringify_value(new)
    if not current_str:
        return new_str
    if not new_str or new_str in current_str.split(" — "):
        return current_str
    return f"{current_str} — {new_str}"


def _canonical_qualifiers(value: object) -> _CanonicalQualifiers | None:
    """Return canonical qualifiers, or ``None`` when the whole object is invalid."""
    if value is _MISSING_QUALIFIERS:
        return ()
    if not isinstance(value, dict):
        return None

    qualifiers: list[tuple[str, _CanonicalJsonValue]] = []
    try:
        for key, qualifier_value in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            qualifiers.append((key, _canonical_json_value(qualifier_value)))
    except (RecursionError, TypeError, ValueError):
        # Never retain a valid subset: that could collapse malformed input onto a
        # different valid overlay identity and merge their metric values.
        return None
    return tuple(sorted(qualifiers))


def _canonical_json_value(value: object) -> _CanonicalJsonValue:
    """Return a sortable tagged representation of one finite, acyclic JSON value."""
    return _canonical_json_value_inner(value, set())


def _canonical_json_value_inner(
    value: object, active_container_ids: set[int]
) -> _CanonicalJsonValue:
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("number", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return ("number", json.dumps(value, allow_nan=False))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        container_id = _enter_canonical_json_container(value, active_container_ids)
        try:
            return (
                "array",
                tuple(
                    _canonical_json_value_inner(item, active_container_ids)
                    for item in value
                ),
            )
        finally:
            active_container_ids.remove(container_id)
    if isinstance(value, dict):
        container_id = _enter_canonical_json_container(value, active_container_ids)
        try:
            items: list[tuple[str, _CanonicalJsonValue]] = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                items.append(
                    (key, _canonical_json_value_inner(item, active_container_ids))
                )
            return ("object", tuple(sorted(items)))
        finally:
            active_container_ids.remove(container_id)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _enter_canonical_json_container(
    value: object, active_container_ids: set[int]
) -> int:
    container_id = id(value)
    if container_id in active_container_ids:
        raise ValueError("JSON values must not contain cycles")
    active_container_ids.add(container_id)
    return container_id


def _overlay_display_names(
    overlay_keys: list[_OverlayKey],
) -> dict[_OverlayKey, str]:
    """Return collision-free display names for canonically ordered overlay keys."""
    ambiguous_scopes = _ambiguous_metric_qualifier_scopes(overlay_keys)
    display_names: dict[_OverlayKey, str] = {}
    allocated: dict[str, _OverlayKey] = {}
    for overlay_key in overlay_keys:
        base_name = _overlay_base_display_name(overlay_key, ambiguous_scopes)
        display_names[overlay_key] = _allocate_overlay_display_name(
            base_name, overlay_key, allocated
        )
    return display_names


def _overlay_base_display_name(
    overlay_key: _OverlayKey,
    ambiguous_scopes: set[tuple[str, str, str]],
) -> str:
    field_name = overlay_key.name
    if (
        overlay_key.category == "metric"
        and overlay_key.unit
        and not _metric_name_has_unit_suffix(overlay_key.name, overlay_key.unit)
    ):
        field_name = f"{field_name} ({overlay_key.unit})"
    qualifier_summary = _metric_qualifier_summary(overlay_key, ambiguous_scopes)
    if qualifier_summary:
        field_name = f"{field_name} [{qualifier_summary}]"
    return field_name


def _ambiguous_metric_qualifier_scopes(
    overlay_keys: list[_OverlayKey],
) -> set[tuple[str, str, str]]:
    """Return single-scope summaries that need an explicit scope label."""
    scopes_by_summary: dict[tuple[str, str, str], set[str]] = {}
    for overlay_key in overlay_keys:
        if overlay_key.category != "metric":
            continue
        breakdown_summary = _qualifier_summary(overlay_key.breakdown_qualifiers)
        metric_summary = _qualifier_summary(overlay_key.metric_qualifiers)
        if bool(breakdown_summary) == bool(metric_summary):
            continue
        summary = breakdown_summary or metric_summary
        scope = "breakdown" if breakdown_summary else "metric"
        scopes_by_summary.setdefault(
            (overlay_key.name, overlay_key.unit, summary), set()
        ).add(scope)
    return {
        identity for identity, scopes in scopes_by_summary.items() if len(scopes) > 1
    }


_MAX_QUALIFIER_DISPLAY_BYTES = 120
_STABLE_DIGEST_HEX_LENGTH = 12


def _metric_qualifier_summary(
    overlay_key: _OverlayKey,
    ambiguous_scopes: set[tuple[str, str, str]],
) -> str:
    """Return a readable, length-safe summary without using it as identity."""
    if overlay_key.category != "metric":
        return ""
    breakdown_summary = _qualifier_summary(overlay_key.breakdown_qualifiers)
    metric_summary = _qualifier_summary(overlay_key.metric_qualifiers)
    if breakdown_summary and metric_summary:
        summary = f"breakdown {breakdown_summary}; metric {metric_summary}"
    else:
        summary = breakdown_summary or metric_summary
        if not summary:
            return ""
        identity = (overlay_key.name, overlay_key.unit, summary)
        if identity in ambiguous_scopes:
            scope = "breakdown" if breakdown_summary else "metric"
            summary = f"{scope} {summary}"
    return _bounded_text_with_digest(
        summary,
        _MAX_QUALIFIER_DISPLAY_BYTES,
        _qualifier_identity_digest(overlay_key),
        separator="…#",
    )


def _qualifier_identity_digest(overlay_key: _OverlayKey) -> str:
    """Return a stable digest of complete, scope-aware canonical qualifiers."""
    canonical_identity = json.dumps(
        [overlay_key.breakdown_qualifiers, overlay_key.metric_qualifiers],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_identity).hexdigest()[:_STABLE_DIGEST_HEX_LENGTH]


def _qualifier_summary(qualifiers: _CanonicalQualifiers) -> str:
    return "; ".join(
        f"{_format_qualifier_key(key)}={_format_canonical_json_value(value, True)}"
        for key, value in qualifiers
    )


_QUALIFIER_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.+-]*$")


def _format_qualifier_key(value: str) -> str:
    if _QUALIFIER_TOKEN.fullmatch(value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_canonical_json_value(
    value: _CanonicalJsonValue, allow_bare_string: bool = False
) -> str:
    kind = value[0]
    if kind == "null":
        return "null"
    if kind == "boolean":
        return "true" if value[1] else "false"
    if kind == "number":
        return str(value[1])
    if kind == "string":
        string_value = str(value[1])
        if (
            allow_bare_string
            and _QUALIFIER_TOKEN.fullmatch(string_value)
            and string_value not in {"false", "null", "true"}
        ):
            return string_value
        return json.dumps(string_value, ensure_ascii=False)
    if kind == "array":
        return (
            "["
            + ",".join(_format_canonical_json_value(item) for item in value[1])
            + "]"
        )
    if kind == "object":
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:"
                f"{_format_canonical_json_value(item)}"
                for key, item in value[1]
            )
            + "}"
        )
    raise ValueError(f"Unknown canonical JSON value kind: {kind}")


def _allocate_overlay_display_name(
    base_name: str,
    overlay_key: _OverlayKey,
    allocated: dict[str, _OverlayKey],
) -> str:
    candidate = base_name
    previous_key = allocated.get(candidate.casefold())
    if previous_key is not None:
        qualifier: str
        if overlay_key.category != previous_key.category:
            qualifier = overlay_key.category
        elif overlay_key.result_kind != previous_key.result_kind:
            qualifier = overlay_key.result_kind
        elif overlay_key.result_index != previous_key.result_index:
            qualifier = f"result {overlay_key.result_index + 1}"
        else:
            qualifier = overlay_key.category
        candidate = f"{base_name} [{qualifier}]"

    suffix = 2
    while candidate.casefold() in allocated:
        candidate = f"{base_name} [{qualifier} {suffix}]"
        suffix += 1
    allocated[candidate.casefold()] = overlay_key
    return candidate


def _metric_name_has_unit_suffix(metric_name: str, unit: str) -> bool:
    unit_text = unit.strip().lower()
    if not unit_text:
        return False

    unit_suffixes = {_normalise_unit_suffix(unit_text)}
    if unit_text == "%":
        unit_suffixes.update({"percent", "percentage"})

    name = _normalise_unit_suffix(metric_name)
    return any(
        name == suffix or name.endswith(f"_{suffix}") for suffix in unit_suffixes
    )


def _normalise_unit_suffix(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


_KIND_ORDER = {
    "compatibility": 0,
    "performance": 1,
    "optimization": 2,
}

_OVERLAY_CATEGORY_ORDER = {
    "placement": 0,
    "status": 1,
    "checks": 2,
    "metric": 3,
    "advice": 4,
}


def _overlay_sort_key(
    overlay_key: _OverlayKey,
) -> tuple[
    int,
    str,
    int,
    str,
    str,
    _CanonicalQualifiers,
    _CanonicalQualifiers,
    int,
]:
    return (
        _KIND_ORDER.get(overlay_key.result_kind, len(_KIND_ORDER)),
        overlay_key.result_kind,
        _OVERLAY_CATEGORY_ORDER[overlay_key.category],
        overlay_key.name,
        overlay_key.unit,
        overlay_key.breakdown_qualifiers,
        overlay_key.metric_qualifiers,
        overlay_key.result_index,
    )


def _stringify_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float):
        return str(value)
    return json.dumps(value, sort_keys=True)


_PLACEMENT_COLORS = {
    "npu": "#2e7d32",
    "nx": "#2e7d32",
    "cpu": "#c62828",
    "gpu": "#1565c0",
}


def _placement_color(placement: str) -> str:
    return _PLACEMENT_COLORS.get(placement.lower(), "#757575")


def _check_status_color(status: str) -> str | None:
    colors = {
        "fail": "#c62828",
        "failed": "#c62828",
        "error": "#c62828",
        "warn": "#f9a825",
        "warning": "#f9a825",
    }
    return colors.get(status.lower())


def _severity_color(severity: str) -> str:
    colors = {
        "info": "#1565c0",
        "warning": "#f9a825",
        "error": "#c62828",
    }
    return colors.get(severity.lower(), "#757575")
