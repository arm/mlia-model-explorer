# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""ExecuTorch graph integration; generic writers consume its contributions."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from mlia.core import output_schema as schema
from mlia.core.errors import ConfigurationError
from mlia.plugins.analysis import AnalysisRunResult
from mlia.utils.filesystem import sha256

from .formats import _FormatIntegration
from .graphs import (
    _allocate_entity_namespace_paths,
    _append_unique_items,
    _entities_by_id,
    _entity_children_by_parent,
    _entity_display_name,
    _entity_id,
    _entity_kind,
    _entity_parent_ids,
    _namespace_group_node_id,
)
from .models import (
    _SOURCE_OPERATOR_KIND,
    _GeneratedModelArtifact,
    _GraphContribution,
    _HierarchyPresentation,
    _MetricPresentation,
    _NodeRef,
    _SourceModelArtifacts,
    _SupplementaryArtifacts,
)
from .navigation import (
    _merge_entity_ref_maps,
    _namespace_entity_refs_as_group_nodes,
    _related_entity_source_refs,
)
from .serialization import _safe_filename
from .source_mapping import _node_string_attr, _vgf_source_operator_node_ids
from .source_model import (
    _graph_collection_label,
    _jsonable,
    _run_model_explorer_adapter,
    _without_none,
)

_PTE_COMPANION_ADAPTER: dict[str, Any] = {
    "module": "pte_adapter_model_explorer.main",
    "extra": "executorch",
    "package": "pte-adapter-model-explorer",
}
_PTE_RUNTIME_COLLECTION_LABEL = "PTE - VGF Backend Hierarchy"
_ETDUMP_INFORMATION_COLLECTION_LABEL = "ETDump Backend Trace"


class _ETRecordIntegration(_FormatIntegration):
    """Present analysis-provided ETRecord identities and companion associations."""

    hierarchy_presentation = _HierarchyPresentation(
        hidden_kinds=frozenset({"backend_operator"}),
        prefer_specific_source_lineages=True,
        collection_labels={"chain": "ETRecord Chain Mapping"},
        group_node_configs=[{"namespaceRegex": ".*", "layoutDirection": 0}],
        preferred_graph_names=("segment_cascade_chain",),
        additional_hierarchy_paths=(("chain", "source_operator"),),
    )
    metric_presentation = _MetricPresentation(
        normalize_colors_across_views=True,
        hidden_statistics_by_metric={
            name: ("sum",)
            for name in (
                "Runtime Total Avg",
                "Runtime Total P50",
                "Runtime Total Min",
                "Runtime Total Max",
                "Runtime Event Count",
            )
        },
    )
    has_source_mapping = True

    def source_collection(
        self, model_path: Path, response: dict[str, Any]
    ) -> Any | None:
        collections = response.get("graphCollections") or []
        if not collections:
            return None
        matches = [
            collection
            for collection in collections
            if _graph_collection_label(collection) == model_path.name
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches and len(collections) == 1:
            return collections[0]
        raise RuntimeError(
            "ETRecord adapter must provide exactly one identifiable source collection."
        )

    def source_operator_refs(
        self,
        model_path: Path,
        collection: dict[str, Any],
        output: dict[str, Any] | None,
    ) -> dict[str, list[_NodeRef]]:
        return _source_operator_refs(collection, output)

    def supplementary_artifacts(
        self,
        result: AnalysisRunResult,
        model_path: Path | None,
        root: Path,
        output: dict[str, Any],
        source: _SourceModelArtifacts,
    ) -> _SupplementaryArtifacts:
        companion = _companion_model_artifact(
            _companion_model_path(result), root, output
        )
        trace = _etdump_information_artifact(
            model_path,
            root,
            output,
        )
        graphs = [graph for graph in (trace, companion) if graph is not None]
        refs = _merge_entity_ref_maps(
            *(graph.sync_refs for graph in graphs),
            _related_entity_source_refs(
                output, source.source_operator_to_node_ids, "backend_operator"
            ),
        )
        return _SupplementaryArtifacts(graphs, refs)

    def preferred_graph(
        self, generated: list[_GeneratedModelArtifact]
    ) -> dict[str, str] | None:
        for artifact in generated:
            kinds = tuple(
                kind
                for kind in artifact.descriptor.hierarchy.backend_kinds
                if kind not in self.hierarchy_presentation.hidden_kinds
            )
            if kinds != ("chain",):
                continue
            collection_label = str(artifact.graph_collection["label"])
            mapped_graph_ids = {
                ref.graph_id
                for refs in artifact.projection.group_refs_by_entity_id.values()
                for ref in refs
                if ref.collection_label == collection_label
            }
            for graph in artifact.graph_collection.get("graphs", []):
                if graph["id"] in mapped_graph_ids:
                    return {
                        "collectionLabel": collection_label,
                        "graphId": str(graph["id"]),
                    }
        return None


def _companion_model_path(result: AnalysisRunResult) -> Path | None:
    """Locate the companion already validated by the analysis workflow."""
    companion = (
        (result.output or {})
        .get("extensions", {})
        .get("executorch", {})
        .get("companion_model")
    )
    if companion is None:
        return None
    path = Path(companion["path"])
    if sha256(path) != companion["sha256"]:
        raise ConfigurationError(
            "Companion model changed after analysis; run analysis again."
        )
    return path


def _companion_model_artifact(
    pte_path: Path | None,
    root: Path,
    output: dict[str, Any],
) -> _GraphContribution | None:
    if pte_path is None:
        return None

    adapter_response = _run_model_explorer_adapter(pte_path, _PTE_COMPANION_ADAPTER)
    raw_collections = adapter_response.get("graphCollections") or []
    if not isinstance(raw_collections, list) or not raw_collections:
        raise RuntimeError("PTE adapter did not return any graph collections.")

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_stem = _safe_filename(pte_path.stem)
    for legacy_suffix in ("pte-embedded-vgf", "pte-compiled-execution"):
        (model_dir / f"{model_stem}-{legacy_suffix}.json").unlink(missing_ok=True)
        shutil.rmtree(
            root / "node-data" / f"{model_stem}-{legacy_suffix}",
            ignore_errors=True,
        )

    graph_collection = _pte_vgf_graph_collection(raw_collections)
    path = model_dir / f"{model_stem}-pte-runtime.json"

    operator_refs: dict[tuple[int, str], list[_NodeRef]] = {}
    embedded_hierarchy_refs = _pte_embedded_vgf_operator_refs(
        output,
        graph_collection,
        operator_refs,
    )
    return _GraphContribution(
        path,
        graph_collection,
        embedded_hierarchy_refs,
        _merge_entity_ref_maps(
            {key: tuple(refs) for key, refs in operator_refs.items()},
            _namespace_entity_refs_as_group_nodes(embedded_hierarchy_refs),
        ),
    )


def _pte_vgf_graph_collection(raw_collections: list[Any]) -> dict[str, Any]:
    """Return only delegate graphs referenced by VgfBackend runtime calls."""
    collections = [_without_none(_jsonable(item)) for item in raw_collections]
    if not all(isinstance(item, dict) for item in collections):
        raise RuntimeError("PTE adapter returned an invalid graph collection.")

    vgf_graph_ids: list[str] = []
    for collection in collections:
        for graph in collection.get("graphs", []) or []:
            if not isinstance(graph, dict):
                continue
            for node in graph.get("nodes", []) or []:
                if not isinstance(node, dict) or node.get("label") != "VgfBackend":
                    continue
                for graph_id in node.get("subgraphIds", []) or []:
                    if isinstance(graph_id, str) and graph_id not in vgf_graph_ids:
                        vgf_graph_ids.append(graph_id)

    if not vgf_graph_ids:
        raise RuntimeError(
            "Companion model does not contain an embedded VgfBackend delegate. "
            "Omit --companion-model for portable or non-VGF programs."
        )

    graphs_by_id = {
        str(graph["id"]): graph
        for collection in collections
        for graph in collection.get("graphs", []) or []
        if isinstance(graph, dict) and graph.get("id")
    }
    missing_graph_ids = [
        graph_id for graph_id in vgf_graph_ids if graph_id not in graphs_by_id
    ]
    if missing_graph_ids:
        raise RuntimeError(
            "PTE adapter did not return VGF delegate graph(s) referenced by the "
            f"runtime program: {missing_graph_ids!r}."
        )
    return {
        "label": _PTE_RUNTIME_COLLECTION_LABEL,
        "graphs": [graphs_by_id[graph_id] for graph_id in vgf_graph_ids],
    }


def _etdump_information_artifact(
    model_path: Path | None,
    root: Path,
    output: dict[str, Any],
) -> _GraphContribution | None:
    """Write a segment/cascade/chain view with ETDump backend operations."""
    collection, overlay_refs, sync_refs = _etdump_information_collection(
        output,
    )
    if collection is None:
        return None

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_stem = _safe_filename(model_path.stem) if model_path else "model"
    path = model_dir / f"{model_stem}-etdump-information.json"
    return _GraphContribution(path, collection, overlay_refs, sync_refs)


def _etdump_information_collection(
    output: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    dict[tuple[int, str], tuple[_NodeRef, ...]],
    dict[tuple[int, str], tuple[_NodeRef, ...]],
]:
    """Build ETDump hierarchy graphs and canonical hierarchy references."""
    graphs: list[dict[str, Any]] = []
    overlay_refs: dict[tuple[int, str], list[_NodeRef]] = {}
    sync_refs: dict[tuple[int, str], list[_NodeRef]] = {}
    results = output.get("results", [])
    if not isinstance(results, list):
        return None, {}, {}

    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        entities = _entities_by_id(result)
        children_by_parent = _entity_children_by_parent(entities)
        chains = [
            entity for entity in entities.values() if _entity_kind(entity) == "chain"
        ]
        namespaces_by_chain = _chain_entity_namespaces(entities)
        graph_id = f"etdump_information-{result_index}"
        nodes: list[dict[str, Any]] = []
        first_node_id_by_chain: dict[str, str] = {}
        last_node_id_by_chain: dict[str, str] = {}
        chain_names_by_id: dict[str, str] = {}
        for chain_index, chain in enumerate(chains):
            chain_id = _entity_id(chain)
            stripe_entities = [
                entities[child_id]
                for child_id in children_by_parent.get(chain_id, [])
                if child_id in entities and _entity_kind(entities[child_id]) == "stripe"
            ]
            backend_operations = _compiled_chain_backend_operations(stripe_entities)
            if not backend_operations:
                continue

            entity_namespaces = namespaces_by_chain[chain_id]
            namespace = entity_namespaces[-1][1]

            source_entities = [
                entities[child_id]
                for child_id in children_by_parent.get(chain_id, [])
                if child_id in entities
                and _entity_kind(entities[child_id]) == _SOURCE_OPERATOR_KIND
            ]
            stripe_ids = [
                str(attributes["stripe_id"])
                for stripe in stripe_entities
                if isinstance((attributes := stripe.get("attributes")), dict)
                and attributes.get("stripe_id") is not None
            ]
            previous_node_id: str | None = None
            for operation_index, operation in enumerate(backend_operations):
                node_id = (
                    f"etdump_operation_{result_index}_{chain_index}_{operation_index}"
                )
                attrs = [
                    {"key": "Execution level", "value": "ETDump backend operation"},
                    {"key": "Chain", "value": _entity_display_name(chain)},
                    {"key": "TOSA ID", "value": operation["tosa_id"]},
                    {
                        "key": "SPIR-V IDs",
                        "value": ", ".join(operation["spirv_ids"]),
                    },
                    {
                        "key": "API labels",
                        "value": ", ".join(operation["api_labels"]),
                    },
                    {"key": "Observed stripe IDs", "value": ", ".join(stripe_ids)},
                    {
                        "key": "Related ETRecord operators",
                        "value": ", ".join(
                            _entity_display_name(entity) for entity in source_entities
                        ),
                    },
                    {
                        "key": "Statistic scope",
                        "value": (
                            "Profiling is shown on the chain group; "
                            "no per-operator timing"
                        ),
                    },
                ]
                incoming_edges = []
                if previous_node_id is not None:
                    incoming_edges.append(
                        {
                            "sourceNodeId": previous_node_id,
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": "0",
                        }
                    )
                nodes.append(
                    {
                        "id": node_id,
                        "label": operation["operator"],
                        "namespace": namespace,
                        "subgraphIds": [],
                        "attrs": [attr for attr in attrs if attr["value"]],
                        "incomingEdges": incoming_edges,
                        "outputsMetadata": [{"id": "0", "attrs": []}],
                        "inputsMetadata": (
                            [{"id": "0", "attrs": []}]
                            if previous_node_id is not None
                            else []
                        ),
                    }
                )
                operator_ref = _NodeRef(
                    _ETDUMP_INFORMATION_COLLECTION_LABEL, graph_id, node_id
                )
                for entity_id in operation["entity_ids"]:
                    if entity_id in entities:
                        _append_unique_items(
                            sync_refs.setdefault((result_index, entity_id), []),
                            [operator_ref],
                        )
                _append_operator_mapping_attrs(
                    nodes[-1],
                    operation["entity_ids"],
                    entities,
                    "Compiler identity",
                    children_by_parent,
                )
                previous_node_id = node_id

            first_node_id_by_chain[chain_id] = (
                f"etdump_operation_{result_index}_{chain_index}_0"
            )
            if previous_node_id is not None:
                last_node_id_by_chain[chain_id] = previous_node_id
            chain_names_by_id[chain_id] = _entity_display_name(chain)

            for entity, entity_namespace in entity_namespaces:
                entity_id = _entity_id(entity)
                group_node_id = _namespace_group_node_id(entity_namespace)
                key = (result_index, entity_id)
                _append_unique_items(
                    overlay_refs.setdefault(key, []),
                    [
                        _NodeRef(
                            _ETDUMP_INFORMATION_COLLECTION_LABEL,
                            graph_id,
                            group_node_id.removesuffix("___group___"),
                        )
                    ],
                )
                _append_unique_items(
                    sync_refs.setdefault(key, []),
                    [
                        _NodeRef(
                            _ETDUMP_INFORMATION_COLLECTION_LABEL,
                            graph_id,
                            group_node_id,
                        )
                    ],
                )

        node_by_id = {
            str(node["id"]): node for node in nodes if isinstance(node.get("id"), str)
        }
        predecessors_by_target: dict[str, list[str]] = {}
        for relationship in (
            output.get("extensions", {})
            .get("executorch", {})
            .get("chain_relationships", [])
        ):
            if relationship["result_index"] != result_index:
                continue
            source_chain_id, target_chain_id = (
                relationship["source_id"],
                relationship["target_id"],
            )
            source_node_id = last_node_id_by_chain.get(source_chain_id)
            target_node_id = first_node_id_by_chain.get(target_chain_id)
            if source_node_id is None or target_node_id is None:
                continue
            target_node = node_by_id[target_node_id]
            incoming_edges = target_node.setdefault("incomingEdges", [])
            if any(
                edge.get("sourceNodeId") == source_node_id
                for edge in incoming_edges
                if isinstance(edge, dict)
            ):
                continue
            inputs_metadata = target_node.setdefault("inputsMetadata", [])
            input_id = f"etrecord_chain_{len(inputs_metadata)}"
            inputs_metadata.append({"id": input_id, "attrs": []})
            incoming_edges.append(
                {
                    "sourceNodeId": source_node_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": input_id,
                }
            )
            _append_unique_items(
                predecessors_by_target.setdefault(target_node_id, []),
                [chain_names_by_id.get(source_chain_id, source_chain_id)],
            )
        for target_node_id, predecessor_names in predecessors_by_target.items():
            target_node = node_by_id[target_node_id]
            _append_node_attr(
                target_node,
                "ETRecord predecessor chains",
                ", ".join(predecessor_names),
            )
            _append_node_attr(
                target_node,
                "Inter-chain relationship",
                "Inferred from ETRecord source-operator topology",
            )

        if nodes:
            graphs.append({"id": graph_id, "nodes": nodes})

    if not graphs:
        return None, {}, {}
    return (
        {"label": _ETDUMP_INFORMATION_COLLECTION_LABEL, "graphs": graphs},
        {key: tuple(refs) for key, refs in overlay_refs.items()},
        {key: tuple(refs) for key, refs in sync_refs.items()},
    )


def _parent_entity_of_kind(
    entity: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    kind: str,
) -> dict[str, Any] | None:
    """Return the first direct parent with the requested entity kind."""
    return next(
        (
            parent
            for parent_id in _entity_parent_ids(entity)
            if (parent := entities.get(parent_id)) is not None
            and _entity_kind(parent) == kind
        ),
        None,
    )


def _compiled_chain_backend_operations(
    stripe_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return deduplicated compiler operations and their shared entity IDs."""
    operations: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for stripe in stripe_entities:
        attributes = stripe.get("attributes")
        if not isinstance(attributes, dict):
            continue
        for operation in attributes.get("backend_operations", []) or []:
            if not isinstance(operation, dict):
                continue
            operator = str(operation.get("operator") or "BackendOp")
            tosa_id = str(operation.get("tosa_id") or "")
            api_labels = tuple(
                str(label) for label in operation.get("api_labels", []) or []
            )
            identity = (tosa_id, operator, api_labels)
            compiled = operations.setdefault(
                identity,
                {
                    "operator": operator,
                    "tosa_id": tosa_id,
                    "api_labels": api_labels,
                    "spirv_ids": tuple(
                        match.group(1)
                        for label in api_labels
                        for match in re.finditer(r"spirv_id_([0-9]+)", label)
                    ),
                    "entity_ids": [],
                },
            )
            entity_id = operation.get("entity_id")
            if isinstance(entity_id, str) and entity_id:
                _append_unique_items(compiled["entity_ids"], [entity_id])
    return list(operations.values())


def _companion_graph_matches_method(graph_id: str, method: str) -> bool:
    """Compare the method encoded in a PTE delegate graph's public identity."""
    # Adapter releases emit either method_chain_instruction or a model-qualified
    # version of that identity. Preserve the method check for both forms.
    instruction_id = graph_id.rsplit("::", 1)[-1]
    match = re.fullmatch(r"(.+)_[0-9]+_[0-9]+", instruction_id)
    return match is not None and match.group(1) == method


def _pte_embedded_vgf_operator_refs(
    output: dict[str, Any],
    graph_collection: dict[str, Any] | None,
    operator_refs: dict[tuple[int, str], list[_NodeRef]] | None = None,
) -> dict[tuple[int, str], tuple[_NodeRef, ...]]:
    """Translate structural companion links into adapter references and layout."""
    if graph_collection is None:
        return {}
    companion = (
        output.get("extensions", {}).get("executorch", {}).get("companion_model", {})
    )
    delegates = {
        delegate["id"]: delegate for delegate in companion.get("delegates", [])
    }
    graphs = graph_collection.get("graphs", [])
    graphs_by_instruction: dict[str, list[dict[str, Any]]] = {}
    nodes_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for graph in graphs:
        graph_id = graph["id"]
        graphs_by_instruction.setdefault(graph_id.rsplit("::", 1)[-1], []).append(graph)
        for node in graph.get("nodes", []):
            nodes_by_identity[(graph_id, node["id"])] = node
    references: dict[tuple[str, str], list[_NodeRef]] = {}
    for identity, refs in _vgf_source_operator_node_ids(graph_collection).items():
        for ref in refs:
            references.setdefault((ref.graph_id, identity), []).append(ref)
    if operator_refs is None:
        operator_refs = {}
    hierarchy_refs: dict[tuple[int, str], list[_NodeRef]] = {}
    entities_by_result = {
        index: _entities_by_id(result)
        for index, result in enumerate(output.get("results", []))
        if isinstance(result, dict)
    }
    children_by_result = {
        index: _entity_children_by_parent(entities)
        for index, entities in entities_by_result.items()
    }
    namespaces_by_result = {
        index: _chain_entity_namespaces(entities)
        for index, entities in entities_by_result.items()
    }
    assigned = set()
    for association in companion.get("operator_associations", []):
        delegate = delegates[association["delegate_id"]]
        instruction = (
            f"{delegate['method']}_{delegate['chain_index']}_"
            f"{delegate['instruction_id']}"
        )
        matching_graphs = graphs_by_instruction.get(instruction, [])
        if len(matching_graphs) != 1 or not _companion_graph_matches_method(
            matching_graphs[0]["id"], delegate["method"]
        ):
            raise ConfigurationError(
                "PTE adapter must expose one graph for deployed instruction "
                f"{instruction!r}"
            )
        graph = matching_graphs[0]
        graph_id = graph["id"]
        identity = schema.vgf_source_operator_id(
            association["segment_index"], association["spirv_id"]
        )
        candidates = references.get((graph_id, identity), [])
        if len(candidates) != 1:
            reasons = [
                _node_string_attr(node, "reason")
                for node in graph.get("nodes", [])
                if "(unavailable)" in node.get("label", "")
            ]
            if any(reasons):
                raise ConfigurationError(
                    "Companion graph is unavailable: "
                    + "; ".join(reason for reason in reasons if reason)
                )
            raise ConfigurationError(
                "PTE/VGF adapter must expose the structural identity "
                f"{identity!r} in {graph_id!r}. "
                "Install adapters that preserve segment and SPIR-V result IDs."
            )
        ref = candidates[0]
        node = nodes_by_identity[(graph_id, ref.node_id)]
        result_index = association["result_index"]
        entities = entities_by_result[result_index]
        backend_ids = association["backend_operator_ids"]
        for entity_id in backend_ids:
            _append_unique_items(
                operator_refs.setdefault((result_index, entity_id), []), [ref]
            )
        _append_operator_mapping_attrs(
            node,
            backend_ids,
            entities,
            "Structural SPIR-V association",
            children_by_result[result_index],
        )
        _append_node_attr(node, "ETDump API label", association["api_label"])
        _append_node_attr(node, "ETDump SPIR-V ID", str(association["spirv_id"]))
        _append_node_attr(node, "ETDump chains", ", ".join(association["chain_names"]))
        _append_node_attr(
            node,
            "ETDump compiler operations",
            ", ".join(association["compiler_operations"]),
        )
        chain_id = association["chain_id"]
        _append_node_attr(
            node,
            "ETDump statistic scope",
            "Profiling is shown on the chain group; no per-operator timing"
            if chain_id
            else "Multiple associated chains; no unique chain aggregate",
        )
        if chain_id is not None:
            namespaces = namespaces_by_result[result_index][chain_id]
            for entity, namespace in namespaces:
                group_ref = _NodeRef(
                    _PTE_RUNTIME_COLLECTION_LABEL,
                    graph_id,
                    _namespace_group_node_id(namespace).removesuffix("___group___"),
                )
                _append_unique_items(
                    hierarchy_refs.setdefault((result_index, _entity_id(entity)), []),
                    [group_ref],
                )
            if namespaces:
                if node.get("namespace"):
                    _append_node_attr(node, "PTE embedded namespace", node["namespace"])
                node["namespace"] = namespaces[-1][1]
                assigned.add(ref)
    for graph in graphs:
        for node in graph.get("nodes", []):
            ref = _NodeRef(_PTE_RUNTIME_COLLECTION_LABEL, graph["id"], node["id"])
            if ref not in assigned and node.get("namespace"):
                _append_node_attr(node, "PTE embedded namespace", node["namespace"])
                _append_node_attr(
                    node, "ETDump hierarchy mapping", "No unique chain association"
                )
    return {key: tuple(refs) for key, refs in hierarchy_refs.items()}


def _append_operator_mapping_attrs(
    node: dict[str, Any],
    entity_ids: list[str],
    entities: dict[str, dict[str, Any]],
    strategy: str,
    children_by_parent: dict[str, list[str]],
) -> None:
    """Expose navigation identity and distinguish heuristic source provenance."""
    if not entity_ids:
        return
    _append_node_attr(node, "MLIA operator identities", ", ".join(entity_ids))
    _append_node_attr(node, "Operator mapping", strategy)
    associations = []
    for entity_id in entity_ids:
        entity = entities.get(entity_id, {})
        if not children_by_parent.get(entity_id):
            continue
        attributes = entity.get("attributes", {})
        associations.append(
            "Heuristic representative source "
            f"({attributes.get('mapping_confidence', 'unknown')}): "
            f"{attributes.get('mapping_strategy', 'representative source association')}"
        )
    if associations:
        _append_node_attr(
            node, "ETRecord source association", "; ".join(sorted(set(associations)))
        )


def _append_node_attr(node: dict[str, Any], key: str, value: str) -> None:
    """Append one non-empty node attribute unless the same key already exists."""
    if not value:
        return
    attrs = node.setdefault("attrs", [])
    if not isinstance(attrs, list):
        return
    if any(isinstance(attr, dict) and attr.get("key") == key for attr in attrs):
        return
    attrs.append({"key": key, "value": value})


def _chain_entity_namespaces(
    entities: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], str]]]:
    """Allocate one identity-preserving hierarchy for PTE and measured views."""
    lineages = {}
    components = {}
    prefixes = {}
    for chain_id, chain in entities.items():
        if _entity_kind(chain) != "chain":
            continue
        cascade = _parent_entity_of_kind(chain, entities, "cascade")
        segment = _parent_entity_of_kind(
            cascade if cascade is not None else chain, entities, "segment"
        )
        lineage = tuple(
            _entity_id(entity)
            for entity in (segment, cascade, chain)
            if entity is not None
        )
        lineages[chain_id] = lineage
        for index, entity_id in enumerate(lineage, 1):
            prefix = lineage[:index]
            components[prefix] = _entity_display_name(entities[entity_id])
            prefixes[prefix] = tuple(lineage[:length] for length in range(1, index + 1))
    namespaces = _allocate_entity_namespace_paths(
        components, {prefix: prefix[-1] for prefix in components}, prefixes
    )
    return {
        chain_id: [
            (entities[entity_id], namespaces[lineage[:index]])
            for index, entity_id in enumerate(lineage, 1)
        ]
        for chain_id, lineage in lineages.items()
    }


def _legacy_source_operator_node_ids(
    graph_collection: dict[str, Any],
    output: dict[str, Any] | None,
) -> dict[str, list[_NodeRef]]:
    """Map the selected ETRecord method using analysis identity metadata."""
    source_graph = (
        (output or {})
        .get("extensions", {})
        .get("executorch", {})
        .get("source_graph", {})
    )
    if not source_graph:
        raise ConfigurationError(
            "ETRecord source identity metadata is missing; rerun analysis "
            "or upgrade the ETRecord adapter."
        )

    graph_id = source_graph.get("method")
    selected_graphs = [
        graph
        for graph in graph_collection.get("graphs", []) or []
        if isinstance(graph, dict) and graph.get("id") == graph_id
    ]
    if not isinstance(graph_id, str) or not graph_id or len(selected_graphs) != 1:
        raise ConfigurationError(
            "ETRecord adapter must emit exactly one graph for selected method "
            f"{graph_id!r}; found {len(selected_graphs)}."
        )

    graph = selected_graphs[0]
    collection_label = str(graph_collection["label"])
    operation_kinds = {"call_function", "call_method", "call_module"}
    source_nodes = [
        node for node in source_graph.get("nodes", []) if node["op"] in operation_kinds
    ]
    adapter_nodes = [
        node
        for node in graph.get("nodes", []) or []
        if isinstance(node, dict) and _node_string_attr(node, "Op Type") is not None
    ]
    if len(source_nodes) != len(adapter_nodes):
        raise RuntimeError(
            f"ETRecord adapter graph {graph_id!r} emitted {len(adapter_nodes)} "
            f"operation nodes for {len(source_nodes)} ETRecord operations."
        )

    mappings: dict[str, list[_NodeRef]] = {}
    for source_node, adapter_node in zip(source_nodes, adapter_nodes):
        node_id = adapter_node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise RuntimeError(
                f"ETRecord adapter graph {graph_id!r} contains an operation "
                "node without an ID."
            )
        source_handle = source_node.get("debug_handle")
        adapter_handle = _node_string_attr(adapter_node, "debug_handle")
        if (
            source_handle is not None
            and adapter_handle is not None
            and str(source_handle) != adapter_handle
        ):
            raise RuntimeError(
                f"ETRecord adapter operation order mismatch in graph {graph_id!r}: "
                f"source node {source_node['name']!r} has debug handle "
                f"{source_handle}, but adapter node {node_id!r} has "
                f"{adapter_handle}."
            )
        canonical_id = source_node["id"]
        mappings[canonical_id] = [_NodeRef(collection_label, graph_id, node_id)]
    return mappings


def _normalize_selected_method_graph(
    collection: dict[str, Any], output: dict[str, Any] | None
) -> None:
    """Normalize legacy labels and open the confirmed method's source view first."""
    source_graph = (
        (output or {})
        .get("extensions", {})
        .get("executorch", {})
        .get("source_graph", {})
    )
    method = source_graph.get("method")
    if not isinstance(method, str) or not method:
        return
    graphs = [
        graph for graph in collection.get("graphs", []) or [] if isinstance(graph, dict)
    ]
    if (
        source_graph.get("methods") == [method]
        and len(graphs) == 1
        and graphs[0].get("id") == "forward"
    ):
        graphs[0]["id"] = method
    matches = [graph for graph in graphs if graph.get("id") == method]
    if len(matches) != 1:
        raise ConfigurationError(
            "ETRecord adapter must emit exactly one graph for selected method "
            f"{method!r}; found {len(matches)}."
        )
    collection["graphs"] = [
        matches[0],
        *(graph for graph in graphs if graph is not matches[0]),
    ]


def _source_operator_refs(
    collection: dict[str, Any],
    output: dict[str, Any] | None,
) -> dict[str, list[_NodeRef]]:
    """Consume native node names and method-qualified canonical identities."""
    _normalize_selected_method_graph(collection, output)
    mappings: dict[str, list[_NodeRef]] = {}
    collection_label = str(collection["label"])
    for graph in collection.get("graphs", []) or []:
        if (
            not isinstance(graph, dict)
            or not isinstance(graph.get("id"), str)
            or not graph["id"]
        ):
            continue
        method = graph["id"]
        nodes = [
            node
            for node in graph.get("nodes", []) or []
            if isinstance(node, dict) and _node_string_attr(node, "Op Type") is not None
        ]
        if any(_node_string_attr(node, "Node Name") is None for node in nodes):
            # Older adapter releases lack identity metadata. Preserve their
            # validation through the previous parser boundary rather than guess.
            return _legacy_source_operator_node_ids(collection, output)
        prefix = f"source_operator/etrecord/{method}/"
        nodes_by_name: dict[str, list[dict[str, Any]]] = {}
        nodes_by_handle: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            name = _native_source_name(node)
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id:
                raise RuntimeError(
                    f"ETRecord adapter graph {method!r} contains an operation "
                    "node without an ID."
                )
            canonical = prefix + str(name)
            if canonical in mappings:
                raise RuntimeError(
                    f"ETRecord adapter graph {method!r} has ambiguous source "
                    f"name {name!r}."
                )
            mappings[canonical] = [_NodeRef(collection_label, method, node_id)]
            nodes_by_name[name] = [node]
            native_handle = _node_string_attr(node, "debug_handle")
            if native_handle is not None:
                nodes_by_handle.setdefault(native_handle, []).append(node)
        expected = {
            entity_id: entity
            for result in (output or {}).get("results", []) or []
            if isinstance(result, dict)
            for entity_id, entity in _entities_by_id(result).items()
            if _entity_kind(entity) == "source_operator"
            and entity_id.startswith(prefix)
        }
        for entity_id, entity in expected.items():
            name = str(entity.get("name") or entity_id[len(prefix) :])
            handle = entity.get("attributes", {}).get("debug_handle")
            candidates = nodes_by_name.get(name, [])
            if not candidates and handle is not None:
                candidates = nodes_by_handle.get(str(handle), [])
            if len(candidates) != 1:
                raise RuntimeError(
                    f"ETRecord source identity {entity_id!r} requires exactly "
                    f"one adapter operation; found {len(candidates)}."
                )
            node = candidates[0]
            native_handle = _node_string_attr(node, "debug_handle")
            if (
                handle is not None
                and native_handle is not None
                and str(handle) != native_handle
            ):
                raise RuntimeError(
                    f"ETRecord source identity {entity_id!r} has debug handle "
                    f"{handle}, but adapter node {node['id']!r} has {native_handle}."
                )
            mappings[entity_id] = [_NodeRef(collection_label, method, str(node["id"]))]
    return mappings


def _native_source_name(node: dict[str, Any]) -> str:
    """Undo ExecuTorch FXOperatorGraph's single underscore suffix."""
    name = _node_string_attr(node, "Node Name") or ""
    return name[:-1] if name.endswith("_") else name
