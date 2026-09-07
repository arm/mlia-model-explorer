# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared models for artifact generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mlia.core import output_schema as schema

_SOURCE_OPERATOR_KIND = schema.ENTITY_KIND_SOURCE_OPERATOR
_MODEL_KIND = schema.ENTITY_KIND_MODEL
_CODE_STACK_KIND = schema.ENTITY_KIND_CODE_STACK
_MODEL_EXPLORER_NAMESPACE_KIND = "model_explorer_namespace"
_BUILT_IN_ENTITY_KINDS = frozenset({_SOURCE_OPERATOR_KIND, _MODEL_KIND})
_ONLY_SHOW_STATS_BEARING_ENTITY_NAMESPACES = False


@dataclass(frozen=True)
class ModelExplorerArtifacts:
    """Paths generated for Model Explorer consumption."""

    root: Path
    manifest: Path
    standardized_output: Path
    node_data: list[Path]
    generated_models: list[Path]
    sync_navigation: Path
    entity_node_mappings: Path


@dataclass(frozen=True)
class _NodeRef:
    """Unambiguous reference to one node in one Model Explorer graph."""

    collection_label: str
    graph_id: str
    node_id: str

    def to_json(self) -> dict[str, str]:
        """Return the stable JSON representation used by mapping artifacts."""
        return {
            "collectionLabel": self.collection_label,
            "graphId": self.graph_id,
            "nodeId": self.node_id,
        }


@dataclass(frozen=True)
class _SourceModelArtifacts:
    """Serialized source model graph artifacts and explicit source mappings."""

    paths: list[Path]
    graph_collection: dict[str, Any] | None = None
    source_operator_to_node_ids: dict[str, list[_NodeRef]] = field(default_factory=dict)


@dataclass(frozen=True)
class _NodeDataArtifact:
    """Generated Model Explorer node-data artifact and its logical overlay."""

    path: Path
    graph_collection_label: str
    overlay_key: _OverlayKey
    display_name: str


@dataclass(frozen=True)
class _GeneratedHierarchyProjection:
    """Explicit correspondence between a hierarchy graph and canonical entities."""

    generated_refs_by_canonical_ref: dict[_NodeRef, tuple[_NodeRef, ...]]
    canonical_ref_by_generated_ref: dict[_NodeRef, _NodeRef]
    source_operator_by_generated_ref: dict[_NodeRef, str]
    group_refs_by_entity_id: dict[str, tuple[_NodeRef, ...]]
    terminal_complete_path_group_refs: frozenset[_NodeRef]


@dataclass(frozen=True)
class _GeneratedModelArtifact:
    """Generated hierarchy graph collection retained with its serialized path."""

    path: Path
    descriptor: _GeneratedEntityHierarchy
    graph_collection: dict[str, Any]
    projection: _GeneratedHierarchyProjection


@dataclass(frozen=True)
class _HierarchyPathProjection:
    """All complete runtime namespace paths for one entity hierarchy."""

    entity_paths_by_id: dict[str, tuple[str, ...]]
    source_operator_paths_by_id: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _HierarchyGraphCollection:
    """Generated graph JSON plus its explicit canonical correspondence."""

    graph_collection: dict[str, Any]
    projection: _GeneratedHierarchyProjection


@dataclass(frozen=True)
class _GraphGroup:
    """One artificial namespace group in a concrete graph projection."""

    node_ref: _NodeRef
    namespace_depth: int
    ordered_leaf_ids: tuple[str, ...]
    canonical_leaf_refs: frozenset[_NodeRef]


@dataclass(frozen=True)
class _GraphProjection:
    """One graph projected from an originating source graph."""

    origin: tuple[str, str]
    collection_label: str
    graph_id: str
    ordered_leaf_refs: tuple[_NodeRef, ...]
    ordered_leaf_node_ids: tuple[str, ...]
    leaf_ids_by_canonical_ref: dict[_NodeRef, tuple[str, ...]]
    groups: tuple[_GraphGroup, ...]
    terminal_complete_path_group_ids: frozenset[str]


@dataclass(frozen=True)
class _EntityKindHierarchy:
    """One declared entity-kind hierarchy rendered as a generated graph."""

    path: tuple[str, ...]
    recursive_kinds: frozenset[str] = frozenset()

    @property
    def backend_kinds(self) -> tuple[str, ...]:
        """Return hierarchy kinds that are not built into MLIA schema."""
        return tuple(kind for kind in self.path if kind not in _BUILT_IN_ENTITY_KINDS)

    @property
    def graph_name(self) -> str:
        """Return the stable generated graph name for this hierarchy."""
        from .graphs import _entity_hierarchy_graph_name

        return _entity_hierarchy_graph_name(self.backend_kinds)


@dataclass(frozen=True)
class _GeneratedEntityHierarchy:
    """One result-local generated graph collection for an entity hierarchy."""

    result_index: int
    result_kind: str
    hierarchy: _EntityKindHierarchy
    graph_name: str
    collection_label: str


_OverlayCategory = Literal["placement", "status", "metric", "checks", "advice"]
_CanonicalJsonValue = tuple[Any, ...]
_CanonicalQualifiers = tuple[tuple[str, _CanonicalJsonValue], ...]
_MISSING_QUALIFIERS = object()


@dataclass(frozen=True)
class _OverlayKey:
    """Stable semantic identity for one logical node-data overlay."""

    result_index: int
    result_kind: str
    category: _OverlayCategory
    name: str
    unit: str = ""
    #: Qualifiers attached to the containing breakdown and shared by every metric
    #: in that breakdown, stored in canonical immutable JSON form.
    breakdown_qualifiers: _CanonicalQualifiers = ()
    #: Qualifiers attached to the individual metric and distinguishing variants of
    #: that metric, stored in canonical immutable JSON form.
    metric_qualifiers: _CanonicalQualifiers = ()


@dataclass(frozen=True)
class _LogicalOverlay:
    """Graph-independent overlay data with resolved presentation."""

    key: _OverlayKey
    display_name: str
    node_data: dict[str, Any]


@dataclass
class _NodeAnnotation:
    """Accumulated textual Model Explorer node-data annotation."""

    value_parts: list[str] = field(default_factory=list)
    bg_color: str | None = None
    text_color: str | None = None

    def add(
        self,
        value: object,
        *,
        bg_color: str | None = None,
        text_color: str | None = None,
    ) -> None:
        """Append a display value and optional colors."""
        from .node_data import _stringify_value

        value_str = _stringify_value(value)
        if value_str and value_str not in self.value_parts:
            self.value_parts.append(value_str)
        if bg_color and not self.bg_color:
            self.bg_color = bg_color
        if text_color and not self.text_color:
            self.text_color = text_color

    def to_node_data_result(self) -> dict[str, Any]:
        """Convert to Model Explorer node-data result JSON."""
        result: dict[str, Any] = {"value": " — ".join(self.value_parts)}
        if self.bg_color:
            result["bgColor"] = self.bg_color
        if self.text_color:
            result["textColor"] = self.text_color
        return result
