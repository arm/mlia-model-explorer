# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Private format boundary for supplementary graphs and presentation policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError
from mlia.plugins.analysis import AnalysisRunResult

from .models import (
    _GeneratedModelArtifact,
    _HierarchyPresentation,
    _MetricPresentation,
    _NodeRef,
    _SourceModelArtifacts,
    _SupplementaryArtifacts,
)


class _FormatIntegration:
    """Default integration preserving the existing source/projection workflow."""

    hierarchy_presentation = _HierarchyPresentation()
    metric_presentation = _MetricPresentation()
    has_source_mapping = False

    def source_collection(
        self, model_path: Path, response: dict[str, Any]
    ) -> Any | None:
        from .source_model import _graph_collection_label

        collections = response.get("graphCollections") or []
        if not collections:
            return None
        if len(collections) != 1:
            labels = ", ".join(
                _graph_collection_label(collection) for collection in collections
            )
            raise RuntimeError(
                "Model Explorer adapter returned multiple graph collections; "
                "mlia-model-explorer currently requires exactly one. "
                f"Returned collections: {labels}."
            )
        return collections[0]

    def source_operator_refs(
        self,
        model_path: Path,
        collection: dict[str, Any],
        output: dict[str, Any] | None,
    ) -> dict[str, list[_NodeRef]]:
        raise NotImplementedError

    def supplementary_artifacts(
        self,
        result: AnalysisRunResult,
        model_path: Path | None,
        root: Path,
        output: dict[str, Any],
        source: _SourceModelArtifacts,
    ) -> _SupplementaryArtifacts:
        if result.args.get("companion_model") is not None:
            raise ConfigurationError(
                "--companion-model is not supported for this model format."
            )
        return _SupplementaryArtifacts()

    def preferred_graph(
        self, generated: list[_GeneratedModelArtifact]
    ) -> dict[str, str] | None:
        return None


def _format_integration(model_path: Path | None) -> _FormatIntegration:
    """Dispatch format ownership without exposing it to artifact consumers."""
    if model_path is not None and model_path.suffix.lower() in {".etrecord", ".bin"}:
        from .etrecord import _ETRecordIntegration

        return _ETRecordIntegration()
    return _FormatIntegration()
