# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Artifact generation for the Model Explorer analysis plugin."""

from __future__ import annotations

from mlia.plugins.analysis import AnalysisRunResult

from ._artifacts.formats import _format_integration
from ._artifacts.graphs import (
    _generated_model_artifacts_by_result,
    _write_entity_subgraph_model_file,
)
from ._artifacts.models import ModelExplorerArtifacts
from ._artifacts.navigation import (
    _entity_node_mappings,
    _merge_entity_ref_maps,
    _sync_navigation_data,
)
from ._artifacts.node_data import (
    _entity_metric_overlays,
    _node_data_overlays,
    _write_logical_node_data_files,
)
from ._artifacts.serialization import _manifest, _write_json
from ._artifacts.source_model import (
    _empty_output,
    _model_path,
    _write_source_model_files,
)

__all__ = ["ModelExplorerArtifacts", "write_artifacts"]


def write_artifacts(result: AnalysisRunResult) -> ModelExplorerArtifacts:
    """Write Model Explorer artifact files for an MLIA run."""
    root = result.output_dir / "model-explorer"
    node_data_dir = root / "node-data"
    root.mkdir(parents=True, exist_ok=True)
    node_data_dir.mkdir(parents=True, exist_ok=True)

    output = result.output or _empty_output(result)
    standardized_output_path = root / "standardized-output.json"
    _write_json(standardized_output_path, output)
    visualization_output = output

    model_path = _model_path(result)
    integration = _format_integration(model_path)
    source_model_artifacts = _write_source_model_files(
        model_path, root, visualization_output
    )
    supplementary = integration.supplementary_artifacts(
        result, model_path, root, visualization_output, source_model_artifacts
    )
    for contribution in supplementary.graphs:
        _write_json(contribution.path, contribution.graph_collection)
    model_explorer_model_paths = source_model_artifacts.paths
    source_model_path = (
        model_explorer_model_paths[0] if model_explorer_model_paths else None
    )
    source_operator_to_node_ids = source_model_artifacts.source_operator_to_node_ids
    generated_model_artifacts = _write_entity_subgraph_model_file(
        model_path,
        root,
        visualization_output,
        source_model_artifacts.graph_collection,
        source_operator_to_node_ids,
        presentation=integration.hierarchy_presentation,
    )
    generated_model_paths = [artifact.path for artifact in generated_model_artifacts]
    generated_artifacts_by_result = _generated_model_artifacts_by_result(
        generated_model_artifacts
    )
    sync_entity_refs = supplementary.sync_entity_refs
    supplementary_paths = [contribution.path for contribution in supplementary.graphs]

    node_data_directory_names = {
        artifact.descriptor.collection_label: artifact.path.stem
        for artifact in generated_model_artifacts
    }
    if (
        source_model_path is not None
        and source_model_artifacts.graph_collection is not None
    ):
        node_data_directory_names[
            str(source_model_artifacts.graph_collection["label"])
        ] = source_model_path.stem
    for contribution in supplementary.graphs:
        node_data_directory_names[str(contribution.graph_collection["label"])] = (
            contribution.path.stem
        )
    results = visualization_output.get("results", [])
    if not isinstance(results, list):
        results = []
    logical_overlays = _node_data_overlays(
        results,
        generated_artifacts_by_result,
        source_model_artifacts.graph_collection,
        source_operator_to_node_ids,
        presentation=integration.metric_presentation,
    )
    logical_overlays.extend(
        _entity_metric_overlays(
            visualization_output,
            _merge_entity_ref_maps(
                *(item.metric_refs for item in supplementary.graphs)
            ),
            {str(item.graph_collection["label"]) for item in supplementary.graphs},
            presentation=integration.metric_presentation,
        )
    )
    node_data_artifacts = _write_logical_node_data_files(
        logical_overlays,
        node_data_dir,
        node_data_directory_names,
    )
    node_data_paths = [artifact.path for artifact in node_data_artifacts]
    sync_navigation_path = root / "sync-navigation.json"
    _write_json(
        sync_navigation_path,
        _sync_navigation_data(
            source_model_artifacts.graph_collection,
            generated_model_artifacts,
            sync_entity_refs,
            [item.graph_collection for item in supplementary.graphs],
        ),
    )
    entity_node_mappings_path = root / "entity-node-mappings.json"
    _write_json(
        entity_node_mappings_path,
        _entity_node_mappings(
            source_model_artifacts.graph_collection,
            visualization_output,
            generated_model_artifacts,
            source_operator_to_node_ids,
            sync_entity_refs,
        ),
    )
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        _manifest(
            result,
            visualization_output,
            standardized_output_path,
            source_model_path,
            [
                *model_explorer_model_paths[1:],
                *generated_model_paths,
                *supplementary_paths,
            ],
            node_data_artifacts,
            sync_navigation_path,
            entity_node_mappings_path,
            preferred_graph=integration.preferred_graph(generated_model_artifacts),
        ),
    )

    return ModelExplorerArtifacts(
        root=root,
        manifest=manifest_path,
        standardized_output=standardized_output_path,
        node_data=node_data_paths,
        generated_models=[*generated_model_paths, *supplementary_paths],
        sync_navigation=sync_navigation_path,
        entity_node_mappings=entity_node_mappings_path,
    )
