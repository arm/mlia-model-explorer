# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Model Explorer analysis plugin."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from mlia.core.errors import ConfigurationError, UnsupportedConfigurationError
from mlia.core.output_validation import SchemaValidationError
from mlia.plugins.analysis import AnalysisPluginRegistry, AnalysisRunResult

from mlia.model_explorer import bundler as bundler_module
from mlia.model_explorer._artifacts import graphs as graphs_module
from mlia.model_explorer._artifacts import models as artifact_models
from mlia.model_explorer._artifacts import node_data as node_data_module
from mlia.model_explorer._artifacts import serialization as serialization_module
from mlia.model_explorer._artifacts import source_mapping as source_mapping_module
from mlia.model_explorer._artifacts import source_model as source_model_module
from mlia.model_explorer.artifacts import write_artifacts
from mlia.model_explorer.bundler import build_artifacts_bundle, open_artifacts_bundle
from mlia.model_explorer.graph_data import model_explorer_normalized_namespace
from mlia.model_explorer.plugin import ModelExplorerAnalysisPlugin

_DEFAULT_OUTPUT = object()


def _native_source_operator_ids(model_name: str) -> list[str]:
    """Return canonical source identities represented by the fake adapter output."""
    suffix = Path(model_name).suffix.lower()
    if suffix == ".tflite":
        return ["source_operator/operator/0", "source_operator/operator/1"]
    if suffix == ".vgf":
        return [
            "source_operator/segment_0/spirv-0",
            "source_operator/segment_0/spirv-1",
        ]
    if suffix == ".spv":
        return ["source_operator/segment_0/spirv-60"]
    return []


def _add_minimal_native_source_leaves(output: dict, model_name: str) -> dict:
    """Give non-mapping tests realistic descendant source operators."""
    source_ids = _native_source_operator_ids(model_name)
    if not source_ids:
        return output
    for result in output.get("results", []):
        if not isinstance(result, dict):
            continue
        entities = result.get("entities", [])
        if not isinstance(entities, list):
            continue
        result["entities"] = entities = [dict(entity) for entity in entities]
        existing_ids = {
            entity.get("id") for entity in entities if isinstance(entity, dict)
        }
        parents = {
            entity.get("id")
            for entity in entities
            if isinstance(entity, dict) and entity.get("child_ids")
        }
        parents.update(
            parent_id
            for entity in entities
            if isinstance(entity, dict)
            for parent_id in entity.get("parent_ids", []) or []
            if isinstance(parent_id, str)
        )
        terminals = [
            entity
            for entity in entities
            if isinstance(entity, dict)
            and entity.get("kind") != "source_operator"
            and isinstance(entity.get("id"), str)
            and entity["id"] not in parents
        ]
        for index, terminal in enumerate(terminals):
            source_id = source_ids[index % len(source_ids)]
            if source_id in existing_ids:
                source = next(
                    entity for entity in entities if entity.get("id") == source_id
                )
                source_parents = source.setdefault("parent_ids", [])
                if terminal["id"] not in source_parents:
                    source_parents.append(terminal["id"])
                continue
            entities.append(
                {
                    "id": source_id,
                    "name": source_id.rsplit("/", 1)[-1],
                    "kind": "source_operator",
                    "parent_ids": [terminal["id"]],
                }
            )
            existing_ids.add(source_id)
    return output


def _run_result(
    tmp_path: Path,
    *,
    output: dict | None | object = _DEFAULT_OUTPUT,
    model_exists: bool = True,
    mode: str = "files",
    model_name: str = "model.bin",
    settings: dict[str, object] | None = None,
    cli_model: bool = True,
) -> AnalysisRunResult:
    model = tmp_path / model_name
    if model_exists:
        model.write_bytes(b"model")
    ctx = MagicMock()
    ctx.output_dir = tmp_path / "mlia-output"
    ctx.output_dir.mkdir(exist_ok=True)
    default_output = {
        "schema_version": "1.2.0",
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "timestamp": "2026-07-17T12:00:00Z",
        "tool": {"name": "mlia", "version": "0.11.0"},
        "model": {
            "name": model.name,
            "format": model.suffix.lstrip(".") or "unknown",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "target": {
            "profile_name": "test-target",
            "target_type": "test",
            "components": [{"type": "npu", "family": "test"}],
            "configuration": {},
        },
        "context": {},
        "backends": [
            {
                "id": "test",
                "name": "test",
                "version": "1.0.0",
                "configuration": {},
            }
        ],
        "results": [
            {
                "kind": "compatibility",
                "status": "partial",
                "producer": "test",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "kind": "source_operator",
                        "name": "CONV_2D",
                        "placement": "NPU",
                    }
                ],
                "checks": [
                    {
                        "id": "npu_support_op_0",
                        "status": "fail",
                        "entity_id": "source_operator/operator/0",
                        "details": {
                            "reasons": [
                                {"description": "missing quantization parameters"}
                            ]
                        },
                    }
                ],
                "advice": [
                    {
                        "id": "advice-1",
                        "category": "compatibility",
                        "severity": "info",
                        "message": "test advice",
                        "affected_entity_ids": ["source_operator/operator/0"],
                    }
                ],
            }
        ],
    }
    resolved_output = default_output if output is _DEFAULT_OUTPUT else output
    if isinstance(resolved_output, dict):
        resolved_output = _add_minimal_native_source_leaves(resolved_output, model_name)
    return AnalysisRunResult(
        output=resolved_output,
        context=ctx,
        args={"command": "check", "model_explorer": mode},
        command_name="check",
        parameters={
            "model": str(model) if cli_model else None,
            "target_profile": "test-target",
        },
        settings=settings or {},
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_bundle_data(index: Path) -> dict:
    html = index.read_text(encoding="utf-8")
    marker = '<script id="mlia-model-explorer-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _test_hierarchy(graph_name: str):
    return next(
        hierarchy
        for hierarchy in graphs_module._result_entity_kind_hierarchies(
            {
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {"kind": "segment"},
                    {"kind": "cascade"},
                    {"kind": "chain"},
                    {"kind": "source_operator"},
                    {"kind": "nn_module"},
                ],
            }
        )
        if hierarchy.graph_name == graph_name
    )


_TEST_ENTITY_KINDS = [
    {"id": "segment", "child_kinds": ["cascade"]},
    {"id": "cascade", "parent_kinds": ["segment"], "child_kinds": ["chain"]},
    {"id": "chain", "parent_kinds": ["cascade"], "child_kinds": ["source_operator"]},
    {
        "id": "nn_module",
        "parent_kinds": ["nn_module"],
        "child_kinds": ["source_operator"],
    },
]


class _FakeModelExplorerConfig:
    """Minimal fake for Model Explorer config tests."""

    def __init__(self) -> None:
        """Create an empty fake config."""
        self.models: list[dict[str, object]] = []
        self.node_data: list[dict[str, str]] = []
        self.ui_state: dict[str, object] | None = None
        self.selected_node_data: list[dict[str, str]] = []
        self.sync_navigation: str | None = None
        self.sync_navigation_data_path: str | None = None
        self.keep_layers_with_a_single_child: bool | None = None
        self.artificial_layer_node_count_threshold: int | None = None

    def add_model_from_path(self, path: str, **kwargs: object) -> None:
        """Record a model path."""
        self.models.append({"path": path, **kwargs})

    def add_graph_json_from_path(self, path: str, **kwargs: object) -> None:
        """Record a graph JSON path."""
        self.models.append({"path": path, **kwargs})

    def add_node_data(
        self, *, name: str, node_data: str, model_name: str | None = None
    ) -> None:
        """Record in-memory node data."""
        self.node_data.append(
            {"name": name, "node_data": node_data, "model_name": model_name or ""}
        )

    def set_initial_ui_state(self, ui_state: dict[str, object]) -> None:
        """Record the initial UI state."""
        self.ui_state = ui_state

    def set_sync_navigation(self, mode: str) -> None:
        """Record the sync navigation mode."""
        self.sync_navigation = mode

    def set_selected_node_data(self, selected_node_data: list[dict[str, str]]) -> None:
        """Record selected node data preferences."""
        self.selected_node_data = selected_node_data

    def set_keep_layers_with_a_single_child(self, keep: bool = True) -> None:
        """Record whether single-child layers are kept."""
        self.keep_layers_with_a_single_child = keep

    def set_artificial_layer_node_count_threshold(self, threshold: int) -> None:
        """Record the artificial layer threshold."""
        self.artificial_layer_node_count_threshold = threshold

    def set_sync_navigation_data_from_path(self, path: str) -> None:
        """Record a sync navigation data path."""
        self.sync_navigation_data_path = path
        self.sync_navigation = "from_cns"


class _FakeAdapter:
    """Fake VGF adapter shaped like the real VGF/SPIR-V adapter output."""

    metadata = SimpleNamespace(fileExts=["vgf"])

    def convert(self, model_path: str, settings: dict) -> dict:
        """Return SPIR-V operation nodes with native IDs and attributes."""
        return {
            "graphs": [
                {
                    "id": "Main",
                    "nodes": [
                        {
                            "id": "0__op_0",
                            "label": Path(model_path).stem,
                            "namespace": "original/group",
                            "attrs": [
                                {"key": "spv id", "value": "%0"},
                                {"key": "dtype", "value": "int8"},
                            ],
                        },
                        {
                            "id": "0__op_1",
                            "label": "consumer",
                            "namespace": "original/group",
                            "attrs": [{"key": "spv id", "value": "%1"}],
                            "incomingEdges": [{"sourceNodeId": "0__op_0"}],
                        },
                    ],
                }
            ]
        }


class _SpirvIdAdapter:
    """Fake adapter with real VGF/SPIR-V node metadata."""

    metadata = SimpleNamespace(fileExts=["vgf"])

    def convert(self, model_path: str, settings: dict) -> dict:
        return {
            "graphs": [
                {
                    "id": "Main",
                    "nodes": [
                        {
                            "id": "0__op_60",
                            "label": Path(model_path).stem,
                            "namespace": "graph_segment_0",
                            "attrs": [{"key": "spv id", "value": "%60"}],
                        }
                    ],
                }
            ]
        }


class _StandaloneSpirvAdapter:
    """Fake standalone SPIR-V adapter with native result-ID metadata."""

    metadata = SimpleNamespace(fileExts=["spv"])

    def convert(self, model_path: str, settings: dict) -> dict:
        return {
            "graphs": [
                {
                    "id": "Main",
                    "nodes": [
                        {
                            "id": "op_60",
                            "label": Path(model_path).stem,
                            "namespace": "graph",
                            "attrs": [{"key": "spv id", "value": "%60"}],
                        }
                    ],
                }
            ]
        }


class _OperatorIndexAdapter:
    """Fake TFLite adapter with native operator-index node IDs."""

    metadata = SimpleNamespace(fileExts=["tflite"])

    def convert(self, model_path: str, settings: dict) -> dict:
        return {
            "graphCollections": [
                {
                    "label": "TFLite",
                    "graphs": [
                        {
                            "id": "main",
                            "nodes": [
                                {
                                    "id": "0",
                                    "label": Path(model_path).stem,
                                    "namespace": "",
                                },
                                {
                                    "id": "1",
                                    "label": "consumer",
                                    "namespace": "",
                                    "incomingEdges": [{"sourceNodeId": "0"}],
                                },
                            ],
                        }
                    ],
                }
            ]
        }


class _TfliteOperatorIndexAdapter(_OperatorIndexAdapter):
    """Alias emphasizing that operator-index output is TFLite-specific."""


class _NamespaceMetricAdapter:
    """Fake TFLite adapter with a faithful nested namespace hierarchy."""

    metadata = SimpleNamespace(fileExts=["tflite"])

    def convert(self, model_path: str, settings: dict) -> dict:
        return {
            "graphs": [
                {
                    "id": "main",
                    "nodes": [
                        {
                            "id": "0",
                            "label": "conv",
                            "namespace": "stage/convolution",
                        },
                        {
                            "id": "1",
                            "label": "resize",
                            "namespace": "stage/resize",
                        },
                    ],
                }
            ]
        }


class _NamespaceTensorCollisionAdapter:
    """Fake TFLite graph whose constant tensor name matches its namespace."""

    metadata = SimpleNamespace(fileExts=["tflite"])

    def convert(self, model_path: str, settings: dict) -> dict:
        return {
            "graphs": [
                {
                    "id": "main",
                    "nodes": [
                        {
                            "id": "2",
                            "label": "pseudo_const",
                            "namespace": "stage/convolution",
                            "outputsMetadata": [
                                {
                                    "id": "0",
                                    "attrs": [
                                        {
                                            "key": "tensor_name",
                                            "value": "stage/convolution",
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "id": "0",
                            "label": "conv",
                            "namespace": "stage/convolution",
                            "incomingEdges": [{"sourceNodeId": "2"}],
                        },
                    ],
                }
            ]
        }


class _SingleGraphCollectionAdapter:
    """Fake adapter returning one graph collection with its own label."""

    metadata = SimpleNamespace(fileExts=["vgf"])

    def convert(self, model_path: str, settings: dict) -> dict:
        """Return one graph collection whose label MLIA should overwrite."""
        return {
            "graphCollections": [
                {
                    "label": "Adapter Label",
                    "graphs": [
                        {
                            "id": "main",
                            "nodes": [
                                {
                                    "id": "0",
                                    "label": Path(model_path).stem,
                                    "namespace": "",
                                }
                            ],
                        }
                    ],
                }
            ]
        }


class _MultipleGraphCollectionAdapter:
    """Fake adapter returning multiple graph collections."""

    metadata = SimpleNamespace(fileExts=["vgf"])

    def convert(self, model_path: str, settings: dict) -> dict:
        """Return multiple graph collections whose labels MLIA should overwrite."""
        return {
            "graphCollections": [
                {
                    "label": "Left Adapter Label",
                    "graphs": [{"id": "left", "nodes": [{"id": "0", "label": "left"}]}],
                },
                {
                    "label": "Right Adapter Label",
                    "graphs": [
                        {"id": "right", "nodes": [{"id": "1", "label": "right"}]}
                    ],
                },
            ]
        }


class _MatchingNamespaceAdapter:
    """Fake adapter whose source graph already has an entity namespace."""

    metadata = SimpleNamespace(fileExts=["vgf"])

    def convert(self, model_path: str, settings: dict) -> dict:
        """Return source namespaces that partially overlap entity namespaces."""
        return {
            "graphs": [
                {
                    "id": "main",
                    "nodes": [
                        {
                            "id": "node_0",
                            "label": "segment op",
                            "namespace": "segment 0",
                        },
                        {
                            "id": "node_1",
                            "label": "source only op",
                            "namespace": "source only",
                        },
                    ],
                }
            ]
        }


class _EscapedNamespaceAdapter:
    """Fake adapter with an escaped slash in a source namespace component."""

    metadata = SimpleNamespace(fileExts=["vgf"])

    def convert(self, model_path: str, settings: dict) -> dict:
        """Return a source namespace where escaped slash is not a separator."""
        return {
            "graphs": [
                {
                    "id": "main",
                    "nodes": [
                        {
                            "id": "node_0",
                            "label": "escaped",
                            "namespace": r"source\/part/child",
                        },
                    ],
                }
            ]
        }


def test_plugin_registers_with_registry() -> None:
    """Entry-point plugin registers an analysis plugin instance."""
    registry = AnalysisPluginRegistry()

    ModelExplorerAnalysisPlugin.register(registry)

    assert len(registry.plugins) == 1
    options = registry.cli_options()
    assert len(options) == 1
    assert isinstance(options[0], click.Option)
    assert options[0].opts == ["--model-explorer"]
    assert registry.enabled_plugins({"model_explorer": "files"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_json_artifact_writer_rejects_non_finite_numbers(
    tmp_path: Path, value: float
) -> None:
    """Artifact JSON must remain readable by strict JSON consumers."""
    output_path = tmp_path / "output.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        serialization_module._write_json(output_path, {"value": value})

    assert not output_path.exists()


def test_write_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Artifact generation writes manifest, output, mapping, and node data."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.tflite"))

    assert artifacts.manifest.exists()
    assert artifacts.standardized_output.exists()
    assert artifacts.sync_navigation.exists()
    assert artifacts.entity_node_mappings.exists()
    assert all(path.exists() for path in artifacts.node_data)
    assert not (artifacts.root / "models" / "model.tflite").exists()

    manifest = _read_json(artifacts.manifest)
    assert manifest["command"] == "check"
    assert manifest["model"]["name"] == "model.tflite"
    assert manifest["model_path"] == "models/model-source.json"
    assert manifest["standardized_output"] == "standardized-output.json"
    assert manifest["sync_navigation"] == "sync-navigation.json"
    assert manifest["entity_node_mappings"] == "entity-node-mappings.json"
    node = {"collectionLabel": "Source Model", "graphId": "main", "nodeId": "0"}
    assert _read_json(artifacts.entity_node_mappings) == {
        "results": [
            {
                "resultIndex": 0,
                "resultKind": "compatibility",
                "mappings": [
                    {
                        "node": node,
                        "entityId": "source_operator/operator/0",
                    }
                ],
            }
        ]
    }
    assert _read_json(artifacts.sync_navigation) == {
        "type": "sync_navigation",
        "graphMappings": [],
    }
    assert manifest["node_data"] == [
        {
            "path": "node-data/model-source/placement.json",
            "graph_collection_label": "Source Model",
        },
        {
            "path": "node-data/model-source/checks.json",
            "graph_collection_label": "Source Model",
        },
        {
            "path": "node-data/model-source/advice.json",
            "graph_collection_label": "Source Model",
        },
    ]
    assert manifest["nodeDataOverlayKeysAndDisplayNames"] == [
        {
            "overlayKey": {
                "resultIndex": 0,
                "resultKind": "compatibility",
                "category": "placement",
                "name": "placement",
                "unit": "",
                "breakdownQualifiers": {},
                "metricQualifiers": {},
            },
            "displayName": "placement",
        },
        {
            "overlayKey": {
                "resultIndex": 0,
                "resultKind": "compatibility",
                "category": "checks",
                "name": "checks",
                "unit": "",
                "breakdownQualifiers": {},
                "metricQualifiers": {},
            },
            "displayName": "checks",
        },
        {
            "overlayKey": {
                "resultIndex": 0,
                "resultKind": "compatibility",
                "category": "advice",
                "name": "advice",
                "unit": "",
                "breakdownQualifiers": {},
                "metricQualifiers": {},
            },
            "displayName": "advice",
        },
    ]
    assert "mappings" not in manifest
    assert not (artifacts.root / "mapping").exists()

    placement = _read_json(
        artifacts.root / "node-data" / "model-source" / "placement.json"
    )
    assert placement["main"]["name"] == "placement"
    assert placement["main"]["results"] == {"0": {"value": "NPU", "bgColor": "#2e7d32"}}

    checks = _read_json(artifacts.root / "node-data" / "model-source" / "checks.json")
    assert checks["main"]["name"] == "checks"
    assert checks["main"]["results"] == {
        "0": {"value": "missing quantization parameters", "bgColor": "#c62828"}
    }

    advice = _read_json(artifacts.root / "node-data" / "model-source" / "advice.json")
    assert advice["main"]["name"] == "advice"
    assert advice["main"]["results"] == {
        "0": {"value": "test advice", "bgColor": "#1565c0"}
    }


def test_write_artifacts_with_missing_standardized_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Artifact generation handles runs with no standardized output."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    artifacts = write_artifacts(
        _run_result(tmp_path, output=None, model_name="model.tflite")
    )

    output = _read_json(artifacts.standardized_output)
    assert output["results"] == []
    assert output["model"]["name"] == "model.tflite"
    assert artifacts.node_data == []


def test_write_artifacts_without_model_copy_when_model_missing(tmp_path: Path) -> None:
    """Missing model paths are tolerated and omitted from the manifest."""
    artifacts = write_artifacts(_run_result(tmp_path, model_exists=False))

    manifest = _read_json(artifacts.manifest)
    assert manifest["model_path"] is None
    assert not (artifacts.root / "models" / "model.bin").exists()


def test_write_artifacts_uses_materialized_output_model_without_cli_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A backend-materialized model is used when no CLI model was supplied."""
    fake_module = SimpleNamespace(SpirvAdapter=_StandaloneSpirvAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    result = _run_result(
        tmp_path,
        model_exists=False,
        model_name="captured.spv",
        cli_model=False,
    )
    model_data = b"captured SPIR-V"
    materialized = result.output_dir / "captured.spv"
    materialized.write_bytes(model_data)
    assert isinstance(result.output, dict)
    result.output["model"]["hash"] = hashlib.sha256(model_data).hexdigest()
    result.output["model"]["size_bytes"] = len(model_data)
    result.output["results"][0]["entities"].append(
        {
            "id": "source_operator/segment_0/spirv-60",
            "kind": "source_operator",
            "name": "captured operation",
        }
    )

    generated = write_artifacts(result)

    manifest = _read_json(generated.manifest)
    assert manifest["model_path"] == "models/captured-source.json"
    mappings = _read_json(generated.entity_node_mappings)
    assert any(
        mapping["entityId"] == "source_operator/segment_0/spirv-60"
        for mapping in mappings["results"][0]["mappings"]
    )


def test_write_artifacts_rejects_mismatched_materialized_output_model(
    tmp_path: Path,
) -> None:
    """Output-root model discovery authenticates the standardized model identity."""
    result = _run_result(
        tmp_path,
        model_exists=False,
        model_name="captured.spv",
        cli_model=False,
    )
    (result.output_dir / "captured.spv").write_bytes(b"wrong model")

    with pytest.raises(ConfigurationError, match="standardized-output hash"):
        write_artifacts(result)


def test_write_artifacts_skips_global_advice_without_affected_entity_ids(
    tmp_path: Path,
) -> None:
    """Global advice is kept in standardized output but not node data."""
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "compatibility",
                "advice": [
                    {
                        "id": "global-advice",
                        "category": "compatibility",
                        "severity": "info",
                        "message": "global advice",
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_exists=False)
    )

    assert artifacts.node_data == []
    assert not (
        artifacts.root / "node-data" / "model-source" / "compatibility.json"
    ).exists()


def test_write_artifacts_uses_entity_references_for_node_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Node-data overlays resolve checks, advice, and breakdowns through entities."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "source_operator/operator/1",
                        "name": "source-op",
                        "kind": "source_operator",
                        "status": "source-status",
                    },
                    {
                        "id": "subgraph-entity",
                        "name": "subgraph",
                        "kind": "chain",
                        "status": "subgraph-status",
                    },
                    {
                        "id": "model-entity",
                        "name": "model",
                        "kind": "model",
                        "status": "model-status",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/1",
                        "metrics": [{"name": "cycles", "value": 1, "unit": "cycles"}],
                    },
                    {
                        "entity_id": "subgraph-entity",
                        "metrics": [{"name": "cycles", "value": 100, "unit": "cycles"}],
                    },
                    {
                        "entity_id": "model-entity",
                        "metrics": [
                            {"name": "cycles", "value": 1000, "unit": "cycles"}
                        ],
                    },
                ],
                "checks": [
                    {
                        "id": "source-check",
                        "status": "fail",
                        "entity_id": "source_operator/operator/1",
                        "details": {"reasons": [{"description": "source check"}]},
                    },
                    {
                        "id": "subgraph-check",
                        "status": "fail",
                        "entity_id": "subgraph-entity",
                        "details": {"reasons": [{"description": "subgraph check"}]},
                    },
                ],
                "advice": [
                    {
                        "severity": "warning",
                        "message": "source advice",
                        "affected_entity_ids": [
                            "source_operator/operator/1",
                            "subgraph-entity",
                        ],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    assert [
        path.relative_to(artifacts.root / "node-data").as_posix()
        for path in artifacts.node_data
    ] == [
        "model-source/status.json",
        "model-hierarchy-segment_cascade_chain/status.json",
        "model-source/checks.json",
        "model-hierarchy-segment_cascade_chain/checks.json",
        "model-source/cycles.json",
        "model-hierarchy-segment_cascade_chain/cycles.json",
        "model-source/advice.json",
        "model-hierarchy-segment_cascade_chain/advice.json",
    ]
    status = _read_json(artifacts.root / "node-data" / "model-source" / "status.json")
    assert status["main"]["results"] == {
        "0": {"value": "subgraph-status"},
        "1": {"value": "source-status"},
    }
    cycles = _read_json(artifacts.root / "node-data" / "model-source" / "cycles.json")
    assert cycles["main"]["results"] == {"1": {"value": 1}}
    generated_cycles = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "cycles.json"
    )
    assert generated_cycles["segment_cascade_chain-main"]["results"] == {
        "1": {"value": 1},
        "subgraph": {"value": 100},
    }
    checks = _read_json(artifacts.root / "node-data" / "model-source" / "checks.json")
    assert checks["main"]["results"] == {
        "0": {"value": "subgraph check", "bgColor": "#c62828"},
        "1": {"value": "source check", "bgColor": "#c62828"},
    }
    advice = _read_json(artifacts.root / "node-data" / "model-source" / "advice.json")
    assert advice["main"]["results"] == {
        "0": {"value": "source advice", "bgColor": "#f9a825"},
        "1": {"value": "source advice", "bgColor": "#f9a825"},
    }


def test_write_artifacts_maps_each_source_node_to_its_source_operator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Source nodes map directly to source operators, not ancestor entities."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "combined-ops",
                        "name": "combined ops",
                        "kind": "chain",
                        "child_ids": [
                            "source_operator/operator/0",
                            "source_operator/operator/1",
                        ],
                    },
                    {"id": "source_operator/operator/0", "kind": "source_operator"},
                    {"id": "source_operator/operator/1", "kind": "source_operator"},
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    mappings = _read_json(artifacts.entity_node_mappings)["results"][0]["mappings"]
    assert mappings == [
        {
            "node": {
                "collectionLabel": "Source Model",
                "graphId": "main",
                "nodeId": "0",
            },
            "entityId": "source_operator/operator/0",
        },
        {
            "node": {
                "collectionLabel": "Source Model",
                "graphId": "main",
                "nodeId": "1",
            },
            "entityId": "source_operator/operator/1",
        },
    ]
    assert all(mapping["entityId"] != "combined-ops" for mapping in mappings)


def test_write_artifacts_groups_source_model_with_entity_namespaces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Entity kind hierarchy groups the cloned source graph."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "cascade-0",
                        "name": "cascade 0",
                        "kind": "cascade",
                        "child_ids": ["chain-0"],
                    },
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                        "parent_ids": ["cascade-0"],
                    },
                    {
                        "id": "source_operator/segment_0/spirv-0",
                        "name": "source op 0",
                        "kind": "source_operator",
                        "parent_ids": ["chain-0"],
                    },
                    {
                        "id": "chain-1",
                        "name": "chain 1",
                        "kind": "chain",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain-0",
                        "metrics": [{"name": "cycles", "value": 100, "unit": "cycles"}],
                    },
                    {
                        "entity_id": "cascade-0",
                        "metrics": [{"name": "cycles", "value": 10, "unit": "cycles"}],
                    },
                    {
                        "entity_id": "source_operator/segment_0/spirv-0",
                        "metrics": [{"name": "cycles", "value": 1, "unit": "cycles"}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert [path.name for path in artifacts.generated_models] == [
        "model-hierarchy-segment_cascade_chain.json"
    ]
    source_model = _read_json(artifacts.root / "models" / "model-source.json")
    assert (
        source_model["graphs"][0]["nodes"]
        == _FakeAdapter().convert(str(tmp_path / "model.vgf"), {})["graphs"][0]["nodes"]
    )
    sync_navigation = _read_json(artifacts.sync_navigation)
    assert {
        "from": {
            "collectionLabel": "segment_cascade_chain",
            "graphId": "segment_cascade_chain-Main",
            "nodeId": "cascade 0/chain 0___group___",
        },
        "to": {"Source Model": {"Main": ["0__op_0"]}},
    } in sync_navigation["graphMappings"]
    mappings = _read_json(artifacts.entity_node_mappings)["results"][0]["mappings"]
    assert {
        "node": {
            "collectionLabel": "segment_cascade_chain",
            "graphId": "segment_cascade_chain-Main",
            "nodeId": "cascade 0___group___",
        },
        "entityId": "cascade-0",
    } in mappings
    assert {
        "node": {
            "collectionLabel": "segment_cascade_chain",
            "graphId": "segment_cascade_chain-Main",
            "nodeId": "cascade 0/chain 0___group___",
        },
        "entityId": "chain-0",
    } in mappings
    assert {
        "node": {
            "collectionLabel": "Source Model",
            "graphId": "Main",
            "nodeId": "0__op_0",
        },
        "entityId": "source_operator/segment_0/spirv-0",
    } in mappings
    generated_source_operator_mappings = [
        mapping
        for mapping in mappings
        if mapping["node"]["collectionLabel"] == "segment_cascade_chain"
        and mapping["entityId"] == "source_operator/segment_0/spirv-0"
    ]
    assert len(generated_source_operator_mappings) == 2
    assert all(
        "__mlia_sub_operator__" in mapping["node"]["nodeId"]
        for mapping in generated_source_operator_mappings
    )
    generated_source_operator_node_ids = {
        mapping["node"]["nodeId"]
        for mapping in generated_source_operator_mappings
    }
    assert not any(
        mapping["node"]["collectionLabel"] == "Source Model"
        and mapping["entityId"] in {"cascade-0", "chain-0"}
        for mapping in mappings
    )
    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    assert entity_model["label"] == "segment_cascade_chain"
    assert entity_model["graphs"][0]["id"] == "segment_cascade_chain-Main"
    assert [node["namespace"] for node in entity_model["graphs"][0]["nodes"]] == [
        "cascade 0/chain 0",
        "chain 1",
        "",
    ]
    assert [node["label"] for node in entity_model["graphs"][0]["nodes"]] == [
        "model (#1)",
        "model (#2)",
        "consumer",
    ]
    source_cycles = _read_json(
        artifacts.root / "node-data" / "model-source" / "cycles.json"
    )
    assert source_cycles["Main"]["results"]["0__op_0"] == {"value": 1}

    cycles = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "cycles.json"
    )
    generated_cycle_results = cycles["segment_cascade_chain-Main"]["results"]
    assert generated_cycle_results == {
        "cascade 0": {"value": 10},
        "cascade 0/chain 0": {"value": 100},
    }
    assert generated_source_operator_node_ids.isdisjoint(generated_cycle_results)

    manifest = _read_json(artifacts.manifest)
    assert manifest["model_path"] == "models/model-source.json"
    assert manifest["model_paths"] == [
        "models/model-source.json",
        "models/model-hierarchy-segment_cascade_chain.json",
    ]
    assert manifest["generated_models"] == [
        "models/model-hierarchy-segment_cascade_chain.json"
    ]


def test_write_artifacts_preserves_escaped_source_namespace_mapping_prefixes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Escaped slashes in source namespaces do not split mapping prefixes."""
    fake_module = SimpleNamespace(VGFAdapter=_EscapedNamespaceAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )

    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.vgf"))

    assert _read_json(artifacts.sync_navigation) == {
        "type": "sync_navigation",
        "graphMappings": [],
    }


def test_write_artifacts_omits_leaf_mappings_for_matching_namespace_groups(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Same namespace groups on both sides rely on Model Explorer ID fallback."""
    fake_module = SimpleNamespace(VGFAdapter=_MatchingNamespaceAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "segment-0",
                        "name": "segment 0",
                        "kind": "segment",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "segment-0",
                        "metrics": [{"name": "cycles", "value": 1}],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert _read_json(artifacts.sync_navigation) == {
        "type": "sync_navigation",
        "graphMappings": [],
    }


def test_write_artifacts_keeps_statsless_parent_subgraphs_in_entity_namespaces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Visual namespaces include all parent subgraph entity levels."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "segment-0",
                        "name": "segment 0",
                        "kind": "segment",
                    },
                    {
                        "id": "cascade-0",
                        "name": "cascade 0",
                        "kind": "cascade",
                        "parent_ids": ["segment-0"],
                    },
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                        "parent_ids": ["cascade-0"],
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "cascade-0",
                        "metrics": [{"name": "cycles", "value": 10}],
                    },
                    {
                        "entity_id": "chain-0",
                        "metrics": [{"name": "cycles", "value": 100}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    assert (
        entity_model["graphs"][0]["nodes"][0]["namespace"]
        == "segment 0/cascade 0/chain 0"
    )
    cycles = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "cycles.json"
    )
    assert cycles["segment_cascade_chain-Main"]["results"] == {
        "segment 0/cascade 0": {"value": 10},
        "segment 0/cascade 0/chain 0": {"value": 100},
    }


def test_complete_entity_paths_strip_visible_parent_name_prefixes() -> None:
    """Child labels do not repeat a visible parent segment name."""
    entities = {
        "segment-10": {
            "id": "segment-10",
            "name": "graph_segment_10",
            "kind": "segment",
        },
        "cascade-1": {
            "id": "cascade-1",
            "name": "graph_segment_10/Cascade 1",
            "kind": "cascade",
            "parent_ids": ["segment-10"],
        },
        "chain-2": {
            "id": "chain-2",
            "name": "graph_segment_10/Chain 2",
            "kind": "chain",
            "parent_ids": ["cascade-1"],
        },
    }
    hierarchy = _test_hierarchy("segment_cascade_chain")

    projection = graphs_module._entity_hierarchy_complete_paths(
        entities, hierarchy, set()
    )

    assert projection.entity_paths_by_id == {
        "segment-10": ("graph_segment_10",),
        "cascade-1": ("graph_segment_10/Cascade 1",),
        "chain-2": ("graph_segment_10/Cascade 1/Chain 2",),
    }


def test_complete_entity_paths_keep_hidden_parent_context(monkeypatch) -> None:
    """Hidden parent names remain as context on the first visible child."""
    monkeypatch.setattr(
        graphs_module,
        "_ONLY_SHOW_STATS_BEARING_ENTITY_NAMESPACES",
        True,
    )
    entities = {
        "segment-10": {
            "id": "segment-10",
            "name": "graph_segment_10",
            "kind": "segment",
        },
        "cascade-1": {
            "id": "cascade-1",
            "name": "graph_segment_10/Cascade 1",
            "kind": "cascade",
            "parent_ids": ["segment-10"],
        },
        "chain-2": {
            "id": "chain-2",
            "name": "graph_segment_10/Chain 2",
            "kind": "chain",
            "parent_ids": ["cascade-1"],
        },
    }
    hierarchy = _test_hierarchy("segment_cascade_chain")

    projection = graphs_module._entity_hierarchy_complete_paths(
        entities,
        hierarchy,
        {"cascade-1", "chain-2"},
    )

    assert projection.entity_paths_by_id == {
        "cascade-1": (r"graph_segment_10\/Cascade 1",),
        "chain-2": (r"graph_segment_10\/Cascade 1/Chain 2",),
    }


def test_complete_entity_paths_are_unique_and_deterministic() -> None:
    """Normalized collisions are disambiguated independently of entity order."""
    entities = {
        "parent-1": {"id": "parent-1", "name": "parent1", "kind": "cascade"},
        "parent-2": {"id": "parent-2", "name": "parent2", "kind": "cascade"},
        "child/1": {
            "id": "child/1",
            "name": "x;shared",
            "kind": "chain",
            "parent_ids": ["parent-1"],
        },
        "child;2": {
            "id": "child;2",
            "name": "y;shared",
            "kind": "chain",
            "parent_ids": ["parent-2"],
        },
    }
    hierarchy = _test_hierarchy("segment_cascade_chain")

    first = graphs_module._entity_hierarchy_complete_paths(
        entities, hierarchy, set()
    )
    reordered = graphs_module._entity_hierarchy_complete_paths(
        dict(reversed(list(entities.items()))), hierarchy, set()
    )

    assert first == reordered
    all_paths = [
        path
        for paths in first.entity_paths_by_id.values()
        for path in paths
    ]
    normalized = [
        model_explorer_normalized_namespace(path)
        for path in all_paths
    ]
    assert all(normalized)
    assert len(normalized) == len(set(normalized))


def test_complete_entity_paths_enumerate_multi_parent_paths() -> None:
    """Backend entities and source operators retain every complete ancestry path."""
    entities = {
        "cascade-a": {"id": "cascade-a", "name": "Cascade A", "kind": "cascade"},
        "cascade-b": {"id": "cascade-b", "name": "Cascade B", "kind": "cascade"},
        "chain": {
            "id": "chain",
            "name": "Chain",
            "kind": "chain",
            "parent_ids": ["cascade-b", "cascade-a"],
        },
        "op": {
            "id": "op",
            "kind": "source_operator",
            "parent_ids": ["chain"],
        },
    }
    hierarchy = _test_hierarchy("segment_cascade_chain")

    projection = graphs_module._entity_hierarchy_complete_paths(
        entities, hierarchy, set()
    )

    assert projection.entity_paths_by_id["chain"] == (
        "Cascade A/Chain",
        "Cascade B/Chain",
    )
    assert projection.source_operator_paths_by_id["op"] == (
        "Cascade A/Chain",
        "Cascade B/Chain",
    )


def test_write_artifacts_disambiguates_duplicate_entity_namespace_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Duplicate entity names remain separate namespace groups and node-data keys."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "cascade-a",
                        "name": "duplicate",
                        "kind": "cascade",
                    },
                    {
                        "id": "cascade-b",
                        "name": "duplicate",
                        "kind": "cascade",
                    },
                    {
                        "id": "chain-a",
                        "name": "op",
                        "kind": "chain",
                        "parent_ids": ["cascade-a"],
                    },
                    {
                        "id": "chain-b",
                        "name": "op",
                        "kind": "chain",
                        "parent_ids": ["cascade-b"],
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "cascade-a",
                        "metrics": [{"name": "cycles", "value": 10}],
                    },
                    {
                        "entity_id": "cascade-b",
                        "metrics": [{"name": "cycles", "value": 20}],
                    },
                    {
                        "entity_id": "chain-a",
                        "metrics": [{"name": "cycles", "value": 100}],
                    },
                    {
                        "entity_id": "chain-b",
                        "metrics": [{"name": "cycles", "value": 200}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    assert [node["namespace"] for node in entity_model["graphs"][0]["nodes"]] == [
        "duplicate [cascade-a]/op [chain-a]",
        "duplicate [cascade-b]/op [chain-b]",
    ]
    cycles = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "cycles.json"
    )
    assert cycles["segment_cascade_chain-Main"]["results"] == {
        "duplicate [cascade-a]": {"value": 10},
        "duplicate [cascade-a]/op [chain-a]": {"value": 100},
        "duplicate [cascade-b]": {"value": 20},
        "duplicate [cascade-b]/op [chain-b]": {"value": 200},
    }


def test_write_artifacts_escapes_entity_name_namespace_separators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Entity names with slashes remain one visual namespace component."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "cascade-0",
                        "name": "cascade/0",
                        "kind": "cascade",
                    },
                    {
                        "id": "chain-0",
                        "name": "chain/0",
                        "kind": "chain",
                        "parent_ids": ["cascade-0"],
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "cascade-0",
                        "metrics": [{"name": "cycles", "value": 10}],
                    },
                    {
                        "entity_id": "chain-0",
                        "metrics": [{"name": "cycles", "value": 100}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    assert entity_model["graphs"][0]["nodes"][0]["namespace"] == r"cascade\/0/chain\/0"
    sync_navigation = _read_json(artifacts.sync_navigation)
    generated_group_ids = {
        entry["from"]["nodeId"]
        for entry in sync_navigation["graphMappings"]
        if entry["from"]["collectionLabel"] == "segment_cascade_chain"
    }
    assert generated_group_ids == {
        "0__op_0",
        "0__op_1",
        r"cascade\/0___group___",
        r"cascade\/0/chain\/0___group___",
    }
    cycles = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "cycles.json"
    )
    assert cycles["segment_cascade_chain-Main"]["results"] == {
        r"cascade\/0": {"value": 10},
        r"cascade\/0/chain\/0": {"value": 100},
    }


def test_write_artifacts_keeps_parent_subgraphs_result_local(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Each result gets its own hierarchy graph and metric namespace state."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    entities = [
        {
            "id": "segment-0",
            "name": "segment 0",
            "kind": "segment",
        },
        {
            "id": "cascade-0",
            "name": "cascade 0",
            "kind": "cascade",
            "parent_ids": ["segment-0"],
        },
        {
            "id": "chain-0",
            "name": "chain 0",
            "kind": "chain",
            "parent_ids": ["cascade-0"],
        },
    ]
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": entities,
                "breakdowns": [
                    {
                        "entity_id": "cascade-0",
                        "metrics": [{"name": "cycles", "value": 10}],
                    },
                    {
                        "entity_id": "chain-0",
                        "metrics": [{"name": "cycles", "value": 100}],
                    },
                ],
            },
            {
                "kind": "memory",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": entities,
                "breakdowns": [
                    {
                        "entity_id": "segment-0",
                        "metrics": [{"name": "bytes", "value": 1000}],
                    }
                ],
            },
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert [path.name for path in artifacts.generated_models] == [
        "model-hierarchy-segment_cascade_chain-result-1.json",
        "model-hierarchy-segment_cascade_chain-result-2.json",
    ]
    performance_model = _read_json(
        artifacts.root
        / "models"
        / "model-hierarchy-segment_cascade_chain-result-1.json"
    )
    memory_model = _read_json(
        artifacts.root
        / "models"
        / "model-hierarchy-segment_cascade_chain-result-2.json"
    )
    assert performance_model["label"] == (
        "segment_cascade_chain [performance result 1]"
    )
    assert memory_model["label"] == "segment_cascade_chain [memory result 2]"
    assert performance_model["graphs"][0]["nodes"][0]["namespace"] == (
        "segment 0/cascade 0/chain 0"
    )
    assert memory_model["graphs"][0]["nodes"][0]["namespace"] == (
        "segment 0/cascade 0/chain 0"
    )
    cycles = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain-result-1"
        / "cycles.json"
    )
    assert cycles["segment_cascade_chain-result-1-Main"]["results"] == {
        "segment 0/cascade 0": {"value": 10},
        "segment 0/cascade 0/chain 0": {"value": 100},
    }
    assert "segment_cascade_chain-result-2-Main" not in cycles
    memory = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain-result-2"
        / "bytes.json"
    )
    assert memory["segment_cascade_chain-result-2-Main"]["results"] == {
        "segment 0": {"value": 1000}
    }
    assert "segment_cascade_chain-result-1-Main" not in memory


def test_write_artifacts_uses_builtin_source_operator_metadata_relationship(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Declared built-in source_operator metadata still drives hierarchy tracing."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": [
                    {"id": "segment", "child_kinds": ["cascade"]},
                    {
                        "id": "cascade",
                        "parent_kinds": ["segment"],
                        "child_kinds": ["chain"],
                    },
                    {"id": "chain", "parent_kinds": ["cascade"]},
                    {"id": "source_operator", "parent_kinds": ["chain"]},
                ],
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert [path.name for path in artifacts.generated_models] == [
        "model-hierarchy-segment_cascade_chain.json"
    ]
    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    assert entity_model["graphs"][0]["nodes"][0]["namespace"] == "chain 0"


def test_write_artifacts_uses_parent_ids_as_canonical_visual_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Entity parent_ids win over conflicting child_ids."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "cascade-a",
                        "name": "cascade A",
                        "kind": "cascade",
                        "child_ids": ["chain-conflict", "chain-orphan"],
                    },
                    {
                        "id": "cascade-b",
                        "name": "cascade B",
                        "kind": "cascade",
                    },
                    {
                        "id": "chain-conflict",
                        "name": "chain conflict",
                        "kind": "chain",
                        "parent_ids": ["cascade-b"],
                    },
                    {
                        "id": "chain-orphan",
                        "name": "chain orphan",
                        "kind": "chain",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "cascade-b",
                        "metrics": [{"name": "cycles", "value": 1}],
                    },
                    {
                        "entity_id": "chain-conflict",
                        "metrics": [{"name": "cycles", "value": 2}],
                    },
                    {
                        "entity_id": "chain-orphan",
                        "metrics": [{"name": "cycles", "value": 3}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    nodes = entity_model["graphs"][0]["nodes"]
    assert nodes[0]["namespace"] == "cascade B/chain conflict"
    assert nodes[1]["namespace"] == "cascade A/chain orphan"


def test_write_artifacts_uses_child_ids_when_parent_ids_are_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Entity child_ids infer hierarchy parents when child parent_ids are absent."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "segment-0",
                        "name": "segment 0",
                        "kind": "segment",
                        "child_ids": ["cascade-0"],
                    },
                    {
                        "id": "cascade-0",
                        "name": "cascade 0",
                        "kind": "cascade",
                        "child_ids": ["chain-0"],
                    },
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain-0",
                        "metrics": [{"name": "cycles", "value": 1}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    entity_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-segment_cascade_chain.json"
    )
    assert (
        entity_model["graphs"][0]["nodes"][0]["namespace"]
        == "segment 0/cascade 0/chain 0"
    )


def test_write_artifacts_emits_separate_overlays_per_result_field(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Common result fields are emitted as focused node-data files."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "op",
                        "kind": "source_operator",
                        "status": "estimated",
                    },
                    {
                        "id": "source_operator/operator/1",
                        "name": "op-extra",
                        "kind": "source_operator",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "metrics": [
                            {"name": "npu_cycles", "value": 123, "unit": "cycles"},
                            {"name": "sram", "value": 456, "unit": "bytes"},
                        ],
                    },
                    {
                        "entity_id": "source_operator/operator/1",
                        "metrics": [
                            {"name": "npu_cycles", "value": 7, "unit": "cycles"},
                            {"name": "sram", "value": "extra", "unit": "bytes"},
                        ],
                    },
                ],
                "advice": [
                    {
                        "severity": "warning",
                        "message": "try another layout",
                        "affected_entity_ids": ["source_operator/operator/0"],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    assert [path.name for path in artifacts.node_data] == [
        "status.json",
        "npu_cycles.json",
        "sram (bytes).json",
        "advice.json",
    ]

    status = _read_json(artifacts.root / "node-data" / "model-source" / "status.json")
    assert status["main"]["results"] == {"0": {"value": "estimated"}}

    npu_cycles = _read_json(
        artifacts.root / "node-data" / "model-source" / "npu_cycles.json"
    )
    assert npu_cycles["main"]["name"] == "npu_cycles"
    assert npu_cycles["main"]["gradient"] == [
        {"stop": 0, "bgColor": "#2e7d32"},
        {"stop": 1, "bgColor": "#c62828"},
    ]
    assert npu_cycles["main"]["results"] == {
        "0": {"value": 123},
        "1": {"value": 7},
    }

    sram = _read_json(
        artifacts.root / "node-data" / "model-source" / "sram (bytes).json"
    )
    assert sram["main"]["name"] == "sram (bytes)"
    assert sram["main"]["results"] == {
        "0": {"value": 456},
        "1": {"value": "extra"},
    }

    advice = _read_json(artifacts.root / "node-data" / "model-source" / "advice.json")
    assert advice["main"]["name"] == "advice"
    assert advice["main"]["results"] == {
        "0": {"value": "try another layout", "bgColor": "#f9a825"}
    }


def _metric_overlay_output(breakdowns: list[dict]) -> dict:
    """Return minimal standardized output for production-path metric overlay tests."""
    return {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "op",
                        "kind": "source_operator",
                    }
                ],
                "breakdowns": breakdowns,
            }
        ],
    }


def _write_metric_overlay_artifacts(
    tmp_path: Path, monkeypatch, breakdowns: list[dict]
):
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    return write_artifacts(
        _run_result(
            tmp_path,
            output=_metric_overlay_output(breakdowns),
            model_name="model.tflite",
        )
    )


def test_source_namespace_metrics_are_projected_through_faithful_hierarchy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Core projection adds source namespace group metrics without flattening."""
    fake_module = SimpleNamespace(TfliteAdapter=_NamespaceMetricAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "conv",
                        "kind": "source_operator",
                    },
                    {
                        "id": "source_operator/operator/1",
                        "name": "resize",
                        "kind": "source_operator",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "metrics": [
                            {
                                "name": "cycles",
                                "value": 10,
                                "aggregation": "sum",
                            }
                        ],
                    },
                    {
                        "entity_id": "source_operator/operator/1",
                        "metrics": [
                            {
                                "name": "cycles",
                                "value": 20,
                                "aggregation": "sum",
                            }
                        ],
                    },
                ],
            }
        ],
    }

    generated = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    cycles = _read_json(
        generated.root / "node-data" / "model-source" / "cycles.json"
    )
    assert cycles["main"]["results"] == {
        "0": {"value": 10},
        "1": {"value": 20},
        "stage___group___": {"value": 30},
        "stage/convolution___group___": {"value": 10},
        "stage/resize___group___": {"value": 20},
    }


def test_source_namespace_metrics_use_explicit_group_ids_when_tensor_names_collide(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Namespace metrics cannot resolve to constants with matching tensor names."""
    fake_module = SimpleNamespace(TfliteAdapter=_NamespaceTensorCollisionAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = _metric_overlay_output(
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {
                        "name": "cycles",
                        "value": 10,
                        "aggregation": "sum",
                    }
                ],
            }
        ]
    )

    generated = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    cycles = _read_json(
        generated.root / "node-data" / "model-source" / "cycles.json"
    )
    assert cycles["main"]["results"] == {
        "0": {"value": 10},
        "stage___group___": {"value": 10},
        "stage/convolution___group___": {"value": 10},
    }


def test_source_namespace_metrics_copy_complete_parent_entity_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A source namespace may copy a complete parent without leaf statistics."""
    fake_module = SimpleNamespace(TfliteAdapter=_NamespaceMetricAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": [
                    {"id": "chain", "child_kinds": ["source_operator"]}
                ],
                "entities": [
                    {
                        "id": "chain/0",
                        "name": "chain",
                        "kind": "chain",
                        "child_ids": [
                            "source_operator/operator/0",
                            "source_operator/operator/1",
                        ],
                    },
                    {
                        "id": "source_operator/operator/0",
                        "name": "conv",
                        "kind": "source_operator",
                        "parent_ids": ["chain/0"],
                    },
                    {
                        "id": "source_operator/operator/1",
                        "name": "resize",
                        "kind": "source_operator",
                        "parent_ids": ["chain/0"],
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain/0",
                        "metrics": [
                            {
                                "name": "cycles",
                                "value": 30,
                                "unit": "cycles",
                                "aggregation": "sum",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    generated = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    cycles = _read_json(
        generated.root / "node-data" / "model-source" / "cycles.json"
    )
    assert cycles["main"]["results"] == {
        "stage___group___": {"value": 30}
    }


def test_metric_qualifiers_create_distinct_overlays_without_merging_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Metric qualifier variants retain separate logical overlays and values."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "op",
                        "kind": "source_operator",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "metrics": [
                            {
                                "name": "latency",
                                "value": 20,
                                "unit": "ms",
                                "qualifiers": {"engine": "cpu"},
                            },
                            {
                                "name": "latency",
                                "value": 10,
                                "unit": "ms",
                                "qualifiers": {"engine": "npu"},
                            },
                        ],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    assert [path.name for path in artifacts.node_data] == [
        "latency (ms) [engine=cpu].json",
        "latency (ms) [engine=npu].json",
    ]
    assert [_read_json(path)["main"]["name"] for path in artifacts.node_data] == [
        "latency (ms) [engine=cpu]",
        "latency (ms) [engine=npu]",
    ]
    assert [_read_json(path)["main"]["results"] for path in artifacts.node_data] == [
        {"0": {"value": 20}},
        {"0": {"value": 10}},
    ]
    assert _read_json(artifacts.manifest)["nodeDataOverlayKeysAndDisplayNames"] == [
        {
            "overlayKey": {
                "resultIndex": 0,
                "resultKind": "performance",
                "category": "metric",
                "name": "latency",
                "unit": "ms",
                "breakdownQualifiers": {},
                "metricQualifiers": {"engine": "cpu"},
            },
            "displayName": "latency (ms) [engine=cpu]",
        },
        {
            "overlayKey": {
                "resultIndex": 0,
                "resultKind": "performance",
                "category": "metric",
                "name": "latency",
                "unit": "ms",
                "breakdownQualifiers": {},
                "metricQualifiers": {"engine": "npu"},
            },
            "displayName": "latency (ms) [engine=npu]",
        },
    ]


def test_breakdown_qualifiers_create_distinct_overlays_without_merging_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Breakdown qualifier variants retain separate logical overlays and values."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "op",
                        "kind": "source_operator",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "qualifiers": {"phase": "warm"},
                        "metrics": [{"name": "cycles", "value": 10}],
                    },
                    {
                        "entity_id": "source_operator/operator/0",
                        "qualifiers": {"phase": "cold"},
                        "metrics": [{"name": "cycles", "value": 20}],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    assert [path.name for path in artifacts.node_data] == [
        "cycles [phase=cold].json",
        "cycles [phase=warm].json",
    ]
    assert [_read_json(path)["main"]["results"] for path in artifacts.node_data] == [
        {"0": {"value": 20}},
        {"0": {"value": 10}},
    ]


def test_breakdown_and_metric_qualifier_scopes_remain_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Equal qualifier content at different scopes remains distinct and named."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "op",
                        "kind": "source_operator",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "qualifiers": {"phase": "warm"},
                        "metrics": [{"name": "cycles", "value": 10}],
                    },
                    {
                        "entity_id": "source_operator/operator/0",
                        "metrics": [
                            {
                                "name": "cycles",
                                "value": 20,
                                "qualifiers": {"phase": "warm"},
                            }
                        ],
                    },
                    {
                        "entity_id": "source_operator/operator/0",
                        "qualifiers": {"phase": "warm"},
                        "metrics": [
                            {
                                "name": "cycles",
                                "value": 30,
                                "qualifiers": {"phase": "warm"},
                            }
                        ],
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    assert [path.name for path in artifacts.node_data] == [
        "cycles [metric phase=warm].json",
        "cycles [breakdown phase=warm].json",
        "cycles [breakdown phase=warm; metric phase=warm].json",
    ]
    assert [_read_json(path)["main"]["results"] for path in artifacts.node_data] == [
        {"0": {"value": 20}},
        {"0": {"value": 10}},
        {"0": {"value": 30}},
    ]


def test_qualifier_canonicalization_preserves_json_types_and_object_order() -> None:
    """Canonical qualifiers are hashable, typed, ordered, and presentation-safe."""
    first = node_data_module._canonical_qualifiers(
        {
            "optional": None,
            "config": {
                "shape": [1, 2.5, True, "2", None],
                "enabled": False,
            },
        }
    )
    reordered = node_data_module._canonical_qualifiers(
        {
            "config": {
                "enabled": False,
                "shape": [1, 2.5, True, "2", None],
            },
            "optional": None,
        }
    )

    assert first is not None
    assert reordered is not None
    assert first == reordered
    assert hash(first) == hash(reordered)
    assert node_data_module._qualifier_summary(first) == (
        'config={"enabled":false,"shape":[1,2.5,true,"2",null]}; optional=null'
    )
    assert (
        len(
            {
                node_data_module._canonical_qualifiers({"value": None}),
                node_data_module._canonical_qualifiers({"value": False}),
                node_data_module._canonical_qualifiers({"value": 0}),
                node_data_module._canonical_qualifiers({"value": 0.0}),
                node_data_module._canonical_qualifiers({"value": "0"}),
                node_data_module._canonical_qualifiers({"value": [0]}),
                node_data_module._canonical_qualifiers({"value": {"item": 0}}),
            }
        )
        == 7
    )
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    assert (
        node_data_module._canonical_qualifiers(
            {"kept": "value", "invalid": object(), "not_finite": float("nan")}
        )
        is None
    )
    assert node_data_module._canonical_qualifiers({"cycle": cycle}) is None
    assert node_data_module._canonical_qualifiers({1: "not a string key"}) is None
    assert node_data_module._canonical_qualifiers(["not", "an", "object"]) is None


def test_overlay_key_json_restores_complete_typed_qualifiers() -> None:
    """Manifest overlay keys expose canonical qualifier identity as typed JSON."""
    qualifiers = node_data_module._canonical_qualifiers(
        {
            "null": None,
            "boolean": True,
            "integer": 1,
            "float": 1.0,
            "string": "value",
            "array": [0, False],
            "object": {"nested": "value"},
        }
    )
    assert qualifiers is not None

    serialized = serialization_module._overlay_key_json(
        artifact_models._OverlayKey(
            2,
            "performance",
            "metric",
            "latency",
            "ms",
            metric_qualifiers=qualifiers,
        )
    )

    assert serialized == {
        "resultIndex": 2,
        "resultKind": "performance",
        "category": "metric",
        "name": "latency",
        "unit": "ms",
        "breakdownQualifiers": {},
        "metricQualifiers": {
            "array": [0, False],
            "boolean": True,
            "float": 1.0,
            "integer": 1,
            "null": None,
            "object": {"nested": "value"},
            "string": "value",
        },
    }
    assert type(serialized["metricQualifiers"]["integer"]) is int
    assert type(serialized["metricQualifiers"]["float"]) is float


def test_long_qualifier_naming_is_bounded_and_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Long valid qualifiers produce stable readable names and safe components."""
    breakdowns = [
        {
            "entity_id": "source_operator/operator/0",
            "metrics": [
                {
                    "name": "latency",
                    "unit": "ms",
                    "value": 7,
                    "qualifiers": {"engine": "n" * 300},
                }
            ],
        }
    ]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _write_metric_overlay_artifacts(first_dir, monkeypatch, breakdowns)
    second = _write_metric_overlay_artifacts(second_dir, monkeypatch, breakdowns)

    assert [path.name for path in first.node_data] == [
        path.name for path in second.node_data
    ]
    display_name = _read_json(first.node_data[0])["main"]["name"]
    assert display_name.startswith("latency (ms) [engine=")
    assert "n" * 300 not in display_name
    digest = display_name.rsplit("…#", 1)[1].removesuffix("]")
    assert len(digest) == node_data_module._STABLE_DIGEST_HEX_LENGTH
    assert all(character in "0123456789abcdef" for character in digest)
    assert len(first.node_data[0].name.encode("utf-8")) <= (
        serialization_module._MAX_FILENAME_COMPONENT_BYTES
    )
    assert _read_json(first.node_data[0])["main"]["results"] == {"0": {"value": 7}}


def test_large_nested_qualifiers_generate_with_bounded_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Large nested JSON qualifier identities remain safe on the production path."""
    nested = {
        "enabled": True,
        "layers": [
            {"name": f"layer-{index}", "shape": [index, index + 1, index + 2]}
            for index in range(60)
        ],
    }
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {
                        "name": "cycles",
                        "value": 11,
                        "qualifiers": {"config": nested},
                    }
                ],
            }
        ],
    )

    graph = _read_json(artifacts.node_data[0])["main"]
    assert graph["name"].startswith('cycles [config={"enabled":true,"layers":[')
    assert "…#" in graph["name"]
    assert len(artifacts.node_data[0].name.encode("utf-8")) <= (
        serialization_module._MAX_FILENAME_COMPONENT_BYTES
    )
    assert graph["results"] == {"0": {"value": 11}}


def test_long_qualifiers_with_identical_prefixes_remain_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Full canonical qualifier identity, not its visible prefix, drives naming."""
    common_prefix = "accelerator-" + "x" * 300
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {
                        "name": "latency",
                        "value": 1,
                        "qualifiers": {"engine": f"{common_prefix}-alpha"},
                    },
                    {
                        "name": "latency",
                        "value": 2,
                        "qualifiers": {"engine": f"{common_prefix}-beta"},
                    },
                ],
            }
        ],
    )

    graphs = [_read_json(path)["main"] for path in artifacts.node_data]
    display_names = [graph["name"] for graph in graphs]
    assert len(set(display_names)) == 2
    assert display_names[0].split("…#", 1)[0] == display_names[1].split("…#", 1)[0]
    assert len({path.name.casefold() for path in artifacts.node_data}) == 2
    assert [graph["results"] for graph in graphs] == [
        {"0": {"value": 1}},
        {"0": {"value": 2}},
    ]


def test_qualifier_filename_normalization_collisions_are_allocated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Distinct qualifier displays survive normalization to the same safe stem."""
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {
                        "name": "latency",
                        "value": 1,
                        "qualifiers": {"engine": "a/b"},
                    },
                    {
                        "name": "latency",
                        "value": 2,
                        "qualifiers": {"engine": "a:b"},
                    },
                ],
            }
        ],
    )

    graphs = [_read_json(path)["main"] for path in artifacts.node_data]
    assert graphs[0]["name"] != graphs[1]["name"]
    assert artifacts.node_data[1].stem == f"{artifacts.node_data[0].stem} (2)"
    assert [graph["results"] for graph in graphs] == [
        {"0": {"value": 1}},
        {"0": {"value": 2}},
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("COM¹", "_COM¹"),
        ("cOm¹", "_cOm¹"),
        ("COM²", "_COM²"),
        ("com²", "_com²"),
        ("COM³", "_COM³"),
        ("CoM³", "_CoM³"),
        ("LPT¹", "_LPT¹"),
        ("lPt¹", "_lPt¹"),
        ("LPT²", "_LPT²"),
        ("lpt²", "_lpt²"),
        ("LPT³", "_LPT³"),
        ("LpT³", "_LpT³"),
        ("COM¹.results", "_COM¹.results"),
        ("lPt².data", "_lPt².data"),
    ],
)
def test_safe_filename_escapes_windows_superscript_device_names(
    value: str, expected: str
) -> None:
    """Windows superscript device aliases are unsafe with any case or extension."""
    assert serialization_module._safe_filename(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CON", "_CON"),
        ("prn.txt", "_prn.txt"),
        ("COM9", "_COM9"),
        ("lPt1.log", "_lPt1.log"),
        ("COM0", "COM0"),
        ("COM⁴", "COM⁴"),
        ("COM¹x", "COM¹x"),
        ("XLPT²", "XLPT²"),
    ],
)
def test_safe_filename_keeps_existing_reserved_and_lookalike_behavior(
    value: str, expected: str
) -> None:
    """The superscript fix does not broaden matching beyond device-name stems."""
    assert serialization_module._safe_filename(value) == expected


def test_superscript_device_names_get_safe_resolved_artifact_filenames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Production writes escape reserved stems before existing and JSON extensions."""
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {"name": "COM¹.results", "value": 1},
                    {"name": "lPt³.data", "value": 2},
                ],
            }
        ],
    )

    assert [path.resolve().name for path in artifacts.node_data] == [
        "_COM¹.results.json",
        "_lPt³.data.json",
    ]
    assert [_read_json(path)["main"]["name"] for path in artifacts.node_data] == [
        "COM¹.results",
        "lPt³.data",
    ]


def test_safe_filename_truncates_at_a_multibyte_utf8_boundary() -> None:
    """Drop partial code points while keeping filename truncation within budget."""
    digest_suffix_bytes = len(
        ("--" + "0" * node_data_module._STABLE_DIGEST_HEX_LENGTH).encode("utf-8")
    )
    prefix_budget = serialization_module._MAX_FILENAME_STEM_BYTES - digest_suffix_bytes
    value = "a" * (prefix_budget - 1) + "😀" + "tail" * 10

    safe = serialization_module._safe_filename(value)
    encoded = safe.encode("utf-8")

    assert safe.startswith("a" * (prefix_budget - 1) + "--")
    assert "😀" not in safe
    assert encoded.decode("utf-8") == safe
    assert len(encoded) <= serialization_module._MAX_FILENAME_STEM_BYTES


def test_filename_collision_suffix_is_allocated_near_the_byte_limit(
    monkeypatch,
) -> None:
    """Collision suffixes fit by truncating a stem already at the component limit."""
    extension = serialization_module._NODE_DATA_FILENAME_EXTENSION
    max_stem_bytes = serialization_module._MAX_FILENAME_COMPONENT_BYTES - len(
        extension.encode("utf-8")
    )
    monkeypatch.setattr(
        serialization_module, "_MAX_FILENAME_STEM_BYTES", max_stem_bytes
    )
    value = "x" * max_stem_bytes
    allocated: set[str] = set()

    first = serialization_module._allocate_safe_filename(value, allocated)
    second = serialization_module._allocate_safe_filename(value, allocated)

    assert first == value
    assert second == f"{'x' * (max_stem_bytes - len(' (2)'))} (2)"
    assert len(f"{second}{extension}".encode("utf-8")) == (
        serialization_module._MAX_FILENAME_COMPONENT_BYTES
    )
    assert allocated == {first.casefold(), second.casefold()}


def test_malformed_metric_qualifiers_skip_only_that_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An invalid qualifier object cannot collapse into and sum with a valid one."""
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {
                        "name": "cycles",
                        "value": 10,
                        "qualifiers": {"engine": "cpu"},
                    },
                    {
                        "name": "cycles",
                        "value": 20,
                        "qualifiers": {
                            "engine": "cpu",
                            "config": {1: "non-string nested key"},
                        },
                    },
                ],
            }
        ],
    )

    assert [path.name for path in artifacts.node_data] == ["cycles [engine=cpu].json"]
    assert _read_json(artifacts.node_data[0])["main"]["results"] == {"0": {"value": 10}}


def test_malformed_breakdown_qualifiers_skip_only_that_breakdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Invalid breakdown qualifiers reject all its metrics but not valid siblings."""
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "qualifiers": ["not", "an", "object"],
                "metrics": [{"name": "cycles", "value": 20}],
            },
            {
                "entity_id": "source_operator/operator/0",
                "qualifiers": {"phase": "warm"},
                "metrics": [{"name": "cycles", "value": 10}],
            },
        ],
    )

    assert [path.name for path in artifacts.node_data] == ["cycles [phase=warm].json"]
    assert _read_json(artifacts.node_data[0])["main"]["results"] == {"0": {"value": 10}}


def test_nonfinite_metric_qualifiers_are_rejected_at_artifact_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Non-finite nested values must not leak into standardized JSON artifacts."""
    with pytest.raises(ValueError, match="Out of range float values"):
        _write_metric_overlay_artifacts(
            tmp_path,
            monkeypatch,
            [
                {
                    "entity_id": "source_operator/operator/0",
                    "metrics": [
                        {
                            "name": "cycles",
                            "value": 1,
                            "qualifiers": {"ratio": float("nan")},
                        }
                    ],
                }
            ],
        )


def test_nonobject_metric_qualifiers_are_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Malformed qualifier shapes do not prevent valid overlays being written."""
    artifacts = _write_metric_overlay_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "entity_id": "source_operator/operator/0",
                "metrics": [
                    {
                        "name": "cycles",
                        "value": 2,
                        "qualifiers": "cpu",
                    },
                    {"name": "cycles", "value": 3},
                ],
            }
        ],
    )

    assert [path.name for path in artifacts.node_data] == ["cycles.json"]
    assert _read_json(artifacts.node_data[0])["main"] == {
        "name": "cycles",
        "results": {"0": {"value": 3}},
        "showExpandedSummaryOnGroupNode": True,
        "showLabelCountColumnsInChildrenStatsTable": True,
        "gradient": [
            {"stop": 0, "bgColor": "#2e7d32"},
            {"stop": 1, "bgColor": "#c62828"},
        ],
    }


def test_overlay_display_name_collisions_still_use_category_and_result_identity() -> (
    None
):
    """Qualifier naming preserves existing category and result collision rules."""
    status = artifact_models._OverlayKey(0, "performance", "status", "status")
    status_metric = artifact_models._OverlayKey(0, "performance", "metric", "status")
    performance_latency = artifact_models._OverlayKey(
        0, "performance", "metric", "latency"
    )
    optimization_latency = artifact_models._OverlayKey(
        1, "optimization", "metric", "latency"
    )
    keys = sorted(
        [
            optimization_latency,
            performance_latency,
            status_metric,
            status,
        ],
        key=node_data_module._overlay_sort_key,
    )

    assert node_data_module._overlay_display_names(keys) == {
        status: "status",
        status_metric: "status [metric]",
        performance_latency: "latency",
        optimization_latency: "latency [optimization]",
    }


def test_write_artifacts_disambiguates_normalized_node_data_filenames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Distinct overlay names survive collisions after filename normalization."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "source_operator/operator/0",
                        "name": "op",
                        "kind": "source_operator",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/operator/0",
                        "metrics": [
                            {"name": "latency/read", "value": 1, "unit": "ticks"},
                            {"name": "latency:read", "value": 2, "unit": "ticks"},
                        ],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.tflite")
    )

    assert [path.name for path in artifacts.node_data] == [
        "latency-read (ticks).json",
        "latency-read (ticks) (2).json",
    ]
    first = _read_json(artifacts.node_data[0])
    second = _read_json(artifacts.node_data[1])
    assert first["main"] == {
        "name": "latency/read (ticks)",
        "results": {"0": {"value": 1}},
        "showExpandedSummaryOnGroupNode": True,
        "showLabelCountColumnsInChildrenStatsTable": True,
        "gradient": [
            {"stop": 0, "bgColor": "#2e7d32"},
            {"stop": 1, "bgColor": "#c62828"},
        ],
    }
    assert second["main"] == {
        "name": "latency:read (ticks)",
        "results": {"0": {"value": 2}},
        "showExpandedSummaryOnGroupNode": True,
        "showLabelCountColumnsInChildrenStatsTable": True,
        "gradient": [
            {"stop": 0, "bgColor": "#2e7d32"},
            {"stop": 1, "bgColor": "#c62828"},
        ],
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"code_stack": {"exclude": ["**/site-packages/torch/**"]}},
        {"obsolete": False},
    ],
)
def test_plugin_rejects_obsolete_nonempty_settings(
    tmp_path: Path,
    monkeypatch,
    settings: dict[str, object],
) -> None:
    """Removed Model Explorer settings fail instead of being silently ignored."""
    registry = AnalysisPluginRegistry()
    ModelExplorerAnalysisPlugin.register(registry)
    plugin = registry.plugins[0]
    write = MagicMock()
    monkeypatch.setattr("mlia.model_explorer.plugin.write_artifacts", write)

    with pytest.raises(ConfigurationError, match="no longer supported"):
        plugin.run(_run_result(tmp_path, settings=settings))

    write.assert_not_called()


def test_plugin_rejects_invalid_output_before_writing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Plugin entry validates standardized output before artifact generation."""
    registry = AnalysisPluginRegistry()
    ModelExplorerAnalysisPlugin.register(registry)
    plugin = registry.plugins[0]
    write = MagicMock()
    monkeypatch.setattr("mlia.model_explorer.plugin.write_artifacts", write)

    with pytest.raises(SchemaValidationError):
        plugin.run(_run_result(tmp_path, output={"schema_version": "1.2.0"}))

    write.assert_not_called()


def test_plugin_run_files_writes_bundle_without_opening(
    tmp_path: Path, monkeypatch
) -> None:
    """Files mode writes artifacts and builds the standalone HTML page."""
    registry = AnalysisPluginRegistry()
    ModelExplorerAnalysisPlugin.register(registry)
    plugin = registry.plugins[0]
    build_bundle = MagicMock(return_value=tmp_path / "index.html")
    open_bundle = MagicMock()
    monkeypatch.setattr(
        "mlia.model_explorer.plugin.build_artifacts_bundle", build_bundle
    )
    monkeypatch.setattr("mlia.model_explorer.plugin.open_artifacts_bundle", open_bundle)

    plugin.run(_run_result(tmp_path, mode="files", model_exists=False))

    build_bundle.assert_called_once()
    assert build_bundle.call_args.args[0].name == "model-explorer"
    open_bundle.assert_not_called()


def test_plugin_run_open_mode_opens_standalone_html(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Open mode opens the generated standalone HTML in the default browser."""
    registry = AnalysisPluginRegistry()
    ModelExplorerAnalysisPlugin.register(registry)
    plugin = registry.plugins[0]
    build_bundle = MagicMock()
    open_bundle = MagicMock(return_value=tmp_path / "index.html")
    monkeypatch.setattr(
        "mlia.model_explorer.plugin.build_artifacts_bundle", build_bundle
    )
    monkeypatch.setattr("mlia.model_explorer.plugin.open_artifacts_bundle", open_bundle)

    plugin.run(_run_result(tmp_path, mode="open", model_exists=False))

    build_bundle.assert_not_called()
    open_bundle.assert_called_once()
    assert open_bundle.call_args.args[0].name == "model-explorer"


def test_plugin_rejects_model_format_without_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unsupported model formats fail before a bundle is built or opened."""
    registry = AnalysisPluginRegistry()
    ModelExplorerAnalysisPlugin.register(registry)
    plugin = registry.plugins[0]
    build_bundle = MagicMock()
    open_bundle = MagicMock()
    monkeypatch.setattr(
        "mlia.model_explorer.plugin.build_artifacts_bundle", build_bundle
    )
    monkeypatch.setattr("mlia.model_explorer.plugin.open_artifacts_bundle", open_bundle)

    with pytest.raises(
        UnsupportedConfigurationError,
        match=r"Model Explorer does not support model format '\.unsupported'\.",
    ):
        plugin.run(_run_result(tmp_path, mode="open", model_name="model.unsupported"))

    build_bundle.assert_not_called()
    open_bundle.assert_not_called()


def test_build_artifacts_bundle_writes_standalone_html(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Standalone HTML embeds the visualizer, graph collections, and node data."""
    fake_module = SimpleNamespace(TfliteAdapter=_OperatorIndexAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.tflite"))

    index = build_artifacts_bundle(artifacts.root)

    bundle = artifacts.root / "bundle"
    assert index == bundle / "index.html"
    assert index.exists()
    assert [path.name for path in bundle.iterdir()] == ["index.html"]
    html = index.read_text(encoding="utf-8")
    assert "<title>model.tflite - MLIA Model Explorer</title>" in html
    assert "const embeddedAssets =" in html
    assert "modelExplorer.createWorker" in html
    assert "document.createElement('model-explorer-visualizer')" in html
    assert 'id="message"' not in html
    assert 'id="left-graph-select"' in html
    assert 'id="add-pane-button"' in html
    assert 'id="right-graph-select"' in html
    data = _read_bundle_data(index)
    assert data["graphCollections"][0]["label"] == "Source Model"
    runs = data["nodeDataProviderRuns"]
    assert [run["name"] for run in runs] == ["placement", "checks", "advice"]
    assert [run["collectionLabel"] for run in runs] == [
        "Source Model",
        "Source Model",
        "Source Model",
    ]
    assert all(set(run) == {"collectionLabel", "name", "data"} for run in runs)
    assert all(set(run["data"]) == {"main"} for run in runs)
    assert "nodeData" not in data
    assert data["hostContractVersion"] == 1
    assert data["visualizerConfig"]["hideTitleBar"] is True
    assert data["visualizerConfig"]["hideLegends"] is True
    assert data["visualizerConfig"]["keepLayersWithASingleChild"] is True
    assert (
        data["visualizerConfig"]["artificialLayerNodeCountThreshold"] == 1_000_000_000
    )
    assert data["visualizerConfig"]["syncNavigationData"] == {
        "type": "sync_navigation",
        "graphMappings": [],
    }


def test_bundle_host_installs_inputs_and_controls_graph_selectors() -> None:
    """The host script owns selectors and drives the real custom-element API."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the generated host script")

    data = {
        "graphCollections": [
            {
                "label": "First",
                "graphs": [
                    {"id": "main", "nodes": []},
                    {"id": "alternate", "nodes": []},
                ],
            },
            {"label": "Second", "graphs": [{"id": "main", "nodes": []}]},
        ],
        "visualizerConfig": {"hideTitleBar": True},
        "nodeDataProviderRuns": [
            {
                "collectionLabel": "First",
                "name": "latency",
                "data": {"main": {"results": {}}},
            }
        ],
        "selectedNodeData": [{"modelName": "First", "name": "latency"}],
        "initialUiState": {
            "paneStates": [
                {
                    "selectedGraphId": "main",
                    "selectedCollectionLabel": "First",
                    "widthFraction": 1,
                }
            ]
        },
    }
    script = f"""
const data = {json.dumps(data)};
const operations = [];
const listeners = {{}};
const selectGraphCalls = [];
function element(id) {{
  return {{
    id,
    style: {{display: 'none'}},
    hidden: false,
    disabled: false,
    value: '',
    textContent: '',
    children: [],
    handlers: {{}},
    appendChild(child) {{ this.children.push(child); }},
    addEventListener(name, handler) {{ this.handlers[name] = handler; }},
  }};
}}
const elements = {{
  'mlia-model-explorer-data': {{textContent: JSON.stringify(data)}},
  'graph-toolbar': element('graph-toolbar'),
  'left-graph-control': element('left-graph-control'),
  'right-graph-control': element('right-graph-control'),
  'add-pane-button': element('add-pane-button'),
  'left-graph-select': element('left-graph-select'),
  'right-graph-select': element('right-graph-select'),
  'visualizer-host': element('visualizer-host'),
}};
const visualizer = new Proxy({{
  addEventListener(name, handler) {{ listeners[name] = handler; }},
  selectGraph(...args) {{ selectGraphCalls.push(args); }},
}}, {{
  set(target, property, value) {{
    operations.push({{type: 'set', property}});
    target[property] = value;
    return true;
  }}
}});
global.document = {{
  getElementById(id) {{ return elements[id] || null; }},
  createElement(name) {{
    if (name === 'option') return element('option');
    if (name !== 'model-explorer-visualizer') throw new Error(name);
    operations.push({{type: 'create'}});
    return visualizer;
  }},
  body: element('body'),
}};
elements['visualizer-host'].appendChild = (value) => {{
  if (value !== visualizer) throw new Error('unexpected element');
  operations.push({{type: 'append'}});
}};
{bundler_module._HOST_JS}
queueMicrotask(() => {{
  listeners.uiStateChanged({{detail: {{paneStates: [{{
    selectedGraphId: 'main',
    selectedCollectionLabel: 'First',
    widthFraction: 1,
  }}]}}}});
  const addPaneInitiallyVisible = !elements['add-pane-button'].hidden;
  const addPaneInitiallyEnabled = !elements['add-pane-button'].disabled;
  elements['add-pane-button'].handlers.click();
  elements['left-graph-select'].value = '2';
  elements['left-graph-select'].handlers.change();
  visualizer.graphCollections[1].graphs[0].id = 'main (1)';
  listeners.uiStateChanged({{detail: {{paneStates: [
    {{
      selectedGraphId: 'alternate',
      selectedCollectionLabel: 'First',
      widthFraction: 0.35,
    }},
    {{
      selectedGraphId: 'main (1)',
      selectedCollectionLabel: 'Second',
      widthFraction: 0.65,
    }},
  ]}}}});
  process.stdout.write(JSON.stringify({{
    operations,
    selectGraphCalls,
    visualizer,
    leftOptions: elements['left-graph-select'].children.map(
      (item) => item.textContent
    ),
    leftValue: elements['left-graph-select'].value,
    rightValue: elements['right-graph-select'].value,
    rightHidden: elements['right-graph-control'].hidden,
    addPaneInitiallyVisible,
    addPaneInitiallyEnabled,
    addPaneHidden: elements['add-pane-button'].hidden,
    addPaneDisabled: elements['add-pane-button'].disabled,
    toolbarColumns: elements['graph-toolbar'].style.gridTemplateColumns,
  }}));
}});
"""

    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["operations"] == [
        {"type": "create"},
        {"type": "set", "property": "graphCollections"},
        {"type": "set", "property": "config"},
        {"type": "set", "property": "nodeDataProviderRuns"},
        {"type": "set", "property": "selectedNodeData"},
        {"type": "set", "property": "syncNodeDataOverlaySelection"},
        {"type": "set", "property": "initialUiState"},
        {"type": "append"},
    ]
    assert result["leftOptions"] == [
        "First / main",
        "First / alternate",
        "Second",
    ]
    assert result["selectGraphCalls"] == [
        ["main", "First", 1],
        ["main", "Second", 0],
    ]
    assert result["leftValue"] == "1"
    assert result["rightValue"] == "2"
    assert result["rightHidden"] is False
    assert result["addPaneInitiallyVisible"] is True
    assert result["addPaneInitiallyEnabled"] is True
    assert result["addPaneHidden"] is True
    assert result["addPaneDisabled"] is True
    assert result["toolbarColumns"] == "35% 65%"
    assert result["visualizer"]["nodeDataProviderRuns"] == data["nodeDataProviderRuns"]
    assert result["visualizer"]["syncNodeDataOverlaySelection"] is True
    assert "modelGraphProcessed" not in result["visualizer"]


def test_build_artifacts_bundle_escapes_model_name_in_page_title(
    tmp_path: Path,
) -> None:
    """Standalone HTML title includes the model name escaped for HTML."""
    output = {
        "schema_version": "1.2.0",
        "model": {"name": "model<&>.bin"},
        "target": {},
        "backends": [],
        "results": [],
    }
    artifacts = write_artifacts(
        _run_result(
            tmp_path,
            output=output,
            model_name="model.bin",
            model_exists=False,
        )
    )

    index = build_artifacts_bundle(artifacts.root)

    html = index.read_text(encoding="utf-8")
    assert "<title>model&lt;&amp;&gt;.bin - MLIA Model Explorer</title>" in html


def test_build_artifacts_bundle_opens_source_and_overlay_rich_entity_graph_side_by_side(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Initial split view prefers the generated graph with greatest data coverage."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    },
                    {
                        "id": "module-0",
                        "name": "module 0",
                        "kind": "nn_module",
                        "placement": "NPU",
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain-0",
                        "metrics": [
                            {"name": "total_cycles", "value": 456, "unit": "cycles"}
                        ],
                    },
                    {
                        "entity_id": "module-0",
                        "metrics": [
                            {"name": "total_cycles", "value": 123, "unit": "cycles"}
                        ],
                    },
                ],
            }
        ],
    }
    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    index = build_artifacts_bundle(artifacts.root)

    data = _read_bundle_data(index)
    assert [collection["label"] for collection in data["graphCollections"]] == [
        "Source Model",
        "nn_module",
        "segment_cascade_chain",
    ]
    assert data["initialUiState"] == {
        "paneStates": [
            {
                "deepestExpandedGroupNodeIds": [],
                "selectedNodeId": "",
                "selectedGraphId": "Main",
                "selectedCollectionLabel": "Source Model",
                "widthFraction": 0.5,
            },
            {
                "deepestExpandedGroupNodeIds": [],
                "selectedNodeId": "",
                "selectedGraphId": "nn_module-Main",
                "selectedCollectionLabel": "nn_module",
                "widthFraction": 0.5,
                "selected": True,
            },
        ]
    }
    assert data["selectedNodeData"] == [
        {"modelName": "nn_module", "name": "total_cycles"},
        {"modelName": "segment_cascade_chain", "name": "total_cycles"},
    ]


def test_build_artifacts_bundle_falls_back_to_op_cycles_selected_node_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Op cycles are selected by default when total cycles are unavailable."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain-0",
                        "metrics": [
                            {"name": "op_cycles", "value": 123, "unit": "cycles"}
                        ],
                    }
                ],
            }
        ],
    }
    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    index = build_artifacts_bundle(artifacts.root)

    data = _read_bundle_data(index)
    assert data["selectedNodeData"] == [
        {"modelName": "Source Model", "name": "op_cycles"},
        {"modelName": "segment_cascade_chain", "name": "op_cycles"},
    ]


def test_build_artifacts_bundle_skips_selected_node_data_without_default_overlays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """No selected node-data default is emitted without known cycle overlays."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain-0",
                        "metrics": [
                            {"name": "npu_cycles", "value": 123, "unit": "cycles"}
                        ],
                    }
                ],
            }
        ],
    }
    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    index = build_artifacts_bundle(artifacts.root)

    data = _read_bundle_data(index)
    assert data["selectedNodeData"] == []


def test_write_artifacts_emits_one_graph_per_declared_entity_kind_hierarchy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Multiple source-operator-rooted entity-kind paths produce multiple graphs."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    },
                    {
                        "id": "module-0",
                        "name": "module 0",
                        "kind": "nn_module",
                    },
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert [path.name for path in artifacts.generated_models] == [
        "model-hierarchy-nn_module.json",
        "model-hierarchy-segment_cascade_chain.json",
    ]
    nn_module_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-nn_module.json"
    )
    assert nn_module_model["label"] == "nn_module"
    assert nn_module_model["graphs"][0]["id"] == "nn_module-Main"
    assert nn_module_model["graphs"][0]["nodes"][0]["namespace"] == ""
    assert nn_module_model["graphs"][0]["nodes"][1]["namespace"] == "module 0"


def test_write_artifacts_renders_supplied_code_stack_without_second_transformation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Model Explorer renders already-collapsed standardized output unchanged."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "code_stack/root",
                        "name": "model.py:10",
                        "kind": "code_stack",
                        "attributes": {"file": "src/model.py", "line": 10},
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    code_stack_model = _read_json(
        artifacts.root / "models" / "model-hierarchy-code_stack.json"
    )
    namespaces = [node["namespace"] for node in code_stack_model["graphs"][0]["nodes"]]
    assert "model.py:10" in namespaces
    assert _read_json(artifacts.standardized_output) == output


def test_write_artifacts_targets_namespace_overlays_to_matching_graph_collections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Node-data overlays are scoped to graph collections containing their keys."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain-0",
                        "metrics": [{"name": "latency", "unit": "cycles", "value": 10}],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    metric_overlay = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "latency (cycles).json"
    )
    assert metric_overlay["segment_cascade_chain-Main"]["results"] == {
        "chain 0": {"value": 10}
    }
    manifest = _read_json(artifacts.manifest)
    assert manifest["node_data"] == [
        {
            "path": "node-data/model-source/latency (cycles).json",
            "graph_collection_label": "Source Model",
        },
        {
            "path": (
                "node-data/model-hierarchy-segment_cascade_chain/latency (cycles).json"
            ),
            "graph_collection_label": "segment_cascade_chain",
        },
    ]


def test_write_artifacts_omits_node_data_with_no_matching_graph_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Node-data files with no matching graph collection are omitted entirely."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "ghost-op",
                        "name": "ghost op",
                        "kind": "source_operator",
                        "status": "estimated",
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert artifacts.node_data == []
    assert _read_json(artifacts.manifest)["node_data"] == []
    assert not (artifacts.root / "node-data" / "model-source" / "status.json").exists()


def test_write_artifacts_targets_one_overlay_to_multiple_graph_collections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A node-data overlay is listed once for each matching graph collection."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "chain-0",
                        "name": "chain 0",
                        "kind": "chain",
                        "status": "estimated",
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    assert artifacts.node_data == [
        artifacts.root / "node-data" / "model-source" / "status.json",
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "status.json",
    ]
    source_status = _read_json(
        artifacts.root / "node-data" / "model-source" / "status.json"
    )
    generated_status = _read_json(
        artifacts.root
        / "node-data"
        / "model-hierarchy-segment_cascade_chain"
        / "status.json"
    )
    assert set(source_status) == {"Main"}
    assert set(generated_status) == {"segment_cascade_chain-Main"}
    assert source_status["Main"]["results"] == {"0__op_0": {"value": "estimated"}}
    assert generated_status["segment_cascade_chain-Main"]["results"] == {
        "0__op_0": {"value": "estimated"}
    }
    manifest = _read_json(artifacts.manifest)
    assert manifest["node_data"] == [
        {
            "path": "node-data/model-source/status.json",
            "graph_collection_label": "Source Model",
        },
        {
            "path": "node-data/model-hierarchy-segment_cascade_chain/status.json",
            "graph_collection_label": "segment_cascade_chain",
        },
    ]
    assert manifest["nodeDataOverlayKeysAndDisplayNames"] == [
        {
            "overlayKey": {
                "resultIndex": 0,
                "resultKind": "performance",
                "category": "status",
                "name": "status",
                "unit": "",
                "breakdownQualifiers": {},
                "metricQualifiers": {},
            },
            "displayName": "status",
        }
    ]


def test_open_artifacts_bundle_opens_file_url(tmp_path: Path, monkeypatch) -> None:
    """Open mode opens the generated standalone HTML via file URL."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.vgf"))
    opened_urls: list[str] = []
    monkeypatch.setattr(bundler_module.webbrowser, "open", opened_urls.append)

    index = open_artifacts_bundle(artifacts.root)

    assert index == artifacts.root / "bundle" / "index.html"
    assert opened_urls == [index.resolve().as_uri()]


def test_write_artifacts_runs_adapter_and_writes_source_graph_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Known source model formats are converted to Model Explorer JSON upfront."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    import_module = MagicMock(return_value=fake_module)
    monkeypatch.setattr(source_model_module.importlib, "import_module", import_module)

    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.vgf"))

    import_module.assert_called_once_with("vgf_adapter_model_explorer.main")
    source_model = _read_json(artifacts.root / "models" / "model-source.json")
    expected = _FakeAdapter().convert(str(tmp_path / "model.vgf"), {})
    expected["label"] = "Source Model"
    assert source_model == expected
    manifest = _read_json(artifacts.manifest)
    assert manifest["model_path"] == "models/model-source.json"
    assert manifest.get("model_paths") is None
    assert not (artifacts.root / "models" / "model.tflite").exists()


def test_write_artifacts_runs_onnx_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """ONNX models use the optional adapter and canonical source mapping."""

    class ONNXAdapter:
        metadata = SimpleNamespace(fileExts=["onnx"])

        def convert(self, model_path: str, settings: dict) -> dict:
            return {
                "graphs": [
                    {
                        "id": "main",
                        "nodes": [
                            {
                                "id": "adapter_node",
                                "label": "Identity",
                                "outputsMetadata": [
                                    {
                                        "id": "0",
                                        "attrs": [
                                            {"key": "__tensor_tag", "value": "y"}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

    source_model = SimpleNamespace(
        graph=SimpleNamespace(
            name="main",
            node=[
                SimpleNamespace(
                    name="",
                    output=["y"],
                    op_type="Identity",
                    domain="",
                    attribute=[],
                )
            ],
        ),
        functions=[],
    )
    fake_module = SimpleNamespace(ONNXAdapter=ONNXAdapter)
    import_module = MagicMock(return_value=fake_module)
    monkeypatch.setattr(source_model_module.importlib, "import_module", import_module)
    monkeypatch.setattr(
        source_mapping_module,
        "_load_onnx_model",
        MagicMock(return_value=source_model),
    )

    generated = write_artifacts(_run_result(tmp_path, model_name="model.onnx"))

    import_module.assert_called_once_with("model_explorer_onnx.main")
    source_graph = _read_json(generated.root / "models" / "model-source.json")
    assert source_graph["graphs"][0]["nodes"][0]["id"] == "adapter_node"


def test_write_artifacts_overwrites_single_adapter_graph_collection_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Single adapter graph collection labels are normalized to Source Model."""
    fake_module = SimpleNamespace(VGFAdapter=_SingleGraphCollectionAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )

    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.vgf"))

    source_model = _read_json(artifacts.root / "models" / "model-source.json")
    assert source_model["label"] == "Source Model"
    assert source_model["graphs"][0]["nodes"][0]["id"] == "0"
    manifest = _read_json(artifacts.manifest)
    assert manifest["node_data"] == []


def test_write_artifacts_rejects_multiple_adapter_graph_collections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Multiple adapter graph collections fail instead of being flattened."""
    fake_module = SimpleNamespace(VGFAdapter=_MultipleGraphCollectionAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )

    with pytest.raises(RuntimeError) as err:
        write_artifacts(_run_result(tmp_path, model_name="model.vgf"))

    assert str(err.value) == (
        "Model Explorer adapter returned multiple graph collections; "
        "mlia-model-explorer currently requires exactly one. "
        "Returned collections: Left Adapter Label, Right Adapter Label."
    )


def test_write_artifacts_runs_builtin_tflite_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """TFLite source models use the built-in Python Model Explorer adapter."""
    fake_module = SimpleNamespace(TfliteAdapter=_TfliteOperatorIndexAdapter)
    import_module = MagicMock(return_value=fake_module)
    monkeypatch.setattr(source_model_module.importlib, "import_module", import_module)

    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.tflite"))

    import_module.assert_called_once_with(
        "model_explorer.builtin_tflite_flatbuffer_adapter"
    )
    source_model = _read_json(artifacts.root / "models" / "model-source.json")
    assert source_model["label"] == "Source Model"
    assert source_model["graphs"][0]["nodes"][0]["id"] == "0"
    manifest = _read_json(artifacts.manifest)
    assert manifest["model_path"] == "models/model-source.json"
    assert not (artifacts.root / "models" / "model.tflite").exists()


def test_build_artifacts_bundle_uses_generated_graph_json_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Serialized Model Explorer graph files are embedded for the visualizer."""
    fake_module = SimpleNamespace(VGFAdapter=_FakeAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    artifacts = write_artifacts(_run_result(tmp_path, model_name="model.vgf"))

    index = build_artifacts_bundle(artifacts.root)

    data = _read_bundle_data(index)
    assert data["graphCollections"] == [
        _read_json(artifacts.root / "models" / "model-source.json")
    ]


def test_write_artifacts_reports_missing_required_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Missing optional adapters are reported while writing artifacts."""
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(
            side_effect=ImportError("No module named 'vgf_adapter_model_explorer'")
        ),
    )

    with pytest.raises(RuntimeError) as err:
        write_artifacts(_run_result(tmp_path, model_name="model.vgf"))

    message = str(err.value)
    assert "vgf_adapter_model_explorer.main" in message
    assert "mlia-model-explorer[vgf]" in message
    assert "vgf-adapter-model-explorer" in message


def test_write_artifacts_maps_vgf_spirv_source_locations_to_stable_node_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """VGF/SPIR-V source locations map directly to stable adapter node ids."""
    fake_module = SimpleNamespace(VGFAdapter=_SpirvIdAdapter)
    monkeypatch.setattr(
        source_model_module.importlib,
        "import_module",
        MagicMock(return_value=fake_module),
    )
    output = {
        "schema_version": "1.2.0",
        "model": {},
        "target": {},
        "backends": [],
        "results": [
            {
                "kind": "performance",
                "entity_kinds": _TEST_ENTITY_KINDS,
                "entities": [
                    {
                        "id": "source_operator/segment_0/spirv-60",
                        "name": "conv",
                        "kind": "source_operator",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "source_operator/segment_0/spirv-60",
                        "metrics": [
                            {"name": "op_cycles", "value": 123, "unit": "cycles"}
                        ],
                    }
                ],
            }
        ],
    }

    artifacts = write_artifacts(
        _run_result(tmp_path, output=output, model_name="model.vgf")
    )

    op_cycles = _read_json(
        artifacts.root / "node-data" / "model-source" / "op_cycles.json"
    )
    assert op_cycles["Main"]["results"] == {
        "0__op_60": {"value": 123},
        "graph_segment_0___group___": {"value": 123},
    }
