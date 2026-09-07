# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for canonical PT2 source-operator mapping."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlia.model_explorer._artifacts import source_mapping, source_model


def _source_node(name: str, op: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, op=op)


def _exported_program(nodes: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
    )


def _graph_collection(*node_ids: str) -> dict[str, Any]:
    return {
        "label": "Source Model",
        "graphs": [
            {
                "id": "graph",
                "nodes": [{"id": node_id, "label": node_id} for node_id in node_ids],
            }
        ],
    }


def _mapping(
    monkeypatch: pytest.MonkeyPatch,
    exported_program: SimpleNamespace,
    graph_collection: dict[str, Any],
) -> dict[str, list[source_mapping._NodeRef]]:
    loads: list[Path] = []

    def load_exported_program(path: Path) -> SimpleNamespace:
        loads.append(path)
        return exported_program

    monkeypatch.setattr(
        source_mapping, "_load_pt2_exported_program", load_exported_program
    )
    mapping = source_mapping._source_operator_node_ids(
        Path("model.pt2"), graph_collection
    )
    assert loads == [Path("model.pt2")]
    return mapping


def _ref(node_id: str) -> source_mapping._NodeRef:
    return source_mapping._NodeRef("Source Model", "graph", node_id)


def test_maps_top_level_operations_by_exact_fx_node_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [
        _source_node("weight", "placeholder"),
        _source_node("conv2d", "call_function"),
        _source_node("activation", "call_module"),
        _source_node("reshape", "call_method"),
        _source_node("constant", "get_attr"),
        _source_node("output", "output"),
    ]

    mapping = _mapping(
        monkeypatch,
        _exported_program(nodes),
        _graph_collection(*(node.name for node in nodes)),
    )

    assert mapping == {
        "source_operator/conv2d": [_ref("conv2d")],
        "source_operator/activation": [_ref("activation")],
        "source_operator/reshape": [_ref("reshape")],
    }


def test_rejects_adapter_nodes_that_do_not_match_exported_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="missing node IDs.*unexpected node IDs"):
        _mapping(
            monkeypatch,
            _exported_program([_source_node("relu", "call_function")]),
            _graph_collection("different"),
        )


def test_rejects_duplicate_adapter_node_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_collection = _graph_collection("relu")
    graph_collection["graphs"][0]["nodes"].append(
        {"id": "relu", "label": "duplicate"}
    )

    with pytest.raises(RuntimeError, match="duplicate node ID 'relu'"):
        _mapping(
            monkeypatch,
            _exported_program([_source_node("relu", "call_function")]),
            graph_collection,
        )


def test_requires_one_top_level_adapter_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one top-level adapter graph"):
        _mapping(
            monkeypatch,
            _exported_program([_source_node("relu", "call_function")]),
            {"label": "Source Model", "graphs": []},
        )


def test_real_builtin_adapter_maps_saved_exported_program(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("model_explorer.builtin_pytorch_exportedprogram_adapter")

    model = torch.nn.Sequential(torch.nn.ReLU())
    model_path = tmp_path / "model.pt2"
    torch.export.save(torch.export.export(model, (torch.ones(1),)), model_path)

    generated = source_model._write_source_model_files(model_path, tmp_path)
    loaded = torch.export.load(model_path)
    operation_names = {
        node.name
        for node in loaded.graph_module.graph.nodes
        if node.op in source_mapping._PT2_OPERATION_NODE_OPS
    }

    assert len(generated.paths) == 1
    assert generated.graph_collection is not None
    assert set(generated.source_operator_to_node_ids) == {
        f"source_operator/{node_name}" for node_name in operation_names
    }
    assert {
        node_ref.node_id
        for node_refs in generated.source_operator_to_node_ids.values()
        for node_ref in node_refs
    } == operation_names
