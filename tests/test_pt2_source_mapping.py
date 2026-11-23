# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for canonical PT2 source-operator mapping."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest

from mlia.model_explorer._artifacts import source_mapping

_ASSERT = unittest.TestCase()


def _source_node(name: str, op: str) -> dict[str, str]:
    return {"name": name, "op": op}


def _source_facts(nodes: list[dict[str, str]]) -> dict:
    return {"nodes": nodes}


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
    facts: dict,
    graph_collection: dict[str, Any],
) -> dict[str, list[source_mapping._NodeRef]]:
    loads: list[Path] = []

    def load_facts(path: Path) -> dict:
        loads.append(path)
        return facts

    monkeypatch.setattr(source_mapping, "model_facts", load_facts)
    mapping = source_mapping._source_operator_node_ids(
        Path("model.pt2"), graph_collection
    )
    _ASSERT.assertEqual(loads, [Path("model.pt2")])
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
        _source_facts(nodes),
        _graph_collection(*(node["name"] for node in nodes)),
    )

    _ASSERT.assertEqual(
        mapping,
        {
            "source_operator/conv2d": [_ref("conv2d")],
            "source_operator/activation": [_ref("activation")],
            "source_operator/reshape": [_ref("reshape")],
        },
    )


def test_rejects_adapter_nodes_that_do_not_match_exported_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="missing node IDs.*unexpected node IDs"):
        _mapping(
            monkeypatch,
            _source_facts([_source_node("relu", "call_function")]),
            _graph_collection("different"),
        )


def test_rejects_duplicate_adapter_node_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_collection = _graph_collection("relu")
    graph_collection["graphs"][0]["nodes"].append({"id": "relu", "label": "duplicate"})

    with pytest.raises(RuntimeError, match="duplicate node ID 'relu'"):
        _mapping(
            monkeypatch,
            _source_facts([_source_node("relu", "call_function")]),
            graph_collection,
        )


def test_requires_one_top_level_adapter_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one top-level adapter graph"):
        _mapping(
            monkeypatch,
            _source_facts([_source_node("relu", "call_function")]),
            {"label": "Source Model", "graphs": []},
        )
