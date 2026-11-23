# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for canonical ONNX source-operator mapping."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest
from mlia.core.errors import UnsupportedConfigurationError

from mlia.model_explorer._artifacts import source_mapping

_ASSERT = unittest.TestCase()


def _source_node(
    *,
    name: str = "",
    outputs: tuple[str, ...] = ("y",),
    op_type: str = "Identity",
    domain: str = "",
    has_subgraphs: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "output": list(outputs),
        "op_type": op_type,
        "domain": domain,
        "has_subgraphs": has_subgraphs,
    }


def _model(
    nodes: list[dict], *, graph_name: str = "main", has_functions: bool = False
) -> dict:
    return {"name": graph_name, "nodes": nodes, "has_functions": has_functions}


def _adapter_node(
    node_id: str,
    label: str,
    *tensor_tags: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "outputsMetadata": [
            {
                "id": str(index),
                "attrs": [{"key": "__tensor_tag", "value": tensor_tag}],
            }
            for index, tensor_tag in enumerate(tensor_tags)
        ],
    }


def _graph_collection(
    nodes: list[dict[str, Any]], *, graph_id: str = "main"
) -> dict[str, Any]:
    return {
        "label": "Source Model",
        "graphs": [{"id": graph_id, "nodes": nodes}],
    }


def _mapping(
    monkeypatch: pytest.MonkeyPatch,
    model: dict,
    graph_collection: dict[str, Any],
) -> dict[str, list[source_mapping._NodeRef]]:
    loads: list[Path] = []

    def load_model(path: Path) -> dict:
        loads.append(path)
        return model

    monkeypatch.setattr(source_mapping, "model_facts", load_model)
    mapping = source_mapping._source_operator_node_ids(
        Path("model.onnx"), graph_collection
    )
    _ASSERT.assertEqual(loads, [Path("model.onnx")])
    return mapping


def _ref(graph_id: str, node_id: str) -> source_mapping._NodeRef:
    return source_mapping._NodeRef("Source Model", graph_id, node_id)


def test_named_node_requires_matching_name_output_and_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        monkeypatch,
        _model([_source_node(name="named", outputs=("named_out",))]),
        _graph_collection([_adapter_node("named", "Identity", "named_out")]),
    )

    _ASSERT.assertEqual(mapping, {"source_operator/0": [_ref("main", "named")]})


def test_duplicate_source_names_use_output_and_label_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        monkeypatch,
        _model(
            [
                _source_node(name="dup", outputs=("first",)),
                _source_node(name="dup", outputs=("second",)),
            ]
        ),
        _graph_collection(
            [
                _adapter_node("dup", "Identity", "first"),
                _adapter_node("dup_1", "Identity", "second"),
            ]
        ),
    )

    _ASSERT.assertEqual(
        mapping,
        {
            "source_operator/0": [_ref("main", "dup")],
            "source_operator/1": [_ref("main", "dup_1")],
        },
    )


def test_unnamed_node_uses_actual_adapter_id_and_output_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        monkeypatch,
        _model([_source_node(outputs=("y",))]),
        _graph_collection([_adapter_node("node_7", "Identity", "y")]),
    )

    _ASSERT.assertEqual(mapping, {"source_operator/0": [_ref("main", "node_7")]})


def test_multiple_outputs_ignore_omitted_optional_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        monkeypatch,
        _model(
            [
                _source_node(
                    outputs=("values", "", "indices"),
                    op_type="Custom",
                    domain="example.domain",
                )
            ]
        ),
        _graph_collection(
            [
                _adapter_node(
                    "node",
                    "example.domain::Custom",
                    "values",
                    "adapter_optional_placeholder",
                    "indices",
                )
            ]
        ),
    )

    _ASSERT.assertEqual(mapping, {"source_operator/0": [_ref("main", "node")]})


def test_adapter_only_input_output_and_initializer_nodes_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        monkeypatch,
        _model([_source_node(outputs=("y",))]),
        _graph_collection(
            [
                _adapter_node("[value] x", "Input", "x"),
                _adapter_node("[initializer] w", "Initializer", "w"),
                _adapter_node("[value] initialized", "InitializedInput", "initialized"),
                _adapter_node("node", "Identity", "y"),
                _adapter_node("[value] y", "Output"),
            ]
        ),
    )

    _ASSERT.assertEqual(mapping, {"source_operator/0": [_ref("main", "node")]})


def test_missing_tensor_tags_block_unnamed_node_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="found zero candidates.*output tensor"):
        _mapping(
            monkeypatch,
            _model([_source_node(outputs=("y",))]),
            _graph_collection([_adapter_node("node", "Identity")]),
        )


def test_name_and_output_evidence_disagreement_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="evidence disagrees"):
        _mapping(
            monkeypatch,
            _model([_source_node(name="named", outputs=("y",))]),
            _graph_collection(
                [
                    _adapter_node("named", "Identity", "other"),
                    _adapter_node("different", "Identity", "y"),
                ]
            ),
        )


def test_ambiguous_output_evidence_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="found multiple candidates"):
        _mapping(
            monkeypatch,
            _model([_source_node(outputs=("y",))]),
            _graph_collection(
                [
                    _adapter_node("node", "Identity", "y"),
                    _adapter_node("node_1", "Identity", "y"),
                ]
            ),
        )


def test_operation_type_and_domain_label_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="operation label mismatch"):
        _mapping(
            monkeypatch,
            _model(
                [
                    _source_node(
                        outputs=("y",), op_type="Identity", domain="custom.domain"
                    )
                ]
            ),
            _graph_collection([_adapter_node("node", "Identity", "y")]),
        )


def test_incomplete_mapping_rejects_unmapped_adapter_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="mapping is incomplete"):
        _mapping(
            monkeypatch,
            _model([_source_node(outputs=("y",))]),
            _graph_collection(
                [
                    _adapter_node("node", "Identity", "y"),
                    _adapter_node("extra", "Relu", "extra_out"),
                ]
            ),
        )


def test_mapping_must_be_one_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="not one-to-one"):
        _mapping(
            monkeypatch,
            _model(
                [
                    _source_node(outputs=("shared",)),
                    _source_node(outputs=("shared",)),
                ]
            ),
            _graph_collection([_adapter_node("node", "Identity", "shared")]),
        )


def test_graph_id_mismatch_is_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="graph mismatch"):
        _mapping(
            monkeypatch,
            _model([_source_node()], graph_name="original"),
            _graph_collection(
                [_adapter_node("node", "Identity", "y")], graph_id="adapter"
            ),
        )


def test_onnx_subgraphs_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(
        UnsupportedConfigurationError, match="does not support subgraphs"
    ):
        _mapping(
            monkeypatch,
            _model([_source_node(has_subgraphs=True)]),
            _graph_collection([_adapter_node("node", "Identity", "y")]),
        )


def test_onnx_functions_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(
        UnsupportedConfigurationError, match="does not support ONNX functions"
    ):
        _mapping(
            monkeypatch,
            _model([_source_node()], has_functions=True),
            _graph_collection([_adapter_node("node", "Identity", "y")]),
        )
