# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for canonical ONNX source-operator mapping."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlia.core.errors import UnsupportedConfigurationError

from mlia.model_explorer._artifacts import source_mapping, source_model

_ASSERT = unittest.TestCase()


def _source_node(
    *,
    name: str = "",
    outputs: tuple[str, ...] = ("y",),
    op_type: str = "Identity",
    domain: str = "",
    attributes: tuple[Any, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        output=list(outputs),
        op_type=op_type,
        domain=domain,
        attribute=list(attributes),
    )


def _model(
    nodes: list[SimpleNamespace],
    *,
    graph_name: str = "main",
    functions: tuple[Any, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        graph=SimpleNamespace(name=graph_name, node=nodes),
        functions=list(functions),
    )


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
    model: SimpleNamespace,
    graph_collection: dict[str, Any],
) -> dict[str, list[source_mapping._NodeRef]]:
    loads: list[Path] = []

    def load_model(path: Path) -> SimpleNamespace:
        loads.append(path)
        return model

    monkeypatch.setattr(source_mapping, "_load_onnx_model", load_model)
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
    subgraph_attribute = SimpleNamespace(g=object(), graphs=[])
    with pytest.raises(
        UnsupportedConfigurationError, match="does not support subgraphs"
    ):
        _mapping(
            monkeypatch,
            _model([_source_node(attributes=(subgraph_attribute,))]),
            _graph_collection([_adapter_node("node", "Identity", "y")]),
        )


def test_onnx_functions_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(
        UnsupportedConfigurationError, match="does not support ONNX functions"
    ):
        _mapping(
            monkeypatch,
            _model([_source_node()], functions=(object(),)),
            _graph_collection([_adapter_node("node", "Identity", "y")]),
        )


def test_real_adapter_handles_initialized_input_and_duplicate_names(tmp_path) -> None:
    """Real adapter auxiliaries and NameFixPass renames preserve source identity."""
    onnx = pytest.importorskip("onnx")
    adapter_module = pytest.importorskip("model_explorer_onnx.main")

    model_input = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    model_output = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    initializer = onnx.helper.make_tensor("x", onnx.TensorProto.FLOAT, [1], [1.0])
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Identity", ["x"], ["first"], name="dup"),
                onnx.helper.make_node("Identity", ["first"], ["y"], name="dup"),
            ],
            "main",
            [model_input],
            [model_output],
            [initializer],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    model_path = tmp_path / "initialized-duplicate.onnx"
    onnx.save(model, model_path)

    adapter_response = adapter_module.ONNXAdapter().convert(str(model_path), {})
    graph_collection = source_model._source_model_graph_collection(
        model_path, adapter_response
    )
    mapping = source_mapping._source_operator_node_ids(model_path, graph_collection)

    _ASSERT.assertIn(
        "InitializedInput",
        {node["label"] for node in graph_collection["graphs"][0]["nodes"]},
    )
    _ASSERT.assertEqual(
        mapping,
        {
            "source_operator/0": [_ref("main", "dup")],
            "source_operator/1": [_ref("main", "dup_1")],
        },
    )
