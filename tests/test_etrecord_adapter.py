# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for external ETRecord Model Explorer adapter integration."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from mlia.core.errors import ConfigurationError

from mlia.model_explorer._artifacts import etrecord, source_mapping


@pytest.mark.parametrize("native_identity", [False, True])
@pytest.mark.parametrize("methods", [["decode"], ["forward", "decode"]])
def test_single_program_adapter_uses_confirmed_analysis_method(
    native_identity, methods
):
    """Legacy forward graph labels are corrected only for proven single-method input."""
    attributes = [
        {"key": "Op Type", "value": "aten.relu.default"},
        {"key": "debug_handle", "value": "42"},
    ]
    if native_identity:
        attributes.append({"key": "Node Name", "value": "relu"})
    collection = {
        "label": "Source Model",
        "graphs": [{"id": "forward", "nodes": [{"id": "relu_", "attrs": attributes}]}],
    }
    source_id = "source_operator/etrecord/decode/relu"
    output = {
        "extensions": {
            "executorch": {
                "source_graph": {
                    "method": "decode",
                    "methods": methods,
                    "nodes": [
                        {
                            "id": source_id,
                            "name": "relu",
                            "op": "call_function",
                            "debug_handle": 42,
                        }
                    ],
                }
            }
        }
    }
    if len(methods) > 1:
        with pytest.raises(ConfigurationError, match="selected method 'decode'"):
            source_mapping._source_operator_node_ids(
                Path("model.etrecord"), collection, output=output
            )
        assert collection["graphs"][0]["id"] == "forward"
        return

    mappings = source_mapping._source_operator_node_ids(
        Path("model.etrecord"), collection, output=output
    )
    assert collection["graphs"][0]["id"] == "decode"
    assert list(mappings) == [source_id]
    assert mappings[source_id][0].graph_id == "decode"
    assert mappings[source_id][0].node_id == "relu_"


@pytest.mark.parametrize(
    "suffix,mismatch,expected_error",
    [
        (".etrecord", None, None),
        (".bin", None, None),
        (".etrecord", "count", "operation nodes"),
        (".etrecord", "handle", "order mismatch"),
        (".etrecord", "node_id", "without an ID"),
    ],
)
def test_external_etrecord_nodes_map_to_stable_source_identities(
    monkeypatch,
    suffix: str,
    mismatch: str | None,
    expected_error: str | None,
) -> None:
    """External adapter IDs map back to original ETRecord node names."""

    def unexpected_import(name):
        raise AssertionError(f"Mapping must not deserialize ETRecord again: {name}")

    monkeypatch.setattr(source_mapping.importlib, "import_module", unexpected_import)
    output = {
        "extensions": {
            "executorch": {
                "source_graph": {
                    "method": "forward",
                    "nodes": [
                        {
                            "id": "source_operator/etrecord/forward/conv",
                            "name": "conv",
                            "op": "call_function",
                            "debug_handle": 4,
                        }
                    ],
                    "edges": [],
                }
            }
        }
    }
    collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": "forward",
                "nodes": [
                    {"id": "GraphInputs", "attrs": []},
                    {
                        "id": "conv_",
                        "attrs": [
                            {"key": "Op Type", "value": "aten.convolution.default"},
                            {"key": "debug_handle", "value": "4"},
                        ],
                    },
                    {"id": "GraphOutputs", "attrs": []},
                ],
            }
        ],
    }

    adapter_nodes = collection["graphs"][0]["nodes"]
    if mismatch == "count":
        adapter_nodes.append({**adapter_nodes[1], "id": "extra_"})
    elif mismatch == "handle":
        adapter_nodes[1]["attrs"][1]["value"] = "5"
    elif mismatch == "node_id":
        adapter_nodes[1]["id"] = ""

    if expected_error is not None:
        with pytest.raises(RuntimeError, match=expected_error):
            source_mapping._source_operator_node_ids(
                Path(f"model{suffix}"), collection, output=output
            )
        return

    mappings = source_mapping._source_operator_node_ids(
        Path(f"model{suffix}"), collection, output=output
    )

    assert list(mappings) == ["source_operator/etrecord/forward/conv"]
    assert mappings[next(iter(mappings))][0].node_id == "conv_"


@pytest.mark.parametrize("graph_ids", [["decode"], ["forward", "forward"]])
def test_legacy_adapter_requires_one_graph_for_selected_method(graph_ids) -> None:
    """Legacy adapters must identify the selected graph without ambiguity."""
    collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": graph_id,
                "nodes": [
                    {
                        "id": "conv_",
                        "attrs": [
                            {"key": "Op Type", "value": "aten.convolution.default"}
                        ],
                    }
                ],
            }
            for graph_id in graph_ids
        ],
    }
    output = {
        "extensions": {
            "executorch": {"source_graph": {"method": "forward", "nodes": []}}
        }
    }

    with pytest.raises(
        ConfigurationError, match="exactly one graph for selected method 'forward'"
    ):
        source_mapping._source_operator_node_ids(
            Path("model.etrecord"), collection, output=output
        )


@pytest.mark.parametrize("suffix", [".etrecord", ".bin"])
def test_source_mapping_uses_adapter_metadata_without_reparsing(monkeypatch, suffix):
    """Method-scoped identity metadata survives reordered adapter nodes."""

    def unexpected_import(name):
        raise AssertionError(f"Mapping must not deserialize ETRecord again: {name}")

    monkeypatch.setattr(source_mapping.importlib, "import_module", unexpected_import)
    collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": "forward",
                "nodes": [
                    {
                        "id": "relu_",
                        "attrs": [
                            {"key": "Op Type", "value": "aten.relu.default"},
                            {"key": "Node Name", "value": "relu"},
                            {"key": "debug_handle", "value": "5"},
                        ],
                    },
                    {
                        "id": "conv_",
                        "attrs": [
                            {"key": "Op Type", "value": "aten.convolution.default"},
                            {"key": "Node Name", "value": "conv"},
                            {"key": "debug_handle", "value": "4"},
                        ],
                    },
                ],
            }
        ],
    }
    output = {
        "results": [
            {
                "entities": [
                    {
                        "id": "source_operator/etrecord/forward/conv",
                        "kind": "source_operator",
                        "name": "conv",
                        "attributes": {"debug_handle": 4},
                    },
                    {
                        "id": "source_operator/etrecord/forward/relu",
                        "kind": "source_operator",
                        "name": "relu",
                        "attributes": {"debug_handle": 5},
                    },
                ]
            }
        ]
    }
    mappings = source_mapping._source_operator_node_ids(
        Path("model" + suffix), collection, output=output
    )
    assert mappings["source_operator/etrecord/forward/conv"][0].node_id == "conv_"
    assert mappings["source_operator/etrecord/forward/relu"][0].node_id == "relu_"


def test_native_identity_names_keep_repeated_handles_method_scoped(monkeypatch):
    """Shared debug handles do not merge different operations or methods."""
    monkeypatch.setattr(
        source_mapping.importlib,
        "import_module",
        lambda name: pytest.fail(f"Unexpected reparse: {name}"),
    )
    collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": method,
                "nodes": [
                    {
                        "id": name + "_",
                        "attrs": [
                            {"key": "Op Type", "value": "aten." + name},
                            {"key": "Node Name", "value": name + "_"},
                            {"key": "debug_handle", "value": "42"},
                        ],
                    }
                    for name in ("conv", "rescale")
                ],
            }
            for method in ("forward", "decode")
        ],
    }
    output = {
        "results": [
            {
                "entities": [
                    {
                        "id": f"source_operator/etrecord/{method}/{name}",
                        "kind": "source_operator",
                        "name": name,
                        "attributes": {"debug_handle": 42},
                    }
                    for method in ("forward", "decode")
                    for name in ("conv", "rescale")
                ]
            }
        ]
    }
    mappings = source_mapping._source_operator_node_ids(
        Path("model.etrecord"), collection, output=output
    )
    for method in ("forward", "decode"):
        for name in ("conv", "rescale"):
            ref = mappings[f"source_operator/etrecord/{method}/{name}"][0]
            assert ref.graph_id == method
            assert ref.node_id == name + "_"


def test_native_source_mapping_indexes_names_and_handles_once(monkeypatch):
    """Mapping a large method does not scan every node for each source entity."""
    size = 100
    collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": "forward",
                "nodes": [
                    {
                        "id": f"node_{index}",
                        "attrs": [
                            {"key": "Op Type", "value": "aten.relu.default"},
                            {"key": "Node Name", "value": f"relu_{index}_"},
                            {"key": "debug_handle", "value": str(index)},
                        ],
                    }
                    for index in range(size)
                ],
            }
        ],
    }
    output = {
        "results": [
            {
                "entities": [
                    {
                        "id": f"source_operator/etrecord/forward/source_{index}",
                        "kind": "source_operator",
                        "name": f"relu_{index}" if index % 2 else f"renamed_{index}",
                        "attributes": {"debug_handle": index},
                    }
                    for index in range(size)
                ]
            }
        ]
    }
    names = Mock(wraps=etrecord._native_source_name)
    attributes = Mock(wraps=etrecord._node_string_attr)
    monkeypatch.setattr(etrecord, "_native_source_name", names)
    monkeypatch.setattr(etrecord, "_node_string_attr", attributes)

    mappings = source_mapping._source_operator_node_ids(
        Path("model.etrecord"), collection, output=output
    )

    for index in range(size):
        assert (
            mappings[f"source_operator/etrecord/forward/source_{index}"][0].node_id
            == f"node_{index}"
        )
    assert names.call_count <= size
    assert attributes.call_count <= 6 * size
