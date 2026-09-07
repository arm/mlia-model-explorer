# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused correctness tests for graph-qualified mappings and cyclic closure."""

import json
from pathlib import Path

import pytest
from mlia.core.errors import UnsupportedConfigurationError

from mlia.model_explorer._artifacts import (
    graphs,
    models,
    navigation,
    node_data,
    source_mapping,
)


def _ref(graph_id: str, node_id: str) -> models._NodeRef:
    return models._NodeRef("Source Model", graph_id, node_id)


def test_result_entity_node_ids_close_cycles_and_self_cycles() -> None:
    node_a = _ref("main", "a")
    node_b = _ref("main", "b")
    node_c = _ref("main", "c")
    result = {
        "entities": [
            {"id": "shared", "kind": "chain", "child_ids": ["cycle", "op-a"]},
            {"id": "cycle", "kind": "cascade", "child_ids": ["shared", "op-b"]},
            {"id": "self", "kind": "chain", "child_ids": ["self", "op-c"]},
            {"id": "op-a", "kind": "source_operator"},
            {"id": "op-b", "kind": "source_operator"},
            {"id": "op-c", "kind": "source_operator"},
        ]
    }

    resolved = graphs._result_entity_node_ids(
        result,
        {"op-a": [node_a], "op-b": [node_b], "op-c": [node_c]},
    )

    assert resolved["shared"] == [node_a, node_b]
    assert resolved["cycle"] == resolved["shared"]
    assert resolved["self"] == [node_c]


def test_tflite_operator_zero_is_distinct_in_each_subgraph() -> None:
    graph_collection = {
        "label": "Source Model",
        "graphs": [
            {"id": "main", "nodes": [{"id": "0", "label": "PRIMARY"}]},
            {"id": "subgraph_1", "nodes": [{"id": "0", "label": "NESTED"}]},
        ],
    }

    mapping = source_mapping._source_operator_node_ids(
        Path("model.tflite"), graph_collection
    )

    assert mapping == {
        "source_operator/operator/0": [_ref("main", "0")],
        "source_operator/subgraph/1/operator/0": [_ref("subgraph_1", "0")],
    }


def test_collection_scoped_overlay_routes_colliding_node_ids() -> None:
    primary = _ref("main", "0")
    nested = _ref("subgraph_1", "0")
    raw = {
        "name": "status",
        "results": {
            primary: {"value": "primary"},
            nested: {"value": "nested"},
        },
    }

    data_by_collection = node_data._collection_scoped_node_data("status", raw)

    assert set(data_by_collection) == {"Source Model"}
    graph_data = data_by_collection["Source Model"]
    assert graph_data["main"]["results"] == {"0": {"value": "primary"}}
    assert graph_data["subgraph_1"]["results"] == {"0": {"value": "nested"}}


def test_generated_namespace_refs_target_exact_graphs() -> None:
    hierarchy = models._EntityKindHierarchy(("chain", "source_operator"))
    generated = models._GeneratedEntityHierarchy(
        0, "performance", hierarchy, "chain", "chain"
    )
    source_ref = _ref("main", "0")
    generated_ref = models._NodeRef("chain", "chain-main", "0")
    expected_ref = models._NodeRef("chain", "chain-main", "group")
    group_ref = models._NodeRef("chain", "chain-main", "group___group___")
    projection = models._GeneratedHierarchyProjection(
        generated_refs_by_canonical_ref={source_ref: (generated_ref,)},
        canonical_ref_by_generated_ref={generated_ref: source_ref},
        source_operator_by_generated_ref={generated_ref: "op"},
        group_refs_by_entity_id={"group": (group_ref,)},
        terminal_complete_path_group_refs=frozenset({group_ref}),
    )
    artifact = models._GeneratedModelArtifact(
        Path("chain.json"),
        generated,
        {"label": "chain", "graphs": []},
        projection,
    )

    refs = node_data._result_entity_namespace_refs([artifact])

    assert refs == {"group": [expected_ref]}

    overlay_refs = node_data._result_overlay_node_ids(
        [artifact], {"op": [source_ref]}
    )
    assert overlay_refs == {"op": [source_ref, generated_ref]}

    data_by_collection = node_data._collection_scoped_node_data(
        "cycles", {"results": {expected_ref: {"value": 7}}}
    )

    assert set(data_by_collection) == {"chain"}
    graph_data = data_by_collection["chain"]
    assert set(graph_data) == {"chain-main"}
    assert graph_data["chain-main"]["results"] == {"group": {"value": 7}}


def test_overlay_identity_distinguishes_metric_from_same_named_field(
    tmp_path: Path,
) -> None:
    output = {
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "op",
                        "kind": "source_operator",
                        "status": "estimated",
                    }
                ],
                "breakdowns": [
                    {
                        "entity_id": "op",
                        "metrics": [{"name": "status", "value": 7}],
                    }
                ],
            }
        ]
    }

    generated = node_data._write_node_data_files(
        output,
        tmp_path,
        {"Source Model": "model-source"},
        {},
        {
            "label": "Source Model",
            "graphs": [{"id": "main", "nodes": [{"id": "0"}]}],
        },
        {"op": [_ref("main", "0")]},
    )

    paths = list(dict.fromkeys(item.path for item in generated))
    assert [path.name for path in paths] == [
        "status.json",
        "status [metric].json",
    ]
    field_overlay = json.loads(paths[0].read_text(encoding="utf-8"))
    metric_overlay = json.loads(paths[1].read_text(encoding="utf-8"))
    assert field_overlay["main"]["name"] == "status"
    assert field_overlay["main"]["results"] == {"0": {"value": "estimated"}}
    assert "gradient" not in field_overlay["main"]
    assert metric_overlay["main"]["name"] == ("status [metric]")
    assert metric_overlay["main"]["results"] == {"0": {"value": 7}}
    assert "gradient" in metric_overlay["main"]


def test_mapping_artifact_uses_structured_node_references() -> None:
    graph_collection = {
        "label": "Source Model",
        "graphs": [{"id": "main", "nodes": [{"id": "0"}]}],
    }
    output = {
        "results": [
            {
                "kind": "performance",
                "entities": [{"id": "op", "kind": "source_operator"}],
            }
        ]
    }

    mapping = navigation._entity_node_mappings(
        graph_collection,
        output,
        {},
        {"op": [_ref("main", "0")]},
    )["results"][0]

    node = {"collectionLabel": "Source Model", "graphId": "main", "nodeId": "0"}
    assert mapping["mappings"] == [{"node": node, "entityId": "op"}]


def test_source_operator_mapping_rejects_tosa_without_identity_contract() -> None:
    """TOSA fails without a verified native source-operator identity contract."""
    graph_collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": "main",
                "nodes": [
                    {"id": "0", "label": "CONV2D"},
                    {"id": "operator/1", "label": "ADD"},
                ],
            }
        ],
    }

    with pytest.raises(
        UnsupportedConfigurationError,
        match=(
            "Model Explorer does not support source-operator mapping for model "
            "format '\\.tosa'\\."
        ),
    ):
        source_mapping._source_operator_node_ids(Path("model.tosa"), graph_collection)


def test_generated_hierarchies_and_metrics_are_result_local() -> None:
    hierarchy_metadata = [
        {"id": "cascade", "child_kinds": ["chain"]},
        {
            "id": "chain",
            "parent_kinds": ["cascade"],
            "child_kinds": ["source_operator"],
        },
    ]
    output = {
        "results": [
            {
                "kind": "performance",
                "entity_kinds": hierarchy_metadata,
                "entities": [
                    {"id": "root", "name": "first root", "kind": "cascade"},
                    {
                        "id": "ancestor",
                        "name": "first ancestor",
                        "kind": "chain",
                        "parent_ids": ["root"],
                    },
                    {
                        "id": "shared-op",
                        "kind": "source_operator",
                        "parent_ids": ["ancestor"],
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "ancestor",
                        "metrics": [{"name": "cycles", "value": 7}],
                    }
                ],
            },
            {
                "kind": "memory",
                "entity_kinds": hierarchy_metadata,
                "entities": [
                    {"id": "root", "name": "second root", "kind": "cascade"},
                    {
                        "id": "ancestor",
                        "name": "second ancestor",
                        "kind": "chain",
                    },
                    {
                        "id": "shared-op",
                        "kind": "source_operator",
                        "parent_ids": ["ancestor"],
                    },
                ],
            },
        ]
    }
    source = {
        "label": "Source Model",
        "graphs": [{"id": "main", "nodes": [{"id": "0"}]}],
    }
    source_mapping = {"shared-op": [_ref("main", "0")]}

    generated = graphs._generated_entity_hierarchies(output)

    assert [(item.graph_name, item.collection_label) for item in generated] == [
        ("cascade_chain-result-1", "cascade_chain [performance result 1]"),
        ("cascade_chain-result-2", "cascade_chain [memory result 2]"),
    ]

    first_generated = graphs._source_model_with_entity_namespaces(
        output["results"][0],
        source,
        generated[0].hierarchy,
        generated[0].graph_name,
        generated[0].collection_label,
        source_mapping,
    )
    second_generated = graphs._source_model_with_entity_namespaces(
        output["results"][1],
        source,
        generated[1].hierarchy,
        generated[1].graph_name,
        generated[1].collection_label,
        source_mapping,
    )
    first = first_generated.graph_collection
    second = second_generated.graph_collection

    assert first["graphs"][0]["nodes"][0]["namespace"] == ("first root/first ancestor")
    assert second["graphs"][0]["nodes"][0]["namespace"] == "second ancestor"
    assert first["graphs"][0]["id"] == "cascade_chain-result-1-main"
    assert second["graphs"][0]["id"] == "cascade_chain-result-2-main"

    generated_artifacts = [
        models._GeneratedModelArtifact(
            Path("first.json"), generated[0], first, first_generated.projection
        ),
        models._GeneratedModelArtifact(
            Path("second.json"), generated[1], second, second_generated.projection
        ),
    ]
    mappings = navigation._entity_node_mappings(
        source,
        output,
        generated_artifacts,
        source_mapping,
    )["results"]
    assert mappings[0]["resultIndex"] == 0
    assert mappings[0]["resultKind"] == "performance"
    assert mappings[1]["resultIndex"] == 1
    assert mappings[1]["resultKind"] == "memory"
    first_group = {
        "node": {
            "collectionLabel": "cascade_chain [performance result 1]",
            "graphId": "cascade_chain-result-1-main",
            "nodeId": "first root/first ancestor___group___",
        },
        "entityId": "ancestor",
    }
    second_group = {
        "node": {
            "collectionLabel": "cascade_chain [memory result 2]",
            "graphId": "cascade_chain-result-2-main",
            "nodeId": "second ancestor___group___",
        },
        "entityId": "ancestor",
    }
    first_leaf = {
        "node": {
            "collectionLabel": "cascade_chain [performance result 1]",
            "graphId": "cascade_chain-result-1-main",
            "nodeId": "0",
        },
        "entityId": "shared-op",
    }
    second_leaf = {
        "node": {
            "collectionLabel": "cascade_chain [memory result 2]",
            "graphId": "cascade_chain-result-2-main",
            "nodeId": "0",
        },
        "entityId": "shared-op",
    }
    assert first_group in mappings[0]["mappings"]
    assert first_group not in mappings[1]["mappings"]
    assert second_group in mappings[1]["mappings"]
    assert second_group not in mappings[0]["mappings"]
    assert first_leaf in mappings[0]["mappings"]
    assert first_leaf not in mappings[1]["mappings"]
    assert second_leaf in mappings[1]["mappings"]
    assert second_leaf not in mappings[0]["mappings"]


def test_hierarchy_only_includes_source_adjacent_kinds_present_in_result() -> None:
    result = {
        "entity_kinds": [
            {"id": "segment", "child_kinds": ["cascade"]},
            {
                "id": "cascade",
                "parent_kinds": ["segment"],
                "child_kinds": ["chain"],
            },
            {
                "id": "chain",
                "parent_kinds": ["cascade"],
                "child_kinds": ["source_operator", "performance_group"],
            },
            {
                "id": "performance_group",
                "parent_kinds": ["chain"],
                "child_kinds": ["source_operator"],
            },
        ],
        "entities": [
            {"id": "segment-0", "kind": "segment"},
            {"id": "cascade-0", "kind": "cascade"},
            {"id": "chain-0", "kind": "chain"},
            {"id": "source_operator/operator/0", "kind": "source_operator"},
        ],
    }

    assert graphs._result_entity_kind_hierarchies(result) == [
        models._EntityKindHierarchy(
            ("segment", "cascade", "chain", "source_operator")
        )
    ]


def test_hierarchy_omits_coarse_prefix_covered_by_finer_path() -> None:
    result = {
        "entity_kinds": [
            {
                "id": "segment",
                "child_kinds": ["cascade", "source_operator"],
            },
            {
                "id": "cascade",
                "parent_kinds": ["segment"],
                "child_kinds": ["chain"],
            },
            {
                "id": "chain",
                "parent_kinds": ["cascade"],
                "child_kinds": ["source_operator"],
            },
        ],
        "entities": [
            {"id": "segment-0", "name": "segment 0", "kind": "segment"},
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
            {
                "id": "nested-op",
                "kind": "source_operator",
                "parent_ids": ["chain-0"],
            },
            {
                "id": "structural-op",
                "kind": "source_operator",
                "parent_ids": ["segment-0"],
            },
        ],
    }

    hierarchies = graphs._result_entity_kind_hierarchies(result)

    assert hierarchies == [
        models._EntityKindHierarchy(
            ("segment", "cascade", "chain", "source_operator")
        )
    ]

    source = {
        "label": "Source Model",
        "graphs": [
            {
                "id": "main",
                "nodes": [{"id": "nested"}, {"id": "structural"}],
            }
        ],
    }
    generated = graphs._source_model_with_entity_namespaces(
        result,
        source,
        hierarchies[0],
        hierarchies[0].graph_name,
        hierarchies[0].graph_name,
        {
            "nested-op": [_ref("main", "nested")],
            "structural-op": [_ref("main", "structural")],
        },
    )

    assert [
        node["namespace"]
        for node in generated.graph_collection["graphs"][0]["nodes"]
    ] == [
        "segment 0/cascade 0/chain 0",
        "segment 0",
    ]


def test_hierarchy_keeps_divergent_fine_paths() -> None:
    result = {
        "entity_kinds": [
            {"id": "root", "child_kinds": ["left", "right"]},
            {
                "id": "left",
                "parent_kinds": ["root"],
                "child_kinds": ["source_operator"],
            },
            {
                "id": "right",
                "parent_kinds": ["root"],
                "child_kinds": ["source_operator"],
            },
        ],
        "entities": [
            {"id": "root-0", "kind": "root"},
            {"id": "left-0", "kind": "left"},
            {"id": "right-0", "kind": "right"},
            {"id": "source-0", "kind": "source_operator"},
        ],
    }

    assert graphs._result_entity_kind_hierarchies(result) == [
        models._EntityKindHierarchy(("root", "left", "source_operator")),
        models._EntityKindHierarchy(("root", "right", "source_operator")),
    ]


def test_hierarchy_records_recursive_entity_kinds() -> None:
    result = {
        "entity_kinds": [
            {
                "id": "nn_module",
                "parent_kinds": ["nn_module"],
                "child_kinds": ["source_operator"],
            }
        ],
        "entities": [{"id": "module-0", "kind": "nn_module"}],
    }

    assert graphs._result_entity_kind_hierarchies(result) == [
        models._EntityKindHierarchy(
            ("nn_module", "source_operator"), frozenset({"nn_module"})
        )
    ]


def test_hierarchy_discovers_standalone_and_code_line_stack_paths() -> None:
    """Well-known line parents should add a view without replacing stack view."""
    result = {
        "entities": [
            {"id": "line-0", "kind": "code_line"},
            {"id": "stack-0", "kind": "code_stack"},
            {"id": "source-0", "kind": "source_operator"},
        ]
    }

    assert graphs._result_entity_kind_hierarchies(result) == [
        models._EntityKindHierarchy(
            ("code_line", "code_stack", "source_operator"),
            frozenset({"code_stack"}),
        ),
        models._EntityKindHierarchy(
            ("code_stack", "source_operator"),
            frozenset({"code_stack"}),
        ),
    ]


def test_code_line_stack_hierarchy_retains_recursive_caller_lineages() -> None:
    """Stack recursion should expose source nodes beneath leaf and caller lines."""
    result = {
        "entities": [
            {
                "id": "line-20",
                "kind": "code_line",
                "name": "line 20",
                "child_ids": ["stack-20"],
            },
            {
                "id": "line-39",
                "kind": "code_line",
                "name": "line 39",
                "child_ids": ["stack-39"],
            },
            {
                "id": "stack-39",
                "kind": "code_stack",
                "name": "stack 39",
                "parent_ids": ["line-39"],
                "child_ids": ["stack-20"],
            },
            {
                "id": "stack-20",
                "kind": "code_stack",
                "name": "stack 20",
                "parent_ids": ["line-20", "stack-39"],
                "child_ids": ["source"],
            },
            {
                "id": "source",
                "kind": "source_operator",
                "name": "source",
                "parent_ids": ["stack-20"],
            },
        ]
    }
    hierarchy = next(
        item
        for item in graphs._result_entity_kind_hierarchies(result)
        if item.graph_name == "code_line_code_stack"
    )

    projection = graphs._entity_hierarchy_complete_paths(
        graphs._entities_by_id(result), hierarchy, set()
    )

    assert projection.source_operator_paths_by_id["source"] == (
        "line 20/stack 20",
        "line 39/stack 39/stack 20",
    )
