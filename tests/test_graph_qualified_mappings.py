# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused correctness tests for graph-qualified mappings and cyclic closure."""

import json
import unittest
from pathlib import Path

import pytest
from mlia.core.errors import UnsupportedConfigurationError

from mlia.model_explorer._artifacts import etrecord as et_integration
from mlia.model_explorer._artifacts import (
    graphs,
    models,
    navigation,
    node_data,
    source_mapping,
)
from mlia.model_explorer._artifacts.formats import _format_integration

_ASSERT = unittest.TestCase()


def _ref(graph_id: str, node_id: str) -> models._NodeRef:
    return models._NodeRef("Source Model", graph_id, node_id)


@pytest.mark.parametrize("suffix", [".vgf", ".etrecord"])
@pytest.mark.parametrize("adapter_layout", [False, True])
def test_generated_hierarchy_preserves_layout_outside_etrecord(
    tmp_path: Path, suffix: str, adapter_layout: bool
) -> None:
    """Hierarchy export preserves existing layouts and scopes ETRecord's override."""
    original_layout = [{"namespaceRegex": "Existing", "layoutDirection": 1}]
    source_graph = {"id": "main", "nodes": [{"id": "conv", "label": "Conv2d"}]}
    if adapter_layout:
        source_graph["groupNodeConfigs"] = original_layout
    source = {"label": "Source Model", "graphs": [source_graph]}
    output = {
        "results": [
            {
                "kind": "performance",
                "entity_kinds": [{"id": "chain", "child_kinds": ["source_operator"]}],
                "entities": [
                    {"id": "chain", "kind": "chain", "name": "Chain"},
                    {
                        "id": "source",
                        "kind": "source_operator",
                        "parent_ids": ["chain"],
                    },
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain",
                        "metrics": [
                            {"name": "total_cycles", "value": 100, "unit": "cycles"}
                        ],
                    }
                ],
            }
        ]
    }

    artifacts = graphs._write_entity_subgraph_model_file(
        Path("model" + suffix),
        tmp_path,
        output,
        source,
        {"source": [_ref("main", "conv")]},
        presentation=_format_integration(Path("model" + suffix)).hierarchy_presentation,
    )

    assert len(artifacts) == 1
    graph = json.loads(artifacts[0].path.read_text())["graphs"][0]
    assert graph["nodes"][0]["namespace"] == "Chain"
    if suffix in {".etrecord", ".bin"}:
        assert graph["groupNodeConfigs"] == [
            {"namespaceRegex": ".*", "layoutDirection": 0}
        ]
    elif adapter_layout:
        assert graph["groupNodeConfigs"] == original_layout
    else:
        assert "groupNodeConfigs" not in graph
    assert source_graph.get("groupNodeConfigs") == (
        original_layout if adapter_layout else None
    )


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

    _ASSERT.assertEqual(resolved["shared"], [node_a, node_b])
    _ASSERT.assertEqual(resolved["cycle"], resolved["shared"])
    _ASSERT.assertEqual(resolved["self"], [node_c])


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

    _ASSERT.assertEqual(
        mapping,
        {
            "source_operator/operator/0": [_ref("main", "0")],
            "source_operator/subgraph/1/operator/0": [_ref("subgraph_1", "0")],
        },
    )


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

    _ASSERT.assertEqual(set(data_by_collection), {"Source Model"})
    graph_data = data_by_collection["Source Model"]
    _ASSERT.assertEqual(graph_data["main"]["results"], {"0": {"value": "primary"}})
    _ASSERT.assertEqual(graph_data["subgraph_1"]["results"], {"0": {"value": "nested"}})


def test_mixed_hierarchy_metric_colors_are_normalized_by_entity_kind() -> None:
    overlay_key = models._OverlayKey(0, "performance", "metric", "cycles")
    chain_a = models._NodeRef("chain", "chain-forward", "Chain A")
    chain_a_full = models._NodeRef(
        "segment_cascade_chain", "full-forward", "Segment/Cascade/Chain A"
    )
    chain_b = models._NodeRef("chain", "chain-forward", "Chain B")
    chain_b_full = models._NodeRef(
        "segment_cascade_chain", "full-forward", "Segment/Cascade/Chain B"
    )
    cascade_a = models._NodeRef(
        "segment_cascade_chain", "full-forward", "Segment/Cascade A"
    )
    cascade_b = models._NodeRef(
        "segment_cascade_chain", "full-forward", "Segment/Cascade B"
    )
    segment = models._NodeRef("segment_cascade_chain", "full-forward", "Segment")
    entries = {
        chain_a: {"value": 10},
        chain_a_full: {"value": 10},
        chain_b: {"value": 20},
        chain_b_full: {"value": 20},
        cascade_a: {"value": 500},
        cascade_b: {"value": 1000},
        segment: {"value": 2000},
    }
    kinds = {
        chain_a: {"chain"},
        chain_a_full: {"chain"},
        chain_b: {"chain"},
        chain_b_full: {"chain"},
        cascade_a: {"cascade"},
        cascade_b: {"cascade"},
        segment: {"segment"},
    }

    node_data._apply_level_metric_colors(
        {overlay_key: entries},
        {overlay_key: kinds},
        {"chain", "segment_cascade_chain"},
    )

    assert entries[chain_a] == {"value": 10, "bgColor": "#a0ffb0"}
    assert entries[chain_a_full] == entries[chain_a]
    assert entries[chain_b] == {"value": 20, "bgColor": "#ff8888"}
    assert entries[chain_b_full] == entries[chain_b]
    assert entries[cascade_a] == {"value": 500, "bgColor": "#a0ffb0"}
    assert entries[cascade_b] == {"value": 1000, "bgColor": "#ff8888"}
    assert entries[segment] == {"value": 2000, "bgColor": "#a0ffb0"}


def test_standard_etdump_metrics_use_standard_gradient() -> None:
    overlay_key = models._OverlayKey(
        0,
        "performance",
        "metric",
        "Runtime Total Avg",
        "ms",
    )
    node_ref = models._NodeRef("Source Model", "forward", "conv")

    data = node_data._graph_node_data(
        overlay_key,
        {node_ref: {"value": 1.5}},
        presentation=et_integration._ETRecordIntegration.metric_presentation,
    )

    assert data["gradient"] == [
        {"stop": 0, "bgColor": "#a0ffb0"},
        {"stop": 0.5, "bgColor": "#fff855"},
        {"stop": 1, "bgColor": "#ff8888"},
    ]
    assert data["hideAggregatedStats"] == ["sum"]


def test_etrecord_subset_view_keeps_hierarchy_colors(tmp_path: Path) -> None:
    """A companion with only one mapped chain keeps the full view's scale."""
    result = {
        "kind": "performance",
        "entity_kinds": [
            {"id": "chain", "child_kinds": ["source_operator"]},
        ],
        "entities": [
            entity
            for index in range(2)
            for entity in (
                {"id": f"chain-{index}", "kind": "chain"},
                {
                    "id": f"op-{index}",
                    "kind": "source_operator",
                    "parent_ids": [f"chain-{index}"],
                },
            )
        ],
        "breakdowns": [
            {
                "entity_id": f"chain-{index}",
                "metrics": [{"name": "cycles", "value": value}],
            }
            for index, value in enumerate((10, 100))
        ]
        + [
            {
                "entity_id": "chain-1",
                "qualifiers": {"capture": "other"},
                "metrics": [{"name": "cycles", "value": 10000}],
            }
        ],
    }
    source = {
        "label": "Source Model",
        "graphs": [{"id": "forward", "nodes": [{"id": str(i)} for i in range(2)]}],
    }
    source_refs = {f"op-{i}": [_ref("forward", str(i))] for i in range(2)}
    integration = et_integration._ETRecordIntegration()
    generated = graphs._write_entity_subgraph_model_file(
        Path("model.etrecord"),
        tmp_path,
        {"results": [result]},
        source,
        source_refs,
        presentation=integration.hierarchy_presentation,
    )
    hierarchy_overlays = node_data._node_data_overlays(
        [result],
        {0: generated},
        source,
        source_refs,
        presentation=integration.metric_presentation,
    )
    companion_ref = models._NodeRef("Companion", "forward_0_0", "Chain 1")
    companion_overlays = node_data._entity_metric_overlays(
        {"results": [result]},
        {(0, "chain-1"): (companion_ref,)},
        {"Companion"},
        presentation=integration.metric_presentation,
    )
    assert len(hierarchy_overlays) == len(companion_overlays) == 2
    for hierarchy, companion in zip(hierarchy_overlays, companion_overlays):
        assert hierarchy.key == companion.key
        expected_color = "#a0ffb0" if hierarchy.key.breakdown_qualifiers else "#ff8888"
        expected_value = 10000 if hierarchy.key.breakdown_qualifiers else 100
        chain_ref = generated[0].projection.group_refs_by_entity_id["chain-1"][0]
        chain_data = hierarchy.node_data["results"][
            models._NodeRef(
                chain_ref.collection_label,
                chain_ref.graph_id,
                chain_ref.node_id.removesuffix("___group___"),
            )
        ]
        assert (
            companion.node_data["results"][companion_ref]
            == chain_data
            == {
                "value": expected_value,
                "bgColor": expected_color,
            }
        )


def test_generic_metrics_keep_main_aggregation_despite_etdump_display_name() -> None:
    """Only explicit format policy changes browser aggregate presentation."""
    key = models._OverlayKey(0, "performance", "metric", "Runtime Total Avg", "ms")
    ref = models._NodeRef("Source Model", "main", "conv")

    data = node_data._graph_node_data(key, {ref: {"value": 1.5}})

    assert "hideAggregatedStats" not in data


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "#a0ffb0"),
        (50, "#fff855"),
        (100, "#ff8888"),
    ],
)
def test_metric_gradient_uses_requested_min_mid_max_stops(
    value: float,
    expected: str,
) -> None:
    assert node_data._metric_gradient_color(value, 0, 100) == expected


def test_standard_etdump_result_uses_source_namespace_projection() -> None:
    entity_id = "source_operator/etrecord/forward/linear"
    node_ref = models._NodeRef("Source Model", "forward", "linear")
    result = {
        "kind": "performance",
        "producer": "neural-technology-profiling-data",
        "entities": [
            {
                "id": entity_id,
                "kind": "source_operator",
                "name": "linear",
            }
        ],
        "breakdowns": [
            {
                "id": "runtime/linear",
                "entity_id": entity_id,
                "metrics": [
                    {
                        "name": "Runtime Total Avg",
                        "value": 0.5,
                        "unit": "ms",
                    }
                ],
            }
        ],
    }
    source_graph_collection = {
        "label": "Source Model",
        "graphs": [
            {
                "id": "forward",
                "nodes": [
                    {
                        "id": "linear",
                        "namespace": "fc",
                    }
                ],
            }
        ],
    }

    breakdowns, entities, refs = node_data._source_namespace_projection(
        result,
        source_graph_collection,
        {entity_id: [node_ref]},
    )
    assert len(breakdowns) == 1
    assert {entity["name"] for entity in entities.values()} == {"fc"}
    assert list(refs.values()) == [
        [models._NodeRef("Source Model", "forward", "fc___group___")]
    ]


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

    _ASSERT.assertEqual(refs, {"group": [expected_ref]})

    overlay_refs = node_data._result_overlay_node_ids([artifact], {"op": [source_ref]})
    _ASSERT.assertEqual(overlay_refs, {"op": [source_ref, generated_ref]})

    data_by_collection = node_data._collection_scoped_node_data(
        "cycles", {"results": {expected_ref: {"value": 7}}}
    )

    _ASSERT.assertEqual(set(data_by_collection), {"chain"})
    graph_data = data_by_collection["chain"]
    _ASSERT.assertEqual(set(graph_data), {"chain-main"})
    _ASSERT.assertEqual(graph_data["chain-main"]["results"], {"group": {"value": 7}})


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

    overlays = node_data._node_data_overlays(
        output["results"],
        {},
        {
            "label": "Source Model",
            "graphs": [{"id": "main", "nodes": [{"id": "0"}]}],
        },
        {"op": [_ref("main", "0")]},
    )
    generated = node_data._write_logical_node_data_files(
        overlays, tmp_path, {"Source Model": "model-source"}
    )

    paths = list(dict.fromkeys(item.path for item in generated))
    _ASSERT.assertEqual(
        [path.name for path in paths],
        [
            "status.json",
            "status [metric].json",
        ],
    )
    field_overlay = json.loads(paths[0].read_text(encoding="utf-8"))
    metric_overlay = json.loads(paths[1].read_text(encoding="utf-8"))
    _ASSERT.assertEqual(field_overlay["main"]["name"], "status")
    _ASSERT.assertEqual(field_overlay["main"]["results"], {"0": {"value": "estimated"}})
    _ASSERT.assertNotIn("gradient", field_overlay["main"])
    _ASSERT.assertEqual(metric_overlay["main"]["name"], "status [metric]")
    _ASSERT.assertEqual(metric_overlay["main"]["results"], {"0": {"value": 7}})
    _ASSERT.assertIn("gradient", metric_overlay["main"])


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
    _ASSERT.assertEqual(mapping["mappings"], [{"node": node, "entityId": "op"}])


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

    _ASSERT.assertEqual(
        [(item.graph_name, item.collection_label) for item in generated],
        [
            ("cascade_chain-result-1", "cascade_chain [performance result 1]"),
            ("cascade_chain-result-2", "cascade_chain [memory result 2]"),
        ],
    )

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

    _ASSERT.assertEqual(
        first["graphs"][0]["nodes"][0]["namespace"],
        "first root/first ancestor",
    )
    _ASSERT.assertEqual(second["graphs"][0]["nodes"][0]["namespace"], "second ancestor")
    _ASSERT.assertEqual(first["graphs"][0]["id"], "cascade_chain-result-1-main")
    _ASSERT.assertEqual(second["graphs"][0]["id"], "cascade_chain-result-2-main")

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
    _ASSERT.assertEqual(mappings[0]["resultIndex"], 0)
    _ASSERT.assertEqual(mappings[0]["resultKind"], "performance")
    _ASSERT.assertEqual(mappings[1]["resultIndex"], 1)
    _ASSERT.assertEqual(mappings[1]["resultKind"], "memory")
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
    _ASSERT.assertIn(first_group, mappings[0]["mappings"])
    _ASSERT.assertNotIn(first_group, mappings[1]["mappings"])
    _ASSERT.assertIn(second_group, mappings[1]["mappings"])
    _ASSERT.assertNotIn(second_group, mappings[0]["mappings"])
    _ASSERT.assertIn(first_leaf, mappings[0]["mappings"])
    _ASSERT.assertNotIn(first_leaf, mappings[1]["mappings"])
    _ASSERT.assertIn(second_leaf, mappings[1]["mappings"])
    _ASSERT.assertNotIn(second_leaf, mappings[0]["mappings"])


@pytest.mark.parametrize("result_count", [1, 2])
def test_additional_entity_hierarchy_adds_chain_only_view(result_count: int) -> None:
    """Format presentation adds views using relationships in each result."""
    output = {
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
                    {
                        "id": "chain",
                        "parent_kinds": ["cascade"],
                        "child_kinds": ["source_operator"],
                    },
                ],
                "entities": [
                    {"id": "segment-0", "kind": "segment"},
                    {
                        "id": "cascade-0",
                        "kind": "cascade",
                        "parent_ids": ["segment-0"],
                    },
                    {
                        "id": "chain-0",
                        "kind": "chain",
                        "parent_ids": ["cascade-0"],
                    },
                    {
                        "id": "op-0",
                        "kind": "source_operator",
                        "parent_ids": ["chain-0"],
                    },
                ],
            }
        ],
    }

    output["results"] *= result_count
    generated = graphs._generated_entity_hierarchies(
        output,
        presentation=models._HierarchyPresentation(
            additional_hierarchy_paths=(("chain", "source_operator"),)
        ),
    )

    assert [item.graph_name for item in generated] == [
        name + (f"-result-{index + 1}" if result_count > 1 else "")
        for index in range(result_count)
        for name in ("chain", "segment_cascade_chain")
    ]
    assert [item.hierarchy.path for item in generated] == [
        ("chain", "source_operator"),
        ("segment", "cascade", "chain", "source_operator"),
    ] * result_count

    etrecord_generated = graphs._generated_entity_hierarchies(
        output,
        presentation=_format_integration(Path("model.etrecord")).hierarchy_presentation,
    )
    assert [item.collection_label for item in etrecord_generated] == [
        label + (f" [performance result {index + 1}]" if result_count > 1 else "")
        for label in ("segment_cascade_chain", "ETRecord Chain Mapping")
        for index in range(result_count)
    ]
    assert {item.graph_name for item in etrecord_generated} == {
        item.graph_name for item in generated
    }


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

    _ASSERT.assertEqual(
        graphs._result_entity_kind_hierarchies(result),
        [
            models._EntityKindHierarchy(
                ("segment", "cascade", "chain", "source_operator")
            )
        ],
    )


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

    _ASSERT.assertEqual(
        hierarchies,
        [
            models._EntityKindHierarchy(
                ("segment", "cascade", "chain", "source_operator")
            )
        ],
    )

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

    _ASSERT.assertEqual(
        [
            node["namespace"]
            for node in generated.graph_collection["graphs"][0]["nodes"]
        ],
        [
            "segment 0/cascade 0/chain 0",
            "segment 0",
        ],
    )


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

    _ASSERT.assertEqual(
        graphs._result_entity_kind_hierarchies(result),
        [
            models._EntityKindHierarchy(("root", "left", "source_operator")),
            models._EntityKindHierarchy(("root", "right", "source_operator")),
        ],
    )


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

    _ASSERT.assertEqual(
        graphs._result_entity_kind_hierarchies(result),
        [
            models._EntityKindHierarchy(
                ("nn_module", "source_operator"), frozenset({"nn_module"})
            )
        ],
    )


def test_hierarchy_discovers_standalone_and_code_line_stack_paths() -> None:
    """Well-known line parents should add a view without replacing stack view."""
    result = {
        "entities": [
            {"id": "line-0", "kind": "code_line"},
            {"id": "stack-0", "kind": "code_stack"},
            {"id": "source-0", "kind": "source_operator"},
        ]
    }

    _ASSERT.assertEqual(
        graphs._result_entity_kind_hierarchies(result),
        [
            models._EntityKindHierarchy(
                ("code_line", "code_stack", "source_operator"),
                frozenset({"code_stack"}),
            ),
            models._EntityKindHierarchy(
                ("code_stack", "source_operator"),
                frozenset({"code_stack"}),
            ),
        ],
    )


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

    _ASSERT.assertEqual(
        projection.source_operator_paths_by_id["source"],
        (
            "line 20/stack 20",
            "line 39/stack 39/stack 20",
        ),
    )


def test_operator_source_refs_and_routes_stay_result_and_graph_scoped():
    """Many compiler identities may share a source node without crossing results."""
    source0 = models._NodeRef("Source Model", "forward", "conv")
    source1 = models._NodeRef("Source Model", "decode", "conv")
    result0 = {
        "entities": [
            {"id": "backend/0", "kind": "backend_operator", "child_ids": ["source/0"]},
            {"id": "source/0", "kind": "source_operator"},
        ]
    }
    result1 = {
        "entities": [
            {"id": "backend/0", "kind": "backend_operator", "child_ids": ["source/1"]},
            {"id": "source/1", "kind": "source_operator"},
        ]
    }
    output = {"results": [result0, result1]}
    refs = navigation._related_entity_source_refs(
        output, {"source/0": [source0], "source/1": [source1]}, "backend_operator"
    )
    assert refs == {(0, "backend/0"): (source0,), (1, "backend/0"): (source1,)}
    pte0 = models._NodeRef("PTE", "partition-0", "op")
    trace0 = models._NodeRef("ETDump", "result-0", "op")
    trace1 = models._NodeRef("ETDump", "result-1", "op")
    merged = navigation._merge_entity_ref_maps(
        refs, {(0, "backend/0"): (pte0, trace0), (1, "backend/0"): (trace1,)}
    )
    routes = navigation._sync_navigation_data(None, [], merged)["graphMappings"]
    route0 = next(r for r in routes if r["from"] == source0.to_json())
    assert route0["to"] == {
        "PTE": {"partition-0": ["op"]},
        "ETDump": {"result-0": ["op"], "result-1": []},
        "Source Model": {"decode": []},
    }
    route1 = next(r for r in routes if r["from"] == source1.to_json())
    assert route1["to"] == {
        "ETDump": {"result-0": [], "result-1": ["op"]},
        "PTE": {"partition-0": []},
        "Source Model": {"forward": []},
    }


@pytest.mark.parametrize("other_chain", [False, True])
@pytest.mark.parametrize("compiled_count", [1, 2])
def test_backend_operator_path_replaces_direct_source_chain_path(
    other_chain: bool,
    compiled_count: int,
) -> None:
    """A compiled association refines its own chain while preserving others."""
    entities = {
        "chain-a": {"id": "chain-a", "kind": "chain", "name": "Chain A"},
        "compiled": {
            "id": "compiled",
            "kind": "backend_operator",
            "name": "Conv2D",
            "parent_ids": ["chain-a"],
        },
        "source": {
            "id": "source",
            "kind": "source_operator",
            "name": "conv",
            "parent_ids": ["chain-a", "compiled"],
        },
    }
    if compiled_count == 2:
        entities["compiled-2"] = {
            "id": "compiled-2",
            "kind": "backend_operator",
            "name": "Rescale",
            "parent_ids": ["chain-a"],
        }
        entities["source"]["parent_ids"].append("compiled-2")
    if other_chain:
        entities["chain-b"] = {"id": "chain-b", "kind": "chain", "name": "Chain B"}
        entities["source"]["parent_ids"].append("chain-b")
    hierarchy = models._EntityKindHierarchy(
        ("chain", "backend_operator", "source_operator"), frozenset()
    )
    projection = graphs._entity_hierarchy_complete_paths(
        entities,
        hierarchy,
        {"chain-a", "chain-b", "compiled"},
        presentation=et_integration._ETRecordIntegration.hierarchy_presentation,
    )
    expected = ("Chain A", "Chain B") if other_chain else ("Chain A",)
    _ASSERT.assertEqual(projection.source_operator_paths_by_id["source"], expected)
    _ASSERT.assertNotIn("compiled", projection.entity_paths_by_id)
    _ASSERT.assertEqual(entities["compiled"]["parent_ids"], ["chain-a"])
    generated, _ = graphs._entity_hierarchy_graph(
        {"id": "main", "nodes": [{"id": "source", "label": "Conv2d"}]},
        {"main": "generated"},
        {"source": expected},
    )
    _ASSERT.assertEqual(len(generated["nodes"]), len(expected))
    if not other_chain:
        _ASSERT.assertEqual(generated["nodes"][0]["label"], "Conv2d")


def test_generic_compiler_kind_retains_main_naming_rule() -> None:
    assert (
        graphs._entity_hierarchy_graph_name(
            ("segment", "cascade", "chain", "backend_operator")
        )
        == "segment_cascade_chain_backend_operator"
    )


@pytest.mark.parametrize("suffix", [".vgf", ".pte", ".onnx"])
def test_non_etrecord_adapter_preserves_main_collection_validation(suffix: str) -> None:
    from mlia.model_explorer._artifacts.source_model import (
        _source_model_graph_collection,
    )

    with pytest.raises(RuntimeError, match="multiple graph collections"):
        _source_model_graph_collection(
            Path("model" + suffix),
            {
                "graphCollections": [
                    {"label": "model" + suffix, "graphs": []},
                    {"label": "delegate", "graphs": []},
                ]
            },
        )


def test_etrecord_adapter_rejects_ambiguous_source_collections() -> None:
    from mlia.model_explorer._artifacts.source_model import (
        _source_model_graph_collection,
    )

    with pytest.raises(RuntimeError, match="source collection"):
        _source_model_graph_collection(
            Path("model.etrecord"),
            {
                "graphCollections": [
                    {"label": "first", "graphs": []},
                    {"label": "second", "graphs": []},
                ]
            },
        )


def test_non_etrecord_backend_operator_hierarchy_retains_main_paths() -> None:
    entities = {
        "chain": {"id": "chain", "kind": "chain", "name": "Chain"},
        "backend": {
            "id": "backend",
            "kind": "backend_operator",
            "name": "Conv",
            "parent_ids": ["chain"],
        },
        "source": {
            "id": "source",
            "kind": "source_operator",
            "parent_ids": ["chain", "backend"],
        },
    }
    projection = graphs._entity_hierarchy_complete_paths(
        entities,
        models._EntityKindHierarchy(("chain", "backend_operator", "source_operator")),
        {"chain", "backend"},
    )
    assert projection.source_operator_paths_by_id["source"] == ("Chain", "Chain/Conv")


@pytest.mark.parametrize("prefix", ["", "model.pte::"])
def test_companion_graph_method_is_validated(prefix: str) -> None:
    assert et_integration._companion_graph_matches_method(
        f"{prefix}decode_0_1", "decode"
    )
    assert not et_integration._companion_graph_matches_method(
        f"{prefix}decode_0_1", "forward"
    )
    assert et_integration._companion_graph_matches_method(
        f"{prefix}foo_bar_0_1", "foo_bar"
    )


@pytest.mark.parametrize("graph_id", ["forward", "forward_x_0", "forward_0_x"])
def test_companion_graph_requires_instruction_identity(graph_id: str) -> None:
    assert not et_integration._companion_graph_matches_method(graph_id, "forward")


def test_unavailable_delegate_reports_native_library_reason():
    output = {
        "results": [{"entities": []}],
        "extensions": {
            "executorch": {
                "companion_model": {
                    "delegates": [
                        {
                            "id": "delegate",
                            "method": "forward",
                            "chain_index": 0,
                            "instruction_id": 0,
                        }
                    ],
                    "operator_associations": [
                        {"delegate_id": "delegate", "segment_index": 0, "spirv_id": 42}
                    ],
                }
            }
        },
    }
    collection = {
        "label": "PTE - VGF Backend Hierarchy",
        "graphs": [
            {
                "id": "model.pte::forward_0_0",
                "nodes": [
                    {
                        "id": "unavailable",
                        "label": "segment (unavailable)",
                        "attrs": [
                            {
                                "key": "reason",
                                "value": "SPIRV-Tools shared library is missing",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    from mlia.core.errors import ConfigurationError

    with pytest.raises(
        ConfigurationError, match="SPIRV-Tools shared library is missing"
    ):
        et_integration._pte_embedded_vgf_operator_refs(output, collection)


def test_additional_navigation_blocks_unrelated_same_id_graphs():
    """Repeated SPIR-V IDs must not activate the visualizer's same-ID fallback."""
    collection = {
        "label": "PTE",
        "graphs": [
            {"id": graph_id, "nodes": [{"id": "0__op_42"}]}
            for graph_id in ("forward_0_0", "forward_0_1")
        ],
    }
    first = models._NodeRef("PTE", "forward_0_0", "0__op_42")
    source = models._NodeRef("Source Model", "forward", "conv")
    data = navigation._sync_navigation_data(
        None, [], {(0, "compiler/42"): (first, source)}, [collection]
    )
    for graph_id, other_graph in (
        ("forward_0_0", "forward_0_1"),
        ("forward_0_1", "forward_0_0"),
    ):
        route = next(
            item
            for item in data["graphMappings"]
            if item["from"] == models._NodeRef("PTE", graph_id, "0__op_42").to_json()
        )
        assert route["to"]["PTE"][other_graph] == []


def test_overlay_names_are_shared_across_main_and_supplementary_views(tmp_path):
    """A result missing source links must still reserve its overlay name."""
    output = {
        "results": [
            {
                "kind": "performance",
                "entities": [
                    {
                        "id": "chain",
                        "kind": "chain",
                        "child_ids": ["op"] if index else [],
                    },
                    {"id": "op", "kind": "source_operator"},
                ],
                "breakdowns": [
                    {
                        "entity_id": "chain",
                        "metrics": [{"name": "total_cycles", "value": index + 1}],
                    }
                ],
            }
            for index in range(2)
        ]
    }
    source_ref = models._NodeRef("Source Model", "forward", "conv")
    main = node_data._node_data_overlays(
        output["results"],
        {},
        {
            "label": "Source Model",
            "graphs": [{"id": "forward", "nodes": [{"id": "conv"}]}],
        },
        {"op": [source_ref]},
    )
    supplementary = node_data._entity_metric_overlays(
        output,
        {
            (index, "chain"): (models._NodeRef("Trace", str(index), "Chain"),)
            for index in range(2)
        },
        {"Trace"},
    )
    artifacts = node_data._write_logical_node_data_files(
        [*main, *supplementary], tmp_path, {"Source Model": "source", "Trace": "trace"}
    )
    names = {
        item.display_name
        for item in artifacts
        if item.overlay_key.result_index == 1
        and item.overlay_key.name == "total_cycles"
    }
    assert len(names) == 1
    for item in artifacts:
        assert all(
            graph["name"] == item.display_name
            for graph in json.loads(item.path.read_text()).values()
        )
