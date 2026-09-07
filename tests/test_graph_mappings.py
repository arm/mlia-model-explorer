# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Runtime tests for graph-qualified sync-navigation mappings."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mlia.plugins.analysis import AnalysisRunResult

from mlia.model_explorer import bundler as bundler_module
from mlia.model_explorer._artifacts import graphs as graphs_module
from mlia.model_explorer._artifacts import models as artifact_models
from mlia.model_explorer._artifacts import navigation as navigation_module
from mlia.model_explorer._artifacts import source_model as source_model_module
from mlia.model_explorer.artifacts import ModelExplorerArtifacts, write_artifacts
from mlia.model_explorer.bundler import build_artifacts_bundle


def _node(node_id: str, namespace: str) -> dict:
    result_id = node_id.rsplit("_", 1)[-1]
    return {
        "id": node_id,
        "label": node_id,
        "namespace": namespace,
        "attrs": [{"key": "spv id", "value": f"%{result_id}"}],
    }


def _source_operator(result_id: int, *parent_ids: str) -> dict:
    return {
        "id": f"source_operator/segment_0/spirv-{result_id}",
        "kind": "source_operator",
        "parent_ids": list(parent_ids),
    }


def _result(
    entity_kinds: list[dict],
    entities: list[dict],
    kind: str = "performance",
) -> dict:
    return {
        "kind": kind,
        "entity_kinds": entity_kinds,
        "entities": entities,
    }


def _write_artifacts(
    tmp_path: Path,
    monkeypatch,
    graphs: list[dict],
    results: list[dict],
) -> ModelExplorerArtifacts:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.vgf"
    model.write_bytes(b"model")
    output = {
        "schema_version": "1.2.0",
        "model": {"name": model.name},
        "target": {},
        "backends": [],
        "results": results,
    }
    context = MagicMock()
    context.output_dir = tmp_path / "output"
    context.output_dir.mkdir()
    monkeypatch.setattr(
        source_model_module,
        "_run_model_explorer_adapter",
        lambda _model_path, _adapter_info: {"graphs": graphs},
    )
    run_result = AnalysisRunResult(
        output=output,
        context=context,
        args={"command": "check", "model_explorer": "files"},
        command_name="check",
        parameters={"model": str(model)},
    )
    return write_artifacts(run_result)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_bundle_data(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    marker = '<script id="mlia-model-explorer-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _graph_mappings(generated: ModelExplorerArtifacts) -> list[dict]:
    return _read_json(generated.sync_navigation)["graphMappings"]


def _mapping_by_source(
    generated: ModelExplorerArtifacts,
) -> dict[tuple[str, str, str], dict]:
    return {
        (
            entry["from"]["collectionLabel"],
            entry["from"]["graphId"],
            entry["from"]["nodeId"],
        ): entry["to"]
        for entry in _graph_mappings(generated)
    }


def test_empty_output_emits_empty_graph_mappings(tmp_path: Path, monkeypatch) -> None:
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [],
    )

    assert _read_json(generated.sync_navigation) == {
        "type": "sync_navigation",
        "graphMappings": [],
    }


def test_exact_groups_map_bidirectionally_and_between_generated_graphs(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "outer", "child_kinds": ["inner"]},
        {
            "id": "inner",
            "parent_kinds": ["outer"],
            "child_kinds": ["source_operator"],
        },
        {"id": "peer", "child_kinds": ["source_operator"]},
    ]
    entities: list[dict] = [
        {"id": "outer", "name": "outer", "kind": "outer"},
        {
            "id": "inner",
            "name": "inner",
            "kind": "inner",
            "parent_ids": ["outer"],
        },
        {"id": "peer", "name": "peer", "kind": "peer"},
        _source_operator(0, "inner", "peer"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [_result(entity_kinds, entities)],
    )
    mappings = _mapping_by_source(generated)

    source_to = mappings[("Source Model", "main", "source___group___")]
    assert source_to == {
        "outer_inner": {
            "outer_inner-main": ["outer/inner___group___"],
        },
        "peer": {"peer-main": ["peer___group___"]},
    }
    inner_to = mappings[("outer_inner", "outer_inner-main", "outer/inner___group___")]
    assert inner_to == {
        "Source Model": {"main": ["source___group___"]},
        "peer": {"peer-main": ["peer___group___"]},
    }
    peer_to = mappings[("peer", "peer-main", "peer___group___")]
    assert peer_to["outer_inner"] == {"outer_inner-main": ["outer/inner___group___"]}


def test_deepest_exact_target_is_selected_for_single_child_chain(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "outer", "child_kinds": ["inner"]},
        {
            "id": "inner",
            "parent_kinds": ["outer"],
            "child_kinds": ["source_operator"],
        },
    ]
    entities: list[dict] = [
        {"id": "outer", "name": "outer", "kind": "outer"},
        {
            "id": "inner",
            "name": "inner",
            "kind": "inner",
            "parent_ids": ["outer"],
        },
        _source_operator(0, "inner"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [_result(entity_kinds, entities)],
    )

    source_to = _mapping_by_source(generated)[
        ("Source Model", "main", "source___group___")
    ]
    assert source_to["outer_inner"]["outer_inner-main"] == ["outer/inner___group___"]


def test_leaf_list_fallback_preserves_target_graph_order(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities = [
        {"id": "first", "name": "first", "kind": "partition"},
        {"id": "second", "name": "second", "kind": "partition"},
        _source_operator(1, "first"),
        _source_operator(0, "second"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    _node("0__op_0", "source"),
                    _node("0__op_1", "source"),
                ],
            }
        ],
        [_result(entity_kinds, entities)],
    )

    source_to = _mapping_by_source(generated)[
        ("Source Model", "main", "source___group___")
    ]
    assert source_to == {
        "partition": {
            "partition-main": ["0__op_0", "0__op_1"],
        }
    }


def test_same_ids_in_different_source_graphs_have_explicit_empty_routes(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities = [
        {"id": "partition", "name": "partition", "kind": "partition"},
        _source_operator(0, "partition"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {"id": "first/subgraph", "nodes": [_node("0__op_0", "shared")]},
            {"id": r"first\subgraph", "nodes": [_node("0__op_0", "shared")]},
        ],
        [_result(entity_kinds, entities)],
    )
    mappings = _mapping_by_source(generated)

    assert mappings[("Source Model", "first/subgraph", "shared___group___")] == {
        "partition": {
            "partition-first/subgraph": ["partition___group___"],
            r"partition-first\subgraph": [],
        },
        "Source Model": {r"first\subgraph": []},
    }
    assert mappings[("Source Model", r"first\subgraph", "shared___group___")] == {
        "Source Model": {"first/subgraph": []},
        "partition": {
            "partition-first/subgraph": [],
            r"partition-first\subgraph": ["partition___group___"],
        },
    }
    assert mappings[
        ("partition", "partition-first/subgraph", "partition___group___")
    ] == {
        "Source Model": {
            "first/subgraph": ["shared___group___"],
            r"first\subgraph": [],
        },
        "partition": {r"partition-first\subgraph": []},
    }

    projection_identities = {
        ("Source Model", "first/subgraph"),
        ("partition", "partition-first/subgraph"),
        ("Source Model", r"first\subgraph"),
        ("partition", r"partition-first\subgraph"),
    }
    for source, targets in mappings.items():
        target_identities = [
            (collection_label, graph_id)
            for collection_label, graphs_by_id in targets.items()
            for graph_id in graphs_by_id
        ]
        assert target_identities == list(dict.fromkeys(target_identities))
        assert set(target_identities) == projection_identities - {source[:2]}


def test_repeated_group_ids_across_results_stay_collection_qualified(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {"id": "partition", "name": "same", "kind": "partition"},
            _source_operator(0, "partition"),
        ],
    )
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "same")]}],
        [result, {**result, "kind": "memory"}],
    )
    assert [path.name for path in generated.generated_models] == [
        "model-hierarchy-partition-result-1.json",
        "model-hierarchy-partition-result-2.json",
    ]
    mappings = _mapping_by_source(generated)

    source_to = mappings[("Source Model", "main", "same___group___")]
    assert source_to == {
        "partition [performance result 1]": {
            "partition-result-1-main": ["same___group___"],
        },
        "partition [memory result 2]": {
            "partition-result-2-main": ["same___group___"],
        },
    }
    first_generated_to = mappings[
        (
            "partition [performance result 1]",
            "partition-result-1-main",
            "same___group___",
        )
    ]
    assert first_generated_to == {
        "Source Model": {"main": ["same___group___"]},
        "partition [memory result 2]": {
            "partition-result-2-main": ["same___group___"],
        },
    }


def test_escaped_namespace_paths_are_preserved(tmp_path: Path, monkeypatch) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities = [
        {"id": "partition", "name": "target", "kind": "partition"},
        _source_operator(0, "partition"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [_node("0__op_0", r"source\/part/child")],
            }
        ],
        [_result(entity_kinds, entities)],
    )
    source_ids = [
        entry["from"]["nodeId"]
        for entry in _graph_mappings(generated)
        if entry["from"]["collectionLabel"] == "Source Model"
        and entry["from"]["nodeId"].endswith("___group___")
    ]

    assert source_ids == [
        r"source\/part___group___",
        r"source\/part/child___group___",
    ]


def test_graph_mappings_include_leaf_sources_and_are_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities = [
        {"id": "partition", "name": "target", "kind": "partition"},
        _source_operator(0, "partition"),
        _source_operator(1, "partition"),
    ]
    graphs = [
        {
            "id": "main",
            "nodes": [
                _node("0__op_0", "source"),
                _node("0__op_1", "source"),
            ],
        }
    ]
    first = _write_artifacts(
        tmp_path / "first", monkeypatch, graphs, [_result(entity_kinds, entities)]
    )
    second = _write_artifacts(
        tmp_path / "second", monkeypatch, graphs, [_result(entity_kinds, entities)]
    )

    first_data = _read_json(first.sync_navigation)
    assert first_data == _read_json(second.sync_navigation)
    assert set(first_data) == {"type", "graphMappings"}
    assert first_data["type"] == "sync_navigation"
    assert all(set(entry) == {"from", "to"} for entry in first_data["graphMappings"])
    assert any(
        not entry["from"]["nodeId"].endswith("___group___")
        for entry in first_data["graphMappings"]
    )
    source_refs = [
        (
            entry["from"]["collectionLabel"],
            entry["from"]["graphId"],
            entry["from"]["nodeId"],
        )
        for entry in first_data["graphMappings"]
    ]
    assert len(source_refs) == len(set(source_refs))
    assert all(
        len(node_ids) == len(dict.fromkeys(node_ids))
        for entry in first_data["graphMappings"]
        for graphs_by_id in entry["to"].values()
        for node_ids in graphs_by_id.values()
    )


def test_bundle_contains_graph_mappings_artifact_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities: list[dict] = [
        {"id": "partition", "name": "target", "kind": "partition"},
        _source_operator(0, "partition"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [_result(entity_kinds, entities)],
    )
    expected = _read_json(generated.sync_navigation)

    index = build_artifacts_bundle(generated.root)
    html = index.read_text(encoding="utf-8")
    marker = '<script id="mlia-model-explorer-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    bundle_data = json.loads(html[start:end])

    assert bundle_data["visualizerConfig"]["syncNavigationData"] == expected


def test_colliding_hierarchy_kind_paths_get_distinct_artifact_identities(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "a", "child_kinds": ["b"]},
        {
            "id": "b",
            "parent_kinds": ["a"],
            "child_kinds": ["source_operator"],
        },
        {"id": "a_b", "child_kinds": ["source_operator"]},
    ]
    entities: list[dict] = [
        {"id": "a", "name": "a", "kind": "a"},
        {"id": "b", "name": "b", "kind": "b", "parent_ids": ["a"]},
        {"id": "a_b", "name": "a_b", "kind": "a_b"},
        _source_operator(0, "b", "a_b"),
    ]

    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [_result(entity_kinds, entities)],
    )

    assert [path.name for path in generated.generated_models] == [
        "model-hierarchy-a_b-result-1-hierarchy-1.json",
        "model-hierarchy-a_b-result-1-hierarchy-2.json",
    ]
    collections = [_read_json(path) for path in generated.generated_models]
    assert [collection["label"] for collection in collections] == [
        "a_b [performance result 1, hierarchy 1: a > b]",
        "a_b [performance result 1, hierarchy 2: a_b]",
    ]
    assert [collection["graphs"][0]["id"] for collection in collections] == [
        "a_b-result-1-hierarchy-1-main",
        "a_b-result-1-hierarchy-2-main",
    ]
    source_targets = _mapping_by_source(generated)[
        ("Source Model", "main", "source___group___")
    ]
    assert set(source_targets) == {
        "a_b [performance result 1, hierarchy 1: a > b]",
        "a_b [performance result 1, hierarchy 2: a_b]",
    }


def test_generated_subgraph_links_and_all_cross_artifact_ids_are_rewritten(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {"id": "partition", "name": "partition", "kind": "partition"},
            _source_operator(0, "partition"),
            _source_operator(1, "partition"),
            _source_operator(2, "partition"),
        ],
    )
    result["breakdowns"] = [
        {
            "entity_id": "partition",
            "metrics": [{"name": "cycles", "value": 7}],
        }
    ]
    graphs = [
        {
            "id": "root/graph",
            "nodes": [
                {
                    **_node("0__op_0", "source/root"),
                    "subgraphIds": [r"child\graph", "nested"],
                }
            ],
        },
        {
            "id": r"child\graph",
            "nodes": [
                {
                    **_node("0__op_1", "source/child"),
                    "subgraphIds": ["nested"],
                }
            ],
        },
        {
            "id": "nested",
            "nodes": [_node("0__op_2", "source/nested")],
        },
    ]

    generated = _write_artifacts(tmp_path, monkeypatch, graphs, [result])
    generated_collection = _read_json(generated.generated_models[0])
    generated_graphs = {graph["id"]: graph for graph in generated_collection["graphs"]}

    assert set(generated_graphs) == {
        "partition-root/graph",
        r"partition-child\graph",
        "partition-nested",
    }
    assert generated_graphs["partition-root/graph"]["nodes"][0]["subgraphIds"] == [
        r"partition-child\graph",
        "partition-nested",
    ]
    assert generated_graphs[r"partition-child\graph"]["nodes"][0]["subgraphIds"] == [
        "partition-nested"
    ]

    mappings = _read_json(generated.entity_node_mappings)["results"][0]["mappings"]
    generated_leaf_graph_ids = {
        mapping["node"]["graphId"]
        for mapping in mappings
        if mapping["node"]["collectionLabel"] == "partition"
        and mapping["node"]["nodeId"].startswith("0__op_")
    }
    assert generated_leaf_graph_ids == set(generated_graphs)

    cycles = _read_json(
        generated.root / "node-data" / "model-hierarchy-partition" / "cycles.json"
    )
    assert set(cycles) == set(generated_graphs)
    assert all(
        data["results"] == {"partition": {"value": 7}} for data in cycles.values()
    )

    index = build_artifacts_bundle(generated.root)
    bundle_data = _read_bundle_data(index)
    assert bundle_data["initialUiState"]["paneStates"][0]["selectedGraphId"] == (
        "root/graph"
    )
    assert bundle_data["initialUiState"]["paneStates"][1]["selectedGraphId"] == (
        "partition-root/graph"
    )
    sync_targets = _mapping_by_source(generated)[
        ("Source Model", r"child\graph", "source___group___")
    ]
    assert sync_targets == {
        "Source Model": {"root/graph": [], "nested": []},
        "partition": {
            "partition-root/graph": [],
            r"partition-child\graph": ["partition___group___"],
            "partition-nested": [],
        },
    }


@pytest.mark.parametrize(
    ("namespace", "expected_groups"),
    [
        ("legacy;actual/child", ["actual", "actual/child"]),
        (";legacy;;actual/child;", ["actual", "actual/child"]),
        (";actual", [";actual"]),
        ("actual;", ["actual;"]),
        (";", [";"]),
        (r"legacy;actual\/part/child", [r"actual\/part", r"actual\/part/child"]),
    ],
)
def test_source_namespace_projection_matches_model_explorer_normalization(
    tmp_path: Path,
    monkeypatch,
    namespace: str,
    expected_groups: list[str],
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities = [
        {"id": "partition", "name": "target", "kind": "partition"},
        _source_operator(0, "partition"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", namespace)]}],
        [_result(entity_kinds, entities)],
    )

    source_group_ids = [
        entry["from"]["nodeId"].removesuffix("___group___")
        for entry in _graph_mappings(generated)
        if entry["from"]["collectionLabel"] == "Source Model"
        and entry["from"]["nodeId"].endswith("___group___")
    ]
    assert source_group_ids == expected_groups


@pytest.mark.parametrize(
    ("outer_name", "expected_groups"),
    [
        ("legacy;actual", ["actual", "actual/child"]),
        (";legacy;;actual", ["actual", "actual/child"]),
        (";actual", [";actual", ";actual/child"]),
        ("actual;", ["child"]),
        (";", [";", ";/child"]),
        ("legacy;actual/part", [r"actual\/part", r"actual\/part/child"]),
    ],
)
def test_generated_namespace_references_match_runtime_normalization(
    tmp_path: Path,
    monkeypatch,
    outer_name: str,
    expected_groups: list[str],
) -> None:
    entity_kinds = [
        {"id": "outer", "child_kinds": ["inner"]},
        {
            "id": "inner",
            "parent_kinds": ["outer"],
            "child_kinds": ["source_operator"],
        },
    ]
    result = _result(
        entity_kinds,
        [
            {"id": "outer", "name": outer_name, "kind": "outer"},
            {
                "id": "inner",
                "name": "child",
                "kind": "inner",
                "parent_ids": ["outer"],
            },
            _source_operator(0, "inner"),
        ],
    )
    result["breakdowns"] = [
        {"entity_id": "inner", "metrics": [{"name": "cycles", "value": 9}]}
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [result],
    )
    expected_group_ids = {f"{namespace}___group___" for namespace in expected_groups}

    generated_from_ids = {
        entry["from"]["nodeId"]
        for entry in _graph_mappings(generated)
        if entry["from"]["collectionLabel"] == "outer_inner"
        and entry["from"]["nodeId"].endswith("___group___")
    }
    assert generated_from_ids == expected_group_ids
    entity_mappings = _read_json(generated.entity_node_mappings)["results"][0][
        "mappings"
    ]
    mapped_group_ids = {
        mapping["node"]["nodeId"]
        for mapping in entity_mappings
        if mapping["node"]["collectionLabel"] == "outer_inner"
        and mapping["node"]["nodeId"].endswith("___group___")
    }
    assert mapped_group_ids <= expected_group_ids
    assert f"{expected_groups[-1]}___group___" in mapped_group_ids
    cycles = _read_json(
        generated.root / "node-data" / "model-hierarchy-outer_inner" / "cycles.json"
    )
    assert cycles["outer_inner-main"]["results"] == {
        expected_groups[-1]: {"value": 9}
    }


def test_normalized_entity_sibling_collisions_stay_distinct_across_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {
                "id": "legacy-entity",
                "name": "legacy;actual",
                "kind": "partition",
            },
            {"id": "actual-entity", "name": "actual", "kind": "partition"},
            _source_operator(0, "legacy-entity"),
            _source_operator(1, "actual-entity"),
        ],
    )
    result["breakdowns"] = [
        {
            "entity_id": "legacy-entity",
            "metrics": [{"name": "cycles", "value": 10}],
        },
        {
            "entity_id": "actual-entity",
            "metrics": [{"name": "cycles", "value": 20}],
        },
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    _node("0__op_0", "source"),
                    _node("0__op_1", "source"),
                ],
            }
        ],
        [result],
    )

    generated_collection = _read_json(generated.generated_models[0])
    generated_graph = generated_collection["graphs"][0]
    assert [node["namespace"] for node in generated_graph["nodes"]] == [
        "legacy;actual [legacy-entity]",
        "actual [actual-entity]",
    ]
    generated_group_ids = {
        entry["from"]["nodeId"]
        for entry in _graph_mappings(generated)
        if entry["from"]["collectionLabel"] == "partition"
        and entry["from"]["nodeId"].endswith("___group___")
    }
    assert generated_group_ids == {
        "actual [legacy-entity]___group___",
        "actual [actual-entity]___group___",
    }

    mappings = _mapping_by_source(generated)
    assert mappings[("Source Model", "main", "source___group___")]["partition"] == {
        "partition-main": ["0__op_0", "0__op_1"]
    }
    assert mappings[
        ("partition", "partition-main", "actual [legacy-entity]___group___")
    ]["Source Model"] == {"main": ["0__op_0"]}
    assert mappings[
        ("partition", "partition-main", "actual [actual-entity]___group___")
    ]["Source Model"] == {"main": ["0__op_1"]}

    entity_mappings = _read_json(generated.entity_node_mappings)["results"][0][
        "mappings"
    ]
    mapped_entities_by_node = {
        mapping["node"]["nodeId"]: mapping["entityId"]
        for mapping in entity_mappings
        if mapping["node"]["collectionLabel"] == "partition"
        and mapping["node"]["nodeId"].endswith("___group___")
    }
    assert mapped_entities_by_node == {
        "actual [legacy-entity]___group___": "legacy-entity",
        "actual [actual-entity]___group___": "actual-entity",
    }
    cycles = _read_json(
        generated.root / "node-data" / "model-hierarchy-partition" / "cycles.json"
    )
    assert cycles["partition-main"]["results"] == {
        "actual [actual-entity]": {"value": 20},
        "actual [legacy-entity]": {"value": 10},
    }


def test_cross_parent_runtime_collisions_stay_distinct_across_all_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "outer", "child_kinds": ["inner"]},
        {
            "id": "inner",
            "parent_kinds": ["outer"],
            "child_kinds": ["source_operator"],
        },
    ]
    entities: list[dict] = [
        {"id": "parent-1", "name": "parent1", "kind": "outer"},
        {"id": "parent-2", "name": "parent2", "kind": "outer"},
        {
            "id": "child/1",
            "name": "x;shared",
            "kind": "inner",
            "parent_ids": ["parent-1"],
        },
        {
            "id": "child;2",
            "name": "y;shared",
            "kind": "inner",
            "parent_ids": ["parent-2"],
        },
        _source_operator(0, "child/1"),
        _source_operator(1, "child;2"),
    ]
    breakdowns = [
        {"entity_id": "parent-1", "metrics": [{"name": "cycles", "value": 1}]},
        {"entity_id": "child/1", "metrics": [{"name": "cycles", "value": 10}]},
        {"entity_id": "parent-2", "metrics": [{"name": "cycles", "value": 2}]},
        {"entity_id": "child;2", "metrics": [{"name": "cycles", "value": 20}]},
    ]
    result = _result(entity_kinds, entities)
    result["breakdowns"] = breakdowns
    graphs = [
        {
            "id": "main",
            "nodes": [
                _node("0__op_0", "source"),
                _node("0__op_1", "source"),
            ],
        }
    ]
    generated = _write_artifacts(tmp_path / "first", monkeypatch, graphs, [result])
    reordered_result = _result(entity_kinds, list(reversed(entities)))
    reordered_result["breakdowns"] = list(reversed(breakdowns))
    reordered = _write_artifacts(
        tmp_path / "reordered", monkeypatch, graphs, [reordered_result]
    )

    expected_paths = {
        "parent1",
        "parent1/x%3Bshared [child%2F1]",
        "parent2",
        "parent2/y%3Bshared [child%3B2]",
    }
    generated_collection = _read_json(generated.generated_models[0])
    generated_graph = generated_collection["graphs"][0]
    assert {node["namespace"] for node in generated_graph["nodes"]} == {
        "parent1/x%3Bshared [child%2F1]",
        "parent2/y%3Bshared [child%3B2]",
    }
    assert _read_json(reordered.generated_models[0]) == generated_collection

    cycles = _read_json(
        generated.root / "node-data" / "model-hierarchy-outer_inner" / "cycles.json"
    )
    assert cycles["outer_inner-main"]["results"] == {
        "parent1": {"value": 1},
        "parent1/x%3Bshared [child%2F1]": {"value": 10},
        "parent2": {"value": 2},
        "parent2/y%3Bshared [child%3B2]": {"value": 20},
    }
    reordered_cycles = _read_json(
        reordered.root
        / "node-data"
        / "model-hierarchy-outer_inner"
        / "cycles.json"
    )
    assert reordered_cycles == cycles

    entity_mappings = _read_json(generated.entity_node_mappings)["results"][0][
        "mappings"
    ]
    mapped_groups = {
        mapping["entityId"]: mapping["node"]["nodeId"]
        for mapping in entity_mappings
        if mapping["node"]["collectionLabel"] == "outer_inner"
        and mapping["node"]["nodeId"].endswith("___group___")
    }
    assert mapped_groups == {
        "child/1": "parent1/x%3Bshared [child%2F1]___group___",
        "child;2": "parent2/y%3Bshared [child%3B2]___group___",
        "parent-1": "parent1___group___",
        "parent-2": "parent2___group___",
    }
    assert _read_json(reordered.entity_node_mappings) == _read_json(
        generated.entity_node_mappings
    )

    generated_routes = {
        entry["from"]["nodeId"].removesuffix("___group___"): entry["to"][
            "Source Model"
        ]["main"]
        for entry in _graph_mappings(generated)
        if entry["from"]["collectionLabel"] == "outer_inner"
        and entry["from"]["nodeId"].endswith("___group___")
    }
    assert set(generated_routes) == expected_paths
    assert generated_routes == {
        "parent1": ["0__op_0"],
        "parent1/x%3Bshared [child%2F1]": ["0__op_0"],
        "parent2": ["0__op_1"],
        "parent2/y%3Bshared [child%3B2]": ["0__op_1"],
    }
    assert _read_json(reordered.sync_navigation) == _read_json(
        generated.sync_navigation
    )


def test_secondary_cross_branch_collisions_reallocate_production_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "outer", "child_kinds": ["inner"]},
        {
            "id": "inner",
            "parent_kinds": ["outer"],
            "child_kinds": ["source_operator"],
        },
    ]
    entities: list[dict] = [
        {"id": "p1", "name": "parent1", "kind": "outer"},
        {"id": "p2", "name": "parent2", "kind": "outer"},
        {"id": "p3", "name": "parent3", "kind": "outer"},
        {"id": "a", "name": "x;shared", "kind": "inner", "parent_ids": ["p1"]},
        {"id": "b", "name": "y;shared", "kind": "inner", "parent_ids": ["p2"]},
        {"id": "c", "name": "z;shared", "kind": "inner", "parent_ids": ["p3"]},
        {
            "id": "d",
            "name": "x%3Bshared [a]",
            "kind": "inner",
            "parent_ids": ["p1"],
        },
        {
            "id": "e",
            "name": "x%3Bshared [a entity-id]",
            "kind": "inner",
            "parent_ids": ["p1"],
        },
        *[
            _source_operator(index, entity_id)
            for index, entity_id in enumerate("abcde")
        ],
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [_node(f"0__op_{index}", "source") for index in range(5)],
            }
        ],
        [_result(entity_kinds, entities)],
    )

    generated_graph = _read_json(generated.generated_models[0])["graphs"][0]
    assert [node["namespace"] for node in generated_graph["nodes"]] == [
        "parent1/MLIA entity a",
        "parent2/y%3Bshared [b]",
        "parent3/z%3Bshared [c]",
        "parent1/x%3Bshared [a] [d]",
        "parent1/x%3Bshared [a entity-id] [e]",
    ]
    generated_routes = {
        entry["from"]["nodeId"].removesuffix("___group___"): entry["to"][
            "Source Model"
        ]["main"]
        for entry in _graph_mappings(generated)
        if entry["from"]["collectionLabel"] == "outer_inner"
        and entry["from"]["nodeId"].endswith("___group___")
    }
    assert generated_routes == {
        "parent1": ["0__op_0", "0__op_3", "0__op_4"],
        "parent1/MLIA entity a": ["0__op_0"],
        "parent1/x%3Bshared [a entity-id] [e]": ["0__op_4"],
        "parent1/x%3Bshared [a] [d]": ["0__op_3"],
        "parent2": ["0__op_1"],
        "parent2/y%3Bshared [b]": ["0__op_1"],
        "parent3": ["0__op_2"],
        "parent3/z%3Bshared [c]": ["0__op_2"],
    }


def test_serialized_group_suffix_node_ids_remain_canonical_leaves(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        source_model_module,
        "_source_operator_node_ids",
        lambda _model_path, _graph_collection: {
            "source_operator/segment_0/spirv-0": [
                artifact_models._NodeRef("Source Model", "main", "leaf___group___")
            ],
            "source_operator/segment_0/spirv-1": [
                artifact_models._NodeRef("Source Model", "main", "ordinary")
            ],
        },
    )
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {"id": "suffix", "name": "target suffix", "kind": "partition"},
            {"id": "ordinary", "name": "target ordinary", "kind": "partition"},
            _source_operator(0, "suffix"),
            _source_operator(1, "ordinary"),
        ],
    )
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    {
                        **_node("leaf___group___", "source/enclosing"),
                        "attrs": [{"key": "spv id", "value": "%0"}],
                    },
                    {
                        **_node("ordinary", "source/enclosing"),
                        "attrs": [{"key": "spv id", "value": "%1"}],
                    },
                ],
            }
        ],
        [result],
    )

    source_projection = graphs_module._graph_projection(
        ("Source Model", "main"),
        "Source Model",
        _read_json(generated.root / "models" / "model-source.json")["graphs"][0],
    )
    assert [ref.node_id for ref in source_projection.ordered_leaf_refs] == [
        "leaf___group___",
        "ordinary",
    ]
    assert {
        group.node_ref.node_id: group.ordered_leaf_ids
        for group in source_projection.groups
    } == {
        "source___group___": ("leaf___group___", "ordinary"),
        "source/enclosing___group___": ("leaf___group___", "ordinary"),
    }

    mappings = _mapping_by_source(generated)
    for source_group_id in (
        "source___group___",
        "source/enclosing___group___",
    ):
        assert mappings[("Source Model", "main", source_group_id)]["partition"] == {
            "partition-main": ["leaf___group___", "ordinary"]
        }
    assert mappings[
        ("partition", "partition-main", "target suffix___group___")
    ]["Source Model"] == {"main": ["leaf___group___"]}
    assert mappings[
        ("partition", "partition-main", "target ordinary___group___")
    ]["Source Model"] == {"main": ["ordinary"]}

    entity_mappings = _read_json(generated.entity_node_mappings)["results"][0][
        "mappings"
    ]
    mapped_nodes = {
        (
            mapping["node"]["collectionLabel"],
            mapping["node"]["nodeId"],
            mapping["entityId"],
        )
        for mapping in entity_mappings
    }
    assert (
        "Source Model",
        "leaf___group___",
        "source_operator/segment_0/spirv-0",
    ) in mapped_nodes
    assert (
        "partition",
        "leaf___group___",
        "source_operator/segment_0/spirv-0",
    ) in mapped_nodes
    assert ("partition", "target suffix___group___", "suffix") in mapped_nodes


def test_serialized_node_id_colliding_with_runtime_group_is_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "Serialized graph node IDs collide with Model Explorer runtime-created "
            "namespace group IDs.*source___group___"
        ),
    ):
        graphs_module._graph_projection(
            ("Source Model", "main"),
            "Source Model",
            {
                "id": "main",
                "nodes": [
                    {
                        "id": "source___group___",
                        "label": "ambiguous",
                        "namespace": "source",
                    }
                ],
            },
        )


def test_duplicate_source_graph_ids_are_rejected_during_artifact_generation(
    tmp_path: Path, monkeypatch
) -> None:
    result = _result(
        [{"id": "partition", "child_kinds": ["source_operator"]}],
        [
            {"id": "partition", "kind": "partition"},
            _source_operator(0, "partition"),
        ],
    )
    graphs = [
        {"id": "main", "nodes": [_node("0__op_0", "source")]},
        {"id": "main", "nodes": [_node("0__op_1", "source")]},
    ]

    with pytest.raises(
        RuntimeError, match="Duplicate graph ID 'main' in source graphs"
    ):
        _write_artifacts(tmp_path, monkeypatch, graphs, [result])


def test_duplicate_node_ids_are_rejected_during_mapping_generation(
    tmp_path: Path, monkeypatch
) -> None:
    result = _result(
        [{"id": "partition", "child_kinds": ["source_operator"]}],
        [
            {"id": "partition", "kind": "partition"},
            _source_operator(0, "partition"),
        ],
    )
    graph = {
        "id": "main",
        "nodes": [
            _node("0__op_0", "source"),
            _node("0__op_0", "other"),
        ],
    }

    with pytest.raises(RuntimeError, match="Duplicate node ID '0__op_0'"):
        _write_artifacts(tmp_path, monkeypatch, [graph], [result])


def test_missing_generated_projection_is_rejected() -> None:
    hierarchy = artifact_models._EntityKindHierarchy(("partition", "source_operator"))
    descriptor = artifact_models._GeneratedEntityHierarchy(
        0, "performance", hierarchy, "partition", "partition"
    )
    artifact = artifact_models._GeneratedModelArtifact(
        Path("partition.json"),
        descriptor,
        {"label": "partition", "graphs": []},
        artifact_models._GeneratedHierarchyProjection(
            {}, {}, {}, {}, frozenset()
        ),
    )
    source = {
        "label": "Source Model",
        "graphs": [{"id": "main", "nodes": [{"id": "0", "namespace": "source"}]}],
    }

    with pytest.raises(RuntimeError, match="missing the projection.*partition-main"):
        navigation_module._sync_navigation_data(source, [artifact])


def test_missing_target_canonical_leaf_is_rejected() -> None:
    origin = ("Source Model", "main")
    source = graphs_module._graph_projection(
        origin,
        "Source Model",
        {
            "id": "main",
            "nodes": [
                {"id": "a", "namespace": "source"},
                {"id": "b", "namespace": "source"},
            ],
        },
    )
    target = graphs_module._graph_projection(
        origin,
        "target",
        {"id": "target-main", "nodes": [{"id": "a", "namespace": "target"}]},
    )

    with pytest.raises(RuntimeError, match="missing canonical target leaves.*'b'"):
        navigation_module._graph_mapping_target_ids(source.groups[0], target)


def test_repeated_deepest_exact_groups_are_all_returned() -> None:
    canonical_ref = artifact_models._NodeRef("Source Model", "main", "a")
    source_group = artifact_models._GraphGroup(
        artifact_models._NodeRef("source", "main", "source___group___"),
        1,
        ("a",),
        frozenset({canonical_ref}),
    )
    target = artifact_models._GraphProjection(
        origin=("Source Model", "main"),
        collection_label="target",
        graph_id="target-main",
        ordered_leaf_refs=(canonical_ref,),
        ordered_leaf_node_ids=("a",),
        leaf_ids_by_canonical_ref={canonical_ref: ("a",)},
        groups=(
            artifact_models._GraphGroup(
                artifact_models._NodeRef("target", "target-main", "x___group___"),
                2,
                ("a",),
                frozenset({canonical_ref}),
            ),
            artifact_models._GraphGroup(
                artifact_models._NodeRef("target", "target-main", "y___group___"),
                2,
                ("a",),
                frozenset({canonical_ref}),
            ),
        ),
        terminal_complete_path_group_ids=frozenset(
            {"x___group___", "y___group___"}
        ),
    )

    assert navigation_module._graph_mapping_target_ids(source_group, target) == [
        "x___group___",
        "y___group___",
    ]


def test_unequal_depth_complete_paths_all_receive_sync_navigation_targets(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {
            "id": "layer",
            "parent_kinds": ["layer"],
            "child_kinds": ["layer", "source_operator"],
        }
    ]
    entities = [
        {"id": "root", "name": "Root", "kind": "layer"},
        {
            "id": "deep",
            "name": "Deep",
            "kind": "layer",
            "parent_ids": ["root"],
        },
        {"id": "shallow", "name": "Shallow", "kind": "layer"},
        _source_operator(0, "deep", "shallow"),
    ]

    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [_result(entity_kinds, entities)],
    )

    target_ids = _mapping_by_source(generated)[
        ("Source Model", "main", "source___group___")
    ]["layer"]["layer-main"]
    assert target_ids == [
        "Root/Deep___group___",
        "Shallow___group___",
    ]


def test_generated_projection_requires_explicit_complete_canonical_mapping() -> None:
    origin = ("Source Model", "main")
    graph = {
        "id": "generated-main",
        "nodes": [{"id": "generated", "namespace": "target"}],
    }

    with pytest.raises(
        RuntimeError,
        match="canonical mapping is incomplete.*never treated as canonical",
    ):
        graphs_module._graph_projection(origin, "generated", graph, {})

    generated_ref = artifact_models._NodeRef(
        "generated", "generated-main", "generated"
    )
    unexpected_ref = artifact_models._NodeRef(
        "generated", "generated-main", "unexpected"
    )
    canonical_ref = artifact_models._NodeRef("Source Model", "main", "source")
    with pytest.raises(RuntimeError, match="contains unexpected nodes"):
        graphs_module._graph_projection(
            origin,
            "generated",
            graph,
            {
                generated_ref: canonical_ref,
                unexpected_ref: canonical_ref,
            },
        )


def test_generated_artifact_rejects_inconsistent_forward_canonical_mapping() -> None:
    hierarchy = artifact_models._EntityKindHierarchy(("partition", "source_operator"))
    descriptor = artifact_models._GeneratedEntityHierarchy(
        0, "performance", hierarchy, "partition", "partition"
    )
    source_ref = artifact_models._NodeRef("Source Model", "main", "source")
    generated_ref = artifact_models._NodeRef(
        "partition", "partition-main", "generated"
    )
    artifact = artifact_models._GeneratedModelArtifact(
        Path("partition.json"),
        descriptor,
        {
            "label": "partition",
            "graphs": [
                {
                    "id": "partition-main",
                    "nodes": [{"id": "generated", "namespace": "target"}],
                }
            ],
        },
        artifact_models._GeneratedHierarchyProjection(
            generated_refs_by_canonical_ref={},
            canonical_ref_by_generated_ref={generated_ref: source_ref},
            source_operator_by_generated_ref={},
            group_refs_by_entity_id={},
            terminal_complete_path_group_refs=frozenset(),
        ),
    )
    source = {
        "label": "Source Model",
        "graphs": [
            {"id": "main", "nodes": [{"id": "source", "namespace": "source"}]}
        ],
    }

    with pytest.raises(RuntimeError, match="forward canonical mapping"):
        navigation_module._sync_navigation_data(source, [artifact])


def test_split_self_edge_is_rejected_with_targeted_error(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {"id": "a", "name": "A", "kind": "partition"},
            {"id": "b", "name": "B", "kind": "partition"},
            _source_operator(0, "a", "b"),
        ],
    )
    graph = {
        "id": "main",
        "nodes": [
            {
                **_node("0__op_0", "source"),
                "incomingEdges": [{"sourceNodeId": "0__op_0"}],
            }
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="Cannot split self-referential source node '0__op_0'.*linear",
    ):
        _write_artifacts(tmp_path, monkeypatch, [graph], [result])


def test_single_path_self_edge_is_preserved(tmp_path: Path, monkeypatch) -> None:
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {"id": "a", "name": "A", "kind": "partition"},
            _source_operator(0, "a"),
        ],
    )
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    {
                        **_node("0__op_0", "source"),
                        "incomingEdges": [{"sourceNodeId": "0__op_0"}],
                    }
                ],
            }
        ],
        [result],
    )

    node = _read_json(generated.generated_models[0])["graphs"][0]["nodes"][0]
    assert node["id"] == "0__op_0"
    assert node["incomingEdges"] == [{"sourceNodeId": "0__op_0"}]


def test_distinct_node_cycle_is_preserved_without_extra_rewritten_edges() -> None:
    graph, generated_ids = graphs_module._entity_hierarchy_graph(
        {
            "id": "main",
            "nodes": [
                {"id": "a", "incomingEdges": [{"sourceNodeId": "b"}]},
                {"id": "b", "incomingEdges": [{"sourceNodeId": "a"}]},
            ],
        },
        {"main": "hierarchy-main"},
        {"a": ("A1", "A2"), "b": ("B1", "B2")},
    )

    nodes = {node["id"]: node for node in graph["nodes"]}
    a_first, a_last = generated_ids["a"]
    b_first, b_last = generated_ids["b"]
    assert nodes[a_first]["incomingEdges"] == [{"sourceNodeId": b_last}]
    assert nodes[a_last]["incomingEdges"] == [{"sourceNodeId": a_first}]
    assert nodes[b_first]["incomingEdges"] == [{"sourceNodeId": a_last}]
    assert nodes[b_last]["incomingEdges"] == [{"sourceNodeId": b_first}]
    assert sum(len(node.get("incomingEdges", [])) for node in nodes.values()) == 4


def test_sub_operator_ids_avoid_source_collisions_and_remain_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    long_node_id = "canonical-" + ("x" * 500)
    colliding_source_id = graphs_module._sub_operator_node_id(
        "main", long_node_id, "A"
    )
    source_operator_id = "source_operator/custom"
    monkeypatch.setattr(
        source_model_module,
        "_source_operator_node_ids",
        lambda _model_path, _graph_collection: {
            source_operator_id: [
                artifact_models._NodeRef("Source Model", "main", long_node_id)
            ]
        },
    )
    result = _result(
        [{"id": "partition", "child_kinds": ["source_operator"]}],
        [
            {"id": "a", "name": "A", "kind": "partition"},
            {"id": "b", "name": "B", "kind": "partition"},
            {
                "id": source_operator_id,
                "kind": "source_operator",
                "parent_ids": ["b", "a"],
            },
        ],
    )
    graphs = [
        {
            "id": "main",
            "nodes": [
                {"id": long_node_id, "label": "split"},
                {"id": colliding_source_id, "label": "reserved collision"},
            ],
        }
    ]

    first = _write_artifacts(tmp_path / "first", monkeypatch, graphs, [result])
    second = _write_artifacts(tmp_path / "second", monkeypatch, graphs, [result])
    first_graph = _read_json(first.generated_models[0])["graphs"][0]
    second_graph = _read_json(second.generated_models[0])["graphs"][0]
    split_nodes = [
        node
        for node in first_graph["nodes"]
        if node.get("namespace") in {"A", "B"}
    ]

    assert first_graph == second_graph
    assert [node["namespace"] for node in split_nodes] == ["A", "B"]
    assert all(len(node["id"]) <= 160 for node in split_nodes)
    assert len({node["id"] for node in first_graph["nodes"]}) == 3
    assert colliding_source_id in {node["id"] for node in first_graph["nodes"]}
    assert split_nodes[0]["id"] != colliding_source_id


def test_each_generated_hierarchy_splits_source_operator_independently(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "left", "child_kinds": ["source_operator"]},
        {"id": "right", "child_kinds": ["source_operator"]},
    ]
    result = _result(
        entity_kinds,
        [
            {"id": "left-a", "name": "Left A", "kind": "left"},
            {"id": "left-b", "name": "Left B", "kind": "left"},
            {"id": "right-a", "name": "Right A", "kind": "right"},
            {"id": "right-b", "name": "Right B", "kind": "right"},
            _source_operator(0, "right-b", "left-b", "right-a", "left-a"),
        ],
    )

    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [{"id": "main", "nodes": [_node("0__op_0", "source")]}],
        [result],
    )
    collections = {
        collection["label"]: collection
        for collection in map(_read_json, generated.generated_models)
    }

    assert set(collections) == {"left", "right"}
    assert [
        node["namespace"] for node in collections["left"]["graphs"][0]["nodes"]
    ] == ["Left A", "Left B"]
    assert [
        node["namespace"] for node in collections["right"]["graphs"][0]["nodes"]
    ] == ["Right A", "Right B"]


def test_one_source_operator_can_split_in_multiple_graphs_with_subgraph_links(
    tmp_path: Path, monkeypatch
) -> None:
    source_operator_id = "source_operator/shared"
    monkeypatch.setattr(
        source_model_module,
        "_source_operator_node_ids",
        lambda _model_path, _graph_collection: {
            source_operator_id: [
                artifact_models._NodeRef("Source Model", "main", "main-op"),
                artifact_models._NodeRef("Source Model", "nested", "nested-op"),
            ]
        },
    )
    result = _result(
        [{"id": "partition", "child_kinds": ["source_operator"]}],
        [
            {"id": "a", "name": "A", "kind": "partition"},
            {"id": "b", "name": "B", "kind": "partition"},
            {
                "id": source_operator_id,
                "kind": "source_operator",
                "parent_ids": ["a", "b"],
            },
        ],
    )
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    {
                        "id": "main-op",
                        "label": "main",
                        "subgraphIds": ["nested"],
                    }
                ],
            },
            {"id": "nested", "nodes": [{"id": "nested-op", "label": "nested"}]},
        ],
        [result],
    )

    collection = _read_json(generated.generated_models[0])
    graphs = {graph["id"]: graph for graph in collection["graphs"]}
    assert set(graphs) == {"partition-main", "partition-nested"}
    for graph in graphs.values():
        assert [node["namespace"] for node in graph["nodes"]] == ["A", "B"]
    assert all(
        node["subgraphIds"] == ["partition-nested"]
        for node in graphs["partition-main"]["nodes"]
    )

    mappings = _read_json(generated.entity_node_mappings)["results"][0]["mappings"]
    generated_refs = {
        (mapping["node"]["graphId"], mapping["node"]["nodeId"])
        for mapping in mappings
        if mapping["entityId"] == source_operator_id
        and mapping["node"]["collectionLabel"] == "partition"
    }
    assert len(generated_refs) == 4
    assert {graph_id for graph_id, _ in generated_refs} == {
        "partition-main",
        "partition-nested",
    }


def test_mixed_level_parents_generate_every_complete_runtime_path(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "root", "child_kinds": ["mid", "source_operator"]},
        {
            "id": "mid",
            "parent_kinds": ["root"],
            "child_kinds": ["source_operator"],
        },
    ]
    source_operator_id = "source_operator/segment_0/spirv-0"
    entities: list[dict] = [
        {
            "id": "root-direct",
            "name": "Direct Root",
            "kind": "root",
        },
        {"id": "root-deep", "name": "Deep Root", "kind": "root"},
        {
            "id": "root-inferred",
            "name": "Ignored Inferred Root",
            "kind": "root",
            # Explicit source parent_ids take precedence over inferred child_ids.
            "child_ids": [source_operator_id],
        },
        {
            "id": "mid",
            "name": "Mid",
            "kind": "mid",
            "parent_ids": ["root-deep"],
        },
        {
            **_source_operator(0, "root-direct", "mid"),
            "placement": "NPU",
            "status": "estimated",
        },
    ]
    result = _result(entity_kinds, entities)
    result["checks"] = [
        {
            "entity_id": source_operator_id,
            "status": "fail",
            "details": {"reasons": [{"description": "mixed-level check"}]},
        }
    ]
    result["advice"] = [
        {
            "severity": "warning",
            "message": "mixed-level advice",
            "affected_entity_ids": [source_operator_id],
        }
    ]
    source_label = "Fused Conv2D / Bias + ReLU"
    graphs = [
        {
            "id": "main",
            "nodes": [{**_node("0__op_0", "source"), "label": source_label}],
        }
    ]

    generated = _write_artifacts(tmp_path / "first", monkeypatch, graphs, [result])
    reordered_result = _result(entity_kinds, list(reversed(entities)))
    reordered_result.update({"checks": result["checks"], "advice": result["advice"]})
    reordered = _write_artifacts(
        tmp_path / "reordered", monkeypatch, graphs, [reordered_result]
    )

    hierarchy = _read_json(generated.generated_models[0])
    reordered_hierarchy = _read_json(reordered.generated_models[0])
    assert hierarchy == reordered_hierarchy
    hierarchy_graph = hierarchy["graphs"][0]
    sub_operators = [
        node
        for node in hierarchy_graph["nodes"]
        if "__mlia_sub_operator__" in node["id"]
    ]
    assert [node["namespace"] for node in sub_operators] == [
        "Deep Root/Mid",
        "Direct Root",
    ]
    assert [node["label"] for node in sub_operators] == [
        f"{source_label} (#1)",
        f"{source_label} (#2)",
    ]
    source_graph = _read_json(
        generated.root / "models" / "model-source.json"
    )["graphs"][0]
    assert source_graph["nodes"][0]["label"] == source_label
    sub_operator_ids = {node["id"] for node in sub_operators}

    mappings = _read_json(generated.entity_node_mappings)
    assert mappings == _read_json(reordered.entity_node_mappings)
    entity_mappings = mappings["results"][0]["mappings"]
    assert {
        mapping["node"]["nodeId"]
        for mapping in entity_mappings
        if mapping["entityId"] == source_operator_id
        and mapping["node"]["collectionLabel"] == "root_mid"
    } == sub_operator_ids
    group_ids_by_entity = {
        entity_id: {
            mapping["node"]["nodeId"]
            for mapping in entity_mappings
            if mapping["entityId"] == entity_id
            and mapping["node"]["collectionLabel"] == "root_mid"
        }
        for entity_id in ("root-direct", "root-deep", "mid", "root-inferred")
    }
    assert group_ids_by_entity == {
        "root-direct": {"Direct Root___group___"},
        "root-deep": {"Deep Root___group___"},
        "mid": {"Deep Root/Mid___group___"},
        "root-inferred": set(),
    }

    navigation = _mapping_by_source(generated)
    assert _read_json(generated.sync_navigation) == _read_json(
        reordered.sync_navigation
    )
    assert navigation[("Source Model", "main", "source___group___")]["root_mid"][
        "root_mid-main"
    ] == ["Deep Root/Mid___group___", "Direct Root___group___"]
    assert navigation[("Source Model", "main", "0__op_0")]["root_mid"][
        "root_mid-main"
    ] == [node["id"] for node in sub_operators]
    for sub_operator in sub_operators:
        assert navigation[
            ("root_mid", "root_mid-main", sub_operator["id"])
        ]["Source Model"]["main"] == ["0__op_0"]

    node_data_dir = generated.root / "node-data" / "model-hierarchy-root_mid"
    reordered_node_data_dir = (
        reordered.root / "node-data" / "model-hierarchy-root_mid"
    )
    for filename in ("placement.json", "status.json", "checks.json", "advice.json"):
        overlay = _read_json(node_data_dir / filename)
        assert overlay == _read_json(reordered_node_data_dir / filename)
        assert set(overlay["root_mid-main"]["results"]) == sub_operator_ids


def test_multi_parent_source_operator_becomes_deterministic_sub_operator_chain(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "cascade", "child_kinds": ["chain"]},
        {
            "id": "chain",
            "parent_kinds": ["cascade"],
            "child_kinds": ["source_operator"],
        },
    ]
    entities: list[dict] = [
        {"id": "cascade-a", "name": "Cascade A", "kind": "cascade"},
        {"id": "cascade-b", "name": "Cascade B", "kind": "cascade"},
        {"id": "cascade-c", "name": "Cascade C", "kind": "cascade"},
        {"id": "cascade-z", "name": "Cascade Z", "kind": "cascade"},
        {
            "id": "chain-a",
            "name": "Chain A",
            "kind": "chain",
            "parent_ids": ["cascade-a"],
        },
        {
            "id": "chain-shared",
            "name": "Shared",
            "kind": "chain",
            "parent_ids": ["cascade-c", "cascade-b"],
        },
        {
            "id": "chain-z",
            "name": "Chain Z",
            "kind": "chain",
            "parent_ids": ["cascade-z"],
        },
        {
            **_source_operator(1, "chain-z", "chain-shared", "chain-a"),
            "placement": "NPU",
            "status": "estimated",
        },
    ]
    result = _result(entity_kinds, entities)
    result["breakdowns"] = [
        {
            "entity_id": "source_operator/segment_0/spirv-1",
            "metrics": [
                {"name": "cycles", "value": 7, "aggregation": "sum"}
            ],
        },
        {
            "entity_id": "chain-shared",
            "metrics": [
                {"name": "cycles", "value": 11, "aggregation": "sum"}
            ],
        },
    ]
    result["checks"] = [
        {
            "entity_id": "source_operator/segment_0/spirv-1",
            "status": "fail",
            "details": {"reasons": [{"description": "check message"}]},
        }
    ]
    result["advice"] = [
        {
            "severity": "warning",
            "message": "advice message",
            "affected_entity_ids": ["source_operator/segment_0/spirv-1"],
        }
    ]
    graphs = [
        {
            "id": "main",
            "nodes": [
                {"id": "input", "label": "input", "namespace": "input"},
                {
                    **_node("0__op_1", "source/op"),
                    "incomingEdges": [
                        {
                            "sourceNodeId": "input",
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": "0",
                        }
                    ],
                    "outputsMetadata": [{"id": "0", "attrs": []}],
                },
                {
                    "id": "consumer-0",
                    "label": "consumer-0",
                    "namespace": "output",
                    "incomingEdges": [
                        {
                            "sourceNodeId": "0__op_1",
                            "sourceNodeOutputId": "0",
                            "targetNodeInputId": "0",
                        }
                    ],
                },
                {
                    "id": "consumer-1",
                    "label": "consumer-1",
                    "namespace": "output",
                    "incomingEdges": [
                        {
                            "sourceNodeId": "0__op_1",
                            "sourceNodeOutputId": "1",
                            "targetNodeInputId": "0",
                        }
                    ],
                },
            ],
        }
    ]

    generated = _write_artifacts(tmp_path / "first", monkeypatch, graphs, [result])
    reordered_result = _result(entity_kinds, list(reversed(entities)))
    reordered_result.update(
        {
            "breakdowns": list(reversed(result["breakdowns"])),
            "checks": result["checks"],
            "advice": result["advice"],
        }
    )
    reordered = _write_artifacts(
        tmp_path / "reordered", monkeypatch, graphs, [reordered_result]
    )

    hierarchy_graph = _read_json(generated.generated_models[0])["graphs"][0]
    nodes = hierarchy_graph["nodes"]
    sub_operators = [
        node for node in nodes if "__mlia_sub_operator__" in node["id"]
    ]
    expected_paths = [
        "Cascade A/Chain A",
        "Cascade B/Shared",
        "Cascade C/Shared",
        "Cascade Z/Chain Z",
    ]
    assert [node["namespace"] for node in sub_operators] == expected_paths
    assert len({node["id"] for node in sub_operators}) == len(expected_paths)
    assert all(
        node["id"].startswith("0__op_1__mlia_sub_operator__")
        for node in sub_operators
    )
    assert not any(node["id"] == "0__op_1" for node in nodes)
    assert sub_operators[0]["incomingEdges"] == [
        {
            "sourceNodeId": "input",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]
    for previous, current in zip(sub_operators, sub_operators[1:]):
        assert current["incomingEdges"] == [
            {
                "sourceNodeId": previous["id"],
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            },
            {
                "sourceNodeId": previous["id"],
                "sourceNodeOutputId": "1",
                "targetNodeInputId": "1",
            },
        ]
    last_sub_operator_id = sub_operators[-1]["id"]
    consumers = {
        node["id"]: node
        for node in nodes
        if node["id"].startswith("consumer-")
    }
    assert (
        consumers["consumer-0"]["incomingEdges"][0]["sourceNodeId"]
        == last_sub_operator_id
    )
    assert (
        consumers["consumer-1"]["incomingEdges"][0]["sourceNodeId"]
        == last_sub_operator_id
    )
    assert sum(len(node.get("incomingEdges", [])) for node in consumers.values()) == 2

    reordered_graph = _read_json(reordered.generated_models[0])
    assert reordered_graph == _read_json(generated.generated_models[0])

    entity_mappings = _read_json(generated.entity_node_mappings)["results"][0][
        "mappings"
    ]
    source_operator_id = "source_operator/segment_0/spirv-1"
    mapped_sub_operator_ids = {
        mapping["node"]["nodeId"]
        for mapping in entity_mappings
        if mapping["node"]["collectionLabel"] == "cascade_chain"
        and mapping["entityId"] == source_operator_id
    }
    assert mapped_sub_operator_ids == {node["id"] for node in sub_operators}
    shared_group_ids = {
        mapping["node"]["nodeId"]
        for mapping in entity_mappings
        if mapping["entityId"] == "chain-shared"
    }
    assert shared_group_ids == {
        "Cascade B/Shared___group___",
        "Cascade C/Shared___group___",
    }

    node_data_dir = generated.root / "node-data" / "model-hierarchy-cascade_chain"
    for filename in ("placement.json", "status.json", "checks.json", "advice.json"):
        overlay_results = _read_json(node_data_dir / filename)["cascade_chain-main"][
            "results"
        ]
        assert set(overlay_results) == {node["id"] for node in sub_operators}
    generated_cycles = _read_json(node_data_dir / "cycles.json")[
        "cascade_chain-main"
    ]["results"]
    assert generated_cycles == {
        "Cascade B/Shared": {"value": 11},
        "Cascade C/Shared": {"value": 11},
    }
    assert not mapped_sub_operator_ids.intersection(generated_cycles)
    source_cycles = _read_json(
        generated.root / "node-data" / "model-source" / "cycles.json"
    )["main"]["results"]
    assert source_cycles["0__op_1"] == {"value": 7}
    assert not mapped_sub_operator_ids.intersection(source_cycles)

    navigation = _mapping_by_source(generated)
    source_to_generated = navigation[
        ("Source Model", "main", "source/op___group___")
    ]["cascade_chain"]["cascade_chain-main"]
    assert source_to_generated == [
        f"{path}___group___" for path in expected_paths
    ]
    source_leaf_targets = navigation[
        ("Source Model", "main", "0__op_1")
    ]["cascade_chain"]["cascade_chain-main"]
    assert source_leaf_targets == [node["id"] for node in sub_operators]
    for sub_operator in sub_operators:
        assert navigation[
            ("cascade_chain", "cascade_chain-main", sub_operator["id"])
        ]["Source Model"]["main"] == ["0__op_1"]


def test_split_nodes_preserve_unusable_source_label_behavior() -> None:
    graph = {
        "id": "main",
        "nodes": [
            {"id": "missing"},
            {"id": "empty", "label": ""},
            {"id": "non-string", "label": 7},
        ],
    }

    generated, _ = graphs_module._entity_hierarchy_graph(
        graph,
        {"main": "partition-main"},
        {
            "missing": ("A", "B"),
            "empty": ("A", "B"),
            "non-string": ("A", "B"),
        },
    )
    nodes = generated["nodes"]

    assert all("label" not in node for node in nodes[:2])
    assert [node["label"] for node in nodes[2:4]] == ["", ""]
    assert [node["label"] for node in nodes[4:]] == [7, 7]


def test_single_complete_path_keeps_original_node_and_dataflow(
    tmp_path: Path, monkeypatch
) -> None:
    entity_kinds = [
        {"id": "cascade", "child_kinds": ["chain"]},
        {
            "id": "chain",
            "parent_kinds": ["cascade"],
            "child_kinds": ["source_operator"],
        },
    ]
    result = _result(
        entity_kinds,
        [
            {"id": "cascade", "name": "Cascade", "kind": "cascade"},
            {
                "id": "chain",
                "name": "Chain",
                "kind": "chain",
                "parent_ids": ["cascade"],
            },
            _source_operator(1, "chain"),
        ],
    )
    graphs = [
        {
            "id": "main",
            "nodes": [
                {"id": "input", "label": "input", "namespace": "input"},
                {
                    **_node("0__op_1", "source/op"),
                    "incomingEdges": [{"sourceNodeId": "input"}],
                },
                {
                    "id": "consumer",
                    "label": "consumer",
                    "namespace": "output",
                    "incomingEdges": [{"sourceNodeId": "0__op_1"}],
                },
            ],
        }
    ]

    generated = _write_artifacts(tmp_path, monkeypatch, graphs, [result])
    nodes = _read_json(generated.generated_models[0])["graphs"][0]["nodes"]
    generated_by_id = {node["id"]: node for node in nodes}

    assert "0__op_1" in generated_by_id
    assert not any("__mlia_sub_operator__" in node_id for node_id in generated_by_id)
    assert generated_by_id["0__op_1"]["namespace"] == "Cascade/Chain"
    assert generated_by_id["0__op_1"]["label"] == "0__op_1"
    assert generated_by_id["0__op_1"]["incomingEdges"] == [
        {"sourceNodeId": "input"}
    ]
    assert generated_by_id["consumer"]["incomingEdges"] == [
        {"sourceNodeId": "0__op_1"}
    ]


def _headless_browser() -> str | None:
    for executable in (
        "msedge",
        "chrome",
        "chromium",
        "chromium-browser",
    ):
        resolved = shutil.which(executable)
        if resolved:
            return resolved
    return None


def _run_headless_browser(browser: str, index: Path, budget_ms: int = 5000):
    return subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            "--dump-dom",
            f"--virtual-time-budget={budget_ms}",
            index.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_single_pane_hides_right_graph_control_in_browser(tmp_path: Path) -> None:
    """The single-pane toolbar must not render any part of the right control."""
    browser = _headless_browser()
    if browser is None:
        pytest.skip("A Chromium browser is required for the toolbar layout test")

    data = {
        "hostContractVersion": 1,
        "graphCollections": [
            {
                "label": "Source Model",
                "graphs": [{"id": "main", "nodes": []}],
            }
        ],
        "nodeDataProviderRuns": [],
        "initialUiState": {
            "paneStates": [
                {
                    "deepestExpandedGroupNodeIds": [],
                    "selectedNodeId": "",
                    "selectedGraphId": "main",
                    "selectedCollectionLabel": "Source Model",
                    "widthFraction": 1,
                }
            ]
        },
        "selectedNodeData": [],
        "visualizerConfig": {},
        "manifest": {},
    }
    probe = """
<script>
const rightControl = document.getElementById('right-graph-control');
const addPaneButton = document.getElementById('add-pane-button');
document.documentElement.dataset.rightGraphHidden = String(rightControl.hidden);
document.documentElement.dataset.rightGraphDisplay =
  getComputedStyle(rightControl).display;
document.documentElement.dataset.addPaneHidden = String(addPaneButton.hidden);
document.documentElement.dataset.addPaneDisplay =
  getComputedStyle(addPaneButton).display;
rightControl.hidden = false;
document.documentElement.dataset.rightGraphShownDisplay =
  getComputedStyle(rightControl).display;
</script>
"""
    index = tmp_path / "index.html"
    index.write_text(
        bundler_module._index_html(data).replace("</body>", f"{probe}</body>"),
        encoding="utf-8",
    )

    completed = _run_headless_browser(browser, index)

    assert completed.returncode == 0, completed.stderr
    assert 'data-right-graph-hidden="true"' in completed.stdout
    assert 'data-right-graph-display="none"' in completed.stdout
    assert 'data-add-pane-hidden="false"' in completed.stdout
    assert 'data-add-pane-display="flex"' in completed.stdout
    assert 'data-right-graph-shown-display="flex"' in completed.stdout


def test_vendored_visualizer_executes_graph_mapping_validation(tmp_path: Path) -> None:
    browser = _headless_browser()
    if browser is None:
        pytest.skip(
            "A Chromium browser is required for the embedded runtime smoke test"
        )

    data = {
        "graphCollections": [
            {
                "label": "Source Model",
                "graphs": [
                    {
                        "id": "main",
                        "nodes": [{"id": "a", "label": "a", "namespace": "source"}],
                    }
                ],
            },
            {
                "label": "target",
                "graphs": [
                    {
                        "id": "target-main",
                        "nodes": [{"id": "a", "label": "a", "namespace": "target"}],
                    }
                ],
            },
        ],
        "nodeDataProviderRuns": [],
        "initialUiState": {
            "paneStates": [
                {
                    "deepestExpandedGroupNodeIds": [],
                    "selectedNodeId": "",
                    "selectedGraphId": "main",
                    "selectedCollectionLabel": "Source Model",
                    "widthFraction": 0.5,
                },
                {
                    "deepestExpandedGroupNodeIds": [],
                    "selectedNodeId": "",
                    "selectedGraphId": "target-main",
                    "selectedCollectionLabel": "target",
                    "widthFraction": 0.5,
                    "selected": True,
                },
            ]
        },
        "selectedNodeData": [],
        "visualizerConfig": {
            "syncNavigationData": {
                "type": "sync_navigation",
                "graphMappings": {},
            }
        },
        "manifest": {},
    }
    runtime_probe = """
<script>
const graphMappingsProbe = setInterval(() => {
  if (document.body.textContent.includes('graphMappings must be an array.')) {
    document.documentElement.dataset.graphMappingsRuntime = 'validated';
    clearInterval(graphMappingsProbe);
  }
}, 10);
</script>
"""
    html = bundler_module._index_html(data).replace(
        "</body>", f"{runtime_probe}</body>"
    )
    index = tmp_path / "index.html"
    index.write_text(html, encoding="utf-8")

    completed = _run_headless_browser(browser, index)

    assert completed.returncode == 0, completed.stderr
    assert 'data-graph-mappings-runtime="validated"' in completed.stdout


def test_vendored_visualizer_applies_mappings_and_suppresses_cross_family_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    browser = _headless_browser()
    if browser is None:
        pytest.skip(
            "A Chromium browser is required for the embedded runtime integration test"
        )

    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    entities = [
        {"id": "partition", "name": "mapped", "kind": "partition"},
        _source_operator(0, "partition"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {"id": "main", "nodes": [_node("0__op_0", "shared")]},
            {"id": "unrelated", "nodes": [_node("0__op_0", "shared")]},
        ],
        [_result(entity_kinds, entities)],
    )
    source_route = _mapping_by_source(generated)[
        ("Source Model", "main", "shared___group___")
    ]
    assert source_route["partition"]["partition-main"] == [
        "mapped___group___"
    ]
    assert source_route["Source Model"]["unrelated"] == []
    assert source_route["partition"]["partition-unrelated"] == []

    index = build_artifacts_bundle(generated.root)
    runtime_probe = """
<script>
(async () => {
  const root = document.documentElement;
  const waitFor = async (predicate, description, timeoutMs = 20000) => {
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (predicate()) return;
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error(`Timed out waiting for ${description}`);
  };

  try {
    await customElements.whenDefined('model-explorer-visualizer');
    const visualizer = document.querySelector('model-explorer-visualizer');
    if (!visualizer) throw new Error('visualizer element was not created');

    const processed = [];
    const selections = [];
    let latestUiState;
    visualizer.addEventListener('modelGraphProcessed', (event) => {
      processed.push(event.detail);
    });
    visualizer.addEventListener('selectedNodeChanged', (event) => {
      if (event.detail?.nodeId) selections.push(event.detail);
    });
    visualizer.addEventListener('uiStateChanged', (event) => {
      latestUiState = event.detail;
    });

    await waitFor(
      () => !document.getElementById('left-graph-select')?.disabled,
      'the MLIA graph selectors'
    );
    const leftSelect = document.getElementById('left-graph-select');
    const unrelatedOption = Array.from(leftSelect.options).find(
      (option) => option.textContent === 'Source Model / unrelated'
    );
    if (!unrelatedOption) throw new Error('missing unrelated source option');
    leftSelect.value = unrelatedOption.value;
    leftSelect.dispatchEvent(new Event('change'));
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'Source Model' &&
        event.modelGraph?.id === 'unrelated'
      ),
      'the source graph selector change'
    );
    const rightSelect = document.getElementById('right-graph-select');
    const partitionOption = Array.from(rightSelect.options).find(
      (option) => option.textContent === 'partition / partition-unrelated'
    );
    if (!partitionOption) throw new Error('missing unrelated partition option');
    rightSelect.value = partitionOption.value;
    rightSelect.dispatchEvent(new Event('change'));
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'partition' &&
        event.modelGraph?.id === 'partition-unrelated'
      ),
      'the generated graph selector change'
    );
    visualizer.selectGraph('main', 'Source Model', 0);
    visualizer.selectGraph('partition-main', 'partition', 1);
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'Source Model' &&
        event.modelGraph?.id === 'main'
      ) && processed.some((event) =>
        event.modelGraph?.collectionLabel === 'partition' &&
        event.modelGraph?.id === 'partition-main'
      ),
      'both selected primary graphs'
    );
    visualizer.selectNode(
      'shared___group___', 'main', 'Source Model', 0
    );
    await waitFor(
      () => selections.some((selection) =>
        selection.collectionLabel === 'partition' &&
        selection.graphId === 'partition-main' &&
        selection.nodeId === 'mapped___group___' &&
        selection.selectionSource === 'sync'
      ),
      'the mapped target group selection'
    );
    root.dataset.graphMappingsMapped = 'selected';

    selections.length = 0;
    visualizer.selectNode('0__op_0', 'unrelated', 'Source Model', 1);
    await waitFor(
      () => selections.some((selection) =>
        selection.collectionLabel === 'Source Model' &&
        selection.graphId === 'unrelated' &&
        selection.nodeId === '0__op_0' &&
        selection.selectionSource === 'user'
      ),
      'the unrelated graph in the second pane'
    );

    selections.length = 0;
    visualizer.selectNode(
      'shared___group___', 'main', 'Source Model', 0
    );
    await waitFor(
      () => selections.some((selection) =>
        selection.collectionLabel === 'Source Model' &&
        selection.graphId === 'main' &&
        selection.nodeId === 'shared___group___' &&
        selection.selectionSource === 'user'
      ),
      'the cross-family source selection'
    );
    await new Promise((resolve) => setTimeout(resolve, 500));

    const incorrectFallback = selections.some((selection) =>
      selection.collectionLabel === 'Source Model' &&
      selection.graphId === 'unrelated' &&
      selection.nodeId === 'shared___group___' &&
      selection.selectionSource === 'sync'
    );
    const rightPane = latestUiState?.paneStates?.[1];
    if (incorrectFallback) {
      throw new Error('same-id fallback selected the unrelated group');
    }
    if (
      rightPane?.selectedCollectionLabel !== 'Source Model' ||
      rightPane?.selectedGraphId !== 'unrelated' ||
      rightPane?.selectedNodeId !== '0__op_0'
    ) {
      throw new Error(`unexpected right pane state: ${JSON.stringify(rightPane)}`);
    }
    root.dataset.graphMappingsEmptyRoute = 'suppressed';
  } catch (error) {
    root.dataset.graphMappingsError = String(error);
  }
})();
</script>
"""
    html = index.read_text(encoding="utf-8").replace(
        "</body>", f"{runtime_probe}</body>"
    )
    index.write_text(html, encoding="utf-8")

    completed = _run_headless_browser(browser, index, budget_ms=30000)

    assert completed.returncode == 0, completed.stderr
    assert 'data-graph-mappings-mapped="selected"' in completed.stdout
    assert 'data-graph-mappings-empty-route="suppressed"' in completed.stdout
    assert "data-graph-mappings-error=" not in completed.stdout


def test_vendored_visualizer_keeps_cross_parent_collision_groups_distinct(
    tmp_path: Path, monkeypatch
) -> None:
    browser = _headless_browser()
    if browser is None:
        pytest.skip("A Chromium browser is required for the hierarchy runtime test")

    entity_kinds = [
        {"id": "outer", "child_kinds": ["inner"]},
        {
            "id": "inner",
            "parent_kinds": ["outer"],
            "child_kinds": ["source_operator"],
        },
    ]
    entities: list[dict] = [
        {"id": "parent-1", "name": "parent1", "kind": "outer"},
        {"id": "parent-2", "name": "parent2", "kind": "outer"},
        {
            "id": "child/1",
            "name": "x;shared",
            "kind": "inner",
            "parent_ids": ["parent-1"],
        },
        {
            "id": "child;2",
            "name": "y;shared",
            "kind": "inner",
            "parent_ids": ["parent-2"],
        },
        _source_operator(0, "child/1"),
        _source_operator(1, "child;2"),
    ]
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    _node("0__op_0", "source"),
                    _node("0__op_1", "source"),
                ],
            }
        ],
        [_result(entity_kinds, entities)],
    )
    index = build_artifacts_bundle(generated.root)
    runtime_probe = """
<script>
(async () => {
  const root = document.documentElement;
  const waitFor = async (predicate, description, timeoutMs = 20000) => {
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (predicate()) return;
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error(`Timed out waiting for ${description}`);
  };

  try {
    await customElements.whenDefined('model-explorer-visualizer');
    const visualizer = document.querySelector('model-explorer-visualizer');
    if (!visualizer) throw new Error('visualizer element was not created');

    const processed = [];
    visualizer.addEventListener('modelGraphProcessed', (event) => {
      processed.push(event.detail);
    });
    await waitFor(
      () => !document.getElementById('left-graph-select')?.disabled,
      'the MLIA graph selectors'
    );
    visualizer.selectGraph('main', 'Source Model', 1);
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'Source Model' &&
        event.paneIndex === 1
      ),
      'the source graph selector change'
    );
    visualizer.selectGraph('outer_inner-main', 'outer_inner', 1);
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'outer_inner'
      ),
      'the generated hierarchy selector change'
    );
    const graph = processed.find((event) =>
      event.modelGraph.collectionLabel === 'outer_inner'
    ).modelGraph;
    const expected = {
      'parent1___group___': ['0__op_0'],
      'parent1/x%3Bshared [child%2F1]___group___': ['0__op_0'],
      'parent2___group___': ['0__op_1'],
      'parent2/y%3Bshared [child%3B2]___group___': ['0__op_1'],
    };
    for (const [groupId, leaves] of Object.entries(expected)) {
      const group = graph.nodesById[groupId];
      if (group?.nodeType !== 1) {
        throw new Error(`missing runtime group ${groupId}`);
      }
      if (JSON.stringify(group.descendantsOpNodeIds) !== JSON.stringify(leaves)) {
        throw new Error(
          `wrong leaves for ${groupId}: ${JSON.stringify(group.descendantsOpNodeIds)}`
        );
      }
    }
    if (graph.nodesById['shared___group___']) {
      throw new Error('cross-parent entities merged into shared___group___');
    }
    root.dataset.hierarchyCollisionRuntime = 'distinct';
  } catch (error) {
    root.dataset.hierarchyCollisionError = String(error);
  }
})();
</script>
"""
    html = index.read_text(encoding="utf-8").replace(
        "</body>", f"{runtime_probe}</body>"
    )
    index.write_text(html, encoding="utf-8")

    completed = _run_headless_browser(browser, index, budget_ms=30000)

    assert completed.returncode == 0, completed.stderr
    assert 'data-hierarchy-collision-runtime="distinct"' in completed.stdout
    assert "data-hierarchy-collision-error=" not in completed.stdout


def test_vendored_visualizer_keeps_group_suffix_leaf_addressable(
    tmp_path: Path, monkeypatch
) -> None:
    browser = _headless_browser()
    if browser is None:
        pytest.skip("A Chromium browser is required for the suffix-leaf runtime test")

    monkeypatch.setattr(
        source_model_module,
        "_source_operator_node_ids",
        lambda _model_path, _graph_collection: {
            "source_operator/segment_0/spirv-0": [
                artifact_models._NodeRef("Source Model", "main", "leaf___group___")
            ],
            "source_operator/segment_0/spirv-1": [
                artifact_models._NodeRef("Source Model", "main", "ordinary")
            ],
        },
    )
    entity_kinds = [{"id": "partition", "child_kinds": ["source_operator"]}]
    result = _result(
        entity_kinds,
        [
            {"id": "suffix", "name": "target suffix", "kind": "partition"},
            {"id": "ordinary", "name": "target ordinary", "kind": "partition"},
            _source_operator(0, "suffix"),
            _source_operator(1, "ordinary"),
        ],
    )
    generated = _write_artifacts(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "main",
                "nodes": [
                    {
                        **_node("leaf___group___", "source/enclosing"),
                        "attrs": [{"key": "spv id", "value": "%0"}],
                    },
                    {
                        **_node("ordinary", "source/enclosing"),
                        "attrs": [{"key": "spv id", "value": "%1"}],
                    },
                ],
            }
        ],
        [result],
    )
    index = build_artifacts_bundle(generated.root)
    runtime_probe = """
<script>
(async () => {
  const root = document.documentElement;
  const waitFor = async (predicate, description, timeoutMs = 20000) => {
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (predicate()) return;
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error(`Timed out waiting for ${description}`);
  };

  try {
    await customElements.whenDefined('model-explorer-visualizer');
    const visualizer = document.querySelector('model-explorer-visualizer');
    if (!visualizer) throw new Error('visualizer element was not created');

    const processed = [];
    const selections = [];
    visualizer.addEventListener('modelGraphProcessed', (event) => {
      processed.push(event.detail);
    });
    visualizer.addEventListener('selectedNodeChanged', (event) => {
      if (event.detail?.nodeId) selections.push(event.detail);
    });

    await waitFor(
      () => !document.getElementById('left-graph-select')?.disabled,
      'the MLIA graph selectors'
    );
    visualizer.selectGraph('main', 'Source Model', 1);
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'Source Model' &&
        event.paneIndex === 1
      ),
      'the source graph selector change'
    );
    visualizer.selectGraph('partition-main', 'partition', 1);
    await waitFor(
      () => processed.some((event) =>
        event.modelGraph?.collectionLabel === 'partition'
      ),
      'both suffix-leaf graph projections'
    );
    const sourceGraph = processed.find((event) =>
      event.modelGraph.collectionLabel === 'Source Model'
    ).modelGraph;
    const targetGraph = processed.find((event) =>
      event.modelGraph.collectionLabel === 'partition'
    ).modelGraph;
    const suffixLeaf = sourceGraph.nodesById['leaf___group___'];
    const enclosingGroup = sourceGraph.nodesById['source/enclosing___group___'];
    if (suffixLeaf?.nodeType !== 0) {
      throw new Error(`suffix leaf was not an op node: ${JSON.stringify(suffixLeaf)}`);
    }
    if (targetGraph.nodesById['leaf___group___']?.nodeType !== 0) {
      throw new Error('generated suffix leaf was not an op node');
    }
    if (
      enclosingGroup?.nodeType !== 1 ||
      !enclosingGroup.descendantsOpNodeIds?.includes('leaf___group___') ||
      !enclosingGroup.descendantsOpNodeIds?.includes('ordinary')
    ) {
      throw new Error(
        `enclosing group lost canonical leaves: ${JSON.stringify(enclosingGroup)}`
      );
    }
    root.dataset.groupSuffixLeafRuntime = 'processed';

    visualizer.selectNode('leaf___group___', 'main', 'Source Model', 0);
    await waitFor(
      () => selections.some((selection) =>
        selection.collectionLabel === 'partition' &&
        selection.graphId === 'partition-main' &&
        selection.nodeId === 'leaf___group___' &&
        selection.selectionSource === 'sync'
      ),
      'suffix leaf sync selection'
    );
    root.dataset.groupSuffixLeafSelection = 'synced';
  } catch (error) {
    root.dataset.groupSuffixLeafError = String(error);
  }
})();
</script>
"""
    html = index.read_text(encoding="utf-8").replace(
        "</body>", f"{runtime_probe}</body>"
    )
    index.write_text(html, encoding="utf-8")

    completed = _run_headless_browser(browser, index, budget_ms=30000)

    assert completed.returncode == 0, completed.stderr
    assert 'data-group-suffix-leaf-runtime="processed"' in completed.stdout
    assert 'data-group-suffix-leaf-selection="synced"' in completed.stdout
    assert "data-group-suffix-leaf-error=" not in completed.stdout
