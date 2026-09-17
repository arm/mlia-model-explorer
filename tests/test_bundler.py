# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for standalone Model Explorer bundle selection."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import pytest

from mlia.model_explorer import bundler

_ASSERT = unittest.TestCase()


def _graph_collection(
    label: str,
    graph_id: str,
    nodes: list[dict[str, str]],
) -> dict[str, Any]:
    return _multi_graph_collection(label, [(graph_id, nodes)])


def _multi_graph_collection(
    label: str,
    graphs: list[tuple[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    return {
        "label": label,
        "graphs": [{"id": graph_id, "nodes": nodes} for graph_id, nodes in graphs],
    }


def _run(
    collection_label: str,
    name: str,
    data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "collectionLabel": collection_label,
        "name": name,
        "data": data,
    }


def _graph_data(results: dict[str, Any], *, name: str = "test") -> dict[str, Any]:
    return {"name": name, "results": results}


def _write_payload(root: Path, name: str, payload: object) -> Path:
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_node_data_provider_runs_read_graph_indexed_payload(tmp_path: Path) -> None:
    """Standard Model Explorer graph-indexed data becomes one provider run."""
    payload_path = _write_payload(
        tmp_path,
        "cycles.json",
        {"main": _graph_data({"node": {"value": 12}}, name="cycles")},
    )

    runs = bundler._node_data_provider_runs(
        tmp_path,
        [
            {
                "path": payload_path.name,
                "graph_collection_label": "Source Model",
            }
        ],
    )

    _ASSERT.assertEqual(
        runs,
        [
            {
                "collectionLabel": "Source Model",
                "name": "cycles",
                "data": {
                    "main": {
                        "name": "cycles",
                        "results": {"node": {"value": 12}},
                    }
                },
            }
        ],
    )


def test_node_data_provider_runs_reject_unindexed_payload(tmp_path: Path) -> None:
    """Node-data artifacts must use Model Explorer's graph-indexed format."""
    payload_path = _write_payload(
        tmp_path,
        "placement.json",
        _graph_data({"node": {"value": "NPU"}}, name="placement"),
    )

    with pytest.raises(RuntimeError, match="graph-indexed JSON object"):
        bundler._node_data_provider_runs(
            tmp_path,
            [
                {
                    "path": payload_path.name,
                    "graph_collection_label": "Source Model",
                }
            ],
        )


def test_node_data_provider_runs_allow_same_effective_name_on_disjoint_graphs(
    tmp_path: Path,
) -> None:
    """Overlay identity includes the graph, so disjoint reuse is unambiguous."""
    first_path = _write_payload(
        tmp_path,
        "first.json",
        {"main": _graph_data({"node": {"value": 1}}, name="duplicate")},
    )
    second_path = _write_payload(
        tmp_path,
        "second.json",
        {"other": _graph_data({"node": {"value": 2}}, name="duplicate")},
    )

    runs = bundler._node_data_provider_runs(
        tmp_path,
        [
            {
                "path": first_path.name,
                "graph_collection_label": "Source Model",
            },
            {
                "path": second_path.name,
                "graph_collection_label": "Source Model",
            },
        ],
    )

    _ASSERT.assertEqual([run["name"] for run in runs], ["duplicate", "duplicate"])


def test_node_data_provider_runs_reject_same_graph_effective_name(
    tmp_path: Path,
) -> None:
    """Two applicable runs cannot expose the same name on one graph."""
    first_path = _write_payload(
        tmp_path,
        "first.json",
        {"main": _graph_data({"node": {"value": 1}}, name="duplicate")},
    )
    second_path = _write_payload(
        tmp_path,
        "second.json",
        {"main": _graph_data({"node": {"value": 2}}, name="duplicate")},
    )

    with pytest.raises(
        RuntimeError,
        match="Duplicate node-data provider name 'duplicate' for graph 'main'",
    ):
        bundler._node_data_provider_runs(
            tmp_path,
            [
                {
                    "path": first_path.name,
                    "graph_collection_label": "Source Model",
                },
                {
                    "path": second_path.name,
                    "graph_collection_label": "Source Model",
                },
            ],
        )


def test_selected_node_data_uses_graph_level_effective_name() -> None:
    """Default selection follows graph-level names, not only run fallbacks."""
    runs = [
        _run(
            "collection",
            "fallback",
            {
                "main": _graph_data(
                    {"node": {"value": 1}},
                    name="total_cycles",
                )
            },
        )
    ]

    _ASSERT.assertEqual(
        bundler._selected_node_data(runs),
        [{"modelName": "collection", "name": "total_cycles"}],
    )


def test_initial_ui_state_prefers_greater_valid_node_data_coverage() -> None:
    """Valid value coverage wins even when another graph has more overlay runs."""
    source = _graph_collection("Source Model", "source", [{"id": "source-node"}])
    more_runs = _graph_collection(
        "more-runs",
        "more-graph",
        [
            {"id": "only-valid-node", "namespace": "covered/group"},
            {"id": "uncovered-node", "namespace": "uncovered/group"},
        ],
    )
    more_coverage = _graph_collection(
        "more-coverage",
        "coverage-graph",
        [
            {"id": "node-a", "namespace": "namespace/child"},
            {"id": "node-b", "namespace": "present-without-overlay"},
        ],
    )
    lower_coverage = _graph_collection(
        "lower-coverage",
        "lower-graph",
        [{"id": "lower-node", "namespace": "lower/group"}],
    )
    runs = [
        _run(
            "more-runs",
            "overlay-1",
            {"more-graph": _graph_data({"only-valid-node": {"value": 1}})},
        ),
        _run(
            "more-runs",
            "overlay-2",
            {"more-graph": _graph_data({"missing-node": {"value": 2}})},
        ),
        _run(
            "more-runs",
            "overlay-3",
            {"missing-graph": _graph_data({"covered/group": {"value": 3}})},
        ),
        _run(
            "more-coverage",
            "one-rich-overlay",
            {
                "coverage-graph": _graph_data(
                    {
                        "node-a": {"value": 0},
                        "namespace": {"value": False},
                        "namespace/child": {"value": ""},
                        "missing-node": {"value": 4},
                        "present-without-overlay": {"bgColor": "red"},
                        "node-b": {"value": None},
                    }
                )
            },
        ),
        _run(
            "lower-coverage",
            "lower-overlay",
            {"lower-graph": _graph_data({"lower-node": {"value": 1}})},
        ),
    ]

    _ASSERT.assertEqual(bundler._node_data_provider_run_coverage(more_runs, runs), 1)
    _ASSERT.assertEqual(
        bundler._node_data_provider_run_coverage(more_coverage, runs), 3
    )
    _ASSERT.assertEqual(
        bundler._node_data_provider_run_coverage(lower_coverage, runs), 1
    )

    initial_ui_state = bundler._initial_ui_state(
        [source, more_runs, more_coverage, lower_coverage],
        runs,
    )

    _ASSERT.assertIsNot(initial_ui_state, None)
    _ASSERT.assertNotIn("selected", initial_ui_state["paneStates"][0])
    _ASSERT.assertEqual(initial_ui_state["paneStates"][0]["selectedGraphId"], "source")
    _ASSERT.assertIs(initial_ui_state["paneStates"][1]["selected"], True)
    _ASSERT.assertEqual(
        initial_ui_state["paneStates"][1]["selectedGraphId"],
        "coverage-graph",
    )
    _ASSERT.assertEqual(
        initial_ui_state["paneStates"][1]["selectedCollectionLabel"],
        "more-coverage",
    )


def test_initial_ui_state_preserves_first_candidate_on_coverage_tie() -> None:
    """Equal coverage keeps the existing deterministic first-candidate choice."""
    source = _graph_collection("Source Model", "source", [{"id": "source-node"}])
    first = _graph_collection("first", "first-graph", [{"id": "first-node"}])
    second = _graph_collection("second", "second-graph", [{"id": "second-node"}])
    runs = [
        _run(
            "first",
            "first-overlay",
            {"first-graph": _graph_data({"first-node": {"value": 1}})},
        ),
        _run(
            "second",
            "second-overlay",
            {"second-graph": _graph_data({"second-node": {"value": 2}})},
        ),
    ]

    initial_ui_state = bundler._initial_ui_state([source, first, second], runs)

    _ASSERT.assertIsNot(initial_ui_state, None)
    _ASSERT.assertEqual(
        initial_ui_state["paneStates"][1]["selectedGraphId"], "first-graph"
    )
    _ASSERT.assertEqual(
        initial_ui_state["paneStates"][1]["selectedCollectionLabel"],
        "first",
    )


def test_node_data_provider_run_coverage_is_collection_scoped() -> None:
    """A run for another collection never contributes to pane selection."""
    collection = _graph_collection(
        "candidate",
        "candidate-graph",
        [{"id": "node", "namespace": "group/child"}],
    )
    runs = [
        _run(
            "other-collection",
            "other-overlay",
            {
                "candidate-graph": _graph_data(
                    {"node": {"value": 1}, "group": {"value": 2}}
                )
            },
        )
    ]

    _ASSERT.assertEqual(bundler._node_data_provider_run_coverage(collection, runs), 0)


def test_initial_ui_state_ignores_later_graph_coverage() -> None:
    """Values for later graphs cannot influence the initial first-graph pane."""
    source = _graph_collection("Source Model", "source", [{"id": "source-node"}])
    later_heavy = _multi_graph_collection(
        "later-heavy",
        [
            ("later-heavy-first", [{"id": "first-node"}]),
            (
                "later-heavy-second",
                [{"id": "later-node", "namespace": "later/group"}],
            ),
        ],
    )
    first_graph_rich = _multi_graph_collection(
        "first-graph-rich",
        [
            (
                "first-graph-rich-first",
                [{"id": "rich-node", "namespace": "rich/group"}],
            ),
            ("first-graph-rich-second", [{"id": "unused-later-node"}]),
        ],
    )
    runs = [
        _run(
            "later-heavy",
            "qualified-later-heavy",
            {
                "later-heavy-first": _graph_data({"first-node": {"value": "first"}}),
                "later-heavy-second": _graph_data(
                    {
                        "later-node": {"value": 1},
                        "later": {"value": 2},
                        "later/group": {"value": 3},
                    }
                ),
            },
        ),
        _run(
            "first-graph-rich",
            "qualified-first-rich",
            {
                "first-graph-rich-first": _graph_data(
                    {"rich-node": {"value": 0}, "rich": {"value": False}}
                )
            },
        ),
    ]

    _ASSERT.assertEqual(bundler._node_data_provider_run_coverage(later_heavy, runs), 1)
    _ASSERT.assertEqual(
        bundler._node_data_provider_run_coverage(first_graph_rich, runs), 2
    )

    initial_ui_state = bundler._initial_ui_state(
        [source, later_heavy, first_graph_rich], runs
    )

    _ASSERT.assertIsNot(initial_ui_state, None)
    _ASSERT.assertEqual(initial_ui_state["paneStates"][0]["selectedGraphId"], "source")
    _ASSERT.assertEqual(
        initial_ui_state["paneStates"][1]["selectedGraphId"],
        "first-graph-rich-first",
    )
    _ASSERT.assertEqual(
        initial_ui_state["paneStates"][1]["selectedCollectionLabel"],
        "first-graph-rich",
    )
