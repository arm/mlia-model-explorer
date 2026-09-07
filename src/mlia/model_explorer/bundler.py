# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Standalone HTML bundle helpers for Model Explorer artifacts."""

from __future__ import annotations

import html
import json
import logging
import re
import shutil
import webbrowser
from pathlib import Path
from typing import Any

from mlia.model_explorer.graph_data import graph_node_data_keys

logger = logging.getLogger(__name__)

_ARTIFICIAL_LAYER_NODE_COUNT_THRESHOLD = 1_000_000_000
# Contract consumed by embedders that safely update the generated host data.
_HOST_CONTRACT_VERSION = 1
_DEFAULT_SELECTED_NODE_DATA_NAMES = (
    "total_cycles",
    "op_cycles",
)
_VISUALIZER_DIR = Path(__file__).parent / "model-explorer-visualizer"
_VISUALIZER_JS = _VISUALIZER_DIR / "main_embedded.js"
_BUNDLE_DIR_NAME = "bundle"


def build_artifacts_bundle(root: Path) -> Path:
    """Build a standalone Model Explorer HTML page for generated artifacts.

    The returned path points at the generated ``index.html``. The page embeds the
    Model Explorer visualizer JavaScript, graph collections, node-data overlays,
    sync-navigation data, and host glue code directly in the HTML so it can be
    opened from ``file://`` without loading sibling runtime files.
    """
    manifest = _read_manifest(root)
    bundle = root / _BUNDLE_DIR_NAME
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    data = _bundle_data(root, manifest)
    (bundle / "index.html").write_text(_index_html(data), encoding="utf-8")
    logger.info("Model Explorer standalone HTML bundle written to %s", bundle)
    return bundle / "index.html"


def open_artifacts_bundle(root: Path) -> Path:
    """Build and open a standalone Model Explorer HTML page in the browser."""
    index = build_artifacts_bundle(root)
    webbrowser.open(index.resolve().as_uri())
    return index


def _visualizer_script() -> str:
    if not _VISUALIZER_JS.exists():
        raise RuntimeError(
            "The vendored Model Explorer visualizer is missing. "
            "Run src/mlia/model_explorer/model-explorer-visualizer/update.py "
            "with a local Model Explorer checkout. Missing: "
            f"{_VISUALIZER_JS}"
        )
    return _escape_script_text(_VISUALIZER_JS.read_text(encoding="utf-8"))


def _bundle_data(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    model_paths = _artifact_paths(root, manifest.get("model_paths", []))
    if not model_paths:
        model_path = _artifact_path(root, manifest.get("model_path"))
        if model_path is not None and model_path.exists():
            model_paths = [model_path]

    graph_collections = [
        graph_collection
        for path in model_paths
        if path.suffix.lower() == ".json"
        for graph_collection in [_read_graph_collection(path)]
        if graph_collection is not None
    ]
    node_data_provider_runs = _node_data_provider_runs(
        root,
        manifest.get("node_data", []),
    )
    sync_navigation = _read_optional_json(root, manifest.get("sync_navigation"))
    initial_ui_state = _initial_ui_state(graph_collections, node_data_provider_runs)
    selected_node_data = _selected_node_data(node_data_provider_runs)

    visualizer_config: dict[str, Any] = {
        "hideTitleBar": True,
        "hideLegends": True,
        "keepLayersWithASingleChild": True,
        "artificialLayerNodeCountThreshold": _ARTIFICIAL_LAYER_NODE_COUNT_THRESHOLD,
    }
    if sync_navigation is not None:
        visualizer_config["syncNavigationData"] = sync_navigation

    return {
        "hostContractVersion": _HOST_CONTRACT_VERSION,
        "graphCollections": graph_collections,
        "nodeDataProviderRuns": node_data_provider_runs,
        "initialUiState": initial_ui_state,
        "selectedNodeData": selected_node_data,
        "visualizerConfig": visualizer_config,
        "manifest": manifest,
    }


def _read_graph_collection(path: Path) -> dict[str, Any] | None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("graphs"), list):
        return value
    logger.warning("Skipping unsupported Model Explorer graph JSON: %s", path)
    return None


def _node_data_provider_runs(
    root: Path,
    values: object,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for artifact in _node_data_artifacts(root, values):
        path = Path(artifact["node_data_path"])
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or not data
            or not all(
                isinstance(graph_data, dict)
                and isinstance(graph_data.get("results"), dict)
                for graph_data in data.values()
            )
        ):
            raise RuntimeError(
                f"Node-data payload must be a graph-indexed JSON object: {path}"
            )

        collection_label = artifact["graph_collection_label"]
        name = _node_data_provider_run_name(data, path.stem)
        runs.append(
            {
                "collectionLabel": collection_label,
                "name": name,
                "data": data,
            }
        )
    _validate_unique_node_data_provider_run_names(runs)
    return runs


def _validate_unique_node_data_provider_run_names(
    runs: list[dict[str, Any]],
) -> None:
    """Reject ambiguous public overlay identities on the same collection graph."""
    run_index_by_identity: dict[tuple[str, str, str], int] = {}
    for run_index, run in enumerate(runs):
        collection_label = str(run.get("collectionLabel") or "")
        run_name = str(run.get("name") or "")
        data = run.get("data")
        if not isinstance(data, dict):
            continue
        for graph_id, graph_data in data.items():
            if not isinstance(graph_id, str):
                continue
            graph_name = (
                graph_data.get("name") if isinstance(graph_data, dict) else None
            )
            effective_name = (
                graph_name if isinstance(graph_name, str) and graph_name else run_name
            )
            identity = (collection_label, graph_id, effective_name)
            previous_index = run_index_by_identity.get(identity)
            if previous_index is not None and previous_index != run_index:
                raise RuntimeError(
                    "Duplicate node-data provider name "
                    f"{effective_name!r} for graph {graph_id!r} in collection "
                    f"{collection_label!r}. Effective names must be unique per "
                    "collection graph."
                )
            run_index_by_identity[identity] = run_index


def _effective_node_data_provider_run_names(run: dict[str, Any]) -> set[str]:
    """Return the collection-scoped public names exposed by one run."""
    run_name = str(run.get("name") or "")
    data = run.get("data")
    graph_data_values = list(data.values()) if isinstance(data, dict) else []
    effective_names = {
        graph_name if isinstance(graph_name, str) and graph_name else run_name
        for graph_data in graph_data_values
        for graph_name in [
            graph_data.get("name") if isinstance(graph_data, dict) else None
        ]
    }
    return effective_names or {run_name}


def _node_data_provider_run_name(
    data: dict[str, Any],
    fallback: str,
) -> str:
    return next(
        (
            graph_data["name"]
            for graph_data in data.values()
            if isinstance(graph_data, dict)
            and isinstance(graph_data.get("name"), str)
            and graph_data["name"]
        ),
        fallback,
    )


def _initial_ui_state(
    graph_collections: list[dict[str, Any]],
    node_data_provider_runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(graph_collections) < 2:
        return None

    source_graph = _first_graph(graph_collections[0])
    generated_graphs = [
        (graph, collection)
        for collection in graph_collections[1:]
        for graph in [_first_graph(collection)]
        if graph
    ]
    if source_graph is None or not generated_graphs:
        return None

    selected_generated_graph, _ = max(
        generated_graphs,
        key=lambda candidate: _node_data_provider_run_coverage(
            candidate[1], node_data_provider_runs
        ),
    )
    initial_graphs = [source_graph, selected_generated_graph]
    width_fraction = 1.0 / len(initial_graphs)
    return {
        "paneStates": [
            _pane_state(graph, selected=index == 1, width_fraction=width_fraction)
            for index, graph in enumerate(initial_graphs)
        ]
    }


def _node_data_provider_run_coverage(
    graph_collection: dict[str, Any],
    node_data_provider_runs: list[dict[str, Any]],
) -> int:
    """Count valid upfront node-data values for the collection's first graph."""
    first_graph = _first_graph_data(graph_collection)
    if first_graph is None:
        return 0
    collection_label = str(graph_collection.get("label") or "")
    graph_id = first_graph["id"]
    graph_keys = graph_node_data_keys(first_graph)
    return sum(
        _graph_node_data_value_coverage(run.get("data", {}).get(graph_id), graph_keys)
        for run in node_data_provider_runs
        if run.get("collectionLabel") == collection_label
        and isinstance(run.get("data"), dict)
    )


def _graph_node_data_value_coverage(
    graph_data: object,
    graph_keys: set[str],
) -> int:
    if not isinstance(graph_data, dict):
        return 0
    results = graph_data.get("results")
    if not isinstance(results, dict):
        return 0
    return sum(
        1
        for key, entry in results.items()
        if isinstance(key, str)
        and key in graph_keys
        and isinstance(entry, dict)
        and entry.get("value") is not None
    )


def _selected_node_data(
    node_data_provider_runs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    collection_labels = sorted(
        {
            collection_label
            for run in node_data_provider_runs
            for collection_label in [run.get("collectionLabel")]
            if isinstance(collection_label, str) and collection_label
        }
    )
    for collection_label in collection_labels:
        for default_name in _DEFAULT_SELECTED_NODE_DATA_NAMES:
            if any(
                run.get("collectionLabel") == collection_label
                and default_name in _effective_node_data_provider_run_names(run)
                for run in node_data_provider_runs
            ):
                # Model Explorer's current public input calls the graph collection
                # label "modelName". Keep that API-boundary spelling until the
                # upstream input is renamed independently.
                selected.append({"modelName": collection_label, "name": default_name})
                break
    return selected


def _first_graph_data(graph_collection: dict[str, Any]) -> dict[str, Any] | None:
    graphs = graph_collection.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        return None
    graph = graphs[0]
    if not isinstance(graph, dict) or not isinstance(graph.get("id"), str):
        return None
    return graph


def _first_graph(graph_collection: dict[str, Any]) -> dict[str, str] | None:
    graph = _first_graph_data(graph_collection)
    if graph is None:
        return None
    return {
        "graph_id": graph["id"],
        "collection_label": str(graph_collection.get("label") or ""),
    }


def _pane_state(
    graph: dict[str, str],
    *,
    selected: bool = False,
    width_fraction: float = 0.5,
) -> dict[str, object]:
    pane_state: dict[str, object] = {
        "deepestExpandedGroupNodeIds": [],
        "selectedNodeId": "",
        "selectedGraphId": graph["graph_id"],
        "selectedCollectionLabel": graph["collection_label"],
        "widthFraction": width_fraction,
    }
    if selected:
        pane_state["selected"] = True
    return pane_state


def _read_optional_json(root: Path, value: object) -> dict[str, Any] | None:
    path = _artifact_path(root, value)
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _artifact_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _node_data_artifacts(root: Path, values: object) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    artifacts = []
    for value in values:
        if not isinstance(value, dict):
            continue
        path = _artifact_path(root, value.get("path"))
        graph_collection_label = value.get("graph_collection_label")
        if (
            path is not None
            and path.exists()
            and isinstance(graph_collection_label, str)
            and graph_collection_label
        ):
            artifacts.append(
                {
                    "node_data_path": str(path),
                    "graph_collection_label": graph_collection_label,
                }
            )
    return artifacts


def _artifact_paths(root: Path, values: object) -> list[Path]:
    if not isinstance(values, list):
        return []
    paths = []
    for value in values:
        path = _artifact_path(root, value)
        if path is not None and path.exists():
            paths.append(path)
    return paths


def _index_html(data: dict[str, Any]) -> str:
    return _INDEX_HTML.format(
        title=html.escape(_page_title(data), quote=False),
        data_json=_escape_json_script_data(data),
        visualizer_js=_visualizer_script(),
        host_js=_escape_script_text(_HOST_JS),
    )


def _page_title(data: dict[str, Any]) -> str:
    manifest = data.get("manifest", {})
    model = manifest.get("model", {}) if isinstance(manifest, dict) else {}
    model_name = model.get("name") if isinstance(model, dict) else None
    if isinstance(model_name, str) and model_name:
        return f"{model_name} - MLIA Model Explorer"
    return "MLIA Model Explorer"


def _escape_json_script_data(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False).replace("<", r"\u003c")


_SCRIPT_END_TAG = re.compile(r"</script", re.IGNORECASE)


def _escape_script_text(value: str) -> str:
    return _SCRIPT_END_TAG.sub(r"<\\/script", value)


_INDEX_HTML = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{title}</title>
    <style>
      html, body {{
        height: 100%;
        margin: 0;
        font-family: Arial, sans-serif;
        overflow: hidden;
      }}
      #mlia-app {{
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 0;
      }}
      #graph-toolbar {{
        align-items: center;
        background: #f7f8fa;
        border-bottom: 1px solid #d8dce3;
        box-sizing: border-box;
        display: grid;
        flex: 0 0 auto;
        grid-template-columns: 50% 50%;
        min-height: 48px;
        padding: 8px 0;
      }}
      .graph-select-control {{
        align-items: center;
        box-sizing: border-box;
        display: flex;
        gap: 8px;
        min-width: 0;
      }}
      .graph-select-control[hidden] {{
        display: none;
      }}
      #left-graph-control {{
        justify-content: flex-end;
        padding-right: 10px;
      }}
      #right-graph-control {{
        justify-content: flex-start;
        padding-left: 10px;
      }}
      .graph-select-control label {{
        color: #3c4043;
        font-size: 13px;
        font-weight: 600;
        white-space: nowrap;
      }}
      .graph-select-control select {{
        background: #fff;
        border: 1px solid #aeb4bd;
        border-radius: 4px;
        color: #202124;
        font: inherit;
        max-width: min(38vw, 520px);
        min-width: min(28vw, 280px);
        padding: 5px 28px 5px 8px;
      }}
      #add-pane-button {{
        align-items: center;
        background: #fff;
        border: 1px solid #aeb4bd;
        border-radius: 4px;
        color: #3c4043;
        cursor: pointer;
        display: inline-flex;
        font: 20px/1 Arial, sans-serif;
        height: 29px;
        justify-content: center;
        padding: 0;
        width: 29px;
      }}
      #add-pane-button:hover:not(:disabled) {{
        border-color: #6f7680;
        color: #202124;
      }}
      #add-pane-button:disabled {{
        cursor: default;
        opacity: 0.5;
      }}
      #add-pane-button[hidden] {{
        display: none;
      }}
      #visualizer-host {{ flex: 1 1 auto; min-height: 0; min-width: 0; }}
      model-explorer-visualizer {{ display: block; height: 100%; width: 100%; }}
      @media (max-width: 760px) {{
        #graph-toolbar {{
          align-items: stretch;
          display: flex;
          flex-direction: column;
          gap: 6px;
          padding: 8px 12px;
        }}
        .graph-select-control {{
          flex-basis: auto;
          justify-content: flex-start;
          padding-left: 0;
          padding-right: 0;
        }}
        .graph-select-control select {{
          flex: 1;
          max-width: none;
          min-width: 0;
          width: 100%;
        }}
      }}
    </style>
    <script>{visualizer_js}</script>
    <script id=\"mlia-model-explorer-data\" type=\"application/json\">
      {data_json}
    </script>
  </head>
  <body>
    <div id=\"mlia-app\">
      <div id=\"graph-toolbar\" role=\"toolbar\" aria-label=\"Displayed graphs\">
        <div id=\"left-graph-control\" class=\"graph-select-control\">
          <label for=\"left-graph-select\">Left graph</label>
          <select
            id=\"left-graph-select\"
            aria-label=\"Graph displayed in the left pane\"
          ></select>
          <button
            id=\"add-pane-button\"
            type=\"button\"
            aria-label=\"Open second pane\"
            title=\"Open second pane\"
            disabled
          >+</button>
        </div>
        <div id=\"right-graph-control\" class=\"graph-select-control\">
          <label for=\"right-graph-select\">Right graph</label>
          <select
            id=\"right-graph-select\"
            aria-label=\"Graph displayed in the right pane\"
          ></select>
        </div>
      </div>
      <main id=\"visualizer-host\"></main>
    </div>
    <script>{host_js}</script>
  </body>
</html>
"""

_HOST_JS = """(() => {
  const dataElement = document.getElementById('mlia-model-explorer-data');
  const data = dataElement ? JSON.parse(dataElement.textContent || '{}') : {};
  const visualizer = document.createElement('model-explorer-visualizer');
  const graphCollections = data.graphCollections || [];
  const graphOptions = [];
  for (const collection of graphCollections) {
    const graphs = collection.graphs || [];
    for (const graph of graphs) {
      graphOptions.push({
        collectionGraphCount: graphs.length,
        collectionLabel: collection.label || '',
        publicGraphId: graph.id,
        graph,
      });
    }
  }

  const toolbar = document.getElementById('graph-toolbar');
  const addPaneButton = document.getElementById('add-pane-button');
  const paneControls = [
    {
      container: document.getElementById('left-graph-control'),
      select: document.getElementById('left-graph-select'),
    },
    {
      container: document.getElementById('right-graph-control'),
      select: document.getElementById('right-graph-select'),
    },
  ];

  const optionLabel = (option) => {
    if (!option.collectionLabel) return option.publicGraphId;
    return option.collectionGraphCount === 1
      ? option.collectionLabel
      : `${option.collectionLabel} / ${option.publicGraphId}`;
  };
  for (const paneControl of paneControls) {
    if (!paneControl.select) continue;
    for (let index = 0; index < graphOptions.length; index += 1) {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = optionLabel(graphOptions[index]);
      paneControl.select.appendChild(option);
    }
    paneControl.select.disabled = true;
  }

  let selectorsReady = false;
  const synchronizeSelectors = (uiState) => {
    const paneStates = uiState?.paneStates || [];
    const hasTwoPanes = paneStates.length >= 2;
    if (toolbar) {
      const requestedFraction = paneStates[0]?.widthFraction;
      const leftFraction = hasTwoPanes && Number.isFinite(requestedFraction)
        ? Math.min(1, Math.max(0, requestedFraction))
        : 1;
      toolbar.style.gridTemplateColumns =
        `${leftFraction * 100}% ${(1 - leftFraction) * 100}%`;
    }
    if (paneControls[1].container) {
      paneControls[1].container.hidden = !hasTwoPanes;
    }
    if (addPaneButton) {
      addPaneButton.hidden = hasTwoPanes;
      addPaneButton.disabled =
        hasTwoPanes || !selectorsReady || !paneStates[0]?.selectedGraphId;
    }
    for (let paneIndex = 0; paneIndex < paneControls.length; paneIndex += 1) {
      const paneState = paneStates[paneIndex];
      const select = paneControls[paneIndex].select;
      if (!paneState || !select) continue;
      const optionIndex = graphOptions.findIndex((option) =>
        option.collectionLabel === paneState.selectedCollectionLabel &&
        (option.graph.id === paneState.selectedGraphId ||
          option.publicGraphId === paneState.selectedGraphId)
      );
      if (optionIndex >= 0) select.value = String(optionIndex);
    }
  };

  if (addPaneButton) {
    addPaneButton.addEventListener('click', () => {
      const leftSelect = paneControls[0].select;
      const option = leftSelect
        ? graphOptions[Number(leftSelect.value)]
        : undefined;
      if (!option) return;
      addPaneButton.disabled = true;
      visualizer.selectGraph(
        option.publicGraphId,
        option.collectionLabel || undefined,
        1,
      );
    });
  }

  for (let paneIndex = 0; paneIndex < paneControls.length; paneIndex += 1) {
    const select = paneControls[paneIndex].select;
    if (!select) continue;
    select.addEventListener('change', () => {
      const option = graphOptions[Number(select.value)];
      if (option) {
        visualizer.selectGraph(
          option.publicGraphId,
          option.collectionLabel || undefined,
          paneIndex,
        );
      }
    });
  }
  visualizer.addEventListener('uiStateChanged', (event) => {
    const hasSelectedGraph = (event.detail?.paneStates || []).some(
      (paneState) => Boolean(paneState.selectedGraphId)
    );
    selectorsReady = hasSelectedGraph;
    synchronizeSelectors(event.detail);
    if (hasSelectedGraph) {
      for (const paneControl of paneControls) {
        if (paneControl.select) {
          paneControl.select.disabled = graphOptions.length === 0;
        }
      }
    }
  });

  visualizer.graphCollections = graphCollections;
  visualizer.config = data.visualizerConfig || {};
  visualizer.nodeDataProviderRuns = data.nodeDataProviderRuns || [];
  visualizer.selectedNodeData = data.selectedNodeData || [];
  visualizer.syncNodeDataOverlaySelection = true;
  if (data.initialUiState) {
    visualizer.initialUiState = data.initialUiState;
  }
  synchronizeSelectors(data.initialUiState);

  const host = document.getElementById('visualizer-host');
  (host || document.body).appendChild(visualizer);
})();
"""
