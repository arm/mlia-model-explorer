# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Locate the source model and write its display graph and operator mappings.

The runtime reads model files; this module supplies display settings and
normalizes the returned JSON for the report.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError, UnsupportedConfigurationError
from mlia.plugins.analysis import AnalysisRunResult

from .models import _SourceModelArtifacts
from .serialization import _safe_filename, _write_json
from .source_mapping import _source_operator_node_ids

_SOURCE_MODEL_COLLECTION_LABEL = "Source Model"

# Reader modules and package dependencies belong to the isolated runtime.
_ADAPTERS_BY_SUFFIX: dict[str, dict[str, Any]] = {
    ".pt2": {
        "settings": {"const_element_count_limit": 16},
    },
    ".tflite": {
        # The Arm adapter optionally folds pseudo-const nodes because their IDs can
        # collide with nodes and groups targeted by MLIA overlay data, causing
        # colours to appear on the wrong boxes. The supported upstream built-in
        # TFLite adapter ignores unrecognised setting entries, so this is safe to
        # pass unconditionally.
        "settings": {"fold_pseudo_const_nodes": True},
    },
    ".onnx": {},
    ".spv": {},
    ".vgf": {},
    ".tosa": {},
    ".etrecord": {},
    ".bin": {
        "adapter_suffix": ".etrecord",
    },
}


def _empty_output(result: AnalysisRunResult) -> dict[str, Any]:
    model_path = _model_path(result)
    return {
        "schema_version": None,
        "model": {"name": model_path.name if model_path else None},
        "target": {},
        "backends": [],
        "results": [],
    }


def _model_path(result: AnalysisRunResult) -> Path | None:
    model = result.parameters.get("model")
    if model is not None:
        path = Path(str(model)).expanduser()
        return path if path.exists() else None
    return _materialized_output_model_path(result)


def _materialized_output_model_path(result: AnalysisRunResult) -> Path | None:
    """Resolve and authenticate an effective model placed in the output root."""
    output = result.output
    if not isinstance(output, dict):
        return None
    model = output.get("model")
    if not isinstance(model, dict):
        return None
    name = model.get("name")
    expected_hash = model.get("hash")
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or not isinstance(expected_hash, str)
    ):
        return None

    output_dir = result.output_dir.resolve()
    candidate = result.output_dir / name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != output_dir or not resolved.is_file():
        return None

    expected_size = model.get("size_bytes")
    if isinstance(expected_size, int) and resolved.stat().st_size != expected_size:
        raise ConfigurationError(
            f"Materialized model '{resolved}' does not match standardized-output size."
        )
    actual_hash = _file_sha256(resolved)
    if actual_hash.lower() != expected_hash.lower():
        raise ConfigurationError(
            f"Materialized model '{resolved}' does not match standardized-output hash."
        )
    return resolved


def _file_sha256(path: Path) -> str:
    """Return a file SHA-256 without loading the entire model into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_model(model_path: Path | None, root: Path) -> Path | None:
    if model_path is None:
        return None
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / model_path.name
    if model_path.resolve() != destination.resolve():
        shutil.copy2(model_path, destination)
    return destination


def _write_source_model_files(
    model_path: Path | None,
    root: Path,
    output: dict[str, Any] | None = None,
) -> _SourceModelArtifacts:
    """Write Model Explorer JSON for the source model when an adapter is known."""
    if model_path is None:
        return _SourceModelArtifacts([])

    model_format = model_path.suffix.lower()
    adapter_info = _ADAPTERS_BY_SUFFIX.get(model_format)
    if not isinstance(adapter_info, dict):
        unsupported_format = model_format or "<none>"
        raise UnsupportedConfigurationError(
            f"Model Explorer does not support model format '{unsupported_format}'."
        )

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{_safe_filename(model_path.stem)}-source.json"
    adapter_response = _run_model_explorer_adapter(model_path, adapter_info)
    graph_collection = _source_model_graph_collection(model_path, adapter_response)
    from .formats import _format_integration

    mapping_options = (
        {"output": output} if _format_integration(model_path).has_source_mapping else {}
    )
    source_operator_to_node_ids = _source_operator_node_ids(
        model_path, graph_collection, **mapping_options
    )
    _write_json(path, graph_collection)
    return _SourceModelArtifacts(
        [path],
        graph_collection,
        source_operator_to_node_ids,
    )


def _run_model_explorer_adapter(
    model_path: Path, adapter_info: dict[str, Any]
) -> dict[str, Any]:
    """Convert the model through the selected isolated adapter toolchain."""
    from ..execution import convert_model

    return convert_model(model_path, adapter_info)


def _source_model_graph_collection(
    model_path: Path, adapter_response: dict[str, Any]
) -> dict[str, Any]:
    """Convert adapter output to MLIA's source graph collection shape."""
    # Model Explorer adapters are not completely uniform: some return a
    # top-level ``graphs`` list, while others return one or more
    # ``graphCollections``. Source-collection selection is owned by the format
    # integration; ordinary adapters retain the single-collection contract.
    live_graph_collection = _source_adapter_graph_collection(
        model_path, adapter_response
    )
    if live_graph_collection is not None:
        graph_collection = _without_none(live_graph_collection)
        if not isinstance(graph_collection, dict):
            raise RuntimeError(
                "Model Explorer adapter returned an invalid graph collection."
            )
        graph_collection["label"] = _SOURCE_MODEL_COLLECTION_LABEL
        return graph_collection

    graphs = adapter_response.get("graphs") or []
    return {
        "label": _SOURCE_MODEL_COLLECTION_LABEL,
        "graphs": _without_none(graphs),
    }


def _without_none(value: Any) -> Any:
    """Recursively omit None values from adapter output."""
    if isinstance(value, dict):
        return {
            key: _without_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _source_adapter_graph_collection(
    model_path: Path, adapter_response: dict[str, Any]
) -> Any | None:
    """Select the analyzed collection through its format integration."""
    from .formats import _format_integration

    return _format_integration(model_path).source_collection(
        model_path, adapter_response
    )


def _graph_collection_label(graph_collection: Any) -> str:
    if isinstance(graph_collection, dict):
        return str(graph_collection.get("label") or "")
    return str(getattr(graph_collection, "label", "") or "")
