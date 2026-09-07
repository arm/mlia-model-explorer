# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Source-model discovery, adapter execution, and serialization."""

from __future__ import annotations

import hashlib
import importlib
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError, UnsupportedConfigurationError
from mlia.plugins.analysis import AnalysisRunResult

from .models import _SourceModelArtifacts
from .serialization import _safe_filename, _write_json
from .source_mapping import _source_operator_node_ids

_SOURCE_MODEL_COLLECTION_LABEL = "Source Model"

_ADAPTERS_BY_SUFFIX = {
    ".pt2": {
        "module": "model_explorer.builtin_pytorch_exportedprogram_adapter",
        "extra": "",
        "package": "ai-edge-model-explorer",
        "settings": {"const_element_count_limit": 16},
    },
    ".tflite": {
        "module": "model_explorer.builtin_tflite_flatbuffer_adapter",
        "extra": "",
        "package": "ai-edge-model-explorer",
        # The Arm adapter optionally folds pseudo-const nodes because their IDs can
        # collide with nodes and groups targeted by MLIA overlay data, causing
        # colours to appear on the wrong boxes. The supported upstream built-in
        # TFLite adapter ignores unrecognised setting entries, so this is safe to
        # pass unconditionally.
        "settings": {"fold_pseudo_const_nodes": True},
    },
    ".onnx": {
        "module": "model_explorer_onnx.main",
        "extra": "onnx",
        "package": "model-explorer-onnx",
    },
    ".spv": {
        "module": "spirv_adapter_model_explorer.main",
        "extra": "spirv",
        "package": "spirv-adapter-model-explorer",
    },
    ".vgf": {
        "module": "vgf_adapter_model_explorer.main",
        "extra": "vgf",
        "package": "vgf-adapter-model-explorer",
    },
    ".tosa": {
        "module": "tosa_adapter_model_explorer.main",
        "extra": "tosa",
        "package": "tosa-adapter-model-explorer",
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
    model_path: Path | None, root: Path
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
    source_operator_to_node_ids = _source_operator_node_ids(
        model_path, graph_collection
    )
    _write_json(path, graph_collection)
    return _SourceModelArtifacts(
        [path], graph_collection, source_operator_to_node_ids
    )


def _run_model_explorer_adapter(
    model_path: Path, adapter_info: dict[str, Any]
) -> dict[str, Any]:
    """Run the Model Explorer adapter for a source model."""
    module_name = adapter_info["module"]
    try:
        module = importlib.import_module(module_name)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise RuntimeError(_adapter_import_error_message(adapter_info, err)) from err

    adapter_class = _adapter_class(module, model_path.suffix)
    if adapter_class is None:
        raise RuntimeError(
            f"Model Explorer adapter module {module_name} does not provide an "
            f"adapter for {model_path.suffix}."
        )

    return adapter_class().convert(
        str(model_path), dict(adapter_info.get("settings", {}))
    )


def _adapter_class(module: Any, suffix: str) -> type | None:
    """Find an adapter class in an imported Model Explorer adapter module."""
    expected_ext = suffix.lower().lstrip(".")
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        metadata = getattr(value, "metadata", None)
        file_exts = getattr(metadata, "fileExts", [])
        if expected_ext in [
            str(file_ext).lower().lstrip(".") for file_ext in file_exts
        ]:
            return value
    return None


def _adapter_import_error_message(adapter_info: dict[str, Any], err: Exception) -> str:
    install_lines = []
    if adapter_info["extra"]:
        install_lines.extend(
            [
                "Install the required optional dependency, for example:",
                "  python -m pip install "
                f'"mlia-model-explorer[{adapter_info["extra"]}]"',
                "",
                "or install the adapter package directly, for example:",
            ]
        )
    else:
        install_lines.append("Install the required package, for example:")
    install_lines.append(f"  python -m pip install {adapter_info['package']}")
    return (
        "Model Explorer needs an adapter package for this model file, but it "
        f"could not be imported:\n\n  {adapter_info['module']}\n"
        f"  {type(err).__name__}: {err}\n\n"
        + "\n".join(install_lines)
    )


def _source_model_graph_collection(
    model_path: Path, adapter_response: dict[str, Any]
) -> dict[str, Any]:
    """Convert adapter output to MLIA's source graph collection shape."""
    # Model Explorer adapters are not completely uniform: some return a
    # top-level ``graphs`` list, while others return one or more
    # ``graphCollections``. MLIA writes one source graph collection artifact and
    # later uses the collection label as the stable model-name discriminator for
    # node-data overlays. Normalize all adapter shapes to that single artifact
    # and always use MLIA's source label so empty or adapter-specific labels do
    # not make overlays unmatchable or unstable.
    graph_collections = adapter_response.get("graphCollections") or []
    if graph_collections:
        if len(graph_collections) != 1:
            # Graph IDs are scoped by collection, and nodes can link to other
            # graphs through subgraphIds. Supporting multiple collections would
            # require preserving that scope or renaming graph IDs while rewriting
            # every graph reference and all downstream mappings/node data. Reject
            # the unsupported shape rather than flattening it ambiguously.
            labels = [
                str(collection.get("label") or "<unnamed>")
                if isinstance(collection, dict)
                else "<invalid>"
                for collection in graph_collections
            ]
            raise RuntimeError(
                "Model Explorer adapter returned multiple graph collections; "
                "mlia-model-explorer currently requires exactly one. "
                f"Returned collections: {', '.join(labels)}."
            )
        graph_collection = _without_none(_jsonable(graph_collections[0]))
        graph_collection["label"] = _SOURCE_MODEL_COLLECTION_LABEL
        return graph_collection

    graphs = adapter_response.get("graphs") or []
    return {
        "label": _SOURCE_MODEL_COLLECTION_LABEL,
        "graphs": _without_none(_jsonable(graphs)),
    }


def _jsonable(value: Any) -> Any:
    """Convert dataclass adapter objects to JSON-compatible values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _without_none(value: Any) -> Any:
    """Recursively omit None values from adapter output."""
    if isinstance(value, dict):
        return {
            key: _without_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value
