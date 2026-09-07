# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Artifact filenames, manifests, and JSON serialization."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mlia.plugins.analysis import AnalysisRunResult

from .models import (
    _CanonicalJsonValue,
    _CanonicalQualifiers,
    _NodeDataArtifact,
    _OverlayKey,
)

_NODE_DATA_FILENAME_EXTENSION = ".json"
_MAX_FILENAME_COMPONENT_BYTES = 180
_FILENAME_COLLISION_SUFFIX_RESERVE_BYTES = 24
_MAX_FILENAME_STEM_BYTES = (
    _MAX_FILENAME_COMPONENT_BYTES
    - len(_NODE_DATA_FILENAME_EXTENSION.encode("utf-8"))
    - _FILENAME_COLLISION_SUFFIX_RESERVE_BYTES
)
_WINDOWS_RESERVED_FILENAME_STEMS = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    "com¹",
    "com²",
    "com³",
    *(f"lpt{index}" for index in range(1, 10)),
    "lpt¹",
    "lpt²",
    "lpt³",
}
_STABLE_DIGEST_HEX_LENGTH = 12


def _bounded_text_with_digest(
    value: str,
    max_bytes: int,
    digest: str,
    *,
    separator: str,
) -> str:
    """Bound UTF-8 text while retaining a stable suffix for the complete value."""
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    suffix = f"{separator}{digest}"
    prefix_budget = max_bytes - len(suffix.encode("utf-8"))
    prefix = _truncate_utf8(value, prefix_budget).rstrip()
    return f"{prefix}{suffix}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    return value.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _safe_filename(value: str) -> str:
    safe = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "-", value.strip()).strip(" .-")
    safe = safe or "result"
    if safe.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_FILENAME_STEMS:
        safe = f"_{safe}"
    return _bounded_text_with_digest(
        safe,
        _MAX_FILENAME_STEM_BYTES,
        hashlib.sha256(value.encode("utf-8")).hexdigest()[:_STABLE_DIGEST_HEX_LENGTH],
        separator="--",
    ).rstrip(" .")


def _allocate_safe_filename(value: str, allocated: set[str]) -> str:
    """Return a deterministic collision-free, component-length-safe stem."""
    base = _safe_filename(value)
    candidate = base
    suffix_number = 2
    while candidate.casefold() in allocated:
        suffix = f" ({suffix_number})"
        base_budget = (
            _MAX_FILENAME_COMPONENT_BYTES
            - len(_NODE_DATA_FILENAME_EXTENSION.encode("utf-8"))
            - len(suffix.encode("utf-8"))
        )
        candidate = f"{_truncate_utf8(base, base_budget).rstrip(' .')}{suffix}"
        suffix_number += 1
    allocated.add(candidate.casefold())
    return candidate


def _manifest(
    result: AnalysisRunResult,
    output: dict[str, Any],
    standardized_output_path: Path,
    model_path: Path | None,
    generated_model_paths: list[Path],
    node_data_artifacts: list[_NodeDataArtifact],
    sync_navigation_path: Path,
    entity_node_mappings_path: Path,
) -> dict[str, Any]:
    root = result.output_dir / "model-explorer"
    model_paths = [path for path in [model_path, *generated_model_paths] if path]
    manifest: dict[str, Any] = {
        "schema_version": "0.1.0",
        "source": "mlia-model-explorer",
        "command": result.command_name,
        "target": output.get("target", {}),
        "model": output.get("model", {}),
        "model_path": _relative_path(model_path, root) if model_path else None,
        "standardized_output": _relative_path(standardized_output_path, root),
        "sync_navigation": _relative_path(sync_navigation_path, root),
        "entity_node_mappings": _relative_path(entity_node_mappings_path, root),
        "node_data": [
            _node_data_manifest_entry(artifact, root)
            for artifact in node_data_artifacts
        ],
        "nodeDataOverlayKeysAndDisplayNames": (
            _node_data_overlay_keys_and_display_names(node_data_artifacts)
        ),
    }
    if generated_model_paths:
        manifest["model_paths"] = [_relative_path(path, root) for path in model_paths]
        manifest["generated_models"] = [
            _relative_path(path, root) for path in generated_model_paths
        ]
    return manifest


def _node_data_manifest_entry(
    artifact: _NodeDataArtifact, root: Path
) -> dict[str, str]:
    return {
        "path": _relative_path(artifact.path, root),
        "graph_collection_label": artifact.graph_collection_label,
    }


def _node_data_overlay_keys_and_display_names(
    artifacts: list[_NodeDataArtifact],
) -> list[dict[str, Any]]:
    """Return each logical overlay identity and public display name once."""
    entries: list[dict[str, Any]] = []
    seen: set[_OverlayKey] = set()
    for artifact in artifacts:
        if artifact.overlay_key in seen:
            continue
        seen.add(artifact.overlay_key)
        entries.append(
            {
                "overlayKey": _overlay_key_json(artifact.overlay_key),
                "displayName": artifact.display_name,
            }
        )
    return entries


def _overlay_key_json(key: _OverlayKey) -> dict[str, Any]:
    """Return the complete logical overlay identity as typed JSON data."""
    return {
        "resultIndex": key.result_index,
        "resultKind": key.result_kind,
        "category": key.category,
        "name": key.name,
        "unit": key.unit,
        "breakdownQualifiers": _canonical_qualifiers_json(key.breakdown_qualifiers),
        "metricQualifiers": _canonical_qualifiers_json(key.metric_qualifiers),
    }


def _canonical_qualifiers_json(qualifiers: _CanonicalQualifiers) -> dict[str, Any]:
    return {name: _canonical_json_value_json(value) for name, value in qualifiers}


def _canonical_json_value_json(value: _CanonicalJsonValue) -> Any:
    """Restore one canonical tagged value to its JSON representation."""
    tag = value[0]
    if tag == "null":
        return None
    if tag == "boolean" or tag == "string":
        return value[1]
    if tag == "number":
        return json.loads(value[1])
    if tag == "array":
        return [_canonical_json_value_json(item) for item in value[1]]
    if tag == "object":
        return {name: _canonical_json_value_json(item) for name, item in value[1]}
    raise ValueError(f"Unsupported canonical JSON tag: {tag!r}")


def _relative_path(path: Path, root: Path) -> str:
    """Return a manifest path relative to the artifact root when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Python otherwise serializes NaN and infinities even though JSON forbids
    # them. Fail at the artifact boundary rather than writing files that strict
    # consumers such as JSON.parse cannot read.
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
