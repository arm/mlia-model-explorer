# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for matching Model Explorer graph nodes to node-data entries."""

from __future__ import annotations

from typing import Any

_ESCAPED_SLASH_TOKEN = "___escaped_slash___"  # nosec B105: sentinel string, not a secret.


def graph_node_data_keys(graph: dict[str, Any]) -> set[str]:
    """Return node IDs and namespace group keys addressable by node data."""
    keys: set[str] = set()
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str):
            keys.add(node_id)
        namespace = node.get("namespace")
        if isinstance(namespace, str) and namespace:
            keys.update(model_explorer_namespace_prefixes(namespace))
    return keys


def model_explorer_namespace_components(namespace: str) -> list[str]:
    """Return hierarchy components using Model Explorer's namespace rules.

    Model Explorer removes empty semicolon-separated parts and uses the final
    part only when multiple non-empty parts remain. With zero or one non-empty
    part, it preserves the original raw namespace. Within that namespace,
    ``\\/`` represents a literal slash rather than a hierarchy separator.
    """
    alternatives = [part for part in namespace.split(";") if part]
    actual_namespace = alternatives[-1] if len(alternatives) > 1 else namespace
    return [
        part.replace(_ESCAPED_SLASH_TOKEN, r"\/")
        for part in actual_namespace.replace(r"\/", _ESCAPED_SLASH_TOKEN).split("/")
        if part
    ]


def model_explorer_normalized_namespace(namespace: str) -> str:
    """Return the exact namespace Model Explorer uses to create groups."""
    return "/".join(model_explorer_namespace_components(namespace))


def model_explorer_namespace_prefixes(namespace: str) -> list[str]:
    """Return artificial-group namespaces in shallow-to-deep order."""
    components = model_explorer_namespace_components(namespace)
    return ["/".join(components[: index + 1]) for index in range(len(components))]


def model_explorer_namespace_depth(namespace: str) -> int:
    """Return the number of Model Explorer hierarchy components."""
    return len(model_explorer_namespace_components(namespace))
