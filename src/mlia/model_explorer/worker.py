# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Read supported model formats and write graph data for Model Explorer."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert adapter dataclasses and tuples to JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _source_facts(model: Path) -> dict[str, Any]:
    """Read node identities needed to map PT2 and ONNX models onto displayed graphs."""
    if model.suffix.lower() == ".pt2":
        import torch

        program = torch.export.load(model)
        return {
            "nodes": [
                {"name": node.name, "op": node.op} for node in program.graph.nodes
            ]
        }
    if model.suffix.lower() == ".onnx":
        import onnx

        graph = onnx.load(str(model), load_external_data=False)
        return {
            "has_functions": bool(graph.functions),
            "name": graph.graph.name,
            "nodes": [
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "domain": node.domain,
                    "output": list(node.output),
                    "has_subgraphs": any(
                        attribute.HasField("g") or attribute.graphs
                        for attribute in node.attribute
                    ),
                }
                for node in graph.graph.node
            ],
        }
    return {}


def _adapters() -> dict[str, str]:
    """Load the packaged mapping from model extensions to reader modules."""
    return json.loads(
        (resources.files("mlia_runtime_assets") / "resources/adapters.json").read_text()
    )


def run(request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Execute one whole adapter conversion plus required native source reads."""
    if request["operation"] != "model-explorer/process":
        raise ValueError("Unsupported Model Explorer operation")
    payload = request["payload"]
    model = Path(payload["model"])
    suffix = str(payload.get("adapter_suffix", model.suffix)).lower()
    module_name = _adapters().get(suffix)
    if module_name is None:
        raise ValueError(f"Selected Model Explorer runtime does not support {suffix}")
    module = importlib.import_module(module_name)
    for value in vars(module).values():
        if isinstance(value, type) and suffix.lstrip(".") in [
            str(ext).lower().lstrip(".")
            for ext in getattr(getattr(value, "metadata", None), "fileExts", [])
        ]:
            response = value().convert(str(model), dict(payload.get("options", {})))
            break
    else:
        raise ValueError(f"Adapter {module_name} does not provide {suffix}")
    output = Path(request["output_dir"]) / "conversion.json"
    output.write_text(
        json.dumps(
            {
                "adapter_response": _jsonable(response),
                "source_facts": _source_facts(model),
            },
            allow_nan=False,
        )
    )
    return {}, [{"path": str(output), "kind": "model-explorer-conversion"}]


def validate_runtime(request: dict[str, Any]) -> None:
    """Check adapter imports before marking the toolchain ready."""
    for module in set(_adapters().values()):
        importlib.import_module(module)
