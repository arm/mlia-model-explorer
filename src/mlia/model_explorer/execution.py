# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Select a Model Explorer runtime and read its graph data and source mappings."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError
from mlia.plugins.analysis import AnalysisRunResult
from mlia.runtime.catalog import ExecutionProfile, load_profile
from mlia.runtime.session import RuntimeSession, execution_session

_OPERATION = "model-explorer/process"


def select_conversion_profile(
    model: Path, selection: str | Path | None, parameters: Mapping[str, object]
) -> ExecutionProfile:
    """Resolve a dedicated profile or the conversion binding in a combined one."""
    selected = (
        selection
        or parameters.get("execution_profile")
        or os.environ.get("MLIA_EXECUTION_PROFILE")
    )
    profile = load_profile(str(selected), model=model) if selected else None
    if profile is None or (
        selection is None
        and not any(op == _OPERATION for op, _ in profile.stages.values())
    ):
        profile = load_profile("arm/model-explorer", model=model)
    stages = {
        name: stage for name, stage in profile.stages.items() if stage[0] == _OPERATION
    }
    if not stages:
        raise ConfigurationError(
            f"Model Explorer profile {profile.data['id']} must define {_OPERATION}"
        )
    for _, definition in stages.values():
        contract = definition.data["operations"][_OPERATION]["result_schema"]
        if contract != "model-explorer/1":
            raise ConfigurationError(
                "Model Explorer requires result contract model-explorer/1; "
                f"got {contract}"
            )
    profile = replace(
        profile,
        data={
            **profile.data,
            "stages": {name: profile.data["stages"][name] for name in stages},
        },
        stages=stages,
    )
    _validate_input(profile, model)
    return profile


def _validate_input(profile: ExecutionProfile, model: Path) -> None:
    """Check source and companion formats against the selected processing stage."""
    suffix = model.suffix.lower().lstrip(".")
    stage = next(iter(profile.data["stages"].values()))
    for inputs in (profile.data.get("inputs"), stage.get("inputs")):
        if inputs and suffix not in inputs:
            raise ConfigurationError(
                f"Model Explorer profile {profile.data['id']} does not accept "
                f"model representation {model.suffix}"
            )


class _ConversionSession:
    """Lazily provision one adapter toolchain and retain its native source facts."""

    def __init__(
        self, result: AnalysisRunResult, selection: str | Path | None, stack: ExitStack
    ):
        self.result = result
        self.selection = selection
        self.stack = stack
        self.runtime: RuntimeSession | None = None
        self.facts: dict[Path, dict[str, Any]] = {}

    def convert(self, model: Path, adapter: dict[str, Any]) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            profile = select_conversion_profile(
                model, self.selection, self.result.parameters
            )
            accepted = (
                True
                if self.result.parameters.get("i_agree_to_the_contained_eula")
                else False
                if self.result.parameters.get("noninteractive")
                else None
            )
            runtime = self.stack.enter_context(
                execution_session(
                    profile,
                    self.result.output_dir / "model-explorer" / "runtime",
                    accept_eula=accepted,
                    prompt=self.result.prompt,
                )
            )
            self.runtime = runtime
        _validate_input(runtime.profile, model)
        options = dict(adapter.get("settings", {}))
        for stage in runtime.profile.data["stages"].values():
            options.update(stage.get("options", {}).get("options", {}))
        response = runtime.invoke(
            _OPERATION,
            {
                "model": str(model.resolve()),
                "adapter_suffix": adapter.get("adapter_suffix", model.suffix.lower()),
                "options": options,
            },
        )
        artifacts = response["artifacts"]
        if (
            len(artifacts) != 1
            or artifacts[0].get("kind") != "model-explorer-conversion"
        ):
            raise ConfigurationError(
                "Model Explorer runtime must return exactly one conversion artifact"
            )
        try:
            data = json.loads(Path(artifacts[0]["path"]).read_text())
            graph, facts = data["adapter_response"], data["source_facts"]
            if not isinstance(graph, dict) or not isinstance(facts, dict):
                raise ValueError("Expected graph and native source facts objects")
            _validate_facts(model, facts)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(
                f"Invalid Model Explorer runtime result: {exc}"
            ) from exc
        self.facts[model.resolve()] = facts
        return graph


def _validate_facts(model: Path, facts: dict[str, Any]) -> None:
    """Validate only native evidence required by the consuming mapping code."""
    suffix = model.suffix.lower()
    if suffix not in {".pt2", ".onnx"}:
        return
    nodes = facts["nodes"]
    if not isinstance(nodes, list):
        raise ValueError("source_facts.nodes must be an array")
    fields = ("name", "op") if suffix == ".pt2" else ("name", "op_type", "domain")
    for node in nodes:
        if not isinstance(node, dict) or any(
            not isinstance(node[field], str) for field in fields
        ):
            raise ValueError("Invalid source node evidence")
        if suffix == ".onnx" and (
            not isinstance(node["has_subgraphs"], bool)
            or not isinstance(node["output"], list)
            or any(not isinstance(name, str) for name in node["output"])
        ):
            raise ValueError("Invalid ONNX source node evidence")
    if suffix == ".onnx" and (
        not isinstance(facts["has_functions"], bool)
        or not isinstance(facts["name"], str)
    ):
        raise ValueError("Invalid ONNX graph evidence")


_CURRENT: ContextVar[_ConversionSession | None] = ContextVar(
    "model_explorer_conversion", default=None
)


@contextmanager
def conversion_session(
    result: AnalysisRunResult, selection: str | Path | None
) -> Iterator[None]:
    """Scope conversion state to one API call, including companion adapters."""
    with ExitStack() as stack:
        token = _CURRENT.set(_ConversionSession(result, selection, stack))
        try:
            yield
        finally:
            _CURRENT.reset(token)


def convert_model(model: Path, adapter: dict[str, Any]) -> dict[str, Any]:
    """Run the adapter; never import it into the frontend."""
    session = _CURRENT.get()
    if session is None:
        raise ConfigurationError(
            "Model conversion requires write_artifacts or the "
            "Model Explorer analysis plugin"
        )
    return session.convert(model, adapter)


def model_facts(model: Path) -> dict[str, Any]:
    """Read native evidence from the same worker conversion as the source graph."""
    session = _CURRENT.get()
    if session is None or model.resolve() not in session.facts:
        raise ConfigurationError(f"No Model Explorer source evidence for {model}")
    return session.facts[model.resolve()]
