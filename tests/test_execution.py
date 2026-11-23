# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Profile selection and reuse across source and companion model conversions."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import load_profile

from mlia.model_explorer.execution import (
    conversion_session,
    convert_model,
    select_conversion_profile,
)


@pytest.fixture
def profile_file(tmp_path):
    """Bind a real catalog runtime without provisioning its dependencies."""
    profile = load_profile("arm/model-explorer")
    _, definition = next(iter(profile.stages.values()))
    path = tmp_path / "profile.toml"
    path.write_text(
        'schema_version = 1\nid = "test/explorer"\ninputs = ["tflite"]\n'
        '[stages.process]\noperation = "model-explorer/process"\n'
        f'runtime = "{definition.name}"\n'
    )
    return path


def test_selects_stage_for_model_input(profile_file):
    """A stage restricted to TFLite is usable for a TFLite model."""
    profile_file.write_text(profile_file.read_text() + 'inputs = ["tflite"]\n')
    selected = select_conversion_profile(Path("model.tflite"), profile_file, {})
    assert list(selected.stages) == ["process"]
    assert (
        selected.profile_sha256
        == load_profile(profile_file, model=Path("model.tflite")).profile_sha256
    )


@pytest.mark.parametrize("source", ["explicit", "combined", "environment"])
def test_rejects_input_outside_selected_profile(profile_file, monkeypatch, source):
    """Input restrictions apply regardless of how the user selects a profile."""
    selection = profile_file if source == "explicit" else None
    parameters = (
        {"execution_profile": str(profile_file)} if source == "combined" else {}
    )
    if source == "environment":
        monkeypatch.setenv("MLIA_EXECUTION_PROFILE", str(profile_file))
    with pytest.raises(ConfigurationError, match="does not accept.*.pt2"):
        select_conversion_profile(Path("model.pt2"), selection, parameters)


@pytest.mark.parametrize("restricted", [False, True])
def test_source_and_companion_share_runtime(
    tmp_path, profile_file, monkeypatch, restricted
):
    """Reuse the toolchain, but reject companion formats excluded by its profile."""
    sessions = []
    inputs = []

    @contextmanager
    def execute(profile, output, **_kwargs):
        sessions.append(profile)

        def invoke(operation, payload):
            assert operation == "model-explorer/process"
            inputs.append(Path(payload["model"]).suffix)
            output.mkdir(parents=True, exist_ok=True)
            result = output / "conversion.json"
            result.write_text(json.dumps({"adapter_response": {}, "source_facts": {}}))
            return {
                "artifacts": [
                    {"path": str(result), "kind": "model-explorer-conversion"}
                ]
            }

        yield SimpleNamespace(profile=profile, invoke=invoke)

    monkeypatch.setattr("mlia.model_explorer.execution.execution_session", execute)
    result = SimpleNamespace(parameters={}, output_dir=tmp_path, prompt=None)
    with conversion_session(result, profile_file if restricted else None):
        assert convert_model(tmp_path / "model.tflite", {}) == {}
        if restricted:
            with pytest.raises(ConfigurationError, match="does not accept.*.pte"):
                convert_model(tmp_path / "companion.pte", {})
        else:
            assert convert_model(tmp_path / "companion.pte", {}) == {}
    assert len(sessions) == 1
    assert inputs == ([".tflite"] if restricted else [".tflite", ".pte"])
