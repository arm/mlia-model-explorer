# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public artifact-generation package API."""

from mlia.model_explorer import artifacts
from mlia.model_explorer.artifacts import ModelExplorerArtifacts, write_artifacts


def test_public_artifact_imports() -> None:
    """The package conversion preserves the supported public import forms."""
    assert artifacts.ModelExplorerArtifacts is ModelExplorerArtifacts
    assert artifacts.write_artifacts is write_artifacts
    assert artifacts.__all__ == ["ModelExplorerArtifacts", "write_artifacts"]
