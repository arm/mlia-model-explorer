# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public artifact-generation package API."""

import unittest

from mlia.model_explorer import artifacts
from mlia.model_explorer.artifacts import ModelExplorerArtifacts, write_artifacts

_ASSERT = unittest.TestCase()


def test_public_artifact_imports() -> None:
    """The package conversion preserves the supported public import forms."""
    _ASSERT.assertIs(artifacts.ModelExplorerArtifacts, ModelExplorerArtifacts)
    _ASSERT.assertIs(artifacts.write_artifacts, write_artifacts)
    _ASSERT.assertEqual(
        artifacts.__all__, ["ModelExplorerArtifacts", "write_artifacts"]
    )
