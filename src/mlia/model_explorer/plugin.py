# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model Explorer post-analysis plugin."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import click
from mlia.core.errors import ConfigurationError
from mlia.core.output_validation import validate_standardized_output
from mlia.plugins.analysis import AnalysisPluginRegistry, AnalysisRunResult
from mlia.plugins.plugins import AnalysisPlugin

from mlia.model_explorer.artifacts import write_artifacts
from mlia.model_explorer.bundler import build_artifacts_bundle, open_artifacts_bundle

logger = logging.getLogger(__name__)

MODEL_EXPLORER_MODES = ("files", "open")


class _ModelExplorerAnalysisPlugin:
    """Export MLIA analysis results for Model Explorer."""

    name = "model-explorer"

    def cli_options(self) -> list[click.Option]:
        """Return Model Explorer CLI options."""
        return [
            click.Option(
                ["--model-explorer"],
                type=click.Choice(MODEL_EXPLORER_MODES),
                metavar="MODE",
                help=(
                    "Export Model Explorer artifacts after analysis. MODE is one of "
                    "'files' or 'open'."
                ),
            )
        ]

    def enabled(self, args: Mapping[str, object]) -> bool:
        """Return whether the plugin is enabled."""
        return args.get("model_explorer") is not None

    def run(self, result: AnalysisRunResult) -> None:
        """Export artifacts and optionally open them."""
        if result.settings:
            raise ConfigurationError(
                "Model Explorer plugin settings are no longer supported; "
                "configure standardized-output filtering in the core "
                "[filtering] table instead."
            )
        validate_standardized_output(result.output or {}, use_jsonschema=False)
        mode = cast(str, result.args["model_explorer"])
        artifacts = write_artifacts(result)
        logger.info("Model Explorer artifacts written to %s", artifacts.root)
        logger.info("Model Explorer manifest: %s", artifacts.manifest)

        if mode == "open":
            index = open_artifacts_bundle(artifacts.root)
        else:
            index = build_artifacts_bundle(artifacts.root)
        logger.info("Model Explorer standalone HTML: %s", index)


class ModelExplorerAnalysisPlugin(AnalysisPlugin[AnalysisPluginRegistry]):
    """MLIA entry point for the Model Explorer analysis plugin."""

    plugin_interface_version = "0.0.2"

    @staticmethod
    def register(registry: AnalysisPluginRegistry) -> None:
        """Register the Model Explorer analysis plugin."""
        registry.register(_ModelExplorerAnalysisPlugin())
