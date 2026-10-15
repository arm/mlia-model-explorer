# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Check installed package compatibility through MLIA's plugin loader."""

from mlia.plugins.analysis import AnalysisPluginRegistry
from mlia.plugins.plugins import load_analysis_plugins


def test_installed_plugin_is_discovered_by_mlia() -> None:
    """Exercise entry-point metadata and core/interface version checks."""
    registry = AnalysisPluginRegistry()

    load_analysis_plugins(registry)

    plugin = registry.items["model-explorer"]
    assert registry.plugin_interface_versions["model-explorer"] == "0.0.2"
    assert plugin.enabled({"model_explorer": "files"})
    assert not plugin.enabled({})
    options = {option.name: option for option in plugin.cli_options()}
    assert options["model_explorer"].type.choices == ("files", "open")
