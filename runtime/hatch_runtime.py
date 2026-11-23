# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Include runtime resources using the shared core packaging implementation."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from mlia.runtime.build import runtime_resource_files


class RuntimeBuildHook(BuildHookInterface):
    """Bind runtime resources to this provider's resolved wheel version."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Include the verified files for this wheel or source distribution."""
        build_data.setdefault("force_include", {}).update(
            runtime_resource_files(
                Path(self.root),
                self.metadata.name,
                self.metadata.version,
                self.target_name,
            )
        )
