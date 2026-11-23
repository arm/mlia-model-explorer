<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# Model Explorer runtime

The package definition is [model-explorer/pyproject.toml](model-explorer/pyproject.toml).
`model-explorer/process` accepts supported model files and returns display graphs and source-node mappings.

`uv build` embeds the worker wheel in the plugin. MLIA installs it and its
requirements in a separate environment on first use. Its version matches the
plugin wheel. Update the package requirements and canonical vendor inputs when
changing the toolchain.

The worker implementation is [Model Explorer worker](../src/mlia/model_explorer/worker.py).
