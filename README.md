<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# MLIA Model Explorer Plugin

This package provides a post-analysis plugin for ML Inference Advisor (MLIA).
It exports MLIA standardized output into Model Explorer-oriented artifacts and
builds a standalone HTML page using the vendored embedded Model Explorer
visualizer component.

Model conversion uses the `ai-edge-model-explorer` Python APIs, while browser
rendering uses the vendored embedded Model Explorer visualizer. The generated
page is self-contained and does not start the Python Model Explorer web server.

## Documentation

Structured documentation lives in [docs/README.md](docs/README.md). Start with:

- [Usage](docs/source/usage.md) for installation, CLI modes, and troubleshooting.
- [High-level architecture](docs/source/high_level_architecture.md) and
  [execution flow](docs/source/execution_flow.md) for the conversion pipeline.
- [Artifacts](docs/source/artifacts.md) for the generated directory layout.
- [Model adapters](docs/source/adapters.md) for format support and limitations.
- [Python API](docs/source/api.md) and
  [development notes](docs/source/development.md) for integration and maintenance.

## Installation

Install the plugin in the same Python environment as MLIA:

```bash
python -m pip install mlia-model-explorer
```

The base package supports `.pt2` and `.tflite`. Optional extras provide `.onnx`,
`.spv` and `.vgf` adapters; see the [model adapter matrix](docs/source/adapters.md)
for installation commands and format-specific details.

## Usage

```bash
mlia check MODEL --target-profile TARGET --compatibility --model-explorer files
mlia check MODEL --target-profile TARGET --compatibility --model-explorer open
```

The `files` mode writes artifacts and builds the standalone HTML page. The `open`
mode additionally opens that page in the default browser.

Generated artifacts are written under `mlia-output/model-explorer/`. The
standalone HTML page is written to `mlia-output/model-explorer/bundle/index.html`.
It can be opened directly from `file://` without separate runtime or data files.
