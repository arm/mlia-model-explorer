<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# MLIA Model Explorer Plugin

The `mlia-model-explorer` package registers the `model_explorer` post-analysis
plugin with ML Inference Advisor (MLIA).

[Read the documentation on Arm Developer](https://developer.arm.com/dev-docs/mlia-model-explorer/).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [ExecuTorch ETRecord profiling](#executorch-etrecord-profiling)
- [Repository contents](#repository-contents)
- [Reporting bugs](#reporting-bugs)
- [Development (uv)](#development-uv)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Releases](#releases)
- [License](#license)
- [Trademarks and copyrights](#trademarks-and-copyrights)

## Overview

The plugin exports MLIA standardized output into Model Explorer-oriented
artifacts and builds a standalone HTML page using the vendored embedded Model
Explorer visualizer component.

Model conversion uses the `ai-edge-model-explorer` Python APIs, while browser
rendering uses the vendored embedded Model Explorer visualizer. The generated
page is self-contained and does not start the Python Model Explorer web server.

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

## ExecuTorch ETRecord profiling

Use the matching ETRecord and ETDump to visualize source runtime timings and,
when captured, VGF cycle measurements. An optional matching PTE adds its compiler
graph. This workflow requires MLIA with the ExecuTorch extras installed:

```bash
uv run --with "mlia-neural-technology[executorch]" \
  --with "mlia-model-explorer[executorch]" \
  mlia check model.etrecord \
  --target-profile neural-technology \
  --performance \
  --profiling-data model.etdump \
  --companion-model model.pte \
  --model-explorer open
```

Only `.pte` companions with VGF content and matching captured VGF profiling are
currently supported. Omit `--companion-model` when no companion graph is needed.
Use `--model-explorer files` to generate the visualization without opening it.
See [ETRecord profiling](docs/source/etrecord.md) for method selection,
available views and measurement limitations. Companion input is also available
through the Python API; clients without a companion-model selector can use the
CLI or Python API for this workflow.

## Repository contents

- `src/mlia/model_explorer/`: plugin registration, artifact generation, model
  adapters, and the vendored browser visualizer.
- `tests/`: plugin, artifact, and browser tests.
- `docs/`: usage, architecture, API, and development documentation.
- `pre_commit_hooks/`: repository quality checks.
- `pyproject.toml`: package metadata, dependencies, and test configuration.

## Reporting bugs

Report plugin-specific bugs in the
[`arm/mlia-model-explorer` issue tracker](https://github.com/arm/mlia-model-explorer/issues).
For shared MLIA functionality, use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues).

## Development (uv)

Install development dependencies and run the tests against the declared
dependency versions:

```bash
uv sync --group dev
uv run pytest
uv run pre-commit run --all-files
uv build --wheel
```

For testing with local MLIA or Model Explorer checkouts, follow the
[development notes](docs/source/development.md) and use `dev-local.py`.

## Documentation

Structured documentation lives in [docs/README.md](docs/README.md). Start with:

- [Usage](docs/source/usage.md) for installation, CLI modes, and troubleshooting.
- [High-level architecture](docs/source/high_level_architecture.md) and
  [execution flow](docs/source/execution_flow.md) for the conversion pipeline.
- [Artifacts](docs/source/artifacts.md) for the generated directory layout.
- [Model adapters](docs/source/adapters.md) for format support and limitations.
- [Python API](docs/source/api.md) and
  [development notes](docs/source/development.md) for integration and maintenance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, commit sign-off, and
pull request requirements.

## Releases

See [MLIA Model Explorer releases](https://github.com/arm/mlia-model-explorer/releases)
for the release history.

## License

This project is licensed under Apache License 2.0. See
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Trademarks and copyrights

- Arm is a registered trademark or trademark of Arm Limited (or its subsidiaries) in the U.S. and/or elsewhere.
- Model Explorer is a Google project. Copyright and third-party license notices
  for the vendored visualizer are in
  [`main_embedded.js.license`](src/mlia/model_explorer/model-explorer-visualizer/main_embedded.js.license).
- PyTorch and ExecuTorch are trademarks of The Linux Foundation.
- Python is a registered trademark of the PSF.
