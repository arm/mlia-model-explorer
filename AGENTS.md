<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# Agent guidance

## Local sibling checkouts

When a task needs the local MLIA or Model Explorer sibling checkouts, run project
commands through the tracked Python helper:

```sh
python dev-local.py pytest
python dev-local.py mlia check ...
```

The helper loads editable dependencies from the gitignored
`local-overrides.txt`. Follow the format documented in
`docs/source/development.md`; paths in that file are machine-specific and must
not be committed.

Use normal `uv run ...` instead when testing the published dependency versions.
Do not add local `[tool.uv.sources]` entries to `pyproject.toml`, because they keep
the tracked file dirty and are not portable.

An editable `ai-edge-model-explorer` override changes only the Python package. If
a task changes the browser visualizer, rebuild and copy the vendored runtime with
`src/mlia/model_explorer/model-explorer-visualizer/update.py` as well.
