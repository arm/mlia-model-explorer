<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# Vendored Model Explorer Visualizer

This directory contains the pre-built embedded Model Explorer custom element used by `mlia-model-explorer` to create standalone browser pages without starting the Python `ai-edge-model-explorer` server.

Vendored runtime file:

- `main_embedded.js` registers `<model-explorer-visualizer>` and embeds the visualizer worker and static assets.

The checked-in bundle is built from a modified version of Model Explorer.

## Updating

From the `mlia-model-explorer` repository root, run:

```console
python src/mlia/model_explorer/model-explorer-visualizer/update.py E:\model-explorer\model-explorer
```

The script runs the upstream custom-element npm build (`npm run build-npm`) in the supplied Model Explorer checkout, then replaces the vendored file from:

```text
src/ui/custom_element_npm/dist/main_embedded.js
```

Commit the real generated file only. Do not commit symlinks, `__pycache__` directories, or other generated Python bytecode.
