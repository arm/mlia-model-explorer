<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# Vendored Model Explorer Visualizer

This directory contains the pre-built embedded Model Explorer custom element used by `mlia-model-explorer` to create standalone browser pages without starting the Python `ai-edge-model-explorer` server.

Vendored runtime file:

- `main_embedded.js` registers `<model-explorer-visualizer>` and embeds the visualizer worker and static assets.
- `main_embedded.js.license` contains generated combined license and copyright notices, bound to the JavaScript file by its SHA-256 hash. Keep both files together when redistributing the runtime.

The checked-in bundle is built from a modified version of Model Explorer.

## Updating

From the `mlia-model-explorer` repository root, run:

```console
python src/mlia/model_explorer/model-explorer-visualizer/update.py E:\model-explorer\model-explorer
```

The script runs the upstream custom-element npm build (`npm run build-npm`) in the supplied Model Explorer checkout, then validates and copies both outputs:

```text
src/ui/custom_element_npm/dist/main_embedded.js
src/ui/custom_element_npm/dist/main_embedded.js.license
```

A missing output or a license whose bundle hash does not match stops the update before either vendored file is replaced. The two files are staged before replacement so a copy failure leaves the existing pair unchanged.

Commit both real generated files. Do not edit the license output manually or commit symlinks, `__pycache__` directories, or other generated Python bytecode.
