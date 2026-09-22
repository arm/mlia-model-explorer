<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Usage

## Installation

Install the plugin in the same Python environment as MLIA:

```bash
python -m pip install mlia-model-explorer
```

The base package includes Model Explorer's built-in PyTorch ExportedProgram and
TFLite conversion support. Install an extra when the model format needs a
separate adapter, for example:

```bash
python -m pip install "mlia-model-explorer[onnx]"
python -m pip install "mlia-model-explorer[spirv]"
python -m pip install "mlia-model-explorer[vgf]"
```

See [Model Adapters](adapters.md) before choosing an extra; installing an adapter
does not by itself guarantee that this plugin has a verified source-identity
mapping for the format.

## Generate a standalone visualization

Use `files` when another process will archive, publish, or open the result:

```bash
mlia check MODEL --target-profile TARGET --compatibility --model-explorer files
```

Use `open` for an interactive local run:

```bash
mlia check MODEL --target-profile TARGET --compatibility --model-explorer open
```

With the default MLIA output directory, artifacts are written under
`mlia-output/model-explorer/` and the page is
`mlia-output/model-explorer/bundle/index.html`. The HTML is self-contained and
can be moved or opened directly with a `file://` URL.

## Interpreting the view

When both are available, the initial layout shows the source model beside the
generated MLIA hierarchy with the richest overlay coverage. Select placement,
status, metric, check, or advice overlays from the node-data controls. Synchronized
navigation uses explicit mappings so selecting related nodes across graph panes
does not depend on coincidentally matching node IDs.

Generated hierarchies reflect standardized output exactly. Configure entity or
code-stack collapse through MLIA core's `[[filtering.collapse]]` settings; legacy
Model Explorer plugin settings are rejected.

## Collapsed entity frames

Entity collapse is owned by MLIA core and is applied before this plugin receives
standardized output. Model Explorer does not parse filtering settings or perform
a second transformation; its standardized-output artifact, generated hierarchy,
mappings, and node data all reflect the processed result unchanged.

Configure structured `[[filtering.collapse]]` rules in MLIA's user `config.toml`.
MLIA loads this file from the platform-standard configuration directory, such as
`%LOCALAPPDATA%\arm\mlia\config.toml` on Windows or
`$XDG_CONFIG_HOME/mlia/config.toml` on Linux. See the
[MLIA configuration documentation](https://github.com/Arm/mlia#configuration)
for defaults, raw-value glob semantics, and how to disable collapse.

## PyTorch code stacks

For `.pt2` models, prefer `torch.export.export(..., strict=True)`. Strict export
uses PyTorch Dynamo's logical source stack and generally preserves ordinary
helper-function frames and their individual call-site lines.

In recent PyTorch versions, non-strict export may fall back to live FX tracing
that filters stack information mainly to `forward` methods. Operations on
different lines inside an ordinary helper function can then receive the same
recorded call site. Later PyTorch or ExecuTorch transformations may preserve,
replace, or append frames, so displayed stacks remain provenance information
rather than a guaranteed complete runtime Python call stack.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Adapter import error | Install the extra or adapter package named in the error, in the same environment as MLIA. |
| Unsupported model format or mapping | Check the [adapter matrix](adapters.md); the plugin fails rather than guessing node identity. |
| Materialized model hash or size mismatch | Regenerate the backend output and ensure the model file is directly inside the MLIA output root. |
| Plugin settings are no longer supported | Move collapse and filtering rules to MLIA core configuration. |
| Vendored visualizer is missing | Restore it from the package or follow the contributor update procedure in [Development](development.md). |
| Page does not open automatically | Use `files`, then open `bundle/index.html` manually in a modern browser. |
