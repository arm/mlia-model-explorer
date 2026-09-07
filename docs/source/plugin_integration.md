<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Plugin Integration

## Discovery

Installing the package registers `ModelExplorerAnalysisPlugin` in the
`mlia.plugin.analysis` Python entry-point group. MLIA loads that entry point and
the plugin registers the `model-explorer` post-analysis implementation.

The plugin contributes one Click option:

```text
--model-explorer MODE
```

`MODE` is either `files` or `open`. Without the option, the plugin is disabled
and has no effect on the analysis run.

## Invocation contract

MLIA supplies an `AnalysisRunResult` after analysis. The plugin uses its:

- Parsed arguments to select the mode.
- Parameters to locate the command-line model.
- Standardized output to generate hierarchy and overlay data.
- Output directory as the parent of `model-explorer/`.

Before writing files, the plugin rejects legacy plugin-specific settings and
validates the standardized output through MLIA core. Filtering configuration is
therefore applied by MLIA before this plugin runs; the plugin visualizes the
processed result without applying another collapse or filter pass.

## Modes

Both modes generate the same artifacts and standalone HTML:

- The `files` mode stops after building `bundle/index.html`.
- The `open` mode additionally opens the resolved `file://` URL in the default browser.

The page does not start an HTTP server and does not depend on a separately
installed Model Explorer server UI.

## Failure behavior

Configuration, validation, adapter, mapping, serialization, and bundle errors are
allowed to propagate to MLIA's normal command error handling. The plugin does not
silently omit an unsupported model format or an ambiguous source mapping because
that could attach results to the wrong graph nodes.
