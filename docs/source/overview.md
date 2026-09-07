<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Overview

`mlia-model-explorer` is an MLIA analysis plugin. It runs after analysis, consumes
MLIA's standardized output, and creates an interactive Model Explorer view of the
source graph, MLIA entity hierarchies, metrics, checks, and advice.

## What this package owns

The package owns:

- The `--model-explorer files|open` analysis-plugin option.
- Conversion of supported source models through Model Explorer adapters.
- Projection of MLIA entities into generated hierarchy graphs.
- Node-data overlays and cross-graph navigation mappings.
- A self-contained `index.html` using the vendored embedded visualizer.

The Python `ai-edge-model-explorer` package supplies conversion APIs and built-in
adapters. It does not serve the generated page. Browser rendering uses the
checked-in `main_embedded.js`, with graph data and overlays embedded directly in
the HTML.

## Ownership boundaries

MLIA core owns analysis, standardized-output validation, entity definitions, and
filtering such as code-stack collapse. Adapter packages own translation from
their native model formats into Model Explorer graph JSON. This plugin owns the
correlation between those graphs and MLIA entities, plus the standalone browser
experience.

The artifact manifest and mapping files carry schema or contract versions, but
their current field layouts are implementation contracts rather than a frozen
external interchange standard. See [Artifacts](artifacts.md) for their behavioral
roles.
