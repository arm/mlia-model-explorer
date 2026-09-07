<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Model Explorer Documentation

This directory contains the MkDocs content for the `mlia-model-explorer`
repository. The top-level `README.md` remains the quick-start guide; these pages
cover plugin integration, artifact generation, format support, and maintenance.

## Build

Install only the documentation dependencies, then build from the repository root:

```bash
uv sync --no-install-project --only-group docs
uv run --only-group docs mkdocs build --strict
```

For local preview:

```bash
uv run --only-group docs mkdocs serve
```

The generated site is written to `.mkdocs/site/`.

## Included pages

- The `source/index.md` page is the documentation landing page.
- The `source/overview.md` page describes the package role and ownership boundaries.
- The `source/high_level_architecture.md` page presents the static component view.
- The `source/plugin_integration.md` page covers MLIA discovery and invocation.
- The `source/execution_flow.md` page follows the conversion and bundle flow.
- The `source/usage.md` page covers installation, commands, and troubleshooting.
- The `source/api.md` page documents the supported Python artifact API.
- The `source/artifacts.md` page explains the generated file layout and behavior.
- The `source/adapters.md` page lists model-format adapters and mapping limitations.
- The `source/development.md` page provides contributor and maintenance guidance.

## Scope

The site documents this plugin and the contracts it consumes from MLIA. General
MLIA configuration and standardized-output schema details belong in the core
MLIA documentation; Model Explorer adapter internals belong in their owning
projects.
