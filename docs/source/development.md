<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development

## Published dependency environment

Use normal `uv` commands when testing the dependency versions declared by this
repository:

```bash
uv sync --group dev
uv run pytest
```

Artifact and browser tests cover public imports, adapter mappings, graph-qualified
navigation, overlay routing, bundle selection, escaping, and execution of the
vendored visualizer when a supported headless browser is available.

## Local sibling checkouts

To test against sibling MLIA or Model Explorer checkouts without modifying
`pyproject.toml`, put the machine-specific editable sources in the gitignored
`local-overrides.txt` file:

```text
-e E:/mlia/mlia
-e E:/model-explorer/model-explorer/src/server/package
```

Run project commands through the tracked Python helper, which layers those
requirements over the normal project environment:

```sh
python dev-local.py pytest
python dev-local.py mlia check MODEL --target-profile TARGET --compatibility --model-explorer files
```

Use ordinary `uv run ...` when validating published dependency versions. Do not
add machine-specific `[tool.uv.sources]` entries to `pyproject.toml`.

The editable Model Explorer override changes only its Python package. Browser
changes must also be rebuilt and copied with the updater described below;
updating the editable package does not change this plugin's standalone runtime
or Model Explorer's packaged `web_app`.

## Building the documentation

From the repository root:

```bash
uv sync --no-install-project --only-group docs
uv run --only-group docs mkdocs build --strict
```

For local preview:

```bash
uv run --only-group docs mkdocs serve
```

Keep `mkdocs.yml`, `docs/README.md`, and the top-level documentation links aligned
when adding or renaming a page. The site is Markdown-first so the source remains
readable directly in the repository.

## Updating the vendored visualizer

An editable `ai-edge-model-explorer` checkout changes only Python conversion
code. Browser changes must be built and copied into this repository separately:

```text
python src/mlia/model_explorer/model-explorer-visualizer/update.py <model-explorer-checkout>
```

The updater runs Model Explorer's `npm run build-npm` and replaces
`main_embedded.js` with `src/ui/custom_element_npm/dist/main_embedded.js` from the
checkout. Update the provenance note in the visualizer README, commit the actual
generated file rather than a symlink, and rerun the Python and headless-browser
tests.

## Maintenance boundaries

- Keep general MLIA configuration and standardized-output schema guidance in the
  core MLIA documentation.
- Keep native conversion behavior in the owning adapter project.
- Document here the plugin's mapping rules, generated artifacts, embedded runtime,
  and cross-project compatibility requirements.
