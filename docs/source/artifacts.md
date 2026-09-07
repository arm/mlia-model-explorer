<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Artifacts

## Directory layout

A typical run produces:

```text
model-explorer/
├── manifest.json
├── standardized-output.json
├── sync-navigation.json
├── entity-node-mappings.json
├── models/
│   ├── <model>-source.json
│   └── <model>-hierarchy-<entity-kinds>.json
├── node-data/
│   └── <graph-collection>/
│       └── <overlay>.json
└── bundle/
    └── index.html
```

The exact number and names of hierarchy and overlay files depend on the MLIA
results. Filenames are sanitized, length-bounded, and collision-disambiguated for
portable output.

## File roles

| Artifact | Purpose |
| --- | --- |
| `standardized-output.json` | Preserves the MLIA result supplied to visualization. |
| `models/*-source.json` | Contains the adapter-produced source graph collection, normalized to the `Source Model` label. |
| `models/*-hierarchy-*.json` | Contains generated graphs for result-local entity-kind paths. |
| `node-data/**/*.json` | Provides graph-indexed placement, status, metric, check, or advice overlays for named graph collections. |
| `sync-navigation.json` | Contains graph-qualified mappings used to synchronize panes. |
| `entity-node-mappings.json` | Relates MLIA entity IDs to collection, graph, and node-qualified references. |
| `manifest.json` | Records run metadata, artifact paths, overlay routing, and logical overlay display names. |
| `bundle/index.html` | Embeds all usable artifacts, host configuration, and the browser runtime in one file. |

## Graph and node identity

Graph collection labels scope graph IDs, and graph IDs scope node IDs. Mapping
records therefore use all three values. Consumers should not join data using a
bare node ID, because separate subgraphs or generated collections may legitimately
reuse it.

Model Explorer also creates artificial group rows from namespace prefixes. The
plugin normalizes escaped slashes and alternative namespaces using the same rules
as the browser runtime so overlays target the visible group rather than an
unrelated node with similar text.

## Overlay behavior

Each node-data file is a graph-indexed JSON object and is routed only to graph
collections containing matching node or namespace keys. Logical overlay identity
includes result index and kind, category, name, unit, and qualifiers; display
names are disambiguated when those properties would otherwise appear identical.

Qualitative facts may be repeated on the source nodes participating in a higher
level entity. Metrics use more conservative projection rules so a group total is
not presented as though it were measured independently for every child.

## Contract versioning

`manifest.json` currently carries its own schema version, and the embedded page
carries a host contract version. These versions allow the plugin and vendored
runtime to evolve together. The files are useful for inspection and integration,
but their complete field layouts are not declared as a frozen public interchange
schema.
