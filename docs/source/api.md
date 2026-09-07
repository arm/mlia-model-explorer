<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Python API

## Supported surface

Most callers should enable the plugin through `mlia check`. Integrations that
already receive an MLIA `AnalysisRunResult` can invoke the artifact writer
directly:

```python
from mlia.model_explorer.artifacts import ModelExplorerArtifacts, write_artifacts

generated: ModelExplorerArtifacts = write_artifacts(analysis_run_result)
print(generated.manifest)
```

The supported import surface of `mlia.model_explorer.artifacts` is:

- The `write_artifacts(result)` function writes graph, overlay, navigation, and
  manifest files under `<result.output_dir>/model-explorer`.
- The `ModelExplorerArtifacts` class provides immutable paths describing the
  generated root, manifest, standardized output, node-data files, generated
  models, sync-navigation data, and entity-node mappings.

`write_artifacts` does not build `bundle/index.html`; the normal analysis plugin
does that after writing artifacts.

## Input expectations

The supplied `AnalysisRunResult` must follow the MLIA analysis-plugin contract.
In particular, its standardized output should already have been validated and
filtered by MLIA core. Direct callers are responsible for validation if they
bypass normal plugin invocation.

The writer accepts an absent output by producing a minimal empty result. A model
that is present must be supported and must map unambiguously to source-operator
identities.

## Stability

The path container and writer imports above are the intentionally supported
package API. Helpers in `bundler`, `graph_data`, and `_artifacts` support the
plugin implementation and may evolve with the embedded visualizer and artifact
contract.
