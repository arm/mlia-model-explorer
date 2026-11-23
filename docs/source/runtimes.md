<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->
# Model Explorer runtime

Model Explorer installs its model readers in a separate environment on first use.
The same environment handles every supported format and is reused for later checks.

## Selection and installation

One bundled profile, `arm/model-explorer`, accepts TFLite, PT2, ETRecord
(`.etrecord` or `.bin`), PTE, ONNX, VGF, SPIR-V and TOSA inputs. All adapters run
in `arm/model-explorer/public@VERSION`; source and companion models reuse the
same environment. The worker selects the adapter for each file. The existing
[format limitations](adapters.md) still apply, including TOSA source mapping and
standalone PTE restrictions.

```bash
mlia check model.pt2 -t ai-tile --model-explorer files \
  --model-explorer-execution-profile arm/model-explorer
mlia check model.pt2 -t ai-tile --model-explorer files --show-plan
```

`--model-explorer-execution-profile` accepts a catalog profile name or a TOML
file. It takes precedence over a `model-explorer/process` binding in the main
`--execution-profile` (or `MLIA_EXECUTION_PROFILE`). When the main profile has no
adapter binding, the shared default is used. The adapter profile is independent
of the target backend profile; one operation cannot be bound twice.

Example custom profile (replace the version with an installed provider version):

```toml
schema_version = 1
id = "my/model-explorer"
inputs = ["tflite", "pt2", "etrecord", "bin", "pte", "onnx", "vgf", "spv", "tosa"]

[stages.process]
operation = "model-explorer/process"
runtime = "arm/model-explorer/public@VERSION"

[stages.process.options.options]
const_element_count_limit = 16
```

The runtime version matches the installed `mlia-model-explorer` wheel version.

The selected runtime installs automatically on first processing and is reused.
Its combined dependency set is installed once, regardless of the first input
format. Generated HTML and graph files are saved with the analysis output.
Its dependencies live under
`~/.mlia/runtimes/arm/model-explorer/public@VERSION/ARCH/venv`, or the directory
selected by `MLIA_RUNTIME_HOME`. Core displays installation progress and applies
its usual EULA and noninteractive policy. The public bundled runtime does
not declare an interactive EULA. Dependencies requiring different system libraries
or operating systems remain the caller's responsibility.

Inspect packages without activating a runtime:

```bash
uv pip list --python /path/to/runtime/venv/bin/python
```

## Inspecting a run

Target execution records remain in `mlia-output/execution-plan.json`.
Post-analysis records and request/result/log files are separate under
`mlia-output/model-explorer/runtime/`. They identify the exact adapter runtime
and each conversion, including a PTE companion conversion where applicable.

The Python API supports `write_artifacts(result, execution_profile=...)` with
the same selection and installation behavior. See [Python API](api.md).
