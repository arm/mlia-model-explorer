<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Model Adapters

The plugin selects a Model Explorer adapter from the effective model's filename
suffix. It then requires a verified mapping from MLIA source-operator entity IDs
to the adapter's graph nodes. Adapter installation and end-to-end mapping support
are therefore separate requirements.

## Support matrix

| Format | Dependency | Source-identity behavior | End-to-end status |
| --- | --- | --- | --- |
| `.pt2` | Base `ai-edge-model-explorer` dependency | Matches top-level exported-program FX node names to one adapter graph. | Supported |
| `.tflite` | Base `ai-edge-model-explorer` dependency | Uses operator indices within each flatbuffer subgraph; identical indices remain graph-qualified. | Supported |
| `.onnx` | `mlia-model-explorer[onnx]` | Correlates top-level ONNX nodes through name, output tensors, and operation labels, requiring a complete one-to-one match. | Supported with restrictions |
| `.spv` | `mlia-model-explorer[spirv]` | Uses native SPIR-V result-ID metadata exposed by the adapter. | Supported |
| `.vgf` | `mlia-model-explorer[vgf]` | Combines VGF segment identity with native SPIR-V result-ID metadata. | Supported |
| `.tosa` | `mlia-model-explorer[tosa]` | No verified native source-operator identity contract is implemented. | Not yet supported end to end |

## Common adapter requirements

Adapter output must normalize to exactly one graph collection. Multiple graph
collections are rejected because flattening them could collide graph IDs or break
subgraph references. Within the accepted collection, graph and node IDs used for
mapping must be valid and unambiguous.

The source graph collection is always relabeled `Source Model`; this stable label
is used to route node-data overlays and qualify mapping references.

## ONNX restrictions

ONNX mapping currently supports the top-level `ModelProto.graph` only. Models
with ONNX functions or graph-valued attributes are rejected because their node
identity requires a future graph-qualified native ID format. Initializer, input,
and output helper nodes emitted by the adapter are not treated as source
operations.

Names alone are insufficient when they are duplicated. The mapper combines all
available authoritative evidence and rejects missing, ambiguous, conflicting, or
incomplete matches rather than attaching MLIA data to a plausible-looking node.

## Adapter and visualizer versions

The Python adapter package converts models; it does not provide the runtime used
by the standalone page. Adapter packages may themselves depend on
`ai-edge-model-explorer` because they use its adapter API and graph-building
helpers; this remains separate from serving the UI.

Both `files` and `open` use the vendored standalone Model Explorer runtime.

Loading these artifacts in a separately installed Model Explorer server is
unsupported until that project builds its `web_app`, bumps the package version,
and publishes a release containing the graph-mapping runtime. Do not infer or
substitute an upstream version that has not been published.
