---
# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
sidebar_position: 7
---

# ETRecord profiling views

Visualize a matching ETRecord and ETDump to explore runtime timings and captured
VGF profiling. Other model formats are covered in [Usage](usage.md) and
[Model adapters](adapters.md).

## Run the visualization

Supply the `.etrecord` (or `.bin`) and its matching `.etdump` (or `.etdp`). Add a
matching `.pte` if you also want its VGF compiler graph. The following command
installs the required ExecuTorch extras for this run and opens the visualization:

```bash
uv run --with "mlia-neural-technology[executorch]" \
  --with "mlia-model-explorer[executorch]" \
  mlia check model.etrecord \
  --target-profile neural-technology \
  --performance \
  --profiling-data run.etdump \
  --companion-model model.pte \
  --output-dir mlia-etrecord-result \
  --model-explorer open
```

This workflow currently supports only a `.pte` companion containing a VGF
delegate, with matching VGF profiling in the ETDump. Other companion formats
and non-VGF companion views are not supported. The Model Explorer ExecuTorch
extra includes the required ETRecord, PTE and VGF adapters.

Omit `--companion-model` when no companion view is needed. Captures containing
only ordinary runtime events remain supported and show available source timings.

For a multi-method record, add
`--neural-technology-profiling-data.etrecord-method decode`, replacing `decode`
with the captured method. Single-method selection is automatic.

Use `--model-explorer files` to generate files without opening the browser.
The standalone page is written under the output directory at
`mlia-output/model-explorer/bundle/index.html`. Analysis results are saved in
`mlia-output/mlia-output.json`. Companion input is also supported through the
Python API's `companion_model` argument. Supplying a companion does not enable
visualization; `--model-explorer` must be requested explicitly.

## Explore the results

| View | Contents |
| --- | --- |
| `Source Model` | Source operators and available runtime timings. |
| `segment_cascade_chain` | The captured NX hierarchy and its profiling. |
| `ETRecord Chain Mapping` | Chains associated with source operators; the initial detailed view when available. |
| `ETDump Backend Trace` | Compiler operations and their measured chain groups. |
| `PTE - VGF Backend Hierarchy` | The optional companion compiler graph with associated chain groups. |

Select a mapped operator or group to navigate between related views. Some views
or links may be absent when the capture does not provide the required
relationships. Unmatched compiler operations remain visible; a missing link
does not establish that an operation was not executed.

## Measurements and limitations

- Runtime timings are in **milliseconds**; VGF counters are in **cycles**.
  Cycle counters describe captured activity, not elapsed latency.
- PTE and ETDump views show VGF profiling on chain and cascade groups.
  A group's measurement is not an independent timing for every operator it
  contains. Source associations may be incomplete or approximate.
- Colours use the same scale for the same entity kind across views. The viewer
  selects `total_cycles` when available, otherwise `op_cycles`.
- Parent and child groups may describe overlapping work. Do not sum measurements
  across hierarchy levels. Shared runtime timings, P50, minimum and maximum
  values are not additive; unavailable group aggregates are omitted.
- Independent measurements are combined within each recorded execution before
  summarizing repeated executions. Combining group averages requires matching
  execution coverage.
- Totals describe recorded observations. Missing measurements are not estimated,
  and a successful visualization does not prove a complete inference was
  captured. Use the reported execution coverage and warnings when interpreting
  summaries. Model cycle summaries are unavailable when VGF instructions or
  segments cover different recorded executions.
- Companion matching checks structural compatibility and assumes you supplied
  the matching PTE. It does not prove that this PTE produced the capture. Before
  generating the companion view, its digest is compared with the analyzed file's
  recorded digest; a changed file requires analysis again.

## Supported VGF captures

VGF decoding supports statistics modes 0 and 1 for Mali-G1 and Mali-G2:

| Device | Mode 0 words per task | Mode 1 words per task |
| --- | --- | --- |
| Mali-G1 | 16 | 32 |
| Mali-G2 | 16 | 64 |

A Mali-G2 mode-0 layout with 64 words per task is not supported. Declared
counter-layout metadata must use schema version 1; unsupported or conflicting
settings are rejected.

Recorded mode and device settings are used when available. Legacy schema-v1 VGF
captures do not record these settings, so analysis defaults to mode 1 and
Mali-G2 when they are missing. Check the reported assumptions and warnings:
successful decoding does not verify that those defaults match the capture.

For counter layouts, aggregation rules and capture-completeness details, see
[Neural Technology ETRecord profiling](https://github.com/arm/mlia-neural-technology/blob/main/docs/source/etrecord.md).
See [Artifacts](artifacts.md) for the generated files. If an adapter is missing
or incompatible, install the extra or version named in the error. For companion
matching errors, check that all artifacts belong to the same deployed program
and selected method.
